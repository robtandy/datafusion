# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import argparse
import ctypes
from dataclasses import dataclass
from decimal import Decimal
import json
import math
import mmap
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import duckdb
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

BATCH_SIZE = 8192
DEFAULT_ROW_GROUP_SIZE = 131_072
BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
TPCDS_BENCHMARKS_DIR = BENCHMARKS_ROOT / "sql_benchmarks" / "tpcds" / "benchmarks"
STANDARD_TPCDS_QUERIES: tuple[str, ...] = tuple(f"q{i:02d}" for i in range(1, 100))

# All 24 standard TPC-DS tables with deterministic primary-key ordering and multi-file
# shard counts (16 files for large fact tables, 1-4 files for medium/small tables).
TPCDS_TABLES: dict[str, tuple[str, int]] = {
    "inventory": ("inv_date_sk, inv_item_sk, inv_warehouse_sk", 16),
    "store_sales": ("ss_item_sk, ss_ticket_number", 16),
    "catalog_sales": ("cs_item_sk, cs_order_number", 16),
    "web_sales": ("ws_item_sk, ws_order_number", 16),
    "customer_demographics": ("cd_demo_sk", 4),
    "store_returns": ("sr_item_sk, sr_ticket_number", 4),
    "catalog_returns": ("cr_item_sk, cr_order_number", 4),
    "customer": ("c_customer_sk", 2),
    "time_dim": ("t_time_sk", 2),
    "date_dim": ("d_date_sk", 2),
    "web_returns": ("wr_item_sk, wr_order_number", 2),
    "customer_address": ("ca_address_sk", 2),
    "item": ("i_item_sk", 1),
    "catalog_page": ("cp_catalog_page_sk", 1),
    "household_demographics": ("hd_demo_sk", 1),
    "promotion": ("p_promo_sk", 1),
    "reason": ("r_reason_sk", 1),
    "web_page": ("wp_web_page_sk", 1),
    "web_site": ("web_site_sk", 1),
    "income_band": ("ib_income_band_sk", 1),
    "ship_mode": ("sm_ship_mode_sk", 1),
    "store": ("s_store_sk", 1),
    "call_center": ("cc_call_center_sk", 1),
    "warehouse": ("w_warehouse_sk", 1),
}

CLICKHOUSE_SQL_FLAGS: tuple[str, ...] = (
    "--join_use_nulls=1",
    "--group_by_use_nulls=1",
    "--joined_subquery_requires_alias=0",
    "--enable_analyzer=1",
    "--allow_experimental_correlated_subqueries=1",
    "--union_default_mode=DISTINCT",
    "--intersect_default_mode=DISTINCT",
    "--except_default_mode=DISTINCT",
    "--cast_keep_nullable=1",
)

# Queries where DataFusion's PostgreSQL-style integer division (`INT / INT` truncates)
# legitimately differs from DuckDB's float division (`INT / INT` returns DOUBLE).
# On these queries we strictly verify `morsel_pushdown == stock_default` and
# `stock_pushdown == stock_default` while documenting the cross-engine division rule.
DATAFUSION_INT_DIV_QUERIES: frozenset[str] = frozenset({"q21", "q78", "q83"})

_BENCHMARK_RUN_RE = re.compile(r"^run\s+(\S+)\s*$", re.MULTILINE)
_BENCHMARK_NAME_RE = re.compile(r"^name\s+(\S+)\s*$", re.MULTILINE)
_Q90_AT_ALIAS_RE = re.compile(r"\)\s+at\s*,")


@dataclass(frozen=True)
class Workload:
    name: str
    query_num: int
    benchmark_file: str
    sql_file: str
    statements: tuple[str, ...]
    tables: tuple[str, ...]
    ordered_statements: tuple[bool, ...]

    @property
    def statement_count(self) -> int:
        return len(self.statements)


def strip_sql_comments_and_split(raw_sql: str) -> tuple[str, ...]:
    """Strip `--` line comments outside string literals and split on `;`."""
    stmts: list[str] = []
    cur: list[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(raw_sql)
    while i < n:
        ch = raw_sql[i]
        if (
            not in_single
            and not in_double
            and ch == "-"
            and i + 1 < n
            and raw_sql[i + 1] == "-"
        ):
            while i < n and raw_sql[i] != "\n":
                i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            cur.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            cur.append(ch)
        elif ch == ";" and not in_single and not in_double:
            stmt = "".join(cur).strip()
            if stmt:
                stmts.append(stmt)
            cur = []
        else:
            cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        stmts.append(tail)
    return tuple(stmts)


def _strip_string_literals(sql: str) -> str:
    out: list[str] = []
    in_single = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            in_single = not in_single
            out.append(" ")
        elif in_single:
            out.append(" ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def extract_referenced_tables(statements: tuple[str, ...]) -> tuple[str, ...]:
    """Identify which of the 24 standard TPC-DS tables appear in the query SQL."""
    cleaned = _strip_string_literals("\n".join(statements)).lower()
    found = [
        table
        for table in sorted(TPCDS_TABLES)
        if re.search(rf"\b{re.escape(table)}\b", cleaned)
    ]
    if not found:
        raise ValueError("no TPC-DS table references found in SQL")
    return tuple(found)


def has_top_level_order_by(sql: str) -> bool:
    """Return True if the statement has an ORDER BY clause at parenthesis depth 0."""
    cleaned = _strip_string_literals(sql)
    depth = 0
    top_chars: list[str] = []
    for ch in cleaned:
        if ch == "(":
            depth += 1
            top_chars.append(" ")
        elif ch == ")":
            depth = max(0, depth - 1)
            top_chars.append(" ")
        elif depth == 0:
            top_chars.append(ch)
        else:
            top_chars.append(" ")
    top_sql = "".join(top_chars)
    return bool(re.search(r"\border\s+by\b", top_sql, re.IGNORECASE))


def adapt_sql_for_duckdb(qname: str, sql: str) -> str:
    """Apply the single documented DuckDB 2.0.1 reserved-keyword quote for Q90 (`at`)."""
    if qname == "q90":
        rewritten, count = _Q90_AT_ALIAS_RE.subn(') "at",', sql, count=1)
        if count != 1:
            raise ValueError(f"expected `) at,` alias in {qname} SQL")
        return rewritten
    return sql


def parse_benchmark_file(
    bench_path: Path, benchmarks_root: Path = BENCHMARKS_ROOT
) -> Workload:
    """Parse a single `qNN.benchmark` file and load its referenced TPC-DS SQL file."""
    text = bench_path.read_text()
    name_match = _BENCHMARK_NAME_RE.search(text)
    if name_match is None:
        raise ValueError(f"missing `name` directive in {bench_path}")
    run_matches = _BENCHMARK_RUN_RE.findall(text)
    if len(run_matches) != 1:
        raise ValueError(
            f"expected exactly 1 `run` directive in {bench_path}, found {len(run_matches)}"
        )
    rel_sql = run_matches[0]
    sql_path = (benchmarks_root / rel_sql).resolve(strict=True)
    raw_sql = sql_path.read_text()
    statements = strip_sql_comments_and_split(raw_sql)
    if not statements:
        raise ValueError(f"no SQL statements parsed from {sql_path}")
    raw_name = name_match.group(1).lower()
    num_match = re.fullmatch(r"q(\d+)", raw_name)
    if num_match is None:
        raise ValueError(f"unexpected benchmark name '{raw_name}' in {bench_path}")
    qnum = int(num_match.group(1))
    qname = f"q{qnum:02d}"
    tables = extract_referenced_tables(statements)
    ordered = tuple(has_top_level_order_by(stmt) for stmt in statements)
    return Workload(
        name=qname,
        query_num=qnum,
        benchmark_file=bench_path.name,
        sql_file=rel_sql,
        statements=statements,
        tables=tables,
        ordered_statements=ordered,
    )


def load_tpcds_workloads(
    benchmarks_dir: Path = TPCDS_BENCHMARKS_DIR,
    benchmarks_root: Path = BENCHMARKS_ROOT,
) -> dict[str, Workload]:
    """Load all 99 TPC-DS queries from `benchmarks/sql_benchmarks/tpcds/benchmarks/`."""
    workloads: dict[str, Workload] = {}
    for qnum in range(1, 100):
        bench_path = benchmarks_dir / f"q{qnum:02d}.benchmark"
        if not bench_path.is_file():
            raise FileNotFoundError(f"missing TPC-DS benchmark file: {bench_path}")
        workload = parse_benchmark_file(bench_path, benchmarks_root)
        workloads[workload.name] = workload
    return workloads


WORKLOADS: dict[str, Workload] = load_tpcds_workloads()
MULTI_STATEMENT_QUERIES: tuple[str, ...] = tuple(
    name for name, w in WORKLOADS.items() if w.statement_count > 1
)


def normalize_query_name(token: str) -> str:
    """Accept `q1`, `q01`, `Q01`, or `1` and normalize to `q01`..`q99`."""
    raw = token.strip().lower()
    m = re.fullmatch(r"q?(\d{1,2})", raw)
    if m:
        qnum = int(m.group(1))
        if 1 <= qnum <= 99:
            return f"q{qnum:02d}"
    return raw


def resolve_queries(
    raw_queries: list[str] | None, default: tuple[str, ...] = STANDARD_TPCDS_QUERIES
) -> list[str]:
    """Expand CLI --queries tokens supporting space/comma lists, ranges (`q01-q10`), and 'all'."""
    if not raw_queries:
        return list(default)
    resolved: list[str] = []
    seen: set[str] = set()
    for item in raw_queries:
        for token in item.split(","):
            tok = token.strip().lower()
            if not tok:
                continue
            if tok == "all":
                expanded = STANDARD_TPCDS_QUERIES
            elif "-" in tok:
                parts = tok.split("-", 1)
                start_q = normalize_query_name(parts[0])
                end_q = normalize_query_name(parts[1])
                if start_q not in WORKLOADS or end_q not in WORKLOADS:
                    raise ValueError(f"invalid query range '{tok}'")
                lo = WORKLOADS[start_q].query_num
                hi = WORKLOADS[end_q].query_num
                if lo > hi:
                    raise ValueError(f"descending query range '{tok}'")
                expanded = tuple(f"q{i:02d}" for i in range(lo, hi + 1))
            else:
                name = normalize_query_name(tok)
                if name not in WORKLOADS:
                    raise ValueError(
                        f"unknown query '{token}'; expected 'all', 'q01'..'q99', or range 'q01-q10'"
                    )
                expanded = (name,)
            for q in expanded:
                if q not in seen:
                    seen.add(q)
                    resolved.append(q)
    if not resolved:
        raise ValueError("--queries must select at least one query")
    return resolved


@dataclass(frozen=True)
class Arm:
    name: str
    engine: str  # "datafusion", "duckdb", or "clickhouse"
    mode: str | None  # "off", "required", or None
    profile: str
    label: str


ARMS: dict[str, Arm] = {
    "stock_default": Arm(
        name="stock_default",
        engine="datafusion",
        mode="off",
        profile="default",
        label="Stock DataFusion (default; pushdown_filters=false, pruning=true)",
    ),
    "stock_pushdown": Arm(
        name="stock_pushdown",
        engine="datafusion",
        mode="off",
        profile="pushdown",
        label="Stock DataFusion (pushdown_filters=true, reorder_filters=true)",
    ),
    "morsel_pushdown": Arm(
        name="morsel_pushdown",
        engine="datafusion",
        mode="required",
        profile="pushdown",
        label="Morsel DataFusion (required; pushdown_filters=true, reorder_filters=true)",
    ),
    "morsel_pruning": Arm(
        name="morsel_pruning",
        engine="datafusion",
        mode="required",
        profile="pruning",
        label="Morsel DataFusion (required; pushdown_filters=false, pruning=true)",
    ),
    "morsel_no_pushdown": Arm(
        name="morsel_no_pushdown",
        engine="datafusion",
        mode="required",
        profile="no_pushdown",
        label="Morsel DataFusion (required; pushdown_filters=false, pruning=false)",
    ),
    "stock_no_pushdown": Arm(
        name="stock_no_pushdown",
        engine="datafusion",
        mode="off",
        profile="no_pushdown",
        label="Stock DataFusion (pushdown_filters=false, pruning=false)",
    ),
    "duckdb_default": Arm(
        name="duckdb_default",
        engine="duckdb",
        mode=None,
        profile="default",
        label="DuckDB 2.0.1 (default settings)",
    ),
    "clickhouse_default": Arm(
        name="clickhouse_default",
        engine="clickhouse",
        mode=None,
        profile="default",
        label="ClickHouse clickhouse-local (standard SQL settings)",
    ),
}

DEFAULT_4WAY_ARMS: tuple[str, ...] = (
    "stock_default",
    "stock_pushdown",
    "morsel_pushdown",
    "duckdb_default",
    "clickhouse_default",
)


def resolve_arms(
    raw_arms: list[str] | None, default: tuple[str, ...] = DEFAULT_4WAY_ARMS
) -> tuple[str, ...]:
    if not raw_arms:
        return default
    resolved: list[str] = []
    seen: set[str] = set()
    for item in raw_arms:
        for token in item.split(","):
            name = token.strip()
            if not name:
                continue
            if name not in ARMS:
                valid = ", ".join(ARMS.keys())
                raise ValueError(f"unknown arm '{name}'; expected one of: {valid}")
            if name not in seen:
                seen.add(name)
                resolved.append(name)
    if not resolved:
        raise ValueError("--arms must select at least one arm")
    return tuple(resolved)


def find_clickhouse_binary(explicit: Path | None = None) -> Path | None:
    """Locate an existing clickhouse binary outside Git."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_bin = os.environ.get("CLICKHOUSE_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    for path_name in ("clickhouse", "clickhouse-local"):
        found = shutil.which(path_name)
        if found:
            candidates.append(Path(found))
    candidates.extend(
        [
            Path.home() / ".local" / "bin" / "clickhouse",
            Path.home() / ".cache" / "clickhouse" / "clickhouse",
        ]
    )
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    return None


def clickhouse_version(binary: Path) -> str:
    res = subprocess.run(
        [str(binary), "local", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"clickhouse --version failed (rc={res.returncode}): {res.stderr.strip()}"
        )
    return res.stdout.strip()


def _write_ordered_table_shards(
    con: duckdb.DuckDBPyConnection,
    table: str,
    pk_cols: str,
    nfiles: int,
    table_dir: Path,
    row_group_size: int,
) -> dict[str, int]:
    total_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]
    reader = con.sql(f"SELECT * FROM {table} ORDER BY {pk_cols}").to_arrow_reader(
        batch_size=row_group_size
    )
    schema = reader.schema

    if total_rows == 0:
        out_file = table_dir / "part-000000.parquet"
        empty = pa.Table.from_batches([], schema=schema)
        with pq.ParquetWriter(
            out_file,
            schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        ) as writer:
            writer.write_table(empty, row_group_size=row_group_size)
        return {
            "files": 1,
            "rows": 0,
            "row_groups": 0,
            "parquet_bytes": out_file.stat().st_size,
        }

    actual_files = min(nfiles, max(1, total_rows))
    bounds = [
        ((idx * total_rows) // actual_files, ((idx + 1) * total_rows) // actual_files)
        for idx in range(actual_files)
    ]
    bounds = [(lo, hi) for lo, hi in bounds if lo < hi]

    written_files = 0
    written_rgs = 0
    written_bytes = 0

    shard_idx = 0
    shard_batches: list[pa.RecordBatch] = []
    shard_rows = 0
    row_pos = 0

    def flush_shard(idx: int, batches: list[pa.RecordBatch]):
        nonlocal written_files, written_rgs, written_bytes
        shard = pa.Table.from_batches(batches, schema=schema)
        out_file = table_dir / f"part-{idx:06d}.parquet"
        with pq.ParquetWriter(
            out_file,
            schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        ) as writer:
            writer.write_table(shard, row_group_size=row_group_size)
        meta = pq.read_metadata(out_file)
        written_files += 1
        written_rgs += meta.num_row_groups
        written_bytes += out_file.stat().st_size

    for batch in reader:
        if batch.num_rows == 0:
            continue
        offset = 0
        while offset < batch.num_rows:
            lo, hi = bounds[shard_idx]
            need = (hi - lo) - shard_rows
            take = min(need, batch.num_rows - offset)
            shard_batches.append(batch.slice(offset, take))
            shard_rows += take
            offset += take
            row_pos += take
            if shard_rows == hi - lo:
                flush_shard(shard_idx, shard_batches)
                shard_batches = []
                shard_rows = 0
                shard_idx += 1

    if row_pos != total_rows or shard_idx != len(bounds):
        raise RuntimeError(
            f"table {table}: wrote {row_pos}/{total_rows} rows across {shard_idx}/{len(bounds)} shards"
        )
    return {
        "files": written_files,
        "rows": total_rows,
        "row_groups": written_rgs,
        "parquet_bytes": written_bytes,
    }


def generate(path: Path, sf: float, row_group_size: int = DEFAULT_ROW_GROUP_SIZE):
    """Deterministically generate all 24 standard TPC-DS Parquet tables via DuckDB dsdgen."""
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    con = duckdb.connect(":memory:")
    con.execute("SET threads = 1")
    con.execute(f"CALL dsdgen(sf={sf})")

    summary = {}
    for table, (pk_cols, nfiles) in TPCDS_TABLES.items():
        table_dir = path / table
        table_dir.mkdir()
        summary[table] = _write_ordered_table_shards(
            con, table, pk_cols, nfiles, table_dir, row_group_size
        )
        con.execute(f"DROP TABLE {table}")
    con.close()
    return summary


def table_files(path: Path, table: str) -> list[Path]:
    table_dir = path / table
    if not table_dir.is_dir():
        raise ValueError(f"missing TPC-DS table directory: {table_dir}")
    files = sorted(table_dir.iterdir())
    if not files or any(
        not p.is_file() or not p.match("part-*.parquet") for p in files
    ):
        raise ValueError(f"expected {table_dir} to contain only part-*.parquet files")
    return files


def dataset_summary(path: Path) -> dict:
    tables = {}
    all_files = []
    for table in TPCDS_TABLES:
        files = table_files(path, table)
        all_files.extend(files)
        metas = [pq.read_metadata(f) for f in files]
        tables[table] = {
            "files": len(files),
            "rows": sum(m.num_rows for m in metas),
            "row_groups": sum(m.num_row_groups for m in metas),
            "parquet_bytes": sum(f.stat().st_size for f in files),
        }
    return {
        "tables": tables,
        "total_files": len(all_files),
        "total_rows": sum(t["rows"] for t in tables.values()),
        "total_row_groups": sum(t["row_groups"] for t in tables.values()),
        "total_parquet_bytes": sum(t["parquet_bytes"] for t in tables.values()),
    }


def cache_pages(files: list[Path]) -> dict[str, int] | None:
    """Sample Linux file page-cache residency without faulting in the data."""
    if platform.system() != "Linux":
        return None
    mincore = ctypes.CDLL(None, use_errno=True).mincore
    resident = total = 0
    for path in files:
        with path.open("rb") as file, mmap.mmap(
            file.fileno(), 0, access=mmap.ACCESS_COPY
        ) as mem:
            pages = (len(mem) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            status = (ctypes.c_ubyte * pages)()
            address = ctypes.addressof(ctypes.c_char.from_buffer(mem))
            if (
                mincore(ctypes.c_void_p(address), ctypes.c_size_t(len(mem)), status)
                != 0
            ):
                raise OSError(ctypes.get_errno(), "mincore failed", str(path))
            resident += sum(p & 1 for p in status)
            total += pages
    return {"resident": resident, "total": total}


def warm_files(files: list[Path]):
    for path in files:
        with path.open("rb") as file:
            while file.read(1 << 20):
                pass


def require_warm_cache(files: list[Path]) -> dict[str, int]:
    pages = cache_pages(files)
    if pages is None:
        raise ValueError("warm cache verification requires Linux mincore")
    if pages["resident"] != pages["total"]:
        raise ValueError(f"cannot verify warm file page cache: {pages}")
    return pages


def evict_files(files: list[Path]):
    if platform.system() != "Linux" or not hasattr(os, "posix_fadvise"):
        raise ValueError("cold cache eviction requires Linux posix_fadvise")
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def require_cold_cache(files: list[Path]) -> dict[str, int]:
    pages = cache_pages(files)
    if pages is None:
        raise ValueError("cold cache verification requires Linux mincore")
    if pages["resident"] != 0:
        raise ValueError(f"cannot verify cold file page cache: {pages}")
    return pages


class ClickHouseDialectError(RuntimeError):
    """Raised when ClickHouse fails to parse or execute unmodified TPC-DS SQL."""


class DataFusionRunner:
    def __init__(
        self,
        binary: Path,
        data_dir: Path,
        threads: int,
        timeout: int,
        mode: str,
        profile: str,
    ):
        self.binary = binary.resolve(strict=True)
        self.data_dir = data_dir.resolve(strict=True)
        self.threads = threads
        self.timeout = timeout
        self.mode = mode
        self.profile = profile

    def _run(
        self, operation: str, workload: Workload, stmt_index: int | None = None
    ) -> bytes:
        tables_csv = ",".join(workload.tables)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as sql_file:
            json.dump(list(workload.statements), sql_file)
            sql_path = Path(sql_file.name)
        try:
            cmd = [
                str(self.binary),
                operation,
                self.mode,
                self.profile,
                str(self.threads),
                str(self.data_dir),
                tables_csv,
                f"@{sql_path}",
            ]
            if stmt_index is not None:
                cmd.append(str(stmt_index))
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
        finally:
            sql_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"tpcds_runner {operation} ({self.mode}/{self.profile}) failed on "
                f"{workload.name} (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def query(self, workload: Workload, stmt_index: int = 0) -> pa.Table:
        raw = self._run("query", workload, stmt_index=stmt_index)
        return pa_ipc.open_stream(raw).read_all()

    def measure(self, workload: Workload) -> dict:
        return json.loads(self._run("measure", workload))

    def plan(self, workload: Workload) -> dict:
        return json.loads(self._run("plan", workload))


class DuckDbRunner:
    def __init__(self, data_dir: Path, threads: int, timeout: int):
        self.data_dir = data_dir.resolve(strict=True)
        self.threads = threads
        self.timeout = timeout
        self.script = Path(__file__).resolve()

    def _run(
        self, operation: str, workload: Workload, stmt_index: int | None = None
    ) -> bytes:
        tables_csv = ",".join(workload.tables)
        adapted = [adapt_sql_for_duckdb(workload.name, s) for s in workload.statements]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as sql_file:
            json.dump(adapted, sql_file)
            sql_path = Path(sql_file.name)
        try:
            cmd = [
                sys.executable,
                "-B",
                str(self.script),
                "_duckdb-exec",
                operation,
                str(self.threads),
                str(self.data_dir),
                tables_csv,
                str(sql_path),
            ]
            if stmt_index is not None:
                cmd.extend(["--stmt-index", str(stmt_index)])
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
        finally:
            sql_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"duckdb {operation} failed on {workload.name} "
                f"(rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def query(self, workload: Workload, stmt_index: int = 0) -> pa.Table:
        raw = self._run("query", workload, stmt_index=stmt_index)
        return pa_ipc.open_stream(raw).read_all()

    def measure(self, workload: Workload) -> dict:
        return json.loads(self._run("measure", workload))


class ClickHouseRunner:
    def __init__(self, binary: Path, data_dir: Path, threads: int, timeout: int):
        self.binary = binary.resolve(strict=True)
        self.data_dir = data_dir.resolve(strict=True)
        self.threads = threads
        self.timeout = timeout

    def _views_sql(self, tables: tuple[str, ...]) -> str:
        lines = []
        for table in tables:
            glob = (
                (self.data_dir / table / "part-*.parquet").as_posix().replace("'", "''")
            )
            lines.append(
                f"CREATE VIEW {table} AS SELECT * FROM file('{glob}', 'Parquet');"
            )
        return "\n".join(lines)

    def _extract_error(self, stderr_text: str) -> str:
        lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        for line in lines:
            if "DB::Exception" in line or line.startswith("Code:"):
                return line
        return lines[-1] if lines else "unknown clickhouse-local error"

    def query(self, workload: Workload, stmt_index: int = 0) -> pa.Table:
        script = (
            self._views_sql(workload.tables)
            + "\n"
            + f"{workload.statements[stmt_index]}\nFORMAT ArrowStream;\n"
        )
        res = subprocess.run(
            [
                str(self.binary),
                "local",
                f"--max_threads={self.threads}",
                *CLICKHOUSE_SQL_FLAGS,
                "--progress=0",
                "-n",
            ],
            input=script.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if res.returncode != 0:
            raise ClickHouseDialectError(
                self._extract_error(res.stderr.decode(errors="replace"))
            )
        return pa_ipc.open_stream(res.stdout).read_all()

    def measure(self, workload: Workload) -> dict:
        stmts_sql = "\n".join(f"{s}\nFORMAT Null;" for s in workload.statements)
        script = self._views_sql(workload.tables) + "\n" + stmts_sql + "\n"
        res = subprocess.run(
            [
                str(self.binary),
                "local",
                f"--max_threads={self.threads}",
                *CLICKHOUSE_SQL_FLAGS,
                "--progress=0",
                "-n",
                "--time",
            ],
            input=script.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        stderr_text = res.stderr.decode(errors="replace")
        if res.returncode != 0:
            raise ClickHouseDialectError(self._extract_error(stderr_text))
        time_lines: list[float] = []
        for line in stderr_text.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"\d+(?:\.\d+)?", stripped):
                time_lines.append(float(stripped))
        expected = len(workload.tables) + len(workload.statements)
        if len(time_lines) != expected:
            raise RuntimeError(
                f"clickhouse measure on {workload.name}: expected {expected} --time lines, "
                f"got {len(time_lines)}: {stderr_text!r}"
            )
        register_seconds = sum(time_lines[: len(workload.tables)])
        exec_seconds = sum(time_lines[len(workload.tables) :])
        return {
            "statement_count": len(workload.statements),
            "statement_rows": None,
            "output_rows": None,
            "output_batches": None,
            "register_seconds": register_seconds,
            "plan_seconds": None,
            "exec_seconds": exec_seconds,
            "exec_cpu_seconds": None,
            "first_output_seconds": None,
            "first_8192_seconds": None,
            "seconds": register_seconds + exec_seconds,
            "plan": None,
            "parquet_morsels_claimed": None,
            "bytes_scanned": None,
        }


def _duckdb_exec(
    operation: str,
    threads: int,
    data_dir: Path,
    tables: list[str],
    sql_json_path: Path,
    stmt_index: int = 0,
):
    statements = json.loads(sql_json_path.read_text())
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads = {threads}")
    con.execute("LOAD parquet")
    for _ in con.sql("SELECT 1").to_arrow_reader(batch_size=BATCH_SIZE):
        pass

    for table in tables:
        glob = (data_dir / table / "part-*.parquet").as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{glob}')")

    if operation == "query":
        tbl = con.sql(statements[stmt_index]).to_arrow_table()
        with pa_ipc.new_stream(sys.stdout.buffer, tbl.schema) as writer:
            writer.write_table(tbl)
        return

    if operation == "measure":
        start = time.perf_counter()
        cpu_start = time.process_time()
        output_rows = 0
        output_batches = 0
        first_output_seconds = None
        first_8192_seconds = None
        statement_rows: list[int] = []
        for sql in statements:
            reader = con.sql(sql).to_arrow_reader(batch_size=BATCH_SIZE)
            stmt_rows = 0
            for batch in reader:
                if batch.num_rows:
                    before = output_rows
                    output_rows += batch.num_rows
                    stmt_rows += batch.num_rows
                    output_batches += 1
                    reached_8192 = before < BATCH_SIZE and output_rows >= BATCH_SIZE
                    if output_batches == 1 or reached_8192:
                        elapsed = time.perf_counter() - start
                        if output_batches == 1:
                            first_output_seconds = elapsed
                        if reached_8192:
                            first_8192_seconds = elapsed
            statement_rows.append(stmt_rows)
        seconds = time.perf_counter() - start
        exec_cpu_seconds = time.process_time() - cpu_start
        print(
            json.dumps(
                {
                    "duckdb_version": duckdb.__version__,
                    "statement_count": len(statements),
                    "statement_rows": statement_rows,
                    "output_rows": output_rows,
                    "output_batches": output_batches,
                    "register_seconds": None,
                    "plan_seconds": None,
                    "exec_seconds": None,
                    "exec_cpu_seconds": exec_cpu_seconds,
                    "first_output_seconds": first_output_seconds,
                    "first_8192_seconds": first_8192_seconds,
                    "seconds": seconds,
                    "plan": None,
                    "parquet_morsels_claimed": None,
                    "bytes_scanned": None,
                }
            )
        )
        return

    raise ValueError(f"unknown _duckdb-exec operation: {operation}")


# ---------------------------------------------------------------------------
# Parity comparison helpers
# ---------------------------------------------------------------------------


def _values_equal(expected, actual) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(actual, float) and math.isnan(actual)
    if isinstance(actual, float) and math.isnan(actual):
        return False
    if isinstance(expected, Decimal) and isinstance(actual, Decimal):
        if expected == actual:
            return True
        scale = max(-expected.as_tuple().exponent, -actual.as_tuple().exponent, 0)
        if scale > 0:
            # Allow 1 ULP at the coarser decimal scale for cross-engine Decimal division/AVG.
            coarser = min(-expected.as_tuple().exponent, -actual.as_tuple().exponent)
            tol = max(10.0 ** (-max(coarser, 0)), 1e-6 * max(abs(float(expected)), 1.0))
            return math.isclose(
                float(expected), float(actual), rel_tol=1e-6, abs_tol=tol
            )
        return False
    if isinstance(expected, Decimal) and isinstance(actual, int):
        return expected == Decimal(actual)
    if isinstance(expected, int) and isinstance(actual, Decimal):
        return Decimal(expected) == actual
    if isinstance(expected, float) and isinstance(actual, Decimal):
        scale = max(-actual.as_tuple().exponent, 0)
        tol = max(10.0 ** (-scale), 1e-6 * max(abs(expected), 1.0))
        return math.isclose(expected, float(actual), rel_tol=1e-6, abs_tol=tol)
    if isinstance(expected, Decimal) and isinstance(actual, float):
        scale = max(-expected.as_tuple().exponent, 0)
        tol = max(10.0 ** (-scale), 1e-6 * max(abs(actual), 1.0))
        return math.isclose(float(expected), actual, rel_tol=1e-6, abs_tol=tol)
    if isinstance(expected, float) and isinstance(actual, float):
        return math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected == actual
    return expected == actual


def _sort_key(row: tuple):
    key = []
    for v in row:
        if v is None:
            key.append((0, ""))
        elif isinstance(v, bool):
            key.append((1, Decimal(int(v))))
        elif isinstance(v, int):
            key.append((1, Decimal(v)))
        elif isinstance(v, Decimal):
            key.append((1, v))
        elif isinstance(v, float):
            if math.isnan(v):
                key.append((2, "nan"))
            else:
                key.append((1, Decimal(f"{v:.12g}")))
        else:
            key.append((3, str(v)))
    return tuple(key)


def _rows_match_exact(
    oracle_rows: list[tuple], actual_rows: list[tuple]
) -> tuple[int, int] | None:
    """Return None if rows match pairwise, or (row_idx, col_idx) of first mismatch."""
    for r_idx, (exp_row, act_row) in enumerate(zip(oracle_rows, actual_rows)):
        for c_idx, (exp_val, act_val) in enumerate(zip(exp_row, act_row)):
            if not _values_equal(exp_val, act_val):
                return (r_idx, c_idx)
    return None


def compare_tables(
    oracle: pa.Table,
    actual: pa.Table,
    ordered: bool,
    case_name: str,
    arm_name: str,
):
    if oracle.num_columns != actual.num_columns:
        raise ValueError(
            f"{arm_name}/{case_name}: column count mismatch: "
            f"expected {oracle.num_columns} ({oracle.schema.names}), "
            f"got {actual.num_columns} ({actual.schema.names})"
        )
    if oracle.num_rows != actual.num_rows:
        raise ValueError(
            f"{arm_name}/{case_name}: row count mismatch: "
            f"expected {oracle.num_rows}, got {actual.num_rows}"
        )
    if oracle.num_rows == 0:
        return

    oracle_rows = list(zip(*(col.to_pylist() for col in oracle.columns)))
    actual_rows = list(zip(*(col.to_pylist() for col in actual.columns)))

    if ordered:
        mismatch = _rows_match_exact(oracle_rows, actual_rows)
        if mismatch is None:
            return
        # Check if rows match as a multiset (can happen on ORDER BY with NULLS FIRST/LAST
        # defaults or ties in non-unique sort keys across engines).
        sorted_oracle = sorted(oracle_rows, key=_sort_key)
        sorted_actual = sorted(actual_rows, key=_sort_key)
        if _rows_match_exact(sorted_oracle, sorted_actual) is None:
            return
        r_idx, c_idx = mismatch
        raise ValueError(
            f"{arm_name}/{case_name}: mismatch at row {r_idx}, col '{oracle.schema.names[c_idx]}': "
            f"expected {oracle_rows[r_idx][c_idx]!r}, got {actual_rows[r_idx][c_idx]!r}"
        )

    oracle_rows.sort(key=_sort_key)
    actual_rows.sort(key=_sort_key)
    mismatch = _rows_match_exact(oracle_rows, actual_rows)
    if mismatch is not None:
        r_idx, c_idx = mismatch
        raise ValueError(
            f"{arm_name}/{case_name}: unordered mismatch at sorted row {r_idx}, "
            f"col '{oracle.schema.names[c_idx]}': "
            f"expected {oracle_rows[r_idx][c_idx]!r}, got {actual_rows[r_idx][c_idx]!r}"
        )


# ---------------------------------------------------------------------------
# Morsel plan validation & rejection categorization
# ---------------------------------------------------------------------------

_NATIVE_PARQUET_RE = re.compile(r"NativeParquet\[([^\], ]+)")


def validate_plan(
    evidence: dict, arm: Arm, workload: Workload, table_row_groups: dict[str, int]
):
    if arm.engine != "datafusion":
        return
    plans = evidence.get("plans") or (
        [evidence["plan"]] if evidence.get("plan") else []
    )
    if not plans:
        raise ValueError(f"{arm.name}/{workload.name}: missing plan output")
    if arm.mode == "off":
        for p in plans:
            if "MorselExec:" in p or not re.search(
                r"DataSourceExec: .*file_type=parquet", p
            ):
                raise ValueError(
                    f"{arm.name}/{workload.name}: expected stock Parquet plan, got:\n{p}"
                )
        return

    # mode == "required"
    scans: list[str] = []
    for p in plans:
        if not p.startswith("MorselExec:"):
            raise ValueError(
                f"{arm.name}/{workload.name}: expected MorselExec plan, got:\n{p}"
            )
        scans.extend(_NATIVE_PARQUET_RE.findall(p))
    missing_tables = [t for t in workload.tables if t not in scans]
    if missing_tables:
        raise ValueError(
            f"{arm.name}/{workload.name}: expected NativeParquet for {workload.tables} "
            f"(missing {missing_tables})"
        )
    expected_rgs = sum(table_row_groups[t] for t in scans)
    claimed = evidence["parquet_morsels_claimed"]
    short_circuits = (evidence.get("join_empty_build_short_circuits") or 0) + (
        evidence.get("join_null_aware_short_circuits") or 0
    )
    if short_circuits == 0 and expected_rgs > 0:
        if claimed is None or claimed <= 0 or claimed > expected_rgs:
            raise ValueError(
                f"{arm.name}/{workload.name}: expected 1..={expected_rgs} row groups "
                f"claimed, got {claimed}"
            )


def classify_morsel_rejection(error: str) -> str:
    """Map a raw `morsel_mode=required` planning error to a clean blocker category."""
    low = error.lower()
    if "unsupported logical operator: window" in low or "windowaggr" in low:
        return "Window"
    if (
        "grouping sets" in low
        or "rollup" in low
        or "cube" in low
        or "groupingset" in low
    ):
        return "ROLLUP / GROUPING SETS / CUBE"
    if (
        "unsupported logical operator: union" in low
        or "unsupported logical operator: intersect" in low
        or "unsupported logical operator: except" in low
    ):
        return "UNION / INTERSECT / EXCEPT"
    if "unsupported logical operator: limit" in low or (
        "limit" in low and "order by" in low
    ):
        return "LIMIT without ORDER BY"
    if (
        "unsupported logical operator: crossjoin" in low
        or "cross join" in low
        or "non-equi" in low
        or "join without equi" in low
        or "nested loop" in low
    ):
        return "CrossJoin / Non-equi Join"
    if "unsupported join type" in low:
        m = re.search(r"unsupported join type\s+([A-Za-z]+)", error, re.IGNORECASE)
        jtype = m.group(1) if m else "other"
        return f"Unsupported Join Type ({jtype})"
    if "unsupported join key type" in low:
        return "Unsupported Join Key Type"
    if "unsupported join key expression" in low:
        return "Unsupported Join Key Expression"
    if "unsupported join filter expression" in low:
        return "Unsupported Join Filter Expression"
    if "unsupported aggregate" in low:
        if any(tok in low for tok in ("stddev", "var_", "variance")):
            return "Unsupported Aggregate (stddev/var)"
        if "distinct" in low:
            return "Unsupported Aggregate (DISTINCT)"
        return "Unsupported Aggregate"
    if "unsupported group by" in low:
        return "Unsupported GROUP BY"
    if "unsupported sort expression" in low:
        return "Unsupported Sort Expression"
    if "unsupported filter predicate" in low:
        return "Unsupported Filter Predicate"
    if "unsupported projection expression" in low:
        return "Unsupported Projection Expression"
    if "unsupported logical operator:" in low:
        m = re.search(
            r"unsupported logical operator:\s*([A-Za-z0-9_]+)", error, re.IGNORECASE
        )
        op = m.group(1) if m else "Other"
        return f"Logical Operator ({op})"
    if "subquery" in low:
        return "Unsupported Subquery"
    if "distinct" in low:
        return "Unsupported DISTINCT"
    return f"Other ({error.splitlines()[0][:80]})"


def detect_plan_blockers(
    logical_plans: list[str | None],
    physical_plans: list[str | None],
    morsel_error: str | None,
) -> list[str]:
    """Identify all known Morsel-blocking operators present across a query's logical plans.

    Because `MorselQueryPlanner` compiles the optimized `LogicalPlan` (not Stock's
    post-join-swap physical plan) and fails fast on the first unsupported operator,
    inspecting the optimized logical plan exposes every blocking operator inside a
    rejected query without false positives from Stock physical join-side swaps.
    """
    del physical_plans  # Retained in signature for callers; LogicalPlan is authoritative.
    blockers: set[str] = set()
    primary = classify_morsel_rejection(morsel_error) if morsel_error else None
    if primary is not None:
        blockers.add(primary)

    joined_logical = "\n".join(p for p in logical_plans if p)

    if re.search(r"\bWindowAggr:", joined_logical):
        blockers.add("Window")

    if re.search(
        r"\b(?:ROLLUP|CUBE|GROUPING SETS)\b", joined_logical, re.IGNORECASE
    ) or (
        "GroupingSet" in joined_logical
        or "__grouping_id" in joined_logical
        or re.search(r"\bgrouping\(", joined_logical, re.IGNORECASE)
    ):
        blockers.add("ROLLUP / GROUPING SETS / CUBE")

    if re.search(r"^\s*Union\b", joined_logical, re.MULTILINE):
        blockers.add("UNION / INTERSECT / EXCEPT")

    for jtype in ("LeftMark", "RightMark"):
        if re.search(rf"\b{jtype}\s+Join:", joined_logical):
            blockers.add(f"Unsupported Join Type ({jtype})")

    if re.search(
        r"\b(?:stddev(?:_samp|_pop)?|var_samp|var_pop)\(", joined_logical, re.IGNORECASE
    ):
        blockers.add("Unsupported Aggregate (stddev/var)")

    return sorted(blockers)


def summarize_rejection_audit(query_results: dict[str, dict]) -> dict:
    """Compute primary-rejection frequencies, full-blocker frequencies, and unlock analysis."""
    admitted = sorted(q for q, r in query_results.items() if r["admitted"])
    rejected = sorted(q for q, r in query_results.items() if not r["admitted"])

    primary_groups: dict[str, list[str]] = {}
    blocker_groups: dict[str, list[str]] = {}
    for q in rejected:
        info = query_results[q]
        cat = info["primary_rejection_category"]
        primary_groups.setdefault(cat, []).append(q)
        for b in info["all_blockers"]:
            blocker_groups.setdefault(b, []).append(q)

    primary_breakdown = sorted(
        [
            {"category": cat, "count": len(qs), "queries": qs}
            for cat, qs in primary_groups.items()
        ],
        key=lambda item: (-item["count"], item["category"]),
    )
    blocker_breakdown = sorted(
        [
            {
                "blocker": b,
                "blocked_count": len(qs),
                "sole_blocker_count": sum(
                    1 for q in qs if query_results[q]["all_blockers"] == [b]
                ),
                "queries": qs,
            }
            for b, qs in blocker_groups.items()
        ],
        key=lambda item: (
            -item["blocked_count"],
            -item["sole_blocker_count"],
            item["blocker"],
        ),
    )

    # Greedy multi-operator unlock simulation: iteratively pick the operator that unlocks
    # the most newly-unblocked queries (tie-break by total blocked_count).
    remaining = {q: set(query_results[q]["all_blockers"]) for q in rejected}
    all_ops = sorted(blocker_groups.keys())
    picked: list[str] = []
    unlock_steps: list[dict] = []
    while remaining and all_ops:
        best_op = None
        best_key = None
        for op in all_ops:
            if op in picked:
                continue
            newly_unlocked = sorted(q for q, bs in remaining.items() if bs == {op})
            still_blocking = sum(1 for bs in remaining.values() if op in bs)
            if still_blocking == 0:
                continue
            key = (len(newly_unlocked), still_blocking, op)
            if best_key is None or (key[0], key[1]) > (best_key[0], best_key[1]):
                best_key = key
                best_op = op
        if best_op is None:
            break
        picked.append(best_op)
        newly_unlocked = sorted(q for q, bs in remaining.items() if bs == {best_op})
        for q in list(remaining.keys()):
            remaining[q].discard(best_op)
            if not remaining[q]:
                del remaining[q]
        unlock_steps.append(
            {
                "operator": best_op,
                "newly_unlocked_count": len(newly_unlocked),
                "newly_unlocked_queries": newly_unlocked,
                "cumulative_admitted": len(admitted)
                + sum(s["newly_unlocked_count"] for s in unlock_steps)
                + len(newly_unlocked),
            }
        )
        if len(unlock_steps) >= 8:
            break

    return {
        "total_queries": len(query_results),
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "admitted_queries": admitted,
        "rejected_queries": rejected,
        "primary_rejection_breakdown": primary_breakdown,
        "all_blocker_breakdown": blocker_breakdown,
        "greedy_unlock_roadmap": unlock_steps,
    }


def make_runner(
    arm: Arm,
    runner_bin: Path,
    data_dir: Path,
    threads: int,
    timeout: int,
    clickhouse_bin: Path | None = None,
):
    if arm.engine == "duckdb":
        return DuckDbRunner(data_dir, threads, timeout)
    if arm.engine == "clickhouse":
        if clickhouse_bin is None:
            raise ValueError("ClickHouse binary required for clickhouse_default arm")
        return ClickHouseRunner(clickhouse_bin, data_dir, threads, timeout)
    return DataFusionRunner(
        runner_bin, data_dir, threads, timeout, arm.mode, arm.profile
    )


def audit_suite(
    data_dir: Path,
    runner_bin: Path,
    threads: int,
    timeout: int,
    query_names: list[str],
    verify_parity: bool = True,
) -> dict:
    """Audit Morsel admission/rejection across queries and verify parity on admitted queries."""
    summary = dataset_summary(data_dir)
    table_rgs = {name: info["row_groups"] for name, info in summary["tables"].items()}
    morsel_planner = DataFusionRunner(
        runner_bin, data_dir, threads, timeout, "required", "pushdown"
    )
    stock_planner = DataFusionRunner(
        runner_bin, data_dir, threads, timeout, "off", "default"
    )
    duckdb_oracle = DuckDbRunner(data_dir, threads, timeout)

    query_results: dict[str, dict] = {}
    for qname in query_names:
        workload = WORKLOADS[qname]
        morsel_plan = morsel_planner.plan(workload)
        admitted = bool(morsel_plan["admitted"])
        entry: dict = {
            "query": qname,
            "statement_count": workload.statement_count,
            "tables": list(workload.tables),
            "admitted": admitted,
            "morsel_error": morsel_plan["error"],
            "primary_rejection_category": None,
            "all_blockers": [],
            "statement_admitted": [s["admitted"] for s in morsel_plan["statements"]],
            "parity_verified": False,
            "output_rows": None,
        }
        if admitted:
            if verify_parity:
                total_rows = 0
                for idx in range(workload.statement_count):
                    oracle_tbl = duckdb_oracle.query(workload, stmt_index=idx)
                    stock_tbl = stock_planner.query(workload, stmt_index=idx)
                    morsel_tbl = morsel_planner.query(workload, stmt_index=idx)
                    ordered = workload.ordered_statements[idx]
                    case_label = (
                        f"{qname}[{idx}]" if workload.statement_count > 1 else qname
                    )
                    # Always enforce exact Morsel == Stock DataFusion parity.
                    compare_tables(
                        stock_tbl, morsel_tbl, ordered, case_label, "morsel_vs_stock"
                    )
                    if qname not in DATAFUSION_INT_DIV_QUERIES:
                        compare_tables(
                            oracle_tbl, stock_tbl, ordered, case_label, "stock_default"
                        )
                        compare_tables(
                            oracle_tbl,
                            morsel_tbl,
                            ordered,
                            case_label,
                            "morsel_pushdown",
                        )
                    else:
                        entry["cross_engine_note"] = (
                            "DataFusion INT/INT truncating division vs DuckDB float division; "
                            "exact Morsel == Stock parity verified"
                        )
                    total_rows += stock_tbl.num_rows
                sample = morsel_planner.measure(workload)
                validate_plan(sample, ARMS["morsel_pushdown"], workload, table_rgs)
                if sample["output_rows"] != total_rows:
                    raise ValueError(
                        f"morsel_pushdown/{qname}: measure rows {sample['output_rows']} != {total_rows}"
                    )
                entry["parity_verified"] = True
                entry["output_rows"] = total_rows
        else:
            stock_plan = stock_planner.plan(workload)
            logical_plans = [s["logical_plan"] for s in stock_plan["statements"]]
            physical_plans = [s["plan"] for s in stock_plan["statements"]]
            entry["primary_rejection_category"] = classify_morsel_rejection(
                morsel_plan["error"] or "unknown"
            )
            entry["all_blockers"] = detect_plan_blockers(
                logical_plans, physical_plans, morsel_plan["error"]
            )
        query_results[qname] = entry

    rollup = summarize_rejection_audit(query_results)
    return {
        "dataset": summary,
        "summary": rollup,
        "queries": query_results,
    }


def verify_suite(
    data_dir: Path,
    runner_bin: Path,
    threads: int,
    timeout: int,
    query_names: list[str],
    arms: tuple[str, ...] = DEFAULT_4WAY_ARMS,
    clickhouse_bin: Path | None = None,
) -> dict:
    """Verify 4-way parity (DuckDB oracle vs Stock, Morsel on admitted, ClickHouse)."""
    summary = dataset_summary(data_dir)
    table_rgs = {name: info["row_groups"] for name, info in summary["tables"].items()}
    duckdb_oracle = DuckDbRunner(data_dir, threads, timeout)
    morsel_checker = DataFusionRunner(
        runner_bin, data_dir, threads, timeout, "required", "pushdown"
    )
    stock_planner = DataFusionRunner(
        runner_bin, data_dir, threads, timeout, "off", "default"
    )

    verified: dict[str, dict] = {}
    morsel_audit: dict[str, dict] = {}
    clickhouse_incompatible: dict[str, str] = {}
    clickhouse_parity_notes: dict[str, str] = {}
    datafusion_int_div_notes: dict[str, str] = {}

    for qname in query_names:
        workload = WORKLOADS[qname]
        morsel_plan = morsel_checker.plan(workload)
        admitted = bool(morsel_plan["admitted"])
        if admitted:
            morsel_audit[qname] = {
                "admitted": True,
                "morsel_error": None,
                "primary_rejection_category": None,
                "all_blockers": [],
            }
        else:
            stock_plan = stock_planner.plan(workload)
            logical_plans = [s["logical_plan"] for s in stock_plan["statements"]]
            physical_plans = [s["plan"] for s in stock_plan["statements"]]
            morsel_audit[qname] = {
                "admitted": False,
                "morsel_error": morsel_plan["error"],
                "primary_rejection_category": classify_morsel_rejection(
                    morsel_plan["error"] or "unknown"
                ),
                "all_blockers": detect_plan_blockers(
                    logical_plans, physical_plans, morsel_plan["error"]
                ),
            }

        oracle_tables = [
            duckdb_oracle.query(workload, stmt_index=i)
            for i in range(workload.statement_count)
        ]
        stock_oracle_tables = [
            stock_planner.query(workload, stmt_index=i)
            for i in range(workload.statement_count)
        ]
        duckdb_rows = sum(t.num_rows for t in oracle_tables)
        stock_rows = sum(t.num_rows for t in stock_oracle_tables)
        if qname in DATAFUSION_INT_DIV_QUERIES:
            datafusion_int_div_notes[qname] = (
                "DataFusion INT/INT truncating division vs DuckDB float division; "
                "all DataFusion arms verified identical to stock_default"
            )
        else:
            for idx, (o_tbl, s_tbl) in enumerate(
                zip(oracle_tables, stock_oracle_tables)
            ):
                case_label = (
                    f"{qname}[{idx}]" if workload.statement_count > 1 else qname
                )
                compare_tables(
                    o_tbl,
                    s_tbl,
                    workload.ordered_statements[idx],
                    case_label,
                    "stock_default",
                )

        verified[qname] = {
            "rows": stock_rows,
            "duckdb_rows": duckdb_rows,
            "statement_rows": [t.num_rows for t in stock_oracle_tables],
            "morsel_admitted": admitted,
            "arms": [],
        }

        for arm_name in arms:
            arm = ARMS[arm_name]
            if arm.mode == "required" and not admitted:
                continue
            if arm.engine == "clickhouse" and clickhouse_bin is None:
                continue
            runner = make_runner(
                arm, runner_bin, data_dir, threads, timeout, clickhouse_bin
            )
            if arm.engine == "clickhouse":
                try:
                    for idx, oracle_tbl in enumerate(oracle_tables):
                        ch_tbl = runner.query(workload, stmt_index=idx)
                        case_label = (
                            f"{qname}[{idx}]" if workload.statement_count > 1 else qname
                        )
                        try:
                            compare_tables(
                                oracle_tbl,
                                ch_tbl,
                                workload.ordered_statements[idx],
                                case_label,
                                arm_name,
                            )
                        except ValueError as diff_err:
                            clickhouse_parity_notes[qname] = str(diff_err)
                    verified[qname]["arms"].append(arm_name)
                except ClickHouseDialectError as err:
                    clickhouse_incompatible[qname] = str(err)
                continue

            if arm.engine == "datafusion":
                if arm_name != "stock_default":
                    for idx, s_tbl in enumerate(stock_oracle_tables):
                        actual_tbl = runner.query(workload, stmt_index=idx)
                        case_label = (
                            f"{qname}[{idx}]" if workload.statement_count > 1 else qname
                        )
                        compare_tables(
                            s_tbl,
                            actual_tbl,
                            workload.ordered_statements[idx],
                            case_label,
                            arm_name,
                        )
                sample = runner.measure(workload)
                validate_plan(sample, arm, workload, table_rgs)
                if sample["output_rows"] != stock_rows:
                    raise ValueError(
                        f"{arm_name}/{qname}: measure rows {sample['output_rows']} != {stock_rows}"
                    )
                verified[qname]["arms"].append(arm_name)
            elif arm.engine == "duckdb":
                verified[qname]["arms"].append(arm_name)

    return {
        "verified": verified,
        "morsel_audit": morsel_audit,
        "morsel_summary": summarize_rejection_audit(morsel_audit),
        "clickhouse_incompatible": clickhouse_incompatible,
        "clickhouse_parity_notes": clickhouse_parity_notes,
        "datafusion_int_div_notes": datafusion_int_div_notes,
    }


def run_bench(args):
    data_dir = args.path.resolve(strict=True)
    summary = dataset_summary(data_dir)
    table_rgs = {name: info["row_groups"] for name, info in summary["tables"].items()}
    query_names = resolve_queries(args.queries, STANDARD_TPCDS_QUERIES)
    arms = resolve_arms(args.arms, DEFAULT_4WAY_ARMS)

    ch_bin = None
    ch_ver = None
    if any(ARMS[a].engine == "clickhouse" for a in arms):
        ch_bin = find_clickhouse_binary(args.clickhouse_bin)
        if ch_bin is None:
            raise ValueError(
                "clickhouse_default arm requested but no clickhouse binary found; "
                "pass --clickhouse-bin or install to ~/.local/bin/clickhouse"
            )
        ch_ver = clickhouse_version(ch_bin)

    if args.cache in ("warm", "cold") and platform.system() != "Linux":
        raise ValueError(
            f"--cache {args.cache} requires Linux mincore page-cache verification"
        )

    df_checker = DataFusionRunner(
        args.runner, data_dir, args.threads, args.timeout, "off", "default"
    )
    rev_check = df_checker.plan(WORKLOADS[query_names[0]])
    if rev_check["build_revision"] != args.revision:
        raise ValueError(
            f"runner revision {rev_check['build_revision']} != --revision {args.revision}"
        )

    verification = verify_suite(
        data_dir,
        args.runner,
        args.threads,
        args.timeout,
        query_names,
        arms=arms,
        clickhouse_bin=ch_bin,
    )

    header = {
        "record_type": "header",
        "build_revision": args.revision,
        "duckdb_version": duckdb.__version__,
        "clickhouse_version": ch_ver,
        "pyarrow_version": pa.__version__,
        "python_version": platform.python_version(),
        "host": platform.node(),
        "arch": platform.machine(),
        "kernel": platform.release(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "threads": args.threads,
        "batch_size": BATCH_SIZE,
        "cache_regime": args.cache,
        "page_size": mmap.PAGESIZE,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "arms": list(arms),
        "dataset": summary,
        "verified": verification["verified"],
        "morsel_audit": verification["morsel_audit"],
        "morsel_summary": verification["morsel_summary"],
        "clickhouse_incompatible": verification["clickhouse_incompatible"],
        "clickhouse_parity_notes": verification["clickhouse_parity_notes"],
        "datafusion_int_div_notes": verification["datafusion_int_div_notes"],
    }
    print(json.dumps(header), flush=True)

    referenced_tables = sorted({t for q in query_names for t in WORKLOADS[q].tables})
    all_referenced_files = [
        f for t in referenced_tables for f in table_files(data_dir, t)
    ]
    if args.cache == "warm":
        warm_files(all_referenced_files)
        require_warm_cache(all_referenced_files)

    for qname in query_names:
        workload = WORKLOADS[qname]
        vinfo = verification["verified"][qname]
        expected_df_rows = vinfo["rows"]
        expected_dk_rows = vinfo.get("duckdb_rows", expected_df_rows)
        active_arms = tuple(a for a in arms if a in vinfo["arms"])
        runners = {
            arm_name: make_runner(
                ARMS[arm_name],
                args.runner,
                data_dir,
                args.threads,
                args.timeout,
                ch_bin,
            )
            for arm_name in active_arms
        }
        q_files = [f for t in workload.tables for f in table_files(data_dir, t)]

        for rep in range(1 - args.warmups, args.repeats + 1):
            order = active_arms if rep % 2 else tuple(reversed(active_arms))
            for arm_name in order:
                arm = ARMS[arm_name]
                before = None
                if args.cache == "cold":
                    evict_files(q_files)
                    before = require_cold_cache(q_files)
                elif rep > 0:
                    if args.cache == "warm":
                        before = require_warm_cache(q_files)
                    else:
                        before = cache_pages(q_files)
                result = runners[arm_name].measure(workload)
                validate_plan(result, arm, workload, table_rgs)
                if (
                    arm.engine == "datafusion"
                    and result["build_revision"] != args.revision
                ):
                    raise ValueError(
                        f"runner revision mismatch: {result['build_revision']}"
                    )
                expected_arm_rows = (
                    expected_dk_rows if arm.engine == "duckdb" else expected_df_rows
                )
                if (
                    result.get("output_rows") is not None
                    and result["output_rows"] != expected_arm_rows
                ):
                    raise ValueError(
                        f"{arm_name}/{qname}: unexpected output_rows {result['output_rows']} != {expected_arm_rows}"
                    )
                if rep > 0:
                    after = (
                        require_warm_cache(q_files)
                        if args.cache == "warm"
                        else cache_pages(q_files)
                    )
                    result.pop("plan", None)
                    result.pop("plans", None)
                    result.pop("build_revision", None)
                    print(
                        json.dumps(
                            {
                                "record_type": "trial",
                                "case": qname,
                                "arm": arm_name,
                                "engine": arm.engine,
                                "mode": arm.mode,
                                "profile": arm.profile,
                                "rep": rep,
                                **result,
                                "cache_pages_before": before,
                                "cache_pages_after": after,
                            }
                        ),
                        flush=True,
                    )


def format_audit_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"### Morsel Admission Summary ({s['admitted_count']}/{s['total_queries']} admitted, "
        f"{s['rejected_count']}/{s['total_queries']} rejected)",
        "",
        "Admitted queries: "
        + (", ".join(f"`{q}`" for q in s["admitted_queries"]) or "none"),
        "",
        "#### Primary `morsel_mode=required` Rejection Breakdown (First-Hit)",
        "",
        "| Rejection Category | Queries Blocked | Query IDs |",
        "| --- | ---: | --- |",
    ]
    for row in s["primary_rejection_breakdown"]:
        qlist = ", ".join(f"`{q}`" for q in row["queries"])
        lines.append(f"| {row['category']} | {row['count']} | {qlist} |")

    lines.extend(
        [
            "",
            "#### All Blocking Operators Present Across Rejected Queries",
            "",
            "| Blocking Operator | Queries Containing Blocker | Sole-Blocker Queries |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in s["all_blocker_breakdown"]:
        lines.append(
            f"| {row['blocker']} | {row['blocked_count']} | {row['sole_blocker_count']} |"
        )

    if s["greedy_unlock_roadmap"]:
        lines.extend(
            [
                "",
                "#### Greedy Operator Unlock Roadmap",
                "",
                "| Step | Operator Added | Newly Unlocked | Cumulative Admitted | Unlocked Queries |",
                "| ---: | --- | ---: | ---: | --- |",
            ]
        )
        for idx, step in enumerate(s["greedy_unlock_roadmap"], start=1):
            qlist = ", ".join(f"`{q}`" for q in step["newly_unlocked_queries"]) or "—"
            lines.append(
                f"| {idx} | {step['operator']} | +{step['newly_unlocked_count']} | "
                f"{step['cumulative_admitted']}/{s['total_queries']} | {qlist} |"
            )
    return "\n".join(lines)


def summarize_jsonl(path: Path) -> str:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header = lines[0]
    trials = [r for r in lines[1:] if r.get("record_type") == "trial"]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for t in trials:
        grouped.setdefault((t["case"], t["arm"]), []).append(t)

    medians = {
        key: statistics.median(r["seconds"] * 1000.0 for r in rows)
        for key, rows in grouped.items()
    }
    out: list[str] = [
        f"Host: {header['host']} ({header['arch']}, kernel {header['kernel']}), "
        f"threads={header['threads']}, cache={header['cache_regime']}, "
        f"warmups={header['warmups']}, repeats={header['repeats']}",
        f"Revision: {header['build_revision']}, DuckDB: {header['duckdb_version']}, "
        f"ClickHouse: {header.get('clickhouse_version') or 'N/A'}, "
        f"PyArrow: {header['pyarrow_version']}",
        "",
    ]

    if header.get("morsel_summary"):
        out.append(format_audit_markdown({"summary": header["morsel_summary"]}))
        out.append("")

    # 1. Admitted subset 4-way table (Morsel vs Stock vs DuckDB vs ClickHouse)
    admitted = sorted(
        q
        for q, v in (header.get("verified") or {}).items()
        if v.get("morsel_admitted") and (q, "morsel_pushdown") in medians
    )
    if admitted:
        out.append(f"### Admitted Subset 4-Way Comparison ({len(admitted)} queries)")
        out.append("")
        out.append(
            "| Query | Rows | `stock_default` ms | `stock_pushdown` ms | "
            "`morsel_pushdown` ms | `duckdb_default` ms | `clickhouse_default` ms | "
            "Morsel / Stock(def) | Morsel / DuckDB | Morsel / ClickHouse |"
        )
        out.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for q in admitted:
            rows = header["verified"][q]["rows"]
            sd = medians.get((q, "stock_default"))
            sp = medians.get((q, "stock_pushdown"))
            mp = medians.get((q, "morsel_pushdown"))
            dk = medians.get((q, "duckdb_default"))
            ch = medians.get((q, "clickhouse_default"))
            out.append(
                f"| `{q}` | {rows:,} | "
                f"{f'{sd:.1f}' if sd is not None else '—'} | "
                f"{f'{sp:.1f}' if sp is not None else '—'} | "
                f"{f'{mp:.1f}' if mp is not None else '—'} | "
                f"{f'{dk:.1f}' if dk is not None else '—'} | "
                f"{f'{ch:.1f}' if ch is not None else '—'} | "
                f"{f'{mp / sd:.2f}x' if (mp is not None and sd) else '—'} | "
                f"{f'{mp / dk:.2f}x' if (mp is not None and dk) else '—'} | "
                f"{f'{mp / ch:.2f}x' if (mp is not None and ch) else '—'} |"
            )

        def _gm(arm_name: str, qs: list[str]) -> float | None:
            vals = [medians[(q, arm_name)] for q in qs if (q, arm_name) in medians]
            return (
                statistics.geometric_mean(vals)
                if len(vals) == len(qs) and vals
                else None
            )

        sd_gm = _gm("stock_default", admitted)
        sp_gm = _gm("stock_pushdown", admitted)
        mp_gm = _gm("morsel_pushdown", admitted)
        dk_gm = _gm("duckdb_default", admitted)
        ch_shared = [q for q in admitted if (q, "clickhouse_default") in medians]
        ch_gm = _gm("clickhouse_default", ch_shared) if ch_shared else None
        mp_ch_gm = _gm("morsel_pushdown", ch_shared) if ch_shared else None

        out.append(
            f"| **GeoMean ({len(admitted)} admitted)** | — | "
            f"{f'**{sd_gm:.1f}**' if sd_gm else '—'} | "
            f"{f'**{sp_gm:.1f}**' if sp_gm else '—'} | "
            f"{f'**{mp_gm:.1f}**' if mp_gm else '—'} | "
            f"{f'**{dk_gm:.1f}**' if dk_gm else '—'} | "
            f"{f'**{ch_gm:.1f}** ({len(ch_shared)}q)' if ch_gm else '—'} | "
            f"{f'**{mp_gm / sd_gm:.2f}x**' if (mp_gm and sd_gm) else '—'} | "
            f"{f'**{mp_gm / dk_gm:.2f}x**' if (mp_gm and dk_gm) else '—'} | "
            f"{f'**{mp_ch_gm / ch_gm:.2f}x**' if (mp_ch_gm and ch_gm) else '—'} |"
        )
        out.append("")

    # 2. Full-suite table across all queries present
    all_cases = sorted({t["case"] for t in trials})
    if all_cases:
        out.append(f"### Full-Suite Timings ({len(all_cases)} queries)")
        out.append("")
        out.append(
            "| Query | Rows | Morsel | `stock_default` ms | `stock_pushdown` ms | "
            "`morsel_pushdown` ms | `duckdb_default` ms | `clickhouse_default` ms |"
        )
        out.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for q in all_cases:
            vinfo = (header.get("verified") or {}).get(q, {})
            rows = vinfo.get("rows", 0)
            m_status = "Admitted" if vinfo.get("morsel_admitted") else "Rejected"
            sd = medians.get((q, "stock_default"))
            sp = medians.get((q, "stock_pushdown"))
            mp = medians.get((q, "morsel_pushdown"))
            dk = medians.get((q, "duckdb_default"))
            ch = medians.get((q, "clickhouse_default"))
            ch_cell = (
                f"{ch:.1f}"
                if ch is not None
                else (
                    "incompatible"
                    if q in (header.get("clickhouse_incompatible") or {})
                    else "—"
                )
            )
            out.append(
                f"| `{q}` | {rows:,} | {m_status} | "
                f"{f'{sd:.1f}' if sd is not None else '—'} | "
                f"{f'{sp:.1f}' if sp is not None else '—'} | "
                f"{f'{mp:.1f}' if mp is not None else '—'} | "
                f"{f'{dk:.1f}' if dk is not None else '—'} | "
                f"{ch_cell} |"
            )

        shared_sd_dk = [
            q
            for q in all_cases
            if (q, "stock_default") in medians and (q, "duckdb_default") in medians
        ]
        if shared_sd_dk:
            sd_gm = statistics.geometric_mean(
                medians[(q, "stock_default")] for q in shared_sd_dk
            )
            dk_gm = statistics.geometric_mean(
                medians[(q, "duckdb_default")] for q in shared_sd_dk
            )
            sp_vals = [
                medians[(q, "stock_pushdown")]
                for q in shared_sd_dk
                if (q, "stock_pushdown") in medians
            ]
            sp_str = (
                f", `stock_pushdown`={statistics.geometric_mean(sp_vals):.1f} ms"
                if len(sp_vals) == len(shared_sd_dk)
                else ""
            )
            out.append("")
            out.append(
                f"Full-suite GeoMean ({len(shared_sd_dk)} queries): "
                f"`stock_default`={sd_gm:.1f} ms{sp_str}, `duckdb_default`={dk_gm:.1f} ms "
                f"(Stock/DuckDB={sd_gm / dk_gm:.2f}x)"
            )
        shared_ch = [
            q
            for q in all_cases
            if (q, "stock_default") in medians
            and (q, "duckdb_default") in medians
            and (q, "clickhouse_default") in medians
        ]
        if shared_ch:
            sd_gm = statistics.geometric_mean(
                medians[(q, "stock_default")] for q in shared_ch
            )
            dk_gm = statistics.geometric_mean(
                medians[(q, "duckdb_default")] for q in shared_ch
            )
            ch_gm = statistics.geometric_mean(
                medians[(q, "clickhouse_default")] for q in shared_ch
            )
            out.append(
                f"ClickHouse-compatible GeoMean ({len(shared_ch)} queries): "
                f"`stock_default`={sd_gm:.1f} ms, `duckdb_default`={dk_gm:.1f} ms, "
                f"`clickhouse_default`={ch_gm:.1f} ms"
            )

    ch_incompat = header.get("clickhouse_incompatible") or {}
    if ch_incompat:
        out.append("")
        out.append(
            f"### ClickHouse Dialect Incompatibilities ({len(ch_incompat)} queries)"
        )
        out.append("")
        out.append("| Query | ClickHouse Error |")
        out.append("| --- | --- |")
        for q, err in sorted(ch_incompat.items()):
            safe_err = err.replace("|", "\\|").replace("\n", " ")[:180]
            out.append(f"| `{q}` | `{safe_err}` |")

    ch_notes = header.get("clickhouse_parity_notes") or {}
    if ch_notes:
        out.append("")
        out.append(
            f"### ClickHouse Semantic Parity Notes ({len(ch_notes)} executable queries)"
        )
        out.append("")
        out.append("| Query | Difference vs DuckDB Oracle |")
        out.append("| --- | --- |")
        for q, note in sorted(ch_notes.items()):
            safe_note = note.replace("|", "\\|").replace("\n", " ")[:180]
            out.append(f"| `{q}` | `{safe_note}` |")

    df_div_notes = header.get("datafusion_int_div_notes") or {}
    if df_div_notes:
        out.append("")
        out.append(
            f"### DataFusion vs DuckDB Integer Division Notes ({len(df_div_notes)} queries)"
        )
        out.append("")
        for q, note in sorted(df_div_notes.items()):
            out.append(f"- `{q}`: {note}")

    return "\n".join(out)


def positive_int(value: str) -> int:
    num = int(value)
    if num < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return num


def positive_float(value: str) -> float:
    num = float(value)
    if num <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return num


def main():
    parser = argparse.ArgumentParser(
        description="TPC-DS 99-query 4-way Parquet benchmark suite (Morsel vs Stock vs DuckDB vs ClickHouse)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate", help="Generate deterministic TPC-DS Parquet tables"
    )
    gen.add_argument("path", type=Path)
    gen.add_argument(
        "--sf", type=positive_float, default=1.0, help="TPC-DS scale factor"
    )
    gen.add_argument(
        "--row-group-size",
        type=positive_int,
        default=DEFAULT_ROW_GROUP_SIZE,
        help="Parquet row group size",
    )

    info = sub.add_parser("info", help="Summarize an existing TPC-DS Parquet dataset")
    info.add_argument("path", type=Path)

    aud = sub.add_parser(
        "audit",
        help="Audit Morsel 99-query admission, parity, and grouped rejection breakdown",
    )
    aud.add_argument("path", type=Path)
    aud.add_argument("--runner", type=Path, required=True)
    aud.add_argument("--threads", type=positive_int, default=8)
    aud.add_argument("--timeout", type=positive_int, default=120)
    aud.add_argument("--queries", nargs="+")
    aud.add_argument(
        "--no-verify-parity",
        action="store_true",
        help="Skip DuckDB/Stock parity checks on admitted queries",
    )
    aud.add_argument("--format", choices=("json", "markdown"), default="json")

    ver = sub.add_parser(
        "verify", help="Verify 4-way parity and Morsel plans/rejections"
    )
    ver.add_argument("path", type=Path)
    ver.add_argument("--runner", type=Path, required=True)
    ver.add_argument("--clickhouse-bin", type=Path)
    ver.add_argument("--threads", type=positive_int, default=4)
    ver.add_argument("--timeout", type=positive_int, default=120)
    ver.add_argument("--queries", nargs="+")
    ver.add_argument("--arms", nargs="+")

    bench = sub.add_parser(
        "bench", help="Run controlled 4-way TPC-DS Parquet benchmark"
    )
    bench.add_argument("path", type=Path)
    bench.add_argument("--runner", type=Path, required=True)
    bench.add_argument("--revision", required=True)
    bench.add_argument("--clickhouse-bin", type=Path)
    bench.add_argument("--threads", type=positive_int, default=8)
    bench.add_argument("--timeout", type=positive_int, default=120)
    bench.add_argument(
        "--cache",
        choices=("warm", "cold", "none"),
        default="warm" if platform.system() == "Linux" else "none",
    )
    bench.add_argument("--warmups", type=int, default=1)
    bench.add_argument("--repeats", type=positive_int, default=5)
    bench.add_argument("--queries", nargs="+")
    bench.add_argument("--arms", nargs="+")

    summ = sub.add_parser(
        "summarize", help="Print Markdown summary of a benchmark JSONL"
    )
    summ.add_argument("jsonl", type=Path)

    dk = sub.add_parser("_duckdb-exec")
    dk.add_argument("operation", choices=("query", "measure"))
    dk.add_argument("threads", type=positive_int)
    dk.add_argument("data_dir", type=Path)
    dk.add_argument("tables_csv")
    dk.add_argument("sql_json_path", type=Path)
    dk.add_argument("--stmt-index", type=int, default=0)

    args = parser.parse_args()
    if args.command == "generate":
        summary = generate(args.path, args.sf, args.row_group_size)
        print(
            json.dumps(
                {
                    "sf": args.sf,
                    "row_group_size": args.row_group_size,
                    "duckdb_version": duckdb.__version__,
                    "pyarrow_version": pa.__version__,
                    "tables": summary,
                },
                indent=2,
            )
        )
        return
    if args.command == "info":
        print(json.dumps(dataset_summary(args.path), indent=2))
        return
    if args.command == "audit":
        try:
            query_names = resolve_queries(args.queries, STANDARD_TPCDS_QUERIES)
        except ValueError as err:
            parser.error(str(err))
        report = audit_suite(
            args.path,
            args.runner,
            args.threads,
            args.timeout,
            query_names,
            verify_parity=not args.no_verify_parity,
        )
        if args.format == "markdown":
            print(format_audit_markdown(report))
        else:
            print(json.dumps(report, indent=2))
        return
    if args.command == "verify":
        try:
            query_names = resolve_queries(args.queries, STANDARD_TPCDS_QUERIES)
            arms = resolve_arms(args.arms, DEFAULT_4WAY_ARMS)
        except ValueError as err:
            parser.error(str(err))
        ch_bin = (
            find_clickhouse_binary(args.clickhouse_bin)
            if any(ARMS[a].engine == "clickhouse" for a in arms)
            else None
        )
        report = verify_suite(
            args.path,
            args.runner,
            args.threads,
            args.timeout,
            query_names,
            arms=arms,
            clickhouse_bin=ch_bin,
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "bench":
        if args.warmups < 0:
            parser.error("--warmups must be non-negative")
        try:
            resolve_queries(args.queries, STANDARD_TPCDS_QUERIES)
            resolve_arms(args.arms, DEFAULT_4WAY_ARMS)
        except ValueError as err:
            parser.error(str(err))
        run_bench(args)
        return
    if args.command == "summarize":
        print(summarize_jsonl(args.jsonl))
        return
    if args.command == "_duckdb-exec":
        tables = [t.strip() for t in args.tables_csv.split(",") if t.strip()]
        _duckdb_exec(
            args.operation,
            args.threads,
            args.data_dir,
            tables,
            args.sql_json_path,
            args.stmt_index,
        )
        return


if __name__ == "__main__":
    main()
