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

# Standard TPC-H tables with deterministic primary-key ordering and multi-file splits.
TPCH_TABLES: dict[str, tuple[str, int]] = {
    "lineitem": ("l_orderkey, l_linenumber", 16),
    "orders": ("o_orderkey", 8),
    "partsupp": ("ps_partkey, ps_suppkey", 4),
    "customer": ("c_custkey", 2),
    "part": ("p_partkey", 2),
    "supplier": ("s_suppkey", 1),
    "nation": ("n_nationkey", 1),
    "region": ("r_regionkey", 1),
}


@dataclass(frozen=True)
class Workload:
    name: str
    tables: tuple[str, ...]
    sql: str
    ordered: bool
    morsel_admitted: bool
    is_count: bool
    description: str
    is_aggregate: bool = False


WORKLOADS: dict[str, Workload] = {
    "q6": Workload(
        name="q6",
        tables=("lineitem",),
        sql=(
            "SELECT\n"
            "    SUM(l_extendedprice * l_discount) AS revenue\n"
            "FROM\n"
            "    lineitem\n"
            "WHERE\n"
            "    l_shipdate >= DATE '1994-01-01'\n"
            "    AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR\n"
            "    AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01\n"
            "    AND l_quantity < 24"
        ),
        ordered=True,
        morsel_admitted=True,
        is_count=False,
        is_aggregate=True,
        description="Standard TPC-H Q6 (filtered SUM over lineitem)",
    ),
    "q1_unordered": Workload(
        name="q1_unordered",
        tables=("lineitem",),
        sql=(
            "SELECT\n"
            "    l_returnflag,\n"
            "    l_linestatus,\n"
            "    SUM(l_quantity) AS sum_qty,\n"
            "    SUM(l_extendedprice) AS sum_base_price,\n"
            "    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,\n"
            "    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,\n"
            "    AVG(l_quantity) AS avg_qty,\n"
            "    AVG(l_extendedprice) AS avg_price,\n"
            "    AVG(l_discount) AS avg_disc,\n"
            "    COUNT(*) AS count_order\n"
            "FROM\n"
            "    lineitem\n"
            "WHERE\n"
            "    l_shipdate <= DATE '1998-09-02'\n"
            "GROUP BY\n"
            "    l_returnflag,\n"
            "    l_linestatus"
        ),
        ordered=False,
        morsel_admitted=True,
        is_count=False,
        is_aggregate=True,
        description="Unordered TPC-H Q1 (grouped multi-aggregate without ORDER BY over lineitem)",
    ),
    "q1": Workload(
        name="q1",
        tables=("lineitem",),
        sql=(
            "SELECT\n"
            "    l_returnflag,\n"
            "    l_linestatus,\n"
            "    SUM(l_quantity) AS sum_qty,\n"
            "    SUM(l_extendedprice) AS sum_base_price,\n"
            "    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,\n"
            "    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,\n"
            "    AVG(l_quantity) AS avg_qty,\n"
            "    AVG(l_extendedprice) AS avg_price,\n"
            "    AVG(l_discount) AS avg_disc,\n"
            "    COUNT(*) AS count_order\n"
            "FROM\n"
            "    lineitem\n"
            "WHERE\n"
            "    l_shipdate <= DATE '1998-09-02'\n"
            "GROUP BY\n"
            "    l_returnflag,\n"
            "    l_linestatus\n"
            "ORDER BY\n"
            "    l_returnflag,\n"
            "    l_linestatus"
        ),
        ordered=True,
        morsel_admitted=False,
        is_count=False,
        description="Standard TPC-H Q1 (grouped multi-aggregate + ORDER BY over lineitem)",
    ),
    "q3": Workload(
        name="q3",
        tables=("customer", "orders", "lineitem"),
        sql=(
            "SELECT\n"
            "    l_orderkey,\n"
            "    SUM(l_extendedprice * (1 - l_discount)) AS revenue,\n"
            "    o_orderdate,\n"
            "    o_shippriority\n"
            "FROM\n"
            "    customer,\n"
            "    orders,\n"
            "    lineitem\n"
            "WHERE\n"
            "    c_mktsegment = 'BUILDING'\n"
            "    AND c_custkey = o_custkey\n"
            "    AND l_orderkey = o_orderkey\n"
            "    AND o_orderdate < DATE '1995-03-15'\n"
            "    AND l_shipdate > DATE '1995-03-15'\n"
            "GROUP BY\n"
            "    l_orderkey,\n"
            "    o_orderdate,\n"
            "    o_shippriority\n"
            "ORDER BY\n"
            "    revenue DESC,\n"
            "    o_orderdate\n"
            "LIMIT 10"
        ),
        ordered=True,
        morsel_admitted=False,
        is_count=False,
        description="Standard TPC-H Q3 (3-way join + filter + GROUP BY + ORDER BY LIMIT 10)",
    ),
    "q12": Workload(
        name="q12",
        tables=("lineitem", "orders"),
        sql=(
            "SELECT\n"
            "    l_shipmode,\n"
            "    SUM(CASE\n"
            "        WHEN o_orderpriority = '1-URGENT'\n"
            "            OR o_orderpriority = '2-HIGH'\n"
            "            THEN 1\n"
            "        ELSE 0\n"
            "    END) AS high_line_count,\n"
            "    SUM(CASE\n"
            "        WHEN o_orderpriority <> '1-URGENT'\n"
            "            AND o_orderpriority <> '2-HIGH'\n"
            "            THEN 1\n"
            "        ELSE 0\n"
            "    END) AS low_line_count\n"
            "FROM\n"
            "    lineitem\n"
            "    JOIN orders ON l_orderkey = o_orderkey\n"
            "WHERE\n"
            "    l_shipmode IN ('MAIL', 'SHIP')\n"
            "    AND l_commitdate < l_receiptdate\n"
            "    AND l_shipdate < l_commitdate\n"
            "    AND l_receiptdate >= DATE '1994-01-01'\n"
            "    AND l_receiptdate < DATE '1994-01-01' + INTERVAL '1' YEAR\n"
            "GROUP BY\n"
            "    l_shipmode\n"
            "ORDER BY\n"
            "    l_shipmode"
        ),
        ordered=True,
        morsel_admitted=False,
        is_count=False,
        description="Standard TPC-H Q12 (join + filter + GROUP BY + ORDER BY)",
    ),
    "q6_count": Workload(
        name="q6_count",
        tables=("lineitem",),
        sql=(
            "SELECT\n"
            "    COUNT(*) AS n\n"
            "FROM\n"
            "    lineitem\n"
            "WHERE\n"
            "    l_shipdate >= DATE '1994-01-01'\n"
            "    AND l_shipdate < DATE '1995-01-01'\n"
            "    AND l_discount BETWEEN 0.05 AND 0.07\n"
            "    AND l_quantity < 24"
        ),
        ordered=True,
        morsel_admitted=True,
        is_count=True,
        description="Morsel-admitted filtered COUNT(*) on TPC-H lineitem (Q6 predicate)",
    ),
    "q6_scan": Workload(
        name="q6_scan",
        tables=("lineitem",),
        sql=(
            "SELECT\n"
            "    l_orderkey,\n"
            "    l_extendedprice * l_discount AS rev\n"
            "FROM\n"
            "    lineitem\n"
            "WHERE\n"
            "    l_shipdate >= DATE '1994-01-01'\n"
            "    AND l_shipdate < DATE '1995-01-01'\n"
            "    AND l_discount BETWEEN 0.05 AND 0.07\n"
            "    AND l_quantity < 24"
        ),
        ordered=False,
        morsel_admitted=True,
        is_count=False,
        description="Morsel-admitted unordered scan + filter + projection on TPC-H lineitem",
    ),
}


@dataclass(frozen=True)
class Arm:
    name: str
    engine: str  # "datafusion" or "duckdb"
    mode: str | None  # "off", "required", or None for duckdb
    profile: str  # "no_pushdown", "default", "pruning", "pushdown"
    label: str


ARMS: dict[str, Arm] = {
    "stock_no_pushdown": Arm(
        name="stock_no_pushdown",
        engine="datafusion",
        mode="off",
        profile="no_pushdown",
        label="Stock DataFusion (pushdown_filters=false, pruning=false)",
    ),
    "morsel_no_pushdown": Arm(
        name="morsel_no_pushdown",
        engine="datafusion",
        mode="required",
        profile="no_pushdown",
        label="Morsel DataFusion (required, pushdown_filters=false, pruning=false)",
    ),
    "morsel_pruning": Arm(
        name="morsel_pruning",
        engine="datafusion",
        mode="required",
        profile="pruning",
        label="Morsel DataFusion (required, pushdown_filters=false, pruning=true)",
    ),
    "morsel_pushdown": Arm(
        name="morsel_pushdown",
        engine="datafusion",
        mode="required",
        profile="pushdown",
        label="Morsel DataFusion (required, pushdown_filters=true, reorder_filters=true)",
    ),
    "stock_default": Arm(
        name="stock_default",
        engine="datafusion",
        mode="off",
        profile="default",
        label="Stock DataFusion (default settings; pushdown_filters=false, pruning=true)",
    ),
    "stock_pushdown": Arm(
        name="stock_pushdown",
        engine="datafusion",
        mode="off",
        profile="pushdown",
        label="Stock DataFusion (pushdown_filters=true, reorder_filters=true)",
    ),
    "stock_pushdown_no_repartition": Arm(
        name="stock_pushdown_no_repartition",
        engine="datafusion",
        mode="off",
        profile="pushdown_no_repartition",
        label="Stock DataFusion (pushdown_filters=true, reorder_filters=true, repartition_file_scans=false)",
    ),
    "duckdb_default": Arm(
        name="duckdb_default",
        engine="duckdb",
        mode=None,
        profile="default",
        label="DuckDB (default settings)",
    ),
}

DEFAULT_ROADMAP_ARMS = ("stock_default", "stock_pushdown", "duckdb_default")
DEFAULT_MORSEL_ARMS = (
    "stock_no_pushdown",
    "morsel_no_pushdown",
    "morsel_pruning",
    "stock_default",
    "stock_pushdown",
    "morsel_pushdown",
    "duckdb_default",
)


def generate(path: Path, sf: float, row_group_size: int = DEFAULT_ROW_GROUP_SIZE):
    """Deterministically generate standard TPC-H Parquet tables via DuckDB dbgen."""
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    con = duckdb.connect(":memory:")
    con.execute("SET threads = 1")
    con.execute(f"CALL dbgen(sf={sf})")

    summary = {}
    for table, (pk_cols, nfiles) in TPCH_TABLES.items():
        table_dir = path / table
        table_dir.mkdir()
        arrow_table = con.sql(
            f"SELECT * FROM {table} ORDER BY {pk_cols}"
        ).to_arrow_table()
        total_rows = arrow_table.num_rows
        actual_files = min(nfiles, max(1, total_rows))
        written_files = 0
        written_rgs = 0
        written_bytes = 0
        for idx in range(actual_files):
            lo = (idx * total_rows) // actual_files
            hi = ((idx + 1) * total_rows) // actual_files
            if lo == hi:
                continue
            shard = arrow_table.slice(lo, hi - lo)
            out_file = table_dir / f"part-{idx:06d}.parquet"
            with pq.ParquetWriter(
                out_file,
                shard.schema,
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
        summary[table] = {
            "files": written_files,
            "rows": total_rows,
            "row_groups": written_rgs,
            "parquet_bytes": written_bytes,
        }
    con.close()
    return summary


def table_files(path: Path, table: str) -> list[Path]:
    table_dir = path / table
    if not table_dir.is_dir():
        raise ValueError(f"missing TPC-H table directory: {table_dir}")
    files = sorted(table_dir.iterdir())
    if not files or any(
        not p.is_file() or not p.match("part-*.parquet") for p in files
    ):
        raise ValueError(f"expected {table_dir} to contain only part-*.parquet files")
    return files


def dataset_summary(path: Path) -> dict:
    tables = {}
    all_files = []
    for table in TPCH_TABLES:
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

    def _run(self, operation: str, workload: Workload) -> bytes:
        tables_csv = ",".join(workload.tables)
        result = subprocess.run(
            [
                str(self.binary),
                operation,
                self.mode,
                self.profile,
                str(self.threads),
                str(self.data_dir),
                tables_csv,
                workload.sql,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tpch_runner {operation} ({self.mode}/{self.profile}) failed on "
                f"{workload.name} (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def query(self, workload: Workload) -> pa.Table:
        raw = self._run("query", workload)
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

    def _run(self, operation: str, workload: Workload) -> bytes:
        tables_csv = ",".join(workload.tables)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(self.script),
                "_duckdb-exec",
                operation,
                str(self.threads),
                str(self.data_dir),
                tables_csv,
                workload.sql,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"duckdb {operation} failed on {workload.name} "
                f"(rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def query(self, workload: Workload) -> pa.Table:
        raw = self._run("query", workload)
        return pa_ipc.open_stream(raw).read_all()

    def measure(self, workload: Workload) -> dict:
        return json.loads(self._run("measure", workload))

    def profile(self, workload: Workload) -> dict:
        return json.loads(self._run("profile", workload))


def _sum_duckdb_operator_timings(nodes: list[dict], out: dict[str, float]):
    for node in nodes:
        op = node.get("type") or node.get("name")
        if op and "timing" in node:
            out[op] = out.get(op, 0.0) + float(node["timing"]) * 1000.0
        _sum_duckdb_operator_timings(node.get("children", []), out)


def _duckdb_exec(
    operation: str, threads: int, data_dir: Path, tables: list[str], sql: str
):
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads = {threads}")
    # Load Parquet extension and initialize Arrow bridge before starting the clock so
    # one-time extension loading does not skew query timing vs statically linked DataFusion.
    con.execute("LOAD parquet")
    for _ in con.sql("SELECT 1").to_arrow_reader(batch_size=BATCH_SIZE):
        pass

    for table in tables:
        glob = (data_dir / table / "part-*.parquet").as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{glob}')")

    if operation == "query":
        tbl = con.sql(sql).to_arrow_table()
        with pa_ipc.new_stream(sys.stdout.buffer, tbl.schema) as writer:
            writer.write_table(tbl)
        return

    if operation == "measure":
        start = time.perf_counter()
        cpu_start = time.process_time()
        reader = con.sql(sql).to_arrow_reader(batch_size=BATCH_SIZE)
        output_rows = 0
        output_batches = 0
        first_output_seconds = None
        first_8192_seconds = None
        for batch in reader:
            if batch.num_rows:
                before = output_rows
                output_rows += batch.num_rows
                output_batches += 1
                reached_8192 = before < BATCH_SIZE and output_rows >= BATCH_SIZE
                if output_batches == 1 or reached_8192:
                    elapsed = time.perf_counter() - start
                    if output_batches == 1:
                        first_output_seconds = elapsed
                    if reached_8192:
                        first_8192_seconds = elapsed
        seconds = time.perf_counter() - start
        exec_cpu_seconds = time.process_time() - cpu_start
        plan_rows = con.execute(f"EXPLAIN {sql}").fetchall()
        plan = "\n".join(str(r[1]) for r in plan_rows)
        print(
            json.dumps(
                {
                    "duckdb_version": duckdb.__version__,
                    "output_rows": output_rows,
                    "output_batches": output_batches,
                    "register_seconds": None,
                    "plan_seconds": None,
                    "exec_seconds": None,
                    "exec_cpu_seconds": exec_cpu_seconds,
                    "first_output_seconds": first_output_seconds,
                    "first_8192_seconds": first_8192_seconds,
                    "seconds": seconds,
                    "plan": plan,
                    "parquet_morsels_claimed": None,
                    "count_finalizations": None,
                    "aggregate_finalizations": None,
                    "bytes_scanned": None,
                    "worker_claims": None,
                    "worker_completions": None,
                    "morsel_decode_work_ms": None,
                    "morsel_process_work_ms": None,
                    "morsel_ready_wait_ms": None,
                    "morsel_output_wait_ms": None,
                    "morsel_pending_io_ms": None,
                    "morsel_peak_ready_depth": None,
                    "morsel_peak_output_depth": None,
                    "operator_compute_ms": None,
                }
            )
        )
        return

    if operation == "profile":
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "duckdb_profile.json"
            escaped = profile_path.as_posix().replace("'", "''")
            con.execute("PRAGMA enable_profiling = 'json'")
            con.execute(f"PRAGMA profiling_output = '{escaped}'")
            rows = con.execute(sql).fetchall()
            con.close()
            prof = json.loads(profile_path.read_text())
        operator_timing_ms: dict[str, float] = {}
        _sum_duckdb_operator_timings(prof.get("operator", []), operator_timing_ms)
        total_operator_ms = sum(operator_timing_ms.values())
        print(
            json.dumps(
                {
                    "duckdb_version": duckdb.__version__,
                    "output_rows": len(rows),
                    "bytes_read": prof.get("io", {}).get("total_bytes_read"),
                    "total_operator_ms": total_operator_ms,
                    "operator_timing_ms": dict(sorted(operator_timing_ms.items())),
                }
            )
        )
        return

    raise ValueError(f"unknown _duckdb-exec operation: {operation}")


def _values_equal(expected, actual) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, Decimal) and isinstance(actual, Decimal):
        return expected == actual
    if isinstance(expected, Decimal) and isinstance(actual, int):
        return expected == Decimal(actual)
    if isinstance(expected, int) and isinstance(actual, Decimal):
        return Decimal(expected) == actual
    # DuckDB returns float64 for AVG(DECIMAL(15,2)) while DataFusion returns Decimal128(19,6).
    if isinstance(expected, float) and isinstance(actual, Decimal):
        scale = -actual.as_tuple().exponent
        tol = max(10.0 ** (-scale), 1e-9 * abs(expected))
        return math.isclose(expected, float(actual), rel_tol=1e-9, abs_tol=tol)
    if isinstance(expected, Decimal) and isinstance(actual, float):
        scale = -expected.as_tuple().exponent
        tol = max(10.0 ** (-scale), 1e-9 * abs(actual))
        return math.isclose(float(expected), actual, rel_tol=1e-9, abs_tol=tol)
    if isinstance(expected, float) and isinstance(actual, float):
        return math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)
    return expected == actual


def _sort_key(row: tuple):
    key = []
    for v in row:
        if v is None:
            key.append((0, 0))
        elif isinstance(v, bool):
            key.append((1, int(v)))
        elif isinstance(v, int):
            key.append((1, Decimal(v)))
        elif isinstance(v, float):
            key.append((1, Decimal(str(v)) if not math.isnan(v) else Decimal("NaN")))
        else:
            key.append((1, v))
    return tuple(key)


def compare_tables(
    oracle: pa.Table, actual: pa.Table, ordered: bool, case_name: str, arm_name: str
):
    oracle_names = [n.lower() for n in oracle.schema.names]
    actual_names = [n.lower() for n in actual.schema.names]
    if oracle_names != actual_names:
        raise ValueError(
            f"{arm_name}/{case_name}: column mismatch: expected {oracle_names}, got {actual_names}"
        )
    if oracle.num_rows != actual.num_rows:
        raise ValueError(
            f"{arm_name}/{case_name}: row count mismatch: expected {oracle.num_rows}, got {actual.num_rows}"
        )
    if oracle.num_rows == 0:
        return

    oracle_cols = [col.to_pylist() for col in oracle.columns]
    actual_cols = [col.to_pylist() for col in actual.columns]
    oracle_rows = list(zip(*oracle_cols))
    actual_rows = list(zip(*actual_cols))

    if not ordered:
        # Sort both sides by exact comparable keys for multiset equality.
        oracle_rows.sort(key=_sort_key)
        actual_rows.sort(key=_sort_key)

    for row_idx, (exp_row, act_row) in enumerate(zip(oracle_rows, actual_rows)):
        for col_idx, (exp_val, act_val) in enumerate(zip(exp_row, act_row)):
            if not _values_equal(exp_val, act_val):
                col_name = oracle_names[col_idx]
                raise ValueError(
                    f"{arm_name}/{case_name}: mismatch at row {row_idx}, col '{col_name}': "
                    f"expected {exp_val!r}, got {act_val!r}"
                )


def validate_plan(
    evidence: dict, arm: Arm, workload: Workload, table_row_groups: dict[str, int]
):
    if arm.engine != "datafusion":
        return
    plan = evidence["plan"]
    if arm.mode == "off":
        if "MorselExec:" in plan or not re.search(
            r"DataSourceExec: .*file_type=parquet", plan
        ):
            raise ValueError(
                f"{arm.name}/{workload.name}: expected a stock Parquet plan, got:\n{plan}"
            )
        return

    # mode == "required"
    table = workload.tables[0]
    if not plan.startswith("MorselExec:") or f"NativeParquet[{table}" not in plan:
        raise ValueError(
            f"{arm.name}/{workload.name}: expected MorselExec with NativeParquet[{table}...], got:\n{plan}"
        )
    expected_rgs = table_row_groups[table]
    claimed = evidence["parquet_morsels_claimed"]
    if arm.profile == "no_pushdown":
        if claimed != expected_rgs:
            raise ValueError(
                f"{arm.name}/{workload.name}: expected to claim all {expected_rgs} "
                f"row groups under no_pushdown, got {claimed}"
            )
    else:
        if claimed <= 0 or claimed > expected_rgs:
            raise ValueError(
                f"{arm.name}/{workload.name}: expected 1..={expected_rgs} row groups "
                f"claimed under {arm.profile}, got {claimed}"
            )
    if ("CountSink" in plan) != workload.is_count or evidence[
        "count_finalizations"
    ] != int(workload.is_count):
        raise ValueError(
            f"{arm.name}/{workload.name}: unexpected CountSink or count_finalizations "
            f"(is_count={workload.is_count}, count_finalizations={evidence['count_finalizations']})"
        )
    agg_finalizations = evidence.get("aggregate_finalizations", 0)
    if ("AggregateSink" in plan) != workload.is_aggregate or agg_finalizations != int(
        workload.is_aggregate
    ):
        raise ValueError(
            f"{arm.name}/{workload.name}: unexpected AggregateSink or aggregate_finalizations "
            f"(is_aggregate={workload.is_aggregate}, aggregate_finalizations={agg_finalizations})"
        )


def make_runner(arm: Arm, runner_bin: Path, data_dir: Path, threads: int, timeout: int):
    if arm.engine == "duckdb":
        return DuckDbRunner(data_dir, threads, timeout)
    return DataFusionRunner(
        runner_bin, data_dir, threads, timeout, arm.mode, arm.profile
    )


def verify_suite(
    data_dir: Path,
    runner_bin: Path,
    threads: int,
    timeout: int,
    query_names: list[str],
    morsel_arms: tuple[str, ...] = DEFAULT_MORSEL_ARMS,
    roadmap_arms: tuple[str, ...] = DEFAULT_ROADMAP_ARMS,
) -> dict:
    summary = dataset_summary(data_dir)
    table_rgs = {name: info["row_groups"] for name, info in summary["tables"].items()}
    duckdb_oracle = DuckDbRunner(data_dir, threads, timeout)

    verified = {}
    morsel_rejections = {}
    evidence = []

    for qname in query_names:
        workload = WORKLOADS[qname]
        oracle_table = duckdb_oracle.query(workload)
        verified[qname] = {"rows": oracle_table.num_rows, "arms": []}

        arm_names = morsel_arms if workload.morsel_admitted else roadmap_arms
        for arm_name in arm_names:
            arm = ARMS[arm_name]
            runner = make_runner(arm, runner_bin, data_dir, threads, timeout)
            if arm.engine != "duckdb":
                actual_table = runner.query(workload)
                compare_tables(
                    oracle_table, actual_table, workload.ordered, qname, arm_name
                )
            sample = runner.measure(workload)
            validate_plan(sample, arm, workload, table_rgs)
            if sample["output_rows"] != oracle_table.num_rows:
                raise ValueError(
                    f"{arm_name}/{qname}: measure output_rows {sample['output_rows']} "
                    f"!= verified rows {oracle_table.num_rows}"
                )
            verified[qname]["arms"].append(arm_name)
            evidence.append(
                {
                    "case": qname,
                    "arm": arm_name,
                    "label": arm.label,
                    "engine": arm.engine,
                    "mode": arm.mode,
                    "profile": arm.profile,
                    "output_rows": sample["output_rows"],
                    "parquet_morsels_claimed": sample["parquet_morsels_claimed"],
                    "count_finalizations": sample["count_finalizations"],
                    "aggregate_finalizations": sample["aggregate_finalizations"],
                    "bytes_scanned": sample["bytes_scanned"],
                    "plan": sample["plan"],
                }
            )

        if not workload.morsel_admitted:
            morsel_checker = DataFusionRunner(
                runner_bin, data_dir, threads, timeout, "required", "no_pushdown"
            )
            plan_result = morsel_checker.plan(workload)
            if plan_result["admitted"]:
                raise ValueError(
                    f"{qname}: unexpectedly admitted under morsel_mode=required; "
                    "update WORKLOADS if support has landed"
                )
            morsel_rejections[qname] = plan_result["error"]

    return {
        "verified": verified,
        "morsel_rejections": morsel_rejections,
        "evidence": evidence,
    }


def run_bench(args):
    data_dir = args.path.resolve(strict=True)
    summary = dataset_summary(data_dir)
    table_rgs = {name: info["row_groups"] for name, info in summary["tables"].items()}
    query_names = args.queries or list(WORKLOADS.keys())
    morsel_arms = tuple(args.morsel_arms) if args.morsel_arms else DEFAULT_MORSEL_ARMS
    roadmap_arms = (
        tuple(args.roadmap_arms) if args.roadmap_arms else DEFAULT_ROADMAP_ARMS
    )

    if args.cache == "warm" and platform.system() != "Linux":
        raise ValueError("--cache warm requires Linux mincore page-cache verification")

    verification = verify_suite(
        data_dir,
        args.runner,
        args.threads,
        args.timeout,
        query_names,
        morsel_arms=morsel_arms,
        roadmap_arms=roadmap_arms,
    )

    # Check runner build revision on every DataFusion evidence record.
    df_checker = DataFusionRunner(
        args.runner, data_dir, args.threads, args.timeout, "off", "default"
    )
    rev_check = df_checker.plan(WORKLOADS[query_names[0]])
    if rev_check["build_revision"] != args.revision:
        raise ValueError(
            f"runner revision {rev_check['build_revision']} != --revision {args.revision}"
        )

    header = {
        "record_type": "header",
        "build_revision": args.revision,
        "duckdb_version": duckdb.__version__,
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
        "dataset": summary,
        "verified": verification["verified"],
        "morsel_rejections": verification["morsel_rejections"],
        "evidence": verification["evidence"],
    }
    print(json.dumps(header), flush=True)

    # Pre-warm all referenced table files once if warm regime is requested.
    referenced_tables = sorted({t for q in query_names for t in WORKLOADS[q].tables})
    all_referenced_files = [
        f for t in referenced_tables for f in table_files(data_dir, t)
    ]
    if args.cache == "warm":
        warm_files(all_referenced_files)
        require_warm_cache(all_referenced_files)

    for qname in query_names:
        workload = WORKLOADS[qname]
        expected_rows = verification["verified"][qname]["rows"]
        arm_names = morsel_arms if workload.morsel_admitted else roadmap_arms
        runners = {
            arm_name: make_runner(
                ARMS[arm_name], args.runner, data_dir, args.threads, args.timeout
            )
            for arm_name in arm_names
        }
        q_files = [f for t in workload.tables for f in table_files(data_dir, t)]

        for rep in range(1 - args.warmups, args.repeats + 1):
            order = arm_names if rep % 2 else tuple(reversed(arm_names))
            for arm_name in order:
                arm = ARMS[arm_name]
                before = None
                if rep > 0:
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
                if result["output_rows"] != expected_rows:
                    raise ValueError(
                        f"{arm_name}/{qname}: unexpected output_rows {result['output_rows']}"
                    )
                if rep > 0:
                    after = (
                        require_warm_cache(q_files)
                        if args.cache == "warm"
                        else cache_pages(q_files)
                    )
                    result.pop("plan", None)
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


def summarize_jsonl(path: Path):
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header = lines[0]
    trials = [r for r in lines[1:] if r.get("record_type") == "trial"]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for t in trials:
        grouped.setdefault((t["case"], t["arm"]), []).append(t)

    print(
        f"Host: {header['host']} ({header['arch']}, kernel {header['kernel']}), "
        f"threads={header['threads']}, cache={header['cache_regime']}, "
        f"warmups={header['warmups']}, repeats={header['repeats']}"
    )
    print(
        f"Revision: {header['build_revision']}, DuckDB: {header['duckdb_version']}, "
        f"PyArrow: {header['pyarrow_version']}"
    )
    print()
    print(
        "| Query | Arm | Median total ms (min-max) | Median exec ms (min-max) | "
        "Output rows | Batches | Morsels claimed |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for (case, arm), rows in grouped.items():
        totals = [r["seconds"] * 1000.0 for r in rows]
        med_tot = statistics.median(totals)
        execs = [
            r["exec_seconds"] * 1000.0 for r in rows if r["exec_seconds"] is not None
        ]
        if execs:
            exec_str = (
                f"{statistics.median(execs):.1f} ({min(execs):.1f}-{max(execs):.1f})"
            )
        else:
            exec_str = "—"
        morsels = rows[0].get("parquet_morsels_claimed")
        morsels_str = str(morsels) if morsels is not None else "—"
        print(
            f"| `{case}` | `{arm}` | "
            f"{med_tot:.1f} ({min(totals):.1f}-{max(totals):.1f}) | "
            f"{exec_str} | {rows[0]['output_rows']:,} | "
            f"{rows[0]['output_batches']:,} | {morsels_str} |"
        )

    morsel_groups = [
        ((case, arm), rows)
        for (case, arm), rows in grouped.items()
        if any(r.get("morsel_decode_work_ms") is not None for r in rows)
    ]
    if morsel_groups:
        print()
        print(
            "| Query | Arm | Decode work ms | Process work ms | Ready wait ms | "
            "Output wait ms | Pending IO ms | Peak ready/out |"
        )
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for (case, arm), rows in morsel_groups:
            decode = statistics.median(
                r["morsel_decode_work_ms"]
                for r in rows
                if r.get("morsel_decode_work_ms") is not None
            )
            process = statistics.median(
                r["morsel_process_work_ms"]
                for r in rows
                if r.get("morsel_process_work_ms") is not None
            )
            ready = statistics.median(
                r["morsel_ready_wait_ms"]
                for r in rows
                if r.get("morsel_ready_wait_ms") is not None
            )
            output = statistics.median(
                r["morsel_output_wait_ms"]
                for r in rows
                if r.get("morsel_output_wait_ms") is not None
            )
            pending_io = statistics.median(
                r["morsel_pending_io_ms"]
                for r in rows
                if r.get("morsel_pending_io_ms") is not None
            )
            peak_ready = max(
                r["morsel_peak_ready_depth"]
                for r in rows
                if r.get("morsel_peak_ready_depth") is not None
            )
            peak_out = max(
                r["morsel_peak_output_depth"]
                for r in rows
                if r.get("morsel_peak_output_depth") is not None
            )
            print(
                f"| `{case}` | `{arm}` | {decode:.1f} | {process:.1f} | "
                f"{ready:.1f} | {output:.1f} | {pending_io:.1f} | "
                f"{peak_ready}/{peak_out} |"
            )


def run_duckdb_profile(args):
    data_dir = args.path.resolve(strict=True)
    query_names = args.queries or [
        name for name, w in WORKLOADS.items() if w.morsel_admitted
    ]
    if args.cache == "warm" and platform.system() != "Linux":
        raise ValueError("--cache warm requires Linux mincore page-cache verification")

    referenced_tables = sorted({t for q in query_names for t in WORKLOADS[q].tables})
    all_referenced_files = [
        f for t in referenced_tables for f in table_files(data_dir, t)
    ]
    if args.cache == "warm":
        warm_files(all_referenced_files)
        require_warm_cache(all_referenced_files)

    runner = DuckDbRunner(data_dir, args.threads, args.timeout)
    results = {}
    for qname in query_names:
        workload = WORKLOADS[qname]
        q_files = [f for t in workload.tables for f in table_files(data_dir, t)]
        samples = []
        for rep in range(1 - args.warmups, args.repeats + 1):
            if rep > 0 and args.cache == "warm":
                require_warm_cache(q_files)
            sample = runner.profile(workload)
            if rep > 0:
                if args.cache == "warm":
                    require_warm_cache(q_files)
                samples.append(sample)

        op_names = sorted({op for s in samples for op in s["operator_timing_ms"]})
        op_medians = {
            op: statistics.median(s["operator_timing_ms"].get(op, 0.0) for s in samples)
            for op in op_names
        }
        totals = [s["total_operator_ms"] for s in samples]
        bytes_reads = [s["bytes_read"] for s in samples if s["bytes_read"] is not None]
        results[qname] = {
            "output_rows": samples[0]["output_rows"],
            "bytes_read_median": (
                int(statistics.median(bytes_reads)) if bytes_reads else None
            ),
            "total_operator_ms_median": statistics.median(totals),
            "total_operator_ms_min": min(totals),
            "total_operator_ms_max": max(totals),
            "operator_timing_ms_median": op_medians,
        }

    report = {
        "duckdb_version": duckdb.__version__,
        "host": platform.node(),
        "arch": platform.machine(),
        "kernel": platform.release(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "threads": args.threads,
        "cache_regime": args.cache,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "methodology": {
            "isolation": "fresh_process_and_connection",
            "execution": "con.execute(sql).fetchall()",
            "json_field": "operator[].timing summed in ms grouped by type",
            "aggregation": "median",
        },
        "queries": results,
    }
    if args.format == "markdown":
        print(
            f"DuckDB {report['duckdb_version']} operator profile on {report['host']} "
            f"(threads={args.threads}, cache={args.cache}, fresh process/connection, "
            f"warmups={args.warmups}, repeats={args.repeats}, median of operator[].timing by type)"
        )
        print()
        print(
            "| Query | Output rows | Bytes read | Total op ms (min-max) | Operator breakdown (median ms) |"
        )
        print("| --- | ---: | ---: | ---: | --- |")
        for qname, info in results.items():
            ops_str = ", ".join(
                f"`{op}`={ms:.1f}"
                for op, ms in info["operator_timing_ms_median"].items()
            )
            bytes_str = (
                f"{info['bytes_read_median']:,}"
                if info["bytes_read_median"] is not None
                else "—"
            )
            print(
                f"| `{qname}` | {info['output_rows']:,} | {bytes_str} | "
                f"{info['total_operator_ms_median']:.1f} "
                f"({info['total_operator_ms_min']:.1f}-{info['total_operator_ms_max']:.1f}) | "
                f"{ops_str} |"
            )
        return
    print(json.dumps(report, indent=2))


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
        description="TPC-H Parquet benchmark suite (Stock DataFusion vs DuckDB vs Morsel)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate deterministic TPC-H Parquet tables")
    gen.add_argument("path", type=Path)
    gen.add_argument(
        "--sf", type=positive_float, default=1.0, help="TPC-H scale factor"
    )
    gen.add_argument(
        "--row-group-size",
        type=positive_int,
        default=DEFAULT_ROW_GROUP_SIZE,
        help="Parquet row group size",
    )

    info = sub.add_parser("info", help="Summarize an existing TPC-H Parquet dataset")
    info.add_argument("path", type=Path)

    ver = sub.add_parser("verify", help="Verify exact parity and morsel plans")
    ver.add_argument("path", type=Path)
    ver.add_argument("--runner", type=Path, required=True)
    ver.add_argument("--threads", type=positive_int, default=4)
    ver.add_argument("--timeout", type=positive_int, default=120)
    ver.add_argument("--queries", nargs="+", choices=list(WORKLOADS.keys()))
    ver.add_argument("--morsel-arms", nargs="+", choices=list(ARMS.keys()))
    ver.add_argument("--roadmap-arms", nargs="+", choices=list(ARMS.keys()))

    bench = sub.add_parser("bench", help="Run controlled 3-way TPC-H benchmark")
    bench.add_argument("path", type=Path)
    bench.add_argument("--runner", type=Path, required=True)
    bench.add_argument("--revision", required=True)
    bench.add_argument("--threads", type=positive_int, default=8)
    bench.add_argument("--timeout", type=positive_int, default=120)
    bench.add_argument(
        "--cache",
        choices=("warm", "none"),
        default="warm" if platform.system() == "Linux" else "none",
    )
    bench.add_argument("--warmups", type=int, default=1)
    bench.add_argument("--repeats", type=positive_int, default=5)
    bench.add_argument("--queries", nargs="+", choices=list(WORKLOADS.keys()))
    bench.add_argument("--morsel-arms", nargs="+", choices=list(ARMS.keys()))
    bench.add_argument("--roadmap-arms", nargs="+", choices=list(ARMS.keys()))

    summ = sub.add_parser(
        "summarize", help="Print Markdown summary of a benchmark JSONL"
    )
    summ.add_argument("jsonl", type=Path)

    dprof = sub.add_parser(
        "duckdb-profile",
        help="Collect fresh-process DuckDB JSON operator timing breakdowns",
    )
    dprof.add_argument("path", type=Path)
    dprof.add_argument("--threads", type=positive_int, default=8)
    dprof.add_argument("--timeout", type=positive_int, default=120)
    dprof.add_argument(
        "--cache",
        choices=("warm", "none"),
        default="warm" if platform.system() == "Linux" else "none",
    )
    dprof.add_argument("--warmups", type=int, default=1)
    dprof.add_argument("--repeats", type=positive_int, default=7)
    dprof.add_argument("--queries", nargs="+", choices=list(WORKLOADS.keys()))
    dprof.add_argument("--format", choices=("json", "markdown"), default="json")

    # Internal subprocess entrypoint for isolated DuckDB runs.
    dk = sub.add_parser("_duckdb-exec")
    dk.add_argument("operation", choices=("query", "measure", "profile"))
    dk.add_argument("threads", type=positive_int)
    dk.add_argument("data_dir", type=Path)
    dk.add_argument("tables_csv")
    dk.add_argument("sql")

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
    if args.command == "verify":
        report = verify_suite(
            args.path,
            args.runner,
            args.threads,
            args.timeout,
            args.queries or list(WORKLOADS.keys()),
            morsel_arms=(
                tuple(args.morsel_arms) if args.morsel_arms else DEFAULT_MORSEL_ARMS
            ),
            roadmap_arms=(
                tuple(args.roadmap_arms) if args.roadmap_arms else DEFAULT_ROADMAP_ARMS
            ),
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "bench":
        if args.warmups < 0:
            parser.error("--warmups must be non-negative")
        run_bench(args)
        return
    if args.command == "summarize":
        summarize_jsonl(args.jsonl)
        return
    if args.command == "duckdb-profile":
        if args.warmups < 0:
            parser.error("--warmups must be non-negative")
        run_duckdb_profile(args)
        return
    if args.command == "_duckdb-exec":
        tables = [t.strip() for t in args.tables_csv.split(",") if t.strip()]
        _duckdb_exec(args.operation, args.threads, args.data_dir, tables, args.sql)
        return


if __name__ == "__main__":
    main()
