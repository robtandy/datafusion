# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Profiling harness for TPC-H SF1 q6_count and q6_scan vs DuckDB (#180)."""

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

DEFAULT_BATCH_SIZE = 8192


@dataclass(frozen=True)
class Workload:
    name: str
    tables: tuple[str, ...]
    sql: str
    ordered: bool
    is_count: bool


WORKLOADS: dict[str, Workload] = {
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
        is_count=True,
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
        is_count=False,
    ),
}

PROBES: dict[str, Workload] = {
    "p0_count_star": Workload(
        name="p0_count_star",
        tables=("lineitem",),
        sql="SELECT COUNT(*) AS n FROM lineitem",
        ordered=True,
        is_count=True,
    ),
    "p1_shipdate_only": Workload(
        name="p1_shipdate_only",
        tables=("lineitem",),
        sql=(
            "SELECT COUNT(*) AS n FROM lineitem "
            "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'"
        ),
        ordered=True,
        is_count=True,
    ),
    "p2_discount_only": Workload(
        name="p2_discount_only",
        tables=("lineitem",),
        sql=(
            "SELECT COUNT(*) AS n FROM lineitem "
            "WHERE l_discount BETWEEN 0.05 AND 0.07"
        ),
        ordered=True,
        is_count=True,
    ),
    "p3_quantity_only": Workload(
        name="p3_quantity_only",
        tables=("lineitem",),
        sql="SELECT COUNT(*) AS n FROM lineitem WHERE l_quantity < 24",
        ordered=True,
        is_count=True,
    ),
    "p4_shipdate_discount": Workload(
        name="p4_shipdate_discount",
        tables=("lineitem",),
        sql=(
            "SELECT COUNT(*) AS n FROM lineitem "
            "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
            "AND l_discount BETWEEN 0.05 AND 0.07"
        ),
        ordered=True,
        is_count=True,
    ),
    "p5_q6_count": WORKLOADS["q6_count"],
    "p6_q6_scan_no_mul": Workload(
        name="p6_q6_scan_no_mul",
        tables=("lineitem",),
        sql=(
            "SELECT l_orderkey, l_extendedprice, l_discount FROM lineitem "
            "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
            "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"
        ),
        ordered=False,
        is_count=False,
    ),
    "p7_q6_scan": WORKLOADS["q6_scan"],
    "p8_orderkey_only": Workload(
        name="p8_orderkey_only",
        tables=("lineitem",),
        sql="SELECT COUNT(*) AS n FROM lineitem WHERE l_orderkey > 0",
        ordered=True,
        is_count=True,
    ),
    "p9_extprice_only": Workload(
        name="p9_extprice_only",
        tables=("lineitem",),
        sql="SELECT COUNT(*) AS n FROM lineitem WHERE l_extendedprice > 0",
        ordered=True,
        is_count=True,
    ),
    "p10_qty_first_order": Workload(
        name="p10_qty_first_order",
        tables=("lineitem",),
        sql=(
            "SELECT COUNT(*) AS n FROM lineitem "
            "WHERE l_quantity < 24 AND l_discount BETWEEN 0.05 AND 0.07 "
            "AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'"
        ),
        ordered=True,
        is_count=True,
    ),
}


@dataclass(frozen=True)
class Arm:
    name: str
    engine: str  # "datafusion" or "duckdb"
    mode: str | None
    profile: str
    label: str


ARMS: dict[str, Arm] = {
    "stock_no_pushdown": Arm(
        name="stock_no_pushdown",
        engine="datafusion",
        mode="off",
        profile="no_pushdown",
        label="Stock DataFusion (off, pushdown=false, pruning=false)",
    ),
    "stock_default": Arm(
        name="stock_default",
        engine="datafusion",
        mode="off",
        profile="default",
        label="Stock DataFusion (off, defaults)",
    ),
    "stock_pushdown": Arm(
        name="stock_pushdown",
        engine="datafusion",
        mode="off",
        profile="pushdown",
        label="Stock DataFusion (off, pushdown=true, reorder=true)",
    ),
    "stock_pushdown_norep": Arm(
        name="stock_pushdown_norep",
        engine="datafusion",
        mode="off",
        profile="pushdown_no_repartition",
        label="Stock DataFusion (off, pushdown=true, repartition=false)",
    ),
    "morsel_no_pushdown": Arm(
        name="morsel_no_pushdown",
        engine="datafusion",
        mode="required",
        profile="no_pushdown",
        label="Morsel DataFusion (required, pushdown=false, pruning=false)",
    ),
    "morsel_pruning": Arm(
        name="morsel_pruning",
        engine="datafusion",
        mode="required",
        profile="pruning",
        label="Morsel DataFusion (required, pushdown=false, pruning=true)",
    ),
    "morsel_pushdown": Arm(
        name="morsel_pushdown",
        engine="datafusion",
        mode="required",
        profile="pushdown",
        label="Morsel DataFusion (required, pushdown=true, reorder=true)",
    ),
    "morsel_pushdown_noreorder": Arm(
        name="morsel_pushdown_noreorder",
        engine="datafusion",
        mode="required",
        profile="pushdown_no_reorder",
        label="Morsel DataFusion (required, pushdown=true, reorder=false)",
    ),
    "duckdb_default": Arm(
        name="duckdb_default",
        engine="duckdb",
        mode=None,
        profile="default",
        label="DuckDB 2.0.1 (defaults)",
    ),
}

BASELINE_ARMS = (
    "stock_no_pushdown",
    "morsel_no_pushdown",
    "morsel_pruning",
    "morsel_pushdown",
    "morsel_pushdown_noreorder",
    "stock_default",
    "stock_pushdown",
    "duckdb_default",
)


def table_files(data_dir: Path, table: str) -> list[Path]:
    table_dir = data_dir / table
    files = sorted(table_dir.glob("part-*.parquet"))
    if not files:
        raise ValueError(f"no part-*.parquet files found in {table_dir}")
    return files


def cache_pages(files: list[Path]) -> dict[str, int] | None:
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


def _values_equal(expected, actual) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, Decimal) and isinstance(actual, Decimal):
        return expected == actual
    if isinstance(expected, Decimal) and isinstance(actual, int):
        return expected == Decimal(actual)
    if isinstance(expected, int) and isinstance(actual, Decimal):
        return Decimal(expected) == actual
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
            f"{arm_name}/{case_name}: column mismatch {oracle_names} vs {actual_names}"
        )
    if oracle.num_rows != actual.num_rows:
        raise ValueError(
            f"{arm_name}/{case_name}: row count mismatch {oracle.num_rows} vs {actual.num_rows}"
        )
    if oracle.num_rows == 0:
        return
    oracle_rows = list(zip(*(col.to_pylist() for col in oracle.columns)))
    actual_rows = list(zip(*(col.to_pylist() for col in actual.columns)))
    if not ordered:
        oracle_rows.sort(key=_sort_key)
        actual_rows.sort(key=_sort_key)
    for row_idx, (exp_row, act_row) in enumerate(zip(oracle_rows, actual_rows)):
        for col_idx, (exp_val, act_val) in enumerate(zip(exp_row, act_row)):
            if not _values_equal(exp_val, act_val):
                raise ValueError(
                    f"{arm_name}/{case_name}: row {row_idx} col {oracle_names[col_idx]}: "
                    f"{exp_val!r} != {act_val!r}"
                )


def summarize_morsel_runs(morsel_runs: list[list[dict]]) -> dict | None:
    """Aggregate #138 per-run metrics from runner JSON into compact summary."""
    if not morsel_runs:
        return None
    run = morsel_runs[-1]
    sums: dict[str, int] = {}
    per_worker: dict[str, dict[int, int]] = {}
    for entry in run:
        name = entry["name"]
        val = int(entry["value"])
        worker = entry.get("worker")
        sums[name] = sums.get(name, 0) + val
        if worker is not None:
            per_worker.setdefault(name, {})[int(worker)] = val
    ns_to_ms = lambda name: round(sums.get(name, 0) / 1e6, 3)
    workers_claimed = [
        v for _, v in sorted(per_worker.get("morsels_claimed", {}).items())
    ]
    workers_completed = [
        v for _, v in sorted(per_worker.get("morsels_completed", {}).items())
    ]
    return {
        "pending_io_ms": ns_to_ms("morsel_pending_io"),
        "ready_wait_ms": ns_to_ms("morsel_ready_wait"),
        "decode_ms": ns_to_ms("morsel_decode_work"),
        "process_ms": ns_to_ms("morsel_process_work"),
        "output_wait_ms": ns_to_ms("morsel_output_wait"),
        "morsels_claimed": sums.get("morsels_claimed", 0),
        "morsels_completed": sums.get("morsels_completed", 0),
        "per_worker_claimed": workers_claimed,
        "per_worker_completed": workers_completed,
        "peak_ready_depth": sums.get("morsel_peak_ready_depth", 0),
        "peak_ready_bytes": sums.get("morsel_peak_ready_compressed_bytes_estimate", 0),
        "peak_output_depth": sums.get("morsel_peak_output_depth", 0),
        "peak_output_bytes": sums.get("morsel_peak_output_arrow_bytes", 0),
    }


def run_df(
    runner: Path,
    operation: str,
    arm: Arm,
    workload: Workload,
    data_dir: Path,
    threads: int,
    batch_size: int,
    metrics: bool,
    timeout: int,
    warmups: int | None = None,
    repeats: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> bytes:
    cmd = [
        str(runner),
        operation,
        arm.mode,
        arm.profile,
        str(threads),
        str(batch_size),
        "on" if metrics else "off",
        str(data_dir),
        ",".join(workload.tables),
        workload.sql,
    ]
    if warmups is not None:
        cmd.append(str(warmups))
    if repeats is not None:
        cmd.append(str(repeats))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"profile_runner {operation} ({arm.name}) failed on {workload.name}: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout


def _duckdb_exec_fresh(
    operation: str,
    threads: int,
    batch_size: int,
    data_dir: Path,
    tables: list[str],
    sql: str,
):
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads = {threads}")
    con.execute("LOAD parquet")
    for _ in con.sql("SELECT 1").to_arrow_reader(batch_size=batch_size):
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
        reader = con.sql(sql).to_arrow_reader(batch_size=batch_size)
        output_rows = 0
        output_batches = 0
        first_output_seconds = None
        first_8192_seconds = None
        for batch in reader:
            if batch.num_rows:
                before = output_rows
                output_rows += batch.num_rows
                output_batches += 1
                reached_8192 = before < 8192 and output_rows >= 8192
                if output_batches == 1 or reached_8192:
                    elapsed = time.perf_counter() - start
                    if output_batches == 1:
                        first_output_seconds = elapsed
                    if reached_8192:
                        first_8192_seconds = elapsed
        seconds = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_start
        print(
            json.dumps(
                {
                    "duckdb_version": duckdb.__version__,
                    "output_rows": output_rows,
                    "output_batches": output_batches,
                    "seconds": seconds,
                    "cpu_seconds": cpu_seconds,
                    "first_output_seconds": first_output_seconds,
                    "first_8192_seconds": first_8192_seconds,
                }
            )
        )
        return
    raise ValueError(f"unknown _duckdb-exec operation: {operation}")


def _duckdb_exec_warm(
    threads: int,
    batch_size: int,
    data_dir: Path,
    tables: list[str],
    sql: str,
    warmups: int,
    repeats: int,
    drain_mode: str,
    profile_operators: bool,
):
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads = {threads}")
    con.execute("LOAD parquet")
    for _ in con.sql("SELECT 1").to_arrow_reader(batch_size=batch_size):
        pass
    view_start = time.perf_counter()
    for table in tables:
        glob = (data_dir / table / "part-*.parquet").as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{glob}')")
    view_seconds = time.perf_counter() - view_start

    def run_once():
        t0 = time.perf_counter()
        c0 = time.process_time()
        output_rows = 0
        output_batches = 0
        if drain_mode == "arrow":
            reader = con.sql(sql).to_arrow_reader(batch_size=batch_size)
            for batch in reader:
                if batch.num_rows:
                    output_rows += batch.num_rows
                    output_batches += 1
        elif drain_mode == "native":
            rows = con.execute(sql).fetchall()
            output_rows = len(rows)
            output_batches = 1
        elif drain_mode == "execute_only":
            # Drain inside C++ without Python tuple materialization or Arrow export.
            res = con.execute(
                f"SELECT COUNT(*) FROM ({sql}) _t"
                if "COUNT(*)" not in sql.upper()
                else sql
            ).fetchall()
            output_rows = int(res[0][0])
            output_batches = 1
        else:
            raise ValueError(f"unknown drain_mode: {drain_mode}")
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        return {
            "exec_seconds": wall,
            "exec_cpu_seconds": cpu,
            "output_rows": output_rows,
            "output_batches": output_batches,
        }

    first_run = run_once()
    for _ in range(max(0, warmups - 1)):
        run_once()
    iterations = [run_once() for _ in range(repeats)]

    operator_profile = None
    if profile_operators:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            prof_path = tf.name
        try:
            con.execute("PRAGMA enable_profiling = 'json'")
            con.execute(f"PRAGMA profiling_output = '{prof_path}'")
            con.execute("PRAGMA profiling_mode = 'detailed'")
            # Run once with profiling enabled
            reader = con.sql(sql).to_arrow_reader(batch_size=batch_size)
            for _ in reader:
                pass
            operator_profile = json.loads(Path(prof_path).read_text())
        finally:
            Path(prof_path).unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "duckdb_version": duckdb.__version__,
                "drain_mode": drain_mode,
                "view_seconds": view_seconds,
                "first_run": first_run,
                "iterations": iterations,
                "operator_profile": operator_profile,
            }
        )
    )


def run_duckdb_subprocess(
    subcmd: str,
    workload: Workload,
    data_dir: Path,
    threads: int,
    batch_size: int,
    timeout: int,
    extra_args: list[str] | None = None,
) -> bytes:
    cmd = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        subcmd,
        str(threads),
        str(batch_size),
        str(data_dir),
        ",".join(workload.tables),
        workload.sql,
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"duckdb {subcmd} failed on {workload.name}: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout


def validate_plan(sample: dict, arm: Arm, workload: Workload, expected_rgs: int):
    if arm.engine != "datafusion":
        return
    plan = sample["plan"]
    if arm.mode == "off":
        if "MorselExec:" in plan or not re.search(
            r"DataSourceExec: .*file_type=parquet", plan
        ):
            raise ValueError(f"{arm.name}/{workload.name}: bad stock plan:\n{plan}")
        return
    table = workload.tables[0]
    if not plan.startswith("MorselExec:") or not re.search(
        rf"NativeParquet\[{re.escape(table)}(?:\]|,)", plan
    ):
        raise ValueError(f"{arm.name}/{workload.name}: bad morsel plan:\n{plan}")
    claimed = sample["parquet_morsels_claimed"]
    if claimed <= 0 or claimed > expected_rgs:
        raise ValueError(
            f"{arm.name}/{workload.name}: expected 1..={expected_rgs} RGs, got {claimed}"
        )
    if ("CountSink" in plan) != workload.is_count or sample[
        "count_finalizations"
    ] != int(workload.is_count):
        raise ValueError(
            f"{arm.name}/{workload.name}: CountSink/finalizations mismatch"
        )


def cmd_baseline(args):
    data_dir = Path(args.data_dir).resolve(strict=True)
    runner = Path(args.runner).resolve(strict=True)
    files = table_files(data_dir, "lineitem")
    expected_rgs = sum(pq.read_metadata(f).num_row_groups for f in files)
    warm_files(files)
    require_warm_cache(files)

    arm_names = args.arms.split(",") if args.arms else list(BASELINE_ARMS)
    query_names = args.queries.split(",") if args.queries else ["q6_count", "q6_scan"]
    metrics = args.metrics

    # Verify exact parity against DuckDB oracle first.
    oracles: dict[str, pa.Table] = {}
    for qname in query_names:
        wl = WORKLOADS[qname]
        raw = run_duckdb_subprocess(
            "_duckdb-fresh",
            wl,
            data_dir,
            args.threads,
            args.batch_size,
            args.timeout,
            ["query"],
        )
        oracle = pa_ipc.open_stream(raw).read_all()
        oracles[qname] = oracle
        for arm_name in arm_names:
            arm = ARMS[arm_name]
            if arm.engine == "duckdb":
                continue
            df_raw = run_df(
                runner,
                "query",
                arm,
                wl,
                data_dir,
                args.threads,
                args.batch_size,
                False,
                args.timeout,
            )
            actual = pa_ipc.open_stream(df_raw).read_all()
            compare_tables(oracle, actual, wl.ordered, qname, arm_name)
        print(
            json.dumps(
                {
                    "kind": "verify",
                    "case": qname,
                    "oracle_rows": oracle.num_rows,
                    "arms": arm_names,
                }
            ),
            flush=True,
        )

    for qname in query_names:
        wl = WORKLOADS[qname]
        expected_rows = oracles[qname].num_rows
        total_rounds = args.warmups + args.repeats
        for rnd in range(total_rounds):
            measured = rnd >= args.warmups
            order = list(arm_names) if rnd % 2 == 0 else list(reversed(arm_names))
            for arm_name in order:
                arm = ARMS[arm_name]
                before_pages = require_warm_cache(files)
                if arm.engine == "duckdb":
                    sample = json.loads(
                        run_duckdb_subprocess(
                            "_duckdb-fresh",
                            wl,
                            data_dir,
                            args.threads,
                            args.batch_size,
                            args.timeout,
                            ["measure"],
                        )
                    )
                else:
                    sample = json.loads(
                        run_df(
                            runner,
                            "measure",
                            arm,
                            wl,
                            data_dir,
                            args.threads,
                            args.batch_size,
                            metrics,
                            args.timeout,
                        )
                    )
                    validate_plan(sample, arm, wl, expected_rgs)
                after_pages = require_warm_cache(files)
                if sample["output_rows"] != expected_rows:
                    raise ValueError(
                        f"{arm_name}/{qname}: row count {sample['output_rows']} != {expected_rows}"
                    )
                if not measured:
                    continue
                record = {
                    "kind": "baseline",
                    "case": qname,
                    "arm": arm_name,
                    "round": rnd - args.warmups + 1,
                    "batch_size": args.batch_size,
                    "metrics_enabled": metrics,
                    "pages_before": before_pages,
                    "pages_after": after_pages,
                    "seconds": sample["seconds"],
                    "cpu_seconds": sample.get("cpu_seconds"),
                    "register_seconds": sample.get("register_seconds"),
                    "plan_seconds": sample.get("plan_seconds"),
                    "exec_seconds": sample.get("exec_seconds"),
                    "exec_cpu_seconds": sample.get("exec_cpu_seconds"),
                    "first_output_seconds": sample.get("first_output_seconds"),
                    "first_8192_seconds": sample.get("first_8192_seconds"),
                    "output_rows": sample["output_rows"],
                    "output_batches": sample["output_batches"],
                    "bytes_scanned": sample.get("bytes_scanned"),
                    "parquet_morsels_claimed": sample.get("parquet_morsels_claimed"),
                    "plan_metrics": sample.get("plan_metrics"),
                    "morsel_summary": summarize_morsel_runs(
                        sample.get("morsel_runs", [])
                    ),
                }
                if rnd == args.warmups and "plan" in sample:
                    record["plan"] = sample["plan"]
                print(json.dumps(record), flush=True)


def cmd_warm(args):
    data_dir = Path(args.data_dir).resolve(strict=True)
    runner = Path(args.runner).resolve(strict=True)
    files = table_files(data_dir, "lineitem")
    warm_files(files)
    require_warm_cache(files)

    arm_names = args.arms.split(",") if args.arms else list(BASELINE_ARMS)
    query_names = args.queries.split(",") if args.queries else ["q6_count", "q6_scan"]

    for qname in query_names:
        wl = WORKLOADS[qname]
        for arm_name in arm_names:
            arm = ARMS[arm_name]
            before_pages = require_warm_cache(files)
            if arm.engine == "duckdb":
                for drain_mode in ("arrow", "native", "execute_only"):
                    res = json.loads(
                        run_duckdb_subprocess(
                            "_duckdb-warm",
                            wl,
                            data_dir,
                            args.threads,
                            args.batch_size,
                            args.timeout,
                            [
                                str(args.warmups),
                                str(args.repeats),
                                drain_mode,
                                "1" if drain_mode == "arrow" else "0",
                            ],
                        )
                    )
                    require_warm_cache(files)
                    print(
                        json.dumps(
                            {
                                "kind": "warm_duckdb",
                                "case": qname,
                                "arm": f"duckdb_{drain_mode}",
                                "batch_size": args.batch_size,
                                "pages": before_pages,
                                "result": res,
                            }
                        ),
                        flush=True,
                    )
            else:
                res_plan = json.loads(
                    run_df(
                        runner,
                        "measure_warm",
                        arm,
                        wl,
                        data_dir,
                        args.threads,
                        args.batch_size,
                        args.metrics,
                        args.timeout,
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                )
                res_exec = (
                    json.loads(
                        run_df(
                            runner,
                            "measure_exec_loop",
                            arm,
                            wl,
                            data_dir,
                            args.threads,
                            args.batch_size,
                            args.metrics,
                            args.timeout,
                            warmups=args.warmups,
                            repeats=args.repeats,
                        )
                    )
                    if arm.mode == "required"
                    else None
                )
                require_warm_cache(files)
                # Attach summarized morsel metrics for each iteration
                for it in res_plan["iterations"]:
                    it["morsel_summary"] = summarize_morsel_runs(
                        it.pop("morsel_runs", [])
                    )
                if res_exec is not None:
                    for it in res_exec["iterations"]:
                        it["morsel_summary"] = summarize_morsel_runs(
                            it.pop("morsel_runs", [])
                        )
                print(
                    json.dumps(
                        {
                            "kind": "warm_df",
                            "case": qname,
                            "arm": arm_name,
                            "batch_size": args.batch_size,
                            "metrics_enabled": args.metrics,
                            "pages": before_pages,
                            "plan_and_exec": res_plan,
                            "exec_only": res_exec,
                        }
                    ),
                    flush=True,
                )


def cmd_batch_sweep(args):
    data_dir = Path(args.data_dir).resolve(strict=True)
    runner = Path(args.runner).resolve(strict=True)
    files = table_files(data_dir, "lineitem")
    warm_files(files)
    require_warm_cache(files)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    arm_names = (
        args.arms.split(",")
        if args.arms
        else ["morsel_no_pushdown", "morsel_pushdown", "stock_no_pushdown"]
    )
    query_names = args.queries.split(",") if args.queries else ["q6_count", "q6_scan"]

    for qname in query_names:
        wl = WORKLOADS[qname]
        for bs in batch_sizes:
            for arm_name in arm_names:
                arm = ARMS[arm_name]
                require_warm_cache(files)
                op = "measure_exec_loop" if arm.mode == "required" else "measure_warm"
                res = json.loads(
                    run_df(
                        runner,
                        op,
                        arm,
                        wl,
                        data_dir,
                        args.threads,
                        bs,
                        args.metrics,
                        args.timeout,
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                )
                require_warm_cache(files)
                for it in res["iterations"]:
                    it["morsel_summary"] = summarize_morsel_runs(
                        it.pop("morsel_runs", [])
                    )
                print(
                    json.dumps(
                        {
                            "kind": "batch_sweep",
                            "case": qname,
                            "arm": arm_name,
                            "batch_size": bs,
                            "metrics_enabled": args.metrics,
                            "result": res,
                        }
                    ),
                    flush=True,
                )


def cmd_probes(args):
    data_dir = Path(args.data_dir).resolve(strict=True)
    runner = Path(args.runner).resolve(strict=True)
    files = table_files(data_dir, "lineitem")
    warm_files(files)
    require_warm_cache(files)

    probe_names = args.probes.split(",") if args.probes else list(PROBES.keys())
    arm_names = (
        args.arms.split(",")
        if args.arms
        else [
            "morsel_no_pushdown",
            "morsel_pushdown",
            "morsel_pushdown_noreorder",
            "stock_no_pushdown",
            "duckdb_default",
        ]
    )

    for pname in probe_names:
        wl = PROBES[pname]
        for arm_name in arm_names:
            arm = ARMS[arm_name]
            require_warm_cache(files)
            if arm.engine == "duckdb":
                res = json.loads(
                    run_duckdb_subprocess(
                        "_duckdb-warm",
                        wl,
                        data_dir,
                        args.threads,
                        args.batch_size,
                        args.timeout,
                        [str(args.warmups), str(args.repeats), "arrow", "0"],
                    )
                )
                require_warm_cache(files)
                print(
                    json.dumps(
                        {
                            "kind": "probe_duckdb",
                            "probe": pname,
                            "arm": arm_name,
                            "result": res,
                        }
                    ),
                    flush=True,
                )
            else:
                op = "measure_exec_loop" if arm.mode == "required" else "measure_warm"
                try:
                    res = json.loads(
                        run_df(
                            runner,
                            op,
                            arm,
                            wl,
                            data_dir,
                            args.threads,
                            args.batch_size,
                            args.metrics,
                            args.timeout,
                            warmups=args.warmups,
                            repeats=args.repeats,
                        )
                    )
                except RuntimeError as exc:
                    if "morsel_mode=required:" in str(exc):
                        print(
                            json.dumps(
                                {
                                    "kind": "probe_skipped",
                                    "probe": pname,
                                    "arm": arm_name,
                                    "reason": str(exc),
                                }
                            ),
                            flush=True,
                        )
                        continue
                    raise
                require_warm_cache(files)
                for it in res["iterations"]:
                    it["morsel_summary"] = summarize_morsel_runs(
                        it.pop("morsel_runs", [])
                    )
                print(
                    json.dumps(
                        {
                            "kind": "probe_df",
                            "probe": pname,
                            "arm": arm_name,
                            "result": res,
                        }
                    ),
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Internal DuckDB helpers
    p_dfresh = sub.add_parser("_duckdb-fresh")
    p_dfresh.add_argument("threads", type=int)
    p_dfresh.add_argument("batch_size", type=int)
    p_dfresh.add_argument("data_dir")
    p_dfresh.add_argument("tables_csv")
    p_dfresh.add_argument("sql")
    p_dfresh.add_argument("operation")

    p_dwarm = sub.add_parser("_duckdb-warm")
    p_dwarm.add_argument("threads", type=int)
    p_dwarm.add_argument("batch_size", type=int)
    p_dwarm.add_argument("data_dir")
    p_dwarm.add_argument("tables_csv")
    p_dwarm.add_argument("sql")
    p_dwarm.add_argument("warmups", type=int)
    p_dwarm.add_argument("repeats", type=int)
    p_dwarm.add_argument("drain_mode")
    p_dwarm.add_argument("profile_operators")

    def add_common(p):
        p.add_argument("data_dir")
        p.add_argument("--runner", required=True)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        p.add_argument("--timeout", type=int, default=120)
        p.add_argument("--warmups", type=int, default=1)
        p.add_argument("--repeats", type=int, default=7)
        p.add_argument("--metrics", action="store_true")
        p.add_argument("--arms", default=None)
        p.add_argument("--queries", default=None)

    p_base = sub.add_parser("baseline")
    add_common(p_base)

    p_warm = sub.add_parser("warm")
    add_common(p_warm)

    p_bs = sub.add_parser("batch-sweep")
    add_common(p_bs)
    p_bs.add_argument("--batch-sizes", default="2048,4096,8192,16384")

    p_pr = sub.add_parser("probes")
    add_common(p_pr)
    p_pr.add_argument("--probes", default=None)

    args = parser.parse_args()
    if args.cmd == "_duckdb-fresh":
        tables = [t for t in args.tables_csv.split(",") if t]
        _duckdb_exec_fresh(
            args.operation,
            args.threads,
            args.batch_size,
            Path(args.data_dir),
            tables,
            args.sql,
        )
    elif args.cmd == "_duckdb-warm":
        tables = [t for t in args.tables_csv.split(",") if t]
        _duckdb_exec_warm(
            args.threads,
            args.batch_size,
            Path(args.data_dir),
            tables,
            args.sql,
            args.warmups,
            args.repeats,
            args.drain_mode,
            args.profile_operators == "1",
        )
    elif args.cmd == "baseline":
        cmd_baseline(args)
    elif args.cmd == "warm":
        cmd_warm(args)
    elif args.cmd == "batch-sweep":
        cmd_batch_sweep(args)
    elif args.cmd == "probes":
        cmd_probes(args)


if __name__ == "__main__":
    main()
