# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Reproducible quiet-host diagnostic profiler for TPC-H SF5 Q14, Q17, and Q20 (#215).

Captures side-by-side physical plans (`EXPLAIN` + `EXPLAIN ANALYZE VERBOSE`),
warm-cache wall/CPU/`#138` metrics, Arm PMUv3 `perf stat` counters, and
3,999 Hz `perf record` leaf symbol profiles for `stock_pushdown` vs.
`morsel_pushdown` (and optional scratch ablation runners).
"""

import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import pyarrow.ipc as pa_ipc

REPO_ROOT = Path(__file__).resolve().parents[3]
QUERIES_DIR = REPO_ROOT / "benchmarks" / "queries"
PERF_EVENTS = "cycles:u,instructions:u,r17:u,task-clock,context-switches,cpu-migrations"

QUERY_SPECS: dict[str, tuple[str, ...]] = {
    "q14": ("lineitem", "part"),
    "q17": ("lineitem", "part"),
    "q20": ("supplier", "nation", "partsupp", "part", "lineitem"),
}

ARMS: dict[str, tuple[str, str]] = {
    "stock_default": ("off", "default"),
    "stock_pushdown": ("off", "pushdown"),
    "morsel_no_pushdown": ("required", "no_pushdown"),
    "morsel_pushdown": ("required", "pushdown"),
}

_DURATION_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(ns|µs|us|ms|s)$")
_REPORT_LINE_RE = re.compile(r"^\s*([0-9]+\.[0-9]+)%\s+\S+\s+\S+\s+\[.\]\s+(.+?)\s*$")


def load_sql(query_name: str) -> str:
    return (QUERIES_DIR / f"{query_name}.sql").read_text().strip().rstrip(";")


def parquet_files_for(data_dir: Path, tables: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for table in tables:
        files.extend(sorted((data_dir / table).glob("*.parquet")))
    return files


def mincore_resident_ratio(files: list[Path]) -> tuple[int, int]:
    """Return `(resident_pages, total_pages)` across all listed files via Linux `mincore`."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    libc.mincore.restype = ctypes.c_int
    page_size = os.sysconf("SC_PAGE_SIZE")
    resident = 0
    total = 0
    for path in files:
        size = path.stat().st_size
        if size == 0:
            continue
        pages = (size + page_size - 1) // page_size
        total += pages
        with path.open("rb") as fh:
            mm = mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_COPY)
            try:
                addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                vec = ctypes.create_string_buffer(pages)
                rc = libc.mincore(addr, size, vec)
                if rc != 0:
                    err = ctypes.get_errno()
                    raise OSError(err, f"mincore failed for {path}")
                resident += sum(1 for b in vec.raw if b & 1)
            finally:
                mm.close()
    return resident, total


def require_warm_cache(files: list[Path]) -> tuple[int, int]:
    resident, total = mincore_resident_ratio(files)
    if resident != total:
        # Touch files sequentially to populate page cache, then re-verify.
        for path in files:
            with path.open("rb") as fh:
                while fh.read(4 * 1024 * 1024):
                    pass
        resident, total = mincore_resident_ratio(files)
        if resident != total:
            raise RuntimeError(
                f"warm page cache incomplete: {resident}/{total} pages resident"
            )
    return resident, total


def check_host_idle(min_idle_pct: int, max_wait_s: int) -> dict[str, int]:
    """Poll `vmstat -y 1 2` until CPU idle% >= `min_idle_pct` or timeout."""
    deadline = time.monotonic() + max_wait_s
    last: dict[str, int] = {}
    while True:
        proc = subprocess.run(
            ["vmstat", "-y", "1", "2"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        cols = lines[-1].split()
        last = {
            "runnable": int(cols[0]),
            "us": int(cols[12]),
            "sy": int(cols[13]),
            "id": int(cols[14]),
            "wa": int(cols[15]),
        }
        if last["id"] >= min_idle_pct or time.monotonic() >= deadline:
            return last
        time.sleep(2)


def parse_duration_ms(raw: str) -> float:
    m = _DURATION_RE.match(raw.strip())
    if not m:
        raise ValueError(f"cannot parse duration: {raw!r}")
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "ns":
        return val / 1e6
    if unit in ("µs", "us"):
        return val / 1e3
    if unit == "ms":
        return val
    if unit == "s":
        return val * 1e3
    raise ValueError(f"unexpected unit: {unit!r}")


def perf_env(perf_lib_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if perf_lib_dir is not None:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{perf_lib_dir}:{existing}" if existing else str(perf_lib_dir)
        )
    return env


def parse_perf_stat(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw_val, event = parts[0].strip(), parts[2].strip()
        if not raw_val or raw_val.startswith("<"):
            continue
        try:
            out[event] = float(raw_val)
        except ValueError:
            continue
    return out


def categorize_symbol(sym: str) -> str:
    s = sym.lower()
    if any(
        tok in s
        for tok in (
            "zstd_",
            "huf_",
            "fse_",
            "xxh",
            "bit_reload",
            "zstd_decompress",
        )
    ):
        return "zstd_decompress"
    if any(
        tok in s
        for tok in (
            "eval_staged",
            "filter_record_batch",
            "filter_primitive",
            "filter_native",
            "arrow_select::filter",
            "booleanbufferbuilder",
        )
    ):
        return "rowfilter_eval_staged_filter"
    if any(
        tok in s
        for tok in (
            "joinbloomfilterexpr",
            "splitblockbloomfilter",
            "hashtablelookupexpr",
            "partitioned_hash_eval",
            "contain_hashes",
            "sort_rows_by_partition",
        )
    ):
        return "dynamic_filter_lookup"
    if any(
        tok in s
        for tok in (
            "parquet::arrow::arrow_reader",
            "parquet::column",
            "parquet::encodings",
            "parquet::util::bit_util",
            "parquet::file",
            "parquet::data_type",
            "decode_values",
            "read_records",
            "rledecoder",
            "fixedlenbytearray",
            "bytearrayreader",
            "primitiverecordreader",
            "row_filter",
            "predicate_cache",
            "cached_array",
        )
    ):
        return "parquet_decode_and_rf"
    if any(
        tok in s
        for tok in (
            "hash_join",
            "radix_join",
            "radix::",
            "hash_table",
            "hashbrown",
            "ahash",
            "create_hashes",
            "take_arrays",
            "arrow_select::take",
            "concat_batches",
        )
    ):
        return "hash_join_and_take"
    if any(
        tok in s
        for tok in (
            "aggregate",
            "group_values",
            "accumulator",
            "like::",
            "arrow_string",
            "arrow_arith",
            "arrow_ord",
            "arrow_cmp",
        )
    ):
        return "expr_agg_cmp"
    if any(
        tok in s
        for tok in (
            "malloc",
            "free",
            "realloc",
            "calloc",
            "memcpy",
            "memmove",
            "memset",
            "alloc::alloc",
            "allocate_aligned",
        )
    ):
        return "alloc_and_mem"
    if any(
        tok in s
        for tok in (
            "tokio::",
            "parking_lot",
            "futex",
            "pthread",
            "object_store",
            "std::fs",
            "pread",
        )
    ):
        return "runtime_sync_io"
    return "other"


def parse_perf_report(report_text: str) -> tuple[list[dict], dict[str, float]]:
    symbols: list[dict] = []
    buckets: dict[str, float] = {}
    for line in report_text.splitlines():
        m = _REPORT_LINE_RE.match(line)
        if not m:
            continue
        pct = float(m.group(1))
        sym = m.group(2)
        cat = categorize_symbol(sym)
        buckets[cat] = round(buckets.get(cat, 0.0) + pct, 3)
        symbols.append({"pct": pct, "category": cat, "symbol": sym})
    return symbols, dict(sorted(buckets.items(), key=lambda kv: -kv[1]))


def run_measure(
    runner: Path,
    mode: str,
    profile: str,
    threads: int,
    data_dir: Path,
    tables: tuple[str, ...],
    sql: str,
    timeout_s: int,
) -> dict:
    cmd = [
        str(runner),
        "measure",
        mode,
        profile,
        str(threads),
        str(data_dir),
        ",".join(tables),
        sql,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=True,
    )
    return json.loads(proc.stdout)


def cmd_explain(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(args.runner)
    data_dir = Path(args.data_dir)
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    summary: dict[str, dict] = {}
    for q in queries:
        tables = QUERY_SPECS[q]
        sql = load_sql(q)
        files = parquet_files_for(data_dir, tables)
        require_warm_cache(files)
        summary[q] = {}
        for arm in arms:
            mode, profile = ARMS[arm]
            p_plan = subprocess.run(
                [
                    str(runner),
                    "plan",
                    mode,
                    profile,
                    str(args.threads),
                    str(data_dir),
                    ",".join(tables),
                    sql,
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=True,
            )
            plan_obj = json.loads(p_plan.stdout)
            (out_dir / f"{q}_{arm}_plan.txt").write_text(plan_obj["plan"] or "")

            p_eav = subprocess.run(
                [
                    str(runner),
                    "query",
                    mode,
                    profile,
                    str(args.threads),
                    str(data_dir),
                    ",".join(tables),
                    f"EXPLAIN ANALYZE VERBOSE {sql}",
                ],
                capture_output=True,
                timeout=args.timeout,
                check=True,
            )
            tbl = pa_ipc.open_stream(p_eav.stdout).read_all()
            blocks = [
                f"--- {pt} ---\n{ps}"
                for pt, ps in zip(tbl.column(0).to_pylist(), tbl.column(1).to_pylist())
            ]
            eav_text = "\n".join(blocks)
            (out_dir / f"{q}_{arm}_eav.txt").write_text(eav_text)

            meas = run_measure(
                runner,
                mode,
                profile,
                args.threads,
                data_dir,
                tables,
                sql,
                args.timeout,
            )
            (out_dir / f"{q}_{arm}_measure.json").write_text(json.dumps(meas, indent=2))
            summary[q][arm] = {
                "wall_ms": round(meas["seconds"] * 1000.0, 2),
                "exec_ms": round(meas["exec_seconds"] * 1000.0, 2),
                "exec_cpu_ms": (
                    round(meas["exec_cpu_seconds"] * 1000.0, 2)
                    if meas["exec_cpu_seconds"] is not None
                    else None
                ),
                "bytes_scanned": meas["bytes_scanned"],
                "morsels_claimed": meas["parquet_morsels_claimed"],
                "dyn_filters": meas["join_dynamic_filters_pushed"],
                "adaptive_rf_rgs": meas["adaptive_row_filter_rgs"],
                "adaptive_bulk_rgs": meas["adaptive_bulk_decode_rgs"],
                "adaptive_switches": meas["adaptive_mode_switches"],
            }
    print(json.dumps(summary, indent=2))


def cmd_perf(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(args.runner)
    data_dir = Path(args.data_dir)
    perf_bin = str(args.perf)
    perf_lib = Path(args.perf_lib_dir) if args.perf_lib_dir else None
    env = perf_env(perf_lib)
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    pre_vm = check_host_idle(args.min_idle_pct, args.max_idle_wait)
    results: dict[str, dict] = {"vmstat_pre": pre_vm, "queries": {}}

    for q in queries:
        tables = QUERY_SPECS[q]
        sql = load_sql(q)
        files = parquet_files_for(data_dir, tables)
        res_pages, tot_pages = require_warm_cache(files)
        q_out: dict[str, dict] = {
            "mincore_pages": [res_pages, tot_pages],
            "arms": {},
        }

        # Warmup each arm once before measured perf stat repeats.
        for arm in arms:
            mode, profile = ARMS[arm]
            run_measure(
                runner,
                mode,
                profile,
                args.threads,
                data_dir,
                tables,
                sql,
                args.timeout,
            )

        # Counterbalanced perf stat repeats.
        per_arm_trials: dict[str, list[dict]] = {a: [] for a in arms}
        for r in range(args.stat_repeats):
            order = arms if r % 2 == 0 else list(reversed(arms))
            for arm in order:
                mode, profile = ARMS[arm]
                require_warm_cache(files)
                stat_path = out_dir / f"{q}_{arm}_stat_{r}.csv"
                cmd = [
                    perf_bin,
                    "stat",
                    "-x,",
                    "-o",
                    str(stat_path),
                    "-e",
                    PERF_EVENTS,
                    "--",
                    str(runner),
                    "measure",
                    mode,
                    profile,
                    str(args.threads),
                    str(data_dir),
                    ",".join(tables),
                    sql,
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=args.timeout,
                    check=True,
                )
                require_warm_cache(files)
                meas = json.loads(proc.stdout)
                counters = parse_perf_stat(stat_path)
                per_arm_trials[arm].append({"measure": meas, "counters": counters})

        # Summarize stat medians and run perf record once per arm.
        for arm in arms:
            trials = per_arm_trials[arm]
            wall_ms = [t["measure"]["seconds"] * 1000.0 for t in trials]
            exec_ms = [t["measure"]["exec_seconds"] * 1000.0 for t in trials]
            cpu_ms = [t["measure"]["exec_cpu_seconds"] * 1000.0 for t in trials]
            counter_keys = sorted({k for t in trials for k in t["counters"].keys()})
            med_counters = {
                k: statistics.median(
                    [t["counters"][k] for t in trials if k in t["counters"]]
                )
                for k in counter_keys
            }
            ipc = (
                med_counters["instructions:u"] / med_counters["cycles:u"]
                if med_counters.get("cycles:u", 0.0) > 0
                else None
            )

            mode, profile = ARMS[arm]
            perf_data = out_dir / f"{q}_{arm}.perf"
            report_path = out_dir / f"{q}_{arm}.report"
            require_warm_cache(files)
            subprocess.run(
                [
                    perf_bin,
                    "record",
                    "-q",
                    "-F",
                    "3999",
                    "-e",
                    "cycles:u",
                    "-o",
                    str(perf_data),
                    "--",
                    str(runner),
                    "measure",
                    mode,
                    profile,
                    str(args.threads),
                    str(data_dir),
                    ",".join(tables),
                    sql,
                ],
                capture_output=True,
                env=env,
                timeout=args.timeout,
                check=True,
            )
            rep = subprocess.run(
                [
                    perf_bin,
                    "report",
                    "--stdio",
                    "--no-children",
                    "--percent-limit",
                    "0.3",
                    "-i",
                    str(perf_data),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=args.timeout,
                check=True,
            )
            report_path.write_text(rep.stdout)
            symbols, buckets = parse_perf_report(rep.stdout)

            # Pick median-wall trial for detailed morsel pipeline breakdown.
            med_trial = sorted(trials, key=lambda t: t["measure"]["seconds"])[
                len(trials) // 2
            ]["measure"]
            q_out["arms"][arm] = {
                "wall_ms_median": round(statistics.median(wall_ms), 2),
                "wall_ms_minmax": [
                    round(min(wall_ms), 2),
                    round(max(wall_ms), 2),
                ],
                "exec_ms_median": round(statistics.median(exec_ms), 2),
                "exec_cpu_ms_median": round(statistics.median(cpu_ms), 2),
                "bytes_scanned": med_trial["bytes_scanned"],
                "counters_median": med_counters,
                "ipc": round(ipc, 3) if ipc is not None else None,
                "perf_categories_pct": buckets,
                "top_symbols": symbols[:25],
                "morsel_summary": (
                    {
                        "decode_work_ms": med_trial["morsel_decode_work_ms"],
                        "process_work_ms": med_trial["morsel_process_work_ms"],
                        "ready_wait_ms": med_trial["morsel_ready_wait_ms"],
                        "output_wait_ms": med_trial["morsel_output_wait_ms"],
                        "pending_io_ms": med_trial["morsel_pending_io_ms"],
                        "join_combine_work_ms": med_trial[
                            "morsel_join_combine_work_ms"
                        ],
                        "join_finalize_work_ms": med_trial[
                            "morsel_join_finalize_work_ms"
                        ],
                        "adaptive_rf_rgs": med_trial["adaptive_row_filter_rgs"],
                        "adaptive_bulk_rgs": med_trial["adaptive_bulk_decode_rgs"],
                        "adaptive_switches": med_trial["adaptive_mode_switches"],
                        "dyn_filters": med_trial["join_dynamic_filters_pushed"],
                        "pipelines": med_trial["morsel_pipelines"],
                    }
                    if mode == "required"
                    else None
                ),
            }
        results["queries"][q] = q_out

    results["vmstat_post"] = check_host_idle(0, 1)
    summary_path = out_dir / "perf_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("explain", "perf"):
        p = sub.add_parser(name)
        p.add_argument("--runner", required=True)
        p.add_argument("--data-dir", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--queries", default="q14,q17,q20")
        p.add_argument("--arms", default="stock_pushdown,morsel_pushdown")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--timeout", type=int, default=120)
        if name == "perf":
            p.add_argument("--perf", required=True)
            p.add_argument("--perf-lib-dir", default=None)
            p.add_argument("--stat-repeats", type=int, default=5)
            p.add_argument("--min-idle-pct", type=int, default=85)
            p.add_argument("--max-idle-wait", type=int, default=120)

    args = parser.parse_args()
    if args.cmd == "explain":
        cmd_explain(args)
    else:
        cmd_perf(args)


if __name__ == "__main__":
    main()
