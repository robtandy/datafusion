# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Run controlled scratch ablations built from ablations.patch (#180)."""

import argparse
import json
from pathlib import Path
import statistics

import pyarrow.ipc as pa_ipc

from profile_harness import (
    ARMS,
    WORKLOADS,
    compare_tables,
    require_warm_cache,
    run_df,
    run_duckdb_subprocess,
    summarize_morsel_runs,
    table_files,
    warm_files,
)

VARIANTS: list[tuple[str, str, dict[str, str]]] = [
    ("native", "morsel_no_pushdown", {}),
    ("no_batch_yield", "morsel_no_pushdown", {"MORSEL_SCRATCH_NO_YIELD": "1"}),
    (
        "short_circuit",
        "morsel_no_pushdown",
        {"MORSEL_SCRATCH_SHORT_CIRCUIT": "1"},
    ),
    (
        "no_yield_and_short_circuit",
        "morsel_no_pushdown",
        {"MORSEL_SCRATCH_NO_YIELD": "1", "MORSEL_SCRATCH_SHORT_CIRCUIT": "1"},
    ),
    ("pushdown_base", "morsel_pushdown", {}),
    ("pushdown_noreorder", "morsel_pushdown_noreorder", {}),
    (
        "pushdown_grouped",
        "morsel_pushdown",
        {"MORSEL_SCRATCH_GROUP_ROW_FILTER": "1"},
    ),
    (
        "pushdown_grouped_noreorder",
        "morsel_pushdown_noreorder",
        {"MORSEL_SCRATCH_GROUP_ROW_FILTER": "1"},
    ),
]


FLBA_VARIANTS: list[tuple[str, str, dict[str, str]]] = [
    ("baseline_flba", "morsel_no_pushdown", {}),
    ("direct_flba_dec128", "morsel_no_pushdown", {"MORSEL_SCRATCH_FLBA_DIRECT": "1"}),
    ("baseline_pushdown", "morsel_pushdown", {}),
    ("direct_flba_pushdown", "morsel_pushdown", {"MORSEL_SCRATCH_FLBA_DIRECT": "1"}),
    ("direct_flba_stock", "stock_no_pushdown", {"MORSEL_SCRATCH_FLBA_DIRECT": "1"}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--suite", choices=("datafusion", "flba"), default="datafusion")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    variants = VARIANTS if args.suite == "datafusion" else FLBA_VARIANTS
    data_dir = Path(args.data_dir).resolve(strict=True)
    runner = Path(args.runner).resolve(strict=True)
    files = table_files(data_dir, "lineitem")
    warm_files(files)
    require_warm_cache(files)

    # Verify exact parity against DuckDB oracle for every variant first.
    for case in ("q6_count", "q6_scan"):
        wl = WORKLOADS[case]
        oracle_raw = run_duckdb_subprocess(
            "_duckdb-fresh",
            wl,
            data_dir,
            args.threads,
            args.batch_size,
            args.timeout,
            ["query"],
        )
        oracle = pa_ipc.open_stream(oracle_raw).read_all()
        for vname, arm_name, env in variants:
            arm = ARMS[arm_name]
            actual_raw = run_df(
                runner,
                "query",
                arm,
                wl,
                data_dir,
                args.threads,
                args.batch_size,
                False,
                args.timeout,
                extra_env=env,
            )
            actual = pa_ipc.open_stream(actual_raw).read_all()
            compare_tables(oracle, actual, wl.ordered, case, vname)
        print(
            json.dumps(
                {
                    "kind": "verify_ablation",
                    "suite": args.suite,
                    "case": case,
                    "oracle_rows": oracle.num_rows,
                    "variants": [v[0] for v in variants],
                }
            ),
            flush=True,
        )

    for case in ("q6_count", "q6_scan"):
        wl = WORKLOADS[case]
        for vname, arm_name, env in variants:
            arm = ARMS[arm_name]
            op = "measure_exec_loop" if arm.mode == "required" else "measure_warm"
            require_warm_cache(files)
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
                    extra_env=env,
                )
            )
            require_warm_cache(files)
            fresh_tots = []
            if args.fresh:
                for rnd in range(1 + args.repeats):
                    require_warm_cache(files)
                    s = json.loads(
                        run_df(
                            runner,
                            "measure",
                            arm,
                            wl,
                            data_dir,
                            args.threads,
                            args.batch_size,
                            False,
                            args.timeout,
                            extra_env=env,
                        )
                    )
                    require_warm_cache(files)
                    if rnd >= 1:
                        fresh_tots.append(s["seconds"] * 1000.0)
            for it in res["iterations"]:
                it["morsel_summary"] = summarize_morsel_runs(it.pop("morsel_runs", []))
            execs = [it["exec_seconds"] * 1000.0 for it in res["iterations"]]
            cpus = [it["exec_cpu_seconds"] * 1000.0 for it in res["iterations"]]
            ms_list = [
                it["morsel_summary"] for it in res["iterations"] if it["morsel_summary"]
            ]
            print(
                json.dumps(
                    {
                        "kind": "ablation",
                        "suite": args.suite,
                        "case": case,
                        "variant": vname,
                        "arm": arm_name,
                        "env": env,
                        "median_exec_ms": statistics.median(execs),
                        "min_exec_ms": min(execs),
                        "max_exec_ms": max(execs),
                        "median_cpu_ms": statistics.median(cpus),
                        "median_fresh_total_ms": (
                            statistics.median(fresh_tots) if fresh_tots else None
                        ),
                        "median_decode_ms": (
                            statistics.median(m["decode_ms"] for m in ms_list)
                            if ms_list
                            else None
                        ),
                        "median_process_ms": (
                            statistics.median(m["process_ms"] for m in ms_list)
                            if ms_list
                            else None
                        ),
                        "median_ready_wait_ms": (
                            statistics.median(m["ready_wait_ms"] for m in ms_list)
                            if ms_list
                            else None
                        ),
                        "median_pending_io_ms": (
                            statistics.median(m["pending_io_ms"] for m in ms_list)
                            if ms_list
                            else None
                        ),
                        "iterations": res["iterations"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
