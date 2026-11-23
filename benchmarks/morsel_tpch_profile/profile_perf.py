# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Capture Arm PMUv3 counters and leaf CPU profiles for DataFusion and DuckDB (#180)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from profile_harness import ARMS, WORKLOADS, require_warm_cache, table_files, warm_files

EVENTS = "cycles:u,instructions:u,r17:u,task-clock,context-switches,cpu-migrations"
HARNESS = Path(__file__).resolve().parent / "profile_harness.py"


def parse_perf_stat(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw_val, _, event = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not raw_val or raw_val.startswith("<"):
            continue
        try:
            metrics[event] = float(raw_val)
        except ValueError:
            continue
    return metrics


def df_exec_loop_cmd(
    runner: Path,
    arm_name: str,
    case: str,
    data_dir: Path,
    threads: int,
    batch_size: int,
    warmups: int,
    repeats: int,
) -> list[str]:
    arm = ARMS[arm_name]
    wl = WORKLOADS[case]
    # Stock DataSourceExec consumes FileGroup work-stealing cursors on first execute(),
    # so use measure_warm (re-plan per iter, ~1.2 ms) for stock and measure_exec_loop for morsel.
    op = "measure_exec_loop" if arm.mode == "required" else "measure_warm"
    return [
        str(runner),
        op,
        arm.mode,
        arm.profile,
        str(threads),
        str(batch_size),
        "off",
        str(data_dir),
        ",".join(wl.tables),
        wl.sql,
        str(warmups),
        str(repeats),
    ]


def duckdb_warm_cmd(
    case: str,
    data_dir: Path,
    threads: int,
    batch_size: int,
    warmups: int,
    repeats: int,
    drain_mode: str = "arrow",
) -> list[str]:
    wl = WORKLOADS[case]
    return [
        sys.executable,
        "-B",
        str(HARNESS),
        "_duckdb-warm",
        str(threads),
        str(batch_size),
        str(data_dir),
        ",".join(wl.tables),
        wl.sql,
        str(warmups),
        str(repeats),
        drain_mode,
        "0",
    ]


def run_stat(
    perf: str,
    cmd: list[str],
    stat_path: Path,
    files: list[Path],
    timeout: int,
) -> tuple[dict, dict[str, float]]:
    require_warm_cache(files)
    proc = subprocess.run(
        [perf, "stat", "-x,", "-o", str(stat_path), "-e", EVENTS, "--", *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"perf stat failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    require_warm_cache(files)
    return json.loads(proc.stdout), parse_perf_stat(stat_path)


def run_record(
    perf: str,
    cmd: list[str],
    stem: Path,
    files: list[Path],
    timeout: int,
) -> dict:
    require_warm_cache(files)
    perf_data = Path(f"{stem}.perf")
    report_path = Path(f"{stem}.report")
    proc = subprocess.run(
        [
            perf,
            "record",
            "-q",
            "-F",
            "3999",
            "-e",
            "cycles:u",
            "-o",
            str(perf_data),
            "--",
            *cmd,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"perf record failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    require_warm_cache(files)
    with report_path.open("w") as out:
        subprocess.run(
            [
                perf,
                "report",
                "--stdio",
                "--no-children",
                "--sort",
                "symbol",
                "-F",
                "overhead,sample,symbol",
                "-i",
                str(perf_data),
            ],
            stdout=out,
            check=True,
            timeout=60,
        )
    return json.loads(proc.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--perf", default=os.environ.get("PERF_BIN", "perf"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--stat-reps", type=int, default=3)
    parser.add_argument("--loop-iters", type=int, default=20)
    parser.add_argument("--skip-steady-stat", action="store_true")
    parser.add_argument("--skip-batch-stat", action="store_true")
    parser.add_argument("--skip-record", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve(strict=True)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(args.runner).resolve(strict=True)
    files = table_files(data_dir, "lineitem")
    warm_files(files)
    require_warm_cache(files)

    stat_arms = [
        "stock_no_pushdown",
        "morsel_no_pushdown",
        "morsel_pushdown",
        "stock_pushdown",
        "duckdb_default",
    ]

    # 1. Steady-state perf stat: measure baseline (0 repeats) and N repeats to subtract startup.
    if not args.skip_steady_stat:
        for case in ("q6_count", "q6_scan"):
            iters = (
                args.loop_iters if case == "q6_count" else max(10, args.loop_iters // 2)
            )
            for arm_name in stat_arms:
                # 0-repeat baseline (1 warmup only) for startup subtraction
                if arm_name == "duckdb_default":
                    base_cmd = duckdb_warm_cmd(case, data_dir, args.threads, 8192, 1, 0)
                    full_cmd = duckdb_warm_cmd(
                        case, data_dir, args.threads, 8192, 1, iters
                    )
                else:
                    base_cmd = df_exec_loop_cmd(
                        runner, arm_name, case, data_dir, args.threads, 8192, 1, 0
                    )
                    full_cmd = df_exec_loop_cmd(
                        runner, arm_name, case, data_dir, args.threads, 8192, 1, iters
                    )
                for rep in range(1, args.stat_reps + 1):
                    _, base_counters = run_stat(
                        args.perf,
                        base_cmd,
                        out_dir / f"{case}.{arm_name}.base.{rep}.stat",
                        files,
                        args.timeout,
                    )
                    full_res, full_counters = run_stat(
                        args.perf,
                        full_cmd,
                        out_dir / f"{case}.{arm_name}.full.{rep}.stat",
                        files,
                        args.timeout,
                    )
                    per_iter = {
                        k: (full_counters[k] - base_counters.get(k, 0.0)) / iters
                        for k in full_counters
                    }
                    print(
                        json.dumps(
                            {
                                "kind": "perf_stat_steady",
                                "case": case,
                                "arm": arm_name,
                                "rep": rep,
                                "iters": iters,
                                "base_counters": base_counters,
                                "full_counters": full_counters,
                                "per_iter_counters": per_iter,
                                "median_exec_ms": sorted(
                                    it["exec_seconds"] * 1000.0
                                    for it in full_res["iterations"]
                                )[len(full_res["iterations"]) // 2],
                            }
                        ),
                        flush=True,
                    )

    # 2. Batch-size perf stat on morsel_no_pushdown (2K, 4K, 8K, 16K).
    if not args.skip_batch_stat:
        for case in ("q6_count", "q6_scan"):
            iters = (
                args.loop_iters if case == "q6_count" else max(10, args.loop_iters // 2)
            )
            for bs in (2048, 4096, 8192, 16384):
                base_cmd = df_exec_loop_cmd(
                    runner, "morsel_no_pushdown", case, data_dir, args.threads, bs, 1, 0
                )
                full_cmd = df_exec_loop_cmd(
                    runner,
                    "morsel_no_pushdown",
                    case,
                    data_dir,
                    args.threads,
                    bs,
                    1,
                    iters,
                )
                _, base_counters = run_stat(
                    args.perf,
                    base_cmd,
                    out_dir / f"{case}.bs{bs}.base.stat",
                    files,
                    args.timeout,
                )
                full_res, full_counters = run_stat(
                    args.perf,
                    full_cmd,
                    out_dir / f"{case}.bs{bs}.full.stat",
                    files,
                    args.timeout,
                )
                per_iter = {
                    k: (full_counters[k] - base_counters.get(k, 0.0)) / iters
                    for k in full_counters
                }
                print(
                    json.dumps(
                        {
                            "kind": "perf_stat_batch",
                            "case": case,
                            "arm": "morsel_no_pushdown",
                            "batch_size": bs,
                            "iters": iters,
                            "per_iter_counters": per_iter,
                            "median_exec_ms": sorted(
                                it["exec_seconds"] * 1000.0
                                for it in full_res["iterations"]
                            )[len(full_res["iterations"]) // 2],
                        }
                    ),
                    flush=True,
                )

    # 3. perf record leaf symbol profiles on steady-state warm loops.
    if not args.skip_record:
        record_arms = [
            "morsel_no_pushdown",
            "morsel_pushdown",
            "stock_no_pushdown",
            "duckdb_default",
        ]
        for case in ("q6_count", "q6_scan"):
            iters = 30 if case == "q6_count" else 15
            for arm_name in record_arms:
                if arm_name == "duckdb_default":
                    cmd = duckdb_warm_cmd(case, data_dir, args.threads, 8192, 2, iters)
                else:
                    cmd = df_exec_loop_cmd(
                        runner, arm_name, case, data_dir, args.threads, 8192, 2, iters
                    )
                stem = out_dir / f"record.{case}.{arm_name}"
                res = run_record(args.perf, cmd, stem, files, args.timeout)
                print(
                    json.dumps(
                        {
                            "kind": "perf_record",
                            "case": case,
                            "arm": arm_name,
                            "iters": iters,
                            "report": f"{stem}.report",
                            "median_exec_ms": sorted(
                                it["exec_seconds"] * 1000.0 for it in res["iterations"]
                            )[len(res["iterations"]) // 2],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
