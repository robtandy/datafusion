# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Run counterbalanced quiet-host SF5 scratch ablations for TPC-H Q14, Q17, and Q20 (#215)."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import pyarrow.ipc as pa_ipc

from profile_q14_q17_q20 import (
    QUERY_SPECS,
    check_host_idle,
    load_sql,
    parquet_files_for,
    require_warm_cache,
)

ABLATION_CONFIGS: dict[str, list[tuple[str, bool, str, str, dict[str, str]]]] = {
    "q14": [
        ("stock_pushdown", False, "off", "pushdown", {}),
        ("morsel_baseline", False, "required", "pushdown", {}),
        (
            "+radix_nonatomic_bloom (part)",
            True,
            "required",
            "pushdown",
            {"MORSEL_SCRATCH_NONATOMIC_BLOOM": "1"},
        ),
        (
            "+skip_1m_bloom (part)",
            True,
            "required",
            "pushdown",
            {"MORSEL_SCRATCH_SKIP_LARGE_BLOOM": "500000"},
        ),
        (
            "+stats+wave1_rf (lineitem)",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_STATS_BUILD_SIDE": "1",
                "MORSEL_SCRATCH_FIX_WAVE1_RF": "1",
            },
        ),
        (
            "+stats+wave1+radix_bloom",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_STATS_BUILD_SIDE": "1",
                "MORSEL_SCRATCH_FIX_WAVE1_RF": "1",
                "MORSEL_SCRATCH_NONATOMIC_BLOOM": "1",
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
            },
        ),
        (
            "+stats+wave1+radix_bloom+ra2x",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_STATS_BUILD_SIDE": "1",
                "MORSEL_SCRATCH_FIX_WAVE1_RF": "1",
                "MORSEL_SCRATCH_NONATOMIC_BLOOM": "1",
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
                "MORSEL_SCRATCH_READ_AHEAD_2X": "1",
            },
        ),
    ],
    "q17": [
        ("stock_pushdown", False, "off", "pushdown", {}),
        ("morsel_baseline", False, "required", "pushdown", {}),
        (
            "+infallible_bloom",
            True,
            "required",
            "pushdown",
            {"MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1"},
        ),
        (
            "+inf_bloom+no_range",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
            },
        ),
        (
            "+inf+no_rng+bloom32b",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
                "MORSEL_SCRATCH_BLOOM_BITS": "32",
            },
        ),
        (
            "+inf+no_rng+b32+force_rf",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
                "MORSEL_SCRATCH_BLOOM_BITS": "32",
                "MORSEL_SCRATCH_FORCE_RF_ON_DYN": "1",
            },
        ),
        (
            "+inf+no_rng+b32+frf+ra2x",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
                "MORSEL_SCRATCH_BLOOM_BITS": "32",
                "MORSEL_SCRATCH_FORCE_RF_ON_DYN": "1",
                "MORSEL_SCRATCH_READ_AHEAD_2X": "1",
            },
        ),
    ],
    "q20": [
        ("stock_pushdown", False, "off", "pushdown", {}),
        ("morsel_baseline", False, "required", "pushdown", {}),
        (
            "+q20_suppkey_push",
            True,
            "required",
            "pushdown",
            {"MORSEL_SCRATCH_Q20_SUPPKEY_PUSH": "1"},
        ),
        (
            "+suppkey+inf+no_rng",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_Q20_SUPPKEY_PUSH": "1",
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
            },
        ),
        (
            "+suppkey+inf+no_rng+b32+frf",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_Q20_SUPPKEY_PUSH": "1",
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
                "MORSEL_SCRATCH_BLOOM_BITS": "32",
                "MORSEL_SCRATCH_FORCE_RF_ON_DYN": "1",
            },
        ),
        (
            "+suppkey+inf+no_rng+b32+frf+ra2x",
            True,
            "required",
            "pushdown",
            {
                "MORSEL_SCRATCH_Q20_SUPPKEY_PUSH": "1",
                "MORSEL_SCRATCH_INFALLIBLE_BLOOM": "1",
                "MORSEL_SCRATCH_NO_RANGE_WITH_BLOOM": "1",
                "MORSEL_SCRATCH_BLOOM_BITS": "32",
                "MORSEL_SCRATCH_FORCE_RF_ON_DYN": "1",
                "MORSEL_SCRATCH_READ_AHEAD_2X": "1",
            },
        ),
    ],
}


def query_result(
    runner: Path,
    mode: str,
    profile: str,
    threads: int,
    data_dir: Path,
    tables: tuple[str, ...],
    sql: str,
    env_extra: dict[str, str],
    timeout_s: int,
) -> dict:
    env = os.environ.copy()
    env.update(env_extra)
    proc = subprocess.run(
        [
            str(runner),
            "query",
            mode,
            profile,
            str(threads),
            str(data_dir),
            ",".join(tables),
            sql,
        ],
        capture_output=True,
        env=env,
        timeout=timeout_s,
        check=True,
    )
    return pa_ipc.open_stream(proc.stdout).read_all().to_pydict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runner", required=True)
    parser.add_argument("--scratch-runner", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--queries", default="q14,q17,q20")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--min-idle-pct", type=int, default=85)
    parser.add_argument("--max-idle-wait", type=int, default=120)
    args = parser.parse_args()

    base_runner = Path(args.baseline_runner)
    scratch_runner = Path(args.scratch_runner)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    all_out: dict[str, dict] = {}
    for q in queries:
        tables = QUERY_SPECS[q]
        sql = load_sql(q)
        files = parquet_files_for(data_dir, tables)
        res_pages, tot_pages = require_warm_cache(files)
        configs = ABLATION_CONFIGS[q]

        ref = query_result(
            base_runner,
            "off",
            "pushdown",
            args.threads,
            data_dir,
            tables,
            sql,
            {},
            args.timeout,
        )
        for name, use_scratch, mode, prof, extra in configs:
            if extra:
                rbin = scratch_runner if use_scratch else base_runner
                got = query_result(
                    rbin,
                    mode,
                    prof,
                    args.threads,
                    data_dir,
                    tables,
                    sql,
                    extra,
                    args.timeout,
                )
                if got != ref:
                    raise AssertionError(f"{q} parity mismatch for {name}")

        pre = check_host_idle(args.min_idle_pct, args.max_idle_wait)
        # Warmup
        for _, use_scratch, mode, prof, extra in configs[:2]:
            rbin = scratch_runner if use_scratch else base_runner
            env = os.environ.copy()
            env.update(extra)
            subprocess.run(
                [
                    str(rbin),
                    "measure",
                    mode,
                    prof,
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

        trials_by_cfg: dict[str, list[dict]] = {c[0]: [] for c in configs}
        for r in range(args.repeats):
            order = configs if r % 2 == 0 else list(reversed(configs))
            for name, use_scratch, mode, prof, extra in order:
                require_warm_cache(files)
                rbin = scratch_runner if use_scratch else base_runner
                env = os.environ.copy()
                env.update(extra)
                proc = subprocess.run(
                    [
                        str(rbin),
                        "measure",
                        mode,
                        prof,
                        str(args.threads),
                        str(data_dir),
                        ",".join(tables),
                        sql,
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=args.timeout,
                    check=True,
                )
                trials_by_cfg[name].append(json.loads(proc.stdout))

        post = check_host_idle(0, 1)
        q_summary: dict[str, dict] = {
            "mincore_pages": [res_pages, tot_pages],
            "vmstat_pre": pre,
            "vmstat_post": post,
            "configs": {},
        }
        for name, _, mode, _, _ in configs:
            trials = trials_by_cfg[name]
            med = sorted(trials, key=lambda x: x["seconds"])[len(trials) // 2]
            walls = [t["seconds"] * 1000.0 for t in trials]
            execs = [t["exec_seconds"] * 1000.0 for t in trials]
            cpus = [t["exec_cpu_seconds"] * 1000.0 for t in trials]
            q_summary["configs"][name] = {
                "wall_median_ms": round(statistics.median(walls), 2),
                "wall_minmax_ms": [round(min(walls), 2), round(max(walls), 2)],
                "exec_median_ms": round(statistics.median(execs), 2),
                "cpu_median_ms": round(statistics.median(cpus), 2),
                "median_trial": med,
            }
        all_out[q] = q_summary
        (out_dir / f"{q}_ablations.json").write_text(json.dumps(q_summary, indent=2))
    print(json.dumps(all_out, indent=2))


if __name__ == "__main__":
    main()
