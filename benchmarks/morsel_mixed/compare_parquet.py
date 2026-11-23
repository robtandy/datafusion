# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Paired warm-cache Parquet comparison using one same-runtime harness binary."""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "morsel_parquet"))
from harness import (  # noqa: E402
    LIMITS,
    SEED,
    cache_pages,
    expected_mask,
    fixture_files,
    positive,
    prepare_cache,
    workloads,
)

REVISION = "d649cacf00fee7125f45ea1427a023b2eab732b1"


def fixture(path, table, selectivity):
    files = fixture_files(path)
    metadata = [pq.read_metadata(file) for file in files]
    rows = sum(m.num_rows for m in metadata)
    return files, {
        "path": str(path),
        "input_rows": rows,
        "row_groups": sum(m.num_row_groups for m in metadata),
        "files": len(files),
        "parquet_bytes": sum(file.stat().st_size for file in files),
        "sql": workloads()[f"count_{selectivity}"].replace(
            "FROM fixture", f"FROM {table}"
        ),
        "expected_scalar": int(expected_mask(rows, LIMITS[selectivity]).sum()),
    }


def warm(files):
    for path in files:
        with path.open("rb") as file:
            while file.read(1 << 20):
                pass
    return prepare_cache(files, "warm")


def validate(
    report,
    mode,
    mixed_first,
    partitions,
    revision=REVISION,
    pushdown_filters=False,
):
    if report["build_revision"] != revision:
        raise ValueError(f"expected a binary built against {revision}")
    if report["runtime"]["morsel_mode"] != mode or not report["valid"]:
        raise ValueError(f"invalid {mode} report")
    phases = ["mixed", "isolated"] if mixed_first else ["isolated", "mixed"]
    if [run["mode"] for run in report["runs"]] != phases:
        raise ValueError(f"expected phases {phases}")
    for table in ("small", "large"):
        plan = report[table]["plan"]
        if mode == "required":
            prefix = (
                f"MorselExec: workers={partitions}, NativeParquet[{table}, predicate="
                if pushdown_filters
                else f"MorselExec: workers={partitions}, NativeParquet[{table}]"
            )
            expected = plan.startswith(prefix) and "CountSink" in plan
        else:
            expected = "MorselExec:" not in plan and re.search(
                r"DataSourceExec: .*file_type=parquet", plan
            )
        if not expected:
            raise ValueError(f"unexpected {mode}/{table} plan: {plan}")


def pair_schedule(pair):
    first, second = ("off", "required") if pair % 2 == 0 else ("required", "off")
    # Balance phase order independently of which backend runs first.
    mixed_first = pair % 4 in (1, 2)
    return (first, mixed_first), (second, mixed_first)


def compare(args):
    if platform.system() != "Linux":
        raise ValueError("the comparison requires Linux file page-cache verification")
    small_files, small = fixture(args.small.resolve(), "small", "low")
    large_files, large = fixture(args.large.resolve(), "large", "high")
    if small["input_rows"] >= large["input_rows"]:
        raise ValueError("the large fixture must have more rows than the small fixture")
    files = small_files + large_files
    emit(
        {
            "kind": "setup",
            "build_revision": args.revision,
            "pushdown_filters": args.pushdown_filters,
            "morsel_metrics": args.morsel_metrics,
            "host": platform.node(),
            "kernel": platform.release(),
            "arch": platform.machine(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "page_size": os.sysconf("SC_PAGE_SIZE"),
            "cache_regime": "warm",
            "seed": SEED,
            "small": small,
            "large": large,
            "pairs_per_interval": args.pairs,
            "intervals_ms": args.interval_ms,
        }
    )
    command = [str(args.runner.resolve()), "--repetitions", "1"]
    if args.pushdown_filters:
        command.append("--pushdown-filters")
    if args.morsel_metrics:
        command.append("--morsel-metrics")
    for option in (
        "requests", "workers", "blocking_threads", "partitions", "background_lanes"
    ):
        command += [f"--{option.replace('_', '-')}", str(getattr(args, option))]
    for table, info in (("small", small), ("large", large)):
        command += [
            f"--{table}-fixture",
            info["path"],
            f"--{table}-sql",
            info["sql"],
            f"--{table}-expected-scalar",
            str(info["expected_scalar"]),
        ]
    for interval in args.interval_ms:
        for pair in range(args.pairs):
            for mode, mixed_first in pair_schedule(pair):
                phase_flag = ["--mixed-first"] if mixed_first else []
                before = warm(files)
                completed = subprocess.run(
                    command
                    + ["--morsel-mode", mode, "--interval-ms", str(interval)]
                    + phase_flag,
                    stdout=subprocess.PIPE,
                    timeout=args.timeout,
                    check=False,
                )
                after = cache_pages(files)
                if completed.returncode:
                    # Preserve the harness's JSON diagnostics for invalid runs.
                    sys.stderr.buffer.write(completed.stdout)
                    completed.check_returncode()
                report = json.loads(completed.stdout)
                validate(
                    report,
                    mode,
                    mixed_first,
                    args.partitions,
                    revision=args.revision,
                    pushdown_filters=args.pushdown_filters,
                )
                for run in report["runs"]:
                    background = run["background"]
                    background["input_rows_per_second"] = (
                        background["completed_in_window_per_second"]
                        * large["input_rows"]
                    )
                emit(
                    {
                        "kind": "trial",
                        "pair": pair,
                        "interval_ms": interval,
                        "cache_pages_before": before,
                        "cache_pages_after": after,
                        "report": report,
                    }
                )


def emit(record):
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    for table in ("small", "large"):
        parser.add_argument(
            f"--{table}",
            type=Path,
            required=True,
            help=f"{table} #130 fixture directory",
        )
    parser.add_argument("--interval-ms", type=positive, nargs="+", default=[20, 5])
    parser.add_argument("--pairs", type=positive, default=4)
    parser.add_argument("--requests", type=positive, default=600)
    parser.add_argument("--workers", type=positive, default=4)
    parser.add_argument("--blocking-threads", type=positive, default=2)
    parser.add_argument("--partitions", type=positive, default=4)
    parser.add_argument("--background-lanes", type=positive, default=3)
    parser.add_argument(
        "--revision",
        default=REVISION,
        help="expected MORSEL_BENCH_REVISION embedded in the runner binary",
    )
    parser.add_argument(
        "--pushdown-filters",
        action="store_true",
        help="enable Parquet row-level filter pushdown on both backends",
    )
    parser.add_argument(
        "--morsel-metrics",
        action="store_true",
        help="collect #138 detailed morsel metrics on native runs",
    )
    parser.add_argument(
        "--timeout", type=positive, default=60, help="seconds per harness invocation"
    )
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
