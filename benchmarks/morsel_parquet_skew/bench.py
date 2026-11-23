# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Use #128's data model, SQL, exact-result oracle and warm-cache check.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "morsel_parquet"))
from harness import (
    SCHEMA,
    SEED,
    StreamRunner,
    cache_pages,
    fixture_files,
    mix,
    payloads,
    positive,
    scores,
    validate_path,
    verify,
    workloads,
)

WORKERS = 8
LAYOUTS = {
    # The files with 24 groups land in distinct two-file stock partitions.
    "file-skew": ([24, 1, 24, 1, 2, 2] + [1] * 10, 131_072),
    "large-row-groups": ([2, 2], 2_097_152),
}
MODES = ("off", "required")


class EnvStreamRunner(StreamRunner):
    def __init__(self, binary, path, threads, timeout, mode, env_overrides=None):
        super().__init__(binary, path, threads, timeout, mode)
        self.env = os.environ.copy()
        if env_overrides:
            self.env.update(env_overrides)

    def _run(self, operation, sql, output):
        return subprocess.run(
            [str(self.binary), operation, *self.args, sql],
            stdout=output,
            check=True,
            timeout=self.timeout,
            env=self.env,
        ).stdout


def layout(case, smoke):
    groups, group_rows = LAYOUTS[case]
    return groups, group_rows // 1024 if smoke else group_rows


def generate(path, groups, group_rows, write_page_index=False):
    path.mkdir(parents=True)
    start = 0
    for file, num_groups in enumerate(groups):
        with pq.ParquetWriter(
            path / f"part-{file:06d}.parquet",
            SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            write_statistics=True,
            write_page_index=write_page_index,
            version="2.6",
            data_page_version="1.0",
        ) as writer:
            for _ in range(num_groups):
                ids = np.arange(start, start + group_rows, dtype=np.int64)
                score, delta = scores(ids)
                table = pa.Table.from_arrays(
                    [
                        pa.array(ids),
                        pa.array(score, mask=ids % 19 == 0),
                        pa.array(delta, mask=ids % 23 == 0),
                        pa.array(payloads(ids), mask=ids % 11 == 0),
                        pa.array((mix(ids, SEED ^ (1 << 61)) >> 16).astype(np.int64)),
                    ],
                    schema=SCHEMA,
                )
                writer.write_table(table, row_group_size=group_rows)
                start += group_rows


def check_layout(files, groups, group_rows):
    if len(files) != len(groups):
        raise ValueError(f"expected {len(groups)} files, found {len(files)}")
    for path, count in zip(files, groups, strict=True):
        meta = pq.read_metadata(path)
        if meta.num_row_groups != count or any(
            meta.row_group(i).num_rows != group_rows for i in range(count)
        ):
            raise ValueError(
                f"{path}: expected {count} row groups of {group_rows} rows"
            )


def validate_sample(
    result,
    mode,
    query,
    row_groups,
    expected,
    revision,
    allow_split=False,
    pushdown_filters=False,
):
    split_rgs = result.get("parquet_split_row_groups", 0)
    split_slices = result.get("parquet_split_slices", 0)
    expected_morsels = row_groups
    if mode != "off" and allow_split and split_rgs > 0:
        expected_morsels = split_slices + (row_groups - split_rgs)
        if expected_morsels <= row_groups:
            raise ValueError(
                f"{mode}/{query}: split_rgs={split_rgs} must yield more morsels than {row_groups}"
            )
    plan = result["plan"]
    if mode == "off":
        validate_path(result, mode, query, expected_morsels)
    else:
        if not plan.startswith("MorselExec:") or not re.search(
            r"NativeParquet\[fixture(?:,|\])", plan
        ):
            raise ValueError(
                f"{query}: expected a native Parquet morsel plan, got {plan}"
            )
        if result["parquet_morsels_claimed"] != expected_morsels:
            raise ValueError(
                f"{query}: expected to claim {expected_morsels} Parquet morsels, got {result['parquet_morsels_claimed']}"
            )
        count = query.startswith("count_")
        if ("CountSink" in plan) != count or result["count_finalizations"] != int(
            count
        ):
            raise ValueError(
                f"{query}: unexpected native count sink or finalization count"
            )
    if result["build_revision"] != revision or result["output_rows"] != expected:
        raise ValueError(
            f"{mode}/{query}: unexpected build revision or output row count"
        )
    claims, completions = result["worker_claims"], result["worker_completions"]
    if mode == "off":
        if claims is not None or completions is not None:
            raise ValueError(f"{mode}/{query}: unexpected native metrics")
    elif (
        not isinstance(claims, list)
        or not isinstance(completions, list)
        or len(claims) != WORKERS
        or len(completions) != WORKERS
        or any(type(n) is not int or n < 0 for n in claims)
        or any(type(n) is not int or n < 0 for n in completions)
        or sum(claims) != expected_morsels
        or sum(completions) != expected_morsels
        or (not pushdown_filters and claims != completions)
    ):
        raise ValueError(f"{mode}/{query}: invalid per-worker claims/completions")


def warm_pages(files):
    # A shared host can reclaim pages between trials. Reheat outside the timer,
    # but do not label a partially resident fixture as warm.
    for _ in range(2):
        pages = cache_pages(files)
        if pages["resident"] == pages["total"]:
            return pages
        for path in files:
            with path.open("rb") as file:
                while file.read(1 << 20):
                    pass
    pages = cache_pages(files)
    if pages["resident"] != pages["total"]:
        raise ValueError(f"cannot verify warm file page cache: {pages}")
    return pages


def benchmark(args, groups, group_rows):
    if platform.system() != "Linux":
        raise ValueError("the warm-cache check requires Linux")
    files = fixture_files(args.path)
    check_layout(files, groups, group_rows)
    row_groups = sum(groups)
    rows = row_groups * group_rows

    base_env = {
        "MORSEL_ENABLE_PAGE_INDEX": "1" if args.enable_page_index else "0",
        "MORSEL_PUSHDOWN_FILTERS": "1" if args.pushdown_filters else "0",
    }
    if args.compare_split:
        arm_specs = (
            ("off", "off", base_env),
            ("legacy", "required", {**base_env, "MORSEL_SPLIT_ENABLED": "0"}),
            ("split", "required", {**base_env, "MORSEL_SPLIT_ENABLED": "1"}),
        )
    else:
        native_env = (
            {**base_env, "MORSEL_SPLIT_ENABLED": "1"} if args.enable_split else base_env
        )
        arm_specs = (("off", "off", base_env), ("required", "required", native_env))
    arm_names = tuple(name for name, _, _ in arm_specs)
    arm_modes = {name: mode for name, mode, _ in arm_specs}
    backends = {
        name: EnvStreamRunner(args.runner, args.path, WORKERS, args.timeout, mode, env)
        for name, mode, env in arm_specs
    }
    allow_split = args.enable_split or args.compare_split

    # No timings are emitted until all arms pass both the oracle and plan checks.
    matches = verify(backends[arm_names[0]], rows)
    for name in arm_names[1:]:
        if verify(backends[name], rows) != matches:
            raise ValueError(f"oracle mismatch for arm {name}")
    queries = workloads()
    expected = {f"scan_{name}": n for name, n in matches.items()}
    expected.update({f"count_{name}": 1 for name in matches})

    def sample(arm, query):
        result = backends[arm].measure(queries[query])
        validate_sample(
            result,
            arm_modes[arm],
            query,
            row_groups,
            expected[query],
            args.revision,
            allow_split=allow_split and arm != "legacy",
            pushdown_filters=args.pushdown_filters,
        )
        return result

    evidence = []
    for query in queries:
        for arm in arm_names:
            result = sample(arm, query)
            evidence.append(
                {
                    "mode": arm,
                    "query": query,
                    **{
                        key: result[key]
                        for key in (
                            "plan",
                            "parquet_morsels_claimed",
                            "count_finalizations",
                        )
                    },
                }
            )
    print(
        json.dumps(
            {
                "fixture": args.case,
                "smoke": args.smoke,
                "host": platform.node(),
                "kernel": platform.release(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "workers": WORKERS,
                "build_revision": args.revision,
                "cache_regime": "warm",
                "enable_page_index": args.enable_page_index,
                "pushdown_filters": args.pushdown_filters,
                "compare_split": args.compare_split,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "row_groups_per_file": groups,
                "rows_per_group": group_rows,
                "input_rows": rows,
                "parquet_bytes": sum(path.stat().st_size for path in files),
                "verified_matches": matches,
                "evidence": evidence,
            }
        ),
        flush=True,
    )
    for rep in range(1 - args.warmups, args.repeats + 1):
        if len(arm_names) == 2:
            order = arm_names if rep % 2 else arm_names[::-1]
        else:
            shift = rep % len(arm_names)
            order = arm_names[shift:] + arm_names[:shift]
        for query in queries:
            for arm in order:
                pages = warm_pages(files)
                result = sample(arm, query)
                if rep > 0:
                    del result["plan"], result["build_revision"]
                    print(
                        json.dumps(
                            {
                                "rep": rep,
                                "query": query,
                                "mode": arm,
                                "cache_pages_before": pages,
                                **result,
                            }
                        ),
                        flush=True,
                    )


def main():
    parser = argparse.ArgumentParser(description="Parquet morsel skew A/B")
    commands = parser.add_subparsers(dest="command", required=True)
    generate_cmd = commands.add_parser("generate")
    ab = commands.add_parser("ab")
    for cmd in (generate_cmd, ab):
        cmd.add_argument("case", choices=LAYOUTS)
        cmd.add_argument("path", type=Path)
        cmd.add_argument("--smoke", action="store_true", help="use smaller row groups")
    generate_cmd.add_argument(
        "--page-index",
        action="store_true",
        help="write Parquet ColumnIndex and OffsetIndex",
    )
    ab.add_argument("--runner", type=Path, required=True)
    ab.add_argument("--revision", required=True)
    ab.add_argument("--timeout", type=positive, default=120, help="seconds per query")
    ab.add_argument("--warmups", type=int, default=1)
    ab.add_argument("--repeats", type=positive, default=5)
    ab.add_argument(
        "--enable-page-index",
        action="store_true",
        help="enable Parquet page/offset index reads in the runner",
    )
    ab.add_argument(
        "--pushdown-filters",
        action="store_true",
        help="enable Parquet row-filter pushdown in the runner",
    )
    ab.add_argument(
        "--enable-split",
        action="store_true",
        help="opt into intra-row-group splitting in the required arm",
    )
    ab.add_argument(
        "--compare-split",
        action="store_true",
        help="run paired 3-arm comparison: stock (off), legacy native (split off), and split native",
    )
    args = parser.parse_args()
    groups, group_rows = layout(args.case, args.smoke)
    if args.command == "generate":
        generate(args.path, groups, group_rows, write_page_index=args.page_index)
        return
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    benchmark(args, groups, group_rows)


if __name__ == "__main__":
    main()
