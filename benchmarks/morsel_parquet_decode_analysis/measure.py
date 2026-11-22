# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Replay #155 against the pinned #128 harness; raw NDJSON goes to stdout."""

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "morsel_parquet"))
import harness

WORKER_KEYS = {
    "morsels_claimed",
    "morsels_completed",
    "morsel_decode_work",
    "morsel_process_work",
    "morsel_ready_wait",
    "morsel_output_wait",
}
RUN_KEYS = {
    "morsel_pending_io",
    "morsel_peak_ready_depth",
    "morsel_peak_ready_compressed_bytes_estimate",
    "morsel_peak_output_depth",
    "morsel_peak_output_arrow_bytes",
}
VARIANTS = {
    "stock": ("off", {}),
    "native": ("required", {}),
    "unmeasured": ("required", {"MORSEL_NO_METRICS": "1"}),
    "direct": ("required", {"MORSEL_SCRATCH_DIRECT": "1"}),
    "row_group_yield": ("required", {"MORSEL_SCRATCH_NO_YIELD": "1"}),
    "direct_row_group_yield": (
        "required",
        {"MORSEL_SCRATCH_DIRECT": "1", "MORSEL_SCRATCH_NO_YIELD": "1"},
    ),
    "producer8": ("required", {"MORSEL_SCRATCH_PRODUCER": "1"}),
    "output32": ("required", {"MORSEL_SCRATCH_OUTPUT": "1"}),
    "both_queues": (
        "required",
        {"MORSEL_SCRATCH_PRODUCER": "1", "MORSEL_SCRATCH_OUTPUT": "1"},
    ),
    "output32_unmeasured": (
        "required",
        {"MORSEL_SCRATCH_OUTPUT": "1", "MORSEL_NO_METRICS": "1"},
    ),
    "prune": ("required", {"MORSEL_SCRATCH_PRUNE": "1"}),
    "mask": ("required", {"MORSEL_SCRATCH_COUNT": "1"}),
    "stock_blocking16": ("off", {"MORSEL_SCRATCH_BLOCKING": "1"}),
    "native_blocking16": ("required", {"MORSEL_SCRATCH_BLOCKING": "1"}),
}


def validate_metrics(runs, detailed, workers):
    if not detailed:
        if runs:
            raise ValueError("unexpected detailed report")
        return {}
    if len(runs) != 1:
        raise ValueError(f"expected one detailed report, got {len(runs)}")
    counters = {(m["worker"], m["name"]): m["value"] for m in runs[0]}
    expected = {(None, name) for name in RUN_KEYS} | {
        (str(i), name) for i in range(workers) for name in WORKER_KEYS
    }
    if counters.keys() != expected or len(counters) != len(runs[0]):
        raise ValueError("missing, duplicate or unexpected detailed metrics")
    result = {key: counters[None, key] for key in RUN_KEYS}
    result.update(
        {key: [counters[str(i), key] for i in range(workers)] for key in WORKER_KEYS}
    )
    if (
        result["morsels_claimed"] != result["morsels_completed"]
        or sum(result["morsels_completed"]) != 128
    ):
        raise ValueError("expected exactly 128 completions and matching worker claims")
    return result


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("revision")
    parser.add_argument("workers", type=int, choices=[1, 4, 8])
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=["stock", "native"]
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=harness.workloads(),
        default=list(harness.workloads()),
    )
    parser.add_argument("--repeats", type=harness.positive, default=8)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    files = harness.fixture_files(args.fixture)
    row_groups = [pq.ParquetFile(f).metadata for f in files]
    if len(files) != 16 or any(
        f.num_row_groups != 8
        or any(f.row_group(i).num_rows != 131072 for i in range(f.num_row_groups))
        for f in row_groups
    ):
        raise ValueError("expected the #128 large fixture")
    emit(
        {
            "revision": args.revision,
            "workers": args.workers,
            "cpus": sorted(os.sched_getaffinity(0)),
            "fixture_bytes": sum(f.stat().st_size for f in files),
            "profile": args.profile,
        }
    )
    for key in set(k for _, flags in VARIANTS.values() for k in flags) | {
        "MORSEL_SCRATCH_PROFILE"
    }:
        os.environ.pop(key, None)
    if args.profile:
        os.environ["MORSEL_SCRATCH_PROFILE"] = "1"

    rows = 16 * 8 * 131072
    for variant in args.variants:
        mode, flags = VARIANTS[variant]
        os.environ.update(flags)
        backend = harness.StreamRunner(
            args.runner, args.fixture, args.workers, 90, mode
        )
        # Mask targets filtered COUNT only; run the unchanged scan path in the oracle.
        emit({"variant": variant, "verified_matches": harness.verify(backend, rows)})
        for key in flags:
            del os.environ[key]

    for path in files:
        with path.open("rb") as file:
            while file.read(1 << 20):
                pass

    for rep in range(args.repeats + 1):
        # Rotate order to balance early/late slots even when there are >2 variants.
        offset = rep % len(args.variants)
        order = args.variants[offset:] + args.variants[:offset]
        for case in args.cases:
            for variant in order:
                mode, flags = VARIANTS[variant]
                if case.startswith("scan_") and variant == "mask":
                    continue
                if case.startswith("count_") and variant == "prune":
                    continue
                os.environ.update(flags)
                backend = harness.StreamRunner(
                    args.runner, args.fixture, args.workers, 90, mode
                )
                before = harness.prepare_cache(files, "warm")
                result = backend.measure(harness.workloads()[case])
                after = harness.prepare_cache(files, "warm")
                for key in flags:
                    del os.environ[key]
                harness.validate_path(result, mode, case, 128)
                n = {"scan_low": 759841, "scan_high": 9883940}.get(case, 1)
                if (
                    result["build_revision"] != args.revision
                    or result["output_rows"] != n
                ):
                    raise ValueError(
                        f"unexpected build revision or output row count: {result}"
                    )
                expected_bytes = 196691567 if case.startswith("scan_") else 75280626
                if result["bytes_scanned"] != expected_bytes:
                    raise ValueError(
                        f"unexpected bytes scanned: {result['bytes_scanned']}"
                    )
                result["detail"] = validate_metrics(
                    result.pop("runs"),
                    mode == "required" and "MORSEL_NO_METRICS" not in flags,
                    args.workers,
                )
                if args.profile:
                    calls = {key: count for key, count, _, _ in result["phases"]}
                    if calls != {
                        "batch": 2176,
                        "project": 2048,
                        "reader": 272,
                        "range": 128,
                    }:
                        raise ValueError(
                            f"unexpected stock/native decode shape: {calls}"
                        )
                if rep:
                    emit(
                        dict(
                            result,
                            variant=variant,
                            case=case,
                            rep=rep,
                            cache_pages_before=before,
                            cache_pages_after=after,
                        )
                    )


if __name__ == "__main__":
    main()
