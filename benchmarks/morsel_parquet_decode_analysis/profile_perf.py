# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Capture Arm PMUv3 counters and leaf profiles for #155 on warm Linux files."""

import json
import os
import subprocess
import sys
from pathlib import Path

from measure import harness

VARIANTS = [
    ("stock", "off", {}),
    ("native", "required", {}),
    ("row_group_yield", "required", {"MORSEL_SCRATCH_NO_YIELD": "1"}),
    ("producer8", "required", {"MORSEL_SCRATCH_PRODUCER": "1"}),
    (
        "both_queues",
        "required",
        {"MORSEL_SCRATCH_PRODUCER": "1", "MORSEL_SCRATCH_OUTPUT": "1"},
    ),
]
CASES = ("scan_high", "count_high")


def main(work, perf):
    fixture = work / "large"
    binary = work / "ablations"
    files = harness.fixture_files(fixture)
    env = {
        key: val
        for key, val in os.environ.items()
        if not key.startswith("MORSEL_SCRATCH_") and key != "MORSEL_NO_METRICS"
    }

    def invocation(case, mode):
        return [
            str(binary),
            "measure",
            mode,
            "8",
            harness.fixture_ddl(fixture),
            harness.workloads()[case],
        ]

    def run(command, case, mode, flags):
        harness.prepare_cache(files, "warm")
        result = json.loads(
            subprocess.check_output(
                [perf, *command, "--", *invocation(case, mode)],
                env={**env, **flags},
                timeout=90,
            )
        )
        harness.validate_path(result, mode, case, 128)
        harness.prepare_cache(files, "warm")
        return result

    for rep in range(1, 6):
        offset = rep % len(VARIANTS)
        for case in CASES:
            for name, mode, flags in VARIANTS[offset:] + VARIANTS[:offset]:
                stem = work / f"{case}.{name}.{rep}"
                result = run(
                    [
                        "stat",
                        "-x,",
                        "-o",
                        f"{stem}.stat",
                        "-e",
                        "cycles:u,instructions:u,r17:u,task-clock,context-switches,cpu-migrations",
                    ],
                    case,
                    mode,
                    flags,
                )
                print(
                    json.dumps(
                        dict(
                            kind="stat", case=case, variant=name, rep=rep, result=result
                        )
                    ),
                    flush=True,
                )
    for case in CASES:
        for name, mode, flags in VARIANTS[:2]:
            stem = work / f"{case}.{name}"
            result = run(
                ["record", "-q", "-F", "3999", "-e", "cycles:u", "-o", f"{stem}.perf"],
                case,
                mode,
                flags,
            )
            with open(f"{stem}.report", "w") as report:
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
                        f"{stem}.perf",
                    ],
                    stdout=report,
                    check=True,
                    timeout=60,
                )
            print(
                json.dumps(dict(kind="record", case=case, variant=name, result=result)),
                flush=True,
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
