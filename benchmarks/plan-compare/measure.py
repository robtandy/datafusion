#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Measure the queries of `experiments.sql` on the pipeline engine: our
engine only (no timings of DuckDB).

`experiments.sql` has benchmark queries next to variants that give them the
shape of DuckDB's plan, as far as SQL can (DuckDB's join order written out
with explicit joins, a common table expression `MATERIALIZED`, a top-N that
reads its sort keys only), to estimate what the divergences of
docs/planner-comparison.md likely cost. Runs `dump_plans --full` (an optimized
build, with profiling on) `--runs` times, each time on `--rounds` rounds of all
queries (so that every process runs every query several times, interleaved
with the others), and prints, for every query, the medians of its elapsed time
and of the self time of its operators, from the dumps of its profiles. See
README.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture import parse_queries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=pathlib.Path,
        default=HERE / "experiments.sql",
        help="the queries (default: %(default)s)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="the runs (default: %(default)s)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="the rounds of all queries of a run (default: %(default)s)",
    )
    args = parser.parse_args()
    manifest = HERE / "dump_plans" / "Cargo.toml"
    subprocess.run(
        ["cargo", "build", "--locked", "--release", "--manifest-path", str(manifest)],
        check=True,
    )
    binary = HERE / "dump_plans" / "target" / "release" / "plan-compare-dump-plans"
    queries = parse_queries(args.queries.read_text(encoding="utf-8"))
    names = [name for name, _ in queries]
    elapsed: dict[str, list[float]] = {name: [] for name in names}
    operators: dict[str, list[float]] = {name: [] for name in names}
    with tempfile.TemporaryDirectory() as out:
        rounds = pathlib.Path(out) / "rounds.sql"
        rounds.write_text(
            "\n".join(
                f"-- {name}__{round_}\n{sql};\n"
                for round_ in range(args.rounds)
                for name, sql in queries
            ),
            encoding="utf-8",
        )
        for _ in range(args.runs):
            # `MATERIALIZED` common table expressions need the PostgreSQL
            # dialect, which plans the benchmark's queries as the default.
            subprocess.run(
                [
                    str(binary),
                    "--full",
                    "--set",
                    "datafusion.sql_parser.dialect=PostgreSQL",
                    "--queries",
                    str(rounds),
                    "--out",
                    out,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            for name in names:
                for round_ in range(args.rounds):
                    path = pathlib.Path(out) / f"{name}__{round_}.json"
                    dump = json.loads(path.read_text(encoding="utf-8"))
                    metrics = dump["executor"]["metrics"]
                    elapsed[name].append(metrics["elapsed_time"]["value"] / 1e6)
                    operators[name].append(
                        sum(
                            op["metrics"]["self_time"]["value"]
                            for op in dump["operators"]
                        )
                        / 1e6
                    )
    print(f"{'query':<24} {'elapsed (ms)':>12} {'operators (ms)':>14}")
    for name in names:
        print(
            f"{name:<24} {statistics.median(elapsed[name]):>12.2f} "
            f"{statistics.median(operators[name]):>14.2f}"
        )


if __name__ == "__main__":
    main()
