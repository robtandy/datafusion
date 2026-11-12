#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Capture DuckDB's plans of the queries of the pipeline engine's benchmark.

Runs `EXPLAIN` for every query of `queries.sql` (written by `materialize.sh`)
in the DuckDB CLI built by `build_duckdb.sh`, over views of the Parquet files
that `materialize.sh` wrote, and writes the output of every query to
`duckdb/<query>.txt`. Checks first that the CLI is built from the commit of the
captured plans and that the data is that of the captured plans (`--force`
skips these checks). See README.md.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

#: The commit of DuckDB whose plans are captured, see build_duckdb.sh.
DUCKDB_COMMIT = "e55f794cae98443cfaf302f41ce8a83a95a7872a"

#: The tables of the benchmark: a directory of Parquet files in the data
#: directory, and a view over them in DuckDB.
TABLES = (
    "region",
    "nation",
    "supplier",
    "part",
    "partsupp",
    "customer",
    "orders",
    "lineitem",
    "hits",
)

#: The rows of the tables of the captured data, TPC-H SF 0.1 and 1,000,000
#: `hits` (see materialize.sh).
EXPECTED_ROWS = {
    "region": 5,
    "nation": 25,
    "supplier": 1_000,
    "part": 20_000,
    "partsupp": 80_000,
    "customer": 15_000,
    "orders": 150_000,
    "lineitem": 601_265,
    "hits": 1_000_000,
}

#: The Parquet files of every table of the captured data, one per thread.
EXPECTED_FILES = 16

#: The dialect adjustments of the queries for DuckDB: (query, text, the text
#: for DuckDB); every occurrence of the text is replaced. See README.md.
ADJUSTMENTS = (
    # DuckDB has no `to_timestamp_seconds`. `make_timestamp(BIGINT)` makes a
    # `TIMESTAMP` (without time zone, as the `Timestamp(Second, None)` of
    # DataFusion's `to_timestamp_seconds`) of microseconds since the epoch.
    (
        "hits42",
        'to_timestamp_seconds("EventTime")',
        'make_timestamp("EventTime" * 1000000)',
    ),
)


def parse_queries(script: str) -> list[tuple[str, str]]:
    """The (name, SQL) of the queries of `queries.sql`: every query is on the
    lines after a line `-- <name>`, up to the line that ends with `;` (none of
    the queries contains a `;`), as `PIPELINE_BENCH_MODE=queries` prints
    them."""
    queries = []
    name = None
    sql: list[str] = []
    for line in script.splitlines():
        if name is None:
            if line.startswith("-- "):
                name = line[len("-- ") :].strip()
                sql = []
            elif line.strip():
                raise SystemExit(f"error: unexpected line {line!r} in the queries")
        elif line.endswith(";"):
            sql.append(line[:-1])
            queries.append((name, "\n".join(sql)))
            name = None
        else:
            sql.append(line)
    if name is not None:
        raise SystemExit(f"error: query {name} does not end with `;`")
    return queries


def adjust(name: str, sql: str) -> str:
    """The SQL of query `name` for DuckDB, with its ADJUSTMENTS."""
    for query, text, replacement in ADJUSTMENTS:
        if query != name:
            continue
        if text not in sql:
            raise SystemExit(f"error: {name} has no {text!r} to adjust")
        sql = sql.replace(text, replacement)
    return sql


class DuckDB:
    """The DuckDB CLI, run in the data directory (so that the paths of the
    Parquet files are relative to it) on an in-memory database."""

    def __init__(self, path: pathlib.Path, data_dir: pathlib.Path) -> None:
        self.path = path
        self.data_dir = data_dir

    def run(self, script: str, *options: str) -> str:
        """The output of `script`: fails if DuckDB fails or reports anything
        on stderr. The options make the output independent of the
        environment: batch mode, no `~/.duckdbrc`, and no "agent mode" (which
        DuckDB turns on when the environment variable of a coding agent is
        set, and which renders compact plans)."""
        result = subprocess.run(
            [str(self.path), "-batch", "-bail", "-no-init", "-no-agent", *options],
            input=script,
            capture_output=True,
            encoding="utf-8",
            cwd=self.data_dir,
            check=False,
        )
        if result.returncode != 0 or result.stderr:
            raise SystemExit(
                f"error: DuckDB failed (exit status {result.returncode}):\n"
                f"{result.stderr}\nin\n{script}"
            )
        return result.stdout

    def version(self) -> tuple[str, str]:
        """The library version and the source id (commit) of `PRAGMA version`."""
        output = self.run(
            "SELECT library_version, source_id FROM pragma_version();",
            "-list",
            "-noheader",
        )
        version, source_id = output.strip().split("|")
        return version, source_id

    def tables(self) -> dict[str, tuple[int, int]]:
        """The Parquet files and rows of every table."""
        script = (
            "\nUNION ALL\n".join(
                f"SELECT '{table}', "
                f"(SELECT count(*) FROM glob('{table}/*.parquet')), "
                f"(SELECT count(*) FROM read_parquet('{table}/*.parquet'))"
                for table in TABLES
            )
            + ";"
        )
        tables = {}
        for line in self.run(script, "-list", "-noheader").splitlines():
            table, files, rows = line.split("|")
            tables[table] = (int(files), int(rows))
        return tables

    def explain(self, sql: str, threads: int) -> str:
        """The output of `EXPLAIN sql` on `threads` threads, over a view of
        the Parquet files of every table."""
        script = [f"SET threads = {threads};"]
        script += [
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{table}/*.parquet');"
            for table in TABLES
        ]
        script.append(f"EXPLAIN {sql};")
        return self.run("\n".join(script) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duckdb",
        type=pathlib.Path,
        default=os.environ.get(
            "DUCKDB", pathlib.Path.home() / "github/duckdb/build/release/duckdb"
        ),
        help="the DuckDB CLI (default: $DUCKDB or ~/github/duckdb/build/release/duckdb)",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=HERE / "data",
        help="the data directory of materialize.sh (default: %(default)s)",
    )
    parser.add_argument(
        "--queries",
        type=pathlib.Path,
        default=HERE / "queries.sql",
        help="the queries of materialize.sh (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=HERE / "duckdb",
        help="the directory of the plans (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=EXPECTED_FILES,
        help="DuckDB's threads (default: %(default)s, the threads of the data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="capture with another DuckDB build or other data",
    )
    args = parser.parse_args()

    duckdb = DuckDB(args.duckdb.expanduser().resolve(), args.data_dir.resolve())
    version, source_id = duckdb.version()
    print(f"DuckDB {version} ({source_id}) at {duckdb.path}")
    problems = []
    if not DUCKDB_COMMIT.startswith(source_id):
        problems.append(f"DuckDB is built from {source_id}, not from {DUCKDB_COMMIT}")
    print(f"{'table':<9} {'files':>5} {'rows':>9}  (in {duckdb.data_dir})")
    for table, (files, rows) in duckdb.tables().items():
        print(f"{table:<9} {files:>5} {rows:>9}")
        if (files, rows) != (EXPECTED_FILES, EXPECTED_ROWS[table]):
            problems.append(
                f"{table} has {files} files of {rows} rows, not "
                f"{EXPECTED_FILES} files of {EXPECTED_ROWS[table]} rows"
            )
    if problems and not args.force:
        problems = "\n".join(f"  {problem}" for problem in problems)
        sys.exit(f"error: not the DuckDB build or the data of the plans:\n{problems}")

    queries = parse_queries(args.queries.read_text(encoding="utf-8"))
    names = [name for name, _ in queries]
    unknown = {query for query, _, _ in ADJUSTMENTS} - set(names)
    if unknown:
        sys.exit(f"error: adjustments of unknown queries {sorted(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, sql in queries:
        plan = duckdb.explain(adjust(name, sql), args.threads)
        (args.out / f"{name}.txt").write_text(plan, encoding="utf-8", newline="\n")
    # Remove the plans of queries that are no longer in the list.
    for plan in args.out.glob("*.txt"):
        if plan.stem not in names:
            plan.unlink()
    print(f"Wrote the plans of {len(queries)} queries to {args.out}")


if __name__ == "__main__":
    main()
