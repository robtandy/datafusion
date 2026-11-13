<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# DuckDB plans of the pipeline engine's benchmark queries

The pipeline engine (`datafusion/physical-pipeline`) is modelled on DuckDB's
morsel-driven execution, and the planner work that follows compares its plans
with DuckDB's. This directory holds DuckDB's plans (`EXPLAIN`) of the 47
queries of the engine's benchmark,
`datafusion/physical-pipeline/benches/pipeline_vs_default.rs` (the 22 TPC-H
queries and the 25 ClickBench queries), over the Parquet files that the
benchmark reads, our plans of the same queries, the differ that compares them
(see [The comparison](#the-comparison) and `docs/planner-comparison.md`), and
the scripts that reproduce all of it. Plan shapes only: no timings of DuckDB.

| Path                    | Content                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| `duckdb/<query>.txt`    | the output of DuckDB's `EXPLAIN` of the query `<query>` (`q1` … `q22`, `hits0` … `hits42`)     |
| `duckdb/<query>.json`   | the same plan in JSON (`EXPLAIN (FORMAT JSON)`), which the differ reads                        |
| `pipeline/<query>.json` | our plan of the query: the plan-shape part of the JSON dump of the profile of its execution    |
| `queries.sql`           | the 47 queries, as the benchmark prints them (`PIPELINE_BENCH_MODE=queries`)                   |
| `build_duckdb.sh`       | builds the DuckDB CLI at the commit of the plans                                               |
| `materialize.sh`        | writes the data (`PIPELINE_BENCH_MODE=data`) and `queries.sql` with the benchmark              |
| `capture.py`            | runs `EXPLAIN` for every query in DuckDB and writes `duckdb/`                                  |
| `dump_plans/`           | a Rust program that plans and runs every query with the pipeline engine and writes `pipeline/` |
| `differ.py`             | compares the plans and writes `docs/planner-comparison.md`                                     |
| `test_differ.py`        | the tests of the differ                                                                        |
| `test_measure.py`       | the tests of `measure.py`                                                                      |
| `experiments.sql`       | variants of queries with the shape of DuckDB's plans, for the likely costs of the divergences  |
| `measure.py`            | measures the queries of `experiments.sql` on our engine                                        |
| `data/`                 | the Parquet files that `materialize.sh` writes (ignored by git, see `benchmarks/.gitignore`)   |

## DuckDB

All plans were captured with one build of DuckDB, the head of `main` of
`github.com/duckdb/duckdb`:

- **Commit**: `5d254ad870763a9724af68b937ebf4e1440ad040` (2026-11-12,
  "Release the read-ahead memory reservation when the scan is exhausted
  (#28687)"), the head of `main` when it was fetched on 2026-11-12 at 23:36
  UTC (`git ls-remote https://github.com/duckdb/duckdb refs/heads/main`).
  `build_duckdb.sh` clones it into `~/github/duckdb-plan-compare`. The clone
  `~/github/duckdb` on dduck1 is not used: it is a clone of the fork
  `github.com/robtandy/duckdb`, whose `main` (`e55f794cae`, 2026-10-20) is
  4,192 commits behind, and `~/github/dduck` builds against it, so it is not
  moved to another commit. The source of DuckDB that made these plans is the
  checkout of `build_duckdb.sh`.
- **`PRAGMA version`**:

  | library_version | source_id  | codename            |
  | --------------- | ---------- | ------------------- |
  | v2.1.0-dev93719 | 5d254ad870 | Development Version |

  This is DuckDB 2, not 1.5.x: the history of this commit contains DuckDB
  v2.0.0 (released on 2026-10-20 from the branch `v2.0-cyanoptera`, which
  `main` merged), and `main` has been versioned 2.1 since 2026-09-02, so
  builds of `main` report `v2.1.0-dev<N>`, where `<N>` is the number of
  commits in the history of the checkout: 93,719 (a shallow clone reports 1,
  with the same `source_id`).

- **Build command**, in the DuckDB checkout, with `ninja` on the `PATH`
  (`~/.local/bin/ninja` on dduck1):

  ```bash
  GEN=ninja make release
  ```

  DuckDB's release build, which ran

  ```bash
  mkdir -p ./build/release && cd build/release && cmake -G "Ninja" -DFORCE_COLORED_OUTPUT=1 -DLINK_CORE_EXTENSIONS=OFF -DSTATICALLY_LINK_EXTENSIONS="" -DDUCKDB_CAPABILITIES="" -DLOCAL_EXTENSION_REPO="" -DDUCKDB_EXPLICIT_VERSION="" -DCMAKE_BUILD_TYPE=Release ../.. && cmake --build . --config Release
  ```

  and built the CLI `build/release/duckdb` with the extensions
  `core_functions`, `parquet`, `json` and `icu` linked in, with GCC 13.3.0,
  CMake 3.28.3 and Ninja 1.11.1, in 11 minutes on dduck1 (16-core
  Neoverse-N1, aarch64). `build_duckdb.sh` runs this command (see
  [Reproduce](#reproduce-from-a-clean-workspace)).

## Data

`materialize.sh` writes the data with the benchmark, whose `parquet` source
writes the same files to a temporary directory: the 8 TPC-H tables (the
synthetic data of `datafusion/physical-pipeline/tests/tpch_parity.rs`) and the
synthetic ClickBench `hits` table, at the scale of an optimized build of the
benchmark on 16 threads (TPC-H SF 0.1, 1,000,000 `hits` rows, 16 threads).
The command it runs, from the root of the repository:

```bash
PIPELINE_BENCH_MODE=data PIPELINE_BENCH_DATA_DIR=$PWD/benchmarks/plan-compare/data \
PIPELINE_BENCH_SCALE=0.1 PIPELINE_BENCH_HITS=1000000 PIPELINE_BENCH_THREADS=16 \
cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench pipeline_vs_default -- pipeline_vs_default --exact
```

The output paths are `benchmarks/plan-compare/data/<table>/part-<i>.parquet`,
for `<i>` from 0 to 15: one file per table per thread. Every table is split
into 16 files of about the same number of rows (so 11 of the 16 files of
`region` are empty), written in batches of 8,192 rows, with row groups of at
most 131,072 rows. The data is deterministic: writing it again gives the same
bytes.

| Table      |      Rows | Files |      Bytes |
| ---------- | --------: | ----: | ---------: |
| `region`   |         5 |    16 |     10,964 |
| `nation`   |        25 |    16 |     25,550 |
| `supplier` |     1,000 |    16 |    186,471 |
| `part`     |    20,000 |    16 |  2,362,794 |
| `partsupp` |    80,000 |    16 |  5,923,564 |
| `customer` |    15,000 |    16 |  2,240,822 |
| `orders`   |   150,000 |    16 | 12,629,298 |
| `lineitem` |   601,265 |    16 | 48,026,712 |
| `hits`     | 1,000,000 |    16 | 91,457,958 |

DuckDB reads the columns of the TPC-H tables as `BIGINT` (the keys),
`INTEGER`, `DECIMAL(15,2)` (prices, quantities, discounts, taxes and
balances), `DATE` and `VARCHAR` (the `Utf8View` strings of DataFusion), and
those of `hits` as `BIGINT`, `INTEGER`, `SMALLINT`, `DATE` (`EventDate`) and
`VARCHAR`; `EventTime` is a `BIGINT` of seconds, as in ClickBench's
`hits.parquet`.

The benchmark runs both engines over these files with

```bash
PIPELINE_BENCH_SOURCE=parquet PIPELINE_BENCH_DATA_DIR=$PWD/benchmarks/plan-compare/data \
PIPELINE_BENCH_THREADS=16 \
cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench pipeline_vs_default -- pipeline_vs_default --exact
```

(which writes the same files again first).

## Queries

`materialize.sh` also writes `queries.sql`, the output of
`PIPELINE_BENCH_MODE=queries` without the lines of libtest: every query on the
lines after a line with its name (`-- q1`), terminated by `;`. `q1` … `q22`
are the TPC-H queries of the parity suite (those of DataFusion's TPC-H
benchmark, Q15 with a `WITH` clause), and `hits0` … `hits42` the ClickBench
queries of the benchmark (named after their 0-based ClickBench number, with
tie-breakers in their `ORDER BY`s and fewer sums in `hits29`).

DuckDB runs 46 of the 47 queries unchanged. One needs a dialect adjustment,
which `capture.py` applies (see `ADJUSTMENTS`):

| Query    | In the benchmark                    | In DuckDB                               | Why                                                                                                                                                                                                    |
| -------- | ----------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `hits42` | `to_timestamp_seconds("EventTime")` | `make_timestamp("EventTime" * 1000000)` | DuckDB has no `to_timestamp_seconds`; `make_timestamp(BIGINT)` makes a `TIMESTAMP` (without time zone, like DataFusion's `Timestamp(Second, None)`) of microseconds since the epoch. Both occurrences. |

## Capture

```bash
benchmarks/plan-compare/capture.py --duckdb ~/github/duckdb-plan-compare/build/release/duckdb
```

(`--duckdb` defaults to that path.) `capture.py` first checks that the CLI is
built from the commit above (`PRAGMA version`) and that the data directory
holds the data above (16 files and the rows of every table), and then runs,
for every query of `queries.sql`, the DuckDB CLI on an in-memory database in
the data directory, `duckdb -batch -bail -no-init -no-agent`, with the script

```sql
SET threads = 16;
CREATE VIEW region AS SELECT * FROM read_parquet('region/*.parquet');
-- ... a view for every table: nation, supplier, part, partsupp, customer,
-- orders, lineitem, hits
EXPLAIN <query>;
```

and writes its output to `duckdb/<query>.txt`, and that of the same script
with `EXPLAIN (FORMAT JSON) <query>;` to `duckdb/<query>.json` (and removes the
plans of queries that are no longer in `queries.sql`). It takes 10 seconds.

- The output is DuckDB's physical plan (`explain_output` is `physical_only` by
  default), as the CLI prints it when its output is not a terminal: every
  operator, with all its details (no operator is folded and no detail is
  truncated), the root at the top. `~<N> rows` are DuckDB's estimated
  cardinalities. The paths of the files are relative to the data directory,
  so the plans do not depend on where the data is.
- `-no-init` skips `~/.duckdbrc`, and `-no-agent` the CLI's "agent mode",
  which it turns on when the environment variable of an AI coding agent (such
  as `AI_AGENT`) is set and its output is not a terminal, and in which it
  prints compact plans instead.
- Running the capture again reproduces the plans byte for byte (with the same
  build and data, DuckDB's estimates do not change either), and DuckDB plans
  these queries the same on 1 thread as on 16.
- The JSON plan is the same tree as the text plan, rendered by DuckDB's JSON
  tree renderer: every operator is an object with its `name` (`READ_PARQUET`,
  `HASH_JOIN`, ...), its `children` (the right child of a join is its build
  side) and its `extra_info`, the details of the text plan by name, a list for
  the details of several lines (e.g. one item per projection, filter or join
  condition), with `Estimated Cardinality`.

## Our plans

`dump_plans/` is a Rust program (a package of its own, outside of the
workspace of the repository, with the versions of its `Cargo.lock`) that
plans and runs every query of `queries.sql` with the pipeline engine over the
same Parquet files, as the benchmark's `parquet` source does (the session
configuration of the benchmark: 16 target partitions, `PipelineQueryPlanner`,
trusting DataFusion's built-in scalar and aggregate functions), with profiling
on, and writes the JSON dump of the profile of the execution (the schema of
`datafusion/physical-pipeline/docs/profiling.md`) to `pipeline/<query>.json`:

```bash
cargo run --locked --manifest-path benchmarks/plan-compare/dump_plans/Cargo.toml
```

(`--data-dir`, `--queries` and `--out` default to `data/`, `queries.sql` and
`pipeline/`, `--threads` to 16.) It writes the plan-shape part of the dump: its
`schema`, `version` and `root`, the `id`, `kind`, `name`, `inputs` and
`pipelines` of every operator and its metrics `rows_in` and `rows_out` (the
rows it consumed and produced), and the `id`, `meta_pipeline`, `source`,
`operators`, `sink`, `dependencies` and `meta_dependencies` of every pipeline,
and of the meta pipelines; `--full` writes the whole dump, with all the
timings. The paths of the files in the descriptions of the scans are relative
to the data directory, as in DuckDB's plans. Running it again reproduces the
plans, but for the rows of the operators below dynamic filters and the
thresholds of the dynamic filters of TopKs in the descriptions of the scans,
which depend on the order in which the threads read their morsels. A build in
the `dev` profile writes the plans of the 47 queries in 10 seconds.

## The comparison

`differ.py` reads both plans of every query from their machine-readable form
(DuckDB's `duckdb/<query>.json` and our `pipeline/<query>.json`), normalizes
them into one fingerprint (for every scan the columns it reads and the filters
pushed into it; the join order, the join operators and their build sides; the
locations of the aggregations, top-N, sorts and limits; and the pipeline
breaks, at the locations of their operators), compares the fingerprints,
groups the differences of every query by their cause into its divergences, and
classifies every divergence as a rule gap (our optimizer lacks a rule that
DuckDB has) or as cost/statistics-driven (the rules agree, the choice differs).
It compares the filters of the scans by their meaning (normalized terms of an
expression, an operator and its constants, which the filters of the other scan
must imply: a missing or changed bound is a divergence, the same filter in the
other dialect is not), and the placement of operators by their locations (the
skeleton of the plan below them, with blocks of joins as the multiset of their
inputs, and the operators above them: an operator or a pipeline break at
another place is a divergence, also with as many of every kind). See "How the
plans are compared" in the report. It prints the verdict of every query
(matches, rule gap or cost/statistics-driven) and the summary of
`docs/planner-comparison.md`, which it writes with `--report`:

```bash
benchmarks/plan-compare/differ.py            # verdicts and summary
benchmarks/plan-compare/differ.py --verbose  # and the findings of every query
benchmarks/plan-compare/differ.py --report   # and write docs/planner-comparison.md
benchmarks/plan-compare/differ.py --check    # fail if it is not up to date
python3 -m unittest discover -s benchmarks/plan-compare -p 'test_*.py'
```

It needs Python 3.9 or later and nothing else, and fails if a divergence has
no cause (a new kind of divergence, to classify in `CAUSES`), or no note on
what it likely costs (`NOTES`). The report documents the normalization, the
causes and what the comparison leaves out.

The likely costs of the notes are measured on our engine where SQL can give a
query the shape of DuckDB's plan (`experiments.sql`: DuckDB's join order
written out with explicit joins, the common table expression of q15
`MATERIALIZED`, and the top-N of hits23 over its sort keys only):

```bash
benchmarks/plan-compare/measure.py
```

builds `dump_plans` in the `release` profile (and runs the executable that
cargo reports, wherever its target directory is, e.g. `CARGO_TARGET_DIR`) and
runs it 6 times on 6 rounds of all queries of `experiments.sql` (with the
PostgreSQL dialect, which parses `MATERIALIZED` and plans the benchmark's
queries as the default dialect), and prints the medians (of 36 executions,
those that the notes quote) of the elapsed time of every query and of the self
time of its operators, from their profiles (so with profiling on). These are
timings of our engine only, on 16 threads; on a busy machine, the differences
of less than 3% are noise.

## Reproduce from a clean workspace

Needs git, CMake, Ninja, a C++ compiler, Python 3 and the Rust toolchain of the
repository. From the root of a checkout of this repository:

```bash
# 1. Build the DuckDB CLI at the commit of the plans (11 minutes on 16
#    cores). If the directory does not exist, clones the history of `main` of
#    github.com/duckdb/duckdb into it (without the files of other commits)
#    and checks out the commit; an existing checkout must be at the commit.
benchmarks/plan-compare/build_duckdb.sh ~/github/duckdb-plan-compare

# 2. Write the data to benchmarks/plan-compare/data and the queries to
#    benchmarks/plan-compare/queries.sql (builds the benchmark with the
#    release-nonlto profile).
benchmarks/plan-compare/materialize.sh

# 3. Capture the plans in benchmarks/plan-compare/duckdb.
benchmarks/plan-compare/capture.py --duckdb ~/github/duckdb-plan-compare/build/release/duckdb

# 4. Dump our plans in benchmarks/plan-compare/pipeline.
cargo run --locked --manifest-path benchmarks/plan-compare/dump_plans/Cargo.toml

# 5. Compare them, and write docs/planner-comparison.md.
benchmarks/plan-compare/differ.py --report

# Nothing changed: the queries and the plans are reproduced (but for the rows
# below dynamic filters in our plans, see "Our plans", and so in the excerpts
# of the report), and so are the verdicts and the summary.
git status --short benchmarks/plan-compare docs/planner-comparison.md
```

To capture the plans of another commit of DuckDB, change `DUCKDB_COMMIT` in
`build_duckdb.sh` and `capture.py`, build it and capture again. To capture the
plans over other data (other `PIPELINE_BENCH_SCALE`, `PIPELINE_BENCH_HITS` or
`PIPELINE_BENCH_THREADS`), write it to another directory with
`materialize.sh <dir>` and capture with
`capture.py --data-dir <dir> --out <plans> --threads <threads> --force`
(`materialize.sh` writes `queries.sql` again, with the same queries).
