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
benchmark reads, and the scripts that reproduce them. These are fixtures only:
plan shapes, no timings and no comparison.

| Path                 | Content                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `duckdb/<query>.txt` | the output of DuckDB's `EXPLAIN` of the query `<query>` (`q1` … `q22`, `hits0` … `hits42`)   |
| `queries.sql`        | the 47 queries, as the benchmark prints them (`PIPELINE_BENCH_MODE=queries`)                 |
| `build_duckdb.sh`    | builds the DuckDB CLI at the commit of the plans                                             |
| `materialize.sh`     | writes the data (`PIPELINE_BENCH_MODE=data`) and `queries.sql` with the benchmark            |
| `capture.py`         | runs `EXPLAIN` for every query in DuckDB and writes `duckdb/`                                |
| `data/`              | the Parquet files that `materialize.sh` writes (ignored by git, see `benchmarks/.gitignore`) |

## DuckDB

All plans were captured with one build of DuckDB, the head of `main`:

- **Commit**: `e55f794cae98443cfaf302f41ce8a83a95a7872a` (2026-10-20, "Test:
  replace deprecated `__TEST_DIR__` placeholder with `{TEST_DIR}` (#27533)").
  It is the head of `main` of `~/github/duckdb` on dduck1 (a shallow clone of
  `github.com/robtandy/duckdb`, a fork of `github.com/duckdb/duckdb`) after
  `git fetch origin main` on 2026-11-12, and a commit of the `main` of
  `github.com/duckdb/duckdb`. `~/github/duckdb` is the DuckDB source that
  the work on the pipeline engine reads.
- **`PRAGMA version`**:

  | library_version | source_id  | codename            |
  | --------------- | ---------- | ------------------- |
  | v2.1.0-dev1     | e55f794cae | Development Version |

  This is DuckDB 2, not 1.5.x: DuckDB releases 2.0 from the branch
  `v2.0-cyanoptera` (where it tagged v2.0.0 on 2026-10-20), and `main` has
  been versioned 2.1 since 2026-09-02, so builds of `main` report
  `v2.1.0-dev<N>`. This commit of `main` contains the release branch up to
  2026-10-16 (their merge base; v2.0.0 has 108 more commits) and 1,021
  commits that v2.0.0 does not. `<N>` is the number of commits in the history
  of the checkout, 1 in a shallow clone; a full clone reports a larger `<N>`
  with the same `source_id`.

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
  CMake 3.28.3 and Ninja 1.11.1, in 12 minutes on dduck1 (16-core
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
benchmarks/plan-compare/capture.py --duckdb ~/github/duckdb/build/release/duckdb
```

`capture.py` first checks that the CLI is built from the commit above
(`PRAGMA version`) and that the data directory holds the data above (16 files
and the rows of every table), and then runs, for every query of
`queries.sql`, the DuckDB CLI on an in-memory database in the data directory,
`duckdb -batch -bail -no-init -no-agent`, with the script

```sql
SET threads = 16;
CREATE VIEW region AS SELECT * FROM read_parquet('region/*.parquet');
-- ... a view for every table: nation, supplier, part, partsupp, customer,
-- orders, lineitem, hits
EXPLAIN <query>;
```

and writes its output to `duckdb/<query>.txt` (and removes the plans of
queries that are no longer in `queries.sql`). It takes 5 seconds.

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

## Reproduce from a clean workspace

Needs git, CMake, Ninja, a C++ compiler, Python 3 and the Rust toolchain of the
repository. From the root of a checkout of this repository:

```bash
# 1. Build the DuckDB CLI at the commit of the plans (12 minutes on 16 cores).
#    Clones it to the directory if the directory does not exist; an existing
#    checkout must be at the commit already.
benchmarks/plan-compare/build_duckdb.sh ~/github/duckdb

# 2. Write the data to benchmarks/plan-compare/data and the queries to
#    benchmarks/plan-compare/queries.sql (builds the benchmark with the
#    release-nonlto profile).
benchmarks/plan-compare/materialize.sh

# 3. Capture the plans in benchmarks/plan-compare/duckdb.
benchmarks/plan-compare/capture.py --duckdb ~/github/duckdb/build/release/duckdb

# Nothing changed: the queries and plans are reproduced.
git status --short benchmarks/plan-compare
```

To capture the plans of another commit of DuckDB, change `DUCKDB_COMMIT` in
`build_duckdb.sh` and `capture.py`, build it and capture again. To capture the
plans over other data (other `PIPELINE_BENCH_SCALE`, `PIPELINE_BENCH_HITS` or
`PIPELINE_BENCH_THREADS`), write it to another directory with
`materialize.sh <dir>` and capture with
`capture.py --data-dir <dir> --out <plans> --threads <threads> --force`
(`materialize.sh` writes `queries.sql` again, with the same queries).
