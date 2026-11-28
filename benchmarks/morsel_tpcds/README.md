<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel TPC-DS 99-Query SF=1 4-Way Parquet Benchmark Suite

This fork-only benchmark suite compares **Morsel DataFusion**
(`morsel_mode=required`), **Stock DataFusion** (`stock_default` and
`stock_pushdown`), **DuckDB 2.0.1**, and **ClickHouse (`clickhouse-local`)**
across all **99 standard TPC-DS queries**
(`benchmarks/sql_benchmarks/tpcds/benchmarks/q01.benchmark`–`q99.benchmark`) on
identical deterministic SF=1 Parquet files (#231). It lives strictly under
`benchmarks/morsel_tpcds/` and follows the detached-example build pattern from
`benchmarks/morsel_tpch/` with zero edits to workspace `Cargo.toml` or
production crates.

## Dataset Generation and Layout

`harness.py generate` uses DuckDB's built-in `tpcds` extension
(`SET threads = 1; CALL dsdgen(sf=...)`) and exports all 24 standard TPC-DS
tables deterministically ordered by each table's primary key into multi-file
Parquet directories (`<data_dir>/<table>/part-000000.parquet`, ...). Keep all
generated Parquet files and `.jsonl` run outputs outside Git (e.g., under
`$HOME/.cache/`).

Writer settings:

- Parquet format version `2.6`, data page version `1.0`
- Compression: Zstandard (`zstd`), level `3`
- Dictionary encoding: enabled (`use_dictionary=True`)
- Column min/max statistics: enabled (`write_statistics=True`)
- Default row-group size: `131,072` rows (`16 × 8,192` DataFusion batch size,
  configurable via `--row-group-size`)

Two primary scale modes are supported (or any positive `--sf`):

- **Smoke mode (`--sf 0.01`):** 278,020 total rows across 24 tables / 98 files
  / 98 row groups (7,542,834 bytes / 7.2 MiB on disk); generates in ~1 second
  and is used by `test_harness.py` to verify byte-for-byte determinism, query
  parsing, 4-way execution, parity checks on admitted queries, Morsel rejection
  logging, and ClickHouse dialect error recording.
- **SF1 benchmark mode (`--sf 1`):** 19,557,375 total rows across 24 tables /
  98 files / 206 row groups (458,185,563 bytes / 437.0 MiB on disk).

### SF1 Dataset Layout (`--sf 1 --row-group-size 131072`)

| Table                    | Primary key ordering                         |  Files |           Rows | Row groups |   Parquet bytes |
| ------------------------ | -------------------------------------------- | -----: | -------------: | ---------: | --------------: |
| `inventory`              | `inv_date_sk, inv_item_sk, inv_warehouse_sk` |     16 |     11,745,000 |         96 |      29,822,248 |
| `store_sales`            | `ss_item_sk, ss_ticket_number`               |     16 |      2,880,404 |         32 |     172,978,255 |
| `catalog_sales`          | `cs_item_sk, cs_order_number`                |     16 |      1,441,548 |         16 |     137,846,205 |
| `web_sales`              | `ws_item_sk, ws_order_number`                |     16 |        719,384 |         16 |      70,463,104 |
| `customer_demographics`  | `cd_demo_sk`                                 |      4 |      1,920,800 |         16 |       5,854,178 |
| `store_returns`          | `sr_item_sk, sr_ticket_number`               |      4 |        287,514 |          4 |      15,488,437 |
| `catalog_returns`        | `cr_item_sk, cr_order_number`                |      4 |        144,067 |          4 |      11,167,490 |
| `customer`               | `c_customer_sk`                              |      2 |        100,000 |          2 |       4,503,577 |
| `time_dim`               | `t_time_sk`                                  |      2 |         86,400 |          2 |         795,727 |
| `date_dim`               | `d_date_sk`                                  |      2 |         73,049 |          2 |       1,327,863 |
| `web_returns`            | `wr_item_sk, wr_order_number`                |      2 |         71,763 |          2 |       5,396,941 |
| `customer_address`       | `ca_address_sk`                              |      2 |         50,000 |          2 |         887,282 |
| `item`                   | `i_item_sk`                                  |      1 |         18,000 |          1 |       1,132,141 |
| `catalog_page`           | `cp_catalog_page_sk`                         |      1 |         11,718 |          1 |         429,284 |
| `household_demographics` | `hd_demo_sk`                                 |      1 |          7,200 |          1 |          27,094 |
| `promotion`              | `p_promo_sk`                                 |      1 |            300 |          1 |          17,244 |
| `reason`                 | `r_reason_sk`                                |      1 |             75 |          1 |           2,001 |
| `web_page`               | `wp_web_page_sk`                             |      1 |             60 |          1 |           5,711 |
| `web_site`               | `web_site_sk`                                |      1 |             30 |          1 |          11,221 |
| `income_band`            | `ib_income_band_sk`                          |      1 |             20 |          1 |           1,433 |
| `ship_mode`              | `sm_ship_mode_sk`                            |      1 |             20 |          1 |           2,664 |
| `store`                  | `s_store_sk`                                 |      1 |             12 |          1 |          10,246 |
| `call_center`            | `cc_call_center_sk`                          |      1 |              6 |          1 |          10,599 |
| `warehouse`              | `w_warehouse_sk`                             |      1 |              5 |          1 |           4,618 |
| **Total**                |                                              | **98** | **19,557,375** |    **206** | **458,185,563** |

Large fact tables (`inventory`, `store_sales`, `catalog_sales`, `web_sales`)
are sharded into 16 files each so both 8-worker Stock DataFusion and Morsel's
native Parquet source (`repartition_file_scans=false`) get full multi-file
parallelism without tiny tail row groups.

## Query Extraction and Cross-Engine Notes

All 99 queries (`q01`–`q99`) are parsed directly from
`benchmarks/sql_benchmarks/tpcds/benchmarks/q01.benchmark`–`q99.benchmark`,
resolving each file's `run ../datafusion/core/tests/tpc-ds/<N>.sql` directive
and extracting only the tables referenced by each query.

- **Multi-statement queries (`q14`, `q23`, `q24`, `q39`):** Four standard
  TPC-DS query files contain two independent `SELECT` statements separated by
  `;`. `harness.py` executes both statements sequentially in a single session
  for timing (`measure`), verifies exact parity on every statement individually
  (`query`), and marks a query as Morsel-admitted only if all of its statements
  plan natively under `morsel_mode=required`.
- **DuckDB 2.0.1 `q90` alias quoting:** `90.sql` uses an unquoted derived-table
  alias `) at,`, which DuckDB 2.0.1 rejects as a reserved keyword. `harness.py`
  applies a targeted single-query rewrite `) at,` → `) "at",` for DuckDB only;
  DataFusion and ClickHouse execute the unmodified SQL.
- **DataFusion vs. DuckDB integer division (`q21`, `q78`, `q83`):** DataFusion
  follows PostgreSQL truncating integer division (`INT / INT`), whereas DuckDB
  promotes `INT / INT` to `DOUBLE`. On `q21`, `q78`, and `q83`, `harness.py`
  enforces exact `morsel_pushdown == stock_default` and
  `stock_pushdown == stock_default` parity while documenting the cross-engine
  division semantics difference. On all other **96/99** queries, Stock
  DataFusion matches DuckDB 2.0.1 exactly.
- **ClickHouse (`clickhouse-local`) standard-SQL settings:** ClickHouse runs
  the unmodified SQL with standard-SQL compatibility flags
  (`--join_use_nulls=1 --group_by_use_nulls=1 --joined_subquery_requires_alias=0
--enable_analyzer=1 --allow_experimental_correlated_subqueries=1
--union_default_mode=DISTINCT --intersect_default_mode=DISTINCT
--except_default_mode=DISTINCT --cast_keep_nullable=1`). Any remaining
  ClickHouse dialect incompatibilities (14 queries) are recorded verbatim in
  the JSONL header and summary report without silently skipping.

## Controlled Arms, Timing Boundary, and Parity

Each timed arm runs in a fresh process per repetition with pinned CPU affinity
(`taskset -c 0-7`, 8 threads, batch size `8,192`) and alternating arm order
across odd and even rounds.

| Arm                  | Engine       | `morsel_mode` | Settings                                                                                                               |
| -------------------- | ------------ | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `stock_default`      | DataFusion   | `off`         | DataFusion defaults (`pushdown_filters=false`, `pruning=true`, `repartition_file_scans=true`)                          |
| `stock_pushdown`     | DataFusion   | `off`         | DataFusion defaults + `pushdown_filters=true`, `reorder_filters=true`                                                  |
| `morsel_pushdown`    | DataFusion   | `required`    | `pushdown_filters=true`, `reorder_filters=true`, `repartition_file_scans=false`, `pruning=true`                        |
| `duckdb_default`     | DuckDB 2.0.1 | —             | `SET threads = 8`, lazy `read_parquet(...)` views, drained via `to_arrow_reader(8192)`                                 |
| `clickhouse_default` | ClickHouse   | —             | `clickhouse local --max_threads=8` + standard-SQL flags, `CREATE VIEW ... file(..., 'Parquet')` + `FORMAT Null --time` |

- **Matching timing boundary:** DataFusion starts the timer right before
  `register_parquet` for only the tables referenced by the query and drains
  `execute_stream` without serializing output. DuckDB starts the timer before
  draining `to_arrow_reader(8192)` over lazy `read_parquet(...)` views (which
  reads Parquet footers during query execution). ClickHouse reports
  `register_seconds` (sum of `CREATE VIEW` footer extraction times) +
  `exec_seconds` (sum of `SELECT ... FORMAT Null` times via `--time`),
  excluding binary startup overhead while including footer extraction,
  planning, and execution drain.
- **Warm page cache (`--cache warm`):** Pre-warms all Parquet files into the
  Linux page cache and asserts 100% 4 KiB page residency via `mincore` before
  and after every measured trial.
- **Parity verification:** Transfers lossless Arrow IPC streams (`StreamWriter`
  in Rust, `to_arrow_table()` in DuckDB, `FORMAT ArrowStream` in ClickHouse)
  before timing begins. Enforces exact `morsel_pushdown == stock_default`
  parity on **62/62** admitted queries and exact `stock_default == duckdb_default`
  parity on **96/99** queries (with the 3 `INT / INT` queries documented above).

## Reproduction Commands

From the repository root on a Linux `aarch64` host with `uv`, `taskset`, and
the pinned Rust toolchain:

```bash
set -euo pipefail
b=benchmarks/morsel_tpcds
work="$HOME/.cache/morsel-tpcds"
mkdir -p "$work"
revision=$(git rev-parse HEAD)

# 1. Reuse an existing ClickHouse static binary or install outside Git.
ch_bin="$HOME/.local/bin/clickhouse"
if [ ! -x "$ch_bin" ] && [ -x "$HOME/.cache/clickhouse/clickhouse" ]; then
  ch_bin="$HOME/.cache/clickhouse/clickhouse"
fi
if [ ! -x "$ch_bin" ]; then
  mkdir -p "$HOME/.local/bin"
  (cd "$HOME/.local/bin" && curl -fsSL https://clickhouse.com/ | sh)
  ch_bin="$HOME/.local/bin/clickhouse"
fi

# 2. Build the standalone runner as a detached Cargo example (zero workspace edits).
git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$b/tpcds_runner.rs" "$work/pinned/datafusion/core/examples/morsel_tpcds_runner.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$revision" CARGO_TARGET_DIR="$work/target"     RUSTC_WRAPPER=sccache     cargo build --locked --profile release-nonlto -j 10       -p datafusion --features math_expressions       --example morsel_tpcds_runner
)
runner="$work/target/release-nonlto/examples/morsel_tpcds_runner"

# 3. Create the pinned Python environment.
uv venv --python 3.12.10 "$work/venv"
uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"

# 4. Run the fast deterministic --sf 0.01 smoke unittest suite.
MORSEL_TPCDS_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision"   CLICKHOUSE_BIN="$ch_bin"   "$py" -B -m unittest discover -s "$b" -v

# 5. Generate deterministic SF=1 TPC-DS Parquet tables outside Git.
data_dir="$work/tpcds_sf1"
"$py" -B "$b/harness.py" generate "$data_dir" --sf 1 --row-group-size 131072

# 6. Run the 99-query Morsel admission & rejection audit with parity verification.
"$py" -B "$b/harness.py" audit "$data_dir" --runner "$runner"   --threads 8 --timeout 120 --format markdown

# 7. Run the quiet-host SF=1 4-way warm-cache benchmark on 8 pinned cores.
vmstat -y 1 3
taskset -c 0-7 "$py" -B "$b/harness.py" bench "$data_dir"   --runner "$runner" --revision "$revision" --clickhouse-bin "$ch_bin"   --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 5   --queries all > "$work/sf1_warm_4way.jsonl"
vmstat -y 1 3

# 8. Print Markdown summary tables.
"$py" -B "$b/harness.py" summarize "$work/sf1_warm_4way.jsonl"

# 9. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```

## Quiet-Host SF=1 4-Way Baseline: `rob-tandy-dduck2`, 2026-11-28 UTC

### Environment and Controls

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 vCPUs (1 thread per core, single NUMA node, 32 MiB L3),
  61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  `--threads 8`, batch size `8,192`.
- **Host idle verification:** `vmstat -y 1 3` immediately before the run showed
  **90–93% idle** (`r=1–2`); immediately after showed **84–93% idle** (`r=2`).
  Across all 2,220 measured trials, the median middle-3 spread across the 5
  repeats was **2.6%** (p90 **6.2%**).
- **Build and versions:**
  - DataFusion revision: `1663c620eede06cfde33ebb0fa1beeef0868b714`
    (`release-nonlto`, `-p datafusion --features math_expressions`)
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`
  - DuckDB `2.0.1`, ClickHouse `26.11.1.833` (official static `aarch64` build),
    Python `3.12.10`, PyArrow `26.0.1`, NumPy `2.5.5`
- **Page-cache verification:** All **2,220 / 2,220** measured trials started
  and ended with **100% 4 KiB page residency** via `mincore` (0 residency
  drops).
- **Protocol:** 1 unmeasured warmup round + 5 measured rounds per query/arm
  with alternating arm order. Values below are medians in milliseconds.

### Executive Summary

| Subset                               | Queries | `stock_default` GeoMean (Sum) | `stock_pushdown` GeoMean (Sum) | `morsel_pushdown` GeoMean (Sum) | `duckdb_default` GeoMean (Sum) | `clickhouse_default` GeoMean (Sum) |
| ------------------------------------ | ------: | ----------------------------: | -----------------------------: | ------------------------------: | -----------------------------: | ---------------------------------: |
| **Morsel-admitted subset**           |  **62** |         85.7 ms (11,105.1 ms) |           75.3 ms (7,388.0 ms) |        **68.2 ms (6,878.6 ms)** |           67.7 ms (5,211.0 ms) |                                  — |
| **Admitted ∩ ClickHouse-compatible** |  **52** |         93.5 ms (10,439.0 ms) |           80.8 ms (6,756.6 ms) |        **74.0 ms (6,316.6 ms)** |           70.7 ms (4,627.0 ms) |             181.1 ms (12,425.0 ms) |
| **Full TPC-DS suite**                |  **99** |         95.1 ms (17,448.1 ms) |          84.6 ms (13,141.5 ms) |                               — |       **73.3 ms (9,547.5 ms)** |                                  — |
| **Full ∩ ClickHouse-compatible**     |  **85** |         97.7 ms (15,891.2 ms) |          86.0 ms (11,656.6 ms) |                               — |       **74.6 ms (8,522.5 ms)** |             186.4 ms (21,514.0 ms) |

Headline ratios on the **62 natively admitted TPC-DS queries**:

- **`morsel_pushdown` vs. `stock_default`:** **0.80×** geomean (**1.26×
  faster**, winning **48 / 62** queries; sum 6,878.6 ms vs. 11,105.1 ms =
  **1.61× faster** total runtime).
- **`morsel_pushdown` vs. `stock_pushdown`:** **0.91×** geomean (**1.10×
  faster**, winning **48 / 62** queries).
- **`morsel_pushdown` vs. `duckdb_default` (2.0.1):** **1.01×** geomean
  (68.2 ms vs. 67.7 ms — within **0.7%** of DuckDB across 62 queries, winning
  **24 / 62** queries).
- **`morsel_pushdown` vs. `clickhouse_default` (26.11.1.833, 52 compatible
  admitted queries):** **0.41×** geomean (**2.45× faster**, winning
  **51 / 52** queries).

### 1. 99-Query Morsel Admission and Rejection Breakdown

Under `morsel_mode=required`, **62 / 99 (62.6%)** standard TPC-DS queries plan
and execute natively in `MorselExec` with **62 / 62 exact parity against Stock
DataFusion** (59 / 62 exact against DuckDB 2.0.1, plus 3 documented `INT / INT`
division queries `q21`, `q78`, `q83` matching Stock DataFusion identically).
Only **5 missing logical operators** account for all **37 / 99** rejected
queries.

#### Primary `morsel_mode=required` Rejection Breakdown (First-Hit)

| Rejection category                 | Queries blocked | Query IDs                                                                                        |
| ---------------------------------- | --------------: | ------------------------------------------------------------------------------------------------ |
| Window                             |              14 | `q12`, `q20`, `q36`, `q44`, `q47`, `q51`, `q53`, `q57`, `q63`, `q67`, `q70`, `q86`, `q89`, `q98` |
| UNION / INTERSECT / EXCEPT         |              11 | `q02`, `q23`, `q33`, `q49`, `q54`, `q56`, `q60`, `q66`, `q71`, `q75`, `q76`                      |
| ROLLUP / GROUPING SETS / CUBE      |               7 | `q05`, `q14`, `q18`, `q22`, `q27`, `q77`, `q80`                                                  |
| Unsupported Join Type (LeftMark)   |               3 | `q10`, `q35`, `q45`                                                                              |
| Unsupported Aggregate (stddev/var) |               2 | `q17`, `q39`                                                                                     |

#### All Blocking Operators Present Across Rejected Queries (LogicalPlan Audit)

Because `MorselQueryPlanner` fails fast on the first unsupported operator,
some rejected queries contain two blocking operators (e.g., `q05`, `q14`,
`q77`, `q80` contain both `Union` and `ROLLUP`; `q36`, `q67`, `q70`, `q86`
contain both `Window` and `ROLLUP`; `q49` contains both `Union` and `Window`).

| Blocking operator                  | Queries containing blocker | Sole-blocker queries | Query IDs                                                                                               |
| ---------------------------------- | -------------------------: | -------------------: | ------------------------------------------------------------------------------------------------------- |
| UNION / INTERSECT / EXCEPT         |                         15 |                   10 | `q02`, `q05`, `q14`, `q23`, `q33`, `q49`, `q54`, `q56`, `q60`, `q66`, `q71`, `q75`, `q76`, `q77`, `q80` |
| Window                             |                         15 |                   10 | `q12`, `q20`, `q36`, `q44`, `q47`, `q49`, `q51`, `q53`, `q57`, `q63`, `q67`, `q70`, `q86`, `q89`, `q98` |
| ROLLUP / GROUPING SETS / CUBE      |                         11 |                    3 | `q05`, `q14`, `q18`, `q22`, `q27`, `q36`, `q67`, `q70`, `q77`, `q80`, `q86`                             |
| Unsupported Join Type (LeftMark)   |                          3 |                    3 | `q10`, `q35`, `q45`                                                                                     |
| Unsupported Aggregate (stddev/var) |                          2 |                    2 | `q17`, `q39`                                                                                            |

#### Operator Prioritization / Greedy Unlock Roadmap

Implementing just **3 operators** (`Union`, `Window`, and `ROLLUP / GROUPING
SETS / CUBE`) unlocks **+32 queries**, lifting TPC-DS coverage from **62 / 99
(62.6%)** to **94 / 99 (94.9%)**:

| Priority step | Operator added                     | Newly unlocked | Cumulative admitted | Unlocked queries                                                            |
| ------------: | ---------------------------------- | -------------: | ------------------: | --------------------------------------------------------------------------- |
|             1 | UNION / INTERSECT / EXCEPT         |            +10 |       72/99 (72.7%) | `q02`, `q23`, `q33`, `q54`, `q56`, `q60`, `q66`, `q71`, `q75`, `q76`        |
|             2 | Window                             |            +11 |       83/99 (83.8%) | `q12`, `q20`, `q44`, `q47`, `q49`, `q51`, `q53`, `q57`, `q63`, `q89`, `q98` |
|             3 | ROLLUP / GROUPING SETS / CUBE      |            +11 |       94/99 (94.9%) | `q05`, `q14`, `q18`, `q22`, `q27`, `q36`, `q67`, `q70`, `q77`, `q80`, `q86` |
|             4 | Unsupported Join Type (LeftMark)   |             +3 |       97/99 (98.0%) | `q10`, `q35`, `q45`                                                         |
|             5 | Unsupported Aggregate (stddev/var) |             +2 |      99/99 (100.0%) | `q17`, `q39`                                                                |

_(Note: unlock counts are derived from top-level optimized `LogicalPlan`
operators; implementing an operator can occasionally expose secondary
expression-level constraints inside newly unlocked subtrees.)_

### 2. Admitted Subset 4-Way SF=1 Timings (62 Queries)

| Query                     | Rows | `stock_default` ms | `stock_pushdown` ms | `morsel_pushdown` ms | `duckdb_default` ms | `clickhouse_default` ms | Morsel / Stock(def) | Morsel / Stock(pd) | Morsel / DuckDB | Morsel / ClickHouse |
| ------------------------- | ---: | -----------------: | ------------------: | -------------------: | ------------------: | ----------------------: | ------------------: | -----------------: | --------------: | ------------------: |
| `q01`                     |    0 |               19.0 |                19.3 |                 13.2 |                32.5 |                       — |               0.69x |              0.69x |           0.41x |                   — |
| `q03`                     |   38 |               33.9 |                25.1 |                 22.7 |                22.8 |                    72.0 |               0.67x |              0.91x |           1.00x |               0.32x |
| `q04`                     |    8 |             1004.5 |               786.4 |                833.0 |               355.8 |                  1357.0 |               0.83x |              1.06x |           2.34x |               0.61x |
| `q06`                     |   41 |               73.0 |                72.9 |                 71.4 |                68.5 |                   153.0 |               0.98x |              0.98x |           1.04x |               0.47x |
| `q07`                     |  100 |              133.6 |               103.3 |                 83.3 |               108.9 |                   232.0 |               0.62x |              0.81x |           0.76x |               0.36x |
| `q08`                     |    0 |               71.7 |                36.3 |                 36.0 |                66.0 |                   130.0 |               0.50x |              0.99x |           0.55x |               0.28x |
| `q09`                     |    1 |              257.0 |               273.5 |                286.5 |               208.2 |                   388.0 |               1.11x |              1.05x |           1.38x |               0.74x |
| `q11`                     |   88 |              497.7 |               402.3 |                396.2 |               161.4 |                   623.0 |               0.80x |              0.98x |           2.45x |               0.64x |
| `q13`                     |    1 |              201.8 |               181.3 |                162.8 |               146.3 |                   259.0 |               0.81x |              0.90x |           1.11x |               0.63x |
| `q15`                     |  100 |               70.9 |                39.3 |                 47.6 |                51.5 |                    95.0 |               0.67x |              1.21x |           0.92x |               0.50x |
| `q16`                     |    1 |               23.5 |                22.8 |                 19.5 |                63.1 |                       — |               0.83x |              0.85x |           0.31x |                   — |
| `q19`                     |  100 |               76.8 |                66.3 |                 57.3 |                59.2 |                   145.0 |               0.75x |              0.86x |           0.97x |               0.40x |
| `q21`                     |  100 |               39.4 |                40.2 |                 28.9 |                25.0 |                   112.0 |               0.73x |              0.72x |           1.16x |               0.26x |
| `q24`                     |    0 |              429.7 |               180.5 |                191.7 |               146.4 |                   354.0 |               0.45x |              1.06x |           1.31x |               0.54x |
| `q25`                     |    0 |              200.4 |                74.2 |                101.6 |                72.7 |                   216.0 |               0.51x |              1.37x |           1.40x |               0.47x |
| `q26`                     |  100 |               73.6 |                62.2 |                 53.7 |                75.4 |                   178.0 |               0.73x |              0.86x |           0.71x |               0.30x |
| `q28`                     |    1 |              288.5 |               320.2 |                301.3 |               168.5 |                   251.0 |               1.04x |              0.94x |           1.79x |               1.20x |
| `q29`                     |    0 |              169.9 |               112.8 |                114.7 |                77.1 |                   230.0 |               0.67x |              1.02x |           1.49x |               0.50x |
| `q30`                     |   92 |               61.6 |                58.6 |                 56.9 |                53.0 |                   149.0 |               0.92x |              0.97x |           1.07x |               0.38x |
| `q31`                     |   51 |              203.7 |               252.2 |                208.2 |                91.2 |                   415.0 |               1.02x |              0.83x |           2.28x |               0.50x |
| `q32`                     |    1 |               51.3 |                39.3 |                 37.1 |                41.3 |                       — |               0.72x |              0.94x |           0.90x |                   — |
| `q34`                     |    0 |               20.1 |                19.9 |                 15.2 |                46.7 |                    95.0 |               0.75x |              0.76x |           0.33x |               0.16x |
| `q37`                     |    0 |               26.1 |                19.8 |                 17.8 |                26.6 |                    63.0 |               0.68x |              0.90x |           0.67x |               0.28x |
| `q38`                     |    1 |              110.4 |               120.9 |                132.2 |               116.0 |                   247.0 |               1.20x |              1.09x |           1.14x |               0.54x |
| `q40`                     |  100 |               72.3 |                48.2 |                 43.9 |                43.3 |                   116.0 |               0.61x |              0.91x |           1.01x |               0.38x |
| `q41`                     |    0 |               27.7 |                25.4 |                 23.5 |                27.8 |                    60.0 |               0.85x |              0.92x |           0.85x |               0.39x |
| `q42`                     |   10 |               45.7 |                37.0 |                 30.2 |                30.7 |                    76.0 |               0.66x |              0.81x |           0.98x |               0.40x |
| `q43`                     |    0 |               17.0 |                16.9 |                 13.4 |                21.0 |                    53.0 |               0.79x |              0.79x |           0.64x |               0.25x |
| `q46`                     |    0 |               32.5 |                33.9 |                 31.9 |                50.2 |                    93.0 |               0.98x |              0.94x |           0.64x |               0.34x |
| `q48`                     |    1 |              142.3 |               150.0 |                123.8 |               121.7 |                   231.0 |               0.87x |              0.83x |           1.02x |               0.54x |
| `q50`                     |    6 |              111.0 |               110.2 |                 82.6 |                76.8 |                       — |               0.74x |              0.75x |           1.08x |                   — |
| `q52`                     |  100 |               46.2 |                37.8 |                 32.3 |                32.3 |                    74.0 |               0.70x |              0.86x |           1.00x |               0.44x |
| `q55`                     |   69 |               44.3 |                35.0 |                 29.2 |                28.9 |                    73.0 |               0.66x |              0.83x |           1.01x |               0.40x |
| `q58`                     |    0 |              119.5 |               141.6 |                 77.8 |                88.2 |                       — |               0.65x |              0.55x |           0.88x |                   — |
| `q59`                     |  100 |              153.2 |               173.2 |                111.4 |                70.7 |                   209.0 |               0.73x |              0.64x |           1.58x |               0.53x |
| `q61`                     |    1 |               29.5 |                28.8 |                 20.8 |                68.4 |                   120.0 |               0.70x |              0.72x |           0.30x |               0.17x |
| `q62`                     |  100 |               42.1 |                46.8 |                 39.3 |                33.5 |                       — |               0.93x |              0.84x |           1.17x |                   — |
| `q64`                     |   10 |              407.0 |               281.2 |                509.1 |               177.5 |                   828.0 |               1.25x |              1.81x |           2.87x |               0.61x |
| `q65`                     |  100 |              105.9 |               151.5 |                105.3 |                78.0 |                   189.0 |               0.99x |              0.70x |           1.35x |               0.56x |
| `q68`                     |    0 |               22.8 |                22.6 |                 16.7 |                52.8 |                    96.0 |               0.73x |              0.74x |           0.32x |               0.17x |
| `q69`                     |  100 |               93.2 |               120.3 |                100.8 |                91.9 |                   271.0 |               1.08x |              0.84x |           1.10x |               0.37x |
| `q72`                     |  100 |             3068.2 |               243.8 |                187.3 |               173.7 |                   478.0 |               0.06x |              0.77x |           1.08x |               0.39x |
| `q73`                     |    0 |               19.8 |                19.2 |                 15.0 |                58.8 |                    92.0 |               0.76x |              0.78x |           0.26x |               0.16x |
| `q74`                     |   90 |              311.4 |               272.4 |                247.4 |               166.3 |                   545.0 |               0.79x |              0.91x |           1.49x |               0.45x |
| `q78`                     |  100 |              679.7 |               655.4 |                279.0 |               186.6 |                   385.0 |               0.41x |              0.43x |           1.50x |               0.72x |
| `q79`                     |  100 |              129.3 |               157.6 |                133.7 |               101.0 |                   198.0 |               1.03x |              0.85x |           1.32x |               0.68x |
| `q81`                     |   96 |               53.7 |                55.3 |                 56.8 |                44.4 |                   149.0 |               1.06x |              1.03x |           1.28x |               0.38x |
| `q82`                     |    1 |               39.6 |                39.1 |                 30.6 |                28.5 |                   147.0 |               0.77x |              0.78x |           1.07x |               0.21x |
| `q83`                     |    4 |               48.8 |                49.9 |                 46.4 |                53.7 |                   176.0 |               0.95x |              0.93x |           0.86x |               0.26x |
| `q84`                     |   22 |               45.1 |                47.8 |                 95.9 |                42.6 |                   139.0 |               2.13x |              2.01x |           2.25x |               0.69x |
| `q85`                     |    3 |              116.0 |               112.0 |                 98.4 |                85.7 |                   351.0 |               0.85x |              0.88x |           1.15x |               0.28x |
| `q87`                     |    1 |              107.4 |               124.0 |                127.5 |               117.2 |                   248.0 |               1.19x |              1.03x |           1.09x |               0.51x |
| `q88`                     |    1 |              208.7 |               229.6 |                227.7 |               253.6 |                   535.0 |               1.09x |              0.99x |           0.90x |               0.43x |
| `q90`                     |    1 |               37.4 |                37.3 |                 34.3 |                44.2 |                   112.0 |               0.92x |              0.92x |           0.78x |               0.31x |
| `q91`                     |    4 |               56.3 |                60.1 |                 54.8 |                47.8 |                   147.0 |               0.97x |              0.91x |           1.15x |               0.37x |
| `q92`                     |    1 |               38.9 |                35.8 |                 32.3 |                39.5 |                       — |               0.83x |              0.90x |           0.82x |                   — |
| `q93`                     |  100 |              107.3 |               106.5 |                130.3 |                60.8 |                   210.0 |               1.21x |              1.22x |           2.14x |               0.62x |
| `q94`                     |    1 |               52.8 |                50.0 |                 45.9 |                63.2 |                       — |               0.87x |              0.92x |           0.73x |                   — |
| `q95`                     |    1 |              145.5 |               103.2 |                158.6 |               101.9 |                       — |               1.09x |              1.54x |           1.56x |                   — |
| `q96`                     |    1 |               44.3 |                48.9 |                 38.9 |                37.0 |                    95.0 |               0.88x |              0.80x |           1.05x |               0.41x |
| `q97`                     |    1 |               82.0 |                87.7 |                101.7 |                82.7 |                   165.0 |               1.24x |              1.16x |           1.23x |               0.62x |
| `q99`                     |   90 |               62.5 |                62.2 |                 55.9 |                43.7 |                       — |               0.89x |              0.90x |           1.28x |                   — |
| **GeoMean (62 admitted)** |    — |           **85.7** |            **75.3** |             **68.2** |            **67.7** |         **181.1** (52q) |           **0.80x** |          **0.91x** |       **1.01x** |     **0.41x** (52q) |
| **Sum (62 admitted)**     |    — |        **11105.1** |          **7388.0** |           **6878.6** |          **5211.0** |       **12425.0** (52q) |           **0.62x** |          **0.93x** |       **1.32x** |     **0.51x** (52q) |

### 3. Full 99-Query SF=1 Timings (Stock vs. DuckDB vs. ClickHouse + Morsel)

ClickHouse cells marked with `*` executed cleanly with a documented ClickHouse
numeric/semantic difference vs. DuckDB (see Section 5).

| Query               |  Rows | Morsel status                      | `stock_default` ms | `stock_pushdown` ms | `morsel_pushdown` ms | `duckdb_default` ms | `clickhouse_default` ms |
| ------------------- | ----: | ---------------------------------- | -----------------: | ------------------: | -------------------: | ------------------: | ----------------------: |
| `q01`               |     0 | Admitted                           |               19.0 |                19.3 |                 13.2 |                32.5 |            incompatible |
| `q02`               | 2,513 | UNION / INTERSECT / EXCEPT         |              122.4 |               119.3 |                    — |                64.3 |                  172.0* |
| `q03`               |    38 | Admitted                           |               33.9 |                25.1 |                 22.7 |                22.8 |                    72.0 |
| `q04`               |     8 | Admitted                           |             1004.5 |               786.4 |                833.0 |               355.8 |                  1357.0 |
| `q05`               |   100 | ROLLUP / GROUPING SETS / CUBE      |              151.4 |               134.7 |                    — |               129.1 |                   264.0 |
| `q06`               |    41 | Admitted                           |               73.0 |                72.9 |                 71.4 |                68.5 |                   153.0 |
| `q07`               |   100 | Admitted                           |              133.6 |               103.3 |                 83.3 |               108.9 |                   232.0 |
| `q08`               |     0 | Admitted                           |               71.7 |                36.3 |                 36.0 |                66.0 |                   130.0 |
| `q09`               |     1 | Admitted                           |              257.0 |               273.5 |                286.5 |               208.2 |                   388.0 |
| `q10`               |     2 | Unsupported Join Type (LeftMark)   |              100.4 |               122.0 |                    — |               104.8 |                   271.0 |
| `q11`               |    88 | Admitted                           |              497.7 |               402.3 |                396.2 |               161.4 |                  623.0* |
| `q12`               |   100 | Window                             |               40.1 |                34.1 |                    — |                32.3 |                    74.0 |
| `q13`               |     1 | Admitted                           |              201.8 |               181.3 |                162.8 |               146.3 |                   259.0 |
| `q14`               |   200 | ROLLUP / GROUPING SETS / CUBE      |              737.3 |               719.4 |                    — |               486.0 |                  1472.0 |
| `q15`               |   100 | Admitted                           |               70.9 |                39.3 |                 47.6 |                51.5 |                    95.0 |
| `q16`               |     1 | Admitted                           |               23.5 |                22.8 |                 19.5 |                63.1 |            incompatible |
| `q17`               |     1 | Unsupported Aggregate (stddev/var) |              170.2 |                98.3 |                    — |                74.6 |                  212.0* |
| `q18`               |   100 | ROLLUP / GROUPING SETS / CUBE      |              144.4 |               103.2 |                    — |               109.0 |                   256.0 |
| `q19`               |   100 | Admitted                           |               76.8 |                66.3 |                 57.3 |                59.2 |                   145.0 |
| `q20`               |   100 | Window                             |               49.8 |                38.7 |                    — |                36.1 |                    80.0 |
| `q21`               |   100 | Admitted                           |               39.4 |                40.2 |                 28.9 |                25.0 |                   112.0 |
| `q22`               |   100 | ROLLUP / GROUPING SETS / CUBE      |              354.6 |               363.9 |                    — |                70.5 |                   226.0 |
| `q23`               |     1 | UNION / INTERSECT / EXCEPT         |             1126.2 |               848.0 |                    — |               848.3 |                  1269.0 |
| `q24`               |     0 | Admitted                           |              429.7 |               180.5 |                191.7 |               146.4 |                   354.0 |
| `q25`               |     0 | Admitted                           |              200.4 |                74.2 |                101.6 |                72.7 |                   216.0 |
| `q26`               |   100 | Admitted                           |               73.6 |                62.2 |                 53.7 |                75.4 |                   178.0 |
| `q27`               |     1 | ROLLUP / GROUPING SETS / CUBE      |               25.3 |                20.8 |                    — |                41.6 |                  143.0* |
| `q28`               |     1 | Admitted                           |              288.5 |               320.2 |                301.3 |               168.5 |                   251.0 |
| `q29`               |     0 | Admitted                           |              169.9 |               112.8 |                114.7 |                77.1 |                   230.0 |
| `q30`               |    92 | Admitted                           |               61.6 |                58.6 |                 56.9 |                53.0 |                   149.0 |
| `q31`               |    51 | Admitted                           |              203.7 |               252.2 |                208.2 |                91.2 |                  415.0* |
| `q32`               |     1 | Admitted                           |               51.3 |                39.3 |                 37.1 |                41.3 |            incompatible |
| `q33`               |   100 | UNION / INTERSECT / EXCEPT         |               95.1 |                95.3 |                    — |                99.5 |                   207.0 |
| `q34`               |     0 | Admitted                           |               20.1 |                19.9 |                 15.2 |                46.7 |                    95.0 |
| `q35`               |   100 | Unsupported Join Type (LeftMark)   |              115.7 |               136.3 |                    — |               110.8 |                   291.0 |
| `q36`               |     1 | Window                             |               20.6 |                20.3 |                    — |                32.8 |                   70.0* |
| `q37`               |     0 | Admitted                           |               26.1 |                19.8 |                 17.8 |                26.6 |                    63.0 |
| `q38`               |     1 | Admitted                           |              110.4 |               120.9 |                132.2 |               116.0 |                   247.0 |
| `q39`               |   153 | Unsupported Aggregate (stddev/var) |              194.2 |               203.1 |                    — |               117.0 |                   408.0 |
| `q40`               |   100 | Admitted                           |               72.3 |                48.2 |                 43.9 |                43.3 |                   116.0 |
| `q41`               |     0 | Admitted                           |               27.7 |                25.4 |                 23.5 |                27.8 |                    60.0 |
| `q42`               |    10 | Admitted                           |               45.7 |                37.0 |                 30.2 |                30.7 |                    76.0 |
| `q43`               |     0 | Admitted                           |               17.0 |                16.9 |                 13.4 |                21.0 |                    53.0 |
| `q44`               |     0 | Window                             |               19.2 |                19.3 |                    — |                33.3 |                    67.0 |
| `q45`               |    16 | Unsupported Join Type (LeftMark)   |               48.7 |                45.7 |                    — |                51.2 |                   118.0 |
| `q46`               |     0 | Admitted                           |               32.5 |                33.9 |                 31.9 |                50.2 |                    93.0 |
| `q47`               |   100 | Window                             |              322.9 |               344.3 |                    — |               110.6 |            incompatible |
| `q48`               |     1 | Admitted                           |              142.3 |               150.0 |                123.8 |               121.7 |                   231.0 |
| `q49`               |    31 | UNION / INTERSECT / EXCEPT         |              173.5 |               134.0 |                    — |               121.5 |                   262.0 |
| `q50`               |     6 | Admitted                           |              111.0 |               110.2 |                 82.6 |                76.8 |            incompatible |
| `q51`               |   100 | Window                             |              191.6 |               205.4 |                    — |               186.6 |                   309.0 |
| `q52`               |   100 | Admitted                           |               46.2 |                37.8 |                 32.3 |                32.3 |                    74.0 |
| `q53`               |    14 | Window                             |               54.7 |                46.3 |                    — |                33.4 |                   103.0 |
| `q54`               |     0 | UNION / INTERSECT / EXCEPT         |              102.7 |                60.7 |                    — |                70.9 |                   156.0 |
| `q55`               |    69 | Admitted                           |               44.3 |                35.0 |                 29.2 |                28.9 |                    73.0 |
| `q56`               |    78 | UNION / INTERSECT / EXCEPT         |               99.1 |                92.0 |                    — |                88.1 |                   189.0 |
| `q57`               |   100 | Window                             |              171.8 |               169.1 |                    — |                63.5 |            incompatible |
| `q58`               |     0 | Admitted                           |              119.5 |               141.6 |                 77.8 |                88.2 |            incompatible |
| `q59`               |   100 | Admitted                           |              153.2 |               173.2 |                111.4 |                70.7 |                   209.0 |
| `q60`               |   100 | UNION / INTERSECT / EXCEPT         |               99.8 |                99.3 |                    — |                97.3 |                   198.0 |
| `q61`               |     1 | Admitted                           |               29.5 |                28.8 |                 20.8 |                68.4 |                   120.0 |
| `q62`               |   100 | Admitted                           |               42.1 |                46.8 |                 39.3 |                33.5 |            incompatible |
| `q63`               |    44 | Window                             |               56.3 |                46.3 |                    — |                34.9 |                    99.0 |
| `q64`               |    10 | Admitted                           |              407.0 |               281.2 |                509.1 |               177.5 |                   828.0 |
| `q65`               |   100 | Admitted                           |              105.9 |               151.5 |                105.3 |                78.0 |                   189.0 |
| `q66`               |     5 | UNION / INTERSECT / EXCEPT         |              144.2 |               118.1 |                    — |               115.8 |            incompatible |
| `q67`               |   100 | Window                             |              231.6 |               249.8 |                    — |               214.9 |                   821.0 |
| `q68`               |     0 | Admitted                           |               22.8 |                22.6 |                 16.7 |                52.8 |                    96.0 |
| `q69`               |   100 | Admitted                           |               93.2 |               120.3 |                100.8 |                91.9 |                   271.0 |
| `q70`               |     3 | Window                             |              201.5 |               229.5 |                    — |                81.3 |                   206.0 |
| `q71`               | 1,102 | UNION / INTERSECT / EXCEPT         |               94.4 |                73.0 |                    — |                74.8 |                   156.0 |
| `q72`               |   100 | Admitted                           |             3068.2 |               243.8 |                187.3 |               173.7 |                   478.0 |
| `q73`               |     0 | Admitted                           |               19.8 |                19.2 |                 15.0 |                58.8 |                    92.0 |
| `q74`               |    90 | Admitted                           |              311.4 |               272.4 |                247.4 |               166.3 |                   545.0 |
| `q75`               |   100 | UNION / INTERSECT / EXCEPT         |              251.9 |               222.0 |                    — |               151.2 |            incompatible |
| `q76`               |   100 | UNION / INTERSECT / EXCEPT         |               83.3 |                91.6 |                    — |                82.7 |                   154.0 |
| `q77`               |    44 | ROLLUP / GROUPING SETS / CUBE      |              134.0 |               150.8 |                    — |               113.5 |                   264.0 |
| `q78`               |   100 | Admitted                           |              679.7 |               655.4 |                279.0 |               186.6 |                  385.0* |
| `q79`               |   100 | Admitted                           |              129.3 |               157.6 |                133.7 |               101.0 |                   198.0 |
| `q80`               |   100 | ROLLUP / GROUPING SETS / CUBE      |              252.1 |               155.6 |                    — |               137.3 |                   295.0 |
| `q81`               |    96 | Admitted                           |               53.7 |                55.3 |                 56.8 |                44.4 |                   149.0 |
| `q82`               |     1 | Admitted                           |               39.6 |                39.1 |                 30.6 |                28.5 |                   147.0 |
| `q83`               |     4 | Admitted                           |               48.8 |                49.9 |                 46.4 |                53.7 |                   176.0 |
| `q84`               |    22 | Admitted                           |               45.1 |                47.8 |                 95.9 |                42.6 |                   139.0 |
| `q85`               |     3 | Admitted                           |              116.0 |               112.0 |                 98.4 |                85.7 |                   351.0 |
| `q86`               |   100 | Window                             |               30.4 |                36.5 |                    — |                30.3 |                    75.0 |
| `q87`               |     1 | Admitted                           |              107.4 |               124.0 |                127.5 |               117.2 |                   248.0 |
| `q88`               |     1 | Admitted                           |              208.7 |               229.6 |                227.7 |               253.6 |                   535.0 |
| `q89`               |   100 | Window                             |               61.3 |                58.6 |                    — |                43.2 |                   112.0 |
| `q90`               |     1 | Admitted                           |               37.4 |                37.3 |                 34.3 |                44.2 |                   112.0 |
| `q91`               |     4 | Admitted                           |               56.3 |                60.1 |                 54.8 |                47.8 |                   147.0 |
| `q92`               |     1 | Admitted                           |               38.9 |                35.8 |                 32.3 |                39.5 |            incompatible |
| `q93`               |   100 | Admitted                           |              107.3 |               106.5 |                130.3 |                60.8 |                   210.0 |
| `q94`               |     1 | Admitted                           |               52.8 |                50.0 |                 45.9 |                63.2 |            incompatible |
| `q95`               |     1 | Admitted                           |              145.5 |               103.2 |                158.6 |               101.9 |            incompatible |
| `q96`               |     1 | Admitted                           |               44.3 |                48.9 |                 38.9 |                37.0 |                    95.0 |
| `q97`               |     1 | Admitted                           |               82.0 |                87.7 |                101.7 |                82.7 |                   165.0 |
| `q98`               | 2,486 | Window                             |               70.1 |                48.4 |                    — |                43.6 |                    90.0 |
| `q99`               |    90 | Admitted                           |               62.5 |                62.2 |                 55.9 |                43.7 |            incompatible |
| **GeoMean (99/99)** |     — | **62/99 admitted**                 |           **95.1** |            **84.6** |       **68.2** (62q) |            **73.3** |         **186.4** (85q) |
| **Sum (99/99)**     |     — | **62/99 admitted**                 |        **17448.1** |         **13141.5** |     **6878.6** (62q) |          **9547.5** |       **21514.0** (85q) |

### 4. ClickHouse Dialect Incompatibilities (14 Queries)

ClickHouse executed **85 / 99** unmodified TPC-DS queries cleanly and rejected
**14 / 99** queries due to ClickHouse SQL dialect differences (primarily
single-quoted column aliases like `as 'order count'` / `as '30 days'`,
case-sensitive `SR_FEE`, and CTE/window identifier scoping):

| Query | ClickHouse error                                                                                                                                         |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q01` | `Code: 47. DB::Exception: Unknown expression or function identifier `SR_FEE`. Maybe you meant: ['sr_fee']. In scope  customer_total_return AS ctr1. ...` |
| `q16` | `Code: 62. DB::Exception: Syntax error: failed at position 48 ('order count') (line 2, col 39): 'order count'`                                           |
| `q32` | `Code: 62. DB::Exception: Syntax error: failed at position 38 ('excess discount amount') (line 1, col 38): 'excess discount amount'`                     |
| `q47` | `Code: 47. DB::Exception: Unknown expression or function identifier `d_year`. In scope WITH v1 AS (SELECT i_category, i_brand, s_store_name, s_compa...` |
| `q50` | `Code: 62. DB::Exception: Syntax error: failed at position 242 ('30 days') (line 12, col 88): '30 days'`                                                 |
| `q57` | `Code: 47. DB::Exception: Unknown expression or function identifier `d_year`. In scope WITH v1 AS (SELECT i_category, i_brand, cc_name, d_year, d_mo...` |
| `q58` | `Code: 207. DB::Exception: JOIN , ... ambiguous identifier 'item_id'. In scope WITH ss_items AS (SELECT i_item_id AS item_id, sum(ss_ext_sales_price...` |
| `q62` | `Code: 62. DB::Exception: Syntax error: failed at position 149 ('30 days') (line 5, col 84): '30 days'`                                                  |
| `q66` | `Code: 184. DB::Exception: Aggregate function sum(jan_sales) AS jan_sales is found inside another aggregate function in query. (ILLEGAL_AGGREGATION)`    |
| `q75` | `Code: 43. DB::Exception: Illegal type Variant(Decimal(9, 2), Float64) of argument for aggregate function SUM. If it is the inferred common type of ...` |
| `q92` | `Code: 62. DB::Exception: Syntax error: failed at position 42 ('Excess Discount Amount') (line 2, col 33): 'Excess Discount Amount'`                     |
| `q94` | `Code: 62. DB::Exception: Syntax error: failed at position 48 ('order count') (line 2, col 39): 'order count'`                                           |
| `q95` | `Code: 62. DB::Exception: Syntax error: failed at position 273 ('order count') (line 7, col 39): 'order count'`                                          |
| `q99` | `Code: 62. DB::Exception: Syntax error: failed at position 148 ('30 days') (line 5, col 84): '30 days'`                                                  |

### 5. Cross-Engine Parity Notes

#### ClickHouse Semantic Parity Notes (7 Executable Queries)

On 7 of the 85 executable queries (`q02`, `q11`, `q17`, `q27`, `q31`, `q36`,
`q78`), ClickHouse ran to completion with documented ClickHouse semantic
differences relative to DuckDB 2.0.1 and DataFusion (fixed-scale Decimal
division truncation in `q02`/`q11`/`q31`/`q78`, `stddevSamp` returning `NaN`
instead of `NULL` for 1-row groups in `q17`, and `GROUPING()` / `ROLLUP`
filtering in `q27`/`q36`):

| Query | Observed difference vs. DuckDB oracle                                                                                      |
| ----- | -------------------------------------------------------------------------------------------------------------------------- |
| `q02` | `clickhouse_default/q02: mismatch at row 0, col 'round((thu_sales1 / thu_sales2), 2)': expected 3.18, got Decimal('3.17')` |
| `q11` | `clickhouse_default/q11: row count mismatch: expected 88, got 87`                                                          |
| `q17` | `clickhouse_default/q17: mismatch at row 0, col 'store_sales_quantitystdev': expected None, got nan`                       |
| `q27` | `clickhouse_default/q27: row count mismatch: expected 1, got 0`                                                            |
| `q31` | `clickhouse_default/q31: row count mismatch: expected 51, got 50`                                                          |
| `q36` | `clickhouse_default/q36: row count mismatch: expected 1, got 0`                                                            |
| `q78` | `clickhouse_default/q78: mismatch at row 48, col 'ratio': expected 2.13, got 2.12`                                         |

#### DataFusion vs. DuckDB Integer Division Notes (3 Queries)

- `q21`, `q78`, `q83`: DataFusion uses PostgreSQL-style truncating integer
  division (`INT / INT`), while DuckDB promotes `INT / INT` to `DOUBLE`. All
  three DataFusion arms (`stock_default`, `stock_pushdown`, `morsel_pushdown`)
  produce byte-for-byte identical results on all three queries.

### Key Observations and Next Priorities

- **Top 3 operators unlock 94.9% of TPC-DS:** `Union` (+10), `Window` (+11),
  and `ROLLUP / GROUPING SETS / CUBE` (+11) lift Morsel admission from 62/99 to
  94/99; adding `LeftMark` join (+3) and `stddev` (+2) reaches 99/99.
- **Morsel already matches DuckDB geomean across 62 TPC-DS queries:** At
  **68.2 ms** vs. DuckDB's **67.7 ms** (1.01×), Morsel beats `stock_default`
  (85.7 ms, 1.26× faster) and `stock_pushdown` (75.3 ms, 1.10× faster) on
  48/62 queries, and is **2.45× faster** than ClickHouse (181.1 ms across 52
  compatible admitted queries).
- **Standout Morsel wins:** `q72` (187.3 ms vs. `stock_default` 3,068.2 ms =
  **16.4×** faster; vs. `stock_pushdown` 243.8 ms = 1.30× faster), `q78`
  (279.0 ms vs. 679.7 ms `stock_default` / 655.4 ms `stock_pushdown` =
  **2.35–2.44×** faster), `q58` (77.8 ms vs. 141.6 ms `stock_pushdown`), and
  empty-build short-circuit / dynamic filter pushdown wins on `q01`, `q08`,
  `q16`, `q34`, `q61`, `q68`, `q73` (2–4× faster than DuckDB).
- **Optimization targets exposed by the 62-query suite:**
  - `q84` (95.9 ms vs. 45.1 ms `stock_default` / 42.6 ms DuckDB — 2.13× slower
    than Stock),
  - `q64` (509.1 ms vs. 281.2 ms `stock_pushdown` / 177.5 ms DuckDB — 18-way
    join + multi-CTE),
  - `q95` (158.6 ms vs. 103.2 ms `stock_pushdown` / 101.9 ms DuckDB),
  - `q04` (833.0 ms vs. 355.8 ms DuckDB), `q11` (396.2 ms vs. 161.4 ms DuckDB),
    `q31` (208.2 ms vs. 91.2 ms DuckDB), and `q93` (130.3 ms vs. 60.8 ms
    DuckDB).
