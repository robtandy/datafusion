<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel TPC-H Parquet Benchmark Suite

This fork-only benchmark suite compares **Stock DataFusion** (`morsel_mode=off`),
**Morsel DataFusion** (`morsel_mode=required`), and **DuckDB** on the exact same
standard TPC-H Parquet files (#165). It lives entirely under
`benchmarks/morsel_tpch/` and reuses the detached-example build pattern from
`benchmarks/morsel_parquet/` without modifying any workspace `Cargo.toml` or
production crate.

## Dataset Generation and Layout

`harness.py generate` uses DuckDB's built-in `tpch` extension (`SET threads = 1;
CALL dbgen(sf=...)`) and exports all eight standard TPC-H tables deterministically
ordered by each table's primary key into multi-file Parquet directories
(`<data_dir>/<table>/part-000000.parquet`, ...). Keep all generated Parquet files
and `.jsonl` run outputs outside Git.

Writer settings (standard analytical Parquet layout):

- Parquet format version `2.6`, data page version `1.0`
- Compression: Zstandard (`zstd`), level `3`
- Dictionary encoding: enabled (`use_dictionary=True`)
- Column min/max statistics: enabled (`write_statistics=True`)
- Default row-group size: `131,072` rows (`16 × 8,192` DataFusion batch size,
  configurable via `--row-group-size`)

Three common scale modes are supported (or any positive `--sf`):

- **Smoke mode (`--sf 0.01`):** 86,805 total rows across 35 files; generates in
  under one second and is used by `test_harness.py` to verify byte-for-byte
  determinism, exact DuckDB vs. Stock DataFusion parity across all 22 standard
  TPC-H queries (`q1`–`q22`), and `morsel_mode=required` plans/rejections.
- **SF1 benchmark mode (`--sf 1`):** 8,661,245 total rows across 35 files and
  79 row groups (266,865,655 bytes / 254.5 MiB on disk).
- **SF5 benchmark mode (`--sf 5`):** 43,299,825 total rows across 35 files and
  353 row groups (1,400,570,073 bytes / 1.30 GiB on disk; 29,999,795 `lineitem`
  rows across 16 files / 240 row groups; see
  [sf5_22q_wave4_scorecard.md](sf5_22q_wave4_scorecard.md) for the Wave-4
  post-`#206`/`#207`/`#208` idle-`dduck2` **22/22 3-way SF5 scorecard**,
  [sf5_22q_final_scorecard.md](sf5_22q_final_scorecard.md) for the earlier
  post-`#193`/`#194` idle-`dduck2` **22/22 3-way SF5 scorecard + spill check**,
  [sf5_22q_scorecard.md](sf5_22q_scorecard.md) for the initial `#192`
  22/22 SF5 scorecard, and [sf5_baseline_22q.md](sf5_baseline_22q.md)
  for the earlier pre-`#191` Stock vs. DuckDB 2.0.1 baseline).

### SF1 Dataset Layout (`--sf 1 --row-group-size 131072`)

| Table      | Ordering key               |  Files |          Rows | Row groups |   Parquet bytes |
| ---------- | -------------------------- | -----: | ------------: | ---------: | --------------: |
| `lineitem` | `l_orderkey, l_linenumber` |     16 |     6,001,215 |         48 |     177,150,918 |
| `orders`   | `o_orderkey`               |      8 |     1,500,000 |         16 |      44,579,195 |
| `partsupp` | `ps_partkey, ps_suppkey`   |      4 |       800,000 |          8 |      30,433,210 |
| `customer` | `c_custkey`                |      2 |       150,000 |          2 |       9,203,182 |
| `part`     | `p_partkey`                |      2 |       200,000 |          2 |       4,867,502 |
| `supplier` | `s_suppkey`                |      1 |        10,000 |          1 |         627,442 |
| `nation`   | `n_nationkey`              |      1 |            25 |          1 |           2,663 |
| `region`   | `r_regionkey`              |      1 |             5 |          1 |           1,543 |
| **Total**  |                            | **35** | **8,661,245** |     **79** | **266,865,655** |

For `lineitem` at SF1, 16 files with 131,072-row row groups produce 3 balanced
row groups per file (two full 131,072-row groups and one ~112,932-row group per
file; 48 row groups total), avoiding tiny tail row groups while giving both
8-worker DataFusion (`repartition_file_scans=false`) and Morsel's native Parquet
source (`NativeParquet[lineitem]`) full multi-file parallelism.

## Workloads

All 22 standard TPC-H queries (`q1`–`q22`) are loaded directly from
`benchmarks/queries/q1.sql`..`q22.sql` (rewriting `q15`'s `CREATE VIEW
revenue0` / `DROP VIEW revenue0` into a single-statement CTE for clean fresh
sessions across engines, and substituting TPC-H clause 2.4.11.3 `0.0001 / SF`
in `q11`). Use `--queries all` for all 22 standard queries, `--queries morsel`
for currently-admitted Morsel workloads, or space-/comma-separated subsets such
as `--queries q1,q3,q6,q12` or `--queries q6_count q6_scan q1_unordered`.

### Morsel-Native Coverage (22 / 22 Standard TPC-H Queries + 3 Slices)

After `#191` (general multi-stage morsel DAG, `HAVING`, `DISTINCT`, and
uncorrelated scalar subqueries), **all 22 standard TPC-H queries (`q1`–`q22`)
plus 3 single-table slices (`q1_unordered`, `q6_count`, `q6_scan`) — 25/25
workloads** — run directly over the real TPC-H Parquet schemas (`l_quantity`,
`l_extendedprice`, `l_discount`, `l_tax` as `DECIMAL(15,2)`, `l_shipdate` as
`DATE32`, `l_orderkey` / `p_partkey` as `INT64`, string columns as `Utf8View`)
with zero casts or schema workarounds; `morsel_mode=required` admits all 25
natively under `MorselExec` with zero fallbacks and verified DuckDB parity:

- See [sf5_22q_wave4_scorecard.md](sf5_22q_wave4_scorecard.md) for the Wave-4
  post-`#206`/`#207`/`#208` idle-`dduck2` **22/22 SF5 3-way warm-cache
  scorecard** comparing single-config `morsel_pushdown` vs. `stock_default`,
  `stock_pushdown`, and DuckDB 2.0.1 with before/after `#195` attribution and
  `#138` metrics (`#209`).
- See [sf5_22q_final_scorecard.md](sf5_22q_final_scorecard.md) for the earlier
  post-`#193`/`#194` idle-`dduck2` **22/22 SF5 3-way warm-cache scorecard**
  (before vs. after, `bytes_scanned` dynamic join pushdown analysis, and
  tight-memory SF5 spill completion under `RLIMIT_NOFILE=1024`, `#195`).
- See [sf5_22q_scorecard.md](sf5_22q_scorecard.md) for the initial idle-`dduck2`
  **full 22/22 SF5 3-way warm-cache scorecard** (Morsel `required` vs. Stock
  `off` vs. DuckDB 2.0.1) with geometric means and `#138` per-pipeline
  wait/work breakdowns (`#192`).
- See [checkpoint_b_results.md](checkpoint_b_results.md) for the idle-`dduck2`
  **Checkpoint B** SF1 3-way warm + cold scorecard across `q1`, `q3`, `q6`, and
  `q12` with `#138` per-pipeline wait/work breakdowns (`#174`).
- See [sf5_baseline_22q.md](sf5_baseline_22q.md) for the pre-`#191` SF5
  22-query Stock vs. DuckDB baseline and historical Morsel admission matrix
  (`#189`).
- See [checkpoint_a_results.md](checkpoint_a_results.md) for the idle-`dduck2`
  Checkpoint A SF1 3-way scorecard (`#173`).
- See [morsel_aggregate.md](morsel_aggregate.md) for the `#168` `dduck1`
  aggregate note.

Checkpoint B standard queries (`--queries q1,q3,q6,q12`):

- **`q1`**: Filtered grouped 8-aggregate (`SUM`, `AVG`, `COUNT`) + `ORDER BY`
  (`SortSink`) over `lineitem`.
- **`q3`**: 3-way Inner `HashJoin` (`customer ⋈ orders ⋈ lineitem`) + filter +
  `GROUP BY` + `ORDER BY ... LIMIT 10` (`TopNSink`).
- **`q6`**: Filtered ungrouped `SUM(l_extendedprice * l_discount)` over
  `lineitem`.
- **`q12`**: 2-way Inner `HashJoin` (`orders ⋈ lineitem`) + filter +
  conditional `SUM` + `GROUP BY` + `ORDER BY` (`SortSink`).

Additional admitted standard queries (`q5`, `q10`, `q14`, `q19`) and slices:

- **`q5`**: 6-way Inner `HashJoin` + `GROUP BY` + `ORDER BY` (`SortSink`).
- **`q10`**: 4-way Inner `HashJoin` + `GROUP BY` + `ORDER BY ... LIMIT 20`
  (`TopNSink`).
- **`q14`**: Two-pipeline native Inner `HashJoin` (`part` build, filtered
  `lineitem` probe) + conditional `SUM` ratio (`#188`).
- **`q19`**: Two-pipeline native Inner `HashJoin` (`part` build, `lineitem`
  probe) with disjunctive 3-branch `OR` join filter + ungrouped `SUM` (`#188`).
- **`q1_unordered`**: TPC-H Q1 without `ORDER BY`.
- **`q6_count`**:

  ```sql
  SELECT COUNT(*) AS n
  FROM lineitem
  WHERE l_shipdate >= DATE '1994-01-01'
    AND l_shipdate < DATE '1995-01-01'
    AND l_discount BETWEEN 0.05 AND 0.07
    AND l_quantity < 24
  ```

- **`q6_scan`**:

  ```sql
  SELECT l_orderkey, l_extendedprice * l_discount AS rev
  FROM lineitem
  WHERE l_shipdate >= DATE '1994-01-01'
    AND l_shipdate < DATE '1995-01-01'
    AND l_discount BETWEEN 0.05 AND 0.07
    AND l_quantity < 24
  ```

## Controlled Arms, Timing, and Parity Rules

Each timed arm runs in a **fresh process** per query repetition, inheriting the
same CPU affinity (`taskset -c 0-7`) and 8-thread budget (`target_partitions=8`,
Tokio `worker_threads=8` and `max_blocking_threads=8` for DataFusion;
`SET threads = 8` for DuckDB). Arm execution order alternates between odd and
even rounds.

### Scan Profiles and Arm Labels

| Arm                             | Engine     | `morsel_mode` | Scan profile settings                                                                                                                                                             |
| ------------------------------- | ---------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stock_no_pushdown`             | DataFusion | `off`         | `pushdown_filters=false`, `pruning=false`, `enable_page_index=false`, `bloom_filter_on_read=false`, `repartition_file_scans=false`, `collect_statistics=false`                    |
| `morsel_no_pushdown`            | DataFusion | `required`    | Same as `stock_no_pushdown` (exact apples-to-apples scan configuration matching `benchmarks/morsel_parquet`)                                                                      |
| `morsel_pruning`                | DataFusion | `required`    | `pushdown_filters=false`, `repartition_file_scans=false`, `pruning=true`, `enable_page_index=true`, `bloom_filter_on_read=true`, `collect_statistics=true`                        |
| `morsel_pushdown`               | DataFusion | `required`    | `pushdown_filters=true`, `reorder_filters=true`, `repartition_file_scans=false`, `pruning=true`, `enable_page_index=true`, `bloom_filter_on_read=true`, `collect_statistics=true` |
| `stock_default`                 | DataFusion | `off`         | DataFusion defaults (`pushdown_filters=false`, `pruning=true`, `enable_page_index=true`, `repartition_file_scans=true`, `collect_statistics=true`)                                |
| `stock_pushdown`                | DataFusion | `off`         | DataFusion defaults + `pushdown_filters=true`, `reorder_filters=true`                                                                                                             |
| `stock_pushdown_no_repartition` | DataFusion | `off`         | Same as `morsel_pushdown` (`pushdown_filters=true`, `reorder_filters=true`, `repartition_file_scans=false`) for apples-to-apples pushdown comparison                              |
| `duckdb_default`                | DuckDB     | —             | DuckDB defaults (`SET threads = 8`, `read_parquet(...)` with built-in pruning and filter pushdown)                                                                                |

Note on `repartition_file_scans`: `morsel_mode=required` currently rejects
byte-range split Parquet files (`file.range.is_some()`), which `stock_default`
enables via `repartition_file_scans=true`. Therefore `morsel_no_pushdown`,
`morsel_pruning`, and `morsel_pushdown` all set `repartition_file_scans=false`,
while `stock_default` and `stock_pushdown` retain DataFusion's unmodified
defaults.

### Timing Boundary and Warm Page Cache

- **DataFusion (`tpch_runner.rs`)**: Starts the clock right before
  `register_parquet` for only the tables referenced by the query (matching
  DuckDB's lazy view binding), then runs SQL planning, physical planning, and
  drains `execute_stream` at batch size `8,192` without serializing output.
  Reports `register_seconds`, `plan_seconds`, `exec_seconds`,
  `exec_cpu_seconds`, `first_output_seconds`, `first_8192_seconds`, total
  `seconds`, Stock `operator_compute_ms`, and (when `MORSEL_METRICS=1`, the
  default on `measure`) `MorselMetrics` wait/work breakdowns
  (`morsel_decode_work_ms`, `morsel_process_work_ms`, `morsel_ready_wait_ms`,
  `morsel_output_wait_ms`, `morsel_pending_io_ms`, coordinator join/sort/topn
  combine/finalize times, peak queue depths, per-worker `worker_claims` /
  `worker_completions`, and per-pipeline `morsel_pipelines` breakdowns).
- **DuckDB (`harness.py _duckdb-exec`)**: Opens a fresh in-memory connection,
  sets `threads`, preloads `LOAD parquet`, warms the Arrow C-stream bridge with
  `SELECT 1` (so one-time dynamic extension loading does not penalize DuckDB vs
  statically linked DataFusion), registers lazy `CREATE VIEW ... AS SELECT *
FROM read_parquet(...)` views for the queried tables, and starts the clock
  immediately before draining `con.sql(sql).to_arrow_reader(batch_size=8192)`.
- **Warm page cache (`--cache warm`)**: Reads all referenced Parquet files into
  the Linux page cache once before timing and checks via `mincore` that 100% of
  4 KiB pages of the queried tables are resident **before and after** every
  timed trial. Any residency drop aborts the run.
- **Cold page cache (`--cache cold`)**: Evicts all queried Parquet files from
  the Linux page cache via `fsync` + `posix_fadvise(POSIX_FADV_DONTNEED)`
  before each trial (no `sudo` required; requires disk-backed storage, not
  `tmpfs`) and verifies via `mincore` that `0 / total` 4 KiB pages are resident
  before each timed trial.

### Result Parity Against DuckDB Oracle

Before any timing begins, `bench` (and `verify`) transfers full query results via
lossless Arrow IPC streams (`arrow::ipc::writer::StreamWriter` in Rust,
`to_arrow_table()` in DuckDB) and checks exact parity against DuckDB:

- Ordered queries (`q6`, `q1`, `q3`, `q12`, `q6_count`) require exact row order
  and exact values.
- Unordered queries (`q1_unordered`, `q6_scan`) check exact multiset equality
  across all rows (sorted by exact keys, verifying multiplicity and exact
  `DECIMAL` values).
- Cross-engine type/name normalization is restricted to documented cases:
  - `q1` / `q1_unordered` `AVG(DECIMAL(15,2))` and `q8` `mkt_share`: DuckDB
    returns `float64`, while DataFusion returns `DECIMAL(19,6)` /
    `DECIMAL(38,8)`; verified within the Decimal's scale (`abs_tol=10^-scale`,
    `rel_tol=1e-9`).
  - `q14` / `q17` floating-point ratios: `math.isclose(..., rel_tol=1e-9,
abs_tol=1e-9)`.
  - `q12` `SUM(CASE ... THEN 1 ELSE 0 END)`: DuckDB returns `DECIMAL(38,0)`
    while DataFusion returns `INT64`; verified by exact integer/Decimal
    equality.
  - `q7` / `q8` / `q9` `EXTRACT(YEAR FROM ...)`: DuckDB returns `INT64` while
    DataFusion returns `INT32`; verified by exact integer equality.
  - `q18` unaliased `sum(l_quantity)`: DuckDB names the column
    `sum(l_quantity)` while DataFusion qualifies it as
    `sum(lineitem.l_quantity)`; normalized by stripping table qualifiers inside
    unaliased aggregate names.
- Every timed DataFusion trial re-validates the runner git revision, drained row
  count, and physical plan (`DataSourceExec` without `MorselExec` for `off`;
  `MorselExec` + `NativeParquet[lineitem...]`, row-group claims, `CountSink` /
  `count_finalizations`, and `AggregateSink` / `aggregate_finalizations`
  matching the workload for `required`).

## Reproduction Commands

From the repository root on a Linux host with `uv`, `taskset`, and the pinned
Rust toolchain:

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work=$(mktemp -d "${MORSEL_TPCH_WORK_ROOT:-$HOME/.cache}/morsel-tpch.XXXXXX")
revision=$(git rev-parse HEAD)

# 1. Build the standalone runner as a detached Cargo example (no workspace edits).
git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$b/tpch_runner.rs" "$work/pinned/datafusion/core/examples/morsel_tpch_runner.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$revision" CARGO_TARGET_DIR="$work/target" \
    RUSTC_WRAPPER=sccache \
    timeout --kill-after=20s 35m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features \
      --features parquet,sql,datetime_expressions,unicode_expressions \
      --example morsel_tpch_runner
)
# Omit RUSTC_WRAPPER=sccache if sccache is not installed.
runner="$work/target/release-nonlto/examples/morsel_tpch_runner"

# 2. Create the pinned Python environment.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"

# 3. Run the deterministic --sf 0.01 unit + end-to-end parity and plan smoke tests.
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 180s "$py" -B -m unittest discover -s "$b" -v

# 4. Generate deterministic SF1 TPC-H Parquet tables outside Git.
data_dir="$work/tpch_sf1"
timeout 180s "$py" -B "$b/harness.py" generate "$data_dir" \
  --sf 1 --row-group-size 131072

# 5. Run the controlled warm-cache 3-way SF1 benchmark on 8 pinned cores
#    (7-workload SF1 suite; see sf5_baseline_22q.md for the all-22-query SF5 command).
vmstat -y 1 3
timeout --kill-after=10s 300s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q6,q1_unordered,q1,q3,q12,q6_count,q6_scan \
  > "$work/sf1_warm.jsonl"
vmstat -y 1 3

# 6. Print summary tables.
"$py" -B "$b/harness.py" summarize "$work/sf1_warm.jsonl"

# 7. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```

## SF1 Warm-Cache Baseline: `rob-tandy-dduck2`, 2026-11-22 UTC

### Environment and Controls

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 vCPUs (1 thread per core), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  `--threads 8`, batch size `8,192`.
- **Host idle check:** `vmstat -y 1 3` immediately before the run showed
  83–89% host CPU idle; immediately after showed 90–92% idle (no concurrent
  builds; workspace was not exclusively reserved).
- **Build & versions:**
  - DataFusion revision: `4dbbaf33d0a446d570765fa3cda605936c67ac93`
    (`release-nonlto`, `--locked --no-default-features --features parquet,sql`)
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`,
    Clang `18.1.3`, Tokio `1.54.1`
  - Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`
- **Storage & page-cache verification:** SF1 Parquet dataset (266,865,655 bytes,
  35 files, 79 row groups) generated under `$HOME/.cache` on local EC2 NVMe
  instance storage (`/dev/mapper/vg_ws_ephemeral-volume1p1`, Btrfs,
  `compress=zstd:1`). All 168 measured trials started and ended with **100% of
  4 KiB file pages resident** via `mincore`: `43,256/43,256` pages for
  `lineitem`-only queries (`q6`, `q1`, `q6_count`, `q6_scan`), `54,144/54,144`
  for `q12`, and `56,392/56,392` for `q3`.
- **Protocol:** 1 unmeasured warmup round and 7 measured rounds per query/arm
  with alternating arm order. Timings below are medians in milliseconds with
  observed min–max in parentheses.

### 1. Standard TPC-H SF1 Roadmap Queries (Stock DataFusion vs DuckDB)

Morsel DataFusion (`morsel_mode=required`) is **not yet admitted** on `q6`,
`q1`, `q3`, and `q12` (verified rejected at planning time; see rejection reasons
above).

| Query | Rows | Arm (`mode` / settings)                           | Median total ms (min–max) | Median exec ms (min–max) | `bytes_scanned` | Morsel status    |
| ----- | ---: | ------------------------------------------------- | ------------------------: | -----------------------: | --------------: | ---------------- |
| `q6`  |    1 | `stock_default` (`off`, `pushdown_filters=false`) |         98.8 (96.7–100.4) |         89.0 (86.6–90.7) |      47,682,742 | Not yet admitted |
| `q6`  |    1 | `stock_pushdown` (`off`, `pushdown_filters=true`) |       134.4 (131.9–137.1) |      124.8 (122.1–127.6) |      47,682,742 | Not yet admitted |
| `q6`  |    1 | `duckdb_default` (DuckDB defaults)                |          59.1 (52.3–61.7) |                        — |               — | Not yet admitted |
| `q1`  |    4 | `stock_default` (`off`, `pushdown_filters=false`) |       209.5 (203.4–213.1) |      198.3 (192.1–201.7) |      51,899,102 | Not yet admitted |
| `q1`  |    4 | `stock_pushdown` (`off`, `pushdown_filters=true`) |       195.8 (191.6–197.2) |      184.9 (180.7–186.2) |      51,899,102 | Not yet admitted |
| `q1`  |    4 | `duckdb_default` (DuckDB defaults)                |       111.3 (106.9–117.6) |                        — |               — | Not yet admitted |
| `q3`  |   10 | `stock_default` (`off`, `pushdown_filters=false`) |       128.9 (125.1–129.9) |      115.6 (111.4–116.5) |      65,276,113 | Not yet admitted |
| `q3`  |   10 | `stock_pushdown` (`off`, `pushdown_filters=true`) |         99.5 (98.4–105.6) |         86.5 (85.1–92.2) |      65,276,113 | Not yet admitted |
| `q3`  |   10 | `duckdb_default` (DuckDB defaults)                |         96.9 (94.7–103.8) |                        — |               — | Not yet admitted |
| `q12` |    2 | `stock_default` (`off`, `pushdown_filters=false`) |          65.9 (63.8–69.5) |         53.0 (51.1–56.4) |      44,405,916 | Not yet admitted |
| `q12` |    2 | `stock_pushdown` (`off`, `pushdown_filters=true`) |          83.5 (80.0–87.3) |         70.4 (67.0–74.4) |      44,405,916 | Not yet admitted |
| `q12` |    2 | `duckdb_default` (DuckDB defaults)                |          64.3 (61.8–74.6) |                        — |               — | Not yet admitted |

Registration (`register_seconds`) + planning (`plan_seconds`) accounted for
9.7–13.4 ms across DataFusion roadmap runs (`4.4–5.9` ms registration,
`5.0–8.0` ms planning).

### 2. Live 3-Way SF1 Comparison on Admitted Morsel Slices (`q6_count`, `q6_scan`)

Both `q6_count` (1 output row, `n = 114,160`) and `q6_scan` (114,160 output
rows, 1.90% selectivity over 6,001,215 `lineitem` rows) matched DuckDB exactly
across all six arms. Both Morsel arms executed natively as
`MorselExec: workers=8, NativeParquet[lineitem]` and claimed all 48 row groups
(`count_finalizations=1` for `q6_count`, `0` for `q6_scan`), scanning the exact
same `bytes_scanned` as Stock DataFusion (`16,509,334` bytes for `q6_count`;
`57,252,505` bytes for `q6_scan`).

| Query      | Arm (`mode` / profile)                                                       | Batches | Morsels claimed | First batch ms | First 8,192 rows ms | Median exec ms | Median total ms (min–max) |
| ---------- | ---------------------------------------------------------------------------- | ------: | --------------: | -------------: | ------------------: | -------------: | ------------------------: |
| `q6_count` | `stock_no_pushdown` (`off`, `pushdown_filters=false`, `pruning=false`)       |       1 |               0 |           49.5 |                   — |           41.2 |          49.5 (46.6–50.0) |
| `q6_count` | `morsel_no_pushdown` (`required`, `pushdown_filters=false`, `pruning=false`) |       1 |              48 |           47.3 |                   — |           40.0 |          47.3 (45.5–48.5) |
| `q6_count` | `morsel_pruning` (`required`, `pushdown_filters=false`, `pruning=true`)      |       1 |              48 |           49.2 |                   — |           40.9 |          49.2 (45.2–50.5) |
| `q6_count` | `stock_default` (`off`, default: `pushdown_filters=false`, `pruning=true`)   |       1 |               0 |           56.4 |                   — |           47.1 |          56.4 (54.5–57.5) |
| `q6_count` | `stock_pushdown` (`off`, `pushdown_filters=true`, `reorder_filters=true`)    |       1 |               0 |           68.7 |                   — |           59.4 |          68.7 (67.0–74.4) |
| `q6_count` | `duckdb_default` (DuckDB defaults)                                           |       1 |               — |           24.6 |                   — |              — |          24.8 (23.3–25.9) |
| `q6_scan`  | `stock_no_pushdown` (`off`, `pushdown_filters=false`, `pruning=false`)       |      16 |               0 |           55.0 |                55.0 |           80.0 |         87.9 (85.1–109.7) |
| `q6_scan`  | `morsel_no_pushdown` (`required`, `pushdown_filters=false`, `pruning=false`) |     736 |              48 |           15.8 |                23.8 |           99.1 |       105.8 (103.6–114.3) |
| `q6_scan`  | `morsel_pruning` (`required`, `pushdown_filters=false`, `pruning=true`)      |     736 |              48 |           16.1 |                24.6 |          101.1 |       109.3 (105.8–120.4) |
| `q6_scan`  | `stock_default` (`off`, default: `pushdown_filters=false`, `pruning=true`)   |   17–18 |               0 |           57.0 |                57.0 |          100.3 |       108.9 (105.8–112.5) |
| `q6_scan`  | `stock_pushdown` (`off`, `pushdown_filters=true`, `reorder_filters=true`)    |      48 |               0 |           31.7 |                34.3 |          139.4 |       148.6 (145.5–162.8) |
| `q6_scan`  | `duckdb_default` (DuckDB defaults)                                           |      48 |               — |           29.2 |                36.4 |              — |          65.4 (63.6–79.5) |

### Key Observations and Limitations

- **Morsel vs Stock DataFusion on `q6_count`**: With matching
  `pushdown_filters=false`, `morsel_no_pushdown` (47.3 ms total / 40.0 ms exec)
  slightly edges out `stock_no_pushdown` (49.5 ms total / 41.2 ms exec) and is
  1.15–1.19× faster than `stock_default` (56.4 ms). DuckDB remains 1.91× faster
  at 24.8 ms.
- **Morsel vs Stock DataFusion on `q6_scan`**: Under `pushdown_filters=false`,
  `morsel_no_pushdown` drains in 105.8 ms (1.20× `stock_no_pushdown`'s 87.9 ms,
  comparable to `stock_default`'s 108.9 ms, and 1.62× DuckDB's 65.4 ms). Because
  `MorselExec` does not coalesce filtered output batches (emitting 736 batches
  vs Stock's 16–18 coalesced batches), it reaches both the first batch (15.8 ms)
  and the first 8,192 output rows (23.8 ms) earlier than both Stock DataFusion
  (55.0–57.0 ms) and DuckDB (36.4 ms).
- **Row-group pruning and `pushdown_filters=true` on TPC-H `lineitem`**: Because
  standard TPC-H `lineitem` is ordered by `(l_orderkey, l_linenumber)`, date,
  discount, and quantity values span the full domain within every row group; all
  48 row groups pass min/max statistics pruning (`morsel_pruning` claims 48/48).
  Enabling `pushdown_filters=true` helps join/dense-filter queries (`q3`: 99.5
  vs 128.9 ms, nearly matching DuckDB's 96.9 ms; `q1`: 195.8 vs 209.5 ms), but
  adds row-selection overhead on `q6`/`q6_count`/`q6_scan` (e.g., `q6_scan`:
  148.6 vs 108.9 ms).
- **Limitations**:
  - Warm page-cache results measure in-memory Parquet decode, filter, join, and
    aggregation speed, not cold storage I/O.
  - DuckDB timings reflect DuckDB's C++ Arrow reader drain
    (`to_arrow_reader(8192)`), which includes Parquet footer reads and Arrow
    conversion; DataFusion's `total ms` includes `register_parquet` + planning +
    `execute_stream` drain without Arrow IPC serialization.
  - `q6` and `q1_unordered` are now natively admitted in `morsel_mode=required`
    (see [morsel_aggregate.md](morsel_aggregate.md)); `q1` (with `ORDER BY`),
    `q3`, and `q12` remain roadmap queries until `HashJoin` and `Sort`/`TopN`
    are implemented in `MorselQueryPlanner`.
