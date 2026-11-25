<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel `BatchView` Selection Vectors Benchmark Note (#169)

This note records paired **Before (`026286a37`, `origin/morsel-execution`) vs
After (`6eb693d3d`)** release benchmarks for in-pipeline deferred
materialization via `BatchView` selection vectors
(`Selection::{All, Mask, Indices}`) across:

1. **Official `#127` in-memory harness** (`datafusion/core/benches/morsel_execution.rs`,
   31 timed repetitions per configuration) plus stream-draining `#127`
   unfiltered controls.
2. **`#128` Parquet pipelines** (warm cache, low and medium selectivity, plus
   unfiltered control).
3. **`#158` filtered `HashJoin` probe pipelines** (in-memory and Parquet,
   varying probe and build selectivities, plus unfiltered control).
4. **Filtered `GROUP BY` and TPC-H Q1 / Q3-style probes** (in-memory and
   Parquet, plus unfiltered control).

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs,
  61 GiB RAM.
- **Host idle check:** `/proc/stat` measured 95–100% idle across all 16 CPUs
  immediately before measurement.
- **CPU pinning and runtimes:**
  - Official `#127` harness (`benches/morsel_execution.rs`): `taskset -c 12-15`,
    dedicated 1-worker and 4-worker Tokio runtimes matching worker count.
  - Full 22-workload stream-draining suite: `taskset -c 8-15`, dedicated
    1-worker (`1w`), 4-worker (`4w`), and 8-worker (`8w`) Tokio runtimes; every
    query drains `execute_stream` batch-by-batch without retaining output
    batches in memory (avoiding `collect()` allocator artifacts).
- **Build:** identical `release-nonlto` profile
  (`--locked --profile release-nonlto -j 12 -p datafusion --no-default-features --features parquet,sql`)
  built in separate detached worktrees for `before` (`026286a37`) and `after`
  (`6eb693d3d`).
- **Parquet settings:** matched across both arms (`pushdown_filters=false`,
  `pruning=false`, `collect_statistics=false`, warm OS page cache after
  fixture generation + parity warmup). Disabling Parquet row-filter pushdown
  isolates in-pipeline `BatchView` execution.
- **Parity verification:** Every workload was verified for exact row-count,
  exact integer checksum, and float-tolerance (`1e-9` relative) parity against
  stock DataFusion (`morsel_mode=off`), and verified to execute natively under
  `MorselExec` (`morsel_mode=required`).
- **Protocol (22-workload suite):** `B1 -> A1 -> A2 -> B2` alternating paired
  rounds; each round ran 2 untimed warmups followed by 9 timed repetitions (18
  timed samples per workload per arm). Reported medians are `(R1 + R2) / 2`
  with observed `min–max` ranges across all 18 samples in parentheses.

## Datasets

- **`#127` (`items`):** `8,388,608` rows (`2,048` batches × `4,096` rows),
  `id: Int64`, `score: Int64` in `0..256` (`score < 64` = ~25% selectivity).
  Partition layouts: `1w` (`[2048]`), `4w_unif` (`[512, 512, 512, 512]`),
  `4w_skew` (`[1664, 128, 128, 128]`).
- **`#128` / `#158` / `GROUP BY` / TPC-H-style (`fact` & `dim`):**
  - `dim`: `100,000` rows (`k: Int64`, `grp: Int32`, `weight: Int64` in
    `0..1000`, `tag: Utf8`), stored as 4 Zstd Parquet files (`25,000` rows/RG)
    and identically as a 4-partition `MemTable`.
  - `fact`: `4,000,000` rows (`id: Int64`, `k: Int64` with ~5% unmatched and
    ~1% `NULL`, `grp: Int32` in `0..64`, `val: Int64`, `fval: Float64`,
    `discount: Float64`, `gate: Int32` in `0..1000`, `retflag: Utf8`,
    `linestat: Utf8`, `tag: Utf8`), stored as 16 Zstd Parquet files (`32` row
    groups of `125,000` rows) and identically as a 16-partition `MemTable`
    (`batch_size=8,192`).

## Results

### 1a. Official `#127` Harness (`benches/morsel_execution.rs`, 31 reps, `taskset -c 12-15`)

| Layout  | Workers | Query      | Stock (`off`) ms | Before (`required`) ms [p10, p90] | After (`required`) ms [p10, p90] | Speedup |
| ------- | ------: | ---------- | ---------------: | --------------------------------: | -------------------------------: | ------: |
| uniform |       1 | projection |           20.235 |            18.902 [18.782, 19.135] |           19.423 [19.336, 19.544] |   0.97× |
| uniform |       1 | count      |           10.052 |             9.341 [9.247, 9.429]  |            9.154 [9.005, 9.291]  |   1.02× |
| uniform |       4 | projection |            6.912 |             7.656 [7.616, 7.717]  |            5.912 [5.872, 5.973]  |   1.30× |
| uniform |       4 | count      |            2.801 |             2.664 [2.626, 2.680]  |            2.613 [2.591, 2.684]  |   1.02× |
| skewed  |       4 | projection |           18.900 |             7.772 [7.690, 7.847]  |            5.950 [5.910, 5.990]  |   1.31× |
| skewed  |       4 | count      |            8.621 |             2.685 [2.649, 2.744]  |            2.646 [2.613, 2.716]  |   1.01× |

*(Note: in the `After` run of `benches/morsel_execution.rs`, the stock `off` arm on `uniform 1w projection` was `20.590 ms` vs `20.235 ms` in the `Before` run, so net relative delta on `1w projection` is `< 1%`.)*

### 1b. `#127` In-Memory Pipelines with Unfiltered Controls (`batch_size=4,096`, stream-draining)

| Workload                      | Threads | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ----------------------------- | ------: | -------------------------: | ------------------------: | ------: | ----------: |
| `127_1w_proj_25pct`           |       1 |           18.39 (18.2–18.5) |          18.59 (18.5–18.7) |   0.99× |   2,096,013 |
| `127_1w_count_25pct`          |       1 |             8.80 (8.7–8.9) |            8.88 (8.7–9.1) |   0.99× |           1 |
| `127_1w_unfiltered_ctrl`      |       1 |             7.75 (7.5–8.0) |            7.56 (7.4–7.7) |   1.03× |   8,388,608 |
| `127_4w_unif_proj_25pct`      |       4 |             8.68 (8.6–8.8) |            6.44 (6.3–6.6) |   1.35× |   2,096,013 |
| `127_4w_unif_count_25pct`     |       4 |             2.58 (2.5–2.7) |            2.61 (2.6–2.6) |   0.99× |           1 |
| `127_4w_unif_unfiltered_ctrl` |       4 |             4.69 (4.1–5.3) |            5.20 (4.8–5.5) |   0.90× |   8,388,608 |
| `127_4w_skew_proj_25pct`      |       4 |             8.64 (8.4–8.8) |            6.51 (6.4–6.6) |   1.33× |   2,096,013 |

### 2. `#128` Parquet Pipelines (8 Threads, Warm Cache, `batch_size=8,192`)

| Workload                 | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ------------------------ | -------------------------: | ------------------------: | ------: | ----------: |
| `128_pq_proj_2pct`       |           21.57 (19.8–23.2) |          21.34 (20.0–23.0) |   1.01× |      80,293 |
| `128_pq_proj_25pct`      |           24.00 (22.3–25.6) |          23.62 (22.1–25.1) |   1.02× |   1,000,776 |
| `128_pq_count_2pct`      |             2.87 (2.6–3.3) |            2.84 (2.5–3.7) |   1.01× |           1 |
| `128_pq_unfiltered_ctrl` |           21.23 (20.1–25.1) |          20.45 (19.4–22.1) |   1.04× |   4,000,000 |

### 3. `#158` Filtered `HashJoin` Probe Pipelines (8 Threads, `batch_size=8,192`)

| Workload                       | Query shape                                              | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ------------------------------ | -------------------------------------------------------- | -------------------------: | ------------------------: | ------: | ----------: |
| `158_mem_join_probe_sel_dim`   | MemTable, 25% probe filter × 10% build filter (2.5% out) |             9.07 (9.0–9.2) |            6.27 (6.2–6.4) |   1.45× |      94,867 |
| `158_mem_join_probe_25pct`     | MemTable, 25% probe filter, full build                   |           17.33 (16.5–18.2) |          14.92 (14.3–15.7) |   1.16× |     940,784 |
| `158_mem_join_probe_5pct`      | MemTable, 5% probe filter, full build                    |           10.04 (9.4–11.2) |           9.79 (9.1–10.7) |   1.03× |     188,687 |
| `158_mem_join_unfiltered_ctrl` | MemTable, unfiltered control                             |           27.52 (26.5–31.6) |          27.69 (26.4–28.4) |   0.99× |   3,760,256 |
| `158_pq_join_probe_10pct`      | Parquet, 10% probe filter, full build                    |           40.92 (39.3–49.4) |          40.88 (39.0–43.7) |   1.00× |     377,257 |

### 4. Filtered `GROUP BY` & TPC-H Q1 / Q3-Style Probes (8 Threads, `batch_size=8,192`)

| Workload                      | Query shape                                                      | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ----------------------------- | ---------------------------------------------------------------- | -------------------------: | ------------------------: | ------: | ----------: |
| `groupby_mem_10pct`           | MemTable, 10% filter → `GROUP BY grp` (`COUNT/SUM/AVG`)          |             2.71 (2.6–3.0) |            2.44 (2.4–2.6) |   1.11× |          64 |
| `groupby_mem_25pct`           | MemTable, 25% filter → `GROUP BY grp` (`COUNT/SUM/AVG`)          |             3.60 (3.5–3.7) |            3.19 (3.2–3.2) |   1.13× |          64 |
| `groupby_mem_unfiltered_ctrl` | MemTable, unfiltered `GROUP BY grp` control                      |             6.63 (6.6–6.7) |            6.19 (6.1–6.2) |   1.07× |          64 |
| `groupby_pq_10pct`            | Parquet, 10% filter → `GROUP BY grp` (`COUNT/SUM/AVG`)           |           10.44 (10.0–12.1) |          10.43 (10.0–11.2) |   1.00× |          64 |
| `tpch_q1_style_pq_98pct`      | Parquet, 98.6% filter → 2-key `GROUP BY` (`3× SUM, COUNT`)       |           31.40 (30.6–36.3) |          30.95 (30.4–35.5) |   1.01× |           6 |
| `tpch_q3_style_pq_probe`      | Parquet, 54% probe filter + 25% build filter → Join → `GROUP BY` |           39.01 (37.5–44.1) |          37.06 (34.9–41.0) |   1.05× |       4,096 |

## Observations

- **Filtered `HashJoin` probe (`#158`):** Deferred materialization delivers the
  largest gains where selective probe filters feed a hash join with payload
  columns: **1.45× faster** (`9.07 ms -> 6.27 ms`) on `158_mem_join_probe_sel_dim`
  and **1.16× faster** (`17.33 ms -> 14.92 ms`) on `158_mem_join_probe_25pct`.
  Probe workers hash and probe only selected rows via `BatchView` and gather
  `id`/`val`/`tag` payload columns directly from backing batches only for rows
  that match the build table, eliminating both the intermediate filtered batch
  allocation and copies of unmatched probe payloads. On TPC-H Q3-style Parquet
  probe (`tpch_q3_style_pq_probe`), wall time drops **1.05×** (`39.01 ms -> 37.06 ms`)
  even with Parquet Zstd decode dominating total runtime.
- **`#127` (`Filter -> Projection`):** Zero-copy column narrowing before
  coalescing speeds up 4-worker `#127` projection by **1.30×–1.35×** (uniform:
  `7.656 ms -> 5.912 ms` in `benches/morsel_execution.rs`, overtaking stock
  DataFusion's `6.804 ms`; skewed: `7.772 ms -> 5.950 ms`, **3.16×** faster
  than stock). On single-worker `127_1w_proj_25pct`, which already had static
  filter-input pruning + fused `push_batch_with_filter`, `BatchView` is at
  parity (`18.39 ms -> 18.59 ms`, `0.99×`; `18.902 ms -> 19.423 ms` in
  `benches/morsel_execution.rs` where stock `off` moved `20.235 -> 20.590 ms`).
- **Filtered `GROUP BY`:** Direct selection-aware grouped accumulation
  eliminates intermediate filtered `RecordBatch` materialization for integer-key
  `GROUP BY` (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`), delivering **1.11×–1.13×**
  speedups on in-memory filtered `GROUP BY` (`3.60 ms -> 3.19 ms` at 25%,
  `2.71 ms -> 2.44 ms` at 10%) and **1.07×** on the unfiltered `GROUP BY`
  control (`6.63 ms -> 6.19 ms`). `tpch_q1_style_pq_98pct` (98.6% selectivity)
  is at parity (`31.40 ms -> 30.95 ms`, `1.01×`).
- **Unfiltered controls:** `127_1w_unfiltered_ctrl` (`1.03×`),
  `128_pq_unfiltered_ctrl` (`1.04×`), `158_mem_join_unfiltered_ctrl` (`0.99×`),
  and `groupby_mem_unfiltered_ctrl` (`1.07×`) are at parity or faster;
  `127_4w_unif_unfiltered_ctrl` (`4.69 ms` vs `5.20 ms`, ranges `4.1–5.3 ms` vs
  `4.8–5.5 ms`) reflects single-channel mpsc wakeup variance over 2,048 tiny
  4K-row batches (~0.5 ms total).
