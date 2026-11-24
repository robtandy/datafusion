<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel `BatchView` Selection Vectors Benchmark Note (#169)

This note records a paired **Before (`f044eda78`) vs After (`8f21aee9f`)**
release benchmark for in-pipeline deferred materialization via `BatchView`
selection vectors (`Selection::{All, Mask, Indices}`) across:

1. **`#127` in-memory pipelines** (1-worker uniform, 4-worker uniform, 4-worker
   skewed, plus unfiltered controls).
2. **`#128` Parquet pipelines** (warm cache, low and medium selectivity, plus
   unfiltered control).
3. **`#158` filtered `HashJoin` probe pipelines** (in-memory and Parquet,
   varying probe and build selectivities, plus unfiltered control).
4. **Filtered `GROUP BY` and TPC-H Q1 / Q3-style probes** (in-memory and
   Parquet, plus unfiltered control).

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs,
  61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 8-15` (8 dedicated idle logical CPUs;
  CPUs 8–15 measured 88–95% idle via `/proc/stat` prior to measurement).
- **Build:** identical `release-nonlto` profile
  (`--locked --profile release-nonlto -j 12 -p datafusion --no-default-features --features parquet,sql`)
  built in separate detached worktrees for `before` (`f044eda78d028a4088235ac923f555a832d843bf`)
  and `after` (`8f21aee9f`).
- **Parquet settings:** matched across both arms (`pushdown_filters=false`,
  `pruning=false`, `collect_statistics=false`, warm OS page cache after
  fixture generation + parity warmup). Disabling Parquet row-filter pushdown
  isolates in-pipeline `BatchView` execution.
- **Parity verification:** Every single workload was verified for exact
  row-count, exact integer checksum, and float-tolerance (`1e-9` relative)
  parity against stock DataFusion (`morsel_mode=off`), and verified to execute
  natively under `MorselExec` (`morsel_mode=required`).
- **Protocol:** `B1 -> A1 -> A2 -> B2` alternating paired rounds; each round
  ran 2 untimed warmups followed by 9 timed repetitions (18 timed samples per
  workload per arm). Medians below are the average of the two paired round
  medians (`(R1 + R2) / 2`), with full observed `min–max` ranges in parentheses.

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

### 1. `#127` In-Memory Pipelines (`batch_size=4,096`)

| Workload                      | Threads | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ----------------------------- | ------: | -------------------------: | ------------------------: | ------: | ----------: |
| `127_1w_proj_25pct`           |       1 |           21.99 (19.9–24.6) |          24.59 (20.6–26.0) |   0.89× |   2,096,013 |
| `127_1w_count_25pct`          |       1 |           25.49 (24.1–26.9) |          25.38 (25.1–25.7) |   1.00× |           1 |
| `127_1w_unfiltered_ctrl`      |       1 |           38.88 (37.6–40.4) |          38.99 (38.0–40.0) |   1.00× |   8,388,608 |
| `127_4w_unif_proj_25pct`      |       4 |           11.99 (10.6–12.7) |           10.51 (9.9–11.2) |   1.14× |   2,096,013 |
| `127_4w_unif_count_25pct`     |       4 |             3.27 (3.2–3.5) |             3.18 (3.1–3.3) |   1.03× |           1 |
| `127_4w_unif_unfiltered_ctrl` |       4 |            8.05 (5.2–11.6) |            6.89 (4.8–13.4) |   1.17× |   8,388,608 |
| `127_4w_skew_proj_25pct`      |       4 |           11.73 (10.9–12.2) |            8.87 (8.3–9.3) |   1.32× |   2,096,013 |

### 2. `#128` Parquet Pipelines (8 Threads, Warm Cache, `batch_size=8,192`)

| Workload                 | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ------------------------ | -------------------------: | ------------------------: | ------: | ----------: |
| `128_pq_proj_2pct`       |           21.06 (18.6–22.8) |          20.27 (19.6–21.5) |   1.04× |      80,293 |
| `128_pq_proj_25pct`      |           23.27 (21.9–26.5) |          22.76 (21.4–24.4) |   1.02× |   1,000,776 |
| `128_pq_count_2pct`      |             3.19 (2.7–4.6) |            2.83 (2.6–3.6) |   1.13× |           1 |
| `128_pq_unfiltered_ctrl` |           25.10 (23.8–29.1) |          23.36 (22.3–25.3) |   1.07× |   4,000,000 |

### 3. `#158` Filtered `HashJoin` Probe Pipelines (8 Threads, `batch_size=8,192`)

| Workload                       | Query shape                                             | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ------------------------------ | ------------------------------------------------------- | -------------------------: | ------------------------: | ------: | ----------: |
| `158_mem_join_probe_sel_dim`   | MemTable, 25% probe filter × 10% build filter (2.5% out) |             9.14 (9.1–9.2) |            6.18 (6.1–6.4) |   1.48× |      94,867 |
| `158_mem_join_probe_25pct`     | MemTable, 25% probe filter, full build                  |           18.25 (17.3–20.0) |          15.09 (14.2–17.1) |   1.21× |     940,784 |
| `158_mem_join_probe_5pct`      | MemTable, 5% probe filter, full build                   |           10.38 (9.8–11.8) |           9.62 (9.0–10.9) |   1.08× |     188,687 |
| `158_mem_join_unfiltered_ctrl` | MemTable, unfiltered control                            |           29.30 (27.8–31.2) |          28.02 (26.5–29.2) |   1.05× |   3,760,256 |
| `158_pq_join_probe_10pct`      | Parquet, 10% probe filter, full build                   |           42.11 (39.9–48.3) |          40.10 (38.6–42.7) |   1.05× |     377,257 |

### 4. Filtered `GROUP BY` & TPC-H Q1 / Q3-Style Probes (8 Threads, `batch_size=8,192`)

| Workload                      | Query shape                                                    | Before median ms (min–max) | After median ms (min–max) | Speedup | Output rows |
| ----------------------------- | -------------------------------------------------------------- | -------------------------: | ------------------------: | ------: | ----------: |
| `groupby_mem_10pct`           | MemTable, 10% filter → `GROUP BY grp` (`COUNT/SUM/AVG`)        |             2.68 (2.6–2.8) |            2.43 (2.4–2.5) |   1.10× |          64 |
| `groupby_mem_25pct`           | MemTable, 25% filter → `GROUP BY grp` (`COUNT/SUM/AVG`)        |             3.62 (3.6–3.7) |            3.12 (3.1–3.2) |   1.16× |          64 |
| `groupby_mem_unfiltered_ctrl` | MemTable, unfiltered `GROUP BY grp` control                    |             6.62 (6.5–6.7) |            6.15 (6.1–6.2) |   1.08× |          64 |
| `groupby_pq_10pct`            | Parquet, 10% filter → `GROUP BY grp` (`COUNT/SUM/AVG`)         |           11.36 (10.3–13.0) |          10.38 (9.6–10.9) |   1.09× |          64 |
| `tpch_q1_style_pq_98pct`      | Parquet, 98.6% filter → 2-key `GROUP BY` (`3× SUM, COUNT`)     |           31.11 (30.3–35.9) |          32.74 (30.7–36.1) |   0.95× |           6 |
| `tpch_q3_style_pq_probe`      | Parquet, 54% probe filter + 25% build filter → Join → `GROUP BY` |           39.38 (37.9–46.0) |          36.03 (34.8–37.6) |   1.09× |       4,096 |

## Observations

- **Filtered `HashJoin` probe (`#158`):** Deferred materialization delivers the
  largest gains where selective probe filters feed a hash join with payload
  columns: **1.48× faster** (`9.14 ms -> 6.18 ms`) on `158_mem_join_probe_sel_dim`
  and **1.21× faster** (`18.25 ms -> 15.09 ms`) on `158_mem_join_probe_25pct`.
  Probe workers hash and probe only selected rows via `BatchView` and gather
  `id`/`val`/`tag` payload columns directly from backing batches only for rows
  that match the build table, eliminating both the intermediate filtered batch
  allocation and copies of unmatched probe payloads. On TPC-H Q3-style Parquet
  probe (`tpch_q3_style_pq_probe`), wall time drops **1.09×** (`39.38 ms -> 36.03 ms`)
  even with Parquet decode dominating total runtime.
- **Filtered `GROUP BY`:** Direct selection-aware grouped accumulation
  eliminates intermediate filtered `RecordBatch` materialization for integer-key
  `GROUP BY` (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`), delivering **1.10×–1.16×**
  speedups on in-memory filtered `GROUP BY` (`3.62 ms -> 3.12 ms` at 25%,
  `2.68 ms -> 2.43 ms` at 10%) and **1.09×** on Parquet (`11.36 ms -> 10.38 ms`),
  while also speeding up the unfiltered `GROUP BY` control by **1.08×**
  (`6.62 ms -> 6.15 ms`).
- **`#127` & `#128` (`Filter -> Projection`):** Zero-copy column narrowing
  before coalescing speeds up 4-worker `#127` by **1.14×** (uniform) and
  **1.32×** (skewed), and improves `#128` Parquet by **1.02×–1.13×**. On
  single-worker `127_1w_proj_25pct`, which already had pre-existing filter
  pruning + fused `push_batch_with_filter`, `BatchView` adds ~2.6 ms over 2,048
  batches (`21.99 ms -> 24.59 ms`, min `19.9 -> 20.6 ms`) from eager bit-count
  normalization per batch. On `tpch_q1_style_pq_98pct` (98.6% selectivity),
  observed ranges overlap (`30.3–35.9 ms` vs `30.7–36.1 ms`) because almost all
  rows survive into the multi-expression `Projection` before the 2-key string
  `GroupValues` aggregate.
- **Unfiltered controls:** Every unfiltered control (`127_1w_unfiltered_ctrl`,
  `127_4w_unif_unfiltered_ctrl`, `128_pq_unfiltered_ctrl`,
  `158_mem_join_unfiltered_ctrl`, `groupby_mem_unfiltered_ctrl`) shows **zero
  regression** (`1.00×–1.17×`).
