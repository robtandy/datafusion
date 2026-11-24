<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF1 Checkpoint A Scorecard: Stock DataFusion vs. Morsel DataFusion vs. DuckDB (#173)

This scorecard records the controlled idle-host **TPC-H Checkpoint A** 3-way
comparison on standard **TPC-H SF1 Parquet** (`lineitem`: 6,001,215 rows,
16 Zstd files, 48 row groups — 32 full 131,072-row groups and 16 tail groups of
~112,932 rows, 3 per file — totaling 177,150,918 bytes) on `rob-tandy-dduck2`
after `#149` (worker-local coalescing), `#150` (Parquet `RowFilter` pushdown),
`#166` (direct mask-bit `COUNT(*)` + filter-column pruning), `#167`
(intra-row-group Parquet splitting), `#140` (grouped `COUNT(*)`), and `#168`
(multi-aggregate `GROUP BY` + multi-column/string keys).

This scorecard does not replace the pinned pre-`#150` baseline in
[README.md](README.md#sf1-warm-cache-baseline-rob-tandy-dduck2-2026-11-22-utc).

## Executive Summary

All four Checkpoint A queries (`q6`, `q1_unordered`, `q6_count`, `q6_scan`)
passed exact lossless Arrow IPC parity against **DuckDB 2.0.1** and executed
natively under `MorselExec: workers=8, NativeParquet[lineitem...]` with
`morsel_mode=required` (zero fallback, all 48 row groups claimed, exact
`bytes_scanned` parity with Stock DataFusion).

| Query          |         Best Morsel (`required`)\* | Stock Default (`off`) |                  Best Stock (`off`, tuned) | DuckDB 2.0.1 |    Morsel vs. Stock Default | Morsel vs. Best Stock | Morsel vs. DuckDB |
| -------------- | ---------------------------------: | --------------------: | -----------------------------------------: | -----------: | --------------------------: | --------------------: | ----------------: |
| `q6`           | **88.3 ms** (`morsel_no_pushdown`) |              102.4 ms |              87.7 ms (`stock_no_pushdown`) |      59.3 ms | **1.16× faster (−14.1 ms)** |       0.99× (+0.6 ms) |  1.49× (+29.0 ms) |
| `q1_unordered` |   **170.4 ms** (`morsel_pushdown`) |              218.4 ms | 171.1 ms (`stock_pushdown_no_repartition`) |     112.7 ms | **1.28× faster (−48.0 ms)** |   **1.00× (−0.7 ms)** |  1.51× (+57.7 ms) |
| `q6_count`     | **48.3 ms** (`morsel_no_pushdown`) |               56.9 ms |              49.2 ms (`stock_no_pushdown`) |      26.6 ms |  **1.18× faster (−8.6 ms)** |   **1.02× (−0.9 ms)** |  1.82× (+21.7 ms) |
| `q6_scan`      | **97.6 ms** (`morsel_no_pushdown`) |              111.9 ms |              94.0 ms (`stock_no_pushdown`) |      67.2 ms | **1.15× faster (−14.3 ms)** |       0.96× (+3.6 ms) |  1.45× (+30.4 ms) |

\*Morsel timings in the main 224-trial run were recorded with `MORSEL_METRICS=1`
enabled so every trial captures wait/work breakdowns; a separate
(non-interleaved) `MORSEL_METRICS=0` follow-up run was `0–3 ms` faster on most
arms (`q6`: `86.0 ms`, `q1_unordered`: `166.9 ms`, `q6_count`: `48.1 ms`,
`q6_scan`: `96.5 ms`), which is comparable to the `+0.6 ms` and `+3.6 ms`
"Morsel vs. Best Stock" differences on `q6` and `q6_scan`.

Key takeaways:

- **Morsel beats unmodified Stock DataFusion (`stock_default` and
  `stock_pushdown`) by 1.15–1.28× on all four queries** (`−8.6` to `−48.0 ms`
  vs. `stock_default`; `−7.8` to `−32.7 ms` for `morsel_pushdown` vs.
  `stock_pushdown`).
- **Why unmodified Stock DataFusion loses 7.7–42.1 ms on SF1 `lineitem`
  (`stock_default` vs. `stock_no_pushdown`):** Even on this balanced 16-file /
  48-row-group dataset, Stock's default `repartition_file_scans=true` combined
  with `RowGroupRangeAssignment::StartOffset` slices files blindly by byte
  offset (e.g., assigning `part-000008.parquet:0..3520` to partition 3),
  producing a static **`[6, 6, 6, 7, 6, 6, 6, 5]`** row-group skew across 8
  partitions and creating a `7/6 = 1.17×` tail straggler despite identical
  aggregate CPU work. Morsel's dynamic row-group claiming automatically
  balances all 8 workers (`6..6` per worker on `q1_unordered`
  `morsel_no_pushdown`/`morsel_pruning`) without requiring manual
  `repartition_file_scans=false`. When Stock is manually tuned with
  `repartition_file_scans=false` (`stock_no_pushdown` /
  `stock_pushdown_no_repartition`), Morsel matches or edges out Stock on
  `q6_count` (`48.3` vs. `49.2 ms`) and `q1_unordered` pushdown (`170.4` vs.
  `171.1 ms`) and stays within `0.7–3.8%` on `q6` and `q6_scan`.
- **Progress vs. `#165` pre-`#149`/`#150`/`#166`/`#167` baseline on `dduck2`:**
  - `q6` and `q1_unordered` are newly admitted natively under
    `morsel_mode=required` (previously rejected at planning time).
  - `q6_scan` drops **`−8.2 ms` (`−7.8%`, `105.8 → 97.6 ms` total; `99.1 →
90.0 ms` exec)** and collapses output batches from **736 → 17** (`#149`
    worker-local coalescing), flipping from roughly tied with `stock_default`
    (`105.8` vs. `108.9 ms`) to **1.15× faster** (`97.6` vs. `111.9 ms`).
  - `q6_count` remains steady at **`48.3 ms`** (`47.3 ms` in `#165`), ahead of
    every Stock arm (`49.2–71.3 ms`), because `decode_work` already accounts for
    `87.9%` of worker time at 1.90% selectivity.
- **What explains the remaining 1.45–1.82× gap to DuckDB:**
  - **Parquet `FLBA(7)` → `Decimal128` decode is the #1 bottleneck across all
    four queries, explaining `76.5–79.7%` of the aggregate operator-CPU gap
    (`Morsel (decode + process) − DuckDB total_operator_ms`, or `48.9–59.6%` of
    the total fresh-process wall-clock gap dividing aggregate CPU delta by 8
    cores).** Comparing Morsel `decode_work` against DuckDB's fresh-process
    `TABLE_SCAN` operator CPU across 8 cores: `q6` `513.3` vs. `392.5 ms`
    (`+120.8 ms` CPU / `+15.1 ms` over 8 cores = `76.9%` of the `157.0 ms`
    operator-CPU gap, `52.1%` of the `29.0 ms` wall gap), `q1_unordered`
    `718.3` vs. `443.4 ms` (`+274.9 ms` CPU / `+34.4 ms` over 8 cores = `79.7%`
    of the `344.8 ms` operator-CPU gap, `59.6%` of the `57.7 ms` wall gap),
    `q6_count` `242.7` vs. `140.0 ms` (`+102.7 ms` CPU / `+12.8 ms` over 8
    cores = `76.5%` of the `134.3 ms` operator-CPU gap, `59.2%` of the
    `21.7 ms` wall gap), and `q6_scan` `568.4` vs. `449.5 ms` (`+118.9 ms` CPU
    / `+14.9 ms` over 8 cores = `78.0%` of the `152.5 ms` operator-CPU gap,
    `48.9%` of the `30.4 ms` wall gap). As isolated in `#180`, `parquet 60.1.0`
    `FixedLenByteArrayReader` performs two-pass 7-byte `memcpy` +
    `sign_extend_be` per row instead of pre-decoding small dictionary pages
    into `i128`.
  - **Expression evaluation + 8-aggregate hash aggregation explains `21.5%` of
    the `q1_unordered` aggregate operator-CPU gap (`16.1%` of the `57.7 ms`
    wall-clock gap)** (`401.0 ms` Morsel `process_work` under `morsel_pushdown`
    vs. `326.9 ms` DuckDB `PROJECTION 73.0 + HASH_GROUP_BY 253.8 +
RESULT_COLLECTOR 0.1`, `+74.1 ms` CPU / `+9.3 ms` over 8 cores). DuckDB's
    optimizer rewrites `AVG(l_quantity)` and `AVG(l_extendedprice)` into
    `SUM / COUNT` reusing the already-computed `SUM` accumulators (avoiding 2
    redundant 128-bit sums over 5.92M rows), whereas DataFusion maintains 7
    independent `Decimal128` sums and hashes `(Utf8View, Utf8View)` keys
    through Arrow `RowConverter` / `GroupValuesRows`.
  - **Morsel scheduling / wait overhead is minimal:** aggregate
    `morsel_ready_wait` across all 8 workers is `7.3–24.2 ms` (`0.9–3.0 ms` per
    worker) on the fastest arms, and `morsel_output_wait` is `0.0 ms` on every
    query and arm.

## Environment and Measurement Controls

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 vCPUs (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 pinned CPUs), `--threads 8`
  (Tokio `worker_threads=8`, `max_blocking_threads=8`, `target_partitions=8`;
  DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification:** `vmstat -y 1 3` immediately before the 224-trial
  scorecard showed **92–93% host CPU idle** (`us 6–7%, sy 0–1%`); immediately
  after showed **97–100% host CPU idle** (`us 0–2%, sy 0–1%`). No competing
  builds or benchmarks ran.
- **Build & versions:**
  - DataFusion revision: `dd782ac575c6635635eb472936f0e4c3bcdde449`
    (`agent/morsel-issue-173`, extending only `benchmarks/morsel_tpch/` on top
    of `morsel-execution` HEAD `b339aa2ba` after `#168`), built with
    `--locked --profile release-nonlto -p datafusion --no-default-features --features parquet,sql`.
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`,
    Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`.
- **Dataset & warm page-cache verification:** Standard deterministic `#165` SF1
  TPC-H Parquet dataset generated under `$HOME/.cache/issue173/tpch_sf1` on
  local EC2 NVMe Btrfs storage (`lineitem`: 16 files, 6,001,215 rows, 48 row
  groups, `177,150,918` bytes — byte-for-byte identical to `#165`). All **224 /
  224** timed trials started and ended with **43,256 / 43,256 (100%)** 4 KiB
  pages resident via Linux `mincore`.
- **Protocol:** Fresh process per trial, 1 unmeasured warmup round + 7 measured
  counterbalanced rounds (alternating arm order between odd and even rounds).
  All timings below are medians in milliseconds with observed min–max in
  parentheses.
- **Morsel metrics instrumentation check:** All 224 scorecard trials ran with
  `MORSEL_METRICS=1` enabled so every Morsel trial emits wait/work breakdowns.
  A separate (non-interleaved) `MORSEL_METRICS=0` follow-up run
  (`1 warmup + 7 repeats`) showed `0–3 ms` lower medians on most arms across
  the 4 queries: `q6` `86.0` vs. `88.3 ms` (`morsel_no_pushdown`), `119.4` vs.
  `122.7 ms` (`morsel_pushdown`); `q1_unordered` `182.7` vs. `181.8 ms`
  (`morsel_no_pushdown`), `166.9` vs. `170.4 ms` (`morsel_pushdown`); `q6_count`
  `48.1` vs. `48.3 ms` (`morsel_no_pushdown`), `61.3` vs. `63.5 ms`
  (`morsel_pushdown`); `q6_scan` `96.5` vs. `97.6 ms` (`morsel_no_pushdown`),
  `134.1` vs. `132.2 ms` (`morsel_pushdown`).

## 1. Full 3-Way SF1 Scorecard (Fresh Process, Warm Page Cache, 8 Threads)

All DataFusion arms scanned the exact same Parquet bytes (`47,682,742` on `q6`,
`51,899,102` on `q1_unordered`, `16,509,334` on `q6_count`, `57,252,505` on
`q6_scan`). Every `morsel_mode=required` trial claimed `48 / 48` row groups,
with `aggregate_finalizations=1` on `q6` and `q1_unordered`,
`count_finalizations=1` on `q6_count`, and zero fallbacks.

### TPC-H Q6 (Filtered Ungrouped `SUM`, 1 Row Output, 1.90% Selectivity)

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | Batches | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |          87.7 (83.3–95.3) |     4.9 + 4.0 |         78.8 (75.0–86.3) |       544.9 |       1 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |      **88.3 (84.7–92.7)** |     4.8 + 3.2 |     **80.4 (76.5–84.6)** |       564.1 |       1 |      48 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |          90.0 (86.3–97.1) |     4.9 + 4.3 |         80.8 (77.0–88.0) |       575.7 |       1 |      48 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |       102.4 (100.5–106.0) |     4.8 + 5.2 |         92.5 (90.4–95.7) |       553.0 |       1 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |       120.4 (119.6–125.3) |     4.8 + 5.0 |      110.8 (109.7–115.6) |       740.1 |       1 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |       122.7 (119.2–125.6) |     5.0 + 4.3 |      113.5 (110.4–115.8) |       730.0 |       1 |      48 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |       137.4 (134.9–141.4) |     4.9 + 5.2 |      127.5 (124.9–131.2) |       736.0 |       1 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |      **59.3 (55.1–69.5)** |             — |                        — |       369.5 |       1 |       — |

### TPC-H Q1 Unordered (Grouped 8-Aggregate on `(l_returnflag, l_linestatus)`, 4 Rows, 98.6% Selectivity)

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | Batches | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |       176.3 (175.0–187.4) |     4.8 + 5.4 |      166.2 (164.8–177.2) |     1,270.3 |       3 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |       181.8 (170.0–185.1) |     4.7 + 4.1 |      172.9 (161.5–176.2) |     1,251.2 |       1 |      48 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |       175.3 (173.1–178.4) |     4.8 + 5.0 |      165.5 (163.1–168.8) |     1,263.1 |       1 |      48 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |       218.4 (214.6–222.2) |     4.8 + 6.4 |      207.4 (203.5–211.1) |     1,282.1 |       3 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |       171.1 (158.0–194.1) |     4.9 + 6.0 |      160.9 (147.3–183.0) |     1,130.7 |       3 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |   **170.4 (163.2–179.5)** |     4.7 + 5.0 |  **160.9 (153.3–169.2)** |     1,152.4 |       1 |      48 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |       203.1 (199.9–206.0) |     4.8 + 6.2 |      191.9 (189.0–195.1) |     1,142.5 |       3 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |   **112.7 (109.5–118.4)** |             — |                        — |       772.2 |       4 |       — |

### Updated `q6_count` (Filtered `COUNT(*)`, 1 Row Output, 1.90% Selectivity)

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | Batches | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |          49.2 (46.7–90.3) |     4.5 + 3.5 |         40.9 (38.5–82.7) |       279.5 |       1 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |      **48.3 (45.0–50.2)** |     4.7 + 2.8 |     **40.7 (37.4–42.5)** |       289.1 |       1 |      48 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |          50.7 (46.8–57.1) |     4.7 + 4.0 |         42.2 (38.5–48.2) |       287.9 |       1 |      48 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |          56.9 (55.8–58.2) |     4.6 + 4.8 |         47.5 (46.4–48.5) |       280.6 |       1 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |          64.4 (58.3–73.8) |     4.6 + 4.7 |         54.9 (49.0–64.5) |       342.6 |       1 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |          63.5 (60.9–65.0) |     5.0 + 4.0 |         54.5 (52.6–56.0) |       349.2 |       1 |      48 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |          71.3 (70.2–73.2) |     4.8 + 4.7 |         62.0 (60.9–63.7) |       351.4 |       1 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |      **26.6 (24.1–31.5)** |             — |                        — |       147.8 |       1 |       — |

### Updated `q6_scan` (Filtered Scan + Projection, 114,160 Rows Output, 1.90% Selectivity)

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | First batch / 8k ms | Batches | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------------------: | ------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |         94.0 (90.9–108.4) |     4.9 + 3.3 |        85.8 (82.8–101.0) |       601.8 |         57.4 / 57.4 |      16 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |     **97.6 (95.1–102.9)** |     4.8 + 2.6 |     **90.0 (87.6–95.2)** |       639.1 |         60.4 / 60.4 |      17 |      48 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |         99.1 (96.2–102.0) |     4.7 + 3.7 |         90.4 (88.3–92.9) |       639.9 |         61.7 / 61.7 |      16 |      48 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |       111.9 (111.0–115.4) |     4.7 + 4.5 |      102.7 (101.5–106.4) |       609.7 |         58.4 / 58.4 |      18 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |       127.8 (125.0–133.5) |     4.5 + 4.2 |      119.2 (115.9–124.8) |       820.7 |         31.9 / 33.9 |      48 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |       132.2 (130.7–139.2) |     4.7 + 3.7 |      123.8 (122.6–131.0) |       822.5 |         78.7 / 78.7 |      17 |      48 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |       152.8 (147.3–155.5) |     4.6 + 4.2 |      143.8 (138.8–146.5) |       823.7 |         32.0 / 33.8 |      48 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |      **67.2 (64.0–74.3)** |             — |                        — |       449.3 |         32.3 / 39.0 |      48 |       — |

## 2. Comparison Against `#165` Pre-`#149`/`#150`/`#166`/`#167` Baseline (`rob-tandy-dduck2`)

Both campaigns ran on the same host (`rob-tandy-dduck2`, 8 pinned cores
`taskset -c 0-7`, identical SF1 Parquet dataset, warm page cache verified via
`mincore`, 1 warmup + 7 repeats):

| Query          | Arm                  | `#165` (`4dbbaf33d`) total / exec ms | Checkpoint A (`dd782ac57`) total / exec ms |                                 Delta | Notes                                                                                    |
| -------------- | -------------------- | -----------------------------------: | -----------------------------------------: | ------------------------------------: | ---------------------------------------------------------------------------------------- |
| `q6`           | `morsel_no_pushdown` |                         Not admitted |                            **88.3 / 80.4** |                        Newly admitted | Beats `stock_default` (`102.4 ms`) by 1.16×                                              |
| `q1_unordered` | `morsel_pushdown`    |                         Not admitted |                          **170.4 / 160.9** |                        Newly admitted | Beats `stock_default` (`218.4 ms`) by 1.28× and `stock_pushdown` (`203.1 ms`) by 1.19×   |
| `q6_count`     | `morsel_no_pushdown` |                          47.3 / 40.0 |                            **48.3 / 40.7** |                +1.0 / +0.7 ms (~flat) | Decode is 87.9% of worker time at 1.90% selectivity; still beats all Stock arms          |
| `q6_count`     | `morsel_pruning`     |                          49.2 / 40.9 |                            **50.7 / 42.2** |                +1.5 / +1.3 ms (~flat) | Beats `stock_default` (`56.9 ms`) by 1.12×                                               |
| `q6_scan`      | `morsel_no_pushdown` |           105.8 / 99.1 (736 batches) |               **97.6 / 90.0 (17 batches)** |    **−8.2 / −9.1 ms (−7.8% / −9.2%)** | `#149` coalescing cuts batches 43×; flips ahead of `stock_default` (`111.9 ms`) by 1.15× |
| `q6_scan`      | `morsel_pruning`     |          109.3 / 101.1 (736 batches) |               **99.1 / 90.4 (16 batches)** | **−10.2 / −10.7 ms (−9.3% / −10.6%)** | First-batch latency shifts `16.1 → 61.7 ms` (full 8,192-row batch)                       |

## 3. Wait/Work Breakdown: Morsel vs. Stock vs. DuckDB

### Morsel Native Wait/Work Metrics (Summed Across 8 Workers, Median ms)

| Query          | Arm                  | Decode work ms (% work) | Process work ms (% work) | Ready wait ms | Output wait ms | Pending IO ms | Peak ready/out | Claims/worker (min..max) |
| -------------- | -------------------- | ----------------------: | -----------------------: | ------------: | -------------: | ------------: | -------------: | -----------------------: |
| `q6`           | `morsel_no_pushdown` |           513.3 (92.2%) |              43.4 (7.8%) |          14.9 |            0.0 |          91.8 |            8/1 |                     5..7 |
| `q6`           | `morsel_pruning`     |           518.3 (92.2%) |              44.0 (7.8%) |          16.7 |            0.0 |         105.2 |            8/1 |                     5..7 |
| `q6`           | `morsel_pushdown`    |           697.4 (99.9%) |               1.0 (0.1%) |          46.9 |            0.0 |         118.1 |            7/1 |                     3..9 |
| `q1_unordered` | `morsel_no_pushdown` |           715.3 (57.6%) |            526.9 (42.4%) |          18.3 |            0.0 |         132.3 |            8/1 |                     6..6 |
| `q1_unordered` | `morsel_pruning`     |           701.2 (57.2%) |            525.4 (42.8%) |          16.9 |            0.0 |         121.7 |            8/1 |                     6..6 |
| `q1_unordered` | `morsel_pushdown`    |           718.3 (64.2%) |            401.0 (35.8%) |          24.2 |            0.0 |          85.7 |            8/1 |                    1..15 |
| `q6_count`     | `morsel_no_pushdown` |           242.7 (87.9%) |             33.3 (12.1%) |           7.3 |            0.0 |          28.0 |            8/1 |                     3..7 |
| `q6_count`     | `morsel_pruning`     |           242.7 (87.8%) |             33.7 (12.2%) |          12.0 |            0.0 |          28.8 |            8/1 |                     4..8 |
| `q6_count`     | `morsel_pushdown`    |          334.0 (100.0%) |               0.0 (0.0%) |          38.3 |            0.0 |          67.4 |            5/1 |                    3..10 |
| `q6_scan`      | `morsel_no_pushdown` |           568.4 (91.8%) |              50.8 (8.2%) |          20.5 |            0.0 |         118.4 |            8/7 |                     5..7 |
| `q6_scan`      | `morsel_pruning`     |           577.9 (91.8%) |              52.0 (8.2%) |          16.8 |            0.0 |         113.9 |            8/7 |                     4..7 |
| `q6_scan`      | `morsel_pushdown`    |           786.1 (99.4%) |               4.4 (0.6%) |          36.4 |            0.0 |         125.9 |            8/5 |                    3..11 |

### Cross-Engine Operator CPU Decomposition (Aggregate ms Across 8 Threads)

Stock operator timings are median `elapsed_compute` summed across plan nodes
from the main scorecard run. DuckDB operator timings were collected with
`harness.py duckdb-profile` (fresh process and fresh `:memory:` connection per
repetition, `SET threads = 8`, `taskset -c 0-7`, `con.execute(sql).fetchall()`
with `PRAGMA enable_profiling = 'json'`, warm page cache verified via
`mincore`, 1 warmup + 7 repeats, taking the median of `operator[].timing`
summed in ms grouped by `type`, idle host `vmstat` 92–93% before / 95–99%
after; note that the median of per-run totals can differ slightly from the sum
of per-operator medians).

| Query          | Engine / Arm                          |                           Scan + Decode ms |                                        Filter + Proj + Agg ms | Total Op Work ms | Wall ms |
| -------------- | ------------------------------------- | -----------------------------------------: | ------------------------------------------------------------: | ---------------: | ------: |
| `q6`           | Stock `stock_no_pushdown`             |                   486.9 (`DataSourceExec`) |                              41.0 (`Filter` 40.2 + `Agg` 0.8) |            527.9 |    87.7 |
| `q6`           | Morsel `morsel_no_pushdown`           |                      513.3 (`decode_work`) |                                         43.4 (`process_work`) |            556.7 |    88.3 |
| `q6`           | DuckDB `duckdb_default`               |                       392.5 (`TABLE_SCAN`) |                        7.2 (`PROJ` 5.6 + `UNGROUPED_AGG` 1.5) |            399.7 |    59.3 |
| `q1_unordered` | Stock `stock_no_pushdown`             |                   704.5 (`DataSourceExec`) |             485.9 (`Filter` 98.8 + `Proj` 71.9 + `Agg` 315.2) |          1,190.4 |   176.3 |
| `q1_unordered` | Stock `stock_pushdown_no_repartition` | 793.8 (`DataSourceExec`, incl. fused proj) |                                       323.9 (`AggregateExec`) |          1,117.8 |   171.1 |
| `q1_unordered` | Morsel `morsel_no_pushdown`           |                      715.3 (`decode_work`) |                                        526.9 (`process_work`) |          1,242.2 |   181.8 |
| `q1_unordered` | Morsel `morsel_pushdown`              |                      718.3 (`decode_work`) |                                        401.0 (`process_work`) |          1,119.3 |   170.4 |
| `q1_unordered` | DuckDB `duckdb_default`               |                       443.4 (`TABLE_SCAN`) | 326.9 (`PROJ` 73.0 + `HASH_GROUP_BY` 253.8 + `COLLECTOR` 0.1) |            774.5 |   112.7 |
| `q6_count`     | Stock `stock_no_pushdown`             |                   229.6 (`DataSourceExec`) |                              34.1 (`Filter` 34.0 + `Agg` 0.1) |            263.8 |    49.2 |
| `q6_count`     | Morsel `morsel_no_pushdown`           |                      242.7 (`decode_work`) |                                         33.3 (`process_work`) |            276.0 |    48.3 |
| `q6_count`     | DuckDB `duckdb_default`               |                       140.0 (`TABLE_SCAN`) |                        1.7 (`PROJ` 1.0 + `UNGROUPED_AGG` 0.7) |            141.7 |    26.6 |
| `q6_scan`      | Stock `stock_no_pushdown`             |                   539.6 (`DataSourceExec`) |                             42.0 (`Filter` 41.5 + `Proj` 0.5) |            581.6 |    94.0 |
| `q6_scan`      | Morsel `morsel_no_pushdown`           |                      568.4 (`decode_work`) |                                         50.8 (`process_work`) |            619.2 |    97.6 |
| `q6_scan`      | DuckDB `duckdb_default`               |                       449.5 (`TABLE_SCAN`) |                          18.1 (`PROJ` 6.3 + `COLLECTOR` 11.8) |            466.7 |    67.2 |

## 4. Analysis: Where Morsel Wins, Where DuckDB Leads, and Why

### Where Morsel Beats Stock DataFusion

1. **Dynamic row-group work-stealing beats default `repartition_file_scans=true`
   by 1.15–1.28× across all four queries.** Stock's default static byte-range
   partitioner (`stock_default` and `stock_pushdown`) splits the 16 `lineitem`
   files across 8 partitions by raw byte offset. Combined with DataFusion's
   default `RowGroupRangeAssignment::StartOffset`, small file-header slices
   (such as `part-000008.parquet:0..3520`) capture the first row group of the
   next file, assigning `[6, 6, 6, 7, 6, 6, 6, 5]` row groups across the 8
   partitions. That static 7-RG straggler drops effective CPU parallelism from
   `7.0–7.6×` down to `~6.0×` (`q1_unordered`: `166.2 → 207.4 ms` exec on
   identical `1,270–1,282 ms` CPU). Morsel eliminates this automatically via
   dynamic row-group claiming (`6..6` on `q1_unordered` `morsel_no_pushdown` /
   `morsel_pruning`).
2. **Lower planning overhead and zero `RepartitionExec` shuffle on grouped
   aggregates.** Morsel's single-stage `MorselExec` with worker-local
   `AggregateSink` + coordinator finalization plans `0.9–1.4 ms` faster than
   Stock's two-phase `Partial Aggregate -> Hash Repartition -> FinalPartitioned
Aggregate` plan and emits a single finalized output batch (`1` batch vs.
   Stock's `3` partitioned batches on `q1_unordered`).
3. **`pushdown_filters=true` is a clear win on `q1_unordered` (`181.8 →
170.4 ms`, `−11.4 ms`), but regresses Q6-family queries (`+15` to `+35 ms`).**
   - On **`q1_unordered`**, the filter is a single predicate
     (`l_shipdate <= DATE '1998-09-02'`, 98.6% selectivity). `RowFilter` creates
     one `ArrowPredicate` with zero redundant column re-decoding (`decode_work`
     stays flat at `715.3 → 718.3 ms`), while avoiding post-decode filtering and
     6-column copying of 5.92M surviving rows saves **`−125.9 ms` of
     `process_work`** (`526.9 → 401.0 ms`).
   - On **`q6`, `q6_count`, and `q6_scan`**, `RowFilter` splits the 4-conjunct
     predicate into 5 separate `ArrowPredicate`s and evaluates smallest-column-first
     (`reorder_filters=true`), re-decoding `l_shipdate` and `l_discount`
     multiple times and adding `+91` to `+218 ms` of `decode_work` (confirming
     `#180`). Note also that under `morsel_pushdown`, async `RowFilter` yields
     increase per-worker claim variance (`1..15` on `q1_unordered`, `3..11` on
     `q6_scan`) and raise `morsel_ready_wait` (`18.3 → 24.2 ms` on
     `q1_unordered`, `14.9 → 46.9 ms` on `q6`), though `q1_unordered` still
     achieves `7.16×` CPU parallelism (`1,152.4 / 160.9 ms`).

### Where DuckDB Is Still Ahead — Root-Cause Attribution

1. **Parquet `FLBA(7)` → `Decimal128` decoding (`76.5–79.7%` of the aggregate
   operator-CPU gap; `48.9–59.6%` of the total wall-clock gap divided across 8
   cores).**
   - All four TPC-H `lineitem` monetary/quantity columns (`l_quantity`,
     `l_extendedprice`, `l_discount`, `l_tax`) are stored as dictionary-encoded
     `FIXED_LEN_BYTE_ARRAY(7)` in standard PyArrow Parquet.
   - `arrow-rs` / `parquet 60.1.0` `FixedLenByteArrayReader` performs two-pass
     7-byte `memcpy` + `sign_extend_be` per row (`18M` rows on `q6`, `24M` rows
     on `q1_unordered`), rather than decoding the tiny dictionary pages once
     into `Vec<i128>`.
   - Comparing Morsel `decode_work` vs. DuckDB fresh-process `TABLE_SCAN`, this
     gap is **`+120.8 ms` CPU (`+15.1 ms` over 8 cores)** on `q6`,
     **`+274.9 ms` CPU (`+34.4 ms` over 8 cores)** on `q1_unordered`,
     **`+102.7 ms` CPU (`+12.8 ms` over 8 cores)** on `q6_count`, and
     **`+118.9 ms` CPU (`+14.9 ms` over 8 cores)** on `q6_scan`. On `q6`,
     DuckDB's late materialization inside `TABLE_SCAN` also avoids decoding
     `l_extendedprice` (31.2 MB compressed) for filtered-out vectors.
2. **Expression evaluation and 8-aggregate hash aggregation on `q1_unordered`
   (`21.5%` of the aggregate operator-CPU gap; `16.1%` of the `57.7 ms`
   wall-clock gap, `+74.1 ms` CPU / `+9.3 ms` over 8 cores).**
   - DuckDB's `EXPLAIN` / JSON profile shows two structural advantages on Q1's
     5.92M surviving rows:
     1. **Aggregate CSE:** DuckDB rewrites `AVG(l_quantity)` and
        `AVG(l_extendedprice)` into `SUM(col) / COUNT(col)`, reusing the
        existing `SUM` accumulators and eliminating two redundant 128-bit sums
        over 5.92M rows. Both Stock and Morsel DataFusion accumulate all 8
        aggregates independently (`4× SUM` + `3× AVG` + `COUNT(*)` = 7
        independent `Decimal128` sums + 4 counts).
     2. **Compact string group keys & vectorized accumulation:** DuckDB hashes
        the 1-byte `(l_returnflag, l_linestatus)` keys directly in its tight
        vectorized `HASH_GROUP_BY` (`253.8 ms`, plus `73.0 ms` `PROJECTION` and
        `0.1 ms` `RESULT_COLLECTOR` = `326.9 ms`), whereas DataFusion encodes
        `(Utf8View, Utf8View)` through Arrow `RowConverter` / `GroupValuesRows`
        and evaluates checked `Decimal128` arithmetic (`401.0 ms` in Morsel
        `morsel_pushdown` `process_work`; in Stock `stock_no_pushdown`,
        `ProjectionExec 71.9 + AggregateExec 315.2 = 387.1 ms` excluding
        `FilterExec`, while in `stock_pushdown_no_repartition`, `AggregateExec`
        is `323.9 ms` with the `__common_expr_1` projection fused into
        `DataSourceExec`'s `793.8 ms`).
3. **Post-decode filter evaluation when `pushdown_filters=false` on Q6-family
   queries (`+31.6–36.2 ms` CPU / `+4.0–4.5 ms` over 8 cores).**
   - Because `RowFilter` regresses on Q6's 4-conjunct predicate, Morsel's best
     Q6 arm runs with `pushdown_filters=false`, paying `33.3–50.8 ms` of
     post-decode `process_work` compared to DuckDB's `1.7–18.1 ms` post-scan
     work (where the filter ran inside `TABLE_SCAN`).
4. **Fresh-process registration + planning (`7.4–9.7 ms`).**
   - DataFusion spends `4.7–5.0 ms` in `register_parquet` (reading 16 Parquet
     footers for schema inference) plus `2.6–5.0 ms` in logical/physical
     planning (`7.4–9.7 ms` total). DuckDB's fresh-process total includes footer
     reads, planning, and Arrow C-stream export in its single wall timing.

## 5. Reproduction Commands

From the repository root on Linux (`rob-tandy-dduck2`; main 224-trial scorecard
recorded at `dd782ac575c6635635eb472936f0e4c3bcdde449`, DuckDB operator profile
recorded using `harness.py duckdb-profile` on this branch with identical engine
code):

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work="$HOME/.cache/issue173"
mkdir -p "$work"
revision=$(git rev-parse HEAD)

# 1. Build the standalone runner as a detached Cargo example (no workspace edits).
git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$work/pinned/$b/tpch_runner.rs" \
  "$work/pinned/datafusion/core/examples/morsel_tpch_runner.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$revision" CARGO_TARGET_DIR="$work/target" \
    RUSTC_WRAPPER=sccache \
    timeout --kill-after=20s 35m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql \
      --example morsel_tpch_runner
)
runner="$work/target/release-nonlto/examples/morsel_tpch_runner"

# 2. Create the pinned Python environment.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"

# 3. Run unit + deterministic --sf 0.01 end-to-end parity, plan, and metrics tests.
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 180s "$py" -B -m unittest discover -s "$b" -v

# 4. Generate deterministic SF1 TPC-H Parquet tables outside Git.
data_dir="$work/tpch_sf1"
timeout 180s "$py" -B "$b/harness.py" generate "$data_dir" \
  --sf 1 --row-group-size 131072

# 5. Run the Checkpoint A 3-way SF1 warm-cache benchmark on 8 pinned cores.
vmstat -y 1 3
timeout --kill-after=10s 300s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q6 q1_unordered q6_count q6_scan \
  --morsel-arms stock_no_pushdown morsel_no_pushdown morsel_pruning stock_default \
    stock_pushdown_no_repartition morsel_pushdown stock_pushdown duckdb_default \
  > "$work/sf1_checkpoint_a.jsonl"
vmstat -y 1 3

# 6. Print timing + wait/work summary tables.
"$py" -B "$b/harness.py" summarize "$work/sf1_checkpoint_a.jsonl"

# 7. Collect fresh-process DuckDB JSON operator timing breakdown (Section 3).
vmstat -y 1 3
timeout --kill-after=10s 180s taskset -c 0-7 "$py" -B "$b/harness.py" duckdb-profile \
  "$data_dir" --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q6 q1_unordered q6_count q6_scan \
  > "$work/sf1_duckdb_profile.json"
vmstat -y 1 3

# 8. Separate (non-interleaved) instrumentation-overhead check (MORSEL_METRICS=0).
MORSEL_METRICS=0 timeout --kill-after=10s 180s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q6 q1_unordered q6_count q6_scan \
  --morsel-arms morsel_no_pushdown morsel_pushdown \
  > "$work/sf1_metrics_off.jsonl"
"$py" -B "$b/harness.py" summarize "$work/sf1_metrics_off.jsonl"

# 9. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```
