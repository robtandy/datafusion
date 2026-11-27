<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Stock-Gap Fix Verification: TPC-H SF=5 `q9`, `q10`, and `q13` (#217)

This report records the paired quiet-host SF=5 before/after benchmark results
and 22-query TPC-H non-regression verification for the root-cause fixes
identified by `#214` ([`q9_q10_q13.md`](q9_q10_q13.md)) and implemented in
`#217`.

## Summary of Results (`morsel_pushdown` vs. Best Stock, SF=5 Warm Cache, 8 Threads)

All three target queries **eliminate 100% of the remaining Stock gap** and
**beat the fastest Stock configuration** on both wall time and total execution
CPU with exact DuckDB 2.0.1 parity and `morsel_mode = required`:

| Query | Baseline `morsel_pushdown` | **Fixed `morsel_pushdown`** | Before -> After Delta | Paired Best Stock (`stock_default` / `stock_pushdown`) | **Fixed `morsel_pushdown` vs. Best Stock** |
| ----- | -------------------------: | --------------------------: | --------------------: | -----------------------------------------------------: | -----------------------------------------: |
| `q9`  |               `1,365.2 ms` |              **`861.6 ms`** | **`-503.6 ms` (`0.631x`)** |                            `1,000.4 ms` (`stock_default`) | **`-138.8 ms` (`0.861x`, `-1,096 ms` CPU)** |
| `q10` |                 `582.5 ms` |              **`496.5 ms`** |  **`-86.0 ms` (`0.852x`)** |                           `517.5 ms` (`stock_pushdown`) | **`-21.0 ms` (`0.959x`, `-52 ms` CPU)**     |
| `q13` |                 `572.4 ms` |              **`375.6 ms`** | **`-196.8 ms` (`0.656x`)** |                             `461.4 ms` (`stock_default`) | **`-85.8 ms` (`0.814x`, `-928 ms` CPU)**    |

Across all 22 TPC-H SF=5 queries (`morsel_pushdown`, 3 paired repeats, 22/22
exact DuckDB parity), total suite wall time dropped from **`9,840.2 ms` to
`8,839.8 ms` (`-1,000.4 ms`, `-10.2%`)** and total exec CPU dropped from
**`64,812.9 ms` to `58,406.2 ms` (`-6,406.7 ms`, `-9.9%`)**. In a follow-up
strictly alternating 5-repeat paired check on `q4`, `q7`, `q12`, `q20`, and
`q22`, `q4` (`-2.4 ms`), `q7` (`-24.9 ms`), and `q12` (`-5.3 ms`) all improved,
while `q20` (`+1.1 ms`, `1.004x`) and `q22` (`+3.4 ms`, `1.029x`, overlapping
min-max ranges) stayed within run-to-run noise.

---

## Implemented Root-Cause Fixes (`datafusion/core/src/morsel/`)

All seven recommendations from `#214` were implemented inside
`datafusion/core/src/morsel/` with zero changes to stock physical operators or
DataFusion defaults:

1. **Selectivity & Domain-Saturation Gating + Lock-Free Radix-Partitioned Bloom Filters (`parquet.rs`, `join_filter.rs`, `hash_join.rs`):**
   - Replaced contended `AtomicSplitBlockBloomFilter` (`AtomicU32::fetch_or`
     across all 8 workers) with lock-free Level-0 radix-partitioned
     `SplitBlockBloomFilter` slices (`num_blocks >= 64` maps partition `p` from
     bits `[51..57)` to a disjoint contiguous slice of `blocks`, with zero
     atomics and L1/L2-local non-atomic writes).
   - Skipped Bloom construction on unfiltered base tables above 16,384 rows
     (such as `partsupp` 4M, `customer` 750K, `supplier` 50K) and on single
     integer keys whose `[min, max]` domain is >= 75% saturated (already covered
     by pushed min/max bounds), while preserving Bloom construction for
     filtered builds, `Stage` sources, `BuildSemi`/`BuildAnti` filtered builds
     (`q21`), and backward dynamic-filter-pruned builds (`q18`).
2. **Eliminated 58K-Batch Radix Slice Fragmentation (`RadixChunk`) + Parallel Table Drop (`radix.rs`, `radix_agg.rs`, `radix_join.rs`, `aggregate.rs`, `group_count.rs`, `hash_join.rs`, `morsel.rs`):**
   - Replaced `Vec<Vec<RecordBatch>>` (which eagerly created 64 sliced
     `RecordBatch` objects per 8K batch — 58K slices on `q9` `orders` and `q13`
     `AggregateSink`) with `Vec<RadixChunk>` (1 permuted `RecordBatch` +
     `Vec<(u32, u32)>` partition ranges), removing single-threaded `.clone()`
     and `drop()` storms on both radix `GROUP BY` and radix `HashJoin` while
     preserving clean OOM spill fallback.
   - Added zero-slice `merge_partial_slice` to `DirectGroupedRadixMerger` and
     `GroupCountRadixMerger`, and direct primitive payload copying in
     `build_single_table_from_radix_chunks`.
   - Replaced synchronous single-threaded `drop(finalized_tables)` on large
     joins with awaited parallel partition drops across workers (freeing 100% of
     `MemoryPool` leases before query completion — genuine wall-time reduction,
     not teardown deferral).
3. **Join Build-Side Estimation for Two-Sided Ranges & Projected Width Fraction (`morsel.rs`, `parquet.rs`):**
   - Scaled `Source::Parquet` whole-file bytes by the projected-to-file-schema
     column width ratio (`projected_width / file_schema_width`, weighting
     `Utf8`/`Utf8View`/`Binary` = 4, `Decimal` = 2, fixed-width = 1), avoiding
     double-counting width on small wide dimension tables while recognizing
     narrow projections on large tables.
   - Gave two-sided constant range predicates (`col >= lo AND col < hi`) shift
     weight `2` (`1/4` selectivity) and kept comparable-size `Stage` inputs on
     the probe side so cross-stage dynamic filters and empty-build
     short-circuits stay active. This flips `q10` to build narrow filtered
     `orders` (`286K x 2 ints`) and prune 71.6% of wide `customer` (`750K x 7
     cols`), without flipping `q5`, `q7`, `q8`, `q17`, or small wide dims.
4. **Sibling Forward Dynamic Filter Propagation & Topological Build Ordering (`parquet.rs`, `dag.rs`, `morsel.rs`, `hash_join.rs`):**
   - Propagated dynamic join filters forward across `Inner`/`ProbeSemi` sibling
     joins sharing a probe key column when the earlier join is filtered and the
     later sibling is unfiltered (`part -> partsupp` on `q9`, shrinking
     `partsupp` ~18x from `4.0M` to `218K` rows).
   - Replaced fixed reverse build order in `run_build_pipelines` with Kahn's
     topological sort over dynamic-filter dependencies (breaking ties in reverse
     probe order and accumulating `short_circuit_op` with `prev.max(op_idx)` so
     earlier redundant build-emission pipelines are skipped).
5. **Specialized 2-Key Integer `FlatIntTable<(i32, i32)>` & `FlatIntTable<(i64, i64)>` (`hash_join.rs`, `join_filter.rs`):**
   - Added `KeyKind::Int32Pair` / `KeyKind::Int64Pair` with `mix_u64_pair`,
     eliminating `arrow_row::RowConverter` and `memcmp` on `(ps_suppkey,
     ps_partkey)` in `q9` across both single-table and 64-way radix paths.
6. **Build-Key Probe Substitution & Zero-Copy Contiguous Probe Output (`hash_join.rs`):**
   - On matched rows where a projected `JoinOutputCol::Build` equals a plain
     non-float probe equijoin key (`c_custkey == o_custkey` on `q13`), read
     directly from the L1-resident probe batch instead of gathering from build
     partitions; when `probe_indices` is a contiguous ascending range, emit
     zero-copy `Arc::clone` / `.slice` instead of `arrow::compute::take`.
7. **Dense Integer `ArrayMap` (`DenseIntTable`) & Density-Gated `R = 0` Budget (`radix_join.rs`, `hash_join.rs`):**
   - Added direct-indexed `DenseIntTable` (4 bytes/slot `Vec<u32>`, zero
     hashing, lazy `next_row` empty for unique PK builds) mirroring Stock's
     `ArrayMap` heuristic (>= 15% density, <= 4M span), covering `nation` (25),
     `supplier` (50K, 200 KiB), and `customer` (750K, 3 MiB).
   - Kept `RADIX_JOIN_THRESHOLD_ROWS = 16_384` for all non-dense builds
     (including sparse single-int builds like `q4`/`q7`/`q10`/`q12` `orders`
     and `q9` `part` + 2-key `partsupp`), allowing `R = 0` above `16_384` rows
     (up to `1_048_576` rows / `16 MiB`) **only** when running integer bounds
     qualify for `DenseIntTable` (verified strictly at `>= 15%` density in
     `finalize_with_spill`, falling back to `flush_r0_to_radix` otherwise).

---

## Measurement Environment and Controls

- **Host:** `rob-tandy-dduck1`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1014-aws`,
  16 Arm Neoverse-N1 cores (64 KiB L1d / 1 MiB L2 per core, 32 MiB shared L3),
  61 GiB RAM.
- **CPU pinning & threads:** `taskset -c 0-7`, `--threads 8`, batch size
  `8,192`, `--cache warm` (100% Linux `mincore` page residency verified before
  and after every trial).
- **Dataset:** Deterministic SF=5 TPC-H Parquet dataset
  (`/home/bits/.cache/issue193/tpch_sf5`, 43,299,825 rows, 35 files, 353 row
  groups, Zstd level 3, `row_group_size = 131,072`).
- **Revisions compared:**
  - **Baseline (`BASE`):** `e0384274ee3bddd00f20010ebeed6612dfe691ce`
    (`morsel-execution` HEAD prior to `#217`).
  - **Fixed (`AFTER`):** `963e57d32c8b0239c11303f3fc1e08a49751d761` (exact
    production code and tests of this PR).
- **Host idle checks (`vmstat -y 1 3`):**
  - Before run: `86–99%` idle; mid-run (between `q9/q10/q13` and 22-query
    suite): `98–99%` idle; after run: `87–98%` idle. Every trial verified exact
    DuckDB 2.0.1 parity before timing.

---

## 1. Paired SF=5 Before/After Results: `q9`, `q10`, and `q13` (5 Repeats)

Medians in milliseconds with observed `(min-max)` across 5 counterbalanced
repeats (`1` warmup + `5` measured):

| Query | Arm                   | Baseline Total ms (min-max) | **Fixed Total ms (min-max)** | Baseline Exec CPU ms | **Fixed Exec CPU ms** | Baseline JoinFin ms | **Fixed JoinFin ms** | Delta Total ms (Ratio) |
| ----- | --------------------- | --------------------------: | ---------------------------: | -------------------: | --------------------: | ------------------: | -------------------: | ---------------------: |
| `q9`  | `stock_default`       |   `984.7` (980.3-999.4)     |  `1,000.4` (977.9-1,012.7)   |            `6,973.9` |             `6,993.1` |                   — |                    — |     `+15.7` (`1.016x`) |
| `q9`  | `stock_pushdown`      | `1,489.3` (1,463.9-1,497.4) |  `1,505.4` (1,469.9-1,512.8) |           `10,561.9` |            `10,662.5` |                   — |                    — |     `+16.1` (`1.011x`) |
| `q9`  | `morsel_no_pushdown`  | `1,249.3` (1,237.5-1,260.9) |  `1,004.1` (994.3-1,014.0)   |            `8,652.2` |             `7,254.7` |             `176.2` |               `75.7` | **`-245.2` (`0.804x`)** |
| `q9`  | **`morsel_pushdown`** | `1,365.2` (1,360.2-1,374.2) |    **`861.6`** (847.6-904.8) |            `8,981.1` |         **`5,897.3`** |             `259.4` |           **`44.6`** | **`-503.6` (`0.631x`)** |
| `q10` | `stock_default`       |     `621.7` (604.8-656.9)   |      `623.9` (618.7-648.6)   |            `4,129.3` |             `4,194.5` |                   — |                    — |      `+2.2` (`1.003x`) |
| `q10` | `stock_pushdown`      |     `518.7` (513.6-528.5)   |      `517.5` (507.8-524.1)   |            `3,257.5` |             `3,254.7` |                   — |                    — |      `-1.1` (`0.998x`) |
| `q10` | `morsel_no_pushdown`  |     `727.0` (724.7-734.8)   |      `641.9` (634.2-651.6)   |            `5,119.3` |             `4,575.9` |              `73.4` |               `13.4` |  **`-85.1` (`0.883x`)** |
| `q10` | **`morsel_pushdown`** |     `582.5` (579.2-591.9)   |    **`496.5`** (494.6-500.1) |            `3,721.1` |         **`3,202.3`** |              `90.9` |           **`18.7`** |  **`-86.0` (`0.852x`)** |
| `q13` | `stock_default`       |     `455.2` (445.7-470.3)   |      `461.4` (447.6-472.0)   |            `3,335.7` |             `3,327.1` |                   — |                    — |      `+6.2` (`1.014x`) |
| `q13` | `stock_pushdown`      |     `530.9` (517.5-578.4)   |      `538.9` (511.7-560.2)   |            `3,744.4` |             `3,755.5` |                   — |                    — |      `+8.0` (`1.015x`) |
| `q13` | `morsel_no_pushdown`  |     `532.2` (520.0-557.6)   |      `366.0` (356.7-378.6)   |            `3,479.1` |             `2,412.3` |               `9.0` |               `11.4` | **`-166.2` (`0.688x`)** |
| `q13` | **`morsel_pushdown`** |     `572.4` (563.4-581.6)   |    **`375.6`** (365.3-397.2) |            `3,549.6` |         **`2,399.4`** |              `24.9` |           **`11.2`** | **`-196.8` (`0.656x`)** |

### Per-Pipeline `morsel_pushdown` Work Attribution (Medians from Paired 5-Repeat Run)

| Query | Pipeline                     | Base Decode ms | After Decode ms | Base Process ms | **After Process ms** | Base JoinFin ms | **After JoinFin ms** | Primary Driver |
| ----- | ---------------------------- | -------------: | --------------: | --------------: | -------------------: | --------------: | -------------------: | -------------- |
| `q9`  | `build_0` (`part`)           |        `106.0` |         `111.6` |           `0.2` |                `0.3` |           `1.6` |            **`0.3`** | Parallel 64-way radix + lock-free partition-slice Bloom |
| `q9`  | `build_1` (`supplier`)       |          `1.4` |           `1.4` |           `0.8` |            **`0.2`** |           `1.7` |            **`0.4`** | Unfiltered Bloom skip + 200 KiB `DenseIntTable` |
| `q9`  | `build_2` (`partsupp`)       |        `222.1` |         `192.5` |          `95.2` |            **`9.3`** |         `144.9` |            **`0.8`** | `part -> partsupp` sibling pushdown (`4M -> 218K`) + unfiltered Bloom skip + parallel radix `Int64Pair` |
| `q9`  | `build_3` (`orders`)         |        `237.7` |         `243.2` |         `187.0` |          **`137.4`** |         `111.3` |           **`43.1`** | `RadixChunk` zero-slice + direct `Date32` payload copy |
| `q9`  | `probe` (`lineitem`)         |      `4,608.3` |     **`3,711.4`** |       `1,853.2` |          **`899.1`** |           `0.0` |                `0.0` | `Int64Pair` + L3-resident `partsupp` + `DenseIntTable` `supplier`/`nation` + zero-copy contiguous probe |
| `q10` | `build_0_build`              |        `337.2` |         `413.3` |          `75.6` |           **`26.2`** |          `70.3` |            **`1.9`** | Flipped build side to filtered `orders` (`286K x 2`) + parallel radix + partition-slice Bloom |
| `q10` | `build_0`                    |        `371.5` |         `354.3` |         `287.2` |          **`102.1`** |          `20.5` |           **`16.5`** | `orders` Bloom prunes `71.6%` of wide `customer` rows |
| `q13` | `stage_0_build` (`customer`) |         `26.4` |          `30.7` |          `23.6` |            **`7.7`** |          `24.9` |           **`11.2`** | Unfiltered Bloom skip + contiguous 3 MiB `DenseIntTable` |
| `q13` | `stage_0_probe` (`orders`)   |      `1,495.4` |       `1,418.3` |       `1,513.7` |          **`784.8`** |           `0.0` |                `0.0` | `DenseIntTable` + `build_key_probe_subst` + zero-copy contiguous probe + `RadixChunk` agg finalize |

---

## 2. Paired SF=5 22-Query TPC-H Non-Regression Check (`morsel_pushdown`, 3 Repeats)

Every query passed exact DuckDB 2.0.1 Arrow IPC parity and `morsel_mode =
required` execution:

| Query     | Baseline ms | **Fixed ms** | Delta ms |    Ratio | Base JoinFin ms | **Fixed JoinFin ms** | Baseline Exec CPU ms | **Fixed Exec CPU ms** |
| --------- | ----------: | -----------: | -------: | -------: | --------------: | -------------------: | -------------------: | --------------------: |
| `q1`      |     `573.2` |      `581.3` |   `+8.1` | `1.014x` |           `0.0` |                `0.0` |            `4,310.5` |             `4,387.5` |
| `q2`      |     `163.7` |      `157.3` |   `-6.3` | `0.961x` |           `1.9` |                `1.7` |              `755.1` |               `716.9` |
| `q3`      |     `378.1` |      `352.6` |  `-25.5` | `0.933x` |          `27.2` |                `6.0` |            `2,431.8` |             `2,304.4` |
| `q4`      |     `216.4` |      `207.0` |   `-9.4` | `0.956x` |           `8.8` |                `2.2` |            `1,360.9` |             `1,314.0` |
| `q5`      |     `423.1` |      `411.7` |  `-11.4` | `0.973x` |          `11.8` |                `2.2` |            `2,773.7` |             `2,711.2` |
| `q6`      |     `333.2` |      `330.4` |   `-2.8` | `0.991x` |           `0.0` |                `0.0` |            `2,449.2` |             `2,458.8` |
| `q7`      |     `417.7` |      `400.4` |  `-17.2` | `0.959x` |          `21.1` |                `4.7` |            `2,704.6` |             `2,610.8` |
| `q8`      |     `538.6` |      `505.9` |  `-32.7` | `0.939x` |          `21.0` |                `4.5` |            `3,562.3` |             `3,284.0` |
| **`q9`**  |   `1,368.7` |    **`864.7`** | **`-504.1`** | **`0.632x`** |         `260.2` |           **`45.3`** |            `8,969.7` |         **`5,915.3`** |
| **`q10`** |     `581.2` |    **`495.9`** |  **`-85.3`** | **`0.853x`** |          `91.0` |           **`18.4`** |            `3,718.9` |         **`3,201.1`** |
| `q11`     |     `125.8` |      `125.9` |   `+0.2` | `1.001x` |           `0.1` |                `0.1` |              `758.9` |               `753.3` |
| `q12`     |     `194.8` |      `194.8` |   `+0.1` | `1.000x` |           `5.4` |                `1.3` |            `1,296.2` |             `1,281.3` |
| **`q13`** |     `562.1` |    **`367.4`** | **`-194.7`** | **`0.654x`** |          `25.0` |           **`11.7`** |            `3,545.6` |         **`2,391.9`** |
| `q14`     |     `326.8` |      `303.0` |  `-23.8` | `0.927x` |          `33.5` |                `6.3` |            `2,127.0` |             `1,837.3` |
| `q15`     |     `680.9` |      `665.8` |  `-15.0` | `0.978x` |           `3.1` |                `1.6` |            `4,966.5` |             `4,891.8` |
| `q16`     |     `161.8` |      `161.6` |   `-0.2` | `0.999x` |           `5.8` |                `1.7` |              `703.8` |               `681.2` |
| `q17`     |     `475.9` |      `482.2` |   `+6.3` | `1.013x` |           `0.7` |                `0.4` |            `3,307.9` |             `3,329.4` |
| `q18`     |     `634.5` |      `615.7` |  `-18.8` | `0.970x` |          `27.3` |                `9.3` |            `4,226.2` |             `4,024.4` |
| `q19`     |     `390.5` |      `386.9` |   `-3.6` | `0.991x` |           `0.7` |                `1.8` |            `2,478.5` |             `2,551.5` |
| `q20`     |     `312.9` |      `332.0` |  `+19.0` | `1.061x` |           `0.7` |                `0.7` |            `1,936.9` |             `1,957.6` |
| `q21`     |     `862.9` |      `772.1` |  `-90.8` | `0.895x` |         `137.5` |               `37.7` |            `5,762.8` |             `5,129.8` |
| `q22`     |     `117.5` |      `125.0` |   `+7.5` | `1.064x` |           `3.6` |                `1.3` |              `666.0` |               `672.9` |
| **Total (22q)** | **`9,840.2`** | **`8,839.8`** | **`-1,000.4`** | **`0.898x`** | — | — | **`64,812.9`** | **`58,406.2`** |

### Strictly Alternating 5-Repeat Paired Spot-Check (`q4`, `q7`, `q12`, `q20`, `q22`)

Because `q20` and `q22` ran near the end of the 3-repeat sweep (when host idle
dipped briefly to `87%`) and `q4`/`q7`/`q12` were earlier sparse-int
regression candidates, we re-checked all five with strictly alternating
trial-by-trial execution (`B-A-A-B-B-A-A-B-B-A`, 1 warmup + 5 measured repeats
on `taskset -c 0-7`):

| Query | Baseline ms (min-max) | **Fixed ms (min-max)** | Delta ms (Ratio) | Base JoinFin ms | **Fixed JoinFin ms** | Base CPU ms | **Fixed CPU ms** |
| ----- | --------------------: | ---------------------: | ---------------: | --------------: | -------------------: | ----------: | ---------------: |
| `q4`  | `211.7` (210.9-221.0) |  `209.4` (201.2-218.2) |  `-2.4` (`0.989x`) |           `8.8` |                `2.2` |   `1,348.2` |        `1,317.9` |
| `q7`  | `414.7` (406.9-418.4) |  `389.9` (384.6-395.2) | `-24.9` (`0.940x`) |          `21.9` |                `4.9` |   `2,720.0` |        `2,578.6` |
| `q12` | `194.8` (192.6-195.3) |  `189.5` (185.1-191.9) |  `-5.3` (`0.973x`) |           `5.6` |                `1.3` |   `1,295.1` |        `1,265.4` |
| `q20` | `309.8` (301.9-313.7) |  `310.9` (305.6-315.9) |  `+1.1` (`1.004x`) |           `0.7` |                `0.7` |   `1,915.3` |        `1,937.7` |
| `q22` | `118.9` (115.3-119.6) |  `122.4` (117.9-123.6) |  `+3.4` (`1.029x`) |           `3.7` |                `1.1` |     `664.9` |          `675.4` |

`q4`, `q7`, and `q12` are all faster than baseline (`q7` and `q12` with
non-overlapping min-max ranges), and `q20` (`+1.1 ms`) / `q22` (`+3.4 ms`) have
overlapping min-max ranges with baseline.

---

## 3. Correctness, Spill, and Lint Verification

- **Unit & Integration Suites (238/238 passed, 0 failed):**
  - `cargo test -p datafusion --lib morsel::` (112 passed)
  - `morsel_aggregate` (13 passed), `morsel_batch_view` (6 passed),
    `morsel_execution` (17 passed), `morsel_join` (28 passed),
    `morsel_parquet` (34 passed), `morsel_sort` (10 passed),
    `morsel_spill` (12 passed), `morsel_tpch_coverage` (6 passed, including
    22/22 TPC-H parity)
- **Feature-gate & Lint checks:**
  - `cargo fmt --all --check`: passed
  - `cargo clippy --all-targets --all-features -- -D warnings`: passed (0 warnings)
  - `cargo check -p datafusion --no-default-features`: passed (0 warnings)

## 4. Reproduction Commands

```bash
set -euo pipefail
work="$HOME/.cache/issue217"
py="/home/bits/.cache/issue193/venv/bin/python"
data_dir="/home/bits/.cache/issue193/tpch_sf5"
after_rev="963e57d32c8b0239c11303f3fc1e08a49751d761"

# Paired 5-repeat SF=5 benchmark on q9, q10, q13 (repeat for base and fixed runner)
taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" \
  --runner "$work/morsel_tpch_runner_after3" \
  --revision "$after_rev" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries q9,q10,q13 \
  --morsel-arms stock_default,stock_pushdown,morsel_no_pushdown,morsel_pushdown \
  > "$work/v3_q9_q10_q13_after.jsonl"

# Paired 3-repeat SF=5 non-regression across all 22 TPC-H queries
taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" \
  --runner "$work/morsel_tpch_runner_after3" \
  --revision "$after_rev" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 3 \
  --queries all \
  --morsel-arms morsel_pushdown \
  > "$work/v3_22q_after.jsonl"
```
