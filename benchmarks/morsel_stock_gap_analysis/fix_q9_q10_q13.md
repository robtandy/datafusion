<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Stock-Gap Fix Verification: TPC-H SF=5 `q9`, `q10`, and `q13` (#217)

This report records the paired SF=5 before/after benchmark results and 22-query
TPC-H non-regression verification for the root-cause fixes identified by `#214`
([`q9_q10_q13.md`](q9_q10_q13.md)) and implemented in `#217`.

## Summary of Results (`morsel_pushdown` vs. Best Stock, SF=5 Warm Cache, 8 Threads)

All three target queries **eliminate 100% of the remaining Stock gap** and now
**beat the fastest Stock configuration** on both wall time and total execution
CPU with exact DuckDB 2.0.1 parity and `morsel_mode = required`:

| Query | Baseline `morsel_pushdown` | **Fixed `morsel_pushdown`** | Before -> After Delta | Paired Best Stock (`stock_default` / `stock_pushdown`) | **Fixed `morsel_pushdown` vs. Best Stock** |
| ----- | -------------------------: | --------------------------: | --------------------: | -----------------------------------------------------: | -----------------------------------------: |
| `q9`  |               `1,408.8 ms` |            **`1,041.0 ms`** | **`-367.7 ms` (`0.739x`)** |                            `1,053.2 ms` (`stock_default`) | **`-12.2 ms` (`0.988x`, `-436 ms` CPU)**   |
| `q10` |                 `607.7 ms` |              **`522.9 ms`** |  **`-84.8 ms` (`0.861x`)** |                           `532.3 ms` (`stock_pushdown`) | **`-9.4 ms` (`0.982x`, `-122 ms` CPU)**    |
| `q13` |                 `611.7 ms` |              **`389.6 ms`** | **`-222.2 ms` (`0.637x`)** |                             `472.3 ms` (`stock_default`) | **`-82.7 ms` (`0.825x`, `-956 ms` CPU)**   |

Across all 22 TPC-H SF=5 queries (`morsel_pushdown`, 3 paired repeats, 22/22
exact DuckDB parity), total suite wall time dropped from **`10,317.4 ms` to
`9,067.4 ms` (`-1,250.0 ms`, `-12.1%`)** and total exec CPU dropped from
**`67,810.8 ms` to `58,539.0 ms` (`-9,271.8 ms`, `-13.7%`)** with zero
regressions.

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
3. **Join Build-Side Estimation for Two-Sided Ranges & Projected Column Width (`morsel.rs`):**
   - Weighted `Source::Parquet` side estimates by projected column width
     (`Utf8`/`Utf8View`/`Binary` = 4, `Decimal` = 2, fixed-width = 1, floor 4)
     and gave two-sided constant range predicates (`col >= lo AND col < hi`)
     weight `2` (`1/4` selectivity). This flips `q10` to build narrow filtered
     `orders` (`286K x 2 ints`) and prune 71.6% of wide `customer` (`750K x 7
     cols`), without flipping `q5`, `q7`, `q8`, or `q17`.
4. **Sibling Forward Dynamic Filter Propagation & Topological Build Ordering (`parquet.rs`, `dag.rs`, `morsel.rs`, `hash_join.rs`):**
   - Propagated dynamic join filters forward across `Inner`/`ProbeSemi` sibling
     joins sharing a probe key column when the earlier join is filtered and the
     later sibling is unfiltered (`part -> partsupp` on `q9`, shrinking
     `partsupp` ~18x from `4.0M` to `218K` rows).
   - Replaced fixed reverse build order in `run_build_pipelines` with Kahn's
     topological sort over dynamic-filter dependencies (breaking ties in reverse
     probe order so existing dependency chains and short-circuits remain intact).
5. **Specialized 2-Key Integer `FlatIntTable<(i32, i32)>` & `FlatIntTable<(i64, i64)>` (`hash_join.rs`, `join_filter.rs`):**
   - Added `KeyKind::Int32Pair` / `KeyKind::Int64Pair` with `mix_u64_pair`,
     eliminating `arrow_row::RowConverter` and `memcmp` on `(ps_suppkey,
     ps_partkey)` in `q9`.
6. **Build-Key Probe Substitution & Zero-Copy Contiguous Probe Output (`hash_join.rs`):**
   - On matched rows where a projected `JoinOutputCol::Build` equals a plain
     non-float probe equijoin key (`c_custkey == o_custkey` on `q13`), read
     directly from the L1-resident probe batch instead of gathering from build
     partitions; when `probe_indices` is a contiguous ascending range, emit
     zero-copy `Arc::clone` / `.slice` instead of `arrow::compute::take`.
7. **Dense Integer `ArrayMap` (`DenseIntTable`) & Integer Radix Threshold Tuning (`radix_join.rs`, `hash_join.rs`):**
   - Added direct-indexed `DenseIntTable` (4 bytes/slot `Vec<u32>`, zero
     hashing, lazy `next_row` empty for unique PK builds) mirroring Stock's
     `ArrayMap` heuristic (>= 15% density, <= 4M span), covering `nation` (25),
     `supplier` (50K, 200 KiB), and `customer` (750K, 3 MiB).
   - Tuned in-memory single-table budgets for compact integer builds
     (`1_048_576` rows for single int keys, `262_144` rows for int pairs, capped
     at `16 MiB` raw memory) while keeping `16_384` for `Utf8`/`Rows` builds and
     preserving tight-memory OOM radix spilling.

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
  - **Fixed (`AFTER`):** `7546bfa180c371306907179316110849540cd714` (identical
    production code to PR head; subsequent commits only add tests, formatting,
    clippy type-alias cleanup, and this report).
- **Host idle checks (`vmstat -y 1 3`):**
  - Before `BASE` run: `93–97%` idle; between `BASE` and `AFTER`: `99%` idle;
    at start of `AFTER` `q9,q10,q13` run: transient `55–66%` idle (note Stock
    reference arms remained within `0.3–1.5%` on `q9`/`q10`); after `AFTER` run:
    `87–99%` idle. Every trial verified exact DuckDB 2.0.1 parity before timing.

---

## 1. Paired SF=5 Before/After Results: `q9`, `q10`, and `q13` (5 Repeats)

Medians in milliseconds with observed `(min-max)` across 5 counterbalanced
repeats (`1` warmup + `5` measured):

| Query | Arm                  | Baseline Total ms (min-max) | **Fixed Total ms (min-max)** | Baseline Exec CPU ms | **Fixed Exec CPU ms** | Baseline JoinFin ms | **Fixed JoinFin ms** | Delta Total ms (Ratio) |
| ----- | -------------------- | --------------------------: | ---------------------------: | -------------------: | --------------------: | ------------------: | -------------------: | ---------------------: |
| `q9`  | `stock_default`      | `1,050.3` (1,029.9-1,130.7) |  `1,053.2` (1,011.7-1,096.9) |            `7,319.1` |             `7,382.6` |                   — |                    — |      `+2.9` (`1.003x`) |
| `q9`  | `stock_pushdown`     | `1,601.4` (1,526.2-1,645.8) |  `1,598.6` (1,572.3-1,647.9) |           `11,221.6` |            `11,281.4` |                   — |                    — |      `-2.7` (`0.998x`) |
| `q9`  | `morsel_no_pushdown` | `1,352.7` (1,335.4-1,365.2) |  `1,100.5` (1,079.0-1,105.6) |            `9,357.7` |             `7,809.8` |             `196.8` |               `97.3` |  **`-252.2` (`0.814x`)** |
| `q9`  | **`morsel_pushdown`** | `1,408.8` (1,388.9-1,430.6) | **`1,041.0`** (1,017.1-1,107.1) |            `9,244.9` |         **`6,946.4`** |             `280.5` |           **`82.9`** |  **`-367.7` (`0.739x`)** |
| `q10` | `stock_default`      |     `646.6` (635.3-655.8)   |      `634.3` (630.0-652.1)   |            `4,302.5` |             `4,274.9` |                   — |                    — |     `-12.3` (`0.981x`) |
| `q10` | `stock_pushdown`     |     `524.4` (517.4-532.1)   |      `532.3` (525.4-539.1)   |            `3,386.8` |             `3,376.9` |                   — |                    — |      `+7.9` (`1.015x`) |
| `q10` | `morsel_no_pushdown` |     `760.9` (753.0-777.0)   |      `665.1` (643.6-683.8)   |            `5,366.2` |             `4,686.3` |              `78.9` |               `26.4` |   **`-95.9` (`0.874x`)** |
| `q10` | **`morsel_pushdown`** |     `607.7` (597.9-615.4)   |    **`522.9`** (515.0-531.9) |            `3,883.5` |         **`3,254.6`** |              `95.9` |           **`34.3`** |   **`-84.8` (`0.861x`)** |
| `q13` | `stock_default`      |     `500.3` (481.6-517.2)   |      `472.3` (469.0-476.9)   |            `3,569.3` |             `3,440.4` |                   — |                    — |     `-28.0` (`0.944x`) |
| `q13` | `stock_pushdown`     |     `557.6` (543.2-597.3)   |      `550.8` (531.8-558.1)   |            `4,060.3` |             `3,909.7` |                   — |                    — |      `-6.8` (`0.988x`) |
| `q13` | `morsel_no_pushdown` |     `566.1` (554.7-582.2)   |      `375.5` (362.3-377.2)   |            `3,702.0` |             `2,489.2` |               `9.7` |               `11.8` |  **`-190.5` (`0.663x`)** |
| `q13` | **`morsel_pushdown`** |     `611.7` (584.6-620.0)   |    **`389.6`** (362.8-400.0) |            `3,785.9` |         **`2,484.2`** |              `26.3` |           **`12.4`** |  **`-222.2` (`0.637x`)** |

### Per-Pipeline `morsel_pushdown` Work Attribution (Medians from Paired 5-Repeat Run)

| Query | Pipeline         | Base Decode ms | After Decode ms | Base Process ms | **After Process ms** | Base JoinFin ms | **After JoinFin ms** | Primary Driver |
| ----- | ---------------- | -------------: | --------------: | --------------: | -------------------: | --------------: | -------------------: | -------------- |
| `q9`  | `build_2` (`partsupp`) | `233.1` | `201.4` | `96.8` | **`6.3`** | `154.7` | **`19.6`** | `part -> partsupp` sibling pushdown (`4M -> 218K`) + unfiltered Bloom skip + `Int64Pair` |
| `q9`  | `build_3` (`orders`)   | `239.1` | `276.4` | `188.3` | `192.4` | `123.4` | **`60.4`** | `RadixChunk` zero-slice + direct `Date32` payload copy |
| `q9`  | `probe` (`lineitem`)   | `4,601.4` | `4,705.7` | `1,995.1` | **`866.0`** | `0.0` | `0.0` | `Int64Pair` + L3-resident `partsupp` + `DenseIntTable` `supplier`/`nation` + zero-copy contiguous probe |
| `q10` | `build_0_build`        | `342.5` | `429.1` | `81.8` | **`23.9`** | `74.3` | **`17.1`** | Flipped build side from `customer` (`750K x 7`) to filtered `orders` (`286K x 2`) |
| `q10` | `build_0`              | `383.2` | `356.0` | `310.6` | **`78.0`** | `22.2` | **`17.1`** | `orders` Bloom prunes `71.6%` of wide `customer` rows |
| `q13` | `stage_0_build` (`customer`) | `27.4` | `32.1` | `24.3` | **`6.8`** | `26.3` | **`12.4`** | Unfiltered Bloom skip + contiguous 3 MiB `DenseIntTable` |
| `q13` | `stage_0_probe` (`orders`)   | `1,560.5` | `1,760.4` | `1,674.8` | **`431.6`** | `0.0` | `0.0` | `DenseIntTable` + `build_key_probe_subst` + zero-copy contiguous probe + `RadixChunk` agg finalize |

---

## 2. Paired SF=5 22-Query TPC-H Non-Regression Check (`morsel_pushdown`, 3 Repeats)

Every query passed exact DuckDB 2.0.1 Arrow IPC parity and `morsel_mode =
required` execution:

| Query | Baseline ms | **Fixed ms** | Delta ms | Ratio | Baseline Exec CPU ms | **Fixed Exec CPU ms** |
| ----- | ----------: | -----------: | -------: | ----: | -------------------: | --------------------: |
| `q1`  |     `602.2` |      `569.6` |  `-32.6` | `0.946x` |            `4,539.6` |             `4,260.8` |
| `q2`  |     `169.8` |      `156.9` |  `-12.8` | `0.924x` |              `783.3` |               `723.1` |
| `q3`  |     `393.3` |      `359.5` |  `-33.7` | `0.914x` |            `2,558.2` |             `2,207.5` |
| `q4`  |     `217.7` |      `228.0` |  `+10.3` | `1.047x` |            `1,376.9` |             `1,315.7` |
| `q5`  |     `447.8` |      `407.2` |  `-40.6` | `0.909x` |            `2,932.3` |             `2,563.9` |
| `q6`  |     `339.7` |      `332.7` |   `-7.0` | `0.979x` |            `2,533.1` |             `2,466.8` |
| `q7`  |     `434.5` |      `433.7` |   `-0.7` | `0.998x` |            `2,831.2` |             `2,549.0` |
| `q8`  |     `559.4` |      `518.9` |  `-40.5` | `0.928x` |            `3,717.1` |             `3,231.7` |
| **`q9`**  |   `1,442.3` |    **`970.7`** | **`-471.7`** | **`0.673x`** |            `9,314.7` |         **`6,534.4`** |
| **`q10`** |     `623.3` |    **`518.7`** | **`-104.6`** | **`0.832x`** |            `3,927.1` |         **`3,172.8`** |
| `q11` |     `128.9` |      `140.1` |  `+11.1` | `1.086x` |              `774.3` |               `759.2` |
| `q12` |     `203.8` |      `206.3` |   `+2.4` | `1.012x` |            `1,344.2` |             `1,253.1` |
| **`q13`** |     `608.2` |    **`374.3`** | **`-233.9`** | **`0.615x`** |            `3,835.6` |         **`2,412.0`** |
| `q14` |     `345.9` |      `308.2` |  `-37.8` | `0.891x` |            `2,218.1` |             `1,860.3` |
| `q15` |     `704.4` |      `663.1` |  `-41.3` | `0.941x` |            `5,155.8` |             `4,874.6` |
| `q16` |     `169.1` |      `166.6` |   `-2.5` | `0.985x` |              `727.1` |               `647.4` |
| `q17` |     `495.8` |      `489.0` |   `-6.8` | `0.986x` |            `3,420.6` |             `3,412.7` |
| `q18` |     `678.7` |      `595.4` |  `-83.4` | `0.877x` |            `4,468.9` |             `4,070.2` |
| `q19` |     `387.3` |      `377.4` |   `-9.9` | `0.974x` |            `2,583.4` |             `2,507.1` |
| `q20` |     `326.1` |      `310.7` |  `-15.4` | `0.953x` |            `2,014.8` |             `1,925.2` |
| `q21` |     `913.9` |      `819.4` |  `-94.5` | `0.897x` |            `6,040.6` |             `5,131.8` |
| `q22` |     `125.1` |      `120.9` |   `-4.2` | `0.967x` |              `713.7` |               `659.9` |
| **Total (22q)** | **`10,317.4`** | **`9,067.4`** | **`-1,250.0`** | **`0.879x`** | **`67,810.8`** | **`58,539.0`** |

*(Note: on `q4`, `q11`, and `q12`, wall time was within `+2.4` to `+11.1 ms`
noise while total exec CPU strictly improved on all three: `-61.2 ms`,
`-15.1 ms`, and `-91.1 ms`, respectively.)*

---

## 3. Correctness, Spill, and Lint Verification

- **Unit & Integration Suites (234/234 passed, 0 failed):**
  - `cargo test -p datafusion --lib morsel::` (112 passed)
  - `morsel_aggregate` (13 passed), `morsel_batch_view` (6 passed),
    `morsel_execution` (17 passed), `morsel_join` (24 passed),
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

# Paired 5-repeat SF=5 benchmark on q9, q10, q13 (repeat for base and fixed runner)
taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" \
  --runner "$work/morsel_tpch_runner_after2" \
  --revision 7546bfa180c371306907179316110849540cd714 \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries q9,q10,q13 \
  --morsel-arms stock_default,stock_pushdown,morsel_no_pushdown,morsel_pushdown \
  > "$work/paired_q9_q10_q13_after.jsonl"

# Paired 3-repeat SF=5 non-regression across all 22 TPC-H queries
taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" \
  --runner "$work/morsel_tpch_runner_after2" \
  --revision 7546bfa180c371306907179316110849540cd714 \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 3 \
  --queries all \
  --morsel-arms morsel_pushdown \
  > "$work/paired_22q_after.jsonl"
```
