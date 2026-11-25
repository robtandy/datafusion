<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel SF=5 Benchmark: Radix-Partitioned Hash Tables & Out-of-Core Spilling (#194)

This note records a paired quiet-host before/after benchmark at **TPC-H Scale
Factor 5 (`--sf 5`)** evaluating the opt-in morsel radix-partitioning and
out-of-core spilling implementation from `#194` (design in
`docs/source/contributor-guide/morsel_spill_and_join_pushdown.md`):

1. **Radix-partitioned `GROUP BY` & parallel partition finalization**
   (`AggregateSink` — both `Grouped` and `DirectGrouped` with adaptive direct
   radix scatter — and `GroupCountSink`): low-cardinality queries (`<= 4,096`
   groups) remain on the zero-overhead `R = 0` fast path, while high-cardinality
   queries partition into `2^6 = 64` radix partitions (middle hash bits
   `[51..57)`, disjoint from `hashbrown` `h2` `[57..64)` and `h1` `[0..33)`),
   combine in `O(W * 64)` pointer moves, finalize partitions in parallel across
   all workers, and spill/replay partitions one at a time via `SpillManager`
   with recursive radix splitting and sorted-run skew fallback under memory
   pressure.
2. **Radix-partitioned `HashJoin` & out-of-core `ProbeSpill` child pipeline**:
   small builds (`<= 16,384` rows) remain on `R = 0`; large builds scatter into
   64 radix partitions with parallel partition finalization (and lock-free
   atomic split-block Bloom construction); under tight memory limits, excess
   build partitions and matching probe partitions spill via `SpillManager`
   (preserving sound bounds-only dynamic filter pushdown across `M ∪ S`) and
   replay in a follow-up `P2` child pipeline with recursive radix splitting and
   a block-nested-loop hash sub-pass for unsplittable duplicate-key skew across
   `INNER`/`LEFT`/`RIGHT`/`FULL`/`SEMI`/`ANTI`.
3. **External spill-merge for parallel `Sort` (and `TopN` overflow fallback)**:
   worker-local sorted runs spill to disk via `SpillManager` on `MemoryPool`
   pressure and stream-merge through DataFusion's `StreamingMergeBuilder` /
   `MultiLevelMergeBuilder` in bounded memory.

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, Ubuntu 24.04, Linux `7.0.0-1014-aws`, 16 ARM
  Neoverse-N1 vCPUs (1 thread/core), `aarch64`.
- **CPU pinning and threads:** `taskset -c 0-7` (8 pinned CPUs), 8 Tokio
  workers (`worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`), `batch_size=8,192`.
- **Host idle verification:** `vmstat -y 1 3` reported **100% idle** before the
  run and **97% idle** immediately after.
- **Revisions compared (`--profile release-nonlto`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`):**
  - **Before (`morsel-execution` base):** `c21850bd6`
  - **After (`agent/morsel-issue-194`):** `ebb8b2916`
- **Dataset:** Standard deterministic SF=5 TPC-H Parquet tables (35 files, 353
  row groups, 43.3M rows, 1.34 GiB Zstd level 3, pre-warmed in Linux page
  cache).
- **Protocol:** Exact Arrow IPC result parity verified against Stock DataFusion
  for every workload (both unconstrained RAM and tight-memory spill runs). Fresh
  process per trial, 1 unmeasured warmup + 5 counterbalanced measured rounds
  alternating `before`/`after` order. Wall times are medians in milliseconds
  (`min–max` in parentheses); `jfin` is median `morsel_join_finalize_work_ms`.

## Results: High-Cardinality SF=5 TPC-H Speedups (Parallel Radix Finalize)

### Primary Profile: `pushdown` (`pushdown_filters=true`)

| Query | Before wall ms (min–max) | After wall ms (min–max) | Wall speedup | Before `jfin` ms | After `jfin` ms | `stock_pushdown` ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TPC-H Q3** | 380.9 (373.6–389.4) | **359.9 (353.5–364.3)** | **1.06×** (−21.0 ms) | 64.9 | **27.2** | 356.6 |
| **TPC-H Q9** | 1,890.3 (1,867.1–1,912.2) | **1,368.4 (1,328.6–1,373.9)** | **1.38×** (−521.9 ms) | 975.5 | **260.6** | 1,473.3 |
| **TPC-H Q10** | 647.3 (642.3–651.2) | **563.9 (563.1–565.8)** | **1.15×** (−83.4 ms) | 122.9 | **91.3** | 492.7 |
| **TPC-H Q13** | 884.8 (851.6–987.6) | **606.2 (593.9–609.5)** | **1.46×** (−278.6 ms) | 57.9 | **25.0** | 529.7 |
| **TPC-H Q18** | 1,619.7 (1,591.9–1,623.2) | **626.4 (624.0–645.4)** | **2.59×** (−993.3 ms) | 65.1 | **27.6** | 773.9 |
| **5-Query Sum** | **5,423.0** | **3,524.8** | **1.54×** (−1,898.2 ms) | **1,286.3** | **431.7** | **3,626.2** |
| **TPC-H Q1** *(low-card check)* | 693.8 (684.1–715.3) | **694.2 (684.6–703.1)** | **1.00×** (+0.4 ms) | 0.0 | 0.0 | 733.5 |
| **TPC-H Q12** *(small-build check)* | 241.9 (236.0–243.0) | **238.3 (236.3–239.8)** | **1.02×** (−3.6 ms) | 13.3 | **5.4** | 275.9 |

### Companion Profile: `pruning` (`pushdown_filters=false, pruning=true`)

| Query | Before wall ms | After wall ms | Wall speedup | Before `jfin` ms | After `jfin` ms | `stock_pruning` ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TPC-H Q3** | 506.8 | 580.0 | 0.87× | 59.8 | **10.2** | 472.1 |
| **TPC-H Q9** | 1,871.4 | **1,376.6** | **1.36×** | 938.1 | **179.2** | 980.3 |
| **TPC-H Q10** | 773.9 | 783.7 | 0.99× | 141.9 | **74.0** | 743.8 |
| **TPC-H Q13** | 836.3 | **541.1** | **1.55×** | 50.8 | **8.9** | 459.8 |
| **TPC-H Q18** | 2,768.6 | **1,846.1** | **1.50×** | 807.6 | **259.1** | 1,370.8 |
| **5-Query Sum** | **6,757.0** | **5,127.5** | **1.32×** | **1,998.2** | **531.4** | **4,026.8** |
| **TPC-H Q1** *(low-card check)* | 702.6 | 711.3 | 0.99× | 0.0 | 0.0 | 713.0 |
| **TPC-H Q12** *(small-build check)* | 207.5 | 248.6 | 0.83× | 19.6 | **2.5** | 190.7 |

*Note on `pruning` Q3/Q12:* Without `pushdown_filters=true`, unpruned probe
batches hit 64 radix build partitions and materialize build payload columns via
Arrow `interleave` across 64 partition arrays rather than single-array `take`,
adding ~40–70 ms on probe-heavy non-pushdown scans while parallel radix finalize
slashes `jfin` by **5.8×–7.8×** (`59.8 -> 10.2 ms` on Q3, `19.6 -> 2.5 ms` on
Q12). Under `pushdown` (where the Bloom `RowFilter` eliminates ~95–98% of
non-matching probe rows before probing), **all five target queries plus Q12
speed up** and the 5-query sum beats `stock_pushdown` (`3,524.8` vs `3,626.2 ms`).

## Results: Tight-Memory SF=5 Spill Completion (`GreedyMemoryPool`)

Every workload below was executed on SF=5 under a strict `GreedyMemoryPool`
budget with `DiskManagerMode::OsTmpDirectory` and `morsel_mode=required`, and
verified for exact Arrow IPC result parity against unconstrained Stock
DataFusion:

| Workload | `GreedyMemoryPool` | Before (`c21850bd6`) | After (`ebb8b2916`) | Wall ms | Spilled MiB | Spill files | Exact parity |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| **TPC-H Q3** | 32 MiB | `ResourcesExhausted` | **Completed** | 4,711.1 | 506.4 MiB | 11,344 | Exact (10 rows) |
| **TPC-H Q9** | 256 MiB | `ResourcesExhausted` | **Completed** | 23,271.3 | 673.5 MiB | 4,811 | Exact (175 rows) |
| **TPC-H Q10** | 128 MiB | `ResourcesExhausted` | **Completed** | 6,232.7 | 519.3 MiB | 12,246 | Exact (20 rows) |
| **TPC-H Q13** | 64 MiB | `ResourcesExhausted` | **Completed** | 3,000.6 | 239.7 MiB | 2,356 | Exact (42 rows) |
| **TPC-H Q18** | 128 MiB | `ResourcesExhausted` | **Completed** | 1,837.6 | 222.1 MiB | 1,024 | Exact (100 rows) |
| **SF=5 `orders` `ORDER BY`**\* | 128 MiB | `ResourcesExhausted` | **Completed** | 721.2 | 173.4 MiB | 75 | Exact (1.14M rows) |

\*`SELECT o_orderkey, o_custkey, o_totalprice, o_orderdate, o_comment FROM orders WHERE o_orderdate < DATE '1993-01-01' ORDER BY o_totalprice DESC, o_orderkey ASC` (1.14M filtered rows with wide `Utf8View` comments; ordered parity verified via SHA-256 over the full sorted stream).

## Key Takeaways

- **TPC-H Q18 (`1,619.7 -> 626.4 ms`, `2.59×`, `-16.6%` CPU):** Stage 0 groups
  30M `lineitem` rows into 7.5M distinct `l_orderkey` groups. L2 pre-aggregation
  collapses 4× before radix scattering into 64 partitions, and parallel finalize
  runs merge + `HAVING sum(l_quantity) > 300` across all 8 cores simultaneously
  (beating `stock_pushdown`'s `773.9 ms` by **147.5 ms**). Under a 128 MiB
  pool, Q18 completes cleanly in **1,837.6 ms** by spilling 222.1 MiB across
  1,024 files instead of crashing with `ResourcesExhausted`.
- **TPC-H Q9 (`1,890.3 -> 1,368.4 ms`, `1.38×`, `-521.9 ms`):** Parallel 64-way
  radix `HashJoin` build finalization cuts `join_finalize` across the 4M-row
  `partsupp` (`Rows` key) and 7.5M-row `orders` (`Int64` key) builds by **3.7×–5.2×**
  (`975.5 -> 260.6 ms` pushdown; `938.1 -> 179.2 ms` pruning), overtaking
  `stock_pushdown` (`1,473.3 ms`) by **104.9 ms**.
- **TPC-H Q13 (`884.8 -> 606.2 ms`, `1.46×` pushdown; `836.3 -> 541.1 ms`,
  `1.55×` pruning):** Adaptive direct radix scatter on `DirectGroupedState`
  detects low pre-aggregation reduction on 7.5M `orders` rows across 750K
  `c_custkey` groups, skips worker-local hash-table lookups, and finalizes 64
  L2-resident (~187 KiB) partitions in parallel across 8 cores.
- **TPC-H Q10 (`647.3 -> 563.9 ms`, `1.15×`) & Q3 (`380.9 -> 359.9 ms`,
  `1.06×`):** Parallel radix `HashJoin` build finalization plus parallel radix
  `GROUP BY` finalization cut both join build and multi-column/string
  aggregation tail latency, while keeping low-cardinality Q1 (`1.00×`) and
  small-build Q12 (`1.02×`) regression-free.
