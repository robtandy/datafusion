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
   64 radix partitions with parallel partition finalization (lock-free atomic
   split-block Bloom construction, single shared non-spillable resident
   `MemoryConsumer` for `FairSpillPool` fairness), batch-level inlined flat-int
   radix probing with direct primitive / `StringView` build-column gather; under
   tight memory limits, excess build partitions and matching probe partitions
   spill via `SpillManager` (preserving sound bounds-only dynamic filter
   pushdown across `M ∪ S`) and replay in a follow-up `P2` child pipeline with
   recursive radix splitting and a block-nested-loop hash sub-pass for
   unsplittable duplicate-key skew across
   `INNER`/`LEFT`/`RIGHT`/`FULL`/`SEMI`/`ANTI`.
3. **External spill-merge for parallel `Sort` (and `TopN` overflow fallback)**:
   worker-local sorted runs spill to disk via `SpillManager` on `MemoryPool`
   pressure and stream-merge through DataFusion's `StreamingMergeBuilder` /
   `MultiLevelMergeBuilder` in bounded memory, reusing the pipeline's
   `SpillContext` across finalize.

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, Ubuntu 24.04, Linux `7.0.0-1014-aws`, 16 ARM
  Neoverse-N1 vCPUs (1 thread/core), `aarch64`.
- **CPU pinning and threads:** `taskset -c 0-7` (8 pinned CPUs), 8 Tokio
  workers (`worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`), `batch_size=8,192`.
- **Host idle verification:** `vmstat -y 1 3` reported **98% idle** before the
  run and **97% idle** immediately after.
- **Revisions compared (`--profile release-nonlto`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`):**
  - **Before (`morsel-execution` base):** `c21850bd6`
  - **After (`agent/morsel-issue-194`):** `1d8b6ffb4` (paired full-RAM + tight-memory suite) and `45fca6ad9` (`q10_128mb` nested-P2 parent spill headroom fix)
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
| **TPC-H Q3** | 386.5 (376.2–388.2) | **354.3 (350.8–361.1)** | **1.09×** (−32.2 ms) | 65.8 | **27.2** | 356.3 |
| **TPC-H Q9** | 1,895.2 (1,873.1–1,906.0) | **1,337.0 (1,313.9–1,363.3)** | **1.42×** (−558.2 ms) | 975.8 | **263.1** | 1,449.8 |
| **TPC-H Q10** | 641.7 (634.1–648.4) | **562.7 (559.4–571.5)** | **1.14×** (−79.0 ms) | 119.1 | **91.4** | 499.6 |
| **TPC-H Q13** | 926.9 (903.5–950.8) | **552.8 (546.5–563.4)** | **1.68×** (−374.1 ms) | 58.1 | **25.3** | 527.8 |
| **TPC-H Q18** | 1,621.6 (1,586.2–1,625.9) | **638.6 (629.3–645.9)** | **2.54×** (−983.0 ms) | 65.6 | **27.8** | 773.6 |
| **5-Query Sum** | **5,471.9** | **3,445.4** | **1.59×** (−2,026.5 ms) | **1,284.4** | **434.8** | **3,607.1** |
| **TPC-H Q1** *(low-card check)* | 690.8 (682.6–704.8) | **699.3 (686.1–701.7)** | **0.99×** (+8.5 ms) | 0.0 | 0.0 | 729.6 |
| **TPC-H Q12** *(small-build check)* | 239.2 (237.1–244.9) | **236.8 (234.9–246.1)** | **1.01×** (−2.4 ms) | 13.1 | **5.9** | 275.4 |

### Companion Profile: `pruning` (`pushdown_filters=false, pruning=true`)

| Query | Before wall ms (min–max) | After wall ms (min–max) | Wall speedup | Before `jfin` ms | After `jfin` ms | `stock_pruning` ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TPC-H Q3** | 510.8 (500.3–512.4) | **519.6 (509.4–521.0)** | **0.98×** (+8.8 ms) | 60.7 | **10.3** | 478.4 |
| **TPC-H Q9** | 1,857.8 (1,839.6–1,878.1) | **1,234.6 (1,228.6–1,236.8)** | **1.50×** (−623.2 ms) | 929.4 | **178.4** | 971.4 |
| **TPC-H Q10** | 775.4 (767.0–780.4) | **715.1 (712.1–723.8)** | **1.08×** (−60.3 ms) | 142.6 | **72.4** | 727.6 |
| **TPC-H Q13** | 881.4 (849.2–908.9) | **534.4 (521.6–582.5)** | **1.65×** (−347.0 ms) | 52.3 | **8.9** | 466.4 |
| **TPC-H Q18** | 2,777.1 (2,726.6–2,801.4) | **1,730.8 (1,708.1–1,744.8)** | **1.60×** (−1,046.3 ms) | 790.9 | **274.4** | 1,366.0 |
| **5-Query Sum** | **6,802.5** | **4,734.5** | **1.44×** (−2,068.0 ms) | **1,975.9** | **544.4** | **4,009.8** |
| **TPC-H Q1** *(low-card check)* | 701.5 (694.0–709.4) | **694.2 (690.0–702.4)** | **1.01×** (−7.3 ms) | 0.0 | 0.0 | 704.5 |
| **TPC-H Q12** *(small-build check)* | 204.0 (200.2–207.2) | **198.2 (195.8–202.5)** | **1.03×** (−5.8 ms) | 18.4 | **2.4** | 195.6 |

## Results: Tight-Memory SF=5 Spill Completion (`GreedyMemoryPool`)

Every workload below was executed on SF=5 under a strict `GreedyMemoryPool`
budget with `DiskManagerMode::OsTmpDirectory` and `morsel_mode=required`, and
verified for exact Arrow IPC result parity against unconstrained Stock
DataFusion:

| Workload | `GreedyMemoryPool` | Before (`c21850bd6`) | After (`45fca6ad9`) | Wall ms | Spilled MiB | Spill files | Exact parity |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| **TPC-H Q3** | 32 MiB | `ResourcesExhausted` | **Completed** | 4,660.6 | 506.4 MiB | 11,344 | Exact (10 rows) |
| **TPC-H Q9** | 256 MiB | `ResourcesExhausted` | **Completed** | 22,804.6 | 673.5 MiB | 4,784 | Exact (175 rows) |
| **TPC-H Q10** | 128 MiB | `ResourcesExhausted` | **Completed** | 5,046.2 | 472.7 MiB | 7,675 | Exact (20 rows) |
| **TPC-H Q13** | 64 MiB | `ResourcesExhausted` | **Completed** | 3,041.0 | 239.7 MiB | 2,452 | Exact (42 rows) |
| **TPC-H Q18** | 128 MiB | `ResourcesExhausted` | **Completed** | 1,851.4 | 222.0 MiB | 1,024 | Exact (100 rows) |
| **SF=5 `orders` `ORDER BY`**\* | 128 MiB | `ResourcesExhausted` | **Completed** | 720.0 | 173.8 MiB | 75 | Exact (1.14M rows) |

\*`SELECT o_orderkey, o_custkey, o_totalprice, o_orderdate, o_comment FROM orders WHERE o_orderdate < DATE '1993-01-01' ORDER BY o_totalprice DESC, o_orderkey ASC` (1.14M filtered rows with wide `Utf8View` comments; ordered parity verified via SHA-256 over the full sorted stream).

## Key Takeaways

- **TPC-H Q18 (`1,621.6 -> 638.6 ms`, `2.54×` pushdown; `2,777.1 -> 1,730.8 ms`,
  `1.60×` pruning):** Stage 0 groups 30M `lineitem` rows into 7.5M distinct
  `l_orderkey` groups. L2 pre-aggregation collapses 4× before radix scattering
  into 64 partitions, and parallel finalize runs merge +
  `HAVING sum(l_quantity) > 300` across all 8 cores simultaneously (beating
  `stock_pushdown`'s `773.6 ms` by **135.0 ms**). Under a 128 MiB pool, Q18
  completes cleanly in **1,851.4 ms** by spilling 222.0 MiB across 1,024 files.
- **TPC-H Q13 (`926.9 -> 552.8 ms`, `1.68×` pushdown; `881.4 -> 534.4 ms`,
  `1.65×` pruning):** Adaptive direct radix scatter on `DirectGroupedState`
  skips worker-local hash-table lookups on 7.5M `orders` rows across 750K
  `c_custkey` groups, finalizes 64 L2-resident (~187 KiB) partitions in
  parallel across 8 cores, and lowers total process CPU by **−7.1%**.
- **TPC-H Q9 (`1,895.2 -> 1,337.0 ms`, `1.42×` pushdown; `1,857.8 -> 1,234.6 ms`,
  `1.50×` pruning):** Parallel 64-way radix `HashJoin` build finalization cuts
  `join_finalize` across the 4M-row `partsupp` (`Rows` key) and 7.5M-row
  `orders` (`Int64` key) builds by **3.7×–5.2×** (`975.8 -> 263.1 ms` pushdown;
  `929.4 -> 178.4 ms` pruning), beating `stock_pushdown` (`1,449.8 ms`) by
  **112.8 ms**.
- **TPC-H Q10 (`641.7 -> 562.7 ms`, `1.14×` pushdown; `775.4 -> 715.1 ms`,
  `1.08×` pruning) & Q3 (`386.5 -> 354.3 ms`, `1.09×` pushdown; `510.8 -> 519.6 ms`,
  `0.98×` pruning):** Batch-level inlined flat-int radix probing plus direct
  primitive/`StringView` build-column gather eliminate the earlier `interleave`
  probe overhead on `pushdown_filters=false` (bringing Q12 to **1.03×** on
  `pruning` and **1.01×** on `pushdown`, and Q10 to **1.08×** on `pruning`,
  beating `stock_pruning` `727.6 ms`).
