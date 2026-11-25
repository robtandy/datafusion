<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel SF=5 Benchmark: Dynamic Join Filter Pushdown & Same-Column `RowFilter` Grouping (#193)

This note records a paired quiet-host before/after benchmark at **TPC-H Scale
Factor 5 (`--sf 5`)** measuring the two opt-in morsel Parquet optimizations
implemented in `#193`:

1. **Dynamic join filter pushdown (`min/max` bounds + small `InList` +
   cache-resident 256-bit split-block Bloom filter)** collected during
   `HashJoinBuildSink` `sink`/`combine`/`finalize`, published strictly before
   probe Parquet I/O lanes open files, with reverse-order cascading propagation
   across multi-join build chains (`region -> nation -> supplier -> customer ->
   orders -> lineitem`), filtered build-side size discounting, and per-conjunct
   tautology elimination in `RowFilter`.
2. **Same-column Parquet `RowFilter` conjunct grouping** on the native buffered
   morsel path (leaving stock DataFusion `morsel_mode=off` completely
   unchanged), merging pushable conjuncts over the exact same leaf column set
   into a single `ArrowPredicate` so each predicate column is decoded once
   instead of 2–3×.

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, Ubuntu 24.04, Linux `7.0.0-1014-aws`, 16 ARM
  Neoverse-N1 vCPUs (1 thread/core), `aarch64`.
- **CPU pinning and threads:** `taskset -c 0-7` (8 pinned CPUs), 8 Tokio
  workers (`worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`), `batch_size=8,192`.
- **Host idle verification:** `vmstat -y 1 3` reported **97–98% idle**
  immediately before the benchmark and **96–99% idle** immediately after.
- **Revisions compared (same `tpch_runner.rs`, `--profile release-nonlto`,
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`):**
  - **Before (`morsel-execution` base):** `3ba7893d9`
  - **After (`agent/morsel-issue-193`):** `e58bd78d6` (Q3/Q5/Q6/Q12 5-round paired table) and `87755b449` (full 22-query SF=5 verification after fixing guarded `RowFilter` staging, `pruning=false` gating, and Q7/Q20 build-side estimates)
- **Dataset:** Standard deterministic SF=5 TPC-H Parquet tables generated via
  `benchmarks/morsel_tpch/harness.py generate ... --sf 5 --row-group-size 131072`
  (35 files, 353 row groups, 43.3M total rows, 1.34 GiB Zstd level 3, pre-read
  into Linux page cache).
- **Protocol:** Exact Arrow IPC result parity verified against Stock DataFusion
  for every workload and arm before timing. Fresh process per trial, 1
  unmeasured warmup round + 5 counterbalanced measured rounds alternating arm
  order. Timings are medians in milliseconds (`min–max` in parentheses); CPU is
  median process `exec_cpu_seconds`.

## Results: TPC-H SF=5 Q3, Q5, Q6, `q6_count`, Q12 & Dynamic Key-Range Join

### Primary Comparison: `morsel_pushdown` (`pushdown_filters=true`) Before vs. After

| Workload | Before wall ms (min–max) | After wall ms (min–max) | Wall speedup | Before CPU ms | After CPU ms | CPU delta | `bytes_scanned` Before → After | `stock_pushdown` wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TPC-H Q3** | 548.6 (543.9–554.3) | **392.5 (387.4–397.7)** | **1.40×** (−156.1 ms) | 3,534.6 | **2,154.5** | **−39.0%** | 334,406,980 → 334,406,980 | 369.8 (2,373.6 ms CPU) |
| **TPC-H Q5** | 735.3 (727.6–739.8) | **409.5 (401.1–412.4)** | **1.80×** (−325.8 ms) | 4,495.6 | **2,526.7** | **−43.8%** | 374,043,457 → 374,043,457 | 464.9 (3,114.9 ms CPU) |
| **TPC-H Q6** | 484.7 (477.6–489.1) | **455.4 (452.9–476.2)** | **1.06×** (−29.3 ms) | 3,320.6 | **3,137.9** | **−5.5%** | 238,671,437 → 238,671,437 | 495.6 (3,360.4 ms CPU) |
| **TPC-H `q6_count`** | 244.3 (240.0–253.1) | **221.4 (216.6–229.1)** | **1.10×** (−22.9 ms) | 1,611.9 | **1,420.7** | **−11.9%** | 82,528,403 → 82,528,403 | 256.4 (1,615.1 ms CPU) |
| **TPC-H Q12** | 895.7 (883.1–905.2) | **251.9 (249.0–255.3)** | **3.56×** (−643.8 ms) | 2,167.5 | **1,541.6** | **−28.9%** | 221,292,133 → 221,292,133 | 283.2 (1,788.5 ms CPU) |
| **SF=5 dynamic range join**\* | 311.1 (307.7–312.7) | **69.9 (68.9–71.7)** | **4.45×** (−241.2 ms) | 2,240.5 | **295.2** | **−86.8%** | **237,374,364 → 25,625,691 (−89.2%)** | 125.8 (342.1 ms CPU) |

\*`sf5_dyn_orderkey_range`:
`SELECT COUNT(*) AS cnt, SUM(l.l_extendedprice * (1 - l.l_discount)) AS rev FROM orders o INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey WHERE o.o_orderkey + 0 >= 1000000 AND o.o_orderkey + 0 < 1500000`.
Using `o_orderkey + 0` prevents static equivalence propagation to `l_orderkey`
so all row-group pruning on 30M-row `lineitem` (`240 → 5` `lineitem` row groups,
`304 → 69` total morsels claimed) comes purely from runtime dynamic join min/max
pushdown.

### Companion Comparison: `morsel_pruning` (`pushdown_filters=false, pruning=true`) & `stock_default`

| Workload | `morsel_pruning` Before | `morsel_pruning` After | Speedup | `bytes_scanned` Before → After | `stock_default` wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| **TPC-H Q3** | 516.0 ms | 521.0 ms | 0.99× | 334,406,980 → 334,406,980 | 497.0 ms |
| **TPC-H Q5** | 732.8 ms | 736.4 ms | 1.00× | 374,043,457 → 374,043,457 | 682.7 ms |
| **TPC-H Q6** | 338.2 ms | 337.2 ms | 1.00× | 238,671,437 → 238,671,437 | 353.4 ms |
| **TPC-H `q6_count`** | 183.6 ms | 186.1 ms | 0.99× | 82,528,403 → 82,528,403 | 198.7 ms |
| **TPC-H Q12** | 847.1 ms | **211.5 ms** | **4.01×** (−635.6 ms) | 221,292,133 → 221,292,133 | 207.2 ms |
| **SF=5 dynamic range join** | 310.7 ms | **68.9 ms** | **4.51×** (−241.8 ms) | **237,374,364 → 25,625,691 (−89.2%)** | 109.8 ms |

### Full 22-Query SF=5 Regression & Speedup Check (`3ba7893d9` vs `87755b449`)

Paired quiet-host check (`vmstat` 98–99% idle before, 95–99% idle after; 1 warmup + 3 counterbalanced measured rounds per profile on 8 pinned CPUs, `taskset -c 0-7`):

| Query | `pruning` Before | `pruning` After | `pruning` Speedup | `pushdown` Before | `pushdown` After | `pushdown` Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q1` | 703.1 ms | 704.7 ms | 1.00× | 707.1 ms | 705.3 ms | 1.00× |
| `q2` | 272.4 ms | 272.1 ms | 1.00× | 278.6 ms | 246.9 ms | **1.13×** |
| `q3` | 525.9 ms | 526.1 ms | 1.00× | 554.6 ms | 393.3 ms | **1.41×** |
| `q4` | 262.3 ms | 263.7 ms | 0.99× | 248.9 ms | 224.3 ms | **1.11×** |
| `q5` | 743.0 ms | 760.5 ms | 0.98× | 743.9 ms | 410.0 ms | **1.81×** |
| `q6` | 338.1 ms | 335.8 ms | 1.01× | 491.7 ms | 466.3 ms | **1.05×** |
| `q7` | 1,338.0 ms | 1,355.3 ms | 0.99× | 1,355.0 ms | 438.3 ms | **3.09×** |
| `q8` | 867.6 ms | 879.1 ms | 0.99× | 884.4 ms | 570.4 ms | **1.55×** |
| `q9` | 1,887.4 ms | 1,893.1 ms | 1.00× | 1,875.5 ms | 1,939.3 ms | 0.97× |
| `q10` | 798.5 ms | 799.3 ms | 1.00× | 769.6 ms | 662.5 ms | **1.16×** |
| `q11` | 151.0 ms | 151.0 ms | 1.00× | 153.6 ms | 140.4 ms | **1.09×** |
| `q12` | 858.5 ms | 215.6 ms | **3.98×** | 902.4 ms | 258.9 ms | **3.49×** |
| `q13` | 890.5 ms | 895.1 ms | 0.99× | 864.2 ms | 892.3 ms | 0.97× |
| `q14` | 450.9 ms | 450.8 ms | 1.00× | 346.2 ms | 372.1 ms | 0.93× |
| `q15` | 695.3 ms | 696.3 ms | 1.00× | 903.9 ms | 896.4 ms | 1.01× |
| `q16` | 209.2 ms | 206.0 ms | 1.02× | 216.5 ms | 210.1 ms | 1.03× |
| `q17` | 2,399.2 ms | 2,433.9 ms | 0.99× | 2,413.3 ms | 2,307.5 ms | **1.05×** |
| `q18` | 2,799.7 ms | 2,845.3 ms | 0.98× | 2,808.2 ms | 1,586.4 ms | **1.77×** |
| `q19` | 512.2 ms | 507.8 ms | 1.01× | 671.8 ms | 401.5 ms | **1.67×** |
| `q20` | 1,038.0 ms | 1,038.9 ms | 1.00× | 1,061.3 ms | 1,041.9 ms | **1.02×** |
| `q21` | 4,473.6 ms | 2,763.4 ms | **1.62×** | 4,508.8 ms | 2,649.6 ms | **1.70×** |
| `q22` | 132.8 ms | 135.1 ms | 0.98× | 138.3 ms | 118.1 ms | **1.17×** |
| **22-Q Geomean** | **676.8 ms** | **624.2 ms** | **1.08×** | **702.3 ms** | **534.3 ms** | **1.31×** |
| **22-Q Sum** | **22,347.0 ms** | **20,128.8 ms** | **1.11×** | **22,897.5 ms** | **16,931.9 ms** | **1.35×** |
| `q6_count` | 190.2 ms | 184.6 ms | 1.03× | 249.0 ms | 219.3 ms | **1.14×** |
| `sf5_dyn_orderkey_range` | 313.7 ms | 69.7 ms | **4.50×** | 316.6 ms | 70.6 ms | **4.48×** |

## Key Takeaways

- **TPC-H Q3 (`548.6 → 392.5 ms`, `1.40×`, `−39.0%` CPU):** `customer` pushes an
  L2-resident `o_custkey` Bloom filter into `orders` and `orders ⋈ customer`
  pushes an `l_orderkey` Bloom filter into `lineitem`, pruning ~95% of
  `lineitem` rows inside `RowFilter` before decoding `l_extendedprice` /
  `l_discount` and before probing `HashJoin`. Morsel now beats `stock_default`
  (`497.0 ms`) by **`1.27×`** and uses **9.2% less CPU** than `stock_pushdown`
  (`2,154.5` vs `2,373.6 ms`).
- **TPC-H Q5 (`735.3 → 409.5 ms`, `1.80×`, `−43.8%` CPU):** Reverse-order build
  execution cascades dynamic filters down the entire 6-way join chain
  (`region(1) → nation(5) → supplier(10K) → customer(150K) → orders(228K) →
  lineitem`), shrinking `join_finalize` by **5.6×** (`130.9 → 23.2 ms`) and
  beating **`stock_pushdown` (`464.9 ms`) by `1.14×`** (`−55.4 ms`, `−18.9%`
  CPU) and **`stock_default` (`682.7 ms`) by `1.67×`**.
- **TPC-H Q6 (`484.7 → 455.4 ms`, `−29.3 ms`) & `q6_count` (`244.3 → 221.4 ms`,
  `−22.9 ms`, `−11.9%` CPU):** Grouping same-column `RowFilter` conjuncts
  collapses Q6's 5 predicates into 3 (`l_shipdate`, `l_discount`, `l_quantity`),
  decoding each predicate column once and beating `stock_pushdown` by
  **`1.09×–1.16×`**.
- **TPC-H Q12 (`895.7 → 251.9 ms`, `3.56×` pushdown; `847.1 → 211.5 ms`,
  `4.01×` pruning):** Discounting filtered scans places selective `lineitem`
  (~155K rows) on build instead of unfiltered `orders` (7.5M rows), dropping
  `join_finalize` from `616.8 → 13.3 ms` (**46×**) and pushing a 512 KiB
  L2-resident `o_orderkey` Bloom filter into `orders`, beating `stock_pushdown`
  (`283.2 ms`) by **`1.12×`**.
- **Dynamic row-group statistics pruning (`sf5_dyn_orderkey_range`):** As
  analyzed in `#170`, standard TPC-H `o_orderdate` / `l_receiptdate` predicates
  are independent of primary keys `o_orderkey` / `l_orderkey`, so Q3/Q5/Q12
  scan the same Parquet bytes and gain from worker-side Bloom `RowFilter` +
  smaller hash builds. On key-clustered joins (`sf5_dyn_orderkey_range`),
  dynamic min/max pushdown prunes **235 / 240 `lineitem` row groups** before
  probe I/O starts, slashing `bytes_scanned` by **`−89.2%`** (`237.4 MB → 25.6 MB`)
  and wall time by **`4.51×`** (`310.7 → 68.9 ms`), beating `stock_default`
  (`109.8 ms`) by **`1.59×`** and `stock_pushdown` (`125.8 ms`) by **`1.80×`**.

## Reproduction Commands

```bash
set -euo pipefail
work="$HOME/.cache/issue193"
mkdir -p "$work"

# 1. Generate SF=5 TPC-H Parquet tables once.
uv venv --python 3.12 "$work/venv"
uv pip install --python "$work/venv/bin/python" -r benchmarks/morsel_tpch/requirements.txt
"$work/venv/bin/python" -B benchmarks/morsel_tpch/harness.py generate \
  "$work/tpch_sf5" --sf 5 --row-group-size 131072

# 2. Build detached `before` (3ba7893d9) and `after` (HEAD) release runners.
for pair in "before:3ba7893d9" "after:HEAD"; do
  label=${pair%%:*}; rev=${pair#*:}
  git worktree add --detach "$work/${label}_wt" "$rev"
  mkdir -p "$work/${label}_wt/datafusion/core/examples"
  cp benchmarks/morsel_tpch/tpch_runner.rs \
     "$work/${label}_wt/datafusion/core/examples/morsel_tpch_runner.rs"
  (cd "$work/${label}_wt" && CARGO_TARGET_DIR="$work/target_$label" \
    cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features \
      --features parquet,sql,datetime_expressions,unicode_expressions \
      --example morsel_tpch_runner)
  cp "$work/target_$label/release-nonlto/examples/morsel_tpch_runner" \
     "$work/runner_$label"
  git worktree remove --force "$work/${label}_wt"
done

# 3. Run paired warm-cache benchmark on 8 pinned cores (`taskset -c 0-7`).
#    For each query in q3, q5, q6, q6_count, q12, and sf5_dyn_orderkey_range,
#    invoke `$work/runner_{before,after} measure required {pruning,pushdown} 8 ...`
#    for 1 warmup + 5 counterbalanced rounds after verifying exact Arrow IPC parity.
```
