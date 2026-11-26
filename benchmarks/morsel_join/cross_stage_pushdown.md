<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel SF=5 Benchmark: Cross-`Aggregate` Dynamic Join Filter Pushdown (#206)

This note records a paired quiet-host before/after benchmark at **TPC-H Scale
Factor 5 (`--sf 5`)** measuring cross-`Aggregate` / cross-stage dynamic join
filter propagation and join build-side estimation improvements in `#206`:

1. **Cross-`Aggregate` / cross-stage dynamic join filter pushdown (`parquet.rs`,
   `planner.rs`, `dag.rs`, `morsel.rs`):** A whole-DAG dynamic join filter
   binding pass propagates finalized build-side filters (`min/max` bounds,
   `InList`, and `SplitBlockBloomFilter`) across single-consumer `StageSource`
   boundaries backed by grouped `Aggregate`, `GroupCount`, `Sort`, or `Result`
   stages (strictly skipping ungrouped aggregates, `TopN`/`Limit` stages,
   `HAVING`-altered or aggregate-derived key expressions, probe-preserving
   joins, and multi-consumer stages). At runtime, `Pipeline::prepare_execution`
   executes join build pipelines before ensuring the probe `StageSource` so
   published filters prune upstream `ParquetScan`s before aggregation (and
   empty-build short-circuits skip the aggregate stage completely).
2. **Cascading `BuildSemi`/`BuildAnti` build-side filter pushdown (`parquet.rs`):**
   Downstream probe-filtering joins (such as `supplier ⋈ nation` in TPC-H Q20)
   now cascade dynamic filters into earlier `BuildSemi` / `BuildAnti` build
   pipelines (`partsupp ⋉ part`), shrinking both the intermediate join build
   and the dynamic filter pushed across `stage_0` into `lineitem`.
3. **Selective join build-side estimation (`morsel.rs`):**
   `estimate_join_side_size` now accounts for equality predicate selectivity,
   propagates filter selectivity across probe-filtering `Inner` / `ProbeSemi`
   joins, bounds `BuildSemi` / `BuildAnti` outputs by their build side, and
   prefers placing non-`Stage` inputs on build on exact ties. This places the
   filtered join side on build in TPC-H **Q17** (`lineitem ⋈ part`), **Q2**
   (`partsupp ⋈ part ⋈ ...`), and **Q21** (`(l1 ⋈ supplier ⋈ orders ⋈ nation) ⋉
l2` instead of 19M-row `l3`).

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1014-aws`, 16
  ARM Neoverse-N1 vCPUs (1 thread/core), `aarch64`.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated CPUs), 8 Tokio
  workers (`worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`; DuckDB `SET threads = 8`), `batch_size=8,192`.
- **Host idle verification (`vmstat -y 1 3`):** **96–97% idle** immediately
  before the benchmark and **99–100% idle** immediately after.
- **Revisions compared (same `benchmarks/morsel_tpch/tpch_runner.rs`,
  `--profile release-nonlto`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, DuckDB
  `2.0.1`):**
  - **Before (`morsel-execution` base):** `836e975c5`
  - **After (`agent/morsel-issue-206`):** this PR
- **Dataset:** Standard deterministic SF=5 TPC-H Parquet tables (`35` Zstd
  level-3 files, `353` row groups at `row_group_size=131,072`, `43,299,825`
  rows, `1.34 GiB`). Linux `mincore` verified **100% page residency
  (`341,955 / 341,955` 4 KiB pages)** both before and after the benchmark.
- **Protocol:** Exact Arrow IPC result parity verified against Stock DataFusion
  (`stock_pushdown`) across **all 22/22 SF=5 TPC-H queries** before timing
  (plus all 21/21 SF=0.01 DuckDB parity/spill harness unittests). Fresh process
  per trial with `MORSEL_METRICS=1`, 1 unmeasured warmup round + 5
  counterbalanced measured rounds for the primary focus set (`q17`, `q20`,
  `q21`, `q9`, `q2`) and 1 warmup + 3 counterbalanced measured rounds for the
  full 22/22 regression check. Timings are medians in milliseconds (`min–max`
  in parentheses); CPU is median process `exec_cpu_seconds`.

## Results: Primary Paired Comparison (`q17`, `q20`, `q21`, `q9`, `q2`)

### `morsel_pushdown` (`pushdown_filters=true`) Before vs. After (5 Rounds)

| Query   | Before wall ms (min–max) | After wall ms (min–max) |         Wall speedup | Before CPU ms | After CPU ms |  CPU delta | Process work Before → After | `join_finalize` Before → After | `stock_pushdown` | `stock_default` | DuckDB 2.0.1 |
| ------- | -----------------------: | ----------------------: | -------------------: | ------------: | -----------: | ---------: | --------------------------: | -----------------------------: | ---------------: | --------------: | -----------: |
| **q17** |  1,298.2 (1249.0–1325.2) | **482.5 (477.6–488.3)** | **2.69×** (−815.7ms) |       7,326.6 |  **3,418.4** | **−53.3%** |       2,261.9 → **14.4 ms** |              28.0 → **0.8 ms** |         457.4 ms |      1,277.8 ms |     410.4 ms |
| **q20** |      780.8 (738.9–794.5) | **311.2 (310.0–342.8)** | **2.51×** (−469.6ms) |       3,851.6 |  **2,016.4** | **−47.6%** |          808.3 → **6.8 ms** |               1.4 → **0.7 ms** |         309.9 ms |        478.2 ms |     301.7 ms |
| **q21** |  1,550.1 (1538.1–1586.3) | **948.4 (934.9–973.2)** | **1.63×** (−601.7ms) |       8,116.7 |  **6,186.2** | **−23.8%** |        944.5 → **639.8 ms** |           532.5 → **153.0 ms** |       1,555.6 ms |      1,427.5 ms |     738.4 ms |
| **q9**  |  1,453.0 (1420.5–1459.6) | 1,436.9 (1420.8–1463.2) |      1.01× (−16.1ms) |       9,463.9 |      9,388.1 |      −0.8% |        2,351.5 → 2,319.1 ms |               296.2 → 290.5 ms |       1,586.7 ms |      1,079.1 ms |     824.9 ms |
| **q2**  |      191.3 (183.8–206.1) | **164.6 (161.0–165.8)** |  **1.16×** (−26.7ms) |         963.1 |    **794.6** | **−17.5%** |            137.6 → 126.0 ms |              22.5 → **1.9 ms** |         164.0 ms |        173.3 ms |     130.4 ms |

### Companion `morsel_pruning` (`pushdown_filters=false, pruning=true`) Before vs. After (5 Rounds)

| Query   | Before wall ms (min–max) |     After wall ms (min–max) |   Speedup |
| ------- | -----------------------: | --------------------------: | --------: |
| **q17** |  1,360.7 (1336.9–1407.5) |     1,352.1 (1341.4–1488.7) |     1.01× |
| **q20** |      746.0 (718.1–774.4) |         732.5 (706.2–740.9) |     1.02× |
| **q21** |  2,213.5 (2168.4–2227.5) | **1,851.7 (1816.5–1862.1)** | **1.20×** |
| **q9**  |  1,412.8 (1398.9–1432.8) |     1,414.3 (1410.6–1440.4) |     1.00× |
| **q2**  |      246.3 (243.6–252.2) |         248.5 (246.7–253.8) |     0.99× |

_(Note: standard TPC-H primary/foreign keys `l_partkey`, `l_suppkey`, and
`l_orderkey` span the full key domain across all row groups, so `pruning=true`
with `pushdown_filters=false` sees flat `bytes_scanned` and Q21's `1.20×`
pruning gain comes purely from eliminating the 19M-row `l3` hash build.)_

## Full 22-Query SF=5 Regression & 3-Way Check (3 Counterbalanced Rounds)

Single-config `morsel_pushdown` now **beats `stock_pushdown`** on both 22/22
geometric mean (**`404.2` vs. `425.3 ms`, `0.95×`**) and 22/22 total wall time
(**`10,833.7` vs. `11,937.5 ms`, `0.91×`, −1.10 s**), beats `stock_default`
(`467.3 ms` geomean / `13,557.7 ms` sum) by **`1.16×` (`1.25×` on sum)**, and
closes the gap to DuckDB 2.0.1 (`318.7 ms`) from **`1.43×` → `1.27×`** with
zero regressions across the remaining 17 queries:

| Query       | Before `morsel_pushdown` | After `morsel_pushdown` | Before → After | `stock_pushdown` | `stock_default` |   DuckDB 2.0.1 | After vs. `stock_pushdown` | After vs. DuckDB |
| ----------- | -----------------------: | ----------------------: | -------------: | ---------------: | --------------: | -------------: | -------------------------: | ---------------: |
| `q1`        |                 725.8 ms |                711.5 ms |          1.02× |         757.6 ms |        793.3 ms |       481.1 ms |                  **0.94×** |            1.48× |
| `q2`        |                 187.3 ms |            **161.2 ms** |      **1.16×** |         159.4 ms |        170.6 ms |       122.5 ms |                      1.01× |            1.32× |
| `q3`        |                 398.6 ms |                393.7 ms |          1.01× |         392.7 ms |        529.0 ms |       359.9 ms |                      1.00× |            1.09× |
| `q4`        |                 212.3 ms |                215.9 ms |          0.98× |         279.4 ms |        186.5 ms |       194.7 ms |                  **0.77×** |            1.11× |
| `q5`        |                 438.7 ms |                440.6 ms |          1.00× |         488.5 ms |        732.2 ms |       386.0 ms |                  **0.90×** |            1.14× |
| `q6`        |                 475.4 ms |                473.6 ms |          1.00× |         508.5 ms |        364.5 ms |       206.0 ms |                  **0.93×** |            2.30× |
| `q7`        |                 428.1 ms |                427.2 ms |          1.00× |         505.8 ms |        934.7 ms |       425.0 ms |                  **0.84×** |            1.01× |
| `q8`        |                 550.8 ms |                549.4 ms |          1.00× |         653.8 ms |        667.1 ms |       468.5 ms |                  **0.84×** |            1.17× |
| `q9`        |               1,430.2 ms |              1,443.7 ms |          0.99× |       1,601.7 ms |      1,081.5 ms |       835.2 ms |                  **0.90×** |            1.73× |
| `q10`       |                 618.9 ms |                619.7 ms |          1.00× |         534.4 ms |        664.9 ms |       473.4 ms |                      1.16× |            1.31× |
| `q11`       |                 133.4 ms |                133.1 ms |          1.00× |         116.7 ms |        104.7 ms |        97.4 ms |                      1.14× |            1.37× |
| `q12`       |                 257.0 ms |                255.6 ms |          1.01× |         285.7 ms |        200.6 ms |       223.9 ms |                  **0.89×** |            1.14× |
| `q13`       |                 610.1 ms |                616.9 ms |          0.99× |         583.5 ms |        533.1 ms |       561.1 ms |                      1.06× |            1.10× |
| `q14`       |                 332.9 ms |                336.4 ms |          0.99× |         287.6 ms |        405.5 ms |       352.7 ms |                      1.17× |        **0.95×** |
| `q15`       |                 948.0 ms |                946.3 ms |          1.00× |         931.1 ms |        735.0 ms |       272.9 ms |                      1.02× |            3.47× |
| `q16`       |                 166.2 ms |                167.0 ms |          1.00× |         139.7 ms |        124.6 ms |       174.3 ms |                      1.20× |        **0.96×** |
| `q17`       |               1,284.5 ms |            **482.5 ms** |      **2.66×** |         451.9 ms |      1,261.3 ms |       413.6 ms |                      1.07× |            1.17× |
| `q18`       |                 694.8 ms |                674.4 ms |          1.03× |         826.0 ms |      1,494.8 ms |       480.3 ms |                  **0.82×** |            1.40× |
| `q19`       |                 411.2 ms |                405.2 ms |          1.01× |         407.5 ms |        555.1 ms |       335.1 ms |                  **0.99×** |            1.21× |
| `q20`       |                 767.0 ms |            **314.7 ms** |      **2.44×** |         304.0 ms |        470.7 ms |       307.6 ms |                      1.03× |            1.02× |
| `q21`       |               1,553.2 ms |            **946.7 ms** |      **1.64×** |       1,585.8 ms |      1,433.0 ms |       736.6 ms |                  **0.60×** |            1.29× |
| `q22`       |                 119.9 ms |                118.5 ms |          1.01× |         136.3 ms |        114.9 ms |       148.7 ms |                  **0.87×** |        **0.80×** |
| **GeoMean** |             **454.2 ms** |            **404.2 ms** |      **1.12×** |     **425.3 ms** |    **467.3 ms** |   **318.7 ms** |                  **0.95×** |        **1.27×** |
| **Sum**     |          **12,744.4 ms** |         **10,833.7 ms** |      **1.18×** |  **11,937.5 ms** | **13,557.7 ms** | **8,056.4 ms** |                  **0.91×** |        **1.34×** |

## Tight-Memory SF=5 Spill Verification (`RLIMIT_NOFILE=1024`)

Cross-`Aggregate` pushdown shrinks the grouped `Aggregate(lineitem)` stages in
Q17 and Q20 from 1M–1.4M groups down to ~1K–2K groups, eliminating spilling
entirely under the `#195` 128 MiB `GreedyMemoryPool` budget while Q21 continues
to spill cleanly at 256 MiB with verified exact Arrow IPC parity:

| Workload | Memory budget |    Wall ms | `spill_count` | `spilled_bytes` | Parity vs. Stock |
| -------- | ------------: | ---------: | ------------: | --------------: | ---------------- |
| `q17`    |       128 MiB |   483.8 ms |             0 |             0 B | Exact            |
| `q20`    |       128 MiB |   318.3 ms |             0 |             0 B | Exact            |
| `q21`    |       256 MiB | 3,991.1 ms |           224 |   516,746,288 B | Exact            |

## Analysis of Q17, Q20, Q21, Q2, and Q9

- **TPC-H Q17 (`1,298.2 → 482.5 ms`, `2.69×`, `−53.3%` CPU):** Putting
  `lineitem ⋈ part(Brand#23, MED BOX)` (~30K rows, ~1K distinct `p_partkey`s)
  on build and pushing its dynamic filter across `StageSource[stage_0]` into
  `stage_0`'s `lineitem` scan drops total `morsel_process_work_ms` by **157×**
  (`2,261.9 → 14.4 ms`) and `join_finalize` by **35×** (`28.0 → 0.8 ms`).
- **TPC-H Q20 (`780.8 → 311.2 ms`, `2.51×`, `−47.6%` CPU):** Cascading
  `supplier ⋈ nation('CANADA')` into `partsupp ⋉ part('forest%')` through
  `BuildSemi` shrinks the `partsupp` build to ~1.8K `(ps_partkey, ps_suppkey)`
  pairs, which then push across `StageSource[stage_0]` into `lineitem`, cutting
  `morsel_process_work_ms` by **119×** (`808.3 → 6.8 ms`) and matching both
  `stock_pushdown` (`309.9 ms`) and DuckDB 2.0.1 (`301.7 ms`).
- **TPC-H Q21 (`1,550.1 → 948.4 ms`, `1.63×`, `−23.8%` CPU):** Bounding
  `BuildSemi` outputs by their build side fixes a build-side inversion on the
  final `LeftAnti` join, moving `(l1 ⋈ supplier ⋈ orders ⋈ nation) ⋉ l2`
  (~380K rows) onto build (`BuildAnti`) instead of 19M-row `l3`. This cuts
  `join_finalize` by **3.48×** (`532.5 → 153.0 ms`), pushes an `l_orderkey`
  dynamic filter into `l3`, and beats `stock_pushdown` (`1,555.6 ms`) by
  **`1.64×`**.
- **TPC-H Q2 (`191.3 → 164.6 ms`, `1.16×`):** Places filtered
  `partsupp ⋈ part ⋈ supplier ⋈ nation ⋈ region` on build and pushes `p_partkey`
  bounds across `StageSource[stage_0]`, cutting `join_finalize` from
  `22.5 → 1.9 ms`.
- **TPC-H Q9 (`1,453.0 → 1,436.9 ms`, `0.90×` vs. `stock_pushdown`):**
  Inspection confirmed Q9 has no cross-stage boundary and no build-side
  inversion (all five dimension/bridge tables sit on build against 30M-row
  `lineitem`). Q9 already beats `stock_pushdown` (`1,586.7 ms`); its gap to
  `stock_default` (`1,079.1 ms`) stems from evaluating unfiltered 4M-row
  `partsupp`'s 8 MiB Bloom `RowFilter` on `lineitem` and not propagating
  sibling probe-key equivalence (`part.p_partkey = l_partkey =
partsupp.ps_partkey`) forward across builds, which is left for a separate
  targeted follow-up.

## Reproduction Commands

```bash
set -euo pipefail
work="$HOME/.cache/issue206"
data_dir="$HOME/.cache/issue193/tpch_sf5"
py="$HOME/.cache/issue193/venv/bin/python"
mkdir -p "$work"

# 1. Build detached before (836e975c5) and after (HEAD) release runners.
for pair in "before:836e975c5" "after:HEAD"; do
  label=${pair%%:*}; rev=${pair#*:}
  git worktree add --detach "$work/${label}_wt" "$rev"
  mkdir -p "$work/${label}_wt/datafusion/core/examples"
  cp benchmarks/morsel_tpch/tpch_runner.rs \
     "$work/${label}_wt/datafusion/core/examples/morsel_tpch_runner.rs"
  (
    cd "$work/${label}_wt"
    MORSEL_BENCH_REVISION="$rev" CARGO_TARGET_DIR="$work/target_$label" \
      RUSTC_WRAPPER=sccache \
      cargo build --locked --profile release-nonlto -j 10 \
        -p datafusion --no-default-features \
        --features parquet,sql,datetime_expressions,unicode_expressions \
        --example morsel_tpch_runner
  )
  cp "$work/target_$label/release-nonlto/examples/morsel_tpch_runner" \
     "$work/runner_$label"
  git worktree remove --force "$work/${label}_wt"
done

# 2. Verify SF=0.01 DuckDB parity/spill unittests (21/21).
MORSEL_TPCH_RUNNER="$work/runner_after" MORSEL_BENCH_REVISION="$(git rev-parse HEAD)" \
  "$py" -B -m unittest discover -s benchmarks/morsel_tpch -v

# 3. Verify 22/22 SF=5 Arrow IPC parity and run quiet-host paired benchmark
#    (`taskset -c 0-7`, `--threads 8`, `MORSEL_METRICS=1`, 1 warmup + 5 focus
#    rounds and 1 warmup + 3 full 22-query rounds, plus tight-memory spill check
#    for q17@128 MiB, q20@128 MiB, q21@256 MiB under `ulimit -n 1024`).
```
