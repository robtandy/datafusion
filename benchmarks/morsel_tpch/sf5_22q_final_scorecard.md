<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF5 Final 22/22 3-Way Scorecard: Post-Dynamic Join Pushdown & Radix Spilling (#195)

This scorecard reports the **final complete 22/22 standard TPC-H (`q1`–`q22`)
3-way comparison at Scale Factor 5 (`--sf 5`)** between **Morsel DataFusion
(`morsel_mode=required`)**, **Stock DataFusion (`morsel_mode=off`, both
`stock_default` and `stock_pushdown`)**, and **DuckDB 2.0.1** on quiet
`rob-tandy-dduck2` (8 pinned ARM Neoverse-N1 cores, verified 100% warm Linux
page cache via `mincore`) after **dynamic join filter pushdown + same-column
`RowFilter` grouping (`#193`)** and **parallel radix-partitioned hash tables +
out-of-core spilling (`#194`)** landed on `morsel-execution`.

It directly compares **before (`#192`, [sf5_22q_scorecard.md](sf5_22q_scorecard.md))**
vs. **after (post-`#194`)** on the exact same SF5 Parquet dataset and host,
tracks `bytes_scanned` and dynamic join filter pushdown reductions across all
22 TPC-H queries plus a key-clustered SF5 range join, and verifies
**tight-memory SF5 out-of-core spill completion** across `Aggregate`,
`HashJoin`, and `Sort` under `GreedyMemoryPool` and `RLIMIT_NOFILE=1024`.

## Executive Summary

- **Morsel DataFusion now beats unmodified Stock DataFusion (`stock_default`)
  across the full 22/22 TPC-H SF5 suite — both in a single configuration
  (`morsel_pushdown`) and in best-per-query mode:**
  - **Single-config `morsel_pushdown` (`morsel_mode=required`):** 22/22
    geometric mean drops from **`716.3 ms → 461.0 ms` (`1.55×` faster)** and
    22-query sum drops from **`23,240.9 ms → 12,757.6 ms` (`1.82×` faster,
    −10.48 s!)**, beating **`stock_default` (`474.2 ms` geomean / `13,398.7 ms`
    sum) at `0.97×` (`0.95×` on sum)** and slashing the gap to DuckDB 2.0.1
    (`311.6 ms`) from **`2.36× → 1.48×`**.
  - **Best Morsel per query (`morsel_mode=required`):** 22/22 geometric mean
    drops from **`679.7 ms → 443.2 ms` (`1.53×` faster)** and 22-query sum
    drops from **`22,490.5 ms → 12,247.1 ms` (`1.84×` faster, −10.24 s!)**,
    beating **`stock_default` (`474.2 ms`) at `0.93×` (`1.07×` faster; `0.91×`
    on sum)**, landing within **`1.02×` of `stock_pushdown` (`434.2 ms`)**, and
    narrowing the gap to DuckDB 2.0.1 from **`2.24× → 1.42×`**.
- **100% native admission, 100% DuckDB parity, 100% warm cache, zero output
  backpressure:** All `22 / 22` standard TPC-H queries passed lossless Arrow
  IPC parity against **DuckDB 2.0.1** at both `--sf 0.01` (21/21 `morsel_tpch`
  unittests passing) and `--sf 5` across all 6 arms, executed strictly under
  `MorselExec: workers=8` with `morsel_mode=required` (`0` fallbacks, `0`
  rejections), maintained **100% `mincore` page residency on all `660 / 660`
  measured trials**, and recorded **`0.0 ms` `morsel_output_wait`** across all
  330 measured Morsel trials.

| Arm                                                | `#192` GeoMean | Post-`#194` GeoMean | `#192 → #194` Gain | Post-`#194` Sum (22/22) | Ratio vs. `stock_default` | Ratio vs. `stock_pushdown` | Ratio vs. `duckdb_default` |
| -------------------------------------------------- | -------------: | ------------------: | -----------------: | ----------------------: | ------------------------: | -------------------------: | -------------------------: |
| Stock DataFusion (`stock_default`, `off`)          |       462.6 ms |            474.2 ms |                  — |             13,398.7 ms |                     1.00× |                      1.09× |                      1.52× |
| Stock DataFusion (`stock_pushdown`, `off`)         |              — |            434.2 ms |                  — |             11,944.4 ms |                     0.92× |                      1.00× |                      1.39× |
| Morsel DataFusion (`morsel_no_pushdown`, `req`)    |       690.7 ms |            563.9 ms |          **1.22×** |             15,901.3 ms |                     1.19× |                      1.30× |                      1.81× |
| Morsel DataFusion (`morsel_pruning`, `req`)        |              — |            568.4 ms |                  — |             16,057.2 ms |                     1.20× |                      1.31× |                      1.82× |
| **Morsel DataFusion (`morsel_pushdown`, `req`)**   |       716.3 ms |        **461.0 ms** |          **1.55×** |         **12,757.6 ms** |                 **0.97×** |                      1.06× |                      1.48× |
| **Best Morsel per query (`morsel_mode=required`)** |       679.7 ms |        **443.2 ms** |          **1.53×** |         **12,247.1 ms** |                 **0.93×** |                      1.02× |                  **1.42×** |
| **DuckDB 2.0.1 (`duckdb_default`)**                |       304.0 ms |        **311.6 ms** |                  — |          **7,884.5 ms** |                 **0.66×** |                  **0.72×** |                  **1.00×** |

- **Headline per-query wins and speedups (`#192` → post-`#194`):**
  - **Massive speedups across the `#192` bottleneck queries:** **`q18`**
    `2,812.5 → 693.3 ms` (**`4.06×`**), **`q12`** `856.8 → 231.6 ms`
    (**`3.70×`**), **`q7`** `1,361.1 → 444.4 ms` (**`3.06×`**), **`q21`**
    `4,418.0 → 1,544.1 ms` (**`2.86×`**, `−2,873.9 ms`), **`q17`**
    `2,506.7 → 1,238.2 ms` (**`2.02×`**, `−1,268.4 ms`), **`q5`**
    `756.7 → 441.3 ms` (**`1.71×`**), **`q13`** `930.9 → 577.0 ms`
    (**`1.61×`**), **`q8`** `876.0 → 554.4 ms` (**`1.58×`**), **`q20`**
    `1,065.2 → 732.1 ms` (**`1.46×`**), **`q2`** `282.3 → 194.2 ms`
    (**`1.45×`**), **`q9`** `1,899.5 → 1,378.7 ms` (**`1.38×`**), **`q3`**
    `537.1 → 401.3 ms` (**`1.34×`**), **`q10`** `780.9 → 609.0 ms`
    (**`1.28×`**), and **`q19`** `519.7 → 414.5 ms` (**`1.25×`**).
  - **Head-to-head wins vs. Stock DataFusion and DuckDB 2.0.1:**
    - Best Morsel beats **`stock_default` on `11 / 22` queries** (up from
      `5 / 22` in `#192`): `q1` (`0.93×`), `q3` (`0.74×`), `q5` (`0.62×`),
      `q6` (`0.94×`), `q7` (`0.49×` / `2.05×` faster!), `q8` (`0.84×`), `q10`
      (`0.94×`), `q14` (`0.84×`), `q15` (`0.96×`), `q18` (`0.46×` / `2.15×`
      faster!), and `q19` (`0.74×`).
    - Best Morsel beats **`stock_pushdown` on `12 / 22` queries** (`q1`, `q4`,
      `q5`, `q6`, `q7`, `q8`, `q9`, `q12`, `q15`, `q18`, `q19`, `q22`);
      single-config `morsel_pushdown` beats `stock_pushdown` on **`11 / 22`**
      queries!
    - Morsel **beats DuckDB 2.0.1 on `q22`** (`127.7` vs. `147.2 ms`,
      **`0.87×`** / `1.15×` faster) and lands within **1–8% of DuckDB 2.0.1**
      on **`q14`** (`339.2` vs. `335.8 ms`, `1.01×`), **`q12`** (`231.6` vs.
      `225.2 ms`, `1.03×`), **`q7`** (`444.4` vs. `418.8 ms`, `1.06×`), and
      **`q13`** (`577.0` vs. `531.8 ms`, `1.08×`).
  - **Elimination of serial coordinator bottlenecks:** Total `join_finalize`
    wall time across all 22 queries fell **`6.89×`** from **`7,258.4 ms →
1,053.4 ms` (`−6,205.0 ms`)**, while total process CPU dropped **`−17.8%`**
    (`93.3 s → 76.7 s`).
- **Dynamic join filter pushdown & `bytes_scanned` reductions:** Morsel pushed
  dynamic join filters on **`19 / 22` standard TPC-H queries** (all join
  queries with Parquet probes), cutting `bytes_scanned` on standard TPC-H
  `q18` (`−703 KB`, `550 → 547` RGs), `q19` (`−224 KB`, below `stock_default`),
  and `q3` (`−30 KB` vs. `morsel_no_pushdown`), and skipping millions of
  payload column decodes via L2-resident Bloom `RowFilter`s. On the
  key-clustered SF5 range join (`sf5_dyn_orderkey_range`), dynamic min/max
  pushdown pruned **`235 / 240` `lineitem` row groups**, slashing
  `bytes_scanned` by **`−89.2%`** (`237.4 MB → 25.6 MB`) and wall time by
  **`4.40×`** (`325.9 → 74.1 ms`, beating `stock_default` `117.9 ms` by
  **`1.59×`** and `stock_pushdown` `140.9 ms` by **`1.90×`**).
- **Tight-memory SF5 spill completion (`9 / 9` workloads, `RLIMIT_NOFILE=1024`):**
  Every tight-memory SF5 workload (`q3` @ 32 MiB, `q13` @ 64 MiB, `q10`/`q17`/
  `q18`/`q20`/`sf5_orders_sort` @ 128 MiB, `q9`/`q21` @ 256 MiB) completed
  cleanly under strict `GreedyMemoryPool` budgets (`142.3 MiB` to `1,787.2 MiB`
  spilled across `64–458` bounded spill files) with **100% verified exact
  result parity** against DuckDB 2.0.1 / Stock DataFusion.

## Environment, Controls, and Reproduction Commands

### Host and Build Metadata

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated CPUs),
  `--threads 8` (Tokio `worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`; DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification (`vmstat -y 1 3`):**
  - Immediately before 22/22 scorecard: **89–92% idle** (`us 7–11%, sy 0–1%`).
  - Immediately after 22/22 scorecard: **96–99% idle** (`us 1–4%, sy 0%`).
- **Build & versions:**
  - DataFusion revision: `d78ecdab8e9631fd4f87b839a6118aa8be7d9405`
    (`agent/morsel-issue-195`, runner spill/dynamic-filter metric reporting on
    top of `morsel-execution` HEAD `c9fb014ce` after `#193` and `#194`), built
    with `--locked --profile release-nonlto -p datafusion --no-default-features
--features parquet,sql,datetime_expressions,unicode_expressions --example
morsel_tpch_runner`.
  - Rust `1.99.1` (`58cf8f9c3 2026-10-13`, `aarch64-unknown-linux-gnu`), Cargo
    `1.99.1`, Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`.
- **Dataset & page-cache verification:** Deterministic `#165`/`#189`/`#192` SF5
  TPC-H Parquet dataset (`43,299,825` rows, `35` Zstd level-3 files, `353` row
  groups at `row_group_size=131,072`, `1,400,570,073` bytes / `1,335.7 MiB` on
  local EC2 NVMe Btrfs storage under `$HOME/.cache/issue189/tpch_sf5`). All
  **`660 / 660` measured trials** started and ended with **100% of 4 KiB
  Parquet file pages resident** via Linux `mincore`.
- **Protocol:** Fresh process per trial, `1` unmeasured warmup round + `5`
  measured counterbalanced rounds (alternating arm order between odd and even
  rounds) with `MORSEL_METRICS=1` enabled on every Morsel trial. All timings
  below are medians in milliseconds with observed `min–max` in parentheses.

### Exact Reproduction Commands

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work="$HOME/.cache/issue195"
mkdir -p "$work"
revision=$(git rev-parse HEAD)

# 1. Build detached tpch_runner with full TPC-H expression features.
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
runner="$work/target/release-nonlto/examples/morsel_tpch_runner"

# 2. Create pinned Python venv and run all 21/21 SF=0.01 parity/plan/spill unittests.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 240s "$py" -B -m unittest discover -s "$b" -v

# 3. Generate deterministic SF5 TPC-H Parquet tables outside Git (or reuse existing SF5 dir).
data_dir="$HOME/.cache/issue189/tpch_sf5"
# timeout 600s "$py" -B "$b/harness.py" generate "$data_dir" --sf 5 --row-group-size 131072

# 4. Run quiet-host warm-cache SF5 22/22 6-arm scorecard (660 timed trials).
vmstat -y 1 3
timeout --kill-after=10s 2400s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries all \
  --morsel-arms stock_default,stock_pushdown,morsel_no_pushdown,morsel_pruning,morsel_pushdown,duckdb_default \
  > "$work/sf5_22q_final_scorecard.jsonl"
vmstat -y 1 3

# 5. Print summary and 3-way geometric mean table.
"$py" -B "$b/harness.py" summarize "$work/sf5_22q_final_scorecard.jsonl"

# 6. Example tight-memory SF5 spill invocation (repeat for q3@32, q13@64, q10/q17/q18/q20@128, q9/q21@256).
(
  ulimit -n 1024
  MORSEL_MEMORY_LIMIT_MB=128 taskset -c 0-7 "$runner" measure required pushdown 8 \
    "$data_dir" "customer,orders,lineitem" "$(cat benchmarks/queries/q18.sql)"
)

# 7. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```

## 1. Full 22/22 TPC-H SF5 3-Way Warm-Cache Scorecard (Post-`#194`)

Every query verified exact Arrow IPC parity against DuckDB 2.0.1 across all 6
arms before timing and ran with `morsel_mode=required` (`0` fallback, `100%`
`mincore` page residency on all `660 / 660` trials). Ratios compare the faster
Morsel arm (`Best Morsel`) against `stock_default`, `stock_pushdown`, and
`duckdb_default` (`< 1.00×` means Morsel is faster; bold highlights wins).

| Query               |   Rows | `stock_default` ms (min–max) | `stock_pushdown` ms (min–max) | `morsel_no_pushdown` ms (min–max) | `morsel_pruning` ms (min–max) | `morsel_pushdown` ms (min–max) | `duckdb_default` ms (min–max) | Best / Stock Def | Best / Stock PD | Best / DuckDB |
| ------------------- | -----: | ---------------------------: | ----------------------------: | --------------------------------: | ----------------------------: | -----------------------------: | ----------------------------: | ---------------: | --------------: | ------------: |
| `q1`                |      4 |          776.3 (773.1–807.8) |           762.7 (756.6–787.0) |           **720.3** (713.7–734.6) |           725.8 (717.4–734.3) |        **721.7** (709.5–727.8) |           463.0 (460.4–468.2) |        **0.93×** |       **0.94×** |         1.56× |
| `q2`                |    100 |          179.8 (176.6–188.3) |           171.8 (169.7–183.2) |               250.9 (231.4–257.0) |           249.5 (232.3–254.8) |            194.2 (186.0–239.0) |           127.5 (120.7–130.8) |            1.08× |           1.13× |         1.52× |
| `q3`                |     10 |          542.4 (536.4–546.9) |           393.3 (390.5–402.4) |               564.6 (560.7–569.8) |           563.6 (556.2–573.2) |        **401.3** (392.1–407.4) |           355.5 (350.0–358.8) |        **0.74×** |           1.02× |         1.13× |
| `q4`                |      5 |          206.2 (204.1–229.7) |           295.6 (286.6–320.8) |               312.9 (303.3–332.4) |           312.3 (298.5–329.9) |        **226.2** (223.8–228.2) |           188.6 (184.7–219.1) |            1.10× |       **0.77×** |         1.20× |
| `q5`                |      5 |          716.3 (691.9–735.7) |           484.0 (473.0–493.7) |               742.5 (733.9–749.3) |           751.1 (729.4–796.9) |        **441.3** (437.0–453.2) |           377.9 (373.9–381.9) |        **0.62×** |       **0.91×** |         1.17× |
| `q6`                |      1 |          377.1 (363.6–379.8) |           517.6 (513.3–541.4) |               357.7 (356.7–371.5) |       **353.3** (349.3–359.0) |            480.2 (472.3–499.0) |           207.2 (205.1–207.9) |        **0.94×** |       **0.68×** |         1.71× |
| `q7`                |      4 |          909.0 (894.5–944.1) |           513.1 (503.3–528.0) |         1,040.6 (1,001.4–1,059.0) |     1,069.7 (1,036.6–1,077.4) |        **444.4** (430.3–447.5) |           418.8 (406.8–421.3) |        **0.49×** |       **0.87×** |         1.06× |
| `q8`                |      2 |          658.2 (655.5–670.0) |           655.9 (641.6–673.8) |               717.2 (706.8–738.0) |           713.4 (709.7–732.4) |        **554.4** (534.7–563.5) |           454.7 (448.4–471.2) |        **0.84×** |       **0.85×** |         1.22× |
| `q9`                |    175 |    1,044.8 (1,030.2–1,061.1) |     1,555.9 (1,521.3–1,639.9) |     **1,378.7** (1,344.1–1,404.3) |     1,383.4 (1,313.3–1,410.5) |      1,418.0 (1,391.7–1,431.2) |           808.1 (771.5–811.5) |            1.32× |       **0.89×** |         1.71× |
| `q10`               |     20 |          650.4 (635.1–685.6) |           532.4 (525.9–587.9) |               767.0 (751.8–785.4) |           760.2 (754.1–793.9) |        **609.0** (597.9–621.0) |           453.5 (446.3–459.8) |        **0.94×** |           1.14× |         1.34× |
| `q11`               |  4,606 |          115.7 (107.6–120.8) |           128.3 (126.2–132.8) |               200.3 (196.5–204.1) |           206.4 (196.9–222.2) |            144.4 (140.1–147.2) |             97.8 (91.5–101.9) |            1.25× |           1.13× |         1.48× |
| `q12`               |      2 |          222.3 (220.5–224.6) |           298.5 (296.6–308.2) |               237.1 (223.6–238.2) |       **231.6** (229.1–244.5) |            267.7 (263.7–313.8) |           225.2 (221.7–227.1) |            1.04× |       **0.78×** |         1.03× |
| `q13`               |     42 |          499.2 (481.6–543.8) |           557.9 (535.5–588.1) |               577.0 (554.6–618.3) |           594.9 (542.0–629.7) |            596.8 (570.7–598.9) |           531.8 (522.4–564.0) |            1.16× |           1.03× |         1.08× |
| `q14`               |      1 |          402.6 (400.9–407.5) |           299.5 (295.2–319.6) |               423.6 (416.6–427.0) |           423.4 (415.2–424.6) |        **339.2** (335.7–351.6) |           335.8 (329.0–365.0) |        **0.84×** |           1.13× |         1.01× |
| `q15`               |      1 |          743.0 (737.0–747.5) |         936.6 (934.9–1,034.4) |           **710.3** (703.5–719.2) |           726.7 (705.7–754.3) |            955.3 (933.3–965.7) |           270.7 (267.0–283.5) |        **0.96×** |       **0.76×** |         2.62× |
| `q16`               | 27,724 |          136.9 (136.0–142.5) |           148.1 (144.8–149.5) |               173.0 (171.5–174.7) |           174.8 (167.7–181.7) |            171.9 (169.8–177.2) |           144.6 (140.5–148.3) |            1.26× |           1.16× |         1.19× |
| `q17`               |      1 |    1,201.5 (1,168.3–1,218.4) |           463.1 (455.2–508.1) |         1,387.0 (1,361.1–1,404.8) |     1,378.4 (1,309.7–1,404.0) |      1,238.2 (1,192.7–1,284.1) |           401.6 (399.8–405.6) |            1.03× |           2.67× |         3.08× |
| `q18`               |    100 |    1,493.2 (1,458.5–1,514.9) |           840.7 (828.3–851.4) |         1,883.1 (1,868.5–1,896.1) |     1,881.0 (1,856.4–1,909.8) |        **693.3** (681.3–705.0) |           481.0 (474.4–504.9) |        **0.46×** |       **0.82×** |         1.44× |
| `q19`               |      1 |          562.4 (557.2–566.8) |           423.6 (420.1–433.2) |               528.6 (515.4–529.0) |           533.4 (525.3–540.5) |        **414.5** (410.0–432.5) |           329.3 (327.0–342.1) |        **0.74×** |       **0.98×** |         1.26× |
| `q20`               |    901 |          478.1 (469.6–508.4) |           320.5 (317.2–321.2) |               732.1 (724.9–769.5) |           749.4 (721.6–769.6) |            774.1 (750.8–782.8) |           305.4 (299.5–311.1) |            1.53× |           2.28× |         2.40× |
| `q21`               |    100 |    1,357.1 (1,338.7–1,396.0) |     1,502.4 (1,485.0–1,552.6) |         2,025.1 (2,009.3–2,127.1) |     2,097.6 (2,010.8–2,138.3) |      1,544.1 (1,485.2–1,570.8) |           759.7 (740.6–762.2) |            1.14× |           1.03× |         2.03× |
| `q22`               |      7 |          126.1 (123.8–130.8) |           142.8 (142.1–155.2) |               171.8 (170.6–180.7) |           177.4 (176.5–187.9) |        **127.7** (127.0–128.7) |           147.2 (143.6–148.4) |            1.01× |       **0.89×** |     **0.87×** |
| **GeoMean (22/22)** |      — |                 **474.2 ms** |                  **434.2 ms** |                      **563.9 ms** |                  **568.4 ms** |                   **461.0 ms** |                  **311.6 ms** |        **0.93×** |           1.02× |     **1.42×** |
| **Sum (22/22)**     |      — |              **13,398.7 ms** |               **11,944.4 ms** |                   **15,901.3 ms** |               **16,057.2 ms** |                **12,757.6 ms** |                **7,884.5 ms** |        **0.91×** |           1.03× |     **1.55×** |

_(Notice the complete reversal from `#192`: whereas `morsel_no_pushdown` won
`16 / 22` queries in `#192`, **`morsel_pushdown` now wins `15 / 22` queries**
and lowers the single-config Morsel geomean by `102.9 ms` vs.
`morsel_no_pushdown` (`461.0` vs. `563.9 ms`), because dynamic Bloom
`RowFilter` pushdown + same-column `RowFilter` grouping turns pushdown into a
major win on almost every join query.)_

## 2. Before vs. After Comparison (`#192` vs. Post-`#194`)

Comparing the initial 22/22 SF5 scorecard (`#192`, `3de519ad2`) directly
against post-`#194` (`d78ecdab8`) on `rob-tandy-dduck2`:

| Query               | `#192` `morsel_pushdown` | Post-`#194` `morsel_pushdown` | `pushdown` Speedup | `#192` Best Morsel | Post-`#194` Best Morsel | Best Speedup | Best Delta ms | `morsel_pushdown` / Stock Def | Best / Stock Def | Best / DuckDB |
| ------------------- | -----------------------: | ----------------------------: | -----------------: | -----------------: | ----------------------: | -----------: | ------------: | ----------------------------: | ---------------: | ------------: |
| `q1`                |                    713.2 |                         721.7 |              0.99× |              713.2 |                   720.3 |        0.99× |          +7.1 |                     **0.93×** |        **0.93×** |         1.56× |
| `q2`                |                    287.9 |                         194.2 |          **1.48×** |              282.3 |                   194.2 |    **1.45×** |         −88.2 |                         1.08× |            1.08× |         1.52× |
| `q3`                |                    562.0 |                         401.3 |          **1.40×** |              537.1 |                   401.3 |    **1.34×** |        −135.7 |                     **0.74×** |        **0.74×** |         1.13× |
| `q4`                |                    261.1 |                         226.2 |          **1.15×** |              261.1 |                   226.2 |    **1.15×** |         −35.0 |                         1.10× |            1.10× |         1.20× |
| `q5`                |                    760.4 |                         441.3 |          **1.72×** |              756.7 |                   441.3 |    **1.71×** |        −315.4 |                     **0.62×** |        **0.62×** |         1.17× |
| `q6`                |                    485.0 |                         480.2 |              1.01× |              347.7 |                   353.3 |        0.98× |          +5.6 |                         1.27× |        **0.94×** |         1.71× |
| `q7`                |                  1,411.6 |                         444.4 |          **3.18×** |            1,361.1 |                   444.4 |    **3.06×** |        −916.7 |                     **0.49×** |        **0.49×** |         1.06× |
| `q8`                |                    900.0 |                         554.4 |          **1.62×** |              876.0 |                   554.4 |    **1.58×** |        −321.6 |                     **0.84×** |        **0.84×** |         1.22× |
| `q9`                |                  1,903.7 |                       1,418.0 |          **1.34×** |            1,899.5 |                 1,378.7 |    **1.38×** |        −520.7 |                         1.36× |            1.32× |         1.71× |
| `q10`               |                    780.9 |                         609.0 |          **1.28×** |              780.9 |                   609.0 |    **1.28×** |        −172.0 |                     **0.94×** |        **0.94×** |         1.34× |
| `q11`               |                    160.7 |                         144.4 |          **1.11×** |              156.7 |                   144.4 |    **1.09×** |         −12.4 |                         1.25× |            1.25× |         1.48× |
| `q12`               |                    912.8 |                         267.7 |          **3.41×** |              856.8 |                   231.6 |    **3.70×** |        −625.2 |                         1.20× |            1.04× |         1.03× |
| `q13`               |                    930.9 |                         596.8 |          **1.56×** |              930.9 |                   577.0 |    **1.61×** |        −354.0 |                         1.20× |            1.16× |         1.08× |
| `q14`               |                    358.8 |                         339.2 |          **1.06×** |              358.8 |                   339.2 |    **1.06×** |         −19.6 |                     **0.84×** |        **0.84×** |         1.01× |
| `q15`               |                    902.3 |                         955.3 |              0.94× |              702.4 |                   710.3 |        0.99× |          +7.9 |                         1.29× |        **0.96×** |         2.62× |
| `q16`               |                    220.3 |                         171.9 |          **1.28×** |              212.1 |                   171.9 |    **1.23×** |         −40.3 |                         1.26× |            1.26× |         1.19× |
| `q17`               |                  2,506.7 |                       1,238.2 |          **2.02×** |            2,506.7 |                 1,238.2 |    **2.02×** |      −1,268.4 |                         1.03× |            1.03× |         3.08× |
| `q18`               |                  2,818.6 |                         693.3 |          **4.07×** |            2,812.5 |                   693.3 |    **4.06×** |      −2,119.3 |                     **0.46×** |        **0.46×** |         1.44× |
| `q19`               |                    680.8 |                         414.5 |          **1.64×** |              519.7 |                   414.5 |    **1.25×** |        −105.2 |                     **0.74×** |        **0.74×** |         1.26× |
| `q20`               |                  1,097.3 |                         774.1 |          **1.42×** |            1,065.2 |                   732.1 |    **1.46×** |        −333.1 |                         1.62× |            1.53× |         2.40× |
| `q21`               |                  4,450.8 |                       1,544.1 |          **2.88×** |            4,418.0 |                 1,544.1 |    **2.86×** |      −2,873.9 |                         1.14× |            1.14× |         2.03× |
| `q22`               |                    135.2 |                         127.7 |          **1.06×** |              135.0 |                   127.7 |    **1.06×** |          −7.3 |                         1.01× |            1.01× |     **0.87×** |
| **GeoMean (22/22)** |             **716.3 ms** |                  **461.0 ms** |          **1.55×** |       **679.7 ms** |            **443.2 ms** |    **1.53×** | **−236.5 ms** |                     **0.97×** |        **0.93×** |     **1.42×** |
| **Sum (22/22)**     |          **23,240.9 ms** |               **12,757.6 ms** |          **1.82×** |    **22,490.5 ms** |         **12,247.1 ms** |    **1.84×** | **−10,243.4** |                     **0.95×** |        **0.91×** |     **1.55×** |

_(Note: the ~2.5% ambient drift on `stock_default` (`462.6 → 474.2 ms`) and
`duckdb_default` (`304.0 → 311.6 ms`) between `#192` and `#195` explains the
`0.98–0.99×` (`+5.6` to `+7.9 ms`) on untouched single-table / no-join queries
`q1`, `q6`, and `q15`; relative to `stock_default`, all three held or improved
their ratios: `q1` `0.93× → 0.93×`, `q6` `0.96× → 0.94×`, `q15`
`0.98× → 0.96×`.)_

## 3. Dynamic Join Filter Pushdown & `bytes_scanned` Reductions

### Standard 22/22 TPC-H SF5 (`bytes_scanned`, Row Groups Claimed, and Dynamic Filters Pushed)

Morsel pushed dynamic join filters (`join_dynamic_filters_pushed > 0`) on
**`19 / 22` standard TPC-H queries** — every query with a Parquet-probed hash
join (`q1`/`q6` have no joins; `q15` joins a materialized `StageSource` against
`supplier`).

| Query | `dyn_filters` (`pruning`/`pushdown`) | RGs claimed (`no_pd → pd`) | `#192` `morsel_pushdown` bytes | Post-`#194` `morsel_pushdown` bytes | `bytes_scanned` delta vs. `#192` | `stock_default` bytes | Primary dynamic pushdown effect                                                                |
| ----- | -----------------------------------: | -------------------------: | -----------------------------: | ----------------------------------: | -------------------------------: | --------------------: | ---------------------------------------------------------------------------------------------- |
| `q3`  |                                `4/6` |                `310 → 310` |                    334,406,980 |                         334,406,980 |      `0` (`−30,128` vs. `no_pd`) |           334,406,980 | Bloom `RowFilter` prunes ~95% of `lineitem` payload decodes (`537 → 401 ms`)                   |
| `q5`  |                              `13/15` |                `313 → 313` |                    374,043,457 |                         374,043,457 |                              `0` |           374,043,457 | Reverse-order cascade `region → … → lineitem` (`−43%` CPU, `757 → 441 ms`)                     |
| `q7`  |                              `12/15` |                `313 → 313` |                    408,905,403 |                         408,905,403 |                              `0` |           408,905,403 | `nation(2) → supplier/customer/orders → lineitem` (`1,361 → 444 ms`, `3.06×`)                  |
| `q8`  |                              `15/19` |                `322 → 322` |                    531,613,209 |                         531,613,209 |                              `0` |           531,613,209 | `part(0.7%)` Bloom `RowFilter` skips 99%+ of `lineitem` probe (`876 → 554 ms`)                 |
| `q12` |                                `2/3` |                `304 → 304` |                    221,292,133 |                         221,292,133 |                              `0` |           221,292,133 | Filtered-scan build flip (`lineitem` 155k build) + `o_orderkey` Bloom (`857 → 232 ms`)         |
| `q18` |                                `6/9` |            **`550 → 547`** |                    256,827,407 |                     **256,124,246** |                   **`−703,161`** |           237,511,219 | Dynamic bounds prune 3 RGs + `301`-key Bloom slashes `orders`/`lineitem` (`2,813 → 693 ms`)    |
| `q19` |                                `2/3` |                `248 → 248` |                    369,999,158 |                     **369,774,803** |                   **`−224,355`** |           369,999,158 | `part` Bloom skips `lineitem` columns + beats `stock_default` `bytes_scanned` (`520 → 415 ms`) |
| `q21` |                               `7/10` |                `786 → 786` |                    606,479,991 |                         606,479,991 |                              `0` |           606,479,991 | `SAUDI ARABIA` cascade shrinks `build_0` (`4,418 → 1,544 ms`, `2.86×`)                         |

- **Why standard TPC-H `bytes_scanned` changes modestly while wall time drops
  up to `4.07×`:** As documented in `#170`/`#193`, standard TPC-H `lineitem`
  and `orders` are clustered by primary key (`orderkey`), whereas TPC-H
  predicates filter on dates, nations, and part attributes distributed
  uniformly across all `orderkey`s. Even on Stock DataFusion, `bytes_scanned`
  is identical to unpruned scans on `21 / 22` queries (only `q18` prunes
  `19.3 MB` via Stock's `301`-key `KEY_RANGE_BITMAP` page-level pruning; Morsel
  prunes `703 KB` / 3 row groups via min/max bounds and still beats
  `stock_default` by **`2.15×`** and `stock_pushdown` by **`1.21×`** on `q18`).
  The massive runtime gains on standard TPC-H come from **late-materialization
  L2-resident Bloom `RowFilter`s** (skipping non-matching Parquet payload
  decodes before Arrow materialization) and **reverse-order cascading dynamic
  filters** shrinking upstream hash-join builds.

### Key-Clustered SF5 Dynamic Range Join (`sf5_dyn_orderkey_range`)

To directly isolate dynamic row-group statistics pruning on key-clustered data
at SF5 on the same `rob-tandy-dduck2` host and runner (`1` warmup + `5`
counterbalanced warm-cache repeats, verified exact DuckDB 2.0.1 Arrow IPC
parity):

```sql
SELECT COUNT(*) AS cnt, SUM(l.l_extendedprice * (1 - l.l_discount)) AS rev
FROM orders o INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE o.o_orderkey + 0 >= 1000000 AND o.o_orderkey + 0 < 1500000
```

_(Using `o.o_orderkey + 0` prevents static equivalence propagation to
`l_orderkey`, so all `lineitem` row-group pruning comes strictly from runtime
dynamic join min/max pushdown.)_

| Arm                   | Median wall ms (min–max) | Median CPU ms | `bytes_scanned` | `bytes_scanned` vs. `no_pushdown` |                   Morsels claimed | Dynamic filters pushed |
| --------------------- | -----------------------: | ------------: | --------------: | --------------------------------: | --------------------------------: | ---------------------: |
| `morsel_no_pushdown`  |      325.9 (317.9–328.7) |       2,352.1 |     237,374,364 |                              0.0% |                               304 |                      2 |
| **`morsel_pruning`**  |     **74.1** (73.1–75.0) |     **314.5** |  **25,625,691** |                        **−89.2%** | **69** (`240 → 5` `lineitem` RGs) |                      2 |
| **`morsel_pushdown`** |     **75.7** (73.4–76.4) |     **334.6** |  **25,625,691** |                        **−89.2%** | **69** (`240 → 5` `lineitem` RGs) |                      3 |
| `stock_default`       |      117.9 (112.4–119.4) |         318.9 |      25,625,691 |                            −89.2% |                                 — |                      — |
| `stock_pushdown`      |      140.9 (134.5–147.0) |         361.5 |      25,625,691 |                            −89.2% |                                 — |                      — |
| `duckdb_default`      |         39.3 (38.1–43.8) |         128.4 |               — |                                 — |                                 — |                      — |

Dynamic min/max pushdown prunes **`235 / 240` `lineitem` row groups** before
probe I/O begins, slashing `bytes_scanned` by **`−89.2%`** (`−211.7 MB`), CPU
by **`−86.6%`**, and wall time by **`4.40×`**, beating `stock_default` by
**`1.59×`** and `stock_pushdown` by **`1.90×`**.

## 4. Native `MorselExec` Execution & `#138` Wait/Work Summary (Post-`#194`)

Every row below reports the faster Morsel arm (`Best Morsel`) from the final
SF5 run (`MORSEL_METRICS=1`). `Decode`, `Process`, and `Ready wait` are summed
across 8 workers (`Output wait` was **`0.0 ms`** on all 330 measured Morsel
trials). Post-`#194`, `Join fin` is the wall time of parallel 64-way radix
`HashJoinBuildSink::finalize` (down **`6.89×`** from `#192`'s `7,258.4 ms` to
**`1,053.4 ms`**!), and `Agg fin / other ms` (`max(0, exec - (decode +
process)/8 - join_fin - sort_topn_fin)`) includes parallel 64-way radix
`AggregateSink::finalize` wall time plus pipeline synchronization overhead.

| Query             | Best Morsel arm      | `mincore` pages (100%) | RGs claimed | Dyn filt | Stages / `jb` / `agg` / `gc` / `s` |      Exec ms |  Exec CPU ms | Decode ms | Process ms | Ready wait ms | Join fin ms | Sort/TopN fin ms | Agg fin / other ms |
| ----------------- | -------------------- | ---------------------: | ----------: | -------: | ---------------------------------- | -----------: | -----------: | --------: | ---------: | ------------: | ----------: | ---------------: | -----------------: |
| `q1`              | `morsel_no_pushdown` |                229,147 |         240 |        0 | `0 / 0 / 1 / 0 / Sort`             |        708.9 |      5,573.1 |   3,204.2 |    2,274.5 |          16.2 |         0.0 |              0.1 |               23.9 |
| `q2`              | `morsel_pushdown`    |                 44,595 |          78 |       21 | `1 / 8 / 1 / 0 / TopN`             |        182.2 |        941.3 |     577.9 |      127.3 |         122.0 |        22.2 |              0.4 |               71.4 |
| `q3`              | `morsel_pushdown`    |                297,360 |         310 |        6 | `0 / 2 / 1 / 0 / TopN`             |        385.9 |      2,564.2 |   1,946.2 |      162.8 |         139.0 |        29.1 |              0.2 |               92.9 |
| `q4`              | `morsel_pushdown`    |                286,442 |         304 |        3 | `0 / 1 / 1 / 0 / Sort`             |        212.1 |      1,381.6 |   1,120.6 |       77.3 |          68.7 |        10.0 |              0.0 |               52.3 |
| `q5`              | `morsel_pushdown`    |                298,107 |         313 |       15 | `0 / 5 / 1 / 0 / Sort`             |        424.4 |      2,872.9 |   2,360.0 |      152.6 |         167.7 |        12.3 |              0.0 |               98.1 |
| `q6`              | `morsel_pruning`     |                229,147 |         240 |        0 | `0 / 0 / 1 / 0 / —`                |        340.6 |      2,651.4 |   2,330.6 |      220.5 |          16.6 |         0.0 |              0.0 |               21.7 |
| `q7`              | `morsel_pushdown`    |                298,106 |         313 |       15 | `0 / 5 / 1 / 0 / Sort`             |        425.3 |      2,850.0 |   2,309.2 |       81.7 |         208.4 |        22.8 |              0.0 |              103.6 |
| `q8`              | `morsel_pushdown`    |                303,951 |         322 |       19 | `0 / 7 / 1 / 0 / Sort`             |        534.4 |      3,624.6 |   3,045.2 |       52.0 |         235.8 |        21.5 |              0.0 |              125.8 |
| `q9`              | `morsel_no_pushdown` |                331,036 |         346 |       13 | `0 / 5 / 1 / 0 / Sort`             |      1,361.8 |      9,444.1 |   4,172.1 |    4,228.1 |          82.1 |       198.7 |              0.0 |              113.1 |
| `q10`             | `morsel_pushdown`    |                297,361 |         311 |        9 | `0 / 3 / 1 / 0 / TopN`             |        592.6 |      3,797.5 |   2,065.7 |      759.6 |         174.6 |        97.7 |              0.3 |              141.5 |
| `q11`             | `morsel_pushdown`    |                 38,750 |          68 |        8 | `0 / 4 / 2 / 0 / Sort`             |        134.9 |        817.1 |     749.8 |       30.3 |          68.5 |         0.1 |              0.5 |               36.8 |
| `q12`             | `morsel_pruning`     |                286,442 |         304 |        2 | `0 / 1 / 1 / 0 / Sort`             |        215.3 |      1,578.7 |     826.1 |      617.6 |          22.8 |         2.9 |              0.0 |               32.0 |
| `q13`             | `morsel_no_pushdown` |                 68,213 |          70 |        2 | `1 / 1 / 1 / 1 / Sort`             |        569.9 |      3,703.3 |   1,161.8 |    2,194.1 |          34.0 |         9.8 |              0.0 |              140.6 |
| `q14`             | `morsel_pushdown`    |                234,991 |         248 |        3 | `0 / 1 / 1 / 0 / —`                |        324.2 |      2,157.7 |   1,506.5 |      172.0 |         118.9 |        35.4 |              0.0 |               79.0 |
| `q15`             | `morsel_no_pushdown` |                229,892 |         481 |        0 | `2 / 1 / 3 / 0 / Sort`             |        697.3 |      5,241.6 |   4,419.8 |      538.3 |          42.9 |         2.5 |              0.0 |               75.0 |
| `q16`             | `morsel_pushdown`    |                 44,593 |          41 |        3 | `1 / 2 / 2 / 0 / Sort`             |        162.7 |        746.0 |     255.4 |      259.1 |         123.5 |         6.1 |              5.5 |               86.7 |
| `q17`             | `morsel_pushdown`    |                234,991 |         488 |        6 | `1 / 2 / 2 / 0 / —`                |      1,223.9 |      7,077.1 |   3,106.7 |    2,156.9 |         105.6 |        27.2 |              0.0 |              538.7 |
| `q18`             | `morsel_pushdown`    |                297,360 |         547 |        9 | `1 / 3 / 2 / 0 / TopN`             |        677.7 |      4,619.6 |   2,117.3 |    1,204.2 |         122.1 |        32.8 |              0.1 |              229.6 |
| `q19`             | `morsel_pushdown`    |                234,991 |         248 |        3 | `0 / 1 / 1 / 0 / —`                |        397.6 |      2,728.9 |   2,464.7 |       17.2 |         186.9 |         1.1 |              0.0 |               86.3 |
| `q20`             | `morsel_no_pushdown` |                273,741 |         282 |        3 | `1 / 4 / 1 / 0 / Sort`             |        717.3 |      3,715.9 |   1,967.5 |    1,088.9 |          69.4 |         1.5 |              0.1 |              333.7 |
| `q21`             | `morsel_pushdown`    |                287,188 |         786 |       10 | `0 / 5 / 1 / 0 / TopN`             |      1,526.4 |      7,916.0 |   3,943.8 |      941.2 |         192.2 |       515.7 |              0.0 |              400.1 |
| `q22`             | `morsel_pushdown`    |                 68,213 |          76 |        3 | `0 / 1 / 2 / 0 / Sort`             |        117.7 |        716.3 |     538.2 |      110.8 |         149.7 |         3.9 |              0.0 |               32.6 |
| **Total (22/22)** | —                    |                      — |           — |        — | —                                  | **11,933.2** | **76,718.8** |         — |          — |             — | **1,053.4** |          **7.4** |        **2,915.3** |

### Highlights from `#138` Pipeline Breakdowns (Before vs. After)

- **`q18` (`2,812.5 → 693.3 ms`, `4.06×`; beats `stock_default` `1,493.2 ms` by
  `2.15×` and `stock_pushdown` `840.7 ms` by `1.21×`):** `stage_0`'s 7.5M-group
  `l_orderkey` aggregate moved from serial merge (`~1,116 ms`) to parallel
  64-way radix finalize; `build_0` (`orders ⋈ customer`) dropped `join_finalize`
  from `726.9 → 1.0 ms` via `stage_0`'s `301`-key dynamic filter; and `probe`
  `lineitem` CPU plunged `4.67×` (`2,702.0 → 578.1 ms`) while pruning 3 RGs
  (`−703 KB` `bytes_scanned`).
- **`q12` (`856.8 → 231.6 ms`, `3.70×`; `1.03×` of DuckDB `225.2 ms`):**
  Filtered build-side discounting moved `lineitem` (`~155k` rows) onto build
  instead of 7.5M unfiltered `orders`, cutting `join_finalize` **`216×`**
  (`627.5 → 2.9 ms`).
- **`q7` (`1,361.1 → 444.4 ms`, `3.06×`; beats `stock_default` `909.0 ms` by
  `2.05×`, `stock_pushdown` `513.1 ms` by `1.15×`; `1.06×` of DuckDB
  `418.8 ms`):** Cascading `nation(2)` dynamic filters + parallel radix build
  cut `join_finalize` **`28.3×`** (`644.7 → 22.8 ms`) and `probe` process CPU
  **`30.1×`** (`2,052.8 → 68.2 ms`).
- **`q21` (`4,418.0 → 1,544.1 ms`, `2.86×`, `−2,873.9 ms`):** Parallel 64-way
  radix `HashJoin` build finalize + `SAUDI ARABIA` dynamic filter cascade cut
  `join_finalize` **`6.58×`** (`3,393.2 → 515.7 ms`) and `probe` process CPU
  **`11.0×`** (`2,560.7 → 233.4 ms`).
- **`q5` (`756.7 → 441.3 ms`, `1.71×`) & `q8` (`876.0 → 554.4 ms`, `1.58×`):**
  Both now beat **both** `stock_default` (`0.62×` / `0.84×`) and
  `stock_pushdown` (`0.91×` / `0.85×`) as cascading dynamic filters shrink
  `join_finalize` by `11.3–11.4×` (`q5`: `139.9 → 12.3 ms`; `q8`:
  `243.3 → 21.5 ms`).
- **Remaining headroom vs. `stock_pushdown` and DuckDB 2.0.1:**
  1. **Cross-stage dynamic filter pushdown (`q17`, `q20`):** Accounts for
     **100% of the remaining gap between Best Morsel (`443.2 ms`) and
     `stock_pushdown` (`434.2 ms`)** — pushing `part`'s `0.1%` (`q17`) and
     `1.1%` (`q20`) `p_partkey` filters across the `StageSource` boundary into
     `stage_0`'s `lineitem` scan would take `q17` (`1,238 → ~450 ms`) and `q20`
     (`732 → ~320 ms`) straight to DuckDB/Stock-pushdown speed.
  2. **Stage CSE on `q15` (`710.3 ms` vs. DuckDB `270.7 ms`):** Reusing
     `revenue0`'s `stage_0` across the `MAX` scalar subquery and `supplier` join
     avoids scanning `lineitem` twice (`605 MB → ~302 MB`).
  3. **Large `Rows`-key radix hash join probe/build (`q9`, `q21`) and
     `Decimal128` Parquet decode (`q1`, `q6`):** Narrowing the remaining `1.42×`
     gap to DuckDB 2.0.1.

## 5. Tight-Memory SF5 Out-of-Core Spill Completion (`GreedyMemoryPool`, `RLIMIT_NOFILE=1024`)

Every workload below ran on SF5 (`taskset -c 0-7`, 8 threads,
`morsel_mode=required`, `pushdown` profile) directly via `tpch_runner.rs` with
`MORSEL_MEMORY_LIMIT_MB` enforcing a strict `GreedyMemoryPool` budget,
`DiskManagerMode::OsTmpDirectory`, and `ulimit -n 1024` (`RLIMIT_NOFILE=1024`).
Prior to `#194`, all 9 workloads aborted with `ResourcesExhausted` under these
tight limits; post-`#194`, **all `9 / 9` complete cleanly with bounded open
spill files and verified exact result parity**:

| Workload                      | Spill operators exercised          | `GreedyMemoryPool` | Pre-`#194` status    | Post-`#194` status |  Wall ms | Spilled MiB | Spill files (`RLIMIT_NOFILE=1024`) | Spilled rows | Parity verification                                                |
| ----------------------------- | ---------------------------------- | -----------------: | -------------------- | ------------------ | -------: | ----------: | ---------------------------------: | -----------: | ------------------------------------------------------------------ |
| **TPC-H `q3`**                | Radix `HashJoin` + `Agg`           |             32 MiB | `ResourcesExhausted` | **Completed**      |  2,551.4 |   503.4 MiB |                                112 |   12,856,195 | Exact Arrow IPC vs. DuckDB 2.0.1 (10 rows)                         |
| **TPC-H `q13`**               | Radix `LEFT JOIN` + 750k-group Agg |             64 MiB | `ResourcesExhausted` | **Completed**      |  2,396.0 |   239.3 MiB |                                176 |   13,979,676 | Exact Arrow IPC vs. DuckDB 2.0.1 (42 rows)                         |
| **TPC-H `q10`**               | 4-way Radix `HashJoin` + `Agg`     |            128 MiB | `ResourcesExhausted` | **Completed**      |  3,396.4 |   470.0 MiB |                                224 |    6,804,811 | Exact Arrow IPC vs. DuckDB 2.0.1 (20 rows)                         |
| **TPC-H `q17`**               | 1.0M-group Radix `Agg` + Join      |            128 MiB | `ResourcesExhausted` | **Completed**      |  3,926.3 |   957.9 MiB |                                176 |   30,658,989 | Exact Arrow IPC vs. DuckDB 2.0.1 (1 row)                           |
| **TPC-H `q18`**               | 7.5M-group Radix `Agg` + 3-way JN  |            128 MiB | `ResourcesExhausted` | **Completed**      |  3,683.7 |   767.9 MiB |                                288 |   30,631,588 | Exact Arrow IPC vs. DuckDB 2.0.1 (100 rows)                        |
| **TPC-H `q20`**               | 2.7M-group Radix `Agg` + Semi-JN   |            128 MiB | `ResourcesExhausted` | **Completed**      |  1,212.5 |   142.3 MiB |                                 64 |    4,541,197 | Exact Arrow IPC vs. DuckDB 2.0.1 (901 rows)                        |
| **SF5 `orders` `ORDER BY`**\* | External `Sort` spill-merge        |            128 MiB | `ResourcesExhausted` | **Completed**      |    748.5 |   172.8 MiB |                                 74 |    1,802,376 | Exact ordered SHA-256 vs. Stock (`1,139,873` rows, `897470ceba5a`) |
| **TPC-H `q9`**                | 6-way Radix `HashJoin` build/probe |            256 MiB | `ResourcesExhausted` | **Completed**      | 21,742.6 |   671.5 MiB |                                224 |   13,947,112 | Exact Arrow IPC vs. DuckDB 2.0.1 (175 rows)                        |
| **TPC-H `q21`**               | 5-way Semi/Anti Radix `HashJoin`   |            256 MiB | `ResourcesExhausted` | **Completed**      | 23,749.0 | 1,787.2 MiB |                                458 |   60,546,842 | Exact Arrow IPC vs. DuckDB 2.0.1 (100 rows)                        |

\*`SELECT o_orderkey, o_custkey, o_totalprice, o_orderdate, o_comment FROM orders WHERE o_orderdate < DATE '1993-01-01' ORDER BY o_totalprice DESC, o_orderkey ASC` (`1,139,873` wide rows with `Utf8View` comments).

## Summary

Landing dynamic join filter pushdown (`#193`) and parallel radix-partitioned
hash tables with out-of-core spilling (`#194`) transforms the 22/22 TPC-H SF5
morsel engine from `1.47×` slower than `stock_default` (`#192`: `679.7` vs.
`462.6 ms`) to **faster than `stock_default` both in single-config
`morsel_pushdown` (`461.0` vs. `474.2 ms`, `0.97×`) and in best-per-query mode
(`443.2` vs. `474.2 ms`, `0.93×`)**, cuts the 22-query sum by **`10.2–10.5 s`
(`1.82–1.84×`)**, reduces total `join_finalize` wall time by **`6.89×`**
(`7,258.4 → 1,053.4 ms`), narrows the 22/22 geomean gap vs. DuckDB 2.0.1 from
**`2.24× → 1.42×`**, achieves **`−89.2%` `bytes_scanned`** on key-clustered
dynamic joins, and completes all **9 / 9 tight-memory SF5 spill workloads**
(`32–256 MiB`, up to `1.79 GiB` / `60.5M` rows spilled) under
`RLIMIT_NOFILE=1024` with 100% verified exact parity.
