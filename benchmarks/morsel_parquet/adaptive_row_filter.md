<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Runtime-Adaptive Per-RG `RowFilter` Switching & `AdaptiveFilter` Conjunct Reordering (#207)

Before: `1fe19bf025407738f5ab8c13582096c86877226a` (`morsel-execution` after #206).
After: `agent/morsel-issue-207` (runtime-adaptive per-RG `RowFilter` vs.
`BulkDecode + BatchView` switching + DuckDB-style `AdaptiveFilter` conjunct
reordering).

## Executive Summary

- **Single-config `morsel_pushdown` recovers the narrow/dense scan regressions
  while keeping or improving the dynamic-join `RowFilter` wins across all
  22/22 SF=5 TPC-H queries:**
  - **22/22 GeoMean:** `403.9 → 386.4 ms` (**`1.05×`** faster vs. pre-#207
    `morsel_pushdown`, `−17.5 ms`; **`1.45×`** faster vs. `morsel_no_pushdown`
    at `560.9 ms`; **`1.21×`** faster vs. `stock_default` at `466.4 ms`;
    **`1.10×`** faster vs. `stock_pushdown` at `424.2 ms`; `1.21×` of DuckDB
    2.0.1 at `320.2 ms`).
  - **22/22 Sum:** `10,840.1 → 10,404.5 ms` (`−435.6 ms` vs. pre-#207
    `morsel_pushdown`, `−5,440.7 ms` vs. `morsel_no_pushdown`, `−3,121.3 ms`
    vs. `stock_default`, `−1,511.3 ms` vs. `stock_pushdown`).
  - **Oracle gap reduced by 88%:** The geometric-mean gap between single-config
    `morsel_pushdown` (`386.4 ms`) and per-query oracle `Best Morsel`
    (`384.2 ms`) drops from **`17.8 ms` in `#195` to `2.2 ms`**.
- **Headline per-query recoveries and wins (5-round paired SF=5 medians):**
  - **`q6`:** `461.0 → 345.8 ms` (**`1.33×`**, `−115.2 ms`; converges to
    `BulkDecode` on `236 / 240` RGs, within `1.4%` of `morsel_no_pushdown`
    `341.0 ms` and faster than `stock_default` `368.9 ms` / `stock_pushdown`
    `512.0 ms`).
  - **`q15`:** `942.7 → 725.4 ms` (**`1.30×`**, `−217.3 ms`; converges to
    `BulkDecode` on `451 / 480` RGs, within `2.5%` of `morsel_no_pushdown`
    `707.6 ms` and faster than `stock_default` `727.3 ms` / `stock_pushdown`
    `921.0 ms`).
  - **`q12`:** `254.7 → 187.8 ms` (**`1.36×`**, `−66.9 ms`; **beats every arm
    including `morsel_no_pushdown` `216.9 ms`, `stock_default` `201.9 ms`, and
    DuckDB 2.0.1 `222.9 ms`** because `lineitem` switches to `BulkDecode` while
    `orders` keeps Bloom `RowFilter`: `64` RF / `240` Bulk RGs).
  - **`q9`:** `1,446.5 → 1,424.7 ms` (**`1.02×`**, `−21.8 ms`; within `0.9%`
    of `morsel_no_pushdown` `1,411.5 ms` and `161.1 ms` faster than
    `stock_pushdown` `1,585.8 ms`).
  - **Dynamic-join `RowFilter` wins retained or improved:** `q4`
    (`218.0 → 207.3 ms`, `1.05×`), `q5` (`440.2 → 439.7 ms`), `q8`
    (`546.7 → 538.5 ms`, `1.02×`), `q11` (`132.6 → 119.6 ms`, `1.11×`), `q17`
    (`482.9 → 486.4 ms` vs. `1,374.8 ms` no-pushdown), `q18`
    (`686.1 → 663.8 ms`, `1.03×`, `297` RF / `4` Bulk RGs vs. `1,945.9 ms`
    no-pushdown), `q19` (`413.2 → 392.8 ms`, `1.05×`), `q20`
    (`315.9 → 301.3 ms`, `1.05×`, beats DuckDB `308.9 ms`), `q21`
    (`957.5 → 926.5 ms`, `1.03×`), `q3` (`391.3 → 403.6 ms` vs. `566.6 ms`
    no-pushdown), `q14` (`328.5 → 339.0 ms`, beats DuckDB `343.9 ms`).
  - **Head-to-head vs. DuckDB 2.0.1:** Single-config `morsel_pushdown` beats
    DuckDB on `5 / 22` queries (`q12`, `q14`, `q16`, `q20`, `q22`).

## Method & Environment

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (8 pinned via `taskset -c 0-7`), 61 GiB RAM.
- **Idle verification:** `vmstat -y 1 3` immediately before the run showed
  `97–99%` CPU idle; immediately after showed `98–99%` CPU idle.
- **Dataset & cache residency:** SF=5 TPC-H Parquet (`35` files, `341,955 /
  341,955` pages verified `100%` resident via `mincore`), `Q11` fraction scaled
  to `0.0001 / 5` (`4,606` output rows).
- **Build:** `cargo build --locked --profile release-nonlto -p datafusion
  --no-default-features --features parquet,sql,datetime_expressions,unicode_expressions
  --example morsel_tpch_runner`, `--threads 8`, batch size `8,192`.
- **Verification:** `22 / 22` SF=5 queries passed lossless Arrow IPC parity
  against `stock_pushdown` before timing.
- **Sampling:** 1 warmup round + 5 measured interleaved rounds (alternating arm
  order each round); table reports median elapsed ms (`min–max` for
  `after_pushdown`).

## Full 22/22 SF=5 TPC-H Paired Results

| Query       | `before_pd` (ms) | `after_pd` ms (min–max)     | `after_nopd` (ms) | `stock_def` (ms) | `stock_pd` (ms) | `duckdb` (ms) | `after` vs `before` | `after` vs `nopd` | Adaptive RGs (RF / Bulk) |
| ----------- | ---------------: | --------------------------: | ----------------: | ---------------: | --------------: | ------------: | ------------------: | ----------------: | -----------------------: |
| `q1`        |            732.3 |         746.4 (730.2–751.5) |             739.0 |            797.0 |           764.8 |         485.4 |               0.98× |             0.99× |                 154 / 86 |
| `q2`        |            159.5 |         162.5 (158.8–165.8) |             248.8 |            173.2 |           160.6 |         126.8 |               0.98× |         **1.53×** |                   74 / 4 |
| `q3`        |            391.3 |         403.6 (389.3–412.2) |             566.6 |            524.9 |           388.1 |         358.1 |               0.97× |         **1.40×** |                  306 / 4 |
| `q4`        |            218.0 |     **207.3** (206.0–215.6) |             313.2 |            189.6 |           284.8 |         197.1 |           **1.05×** |         **1.51×** |                 270 / 34 |
| `q5`        |            440.2 |     **439.7** (430.5–443.4) |             793.1 |            731.7 |           478.5 |         386.3 |           **1.00×** |         **1.80×** |                  308 / 5 |
| `q6`        |            461.0 |     **345.8** (334.6–347.6) |             341.0 |            368.9 |           512.0 |         209.1 |           **1.33×** |             0.99× |                  4 / 236 |
| `q7`        |            424.7 |         464.4 (460.8–479.2) |           1,080.1 |            920.3 |           508.6 |         424.5 |               0.91× |         **2.33×** |                 261 / 52 |
| `q8`        |            546.7 |     **538.5** (530.4–548.4) |             727.8 |            664.9 |           644.2 |         466.0 |           **1.02×** |         **1.35×** |                 280 / 41 |
| `q9`        |          1,446.5 |   **1,424.7** (1410.3–1457.2) |           1,411.5 |          1,069.4 |         1,585.8 |         821.2 |           **1.02×** |             0.99× |                  245 / 4 |
| `q10`       |            614.3 |         629.9 (618.0–637.6) |             777.4 |            636.4 |           533.7 |         462.7 |               0.98× |         **1.23×** |                  305 / 5 |
| `q11`       |            132.6 |     **119.6** (114.1–127.5) |             203.9 |            107.8 |           112.3 |         111.4 |           **1.11×** |         **1.71×** |                  28 / 40 |
| `q12`       |            254.7 |     **187.8** (181.3–189.5) |             216.9 |            201.9 |           291.7 |         222.9 |           **1.36×** |         **1.15×** |                 64 / 240 |
| `q13`       |            605.3 |         614.6 (606.1–628.4) |             574.8 |            507.3 |           575.0 |         559.9 |               0.98× |             0.94× |                   62 / 2 |
| `q14`       |            328.5 |         339.0 (331.9–341.1) |             427.5 |            404.1 |           291.4 |         343.9 |               0.97× |         **1.26×** |                  237 / 3 |
| `q15`       |            942.7 |     **725.4** (717.4–748.8) |             707.6 |            727.3 |           921.0 |         270.9 |           **1.30×** |             0.98× |                 29 / 451 |
| `q16`       |            168.1 |     **165.7** (162.4–168.8) |             170.3 |            125.3 |           135.1 |         172.0 |           **1.01×** |         **1.03×** |                   39 / 2 |
| `q17`       |            482.9 |         486.4 (483.6–498.7) |           1,374.8 |          1,296.7 |           460.3 |         410.5 |               0.99× |         **2.83×** |                  480 / 8 |
| `q18`       |            686.1 |     **663.8** (634.6–670.2) |           1,945.9 |          1,518.7 |           850.3 |         477.0 |           **1.03×** |         **2.93×** |                  297 / 4 |
| `q19`       |            413.2 |     **392.8** (390.9–401.1) |             493.4 |            559.4 |           410.1 |         336.3 |           **1.05×** |         **1.26×** |                  245 / 3 |
| `q20`       |            315.9 |     **301.3** (296.7–304.0) |             723.8 |            461.3 |           306.2 |         308.9 |           **1.05×** |         **2.40×** |                 264 / 18 |
| `q21`       |            957.5 |     **926.5** (895.6–942.1) |           1,836.7 |          1,426.5 |         1,569.6 |         738.0 |           **1.03×** |         **1.98×** |                 692 / 94 |
| `q22`       |            118.4 |         119.0 (118.3–126.6) |             171.0 |            113.2 |           131.6 |         149.2 |               1.00× |         **1.44×** |                   74 / 2 |
| **GeoMean** |        **403.9** |                   **386.4** |         **560.9** |        **466.4** |       **424.2** |     **320.2** |           **1.05×** |         **1.45×** |                        — |
| **Sum**     |     **10,840.1** |                **10,404.5** |      **15,845.2** |     **13,525.8** |    **11,915.8** |   **8,038.1** |           **1.04×** |         **1.52×** |                        — |

## Remaining Shortfall Note

- **`q7` (`424.7 → 464.4 ms`, `+39.7 ms` / `0.91×` vs. `before_pd`):**
  `q7` still achieves a **`2.33×` speedup over `morsel_no_pushdown`** (`1,080.1 ms`),
  beats `stock_default` (`920.3 ms`) by **`1.98×`**, and beats `stock_pushdown`
  (`508.6 ms`) by **`44.2 ms`**. The `+39.7 ms` delta vs. static `before_pd`
  comes from `RowFilterOrderBandit` running 2-RG trial swaps on `lineitem`'s
  3-predicate `RowFilter` (`l_shipdate`, `l_suppkey`, `l_orderkey`) where
  `l_suppkey` comes from an unfiltered `supplier` build side (`100%` pass rate)
  and `orders` choosing `BulkDecode` (`50 / 60` RGs) with `BatchView` Bloom
  filtering.
