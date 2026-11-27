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

- **Overall 22/22 SF=5 TPC-H impact (5-round paired medians):**
  - **22/22 GeoMean:** `377.4 → 361.8 ms` (**`1.04×`** faster vs. pre-#207
    `morsel_pushdown`, `−15.6 ms`; **`1.40×`** faster vs. `morsel_no_pushdown`
    at `505.5 ms`; **`1.19×`** faster vs. `stock_default` at `429.7 ms`;
    **`1.10×`** faster vs. `stock_pushdown` at `399.4 ms`; `1.19×` of DuckDB
    2.0.1 at `303.5 ms`).
  - **22/22 Sum:** `10,110.7 → 9,710.1 ms` (`−400.6 ms` vs. pre-#207
    `morsel_pushdown`, `−4,580.9 ms` vs. `morsel_no_pushdown`, `−2,646.1 ms`
    vs. `stock_default`, `−1,416.8 ms` vs. `stock_pushdown`).
  - **Oracle gap:** The geometric-mean gap between single-config
    `morsel_pushdown` (`361.8 ms`) and per-query oracle `Best Morsel`
    (`357.8 ms`) drops from `17.8 ms` in `#195` to **`3.9 ms`**.
- **Target narrow/dense scan recoveries achieved:**
  - **`q6`:** `449.8 → 322.5 ms` (**`1.39×`**, `−127.3 ms`; `4` RF / `236` Bulk
    RGs, matching `morsel_no_pushdown` `320.9 ms` and beating `stock_default`
    `341.7 ms` / `stock_pushdown` `484.9 ms`).
  - **`q15`:** `898.1 → 674.8 ms` (**`1.33×`**, `−223.3 ms`; `30` RF / `450`
    Bulk RGs, within `3.4%` of `morsel_no_pushdown` `652.8 ms` and beating
    `stock_default` `677.3 ms` / `stock_pushdown` `863.5 ms`).
  - **`q12`:** `239.7 → 181.8 ms` (**`1.32×`**, `−57.9 ms`; `79` RF / `225` Bulk
    RGs — `lineitem` switches to `BulkDecode` while `orders` keeps Bloom
    `RowFilter`, beating `morsel_no_pushdown` `197.8 ms`, `stock_default`
    `193.1 ms`, and DuckDB `212.1 ms`).
- **Dynamic-join `RowFilter` wins preserved or improved:**
  - `q3` (`361.5 → 362.8 ms`, `1.00×`), `q4` (`205.4 → 198.9 ms`, `1.03×`),
    `q5` (`408.2 → 407.5 ms`, `1.00×`), `q7` (`401.6 → 399.2 ms`, `1.01×`,
    `2.32×` vs. `no_pushdown`, matching DuckDB `399.5 ms`), `q8`
    (`523.5 → 522.1 ms`, `1.00×`, `1.30×` vs. `no_pushdown`), `q11`
    (`125.7 → 116.2 ms`, `1.08×`), `q14` (`312.4 → 317.8 ms`, `0.98×`, ahead of
    DuckDB `326.3 ms`), `q17` (`458.9 → 462.7 ms`, `0.99×`, `2.85×` vs.
    `no_pushdown`), `q18` (`636.1 → 620.8 ms`, `1.02×`, `297` RF / `4` Bulk
    RGs, `2.80×` vs. `no_pushdown`), `q19` (`386.7 → 366.5 ms`, `1.06×`), `q20`
    (`293.4 → 295.1 ms`, `0.99×`, `2.22×` vs. `no_pushdown`), `q21`
    (`876.4 → 845.5 ms`, `1.04×`).
- **Measured shortfalls (target missed or slight regression vs. `before_pd`):**
  - **`q9` target missed:** `1,327.0 → 1,354.1 ms` (`0.98×` vs. `before_pd`,
    and `7.5%` slower than `morsel_no_pushdown` at `1,259.5 ms`). On `q9`,
    `lineitem` committed `RowFilter` (`245` RF / `4` Bulk RGs) because per-RG
    probe cost differences are small relative to the 4-way hash join and
    aggregate CPU (`q9` remains `111.0 ms` faster than `stock_pushdown` at
    `1,465.1 ms`).
  - **`q10`:** `568.1 → 584.4 ms` (`0.97×`, `+16.3 ms` vs. `before_pd`, driven
    by the 2-RG initial `BulkDecode` probe on `orders` and `lineitem`; still
    `1.25×` faster than `morsel_no_pushdown` at `730.5 ms` and `35.4 ms` ahead
    of `stock_default`).
  - **Small single-table / dimension-heavy deltas (`≤ 5%`):** `q1`
    (`683.8 → 703.6 ms`, `0.97×`), `q13` (`543.5 → 552.0 ms`, `0.98×`), `q14`
    (`312.4 → 317.8 ms`, `0.98×`), `q16` (`154.2 → 162.5 ms`, `0.95×`).

## Method & Environment

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (8 pinned via `taskset -c 0-7`), 61 GiB RAM.
- **Idle verification:** `vmstat -y 1 3` immediately before the run showed
  `97–99%` CPU idle; immediately after showed `95–97%` CPU idle.
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

| Query       | `before_pd` (ms) | `after_pd` ms (min–max)       | `after_nopd` (ms) | `stock_def` (ms) | `stock_pd` (ms) | `duckdb` (ms) | `after` vs `before` | `after` vs `nopd` | Adaptive RGs (RF / Bulk) |
| ----------- | ---------------: | ----------------------------: | ----------------: | ---------------: | --------------: | ------------: | ------------------: | ----------------: | -----------------------: |
| `q1`        |            683.8 |           703.6 (694.5–708.2) |             692.5 |            742.2 |           725.3 |         460.5 |               0.97× |             0.98× |                 208 / 32 |
| `q2`        |            148.4 |           150.0 (148.4–158.5) |             212.8 |            158.7 |           150.4 |         121.5 |               0.99× |         **1.42×** |                   74 / 4 |
| `q3`        |            361.5 |           362.8 (356.7–366.7) |             509.2 |            496.3 |           365.1 |         336.8 |           **1.00×** |         **1.40×** |                 260 / 50 |
| `q4`        |            205.4 |       **198.9** (197.3–208.0) |             277.9 |            177.5 |           269.6 |         185.7 |           **1.03×** |         **1.40×** |                 240 / 64 |
| `q5`        |            408.2 |       **407.5** (394.8–414.4) |             707.6 |            670.2 |           455.1 |         366.2 |           **1.00×** |         **1.74×** |                 273 / 40 |
| `q6`        |            449.8 |       **322.5** (318.5–327.3) |             320.9 |            341.7 |           484.9 |         202.4 |           **1.39×** |             1.00× |                  4 / 236 |
| `q7`        |            401.6 |       **399.2** (394.7–430.3) |             925.6 |            822.0 |           472.8 |         399.5 |           **1.01×** |         **2.32×** |                 262 / 51 |
| `q8`        |            523.5 |       **522.1** (510.2–532.6) |             677.0 |            623.6 |           609.3 |         441.6 |           **1.00×** |         **1.30×** |                 270 / 51 |
| `q9`        |          1,327.0 |       1,354.1 (1332.1–1375.5) |           1,259.5 |            978.3 |         1,465.1 |         763.6 |               0.98× |             0.93× |                  245 / 4 |
| `q10`       |            568.1 |           584.4 (567.4–597.1) |             730.5 |            619.8 |           498.4 |         443.0 |               0.97× |         **1.25×** |                  306 / 4 |
| `q11`       |            125.7 |       **116.2** (113.3–121.8) |             169.1 |             94.4 |           113.4 |         105.7 |           **1.08×** |         **1.45×** |                  33 / 35 |
| `q12`       |            239.7 |       **181.8** (176.2–185.5) |             197.8 |            193.1 |           277.1 |         212.1 |           **1.32×** |         **1.09×** |                 79 / 225 |
| `q13`       |            543.5 |           552.0 (534.8–594.5) |             525.3 |            455.3 |           520.9 |         513.0 |               0.98× |             0.95× |                   61 / 3 |
| `q14`       |            312.4 |           317.8 (313.2–321.2) |             394.5 |            379.7 |           274.3 |         326.3 |               0.98× |         **1.24×** |                  237 / 3 |
| `q15`       |            898.1 |       **674.8** (668.4–753.9) |             652.8 |            677.3 |           863.5 |         259.3 |           **1.33×** |             0.97× |                 30 / 450 |
| `q16`       |            154.2 |           162.5 (156.5–165.6) |             152.4 |            115.5 |           128.2 |         164.0 |               0.95× |             0.94× |                   39 / 2 |
| `q17`       |            458.9 |           462.7 (460.6–466.6) |           1,319.5 |          1,133.0 |           432.2 |         396.6 |               0.99× |         **2.85×** |                  480 / 8 |
| `q18`       |            636.1 |       **620.8** (610.8–639.2) |           1,735.5 |          1,334.7 |           794.8 |         453.5 |           **1.02×** |         **2.80×** |                  297 / 4 |
| `q19`       |            386.7 |       **366.5** (360.6–378.5) |             465.4 |            530.5 |           394.4 |         319.3 |           **1.06×** |         **1.27×** |                  245 / 3 |
| `q20`       |            293.4 |           295.1 (289.7–302.7) |             653.8 |            427.1 |           289.1 |         293.6 |               0.99× |         **2.22×** |                  276 / 6 |
| `q21`       |            876.4 |       **845.5** (840.7–861.4) |           1,563.5 |          1,278.4 |         1,419.6 |         702.0 |           **1.04×** |         **1.85×** |                535 / 251 |
| `q22`       |            108.4 |           109.5 (109.0–113.3) |             147.8 |            106.8 |           123.2 |         138.6 |               0.99× |         **1.35×** |                   74 / 2 |
| **GeoMean** |        **377.4** |                     **361.8** |         **505.5** |        **429.7** |       **399.4** |     **303.5** |           **1.04×** |         **1.40×** |                        — |
| **Sum**     |     **10,110.7** |                   **9,710.1** |      **14,291.0** |     **12,356.2** |    **11,126.9** |   **7,604.8** |           **1.04×** |         **1.47×** |                        — |

## Notes

- Sorting initial `RowFilter` candidates by `(required_bytes, expr_cost, idx)`
  keeps narrow columns (e.g. `Date32` `l_shipdate`, `l_partkey`) ahead of wider
  columns while `RowFilterOrderBandit` adapts dynamically across row groups,
  eliminating the earlier `q7` and `q8` regressions (`q7`: `401.6 → 399.2 ms`,
  `q8`: `523.5 → 522.1 ms`).
- On `q9`, the target (`~1,260 ms` matching `morsel_no_pushdown`) was not met
  (`1,354.1 ms` vs. `1,259.5 ms`). Because `lineitem` feeds a 4-way hash join
  and grouped aggregate, probe-window noise exceeds the ~5% per-RG difference
  between `RowFilter` and `BulkDecode`, so `lineitem` committed `RowFilter`.
