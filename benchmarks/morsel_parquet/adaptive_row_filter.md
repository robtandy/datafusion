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
  - **22/22 GeoMean:** `376.4 → 358.5 ms` (**`1.05×`** faster vs. pre-#207
    `morsel_pushdown`, `−17.9 ms`; **`1.40×`** faster vs. `morsel_no_pushdown`
    at `501.3 ms`; **`1.20×`** faster vs. `stock_default` at `429.3 ms`;
    **`1.10×`** faster vs. `stock_pushdown` at `395.4 ms`; `1.19×` of DuckDB
    2.0.1 at `301.2 ms`).
  - **22/22 Sum:** `10,100.7 → 9,621.5 ms` (`−479.2 ms` vs. pre-#207
    `morsel_pushdown`, `−4,498.7 ms` vs. `morsel_no_pushdown`, `−2,681.2 ms`
    vs. `stock_default`, `−1,449.9 ms` vs. `stock_pushdown`).
  - **Oracle gap reduced by 84%:** The geometric-mean gap between single-config
    `morsel_pushdown` (`358.5 ms`) and per-query oracle `Best Morsel`
    (`355.5 ms`) drops from **`17.8 ms` in `#195` to `2.9 ms`**.
- **Headline per-query recoveries and wins (5-round paired SF=5 medians):**
  - **`q6`:** `449.6 → 321.0 ms` (**`1.40×`**, `−128.6 ms`; converges to
    `BulkDecode` on `236 / 240` RGs, matching `morsel_no_pushdown` `320.8 ms`
    and beating `stock_default` `337.8 ms` / `stock_pushdown` `485.1 ms`).
  - **`q15`:** `902.6 → 672.1 ms` (**`1.34×`**, `−230.5 ms`; converges to
    `BulkDecode` on `450 / 480` RGs, within `2.9%` of `morsel_no_pushdown`
    `653.2 ms` and beating `stock_pushdown` `885.9 ms`).
  - **`q12`:** `237.3 → 174.4 ms` (**`1.36×`**, `−62.9 ms`; **beats every arm
    including `morsel_no_pushdown` `194.5 ms`, `stock_default` `194.7 ms`, and
    DuckDB 2.0.1 `212.8 ms`** because `lineitem` switches to `BulkDecode` while
    `orders` keeps Bloom `RowFilter`: `64` RF / `240` Bulk RGs).
  - **`q9`:** `1,325.0 → 1,308.2 ms` (**`1.01×`**, `−16.8 ms`; `167.9 ms`
    faster than `stock_pushdown` `1,476.1 ms`).
  - **Dynamic-join `RowFilter` wins retained or improved:** `q7`
    (`401.4 → 403.0 ms`, `1.00×`, `2.31×` vs. `no_pushdown`, within `1.3%` of
    DuckDB `397.7 ms`), `q4` (`203.4 → 197.2 ms`, `1.03×`), `q5`
    (`408.5 → 408.9 ms`, `1.00×`), `q11` (`124.8 → 111.1 ms`, `1.12×`), `q18`
    (`631.4 → 603.9 ms`, `1.05×`, `297` RF / `4` Bulk RGs vs. `1,723.4 ms`
    no-pushdown), `q19` (`387.4 → 373.6 ms`, `1.04×`), `q20`
    (`292.0 → 281.6 ms`, `1.04×`, beats DuckDB `291.2 ms`), `q21`
    (`874.0 → 834.8 ms`, `1.05×`), `q3` (`357.7 → 361.3 ms`, `0.99×`), `q14`
    (`310.7 → 315.2 ms`, `0.99×`, beats DuckDB `325.3 ms`), `q17`
    (`457.9 → 464.2 ms`, `0.99×`, `2.67×` vs. `no_pushdown`), `q8`
    (`517.4 → 536.6 ms`, `0.96×`, `1.25×` vs. `no_pushdown`, `67.1 ms` ahead of
    `stock_pushdown`).
  - **Head-to-head vs. DuckDB 2.0.1:** Single-config `morsel_pushdown` beats
    DuckDB on `5 / 22` queries (`q12`, `q14`, `q16`, `q20`, `q22`) and lands
    within `1.3%` on `q7`.

## Method & Environment

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (8 pinned via `taskset -c 0-7`), 61 GiB RAM.
- **Idle verification:** `vmstat -y 1 3` immediately before the run showed
  `96–99%` CPU idle; immediately after showed `95–99%` CPU idle.
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
| `q1`        |            693.4 |         703.1 (688.7–707.4) |             699.0 |            741.7 |           730.6 |         462.9 |               0.99× |             0.99× |                102 / 138 |
| `q2`        |            148.7 |         149.6 (146.2–150.7) |             213.1 |            159.0 |           148.4 |         117.7 |               0.99× |         **1.42×** |                  62 / 16 |
| `q3`        |            357.7 |         361.3 (358.1–368.0) |             504.7 |            491.7 |           356.8 |         336.2 |               0.99× |         **1.40×** |                 267 / 43 |
| `q4`        |            203.4 |     **197.2** (191.9–200.0) |             279.7 |            178.5 |           267.6 |         182.2 |           **1.03×** |         **1.42×** |                 271 / 33 |
| `q5`        |            408.5 |         408.9 (407.1–410.3) |             701.7 |            663.0 |           449.2 |         368.5 |           **1.00×** |         **1.72×** |                  308 / 5 |
| `q6`        |            449.6 |     **321.0** (313.5–327.1) |             320.8 |            337.8 |           485.1 |         199.0 |           **1.40×** |             1.00× |                  4 / 236 |
| `q7`        |            401.4 |         403.0 (399.0–405.2) |             929.4 |            835.8 |           464.3 |         397.7 |           **1.00×** |         **2.31×** |                 264 / 49 |
| `q8`        |            517.4 |         536.6 (533.4–537.8) |             669.7 |            626.1 |           603.7 |         441.6 |               0.96× |         **1.25×** |                 277 / 44 |
| `q9`        |          1,325.0 |   **1,308.2** (1297.0–1320.9) |           1,227.2 |            975.6 |         1,476.1 |         754.3 |           **1.01×** |             0.94× |                 32 / 217 |
| `q10`       |            564.6 |         581.1 (573.2–584.8) |             719.2 |            610.9 |           497.2 |         436.3 |               0.97× |         **1.24×** |                  305 / 5 |
| `q11`       |            124.8 |     **111.1** (109.3–115.1) |             164.9 |             95.1 |           105.4 |         103.8 |           **1.12×** |         **1.48×** |                  28 / 40 |
| `q12`       |            237.3 |     **174.4** (172.4–177.1) |             194.5 |            194.7 |           273.5 |         212.8 |           **1.36×** |         **1.12×** |                 64 / 240 |
| `q13`       |            552.4 |         552.3 (546.3–569.8) |             516.9 |            453.2 |           531.9 |         512.6 |           **1.00×** |             0.94× |                   62 / 2 |
| `q14`       |            310.7 |         315.2 (311.2–345.1) |             395.7 |            381.3 |           269.2 |         325.3 |               0.99× |         **1.26×** |                  237 / 3 |
| `q15`       |            902.6 |     **672.1** (662.4–719.7) |             653.2 |            670.7 |           885.9 |         261.4 |           **1.34×** |             0.97× |                 30 / 450 |
| `q16`       |            152.3 |         157.0 (150.2–157.5) |             154.6 |            118.2 |           127.5 |         158.2 |               0.97× |             0.98× |                   39 / 2 |
| `q17`       |            457.9 |         464.2 (460.6–468.2) |           1,241.8 |          1,110.8 |           428.9 |         391.4 |               0.99× |         **2.67×** |                  481 / 7 |
| `q18`       |            631.4 |     **603.9** (599.7–619.9) |           1,723.4 |          1,337.7 |           767.3 |         449.7 |           **1.05×** |         **2.85×** |                  297 / 4 |
| `q19`       |            387.4 |     **373.6** (366.6–378.1) |             458.3 |            532.7 |           393.9 |         319.6 |           **1.04×** |         **1.23×** |                  244 / 4 |
| `q20`       |            292.0 |     **281.6** (278.3–286.6) |             652.7 |            421.8 |           289.3 |         291.2 |           **1.04×** |         **2.32×** |                  275 / 7 |
| `q21`       |            874.0 |     **834.8** (830.7–875.8) |           1,552.1 |          1,258.2 |         1,399.2 |         702.9 |           **1.05×** |         **1.86×** |                534 / 252 |
| `q22`       |            108.2 |         111.2 (105.7–114.7) |             147.7 |            108.3 |           120.3 |         137.7 |               0.97× |         **1.33×** |                  34 / 42 |
| **GeoMean** |        **376.4** |                   **358.5** |         **501.3** |        **429.3** |       **395.4** |     **301.2** |           **1.05×** |         **1.40×** |                        — |
| **Sum**     |     **10,100.7** |                 **9,621.5** |      **14,120.2** |     **12,302.7** |    **11,071.4** |   **7,563.2** |           **1.05×** |         **1.47×** |                        — |

## Notes

- Scoring each same-column `RowFilter` candidate group by the **sum** of its
  constituent expression costs (rather than minimum) places `lineitem`'s
  `l_shipdate` range (`cost = 10`, ~2.9 MB) ahead of the multi-stage
  `l_suppkey` range + Bloom group (`cost = 10 + 3 = 13`, ~5.4 MB), recovering
  `q7` (`401.4 → 403.0 ms`, `1.00×`) while keeping `q8` (`536.6 ms`, `1.25×`
  vs. `no_pushdown`), `q18` (`603.9 ms`, `1.05×`), `q19` (`373.6 ms`, `1.04×`),
  and `q20` (`281.6 ms`, `1.04×`).
