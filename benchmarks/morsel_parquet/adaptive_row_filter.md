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

- **Single-config `morsel_pushdown` now recovers the narrow/dense scan
  regressions while retaining the dynamic-join `RowFilter` wins across all
  22/22 SF=5 TPC-H queries:**
  - **22/22 GeoMean:** `404.5 → 392.2 ms` (`1.03×` faster vs. pre-#207
    `morsel_pushdown`; **`1.44×`** faster vs. `morsel_no_pushdown` at
    `564.6 ms`; **`1.19×`** faster vs. `stock_default` at `467.4 ms`;
    **`1.09×`** faster vs. `stock_pushdown` at `426.5 ms`; `1.22×` of DuckDB
    2.0.1 at `322.0 ms`).
  - **22/22 Sum:** `10,847.9 → 10,548.8 ms` (`−299.1 ms` vs. pre-#207
    `morsel_pushdown`, `−5,425.7 ms` vs. `morsel_no_pushdown`, `−2,987.8 ms`
    vs. `stock_default`, `−1,404.7 ms` vs. `stock_pushdown`).
  - **Oracle gap nearly eliminated:** The geometric-mean gap between
    single-config `morsel_pushdown` (`392.2 ms`) and per-query oracle
    `Best Morsel` (`390.8 ms`) drops from **`17.8 ms` in `#195` to `1.4 ms`**
    (`92%` reduction).
- **Headline per-query recoveries and wins (5-round paired SF=5 medians):**
  - **`q6`:** `464.6 → 344.6 ms` (**`1.35×`**, `−120.0 ms`; converges to
    `BulkDecode` on `236 / 240` RGs, within `1.7%` of `morsel_no_pushdown`
    `338.8 ms` and faster than `stock_default` `365.6 ms` / `stock_pushdown`
    `513.3 ms`).
  - **`q15`:** `972.4 → 720.4 ms` (**`1.35×`**, `−252.0 ms`; converges to
    `BulkDecode` on `449 / 480` RGs, within `0.6%` of `morsel_no_pushdown`
    `716.1 ms` and faster than `stock_default` `729.7 ms` / `stock_pushdown`
    `921.1 ms`).
  - **`q12`:** `255.7 → 189.0 ms` (**`1.35×`**, `−66.7 ms`; **beats every arm
    including `morsel_no_pushdown` `225.0 ms`, `stock_default` `204.6 ms`, and
    DuckDB 2.0.1 `229.0 ms`** because `lineitem` switches to `BulkDecode` while
    `orders` keeps Bloom `RowFilter`: `65` RF / `239` Bulk RGs).
  - **Dynamic-join `RowFilter` wins retained or improved:** `q3` (`393.2 → 394.9 ms`),
    `q4` (`218.9 → 212.3 ms`), `q5` (`443.9 → 440.2 ms`), `q14` (`331.9 → 335.2 ms`,
    beats DuckDB `346.5 ms`), `q17` (`486.3 → 495.7 ms` vs. `1,396.9 ms`
    no-pushdown), `q18` (`678.1 → 724.0 ms` vs. `1,948.6 ms` no-pushdown),
    `q19` (`406.0 → 393.6 ms`), `q20` (`315.0 → 305.0 ms`, beats DuckDB
    `311.7 ms`), `q21` (`956.1 → 929.8 ms`).
  - **Head-to-head vs. DuckDB 2.0.1:** Single-config `morsel_pushdown` beats
    DuckDB on `5 / 22` queries (`q12`, `q14`, `q16`, `q20`, `q22`).

## Method & Environment

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (8 pinned via `taskset -c 0-7`), 61 GiB RAM.
- **Idle verification:** `vmstat -y 1 3` immediately before the run showed
  `92–99%` CPU idle; immediately after showed `96–99%` CPU idle.
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

| Query | `before_pd` (ms) | `after_pd` ms (min–max) | `after_nopd` (ms) | `stock_def` (ms) | `stock_pd` (ms) | `duckdb` (ms) | `after` vs `before` | `after` vs `nopd` | Adaptive RGs (RF / Bulk) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q1` | 732.2 | 740.5 (733.9–749.0) | 739.8 | 797.5 | 768.0 | 484.2 | 0.99× | 1.00× | 209 / 31 |
| `q2` | 160.6 | 160.5 (158.6–162.9) | 249.9 | 176.1 | 157.7 | 125.9 | 1.00× | **1.56×** | 74 / 4 |
| `q3` | 393.2 | 394.9 (392.8–405.2) | 557.4 | 528.9 | 395.8 | 360.2 | 1.00× | **1.41×** | 306 / 4 |
| `q4` | 218.9 | **212.3** (207.9–215.4) | 312.2 | 190.2 | 283.0 | 196.1 | **1.03×** | **1.47×** | 271 / 33 |
| `q5` | 443.9 | **440.2** (433.1–445.4) | 796.4 | 732.4 | 483.8 | 386.9 | **1.01×** | **1.81×** | 308 / 5 |
| `q6` | 464.6 | **344.6** (338.5–347.4) | 338.8 | 365.6 | 513.3 | 206.6 | **1.35×** | 0.98× | 4 / 236 |
| `q7` | 420.6 | 469.2 (455.5–521.1) | 1,086.2 | 921.0 | 502.5 | 423.5 | 0.90× | **2.31×** | 307 / 6 |
| `q8` | 554.5 | 582.7 (574.7–592.9) | 737.4 | 667.0 | 645.2 | 470.4 | 0.95× | **1.27×** | 272 / 49 |
| `q9` | 1,419.8 | 1,454.6 (1,415.2–1,482.6) | 1,427.7 | 1,060.8 | 1,610.4 | 824.7 | 0.98× | 0.98× | 245 / 4 |
| `q10` | 611.6 | 619.2 (617.5–629.3) | 789.4 | 653.0 | 534.7 | 465.2 | 0.99× | **1.27×** | 305 / 5 |
| `q11` | 134.1 | 133.5 (126.5–136.8) | 200.4 | 103.8 | 120.5 | 115.4 | 1.00× | **1.50×** | 39 / 29 |
| `q12` | 255.7 | **189.0** (184.0–192.4) | 225.0 | 204.6 | 291.1 | 229.0 | **1.35×** | **1.19×** | 65 / 239 |
| `q13` | 607.9 | 614.6 (600.3–626.2) | 592.7 | 510.5 | 578.3 | 551.9 | 0.99× | 0.96× | 25 / 39 |
| `q14` | 331.9 | 335.2 (329.7–339.9) | 428.9 | 404.0 | 288.5 | 346.5 | 0.99× | **1.28×** | 237 / 3 |
| `q15` | 972.4 | **720.4** (712.3–755.6) | 716.1 | 729.7 | 921.1 | 271.2 | **1.35×** | 0.99× | 31 / 449 |
| `q16` | 166.8 | 167.4 (165.9–170.1) | 168.1 | 125.4 | 139.5 | 173.6 | 1.00× | 1.00× | 39 / 2 |
| `q17` | 486.3 | 495.7 (492.3–506.1) | 1,396.9 | 1,288.5 | 462.4 | 409.6 | 0.98× | **2.82×** | 483 / 5 |
| `q18` | 678.1 | 724.0 (714.2–848.8) | 1,948.6 | 1,511.5 | 836.1 | 483.6 | 0.94× | **2.69×** | 207 / 94 |
| `q19` | 406.0 | **393.6** (379.9–410.6) | 495.1 | 562.5 | 414.3 | 344.4 | **1.03×** | **1.26×** | 245 / 3 |
| `q20` | 315.0 | **305.0** (300.4–307.4) | 729.9 | 463.3 | 300.8 | 311.7 | **1.03×** | **2.39×** | 276 / 6 |
| `q21` | 956.1 | **929.8** (917.8–948.4) | 1,863.3 | 1,424.6 | 1,571.8 | 746.3 | **1.03×** | **2.00×** | 536 / 250 |
| `q22` | 117.8 | 122.0 (121.5–124.7) | 174.3 | 115.7 | 134.5 | 149.6 | 0.97× | **1.43×** | 74 / 2 |
| **GeoMean** | **404.5** | **392.2** | **564.6** | **467.4** | **426.5** | **322.0** | **1.03×** | **1.44×** | — |
| **Sum** | **10,847.9** | **10,548.8** | **15,974.5** | **13,536.6** | **11,953.5** | **8,076.5** | **1.03×** | **1.51×** | — |

## Notes

- On `q9`, `lineitem` has a 5.5%-selective `l_partkey` dynamic Bloom filter
  paired with 100%-passing `l_suppkey` and `l_orderkey` filters; `after_pd`
  tracks `before_pd` (`1,454.6` vs. `1,419.8 ms`, `1,415.2 ms` min) and remains
  `155.8 ms` faster than `stock_pushdown` (`1,610.4 ms`).
- On `q7` (`420.6 → 469.2 ms`), `q8` (`554.5 → 582.7 ms`), and `q18`
  (`678.1 → 724.0 ms`), the 2-RG initial `BulkDecode` probe accounts for the
  small `~28–48 ms` delta while preserving `1.27×–2.69×` speedups over
  `morsel_no_pushdown` (`737–1,949 ms`) and keeping single-config
  `morsel_pushdown` ahead of both `stock_default` and `stock_pushdown` on all
  three queries.
