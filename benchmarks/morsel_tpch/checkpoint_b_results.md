<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF1 Checkpoint B Scorecard: Stock DataFusion vs. Morsel DataFusion vs. DuckDB (#174)

This scorecard records the controlled idle-host **TPC-H Checkpoint B** 3-way
comparison (**Morsel DataFusion (`required`) vs. Stock DataFusion (`off`) vs.
DuckDB 2.0.1**) across **TPC-H Q1, Q3, Q6, and Q12** on standard **TPC-H SF1
Parquet** (`8,661,245` total rows, `35` Zstd files, `79` row groups,
`266,865,655` bytes) on `rob-tandy-dduck2` after `#168` (`GROUP BY`), `#158` /
`#188` (native `HashJoin`), `#169` (`BatchView`), and `#159` (native parallel
`TopN` & `Sort`) landed on `morsel-execution`.

This scorecard complements the Checkpoint A (`#173`) single-table report in
[checkpoint_a_results.md](checkpoint_a_results.md) and the pinned pre-`#150`
baseline in [README.md](README.md#sf1-warm-cache-baseline-rob-tandy-dduck2-2026-11-22-utc).

## Executive Summary

All four Checkpoint B standard TPC-H queries (`q1`, `q3`, `q6`, `q12`) passed
exact lossless Arrow IPC ordered result parity against **DuckDB 2.0.1** and
executed natively under `MorselExec: workers=8` with `morsel_mode=required`
(zero fallback, zero SQL/expression edge cases, exact `bytes_scanned` parity
with Stock DataFusion, and full `#138` per-pipeline wait/work breakdowns).

### Warm Page-Cache 3-Way Scorecard (8 Pinned Cores, Fresh Process, Median Total ms)

| Query                        | Best Morsel (`required`)\*          | Stock Default (`off`) | Best Stock (`off`, tuned)                  | DuckDB 2.0.1 |    Morsel vs. Stock Default | Morsel vs. Best Stock | Morsel vs. DuckDB |
| ---------------------------- | ----------------------------------- | --------------------: | ------------------------------------------ | -----------: | --------------------------: | --------------------: | ----------------: |
| `q1`                         | **163.2 ms** (`morsel_pushdown`)    |              208.6 ms | 159.4 ms (`stock_pushdown_no_repartition`) |     111.8 ms | **1.28× faster (−45.4 ms)** |       0.98× (+3.7 ms) |  1.46× (+51.3 ms) |
| `q3`                         | **120.7 ms** (`morsel_no_pushdown`) |              125.7 ms | 92.9 ms (`stock_pushdown_no_repartition`)  |      97.4 ms |  **1.04× faster (−5.0 ms)** |      0.77× (+27.8 ms) |  1.24× (+23.3 ms) |
| `q6`                         | **84.8 ms** (`morsel_pruning`)      |               97.9 ms | 83.0 ms (`stock_no_pushdown`)              |      54.2 ms | **1.16× faster (−13.1 ms)** |       0.98× (+1.7 ms) |  1.56× (+30.5 ms) |
| `q12`                        | **174.3 ms** (`morsel_no_pushdown`) |               64.5 ms | 60.8 ms (`stock_no_pushdown`)              |      62.3 ms |           0.37× (+109.8 ms) |     0.35× (+113.5 ms) | 2.80× (+112.0 ms) |
| **Geomean (4q)**             | **130.6 ms**                        |          **113.4 ms** | **93.0 ms**                                |  **77.9 ms** |           1.15× slower (4q) |          1.40× slower |      1.68× slower |
| **Geomean (`q1`,`q3`,`q6`)** | **118.6 ms**                        |          **136.9 ms** | **107.1 ms**                               |  **83.9 ms** |       **1.15× faster (3q)** |          1.11× slower |      1.41× slower |

\*Main 224-trial scorecard recorded with `MORSEL_METRICS=1` so every Morsel
trial captures `#138` per-pipeline metrics. A separate (non-interleaved)
`MORSEL_METRICS=0` run gave `q1`: **`156.9 ms`** (`morsel_pushdown`, faster
than even tuned `stock_pushdown_no_repartition`'s `159.4 ms`), `q3`: `121.6 ms`,
`q6`: `85.2 ms` (`morsel_no_pushdown`), and `q12`: `180.0 ms`.

### Headline Takeaways

1. **Morsel beats unmodified Stock DataFusion (`stock_default`) on Q1, Q3, and
   Q6 — by `1.15×` geomean warm and `1.20×` geomean cold:**
   - **Q1 (full query with `ORDER BY`):** `163.2 ms` (`156.9 ms` with metrics
     off) beats `stock_default` (`208.6 ms`) by **`1.28×` (`−45.4 ms`)** and
     `stock_pushdown` (`197.4 ms`) by **`1.21×` (`−34.2 ms`)**, and is even
     faster than Checkpoint A's `q1_unordered` (`170.4 ms`) thanks to `#169`
     `BatchView`. `#159` `SortSink` adds only `0.2 ms` (`0.1 ms` worker +
     `0.1 ms` finalize).
   - **Q3 (3-way join + `GROUP BY` + `TopN 10`):** `morsel_no_pushdown`
     (`120.7 ms`) beats both `stock_default` (`125.7 ms`, `1.04×`) and
     `stock_no_pushdown` (`122.4 ms`, `1.01×`). `#159` `TopNSink` takes just
     `0.6 ms` (`0.4 ms` worker + `0.2 ms` finalize). However, Stock's
     `stock_pushdown_no_repartition` (`92.9 ms`) and `stock_pushdown`
     (`97.6 ms`) leap ahead of both Morsel and DuckDB (`97.4 ms`) via
     **dynamic join filter pushdown (`#170`)**, which cuts Stock's Q3 CPU by
     `−182.9 ms` (`−32.8 ms` wall).
   - **Q6 (filtered revenue `SUM`):** `morsel_pruning` (`84.8 ms`) /
     `morsel_no_pushdown` (`85.7 ms`) beats `stock_default` (`97.9 ms`) by
     **`1.16×` (`−13.1 ms`)** and improves `−2.6` to `−3.5 ms` over Checkpoint
     A (`88.3 ms`) via `#169` `BatchView`.
2. **Q12 reveals a massive, immediately actionable join-build bottleneck (`#138`
   per-pipeline metrics caught it red-handed):**
   - Morsel takes `174.3 ms` (`165.1 ms` exec) on Q12 vs. Stock `64.5 ms` and
     DuckDB `62.3 ms` (`2.80×` slower).
   - `#138` per-pipeline metrics show that **`113.2 ms` (`68.6%` of total exec
     wall time)** is spent in **single-threaded coordinator
     `HashJoinBuildSink::finalize`** on the `build` pipeline, while all
     parallel scan/decode/probe/sort work totals only **`268.4 ms` across 8
     workers** (less than DuckDB's `389.9 ms`!).
   - Root cause verified in `MorselQueryPlanner::estimate_side_size()`: it
     estimates Parquet join inputs purely from raw file bytes
     (`total_file_bytes * 4`) ignoring filter selectivity, so it picks
     **unfiltered `orders` (`1,500,000` rows)** as the build side instead of
     **filtered `lineitem` (`30,988` rows — `48.4×` smaller!)** chosen by both
     Stock DataFusion and DuckDB, and then finalizes that 1.5M-row hash table
     serially on 1 thread.
3. **Under verified cold page cache (`0 / total` resident pages via
   `posix_fadvise` + `mincore`), Morsel's async prefetch queue cuts cold NVMe
   read penalties by ~2× vs. Stock:**
   - Warm-to-cold penalty on `morsel_no_pushdown` vs. `stock_default`:
     `q1` `+4.2` vs. `+11.0 ms`, `q3` `+15.1` vs. `+22.6 ms`, `q6` `+3.8` vs.
     `+12.2 ms`, `q12` `+11.8` vs. `+17.1 ms`. On cold Q6, Morsel (`89.4 ms`)
     widens its lead over `stock_default` (`110.1 ms`) to **`1.23×`** and
     narrows the gap to DuckDB (`62.8 ms`) from `1.56×` warm to **`1.42×`**
     cold.

## Environment and Measurement Controls

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated CPUs),
  `--threads 8` (Tokio `worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`; DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification (`vmstat -y 1 3`):**
  - Warm scorecard (224 trials): **89–93% idle** before, **87–92% idle** after.
  - Cold-cache scorecard (100 trials): **89–93% idle** before, **88–93% idle**
    after.
  - DuckDB operator profile (28 trials): **92–93% idle** before, **97–99%
    idle** after.
  - `MORSEL_METRICS=0` check (56 trials): **88–89% idle** before, **90–99%
    idle** after.
- **Build & versions:**
  - DataFusion revision: `f54afc6b07cd82b9726b24100c722a5f823587bd`
    (`agent/morsel-issue-174`, updating only `benchmarks/morsel_tpch/` on top
    of `morsel-execution` HEAD `8e5a9ed58` after `#159`), built with
    `--locked --profile release-nonlto -p datafusion --no-default-features --features parquet,sql,datetime_expressions,unicode_expressions --example morsel_tpch_runner`.
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`,
    Tokio `1.54.1`, Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`,
    NumPy `2.5.5`.
- **Dataset & page-cache verification:** Deterministic `#165` SF1 TPC-H Parquet
  dataset generated on local EC2 NVMe Btrfs storage
  (`/dev/mapper/vg_ws_ephemeral-volume1p1` under
  `$HOME/.cache/issue174/tpch_sf1`): `lineitem` `6,001,215` rows / `16` files /
  `48` RGs / `177,150,918` bytes (`43,256` 4 KiB pages); `orders` `1,500,000`
  rows / `8` files / `16` RGs / `44,579,195` bytes (`10,888` pages); `customer`
  `150,000` rows / `2` files / `2` RGs / `9,203,182` bytes (`2,248` pages).
  - **Warm campaign:** All **224 / 224** timed trials started and ended with
    **100% resident pages** via `mincore` (`43,256/43,256` for `q1`/`q6`,
    `54,144/54,144` for `q12`, `56,392/56,392` for `q3`).
  - **Cold campaign:** Before every trial, `evict_files()` ran `fsync` +
    `posix_fadvise(POSIX_FADV_DONTNEED)` and `require_cold_cache()` verified
    **`0 / total` resident pages** via `mincore` on all **100 / 100** timed
    trials.
- **Protocol:** Fresh process per trial, counterbalanced alternating arm order
  between odd and even rounds. Warm campaign: `1` warmup + `7` repeats; cold
  campaign: `0` warmups + `5` repeats. All timings are medians in milliseconds
  with observed `min–max` in parentheses.

## 1. Full 3-Way Warm-Cache SF1 Scorecard (8 Threads, `taskset -c 0-7`)

All DataFusion arms scanned identical Parquet bytes (`51,899,102` on `q1`,
`65,276,113`–`65,282,225` on `q3`, `47,682,742` on `q6`, `44,405,916` on
`q12`). Every `morsel_mode=required` trial claimed all Parquet row groups
(`48` on `q1`/`q6`, `66` on `q3` = `2+16+48`, `64` on `q12` = `16+48`), passed
exact ordered DuckDB parity, and emitted the expected finalizations (`q1`:
`agg=1, sort=1`; `q3`: `join_build=2, agg=1, topn=1`; `q6`: `agg=1`; `q12`:
`join_build=1, agg=1, sort=1`).

### TPC-H Q1 (Filtered 8-Aggregate `GROUP BY` + `ORDER BY`, 4 Rows, 98.6% Selectivity)

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |       171.6 (166.6–183.6) |     4.6 + 5.6 |      161.3 (156.7–173.0) |     1,193.9 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |       170.9 (165.2–174.9) |     4.6 + 4.0 |      162.2 (156.6–166.5) |     1,203.7 |      48 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |       173.6 (168.7–209.7) |     4.6 + 4.9 |      164.1 (159.7–200.6) |     1,210.9 |      48 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |       208.6 (206.8–221.0) |     4.4 + 6.5 |      197.6 (195.7–210.2) |     1,200.1 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |       159.4 (157.7–181.9) |     4.2 + 6.2 |      148.8 (147.1–171.3) |     1,094.8 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |   **163.2 (155.8–166.8)** |     4.4 + 4.8 |  **153.9 (146.2–157.9)** |     1,108.7 |      48 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |       197.4 (193.8–198.1) |     4.6 + 6.3 |      186.5 (183.0–187.6) |     1,099.4 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |   **111.8 (104.8–118.2)** |             — |                        — |       734.9 |       — |

_(With `MORSEL_METRICS=0`: `morsel_pushdown` drops to **`156.9 ms`** total /
`148.1 ms` exec; `morsel_no_pushdown` is `169.8 ms` total / `161.6 ms` exec.)_

### TPC-H Q3 (3-Way `customer ⋈ orders ⋈ lineitem` + `GROUP BY` + `TopN 10`, 10 Rows)

SF1 cardinalities: `customer` filtered `30,142` rows → `customer ⋈ orders`
(`o_orderdate < '1995-03-15'`, `727,305` rows) = `147,126` rows → `lineitem`
(`l_shipdate > '1995-03-15'`, `3,241,776` rows) → 3-way join `30,519` rows →
`11,620` groups → `TopN 10`.

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |       122.4 (116.5–140.1) |     5.2 + 6.8 |      110.2 (104.6–127.5) |       739.3 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |   **120.7 (117.4–135.2)** |     5.2 + 4.0 |  **111.7 (108.3–125.3)** |       694.5 |      66 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |       126.3 (117.8–132.0) |     5.4 + 5.4 |      115.4 (107.2–121.2) |       694.0 |      66 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |       125.7 (122.8–128.0) |     5.2 + 7.8 |      112.9 (110.0–114.8) |       624.5 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |     **92.9 (91.1–117.6)** |     5.3 + 7.3 |     **80.4 (78.6–93.3)** |       441.6 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |       132.4 (127.1–137.9) |     5.4 + 5.4 |      121.7 (116.3–126.9) |       711.6 |      66 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |         97.6 (95.9–100.6) |     5.3 + 7.6 |         84.8 (83.3–87.1) |       446.2 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |         97.4 (96.3–136.8) |             — |                        — |       549.9 |       — |

_(With `MORSEL_METRICS=0`: `morsel_no_pushdown` `121.6 ms`, `morsel_pushdown`
`128.5 ms`.)_

### TPC-H Q6 (Filtered Revenue `SUM`, 1 Row, 1.90% Selectivity)

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |          83.0 (78.9–85.7) |     4.2 + 3.7 |         75.0 (71.1–78.0) |       519.7 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |          85.7 (80.0–88.7) |     4.5 + 2.9 |         78.4 (72.5–81.1) |       536.2 |      48 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |      **84.8 (80.9–87.1)** |     4.3 + 4.0 |     **76.4 (72.6–79.0)** |       538.7 |      48 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |          97.9 (96.2–99.1) |     4.6 + 4.9 |         88.9 (86.5–89.5) |       526.6 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |       114.9 (108.8–118.5) |     4.4 + 4.7 |       106.2 (99.8–109.6) |       708.0 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |       115.6 (109.6–120.7) |     4.5 + 4.0 |      107.1 (101.1–112.1) |       704.8 |      48 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |       133.3 (131.3–145.0) |     4.5 + 4.8 |      124.1 (122.1–130.0) |       705.7 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |      **54.2 (52.6–64.9)** |             — |                        — |       357.2 |       — |

_(With `MORSEL_METRICS=0`: `morsel_no_pushdown` `85.2 ms`, `morsel_pushdown`
`114.7 ms`.)_

### TPC-H Q12 (2-Way `orders ⋈ lineitem` + Conditional `SUM` + `GROUP BY` + `ORDER BY`, 2 Rows)

SF1 cardinalities: `lineitem` filtered `30,988` rows (`0.52%` of 6.00M);
`orders` unfiltered `1,500,000` rows.

| Arm (`mode` / scan profile)                                                  | Median total ms (min–max) | Reg + plan ms | Median exec ms (min–max) | Exec CPU ms | Morsels |
| ---------------------------------------------------------------------------- | ------------------------: | ------------: | -----------------------: | ----------: | ------: |
| `stock_no_pushdown` (`off`, `pd=false`, `prune=false`, `repart=false`)       |      **60.8 (57.8–63.7)** |     5.4 + 5.6 |     **49.7 (46.4–52.9)** |       287.4 |       0 |
| `morsel_no_pushdown` (`required`, `pd=false`, `prune=false`, `repart=false`) |   **174.3 (171.8–192.3)** |     5.5 + 4.1 |  **165.1 (162.0–182.7)** |       405.7 |      64 |
| `morsel_pruning` (`required`, `pd=false`, `prune=true`, `repart=false`)      |       179.3 (173.3–184.0) |     5.5 + 5.4 |      168.5 (162.3–172.9) |       410.0 |      64 |
| `stock_default` (`off`, defaults: `pd=false`, `prune=true`, `repart=true`)   |          64.5 (62.8–68.3) |     5.7 + 6.8 |         52.2 (50.5–55.3) |       292.2 |       0 |
| `stock_pushdown_no_repartition` (`off`, `pd=true`, `repart=false`)           |          75.7 (71.3–84.8) |     5.9 + 6.6 |         63.8 (59.0–72.2) |       400.4 |       0 |
| `morsel_pushdown` (`required`, `pd=true`, `repart=false`)                    |       192.1 (189.4–208.5) |     5.5 + 5.4 |      181.1 (178.5–197.4) |       470.5 |      64 |
| `stock_pushdown` (`off`, defaults + `pd=true`, `repart=true`)                |          81.8 (79.9–84.5) |     5.6 + 6.8 |         69.4 (67.9–71.6) |       400.1 |       0 |
| `duckdb_default` (DuckDB 2.0.1 defaults)                                     |          62.3 (58.2–69.8) |             — |                        — |       361.4 |       — |

_(With `MORSEL_METRICS=0`: `morsel_no_pushdown` `180.0 ms`, `morsel_pushdown`
`201.6 ms`.)_

## 2. Verified Page-Cache-Cold Check (`--cache cold`, 8 Threads, 5 Repeats)

Every cold trial evicted all queried Parquet files via `fsync` +
`posix_fadvise(POSIX_FADV_DONTNEED)` and verified **`0 / total` resident 4 KiB
pages** via `mincore` immediately before execution. Across all 100 cold trials,
DataFusion faulted in `30.8–40.6%` of total file pages (the projected Parquet
column chunks + footers), while DuckDB faulted in `25.3–34.9%`.

| Query | Arm                  | Warm ms |       Cold ms (min–max) |  Cold delta |   Pages faulted (after) | Cold `pending_io` / `ready_wait` ms |
| ----- | -------------------- | ------: | ----------------------: | ----------: | ----------------------: | ----------------------------------: |
| `q1`  | `stock_default`      |   208.6 |     219.5 (217.0–223.0) |    +11.0 ms | 14,787 / 43,256 (34.2%) |                                   — |
| `q1`  | `stock_pushdown`     |   197.4 |     204.8 (202.5–207.5) |     +7.4 ms | 14,787 / 43,256 (34.2%) |                                   — |
| `q1`  | `morsel_no_pushdown` |   170.9 |     175.1 (171.7–180.4) |     +4.2 ms | 14,787 / 43,256 (34.2%) |                        183.3 / 41.5 |
| `q1`  | `morsel_pushdown`    |   163.2 | **169.2 (163.4–170.5)** | **+6.0 ms** | 14,787 / 43,256 (34.2%) |                        196.0 / 53.7 |
| `q1`  | `duckdb_default`     |   111.8 | **112.5 (109.9–118.5)** |     +0.7 ms | 12,803 / 43,256 (29.6%) |                                   — |
| `q3`  | `stock_default`      |   125.7 |     148.3 (144.7–159.7) |    +22.6 ms | 22,889 / 56,392 (40.6%) |                                   — |
| `q3`  | `stock_pushdown`     |    97.6 |     119.1 (117.5–121.1) |    +21.5 ms | 21,245 / 56,392 (37.7%) |                                   — |
| `q3`  | `morsel_no_pushdown` |   120.7 | **135.8 (133.5–138.7)** |    +15.1 ms | 22,897 / 56,392 (40.6%) |                        277.4 / 98.9 |
| `q3`  | `morsel_pushdown`    |   132.4 |     142.4 (141.1–144.1) |    +10.0 ms | 22,889 / 56,392 (40.6%) |                       258.8 / 127.9 |
| `q3`  | `duckdb_default`     |    97.4 | **101.7 (100.8–104.4)** |     +4.4 ms | 19,673 / 56,392 (34.9%) |                                   — |
| `q6`  | `stock_default`      |    97.9 |     110.1 (108.5–111.4) |    +12.2 ms | 13,807 / 43,256 (31.9%) |                                   — |
| `q6`  | `stock_pushdown`     |   133.3 |     145.5 (142.2–146.9) |    +12.2 ms | 13,807 / 43,256 (31.9%) |                                   — |
| `q6`  | `morsel_no_pushdown` |    85.7 |    **89.4 (88.2–91.8)** | **+3.8 ms** | 13,807 / 43,256 (31.9%) |                        191.6 / 37.9 |
| `q6`  | `morsel_pushdown`    |   115.6 |     124.7 (121.2–126.6) |     +9.1 ms | 13,807 / 43,256 (31.9%) |                       201.7 / 114.6 |
| `q6`  | `duckdb_default`     |    54.2 |    **62.8 (57.0–66.3)** |     +8.6 ms | 11,900 / 43,256 (27.5%) |                                   — |
| `q12` | `stock_default`      |    64.5 |        81.6 (79.6–85.1) |    +17.1 ms | 16,656 / 54,144 (30.8%) |                                   — |
| `q12` | `stock_pushdown`     |    81.8 |       98.8 (95.7–100.8) |    +17.0 ms | 16,656 / 54,144 (30.8%) |                                   — |
| `q12` | `morsel_no_pushdown` |   174.3 | **186.1 (184.1–189.4)** |    +11.8 ms | 16,656 / 54,144 (30.8%) |                        218.7 / 65.3 |
| `q12` | `morsel_pushdown`    |   192.1 |     202.8 (198.4–213.7) |    +10.7 ms | 16,656 / 54,144 (30.8%) |                       198.3 / 120.4 |
| `q12` | `duckdb_default`     |    62.3 |    **66.1 (65.7–69.4)** |     +3.8 ms | 13,700 / 54,144 (25.3%) |                                   — |

Cold-cache takeaway: Morsel's bounded async prefetch (`ReadyReaders`, peak
depth `8`) hides local NVMe read latency significantly better than Stock's
default pull-driven Parquet scan (`+3.8–15.1 ms` cold delta on
`morsel_no_pushdown` vs. `+11.0–22.6 ms` on `stock_default`), widening Morsel's
cold lead over `stock_default` to **`1.30×`** on `q1`, **`1.09×`** on `q3`, and
**`1.23×`** on `q6`.

## 3. `#138` Per-Pipeline Wait/Work Breakdowns & Cross-Engine Operator Decomposition

### Morsel `#138` Per-Pipeline Metrics (Warm Page Cache, Summed Across 8 Workers, Median ms)

Coordinator work (`join_combine`, `join_finalize`, `sort_finalize`,
`topn_finalize`) is **single-threaded wall time**; `decode`, `process`,
`ready_wait`, and `output_wait` are summed across 8 workers.

| Query | Arm                  | Pipeline      | Decode ms | Process ms | Ready wait ms | Output wait ms | Join comb/fin ms | Sort/TopN fin ms | Claims (min..max) |
| ----- | -------------------- | ------------- | --------: | ---------: | ------------: | -------------: | ---------------: | ---------------: | ----------------: |
| `q1`  | `morsel_no_pushdown` | `main`        |     676.1 |      507.2 |          14.9 |            0.0 |          0.0/0.0 |              0.0 |              6..6 |
| `q1`  | `morsel_no_pushdown` | `sort`        |       0.0 |        0.1 |           0.0 |            0.0 |          0.0/0.0 |              0.1 |              0..1 |
| `q1`  | `morsel_pushdown`    | `main`        |     678.8 |      388.6 |          21.9 |            0.0 |          0.0/0.0 |              0.0 |             2..10 |
| `q1`  | `morsel_pushdown`    | `sort`        |       0.0 |        0.1 |           0.0 |            0.0 |          0.0/0.0 |              0.1 |              0..1 |
| `q3`  | `morsel_no_pushdown` | `build_build` |       5.7 |        1.1 |           4.9 |            0.0 |      0.0/**1.0** |              0.0 |              0..1 |
| `q3`  | `morsel_no_pushdown` | `build`       |      89.7 |       33.0 |          12.8 |            0.0 |      0.0/**9.2** |              0.0 |              1..3 |
| `q3`  | `morsel_no_pushdown` | `probe`       |     402.5 |      121.7 |          14.8 |            0.0 |          0.0/0.0 |              0.0 |              6..6 |
| `q3`  | `morsel_no_pushdown` | `topn`        |       0.0 |        0.4 |           0.0 |            0.0 |          0.0/0.0 |              0.2 |              0..1 |
| `q3`  | `morsel_pushdown`    | `build_build` |       6.8 |        0.1 |          11.2 |            0.0 |      0.0/**1.0** |              0.0 |              0..1 |
| `q3`  | `morsel_pushdown`    | `build`       |      92.8 |       27.2 |          14.4 |            0.0 |      0.0/**9.6** |              0.0 |              0..5 |
| `q3`  | `morsel_pushdown`    | `probe`       |     432.7 |      108.0 |          11.8 |            0.0 |          0.0/0.0 |              0.0 |             2..12 |
| `q3`  | `morsel_pushdown`    | `topn`        |       0.0 |        0.4 |           0.0 |            0.0 |          0.0/0.0 |              0.2 |              0..1 |
| `q6`  | `morsel_no_pushdown` | `main`        |     495.7 |       42.9 |          13.3 |            0.0 |          0.0/0.0 |              0.0 |              6..6 |
| `q6`  | `morsel_pruning`     | `main`        |     480.6 |       43.5 |          16.1 |            0.0 |          0.0/0.0 |              0.0 |              6..6 |
| `q6`  | `morsel_pushdown`    | `main`        |     675.3 |        0.9 |          37.4 |            0.0 |          0.0/0.0 |              0.0 |              4..8 |
| `q12` | `morsel_no_pushdown` | `build`       |      81.4 |       11.9 |           6.1 |            0.0 |    0.0/**113.2** |              0.0 |              2..2 |
| `q12` | `morsel_no_pushdown` | `probe`       |     119.0 |       56.0 |          11.5 |            0.0 |          0.0/0.0 |              0.0 |              5..7 |
| `q12` | `morsel_no_pushdown` | `sort`        |       0.0 |        0.1 |           0.0 |            0.0 |          0.0/0.0 |              0.0 |              0..1 |
| `q12` | `morsel_pushdown`    | `build`       |      83.6 |       11.4 |           5.8 |            0.0 |    0.0/**114.0** |              0.0 |              2..2 |
| `q12` | `morsel_pushdown`    | `probe`       |     216.4 |        7.4 |          28.0 |            0.0 |          0.0/0.0 |              0.0 |              4..8 |
| `q12` | `morsel_pushdown`    | `sort`        |       0.0 |        0.1 |           0.0 |            0.0 |          0.0/0.0 |              0.0 |              0..1 |

_(Total `morsel_pending_io_ms` across pipelines on fastest warm arms: `q1`
`morsel_pushdown` `74.4 ms`, `q3` `morsel_no_pushdown` `111.6 ms`, `q6`
`morsel_pruning` `88.0 ms`, `q12` `morsel_no_pushdown` `69.8 ms`; `output_wait`
is `0.0 ms` on every query, arm, and pipeline.)_

### Cross-Engine Operator CPU Decomposition (Warm Cache, Aggregate ms Across 8 Threads)

Stock timings are median `elapsed_compute` per operator from the main 224-trial
run. DuckDB timings were collected via `harness.py duckdb-profile` (fresh
process/connection per trial, `taskset -c 0-7`, `SET threads = 8`, warm cache
verified via `mincore`, 1 warmup + 7 repeats, median of `operator[].timing`
summed by `type`).

| Query | Engine / Arm                          |                  Scan + Decode ms |                                      Non-Scan Operators (Filter / Join / Agg / Sort) ms | Total Op Work ms | Wall ms |
| ----- | ------------------------------------- | --------------------------------: | --------------------------------------------------------------------------------------: | ---------------: | ------: |
| `q1`  | Stock `stock_pushdown_no_repartition` |   757.1 (`DataSourceExec` + proj) |                                            311.0 (`Agg` 310.7 + `Sort` 0.2 + `SPM` 0.1) |          1,068.1 |   159.4 |
| `q1`  | Morsel `morsel_pushdown`              |             678.8 (`decode_work`) |                                                388.8 (`process` 388.7 + `sort_fin` 0.1) |          1,067.6 |   163.2 |
| `q1`  | DuckDB `duckdb_default`               |              435.8 (`TABLE_SCAN`) |                            317.3 (`PROJ` 65.4 + `HASH_GROUP_BY` 251.7 + `ORDER_BY` 0.3) |            745.2 |   111.8 |
| `q3`  | Stock `stock_default`                 |          485.5 (`DataSourceExec`) |        96.9 (`Filter` 41.3 + `HashJoin` 50.5 + `Agg` 4.3 + `TopK` 0.7 + `Coalesce` 0.1) |            582.4 |   125.7 |
| `q3`  | Stock `stock_pushdown_no_repartition` | 384.7 (`DataSourceExec` + dynflt) |                        18.6 (`HashJoin` 13.7 + `Agg` 4.0 + `TopK` 0.8 + `Coalesce` 0.1) |            403.3 |    92.9 |
| `q3`  | Morsel `morsel_no_pushdown`           |             493.8 (`decode_work`) |                   168.0 (`process` 157.6 + `join_fin` **10.2** serial + `topn_fin` 0.2) |            661.8 |   120.7 |
| `q3`  | DuckDB `duckdb_default`               |              513.6 (`TABLE_SCAN`) |                38.5 (`HASH_JOIN` 33.0 + `HASH_GROUP_BY` 4.6 + `TOP_N` 0.6 + `PROJ` 0.3) |            552.3 |    97.4 |
| `q6`  | Stock `stock_no_pushdown`             |          461.6 (`DataSourceExec`) |                                                        39.8 (`Filter` 39.1 + `Agg` 0.7) |            501.4 |    83.0 |
| `q6`  | Morsel `morsel_pruning`               |             480.6 (`decode_work`) |                                                                   43.5 (`process_work`) |            524.1 |    84.8 |
| `q6`  | DuckDB `duckdb_default`               |              391.7 (`TABLE_SCAN`) |                                                  6.9 (`PROJ` 5.3 + `UNGROUPED_AGG` 1.6) |            401.6 |    54.2 |
| `q12` | Stock `stock_no_pushdown`             |          173.2 (`DataSourceExec`) |      52.0 (`Filter` 33.7 + `HashJoin` 13.7 + `Agg` 4.2 + `Repart` 0.2 + `Sort/SPM` 0.2) |            225.2 |    60.8 |
| `q12` | Morsel `morsel_no_pushdown`           |             200.6 (`decode_work`) |                         181.5 (`process` 68.3 + `join_fin` **113.2** serial on 1 core!) |            382.1 |   174.3 |
| `q12` | DuckDB `duckdb_default`               |              331.7 (`TABLE_SCAN`) | 57.0 (`FILTER` 44.4 + `HASH_JOIN` 8.5 + `HASH_GROUP_BY` 2.6 + `PROJ` 1.3 + `ORDER` 0.1) |            389.9 |    62.3 |

## 4. Prioritized Bottleneck Attribution & Next Engineering Wins

### 1. Selectivity-Aware Join Build-Side Selection + Parallel Radix-Partitioned Hash Join Build (Top Q12 Bottleneck: `−113 ms` Wall!)

- **Measured impact on Q12:** `113.2 ms` out of `165.1 ms` exec (`68.6%` of
  wall time!) is spent in single-threaded `HashJoinBuildSink::finalize` on the
  coordinator.
- **Root cause #1 (build-side inversion):** In
  `datafusion/core/src/morsel.rs`, `estimate_side_size()` estimates Parquet
  join sides purely from raw file bytes (`total_file_bytes * 4`), ignoring
  filter predicates. On Q12, `orders` (`44.6 MB`, `1,500,000` unfiltered rows)
  looks smaller than `lineitem` (`177.2 MB`), even though `lineitem`'s
  predicates filter 6.00M rows down to **`30,988` rows (`48.4×` smaller)**.
  Both Stock DataFusion (`JoinSelection` with filter selectivity stats) and
  DuckDB build on filtered `lineitem` (`30,988` rows) and probe with `orders`
  (`8.5–13.8 ms` total join CPU).
- **Root cause #2 (serial coordinator hash-table build):** Even when a join
  genuinely has a large build side (e.g., `orders ⋈ customer` `147,126` rows in
  Q3 = `9.2 ms` serial finalize, or Q12's `1.5M` rows = `113.2 ms`),
  `HashJoinBuildSink::finalize` builds a single monolithic hash table on 1
  thread. Parallel radix-partitioned build/finalize across 8 workers eliminates
  the serial Amdahl bottleneck and scales to SF5+ large-build joins.
- **Projected win:** Fixing Q12's build-side selection (and/or parallelizing
  build finalize) removes `~110 ms` of serial wall time, projecting Q12 from
  **`174.3 ms → ~58–62 ms`** — matching or beating both `stock_default`
  (`64.5 ms`) and DuckDB (`62.3 ms`) and moving the 4-query Checkpoint B
  geomean from `130.6 ms` to **`~99 ms`** (**`1.15×` faster than `stock_default`
  across all 4 queries**).

### 2. Dynamic Join Filter Pushdown (`#170`) (Top Q3 Bottleneck: `−28 to −33 ms` Wall!)

- **Measured impact on Q3:** Stock `stock_pushdown_no_repartition` (`92.9 ms`)
  and `stock_pushdown` (`97.6 ms`) beat both Morsel (`120.7 ms`) and DuckDB
  (`97.4 ms`). Comparing `stock_default` (`pushdown_filters=false`, `125.7 ms`)
  to `stock_pushdown_no_repartition` (`92.9 ms`):
  - `DataSourceExec` drops **`−100.8 ms` CPU** (`485.5 → 384.7 ms`).
  - `FilterExec` disappears (`41.3 → 0.0 ms`).
  - `HashJoinExec` drops **`−36.8 ms` CPU** (`50.5 → 13.7 ms`).
  - Total process CPU drops **`−182.9 ms`** (`624.5 → 441.6 ms`, `−32.8 ms`
    wall).
- **Why:** Stock's `HashJoinExec` pushes `DynamicFilter` bounds/filters from
  the `customer` (`30,142` rows) and `orders ⋈ customer` (`147,126` rows)
  builds into the `orders` and `lineitem` Parquet `RowFilter`s, skipping
  `l_extendedprice` / `l_discount` `Decimal128` decoding and hash-table probes
  for non-matching rows. Without `#170`, `morsel_pushdown` only pushes static
  date filters (`48–54%` selectivity), adding `+39.3 ms` decode overhead
  (`120.7 → 132.4 ms`).
- **Projected win:** Implementing `#170` dynamic join filter pushdown +
  parallelizing Q3's `10.2 ms` serial `join_finalize` projects Morsel Q3 from
  **`120.7 ms → ~85–95 ms`**, matching tuned Stock (`92.9 ms`) and DuckDB
  (`97.4 ms`).

### 3. Parquet `FLBA(7)` → `Decimal128` Dictionary Pre-Decode & Late Materialization (Top Q1 & Q6 Gap vs. DuckDB)

- **Measured impact on Q1 and Q6:** Consistent with Checkpoint A (`#173`) and
  `#180`, `parquet 60.1.0` `FixedLenByteArrayReader` decodes 7-byte
  big-endian `Decimal128` values per row instead of pre-decoding small
  dictionary pages into `i128`:
  - **Q1:** Morsel `decode_work` `678.8 ms` vs. DuckDB `TABLE_SCAN` `435.8 ms`
    (`+243.0 ms` CPU / `+30.4 ms` over 8 cores = **`75.5%`** of the `322.0 ms`
    aggregate operator-CPU gap).
  - **Q6:** Morsel `decode_work` `480.6 ms` (`morsel_pruning`) vs. DuckDB
    `TABLE_SCAN` `391.7 ms` (`+88.9 ms` CPU / `+11.1 ms` over 8 cores =
    **`72.6%`** of the `122.5 ms` aggregate operator-CPU gap, plus DuckDB's
    late materialization skipping `l_extendedprice` decode for 98.1% filtered
    rows).
- Note that on non-`Decimal128`-heavy scans (`q3` `decode_work` `493.8 ms` vs.
  DuckDB `513.6 ms`; `q12` `decode_work` `200.6 ms` vs. DuckDB `331.7 ms`),
  Morsel's Parquet decode is already faster than DuckDB's `TABLE_SCAN`.

### 4. Multi-Conjunct `RowFilter` Regression & Q1 Aggregate Vectorization / CSE

- **Q6 & Q12 `pushdown_filters=true` regression (`+18` to `+30 ms`):** On
  multi-conjunct predicates (`q6` 4 conjuncts, `q12` 5 conjuncts), DataFusion's
  `RowFilter` splits conjuncts into separate `ArrowPredicate`s and re-decodes
  shared columns multiple times (`q6` decode `480.6 → 675.3 ms`; `q12` probe
  decode `119.0 → 216.4 ms`), forcing the fastest arm to run with
  `pushdown_filters=false` and pay `+33–56 ms` of post-decode filter work.
- **Q1 8-aggregate `process_work` (`388.7 ms` vs. DuckDB `317.3 ms`, `+71.4 ms`
  CPU / `22.2%` of the operator-CPU gap):** DuckDB applies aggregate CSE
  (reusing `SUM` accumulators for `AVG(l_quantity)` and `AVG(l_extendedprice)`)
  and hashes 1-byte `(l_returnflag, l_linestatus)` keys directly instead of
  encoding `Utf8View` pairs through Arrow `RowConverter`.

### 5. Verified Non-Bottlenecks (What Already Works Great)

- **`#159` `SortSink` & `TopNSink`:** `0.2 ms` on Q1 (`0.1 ms` worker +
  `0.1 ms` finalize), `0.6 ms` on Q3 (`0.4 ms` + `0.2 ms`), `0.1 ms` on Q12 —
  matching or beating Stock `SortExec` and DuckDB `ORDER_BY`/`TOP_N`.
- **`#169` `BatchView` zero-copy filtering:** Full Q1 with `ORDER BY`
  (`163.2 ms`) runs `7.2 ms` faster than Checkpoint A's `q1_unordered`
  (`170.4 ms`), and Q6 drops from `88.3 ms` to `84.8–85.7 ms`.
- **Dynamic row-group work-stealing & backpressure:** `6..6` balanced claims on
  `lineitem`, `0.0 ms` `morsel_output_wait` across all 224 warm and 100 cold
  trials, and automatic elimination of Stock `stock_default`'s
  `repartition_file_scans=true` `[6,6,6,7,6,6,6,5]` static straggler skew.

## 5. Live 22-Query Admission Status Post-`#159`

Verified at `--sf 0.01` via `test_harness.py` (`test_sf001_verify_parity_and_morsel_plan_checks`):
**`8 / 22` standard TPC-H queries** are now natively admitted under
`morsel_mode=required` with exact DuckDB parity (`q1`, `q3`, `q5`, `q6`, `q10`,
`q12`, `q14`, `q19`), plus `q1_unordered`, `q6_count`, and `q6_scan` — up from
`3 / 22` in `#189`. The remaining `14 / 22` standard queries reject cleanly at
planning time (`LeftSemi`/`LeftAnti` joins: `q2, q4, q17, q18, q20, q21`;
`SubqueryAlias`: `q7, q8, q9, q13, q22`; scalar `<subquery>` filter predicates:
`q11, q15`; nested `DISTINCT` aggregates: `q16`).

## 6. Exact Reproduction Commands

From the repository root on Linux (`rob-tandy-dduck2`; recorded at revision
`f54afc6b07cd82b9726b24100c722a5f823587bd` on top of `morsel-execution`
`8e5a9ed58`):

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work="$HOME/.cache/issue174"
mkdir -p "$work"
revision=$(git rev-parse HEAD)

# 1. Build the standalone runner as a detached Cargo example (no workspace edits).
git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$work/pinned/$b/tpch_runner.rs" \
  "$work/pinned/datafusion/core/examples/morsel_tpch_runner.rs"
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

# 2. Create the pinned Python environment.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"

# 3. Run unit + deterministic --sf 0.01 22-query parity, plan, and pipeline-metrics tests.
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 180s "$py" -B -m unittest discover -s "$b" -v

# 4. Generate deterministic SF1 TPC-H Parquet tables outside Git (on disk, not tmpfs).
data_dir="$work/tpch_sf1"
timeout 180s "$py" -B "$b/harness.py" generate "$data_dir" \
  --sf 1 --row-group-size 131072

# 5. Run the Checkpoint B 3-way SF1 warm-cache scorecard (8 pinned cores, 224 trials).
vmstat -y 1 3
timeout --kill-after=10s 600s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q1 q3 q6 q12 \
  --morsel-arms stock_no_pushdown morsel_no_pushdown morsel_pruning stock_default \
    stock_pushdown_no_repartition morsel_pushdown stock_pushdown duckdb_default \
  > "$work/sf1_checkpoint_b_warm.jsonl"
vmstat -y 1 3
"$py" -B "$b/harness.py" summarize "$work/sf1_checkpoint_b_warm.jsonl"

# 6. Run the verified page-cache-cold check (0/total resident pages before each trial).
vmstat -y 1 3
timeout --kill-after=10s 600s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 120 --cache cold --warmups 0 --repeats 5 \
  --queries q1 q3 q6 q12 \
  --morsel-arms stock_default stock_pushdown morsel_no_pushdown morsel_pushdown duckdb_default \
  > "$work/sf1_checkpoint_b_cold.jsonl"
vmstat -y 1 3
"$py" -B "$b/harness.py" summarize "$work/sf1_checkpoint_b_cold.jsonl"

# 7. Collect fresh-process DuckDB JSON operator timing breakdowns (Section 3).
vmstat -y 1 3
timeout --kill-after=10s 180s taskset -c 0-7 "$py" -B "$b/harness.py" duckdb-profile \
  "$data_dir" --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q1 q3 q6 q12 \
  > "$work/sf1_checkpoint_b_duckdb_profile.json"
vmstat -y 1 3

# 8. Separate (non-interleaved) instrumentation-overhead check (MORSEL_METRICS=0).
MORSEL_METRICS=0 timeout --kill-after=10s 300s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q1 q3 q6 q12 \
  --morsel-arms morsel_no_pushdown morsel_pushdown \
  > "$work/sf1_checkpoint_b_metrics_off.jsonl"
"$py" -B "$b/harness.py" summarize "$work/sf1_checkpoint_b_metrics_off.jsonl"

# 9. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```
