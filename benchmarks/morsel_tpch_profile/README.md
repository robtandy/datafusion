<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF1 `q6_count` & `q6_scan` Performance Profile vs DuckDB (#180)

This read-only profiling investigation measures `morsel-execution` HEAD
(`9c3079473`, post-`#149` coalescing, `#150` Parquet `RowFilter` pushdown, and
`#166` Parquet direct mask-bit `COUNT(*)` + filter-column pruning) on the exact
standard `#165` TPC-H SF1 `lineitem` Parquet dataset (8 pinned cores, warm page
cache verified via `mincore`) and decomposes every millisecond of the remaining
gap to **DuckDB 2.0.1**.

## Bottom Line

Post-`#149`/`#150`/`#166`, **Morsel DataFusion (`morsel_no_pushdown`)** improved
from `47.3 ms` → **`46.4 ms`** fresh total (`35.6 ms` warm exec-only / `36.8 ms`
warm plan+exec) on `q6_count` and from `105.8 ms` → **`94.0 ms`** fresh total
(`75.4 ms` warm exec-only / `76.5 ms` warm plan+exec; 16 coalesced batches vs.
736 pre-`#149`) on `q6_scan`, beating `stock_default` by `10.1 ms` on `q6_count`
and `13.8 ms` on `q6_scan`. Yet **DuckDB 2.0.1** still finishes in **`25.1 ms`**
fresh (`17.2 ms` warm exec-only / `20.6 ms` warm plan+exec Arrow drain) on
`q6_count` (**`+21.3 ms`** fresh / **`+18.4 ms`** warm exec-only / **`+16.2 ms`**
warm plan+exec gap) and **`68.3 ms`** fresh (`48.6 ms` warm exec-only /
`52.7 ms` warm plan+exec Arrow drain) on `q6_scan` (**`+25.7 ms`** fresh /
**`+26.8 ms`** warm exec-only / **`+23.8 ms`** warm plan+exec gap).

**One single bottleneck explains nearly the entire execution gap on both
queries:**

- **`arrow-rs` (`parquet 60.1.0`) `FixedLenByteArrayReader` two-pass
  `FLBA(7)` → `Decimal128` decoding burns `70.2%` of all CPU cycles on
  `q6_count` and `54.1%` on `q6_scan`.** PyArrow writes `DECIMAL(15,2)` columns
  (`l_discount`, `l_quantity`, `l_extendedprice`) as dictionary-encoded
  `fixed_len_byte_array(7)`. Instead of decoding the 11-entry (`l_discount`) or
  50-entry (`l_quantity`) dictionary page once into `Vec<i128>`,
  `FixedLenByteArrayReader` performs **12–18 million 7-byte `memcpy` calls**
  into a temporary `FixedSizeBinaryArray` (`35.8%` of `q6_count` cycles) and
  then runs a second pass of **12–18 million `sign_extend_be` +
  `i128::from_be_bytes` conversions** (`34.4%` of `q6_count` cycles).
- **Direct empirical proof on the exact, unmodified `#165` SF1 Parquet files
  (single paired quiet-host session):** a ~130-line disposable
  `DirectDecimal128Reader` scratch patch to `parquet 60.1.0` (pre-decoding the
  dictionary once into `Vec<i128>` and gathering `dict[key]` directly) cut
  `morsel_no_pushdown` warm exec-only on `q6_count` from **`35.8 ms` →
  `16.7 ms`** (**`−19.1 ms`, `2.14×` faster — reaching parity with DuckDB's
  `17.2 ms` warm exec-only!**) and fresh total from `45.5 ms` → **`25.8 ms`**
  (**`−19.7 ms`**, at parity with DuckDB's `25.5 ms` fresh). On `q6_scan`, it
  cut warm exec-only from **`76.0 ms` → `50.2 ms`** (**`−25.8 ms`, `1.51×`
  faster** — still **`+1.6 ms`** behind DuckDB's `48.6 ms` warm exec-only, while
  `direct_flba_stock` reaches `47.9 ms` exec-only / `64.8 ms` fresh, at parity
  with DuckDB's `48.6 ms` / `64.9 ms`).
- **The remaining gaps are clearly separated and quantified:**
  - **Fresh-process registration + planning overhead** accounts for **`7.3 ms`**
    in DataFusion (`4.6 ms` Parquet footer reads / schema inference + `2.7 ms`
    first-query planning), compared to DuckDB's `~8.3 ms` total fresh-to-warm
    exec delta on `q6_count` (`25.5 − 17.2 ms`, including footer reads,
    planning, buffer-pool warmup, and Arrow bridge).
  - **Zstd decompression is NOT the bottleneck on `q6_count` (`2.9%` of
    cycles)**, because `l_shipdate`, `l_discount`, and `l_quantity` are
    low-ratio dictionary index pages (`1.00–1.14×`). On `q6_scan`, Zstd
    decompression of `l_extendedprice` (31.2 MB compressed, `1.65×`) +
    `l_orderkey` (9.6 MB) is `25.8%` of cycles (~`19 ms` wall), which DuckDB
    must also decompress because 1.90% uniformly scattered matches touch every
    page.
  - **Post-decode predicate evaluation (`process_work`) is small**: `32.9 ms`
    aggregate across 8 workers (`4.1 ms` wall, `11.4%` of cycles) on `q6_count`
    and `43.2 ms` (`5.4 ms` wall, `5.7%` of cycles) on `q6_scan`. DataFusion
    emits zero casts on `Decimal128` or `Date32`. Naive Arrow `filter()`-based
    conjunct short-circuiting on already-decoded batches actually _increased_
    `process_work` (`33.3 → 38.1 ms`) because materializing 15% of `i128` rows
    costs more than scanning the L2-resident `Decimal128Array`.
  - **`pushdown_filters=true` (`RowFilter`) regresses (`+13.7 ms` on `q6_count`,
    `+34.5 ms` on `q6_scan`)** for two distinct reasons: (a)
    `reorder_filters=true` evaluates smallest-bytes-first (`l_discount` →
    `l_quantity` → `l_shipdate`) instead of SQL order (`l_shipdate` first),
    costing `+4.3 ms` on `q6_count` and `+5.4 ms` on `q6_scan` (`pushdown_base`
    vs. existing config `pushdown_noreorder`), and (b)
    `prebuild_row_filter_candidates` splits `l_discount BETWEEN 0.05 AND 0.07`
    and `l_shipdate >= ... AND < ...` into **5 separate `ArrowPredicate`s**,
    decoding `l_shipdate` 2× and `l_discount` 2–3× while burning `15.3%` of
    `q6_count` cycles in `RowSelection::and_then`. Merging same-column conjuncts
    alone (`pushdown_base → pushdown_grouped`) saves **`−5.8 ms`** on `q6_count`
    and **`−2.9 ms`** on `q6_scan` (`−5.1 ms` / `−1.7 ms` incremental on top of
    `reorder_filters=false`, for **`−9.4 ms` / `−7.1 ms`** combined).
  - **Smaller batch sizes (`2048`) hurt DataFusion despite better L2 locality:**
    hardware counters confirm `2048` cuts L2D cache refills by **33%**
    (`1,626K → 1,091K`), but quadrupling batches adds **`+97.6M` instructions**
    and **`+36.0M` cycles** of per-batch Arrow allocation / `PhysicalExpr` /
    `yield_now()` overhead, making `q6_count` **`+3.8 ms` slower** (`39.9` vs
    `36.1 ms`) and `q6_scan` **`+3.3 ms` slower** (`80.1` vs `76.8 ms`).

## Measurement Conditions & Dataset Layout

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`, 16
  ARM Neoverse-N1 vCPUs (1 thread/core, 64 KiB L1d / 1 MiB L2 per core, 32 MiB
  shared L3), 61 GiB RAM.
- **CPU pinning & threads:** All measured runs pinned via `taskset -c 0-7`
  (`--threads 8`, Tokio `worker_threads=8`, `max_blocking_threads=8`, DuckDB
  `SET threads = 8`). Quiet-host `vmstat -y 1 3` checks before/after measured
  campaigns showed 87–99% idle CPU (no concurrent builds).
- **Revision & toolchain:** `morsel-execution` HEAD
  `9c3079473` (`release-nonlto`, `--locked --no-default-features --features
parquet,sql`), Rust `1.99.1`, `arrow`/`parquet` `60.1.0`, Tokio `1.54.1`,
  Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`, unprivileged
  `perf` `6.8.12` (Arm PMUv3 `r17:u` = `l2d_cache_refill`).
- **Warm page cache:** Standard `#165` SF1 `lineitem` Parquet dataset (16 files,
  6,001,215 rows, 48 row groups of 131,072 rows, 177,150,918 bytes, Zstd level 3,
  dictionary enabled) generated deterministically via
  `benchmarks/morsel_tpch/harness.py generate`. Every timed trial verified
  **43,256 / 43,256 (100%)** resident 4 KiB pages via `mincore` before and after
  execution. Every variant passed exact lossless Arrow IPC result parity against
  DuckDB 2.0.1 (`q6_count`: `n = 114,160`; `q6_scan`: exact multiset equality
  over all 114,160 rows, 1.90% selectivity).

### SF1 `lineitem` Parquet Column Anatomy (All 48 Row Groups)

| Column             | Arrow type         | Parquet physical type     | Dict RGs | Compressed bytes | Uncompressed bytes | Ratio |
| ------------------ | ------------------ | ------------------------- | -------: | ---------------: | -----------------: | ----: |
| `l_discount`       | `Decimal128(15,2)` | `FIXED_LEN_BYTE_ARRAY(7)` |  48 / 48 |        2,659,690 |          3,040,640 | 1.14× |
| `l_quantity`       | `Decimal128(15,2)` | `FIXED_LEN_BYTE_ARRAY(7)` |  48 / 48 |        4,550,640 |          4,554,208 | 1.00× |
| `l_shipdate`       | `Date32`           | `INT32`                   |  48 / 48 |        9,299,004 |          9,518,868 | 1.02× |
| `l_orderkey`       | `Int64`            | `INT64`                   |  48 / 48 |        9,569,763 |         22,827,196 | 2.39× |
| `l_extendedprice`  | `Decimal128(15,2)` | `FIXED_LEN_BYTE_ARRAY(7)` |  48 / 48 |       31,173,408 |         51,342,879 | 1.65× |
| **`q6_count` (3)** | —                  | —                         |        — |   **16,509,334** |     **17,113,716** | 1.04× |
| **`q6_scan` (5)**  | —                  | —                         |        — |   **57,252,505** |     **91,283,791** | 1.59× |

Note the dictionary sizes in RG 0 (131,072 rows): `l_discount` has a **61-byte**
dictionary page (11 values, 0.00–0.10), `l_quantity` has a **225-byte**
dictionary page (50 values, 1–50), and `l_shipdate` has a **5,463-byte**
dictionary page (~2,526 dates).

## 1. Post-`#166` Baseline: Fresh Process & Warm In-Session (SF1, 8 Threads)

### Fresh-Process End-to-End Timings (`metrics=off`, 7 repeats after 1 warmup)

DataFusion starts the clock before `register_parquet` (`reg` + `plan` + `exec` =
`total`); DuckDB creates the lazy `read_parquet` view before timing and drains
`to_arrow_reader(batch_size=8192)`, matching `#165`. Medians in ms (min–max).

| Query      | Arm                         |   Total ms (min–max) |  Reg ms | Plan ms |    Exec ms (min–max) |    CPU ms | Batches |
| ---------- | --------------------------- | -------------------: | ------: | ------: | -------------------: | --------: | ------: |
| `q6_count` | `stock_no_pushdown`         |     47.9 (44.5–57.4) |     4.7 |     3.4 |     39.8 (36.9–49.3) |     265.5 |       1 |
| `q6_count` | **`morsel_no_pushdown`**    | **46.4 (45.2–49.5)** | **4.6** | **2.7** | **39.5 (37.7–42.1)** | **281.1** |   **1** |
| `q6_count` | `morsel_pruning`            |     47.8 (44.9–52.3) |     4.6 |     3.8 |     39.4 (36.6–43.6) |     278.9 |       1 |
| `q6_count` | `morsel_pushdown`           |     61.6 (57.6–64.9) |     4.6 |     3.8 |     53.2 (49.5–55.7) |     341.3 |       1 |
| `q6_count` | `morsel_pushdown_noreorder` |     59.9 (57.8–62.7) |     4.8 |     3.8 |     51.0 (49.0–54.1) |     331.1 |       1 |
| `q6_count` | `stock_default`             |     56.5 (54.5–57.8) |     4.8 |     4.7 |     46.9 (45.1–48.4) |     272.0 |       1 |
| `q6_count` | `stock_pushdown`            |     69.1 (67.2–72.3) |     4.7 |     4.6 |     59.9 (58.4–63.0) |     336.2 |       1 |
| `q6_count` | **`duckdb_default`**        | **25.1 (23.3–26.1)** |       — |       — |                    — | **145.2** |   **1** |
| `q6_scan`  | `stock_no_pushdown`         |     89.8 (87.2–93.7) |     4.5 |     3.2 |     81.5 (79.5–85.4) |     571.4 |      16 |
| `q6_scan`  | **`morsel_no_pushdown`**    | **94.0 (89.7–97.2)** | **4.6** | **2.5** | **86.8 (82.7–89.9)** | **599.0** |  **16** |
| `q6_scan`  | `morsel_pruning`            |    94.3 (90.7–109.0) |     4.6 |     3.6 |    86.3 (82.5–100.6) |     603.1 |      16 |
| `q6_scan`  | `morsel_pushdown`           |  129.1 (126.9–132.7) |     4.6 |     3.6 |  121.3 (118.7–124.5) |     787.6 |      17 |
| `q6_scan`  | `morsel_pushdown_noreorder` |  127.3 (125.3–133.1) |     4.6 |     3.6 |  119.1 (117.5–124.8) |     774.8 |      18 |
| `q6_scan`  | `stock_default`             |  107.8 (105.5–109.7) |     4.5 |     4.4 |    98.8 (96.5–101.0) |     580.5 |      18 |
| `q6_scan`  | `stock_pushdown`            |  149.6 (145.2–151.4) |     4.7 |     4.2 |  140.5 (136.4–142.4) |     793.0 |      48 |
| `q6_scan`  | **`duckdb_default`**        | **68.3 (62.4–71.0)** |       — |       — |                    — | **442.2** |  **48** |

### Warm In-Session Execution (Excluding Table Registration, 7 repeats after 2 warmups)

DuckDB's JSON profile `query.total_time` includes `planner + optimizer +
physical_planner` (~`1.7 ms`). Therefore we compute DuckDB **warm exec-only** as
`total_time − (planner + optimizer + physical_planner)` (median across 7
profiled iterations) so DataFusion `exec-only` compares apples-to-apples against
DuckDB `exec-only`, and DataFusion `plan+exec` compares apples-to-apples against
DuckDB `to_arrow_reader` wall time (which includes DuckDB planning + execution +
Arrow C-stream export).

| Query      | Arm                          | Warm plan ms | Warm exec-only ms | Warm plan+exec ms | Warm CPU ms | Notes                                                                                          |
| ---------- | ---------------------------- | -----------: | ----------------: | ----------------: | ----------: | ---------------------------------------------------------------------------------------------- |
| `q6_count` | `stock_no_pushdown`          |          1.6 |              36.8 |              38.4 |       268.3 | Same `SessionContext`, re-planned each iteration                                               |
| `q6_count` | **`morsel_no_pushdown`**     |      **1.2** |          **35.6** |          **36.8** |   **263.3** | `exec_only` loop: `35.5 ms`                                                                    |
| `q6_count` | `morsel_pruning`             |          1.7 |              36.8 |              38.5 |       277.3 | `statistics_eval_time` ~2.2 ms aggregate, 0 pruned                                             |
| `q6_count` | `morsel_pushdown`            |          1.7 |              50.2 |              51.9 |       325.9 | `exec_only` loop: `48.3 ms`                                                                    |
| `q6_count` | `morsel_pushdown_noreorder`  |          1.7 |              44.5 |              46.2 |       298.2 | `exec_only` loop: `45.0 ms`                                                                    |
| `q6_count` | `stock_default`              |          2.2 |              45.9 |              48.1 |           — | Byte-range splitting adds ~9 ms vs `no_pushdown`                                               |
| `q6_count` | `stock_pushdown`             |          2.2 |              57.8 |              59.9 |           — | `RowFilter` + byte-range split                                                                 |
| `q6_count` | **`duckdb_default` (Arrow)** |      **1.7** |          **17.2** |          **20.6** |   **126.6** | Profile `total_time = 18.9 ms` (`plan 1.7 + exec 17.2`); `fetchall = 19.4 ms`; Arrow `20.6 ms` |
| `q6_scan`  | `stock_no_pushdown`          |          1.4 |              76.4 |              77.8 |       553.1 | Same `SessionContext`, re-planned each iteration                                               |
| `q6_scan`  | **`morsel_no_pushdown`**     |      **1.1** |          **75.4** |          **76.5** |   **565.3** | `exec_only` loop: `76.9 ms`                                                                    |
| `q6_scan`  | `morsel_pruning`             |          1.6 |              79.7 |              81.3 |       580.3 | 0 / 48 row groups pruned                                                                       |
| `q6_scan`  | `morsel_pushdown`            |          1.6 |             109.6 |             111.2 |       755.8 | `exec_only` loop: `112.2 ms`                                                                   |
| `q6_scan`  | `morsel_pushdown_noreorder`  |          1.6 |             105.7 |             107.4 |       729.0 | `exec_only` loop: `109.0 ms`                                                                   |
| `q6_scan`  | `stock_default`              |          2.0 |              92.0 |              94.0 |           — | Byte-range splitting overhead                                                                  |
| `q6_scan`  | `stock_pushdown`             |          2.0 |             129.4 |             131.3 |           — | `RowFilter` + byte-range split                                                                 |
| `q6_scan`  | **`duckdb_default` (Arrow)** |      **1.7** |          **48.6** |          **52.7** |   **369.5** | Profile `total_time = 50.3 ms` (`plan 1.7 + exec 48.6`); Arrow drain `52.7 ms`                 |

### `#138` Morsel Per-Run Metrics & DuckDB Operator Profile

All `#138` durations are sums across the 8 workers (median of 7 runs in ms);
`ready/out` are max observed queue depths; `claims/worker` is from the median
run.

| Query      | Arm                         | Total |  Exec | Pending I/O | Ready wait | Decode work | Process work | Output wait | Peak rdy/out | Claims/worker     |
| ---------- | --------------------------- | ----: | ----: | ----------: | ---------: | ----------: | -----------: | ----------: | -----------: | ----------------- |
| `q6_count` | `morsel_no_pushdown`        |  48.9 |  41.9 |        22.4 |        8.1 |   **254.4** |     **35.5** |         0.0 |        8 / 1 | `4/7/7/7/7/7/6/3` |
| `q6_count` | `morsel_no_pushdown` (warm) |     — |  36.1 |        20.8 |        6.5 |   **222.5** |     **32.9** |         0.0 |        8 / 1 | balanced (48 RGs) |
| `q6_count` | `morsel_pruning`            |  49.5 |  41.3 |        32.0 |       10.4 |       243.2 |         33.0 |         0.0 |        8 / 1 | `6/7/6/5/7/6/6/5` |
| `q6_count` | `morsel_pushdown`           |  60.4 |  52.6 |        68.0 |       37.7 |   **319.2** |          0.0 |         0.0 |        4 / 1 | `6/8/7/6/4/5/5/7` |
| `q6_count` | `morsel_pushdown_noreorder` |  60.1 |  51.6 |        58.2 |       38.9 |       311.3 |          0.0 |         0.0 |        6 / 1 | `7/4/5/8/6/6/7/5` |
| `q6_scan`  | `morsel_no_pushdown`        |  92.9 |  85.6 |       122.8 |       15.7 |   **543.1** |     **49.6** |         0.0 |        8 / 6 | `6/6/6/6/6/6/5/7` |
| `q6_scan`  | `morsel_no_pushdown` (warm) |     — |  76.8 |        62.8 |       16.5 |   **480.2** |     **43.2** |         0.0 |        8 / 6 | balanced (48 RGs) |
| `q6_scan`  | `morsel_pushdown`           | 127.6 | 119.4 |       119.5 |       36.2 |   **771.1** |          3.8 |         0.0 |        6 / 4 | `4/7/5/6/7/6/7/6` |
| `q6_scan`  | `morsel_pushdown_noreorder` | 123.9 | 115.8 |        84.1 |       43.0 |       729.4 |          1.7 |         0.0 |        5 / 5 | `7/7/9/4/6/4/5/6` |

**Key takeaway from `#138`:** `#148` completely eliminated the old producer
starvation bottleneck — `ready_wait` on `morsel_no_pushdown` is now only **0.8 ms
per worker** (`6.5 / 8`) on `q6_count` and **2.1 ms per worker** (`16.5 / 8`) on
`q6_scan`, and `output_wait` is **0.0 ms**. **`decode_work` accounts for 87.1%
of active worker time on `q6_count` and 91.7% on `q6_scan`.**

**DuckDB 2.0.1 internal operator profile (`PRAGMA enable_profiling = 'json'`,
7-repeat median):**

- **`q6_count`** (`total_time = 18.92 ms`, `plan = 1.68 ms`,
  **`exec_only = 17.24 ms`** (`16.45–17.40 ms`), `cpu_time = 122.81 ms`,
  `blocked_thread_time = 0.60 ms`, `bytes_scanned = 16,509,334`):
  - `planner` (`1.12 ms`) + `optimizer` (`0.52 ms`) + `physical_planner`
    (`0.04 ms`) = **`1.68 ms`**
  - `READ_PARQUET` (with all 3 filters pushed): **`121.16 ms` CPU** (160 read
    ops, `5.63 ms` read time, 48/48 RGs)
  - `PROJECTION` (constant `42`): `0.95 ms` CPU; `UNGROUPED_AGGREGATE`:
    `0.70 ms` CPU
- **`q6_scan`** (`total_time = 50.33 ms`, `plan = 1.72 ms`,
  **`exec_only = 48.60 ms`** (`47.74–50.01 ms`), `cpu_time = 364.30 ms`,
  `blocked_thread_time = 1.34 ms`, `bytes_scanned = 57,252,505`):
  - `planner` (`1.17 ms`) + `optimizer` (`0.54 ms`) + `physical_planner`
    (`0.02 ms`) = **`1.72 ms`**
  - `READ_PARQUET` (projects `l_orderkey, l_extendedprice, l_discount`; all 3
    filters pushed): **`353.49 ms` CPU** (158 read ops, `28.55 ms` read time)
  - `PROJECTION` (`l_orderkey, l_extendedprice * l_discount`): `5.54 ms` CPU;
    `RESULT_COLLECTOR`: `5.26 ms` CPU

## 2. Millisecond Waterfall: Decomposing the Gap to DuckDB

To avoid mixing baselines across runs, **every figure in both waterfalls below
comes from one single paired quiet-host session** (`sf1_flba_paired.jsonl` +
`sf1_duckdb_fresh_paired.jsonl` + `sf1_duckdb_warm_paired.jsonl`, 7 repeats
each, `taskset -c 0-7`, 92–99% host idle). Every row is an exact measured delta
between consecutive milestones (`B − A`), with zero fudge factors.

### Waterfall A: `q6_count` (`45.5 ms` Fresh / `35.8 ms` Warm Exec vs. DuckDB `25.5 ms` Fresh / `17.2 ms` Warm Exec-Only)

| Step / Derived Formula                                                           | Fresh total (ms) | Warm exec-only (ms) | Derivation / Evidence                                                                                |
| -------------------------------------------------------------------------------- | ---------------: | ------------------: | ---------------------------------------------------------------------------------------------------- |
| **Start: `baseline_flba` (`morsel_no_pushdown` at HEAD `9c3079473`)**            |      **45.5 ms** |         **35.8 ms** | Paired 7-repeat median (`263.5 ms` CPU; `decode = 213.9 ms`, `process = 32.6 ms`)                    |
| 1. `FLBA(7) → Decimal128` 2-pass decode (`direct_flba_dec128 − baseline_flba`)   |     **−19.7 ms** |        **−19.1 ms** | `25.8 − 45.5 = −19.7 ms` fresh; `16.7 − 35.8 = −19.1 ms` warm (`70.2%` of cycles in `perf record`)   |
| _Subtotal: `direct_flba_dec128` (`morsel_no_pushdown` with direct FLBA decode)_  |        _25.8 ms_ |           _16.7 ms_ | _`117.5 ms` CPU (`decode = 67.4 ms`, `process = 32.7 ms`)_                                           |
| 2. Fresh startup delta: `DF(fresh − warm_exec) − DuckDB(fresh − warm_exec_only)` |      **−0.8 ms** |                   — | `(25.8 − 16.7) − (25.5 − 17.2) = 9.1 − 8.3 = −0.8 ms` (DataFusion `reg+plan+cold` vs DuckDB 1st-run) |
| 3. Post-FLBA warm exec delta: `DuckDB exec_only − direct_flba_dec128`            |      **+0.5 ms** |         **+0.5 ms** | `17.2 − 16.7 = +0.5 ms` (within parity: `15.7–17.9 ms` vs DuckDB `16.5–17.4 ms`)                     |
| **Target: DuckDB 2.0.1 (`duckdb_default`)**                                      |      **25.5 ms** |         **17.2 ms** | Fresh Arrow `25.5 ms`; warm `exec_only = 17.2 ms` (`total 18.9 − plan 1.7`; warm Arrow `20.6 ms`)    |

_Exact sum check:_ Fresh `45.5 − 19.7 − 0.8 + 0.5 = 25.5 ms`; Warm exec-only
`35.8 − 19.1 + 0.5 = 17.2 ms`. Once `FLBA(7) → Decimal128` is fixed,
`morsel_no_pushdown` is at parity with DuckDB on `q6_count` (`16.7 ms` vs.
`17.2 ms` warm exec-only; `25.8 ms` vs. `25.5 ms` fresh total).

### Waterfall B: `q6_scan` (`91.3 ms` Fresh / `76.0 ms` Warm Exec vs. DuckDB `64.9 ms` Fresh / `48.6 ms` Warm Exec-Only)

| Step / Derived Formula                                                               | Fresh total (ms) | Warm exec-only (ms) | Derivation / Evidence                                                                                                             |
| ------------------------------------------------------------------------------------ | ---------------: | ------------------: | --------------------------------------------------------------------------------------------------------------------------------- |
| **Start: `baseline_flba` (`morsel_no_pushdown` at HEAD `9c3079473`)**                |      **91.3 ms** |         **76.0 ms** | Paired 7-repeat median (`563.2 ms` CPU; `decode = 485.8 ms`, `process = 43.7 ms`)                                                 |
| 1. `FLBA(7) → Decimal128` 2-pass decode (`direct_flba_dec128 − baseline_flba`)       |     **−22.9 ms** |        **−25.8 ms** | `68.4 − 91.3 = −22.9 ms` fresh; `50.2 − 76.0 = −25.8 ms` warm (`54.1%` of cycles in `perf record`)                                |
| _Subtotal: `direct_flba_dec128` (`morsel_no_pushdown` with direct FLBA decode)_      |        _68.4 ms_ |           _50.2 ms_ | _`369.6 ms` CPU (`decode = 298.9 ms`, `process = 42.5 ms`); still `+1.6 ms` behind DuckDB exec-only_                              |
| 2. Morsel-vs-Stock `q6_scan` gap (`direct_flba_stock − direct_flba_dec128`)          |      **−3.6 ms** |         **−2.3 ms** | `64.8 − 68.4 = −3.6 ms` fresh; `47.9 − 50.2 = −2.3 ms` warm (`347.2` vs `369.6 ms` CPU: `ready_wait` + `ResultBuffer` coalescing) |
| _Subtotal: `direct_flba_stock` (`stock_no_pushdown` with direct FLBA decode)_        |        _64.8 ms_ |           _47.9 ms_ | _`347.2 ms` CPU; at parity with DuckDB (`48.6 ms` warm exec-only / `64.9 ms` fresh)_                                              |
| 3. Fresh startup delta: `Stock(fresh − warm_exec) − DuckDB(fresh − warm_exec_only)`  |      **−0.6 ms** |                   — | `(64.8 − 47.9) − (64.9 − 48.6) = 16.9 − 16.3 = −0.6 ms`                                                                           |
| 4. Post-FLBA Stock vs DuckDB exec-only delta: `DuckDB exec_only − direct_flba_stock` |      **+0.7 ms** |         **+0.7 ms** | `48.6 − 47.9 = +0.7 ms` (within parity: `46.7–51.3 ms` vs DuckDB `47.7–50.0 ms`)                                                  |
| **Target: DuckDB 2.0.1 (`duckdb_default`)**                                          |      **64.9 ms** |         **48.6 ms** | Fresh Arrow `64.9 ms`; warm `exec_only = 48.6 ms` (`total 50.3 − plan 1.7`; warm Arrow `52.7 ms`)                                 |

_Exact sum check:_ Fresh `91.3 − 22.9 − 3.6 − 0.6 + 0.7 = 64.9 ms`; Warm
exec-only `76.0 − 25.8 − 2.3 + 0.7 = 48.6 ms`.

## 3. Deep-Dive Evidence Across All 5 Investigation Axes

### Axis 1: Hardware Counters, `perf record` Leaf Symbols & SQL Micro-Probes

#### Arm PMUv3 Steady-State Hardware Counters (per iteration, startup subtracted, 8 threads)

| Query      | Arm                  |  Exec ms | Cycles (M) | Instructions (M) |  IPC | L2D refills (K) | Task-clock ms |
| ---------- | -------------------- | -------: | ---------: | ---------------: | ---: | --------------: | ------------: |
| `q6_count` | `stock_no_pushdown`  |     38.5 |      640.3 |          1,409.7 | 2.20 |         1,600.7 |         276.2 |
| `q6_count` | `morsel_no_pushdown` |     36.7 |      614.1 |          1,405.2 | 2.29 |         1,602.6 |         270.2 |
| `q6_count` | `morsel_pushdown`    |     52.4 |      791.1 |          1,619.1 | 2.05 |         2,010.5 |         356.3 |
| `q6_count` | `stock_pushdown`     |     60.4 |      801.2 |          1,631.7 | 2.04 |         2,179.3 |         360.1 |
| `q6_count` | **`duckdb_default`** | **22.4** |  **274.7** |        **476.5** | 1.73 |     **1,296.7** |     **140.5** |
| `q6_scan`  | `stock_no_pushdown`  |     76.4 |    1,300.7 |          2,982.8 | 2.29 |         4,349.1 |         573.5 |
| `q6_scan`  | `morsel_no_pushdown` |     79.4 |    1,317.6 |          2,979.2 | 2.26 |         4,448.3 |         590.0 |
| `q6_scan`  | `morsel_pushdown`    |    118.8 |    1,490.0 |          3,206.5 | 2.15 |         5,065.1 |         795.4 |
| `q6_scan`  | `stock_pushdown`     |    133.5 |    1,449.9 |          3,217.9 | 2.22 |         4,887.5 |         787.2 |
| `q6_scan`  | **`duckdb_default`** | **55.6** |  **833.6** |      **1,989.8** | 2.39 |     **3,113.3** |     **386.7** |

Morsel DataFusion executes **2.95× more instructions** (`1,405.2M` vs. `476.5M`,
or `234` vs. `79` instructions/row) on `q6_count` despite **higher IPC** (`2.29`
vs. `1.73`), proving the gap is excess instructions rather than stalls or lock
contention.

#### `perf record` Cycle Attribution (3,999 Hz `cycles:u`, 31K–57K samples, 0 lost samples)

| Category / Top Demangled Symbol                                                             | `q6_count` `morsel_no_pd` | `q6_count` `morsel_pd` | `q6_scan` `morsel_no_pd` | `q6_scan` `morsel_pd` |
| ------------------------------------------------------------------------------------------- | ------------------------: | ---------------------: | -----------------------: | --------------------: |
| `Decimal128Array::from_unary(&FixedSizeBinaryArray, sign_extend_be)` (`SpecFromIter<i128>`) |                **34.35%** |                 27.53% |               **24.77%** |                22.12% |
| `fixed_len_byte_array::ValueDecoder::read_values`                                           |                **17.25%** |                 13.81% |               **12.00%** |                10.44% |
| `__memcpy_generic` + `memcpy@plt` (7-byte copies from FLBA dict)                            |                **18.58%** |                 15.08% |               **17.33%** |                15.57% |
| **Subtotal: `FLBA(7) → Decimal128` 2-pass decode**                                          |                **70.18%** |             **56.42%** |               **54.10%** |            **48.13%** |
| Zstd decompression (`ZSTD_decompressSequences_default`, `HUF_decompress4X*`, `FSE_*`)       |                     2.90% |                  2.30% |               **25.80%** |                22.60% |
| `RowSelection::and_then` + `RowSelectionPolicy::resolve` + `filter_native`                  |                     0.00% |             **20.55%** |                    0.00% |            **10.61%** |
| `arrow_ord::cmp::apply_op<&[i128]>` (`Decimal128` comparisons)                              |                     7.43% |                  3.70% |                    3.61% |                 2.00% |
| `arrow_ord::cmp::apply_op<&[i32]>` (`Date32` comparisons)                                   |                     1.79% |                  0.50% |                    0.96% |                 0.30% |
| Bit-unpack / RLE (`RleDecoder::get_batch_with_dict`, `BitReader`, `unpack32`)               |                     5.50% |                  3.50% |                    4.40% |                 3.20% |
| Allocator (`malloc`, `_int_malloc`, `cfree`, `__memset_zva64`)                              |                     1.78% |                  3.30% |                    1.70% |                 2.50% |
| Tokio / synchronization / other                                                             |                    10.42% |                  9.73% |                    9.43% |                10.66% |

In **DuckDB 2.0.1's `perf record`** on `q6_count`, the #1 DuckDB symbol is
**`duckdb::DictionaryDecoder::Filter` (`13.48%` of samples)** alongside
`__fastunpack12` (`1.75%`) and `__fastunpack4` (`0.84%`): DuckDB evaluates
range filters directly against the Parquet dictionary page and unpacks matching
dictionary indices into a `SelectionVector` without expanding `FLBA(7)` to
16-byte values per row.

#### Isolated Column & Predicate SQL Micro-Probes (Warm Exec ms / CPU ms, 8 Threads)

| Probe (`lineitem`, 6,001,215 rows)    |  `morsel_no_pushdown` (dec / proc ms) |    `morsel_pushdown` |    `duckdb_default` | DF / DuckDB CPU |
| ------------------------------------- | ------------------------------------: | -------------------: | ------------------: | --------------: |
| `p1_shipdate_only` (`INT32`, 9.3 MB)  |     **5.0 ms** (31.4 cpu; 12.7 / 7.2) |    8.3 ms (58.1 cpu) |  12.2 ms (61.0 cpu) |       **0.52×** |
| `p2_discount_only` (`FLBA7`, 2.7 MB)  | **18.2 ms** (129.3 cpu; 104.5 / 16.2) |  23.4 ms (164.2 cpu) |  12.3 ms (63.1 cpu) |       **2.05×** |
| `p3_quantity_only` (`FLBA7`, 4.6 MB)  |  **16.2 ms** (117.5 cpu; 100.6 / 8.1) |  17.0 ms (116.9 cpu) |  12.4 ms (70.5 cpu) |       **1.67×** |
| `p4_shipdate_discount`                |     22.0 ms (156.1 cpu; 118.6 / 23.7) |  31.9 ms (212.0 cpu) |  16.9 ms (96.1 cpu) |           1.62× |
| `p5_q6_count` (all 3 filter cols)     |     38.8 ms (271.4 cpu; 221.1 / 32.6) |  49.2 ms (325.2 cpu) | 20.7 ms (126.3 cpu) |           2.15× |
| `p10_qty_first_order` (SQL reorder)   |     36.7 ms (272.8 cpu; 223.3 / 33.0) |  50.9 ms (336.3 cpu) | 20.9 ms (126.2 cpu) |           2.16× |
| `p8_orderkey_only` (`INT64`, 9.6 MB)  |     **9.2 ms** (64.5 cpu; 49.2 / 3.8) |   1.0 ms\* (6.3 cpu) |  15.2 ms (90.2 cpu) |       **0.72×** |
| `p9_extprice_only` (`FLBA7`, 31.2 MB) |      33.8 ms (240.3 cpu; 209.8 / 8.8) |   1.0 ms\* (6.3 cpu) | 33.0 ms (223.0 cpu) |           1.08× |
| `p6_q6_scan_no_mul` (no `×`)          |     78.8 ms (574.0 cpu; 499.8 / 41.6) | 110.2 ms (743.5 cpu) | 52.4 ms (368.2 cpu) |           1.56× |
| `p7_q6_scan` (with `×`)               |     76.1 ms (566.8 cpu; 487.2 / 44.0) | 110.8 ms (746.7 cpu) | 53.2 ms (370.0 cpu) |           1.53× |

_\*`p8_orderkey_only` and `p9_extprice_only` have `min > 0` in every row group,
so `morsel_pushdown` (which enables statistics pruning) satisfies `col > 0`
directly via metadata without reading data pages. Note DuckDB probe column shows
`to_arrow_reader` wall time (including ~1.7 ms planning)._

Look at what these probes prove:

1. **Morsel DataFusion is ALREADY faster than DuckDB on `l_shipdate` (`INT32`:
   `5.0 ms` vs. `12.2 ms`) and `l_orderkey` (`INT64`: `9.2 ms` vs. `15.2 ms`)!**
2. **Despite `l_discount` being 3.5× smaller on disk than `l_shipdate` (2.66 MB
   vs. 9.30 MB), `arrow-rs` burns `8.2×` more decode CPU on `l_discount`
   (`104.5 ms` vs. `12.7 ms`)** purely due to `FixedLenByteArrayReader`'s
   7-byte `memcpy` + `from_unary` loop.
3. **Pure additivity vs. DuckDB short-circuiting:** In `morsel_no_pushdown`,
   `p1 + p2 + p3` decode (`12.7 + 104.5 + 100.6 = 217.8 ms`) equals `p5_q6_count`
   (`221.1 ms`) — all 3 columns are 100% decoded and evaluated. In DuckDB,
   `p1 + p2 + p3 = 194.6 ms` CPU drops to `126.3 ms` on `p5_q6_count` (−35% CPU)
   via dictionary-level filter evaluation and `SelectionVector` pruning.
4. **Decimal128 multiply (`l_extendedprice * l_discount`) is negligible:**
   `p6_q6_scan_no_mul` (`78.8 ms`) vs. `p7_q6_scan` (`76.1 ms`) shows `< 1 ms`
   cost on the 114,160 surviving rows.
5. **Proof of `RowFilter` duplicate column decoding:** `p3_quantity_only` (1
   conjunct) has **zero** `RowFilter` overhead (`116.9` vs. `117.5 ms` CPU),
   whereas `p1_shipdate_only` (2 conjuncts on 1 column) nearly **doubles CPU**
   (`31.4 → 58.1 ms`) and `p2_discount_only` (`BETWEEN`, 2 conjuncts) adds
   `+34.9 ms` CPU (`129.3 → 164.2 ms`).

#### Direct Validation: `DirectDecimal128Reader` & `INT64`-Decimal Diagnostic Control

We validated the `FLBA(7) → Decimal128` diagnosis two independent ways (all
pinned `taskset -c 0-7`, 8 threads, 7 repeats, exact DuckDB parity verified):

1. **Direct scratch fix on the exact, unmodified `#165` FLBA(7) Parquet files**
   (`parquet_flba_decimal_scratch.patch` in a disposable worktree, paired in the
   same quiet session with DuckDB fresh + warm 7-repeat profile; this is the
   single dataset backing Waterfalls A and B):

| Query      | Variant (`#165` FLBA(7) files)   | Warm exec-only ms (min–max) | Warm CPU ms | Decode ms | Process ms | Fresh total ms |
| ---------- | -------------------------------- | --------------------------: | ----------: | --------: | ---------: | -------------: |
| `q6_count` | `baseline_flba` (Morsel)         |            35.8 (33.8–39.3) |       263.5 |     213.9 |       32.6 |           45.5 |
| `q6_count` | **`direct_flba_dec128`**         |        **16.7 (15.7–17.9)** |   **117.5** |  **67.4** |       32.7 |       **25.8** |
| `q6_count` | `baseline_pushdown` (Morsel)     |            50.7 (49.5–55.7) |       338.4 |     313.5 |        0.0 |           60.1 |
| `q6_count` | `direct_flba_pushdown`           |            30.9 (29.7–32.9) |       202.7 |     177.5 |        0.0 |           42.3 |
| `q6_count` | `direct_flba_stock`              |            17.0 (16.5–18.2) |       112.2 |         — |          — |           26.1 |
| `q6_count` | _DuckDB 2.0.1 (exec-only)_       |          _17.2 (16.5–17.4)_ |     _122.8_ |         — |          — |         _25.5_ |
| `q6_count` | _DuckDB 2.0.1 (plan+exec Arrow)_ |          _20.6 (20.2–21.2)_ |     _126.6_ |         — |          — |         _25.5_ |
| `q6_scan`  | `baseline_flba` (Morsel)         |            76.0 (74.9–80.9) |       563.2 |     485.8 |       43.7 |           91.3 |
| `q6_scan`  | **`direct_flba_dec128`**         |        **50.2 (47.9–51.6)** |   **369.6** | **298.9** |       42.5 |       **68.4** |
| `q6_scan`  | `baseline_pushdown` (Morsel)     |         109.9 (107.0–114.0) |       748.0 |     704.5 |        1.4 |          132.2 |
| `q6_scan`  | `direct_flba_pushdown`           |            85.0 (83.9–89.9) |       560.9 |     509.7 |        1.5 |          106.0 |
| `q6_scan`  | `direct_flba_stock`              |            47.9 (46.7–51.3) |       347.2 |         — |          — |           64.8 |
| `q6_scan`  | _DuckDB 2.0.1 (exec-only)_       |          _48.6 (47.7–50.0)_ |     _363.9_ |         — |          — |         _64.9_ |
| `q6_scan`  | _DuckDB 2.0.1 (plan+exec Arrow)_ |          _52.7 (51.9–53.0)_ |     _369.5_ |         — |          — |         _64.9_ |

2. **Parquet physical-type diagnostic control (`INT64`-stored `Decimal(15,2)`):**
   Re-exporting SF1 `lineitem` via DuckDB `COPY` (see Section 5 for exact
   command) stores `Decimal(15,2)` as Parquet physical `INT64` (identical
   compressed size on `q6_count`: 16.46 MB vs. 16.51 MB). On those files,
   unmodified `baseline_runner` (`morsel_no_pushdown`) ran `q6_count` in
   **`14.2 ms` warm exec** (`99.6 ms` CPU, `decode = 49.5 ms`) vs. DuckDB's
   `18.1 ms` warm Arrow drain (~`16.4 ms` exec-only), independently confirming
   that DataFusion's `PrimitiveArrayReader` dictionary path reaches parity with
   DuckDB once `FixedLenByteArrayReader` is bypassed.

### Axis 2 & Axis 3: Batch-Size Sweep (`2048` / `4096` / `8192` / `16384`)

Warm in-session execution (`morsel_no_pushdown`, 8 threads, `taskset -c 0-7`),
combining `#138` metrics and Arm PMUv3 hardware counters:

| Query      | Batch size |    Exec ms (min–max) |    CPU ms | Decode ms | Process ms |  Cycles (M) | Instructions (M) | L2D refills (K) |
| ---------- | ---------: | -------------------: | --------: | --------: | ---------: | ----------: | ---------------: | --------------: |
| `q6_count` |       2048 |     39.9 (38.6–41.0) |     297.3 |     228.3 |       47.7 |       672.9 |          1,502.8 |     **1,091.1** |
| `q6_count` |       4096 |     37.5 (35.1–39.1) |     269.6 |     214.0 |       37.3 |       653.5 |          1,436.8 |         1,238.5 |
| `q6_count` |   **8192** | **36.1 (35.6–39.2)** | **272.3** | **222.5** |   **32.9** |   **636.9** |      **1,405.2** |         1,626.2 |
| `q6_count` |      16384 |     38.4 (35.8–41.9) |     272.1 |     224.2 |       31.0 |       626.7 |          1,391.4 |         1,891.5 |
| `q6_scan`  |       2048 |     80.1 (79.2–84.7) |     597.4 |     495.9 |       61.6 |     1,361.1 |          3,112.5 |         3,779.0 |
| `q6_scan`  |       4096 |     77.2 (75.5–85.2) |     577.5 |     495.4 |       48.8 |     1,349.4 |          3,021.1 |         4,212.1 |
| `q6_scan`  |   **8192** | **76.8 (73.9–78.1)** | **562.3** | **480.2** |   **43.2** | **1,276.5** |      **2,979.2** |         4,417.2 |
| `q6_scan`  |      16384 |     75.1 (72.7–77.3) |     556.8 |     480.7 |       39.7 |     1,258.8 |          2,960.6 |         4,176.5 |

Shrinking to DuckDB's 2,048-row vector size **does** reduce L2D cache refills by
33% on `q6_count` (`1,626K → 1,091K`), but Arrow/DataFusion allocates fresh
buffers per batch and runs `PhysicalExpr` dispatch + `yield_now()` 4× as often,
adding `+97.6M` instructions and `+14.8 ms` of `process_work` — a net
**`+3.8 ms` regression** on `q6_count` and **`+3.3 ms` regression** on
`q6_scan`. Keeping `8192` is right for DataFusion until buffer reuse is
introduced.

### Axis 4 & Axis 5: Disposable Scratch Ablations (`ablations.patch`)

All eight variants built from `ablations.patch` in a disposable worktree,
verified against DuckDB oracle parity, pinned `taskset -c 0-7`, 7 repeats warm
(with `pushdown_noreorder` included separately to isolate the existing
`reorder_filters=false` config from same-column conjunct grouping):

| Query      | Variant                                       |       Exec ms (min–max) |    CPU ms | Decode ms | Process ms | Ready wait ms | Pending I/O ms |
| ---------- | --------------------------------------------- | ----------------------: | --------: | --------: | ---------: | ------------: | -------------: |
| `q6_count` | `native` (control)                            |        36.8 (36.1–40.3) |     273.7 |     224.0 |       33.3 |           5.9 |           22.7 |
| `q6_count` | `no_batch_yield`                              |        37.5 (36.1–38.1) |     266.5 |     227.7 |       32.8 |           8.3 |       **72.6** |
| `q6_count` | `short_circuit`                               |        37.8 (36.2–40.3) |     278.4 |     223.4 |   **38.1** |           5.8 |           20.1 |
| `q6_count` | `no_yield_and_short_circuit`                  |        38.5 (36.1–39.0) |     275.3 |     225.5 |       37.6 |           8.5 |           84.9 |
| `q6_count` | `pushdown_base` (5 preds, `reorder=true`)     |        50.6 (50.1–51.4) |     332.2 |     307.6 |        0.0 |          41.6 |           56.3 |
| `q6_count` | `pushdown_noreorder` (5 preds, `false`)       |        46.3 (44.3–47.4) |     307.7 |     281.6 |        0.0 |          35.8 |           36.7 |
| `q6_count` | `pushdown_grouped` (3 preds, `true`)          |        44.8 (44.1–46.0) |     292.4 |     266.5 |        0.0 |          36.1 |           52.0 |
| `q6_count` | **`pushdown_grouped_noreorder`** (3, `false`) |    **41.2 (40.5–42.6)** | **278.5** | **252.5** |        0.0 |          28.6 |           30.6 |
| `q6_scan`  | `native` (control)                            |        78.2 (75.7–79.8) |     579.7 |     497.3 |       46.8 |          16.5 |           64.7 |
| `q6_scan`  | `no_batch_yield`                              |        77.5 (74.2–80.7) |     571.0 |     496.6 |       43.8 |          15.3 |      **152.3** |
| `q6_scan`  | `short_circuit`                               |        78.3 (76.7–81.5) |     581.8 |     491.5 |   **53.4** |          18.8 |           60.3 |
| `q6_scan`  | `no_yield_and_short_circuit`                  |        78.7 (75.9–81.9) |     580.1 |     501.6 |       55.3 |          15.8 |          146.1 |
| `q6_scan`  | `pushdown_base` (5 preds, `reorder=true`)     |     111.0 (109.0–112.8) |     750.5 |     707.7 |        1.2 |          37.7 |           85.0 |
| `q6_scan`  | `pushdown_noreorder` (5 preds, `false`)       |     105.6 (102.9–109.1) |     720.8 |     674.8 |        1.2 |          31.9 |           69.7 |
| `q6_scan`  | `pushdown_grouped` (3 preds, `true`)          |     108.1 (102.5–110.8) |     717.8 |     668.9 |        1.3 |          39.7 |           75.8 |
| `q6_scan`  | **`pushdown_grouped_noreorder`** (3, `false`) | **103.9 (100.0–110.2)** | **696.4** | **648.7** |        1.6 |          35.3 |           59.1 |

Findings from the ablations:

- **`no_batch_yield`**: Within noise on wall time (`+0.7 ms` on `q6_count`,
  `−0.7 ms` on `q6_scan`) while tripling `pending_io` latency (`22.7 → 72.6 ms`
  and `64.7 → 152.3 ms`). Do **not** remove cooperative yields.
- **`short_circuit`** (group conjuncts by column, evaluate `l_shipdate` first,
  then `arrow::compute::filter` the next `Decimal128` column to 15.2% / 4.1%
  rows): Increases `process_work` (`33.3 → 38.1 ms` on `q6_count`,
  `46.8 → 53.4 ms` on `q6_scan`) because copying 15% of `i128` rows + scattering
  bits costs more than scanning contiguous `i128` arrays linearly. Short-circuit
  savings on decoded batches have a hard ceiling of ~`2.5 ms` wall (`7.43%`
  `apply_op<&[i128]>`), which requires zero-copy selection-vector / set-bit
  evaluation rather than Arrow `filter()`.
- **Separating `RowFilter` ordering vs. same-column grouping:**
  - Disabling `reorder_filters` alone (`pushdown_base → pushdown_noreorder`,
    already an existing DataFusion config option) saves **`−4.3 ms`** on
    `q6_count` (`50.6 → 46.3 ms`) and **`−5.4 ms`** on `q6_scan`
    (`111.0 → 105.6 ms`) by evaluating `l_shipdate` first instead of
    smallest-bytes-first (`l_discount` → `l_quantity` → `l_shipdate`).
  - Merging same-column conjuncts alone (`pushdown_base → pushdown_grouped`,
    5 → 3 predicates) saves **`−5.8 ms`** (`−39.8 ms` CPU) on `q6_count`
    (`50.6 → 44.8 ms`) and **`−2.9 ms`** (`−32.7 ms` CPU) on `q6_scan`
    (`111.0 → 108.1 ms`). Applied on top of `reorder_filters=false`
    (`pushdown_noreorder → pushdown_grouped_noreorder`), merging saves an
    additional **`−5.1 ms`** on `q6_count` (`46.3 → 41.2 ms`) and **`−1.7 ms`**
    on `q6_scan` (`105.6 → 103.9 ms`) — **`−9.4 ms` / `−7.1 ms`** combined.
  - Even with both improvements, `RowFilter` still lags `native` by `4.4 ms`
    (`q6_count`) and `25.7 ms` (`q6_scan`) because `RowSelection::and_then`
    churn, Phase-2 `ready_wait`, and decoding `l_discount` a second time for
    `q6_scan` output projection still outweigh skipping rows inside
    already-decompressed pages.

## 4. Ranked Top-3 Engineering Follow-Ups

|  Rank | Optimization                                                                                                 |                                                                                                                                                    Expected savings (`q6_count`) |                                                                                                                                                            Expected savings (`q6_scan`) | Scope & notes                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----: | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Direct `FixedLenByteArray` → `Decimal128` (and `Decimal64`/`Decimal32`) dictionary & plain decoding**      |                                                                   **−19 to −20 ms** (warm: `35.8 → 16.7 ms`, parity with DuckDB `17.2 ms`; fresh: `45.5 → 25.8 ms` vs `25.5 ms`) |                                                                          **−23 to −26 ms** (warm: `76.0 → 50.2 ms` vs DuckDB `48.6 ms` exec-only; fresh: `91.3 → 68.4 ms` vs `64.9 ms`) | Upstream `arrow-rs` (`parquet::arrow::array_reader::fixed_len_byte_array`) or custom morsel Parquet decoder: decode FLBA dictionary page once into `Vec<i128>` and gather `dict[key]` directly, eliminating 12–18M 7-byte `memcpy`s and 12–18M `sign_extend_be` calls. Proven on exact `#165` files!                                                                                                                                                                       |
| **2** | **Merge same-column `RowFilter` conjuncts + Parquet dictionary-level filter pushdown & output-column reuse** | **−5.1 to −5.8 ms** from merging alone on `pushdown=true` (plus `−4.3 ms` from existing `reorder_filters=false`); **−2 to −4 ms** beyond post-FLBA `no_pushdown` via dict filter | **−1.7 to −2.9 ms** from merging alone on `pushdown=true` (plus `−5.4 ms` from `reorder_filters=false`); **−5 to −10 ms** beyond post-FLBA `no_pushdown` via dict filter + column reuse | (a) In `datafusion/datasource-parquet/src/row_filter.rs`, merge same-column conjuncts in `prebuild_row_filter_candidates` so `BETWEEN` / range predicates decode each column once (`−5.8 ms` / `−2.9 ms` at `reorder=true`, `−5.1 ms` / `−1.7 ms` at `reorder=false`); (b) evaluate pushed single-column predicates directly on Parquet dictionary pages (matching `duckdb::DictionaryDecoder::Filter`) and avoid re-decoding `l_discount` in `q6_scan` output projection. |
| **3** | **Close `q6_scan`'s Morsel-vs-Stock gap + cut fresh-process metadata/planning overhead**                     |                                                                                                                                    **−2 to −3 ms** fresh total (`25.8 → ~23 ms`) |                        **−2.3 to −3.6 ms** from closing Morsel-vs-Stock gap (`50.2 → 47.9 ms` warm, `68.4 → 64.8 ms` fresh, reaching DuckDB parity!) plus **−2 to −3 ms** fresh startup | (a) Eliminate `q6_scan`'s `2.3 ms` warm / `3.6 ms` fresh Morsel-vs-Stock gap (`50.2` vs `47.9 ms` post-FLBA, `369.6` vs `347.2 ms` CPU) driven by `ready_wait` (`16.5 ms`) and `ResultBuffer` coalescing copies; (b) reduce fresh `register_parquet` (`4.6 ms` listing + footer reads for 16 files) and `plan_seconds` (`2.7 ms`) by parallelizing footer reads / reusing cached `ParquetMetaData`.                                                                        |

## 5. Exact Reproduction Commands

Run from the repository root on a quiet Linux host (`taskset -c 0-7`, 8 cores,
`uv`, pinned Rust toolchain):

```bash
set -euo pipefail
p=benchmarks/morsel_tpch_profile
tpch=benchmarks/morsel_tpch
work=$(mktemp -d "${MORSEL_PROFILE_WORK_ROOT:-$HOME/.cache}/morsel-tpch-profile.XXXXXX")
rev=$(git rev-parse HEAD)

# 1. Build the standalone profiling runner in a detached worktree (zero repo edits).
git worktree add --detach "$work/pinned" "$rev"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$p/profile_runner.rs" "$work/pinned/datafusion/core/examples/morsel_tpch_profile_runner.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$rev" CARGO_TARGET_DIR="$work/target" \
    RUSTC_WRAPPER=sccache \
    timeout --kill-after=20s 35m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql \
      --example morsel_tpch_profile_runner
)
runner="$work/target/release-nonlto/examples/morsel_tpch_profile_runner"

# 2. Create the pinned Python 3.12.10 environment and generate the standard #165 SF1 dataset.
timeout 120s uv venv --python 3.12.10 "$work/venv"
py="$work/venv/bin/python"
timeout 180s uv pip install --python "$py" -r "$tpch/requirements.txt"
timeout 60s "$py" -B -m unittest discover -s "$p" -v
timeout 180s "$py" -B "$tpch/harness.py" generate "$work/tpch_sf1" --sf 1 --row-group-size 131072

# 3. Baseline (fresh-process, metrics=off), #138 metrics pass, warm in-session, batch sweep, and SQL micro-probes.
vmstat -y 1 3
timeout 240s taskset -c 0-7 "$py" -B "$p/profile_harness.py" baseline \
  "$work/tpch_sf1" --runner "$runner" --threads 8 --warmups 1 --repeats 7 \
  > "$work/sf1_baseline.jsonl"
timeout 240s taskset -c 0-7 "$py" -B "$p/profile_harness.py" baseline \
  "$work/tpch_sf1" --runner "$runner" --threads 8 --warmups 1 --repeats 7 --metrics \
  --arms morsel_no_pushdown,morsel_pruning,morsel_pushdown,morsel_pushdown_noreorder \
  > "$work/sf1_metrics.jsonl"
timeout 240s taskset -c 0-7 "$py" -B "$p/profile_harness.py" warm \
  "$work/tpch_sf1" --runner "$runner" --threads 8 --warmups 2 --repeats 7 --metrics \
  > "$work/sf1_warm.jsonl"
timeout 240s taskset -c 0-7 "$py" -B "$p/profile_harness.py" batch-sweep \
  "$work/tpch_sf1" --runner "$runner" --threads 8 --warmups 2 --repeats 7 --metrics \
  > "$work/sf1_batch_sweep.jsonl"
timeout 300s taskset -c 0-7 "$py" -B "$p/profile_harness.py" probes \
  "$work/tpch_sf1" --runner "$runner" --threads 8 --warmups 2 --repeats 5 --metrics \
  > "$work/sf1_probes.jsonl"

# 4. Arm PMUv3 hardware counters & leaf CPU profiles (unprivileged perf recipe from #155).
# If perf is not on PATH, download linux-tools + libslang2/libnuma1/libtraceevent1 via
# `apt download` and extract with `dpkg-deb -x`, then set PERF_BIN and LD_LIBRARY_PATH.
timeout 360s taskset -c 0-7 "$py" -B "$p/profile_perf.py" \
  "$work/tpch_sf1" "$work/perf_out" --runner "$runner" \
  --threads 8 --stat-reps 3 --loop-iters 20 \
  > "$work/sf1_perf.jsonl"

# 5. Disposable DataFusion scratch ablations (ablations.patch).
git worktree add --detach "$work/ablation_tree" "$rev"
mkdir -p "$work/ablation_tree/datafusion/core/examples"
cp "$p/profile_runner.rs" "$work/ablation_tree/datafusion/core/examples/morsel_tpch_profile_runner.rs"
git -C "$work/ablation_tree" apply "$p/ablations.patch"
(
  cd "$work/ablation_tree"
  MORSEL_BENCH_REVISION="$rev" CARGO_TARGET_DIR="$work/target" \
    RUSTC_WRAPPER=sccache \
    timeout --kill-after=20s 15m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql \
      --example morsel_tpch_profile_runner
)
cp "$work/target/release-nonlto/examples/morsel_tpch_profile_runner" "$work/ablation_runner"
timeout 240s taskset -c 0-7 "$py" -B "$p/run_ablations.py" \
  "$work/tpch_sf1" --runner "$work/ablation_runner" --suite datafusion \
  --threads 8 --warmups 2 --repeats 7 --metrics \
  > "$work/sf1_ablations.jsonl"

# 6. Disposable parquet-60.1.0 DirectDecimal128Reader paired validation (Waterfalls A & B).
parquet_src=$(ls -d "$HOME/.cargo/registry/src"/*/parquet-60.1.0 | head -n 1)
cp -r "$parquet_src" "$work/parquet-60.1.0-patched"
patch "$work/parquet-60.1.0-patched/src/arrow/array_reader/fixed_len_byte_array.rs" \
  "$p/parquet_flba_decimal_scratch.patch"
git worktree add --detach "$work/flba_tree" "$rev"
mkdir -p "$work/flba_tree/datafusion/core/examples"
cp "$p/profile_runner.rs" "$work/flba_tree/datafusion/core/examples/morsel_tpch_profile_runner.rs"
printf '\n[patch.crates-io]\nparquet = { path = "%s/parquet-60.1.0-patched" }\n' "$work" \
  >> "$work/flba_tree/Cargo.toml"
(
  cd "$work/flba_tree"
  MORSEL_BENCH_REVISION="$rev" CARGO_TARGET_DIR="$work/target_flba" \
    RUSTC_WRAPPER=sccache \
    timeout --kill-after=20s 15m cargo build --offline --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql \
      --example morsel_tpch_profile_runner
)
cp "$work/target_flba/release-nonlto/examples/morsel_tpch_profile_runner" "$work/flba_runner"
timeout 240s taskset -c 0-7 "$py" -B "$p/run_ablations.py" \
  "$work/tpch_sf1" --runner "$work/flba_runner" --suite flba \
  --threads 8 --warmups 2 --repeats 7 --metrics --fresh \
  > "$work/sf1_flba_paired.jsonl"
timeout 180s taskset -c 0-7 "$py" -B "$p/profile_harness.py" baseline \
  "$work/tpch_sf1" --runner "$runner" --threads 8 --warmups 1 --repeats 7 \
  --arms duckdb_default > "$work/sf1_duckdb_fresh_paired.jsonl"
timeout 180s taskset -c 0-7 "$py" -B "$p/profile_harness.py" warm \
  "$work/tpch_sf1" --runner "$runner" --threads 8 --warmups 2 --repeats 7 \
  --arms duckdb_default > "$work/sf1_duckdb_warm_paired.jsonl"

# 7. Optional diagnostic control: INT64-stored Decimal(15,2) Parquet (re-exported via DuckDB COPY).
mkdir -p "$work/tpch_sf1_int64dec/lineitem"
"$py" -c "
import duckdb
con = duckdb.connect(':memory:')
con.execute('SET threads = 4')
for i in range(16):
    src = f'$work/tpch_sf1/lineitem/part-{i:06d}.parquet'
    dst = f'$work/tpch_sf1_int64dec/lineitem/part-{i:06d}.parquet'
    con.execute(f\"COPY (SELECT * FROM read_parquet('{src}')) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 131072)\")
"
timeout 120s taskset -c 0-7 "$py" -B "$p/profile_harness.py" warm \
  "$work/tpch_sf1_int64dec" --runner "$runner" --threads 8 --warmups 2 --repeats 7 --metrics \
  --arms morsel_no_pushdown,morsel_pushdown,stock_no_pushdown,duckdb_default \
  > "$work/sf1_int64dec_warm.jsonl"

# 8. Clean up all disposable worktrees.
git worktree remove --force "$work/pinned"
git worktree remove --force "$work/ablation_tree"
git worktree remove --force "$work/flba_tree"
```
