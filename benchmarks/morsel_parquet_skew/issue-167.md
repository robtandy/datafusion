<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Intra-row-group Parquet morsel splitting (#167)

Comparison of stock (`morsel_mode = off`), legacy native (`morsel_mode = required`,
splitting disabled via `ParquetSplitConfig { enabled: false }`), and split native
(`morsel_mode = required`, default splitting: `min_row_group_rows = 1 Mi`,
`target_slice_rows = 128 Ki`, page-aligned boundary selection, `256 MiB`
query-wide compressed split-buffer cap, `16 MiB` per-slice compressed cap,
`128 MiB` decoded workspace ceiling) on the same files and binary built from
`agent/morsel-issue-167` (based on `9c3079473`). This note does not replace the
pinned [#156 baseline](README.md#dduck2-warm-cache-baseline) or the pinned
[#128 baseline](../morsel_parquet/README.md).

## Method

`rob-tandy-dduck1`, November 23, 2026 UTC: Ubuntu 24.04.5, Linux
`7.0.0-1014-aws`, AArch64 Neoverse-N1, Btrfs on local EC2 NVMe instance storage,
Rust `1.99.1 (58cf8f9c3)`, Python `3.12.10` (`numpy 2.2.6`, `pyarrow 20.0.0`).
No builds or lint jobs ran during measurement (`vmstat 1 3` reported 97% CPU
idle before launch). All runs were pinned to CPUs 0–7 (`taskset -c 0-7`) with
8 workers, `batch_size = 8192`, `repartition_file_scans = false`,
`collect_statistics = false`, `pruning = false`, `bloom_filter_on_read = false`,
1 warmup round, and 5 measured paired rounds rotating arm order each round.
Full Linux page-cache residency was verified via `mincore` before every timed
sample. Every arm passed #128's exact NumPy oracle before timing
(`378,916` / `4,941,572` rows on the 8,388,608-row `#156` layouts;
`759,841` / `9,883,940` rows on the 16,777,216-row `#128` uniform layout).

Fixtures (Zstandard level 3, PLAIN encoding, `use_dictionary = False`, statistics
enabled, PyArrow default 1 MiB data pages):

| Fixture                    | Files | Row groups per file            | Rows / RG | Page/Offset Index | Total bytes |
| -------------------------- | ----: | ------------------------------ | --------: | ----------------- | ----------: |
| `large-row-groups-noindex` |     2 | `2, 2` (4 RGs)                 | 2,097,152 | No                | 155,608,030 |
| `large-row-groups-indexed` |     2 | `2, 2` (4 RGs)                 | 2,097,152 | Yes               | 155,607,514 |
| `file-skew-indexed`        |    16 | `24, 1, 24, 1, 2, 2, 1×10`     |   131,072 | Yes               | 155,793,381 |
| `uniform-128rg` (#128)     |    16 | `8 × 16` (128 RGs)             |   131,072 | No                | 311,529,117 |

## 1. `#156` `large-row-groups` no-index control (`pushdown_filters = false`)

With `enable_page_index = true` and splitting enabled, files without an Offset
Index cleanly fall back to unsplit row groups (`parquet_split_row_groups = 0`,
4 morsels across 4 of 8 workers), matching legacy native with zero overhead.

| Query        | Stock ms, median (range) | Native (no-index) ms, median (range) | Active workers | Claims (round 1)           | `bytes_scanned` |
| ------------ | -----------------------: | -----------------------------------: | -------------: | -------------------------- | --------------: |
| `scan_low`   |      299.7 (294.8–302.9) |                  186.8 (184.8–190.2) |            4/8 | `[0, 0, 0, 1, 1, 1, 0, 1]` |      98,222,226 |
| `count_low`  |      145.9 (143.9–147.3) |                    94.1 (92.4–95.0)  |            4/8 | `[0, 0, 0, 1, 1, 1, 0, 1]` |      37,497,725 |
| `scan_high`  |      302.1 (300.4–305.3) |                  193.3 (189.0–196.4) |            4/8 | `[0, 1, 0, 1, 1, 0, 0, 1]` |      98,222,226 |
| `count_high` |      145.5 (143.2–147.3) |                    94.3 (92.3–94.3)  |            4/8 | `[0, 0, 0, 1, 1, 1, 0, 1]` |      37,497,725 |

## 2. `#156` `large-row-groups` indexed (`pushdown_filters = false`)

All 4 large row groups split cleanly into page-aligned slices (`60` slices for
`scan_*`, `32` slices for `count_*`). **All 8 workers claim and complete slices
in every trial**, `bytes_scanned` is **byte-for-byte identical (1.000×)** to
unsplit storage reads, and peak unique compressed buffer residency stays at
20–33 MiB (well below the 256 MiB cap).

| Query        | Stock ms, median (range) | Legacy native ms (4/8 workers) | Split native ms (8/8 workers) | Split vs legacy | Split vs stock | Split claims (round 1)     | `bytes_scanned` (all arms) | Reused bytes | Peak unique bytes |
| ------------ | -----------------------: | -----------------------------: | ----------------------------: | --------------: | -------------: | -------------------------- | -------------------------: | -----------: | ----------------: |
| `scan_low`   |      293.0 (290.4–295.5) |            183.1 (181.2–184.4) |           157.7 (151.5–166.3) |          -13.9% |         -46.2% | `[8, 7, 7, 8, 7, 8, 7, 8]` |                 98,214,742 |   74,547,897 |        34,762,249 |
| `count_low`  |      142.7 (139.6–143.6) |              91.3 (91.0–92.7)  |             66.7 (65.0–68.0)  |          -26.9% |         -53.3% | `[4, 4, 4, 4, 4, 3, 5, 4]` |                 37,495,813 |   16,371,853 |        22,227,078 |
| `scan_high`  |      294.0 (292.3–298.5) |            188.2 (188.0–190.3) |           167.9 (159.2–172.9) |          -10.8% |         -42.9% | `[7, 7, 7, 8, 8, 7, 8, 8]` |                 98,214,742 |   74,547,897 |        34,022,386 |
| `count_high` |      140.6 (139.9–142.9) |              91.5 (90.9–92.5)  |             65.5 (62.9–72.4)  |          -28.4% |         -53.4% | `[4, 4, 4, 4, 4, 4, 4, 4]` |                 37,495,813 |   16,371,853 |        20,595,937 |

Worker ready-wait drops sharply when all 8 workers have claimable slices:
`scan_low` aggregate `morsel_ready_wait` drops from `231.4 ms` → `49.7 ms`, and
`count_low` drops from `98.2 ms` → `31.5 ms`. On `count_*`, snapping slice
boundaries to `delta`'s ~273 Ki-row pages yields `32` slices with only `1.44×`
decode bytes (`16.4 MiB` reused from the broker, zero extra storage reads),
cutting wall time by **26.9–28.4%** over legacy native and **53.3–53.4%** over
stock. On `scan_*`, snapping to `payload`'s ~140 Ki-row pages keeps `payload`
at `1.00×` while `score`, `delta`, and `row_id` straddle boundaries (`1.76×`
total decode bytes, `74.5 MiB` served from the broker with zero duplicate I/O),
yielding **10.8–13.9%** wall-time wins over legacy native and **42.9–46.2%**
over stock.

## 3. `#156` `large-row-groups` indexed (`pushdown_filters = true`)

Splitting also applies to `FilteredBufferedReaders` (with `AsyncRangeBroker`
deduplicating shared boundary pages across Phase-1 and concurrent Phase-2 lanes
and `RowGroupCache` batch-boundary expansion scoped strictly to cached leaves).

| Query        | Stock ms, median (range) | Legacy native ms (4/8 workers) | Split native ms (8/8 workers) | Split vs legacy | Split vs stock | Split claims (round 1)     | `bytes_scanned` (all arms) | Peak unique bytes |
| ------------ | -----------------------: | -----------------------------: | ----------------------------: | --------------: | -------------: | -------------------------- | -------------------------: | ----------------: |
| `scan_low`   |      328.4 (326.3–336.8) |            177.3 (176.1–180.1) |           173.2 (171.8–175.7) |           -2.3% |         -47.3% | `[8, 6, 9, 8, 8, 7, 9, 5]` |                 98,214,742 |        19,500,222 |
| `count_low`  |      159.0 (157.8–159.2) |              88.9 (88.5–89.2)  |             69.4 (64.9–74.1)  |          -21.9% |         -56.4% | `[3, 4, 5, 3, 5, 4, 4, 4]` |                 37,495,813 |        14,181,107 |
| `scan_high`  |      346.1 (345.0–347.9) |            183.7 (183.0–186.2) |           185.6 (184.3–187.2) |           +1.0% |         -46.4% | `[7, 5, 8, 7, 8, 9, 8, 8]` |                 98,214,742 |        18,089,304 |
| `count_high` |      159.9 (159.5–161.0) |              89.3 (88.9–90.0)  |             84.4 (79.6–85.1)  |           -5.5% |         -47.2% | `[4, 4, 4, 4, 3, 4, 4, 5]` |                 37,495,813 |        14,577,361 |

All 8 workers claim work across all 4 queries with zero duplicate storage I/O
(`bytes_scanned` identical across all arms) and lower peak unique compressed
residency (`13.5–18.6 MiB`). On `count_low`, split native drops wall time by
**21.9%** vs legacy native (`88.9 → 69.4 ms`) and **56.4%** vs stock. On
`scan_high` with `pushdown_filters = true`, per-slice `RowFilter` + `RowGroupCache`
boundary decompression overhead (`1242 ms` vs `593 ms` aggregate worker decode
work across 8 vs 4 workers) balances out the 4→8 worker concurrency gain
(`185.6` vs `183.7 ms`, still **46.4%** ahead of stock).

## 4. `#156` `file-skew` indexed control (`pushdown_filters = false`)

Row groups are `131,072` rows (`< 1 Mi` threshold), so `parquet_split_row_groups = 0`:
all 64 row groups stay unsplit and take the fast single-decoder path, spreading
across all 8 workers with zero regression (**2.2–2.3× faster than stock**).

| Query        | Stock ms, median (range) | Legacy native ms (8/8 workers) | Split native ms (8/8 workers) | Split RGs | Claims (round 1)           |
| ------------ | -----------------------: | -----------------------------: | ----------------------------: | --------: | -------------------------- |
| `scan_low`   |      225.9 (222.4–230.8) |              96.8 (95.0–104.9) |              97.5 (96.0–98.4) |         0 | `[8, 8, 8, 9, 7, 7, 8, 9]` |
| `count_low`  |      118.2 (114.6–119.2) |              50.9 (49.9–52.4)  |              51.4 (49.0–54.5) |         0 | `[9, 9, 7, 9, 7, 8, 8, 7]` |
| `scan_high`  |      239.1 (238.0–241.5) |            108.0 (103.9–111.9) |            108.6 (107.1–113.5)|         0 | `[8, 8, 7, 9, 8, 8, 8, 8]` |
| `count_high` |      116.8 (115.3–119.7) |              50.4 (48.9–51.8)  |              50.3 (49.9–51.0) |         0 | `[9, 8, 7, 8, 8, 8, 7, 9]` |

## 5. `#128` uniform 128-RG non-regression check

Row groups are `131,072` rows (`< 1 Mi` threshold), so `parquet_split_row_groups = 0`
and all 128 row groups stay on the unsplit fast path with identical `bytes_scanned`
and zero regression for both `pushdown_filters = false` and `pushdown_filters = true`.

| `pushdown_filters` | Query        | Stock ms, median (range) | Legacy native ms, median (range) | Split native ms, median (range) | Split RGs |
| ------------------ | ------------ | -----------------------: | -------------------------------: | ------------------------------: | --------: |
| `false`            | `scan_low`   |      171.8 (167.0–175.1) |              171.0 (170.6–173.7) |             171.4 (168.9–172.7) |         0 |
| `false`            | `count_low`  |        90.2 (89.4–95.8)  |                87.7 (86.5–91.0)  |               91.4 (87.5–96.4)  |         0 |
| `false`            | `scan_high`  |      193.8 (189.7–195.6) |              195.4 (189.3–206.9) |             193.1 (190.6–200.0) |         0 |
| `false`            | `count_high` |        89.8 (88.5–92.4)  |                88.1 (87.3–92.5)  |               89.7 (88.8–90.0)  |         0 |
| `true`             | `scan_low`   |      211.2 (205.4–215.3) |              216.9 (209.3–220.2) |             213.6 (208.8–217.3) |         0 |
| `true`             | `count_low`  |        90.0 (88.5–92.7)  |                87.5 (85.4–91.8)  |               88.2 (86.7–89.5)  |         0 |
| `true`             | `scan_high`  |      203.8 (202.1–207.6) |              203.8 (196.2–210.2) |             203.6 (202.8–238.7) |         0 |
| `true`             | `count_high` |        92.3 (88.7–93.4)  |                90.1 (87.7–92.2)  |               87.8 (84.7–90.7)  |         0 |
