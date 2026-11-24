<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Skewed-file and large-row-group Parquet benchmarks after #148–#167 (#172)

Engine revision under test: `04fa5fc6456f62ee40fe25311e70894e7deca967`
(`morsel-execution` after `#148` producer scaling, `#149` worker-local output
coalescing and scaled result channel, `#150` Parquet row-level filter pushdown,
`#166` direct mask-bit `COUNT(*)` and filter-column pruning, and `#167`
intra-row-group Parquet morsel splitting). Compared on idle `rob-tandy-dduck2`
against the pinned [#156 baseline](README.md#dduck2-warm-cache-baseline) at
`c2809d3b3ee73936b182429e1646af290d797c6e` (the `#138` metrics merge). This
note does not replace the pinned `#156` baseline in [README.md](README.md).

## Summary

- **`file-skew` flips from 1.1–1.4× slower than stock to 2.15–2.37× faster than
  stock.** On the exact `#156` no-index fixture (`pushdown_filters = false`),
  native median wall time drops **60.9–69.4% (2.56–3.26× faster)** vs
  `c2809d3b`: `scan_low` `321.2 → 98.4 ms`, `count_low` `129.9 → 50.5 ms`,
  `scan_high` `322.7 → 111.8 ms`, and `count_high` `131.5 → 51.4 ms`. Aggregate
  `morsel_ready_wait` across the 8 workers collapses **23.3–57.7×**
  (`1,212.3 → 21.0 ms` on `scan_low`, `409.6 → 14.9 ms` on `count_low`) as
  `#148` raises `morsel_peak_ready_depth` from `2` to `8`. With
  `pushdown_filters = true`, native stays **2.28–2.37× faster than stock**
  (`118.9 / 51.3 / 113.9 / 51.3 ms` vs `281.6 / 116.8 / 265.1 / 119.6 ms`).
- **`large-row-groups` reaches 8/8 worker utilization in every trial with a
  Page/Offset Index and `#167` splitting opted in** (`MORSEL_SPLIT_ENABLED=1`,
  `MORSEL_ENABLE_PAGE_INDEX=1`), up from **4/8** in `#156` and in the paired
  unsplit `legacy` arm. All 4 row groups split cleanly into `60` page-aligned
  slices on `scan_*` and `32` slices on `count_*`, with **byte-for-byte
  identical `bytes_scanned`** across `off`, `legacy`, and `split` (`98,214,742`
  bytes for `scan_*`, `37,495,813` bytes for `count_*`, zero duplicate storage
  reads).
- **Splitting cuts `large-row-groups` wall time by 23.9–43.0% vs `#156`
  (`c2809d3b`) and 45.4–54.9% (1.83–2.22×) vs stock** under
  `pushdown_filters = false`: `scan_low` `207.7 → 155.5 ms`, `count_low`
  `108.8 → 65.8 ms`, `scan_high` `215.1 → 163.6 ms`, and `count_high`
  `112.6 → 64.2 ms`. Aggregate `morsel_ready_wait` drops **5.3–7.5×** vs
  `c2809d3b` (`422.4 → 56.3 ms` on `scan_low`, `204.4 → 38.6 ms` on
  `count_low`).
- **Under `pushdown_filters = true` on `large-row-groups-indexed`**, split
  native still activates **8/8 workers** in every trial and beats stock by
  **1.82–2.35×** (`45.0–57.4%` faster), cutting `count_low` by **23.6%** vs
  unsplit `legacy` (`90.8 → 69.3 ms`) and `count_high` by **8.2%**
  (`92.0 → 84.5 ms`). On the 4-column `scan_*` queries, per-slice `RowFilter` +
  `RowGroupCache` boundary-page re-decompression doubles `morsel_decode_work`
  (`587.8 → 1,146.4 ms` on `scan_low`, `600.2 → 1,245.8 ms` on `scan_high`),
  making `scan_low` roughly flat (`180.3 → 176.2 ms`, `-2.2%`) and `scan_high`
  slightly slower (`186.0 → 193.3 ms`, `+3.9%`) than unsplit 4/8-worker
  `legacy`.
- **Clean fallback and zero regression when splitting does not apply.** On the
  exact `#156` no-index `large-row-groups` fixture with splitting opted in,
  `parquet_split_row_groups = 0` (4/8 workers), while `#148`/`#149`/`#166`
  still speed up native by **11.8–17.4%** over `#156` (`182.9 / 92.2 / 189.6 /
93.0 ms`, `1.54–1.60×` faster than stock) and cut `morsel_ready_wait` by
  **1.8–2.2×**. On `file-skew-indexed` (`131,072` rows/RG `< 1 Mi` split
  threshold), `parquet_split_row_groups = 0` and split matches legacy within
  ~1 ms in both pushdown modes.
- **Exact parity verified across all arms, fixtures, and pushdown modes.**
  Every arm passed `#128`'s exact NumPy oracle before any timing was emitted
  (`378,916` rows at low selectivity, `4,941,572` rows at high selectivity,
  projected values including nulls, and exact count scalars).

## Method

Recorded on `rob-tandy-dduck2`, 2026-11-24 UTC: Ubuntu 24.04.5 LTS, Linux
`7.0.0-1012-aws`, 16 AArch64 Neoverse-N1 vCPUs, Btrfs on local EC2 NVMe
instance storage (`/home/bits/.cache`), Rust `1.99.1 (58cf8f9c3 2026-10-13)`,
Clang/lld `18.1.3`, Python `3.12.10` (`numpy 2.2.6`, `pyarrow 20.0.0`).

All runs were pinned to CPUs 0–7 (`taskset -c 0-7`) with 8 Tokio workers,
8 target partitions, `batch_size = 8192`, `repartition_file_scans = false`,
`collect_statistics = false`, `pruning = false`, `bloom_filter_on_read = false`,
1 warmup round, and 5 measured counterbalanced rounds per query. Full Linux
page-cache residency was verified via `mincore` before every warmup and timed
trial. No competing builds or benchmarks were running during the recorded runs;
the two 1-second `vmstat 1 3` samples (excluding the since-boot line) showed
88–99% whole-host CPU idle before and after every recorded run:

- `c2809d3b` baseline re-run: before 92%, 88% idle; after 93%, 99% idle.
- `file-skew` (no-index): before 88%, 88% idle; after 98%, 99% idle.
- `file-skew-indexed`: before 92%, 93% idle; after 93%, 99% idle.
- `large-row-groups` (no-index): before 89%, 89% idle; after 91%, 97% idle.
- `large-row-groups-indexed`: before 92%, 89% idle; after 97%, 99% idle.

Because `#167` makes intra-row-group splitting opt-in
(`ParquetSplitConfig::default().enabled == false`) and requires a Parquet
Offset Index (`enable_page_index = true` + files written with `--page-index`),
the suite covers four 8,388,608-row fixtures (Zstandard level 3, PLAIN
encoding, `use_dictionary = False`, statistics enabled):

| Fixture                    | Files | Row groups per file (path order) | Rows / RG | Page/Offset Index | Total bytes | Resident 4 KiB pages |
| -------------------------- | ----: | -------------------------------- | --------: | ----------------- | ----------: | -------------------: |
| `file-skew`                |    16 | `24, 1, 24, 1, 2, 2, 1×10`       |   131,072 | No                | 155,785,925 |      38,046 / 38,046 |
| `file-skew-indexed`        |    16 | `24, 1, 24, 1, 2, 2, 1×10`       |   131,072 | Yes               | 155,793,381 |      38,046 / 38,046 |
| `large-row-groups`         |     2 | `2, 2` (4 RGs)                   | 2,097,152 | No                | 155,608,030 |      37,991 / 37,991 |
| `large-row-groups-indexed` |     2 | `2, 2` (4 RGs)                   | 2,097,152 | Yes               | 155,607,514 |      37,991 / 37,991 |

`file-skew` and `large-row-groups` match `#156`'s exact byte counts and
resident page counts (`155,785,925` / `38,046` and `155,608,030` / `37,991`).
All `#138` time metrics (`morsel_ready_wait`, `morsel_output_wait`,
`morsel_decode_work`, `morsel_process_work`) are sums across all 8 workers in
milliseconds (median across the 5 repeats); `morsel_pending_io` is the
query-wide reader-producer pending-I/O timer in milliseconds. Native runs
include `#138` metrics collection overhead, matching `#156`.

Physical plans were verified on every arm and trial:

- Stock (`off`): `DataSourceExec` (`file_type=parquet`), 8 file partitions on
  `file-skew` and 2 file partitions on `large-row-groups`, with no `MorselExec`.
- Native (`required`, `pushdown_filters = false`):
  `MorselExec: workers=8, NativeParquet[fixture] -> Filter[...] -> Projection[...] -> ResultSink`
  (or `Projection[] -> CountSink; FinalizedCountSource -> ...` with
  `count_finalizations = 1` for `count_*`).
- Native (`required`, `pushdown_filters = true`):
  `MorselExec: workers=8, NativeParquet[fixture, predicate=...] -> Projection[...] -> ResultSink`
  (or `CountSink` with `count_finalizations = 1`), with no separate `Filter`
  operator in the morsel pipeline.

## 1. Host comparability & `c2809d3b` `#138` baseline

The `#156` runner at `c2809d3b` did not emit `#138` wait/work durations. To
obtain exact before-vs-after `#138` wait metrics on the same host, `c2809d3b`
was rebuilt in a disposable worktree with `stream_runner.rs` (minus
`ParquetSplitConfig`, which did not exist at `c2809d3b`) and run on the exact
`#156` no-index fixtures (`pushdown_filters = false`, `enable_page_index =
false`):

| Fixture            | Query        | Pinned `#156` Stock ms | Today `c2809d3b` Stock ms | Pinned `#156` Native ms | Today `c2809d3b` Native ms | `ready_wait` ms | `output_wait` ms | `pending_io` ms | `decode` ms | `process` ms | Peak ready / out |
| ------------------ | ------------ | ---------------------: | ------------------------: | ----------------------: | -------------------------: | --------------: | ---------------: | --------------: | ----------: | -----------: | ---------------: |
| `file-skew`        | `scan_low`   |                  226.5 |       229.8 (228.6–231.8) |                   321.2 |        319.4 (318.6–321.1) |         1,212.3 |              0.4 |            48.8 |       539.7 |         78.6 |            2 / 2 |
| `file-skew`        | `count_low`  |                  118.3 |       118.2 (116.4–118.7) |                   129.9 |        129.8 (128.9–130.8) |           409.6 |              0.0 |            21.6 |       240.2 |         67.6 |            2 / 1 |
| `file-skew`        | `scan_high`  |                  242.3 |       242.5 (240.2–246.9) |                   322.7 |        320.9 (320.0–324.2) |         1,194.4 |              0.5 |            48.4 |       538.7 |        104.0 |            2 / 2 |
| `file-skew`        | `count_high` |                  117.4 |       119.6 (116.8–126.3) |                   131.5 |        131.3 (130.2–132.3) |           407.4 |              0.0 |            23.2 |       240.3 |         80.5 |            2 / 1 |
| `large-row-groups` | `scan_low`   |                  289.1 |       294.6 (291.9–298.2) |                   207.7 |        212.0 (210.1–214.7) |           422.4 |              0.1 |            62.7 |       498.3 |         77.6 |            1 / 2 |
| `large-row-groups` | `count_low`  |                  142.2 |       143.7 (142.4–144.7) |                   108.8 |        109.6 (107.9–110.3) |           204.4 |              0.0 |            24.9 |       234.3 |         63.7 |            2 / 1 |
| `large-row-groups` | `scan_high`  |                  290.9 |       295.7 (293.9–296.5) |                   215.1 |        219.7 (216.4–221.1) |           425.7 |              0.1 |            62.5 |       501.8 |        102.0 |            1 / 2 |
| `large-row-groups` | `count_high` |                  143.1 |       142.0 (141.3–144.3) |                   112.6 |        113.2 (112.5–114.7) |           205.3 |              0.0 |            24.7 |       234.2 |         77.7 |            1 / 1 |

Every median matches the pinned `#156` `dduck2` baseline within `0.0–2.1%`,
confirming host comparability. The `#138` metrics confirm `#155`'s diagnosis:
at `c2809d3b`, `file-skew` workers spent `1,194–1,212 ms` aggregate waiting on
the 2-reader ready FIFO (`Peak ready = 2`) during `scan_*` — more than twice
their `539 ms` decode work.

## 2. `file-skew` (exact `#156` no-index fixture)

### `pushdown_filters = false` (apples-to-apples with `#156`)

| Query        | Stock ms, median (range) | Native `#156` (`c2809d3b`) ms | Native `#167` (`04fa5fc6`) ms | vs `#156` | vs Stock | Active workers | Claims (round 1)            |
| ------------ | -----------------------: | ----------------------------: | ----------------------------: | --------: | -------: | -------------: | --------------------------- |
| `scan_low`   |      231.6 (227.6–232.4) |                         321.2 |             98.4 (95.9–101.6) |    -69.4% |   -57.5% |            8/8 | `[8, 8, 8, 8, 8, 8, 8, 8]`  |
| `count_low`  |      118.0 (116.5–120.1) |                         129.9 |              50.5 (50.2–51.6) |    -61.1% |   -57.2% |            8/8 | `[8, 8, 8, 7, 8, 9, 8, 8]`  |
| `scan_high`  |      240.9 (239.2–244.4) |                         322.7 |           111.8 (110.8–113.4) |    -65.3% |   -53.6% |            8/8 | `[7, 8, 9, 8, 8, 10, 7, 7]` |
| `count_high` |      119.1 (118.0–119.4) |                         131.5 |              51.4 (50.3–51.8) |    -60.9% |   -56.8% |            8/8 | `[8, 7, 8, 9, 9, 8, 8, 7]`  |

`#138` metrics before (`c2809d3b`) vs after (`04fa5fc6`, `pushdown_filters =
false`):

| Query        | `ready_wait` ms (before → after) | `output_wait` ms (before → after) | `pending_io` ms (before → after) | `decode` ms (before → after) | `process` ms (before → after) | Peak ready / out (before → after) |
| ------------ | -------------------------------: | --------------------------------: | -------------------------------: | ---------------------------: | ----------------------------: | --------------------------------: |
| `scan_low`   |          1,212.3 → 21.0 (-57.7×) |                         0.4 → 0.0 |                     48.8 → 162.5 |                539.7 → 535.2 |                   78.6 → 87.5 |                         2/2 → 8/7 |
| `count_low`  |            409.6 → 14.9 (-27.6×) |                         0.0 → 0.0 |                      21.6 → 74.0 |                240.2 → 231.2 |                   67.6 → 62.6 |                         2/1 → 8/1 |
| `scan_high`  |          1,194.4 → 23.9 (-49.9×) |                        0.5 → 21.5 |                     48.4 → 153.0 |                538.7 → 528.9 |                 104.0 → 112.8 |                        2/2 → 8/16 |
| `count_high` |            407.4 → 17.5 (-23.3×) |                         0.0 → 0.0 |                      23.2 → 73.0 |                240.3 → 238.3 |                   80.5 → 58.9 |                         2/1 → 8/1 |

Removing the 2-reader bottleneck in `#148` allows all 8 workers to saturate
dynamic row-group stealing across the skewed 16-file layout (where stock's two
heaviest static partitions hold 25 of the 64 row groups each), flipping native
from slower than stock to **2.15–2.35× faster than stock**. `morsel_pending_io`
increases (`48.8 → 162.5 ms`) because `#148` runs more concurrent file-lane I/O
producers in parallel (overlapped with worker decode), driving aggregate worker
`morsel_ready_wait` down by **23–58×**. On `count_high`, `#166`'s direct
mask-bit count also cuts `morsel_process_work` by **26.8%** (`80.5 → 58.9 ms`).
On `scan_high`, `morsel_output_wait` rises to `21.5 ms` (`peak_output_depth =
16`, `#149`'s scaled result channel) as all 8 workers stream 4.94M projected
rows into the single consumer stream.

### `pushdown_filters = true` (`04fa5fc6`)

| Query        | Stock ms, median (range) | Native (`04fa5fc6`) ms | vs Stock | vs `pd=false` | Active workers | Claims (round 1)             | `ready_wait` ms | `output_wait` ms | `decode` ms | `process` ms |
| ------------ | -----------------------: | ---------------------: | -------: | ------------: | -------------: | ---------------------------- | --------------: | ---------------: | ----------: | -----------: |
| `scan_low`   |      281.6 (277.9–284.6) |    118.9 (114.1–122.6) |   -57.8% |        +20.8% |            8/8 | `[8, 7, 9, 7, 10, 9, 7, 7]`  |            50.6 |              0.0 |       699.4 |          0.9 |
| `count_low`  |      116.8 (116.6–118.0) |       51.3 (50.7–52.1) |   -56.1% |         +1.4% |            8/8 | `[9, 8, 7, 8, 8, 9, 7, 8]`   |            43.7 |              0.0 |       298.7 |          0.0 |
| `scan_high`  |      265.1 (258.9–276.7) |    113.9 (111.7–115.9) |   -57.0% |         +1.8% |            8/8 | `[10, 6, 9, 10, 7, 7, 7, 8]` |            42.6 |              7.5 |       644.1 |         11.2 |
| `count_high` |      119.6 (117.4–119.8) |       51.3 (49.7–55.6) |   -57.1% |         -0.3% |            8/8 | `[9, 7, 9, 9, 8, 8, 6, 8]`   |            33.7 |              0.0 |       304.2 |          0.1 |

With `pushdown_filters = true`, predicate evaluation moves inside Parquet
decode (`morsel_process_work` drops to `0.0–11.2 ms`). Native stays
**2.28–2.37× faster than stock** on all four queries. On `scan_low`, both stock
(`231.6 → 281.6 ms`) and native (`98.4 → 118.9 ms`) are slower than
`pushdown_filters = false`: because `(score + 3 * delta) % 1000 < 50` matches
pseudo-randomly across every 8,192-row batch, `RowSelection` cannot skip data
pages of `row_id`/`payload` and incurs per-row selection-mask decode overhead
(`decode + process` rises from `622.7 → 700.3 ms`).

### `file-skew-indexed` control (`--enable-page-index --compare-split`)

With 131,072-row row groups (below `#167`'s `1 Mi`-row split threshold),
`parquet_split_row_groups = 0` in every trial and `split` matches `legacy` with
zero overhead (all arms 8/8 active, `38,046 / 38,046` resident pages):

| `pushdown_filters` | Query        | Stock ms, median (range) | Legacy native (`split=0`) ms | Split native (`split=1`) ms | Split RGs | Split claims (round 1)      |
| ------------------ | ------------ | -----------------------: | ---------------------------: | --------------------------: | --------: | --------------------------- |
| `false`            | `scan_low`   |      231.9 (229.4–234.1) |            99.7 (99.0–101.4) |           99.5 (97.9–103.3) |         0 | `[8, 8, 8, 7, 8, 8, 8, 9]`  |
| `false`            | `count_low`  |      118.3 (117.6–119.4) |             52.1 (50.2–54.0) |            52.4 (51.2–54.8) |         0 | `[9, 7, 7, 9, 8, 8, 9, 7]`  |
| `false`            | `scan_high`  |      247.2 (243.3–248.6) |          113.1 (110.2–115.0) |         112.1 (108.8–117.3) |         0 | `[9, 8, 9, 8, 8, 8, 7, 7]`  |
| `false`            | `count_high` |      120.1 (119.1–122.0) |             52.0 (50.4–55.2) |            52.6 (51.2–54.3) |         0 | `[8, 9, 8, 7, 8, 7, 8, 9]`  |
| `true`             | `scan_low`   |      280.0 (275.8–283.4) |          116.2 (112.0–117.6) |         116.1 (113.0–120.3) |         0 | `[9, 9, 9, 8, 8, 4, 7, 10]` |
| `true`             | `count_low`  |      118.4 (117.2–122.2) |             52.1 (49.2–57.3) |            52.3 (51.0–53.5) |         0 | `[9, 8, 8, 9, 8, 8, 7, 7]`  |
| `true`             | `scan_high`  |      266.1 (245.0–273.3) |          114.0 (110.0–117.3) |         111.7 (110.7–117.5) |         0 | `[6, 7, 6, 6, 9, 9, 12, 9]` |
| `true`             | `count_high` |      118.2 (116.9–119.3) |             50.5 (49.3–60.0) |            51.8 (50.7–54.1) |         0 | `[9, 7, 8, 8, 8, 7, 8, 9]`  |

## 3. `large-row-groups` (exact `#156` no-index fixture)

When run with `--enable-page-index --enable-split` against the exact `#156`
fixture (which has no Parquet Offset Index), `#167` cleanly falls back to
unsplit row groups (`parquet_split_row_groups = 0`, 4/8 active workers, exact
`bytes_scanned` match with stock). Even without splitting, `#148`/`#149`/`#166`
cut native wall time by **11.8–17.4%** vs `#156` (`c2809d3b`) and roughly halve
`morsel_ready_wait` (`422.4 → 226.2 ms` on `scan_low`, `204.4 → 93.0 ms` on
`count_low`):

| `pushdown_filters` | Query        | Stock ms, median (range) | Native `#156` (`c2809d3b`) ms | Native `#167` fallback ms | vs `#156` | vs Stock | Active workers | Claims (round 1)           | `ready_wait` ms | `decode` ms | `process` ms |
| ------------------ | ------------ | -----------------------: | ----------------------------: | ------------------------: | --------: | -------: | -------------: | -------------------------- | --------------: | ----------: | -----------: |
| `false`            | `scan_low`   |      293.5 (291.9–296.7) |                         207.7 |       182.9 (181.8–184.4) |    -11.9% |   -37.7% |            4/8 | `[1, 0, 0, 1, 1, 0, 0, 1]` |           226.2 |       493.4 |         74.2 |
| `false`            | `count_low`  |      143.8 (142.3–144.6) |                         108.8 |          92.2 (91.4–92.8) |    -15.3% |   -35.9% |            4/8 | `[1, 1, 0, 1, 0, 0, 0, 1]` |            93.0 |       230.3 |         54.2 |
| `false`            | `scan_high`  |      295.8 (295.3–298.6) |                         215.1 |       189.6 (188.1–190.4) |    -11.8% |   -35.9% |            4/8 | `[1, 1, 0, 1, 0, 0, 0, 1]` |           231.3 |       492.5 |         94.5 |
| `false`            | `count_high` |      143.5 (141.9–144.5) |                         112.6 |          93.0 (91.8–93.5) |    -17.4% |   -35.2% |            4/8 | `[1, 1, 0, 1, 0, 0, 0, 1]` |            93.8 |       231.7 |         54.7 |
| `true`             | `scan_low`   |      330.7 (329.0–333.7) |                             — |       178.8 (176.7–180.3) |         — |   -45.9% |            4/8 | `[0, 0, 1, 1, 1, 0, 0, 1]` |           484.7 |       584.8 |          0.8 |
| `true`             | `count_low`  |      161.1 (159.7–161.7) |                             — |          89.5 (88.9–91.0) |         — |   -44.5% |            4/8 | `[1, 1, 1, 0, 0, 0, 0, 1]` |           389.9 |       289.6 |          0.0 |
| `true`             | `scan_high`  |      350.7 (349.0–351.8) |                             — |       185.2 (184.9–186.1) |         — |   -47.2% |            4/8 | `[1, 1, 1, 0, 0, 0, 0, 1]` |           484.6 |       598.2 |          6.6 |
| `true`             | `count_high` |      162.0 (161.1–163.1) |                             — |          91.1 (90.3–92.0) |         — |   -43.8% |            4/8 | `[1, 1, 1, 0, 0, 0, 0, 1]` |           391.1 |       291.9 |          0.0 |

## 4. `large-row-groups-indexed` (Page/Offset Index enabled, `#167` splitting)

With a Parquet Page/Offset Index (`generate large-row-groups --page-index`),
`enable_page_index = true`, and `ParquetSplitConfig { enabled: true }`
(`--compare-split`), all 4 row groups split cleanly into page-aligned slices
(`60` slices for `scan_*`, `32` slices for `count_*`). **All 8/8 workers claim
and complete slices in every single trial** (40/40 split trials across both
pushdown modes; 4/8 in every paired `legacy` trial), and `bytes_scanned` is
**byte-for-byte identical (1.000×)** across `off`, `legacy`, and `split`.

### `pushdown_filters = false`

| Query        | Stock ms, median (range) | Legacy native (`4/8`) ms | Split native (`8/8`) ms | Split vs legacy | Split vs `#156` | Split vs Stock | Active workers (all 5 rounds) | Split claims (round 1)     | `bytes_scanned` (all arms) | Reused bytes | Max peak unique bytes |
| ------------ | -----------------------: | -----------------------: | ----------------------: | --------------: | --------------: | -------------: | ----------------------------: | -------------------------- | -------------------------: | -----------: | --------------------: |
| `scan_low`   |      296.7 (293.6–298.0) |      183.7 (182.9–184.4) |     155.5 (151.4–156.4) |          -15.3% |          -25.1% |         -47.6% |                           8/8 | `[8, 8, 7, 7, 8, 6, 8, 8]` |                 98,214,742 |   74,547,897 |            51,348,748 |
| `count_low`  |      144.5 (143.3–146.6) |         93.0 (91.9–93.5) |        65.8 (63.9–66.1) |          -29.2% |          -39.5% |         -54.4% |                           8/8 | `[3, 4, 5, 4, 4, 4, 4, 4]` |                 37,495,813 |   16,371,853 |            34,873,840 |
| `scan_high`  |      299.8 (299.0–300.7) |      189.5 (188.9–190.2) |     163.6 (158.9–168.3) |          -13.7% |          -23.9% |         -45.4% |                           8/8 | `[7, 8, 7, 7, 7, 8, 8, 8]` |                 98,214,742 |   74,547,897 |            51,582,773 |
| `count_high` |      142.3 (141.4–145.3) |         93.2 (92.6–94.1) |        64.2 (60.2–68.7) |          -31.1% |          -43.0% |         -54.9% |                           8/8 | `[3, 4, 4, 5, 4, 4, 4, 4]` |                 37,495,813 |   16,371,853 |            33,369,323 |

`#138` wait and work metrics (`c2809d3b` → HEAD `legacy` → HEAD `split`,
`pushdown_filters = false`):

| Query        | `ready_wait` ms (`c2809d3b` → `legacy` → `split`) | `output_wait` ms (`split`) | `pending_io` ms (`c2809d3b` → `legacy` → `split`) | `decode` ms (`legacy` → `split`) | `process` ms (`legacy` → `split`) |
| ------------ | ------------------------------------------------: | -------------------------: | ------------------------------------------------: | -------------------------------: | --------------------------------: |
| `scan_low`   |              422.4 → 229.8 → 56.3 (-7.5× / -4.1×) |                        0.0 |                                62.7 → 64.1 → 93.8 |            492.4 → 939.4 (1.91×) |                       74.2 → 87.3 |
| `count_low`  |               204.4 → 93.0 → 38.6 (-5.3× / -2.4×) |                        0.0 |                                24.9 → 25.1 → 38.5 |            232.7 → 325.4 (1.40×) |                       54.3 → 59.9 |
| `scan_high`  |              425.7 → 230.5 → 56.8 (-7.5× / -4.1×) |                       15.9 |                                62.5 → 64.1 → 89.8 |            493.2 → 930.4 (1.89×) |                      95.0 → 121.9 |
| `count_high` |               205.3 → 94.6 → 37.8 (-5.4× / -2.5×) |                        0.0 |                                24.7 → 25.8 → 34.5 |            233.6 → 320.2 (1.37×) |                       54.6 → 56.9 |

On `count_*`, snapping boundaries to `delta`'s ~273 Ki-row pages yields `32`
slices with only `1.37–1.40×` aggregate CPU decode work across twice as many
active workers (`4 → 8`), delivering **29.2–31.1%** wall-time wins over legacy
native, **39.5–43.0%** over `#156`, and **2.19–2.22×** over stock. On `scan_*`,
snapping to `payload`'s ~140 Ki-row pages yields `60` slices where `score`,
`delta`, and `row_id` straddle slice boundaries (`71.1 MiB` served from the
`AsyncRangeBroker` with zero duplicate storage I/O, `1.89–1.91×` CPU decode
work), yielding **13.7–15.3%** wins over legacy native, **23.9–25.1%** over
`#156`, and **1.83–1.91×** over stock. Peak unique compressed residency stays
at `31.8–49.2 MiB`, well under `#167`'s `256 MiB` cap.

### `pushdown_filters = true`

| Query        | Stock ms, median (range) | Legacy native (`4/8`) ms | Split native (`8/8`) ms | Split vs legacy | Split vs Stock | Active workers (all 5 rounds) | Split claims (round 1)      | `bytes_scanned` (all arms) | Max peak unique bytes |
| ------------ | -----------------------: | -----------------------: | ----------------------: | --------------: | -------------: | ----------------------------: | --------------------------- | -------------------------: | --------------------: |
| `scan_low`   |      335.1 (333.8–336.0) |      180.3 (179.3–180.9) |     176.2 (173.6–184.7) |           -2.2% |         -47.4% |                           8/8 | `[7, 9, 6, 11, 7, 7, 6, 7]` |                 98,214,742 |            33,576,650 |
| `count_low`  |      162.8 (159.4–171.2) |         90.8 (90.1–91.5) |        69.3 (64.4–73.4) |          -23.6% |         -57.4% |                           8/8 | `[4, 4, 4, 4, 4, 4, 3, 5]`  |                 37,495,813 |            20,519,192 |
| `scan_high`  |      351.1 (350.0–353.2) |      186.0 (185.8–187.5) |     193.3 (183.7–195.6) |           +3.9% |         -45.0% |                           8/8 | `[8, 8, 6, 5, 7, 9, 9, 8]`  |                 98,214,742 |            31,308,761 |
| `count_high` |      162.6 (161.2–164.2) |         92.0 (92.0–92.8) |        84.5 (80.3–89.4) |           -8.2% |         -48.1% |                           8/8 | `[5, 4, 4, 4, 4, 4, 3, 4]`  |                 37,495,813 |            20,959,245 |

`#138` wait and work metrics (`legacy` → `split`, `pushdown_filters = true`):

| Query        | `ready_wait` ms (`legacy` → `split`) | `output_wait` ms (`split`) | `pending_io` ms (`legacy` → `split`) | `decode` ms (`legacy` → `split`) | `process` ms (`legacy` → `split`) |
| ------------ | -----------------------------------: | -------------------------: | -----------------------------------: | -------------------------------: | --------------------------------: |
| `scan_low`   |                 490.2 → 81.9 (-6.0×) |                        0.0 |                         67.6 → 107.3 |          587.8 → 1,146.4 (1.95×) |                         0.8 → 1.2 |
| `count_low`  |                 396.4 → 74.9 (-5.3×) |                        0.0 |                          25.4 → 53.6 |            292.0 → 421.7 (1.44×) |                         0.0 → 0.0 |
| `scan_high`  |                 488.3 → 70.7 (-6.9×) |                        5.6 |                          66.8 → 77.2 |          600.2 → 1,245.8 (2.08×) |                        6.7 → 11.5 |
| `count_high` |                 394.6 → 78.8 (-5.0×) |                        0.0 |                          25.8 → 86.6 |            294.1 → 544.4 (1.85×) |                         0.0 → 0.1 |

With `pushdown_filters = true`, splitting reduces `morsel_ready_wait` by
**5.0–6.9×** and engages all 8 workers across both Phase-1 filtering and
Phase-2 decode (3–11 claims per worker on `scan_*`, 3–5 on `count_*`), keeping
native **1.82–2.35× faster than stock**. As observed on `dduck1` in
[issue-167.md](issue-167.md), `count_low` wins cleanly (`-23.6%` vs legacy) and
`count_high` improves on median (`-8.2%`), whereas on 4-column `scan_*` the
per-slice `RowFilter` + `RowGroupCache` boundary-page re-decompression roughly
doubles `morsel_decode_work` (`1.95–2.08×`), offsetting the `4 → 8` worker
concurrency gain (`-2.2%` on `scan_low`, `+3.9%` on `scan_high` vs legacy).

## Reproduction

Run from the repository root on `rob-tandy-dduck2`:

```bash
set -e
b=benchmarks/morsel_parquet_skew
work=$(mktemp -d "$HOME/.cache/issue172.XXXXXX")
head_rev=04fa5fc6456f62ee40fe25311e70894e7deca967

git worktree add --detach "$work/head" "$head_rev"
mkdir -p "$work/head/datafusion/core/examples"
cp "$b/stream_runner.rs" "$work/head/datafusion/core/examples/morsel_parquet_skew.rs"
(
  cd "$work/head"
  MORSEL_BENCH_REVISION="$head_rev" CARGO_TARGET_DIR="$work/target-head" \
    timeout --kill-after=20s 35m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql --example morsel_parquet_skew
)
runner="$work/target-head/release-nonlto/examples/morsel_parquet_skew"
uv venv --python 3.12.10 "$work/venv"
uv pip install --python "$work/venv/bin/python" -r benchmarks/morsel_parquet/requirements.txt
py="$work/venv/bin/python"

# Verify skew harness unit and smoke tests
MORSEL_PARQUET_RUNNER="$runner" MORSEL_BENCH_REVISION="$head_rev" \
  timeout 120s taskset -c 0-7 "$py" -B -m unittest discover -s "$b" -v

# Generate fixtures
mkdir -p "$work/fixtures" "$work/results"
"$py" -B "$b/bench.py" generate file-skew "$work/fixtures/file-skew"
"$py" -B "$b/bench.py" generate file-skew "$work/fixtures/file-skew-indexed" --page-index
"$py" -B "$b/bench.py" generate large-row-groups "$work/fixtures/large-row-groups"
"$py" -B "$b/bench.py" generate large-row-groups "$work/fixtures/large-row-groups-indexed" --page-index

# 1. Exact #156 file-skew (no page index): pushdown_filters=false and true
taskset -c 0-7 "$py" -B "$b/bench.py" ab file-skew "$work/fixtures/file-skew" \
  --runner "$runner" --revision "$head_rev" --warmups 1 --repeats 5 \
  > "$work/results/head_file-skew_pd0.jsonl"
taskset -c 0-7 "$py" -B "$b/bench.py" ab file-skew "$work/fixtures/file-skew" \
  --runner "$runner" --revision "$head_rev" --pushdown-filters --warmups 1 --repeats 5 \
  > "$work/results/head_file-skew_pd1.jsonl"

# 2. Indexed file-skew control (0 split RGs)
taskset -c 0-7 "$py" -B "$b/bench.py" ab file-skew "$work/fixtures/file-skew-indexed" \
  --runner "$runner" --revision "$head_rev" --enable-page-index --compare-split \
  --warmups 1 --repeats 5 > "$work/results/head_file-skew-indexed_pd0.jsonl"
taskset -c 0-7 "$py" -B "$b/bench.py" ab file-skew "$work/fixtures/file-skew-indexed" \
  --runner "$runner" --revision "$head_rev" --enable-page-index --pushdown-filters \
  --compare-split --warmups 1 --repeats 5 > "$work/results/head_file-skew-indexed_pd1.jsonl"

# 3. Exact #156 large-row-groups (no page index): fallback verification
taskset -c 0-7 "$py" -B "$b/bench.py" ab large-row-groups "$work/fixtures/large-row-groups" \
  --runner "$runner" --revision "$head_rev" --enable-page-index --enable-split \
  --warmups 1 --repeats 5 > "$work/results/head_lrg-noindex_pd0.jsonl"
taskset -c 0-7 "$py" -B "$b/bench.py" ab large-row-groups "$work/fixtures/large-row-groups" \
  --runner "$runner" --revision "$head_rev" --enable-page-index --enable-split \
  --pushdown-filters --warmups 1 --repeats 5 > "$work/results/head_lrg-noindex_pd1.jsonl"

# 4. Indexed large-row-groups: stock vs legacy (4/8) vs split (8/8)
taskset -c 0-7 "$py" -B "$b/bench.py" ab large-row-groups "$work/fixtures/large-row-groups-indexed" \
  --runner "$runner" --revision "$head_rev" --enable-page-index --compare-split \
  --warmups 1 --repeats 5 > "$work/results/head_lrg-indexed_pd0.jsonl"
taskset -c 0-7 "$py" -B "$b/bench.py" ab large-row-groups "$work/fixtures/large-row-groups-indexed" \
  --runner "$runner" --revision "$head_rev" --enable-page-index --pushdown-filters \
  --compare-split --warmups 1 --repeats 5 > "$work/results/head_lrg-indexed_pd1.jsonl"
```

To reproduce the `c2809d3b` `#138` wait-metric baseline table in Section 1,
check out `c2809d3b3ee73936b182429e1646af290d797c6e` in a second detached
worktree, copy `stream_runner.rs` with the `ParquetSplitConfig` lines removed,
build with `MORSEL_BENCH_REVISION=c2809d3b3ee73936b182429e1646af290d797c6e`, and
run `bench.py ab` on `$work/fixtures/file-skew` and
`$work/fixtures/large-row-groups`.
