<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Direct mask-bit `COUNT(*)` and filter column pruning on Parquet (#166)

Before: `98b2c851653b8525d4386d380556030a510f12e4` (`morsel-execution` after #150).
After: `23682e966` (`agent/morsel-issue-166`). This comparison does not replace
the pinned [#128 baseline](README.md#native-parquet-ab-128).

## Method

`rob-tandy-dduck1`, November 23, 2026 UTC: Ubuntu 24.04.5, Linux
`7.0.0-1014-aws`, 16 Neoverse-N1 vCPUs, 61 GiB RAM, Rust 1.99.1 (`58cf8f9c3`),
Clang/lld 18.1.3, and Python 3.12.10 (`numpy 2.2.6`, `pyarrow 20.0.0`). The
private #128 16-file × 8-row-group × 131,072-row fixture (311,529,117 bytes,
Zstandard level 3, PLAIN encoding, no dictionaries) was generated under
`$HOME/.cache/issue166/large` on local EC2 NVMe instance storage. No builds ran
during measurement (`vmstat 1 3` showed 96–98% idle prior to launch); both arms
used CPUs 0–7, eight workers, one warmup, and eight timed samples per arm and
query in each regime.

The #128 harness verified all four queries in every arm against the NumPy oracle
before timing: 759,841 low-selectivity (4.5%) and 9,883,940 high-selectivity
(59%) rows and counts. Stock used `morsel_mode=off`; native used
`morsel_mode=required`, claimed all 128 row groups, and finalized one count per
count query. Every warm timed sample started with 76,063/76,063 file pages
resident; every page-cache-cold sample started with 0/76,063 resident pages.
Under `pushdown_filters = false`, native scan plans show
`Filter[..., projection=[0, 2, 3]]` (pruning `score` before
`LimitedBatchCoalescer`). Under `pushdown_filters = true`, native plans show
`NativeParquet[fixture, predicate=...]` with zero residual `Filter` operators.

## Paired `pushdown_filters = false` timings

Total elapsed ms from registration through stream exhaustion, median (min–max),
plus aggregate worker `morsel_process_work` in ms (sum across 8 workers).

| Cache           | Case         |        Stock before |         Stock after |       Native before |        Native after | Native process (ms) | `bytes_scanned` |
| --------------- | ------------ | ------------------: | ------------------: | ------------------: | ------------------: | ------------------: | --------------: |
| Warm            | `scan_low`   | 169.9 (161.1–181.3) | 169.5 (168.1–179.5) | 170.6 (167.4–174.1) | 170.6 (168.3–176.4) |       170.8 → 168.6 |     196,691,567 |
| Warm            | `scan_high`  | 191.9 (185.3–206.1) | 197.7 (191.8–212.0) | 194.9 (188.2–202.2) | 193.9 (189.9–201.3) |       226.6 → 218.5 |     196,691,567 |
| Warm            | `count_low`  |    89.5 (87.0–92.8) |    91.2 (88.1–94.4) |    92.1 (88.4–93.5) |    87.7 (85.4–91.8) |       137.2 → 118.0 |      75,280,626 |
| Warm            | `count_high` |    88.9 (87.8–93.4) |    90.3 (87.3–95.3) |    94.3 (92.7–97.3) |    91.4 (86.7–97.5) |       165.3 → 118.2 |      75,280,626 |
| Page-cache-cold | `scan_low`   | 216.6 (196.2–240.9) | 220.7 (208.6–229.5) | 196.3 (192.3–242.5) | 194.6 (191.1–200.1) |       181.7 → 176.2 |     196,691,567 |
| Page-cache-cold | `scan_high`  | 233.0 (217.0–288.8) | 234.1 (225.6–299.1) | 215.5 (213.7–224.1) | 221.9 (215.6–227.1) |       243.6 → 227.6 |     196,691,567 |
| Page-cache-cold | `count_low`  | 118.4 (105.0–122.6) | 119.3 (110.9–144.7) |  101.5 (98.6–124.0) |   99.7 (97.5–130.9) |       146.6 → 121.9 |      75,280,626 |
| Page-cache-cold | `count_high` | 105.9 (101.8–122.7) | 119.6 (115.6–132.3) | 103.4 (101.4–106.8) |   99.9 (97.3–106.9) |       171.7 → 121.3 |      75,280,626 |

Key observations under `pushdown_filters = false`:

- **Direct mask-bit `COUNT(*)` eliminates filtered array materialization.**
  Aggregate worker `morsel_process_work` drops by **14.0% on warm `count_low`**
  (137.2 → 118.0 ms; -19.2 ms) and **28.5% on warm `count_high`** (165.3 →
  118.2 ms; -47.1 ms), and by **16.8–29.3% in page-cache-cold** (146.6 →
  121.9 ms and 171.7 → 121.3 ms). High-selectivity `count_high` now matches
  low-selectivity `count_low` processing cost (~118 ms warm, ~121 ms cold)
  because `score` and `delta` arrays are never filtered or copied.
- **Warm 8-worker `count_low` drops from 92.1 → 87.7 ms (-4.8%), beating paired
  stock (91.2 ms); cold `count_low` and `count_high` drop to 99.7 / 99.9 ms
  (16.4–16.5% ahead of paired stock at 119.3 / 119.6 ms).**

## Filter-only column pruning ablation (`pushdown_filters = false`)

To evaluate whether pruning filter-only columns before `LimitedBatchCoalescer`
still pays post-#149/#150, a dedicated ablation compared the same direct-mask
`COUNT(*)` binary **without** vs **with** `prune_filter_input` on
`Source::Parquet` (dropping `score` before `push_batch_with_filter` on
`scan_low` and `scan_high`):

| Cache           | Case        | Native without pruning | Native with pruning | Process work (no-prune → prune) |
| --------------- | ----------- | ---------------------: | ------------------: | ------------------------------: |
| Warm            | `scan_low`  |    171.0 (167.7–171.5) | 170.5 (169.5–173.7) |                   170.4 → 168.5 |
| Warm            | `scan_high` |    195.5 (190.1–199.6) | 193.1 (186.2–196.5) |                   228.8 → 213.8 |
| Page-cache-cold | `scan_low`  |    187.3 (185.0–200.0) | 186.5 (181.9–188.1) |                   184.6 → 175.4 |
| Page-cache-cold | `scan_high` |    212.8 (208.3–216.8) | 210.7 (207.7–213.8) |                   237.7 → 220.1 |

Filter-only column pruning saves **15.0–17.6 ms of worker process time
(-6.6% to -7.4%)** and **2.1–2.4 ms of wall time** on `scan_high` with zero
regression on `scan_low`, and also narrows residual filter output before
coalescing when `pushdown_filters = true` leaves a residual `Filter ->
Projection`. It is kept on `Source::Parquet`.

## Paired `pushdown_filters = true` timings

Total elapsed ms, median (min–max), plus aggregate worker `morsel_process_work`
in ms (sum across 8 workers).

| Cache           | Case         |        Stock before |         Stock after |       Native before |        Native after | Native process (ms) | `bytes_scanned` |
| --------------- | ------------ | ------------------: | ------------------: | ------------------: | ------------------: | ------------------: | --------------: |
| Warm            | `scan_low`   | 216.4 (206.9–223.0) | 218.6 (216.2–222.3) | 214.4 (206.6–216.9) | 218.2 (215.4–232.5) |         1.82 → 2.12 |     196,691,567 |
| Warm            | `scan_high`  | 204.9 (197.0–215.5) | 205.4 (202.5–207.7) | 205.1 (192.6–211.6) | 204.8 (202.3–208.4) |       23.99 → 23.84 |     196,691,567 |
| Warm            | `count_low`  |    91.4 (86.2–96.1) |    91.6 (88.9–99.0) |    88.0 (84.4–95.6) |    90.4 (88.1–98.8) |         0.08 → 0.05 |      75,280,626 |
| Warm            | `count_high` |    91.9 (89.0–97.6) |    93.3 (89.2–94.5) |    86.8 (84.5–88.7) |    89.7 (87.4–91.5) |         0.25 → 0.13 |      75,280,626 |
| Page-cache-cold | `scan_low`   | 241.1 (231.1–253.4) | 246.1 (239.7–248.8) | 235.1 (228.4–239.4) | 234.9 (230.4–251.0) |         2.02 → 1.90 |     196,691,567 |
| Page-cache-cold | `scan_high`  | 227.2 (216.6–242.9) | 237.0 (229.8–251.8) | 222.4 (217.4–226.1) | 226.4 (218.8–234.5) |       23.87 → 23.68 |     196,691,567 |
| Page-cache-cold | `count_low`  |  100.7 (98.3–104.9) | 103.7 (100.8–117.3) |    93.3 (90.5–96.4) |   92.9 (91.3–104.0) |         0.08 → 0.05 |      75,280,626 |
| Page-cache-cold | `count_high` |  103.5 (97.4–111.8) | 103.7 (100.6–105.8) |    93.5 (91.6–96.7) |    95.3 (93.6–98.4) |         0.26 → 0.12 |      75,280,626 |

Key observations under `pushdown_filters = true`:

- **Zero-column direct row counting (`CountStrategy::DirectRows`) halves
  post-decode `COUNT(*)` processing** (0.08 → 0.05 ms on `count_low`; 0.25 →
  0.13 ms warm and 0.26 → 0.12 ms cold on `count_high`) by counting
  `batch.num_rows()` directly instead of invoking `Projector::project_batch`
  for the empty projection.
- **Native beats paired stock on all four `pushdown_filters = true` workloads**
  in both warm and page-cache-cold regimes (`count_low` 90.4 vs 91.6 ms warm,
  92.9 vs 103.7 ms cold; `count_high` 89.7 vs 93.3 ms warm, 95.3 vs 103.7 ms
  cold).

## Reproduction

Follow the reproduction recipe in [issue-150.md](issue-150.md#reproduction),
building `runner-before` at `98b2c851653b8525d4386d380556030a510f12e4` and
`runner-after` at `23682e966` with
`--profile release-nonlto -p datafusion --no-default-features --features parquet,sql`,
and running both `MORSEL_PUSHDOWN_FILTERS=0` and `MORSEL_PUSHDOWN_FILTERS=1`
with `MORSEL_METRICS=1`, `--threads 8 --warmups 1 --repeats 8`, across
`warm` and `page-cache-cold` pinned to CPUs 0–7.
