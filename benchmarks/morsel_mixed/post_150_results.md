<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Mixed-query Parquet tail-latency comparison after #148–#150 (#171)

Engine revision under test: `98b2c851653b8525d4386d380556030a510f12e4`
(`morsel-execution` after #148 producer scaling, #149 batch coalescing and
scaled result channel, and #150 Parquet row-level filter pushdown). Compared
against the pinned [#139 baseline](README.md#pinned-parquet-comparison-139) at
`d649cacf00fee7125f45ea1427a023b2eab732b1`. This note does not replace the
pinned #139 baseline in [README.md](README.md).

## Summary

- **Short-query tail latency improves substantially under `pushdown_filters = false`
  (apples-to-apples with #139).** At lossless 20 offers/s, native mixed p99
  drops from **19.76 ms → 11.58 ms** (**1.71× lower** than #139 and **9.52×
  lower** than stock's 110.26 ms). At 50 offers/s, native keeps 100% completion
  with mixed p99 falling from **20.30 ms → 11.19 ms** (**1.81× lower** than
  #139; stock completes 76.38% with 621.77 ms p99). At 200 offers/s, native
  completion nearly doubles from **43.75% → 83.25%** (161.20 vs 83.19 req/s;
  stock 20.79% / 39.29 req/s) while mixed p99 drops from **231.67 ms →
  119.71 ms** (**1.94× lower**).
- **Background scan throughput does not overtake stock under 3-lane CPU
  saturation.** With `pushdown_filters = false`, native background throughput is
  unchanged at **6.00 vs 6.55 scans/s** at 20 offers/s (−8.4%), **5.79 vs 6.46**
  at 50 offers/s (−10.4%, while serving 32% more foreground queries/s), and
  **5.00 vs 6.17** at 200 offers/s (−19.0%, while serving 4.1× more foreground
  queries/s). With `pushdown_filters = true`, background throughput is **6.05 vs
  6.50** at 20 offers/s (−6.9%), **5.75 vs 6.38** at 50 offers/s (−9.9%), and
  **5.00 vs 6.00** at 200 offers/s (−16.7%). Separate #138 metrics show why:
  under 3 concurrent background lanes on 4 CPUs, `morsel_ready_wait` is already
  tiny (4.5 ms summed over 4 workers) and the 4 CPUs are saturated by
  ~606 ms-CPU/scan of decode + filter work; the remaining ~8% gap vs stock is
  the cost of fine-grained cooperative yielding and channel coordination that
  delivers the 9.5× foreground p99 reduction.
- **Tail-latency regression flagged for `#150` (`pushdown_filters = true`), NOT
  for `#148` or `#149`:**
  - **#148 (producer scaling):** no regression — drove the ~2× drop in mixed
    foreground `service_time` (`11.96 → 6.20 ms` at 20 offers/s) because the
    4-file `small` query now reads all 4 files concurrently instead of
    serializing through one producer lane.
  - **#149 (batch coalescing + scaled result channel):** no regression observed
    (note: both #139 queries use `CountSink` with `peak_output_depth = 1` and
    `output_wait = 0`, so `ResultSink` batch coalescing is not stressed by this
    workload).
  - **#150 (`pushdown_filters = true`):** native still beats stock by **5.53×**
    at 20 offers/s (`22.28 vs 123.14 ms` p99), **30.4×** at 50 offers/s (`22.53
vs 683.98 ms`, 100% vs 69.25% completion), and **3.98×** at 200 offers/s
    (`175.05 vs 696.70 ms`, 62.92% vs 18.29% completion), **but regresses ~2×
    vs `pushdown_filters = false`** (`11.58 → 22.28 ms` at 20/s; `11.19 →
22.53 ms` at 50/s; `119.71 → 175.05 ms` at 200/s, with 200/s completion
    falling from `83.25% → 62.92%`). Root cause: in
    `datafusion/core/src/morsel.rs`, `ClaimedWork::FilterStep` runs
    `planner.plan()` synchronously over an entire 131,072-row row group
    (~4.8 ms uninterrupted CPU time per row group per #138 `decode_work`)
    without yielding, a **16× coarser** cooperative quantum than the ~0.30 ms
    per 8,192-row batch in `pushdown_filters = false`. This appears directly as
    a **12× jump** in lossless mixed foreground `queue_wait` p50 (`0.06 →
0.73 ms` at 20 offers/s with 0 queued requests — pure Tokio worker wake-up
    starvation). Fairer intra-row-group yielding during `RowFilter` evaluation
    or a dedicated CPU pool / priority scheduling is warranted if
    `pushdown_filters = true` is enabled under latency-sensitive mixed loads.

## Method

`rob-tandy-dduck2`, 2026-11-23 UTC: Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
16 aarch64 Neoverse-N1 vCPUs, 61 GiB RAM, Rust 1.99.1 (`58cf8f9c3`), Clang/lld
18.1.3, Python 3.12.10 (`numpy 2.2.6`, `pyarrow 20.0.0`). Both backends used a
single locked release build of `benchmarks/morsel_mixed`
(`CARGO_PROFILE_RELEASE_OPT_LEVEL=2`, matching #139) with
`MORSEL_BENCH_REVISION=98b2c851653b8525d4386d380556030a510f12e4`, CPUs 12–15,
four Tokio workers, up to two blocking threads, four target partitions, 8,192-row
batches, and three continuously reissuing background lanes.

Fixtures were generated under `$HOME/.cache/issue171` on Btrfs on the local
NVMe instance-store volume with the exact #130/#139 shapes and byte counts:

- `small`: 4 files, 8 row groups, 32,768 rows, 627,668 bytes,
  `count_low` (`(score + 3 * delta) % 1000 < 50`, expected scalar `1,456`).
- `large`: 16 files, 128 row groups, 16,777,216 rows, 311,529,117 bytes,
  `count_high` (`(score + 3 * delta) % 1000 < 650`, expected scalar `9,883,940`).

Every arm verified 76,219 / 76,219 4 KiB file pages resident in the Linux page
cache via `mincore` both before and immediately after execution. Three
one-second `vmstat` samples before each comparison showed 87–93% whole-host
CPU idle with no active builds or competing benchmark processes; stock `off`
with `pushdown_filters = false` matched #139's stock latencies and throughput
within 1–5% across all three offer rates.

Each configuration ran four counterbalanced pairs per load level: 200 offers
per phase at 20 offers/s (`--interval-ms 50`), and 600 offers per phase at 50
(`--interval-ms 20`) and 200 offers/s (`--interval-ms 5`). Every arm passed
scalar parity checks and physical-plan verification:

- `off`: `DataSourceExec: ... file_type=parquet`, no `MorselExec`.
- `required`, `pushdown_filters = false`:
  `MorselExec: workers=4, NativeParquet[table] -> Filter[...] -> Projection[] -> CountSink`.
- `required`, `pushdown_filters = true`:
  `MorselExec: workers=4, NativeParquet[table, predicate=...] -> Projection[] -> CountSink`.

All timings and rates below are **medians of the four runs, with min–max across
runs** (omitting the range when all four runs were identical), matching #139.
Completion and drop counts are summed across all four runs. Across the 48 main
trials, zero queries failed; 8,320 of 8,330 completed mixed foreground queries
under `pushdown_filters = false` and 7,596 of 7,611 under
`pushdown_filters = true` started execution while at least one background scan
was in flight.

## Results: `pushdown_filters = false` (apples-to-apples with #139)

### Latency and completion/drop rates

| Offers/s | Phase    | Backend    | Completed (% of offers) | Dropped (% of offers) |                 p50 ms |                 p95 ms |                 p99 ms |
| -------: | :------- | :--------- | ----------------------: | --------------------: | ---------------------: | ---------------------: | ---------------------: |
|       20 | Isolated | `off`      |           800 (100.00%) |             0 (0.00%) |       2.69 (2.67–2.72) |       3.16 (3.00–3.21) |       3.62 (3.29–3.83) |
|       20 | Isolated | `required` |           800 (100.00%) |             0 (0.00%) |       2.26 (2.19–2.32) |       2.55 (2.42–2.67) |       2.88 (2.59–3.34) |
|       20 | Mixed    | `off`      |           800 (100.00%) |             0 (0.00%) |    40.58 (38.66–41.82) |    92.98 (80.86–97.19) |  110.26 (93.80–117.91) |
|       20 | Mixed    | `required` |           800 (100.00%) |             0 (0.00%) |       6.43 (5.99–6.50) |      9.54 (9.06–10.39) |    11.58 (11.01–12.97) |
|       50 | Isolated | `off`      |         2,400 (100.00%) |             0 (0.00%) |       2.51 (2.48–2.54) |       2.87 (2.78–2.95) |       3.40 (3.08–3.48) |
|       50 | Isolated | `required` |         2,400 (100.00%) |             0 (0.00%) |       2.13 (2.10–2.14) |       2.44 (2.39–2.50) |       3.20 (2.97–3.42) |
|       50 | Mixed    | `off`      |          1,833 (76.38%) |          567 (23.62%) | 409.59 (396.11–421.30) | 552.73 (525.03–606.51) | 621.77 (601.57–685.03) |
|       50 | Mixed    | `required` |         2,400 (100.00%) |             0 (0.00%) |       6.14 (6.05–6.22) |      9.61 (9.40–10.22) |    11.19 (11.09–12.48) |
|      200 | Isolated | `off`      |         2,400 (100.00%) |             0 (0.00%) |       2.24 (2.19–2.25) |       2.46 (2.41–2.54) |       2.64 (2.57–4.62) |
|      200 | Isolated | `required` |         2,400 (100.00%) |             0 (0.00%) |       1.83 (1.77–2.00) |       2.17 (2.03–2.25) |       2.56 (2.25–2.78) |
|      200 | Mixed    | `off`      |            499 (20.79%) |        1,901 (79.21%) | 430.56 (398.04–474.00) | 563.24 (542.32–588.07) | 598.01 (591.24–610.45) |
|      200 | Mixed    | `required` |          1,998 (83.25%) |          402 (16.75%) |  101.26 (97.39–103.33) | 112.84 (111.74–114.29) | 119.71 (116.95–123.47) |

### Throughput

| Offers/s | Backend    | Isolated req/s |            Mixed req/s | Mixed bg scans/s | Mixed bg million input rows/s |
| -------: | :--------- | -------------: | ---------------------: | ---------------: | ----------------------------: |
|       20 | `off`      |          20.00 |    19.99 (19.97–20.00) | 6.55 (6.50–6.70) |        109.89 (109.05–112.41) |
|       20 | `required` |          20.00 |                  20.00 |             6.00 |                        100.66 |
|       50 | `off`      |          50.00 |    37.86 (36.39–38.83) | 6.46 (6.33–6.50) |        108.35 (106.26–109.05) |
|       50 | `required` |          50.00 |                  50.00 | 5.79 (5.75–5.92) |           97.17 (96.47–99.27) |
|      200 | `off`      |         200.00 |    39.29 (36.63–41.20) | 6.17 (6.00–6.33) |        103.46 (100.66–106.26) |
|      200 | `required` |         200.00 | 161.20 (158.17–164.76) |             5.00 |                         83.89 |

### Direct comparison vs #139 pinned baseline (`d649cacf0`, mixed phase)

| Offers/s | Backend    | Completion (#139 → post-#150) |  p50 ms (#139 → post-#150) |  p99 ms (#139 → post-#150) | Bg scans/s (#139 → post-#150) |
| -------: | :--------- | ----------------------------: | -------------------------: | -------------------------: | ----------------------------: |
|       20 | `off`      |               100.0% → 100.0% |              43.81 → 40.58 |            111.79 → 110.26 |                   6.60 → 6.55 |
|       20 | `required` |               100.0% → 100.0% |    **12.20 → 6.43** (−47%) |   **19.76 → 11.58** (−41%) |                   6.00 → 6.00 |
|       50 | `off`      |                 74.2% → 76.4% |            440.37 → 409.59 |            653.88 → 621.77 |                   6.50 → 6.46 |
|       50 | `required` |               100.0% → 100.0% |    **12.07 → 6.14** (−49%) |   **20.30 → 11.19** (−45%) |                   5.75 → 5.79 |
|      200 | `off`      |                 21.0% → 20.8% |            424.98 → 430.56 |            627.16 → 598.01 |                   6.17 → 6.17 |
|      200 | `required` |  **43.8% → 83.3%** (+39.5 pp) | **201.41 → 101.26** (−50%) | **231.67 → 119.71** (−48%) |                   5.00 → 5.00 |

Stock `off` is stable across revisions, while native `required` cuts mixed
p50/p95/p99 nearly in half at every offer rate and nearly doubles 200 offers/s
foreground throughput (83.19 → 161.20 req/s) without reducing background scan
throughput.

## Results: `pushdown_filters = true` (#150 enabled on both backends)

### Latency and completion/drop rates

| Offers/s | Phase    | Backend    | Completed (% of offers) | Dropped (% of offers) |                 p50 ms |                 p95 ms |                 p99 ms |
| -------: | :------- | :--------- | ----------------------: | --------------------: | ---------------------: | ---------------------: | ---------------------: |
|       20 | Isolated | `off`      |           800 (100.00%) |             0 (0.00%) |       2.75 (2.70–2.84) |       3.15 (2.96–3.59) |       3.61 (3.48–4.56) |
|       20 | Isolated | `required` |           800 (100.00%) |             0 (0.00%) |       2.30 (2.28–2.31) |       2.46 (2.44–2.79) |       2.89 (2.70–3.10) |
|       20 | Mixed    | `off`      |           800 (100.00%) |             0 (0.00%) |    46.79 (42.75–47.39) |   98.91 (92.90–107.34) | 123.14 (117.75–130.65) |
|       20 | Mixed    | `required` |           800 (100.00%) |             0 (0.00%) |       8.65 (8.56–8.70) |    17.52 (15.78–18.53) |    22.28 (17.89–23.60) |
|       50 | Isolated | `off`      |         2,400 (100.00%) |             0 (0.00%) |       2.54 (2.50–2.60) |       2.84 (2.78–3.06) |       3.15 (2.88–3.70) |
|       50 | Isolated | `required` |         2,400 (100.00%) |             0 (0.00%) |       2.18 (2.13–2.22) |       2.39 (2.37–2.42) |       2.59 (2.53–2.92) |
|       50 | Mixed    | `off`      |          1,662 (69.25%) |          738 (30.75%) | 469.18 (460.97–480.43) | 639.63 (601.14–650.79) | 683.98 (667.38–692.21) |
|       50 | Mixed    | `required` |         2,400 (100.00%) |             0 (0.00%) |       8.66 (8.05–8.81) |    17.13 (15.92–18.91) |    22.53 (20.61–22.99) |
|      200 | Isolated | `off`      |         2,400 (100.00%) |             0 (0.00%) |       2.27 (2.24–2.33) |       2.50 (2.50–2.65) |       2.67 (2.65–3.04) |
|      200 | Isolated | `required` |         2,400 (100.00%) |             0 (0.00%) |       1.87 (1.87–1.89) |       2.14 (2.11–2.19) |       2.42 (2.24–2.66) |
|      200 | Mixed    | `off`      |            439 (18.29%) |        1,961 (81.71%) | 491.26 (462.85–494.49) | 664.65 (634.35–696.49) | 696.70 (666.02–719.30) |
|      200 | Mixed    | `required` |          1,510 (62.92%) |          890 (37.08%) | 132.37 (129.46–138.72) | 166.99 (161.69–170.86) | 175.05 (172.43–194.61) |

### Throughput

| Offers/s | Backend    | Isolated req/s |            Mixed req/s | Mixed bg scans/s | Mixed bg million input rows/s |
| -------: | :--------- | -------------: | ---------------------: | ---------------: | ----------------------------: |
|       20 | `off`      |          20.00 |    19.99 (19.96–20.00) | 6.50 (6.30–6.50) |        109.05 (105.70–109.05) |
|       20 | `required` |          20.00 |                  20.00 | 6.05 (6.00–6.30) |        101.50 (100.66–105.70) |
|       50 | `off`      |          50.00 |    34.24 (33.48–34.88) | 6.38 (6.25–6.42) |        106.95 (104.86–107.65) |
|       50 | `required` |          50.00 |                  50.00 | 5.75 (5.75–6.17) |          96.47 (96.47–103.46) |
|      200 | `off`      |         200.00 |    34.52 (33.95–38.39) |             6.00 |                        100.66 |
|      200 | `required` |         200.00 | 120.74 (117.78–124.46) | 5.00 (5.00–5.67) |           83.89 (83.89–95.07) |

### Foreground latency decomposition (mixed phase, native `required`)

Medians of per-run p50s in ms:

| Offers/s | Configuration                  | `planning` p50 | `queue_wait` p50 | `service_time` p50 | Total p99 | Queued (per run) | Overdue >20 ms (per run) |
| -------: | :----------------------------- | -------------: | ---------------: | -----------------: | --------: | ---------------: | -----------------------: |
|       20 | #139 (`pushdown = false`)      |           1.02 |             0.06 |              11.96 |     19.76 |          0,0,0,0 |                  1,2,0,4 |
|       20 | Post-#150 (`pushdown = false`) |           1.01 |             0.06 |               6.20 |     11.58 |          0,0,0,0 |                  1,0,0,0 |
|       20 | Post-#150 (`pushdown = true`)  |           1.00 |         **0.73** |           **7.53** | **22.28** |          0,0,0,0 |                  9,3,6,2 |
|       50 | #139 (`pushdown = false`)      |           1.01 |             0.06 |              11.87 |     20.30 |         7,10,9,6 |                 7,10,9,6 |
|       50 | Post-#150 (`pushdown = false`) |           1.00 |             0.06 |               5.94 |     11.19 |          0,0,0,0 |                  0,0,0,0 |
|       50 | Post-#150 (`pushdown = true`)  |           0.98 |         **0.74** |           **7.52** | **22.53** |        14,17,8,9 |                14,18,9,9 |
|      200 | #139 (`pushdown = false`)      |           0.99 |           189.73 |              11.85 |    231.67 |  255,261,268,262 |          255,261,268,262 |
|      200 | Post-#150 (`pushdown = false`) |           0.91 |            94.92 |               5.97 |    119.71 |  490,499,509,496 |          488,492,504,493 |
|      200 | Post-#150 (`pushdown = true`)  |           0.89 |           124.73 |               7.84 |    175.05 |  387,381,368,370 |          388,382,369,371 |

Notice `queue_wait` at 20 offers/s: zero foreground requests were queued behind
another foreground request in any run, so `queue_wait` measures pure Tokio
worker wake-up latency from offer to dequeue. Enabling `pushdown_filters = true`
increases that wake-up median **12×** (`0.06 → 0.73 ms`) because background
workers hold Tokio worker threads inside synchronous whole-row-group
`RowFilter` evaluation (`planner.plan()`, ~4.8 ms per row group) instead of
yielding after every 8,192-row batch (~0.30 ms).

## Separate #138 wait and buffering metrics check

Collected in a separate instrumented run (`--morsel-metrics`, 1 pair per load
level on CPUs 12–15, warm page cache verified at 76,219 / 76,219 pages) so
instrumentation stayed off during the paired timings above. Values are medians
of per-query sums across the 4 workers (overlapping monotonic durations, not
partitions of wall time), with max queue peaks observed across all executions
in the run.

### `pushdown_filters = false`

| Offers/s | Phase    | Role       | Executions | Pending I/O ms | Ready wait ms | Decode work ms | Process work ms | Output wait ms | Morsels | Peak ready depth (bytes) | Peak output depth (bytes) |
| -------: | :------- | :--------- | ---------: | -------------: | ------------: | -------------: | --------------: | -------------: | ------: | -----------------------: | ------------------------: |
|       20 | Isolated | Foreground |        200 |           1.47 |          1.45 |           0.94 |            0.35 |           0.00 |       8 |             4 (0.30 MiB) |                   1 (8 B) |
|       20 | Mixed    | Foreground |        200 |           1.93 |          1.58 |           0.93 |            0.32 |           0.00 |       8 |             4 (0.30 MiB) |                   1 (8 B) |
|       20 | Mixed    | Background |         63 |          55.78 |          4.56 |         446.53 |          159.26 |           0.00 |     128 |             4 (9.32 MiB) |                   1 (8 B) |
|       50 | Isolated | Foreground |        600 |           1.33 |          1.38 |           0.95 |            0.35 |           0.00 |       8 |             4 (0.30 MiB) |                   1 (8 B) |
|       50 | Mixed    | Foreground |        600 |           1.90 |          1.58 |           0.96 |            0.32 |           0.00 |       8 |             4 (0.30 MiB) |                   1 (8 B) |
|       50 | Mixed    | Background |         72 |          58.86 |          4.53 |         456.70 |          163.41 |           0.00 |     128 |             4 (9.32 MiB) |                   1 (8 B) |
|      200 | Isolated | Foreground |        600 |           1.24 |          1.26 |           0.91 |            0.33 |           0.00 |       8 |             4 (0.30 MiB) |                   1 (8 B) |
|      200 | Mixed    | Foreground |        496 |           2.27 |          1.58 |           0.92 |            0.31 |           0.00 |       8 |             4 (0.30 MiB) |                   1 (8 B) |
|      200 | Mixed    | Background |         18 |          65.33 |          5.54 |         454.31 |          163.85 |           0.00 |     128 |             4 (9.32 MiB) |                   1 (8 B) |

### `pushdown_filters = true`

| Offers/s | Phase    | Role       | Executions | Pending I/O ms | Ready wait ms | Decode work ms | Process work ms | Output wait ms | Morsels | Peak ready depth (bytes) | Peak output depth (bytes) |
| -------: | :------- | :--------- | ---------: | -------------: | ------------: | -------------: | --------------: | -------------: | ------: | -----------------------: | ------------------------: |
|       20 | Isolated | Foreground |        200 |           1.63 |          1.83 |           1.51 |            0.00 |           0.00 |       8 |             4 (0.07 MiB) |                   1 (8 B) |
|       20 | Mixed    | Foreground |        200 |           1.75 |          0.67 |           1.50 |            0.00 |           0.00 |       8 |             4 (0.07 MiB) |                   1 (8 B) |
|       20 | Mixed    | Background |         63 |         131.70 |          8.53 |         612.26 |            0.30 |           0.00 |     128 |             4 (2.25 MiB) |                   1 (8 B) |
|       50 | Isolated | Foreground |        600 |           1.59 |          1.67 |           1.46 |            0.00 |           0.00 |       8 |             4 (0.07 MiB) |                   1 (8 B) |
|       50 | Mixed    | Foreground |        600 |           1.64 |          0.55 |           1.45 |            0.00 |           0.00 |       8 |             4 (0.07 MiB) |                   1 (8 B) |
|       50 | Mixed    | Background |         72 |         130.37 |          8.08 |         621.68 |            0.30 |           0.00 |     128 |             4 (2.25 MiB) |                   1 (8 B) |
|      200 | Isolated | Foreground |        600 |           1.34 |          1.43 |           1.46 |            0.00 |           0.00 |       8 |             4 (0.07 MiB) |                   1 (8 B) |
|      200 | Mixed    | Foreground |        374 |           1.68 |          0.00 |           1.39 |            0.00 |           0.00 |       8 |             4 (0.07 MiB) |                   1 (8 B) |
|      200 | Mixed    | Background |         18 |         114.07 |          9.03 |         614.88 |            0.32 |           0.00 |     128 |             4 (2.25 MiB) |                   1 (8 B) |

Key takeaways from #138 metrics:

1. **Background scans are CPU-bound, not I/O-starved:** with
   `pushdown_filters = false`, each background scan spends ~606 ms of summed
   worker CPU (`decode_work` 446.5 ms + `process_work` 159.3 ms) and only
   4.56 ms in `ready_wait`. Across 4 affined CPUs, 6.00 scans/s consumes
   ~3,636 ms-CPU/s before foreground work, planning, and Tokio task switching.
2. **Whole-row-group `RowFilter` explains the `pushdown_filters = true` tail
   latency:** under `pushdown_filters = true`, filter evaluation moves from
   per-batch `process_work` (`159.3 → 0.30 ms`) into synchronous whole-row-group
   `decode_work` (`446.5 → 612.3 ms`, ~4.78 ms per 131,072-row group), holding
   Tokio worker threads 16× longer between yields than the ~0.30 ms per-batch
   slice in `pushdown_filters = false`.
3. **Queued compressed-byte estimate drops 4.1× under #150**
   (`9.32 → 2.25 MiB` for `large`, `0.30 → 0.07 MiB` for `small`) because
   `FilteredBufferedReaders` fetches only predicate columns (`score`, `delta`)
   and tracks actual fetched bytes rather than full row-group compressed size.

## Reproduction

From a checkout at `98b2c851653b8525d4386d380556030a510f12e4` (with this
directory's `--revision`, `--pushdown-filters`, and `--morsel-metrics` flag
support):

```bash
set -e
b=benchmarks/morsel_mixed
p=benchmarks/morsel_parquet
work="$HOME/.cache/issue171"
rev=98b2c851653b8525d4386d380556030a510f12e4
mkdir -p "$work"

MORSEL_BENCH_REVISION="$rev" CARGO_TARGET_DIR="$work/target" \
  CARGO_PROFILE_RELEASE_OPT_LEVEL=2 timeout --kill-after=15s 1200s \
  cargo build -j 10 --release --locked --manifest-path "$b/Cargo.toml"

timeout 60s uv venv --python 3.12.10 "$work/venv"
timeout 120s uv pip install --python "$work/venv/bin/python" -r "$p/requirements.txt"
py="$work/venv/bin/python"
timeout 60s "$py" -B "$p/harness.py" generate "$work/small" \
  --files 4 --groups 2 --group-rows 4096
timeout 60s "$py" -B "$p/harness.py" generate "$work/large" \
  --files 16 --groups 8 --group-rows 131072

# Paired runs with pushdown_filters = false (apples-to-apples with #139)
timeout --kill-after=10s 360s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
  --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
  --revision "$rev" --workers 4 --blocking-threads 2 --partitions 4 --background-lanes 3 \
  --pairs 4 --requests 600 --interval-ms 20 5 > "$work/runA-pushdown0-50-200.jsonl"
timeout --kill-after=10s 260s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
  --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
  --revision "$rev" --workers 4 --blocking-threads 2 --partitions 4 --background-lanes 3 \
  --pairs 4 --requests 200 --interval-ms 50 > "$work/runB-pushdown0-20.jsonl"

# Paired runs with pushdown_filters = true (#150)
timeout --kill-after=10s 360s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
  --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
  --revision "$rev" --pushdown-filters --workers 4 --blocking-threads 2 --partitions 4 \
  --background-lanes 3 --pairs 4 --requests 600 --interval-ms 20 5 \
  > "$work/runC-pushdown1-50-200.jsonl"
timeout --kill-after=10s 260s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
  --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
  --revision "$rev" --pushdown-filters --workers 4 --blocking-threads 2 --partitions 4 \
  --background-lanes 3 --pairs 4 --requests 200 --interval-ms 50 \
  > "$work/runD-pushdown1-20.jsonl"

# Separate #138 wait/buffering metrics check (--morsel-metrics, 1 pair each)
for pd in "" "--pushdown-filters"; do
  tag=$([ -n "$pd" ] && echo pd1 || echo pd0)
  timeout --kill-after=10s 240s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
    --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
    --revision "$rev" $pd --morsel-metrics --workers 4 --blocking-threads 2 \
    --partitions 4 --background-lanes 3 --pairs 1 --requests 600 --interval-ms 20 5 \
    > "$work/metrics-$tag-50-200.jsonl"
  timeout --kill-after=10s 240s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
    --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
    --revision "$rev" $pd --morsel-metrics --workers 4 --blocking-threads 2 \
    --partitions 4 --background-lanes 3 --pairs 1 --requests 200 --interval-ms 50 \
    > "$work/metrics-$tag-20.jsonl"
done
```
