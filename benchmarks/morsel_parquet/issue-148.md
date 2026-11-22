<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Parquet producer read-ahead (#148)

Before: `c2809d3b3ee73936b182429e1646af290d797c6e` (includes #138). After:
`5fe71ad1e8b506f0f4dd9576aab0794c76ea483d`. This comparison does not replace the
pinned [#128 baseline](README.md#native-parquet-ab-128).

## Method

`rob-tandy-dduck1`, November 22, 2026 UTC: Ubuntu 24.04.5, Linux
`7.0.0-1014-aws`, 16 Neoverse-N1 vCPUs, 61 GiB RAM, Rust 1.99.1, Clang/lld
18.1.3 and Python 3.12.10. The private fixture was on Btrfs on local EC2
NVMe instance storage (`compress=zstd:1`). No builds ran during either
measurement; three one-second `vmstat` samples immediately before each
regime showed 97–99% host CPU idle. Both arms used CPUs 0–7 and eight workers.
One warmup and eight timed samples per arm and query in each cache regime.

The unmodified #128 harness verified all four queries in both arms against
the same NumPy oracle before timing: 759,841 low-selectivity and 9,883,940
high-selectivity rows (and counts). Stock used `morsel_mode=off`; native used
`morsel_mode=required`, claimed and completed all 128 row groups, and finalized
exactly one count per count query. Both scanned 196,691,567 bytes per scan or
75,280,626 per count, before and after. All timed warm samples started with
76,063/76,063 file pages resident; all page-cache-cold samples started with
zero resident pages. This does not control device caches; the #128 EBS
storage check was not repeated.

## Timings

Total elapsed ms, from registration through stream exhaustion. Median (min–max).

| Cache           | Case         |        Stock before |         Stock after |       Native before |        Native after |
| --------------- | ------------ | ------------------: | ------------------: | ------------------: | ------------------: |
| Warm            | `scan_low`   | 169.8 (165.9–179.3) | 168.5 (163.7–174.3) | 620.2 (615.8–624.1) | 209.2 (204.4–239.6) |
| Warm            | `scan_high`  | 198.5 (189.2–235.2) | 191.6 (188.6–197.5) | 622.0 (612.5–623.3) | 215.6 (207.3–224.4) |
| Warm            | `count_low`  |    90.4 (88.0–94.2) |    91.3 (86.3–96.0) | 244.1 (241.9–251.2) |    90.7 (89.6–93.4) |
| Warm            | `count_high` |   91.3 (89.4–103.8) |    89.9 (87.5–94.7) | 244.7 (243.0–249.1) |    95.8 (91.9–97.6) |
| Page-cache-cold | `scan_low`   | 200.5 (198.0–221.4) | 196.6 (193.2–200.7) | 630.2 (625.2–632.6) | 227.5 (219.5–235.6) |
| Page-cache-cold | `scan_high`  | 222.8 (217.6–228.6) | 214.1 (211.4–222.4) | 631.3 (625.3–637.5) | 237.9 (233.2–272.2) |
| Page-cache-cold | `count_low`  | 104.5 (102.9–119.3) | 103.4 (100.6–105.2) | 252.5 (248.8–265.2) |    97.7 (96.7–98.8) |
| Page-cache-cold | `count_high` | 109.3 (103.9–113.3) | 102.9 (101.2–111.4) | 252.3 (248.5–254.3) |  100.4 (99.2–114.1) |

Median native runs were 2.65–2.96× as fast as before for scans and
2.51–2.69× as fast for counts.
The new scans still took 1.11–1.24× as long as stock; counts were within 7%.
Native time to the first 8,192 rows fell from 29.4 to 21.1 ms (`scan_low`) and
20.7 to 16.9 ms (`scan_high`) warm, and from 34.8 to 28.1 ms and 26.4 to
23.3 ms page-cache-cold. Native first-batch medians for scans were about
2–3 ms later after the change.

## #138 wait and buffering metrics

Medians of per-run totals in ms, before → after. I/O is summed over pending
planner phases; the other columns are summed over all eight workers. These
are overlapping task durations, not parts of elapsed time. Ready and output
wait measure suspension while claiming a reader or obtaining a result permit.

| Cache           | Case         |   Pending I/O |     Ready wait |       Decode work |  Process work | Output wait |
| --------------- | ------------ | ------------: | -------------: | ----------------: | ------------: | ----------: |
| Warm            | `scan_low`   |  85.7 → 196.4 | 2,357.1 → 21.8 | 1,074.5 → 1,004.1 | 155.7 → 163.9 | 1.0 → 196.7 |
| Warm            | `scan_high`  |  86.1 → 193.7 | 2,308.5 → 18.3 |   1,075.8 → 999.2 | 205.7 → 219.0 | 2.6 → 200.8 |
| Warm            | `count_low`  |  41.7 → 123.5 |   764.4 → 12.1 |     487.0 → 454.2 | 131.7 → 135.3 |       0 → 0 |
| Warm            | `count_high` |  40.9 → 135.3 |   737.5 → 11.9 |     485.9 → 452.2 | 159.7 → 166.0 |       0 → 0 |
| Page-cache-cold | `scan_low`   | 287.2 → 449.3 | 2,399.7 → 75.8 | 1,082.2 → 1,005.3 | 155.5 → 167.5 | 4.1 → 254.8 |
| Page-cache-cold | `scan_high`  | 287.0 → 436.6 | 2,344.5 → 67.3 | 1,081.6 → 1,010.5 | 207.9 → 219.5 | 2.7 → 245.1 |
| Page-cache-cold | `count_low`  | 157.4 → 276.0 |   798.8 → 24.0 |     492.3 → 467.9 | 132.3 → 141.8 |       0 → 0 |
| Page-cache-cold | `count_high` | 153.9 → 281.8 |   769.5 → 24.4 |     489.5 → 464.2 | 161.0 → 174.9 |       0 → 0 |

Before the change, ready wait dominated the measured worker delays even
with warm file pages, and output wait was small. Afterward, workers spent
much less time waiting for readers, and output backpressure became the
larger measured worker wait on scans. The increase in summed I/O wait is
consistent with more phases overlapping; this timer excludes synchronous
producer work and does not measure disk latency in isolation.

Across both cache regimes, peak ready depth went from 2 to 8 for every case.
The maximum queued compressed-size estimate rose from 4,903,170 to
19,502,241 bytes. Peak output depth was unchanged (two batches for scans,
one for counts). Maximum queued Arrow bytes were 13,376 → 13,616 for
`scan_low`, 158,664 → 159,112 for `scan_high`, and 8 → 8 for both counts.

The producer now runs up to `min(workers, files, 16)` file lanes with one
pending planner I/O phase per lane. The ready FIFO holds up to
`min(workers, 16)` readers. A lane can read ahead by one row group behind a
full FIFO; it must publish that reader before advancing further. Queue
gauges exclude lane-local readers and the at-most-one reader held by each
decode worker. Ready bytes estimate the full compressed row groups, not
resident memory; the queue and lane limits are reader-count limits.

## Reproduction

Use the [#128 setup](README.md#native-parquet-ab-128) to generate the private
16-file × 8-row-group × 131,072-row fixture (311,529,117 bytes) and set up the
pinned Python dependencies. For each revision, copy `stream_runner.rs` to
`datafusion/core/examples/morsel_metrics_smoke.rs` in its checkout and apply
the [#138 runner changes](../../docs/source/contributor-guide/morsel_metrics.md#parquet-smoke-check).
After the timed drain, include the raw report in the runner's JSON as
`"detailed": detailed`, using:

```rust
let detailed: Vec<_> = metrics
    .take_runs()
    .iter()
    .map(|run| {
        run.iter()
            .map(|m| (m.to_string(), m.value().as_usize()))
            .collect::<Vec<_>>()
    })
    .collect();
```

From each checkout (set `py`, `fixture` and `CARGO_TARGET_DIR` to absolute paths):

```bash
revision=$(git rev-parse HEAD)
MORSEL_BENCH_REVISION="$revision" timeout --kill-after=15s 900s taskset -c 0-11 \
  cargo build --locked --profile release-nonlto -j 8 -p datafusion \
    --no-default-features --features parquet,sql --example morsel_metrics_smoke
runner="$CARGO_TARGET_DIR/release-nonlto/examples/morsel_metrics_smoke"
for cache in warm page-cache-cold; do
  vmstat -y 1 3  # wait for a quiet workspace before measuring
  MORSEL_METRICS=1 timeout --kill-after=10s 240s taskset -c 0-7 "$py" -B \
    benchmarks/morsel_parquet/harness.py ab "$fixture" --runner "$runner" \
    --revision "$revision" --threads 8 --timeout 120 --cache "$cache" \
    --warmups 1 --repeats 8 > "$revision-$cache.jsonl"
done
```
