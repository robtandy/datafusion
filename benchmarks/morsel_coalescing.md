<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Worker-local post-filter coalescing (#149)

Before: `61f4035d226359549517f792b5f0a03be7ca4565` (includes #148).
After: `580c418e35541e3f78102a78feb8e6a3d9a020f2`.
Stock (`off`) and native (`required`) used the same fixtures and flags in each
build. Counts are an unchanged execution-path control. Values are before →
after, with time in ms; `—` means the query emits fewer than 8,192 rows. The
[`MorselMetrics`] collector adds per-send bookkeeping, which can inflate the
apparent speedup from sending fewer batches. The #138 output-wait metric sums
worker suspension time on result permits; it is not wall time. Depth and Arrow
bytes measure only the channel, not worker-local batches, coalescers, or Parquet
readers.

## #127 in-memory A/B

All cases returned the expected 2,094,791 selected rows or a single count.
Session `batch_size` was 4,096. Timing starts at stream execution.

### Timing without the metrics collector

The unmodified benchmark has 31 timed samples per invocation. Each reported
value is the midpoint of two invocation medians for that revision.

| Layout / workers | Query      | Mode   |   First batch |           Total |
| ---------------- | ---------- | ------ | ------------: | --------------: |
| Uniform / 1      | Projection | Stock  | 0.282 → 0.304 | 19.314 → 19.307 |
| Uniform / 1      | Projection | Native | 0.021 → 0.059 | 18.414 → 18.748 |
| Uniform / 1      | Count      | Stock  | 9.629 → 9.321 |   9.634 → 9.326 |
| Uniform / 1      | Count      | Native | 8.919 → 8.436 |   8.922 → 8.439 |
| Uniform / 4      | Projection | Stock  | 0.271 → 0.273 |   6.726 → 6.602 |
| Uniform / 4      | Projection | Native | 0.035 → 0.106 |   7.821 → 7.424 |
| Uniform / 4      | Count      | Stock  | 2.675 → 2.689 |   2.678 → 2.696 |
| Uniform / 4      | Count      | Native | 2.529 → 2.494 |   2.532 → 2.498 |
| Skewed / 4       | Projection | Stock  | 0.213 → 0.206 | 18.303 → 17.795 |
| Skewed / 4       | Projection | Native | 0.035 → 0.102 |   7.875 → 7.382 |
| Skewed / 4       | Count      | Stock  | 8.628 → 8.202 |   8.631 → 8.206 |
| Skewed / 4       | Count      | Native | 2.607 → 2.454 |   2.611 → 2.456 |

Four-worker native projections took 5.1% less total time on uniform input and
6.3% less on skewed input without the collector, at the cost of about 0.07 ms
more to the first batch. Uniform four-worker stock remained faster end to end.
The inline native projection took 1.8% longer overall; small differences should
be interpreted in light of the variation in the unchanged stock and count arms.

### Separate runs with the metrics collector

The unmodified benchmark does not report batch counts or the 8,192-row
checkpoint; these diagnostic values are from the instrumented runs instead.

| Layout / workers | Stock batches | Native batches | Stock first 8,192 | Native first 8,192 |
| ---------------- | ------------: | -------------: | ----------------: | -----------------: |
| Uniform / 1      |     512 → 512 |    2,048 → 512 |     0.345 → 0.335 |      0.089 → 0.089 |
| Uniform / 4      |     512 → 512 |    2,048 → 512 |     0.304 → 0.278 |      0.095 → 0.117 |
| Skewed / 4       |     514 → 514 |    2,048 → 512 |     0.220 → 0.213 |      0.091 → 0.112 |

Both four-worker native projections reduced the median summed permit wait
from 0.115 ms (uniform) and 0.126 ms (skewed) to zero. Median peak queue depth
rose from 2 to 4 batches (8 slots configured), and median peak queued Arrow
bytes rose from 17,112 to 112,184. The inline projection did not use a result
queue. Counts each emitted one batch and native count queue metrics were
unchanged (depth 1, 8 bytes, no permit wait).

## #128 Parquet A/B

Eight workers and `batch_size=8192`. Timing starts at table registration.
Both arms matched the NumPy oracle for 759,841 low-selectivity and 9,883,940
high-selectivity rows (and counts). Native runs claimed all 128 row groups;
the collector also confirmed all 128 completions. Scan and count read
196,691,567 and 75,280,626 bytes respectively in both builds.

### Timing with the metrics collector (`MORSEL_METRICS=1`)

These medians describe the instrumented runs, not collector-off performance.
A warm trial with the collector disabled overlapped a Rust build and was
discarded.

| Cache           | Query        | Mode   |   First batch | First 8,192 |       Batches |         Total |
| --------------- | ------------ | ------ | ------------: | ----------: | ------------: | ------------: |
| Warm            | `scan_low`   | Stock  |   31.2 → 30.0 | 31.2 → 30.0 |       96 → 96 | 169.9 → 163.8 |
| Warm            | `scan_low`   | Native |   16.8 → 31.1 | 20.9 → 31.1 |    2,048 → 96 | 207.2 → 168.0 |
| Warm            | `scan_high`  | Stock  |   16.5 → 15.7 | 16.5 → 15.7 | 2,048 → 2,048 | 192.8 → 189.9 |
| Warm            | `scan_high`  | Native |   17.2 → 16.4 | 18.0 → 16.8 | 2,048 → 2,048 | 216.0 → 193.2 |
| Warm            | `count_low`  | Stock  |   91.0 → 88.8 |           — |         1 → 1 |   91.0 → 88.8 |
| Warm            | `count_low`  | Native |   93.3 → 90.2 |           — |         1 → 1 |   93.3 → 90.2 |
| Warm            | `count_high` | Stock  |   91.2 → 91.7 |           — |         1 → 1 |   91.2 → 91.7 |
| Warm            | `count_high` | Native |   94.8 → 93.6 |           — |         1 → 1 |   94.9 → 93.7 |
| Page-cache-cold | `scan_low`   | Stock  |   39.7 → 37.6 | 39.7 → 37.6 |       96 → 96 | 216.8 → 217.3 |
| Page-cache-cold | `scan_low`   | Native |   24.3 → 43.2 | 30.7 → 43.2 |    2,048 → 96 | 237.4 → 197.3 |
| Page-cache-cold | `scan_high`  | Stock  |   25.0 → 28.1 | 25.8 → 28.2 | 2,048 → 2,048 | 234.4 → 243.6 |
| Page-cache-cold | `scan_high`  | Native |   25.6 → 26.7 | 26.2 → 27.4 | 2,048 → 2,048 | 246.4 → 221.8 |
| Page-cache-cold | `count_low`  | Stock  | 115.6 → 118.6 |           — |         1 → 1 | 115.6 → 118.6 |
| Page-cache-cold | `count_low`  | Native | 101.7 → 101.5 |           — |         1 → 1 | 101.7 → 101.6 |
| Page-cache-cold | `count_high` | Stock  | 118.9 → 122.7 |           — |         1 → 1 | 119.0 → 122.7 |
| Page-cache-cold | `count_high` | Native | 106.5 → 105.0 |           — |         1 → 1 | 106.6 → 105.0 |

Native `scan_low` emitted 95–97 batches after the change instead of 2,048.
With the collector enabled, median total time fell 19% warm and 17%
page-cache-cold, but time to 8,192 rows rose by 10 ms and 12.5 ms respectively.
For `scan_high`, large filtered batches continued to pass through and permit
waits dropped without a change in batch count. This comparison measures the
two changes together.

### Backpressure with the metrics collector (`MORSEL_METRICS=1`)

| Cache           | Native query | Median output wait (ms) | Peak depth (max) | Peak Arrow KiB (max) | Median ready wait (ms) |
| --------------- | ------------ | ----------------------: | ---------------: | -------------------: | ---------------------: |
| Warm            | `scan_low`   |               187.1 → 0 |            2 → 6 |         13.4 → 774.0 |            17.7 → 17.9 |
| Warm            | `scan_high`  |            187.4 → 36.2 |           2 → 16 |      154.9 → 1,222.7 |            17.1 → 17.1 |
| Page-cache-cold | `scan_low`   |               260.9 → 0 |            2 → 7 |         13.1 → 903.0 |            79.8 → 88.0 |
| Page-cache-cold | `scan_high`  |            251.3 → 63.1 |           2 → 16 |      155.3 → 1,223.6 |           87.6 → 101.1 |

The maximum ready queue depth was 8 before and after, independent of the
result queue's new 16-slot capacity. Both counts had no output-permit wait;
peak queued output stayed at one batch (8 bytes).

## Reproduction

Measured on `rob-tandy-dduck1` (16 ARM Neoverse-N1 vCPUs, 61 GiB RAM,
Rust 1.99.1, Clang/lld 18.1.3) on November 22, 2026 UTC. All builds used
`release-nonlto` and only the `parquet,sql` package features.

For #127 timing, use the unmodified `datafusion/core/benches/morsel_execution.rs`:
the [fixture, warmups, 31 samples, and row oracle](morsel_execution.md) are unchanged.
Run the pinned bench binary in before / after / after / before order. For the
separate diagnostic runs, copy the bench to
`datafusion/core/examples/issue149_memory.rs` and install a [`MorselMetrics`]
collector in each session. In the timed drain, count nonempty batches and time
the crossing of 8,192 cumulative rows, just like the #128 runner. Clear the
collector with `take_runs()` after the sample primer and collect the three output
metrics (`morsel_output_wait`, `morsel_peak_output_depth`, and
`morsel_peak_output_arrow_bytes`) with `sum_by_name` after EOF, outside the timer.
For #128, copy the runner with the [#148 metrics additions](morsel_parquet/issue-148.md#reproduction)
to `datafusion/core/examples/issue149_parquet.rs`. Use the same private 16-file,
128-row-group fixture from that comparison. Warm samples started with all
76,063 pages resident. The first cold attempt was discarded after the harness
found 13 resident pages; both cold arms were rerun on one private reflink copy
and each sample started with zero resident pages. This does not control device
caches. Host CPU was at least 96% idle immediately before the in-memory and
warm Parquet runs, but only 33–82% idle in samples before the cold retry; cold
wall-clock differences should be treated as directional.

From each revision (use separate target directories; set `CARGO_TARGET_DIR`,
`py`, `fixture`, and `cold_fixture` to absolute paths; the after Parquet binary
was tagged `issue149-working-tree`):

```sh
# Unmodified #127 timing, without a collector.
timeout --kill-after=15s 600s taskset -c 0-11 \
  cargo build --locked --profile release-nonlto -j 8 -p datafusion \
    --no-default-features --features parquet,sql --bench morsel_execution
# bench_bin is the built morsel_execution executable in release-nonlto/deps.
timeout --kill-after=10s 120s taskset -c 12-15 "$bench_bin" --bench

# Build the separate #127 diagnostic and #128 runners.
revision=$(git rev-parse HEAD)
MORSEL_BENCH_REVISION="$revision" timeout --kill-after=15s 1100s taskset -c 0-11 \
  cargo build --locked --profile release-nonlto -j 8 -p datafusion \
    --no-default-features --features parquet,sql \
    --example issue149_memory --example issue149_parquet
runner_dir="$CARGO_TARGET_DIR/release-nonlto/examples"
timeout --kill-after=10s 120s taskset -c 12-15 "$runner_dir/issue149_memory" --bench
MORSEL_METRICS=1 timeout --kill-after=10s 240s taskset -c 0-7 "$py" -B \
  benchmarks/morsel_parquet/harness.py ab "$fixture" \
  --runner "$runner_dir/issue149_parquet" --revision "$revision" --threads 8 \
  --timeout 120 --cache warm --warmups 1 --repeats 8 > "$revision-warm.jsonl"
# For the cold retry, prepare cold_fixture once with cp -a --reflink=always.
MORSEL_METRICS=1 timeout --kill-after=10s 150s taskset -c 0-7 "$py" -B \
  benchmarks/morsel_parquet/harness.py ab "$cold_fixture" \
  --runner "$runner_dir/issue149_parquet" --revision "$revision" --threads 8 \
  --timeout 120 --cache page-cache-cold --warmups 1 --repeats 8 > "$revision-page-cache-cold.jsonl"
```

[`MorselMetrics`]: ../docs/source/contributor-guide/morsel_metrics.md
