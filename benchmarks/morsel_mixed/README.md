<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Same-runtime mixed-query latency

This standalone, fork-only benchmark runs **stock DataFusion** (`morsel_mode=off`).
It compares a foreground SQL query alone with the same query under concurrent
scans. Separate DataFusion sessions share **one process and one multi-threaded
Tokio runtime** in both modes. The harness needs no external data by default.

## Build and run

From the repository root with its pinned Rust toolchain:

```bash
timeout 1200 cargo build --release --locked --manifest-path benchmarks/morsel_mixed/Cargo.toml
timeout 60 taskset --cpu-list 12-15 benchmarks/morsel_mixed/target/release/morsel-mixed \
  --workers 4 --blocking-threads 2 --partitions 4 --batch-size 8192 \
  --seed 42 --small-rows 4096 --large-rows 8388608 --background-lanes 3 \
  --warmup 2 --repetitions 3 --requests 300 --interval-ms 5 \
  --queue-capacity 16 --overdue-ms 20 --timeout-ms 10000 > mixed.json
```

Choose four available CPUs for the Linux example. `taskset` is optional, but it
limits **all** threads (including the driver and any blocking work) to those CPUs;
Tokio's worker and blocking-thread limits alone are not a CPU quota. Allocation
and setup run on the caller thread, which also drives arrivals. Query tasks are
spawned on Tokio workers in both modes. DataFusion target partitions and batch
size are set in both sessions; target partitions are not a thread count or a
promise about a custom query plan's actual partitioning. The JSON records these
settings, visible CPU affinity on Linux, and the executable's arguments. See
`--help` for all options. Run the prebuilt binary when the selected CPUs are idle;
do not treat a run during a build or other shared-host activity as performance sign-off.

## Workload and clocks

- The generated Arrow tables have `v: BIGINT` from a fixed SplitMix64 seed and
  are split into the configured partitions and batches. Foreground SQL counts
  rows satisfying `v % 17 = 0`; background SQL scans all rows, summing
  `(v * v + v * 13) % 1000033`. Expected values are computed independently of
  DataFusion while generating the input. Warm-up runs both queries sequentially
  before timing, then pairs alternate which mode runs first.
- In mixed mode each background lane starts a query and immediately repeats
  after draining it and yielding to Tokio. The arrival clock starts after each
  lane has polled a stream at least once (or reported an error). Lanes stop
  starting scans at the nominal window end; in-flight scans drain. For each
  foreground sample the harness counts background streams in flight when it
  starts polling its own stream. This proves overlap of query execution, not
  simultaneous CPU residency, and can miss scans that start later in the sample.
  A mixed run without a successful overlapping sample is invalid.
- Requests are scheduled for `epoch + id * interval`, regardless of when
  earlier work completes. The caller thread sleeps until each deadline on the
  monotonic clock, without relying on Tokio's worker timers. OS scheduling can
  still delay the driver. A late driver attempts every offer against the
  original timestamps; `late_offers` counts offers at least one interarrival
  late. The driver never blocks on the bounded FIFO queue: `queued` counts
  accepted requests that had an earlier accepted request outstanding, and
  `dropped` means the channel was full. `overdue` means the time from scheduled
  arrival to finish (or rejection for a drop) exceeded `--overdue-ms`. These
  flags can overlap other counts.
- End-to-end latency is **scheduled arrival to end of stream**. It includes
  `offer_lag` (scheduled to actual offer), `queue_wait` (offer to dequeue,
  including worker wake-up), and `service_time` (dequeue to end of stream).
  `planning` covers dequeue to physical-plan creation, including SQL parsing
  and logical planning. No query plans are reused. Draining counts rows and
  captures a scalar when requested without retaining all output. Result
  assertions run after both workloads drain, outside latency and throughput
  timers. Dataset generation, fixture registration and warm-up are also
  outside the timers.
- The cooperative timeout applies from scheduled arrival for foreground work
  (including queueing) and from query start for background work. There are finite
  request, window, queue and background scan limits (10,000 scans per lane per
  run). An external `timeout` also bounds setup, shutdown and non-yielding work.
  A timed-out query, wrong result, exhausted lane, or missing overlap makes
  the measured run invalid; JSON is still emitted before a nonzero exit.
  Setup/warm-up failures and external termination can prevent a report.

To swap a workload, pass `--small-sql` or `--large-sql` with SQL text, and
optionally `--small-fixture` or `--large-fixture` with a Parquet file/directory
(registered as `small` or `large`, respectively). Each override requires its
own `--*-expected-rows` or `--*-expected-scalar` (a single non-null 64-bit integer
such as `COUNT(*)`). Use read-only queries. Choose background SQL that actually
scans its input; a metadata-only count or constant query does not reproduce this
CPU workload. All overrides use the same arrival and drain logic.

## Reading the JSON

There is one report per invocation. Pair `runs` by zero-based `repetition` and
compare `mode` (`isolated` or `mixed`). Each timing distribution has nearest-rank
p50, p95 and p99 in nanoseconds, a sample count, and sample variance in ns²
(null for fewer than two samples). All distributions use the same set of
successful requests. Per-offer timestamps are relative to that run's scheduled
start. Check drops, failures, overdue and overlap counts before comparing
percentiles; a sample with heavy drops is not directly comparable to a lossless
one.

`offered_per_second` divides all offers by the nominal window (`requests * interval`).
`completed_per_second` divides successful requests by the window extended through
any foreground drain. Background throughput (`completed_in_window_per_second`)
counts validated scans finishing inside the nominal window per second. It does
not credit partial scans or those finishing during warm-up/drain;
`background.completed` also counts successful scans that finish before or after
the nominal window during the measured run.

## Stock validation on dduck2

One release invocation of the commands above (Rust 1.99.1, aarch64 Neoverse-N1,
DataFusion base `d649cacf00fee7125f45ea1427a023b2eab732b1`). It offered 200
requests/s per phase with four CPUs, four async workers, up to two blocking
workers, four target partitions and three background lanes. There were no drops
or failures. The expected foreground count was 236 and the background sum was
4,193,561,459,776. The timings validate the harness, **not** stock versus morsel
performance or a performance threshold.

| Pair | Mode     | Samples | p50 ms | p95 ms | p99 ms | Variance ms² | Overlap | Completed/s | Scans/s |
| ---: | :------- | ------: | -----: | -----: | -----: | -----------: | ------: | ----------: | ------: |
|    0 | Isolated |     300 |   0.95 |   1.07 |   1.46 |        0.015 |       0 |      200.00 |    0.00 |
|    0 | Mixed    |     300 |   5.52 |  23.61 |  30.36 |       48.789 |     299 |      200.00 |   33.33 |
|    1 | Isolated |     300 |   0.97 |   1.11 |   1.19 |        0.005 |       0 |      200.00 |    0.00 |
|    1 | Mixed    |     300 |   4.42 |  21.13 |  27.27 |       40.819 |     298 |      200.00 |   34.67 |
|    2 | Isolated |     300 |   0.98 |   1.15 |   2.01 |        0.024 |       0 |      200.00 |    0.00 |
|    2 | Mixed    |     300 |   4.24 |  18.27 |  27.44 |       33.273 |     299 |      200.00 |   34.00 |

All three lanes overlapped at least one foreground execution in each mixed run.
No offer was a full interval late. Mean offer lag was 0.06–0.07 ms isolated
and 0.14–0.16 ms mixed; the mixed median queue wait was 2.90–4.41 ms.
Mixed runs had 133–160 queued and 12–24 overdue requests; isolated runs had
none. The planning p50 was 0.71–0.74 ms across all six runs.

## Smoke test

```bash
timeout 1200 cargo test --release --locked --manifest-path benchmarks/morsel_mixed/Cargo.toml
```

The integration tests exercise the binary, check fixed arrivals, stock result
validation and real execution overlap, and verify that a wrong expected row
count across several output batches is reported as a failure after draining.
