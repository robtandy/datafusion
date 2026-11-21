<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Same-runtime mixed-query latency

This standalone, fork-only benchmark compares a foreground SQL query alone
with the same query under concurrent scans. Separate DataFusion sessions share
**one process and one multi-threaded Tokio runtime** within each run. The harness
needs no external data for the default stock DataFusion workload. The Parquet
comparison below runs the same harness in both `morsel_mode=off` and
`morsel_mode=required`.

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
load. All overrides use the same arrival and drain logic. With
`--morsel-mode required`, every executed query must have a `MorselExec` root;
unsupported SQL fails instead of falling back. Parquet and required-mode plans
are recorded before timing. `--mixed-first` reverses the phase order of the
first repetition; subsequent repetitions still alternate.

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
any foreground drain. `completion_rate` and `drop_rate` are fractions of offers.
Background throughput (`completed_in_window_per_second`)
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

## Pinned Parquet comparison (#139)

Use the same engine revision as the [single-query comparison](../morsel_parquet/README.md):
`d649cacf00fee7125f45ea1427a023b2eab732b1`, the #126 merge, before later
optimizations or instrumentation. Build **one** executable for both backends;
copy only this standalone harness into a detached checkout. The runner rejects
reports whose build revision does not match the pin.

From the harness checkout on a Linux remote workspace, with `uv`, `taskset`,
and a private directory on the device under test:

```bash
set -e
b=benchmarks/morsel_mixed
p=benchmarks/morsel_parquet
work=$(mktemp -d "$HOME/.cache/issue139.XXXXXX")
revision=d649cacf00fee7125f45ea1427a023b2eab732b1

git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/$b"
cp -r "$b"/{Cargo.toml,Cargo.lock,src,tests} "$work/pinned/$b/"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$(git rev-parse HEAD)" CARGO_TARGET_DIR="$work/target" \
    CARGO_PROFILE_RELEASE_OPT_LEVEL=2 timeout --kill-after=15s 20m \
    cargo build -j 10 --release --locked --manifest-path "$b/Cargo.toml"
)
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$p/requirements.txt"
py="$work/venv/bin/python"
timeout 60s "$py" -B "$p/harness.py" generate "$work/small" \
  --files 4 --groups 2 --group-rows 4096
timeout 60s "$py" -B "$p/harness.py" generate "$work/large" \
  --files 16 --groups 8 --group-rows 131072

# Check that the host is quiet and no builds are running. Adapt the CPU list.
vmstat -y 1 3
ps -eo pid,pcpu,comm --sort=-pcpu | head -20
timeout --kill-after=10s 360s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
  --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
  --workers 4 --blocking-threads 2 --partitions 4 --background-lanes 3 \
  --pairs 4 --requests 600 --interval-ms 20 5 > "$work/mixed-parquet.jsonl"

# Lower-load follow-up to compare tails without foreground drops.
vmstat -y 1 3
ps -eo pid,pcpu,comm --sort=-pcpu | head -20
timeout --kill-after=10s 260s taskset --cpu-list 12-15 "$py" -B "$b/compare_parquet.py" \
  --runner "$work/target/release/morsel-mixed" --small "$work/small" --large "$work/large" \
  --workers 4 --blocking-threads 2 --partitions 4 --background-lanes 3 \
  --pairs 4 --requests 200 --interval-ms 50 > "$work/mixed-parquet-low.jsonl"
```

- The small query uses the low-selectivity filtered count from #128; the long
  scan uses the high-selectivity filtered count on the larger fixture. Native
  execution at the pin does not admit the in-memory harness's default `SUM`,
  so both arms use the admitted `COUNT(*)` with a two-column modulo filter.
  This still reads row values throughout the input; footer counts cannot answer
  it. The returned scalar is checked against the #130 generator's independent
  array oracle for **every** completed query, including in-flight scans draining
  past the window. Each arm checks the physical plans of both Parquet queries:
  stock `DataSourceExec` or `MorselExec` with `NativeParquet` and `CountSink`.
- As in #128, the Parquet sessions disable listing-statistics collection,
  repartitioning of file scans, Parquet row-filter pushdown, and row-group,
  page-index and Bloom pruning. Both sessions use 8,192-row batches; native
  workers **per pipeline** are capped by the same target-partitions setting used
  by stock.
  This is not a per-process task limit; the four affined CPUs and shared Tokio
  runtime are the process-wide budget. Both backends drain to the same scalar,
  using the same harness clocks, two warm-up queries per workload, bounded
  foreground FIFO, and three continuously reissuing background lanes. This
  matches background **concurrency**, not background arrival rate: faster
  completions allow the lane to submit another scan sooner.
- Each arm is a fresh process; within it, its isolated and mixed phases share
  the runtime and sessions. Arms alternate which backend starts the pair, and
  both backends in a pair use the same phase order (balanced over four pairs).
  Files are read into the Linux page cache before each arm and verified fully
  resident with `mincore`. Residency is sampled again after the arm, not
  continuously. Registration and warm-up precede timing; this is not a cold-I/O
  test and does not reset filesystem metadata, allocator, CPU, hypervisor or
  storage-device caches. Affinity does not reserve the CPUs or isolate their
  caches, memory bandwidth, or storage from other activity.
- The first NDJSON line records fixture shape, expected counts, host, affinity,
  revision and load levels. Each subsequent line holds one arm of a pair:
  match `interval_ms` and `pair`, then compare the same phase under `off` and
  `required`. The nested report retains the individual offer records, p50/p95/p99
  and variance, samples, completion/drop fractions, overlap, and foreground
  and background throughput. Background **logical input rows/s** credits only
  complete scans finishing inside the nominal offer window. A comparison that
  exits nonzero is incomplete; do not use preceding lines as a full comparison.

### Pinned comparison on dduck2, 2026-11-21 UTC

`rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`, 16 aarch64
Neoverse-N1 vCPUs (one thread per core), 61 GiB RAM. Both backends used the
locked release build above (opt-level 2, Rust 1.99.1, Clang/lld 18.1.3) with
CPUs 12–15, four Tokio workers, up to two blocking threads and four target
partitions. The fixtures were on Btrfs on the local NVMe instance-store
volume under `$HOME/.cache`. The short input was 32,768 rows, eight row groups
and 627,668 bytes across four files (expected count 1,456); the long input was
16,777,216 rows, 128 row groups and 311,529,117 bytes across 16 files (expected
count 9,883,940). All 76,219 file pages (4 KiB) were resident before and
just after every arm. One-second `vmstat` samples immediately before each
comparison showed 98–99% CPU idle; no builds were active. The workspace was
not reserved, and the samples do not exclude transient host activity.

Each load has four pairs: 600 offers per phase at 50 and 200 offers/s, and
200 per phase at 20 offers/s. The 20 offers/s follow-up was chosen after the
initial comparison showed drops at both higher rates. Latencies are scheduled
arrival to stream exhaustion among completed requests. Timings and throughput
below are **medians of the four runs, with min–max across runs**, not pooled
percentiles. Counts and percentages are over all offers for the row. No query
or scalar checks failed; 7,330 of 7,334 completed mixed foreground queries
started execution while a scan was in flight.

| Offers/s | Phase    | Backend    | Completed samples (% of offers) | Dropped (% of offers) |                 p50 ms |                 p95 ms |                 p99 ms |
| -------: | :------- | :--------- | ------------------------------: | --------------------: | ---------------------: | ---------------------: | ---------------------: |
|       20 | Isolated | `off`      |                   800 (100.00%) |             0 (0.00%) |       2.72 (2.71–2.76) |       3.24 (3.08–3.43) |       3.59 (3.45–4.09) |
|       20 | Isolated | `required` |                   800 (100.00%) |             0 (0.00%) |       2.77 (2.75–2.80) |       2.94 (2.91–3.04) |       3.15 (3.09–3.39) |
|       20 | Mixed    | `off`      |                   800 (100.00%) |             0 (0.00%) |    43.81 (36.23–44.78) |   87.69 (77.29–113.44) |  111.79 (99.68–124.49) |
|       20 | Mixed    | `required` |                   800 (100.00%) |             0 (0.00%) |    12.20 (11.81–12.25) |    17.36 (16.69–17.76) |    19.76 (19.62–22.26) |
|       50 | Isolated | `off`      |                 2,400 (100.00%) |             0 (0.00%) |       2.61 (2.57–2.67) |       2.92 (2.85–3.12) |       3.44 (3.10–3.55) |
|       50 | Isolated | `required` |                 2,400 (100.00%) |             0 (0.00%) |       2.70 (2.66–2.75) |       2.91 (2.84–3.12) |       3.14 (2.96–3.49) |
|       50 | Mixed    | `off`      |                  1,781 (74.21%) |          619 (25.79%) | 440.37 (395.02–448.69) | 604.81 (576.06–646.65) | 653.88 (600.55–707.71) |
|       50 | Mixed    | `required` |                 2,400 (100.00%) |             0 (0.00%) |    12.07 (11.78–12.08) |    17.39 (17.31–17.92) |    20.30 (19.86–21.14) |
|      200 | Isolated | `off`      |                 2,400 (100.00%) |             0 (0.00%) |       2.32 (2.32–2.42) |       2.68 (2.65–2.84) |       3.23 (2.74–3.84) |
|      200 | Isolated | `required` |                 2,400 (100.00%) |             0 (0.00%) |       2.41 (2.40–2.43) |       2.64 (2.59–2.65) |       3.07 (2.76–3.81) |
|      200 | Mixed    | `off`      |                    503 (20.96%) |        1,897 (79.04%) | 424.98 (395.70–445.91) | 583.51 (563.99–609.22) | 627.16 (596.03–642.10) |
|      200 | Mixed    | `required` |                  1,050 (43.75%) |        1,350 (56.25%) | 201.41 (196.32–206.35) | 224.40 (223.26–229.00) | 231.67 (230.03–234.68) |

| Offers/s | Backend    | Isolated foreground completions/s | Mixed foreground completions/s | Mixed background scans/s | Mixed background million input rows/s |
| -------: | :--------- | --------------------------------: | -----------------------------: | -----------------------: | ------------------------------------: |
|       20 | `off`      |                                20 |            19.99 (19.93–20.00) |         6.60 (6.40–6.70) |                110.73 (107.37–112.41) |
|       20 | `required` |                                20 |                          20.00 |         6.00 (5.80–6.00) |                 100.66 (97.31–100.66) |
|       50 | `off`      |                                50 |            36.64 (35.35–37.96) |         6.50 (6.42–6.58) |                109.05 (107.65–110.45) |
|       50 | `required` |                                50 |                          50.00 |                     5.75 |                                 96.47 |
|      200 | `off`      |                               200 |            39.16 (38.03–41.21) |         6.17 (6.00–6.33) |                103.46 (100.66–106.26) |
|      200 | `required` |                               200 |            83.19 (80.26–85.29) |                     5.00 |                                 83.89 |

At the lossless 20 offers/s level, the isolated p99s were similar, but the
native mixed p99 was 19.76 ms vs. 111.79 ms for stock. Native gave up about 9%
of long-scan throughput (6.00 vs. 6.60 scans/s). At 50 offers/s, native
completed all foreground requests while stock dropped 25.79%; at 200 offers/s
both dropped requests, but native completed 43.75% vs. 20.96% for stock.
Native mixed p99 was also lower at those rates, but the percentiles describe
different subsets of requests and are not lossless speedup measurements.
Native background throughput was 12% lower at 50 and 19% lower at 200 offers/s
by median scans/s. This is a trade-off under the measured three-lane load,
not a single-query scan throughput result or a measurement of the scheduler
in isolation.

## Smoke test

```bash
timeout 1200 cargo test --release --locked --manifest-path benchmarks/morsel_mixed/Cargo.toml
```

The integration tests exercise the binary, check fixed arrivals, stock result
validation and real execution overlap, and verify that a wrong expected row
count across several output batches is reported as a failure after draining.
The Parquet test checks both native plans and result counts, and verifies that
unsupported SQL fails in required mode.
