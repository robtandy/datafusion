---
orphan: true
---

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Experimental morsel execution metrics

Attach a `MorselMetrics` extension to `SessionConfig` (or the config of an
individual `TaskContext`). This is independent of `morsel_mode`: it measures
only native runs, not stock queries or fallbacks. For example, with `fixture`
registered as a supported table on the session:

```rust
use datafusion::morsel::MorselMetrics;
use datafusion::prelude::{SessionConfig, SessionContext};
use std::sync::Arc;

let metrics = Arc::new(MorselMetrics::default());
let mut config = SessionConfig::new().with_extension(Arc::clone(&metrics));
config.options_mut().execution.morsel_mode = datafusion::common::config::MorselMode::Required;
let ctx = SessionContext::new_with_config(config);
// Register `fixture` on `ctx` before executing the query.
ctx.sql("SELECT * FROM fixture").await?.collect().await?;
for run in metrics.take_runs() {
    for metric in &run {
        println!("{metric}"); // Includes the worker label where applicable.
    }
}
```

`take_runs()` drains one DataFusion `MetricsSet` per execution, in completion
order. For correlation with concurrent executions, give each its own collector
via its task context. An unpolled stream has no report. A failed or cancelled run
reports partial measurements after the stream and its tasks release the run;
that may occur after the caller sees an error or drops the stream. These sets
are separate from `ExecutionPlan::metrics()` and `EXPLAIN ANALYZE`, which still
show the original plan-lifetime metrics.

| Metric                                                                   | Units and boundary                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `morsel_pending_io`                                                      | Nanoseconds between a pending planner I/O poll and its next poll (or drop), summed over I/O phases, including footer/index/data reads during execution. Excludes waiting for an I/O slot, synchronous poll work, and registration/planning before execution.                                  |
| `morsel_ready_wait`                                                      | Nanoseconds suspended in a worker's ready-reader claim, including receiver-lock wait and a wait that ends with EOF. Not I/O performed by that worker. Zero for in-memory input.                                                                                                               |
| `morsel_decode_work`                                                     | Nanoseconds of synchronous reader setup and batch polling; zero for in-memory input.                                                                                                                                                                                                          |
| `morsel_process_work`                                                    | Nanoseconds of in-memory batch reads, worker filters/projections and local count updates. Excludes source-side reader preparation, count combine/finalize and the final count result projection.                                                                                              |
| `morsel_output_wait`                                                     | Nanoseconds suspended obtaining an output-channel permit; measures result backpressure, not time the consumer spends waiting for data. Zero for inline output and worker-local counts.                                                                                                        |
| `morsels_claimed`, `morsels_completed`                                   | Number of Parquet readers (one per row group) or in-memory ranges of up to four input batches. A claimed reader completes on reader EOF; an in-memory range completes after its last batch is processed and any result handed off. Filtered-out batches still count as work.                  |
| `morsel_peak_ready_depth`, `morsel_peak_ready_compressed_bytes_estimate` | Peak enqueued readers and the maximum sum of their full row-group compressed sizes from the Parquet footer. The bytes are a proxy: they include unprojected columns and do not measure resident reader allocations.                                                                           |
| `morsel_peak_output_depth`, `morsel_peak_output_arrow_bytes`             | Peak enqueued result batches and peak Arrow backing-buffer capacity reachable from queued batches, in bytes. Overlapping ranges are counted once across columns, nested arrays and batches. This excludes array/schema objects, allocator overhead and storage capacity not exposed by Arrow. |

Worker metrics carry a `worker` label (an index within the run, not a thread
ID). Use `metric.value().as_usize()` for raw counts, bytes and nanoseconds.
Sum individual worker values for a run total; concurrent phases can overlap.
These are monotonic-clock durations, not thread CPU times. Waits include time
until the task gets polled again, including cooperative/runtime scheduling.
Explicit pipeline yields are outside the timers. The timed buckets do not
partition total query wall time.

Queue peaks count only published, not-yet-dequeued entries. Ready reservations
use the same two permits as ready entries; at most one reader per worker can
also be held outside that queue. The output queue is bounded to two batches,
plus at most one batch per worker parked waiting to send; the count sink enqueues
at most one final batch. Bytes held in those tasks, I/O requests, decoder
state, source tables or batches already delivered to the caller are not queue
bytes. The item capacities are **not** memory limits. Peak byte and item counts
need not occur at the same instant.

## Parquet smoke check

The original [#128 comparison](https://github.com/robtandy/datafusion/blob/4c43c428f0b2a6c4c1013875be01093707e063ec/benchmarks/morsel_parquet/README.md)
remains pinned to `d649cacf00fee7125f45ea1427a023b2eab732b1`. Before changing
runtime code, the pinned runner was rechecked on `rob-tandy-dduck1` with the
2-file smoke fixture, 4 workers and `scan_high`: 3,506 rows, 6 output batches,
6 claims, and 6.01 ms total in that one sanity run. This does not replace or
update the baseline.

For a matched check of this revision, copy the #128 `stream_runner.rs` to a
local `datafusion/core/examples/morsel_metrics_smoke.rs`. Just before building
`SessionContext`, add the following (use
`datafusion::morsel::MorselMetrics`):

```rust
let metrics = Arc::new(MorselMetrics::default());
if std::env::var_os("MORSEL_METRICS").is_some() {
    config.set_extension(Arc::clone(&metrics));
}
```

Call `metrics.take_runs()` **after** recording `seconds`, so report formatting
remains outside the timed drain. Use the same runner, Parquet fixture and #128
SQL for both arms:

```bash
MORSEL_BENCH_REVISION=$(git rev-parse HEAD) timeout --kill-after=15s 900s \
  cargo build --locked --profile release-nonlto -p datafusion \
  --no-default-features --features parquet,sql --example morsel_metrics_smoke
runner=target/release-nonlto/examples/morsel_metrics_smoke
# Adjust CPU affinity to the host. Use the #128 fixture registration and SQL.
env -u MORSEL_METRICS timeout 30s taskset -c 12-15 "$runner" measure required 4 "$ddl" "$sql"
MORSEL_METRICS=1 timeout 30s taskset -c 12-15 "$runner" measure required 4 "$ddl" "$sql"
```

The environment variable is only part of this scratch example, not a
DataFusion option. With the runtime changes in `52e0a64e3` on `rob-tandy-dduck1`
(Rust 1.99.1), both arms passed the #128 NumPy oracle on the 6,000-row fixture. A timing smoke
check used the pre-read 16-file, 128-row-group fixture, four workers on CPUs
12–15, and six matched pairs per case after one warmup pair, alternating which
arm ran first. Every pair matched plans, output rows/batches and legacy source
counters. Native measured runs reported 128 claims and completions; the stock
control produced no detailed reports in either arm.

| Case                | Total ms, off median (range) | Total ms, on median (range) | First batch ms, off → on median |
| ------------------- | ---------------------------: | --------------------------: | ------------------------------: |
| Native `scan_low`   |          612.0 (590.4–703.7) |         605.7 (586.4–650.0) |                     16.0 → 16.7 |
| Native `count_low`  |          249.4 (227.6–257.8) |         246.0 (231.0–284.6) |                   249.3 → 246.0 |
| Native `scan_high`  |          699.0 (597.1–772.9) |         671.2 (647.7–719.5) |                     18.0 → 16.6 |
| Native `count_high` |          258.7 (237.7–276.7) |         257.4 (218.6–284.0) |                   258.6 → 257.4 |
| Stock `scan_high`   |          470.0 (388.8–536.7) |         427.8 (366.3–454.3) |                     19.7 → 16.4 |

These are noisy smoke timings, not a claim of a speedup or a bound on overhead.
For example, one measured native `scan_high` run evenly split 128 readers
(31–33 per worker). The totals were 137 ms pending I/O, 302 ms ready wait,
1,167 ms decode, 236 ms process and 542 ms output wait over a 720 ms drain
(including registration/planning). Both queues peaked at two entries, with
158,272 bytes of Arrow buffers in the output queue and a 4,867,802-byte
compressed-size estimate in the ready queue.
