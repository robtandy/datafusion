<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Single-worker morsel execution regression (#144)

Native filtering materializes data that stock does not. Dropping the unused predicate column explains most of the filtered projection gap, and counting true predicate bits closes the filtered count gap. Explicit yields and streaming handoffs matter, but yielding once per morsel is not a safe global change: it more than doubles four-worker streaming time in these runs.

## Baseline reproduction

The regression from [#141](https://github.com/robtandy/datafusion/pull/141) reproduces without a concurrent build. On `rob-tandy-dduck2`, two reruns with the **unmodified** #141 executable gave these medians (round 1 / round 2, ms):

| One worker, 4,096 rows/input batch |   Stock first |    Native first |     Stock total |    Native total |
| ---------------------------------- | ------------: | --------------: | --------------: | --------------: |
| Filtered projection                | 0.560 / 0.600 |   0.032 / 0.039 | 19.213 / 19.321 | 25.692 / 25.945 |
| Filtered `COUNT(*)`                | 9.388 / 9.556 | 13.686 / 13.859 |   9.392 / 9.560 | 13.688 / 13.861 |

This is **34% extra total time for projection and 45–46% for count**. Native streaming delivered its first smaller batch sooner; these are not times to an equal number of rows. Count's first result is the finalized row. The four-worker baseline also reproduced the direction of #141: at round 1, native vs. stock total was 7.321 vs. 6.536 ms for balanced projection and 3.836 vs. 2.640 ms for balanced count, but 7.375 vs. 17.765 ms and 3.835 vs. 8.199 ms, respectively, under partition skew.

## Measurement protocol

- Pinned `f05fbe99e5d4bdd0e452eef7a44663c0c16873e5` from `refs/pull/141/head`. The two checkouts and Cargo target directories were separate. Each process compared `off` and `required` in the same binary with the same Tokio and DataFusion settings; scratch variants changed only native execution. See [exact build/run commands and scratch recipes](morsel_single_worker_analysis/reproduction.md).
- Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`, 16 aarch64 Neoverse-N1 vCPUs, 61 GiB RAM, one NUMA node, no cgroup CPU quota. Rust and Cargo 1.99.1 from the pinned toolchain, Clang/lld 18.1.3, locked Arrow 60.1.0 and Tokio 1.54.1, optimized `release-nonlto` (16 codegen units), only the `sql` package feature. The guest exposed no CPU frequency governor; none was set. The unmodified reruns and four-worker ablations used affinity 12–15, with one or four Tokio pool threads as appropriate; single-worker ablations pinned the whole process to CPU 12. The drain runs on the query pool, with no separate consumer execution core or I/O/time drivers.
- The fixture is a preloaded `MemTable`: 8,388,608 rows, two non-null `Int64` columns (~128 MiB), `SmallRng` seed 127. Use 2,048 × 4,096 and 512 × 16,384 input batches, setting the session batch size to match. Both have 2,094,791 passing rows (`score < 64`). At 4,096 rows per batch, the four-worker layouts are `[512, 512, 512, 512]` and `[1664, 128, 128, 128]`; the latter skews partition sizes, not per-row cost. The short startup cases hold rows/input batch fixed at 4,096 but use 1 or 16 input batches.
- First is the time to the first **non-empty** batch after starting `execute_stream`; total ends after draining to EOF. Plan creation/reset, preloading, and the oracle run are untimed. Each invocation alternates stock/native order, discards five warmups, and collects 31 repetitions. Two full sweeps reverse the native variant order; the startup probes use one sweep. Schema, exact unordered IDs (or count value), and physical plan are checked on every case/mode before timing; the consumer checks row counts on every timed run. No wall-time threshold is a unit-test assertion. This pin predates #141's same-mode priming; with 31 alternating runs the median can favor one immediately preceding mode. Treat first-batch times, especially stock projection, as specific to this ordering rather than order-independent latency estimates.

### Verified plan shapes

The relevant physical nodes in output-to-input order were:

```text
stock, filtered projection:
ProjectionExec: expr=[id@0 + 1 as result]
  FilterExec: score@1 < 64, projection=[id@0]
    DataSourceExec: partitions=N, partition_sizes=[...]

stock, filtered count (one worker):
ProjectionExec: expr=[count(Int64(1))@0 as result]
  AggregateExec: mode=Single, gby=[], aggr=[count(Int64(1))]
    FilterExec: score@0 < 64, projection=[]
      DataSourceExec: partitions=1, partition_sizes=[...]

native, filtered projection:
MorselExec: workers=N, MemTable[items, projection=Some([0, 1])] -> Filter[score@1 < 64] -> Projection[id@0 + 1 AS result] -> ResultSink[capacity=2]

native, filtered count:
MorselExec: workers=N, MemTable[items, projection=Some([1])] -> Filter[score@0 < 64] -> Projection[] -> CountSink; FinalizedCountSource -> Projection[count(Int64(1))@0 AS result] -> ResultSink[capacity=2]
```

With four workers, stock count instead has `AggregateExec(Final) → CoalescePartitionsExec → AggregateExec(Partial) → FilterExec → DataSourceExec`; stock streaming projection's partitions are merged by `execute_stream`. The unfiltered projection control has only `ProjectionExec → DataSourceExec` in stock and `MemTable[projection=Some([0])] → Projection → ResultSink` in `MorselExec`. Native explains describe the lowered plan; the temporary dispatch selects the ablation at execution, so an `inline` plan still prints `ResultSink` but does not instantiate the channel.

### Host load

Ran 2026-11-18 UTC. No concurrent builds were observed during timing; both build commands returned before executing the binaries. The one-minute load averages immediately after building were still 8.86 and 11.10 (16 vCPUs) for the original and extended runs. In the extended run after the 15-second cooldown started, host-wide busy time in one-second `/proc/stat` samples ranged from 0.6% to 36.1% including the benchmark, with no recorded steal; for example, one pre-run sample showed CPUs 12–15 at 2%, 0%, 1%, and 0% busy. Load fell to 3.80 by the end. This was a quiet interval on a shared VM, not an isolated host. In the extended runs, the stock projection median ranged from 19.517 to 19.937 ms at 4,096 rows/batch and 12.855 to 13.123 ms at 16,384; stock count ranged from 9.508 to 9.934 and 5.152 to 5.505 ms, respectively.

## One-worker ablations

- `baseline`: original native worker, with an explicit `yield_now` after every input batch.
- `budget`: same worker topology, operators and batch sizes; yield after each claimed morsel of up to four input batches. At 4,096 rows/batch this reduces explicit yields from 2,048 to 512, and at 16,384 from 512 to 128. Channel waits and per-input cancellation checks are unchanged.
- `inline`: starting from `budget`, let the drain poll the one-worker pipeline directly, removing the supervisor, worker task, joins, result channel and atomic claims. Same source reads, filters, projection and count operators. This is a diagnostic for the combined topology cost, not a measurement of queue instructions alone.
- `prune`: for filtered projection only, evaluate the mask on `id, score`, then apply it to `id` alone, keeping per-input outputs, sends and yields. `score` is not needed after the predicate. The test guard and input projection stay in the timer.
- `material`: for filtered projection, use Arrow's fused filter/coalescer before projection, still retaining `id, score`. For count, accumulate true bits from the predicate mask instead of making a filtered `score` batch followed by a zero-column projection. Both retain the baseline yield cadence. `both` adds `budget` to `material`.

Medians in ms, **round 1 / round 2**. The stock row is from the same executable and invocation as `baseline` in each round. `prune` does not change count, so its redundant count timings are omitted. For scale, at 4,096 rows the round-1 p10–p90 total intervals were 19.207–19.695 ms (stock), 25.986–26.337 ms (baseline), and 20.801–21.032 ms (prune) for projection; 9.137–9.859 ms (stock), 13.726–14.007 ms (baseline), and 8.360–8.608 ms (material) for count. These are empirical percentiles, not confidence intervals.

| Rows/input batch | Variant     | Filtered projection first | Filtered projection total |     Count first |     Count total |
| ---------------: | ----------- | ------------------------: | ------------------------: | --------------: | --------------: |
|             4096 | stock `off` |             0.563 / 0.617 |           19.519 / 19.937 |   9.504 / 9.847 |   9.508 / 9.860 |
|             4096 | baseline    |             0.033 / 0.034 |           26.134 / 26.323 | 13.864 / 14.095 | 13.866 / 14.098 |
|             4096 | budget      |             0.064 / 0.054 |           25.407 / 25.284 | 13.244 / 13.103 | 13.246 / 13.105 |
|             4096 | inline      |             0.026 / 0.028 |           23.947 / 24.170 | 12.978 / 13.290 | 12.978 / 13.291 |
|             4096 | prune       |             0.029 / 0.032 |           20.931 / 21.061 |               — |               — |
|             4096 | material    |             0.091 / 0.093 |           26.904 / 27.089 |   8.473 / 8.432 |   8.475 / 8.435 |
|             4096 | both        |             0.123 / 0.125 |           26.029 / 26.371 |   7.851 / 8.155 |   7.853 / 8.157 |
|            16384 | stock `off` |             0.242 / 0.236 |           13.041 / 12.956 |   5.387 / 5.310 |   5.397 / 5.319 |
|            16384 | baseline    |             0.061 / 0.061 |           18.643 / 18.471 |   9.954 / 9.879 |   9.956 / 9.882 |
|            16384 | budget      |             0.130 / 0.127 |           18.529 / 18.443 |   9.711 / 9.597 |   9.713 / 9.599 |
|            16384 | inline      |             0.048 / 0.047 |           17.884 / 17.815 |   9.671 / 9.579 |   9.673 / 9.580 |
|            16384 | prune       |             0.043 / 0.049 |           12.759 / 12.514 |               — |               — |
|            16384 | material    |             0.218 / 0.220 |           20.229 / 20.020 |   4.964 / 4.867 |   4.966 / 4.869 |
|            16384 | both        |             0.320 / 0.321 |           20.144 / 19.740 |   4.967 / 4.644 |   4.969 / 4.646 |

### Output batch shape

Observed in the untimed oracle drain; sizes are min/median/max rows. Count emits **one batch of one row** in every case, including the parallel cases. It does not hand off filtered input batches to the drain.

| Streaming query       | Rows/input batch | Mode(s)                            | Output batches | Rows/batch (min/median/max)      |
| --------------------- | ---------------: | ---------------------------------- | -------------: | -------------------------------- |
| Filtered projection   |             4096 | stock / material / both            |            512 | 1735 / 4096 / 4096 (511 full)    |
| Filtered projection   |             4096 | baseline / budget / inline / prune |           2048 | 922 / 1023 / 1124                |
| Filtered projection   |            16384 | stock / material / both            |            128 | 14023 / 16384 / 16384 (127 full) |
| Filtered projection   |            16384 | baseline / budget / inline / prune |            512 | 3923 / 4092 / 4259               |
| Unfiltered projection |             4096 | all                                |           2048 | 4096 / 4096 / 4096               |
| Unfiltered projection |            16384 | all                                |            512 | 16384 / 16384 / 16384            |

For four-worker filtered projection (round 1), stock emitted 512 (balanced) or 514 (skewed) batches, with minima of 3,065 or 25 rows, respectively; the median and maximum were 4,096 in both. Native baseline and budget still sent 2,048 batches at 922/1,023/1,124 rows. Coalesced native output has a separate tail per worker and can vary by claims: `material` emitted 513 (balanced) and 512 (skewed) in that drain, with median and maximum 4,096 and minima of 170 and 3,183, respectively.

### Startup and streaming control

The short runs bound the scale of startup **plus execution and teardown**; they do not time `spawn` in isolation. Medians (µs) for the **native** variants:

| Input batches | Variant  | Filtered projection first | Filtered projection total | Count first | Count total |
| ------------: | -------- | ------------------------: | ------------------------: | ----------: | ----------: |
|             1 | baseline |                        21 |                        24 |          21 |          22 |
|             1 | budget   |                        21 |                        24 |          21 |          22 |
|             1 | inline   |                        18 |                        19 |          16 |          16 |
|            16 | baseline |                        25 |                       181 |         106 |         106 |
|            16 | budget   |                        41 |                       171 |          97 |          98 |
|            16 | inline   |                        18 |                       153 |          91 |          92 |

At one batch, `budget` and `inline` differ by about 5–6 µs total, not milliseconds. At 2,048 batches, removing the topology from `budget` saves 1.11–1.46 ms for projection with unchanged outputs, but changes count by only +0.19 to −0.27 ms. Without any filter, at 4,096 rows/batch the projection control also improves from 9.132–9.456 ms (`budget`) to 7.871–7.979 ms (`inline`), with no shape change. Its round-1 first/total medians were 0.390/8.385 ms for stock, 0.028/9.677 ms for native baseline, 0.037/9.132 ms for budget, and 0.016/7.871 ms for inline. This points to real per-batch streaming handoff cost, not to startup or one result-channel operation per COUNT input batch. The inline delta also includes atomic claims/checks and changed poll timing; it is not additive with the other ablations.

### CPU accounting

`getrusage(RUSAGE_THREAD)` deltas bracket the drain on the single Tokio worker, including its producer/supervisor polls but not the resting main thread. At 4,096 rows/batch, round-1 median user CPU for stock / baseline / budget / inline was 19.01 / 26.00 / 25.00 / 24.00 ms for projection and 9.01 / 14.00 / 13.00 / 13.00 ms for count. `prune` used 21.00 ms user CPU for projection; the count `material` path used 8.99 ms. Across the one-worker sweeps median system CPU was at most 2 µs, and median voluntary and involuntary OS context switches were zero. CPU accounting in this guest is coarse; these counters distinguish user vs. kernel time, not individual Tokio polls. No privileged changes or hardware sampling were used (`perf` was not installed).

Tokio's `yield_now()` returns `Pending` and defers the task's wake to the runtime; the executor may poll it again on the same OS thread without ever descheduling that thread, and may even poll it again before another task. It does not imply a kernel thread context switch. The controlled yield relocation shows a user-space scheduling cost plus possible wake/order effects, not a syscall cost.

## Four-worker guardrails

4,096 rows/input batch, round-1 medians in ms:

| Layout   | Mode        | Filtered projection first | Filtered projection total | Count first | Count total |
| -------- | ----------- | ------------------------: | ------------------------: | ----------: | ----------: |
| Balanced | stock `off` |                     0.296 |                     6.642 |       2.684 |       2.687 |
| Balanced | baseline    |                     0.049 |                     7.661 |       3.894 |       3.896 |
| Balanced | budget      |                     0.060 |                    17.555 |       3.616 |       3.618 |
| Balanced | material    |                     0.114 |                     7.355 |       2.464 |       2.466 |
| Balanced | both        |                     0.096 |                    12.099 |       2.167 |       2.170 |
| Skewed   | stock `off` |                     0.211 |                    18.207 |       8.660 |       8.662 |
| Skewed   | baseline    |                     0.049 |                     7.664 |       4.035 |       4.038 |
| Skewed   | budget      |                     0.065 |                    17.542 |       3.662 |       3.664 |
| Skewed   | material    |                     0.121 |                     7.473 |       2.504 |       2.506 |
| Skewed   | both        |                     0.099 |                    12.570 |       2.193 |       2.195 |

The bounded budget **fails the streaming guardrail**: about 2.3× the baseline total in both layouts, also in reverse order (17.488 vs. 7.545 ms balanced, 17.306 vs. 7.604 ms skewed). It almost erases the skew win. The unfiltered control similarly rises from 3.506 to 8.582 ms balanced and from 3.516 to 8.592 ms skewed, so no filter change is required to trigger it. Changing when producers poll/wait on a capacity-two channel is a plausible contributor, but this experiment did not instrument sends, wakes or task placement on the four-worker pool; the mechanism for that regression remains unresolved. Count, with no per-batch send, does not have that guardrail failure.

## What explains the gap

1. **The explicit task yield is measurable but secondary.** Budget saves 0.62–1.04 ms for the two original workloads with 2,048 batches, and at most 0.29 ms with 512 batches. Projection first-result latency rises when moving the yield later; it can keep running until the bounded channel applies backpressure. Reducing yields globally would exchange small single-worker gains for a large parallel streaming regression.
2. **Filter materialization is the strongest tested cause.** Stock projects out `score` before the filtered projection materializes rows; pruning `score` in the native worker saves about 5.2 ms of the 6.4–6.6 ms gap at 4,096 and removes the gap at 16,384 without changing yields, sends, output sizes or work claims. Coalescing the native output to stock's batch shape but **still copying `score`** makes the single-worker projection slower, even with fewer channel messages. That ablation changes copying, projector invocation count and handoff frequency together and does not price coalescing in isolation. For count, stock feeds zero-column filtered batches to the aggregate while native constructs a filtered `score` array only to discard it. Using the mask cardinality without changing cadence or row handoff saves 5.0–5.7 ms and closes the count gap at both sizes.
3. **Residual:** At 4,096, projection with pruning is still about 1.1–1.4 ms slower than stock. The inline and unfiltered controls show handoff can cost on this scale, but they did not combine pruning with inline or reproduce the stock pipeline. Differences include fused filtering, batch shape, stream/poll behavior and work claiming. Do not assign the residual to one queue or sum the isolated effects.

## Recommended follow-up

**For native filtered `COUNT(*)`, accumulate the predicate mask's true count when the only input operators are `Filter` or `Filter → Projection[]`.** Keep the fallback for every other operator shape, the per-batch yield and closure check, supervisor/error propagation, and the one-row finalization path. Use Arrow's validity-aware `true_count` so null predicates do not count; don't elide predicate evaluation or swallow evaluation errors. This is a narrow optimization with a substantial gain and no added buffering.

Fairness/cancellation: the worker still explicitly yields after each synchronous input batch. There is still no mid-batch preemption for a costly predicate, and cheaper batches can change which worker wins a claim (not a guarantee of fair sharing). Receiver drop is still checked between inputs, and worker/peer aborts still take effect when the tasks return to Tokio. The per-morsel experiment, by contrast, can run up to four batches before a scheduling point if the channel does not block; bounded batches are not a time bound. Before landing a runtime change, retain tests for NULL, empty and all-rejected batches, error and early drop before finalization, and check short-query tails under background load using the #137 mixed-query harness. Do not ship the global budget or inline topology from this investigation.

### Count-only confirmation

Applied just the [count mask diff](morsel_single_worker_analysis/reproduction.md#minimal-count-only-confirmation-diff) in a third checkout and reran the unmodified #141 harness at 4,096 rows/batch. Runs were ordered original, patched, patched, original to reduce order bias; the one-minute host load fell from 5.40 to 4.03 with no concurrent builds. `--verify` and each timed case's untimed plan and exact result checks passed. Native count medians in ms (round 1 / round 2):

| Layout        |  Original first |    Mask first |  Original total |    Mask total |
| ------------- | --------------: | ------------: | --------------: | ------------: |
| One worker    | 13.851 / 14.075 | 8.488 / 8.371 | 13.853 / 14.077 | 8.491 / 8.374 |
| Four balanced |   3.829 / 3.879 | 2.441 / 2.441 |   3.835 / 3.882 | 2.443 / 2.443 |
| Four skewed   |   3.854 / 3.875 | 2.497 / 2.445 |   3.856 / 3.877 | 2.500 / 2.447 |

The patched binary's own stock totals were 9.896 / 9.338 ms, 2.747 / 2.645 ms and 8.548 / 8.193 ms, respectively. Native streaming projection totals moved by at most ~3.1% between the two executables in these rounds, not the ≥2× budget failure. The unmodified streaming path is a guardrail here, not evidence of statistical equivalence. The scratch patch passed nine integration and eight morsel unit tests, including nullable predicates, worker error and early drop; [commands and the unit-test feature retry](morsel_single_worker_analysis/reproduction.md) are recorded separately.

## Checks and scope

Only this report and its reproduction notes are submitted. The original and extended release harnesses passed `--verify` and the untimed correctness checks in all timed cases. No production runtime, #127 benchmark source, or #126 Parquet code changed. The full extended workspace test suite and the mixed-query workload were not run for this Markdown-only investigation.

`cargo fmt --all`, `cargo fmt --all --check`, and `timeout 900 taskset -c 0-11 cargo clippy --locked -j 4 --all-targets --all-features -- -D warnings` passed. `timeout 180 ./ci/scripts/doc_prettier_check.sh --write --allow-dirty` passed; that script does not cover `benchmarks/`, so the two notes were also checked directly with `npx prettier@3.10.1 --check`. `git diff --check` passed. `./dev/rust_lint.sh` stopped at its system Python PyYAML prerequisite.
