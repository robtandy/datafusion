<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Profiling the pipeline engine

The executor of the pipeline engine can profile its executions. A profile
records, for every operator, how long it ran and how long the threads were
blocked on it, the morsels, rows and bytes it consumed and produced, and the
memory it reserved; for every pipeline, every thread of a pipeline and the
execution as a whole, the morsels that were distributed, the time the threads
waited in the work queue of the runtime, and their active and idle time. So a
profile tells where the time of a query goes: which operator is slow, and
whether the query is CPU-bound or starved.

`EXPLAIN ANALYZE` shows the metrics that fit on its lines. The JSON dump of a
profile has all of them, with the operator tree and the pipelines it
decomposes into, in the schema [below](#json-schema). Tools read it (the
operator microbenchmarks and the comparison of our plans with DuckDB's): it is
a contract, see [Stability](#stability).

The code is in `src/profiling.rs`, the instrumentation of the push loop in
`src/executor.rs`.

## Enabling profiling

Profiling is off by default. A session enables it with the option
`pipeline_profiling.enabled` of the configuration extension
`ProfilingOptions`, which it registers:

```rust
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_physical_pipeline::profiling::ProfilingOptions;

// Profiles every execution of the session.
let config = SessionConfig::new().with_option_extension(ProfilingOptions::new(true));

// Profiles the executions once `SET pipeline_profiling.enabled = true`
// switched profiling on, in this session only.
let config = SessionConfig::new().with_option_extension(ProfilingOptions::default());
```

`Executor::with_profiling` enables (or disables) profiling for an executor,
whatever its session (e.g. to profile an operator tree that is not planned
from SQL).

Once an execution completed, its profile is `PipelineRootExec::profile` (the
profile of the last execution of the plan), `PipelineStream::profile`,
`ExecutionMetrics::profile` or `Executor::profile`. `QueryProfile::to_json`
(or `to_json_string`) dumps it:

```rust
let plan = ctx.sql(sql).await?.create_physical_plan().await?;
let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
let root = plan.downcast_ref::<PipelineRootExec>().unwrap();
let json = root.profile().unwrap().to_json_string();
```

A failed or cancelled execution has no profile.

## What is measured

### Morsels, rows and bytes

A _morsel_ is a `DataChunk`: at most 2048 rows, which a source hands to a
thread of a pipeline, and which flows through the operators of the pipeline
into its sink. Every operator counts the morsels, the rows and the bytes that
it consumed (`*_in`: the input chunks of `execute` and `sink`, each counted
once, also when the operator is called several times with the same chunk) and
that it produced (`*_out`: the non-empty output chunks of `get_data`,
`execute` and `final_execute`). The morsels of the source of a pipeline are
the morsels that the pipeline _distributed_ to its threads.

The bytes of a chunk are the bytes of its rows: for every column, the bytes of
the values of the rows of its slice (chunks are often slices of larger
batches), and of its null mask. Views count their 16 bytes per row and the
bytes of the values that are not inlined; strings with offsets and lists their
offsets and the values of their rows; dictionaries their keys and all of their
values. The bytes of a chunk with a selection vector are scaled to the
selected rows (they are an estimate). `profiling::chunk_bytes` and
`profiling::array_bytes` compute them, so that other measurements (e.g. the
operator microbenchmarks) count bytes in the same way.

### Time

Every thread of a pipeline (a _worker_, which runs a `PipelineExecutor`) is,
at any time, in one of three states, whose times add up exactly to its elapsed
time, from its spawn to its completion:

- **active** (`active_time`): in a poll of its task. It then calls the
  operators of the pipeline, whose _self time_ (`self_time`) is the time spent
  in their calls (`get_data`, `execute`, `final_execute`, `sink`, `combine`,
  the creation of their local states and the polls of the futures that they
  wait for), or runs the push loop between the calls (the _overhead_ of the
  executor).
- **blocked** (`blocked_time`): suspended because the last call returned
  `Blocked` (a source waiting for its input, e.g. I/O or a bridged
  DataFusion stream, an operator waiting for a future, a sink waiting for room
  in a full queue), until its task is polled again. The time is the blocked
  time of the operator of that call.
- **queued** (`queue_wait_time`): ready to run, but waiting for a thread of
  the runtime in its work queue: from its spawn to its first poll, and after
  it yielded (every 128 operator calls).

A thread that is woken while blocked waits in the work queue too, which
counts as blocked time.

Before the threads of a pipeline start, the executor creates the global states
of its operators (`setup_time`), and once all pipelines of a meta pipeline
completed, it finalizes their sink (`finalize_time`). It does so on the task
of the execution; these calls are part of the self time of their operators
and of the active time of the execution.

The `idle_time` of a pipeline is `threads * elapsed_time - active_time`: the
time that its threads could have been active but were not, because they were
blocked, queued, not started yet, or already done while other threads still
ran. The `idle_time` of the execution is
`threads * elapsed_time - active_time` too, with the `threads` of the
executor: it also counts the time that the execution ran pipelines with fewer
threads, or waited for pipelines to complete. So a CPU-bound query has an active time close to
`threads * elapsed_time`; a starved query has a large idle time: a large
blocked time if it waits for its input, a large queue wait if it waits for the
CPU, or else too few threads, too little parallelism.

### Additivity

The self times and the blocked times of all operators, the work-queue wait
and the overhead of the executor (`overhead_time`, the active time that is not
the self time of an operator) add up exactly to the `thread_time` of the
execution, the wall time of all of its threads: the sum of the elapsed times
of all workers, the setup of the pipelines and the finalization of the meta
pipelines. On one thread, with pipelines that run one after the other, the
thread time is the elapsed time of the execution, but for the time it takes
to schedule its pipelines.

So the sum of the self times and the blocked times of the operators equals
the thread time (and, on one thread, the elapsed time) of the execution, but
for the overhead of the executor and the work-queue wait. The tolerance is
**10%** of the thread time: the tests
`timings_add_up_to_the_elapsed_time_on_one_thread` and
`timings_add_up_to_the_thread_time` assert it, in debug builds, for a plan of
11 operators and 5 pipelines (the scans of two tables, a filter, two
projections, a hash join, an aggregation and a sort of 200,000 rows) on one
and on four threads, and `the_time_the_threads_wait_is_the_blocked_time_of_the_operator`
for a source that blocks. In these tests, the operators account for all but
about 1.3% of the thread time (and of the elapsed time on one thread), of
which the overhead of the executor is the larger part; the work-queue wait is
larger on a busy machine.

### Memory

The memory pool of a profiled execution wraps the pool of its session: it
forwards every call (so that the limits, the errors and the spilling are
those of the session), and attributes every memory consumer to the operator
whose call (or state creation, or finalization) registered it. The
`memory_peak` of an operator is the high-water mark of the memory reserved by
its consumers, all of its threads together. Consumers that are registered
outside of the calls of the operators (e.g. by tasks that a bridged plan
spawns) count towards the `unattributed_memory_peak` of the execution, and
all of them towards its `memory_peak`.

## Overhead

When profiling is off, the executor records only the metrics of
`OperatorMetrics` (as before), and checks once per operator call whether it
profiles. When profiling is on, it also takes a timestamp per poll of a
thread, computes the bytes of every chunk (which costs a loop over the views
of view columns), updates the counters of the thread (merged once per thread
into the profile of the pipeline), and records the reservations of the memory
pool (a lookup in a sharded map per reservation change).

## `EXPLAIN ANALYZE`

If the execution of an `EXPLAIN ANALYZE` was profiled, the rendering of its
plan also shows these development metrics of the profile (which
`datafusion.explain.analyze_level` and `datafusion.explain.analyze_categories`,
or the options of the statement, select like the other metrics, by the
category of their unit):

- on the line of the `PipelineRootExec`, `profile=[...]`: `elapsed_time`,
  `threads`, `morsels`, `active_time`, `idle_time`, `blocked_time`,
  `queue_wait_time` and `memory_peak` of the execution;
- on the line of every operator, after its other metrics: `blocked_time`,
  `rows_in`, `bytes_in`, `bytes_out` and `memory_peak`;
- on the line of every pipeline, `profile=[...]`: `threads`, `morsels`,
  `elapsed_time`, `active_time`, `idle_time`, `blocked_time` and
  `queue_wait_time`.

For example, `EXPLAIN ANALYZE SELECT d, COUNT(*) FROM t1 GROUP BY d` (over
the table `t1` of the tests, on 4 threads, in a debug build) shows

```text
PipelineRootExec: pipelines=2, profile=[elapsed_time=3.16ms, threads=4, morsels=12, active_time=7.26ms, idle_time=5.40ms, blocked_time=0ns, queue_wait_time=406.08µs, memory_peak=102.5 KB]
  PhysicalProjection: expr=[d@0 as d, count(Int64(1))@1 as count(*)], pipelines=[0], metrics=[output_rows=7, output_chunks=6, elapsed_compute=30.43µs, blocked_time=0ns, rows_in=7, bytes_in=135.0 B, bytes_out=135.0 B, memory_peak=0.0 B]
    PhysicalHashAggregate: gby=[d@0 as d], aggr=[count(Int64(1))], pipelines=[0, 1], metrics=[output_rows=7, output_chunks=6, sink_rows=5.00 K, sink_chunks=6, elapsed_compute=6.78ms, blocked_time=0ns, rows_in=5.00 K, bytes_in=40.4 KB, bytes_out=135.0 B, memory_peak=102.5 KB]
      MemoryScanSource: partitions=4, rows=5000, chunks=6, pipelines=[1], metrics=[output_rows=5.00 K, output_chunks=6, elapsed_compute=35.86µs, blocked_time=0ns, rows_in=0, bytes_in=0.0 B, bytes_out=40.4 KB, memory_peak=0.0 B]
  Pipeline 1: MemoryScanSource -> PhysicalHashAggregate, profile=[threads=4, morsels=6, elapsed_time=2.43ms, active_time=6.16ms, idle_time=3.57ms, blocked_time=0ns, queue_wait_time=179.75µs]
  Pipeline 0: PhysicalHashAggregate -> PhysicalProjection -> CollectorSink, after pipelines [1], profile=[threads=4, morsels=6, elapsed_time=472.45µs, active_time=1.08ms, idle_time=813.93µs, blocked_time=0ns, queue_wait_time=226.34µs], metrics=[output_rows=7, elapsed_compute=6.94ms, output_bytes=48.3 KB, output_batches=6]
```

(DataFusion appends the metrics of the `PipelineRootExec` itself to the last
line.)

The JSON dump has all metrics.

## JSON schema

`QueryProfile::to_json` returns one JSON object. All metrics are objects
`{"value": <value>, "unit": "<unit>"}` whose value is a non-negative integer;
the `metrics` of every object map the names of its metrics to them. Operators,
pipelines and meta pipelines are listed by id: the id of every one is its
index in its array, and the other objects refer to them by id. Times are
summed over all threads; points in time (`*_at`) are the times since the
execution started.

#### Units

| Unit      | Meaning                                                                |
| --------- | ---------------------------------------------------------------------- |
| `ns`      | nanoseconds: durations, and points in time since the execution started |
| `rows`    | rows                                                                   |
| `morsels` | morsels: chunks of at most 2048 rows                                   |
| `bytes`   | bytes, see [Morsels, rows and bytes](#morsels-rows-and-bytes)          |
| `count`   | a number of things, e.g. of threads or of calls                        |

#### Top level

| Field            | Type   | Meaning                                                     |
| ---------------- | ------ | ----------------------------------------------------------- |
| `schema`         | string | `"datafusion-physical-pipeline/profile"`                    |
| `version`        | number | the version of the schema, `1`, see [Stability](#stability) |
| `root`           | number | the id of the root of the operator tree: always `0`         |
| `executor`       | object | the execution as a whole                                    |
| `operators`      | array  | the operators, by id                                        |
| `pipelines`      | array  | the pipelines, by id (the ids of `EXPLAIN`)                 |
| `meta_pipelines` | array  | the meta pipelines, by id                                   |

#### Executor fields

| Field     | Type   | Meaning                      |
| --------- | ------ | ---------------------------- |
| `metrics` | object | the metrics of the execution |

#### Executor metrics

| Metric                     | Unit      | Meaning                                                                                             |
| -------------------------- | --------- | --------------------------------------------------------------------------------------------------- |
| `threads`                  | `count`   | the maximum number of threads of a pipeline (`target_partitions` for a `PipelineRootExec`)          |
| `pipelines`                | `count`   | the pipelines                                                                                       |
| `workers`                  | `count`   | the threads of all pipelines                                                                        |
| `morsels`                  | `morsels` | the morsels that the sources of all pipelines distributed                                           |
| `elapsed_time`             | `ns`      | the wall time of the execution                                                                      |
| `thread_time`              | `ns`      | the wall time of all threads: `active_time + blocked_time + queue_wait_time`                        |
| `active_time`              | `ns`      | the active time of all pipelines (with their setup), and the time of the finalizations              |
| `blocked_time`             | `ns`      | the blocked time of all threads                                                                     |
| `queue_wait_time`          | `ns`      | the work-queue wait of all threads                                                                  |
| `idle_time`                | `ns`      | `threads * elapsed_time - active_time` (or `0`)                                                     |
| `overhead_time`            | `ns`      | `active_time` minus the self times of all operators: the time of the executor itself                |
| `memory_peak`              | `bytes`   | the high-water mark of the memory reserved in the memory pool                                       |
| `unattributed_memory_peak` | `bytes`   | the high-water mark of the memory of the consumers registered outside of the calls of the operators |

#### Operator fields

The operator tree below the sink of the root meta pipeline (the
`CollectorSink` of the execution, id `0`), in pre-order, every operator once
(an operator that is the input of several operators is listed once), then the
operators that the pipelines have but the tree does not (e.g. the
`JoinHashTableBuild` source of the pipeline that builds the hash table of a
hash join, which produces no rows).

| Field       | Type   | Meaning                                                                                                             |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------------------- |
| `id`        | number | the id of the operator                                                                                              |
| `kind`      | string | the kind of the operator, the same for all operators of its type: `PhysicalOperator::name`, e.g. `PhysicalHashJoin` |
| `name`      | string | the operator as `EXPLAIN` describes it, with its parameters: `PhysicalOperator::description`                        |
| `inputs`    | array  | the ids of its inputs (its children), in order                                                                      |
| `pipelines` | array  | the pipelines it takes part in: objects with the id of the `pipeline` and the `role` of the operator in it          |
| `metrics`   | object | its metrics, summed over all threads of all of its pipelines                                                        |

A `role` is `"source"`, `"operator"` or `"sink"`. A pipeline breaker is the
sink of a pipeline and the source (or an operator, for the probe side of a
join) of another: its inputs whose pipelines it is the sink of are the inputs
that it consumes before it produces any output (e.g. the build side of a hash
join).

#### Operator metrics

| Metric         | Unit      | Meaning                                                                                                     |
| -------------- | --------- | ----------------------------------------------------------------------------------------------------------- |
| `self_time`    | `ns`      | the time in the calls of the operator, with the creation of its states and the finalization of its sink     |
| `blocked_time` | `ns`      | the time that threads were suspended because a call of the operator returned `Blocked`                      |
| `calls`        | `count`   | the calls of `get_data`, `execute`, `final_execute` and `sink`                                              |
| `rows_in`      | `rows`    | the rows of the chunks that it consumed, as an operator and as a sink                                       |
| `morsels_in`   | `morsels` | the chunks that it consumed                                                                                 |
| `bytes_in`     | `bytes`   | the bytes of the chunks that it consumed                                                                    |
| `rows_out`     | `rows`    | the rows of the chunks that it produced, as a source and as an operator                                     |
| `morsels_out`  | `morsels` | the non-empty chunks that it produced                                                                       |
| `bytes_out`    | `bytes`   | the bytes of the chunks that it produced                                                                    |
| `memory_peak`  | `bytes`   | the high-water mark of the memory reserved by the consumers that it registered, all of its threads together |

#### Pipeline fields

| Field               | Type    | Meaning                                                                                         |
| ------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `id`                | number  | the id of the pipeline                                                                          |
| `meta_pipeline`     | number  | the id of its meta pipeline                                                                     |
| `source`            | number  | the id of its source                                                                            |
| `operators`         | array   | the ids of its operators, from the source to the sink                                           |
| `sink`              | number  | the id of its sink                                                                              |
| `dependencies`      | array   | the ids of the pipelines of its meta pipeline that complete before it starts                    |
| `meta_dependencies` | array   | the ids of the meta pipelines that are finalized before it starts                               |
| `skipped`           | boolean | whether it skipped its work (an operator or its sink had finished, or could not produce output) |
| `metrics`           | object  | its metrics                                                                                     |
| `stages`            | array   | its stages: its source, its operators and its sink, in this order                               |
| `workers`           | array   | its threads, by thread index                                                                    |

#### Pipeline metrics

| Metric            | Unit      | Meaning                                                               |
| ----------------- | --------- | --------------------------------------------------------------------- |
| `threads`         | `count`   | its threads                                                           |
| `morsels`         | `morsels` | the morsels that its source distributed to its threads                |
| `rows`            | `rows`    | the rows of these morsels                                             |
| `started_at`      | `ns`      | when it started (its setup), since the execution started              |
| `finished_at`     | `ns`      | when its last thread completed                                        |
| `elapsed_time`    | `ns`      | `finished_at - started_at`                                            |
| `setup_time`      | `ns`      | the time it took to create its global states and to start its threads |
| `active_time`     | `ns`      | `setup_time`, and the active time of its threads                      |
| `blocked_time`    | `ns`      | the blocked time of its threads                                       |
| `queue_wait_time` | `ns`      | the work-queue wait of its threads                                    |
| `idle_time`       | `ns`      | `threads * elapsed_time - active_time` (or `0`)                       |

#### Stage fields

| Field      | Type   | Meaning                                                        |
| ---------- | ------ | -------------------------------------------------------------- |
| `operator` | number | the id of the operator                                         |
| `role`     | string | its role in the pipeline: `"source"`, `"operator"` or `"sink"` |
| `metrics`  | object | the metrics of the operator in this pipeline                   |

#### Stage metrics

The metrics of the operator in this pipeline, summed over the threads of the
pipeline. The metrics of an operator are the sums of the metrics of its
stages, but for the finalization of its sink, which is part of its
`self_time` only, and for its `memory_peak`.

| Metric         | Unit      | Meaning          |
| -------------- | --------- | ---------------- |
| `self_time`    | `ns`      | as for operators |
| `blocked_time` | `ns`      | as for operators |
| `calls`        | `count`   | as for operators |
| `rows_in`      | `rows`    | as for operators |
| `morsels_in`   | `morsels` | as for operators |
| `bytes_in`     | `bytes`   | as for operators |
| `rows_out`     | `rows`    | as for operators |
| `morsels_out`  | `morsels` | as for operators |
| `bytes_out`    | `bytes`   | as for operators |

#### Worker fields

| Field     | Type   | Meaning                                 |
| --------- | ------ | --------------------------------------- |
| `thread`  | number | the index of the thread in the pipeline |
| `metrics` | object | its metrics                             |

#### Worker metrics

| Metric            | Unit      | Meaning                                                                                   |
| ----------------- | --------- | ----------------------------------------------------------------------------------------- |
| `started_at`      | `ns`      | when the thread was spawned, since the execution started                                  |
| `finished_at`     | `ns`      | when it completed                                                                         |
| `elapsed_time`    | `ns`      | its time from its spawn to its completion: `active_time + blocked_time + queue_wait_time` |
| `active_time`     | `ns`      | its time in polls of its task                                                             |
| `blocked_time`    | `ns`      | its time suspended on a `Blocked` result                                                  |
| `queue_wait_time` | `ns`      | its time ready to run, waiting for a thread of the runtime                                |
| `morsels`         | `morsels` | the morsels it got from the source                                                        |
| `rows`            | `rows`    | the rows of these morsels                                                                 |

#### Meta pipeline fields

| Field       | Type           | Meaning                                   |
| ----------- | -------------- | ----------------------------------------- |
| `id`        | number         | the id of the meta pipeline               |
| `sink`      | number         | the id of its sink                        |
| `pipelines` | array          | the ids of its pipelines                  |
| `children`  | array          | the ids of its child meta pipelines       |
| `parent`    | number or null | the id of its parent, `null` for the root |
| `metrics`   | object         | its metrics                               |

#### Meta pipeline metrics

| Metric          | Unit | Meaning                                                                     |
| --------------- | ---- | --------------------------------------------------------------------------- |
| `finalize_time` | `ns` | the time it took to finalize its sink (part of the `self_time` of the sink) |
| `finalized_at`  | `ns` | when its sink was finalized, since the execution started                    |

### Example

The profile of the execution of `SELECT d, COUNT(*) FROM t1 GROUP BY d` of
[`EXPLAIN ANALYZE`](#explain-analyze), with one element of every array (of
the operators, the pipeline, the stages, the threads and the meta pipelines):

```json
{
  "schema": "datafusion-physical-pipeline/profile",
  "version": 1,
  "root": 0,
  "executor": {
    "metrics": {
      "threads": { "value": 4, "unit": "count" },
      "pipelines": { "value": 2, "unit": "count" },
      "workers": { "value": 8, "unit": "count" },
      "morsels": { "value": 12, "unit": "morsels" },
      "elapsed_time": { "value": 3128454, "unit": "ns" },
      "thread_time": { "value": 7473412, "unit": "ns" },
      "active_time": { "value": 7110635, "unit": "ns" },
      "blocked_time": { "value": 0, "unit": "ns" },
      "queue_wait_time": { "value": 362777, "unit": "ns" },
      "idle_time": { "value": 5403181, "unit": "ns" },
      "overhead_time": { "value": 260095, "unit": "ns" },
      "memory_peak": { "value": 107260, "unit": "bytes" },
      "unattributed_memory_peak": { "value": 0, "unit": "bytes" }
    }
  },
  "operators": [
    {
      "id": 2,
      "kind": "PhysicalHashAggregate",
      "name": "PhysicalHashAggregate: gby=[d@0 as d], aggr=[count(Int64(1))]",
      "inputs": [3],
      "pipelines": [
        { "pipeline": 0, "role": "source" },
        { "pipeline": 1, "role": "sink" }
      ],
      "metrics": {
        "self_time": { "value": 6714862, "unit": "ns" },
        "blocked_time": { "value": 0, "unit": "ns" },
        "calls": { "value": 16, "unit": "count" },
        "rows_in": { "value": 5000, "unit": "rows" },
        "morsels_in": { "value": 6, "unit": "morsels" },
        "bytes_in": { "value": 41370, "unit": "bytes" },
        "rows_out": { "value": 7, "unit": "rows" },
        "morsels_out": { "value": 6, "unit": "morsels" },
        "bytes_out": { "value": 135, "unit": "bytes" },
        "memory_peak": { "value": 107260, "unit": "bytes" }
      }
    }
  ],
  "pipelines": [
    {
      "id": 1,
      "meta_pipeline": 1,
      "source": 3,
      "operators": [],
      "sink": 2,
      "dependencies": [],
      "meta_dependencies": [],
      "skipped": false,
      "metrics": {
        "threads": { "value": 4, "unit": "count" },
        "morsels": { "value": 6, "unit": "morsels" },
        "rows": { "value": 5000, "unit": "rows" },
        "started_at": { "value": 172249, "unit": "ns" },
        "finished_at": { "value": 2572193, "unit": "ns" },
        "elapsed_time": { "value": 2399944, "unit": "ns" },
        "setup_time": { "value": 35725, "unit": "ns" },
        "active_time": { "value": 6064556, "unit": "ns" },
        "blocked_time": { "value": 0, "unit": "ns" },
        "queue_wait_time": { "value": 180085, "unit": "ns" },
        "idle_time": { "value": 3535220, "unit": "ns" }
      },
      "stages": [
        {
          "operator": 3,
          "role": "source",
          "metrics": {
            "self_time": { "value": 28437, "unit": "ns" },
            "blocked_time": { "value": 0, "unit": "ns" },
            "calls": { "value": 10, "unit": "count" },
            "rows_in": { "value": 0, "unit": "rows" },
            "morsels_in": { "value": 0, "unit": "morsels" },
            "bytes_in": { "value": 0, "unit": "bytes" },
            "rows_out": { "value": 5000, "unit": "rows" },
            "morsels_out": { "value": 6, "unit": "morsels" },
            "bytes_out": { "value": 41370, "unit": "bytes" }
          }
        }
      ],
      "workers": [
        {
          "thread": 0,
          "metrics": {
            "started_at": { "value": 209508, "unit": "ns" },
            "finished_at": { "value": 2519040, "unit": "ns" },
            "elapsed_time": { "value": 2309532, "unit": "ns" },
            "active_time": { "value": 2261861, "unit": "ns" },
            "blocked_time": { "value": 0, "unit": "ns" },
            "queue_wait_time": { "value": 47671, "unit": "ns" },
            "morsels": { "value": 1, "unit": "morsels" },
            "rows": { "value": 2048, "unit": "rows" }
          }
        }
      ]
    }
  ],
  "meta_pipelines": [
    {
      "id": 1,
      "sink": 2,
      "pipelines": [1],
      "children": [],
      "parent": 0,
      "metrics": {
        "finalize_time": { "value": 17386, "unit": "ns" },
        "finalized_at": { "value": 2607688, "unit": "ns" }
      }
    }
  ]
}
```

## Stability

The schema is versioned (`version`). A change of the meaning, the unit or the
name of a field or a metric, or the removal of one, increments the version;
adding fields or metrics does not, so readers must ignore what they do not
know. The test `the_json_dump_parses_and_matches_the_documented_schema` checks
that the dump has exactly the fields and metrics, with the units, of the
tables of this document.
