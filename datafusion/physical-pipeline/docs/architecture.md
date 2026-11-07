<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# `datafusion-physical-pipeline` architecture

`datafusion-physical-pipeline` is a push-based, DuckDB-inspired pipeline
execution engine for DataFusion. Queries are parsed, analyzed, optimized and
physically planned by DataFusion exactly as usual. The resulting
`ExecutionPlan` is then _lowered_ into a tree of `PhysicalOperator`s, which is
decomposed into _pipelines_ that worker threads execute by pushing
`DataChunk`s from a source, through streaming operators, into a sink.

This document describes the design of the engine and the contracts that
operators must follow. It is the reference for everybody implementing native
operators in this crate.

## Hard requirement: byte-for-byte compatibility

Every query executed through `PipelineQueryPlanner` must produce **exactly**
the output of default DataFusion:

- the same output `SchemaRef`: field names, data types, nullability and
  metadata (field and schema level);
- the same Arrow data, compared byte for byte after canonicalization (see
  [Differential testing](#differential-testing)). Only the row order may
  differ, and only if the query has no top-level `ORDER BY`.

To make this achievable, the engine never re-plans a query: it starts from
the `ExecutionPlan` produced by DataFusion's `DefaultPhysicalPlanner`, reuses
its schemas and its `PhysicalExpr`s (with all coercions applied), and keeps
all scan pushdowns. Native operators must reproduce the semantics of the
DataFusion operator they replace, including error behavior.

## Overview

```text
SQL / DataFrame
     |  DataFusion: parse, analyze, optimize (logical)
     v
LogicalPlan
     |  PipelineQueryPlanner::create_physical_plan
     |    1. DefaultPhysicalPlanner -> optimized ExecutionPlan
     |    2. lower_execution_plan   -> PhysicalOperator tree
     v
PipelineRootExec (an ExecutionPlan with one output partition)
     |  execute(0): Executor::execute_stream
     |    1. PipelineGraph::try_new(CollectorSink(root)) -> pipelines
     |    2. Executor::execute: run pipelines in dependency order,
     |       each on up to `target_partitions` threads
     v
SendableRecordBatchStream (fed by CollectorSink)
```

## Crate layout

| Module                                                                             | Contents                                                                                  |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `chunk.rs`                                                                         | `DataChunk`, `SelectionVector`, `DEFAULT_VECTOR_SIZE`                                     |
| `operator.rs`                                                                      | `PhysicalOperator`, state traits, result enums, `TaskInterrupt`, `ExecutionContext`       |
| `pipeline.rs`                                                                      | `Pipeline`, `MetaPipeline`, `PipelineGraph`, `PipelineBuilder`, `MetaPipelineBuilder`     |
| `executor.rs`                                                                      | `PipelineExecutor` (push loop), `Executor` (scheduler), `CollectorSink`, `PipelineStream` |
| `bridge.rs`                                                                        | `BridgeOperator` (any `ExecutionPlan` as a source), `PipelineRootExec`                    |
| `planner.rs`                                                                       | `PipelineQueryPlanner`, `lower_execution_plan`                                            |
| `explain.rs`                                                                       | `DisplayAs` for `PipelineRootExec`, pipeline / metrics rendering                          |
| `test_utils.rs`                                                                    | differential byte-equivalence test harness (test only)                                    |
| `tuple_data/`, `sources/`, `operators/`, `join/`, `aggregate/`, `sort/`, `window/` | native operators (Waves 2-4)                                                              |

All dependencies needed by the native operators are declared in `Cargo.toml`
up front, and all modules are declared in `lib.rs`, so that work on different
operators touches disjoint files.

## Data: `DataChunk` and `SelectionVector`

A `DataChunk` is the unit of data that flows through a pipeline:

- `schema: SchemaRef`: the schema of the chunk (the schema of the operator
  that produced it);
- `columns: Vec<ArrayRef>`: the _physical_ columns, all of the same length;
- `selection: Option<SelectionVector>`: the physical rows that are logically
  part of the chunk, in output order;
- `len`: the logical number of rows (the selection length if there is a
  selection, otherwise the physical length). Stored explicitly so that chunks
  without columns still have a row count.

Chunks hold at most `DEFAULT_VECTOR_SIZE` (2048) rows. Sources split larger
batches with zero-copy slices.

A `SelectionVector` is an immutable, reference-counted list of `u32` row
indices (an Arrow `ScalarBuffer<u32>`, so it can be passed to the `take`
kernel without copying). Selective operators (filters) attach or refine a
selection instead of copying the columns:

- `SelectionVector::from_boolean_mask` selects the `true` entries of a
  predicate result (`NULL` is not selected, as in `WHERE`);
- `DataChunk::with_selection(sel)` interprets `sel` relative to the _logical_
  rows of the chunk and composes it with the existing selection
  (`SelectionVector::slice_sel`);
- `DataChunk::slice` slices the selection if there is one, otherwise the
  columns, and never copies data;
- `DataChunk::compact` / `into_record_batch` materialize the logical rows as
  a `RecordBatch`. They only copy (with `take`) if there is a selection that is
  not a contiguous range.

Rules for operators:

- `DataChunk::columns()` are the **physical** columns. An operator that reads
  values must either respect the selection or call `compacted_column` /
  `compact`. Never evaluate a fallible expression on unselected rows: a row
  removed by a filter must not cause an error (e.g. a division by zero).
- An output chunk is _cleared_ (`DataChunk::reset`) before an operator is
  called: it has no rows and no columns. Operators write their output by
  assigning a new chunk, e.g. `*output = DataChunk::from_record_batch(batch)`.
- The schema of produced chunks must be exactly the operator's schema,
  including metadata, because it ends up in the query result.

## Operators: `PhysicalOperator`

`PhysicalOperator` mirrors DuckDB's `PhysicalOperator`. An operator can play
up to three roles; all methods have defaults, so an operator implements only
the roles it plays.

| Role     | Methods                                                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Source   | `is_source`, `parallel_source`, `get_global_source_state`, `get_local_source_state`, `get_data`                              |
| Operator | `parallel_operator`, `get_global_operator_state`, `get_operator_state`, `execute`, `requires_final_execute`, `final_execute` |
| Sink     | `is_sink`, `parallel_sink`, `get_global_sink_state`, `get_local_sink_state`, `sink`, `combine`, `finalize`                   |
| All      | `name`, `schema`, `children`, `description` (for `EXPLAIN`), `build_pipelines`                                               |

Operators are immutable (`&self` everywhere) and can be executed several
times, concurrently. All execution state lives in state objects created by the
operator and owned by the executor:

- **global** states (`GlobalSourceState`, `GlobalOperatorState`,
  `GlobalSinkState`) exist once per execution and are shared by all threads,
  so they are `Send + Sync` and use interior mutability;
- **local** states (`LocalSourceState`, `OperatorState`, `LocalSinkState`)
  exist once per thread (per pipeline), so they are only `Send`.

States are passed as trait objects; downcast them with `as_any` /
`as_any_mut`, `downcast_ref` / `downcast_mut` (and `downcast_arc` for shared
global sink states), or with the `downcast_state` helpers, which return an
internal error instead of panicking. `EmptyState` is the default state of
every role.

Every method receives an `ExecutionContext` with the `TaskContext`
(configuration, memory pool, functions), the `thread_id` in
`0..num_threads`, the waker of the current task (see
[Blocking](#blocking-taskinterrupt)) and the per-execution registries:

- `ctx.global_sink_state(self)`: the global sink state of this operator in
  the current execution. This is how an operator that is a sink _and_ a
  source or operator reaches the data it collected (hash join: the hash
  table, aggregate: the groups, sort: the sorted runs).
- `ctx.metrics(self)`: the `OperatorMetrics` of the operator.

### Source role

`get_data(ctx, gstate, lstate) -> Result<SourceResult>` returns

- `SourceResult::Chunk(chunk)`: the next chunk (at most
  `DEFAULT_VECTOR_SIZE` rows; empty chunks are skipped), the source is called
  again;
- `SourceResult::Finished`: the source is exhausted _for this thread_;
- `SourceResult::Blocked(interrupt)`: no data right now.

Unlike DuckDB, the chunk is returned by value in `SourceResult::Chunk`
instead of being written to an output argument: Arrow chunks are cheap to
move, and this makes it impossible to return data together with `Finished`.

Parallel sources (`parallel_source() == true`) are called concurrently by
several threads with their own local states and must hand out disjoint
morsels (typically through an atomic cursor in the global state).
`GlobalSourceState::max_threads` bounds the number of threads (e.g. a bridged
plan with 3 partitions runs on at most 3 threads).

### Operator role

`execute(ctx, input, output, gstate, lstate) -> Result<OperatorResult>`
processes a non-empty `input` chunk into `output`:

- `NeedMoreInput`: `input` is consumed; `output` (possibly empty) is pushed
  down.
- `HaveMoreOutput`: `output` is pushed down, then the operator is called again
  with the **same** `input` (e.g. a join that produces more than 2048 rows for
  one probe chunk keeps its cursor in the local state).
- `Finished`: the operator will produce no more output, on any thread.
  `output` (possibly non-empty) is still pushed down and the operators
  downstream are still flushed, but the finished operator and the operators
  upstream of it are not called again: all threads of all pipelines that
  contain the operator stop pulling from their sources. Threads blocked in
  their source are woken up, and pipelines that did not start yet skip their
  source. A thread that was already calling the operator concurrently may
  call it once more.
- `Blocked(interrupt)`: the operator is called again with the same input
  once the task is resumed; `output` is ignored.

Operators that buffer rows (e.g. `CachingOperator`) return
`requires_final_execute() == true`; after the source is exhausted the
executor calls `final_execute(ctx, output, gstate, lstate)` until it returns
`OperatorFinalizeResult::Finished`, pushing every non-empty output down the
pipeline. Operators are flushed in pipeline order, from the source side.

The global operator state is shared by all pipelines that contain the
operator (operators above a `UNION ALL` are part of every union pipeline), so
for example a streaming `LIMIT` counts the rows of all its inputs.

### Sink role

- `sink(ctx, input, gstate, lstate) -> Result<SinkResult>` consumes a
  non-empty chunk: `NeedMoreInput`, `Finished` or `Blocked` (called again with
  the same chunk once resumed). `Finished` means that no more input is needed
  from any thread (e.g. a `LIMIT` is reached): all threads of all pipelines
  that end in the sink stop, without flushing their operators. Threads blocked
  in their source are woken up, and pipelines that did not start yet skip
  their source. Every thread still calls `combine` and the sink is finalized
  once. A thread that was already pushing a chunk concurrently may still sink
  it, so the sink should keep returning `Finished`.
- `combine(ctx, gstate, lstate)` is called once per thread when the thread
  finished its part of the pipeline: merge the thread-local state into the
  global state.
- `finalize(ctx, gstate) -> Result<SinkFinalize>` is called once per
  execution, after all pipelines of the sink's meta pipeline completed and
  before any pipeline that reads from the sink starts. Return
  `SinkFinalize::NoOutputPossible` if the operator knows it cannot produce
  output (e.g. an inner hash join with an empty build side): all pipelines in
  which the operator is the source or an operator are then skipped. Their
  sinks still receive a `combine` call (with an empty local state) and are
  still finalized, so e.g. an ungrouped aggregate above the join still
  produces its single row.

The global sink state is created once per execution, before the first
pipeline that sinks into it starts, and stays alive until the execution
completes.

### Blocking: `TaskInterrupt`

Pipelines run as async tasks, but operator methods are synchronous. An
operator that has to wait returns a `Blocked(TaskInterrupt)` result:

- `TaskInterrupt::waker_registered()`: the operator registered
  `ctx.waker()` with what it waits for, e.g. by polling a stream with
  `Context::from_waker(ctx.waker())` (this is what `BridgeOperator` does). The
  task is resumed when the waker is woken. Returning this without registering
  the waker stalls the task forever.
- `TaskInterrupt::wait_for(future)`: the task is resumed once `future`
  completes. The executor drives `future` to completion before it retries the
  blocked call; it drops `future` early only if the call is not needed any
  more (the sink, or an operator at or after the blocked one, finished).
- `TaskInterrupt::yield_now()`: the task yields and is resumed immediately.

The executor re-invokes exactly the call that blocked, with the same input.
Operators must never block a thread (no `block_on`, no blocking locks held for
a long time, no synchronous I/O).

## Pipelines

A **pipeline** is a chain `source -> operators... -> sink`. A **meta
pipeline** is a set of pipelines that share a sink; the sink is finalized once
all of them completed. Meta pipelines form a tree: a pipeline breaker below a
pipeline gets its own child meta pipeline, and the pipelines that read from
the breaker depend on that child.

`PipelineGraph::try_new(sink)` builds the graph for the tree below `sink`
(during execution, a `CollectorSink`). It calls `build_pipelines` on the root
operator with the root meta pipeline's base pipeline. The default
implementation (`MetaPipelineBuilder::build_default`) is:

- a **sink** becomes the source of the current pipeline, and its single child
  is built into a new child meta pipeline whose sink it is
  (`build_sink`);
- a **leaf** becomes the source of the current pipeline (`build_source`);
- an operator with **one child** is added to the current pipeline, which
  continues into the child (`build_streaming_operator`).

Operators with several children or several roles override
`build_pipelines` using the `MetaPipelineBuilder` API. The operator whose
`build_pipelines` runs is available as `meta.current_operator()`.

| `MetaPipelineBuilder` method                            | Purpose                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `build(current, child)`                                 | recurse: `child.build_pipelines(current, meta)`                                            |
| `set_source(current, op)` / `add_operator(current, op)` | add an operator (operators are added from the sink side to the source side)                |
| `create_child_meta_pipeline(current, sink)`             | new child meta pipeline with `sink`; `current` depends on it                               |
| `create_union_pipeline(current, order_matters)`         | new pipeline with the same sink and the operators of `current` so far                      |
| `create_child_pipeline(current, source, last)`          | new pipeline that reads from `source` after `current` (and pipelines created after `last`) |
| `add_dependency(pipeline, dependency)`                  | extra ordering between pipelines of the same meta pipeline                                 |
| `pipelines()`                                           | the pipelines of the meta pipeline created so far                                          |

A hash join builds its pipelines like DuckDB's `PhysicalJoin::BuildJoinPipelines`:

```rust
fn build_pipelines(
    &self,
    current: &mut PipelineBuilder,
    meta: &mut MetaPipelineBuilder<'_>,
) -> Result<()> {
    let this = meta.current_operator()?;
    // The join is an operator of the probe pipeline.
    meta.add_operator(current, Arc::clone(&this))?;
    let last_pipeline = *meta.pipelines().last().unwrap();
    // The build side sinks into the join, in a child meta pipeline.
    let (mut build, mut build_pipeline) =
        meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
    build.build(&mut build_pipeline, &self.children[1])?;
    drop(build);
    // The probe side continues the current pipeline.
    meta.build(current, &self.children[0])?;
    // RIGHT / FULL joins scan the unmatched build rows after probing.
    if self.emits_unmatched_build_rows() {
        meta.create_child_pipeline(current, this, &last_pipeline)?;
    }
    Ok(())
}
```

and a union like DuckDB's `PhysicalUnion::BuildPipelines`:

```rust
let mut union_pipeline = meta.create_union_pipeline(current, false)?;
meta.build(current, &self.children[0])?;
meta.build(&mut union_pipeline, &self.children[1])
```

For example, `SELECT k, COUNT(*) FROM probe JOIN build USING (k) GROUP BY k`
decomposes into

```text
MetaPipeline 0 (sink: CollectorSink) children: [1]
  Pipeline 0: HashAggregate -> CollectorSink depends on meta pipelines [1]
MetaPipeline 1 (sink: HashAggregate) children: [2]
  Pipeline 1: Scan(probe) -> HashJoin -> HashAggregate depends on meta pipelines [2]
MetaPipeline 2 (sink: HashJoin)
  Pipeline 2: Scan(build) -> HashJoin
```

`explain::format_pipelines` renders the pipelines of an operator tree in
this format.

## Execution

### Scheduling: `Executor`

The executor turns the graph into a DAG of events: one _run_ event per
pipeline and one _finalize_ event per meta pipeline.

- A pipeline runs once the pipelines it depends on (same meta pipeline) are
  complete and the child meta pipelines it depends on are finalized.
- A meta pipeline is finalized once all of its pipelines are complete.

Events whose dependencies are satisfied run concurrently. A pipeline runs on
`n` worker tasks (`datafusion_common_runtime::JoinSet`), where `n` is the
`num_threads` of the executor (`target_partitions` for `PipelineRootExec`),
limited by `GlobalSourceState::max_threads`, and `1` if the source, the sink
or any operator of the pipeline is not parallel. Each worker creates its local
states, runs a `PipelineExecutor`, and finally calls `combine`.

An operator that is not parallel in a role (`parallel_source`,
`parallel_operator` or `parallel_sink` returns `false`) is never called
concurrently, even if it is part of several pipelines (the pipelines of the
inputs of a union share the sink and the operators above the union): the
executor runs such pipelines one after the other, in an order compatible with
their dependencies.

Errors of any worker cancel the execution and are returned by the output
stream; panics are propagated. Dropping the output stream aborts all tasks.

### The push loop: `PipelineExecutor`

`PipelineExecutor` is DuckDB's push loop, written as a poll function. Stage
`0` is the source and stage `k` is `operators[k - 1]`; `chunks[k]` holds the
output of stage `k`, and `chunks[n]` is pushed into the sink.

1. If the sink did not accept the last chunk (blocked), push it again.
2. Else, if an operator returned `HaveMoreOutput` (its stage is on the
   `in_process_operators` stack), call the most downstream in-process operator again
   with the same input and push its output down.
3. Else, if the source is not exhausted, get a chunk from the source and
   push it through the operators: an operator with an empty output sends the
   loop back to the most downstream in-process operator, or to the source.
4. Else, flush the operators that require `final_execute`, in order.
5. Else, the pipeline is complete for this thread: `combine`.

Every poll makes at most 128 operator calls (`get_data`, `execute`,
`final_execute` and `sink`), including the repeated calls of an operator that
returns `HaveMoreOutput` while the operators downstream discard its output.
When the budget is exhausted, the position in the pipeline is kept on the
`in_process_operators` stack (or as a pending sink chunk) and the task yields
like tokio's cooperative scheduling (`tokio::task::yield_now`), so that a
pipeline never starves the other tasks of the runtime and is cancelled
promptly.

### Results: `CollectorSink` and `PipelineStream`

The root pipeline sinks into a `CollectorSink`, which compacts every chunk
(applying any selection vector) into a `RecordBatch` and hands it to the
`PipelineStream` through a bounded queue. If the queue is full, the sink
returns `Blocked` (backpressure); if the stream was dropped, it returns
`Finished`. The execution starts when the stream is first polled.

### Metrics

The executor records, for every operator, the produced rows and chunks
(source and operator roles), the consumed rows and chunks (sink role) and the
time spent in its methods (`OperatorMetrics`). `PipelineStream::metrics` and
`PipelineRootExec::execution_metrics` expose them per execution;
`PipelineRootExec::metrics` reports `output_rows` and the total
`elapsed_compute` for `EXPLAIN ANALYZE`.

## DataFusion integration

### `BridgeOperator`

`BridgeOperator` runs an arbitrary `ExecutionPlan` as a parallel source:
threads claim the plan's output partitions through an atomic counter, execute
them, and poll the partition streams with the task's waker. A thread
multiplexes the partitions it claimed: it polls their streams round robin,
keeps the pending ones, and claims another partition (up to its fair share,
`ceil(partitions / threads)`) when all of them are pending; for unbounded
plans it claims its fair share right away. It returns
`Blocked(waker_registered)` only when all of its streams are pending. So, as
in DataFusion, an idle or never-ending partition does not hold back the
other partitions, even when there are more partitions than threads. Batches
are sliced into chunks of at most `DEFAULT_VECTOR_SIZE` rows without copying.
The output of a bridged plan is byte-identical to DataFusion's, because it
_is_ DataFusion's output.

### `PipelineRootExec`

`PipelineRootExec` is the `ExecutionPlan` returned by the planner. It has a
single output partition and the schema of the original plan (the root
operator must have exactly that schema). If the original plan has a single
partition, its equivalence properties (including the output ordering) are
kept: a single-partition input is executed by a single thread, which
preserves order. It is a leaf from DataFusion's point of view. `DisplayAs`
(in `explain.rs`) shows the lowered operator tree.

Because the bridged plans are hidden from DataFusion's plan traversals,
`PipelineRootExec::reset_state` resets the whole original plan
(`reset_plan_states`) and lowers it again, so that
`reset_plan_states(pipeline_root)` makes the plan executable again, as for any
DataFusion plan.

### `PipelineQueryPlanner`

```rust
let planner = PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_query_planner(Arc::new(planner))
    .build();
let ctx = SessionContext::new_with_state(state);
```

The planner wraps a DataFusion `PhysicalPlanner` (this crate does not depend
on the `datafusion` core crate), creates the optimized `ExecutionPlan` with
it, lowers it with `lower_execution_plan` and returns a `PipelineRootExec`.

`lower_execution_plan` is the extension point for native operators: it
recognizes `ExecutionPlan` nodes (and node patterns, e.g. two-stage
aggregations or `SortPreservingMergeExec -> SortExec`) and replaces them with
native operators whose children are lowered recursively. Any node it does not
recognize is wrapped, with its whole subtree, in a `BridgeOperator`, so every
query can be executed. Currently no node is lowered natively: every plan runs
through a single `BridgeOperator`.

## Differential testing

`test_utils.rs` provides the harness that proves byte compatibility:

```rust
#[tokio::test]
async fn filter_parity() {
    crate::test_utils::assert_byte_eq_with_datafusion(
        "SELECT a, b FROM t1 WHERE a > 3",
    )
    .await;
}
```

- `run_sql_with_both_engines(sql)` runs a query on a default
  `SessionContext` and on a `SessionContext` with `PipelineQueryPlanner`
  (otherwise identically configured, `target_partitions = 4`) and returns both
  results.
- `assert_byte_eq_with_datafusion(sql)` asserts that the results are equal:
  1. the plan schemas and all batch schemas are exactly equal (fields, data
     types, nullability, metadata);
  2. the Arrow IPC serializations of the canonical results are byte-for-byte
     equal.
- `*_with_setup(sql, async |ctx| { ... })` variants register custom tables
  in both sessions; `assert_byte_eq_with_datafusion_with_config` also takes a
  `SessionConfig`.

The canonical form of a result concatenates all batches, sorts the rows if
the query has no top-level `ORDER BY` (`has_top_level_order_by`), and rebuilds
the arrays through the Arrow row format. This removes differences that are not
part of the result (batch boundaries, row order of unordered results, bytes
under null slots, buffer layout of view arrays, dictionary encoding) while
comparing every value bit for bit (`-0.0` differs from `0.0`).

Test queries must be deterministic: no `LIMIT` without a total `ORDER BY`, no
ties in `ORDER BY`, no aggregates whose result depends on the input order
(floating point sums, `ARRAY_AGG` without `ORDER BY`, ...).

The default tables, registered by `register_test_tables`, are:

- `t1`: 5000 rows in 4 skewed partitions (one of them empty), columns `id`
  (unique `BIGINT NOT NULL`), `a INT`, `b BIGINT`, `c DOUBLE` (with `NaN`,
  `-0.0`), `d VARCHAR`, `e BOOLEAN`, `f DATE`, `g DECIMAL(12, 2)`, `h`
  (`Utf8View`), `ts TIMESTAMP`, all with nulls;
- `t2`: 150 rows in 3 partitions, `k INT` (joins with `t1.a`, duplicates and
  nulls), `name VARCHAR`, `w BIGINT`;
- `t_empty`: the schema of `t1` without rows.

`test_utils` uses the `datafusion` dev-dependency and is therefore only
compiled for tests. Integration tests can include it as a module, because it
only uses the public API of the crate:

```rust
#[path = "../src/test_utils.rs"]
mod test_utils;
```

## Conventions

- Use `datafusion_common::instant::Instant`, `SpawnedTask` / `JoinSet` from
  `datafusion-common-runtime` and `Vec::try_reserve` (see `clippy.toml`).
- Operators never block a thread; wait with `TaskInterrupt`.
- Keep chunks at most `DEFAULT_VECTOR_SIZE` rows.
- Never change the schema of the data: reuse the `SchemaRef`s and
  `PhysicalExpr`s of the DataFusion plan being lowered.
- Prove every native operator with `assert_byte_eq_with_datafusion` tests,
  including empty inputs, nulls, all relevant data types and skewed
  partitions.
