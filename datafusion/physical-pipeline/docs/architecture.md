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
     |    1. DefaultPhysicalPlanner             -> optimized ExecutionPlan
     |    2. lower_execution_plan_with_runtime  -> PhysicalOperator tree
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
| `planner.rs`                                                                       | `PipelineQueryPlanner`, `lower_execution_plan*` (the lowering of `ExecutionPlan`s)        |
| `explain.rs`                                                                       | `DisplayAs` for `PipelineRootExec`, pipeline / metrics rendering                          |
| `test_utils.rs`                                                                    | differential byte-equivalence test harness (test only)                                    |
| `tuple_data/`, `sources/`, `operators/`, `join/`, `aggregate/`, `sort/`, `window/` | native operators (Waves 2-5), see [Native operators](#native-operators)                   |

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
| `build_union(current, inputs, order_matters)`           | build the pipelines of a union of any number of inputs                                     |
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

and a union with `build_union`, which implements DuckDB's
`PhysicalUnion::BuildPipelines` for any number of inputs:

```rust
meta.build_union(current, &self.children, order_matters)
```

The first input continues `current`; every other input gets a union pipeline
that shares the operators above the union and the sink. If the order matters,
the pipelines of every input only start once all pipelines of the inputs
before it completed, including the pipelines created for nested unions, so
that `Union(Union(A, B), C)` emits `A`, `B` and then `C`. (A union with two
inputs can also use `create_union_pipeline` directly, creating the union
pipeline before building the first input into `current` and building the
second input into the union pipeline afterwards.)

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

Work that is no longer needed is cancelled. When a sink or an operator
finishes, or a sink is finalized with `NoOutputPossible`, the pipelines that
did not start yet and contain it skip their work. A child meta pipeline whose
output only these pipelines would read is then cancelled too, recursively:
its sink is marked as finished, so that its threads stop (also when blocked in
an idle source), still combine, and its sink is still finalized once. Child
meta pipelines that some other pipeline still needs keep running.

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
it, lowers it with `lower_execution_plan_with_runtime` (with the options and
the runtime of its session) and returns a `PipelineRootExec`. `EXPLAIN` shows
the lowered operator tree and its pipelines, `EXPLAIN ANALYZE` the metrics of
every operator (see `explain.rs`).

The lowering is the extension point for native operators: it recognizes
`ExecutionPlan` nodes (and node patterns, e.g. two-stage aggregations or
`SortPreservingMergeExec -> SortExec`) and replaces them with native
operators whose children are lowered recursively:

| DataFusion plan                                                                                      | Native operator                                                             |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `SortPreservingMergeExec -> SortExec`, `SortExec`                                                    | `PhysicalSort`, or `PhysicalTopK` with a fetch                              |
| `AggregateExec(Final \| FinalPartitioned) -> ... -> AggregateExec(Partial)`, `AggregateExec(Single)` | `PhysicalHashAggregate`                                                     |
| `HashJoinExec` (`CollectLeft` or `Partitioned`)                                                      | `PhysicalHashJoin`                                                          |
| `WindowAggExec`, `BoundedWindowAggExec`, with the sorts and repartitionings below them               | `PhysicalWindow`                                                            |
| `SortPreservingMergeExec` of the sorted partitions of a window                                       | `PhysicalSort` of the `PhysicalWindow`                                      |
| `NestedLoopJoinExec`, `CrossJoinExec`                                                                | `PhysicalNestedLoopJoin`, `PhysicalCrossJoin`                               |
| `GlobalLimitExec`, `LocalLimitExec`, `CoalescePartitionsExec` with a fetch                           | `PhysicalLimit`                                                             |
| `UnionExec`, `InterleaveExec`                                                                        | `PhysicalUnion`                                                             |
| `PlaceholderRowExec`                                                                                 | `PhysicalPlaceholderRow`                                                    |
| `FilterExec`                                                                                         | `PhysicalFilter` in a `CachingOperator`                                     |
| `ProjectionExec`                                                                                     | `PhysicalProjection`                                                        |
| `DataSourceExec` (memory, Parquet, CSV), `EmptyExec`                                                 | `MemoryScanSource`, `ParquetScanSource`, `CsvScanSource`, `EmptyScanSource` |
| `RepartitionExec`, `CoalescePartitionsExec`, `CoalesceBatchesExec`                                   | removed                                                                     |

The nodes that only distribute rows over partitions are removed: the threads
of a pipeline read the chunks of its source directly, and the pipeline
breakers (the build sides of joins, aggregations, sorts and windows) are
shared by all threads, which sink their rows into them (and the aggregations
and windows partition them by their keys themselves), so no rows are shuffled
through channels.

Any node it does not recognize (e.g. a sort-merge join, a scalar subquery, a
recursive query or a custom node) is wrapped, with its whole subtree, in a
`BridgeOperator`, so every query can be executed. So are the nodes whose
results or errors a native operator would not reproduce exactly. Every
native operator documents which nodes it reproduces in its `try_lower` (or
`supports_*`) function, and the module documentation of `planner.rs` lists
the rules of the planner itself. In short, DataFusion executes:

- the plans below a limit that stops the execution early (but for sorts and
  windows, which read all of their input in both engines);
- operators with an expression whose errors depend on the batches it is
  evaluated on, or with more than one part that may fail (which error comes
  first depends on the batches), and, on several threads, operators with a
  part that may fail (the threads evaluate the chunks of a partition in
  parallel, DataFusion its rows in order): e.g. the decimal arithmetic of
  TPC-H, which may overflow;
- nodes that coalesce or concatenate batches of types whose concatenation may
  fail (dictionaries, run-end encodings, strings and lists with 32-bit
  offsets), and, below a node of the same pipeline that may fail, nodes that
  pull ahead of their consumers (`FilterExec`, coalescing repartitions, hash
  and nested loop joins, which coalesce their output);
- joins and TopKs whose dynamic filters prune the reads of inputs that may
  fail (e.g. Parquet scans), and windows above TopKs that may end some of
  their partitions early.

Windows, nested loop joins and cross joins keep all of their rows in memory
without reserving it in the memory pool, while the DataFusion nodes that they
replace reserve their memory (and fail, or spill, once the pool is
exhausted): they are only lowered for sessions whose memory pool has no
limit. `lower_execution_plan` (for any session) and
`PipelineRootExec::reset_state` lower as if the pool had a limit, and as if
the session had the options for which the fewest nodes can be lowered.

## Native operators

| Module                               | Operators                                                                                                                                         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sources/`                           | `MemoryScanSource`, `ParquetScanSource` and `CsvScanSource` (morsel-driven), `EmptyScanSource`                                                    |
| `operators/`                         | `PhysicalFilter` (with the `AdaptiveFilter`), `PhysicalProjection`, `CachingOperator`, `PhysicalLimit`, `PhysicalUnion`, `PhysicalPlaceholderRow` |
| `tuple_data/`                        | `TupleDataCollection` (row-major storage), its radix partitioning and the `RowMatcher`, used by joins and aggregations                            |
| `join/hash_join.rs`, `hash_table.rs` | `PhysicalHashJoin` over a shared `JoinHashTable`                                                                                                  |
| `join/nested_loop.rs`                | `PhysicalNestedLoopJoin`, `PhysicalCrossJoin`                                                                                                     |
| `aggregate/`                         | `PhysicalHashAggregate` (thread-local tables, radix-partitioned by `RadixHashAggregate`)                                                          |
| `sort/`                              | `PhysicalSort`, `PhysicalTopK`                                                                                                                    |
| `window/`                            | `PhysicalWindow`                                                                                                                                  |

The joins, aggregations, sorts and windows are pipeline breakers: their
inputs sink into them on all threads, they finalize (in parallel where it
pays), and they are the sources of the pipelines that read their output.
Joins are also operators of their probe pipelines.

## Native coverage of TPC-H

`tests/tpch_parity.rs` runs the 22 TPC-H queries and a stress corpus of 75
analytical queries (correlated subqueries, `EXISTS`, `IN`, multi-stage
aggregations, grouping sets, windows, skewed hash joins, nested loop and cross
joins, TopKs, set operations) over `MemTable`s and Parquet files on both
engines, and asserts their byte parity. `tpch_queries_are_lowered_as_listed`
asserts the exact lowering of every TPC-H query (`TPCH_COVERAGE`): the
numbers of native and of all operators, and the DataFusion plans that run in
`BridgeOperator`s, each with why the native operators leave it to DataFusion.
On 4 threads, with the sessions of the suite (which do not trust DataFusion's
built-in functions, see `TrustedFunctions`):

- Over `MemTable`s, 16 of the 22 queries run fully natively: Q1 to Q6, Q10,
  Q12 to Q16 and Q18 to Q21. Their decimal arithmetic, which can only fail
  with arithmetic overflows, is evaluated natively on any number of threads,
  and so are the scalar subqueries of Q15 (`ScalarSubqueryExec`).
- The native projection refuses the projections of Q7 and Q9 (`date_part`,
  which the classification of errors only knows in sessions that trust
  DataFusion's built-in functions), of Q8 (a decimal division by a sum that
  may be 0, which may fail with several kinds of errors) and of Q22
  (`substr`, likewise untrusted, whose repartitioning over `MemTable`s, and
  filter over Parquet, stay in DataFusion too).
- The filter and the projection of Q11 cast decimals, which may fail with
  other errors than overflows (whose messages name the values that fail), so
  they stay in DataFusion on several threads; the semi join of Q17 has a
  residual filter with decimal arithmetic, which DataFusion evaluates on whole
  batches of candidate pairs.
- Over Parquet, 13 queries run fully natively: the hash joins of Q15 and Q20
  have dynamic filters that prune the reads of the Parquet scans below them
  (whose decimal arithmetic may fail on the rows that DataFusion skips), and
  the TopK of Q18 a dynamic filter that prunes the scan of `orders`: they stay
  in DataFusion too.

All window queries of the stress corpus run on native windows but the two
whose windows are stacked on windows of other partitions or orders, which
DataFusion sorts or merges in between (`PhysicalWindow::try_lower` does not
look through them), and all of its nested loop and cross joins run natively
(`windows_and_loop_joins_of_the_stress_corpus_are_native`).

## Benchmarks

`benches/pipeline_vs_default.rs` measures the wall-clock time of the 22
TPC-H queries (over the synthetic data of the parity suite) and of 25
ClickBench-style queries (over a synthetic `hits` table) on default DataFusion
and on the pipeline engine, both with `target_partitions` set to the number of
threads, over `MemTable`s and over Parquet files, and checks that both engines
return the same rows (every row of the tables below did):

```text
cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench pipeline_vs_default -- pipeline_vs_default --nocapture
PIPELINE_BENCH_SCALE=1 PIPELINE_BENCH_HITS=10000000 cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench pipeline_vs_default -- pipeline_vs_default --nocapture
```

Both sessions trust DataFusion's built-in functions (`TrustedFunctions`) and
its built-in aggregate functions
(`PhysicalHashAggregate::trusted_aggregate_functions`), and the strings are
`Utf8View` (as DataFusion reads them from Parquet). With
`PIPELINE_BENCH_TRUSTED_AGGREGATES=false`, the sessions do not trust the
aggregate functions: the rows of the native aggregations with aggregates are
then all aggregated in the thread-local tables, as before the aggregate
functions could be trusted (the _Before_ columns below), while with trusted
aggregate functions the rows of the threads whose groups hardly repeat bypass
their tables and are aggregated once, by the final aggregation of their radix
partitions (the _After_ columns, see `rows_can_bypass_tables`).

The measurements below were taken on a 16-core Neoverse-N1 machine with 16
threads, with the same build for both runs: the median of 5 executions, in ms
(without planning), and the native operators / all operators of the lowered
plan (a plan of one operator runs entirely in DataFusion, in a
`BridgeOperator`; the queries whose plans are partly native leave the
operators of [Native coverage of TPC-H](#native-coverage-of-tpc-h) to
DataFusion). _DataFusion_ and _Speedup_ (DataFusion / After) are those of the
run with trusted aggregate functions. The machine was shared, so times differ
by up to 10-15% from run to run (e.g. the TPC-H queries, which hardly have
high-cardinality aggregations, in the two runs).

### TPC-H SF 0.1, 1,000,000 `hits` rows

| Query | Rows | MemTable: DataFusion | Before | After | Speedup | Native | Parquet: DataFusion | Before | After | Speedup | Native |
| ----- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q1 | 4 | 7.93 | 4.26 | 4.32 | 1.84x | 7/7 | 10.89 | 9.51 | 9.54 | 1.14x | 7/7 |
| q2 | 48 | 5.84 | 4.45 | 4.47 | 1.31x | 26/26 | 8.64 | 11.40 | 11.59 | 0.75x | 26/26 |
| q3 | 10 | 3.65 | 4.33 | 4.81 | 0.76x | 14/14 | 8.20 | 10.01 | 10.23 | 0.80x | 14/14 |
| q4 | 5 | 5.45 | 5.15 | 5.69 | 0.96x | 10/10 | 4.98 | 6.28 | 6.17 | 0.81x | 10/10 |
| q5 | 5 | 4.09 | 6.76 | 6.54 | 0.63x | 18/18 | 8.61 | 13.96 | 13.63 | 0.63x | 18/18 |
| q6 | 1 | 0.79 | 1.80 | 0.96 | 0.82x | 5/5 | 2.97 | 3.38 | 3.50 | 0.85x | 5/5 |
| q7 | 4 | 20.02 | 14.09 | 15.29 | 1.31x | 21/21 | 9.60 | 19.19 | 18.91 | 0.51x | 21/21 |
| q8 | 2 | 3.65 | 4.23 | 4.30 | 0.85x | 1/2 | 10.37 | 11.63 | 11.97 | 0.87x | 1/2 |
| q9 | 175 | 9.29 | 9.95 | 10.20 | 0.91x | 17/17 | 13.26 | 18.91 | 18.50 | 0.72x | 17/17 |
| q10 | 20 | 6.23 | 7.12 | 7.62 | 0.82x | 14/14 | 10.24 | 13.25 | 13.52 | 0.76x | 14/14 |
| q11 | 2443 | 3.24 | 3.28 | 3.23 | 1.00x | 4/6 | 6.71 | 6.93 | 7.01 | 0.96x | 4/6 |
| q12 | 2 | 4.20 | 4.36 | 4.24 | 0.99x | 8/8 | 5.42 | 5.98 | 5.84 | 0.93x | 8/8 |
| q13 | 39 | 6.28 | 6.00 | 6.07 | 1.03x | 10/10 | 7.66 | 10.92 | 10.67 | 0.72x | 10/10 |
| q14 | 1 | 2.04 | 2.12 | 2.21 | 0.92x | 8/8 | 4.77 | 5.20 | 5.08 | 0.94x | 8/8 |
| q15 | 1 | 4.13 | 3.20 | 3.23 | 1.28x | 18/18 | 8.53 | 8.25 | 8.30 | 1.03x | 9/10 |
| q16 | 2925 | 6.66 | 4.93 | 4.80 | 1.39x | 13/13 | 7.57 | 8.34 | 8.69 | 0.87x | 13/13 |
| q17 | 1 | 7.85 | 7.46 | 8.59 | 0.91x | 2/3 | 17.84 | 18.39 | 18.13 | 0.98x | 2/3 |
| q18 | 100 | 36.47 | 23.15 | 23.04 | 1.58x | 12/12 | 16.61 | 15.71 | 16.00 | 1.04x | 0/1 |
| q19 | 1 | 2.41 | 1.79 | 1.81 | 1.33x | 9/9 | 8.46 | 8.37 | 8.14 | 1.04x | 9/9 |
| q20 | 17 | 4.20 | 4.52 | 4.59 | 0.92x | 18/18 | 9.96 | 9.26 | 10.18 | 0.98x | 1/2 |
| q21 | 45 | 15.39 | 17.57 | 17.70 | 0.87x | 22/22 | 14.70 | 26.30 | 25.86 | 0.57x | 22/22 |
| q22 | 7 | 2.31 | 2.44 | 2.40 | 0.96x | 15/15 | 5.98 | 6.22 | 5.97 | 1.00x | 15/15 |
| hits0 | 1 | 0.06 | 0.09 | 0.08 | 0.75x | 2/2 | 0.07 | 0.11 | 0.09 | 0.78x | 2/2 |
| hits1 | 1 | 0.36 | 0.50 | 0.47 | 0.77x | 5/5 | 1.65 | 1.80 | 1.69 | 0.98x | 5/5 |
| hits2 | 1 | 0.47 | 0.63 | 0.58 | 0.81x | 3/3 | 1.86 | 1.60 | 1.88 | 0.99x | 3/3 |
| hits4 | 1 | 7.67 | 4.86 | 4.82 | 1.59x | 4/4 | 8.76 | 9.07 | 8.84 | 0.99x | 4/4 |
| hits5 | 1 | 6.69 | 4.51 | 4.77 | 1.40x | 4/4 | 8.99 | 8.63 | 8.92 | 1.01x | 4/4 |
| hits6 | 1 | 0.31 | 0.42 | 0.46 | 0.67x | 2/2 | 0.07 | 0.08 | 0.09 | 0.78x | 2/2 |
| hits7 | 30 | 1.67 | 0.85 | 0.90 | 1.86x | 6/6 | 2.78 | 2.04 | 2.14 | 1.30x | 6/6 |
| hits8 | 10 | 14.06 | 10.38 | 10.88 | 1.29x | 5/5 | 13.71 | 12.71 | 12.46 | 1.10x | 5/5 |
| hits10 | 8 | 4.85 | 3.48 | 3.60 | 1.35x | 7/7 | 7.06 | 6.50 | 6.48 | 1.09x | 7/7 |
| hits12 | 10 | 6.84 | 4.84 | 4.95 | 1.38x | 6/6 | 8.47 | 9.51 | 8.87 | 0.95x | 6/6 |
| hits14 | 10 | 8.34 | 7.02 | 7.81 | 1.07x | 6/6 | 10.55 | 10.34 | 11.71 | 0.90x | 6/6 |
| hits15 | 10 | 8.50 | 5.95 | 6.33 | 1.34x | 4/4 | 10.41 | 9.94 | 10.47 | 0.99x | 4/4 |
| hits16 | 10 | 16.13 | 13.92 | 14.40 | 1.12x | 4/4 | 16.48 | 18.38 | 18.19 | 0.91x | 4/4 |
| hits20 | 1 | 2.83 | 1.76 | 1.90 | 1.49x | 5/5 | 5.71 | 5.47 | 6.02 | 0.95x | 5/5 |
| hits21 | 10 | 5.05 | 4.33 | 3.85 | 1.31x | 6/6 | 9.79 | 9.45 | 9.94 | 0.98x | 6/6 |
| hits23 | 10 | 4.59 | 4.26 | 3.45 | 1.33x | 4/4 | 15.60 | 15.45 | 16.46 | 0.95x | 0/1 |
| hits25 | 10 | 1.76 | 1.38 | 1.18 | 1.49x | 4/4 | 3.44 | 3.03 | 3.08 | 1.12x | 4/4 |
| hits27 | 25 | 5.63 | 3.92 | 3.57 | 1.58x | 8/8 | 9.31 | 9.28 | 9.22 | 1.01x | 0/1 |
| hits29 | 1 | 0.44 | 0.59 | 0.59 | 0.75x | 4/4 | 1.32 | 1.24 | 1.20 | 1.10x | 3/3 |
| hits30 | 10 | 7.15 | 6.01 | 6.92 | 1.03x | 6/6 | 9.82 | 12.18 | 11.30 | 0.87x | 6/6 |
| hits32 | 10 | 16.11 | 14.35 | 13.12 | 1.23x | 4/4 | 16.81 | 17.64 | 16.36 | 1.03x | 4/4 |
| hits33 | 10 | 16.23 | 15.40 | 16.87 | 0.96x | 4/4 | 18.74 | 23.96 | 21.96 | 0.85x | 4/4 |
| hits35 | 10 | 10.28 | 8.05 | 8.15 | 1.26x | 5/5 | 10.48 | 11.45 | 10.41 | 1.01x | 5/5 |
| hits36 | 10 | 5.33 | 4.96 | 4.52 | 1.18x | 6/6 | 9.97 | 10.21 | 10.80 | 0.92x | 6/6 |
| hits42 | 10 | 2.35 | 2.00 | 2.15 | 1.09x | 3/4 | 5.54 | 5.02 | 5.03 | 1.10x | 3/4 |

| Source   | Trusted aggregates | DataFusion (ms) | Pipeline (ms) | Speedup | Geometric mean speedup | Fully native queries |
| -------- | ------------------ | --------------: | ------------: | ------: | ---------------------: | -------------------: |
| MemTable | no (before) | 299.14 | 267.39 | 1.12x | 1.04x | 43/47 |
| MemTable | yes (after) | 315.81 | 272.45 | 1.16x | 1.10x | 43/47 |
| Parquet | no (before) | 412.39 | 462.48 | 0.89x | 0.92x | 38/47 |
| Parquet | yes (after) | 409.38 | 461.06 | 0.89x | 0.91x | 38/47 |

### TPC-H SF 1, 10,000,000 `hits` rows

| Query | Rows | MemTable: DataFusion | Before | After | Speedup | Native | Parquet: DataFusion | Before | After | Speedup | Native |
| ----- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q1 | 4 | 32.24 | 32.66 | 32.80 | 0.98x | 7/7 | 63.93 | 63.92 | 66.83 | 0.96x | 7/7 |
| q2 | 100 | 13.71 | 12.82 | 13.56 | 1.01x | 26/26 | 17.78 | 28.89 | 29.30 | 0.61x | 26/26 |
| q3 | 10 | 19.83 | 19.79 | 20.34 | 0.97x | 14/14 | 34.24 | 43.24 | 47.15 | 0.73x | 14/14 |
| q4 | 5 | 24.96 | 34.42 | 35.64 | 0.70x | 10/10 | 19.01 | 26.29 | 26.18 | 0.73x | 10/10 |
| q5 | 5 | 38.36 | 36.31 | 38.24 | 1.00x | 18/18 | 46.54 | 67.03 | 69.25 | 0.67x | 18/18 |
| q6 | 1 | 3.91 | 5.68 | 5.74 | 0.68x | 5/5 | 15.26 | 20.14 | 19.88 | 0.77x | 5/5 |
| q7 | 4 | 150.17 | 84.81 | 86.71 | 1.73x | 21/21 | 55.38 | 100.49 | 109.60 | 0.51x | 21/21 |
| q8 | 2 | 14.68 | 14.08 | 14.29 | 1.03x | 1/2 | 40.17 | 36.61 | 39.29 | 1.02x | 1/2 |
| q9 | 175 | 86.96 | 57.69 | 65.33 | 1.33x | 17/17 | 60.60 | 100.77 | 106.72 | 0.57x | 17/17 |
| q10 | 20 | 28.04 | 25.93 | 26.13 | 1.07x | 14/14 | 43.31 | 54.99 | 57.35 | 0.76x | 14/14 |
| q11 | 746 | 6.11 | 6.12 | 6.13 | 1.00x | 4/6 | 12.86 | 12.04 | 12.54 | 1.03x | 4/6 |
| q12 | 2 | 14.37 | 18.71 | 18.82 | 0.76x | 8/8 | 23.14 | 25.16 | 27.93 | 0.83x | 8/8 |
| q13 | 41 | 25.96 | 36.62 | 37.00 | 0.70x | 10/10 | 42.88 | 47.58 | 62.03 | 0.69x | 10/10 |
| q14 | 1 | 5.53 | 5.25 | 5.32 | 1.04x | 8/8 | 19.58 | 20.36 | 23.17 | 0.85x | 8/8 |
| q15 | 1 | 10.38 | 9.37 | 9.91 | 1.05x | 9/10 | 36.36 | 33.37 | 37.85 | 0.96x | 9/10 |
| q16 | 18399 | 15.06 | 14.61 | 14.51 | 1.04x | 13/13 | 18.23 | 20.19 | 22.45 | 0.81x | 13/13 |
| q17 | 1 | 72.41 | 85.20 | 74.17 | 0.98x | 2/3 | 125.21 | 119.42 | 130.05 | 0.96x | 2/3 |
| q18 | 100 | 166.56 | 141.10 | 149.41 | 1.11x | 2/3 | 116.84 | 112.57 | 114.74 | 1.02x | 0/1 |
| q19 | 1 | 18.18 | 11.87 | 12.24 | 1.49x | 9/9 | 36.32 | 35.73 | 39.69 | 0.92x | 9/9 |
| q20 | 178 | 19.32 | 18.06 | 18.85 | 1.02x | 7/8 | 34.61 | 30.24 | 32.84 | 1.05x | 1/2 |
| q21 | 100 | 158.62 | 110.28 | 125.56 | 1.26x | 22/22 | 99.24 | 168.92 | 175.90 | 0.56x | 22/22 |
| q22 | 7 | 7.05 | 7.46 | 7.29 | 0.97x | 15/15 | 12.49 | 13.82 | 16.21 | 0.77x | 15/15 |
| hits0 | 1 | 0.10 | 0.12 | 0.13 | 0.77x | 2/2 | 0.07 | 0.09 | 0.09 | 0.78x | 2/2 |
| hits1 | 1 | 0.99 | 2.19 | 2.36 | 0.42x | 5/5 | 6.56 | 8.08 | 7.82 | 0.84x | 5/5 |
| hits2 | 1 | 1.57 | 2.47 | 2.50 | 0.63x | 3/3 | 7.29 | 8.99 | 8.49 | 0.86x | 3/3 |
| hits4 | 1 | 65.31 | 26.04 | 25.44 | 2.57x | 4/4 | 76.68 | 37.50 | 37.18 | 2.06x | 4/4 |
| hits5 | 1 | 64.13 | 61.75 | 52.17 | 1.23x | 4/4 | 83.20 | 75.79 | 80.75 | 1.03x | 4/4 |
| hits6 | 1 | 0.93 | 1.46 | 1.40 | 0.66x | 2/2 | 0.08 | 0.09 | 0.10 | 0.80x | 2/2 |
| hits7 | 30 | 3.63 | 3.39 | 3.12 | 1.16x | 6/6 | 8.44 | 8.75 | 9.13 | 0.92x | 6/6 |
| hits8 | 10 | 92.32 | 69.49 | 62.66 | 1.47x | 5/5 | 108.99 | 82.30 | 85.69 | 1.27x | 5/5 |
| hits10 | 8 | 22.56 | 21.02 | 19.67 | 1.15x | 7/7 | 35.69 | 34.51 | 37.43 | 0.95x | 7/7 |
| hits12 | 10 | 71.40 | 61.01 | 48.49 | 1.47x | 6/6 | 78.07 | 80.13 | 60.47 | 1.29x | 6/6 |
| hits14 | 10 | 69.67 | 77.88 | 62.55 | 1.11x | 6/6 | 79.75 | 91.39 | 79.28 | 1.01x | 6/6 |
| hits15 | 10 | 81.31 | 54.00 | 29.85 | 2.72x | 4/4 | 97.17 | 68.75 | 43.99 | 2.21x | 4/4 |
| hits16 | 10 | 123.31 | 155.60 | 99.16 | 1.24x | 4/4 | 154.06 | 186.40 | 136.24 | 1.13x | 4/4 |
| hits20 | 1 | 13.22 | 15.05 | 14.15 | 0.93x | 5/5 | 40.14 | 38.39 | 40.49 | 0.99x | 5/5 |
| hits21 | 10 | 22.75 | 25.47 | 24.66 | 0.92x | 6/6 | 56.76 | 56.12 | 61.78 | 0.92x | 6/6 |
| hits23 | 10 | 23.92 | 28.79 | 27.03 | 0.88x | 4/4 | 128.07 | 119.89 | 128.41 | 1.00x | 0/1 |
| hits25 | 10 | 6.08 | 7.21 | 6.77 | 0.90x | 4/4 | 18.37 | 17.16 | 17.75 | 1.03x | 4/4 |
| hits27 | 25 | 26.27 | 25.29 | 24.13 | 1.09x | 8/8 | 49.12 | 50.34 | 52.20 | 0.94x | 0/1 |
| hits29 | 1 | 1.76 | 3.14 | 2.47 | 0.71x | 4/4 | 4.05 | 4.62 | 4.55 | 0.89x | 3/3 |
| hits30 | 10 | 49.95 | 51.65 | 44.50 | 1.12x | 6/6 | 63.73 | 80.29 | 66.62 | 0.96x | 6/6 |
| hits32 | 10 | 148.26 | 155.61 | 89.97 | 1.65x | 4/4 | 165.68 | 159.41 | 108.23 | 1.53x | 4/4 |
| hits33 | 10 | 232.60 | 208.18 | 135.15 | 1.72x | 4/4 | 237.71 | 255.36 | 160.34 | 1.48x | 4/4 |
| hits35 | 10 | 69.70 | 74.57 | 53.44 | 1.30x | 5/5 | 77.46 | 91.86 | 61.30 | 1.26x | 5/5 |
| hits36 | 10 | 37.28 | 38.99 | 38.71 | 0.96x | 6/6 | 59.98 | 75.51 | 69.59 | 0.86x | 6/6 |
| hits42 | 10 | 7.62 | 7.76 | 8.01 | 0.95x | 3/4 | 21.59 | 22.25 | 21.93 | 0.98x | 3/4 |

| Source   | Trusted aggregates | DataFusion (ms) | Pipeline (ms) | Speedup | Geometric mean speedup | Fully native queries |
| -------- | ------------------ | --------------: | ------------: | ------: | ---------------------: | -------------------: |
| MemTable | no (before) | 2205.44 | 1966.98 | 1.12x | 1.01x | 40/47 |
| MemTable | yes (after) | 2169.05 | 1696.48 | 1.28x | 1.06x | 40/47 |
| Parquet | no (before) | 2503.64 | 2835.78 | 0.88x | 0.87x | 38/47 |
| Parquet | yes (after) | 2632.73 | 2646.77 | 0.99x | 0.93x | 38/47 |

Of note:

- The rows of the high-cardinality aggregations of `hits15`, `hits16`,
  `hits30`, `hits32`, `hits33`, `hits35` and `hits36` (`COUNT(*)`, `SUM` and
  `AVG` by user, search phrase, client IP, watch id or URL) bypass the
  thread-local tables with trusted aggregate functions. With 1,000,000 rows,
  the threads have few rows each (62,500 on 16 threads), and their tables
  rarely fill up, so the bypass hardly matters. With 10,000,000 rows, it makes
  `hits15` 1.8x (MemTable) and 1.6x (Parquet) faster than before, `hits16`,
  `hits32` and `hits33` between 1.4x and 1.7x, `hits35` 1.4x to 1.5x and
  `hits30` 1.2x, while `hits36` filters most of its rows out before its
  aggregation, which hardly changes. Over `MemTable`s, `hits15` is 2.7x faster than
  DataFusion, `hits32` and `hits33` 1.7x, and `hits16` (1.3x slower than
  DataFusion before) 1.2x faster; the totals of all queries are 14% (MemTable)
  and 7% (Parquet) lower.
- With DataFusion's built-in functions trusted, all TPC-H queries run fully
  natively over `MemTable`s at SF 0.1 but Q8 (a decimal division), Q11 (casts
  of decimals) and Q17 (the residual filter of its semi join). At SF 1, the
  join of Q15 and the semi joins of Q18 and Q20 are `Partitioned` hash joins
  that the native hash join leaves to DataFusion (see
  `PhysicalHashJoin::try_lower`). Over Parquet, the hash joins of Q15 and
  Q20 and the TopK of Q18 have dynamic filters that prune the Parquet scans
  below them (see [Native coverage of TPC-H](#native-coverage-of-tpc-h)).
- The ClickBench-style queries run fully natively but for the aggregation of
  `hits42` (by `to_timestamp_seconds`, whose errors the classification of
  errors does not reproduce on several threads), and over Parquet `hits23` (a
  TopK whose dynamic filter prunes the Parquet scan) and `hits27` (a TopK by a
  floating-point average, which the native TopK only lowers over inputs that
  execute again as they executed once, see `executes_again_as_if_for_the_first_time`). The queries of
  less than a millisecond or two (`hits0`, `hits1`, `hits2`, `hits6`,
  `hits29`) pay the fixed cost of setting up the pipelines.

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
