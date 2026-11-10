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
On 4 threads:

- Over `MemTable`s, Q2, Q4, Q12, Q13, Q16, Q18, Q20 and Q21 run fully
  natively.
- Q1, Q3, Q5, Q10 and Q19 aggregate decimal arithmetic with more than one part
  that may overflow, Q6 (an aggregation) and Q14 (a projection) decimal
  arithmetic with one part, on several threads: these operators stay in
  DataFusion, with their inputs.
- The native projection refuses the projections of Q7 and Q9 (`date_part`,
  which the classification of errors only knows in sessions that trust
  DataFusion's built-in functions; their decimal arithmetic would keep them in
  DataFusion on several threads anyway) and of Q8 (a decimal division, which
  may fail with several kinds of errors).
- Q11, Q15 and Q22 have scalar subqueries (`ScalarSubqueryExec`), and the
  semi join of Q17 a residual filter with decimal arithmetic.
- Over Parquet, the hash joins of Q2, Q4, Q12, Q13, Q16, Q17, Q20 and Q21 and
  the TopK of Q18 have dynamic filters that prune the reads of the Parquet
  scans below them (which may fail to read or decode the rows that DataFusion
  skips): they stay in DataFusion too, so no query runs fully natively.

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

Both sessions trust DataFusion's built-in functions, and the strings are
`Utf8View` (as DataFusion reads them from Parquet). The measurements below were
taken on a 16-core Neoverse-N1 machine with 16 threads: the median of 5
executions, in ms (without planning), and the native operators / all
operators of the lowered plan (a plan of one operator runs entirely in
DataFusion, in a `BridgeOperator`; the queries whose plans are partly native
leave the operators of [Native coverage of TPC-H](#native-coverage-of-tpc-h)
to DataFusion).

### TPC-H SF 0.1, 1,000,000 `hits` rows

| Query  | Rows | MemTable: DataFusion | Pipeline | Speedup | Native | Parquet: DataFusion | Pipeline | Speedup | Native |
| ------ | ---: | -------------------: | -------: | ------: | -----: | ------------------: | -------: | ------: | -----: |
| q1     |    4 |                 6.16 |     5.19 |   1.19x |    2/3 |               10.27 |    10.24 |   1.00x |    2/3 |
| q2     |   48 |                 5.24 |     8.73 |   0.60x |  26/26 |                8.13 |    10.29 |   0.79x |    4/6 |
| q3     |   10 |                 3.20 |     3.74 |   0.85x |    2/3 |                8.05 |     7.73 |   1.04x |    2/3 |
| q4     |    5 |                 4.70 |     5.45 |   0.86x |  10/10 |                4.79 |     4.43 |   1.08x |    3/4 |
| q5     |    5 |                 4.02 |     4.65 |   0.86x |    2/3 |                8.49 |     9.12 |   0.93x |    2/3 |
| q6     |    1 |                 0.66 |     0.72 |   0.92x |    1/2 |                2.90 |     3.15 |   0.92x |    1/2 |
| q7     |    4 |                22.17 |    15.96 |   1.39x |    3/4 |                9.02 |     9.02 |   1.00x |    3/4 |
| q8     |    2 |                 3.54 |     4.03 |   0.88x |    1/2 |               10.10 |    11.00 |   0.92x |    1/2 |
| q9     |  175 |                 7.46 |     7.37 |   1.01x |    3/4 |               12.12 |    12.06 |   1.01x |    3/4 |
| q10    |   20 |                 5.23 |     5.99 |   0.87x |    2/3 |                9.66 |    10.13 |   0.95x |    2/3 |
| q11    | 2443 |                 3.11 |     3.07 |   1.01x |    0/1 |                6.03 |     6.18 |   0.98x |    0/1 |
| q12    |    2 |                 4.07 |     4.19 |   0.97x |    8/8 |                5.92 |     5.74 |   1.03x |    3/4 |
| q13    |   39 |                 5.64 |     5.58 |   1.01x |  10/10 |                7.35 |     6.83 |   1.08x |    5/6 |
| q14    |    1 |                 1.65 |     1.76 |   0.93x |    2/3 |                4.04 |     4.83 |   0.84x |    2/3 |
| q15    |    1 |                 3.87 |     3.76 |   1.03x |    0/1 |                8.17 |     7.94 |   1.03x |    0/1 |
| q16    | 2925 |                 6.22 |     5.79 |   1.07x |  13/13 |                7.23 |     8.14 |   0.89x |    8/9 |
| q17    |    1 |                 5.10 |     5.25 |   0.97x |    2/3 |               17.21 |    16.94 |   1.02x |    2/3 |
| q18    |  100 |                33.15 |    20.99 |   1.58x |  12/12 |               15.59 |    15.06 |   1.04x |    0/1 |
| q19    |    1 |                 2.26 |     2.32 |   0.97x |    1/2 |                8.14 |     8.26 |   0.98x |    1/2 |
| q20    |   17 |                 3.71 |     6.45 |   0.58x |  18/18 |                9.53 |     9.61 |   0.99x |    1/2 |
| q21    |   45 |                 9.80 |    17.49 |   0.56x |  22/22 |               13.99 |    21.52 |   0.65x |    7/8 |
| q22    |    7 |                 2.23 |     2.31 |   0.97x |    0/1 |                5.56 |     5.52 |   1.01x |    0/1 |
| hits0  |    1 |                 0.07 |     0.15 |   0.51x |    2/2 |                0.07 |     0.17 |   0.43x |    2/2 |
| hits1  |    1 |                 0.36 |     0.51 |   0.70x |    5/5 |                1.63 |     1.67 |   0.98x |    5/5 |
| hits2  |    1 |                 0.46 |     0.64 |   0.72x |    3/3 |                1.70 |     1.66 |   1.02x |    3/3 |
| hits4  |    1 |                 5.26 |     4.58 |   1.15x |    4/4 |                7.97 |     7.76 |   1.03x |    4/4 |
| hits5  |    1 |                 5.06 |     4.49 |   1.13x |    4/4 |                7.13 |     7.64 |   0.93x |    4/4 |
| hits6  |    1 |                 0.31 |     0.47 |   0.65x |    2/2 |                0.08 |     0.16 |   0.48x |    2/2 |
| hits7  |   30 |                 1.71 |     0.82 |   2.07x |    6/6 |                2.72 |     2.02 |   1.35x |    6/6 |
| hits8  |   10 |                11.43 |     9.01 |   1.27x |    5/5 |               12.95 |    12.41 |   1.04x |    5/5 |
| hits10 |    8 |                 4.51 |     2.75 |   1.64x |    7/7 |                6.13 |     5.80 |   1.06x |    7/7 |
| hits12 |   10 |                 5.74 |     4.68 |   1.23x |    6/6 |                8.35 |     8.27 |   1.01x |    6/6 |
| hits14 |   10 |                 7.50 |     6.22 |   1.20x |    6/6 |                9.21 |    11.01 |   0.84x |    6/6 |
| hits15 |   10 |                 7.07 |     5.49 |   1.29x |    4/4 |                9.40 |     8.53 |   1.10x |    4/4 |
| hits16 |   10 |                13.96 |    12.92 |   1.08x |    4/4 |               16.34 |    17.36 |   0.94x |    4/4 |
| hits20 |    1 |                 1.68 |     1.79 |   0.94x |    5/5 |                6.01 |     6.00 |   1.00x |    5/5 |
| hits21 |   10 |                 4.27 |     3.59 |   1.19x |    6/6 |                8.75 |     9.23 |   0.95x |    6/6 |
| hits23 |   10 |                 3.74 |     3.26 |   1.15x |    4/4 |               15.33 |    16.66 |   0.92x |    0/1 |
| hits25 |   10 |                 1.52 |     1.16 |   1.31x |    4/4 |                3.31 |     3.47 |   0.95x |    0/1 |
| hits27 |   25 |                 5.26 |     4.22 |   1.25x |    0/1 |                8.93 |     9.15 |   0.98x |    0/1 |
| hits29 |    1 |                 0.42 |     0.61 |   0.69x |    4/4 |                1.28 |     1.28 |   1.00x |    3/3 |
| hits30 |   10 |                 6.36 |     5.89 |   1.08x |    6/6 |                9.67 |    11.62 |   0.83x |    6/6 |
| hits32 |   10 |                14.01 |    13.33 |   1.05x |    4/4 |               16.58 |    18.02 |   0.92x |    4/4 |
| hits33 |   10 |                14.85 |    14.97 |   0.99x |    4/4 |               18.53 |    19.52 |   0.95x |    4/4 |
| hits35 |   10 |                 9.75 |     8.02 |   1.22x |    5/5 |               11.89 |    11.29 |   1.05x |    5/5 |
| hits36 |   10 |                 4.64 |     4.13 |   1.12x |    6/6 |                9.36 |    10.19 |   0.92x |    6/6 |
| hits42 |   10 |                 2.03 |     1.90 |   1.07x |    3/4 |                4.93 |     4.93 |   1.00x |    3/4 |

| Source   | DataFusion (ms) | Pipeline (ms) | Speedup | Geometric mean speedup | Fully native queries |
| -------- | --------------: | ------------: | ------: | ---------------------: | -------------------: |
| MemTable |          275.13 |        256.40 |   1.07x |                  1.00x |                31/47 |
| Parquet  |          390.53 |        409.57 |   0.95x |                  0.94x |                21/47 |

### TPC-H SF 1, 10,000,000 `hits` rows

| Query  |  Rows | MemTable: DataFusion | Pipeline | Speedup | Native | Parquet: DataFusion | Pipeline | Speedup | Native |
| ------ | ----: | -------------------: | -------: | ------: | -----: | ------------------: | -------: | ------: | -----: |
| q1     |     4 |                32.65 |    31.18 |   1.05x |    2/3 |               56.02 |    55.85 |   1.00x |    2/3 |
| q2     |   100 |                12.00 |    18.57 |   0.65x |  26/26 |               14.24 |    17.96 |   0.79x |    4/6 |
| q3     |    10 |                19.26 |    18.73 |   1.03x |    2/3 |               31.01 |    31.32 |   0.99x |    2/3 |
| q4     |     5 |                21.42 |    32.76 |   0.65x |  10/10 |               17.61 |    18.01 |   0.98x |    3/4 |
| q5     |     5 |                34.69 |    35.18 |   0.99x |    2/3 |               44.29 |    43.68 |   1.01x |    2/3 |
| q6     |     1 |                 2.96 |     3.07 |   0.97x |    1/2 |               14.07 |    14.72 |   0.96x |    1/2 |
| q7     |     4 |               146.19 |   149.45 |   0.98x |    3/4 |               50.74 |    48.41 |   1.05x |    3/4 |
| q8     |     2 |                13.11 |    13.85 |   0.95x |    1/2 |               34.45 |    36.50 |   0.94x |    1/2 |
| q9     |   175 |                81.96 |    85.25 |   0.96x |    3/4 |               53.41 |    53.20 |   1.00x |    3/4 |
| q10    |    20 |                26.21 |    27.46 |   0.95x |    2/3 |               39.66 |    39.37 |   1.01x |    2/3 |
| q11    |   746 |                 5.83 |     5.86 |   0.99x |    0/1 |               11.26 |    10.83 |   1.04x |    0/1 |
| q12    |     2 |                12.85 |    19.19 |   0.67x |    8/8 |               21.54 |    20.59 |   1.05x |    3/4 |
| q13    |    41 |                24.06 |    38.70 |   0.62x |  10/10 |               33.90 |    31.32 |   1.08x |    5/6 |
| q14    |     1 |                 4.86 |     4.87 |   1.00x |    2/3 |               17.17 |    17.40 |   0.99x |    2/3 |
| q15    |     1 |                10.05 |     9.97 |   1.01x |    0/1 |               29.73 |    30.99 |   0.96x |    0/1 |
| q16    | 18399 |                16.36 |    17.82 |   0.92x |  13/13 |               16.83 |    19.07 |   0.88x |    8/9 |
| q17    |     1 |                64.87 |    64.51 |   1.01x |    2/3 |              107.97 |   108.49 |   1.00x |    2/3 |
| q18    |   100 |               121.15 |   126.20 |   0.96x |    2/3 |              106.33 |   107.27 |   0.99x |    0/1 |
| q19    |     1 |                10.31 |    10.33 |   1.00x |    1/2 |               34.87 |    35.04 |   1.00x |    1/2 |
| q20    |   178 |                16.45 |    18.97 |   0.87x |    7/8 |               29.98 |    32.13 |   0.93x |    1/2 |
| q21    |   100 |               140.54 |   126.21 |   1.11x |  22/22 |               88.97 |   144.87 |   0.61x |    7/8 |
| q22    |     7 |                 7.12 |     7.01 |   1.02x |    0/1 |               11.65 |    11.67 |   1.00x |    0/1 |
| hits0  |     1 |                 0.09 |     0.16 |   0.53x |    2/2 |                0.07 |     0.17 |   0.45x |    2/2 |
| hits1  |     1 |                 0.90 |     2.19 |   0.41x |    5/5 |                5.09 |     7.53 |   0.68x |    5/5 |
| hits2  |     1 |                 1.51 |     2.51 |   0.60x |    3/3 |                7.60 |     8.35 |   0.91x |    3/3 |
| hits4  |     1 |                53.33 |    35.40 |   1.51x |    4/4 |               64.22 |    53.89 |   1.19x |    4/4 |
| hits5  |     1 |                61.05 |    49.42 |   1.24x |    4/4 |               72.71 |    73.62 |   0.99x |    4/4 |
| hits6  |     1 |                 0.97 |     1.43 |   0.68x |    2/2 |                0.07 |     0.17 |   0.45x |    2/2 |
| hits7  |    30 |                 3.61 |     3.40 |   1.06x |    6/6 |                7.23 |     8.73 |   0.83x |    6/6 |
| hits8  |    10 |                83.27 |    83.95 |   0.99x |    5/5 |               98.13 |   111.72 |   0.88x |    5/5 |
| hits10 |     8 |                21.15 |    19.73 |   1.07x |    7/7 |               33.04 |    39.72 |   0.83x |    7/7 |
| hits12 |    10 |                58.70 |    50.62 |   1.16x |    6/6 |               70.48 |    74.74 |   0.94x |    6/6 |
| hits14 |    10 |                57.77 |    61.42 |   0.94x |    6/6 |               73.17 |    87.60 |   0.84x |    6/6 |
| hits15 |    10 |                71.97 |    49.58 |   1.45x |    4/4 |               79.74 |    66.44 |   1.20x |    4/4 |
| hits16 |    10 |               120.02 |   154.22 |   0.78x |    4/4 |              139.82 |   170.32 |   0.82x |    4/4 |
| hits20 |     1 |                12.90 |    14.28 |   0.90x |    5/5 |               36.07 |    36.88 |   0.98x |    5/5 |
| hits21 |    10 |                21.89 |    22.84 |   0.96x |    6/6 |               53.10 |    55.64 |   0.95x |    6/6 |
| hits23 |    10 |                21.31 |    26.59 |   0.80x |    4/4 |              113.08 |   111.91 |   1.01x |    0/1 |
| hits25 |    10 |                 5.86 |     6.68 |   0.88x |    4/4 |               16.34 |    17.06 |   0.96x |    0/1 |
| hits27 |    25 |                25.27 |    25.01 |   1.01x |    0/1 |               47.44 |    47.70 |   0.99x |    0/1 |
| hits29 |     1 |                 1.09 |     2.50 |   0.44x |    4/4 |                4.09 |     4.50 |   0.91x |    3/3 |
| hits30 |    10 |                40.04 |    47.76 |   0.84x |    6/6 |               62.97 |    77.65 |   0.81x |    6/6 |
| hits32 |    10 |               134.48 |   117.80 |   1.14x |    4/4 |              153.41 |   148.33 |   1.03x |    4/4 |
| hits33 |    10 |               199.89 |   182.34 |   1.10x |    4/4 |              223.02 |   221.15 |   1.01x |    4/4 |
| hits35 |    10 |                66.28 |    63.52 |   1.04x |    5/5 |               70.72 |    77.08 |   0.92x |    5/5 |
| hits36 |    10 |                29.74 |    33.78 |   0.88x |    6/6 |               58.43 |    70.21 |   0.83x |    6/6 |
| hits42 |    10 |                 6.86 |     5.44 |   1.26x |    3/4 |               21.12 |    21.18 |   1.00x |    3/4 |

| Source   | DataFusion (ms) | Pipeline (ms) | Speedup | Geometric mean speedup | Fully native queries |
| -------- | --------------: | ------------: | ------: | ---------------------: | -------------------: |
| MemTable |         1924.86 |       1927.72 |   1.00x |                  0.91x |                29/47 |
| Parquet  |         2376.86 |       2520.93 |   0.94x |                  0.91x |                21/47 |

Of note:

- The TPC-H plans that run fully natively over `MemTable`s at SF 0.1 are the
  join-heavy ones (Q2, Q4, Q12, Q13, Q16, Q18, Q20, Q21): Q18 is 1.6x faster
  than DataFusion, Q2, Q20 and Q21 are about 1.7x slower. At SF 1, Q21 is 1.1x
  faster, Q4, Q12 and Q13 are about 1.5x slower, and the semi joins of Q18 and
  Q20 are `Partitioned` hash joins that the native hash join leaves to
  DataFusion (see `PhysicalHashJoin::try_lower`).
- The plans of a single operator (0/1) and the operators that stay in
  DataFusion run at DataFusion's speed: most of TPC-H over Parquet, where the
  hash joins with dynamic filters, and the decimal arithmetic, stay in
  DataFusion.
- The ClickBench-style queries run fully natively but for `hits27` (a TopK by
  a floating-point average, which the native TopK leaves to DataFusion) and the
  aggregation of `hits42` (by `to_timestamp_seconds`, which the classification
  of errors does not know), and over Parquet `hits23` and `hits25` (TopKs whose
  dynamic filters prune the Parquet scan). Over `MemTable`s, the aggregations are up to
  2.1x faster at SF 0.1 (`COUNT(DISTINCT)`, `GROUP BY` with TopKs), and between
  1.3x slower and 1.5x faster with 10,000,000 rows; the queries of less than a
  millisecond (`hits0`, `hits1`, `hits2`, `hits6`, `hits29`) pay the fixed cost
  of setting up the pipelines.

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
