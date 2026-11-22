---
orphan: true
---

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel pipelines: DAG, execution contracts, and selected batches

Design for the opt-in `robtandy/datafusion:morsel-execution` experiment. Nothing
here changes the runtime, the stock planner, or Apache DataFusion defaults. See
[upstream scheduler context][upstream-scheduling] and [parallelism context][upstream-parallelism].

## Starting point and reference

At fork revision `c2809d3b3ee73936b182429e1646af290d797c6e`,
[`MorselQueryPlanner::try_lower`][lowering] walks a single optimized logical
input chain. It builds native physical expressions over `Source::Memory` or
`Source::Parquet`, `Operator::Filter | Projection`, and `Sink::Result | Count`.
[`Pipeline::start`][count-run] already implements one implicit edge: after the
workers return `CountState`s, it combines them, calls
[`CountSink::finalize`][count] once, and feeds `FinalizedCount` through the
result projections. The plan display calls this `FinalizedCountSource`, but it
is a function in the count special case, not a schedulable source. Generalize
that edge before adding more blocking operators.

The DuckDB reference is the workspace snapshot `~/github/duckdb-morsel-reference/`
(`REFERENCE_COMMIT`: `9c1e4c5233af5953e6b35e88fbef5089eaace6e9`). These links
show the corresponding code at a separately pinned public DuckDB revision;
the paths and symbols below refer to the workspace snapshot.

- [`src/execution/physical_operator.cpp`][duck-operator]: `BuildPipelines`
  (line 245) attaches a leaf source or appends a unary operator. A blocking
  sink becomes the current pipeline's source and starts a child meta-pipeline
  for its input. `GetLocalSourceState`, `Sink`, `Combine`, and `Finalize`
  separate the roles.
- [`src/include/duckdb/parallel/meta_pipeline.hpp`][duck-meta] groups pipelines
  sharing a sink. [`src/execution/operator/join/physical_join.cpp`][duck-join]:
  `BuildJoinPipelines` (line 31) builds the RHS as a sink, continues the LHS
  through the join operator, and adds a child pipeline when the join also
  needs to be a source.
- [`src/parallel/executor.cpp`][duck-executor]: `ScheduleEventsInternal`
  (line 81) checks for cycles and orders execution, finish, and completion.
  [`src/parallel/pipeline_executor.cpp`][duck-worker]: `Execute` (line 736)
  resumes operators with pending output; `TryFlushCachingOperators` (line 159)
  drains each cache through the downstream suffix before `PushFinalize`
  combines local sink state.

Reuse these boundaries, not DuckDB's C++ interfaces, event hierarchy, or
thread pool.

## Lowering and DAG

Keep a small internal physical tree of **admitted** scans and operators, with
schemas and bound expressions; lower it from today's optimized logical plan
rather than reverse-engineering a stock optimized `ExecutionPlan`. Walk backward
from the result sink, prepending streaming operators to a suffix in execution
order. A scan fills the source slot. A blocking node creates a new sink group
for its child and fills the current source slot with a reader of that group's
finalized artifact. Every pipeline has one source, a (possibly empty) ordered
operator list, and one sink contribution. For a source that itself supplies
many independent morsels, keep one pipeline and let workers claim from that
source, not one pipeline per morsel or per file.

```rust
struct Pipeline {
    source: SourceSpec,
    operators: Vec<OperatorSpec>,
    sink_group: MetaPipelineId,
    requires: Vec<Dependency>,
}

struct MetaPipeline {
    members: Vec<PipelineId>,
    sink: SinkSpec,
}

enum Dependency {
    Finalized(MetaPipelineId),
    Drained(PipelineId),
}
```

`Finalized(group)` means **all** group members drained, all combines succeeded,
and the supervisor committed the group's complete artifact (which can be `()`).
`Drained(pipeline)` means its sources/producers and workers completed
successfully, including all continuations, flushes, and handoffs of local state.
It does not imply the shared sink has finalized. Validate the expanded graph
with both pipeline and group-finalization vertices (each member precedes its
own group's finalization), compatible schemas/artifact types, and no cycles.
The edges order phases; they are not inter-pipeline batch channels.

```text
P0: Scan(build) -> HashBuildSink               [G0]
P1: Scan(probe) -> Filter -> HashProbe -> Group [G1]  requires Finalized(G0)
P2: FinalizedGroups -> TopNSink                [G2]  requires Finalized(G1)
P3: FinalizedTopN -> Result                    [G3]  requires Finalized(G2)
```

- A grouped aggregate, TopN, or sort has distinct sink and source specs with
  shared immutable config. Without TopN, `FinalizedGroups` feeds the result
  directly; ordered output needs a single ordered reader or an explicit ordered
  merge, not arbitrary concurrent claims from sorted runs.
- Hash join chooses a build input explicitly. Lower it into `G0` with
  `HashBuildSink`, and prepend `HashProbe` to the suffix for the probe input.
  The probe spec binds only after `Finalized(G0)`. Start with inner join. An
  outer join that emits unmatched
  build rows adds a source pipeline that also binds `Finalized(G0)`, but
  starts only after **every** probe pipeline drains. Use `Drained` edges for
  that barrier, not an edge on the sink group the extra pipeline itself
  feeds. Probe workers can hand off reserved match
  bitmaps with their worker exits; the extra source reduces them by partition
  before scanning unmatched rows. This keeps build data immutable and avoids
  a shared per-match lock. Track bitmap memory; do not enable this shape until
  the handoff and merge are implemented.
- For unordered `UNION ALL`, fork the suffix per input, preserve the suffix's
  dependencies on each branch, and use one sink group. Each branch owns its
  worker state; the common sink still finalizes just once. Do not add cross-branch
  ordering or deduplication. The same grouping accommodates a join's additional
  unmatched-row source.

Store immutable specs in `Arc<PipelineGraph>`; keep claims, reservations, and
artifacts in the execution, not on `MorselExec`. Role specs reference typed
artifact slots (e.g. `ArtifactKey<JoinTable>`); a worker factory resolves a key
and clones an `Arc<JoinTable>` at bind time. The sink's mutable build state is
owned separately and consumed to make that artifact. No references into the
plan or another task's mutable state; no global map lookup, lock, or downcast
per row. The scheduler holds a slot's artifact until the last dependent
pipeline binds it; running consumers' `Arc`s then keep it alive only as long
as needed. Keep existing admission: session `off` uses stock unless a native
planner is explicitly installed, `prefer` falls back for the whole unsupported
query, and `required` reports the reason. Never retry a real native error on
the stock path.

## Worker contracts

Proposed private signatures (schematic; `Result` is DataFusion's result type).
`RunCtx` supplies the `TaskContext`, cancellation, and access to tracked
producer/finalize work; `WorkerCtx` adds pipeline/worker identity. These
shared context references are `Sync`; the worker driver owns its mutable
metrics and all `Local*` and `OperatorState` values even when Tokio migrates
it between threads. The
source's shared state is `Sync` (e.g. immutable batches plus an atomic claim
cursor, or a bounded reader queue). Operators bind any shared read-only data
once, before constructing local state. In contrast, `GlobalSinkState` only
needs `Send`: a single coordinator owns it and it is not passed to `sink`.

```rust
trait Source: Send + Sync {
    type GlobalSourceState: Send + Sync + 'static;
    type LocalSourceState: Send + 'static;

    fn open(&self, cx: &RunCtx) -> Result<Self::GlobalSourceState>;
    fn local(&self, global: &Self::GlobalSourceState, cx: &WorkerCtx)
        -> Result<Self::LocalSourceState>;
    fn next(&self, global: &Self::GlobalSourceState, local: &mut Self::LocalSourceState,
            cx: &WorkerCtx) -> impl Future<Output = Result<Option<BatchView>>> + Send;
}

trait Operator: Send + Sync {
    type OperatorState: Send + 'static;

    fn local(&self, cx: &WorkerCtx) -> Result<Self::OperatorState>;
    fn execute(&self, input: &BatchView, local: &mut Self::OperatorState,
               cx: &WorkerCtx) -> Result<OperatorStep>;
    fn final_execute(&self, _local: &mut Self::OperatorState,
                     _cx: &WorkerCtx) -> Result<Option<BatchView>> {
        Ok(None)
    }
}

enum OperatorStep {
    NeedMoreInput(Option<BatchView>),
    HaveMoreOutput(BatchView),
    Finished,
}

trait Sink: Send + Sync {
    type LocalSinkState: Send + 'static;
    type GlobalSinkState: Send + 'static;
    type Finalized: Send + Sync + 'static;

    fn open(&self, cx: &RunCtx) -> Result<Self::GlobalSinkState>;
    fn local(&self, cx: &WorkerCtx) -> Result<Self::LocalSinkState>;
    fn sink(&self, batch: BatchView, local: &mut Self::LocalSinkState,
            cx: &WorkerCtx) -> impl Future<Output = Result<()>> + Send;
    fn combine(&self, global: &mut Self::GlobalSinkState, local: Self::LocalSinkState,
               cx: &RunCtx) -> impl Future<Output = Result<()>> + Send;
    fn finalize(&self, global: Self::GlobalSinkState, cx: &RunCtx)
        -> impl Future<Output = Result<Self::Finalized>> + Send;
}
```

Bind these associated types in concrete worker factories. For heterogeneous
operators, a private object-safe adapter can own the typed operator and state
and forward batch calls; specialize existing filter/projection fast paths
where measured. Do not try to store `dyn Source`/`dyn Sink` with unresolved
associated types or return batches borrowed from mutable scratch across awaits.
A pending `next` waits for work; `None` means no more input for that worker,
not a temporary empty ready queue. [`ParquetScan` and `ReadyReaders`][parquet]
keep pending I/O on the producer side and hand workers decode-ready morsels.
`Sink::sink` awaits backpressure only for streaming sinks; a blocking sink
updates its local state and returns without acquiring a shared table lock.

The worker keeps an owned continuation stack of `(operator index, input view)`
frames. `HaveMoreOutput` yields a nonempty batch and retains the same input
and local cursor for the next call. Drain the current output through the
suffix before resuming the deepest pending frame. `NeedMoreInput` consumes the
input, possibly with one last output; `None` here drops a batch, not the worker.
`Finished` emits nothing and permanently stops this worker's chain at that
operator. Discard its upstream frames and caches, flush only its downstream
suffix, and admit this outcome only when discarding the worker's remaining
input is semantically safe. A query-wide LIMIT stop needs a separate protocol.
Take cooperative budget steps even on no-output paths and within long fanout;
a suspended result send owns one pending batch and resumes without rerunning
an operator or fetching more input.

On normal worker EOF, first exhaust continuations, then call `final_execute`
(DuckDB's `FinalExecute`) repeatedly at each operator in source-to-sink order.
`Some` emits a bounded batch through the **suffix**, including all of its
continuations and backpressure; `None` finishes that operator's flush. This
lets an upstream coalescer feed a second coalescer before it too flushes. Only
then hand off local sink state. Never flush or combine as cleanup after an
error or cancellation.

## Run lifecycle, cancellation, and accounting

Preserve [`MorselExec::execute`][stream]'s lazy start: the first stream poll
creates a run, including its ready list, dependency counts, and terminal
status. Use a supervisor with tracked Tokio tasks via
DataFusion's [`JoinSet` and `SpawnedTask`][tasks]; no task per unopened source
or waiting dependency. Spend a run-wide worker budget on ready pipelines;
queue the rest, and release slots as workers finish. Producers and finalizers
have separate bounded admission so they cannot be starved by workers waiting
for them. Keep Parquet's independent producer/ready bounds and the bounded
terminal output channel; the single-worker in-memory streaming case can drive
the same worker rules inline on consumer polls with no task or channel.

```text
Pipeline: Waiting -> Ready -> Running (including worker flush) -> Drained
Sink group: Open -> Combining -> Finalizing -> Finalized
Run: Active -> Succeeded | Failed(first error) | Cancelled
```

The coordinator consumes each successful worker exit once and can combine
exits as they arrive; other workers may still be sinking into their own
locals. Finalization waits for all pipeline drains and all combines, including
when there were no rows or no workers. Make expensive merge/finalize work
cooperative and tracked so it cannot hide peer errors from the supervisor.
For a future radix table, a single logical `finalize` can hand disjoint
partitions to bounded tracked tasks and join all of them before returning one
artifact. No partition may publish independently. Only the supervisor commits
that artifact and decrements dependent counts in the same success transition;
a worker return, an empty queue, or a closed sender is not a completion signal.
Successful stream EOF requires run success **and** an empty result buffer.

- The first observed execution error or worker failure stops admissions and
  signals terminal status separately from the result queue, even if that
  queue is full and the consumer is idle. Prefer that status over queued
  batches and preserve the original error. Drop closes the receiver and
  requests the same teardown. Use a short run-state gate for the terminal
  transition, artifact commit, and start admission (never held across an
  await); late completions cannot unlock or start a pipeline after cancellation.
  A complete artifact committed earlier is discarded with the run, never
  exposed as a partially built table. Streaming rows already yielded cannot
  be retracted; blocking stages do not emit unfinished results.
- Cancel pending I/O and permit waits, abort peers, join them where the
  supervisor remains live, and discard queued work and unpublished state.
  Give the supervisor teardown access to the result receiver: with `mpsc`,
  share it under a guard used only for `poll_recv`, close, or drain, never
  across an await. On failure close it, abort and join senders (including
  outstanding permit holders), then discard buffered output. This releases
  queue reservations even if the consumer never polls the error. Keep an
  abort-on-drop owner as a backstop, including for producer and finalize
  tasks. Stream drop cannot synchronously join tasks; RAII releases resources
  as tasks unwind or next yield. Check cancellation at bounded work steps,
  including within CPU-heavy probe/merge loops; Tokio cannot preempt a
  synchronous kernel already running. Do not detach blocking work.
- Register run-local [`MemoryConsumer`s][reservation] with
  [`TaskContext::memory_pool`][task-context]. Before allocating or retaining
  execution-owned state, use `MemoryReservation::try_grow`/`try_resize` for
  capacities of tables, bitmaps, cached batches, selections, and queued or
  pending output buffers; also account for owned source buffers under the
  source's budget. Do not re-charge already-accounted shared Arrow buffers
  for every clone, or charge preexisting table storage as a new allocation.
  Keep the charge with the last internal owner (e.g. a backing-batch lease or
  finalized artifact in an `Arc`). Transfer charges with `split`/`take` when
  ownership moves; when copying or rehashing, reserve the transient peak
  while inputs and destination coexist. Release internal output ownership
  when handed to the caller; its subsequent retention is outside this run.
  Drop releases reservations on every exit. A queue's item bound is not a
  byte limit. Without implemented spill, a reservation failure is a query
  error, not an unbounded queue or a wait for memory held by that same stage.
- Keep [#138's `RunMetrics`/`WorkerMetrics`][metrics-source] and
  [documented metrics][metrics-doc]: one optional report per run, published
  after owners release it, including partial measurements on failure/drop.
  Add a `pipeline` label alongside local `worker` IDs; record dependency
  wait and combine/finalize work separately from source ready wait, decode,
  process, and output backpressure. Keep queue peaks distinct from pool
  reservations, and plan-lifetime `ExecutionPlan::metrics()` separate. With
  metrics off, do not allocate a report, collect footprints, or time steps.

## Arrow batch views and materialization

Start with the eager baseline, not a new Arrow encoding. `filter_record_batch`
produces ordinary arrays that work with any `PhysicalExpr`, but can copy wide
columns and leave tiny batches. [`FilterExec`][filter] uses
[`LimitedBatchCoalescer`][coalesce] over Arrow's `BatchCoalescer`. Per worker,
prune columns after evaluating their predicate inputs, coalesce small batches
toward session `batch_size`, and bypass the cache for already-large batches
(after emitting any earlier buffered rows when order matters). A fused
`push_batch_with_filter` can avoid a separate filter/copy pass. `final_execute`
drains the cache at EOF. Coalescing across backing batches is a
materialization point, not a concatenation of unrelated row indices.

For selected-aware paths, use DuckDB's [`SelectionVector`][duck-selection] and
[`DataChunk::Slice`][duck-chunk] as the analogy, but keep Arrow columns intact:

```rust
struct BatchView {
    batch: RecordBatch,
    selection: Selection,
}

enum Selection {
    All,
    Mask { bits: BooleanBuffer, selected: usize },
    Indices(UInt32Array),
}
```

The sketch omits the backing-buffer lease. Selection coordinates are relative
to `batch` (including when it is an Arrow slice). Mask length equals physical
row count; only valid SQL `TRUE` selects, so the selection itself has no
nulls. Validate `PhysicalExpr` array length before using even the count-only
shortcut. Filter indices are non-null, in bounds, strictly increasing, and
unique; choose `UInt32Array` only when row positions fit, otherwise use a
mask or materialize. Counts use selected cardinality, including zero-column
batches. Selections from successive filters intersect in backing coordinates;
if a predicate was evaluated on compacted rows, compose its positions through
the prior selection, never apply compact coordinates to the original batch.
Use `All` for a full selection. Prefer masks for dense filters (one bit per
physical row, cheap popcount) and indices for sparse hot loops (four bytes per
selected row, gather without scanning rejected positions); choose conversion
thresholds by measurement, including conversion and buffer-retention cost.

| Consumer            | Use the selection until it stops paying                                                                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Filtered `COUNT(*)` | Accumulate the selected cardinality with checked arithmetic; do not copy or coalesce rows. Keep today's direct predicate count path where applicable.                                                                                                        |
| Filter → hash probe | Hash only selected keys; retain backing row IDs for payload gathers on actual matches. Probe continuation holds the input view and match cursor; bound each output batch. Join output with repeated pairs is a newly built batch, not this filter selection. |
| Filter → `GROUP BY` | Read only selected key/argument rows; update local groups, with SQL null keys grouped together. Materialize the needed arguments first if the accumulator requires dense Arrow arrays.                                                                       |
| Wide projection     | For pure column projection, share/reorder arrays before compaction. For general expressions, compact only required inputs and rebind column indices; don't evaluate a UDF on rejected rows merely to slice its result.                                       |

For unsupported consumers, explicitly materialize via `filter_record_batch`
(mask) or Arrow `take` (indices), and reset to `All`. Keep exact schema,
nullability, metadata, and row counts (use explicit row count for zero columns).
Only ordinary `RecordBatch`es cross the `MorselExec` stream boundary. Small
selections that retain large backing buffers can cost more than compaction,
especially across a blocking sink or queue; compact when keeping a view would
increase retained memory. Selectivity alone is not enough to pick a policy.

## Module boundaries and migration

Keep `datafusion/core/src/morsel.rs` as the module facade with current public
exports. Below it, keep these private paths under `datafusion/core/src/morsel/`:

| Path                                      | Responsibility                                                                                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `planner.rs`, `fallback.rs`               | Admission, logical-to-native physical lowering, DAG construction; existing whole-query fallback stays in `fallback.rs`.                                      |
| `graph.rs`, `exec.rs`                     | Immutable IDs/specs, schema/dependency validation; `MorselExec` plan properties, display, expression traversal, and lazy Arrow stream boundary in `exec.rs`. |
| `traits.rs`, `driver.rs`, `schedule.rs`   | Role and state contracts; continuation/flush loop; ready queue, task ownership, joins, and artifact commits, respectively.                                   |
| `batch.rs`, `operators.rs`                | Selection/materialization helpers; filter, projection, and worker-local coalescing.                                                                          |
| `memory.rs`, `parquet.rs`, `parquet/`     | Memory source and disjoint batch-range claims; existing Parquet source and producer internals.                                                               |
| `result.rs`, `count.rs`, `metrics.rs`     | Result boundary sink, existing count and finalized count roles, and run instrumentation.                                                                     |
| `aggregate.rs`, `hash_join.rs`, `sort.rs` | Future role adapters and artifacts per blocking operator family, not new variants in a monolithic worker loop.                                               |

1. **Extract, then make the count edge explicit.** Keep the existing scan and
   expression paths, claim sizes, inline poll/yield budget, and count mask
   fast path of [#145]. Use a one-pipeline graph for scans and a two-pipeline
   graph for count. Preserve `EXPLAIN`/`ANALYZE`, expression traversal, schema,
   and final versus incremental emission. Keep [#148]'s Parquet producer
   tuning source-local and [#149]'s coalescing/capacity policy local to driver
   and result sink. If those land first, migrate them, not the old constants;
   do not bundle new queue or batch heuristics with graph extraction.
2. **[#140] after the scan and output tuning.** Add a sink for one admitted
   integer group key and `COUNT(*)`, using local maps and checked merge,
   then a bounded finalized group reader. Start on memory, then cover
   admitted Parquet. Empty grouped input yields no groups (unlike ungrouped
   count). No spill or ordering in this slice. Use stock
   [`AggregateExec`][stock-agg] for semantics, not a
   stock aggregate stream wrapped as a morsel sink.
3. **HashJoin, then TopN/Sort.** Start with one sealed, pool-accounted build,
   inner equijoin, and bounded probe continuation. Use [`HashJoinExec`][stock-join]
   as a reference for key/null semantics and build reservations, not shared
   execution state. Add unmatched scans or radix/spill as later milestones.
   Implement TopN via local candidates and one final merge for supported
   comparators/limits, then full Sort via reserved sorted runs and an ordered
   merge; refer to [`TopK`][stock-topk] and [`SortExec`][stock-sort]. Advertise
   only the ordering actually delivered by the terminal stream. Keep unsupported
   join, grouping, and sort shapes on the existing fallback/rejection path.
4. **Introduce selection-aware consumers separately.** Measure against the
   eager/coalesced path before switching more operators. Compare matched Rust
   builds and CPU budgets against the pre-change fork and stock `off`, using
   the [#127 in-memory harness][mem-bench] (one worker, parallel balanced and
   skewed) and [#128 Parquet harness][parquet-bench] (warm and cold, low and
   high selectivity). Check first batch, first 8,192 rows, full drain, output
   batch sizes/count, #138 waits/peaks, and count parity. Keep an unfiltered
   control and compare metrics-disabled runs as well. Do not trade lower
   average time for worse tails, cancellation, or memory bounds without
   measuring and explaining it.

For runtime PRs, extend [native memory SQL tests][memory-tests],
[native Parquet tests][parquet-tests], and [gated runtime tests][runtime-tests].
Before changing lowering for each new shape, assert that `required` produces
an actual native plan (currently rejected); then compare exact schema and
row multisets to `off` (ordered rows for Sort/TopN with a unique tie-breaker).
For example,
`SELECT k, COUNT(*) AS n FROM t WHERE v > 0 GROUP BY k` with passing keys
`[NULL, 7, 7]` must emit `(NULL, 1)` and `(7, 2)`, and no rows on empty input.
Use barriers/timeouts for build-before-probe, two producers sharing one sink,
EOF with multiple buffered operators, and a failure or drop with full output,
pending I/O, or blocked finalization. Assert exactly-once finalization, no
secondary admissions on failure, no surviving reservations after cleanup,
and repeated/concurrent runs of the same plan. Cover fanout across batch
boundaries, zero-column batches, all/none/NULL masks, and a forced pool failure
at merge peak. This note does not add or enable any runtime test or SQL shape.

[lowering]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel.rs#L90-L281
[count-run]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel.rs#L747-L826
[count]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel/count.rs#L45-L93
[stream]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel.rs#L1001-L1046
[parquet]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel/parquet.rs
[tasks]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/common-runtime/src/lib.rs#L19-L22
[reservation]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/execution/src/memory_pool/mod.rs#L257-L517
[task-context]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/execution/src/task.rs#L121-L123
[metrics-source]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel/metrics.rs#L53-L212
[metrics-doc]: morsel_metrics.md
[filter]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/physical-plan/src/filter.rs#L1830-L1914
[coalesce]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/physical-plan/src/coalesce/mod.rs
[stock-agg]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/physical-plan/src/aggregates/mod.rs#L864
[stock-join]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/physical-plan/src/joins/hash_join/exec.rs#L840
[stock-topk]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/physical-plan/src/topk/mod.rs#L100
[stock-sort]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/physical-plan/src/sorts/sort.rs#L1436
[memory-tests]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/tests/morsel_execution.rs
[parquet-tests]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/tests/morsel_parquet.rs
[runtime-tests]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel/tests.rs
[mem-bench]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/benchmarks/morsel_execution.md
[parquet-bench]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/benchmarks/morsel_parquet/README.md
[duck-operator]: https://github.com/duckdb/duckdb/blob/87d96b7773a10356b11cc15c352125acdc58a9ce/src/execution/physical_operator.cpp#L127-L275
[duck-meta]: https://github.com/duckdb/duckdb/blob/87d96b7773a10356b11cc15c352125acdc58a9ce/src/include/duckdb/parallel/meta_pipeline.hpp#L26-L42
[duck-join]: https://github.com/duckdb/duckdb/blob/87d96b7773a10356b11cc15c352125acdc58a9ce/src/execution/operator/join/physical_join.cpp#L31-L87
[duck-executor]: https://github.com/duckdb/duckdb/blob/87d96b7773a10356b11cc15c352125acdc58a9ce/src/parallel/executor.cpp#L70-L137
[duck-worker]: https://github.com/duckdb/duckdb/blob/87d96b7773a10356b11cc15c352125acdc58a9ce/src/parallel/pipeline_executor.cpp
[duck-selection]: https://github.com/duckdb/duckdb/blob/87d96b7773a10356b11cc15c352125acdc58a9ce/src/include/duckdb/common/types/selection_vector.hpp
[duck-chunk]: https://github.com/duckdb/duckdb/blob/87d96b7773a10356b11cc15c352125acdc58a9ce/src/include/duckdb/common/types/data_chunk.hpp#L149-L158
[upstream-scheduling]: https://github.com/apache/datafusion/issues/21719
[upstream-parallelism]: https://github.com/apache/datafusion/pull/2226
[#140]: https://github.com/robtandy/datafusion/issues/140
[#145]: https://github.com/robtandy/datafusion/issues/145
[#148]: https://github.com/robtandy/datafusion/issues/148
[#149]: https://github.com/robtandy/datafusion/issues/149
