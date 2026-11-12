<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# What runs natively, and what stays in DataFusion

`PipelineQueryPlanner` lowers every node of the `ExecutionPlan` that DataFusion's
`DefaultPhysicalPlanner` produces to the native operators of this crate (see
`planner.rs` and [the architecture](architecture.md#datafusion-integration)).
A node that is not lowered is executed by DataFusion, with its whole subtree,
in a `BridgeOperator`. Under the [correctness policy](compatibility.md), the
only nodes that may stay in DataFusion are user-defined `ExecutionPlan`s: a
node is never left to DataFusion to reproduce its behaviour.

This document lists every `ExecutionPlan` implementation of DataFusion's
crates `datafusion/physical-plan`, `datafusion/datasource*`,
`datafusion/catalog*` and `datafusion/core`, and how the planner lowers it.
`bridge::bridged_plans` returns the plans that a lowered operator tree leaves
to DataFusion.

## Inventory

The column _Native operator_ names the operators that execute a node. A node
that is _removed_ has no operator of its own: the pipeline engine does what it
does by itself (e.g. it has no partitions to distribute rows over, and it
schedules its tasks cooperatively).

### `datafusion/physical-plan`

| `ExecutionPlan`                                            | Native operator                                                                                                                                                                                                |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AggregateExec`                                            | `PhysicalHashAggregate` (with the stages of a two-stage aggregation)                                                                                                                                           |
| `AnalyzeExec` (`EXPLAIN ANALYZE`)                          | `PhysicalAnalyze` (which runs the native plan of the query)                                                                                                                                                    |
| `AsOfJoinExec` (`ASOF JOIN`)                               | `PhysicalAsOfJoin` (the right rows sorted by their keys and match values, which every left row finds by a binary search)                                                                                       |
| `AsyncFuncExec` (async UDFs)                               | `PhysicalAsyncFunc`                                                                                                                                                                                            |
| `BoundedWindowAggExec`, `WindowAggExec`                    | `PhysicalWindow` (with the sorts and repartitionings below them)                                                                                                                                               |
| `BufferExec`                                               | removed                                                                                                                                                                                                        |
| `CoalesceBatchesExec`                                      | removed, or `PhysicalLimit` with a fetch of a single partition (of several, see [conditions](#conditions-in-which-native-operators-refuse-a-node))                                                             |
| `CoalescePartitionsExec`                                   | removed, or `PhysicalLimit` with a fetch                                                                                                                                                                       |
| `CooperativeExec`                                          | removed                                                                                                                                                                                                        |
| `CrossJoinExec`                                            | `PhysicalCrossJoin`                                                                                                                                                                                            |
| `EmptyExec`                                                | `EmptyScanSource`                                                                                                                                                                                              |
| `ExplainExec` (`EXPLAIN`)                                  | `ExplainSource`                                                                                                                                                                                                |
| `FilterExec`                                               | `PhysicalFilter` (in a `CachingOperator`)                                                                                                                                                                      |
| `GlobalLimitExec`, `LocalLimitExec`                        | `PhysicalLimit`                                                                                                                                                                                                |
| `HashJoinExec`                                             | `PhysicalHashJoin`                                                                                                                                                                                             |
| `LazyMemoryExec` (`generate_series`, `range`)              | `LazyMemorySource`                                                                                                                                                                                             |
| `MaterializedCteExec`, `MaterializedCteScanExec`           | `PhysicalMaterializedCte`, `MaterializedCteBody`, `MaterializedCteScan`                                                                                                                                        |
| `NestedLoopJoinExec`                                       | `PhysicalNestedLoopJoin`                                                                                                                                                                                       |
| `PartialSortExec`                                          | `PhysicalPartialSort` (with a fetch of several partitions, one for every partition, in a `PhysicalUnion`; of an input of several pipelines, a `PhysicalSort` or a `PhysicalTopK`, see [below](#partial-sorts)) |
| `PartitionedTopKExec`                                      | `PhysicalPartitionedTopK` over a `PhysicalSort`                                                                                                                                                                |
| `PiecewiseMergeJoinExec`                                   | `PhysicalRangeJoin` (the build rows sorted by their keys, which every probe row finds by a binary search)                                                                                                      |
| `PlaceholderRowExec`                                       | `PhysicalPlaceholderRow`                                                                                                                                                                                       |
| `ProjectionExec`                                           | `PhysicalProjection`                                                                                                                                                                                           |
| `RecursiveQueryExec`, `WorkTableExec`                      | `PhysicalRecursiveCte`, `WorkTableScanSource` ([#61](https://github.com/robtandy/datafusion/issues/61))                                                                                                        |
| `RepartitionExec`                                          | removed (by round robin, by hash or by range, and preserving the order of several partitions)                                                                                                                  |
| `ScalarSubqueryExec`                                       | `PhysicalScalarSubquery`                                                                                                                                                                                       |
| `SortExec`                                                 | `PhysicalSort`, or `PhysicalTopK` with a fetch                                                                                                                                                                 |
| `SortMergeJoinExec`, `SymmetricHashJoinExec`               | not yet: native merge and symmetric joins are [#60](https://github.com/robtandy/datafusion/issues/60)                                                                                                          |
| `SortPreservingMergeExec`                                  | `PhysicalSort` or `PhysicalTopK` of the merged partitions (a merge of sorted rows is their sort); of inputs that never end, `PhysicalStreamingMerge` of the native operators of every partition                |
| `StreamingTableExec` (`information_schema`, `StreamTable`) | `StreamingTableSource`                                                                                                                                                                                         |
| `UnionExec`, `InterleaveExec`                              | `PhysicalUnion`                                                                                                                                                                                                |
| `UnnestExec`                                               | `PhysicalUnnest`                                                                                                                                                                                               |

The test plans of `datafusion_physical_plan::test` (`MockExec`, `BarrierExec`,
`ErrorExec`, `StatisticsExec`, `BlockingExec`, `PanicExec`, `TestMemoryExec`)
and of the tests of the crate are user-defined plans to the planner.

### `datafusion/datasource*`

| `ExecutionPlan`                                                       | Native operator                                                                                                                               |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `DataSourceExec` of a `MemorySourceConfig`                            | `MemoryScanSource`                                                                                                                            |
| `DataSourceExec` of Parquet files                                     | `ParquetScanSource`                                                                                                                           |
| `DataSourceExec` of CSV files                                         | `CsvScanSource`                                                                                                                               |
| `DataSourceExec` of other files                                       | `FileScanSource`: JSON (newline-delimited and arrays), Avro, Arrow IPC (files and streams), and any other `FileSource`, with its `Morselizer` |
| `DataSourceExec` of another `DataSource`                              | stays in DataFusion: a user-defined `DataSource`                                                                                              |
| `DataSinkExec` (`INSERT`, `COPY TO`)                                  | `PhysicalDataSink`, which writes with the `DataSink` of the node                                                                              |
| `MemDeleteExec`, `MemUpdateExec` (`DELETE`, `UPDATE` of a `MemTable`) | `MemoryDmlSource`, which changes the partitions of the table under their locks, as these nodes do                                             |

The crates `datafusion-datasource-{arrow,avro,csv,json,parquet}` implement
`FileSource`s and `DataSink`s, but no `ExecutionPlan`.

### `datafusion/catalog*`

`datafusion-catalog` implements no `ExecutionPlan`: the `DELETE` and `UPDATE`
of a `MemTable` are the `MemDeleteExec` and `MemUpdateExec` of
`datafusion-datasource` (next to the `MemSink` of its `INSERT`), which expose
the partitions, predicates and `SET` expressions that they apply.
`datafusion-catalog-listing` implements no `ExecutionPlan` either: its tables are
scanned by `DataSourceExec`s and written by `DataSinkExec`s. The
`information_schema` tables and the `StreamTable`s of `datafusion-catalog` are
`StreamingTableExec`s.

### `datafusion/core`

`datafusion/core` implements `ExecutionPlan`s only in the tests of its
physical planner (`NoOpExecutionPlan`, `OkExtensionNode`,
`InvariantFailsExtensionNode`, `ExecutableInvariantFails`), which are
user-defined plans to the pipeline engine.

## Nodes that stay in DataFusion

- **User-defined `ExecutionPlan`s** (e.g. of extension nodes, or of
  `TableProvider`s), and `DataSourceExec`s of user-defined `DataSource`s, with
  their subtrees: they are the user's own code.
- **The nodes of [#60](https://github.com/robtandy/datafusion/issues/60)**,
  until it lowers them: `SortMergeJoinExec` and `SymmetricHashJoinExec`.

- A `MaterializedCteExec` one of whose scans stays in DataFusion (e.g. below
  a user-defined node): DataFusion runs the body of a CTE for such a scan,
  so the CTE stays in DataFusion, which runs its body once, for all scans.

## Conditions in which native operators refuse a node

The native operators refuse some nodes of the types they implement, in the
conditions that their `try_lower` functions (or `supports_*` functions)
document; the planner then leaves these nodes to DataFusion. These
conditions belong to the operators, and to the issues that own them (e.g.
[#63](https://github.com/robtandy/datafusion/issues/63) for the memory
accounting of all native operators):

- windows, nested loop joins and cross joins in sessions whose memory pool has
  a limit: they do not reserve the memory of their rows yet;
- sorts, windows and partitioned TopKs of inputs that never end (their native
  operators read all of their input first), and by keys that the native sort
  orders otherwise (unions); and the merges of sorted partitions that never
  end whose partitions the planner cannot split (see
  [partial sorts](#partial-sorts)). `PhysicalStreamingMerge` runs every
  partition in an execution of its own: the partitions that scan a
  materialized CTE around the merge are lowered again by every execution of
  the merge, with scans of the rows of its body;
- aggregations by groups that their input clusters
  (`group_clustering_mode`), and the other aggregations that
  `PhysicalHashAggregate::try_lower` refuses;
- the hash joins that `PhysicalHashJoin::try_lower` refuses (e.g. with a
  fetch, or null-aware with several keys);
- `LocalLimitExec`s of several partitions (and `CoalesceBatchesExec`s with a
  fetch, which DataFusion does not plan any more, of several partitions) that
  a merge with a fetch does not make redundant: they limit every partition;
- the sorts, windows, aggregations and joins of values that address more than
  their memory, and the filters and projections of rows that the native
  operators cannot gather (see the module documentation of `planner.rs`).

## Partial sorts

A `PhysicalPartialSort` sorts the rows of its input as a single thread
receives them, in a single pipeline:

- A `PartialSortExec` with a fetch of several partitions keeps the first rows
  of every partition: it is lowered as the `PhysicalUnion` of a
  `PhysicalPartialSort` of every partition, with the fetch, of the plans of
  the partitions of its input. These are the partitions of scans of in-memory
  batches, of files (a plan for every file group), of streaming tables and
  of the generators of a `LazyMemoryExec`, and of the unions above them, and of the nodes that compute every
  partition of the same partition of their input (projections, filters,
  `LocalLimitExec`s, sorts, aggregations and windows of every partition, and
  the nodes that only change the size of batches).
- The partitions of a hash or range `RepartitionExec` below such a partial
  sort are the rows that an execution of its input routes to them, once,
  with DataFusion's `BatchPartitioner` (also by volatile keys). Of an input
  of a single partition, the rows of every partition keep their order: a
  `PhysicalPartialSortOfPartitions` sorts every partition as its rows arrive
  (it streams, also over inputs that never end, and ends once every
  partition emitted its rows). Of an input of several partitions, whose rows
  arrive in no order, a `PhysicalTopKOfPartitions` keeps the first rows of
  every partition once its input ended (an input that never ends is sorted
  as a single partition). A merge of these partial sorts (which emit at most
  their fetch of every partition) is the native sort of their rows, also
  over inputs that never end.
- The partitions of other inputs (e.g. of a round robin repartitioning, whose
  partitions depend on the batches in which DataFusion reads its rows, or of
  a join) are not kept apart: the partial sort keeps the first rows of all of
  them. A merge above that keeps at most the rows of every partition is the
  native TopK of the input of the partial sorts.
- The rows of an input that runs in several pipelines (e.g. of a union below
  a `CoalescePartitionsExec`) would reach several states of the partial
  sort: such an input is sorted by a `PhysicalSort` (or a `PhysicalTopK` with
  the fetch) instead. DataFusion orders the interleaved partitions of such an
  input by a prefix only if all its rows have the same prefix, which makes
  the partial sort a sort.

## Memory

The operators that this document adds reserve the memory that they hold in
the memory pool of the session, and fail with `ResourcesExhausted` when it
runs out (they do not spill):

- the sources (`LazyMemorySource`, `StreamingTableSource`, `ExplainSource`
  and `FileScanSource`) reserve the batch that a thread emits, and
  `PhysicalAnalyze` holds no rows (it counts the rows of its plan);
- `PhysicalAsOfJoin` reserves the rows of its right input, their keys (in the
  row format) and match values as they arrive, and their concatenations and
  order before it allocates them; `MemoryDmlSource` reserves the rows that
  replace those of a partition until they replace them (the table then holds
  them);
- `PhysicalAsyncFunc` reserves the rows of the chunk it evaluates, and an
  estimate of their results before it evaluates them; `PhysicalUnnest`
  reserves the indices that it takes the rows of an unnest with, and an
  estimate of these rows, before it allocates them (it unnests a single level
  in windows of at most 2048 rows, also of a row that expands into many
  rows), and the rows it did not emit yet; `PhysicalPartialSort` reserves the
  rows of the groups it buffers, and what sorting them allocates, before it
  sorts them; and `PhysicalDataSink` the rows that wait for its writer (and,
  for a memory table, which holds them until it commits them, all rows until
  the writer completes);
- `MaterializedCteBody` reserves the rows of the body of a CTE until the
  execution ends;
- `PhysicalRangeJoin` reserves the rows of its build input and their keys as
  they arrive, and their concatenations and order before it allocates them,
  and the build rows that it emits once it probed all rows;
  `PhysicalStreamingMerge` reserves the batch of every input whose rows it
  merges, and their keys; `PhysicalTopKOfPartitions` the rows that every
  partition holds, and what sorting them allocates, before it sorts them,
  and `PhysicalPartialSortOfPartitions` the groups that it buffers and the
  sorted rows that it did not emit yet; `MaterializedCteScan` reserves its list of the
  batches of the body and of their chunks;
- `PhysicalPartitionedTopK` holds no rows (the `PhysicalSort` below it does,
  which, like the other native sorts, does not reserve its memory yet), but
  reserves the keys of the last row it ranked, and, before it allocates them,
  an estimate of the keys of a chunk in the row format, the indices and ranks
  of the rows that it keeps and these rows.
