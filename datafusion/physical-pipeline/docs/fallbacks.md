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

| `ExecutionPlan`                                            | Native operator                                                                                                                                    |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AggregateExec`                                            | `PhysicalHashAggregate` (with the stages of a two-stage aggregation)                                                                               |
| `AnalyzeExec` (`EXPLAIN ANALYZE`)                          | `PhysicalAnalyze` (which runs the native plan of the query)                                                                                        |
| `AsOfJoinExec` (`ASOF JOIN`)                               | stays in DataFusion, see [below](#nodes-that-stay-in-datafusion)                                                                                   |
| `AsyncFuncExec` (async UDFs)                               | `PhysicalAsyncFunc`                                                                                                                                |
| `BoundedWindowAggExec`, `WindowAggExec`                    | `PhysicalWindow` (with the sorts and repartitionings below them)                                                                                   |
| `BufferExec`                                               | removed                                                                                                                                            |
| `CoalesceBatchesExec`                                      | removed, or `PhysicalLimit` with a fetch of a single partition (of several, see [conditions](#conditions-in-which-native-operators-refuse-a-node)) |
| `CoalescePartitionsExec`                                   | removed, or `PhysicalLimit` with a fetch                                                                                                           |
| `CooperativeExec`                                          | removed                                                                                                                                            |
| `CrossJoinExec`                                            | `PhysicalCrossJoin`                                                                                                                                |
| `EmptyExec`                                                | `EmptyScanSource`                                                                                                                                  |
| `ExplainExec` (`EXPLAIN`)                                  | `ExplainSource`                                                                                                                                    |
| `FilterExec`                                               | `PhysicalFilter` (in a `CachingOperator`)                                                                                                          |
| `GlobalLimitExec`, `LocalLimitExec`                        | `PhysicalLimit`                                                                                                                                    |
| `HashJoinExec`                                             | `PhysicalHashJoin`                                                                                                                                 |
| `LazyMemoryExec` (`generate_series`, `range`)              | `LazyMemorySource`                                                                                                                                 |
| `MaterializedCteExec`, `MaterializedCteScanExec`           | `PhysicalMaterializedCte`, `MaterializedCteBody`, `MaterializedCteScan`                                                                            |
| `NestedLoopJoinExec`                                       | `PhysicalNestedLoopJoin`                                                                                                                           |
| `PartialSortExec`                                          | `PhysicalPartialSort`                                                                                                                              |
| `PartitionedTopKExec`                                      | `PhysicalPartitionedTopK` over a `PhysicalSort`                                                                                                    |
| `PiecewiseMergeJoinExec`                                   | `PhysicalNestedLoopJoin` (the join of its range predicate)                                                                                         |
| `PlaceholderRowExec`                                       | `PhysicalPlaceholderRow`                                                                                                                           |
| `ProjectionExec`                                           | `PhysicalProjection`                                                                                                                               |
| `RecursiveQueryExec`, `WorkTableExec`                      | not yet: native recursive queries are [#61](https://github.com/robtandy/datafusion/issues/61)                                                      |
| `RepartitionExec`                                          | removed (by round robin, by hash or by range, and preserving the order of several partitions)                                                      |
| `ScalarSubqueryExec`                                       | `PhysicalScalarSubquery`                                                                                                                           |
| `SortExec`                                                 | `PhysicalSort`, or `PhysicalTopK` with a fetch                                                                                                     |
| `SortMergeJoinExec`, `SymmetricHashJoinExec`               | not yet: native merge and symmetric joins are [#60](https://github.com/robtandy/datafusion/issues/60)                                              |
| `SortPreservingMergeExec`                                  | `PhysicalSort` or `PhysicalTopK` of the merged partitions (a merge of sorted rows is their sort)                                                   |
| `StreamingTableExec` (`information_schema`, `StreamTable`) | `StreamingTableSource`                                                                                                                             |
| `UnionExec`, `InterleaveExec`                              | `PhysicalUnion`                                                                                                                                    |
| `UnnestExec`                                               | `PhysicalUnnest`                                                                                                                                   |

The test plans of `datafusion_physical_plan::test` (`MockExec`, `BarrierExec`,
`ErrorExec`, `StatisticsExec`, `BlockingExec`, `PanicExec`, `TestMemoryExec`)
and of the tests of the crate are user-defined plans to the planner.

### `datafusion/datasource*`

| `ExecutionPlan`                            | Native operator                                                                                                                               |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `DataSourceExec` of a `MemorySourceConfig` | `MemoryScanSource`                                                                                                                            |
| `DataSourceExec` of Parquet files          | `ParquetScanSource`                                                                                                                           |
| `DataSourceExec` of CSV files              | `CsvScanSource`                                                                                                                               |
| `DataSourceExec` of other files            | `FileScanSource`: JSON (newline-delimited and arrays), Avro, Arrow IPC (files and streams), and any other `FileSource`, with its `Morselizer` |
| `DataSourceExec` of another `DataSource`   | stays in DataFusion: a user-defined `DataSource`                                                                                              |
| `DataSinkExec` (`INSERT`, `COPY TO`)       | `PhysicalDataSink`, which writes with the `DataSink` of the node                                                                              |

The crates `datafusion-datasource-{arrow,avro,csv,json,parquet}` implement
`FileSource`s and `DataSink`s, but no `ExecutionPlan`.

### `datafusion/catalog*`

| `ExecutionPlan`                                                       | Native operator                                                 |
| --------------------------------------------------------------------- | --------------------------------------------------------------- |
| `MemDeleteExec`, `MemUpdateExec` (`DELETE`, `UPDATE` of a `MemTable`) | stay in DataFusion, see [below](#nodes-that-stay-in-datafusion) |

`datafusion-catalog-listing` implements no `ExecutionPlan`: its tables are
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
- **The nodes of [#60](https://github.com/robtandy/datafusion/issues/60) and
  [#61](https://github.com/robtandy/datafusion/issues/61)**, until these issues
  lower them: `SortMergeJoinExec`, `SymmetricHashJoinExec`, `RecursiveQueryExec`
  and `WorkTableExec`.
- **Nodes that do not expose what they compute.** Their definitions are
  private to their crates, so that no other crate can implement them:

  - `AsOfJoinExec` has no accessors for its equality keys, its match
    condition (its comparison operator) and its projection;
  - `MemDeleteExec` and `MemUpdateExec` are private types of
    `datafusion-catalog`, as are the partitions of the `MemTable` that they
    change.

  Lowering them takes accessors in their crates (`AsOfJoinExec::on`,
  `AsOfJoinExec::match_condition` and `AsOfJoinExec::projection`, and public
  `MemDeleteExec` and `MemUpdateExec`).

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

- windows, nested loop joins and cross joins (and the piecewise merge joins
  that are nested loop joins) in sessions whose memory pool has a limit: they
  do not reserve the memory of their rows yet;
- sorts, merges of sorted partitions, windows and partitioned TopKs of inputs
  that never end (their native operators read all of their input first), and
  by keys that the native sort orders otherwise (unions);
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

## Memory

The operators that this document adds reserve the memory that they hold in
the memory pool of the session, and fail with `ResourcesExhausted` when it
runs out (they do not spill):

- the sources (`LazyMemorySource`, `StreamingTableSource`, `ExplainSource`)
  reserve the batch that a thread emits; `FileScanSource` reads with the
  `Morselizer` of its files, as the other file scans do, and
  `PhysicalAnalyze` holds no rows (it counts the rows of its plan);
- `PhysicalAsyncFunc` reserves the rows of the chunk it evaluates, and their
  results, `PhysicalUnnest` the unnested rows it did not emit yet,
  `PhysicalPartialSort` the rows of the groups it buffers and sorts, and
  `PhysicalDataSink` the rows that wait for its writer;
- `MaterializedCteBody` reserves the rows of the body of a CTE until the
  execution ends;
- `PhysicalPartitionedTopK` holds no rows (the `PhysicalSort` below it does,
  which, like the other native sorts, does not reserve its memory yet).
