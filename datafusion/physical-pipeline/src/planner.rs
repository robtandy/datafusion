// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PipelineQueryPlanner`]: plugs the pipeline engine into a DataFusion
//! `SessionContext`.
//!
//! The planner first runs a regular DataFusion [`PhysicalPlanner`]
//! (normally `DefaultPhysicalPlanner`) so that the plan is bound, coerced and
//! optimized exactly as in default DataFusion: the output schema (including
//! field metadata), the physical expressions and all pushdowns into scans are
//! those of DataFusion. The resulting [`ExecutionPlan`] is then *lowered* into
//! a [`PhysicalOperator`] tree by [`lower_execution_plan`], and wrapped in a
//! [`PipelineRootExec`] that runs it with the pipeline engine.
//!
//! # Lowering
//!
//! The lowering walks the plan from its root and replaces every node, or
//! chain of nodes, that a native operator reproduces exactly by that
//! operator, whose inputs are lowered in turn:
//!
//! | DataFusion plan                                                                    | Native operator                                    |
//! | ---------------------------------------------------------------------------------- | -------------------------------------------------- |
//! | `SortPreservingMergeExec -> SortExec`, `SortExec`                                  | [`PhysicalSort`], or [`PhysicalTopK`] with a fetch |
//! | `AggregateExec(Final \| FinalPartitioned) -> ... -> AggregateExec(Partial)`        | [`PhysicalHashAggregate`]                          |
//! | `AggregateExec(Single \| SinglePartitioned)`                                       | [`PhysicalHashAggregate`]                          |
//! | `HashJoinExec` (`CollectLeft` or `Partitioned`)                                    | [`PhysicalHashJoin`]                               |
//! | `WindowAggExec`, `BoundedWindowAggExec` (with the sorts and repartitionings below) | [`PhysicalWindow`]                                 |
//! | `SortPreservingMergeExec` of the partitions of a window                            | [`PhysicalSort`] of the [`PhysicalWindow`]         |
//! | `NestedLoopJoinExec`                                                               | [`PhysicalNestedLoopJoin`]                         |
//! | `CrossJoinExec`                                                                    | [`PhysicalCrossJoin`]                              |
//! | `GlobalLimitExec`, `LocalLimitExec`, `CoalescePartitionsExec` with a fetch         | [`PhysicalLimit`]                                  |
//! | `UnionExec`, `InterleaveExec`                                                      | [`PhysicalUnion`]                                  |
//! | `PlaceholderRowExec`                                                               | [`PhysicalPlaceholderRow`]                         |
//! | `FilterExec`                                                                       | [`PhysicalFilter`] in a [`CachingOperator`]        |
//! | `ProjectionExec`                                                                   | [`PhysicalProjection`]                             |
//! | `ScalarSubqueryExec` (of uncorrelated scalar subqueries)                           | [`PhysicalScalarSubquery`]                         |
//! | `DataSourceExec` (memory, Parquet, CSV), `EmptyExec`                               | the sources of [`try_create_scan_source`]          |
//! | `RepartitionExec`, `CoalescePartitionsExec`, `CoalesceBatchesExec`                 | removed                                            |
//!
//! The nodes that only distribute rows over partitions or change the size of
//! batches are removed: all threads of a pipeline read the chunks of its
//! source directly (scans hand out morsels, the pipeline breakers are shared
//! by all threads and finalize in parallel, and the [`CachingOperator`]s
//! coalesce small chunks), so no rows are shuffled through channels. For the
//! same reason, a two-stage aggregation (`Partial`, the `RepartitionExec` or
//! `CoalescePartitionsExec` that distributes its states, possibly
//! `CoalesceBatchesExec`s and `PartialReduce` stages, and `Final` /
//! `FinalPartitioned`) becomes a single [`PhysicalHashAggregate`], a hash join
//! in either partition mode a single [`PhysicalHashJoin`] (without the
//! repartitioning of its inputs), a sort with the merge of its sorted
//! partitions a single [`PhysicalSort`], and a window with the sorts, merges
//! and repartitionings that partition and sort its input (and the windows
//! stacked below it) a single [`PhysicalWindow`], which partitions and sorts
//! its input itself.
//!
//! Windows, nested loop joins and cross joins keep all of their rows in
//! memory without reserving it in the memory pool, while the DataFusion nodes
//! they replace reserve their memory (and fail or spill once the pool is
//! exhausted): they are only lowered for sessions whose memory pool has no
//! limit, see [`lower_execution_plan_with_runtime`].
//!
//! Every native operator documents (in its `try_lower` or `supports_*`
//! function) which nodes it reproduces exactly, including their errors; any
//! other node, such as a sort-merge join, a recursive query or a custom node,
//! is executed by DataFusion: it is wrapped, with its whole subtree, in a
//! [`BridgeOperator`], so that every plan can be executed. Native operators
//! above a bridged plan read its output as the source of their pipeline.
//!
//! # Plans that DataFusion must execute
//!
//! DataFusion evaluates every node on whole batches before the node above it
//! sees them, a pipeline evaluates all of its operators on a chunk before the
//! next chunk, so which rows are evaluated, and which error comes first, can
//! differ. To keep DataFusion's results and errors exactly, the lowering also
//! leaves these plans to DataFusion:
//!
//! * The plan below a limit that stops the execution early: the input of a
//!   native limit, and the input of a filter with a fetch (unless it is a sort
//!   that reads all of its input, which is lowered natively, with its input).
//!   DataFusion pulls the input of a limit lazily, batch by batch, so the rows
//!   that are evaluated before the limit is reached (and their errors) depend
//!   on its batches (see [`PhysicalFilter::supports_filter_exec`]).
//! * An aggregation or a sort with an expression (a group key, an argument or
//!   a filter of an aggregate, a sort key) whose errors depend on the batches
//!   it is evaluated on (see [`depends_on_batches`]), e.g. the right side of
//!   `a <> 0 AND 10 / a > 1`, which DataFusion evaluates on all rows of a
//!   batch unless at most 20% of them are left, or with several parts that
//!   may fail with different errors (see [`Failures::order_matters`]):
//!   DataFusion evaluates every expression of an operator on a whole batch
//!   before the next one, the native operators on their chunks. (The native
//!   filters and projections refuse such expressions themselves.)
//! * On several threads (a `target_partitions` above 1, or lowering without
//!   a session), a filter, projection, aggregation or sort with a part that
//!   may fail: the threads of a pipeline evaluate the chunks of a partition of
//!   DataFusion, also of a single batch, in parallel, so which of the rows on
//!   which it fails fails first depends on them, while DataFusion fails on
//!   the first of them in its partition (and casts and overflows name the
//!   value that fails). But for the parts that can only fail with overflows,
//!   of decimal arithmetic or of casts of numbers into decimals (see
//!   [`Failures`]), e.g. the arithmetic of TPC-H and the casts with which Q11
//!   compares the sums of its groups with the result of its scalar subquery:
//!   whichever row overflows first, the error is an overflow, as in
//!   DataFusion (see `Lowering::fails_like_datafusion_on_threads`).
//! * A filter, projection, aggregation or sort with a part that may fail
//!   above native operators of the same pipeline that have parts that may fail
//!   too: which of them fails first depends on the batches. For a sort, also
//!   above an input that may fail (expressions, native or not, or sources):
//!   DataFusion evaluates the keys of a sort once it read all of its input,
//!   the native sort whenever it sorts a run of rows of its input. And a sort
//!   of values that address more than their memory (run-end encoded values,
//!   lists of nulls), which DataFusion's sort may fail to concatenate in a few
//!   bytes, and the native sort to interleave, with another error.
//! * A hash join with a build key that may fail: DataFusion evaluates the
//!   build keys once it collected its build side, all keys on a build batch
//!   before the next, the last batch first, the native join on its chunks as
//!   they arrive. And one with a probe key or a filter that may fail (also
//!   without rows, e.g. a cast that their types do not support), or whose
//!   errors depend on the batches: DataFusion evaluates the probe keys on
//!   every probe batch, also on an empty one, the native join on the rows of
//!   its chunks. And one whose output may fail to concatenate, or whose probe
//!   side may fail below a node of the same pipeline that may fail:
//!   DataFusion's join coalesces its output, like `FilterExec`. And one whose
//!   build side (all of its columns) may fail to concatenate, which
//!   DataFusion's join does with its whole build side. And one with
//!   a dynamic filter whose probe side may fail (e.g. decimal arithmetic
//!   above a scan of Parquet files): DataFusion's scans skip the reads that
//!   the filter rejects, which the native join sets to other values.
//! * A TopK whose dynamic filter a node of its input reads (a scan of Parquet
//!   files), through other nodes than those that only pass rows on (scans,
//!   projections, filters, unions, merges and repartitionings of partitions,
//!   without a fetch), or above work that may fail: DataFusion's TopK
//!   tightens the filter, and its scans skip the reads that it rejects, at
//!   other times than the native TopK, so that they read other rows. But for
//!   a TopK whose filter is only read below nodes that read all of their
//!   input before the TopK sees a row (aggregations, sorts and the build
//!   sides of joins, e.g. the aggregation of TPC-H Q18): the filter prunes
//!   nothing in either engine. The native TopK updates the filter, so that the
//!   scans skip the reads that its threshold rules out.
//! * A `FilterExec`, or a node that coalesces batches (a `CoalesceBatchesExec`,
//!   or a `RepartitionExec` over a bounded input, which coalesces the batches
//!   of its output partitions, also below a hash join, which removes the
//!   redistribution of its inputs), whose plan has parts that may fail
//!   (expressions, sources that may fail to read or decode their rows, e.g. a
//!   CSV file with a value that does not parse, or nodes that may fail to
//!   combine their batches, e.g. a repartitioning that merges dictionaries),
//!   below a node of the same pipeline that has a part that may fail too: they
//!   coalesce their output, so they raise the errors of later batches of their
//!   input before the nodes above them see the rows of the earlier ones.
//! * An aggregation that produces a dynamic filter (for `MIN` / `MAX`, which
//!   DataFusion pushes into scans) above nodes that have parts that may fail:
//!   the native aggregation does not update the filter, so these nodes see rows
//!   that DataFusion prunes.
//! * A two-stage aggregation with a stage between its final and its partial
//!   aggregation that does not pass the partial states on unchanged, e.g. a
//!   `PartialReduce` that merges them into other groups (not by their own
//!   group columns, or with other aggregates, filters or a limit), or a
//!   repartitioning of the states by an expression that may fail:
//!   the native aggregation merges all partial states at once, without these
//!   stages. And the sort that DataFusion adds below ordered aggregates (and
//!   that the native aggregation does without) if the aggregates may fail:
//!   DataFusion evaluates them on the sorted rows.
//! * A hash repartitioning by expressions that may fail, which DataFusion
//!   evaluates on every batch, and a node that coalesces batches of types
//!   whose concatenation may fail (e.g. dictionaries, whose merged values may
//!   overflow their keys), also between the stages of an aggregation, or
//!   below a hash join.
//! * A window with a node that it replaces (the window nodes, and the sorts,
//!   merges and repartitionings below them) that evaluates an expression that
//!   may fail or whose errors depend on the batches, or whose batches
//!   DataFusion may fail to combine (of types whose concatenation may fail,
//!   e.g. strings with 32-bit offsets): the native window evaluates its
//!   expressions, and gathers its rows, on other rows together (whole window
//!   partitions, buckets and sorted runs). And a window
//!   above a TopK (through the nodes that it reads all of its input through)
//!   whose input is sorted by a prefix of its keys: DataFusion's TopK ends
//!   such a partition as soon as no later row can make it into the result,
//!   so that a partition of DataFusion's window can emit its rows while the
//!   input of another one is still pending (or fails later), while the native
//!   window reads all of its input first.
//! * A nested loop or cross join below a node of the same pipeline that may
//!   fail: DataFusion joins a probe batch with blocks of build rows, the native
//!   join a chunk of it, so that the rows above come in another order (and
//!   DataFusion's nested loop join coalesces its output, pulling later probe
//!   batches before it emits the rows of earlier ones). And one with types
//!   whose concatenation may fail (e.g. strings with 32-bit offsets) in an input
//!   or in its output: DataFusion's cross join concatenates its build side and
//!   repeats every build row for a whole probe batch, its nested loop join
//!   coalesces its output, and the native joins gather chunks of other rows.
//!   And a nested loop join whose filter may fail or has errors that depend on
//!   the batches: DataFusion evaluates it on other blocks of pairs.
//!
//! Which other nodes a native operator reproduces exactly, including their
//! errors, is up to the operator (see its `try_lower` or `supports_*`
//! function).
//!
//! The pipeline breakers that read all of their input before they produce any
//! output (sorts, aggregations, windows and the build side of joins) separate
//! these concerns: nothing above them runs before their input is exhausted.
//!
//! # `EXPLAIN` and `EXPLAIN ANALYZE`
//!
//! `EXPLAIN` shows the plan that the pipeline engine executes: the
//! `physical_plan` is the [`PipelineRootExec`], with the lowered operator tree
//! and its pipelines (see [`crate::explain`]). `EXPLAIN VERBOSE` also shows the
//! physical plans of DataFusion that it was lowered from. `EXPLAIN ANALYZE`
//! executes the lowered plan and shows the metrics of every operator.
//!
//! # Example
//!
//! ```
//! # use std::sync::Arc;
//! use datafusion::execution::SessionStateBuilder;
//! use datafusion::physical_planner::DefaultPhysicalPlanner;
//! use datafusion::prelude::*;
//! use datafusion_physical_pipeline::PipelineQueryPlanner;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> datafusion::error::Result<()> {
//! let planner = PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
//! let state = SessionStateBuilder::new()
//!     .with_default_features()
//!     .with_query_planner(Arc::new(planner))
//!     .build();
//! let ctx = SessionContext::new_with_state(state);
//! let batches = ctx.sql("SELECT 1 AS one").await?.collect().await?;
//! assert_eq!(batches[0].num_rows(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! [`PhysicalTopK`]: crate::sort::PhysicalTopK

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_common::config::ConfigOptions;
use datafusion_common::format::ExplainFormat;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, ScalarValue, exec_err, internal_err};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_csv::source::CsvSource;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::LogicalPlan;
use datafusion_expr::logical_plan::Explain;
use datafusion_expr::physical_planning_context::{ScalarSubqueryResults, SubqueryIndex};
use datafusion_physical_expr::expressions::{
    Column, DynamicFilterPhysicalExpr, IsNotNullExpr,
};
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, conjunction};
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateInputMode, AggregateMode, GroupClusteringMode,
};
use datafusion_physical_plan::analyze::AnalyzeExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::explain::ExplainExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec};
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::scalar_subquery::ScalarSubqueryExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PhysicalExpr, ReplaceChildrenOptions,
};
use datafusion_session::{PhysicalPlanner, QueryPlanner, Session};

use crate::aggregate::PhysicalHashAggregate;
use crate::bridge::{BridgeOperator, PipelineRootExec};
use crate::chunk::DataChunk;
use crate::explain::{AnalyzedPipelineExec, MetricSelection};
use crate::join::PhysicalHashJoin;
use crate::join::nested_loop::{PhysicalCrossJoin, PhysicalNestedLoopJoin};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, LocalSinkState,
    OperatorResult, OperatorState, PhysicalOperator, SinkFinalize, SinkResult,
    downcast_state,
};
use crate::operators::adaptive_filter::{Failures, depends_on_batches, failures};
use crate::operators::caching::CachingOperator;
use crate::operators::filter::PhysicalFilter;
use crate::operators::limit::PhysicalLimit;
use crate::operators::placeholder_row::PhysicalPlaceholderRow;
use crate::operators::projection::PhysicalProjection;
use crate::operators::union::PhysicalUnion;
use crate::pipeline::{MetaPipelineBuilder, MetaPipelineId, PipelineBuilder};
use crate::sort::{PhysicalSort, PhysicalTopK, TopKDynamicFilter};
use crate::sources::try_create_scan_source;
use crate::window::PhysicalWindow;

/// A [`QueryPlanner`] that executes queries with the pipeline engine.
///
/// See the [module documentation](self) for details.
pub struct PipelineQueryPlanner {
    physical_planner: Arc<dyn PhysicalPlanner>,
}

impl PipelineQueryPlanner {
    /// Create a planner that lowers the plans created by `physical_planner`
    /// (usually `datafusion::physical_planner::DefaultPhysicalPlanner`).
    pub fn new(physical_planner: Arc<dyn PhysicalPlanner>) -> Self {
        Self { physical_planner }
    }

    /// The DataFusion physical planner whose plans are lowered.
    pub fn physical_planner(&self) -> &Arc<dyn PhysicalPlanner> {
        &self.physical_planner
    }

    /// The plan of `EXPLAIN`: DataFusion's `ExplainExec`, whose final
    /// physical plan is the plan that the pipeline engine executes.
    async fn create_explain_plan(
        &self,
        logical_plan: &LogicalPlan,
        explain: &Explain,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = self
            .physical_planner
            .create_physical_plan(logical_plan, session)
            .await?;
        let options = session.config_options();
        let shows_physical_plan = explain.logical_optimization_succeeded
            && !options.explain.logical_plan_only
            && matches!(
                explain.explain_format,
                ExplainFormat::Indent | ExplainFormat::Tree
            );
        let Some(exec) = plan
            .downcast_ref::<ExplainExec>()
            .filter(|_| shows_physical_plan)
        else {
            return Ok(plan);
        };
        // DataFusion's `ExplainExec` reports the errors of physical planning.
        let Ok(input) = self
            .physical_planner
            .create_physical_plan(&explain.plan, session)
            .await
        else {
            return Ok(plan);
        };
        let root =
            create_pipeline_plan_with_runtime(input, options, session.runtime_env())?;
        Ok(Arc::new(crate::explain::explain_pipeline_plan(
            exec, &root, explain, session,
        )))
    }

    /// The plan of `EXPLAIN ANALYZE`: DataFusion's `AnalyzeExec`, executing
    /// (and showing the metrics of) the plan of the pipeline engine, with the
    /// metrics that the statement selects (see [`AnalyzedPipelineExec`]).
    async fn create_analyze_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = self
            .physical_planner
            .create_physical_plan(logical_plan, session)
            .await?;
        let Some(analyze) = plan.downcast_ref::<AnalyzeExec>() else {
            return Ok(plan);
        };
        let root = create_pipeline_plan_with_runtime(
            Arc::clone(analyze.input()),
            session.config_options(),
            session.runtime_env(),
        )?;
        // Shown with the metrics of its operators that the statement selects.
        let input: Arc<dyn ExecutionPlan> = match logical_plan {
            LogicalPlan::Analyze(statement) => Arc::new(AnalyzedPipelineExec::new(
                root,
                MetricSelection::of_analyze(statement, session.config_options()),
            )),
            _ => root,
        };
        Arc::clone(&plan).replace_children(vec![input], recompute())
    }
}

impl fmt::Debug for PipelineQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineQueryPlanner")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl QueryPlanner for PipelineQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = match logical_plan {
            LogicalPlan::Explain(explain) => {
                self.create_explain_plan(logical_plan, explain, session)
                    .await?
            }
            LogicalPlan::Analyze(_) => {
                self.create_analyze_plan(logical_plan, session).await?
            }
            _ => {
                self.physical_planner
                    .create_physical_plan(logical_plan, session)
                    .await?
            }
        };
        let plan: Arc<dyn ExecutionPlan> = create_pipeline_plan_with_runtime(
            plan,
            session.config_options(),
            session.runtime_env(),
        )?;
        Ok(plan)
    }
}

/// Lower `plan` and wrap it into a [`PipelineRootExec`], for any session
/// (see [`lower_execution_plan`]).
pub fn create_pipeline_plan(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<PipelineRootExec>> {
    let root = lower_execution_plan(&plan)?;
    Ok(Arc::new(PipelineRootExec::try_new(root, plan)?))
}

/// Lower `plan`, to be executed in a session with the options `options`, and
/// wrap it into a [`PipelineRootExec`] (see
/// [`lower_execution_plan_with_options`]).
pub fn create_pipeline_plan_with_options(
    plan: Arc<dyn ExecutionPlan>,
    options: &ConfigOptions,
) -> Result<Arc<PipelineRootExec>> {
    create_root(plan, &Lowering::new(options, None))
}

/// Lower `plan`, to be executed in a session with the options `options` and
/// the runtime `runtime`, and wrap it into a [`PipelineRootExec`] (see
/// [`lower_execution_plan_with_runtime`]). This is what the
/// [`PipelineQueryPlanner`] does, with the options and the runtime of its
/// session.
pub fn create_pipeline_plan_with_runtime(
    plan: Arc<dyn ExecutionPlan>,
    options: &ConfigOptions,
    runtime: &RuntimeEnv,
) -> Result<Arc<PipelineRootExec>> {
    create_root(plan, &Lowering::new(options, Some(runtime)))
}

/// Lower `plan` with `lowering` and wrap it into a [`PipelineRootExec`],
/// executing it with DataFusion if the lowered plan does not fit.
fn create_root(
    plan: Arc<dyn ExecutionPlan>,
    lowering: &Lowering<'_>,
) -> Result<Arc<PipelineRootExec>> {
    let root = lowering.lower(&plan, Context::default())?;
    let root = match PipelineRootExec::try_new(root, Arc::clone(&plan)) {
        Ok(root) => root,
        Err(error) => {
            log::warn!(
                "Executing the plan with DataFusion, its lowering failed: {error}"
            );
            PipelineRootExec::try_new(bridge(&plan), plan)?
        }
    };
    Ok(Arc::new(root))
}

/// Lower a DataFusion [`ExecutionPlan`] tree into a [`PhysicalOperator`]
/// tree that can be executed in any session.
///
/// Whether some nodes can be lowered depends on the options of the session
/// that executes them (e.g. a sort of strings, if DataFusion concatenates
/// more than 2 GiB of them, see [`crate::sort::try_lower`]), and on its
/// runtime (windows and loop joins, if its memory pool has no limit, see
/// [`lower_execution_plan_with_runtime`]): this lowers them as if the session
/// had the options and the runtime for which the fewest nodes can be lowered.
/// The planner lowers with the options and the runtime of its session, see
/// [`lower_execution_plan_with_runtime`].
pub fn lower_execution_plan(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn PhysicalOperator>> {
    lower_execution_plan_with_options(plan, &any_session_options())
}

/// Lower a DataFusion [`ExecutionPlan`] tree, to be executed in a session
/// with the options `options`, into a [`PhysicalOperator`] tree.
///
/// Every node (or chain of nodes) that a native operator reproduces exactly
/// is replaced by that operator, and every other node is executed by a
/// [`BridgeOperator`], together with its whole subtree. See the
/// [module documentation](self) for the native operators and the plans that
/// are left to DataFusion.
///
/// A node whose native operator fails to be created (the native operators
/// report the nodes they do not reproduce without an error, so this is not
/// expected) is executed by DataFusion too, with a warning: the lowering
/// never fails, as DataFusion can execute every plan it created.
///
/// The runtime of the session is not known: windows, nested loop joins and
/// cross joins, which are only lowered if its memory pool has no limit, are
/// left to DataFusion, see [`lower_execution_plan_with_runtime`].
pub fn lower_execution_plan_with_options(
    plan: &Arc<dyn ExecutionPlan>,
    options: &ConfigOptions,
) -> Result<Arc<dyn PhysicalOperator>> {
    Lowering::new(options, None).lower(plan, Context::default())
}

/// Lower a DataFusion [`ExecutionPlan`] tree, to be executed in a session
/// with the options `options` and the runtime `runtime`, into a
/// [`PhysicalOperator`] tree (see [`lower_execution_plan_with_options`]).
///
/// The native windows ([`PhysicalWindow`]), nested loop joins
/// ([`PhysicalNestedLoopJoin`]) and cross joins ([`PhysicalCrossJoin`])
/// keep their rows in memory without reserving it in the memory pool, while
/// the DataFusion nodes they replace (the sorts below a window, the build
/// sides of the joins) reserve memory as their input produces it, and fail or
/// spill once the pool is exhausted: they are only lowered if the memory pool
/// of `runtime` has no limit, in which this makes no difference.
pub fn lower_execution_plan_with_runtime(
    plan: &Arc<dyn ExecutionPlan>,
    options: &ConfigOptions,
    runtime: &RuntimeEnv,
) -> Result<Arc<dyn PhysicalOperator>> {
    Lowering::new(options, Some(runtime)).lower(plan, Context::default())
}

/// Options for which the lowering lowers what it lowers for every session:
/// DataFusion's sort concatenates the batches of a partition, which can
/// overflow 32-bit offsets, while they occupy fewer bytes than
/// `sort_in_place_threshold_bytes` (see [`crate::sort::try_lower`]), and the
/// pipelines may run on several threads (see
/// `Lowering::evaluates_in_order`).
fn any_session_options() -> ConfigOptions {
    let mut options = ConfigOptions::default();
    options.execution.sort_in_place_threshold_bytes = usize::MAX;
    options.execution.target_partitions = usize::MAX;
    options
}

/// Where in the plan a node is lowered.
#[derive(Debug, Clone, Copy, Default)]
struct Context {
    /// The node is below a limit that stops the execution early: it is left
    /// to DataFusion, unless it is a sort that reads all of its input.
    limited: bool,
    /// The parts that may fail of the nodes above the node, before the next
    /// pipeline breaker that reads all of its input (see [`failures`]).
    fallible_above: Failures,
}

impl Context {
    /// The context of the input of a pipeline breaker that reads all of its
    /// input before it produces any output, and whose own expressions have
    /// the parts that may fail `failures`.
    fn below_breaker(failures: Failures) -> Self {
        Self {
            limited: false,
            fallible_above: failures,
        }
    }
}

/// Lowers the nodes of a plan, see [`lower_execution_plan_with_options`] and
/// [`lower_execution_plan_with_runtime`].
struct Lowering<'a> {
    options: &'a ConfigOptions,
    /// The runtime of the session that executes the plan, if known: windows
    /// and loop joins are only lowered for runtimes whose memory pool has no
    /// limit.
    runtime: Option<&'a RuntimeEnv>,
}

impl<'a> Lowering<'a> {
    fn new(options: &'a ConfigOptions, runtime: Option<&'a RuntimeEnv>) -> Self {
        Self { options, runtime }
    }
}

impl Lowering<'_> {
    /// Whether the native operators evaluate the rows of every partition of
    /// DataFusion in order, as DataFusion does: if the pipelines run on a
    /// single thread. The executor runs a pipeline on up to
    /// `target_partitions` threads, which evaluate the chunks of a partition,
    /// and of a batch, in parallel: then which of the rows on which an
    /// expression fails fails first depends on the threads, while DataFusion
    /// fails on the first of them in its partition (and casts name the value
    /// that fails).
    fn evaluates_in_order(&self) -> bool {
        self.options.execution.target_partitions <= 1
    }

    /// Whether a native operator whose expressions have the parts that may
    /// fail `failures` (see [`failures`]) fails like DataFusion on the
    /// threads that run its pipelines: if they run on a single thread, which
    /// evaluates the rows in order (see [`Self::evaluates_in_order`]), or if
    /// these parts can only fail with overflows (see
    /// [`Failures::only_overflows`]): whichever row overflows first on
    /// whichever thread, the error is an overflow, as in DataFusion (although
    /// the values it names may be those of another row). These are the
    /// arithmetic overflows of decimals (e.g. of the decimal arithmetic of
    /// TPC-H, `l_extendedprice * (1 - l_discount)`), which raise an
    /// `ArrowError::ArithmeticOverflow`, and the overflows of the values of
    /// the casts of numbers into decimals (e.g. of the casts with which TPC-H
    /// Q11 compares the sums of its groups with the result of its scalar
    /// subquery, `CAST(sum(...) AS Decimal128(38, 15))`), which Arrow reports
    /// as the cast errors or invalid arguments of the values that overflow the
    /// decimal (see [`Failures`]).
    fn fails_like_datafusion_on_threads(&self, failures: Failures) -> bool {
        failures.only_overflows() || self.evaluates_in_order()
    }

    /// Lower `plan`, falling back to DataFusion if a native operator cannot
    /// be created (see [`lower_execution_plan_with_options`]). Never fails;
    /// returns a `Result` for the `try_lower` functions of the operators.
    fn lower(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        ctx: Context,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        self.lower_natively(plan, ctx).or_else(|error| {
            log::warn!(
                "Executing {} with DataFusion, its native lowering failed: {error}",
                plan.name()
            );
            Ok(bridge(plan))
        })
    }

    fn lower_natively(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        ctx: Context,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        // A native sort reads all of its input, also below a limit, and so
        // does a native window, where DataFusion's window does too.
        if let Some(sort) = self.try_lower_sort(plan)? {
            return Ok(sort);
        }
        if let Some(window) = self.try_lower_window(plan)? {
            return Ok(window);
        }
        if let Some(sort) = self.try_lower_merge_of_window(plan)? {
            return Ok(sort);
        }
        if ctx.limited {
            return Ok(bridge(plan));
        }
        if let Some(subquery) = self.try_lower_scalar_subquery(plan, ctx)? {
            return Ok(subquery);
        }
        if let Some(aggregate) = self.try_lower_aggregate(plan)? {
            return Ok(aggregate);
        }
        if let Some(join) = self.try_lower_hash_join(plan, ctx)? {
            return Ok(join);
        }
        if let Some(join) = self.try_lower_loop_join(plan, ctx)? {
            return Ok(join);
        }
        let limited = Context {
            limited: true,
            ..ctx
        };
        if let Some(limit) =
            PhysicalLimit::try_lower(plan, |input| self.lower(input, limited))?
        {
            return Ok(Arc::new(limit));
        }
        if let Some(union) =
            PhysicalUnion::try_lower(plan, |input| self.lower(input, ctx))?
        {
            return Ok(Arc::new(union));
        }
        if let Some(placeholder) = PhysicalPlaceholderRow::try_lower(plan)? {
            return Ok(Arc::new(placeholder));
        }
        if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            return self.lower_filter(plan, filter, ctx);
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            return self.lower_projection(plan, projection, ctx);
        }
        if let Some(scan) = try_create_scan_source(plan)? {
            return Ok(scan);
        }
        if let Some(input) = redistributed_input(plan) {
            // Like `FilterExec` (see `lower_filter`), the nodes that coalesce
            // batches pull later batches of their input, whose errors they
            // raise, before the nodes above them see the rows of earlier ones.
            if coalesces_batches(plan)
                && conflict(ctx.fallible_above, subtree_failures(input, self.runtime))
            {
                return Ok(bridge(plan));
            }
            let lowered = self.lower(input, ctx)?;
            // DataFusion executes a bridged input anyway: keep a
            // repartitioning that spreads it over more partitions, which the
            // threads of the pipeline read in parallel.
            if lowered.is::<BridgeOperator>() && spreads_partitions(plan) {
                return Ok(bridge(plan));
            }
            return Ok(lowered);
        }
        Ok(bridge(plan))
    }

    /// Lower a `ScalarSubqueryExec` to a [`PhysicalScalarSubquery`], which
    /// runs its subqueries, natively too, before its input: see
    /// [`PhysicalScalarSubquery`]. The input is lowered in the context of the
    /// `ScalarSubqueryExec`, which passes its rows on.
    ///
    /// DataFusion reads the batches of a subquery until it read more than one
    /// row (and fails), like a limit: a subquery that may produce more than one
    /// row (see [`produces_at_most_one_row`]) is lowered below a limit that
    /// stops the execution early, so that it is left to DataFusion (but for the
    /// sorts that read all of their input). The usual subqueries, aggregations
    /// without groups (e.g. those of TPC-H Q11, Q15 and Q22), produce exactly
    /// one row and are lowered natively. A `ScalarSubqueryExec` with a
    /// subquery of other than one column, on which DataFusion fails, is left
    /// to DataFusion.
    fn try_lower_scalar_subquery(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        ctx: Context,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let Some(exec) = plan.downcast_ref::<ScalarSubqueryExec>() else {
            return Ok(None);
        };
        if exec
            .subqueries()
            .iter()
            .any(|link| link.plan.schema().fields().len() != 1)
        {
            return Ok(None);
        }
        let input = self.lower(exec.input(), ctx)?;
        let subqueries = exec
            .subqueries()
            .iter()
            .map(|link| {
                let subquery = self.lower(
                    &link.plan,
                    Context {
                        limited: !produces_at_most_one_row(&link.plan),
                        fallible_above: Failures::NONE,
                    },
                )?;
                Ok(Arc::new(ScalarSubqueryResult::try_new(
                    subquery,
                    input.schema(),
                    link.index,
                    exec.results().clone(),
                )?))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Arc::new(PhysicalScalarSubquery::try_new(
            input, subqueries,
        )?)))
    }

    /// Lower a sort (a `SortPreservingMergeExec -> SortExec` chain, or a
    /// `SortExec`) with [`crate::sort::try_lower`].
    ///
    /// DataFusion evaluates the keys of a sort on its batches, the native sort
    /// on its chunks: a sort whose keys have errors that depend on the batches
    /// they are evaluated on (see [`depends_on_batches`]), e.g. the right side
    /// of `a <> 0 AND 10 / a > 1`, which DataFusion evaluates on all rows of a
    /// batch unless at most 20% of them are left, or several parts that may
    /// fail with different errors (which error comes first depends on the
    /// batches, see [`Failures::order_matters`]) is left to DataFusion. So is
    /// a sort with a key that may fail over an input that may fail with
    /// another error (its expressions, native or not, or its sources, see
    /// [`conflict`]): DataFusion evaluates the keys once it read all of its
    /// input, the native sort whenever it sorts a run of its input, before it
    /// read all of it.
    ///
    /// A TopK whose dynamic filter a node of its input reads (a scan of
    /// Parquet files, which skips the reads that the filter rejects) is lowered
    /// if the filter prunes nothing in either engine, or only rows on which the
    /// work of its input cannot fail (see [`TopKFilterPruning`]), so that the
    /// scans read other rows in the two engines without making a difference:
    /// the native TopK then updates the filter, as DataFusion's TopK does.
    fn try_lower_sort(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let Some(sort) = sort_of_chain(plan) else {
            return Ok(None);
        };
        let keys = Judgement::of_ordering(sort.expr(), &sort.input().schema());
        if !keys.evaluates_like_datafusion() {
            return Ok(None);
        }
        // DataFusion's sort concatenates the batches of a partition while they
        // occupy fewer bytes than `sort_in_place_threshold_bytes`, which the
        // native sort accounts for the values whose concatenations their
        // memory bounds (see `crate::sort::try_lower`). Values that address
        // more than their memory may fail to concatenate in a few bytes (see
        // `addresses_more_than_their_memory`), and the native sort, which
        // interleaves them, fails otherwise: such sorts are left to DataFusion.
        if sort
            .input()
            .schema()
            .fields()
            .iter()
            .any(|field| addresses_more_than_their_memory(field.data_type()))
        {
            return Ok(None);
        }
        let fails_like_datafusion = |keys: &Judgement| {
            self.fails_like_datafusion_on_threads(keys.failures)
                && !conflict(keys.failures, subtree_failures(sort.input(), self.runtime))
        };
        if !fails_like_datafusion(&keys) {
            return Ok(None);
        }
        // DataFusion pushes the dynamic filter of a TopK (its threshold) into
        // the scans of its input (e.g. of Parquet files), which skip the files,
        // row groups and rows that it rejects as it tightens; a native TopK
        // publishes its threshold at other times, so that the scans skip other
        // reads. That makes no difference to the rows of the TopK, which only
        // skips rows that cannot make it into its result in both engines, but
        // it does to the work of the nodes that read the pruned rows, which
        // may fail on rows that the other engine skips. So a TopK whose
        // dynamic filter a node of its input reads is lowered natively only
        // (see [`topk_filter_pruning`]) if the filter prunes nothing in either
        // engine (every node that reads it is below a node that reads all of
        // its input before the TopK sees a row, e.g. the aggregation of TPC-H
        // Q18 below its TopK), or if the nodes between the scans that read it
        // and the TopK only pass rows on, and the work of its input cannot
        // fail on any rows (see [`works_alike_on_any_rows`]: e.g. the `LIKE`
        // of ClickBench's `hits23` above its scan). The native TopK then
        // updates the filter, so that the scans prune with its threshold.
        let read_filter = read_topk_filter(sort);
        let to_lower = match &read_filter {
            None => Arc::clone(plan),
            Some((_, id)) => {
                let exact = match topk_filter_pruning(sort.input(), *id) {
                    TopKFilterPruning::Unread | TopKFilterPruning::Settled => true,
                    TopKFilterPruning::Rows => {
                        works_alike_on_any_rows(sort.input(), self.runtime)
                    }
                    TopKFilterPruning::Other => false,
                };
                if !exact {
                    return Ok(None);
                }
                // `crate::sort::try_lower` judges the work on the rows that the
                // filter prunes by the expressions that it knows cannot fail,
                // and leaves the TopKs above other nodes than those that pass
                // rows on to DataFusion: it lowers the TopK of a plan whose
                // `SortExec` has a filter that no node reads, and the native
                // TopK updates the filter of the plan instead.
                with_unread_topk_filter(plan)?
            }
        };
        let input = Context::below_breaker(keys.failures);
        let Some(lowered) = crate::sort::try_lower(&to_lower, self.options, |plan| {
            self.lower(plan, input)
        })?
        else {
            return Ok(None);
        };
        let lowered = match read_filter {
            None => lowered,
            Some((filter, _)) => match updating_filter(&lowered, filter)? {
                Some(lowered) => lowered,
                None => return Ok(None),
            },
        };
        // The keys that the native sort evaluates (those of the top merge, over
        // the projections below it, if any).
        let native_keys = native_sort_keys(&lowered);
        if !native_keys.evaluates_like_datafusion()
            || !fails_like_datafusion(&native_keys)
        {
            return Ok(None);
        }
        Ok(Some(lowered))
    }

    /// Lower an aggregation with [`PhysicalHashAggregate::try_lower`], after
    /// removing the `CoalesceBatchesExec`s and `PartialReduce` stages between
    /// its final and its partial `AggregateExec`, and the sort that only orders
    /// its input for its ordered aggregates (see [`without_ordering_sort`]).
    ///
    /// DataFusion evaluates the group keys, the arguments and the filters of
    /// the aggregates on its batches, one after the other, the native
    /// aggregation on its chunks: an aggregation with an expression whose
    /// errors depend on the batches (see [`depends_on_batches`]) or with
    /// several parts that may fail with different errors (see
    /// [`Failures::order_matters`]) is left to DataFusion, as is an aggregation
    /// with a part that may fail above native operators of the same pipeline
    /// that may fail with another error (see [`conflict`]), and an aggregation
    /// that produces a dynamic filter (which the native aggregation does not
    /// update) above nodes that may fail on the rows that DataFusion would
    /// prune with it (see [`subtree_failures`]). Parts that can only fail
    /// with arithmetic overflows (e.g. the decimal arithmetic of the
    /// aggregates of TPC-H) are evaluated on any number of threads (see
    /// `Lowering::fails_like_datafusion_on_threads`).
    fn try_lower_aggregate(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        if !plan.is::<AggregateExec>() {
            return Ok(None);
        }
        let Some(plan) = aggregation_to_lower(plan)? else {
            return Ok(None);
        };
        let plan = &plan;
        let Some(raw) = raw_aggregate(plan) else {
            return Ok(None);
        };
        let judgement = Judgement::of_plan(raw);
        if !judgement.evaluates_like_datafusion() {
            return Ok(None);
        }
        if !raw.dynamic_expressions_produced().is_empty()
            && !subtree_failures(raw.input(), self.runtime).is_none()
        {
            return Ok(None);
        }
        let own = judgement.failures;
        if !self.fails_like_datafusion_on_threads(own) {
            return Ok(None);
        }
        let input = Context::below_breaker(own);
        let Some(aggregate) = PhysicalHashAggregate::try_lower(plan, |plan| {
            self.lower(without_ordering_sort(raw, own, plan), input)
        })?
        else {
            return Ok(None);
        };
        if conflict(own, native_failures(aggregate.input())) {
            return Ok(None);
        }
        Ok(Some(Arc::new(aggregate)))
    }

    /// Lower a hash join with [`PhysicalHashJoin::try_lower`].
    ///
    /// The join reads all of its build side before it probes, as DataFusion
    /// does: its build side is lowered like the input of a pipeline breaker.
    /// Its probe side continues the pipeline of the nodes above the join,
    /// which evaluates its probe keys and its filter on the chunks of the
    /// pipeline (see [`native_failures`]).
    ///
    /// A join with a build key that may fail is left to DataFusion, which
    /// evaluates the build keys once it collected its build side: all keys on
    /// a build batch before the next, the last batch first. A native join
    /// evaluates them on its chunks as they arrive, so that another error
    /// could come first: of another key, of another row, or of a key whose
    /// errors depend on the batches (see [`depends_on_batches`]).
    fn try_lower_hash_join(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        ctx: Context,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
            return Ok(None);
        };
        let build_schema = join.left().schema();
        let mut build_keys = Judgement::default();
        for (key, _) in join.on() {
            build_keys.add(key, &build_schema);
        }
        if build_keys.batch_dependent || !build_keys.failures.is_none() {
            return Ok(None);
        }
        // DataFusion evaluates the probe keys on every probe batch, also on an
        // empty one, and the filter on the candidate pairs of a probe batch, a
        // native join on the rows of its chunks: probe keys and a filter whose
        // errors depend on the batches (e.g. a cast that their types do not
        // support, which fails without rows), or whose parts may fail with
        // different errors, are left to DataFusion, as are, on several
        // threads, those that may fail with other errors than arithmetic
        // overflows (see `Lowering::fails_like_datafusion_on_threads`). (The
        // native join refuses some of them itself, see
        // [`PhysicalHashJoin::try_lower`].)
        let probe_schema = join.right().schema();
        let mut probe = Judgement::default();
        for (_, key) in join.on() {
            probe.add(key, &probe_schema);
        }
        if let Some(filter) = join.filter() {
            probe.add(filter.expression(), filter.schema());
        }
        if !probe.evaluates_like_datafusion()
            || !self.fails_like_datafusion_on_threads(probe.failures)
        {
            return Ok(None);
        }
        // DataFusion's join coalesces its output, up to `batch_size` rows,
        // before it emits it; a native join passes its output on at once. So
        // DataFusion may fail to concatenate output types whose concatenation
        // may fail, and it pulls later probe batches, whose errors it raises,
        // before the nodes above it see the rows of the earlier ones: such
        // joins are left to DataFusion, as are those whose probe side (which
        // continues the pipeline of the nodes above the join) or probe keys and
        // filter may fail below a node above that may fail with another error
        // (see [`conflict`]).
        if concatenation_may_fail(&join.schema())
            || conflict(
                ctx.fallible_above,
                subtree_failures(join.right(), self.runtime).and(probe.failures),
            )
        {
            return Ok(None);
        }
        // DataFusion concatenates its whole build side, all of its columns
        // (also those that the output of the join does not have), which may
        // fail for their types (e.g. run ends that 32 bits do not hold, also in
        // a list): such joins are left to DataFusion, as the native join does
        // not concatenate its build side.
        if concatenation_may_fail(&join.left().schema()) {
            return Ok(None);
        }
        // DataFusion pushes the dynamic filter of the join into the scans of
        // its probe side, which skip the files, row groups and rows that the
        // filter rejects once the build side is collected; a native join does
        // not skip the same reads. So a join with a dynamic filter whose probe
        // side may fail on the rows that DataFusion skips is left to
        // DataFusion (see [`subtree_failures`]: e.g. a probe side with an
        // expression that may fail, or a source that may fail to read its
        // rows, but for scans of well-formed Parquet files whose expressions
        // cannot fail, as in TPC-H, see [`source_may_fail`]).
        if !plan.dynamic_expressions_produced().is_empty()
            && !subtree_failures(join.right(), self.runtime).is_none()
        {
            return Ok(None);
        }
        // The join removes the redistributions of its inputs: a repartitioning
        // that coalesces batches (see `coalesces_batches`) of types whose
        // concatenation may fail is left to DataFusion, with the join.
        for side in [join.left(), join.right()] {
            if top_repartitions(side).into_iter().any(|repartition| {
                coalesces_batches(repartition)
                    && concatenation_may_fail(&repartition.schema())
            }) {
                return Ok(None);
            }
        }
        let build = Context::below_breaker(Failures::NONE);
        let probe_failures = probe.failures;
        let probe = Context {
            limited: false,
            fallible_above: ctx.fallible_above.and(probe_failures),
        };
        let lowered = PhysicalHashJoin::try_lower(plan, |input| {
            // The join lowers its inputs without the nodes that distribute
            // their rows over partitions.
            let side = if redistributes(join.left(), input)
                && !redistributes(join.right(), input)
            {
                build
            } else {
                probe
            };
            self.lower(input, side)
        })?;
        // The native operators below the join in its probe pipeline evaluate
        // the rows of a chunk before its probe keys and its filter do.
        Ok(lowered
            .filter(|join| !conflict(probe_failures, native_failures(join.probe_side())))
            .map(|join| Arc::new(join) as Arc<dyn PhysicalOperator>))
    }

    /// Lower a window (a `WindowAggExec` or a `BoundedWindowAggExec`, with the
    /// nodes below it that partition and sort its input, and the windows
    /// stacked below it) with [`PhysicalWindow::try_lower`], for a runtime
    /// whose memory pool has no limit (see
    /// [`lower_execution_plan_with_runtime`]).
    ///
    /// The native window reads all of its input before it emits anything, and
    /// is only lowered where DataFusion's window does so too (see
    /// [`PhysicalWindow::try_lower`]), also below a limit: its input is
    /// lowered like the input of a sort, as the input of a pipeline breaker,
    /// and its window functions, partition keys and order keys cannot fail.
    ///
    /// The native window replaces the window nodes and the sorts, merges and
    /// repartitionings below them (see [`window_nodes`]), and evaluates their
    /// expressions, and gathers their rows, on other rows together than
    /// DataFusion (whole window partitions, buckets and sorted runs, rather
    /// than batches): a window with such a node that evaluates an expression
    /// that may fail or whose errors depend on the batches, or that DataFusion
    /// may fail to combine the batches of (see [`may_fail_to_combine`], e.g. a
    /// repartitioning that coalesces strings with 32-bit offsets), is left to
    /// DataFusion. So is a window above a TopK that may end some partitions
    /// early (see [`topk_may_end_early_below`]).
    ///
    /// DataFusion plans a projection of columns between the sort below a
    /// window (or the merge of its sorted partitions) and the input of the
    /// sort, e.g. for a window over an aggregation whose columns it renames,
    /// which [`PhysicalWindow::try_lower`] does not look through: such a window
    /// is lowered as if the projection were below the sort, see
    /// [`with_projection_below_sort`].
    fn try_lower_window(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        if !is_window(plan) {
            return Ok(None);
        }
        let Some(runtime) = self.runtime else {
            return Ok(None);
        };
        let rewritten = with_projection_below_sort(plan)?;
        for plan in std::iter::once(plan).chain(rewritten.as_ref()) {
            let replaced_nodes_may_differ = window_nodes(plan).into_iter().any(|node| {
                let judgement = Judgement::of_plan(node.as_ref());
                judgement.batch_dependent
                    || !judgement.failures.is_none()
                    || may_fail_to_combine(node)
            });
            if replaced_nodes_may_differ || topk_may_end_early_below(plan) {
                continue;
            }
            let input = Context::below_breaker(Failures::NONE);
            if let Some(window) =
                PhysicalWindow::try_lower(plan, runtime, |rows| self.lower(rows, input))?
            {
                return Ok(Some(Arc::new(window)));
            }
        }
        Ok(None)
    }

    /// Lower a `SortPreservingMergeExec` of the sorted partitions of a window
    /// (e.g. of `SELECT ..., ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) FROM
    /// t ORDER BY a, b`), possibly through projections of columns, to a
    /// [`PhysicalSort`] by the keys of the merge (with its fetch) of the native
    /// window: merging the sorted partitions of the window is sorting its rows.
    ///
    /// Only over a window that is lowered to a native window, which DataFusion
    /// lowers only where its window reads all of its input before it emits any
    /// row (see [`PhysicalWindow::try_lower`]): by the time the merge emits a
    /// row, DataFusion read all rows below the window, as the native sort does,
    /// and both evaluate only infallible window functions after that (the keys
    /// of the merge must be columns, which evaluate nothing). So the merge is
    /// lowered also below a limit, which stops DataFusion's merge early, but
    /// for a merge of no rows (`fetch == Some(0)`), which DataFusion ends
    /// without reading its input at all.
    fn try_lower_merge_of_window(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() else {
            return Ok(None);
        };
        // A merge of no rows (`fetch == Some(0)`) emits nothing without reading
        // its input, which the native sort reads all of (and the native window
        // below it, with the errors of its input).
        if self.runtime.is_none()
            || merge.fetch() == Some(0)
            || merge.input().output_partitioning().partition_count() == 0
        {
            return Ok(None);
        }
        let mut node = merge.input();
        while !is_window(node) {
            match node.downcast_ref::<ProjectionExec>() {
                Some(projection)
                    if projection
                        .expr()
                        .iter()
                        .all(|expr| expr.expr.is::<Column>()) =>
                {
                    node = projection.input();
                }
                _ => return Ok(None),
            }
        }
        // The merge produces all rows in its order only if every partition of
        // its input is ordered by it. Its keys are columns (which evaluate
        // nothing), but not of unions, which the native sort orders by the row
        // format, unlike DataFusion's merge.
        let schema = merge.input().schema();
        let keys_are_columns = merge.expr().iter().all(|key| {
            key.expr.is::<Column>()
                && key
                    .expr
                    .data_type(&schema)
                    .is_ok_and(|data_type| !contains_union(&data_type))
        });
        if !keys_are_columns
            || may_fail_to_combine(plan)
            || !merge
                .input()
                .equivalence_properties()
                .ordering_satisfy(merge.expr().clone())?
            || !crate::sort::SortKeys::supports(merge.expr(), &schema)?
        {
            return Ok(None);
        }
        let input = self.lower(merge.input(), Context::below_breaker(Failures::NONE))?;
        let mut operator = &input;
        while let Some(projection) = operator.downcast_ref::<PhysicalProjection>() {
            operator = &projection.children()[0];
        }
        if !operator.is::<PhysicalWindow>() {
            return Ok(None);
        }
        let sort = PhysicalSort::try_new(input, merge.expr().clone(), merge.fetch())?;
        Ok(Some(Arc::new(sort)))
    }

    /// Lower a nested loop join or a cross join with
    /// [`PhysicalNestedLoopJoin::try_lower`] or
    /// [`PhysicalCrossJoin::try_lower`], for a runtime whose memory pool has
    /// no limit (see [`lower_execution_plan_with_runtime`]).
    ///
    /// The join reads all of its build side before it probes, as DataFusion
    /// does: its build side is lowered like the input of a pipeline breaker.
    /// Its probe side continues the pipeline of the nodes above the join.
    ///
    /// DataFusion joins every probe batch with blocks of build rows (the
    /// nested loop join evaluates its filter on these blocks of pairs, and
    /// coalesces its output), the native join every chunk of a probe batch
    /// with blocks of build rows, so that the pairs come in another order, and
    /// in other blocks:
    ///
    /// * a join below a node of the same pipeline that may fail is left to
    ///   DataFusion: which of the rows on which that node fails comes first
    ///   depends on the order of the pairs (and DataFusion's nested loop join
    ///   pulls later probe batches, whose errors it raises, before it emits
    ///   the pairs of earlier ones);
    /// * so is a join with types whose concatenation may fail (see
    ///   [`may_fail_to_concatenate`]) in an input or in its output:
    ///   DataFusion's cross join concatenates its whole build side and repeats
    ///   every build row for a whole probe batch, and its nested loop join
    ///   coalesces its output, while the native joins gather chunks of other
    ///   rows, so that one of them may fail where the other does not;
    /// * and a nested loop join with a filter that may fail, or whose errors
    ///   depend on the blocks of pairs it is evaluated on.
    fn try_lower_loop_join(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        ctx: Context,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let nested_loop = plan.downcast_ref::<NestedLoopJoinExec>();
        let (build, probe) = if let Some(join) = nested_loop {
            (join.left(), join.right())
        } else if let Some(join) = plan.downcast_ref::<CrossJoinExec>() {
            (&join.left, &join.right)
        } else {
            return Ok(None);
        };
        let Some(runtime) = self.runtime else {
            return Ok(None);
        };
        if !ctx.fallible_above.is_none()
            || [build, probe, plan]
                .iter()
                .any(|node| concatenation_may_fail(&node.schema()))
        {
            return Ok(None);
        }
        if let Some(filter) = nested_loop.and_then(NestedLoopJoinExec::filter) {
            let mut judgement = Judgement::default();
            judgement.add(filter.expression(), filter.schema());
            if judgement.batch_dependent || !judgement.failures.is_none() {
                return Ok(None);
            }
        }
        let lower_input = |input: &Arc<dyn ExecutionPlan>| {
            // The probe side continues the pipeline of the nodes above the
            // join, the build side is read before anything is probed.
            let side = if Arc::ptr_eq(input, probe) {
                ctx
            } else {
                Context::below_breaker(Failures::NONE)
            };
            self.lower(input, side)
        };
        let join: Option<Arc<dyn PhysicalOperator>> = if nested_loop.is_some() {
            PhysicalNestedLoopJoin::try_lower(plan, runtime, lower_input)?
                .map(|join| Arc::new(join) as _)
        } else {
            PhysicalCrossJoin::try_lower(plan, runtime, lower_input)?
                .map(|join| Arc::new(join) as _)
        };
        Ok(join)
    }

    /// Lower a `FilterExec` to a [`PhysicalFilter`] wrapped in a
    /// [`CachingOperator`], if that keeps its results and errors exactly, see
    /// [`PhysicalFilter::supports_filter_exec`] and the
    /// [module documentation](self).
    fn lower_filter(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        filter: &FilterExec,
        ctx: Context,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        // `FilterExec` coalesces its output: it pulls later batches of its
        // input, whose errors it raises, before the nodes above it see the
        // rows of earlier ones, which the native filter passes on at once.
        if conflict(ctx.fallible_above, subtree_failures(plan, self.runtime)) {
            return Ok(bridge(plan));
        }
        if PhysicalFilter::supports_filter_exec(filter) {
            let own = failures(filter.predicate(), &filter.input().schema());
            if !self.fails_like_datafusion_on_threads(own) {
                return Ok(bridge(plan));
            }
            let input = self.lower(
                filter.input(),
                Context {
                    limited: filter.fetch().is_some(),
                    fallible_above: ctx.fallible_above.and(own),
                },
            )?;
            if !conflict(own, native_failures(&input)) {
                let filter = PhysicalFilter::try_from_filter_exec(filter, input)?;
                return Ok(Arc::new(CachingOperator::new(Arc::new(filter))));
            }
        }
        Ok(bridge(plan))
    }

    /// Lower a `ProjectionExec` to a [`PhysicalProjection`], if that keeps
    /// its results and errors exactly, see
    /// [`PhysicalProjection::supports_projection_exec`] and the
    /// [module documentation](self).
    fn lower_projection(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        projection: &ProjectionExec,
        ctx: Context,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if PhysicalProjection::supports_projection_exec(projection) {
            let input_schema = projection.input().schema();
            let own = projection
                .expr()
                .iter()
                .map(|expr| failures(&expr.expr, &input_schema))
                .fold(Failures::NONE, Failures::and);
            if !self.fails_like_datafusion_on_threads(own) {
                return Ok(bridge(plan));
            }
            let input = self.lower(
                projection.input(),
                Context {
                    limited: false,
                    fallible_above: ctx.fallible_above.and(own),
                },
            )?;
            if !conflict(own, native_failures(&input)) {
                return Ok(Arc::new(PhysicalProjection::try_from_projection_exec(
                    projection, input,
                )?));
            }
        }
        Ok(bridge(plan))
    }
}

/// Replace the children of a plan, recomputing its properties.
fn recompute() -> ReplaceChildrenOptions {
    ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute)
}

/// `plan`, with its whole subtree, executed by DataFusion.
fn bridge(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn PhysicalOperator> {
    Arc::new(BridgeOperator::new(Arc::clone(plan)))
}

/// The input of `plan`, if `plan` only distributes the rows of its input over
/// partitions or changes the size of its batches, without changing them: the
/// pipeline engine has no partitions, and sizes its chunks itself. These are
///
/// * a `CoalescePartitionsExec` without a fetch (with a fetch it is a limit;
///   without input partitions, DataFusion fails to execute it);
/// * a `RepartitionExec` (round robin or by hash) that does not preserve the
///   order, of at least one partition, whose hash expressions cannot fail
///   (any other is left to DataFusion, which evaluates it on every batch),
///   and that, if it coalesces its output (see [`coalesces_batches`]), does
///   so with types whose concatenation cannot fail;
/// * a `CoalesceBatchesExec` without a fetch, of types whose concatenation
///   cannot fail.
fn redistributed_input(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    let input = plan.children().into_iter().next()?;
    if input.output_partitioning().partition_count() == 0 {
        return None;
    }
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
        return coalesce.fetch().is_none().then_some(input);
    }
    if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
        let schema = input.schema();
        let infallible = match repartition.partitioning() {
            Partitioning::RoundRobinBatch(_) => true,
            Partitioning::Hash(exprs, _) => exprs.iter().all(|expr| {
                !depends_on_batches(expr, &schema) && failures(expr, &schema).is_none()
            }),
            _ => false,
        };
        let coalesces = coalesces_batches(plan) && concatenation_may_fail(&schema);
        return (infallible && !repartition.preserve_order() && !coalesces)
            .then_some(input);
    }
    if is_coalesce_batches(plan) {
        return (plan.fetch().is_none() && !concatenation_may_fail(&plan.schema()))
            .then_some(input);
    }
    None
}

/// Whether `plan` concatenates the batches of its input, up to `batch_size`
/// rows: a `CoalesceBatchesExec`, or a `RepartitionExec` over a bounded input
/// that does not preserve the order, which coalesces the batches of each of
/// its output partitions. Such a node may fail to concatenate them (see
/// [`may_fail_to_concatenate`]), and pulls later batches of its input, whose
/// errors it raises, before the nodes above it see the rows of earlier ones.
fn coalesces_batches(plan: &Arc<dyn ExecutionPlan>) -> bool {
    is_coalesce_batches(plan)
        || plan
            .downcast_ref::<RepartitionExec>()
            .is_some_and(|repartition| {
                !repartition.preserve_order()
                    && !repartition.input().boundedness().is_unbounded()
            })
}

/// Whether concatenating batches of `schema` may fail (see
/// [`may_fail_to_concatenate`]).
fn concatenation_may_fail(schema: &Schema) -> bool {
    schema
        .fields()
        .iter()
        .any(|field| may_fail_to_concatenate(field.data_type()))
}

/// The `RepartitionExec`s of the chain of `RepartitionExec`s and
/// `CoalescePartitionsExec`s at the top of `plan`, which a node may remove
/// below it (e.g. a hash join, from its inputs).
fn top_repartitions(plan: &Arc<dyn ExecutionPlan>) -> Vec<&Arc<dyn ExecutionPlan>> {
    let mut repartitions = vec![];
    let mut node = plan;
    loop {
        if node.is::<RepartitionExec>() {
            repartitions.push(node);
        } else if !node.is::<CoalescePartitionsExec>() {
            return repartitions;
        }
        match node.children().as_slice() {
            [child] => node = *child,
            _ => return repartitions,
        }
    }
}

/// Whether `plan` is a `CoalesceBatchesExec` (which DataFusion does not plan
/// any more, but which other planners may).
#[expect(deprecated)]
fn is_coalesce_batches(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec>()
}

/// Whether concatenating arrays of `data_type`, as the nodes that coalesce
/// batches do (see [`coalesces_batches`]), may fail: it merges the values of
/// dictionaries, which may overflow their keys, adds up run ends, and the
/// offsets of lists, strings and binaries with 32-bit offsets (which lists of
/// nulls, and batches that share large strings, overflow without memory), at
/// any depth.
fn may_fail_to_concatenate(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(..)
        | DataType::RunEndEncoded(..)
        | DataType::List(_)
        | DataType::ListView(_)
        | DataType::Map(..)
        | DataType::Utf8
        | DataType::Binary => true,
        DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _) => may_fail_to_concatenate(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| may_fail_to_concatenate(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| may_fail_to_concatenate(field.data_type())),
        _ => false,
    }
}

/// Whether `plan`, a node whose input is removed (see
/// [`redistributed_input`]), spreads its input over more partitions.
fn spreads_partitions(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<RepartitionExec>()
        && plan.children().iter().all(|input| {
            input.output_partitioning().partition_count()
                < plan.output_partitioning().partition_count()
        })
}

/// Whether `side` is `input`, or only redistributes the rows of `input` over
/// partitions: `input` is below `side`, separated from it only by
/// `RepartitionExec`s and `CoalescePartitionsExec`s.
fn redistributes(side: &Arc<dyn ExecutionPlan>, input: &Arc<dyn ExecutionPlan>) -> bool {
    let mut node = side;
    loop {
        if Arc::ptr_eq(node, input) {
            return true;
        }
        if !(node.is::<RepartitionExec>() || node.is::<CoalescePartitionsExec>()) {
            return false;
        }
        match node.children().as_slice() {
            [child] => node = child,
            _ => return false,
        }
    }
}

/// Whether values of `data_type` are unions, or have unions inside them.
fn contains_union(data_type: &DataType) -> bool {
    match data_type {
        DataType::Union(..) => true,
        DataType::Dictionary(_, values) => contains_union(values),
        DataType::RunEndEncoded(_, values) => contains_union(values.data_type()),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => contains_union(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|field| contains_union(field.data_type()))
        }
        _ => false,
    }
}

/// Whether `plan` is a window node: a `WindowAggExec` or a
/// `BoundedWindowAggExec`.
fn is_window(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<WindowAggExec>() || plan.is::<BoundedWindowAggExec>()
}

/// The nodes of the plan of `window`, a window node, that a native window may
/// replace (see [`PhysicalWindow::try_lower`]), conservatively: `window`, the
/// nodes below it that only distribute and sort its input (`SortExec`s,
/// `SortPreservingMergeExec`s and `CoalescePartitionsExec`s without a fetch,
/// and `RepartitionExec`s), and the window node that `window` may be stacked
/// on (through `ProjectionExec`s, `RepartitionExec`s and
/// `CoalescePartitionsExec`s, which are replaced too), with the nodes that it
/// may replace in turn.
fn window_nodes(window: &Arc<dyn ExecutionPlan>) -> Vec<&Arc<dyn ExecutionPlan>> {
    let mut nodes = vec![window];
    let Some(input) = window.children().into_iter().next() else {
        return nodes;
    };
    let mut node = input;
    loop {
        let distributes = node
            .downcast_ref::<SortExec>()
            .is_some_and(|sort| sort.fetch().is_none())
            || node
                .downcast_ref::<SortPreservingMergeExec>()
                .is_some_and(|merge| merge.fetch().is_none())
            || node
                .downcast_ref::<CoalescePartitionsExec>()
                .is_some_and(|coalesce| coalesce.fetch().is_none())
            || node.is::<RepartitionExec>();
        match node.children().as_slice() {
            [child] if distributes => {
                nodes.push(node);
                node = *child;
            }
            _ => break,
        }
    }
    let mut stacked = vec![];
    let mut node = input;
    loop {
        if is_window(node) {
            nodes.extend(stacked);
            nodes.extend(window_nodes(node));
            break;
        }
        let passes_rows = node.is::<ProjectionExec>()
            || node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>();
        match node.children().as_slice() {
            [child] if passes_rows => {
                stacked.push(node);
                node = *child;
            }
            _ => break,
        }
    }
    nodes
}

/// `window`, a window node, with the projection of columns between the sort
/// below it (or the merge of the sorted partitions of the sort below it) and
/// the input of the sort moved below the sort, `None` if there is none: a
/// `SortExec` without a fetch, by keys that are columns of the projection.
///
/// The sort then sorts the projected rows by the same columns, which is the
/// same sort of the same rows: the plan has the same schema (the window
/// expressions and the keys of the merge are bound to the projection, as
/// before) and produces the same rows in the same order, but for the rows
/// that the keys do not order (ties), which come in any order, as from any
/// sort. A projection of columns evaluates nothing, so it does not matter on
/// which rows it is evaluated together.
///
/// But DataFusion's sort sorts the rows of its input with all of their
/// columns, also those that the projection then drops, while the sort of the
/// rewritten plan (which the native window replaces) only sorts the projected
/// ones. So the rewrite is refused if the sort below `window` may fail to
/// combine its batches (see [`may_fail_to_combine`], e.g. because of a column
/// that the projection drops), or if its keys may fail or have errors that
/// depend on the batches: the guards of the native window only judge the
/// nodes of the rewritten plan.
fn with_projection_below_sort(
    window: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(input) = window.children().into_iter().next() else {
        return Ok(None);
    };
    let merge = input
        .downcast_ref::<SortPreservingMergeExec>()
        .filter(|merge| merge.fetch().is_none());
    let above_sort = match merge {
        Some(merge) => merge.input(),
        None => input,
    };
    let Some(projection) = above_sort.downcast_ref::<ProjectionExec>() else {
        return Ok(None);
    };
    let sort_plan = projection.input();
    let Some(sort) = sort_plan
        .downcast_ref::<SortExec>()
        .filter(|sort| sort.fetch().is_none())
    else {
        return Ok(None);
    };
    let keys = Judgement::of_plan(sort);
    if keys.batch_dependent || !keys.failures.is_none() || may_fail_to_combine(sort_plan)
    {
        return Ok(None);
    }
    let columns = projection
        .expr()
        .iter()
        .map(|expr| expr.expr.downcast_ref::<Column>().map(Column::index))
        .collect::<Option<Vec<_>>>();
    let Some(columns) = columns else {
        return Ok(None);
    };
    // The keys of the sort, as columns of the projection.
    let keys = sort
        .expr()
        .iter()
        .map(|key| {
            let column = key.expr.downcast_ref::<Column>()?.index();
            let output = columns.iter().position(|&input| input == column)?;
            let name = &projection.expr()[output].alias;
            Some(PhysicalSortExpr::new(
                Arc::new(Column::new(name, output)),
                key.options,
            ))
        })
        .collect::<Option<Vec<_>>>();
    let Some(keys) = keys.and_then(LexOrdering::new) else {
        return Ok(None);
    };
    let projected = Arc::clone(above_sort)
        .replace_children(vec![Arc::clone(sort.input())], recompute())?;
    let sorted: Arc<dyn ExecutionPlan> = Arc::new(
        SortExec::new(keys, projected)
            .with_preserve_partitioning(sort.preserve_partitioning()),
    );
    let input = match merge {
        Some(_) => Arc::clone(input).replace_children(vec![sorted], recompute())?,
        None => sorted,
    };
    Ok(Some(
        Arc::clone(window).replace_children(vec![input], recompute())?,
    ))
}

/// Whether a TopK that may end some of its partitions early is among the
/// nodes below `window`, a window node, that the native window reads its input
/// through (see [`PhysicalWindow::try_lower`]): sorts, merges,
/// repartitionings, projections, filters, unions and windows.
///
/// DataFusion's TopK (a `SortExec` with a fetch) whose input is sorted by a
/// prefix of its keys ends a partition as soon as no later row of it can make
/// it into the result, and a `SortExec` with a fetch whose input is sorted by
/// all of its keys passes it through, with a limit: then a partition of
/// DataFusion's window can emit its rows while the input of another partition
/// is still pending (or fails later), while the native window reads all of its
/// input before it emits anything. Such windows are left to DataFusion
/// (conservatively, also those of a single partition).
fn topk_may_end_early_below(window: &Arc<dyn ExecutionPlan>) -> bool {
    let mut nodes = window.children();
    while let Some(node) = nodes.pop() {
        if let Some(sort) = node.downcast_ref::<SortExec>()
            && sort.fetch().is_some()
            && sort
                .input()
                .equivalence_properties()
                .extract_common_sort_prefix(sort.expr().clone())
                .map_or(true, |(prefix, _)| !prefix.is_empty())
        {
            return true;
        }
        if node.is::<SortExec>()
            || node.is::<SortPreservingMergeExec>()
            || node.is::<CoalescePartitionsExec>()
            || node.is::<RepartitionExec>()
            || node.is::<ProjectionExec>()
            || node.is::<FilterExec>()
            || node.is::<UnionExec>()
            || is_window(node)
        {
            nodes.extend(node.children());
        }
    }
    false
}

/// The `SortExec` of `plan`, if `plan` is a `SortExec`, or a chain of
/// `SortPreservingMergeExec`s (and projections between them) above a
/// `SortExec`, as [`crate::sort::try_lower`] lowers them.
fn sort_of_chain(plan: &Arc<dyn ExecutionPlan>) -> Option<&SortExec> {
    let mut node = plan;
    let mut merged = false;
    while node.is::<SortPreservingMergeExec>() || (merged && node.is::<ProjectionExec>())
    {
        merged = true;
        node = node.children().into_iter().next()?;
    }
    node.downcast_ref::<SortExec>()
}

/// The `AggregateExec` that consumes the input rows of the aggregation
/// `plan`: `plan` itself in `Single` or `SinglePartitioned` mode, or the
/// `Partial` aggregation below `plan` in `Final` or `FinalPartitioned` mode,
/// through the nodes that distribute its states.
fn raw_aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<&AggregateExec> {
    let aggregate = plan.downcast_ref::<AggregateExec>()?;
    match aggregate.mode() {
        AggregateMode::Single | AggregateMode::SinglePartitioned => Some(aggregate),
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            let mut node = aggregate.input();
            loop {
                if let Some(partial) = node.downcast_ref::<AggregateExec>() {
                    return (partial.mode() == &AggregateMode::Partial)
                        .then_some(partial);
                }
                if !(node.is::<RepartitionExec>() || node.is::<CoalescePartitionsExec>())
                {
                    return None;
                }
                node = node.children().into_iter().next()?;
            }
        }
        AggregateMode::Partial | AggregateMode::PartialReduce => None,
    }
}

/// The input of `plan`, the input of `raw`, the aggregation that consumes the
/// input rows, if `plan` is a sort that DataFusion added to order these rows
/// for the ordered aggregates of `raw` (e.g. `ARRAY_AGG(x ORDER BY y)`), and
/// otherwise `plan`. The native aggregation orders the values of these
/// aggregates itself: it uses accumulators that do not rely on the order of
/// their input, and sorts the rows of every thread for the aggregates that
/// require it (see [`PhysicalHashAggregate`]). The sort is removed if it sorts
/// every partition of its input, without a fetch, by keys that cannot fail
/// (DataFusion evaluates them on all rows, the native aggregation evaluates
/// them on the rows that pass the filters of the aggregates), unless the
/// expressions of `raw` (with the parts that may fail `own`) may fail:
/// DataFusion evaluates them on the sorted rows, so that which row fails
/// first depends on the sort (and the sort reads all of its input before the
/// aggregation sees any row, so DataFusion raises the errors of the input
/// first).
fn without_ordering_sort<'a>(
    raw: &AggregateExec,
    own: Failures,
    plan: &'a Arc<dyn ExecutionPlan>,
) -> &'a Arc<dyn ExecutionPlan> {
    let Some(sort) = plan.downcast_ref::<SortExec>() else {
        return plan;
    };
    let orders_aggregates = raw
        .required_input_ordering()
        .first()
        .is_some_and(Option::is_some)
        && raw
            .aggr_expr()
            .iter()
            .any(|aggregate| !aggregate.order_bys().is_empty());
    let sorts_partitions = sort.input().output_partitioning().partition_count()
        == plan.output_partitioning().partition_count();
    let keys = Judgement::of_ordering(sort.expr(), &sort.input().schema());
    if orders_aggregates
        && sorts_partitions
        && sort.fetch().is_none()
        && !keys.batch_dependent
        && keys.failures.is_none()
        && own.is_none()
    {
        sort.input()
    } else {
        plan
    }
}

/// `plan`, an aggregation, as [`PhysicalHashAggregate::try_lower`] lowers it:
/// in `Final` or `FinalPartitioned` mode, without the stages between it and
/// its `Partial` aggregation that `try_lower` does not look through
/// (`CoalesceBatchesExec`s and `PartialReduce` aggregations), and `plan`
/// itself otherwise.
///
/// The native aggregation merges all states of the partial aggregation at
/// once, without these stages and the `RepartitionExec`s and
/// `CoalescePartitionsExec`s that distribute the states: returns `None`, so
/// that DataFusion executes the aggregation, if a stage does not pass the
/// partial states on unchanged (see [`passes_states_on`]), e.g. a
/// `PartialReduce` that merges them into other groups, or a repartitioning by
/// an expression that may fail.
fn aggregation_to_lower(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(last) = plan.downcast_ref::<AggregateExec>() else {
        return Ok(Some(Arc::clone(plan)));
    };
    if !matches!(
        last.mode(),
        AggregateMode::Final | AggregateMode::FinalPartitioned
    ) {
        return Ok(Some(Arc::clone(plan)));
    }
    // The stages from the top, down to the partial aggregation.
    let mut stages = vec![];
    let mut node = last.input();
    let partial = loop {
        if let Some(aggregate) = node.downcast_ref::<AggregateExec>() {
            match aggregate.mode() {
                AggregateMode::Partial => break aggregate,
                AggregateMode::PartialReduce => {}
                _ => return Ok(None),
            }
        } else if !(node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>()
            || is_coalesce_batches(node))
        {
            return Ok(None);
        }
        stages.push(node);
        match node.children().as_slice() {
            [input] => node = *input,
            _ => return Ok(None),
        }
    };
    if !stages.iter().all(|stage| passes_states_on(stage, partial)) {
        return Ok(None);
    }
    let distributes = |stage: &Arc<dyn ExecutionPlan>| {
        stage.is::<RepartitionExec>() || stage.is::<CoalescePartitionsExec>()
    };
    if stages.iter().all(|stage| distributes(stage)) {
        return Ok(Some(Arc::clone(plan)));
    }
    // Without the stages that only coalesce batches or reduce states.
    let mut input = Arc::clone(node);
    for stage in stages.iter().rev().filter(|stage| distributes(stage)) {
        input = Arc::clone(stage).replace_children(vec![input], recompute())?;
    }
    Ok(Some(
        Arc::clone(plan).replace_children(vec![input], recompute())?,
    ))
}

/// Whether `stage`, a node between a final aggregation and the `Partial`
/// aggregation `partial` whose states it merges, passes these states on
/// unchanged, as the native aggregation assumes:
///
/// * a `RepartitionExec` that does not preserve the order, round robin, or by
///   hash expressions that cannot fail and whose errors do not depend on the
///   batches (DataFusion evaluates them on every batch of states), and that,
///   if it coalesces its output (see [`coalesces_batches`]), does so with
///   states whose concatenation cannot fail;
/// * a `CoalescePartitionsExec` without a fetch;
/// * a `CoalesceBatchesExec` without a fetch, of states whose concatenation
///   cannot fail;
/// * a `PartialReduce` aggregation that merges the states of `partial` into
///   the same states: it groups by the group columns of the states and merges
///   the aggregates of `partial`, without filters or a limit.
fn passes_states_on(stage: &Arc<dyn ExecutionPlan>, partial: &AggregateExec) -> bool {
    if let Some(repartition) = stage.downcast_ref::<RepartitionExec>() {
        let schema = repartition.input().schema();
        if repartition.preserve_order()
            || (coalesces_batches(stage) && concatenation_may_fail(&schema))
        {
            return false;
        }
        return match repartition.partitioning() {
            Partitioning::RoundRobinBatch(_) => true,
            Partitioning::Hash(exprs, _) => exprs.iter().all(|expr| {
                !depends_on_batches(expr, &schema) && failures(expr, &schema).is_none()
            }),
            _ => false,
        };
    }
    if let Some(coalesce) = stage.downcast_ref::<CoalescePartitionsExec>() {
        return coalesce.fetch().is_none();
    }
    if is_coalesce_batches(stage) {
        return stage.fetch().is_none() && !concatenation_may_fail(&stage.schema());
    }
    stage.downcast_ref::<AggregateExec>().is_some_and(|reduce| {
        reduce.mode() == &AggregateMode::PartialReduce
            && reduce.group_expr() == &partial.group_expr().as_final()
            && reduce.aggr_expr() == partial.aggr_expr()
            && reduce.filter_expr().iter().all(Option::is_none)
            && reduce.limit_options().is_none()
            && reduce.input().schema() == partial.schema()
            && reduce.schema() == partial.schema()
    })
}

/// How the expressions that a node evaluates on the rows of its input can
/// fail.
#[derive(Debug, Clone, Copy, Default)]
struct Judgement {
    /// Whether the errors of an expression depend on the batches it is
    /// evaluated on (see [`depends_on_batches`]).
    batch_dependent: bool,
    /// The parts of the expressions that may fail (see [`failures`]).
    failures: Failures,
}

impl Judgement {
    /// The judgement of the expressions that `plan` evaluates on its input:
    /// judged on the schema of its input if it has a single child, the keys and
    /// the filter of a join on the schemas of the rows they are evaluated on
    /// (the keys of each side on its rows, the filter on the columns of the
    /// candidate pairs that it reads, see [`join_expression_schema`]), and
    /// without schema otherwise, which is conservative (expressions that may
    /// fail then fail on empty batches). The aggregations that merge partial states
    /// evaluate no expression on their input (the arguments of their aggregate
    /// functions are those of the partial aggregations), and the dynamic
    /// filters that `plan` produces (e.g. the threshold of a TopK) compare the
    /// values of its own expressions, which are judged. Expressions that cannot
    /// be listed may fail, depending on the batches.
    fn of_plan(plan: &dyn ExecutionPlan) -> Self {
        let mut judgement = Self::default();
        if plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|aggregate| {
                aggregate.mode().input_mode() == AggregateInputMode::Partial
            })
        {
            return judgement;
        }
        let schema = match plan.children().as_slice() {
            [child] => child.schema(),
            _ => Arc::new(Schema::empty()),
        };
        let produced = plan.dynamic_expressions_produced();
        let listed = plan.apply_expressions(&mut |expr| {
            if !produced
                .iter()
                .any(|dynamic| std::ptr::addr_eq(Arc::as_ptr(dynamic), Arc::as_ptr(expr)))
            {
                match join_expression_schema(plan, expr) {
                    Some(join_schema) => judgement.add(expr, &join_schema),
                    None => judgement.add(expr, &schema),
                }
            }
            Ok(TreeNodeRecursion::Continue)
        });
        if listed.is_err() {
            judgement = Self::unknown();
        }
        judgement
    }

    /// The judgement of expressions that are not known: conservatively, they
    /// may fail, with different kinds of errors, depending on the batches.
    fn unknown() -> Self {
        Self {
            batch_dependent: true,
            failures: Failures::MIXED,
        }
    }

    /// The judgement of the keys of `ordering`, on rows of `schema`.
    fn of_ordering(ordering: &LexOrdering, schema: &Schema) -> Self {
        let mut judgement = Self::default();
        for key in ordering.iter() {
            judgement.add(&key.expr, schema);
        }
        judgement
    }

    fn add(&mut self, expr: &Arc<dyn PhysicalExpr>, schema: &Schema) {
        self.batch_dependent |= depends_on_batches(expr, schema);
        self.failures = self.failures.and(failures(expr, schema));
    }

    /// Whether a native operator that evaluates these expressions on its
    /// chunks, rather than on DataFusion's batches, produces the results and
    /// raises the errors of DataFusion: no expression has errors that depend
    /// on the batches, and at most one part may fail, but for parts that can
    /// only fail with arithmetic overflows, which raise the same error in any
    /// order (DataFusion evaluates every expression on a whole batch before
    /// the next one, so which of several other errors comes first depends on
    /// its batches, see [`Failures::order_matters`]).
    fn evaluates_like_datafusion(&self) -> bool {
        !self.batch_dependent && !self.failures.order_matters()
    }
}

/// The schema of the rows on which `plan`, a hash join or a nested loop join,
/// evaluates its expression `expr`: the schema of its left (build) input for
/// one of its left keys, of its right (probe) input for one of its right keys,
/// and the schema of its filter (the columns of both sides that the filter
/// reads) for its filter. `None` for any other node or expression.
fn join_expression_schema(
    plan: &dyn ExecutionPlan,
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<SchemaRef> {
    let is = |candidate: &Arc<dyn PhysicalExpr>| Arc::ptr_eq(candidate, expr);
    let (keys, filter) = match plan.downcast_ref::<HashJoinExec>() {
        Some(join) => (join.on(), join.filter()),
        None => (&[][..], plan.downcast_ref::<NestedLoopJoinExec>()?.filter()),
    };
    let children = plan.children();
    if let Some(filter) = filter.filter(|filter| is(filter.expression())) {
        return Some(Arc::clone(filter.schema()));
    }
    if keys.iter().any(|(left, _)| is(left)) {
        return children.first().map(|left| left.schema());
    }
    if keys.iter().any(|(_, right)| is(right)) {
        return children.get(1).map(|right| right.schema());
    }
    None
}

/// The judgement of the keys that `sort`, a [`PhysicalSort`] or a
/// [`PhysicalTopK`], evaluates on the rows of its input.
///
/// [`PhysicalTopK`]: crate::sort::PhysicalTopK
fn native_sort_keys(sort: &Arc<dyn PhysicalOperator>) -> Judgement {
    let ordering = if let Some(sort) = sort.downcast_ref::<PhysicalSort>() {
        sort.ordering()
    } else if let Some(topk) = sort.downcast_ref::<PhysicalTopK>() {
        topk.ordering()
    } else {
        // Conservatively: unknown keys may fail, depending on the batches.
        return Judgement::unknown();
    };
    let Some(input) = sort.children().first() else {
        return Judgement::default();
    };
    Judgement::of_ordering(ordering, &input.schema())
}

/// Whether the parts that may fail `above` and `below` (see [`Failures`]),
/// of operators that see the rows of a chunk one after the other (or of a
/// node and of the nodes below it, whose later rows it may read before the
/// nodes above it see the earlier ones), may raise different errors, so that
/// which error comes first depends on which of them sees a row first: if both
/// have parts that may fail, but for parts that can only fail with arithmetic
/// overflows, which raise the same error in any order.
fn conflict(above: Failures, below: Failures) -> bool {
    !above.is_none() && !below.is_none() && above.and(below).order_matters()
}

/// The parts that may fail of all nodes of `plan` (see [`node_failures`]),
/// in the pipelines of `plan`: the subqueries of a `ScalarSubqueryExec` run
/// before its input, in both engines (see [`scalar_subquery_input`]).
fn subtree_failures(
    plan: &Arc<dyn ExecutionPlan>,
    runtime: Option<&RuntimeEnv>,
) -> Failures {
    pipeline_children(plan)
        .into_iter()
        .map(|child| subtree_failures(child, runtime))
        .fold(node_failures(plan, runtime), Failures::and)
}

/// The parts of `plan` itself that may fail: the parts of its expressions
/// that may fail (see [`Judgement::of_plan`], and [`file_scan_failures`] for
/// the scans of Parquet and CSV files, whose expressions are judged on the
/// schema of their table), the sources that may fail to read or decode their
/// rows (see [`source_may_fail`]), and the nodes that may fail to combine
/// their batches (see [`may_fail_to_combine`]), which may fail with several
/// kinds of errors.
fn node_failures(
    plan: &Arc<dyn ExecutionPlan>,
    runtime: Option<&RuntimeEnv>,
) -> Failures {
    let mut failures = file_scan_failures(plan)
        .unwrap_or_else(|| Judgement::of_plan(plan.as_ref()).failures);
    if source_may_fail(plan, runtime) {
        failures = failures.and(Failures::MIXED);
    }
    if may_fail_to_combine(plan) {
        failures = failures.and(Failures::MIXED);
    }
    failures
}

/// The children of `plan` whose rows `plan` reads in its own pipelines: all
/// of them, but for the subqueries of a `ScalarSubqueryExec`, which run (to
/// their end) before its input does.
fn pipeline_children(plan: &Arc<dyn ExecutionPlan>) -> Vec<&Arc<dyn ExecutionPlan>> {
    match scalar_subquery_input(plan) {
        Some(input) => vec![input],
        None => plan.children(),
    }
}

/// The input of `plan`, if it is a `ScalarSubqueryExec`.
fn scalar_subquery_input(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<&Arc<dyn ExecutionPlan>> {
    plan.downcast_ref::<ScalarSubqueryExec>()
        .map(ScalarSubqueryExec::input)
}

/// If `plan` is a scan of Parquet or CSV files, the parts of its
/// expressions, its predicate and its projection (in terms of the schema of
/// its table), that may fail, with the dynamic filters that it reads judged by
/// their children: a dynamic filter evaluates a predicate of the keys of the
/// node that produces it (bounds, `IN` lists or hash table lookups of the build
/// keys of a join, comparisons of the sort keys of a TopK or of the arguments
/// of `MIN` and `MAX`), which cannot fail if its children cannot.
fn file_scan_failures(plan: &Arc<dyn ExecutionPlan>) -> Option<Failures> {
    let scan = plan.downcast_ref::<DataSourceExec>()?;
    let config = scan.data_source().downcast_ref::<FileScanConfig>()?;
    let source = config.file_source();
    if !(source.is::<ParquetSource>() || source.is::<CsvSource>()) {
        return None;
    }
    let schema = source.table_schema().table_schema();
    let projection = source
        .projection()
        .into_iter()
        .flat_map(|projection| projection.iter().map(|expr| Arc::clone(&expr.expr)));
    let mut judgement = Judgement::default();
    for expr in source.filter().into_iter().chain(projection) {
        let Ok(expr) = without_dynamic_filters(expr) else {
            return Some(Failures::MIXED);
        };
        judgement.add(&expr, schema);
    }
    Some(if judgement.batch_dependent {
        judgement.failures.and(Failures::MIXED)
    } else {
        judgement.failures
    })
}

/// `expr` with its dynamic filters (see `DynamicFilterPhysicalExpr`) replaced
/// by expressions that evaluate their children, and nothing that may fail:
/// `child IS NOT NULL AND ...`.
fn without_dynamic_filters(expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
    expr.transform_up(|expr| {
        let Some(filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() else {
            return Ok(Transformed::no(expr));
        };
        let children = filter.children().into_iter().map(|child| {
            Arc::new(IsNotNullExpr::new(Arc::clone(child))) as Arc<dyn PhysicalExpr>
        });
        Ok(Transformed::yes(conjunction(children)))
    })
    .map(|transformed| transformed.data)
}

/// Whether arrays of `data_type` can address more values than their memory
/// holds, so that concatenating a few bytes of them can overflow: run-end
/// encoded arrays with run ends of fewer than 64 bits (which address as many
/// rows as their last run end says), and lists and maps with 32-bit offsets
/// of nulls (which occupy no memory) or of such arrays, at any depth.
fn addresses_more_than_their_memory(data_type: &DataType) -> bool {
    fn addresses(data_type: &DataType, in_32_bit_offsets: bool) -> bool {
        match data_type {
            DataType::Null => in_32_bit_offsets,
            DataType::RunEndEncoded(run_ends, values) => {
                run_ends.data_type() != &DataType::Int64
                    || addresses(values.data_type(), in_32_bit_offsets)
            }
            DataType::List(field)
            | DataType::ListView(field)
            | DataType::Map(field, _) => addresses(field.data_type(), true),
            DataType::LargeList(field)
            | DataType::LargeListView(field)
            | DataType::FixedSizeList(field, _) => {
                addresses(field.data_type(), in_32_bit_offsets)
            }
            DataType::Dictionary(_, values) => addresses(values, in_32_bit_offsets),
            DataType::Struct(fields) => fields
                .iter()
                .any(|field| addresses(field.data_type(), in_32_bit_offsets)),
            DataType::Union(fields, _) => fields
                .iter()
                .any(|(_, field)| addresses(field.data_type(), in_32_bit_offsets)),
            _ => false,
        }
    }
    addresses(data_type, false)
}

/// Whether a node of `plan` consumes the dynamic filter with id `filter`: e.g.
/// a scan, which skips the rows that the filter rejects.
fn consumes_dynamic_filter(plan: &Arc<dyn ExecutionPlan>, filter: u64) -> bool {
    let mut consumes = false;
    let _ = plan.apply(|node| {
        let _ = node.apply_expressions(&mut |expr| {
            expr.apply(|expr| {
                consumes |= expr.expression_id() == Some(filter);
                Ok(TreeNodeRecursion::Continue)
            })
        });
        Ok(match consumes {
            true => TreeNodeRecursion::Stop,
            false => TreeNodeRecursion::Continue,
        })
    });
    consumes
}

/// The dynamic filter of `sort`, a `SortExec`, and its id, if `sort` is a TopK
/// whose dynamic filter a node of its input reads (see
/// [`consumes_dynamic_filter`]).
fn read_topk_filter(sort: &SortExec) -> Option<(Arc<DynamicFilterPhysicalExpr>, u64)> {
    sort.fetch()?;
    let filter = sort
        .dynamic_expressions_produced()
        .into_iter()
        .find(|expr| expr.is::<DynamicFilterPhysicalExpr>())?;
    let id = filter.expression_id()?;
    let filter = (filter as Arc<dyn std::any::Any + Send + Sync>)
        .downcast::<DynamicFilterPhysicalExpr>()
        .ok()?;
    consumes_dynamic_filter(sort.input(), id).then_some((filter, id))
}

/// How the dynamic filter of a TopK prunes the rows of its input (see
/// [`topk_filter_pruning`]). The TopK tightens the filter (its threshold) as
/// it reads its input, at other times in DataFusion than natively, so that
/// the nodes that read the filter (scans of Parquet files) skip other files,
/// row groups and rows in the two engines. Either way, they only skip rows
/// that cannot make it into the result of the TopK, but the nodes that read
/// the pruned rows (the scans, and the nodes above them) work on other rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TopKFilterPruning {
    /// No node reads the filter.
    Unread,
    /// Every node that reads the filter is below a node that reads all of
    /// its rows before it emits any row (see
    /// [`reads_first_child_before_emitting`]), between it and the TopK, which
    /// all rows of the TopK pass (not below one of several inputs of a union,
    /// whose other inputs may pass rows on before): in both engines, the TopK
    /// sees its first row, and the filter gets its first threshold, once these
    /// nodes have read all of their input, so that the filter prunes none of
    /// their rows.
    Settled,
    /// The nodes that read the filter (and are not below such a node), and
    /// the nodes between them and the TopK, only pass rows on (see
    /// [`passes_rows_on`]): the filter prunes rows that cannot make it into
    /// the result, and the work of these nodes on them.
    Rows,
    /// Other nodes are between a node that reads the filter and the TopK,
    /// e.g. the probe side of a join, an aggregation that emits groups as it
    /// reads its input, or a node with a fetch, which stops after a number of
    /// rows that pass the filter (which depend on the rows it prunes).
    Other,
}

/// How the dynamic filter with id `filter` of a TopK prunes the rows of
/// `plan`, the input of the TopK or a node below it whose rows reach the TopK
/// as `plan` produces them (see [`TopKFilterPruning`]).
///
/// The nodes that read the filter below the child that `plan` reads all of
/// before it emits any row (see [`reads_first_child_before_emitting`]) are
/// settled with respect to the rows of `plan`. `plan` emits the rows of its
/// other children as they come: if it streams several children (e.g. a union),
/// the rows of one of them can reach the TopK, and tighten its filter, while
/// the nodes below another one that read the filter are still reading, also
/// if they are settled with respect to the rows of their own child. So the
/// filter prunes nothing only below a barrier that all rows of the TopK pass:
/// a settled child of a node that streams several children is not settled. A
/// join streams its probe side only: it emits rows once it read the matching
/// part of its build side, which is settled if the join reads all of it first,
/// and may still be read otherwise. And a node that may emit rows before the
/// child it streams emits any is no barrier either (see
/// [`emits_rows_after_its_inputs`]).
fn topk_filter_pruning(plan: &Arc<dyn ExecutionPlan>, filter: u64) -> TopKFilterPruning {
    let read_first = reads_first_child_before_emitting(plan, filter);
    // A join emits its rows as its probe side (the second child) passes rows
    // on (or once a partition of it ends), once it read the matching part of
    // its build side: the timing of its rows is that of its probe side. But
    // other parts of its build side may still be read then, unless it reads
    // all of it first.
    let joins = plan.is::<HashJoinExec>()
        || plan.is::<CrossJoinExec>()
        || plan.is::<NestedLoopJoinExec>();
    let mut settled = false;
    let mut built = TopKFilterPruning::Unread;
    let mut streamed = vec![];
    for (index, child) in plan.children().into_iter().enumerate() {
        if index == 0 && read_first {
            settled = consumes_dynamic_filter(child, filter);
        } else if index == 0 && joins {
            if consumes_dynamic_filter(child, filter) {
                built = TopKFilterPruning::Other;
            }
        } else {
            streamed.push(topk_filter_pruning(child, filter));
        }
    }
    if streamed.len() > 1 {
        for pruning in &mut streamed {
            if *pruning == TopKFilterPruning::Settled {
                *pruning = TopKFilterPruning::Other;
            }
        }
    }
    let below = streamed
        .into_iter()
        .max()
        .unwrap_or(TopKFilterPruning::Unread)
        .max(built);
    if below <= TopKFilterPruning::Settled && !reads_dynamic_filter(plan, filter) {
        // A node that may emit rows of its own before the input it streams
        // emits any (e.g. a custom node that emits a row first) can pass a row
        // to the TopK, and tighten its filter, while the nodes below it are
        // still reading.
        if below == TopKFilterPruning::Settled && !emits_rows_after_its_inputs(plan) {
            return TopKFilterPruning::Other;
        }
        return match settled {
            true => TopKFilterPruning::Settled,
            false => below,
        };
    }
    match passes_rows_on(plan) {
        true => below.max(TopKFilterPruning::Rows),
        false => TopKFilterPruning::Other,
    }
}

/// Whether `plan` emits no row before the inputs that it streams emit theirs:
/// its rows are rows of its inputs, or computed from the rows of its inputs
/// that it read before (also at the end of its input, e.g. an aggregation of
/// no group, or the unmatched rows of an outer join once all of its probe
/// side ended), as for projections, filters, coalesced batches,
/// repartitionings, merges of partitions, unions, sorts, aggregations, limits
/// and joins. But for a `Partitioned` hash join that emits rows of its build
/// side at the end of its probe side (e.g. a left join, or a left semi join),
/// which DataFusion's join does as soon as one partition of its probe side
/// ends, without rows, while its other partitions are still read. Any other
/// node, e.g. a custom node, may emit rows of its own first.
fn emits_rows_after_its_inputs(plan: &Arc<dyn ExecutionPlan>) -> bool {
    use datafusion_common::JoinType;
    if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        return join.partition_mode()
            != &datafusion_physical_plan::joins::PartitionMode::Partitioned
            || !matches!(
                join.join_type(),
                JoinType::Left
                    | JoinType::LeftAnti
                    | JoinType::LeftSemi
                    | JoinType::LeftMark
                    | JoinType::Full
            );
    }
    plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || is_coalesce_batches(plan)
        || plan.is::<RepartitionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<SortPreservingMergeExec>()
        || plan.is::<UnionExec>()
        || plan.is::<SortExec>()
        || plan.is::<AggregateExec>()
        || plan.is::<GlobalLimitExec>()
        || plan.is::<LocalLimitExec>()
        || plan.is::<CrossJoinExec>()
        || plan.is::<NestedLoopJoinExec>()
}

/// Whether an expression of `plan` itself (not of its children) reads the
/// dynamic filter with id `filter`, e.g. the predicate of a scan. A node whose
/// expressions cannot be listed may read it.
fn reads_dynamic_filter(plan: &Arc<dyn ExecutionPlan>, filter: u64) -> bool {
    let mut reads = false;
    let listed = plan.apply_expressions(&mut |expr| {
        expr.apply(|expr| {
            reads |= expr.expression_id() == Some(filter);
            Ok(match reads {
                true => TreeNodeRecursion::Stop,
                false => TreeNodeRecursion::Continue,
            })
        })
    });
    reads || listed.is_err()
}

/// Whether `plan` reads all rows of its first child before it emits any row,
/// in DataFusion and natively, so that nothing above `plan` sees a row of
/// the nodes below that child before they read all of their input (or
/// stopped reading it):
///
/// * an aggregation that merges or aggregates all rows of its groups (in
///   `Final`, `FinalPartitioned`, `Single` or `SinglePartitioned` mode)
///   without knowing when a group is complete (no clustering of its input by
///   its groups, see `GroupClusteringMode`), which emits its groups once its
///   input ends (or stops reading it once it has as many groups as its
///   limit);
/// * a sort whose input is not sorted by its keys (DataFusion passes a sorted
///   input through), which emits its rows once its input ends (a TopK whose
///   input is sorted by a prefix of its keys may stop reading it before);
/// * a hash join, a cross join and a nested loop join, which collect their
///   build side (the first child) before they probe it;
///
/// a partition of which reads the corresponding partition of its first child
/// first, if no partition of the child ends before the nodes below it that
/// read the filter read all of their input (see [`ends_after_its_readers`]). The native aggregations, sorts and joins
/// read all of these inputs before they emit any row too: they are pipeline
/// breakers.
fn reads_first_child_before_emitting(plan: &Arc<dyn ExecutionPlan>, filter: u64) -> bool {
    let reads_first = if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        matches!(
            aggregate.mode(),
            AggregateMode::Final
                | AggregateMode::FinalPartitioned
                | AggregateMode::Single
                | AggregateMode::SinglePartitioned
        ) && aggregate.group_clustering_mode() == &GroupClusteringMode::None
    } else if let Some(sort) = plan.downcast_ref::<SortExec>() {
        !sort
            .input()
            .equivalence_properties()
            .ordering_satisfy(sort.expr().clone())
            .unwrap_or(true)
    } else {
        plan.is::<HashJoinExec>()
            || plan.is::<CrossJoinExec>()
            || plan.is::<NestedLoopJoinExec>()
    };
    reads_first
        && plan
            .children()
            .first()
            .is_some_and(|child| ends_after_its_readers(child, filter))
}

/// Whether no partition of `plan` ends before the nodes below it that read
/// the dynamic filter with id `filter` read all of their input: there is no
/// such node, or `plan` has a single partition, or it is a repartitioning or a
/// merge of partitions, every output partition of which ends with the last of
/// its input partitions, or a node that reads every partition of its input to
/// its end before it ends the corresponding partition of its own (projections,
/// filters and coalesced batches without a fetch, aggregations that merge or
/// aggregate all rows of their groups, sorts) above such an input, or a join
/// whose nodes that read the filter are all below its build side (the first
/// child), which it reads all of before it probes it (or skips its probe side),
/// above such a build side. A node that reads all rows of a partition of
/// `plan` before it emits any row then emits no row before the nodes that read
/// the filter below `plan` read all of their input.
fn ends_after_its_readers(plan: &Arc<dyn ExecutionPlan>, filter: u64) -> bool {
    if !consumes_dynamic_filter(plan, filter) {
        return true;
    }
    if plan.output_partitioning().partition_count() <= 1
        || plan.is::<RepartitionExec>()
        || plan.is::<CoalescePartitionsExec>()
    {
        return true;
    }
    if reads_dynamic_filter(plan, filter) {
        return false;
    }
    let children = plan.children();
    let reads_partitions_to_their_end = ((plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || is_coalesce_batches(plan))
        && plan.fetch().is_none())
        || plan.is::<SortExec>()
        || plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|aggregate| {
                !matches!(
                    aggregate.mode(),
                    AggregateMode::Partial | AggregateMode::PartialReduce
                ) && aggregate.group_clustering_mode() == &GroupClusteringMode::None
            });
    if reads_partitions_to_their_end {
        return children.len() == 1 && ends_after_its_readers(children[0], filter);
    }
    if plan.is::<HashJoinExec>()
        || plan.is::<CrossJoinExec>()
        || plan.is::<NestedLoopJoinExec>()
    {
        return match children.as_slice() {
            [build, probe] => {
                !consumes_dynamic_filter(probe, filter)
                    && ends_after_its_readers(build, filter)
            }
            _ => false,
        };
    }
    false
}

/// Whether `plan` passes on the rows of its inputs as they reach it, without
/// combining them with other rows and without a fetch (a node with a fetch
/// stops after a number of rows that pass a filter below it, which depend on
/// the rows that the filter prunes): a scan, a projection, a filter, a union,
/// or a merge or repartitioning of partitions, as `crate::sort::try_lower`
/// judges the nodes between a TopK and the nodes that read its dynamic
/// filter.
fn passes_rows_on(plan: &Arc<dyn ExecutionPlan>) -> bool {
    (plan.is::<DataSourceExec>()
        || plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || plan.is::<UnionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<RepartitionExec>())
        && plan.fetch().is_none()
}

/// Whether the nodes of `plan` (in its pipelines, see [`pipeline_children`])
/// have no part that may fail (see [`node_failures`]: also no source that may
/// fail to read its rows, but for scans of well-formed Parquet files, see
/// [`source_may_fail`]) and no expression whose errors or results depend on
/// the rows that it is evaluated on together (see [`depends_on_batches`],
/// e.g. a volatile function; the scans of files fold that into their
/// failures, see [`file_scan_failures`]): whichever of their rows a dynamic
/// filter prunes, the work of these nodes on the others is the same, and
/// cannot fail.
fn works_alike_on_any_rows(
    plan: &Arc<dyn ExecutionPlan>,
    runtime: Option<&RuntimeEnv>,
) -> bool {
    let batch_dependent = file_scan_failures(plan).is_none()
        && Judgement::of_plan(plan.as_ref()).batch_dependent;
    !batch_dependent
        && node_failures(plan, runtime).is_none()
        && pipeline_children(plan)
            .into_iter()
            .all(|child| works_alike_on_any_rows(child, runtime))
}

/// `plan`, a sort whose `SortExec` (see [`sort_of_chain`]) is a TopK, with
/// that `SortExec` replaced by the same TopK with a new dynamic filter, which
/// no node of its input reads (see `Lowering::try_lower_sort`).
fn with_unread_topk_filter(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(sort) = plan.downcast_ref::<SortExec>() {
        let topk = SortExec::new(sort.expr().clone(), Arc::clone(sort.input()))
            .with_preserve_partitioning(sort.preserve_partitioning())
            .with_fetch(sort.fetch());
        return Ok(Arc::new(topk));
    }
    let Some(child) = plan.children().into_iter().next() else {
        return internal_err!("{} is not a sort", plan.name());
    };
    let child = with_unread_topk_filter(child)?;
    Arc::clone(plan).replace_children(vec![child], recompute())
}

/// `lowered`, the native TopK of a plan whose `SortExec` has a dynamic filter
/// that no node reads (see [`with_unread_topk_filter`]), updating `filter`,
/// the dynamic filter of the `SortExec` of the plan that the nodes of its
/// input read, with the same keys instead: so that they skip the rows that
/// the threshold of the native TopK rules out. `None` for a TopK by float keys
/// that runs DataFusion's TopK if its result may differ (see
/// [`FloatKeyFallback`](crate::sort::FloatKeyFallback)), which would run the
/// plan whose filter no node reads, and for any other operator.
fn updating_filter(
    lowered: &Arc<dyn PhysicalOperator>,
    filter: Arc<DynamicFilterPhysicalExpr>,
) -> Result<Option<Arc<dyn PhysicalOperator>>> {
    let Some(topk) = lowered.downcast_ref::<PhysicalTopK>() else {
        return Ok(None);
    };
    if topk.float_key_fallback().is_some() {
        return Ok(None);
    }
    let mut updating = PhysicalTopK::try_new(
        Arc::clone(topk.input()),
        topk.ordering().clone(),
        topk.k(),
    )?;
    if let Some(dynamic_filter) = topk.dynamic_filter() {
        updating = updating.with_dynamic_filter(TopKDynamicFilter::new(
            filter,
            dynamic_filter.keys().to_vec(),
        ))?;
    }
    Ok(Some(Arc::new(updating)))
}

/// Whether DataFusion may fail to combine the batches of `plan`, a node that
/// it may execute. Every node but those that pass their batches on one by
/// one (projections, merges of partitions, unions, limits, scalar subqueries,
/// and the leaves, see [`source_may_fail`]) may concatenate, coalesce, merge or interleave
/// batches: e.g. `FilterExec` and `RepartitionExec` coalesce their output, a
/// sort merges its sorted runs, and a hash join concatenates its build side.
/// That may fail for types whose concatenation may fail (see
/// [`may_fail_to_concatenate`]): of the output of a `FilterExec`, which
/// coalesces it after its projection, and of the output or an input of any
/// other such node.
fn may_fail_to_combine(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.children().is_empty()
        || plan.is::<ProjectionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<UnionExec>()
        || plan.is::<GlobalLimitExec>()
        || plan.is::<LocalLimitExec>()
        || plan.is::<ScalarSubqueryExec>()
    {
        return false;
    }
    concatenation_may_fail(&plan.schema())
        || (!plan.is::<FilterExec>()
            && plan
                .children()
                .iter()
                .any(|child| concatenation_may_fail(&child.schema())))
}

/// Whether `plan` is a source that may fail to read or decode its rows: every
/// leaf but a scan of memory, of an empty relation or of placeholder rows, and
/// a scan of Parquet files whose expressions (its predicate and its
/// projection) cannot fail (see [`file_scan_failures`]), e.g. those of TPC-H
/// and ClickBench, and whose files have the types of its table (see
/// [`reads_files_of_the_types_of_their_table`]): such files are read without
/// failing, whichever of their files, row groups and rows the dynamic filters
/// of the operators above skip. A Parquet file whose columns the scan casts into
/// the types of its table (e.g. a file of strings for a table of integers,
/// whose schema was given when it was registered) may fail to be read. Scans of
/// CSV files may fail to parse any value of their rows (also of files whose
/// schema DataFusion inferred, from their first rows), and other scans of
/// files, which may fail to read or parse them, and streams may fail too.
fn source_may_fail(plan: &Arc<dyn ExecutionPlan>, runtime: Option<&RuntimeEnv>) -> bool {
    let well_formed_parquet = file_scan_failures(plan).is_some_and(|failures| {
        failures.is_none()
            && plan
                .downcast_ref::<DataSourceExec>()
                .and_then(|scan| scan.data_source().downcast_ref::<FileScanConfig>())
                .is_some_and(|config| {
                    reads_files_of_the_types_of_their_table(config, runtime)
                })
    });
    plan.children().is_empty()
        && !(plan.is::<EmptyExec>()
            || plan.is::<PlaceholderRowExec>()
            || plan
                .downcast_ref::<DataSourceExec>()
                .is_some_and(|scan| scan.data_source().is::<MemorySourceConfig>())
            || well_formed_parquet)
}

/// Whether the files of `config`, a scan of Parquet files in a session with
/// the runtime `runtime`, have the columns of the table of the scan, of its
/// types, so that the scan reads them without casting any value: as their
/// metadata (their footers, which DataFusion caches in the file metadata cache
/// of the runtime when it collects their statistics, see
/// `datafusion.execution.collect_statistics`) shows, with the coercions of
/// DataFusion's Parquet reader: of strings into `Utf8View` (see
/// `apply_file_schema_type_coercions`), and of INT96 timestamps into the unit
/// of `coerce_int96` (see `Int96Coercer`), from which the scan casts them into
/// the unit of the table, which may overflow. (Not its coercion of strings into
/// dictionaries: such files may fail.) The schema of a table that DataFusion
/// inferred from its files has their types, but a schema given when the table
/// was registered may have others, into which the scan casts the values of the
/// files (and fails on those that it cannot cast), or no nulls where the files
/// have nulls. `false` if the metadata of a file is not cached (or not for its
/// current version), if the runtime is not known, and for scans with a custom
/// adapter of their expressions to the schemas of their files.
fn reads_files_of_the_types_of_their_table(
    config: &FileScanConfig,
    runtime: Option<&RuntimeEnv>,
) -> bool {
    use datafusion_datasource_parquet::metadata::CachedParquetMetaData;
    use datafusion_datasource_parquet::{Int96Coercer, apply_file_schema_type_coercions};
    let Some(runtime) = runtime else {
        return false;
    };
    let Some(source) = config.file_source().downcast_ref::<ParquetSource>() else {
        return false;
    };
    // A custom adapter of the expressions of the scan to the schemas of its
    // files may rewrite them otherwise than casting columns.
    if config.expr_adapter_factory.is_some() {
        return false;
    }
    let options = &source.table_parquet_options().global;
    let int96 = options.coerce_int96.map(arrow::datatypes::TimeUnit::from);
    let int96_timezone = options
        .coerce_int96_tz
        .as_ref()
        .map(|timezone| Arc::<str>::from(timezone.as_str()));
    let cache = runtime.cache_manager.get_file_metadata_cache();
    let table = config.file_schema();
    let has_the_types_of_the_table = |file: &datafusion_datasource::PartitionedFile| {
        let Some(entry) = cache.get(&file.object_meta.location) else {
            return false;
        };
        if !entry.is_valid_for(&file.object_meta) {
            return false;
        }
        let Some(metadata) = entry
            .file_metadata
            .as_any()
            .downcast_ref::<CachedParquetMetaData>()
            .map(CachedParquetMetaData::parquet_metadata)
        else {
            return false;
        };
        let file_metadata = metadata.file_metadata();
        let Ok(file_schema) = parquet::arrow::parquet_to_arrow_schema(
            file_metadata.schema_descr(),
            file_metadata.key_value_metadata(),
        ) else {
            return false;
        };
        let mut file_schema =
            apply_file_schema_type_coercions(table, &file_schema).unwrap_or(file_schema);
        if let Some(unit) = &int96
            && let Some(coerced) =
                Int96Coercer::new(file_metadata.schema_descr(), &file_schema, unit)
                    .with_timezone(int96_timezone.clone())
                    .coerce()
        {
            file_schema = coerced;
        }
        // Of the type of the table, and without nulls if the table has none.
        table.fields().iter().all(|field| {
            file_schema
                .field_with_name(field.name())
                .is_ok_and(|file_field| {
                    file_field.data_type() == field.data_type()
                        && (field.is_nullable() || !file_field.is_nullable())
                })
        })
    };
    config
        .file_groups
        .iter()
        .flat_map(|group| group.iter())
        .all(has_the_types_of_the_table)
}

/// The parts that may fail (see [`failures`]) of the native operators that
/// evaluate the chunks of the pipeline of `operator`, from `operator` down to
/// the source of the pipeline: the filters, the projections and the probe
/// keys and filters of hash joins, through the operators that pass on the
/// chunks of their inputs (caching, limits, unions, the probe side of joins,
/// and the inputs of scalar subqueries), down to the sources and pipeline
/// breakers, which belong to other pipelines (and bridged plans, which
/// DataFusion evaluates on whole batches).
fn native_failures(operator: &Arc<dyn PhysicalOperator>) -> Failures {
    if let Some(join) = operator.downcast_ref::<PhysicalHashJoin>() {
        let probe_schema = join.probe_side().schema();
        let mut own = join
            .on()
            .iter()
            .map(|(_, key)| failures(key, &probe_schema))
            .fold(Failures::NONE, Failures::and);
        if let Some(filter) = join.filter() {
            own = own.and(failures(filter.expression(), filter.schema()));
        }
        return own.and(native_failures(join.probe_side()));
    }
    if let Some(join) = operator.downcast_ref::<PhysicalNestedLoopJoin>() {
        return native_failures(join.probe_side());
    }
    if let Some(join) = operator.downcast_ref::<PhysicalCrossJoin>() {
        return native_failures(join.probe_side());
    }
    if let Some(subquery) = operator.downcast_ref::<PhysicalScalarSubquery>() {
        return native_failures(subquery.input());
    }
    if operator.is_source() || operator.is_sink() {
        return Failures::NONE;
    }
    let input_schema = || operator.children()[0].schema();
    let own = if let Some(filter) = operator.downcast_ref::<PhysicalFilter>() {
        failures(filter.predicate(), &input_schema())
    } else if let Some(projection) = operator.downcast_ref::<PhysicalProjection>() {
        let input_schema = input_schema();
        projection
            .exprs()
            .iter()
            .map(|(expr, _)| failures(expr, &input_schema))
            .fold(Failures::NONE, Failures::and)
    } else {
        Failures::NONE
    };
    operator
        .children()
        .iter()
        .map(native_failures)
        .fold(own, Failures::and)
}

/// Whether `plan` produces at most one row, whatever its input: an
/// aggregation without groups of a single output partition (which produces
/// exactly one row), a global limit of at most one row, an empty relation or a
/// placeholder row of a single partition, or a node that passes on some of the
/// rows of its input (a projection, a filter, a sort, a limit, a
/// redistribution, or a scalar subquery) over such an input.
fn produces_at_most_one_row(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let single_partition = plan.output_partitioning().partition_count() == 1;
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return aggregate.group_expr().is_empty() && single_partition;
    }
    if plan.is::<EmptyExec>()
        || (plan.is::<PlaceholderRowExec>() && single_partition)
        || plan
            .downcast_ref::<GlobalLimitExec>()
            .is_some_and(|limit| limit.fetch().is_some_and(|fetch| fetch <= 1))
    {
        return true;
    }
    let passes_rows = plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || plan.is::<SortExec>()
        || plan.is::<SortPreservingMergeExec>()
        || plan.is::<GlobalLimitExec>()
        || plan.is::<LocalLimitExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<RepartitionExec>()
        || is_coalesce_batches(plan)
        || plan.is::<ScalarSubqueryExec>();
    passes_rows
        && pipeline_children(plan)
            .first()
            .is_some_and(|input| produces_at_most_one_row(input))
}

/// The native operator of an uncorrelated scalar subquery (DataFusion's
/// `ScalarSubqueryExec`): it runs the plans of its subqueries, and then its
/// input, whose rows it passes on, and which evaluates the results of the
/// subqueries (`ScalarSubqueryExpr`s, `scalar_subquery(<pending>)`).
///
/// `ScalarSubqueryExec` runs all of its subqueries when its output is first
/// polled, before it executes its input, and populates the
/// `ScalarSubqueryResults` that the `ScalarSubqueryExpr`s of its input read
/// with their results: the value of their single row, a null of their type if
/// they produce no row, and an error if they produce more than one row. A
/// [`PhysicalScalarSubquery`] does the same with pipelines:
///
/// * every subquery is built into a child meta pipeline, whose sink (a
///   `ScalarSubqueryResult`) collects its row and populates its result in the
///   `ScalarSubqueryResults` of the `ScalarSubqueryExec` when it is
///   finalized, and which runs in parallel with those of the other
///   subqueries, as in DataFusion;
/// * the input continues the pipeline of the operator, and all pipelines of
///   the input (also those of the pipeline breakers below it, see
///   [`MetaPipelineBuilder::build_with_dependency`]) wait for all subqueries
///   to be finalized, so that no expression of the input is evaluated before
///   the results are known, and the errors of the subqueries come first, as
///   in DataFusion;
/// * the results of the subqueries are operators of the pipeline of the
///   input, which pass its chunks on, so that the subqueries are canceled
///   when the input does not run (e.g. when an inner join above cannot produce
///   output), as in DataFusion, which then does not poll the
///   `ScalarSubqueryExec`.
///
/// The operator itself takes part in no pipeline. Every execution runs the
/// subqueries again, but keeps the results that an execution of the same plan
/// populated before (the `ScalarSubqueryExec` of DataFusion does not run its
/// subqueries again, and keeps their results, until the plan is reset, see
/// `ExecutionPlan::reset_state`, which clears them).
#[derive(Debug)]
pub struct PhysicalScalarSubquery {
    /// The input, followed by the results of the subqueries.
    children: Vec<Arc<dyn PhysicalOperator>>,
}

impl PhysicalScalarSubquery {
    /// The scalar subquery of the operator tree `input`, which reads the
    /// results of `subqueries`.
    fn try_new(
        input: Arc<dyn PhysicalOperator>,
        subqueries: Vec<Arc<ScalarSubqueryResult>>,
    ) -> Result<Self> {
        if let Some(subquery) = subqueries
            .iter()
            .find(|subquery| subquery.schema() != input.schema())
        {
            return internal_err!(
                "The result of scalar subquery {} passes on rows of another schema than the input",
                subquery.index.as_usize()
            );
        }
        let children = std::iter::once(input)
            .chain(
                subqueries
                    .into_iter()
                    .map(|subquery| subquery as Arc<dyn PhysicalOperator>),
            )
            .collect();
        Ok(Self { children })
    }

    /// The input, whose rows the operator passes on.
    pub fn input(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The results of the subqueries, the sinks of their pipelines; their
    /// children are the plans of the subqueries.
    pub fn subqueries(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children[1..]
    }
}

impl PhysicalOperator for PhysicalScalarSubquery {
    fn name(&self) -> &str {
        "PhysicalScalarSubquery"
    }

    fn schema(&self) -> SchemaRef {
        self.input().schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!(
            "PhysicalScalarSubquery: subqueries={}",
            self.subqueries().len()
        )
    }

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        // The results pass the chunks of the input on (operators are added
        // from the sink side), and the pipelines of the input read them.
        for subquery in self.subqueries() {
            meta.add_operator(current, Arc::clone(subquery))?;
        }
        let mut dependencies = Vec::with_capacity(self.subqueries().len());
        for subquery in self.subqueries() {
            let (mut child, mut pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(subquery))?;
            dependencies.push(child.id());
            let [plan] = subquery.children() else {
                return internal_err!("The result of a scalar subquery has one child");
            };
            child.build(&mut pipeline, plan)?;
        }
        build_waiting_for(meta, current, self.input(), &dependencies)
    }
}

/// Build `input` into `current`, so that every pipeline that this continues
/// or creates waits for the meta pipelines `dependencies` to be finalized
/// (see [`MetaPipelineBuilder::build_with_dependency`], which waits for one
/// of them).
fn build_waiting_for(
    meta: &mut MetaPipelineBuilder<'_>,
    current: &mut PipelineBuilder,
    input: &Arc<dyn PhysicalOperator>,
    dependencies: &[MetaPipelineId],
) -> Result<()> {
    match dependencies {
        [] => meta.build(current, input),
        [dependency] => meta.build_with_dependency(current, input, *dependency),
        [dependency, others @ ..] => {
            let waiting: Arc<dyn PhysicalOperator> = Arc::new(WaitingInput {
                input: Arc::clone(input),
                dependencies: others.to_vec(),
            });
            meta.build_with_dependency(current, &waiting, *dependency)
        }
    }
}

/// `input`, built into the pipeline of the operator above it so that its
/// pipelines wait for the meta pipelines `dependencies` (see
/// [`build_waiting_for`]). It only exists while the pipelines are built, and
/// takes part in none of them.
#[derive(Debug)]
struct WaitingInput {
    input: Arc<dyn PhysicalOperator>,
    dependencies: Vec<MetaPipelineId>,
}

impl PhysicalOperator for WaitingInput {
    fn name(&self) -> &str {
        "WaitingInput"
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        std::slice::from_ref(&self.input)
    }

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        build_waiting_for(meta, current, &self.input, &self.dependencies)
    }
}

/// The result of a scalar subquery of a [`PhysicalScalarSubquery`]: the sink
/// of the pipelines of the subquery (its only child), which collects its row
/// and populates its result in the `ScalarSubqueryResults` of the
/// `ScalarSubqueryExec` when it is finalized, exactly as
/// `ScalarSubqueryExec` does (see `execute_scalar_subquery`): the value of
/// the row (`ScalarValue::try_from_array`), a null of the type of the
/// subquery if it has no row, and the error `Scalar subquery returned more
/// than one row` once it read a second row (its pipelines then stop, as
/// DataFusion stops reading the subquery). And an operator of the pipeline of
/// the input of the scalar subquery, which passes its chunks on.
#[derive(Debug)]
struct ScalarSubqueryResult {
    /// The plan of the subquery.
    children: Vec<Arc<dyn PhysicalOperator>>,
    /// The schema of the chunks of the input of the scalar subquery, which
    /// it passes on.
    schema: SchemaRef,
    /// The slot of the result in `results`.
    index: SubqueryIndex,
    /// The results of the subqueries of the `ScalarSubqueryExec`, which the
    /// `ScalarSubqueryExpr`s of its input read.
    results: ScalarSubqueryResults,
}

impl ScalarSubqueryResult {
    fn try_new(
        subquery: Arc<dyn PhysicalOperator>,
        schema: SchemaRef,
        index: SubqueryIndex,
        results: ScalarSubqueryResults,
    ) -> Result<Self> {
        if subquery.schema().fields().len() != 1 {
            return internal_err!(
                "Scalar subquery must return exactly one column, got {}",
                subquery.schema().fields().len()
            );
        }
        Ok(Self {
            children: vec![subquery],
            schema,
            index,
            results,
        })
    }
}

/// The global sink state of a [`ScalarSubqueryResult`]: the number of rows
/// of the subquery, and the value of the first one.
#[derive(Debug, Default)]
struct SubqueryRows {
    rows: parking_lot::Mutex<(usize, Option<ScalarValue>)>,
}

impl GlobalSinkState for SubqueryRows {}

impl PhysicalOperator for ScalarSubqueryResult {
    fn name(&self) -> &str {
        "ScalarSubqueryResult"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!("ScalarSubqueryResult: index={}", self.index.as_usize())
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        *output = input.clone();
        Ok(OperatorResult::NeedMoreInput)
    }

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        Ok(Arc::new(SubqueryRows::default()))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        _lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<SubqueryRows>(gstate.as_any(), self.name())?;
        let mut rows = state.rows.lock();
        let (count, value) = &mut *rows;
        let first = *count == 0;
        *count = count.saturating_add(input.len());
        // DataFusion fails once it read a second row, without reading more.
        if *count > 1 {
            return Ok(SinkResult::Finished);
        }
        if first {
            *value = Some(ScalarValue::try_from_array(&input.compacted_column(0)?, 0)?);
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<SubqueryRows>(gstate.as_any(), self.name())?;
        let (count, value) = &mut *state.rows.lock();
        if *count > 1 {
            return exec_err!("Scalar subquery returned more than one row");
        }
        let value = match value.take() {
            Some(value) => value,
            None => {
                ScalarValue::try_from(self.children[0].schema().field(0).data_type())?
            }
        };
        // An earlier (or concurrent) execution of the plan may have populated
        // the result, which DataFusion keeps.
        if self.results.get(self.index).is_none()
            && let Err(error) = self.results.set(self.index, value)
            && self.results.get(self.index).is_none()
        {
            return Err(error);
        }
        Ok(SinkFinalize::Ready)
    }

    fn build_pipelines(
        &self,
        _current: &mut PipelineBuilder,
        _meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        internal_err!(
            "The result of a scalar subquery is built by PhysicalScalarSubquery"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{AsArray, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::{
        CsvReadOptions, ParquetReadOptions, SessionConfig, SessionContext,
    };
    use datafusion_common::tree_node::TreeNode;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::{BinaryExpr, Literal, col};
    use datafusion_physical_plan::aggregates::PhysicalGroupBy;
    use datafusion_physical_plan::joins::PartitionMode;
    use datafusion_physical_plan::sorts::partial_sort::PartialSortExec;
    use datafusion_physical_plan::{collect, displayable};

    use crate::bridge::bridged_plans;
    use crate::explain::format_operator_tree;
    use crate::sort::{PhysicalSort, PhysicalTopK};
    use crate::sources::test_harness::{write_csv, write_parquet};
    use crate::sources::{CsvScanSource, MemoryScanSource, ParquetScanSource};
    use crate::test_utils::{
        compare_outputs, datafusion_context, pipeline_context, register_test_tables,
        run_engines, test_session_config, test_table_t1,
    };

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    /// The number of operators of type `T` in the operator tree `operator`.
    fn count<T: PhysicalOperator>(operator: &Arc<dyn PhysicalOperator>) -> usize {
        usize::from(operator.is::<T>())
            + operator.children().iter().map(count::<T>).sum::<usize>()
    }

    /// The number of nodes named `name` in the plan `plan`.
    fn count_nodes(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        let mut nodes = 0;
        plan.apply(|node| {
            nodes += usize::from(node.name() == name);
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        nodes
    }

    async fn setup_test_tables(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)
    }

    /// The tables of [`register_test_tables`], with their `Utf8` columns as
    /// `Utf8View`, the string type of the tables of DataFusion's SQL and of
    /// its Parquet scans: the nodes that coalesce batches (e.g. a
    /// `RepartitionExec`) may fail to concatenate `Utf8` columns, so the
    /// planner leaves them, and the nodes that remove them, to DataFusion.
    fn register_view_tables(ctx: &SessionContext) -> Result<()> {
        let view = |partitions: Vec<Vec<RecordBatch>>| -> Result<_> {
            let schema = partitions[0][0].schema();
            let fields = schema
                .fields()
                .iter()
                .map(|field| match field.data_type() {
                    DataType::Utf8 => {
                        field.as_ref().clone().with_data_type(DataType::Utf8View)
                    }
                    _ => field.as_ref().clone(),
                })
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new(fields));
            let partitions = partitions
                .into_iter()
                .map(|batches| {
                    batches
                        .into_iter()
                        .map(|batch| {
                            let columns = batch
                                .columns()
                                .iter()
                                .zip(schema.fields())
                                .map(|(column, field)| {
                                    arrow::compute::cast(column, field.data_type())
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((schema, partitions))
        };
        let table =
            |schema, partitions| -> Result<Arc<datafusion::datasource::MemTable>> {
                Ok(Arc::new(datafusion::datasource::MemTable::try_new(
                    schema, partitions,
                )?))
            };
        let (schema, t1) = view(test_table_t1()?)?;
        ctx.register_table("t1", table(Arc::clone(&schema), t1)?)?;
        let (t2_schema, t2) = view(crate::test_utils::test_table_t2()?)?;
        ctx.register_table("t2", table(t2_schema, t2)?)?;
        ctx.register_table("t_empty", table(schema, vec![vec![]])?)?;
        Ok(())
    }

    async fn setup_view_tables(ctx: &SessionContext) -> Result<()> {
        register_view_tables(ctx)
    }

    /// The plan of `sql` in default DataFusion, in a session configured with
    /// `config` with the tables of [`register_test_tables`].
    async fn datafusion_plan(sql: &str, config: SessionConfig) -> Arc<dyn ExecutionPlan> {
        let ctx = datafusion_context(config);
        register_test_tables(&ctx).unwrap();
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    /// The plan of `sql` in default DataFusion, in a session configured with
    /// `config` and set up with `setup`.
    async fn datafusion_plan_with<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn ExecutionPlan>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = datafusion_context(config);
        setup(&ctx).await.unwrap();
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    /// Assert that `sql` produces byte-identical results through the
    /// [`PipelineQueryPlanner`] and on default DataFusion (see
    /// `assert_byte_eq_with_datafusion`), in sessions configured with `config`
    /// and set up with `setup`. Returns the operator tree of the pipeline
    /// engine.
    async fn assert_parity_with<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let (expected, actual) =
            run_engines(sql, config, setup).await.unwrap_or_else(|e| {
                panic!("Failed to run query on both engines: {e}\n{sql}")
            });
        let root = actual
            .plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan")
            .operator();
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!(
                "Results differ: {message}\nquery: {sql}\nlowered: {}",
                format_operator_tree(root)
            );
        }
        Arc::clone(root)
    }

    /// [`assert_parity_with`] with the default tables and configuration.
    async fn assert_parity(sql: &str) -> Arc<dyn PhysicalOperator> {
        assert_parity_with(sql, test_session_config(), setup_test_tables).await
    }

    /// Assert that no part of the operator tree `root` of `sql` is executed by
    /// DataFusion: every node is native, so nothing is shuffled through
    /// `RepartitionExec`s either.
    fn assert_native(sql: &str, root: &Arc<dyn PhysicalOperator>) {
        let bridged = bridged_plans(root);
        assert!(
            bridged.is_empty(),
            "Parts of the plan are left to DataFusion\nquery: {sql}\nlowered: {}",
            format_operator_tree(root)
        );
    }

    /// Assert that `sql` either produces byte-identical results through the
    /// [`PipelineQueryPlanner`] and on default DataFusion (if `error` is
    /// `None`), or fails on both with an error that contains `error`.
    async fn assert_outcome(sql: &str, error: Option<&str>) {
        let Some(error) = error else {
            assert_parity(sql).await;
            return;
        };
        for ctx in [
            datafusion_context(test_session_config()),
            pipeline_context(test_session_config()),
        ] {
            register_test_tables(&ctx).unwrap();
            let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await;
            let outcome = match plan {
                Ok(plan) => collect(plan, ctx.task_ctx()).await,
                Err(e) => Err(e),
            };
            match outcome {
                Ok(_) => panic!("{sql} succeeds, expected an error: {error}"),
                Err(e) => assert!(e.to_string().contains(error), "{sql}: {e}"),
            }
        }
    }

    /// A `CoalesceBatchesExec` of `input` (which DataFusion does not plan any
    /// more), with a limit of `fetch` rows.
    #[expect(deprecated)]
    fn coalesce_batches(
        input: Arc<dyn ExecutionPlan>,
        fetch: Option<usize>,
    ) -> datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec {
        datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec::new(input, 8192)
            .with_fetch(fetch)
    }

    /// A configuration with which DataFusion plans `Partitioned` hash joins,
    /// with a `RepartitionExec(Hash)` on both sides.
    fn partitioned_config() -> SessionConfig {
        let mut config = test_session_config();
        let optimizer = &mut config.options_mut().optimizer;
        optimizer.hash_join_single_partition_threshold = 0;
        optimizer.hash_join_single_partition_threshold_rows = 0;
        config
    }

    // ---------------------------------------------------------------------
    // Lowering of the native operators
    // ---------------------------------------------------------------------

    #[test]
    fn unsupported_plans_are_bridged() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let scan: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?;
        let ordering =
            LexOrdering::new(vec![PhysicalSortExpr::new_default(col("a", &schema)?)])
                .unwrap();
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(PartialSortExec::new(ordering, scan, 1));
        let operator = lower_execution_plan(&plan)?;
        let bridge = operator.downcast_ref::<BridgeOperator>().unwrap();
        assert!(Arc::ptr_eq(bridge.plan(), &plan));

        let root = create_pipeline_plan(Arc::clone(&plan))?;
        assert_eq!(root.schema(), schema);
        assert!(Arc::ptr_eq(root.input(), &plan));
        assert!(root.operator().is::<BridgeOperator>());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn plans_whose_native_operators_fail_are_bridged() -> Result<()> {
        // A limit whose input was replaced by an input with another schema,
        // without recomputing its properties: the native limit refuses an
        // input of another schema, DataFusion executes it.
        let schema = |name: &str| {
            Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, true)]))
        };
        let batch = RecordBatch::try_new(
            schema("b"),
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )?;
        let make_plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let limit: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(
                MemorySourceConfig::try_new_exec(&[vec![]], schema("a"), None)?,
                2,
                Some(3),
            ));
            let input = MemorySourceConfig::try_new_exec(
                &[vec![batch.clone()]],
                schema("b"),
                None,
            )?;
            limit.replace_children(
                vec![input],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Keep),
            )
        };
        let plan = make_plan()?;
        assert_eq!(plan.schema(), schema("a"));
        assert!(PhysicalLimit::try_lower(&plan, lower_execution_plan).is_err());
        let root = lower_execution_plan(&plan)?;
        let bridge = root.downcast_ref::<BridgeOperator>().expect("a bridge");
        assert!(Arc::ptr_eq(bridge.plan(), &plan));

        let task_ctx = Arc::new(TaskContext::default());
        let expected = collect(make_plan()?, Arc::clone(&task_ctx)).await?;
        let root: Arc<dyn ExecutionPlan> = create_pipeline_plan(make_plan()?)?;
        let actual = collect(root, task_ctx).await?;
        assert_eq!(actual, expected);
        assert_eq!(actual.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_stage_aggregations_are_one_operator() {
        // The queries, the number of `AggregateExec`s DataFusion plans for
        // them (two stages per aggregation of several partitions) and the
        // number of native aggregations.
        for (sql, stages, aggregations) in [
            (
                "SELECT COUNT(*), COUNT(a), SUM(b), MIN(c), MAX(d), AVG(g) FROM t1",
                2,
                1,
            ),
            (
                "SELECT d, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh, MAX(f) AS mf \
                 FROM t1 GROUP BY d",
                2,
                1,
            ),
            ("SELECT DISTINCT d, e FROM t1", 2, 1),
            (
                "SELECT a, COUNT(*) AS n FROM t1 GROUP BY a HAVING COUNT(*) > 50",
                2,
                1,
            ),
            (
                "SELECT d, e, SUM(g) AS sg, COUNT(*) AS n FROM t1 GROUP BY ROLLUP (d, e)",
                2,
                1,
            ),
            // `COUNT(DISTINCT)` is an aggregation of an aggregation.
            (
                "SELECT e, f, COUNT(DISTINCT a) AS da FROM t1 GROUP BY e, f",
                4,
                2,
            ),
            (
                "SELECT n, COUNT(*) AS groups FROM \
                 (SELECT a, COUNT(*) AS n FROM t1 GROUP BY a) GROUP BY n",
                4,
                2,
            ),
            // A single partition is aggregated in a single stage.
            ("SELECT COUNT(*), SUM(a) FROM t_empty", 1, 1),
        ] {
            let plan = datafusion_plan(sql, test_session_config()).await;
            assert_eq!(
                count_nodes(&plan, "AggregateExec"),
                stages,
                "{sql}\n{}",
                displayable(plan.as_ref()).indent(false)
            );
            let root =
                assert_parity_with(sql, test_session_config(), setup_view_tables).await;
            assert_native(sql, &root);
            assert_eq!(
                count::<PhysicalHashAggregate>(&root),
                aggregations,
                "{sql}\n{}",
                format_operator_tree(&root)
            );
        }

        // Ordered aggregates: the native aggregation orders the values itself,
        // without the sort DataFusion plans below it. Their states are lists:
        // over several partitions, they pass through a repartitioning that may
        // fail to concatenate them (see `may_fail_to_concatenate`), so the
        // aggregation is native over a single partition only.
        let sql = "SELECT d, ARRAY_AGG(id ORDER BY id) AS ids, \
                   FIRST_VALUE(a ORDER BY id DESC) AS last_a, \
                   NTH_VALUE(h, 2 ORDER BY id) AS second_h FROM t1 GROUP BY d";
        let config = test_session_config().with_target_partitions(1);
        let plan = datafusion_plan(sql, config.clone()).await;
        assert_eq!(count_nodes(&plan, "SortExec"), 1);
        let root = assert_parity_with(sql, config, setup_view_tables).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
        assert_eq!(count::<PhysicalSort>(&root), 0);
        let root =
            assert_parity_with(sql, test_session_config(), setup_view_tables).await;
        assert_eq!(count::<PhysicalHashAggregate>(&root), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn hash_joins_are_one_operator_in_both_partition_modes() {
        for (config, mode) in [
            (test_session_config(), PartitionMode::CollectLeft),
            (partitioned_config(), PartitionMode::Partitioned),
        ] {
            for sql in [
                "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
                "SELECT t1.id, t2.w FROM t1 LEFT JOIN t2 ON t1.a = t2.k AND t2.w > 100",
                "SELECT t1.id, t2.k, t2.w FROM t1 RIGHT JOIN t2 ON t1.a = t2.k",
                "SELECT t1.id, t2.k FROM t1 FULL OUTER JOIN t2 ON t1.a = t2.k",
                "SELECT t2.k, t2.name FROM t2 WHERE k IN (SELECT a FROM t1 WHERE e)",
                "SELECT t2.k FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k)",
            ] {
                let plan = datafusion_plan(sql, config.clone()).await;
                let mut modes = vec![];
                plan.apply(|node| {
                    if let Some(join) = node.downcast_ref::<HashJoinExec>() {
                        modes.push(*join.partition_mode());
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .unwrap();
                assert_eq!(modes, vec![mode], "{sql}");
                if mode == PartitionMode::Partitioned {
                    // Both inputs are shuffled by hash in DataFusion.
                    assert_eq!(count_nodes(&plan, "RepartitionExec"), 2, "{sql}");
                }
                let root =
                    assert_parity_with(sql, config.clone(), setup_view_tables).await;
                assert_native(sql, &root);
                assert_eq!(count::<PhysicalHashJoin>(&root), 1, "{sql}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_and_merges_are_one_operator() {
        for (sql, sorts, topks, limits) in [
            ("SELECT * FROM t1 ORDER BY id", 1, 0, 0),
            ("SELECT id, c FROM t1 ORDER BY c NULLS FIRST, id", 1, 0, 0),
            ("SELECT id, a FROM t1 ORDER BY a + b, id", 1, 0, 0),
            (
                "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY d",
                1,
                0,
                0,
            ),
            ("SELECT id, a FROM t1 ORDER BY id DESC LIMIT 25", 0, 1, 0),
            (
                "SELECT id, h FROM t1 ORDER BY h DESC NULLS LAST, id LIMIT 100 OFFSET 7",
                0,
                1,
                1,
            ),
        ] {
            // DataFusion sorts every partition and merges the sorted
            // partitions.
            let plan = datafusion_plan(sql, test_session_config()).await;
            assert_eq!(count_nodes(&plan, "SortPreservingMergeExec"), 1, "{sql}");
            let root =
                assert_parity_with(sql, test_session_config(), setup_view_tables).await;
            assert_native(sql, &root);
            let tree = format_operator_tree(&root);
            assert_eq!(count::<PhysicalSort>(&root), sorts, "{sql}\n{tree}");
            assert_eq!(count::<PhysicalTopK>(&root), topks, "{sql}\n{tree}");
            assert_eq!(count::<PhysicalLimit>(&root), limits, "{sql}\n{tree}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_projections_unions_and_placeholders_are_native() {
        for sql in [
            "SELECT id, a + b AS s, c * 2 AS c2, NOT e AS ne FROM t1 \
             WHERE a > 10 AND d <> 'beta'",
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2",
            "SELECT 1 AS one, 2 AS two",
            "SELECT * FROM t1 WHERE false",
        ] {
            let root = assert_parity(sql).await;
            assert_native(sql, &root);
        }
        let sql = "SELECT id, a FROM t1 WHERE a > 10";
        let root = assert_parity(sql).await;
        assert_native(sql, &root);
        // The native filter is wrapped in a `CachingOperator`.
        let caching = root.downcast_ref::<CachingOperator>().expect("a cache");
        assert!(caching.children()[0].is::<PhysicalFilter>());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn complex_queries_run_natively() {
        for sql in [
            // A join, a filter, an aggregation and a sort.
            "SELECT t2.name, COUNT(*) AS n, SUM(t1.b) AS sb, MIN(t1.h) AS mh \
             FROM t1 JOIN t2 ON t1.a = t2.k WHERE t1.e GROUP BY t2.name ORDER BY t2.name",
            // Three joins, an aggregation and a sort.
            "SELECT a.k, b.name, COUNT(*) AS n, SUM(t1.b) AS sb FROM t2 a \
             JOIN t2 b ON a.k = b.k JOIN t1 ON t1.a = a.k \
             GROUP BY a.k, b.name ORDER BY a.k, b.name",
            // A join with a residual filter and an aggregation of a join.
            "SELECT t1.d, t2.name, COUNT(*) AS n, MAX(t1.ts) AS last \
             FROM t1 JOIN t2 ON t1.a = t2.k AND t1.b > t2.w GROUP BY t1.d, t2.name",
            "SELECT t2.k, COUNT(t1.id) AS n FROM t2 LEFT JOIN t1 ON t1.a = t2.k AND t1.e \
             GROUP BY t2.k ORDER BY t2.k",
            // Aggregations of aggregations, and of unions.
            "SELECT n, COUNT(*) AS groups, SUM(n) AS total FROM \
             (SELECT a, COUNT(*) AS n FROM t1 GROUP BY a) GROUP BY n ORDER BY n",
            // A semi join, a TopK and a limit.
            "SELECT id, a, h FROM t1 WHERE a IN (SELECT k FROM t2 WHERE w > 1000) \
             ORDER BY h DESC NULLS LAST, id LIMIT 20 OFFSET 3",
        ] {
            let root =
                assert_parity_with(sql, test_session_config(), setup_view_tables).await;
            assert_native(sql, &root);
        }
        // Aggregations whose states are lists: natively over a single
        // partition. Over several, their states pass through a repartitioning
        // that may fail to concatenate them (see `may_fail_to_concatenate`),
        // so the aggregations are left to DataFusion.
        for sql in [
            // Of a union.
            "SELECT k, COUNT(*) AS n, COUNT(DISTINCT w) AS dw FROM \
             (SELECT a AS k, b AS w FROM t1 UNION ALL SELECT k, w FROM t2) \
             GROUP BY k ORDER BY k",
            // Ordered aggregates of a join, without the sort of their input.
            "SELECT t1.e, ARRAY_AGG(t2.w ORDER BY t2.w DESC NULLS LAST, t1.id) AS ws \
             FROM t1 JOIN t2 ON t1.a = t2.k GROUP BY t1.e",
        ] {
            let config = test_session_config().with_target_partitions(1);
            let root = assert_parity_with(sql, config, setup_view_tables).await;
            assert_native(sql, &root);
            let root =
                assert_parity_with(sql, test_session_config(), setup_view_tables).await;
            assert_eq!(count::<PhysicalHashAggregate>(&root), 0, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn complex_queries_with_other_partitionings() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT t2.name, COUNT(*) AS n, SUM(t1.b) AS sb FROM t1 \
                 JOIN t2 ON t1.a = t2.k GROUP BY t2.name ORDER BY t2.name",
                "SELECT e, f, COUNT(DISTINCT a) AS da FROM t1 GROUP BY e, f",
                "SELECT id, a FROM t1 ORDER BY id DESC LIMIT 25",
            ] {
                let root =
                    assert_parity_with(sql, config.clone(), setup_view_tables).await;
                assert_native(sql, &root);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_and_loop_joins_are_native() {
        // The queries and the numbers of native windows, nested loop joins and
        // cross joins they are lowered to, with the native operators around
        // them: the sorts and the repartitionings below a window are part of
        // the native window, which partitions and sorts its input itself.
        for (sql, windows, nested_loops, crosses) in [
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
                1,
                0,
                0,
            ),
            (
                "SELECT id, SUM(b) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS s \
                 FROM t1",
                1,
                0,
                0,
            ),
            (
                "SELECT id, d, RANK() OVER (PARTITION BY d ORDER BY a DESC NULLS LAST, id) AS r, \
                 LAG(b, 2) OVER (PARTITION BY d ORDER BY a DESC NULLS LAST, id) AS l, \
                 FIRST_VALUE(h) OVER (PARTITION BY d ORDER BY a DESC NULLS LAST, id) AS f \
                 FROM t1 WHERE a > 3",
                1,
                0,
                0,
            ),
            // Windows of different partitions and orders, stacked on each other
            // (DataFusion sorts between them).
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn, \
                 COUNT(*) OVER (PARTITION BY e ORDER BY id) AS n FROM t1",
                2,
                0,
                0,
            ),
            // A window of an aggregation, an aggregation of a window, and a
            // window of a join.
            (
                "SELECT d, n, RANK() OVER (ORDER BY n DESC, d) AS r FROM \
                 (SELECT d, COUNT(*) AS n FROM t1 GROUP BY d)",
                1,
                0,
                0,
            ),
            (
                "SELECT e, COUNT(*) AS n, MAX(rn) AS m FROM \
                 (SELECT e, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1) \
                 GROUP BY e",
                1,
                0,
                0,
            ),
            (
                "SELECT t2.name, t1.id, SUM(t1.b) OVER (PARTITION BY t2.name ORDER BY t1.id) AS s \
                 FROM t1 JOIN t2 ON t1.a = t2.k",
                1,
                0,
                0,
            ),
            // Nested loop joins, of several join types, and cross joins.
            (
                "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
                0,
                1,
                0,
            ),
            (
                "SELECT t1.id, t2.k, t2.w FROM t1 LEFT JOIN t2 \
                 ON t1.a < t2.k AND t2.w > 1000 WHERE t1.id < 50",
                0,
                1,
                0,
            ),
            (
                "SELECT t2.k, t2.name FROM t2 WHERE EXISTS \
                 (SELECT 1 FROM t1 WHERE t1.a > t2.k AND t1.id < 100)",
                0,
                1,
                0,
            ),
            (
                "SELECT a.k, b.k, a.w FROM t2 a CROSS JOIN t2 b WHERE a.w < 0",
                0,
                0,
                1,
            ),
            // Aggregations, sorts and windows of loop joins.
            (
                "SELECT t2.k, COUNT(*) AS n, SUM(t1.b) AS sb FROM t1 JOIN t2 ON t1.a > t2.k \
                 GROUP BY t2.k ORDER BY t2.k",
                0,
                1,
                0,
            ),
            (
                "SELECT a.k, b.k AS bk, RANK() OVER (PARTITION BY a.k ORDER BY b.k) AS r \
                 FROM t2 a CROSS JOIN t2 b WHERE a.k < 5 AND b.k < 5 ORDER BY a.k, bk, r",
                1,
                0,
                1,
            ),
        ] {
            let root =
                assert_parity_with(sql, test_session_config(), setup_view_tables).await;
            assert_native(sql, &root);
            let tree = format_operator_tree(&root);
            assert_eq!(count::<PhysicalWindow>(&root), windows, "{sql}\n{tree}");
            assert_eq!(
                count::<PhysicalNestedLoopJoin>(&root),
                nested_loops,
                "{sql}\n{tree}"
            );
            assert_eq!(count::<PhysicalCrossJoin>(&root), crosses, "{sql}\n{tree}");
        }
        // The merge of the sorted partitions of a window (for an `ORDER BY` that
        // the window partitions satisfy) is a sort of the native window.
        for sql in [
            "SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1 \
             ORDER BY d, id",
            "SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1 \
             ORDER BY d, id LIMIT 7",
            "SELECT id, rn FROM (SELECT id, d, \
             ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1) ORDER BY d, id",
        ] {
            let plan = datafusion_plan(sql, test_session_config()).await;
            assert_eq!(count_nodes(&plan, "SortPreservingMergeExec"), 1, "{sql}");
            assert_eq!(count_nodes(&plan, "SortExec"), 1, "{sql}");
            let root =
                assert_parity_with(sql, test_session_config(), setup_view_tables).await;
            assert_native(sql, &root);
            let tree = format_operator_tree(&root);
            assert_eq!(count::<PhysicalWindow>(&root), 1, "{sql}\n{tree}");
            assert_eq!(count::<PhysicalSort>(&root), 1, "{sql}\n{tree}");
        }
        // On one thread, and on more threads than partitions.
        for target_partitions in [1, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
                "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
                "SELECT a.k, b.k, a.w FROM t2 a CROSS JOIN t2 b WHERE a.w < 0",
            ] {
                let root =
                    assert_parity_with(sql, config.clone(), setup_view_tables).await;
                assert_native(sql, &root);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Plans left to DataFusion
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_and_loop_joins_need_a_memory_pool_without_limit() -> Result<()> {
        // The native windows and loop joins do not reserve the memory of their
        // rows, the DataFusion nodes they replace do (and fail, or spill, once
        // the pool is exhausted): they are native only for runtimes whose
        // memory pool has no limit.
        let limited = datafusion_execution::runtime_env::RuntimeEnvBuilder::new()
            .with_memory_limit(1 << 30, 1.0)
            .build_arc()?;
        for sql in [
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
            "SELECT a.k, b.k, a.w FROM t2 a CROSS JOIN t2 b WHERE a.w < 0",
        ] {
            let ctx = datafusion_context(test_session_config());
            register_view_tables(&ctx)?;
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let options = ctx.state().config_options().clone();
            let natives = |root: &Arc<dyn PhysicalOperator>| {
                count::<PhysicalWindow>(root)
                    + count::<PhysicalNestedLoopJoin>(root)
                    + count::<PhysicalCrossJoin>(root)
            };
            let unlimited = lower_execution_plan_with_runtime(
                &plan,
                &options,
                &RuntimeEnv::default(),
            )?;
            assert_native(sql, &unlimited);
            assert_eq!(natives(&unlimited), 1, "{sql}");
            for root in [
                lower_execution_plan_with_runtime(&plan, &options, &limited)?,
                lower_execution_plan_with_options(&plan, &options)?,
                lower_execution_plan(&plan)?,
            ] {
                assert_eq!(natives(&root), 0, "{sql}\n{}", format_operator_tree(&root));
            }
            // The planner lowers with the runtime of its session.
            let state = SessionStateBuilder::new()
                .with_config(test_session_config())
                .with_runtime_env(Arc::clone(&limited))
                .with_default_features()
                .with_query_planner(Arc::new(PipelineQueryPlanner::new(Arc::new(
                    datafusion::physical_planner::DefaultPhysicalPlanner::default(),
                ))))
                .build();
            let ctx = SessionContext::new_with_state(state);
            register_view_tables(&ctx)?;
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let root = plan.downcast_ref::<PipelineRootExec>().expect("a pipeline");
            assert_eq!(natives(root.operator()), 0, "{sql}");
        }
        Ok(())
    }

    /// A partition of a `StreamingTableExec` that produces `batch`, and then
    /// what `tail` says.
    #[derive(Debug)]
    struct BatchThen {
        batch: RecordBatch,
        tail: Tail,
    }

    /// What a [`BatchThen`] does after its batch.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Tail {
        /// It ends.
        End,
        /// It never produces anything again.
        Pending,
        /// It fails after a while (an unbounded stream that breaks).
        FailLater,
    }

    impl datafusion_physical_plan::streaming::PartitionStream for BatchThen {
        fn schema(&self) -> &SchemaRef {
            self.batch.schema_ref()
        }

        fn execute(
            &self,
            _ctx: Arc<TaskContext>,
        ) -> datafusion_execution::SendableRecordBatchStream {
            use futures::StreamExt;
            let batch = futures::stream::iter([Ok(self.batch.clone())]);
            let tail = match self.tail {
                Tail::End => futures::stream::empty().boxed(),
                Tail::Pending => futures::stream::pending().boxed(),
                Tail::FailLater => futures::stream::once(async {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    datafusion_common::exec_err!("the stream fails after its batch")
                })
                .boxed(),
            };
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                    self.batch.schema(),
                    batch.chain(tail),
                ),
            )
        }
    }

    /// A stream of a single partition, with `batch` and then `tail`, declared
    /// sorted by `ordering` (if any).
    fn stream_of(
        batch: &RecordBatch,
        tail: Tail,
        ordering: Option<LexOrdering>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(
            datafusion_physical_plan::streaming::StreamingTableExec::try_new(
                batch.schema(),
                vec![Arc::new(BatchThen {
                    batch: batch.clone(),
                    tail,
                })],
                None,
                ordering,
                // (`RepartitionExec` emits the rows of unbounded inputs without
                // waiting for a full batch.)
                tail != Tail::End,
                None,
            )?,
        ))
    }

    /// `ROW_NUMBER() OVER (PARTITION BY partition_by ORDER BY order_by)` over
    /// `input`, in a `BoundedWindowAggExec` (if `bounded`) or a
    /// `WindowAggExec`.
    fn row_numbers(
        input: Arc<dyn ExecutionPlan>,
        partition_by: &str,
        order_by: &str,
        bounded: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::WindowUDF(
                datafusion::functions_window::row_number::row_number_udwf(),
            ),
            "rn".to_string(),
            &[],
            &[col(partition_by, &schema)?],
            &[PhysicalSortExpr::new_default(col(order_by, &schema)?)],
            Arc::new(datafusion_expr::WindowFrame::new(Some(true))),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        Ok(if bounded {
            Arc::new(BoundedWindowAggExec::try_new(
                vec![window_expr],
                input,
                datafusion_physical_plan::InputOrderMode::Sorted,
                true,
            )?)
        } else {
            Arc::new(WindowAggExec::try_new(vec![window_expr], input, true)?)
        })
    }

    /// Lower `plan` for a session with the default runtime (whose memory pool
    /// has no limit, so that windows and loop joins can be native), configured
    /// with `config`.
    fn lower_for_default_runtime(
        plan: Arc<dyn ExecutionPlan>,
        config: &SessionConfig,
    ) -> Result<Arc<PipelineRootExec>> {
        create_pipeline_plan_with_runtime(plan, config.options(), &RuntimeEnv::default())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_topks_that_end_early_are_left_to_datafusion() -> Result<()> {
        // A window over the partitions of a TopK of a hash repartitioning of an
        // input sorted by `k`, a prefix of the keys of the TopK: DataFusion's
        // TopK ends a partition as soon as no later row can make it into the
        // result. The only batch of the input has rows of 3 values of `k`, all
        // in the first partition, so that DataFusion's window emits a row of the
        // first partition while the TopK of the second partition waits for the
        // input, which never ends, or fails later. A native window reads all of
        // its input first: it would wait forever, or fail.
        let config = test_session_config();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let k = col("k", &schema)?;
        let v = col("v", &schema)?;
        let hash = |input| -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(RepartitionExec::try_new(
                input,
                Partitioning::Hash(vec![Arc::clone(&k)], 2),
            )?))
        };
        // The values of `k` that the repartitioning sends to its first
        // partition.
        let candidates = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::Int64Array::from_iter_values(0..64)),
                Arc::new(arrow::array::Int64Array::from_iter_values(0..64)),
            ],
        )?;
        let scan = MemorySourceConfig::try_new_exec(
            &[vec![candidates]],
            Arc::clone(&schema),
            None,
        )?;
        let first = datafusion_physical_plan::collect_partitioned(
            hash(scan)?,
            Arc::new(TaskContext::default()),
        )
        .await?
        .swap_remove(0);
        let mut keys = first
            .iter()
            .flat_map(|batch| {
                let keys = batch
                    .column(0)
                    .as_primitive::<arrow::datatypes::Int64Type>();
                keys.values().to_vec()
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.truncate(3);
        assert_eq!(keys.len(), 3);
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::Int64Array::from_iter_values(
                    keys.iter().flat_map(|&key| [key, key]),
                )),
                Arc::new(arrow::array::Int64Array::from_iter_values(0..6)),
            ],
        )?;
        let by_k = LexOrdering::new([PhysicalSortExpr::new_default(Arc::clone(&k))]);
        let by_k_and_v = LexOrdering::new([
            PhysicalSortExpr::new_default(Arc::clone(&k)),
            PhysicalSortExpr::new_default(Arc::clone(&v)),
        ])
        .expect("sort keys");
        let make = |tail: Tail, sorted: bool, bounded: bool| {
            let input = stream_of(&batch, tail, sorted.then(|| by_k.clone()).flatten())?;
            let topk: Arc<dyn ExecutionPlan> = Arc::new(
                SortExec::new(by_k_and_v.clone(), hash(input)?)
                    .with_fetch(Some(1))
                    .with_preserve_partitioning(true),
            );
            let window = row_numbers(topk, "k", "v", bounded)?;
            let merged: Arc<dyn ExecutionPlan> =
                Arc::new(CoalescePartitionsExec::new(window).with_fetch(Some(1)));
            Ok::<_, datafusion_common::DataFusionError>(merged)
        };
        // The rows of `plan`, within a few seconds.
        let rows = async |plan: Arc<dyn ExecutionPlan>| {
            let task_ctx =
                Arc::new(TaskContext::default().with_session_config(config.clone()));
            let timeout = std::time::Duration::from_secs(10);
            let batches = tokio::time::timeout(timeout, collect(plan, task_ctx)).await;
            let batches = batches.expect("no hang").expect("no error");
            arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap()
        };
        for bounded in [true, false] {
            for tail in [Tail::Pending, Tail::FailLater] {
                // DataFusion returns the first row of the first value of `k`.
                let expected = rows(make(tail, true, bounded)?).await;
                assert_eq!(expected.num_rows(), 1, "{tail:?}");
                let root =
                    lower_for_default_runtime(make(tail, true, bounded)?, &config)?;
                let operator = Arc::clone(root.operator());
                assert_eq!(rows(root).await, expected, "{tail:?}");
                assert_eq!(count::<PhysicalWindow>(&operator), 0, "{tail:?}");
            }
            // A TopK of an input in no order reads all of its partition (here
            // of an input that ends): the window is native, and returns
            // DataFusion's rows.
            let expected = rows(make(Tail::End, false, bounded)?).await;
            let root =
                lower_for_default_runtime(make(Tail::End, false, bounded)?, &config)?;
            assert_eq!(
                count::<PhysicalWindow>(root.operator()),
                1,
                "{}",
                format_operator_tree(root.operator())
            );
            assert_eq!(rows(root).await, expected);
        }
        Ok(())
    }

    /// A one-row column of a `LargeList<RunEndEncoded(Int32, Null)>` list of
    /// `1 << 30` nulls, which takes no memory: the lists of two such rows
    /// need more run ends than 32 bits hold together, so DataFusion fails to
    /// concatenate them.
    fn list_of_many_nulls() -> Result<arrow::array::ArrayRef> {
        use arrow::array::{Array, LargeListArray, NullArray, RunArray};
        use arrow::buffer::OffsetBuffer;
        let values = RunArray::<arrow::datatypes::Int32Type>::try_new(
            &Int32Array::from(vec![1 << 30]),
            &NullArray::new(1),
        )?;
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        Ok(Arc::new(LargeListArray::try_new(
            field,
            OffsetBuffer::new(vec![0, 1 << 30].into()),
            Arc::new(values),
            None,
        )?))
    }

    /// A row of a column `k` of `1` and a column `l` of `list_of_many_nulls`
    /// (or of `0`, of `Int64`, if `!many_nulls`).
    fn row_of_many_nulls(many_nulls: bool) -> Result<RecordBatch> {
        let l = match many_nulls {
            true => list_of_many_nulls()?,
            false => Arc::new(arrow::array::Int64Array::from(vec![0])),
        };
        Ok(RecordBatch::try_from_iter([
            (
                "k",
                Arc::new(Int32Array::from(vec![1])) as arrow::array::ArrayRef,
            ),
            ("l", l),
        ])?)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cross_joins_whose_build_side_may_fail_to_concatenate_stay_in_datafusion()
    -> Result<()> {
        // DataFusion's cross join concatenates its whole build side, also if
        // its probe side is empty: two one-row build batches of lists of
        // `1 << 30` nulls fail. A native cross join does not concatenate them.
        let config = test_session_config();
        let probe_schema =
            Arc::new(Schema::new(vec![Field::new("p", DataType::Int32, false)]));
        let make_plan = |many_nulls: bool| {
            let build = vec![
                row_of_many_nulls(many_nulls).unwrap(),
                row_of_many_nulls(many_nulls).unwrap(),
            ];
            let schema = build[0].schema();
            let probe_schema = Arc::clone(&probe_schema);
            move || -> Arc<dyn ExecutionPlan> {
                Arc::new(CrossJoinExec::new(
                    MemorySourceConfig::try_new_exec(
                        std::slice::from_ref(&build),
                        Arc::clone(&schema),
                        None,
                    )
                    .unwrap(),
                    MemorySourceConfig::try_new_exec(
                        &[vec![]],
                        Arc::clone(&probe_schema),
                        None,
                    )
                    .unwrap(),
                ))
            }
        };
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(config.clone()));
        let error = collect(make_plan(true)(), task_ctx)
            .await
            .expect_err("DataFusion fails")
            .to_string();
        let root =
            assert_plan_outcome(make_plan(true), config.clone(), Some(&error)).await;
        assert_eq!(count::<PhysicalCrossJoin>(&root), 0);
        // With other values, the cross join is native.
        let root = assert_plan_outcome(make_plan(false), config, None).await;
        assert_eq!(count::<PhysicalCrossJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_of_values_that_may_fail_to_combine_stay_in_datafusion() -> Result<()>
    {
        // DataFusion's repartitioning below a window coalesces the rows of a
        // partition, and its sort concatenates them: two rows of lists of
        // `1 << 30` nulls fail. The native window gathers other rows together.
        let config = test_session_config();
        let make_plan = |many_nulls: bool, bounded: bool| {
            let rows = vec![
                row_of_many_nulls(many_nulls).unwrap(),
                row_of_many_nulls(many_nulls).unwrap(),
            ];
            let schema = rows[0].schema();
            move || -> Arc<dyn ExecutionPlan> {
                let scan = MemorySourceConfig::try_new_exec(
                    std::slice::from_ref(&rows),
                    Arc::clone(&schema),
                    None,
                )
                .unwrap();
                let k = col("k", &schema).unwrap();
                let repartitioned = Arc::new(
                    RepartitionExec::try_new(
                        scan,
                        Partitioning::Hash(vec![Arc::clone(&k)], 2),
                    )
                    .unwrap(),
                );
                let sorted = Arc::new(
                    SortExec::new(
                        LexOrdering::new([PhysicalSortExpr::new_default(k)]).unwrap(),
                        repartitioned,
                    )
                    .with_preserve_partitioning(true),
                );
                row_numbers(sorted, "k", "k", bounded).unwrap()
            }
        };
        for bounded in [true, false] {
            let task_ctx =
                Arc::new(TaskContext::default().with_session_config(config.clone()));
            let error = collect(make_plan(true, bounded)(), task_ctx)
                .await
                .expect_err("DataFusion fails")
                .to_string();
            let root = assert_plan_outcome(
                make_plan(true, bounded),
                config.clone(),
                Some(&error),
            )
            .await;
            assert_eq!(count::<PhysicalWindow>(&root), 0);
            // With other values, the window is native.
            let root =
                assert_plan_outcome(make_plan(false, bounded), config.clone(), None)
                    .await;
            assert_eq!(count::<PhysicalWindow>(&root), 1);
        }
        // Strings with 32-bit offsets (`Utf8`), which may fail to concatenate
        // too: windows of them are left to DataFusion, windows of `Utf8View`
        // strings are native.
        let sql =
            "SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1";
        let root = assert_parity(sql).await;
        assert_eq!(count::<PhysicalWindow>(&root), 0);
        let root =
            assert_parity_with(sql, test_session_config(), setup_view_tables).await;
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_projections_of_sorts_that_may_fail_to_combine_stay_in_datafusion()
    -> Result<()> {
        // DataFusion plans a projection of columns between a window and the sort
        // below it, which the planner moves below the sort to lower the window.
        // But DataFusion's sort sorts all columns of its input, also those that
        // the projection drops: two rows of lists of `1 << 30` nulls, which the
        // sort concatenates, fail. A native window, which would sort the
        // projected rows only, returns them.
        let config = test_session_config();
        let make_plan = |many_nulls: bool, bounded: bool| {
            let rows = vec![
                row_of_many_nulls(many_nulls).unwrap(),
                row_of_many_nulls(many_nulls).unwrap(),
            ];
            let schema = rows[0].schema();
            move || -> Arc<dyn ExecutionPlan> {
                let scan = MemorySourceConfig::try_new_exec(
                    std::slice::from_ref(&rows),
                    Arc::clone(&schema),
                    None,
                )
                .unwrap();
                let k = col("k", &schema).unwrap();
                let ordering =
                    LexOrdering::new([PhysicalSortExpr::new_default(Arc::clone(&k))])
                        .unwrap();
                let sorted = Arc::new(SortExec::new(ordering, scan));
                let projected = Arc::new(
                    ProjectionExec::try_new(vec![(k, "k".to_string())], sorted).unwrap(),
                );
                row_numbers(projected, "k", "k", bounded).unwrap()
            }
        };
        for bounded in [true, false] {
            let task_ctx =
                Arc::new(TaskContext::default().with_session_config(config.clone()));
            let error = collect(make_plan(true, bounded)(), task_ctx)
                .await
                .expect_err("DataFusion fails")
                .to_string();
            let root = assert_plan_outcome(
                make_plan(true, bounded),
                config.clone(),
                Some(&error),
            )
            .await;
            assert_eq!(count::<PhysicalWindow>(&root), 0);
            // With other values in the column that the projection drops, the
            // window is native, with the projection below its sort.
            let root =
                assert_plan_outcome(make_plan(false, bounded), config.clone(), None)
                    .await;
            assert_eq!(count::<PhysicalWindow>(&root), 1);
            assert_eq!(count::<PhysicalProjection>(&root), 1);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_no_rows_of_windows_stay_in_datafusion() -> Result<()> {
        // A merge of the sorted partitions of a window with `fetch == Some(0)`
        // emits nothing, without reading its input: DataFusion does not cast
        // the strings below the window, one of which fails to cast. A native
        // sort of the native window would read all rows below the window, and
        // fail.
        use datafusion_physical_expr::expressions::cast;
        let config = test_session_config();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::Int64Array::from_iter_values(0..100)),
                Arc::new(arrow::array::StringViewArray::from_iter_values(
                    (0..100).map(|i| if i == 50 { "x" } else { "1" }),
                )),
            ],
        )?;
        let make_plan = |fetch: Option<usize>, bounded: bool| {
            let (batch, schema) = (batch.clone(), Arc::clone(&schema));
            move || -> Arc<dyn ExecutionPlan> {
                let scan = MemorySourceConfig::try_new_exec(
                    &[vec![batch.clone()]],
                    Arc::clone(&schema),
                    None,
                )
                .unwrap();
                let k = col("k", &schema).unwrap();
                let v =
                    cast(col("s", &schema).unwrap(), &schema, DataType::Int32).unwrap();
                let projected: Arc<dyn ExecutionPlan> = Arc::new(
                    ProjectionExec::try_new(
                        vec![(k, "k".to_string()), (v, "v".to_string())],
                        scan,
                    )
                    .unwrap(),
                );
                let projected_schema = projected.schema();
                let k = col("k", &projected_schema).unwrap();
                let v = col("v", &projected_schema).unwrap();
                let repartitioned = Arc::new(
                    RepartitionExec::try_new(
                        projected,
                        Partitioning::Hash(vec![Arc::clone(&k)], 2),
                    )
                    .unwrap(),
                );
                let ordering = LexOrdering::new([
                    PhysicalSortExpr::new_default(k),
                    PhysicalSortExpr::new_default(v),
                ])
                .unwrap();
                let sorted = Arc::new(
                    SortExec::new(ordering.clone(), repartitioned)
                        .with_preserve_partitioning(true),
                );
                let window = row_numbers(sorted, "k", "v", bounded).unwrap();
                Arc::new(SortPreservingMergeExec::new(ordering, window).with_fetch(fetch))
            }
        };
        for bounded in [true, false] {
            // DataFusion returns no rows, without casting the strings.
            let root =
                assert_plan_outcome(make_plan(Some(0), bounded), config.clone(), None)
                    .await;
            assert_eq!(count::<PhysicalSort>(&root), 0);
            // With a fetch of rows, both engines read all rows below the window,
            // and fail to cast: the merge is a native sort of the native window.
            let root = assert_plan_outcome(
                make_plan(Some(5), bounded),
                config.clone(),
                Some("Cannot cast string 'x'"),
            )
            .await;
            assert_eq!(count::<PhysicalSort>(&root), 1);
            assert_eq!(count::<PhysicalWindow>(&root), 1);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn loop_joins_below_fallible_nodes_stay_in_datafusion() -> Result<()> {
        // DataFusion's cross join joins its probe batch of 3000 rows with every
        // build row in turn, a native cross join every chunk of 2048 rows of it:
        // the projection above it, on one thread, sees the pairs in another
        // order. `CAST(nullif(ps, bs) AS INT)` fails on `'x'` (probe row 2500)
        // with the first build row, and on `'y'` (the first 2048 probe rows)
        // with the second one: DataFusion fails on `'x'` first, a native cross
        // join would make it fail on `'y'` first.
        let build = RecordBatch::try_from_iter([(
            "bs",
            Arc::new(arrow::array::StringViewArray::from(vec!["y", "1"]))
                as arrow::array::ArrayRef,
        )])?;
        let probe = RecordBatch::try_from_iter([(
            "ps",
            Arc::new(arrow::array::StringViewArray::from_iter_values(
                (0..3000).map(|row| match row {
                    ..2048 => "y",
                    2500 => "x",
                    _ => "1",
                }),
            )) as arrow::array::ArrayRef,
        )])?;
        let setup = async |ctx: &SessionContext| {
            single_batch_table("b", build.clone())(ctx).await?;
            single_batch_table("p", probe.clone())(ctx).await
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_option_extension(
                crate::operators::adaptive_filter::TrustedFunctions::new(
                    &datafusion::functions::all_default_functions(),
                ),
            );
        let sql = "SELECT CAST(nullif(ps, bs) AS INT) AS v FROM b CROSS JOIN p";
        let plan = datafusion_plan_with(sql, config.clone(), &setup).await;
        let join = plan
            .apply(|node| {
                Ok(match node.downcast_ref::<CrossJoinExec>() {
                    Some(_) => TreeNodeRecursion::Stop,
                    None => TreeNodeRecursion::Continue,
                })
            })
            .map(|recursion| recursion == TreeNodeRecursion::Stop)?;
        assert!(join, "{}", displayable(plan.as_ref()).indent(false));
        assert_fails_like_datafusion(
            sql,
            config.clone(),
            &setup,
            "Cannot cast string 'x'",
        )
        .await;
        let root = lowered(sql, config.clone(), &setup).await;
        assert_eq!(count::<PhysicalCrossJoin>(&root), 0);
        assert_eq!(count::<PhysicalProjection>(&root), 1);
        // Without a node above it that may fail, the cross join is native.
        let sql = "SELECT nullif(ps, bs) AS v FROM b CROSS JOIN p";
        let root = assert_parity_with(sql, config, &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalCrossJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn plans_below_limits_are_left_to_datafusion() {
        // DataFusion reads the input of a limit lazily: what it evaluates
        // before the limit is reached depends on its batches.
        let sql = "SELECT COUNT(*) AS n FROM (SELECT * FROM t1 LIMIT 5)";
        let root = assert_parity(sql).await;
        let limit = root.children()[0].children()[0]
            .downcast_ref::<PhysicalLimit>()
            .unwrap_or_else(|| panic!("{}", format_operator_tree(&root)));
        assert!(limit.input().is::<BridgeOperator>());
        // Unless a sort that reads all of its input is in between.
        let sql = "SELECT id, a FROM t1 ORDER BY id LIMIT 10 OFFSET 5";
        let root = assert_parity(sql).await;
        assert_native(sql, &root);
        let limit = root.downcast_ref::<PhysicalLimit>().expect("a limit");
        assert!(limit.input().is::<PhysicalTopK>());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_are_those_of_datafusion() {
        for (sql, error) in [
            // The filter removes the rows that would fail.
            ("SELECT 10 / a AS x FROM t1 WHERE a <> 0", None),
            ("SELECT SUM(10 / a) AS x FROM t1 WHERE a <> 0", None),
            (
                "SELECT d, SUM(1000 / a) AS x FROM t1 WHERE a > 0 GROUP BY d ORDER BY d",
                None,
            ),
            ("SELECT 10 / (a - a) AS x FROM t1", Some("Divide by zero")),
            (
                "SELECT COUNT(*) AS n FROM t1 WHERE 10 / a > 1",
                Some("Divide by zero"),
            ),
            (
                "SELECT d, SUM(10 / (a - a)) AS x FROM t1 GROUP BY d",
                Some("Divide by zero"),
            ),
            (
                "SELECT t1.id, 100 / t2.k AS x FROM t1 JOIN t2 ON t1.a = t2.k",
                Some("Divide by zero"),
            ),
            (
                "SELECT id FROM t1 ORDER BY 10 / (a - a), id",
                Some("Divide by zero"),
            ),
            // DataFusion filters the rows of a batch before the aggregation
            // sees them, a pipeline evaluates both on every chunk: the
            // aggregation is left to DataFusion, which fails to cast first.
            (
                "SELECT SUM(10 / a) AS x FROM t1 WHERE CAST(d AS INT) > 0",
                Some("Cannot cast string"),
            ),
            (
                "SELECT d, SUM(10 / a) AS x FROM t1 WHERE CAST(d AS INT) > 0 GROUP BY d",
                Some("Cannot cast string"),
            ),
            // Below a limit.
            (
                "SELECT 10 / (a - a) AS x FROM t1 LIMIT 3",
                Some("Divide by zero"),
            ),
        ] {
            assert_outcome(sql, error).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_of_the_operators_of_a_pipeline_come_in_datafusion_order() -> Result<()>
    {
        // One batch of 3000 rows: `10 / a` fails on row 100, `CAST(s AS INT)`
        // on row 2100. DataFusion filters the whole batch first, and fails to
        // cast. A pipeline (on a single thread, which reads the chunks in
        // order) of a native filter and a native projection (or aggregation)
        // would evaluate both on the first chunk of 2048 rows, and divide by
        // zero first: these operators are left to DataFusion.
        let config = test_session_config().with_target_partitions(1);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    (0..3000).map(|i| i32::from(i != 100)),
                )),
                Arc::new(arrow::array::StringViewArray::from_iter_values(
                    (0..3000).map(|i| if i == 2100 { "x" } else { "1" }),
                )),
            ],
        )?;
        let setup = async |ctx: &SessionContext| {
            let table = datafusion::datasource::MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![batch.clone()]],
            )?;
            ctx.register_table("f", Arc::new(table))?;
            Ok::<_, datafusion_common::DataFusionError>(())
        };
        for sql in [
            "SELECT 10 / a AS x FROM f WHERE CAST(s AS INT) > 0",
            "SELECT SUM(10 / a) AS x FROM f WHERE CAST(s AS INT) > 0",
        ] {
            for ctx in [
                datafusion_context(config.clone()),
                pipeline_context(config.clone()),
            ] {
                setup(&ctx).await?;
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                let error = collect(plan, ctx.task_ctx()).await.unwrap_err();
                assert!(
                    error.to_string().contains("Cannot cast string 'x'"),
                    "{sql}: {error}"
                );
            }
        }
        // The filter alone, and the projection without the filter, are
        // native.
        for sql in [
            "SELECT a FROM f WHERE CAST(s AS INT) > 0",
            "SELECT 10 / a AS x FROM f",
        ] {
            let ctx = pipeline_context(config.clone());
            setup(&ctx).await?;
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let root = plan.downcast_ref::<PipelineRootExec>().unwrap().operator();
            assert_native(sql, root);
        }
        Ok(())
    }

    /// A setup that registers the table `name` of a single batch, `batch`.
    fn single_batch_table(
        name: &'static str,
        batch: RecordBatch,
    ) -> impl AsyncFn(&SessionContext) -> Result<()> {
        async move |ctx: &SessionContext| {
            let table = datafusion::datasource::MemTable::try_new(
                batch.schema(),
                vec![vec![batch.clone()]],
            )?;
            ctx.register_table(name, Arc::new(table))?;
            Ok(())
        }
    }

    /// Assert that `sql` fails with an error that contains `message`, both in
    /// DataFusion and through the [`PipelineQueryPlanner`], in sessions
    /// configured with `config` and set up with `setup`.
    async fn assert_fails_like_datafusion<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        message: &str,
    ) where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        for ctx in [
            datafusion_context(config.clone()),
            pipeline_context(config.clone()),
        ] {
            setup(&ctx).await.unwrap();
            let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await;
            let outcome = match plan {
                Ok(plan) => collect(plan, ctx.task_ctx()).await,
                Err(e) => Err(e),
            };
            match outcome {
                Ok(_) => panic!("{sql} succeeds, expected an error: {message}"),
                Err(e) => assert!(e.to_string().contains(message), "{sql}: {e}"),
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn batch_sensitive_aggregates_and_sort_keys_are_left_to_datafusion() {
        // One batch of 3000 rows: 600 rows with `a = 1`, then 2400 with
        // `a = 0`. DataFusion evaluates the right side of
        // `a <> 0 AND 10 / a > 1` only on the rows that the left side selects
        // if at most 20% of the rows of the batch are left (600 of 3000): it
        // never divides by zero. On the first chunk of 2048 rows, 29% of the
        // rows are left, so a native aggregation or sort, evaluating it on
        // every chunk, would divide by zero.
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(
                (0..3000).map(|i| i32::from(i < 600)),
            ))],
        )
        .unwrap();
        let setup = single_batch_table("g", batch);
        let config = test_session_config().with_target_partitions(1);
        for sql in [
            "SELECT SUM(CAST(a <> 0 AND 10 / a > 1 AS INT)) AS n FROM g",
            "SELECT a, SUM(CAST(a <> 0 AND 10 / a > 1 AS INT)) AS n FROM g GROUP BY a",
            "SELECT a FROM g ORDER BY CAST(a <> 0 AND 10 / a > 1 AS INT), a",
            "SELECT a FROM g ORDER BY CAST(a <> 0 AND 10 / a > 1 AS INT), a LIMIT 5",
        ] {
            let root = assert_parity_with(sql, config.clone(), &setup).await;
            assert_no_native_sort_or_aggregate(sql, &root);
        }
        // Their native forms, with the left side of the conjunction only, whose
        // errors do not depend on the batches. (The classification judges the
        // cast of a boolean to an integer as a part that may fail, whose errors
        // depend on the batches, so they do without it.)
        for sql in [
            "SELECT SUM(CASE WHEN a <> 0 THEN 1 ELSE 0 END) AS n FROM g",
            "SELECT a FROM g ORDER BY a <> 0, a",
        ] {
            let root = assert_parity_with(sql, config.clone(), &setup).await;
            assert_native(sql, &root);
        }
        // Functions and string literals (broadcast to the rows of a batch, e.g.
        // the separator of `STRING_AGG`): whether their errors depend on the
        // batches is up to the classification of errors (see
        // `depends_on_batches`), and the sorts and aggregations that it judges
        // batch-sensitive are left to DataFusion.
        for sql in [
            "SELECT id, d FROM t1 ORDER BY length(d), id",
            "SELECT e, STRING_AGG(h, ',' ORDER BY id) AS hs FROM t1 GROUP BY e",
            "SELECT upper(d) AS u, COUNT(*) AS n FROM t1 GROUP BY upper(d)",
        ] {
            let root = assert_parity(sql).await;
            let mut sensitive = false;
            datafusion_plan(sql, test_session_config())
                .await
                .apply(|node| {
                    if node.is::<SortExec>() || node.is::<AggregateExec>() {
                        sensitive |= !Judgement::of_plan(node.as_ref())
                            .evaluates_like_datafusion();
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .unwrap();
            if sensitive {
                assert_no_native_sort_or_aggregate(sql, &root);
            }
        }
    }

    /// Assert that the operator tree `root` of `sql` has no native sort or
    /// aggregation.
    fn assert_no_native_sort_or_aggregate(sql: &str, root: &Arc<dyn PhysicalOperator>) {
        let natives = count::<PhysicalSort>(root)
            + count::<PhysicalTopK>(root)
            + count::<PhysicalHashAggregate>(root);
        assert_eq!(natives, 0, "{sql}\n{}", format_operator_tree(root));
    }

    /// The operator tree that `sql` is lowered to, in a session configured
    /// with `config` and set up with `setup`.
    async fn lowered<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = pipeline_context(config);
        setup(&ctx).await.unwrap();
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        Arc::clone(plan.downcast_ref::<PipelineRootExec>().unwrap().operator())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_of_the_expressions_of_an_operator_come_in_datafusion_order() {
        // One batch of 3000 rows: `10 / a` fails on row 100, `CAST(s AS INT)`
        // on row 2100. DataFusion evaluates every expression of an
        // aggregation or a sort on the whole batch before the next one, so it
        // fails to cast first; a native operator evaluating both on the first
        // chunk of 2048 rows would divide by zero first.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    (0..3000).map(|i| i32::from(i != 100)),
                )),
                Arc::new(arrow::array::StringViewArray::from_iter_values(
                    (0..3000).map(|i| if i == 2100 { "x" } else { "1" }),
                )),
            ],
        )
        .unwrap();
        let setup = single_batch_table("f", batch);
        let config = test_session_config().with_target_partitions(1);
        for sql in [
            "SELECT SUM(CAST(s AS INT)) AS x, SUM(10 / a) AS y FROM f",
            "SELECT a, SUM(CAST(s AS INT)) AS x, SUM(10 / a) AS y FROM f GROUP BY a",
            "SELECT a FROM f ORDER BY CAST(s AS INT), 10 / a",
        ] {
            assert_fails_like_datafusion(
                sql,
                config.clone(),
                &setup,
                "Cannot cast string 'x'",
            )
            .await;
            let root = lowered(sql, config.clone(), &setup).await;
            assert_no_native_sort_or_aggregate(sql, &root);
        }
        // With a single part that may fail, they are native, and fail like
        // DataFusion.
        for (sql, message) in [
            (
                "SELECT SUM(CAST(s AS INT)) AS x FROM f",
                "Cannot cast string 'x'",
            ),
            ("SELECT a FROM f ORDER BY 10 / a, a", "Divide by zero"),
        ] {
            assert_fails_like_datafusion(sql, config.clone(), &setup, message).await;
            let root = lowered(sql, config.clone(), &setup).await;
            assert_native(sql, &root);
        }
    }

    /// Assert that executing the plan that `make_plan` creates gives the same
    /// outcome in DataFusion and lowered (with [`create_pipeline_plan`]): the
    /// same rows, or errors that contain `error`. Returns the lowered
    /// operator tree.
    async fn assert_plan_outcome(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        config: SessionConfig,
        error: Option<&str>,
    ) -> Arc<dyn PhysicalOperator> {
        // Lowered for the default runtime of the task context.
        let root = lower_for_default_runtime(make_plan(), &config).unwrap();
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let operator = Arc::clone(root.operator());
        let tree = format_operator_tree(&operator);
        let expected = collect(make_plan(), Arc::clone(&task_ctx)).await;
        let actual = collect(root as Arc<dyn ExecutionPlan>, task_ctx).await;
        match (error, expected, actual) {
            (None, Ok(expected), Ok(actual)) => {
                let output = |batches| crate::test_utils::EngineOutput {
                    plan: make_plan(),
                    schema: make_plan().schema(),
                    batches,
                    ordered: false,
                };
                if let Err(message) = compare_outputs(&output(expected), &output(actual))
                {
                    panic!("Results differ: {message}\nlowered: {tree}");
                }
            }
            (Some(error), Err(expected), Err(actual)) => {
                assert!(
                    expected.to_string().contains(error),
                    "DataFusion: {expected}"
                );
                assert!(
                    actual.to_string().contains(error),
                    "pipeline engine: {actual}\nlowered: {tree}"
                );
            }
            (_, expected, actual) => panic!(
                "Outcomes differ: DataFusion {:?}, pipeline engine {:?}\nlowered: {tree}",
                expected.map(|batches| batches.len()),
                actual.map(|batches| batches.len())
            ),
        }
        operator
    }

    /// The plan of `sql` in default DataFusion, on a table `t` of four
    /// partitions with the rows `a = 0, 1, 2, 3` each.
    async fn plan_of_four_partitions(sql: &str) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(0..4))],
        )
        .unwrap();
        let ctx = datafusion_context(test_session_config());
        let table =
            datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]; 4])
                .unwrap();
        ctx.register_table("t", Arc::new(table)).unwrap();
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    /// `plan` with `stage(partial)` above its `Partial` aggregation.
    fn above_partial_aggregation(
        plan: Arc<dyn ExecutionPlan>,
        stage: impl Fn(&AggregateExec, Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let last = plan
            .downcast_ref::<AggregateExec>()
            .expect("a final aggregation");
        let mut node = Arc::clone(last.input());
        let mut above = vec![];
        while !node
            .downcast_ref::<AggregateExec>()
            .is_some_and(|aggregate| aggregate.mode() == &AggregateMode::Partial)
        {
            above.push(Arc::clone(&node));
            node = Arc::clone(node.children()[0]);
        }
        let mut input = stage(last, node);
        for node in above.into_iter().rev() {
            input = node.replace_children(vec![input], recompute()).unwrap();
        }
        plan.replace_children(vec![input], recompute()).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partial_reductions_that_change_the_states_are_left_to_datafusion() {
        let sql = "SELECT a, COUNT(*) AS n FROM t GROUP BY a";
        let plan = plan_of_four_partitions(sql).await;
        let projection = plan.downcast_ref::<ProjectionExec>().expect("a projection");
        let aggregation = Arc::clone(projection.input());
        // A `PartialReduce` stage that merges all states into the group
        // `a = 0`: DataFusion returns `(0, 16)`, which removing it would not.
        let reduce = |groups: PhysicalGroupBy| {
            let aggregation = Arc::clone(&aggregation);
            move || {
                above_partial_aggregation(Arc::clone(&aggregation), |last, partial| {
                    Arc::new(
                        AggregateExec::try_new(
                            AggregateMode::PartialReduce,
                            groups.clone(),
                            last.aggr_expr().to_vec(),
                            last.filter_expr().to_vec(),
                            partial,
                            last.input_schema(),
                        )
                        .unwrap(),
                    )
                })
            }
        };
        let zero = PhysicalGroupBy::new_single(vec![(
            datafusion_physical_expr::expressions::lit(0_i32),
            "a".to_string(),
        )]);
        let make_plan = reduce(zero);
        assert_eq!(make_plan().schema(), aggregation.schema());
        let root = assert_plan_outcome(&make_plan, test_session_config(), None).await;
        assert_eq!(count::<PhysicalHashAggregate>(&root), 0);
        let task_ctx = Arc::new(TaskContext::default());
        let rows = collect(make_plan(), task_ctx).await.unwrap();
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);

        // A `PartialReduce` stage that merges the states into the same groups
        // is removed: the native aggregation merges all states at once.
        let identity = aggregation
            .downcast_ref::<AggregateExec>()
            .unwrap()
            .group_expr()
            .clone();
        let make_plan = reduce(identity);
        let root = assert_plan_outcome(&make_plan, test_session_config(), None).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn redistributions_of_partial_states_that_may_fail_are_left_to_datafusion() {
        let plan = plan_of_four_partitions("SELECT SUM(a) AS n FROM t").await;
        let projection = plan.downcast_ref::<ProjectionExec>().expect("a projection");
        let aggregation = Arc::clone(projection.input());
        let last = aggregation
            .downcast_ref::<AggregateExec>()
            .expect("an aggregation");
        assert_eq!(last.mode(), &AggregateMode::Final);
        let partial = Arc::clone(last.input().children()[0]);
        let state: Arc<dyn PhysicalExpr> =
            Arc::new(Column::new(partial.schema().field(0).name(), 0));
        // A `RepartitionExec` of the partial states by `key` (instead of the
        // `CoalescePartitionsExec`).
        let repartition = |key: Arc<dyn PhysicalExpr>| {
            let (aggregation, partial) = (Arc::clone(&aggregation), Arc::clone(&partial));
            move || {
                let input: Arc<dyn ExecutionPlan> = Arc::new(
                    RepartitionExec::try_new(
                        Arc::clone(&partial),
                        Partitioning::Hash(vec![Arc::clone(&key)], 1),
                    )
                    .unwrap(),
                );
                Arc::clone(&aggregation)
                    .replace_children(vec![input], recompute())
                    .unwrap()
            }
        };
        // By `10 / (sum - sum)`: DataFusion divides by zero, which removing
        // the repartitioning would not.
        let key: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
            datafusion_expr::Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::clone(&state),
                datafusion_expr::Operator::Minus,
                Arc::clone(&state),
            )),
        ));
        let make_plan = repartition(key);
        let root = assert_plan_outcome(
            &make_plan,
            test_session_config(),
            Some("Divide by zero"),
        )
        .await;
        assert_eq!(count::<PhysicalHashAggregate>(&root), 0);
        // By an expression that cannot fail, it is removed.
        let make_plan = repartition(state);
        let root = assert_plan_outcome(&make_plan, test_session_config(), None).await;
        assert_native("SELECT SUM(a) AS n FROM t", &root);
        assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn coalescing_below_fallible_nodes_over_fallible_inputs_stays_in_datafusion() {
        use crate::operators::union::test_support::{
            AfterBatch, INJECTED_FAILURE, batch_plan,
        };

        // `10 / a` over 1000 zeros, then a source error. `FilterExec` and
        // `CoalesceBatchesExec` pull the error before they emit the zeros, so
        // DataFusion raises the source error; native operators that pass the
        // zeros on at once would divide by zero first.
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let zeros = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![0; 1000]))],
        )
        .unwrap();
        let divide = |input: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
            let a = col("a", &input.schema()).unwrap();
            let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
                datafusion_expr::Operator::Divide,
                a,
            ));
            Arc::new(
                ProjectionExec::try_new(vec![(expr, "x".to_string())], input).unwrap(),
            )
        };
        let make_plan = || {
            let source = batch_plan(zeros.clone(), AfterBatch::Fail).unwrap();
            divide(Arc::new(coalesce_batches(source, None)))
        };
        let config = test_session_config().with_target_partitions(1);
        let root =
            assert_plan_outcome(make_plan, config.clone(), Some(INJECTED_FAILURE)).await;
        let projection = root.downcast_ref::<PhysicalProjection>().expect("native");
        let bridge = projection.children()[0]
            .downcast_ref::<BridgeOperator>()
            .expect("a bridge");
        assert!(is_coalesce_batches(bridge.plan()));
        // Without a part that may fail above it, the coalescing is removed.
        let make_plan = || {
            let source = batch_plan(zeros.clone(), AfterBatch::Fail).unwrap();
            let coalesced: Arc<dyn ExecutionPlan> =
                Arc::new(coalesce_batches(source, None));
            let a = col("a", &coalesced.schema()).unwrap();
            Arc::new(
                ProjectionExec::try_new(vec![(a, "x".to_string())], coalesced).unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        let root =
            assert_plan_outcome(make_plan, config.clone(), Some(INJECTED_FAILURE)).await;
        let projection = root.downcast_ref::<PhysicalProjection>().expect("native");
        let bridge = projection.children()[0]
            .downcast_ref::<BridgeOperator>()
            .expect("a bridge");
        assert!(!is_coalesce_batches(bridge.plan()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_below_fallible_nodes_over_fallible_sources_stay_in_datafusion()
    -> Result<()> {
        // A CSV file of 1000 zeros, 7192 negative ones (DataFusion's first
        // batch of 8192 rows), then a value that is not an integer. DataFusion's
        // `FilterExec` coalesces the 1000 zeros that pass `a >= 0` and reads
        // the next batch, which fails to parse, before the projection above
        // divides by zero: the source error comes first, as it does when a
        // sort below an aggregation reads all of its input first.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("f.csv");
        let mut csv = String::from("a\n");
        csv.push_str(&"0\n".repeat(1000));
        csv.push_str(&"-1\n".repeat(7192));
        csv.push_str("x\n");
        csv.push_str(&"1\n".repeat(10));
        std::fs::write(&path, csv)?;
        let path = path.to_str().unwrap().to_string();
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let setup = async |ctx: &SessionContext| {
            ctx.register_csv("f", &path, CsvReadOptions::new().schema(&schema))
                .await?;
            Ok(())
        };
        let config = test_session_config().with_target_partitions(1);
        for sql in [
            "SELECT 10 / a AS x FROM f WHERE a >= 0",
            "SELECT ARRAY_AGG(10 / a ORDER BY a) AS x FROM f WHERE a > -2",
        ] {
            let datafusion = datafusion_context(config.clone());
            setup(&datafusion).await?;
            let error = crate::test_utils::run_sql(&datafusion, sql)
                .await
                .expect_err("DataFusion fails")
                .to_string();
            assert!(!error.contains("Divide by zero"), "{sql}: {error}");
            assert_fails_like_datafusion(sql, config.clone(), &setup, &error).await;
        }
        // Over a source that cannot fail, the filter is native (on one thread,
        // on which the native operators evaluate the rows in order).
        let root = lowered(
            "SELECT 10 / a AS x FROM t1 WHERE a >= 0",
            test_session_config().with_target_partitions(1),
            setup_test_tables,
        )
        .await;
        assert_native("SELECT 10 / a AS x FROM t1 WHERE a >= 0", &root);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_by_fallible_keys_of_fallible_inputs_stay_in_datafusion() -> Result<()>
    {
        // A CSV file of 300000 rows, the first of which is 0, with a value that
        // is not an integer near its end. DataFusion's sort reads all of its
        // input before it evaluates its keys, so it fails to parse the file;
        // the native sort sorts (and evaluates the keys of) runs of
        // `SORT_RUN_ROWS` rows as it reads its input, and would divide by zero
        // first.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("f.csv");
        let mut csv = String::from("a\n0\n");
        csv.push_str(&"1\n".repeat(289_999));
        csv.push_str("x\n");
        csv.push_str(&"1\n".repeat(10_000));
        std::fs::write(&path, csv)?;
        let path = path.to_str().unwrap().to_string();
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let setup = async |ctx: &SessionContext| {
            ctx.register_csv("f", &path, CsvReadOptions::new().schema(&schema))
                .await?;
            Ok(())
        };
        let config = test_session_config().with_target_partitions(1);
        let sql = "SELECT a FROM f ORDER BY 10 / a, a";
        let datafusion = datafusion_context(config.clone());
        setup(&datafusion).await?;
        let error = crate::test_utils::run_sql(&datafusion, sql)
            .await
            .expect_err("DataFusion fails")
            .to_string();
        assert!(!error.contains("Divide by zero"), "{error}");
        assert_fails_like_datafusion(sql, config.clone(), &setup, &error).await;
        let root = lowered(sql, config, &setup).await;
        assert_eq!(count::<PhysicalSort>(&root), 0);
        // Over an input that cannot fail, the sort is native (and `a + 100` is
        // never 0), on one thread, on which it evaluates its keys in order.
        let sql = "SELECT a FROM t1 ORDER BY 10 / (a + 100), id";
        let config = test_session_config().with_target_partitions(1);
        let root = assert_parity_with(sql, config, setup_test_tables).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalSort>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_with_build_keys_that_may_fail_are_left_to_datafusion() -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};
        use datafusion_expr::Operator;
        use datafusion_physical_expr::expressions::{binary, cast, lit};

        // DataFusion evaluates the (two) build keys of a join once it
        // collected its build side: both keys on a build batch before the
        // next, the last batch first. A native join evaluates them on its
        // chunks of up to 2048 rows, as they arrive.
        let config = test_session_config().with_target_partitions(1);
        let probe_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ]));
        let probe = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![1])),
            ],
        )?;
        // The build keys of a join, on the schema of its build side.
        type Keys = dyn Fn(&Schema) -> Result<Vec<Arc<dyn PhysicalExpr>>>;
        // A `CollectLeft` join of `build` (one partition) on `keys` with the
        // row `x = 1, y = 1`.
        let join = |build: Vec<RecordBatch>, keys: &Keys| {
            let schema = build[0].schema();
            let keys = keys(&schema).unwrap();
            let (probe, probe_schema) = (probe.clone(), Arc::clone(&probe_schema));
            move || -> Arc<dyn ExecutionPlan> {
                let on = keys
                    .iter()
                    .cloned()
                    .zip(["x", "y"].map(|name| col(name, &probe_schema).unwrap()))
                    .collect();
                Arc::new(
                    HashJoinExec::try_new(
                        MemorySourceConfig::try_new_exec(
                            std::slice::from_ref(&build),
                            Arc::clone(&schema),
                            None,
                        )
                        .unwrap(),
                        MemorySourceConfig::try_new_exec(
                            &[vec![probe.clone()]],
                            Arc::clone(&probe_schema),
                            None,
                        )
                        .unwrap(),
                        on,
                        None,
                        &JoinType::Inner,
                        None,
                        PartitionMode::CollectLeft,
                        NullEquality::NullEqualsNothing,
                        false,
                    )
                    .unwrap(),
                )
            }
        };
        let ints = |values: Vec<i32>| -> Result<RecordBatch> {
            Ok(RecordBatch::try_from_iter([(
                "a",
                Arc::new(Int32Array::from(values)) as _,
            )])?)
        };

        // 600 rows `a = 1`, then 2400 `a = 0`: DataFusion evaluates the right
        // side of `a <> 0 AND 10 / a > 1` only on the 600 rows that the left
        // side selects in the batch (at most 20% of it), a native join on all
        // rows of its first chunk (more than 20% of which are selected).
        let mut values = vec![1; 600];
        values.extend([0; 2400]);
        let batch_dependent = join(vec![ints(values.clone())?], &|schema| {
            let a = col("a", schema)?;
            let left = binary(Arc::clone(&a), Operator::NotEq, lit(0), schema)?;
            let divide = binary(lit(10), Operator::Divide, Arc::clone(&a), schema)?;
            let right = binary(divide, Operator::Gt, lit(1), schema)?;
            let both = binary(left, Operator::And, right, schema)?;
            Ok(vec![cast(both, schema, DataType::Int32)?, a])
        });
        let root = assert_plan_outcome(&batch_dependent, config.clone(), None).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        // The same rows, joined on keys that cannot fail, natively.
        let infallible = join(vec![ints(values)?], &|schema| {
            Ok(vec![col("a", schema)?, col("a", schema)?])
        });
        let root = assert_plan_outcome(&infallible, config.clone(), None).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        assert!(bridged_plans(&root).is_empty());

        // Two keys that may fail: `CAST(s AS INT)` on row 2100, `10 / a` on
        // row 100. DataFusion casts all rows of the batch first.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let strings = |a: Vec<i32>, s: Vec<String>| -> Result<RecordBatch> {
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(a)),
                    Arc::new(arrow::array::StringViewArray::from(s)),
                ],
            )?)
        };
        let mut a = vec![1; 3000];
        a[100] = 0;
        let mut s = vec!["1".to_string(); 3000];
        s[2100] = "x".to_string();
        let two_failing_keys = join(vec![strings(a, s)?], &|schema| {
            let divide = binary(lit(10), Operator::Divide, col("a", schema)?, schema)?;
            Ok(vec![
                cast(col("s", schema)?, schema, DataType::Int32)?,
                divide,
            ])
        });
        let error = Some("Cannot cast string 'x'");
        let root = assert_plan_outcome(&two_failing_keys, config.clone(), error).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);

        // A single key that may fail, on a row of each of two batches:
        // DataFusion fails on the row of the last batch first.
        let batch = |bad: &str| {
            let mut s = vec!["1".to_string(); 3000];
            s[10] = bad.to_string();
            strings(vec![1; 3000], s)
        };
        let one_failing_key = join(vec![batch("x")?, batch("y")?], &|schema| {
            Ok(vec![
                cast(col("s", schema)?, schema, DataType::Int32)?,
                col("a", schema)?,
            ])
        });
        let error = Some("Cannot cast string 'y'");
        let root = assert_plan_outcome(&one_failing_key, config, error).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn repartitions_that_may_fail_to_coalesce_are_kept() -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};

        // Two batches of 100 rows whose `Dictionary(Int8, Utf8)` values are
        // disjoint. A `RepartitionExec` over a bounded input coalesces the
        // batches of every output partition (up to `batch_size` rows), which
        // merges their dictionaries into one of 200 values: more than keys of
        // 8 bits hold.
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("d", dictionary, false),
        ]));
        let batch = |prefix: &str| {
            let strings = (0..100).map(|i| format!("{prefix}{i}")).collect::<Vec<_>>();
            let values: arrow::array::DictionaryArray<arrow::datatypes::Int8Type> =
                strings.iter().map(String::as_str).collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..100)),
                    Arc::new(values),
                ],
            )
        };
        let batches = vec![batch("a")?, batch("b")?];
        let repartitioned = |partitioning: &Partitioning| -> Arc<dyn ExecutionPlan> {
            let source = MemorySourceConfig::try_new_exec(
                std::slice::from_ref(&batches),
                Arc::clone(&schema),
                None,
            )
            .unwrap();
            Arc::new(RepartitionExec::try_new(source, partitioning.clone()).unwrap())
        };
        let config = test_session_config().with_target_partitions(1);
        let error = Some("Dictionary key bigger than the key type");
        let k = col("k", &schema)?;
        let d = col("d", &schema)?;
        for partitioning in [
            Partitioning::RoundRobinBatch(1),
            Partitioning::Hash(vec![Arc::clone(&d)], 1),
        ] {
            let make_plan = || repartitioned(&partitioning);
            let root = assert_plan_outcome(make_plan, config.clone(), error).await;
            assert!(root.is::<BridgeOperator>(), "{partitioning}");

            // The probe side of a join, whose redistribution the join removes.
            let build_schema =
                Arc::new(Schema::new(vec![Field::new("b", DataType::Int32, false)]));
            let build = RecordBatch::try_new(
                Arc::clone(&build_schema),
                vec![Arc::new(Int32Array::from(vec![1]))],
            )?;
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                let probe = match &partitioning {
                    Partitioning::Hash(..) => Partitioning::Hash(vec![Arc::clone(&k)], 1),
                    other => other.clone(),
                };
                Arc::new(
                    HashJoinExec::try_new(
                        MemorySourceConfig::try_new_exec(
                            &[vec![build.clone()]],
                            Arc::clone(&build_schema),
                            None,
                        )
                        .unwrap(),
                        repartitioned(&probe),
                        vec![(col("b", &build_schema).unwrap(), Arc::clone(&k))],
                        None,
                        &JoinType::Inner,
                        None,
                        PartitionMode::CollectLeft,
                        NullEquality::NullEqualsNothing,
                        false,
                    )
                    .unwrap(),
                )
            };
            let root = assert_plan_outcome(make_plan, config.clone(), error).await;
            assert_eq!(count::<PhysicalHashJoin>(&root), 0, "{partitioning}");
        }

        // Even if the keys hold the values of all batches: a dictionary may fail
        // to coalesce, the repartitioning is kept.
        let make_plan = || {
            let source = MemorySourceConfig::try_new_exec(
                &[vec![batches[0].clone()]],
                Arc::clone(&schema),
                None,
            )
            .unwrap();
            Arc::new(
                RepartitionExec::try_new(source, Partitioning::RoundRobinBatch(1))
                    .unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        let root = assert_plan_outcome(make_plan, config, None).await;
        assert!(
            root.is::<BridgeOperator>(),
            "a dictionary may fail to coalesce"
        );

        // `Utf8` columns, whose 32-bit offsets may overflow: the aggregation of
        // states that a repartitioning coalesces, and the partitioned join
        // whose inputs it coalesces, are left to DataFusion.
        let sql = "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d";
        let root = assert_parity(sql).await;
        assert_eq!(count::<PhysicalHashAggregate>(&root), 0);
        let sql = "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k";
        let root = assert_parity_with(sql, partitioned_config(), setup_test_tables).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn repartitions_below_fallible_nodes_over_fallible_inputs_stay_in_datafusion() {
        use crate::operators::union::test_support::{
            AfterBatch, INJECTED_FAILURE, batch_plan,
        };

        // `10 / a` over 1000 zeros, then a source error. A `RepartitionExec`
        // over a bounded input coalesces its output, so it pulls the error
        // before it emits the zeros, and DataFusion raises the source error.
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let zeros = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![0; 1000]))],
        )
        .unwrap();
        let config = test_session_config().with_target_partitions(1);
        let a = col("a", &schema).unwrap();
        for partitioning in [
            Partitioning::RoundRobinBatch(1),
            Partitioning::Hash(vec![Arc::clone(&a)], 1),
        ] {
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                let source = batch_plan(zeros.clone(), AfterBatch::Fail).unwrap();
                let repartition: Arc<dyn ExecutionPlan> = Arc::new(
                    RepartitionExec::try_new(source, partitioning.clone()).unwrap(),
                );
                let divide: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
                    datafusion_expr::Operator::Divide,
                    Arc::clone(&a),
                ));
                Arc::new(
                    ProjectionExec::try_new(vec![(divide, "x".to_string())], repartition)
                        .unwrap(),
                )
            };
            let root =
                assert_plan_outcome(make_plan, config.clone(), Some(INJECTED_FAILURE))
                    .await;
            let projection = root.downcast_ref::<PhysicalProjection>().expect("native");
            let bridge = projection.children()[0]
                .downcast_ref::<BridgeOperator>()
                .expect("a bridge");
            assert!(bridge.plan().is::<RepartitionExec>(), "{partitioning}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordering_sorts_of_ordered_aggregates_that_may_fail_are_kept() {
        // DataFusion sorts the rows of an ordered aggregate (here `1, 'y'`
        // first) before the aggregation evaluates its arguments, so it fails
        // to cast `'y'`; without the sort, the native aggregation would fail
        // to cast `'x'`, the first row of its input.
        let sql = "SELECT ARRAY_AGG(CAST(column2 AS INT) ORDER BY column1 DESC) \
                   FROM (VALUES (0, 'x'), (1, 'y'))";
        let config = test_session_config().with_target_partitions(1);
        let setup = async |_: &SessionContext| Ok(());
        assert_fails_like_datafusion(
            sql,
            config.clone(),
            &setup,
            "Cannot cast string 'y'",
        )
        .await;
        // The arguments of the aggregate cannot fail: the sort is removed.
        let sql = "SELECT ARRAY_AGG(column2 ORDER BY column1 DESC) AS x \
                   FROM (VALUES (0, 'x'), (1, 'y'))";
        let root = lowered(sql, config, &setup).await;
        assert_eq!(count::<PhysicalSort>(&root), 0);
        assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
        assert!(bridged_plans(&root).is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_with_probe_keys_that_may_fail_are_left_to_datafusion() -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};
        use datafusion_physical_expr::expressions::TryCastExpr;

        // DataFusion evaluates the probe keys of a join on every probe batch,
        // also on an empty one, on which casting to a dictionary of booleans
        // fails (even with `TRY_CAST`). A native join evaluates them on the
        // rows of its chunks: none.
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean));
        let build_schema =
            Arc::new(Schema::new(vec![Field::new("d", dictionary.clone(), true)]));
        let values: arrow::array::DictionaryArray<arrow::datatypes::Int32Type> =
            arrow::array::DictionaryArray::try_new(
                Int32Array::from(vec![0, 1]),
                Arc::new(arrow::array::BooleanArray::from(vec![true, false])),
            )?;
        let build =
            RecordBatch::try_new(Arc::clone(&build_schema), vec![Arc::new(values)])?;
        let probe_schema =
            Arc::new(Schema::new(vec![Field::new("b", DataType::Boolean, true)]));
        let probe = RecordBatch::new_empty(Arc::clone(&probe_schema));
        let make_plan = || -> Arc<dyn ExecutionPlan> {
            let key: Arc<dyn PhysicalExpr> = Arc::new(TryCastExpr::new(
                col("b", &probe_schema).unwrap(),
                dictionary.clone(),
            ));
            Arc::new(
                HashJoinExec::try_new(
                    MemorySourceConfig::try_new_exec(
                        &[vec![build.clone()]],
                        Arc::clone(&build_schema),
                        None,
                    )
                    .unwrap(),
                    MemorySourceConfig::try_new_exec(
                        &[vec![probe.clone()]],
                        Arc::clone(&probe_schema),
                        None,
                    )
                    .unwrap(),
                    vec![(col("d", &build_schema).unwrap(), key)],
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )
                .unwrap(),
            )
        };
        let config = test_session_config().with_target_partitions(1);
        let error = Some("Unsupported output type for dictionary packing");
        let root = assert_plan_outcome(make_plan, config, error).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_over_nodes_that_may_fail_to_coalesce_stay_in_datafusion()
    -> Result<()> {
        // `10 / a` above a filter `a >= 0`, over a node that coalesces its
        // input: 8192 rows (1000 zeros) that it passes on, then two batches of
        // 100 rows whose `Dictionary(Int8, Utf8)` values are disjoint, which
        // it fails to merge. `FilterExec` pulls that error before it emits the
        // zeros, so DataFusion raises it; a native filter would pass the zeros
        // on at once, and the projection divide by zero first.
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("d", dictionary, false),
        ]));
        let batch = |a: Vec<i32>, d: Vec<String>| {
            let d: arrow::array::DictionaryArray<arrow::datatypes::Int8Type> =
                d.iter().map(String::as_str).collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(a)), Arc::new(d)],
            )
        };
        let mut a = vec![0; 1000];
        a.extend(std::iter::repeat_n(-1, 7192));
        let disjoint = |prefix: &str| {
            batch(
                vec![-1; 100],
                (0..100).map(|i| format!("{prefix}{i}")).collect(),
            )
        };
        let batches = vec![
            batch(a, vec!["c".to_string(); 8192])?,
            disjoint("x")?,
            disjoint("y")?,
        ];
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(8192);
        let error = Some("Dictionary key bigger than the key type");
        let a = col("a", &schema)?;
        for coalescing in ["round robin", "hash", "CoalesceBatchesExec"] {
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                let source = MemorySourceConfig::try_new_exec(
                    std::slice::from_ref(&batches),
                    Arc::clone(&schema),
                    None,
                )
                .unwrap();
                let coalesced: Arc<dyn ExecutionPlan> = match coalescing {
                    "round robin" => Arc::new(
                        RepartitionExec::try_new(
                            source,
                            Partitioning::RoundRobinBatch(1),
                        )
                        .unwrap(),
                    ),
                    "hash" => Arc::new(
                        RepartitionExec::try_new(
                            source,
                            Partitioning::Hash(vec![Arc::clone(&a)], 1),
                        )
                        .unwrap(),
                    ),
                    _ => Arc::new(coalesce_batches(source, None)),
                };
                let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                    Arc::clone(&a),
                    datafusion_expr::Operator::GtEq,
                    Arc::new(Literal::new(ScalarValue::Int32(Some(0)))),
                ));
                let filter: Arc<dyn ExecutionPlan> = Arc::new(
                    datafusion_physical_plan::filter::FilterExecBuilder::new(
                        predicate, coalesced,
                    )
                    .apply_projection(Some(vec![0]))
                    .unwrap()
                    .build()
                    .unwrap(),
                );
                let divide: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
                    datafusion_expr::Operator::Divide,
                    col("a", &filter.schema()).unwrap(),
                ));
                Arc::new(
                    ProjectionExec::try_new(vec![(divide, "x".to_string())], filter)
                        .unwrap(),
                )
            };
            let root = assert_plan_outcome(make_plan, config.clone(), error).await;
            assert_eq!(count::<PhysicalFilter>(&root), 0, "{coalescing}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_of_values_that_address_more_than_their_memory_stay_in_datafusion()
    -> Result<()> {
        use arrow::array::{Array, LargeListArray, NullArray, RunArray};
        use arrow::buffer::OffsetBuffer;

        // DataFusion's sort concatenates the batches of a partition while they
        // occupy fewer bytes than `sort_in_place_threshold_bytes`. Two one-row
        // batches whose `LargeList<RunEndEncoded(Int32, Null)>` lists hold
        // `1 << 30` nulls each occupy a few bytes, but need more run ends than
        // 32 bits hold together: DataFusion fails to concatenate them, a native
        // sort fails to interleave them, with another error.
        let list = || -> Result<arrow::array::ArrayRef> {
            let values = RunArray::<arrow::datatypes::Int32Type>::try_new(
                &Int32Array::from(vec![1 << 30]),
                &NullArray::new(1),
            )?;
            let field = Arc::new(Field::new("item", values.data_type().clone(), true));
            Ok(Arc::new(LargeListArray::try_new(
                field,
                OffsetBuffer::new(vec![0, 1 << 30].into()),
                Arc::new(values),
                None,
            )?))
        };
        let row = |k: i32| -> Result<RecordBatch> {
            Ok(RecordBatch::try_from_iter([
                (
                    "k",
                    Arc::new(Int32Array::from(vec![k])) as arrow::array::ArrayRef,
                ),
                ("l", list()?),
            ])?)
        };
        let batches = vec![row(2)?, row(1)?];
        let schema = batches[0].schema();
        let make_plan = || -> Arc<dyn ExecutionPlan> {
            let source = MemorySourceConfig::try_new_exec(
                std::slice::from_ref(&batches),
                Arc::clone(&schema),
                None,
            )
            .unwrap();
            let k = col("k", &schema).unwrap();
            Arc::new(SortExec::new(
                LexOrdering::new(vec![PhysicalSortExpr::new_default(k)]).unwrap(),
                source,
            ))
        };
        let config = test_session_config().with_target_partitions(1);
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(config.clone()));
        let error = collect(make_plan(), task_ctx)
            .await
            .expect_err("DataFusion fails")
            .to_string();
        let root = assert_plan_outcome(make_plan, config, Some(&error)).await;
        assert_eq!(count::<PhysicalSort>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_build_side_may_fail_to_concatenate_are_left_to_datafusion()
    -> Result<()> {
        use arrow::array::{Array, LargeListArray, NullArray, RunArray};
        use arrow::buffer::OffsetBuffer;
        use datafusion_common::{JoinType, NullEquality};

        // DataFusion concatenates all columns of the build side of a join,
        // also those that its output does not have: two one-row build batches
        // whose `LargeList<RunEndEncoded(Int32, Null)>` lists hold `1 << 30`
        // nulls each need more run ends than 32 bits hold together. A native
        // join does not concatenate them.
        let list = || -> Result<arrow::array::ArrayRef> {
            let values = RunArray::<arrow::datatypes::Int32Type>::try_new(
                &Int32Array::from(vec![1 << 30]),
                &NullArray::new(1),
            )?;
            let field = Arc::new(Field::new("item", values.data_type().clone(), true));
            Ok(Arc::new(LargeListArray::try_new(
                field,
                OffsetBuffer::new(vec![0, 1 << 30].into()),
                Arc::new(values),
                None,
            )?))
        };
        let row = || -> Result<RecordBatch> {
            Ok(RecordBatch::try_from_iter([
                (
                    "k",
                    Arc::new(Int32Array::from(vec![1])) as arrow::array::ArrayRef,
                ),
                ("l", list()?),
            ])?)
        };
        let build = vec![row()?, row()?];
        let build_schema = build[0].schema();
        let probe_schema =
            Arc::new(Schema::new(vec![Field::new("p", DataType::Int32, false)]));
        let probe = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )?;
        let make_plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                HashJoinExec::try_new(
                    MemorySourceConfig::try_new_exec(
                        std::slice::from_ref(&build),
                        Arc::clone(&build_schema),
                        None,
                    )
                    .unwrap(),
                    MemorySourceConfig::try_new_exec(
                        &[vec![probe.clone()]],
                        Arc::clone(&probe_schema),
                        None,
                    )
                    .unwrap(),
                    vec![(
                        col("k", &build_schema).unwrap(),
                        col("p", &probe_schema).unwrap(),
                    )],
                    None,
                    &JoinType::Inner,
                    // Only the probe key: `k`, `l`, then `p`.
                    Some(vec![2]),
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )
                .unwrap(),
            )
        };
        let config = test_session_config().with_target_partitions(1);
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(config.clone()));
        let error = collect(make_plan(), task_ctx)
            .await
            .expect_err("DataFusion fails")
            .to_string();
        let root = assert_plan_outcome(make_plan, config, Some(&error)).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_output_coalescing_may_differ_are_left_to_datafusion()
    -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};

        // DataFusion's hash join coalesces its output, up to `batch_size`
        // rows, before it emits it. A native join passes its output on at
        // once.
        let config = test_session_config().with_target_partitions(1);
        let build_schema =
            Arc::new(Schema::new(vec![Field::new("b", DataType::Int32, false)]));
        let build = RecordBatch::try_new(
            Arc::clone(&build_schema),
            vec![Arc::new(Int32Array::from(vec![0]))],
        )?;
        let join = |probe: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
            let k = col("k", &probe.schema()).unwrap();
            Arc::new(
                HashJoinExec::try_new(
                    MemorySourceConfig::try_new_exec(
                        &[vec![build.clone()]],
                        Arc::clone(&build_schema),
                        None,
                    )
                    .unwrap(),
                    probe,
                    vec![(col("b", &build_schema).unwrap(), k)],
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )
                .unwrap(),
            )
        };

        // It may fail to concatenate the output batches of two probe batches
        // of 100 rows whose `Dictionary(Int8, Utf8)` values are disjoint.
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("d", dictionary, false),
        ]));
        let batch = |prefix: &str| {
            let strings = (0..100).map(|i| format!("{prefix}{i}")).collect::<Vec<_>>();
            let d: arrow::array::DictionaryArray<arrow::datatypes::Int8Type> =
                strings.iter().map(String::as_str).collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(vec![0; 100])), Arc::new(d)],
            )
        };
        let batches = vec![batch("x")?, batch("y")?];
        let make_plan = || {
            join(
                MemorySourceConfig::try_new_exec(
                    std::slice::from_ref(&batches),
                    Arc::clone(&schema),
                    None,
                )
                .unwrap(),
            )
        };
        let error = Some("Dictionary key bigger than the key type");
        let root = assert_plan_outcome(make_plan, config.clone(), error).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);

        // `10 / k` above the join of a CSV file, on one thread: its first batch
        // of 8192 rows holds 1000 zeros (that join) and 7192 `-1`s, its next
        // batch a value that does not parse. The join pulls the parse error
        // before it emits the zeros, so DataFusion raises it; a native join
        // would pass the zeros on at once, and the projection divide by zero
        // first.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("f.csv");
        let mut csv = String::from("k\n");
        csv.push_str(&"0\n".repeat(1000));
        csv.push_str(&"-1\n".repeat(7192));
        csv.push_str("x\n");
        std::fs::write(&path, csv)?;
        let ctx = SessionContext::new_with_config(config.clone());
        let schema = Schema::new(vec![Field::new("k", DataType::Int32, true)]);
        ctx.register_csv(
            "f",
            path.to_str().unwrap(),
            CsvReadOptions::new().schema(&schema),
        )
        .await?;
        let scan = ctx.table("f").await?.create_physical_plan().await?;
        let make_plan = |divide: bool| {
            // A copy with its own state: a scan shares the files it reads
            // between its executions.
            let joined = join(
                datafusion_physical_plan::execution_plan::reset_plan_states(Arc::clone(
                    &scan,
                ))
                .unwrap(),
            );
            let k = col("k", &joined.schema()).unwrap();
            let expr: Arc<dyn PhysicalExpr> = if divide {
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
                    datafusion_expr::Operator::Divide,
                    k,
                ))
            } else {
                k
            };
            Arc::new(
                ProjectionExec::try_new(vec![(expr, "x".to_string())], joined).unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        let error = collect(make_plan(true), ctx.task_ctx())
            .await
            .expect_err("DataFusion fails")
            .to_string();
        assert!(!error.contains("Divide by zero"), "{error}");
        let root =
            assert_plan_outcome(|| make_plan(true), config.clone(), Some(&error)).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        // Without a part that may fail above it, the join is native.
        let root = assert_plan_outcome(|| make_plan(false), config, Some(&error)).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_dynamic_filter_prunes_reads_that_may_fail_stay_in_datafusion()
    -> Result<()> {
        // A table `k INT, v INT` of two Parquet files: 100 rows `k = 1` of
        // integers, and rows `k = 0` of strings that fail to cast to `INT`
        // when they are read. DataFusion pushes the dynamic filter of the join
        // (`k = 1`, from its build side) into the scan, which prunes the
        // second file: it never reads it. A native join does not prune the same
        // reads, so that it would fail to cast.
        let dir = tempfile::tempdir()?;
        let good = RecordBatch::try_from_iter([
            (
                "k",
                Arc::new(Int32Array::from(vec![1; 100])) as arrow::array::ArrayRef,
            ),
            ("v", Arc::new(Int32Array::from_iter_values(0..100)) as _),
        ])?;
        let bad = RecordBatch::try_from_iter([
            (
                "k",
                Arc::new(Int32Array::from(vec![0; 10])) as arrow::array::ArrayRef,
            ),
            (
                "v",
                Arc::new(arrow::array::StringViewArray::from(vec!["bad_integer"; 10]))
                    as _,
            ),
        ])?;
        let path = dir.path().join("p");
        write_parquet(&path.join("0.parquet"), good.schema(), &[good], None)?;
        write_parquet(&path.join("1.parquet"), bad.schema(), &[bad], None)?;
        let path = path.to_str().unwrap();
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int32, true),
        ]);
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet(
                "p",
                path,
                ParquetReadOptions::default().schema(&schema),
            )
            .await?;
            ctx.sql("CREATE TABLE b AS SELECT 1 AS x")
                .await?
                .collect()
                .await?;
            Ok(())
        };
        let config = test_session_config().with_target_partitions(1);
        let sql = "SELECT b.x, COUNT(p.v) AS n FROM b JOIN p ON b.x = p.k GROUP BY b.x";
        let plan = datafusion_context(config.clone());
        setup(&plan).await?;
        let plan = plan.sql(sql).await?.create_physical_plan().await?;
        let mut produced = 0;
        plan.apply(|node| {
            if node.is::<HashJoinExec>() {
                produced += node.dynamic_expressions_produced().len();
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(produced, 1, "{}", displayable(plan.as_ref()).indent(false));
        let root = assert_parity_with(sql, config, &setup).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_dynamic_filter_prunes_reads_that_may_fail_stay_in_datafusion()
    -> Result<()> {
        // A table `k INT, v INT, s VARCHAR` (its schema given when it is
        // registered) of two Parquet files: 100 rows `k = 0, v = 100`, and 10
        // rows `k = 1` of strings `v` that fail to cast to `INT` when they are
        // read. The TopK of `ORDER BY k LIMIT 1` reads the first file,
        // whose rows tighten its threshold to `k <= 0`, and DataFusion's scan
        // prunes the second file with its dynamic filter: it never reads it. A
        // native TopK does not prune the same reads, so that it would fail to
        // cast.
        let dir = tempfile::tempdir()?;
        let first = RecordBatch::try_from_iter([
            (
                "k",
                Arc::new(Int32Array::from(vec![0; 100])) as arrow::array::ArrayRef,
            ),
            ("v", Arc::new(Int32Array::from(vec![100; 100])) as _),
            (
                "s",
                Arc::new(arrow::array::StringViewArray::from(vec!["foo"; 100])) as _,
            ),
        ])?;
        let second = RecordBatch::try_from_iter([
            (
                "k",
                Arc::new(Int32Array::from(vec![1; 10])) as arrow::array::ArrayRef,
            ),
            (
                "v",
                Arc::new(arrow::array::StringViewArray::from(vec!["bad_integer"; 10]))
                    as _,
            ),
            (
                "s",
                Arc::new(arrow::array::StringViewArray::from(vec!["foo"; 10])) as _,
            ),
        ])?;
        let path = dir.path().join("p");
        write_parquet(&path.join("0.parquet"), first.schema(), &[first], None)?;
        write_parquet(&path.join("1.parquet"), second.schema(), &[second], None)?;
        let path = path.to_str().unwrap();
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int32, true),
            Field::new("s", DataType::Utf8View, true),
        ]);
        let setup = async |ctx: &SessionContext| {
            register_view_tables(ctx)?;
            ctx.register_parquet(
                "p",
                path,
                ParquetReadOptions::default().schema(&schema),
            )
            .await?;
            Ok(())
        };
        // The scan casts the strings of the second file into the integers of
        // the table: its expressions cannot fail, but its reads can. So they
        // can below the filter `s LIKE '%foo%'` (on batches of 64 rows, the
        // first of which tightens the threshold of DataFusion's TopK).
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        for sql in [
            "SELECT k, v FROM p ORDER BY k LIMIT 1",
            "SELECT k, v FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1",
        ] {
            let root = assert_parity_with(sql, config.clone(), &setup).await;
            assert_eq!(count::<PhysicalTopK>(&root), 0, "{sql}");
        }
        // Over a scan of memory, which cannot fail, the TopK is native.
        let sql = "SELECT id, a FROM t1 ORDER BY a, id LIMIT 3";
        let root = assert_parity_with(sql, config, &setup).await;
        assert_eq!(count::<PhysicalTopK>(&root), 1);
        Ok(())
    }

    /// A table `p` of two Parquet files `(k INT, s VARCHAR)` in `dir`, with the
    /// rows `first` and `second`: the files are well-formed, their columns have
    /// the types of the table.
    fn two_parquet_files(
        dir: &std::path::Path,
        first: (i32, &str, usize),
        second: (i32, &str, usize),
    ) -> Result<String> {
        let path = dir.join("p");
        for (file, (k, s, rows)) in [first, second].into_iter().enumerate() {
            let batch = RecordBatch::try_from_iter([
                (
                    "k",
                    Arc::new(Int32Array::from(vec![k; rows])) as arrow::array::ArrayRef,
                ),
                (
                    "s",
                    Arc::new(arrow::array::StringViewArray::from(vec![s; rows])) as _,
                ),
            ])?;
            write_parquet(
                &path.join(format!("{file}.parquet")),
                batch.schema(),
                &[batch],
                None,
            )?;
        }
        Ok(path.to_str().unwrap().to_string())
    }

    /// A Parquet file at `path` of `rows` rows `(k INT32, v INT96, s UTF8)`: of
    /// `k`, of the INT96 timestamp of midnight of the Julian day `day`, and of
    /// `s`.
    fn write_int96_parquet(
        path: &std::path::Path,
        (k, day, s): (i32, u32, &str),
        rows: usize,
    ) -> Result<()> {
        use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int96, Int96Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        let schema = Arc::new(parquet::schema::parser::parse_message_type(
            "message schema { REQUIRED INT32 k; REQUIRED INT96 v; \
             REQUIRED BYTE_ARRAY s (UTF8); }",
        )?);
        std::fs::create_dir_all(path.parent().expect("a directory"))?;
        let file = std::fs::File::create(path)?;
        let properties = Arc::new(WriterProperties::builder().build());
        let mut writer = SerializedFileWriter::new(file, schema, properties)?;
        let mut row_group = writer.next_row_group()?;
        let mut column = row_group.next_column()?.expect("k");
        column
            .typed::<Int32Type>()
            .write_batch(&vec![k; rows], None, None)?;
        column.close()?;
        let mut column = row_group.next_column()?.expect("v");
        let mut timestamp = Int96::new();
        timestamp.set_data(0, 0, day);
        column
            .typed::<Int96Type>()
            .write_batch(&vec![timestamp; rows], None, None)?;
        column.close()?;
        let mut column = row_group.next_column()?.expect("s");
        column.typed::<ByteArrayType>().write_batch(
            &vec![ByteArray::from(s); rows],
            None,
            None,
        )?;
        column.close()?;
        row_group.close()?;
        writer.close()?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_over_int96_files_that_the_scan_casts_stay_in_datafusion() -> Result<()>
    {
        // A table `k INT, v TIMESTAMP (ns), s VARCHAR` (its schema given when it
        // is registered) of two Parquet files of INT96 timestamps: 100 rows `k =
        // 0` at the epoch, and 10 rows `k = 1` on the Julian day 3,000,000 (in
        // the year 3501). The session reads INT96 timestamps in microseconds
        // (`coerce_int96`), from which the scan casts them into the
        // nanoseconds of the table: the timestamps of the second file overflow.
        // The TopK of `ORDER BY k LIMIT 1` reads the first file, whose first
        // batch of 64 rows tightens its threshold to `k < 0`, and DataFusion's
        // scan prunes the second file. Its reads may fail (the files do not have
        // the types of the table, once read as the session reads them), so the
        // TopK stays in DataFusion.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("p");
        write_int96_parquet(&path.join("0.parquet"), (0, 2_440_588, "foo"), 100)?;
        write_int96_parquet(&path.join("1.parquet"), (1, 3_000_000, "foo"), 10)?;
        let path = path.to_str().unwrap();
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new(
                "v",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new("s", DataType::Utf8View, false),
        ]);
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", path, ParquetReadOptions::default().schema(&schema))
                .await
        };
        let mut config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        config
            .options_mut()
            .set("datafusion.execution.parquet.coerce_int96", "us")?;
        for sql in [
            "SELECT k, v FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1",
            "SELECT k, v FROM p ORDER BY k LIMIT 1",
        ] {
            let root = assert_parity_with(sql, config.clone(), &setup).await;
            assert_eq!(
                count::<PhysicalTopK>(&root),
                0,
                "{sql}\n{}",
                format_operator_tree(&root)
            );
        }
        // Without `coerce_int96`, the reader reads the timestamps in the
        // nanoseconds of the table (the second file, whose timestamps do not
        // fit, is then read as DataFusion reads it), and the TopK is native.
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        let sql = "SELECT k, s FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1";
        let root = assert_parity_with(sql, config, &setup).await;
        assert_eq!(
            count::<PhysicalTopK>(&root),
            1,
            "{}",
            format_operator_tree(&root)
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_dynamic_filter_prunes_casts_that_may_fail_stay_in_datafusion()
    -> Result<()> {
        // A table `k INT, s VARCHAR` of two Parquet files: 100 rows `k = 1` of
        // integers, and 10 rows `k = 0` of strings that fail to cast to `INT`.
        // DataFusion pushes the dynamic filter of the join (`k = 1`, from its
        // build side) into the scan, which prunes the second file: it never
        // casts its strings. A native join does not prune the same reads, so
        // that it could fail to cast.
        let dir = tempfile::tempdir()?;
        let path = two_parquet_files(dir.path(), (1, "7", 100), (0, "bad_integer", 10))?;
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await?;
            ctx.sql("CREATE TABLE b AS SELECT 1 AS x")
                .await?
                .collect()
                .await?;
            Ok(())
        };
        let config = test_session_config().with_target_partitions(1);
        let sql = "SELECT b.x, COUNT(q.v) AS n FROM b \
                   JOIN (SELECT k, CAST(s AS INT) AS v FROM p) q ON b.x = q.k GROUP BY b.x";
        let plan = datafusion_plan_with(sql, config.clone(), &setup).await;
        let mut produced = 0;
        plan.apply(|node| {
            if node.is::<HashJoinExec>() {
                produced += node.dynamic_expressions_produced().len();
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(produced, 1, "{}", displayable(plan.as_ref()).indent(false));
        let root = assert_parity_with(sql, config.clone(), &setup).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        // Without the cast, the work of the probe side cannot fail on the rows
        // that only one of the engines prunes (the files are well-formed): the
        // join is native.
        let sql = "SELECT b.x, COUNT(p.s) AS n FROM b JOIN p ON b.x = p.k GROUP BY b.x";
        let root = assert_parity_with(sql, config, &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_dynamic_filter_prunes_casts_that_may_fail_stay_in_datafusion()
    -> Result<()> {
        // A table `k INT, s VARCHAR` of two Parquet files: 100 rows `k = 0, s =
        // '100'`, and 10 rows `k = 1` of strings that fail to cast to `INT`. The
        // TopK of `ORDER BY k LIMIT 1` reads the first file, whose rows tighten
        // its threshold to `k < 0`, and DataFusion's scan prunes the second
        // file with its dynamic filter: it never casts its strings (which the
        // projection of the scan does). A native TopK tightens its threshold at
        // other times, so that its scan could read the second file, and fail
        // to cast.
        let dir = tempfile::tempdir()?;
        let path =
            two_parquet_files(dir.path(), (0, "100", 100), (1, "bad_integer", 10))?;
        let setup = async |ctx: &SessionContext| {
            register_view_tables(ctx)?;
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await?;
            Ok(())
        };
        let config = test_session_config().with_target_partitions(1);
        let sql = "SELECT k, CAST(s AS INT) AS v FROM p ORDER BY k LIMIT 1";
        let root = assert_parity_with(sql, config.clone(), &setup).await;
        assert_eq!(
            count::<PhysicalTopK>(&root),
            0,
            "{sql}\n{}",
            format_operator_tree(&root)
        );
        // Without the cast, the work on the rows that only one of the engines
        // prunes cannot fail (the files are well-formed): the TopK is native.
        let sql = "SELECT k, s FROM p ORDER BY k LIMIT 1";
        let root = assert_parity_with(sql, config.clone(), &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalTopK>(&root), 1);
        // Over a scan of memory, which cannot fail, the TopK is native.
        let sql = "SELECT id, a FROM t1 ORDER BY a, id LIMIT 3";
        let root = assert_parity_with(sql, config, &setup).await;
        assert_eq!(count::<PhysicalTopK>(&root), 1);
        Ok(())
    }

    /// The plan of `sql` through the [`PipelineQueryPlanner`] (a
    /// [`PipelineRootExec`]), in a session configured with `config` and set up
    /// with `setup`, and the session.
    async fn pipeline_plan<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> (Arc<dyn ExecutionPlan>, SessionContext)
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = pipeline_context(config);
        setup(&ctx).await.unwrap();
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        assert!(plan.is::<PipelineRootExec>(), "{sql}");
        (plan, ctx)
    }

    /// The dynamic filter of the `PhysicalTopK` of `plan`, a
    /// [`PipelineRootExec`], which must be the dynamic filter of the `SortExec`
    /// of the plan it was lowered from, which a node of the input of the
    /// `SortExec` reads.
    fn filter_of_native_topk(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Arc<DynamicFilterPhysicalExpr> {
        let exec = plan.downcast_ref::<PipelineRootExec>().unwrap();
        topk_updating_the_filter_of(exec.operator(), exec.input())
    }

    /// The `PhysicalTopK` of the operator tree `root` (through its first
    /// children), whose dynamic filter must be the dynamic filter of the
    /// `SortExec` of `plan`, the plan that `root` was lowered from, which a
    /// node of the input of the `SortExec` reads.
    fn topk_updating_the_filter_of(
        root: &Arc<dyn PhysicalOperator>,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Arc<DynamicFilterPhysicalExpr> {
        let mut sorts = vec![];
        plan.apply(|node| {
            if let Some(sort) = node.downcast_ref::<SortExec>() {
                sorts.extend(read_topk_filter(sort));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        let [(filter, id)] = sorts.as_slice() else {
            panic!("no TopK whose filter is read: {sorts:?}");
        };
        let topk = std::iter::successors(Some(Arc::clone(root)), |operator| {
            operator.children().first().map(Arc::clone)
        })
        .find(|operator| operator.is::<PhysicalTopK>())
        .unwrap_or_else(|| panic!("no native TopK: {}", format_operator_tree(root)));
        let topk = topk.downcast_ref::<PhysicalTopK>().unwrap();
        let updated = topk.dynamic_filter().expect("a dynamic filter").filter();
        assert_eq!(updated.expression_id(), Some(*id));
        assert!(Arc::ptr_eq(updated, filter));
        Arc::clone(filter)
    }

    /// `files` Parquet files of 500 rows `(k BIGINT, v BIGINT, s VARCHAR)` in
    /// `dir`, in order of `k` (from 0), with `v = 1` but on the rows `zeros`,
    /// and `s` the digits of `k`.
    fn parquet_files_in_order(
        dir: &std::path::Path,
        files: i64,
        zeros: &[i64],
    ) -> Result<String> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        for file in 0..files {
            let k = (file * 500..(file + 1) * 500).collect::<Vec<_>>();
            let v = k.iter().map(|k| i64::from(!zeros.contains(k)));
            let s = k.iter().map(i64::to_string);
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(arrow::array::Int64Array::from(k.clone())),
                    Arc::new(arrow::array::Int64Array::from_iter_values(v)),
                    Arc::new(arrow::array::StringViewArray::from_iter_values(s)),
                ],
            )?;
            let path = dir.join(format!("p/{file:02}.parquet"));
            write_parquet(&path, Arc::clone(&schema), &[batch], None)?;
        }
        Ok(dir.join("p").to_str().unwrap().to_string())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_dynamic_filter_prunes_nothing_are_native() -> Result<()> {
        // DataFusion pushes the dynamic filter of the TopK of `GROUP BY k ORDER
        // BY k LIMIT 3` through the aggregation by `k` into the scan (as the
        // filter of the TopK of TPC-H Q18 into the scan of `orders`), but the
        // aggregation reads all of its input before the TopK sees its first
        // group: in both engines, the filter has no threshold while the scan
        // reads its files, so that it prunes nothing. The TopK is native, and
        // updates the filter, also when the work below the aggregation may fail
        // (`10 / v`, which divides by zero on a row of the last file), on any
        // number of threads.
        let dir = tempfile::tempdir()?;
        let path = parquet_files_in_order(dir.path(), 8, &[3_900])?;
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await
        };
        for target_partitions in [1, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for (sql, error) in [
                (
                    "SELECT k, SUM(v) AS n FROM p GROUP BY k ORDER BY k DESC LIMIT 3",
                    None,
                ),
                (
                    "SELECT k, COUNT(*) AS n FROM p WHERE s LIKE '%7%' \
                     GROUP BY k ORDER BY k LIMIT 5",
                    None,
                ),
                (
                    "SELECT k, SUM(10 / v) AS n FROM p GROUP BY k ORDER BY k LIMIT 3",
                    Some("Divide by zero"),
                ),
            ] {
                match error {
                    None => {
                        let root = assert_parity_with(sql, config.clone(), &setup).await;
                        assert_native(sql, &root);
                    }
                    Some(error) => {
                        assert_fails_like_datafusion(sql, config.clone(), &setup, error)
                            .await;
                    }
                }
                let (plan, ctx) = pipeline_plan(sql, config.clone(), &setup).await;
                let filter = filter_of_native_topk(&plan);
                if error.is_none() {
                    collect(plan, ctx.task_ctx()).await?;
                    assert!(filter.is_complete(), "{sql}");
                    assert_ne!(filter.current()?.to_string(), "true", "{sql}");
                }
            }
        }
        Ok(())
    }

    /// A stream of a single batch, `batch`, which it emits at once, and ends
    /// only after `delay`.
    #[derive(Debug)]
    struct EndsLate {
        batch: RecordBatch,
        delay: std::time::Duration,
    }

    impl datafusion_physical_plan::streaming::PartitionStream for EndsLate {
        fn schema(&self) -> &SchemaRef {
            self.batch.schema_ref()
        }

        fn execute(
            &self,
            _ctx: Arc<TaskContext>,
        ) -> datafusion_execution::SendableRecordBatchStream {
            use futures::StreamExt;
            let batch = self.batch.clone();
            let delay = self.delay;
            let end = futures::stream::once(async move {
                tokio::time::sleep(delay).await;
            })
            .filter_map(|()| async { None });
            let stream = futures::stream::once(async move { Ok(batch) }).chain(end);
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                    self.batch.schema(),
                    stream,
                ),
            )
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_filter_is_read_below_one_of_several_inputs_stay_in_datafusion()
    -> Result<()> {
        // `ORDER BY k LIMIT 1` of the union of a stream of the row `(-1, 7)`,
        // which ends late, and of the groups of an aggregation of 40 Parquet
        // files of `k >= 0`, with `v = 0` on a row of the last file. DataFusion
        // pushes the dynamic filter of the TopK into both inputs of the union,
        // and through the aggregation into the scan. The aggregation reads all
        // of its input before it emits any group, but the other input of the
        // union does not wait for it: DataFusion's TopK tightens the filter to
        // `k < -1` with the row of the stream, while the scan still reads its
        // files, so that it skips the last one, and never divides by zero. A
        // native TopK tightens it only once the stream ends, or once it holds
        // more rows, after the scan read all of its files: the filter of such a
        // TopK prunes the rows below the aggregation, which may fail, so it
        // stays in DataFusion.
        let dir = tempfile::tempdir()?;
        let path = parquet_files_in_order(dir.path(), 40, &[19_999])?;
        let row = RecordBatch::try_from_iter([
            (
                "k",
                Arc::new(arrow::array::Int64Array::from(vec![-1]))
                    as arrow::array::ArrayRef,
            ),
            ("n", Arc::new(arrow::array::Int64Array::from(vec![7])) as _),
        ])?;
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await?;
            let stream = EndsLate {
                batch: row.clone(),
                delay: std::time::Duration::from_millis(500),
            };
            let table = datafusion::catalog::streaming::StreamingTable::try_new(
                row.schema(),
                vec![Arc::new(stream)],
            )?;
            ctx.register_table("s", Arc::new(table))?;
            Ok(())
        };
        let sql = "SELECT k, n FROM (SELECT k, n FROM s \
                   UNION ALL SELECT k, SUM(10 / v) AS n FROM p GROUP BY k) \
                   ORDER BY k LIMIT 1";
        // On a single thread, on which DataFusion reads the files in order.
        {
            let config = test_session_config().with_target_partitions(1);
            let plan = datafusion_plan_with(sql, config.clone(), &setup).await;
            let mut read = 0;
            plan.apply(|node| {
                if let Some(sort) = node.downcast_ref::<SortExec>() {
                    read += usize::from(read_topk_filter(sort).is_some());
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
            assert!(read > 0, "{}", displayable(plan.as_ref()).indent(false));
            let root = assert_parity_with(sql, config, &setup).await;
            assert_eq!(
                count::<PhysicalTopK>(&root),
                0,
                "{}",
                format_operator_tree(&root)
            );
        }
        Ok(())
    }

    /// A node of one partition that emits `row` before it reads its input,
    /// whose rows it then passes on, and that passes the filters of the nodes
    /// above it on to its input.
    #[derive(Debug)]
    struct EmitsRowFirst {
        row: RecordBatch,
        input: Arc<dyn ExecutionPlan>,
        properties: Arc<datafusion_physical_plan::PlanProperties>,
    }

    impl EmitsRowFirst {
        fn new(row: RecordBatch, input: Arc<dyn ExecutionPlan>) -> Self {
            use datafusion_physical_expr::EquivalenceProperties;
            use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
            let properties = datafusion_physical_plan::PlanProperties::new(
                EquivalenceProperties::new(input.schema()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            );
            Self {
                row,
                input,
                properties: Arc::new(properties),
            }
        }
    }

    impl datafusion_physical_plan::DisplayAs for EmitsRowFirst {
        fn fmt_as(
            &self,
            _t: datafusion_physical_plan::DisplayFormatType,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(f, "EmitsRowFirst")
        }
    }

    impl ExecutionPlan for EmitsRowFirst {
        fn name(&self) -> &str {
            "EmitsRowFirst"
        }

        fn properties(&self) -> &Arc<datafusion_physical_plan::PlanProperties> {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self::new(
                self.row.clone(),
                Arc::clone(&children[0]),
            )))
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<datafusion_execution::SendableRecordBatchStream> {
            use futures::StreamExt;
            let input = self.input.execute(partition, context)?;
            let row = self.row.clone();
            let stream = futures::stream::once(async move { Ok(row) }).chain(input);
            Ok(Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                    self.schema(),
                    stream,
                ),
            ))
        }

        fn gather_filters_for_pushdown(
            &self,
            _phase: datafusion_physical_plan::filter_pushdown::FilterPushdownPhase,
            parent_filters: Vec<Arc<dyn PhysicalExpr>>,
            _config: &ConfigOptions,
        ) -> Result<datafusion_physical_plan::filter_pushdown::FilterDescription>
        {
            datafusion_physical_plan::filter_pushdown::FilterDescription::from_children(
                parent_filters,
                &self.children(),
            )
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_filter_is_read_below_nodes_that_emit_rows_first_stay_in_datafusion()
    -> Result<()> {
        // A TopK of `ORDER BY k LIMIT 1` over a node that emits the row `(-1,
        // 7)` before it reads the groups of an aggregation of 40 Parquet files
        // of `k >= 0`, with `v = 0` on a row of the last file. DataFusion pushes
        // the dynamic filter of the TopK through the node and the aggregation
        // into the scan, and tightens it to `k < -1` with the first row of the
        // node, before the aggregation reads any file: its scan skips all of
        // them, and never divides by zero. The aggregation reads all of its
        // input before it emits any group, but the node above it does not wait
        // for it: the native TopK would tighten the filter only after the
        // aggregation read all files. Such a TopK stays in DataFusion.
        let dir = tempfile::tempdir()?;
        let path = parquet_files_in_order(dir.path(), 40, &[19_999])?;
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await
        };
        let config = test_session_config().with_target_partitions(1);
        let options = config.options();
        // The TopK, with its dynamic filter pushed down (every execution needs
        // a plan of its own, as the TopK counts the partitions that complete
        // its filter).
        let make_topk = async || -> Result<Arc<dyn ExecutionPlan>> {
            let groups = datafusion_plan_with(
                "SELECT k, SUM(10 / v) AS n FROM p GROUP BY k",
                config.clone(),
                &setup,
            )
            .await;
            let schema = groups.schema();
            let row = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(arrow::array::Int64Array::from(vec![-1])),
                    Arc::new(arrow::array::Int64Array::from(vec![7])),
                ],
            )?;
            let keys =
                LexOrdering::new(vec![PhysicalSortExpr::new_default(col("k", &schema)?)])
                    .expect("a sort key");
            let topk: Arc<dyn ExecutionPlan> = Arc::new(
                SortExec::new(keys, Arc::new(EmitsRowFirst::new(row, groups)))
                    .with_fetch(Some(1)),
            );
            datafusion_physical_optimizer::PhysicalOptimizerRule::optimize(
                &datafusion_physical_optimizer::filter_pushdown::FilterPushdown::new_post_optimization(),
                topk,
                options,
            )
        };
        let topk = make_topk().await?;
        let sort = topk.downcast_ref::<SortExec>().expect("the TopK");
        assert!(
            read_topk_filter(sort).is_some(),
            "{}",
            displayable(topk.as_ref()).indent(false)
        );
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(config.clone()));
        let expected = collect(Arc::clone(&topk), Arc::clone(&task_ctx)).await?;
        let runtime = RuntimeEnv::default();
        let root =
            create_pipeline_plan_with_runtime(make_topk().await?, options, &runtime)?;
        let tree = format_operator_tree(root.operator());
        let topks = count::<PhysicalTopK>(root.operator());
        let actual = collect(root as Arc<dyn ExecutionPlan>, task_ctx)
            .await
            .unwrap_or_else(|e| panic!("the pipeline engine fails: {e}\n{tree}"));
        let output = |batches| crate::test_utils::EngineOutput {
            schema: topk.schema(),
            plan: Arc::clone(&topk),
            batches,
            ordered: true,
        };
        compare_outputs(&output(expected), &output(actual))
            .unwrap_or_else(|message| panic!("Results differ: {message}\n{tree}"));
        assert_eq!(topks, 0, "{tree}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_filter_is_read_below_the_probe_side_of_a_join_that_settles_it_are_native()
    -> Result<()> {
        // `ORDER BY k DESC LIMIT 3` of the groups of an aggregation of 20
        // Parquet files whose keys are among the groups of another aggregation,
        // with `Partitioned` hash joins (as TPC-H Q18 at SF 1): DataFusion plans
        // a semi join whose probe side is the first aggregation (and the second
        // one its build side), and pushes the
        // dynamic filter of the TopK through the join and the aggregation into
        // the scan. The aggregation reads all of its input before it emits any
        // group, and the join emits its rows as its probe side passes them on:
        // its build side, which the join reads first and which does not read the
        // filter, does not pass rows on to the TopK by itself (unlike the inputs
        // of a union). The filter prunes nothing before the scan read all of its
        // files, in both engines, and the TopK is native.
        let dir = tempfile::tempdir()?;
        let path = parquet_files_in_order(dir.path(), 20, &[])?;
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await?;
            ctx.sql(
                "CREATE TABLE small AS SELECT CAST(value AS BIGINT) AS x \
                 FROM generate_series(0, 9999, 7)",
            )
            .await?
            .collect()
            .await?;
            Ok(())
        };
        let config = partitioned_config();
        let sql = "SELECT g.k, g.n FROM (SELECT k, SUM(v) AS n FROM p GROUP BY k) g \
                   WHERE g.k IN (SELECT x FROM small GROUP BY x HAVING COUNT(*) > 0) \
                   ORDER BY g.k DESC LIMIT 3";
        let plan = datafusion_plan_with(sql, config.clone(), &setup).await;
        let shape = displayable(plan.as_ref()).indent(false).to_string();
        assert!(
            shape.contains("HashJoinExec: mode=Partitioned, join_type=RightSemi"),
            "{shape}"
        );
        let mut read = 0;
        plan.apply(|node| {
            if let Some(sort) = node.downcast_ref::<SortExec>() {
                read += usize::from(read_topk_filter(sort).is_some());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(read, 1, "{}", displayable(plan.as_ref()).indent(false));
        let root = assert_parity_with(sql, config.clone(), &setup).await;
        assert_eq!(
            count::<PhysicalTopK>(&root),
            1,
            "{}",
            format_operator_tree(&root)
        );
        let (plan, _) = pipeline_plan(sql, config, &setup).await;
        filter_of_native_topk(&plan);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_over_scans_through_nodes_that_pass_rows_on_update_their_filter()
    -> Result<()> {
        // 20 Parquet files of 500 rows, in order of `k`. Between the TopK and
        // the scan that reads its dynamic filter, the filter `s LIKE '%7%'` (of
        // ClickBench's `hits23`) and the projections only pass rows on, and the
        // work on the rows that the filter prunes cannot fail: the TopK is
        // native, and updates the filter, so that the scan prunes the files that
        // its threshold rules out, as in DataFusion.
        let dir = tempfile::tempdir()?;
        let path = parquet_files_in_order(dir.path(), 20, &[])?;
        let setup = async |ctx: &SessionContext| {
            register_view_tables(ctx)?;
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await
        };
        for target_partitions in [1, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT * FROM p WHERE s LIKE '%7%' ORDER BY k LIMIT 10",
                "SELECT k, s FROM p WHERE s <> '' AND v > 0 ORDER BY v DESC, k LIMIT 7",
                "SELECT k + 1 AS j, s FROM p WHERE s LIKE '1%' ORDER BY k DESC LIMIT 3",
            ] {
                let root = assert_parity_with(sql, config.clone(), &setup).await;
                assert_native(sql, &root);
                let (plan, _) = pipeline_plan(sql, config.clone(), &setup).await;
                filter_of_native_topk(&plan);
            }
        }
        // On a single thread, the files after the first ones are pruned once
        // the threshold is known.
        let config = test_session_config().with_target_partitions(1);
        let sql = "SELECT * FROM p WHERE s LIKE '%7%' ORDER BY k LIMIT 10";
        let (plan, ctx) = pipeline_plan(sql, config.clone(), &setup).await;
        let filter = filter_of_native_topk(&plan);
        collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert!(filter.is_complete());
        let exec = plan.downcast_ref::<PipelineRootExec>().unwrap();
        let metrics = exec.execution_metrics().expect("metrics");
        let (_, scan) = metrics
            .iter()
            .find(|(operator, _)| operator.is::<ParquetScanSource>())
            .expect("the scan");
        assert!(
            scan.output_rows() < 10_000,
            "the scan read {} rows",
            scan.output_rows()
        );
        // Through the probe side of a join, which is no node that only passes
        // rows on (as `crate::sort::try_lower` judges it), the TopK is left to
        // DataFusion.
        let sql = "SELECT p.k, p.s FROM p JOIN t2 ON p.k = t2.k \
                   ORDER BY p.k DESC, p.s LIMIT 3";
        let plan = datafusion_plan_with(sql, config.clone(), &setup).await;
        let mut read = 0;
        plan.apply(|node| {
            if let Some(sort) = node.downcast_ref::<SortExec>() {
                read += usize::from(read_topk_filter(sort).is_some());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(read, 1, "{}", displayable(plan.as_ref()).indent(false));
        let root = assert_parity_with(sql, config, &setup).await;
        assert_eq!(count::<PhysicalTopK>(&root), 0, "{sql}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fallible_operators_on_several_threads_are_left_to_datafusion() -> Result<()>
    {
        // One batch of 8192 rows: `CAST(s AS INT)` fails on row 0, a string of a
        // million zeros and `x` (slow to parse), and on row 2048, `y`.
        // DataFusion evaluates the rows of a batch in order, so it fails on row
        // 0. On several threads, the native operators evaluate the chunks of
        // the batch in parallel, and the thread of row 2048 would fail first.
        let mut s = vec!["1".to_string(); 8192];
        s[0] = format!("{}x", "0".repeat(1_000_000));
        s[2048] = "y".to_string();
        let batch = RecordBatch::try_from_iter([(
            "s",
            Arc::new(arrow::array::StringViewArray::from(s)) as arrow::array::ArrayRef,
        )])?;
        let setup = single_batch_table("f", batch);
        let error = "Cannot cast string '0000000000";
        // The queries, with the number of native operators that evaluate the
        // cast.
        let casts = |root: &Arc<dyn PhysicalOperator>, operator: &str| match operator {
            "projection" => count::<PhysicalProjection>(root),
            "filter" => count::<PhysicalFilter>(root),
            "aggregation" => count::<PhysicalHashAggregate>(root),
            _ => count::<PhysicalSort>(root),
        };
        for (sql, operator) in [
            ("SELECT CAST(s AS INT) AS x FROM f", "projection"),
            ("SELECT s FROM f WHERE CAST(s AS INT) > 0", "filter"),
            ("SELECT SUM(CAST(s AS INT)) AS x FROM f", "aggregation"),
            ("SELECT s FROM f ORDER BY CAST(s AS INT)", "sort"),
        ] {
            let config = test_session_config();
            assert_fails_like_datafusion(sql, config.clone(), &setup, error).await;
            let root = lowered(sql, config, &setup).await;
            assert_eq!(
                casts(&root, operator),
                0,
                "{sql}\n{}",
                format_operator_tree(&root)
            );
            // On one thread, they evaluate the rows in order, natively.
            let config = test_session_config().with_target_partitions(1);
            assert_fails_like_datafusion(sql, config.clone(), &setup, error).await;
            let root = lowered(sql, config, &setup).await;
            assert!(bridged_plans(&root).is_empty(), "{sql}");
            assert_eq!(casts(&root, operator), 1, "{sql}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn redistributions_are_removed() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        let batch = |values: std::ops::Range<i32>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from_iter_values(values))],
            )
            .unwrap()
        };
        let scan = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(MemorySourceConfig::try_new_exec(
                &[vec![batch(0..3000)], vec![], vec![batch(3000..3010)]],
                Arc::clone(&schema),
                None,
            )?)
        };
        let v = col("v", &schema)?;
        let fallible: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
            datafusion_expr::Operator::Divide,
            Arc::clone(&v),
        ));
        let plans: Vec<(Arc<dyn ExecutionPlan>, bool)> = vec![
            (
                Arc::new(RepartitionExec::try_new(
                    Arc::new(CoalescePartitionsExec::new(scan()?)),
                    Partitioning::RoundRobinBatch(4),
                )?),
                true,
            ),
            (
                Arc::new(RepartitionExec::try_new(
                    scan()?,
                    Partitioning::Hash(vec![Arc::clone(&v)], 4),
                )?),
                true,
            ),
            (Arc::new(coalesce_batches(scan()?, None)), true),
            // DataFusion evaluates the hash expressions on every batch.
            (
                Arc::new(RepartitionExec::try_new(
                    scan()?,
                    Partitioning::Hash(vec![fallible], 4),
                )?),
                false,
            ),
            // A coalescing with a limit.
            (Arc::new(coalesce_batches(scan()?, Some(5))), false),
        ];
        for (plan, removed) in plans {
            let description = displayable(plan.as_ref()).indent(false).to_string();
            let root = lower_execution_plan(&plan)?;
            if removed {
                assert!(root.is::<MemoryScanSource>(), "{description}");
            } else {
                assert!(root.is::<BridgeOperator>(), "{description}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn intermediate_aggregation_stages_are_removed() -> Result<()> {
        // `Final -> CoalesceBatches -> Repartition -> PartialReduce ->
        // Repartition -> Partial`, from DataFusion's `Final -> Repartition ->
        // Partial`. The `PartialReduce` merges the states into the same states,
        // so the native aggregation does without both stages, unless
        // concatenating the states may fail (strings with 32-bit offsets, here
        // the group key `d`), as `CoalesceBatchesExec` does.
        for (sql, native) in [
            (
                "SELECT e, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh FROM t1 GROUP BY e",
                true,
            ),
            (
                "SELECT d, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh FROM t1 GROUP BY d",
                false,
            ),
        ] {
            let make_plan = || async {
                let plan = datafusion_plan(sql, test_session_config()).await;
                plan.transform_down(|node| {
                    let Some(last) = node.downcast_ref::<AggregateExec>() else {
                        return Ok(Transformed::no(node));
                    };
                    if last.mode() != &AggregateMode::FinalPartitioned {
                        return Ok(Transformed::no(node));
                    }
                    let repartition = Arc::clone(last.input());
                    let partial = Arc::clone(repartition.children()[0]);
                    let reduce: Arc<dyn ExecutionPlan> =
                        Arc::new(AggregateExec::try_new(
                            AggregateMode::PartialReduce,
                            last.group_expr().clone(),
                            last.aggr_expr().to_vec(),
                            last.filter_expr().to_vec(),
                            Arc::clone(&partial),
                            last.input_schema(),
                        )?);
                    assert_eq!(reduce.schema(), partial.schema());
                    let below = Arc::clone(&repartition)
                        .replace_children(vec![reduce], recompute())?;
                    let above = repartition.replace_children(vec![below], recompute())?;
                    let coalesce: Arc<dyn ExecutionPlan> =
                        Arc::new(coalesce_batches(above, None));
                    let plan = Arc::clone(&node)
                        .replace_children(vec![coalesce], recompute())?;
                    Ok(Transformed::yes(plan))
                })
                .unwrap()
                .data
            };
            let plan = make_plan().await;
            assert_eq!(count_nodes(&plan, "AggregateExec"), 3);
            let root = lower_execution_plan(&plan)?;
            if native {
                assert_native(sql, &root);
                assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
            } else {
                assert_eq!(count::<PhysicalHashAggregate>(&root), 0, "{sql}");
            }

            let task_ctx = Arc::new(
                TaskContext::default().with_session_config(test_session_config()),
            );
            let expected = collect(make_plan().await, Arc::clone(&task_ctx)).await?;
            let root: Arc<dyn ExecutionPlan> = create_pipeline_plan(make_plan().await)?;
            let actual = collect(Arc::clone(&root), task_ctx).await?;
            let output = |plan: &Arc<dyn ExecutionPlan>, batches| {
                crate::test_utils::EngineOutput {
                    schema: plan.schema(),
                    plan: Arc::clone(plan),
                    batches,
                    ordered: false,
                }
            };
            compare_outputs(&output(&plan, expected), &output(&root, actual))
                .unwrap_or_else(|message| panic!("Results differ: {message}\n{sql}"));
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scans_of_files_are_native() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let batches = test_table_t1()?.into_iter().flatten().collect::<Vec<_>>();
        let (parquet, csv) = (dir.path().join("p"), dir.path().join("c"));
        for (i, batches) in batches.chunks(2).enumerate() {
            let schema = batches[0].schema();
            write_parquet(&parquet.join(format!("{i}.parquet")), schema, batches, None)?;
            let columns = [0, 1, 2, 7];
            let projected = batches
                .iter()
                .map(|batch| batch.project(&columns))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            write_csv(&csv.join(format!("{i}.csv")), &projected)?;
        }
        let (parquet, csv) = (parquet.to_str().unwrap(), csv.to_str().unwrap());
        let setup = async |ctx: &SessionContext| {
            register_view_tables(ctx)?;
            ctx.register_parquet("p", parquet, ParquetReadOptions::default())
                .await?;
            ctx.register_csv("c", csv, CsvReadOptions::new()).await?;
            Ok(())
        };
        let config = test_session_config();
        for (sql, source) in [
            (
                "SELECT d, COUNT(*) AS n, SUM(a) AS sa FROM p GROUP BY d ORDER BY d",
                "ParquetScanSource",
            ),
            (
                "SELECT a, COUNT(*) AS n, MAX(g) AS mg FROM c GROUP BY a",
                "CsvScanSource",
            ),
        ] {
            let root = assert_parity_with(sql, config.clone(), &setup).await;
            assert_native(sql, &root);
            let sources =
                count::<ParquetScanSource>(&root) + count::<CsvScanSource>(&root);
            assert_eq!(sources, 1, "{sql}");
            assert!(format_operator_tree(&root).contains(source), "{sql}");
        }

        // DataFusion pushes the dynamic filter of a hash join into the Parquet
        // scan of its probe side: whether a native join can take it on is up
        // to `PhysicalHashJoin::try_lower`, and the results are DataFusion's
        // either way. Without the filter, the join is native.
        let sql =
            "SELECT t2.name, COUNT(*) AS n FROM p JOIN t2 ON p.a = t2.k GROUP BY t2.name";
        assert_parity_with(sql, config.clone(), &setup).await;
        let mut config = config;
        config
            .options_mut()
            .optimizer
            .enable_join_dynamic_filter_pushdown = false;
        let root = assert_parity_with(sql, config, &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Scalar subqueries
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scalar_subqueries_are_native() -> Result<()> {
        // A `ScalarSubqueryExec` is lowered to a `PhysicalScalarSubquery`,
        // which runs its subqueries (natively too) before the pipelines of its
        // input, whose expressions read their results: constants, also in the
        // arithmetic of a projection, which only overflows, on 4 threads.
        let q11 = "SELECT a, SUM(g * b) AS s FROM t1 GROUP BY a \
                   HAVING SUM(g * b) > (SELECT SUM(g * b) * 0.0001 FROM t1)";
        for (sql, subqueries) in [
            ("SELECT id, a FROM t1 WHERE a > (SELECT AVG(a) FROM t1)", 1),
            (
                "SELECT id, g - (SELECT AVG(g) FROM t1) AS diff FROM t1 WHERE id < 100",
                1,
            ),
            // Several subqueries, and a subquery of a subquery.
            (
                "SELECT id FROM t1 WHERE a = (SELECT MAX(k) FROM t2) \
                 AND b > (SELECT MIN(w) FROM t2)",
                1,
            ),
            (
                "SELECT id FROM t1 \
                 WHERE a > (SELECT AVG(k) FROM t2 WHERE k > (SELECT MIN(a) FROM t1))",
                2,
            ),
            // A subquery whose aggregate is null, and one without rows, whose
            // result is a null.
            (
                "SELECT id, (SELECT MAX(k) FROM t2 WHERE w < -1000000) AS m \
                 FROM t1 WHERE id < 10",
                1,
            ),
            (
                "SELECT id, (SELECT MAX(k) FROM t2 HAVING MAX(k) > 1000) AS m \
                 FROM t1 WHERE id < 10",
                1,
            ),
            // The shape of TPC-H Q11, see below.
            (q11, 1),
        ] {
            let root = assert_parity(sql).await;
            assert_native(sql, &root);
            assert_eq!(count::<PhysicalScalarSubquery>(&root), subqueries, "{sql}");
        }
        // Like TPC-H Q11, the filter casts the sums of its groups into the type
        // in which it compares them with the result, which the projection of
        // the subquery casts into that type: casts of numbers into decimals,
        // which only fail on the values that overflow the decimal (see
        // `Failures::CAST_OVERFLOW`), and which the native filter and
        // projection evaluate on 4 threads.
        let mut casts = vec![];
        datafusion_plan(q11, test_session_config())
            .await
            .apply(|node| {
                let (exprs, input) = if let Some(filter) =
                    node.downcast_ref::<FilterExec>()
                {
                    (vec![Arc::clone(filter.predicate())], filter.input())
                } else if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
                    let exprs =
                        projection.expr().iter().map(|expr| Arc::clone(&expr.expr));
                    (exprs.collect(), projection.input())
                } else {
                    return Ok(TreeNodeRecursion::Continue);
                };
                let schema = input.schema();
                casts.extend(
                    exprs
                        .iter()
                        .map(|expr| failures(expr, &schema))
                        .filter(|failures| !failures.is_none()),
                );
                Ok(TreeNodeRecursion::Continue)
            })?;
        assert_eq!(casts, [Failures::CAST_OVERFLOW; 2], "{q11}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_of_scalar_subqueries_are_those_of_datafusion() -> Result<()> {
        // DataFusion runs the subqueries before the input, so that their errors
        // come first, and fails on a subquery of more than one row, as do the
        // native scalar subqueries (which leave the subqueries that may produce
        // more than one row to DataFusion, as below a limit).
        for (sql, error) in [
            (
                "SELECT id FROM t1 WHERE a = (SELECT k FROM t2)",
                "Scalar subquery returned more than one row",
            ),
            (
                "SELECT 10 / (a - a) AS x FROM t1 \
                 WHERE b > (SELECT MAX(CAST(name AS INT)) FROM t2)",
                "Cannot cast string 'name-",
            ),
        ] {
            assert_outcome(sql, Some(error)).await;
        }

        // A filter that casts a column of 4 partitions into the type of the
        // result of a subquery, which only fails on the row whose value
        // overflows the decimal: by more than its integer holds (a cast
        // error), or by more than its precision (an invalid argument). The
        // native filter fails on it like DataFusion, on 4 threads.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "x",
            DataType::Decimal128(36, 2),
            true,
        )]));
        let sql = "SELECT COUNT(*) AS n FROM f \
                   WHERE CAST(x AS DECIMAL(38, 15)) > \
                   (SELECT CAST(MIN(x) AS DECIMAL(38, 15)) FROM f)";
        for (overflowing, error) in [
            (
                10_i128.pow(30),
                "Cannot cast to Decimal128(38, 15). Overflowing on 1000000000000000000000000000000",
            ),
            (
                10_i128.pow(25),
                "is too large to store in a Decimal128 of precision 38",
            ),
        ] {
            let batch = |values: Vec<Option<i128>>| -> Result<RecordBatch> {
                let values = arrow::array::Decimal128Array::from(values)
                    .with_precision_and_scale(36, 2)?;
                Ok(RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(values)],
                )?)
            };
            let mut partitions = (0..4)
                .map(|_| Ok(vec![batch(vec![Some(100); 3000])?]))
                .collect::<Result<Vec<_>>>()?;
            partitions[2].push(batch(vec![Some(-100), None, Some(overflowing)])?);
            let setup = async |ctx: &SessionContext| {
                let table = datafusion::datasource::MemTable::try_new(
                    Arc::clone(&schema),
                    partitions.clone(),
                )?;
                ctx.register_table("f", Arc::new(table))?;
                Ok(())
            };
            let datafusion = datafusion_context(test_session_config());
            setup(&datafusion).await?;
            let message = crate::test_utils::run_sql(&datafusion, sql)
                .await
                .expect_err("DataFusion fails")
                .to_string();
            assert!(message.contains(error), "{message}");
            assert_fails_like_datafusion(sql, test_session_config(), &setup, &message)
                .await;
            let root = lowered(sql, test_session_config(), &setup).await;
            assert_native(sql, &root);
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // EXPLAIN
    // ---------------------------------------------------------------------

    /// The output of `sql`, an `EXPLAIN`, with the pipeline engine: the plan of
    /// every plan type.
    async fn explain(sql: &str) -> Vec<(String, String)> {
        let ctx = pipeline_context(test_session_config());
        register_view_tables(&ctx).unwrap();
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        batches
            .iter()
            .flat_map(|batch| {
                let types = batch.column(0).as_string::<i32>();
                let plans = batch.column(1).as_string::<i32>();
                types
                    .iter()
                    .zip(plans.iter())
                    .map(|(plan_type, plan)| {
                        (plan_type.unwrap().to_owned(), plan.unwrap().to_owned())
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The plan of type `plan_type` of an `EXPLAIN` output.
    fn plan_of<'a>(explained: &'a [(String, String)], plan_type: &str) -> &'a str {
        explained
            .iter()
            .find(|(t, _)| t == plan_type)
            .map(|(_, plan)| plan.as_str())
            .unwrap_or_else(|| panic!("no {plan_type} in {explained:#?}"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn explain_shows_the_pipelines() {
        let sql = "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d";
        let explained = explain(&format!("EXPLAIN {sql}")).await;
        assert_eq!(explained.len(), 2, "{explained:#?}");
        assert!(plan_of(&explained, "logical_plan").contains("Aggregate:"));
        let plan = plan_of(&explained, "physical_plan");
        assert_eq!(
            plan,
            "PipelineRootExec: pipelines=2\n  \
             PhysicalProjection: expr=[d@0 as d, count(Int64(1))@1 as n], pipelines=[0]\n    \
             PhysicalHashAggregate: gby=[d@0 as d], aggr=[count(Int64(1))], pipelines=[0, 1]\n      \
             MemoryScanSource: partitions=4, rows=5000, chunks=6, pipelines=[1]\n  \
             Pipeline 1: MemoryScanSource -> PhysicalHashAggregate\n  \
             Pipeline 0: PhysicalHashAggregate -> PhysicalProjection -> CollectorSink, \
             after pipelines [1]\n"
        );

        // A native window partitions and sorts its input itself, in pipelines
        // of its own: its input sinks into it, and the threads of its
        // `WindowPartitionSort` pipeline sort its partitions in parallel.
        let explained = explain(
            "EXPLAIN SELECT id, ROW_NUMBER() OVER (PARTITION BY a ORDER BY id) AS rn FROM t1",
        )
        .await;
        let plan = plan_of(&explained, "physical_plan");
        assert!(
            plan.starts_with("PipelineRootExec: pipelines=3\n"),
            "{plan}"
        );
        assert!(
            plan.contains(
                "\n    PhysicalWindow: wdw=[row_number() PARTITION BY [t1.a] ORDER BY \
                 [t1.id ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW], \
                 sort=[a@1 ASC NULLS LAST, id@0 ASC NULLS LAST], pipelines=[0, 1, 2]\n"
            ),
            "{plan}"
        );
        assert!(
            plan.ends_with(
                "  Pipeline 2: WindowPartitionSort -> PhysicalWindow, after pipelines [1]\n  \
                 Pipeline 1: MemoryScanSource -> PhysicalWindow\n  \
                 Pipeline 0: PhysicalWindow -> PhysicalProjection -> CollectorSink, \
                 after pipelines [1, 2]\n"
            ),
            "{plan}"
        );

        // A bridged plan is shown below its bridge: a window with an average of
        // decimals, which the native window leaves to DataFusion (it may
        // overflow on some rows).
        let explained = explain(
            "EXPLAIN SELECT id, AVG(g) OVER (PARTITION BY d ORDER BY id) AS m FROM t1",
        )
        .await;
        let plan = plan_of(&explained, "physical_plan");
        assert!(plan.starts_with("PipelineRootExec: pipelines=1, bridged_plans=1\n"));
        assert!(
            plan.contains(
                "\n    BridgeOperator, pipelines=[0]\n      BoundedWindowAggExec: "
            ),
            "{plan}"
        );
        assert!(
            plan.contains("\n          RepartitionExec: partitioning=Hash"),
            "{plan}"
        );

        // `EXPLAIN VERBOSE` also shows the plans of DataFusion that the plan
        // of the pipeline engine was lowered from.
        let explained = explain(&format!("EXPLAIN VERBOSE {sql}")).await;
        assert!(
            explained.iter().any(|(plan_type, plan)| plan_type
                .starts_with("physical_plan after")
                && plan.contains("RepartitionExec")),
            "{explained:#?}"
        );
        for plan_type in [
            "physical_plan",
            "physical_plan_with_stats",
            "physical_plan_with_schema",
        ] {
            let plan = plan_of(&explained, plan_type);
            assert!(
                plan.starts_with("PipelineRootExec: pipelines=2"),
                "{plan_type}: {plan}"
            );
            assert!(!plan.contains("RepartitionExec"), "{plan_type}: {plan}");
        }
        assert!(
            plan_of(&explained, "physical_plan_with_schema")
                .contains("schema=[d:Utf8View;N")
        );

        // `EXPLAIN FORMAT TREE`.
        let explained = explain(&format!("EXPLAIN FORMAT TREE {sql}")).await;
        let plan = plan_of(&explained, "physical_plan");
        assert!(plan.contains("PipelineRootExec"), "{plan}");
        assert!(plan.contains("pipelines: 2"), "{plan}");
        assert!(!plan.contains("RepartitionExec"), "{plan}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn explain_analyze_selects_the_metrics_of_every_operator() {
        // A native sort of a bridged plan (a window with an average of
        // decimals, which the native window leaves to DataFusion), whose nodes
        // have metrics too.
        let query = "SELECT e, rn FROM (SELECT e, \
                     AVG(g) OVER (PARTITION BY e ORDER BY id) AS rn FROM t1) \
                     ORDER BY rn DESC, e";
        let analyzed = async |options: &str| {
            let explained = explain(&format!("EXPLAIN ({options}) {query}")).await;
            let plan = plan_of(&explained, "Plan with Metrics").to_string();
            assert!(plan.contains("PhysicalSort"), "{plan}");
            assert!(plan.contains("BoundedWindowAggExec"), "{plan}");
            plan
        };
        // Without metrics, of the native operators and of the bridged nodes.
        let plan = analyzed("ANALYZE, METRICS 'none'").await;
        assert!(plan.matches("metrics=[]").count() > 4, "{plan}");
        assert_eq!(
            plan.matches("metrics=[").count(),
            plan.matches("metrics=[]").count(),
            "{plan}"
        );
        // Without the timing metrics.
        let plan = analyzed("ANALYZE, METRICS 'rows', TIMING OFF").await;
        assert!(plan.contains("output_rows="), "{plan}");
        assert!(plan.contains("sink_rows="), "{plan}");
        assert!(!plan.contains("elapsed_compute"), "{plan}");
        // Without the development metrics.
        let plan = analyzed("ANALYZE, LEVEL summary").await;
        assert!(plan.contains("output_rows="), "{plan}");
        assert!(plan.contains("elapsed_compute="), "{plan}");
        for metric in ["output_chunks", "sink_rows", "output_batches"] {
            assert!(!plan.contains(metric), "{metric}: {plan}");
        }
        // All of them, by default.
        let plan = analyzed("ANALYZE").await;
        for metric in [
            "output_rows",
            "output_chunks",
            "sink_rows",
            "elapsed_compute",
        ] {
            assert!(plan.contains(metric), "{metric}: {plan}");
        }
        // In both renderings of `EXPLAIN ANALYZE VERBOSE`, for the native
        // operators and the nodes of the bridged plan.
        let sql =
            format!("EXPLAIN (ANALYZE, VERBOSE, METRICS 'rows', TIMING OFF) {query}");
        let explained = explain(&sql).await;
        for rendering in ["Plan with Metrics", "Plan with Full Metrics"] {
            let plan = plan_of(&explained, rendering);
            for node in ["PhysicalSort", "BoundedWindowAggExec"] {
                let line = plan
                    .lines()
                    .find(|line| line.trim_start().starts_with(node))
                    .unwrap_or_else(|| panic!("no {node}: {plan}"));
                assert!(line.contains("output_rows="), "{rendering}: {line}");
                assert!(!line.contains("elapsed_compute"), "{rendering}: {line}");
            }
            assert!(!plan.contains("elapsed_compute"), "{rendering}: {plan}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn explain_analyze_shows_the_metrics_of_every_operator() {
        let explained = explain(
            "EXPLAIN ANALYZE SELECT e, COUNT(*) AS n FROM t1 WHERE a > 0 GROUP BY e",
        )
        .await;
        let plan = plan_of(&explained, "Plan with Metrics");
        let lines = plan.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 8, "{plan}");
        assert_eq!(lines[0], "PipelineRootExec: pipelines=2", "{plan}");
        let operators = [
            "  PhysicalProjection: expr=[e@0 as e, count(Int64(1))@1 as n], pipelines=[0], \
             metrics=[output_rows=3, output_chunks=",
            "    PhysicalHashAggregate: gby=[e@0 as e], aggr=[count(Int64(1))], \
             pipelines=[0, 1], metrics=[output_rows=3, output_chunks=",
            "      CachingOperator: threshold=256, pipelines=[1], metrics=[output_rows=",
            "        PhysicalFilter: a@0 > 0, projection=[e@1], pipelines=[1], \
             metrics=[output_rows=",
            "          MemoryScanSource: partitions=4, rows=5000, chunks=6, pipelines=[1], \
             metrics=[output_rows=5.00 K, output_chunks=6, elapsed_compute=",
        ];
        for (line, operator) in lines[1..].iter().zip(operators) {
            assert!(line.starts_with(operator), "{line}\n{plan}");
        }
        // The aggregation consumed the rows the filter produced, as a sink.
        let filtered = lines[3]
            .split("output_rows=")
            .nth(1)
            .and_then(|metrics| metrics.split(',').next())
            .unwrap();
        assert!(
            lines[2].contains(&format!("sink_rows={filtered}, ")),
            "{plan}"
        );
        assert_eq!(
            lines[6],
            "  Pipeline 1: MemoryScanSource -> PhysicalFilter -> CachingOperator -> \
             PhysicalHashAggregate",
            "{plan}"
        );
        // DataFusion appends the metrics of the `PipelineRootExec` itself.
        assert!(
            lines[7].starts_with(
                "  Pipeline 0: PhysicalHashAggregate -> PhysicalProjection -> CollectorSink, \
                 after pipelines [1], metrics=[output_rows=3, elapsed_compute="
            ),
            "{plan}"
        );
    }
}

#[cfg(test)]
mod tpch {
    //! The 22 TPC-H queries (of `benchmarks/queries`), on small generated
    //! tables: complex plans of many joins, aggregations, sorts and
    //! subqueries.

    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::compute::concat_batches;
    use arrow::datatypes::SchemaRef;
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;
    use datafusion_common::Result;

    use crate::bridge::{PipelineRootExec, bridged_plans};
    use crate::explain::format_operator_tree;
    use crate::test_utils::{
        compare_outputs, datafusion_context, run_engines, test_session_config,
    };

    /// The SQL that generates every TPC-H table. Every row is a function of
    /// its keys, with the values that the predicates of the queries select.
    const TABLES: &[(&str, &str)] = &[
        (
            "region",
            "SELECT * FROM (VALUES (0, 'AFRICA', 'lar deposits'), \
             (1, 'AMERICA', 'hs use ironic'), (2, 'ASIA', 'ges. thinly even'), \
             (3, 'EUROPE', 'ly final courts'), (4, 'MIDDLE EAST', 'uickly special')) \
             AS r(r_regionkey, r_name, r_comment)",
        ),
        (
            "nation",
            "SELECT * FROM (VALUES (0, 'ALGERIA', 0, 'c0'), (1, 'ARGENTINA', 1, 'c1'), \
             (2, 'BRAZIL', 1, 'c2'), (3, 'CANADA', 1, 'c3'), (4, 'EGYPT', 4, 'c4'), \
             (5, 'ETHIOPIA', 0, 'c5'), (6, 'FRANCE', 3, 'c6'), (7, 'GERMANY', 3, 'c7'), \
             (8, 'INDIA', 2, 'c8'), (9, 'INDONESIA', 2, 'c9'), (10, 'IRAN', 4, 'c10'), \
             (11, 'IRAQ', 4, 'c11'), (12, 'JAPAN', 2, 'c12'), (13, 'JORDAN', 4, 'c13'), \
             (14, 'KENYA', 0, 'c14'), (15, 'MOROCCO', 0, 'c15'), \
             (16, 'MOZAMBIQUE', 0, 'c16'), (17, 'PERU', 1, 'c17'), (18, 'CHINA', 2, 'c18'), \
             (19, 'ROMANIA', 3, 'c19'), (20, 'SAUDI ARABIA', 4, 'c20'), \
             (21, 'VIETNAM', 2, 'c21'), (22, 'RUSSIA', 3, 'c22'), \
             (23, 'UNITED KINGDOM', 3, 'c23'), (24, 'UNITED STATES', 1, 'c24')) \
             AS n(n_nationkey, n_name, n_regionkey, n_comment)",
        ),
        (
            "supplier",
            "SELECT value AS s_suppkey, \
             'Supplier#' || lpad(CAST(value AS VARCHAR), 9, '0') AS s_name, \
             'address ' || CAST(value * 31 % 97 AS VARCHAR) AS s_address, \
             nationkey AS s_nationkey, \
             CAST(10 + nationkey AS VARCHAR) || '-' \
               || lpad(CAST(value * 13 % 1000 AS VARCHAR), 3, '0') || '-' \
               || lpad(CAST(value * 17 % 10000 AS VARCHAR), 4, '0') AS s_phone, \
             CAST(value * 7919 % 1099999 / 100.0 - 999.99 AS DECIMAL(15, 2)) AS s_acctbal, \
             CASE WHEN value % 23 = 0 THEN 'blithely Customer quickly Complaints' \
               ELSE 'final requests ' || CAST(value % 7 AS VARCHAR) END AS s_comment \
             FROM (SELECT value, (value * 7 + value / 25) % 25 AS nationkey \
               FROM generate_series(1, 100))",
        ),
        (
            "part",
            "SELECT value AS p_partkey, \
             CASE value % 5 WHEN 0 THEN 'forest green ' WHEN 1 THEN 'blush green ' \
               WHEN 2 THEN 'forest ivory ' WHEN 3 THEN 'navy chiffon ' \
               ELSE 'lavender red ' END || CAST(value AS VARCHAR) AS p_name, \
             'Manufacturer#' || CAST(1 + value % 5 AS VARCHAR) AS p_mfgr, \
             'Brand#' || CAST(1 + value % 5 AS VARCHAR) \
               || CAST(1 + value / 5 % 5 AS VARCHAR) AS p_brand, \
             CASE value % 6 WHEN 0 THEN 'STANDARD POLISHED BRASS' \
               WHEN 1 THEN 'PROMO BURNISHED COPPER' WHEN 2 THEN 'MEDIUM POLISHED TIN' \
               WHEN 3 THEN 'ECONOMY ANODIZED STEEL' WHEN 4 THEN 'PROMO PLATED BRASS' \
               ELSE 'LARGE BRUSHED NICKEL' END AS p_type, \
             CAST(1 + value * 7 % 50 AS INT) AS p_size, \
             CASE value % 8 WHEN 0 THEN 'SM CASE' WHEN 1 THEN 'SM BOX' \
               WHEN 2 THEN 'MED BAG' WHEN 3 THEN 'MED BOX' WHEN 4 THEN 'LG CASE' \
               WHEN 5 THEN 'LG BOX' WHEN 6 THEN 'SM PACK' ELSE 'LG PKG' END AS p_container, \
             CAST(900 + value % 200 + value / 1000.0 AS DECIMAL(15, 2)) AS p_retailprice, \
             'part comment ' || CAST(value % 11 AS VARCHAR) AS p_comment \
             FROM generate_series(1, 200)",
        ),
        (
            "partsupp",
            "SELECT p.value AS ps_partkey, (p.value + s.value * 25) % 100 + 1 AS ps_suppkey, \
             CAST(1 + (p.value * 31 + s.value * 17) % 9999 AS INT) AS ps_availqty, \
             CAST((1 + (p.value * 97 + s.value * 13) % 100000) / 100.0 AS DECIMAL(15, 2)) \
               AS ps_supplycost, \
             'partsupp comment ' || CAST((p.value + s.value) % 13 AS VARCHAR) AS ps_comment \
             FROM generate_series(1, 200) AS p CROSS JOIN generate_series(0, 3) AS s",
        ),
        (
            "customer",
            "SELECT value AS c_custkey, \
             'Customer#' || lpad(CAST(value AS VARCHAR), 9, '0') AS c_name, \
             'address ' || CAST(value * 17 % 101 AS VARCHAR) AS c_address, \
             value * 11 % 25 AS c_nationkey, \
             CAST(10 + value * 11 % 25 AS VARCHAR) || '-' \
               || lpad(CAST(value * 13 % 1000 AS VARCHAR), 3, '0') || '-' \
               || lpad(CAST(value * 7 % 10000 AS VARCHAR), 4, '0') AS c_phone, \
             CAST(value * 7907 % 1100000 / 100.0 - 999.99 AS DECIMAL(15, 2)) AS c_acctbal, \
             CASE value % 5 WHEN 0 THEN 'BUILDING' WHEN 1 THEN 'AUTOMOBILE' \
               WHEN 2 THEN 'MACHINERY' WHEN 3 THEN 'HOUSEHOLD' \
               ELSE 'FURNITURE' END AS c_mktsegment, \
             'customer comment ' || CAST(value % 7 AS VARCHAR) AS c_comment \
             FROM generate_series(1, 300)",
        ),
        (
            "orders",
            "SELECT value AS o_orderkey, \
             CASE WHEN (1 + value * 37 % 299) % 3 = 0 THEN 2 + value * 37 % 299 \
               ELSE 1 + value * 37 % 299 END AS o_custkey, \
             CASE value % 3 WHEN 0 THEN 'F' WHEN 1 THEN 'O' ELSE 'P' END AS o_orderstatus, \
             CAST(1000 + value * 7919 % 400000 + value / 100.0 AS DECIMAL(15, 2)) \
               AS o_totalprice, \
             arrow_cast(CAST(8035 + value * 13 % 2405 AS INT), 'Date32') AS o_orderdate, \
             CASE value % 5 WHEN 0 THEN '1-URGENT' WHEN 1 THEN '2-HIGH' \
               WHEN 2 THEN '3-MEDIUM' WHEN 3 THEN '4-NOT SPECIFIED' \
               ELSE '5-LOW' END AS o_orderpriority, \
             'Clerk#' || lpad(CAST(1 + value % 50 AS VARCHAR), 9, '0') AS o_clerk, \
             CAST(0 AS INT) AS o_shippriority, \
             CASE WHEN value % 13 = 0 THEN 'carefully special packages requests' \
               ELSE 'final deposits ' || CAST(value % 9 AS VARCHAR) END AS o_comment \
             FROM generate_series(1, 3000)",
        ),
        (
            "lineitem",
            "WITH lines AS (SELECT o.value AS o, l.value AS l, \
               1 + (o.value * 7 + l.value * 13) % 200 AS partkey, \
               8035 + o.value * 13 % 2405 AS orderdate \
               FROM generate_series(1, 3000) AS o CROSS JOIN generate_series(1, 7) AS l \
               WHERE l.value <= CASE WHEN o.value % 101 = 4 THEN 7 \
                 ELSE 1 + o.value * 5 % 7 END), \
             shipped AS (SELECT *, orderdate + 1 + (o + l * 7) % 121 AS shipdate, \
               CASE WHEN o % 101 = 4 THEN 49 + l % 2 ELSE 1 + (o * 3 + l * 7) % 50 END \
                 AS quantity FROM lines) \
             SELECT o AS l_orderkey, partkey AS l_partkey, \
             (partkey + (o + l) % 4 * 25) % 100 + 1 AS l_suppkey, \
             CAST(l AS INT) AS l_linenumber, \
             CAST(quantity AS DECIMAL(15, 2)) AS l_quantity, \
             CAST(quantity * (900 + partkey % 200 + partkey / 1000.0) AS DECIMAL(15, 2)) \
               AS l_extendedprice, \
             CAST((o + l) % 11 / 100.0 AS DECIMAL(15, 2)) AS l_discount, \
             CAST((o * 3 + l) % 9 / 100.0 AS DECIMAL(15, 2)) AS l_tax, \
             CASE WHEN shipdate + 1 + (o + l) % 30 <= 9298 \
               THEN CASE WHEN (o + l) % 2 = 0 THEN 'R' ELSE 'A' END \
               ELSE 'N' END AS l_returnflag, \
             CASE WHEN shipdate > 9298 THEN 'O' ELSE 'F' END AS l_linestatus, \
             arrow_cast(CAST(shipdate AS INT), 'Date32') AS l_shipdate, \
             arrow_cast(CAST(orderdate + 30 + (o * 3 + l) % 61 AS INT), 'Date32') \
               AS l_commitdate, \
             arrow_cast(CAST(shipdate + 1 + (o + l) % 30 AS INT), 'Date32') AS l_receiptdate, \
             CASE (o + l) % 4 WHEN 0 THEN 'DELIVER IN PERSON' WHEN 1 THEN 'COLLECT COD' \
               WHEN 2 THEN 'NONE' ELSE 'TAKE BACK RETURN' END AS l_shipinstruct, \
             CASE (o * 3 + l) % 7 WHEN 0 THEN 'REG AIR' WHEN 1 THEN 'AIR' \
               WHEN 2 THEN 'RAIL' WHEN 3 THEN 'SHIP' WHEN 4 THEN 'TRUCK' \
               WHEN 5 THEN 'MAIL' ELSE 'FOB' END AS l_shipmode, \
             'line comment ' || CAST((o + l) % 17 AS VARCHAR) AS l_comment \
             FROM shipped",
        ),
    ];

    /// The TPC-H tables, generated by DataFusion, in 3 partitions of
    /// different sizes.
    async fn tables() -> Result<Vec<(&'static str, SchemaRef, Vec<Vec<RecordBatch>>)>> {
        let ctx = datafusion_context(test_session_config());
        let mut tables = vec![];
        for &(name, sql) in TABLES {
            let df = ctx.sql(sql).await?;
            let schema = Arc::clone(df.schema().inner());
            let batch = concat_batches(&schema, &df.collect().await?)?;
            let rows = batch.num_rows();
            let partitions = vec![
                vec![batch.slice(0, rows / 2)],
                vec![batch.slice(rows / 2, rows / 8)],
                vec![batch.slice(rows / 2 + rows / 8, rows - rows / 2 - rows / 8)],
            ];
            tables.push((name, schema, partitions));
        }
        Ok(tables)
    }

    /// The query `q{number}.sql` of `benchmarks/queries`, and the statements
    /// to execute before it (the view of query 15).
    fn query(number: usize) -> (String, Vec<String>) {
        let path = format!(
            "{}/../../benchmarks/queries/q{number}.sql",
            env!("CARGO_MANIFEST_DIR")
        );
        let sql =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut statements = sql
            .split(';')
            .map(str::trim)
            .filter(|statement| {
                !statement.is_empty() && !statement.to_lowercase().starts_with("drop")
            })
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let query = statements.pop().unwrap();
        (query, statements)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tpch_queries_match_datafusion() -> Result<()> {
        let tables = tables().await?;
        let mut native = vec![];
        for number in 1..=22 {
            let (sql, statements) = query(number);
            let setup = async |ctx: &SessionContext| {
                for (name, schema, partitions) in &tables {
                    let table =
                        MemTable::try_new(Arc::clone(schema), partitions.clone())?;
                    ctx.register_table(*name, Arc::new(table))?;
                }
                for statement in &statements {
                    ctx.sql(statement).await?.collect().await?;
                }
                Ok(())
            };
            let (expected, actual) = run_engines(&sql, test_session_config(), setup)
                .await
                .unwrap_or_else(|e| panic!("Q{number} failed: {e}"));
            let root = actual
                .plan
                .downcast_ref::<PipelineRootExec>()
                .expect("a pipeline plan")
                .operator();
            if let Err(message) = compare_outputs(&expected, &actual) {
                panic!(
                    "Q{number}: results differ: {message}\nlowered: {}",
                    format_operator_tree(root)
                );
            }
            let rows = actual
                .batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>();
            assert!(rows > 0, "Q{number} selects no rows");
            if bridged_plans(root).is_empty() {
                native.push(number);
            }
        }
        // The other queries leave parts to DataFusion that the native
        // operators do not reproduce exactly: e.g. scalar subqueries, TopKs
        // of strings, or decimal arithmetic (which may fail) in projections
        // and aggregations, which the native operators would evaluate in
        // parallel on several threads (e.g. the `SUM` of Q6).
        assert!(native.contains(&13), "Q13 is not native: {native:?}");
        Ok(())
    }
}
