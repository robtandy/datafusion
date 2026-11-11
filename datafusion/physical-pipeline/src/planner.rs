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
//! chain of nodes, that a native operator implements by that operator, whose
//! inputs are lowered in turn:
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
//! The filters of the pipelines that may stop before their inputs end (below
//! limits, filters with a fetch and aggregations with a soft limit, and in
//! the subqueries of scalar subqueries) are not cached: see
//! `Lowering::lower_in`.
//!
//! The dynamic filter of every native hash join is routed, like DuckDB's join
//! filters, to the scans below its probe side that produce its probe keys
//! unchanged: through projections, filters, sorts, unions, other joins and
//! aggregations grouped by the keys, but never into the null side of an outer
//! join, nor across a node that changes the values of the keys or limits its
//! rows. The scans skip the rows that no build key matches, which the join
//! would drop anyway, also several joins below it: see
//! `Lowering::try_lower_hash_join`.
//!
//! Windows, nested loop joins and cross joins keep all of their rows in
//! memory without reserving it in the memory pool, while the DataFusion nodes
//! they replace reserve their memory (and fail or spill once the pool is
//! exhausted): they are only lowered for sessions whose memory pool has no
//! limit, see [`lower_execution_plan_with_runtime`].
//!
//! Every native operator documents (in its `try_lower` or `supports_*`
//! function) which nodes it lowers; any other node, such as a sort-merge join,
//! a recursive query or a custom node, is executed by DataFusion: it is
//! wrapped, with its whole subtree, in a [`BridgeOperator`], so that every
//! plan can be executed. Native operators above a bridged plan read its output
//! as the source of their pipeline.
//!
//! # Results and errors
//!
//! The pipeline engine returns correct results, with exactly the output schema
//! of the plan that it lowers, but not necessarily the bytes or the errors of
//! DataFusion (see `docs/compatibility.md`): which of several errors is
//! raised, whether an error is raised for rows that the engine never
//! evaluates (e.g. the rows beyond a limit, or the rows that a dynamic filter
//! prunes), the order of rows that SQL does not define, and anything that
//! depends on batches, partitioning or parallelism are the engine's own. So
//! the lowering does not judge how the expressions of a node may fail, nor in
//! which order, on which threads or on which batches the rows are evaluated:
//! it lowers every node that a native operator implements, wherever it is in
//! the plan. In particular,
//!
//! * the plans below limits are lowered like any other plan: a native limit
//!   stops the pipelines below it once it has its rows;
//! * filters, projections, aggregations, sorts, joins and windows are
//!   lowered whatever their expressions may fail with, e.g. the decimal
//!   division of TPC-H Q8 (by zero, or with an overflow), user-defined
//!   functions, and expressions whose errors in DataFusion depend on its
//!   batches, on any number of threads, above and below other operators of
//!   their pipelines that may fail too;
//! * TopKs and hash joins whose dynamic filters the scans below them read are
//!   lowered whatever their inputs evaluate: the native TopK updates the
//!   filter of the plan with its threshold, so that the scans skip the files,
//!   row groups and rows that it rules out (see `Lowering::try_lower_sort`),
//!   as a native hash join sets its filter to the bounds of its build keys;
//! * the nodes that only distribute or coalesce rows are removed whatever the
//!   types of their rows (e.g. strings with 32-bit offsets, whose
//!   concatenation may overflow in DataFusion).
//!
//! A node with a fetch of 0 (e.g. `LIMIT 0`) is lowered to a source of no
//! rows, without its input.
//!
//! DataFusion still executes
//!
//! * the nodes that no native operator implements yet (e.g. sort-merge and
//!   symmetric hash joins, recursive queries, and user-defined
//!   `ExecutionPlan`s), and the nodes that the native operators refuse
//!   themselves (see their `try_lower` functions);
//! * the windows and loop joins of sessions whose memory pool has a limit
//!   (see above);
//! * the sorts, windows, aggregations and joins of values that address more
//!   than their memory, e.g. lists of `1 << 30` nulls (see
//!   `gathers_values_beyond_their_memory`), which the native operators would
//!   materialize as they gather their rows, before the memory pool could
//!   refuse them;
//! * the windows above TopKs of unbounded inputs that DataFusion may end early
//!   (see `ends_unbounded_input_early_below`), which a native window, which
//!   reads all of its input first, would wait for forever;
//! * the merges of the sorted partitions of windows by keys of unions (also
//!   inside other values), which the native sort orders otherwise, as the
//!   sorts by such keys (see `Lowering::try_lower_merge_of_window`).
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

use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering as AtomicOrdering};

use arrow::array::Array as _;
use arrow::compute::{and, prep_null_mask_filter};
use arrow::datatypes::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::format::ExplainFormat;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    JoinSide, JoinType, Result, ScalarValue, exec_err, internal_err,
};
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_csv::source::CsvSource;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::LogicalPlan;
use datafusion_expr::logical_plan::Explain;
use datafusion_expr::physical_planning_context::{ScalarSubqueryResults, SubqueryIndex};
use datafusion_physical_expr::expressions::{Column, DynamicFilterPhysicalExpr};
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::analyze::AnalyzeExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::explain::ExplainExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::utils::{ColumnIndex, build_join_schema};
use datafusion_physical_plan::joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::scalar_subquery::ScalarSubqueryExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::union::{InterleaveExec, UnionExec};
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    ReplaceChildrenOptions,
};
use datafusion_session::{PhysicalPlanner, QueryPlanner, Session};

use crate::aggregate::PhysicalHashAggregate;
use crate::bridge::{BridgeOperator, PipelineRootExec};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk, SelectionVector};
use crate::explain::{AnalyzedPipelineExec, MetricSelection};
use crate::join::PhysicalHashJoin;
use crate::join::hash_join::JoinFilterTarget;
use crate::join::nested_loop::{PhysicalCrossJoin, PhysicalNestedLoopJoin};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorResult, OperatorState, PhysicalOperator,
    SinkFinalize, SinkResult, SourceResult, downcast_state,
};
use crate::operators::caching::CachingOperator;
use crate::operators::filter::PhysicalFilter;
use crate::operators::limit::PhysicalLimit;
use crate::operators::placeholder_row::PhysicalPlaceholderRow;
use crate::operators::projection::PhysicalProjection;
use crate::operators::union::PhysicalUnion;
use crate::pipeline::{MetaPipelineBuilder, MetaPipelineId, PipelineBuilder};
use crate::sort::{PhysicalSort, PhysicalTopK, TopKDynamicFilter};
use crate::sources::{EmptyScanSource, try_create_scan_source};
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
    let root = lowering.lower(&plan)?;
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
/// Every node (or chain of nodes) that a native operator implements is
/// replaced by that operator, and every other node is executed by a
/// [`BridgeOperator`], together with its whole subtree. See the
/// [module documentation](self) for the native operators and the plans that
/// are left to DataFusion.
///
/// A node whose native operator fails to be created (the native operators
/// report the nodes they do not implement without an error, so this is not
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
    Lowering::new(options, None).lower(plan)
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
    Lowering::new(options, Some(runtime)).lower(plan)
}

/// Options for which the lowering lowers what it lowers for every session:
/// DataFusion's sort concatenates the batches of a partition, which can
/// overflow 32-bit offsets, while they occupy fewer bytes than
/// `sort_in_place_threshold_bytes` (see [`crate::sort::try_lower`]).
fn any_session_options() -> ConfigOptions {
    let mut options = ConfigOptions::default();
    options.execution.sort_in_place_threshold_bytes = usize::MAX;
    options
}

/// Lowers the nodes of a plan, see [`lower_execution_plan_with_options`] and
/// [`lower_execution_plan_with_runtime`].
struct Lowering<'a> {
    options: &'a ConfigOptions,
    /// The runtime of the session that executes the plan, if known: windows
    /// and loop joins are only lowered for runtimes whose memory pool has no
    /// limit.
    runtime: Option<&'a RuntimeEnv>,
    /// Whether the node being lowered is in a pipeline that may stop before
    /// its input ends (see [`Self::lower_in`]): below a limit, a filter with a
    /// fetch or an aggregation with a soft limit, or in the subquery of a
    /// scalar subquery, which fails at its second row.
    early_stop: Cell<bool>,
    /// The dynamic filters of the hash joins whose probe sides are being
    /// lowered, routed to the scans below them (see
    /// [`Self::try_lower_hash_join`]).
    join_filters: RefCell<Vec<ScanJoinFilter>>,
}

impl<'a> Lowering<'a> {
    fn new(options: &'a ConfigOptions, runtime: Option<&'a RuntimeEnv>) -> Self {
        Self {
            options,
            runtime,
            early_stop: Cell::new(false),
            join_filters: RefCell::new(vec![]),
        }
    }
}

impl Lowering<'_> {
    /// Lower `plan`, falling back to DataFusion if a native operator cannot
    /// be created (see [`lower_execution_plan_with_options`]). Never fails;
    /// returns a `Result` for the `try_lower` functions of the operators.
    fn lower(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>> {
        self.lower_natively(plan).or_else(|error| {
            log::warn!(
                "Executing {} with DataFusion, its native lowering failed: {error}",
                plan.name()
            );
            Ok(bridge(plan))
        })
    }

    /// Lower `plan` (see [`Self::lower`]) in a pipeline that may stop before
    /// its input ends if `early_stop` (e.g. below a limit), or that runs until
    /// its input ends otherwise (e.g. below a pipeline breaker, such as a
    /// sort, an aggregation or the build side of a join, which reads all of
    /// its input).
    ///
    /// The filters of a pipeline that may stop early are not cached (see
    /// [`CachingOperator`]): a cache holds back the rows of small chunks until
    /// it is full or its input ends, which may never happen, e.g. if a limit
    /// waits for these rows above a stream that stays pending once it has
    /// produced them (DuckDB does not cache in such pipelines either). The
    /// nodes that pass the rows of their inputs on in their pipelines (filters,
    /// projections, unions, limits, the probe sides of joins and the input of
    /// a scalar subquery) keep `early_stop` for their inputs, and the limits,
    /// the filters with a fetch and the aggregations with a soft limit, which
    /// stop their pipelines once they have their rows, set it.
    fn lower_in(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        early_stop: bool,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let outer = self.early_stop.replace(early_stop);
        let lowered = self.lower(plan);
        self.early_stop.set(outer);
        lowered
    }

    /// Lower `input`, an input of the join `join`, in the pipeline of the
    /// join if it is its probe side (its second input, see [`is_within`]),
    /// and in a pipeline that runs until its input ends if it is its build
    /// side (see [`Self::lower_in`]).
    fn lower_join_input(
        &self,
        join: &Arc<dyn ExecutionPlan>,
        input: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let probe =
            matches!(join.children().as_slice(), [_, probe] if is_within(probe, input));
        self.lower_in(input, probe && self.early_stop.get())
    }

    fn lower_natively(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        // A node with a fetch of 0 (e.g. `LIMIT 0`) produces no rows: its
        // input is not executed.
        if plan.fetch() == Some(0) {
            return Ok(no_rows(plan));
        }
        if !gathers_values_beyond_their_memory(plan) {
            if let Some(sort) = self.try_lower_sort(plan)? {
                return Ok(sort);
            }
            if let Some(window) = self.try_lower_window(plan)? {
                return Ok(window);
            }
            if let Some(sort) = self.try_lower_merge_of_window(plan)? {
                return Ok(sort);
            }
            if let Some(aggregate) = self.try_lower_aggregate(plan)? {
                return Ok(aggregate);
            }
            if let Some(join) = self.try_lower_hash_join(plan)? {
                return Ok(join);
            }
            if let Some(join) = self.try_lower_loop_join(plan)? {
                return Ok(join);
            }
        }
        if let Some(subquery) = self.try_lower_scalar_subquery(plan)? {
            return Ok(subquery);
        }
        // A limit with a fetch stops its pipeline once it has its rows.
        let early_stop = self.early_stop.get() || plan.fetch().is_some();
        if let Some(limit) =
            PhysicalLimit::try_lower(plan, |input| self.lower_in(input, early_stop))?
        {
            return Ok(Arc::new(limit));
        }
        if let Some(union) = PhysicalUnion::try_lower(plan, |input| self.lower(input))? {
            return Ok(Arc::new(union));
        }
        if let Some(placeholder) = PhysicalPlaceholderRow::try_lower(plan)? {
            return Ok(Arc::new(placeholder));
        }
        if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            return self.lower_filter(plan, filter);
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            return self.lower_projection(plan, projection);
        }
        if let Some(scan) = self.lower_scan(plan)? {
            return Ok(scan);
        }
        if let Some(input) = redistributed_input(plan) {
            let lowered = self.lower(input)?;
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
    /// [`PhysicalScalarSubquery`].
    ///
    /// A subquery that produces more than one row fails once its second row
    /// reaches its result, which then stops its pipelines (see
    /// `ScalarSubqueryResult`), as DataFusion stops reading it. A
    /// `ScalarSubqueryExec` with a subquery of other than one column, which
    /// is not a valid plan (DataFusion fails on it), is left to DataFusion.
    fn try_lower_scalar_subquery(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
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
        let input = self.lower(exec.input())?;
        let subqueries = exec
            .subqueries()
            .iter()
            .map(|link| {
                Ok(Arc::new(ScalarSubqueryResult::try_new(
                    self.lower_in(&link.plan, true)?,
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
    /// DataFusion pushes the dynamic filter of a TopK (its threshold) into
    /// the scans of its input (e.g. of Parquet files), which skip the files,
    /// row groups and rows that it rejects as it tightens. A TopK whose filter
    /// a node of its input reads is lowered as a TopK whose filter no node
    /// reads (see [`with_unread_topk_filter`]), so that
    /// [`crate::sort::try_lower`] lowers it whatever the work on the rows that
    /// the filter prunes, and the native TopK then updates the filter of the
    /// plan instead (see [`updating_filter`]): the scans skip the reads that
    /// its threshold rules out, which cannot change its result.
    ///
    /// A sort with a fetch of 0 produces no rows: its input is not executed.
    fn try_lower_sort(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let Some(sort) = sort_of_chain(plan) else {
            return Ok(None);
        };
        if sort.fetch() == Some(0) {
            return Ok(Some(no_rows(plan)));
        }
        let read_filter = read_topk_filter(sort);
        let to_lower = match &read_filter {
            None => Arc::clone(plan),
            Some(_) => with_unread_topk_filter(plan)?,
        };
        let Some(lowered) = crate::sort::try_lower(&to_lower, self.options, |plan| {
            self.lower_in(plan, false)
        })?
        else {
            return Ok(None);
        };
        Ok(Some(match read_filter {
            None => lowered,
            Some(filter) => updating_filter(lowered, filter)?,
        }))
    }

    /// Lower an aggregation with [`PhysicalHashAggregate::try_lower`], after
    /// removing the `CoalesceBatchesExec`s and `PartialReduce` stages between
    /// its final and its partial `AggregateExec` (see
    /// [`aggregation_to_lower`]), and the sort that only orders its input for
    /// its ordered aggregates (see [`without_ordering_sort`]).
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
        let Some(raw) = raw_aggregate(&plan) else {
            return Ok(None);
        };
        // An aggregation with a soft limit on its groups (without an order,
        // e.g. of `SELECT DISTINCT k ... LIMIT 3`) may stop once it has them
        // (see `AggregateSpec::soft_limit`): its input may stop early. Any
        // other aggregation reads all of its input.
        let soft_limit = [plan.downcast_ref::<AggregateExec>(), Some(raw)]
            .into_iter()
            .flatten()
            .filter_map(AggregateExec::limit_options)
            .any(|options| options.descending().is_none());
        let aggregate = PhysicalHashAggregate::try_lower(&plan, |input| {
            self.lower_in(without_ordering_sort(raw, input), soft_limit)
        })?;
        Ok(aggregate.map(|aggregate| Arc::new(aggregate) as Arc<dyn PhysicalOperator>))
    }

    /// Lower a hash join with [`PhysicalHashJoin::try_lower`], which lowers
    /// its inputs without the nodes that only distribute their rows, and sets
    /// the dynamic filters of the join (that the scans of its probe side read)
    /// to the bounds and the membership of its build keys.
    ///
    /// The filter of the join is routed, like DuckDB's join filters, to every
    /// scan below its probe side that produces some of its probe keys
    /// unchanged (see [`route_join_filters`]): through projections, filters,
    /// sorts, unions, both inputs of other joins (and from one key of an inner
    /// or semi join to the other) and aggregations grouped by the keys, but
    /// into no input of a join whose unmatched rows it emits (the null side of
    /// an outer join), and across no node that changes their values or limits
    /// its rows. Every such scan gets a filter of the keys it produces, over
    /// its own columns, which the join sets with its own (see
    /// [`PhysicalHashJoin::with_filter_targets`]), and is lowered with it (see
    /// [`Self::lower_scan`]): the scans then skip the rows that cannot match.
    /// E.g. in TPC-H Q21 over memory tables, the scan of `supplier` on the
    /// build side of its join with `lineitem`, two joins below the join with
    /// the nation of the query, produces the suppliers of that nation only, so
    /// that the joins below do not join the lines of the other suppliers. The
    /// pipelines of these scans wait for the build side of the join (see
    /// [`MetaPipelineBuilder::build_with_dependency`]), so its filters are
    /// complete before they open their files.
    fn try_lower_hash_join(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let routes = route_join_filters(plan)?;
        let probe = plan.children().get(1).copied();
        let lower_input = |input: &Arc<dyn ExecutionPlan>| {
            if routes.is_empty() || !probe.is_some_and(|probe| is_within(probe, input)) {
                return self.lower_join_input(plan, input);
            }
            // The scans of the probe side are lowered with the filters routed to
            // them.
            let outer = self.join_filters.borrow().len();
            self.join_filters
                .borrow_mut()
                .extend(routes.iter().map(|route| route.scan_filter.clone()));
            let lowered = self.lower_join_input(plan, input);
            self.join_filters.borrow_mut().truncate(outer);
            lowered
        };
        let Some(join) = PhysicalHashJoin::try_lower(plan, lower_input)? else {
            return Ok(None);
        };
        // The join sets the filters of the scans that were lowered with them.
        let targets = routes
            .into_iter()
            .filter(|route| route.scan_filter.attached.get())
            .map(|route| route.target)
            .collect();
        Ok(Some(Arc::new(join.with_filter_targets(targets)?)))
    }

    /// Lower a scan to a native source (see [`try_create_scan_source`]), with
    /// the dynamic filters of the hash joins above it that are routed to it
    /// (see [`Self::try_lower_hash_join`]):
    ///
    /// * a Parquet scan reads them in its predicate, like the dynamic filters
    ///   that DataFusion pushes into it: it skips the files, row groups and
    ///   pages (by their statistics, page indexes and bloom filters) that they
    ///   reject, and filters its rows with them with `pushdown_filters`;
    /// * the chunks of a scan of a memory table or of CSV files, which cannot
    ///   filter its rows itself, are filtered with them (see
    ///   [`JoinFilteredScan`]).
    fn lower_scan(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let filters = self
            .join_filters
            .borrow()
            .iter()
            .filter(|filter| Arc::ptr_eq(&filter.scan, plan))
            .cloned()
            .collect::<Vec<_>>();
        let Some(exec) = plan
            .downcast_ref::<DataSourceExec>()
            .filter(|_| !filters.is_empty())
        else {
            return try_create_scan_source(plan);
        };
        if exec.downcast_to_file_source::<ParquetSource>().is_some() {
            let predicates = filters
                .iter()
                .map(|filter| Arc::clone(&filter.filter))
                .collect();
            let pushed = exec
                .data_source()
                .try_pushdown_filters(predicates, self.options)?;
            let Some(source) = pushed.updated_node else {
                return try_create_scan_source(plan);
            };
            let filtered: Arc<dyn ExecutionPlan> =
                Arc::new(exec.clone().with_data_source(source));
            let lowered = try_create_scan_source(&filtered)?;
            if lowered.is_some() {
                for filter in &filters {
                    filter.attached.set(true);
                }
            }
            return Ok(lowered);
        }
        let Some(source) = try_create_scan_source(plan)? else {
            return Ok(None);
        };
        for filter in &filters {
            filter.attached.set(true);
        }
        let filters = filters.into_iter().map(|filter| filter.filter).collect();
        Ok(Some(Arc::new(JoinFilteredScan::new(source, filters))))
    }

    /// Lower a window (a `WindowAggExec` or a `BoundedWindowAggExec`, with the
    /// nodes below it that partition and sort its input, and the windows
    /// stacked below it) with [`PhysicalWindow::try_lower`], for a runtime
    /// whose memory pool has no limit (see
    /// [`lower_execution_plan_with_runtime`]).
    ///
    /// DataFusion plans a projection of columns between the sort below a
    /// window (or the merge of its sorted partitions) and the input of the
    /// sort, e.g. for a window over an aggregation whose columns it renames,
    /// which [`PhysicalWindow::try_lower`] does not look through: such a window
    /// is lowered as if the projection were below the sort, see
    /// [`with_projection_below_sort`].
    ///
    /// The native window reads all of its input before it emits anything: a
    /// window above a TopK of an unbounded input that DataFusion may end early
    /// (see [`ends_unbounded_input_early_below`]), which the native window
    /// would wait for forever, is left to DataFusion.
    fn try_lower_window(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        if !is_window(plan) || ends_unbounded_input_early_below(plan) {
            return Ok(None);
        }
        let Some(runtime) = self.runtime else {
            return Ok(None);
        };
        let rewritten = with_projection_below_sort(plan)?;
        for plan in std::iter::once(plan).chain(rewritten.as_ref()) {
            if let Some(window) = PhysicalWindow::try_lower(plan, runtime, |rows| {
                self.lower_in(rows, false)
            })? {
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
    /// Only over a window that is lowered to a native window, for a runtime
    /// whose memory pool has no limit (see
    /// [`lower_execution_plan_with_runtime`]), and if every partition of the
    /// input of the merge is ordered by its keys (otherwise its output would
    /// not be sorted), and by keys that are not of unions (which the native
    /// sort orders otherwise). A merge of no partitions, on which DataFusion
    /// fails, is not lowered either.
    fn try_lower_merge_of_window(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() else {
            return Ok(None);
        };
        if self.runtime.is_none()
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
        // The native sort orders the rows by their keys in the row format,
        // which orders the values of a union by their type first, also for a
        // descending key, and its nulls among the values of their type, unlike
        // DataFusion's merge (see `crate::sort::try_lower`): a merge by keys
        // of unions (also inside other values) would return other rows.
        let schema = merge.input().schema();
        let keys_of_unions = merge.expr().iter().any(|key| {
            key.expr
                .data_type(&schema)
                .map_or(true, |data_type| contains_union(&data_type))
        });
        if keys_of_unions
            || !merge
                .input()
                .equivalence_properties()
                .ordering_satisfy(merge.expr().clone())?
            || !crate::sort::SortKeys::supports(merge.expr(), &schema)?
        {
            return Ok(None);
        }
        let input = self.lower_in(merge.input(), false)?;
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
    fn try_lower_loop_join(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>> {
        let nested_loop = plan.is::<NestedLoopJoinExec>();
        if !nested_loop && !plan.is::<CrossJoinExec>() {
            return Ok(None);
        }
        let Some(runtime) = self.runtime else {
            return Ok(None);
        };
        let lower_input =
            |input: &Arc<dyn ExecutionPlan>| self.lower_join_input(plan, input);
        let join: Option<Arc<dyn PhysicalOperator>> = if nested_loop {
            PhysicalNestedLoopJoin::try_lower(plan, runtime, lower_input)?
                .map(|join| Arc::new(join) as _)
        } else {
            PhysicalCrossJoin::try_lower(plan, runtime, lower_input)?
                .map(|join| Arc::new(join) as _)
        };
        Ok(join)
    }

    /// Lower a `FilterExec` to a [`PhysicalFilter`] wrapped in a
    /// [`CachingOperator`] (but in a pipeline that may stop early, see
    /// [`Self::lower_in`]), unless the native filter does not implement it
    /// (see [`PhysicalFilter::supports_filter_exec`]).
    ///
    /// A filter with a fetch stops its pipeline once it has produced its
    /// rows, as a limit does: its input is lowered in a pipeline that may stop
    /// early. (Its own cache, if any, is emitted when it stops.)
    fn lower_filter(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        filter: &FilterExec,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if !PhysicalFilter::supports_filter_exec(filter) {
            return Ok(bridge(plan));
        }
        let early_stop = self.early_stop.get() || filter.fetch().is_some();
        let input = self.lower_in(filter.input(), early_stop)?;
        let filter = Arc::new(PhysicalFilter::try_from_filter_exec(filter, input)?);
        if self.early_stop.get() {
            return Ok(filter);
        }
        Ok(Arc::new(CachingOperator::new(filter)))
    }

    /// Lower a `ProjectionExec` to a [`PhysicalProjection`], unless the native
    /// projection does not implement it (see
    /// [`PhysicalProjection::supports_projection_exec`]).
    fn lower_projection(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        projection: &ProjectionExec,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if !PhysicalProjection::supports_projection_exec(projection) {
            return Ok(bridge(plan));
        }
        let input = self.lower(projection.input())?;
        Ok(Arc::new(PhysicalProjection::try_from_projection_exec(
            projection, input,
        )?))
    }
}

/// Whether the native operator of `plan`, if it is a sort, a window, an
/// aggregation or a join (which gather the rows of their inputs), would
/// gather values that address more than their memory (see
/// [`addresses_more_than_their_memory`]): in its output, its inputs, or the
/// input of the `SortExec` of a sort (whose columns a projection above it may
/// drop).
///
/// A few bytes of such values can stand for billions of values (e.g. a list
/// of `1 << 30` nulls), which these operators would materialize as they
/// gather their rows, before the memory pool could refuse them (the native
/// operators do not reserve their memory yet): such nodes are left to
/// DataFusion.
fn gathers_values_beyond_their_memory(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let sorted = sort_of_chain(plan).map(|sort| sort.input().schema());
    std::iter::once(plan.schema())
        .chain(plan.children().into_iter().map(|child| child.schema()))
        .chain(sorted)
        .any(|schema| {
            schema
                .fields()
                .iter()
                .any(|field| addresses_more_than_their_memory(field.data_type()))
        })
}

/// Whether arrays of `data_type` can address more values than their memory
/// holds: run-end encoded arrays with run ends of fewer than 64 bits (which
/// address as many rows as their last run end says), and lists and maps with
/// 32-bit offsets of nulls (which occupy no memory) or of such arrays, at any
/// depth.
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

/// The rows of `plan`, a node with a fetch of 0: none, with its schema.
fn no_rows(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn PhysicalOperator> {
    Arc::new(EmptyScanSource::new(plan.schema()))
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
///   order, of at least one partition;
/// * a `CoalesceBatchesExec` without a fetch.
fn redistributed_input(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    let input = plan.children().into_iter().next()?;
    if input.output_partitioning().partition_count() == 0 {
        return None;
    }
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
        return coalesce.fetch().is_none().then_some(input);
    }
    if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
        return redistributes_rows(repartition).then_some(input);
    }
    if is_coalesce_batches(plan) {
        return plan.fetch().is_none().then_some(input);
    }
    None
}

/// Whether `repartition` only distributes the rows of its input over its
/// partitions: round robin or by hash, without preserving the order.
fn redistributes_rows(repartition: &RepartitionExec) -> bool {
    !repartition.preserve_order()
        && matches!(
            repartition.partitioning(),
            Partitioning::RoundRobinBatch(_) | Partitioning::Hash(..)
        )
}

/// Whether `plan` is a `CoalesceBatchesExec` (which DataFusion does not plan
/// any more, but which other planners may).
#[expect(deprecated)]
fn is_coalesce_batches(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec>()
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

/// Whether `plan` is a window node: a `WindowAggExec` or a
/// `BoundedWindowAggExec`.
fn is_window(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<WindowAggExec>() || plan.is::<BoundedWindowAggExec>()
}

/// Whether a TopK of an unbounded input, which DataFusion may end before its
/// input ends, is among the nodes below `window`, a window node, that the
/// native window reads its input through (see [`PhysicalWindow::try_lower`]):
/// sorts, merges, repartitionings, projections, filters, unions and windows.
///
/// DataFusion's TopK (a `SortExec` with a fetch) whose input is sorted by a
/// prefix of its keys ends a partition as soon as no later row of it can make
/// it into the result, so that a partition of DataFusion's window can emit
/// its rows while the input of another partition never ends. The native
/// window reads all of its input before it emits anything: over an unbounded
/// input, it would never emit. (Over a bounded input, it does once the input
/// ends.)
fn ends_unbounded_input_early_below(window: &Arc<dyn ExecutionPlan>) -> bool {
    let mut nodes = window.children();
    while let Some(node) = nodes.pop() {
        if let Some(sort) = node.downcast_ref::<SortExec>()
            && sort.fetch().is_some()
            && sort.input().boundedness().is_unbounded()
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
    let Some(sort) = projection
        .input()
        .downcast_ref::<SortExec>()
        .filter(|sort| sort.fetch().is_none())
    else {
        return Ok(None);
    };
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
/// every partition of its input, without a fetch.
fn without_ordering_sort<'a>(
    raw: &AggregateExec,
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
    if orders_aggregates && sorts_partitions && sort.fetch().is_none() {
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
/// `PartialReduce` that merges them into other groups.
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
/// * a `RepartitionExec` that only distributes them (see
///   [`redistributes_rows`]);
/// * a `CoalescePartitionsExec` without a fetch;
/// * a `CoalesceBatchesExec` without a fetch;
/// * a `PartialReduce` aggregation that merges the states of `partial` into
///   the same states: it groups by the group columns of the states and merges
///   the aggregates of `partial`, without filters or a limit.
fn passes_states_on(stage: &Arc<dyn ExecutionPlan>, partial: &AggregateExec) -> bool {
    if let Some(repartition) = stage.downcast_ref::<RepartitionExec>() {
        return redistributes_rows(repartition);
    }
    if let Some(coalesce) = stage.downcast_ref::<CoalescePartitionsExec>() {
        return coalesce.fetch().is_none();
    }
    if is_coalesce_batches(stage) {
        return stage.fetch().is_none();
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

/// Whether `input` is `side`, or a node below it through nodes of a single
/// child (e.g. the input of a join without the nodes that distribute its
/// rows), see [`Lowering::lower_join_input`].
fn is_within(side: &Arc<dyn ExecutionPlan>, input: &Arc<dyn ExecutionPlan>) -> bool {
    Arc::ptr_eq(side, input)
        || matches!(side.children().as_slice(), [child] if is_within(child, input))
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

/// The dynamic filter of `sort`, a `SortExec`, if `sort` is a TopK whose
/// dynamic filter a node of its input reads (see
/// [`consumes_dynamic_filter`]).
fn read_topk_filter(sort: &SortExec) -> Option<Arc<DynamicFilterPhysicalExpr>> {
    sort.fetch()?;
    let filter = sort
        .dynamic_expressions_produced()
        .into_iter()
        .find(|expr| expr.is::<DynamicFilterPhysicalExpr>())?;
    let id = filter.expression_id()?;
    let filter = (filter as Arc<dyn std::any::Any + Send + Sync>)
        .downcast::<DynamicFilterPhysicalExpr>()
        .ok()?;
    consumes_dynamic_filter(sort.input(), id).then_some(filter)
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

/// `lowered`, the native sort of a plan whose `SortExec` has a dynamic filter
/// that no node reads (see [`with_unread_topk_filter`]), updating `filter`,
/// the dynamic filter of the `SortExec` of the plan that the nodes of its
/// input read, if it is a [`PhysicalTopK`] that updates a filter: with the
/// same keys, so that these nodes skip the rows that the threshold of the
/// native TopK rules out.
///
/// Any other sort is returned as it is, and so is a TopK by float keys that
/// runs DataFusion's TopK if its result may differ (see
/// [`FloatKeyFallback`](crate::sort::FloatKeyFallback)), which would run the
/// plan whose filter no node reads: then nothing updates `filter`, and the
/// nodes that read it prune nothing, which cannot change the result.
fn updating_filter(
    lowered: Arc<dyn PhysicalOperator>,
    filter: Arc<DynamicFilterPhysicalExpr>,
) -> Result<Arc<dyn PhysicalOperator>> {
    let Some(topk) = lowered.downcast_ref::<PhysicalTopK>() else {
        return Ok(lowered);
    };
    let Some(dynamic_filter) = topk.dynamic_filter() else {
        return Ok(lowered);
    };
    if topk.float_key_fallback().is_some() {
        return Ok(lowered);
    }
    let updating = PhysicalTopK::try_new(
        Arc::clone(topk.input()),
        topk.ordering().clone(),
        topk.k(),
    )?
    .with_dynamic_filter(TopKDynamicFilter::new(
        filter,
        dynamic_filter.keys().to_vec(),
    ))?;
    Ok(Arc::new(updating))
}

/// The dynamic filter of a hash join routed to a scan below its probe side
/// (see [`route_join_filters`]).
#[derive(Debug)]
struct JoinFilterRoute {
    /// The filter that the join sets: of its probe keys that the scan
    /// produces.
    target: JoinFilterTarget,
    /// The same filter, over the columns of the scan, that the scan is lowered
    /// with.
    scan_filter: ScanJoinFilter,
}

/// The dynamic filter of a hash join that a scan is lowered with (see
/// [`Lowering::lower_scan`]).
#[derive(Debug, Clone)]
struct ScanJoinFilter {
    /// The scan, a `DataSourceExec` of the plan.
    scan: Arc<dyn ExecutionPlan>,
    /// The filter, over the output columns of the scan.
    filter: Arc<dyn PhysicalExpr>,
    /// Whether the scan was lowered with the filter, which the join then sets.
    attached: Rc<Cell<bool>>,
}

/// The scans below the probe side of `plan`, a hash join, that its dynamic
/// filter is routed to, DuckDB's join filter pushdown: the scans whose output
/// columns are probe keys of the join, unchanged (see [`scan_targets`]), each
/// with a new dynamic filter of these keys (see
/// [`JoinFilterTarget::filter_of`]), over its columns. The join prunes its
/// probe rows that no build key matches, which the scans then skip:
///
/// * only joins that drop their probe rows without a match route filters:
///   none that emits them (the side of an outer join whose rows must be
///   kept), nor a null-aware one, whose probe rows count also if they do not
///   match (their native joins publish no filters either);
/// * only probe keys that are columns, whose filters cannot fail on any row
///   of the scans;
/// * a Parquet scan gets the filter unless it reads the dynamic filter of the
///   join already (which DataFusion pushes into the scans that it reaches,
///   with all the keys): it prunes its files, row groups and pages with it;
/// * a scan of a memory table or of CSV files, which does not prune, gets
///   the filter only if its rows reach the join through other joins,
///   aggregations or sorts, whose work on the rows that the filter rejects it
///   saves: the probe of the join right above it would reject them anyway;
/// * no scan that occurs several times in the probe side (a node that the
///   plan shares between several inputs), which might also be read through
///   nodes that the filter cannot go through.
fn route_join_filters(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<JoinFilterRoute>> {
    let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
        return Ok(vec![]);
    };
    let keeps_unmatched_probe_rows = matches!(
        join.join_type(),
        JoinType::Right | JoinType::Full | JoinType::RightAnti | JoinType::RightMark
    );
    if join.null_aware || keeps_unmatched_probe_rows || join.fetch().is_some() {
        return Ok(vec![]);
    }
    let probe = join.right();
    let columns = join
        .on()
        .iter()
        .map(|(_, probe_key)| probe_key.downcast_ref::<Column>().map(Column::index))
        .collect();
    let mut targets = vec![];
    scan_targets(probe, columns, false, &mut targets);
    let join_filters = plan
        .dynamic_expressions_produced()
        .iter()
        .filter_map(|filter| filter.expression_id())
        .collect::<Vec<_>>();
    let probe_schema = probe.schema();
    let mut routes = vec![];
    for ScanTarget {
        scan,
        columns,
        deep,
    } in targets
    {
        let Some(exec) = scan.downcast_ref::<DataSourceExec>() else {
            continue;
        };
        let useful = if exec.downcast_to_file_source::<ParquetSource>().is_some() {
            !join_filters
                .iter()
                .any(|&filter| consumes_dynamic_filter(&scan, filter))
        } else {
            deep && (exec.data_source().is::<MemorySourceConfig>()
                || exec.downcast_to_file_source::<CsvSource>().is_some())
        };
        if !useful || occurrences(probe, &scan) != 1 {
            continue;
        }
        // The keys that the scan produces, with the same types.
        let scan_schema = scan.schema();
        let (keys, scan_columns): (Vec<usize>, Vec<Arc<dyn PhysicalExpr>>) = columns
            .iter()
            .enumerate()
            .filter_map(|(key, column)| {
                let field = scan_schema.field(column.as_ref().copied()?);
                let probe_key = &join.on()[key].1;
                let same_type = probe_key
                    .data_type(&probe_schema)
                    .is_ok_and(|data_type| &data_type == field.data_type());
                same_type.then(|| {
                    let column = Column::new(field.name(), (*column)?);
                    Some((key, Arc::new(column) as Arc<dyn PhysicalExpr>))
                })?
            })
            .unzip();
        if keys.is_empty() {
            continue;
        }
        let filter = JoinFilterTarget::filter_of(join.on(), &keys)?;
        let scan_filter = Arc::clone(&filter).with_new_children(scan_columns)?;
        routes.push(JoinFilterRoute {
            target: JoinFilterTarget::new(filter, keys),
            scan_filter: ScanJoinFilter {
                scan,
                filter: scan_filter,
                attached: Rc::new(Cell::new(false)),
            },
        });
    }
    Ok(routes)
}

/// How many times `node` occurs in the plan `plan`.
fn occurrences(plan: &Arc<dyn ExecutionPlan>, node: &Arc<dyn ExecutionPlan>) -> usize {
    usize::from(Arc::ptr_eq(plan, node))
        + plan
            .children()
            .into_iter()
            .map(|child| occurrences(child, node))
            .sum::<usize>()
}

/// A scan below the probe side of a hash join whose output columns are probe
/// keys of the join, unchanged (see [`scan_targets`]).
#[derive(Debug)]
struct ScanTarget {
    /// The scan, a `DataSourceExec`.
    scan: Arc<dyn ExecutionPlan>,
    /// For every probe key of the join, the output column of the scan that
    /// holds its values, if any.
    columns: Vec<Option<usize>>,
    /// Whether the rows of the scan reach the join through another join, an
    /// aggregation or a sort.
    deep: bool,
}

/// Add to `targets` the scans below `plan` (see [`ScanTarget`]) whose output
/// columns hold the values of the output columns `columns` of `plan` (for
/// every probe key of a hash join above, the column that holds its values,
/// if any), unchanged, along a path on which removing the rows of the scan
/// whose keys no build key of the join matches only removes rows of `plan`
/// whose keys no build key matches (which the join drops anyway). `deep` is
/// whether the rows of `plan` reach the join through other joins,
/// aggregations or sorts.
///
/// The columns are followed through
///
/// * projections (of columns, not of other expressions), filters, the nodes
///   that only distribute or coalesce rows, sorts, and unions (into all their
///   inputs);
/// * hash joins, nested loop joins and cross joins, into the input of each
///   column whose rows the join emits only with their matches (see
///   [`preserved_inputs`]: both inputs of inner and semi joins, the input of
///   an outer join whose unmatched rows it emits, the output input of an
///   anti or mark join, not the null side of an outer join), and from the
///   key of an inner or semi hash join on one input to the key on the other
///   input, which every pair of rows of the join shares (as DataFusion
///   transfers filters, see `HashJoinExec::gather_filters_for_pushdown`);
/// * aggregations, from the columns of their groups that are columns of
///   their input in all their grouping sets: removing rows of a group
///   removes the group, but changes no other group.
///
/// They are not followed through any other node, nor through nodes with a
/// limit (whose rows other rows would replace), aggregations with a limit on
/// their groups, windows (whose values are computed over other rows), or into
/// scans with a limit.
fn scan_targets(
    plan: &Arc<dyn ExecutionPlan>,
    columns: Vec<Option<usize>>,
    deep: bool,
    targets: &mut Vec<ScanTarget>,
) {
    if columns.iter().all(Option::is_none) || plan.fetch().is_some() {
        return;
    }
    if plan.is::<DataSourceExec>() {
        targets.push(ScanTarget {
            scan: Arc::clone(plan),
            columns,
            deep,
        });
        return;
    }
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let input_columns = columns
            .iter()
            .map(|column| {
                let expr = &projection.expr().get((*column)?)?.expr;
                expr.downcast_ref::<Column>().map(Column::index)
            })
            .collect();
        return scan_targets(projection.input(), input_columns, deep, targets);
    }
    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        let input_columns = match filter.projection() {
            Some(projection) => columns
                .iter()
                .map(|column| projection.get((*column)?).copied())
                .collect(),
            None => columns,
        };
        return scan_targets(filter.input(), input_columns, deep, targets);
    }
    let passes_rows_on = plan.is::<RepartitionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || is_coalesce_batches(plan);
    let sorts = plan.is::<SortExec>() || plan.is::<SortPreservingMergeExec>();
    if passes_rows_on || sorts {
        if let [input] = plan.children().as_slice() {
            scan_targets(input, columns, deep || sorts, targets);
        }
        return;
    }
    if plan.is::<UnionExec>() || plan.is::<InterleaveExec>() {
        for input in plan.children() {
            scan_targets(input, columns.clone(), deep, targets);
        }
        return;
    }
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        if aggregate.limit_options().is_some() {
            return;
        }
        let group_by = aggregate.group_expr();
        let input_columns = columns
            .iter()
            .map(|column| {
                let column = (*column)?;
                let (expr, _) = group_by.expr().get(column)?;
                let in_all_sets = group_by
                    .groups()
                    .iter()
                    .all(|nulls| !nulls.get(column).copied().unwrap_or(true));
                in_all_sets
                    .then(|| expr.downcast_ref::<Column>().map(Column::index))
                    .flatten()
            })
            .collect();
        return scan_targets(aggregate.input(), input_columns, true, targets);
    }
    let (join_type, projection, on) =
        if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
            (*join.join_type(), join.projection.as_deref(), join.on())
        } else if let Some(join) = plan.downcast_ref::<NestedLoopJoinExec>() {
            (*join.join_type(), join.projection().as_deref(), &[][..])
        } else if plan.is::<CrossJoinExec>() {
            (JoinType::Inner, None, &[][..])
        } else {
            return;
        };
    let [left, right] = plan.children()[..] else {
        return;
    };
    let (_, column_indices) =
        build_join_schema(&left.schema(), &right.schema(), &join_type);
    let (left_preserved, right_preserved) = preserved_inputs(join_type);
    let transfers = matches!(
        join_type,
        JoinType::Inner | JoinType::LeftSemi | JoinType::RightSemi
    );
    let mut left_columns = vec![None; columns.len()];
    let mut right_columns = vec![None; columns.len()];
    for (key, column) in columns.iter().enumerate() {
        let Some(column) = *column else {
            continue;
        };
        let column =
            projection.map_or(Some(column), |projection| projection.get(column).copied());
        let Some(ColumnIndex { index, side }) =
            column.and_then(|column| column_indices.get(column))
        else {
            continue;
        };
        let (preserved, own, other) = match side {
            JoinSide::Left => (left_preserved, &mut left_columns, &mut right_columns),
            JoinSide::Right => (right_preserved, &mut right_columns, &mut left_columns),
            JoinSide::None => continue,
        };
        if preserved {
            own[key] = Some(*index);
        }
        // The key on the other input, which every row of the join shares.
        if transfers {
            other[key] = on.iter().find_map(|(left_key, right_key)| {
                let (own_key, other_key) = match side {
                    JoinSide::Left => (left_key, right_key),
                    _ => (right_key, left_key),
                };
                own_key
                    .downcast_ref::<Column>()
                    .filter(|own_key| own_key.index() == *index)?;
                other_key.downcast_ref::<Column>().map(Column::index)
            });
        }
    }
    scan_targets(left, left_columns, true, targets);
    scan_targets(right, right_columns, true, targets);
}

/// Whether removing the rows of the left and of the right input of a join of
/// type `join_type` whose keys no build key of a hash join above matches only
/// removes rows of the join whose keys no build key matches: whether the join
/// emits the rows of the input only with their matches, or alone (both inputs
/// of inner and semi joins, the inputs of outer joins whose unmatched rows
/// they emit, the output inputs of anti and mark joins). Not the null side of
/// an outer join, which other rows would be emitted without (with nulls in its
/// columns). DataFusion's `lr_is_preserved` of hash joins.
fn preserved_inputs(join_type: JoinType) -> (bool, bool) {
    match join_type {
        JoinType::Inner | JoinType::LeftSemi | JoinType::RightSemi => (true, true),
        JoinType::Left | JoinType::LeftAnti | JoinType::LeftMark => (true, false),
        JoinType::Right | JoinType::RightAnti | JoinType::RightMark => (false, true),
        JoinType::Full => (false, false),
    }
}

/// The rows that a [`JoinFilteredScan`] evaluates a filter on before it
/// decides whether to keep it: 32 chunks.
const JOIN_FILTER_SAMPLE_ROWS: usize = 32 * DEFAULT_VECTOR_SIZE;

/// A [`JoinFilteredScan`] keeps evaluating a filter that rejects at least one
/// in this many of the rows it sampled.
const JOIN_FILTER_MIN_REJECTED: usize = 8;

/// A native scan of a memory table or of CSV files whose chunks are filtered
/// with the dynamic filters of hash joins above it, which the planner routed
/// to it (see [`Lowering::try_lower_hash_join`]), as DuckDB applies its join
/// filters in its table scans. The rows of the scan reach the joins through
/// other joins, aggregations or sorts: the rows that a filter rejects (which
/// no build key of its join matches) are not joined, aggregated or sorted
/// before the join drops them.
///
/// A filter is evaluated on the logical rows of every chunk, and the chunk
/// gets a selection of the rows that every filter accepts (a `NULL` rejects a
/// row). As DuckDB's selectivity-optional filters, the scan stops evaluating
/// a filter that rejects fewer than one in [`JOIN_FILTER_MIN_REJECTED`] of the
/// first [`JOIN_FILTER_SAMPLE_ROWS`] rows that it evaluated it on, on all
/// threads: then it would cost more than the work it saves (e.g. the lookup
/// in the hash table of a join whose build side has nearly all the keys).
/// Which rows a filter rejects is not part of the result.
///
/// It has the name of the scan, and describes the filters after it.
#[derive(Debug)]
struct JoinFilteredScan {
    scan: Arc<dyn PhysicalOperator>,
    filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl JoinFilteredScan {
    fn new(scan: Arc<dyn PhysicalOperator>, filters: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self { scan, filters }
    }

    /// The chunk of the rows of `chunk` that every filter that `state` keeps
    /// accepts.
    fn filter(
        &self,
        chunk: DataChunk,
        state: &JoinFilteredScanState,
    ) -> Result<DataChunk> {
        let mut batch = None;
        let mut selected: Option<arrow::array::BooleanArray> = None;
        for (filter, selectivity) in self.filters.iter().zip(&state.selectivities) {
            if !selectivity.is_evaluated() {
                continue;
            }
            let batch = match &batch {
                Some(batch) => batch,
                None => batch.insert(chunk.compact()?),
            };
            let accepted = filter.evaluate(batch)?.into_array(batch.num_rows())?;
            // A `NULL` rejects its row.
            let accepted = as_boolean_array(&accepted)?;
            let accepted = match accepted.nulls() {
                Some(_) => prep_null_mask_filter(accepted),
                None => accepted.clone(),
            };
            selectivity.record(batch.num_rows(), accepted.true_count());
            selected = Some(match selected {
                Some(selected) => and(&selected, &accepted)?,
                None => accepted,
            });
        }
        let (Some(batch), Some(selected)) = (batch, selected) else {
            return Ok(chunk);
        };
        match selected.true_count() {
            rows if rows == batch.num_rows() => Ok(chunk),
            0 => Ok(DataChunk::empty(Arc::clone(chunk.schema()))),
            _ => DataChunk::from_record_batch(batch)
                .with_selection(SelectionVector::from_boolean_mask(&selected)),
        }
    }
}

/// The global state of a [`JoinFilteredScan`].
struct JoinFilteredScanState {
    scan: Arc<dyn GlobalSourceState>,
    /// For every filter, whether it is evaluated.
    selectivities: Vec<FilterSelectivity>,
}

impl GlobalSourceState for JoinFilteredScanState {
    fn max_threads(&self) -> usize {
        self.scan.max_threads()
    }
}

/// How many of the rows that a [`JoinFilteredScan`] sampled a filter on it
/// rejected, and whether the scan keeps evaluating it.
#[derive(Debug, Default)]
struct FilterSelectivity {
    rows: AtomicUsize,
    rejected: AtomicUsize,
    /// [`Self::SAMPLING`], [`Self::KEPT`] or [`Self::DROPPED`].
    decision: AtomicU8,
}

impl FilterSelectivity {
    const SAMPLING: u8 = 0;
    const KEPT: u8 = 1;
    const DROPPED: u8 = 2;

    fn is_evaluated(&self) -> bool {
        self.decision.load(AtomicOrdering::Relaxed) != Self::DROPPED
    }

    /// Record that the filter accepted `accepted` of `rows` rows, and decide
    /// whether to keep it once it was evaluated on enough rows.
    fn record(&self, rows: usize, accepted: usize) {
        if self.decision.load(AtomicOrdering::Relaxed) != Self::SAMPLING {
            return;
        }
        let rejected = rows - accepted;
        let rejected =
            self.rejected.fetch_add(rejected, AtomicOrdering::Relaxed) + rejected;
        let rows = self.rows.fetch_add(rows, AtomicOrdering::Relaxed) + rows;
        if rows >= JOIN_FILTER_SAMPLE_ROWS {
            let decision = if rejected * JOIN_FILTER_MIN_REJECTED >= rows {
                Self::KEPT
            } else {
                Self::DROPPED
            };
            let _ = self.decision.compare_exchange(
                Self::SAMPLING,
                decision,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            );
        }
    }
}

impl PhysicalOperator for JoinFilteredScan {
    fn name(&self) -> &str {
        self.scan.name()
    }

    fn schema(&self) -> SchemaRef {
        self.scan.schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        let filters = self
            .filters
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        format!(
            "{}, join_filters=[{}]",
            self.scan.description(),
            filters.join(", ")
        )
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        self.scan.parallel_source()
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(JoinFilteredScanState {
            scan: self.scan.get_global_source_state(ctx)?,
            selectivities: self
                .filters
                .iter()
                .map(|_| FilterSelectivity::default())
                .collect(),
        }))
    }

    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        let gstate =
            downcast_state::<JoinFilteredScanState>(gstate.as_any(), self.name())?;
        self.scan.get_local_source_state(ctx, gstate.scan.as_ref())
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let gstate =
            downcast_state::<JoinFilteredScanState>(gstate.as_any(), self.name())?;
        match self.scan.get_data(ctx, gstate.scan.as_ref(), lstate)? {
            SourceResult::Chunk(chunk) if !chunk.is_empty() => {
                Ok(SourceResult::Chunk(self.filter(chunk, gstate)?))
            }
            result => Ok(result),
        }
    }
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
    use datafusion_common::tree_node::{Transformed, TreeNode};
    use datafusion_datasource::file_scan_config::FileScanConfig;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_datasource_parquet::ParquetFileReaderFactory;
    use datafusion_datasource_parquet::source::ParquetSource;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::{BinaryExpr, Literal, col};
    use datafusion_physical_plan::aggregates::PhysicalGroupBy;
    use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
    use datafusion_physical_plan::limit::GlobalLimitExec;
    use datafusion_physical_plan::sorts::partial_sort::PartialSortExec;
    use datafusion_physical_plan::{PhysicalExpr, collect, displayable};

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

    /// Assert that `sql` produces the same results through the
    /// [`PipelineQueryPlanner`] and on default DataFusion, compared in their
    /// canonical forms (see `compare_outputs`), in sessions configured with
    /// `config` and set up with `setup`. Returns the operator tree of the
    /// pipeline engine.
    async fn assert_same_results_with<F>(
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

    /// [`assert_same_results_with`] with the default tables and configuration.
    async fn assert_same_results(sql: &str) -> Arc<dyn PhysicalOperator> {
        assert_same_results_with(sql, test_session_config(), setup_test_tables).await
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
            assert_same_results(sql).await;
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
                assert_same_results_with(sql, test_session_config(), setup_view_tables)
                    .await;
            assert_native(sql, &root);
            assert_eq!(
                count::<PhysicalHashAggregate>(&root),
                aggregations,
                "{sql}\n{}",
                format_operator_tree(&root)
            );
        }

        // Ordered aggregates: the native aggregation orders the values itself,
        // without the sort DataFusion plans below it, over one partition or
        // several (whose states, lists, pass through a repartitioning).
        let sql = "SELECT d, ARRAY_AGG(id ORDER BY id) AS ids, \
                   FIRST_VALUE(a ORDER BY id DESC) AS last_a, \
                   NTH_VALUE(h, 2 ORDER BY id) AS second_h FROM t1 GROUP BY d";
        let config = test_session_config().with_target_partitions(1);
        let plan = datafusion_plan(sql, config.clone()).await;
        assert_eq!(count_nodes(&plan, "SortExec"), 1);
        for config in [config, test_session_config()] {
            let root = assert_same_results_with(sql, config, setup_view_tables).await;
            assert_native(sql, &root);
            assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
            assert_eq!(count::<PhysicalSort>(&root), 0);
        }
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
                    assert_same_results_with(sql, config.clone(), setup_view_tables)
                        .await;
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
                assert_same_results_with(sql, test_session_config(), setup_view_tables)
                    .await;
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
            let root = assert_same_results(sql).await;
            assert_native(sql, &root);
        }
        let sql = "SELECT id, a FROM t1 WHERE a > 10";
        let root = assert_same_results(sql).await;
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
                assert_same_results_with(sql, test_session_config(), setup_view_tables)
                    .await;
            assert_native(sql, &root);
        }
        // Aggregations whose states are lists, over one partition or several.
        for sql in [
            // Of a union.
            "SELECT k, COUNT(*) AS n, COUNT(DISTINCT w) AS dw FROM \
             (SELECT a AS k, b AS w FROM t1 UNION ALL SELECT k, w FROM t2) \
             GROUP BY k ORDER BY k",
            // Ordered aggregates of a join, without the sort of their input.
            "SELECT t1.e, ARRAY_AGG(t2.w ORDER BY t2.w DESC NULLS LAST, t1.id) AS ws \
             FROM t1 JOIN t2 ON t1.a = t2.k GROUP BY t1.e",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let root = assert_same_results_with(sql, config, setup_view_tables).await;
                assert_native(sql, &root);
            }
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
                    assert_same_results_with(sql, config.clone(), setup_view_tables)
                        .await;
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
                assert_same_results_with(sql, test_session_config(), setup_view_tables)
                    .await;
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
                assert_same_results_with(sql, test_session_config(), setup_view_tables)
                    .await;
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
                    assert_same_results_with(sql, config.clone(), setup_view_tables)
                        .await;
                assert_native(sql, &root);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Plans left to DataFusion, and plans that are native whatever their
    // expressions, inputs and limits
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
    async fn windows_over_topks_that_end_early_wait_for_bounded_inputs() -> Result<()> {
        // A window over the partitions of a TopK of a hash repartitioning of an
        // input sorted by `k`, a prefix of the keys of the TopK: DataFusion's
        // TopK ends a partition as soon as no later row can make it into the
        // result. The only batch of the input has rows of 3 values of `k`, all
        // in the first partition, so that DataFusion's window emits a row of the
        // first partition while the TopK of the second partition waits for the
        // input. A native window reads all of its input first: over an
        // unbounded input (which never ends, or fails later), it would never
        // emit, so the window is left to DataFusion. Over an input that ends,
        // the window is native, and returns the rows of DataFusion.
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
            // An input that ends, in the order of `k` or in no order.
            for sorted in [true, false] {
                let expected = rows(make(Tail::End, sorted, bounded)?).await;
                let root = lower_for_default_runtime(
                    make(Tail::End, sorted, bounded)?,
                    &config,
                )?;
                assert_eq!(
                    count::<PhysicalWindow>(root.operator()),
                    1,
                    "{}",
                    format_operator_tree(root.operator())
                );
                assert_eq!(rows(root).await, expected, "sorted: {sorted}");
            }
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
    async fn cross_joins_of_values_that_address_more_than_their_memory_are_left_to_datafusion()
    -> Result<()> {
        // Two one-row build batches of lists of `1 << 30` nulls, which take no
        // memory: the native cross join would materialize them as it gathers
        // its rows, so it is left to DataFusion (see
        // `gathers_values_beyond_their_memory`), which fails to concatenate
        // them.
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
        let outcomes = Outcomes::of_plan(make_plan(true), config.clone()).await;
        let error = outcomes.assert_datafusion_fails();
        outcomes.assert_pipeline_fails_with(&[&error]);
        assert_eq!(count::<PhysicalCrossJoin>(&outcomes.operator), 0);
        // With other values, both engines return no rows.
        let outcomes = Outcomes::of_plan(make_plan(false), config).await;
        outcomes.assert_same_rows();
        assert_eq!(count::<PhysicalCrossJoin>(&outcomes.operator), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_of_values_that_may_fail_to_combine_are_native() -> Result<()> {
        // DataFusion's repartitioning below a window coalesces the rows of a
        // partition, and its sort concatenates them: two rows of lists of
        // `1 << 30` nulls fail. The native window would materialize them as it
        // gathers its rows, so it is left to DataFusion (see
        // `gathers_values_beyond_their_memory`). Over other values, the
        // window is native.
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
            let outcomes =
                Outcomes::of_plan(make_plan(true, bounded), config.clone()).await;
            let error = outcomes.assert_datafusion_fails();
            outcomes.assert_pipeline_fails_with(&[&error]);
            assert_eq!(count::<PhysicalWindow>(&outcomes.operator), 0);
            let outcomes =
                Outcomes::of_plan(make_plan(false, bounded), config.clone()).await;
            outcomes.assert_same_rows();
            assert_eq!(count::<PhysicalWindow>(&outcomes.operator), 1);
        }
        // Strings with 32-bit offsets (`Utf8`), which DataFusion may fail to
        // concatenate too, like `Utf8View` strings.
        let sql =
            "SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1";
        let root =
            assert_same_results_with(sql, test_session_config(), setup_test_tables).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        let root =
            assert_same_results_with(sql, test_session_config(), setup_view_tables).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_projections_of_sorts_that_may_fail_to_combine_are_native()
    -> Result<()> {
        // DataFusion plans a projection of columns between a window and the sort
        // below it, which the planner moves below the sort to lower the window.
        // DataFusion's sort sorts all columns of its input, also those that the
        // projection drops: two rows of lists of `1 << 30` nulls, which the
        // sort concatenates, fail. The native window sorts the projected rows
        // only, and returns them: the rows of the plan with other values in the
        // column that the projection drops.
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
            let expected =
                Outcomes::of_plan(make_plan(false, bounded), config.clone()).await;
            expected.assert_same_rows();
            assert_eq!(count::<PhysicalWindow>(&expected.operator), 1);
            assert_eq!(count::<PhysicalProjection>(&expected.operator), 1);
            let outcomes =
                Outcomes::of_plan(make_plan(true, bounded), config.clone()).await;
            outcomes.assert_datafusion_fails();
            outcomes.assert_pipeline_rows(expected.pipeline.as_ref().unwrap());
            assert_eq!(count::<PhysicalWindow>(&outcomes.operator), 1);
        }
        Ok(())
    }

    /// A window function that returns dense unions of two children of
    /// integers, `(type id, value)`: `(1, n)`, `(1, n - 1)`, ... for the first
    /// half of the `n` rows of its partition, and `(0, n / 2)`, ..., `(0, 1)`
    /// for the second half (or, if `nested`, lists of one of these unions),
    /// which it declares to be in descending order (as `ROW_NUMBER` declares
    /// its numbers to be in ascending order): DataFusion orders unions by
    /// their type ids first, then by their values.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct DescendingUnions {
        signature: datafusion_expr::Signature,
        nested: bool,
    }

    impl DescendingUnions {
        fn new(nested: bool) -> Self {
            Self {
                signature: datafusion_expr::Signature::exact(
                    vec![],
                    datafusion_expr::Volatility::Immutable,
                ),
                nested,
            }
        }

        fn union_type() -> DataType {
            DataType::Union(Self::fields(), arrow::datatypes::UnionMode::Dense)
        }

        fn item() -> arrow::datatypes::FieldRef {
            Arc::new(Field::new("item", Self::union_type(), false))
        }

        fn fields() -> arrow::datatypes::UnionFields {
            arrow::datatypes::UnionFields::try_new(
                vec![0, 1],
                vec![
                    Field::new("a", DataType::Int32, false),
                    Field::new("b", DataType::Int32, false),
                ],
            )
            .unwrap()
        }
    }

    impl datafusion_expr::WindowUDFImpl for DescendingUnions {
        fn name(&self) -> &str {
            "descending_unions"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn partition_evaluator(
            &self,
            _args: datafusion_expr::function::PartitionEvaluatorArgs,
        ) -> Result<Box<dyn datafusion_expr::PartitionEvaluator>> {
            Ok(Box::new(DescendingUnionsEvaluator {
                nested: self.nested,
            }))
        }

        fn field(
            &self,
            field_args: datafusion_expr::function::WindowUDFFieldArgs,
        ) -> Result<arrow::datatypes::FieldRef> {
            let data_type = match self.nested {
                true => DataType::List(Self::item()),
                false => Self::union_type(),
            };
            Ok(Field::new(field_args.name(), data_type, false).into())
        }

        fn sort_options(&self) -> Option<arrow::compute::SortOptions> {
            Some(arrow::compute::SortOptions {
                descending: true,
                nulls_first: false,
            })
        }
    }

    /// The evaluator of [`DescendingUnions`].
    #[derive(Debug)]
    struct DescendingUnionsEvaluator {
        nested: bool,
    }

    impl datafusion_expr::PartitionEvaluator for DescendingUnionsEvaluator {
        fn evaluate_all(
            &mut self,
            _values: &[arrow::array::ArrayRef],
            num_rows: usize,
        ) -> Result<arrow::array::ArrayRef> {
            let rows = num_rows as i32;
            let half = rows / 2;
            let type_ids = (0..rows).map(|i| i8::from(i < half)).collect();
            let offsets = (0..rows)
                .map(|i| if i < half { i } else { i - half })
                .collect();
            let values = |range: std::ops::Range<i32>| -> arrow::array::ArrayRef {
                Arc::new(Int32Array::from_iter_values(range.map(|i| rows - i)))
            };
            let unions = Arc::new(arrow::array::UnionArray::try_new(
                DescendingUnions::fields(),
                type_ids,
                Some(offsets),
                vec![values(half..rows), values(0..half)],
            )?);
            if !self.nested {
                return Ok(unions);
            }
            Ok(Arc::new(arrow::array::ListArray::try_new(
                DescendingUnions::item(),
                arrow::buffer::OffsetBuffer::from_lengths(vec![1; num_rows]),
                unions,
                None,
            )?))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_windows_by_keys_of_unions_are_left_to_datafusion() -> Result<()> {
        // `... ORDER BY rn DESC LIMIT 1` over a window whose function returns
        // the unions `(1, 4)`, `(1, 3)`, `(0, 2)`, `(0, 1)` of the rows `id` 0
        // to 3, in descending order: DataFusion's merge returns the row 0. The
        // native sort orders unions by the row format, by their type ids first
        // also for a descending key: it would return another row, a wrong
        // result. So the merge stays in DataFusion (see
        // `Lowering::try_lower_merge_of_window`), as the sort module leaves
        // sorts by keys of unions (also nested) to DataFusion.
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let rows = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(0..4))],
        )?;
        // The window of the unions (or of lists of them), and the merge of its
        // partitions by them, with a fetch of 1.
        let plan = |nested: bool| -> Result<Arc<dyn ExecutionPlan>> {
            let scan = MemorySourceConfig::try_new_exec(
                &[vec![rows.clone()]],
                Arc::clone(&schema),
                None,
            )?;
            let udwf = Arc::new(datafusion_expr::WindowUDF::new_from_impl(
                DescendingUnions::new(nested),
            ));
            let window_expr = datafusion_physical_plan::windows::create_window_expr(
                &datafusion_expr::WindowFunctionDefinition::WindowUDF(udwf),
                "rn".to_string(),
                &[],
                &[],
                &[],
                Arc::new(datafusion_expr::WindowFrame::new(None)),
                Arc::clone(&schema),
                false,
                false,
                None,
            )?;
            let window: Arc<dyn ExecutionPlan> =
                Arc::new(WindowAggExec::try_new(vec![window_expr], scan, true)?);
            let ordering = LexOrdering::new([PhysicalSortExpr::new(
                col("rn", &window.schema())?,
                arrow::compute::SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            )])
            .unwrap();
            Ok(Arc::new(
                SortPreservingMergeExec::new(ordering, window).with_fetch(Some(1)),
            ))
        };
        let config = test_session_config();
        let task_ctx =
            || Arc::new(TaskContext::default().with_session_config(config.clone()));
        let ids = |batches: Vec<RecordBatch>| -> Vec<i32> {
            batches
                .iter()
                .flat_map(|batch| {
                    let ids = batch
                        .column(0)
                        .as_primitive::<arrow::datatypes::Int32Type>();
                    ids.values().to_vec()
                })
                .collect()
        };
        for nested in [false, true] {
            // The window declares the order of the merge: it is not sorted.
            let merge = plan(nested)?;
            let window = Arc::clone(merge.children()[0]);
            let ordering = merge.output_ordering().unwrap().clone();
            assert!(window.equivalence_properties().ordering_satisfy(ordering)?);
            let root = lower_for_default_runtime(plan(nested)?, &config)?;
            let tree = format_operator_tree(root.operator());
            assert!(root.operator().is::<BridgeOperator>(), "{tree}");
            let expected = ids(collect(merge, task_ctx()).await?);
            assert_eq!(expected, vec![0], "nested: {nested}");
            let actual = ids(collect(root as Arc<dyn ExecutionPlan>, task_ctx()).await?);
            assert_eq!(actual, expected, "nested: {nested}: {tree}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_no_rows_of_windows_produce_nothing() -> Result<()> {
        // A merge of the sorted partitions of a window with `fetch == Some(0)`
        // emits nothing, without reading its input (DataFusion does not cast
        // the strings below the window, one of which fails to cast): it is
        // lowered to a source of no rows, which does not execute its input
        // either.
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
            // No rows, without casting the strings.
            let root =
                assert_plan_outcome(make_plan(Some(0), bounded), config.clone(), None)
                    .await;
            assert!(
                root.is::<EmptyScanSource>(),
                "{}",
                format_operator_tree(&root)
            );
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
    async fn loop_joins_below_fallible_nodes_are_native() -> Result<()> {
        // DataFusion's cross join joins its probe batch of 3000 rows with every
        // build row in turn, a native cross join every chunk of 2048 rows of it:
        // the projection above it, on one thread, sees the pairs in another
        // order. `CAST(nullif(ps, bs) AS INT)` fails on `'x'` (probe row 2500)
        // with the first build row, and on `'y'` (the first 2048 probe rows)
        // with the second one: DataFusion fails on `'x'` first, a native cross
        // join may fail on `'y'` first. Which of several errors is raised is
        // the engine's own: the cross join is native.
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
        let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
        outcomes.assert_datafusion_fails_with("Cannot cast string 'x'");
        outcomes.assert_pipeline_fails_with(&[
            "Cannot cast string 'x'",
            "Cannot cast string 'y'",
        ]);
        assert_native(sql, &outcomes.operator);
        assert_eq!(count::<PhysicalCrossJoin>(&outcomes.operator), 1);
        assert_eq!(count::<PhysicalProjection>(&outcomes.operator), 1);
        // Without the cast, both return the same rows.
        let sql = "SELECT nullif(ps, bs) AS v FROM b CROSS JOIN p";
        let root = assert_same_results_with(sql, config, &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalCrossJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn plans_below_limits_are_native() {
        // DataFusion reads the input of a limit lazily, the native limit stops
        // the pipelines below it once it has its rows: which rows they
        // evaluate before (and the errors of these rows) is the engine's own.
        // Without an `ORDER BY`, which rows a limit returns is too, so these
        // queries count them.
        for (sql, rows) in [
            ("SELECT COUNT(*) AS n FROM (SELECT * FROM t1 LIMIT 5)", 5),
            (
                "SELECT COUNT(*) AS n FROM (SELECT id, 10 / a AS x FROM t1 \
                 WHERE a > 0 LIMIT 70 OFFSET 3)",
                70,
            ),
            (
                "SELECT COUNT(*) AS n FROM (SELECT t1.id, t2.name FROM t1 \
                 JOIN t2 ON t1.a = t2.k LIMIT 9)",
                9,
            ),
            (
                "SELECT COUNT(*) AS n FROM (SELECT d, c FROM \
                 (SELECT d, COUNT(*) AS c FROM t1 GROUP BY d) WHERE c > 0 LIMIT 2)",
                2,
            ),
        ] {
            let root =
                assert_same_results_with(sql, test_session_config(), setup_view_tables)
                    .await;
            assert_native(sql, &root);
            assert!(count::<PhysicalLimit>(&root) > 0, "{sql}");
            let ctx = pipeline_context(test_session_config());
            register_view_tables(&ctx).unwrap();
            let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            let n = batches[0]
                .column(0)
                .as_primitive::<arrow::datatypes::Int64Type>()
                .value(0);
            assert_eq!(n, rows, "{sql}");
        }
        // And with a sort that reads all of its input in between.
        let sql = "SELECT id, a FROM t1 ORDER BY id LIMIT 10 OFFSET 5";
        let root = assert_same_results(sql).await;
        assert_native(sql, &root);
        let limit = root.downcast_ref::<PhysicalLimit>().expect("a limit");
        assert!(limit.input().is::<PhysicalTopK>());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_of_native_operators() {
        // The queries, with the errors that the pipeline engine may raise (none
        // if it returns the rows of DataFusion). All of them are native.
        for (sql, errors) in [
            // The filter removes the rows that would fail.
            ("SELECT 10 / a AS x FROM t1 WHERE a <> 0", &[][..]),
            ("SELECT SUM(10 / a) AS x FROM t1 WHERE a <> 0", &[]),
            (
                "SELECT d, SUM(1000 / a) AS x FROM t1 WHERE a > 0 GROUP BY d ORDER BY d",
                &[],
            ),
            ("SELECT 10 / (a - a) AS x FROM t1", &["Divide by zero"]),
            (
                "SELECT COUNT(*) AS n FROM t1 WHERE 10 / a > 1",
                &["Divide by zero"],
            ),
            (
                "SELECT d, SUM(10 / (a - a)) AS x FROM t1 GROUP BY d",
                &["Divide by zero"],
            ),
            (
                "SELECT t1.id, 100 / t2.k AS x FROM t1 JOIN t2 ON t1.a = t2.k",
                &["Divide by zero"],
            ),
            (
                "SELECT id FROM t1 ORDER BY 10 / (a - a), id",
                &["Divide by zero"],
            ),
            // DataFusion filters the rows of a batch before the aggregation
            // sees them, and fails to cast first; a pipeline evaluates both on
            // every chunk, and may divide by zero first.
            (
                "SELECT SUM(10 / a) AS x FROM t1 WHERE CAST(d AS INT) > 0",
                &["Cannot cast string", "Divide by zero"],
            ),
            (
                "SELECT d, SUM(10 / a) AS x FROM t1 WHERE CAST(d AS INT) > 0 GROUP BY d",
                &["Cannot cast string", "Divide by zero"],
            ),
            // Below a limit.
            (
                "SELECT 10 / (a - a) AS x FROM t1 LIMIT 3",
                &["Divide by zero"],
            ),
        ] {
            let outcomes =
                Outcomes::of_sql(sql, test_session_config(), setup_view_tables).await;
            if errors.is_empty() {
                outcomes.assert_same_rows();
            } else {
                outcomes.assert_datafusion_fails();
                outcomes.assert_pipeline_fails_with(errors);
            }
            assert_native(sql, &outcomes.operator);
        }
        // Also when DataFusion evaluates the division on the rows that the
        // filter rejects: unless at most 20% of the rows of a batch are left,
        // it evaluates the right side of an `AND` on all of them, and may
        // divide by zero. The native filter returns the rows of the same
        // query with a `CASE` that guards the division.
        let sql = "SELECT COUNT(*) AS n FROM t1 WHERE a <> 0 AND 10 / a > 1";
        let guarded =
            "SELECT COUNT(*) AS n FROM t1 WHERE CASE WHEN a <> 0 THEN 10 / a > 1 END";
        let reference =
            Outcomes::of_sql(guarded, test_session_config(), setup_view_tables).await;
        reference.assert_same_rows();
        let outcomes =
            Outcomes::of_sql(sql, test_session_config(), setup_view_tables).await;
        outcomes.assert_pipeline_rows(reference.datafusion.as_ref().unwrap());
        assert_native(sql, &outcomes.operator);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operators_of_a_pipeline_that_fail_with_other_errors_are_native() -> Result<()>
    {
        // One batch of 3000 rows: `10 / a` fails on row 100, `CAST(s AS INT)`
        // on row 2100. DataFusion filters the whole batch first, and fails to
        // cast. A pipeline (on a single thread, which reads the chunks in
        // order) of a native filter and a native projection (or aggregation)
        // evaluates both on the first chunk of 2048 rows, and divides by zero
        // first: which of several errors is raised is the engine's own.
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
            let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
            outcomes.assert_datafusion_fails_with("Cannot cast string 'x'");
            outcomes.assert_pipeline_fails_with(&[
                "Cannot cast string 'x'",
                "Divide by zero",
            ]);
            assert_native(sql, &outcomes.operator);
        }
        // So are the expressions of an aggregation or a sort, which DataFusion
        // evaluates on the whole batch one after the other.
        for sql in [
            "SELECT SUM(CAST(s AS INT)) AS x, SUM(10 / a) AS y FROM f",
            "SELECT a, SUM(CAST(s AS INT)) AS x, SUM(10 / a) AS y FROM f GROUP BY a",
            "SELECT a FROM f ORDER BY CAST(s AS INT), 10 / a",
        ] {
            let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
            outcomes.assert_datafusion_fails_with("Cannot cast string 'x'");
            outcomes.assert_pipeline_fails_with(&[
                "Cannot cast string 'x'",
                "Divide by zero",
            ]);
            assert_native(sql, &outcomes.operator);
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

    /// Assert that the hash joins of the plan `plan` are native in the
    /// operator tree of `outcomes` exactly if [`PhysicalHashJoin::try_lower`]
    /// lowers them: the planner leaves no hash join to DataFusion itself (the
    /// plans of these tests produce no dynamic filters).
    fn assert_join_lowered_by_the_native_join(
        outcomes: &Outcomes,
        plan: &Arc<dyn ExecutionPlan>,
    ) {
        let mut joins = 0;
        let mut lowered = 0;
        plan.apply(|node| {
            if node.is::<HashJoinExec>() {
                assert!(node.dynamic_expressions_produced().is_empty());
                joins += 1;
                lowered += usize::from(
                    PhysicalHashJoin::try_lower(node, lower_execution_plan)
                        .unwrap()
                        .is_some(),
                );
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        assert!(joins > 0, "no hash join");
        assert_eq!(
            count::<PhysicalHashJoin>(&outcomes.operator),
            lowered,
            "{}",
            outcomes.tree()
        );
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
    async fn batch_sensitive_aggregates_and_sort_keys_are_native() {
        // One batch of 3000 rows: 600 rows with `a = 1`, then 2400 with
        // `a = 0`. DataFusion evaluates the right side of
        // `a <> 0 AND 10 / a > 1` only on the rows that the left side selects
        // if at most 20% of the rows of the batch are left (600 of 3000): it
        // never divides by zero. On the first chunk of 2048 rows, 29% of the
        // rows are left, so that a native aggregation or sort, which evaluates
        // it on every chunk, may divide by zero: behaviour that depends on the
        // batches is the engine's own. They are native.
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
        ] {
            let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
            outcomes.assert_same_rows_or_fails_with(&["Divide by zero"]);
            assert_native(sql, &outcomes.operator);
            let natives = count::<PhysicalSort>(&outcomes.operator)
                + count::<PhysicalHashAggregate>(&outcomes.operator);
            assert_eq!(natives, 1, "{sql}\n{}", outcomes.tree());
        }
        // Functions and string literals (broadcast to the rows of a batch, e.g.
        // the separator of `STRING_AGG`), whose errors may depend on the
        // batches, are native too.
        for sql in [
            "SELECT id, d FROM t1 ORDER BY length(d), id",
            "SELECT upper(d) AS u, COUNT(*) AS n FROM t1 GROUP BY upper(d)",
        ] {
            let root = assert_same_results(sql).await;
            assert_native(sql, &root);
        }
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

    /// The outcomes of a query or a plan in DataFusion and in the pipeline
    /// engine, with the operator tree that the pipeline engine executes.
    struct Outcomes {
        /// The lowered operator tree.
        operator: Arc<dyn PhysicalOperator>,
        /// The schema of the plan of DataFusion.
        schema: SchemaRef,
        /// Whether the rows are in a defined order (a top-level `ORDER BY`).
        ordered: bool,
        datafusion: Result<Vec<RecordBatch>>,
        pipeline: Result<Vec<RecordBatch>>,
    }

    impl Outcomes {
        /// The outcomes of executing the plan that `make_plan` creates in
        /// DataFusion and lowered for the default runtime (see
        /// [`lower_for_default_runtime`]), in sessions configured with
        /// `config`.
        async fn of_plan(
            make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
            config: SessionConfig,
        ) -> Self {
            let root = lower_for_default_runtime(make_plan(), &config).unwrap();
            let operator = Arc::clone(root.operator());
            let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
            let plan = make_plan();
            let schema = plan.schema();
            let datafusion = collect(plan, Arc::clone(&task_ctx)).await;
            let pipeline = collect(root as Arc<dyn ExecutionPlan>, task_ctx).await;
            Self {
                operator,
                schema,
                ordered: false,
                datafusion,
                pipeline,
            }
        }

        /// The outcomes of `sql` in DataFusion and through the
        /// [`PipelineQueryPlanner`], in sessions configured with `config` and
        /// set up with `setup`.
        async fn of_sql<F>(sql: &str, config: SessionConfig, setup: F) -> Self
        where
            F: AsyncFn(&SessionContext) -> Result<()>,
        {
            let ctx = datafusion_context(config.clone());
            setup(&ctx).await.unwrap();
            let df = ctx.sql(sql).await.unwrap();
            let ordered = crate::test_utils::has_top_level_order_by(df.logical_plan());
            let plan = df.create_physical_plan().await.unwrap();
            let schema = plan.schema();
            let datafusion = collect(plan, ctx.task_ctx()).await;
            let ctx = pipeline_context(config);
            setup(&ctx).await.unwrap();
            let plan = ctx
                .sql(sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            let root = plan
                .downcast_ref::<PipelineRootExec>()
                .unwrap_or_else(|| panic!("not a pipeline plan: {sql}"));
            let operator = Arc::clone(root.operator());
            let pipeline = collect(plan, ctx.task_ctx()).await;
            Self {
                operator,
                schema,
                ordered,
                datafusion,
                pipeline,
            }
        }

        /// The lowered operator tree, rendered.
        fn tree(&self) -> String {
            format_operator_tree(&self.operator)
        }

        /// The canonical form of `batches` (see `canonical_ipc_bytes`).
        fn canonical(&self, batches: &[RecordBatch]) -> Vec<u8> {
            crate::test_utils::canonical_ipc_bytes(&self.schema, batches, self.ordered)
                .unwrap()
        }

        /// Assert that the pipeline engine returns the rows of `expected`.
        fn assert_pipeline_rows(&self, expected: &[RecordBatch]) {
            let actual = self.pipeline.as_ref().unwrap_or_else(|e| {
                panic!("The pipeline engine fails: {e}\nlowered: {}", self.tree())
            });
            let rows = |batches: &[RecordBatch]| {
                batches.iter().map(RecordBatch::num_rows).sum::<usize>()
            };
            assert!(
                self.canonical(expected) == self.canonical(actual),
                "Results differ: {} rows expected, {} rows returned\nlowered: {}",
                rows(expected),
                rows(actual),
                self.tree()
            );
        }

        /// Assert that both engines return the same rows.
        fn assert_same_rows(&self) {
            let expected = self
                .datafusion
                .as_ref()
                .unwrap_or_else(|e| panic!("DataFusion fails: {e}"));
            self.assert_pipeline_rows(expected);
        }

        /// Assert that DataFusion fails, and return its error.
        fn assert_datafusion_fails(&self) -> String {
            match &self.datafusion {
                Ok(_) => panic!("DataFusion succeeds\nlowered: {}", self.tree()),
                Err(e) => e.to_string(),
            }
        }

        /// Assert that DataFusion fails with an error that contains `message`.
        fn assert_datafusion_fails_with(&self, message: &str) {
            let error = self.assert_datafusion_fails();
            assert!(error.contains(message), "DataFusion: {error}");
        }

        /// Assert that the pipeline engine fails with an error that contains
        /// one of `messages`.
        fn assert_pipeline_fails_with(&self, messages: &[&str]) {
            match &self.pipeline {
                Ok(batches) => panic!(
                    "The pipeline engine returns {} rows, expected an error: {messages:?}\n\
                     lowered: {}",
                    batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                    self.tree()
                ),
                Err(e) => {
                    let error = e.to_string();
                    assert!(
                        messages.iter().any(|message| error.contains(message)),
                        "The pipeline engine fails with {error}, expected one of \
                         {messages:?}\nlowered: {}",
                        self.tree()
                    );
                }
            }
        }

        /// Assert that the pipeline engine returns the rows of DataFusion
        /// (which succeeds), or fails with an error that contains one of
        /// `messages`.
        fn assert_same_rows_or_fails_with(&self, messages: &[&str]) {
            match &self.pipeline {
                Ok(_) => self.assert_same_rows(),
                Err(_) => self.assert_pipeline_fails_with(messages),
            }
        }
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
    async fn redistributions_of_partial_states_are_removed() {
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
        // By an expression that cannot fail.
        let expected =
            Outcomes::of_plan(repartition(Arc::clone(&state)), test_session_config())
                .await;
        expected.assert_same_rows();
        assert_native("SELECT SUM(a) AS n FROM t", &expected.operator);
        assert_eq!(count::<PhysicalHashAggregate>(&expected.operator), 1);
        // By `10 / (sum - sum)`: DataFusion divides by zero. The native
        // aggregation, which removes the repartitioning, does not evaluate the
        // key at all, and returns the sum.
        let key: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
            datafusion_expr::Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::clone(&state),
                datafusion_expr::Operator::Minus,
                Arc::clone(&state),
            )),
        ));
        let outcomes = Outcomes::of_plan(repartition(key), test_session_config()).await;
        outcomes.assert_datafusion_fails_with("Divide by zero");
        outcomes.assert_pipeline_rows(expected.pipeline.as_ref().unwrap());
        assert_native("SELECT SUM(a) AS n FROM t", &outcomes.operator);
        assert_eq!(count::<PhysicalHashAggregate>(&outcomes.operator), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn coalescing_below_fallible_nodes_over_fallible_inputs_is_removed() {
        use crate::operators::union::test_support::{
            AfterBatch, INJECTED_FAILURE, batch_plan,
        };

        // `10 / a` over 1000 zeros, then a source error. `FilterExec` and
        // `CoalesceBatchesExec` pull the error before they emit the zeros, so
        // DataFusion raises the source error; native operators that pass the
        // zeros on at once may divide by zero first, which of these errors is
        // raised is the engine's own: the coalescing is removed.
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
        let outcomes = Outcomes::of_plan(make_plan, config.clone()).await;
        outcomes.assert_datafusion_fails_with(INJECTED_FAILURE);
        outcomes.assert_pipeline_fails_with(&[INJECTED_FAILURE, "Divide by zero"]);
        let projection = outcomes
            .operator
            .downcast_ref::<PhysicalProjection>()
            .expect("native");
        let bridge = projection.children()[0]
            .downcast_ref::<BridgeOperator>()
            .expect("a bridge");
        assert!(!is_coalesce_batches(bridge.plan()));
        // So it is without a part that may fail above it.
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
    async fn filters_below_fallible_nodes_over_fallible_sources_are_native() -> Result<()>
    {
        // A CSV file of 1000 zeros, 7192 negative ones (DataFusion's first
        // batch of 8192 rows), then a value that is not an integer. DataFusion's
        // `FilterExec` coalesces the 1000 zeros that pass `a >= 0` and reads
        // the next batch, which fails to parse, before the projection above
        // divides by zero: the source error comes first, as it does when a
        // sort below an aggregation reads all of its input first. The native
        // filter passes the zeros on at once, so that the native operators
        // above it may divide by zero first: which of several errors is raised
        // is the engine's own.
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
            let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
            let error = outcomes.assert_datafusion_fails();
            assert!(!error.contains("Divide by zero"), "{sql}: {error}");
            outcomes.assert_pipeline_fails_with(&[
                "Error while parsing value 'x'",
                "Divide by zero",
            ]);
            assert_eq!(count::<PhysicalFilter>(&outcomes.operator), 1, "{sql}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_by_fallible_keys_of_fallible_inputs_are_native() -> Result<()> {
        // A CSV file of 300000 rows, the first of which is 0, with a value that
        // is not an integer near its end. DataFusion's sort reads all of its
        // input before it evaluates its keys, so it fails to parse the file;
        // the native sort sorts (and evaluates the keys of) runs of
        // `SORT_RUN_ROWS` rows as it reads its input, and may divide by zero
        // first: which of several errors is raised is the engine's own.
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
        let outcomes = Outcomes::of_sql(sql, config, &setup).await;
        let error = outcomes.assert_datafusion_fails();
        assert!(!error.contains("Divide by zero"), "{error}");
        outcomes.assert_pipeline_fails_with(&[
            "Error while parsing value 'x'",
            "Divide by zero",
        ]);
        assert_native(sql, &outcomes.operator);
        assert_eq!(count::<PhysicalSort>(&outcomes.operator), 1);
        // Over an input that cannot fail (and `a + 100` is never 0), on any
        // number of threads.
        let sql = "SELECT a FROM t1 ORDER BY 10 / (a + 100), id";
        for target_partitions in [1, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let root = assert_same_results_with(sql, config, setup_test_tables).await;
            assert_native(sql, &root);
            assert_eq!(count::<PhysicalSort>(&root), 1);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_over_joins_whose_shared_keys_may_fail_on_their_probe_side_are_native()
    -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};
        use datafusion_expr::Operator;
        use datafusion_physical_expr::expressions::{binary, cast, lit};

        // A `CollectLeft` join on one key expression, `CAST(v@0 AS BIGINT)`,
        // that both sides share: the integers `v` of its build side, which
        // cannot fail to cast, and the strings `v` of its probe side, which
        // can: `SORT_RUN_ROWS` rows `'1'`, then `'bad_integer'`, in batches of
        // 4096 rows. A sort by `10 / d` above it, whose first `d` is 0:
        // DataFusion's sort reads all of its input before it evaluates its
        // keys, so it fails to cast the probe key; the native sort sorts (and
        // evaluates the keys of) runs of `SORT_RUN_ROWS` rows as it reads its
        // input, and may divide by zero first: which of several errors is
        // raised is the engine's own, so the sort is native. (Whether the join
        // is native is up to `PhysicalHashJoin::try_lower`.)
        let rows = crate::sort::sort_sink::SORT_RUN_ROWS + 1;
        let build_schema =
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let build = RecordBatch::try_new(
            Arc::clone(&build_schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )?;
        let probe_schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Utf8View, false),
            Field::new("d", DataType::Int32, false),
        ]));
        let mut v = vec!["1"; rows];
        v[rows - 1] = "bad_integer";
        let mut d = vec![1; rows];
        d[0] = 0;
        let probe = (0..rows)
            .step_by(4096)
            .map(|offset| {
                let end = (offset + 4096).min(rows);
                RecordBatch::try_new(
                    Arc::clone(&probe_schema),
                    vec![
                        Arc::new(arrow::array::StringViewArray::from(
                            v[offset..end].to_vec(),
                        )),
                        Arc::new(Int32Array::from(d[offset..end].to_vec())),
                    ],
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let key = cast(col("v", &build_schema)?, &build_schema, DataType::Int64)?;
        let sort = |divide: bool| {
            let (build, probe) = (build.clone(), probe.clone());
            let (build_schema, probe_schema) =
                (Arc::clone(&build_schema), Arc::clone(&probe_schema));
            let key = Arc::clone(&key);
            move || -> Arc<dyn ExecutionPlan> {
                let join: Arc<dyn ExecutionPlan> = Arc::new(
                    HashJoinExec::try_new(
                        MemorySourceConfig::try_new_exec(
                            &[vec![build.clone()]],
                            Arc::clone(&build_schema),
                            None,
                        )
                        .unwrap(),
                        MemorySourceConfig::try_new_exec(
                            std::slice::from_ref(&probe),
                            Arc::clone(&probe_schema),
                            None,
                        )
                        .unwrap(),
                        vec![(Arc::clone(&key), Arc::clone(&key))],
                        None,
                        &JoinType::Inner,
                        None,
                        PartitionMode::CollectLeft,
                        NullEquality::NullEqualsNothing,
                        false,
                    )
                    .unwrap(),
                );
                let schema = join.schema();
                let d = col("d", &schema).unwrap();
                let expr = match divide {
                    true => binary(lit(10), Operator::Divide, d, &schema).unwrap(),
                    false => d,
                };
                let ordering = LexOrdering::new([PhysicalSortExpr::new_default(expr)])
                    .expect("a sort key");
                Arc::new(SortExec::new(ordering, join))
            }
        };
        let mut config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(4096);
        config.options_mut().execution.sort_in_place_threshold_bytes = 1 << 30;
        let error = "Cannot cast string 'bad_integer'";
        let outcomes = Outcomes::of_plan(sort(true), config.clone()).await;
        outcomes.assert_datafusion_fails_with(error);
        outcomes.assert_pipeline_fails_with(&[error, "Divide by zero"]);
        assert_eq!(count::<PhysicalSort>(&outcomes.operator), 1);
        assert_join_lowered_by_the_native_join(&outcomes, &sort(true)());
        // By a key that cannot fail, both engines fail to cast.
        let outcomes = Outcomes::of_plan(sort(false), config).await;
        outcomes.assert_datafusion_fails_with(error);
        outcomes.assert_pipeline_fails_with(&[error]);
        assert_eq!(count::<PhysicalSort>(&outcomes.operator), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_with_build_keys_that_may_fail_are_left_to_the_native_join()
    -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};
        use datafusion_expr::Operator;
        use datafusion_physical_expr::expressions::{binary, cast, lit};

        // DataFusion evaluates the (two) build keys of a join once it
        // collected its build side: both keys on a build batch before the
        // next, the last batch first. A native join evaluates them on its
        // chunks of up to 2048 rows, as they arrive, so that it may raise
        // another error first, which is the engine's own: the planner leaves
        // these joins to the native join (see
        // [`assert_join_lowered_by_the_native_join`]).
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
        let outcomes = Outcomes::of_plan(&batch_dependent, config.clone()).await;
        outcomes.assert_same_rows_or_fails_with(&["Divide by zero"]);
        assert_join_lowered_by_the_native_join(&outcomes, &batch_dependent());
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
        let outcomes = Outcomes::of_plan(&two_failing_keys, config.clone()).await;
        outcomes.assert_datafusion_fails_with("Cannot cast string 'x'");
        outcomes
            .assert_pipeline_fails_with(&["Cannot cast string 'x'", "Divide by zero"]);
        assert_join_lowered_by_the_native_join(&outcomes, &two_failing_keys());

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
        let outcomes = Outcomes::of_plan(&one_failing_key, config).await;
        outcomes.assert_datafusion_fails_with("Cannot cast string 'y'");
        outcomes.assert_pipeline_fails_with(&[
            "Cannot cast string 'x'",
            "Cannot cast string 'y'",
        ]);
        assert_join_lowered_by_the_native_join(&outcomes, &one_failing_key());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn repartitions_that_may_fail_to_coalesce_are_removed() -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};

        // Two batches of 100 rows whose `Dictionary(Int8, Utf8)` values are
        // disjoint. A `RepartitionExec` over a bounded input coalesces the
        // batches of every output partition (up to `batch_size` rows), which
        // merges their dictionaries into one of 200 values: more than keys of
        // 8 bits hold. The native engine removes the repartitioning, and
        // returns the rows: a capacity limit of DataFusion that it does not
        // have.
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
        let error = "Dictionary key bigger than the key type";
        let k = col("k", &schema)?;
        let d = col("d", &schema)?;
        for partitioning in [
            Partitioning::RoundRobinBatch(1),
            Partitioning::Hash(vec![Arc::clone(&d)], 1),
        ] {
            let make_plan = || repartitioned(&partitioning);
            let outcomes = Outcomes::of_plan(make_plan, config.clone()).await;
            outcomes.assert_datafusion_fails_with(error);
            outcomes.assert_pipeline_rows(&batches);
            assert!(outcomes.operator.is::<MemoryScanSource>(), "{partitioning}");

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
            let outcomes = Outcomes::of_plan(make_plan, config.clone()).await;
            outcomes.assert_datafusion_fails_with(error);
            assert_join_lowered_by_the_native_join(&outcomes, &make_plan());
            if count::<PhysicalHashJoin>(&outcomes.operator) == 1 {
                // The row `k = 1` of each of the two batches.
                let rows = outcomes
                    .pipeline
                    .as_ref()
                    .expect("the native join succeeds");
                assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
            }
        }

        // If the keys hold the values of all batches, both engines return the
        // same rows.
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
        let outcomes = Outcomes::of_plan(make_plan, config).await;
        outcomes.assert_same_rows();
        assert!(outcomes.operator.is::<MemoryScanSource>());

        // `Utf8` columns, whose 32-bit offsets may overflow in DataFusion: the
        // aggregation of states that a repartitioning coalesces, and the
        // partitioned join whose inputs it coalesces, are native.
        let sql = "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d";
        let root = assert_same_results(sql).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
        let sql = "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k";
        let root =
            assert_same_results_with(sql, partitioned_config(), setup_test_tables).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn repartitions_below_fallible_nodes_over_fallible_inputs_are_removed() {
        use crate::operators::union::test_support::{
            AfterBatch, INJECTED_FAILURE, batch_plan,
        };

        // `10 / a` over 1000 zeros, then a source error. A `RepartitionExec`
        // over a bounded input coalesces its output, so it pulls the error
        // before it emits the zeros, and DataFusion raises the source error.
        // The native engine removes the repartitioning, and may divide by
        // zero first: which of several errors is raised is its own.
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
            let outcomes = Outcomes::of_plan(make_plan, config.clone()).await;
            outcomes.assert_datafusion_fails_with(INJECTED_FAILURE);
            outcomes.assert_pipeline_fails_with(&[INJECTED_FAILURE, "Divide by zero"]);
            let projection = outcomes
                .operator
                .downcast_ref::<PhysicalProjection>()
                .expect("native");
            let bridge = projection.children()[0]
                .downcast_ref::<BridgeOperator>()
                .expect("a bridge");
            assert!(!bridge.plan().is::<RepartitionExec>(), "{partitioning}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordering_sorts_of_ordered_aggregates_are_removed() {
        // DataFusion sorts the rows of an ordered aggregate (here `1, 'y'`
        // first) before the aggregation evaluates its arguments, so it fails
        // to cast `'y'`; the native aggregation, which orders the values
        // itself, without the sort, may fail to cast `'x'`, the first row of
        // its input, first: which of several errors is raised is its own.
        let sql = "SELECT ARRAY_AGG(CAST(column2 AS INT) ORDER BY column1 DESC) \
                   FROM (VALUES (0, 'x'), (1, 'y'))";
        let config = test_session_config().with_target_partitions(1);
        let setup = async |_: &SessionContext| Ok(());
        let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
        outcomes.assert_datafusion_fails_with("Cannot cast string 'y'");
        outcomes.assert_pipeline_fails_with(&[
            "Cannot cast string 'x'",
            "Cannot cast string 'y'",
        ]);
        assert_eq!(count::<PhysicalSort>(&outcomes.operator), 0);
        assert_eq!(count::<PhysicalHashAggregate>(&outcomes.operator), 1);
        // With arguments that cannot fail, both return the same rows.
        let sql = "SELECT ARRAY_AGG(column2 ORDER BY column1 DESC) AS x \
                   FROM (VALUES (0, 'x'), (1, 'y'))";
        let root = lowered(sql, config, &setup).await;
        assert_eq!(count::<PhysicalSort>(&root), 0);
        assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
        assert!(bridged_plans(&root).is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_with_probe_keys_that_may_fail_are_left_to_the_native_join()
    -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};
        use datafusion_physical_expr::expressions::TryCastExpr;

        // DataFusion evaluates the probe keys of a join on every probe batch,
        // also on an empty one, on which casting to a dictionary of booleans
        // fails (even with `TRY_CAST`). A native join evaluates them on the
        // rows of its chunks: none, so that it returns no rows. Whether an
        // error is raised for rows that the engine never evaluates is its own:
        // the planner leaves the join to the native join (see
        // [`assert_join_lowered_by_the_native_join`]).
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
        let error = "Unsupported output type for dictionary packing";
        let outcomes = Outcomes::of_plan(make_plan, config).await;
        outcomes.assert_datafusion_fails_with(error);
        assert_join_lowered_by_the_native_join(&outcomes, &make_plan());
        match &outcomes.pipeline {
            Ok(rows) => assert!(rows.iter().all(|batch| batch.num_rows() == 0)),
            Err(_) => outcomes.assert_pipeline_fails_with(&[error]),
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_over_nodes_that_may_fail_to_coalesce_are_native() -> Result<()> {
        // `10 / a` above a filter `a >= 0`, over a node that coalesces its
        // input: 8192 rows (1000 zeros) that it passes on, then two batches of
        // 100 rows whose `Dictionary(Int8, Utf8)` values are disjoint, which
        // it fails to merge. `FilterExec` pulls that error before it emits the
        // zeros, so DataFusion raises it. The native engine removes the node
        // (and merges no dictionaries), and its filter passes the zeros on at
        // once: the projection divides by zero.
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
        let error = "Dictionary key bigger than the key type";
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
            let outcomes = Outcomes::of_plan(make_plan, config.clone()).await;
            outcomes.assert_datafusion_fails_with(error);
            outcomes.assert_pipeline_fails_with(&["Divide by zero"]);
            assert_eq!(
                count::<PhysicalFilter>(&outcomes.operator),
                1,
                "{coalescing}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_of_values_that_address_more_than_their_memory_are_left_to_datafusion()
    -> Result<()> {
        use arrow::array::{Array, LargeListArray, NullArray, RunArray};
        use arrow::buffer::OffsetBuffer;

        // DataFusion's sort concatenates the batches of a partition while they
        // occupy fewer bytes than `sort_in_place_threshold_bytes`. Two one-row
        // batches whose `LargeList<RunEndEncoded(Int32, Null)>` lists hold
        // `1 << 30` nulls each occupy a few bytes, but need more run ends than
        // 32 bits hold together: DataFusion fails to concatenate them. A native
        // sort would materialize them as it gathers its rows (the lists take
        // no memory): the sort is left to DataFusion (see
        // `gathers_values_beyond_their_memory`).
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
        let outcomes = Outcomes::of_plan(make_plan, config).await;
        let error = outcomes.assert_datafusion_fails();
        outcomes.assert_pipeline_fails_with(&[&error]);
        assert!(
            outcomes.operator.is::<BridgeOperator>(),
            "{}",
            outcomes.tree()
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_of_values_that_address_more_than_their_memory_are_left_to_datafusion()
    -> Result<()> {
        use arrow::array::{Array, LargeListArray, NullArray, RunArray};
        use arrow::buffer::OffsetBuffer;
        use datafusion_common::{JoinType, NullEquality};

        // DataFusion concatenates all columns of the build side of a join,
        // also those that its output does not have: two one-row build batches
        // whose `LargeList<RunEndEncoded(Int32, Null)>` lists hold `1 << 30`
        // nulls each need more run ends than 32 bits hold together. A native
        // join would materialize them as it gathers its rows (the lists take
        // no memory): the join is left to DataFusion (see
        // `gathers_values_beyond_their_memory`).
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
        let outcomes = Outcomes::of_plan(make_plan, config).await;
        let error = outcomes.assert_datafusion_fails();
        outcomes.assert_pipeline_fails_with(&[&error]);
        assert_eq!(count::<PhysicalHashJoin>(&outcomes.operator), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_output_coalescing_may_differ_are_native() -> Result<()> {
        use datafusion_common::{JoinType, NullEquality};

        // DataFusion's hash join coalesces its output, up to `batch_size`
        // rows, before it emits it. A native join passes its output on at
        // once: whether a concatenation overflows, and which of several errors
        // is raised, is the engine's own. The planner leaves these joins to the
        // native join (see [`assert_join_lowered_by_the_native_join`]).
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
        let outcomes = Outcomes::of_plan(make_plan, config.clone()).await;
        outcomes.assert_datafusion_fails_with("Dictionary key bigger than the key type");
        assert_join_lowered_by_the_native_join(&outcomes, &make_plan());
        if count::<PhysicalHashJoin>(&outcomes.operator) == 1 {
            let rows = outcomes
                .pipeline
                .as_ref()
                .expect("the native join succeeds");
            assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 200);
        }

        // `10 / k` above the join of a CSV file, on one thread: its first batch
        // of 8192 rows holds 1000 zeros (that join) and 7192 `-1`s, its next
        // batch a value that does not parse. The join pulls the parse error
        // before it emits the zeros, so DataFusion raises it; a native join
        // passes the zeros on at once, and the projection may divide by zero
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
        let outcomes = Outcomes::of_plan(|| make_plan(true), config.clone()).await;
        outcomes.assert_pipeline_fails_with(&[&error, "Divide by zero"]);
        assert_join_lowered_by_the_native_join(&outcomes, &make_plan(true));
        assert_eq!(count::<PhysicalProjection>(&outcomes.operator), 1);
        // Without a part that may fail above it, both fail to parse.
        let outcomes = Outcomes::of_plan(|| make_plan(false), config).await;
        outcomes.assert_pipeline_fails_with(&[&error]);
        assert_join_lowered_by_the_native_join(&outcomes, &make_plan(false));
        Ok(())
    }

    /// Assert the behaviour of the plans that `make_plan` makes in `ctx` (a
    /// new one for every execution), lowered with the options of `config`: an
    /// operator, which `native` counts, has a dynamic filter that prunes the
    /// reads of a scan of Parquet files, one of which fails to be read or
    /// decoded, and which DataFusion's filter skips. DataFusion succeeds. The
    /// operator is native. The pipeline engine produces the same rows, or, if
    /// its own filter does not skip that file, fails with an error that
    /// contains `error`: whether an error is raised for rows that DataFusion
    /// never reads is the engine's own (see `docs/compatibility.md`).
    async fn assert_read_errors_allowed(
        ctx: &SessionContext,
        config: &SessionConfig,
        make_plan: impl AsyncFn() -> Result<Arc<dyn ExecutionPlan>>,
        native: fn(&Arc<dyn PhysicalOperator>) -> usize,
        error: &str,
    ) -> Result<()> {
        let plan = make_plan().await?;
        let expected = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .expect("DataFusion skips the file that fails to be read");
        let root = create_pipeline_plan_with_runtime(
            make_plan().await?,
            config.options(),
            &ctx.runtime_env(),
        )?;
        let tree = format_operator_tree(root.operator());
        assert_eq!(native(root.operator()), 1, "{tree}");
        match collect(root as Arc<dyn ExecutionPlan>, ctx.task_ctx()).await {
            Ok(actual) => {
                let output = |batches| crate::test_utils::EngineOutput {
                    schema: plan.schema(),
                    plan: Arc::clone(&plan),
                    batches,
                    ordered: true,
                };
                compare_outputs(&output(expected), &output(actual)).unwrap_or_else(
                    |message| panic!("Results differ: {message}\n{tree}"),
                );
            }
            Err(e) => assert!(e.to_string().contains(error), "{e}\n{tree}"),
        }
        Ok(())
    }

    /// [`assert_read_errors_allowed`] of the plans of `sql`, in a session
    /// with `config` of the tables that `setup` registers, whose scans of files
    /// `transform` changes, if any (e.g. into scans with other reader
    /// factories).
    async fn assert_read_errors_of_query_allowed<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        transform: Option<&dyn Fn(&FileScanConfig) -> FileScanConfig>,
        native: fn(&Arc<dyn PhysicalOperator>) -> usize,
        error: &str,
    ) -> Result<()>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = datafusion_context(config.clone());
        setup(&ctx).await?;
        let make_plan = async || -> Result<Arc<dyn ExecutionPlan>> {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let Some(transform) = transform else {
                return Ok(plan);
            };
            let plan = plan.transform_up(|node| {
                let Some(config) = node
                    .downcast_ref::<DataSourceExec>()
                    .and_then(|scan| scan.data_source().downcast_ref::<FileScanConfig>())
                else {
                    return Ok(Transformed::no(node));
                };
                Ok(Transformed::yes(
                    DataSourceExec::from_data_source(transform(config))
                        as Arc<dyn ExecutionPlan>,
                ))
            })?;
            Ok(plan.data)
        };
        assert_read_errors_allowed(&ctx, &config, make_plan, native, error).await
    }

    /// A scan of files with the explicit schema `schema` for every file
    /// (`PartitionedFile::arrow_schema`), in which the reader reads them.
    fn with_explicit_schemas(
        config: &FileScanConfig,
        schema: &SchemaRef,
    ) -> FileScanConfig {
        use datafusion_datasource::file_groups::FileGroup;
        use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
        let file_groups = config
            .file_groups
            .iter()
            .map(|group| {
                FileGroup::new(
                    group
                        .iter()
                        .map(|file| file.clone().with_arrow_schema(Arc::clone(schema)))
                        .collect(),
                )
            })
            .collect();
        FileScanConfigBuilder::from(config.clone())
            .with_file_groups(file_groups)
            .build()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_dynamic_filter_prunes_reads_that_may_fail_are_native()
    -> Result<()> {
        // A table `k INT, v INT` of two Parquet files: 100 rows `k = 1` of
        // integers, and rows `k = 0` of strings that fail to cast to `INT`
        // when they are read. DataFusion pushes the dynamic filter of the join
        // (`k = 1`, from its build side) into the scan, which prunes the
        // second file: it never reads it. A native join does not prune the same
        // reads, so that it may fail to cast them: whether an error is raised
        // for rows that DataFusion never reads is the engine's own, the join is
        // native.
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
        assert_read_errors_of_query_allowed(
            sql,
            config,
            setup,
            None,
            count::<PhysicalHashJoin>,
            "Cannot cast string 'bad_integer' to value of Int32 type",
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_dynamic_filter_prunes_reads_that_may_fail_are_native()
    -> Result<()> {
        // A table `k INT, v INT, s VARCHAR` (its schema given when it is
        // registered) of two Parquet files: 100 rows `k = 0, v = 100`, and 10
        // rows `k = 1` of strings `v` that fail to cast to `INT` when they are
        // read. The TopK of `ORDER BY k LIMIT 1` reads the first file,
        // whose rows tighten its threshold to `k <= 0`, and DataFusion's scan
        // prunes the second file with its dynamic filter: it never reads it. A
        // native TopK does not prune the same reads, so that it may fail to
        // cast them, which is the engine's own (see `docs/compatibility.md`):
        // the TopK is native, also below the filter `s LIKE '%foo%'` (on
        // batches of 64
        // rows, the first of which tightens the threshold of DataFusion's
        // TopK).
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
            ctx.register_parquet("p", path, ParquetReadOptions::default().schema(&schema))
                .await
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        for sql in [
            "SELECT k, v FROM p ORDER BY k LIMIT 1",
            "SELECT k, v FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1",
        ] {
            assert_read_errors_of_query_allowed(
                sql,
                config.clone(),
                &setup,
                None,
                count::<PhysicalTopK>,
                "Cannot cast string 'bad_integer' to value of Int32 type",
            )
            .await?;
        }
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
    async fn topks_over_int96_files_that_the_scan_casts_are_native() -> Result<()> {
        // A table `k INT, v TIMESTAMP (ns), s VARCHAR` (its schema given when it
        // is registered) of two Parquet files of INT96 timestamps: 100 rows `k =
        // 0` at the epoch, and 10 rows `k = 1` on the Julian day 3,000,000 (in
        // the year 3501). The session reads INT96 timestamps in microseconds
        // (`coerce_int96`), from which the scan casts them into the
        // nanoseconds of the table: the timestamps of the second file overflow.
        // The TopK of `ORDER BY k LIMIT 1` reads the first file, whose first
        // batch of 64 rows tightens its threshold to `k < 0`, and DataFusion's
        // scan prunes the second file. A native TopK may decode it, which is
        // the engine's own: the TopK is native.
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
            assert_read_errors_of_query_allowed(
                sql,
                config.clone(),
                &setup,
                None,
                count::<PhysicalTopK>,
                "converted value exceeds the representable i64 range",
            )
            .await?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_over_files_whose_explicit_schemas_constrain_their_values_are_native()
    -> Result<()> {
        // A table `k INT, v INT[2], s VARCHAR` of two Parquet files of lists of
        // integers, whose explicit schemas (`PartitionedFile::arrow_schema`) are
        // the schema of the table: 100 rows `k = 0` of lists of 2 integers, and
        // 10 rows `k = 1` of lists of 3. The reader reads the lists as lists of
        // exactly 2 integers, which it checks only as it decodes them: it fails
        // to read the second file, which DataFusion's scan prunes with the
        // filter of the TopK. A native TopK may read it, which is the engine's
        // own: the TopK is native.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("p");
        let lists = |k: i32, length: usize, rows: usize| -> Result<RecordBatch> {
            let list =
                arrow::array::ListArray::from_iter_primitive::<
                    arrow::datatypes::Int32Type,
                    _,
                    _,
                >((0..rows).map(|_| Some((0..length as i32).map(Some))));
            Ok(RecordBatch::try_from_iter([
                (
                    "k",
                    Arc::new(Int32Array::from(vec![k; rows])) as arrow::array::ArrayRef,
                ),
                ("v", Arc::new(list) as _),
                (
                    "s",
                    Arc::new(arrow::array::StringViewArray::from(vec!["foo"; rows])) as _,
                ),
            ])?)
        };
        for (file, batch) in [lists(0, 2, 100)?, lists(1, 3, 10)?]
            .into_iter()
            .enumerate()
        {
            write_parquet(
                &path.join(format!("{file}.parquet")),
                batch.schema(),
                &[batch],
                None,
            )?;
        }
        let path = path.to_str().unwrap();
        let item = Arc::new(Field::new("item", DataType::Int32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::FixedSizeList(item, 2), false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", path, ParquetReadOptions::default().schema(&schema))
                .await
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        assert_read_errors_of_query_allowed(
            "SELECT k, v FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1",
            config,
            setup,
            Some(&|config| with_explicit_schemas(config, &schema)),
            count::<PhysicalTopK>,
            "Encountered misaligned row with length 3 (expected length 2)",
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_over_files_with_explicit_schemas_that_the_scan_casts_are_native()
    -> Result<()> {
        // The files of `topks_over_int96_files_that_the_scan_casts_are_native`
        // (INT96 timestamps at the epoch, and on the Julian day 3,000,000), of
        // a table of timestamps in nanoseconds, without `coerce_int96`, but
        // with an explicit schema for every file (`PartitionedFile::arrow_schema`)
        // of timestamps in microseconds: the reader reads the files in that
        // schema, and the scan casts their timestamps into nanoseconds, which
        // overflow for the second file. DataFusion's scan prunes it with the
        // filter of the TopK. A native TopK may read it, which is the engine's
        // own: the TopK is native.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("p");
        write_int96_parquet(&path.join("0.parquet"), (0, 2_440_588, "foo"), 100)?;
        write_int96_parquet(&path.join("1.parquet"), (1, 3_000_000, "foo"), 10)?;
        let path = path.to_str().unwrap();
        let timestamps = |unit| {
            Schema::new(vec![
                Field::new("k", DataType::Int32, false),
                Field::new("v", DataType::Timestamp(unit, None), false),
                Field::new("s", DataType::Utf8View, false),
            ])
        };
        let schema = timestamps(arrow::datatypes::TimeUnit::Nanosecond);
        let reader_schema = Arc::new(timestamps(arrow::datatypes::TimeUnit::Microsecond));
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", path, ParquetReadOptions::default().schema(&schema))
                .await
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        assert_read_errors_of_query_allowed(
            "SELECT k, v FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1",
            config,
            setup,
            Some(&|config| with_explicit_schemas(config, &reader_schema)),
            count::<PhysicalTopK>,
            "converted value exceeds the representable i64 range",
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_over_files_whose_explicit_schemas_do_not_fit_are_native() -> Result<()>
    {
        // A table `k INT, v INT, s VARCHAR` of two Parquet files, whose
        // explicit schemas (`PartitionedFile::arrow_schema`) are the schema of
        // the table: 100 rows `(0, 100, 'foo')`, and 10 rows `(1, 'bad_integer',
        // 'foo')`, whose file stores `v` as strings, which its explicit schema
        // does not fit. The reader fails to read that file, which DataFusion's
        // scan prunes with the filter of the TopK. A native TopK may read it,
        // which is the engine's own: the TopK is native.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("p");
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let first = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![0; 100])),
                Arc::new(Int32Array::from(vec![100; 100])),
                Arc::new(arrow::array::StringViewArray::from(vec!["foo"; 100])),
            ],
        )?;
        let strings = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Utf8View, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let second = RecordBatch::try_new(
            Arc::clone(&strings),
            vec![
                Arc::new(Int32Array::from(vec![1; 10])),
                Arc::new(arrow::array::StringViewArray::from(vec!["bad_integer"; 10])),
                Arc::new(arrow::array::StringViewArray::from(vec!["foo"; 10])),
            ],
        )?;
        write_parquet(&path.join("0.parquet"), first.schema(), &[first], None)?;
        write_parquet(&path.join("1.parquet"), second.schema(), &[second], None)?;
        let path = path.to_str().unwrap();
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", path, ParquetReadOptions::default().schema(&schema))
                .await
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        assert_read_errors_of_query_allowed(
            "SELECT k, v FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1",
            config,
            setup,
            Some(&|config| with_explicit_schemas(config, &schema)),
            count::<PhysicalTopK>,
            "Incompatible supplied Arrow schema",
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_over_files_of_narrow_dictionaries_are_native() -> Result<()> {
        // A table `k INT, v Dictionary(Int8, VARCHAR), s VARCHAR` of two Parquet
        // files (of the types of the table): 100 rows `k = 0`, and 200 rows `k =
        // 1` written as two batches of 100 different values each, in one row
        // group, whose dictionary of 200 values does not fit `Int8` keys. The
        // reader fails to read the second file (only `v IS NOT NULL` reads `v`),
        // which DataFusion's scan prunes with the filter of the TopK. A native
        // TopK may read it, which is the engine's own: the TopK is native.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("p");
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new(
                "v",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let batch = |k: i32, values: Vec<Option<String>>| -> Result<RecordBatch> {
            let rows = values.len();
            let v = values
                .iter()
                .map(Option::as_deref)
                .collect::<arrow::array::DictionaryArray<arrow::datatypes::Int8Type>>();
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(vec![k; rows])),
                    Arc::new(v),
                    Arc::new(arrow::array::StringViewArray::from(vec!["foo"; rows])),
                ],
            )?)
        };
        let first = batch(
            0,
            (0..100)
                .map(|i| (i % 7 != 0).then(|| format!("a{}", i % 10)))
                .collect(),
        )?;
        let wide = |prefix: &str| -> Vec<Option<String>> {
            (0..100)
                .map(|i| (i % 9 != 0).then(|| format!("{prefix}{i}")))
                .collect()
        };
        let second = [batch(1, wide("x"))?, batch(1, wide("y"))?];
        write_parquet(&path.join("0.parquet"), Arc::clone(&schema), &[first], None)?;
        write_parquet(&path.join("1.parquet"), Arc::clone(&schema), &second, None)?;
        let path = path.to_str().unwrap();
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", path, ParquetReadOptions::default())
                .await
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        assert_read_errors_of_query_allowed(
            "SELECT k FROM p WHERE v IS NOT NULL AND s LIKE '%foo%' ORDER BY k LIMIT 1",
            config,
            setup,
            None,
            count::<PhysicalTopK>,
            "dictionary too large for index type",
        )
        .await
    }

    /// A reader factory whose readers fail to read the metadata of the files
    /// whose paths end with `faulty`, and otherwise read files like `inner`.
    #[derive(Debug)]
    struct ReaderFactoryWithFaults {
        inner: Arc<dyn ParquetFileReaderFactory>,
        faulty: &'static str,
    }

    /// A reader of a file whose metadata it fails to read, see
    /// [`ReaderFactoryWithFaults`].
    struct FaultyReader(Box<dyn parquet::arrow::async_reader::AsyncFileReader + Send>);

    impl parquet::arrow::async_reader::AsyncFileReader for FaultyReader {
        fn get_bytes(
            &mut self,
            range: std::ops::Range<u64>,
        ) -> futures::future::BoxFuture<'_, parquet::errors::Result<bytes::Bytes>>
        {
            self.0.get_bytes(range)
        }

        fn get_metadata<'a>(
            &'a mut self,
            _options: Option<&'a parquet::arrow::arrow_reader::ArrowReaderOptions>,
        ) -> futures::future::BoxFuture<
            'a,
            parquet::errors::Result<Arc<parquet::file::metadata::ParquetMetaData>>,
        > {
            Box::pin(async {
                Err(parquet::errors::ParquetError::General(
                    "the metadata of the file cannot be read".to_string(),
                ))
            })
        }
    }

    impl ParquetFileReaderFactory for ReaderFactoryWithFaults {
        fn create_reader(
            &self,
            partition_index: usize,
            partitioned_file: datafusion_datasource::PartitionedFile,
            metadata_size_hint: Option<usize>,
            metrics: &datafusion_physical_plan::metrics::ExecutionPlanMetricsSet,
        ) -> Result<Box<dyn parquet::arrow::async_reader::AsyncFileReader + Send>>
        {
            let faulty = partitioned_file
                .object_meta
                .location
                .as_ref()
                .ends_with(self.faulty);
            let reader = self.inner.create_reader(
                partition_index,
                partitioned_file,
                metadata_size_hint,
                metrics,
            )?;
            Ok(match faulty {
                true => Box::new(FaultyReader(reader)),
                false => reader,
            })
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_over_files_of_custom_readers_are_native() -> Result<()> {
        // Two well-formed Parquet files of the types of their table, of 100 rows
        // `(0, 'foo')` and 10 rows `(1, 'foo')`, which the scan reads with a
        // custom reader factory, whose readers fail to read the metadata of the
        // second file. DataFusion's scan prunes the second file with the filter
        // of the TopK, without reading its metadata. A native TopK may read it,
        // which is the engine's own: the planner does not tell custom readers
        // from DataFusion's (it has no way to), and the TopK is native.
        let dir = tempfile::tempdir()?;
        let path = two_parquet_files(dir.path(), (0, "foo", 100), (1, "foo", 10))?;
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await
        };
        let with_faults = |config: &FileScanConfig| {
            let source = config
                .file_source()
                .downcast_ref::<ParquetSource>()
                .expect("a scan of Parquet files");
            let inner = Arc::clone(
                source
                    .parquet_file_reader_factory()
                    .expect("the reader factory of the scan"),
            );
            let source = source.clone().with_parquet_file_reader_factory(Arc::new(
                ReaderFactoryWithFaults {
                    inner,
                    faulty: "1.parquet",
                },
            ));
            datafusion_datasource::file_scan_config::FileScanConfigBuilder::from(
                config.clone(),
            )
            .with_source(Arc::new(source))
            .build()
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        assert_read_errors_of_query_allowed(
            "SELECT k, s FROM p WHERE s LIKE '%foo%' ORDER BY k LIMIT 1",
            config,
            setup,
            Some(&with_faults),
            count::<PhysicalTopK>,
            "the metadata of the file cannot be read",
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_dynamic_filter_prunes_casts_that_may_fail_are_native()
    -> Result<()> {
        // A table `k INT, s VARCHAR` of two Parquet files: 100 rows `k = 1` of
        // integers, and 10 rows `k = 0` of strings that fail to cast to `INT`.
        // DataFusion pushes the dynamic filter of the join (`k = 1`, from its
        // build side) into the scan, which prunes the second file: it never
        // casts its strings. A native join does not prune the same reads, so
        // that it may fail to cast, as the shape of TPC-H Q15 and Q17 over
        // Parquet (decimal arithmetic above the scan, which may overflow): the
        // join is native, and returns the rows of DataFusion, or fails to cast.
        // (Until `PhysicalHashJoin::try_lower` lowers such a join, the planner
        // lowers it without its dynamic filter, see
        // `Lowering::try_lower_hash_join`.)
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
        for target_partitions in [1, 4] {
            let config = config.clone().with_target_partitions(target_partitions);
            let outcomes = Outcomes::of_sql(sql, config, &setup).await;
            outcomes
                .assert_same_rows_or_fails_with(&["Cannot cast string 'bad_integer'"]);
            assert_native(sql, &outcomes.operator);
            assert_eq!(count::<PhysicalHashJoin>(&outcomes.operator), 1);
        }
        // Without the cast, both return the same rows.
        let sql = "SELECT b.x, COUNT(p.s) AS n FROM b JOIN p ON b.x = p.k GROUP BY b.x";
        let root = assert_same_results_with(sql, config, &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_dynamic_filter_prunes_casts_that_may_fail_are_native()
    -> Result<()> {
        // A table `k INT, s VARCHAR` of two Parquet files: 100 rows `k = 0, s =
        // '100'`, and 10 rows `k = 1` of strings that fail to cast to `INT`. The
        // TopK of `ORDER BY k LIMIT 1` reads the first file, whose rows tighten
        // its threshold to `k < 0`, and DataFusion's scan prunes the second
        // file with its dynamic filter: it never casts its strings (which the
        // projection of the scan does). A native TopK tightens its threshold at
        // other times, so that its scan may read the second file, and fail to
        // cast: whether an error is raised for rows that DataFusion never
        // reads is the engine's own. The TopK is native, and updates the
        // filter.
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
        let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
        outcomes.assert_same_rows_or_fails_with(&["Cannot cast string 'bad_integer'"]);
        assert_native(sql, &outcomes.operator);
        assert_eq!(count::<PhysicalTopK>(&outcomes.operator), 1);
        let (plan, _) = pipeline_plan(sql, config.clone(), &setup).await;
        filter_of_native_topk(&plan);
        // Without the cast, both return the same rows.
        let sql = "SELECT k, s FROM p ORDER BY k LIMIT 1";
        let root = assert_same_results_with(sql, config.clone(), &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalTopK>(&root), 1);
        // Over a scan of memory, which cannot fail, the TopK is native.
        let sql = "SELECT id, a FROM t1 ORDER BY a, id LIMIT 3";
        let root = assert_same_results_with(sql, config, &setup).await;
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
        let [filter] = sorts.as_slice() else {
            panic!("no TopK whose filter is read: {sorts:?}");
        };
        let topk = std::iter::successors(Some(Arc::clone(root)), |operator| {
            operator.children().first().map(Arc::clone)
        })
        .find(|operator| operator.is::<PhysicalTopK>())
        .unwrap_or_else(|| panic!("no native TopK: {}", format_operator_tree(root)));
        let topk = topk.downcast_ref::<PhysicalTopK>().unwrap();
        let updated = topk.dynamic_filter().expect("a dynamic filter").filter();
        assert_eq!(updated.expression_id(), filter.expression_id());
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
                        let root =
                            assert_same_results_with(sql, config.clone(), &setup).await;
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
    async fn topks_whose_filter_is_read_below_one_of_several_inputs_are_native()
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
        // more rows, after the scan read all of its files, so that it may divide
        // by zero: whether an error is raised for rows that DataFusion never
        // reads is the engine's own. The TopK is native, and updates the
        // filter.
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
            let outcomes = Outcomes::of_sql(sql, config.clone(), &setup).await;
            outcomes.assert_same_rows_or_fails_with(&["Divide by zero"]);
            assert_eq!(
                count::<PhysicalTopK>(&outcomes.operator),
                1,
                "{}",
                outcomes.tree()
            );
            let (plan, _) = pipeline_plan(sql, config, &setup).await;
            filter_of_native_topk(&plan);
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
    async fn topks_whose_filter_is_read_below_nodes_that_emit_rows_first_are_native()
    -> Result<()> {
        // A TopK of `ORDER BY k LIMIT 1` over a node that emits the row `(-1,
        // 7)` before it reads the groups of an aggregation of 40 Parquet files
        // of `k >= 0`, with `v = 0` on a row of the last file. DataFusion pushes
        // the dynamic filter of the TopK through the node and the aggregation
        // into the scan, and tightens it to `k < -1` with the first row of the
        // node, before the aggregation reads any file: its scan skips all of
        // them, and never divides by zero. The aggregation reads all of its
        // input before it emits any group, but the node above it does not wait
        // for it. The node (of the user, as an `ExecutionPlan`) is executed by
        // DataFusion, with the aggregation below it, and the TopK above it is
        // native: it updates the filter, with the row that the node emits
        // first, and returns the rows of DataFusion (or divides by zero, if it
        // tightened the filter later, which would be the engine's own).
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
        let bridged = bridged_plans(root.operator());
        match collect(root as Arc<dyn ExecutionPlan>, task_ctx).await {
            Ok(actual) => {
                let output = |batches| crate::test_utils::EngineOutput {
                    schema: topk.schema(),
                    plan: Arc::clone(&topk),
                    batches,
                    ordered: true,
                };
                compare_outputs(&output(expected), &output(actual)).unwrap_or_else(
                    |message| panic!("Results differ: {message}\n{tree}"),
                );
            }
            Err(e) => assert!(e.to_string().contains("Divide by zero"), "{e}\n{tree}"),
        }
        assert_eq!(topks, 1, "{tree}");
        assert_eq!(
            bridged.iter().map(|plan| plan.name()).collect::<Vec<_>>(),
            ["EmitsRowFirst"],
            "{tree}"
        );
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
        let root = assert_same_results_with(sql, config.clone(), &setup).await;
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
                let root = assert_same_results_with(sql, config.clone(), &setup).await;
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
        // Also through the probe side of a join, which is no node that only
        // passes rows on (as `crate::sort::try_lower` judges it): the native
        // TopK updates the filter too.
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
        let root = assert_same_results_with(sql, config.clone(), &setup).await;
        assert_native(sql, &root);
        assert_eq!(count::<PhysicalTopK>(&root), 1, "{sql}");
        let (plan, _) = pipeline_plan(sql, config, &setup).await;
        filter_of_native_topk(&plan);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fallible_operators_on_several_threads_are_native() -> Result<()> {
        // One batch of 8192 rows: `CAST(s AS INT)` fails on row 0, a string of a
        // million zeros and `x` (slow to parse), and on row 2048, `y`.
        // DataFusion evaluates the rows of a batch in order, so it fails on row
        // 0. On several threads, the native operators evaluate the chunks of
        // the batch in parallel, so that the thread of row 2048 may fail first:
        // which of several errors is raised is the engine's own. On one thread,
        // they evaluate the rows in order, as DataFusion.
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
            let outcomes = Outcomes::of_sql(sql, test_session_config(), &setup).await;
            outcomes.assert_datafusion_fails_with(error);
            outcomes.assert_pipeline_fails_with(&[error, "Cannot cast string 'y'"]);
            assert_native(sql, &outcomes.operator);
            assert_eq!(casts(&outcomes.operator, operator), 1, "{sql}");
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
            // DataFusion evaluates the hash expressions on every batch (and
            // divides by zero), the native engine does not evaluate them.
            (
                Arc::new(RepartitionExec::try_new(
                    scan()?,
                    Partitioning::Hash(vec![fallible], 4),
                )?),
                true,
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
        // so the native aggregation does without both stages, also with
        // strings with 32-bit offsets (the group key `d`), which
        // `CoalesceBatchesExec` may fail to concatenate.
        for sql in [
            "SELECT e, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh FROM t1 GROUP BY e",
            "SELECT d, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh FROM t1 GROUP BY d",
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
            assert_native(sql, &root);
            assert_eq!(count::<PhysicalHashAggregate>(&root), 1);

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
            let root = assert_same_results_with(sql, config.clone(), &setup).await;
            assert_native(sql, &root);
            let sources =
                count::<ParquetScanSource>(&root) + count::<CsvScanSource>(&root);
            assert_eq!(sources, 1, "{sql}");
            assert!(format_operator_tree(&root).contains(source), "{sql}");
        }

        // DataFusion pushes the dynamic filter of a hash join into the Parquet
        // scan of its probe side: the join is native, with or without it.
        let sql =
            "SELECT t2.name, COUNT(*) AS n FROM p JOIN t2 ON p.a = t2.k GROUP BY t2.name";
        let mut without_filter = config.clone();
        without_filter
            .options_mut()
            .optimizer
            .enable_join_dynamic_filter_pushdown = false;
        for config in [config, without_filter] {
            let root = assert_same_results_with(sql, config, &setup).await;
            assert_native(sql, &root);
            assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Scalar subqueries
    // ---------------------------------------------------------------------

    /// A source of one partition that produces `batches`, and then stays
    /// pending forever, as a stream that waits for rows that never come.
    #[derive(Debug)]
    struct PendsAfterRows {
        batches: Vec<RecordBatch>,
        properties: Arc<datafusion_physical_plan::PlanProperties>,
    }

    impl PendsAfterRows {
        fn new(batches: Vec<RecordBatch>) -> Self {
            use datafusion_physical_expr::EquivalenceProperties;
            use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
            let properties = datafusion_physical_plan::PlanProperties::new(
                EquivalenceProperties::new(batches[0].schema()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Unbounded {
                    requires_infinite_memory: false,
                },
            );
            Self {
                batches,
                properties: Arc::new(properties),
            }
        }
    }

    impl datafusion_physical_plan::DisplayAs for PendsAfterRows {
        fn fmt_as(
            &self,
            _t: datafusion_physical_plan::DisplayFormatType,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(f, "PendsAfterRows")
        }
    }

    impl ExecutionPlan for PendsAfterRows {
        fn name(&self) -> &str {
            "PendsAfterRows"
        }

        fn properties(&self) -> &Arc<datafusion_physical_plan::PlanProperties> {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<datafusion_execution::SendableRecordBatchStream> {
            use futures::StreamExt;
            let batches = self.batches.clone().into_iter().map(Ok);
            let stream = futures::stream::iter(batches).chain(futures::stream::pending());
            Ok(Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                    self.batches[0].schema(),
                    stream,
                ),
            ))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limits_end_with_the_rows_of_inputs_that_then_stay_pending() -> Result<()> {
        // 128 positive rows, and then nothing more: DataFusion's limits (and
        // filters with a fetch) end as soon as they have their row, and so do
        // the native ones, whose pipelines do not cache the rows of the
        // filters below them (a cache would hold them back until it is full or
        // its input ends, see `Lowering::lower_in`).
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let rows = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(1..=128))],
        )?;
        // A filter of `x > 0` over `input`, whose batches have at most
        // `batch_size` rows.
        let filter = |input: Arc<dyn ExecutionPlan>, batch_size: usize| -> FilterExec {
            let positive = Arc::new(BinaryExpr::new(
                col("x", &schema).unwrap(),
                datafusion_expr::Operator::Gt,
                Arc::new(Literal::new(ScalarValue::Int32(Some(0)))),
            ));
            FilterExec::try_new(positive, input)
                .unwrap()
                .with_batch_size(batch_size)
                .unwrap()
        };
        let source = || -> Arc<dyn ExecutionPlan> {
            Arc::new(PendsAfterRows::new(vec![rows.clone()]))
        };
        // The plan of `WHERE x > 0 LIMIT 1` over several partitions: a merge
        // with a fetch above a filter with the same fetch (which the native
        // limit makes redundant), with the default batch size.
        let merged = || -> Arc<dyn ExecutionPlan> {
            let filter = filter(source(), 8192).with_fetch(Some(1)).unwrap();
            Arc::new(CoalescePartitionsExec::new(filter).with_fetch(Some(1)))
        };
        // A global limit above a filter without a fetch, with batches of 32
        // rows, which DataFusion's filter emits once it has 32 rows.
        let limited = || -> Arc<dyn ExecutionPlan> {
            Arc::new(GlobalLimitExec::new(
                Arc::new(filter(source(), 32)),
                0,
                Some(1),
            ))
        };
        // A filter with a fetch, which stops once it has its row, as a limit
        // does, above a filter without a fetch, with batches of 32 rows: the
        // native filter with the fetch is cached (its cache is emitted once
        // it stops), the one below it is not.
        let filtered = || -> Arc<dyn ExecutionPlan> {
            let inner = Arc::new(filter(source(), 32));
            filter(inner, 8192).with_fetch(Some(1)).unwrap()
        };
        // The shapes, with their numbers of native filters and of caches.
        type Shape<'a> = &'a dyn Fn() -> Arc<dyn ExecutionPlan>;
        let shapes: [(Shape, usize, usize); 3] =
            [(&merged, 1, 0), (&limited, 1, 0), (&filtered, 2, 1)];
        for (plan, filters, caches) in shapes {
            let config = test_session_config();
            let root = create_pipeline_plan(plan())?;
            let name = plan().name().to_owned();
            let operator = root.operator();
            assert_eq!(bridged_plans(operator).len(), 1, "{name}: only the source");
            let count = (
                crate::operators::union::test_support::count::<PhysicalFilter>(operator),
                crate::operators::union::test_support::count::<CachingOperator>(operator),
            );
            assert_eq!(count, (filters, caches), "{name}");
            for (engine, plan) in [("DataFusion", plan()), ("pipeline", root as _)] {
                let task_ctx =
                    Arc::new(TaskContext::default().with_session_config(config.clone()));
                let batches = tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    collect(plan, task_ctx),
                )
                .await
                .unwrap_or_else(|_| panic!("{engine} does not end {name}"))?;
                let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
                assert_eq!(rows, 1, "{engine}: {name}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn soft_limits_end_with_the_rows_of_inputs_that_then_stay_pending() -> Result<()>
    {
        // `SELECT DISTINCT x ... LIMIT 3` over 128 positive rows, and then
        // nothing more: DataFusion's aggregation stops once it has the 3
        // groups of its soft limit (its filter emits batches of 32 rows), and
        // so does the native one, whose input does not cache the rows of its
        // filter (see `Lowering::lower_in`).
        let config = test_session_config()
            .with_batch_size(32)
            .with_target_partitions(1);
        let ctx = datafusion_context(config.clone());
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let rows = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(1..=128))],
        )?;
        ctx.register_batch("t", rows.clone())?;
        let sql = "SELECT DISTINCT x FROM t WHERE x > 0 LIMIT 3";
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        // The scan, replaced by a stream that stays pending after its rows.
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::clone(&plan)
                .transform_up(|node| {
                    Ok(match node.children().is_empty() {
                        true => Transformed::yes(Arc::new(PendsAfterRows::new(vec![
                            rows.clone(),
                        ])) as _),
                        false => Transformed::no(node),
                    })
                })
                .unwrap()
                .data
        };
        let root = create_pipeline_plan(plan())?;
        let operator = root.operator();
        let tree = format_operator_tree(operator);
        assert_eq!(bridged_plans(operator).len(), 1, "only the source: {tree}");
        let count = (
            crate::operators::union::test_support::count::<PhysicalHashAggregate>(
                operator,
            ),
            crate::operators::union::test_support::count::<PhysicalFilter>(operator),
            crate::operators::union::test_support::count::<CachingOperator>(operator),
        );
        assert_eq!(count, (1, 1, 0), "{tree}");
        for (engine, plan) in [("DataFusion", plan()), ("pipeline", root as _)] {
            let task_ctx =
                Arc::new(TaskContext::default().with_session_config(config.clone()));
            let batches = tokio::time::timeout(
                std::time::Duration::from_secs(60),
                collect(plan, task_ctx),
            )
            .await
            .unwrap_or_else(|_| panic!("{engine} does not end {sql}"))?;
            let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
            assert_eq!(rows, 3, "{engine}: {sql}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_pipelines_that_stop_early_are_not_cached() -> Result<()> {
        // The pipelines of the filters below limits (but below pipeline
        // breakers, which read all of their input, and on the build sides of
        // joins) and those of the subqueries of scalar subqueries, which fail
        // at their second row, may stop before their input ends: their
        // filters are not cached.
        for (sql, filters, cached) in [
            ("SELECT id FROM t1 WHERE a > 0", 1, 1),
            ("SELECT id FROM t1 WHERE a > 0 LIMIT 3", 1, 0),
            ("SELECT id FROM t1 WHERE a > 0 LIMIT 3 OFFSET 2", 1, 0),
            (
                "SELECT a, COUNT(*) FROM t1 WHERE b > 0 GROUP BY a LIMIT 3",
                1,
                1,
            ),
            (
                "SELECT t1.id FROM t1 JOIN (SELECT k FROM t2 WHERE w > 0) ON a = k \
                 WHERE b > 0 LIMIT 3",
                2,
                1,
            ),
            (
                "SELECT id, (SELECT k FROM t2 WHERE w > 0) AS m FROM t1 WHERE id < 10",
                2,
                1,
            ),
        ] {
            let plan = datafusion_plan(sql, test_session_config()).await;
            let root = lower_execution_plan(&plan)?;
            assert_native(sql, &root);
            let count = |operator| match operator {
                "filters" => {
                    crate::operators::union::test_support::count::<PhysicalFilter>(&root)
                }
                _ => {
                    crate::operators::union::test_support::count::<CachingOperator>(&root)
                }
            };
            assert_eq!(
                (count("filters"), count("caches")),
                (filters, cached),
                "{sql}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scalar_subqueries_are_native() -> Result<()> {
        // A `ScalarSubqueryExec` is lowered to a `PhysicalScalarSubquery`,
        // which runs its subqueries (natively too) before the pipelines of its
        // input, whose expressions read their results: constants, e.g. in the
        // arithmetic of a projection, or in a filter that casts the sums of
        // its groups into the type of the result (TPC-H Q11).
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
            let root = assert_same_results(sql).await;
            assert_native(sql, &root);
            assert_eq!(count::<PhysicalScalarSubquery>(&root), subqueries, "{sql}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_of_scalar_subqueries_are_those_of_datafusion() -> Result<()> {
        // DataFusion runs the subqueries before the input, so that their errors
        // come first, and fails on a subquery of more than one row, as do the
        // native scalar subqueries (whose result fails once its second row
        // arrives, which stops the pipelines of the subquery).
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
            let root = lowered(sql, test_session_config(), setup_test_tables).await;
            assert_native(sql, &root);
            assert_eq!(count::<PhysicalScalarSubquery>(&root), 1, "{sql}");
        }

        // A filter that casts a column of 4 partitions into the type of the
        // result of a subquery, which only fails on the row whose value
        // overflows the decimal: by more than its integer holds (a cast
        // error), or by more than its precision (an invalid argument). The
        // native filter fails on it too, on 4 threads.
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

        // A bridged plan is shown below its bridge: a recursive query, which
        // has no native operator yet.
        let explained = explain(
            "EXPLAIN WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r \
             WHERE n < 3) SELECT n * 2 AS m FROM r",
        )
        .await;
        let plan = plan_of(&explained, "physical_plan");
        assert!(
            plan.starts_with("PipelineRootExec: pipelines=1, bridged_plans=1\n"),
            "{plan}"
        );
        assert!(
            plan.contains("\n    BridgeOperator, pipelines=[0]\n      RepartitionExec: "),
            "{plan}"
        );
        assert!(
            plan.contains("\n        RecursiveQueryExec: name=r"),
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
        // A native sort of a bridged plan (a recursive query, which has no
        // native operator yet), whose nodes have metrics too.
        let query = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r \
                     WHERE n < 3) SELECT n FROM r ORDER BY n DESC";
        let analyzed = async |options: &str| {
            let explained = explain(&format!("EXPLAIN ({options}) {query}")).await;
            let plan = plan_of(&explained, "Plan with Metrics").to_string();
            assert!(plan.contains("PhysicalSort"), "{plan}");
            assert!(plan.contains("RecursiveQueryExec"), "{plan}");
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
            for node in ["PhysicalSort", "RecursiveQueryExec"] {
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

#[cfg(test)]
mod join_filters {
    //! The routing of the dynamic filters of native hash joins to the scans
    //! below their probe sides (see `Lowering::try_lower_hash_join`).

    use std::path::Path;
    use std::sync::Arc;

    use super::*;
    use crate::bridge::bridged_plans;
    use crate::explain::format_operator_tree;
    use crate::sources::test_harness::write_parquet;
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, run_engines,
        test_session_config,
    };
    use arrow::array::RecordBatch;
    use arrow::compute::concat_batches;
    use arrow::datatypes::SchemaRef;
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use datafusion_common::Result;

    /// The SQL that generates the tables of the TPC-H queries of the tests,
    /// and the column that the rows of a table are sorted by in its Parquet
    /// file: 25 nations of 5 regions, 1,000 suppliers (40 of every nation),
    /// 2,000 customers (80 of every nation), 500 parts (50 of them green), 4
    /// suppliers of every part (of 4 nations), 6,000 orders of 1992 to 1998, 4
    /// lines of every order (of the suppliers of their parts).
    const TABLES: &[(&str, &str, &str)] = &[
        (
            "region",
            "SELECT * FROM (VALUES (0, 'AFRICA'), (1, 'AMERICA'), (2, 'ASIA'), \
             (3, 'EUROPE'), (4, 'MIDDLE EAST')) AS r(r_regionkey, r_name)",
            "r_regionkey",
        ),
        (
            "nation",
            "SELECT * FROM (VALUES (0, 'ALGERIA', 0), (1, 'ARGENTINA', 1), \
             (2, 'BRAZIL', 1), (3, 'CANADA', 1), (4, 'EGYPT', 4), (5, 'ETHIOPIA', 0), \
             (6, 'FRANCE', 3), (7, 'GERMANY', 3), (8, 'INDIA', 2), (9, 'INDONESIA', 2), \
             (10, 'IRAN', 4), (11, 'IRAQ', 4), (12, 'JAPAN', 2), (13, 'JORDAN', 4), \
             (14, 'KENYA', 0), (15, 'MOROCCO', 0), (16, 'MOZAMBIQUE', 0), (17, 'PERU', 1), \
             (18, 'CHINA', 2), (19, 'ROMANIA', 3), (20, 'SAUDI ARABIA', 4), \
             (21, 'VIETNAM', 2), (22, 'RUSSIA', 3), (23, 'UNITED KINGDOM', 3), \
             (24, 'UNITED STATES', 1)) AS n(n_nationkey, n_name, n_regionkey)",
            "n_regionkey",
        ),
        (
            "supplier",
            "SELECT value AS s_suppkey, 'Supplier#' || CAST(value AS VARCHAR) AS s_name, \
             (value - 1) / 40 AS s_nationkey FROM generate_series(1, 1000)",
            "s_nationkey",
        ),
        (
            "customer",
            "SELECT value AS c_custkey, 'Customer#' || CAST(value AS VARCHAR) AS c_name, \
             (value - 1) / 80 AS c_nationkey FROM generate_series(1, 2000)",
            "c_nationkey",
        ),
        (
            "part",
            "SELECT value AS p_partkey, CASE WHEN value % 10 = 0 THEN 'forest green ' \
             ELSE 'navy chiffon ' END || CAST(value AS VARCHAR) AS p_name \
             FROM generate_series(1, 500)",
            "p_partkey",
        ),
        (
            "partsupp",
            "SELECT p.value AS ps_partkey, (p.value + s.value * 251) % 1000 + 1 AS ps_suppkey, \
             CAST((p.value * 7 + s.value) % 100 + 1 AS DECIMAL(15, 2)) AS ps_supplycost \
             FROM generate_series(1, 500) AS p CROSS JOIN generate_series(0, 3) AS s",
            "ps_suppkey",
        ),
        (
            "orders",
            "SELECT value AS o_orderkey, value * 7 % 2000 + 1 AS o_custkey, \
             arrow_cast(CAST(8035 + value * 13 % 2405 AS INT), 'Date32') AS o_orderdate, \
             CASE value % 3 WHEN 0 THEN 'F' WHEN 1 THEN 'O' ELSE 'P' END AS o_orderstatus \
             FROM generate_series(1, 6000)",
            "o_orderkey",
        ),
        (
            "lineitem",
            "WITH lines AS (SELECT o.value AS o, l.value AS l, \
               (o.value * 13 + l.value * 7) % 500 + 1 AS partkey, \
               8035 + o.value * 13 % 2405 AS orderdate \
               FROM generate_series(1, 6000) AS o CROSS JOIN generate_series(1, 4) AS l) \
             SELECT o AS l_orderkey, partkey AS l_partkey, \
             (partkey + (o + l) % 4 * 251) % 1000 + 1 AS l_suppkey, \
             CAST(1 + (o + l) % 50 AS DECIMAL(15, 2)) AS l_quantity, \
             CAST(100 + (o * 7 + l) % 900 AS DECIMAL(15, 2)) AS l_extendedprice, \
             CAST((o + l) % 11 / 100.0 AS DECIMAL(15, 2)) AS l_discount, \
             arrow_cast(CAST(orderdate + 1 + (o + l * 7) % 121 AS INT), 'Date32') \
               AS l_shipdate, \
             arrow_cast(CAST(orderdate + 30 + (o * 3 + l) % 61 AS INT), 'Date32') \
               AS l_commitdate, \
             arrow_cast(CAST(orderdate + 2 + (o + l * 7) % 121 + (o + l) % 30 AS INT), \
               'Date32') AS l_receiptdate \
             FROM lines",
            "l_suppkey",
        ),
    ];

    /// A generated table: its name, schema, rows, and the column that its
    /// Parquet file is sorted by.
    struct Table {
        name: &'static str,
        schema: SchemaRef,
        batch: RecordBatch,
        sort_column: &'static str,
    }

    /// The tables of [`TABLES`].
    async fn tables() -> Result<Vec<Table>> {
        let ctx = datafusion_context(test_session_config());
        let mut tables = vec![];
        for &(name, sql, sort_column) in TABLES {
            let df = ctx.sql(sql).await?;
            let schema = Arc::clone(df.schema().inner());
            let batch = concat_batches(&schema, &df.collect().await?)?;
            tables.push(Table {
                name,
                schema,
                batch,
                sort_column,
            });
        }
        Ok(tables)
    }

    /// Register `tables` in `ctx` as memory tables of 3 partitions.
    fn register_memtables(ctx: &SessionContext, tables: &[Table]) -> Result<()> {
        for table in tables {
            let rows = table.batch.num_rows();
            let partitions = vec![
                vec![table.batch.slice(0, rows / 2)],
                vec![table.batch.slice(rows / 2, rows / 8)],
                vec![
                    table
                        .batch
                        .slice(rows / 2 + rows / 8, rows - rows / 2 - rows / 8),
                ],
            ];
            let memtable = MemTable::try_new(Arc::clone(&table.schema), partitions)?;
            ctx.register_table(table.name, Arc::new(memtable))?;
        }
        Ok(())
    }

    /// Write `tables` to Parquet files in `dir`, one per table, sorted by
    /// their sort columns, in row groups of `row_group_rows` rows.
    async fn write_parquet_tables(
        dir: &Path,
        tables: &[Table],
        row_group_rows: usize,
    ) -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        for table in tables {
            ctx.register_batch(table.name, table.batch.clone())?;
            let sql = format!(
                "SELECT * FROM {} ORDER BY {}",
                table.name, table.sort_column
            );
            let sorted = ctx.sql(&sql).await?.collect().await?;
            let properties = ::parquet::file::properties::WriterProperties::builder()
                .set_max_row_group_row_count(Some(row_group_rows))
                .build();
            let path = dir.join(table.name).join("part-0.parquet");
            write_parquet(&path, Arc::clone(&table.schema), &sorted, Some(properties))?;
        }
        Ok(())
    }

    /// Register the Parquet tables of `dir` (see [`write_parquet_tables`]).
    async fn register_parquet_tables(ctx: &SessionContext, dir: &Path) -> Result<()> {
        for &(name, _, _) in TABLES {
            let path = dir.join(name);
            ctx.register_parquet(
                name,
                path.to_str().expect("a path"),
                ParquetReadOptions::default(),
            )
            .await?;
        }
        Ok(())
    }

    /// The TPC-H query `q{number}.sql` of `benchmarks/queries`.
    fn query(number: usize) -> String {
        let path = format!(
            "{}/../../benchmarks/queries/q{number}.sql",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    /// Run `sql` on both engines (with `config`, over the tables of `setup`),
    /// and assert that their results are the same.
    async fn assert_same_results<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> (EngineOutput, Arc<dyn PhysicalOperator>)
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let (expected, actual) = run_engines(sql, config, setup)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let root = Arc::clone(
            actual
                .plan
                .downcast_ref::<PipelineRootExec>()
                .expect("a pipeline plan")
                .operator(),
        );
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("{sql}: {message}\n{}", format_operator_tree(&root));
        }
        assert!(bridged_plans(&root).is_empty(), "{sql}");
        (actual, root)
    }

    /// A native scan of an execution of the pipeline engine.
    #[derive(Debug)]
    struct Scan {
        /// The names of its columns.
        columns: Vec<String>,
        /// The rows it produced.
        rows: usize,
        /// The dynamic filters of joins that it filters its chunks with (if it
        /// is a [`JoinFilteredScan`]).
        filtered_with: Vec<String>,
        /// The dynamic filters that it reads in its predicate (of a Parquet
        /// scan) that the planner routed to it: not those of the plan.
        routed: Vec<String>,
        /// The row groups that it pruned by their statistics, of all row
        /// groups (of a Parquet scan).
        pruned_row_groups: (usize, usize),
    }

    impl Scan {
        fn has_column(&self, column: &str) -> bool {
            self.columns.iter().any(|name| name == column)
        }
    }

    /// The dynamic filters that the expressions `exprs` read, by expression
    /// id.
    fn dynamic_filters_in<'a>(
        exprs: impl IntoIterator<Item = &'a Arc<dyn PhysicalExpr>>,
    ) -> Vec<(u64, String)> {
        let mut filters = vec![];
        for expr in exprs {
            let _ = expr.apply(|expr| {
                if expr.is::<DynamicFilterPhysicalExpr>()
                    && let Some(id) = expr.expression_id()
                {
                    filters.push((id, expr.to_string()));
                }
                Ok(TreeNodeRecursion::Continue)
            });
        }
        filters
    }

    /// The native scans of `output`, an execution of the pipeline engine (in
    /// the order of the operator tree).
    fn scans(output: &EngineOutput) -> Vec<Scan> {
        let exec = output
            .plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        let metrics = exec.execution_metrics().expect("an execution");
        // The dynamic filters that the scans of the plan read.
        let mut planned = vec![];
        let _ = exec.input().apply(|node| {
            if node.is::<DataSourceExec>() {
                let _ = node.apply_expressions(&mut |expr| {
                    planned
                        .extend(dynamic_filters_in([expr]).into_iter().map(|(id, _)| id));
                    Ok(TreeNodeRecursion::Continue)
                });
            }
            Ok(TreeNodeRecursion::Continue)
        });
        let mut scans = vec![];
        let mut operators = vec![Arc::clone(exec.operator())];
        while let Some(operator) = operators.pop() {
            operators.extend(operator.children().iter().rev().cloned());
            if !operator.is_source() || !operator.children().is_empty() {
                continue;
            }
            let rows = metrics
                .get(operator.as_ref())
                .map_or(0, |metrics| metrics.output_rows());
            let filtered_with = operator
                .downcast_ref::<JoinFilteredScan>()
                .map(|scan| scan.filters.iter().map(ToString::to_string).collect())
                .unwrap_or_default();
            let (routed, pruned_row_groups) =
                match operator.downcast_ref::<crate::sources::ParquetScanSource>() {
                    Some(scan) => {
                        use datafusion_datasource::source::DataSource;
                        use datafusion_physical_plan::metrics::MetricValue;
                        let predicate = scan.config().file_source().filter();
                        let routed = dynamic_filters_in(predicate.as_ref())
                            .into_iter()
                            .filter(|(id, _)| !planned.contains(id))
                            .map(|(_, filter)| filter)
                            .collect();
                        let pruned = match scan
                            .config()
                            .metrics()
                            .clone_inner()
                            .sum_by_name("row_groups_pruned_statistics")
                        {
                            Some(MetricValue::PruningMetrics {
                                pruning_metrics, ..
                            }) => (
                                pruning_metrics.pruned(),
                                pruning_metrics.pruned() + pruning_metrics.matched(),
                            ),
                            _ => (0, 0),
                        };
                        (routed, pruned)
                    }
                    None => (vec![], (0, 0)),
                };
            scans.push(Scan {
                columns: operator
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect(),
                rows,
                filtered_with,
                routed,
                pruned_row_groups,
            });
        }
        scans
    }

    /// The scan of `scans` with the column `column`.
    fn scan_of<'a>(scans: &'a [Scan], column: &str, sql: &str) -> &'a Scan {
        let mut found = scans.iter().filter(|scan| scan.has_column(column));
        match (found.next(), found.next()) {
            (Some(scan), None) => scan,
            _ => panic!("not a single scan of {column} in {scans:#?}\n{sql}"),
        }
    }

    /// TPC-H Q9 of the profits of the suppliers of one nation.
    fn q9_of_germany() -> String {
        let sql = query(9);
        let restricted = sql.replace(
            "p_name like '%green%'",
            "p_name like '%green%' and n_name = 'GERMANY'",
        );
        assert_ne!(sql, restricted);
        restricted
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn deep_scans_of_tpch_join_chains_skip_the_rows_that_cannot_match() -> Result<()>
    {
        // TPC-H Q5, Q7, Q9 (of the suppliers of one nation) and Q21 over memory
        // tables, whose plans join the scans of the dimensions with the nations
        // of the queries several joins below the joins with these nations: the
        // filters of the joins with the nations (and with the suppliers of
        // these nations) are routed to the scans of these dimensions, which
        // produce the rows of the nations of the queries only, and the results
        // are those of DataFusion (which pushes no dynamic filters into memory
        // tables):
        //
        // * Q5 (`r_name = 'ASIA'`, of the nations 8, 9, 12, 18 and 21): the
        //   join with the region filters the scan of `nation` (on the probe
        //   side of the join with `supplier`); its join with the nations
        //   filters the scan of `supplier`, two joins below, and, through the
        //   key `s_nationkey = c_nationkey` of the join of `supplier` with the
        //   join of `customer`, `orders` and `lineitem`, the scan of
        //   `customer`, three joins below: of 5 of the 25 nations, of their 200
        //   suppliers of 1,000, of their 400 customers of 2,000;
        // * Q7 (`FRANCE` and `GERMANY`): the scan of `supplier`, on the build
        //   side of its join with `lineitem`, on the build side of the join with
        //   `orders`, on the build side of the join with `customer`, on the
        //   build side of the join with the nations: their 80 suppliers;
        // * Q9 of `GERMANY`: the scan of `supplier`, on the probe side of its
        //   join with `part` and `lineitem`, three joins below: its 40
        //   suppliers;
        // * Q21 (`SAUDI ARABIA`): the scan of `supplier` on the build side of
        //   its join with `lineitem`, on the build side of the join with
        //   `orders`, on the probe side of the join with the nation: its 40
        //   suppliers.
        let tables = tables().await?;
        let setup = async |ctx: &SessionContext| register_memtables(ctx, &tables);
        for (sql, expected) in [
            (
                query(5),
                &[("n_nationkey", 5), ("s_suppkey", 200), ("c_custkey", 400)][..],
            ),
            (query(7), &[("s_suppkey", 80)][..]),
            (q9_of_germany(), &[("s_suppkey", 40)][..]),
            (query(21), &[("s_suppkey", 40)][..]),
        ] {
            let (actual, _) =
                assert_same_results(&sql, test_session_config(), setup).await;
            assert!(!actual.batches.is_empty(), "{sql}");
            let scans = scans(&actual);
            for &(column, rows) in expected {
                let scan = scan_of(&scans, column, &sql);
                assert_eq!(scan.filtered_with.len(), 1, "{scan:?}\n{sql}");
                assert_eq!(scan.rows, rows, "{scan:?}\n{sql}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn deep_scans_of_tpch_join_chains_prune_their_row_groups() -> Result<()> {
        // The same queries over Parquet files sorted by the nations (and
        // `lineitem` by its suppliers), in row groups of 40 rows, so that the
        // scans prune the row groups of the other nations and suppliers by
        // their statistics.
        //
        // * Q5, with `pushdown_filters`: the scan of `nation` filters its rows
        //   with the filter of the join with the region, so that the join of
        //   the nations filters `supplier` (20 of its 25 row groups pruned) and
        //   `customer` (40 of 50, through the key of the join of `supplier`).
        //   The keys of the join of `supplier` with the join of `customer`,
        //   `orders` and `lineitem`, `l_suppkey = s_suppkey AND c_nationkey =
        //   s_nationkey`, are columns of two inputs of the joins below:
        //   DataFusion pushes its filter nowhere, while the planner routes the
        //   filter of `l_suppkey` to the scan of `lineitem` (the bounds of the
        //   200 suppliers of the nations of Asia, 321 to 880, so that it prunes
        //   the 265 row groups of the other suppliers of 600), and the filter
        //   of `c_nationkey` to the scan of `customer`;
        // * Q7, Q9 of `GERMANY` and Q21: the scan of `supplier` prunes the row
        //   groups of the other nations, and the scan of `lineitem` those of
        //   the other suppliers, with the filters that DataFusion pushes into
        //   them (through the joins below the joins with the nations), which the
        //   native joins set, and which the planner does not route again. In
        //   Q9, the filter of the keys `ps_suppkey = l_suppkey AND ps_partkey =
        //   l_partkey` of the join of `partsupp`, which DataFusion pushes into
        //   `lineitem`, is routed to `supplier` and `part` too, by one key
        //   each, through the keys `s_suppkey = l_suppkey` and `p_partkey =
        //   l_partkey` of their joins with `lineitem` (it prunes nothing:
        //   `partsupp` has all suppliers and parts).
        let tables = tables().await?;
        let dir = tempfile::tempdir()?;
        write_parquet_tables(dir.path(), &tables, 40).await?;
        let setup =
            async |ctx: &SessionContext| register_parquet_tables(ctx, dir.path()).await;
        for (sql, pushdown, pruned, routed) in [
            (
                query(5),
                true,
                &[
                    ("s_suppkey", 20, 25),
                    ("c_custkey", 40, 50),
                    ("l_orderkey", 265, 600),
                ][..],
                &[
                    ("c_custkey", "c_nationkey@2 >= 8 AND c_nationkey@2 <= 21"),
                    ("l_orderkey", "l_suppkey@2 >= 321 AND l_suppkey@2 <= 880"),
                ][..],
            ),
            (
                query(7),
                false,
                &[("s_suppkey", 23, 25), ("l_orderkey", 550, 600)][..],
                &[][..],
            ),
            (
                q9_of_germany(),
                false,
                &[("s_suppkey", 24, 25), ("l_orderkey", 575, 600)][..],
                &[
                    ("s_suppkey", "s_suppkey@0 >= 1 AND s_suppkey@0 <= 1000"),
                    ("p_partkey", "p_partkey@0 >= 1 AND p_partkey@0 <= 500"),
                ][..],
            ),
            (query(21), false, &[("s_name", 24, 25)][..], &[][..]),
        ] {
            let mut config = test_session_config();
            config.options_mut().execution.parquet.pushdown_filters = pushdown;
            let (actual, _) = assert_same_results(&sql, config, setup).await;
            let scans = scans(&actual);
            for &(column, pruned, row_groups) in pruned {
                let scan = scan_of(&scans, column, &sql);
                assert_eq!(
                    scan.pruned_row_groups,
                    (pruned, row_groups),
                    "{scan:?}\n{sql}"
                );
            }
            let mut expected = routed
                .iter()
                .map(|(column, filter)| {
                    let scan = scan_of(&scans, column, &sql);
                    (scan.columns.clone(), format!("DynamicFilter [ {filter} ]"))
                })
                .collect::<Vec<_>>();
            let mut routed = scans
                .iter()
                .flat_map(|scan| {
                    scan.routed
                        .iter()
                        .map(|filter| (scan.columns.clone(), filter.clone()))
                })
                .collect::<Vec<_>>();
            expected.sort();
            routed.sort();
            assert_eq!(routed, expected, "{sql}");
        }
        Ok(())
    }
}
