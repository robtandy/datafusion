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
//! | DataFusion plan                                                             | Native operator                                    |
//! | --------------------------------------------------------------------------- | -------------------------------------------------- |
//! | `SortPreservingMergeExec -> SortExec`, `SortExec`                           | [`PhysicalSort`], or [`PhysicalTopK`] with a fetch |
//! | `AggregateExec(Final \| FinalPartitioned) -> ... -> AggregateExec(Partial)` | [`PhysicalHashAggregate`]                          |
//! | `AggregateExec(Single \| SinglePartitioned)`                                | [`PhysicalHashAggregate`]                          |
//! | `HashJoinExec` (`CollectLeft` or `Partitioned`)                             | [`PhysicalHashJoin`]                               |
//! | `GlobalLimitExec`, `LocalLimitExec`, `CoalescePartitionsExec` with a fetch  | [`PhysicalLimit`]                                  |
//! | `UnionExec`, `InterleaveExec`                                               | [`PhysicalUnion`]                                  |
//! | `PlaceholderRowExec`                                                        | [`PhysicalPlaceholderRow`]                         |
//! | `FilterExec`                                                                | [`PhysicalFilter`] in a [`CachingOperator`]        |
//! | `ProjectionExec`                                                            | [`PhysicalProjection`]                             |
//! | `DataSourceExec` (memory, Parquet, CSV), `EmptyExec`                        | the sources of [`try_create_scan_source`]          |
//! | `RepartitionExec`, `CoalescePartitionsExec`, `CoalesceBatchesExec`          | removed                                            |
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
//! repartitioning of its inputs), and a sort with the merge of its sorted
//! partitions a single [`PhysicalSort`].
//!
//! Every native operator documents (in its `try_lower` or `supports_*`
//! function) which nodes it reproduces exactly, including their errors; any
//! other node, such as a window aggregation, a nested loop or cross join, a
//! sort-merge join, a recursive query or a custom node, is executed by
//! DataFusion: it is wrapped, with its whole subtree, in a [`BridgeOperator`],
//! so that every plan can be executed. Native operators above a bridged plan
//! read its output as the source of their pipeline.
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
//!   batch unless at most 20% of them are left, or with more than one part
//!   that may fail (see [`failing_parts`]): DataFusion evaluates every
//!   expression of an operator on a whole batch before the next one, the
//!   native operators on their chunks. (The native filters and projections
//!   refuse such expressions themselves.)
//! * On several threads (a `target_partitions` above 1, or lowering without
//!   a session), a filter, projection, aggregation or sort with a part that
//!   may fail: the threads of a pipeline evaluate the chunks of a partition of
//!   DataFusion, also of a single batch, in parallel, so which of the rows on
//!   which it fails fails first depends on them, while DataFusion fails on
//!   the first of them in its partition (and casts and overflows name the
//!   value that fails).
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
//!   a dynamic filter whose probe side may fail (e.g. a scan of files):
//!   DataFusion's scans skip the reads that the filter rejects. So do they
//!   for the dynamic filter of a TopK: a TopK whose dynamic filter a node of
//!   its input consumes is left to DataFusion if its input may fail.
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
//!
//! Which other nodes a native operator reproduces exactly, including their
//! errors, is up to the operator (see its `try_lower` or `supports_*`
//! function).
//!
//! The pipeline breakers that read all of their input before they produce any
//! output (sorts, aggregations and the build side of hash joins) separate
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
//! [`PhysicalSort`]: crate::sort::PhysicalSort
//! [`PhysicalTopK`]: crate::sort::PhysicalTopK

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use async_trait::async_trait;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::format::ExplainFormat;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::LogicalPlan;
use datafusion_expr::logical_plan::Explain;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateInputMode, AggregateMode,
};
use datafusion_physical_plan::analyze::AnalyzeExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::explain::ExplainExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PhysicalExpr, ReplaceChildrenOptions,
};
use datafusion_session::{PhysicalPlanner, QueryPlanner, Session};

use crate::aggregate::PhysicalHashAggregate;
use crate::bridge::{BridgeOperator, PipelineRootExec};
use crate::explain::{AnalyzedPipelineExec, MetricSelection};
use crate::join::PhysicalHashJoin;
use crate::operator::PhysicalOperator;
use crate::operators::adaptive_filter::{depends_on_batches, failing_parts};
use crate::operators::caching::CachingOperator;
use crate::operators::filter::PhysicalFilter;
use crate::operators::limit::PhysicalLimit;
use crate::operators::placeholder_row::PhysicalPlaceholderRow;
use crate::operators::projection::PhysicalProjection;
use crate::operators::union::PhysicalUnion;
use crate::sources::try_create_scan_source;

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
        let root = create_pipeline_plan_with_options(input, options)?;
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
        let root = create_pipeline_plan_with_options(
            Arc::clone(analyze.input()),
            session.config_options(),
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
        let plan: Arc<dyn ExecutionPlan> =
            create_pipeline_plan_with_options(plan, session.config_options())?;
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
    let root = lower_execution_plan_with_options(&plan, options)?;
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
/// more than 2 GiB of them, see [`crate::sort::try_lower`]): this lowers them
/// as if the session had the options for which the fewest nodes can be
/// lowered. The planner lowers with the options of its session, see
/// [`lower_execution_plan_with_options`].
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
pub fn lower_execution_plan_with_options(
    plan: &Arc<dyn ExecutionPlan>,
    options: &ConfigOptions,
) -> Result<Arc<dyn PhysicalOperator>> {
    Lowering { options }.lower(plan, Context::default())
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
    /// A node above the node, before the next pipeline breaker that reads all
    /// of its input, has a part that may fail (see [`failing_parts`]).
    fallible_above: bool,
}

impl Context {
    /// The context of the input of a pipeline breaker that reads all of its
    /// input before it produces any output, and whose own expressions have
    /// `failing_parts` parts that may fail.
    fn below_breaker(failing_parts: usize) -> Self {
        Self {
            limited: false,
            fallible_above: failing_parts > 0,
        }
    }
}

/// Lowers the nodes of a plan, see [`lower_execution_plan_with_options`].
struct Lowering<'a> {
    options: &'a ConfigOptions,
}

impl Lowering<'_> {
    /// Whether the native operators evaluate the rows of every partition of
    /// DataFusion in order, as DataFusion does: if the pipelines run on a
    /// single thread. The executor runs a pipeline on up to
    /// `target_partitions` threads, which evaluate the chunks of a partition,
    /// and of a batch, in parallel: then which of the rows on which an
    /// expression fails fails first depends on the threads, while DataFusion
    /// fails on the first of them in its partition. So a native operator with
    /// a part that may fail (see [`failing_parts`]) is only lowered if the
    /// pipelines run on a single thread.
    fn evaluates_in_order(&self) -> bool {
        self.options.execution.target_partitions <= 1
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
        // A native sort reads all of its input, also below a limit.
        if let Some(sort) = self.try_lower_sort(plan)? {
            return Ok(sort);
        }
        if ctx.limited {
            return Ok(bridge(plan));
        }
        if let Some(aggregate) = self.try_lower_aggregate(plan)? {
            return Ok(aggregate);
        }
        if let Some(join) = self.try_lower_hash_join(plan, ctx)? {
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
                && ctx.fallible_above
                && subtree_failing_parts(input) > 0
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

    /// Lower a sort (a `SortPreservingMergeExec -> SortExec` chain, or a
    /// `SortExec`) with [`crate::sort::try_lower`].
    ///
    /// DataFusion evaluates the keys of a sort on its batches, the native sort
    /// on its chunks: a sort whose keys have errors that depend on the batches
    /// they are evaluated on (see [`depends_on_batches`]), e.g. the right side
    /// of `a <> 0 AND 10 / a > 1`, which DataFusion evaluates on all rows of a
    /// batch unless at most 20% of them are left, or more than one part that
    /// may fail (which error comes first depends on the batches) is left to
    /// DataFusion. So is a sort with a key that may fail over an input that
    /// may fail (its expressions, native or not, or its sources): DataFusion
    /// evaluates the keys once it read all of its input, the native sort
    /// whenever it sorts a run of its input, before it read all of it.
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
        let input_may_fail = || subtree_failing_parts(sort.input()) > 0;
        if keys.failing_parts > 0 && (input_may_fail() || !self.evaluates_in_order()) {
            return Ok(None);
        }
        // DataFusion pushes the dynamic filter of a TopK (its threshold) into
        // the scans of its input, which skip the files, row groups and rows
        // that it rejects as it tightens; a native TopK publishes its threshold
        // at other times, so that it does not skip the same reads. So a TopK
        // whose dynamic filter a node of its input consumes is left to
        // DataFusion if its input may fail (e.g. a scan of files, whose values
        // may fail to read or to cast to the types of the table). (The native
        // sort itself leaves to DataFusion the TopKs whose pruned rows the
        // expressions of their input may fail on.)
        if sort.fetch().is_some()
            && let Some(filter) = sort
                .dynamic_expressions_produced()
                .iter()
                .find_map(|expr| expr.expression_id())
            && consumes_dynamic_filter(sort.input(), filter)
            && input_may_fail()
        {
            return Ok(None);
        }
        let input = Context::below_breaker(keys.failing_parts);
        let Some(lowered) =
            crate::sort::try_lower(plan, self.options, |plan| self.lower(plan, input))?
        else {
            return Ok(None);
        };
        // The keys that the native sort evaluates (those of the top merge, over
        // the projections below it, if any).
        let native_keys = native_sort_keys(&lowered);
        if !native_keys.evaluates_like_datafusion()
            || (native_keys.failing_parts > 0
                && (input_may_fail() || !self.evaluates_in_order()))
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
    /// errors depend on the batches (see [`depends_on_batches`]) or with more
    /// than one part that may fail is left to DataFusion, as is an aggregation
    /// with a part that may fail above native operators of the same pipeline
    /// that may fail too, and an aggregation that produces a dynamic filter
    /// (which the native aggregation does not update) above nodes that may fail
    /// on the rows that DataFusion would prune with it.
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
            && subtree_failing_parts(raw.input()) > 0
        {
            return Ok(None);
        }
        let own = judgement.failing_parts;
        if own > 0 && !self.evaluates_in_order() {
            return Ok(None);
        }
        let input = Context::below_breaker(own);
        let Some(aggregate) = PhysicalHashAggregate::try_lower(plan, |plan| {
            self.lower(without_ordering_sort(raw, own, plan), input)
        })?
        else {
            return Ok(None);
        };
        if own > 0 && native_failing_parts(aggregate.input()) > 0 {
            return Ok(None);
        }
        Ok(Some(Arc::new(aggregate)))
    }

    /// Lower a hash join with [`PhysicalHashJoin::try_lower`].
    ///
    /// The join reads all of its build side before it probes, as DataFusion
    /// does: its build side is lowered like the input of a pipeline breaker.
    /// Its probe side continues the pipeline of the nodes above the join (the
    /// probe keys and the filter of a native join cannot fail).
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
        if build_keys.batch_dependent || build_keys.failing_parts > 0 {
            return Ok(None);
        }
        // DataFusion evaluates the probe keys on every probe batch, also on an
        // empty one, and the filter on the candidate pairs of a probe batch, a
        // native join on the rows of its chunks: probe keys and a filter that
        // may fail (also without rows, e.g. a cast that their types do not
        // support), or whose errors depend on the batches, are left to
        // DataFusion. (The native join refuses some of them itself.)
        let probe_schema = join.right().schema();
        let mut probe = Judgement::default();
        for (_, key) in join.on() {
            probe.add(key, &probe_schema);
        }
        if let Some(filter) = join.filter() {
            probe.add(filter.expression(), filter.schema());
        }
        if probe.batch_dependent || probe.failing_parts > 0 {
            return Ok(None);
        }
        // DataFusion's join coalesces its output, up to `batch_size` rows,
        // before it emits it; a native join passes its output on at once. So
        // DataFusion may fail to concatenate output types whose concatenation
        // may fail, and it pulls later probe batches, whose errors it raises,
        // before the nodes above it see the rows of the earlier ones: such
        // joins are left to DataFusion, as are those whose probe side (which
        // continues the pipeline of the nodes above the join) may fail below a
        // node above that may fail.
        if concatenation_may_fail(&join.schema())
            || (ctx.fallible_above && subtree_failing_parts(join.right()) > 0)
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
        // side may fail (e.g. a scan of files, whose values may fail to read or
        // to cast to the types of the table) is left to DataFusion.
        if !plan.dynamic_expressions_produced().is_empty()
            && subtree_failing_parts(join.right()) > 0
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
        let build = Context::below_breaker(0);
        let probe = Context {
            limited: false,
            fallible_above: ctx.fallible_above,
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
        Ok(lowered.map(|join| Arc::new(join) as Arc<dyn PhysicalOperator>))
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
        if ctx.fallible_above && subtree_failing_parts(plan) > 0 {
            return Ok(bridge(plan));
        }
        if PhysicalFilter::supports_filter_exec(filter) {
            let own = failing_parts(filter.predicate(), &filter.input().schema());
            if own > 0 && !self.evaluates_in_order() {
                return Ok(bridge(plan));
            }
            let input = self.lower(
                filter.input(),
                Context {
                    limited: filter.fetch().is_some(),
                    fallible_above: ctx.fallible_above || own > 0,
                },
            )?;
            if own == 0 || native_failing_parts(&input) == 0 {
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
                .map(|expr| failing_parts(&expr.expr, &input_schema))
                .fold(0, usize::saturating_add);
            if own > 0 && !self.evaluates_in_order() {
                return Ok(bridge(plan));
            }
            let input = self.lower(
                projection.input(),
                Context {
                    limited: false,
                    fallible_above: ctx.fallible_above || own > 0,
                },
            )?;
            if own == 0 || native_failing_parts(&input) == 0 {
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
                !depends_on_batches(expr, &schema) && failing_parts(expr, &schema) == 0
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
/// expressions of `raw` (with `own_failing_parts` parts that may fail) may
/// fail: DataFusion evaluates them on the sorted rows, so that which row fails
/// first depends on the sort (and the sort reads all of its input before the
/// aggregation sees any row, so DataFusion raises the errors of the input
/// first).
fn without_ordering_sort<'a>(
    raw: &AggregateExec,
    own_failing_parts: usize,
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
        && keys.failing_parts == 0
        && own_failing_parts == 0
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
                !depends_on_batches(expr, &schema) && failing_parts(expr, &schema) == 0
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
    /// The number of the parts of the expressions that may fail (see
    /// [`failing_parts`]).
    failing_parts: usize,
}

impl Judgement {
    /// The judgement of the expressions that `plan` evaluates on its input:
    /// judged on the schema of its input if it has a single child, and without
    /// schema otherwise, which is conservative (expressions that may fail then
    /// fail on empty batches). The aggregations that merge partial states
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
                judgement.add(expr, &schema);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        if listed.is_err() {
            judgement.batch_dependent = true;
            judgement.failing_parts = usize::MAX;
        }
        judgement
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
        self.failing_parts = self
            .failing_parts
            .saturating_add(failing_parts(expr, schema));
    }

    /// Whether a native operator that evaluates these expressions on its
    /// chunks, rather than on DataFusion's batches, produces the results and
    /// raises the errors of DataFusion: no expression has errors that depend
    /// on the batches, and at most one part may fail (DataFusion evaluates
    /// every expression on a whole batch before the next one, so which of
    /// several errors comes first depends on its batches).
    fn evaluates_like_datafusion(&self) -> bool {
        !self.batch_dependent && self.failing_parts <= 1
    }
}

/// The judgement of the keys that `sort`, a [`PhysicalSort`] or a
/// [`PhysicalTopK`], evaluates on the rows of its input.
///
/// [`PhysicalSort`]: crate::sort::PhysicalSort
/// [`PhysicalTopK`]: crate::sort::PhysicalTopK
fn native_sort_keys(sort: &Arc<dyn PhysicalOperator>) -> Judgement {
    let ordering = if let Some(sort) = sort.downcast_ref::<crate::sort::PhysicalSort>() {
        sort.ordering()
    } else if let Some(topk) = sort.downcast_ref::<crate::sort::PhysicalTopK>() {
        topk.ordering()
    } else {
        // Conservatively: unknown keys may fail, depending on the batches.
        return Judgement {
            batch_dependent: true,
            failing_parts: usize::MAX,
        };
    };
    let Some(input) = sort.children().first() else {
        return Judgement::default();
    };
    Judgement::of_ordering(ordering, &input.schema())
}

/// The parts that may fail of all nodes of `plan`: the parts of their
/// expressions that may fail (see [`Judgement::of_plan`]), the sources that
/// may fail to read or decode their rows (see [`source_may_fail`]), and the
/// nodes that may fail to combine their batches (see
/// [`may_fail_to_combine`]).
fn subtree_failing_parts(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let own = Judgement::of_plan(plan.as_ref())
        .failing_parts
        .saturating_add(usize::from(source_may_fail(plan)))
        .saturating_add(usize::from(may_fail_to_combine(plan)));
    plan.children()
        .into_iter()
        .map(subtree_failing_parts)
        .fold(own, usize::saturating_add)
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

/// Whether DataFusion may fail to combine the batches of `plan`, a node that
/// it may execute. Every node but those that pass their batches on one by
/// one (projections, merges of partitions, unions, limits, and the leaves,
/// see [`source_may_fail`]) may concatenate, coalesce, merge or interleave
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
/// leaf but a scan of memory, of an empty relation or of placeholder rows,
/// e.g. a scan of files, which may fail to read or parse them, or a stream.
fn source_may_fail(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.children().is_empty()
        && !(plan.is::<EmptyExec>()
            || plan.is::<PlaceholderRowExec>()
            || plan
                .downcast_ref::<DataSourceExec>()
                .is_some_and(|scan| scan.data_source().is::<MemorySourceConfig>()))
}

/// The parts that may fail (see [`failing_parts`]) of the native filters and
/// projections that evaluate the chunks of the pipeline of `operator`, from
/// `operator` down to the source of the pipeline: through the operators that
/// pass on the chunks of their inputs (caching, limits, unions, and the probe
/// side of hash joins, whose keys and filters cannot fail), down to the
/// sources and pipeline breakers, which belong to other pipelines (and
/// bridged plans, which DataFusion evaluates on whole batches).
fn native_failing_parts(operator: &Arc<dyn PhysicalOperator>) -> usize {
    if let Some(join) = operator.downcast_ref::<PhysicalHashJoin>() {
        return native_failing_parts(join.probe_side());
    }
    if operator.is_source() || operator.is_sink() {
        return 0;
    }
    let input_schema = || operator.children()[0].schema();
    let own = if let Some(filter) = operator.downcast_ref::<PhysicalFilter>() {
        failing_parts(filter.predicate(), &input_schema())
    } else if let Some(projection) = operator.downcast_ref::<PhysicalProjection>() {
        let input_schema = input_schema();
        projection
            .exprs()
            .iter()
            .map(|(expr, _)| failing_parts(expr, &input_schema))
            .fold(0, usize::saturating_add)
    } else {
        0
    };
    operator
        .children()
        .iter()
        .map(native_failing_parts)
        .fold(own, usize::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{AsArray, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
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

    // ---------------------------------------------------------------------
    // Plans left to DataFusion
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_and_nested_loop_joins_are_bridged() {
        // There is no native window or nested loop join (yet): DataFusion
        // executes them with their inputs, below native operators.
        for (sql, bridged) in [
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
                "BoundedWindowAggExec",
            ),
            (
                "SELECT id, SUM(b) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
                "BoundedWindowAggExec",
            ),
            (
                "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
                "NestedLoopJoinExec",
            ),
        ] {
            let root = assert_parity(sql).await;
            let plans = bridged_plans(&root);
            let names = plans.iter().map(|plan| plan.name()).collect::<Vec<_>>();
            assert_eq!(
                names,
                vec![bridged],
                "{sql}\n{}",
                format_operator_tree(&root)
            );
        }
        // The aggregation above a window is native.
        let sql = "SELECT e, COUNT(*) AS n, MAX(rn) AS m FROM \
                   (SELECT e, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1) \
                   GROUP BY e";
        let root = assert_parity(sql).await;
        assert_eq!(count::<PhysicalHashAggregate>(&root), 1);
        assert_eq!(bridged_plans(&root).len(), 1);
        // And windows of native aggregations and joins.
        for sql in [
            "SELECT d, n, RANK() OVER (ORDER BY n DESC, d) AS r FROM \
             (SELECT d, COUNT(*) AS n FROM t1 GROUP BY d)",
            "SELECT a.k, b.k, a.w FROM t2 a CROSS JOIN t2 b WHERE a.w < 0",
        ] {
            assert_parity(sql).await;
        }
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
        // Their native forms, without the expressions whose errors depend on
        // the batches.
        for sql in [
            "SELECT SUM(CAST(a <> 0 AS INT)) AS n FROM g",
            "SELECT a FROM g ORDER BY CAST(a <> 0 AS INT), a",
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
        let root =
            create_pipeline_plan_with_options(make_plan(), config.options()).unwrap();
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
            Arc::new(datafusion_physical_expr::expressions::Column::new(
                partial.schema().field(0).name(),
                0,
            ));
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
            Arc::new(Literal::new(datafusion_common::ScalarValue::Int64(Some(
                10,
            )))),
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
                Arc::new(Literal::new(datafusion_common::ScalarValue::Int32(Some(
                    10,
                )))),
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
                    Arc::new(Literal::new(datafusion_common::ScalarValue::Int32(Some(
                        10,
                    )))),
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
                    Arc::new(Literal::new(datafusion_common::ScalarValue::Int32(Some(
                        0,
                    )))),
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
                    Arc::new(Literal::new(datafusion_common::ScalarValue::Int32(Some(
                        10,
                    )))),
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
                    Arc::new(Literal::new(datafusion_common::ScalarValue::Int32(Some(
                        10,
                    )))),
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
        // A table `k INT, v INT` of two Parquet files: 100 rows `k = 0, v =
        // 100`, and 10 rows `k = 1` of strings that fail to cast to `INT` when
        // they are read. The TopK of `ORDER BY k LIMIT 1` reads the first file,
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
        ])?;
        let path = dir.path().join("p");
        write_parquet(&path.join("0.parquet"), first.schema(), &[first], None)?;
        write_parquet(&path.join("1.parquet"), second.schema(), &[second], None)?;
        let path = path.to_str().unwrap();
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int32, true),
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
        let config = test_session_config().with_target_partitions(1);
        let sql = "SELECT k, v FROM p ORDER BY k LIMIT 1";
        let root = assert_parity_with(sql, config.clone(), &setup).await;
        assert_eq!(count::<PhysicalTopK>(&root), 0);
        // Over a scan of memory, which cannot fail, the TopK is native.
        let sql = "SELECT id, a FROM t1 ORDER BY a, id LIMIT 3";
        let root = assert_parity_with(sql, config, &setup).await;
        assert_eq!(count::<PhysicalTopK>(&root), 1);
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
            Arc::new(Literal::new(datafusion_common::ScalarValue::Int32(Some(
                10,
            )))),
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
                        return Ok(datafusion_common::tree_node::Transformed::no(node));
                    };
                    if last.mode() != &AggregateMode::FinalPartitioned {
                        return Ok(datafusion_common::tree_node::Transformed::no(node));
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
                    Ok(datafusion_common::tree_node::Transformed::yes(plan))
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

        // A bridged plan is shown below its bridge.
        let explained = explain(
            "EXPLAIN SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
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
        // A native sort of a bridged plan (a window), whose nodes have metrics
        // too.
        let query = "SELECT e, rn FROM (SELECT e, \
                     ROW_NUMBER() OVER (PARTITION BY e ORDER BY id) AS rn FROM t1) \
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
