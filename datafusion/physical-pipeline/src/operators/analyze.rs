// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalAnalyze`]: the execution of the plan of an `EXPLAIN ANALYZE`,
//! and its rendering with its metrics (`AnalyzeExec`).

use std::fmt;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{RecordBatch, StringBuilder};
use arrow::datatypes::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::format::{ExplainFormat, MetricType};
use datafusion_common::instant::Instant;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, internal_err};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::analyze::AnalyzeExec;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::operator_statistics::StatisticsRegistry;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    ExecutionPlanProperties, PhysicalExpr, PlanProperties, ReplaceChildrenOptions,
};
use futures::StreamExt;

use crate::bridge::PipelineRootExec;
use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state_mut,
};

/// The number of batches of its input that a [`PhysicalAnalyze`] reads in one
/// call before it returns to the executor, so that the executor can yield.
const BATCHES_PER_CALL: usize = 32;

/// How the plan of an `EXPLAIN ANALYZE` is rendered with its metrics: the
/// metric types that the statement (or the session) selects, and the
/// statistics registry of the session, which `AnalyzeExec` renders the plan
/// with but does not expose.
#[derive(Debug, Clone)]
pub(crate) struct AnalyzeRendering {
    metric_types: Vec<MetricType>,
    statistics_registry: StatisticsRegistry,
}

impl AnalyzeRendering {
    /// Render with the metric types `metric_types` and the statistics
    /// registry `statistics_registry`, as DataFusion's planner renders an
    /// `AnalyzeExec` of the same statement.
    pub(crate) fn new(
        metric_types: Vec<MetricType>,
        statistics_registry: StatisticsRegistry,
    ) -> Self {
        Self {
            metric_types,
            statistics_registry,
        }
    }

    /// The rendering of a session with the options `options`, without a
    /// statistics registry.
    fn of_session(options: &ConfigOptions) -> Self {
        Self::new(
            options.explain.analyze_level.included_types(),
            StatisticsRegistry::default(),
        )
    }
}

/// The input of an `AnalyzeExec` that the
/// [`PipelineQueryPlanner`](crate::PipelineQueryPlanner) plans: the plan that
/// the `EXPLAIN ANALYZE` executes and renders (all of whose properties,
/// rendering, metrics and execution are those of the plan), with the
/// [`AnalyzeRendering`] of the statement, for [`PhysicalAnalyze`].
#[derive(Debug)]
pub(crate) struct AnalyzedInput {
    input: Arc<dyn ExecutionPlan>,
    rendering: AnalyzeRendering,
}

impl AnalyzedInput {
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        rendering: AnalyzeRendering,
    ) -> Self {
        Self { input, rendering }
    }
}

impl DisplayAs for AnalyzedInput {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.input.fmt_as(t, f)
    }
}

impl ExecutionPlan for AnalyzedInput {
    fn name(&self) -> &str {
        self.input.name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.input.children()
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        self.input.apply_expressions(f)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input: Arc::clone(&self.input).replace_children(children, options)?,
            rendering: self.rendering.clone(),
        }))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input: Arc::clone(&self.input).reset_state()?,
            rendering: self.rendering.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.input.metrics()
    }
}

/// Executes the plan of an `EXPLAIN ANALYZE`, discarding its rows, and then
/// emits the rendering of the plan with its metrics, as `AnalyzeExec` does:
/// `Plan with Metrics` (and in verbose mode `Plan with Full Metrics`,
/// `Output Rows` and `Duration`), in the format of the statement.
///
/// The plan is the plan of the pipeline engine, which executes it natively:
/// the `PipelineRootExec` of the query that the
/// [`PipelineQueryPlanner`](crate::PipelineQueryPlanner) lowered (in an
/// [`AnalyzedInput`]), or the lowered input of another `AnalyzeExec`. The
/// operator runs on a single thread, which reads the partitions of the plan
/// in turn (polled with the waker of its task) and counts their rows; the
/// plan runs its pipelines on threads of its own. The operator holds no rows.
#[derive(Debug)]
pub struct PhysicalAnalyze {
    /// The plan that is executed and rendered.
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    verbose: bool,
    show_statistics: bool,
    metric_categories: Option<Vec<datafusion_common::format::MetricCategory>>,
    format: ExplainFormat,
    rendering: AnalyzeRendering,
}

impl PhysicalAnalyze {
    /// Lower an `AnalyzeExec` to a [`PhysicalAnalyze`]. Its input is the plan
    /// of an [`AnalyzedInput`], or else its input, lowered for a session with
    /// `options` (and `runtime`, if known, see
    /// [`create_pipeline_plan_with_runtime`](crate::planner::create_pipeline_plan_with_runtime)),
    /// which is then rendered with the metric types of the session. Returns
    /// `Ok(None)` for any other node.
    pub fn try_lower(
        plan: &Arc<dyn ExecutionPlan>,
        options: &ConfigOptions,
        runtime: Option<&RuntimeEnv>,
    ) -> Result<Option<Self>> {
        let Some(analyze) = plan.downcast_ref::<AnalyzeExec>() else {
            return Ok(None);
        };
        let (input, rendering) = match analyze.input().downcast_ref::<AnalyzedInput>() {
            Some(analyzed) => (Arc::clone(analyze.input()), analyzed.rendering.clone()),
            None => {
                let input = Arc::clone(analyze.input());
                let root: Arc<dyn ExecutionPlan> = match runtime {
                    Some(runtime) => crate::planner::create_pipeline_plan_with_runtime(
                        input, options, runtime,
                    )?,
                    None => {
                        crate::planner::create_pipeline_plan_with_options(input, options)?
                    }
                };
                (root, AnalyzeRendering::of_session(options))
            }
        };
        Ok(Some(Self {
            input,
            schema: plan.schema(),
            verbose: analyze.verbose(),
            show_statistics: analyze.show_statistics(),
            metric_categories: analyze.metric_categories().map(<[_]>::to_vec),
            format: analyze.format().clone(),
            rendering,
        }))
    }

    /// `displayable` with the metrics that the rendering shows.
    fn annotated<'a>(
        &self,
        displayable: DisplayableExecutionPlan<'a>,
    ) -> DisplayableExecutionPlan<'a> {
        displayable
            .set_metric_types(self.rendering.metric_types.clone())
            .set_metric_categories(self.metric_categories.clone())
    }

    /// The rows of `AnalyzeExec` for the input that produced `rows` rows in
    /// `duration` (`create_output_batch` of DataFusion's `AnalyzeExec`).
    fn output(&self, rows: usize, duration: std::time::Duration) -> Result<RecordBatch> {
        let mut types = StringBuilder::with_capacity(1, 1024);
        let mut plans = StringBuilder::with_capacity(1, 1024);
        let input = self.input.as_ref();
        let annotated = |displayable| self.annotated(displayable);
        match &self.format {
            ExplainFormat::Indent => {
                types.append_value("Plan with Metrics");
                let plan = annotated(DisplayableExecutionPlan::with_metrics(input))
                    .set_show_statistics(self.show_statistics)
                    .set_statistics_registry(self.rendering.statistics_registry.clone())
                    .indent(self.verbose)
                    .to_string();
                plans.append_value(plan);
                if self.verbose {
                    types.append_value("Plan with Full Metrics");
                    let plan =
                        annotated(DisplayableExecutionPlan::with_full_metrics(input))
                            .set_show_statistics(self.show_statistics)
                            .set_statistics_registry(
                                self.rendering.statistics_registry.clone(),
                            )
                            .indent(self.verbose)
                            .to_string();
                    plans.append_value(plan);
                    types.append_value("Output Rows");
                    plans.append_value(rows.to_string());
                    types.append_value("Duration");
                    plans.append_value(format!("{duration:?}"));
                }
            }
            ExplainFormat::PostgresJSON => {
                types.append_value("Plan with Metrics");
                let mut displayable = annotated(match self.verbose {
                    true => DisplayableExecutionPlan::with_full_metrics(input),
                    false => DisplayableExecutionPlan::with_metrics(input),
                });
                if self.verbose {
                    displayable = displayable.set_summary(Some(rows), Some(duration));
                }
                plans.append_value(displayable.pgjson(self.verbose).to_string());
            }
            format @ (ExplainFormat::Tree | ExplainFormat::Graphviz) => {
                return internal_err!(
                    "AnalyzeExec does not support {format} output format"
                );
            }
        }
        Ok(RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![Arc::new(types.finish()), Arc::new(plans.finish())],
        )?)
    }
}

/// The local source state of a [`PhysicalAnalyze`]: the streams of the
/// partitions of the plan that did not end, once it started, and the rows
/// they produced.
#[derive(Default)]
struct AnalyzeState {
    started: Option<Instant>,
    streams: Vec<SendableRecordBatchStream>,
    next: usize,
    rows: usize,
    emitted: bool,
}

impl LocalSourceState for AnalyzeState {}

impl PhysicalOperator for PhysicalAnalyze {
    fn name(&self) -> &str {
        "PhysicalAnalyze"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        let plan = match self.input.downcast_ref::<AnalyzedInput>() {
            Some(analyzed) => Arc::clone(&analyzed.input),
            None => Arc::clone(&self.input),
        };
        let operators = plan
            .downcast_ref::<PipelineRootExec>()
            .map(|root| format!(", root={}", root.operator().name()))
            .unwrap_or_default();
        format!("PhysicalAnalyze: verbose={}{operators}", self.verbose)
    }

    fn is_source(&self) -> bool {
        true
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::<AnalyzeState>::default())
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state_mut::<AnalyzeState>(lstate.as_any_mut(), self.name())?;
        if state.emitted {
            return Ok(SourceResult::Finished);
        }
        let started = match state.started {
            Some(started) => started,
            None => {
                let started = Instant::now();
                for partition in 0..self.input.output_partitioning().partition_count() {
                    state
                        .streams
                        .push(self.input.execute(partition, Arc::clone(&ctx.task_ctx))?);
                }
                *state.started.insert(started)
            }
        };
        let mut cx = Context::from_waker(ctx.waker());
        let mut batches = 0;
        while !state.streams.is_empty() {
            if batches == BATCHES_PER_CALL {
                // Let the executor yield if it has to.
                return Ok(SourceResult::Chunk(DataChunk::empty(self.schema())));
            }
            let mut progress = false;
            let streams = state.streams.len();
            for i in 0..streams {
                let index = (state.next + i) % streams;
                match state.streams[index].poll_next_unpin(&mut cx) {
                    Poll::Ready(Some(batch)) => {
                        state.rows += batch?.num_rows();
                        state.next = index + 1;
                        batches += 1;
                        progress = true;
                        break;
                    }
                    Poll::Ready(None) => {
                        drop(state.streams.remove(index));
                        state.next = index;
                        progress = true;
                        break;
                    }
                    Poll::Pending => {}
                }
            }
            if !progress {
                return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
            }
        }
        let output = self.output(state.rows, started.elapsed())?;
        state.emitted = true;
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(output)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::AsArray;
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::collect;

    use crate::bridge::bridged_plans;
    use crate::operators::lowering_tests::count;
    use crate::test_utils::{
        datafusion_context, pipeline_context, register_test_tables, test_session_config,
    };

    /// The rows of the `EXPLAIN ANALYZE` `sql` in `ctx`: the types of the
    /// plans and the plans.
    async fn analyze(ctx: &SessionContext, sql: &str) -> Result<Vec<(String, String)>> {
        let batches = ctx.sql(sql).await?.collect().await?;
        Ok(batches
            .iter()
            .flat_map(|batch| {
                let types = batch.column(0).as_string::<i32>();
                let plans = batch.column(1).as_string::<i32>();
                (0..batch.num_rows())
                    .map(|i| (types.value(i).to_string(), plans.value(i).to_string()))
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn explain_analyze_is_native() -> Result<()> {
        let ctx = pipeline_context(test_session_config());
        register_test_tables(&ctx)?;
        for sql in [
            "EXPLAIN ANALYZE SELECT a, COUNT(*) FROM t1 GROUP BY a",
            "EXPLAIN ANALYZE VERBOSE SELECT * FROM t1 JOIN t2 ON t1.a = t2.k",
            "EXPLAIN (ANALYZE, LEVEL summary) SELECT * FROM t1 ORDER BY id LIMIT 3",
            "EXPLAIN (ANALYZE, METRICS 'rows', TIMING OFF) SELECT 1",
            "EXPLAIN (ANALYZE, FORMAT pgjson) SELECT a FROM t1 WHERE a > 3",
        ] {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let root = plan.downcast_ref::<PipelineRootExec>().unwrap();
            assert!(bridged_plans(root.operator()).is_empty(), "{sql}");
            assert_eq!(count::<PhysicalAnalyze>(root.operator()), 1, "{sql}");
            let rows = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
            assert_eq!(rows[0].schema(), plan.schema(), "{sql}");
            let types = rows[0]
                .column(0)
                .as_string::<i32>()
                .iter()
                .map(|t| t.unwrap().to_string())
                .collect::<Vec<_>>();
            let expected = if sql.contains("VERBOSE") {
                vec![
                    "Plan with Metrics",
                    "Plan with Full Metrics",
                    "Output Rows",
                    "Duration",
                ]
            } else {
                vec!["Plan with Metrics"]
            };
            assert_eq!(types, expected, "{sql}");
        }
        // The rows that the plan produced.
        let explained = analyze(
            &ctx,
            "EXPLAIN ANALYZE VERBOSE SELECT id FROM t1 WHERE a > 10",
        )
        .await?;
        let rows = ctx
            .sql("SELECT COUNT(*) FROM t1 WHERE a > 10")
            .await?
            .collect()
            .await?;
        let rows = rows[0]
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(explained[2].1, rows.value(0).to_string());
        // The metrics of the native operators.
        let sql = "EXPLAIN ANALYZE SELECT e, COUNT(*) FROM t1 GROUP BY e";
        let (_, plan) = &analyze(&ctx, sql).await?[0];
        assert!(plan.starts_with("PipelineRootExec"), "{plan}");
        assert!(plan.contains("PhysicalHashAggregate"), "{plan}");
        assert!(plan.contains("output_rows=5.00 K"), "{plan}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn analyzes_of_other_plans_lower_their_plans() -> Result<()> {
        // An `AnalyzeExec` of DataFusion's plan: its plan is lowered, and
        // rendered with the native operators.
        let ctx = datafusion_context(test_session_config());
        register_test_tables(&ctx)?;
        let plan = ctx
            .sql("EXPLAIN ANALYZE SELECT d, MAX(b) FROM t1 GROUP BY d")
            .await?
            .create_physical_plan()
            .await?;
        assert!(plan.is::<AnalyzeExec>());
        let state = ctx.state();
        let root: Arc<dyn ExecutionPlan> =
            crate::planner::create_pipeline_plan_with_runtime(
                Arc::clone(&plan),
                state.config_options(),
                state.runtime_env(),
            )?;
        let operator = root.downcast_ref::<PipelineRootExec>().unwrap().operator();
        assert!(bridged_plans(operator).is_empty());
        assert_eq!(count::<PhysicalAnalyze>(operator), 1);
        let rows = collect(root, ctx.task_ctx()).await?;
        let plan = rows[0].column(1).as_string::<i32>().value(0);
        assert!(plan.starts_with("PipelineRootExec"), "{plan}");
        assert!(plan.contains("PhysicalHashAggregate"), "{plan}");
        Ok(())
    }
}
