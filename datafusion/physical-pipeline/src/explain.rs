// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `EXPLAIN` and metrics support for [`PipelineRootExec`].
//!
//! * The [`DisplayAs`] implementation of [`PipelineRootExec`] renders the
//!   lowered operator tree, with the pipelines every operator takes part in,
//!   and the pipelines the tree decomposes into, which is what `EXPLAIN` shows
//!   (see [`PipelineQueryPlanner`](crate::planner::PipelineQueryPlanner)). Once
//!   the plan was executed, every operator also shows its metrics: what
//!   `EXPLAIN ANALYZE` shows.
//! * [`format_operator_tree`] and [`format_pipelines`] render an operator
//!   tree and the pipelines it decomposes into.
//! * [`root_metrics`] provides the metrics reported by
//!   [`ExecutionPlan::metrics`] for `EXPLAIN ANALYZE`.
//! * `EXPLAIN ANALYZE` shows the metrics that the statement (or the session)
//!   selects, by type (`LEVEL`, `datafusion.explain.analyze_level`) and by
//!   category (`METRICS` and `TIMING`,
//!   `datafusion.explain.analyze_categories`). DataFusion applies the
//!   selection to the metrics of the nodes of the plan it shows, of which
//!   the operators of the pipeline engine are not: the planner shows the
//!   plan through an `AnalyzedPipelineExec`, which applies it to them, and
//!   to the nodes of the bridged plans.
//! * If the execution was profiled (see [`ProfilingOptions`]), `EXPLAIN
//!   ANALYZE` also shows the metrics of its [`QueryProfile`] that fit on its
//!   lines (development metrics): those of the execution
//!   (`profile=[...]` on the line of the `PipelineRootExec`), of every
//!   operator (`blocked_time`, `rows_in`, `bytes_in`, `bytes_out` and
//!   `memory_peak`, after its other metrics) and of every pipeline
//!   (`profile=[...]` on its line). [`QueryProfile::to_json`] has all of
//!   them.
//!
//! For example, `EXPLAIN ANALYZE SELECT d, COUNT(*) FROM t GROUP BY d` shows
//!
//! ```text
//! PipelineRootExec: pipelines=2
//!   PhysicalProjection: expr=[d@0 as d, count(Int64(1))@1 as count(*)], pipelines=[0], metrics=[output_rows=7, output_chunks=6, elapsed_compute=24.23µs]
//!     PhysicalHashAggregate: gby=[d@0 as d], aggr=[count(Int64(1))], pipelines=[0, 1], metrics=[output_rows=7, output_chunks=6, sink_rows=5.00 K, sink_chunks=6, elapsed_compute=6.60ms]
//!       MemoryScanSource: partitions=4, rows=5000, chunks=6, pipelines=[1], metrics=[output_rows=5.00 K, output_chunks=6, elapsed_compute=30.95µs]
//!   Pipeline 1: MemoryScanSource -> PhysicalHashAggregate
//!   Pipeline 0: PhysicalHashAggregate -> PhysicalProjection -> CollectorSink, after pipelines [1], metrics=[output_rows=7, elapsed_compute=6.69ms, output_bytes=48.3 KB, output_batches=6]
//! ```
//!
//! The operators are listed from the root of the tree, the pipelines in
//! reverse order, which is an order in which they can run: a pipeline runs
//! after the pipelines it depends on (`after pipelines`) completed, and
//! pipeline 0 produces the output of the plan (into the `CollectorSink` of
//! the execution). The last metrics are those of the [`PipelineRootExec`]
//! itself (see [`root_metrics`]), which DataFusion appends. A plan bridged to
//! DataFusion is shown below its [`BridgeOperator`], with the metrics of its
//! nodes once it was executed.

use std::collections::HashMap;
use std::fmt::{self, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::display::human_readable::{
    human_readable_count, human_readable_duration, human_readable_size,
};
use datafusion_common::display::{PlanType, StringifiedPlan};
use datafusion_common::format::ExplainFormat;
use datafusion_common::format::{ExplainAnalyzeCategories, MetricCategory, MetricType};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, internal_err};
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion_expr::logical_plan::{Analyze, Explain};
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use datafusion_physical_plan::explain::ExplainExec;
use datafusion_physical_plan::metrics::{BaselineMetrics, MetricsSet};
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, PlanProperties,
    ReplaceChildrenOptions, displayable,
};
use datafusion_session::Session;
use futures::{Stream, StreamExt};

use crate::bridge::{BridgeOperator, PipelineRootExec, bridged_plans};
use crate::executor::{ExecutionMetrics, PipelineStream};
use crate::operator::{OperatorMetrics, PhysicalOperator, operator_key};
use crate::pipeline::{PipelineGraph, PipelineId};
#[cfg(doc)]
use crate::profiling::ProfilingOptions;
use crate::profiling::{Metric, MetricUnit, OperatorProfile, QueryProfile};

impl DisplayAs for PipelineRootExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_root(self, &MetricSelection::default(), t, f)
    }
}

/// Render `root` as `t`, with the metrics of its last execution (if any)
/// that `selection` selects.
fn fmt_root(
    root: &PipelineRootExec,
    selection: &MetricSelection,
    t: DisplayFormatType,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let explained =
        ExplainedTree::new(root.operator(), root.execution_metrics(), selection);
    match t {
        DisplayFormatType::Default => explained.fmt_indent(f, false),
        DisplayFormatType::Verbose => explained.fmt_indent(f, true),
        DisplayFormatType::TreeRender => explained.fmt_tree_render(f),
    }
}

/// The metrics that `EXPLAIN ANALYZE` shows: of the types and the categories
/// that the statement selects, or else the session (see
/// `datafusion.explain.analyze_level` and
/// `datafusion.explain.analyze_categories`), all by default.
#[derive(Debug, Clone)]
pub(crate) struct MetricSelection {
    types: Vec<MetricType>,
    /// The categories, or `None` for all of them.
    categories: Option<Vec<MetricCategory>>,
}

impl Default for MetricSelection {
    fn default() -> Self {
        Self {
            types: MetricType::Dev.included_types(),
            categories: None,
        }
    }
}

impl MetricSelection {
    /// The selection of `analyze` in a session with the options `options`,
    /// as DataFusion's planner computes it for its `AnalyzeExec`.
    pub(crate) fn of_analyze(analyze: &Analyze, options: &ConfigOptions) -> Self {
        let level = analyze
            .analyze_level
            .unwrap_or(options.explain.analyze_level);
        let categories = analyze
            .analyze_categories
            .clone()
            .unwrap_or_else(|| options.explain.analyze_categories.clone());
        Self {
            types: level.included_types(),
            categories: match categories {
                ExplainAnalyzeCategories::All => None,
                ExplainAnalyzeCategories::Only(categories) => Some(categories),
            },
        }
    }

    /// Whether a metric of `metric_type` and `category` is shown.
    fn shows(&self, metric_type: MetricType, category: MetricCategory) -> bool {
        self.types.contains(&metric_type)
            && self
                .categories
                .as_ref()
                .is_none_or(|categories| categories.contains(&category))
    }
}

/// An operator tree, with the pipelines it decomposes into and the metrics of
/// an execution, rendered for `EXPLAIN`.
struct ExplainedTree<'a> {
    root: &'a Arc<dyn PhysicalOperator>,
    /// The pipelines of the tree, or the reason why it cannot be decomposed.
    graph: std::result::Result<PipelineGraph, String>,
    /// The pipelines every operator (by [`operator_key`]) takes part in.
    pipelines: HashMap<usize, Vec<PipelineId>>,
    /// The metrics of the last execution, if any.
    metrics: Option<Arc<ExecutionMetrics>>,
    /// The profile of the last execution, if it was profiled.
    profile: Option<Arc<QueryProfile>>,
    /// The metrics that are shown.
    selection: &'a MetricSelection,
}

impl<'a> ExplainedTree<'a> {
    fn new(
        root: &'a Arc<dyn PhysicalOperator>,
        metrics: Option<Arc<ExecutionMetrics>>,
        selection: &'a MetricSelection,
    ) -> Self {
        let graph = pipeline_graph(root, "CollectorSink").map_err(|e| e.to_string());
        let mut pipelines: HashMap<usize, Vec<PipelineId>> = HashMap::new();
        if let Ok(graph) = &graph {
            for pipeline in graph.pipelines() {
                let operators = std::iter::once(pipeline.source())
                    .chain(pipeline.operators())
                    .chain(std::iter::once(pipeline.sink()));
                for operator in operators {
                    let ids = pipelines
                        .entry(operator_key(operator.as_ref()))
                        .or_default();
                    if !ids.contains(&pipeline.id()) {
                        ids.push(pipeline.id());
                    }
                }
            }
        }
        let profile = metrics.as_ref().and_then(|metrics| metrics.profile());
        Self {
            root,
            graph,
            pipelines,
            metrics,
            profile,
            selection,
        }
    }

    /// The indented rendering of `EXPLAIN`: the operator tree, then the
    /// pipelines, see the [module documentation](self).
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, verbose: bool) -> fmt::Result {
        write!(f, "PipelineRootExec: ")?;
        match &self.graph {
            Ok(graph) => write!(f, "pipelines={}", graph.pipelines().len())?,
            Err(e) => write!(f, "pipelines=<{e}>")?,
        }
        let bridged = bridged_plans(self.root).len();
        if bridged > 0 {
            write!(f, ", bridged_plans={bridged}")?;
        }
        if let Some(profile) = &self.profile {
            let metrics = profile.executor().metrics();
            let shown =
                format_profile_metrics(&metrics, EXECUTOR_METRICS, self.selection);
            write!(f, ", profile=[{shown}]")?;
        }
        self.fmt_operator(f, self.root, 1, verbose)?;
        let Ok(graph) = &self.graph else {
            return Ok(());
        };
        for pipeline in graph.pipelines().iter().rev() {
            write!(f, "\n  Pipeline {}: {pipeline}", pipeline.id())?;
            let mut after = pipeline.dependencies().to_vec();
            for &meta in pipeline.meta_dependencies() {
                after.extend_from_slice(graph.meta_pipeline(meta).pipelines());
            }
            after.sort_unstable();
            after.dedup();
            if !after.is_empty() {
                write!(f, ", after pipelines {after:?}")?;
            }
            if let Some(profiled) = self
                .profile
                .as_ref()
                .and_then(|profile| profile.pipelines().get(pipeline.id()))
            {
                let metrics = profiled.metrics();
                let shown =
                    format_profile_metrics(&metrics, PIPELINE_METRICS, self.selection);
                write!(f, ", profile=[{shown}]")?;
            }
        }
        Ok(())
    }

    /// Render `operator` and its children, at `depth`, each on a new line.
    fn fmt_operator(
        &self,
        f: &mut fmt::Formatter<'_>,
        operator: &Arc<dyn PhysicalOperator>,
        depth: usize,
        verbose: bool,
    ) -> fmt::Result {
        let indent = "  ".repeat(depth);
        let bridge = operator.downcast_ref::<BridgeOperator>();
        match bridge {
            Some(_) => write!(f, "\n{indent}BridgeOperator")?,
            None => write!(f, "\n{indent}{}", operator.description())?,
        }
        if let Some(pipelines) = self.pipelines.get(&operator_key(operator.as_ref())) {
            write!(f, ", pipelines={pipelines:?}")?;
        }
        if let Some(metrics) = self
            .metrics
            .as_ref()
            .and_then(|metrics| metrics.get(operator.as_ref()))
        {
            let profiled = self
                .profile
                .as_ref()
                .and_then(|profile| profile.operator(operator.as_ref()));
            write!(
                f,
                ", metrics=[{}]",
                format_operator_metrics(operator, metrics, profiled, self.selection)
            )?;
        }
        if let Some(bridge) = bridge {
            // The plan that DataFusion executes, with the metrics of its
            // nodes once it was executed.
            let plan = bridge.plan().as_ref();
            let displayable = match self.metrics {
                Some(_) => DisplayableExecutionPlan::with_metrics(plan)
                    .set_metric_types(self.selection.types.clone())
                    .set_metric_categories(self.selection.categories.clone()),
                None => DisplayableExecutionPlan::new(plan),
            };
            let nested = "  ".repeat(depth + 1);
            for line in displayable.indent(verbose).to_string().lines() {
                write!(f, "\n{nested}{line}")?;
            }
        }
        for child in operator.children() {
            self.fmt_operator(f, child, depth + 1, verbose)?;
        }
        Ok(())
    }

    /// The rendering of `EXPLAIN FORMAT TREE`: `key=value` lines, see
    /// [`DisplayFormatType::TreeRender`].
    fn fmt_tree_render(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.graph {
            Ok(graph) => {
                writeln!(f, "pipelines={}", graph.pipelines().len())?;
                for pipeline in graph.pipelines() {
                    writeln!(f, "pipeline {}={pipeline}", pipeline.id())?;
                }
            }
            Err(e) => writeln!(f, "pipelines=<{e}>")?,
        }
        let bridged = bridged_plans(self.root).len();
        if bridged > 0 {
            writeln!(f, "bridged_plans={bridged}")?;
        }
        Ok(())
    }
}

/// The metrics of the [`QueryProfile`] of an execution that `EXPLAIN ANALYZE`
/// shows on the line of the `PipelineRootExec`.
const EXECUTOR_METRICS: &[&str] = &[
    "elapsed_time",
    "threads",
    "morsels",
    "active_time",
    "idle_time",
    "blocked_time",
    "queue_wait_time",
    "memory_peak",
];

/// The metrics of the profile of an operator that `EXPLAIN ANALYZE` shows
/// after its other metrics.
const OPERATOR_METRICS: &[&str] = &[
    "blocked_time",
    "rows_in",
    "bytes_in",
    "bytes_out",
    "memory_peak",
];

/// The metrics of the profile of a pipeline that `EXPLAIN ANALYZE` shows on
/// its line.
const PIPELINE_METRICS: &[&str] = &[
    "threads",
    "morsels",
    "elapsed_time",
    "active_time",
    "idle_time",
    "blocked_time",
    "queue_wait_time",
];

/// The metrics `names` of `metrics`, a part of a [`QueryProfile`], that
/// `selection` selects, for `EXPLAIN ANALYZE`: all of them are development
/// metrics, whose category is that of their unit.
fn format_profile_metrics(
    metrics: &[Metric],
    names: &[&str],
    selection: &MetricSelection,
) -> String {
    let mut out = String::new();
    for metric in names
        .iter()
        .filter_map(|name| metrics.iter().find(|metric| metric.name == *name))
    {
        let category = match metric.unit {
            MetricUnit::Nanoseconds => MetricCategory::Timing,
            MetricUnit::Bytes => MetricCategory::Bytes,
            MetricUnit::Rows | MetricUnit::Morsels | MetricUnit::Count => {
                MetricCategory::Rows
            }
        };
        if !selection.shows(MetricType::Dev, category) {
            continue;
        }
        if !out.is_empty() {
            out.push_str(", ");
        }
        let value = match metric.unit {
            MetricUnit::Nanoseconds => human_readable_duration(metric.value),
            MetricUnit::Bytes => human_readable_size(metric.value as usize),
            MetricUnit::Rows | MetricUnit::Morsels | MetricUnit::Count => {
                human_readable_count(metric.value as usize)
            }
        };
        let _ = write!(out, "{}={value}", metric.name);
    }
    out
}

/// The metrics of `operator` in an execution, for `EXPLAIN ANALYZE`, that
/// `selection` selects: the rows and chunks it produced (as a source or an
/// operator) and consumed (as a sink), and the time spent in it, summed over
/// all threads. Their types and categories are those of the corresponding
/// metrics of DataFusion: `output_rows` and `elapsed_compute` are summary
/// metrics, the others development metrics, like `output_batches`. If the
/// execution was profiled, the metrics of its profile (`profiled`) that
/// [`OPERATOR_METRICS`] lists follow.
fn format_operator_metrics(
    operator: &Arc<dyn PhysicalOperator>,
    metrics: &OperatorMetrics,
    profiled: Option<&OperatorProfile>,
    selection: &MetricSelection,
) -> String {
    let count = |value: usize| human_readable_count(value);
    let mut shown = vec![(
        "output_rows",
        count(metrics.output_rows()),
        MetricType::Summary,
        MetricCategory::Rows,
    )];
    shown.push((
        "output_chunks",
        count(metrics.output_chunks()),
        MetricType::Dev,
        MetricCategory::Rows,
    ));
    if operator.is_sink() {
        shown.push((
            "sink_rows",
            count(metrics.sink_rows()),
            MetricType::Dev,
            MetricCategory::Rows,
        ));
        shown.push((
            "sink_chunks",
            count(metrics.sink_chunks()),
            MetricType::Dev,
            MetricCategory::Rows,
        ));
    }
    shown.push((
        "elapsed_compute",
        human_readable_duration(metrics.elapsed_nanos()),
        MetricType::Summary,
        MetricCategory::Timing,
    ));
    let mut out = String::new();
    for (name, value, metric_type, category) in shown {
        if selection.shows(metric_type, category) {
            if !out.is_empty() {
                out.push_str(", ");
            }
            let _ = write!(out, "{name}={value}");
        }
    }
    if let Some(profiled) = profiled {
        let profile =
            format_profile_metrics(&profiled.metrics(), OPERATOR_METRICS, selection);
        if !out.is_empty() && !profile.is_empty() {
            out.push_str(", ");
        }
        out.push_str(&profile);
    }
    out
}

/// The `ExplainExec` of an `EXPLAIN` of the plan that the pipeline engine
/// executes: `exec`, the `ExplainExec` that DataFusion planned for `explain`
/// in `session`, with the final physical plans (also with statistics and
/// schema) of `root`, the lowered plan of the explained query. The logical
/// plans, and in verbose mode the physical plans of DataFusion that `root`
/// was lowered from, are kept.
pub(crate) fn explain_pipeline_plan(
    exec: &ExplainExec,
    root: &Arc<PipelineRootExec>,
    explain: &Explain,
    session: &dyn Session,
) -> ExplainExec {
    let options = &session.config_options().explain;
    let show_statistics = explain.show_statistics.unwrap_or(options.show_statistics);
    let registry = session.statistics_registry().cloned().unwrap_or_default();
    let plan = root.as_ref() as &dyn ExecutionPlan;
    let render = |show_statistics: bool, show_schema: bool| match explain.explain_format {
        ExplainFormat::Tree => displayable(plan)
            .set_tree_maximum_render_width(options.tree_maximum_render_width)
            .tree_render()
            .to_string(),
        _ => displayable(plan)
            .set_show_statistics(show_statistics)
            .set_statistics_registry(registry.clone())
            .set_show_schema(show_schema)
            .indent(explain.verbose)
            .to_string(),
    };
    let plans = exec
        .stringified_plans()
        .iter()
        .map(|plan| {
            let rendered = match plan.plan_type {
                PlanType::FinalPhysicalPlan => {
                    render(show_statistics, options.show_schema)
                }
                PlanType::FinalPhysicalPlanWithStats => render(true, false),
                PlanType::FinalPhysicalPlanWithSchema => render(false, true),
                _ => return plan.clone(),
            };
            StringifiedPlan::new(plan.plan_type.clone(), rendered)
        })
        .collect();
    ExplainExec::new(exec.schema(), plans, exec.verbose())
}

/// The plan that the pipeline engine executes for an `EXPLAIN ANALYZE`, the
/// input of its `AnalyzeExec`: a [`PipelineRootExec`], shown with the metrics
/// of its operators, and of the nodes of its bridged plans, that `selection`
/// selects (see the [module documentation](self)). It executes, and reports
/// the metrics of, the [`PipelineRootExec`].
#[derive(Debug)]
pub(crate) struct AnalyzedPipelineExec {
    /// The [`PipelineRootExec`].
    root: Arc<dyn ExecutionPlan>,
    selection: MetricSelection,
}

impl AnalyzedPipelineExec {
    pub(crate) fn new(root: Arc<PipelineRootExec>, selection: MetricSelection) -> Self {
        Self {
            root: root as Arc<dyn ExecutionPlan>,
            selection,
        }
    }
}

impl DisplayAs for AnalyzedPipelineExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.root.downcast_ref::<PipelineRootExec>() {
            Some(root) => fmt_root(root, &self.selection, t, f),
            None => self.root.fmt_as(t, f),
        }
    }
}

impl ExecutionPlan for AnalyzedPipelineExec {
    fn name(&self) -> &str {
        self.root.name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.root.properties()
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

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("AnalyzedPipelineExec does not have children");
        }
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("AnalyzedPipelineExec does not have children");
        }
        Ok(self)
    }

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            root: Arc::clone(&self.root).reset_state()?,
            selection: self.selection.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.root.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.root.metrics()
    }
}

/// Render an operator tree on one line, e.g.
/// `Projection(BridgeOperator: plan=DataSourceExec: ...)`.
pub fn format_operator_tree(operator: &Arc<dyn PhysicalOperator>) -> String {
    let mut out = String::new();
    write_operator_tree(operator, &mut out);
    out
}

fn write_operator_tree(operator: &Arc<dyn PhysicalOperator>, out: &mut String) {
    out.push_str(&operator.description());
    if operator.children().is_empty() {
        return;
    }
    out.push('(');
    for (i, child) in operator.children().iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_operator_tree(child, out);
    }
    out.push(')');
}

/// Render the pipelines that the operator tree `root` decomposes into, one
/// line per meta pipeline and pipeline. The root pipeline ends in a
/// placeholder `ResultSink`.
pub fn format_pipelines(root: &Arc<dyn PhysicalOperator>) -> Result<String> {
    Ok(pipeline_graph(root, "ResultSink")?.to_string())
}

/// The pipelines of the operator tree `root`, whose root pipeline ends in a
/// placeholder sink named `sink`.
fn pipeline_graph(
    root: &Arc<dyn PhysicalOperator>,
    sink: &'static str,
) -> Result<PipelineGraph> {
    let sink: Arc<dyn PhysicalOperator> = Arc::new(ResultSink {
        name: sink,
        children: vec![Arc::clone(root)],
    });
    PipelineGraph::try_new(sink)
}

/// Render the per-operator metrics of an execution, one line per operator.
pub fn format_execution_metrics(metrics: &ExecutionMetrics) -> String {
    let mut out = String::new();
    for (operator, metrics) in metrics.iter() {
        let _ = writeln!(
            out,
            "{}: output_rows={}, output_chunks={}, sink_rows={}, elapsed={:?}",
            operator.name(),
            metrics.output_rows(),
            metrics.output_chunks(),
            metrics.sink_rows(),
            Duration::from_nanos(metrics.elapsed_nanos())
        );
    }
    out
}

/// The metrics of a [`PipelineRootExec`]: `output_rows` of the root and the
/// total `elapsed_compute` of all operators of its executions.
pub fn root_metrics(exec: &PipelineRootExec) -> MetricsSet {
    exec.metrics_set().clone_inner()
}

/// Wrap the output stream of an execution so that it records the baseline
/// metrics of `exec`.
pub(crate) fn instrument_output_stream(
    exec: &PipelineRootExec,
    stream: PipelineStream,
) -> SendableRecordBatchStream {
    let baseline = BaselineMetrics::new(exec.metrics_set(), 0);
    Box::pin(InstrumentedStream {
        schema: RecordBatchStream::schema(&stream),
        execution: Arc::clone(stream.metrics()),
        inner: stream,
        baseline,
        reported: false,
    })
}

struct InstrumentedStream {
    schema: SchemaRef,
    inner: PipelineStream,
    execution: Arc<ExecutionMetrics>,
    baseline: BaselineMetrics,
    reported: bool,
}

impl InstrumentedStream {
    fn report_elapsed(&mut self) {
        if self.reported {
            return;
        }
        self.reported = true;
        let nanos: u64 = self
            .execution
            .iter()
            .map(|(_, metrics)| metrics.elapsed_nanos())
            .sum();
        self.baseline
            .elapsed_compute()
            .add_duration(Duration::from_nanos(nanos));
    }
}

impl Stream for InstrumentedStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let poll = this.inner.poll_next_unpin(cx);
        if matches!(poll, Poll::Ready(None | Some(Err(_)))) {
            this.report_elapsed();
        }
        this.baseline.record_poll(poll)
    }
}

impl RecordBatchStream for InstrumentedStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Placeholder sink used to render the pipelines of an operator tree.
#[derive(Debug)]
struct ResultSink {
    name: &'static str,
    children: Vec<Arc<dyn PhysicalOperator>>,
}

impl PhysicalOperator for ResultSink {
    fn name(&self) -> &str {
        self.name
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn is_sink(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_physical_plan::{ExecutionPlan, collect, displayable};

    use crate::bridge::BridgeOperator;

    fn root() -> Arc<PipelineRootExec> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let input: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        Arc::new(
            PipelineRootExec::try_new(
                Arc::new(BridgeOperator::new(Arc::clone(&input))),
                input,
            )
            .unwrap(),
        )
    }

    #[test]
    fn display_shows_the_operator_tree() {
        let root = root();
        let display = displayable(root.as_ref()).indent(false).to_string();
        assert_eq!(
            display,
            "PipelineRootExec: pipelines=1, bridged_plans=1\n  \
             BridgeOperator, pipelines=[0]\n    \
             DataSourceExec: partitions=1, partition_sizes=[1]\n  \
             Pipeline 0: BridgeOperator -> CollectorSink\n",
        );
        assert_eq!(
            format_pipelines(root.operator()).unwrap(),
            "MetaPipeline 0 (sink: ResultSink)\n  Pipeline 0: BridgeOperator -> ResultSink\n"
        );
        let tree = displayable(root.as_ref()).tree_render().to_string();
        assert!(tree.contains("PipelineRootExec"), "{tree}");
        assert!(tree.contains("pipelines: 1"), "{tree}");
    }

    #[tokio::test]
    async fn metrics_of_an_execution() -> Result<()> {
        let root = root();
        let output = collect(
            Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            Arc::new(TaskContext::default()),
        )
        .await?;
        assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
        let metrics = root_metrics(&root);
        assert_eq!(metrics.output_rows(), Some(3));
        assert!(metrics.elapsed_compute().is_some());
        let rendered = format_execution_metrics(&root.execution_metrics().unwrap());
        assert!(
            rendered.starts_with("BridgeOperator: output_rows=3, output_chunks=1"),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("CollectorSink: output_rows=0, output_chunks=0, sink_rows=3")
        );

        // Once executed, the display shows the metrics of every operator,
        // and of the bridged plan.
        let display = DisplayableExecutionPlan::with_metrics(root.as_ref())
            .indent(false)
            .to_string();
        let lines = display.lines().collect::<Vec<_>>();
        assert!(
            lines[1].starts_with(
                "  BridgeOperator, pipelines=[0], metrics=[output_rows=3, output_chunks=1, elapsed_compute="
            ),
            "{display}"
        );
        assert!(
            lines[2].starts_with(
                "    DataSourceExec: partitions=1, partition_sizes=[1], metrics=["
            ),
            "{display}"
        );
        assert!(
            lines[3].starts_with(
                "  Pipeline 0: BridgeOperator -> CollectorSink, metrics=[output_rows=3"
            ),
            "{display}"
        );
        Ok(())
    }
}
