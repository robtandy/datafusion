// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Experimental, opt-in morsel execution for in-memory and Parquet queries.

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{BooleanArray, RecordBatch};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::config::MorselMode;
use datafusion_common::nested_struct::adapt_batch_to_schema;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Result, exec_datafusion_err, internal_err, plan_err, project_schema,
};
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::expr::physical_name;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_physical_expr::aggregate::LoweredAggregateBuilder;
use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs, Projector};
use datafusion_physical_expr::{
    EquivalenceProperties, PhysicalExpr, create_physical_expr,
};
use datafusion_physical_plan::execution_plan::{
    Boundedness, EmissionType, SchedulingType, apply_expression_roots,
};
use datafusion_physical_plan::filter::batch_filter;
use datafusion_physical_plan::metrics::{
    ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions,
};
use datafusion_session::{QueryPlanner, Session};
use futures::stream::{self, Stream, StreamExt};
use tokio::sync::mpsc;

use crate::datasource::{MemTable, source_as_provider};
use crate::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};

mod count;
mod fallback;
#[cfg(feature = "parquet")]
mod parquet;

use count::{CountSink, CountState, FinalizedCount};
#[cfg(feature = "parquet")]
use parquet::{ParquetScan, ReadyReaders};
#[cfg(test)]
mod tests;

const BATCHES_PER_MORSEL: usize = 4;
const RESULT_CAPACITY: usize = 2;

/// Opt-in planner for an experimental, unordered pipeline over [`MemTable`]
/// and supported Parquet listing scans (with the `parquet` feature).
/// Lowers scans, filters, and projections with synchronous scalar expressions,
/// and ungrouped `COUNT(*)` over filtered input. Installing this planner
/// explicitly opts in even when the session mode is off. Unsupported queries
/// use [`DefaultPhysicalPlanner`] unless the mode is
/// [`MorselMode::Required`], which produces a planning error instead.
/// This experiment does not provide a stable scheduling API.
///
/// ```
/// use std::num::NonZeroUsize;
/// use std::sync::Arc;
/// use datafusion::execution::{SessionStateBuilder, context::SessionContext};
/// use datafusion::morsel::MorselQueryPlanner;
///
/// let state = SessionStateBuilder::new_with_default_features()
///     .with_query_planner(Arc::new(MorselQueryPlanner::new(NonZeroUsize::new(4).unwrap())))
///     .build();
/// let ctx = SessionContext::new_with_state(state);
/// ```
#[derive(Debug)]
pub struct MorselQueryPlanner {
    workers: NonZeroUsize,
}

enum Lowering {
    Native(MorselExec),
    Unsupported(String),
}

impl MorselQueryPlanner {
    /// Set the maximum number of workers per execution of a native pipeline.
    pub fn new(workers: NonZeroUsize) -> Self {
        Self { workers }
    }

    async fn try_lower(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Lowering> {
        let mut nodes = Vec::new();
        let mut count = None;
        let mut node = plan;
        let scan = loop {
            match node {
                LogicalPlan::Filter(filter) => {
                    if !supported_expr(&filter.predicate)? {
                        return Ok(Lowering::Unsupported(format!(
                            "unsupported filter predicate: {}",
                            filter.predicate
                        )));
                    }
                    nodes.push(node);
                    node = &filter.input;
                }
                LogicalPlan::Projection(projection) => {
                    for expr in &projection.expr {
                        if !supported_expr(expr)? {
                            return Ok(Lowering::Unsupported(format!(
                                "unsupported projection expression: {expr}"
                            )));
                        }
                    }
                    nodes.push(node);
                    node = &projection.input;
                }
                LogicalPlan::Aggregate(aggregate) if count.is_none() => {
                    let Some(expr) = count::count_star(aggregate) else {
                        return Ok(Lowering::Unsupported(
                            "only an ungrouped COUNT(*) over filtered input is supported"
                                .into(),
                        ));
                    };
                    if !nodes
                        .iter()
                        .all(|node| matches!(node, LogicalPlan::Projection(_)))
                    {
                        return Ok(Lowering::Unsupported(
                            "only projection is supported above COUNT(*)".into(),
                        ));
                    }
                    count = Some((aggregate, expr, std::mem::take(&mut nodes)));
                    node = &aggregate.input;
                }
                LogicalPlan::TableScan(scan) => break scan,
                LogicalPlan::Limit(_) => {
                    return Ok(Lowering::Unsupported(format!(
                        "native morsels do not support LIMIT/OFFSET: {}",
                        node.display()
                    )));
                }
                _ => {
                    return Ok(Lowering::Unsupported(format!(
                        "unsupported logical operator: {}",
                        node.display()
                    )));
                }
            }
        };
        if count.is_some()
            && !nodes
                .iter()
                .any(|node| matches!(node, LogicalPlan::Filter(_)))
        {
            return Ok(Lowering::Unsupported(
                "COUNT(*) requires a filter on its input".into(),
            ));
        }
        for (present, feature) in [
            (scan.fetch.is_some(), "LIMIT"),
            (scan.skip.is_some(), "OFFSET"),
            (!scan.statistics_requests.is_empty(), "statistics requests"),
        ] {
            if present {
                return Ok(Lowering::Unsupported(format!(
                    "native morsel scan of {} does not support {feature}",
                    scan.table_name
                )));
            }
        }
        let Ok(provider) = source_as_provider(&scan.source) else {
            return Ok(Lowering::Unsupported(format!(
                "unsupported table source for {}",
                scan.table_name
            )));
        };
        let source = if let Some(table) = provider.downcast_ref::<MemTable>() {
            if !table.sort_order.lock().is_empty() || !scan.filters.is_empty() {
                return Ok(Lowering::Unsupported(format!(
                    "MemTable scan of {} has advertised ordering or scan filters",
                    scan.table_name
                )));
            }
            // Like MemTable::scan, snapshot the batch handles at planning time.
            let mut batches = Vec::new();
            for partition in &table.batches {
                batches.extend(partition.read().await.iter().cloned());
            }
            Source::Memory(MemorySource {
                table: scan.table_name.to_string(),
                batches,
                projection: scan.projection.clone(),
                schema: project_schema(&provider.schema(), scan.projection.as_deref())?,
            })
        } else {
            #[cfg(feature = "parquet")]
            if let Some(table) =
                provider.downcast_ref::<crate::datasource::listing::ListingTable>()
            {
                match ParquetScan::lower(table, scan, session).await? {
                    Ok(source) => Source::Parquet(source),
                    Err(reason) => return Ok(Lowering::Unsupported(reason)),
                }
            } else {
                return Ok(unsupported_provider(&scan.table_name));
            }
            #[cfg(not(feature = "parquet"))]
            return Ok(unsupported_provider(&scan.table_name));
        };
        let (mut operators, mut schema) =
            Self::lower_operators(nodes, Arc::clone(source.schema()), session)?;
        if self.workers == NonZeroUsize::MIN
            && count.is_none()
            && matches!(source, Source::Memory(_))
        {
            prune_filter_input(&mut operators, source.schema());
        }
        let sink = if let Some((aggregate, expr, result_nodes)) = count {
            let field = LoweredAggregateBuilder::new(
                expr,
                aggregate.input.schema(),
                &schema,
                session.execution_props(),
                &PhysicalPlanningContext::default(),
            )
            .build()?
            .aggregate
            .field();
            let count_schema = Arc::new(Schema::new_with_metadata(
                vec![field],
                schema.metadata().clone(),
            ));
            let sink = CountSink::new(Arc::clone(&count_schema));
            let (result_operators, result_schema) =
                Self::lower_operators(result_nodes, count_schema, session)?;
            schema = result_schema;
            Sink::Count {
                sink,
                result_operators,
            }
        } else {
            Sink::Result
        };
        Ok(Lowering::Native(MorselExec::new(
            Pipeline {
                source,
                operators,
                workers: self.workers,
                sink,
                sink_capacity: RESULT_CAPACITY,
            },
            schema,
        )))
    }

    fn lower_operators(
        nodes: Vec<&LogicalPlan>,
        mut schema: SchemaRef,
        session: &dyn Session,
    ) -> Result<(Vec<Operator>, SchemaRef)> {
        let mut operators = Vec::with_capacity(nodes.len());
        let props = session.execution_props();
        let planning_ctx = PhysicalPlanningContext::default();
        for node in nodes.into_iter().rev() {
            match node {
                LogicalPlan::Filter(filter) => {
                    let predicate = create_physical_expr(
                        &filter.predicate,
                        filter.input.schema(),
                        props,
                        &planning_ctx,
                    )?;
                    if predicate.data_type(&schema)? != DataType::Boolean {
                        return plan_err!(
                            "Morsel filter predicate must return Boolean: {predicate}"
                        );
                    }
                    operators.push(Operator::Filter {
                        predicate,
                        projection: None,
                    });
                }
                LogicalPlan::Projection(projection) => {
                    let input = projection.input.schema();
                    let expressions = projection
                        .expr
                        .iter()
                        .map(|expr| {
                            let alias = match expr {
                                Expr::Column(column) => schema
                                    .field(input.index_of_column(column)?)
                                    .name()
                                    .clone(),
                                _ => physical_name(expr)?,
                            };
                            Ok(ProjectionExpr {
                                expr: create_physical_expr(
                                    expr,
                                    input,
                                    props,
                                    &planning_ctx,
                                )?,
                                alias,
                            })
                        })
                        .collect::<Result<ProjectionExprs>>()?;
                    let projector = expressions.make_projector_with_schema_metadata(
                        &schema,
                        projection.schema.as_arrow(),
                    )?;
                    schema = Arc::clone(projector.output_schema());
                    operators.push(Operator::Projection(projector));
                }
                _ => unreachable!(),
            }
        }
        Ok((operators, schema))
    }
}

fn unsupported_provider(table: &datafusion_common::TableReference) -> Lowering {
    Lowering::Unsupported(format!("unsupported table provider for {table}"))
}

fn supported_expr(expr: &Expr) -> Result<bool> {
    use datafusion_functions::core::{
        file_row_index::FileRowIndexFunc, input_file_name::InputFileNameFunc,
    };

    Ok(!expr.exists(|expr| {
        Ok(match expr {
            Expr::ScalarFunction(fun) => {
                fun.func.as_async().is_some()
                    || fun.func.inner().is::<InputFileNameFunc>()
                    || fun.func.inner().is::<FileRowIndexFunc>()
            }
            Expr::ScalarSubquery(_)
            | Expr::Exists(_)
            | Expr::InSubquery(_)
            | Expr::SetComparison(_)
            | Expr::OuterReferenceColumn(..)
            | Expr::AggregateFunction(_)
            | Expr::WindowFunction(_)
            | Expr::GroupingSet(_)
            | Expr::Placeholder(_)
            | Expr::Unnest(_)
            | Expr::HigherOrderFunction(_)
            | Expr::Lambda(_)
            | Expr::LambdaVariable(_) => true,
            _ => false,
        })
    })?)
}

#[async_trait]
impl QueryPlanner for MorselQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let query = match logical_plan {
            LogicalPlan::Show(_) => {
                return DefaultPhysicalPlanner::default()
                    .create_physical_plan(logical_plan, session)
                    .await;
            }
            LogicalPlan::Explain(explain) if !explain.logical_optimization_succeeded => {
                return DefaultPhysicalPlanner::default()
                    .handle_explain(explain, session, None)
                    .await;
            }
            LogicalPlan::Explain(explain) => &explain.plan,
            LogicalPlan::Analyze(analyze) => &analyze.input,
            _ => logical_plan,
        };
        let exec: Arc<dyn ExecutionPlan> = match self.try_lower(query, session).await? {
            Lowering::Native(exec) => Arc::new(exec),
            Lowering::Unsupported(reason) => {
                if session.config_options().execution.morsel_mode == MorselMode::Required
                {
                    return plan_err!("morsel_mode=required: {reason}");
                }
                return fallback::create_physical_plan(logical_plan, session).await;
            }
        };
        match logical_plan {
            LogicalPlan::Explain(explain) => {
                DefaultPhysicalPlanner::default()
                    .handle_explain(explain, session, Some(&exec))
                    .await
            }
            LogicalPlan::Analyze(analyze) => Ok(DefaultPhysicalPlanner::build_analyze(
                analyze, session, exec,
            )),
            _ => Ok(exec),
        }
    }
}

#[derive(Debug)]
enum Operator {
    Filter {
        predicate: Arc<dyn PhysicalExpr>,
        projection: Option<Vec<usize>>,
    },
    Projection(Projector),
}

#[derive(Debug)]
enum Source {
    Memory(MemorySource),
    #[cfg(feature = "parquet")]
    Parquet(ParquetScan),
}

impl Source {
    fn schema(&self) -> &SchemaRef {
        match self {
            Self::Memory(source) => &source.schema,
            #[cfg(feature = "parquet")]
            Self::Parquet(source) => &source.schema,
        }
    }
}

#[derive(Debug)]
struct MemorySource {
    table: String,
    batches: Vec<RecordBatch>,
    projection: Option<Vec<usize>>,
    schema: SchemaRef,
}

impl MemorySource {
    fn read(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let batch = match &self.projection {
            Some(columns) => batch.project(columns)?,
            None => batch.clone(),
        };
        if batch.schema().as_ref() != self.schema.as_ref() {
            adapt_batch_to_schema(batch, &self.schema)
        } else {
            Ok(batch)
        }
    }
}

fn prune_filter_input(operators: &mut [Operator], schema: &Schema) {
    let [
        Operator::Filter { projection, .. },
        Operator::Projection(projector),
    ] = operators
    else {
        return;
    };
    let columns = projector.projection().column_indices();
    if columns.len() == schema.fields().len() {
        return;
    }
    // The predicate still reads the full input; only the selected rows are
    // materialized in the narrowed schema.
    let input_projection = ProjectionExprs::from_indices(&columns, schema);
    let remap = || {
        projector
            .projection()
            .clone()
            .try_map_exprs(|expr| input_projection.project_expr(&expr))?
            .make_projector_with_schema_metadata(
                &schema.project(&columns)?,
                projector.output_schema(),
            )
    };
    if let Ok(remapped) = remap()
        && remapped.output_schema() == projector.output_schema()
    {
        *projection = Some(columns);
        *projector = remapped;
    }
}

fn with_filter_mask<T>(
    batch: &RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
    apply: impl FnOnce(&BooleanArray) -> Result<T>,
) -> Result<T> {
    let mask = predicate.evaluate(batch)?.into_array(batch.num_rows())?;
    let Ok(mask) = as_boolean_array(&mask) else {
        return internal_err!("Cannot create filter_array from non-boolean predicates");
    };
    apply(mask)
}

fn apply_operators(
    operators: &[Operator],
    mut batch: RecordBatch,
) -> Result<RecordBatch> {
    for operator in operators {
        batch = match operator {
            Operator::Filter {
                predicate,
                projection: None,
            } => batch_filter(&batch, predicate)?,
            Operator::Filter {
                predicate,
                projection: Some(columns),
            } => with_filter_mask(&batch, predicate, |mask| {
                // Keep the original filter's handling of mismatched lengths,
                // including when the projection keeps no columns.
                if mask.len() != batch.num_rows() {
                    Ok(filter_record_batch(&batch, mask)?.project(columns)?)
                } else {
                    Ok(filter_record_batch(&batch.project(columns)?, mask)?)
                }
            })?,
            Operator::Projection(projector) => projector.project_batch(&batch)?,
        };
        if batch.num_rows() == 0 {
            break;
        }
    }
    Ok(batch)
}

#[derive(Debug)]
enum Sink {
    Result,
    Count {
        sink: CountSink,
        result_operators: Vec<Operator>,
    },
}

#[derive(Debug)]
struct Pipeline {
    source: Source,
    operators: Vec<Operator>,
    workers: NonZeroUsize,
    sink: Sink,
    sink_capacity: usize,
}

impl Pipeline {
    fn is_inline_memory_stream(&self) -> bool {
        self.workers == NonZeroUsize::MIN
            && matches!(self.source, Source::Memory(_))
            && matches!(self.sink, Sink::Result)
    }

    fn inline_memory_stream(self: Arc<Self>) -> impl Stream<Item = Result<RecordBatch>> {
        stream::try_unfold((self, 0usize), |(pipeline, mut next)| async move {
            let source = match &pipeline.source {
                Source::Memory(source) => source,
                #[cfg(feature = "parquet")]
                Source::Parquet(_) => unreachable!(),
            };
            while let Some(input) = source.batches.get(next) {
                // A drain can immediately poll again, and a filter can consume
                // input without emitting anything. Budget by input, not output.
                if next != 0 && next.is_multiple_of(BATCHES_PER_MORSEL) {
                    tokio::task::yield_now().await;
                }
                next += 1;
                let batch = apply_operators(&pipeline.operators, source.read(input)?)?;
                if batch.num_rows() != 0 {
                    return Ok(Some((batch, (pipeline, next))));
                }
            }
            Ok(None)
        })
    }

    fn memory_count_filter(&self) -> Option<&Arc<dyn PhysicalExpr>> {
        let [
            Operator::Filter {
                predicate,
                projection: None,
            },
            rest @ ..,
        ] = self.operators.as_slice()
        else {
            return None;
        };
        match rest {
            [] => Some(predicate),
            [Operator::Projection(projector)]
                if projector.projection().as_ref().is_empty() =>
            {
                Some(predicate)
            }
            _ => None,
        }
    }

    async fn run_memory_worker(
        &self,
        source: &MemorySource,
        next: &AtomicUsize,
        tx: mpsc::Sender<RecordBatch>,
    ) -> Result<Option<CountState>> {
        let mut count = matches!(self.sink, Sink::Count { .. }).then(CountState::default);
        let count_filter = count.and_then(|_| self.memory_count_filter());
        while !tx.is_closed() {
            let start = next.fetch_add(BATCHES_PER_MORSEL, Ordering::Relaxed);
            if start >= source.batches.len() {
                break;
            }
            let end = (start + BATCHES_PER_MORSEL).min(source.batches.len());
            for input in &source.batches[start..end] {
                if tx.is_closed() {
                    return Ok(None);
                }
                let batch = source.read(input)?;
                if let (Some(local), Some(predicate)) = (&mut count, count_filter) {
                    let rows = with_filter_mask(&batch, predicate, |mask| {
                        if mask.len() == batch.num_rows() {
                            Ok(mask.true_count())
                        } else {
                            Ok(filter_record_batch(&batch, mask)?.num_rows())
                        }
                    })?;
                    local.add_rows(rows)?;
                    tokio::task::yield_now().await;
                } else if !self.consume(batch, &tx, &mut count).await? {
                    return Ok(None);
                }
            }
        }
        Ok(count)
    }

    #[cfg(feature = "parquet")]
    async fn run_parquet_worker(
        &self,
        ready: ReadyReaders,
        tx: mpsc::Sender<RecordBatch>,
    ) -> Result<Option<CountState>> {
        let mut count = matches!(self.sink, Sink::Count { .. }).then(CountState::default);
        while let Some(morsel) = ready.claim().await {
            let mut reader = morsel.into_stream();
            while let Some(batch) = reader.next().await {
                if tx.is_closed() || !self.consume(batch?, &tx, &mut count).await? {
                    return Ok(None);
                }
            }
            tokio::task::yield_now().await;
        }
        Ok(count)
    }

    async fn consume(
        &self,
        batch: RecordBatch,
        tx: &mpsc::Sender<RecordBatch>,
        count: &mut Option<CountState>,
    ) -> Result<bool> {
        let batch = apply_operators(&self.operators, batch)?;
        if let Some(local) = count {
            local.add_rows(batch.num_rows())?;
        } else if batch.num_rows() > 0 && tx.send(batch).await.is_err() {
            return Ok(false);
        }
        tokio::task::yield_now().await;
        Ok(true)
    }

    async fn run_count_result(
        finalized: FinalizedCount,
        operators: &[Operator],
        tx: mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if !tx.is_closed() {
            let batch = apply_operators(operators, finalized.into_batch())?;
            let _ = tx.send(batch).await;
        }
        Ok(())
    }

    fn start(
        self: Arc<Self>,
        context: Arc<TaskContext>,
        metrics: ExecutionPlanMetricsSet,
    ) -> PipelineRun {
        #[cfg(not(feature = "parquet"))]
        let _ = context;
        let (tx, rx) = mpsc::channel(self.sink_capacity);
        // Supervise independently of the consumer: an error must abort peers
        // even when they are waiting for space and the stream isn't polled.
        let supervisor = SpawnedTask::spawn(async move {
            let mut tasks = JoinSet::new();
            match &self.source {
                Source::Memory(source) => {
                    let next = Arc::new(AtomicUsize::new(0));
                    let morsels = source.batches.len().div_ceil(BATCHES_PER_MORSEL);
                    for _ in 0..self.workers.get().min(morsels) {
                        let (pipeline, next, tx) =
                            (Arc::clone(&self), Arc::clone(&next), tx.clone());
                        tasks.spawn(async move {
                            let source = match &pipeline.source {
                                Source::Memory(source) => source,
                                #[cfg(feature = "parquet")]
                                Source::Parquet(_) => unreachable!(),
                            };
                            pipeline.run_memory_worker(source, &next, tx).await
                        });
                    }
                }
                #[cfg(feature = "parquet")]
                Source::Parquet(source) => {
                    let ready = source.start(&context, &mut tasks)?;
                    for _ in 0..self.workers.get() {
                        let (pipeline, ready, tx) =
                            (Arc::clone(&self), ready.clone(), tx.clone());
                        tasks.spawn(async move {
                            pipeline.run_parquet_worker(ready, tx).await
                        });
                    }
                }
            }
            let mut combined = CountState::default();
            while let Some(result) = tasks.join_next().await {
                let local = match result {
                    Ok(result) => result?,
                    Err(err) if err.is_panic() => {
                        std::panic::resume_unwind(err.into_panic())
                    }
                    Err(err) => {
                        return Err(exec_datafusion_err!("Morsel task failed: {err}"));
                    }
                };
                if let Some(local) = local {
                    combined.combine(local)?;
                }
            }
            if let Sink::Count {
                sink,
                result_operators,
            } = &self.sink
                && !tx.is_closed()
            {
                // Run the dependent pipeline only on successful finalization. It
                // has no independent task to wake on error or cancellation.
                #[cfg(test)]
                if let Some(gate) = &sink.finalize_gate {
                    gate.wait().await;
                }
                let finalized = sink.finalize(combined)?;
                MetricBuilder::new(&metrics)
                    .counter("count_finalizations", 0)
                    .add(1);
                Self::run_count_result(finalized, result_operators, tx).await?;
            }
            Ok(())
        });
        PipelineRun {
            rx,
            supervisor: Some(supervisor),
        }
    }
}

struct PipelineRun {
    rx: mpsc::Receiver<RecordBatch>,
    supervisor: Option<SpawnedTask<Result<()>>>,
}

impl PipelineRun {
    async fn next(&mut self) -> Result<Option<RecordBatch>> {
        if let Some(task) = &mut self.supervisor {
            tokio::select! {
                biased;
                status = task.join_unwind_mut() => {
                    status.map_err(|err| exec_datafusion_err!("Morsel supervisor failed: {err}"))??;
                    self.supervisor = None;
                }
                Some(batch) = self.rx.recv() => return Ok(Some(batch)),
            }
        }
        Ok(self.rx.recv().await)
    }
}

#[derive(Debug)]
struct MorselExec {
    pipeline: Arc<Pipeline>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl MorselExec {
    fn new(pipeline: Pipeline, schema: SchemaRef) -> Self {
        let emission = match pipeline.sink {
            Sink::Result => EmissionType::Incremental,
            Sink::Count { .. } => EmissionType::Final,
        };
        let metrics = match &pipeline.source {
            Source::Memory(_) => ExecutionPlanMetricsSet::new(),
            #[cfg(feature = "parquet")]
            Source::Parquet(source) => source.metrics.clone(),
        };
        Self {
            metrics,
            pipeline: Arc::new(pipeline),
            properties: Arc::new(
                PlanProperties::new(
                    EquivalenceProperties::new(schema),
                    Partitioning::UnknownPartitioning(1),
                    emission,
                    Boundedness::Bounded,
                )
                .with_scheduling_type(SchedulingType::Cooperative),
            ),
        }
    }
}

impl DisplayAs for MorselExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pipeline = &self.pipeline;
        write!(f, "MorselExec: workers={}, ", pipeline.workers)?;
        match &pipeline.source {
            Source::Memory(source) => write!(
                f,
                "MemTable[{}, projection={:?}]",
                source.table, source.projection
            )?,
            #[cfg(feature = "parquet")]
            Source::Parquet(source) => write!(f, "NativeParquet[{}]", source.table)?,
        }
        display_operators(&pipeline.operators, f)?;
        if let Sink::Count {
            result_operators, ..
        } = &pipeline.sink
        {
            write!(f, " -> CountSink; FinalizedCountSource")?;
            display_operators(result_operators, f)?;
        }
        if pipeline.is_inline_memory_stream() {
            write!(f, " -> ResultSink[inline]")
        } else {
            write!(f, " -> ResultSink[capacity={}]", pipeline.sink_capacity)
        }
    }
}

fn display_operators(operators: &[Operator], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for operator in operators {
        match operator {
            Operator::Filter {
                predicate,
                projection,
            } => {
                write!(f, " -> Filter[{predicate}")?;
                if let Some(columns) = projection {
                    write!(f, ", projection={columns:?}")?;
                }
                write!(f, "]")?;
            }
            Operator::Projection(projector) => {
                write!(f, " -> {}", projector.projection())?
            }
        }
    }
    Ok(())
}

impl ExecutionPlan for MorselExec {
    fn name(&self) -> &'static str {
        "MorselExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let result_operators = match &self.pipeline.sink {
            Sink::Result => &[][..],
            Sink::Count {
                result_operators, ..
            } => result_operators.as_slice(),
        };
        apply_expression_roots(
            self.pipeline
                .operators
                .iter()
                .chain(result_operators)
                .flat_map(|operator| match operator {
                    Operator::Filter { predicate, .. } => {
                        itertools::Either::Left(std::iter::once(predicate))
                    }
                    Operator::Projection(projector) => itertools::Either::Right(
                        projector.projection().iter().map(|e| &e.expr),
                    ),
                }),
            f,
        )
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

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("MorselExec is a leaf and cannot accept children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!(
                "MorselExec has only partition 0, requested {partition}"
            );
        }
        let pipeline = Arc::clone(&self.pipeline);
        if pipeline.is_inline_memory_stream() {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                pipeline.inline_memory_stream(),
            )));
        }
        let metrics = self.metrics.clone();
        let stream = stream::once(async move {
            stream::try_unfold(pipeline.start(context, metrics), |mut run| async move {
                Ok(run.next().await?.map(|batch| (batch, run)))
            })
        })
        .flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}
