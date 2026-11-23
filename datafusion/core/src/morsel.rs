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
use datafusion_physical_plan::coalesce::LimitedBatchCoalescer;
use datafusion_physical_plan::execution_plan::{
    Boundedness, EmissionType, SchedulingType, apply_expression_roots,
};
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
use parking_lot::Mutex;
use tokio::sync::mpsc;

use crate::datasource::{MemTable, source_as_provider};
use crate::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};

mod count;
mod fallback;
mod metrics;
#[cfg(feature = "parquet")]
mod parquet;

use count::{CountSink, CountState, FinalizedCount};
pub use metrics::MorselMetrics;
use metrics::{RunMetrics, WorkerMetrics};
#[cfg(feature = "parquet")]
use parquet::{ParquetScan, ReadyReaders};
#[cfg(test)]
mod tests;

const BATCHES_PER_MORSEL: usize = 4;
const COUNT_RESULT_CAPACITY: usize = 2;
const MAX_RESULT_CAPACITY: usize = 32;

fn result_capacity(workers: NonZeroUsize) -> usize {
    workers.get().saturating_mul(2).min(MAX_RESULT_CAPACITY)
}

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
        let mut source = if let Some(table) = provider.downcast_ref::<MemTable>() {
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
        #[cfg(feature = "parquet")]
        if let Source::Parquet(ref mut parquet_scan) = source {
            parquet_scan.pushdown_operators(
                &mut operators,
                count.is_some(),
                &scan.filters,
                &scan.projected_schema,
                session,
            )?;
        }
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
        let sink_capacity = match sink {
            Sink::Result => result_capacity(self.workers),
            Sink::Count { .. } => COUNT_RESULT_CAPACITY,
        };
        Ok(Lowering::Native(MorselExec::new(
            Pipeline {
                source,
                operators,
                workers: self.workers,
                sink,
                sink_capacity,
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
    batch: RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
    apply: impl FnOnce(RecordBatch, &BooleanArray) -> Result<T>,
) -> Result<T> {
    let mask = predicate.evaluate(&batch)?.into_array(batch.num_rows())?;
    let Ok(mask) = as_boolean_array(&mask) else {
        return internal_err!("Cannot create filter_array from non-boolean predicates");
    };
    apply(batch, mask)
}

fn filter_batch(
    batch: &RecordBatch,
    mask: &BooleanArray,
    projection: Option<&[usize]>,
) -> Result<RecordBatch> {
    match projection {
        Some(columns) if mask.len() == batch.num_rows() => {
            Ok(filter_record_batch(&batch.project(columns)?, mask)?)
        }
        // Validate mismatched masks on the original batch, even if no columns are kept.
        Some(columns) => Ok(filter_record_batch(batch, mask)?.project(columns)?),
        None => Ok(filter_record_batch(batch, mask)?),
    }
}

fn apply_operators(
    operators: &[Operator],
    mut batch: RecordBatch,
) -> Result<RecordBatch> {
    for operator in operators {
        batch = match operator {
            Operator::Filter {
                predicate,
                projection,
            } => with_filter_mask(batch, predicate, |batch, mask| {
                filter_batch(&batch, mask, projection.as_deref())
            })?,
            Operator::Projection(projector) => projector.project_batch(&batch)?,
        };
        if batch.num_rows() == 0 {
            break;
        }
    }
    Ok(batch)
}

struct ResultBuffer {
    batch_size: usize,
    filters: Vec<Option<LimitedBatchCoalescer>>,
    output: Option<RecordBatch>,
    next_to_finish: usize,
}

impl ResultBuffer {
    fn new(operators: &[Operator], batch_size: usize) -> Self {
        let slots = operators
            .iter()
            .rposition(|op| matches!(op, Operator::Filter { .. }));
        Self {
            batch_size,
            filters: std::iter::repeat_with(|| None)
                .take(slots.map_or(0, |i| i + 1))
                .collect(),
            output: None,
            next_to_finish: 0,
        }
    }

    fn push_from(
        &mut self,
        operators: &[Operator],
        start: usize,
        mut batch: RecordBatch,
    ) -> Result<()> {
        for (index, op) in operators.iter().enumerate().skip(start) {
            match op {
                Operator::Filter {
                    predicate,
                    projection,
                } => {
                    return with_filter_mask(batch, predicate, |batch, mask| {
                        let (batch, mask) = if mask.len() == batch.num_rows() {
                            let batch = match projection {
                                Some(columns) => batch.project(columns)?,
                                None => batch,
                            };
                            (batch, Some(mask))
                        } else {
                            (filter_batch(&batch, mask, projection.as_deref())?, None)
                        };
                        let coalescer = self.filters[index].get_or_insert_with(|| {
                            LimitedBatchCoalescer::new(
                                batch.schema(),
                                self.batch_size,
                                None,
                            )
                        });
                        if let Some(mask) = mask {
                            coalescer.push_batch_with_filter(batch, mask)?;
                        } else {
                            coalescer.push_batch(batch)?;
                        }
                        Ok(())
                    });
                }
                Operator::Projection(projector) => {
                    batch = projector.project_batch(&batch)?
                }
            }
            if batch.num_rows() == 0 {
                return Ok(());
            }
        }
        if batch.num_rows() != 0 {
            debug_assert!(self.output.is_none());
            self.output = Some(batch);
        }
        Ok(())
    }

    fn next(&mut self, operators: &[Operator], eof: bool) -> Result<Option<RecordBatch>> {
        loop {
            if let Some(batch) = self.output.take() {
                return Ok(Some(batch));
            }
            // Drain downstream before feeding it again, so completed batches cannot pile up
            // behind a blocked send. At EOF finish upstream first to feed downstream tails.
            let completed =
                self.filters
                    .iter_mut()
                    .enumerate()
                    .rev()
                    .find_map(|(index, slot)| {
                        Some((index, slot.as_mut()?.next_completed_batch()?))
                    });
            if let Some((index, batch)) = completed {
                self.push_from(operators, index + 1, batch)?;
            } else if eof && self.next_to_finish < self.filters.len() {
                if let Some(coalescer) = &mut self.filters[self.next_to_finish] {
                    coalescer.finish()?;
                }
                self.next_to_finish += 1;
            } else {
                return Ok(None);
            }
        }
    }
}

#[derive(Debug)]
enum Sink {
    Result,
    Count {
        sink: CountSink,
        result_operators: Vec<Operator>,
    },
}

enum WorkerSink {
    Result(ResultBuffer),
    Count(CountState),
}

type WorkerTail = (Vec<RecordBatch>, WorkerMetrics);

#[derive(Clone)]
struct WorkerOutput {
    tx: mpsc::Sender<RecordBatch>,
    batch_size: usize,
    tails: Arc<Mutex<Vec<WorkerTail>>>,
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

    fn inline_memory_stream(
        self: Arc<Self>,
        batch_size: usize,
        worker: WorkerMetrics,
    ) -> impl Stream<Item = Result<RecordBatch>> {
        let buffer = ResultBuffer::new(&self.operators, batch_size);
        stream::try_unfold(
            (self, 0usize, worker, buffer),
            |(pipeline, mut next, mut worker, mut buffer)| async move {
                let source = match &pipeline.source {
                    Source::Memory(source) => source,
                    #[cfg(feature = "parquet")]
                    Source::Parquet(_) => unreachable!(),
                };
                loop {
                    let output = worker.process(|| {
                        buffer.next(&pipeline.operators, next == source.batches.len())
                    })?;
                    if let Some(batch) = output {
                        return Ok(Some((batch, (pipeline, next, worker, buffer))));
                    }
                    let Some(input) = source.batches.get(next) else {
                        return Ok(None);
                    };
                    // A drain can immediately poll again, and a filter can consume
                    // input without emitting anything. Budget by input, not output.
                    if next != 0 && next.is_multiple_of(BATCHES_PER_MORSEL) {
                        tokio::task::yield_now().await;
                    }
                    if next.is_multiple_of(BATCHES_PER_MORSEL) {
                        worker.claim();
                    }
                    next += 1;
                    worker.process(|| {
                        buffer.push_from(&pipeline.operators, 0, source.read(input)?)
                    })?;
                    if next.is_multiple_of(BATCHES_PER_MORSEL)
                        || next == source.batches.len()
                    {
                        worker.complete();
                    }
                }
            },
        )
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

    fn worker_sink(&self, batch_size: usize) -> WorkerSink {
        match self.sink {
            Sink::Result => {
                WorkerSink::Result(ResultBuffer::new(&self.operators, batch_size))
            }
            Sink::Count { .. } => WorkerSink::Count(CountState::default()),
        }
    }

    async fn run_memory_worker(
        &self,
        source: &MemorySource,
        next: &AtomicUsize,
        output: WorkerOutput,
        mut worker: WorkerMetrics,
    ) -> Result<Option<CountState>> {
        let mut sink = self.worker_sink(output.batch_size);
        let tx = &output.tx;
        let count_filter = if matches!(sink, WorkerSink::Count(_)) {
            self.memory_count_filter()
        } else {
            None
        };
        while !tx.is_closed() {
            let start = next.fetch_add(BATCHES_PER_MORSEL, Ordering::Relaxed);
            if start >= source.batches.len() {
                break;
            }
            worker.claim();
            let end = (start + BATCHES_PER_MORSEL).min(source.batches.len());
            for (index, input) in (start..end).zip(&source.batches[start..end]) {
                if tx.is_closed() {
                    return Ok(None);
                }
                let batch = worker.process(|| source.read(input))?;
                if let (WorkerSink::Count(local), Some(predicate)) =
                    (&mut sink, count_filter)
                {
                    worker.process(|| {
                        with_filter_mask(batch, predicate, |batch, mask| {
                            let rows = if mask.len() == batch.num_rows() {
                                mask.true_count()
                            } else {
                                filter_record_batch(&batch, mask)?.num_rows()
                            };
                            local.add_rows(rows)
                        })
                    })?;
                } else if !self.consume(batch, tx, &mut sink, &mut worker).await? {
                    return Ok(None);
                }
                if index + 1 == end {
                    worker.complete();
                }
                tokio::task::yield_now().await;
            }
        }
        self.finish_worker(sink, &output, worker)
    }

    #[cfg(feature = "parquet")]
    async fn run_parquet_worker(
        &self,
        ready: ReadyReaders,
        output: WorkerOutput,
        mut worker: WorkerMetrics,
    ) -> Result<Option<CountState>> {
        let mut sink = self.worker_sink(output.batch_size);
        let tx = &output.tx;
        while let Some(morsel) = worker.ready(ready.claim()).await {
            worker.claim();
            let mut reader = worker.decode(|| morsel.into_stream());
            while let Some(batch) = metrics::decode(&mut worker, reader.next()).await {
                if tx.is_closed()
                    || !self.consume(batch?, tx, &mut sink, &mut worker).await?
                {
                    return Ok(None);
                }
                tokio::task::yield_now().await;
            }
            worker.complete();
            tokio::task::yield_now().await;
        }
        self.finish_worker(sink, &output, worker)
    }

    async fn consume(
        &self,
        batch: RecordBatch,
        tx: &mpsc::Sender<RecordBatch>,
        sink: &mut WorkerSink,
        worker: &mut WorkerMetrics,
    ) -> Result<bool> {
        match sink {
            WorkerSink::Count(local) => worker.process(|| {
                local.add_rows(apply_operators(&self.operators, batch)?.num_rows())?;
                Ok(true)
            }),
            WorkerSink::Result(buffer) => {
                let mut output = worker.process(|| {
                    buffer.push_from(&self.operators, 0, batch)?;
                    buffer.next(&self.operators, false)
                })?;
                while let Some(batch) = output {
                    if !worker.send(tx, batch).await {
                        return Ok(false);
                    }
                    output = worker.process(|| buffer.next(&self.operators, false))?;
                }
                Ok(true)
            }
        }
    }

    fn finish_worker(
        &self,
        sink: WorkerSink,
        output: &WorkerOutput,
        mut worker: WorkerMetrics,
    ) -> Result<Option<CountState>> {
        let mut buffer = match sink {
            WorkerSink::Count(local) => return Ok(Some(local)),
            WorkerSink::Result(buffer) => buffer,
        };
        let mut tail = Vec::new();
        while !output.tx.is_closed() {
            let Some(batch) = worker.process(|| buffer.next(&self.operators, true))?
            else {
                if !tail.is_empty() {
                    output.tails.lock().push((tail, worker));
                }
                break;
            };
            tail.push(batch);
        }
        Ok(None)
    }

    async fn run_count_result(
        finalized: FinalizedCount,
        operators: &[Operator],
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        if !tx.is_closed() {
            let batch = apply_operators(operators, finalized.into_batch())?;
            if let Some(run) = measurement {
                metrics::send(run, None, &tx, batch).await;
            } else {
                let _ = tx.send(batch).await;
            }
        }
        Ok(())
    }

    fn start(
        self: Arc<Self>,
        context: Arc<TaskContext>,
        metrics: ExecutionPlanMetricsSet,
        measurement: Option<Arc<RunMetrics>>,
    ) -> PipelineRun {
        let (tx, rx) = mpsc::channel(self.sink_capacity);
        let output = WorkerOutput {
            tx,
            batch_size: context.session_config().batch_size(),
            tails: Arc::default(),
        };
        #[cfg(not(feature = "parquet"))]
        drop(context);
        // Supervise independently of the consumer: an error must abort peers
        // even when they are waiting for space and the stream isn't polled.
        let worker_measurement = measurement.clone();
        let supervisor = SpawnedTask::spawn(async move {
            let measurement = worker_measurement;
            let mut tasks = JoinSet::new();
            match &self.source {
                Source::Memory(source) => {
                    let next = Arc::new(AtomicUsize::new(0));
                    let morsels = source.batches.len().div_ceil(BATCHES_PER_MORSEL);
                    for id in 0..self.workers.get().min(morsels) {
                        let (pipeline, next, output) =
                            (Arc::clone(&self), Arc::clone(&next), output.clone());
                        let worker = WorkerMetrics::new(measurement.clone(), id);
                        tasks.spawn(async move {
                            let source = match &pipeline.source {
                                Source::Memory(source) => source,
                                #[cfg(feature = "parquet")]
                                Source::Parquet(_) => unreachable!(),
                            };
                            pipeline
                                .run_memory_worker(source, &next, output, worker)
                                .await
                        });
                    }
                }
                #[cfg(feature = "parquet")]
                Source::Parquet(source) => {
                    let ready = source.start(
                        &context,
                        &mut tasks,
                        self.workers,
                        measurement.clone(),
                    )?;
                    for id in 0..self.workers.get() {
                        let (pipeline, ready, output) =
                            (Arc::clone(&self), ready.clone(), output.clone());
                        let worker = WorkerMetrics::new(measurement.clone(), id);
                        tasks.spawn(async move {
                            pipeline.run_parquet_worker(ready, output, worker).await
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
            // A worker's EOF is not necessarily a clean pipeline EOF. Publish its
            // buffered tail only after all workers and any scan producers succeed.
            let tails = std::mem::take(&mut *output.tails.lock());
            let tx = output.tx;
            for (batches, mut worker) in tails {
                for batch in batches {
                    if !worker.send(&tx, batch).await {
                        return Ok(());
                    }
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
                Self::run_count_result(
                    finalized,
                    result_operators,
                    tx,
                    measurement.as_deref(),
                )
                .await?;
            }
            Ok(())
        });
        PipelineRun {
            rx,
            supervisor: Some(supervisor),
            measurement,
        }
    }
}

struct PipelineRun {
    rx: mpsc::Receiver<RecordBatch>,
    supervisor: Option<SpawnedTask<Result<()>>>,
    measurement: Option<Arc<RunMetrics>>,
}

impl PipelineRun {
    async fn next(&mut self) -> Result<Option<RecordBatch>> {
        let output = self.measurement.as_ref().map(|run| &run.output);
        if let Some(task) = &mut self.supervisor {
            tokio::select! {
                biased;
                status = task.join_unwind_mut() => {
                    status.map_err(|err| exec_datafusion_err!("Morsel supervisor failed: {err}"))??;
                    self.supervisor = None;
                }
                Some(batch) = metrics::recv(output, &mut self.rx) => return Ok(Some(batch)),
            }
        }
        Ok(metrics::recv(output, &mut self.rx).await)
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
        #[cfg(feature = "parquet")]
        if let Source::Parquet(source) = &self.pipeline.source
            && source.apply_expressions(f)? == TreeNodeRecursion::Stop
        {
            return Ok(TreeNodeRecursion::Stop);
        }
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
        let collector = context.session_config().get_extension::<MorselMetrics>();
        if pipeline.is_inline_memory_stream() {
            let stream = stream::once(async move {
                pipeline.inline_memory_stream(
                    context.session_config().batch_size(),
                    WorkerMetrics::new(collector.map(RunMetrics::start), 0),
                )
            })
            .flatten();
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                stream,
            )));
        }
        let metrics = self.metrics.clone();
        let stream = stream::once(async move {
            let measurement = collector.map(RunMetrics::start);
            stream::try_unfold(
                pipeline.start(context, metrics, measurement),
                |mut run| async move { Ok(run.next().await?.map(|batch| (batch, run))) },
            )
        })
        .flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}
