// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Experimental, opt-in morsel execution for in-memory queries.

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion_common::nested_struct::adapt_batch_to_schema;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Result, exec_datafusion_err, internal_err, plan_err, project_schema,
};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::expr::physical_name;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs, Projector};
use datafusion_physical_expr::{
    EquivalenceProperties, PhysicalExpr, create_physical_expr,
};
use datafusion_physical_plan::execution_plan::{
    Boundedness, EmissionType, SchedulingType, apply_expression_roots,
};
use datafusion_physical_plan::filter::batch_filter;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions,
};
use datafusion_session::{QueryPlanner, Session};
use futures::stream::{self, FuturesUnordered, StreamExt};
use tokio::sync::mpsc;

use crate::datasource::{MemTable, source_as_provider};
use crate::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};

#[cfg(test)]
mod tests;

const BATCHES_PER_MORSEL: usize = 4;
const RESULT_CAPACITY: usize = 2;

/// Opt-in planner for an experimental, unordered pipeline over [`MemTable`].
/// Only scans, filters, and projections with synchronous scalar expressions
/// are lowered. Queries requiring ordering or other operators use
/// [`DefaultPhysicalPlanner`]. This experiment does not provide a stable
/// scheduling API.
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

impl MorselQueryPlanner {
    /// Set the maximum number of workers per execution of a native pipeline.
    pub fn new(workers: NonZeroUsize) -> Self {
        Self { workers }
    }

    async fn try_lower(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Option<MorselExec>> {
        let mut nodes = Vec::new();
        let mut node = plan;
        let scan = loop {
            match node {
                LogicalPlan::Filter(filter) if supported_expr(&filter.predicate)? => {
                    nodes.push(node);
                    node = &filter.input;
                }
                LogicalPlan::Projection(projection) => {
                    for expr in &projection.expr {
                        if !supported_expr(expr)? {
                            return Ok(None);
                        }
                    }
                    nodes.push(node);
                    node = &projection.input;
                }
                LogicalPlan::TableScan(scan) => break scan,
                _ => return Ok(None),
            }
        };
        // MemTable does not push predicates into its scan; residual Filter nodes
        // above it are evaluated by the pipeline.
        if scan.fetch.is_some()
            || scan.skip.is_some()
            || !scan.filters.is_empty()
            || !scan.statistics_requests.is_empty()
        {
            return Ok(None);
        }
        let Ok(provider) = source_as_provider(&scan.source) else {
            return Ok(None);
        };
        let Some(table) = provider.downcast_ref::<MemTable>() else {
            return Ok(None);
        };
        if !table.sort_order.lock().is_empty() {
            return Ok(None);
        }

        let mut schema = project_schema(&provider.schema(), scan.projection.as_deref())?;
        let source_schema = Arc::clone(&schema);
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
                    operators.push(Operator::Filter(predicate));
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
        // Like MemTable::scan, snapshot the batch handles at planning time.
        let mut batches = Vec::new();
        for partition in &table.batches {
            batches.extend(partition.read().await.iter().cloned());
        }
        Ok(Some(MorselExec::new(
            Pipeline {
                source: MemorySource {
                    table: scan.table_name.to_string(),
                    batches,
                    projection: scan.projection.clone(),
                    schema: source_schema,
                },
                operators,
                workers: self.workers,
                sink_capacity: RESULT_CAPACITY,
            },
            schema,
        )))
    }
}

fn supported_expr(expr: &Expr) -> Result<bool> {
    Ok(!expr.exists(|expr| {
        Ok(match expr {
            Expr::ScalarFunction(fun) => fun.func.as_async().is_some(),
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
        match self.try_lower(logical_plan, session).await? {
            Some(exec) => Ok(Arc::new(exec)),
            None => {
                DefaultPhysicalPlanner::default()
                    .create_physical_plan(logical_plan, session)
                    .await
            }
        }
    }
}

#[derive(Debug)]
enum Operator {
    Filter(Arc<dyn PhysicalExpr>),
    Projection(Projector),
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

#[derive(Debug)]
struct Pipeline {
    source: MemorySource,
    operators: Vec<Operator>,
    workers: NonZeroUsize,
    sink_capacity: usize,
}

impl Pipeline {
    async fn run_worker(
        &self,
        next: &AtomicUsize,
        tx: mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        while !tx.is_closed() {
            let start = next.fetch_add(BATCHES_PER_MORSEL, Ordering::Relaxed);
            if start >= self.source.batches.len() {
                break;
            }
            let end = (start + BATCHES_PER_MORSEL).min(self.source.batches.len());
            for input in &self.source.batches[start..end] {
                if tx.is_closed() {
                    return Ok(());
                }
                let mut batch = self.source.read(input)?;
                for operator in &self.operators {
                    batch = match operator {
                        Operator::Filter(predicate) => batch_filter(&batch, predicate)?,
                        Operator::Projection(projector) => {
                            projector.project_batch(&batch)?
                        }
                    };
                    if batch.num_rows() == 0 {
                        break;
                    }
                }
                if batch.num_rows() > 0 && tx.send(batch).await.is_err() {
                    return Ok(());
                }
                tokio::task::yield_now().await;
            }
        }
        Ok(())
    }

    fn start(self: Arc<Self>) -> PipelineRun {
        let (tx, rx) = mpsc::channel(self.sink_capacity);
        // Supervise independently of the consumer: an error must abort peers
        // even when they are waiting for space and the stream isn't polled.
        let supervisor = SpawnedTask::spawn(async move {
            let next = Arc::new(AtomicUsize::new(0));
            let morsels = self.source.batches.len().div_ceil(BATCHES_PER_MORSEL);
            let mut tasks = FuturesUnordered::new();
            for _ in 0..self.workers.get().min(morsels) {
                if tx.is_closed() {
                    break;
                }
                let (pipeline, next, tx) =
                    (Arc::clone(&self), Arc::clone(&next), tx.clone());
                tasks.push(
                    SpawnedTask::spawn(
                        async move { pipeline.run_worker(&next, tx).await },
                    )
                    .join_unwind(),
                );
            }
            drop(tx);
            while let Some(result) = tasks.next().await {
                result.map_err(|err| {
                    exec_datafusion_err!("Morsel worker failed: {err}")
                })??;
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
}

impl MorselExec {
    fn new(pipeline: Pipeline, schema: SchemaRef) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
            properties: Arc::new(
                PlanProperties::new(
                    EquivalenceProperties::new(schema),
                    Partitioning::UnknownPartitioning(1),
                    EmissionType::Incremental,
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
        let source = &pipeline.source;
        write!(
            f,
            "MorselExec: workers={}, MemTable[{}, projection={:?}]",
            pipeline.workers, source.table, source.projection
        )?;
        for operator in &pipeline.operators {
            match operator {
                Operator::Filter(predicate) => write!(f, " -> Filter[{predicate}]")?,
                Operator::Projection(projector) => {
                    write!(f, " -> {}", projector.projection())?
                }
            }
        }
        write!(f, " -> ResultSink[capacity={}]", pipeline.sink_capacity)
    }
}

impl ExecutionPlan for MorselExec {
    fn name(&self) -> &'static str {
        "MorselExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(
            self.pipeline
                .operators
                .iter()
                .flat_map(|operator| match operator {
                    Operator::Filter(predicate) => {
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
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!(
                "MorselExec has only partition 0, requested {partition}"
            );
        }
        let pipeline = Arc::clone(&self.pipeline);
        let stream = stream::once(async move {
            stream::try_unfold(pipeline.start(), |mut run| async move {
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
