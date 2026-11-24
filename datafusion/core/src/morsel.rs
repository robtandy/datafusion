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
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::config::MorselMode;
use datafusion_common::nested_struct::adapt_batch_to_schema;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Result, exec_datafusion_err, internal_err, plan_err, project_schema,
};
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::expr::physical_name;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_physical_expr::aggregate::LoweredAggregateBuilder;
use datafusion_physical_expr::expressions::Column;
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
mod group_count;
mod metrics;
#[cfg(feature = "parquet")]
mod parquet;

use count::{CountSink, CountState, FinalizedCount};
use group_count::{GroupCountSink, GroupCountState, IntKeyType};
pub use metrics::MorselMetrics;
use metrics::{RunMetrics, WorkerMetrics};
#[cfg(feature = "parquet")]
use parquet::{ClaimedWork, ParquetScan, ReadyReaders};
#[cfg(test)]
mod tests;

const BATCHES_PER_MORSEL: usize = 4;
const COUNT_RESULT_CAPACITY: usize = 2;
const GROUP_COUNT_RESULT_CAPACITY: usize = 2;
const MAX_RESULT_CAPACITY: usize = 32;

fn result_capacity(workers: NonZeroUsize) -> usize {
    workers.get().saturating_mul(2).min(MAX_RESULT_CAPACITY)
}

/// Opt-in planner for an experimental, unordered pipeline over [`MemTable`]
/// and supported Parquet listing scans (with the `parquet` feature).
/// Lowers scans, filters, and projections with synchronous scalar expressions,
/// ungrouped `COUNT(*)` over filtered input, and `GROUP BY <integer column>, COUNT(*)`.
/// Installing this planner explicitly opts in even when the session mode is off.
/// Unsupported queries use [`DefaultPhysicalPlanner`] unless the mode is
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

enum LoweredAggregate<'a> {
    Count {
        aggregate: &'a datafusion_expr::logical_plan::Aggregate,
        count_expr: &'a Expr,
        result_nodes: Vec<&'a LogicalPlan>,
    },
    GroupCount {
        aggregate: &'a datafusion_expr::logical_plan::Aggregate,
        key_expr: &'a Expr,
        key_index: usize,
        key_type: IntKeyType,
        count_expr: &'a Expr,
        result_nodes: Vec<&'a LogicalPlan>,
    },
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
        let mut lowered_aggregate = None;
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
                LogicalPlan::Aggregate(aggregate) if lowered_aggregate.is_none() => {
                    let Some(count_expr) = count::count_star_aggr_expr(aggregate) else {
                        return Ok(Lowering::Unsupported(
                            "only COUNT(*) is supported in native morsel aggregates"
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
                    if aggregate.group_expr.is_empty() {
                        lowered_aggregate = Some(LoweredAggregate::Count {
                            aggregate,
                            count_expr,
                            result_nodes: std::mem::take(&mut nodes),
                        });
                    } else {
                        match group_count::group_by_integer_column(aggregate)? {
                            Ok(key) => {
                                lowered_aggregate = Some(LoweredAggregate::GroupCount {
                                    aggregate,
                                    key_expr: key.expr,
                                    key_index: key.column_index,
                                    key_type: key.key_type,
                                    count_expr,
                                    result_nodes: std::mem::take(&mut nodes),
                                });
                            }
                            Err(reason) => return Ok(Lowering::Unsupported(reason)),
                        }
                    }
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
        if matches!(lowered_aggregate, Some(LoweredAggregate::Count { .. }))
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
                matches!(lowered_aggregate, Some(LoweredAggregate::Count { .. })),
                &scan.filters,
                &scan.projected_schema,
                session,
            )?;
        }
        let prune_filter = match source {
            Source::Memory(_) => self.workers == NonZeroUsize::MIN,
            #[cfg(feature = "parquet")]
            Source::Parquet(_) => true,
        };
        if lowered_aggregate.is_none() && prune_filter {
            prune_filter_input(&mut operators, source.schema());
        }
        let sink = match lowered_aggregate {
            Some(LoweredAggregate::Count {
                aggregate,
                count_expr,
                result_nodes,
            }) => {
                let field = LoweredAggregateBuilder::new(
                    count_expr,
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
            }
            Some(LoweredAggregate::GroupCount {
                aggregate,
                key_expr,
                key_index,
                key_type,
                count_expr,
                result_nodes,
            }) => {
                let Some(input_field) = schema.fields().get(key_index) else {
                    return internal_err!(
                        "Morsel GROUP BY key index {key_index} out of bounds for schema"
                    );
                };
                let Some(physical_key_type) =
                    IntKeyType::from_data_type(input_field.data_type())
                else {
                    return Ok(Lowering::Unsupported(format!(
                        "unsupported GROUP BY key type {}: only a single integer column is supported",
                        input_field.data_type()
                    )));
                };
                if physical_key_type != key_type {
                    return internal_err!(
                        "Morsel GROUP BY key type mismatch between logical and physical schemas"
                    );
                }
                let key_name = physical_name(key_expr)?;
                let key_field = Arc::new(
                    Field::new(
                        &key_name,
                        input_field.data_type().clone(),
                        input_field.is_nullable(),
                    )
                    .with_metadata(input_field.metadata().clone()),
                );
                let count_field = LoweredAggregateBuilder::new(
                    count_expr,
                    aggregate.input.schema(),
                    &schema,
                    session.execution_props(),
                    &PhysicalPlanningContext::default(),
                )
                .build()?
                .aggregate
                .field();
                let agg_schema = Arc::new(Schema::new_with_metadata(
                    vec![key_field, count_field],
                    schema.metadata().clone(),
                ));
                let sink = GroupCountSink::new(
                    Arc::clone(&agg_schema),
                    key_index,
                    physical_key_type,
                    key_name,
                );
                let (result_operators, result_schema) =
                    Self::lower_operators(result_nodes, agg_schema, session)?;
                schema = result_schema;
                Sink::GroupCount {
                    sink,
                    result_operators,
                }
            }
            None => Sink::Result,
        };
        let sink_capacity = match sink {
            Sink::Result => result_capacity(self.workers),
            Sink::Count { .. } => COUNT_RESULT_CAPACITY,
            Sink::GroupCount { .. } => GROUP_COUNT_RESULT_CAPACITY,
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

    fn has_pushed_filter(&self) -> bool {
        match self {
            Self::Memory(_) => false,
            #[cfg(feature = "parquet")]
            Self::Parquet(source) => source.pushed_predicate().is_some(),
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
    if batch.num_columns() == 0 {
        count::validate_mask_length(batch, mask)?;
    }
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
    coalesce_source: bool,
    source_coalescer: Option<LimitedBatchCoalescer>,
    source_finished: bool,
    filters: Vec<Option<LimitedBatchCoalescer>>,
    output: Option<RecordBatch>,
    next_to_finish: usize,
}

impl ResultBuffer {
    fn new(operators: &[Operator], batch_size: usize, coalesce_source: bool) -> Self {
        let slots = operators
            .iter()
            .rposition(|op| matches!(op, Operator::Filter { .. }));
        Self {
            batch_size,
            coalesce_source,
            source_coalescer: None,
            source_finished: !coalesce_source,
            filters: std::iter::repeat_with(|| None)
                .take(slots.map_or(0, |i| i + 1))
                .collect(),
            output: None,
            next_to_finish: 0,
        }
    }

    fn push_input(&mut self, operators: &[Operator], batch: RecordBatch) -> Result<()> {
        if !self.coalesce_source {
            return self.push_from(operators, 0, batch);
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let coalescer = self.source_coalescer.get_or_insert_with(|| {
            LimitedBatchCoalescer::new(batch.schema(), self.batch_size, None)
        });
        coalescer.push_batch(batch)?;
        Ok(())
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
            } else if let Some(batch) = self
                .source_coalescer
                .as_mut()
                .and_then(LimitedBatchCoalescer::next_completed_batch)
            {
                self.push_from(operators, 0, batch)?;
            } else if eof && !self.source_finished {
                if let Some(coalescer) = &mut self.source_coalescer {
                    coalescer.finish()?;
                }
                self.source_finished = true;
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
    GroupCount {
        sink: GroupCountSink,
        result_operators: Vec<Operator>,
    },
}

enum WorkerSink {
    Result(Box<ResultBuffer>),
    Count(CountState),
    GroupCount(GroupCountState),
}

enum WorkerState {
    None,
    Count(CountState),
    GroupCount(GroupCountState),
}

type WorkerTail = (Vec<RecordBatch>, WorkerMetrics);

#[derive(Clone)]
struct WorkerOutput {
    tx: mpsc::Sender<RecordBatch>,
    batch_size: usize,
    tails: Arc<Mutex<Vec<WorkerTail>>>,
}

#[derive(Debug)]
enum CountStrategy<'a> {
    DirectRows,
    DirectMask(&'a Arc<dyn PhysicalExpr>),
    General,
}

#[derive(Debug)]
enum GroupCountStrategy<'a> {
    DirectColumn(usize),
    DirectMask {
        predicate: &'a Arc<dyn PhysicalExpr>,
        key_index: usize,
    },
    General(usize),
}

fn plain_column_index(projector: &Projector, output_index: usize) -> Option<usize> {
    let exprs = projector.projection().as_ref();
    if exprs.iter().all(|p| p.expr.is::<Column>()) {
        exprs
            .get(output_index)?
            .expr
            .downcast_ref::<Column>()
            .map(Column::index)
    } else {
        None
    }
}

async fn abort_and_drain<T: 'static>(tasks: &mut JoinSet<T>) {
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
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
        let buffer = ResultBuffer::new(&self.operators, batch_size, false);
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

    fn count_strategy(&self) -> CountStrategy<'_> {
        match self.operators.as_slice() {
            [] => CountStrategy::DirectRows,
            [Operator::Projection(projector)]
                if projector.projection().as_ref().is_empty() =>
            {
                CountStrategy::DirectRows
            }
            [
                Operator::Filter {
                    predicate,
                    projection: None,
                },
            ] => CountStrategy::DirectMask(predicate),
            [
                Operator::Filter {
                    predicate,
                    projection: None,
                },
                Operator::Projection(projector),
            ] if projector.projection().as_ref().is_empty() => {
                CountStrategy::DirectMask(predicate)
            }
            _ => CountStrategy::General,
        }
    }

    #[cfg(test)]
    fn memory_count_filter(&self) -> Option<&Arc<dyn PhysicalExpr>> {
        match self.count_strategy() {
            CountStrategy::DirectMask(predicate) => Some(predicate),
            CountStrategy::DirectRows | CountStrategy::General => None,
        }
    }

    fn group_count_strategy(&self, key_index: usize) -> GroupCountStrategy<'_> {
        match self.operators.as_slice() {
            [] => GroupCountStrategy::DirectColumn(key_index),
            [Operator::Projection(projector)] => plain_column_index(projector, key_index)
                .map_or(
                    GroupCountStrategy::General(key_index),
                    GroupCountStrategy::DirectColumn,
                ),
            [
                Operator::Filter {
                    predicate,
                    projection: None,
                },
            ] => GroupCountStrategy::DirectMask {
                predicate,
                key_index,
            },
            [
                Operator::Filter {
                    predicate,
                    projection: None,
                },
                Operator::Projection(projector),
            ] => plain_column_index(projector, key_index).map_or(
                GroupCountStrategy::General(key_index),
                |source_key| GroupCountStrategy::DirectMask {
                    predicate,
                    key_index: source_key,
                },
            ),
            _ => GroupCountStrategy::General(key_index),
        }
    }

    fn worker_sink(
        &self,
        batch_size: usize,
        reservation: Option<MemoryReservation>,
    ) -> WorkerSink {
        match &self.sink {
            Sink::Result => WorkerSink::Result(Box::new(ResultBuffer::new(
                &self.operators,
                batch_size,
                self.source.has_pushed_filter(),
            ))),
            Sink::Count { .. } => WorkerSink::Count(CountState::default()),
            Sink::GroupCount { sink, .. } => WorkerSink::GroupCount(
                sink.new_state(reservation.expect("group count reservation")),
            ),
        }
    }

    async fn run_memory_worker(
        &self,
        source: &MemorySource,
        next: &AtomicUsize,
        output: WorkerOutput,
        reservation: Option<MemoryReservation>,
        mut worker: WorkerMetrics,
    ) -> Result<WorkerState> {
        let mut sink = self.worker_sink(output.batch_size, reservation);
        let tx = &output.tx;
        while !tx.is_closed() {
            let start = next.fetch_add(BATCHES_PER_MORSEL, Ordering::Relaxed);
            if start >= source.batches.len() {
                break;
            }
            worker.claim();
            let end = (start + BATCHES_PER_MORSEL).min(source.batches.len());
            for (index, input) in (start..end).zip(&source.batches[start..end]) {
                if tx.is_closed() {
                    return Ok(WorkerState::None);
                }
                let batch = worker.process(|| source.read(input))?;
                if !self.consume(batch, tx, &mut sink, &mut worker).await? {
                    return Ok(WorkerState::None);
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
        reservation: Option<MemoryReservation>,
        mut worker: WorkerMetrics,
    ) -> Result<WorkerState> {
        let mut sink = self.worker_sink(output.batch_size, reservation);
        let tx = &output.tx;
        while let Some(claimed) = worker.ready(ready.claim()).await {
            if tx.is_closed() {
                return Ok(WorkerState::None);
            }
            let morsel = match claimed {
                ClaimedWork::Morsel {
                    morsel,
                    count_claim,
                } => {
                    if count_claim {
                        worker.claim();
                    }
                    morsel
                }
                ClaimedWork::FilterStep {
                    planner,
                    permit,
                    filter_guard,
                    count_claim,
                } => {
                    if count_claim {
                        worker.claim();
                    }
                    let step = worker.decode(|| planner.plan())?;
                    let Some(mut plan) = step else {
                        drop(permit);
                        drop(filter_guard);
                        worker.complete();
                        tokio::task::yield_now().await;
                        continue;
                    };
                    if let Some(morsel) = plan.take_morsels().pop() {
                        drop(permit);
                        drop(filter_guard);
                        morsel
                    } else if let Some(io) = plan.take_pending_planner() {
                        ready.submit_phase2(io, permit, filter_guard);
                        tokio::task::yield_now().await;
                        continue;
                    } else {
                        return internal_err!(
                            "Expected decode morsel or Phase-2 I/O from Parquet RowFilter step"
                        );
                    }
                }
            };
            let mut reader = worker.decode(|| morsel.into_stream());
            while let Some(batch) = metrics::decode(&mut worker, reader.next()).await {
                if tx.is_closed()
                    || !self.consume(batch?, tx, &mut sink, &mut worker).await?
                {
                    return Ok(WorkerState::None);
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
                let rows = match self.count_strategy() {
                    CountStrategy::DirectRows => batch.num_rows(),
                    CountStrategy::DirectMask(predicate) => {
                        count::count_filtered_rows(&batch, predicate)?
                    }
                    CountStrategy::General => {
                        apply_operators(&self.operators, batch)?.num_rows()
                    }
                };
                local.add_rows(rows)?;
                Ok(true)
            }),
            WorkerSink::GroupCount(local) => {
                let Sink::GroupCount { sink, .. } = &self.sink else {
                    unreachable!()
                };
                worker.process(|| {
                    match self.group_count_strategy(sink.key_index) {
                        GroupCountStrategy::DirectColumn(col_index) => {
                            local.add_column(batch.column(col_index))?;
                        }
                        GroupCountStrategy::DirectMask {
                            predicate,
                            key_index,
                        } => {
                            local.add_filtered_batch(&batch, predicate, key_index)?;
                        }
                        GroupCountStrategy::General(col_index) => {
                            let batch = apply_operators(&self.operators, batch)?;
                            if batch.num_rows() > 0 {
                                local.add_column(batch.column(col_index))?;
                            }
                        }
                    }
                    Ok(true)
                })
            }
            WorkerSink::Result(buffer) => {
                let mut output = worker.process(|| {
                    buffer.push_input(&self.operators, batch)?;
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
    ) -> Result<WorkerState> {
        let mut buffer = match sink {
            WorkerSink::Count(local) => return Ok(WorkerState::Count(local)),
            WorkerSink::GroupCount(local) => return Ok(WorkerState::GroupCount(local)),
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
        Ok(WorkerState::None)
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
        // Supervise independently of the consumer: an error must abort peers
        // even when they are waiting for space and the stream isn't polled.
        let worker_measurement = measurement.clone();
        let supervisor = SpawnedTask::spawn(async move {
            let measurement = worker_measurement;
            let group_reservation = match &self.sink {
                Sink::GroupCount { .. } => Some(
                    MemoryConsumer::new("MorselGroupCount")
                        .register(context.memory_pool()),
                ),
                Sink::Result | Sink::Count { .. } => None,
            };
            let mut combined_group = match (&self.sink, &group_reservation) {
                (Sink::GroupCount { sink, .. }, Some(reservation)) => {
                    Some(sink.new_state(reservation.new_empty()))
                }
                _ => None,
            };
            let mut tasks = JoinSet::new();
            match &self.source {
                Source::Memory(source) => {
                    let next = Arc::new(AtomicUsize::new(0));
                    let morsels = source.batches.len().div_ceil(BATCHES_PER_MORSEL);
                    for id in 0..self.workers.get().min(morsels) {
                        let (pipeline, next, output) =
                            (Arc::clone(&self), Arc::clone(&next), output.clone());
                        let reservation =
                            group_reservation.as_ref().map(MemoryReservation::new_empty);
                        let worker = WorkerMetrics::new(measurement.clone(), id);
                        tasks.spawn(async move {
                            let source = match &pipeline.source {
                                Source::Memory(source) => source,
                                #[cfg(feature = "parquet")]
                                Source::Parquet(_) => unreachable!(),
                            };
                            pipeline
                                .run_memory_worker(
                                    source,
                                    &next,
                                    output,
                                    reservation,
                                    worker,
                                )
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
                        measurement.as_ref(),
                    )?;
                    for id in 0..self.workers.get() {
                        let (pipeline, ready, output) =
                            (Arc::clone(&self), ready.clone(), output.clone());
                        let reservation =
                            group_reservation.as_ref().map(MemoryReservation::new_empty);
                        let worker = WorkerMetrics::new(measurement.clone(), id);
                        tasks.spawn(async move {
                            pipeline
                                .run_parquet_worker(ready, output, reservation, worker)
                                .await
                        });
                    }
                }
            }
            drop(group_reservation);
            let mut combined_count = CountState::default();
            while let Some(result) = tasks.join_next().await {
                let local = match result {
                    Ok(Ok(local)) => local,
                    Ok(Err(err)) => {
                        drop(combined_group);
                        abort_and_drain(&mut tasks).await;
                        return Err(err);
                    }
                    Err(err) if err.is_panic() => {
                        drop(combined_group);
                        abort_and_drain(&mut tasks).await;
                        std::panic::resume_unwind(err.into_panic())
                    }
                    Err(err) => {
                        drop(combined_group);
                        abort_and_drain(&mut tasks).await;
                        return Err(exec_datafusion_err!("Morsel task failed: {err}"));
                    }
                };
                match local {
                    WorkerState::None => {}
                    WorkerState::Count(local) => {
                        if let Err(err) = combined_count.combine(local) {
                            drop(combined_group);
                            abort_and_drain(&mut tasks).await;
                            return Err(err);
                        }
                    }
                    WorkerState::GroupCount(local) => {
                        if let Some(combined) = combined_group.as_mut() {
                            match combined.merge_from(local, &output.tx).await {
                                Ok(true) => {}
                                Ok(false) => {
                                    drop(combined_group);
                                    abort_and_drain(&mut tasks).await;
                                    return Ok(());
                                }
                                Err(err) => {
                                    drop(combined_group);
                                    abort_and_drain(&mut tasks).await;
                                    return Err(err);
                                }
                            }
                        }
                    }
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
            match &self.sink {
                Sink::Result => {}
                Sink::Count {
                    sink,
                    result_operators,
                } => {
                    if !tx.is_closed() {
                        // Run the dependent pipeline only on successful finalization. It
                        // has no independent task to wake on error or cancellation.
                        #[cfg(test)]
                        if let Some(gate) = &sink.finalize_gate {
                            gate.wait().await;
                        }
                        if !tx.is_closed() {
                            let finalized = sink.finalize(combined_count)?;
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
                    }
                }
                Sink::GroupCount {
                    sink,
                    result_operators,
                } => {
                    if !tx.is_closed() {
                        #[cfg(test)]
                        if let Some(gate) = &sink.finalize_gate {
                            gate.wait().await;
                        }
                        if !tx.is_closed()
                            && let Some(combined) = combined_group.take()
                        {
                            let finalized = sink.finalize(combined);
                            MetricBuilder::new(&metrics)
                                .counter("group_count_finalizations", 0)
                                .add(1);
                            finalized
                                .emit(
                                    result_operators,
                                    output.batch_size,
                                    tx,
                                    measurement.as_deref(),
                                )
                                .await?;
                        }
                    }
                }
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
            Sink::Count { .. } | Sink::GroupCount { .. } => EmissionType::Final,
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
            Source::Parquet(source) => match source.pushed_predicate() {
                Some(predicate) => {
                    write!(f, "NativeParquet[{}, predicate={predicate}]", source.table)?
                }
                None => write!(f, "NativeParquet[{}]", source.table)?,
            },
        }
        display_operators(&pipeline.operators, f)?;
        match &pipeline.sink {
            Sink::Result => {}
            Sink::Count {
                result_operators, ..
            } => {
                write!(f, " -> CountSink; FinalizedCountSource")?;
                display_operators(result_operators, f)?;
            }
            Sink::GroupCount {
                sink,
                result_operators,
            } => {
                write!(
                    f,
                    " -> GroupCountSink[key={}@{}]; FinalizedGroupCountSource",
                    sink.key_name, sink.key_index
                )?;
                display_operators(result_operators, f)?;
            }
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
            }
            | Sink::GroupCount {
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

#[cfg(test)]
mod parquet_count_tests {
    use super::*;

    #[cfg(all(feature = "parquet", feature = "sql"))]
    use ::parquet::arrow::ArrowWriter;
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use ::parquet::file::properties::WriterProperties;
    use arrow::array::RecordBatchOptions;
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use arrow::array::{Int32Array, StringArray, StructArray};
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use arrow::datatypes::{Field, FieldRef, Fields};
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use datafusion_common::{ScalarValue, exec_err};
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use datafusion_expr::ColumnarValue;
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use datafusion_physical_expr::expressions::Literal;
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use std::fs::File;
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use std::time::Duration;
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use tempfile::TempDir;
    #[cfg(all(feature = "parquet", feature = "sql"))]
    use tokio::time::timeout;

    #[test]
    fn zero_column_filter_batch_rejects_oversized_mask() -> Result<()> {
        let batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(2)),
        )?;
        let oversized = BooleanArray::from(vec![true, true, true]);
        let err = filter_batch(&batch, &oversized, None).unwrap_err();
        assert!(
            err.to_string().contains(
                "Filter predicate of length 3 is larger than target array of length 2"
            ),
            "{err}"
        );
        Ok(())
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    enum TestPhysicalExpr {
        OversizedMask,
        NonBooleanMask,
        Error,
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    impl fmt::Display for TestPhysicalExpr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestPhysicalExpr({self:?})")
        }
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    impl PhysicalExpr for TestPhysicalExpr {
        fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self, f)
        }

        fn return_field(&self, _schema: &Schema) -> Result<FieldRef> {
            Ok(Arc::new(Field::new("mask", DataType::Boolean, false)))
        }

        fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
            match self {
                Self::OversizedMask => Ok(ColumnarValue::Array(Arc::new(
                    BooleanArray::from(vec![true; batch.num_rows() + 1]),
                ))),
                Self::NonBooleanMask => {
                    Ok(ColumnarValue::Array(Arc::new(Int32Array::from(vec![
                        1;
                        batch
                            .num_rows(
                            )
                    ]))))
                }
                Self::Error => exec_err!("injected parquet worker error"),
            }
        }

        fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn PhysicalExpr>>,
        ) -> Result<Arc<dyn PhysicalExpr>> {
            assert!(children.is_empty());
            Ok(self)
        }
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    fn parquet_struct_fixture() -> Result<TempDir> {
        let dir = TempDir::new()?;
        let person_fields = Fields::from(vec![Field::new("name", DataType::Utf8, true)]);
        let names = Arc::new(StringArray::from(vec![
            Some("a"),
            Some("b"),
            None,
            Some("d"),
            Some("e"),
            Some("f"),
            Some("g"),
            Some("h"),
        ]));
        let person = StructArray::try_new(
            person_fields,
            vec![names],
            Some(vec![true, false, true, true, true, false, true, true].into()),
        )?;
        let batch = RecordBatch::try_from_iter([
            ("id", Arc::new(Int32Array::from_iter_values(0..8)) as _),
            (
                "gate",
                Arc::new(Int32Array::from(vec![
                    Some(12),
                    Some(15),
                    None,
                    Some(20),
                    Some(3),
                    Some(18),
                    Some(25),
                    Some(2),
                ])) as _,
            ),
            ("person", Arc::new(person) as _),
        ])?;
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4))
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join("0.parquet"))?,
            batch.schema(),
            Some(properties),
        )?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(dir)
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    async fn parquet_context(
        dir: &TempDir,
        pushdown: bool,
    ) -> Result<crate::prelude::SessionContext> {
        let mut config = crate::prelude::SessionConfig::new().with_target_partitions(4);
        config.options_mut().execution.parquet.pushdown_filters = pushdown;
        config.options_mut().execution.collect_statistics = false;
        let ctx = crate::prelude::SessionContext::new_with_config(config);
        ctx.register_parquet(
            "t",
            dir.path().to_str().unwrap(),
            crate::prelude::ParquetReadOptions::default(),
        )
        .await?;
        Ok(ctx)
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    async fn lower_parquet(
        ctx: &crate::prelude::SessionContext,
        workers: usize,
        sql: &str,
    ) -> Result<MorselExec> {
        let logical = ctx.sql(sql).await?.into_optimized_plan()?;
        let planner = MorselQueryPlanner::new(NonZeroUsize::new(workers).unwrap());
        let Lowering::Native(exec) = planner.try_lower(&logical, &ctx.state()).await?
        else {
            panic!("expected native morsel plan for {sql}: {logical:?}");
        };
        Ok(exec)
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    #[tokio::test]
    async fn parquet_count_strategy_and_filter_pruning_selection() -> Result<()> {
        let dir = parquet_struct_fixture()?;

        // pushdown_filters=false -> DirectMask for COUNT(*), filter pruning for Filter -> Projection.
        let ctx_no_pd = parquet_context(&dir, false).await?;
        let mut plan =
            lower_parquet(&ctx_no_pd, 4, "SELECT COUNT(*) FROM t WHERE gate > 10")
                .await?;
        assert!(matches!(
            plan.pipeline.count_strategy(),
            CountStrategy::DirectMask(_)
        ));
        // Inserting a second Filter forces fallback to General.
        let pipeline = Arc::get_mut(&mut plan.pipeline).unwrap();
        pipeline.operators.insert(
            1,
            Operator::Filter {
                predicate: Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
                projection: None,
            },
        );
        assert!(matches!(pipeline.count_strategy(), CountStrategy::General));

        let scan_plan =
            lower_parquet(&ctx_no_pd, 4, "SELECT id FROM t WHERE gate > 10").await?;
        assert!(
            matches!(
                scan_plan.pipeline.operators.as_slice(),
                [
                    Operator::Filter {
                        projection: Some(cols),
                        ..
                    },
                    Operator::Projection(_)
                ] if cols.as_slice() == [0]
            ),
            "expected filter-only column pruning on multi-worker Parquet scan: {:?}",
            scan_plan.pipeline.operators
        );

        // pushdown_filters=true:
        let ctx_pd = parquet_context(&dir, true).await?;

        // 1. Fully pushed COUNT(*) -> DirectRows with zero-column Parquet output.
        let pushed_count =
            lower_parquet(&ctx_pd, 4, "SELECT COUNT(*) FROM t WHERE gate > 10").await?;
        assert!(matches!(
            pushed_count.pipeline.count_strategy(),
            CountStrategy::DirectRows
        ));
        assert!(pushed_count.pipeline.source.schema().fields().is_empty());

        // 2. Residual struct filter COUNT(*) -> DirectMask on narrowed [person] schema.
        let residual_count = lower_parquet(
            &ctx_pd,
            4,
            "SELECT COUNT(*) FROM t WHERE gate > 10 AND person IS NOT NULL",
        )
        .await?;
        assert!(matches!(
            residual_count.pipeline.count_strategy(),
            CountStrategy::DirectMask(_)
        ));
        assert_eq!(residual_count.pipeline.source.schema().fields().len(), 1);

        // 3. Columnless residual volatile filter COUNT(*) -> DirectMask on zero-column output.
        let zero_col_residual = lower_parquet(
            &ctx_pd,
            4,
            "SELECT COUNT(*) FROM t WHERE gate > 10 AND random() >= 0.0",
        )
        .await?;
        assert!(matches!(
            zero_col_residual.pipeline.count_strategy(),
            CountStrategy::DirectMask(_)
        ));
        assert!(
            zero_col_residual
                .pipeline
                .source
                .schema()
                .fields()
                .is_empty()
        );

        // 4. Residual Filter -> Projection prunes filter-only struct column before coalescing.
        let residual_scan = lower_parquet(
            &ctx_pd,
            4,
            "SELECT id FROM t WHERE gate > 10 AND person IS NOT NULL",
        )
        .await?;
        assert!(
            matches!(
                residual_scan.pipeline.operators.as_slice(),
                [
                    Operator::Filter {
                        projection: Some(cols),
                        ..
                    },
                    Operator::Projection(_)
                ] if cols.as_slice() == [0]
            ),
            "expected residual filter column pruning: {:?}",
            residual_scan.pipeline.operators
        );
        Ok(())
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    #[tokio::test]
    async fn parquet_count_rejects_oversized_and_non_boolean_masks() -> Result<()> {
        let dir = parquet_struct_fixture()?;
        let deadline = Duration::from_secs(10);

        // Test both columned (pushdown=false) and zero-column (pushdown=true + columnless residual)
        // Parquet COUNT(*) pipelines.
        for (pushdown, sql) in [
            (false, "SELECT COUNT(*) FROM t WHERE gate > 10"),
            (
                true,
                "SELECT COUNT(*) FROM t WHERE gate > 10 AND random() >= 0.0",
            ),
        ] {
            let ctx = parquet_context(&dir, pushdown).await?;
            for (expr, expected_err) in [
                (
                    TestPhysicalExpr::OversizedMask,
                    "Filter predicate of length",
                ),
                (TestPhysicalExpr::NonBooleanMask, "non-boolean predicates"),
            ] {
                let mut exec = lower_parquet(&ctx, 2, sql).await?;
                let pipeline = Arc::get_mut(&mut exec.pipeline).unwrap();
                assert_eq!(pipeline.source.schema().fields().is_empty(), pushdown);
                let Operator::Filter { predicate, .. } = &mut pipeline.operators[0]
                else {
                    panic!("expected leading Filter: {:?}", pipeline.operators);
                };
                *predicate = Arc::new(expr);
                assert!(matches!(
                    pipeline.count_strategy(),
                    CountStrategy::DirectMask(_)
                ));
                let weak = Arc::downgrade(&exec.pipeline);
                let mut stream = exec.execute(0, ctx.task_ctx())?;
                let err = timeout(deadline, stream.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap_err();
                assert!(err.to_string().contains(expected_err), "{err}");
                assert!(timeout(deadline, stream.next()).await.unwrap().is_none());
                assert_eq!(
                    exec.metrics()
                        .and_then(|m| m.sum_by_name("count_finalizations"))
                        .map_or(0, |v| v.as_usize()),
                    0
                );
                drop(stream);
                drop(exec);
                timeout(deadline, async {
                    while weak.strong_count() != 0 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("parquet worker tasks did not stop");
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "parquet", feature = "sql"))]
    #[tokio::test]
    async fn parquet_count_general_projection_error_and_cancellation_teardown()
    -> Result<()> {
        let dir = parquet_struct_fixture()?;
        let deadline = Duration::from_secs(10);

        // 1. Non-empty projection above Filter uses General strategy and propagates errors.
        let ctx = parquet_context(&dir, false).await?;
        let mut exec =
            lower_parquet(&ctx, 2, "SELECT COUNT(*) FROM t WHERE gate > 10").await?;
        {
            let pipeline = Arc::get_mut(&mut exec.pipeline).unwrap();
            let input_schema = Arc::clone(pipeline.source.schema());
            let projector = ProjectionExprs::new([ProjectionExpr::new(
                Arc::new(TestPhysicalExpr::Error),
                "err",
            )])
            .make_projector(&input_schema)?;
            pipeline.operators[1] = Operator::Projection(projector);
            assert!(matches!(pipeline.count_strategy(), CountStrategy::General));
        }
        let weak = Arc::downgrade(&exec.pipeline);
        let mut stream = exec.execute(0, ctx.task_ctx())?;
        let err = timeout(deadline, stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(
            err.to_string().contains("injected parquet worker error"),
            "{err}"
        );
        assert_eq!(
            exec.metrics()
                .and_then(|m| m.sum_by_name("count_finalizations"))
                .map_or(0, |v| v.as_usize()),
            0
        );
        drop(stream);
        drop(exec);
        timeout(deadline, async {
            while weak.strong_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("parquet worker tasks did not stop");

        // 2. Dropping Parquet COUNT(*) stream immediately after starting tears down workers
        // without finalizing (both DirectMask with pushdown=false and DirectRows with pushdown=true).
        for pushdown in [false, true] {
            let ctx = parquet_context(&dir, pushdown).await?;
            let exec =
                lower_parquet(&ctx, 2, "SELECT COUNT(*) FROM t WHERE gate > 10").await?;
            let weak = Arc::downgrade(&exec.pipeline);
            let mut stream = exec.execute(0, ctx.task_ctx())?;
            assert!(futures::poll!(stream.next()).is_pending());
            drop(stream);
            assert_eq!(
                exec.metrics()
                    .and_then(|m| m.sum_by_name("count_finalizations"))
                    .map_or(0, |v| v.as_usize()),
                0
            );
            drop(exec);
            timeout(deadline, async {
                while weak.strong_count() != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("parquet workers did not stop after cancellation");
        }
        Ok(())
    }
}
