// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Experimental, opt-in morsel execution for in-memory and Parquet queries.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{BooleanArray, RecordBatch};
use arrow::compute::filter_record_batch;
#[cfg(test)]
use arrow::datatypes::DataType;
use arrow::datatypes::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_common::config::MorselMode;
use datafusion_common::instant::Instant;
use datafusion_common::nested_struct::adapt_batch_to_schema;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_common::{Result, exec_datafusion_err, internal_err, plan_err};
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_physical_expr::expressions::Column;
#[cfg(test)]
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::{ProjectionExprs, Projector};
use datafusion_physical_expr::{EquivalenceProperties, LexOrdering, PhysicalExpr};
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

#[cfg(test)]
use crate::datasource::MemTable;
use crate::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};

mod aggregate;
mod batch;
mod count;
mod dag;
mod fallback;
mod group_count;
mod hash_join;
mod metrics;
#[cfg(feature = "parquet")]
mod parquet;
mod planner;
mod sort;

use aggregate::{AggregateSink, AggregateState, FinalizedAggregate};
use batch::{BatchView, Selection, is_pure_column_projection};
use count::{CountSink, CountState, FinalizedCount};
use dag::{
    DagRuntime, MaterializedStage, ScalarSubqueryState, StageId, StageSource, StageSpec,
    bind_stage_dependencies, collect_pipeline_stage_deps, validate_dag,
};
#[cfg(test)]
use group_count::IntKeyType;
use group_count::{FinalizedGroupCounts, GroupCountSink, GroupCountState};
use hash_join::{
    FinalizedHashTable, HashJoinBuildSink, HashJoinBuildState, HashJoinProbe,
    OperatorStep, ProbeCursor, ProbeScratch,
};
pub use metrics::MorselMetrics;
use metrics::{RunMetrics, WorkerMetrics};
#[cfg(feature = "parquet")]
use parquet::{ClaimedWork, ParquetScan, ReadyReaders};
use sort::{SharedSortMemory, SortSink, SortState, SortedRun, TopNSink, TopNState};
#[cfg(test)]
mod tests;

const BATCHES_PER_MORSEL: usize = 4;
const COUNT_RESULT_CAPACITY: usize = 2;
const GROUP_COUNT_RESULT_CAPACITY: usize = 2;
const AGGREGATE_RESULT_CAPACITY: usize = 2;
const TOPN_RESULT_CAPACITY: usize = 2;
const SORT_RESULT_CAPACITY: usize = 2;
const LIMIT_RESULT_CAPACITY: usize = 2;
const MAX_RESULT_CAPACITY: usize = 32;

fn result_capacity(workers: NonZeroUsize) -> usize {
    workers.get().saturating_mul(2).min(MAX_RESULT_CAPACITY)
}

fn sink_result_capacity(sink: &Sink, workers: NonZeroUsize) -> usize {
    match sink {
        Sink::Result => result_capacity(workers),
        Sink::Count { .. } => COUNT_RESULT_CAPACITY,
        Sink::GroupCount { .. } => GROUP_COUNT_RESULT_CAPACITY,
        Sink::Aggregate { .. } => AGGREGATE_RESULT_CAPACITY,
        Sink::TopN { .. } => TOPN_RESULT_CAPACITY,
        Sink::Sort { .. } => SORT_RESULT_CAPACITY,
        Sink::Limit { .. } => LIMIT_RESULT_CAPACITY,
        Sink::HashJoinBuild(_) => 1,
    }
}

/// Opt-in planner for an experimental morsel pipeline over [`MemTable`]
/// and supported Parquet listing scans (with the `parquet` feature).
/// Lowers scans, filters, and projections with synchronous scalar expressions,
/// `INNER`/`OUTER`/`SEMI`/`ANTI` hash joins and `CROSS JOIN` (two-pipeline build sink +
/// lock-free probe operator + optional unmatched/matched build child pipeline),
/// ungrouped or grouped aggregates (`COUNT`, `SUM`, `MIN`, `MAX`, `AVG`),
/// and ordered `TopN` (`ORDER BY ... LIMIT K [OFFSET O]`) or full parallel `Sort`.
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

type BoxFuture<'a, T> = std::pin::Pin<Box<dyn Future<Output = T> + Send + 'a>>;

enum Lowering {
    Native(MorselExec),
    Unsupported(String),
}

impl MorselQueryPlanner {
    /// Set the maximum number of workers per execution of a native pipeline.
    pub fn new(workers: NonZeroUsize) -> Self {
        Self { workers }
    }
}

fn unsupported_provider_reason(table: &datafusion_common::TableReference) -> String {
    format!("unsupported table provider for {table}")
}

#[derive(Debug, Clone, Copy)]
struct SideEstimate {
    is_empty: bool,
    mem_rows: Option<usize>,
    approx_bytes: u64,
}

fn estimate_side_size(
    source: &Source,
    operators: &[Operator],
    stages: &[StageSpec],
) -> SideEstimate {
    let base = match source {
        Source::Memory(mem) => {
            let rows = mem.batches.iter().map(RecordBatch::num_rows).sum::<usize>();
            let bytes = mem
                .batches
                .iter()
                .map(get_record_batch_memory_size)
                .fold(0usize, usize::saturating_add) as u64;
            SideEstimate {
                is_empty: rows == 0,
                mem_rows: Some(rows),
                approx_bytes: bytes,
            }
        }
        #[cfg(feature = "parquet")]
        Source::Parquet(pq) => {
            let file_bytes = pq.total_file_bytes();
            SideEstimate {
                is_empty: pq.is_empty(),
                mem_rows: None,
                approx_bytes: file_bytes.saturating_mul(4),
            }
        }
        Source::Stage(stage_src) => stages[stage_src.stage_id.0].estimate,
    };
    let mut is_empty = base.is_empty;
    for op in operators {
        if let Operator::HashProbe(probe) = op {
            let build_est = estimate_side_size(
                &probe.build_pipeline.source,
                &probe.build_pipeline.operators,
                stages,
            );
            if build_est.is_empty && probe.empty_build_short_circuits() {
                is_empty = true;
            } else if !build_est.is_empty && probe.needs_build_emission() {
                is_empty = false;
            }
        }
    }
    SideEstimate { is_empty, ..base }
}

fn choose_build_side(left: SideEstimate, right: SideEstimate) -> hash_join::BuildSide {
    use hash_join::BuildSide;
    if left.is_empty && !right.is_empty {
        return BuildSide::Left;
    }
    if right.is_empty && !left.is_empty {
        return BuildSide::Right;
    }
    if let (Some(l_rows), Some(r_rows)) = (left.mem_rows, right.mem_rows) {
        return if l_rows < r_rows {
            BuildSide::Left
        } else {
            BuildSide::Right
        };
    }
    if left.approx_bytes < right.approx_bytes {
        BuildSide::Left
    } else {
        BuildSide::Right
    }
}

fn assign_join_build_ids(operators: &mut [Operator]) -> Result<()> {
    let mut next_id = 0usize;
    for op in operators {
        if let Operator::HashProbe(probe) = op {
            let Some(p) = Arc::get_mut(probe) else {
                return internal_err!(
                    "Expected unique ownership of HashJoinProbe while assigning build IDs"
                );
            };
            p.build_id = next_id;
            next_id += 1;
        }
    }
    Ok(())
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
            Expr::ScalarSubquery(sq) => !sq.outer_ref_columns.is_empty(),
            Expr::Exists(_)
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
    HashProbe(Arc<HashJoinProbe>),
}

#[derive(Debug)]
enum Source {
    Memory(MemorySource),
    #[cfg(feature = "parquet")]
    Parquet(ParquetScan),
    Stage(StageSource),
}

impl Source {
    fn schema(&self) -> &SchemaRef {
        match self {
            Self::Memory(source) => &source.schema,
            #[cfg(feature = "parquet")]
            Self::Parquet(source) => &source.schema,
            Self::Stage(source) => &source.schema,
        }
    }

    fn has_pushed_filter(&self) -> bool {
        match self {
            Self::Memory(_) | Self::Stage(_) => false,
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

fn filter_batch(
    batch: &RecordBatch,
    mask: &BooleanArray,
    projection: Option<&[usize]>,
) -> Result<RecordBatch> {
    if batch.num_columns() == 0 {
        count::validate_mask_length(batch, mask)?;
    }
    if mask.len() == batch.num_rows() {
        let selected = mask.true_count();
        if selected == batch.num_rows() {
            return match projection {
                Some(columns) => Ok(batch.project(columns)?),
                None => Ok(batch.clone()),
            };
        }
        if selected == 0 {
            let schema = match projection {
                Some(columns) => Arc::new(batch.schema().project(columns)?),
                None => batch.schema(),
            };
            return Ok(RecordBatch::new_empty(schema));
        }
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

fn apply_operators_view(
    operators: &[Operator],
    mut view: BatchView,
) -> Result<BatchView> {
    for operator in operators {
        view = match operator {
            Operator::Filter {
                predicate,
                projection,
            } => {
                let filtered = view.filter(predicate)?;
                match projection {
                    Some(columns) => filtered.project_columns(columns)?,
                    None => filtered,
                }
            }
            Operator::Projection(projector) => view.apply_projection(projector)?,
            Operator::HashProbe(_) => {
                return internal_err!(
                    "HashProbe must execute through join continuation driver"
                );
            }
        };
        if view.is_empty() {
            break;
        }
    }
    Ok(view)
}

fn apply_operators(operators: &[Operator], batch: RecordBatch) -> Result<RecordBatch> {
    apply_operators_view(operators, BatchView::all(batch))?.materialize()
}

fn push_view_into_coalescer(
    coalescer: &mut LimitedBatchCoalescer,
    view: BatchView,
) -> Result<()> {
    if view.is_empty() {
        return Ok(());
    }
    match view.into_parts() {
        (batch, Selection::All) => {
            coalescer.push_batch(batch)?;
        }
        (batch, Selection::Mask { bits, .. }) => {
            let mask = BooleanArray::new(bits, None);
            coalescer.push_batch_with_filter(batch, &mask)?;
        }
        (batch, selection @ Selection::Indices(_)) => {
            let materialized = BatchView::try_new(batch, selection)?.materialize()?;
            coalescer.push_batch(materialized)?;
        }
    }
    Ok(())
}

enum NarrowProjectorCache {
    Uninitialized,
    Identity,
    Narrowed {
        columns: Vec<usize>,
        schema: SchemaRef,
        projector: Projector,
    },
}

struct ResultBuffer {
    batch_size: usize,
    coalesce_source: bool,
    source_coalescer: Option<LimitedBatchCoalescer>,
    source_finished: bool,
    /// Slot `i < operators.len()` coalesces filtered input just before a non-pure
    /// `Operator::Projection` at index `i`. Slot `operators.len()` is the terminal
    /// coalescer for filtered batches reaching the `Result` sink.
    coalescers: Vec<Option<LimitedBatchCoalescer>>,
    is_pure_projection: Vec<bool>,
    filter_schemas: Vec<Option<SchemaRef>>,
    narrow_projectors: Vec<NarrowProjectorCache>,
    output: Option<RecordBatch>,
    next_to_finish: usize,
}

impl ResultBuffer {
    fn new(operators: &[Operator], batch_size: usize, coalesce_source: bool) -> Self {
        let has_filter = operators
            .iter()
            .any(|op| matches!(op, Operator::Filter { .. }));
        let slot_count = if has_filter { operators.len() + 1 } else { 0 };
        let op_cache_len = if has_filter { operators.len() } else { 0 };
        let is_pure_projection = if has_filter {
            operators
                .iter()
                .map(|op| {
                    matches!(op, Operator::Projection(p) if is_pure_column_projection(p))
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            batch_size,
            coalesce_source,
            source_coalescer: None,
            source_finished: !coalesce_source,
            coalescers: std::iter::repeat_with(|| None).take(slot_count).collect(),
            is_pure_projection,
            filter_schemas: std::iter::repeat_with(|| None).take(op_cache_len).collect(),
            narrow_projectors: std::iter::repeat_with(|| {
                NarrowProjectorCache::Uninitialized
            })
            .take(op_cache_len)
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

    fn push_view(&mut self, operators: &[Operator], view: BatchView) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        if !self.coalesce_source {
            let saw_filter = !view.is_all();
            return self.push_view_from(operators, 0, view, saw_filter);
        }
        let schema = view.schema();
        let coalescer = self.source_coalescer.get_or_insert_with(|| {
            LimitedBatchCoalescer::new(schema, self.batch_size, None)
        });
        push_view_into_coalescer(coalescer, view)
    }

    fn push_from(
        &mut self,
        operators: &[Operator],
        start: usize,
        mut batch: RecordBatch,
    ) -> Result<()> {
        if self.coalescers.is_empty() {
            for op in operators.iter().skip(start) {
                match op {
                    Operator::Projection(projector) => {
                        batch = projector.project_batch(&batch)?;
                    }
                    Operator::Filter { .. } => unreachable!(),
                    Operator::HashProbe(_) => {
                        return internal_err!(
                            "HashProbe must execute through join continuation driver"
                        );
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
            return Ok(());
        }
        self.push_view_from(operators, start, BatchView::all(batch), false)
    }

    fn ensure_narrow_projector(
        cache: &mut NarrowProjectorCache,
        projector: &Projector,
        input_schema: &Schema,
    ) {
        if !matches!(cache, NarrowProjectorCache::Uninitialized) {
            return;
        }
        let needed = projector.projection().column_indices();
        if needed.len() < input_schema.fields().len() {
            let remap = || -> Result<(SchemaRef, Projector)> {
                let narrow = ProjectionExprs::from_indices(&needed, input_schema);
                let narrow_schema = Arc::new(input_schema.project(&needed)?);
                let remapped = projector
                    .projection()
                    .clone()
                    .try_map_exprs(|expr| narrow.project_expr(&expr))?
                    .make_projector_with_schema_metadata(
                        &narrow_schema,
                        projector.output_schema(),
                    )?;
                Ok((narrow_schema, remapped))
            };
            if let Ok((schema, remapped)) = remap()
                && remapped.output_schema() == projector.output_schema()
            {
                *cache = NarrowProjectorCache::Narrowed {
                    columns: needed,
                    schema,
                    projector: remapped,
                };
                return;
            }
        }
        *cache = NarrowProjectorCache::Identity;
    }

    fn push_view_from(
        &mut self,
        operators: &[Operator],
        start: usize,
        mut view: BatchView,
        mut saw_filter: bool,
    ) -> Result<()> {
        for (index, op) in operators.iter().enumerate().skip(start) {
            match op {
                Operator::Filter {
                    predicate,
                    projection,
                } => {
                    view = view.filter(predicate)?;
                    if let Some(columns) = projection {
                        let schema = match &self.filter_schemas[index] {
                            Some(s) => Arc::clone(s),
                            None => {
                                let s = Arc::new(view.batch().schema().project(columns)?);
                                self.filter_schemas[index] = Some(Arc::clone(&s));
                                s
                            }
                        };
                        view = view.project_columns_with_schema(columns, schema)?;
                    }
                    saw_filter = true;
                }
                Operator::Projection(projector) => {
                    if self.is_pure_projection.get(index).copied().unwrap_or(false) {
                        view = view.apply_projection(projector)?;
                    } else if !self.coalescers.is_empty()
                        && (saw_filter || self.coalescers[index].is_some())
                    {
                        // Narrow away filter-only columns zero-copy before coalescing,
                        // then evaluate expressions on full coalesced batches.
                        Self::ensure_narrow_projector(
                            &mut self.narrow_projectors[index],
                            projector,
                            view.batch().schema().as_ref(),
                        );
                        if let NarrowProjectorCache::Narrowed {
                            columns, schema, ..
                        } = &self.narrow_projectors[index]
                        {
                            view = view.project_columns_with_schema(
                                columns,
                                Arc::clone(schema),
                            )?;
                        }
                        if view.is_empty() {
                            return Ok(());
                        }
                        let schema = view.schema();
                        let coalescer = self.coalescers[index].get_or_insert_with(|| {
                            LimitedBatchCoalescer::new(schema, self.batch_size, None)
                        });
                        push_view_into_coalescer(coalescer, view)?;
                        return Ok(());
                    } else {
                        view = view.apply_projection(projector)?;
                    }
                }
                Operator::HashProbe(_) => {
                    return internal_err!(
                        "HashProbe must execute through join continuation driver"
                    );
                }
            }
            if view.is_empty() {
                return Ok(());
            }
        }

        if view.is_empty() {
            return Ok(());
        }
        let terminal = operators.len();
        if (saw_filter || !view.is_all()) && terminal < self.coalescers.len() {
            let schema = view.schema();
            let coalescer = self.coalescers[terminal].get_or_insert_with(|| {
                LimitedBatchCoalescer::new(schema, self.batch_size, None)
            });
            push_view_into_coalescer(coalescer, view)?;
            return Ok(());
        }
        let batch = view.materialize()?;
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
                self.coalescers
                    .iter_mut()
                    .enumerate()
                    .rev()
                    .find_map(|(index, slot)| {
                        Some((index, slot.as_mut()?.next_completed_batch()?))
                    });
            if let Some((index, batch)) = completed {
                if index == operators.len() {
                    return Ok(Some(batch));
                }
                let projected = match &self.narrow_projectors[index] {
                    NarrowProjectorCache::Narrowed { projector, .. } => {
                        projector.project_batch(&batch)?
                    }
                    NarrowProjectorCache::Identity
                    | NarrowProjectorCache::Uninitialized => {
                        let Operator::Projection(projector) = &operators[index] else {
                            unreachable!()
                        };
                        projector.project_batch(&batch)?
                    }
                };
                if index + 1 == operators.len() {
                    if projected.num_rows() != 0 {
                        return Ok(Some(projected));
                    }
                } else {
                    self.push_view_from(
                        operators,
                        index + 1,
                        BatchView::all(projected),
                        false,
                    )?;
                }
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
            } else if eof && self.next_to_finish < self.coalescers.len() {
                if let Some(coalescer) = &mut self.coalescers[self.next_to_finish] {
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
enum PostSort {
    TopN {
        sink: TopNSink,
        result_operators: Vec<Operator>,
    },
    Sort {
        sink: SortSink,
        result_operators: Vec<Operator>,
    },
}

impl PostSort {
    fn ordering_and_result_ops(&self) -> (&SchemaRef, &LexOrdering, &[Operator]) {
        match self {
            Self::TopN {
                sink,
                result_operators,
            } => (sink.input_schema(), sink.ordering(), result_operators),
            Self::Sort {
                sink,
                result_operators,
            } => (sink.input_schema(), sink.ordering(), result_operators),
        }
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let (sink_status, result_ops) = match self {
            Self::TopN {
                sink,
                result_operators,
            } => (sink.apply_expressions(f)?, result_operators.as_slice()),
            Self::Sort {
                sink,
                result_operators,
            } => (sink.apply_expressions(f)?, result_operators.as_slice()),
        };
        if sink_status == TreeNodeRecursion::Stop {
            return Ok(TreeNodeRecursion::Stop);
        }
        apply_operator_expressions(result_ops, f)
    }

    fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TopN {
                sink,
                result_operators,
            } => {
                write!(f, " -> ")?;
                sink.display_summary(f)?;
                write!(f, "; FinalizedTopNSource")?;
                display_operators(result_operators, f)
            }
            Self::Sort {
                sink,
                result_operators,
            } => {
                write!(f, " -> ")?;
                sink.display_summary(f)?;
                write!(f, "; FinalizedSortSource")?;
                display_operators(result_operators, f)
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
        post_sort: Option<PostSort>,
    },
    GroupCount {
        sink: GroupCountSink,
        result_operators: Vec<Operator>,
        post_sort: Option<PostSort>,
    },
    Aggregate {
        sink: AggregateSink,
        result_operators: Vec<Operator>,
        post_sort: Option<PostSort>,
    },
    TopN {
        sink: TopNSink,
        result_operators: Vec<Operator>,
    },
    Sort {
        sink: SortSink,
        result_operators: Vec<Operator>,
    },
    Limit {
        skip: usize,
        fetch: Option<usize>,
        result_operators: Vec<Operator>,
    },
    HashJoinBuild(Arc<HashJoinBuildSink>),
}

impl Sink {
    fn can_attach_post_sort(&self) -> bool {
        matches!(
            self,
            Self::Count {
                post_sort: None,
                ..
            } | Self::GroupCount {
                post_sort: None,
                ..
            } | Self::Aggregate {
                post_sort: None,
                ..
            }
        )
    }

    fn attach_post_sort(&mut self, ps: PostSort) {
        match self {
            Self::Count { post_sort, .. }
            | Self::GroupCount { post_sort, .. }
            | Self::Aggregate { post_sort, .. } => {
                *post_sort = Some(ps);
            }
            _ => unreachable!("can_attach_post_sort checked before attach_post_sort"),
        }
    }

    fn extend_result_operators(&mut self, ops: Vec<Operator>) -> Result<()> {
        match self {
            Self::Count {
                result_operators,
                post_sort,
                ..
            }
            | Self::GroupCount {
                result_operators,
                post_sort,
                ..
            }
            | Self::Aggregate {
                result_operators,
                post_sort,
                ..
            } => {
                if let Some(ps) = post_sort {
                    match ps {
                        PostSort::TopN {
                            result_operators: ps_ops,
                            ..
                        }
                        | PostSort::Sort {
                            result_operators: ps_ops,
                            ..
                        } => ps_ops.extend(ops),
                    }
                } else {
                    result_operators.extend(ops);
                }
                Ok(())
            }
            Self::TopN {
                result_operators, ..
            }
            | Self::Sort {
                result_operators, ..
            }
            | Self::Limit {
                result_operators, ..
            } => {
                result_operators.extend(ops);
                Ok(())
            }
            Self::Result | Self::HashJoinBuild(_) => {
                internal_err!("Cannot attach result operators to non-blocking sink")
            }
        }
    }

    fn root_sort_ordering(&self) -> Option<(&SchemaRef, &LexOrdering, &[Operator])> {
        match self {
            Self::TopN {
                sink,
                result_operators,
            } => Some((sink.input_schema(), sink.ordering(), result_operators)),
            Self::Sort {
                sink,
                result_operators,
            } => Some((sink.input_schema(), sink.ordering(), result_operators)),
            Self::Count {
                post_sort: Some(ps),
                ..
            }
            | Self::GroupCount {
                post_sort: Some(ps),
                ..
            }
            | Self::Aggregate {
                post_sort: Some(ps),
                ..
            } => Some(ps.ordering_and_result_ops()),
            Self::Result
            | Self::Count {
                post_sort: None, ..
            }
            | Self::GroupCount {
                post_sort: None, ..
            }
            | Self::Aggregate {
                post_sort: None, ..
            }
            | Self::Limit { .. }
            | Self::HashJoinBuild(_) => None,
        }
    }
}

enum WorkerSinkMemory {
    Reservation(MemoryReservation),
    SharedSort(Arc<SharedSortMemory>),
}

impl WorkerSinkMemory {
    fn into_reservation(self) -> MemoryReservation {
        match self {
            Self::Reservation(r) => r,
            Self::SharedSort(_) => panic!("expected MemoryReservation"),
        }
    }

    fn into_shared_sort(self) -> Arc<SharedSortMemory> {
        match self {
            Self::SharedSort(s) => s,
            Self::Reservation(_) => panic!("expected SharedSortMemory"),
        }
    }
}

enum WorkerSink {
    Result(Box<ResultBuffer>),
    Count(CountState),
    GroupCount(GroupCountState),
    Aggregate(AggregateState),
    TopN(TopNState),
    Sort(SortState),
    HashJoinBuild(HashJoinBuildState),
}

enum WorkerState {
    None,
    Count(CountState),
    GroupCount(GroupCountState),
    Aggregate(AggregateState),
    SortedRuns(Vec<SortedRun>),
    HashJoinBuild(HashJoinBuildState),
}

struct JoinWorkerState {
    tables: Arc<Vec<Arc<FinalizedHashTable>>>,
    scratch: ProbeScratch,
    probe_reservation: MemoryReservation,
}

enum BuildRunOutcome {
    Ready {
        tables: Arc<Vec<Arc<FinalizedHashTable>>>,
        short_circuit_op: Option<usize>,
    },
    ShortCircuitAll,
    Cancelled,
}

struct BuildEmissionTargets<'a> {
    combined_count: Option<&'a mut CountState>,
    combined_group: Option<&'a mut Option<GroupCountState>>,
    combined_agg: Option<&'a mut Option<AggregateState>>,
    sorted_runs: Option<&'a mut Vec<SortedRun>>,
    combined_build: Option<&'a mut HashJoinBuildState>,
    sink_reservation: Option<&'a MemoryReservation>,
    shared_sort_memory: Option<&'a Arc<SharedSortMemory>>,
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

fn apply_operator_expressions(
    operators: &[Operator],
    f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
) -> Result<TreeNodeRecursion> {
    for operator in operators {
        let status = match operator {
            Operator::Filter { predicate, .. } => {
                apply_expression_roots(std::iter::once(predicate), f)?
            }
            Operator::Projection(projector) => {
                apply_expression_roots(projector.projection().iter().map(|e| &e.expr), f)?
            }
            Operator::HashProbe(probe) => probe.apply_expressions(f)?,
        };
        if status == TreeNodeRecursion::Stop {
            return Ok(TreeNodeRecursion::Stop);
        }
    }
    Ok(TreeNodeRecursion::Continue)
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
    fn has_join(&self) -> bool {
        self.operators
            .iter()
            .any(|op| matches!(op, Operator::HashProbe(_)))
    }

    fn is_inline_memory_stream(&self) -> bool {
        self.workers == NonZeroUsize::MIN
            && matches!(self.source, Source::Memory(_))
            && matches!(self.sink, Sink::Result)
            && !self.has_join()
    }

    #[cfg(feature = "parquet")]
    fn collect_build_parquet_metrics<'a>(
        &'a self,
        out: &mut Vec<&'a ExecutionPlanMetricsSet>,
    ) {
        for op in &self.operators {
            if let Operator::HashProbe(probe) = op {
                if let Source::Parquet(source) = &probe.build_pipeline.source {
                    out.push(&source.metrics);
                }
                probe.build_pipeline.collect_build_parquet_metrics(out);
            }
        }
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        #[cfg(feature = "parquet")]
        if let Source::Parquet(source) = &self.source
            && source.apply_expressions(f)? == TreeNodeRecursion::Stop
        {
            return Ok(TreeNodeRecursion::Stop);
        }
        match &self.sink {
            Sink::Aggregate { sink, .. } => {
                if sink.apply_expressions(f)? == TreeNodeRecursion::Stop {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Sink::TopN { sink, .. } => {
                if sink.apply_expressions(f)? == TreeNodeRecursion::Stop {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Sink::Sort { sink, .. } => {
                if sink.apply_expressions(f)? == TreeNodeRecursion::Stop {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Sink::HashJoinBuild(sink) => {
                if sink.apply_expressions(f)? == TreeNodeRecursion::Stop {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Sink::Result
            | Sink::Count { .. }
            | Sink::GroupCount { .. }
            | Sink::Limit { .. } => {}
        }
        let (result_operators, post_sort) = match &self.sink {
            Sink::Result | Sink::HashJoinBuild(_) => (&[][..], None),
            Sink::Count {
                result_operators,
                post_sort,
                ..
            }
            | Sink::GroupCount {
                result_operators,
                post_sort,
                ..
            }
            | Sink::Aggregate {
                result_operators,
                post_sort,
                ..
            } => (result_operators.as_slice(), post_sort.as_ref()),
            Sink::TopN {
                result_operators, ..
            }
            | Sink::Sort {
                result_operators, ..
            }
            | Sink::Limit {
                result_operators, ..
            } => (result_operators.as_slice(), None),
        };
        if apply_operator_expressions(&self.operators, f)? == TreeNodeRecursion::Stop {
            return Ok(TreeNodeRecursion::Stop);
        }
        if apply_operator_expressions(result_operators, f)? == TreeNodeRecursion::Stop {
            return Ok(TreeNodeRecursion::Stop);
        }
        if let Some(post_sort) = post_sort
            && post_sort.apply_expressions(f)? == TreeNodeRecursion::Stop
        {
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    }

    fn inline_memory_stream(
        self: Arc<Self>,
        batch_size: usize,
        worker: WorkerMetrics,
    ) -> impl Stream<Item = Result<RecordBatch>> {
        let buffer = Box::new(ResultBuffer::new(&self.operators, batch_size, false));
        stream::try_unfold(
            (self, 0usize, worker, buffer),
            |(pipeline, mut next, mut worker, mut buffer)| async move {
                let source = match &pipeline.source {
                    Source::Memory(source) => source,
                    #[cfg(feature = "parquet")]
                    Source::Parquet(_) => unreachable!(),
                    Source::Stage(_) => unreachable!(),
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
        sink_memory: Option<WorkerSinkMemory>,
    ) -> Result<WorkerSink> {
        Ok(match &self.sink {
            Sink::Result | Sink::Limit { .. } => {
                if self.has_join() {
                    WorkerSink::Result(Box::new(ResultBuffer::new(&[], batch_size, true)))
                } else {
                    WorkerSink::Result(Box::new(ResultBuffer::new(
                        &self.operators,
                        batch_size,
                        self.source.has_pushed_filter(),
                    )))
                }
            }
            Sink::Count { .. } => WorkerSink::Count(CountState::default()),
            Sink::GroupCount { sink, .. } => WorkerSink::GroupCount(
                sink.new_state(
                    sink_memory
                        .expect("group count reservation")
                        .into_reservation(),
                ),
            ),
            Sink::Aggregate { sink, .. } => WorkerSink::Aggregate(
                sink.new_state(
                    sink_memory
                        .expect("aggregate reservation")
                        .into_reservation(),
                )?,
            ),
            Sink::TopN { sink, .. } => {
                WorkerSink::TopN(sink.new_state(
                    sink_memory.expect("topn reservation").into_reservation(),
                )?)
            }
            Sink::Sort { sink, .. } => {
                WorkerSink::Sort(sink.new_state(
                    sink_memory.expect("sort shared memory").into_shared_sort(),
                ))
            }
            Sink::HashJoinBuild(sink) => WorkerSink::HashJoinBuild(
                sink.new_state(
                    sink_memory
                        .expect("hash join build reservation")
                        .into_reservation(),
                ),
            ),
        })
    }

    async fn run_memory_worker(
        &self,
        source: &MemorySource,
        next: &AtomicUsize,
        output: WorkerOutput,
        sink_memory: Option<WorkerSinkMemory>,
        join_context: Option<(Arc<Vec<Arc<FinalizedHashTable>>>, MemoryReservation)>,
        mut worker: WorkerMetrics,
    ) -> Result<WorkerState> {
        let mut sink = self.worker_sink(output.batch_size, sink_memory)?;
        let mut join_state =
            join_context.map(|(tables, probe_reservation)| JoinWorkerState {
                tables,
                scratch: ProbeScratch::default(),
                probe_reservation,
            });
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
                let cont = if let Some(js) = join_state.as_mut() {
                    self.consume_with_joins(
                        batch,
                        tx,
                        output.batch_size,
                        &mut sink,
                        js,
                        &mut worker,
                    )
                    .await?
                } else {
                    self.consume(batch, tx, &mut sink, &mut worker).await?
                };
                if !cont {
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

    #[expect(clippy::too_many_arguments)]
    async fn run_stage_worker(
        &self,
        batches: &[RecordBatch],
        next: &AtomicUsize,
        sequential: bool,
        output: WorkerOutput,
        sink_memory: Option<WorkerSinkMemory>,
        join_context: Option<(Arc<Vec<Arc<FinalizedHashTable>>>, MemoryReservation)>,
        mut worker: WorkerMetrics,
    ) -> Result<WorkerState> {
        let mut sink = self.worker_sink(output.batch_size, sink_memory)?;
        let mut join_state =
            join_context.map(|(tables, probe_reservation)| JoinWorkerState {
                tables,
                scratch: ProbeScratch::default(),
                probe_reservation,
            });
        let tx = &output.tx;
        let stride = if sequential {
            batches.len().max(1)
        } else {
            BATCHES_PER_MORSEL
        };
        while !tx.is_closed() {
            let start = next.fetch_add(stride, Ordering::Relaxed);
            if start >= batches.len() {
                break;
            }
            worker.claim();
            let end = (start + stride).min(batches.len());
            for (index, input) in (start..end).zip(&batches[start..end]) {
                if tx.is_closed() {
                    return Ok(WorkerState::None);
                }
                let batch = input.clone();
                let cont = if let Some(js) = join_state.as_mut() {
                    self.consume_with_joins(
                        batch,
                        tx,
                        output.batch_size,
                        &mut sink,
                        js,
                        &mut worker,
                    )
                    .await?
                } else {
                    self.consume(batch, tx, &mut sink, &mut worker).await?
                };
                if !cont {
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
        sink_memory: Option<WorkerSinkMemory>,
        join_context: Option<(Arc<Vec<Arc<FinalizedHashTable>>>, MemoryReservation)>,
        mut worker: WorkerMetrics,
    ) -> Result<WorkerState> {
        let mut sink = self.worker_sink(output.batch_size, sink_memory)?;
        let mut join_state =
            join_context.map(|(tables, probe_reservation)| JoinWorkerState {
                tables,
                scratch: ProbeScratch::default(),
                probe_reservation,
            });
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
                if tx.is_closed() {
                    return Ok(WorkerState::None);
                }
                let batch = batch?;
                let cont = if let Some(js) = join_state.as_mut() {
                    self.consume_with_joins(
                        batch,
                        tx,
                        output.batch_size,
                        &mut sink,
                        js,
                        &mut worker,
                    )
                    .await?
                } else {
                    self.consume(batch, tx, &mut sink, &mut worker).await?
                };
                if !cont {
                    return Ok(WorkerState::None);
                }
                tokio::task::yield_now().await;
            }
            worker.complete();
            tokio::task::yield_now().await;
        }
        self.finish_worker(sink, &output, worker)
    }

    fn advance_join_chain(
        &self,
        start_op: usize,
        mut view: BatchView,
        join_state: &mut JoinWorkerState,
        frames: &mut Vec<(usize, ProbeCursor)>,
        batch_size: usize,
    ) -> Result<Option<BatchView>> {
        for (idx, operator) in self.operators.iter().enumerate().skip(start_op) {
            match operator {
                Operator::Filter {
                    predicate,
                    projection,
                } => {
                    view = view.filter(predicate)?;
                    if let Some(columns) = projection {
                        view = view.project_columns(columns)?;
                    }
                    if view.is_empty() {
                        return Ok(None);
                    }
                }
                Operator::Projection(projector) => {
                    view = view.apply_projection(projector)?;
                    if view.is_empty() {
                        return Ok(None);
                    }
                }
                Operator::HashProbe(probe) => {
                    let table = &join_state.tables[probe.build_id];
                    let Some(mut cursor) = probe.start_probe_view(
                        view,
                        table,
                        &join_state.probe_reservation,
                    )?
                    else {
                        return Ok(None);
                    };
                    match probe.next_chunk(
                        &mut cursor,
                        table,
                        batch_size,
                        &mut join_state.scratch,
                    )? {
                        OperatorStep::NeedMoreInput(None) => return Ok(None),
                        OperatorStep::NeedMoreInput(Some(out)) => {
                            view = BatchView::all(out);
                        }
                        OperatorStep::HaveMoreOutput(out) => {
                            frames.push((idx, cursor));
                            view = BatchView::all(out);
                        }
                    }
                }
            }
        }
        if view.is_empty() {
            Ok(None)
        } else {
            Ok(Some(view))
        }
    }

    async fn deliver_join_output(
        &self,
        view: BatchView,
        tx: &mpsc::Sender<RecordBatch>,
        sink: &mut WorkerSink,
        worker: &mut WorkerMetrics,
    ) -> Result<bool> {
        match sink {
            WorkerSink::Count(local) => worker.process(|| {
                local.add_rows(view.num_rows())?;
                Ok(true)
            }),
            WorkerSink::GroupCount(local) => {
                let Sink::GroupCount { sink, .. } = &self.sink else {
                    unreachable!()
                };
                worker.process(|| {
                    local.add_view(&view, sink.key_index)?;
                    Ok(true)
                })
            }
            WorkerSink::Aggregate(local) => worker.process(|| {
                local.add_view(&view)?;
                Ok(true)
            }),
            WorkerSink::TopN(local) => worker.process(|| {
                local.add_view(&view)?;
                Ok(true)
            }),
            WorkerSink::Sort(local) => worker.process(|| {
                local.add_view(view)?;
                Ok(true)
            }),
            WorkerSink::HashJoinBuild(local) => worker.process(|| {
                local.add_view(&view)?;
                Ok(true)
            }),
            WorkerSink::Result(buffer) => {
                // Avoid copying already-large dense join output batches through the
                // coalescer; only coalesce sparse/small or filtered batches.
                if view.is_all() && view.num_rows() >= (buffer.batch_size / 2).max(1) {
                    return Ok(worker.send(tx, view.into_parts().0).await);
                }
                let mut output = worker.process(|| {
                    buffer.push_view(&[], view)?;
                    buffer.next(&[], false)
                })?;
                while let Some(out) = output {
                    if !worker.send(tx, out).await {
                        return Ok(false);
                    }
                    output = worker.process(|| buffer.next(&[], false))?;
                }
                Ok(true)
            }
        }
    }

    async fn consume_with_joins(
        &self,
        batch: RecordBatch,
        tx: &mpsc::Sender<RecordBatch>,
        batch_size: usize,
        sink: &mut WorkerSink,
        join_state: &mut JoinWorkerState,
        worker: &mut WorkerMetrics,
    ) -> Result<bool> {
        self.consume_with_joins_from(0, batch, tx, batch_size, sink, join_state, worker)
            .await
    }

    #[expect(clippy::too_many_arguments)]
    async fn consume_with_joins_from(
        &self,
        start_op: usize,
        batch: RecordBatch,
        tx: &mpsc::Sender<RecordBatch>,
        batch_size: usize,
        sink: &mut WorkerSink,
        join_state: &mut JoinWorkerState,
        worker: &mut WorkerMetrics,
    ) -> Result<bool> {
        let mut frames: Vec<(usize, ProbeCursor)> = Vec::new();
        let mut produced = worker.process(|| {
            self.advance_join_chain(
                start_op,
                BatchView::all(batch),
                join_state,
                &mut frames,
                batch_size,
            )
        })?;
        loop {
            if let Some(out) = produced
                && !self.deliver_join_output(out, tx, sink, worker).await?
            {
                return Ok(false);
            }
            let Some((op_idx, mut cursor)) = frames.pop() else {
                return Ok(true);
            };
            if tx.is_closed() {
                return Ok(false);
            }
            tokio::task::yield_now().await;
            produced = worker.process(|| {
                let Operator::HashProbe(probe) = &self.operators[op_idx] else {
                    unreachable!()
                };
                let table = &join_state.tables[probe.build_id];
                match probe.next_chunk(
                    &mut cursor,
                    table,
                    batch_size,
                    &mut join_state.scratch,
                )? {
                    OperatorStep::NeedMoreInput(None) => Ok(None),
                    OperatorStep::NeedMoreInput(Some(chunk)) => self.advance_join_chain(
                        op_idx + 1,
                        BatchView::all(chunk),
                        join_state,
                        &mut frames,
                        batch_size,
                    ),
                    OperatorStep::HaveMoreOutput(chunk) => {
                        frames.push((op_idx, cursor));
                        self.advance_join_chain(
                            op_idx + 1,
                            BatchView::all(chunk),
                            join_state,
                            &mut frames,
                            batch_size,
                        )
                    }
                }
            })?;
        }
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
                        apply_operators_view(&self.operators, BatchView::all(batch))?
                            .num_rows()
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
                            let view = apply_operators_view(
                                &self.operators,
                                BatchView::all(batch),
                            )?;
                            if !view.is_empty() {
                                local.add_view(&view, col_index)?;
                            }
                        }
                    }
                    Ok(true)
                })
            }
            WorkerSink::Aggregate(local) => worker.process(|| {
                let view = apply_operators_view(&self.operators, BatchView::all(batch))?;
                if !view.is_empty() {
                    local.add_view(&view)?;
                }
                Ok(true)
            }),
            WorkerSink::TopN(local) => worker.process(|| {
                let view = apply_operators_view(&self.operators, BatchView::all(batch))?;
                if !view.is_empty() {
                    local.add_view(&view)?;
                }
                Ok(true)
            }),
            WorkerSink::Sort(local) => worker.process(|| {
                let view = apply_operators_view(&self.operators, BatchView::all(batch))?;
                if !view.is_empty() {
                    local.add_view(view)?;
                }
                Ok(true)
            }),
            WorkerSink::HashJoinBuild(local) => worker.process(|| {
                let view = apply_operators_view(&self.operators, BatchView::all(batch))?;
                if !view.is_empty() {
                    local.add_view(&view)?;
                }
                Ok(true)
            }),
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
            WorkerSink::Aggregate(local) => return Ok(WorkerState::Aggregate(local)),
            WorkerSink::TopN(local) => {
                if output.tx.is_closed() {
                    return Ok(WorkerState::None);
                }
                let run = worker.process(|| local.finish(output.batch_size))?;
                return Ok(WorkerState::SortedRuns(run.into_iter().collect()));
            }
            WorkerSink::Sort(local) => {
                if output.tx.is_closed() {
                    return Ok(WorkerState::None);
                }
                let runs = worker.process(|| local.finish())?;
                return Ok(WorkerState::SortedRuns(runs));
            }
            WorkerSink::HashJoinBuild(local) => {
                return Ok(WorkerState::HashJoinBuild(local));
            }
            WorkerSink::Result(buffer) => buffer,
        };
        let tail_ops = if self.has_join() {
            &[][..]
        } else {
            self.operators.as_slice()
        };
        let mut tail = Vec::new();
        while !output.tx.is_closed() {
            let Some(batch) = worker.process(|| buffer.next(tail_ops, true))? else {
                if !tail.is_empty() {
                    output.tails.lock().push((tail, worker));
                }
                break;
            };
            tail.push(batch);
        }
        Ok(WorkerState::None)
    }

    fn run_build_pipelines<'a>(
        &'a self,
        label_prefix: Option<&'a str>,
        stage_inputs: &'a Arc<HashMap<StageId, Arc<MaterializedStage>>>,
        context: &'a Arc<TaskContext>,
        plan_metrics: &'a ExecutionPlanMetricsSet,
        measurement: Option<&'a Arc<RunMetrics>>,
        tx: &'a mpsc::Sender<RecordBatch>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<BuildRunOutcome>> + Send + 'a>>
    {
        Box::pin(async move {
            let probes: Vec<(usize, &Arc<HashJoinProbe>)> = self
                .operators
                .iter()
                .enumerate()
                .filter_map(|(op_idx, op)| match op {
                    Operator::HashProbe(probe) => Some((op_idx, probe)),
                    _ => None,
                })
                .collect();
            if probes.is_empty() {
                return Ok(BuildRunOutcome::Ready {
                    tables: Arc::new(Vec::new()),
                    short_circuit_op: None,
                });
            }
            let multi = probes.len() > 1;
            let mut tables = Vec::with_capacity(probes.len());
            let mut short_circuit_op = None;
            for (idx, (op_idx, probe)) in probes.iter().enumerate() {
                if tx.is_closed() {
                    return Ok(BuildRunOutcome::Cancelled);
                }
                let build_label = match (label_prefix, multi) {
                    (None, false) => "build".to_owned(),
                    (None, true) => format!("build_{idx}"),
                    (Some(prefix), false) => format!("{prefix}_build"),
                    (Some(prefix), true) => format!("{prefix}_build_{idx}"),
                };
                let Some(table) = Self::run_single_build_pipeline(
                    &probe.build_pipeline,
                    &build_label,
                    stage_inputs,
                    context,
                    plan_metrics,
                    measurement,
                    tx,
                )
                .await?
                else {
                    return Ok(BuildRunOutcome::Cancelled);
                };
                if probe.short_circuits_on_table(&table) {
                    if table.is_empty() && probe.empty_build_short_circuits() {
                        MetricBuilder::new(plan_metrics)
                            .counter("join_empty_build_short_circuits", 0)
                            .add(1);
                    } else if probe.kind.is_null_aware_anti() && table.has_null_key() {
                        MetricBuilder::new(plan_metrics)
                            .counter("join_null_aware_short_circuits", 0)
                            .add(1);
                    }
                    let has_later_build_emission = probes[idx + 1..]
                        .iter()
                        .any(|(_, p)| p.needs_build_emission());
                    if !has_later_build_emission {
                        drop(table);
                        drop(tables);
                        return Ok(BuildRunOutcome::ShortCircuitAll);
                    }
                    short_circuit_op = Some(*op_idx);
                }
                tables.push(table);
            }
            Ok(BuildRunOutcome::Ready {
                tables: Arc::new(tables),
                short_circuit_op,
            })
        })
    }

    #[expect(clippy::too_many_arguments)]
    async fn run_build_emission_worker(
        &self,
        op_idx: usize,
        build_id: usize,
        want_matched: bool,
        next_row: &AtomicUsize,
        output: WorkerOutput,
        sink_memory: Option<WorkerSinkMemory>,
        tables: Arc<Vec<Arc<FinalizedHashTable>>>,
        probe_reservation: MemoryReservation,
        mut worker: WorkerMetrics,
    ) -> Result<WorkerState> {
        let mut sink = self.worker_sink(output.batch_size, sink_memory)?;
        let mut join_state = JoinWorkerState {
            tables,
            scratch: ProbeScratch::default(),
            probe_reservation,
        };
        let Operator::HashProbe(probe) = &self.operators[op_idx] else {
            unreachable!()
        };
        let table = Arc::clone(&join_state.tables[build_id]);
        let num_rows = table.num_rows();
        let chunk_rows = output.batch_size.max(1);
        let tx = &output.tx;
        let mut indices = Vec::new();

        while !tx.is_closed() {
            let start = next_row.fetch_add(chunk_rows, Ordering::Relaxed);
            if start >= num_rows {
                break;
            }
            let end = (start + chunk_rows).min(num_rows);
            worker.claim();
            let batch_opt = worker.process(|| -> Result<_> {
                table.collect_build_emission_indices(
                    start,
                    end,
                    want_matched,
                    &mut indices,
                );
                if indices.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(
                        probe.materialize_build_emission_batch(&table, &indices)?,
                    ))
                }
            })?;
            if let Some(batch) = batch_opt {
                let cont = self
                    .consume_with_joins_from(
                        op_idx + 1,
                        batch,
                        tx,
                        output.batch_size,
                        &mut sink,
                        &mut join_state,
                        &mut worker,
                    )
                    .await?;
                if !cont {
                    return Ok(WorkerState::None);
                }
            }
            worker.complete();
            tokio::task::yield_now().await;
        }
        self.finish_worker(sink, &output, worker)
    }

    #[expect(clippy::too_many_arguments)]
    async fn run_build_emission_pipelines(
        self: &Arc<Self>,
        finalized_tables: &Arc<Vec<Arc<FinalizedHashTable>>>,
        short_circuit_op: Option<usize>,
        label_prefix: Option<&str>,
        output: &WorkerOutput,
        context: &Arc<TaskContext>,
        plan_metrics: &ExecutionPlanMetricsSet,
        measurement: Option<&Arc<RunMetrics>>,
        targets: &mut BuildEmissionTargets<'_>,
    ) -> Result<bool> {
        let multi = finalized_tables.len() > 1;
        for (op_idx, op) in self.operators.iter().enumerate() {
            let Operator::HashProbe(probe) = op else {
                continue;
            };
            if !probe.needs_build_emission() {
                continue;
            }
            if short_circuit_op.is_some_and(|sc| op_idx <= sc) {
                continue;
            }
            let table = &finalized_tables[probe.build_id];
            if table.is_empty() {
                continue;
            }
            if output.tx.is_closed() {
                return Ok(false);
            }
            #[cfg(test)]
            if let Some(gate) = &probe.build_emit_gate {
                gate.wait().await;
            }
            if output.tx.is_closed() {
                return Ok(false);
            }

            MetricBuilder::new(plan_metrics)
                .counter("join_build_emission_pipelines", 0)
                .add(1);

            let emit_label = match (label_prefix, multi) {
                (None, false) => "build_emit".to_owned(),
                (None, true) => format!("build_emit_{}", probe.build_id),
                (Some(prefix), false) => format!("{prefix}_emit"),
                (Some(prefix), true) => format!("{prefix}_emit_{}", probe.build_id),
            };
            let chunk_rows = output.batch_size.max(1);
            let num_chunks = table.num_rows().div_ceil(chunk_rows);
            let worker_count = self.workers.get().min(num_chunks).max(1);
            let next_row = Arc::new(AtomicUsize::new(0));
            let want_matched = probe.kind.emit_matched_build();
            let build_id = probe.build_id;
            let probe_reservation = MemoryConsumer::new("MorselHashJoinProbe")
                .register(context.memory_pool());

            let mut tasks = JoinSet::new();
            for id in 0..worker_count {
                let pipeline = Arc::clone(self);
                let next_row = Arc::clone(&next_row);
                let out = output.clone();
                let tables = Arc::clone(finalized_tables);
                let probe_res = probe_reservation.new_empty();
                let sink_memory = if let Some(shared) = targets.shared_sort_memory {
                    Some(WorkerSinkMemory::SharedSort(Arc::clone(shared)))
                } else {
                    targets
                        .sink_reservation
                        .map(|r| WorkerSinkMemory::Reservation(r.new_empty()))
                };
                let worker =
                    WorkerMetrics::for_pipeline(measurement.cloned(), id, &emit_label);
                tasks.spawn(async move {
                    pipeline
                        .run_build_emission_worker(
                            op_idx,
                            build_id,
                            want_matched,
                            &next_row,
                            out,
                            sink_memory,
                            tables,
                            probe_res,
                            worker,
                        )
                        .await
                });
            }
            drop(probe_reservation);

            while let Some(result) = tasks.join_next().await {
                let local = match result {
                    Ok(Ok(local)) => local,
                    Ok(Err(err)) => {
                        abort_and_drain(&mut tasks).await;
                        return Err(err);
                    }
                    Err(err) if err.is_panic() => {
                        abort_and_drain(&mut tasks).await;
                        std::panic::resume_unwind(err.into_panic())
                    }
                    Err(err) => {
                        abort_and_drain(&mut tasks).await;
                        return Err(exec_datafusion_err!(
                            "Morsel join build-emission task failed: {err}"
                        ));
                    }
                };
                match local {
                    WorkerState::None => {}
                    WorkerState::Count(local) => {
                        if let Some(combined) = targets.combined_count.as_deref_mut()
                            && let Err(err) = combined.combine(local)
                        {
                            abort_and_drain(&mut tasks).await;
                            return Err(err);
                        }
                    }
                    WorkerState::GroupCount(local) => {
                        if let Some(Some(combined)) =
                            targets.combined_group.as_deref_mut()
                        {
                            match combined.merge_from(local, &output.tx).await {
                                Ok(true) => {}
                                Ok(false) => {
                                    abort_and_drain(&mut tasks).await;
                                    return Ok(false);
                                }
                                Err(err) => {
                                    abort_and_drain(&mut tasks).await;
                                    return Err(err);
                                }
                            }
                        }
                    }
                    WorkerState::Aggregate(local) => {
                        if let Some(Some(combined)) = targets.combined_agg.as_deref_mut()
                        {
                            match combined.merge_from(local, &output.tx).await {
                                Ok(true) => {}
                                Ok(false) => {
                                    abort_and_drain(&mut tasks).await;
                                    return Ok(false);
                                }
                                Err(err) => {
                                    abort_and_drain(&mut tasks).await;
                                    return Err(err);
                                }
                            }
                        }
                    }
                    WorkerState::SortedRuns(runs) => {
                        if let Some(sorted_runs) = targets.sorted_runs.as_deref_mut() {
                            sorted_runs.extend(runs);
                        }
                        if output.tx.is_closed() {
                            abort_and_drain(&mut tasks).await;
                            return Ok(false);
                        }
                    }
                    WorkerState::HashJoinBuild(local) => {
                        if let Some(combined_build) =
                            targets.combined_build.as_deref_mut()
                        {
                            combined_build.combine(local);
                        }
                        if output.tx.is_closed() {
                            abort_and_drain(&mut tasks).await;
                            return Ok(false);
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    async fn run_single_build_pipeline(
        build_pipeline: &Arc<Pipeline>,
        build_label: &str,
        stage_inputs: &Arc<HashMap<StageId, Arc<MaterializedStage>>>,
        context: &Arc<TaskContext>,
        plan_metrics: &ExecutionPlanMetricsSet,
        measurement: Option<&Arc<RunMetrics>>,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<Arc<FinalizedHashTable>>> {
        let Sink::HashJoinBuild(build_sink) = &build_pipeline.sink else {
            return internal_err!("Expected HashJoinBuild sink on join build pipeline");
        };

        let (nested_tables, nested_short_circuit_op) = if build_pipeline.has_join() {
            match build_pipeline
                .run_build_pipelines(
                    Some(build_label),
                    stage_inputs,
                    context,
                    plan_metrics,
                    measurement,
                    tx,
                )
                .await?
            {
                BuildRunOutcome::Ready {
                    tables,
                    short_circuit_op,
                } => (Some(tables), short_circuit_op),
                BuildRunOutcome::Cancelled => return Ok(None),
                BuildRunOutcome::ShortCircuitAll => {
                    let empty_res = MemoryConsumer::new("MorselHashJoinBuild")
                        .register(context.memory_pool());
                    let empty_state = build_sink.new_state(empty_res);
                    let finalized = build_sink.finalize(empty_state, tx).await?;
                    if finalized.is_some() {
                        MetricBuilder::new(plan_metrics)
                            .counter("join_build_finalizations", 0)
                            .add(1);
                    }
                    return Ok(finalized);
                }
            }
        } else {
            (None, None)
        };

        let build_reservation =
            MemoryConsumer::new("MorselHashJoinBuild").register(context.memory_pool());
        let nested_probe_reservation = nested_tables.as_ref().map(|_| {
            MemoryConsumer::new("MorselHashJoinProbe").register(context.memory_pool())
        });
        let mut combined_build = build_sink.new_state(build_reservation.new_empty());
        let build_output = WorkerOutput {
            tx: tx.clone(),
            batch_size: context.session_config().batch_size(),
            tails: Arc::default(),
        };

        let mut combine_duration = std::time::Duration::ZERO;
        if nested_short_circuit_op.is_none() {
            let mut tasks = JoinSet::new();
            match &build_pipeline.source {
                Source::Memory(source) => {
                    let next = Arc::new(AtomicUsize::new(0));
                    let morsels = source.batches.len().div_ceil(BATCHES_PER_MORSEL);
                    for id in 0..build_pipeline.workers.get().min(morsels) {
                        let (pipeline, next, output) = (
                            Arc::clone(build_pipeline),
                            Arc::clone(&next),
                            build_output.clone(),
                        );
                        let reservation = Some(WorkerSinkMemory::Reservation(
                            build_reservation.new_empty(),
                        ));
                        let join_ctx = nested_tables.as_ref().map(|t| {
                            (
                                Arc::clone(t),
                                nested_probe_reservation
                                    .as_ref()
                                    .expect("nested probe reservation")
                                    .new_empty(),
                            )
                        });
                        let worker = WorkerMetrics::for_pipeline(
                            measurement.cloned(),
                            id,
                            build_label,
                        );
                        tasks.spawn(async move {
                            let source = match &pipeline.source {
                                Source::Memory(source) => source,
                                #[cfg(feature = "parquet")]
                                Source::Parquet(_) => unreachable!(),
                                Source::Stage(_) => unreachable!(),
                            };
                            pipeline
                                .run_memory_worker(
                                    source,
                                    &next,
                                    output,
                                    reservation,
                                    join_ctx,
                                    worker,
                                )
                                .await
                        });
                    }
                }
                Source::Stage(stage_src) => {
                    let stage_data = Arc::clone(
                        stage_inputs
                            .get(&stage_src.stage_id)
                            .expect("bound upstream stage artifact"),
                    );
                    let next = Arc::new(AtomicUsize::new(0));
                    let morsels = stage_data.batches.len().div_ceil(BATCHES_PER_MORSEL);
                    for id in 0..build_pipeline.workers.get().min(morsels) {
                        let (pipeline, stage_data, next, output) = (
                            Arc::clone(build_pipeline),
                            Arc::clone(&stage_data),
                            Arc::clone(&next),
                            build_output.clone(),
                        );
                        let reservation = Some(WorkerSinkMemory::Reservation(
                            build_reservation.new_empty(),
                        ));
                        let join_ctx = nested_tables.as_ref().map(|t| {
                            (
                                Arc::clone(t),
                                nested_probe_reservation
                                    .as_ref()
                                    .expect("nested probe reservation")
                                    .new_empty(),
                            )
                        });
                        let worker = WorkerMetrics::for_pipeline(
                            measurement.cloned(),
                            id,
                            build_label,
                        );
                        tasks.spawn(async move {
                            pipeline
                                .run_stage_worker(
                                    &stage_data.batches,
                                    &next,
                                    false,
                                    output,
                                    reservation,
                                    join_ctx,
                                    worker,
                                )
                                .await
                        });
                    }
                }
                #[cfg(feature = "parquet")]
                Source::Parquet(source) => {
                    let ready = source.start(
                        context,
                        &mut tasks,
                        build_pipeline.workers,
                        measurement,
                    )?;
                    for id in 0..build_pipeline.workers.get() {
                        let (pipeline, ready, output) = (
                            Arc::clone(build_pipeline),
                            ready.clone(),
                            build_output.clone(),
                        );
                        let reservation = Some(WorkerSinkMemory::Reservation(
                            build_reservation.new_empty(),
                        ));
                        let join_ctx = nested_tables.as_ref().map(|t| {
                            (
                                Arc::clone(t),
                                nested_probe_reservation
                                    .as_ref()
                                    .expect("nested probe reservation")
                                    .new_empty(),
                            )
                        });
                        let worker = WorkerMetrics::for_pipeline(
                            measurement.cloned(),
                            id,
                            build_label,
                        );
                        tasks.spawn(async move {
                            pipeline
                                .run_parquet_worker(
                                    ready,
                                    output,
                                    reservation,
                                    join_ctx,
                                    worker,
                                )
                                .await
                        });
                    }
                }
            }
            drop(nested_probe_reservation);

            while let Some(result) = tasks.join_next().await {
                let local = match result {
                    Ok(Ok(local)) => local,
                    Ok(Err(err)) => {
                        drop(combined_build);
                        abort_and_drain(&mut tasks).await;
                        drop(nested_tables);
                        return Err(err);
                    }
                    Err(err) if err.is_panic() => {
                        drop(combined_build);
                        abort_and_drain(&mut tasks).await;
                        drop(nested_tables);
                        std::panic::resume_unwind(err.into_panic())
                    }
                    Err(err) => {
                        drop(combined_build);
                        abort_and_drain(&mut tasks).await;
                        drop(nested_tables);
                        return Err(exec_datafusion_err!(
                            "Morsel build task failed: {err}"
                        ));
                    }
                };
                if let WorkerState::HashJoinBuild(local) = local {
                    let start = measurement.map(|_| Instant::now());
                    combined_build.combine(local);
                    if let Some(start) = start {
                        combine_duration += start.elapsed();
                    }
                }
                if tx.is_closed() {
                    drop(combined_build);
                    abort_and_drain(&mut tasks).await;
                    drop(nested_tables);
                    return Ok(None);
                }
            }
        } else {
            drop(nested_probe_reservation);
        }

        if let Some(tables) = &nested_tables {
            let mut targets = BuildEmissionTargets {
                combined_count: None,
                combined_group: None,
                combined_agg: None,
                sorted_runs: None,
                combined_build: Some(&mut combined_build),
                sink_reservation: Some(&build_reservation),
                shared_sort_memory: None,
            };
            let cont = build_pipeline
                .run_build_emission_pipelines(
                    tables,
                    nested_short_circuit_op,
                    Some(build_label),
                    &build_output,
                    context,
                    plan_metrics,
                    measurement,
                    &mut targets,
                )
                .await;
            match cont {
                Ok(true) => {}
                Ok(false) => {
                    drop(combined_build);
                    drop(build_reservation);
                    drop(nested_tables);
                    return Ok(None);
                }
                Err(err) => {
                    drop(combined_build);
                    drop(build_reservation);
                    drop(nested_tables);
                    return Err(err);
                }
            }
        }
        drop(build_reservation);
        drop(nested_tables);

        if tx.is_closed() {
            return Ok(None);
        }
        #[cfg(test)]
        if let Some(gate) = &build_sink.finalize_gate {
            gate.wait().await;
        }
        if tx.is_closed() {
            return Ok(None);
        }

        let finalize_start = measurement.map(|_| Instant::now());
        let finalized = build_sink.finalize(combined_build, tx).await?;
        if let Some(run) = measurement {
            run.record_pipeline_time(
                build_label,
                "morsel_join_combine_work",
                combine_duration,
            );
            if let Some(start) = finalize_start {
                run.record_pipeline_time(
                    build_label,
                    "morsel_join_finalize_work",
                    start.elapsed(),
                );
            }
        }
        if finalized.is_some() {
            MetricBuilder::new(plan_metrics)
                .counter("join_build_finalizations", 0)
                .add(1);
        }
        Ok(finalized)
    }

    fn post_sort(&self) -> Option<&PostSort> {
        match &self.sink {
            Sink::Count { post_sort, .. }
            | Sink::GroupCount { post_sort, .. }
            | Sink::Aggregate { post_sort, .. } => post_sort.as_ref(),
            Sink::Result
            | Sink::TopN { .. }
            | Sink::Sort { .. }
            | Sink::Limit { .. }
            | Sink::HashJoinBuild(_) => None,
        }
    }

    async fn finalize_direct_sort(
        sink: &SortSink,
        result_operators: &[Operator],
        sorted_runs: Vec<SortedRun>,
        fin: SortFinalizeCtx<'_>,
        tx: mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if tx.is_closed() {
            return Ok(());
        }
        #[cfg(test)]
        if let Some(gate) = &sink.finalize_gate {
            gate.wait().await;
        }
        if tx.is_closed() {
            return Ok(());
        }
        let finalize_start = fin.measurement.map(|_| Instant::now());
        MetricBuilder::new(fin.metrics)
            .counter("sort_finalizations", 0)
            .add(1);
        sink.finalize_and_emit(
            sorted_runs,
            result_operators,
            fin.batch_size,
            fin.context.memory_pool(),
            tx,
            fin.measurement,
        )
        .await?;
        if let (Some(run), Some(start)) = (fin.measurement, finalize_start) {
            run.record_pipeline_time(
                "sort",
                "morsel_sort_finalize_work",
                start.elapsed(),
            );
        }
        Ok(())
    }

    async fn finalize_direct_topn(
        sink: &TopNSink,
        result_operators: &[Operator],
        sorted_runs: Vec<SortedRun>,
        fin: SortFinalizeCtx<'_>,
        tx: mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if tx.is_closed() {
            return Ok(());
        }
        #[cfg(test)]
        if let Some(gate) = &sink.finalize_gate {
            gate.wait().await;
        }
        if tx.is_closed() {
            return Ok(());
        }
        let finalize_start = fin.measurement.map(|_| Instant::now());
        MetricBuilder::new(fin.metrics)
            .counter("topn_finalizations", 0)
            .add(1);
        sink.finalize_and_emit(
            sorted_runs,
            result_operators,
            fin.batch_size,
            fin.context.memory_pool(),
            tx,
            fin.measurement,
        )
        .await?;
        if let (Some(run), Some(start)) = (fin.measurement, finalize_start) {
            run.record_pipeline_time(
                "topn",
                "morsel_topn_finalize_work",
                start.elapsed(),
            );
        }
        Ok(())
    }

    async fn run_post_sort_stage(
        self: &Arc<Self>,
        finalized_agg: FinalizedAggOutput,
        batch_size: usize,
        context: &Arc<TaskContext>,
        metrics: &ExecutionPlanMetricsSet,
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<&Arc<RunMetrics>>,
    ) -> Result<()> {
        if tx.is_closed() {
            return Ok(());
        }
        let Some(post_sort) = self.post_sort() else {
            return internal_err!("Expected PostSort stage on aggregate sink");
        };
        let (consumer_name, stage_label) = match post_sort {
            PostSort::TopN { .. } => ("MorselTopN", "topn"),
            PostSort::Sort { .. } => ("MorselSort", "sort"),
        };
        let sort_workers = match &finalized_agg {
            FinalizedAggOutput::Count(_) => 1,
            FinalizedAggOutput::GroupCount(_) | FinalizedAggOutput::Aggregate(_) => {
                self.workers.get()
            }
        };

        let (agg_tx, agg_rx) = mpsc::channel(result_capacity(self.workers));
        let agg_rx = Arc::new(tokio::sync::Mutex::new(agg_rx));
        let mut tasks: JoinSet<Result<Vec<SortedRun>>> = JoinSet::new();

        // Producer task: streams finalized aggregate output through agg_result_operators.
        {
            let pipeline = Arc::clone(self);
            tasks.spawn(async move {
                match finalized_agg {
                    FinalizedAggOutput::Count(finalized) => {
                        let Sink::Count {
                            result_operators, ..
                        } = &pipeline.sink
                        else {
                            unreachable!()
                        };
                        Self::run_count_result(finalized, result_operators, agg_tx, None)
                            .await?;
                    }
                    FinalizedAggOutput::GroupCount(finalized) => {
                        let Sink::GroupCount {
                            result_operators, ..
                        } = &pipeline.sink
                        else {
                            unreachable!()
                        };
                        finalized
                            .emit(result_operators, batch_size, agg_tx, None)
                            .await?;
                    }
                    FinalizedAggOutput::Aggregate(finalized) => {
                        let Sink::Aggregate {
                            result_operators, ..
                        } = &pipeline.sink
                        else {
                            unreachable!()
                        };
                        finalized
                            .emit(result_operators, batch_size, agg_tx, None)
                            .await?;
                    }
                }
                Ok(Vec::new())
            });
        }

        // Consumer tasks: dynamically claim finalized aggregate batches and feed local TopN / Sort.
        let stage_reservation =
            MemoryConsumer::new(consumer_name).register(context.memory_pool());
        let shared_sort_memory = matches!(post_sort, PostSort::Sort { .. })
            .then(|| SharedSortMemory::new(stage_reservation.new_empty()));
        for id in 0..sort_workers {
            let pipeline = Arc::clone(self);
            let agg_rx = Arc::clone(&agg_rx);
            let output_tx = tx.clone();
            let reservation = stage_reservation.new_empty();
            let shared_sort = shared_sort_memory.as_ref().map(Arc::clone);
            let mut worker =
                WorkerMetrics::for_pipeline(measurement.cloned(), id, stage_label);
            tasks.spawn(async move {
                let Some(ps) = pipeline.post_sort() else {
                    unreachable!()
                };
                enum LocalPostState {
                    TopN(TopNState),
                    Sort(SortState),
                }
                let mut local = match ps {
                    PostSort::TopN { sink, .. } => {
                        LocalPostState::TopN(sink.new_state(reservation)?)
                    }
                    PostSort::Sort { sink, .. } => LocalPostState::Sort(
                        sink.new_state(shared_sort.expect("shared sort memory")),
                    ),
                };
                loop {
                    let maybe_batch = tokio::select! {
                        biased;
                        _ = output_tx.closed() => return Ok(Vec::new()),
                        batch = async { agg_rx.lock().await.recv().await } => batch,
                    };
                    let Some(batch) = maybe_batch else {
                        break;
                    };
                    if output_tx.is_closed() {
                        return Ok(Vec::new());
                    }
                    worker.claim();
                    worker.process(|| match &mut local {
                        LocalPostState::TopN(s) => s.add_view(&BatchView::all(batch)),
                        LocalPostState::Sort(s) => s.add_view(BatchView::all(batch)),
                    })?;
                    worker.complete();
                    tokio::task::yield_now().await;
                }
                if output_tx.is_closed() {
                    return Ok(Vec::new());
                }
                let runs = worker.process(|| match local {
                    LocalPostState::TopN(s) => {
                        Ok(s.finish(batch_size)?.into_iter().collect())
                    }
                    LocalPostState::Sort(s) => s.finish(),
                })?;
                Ok(runs)
            });
        }
        drop(stage_reservation);
        drop(shared_sort_memory);
        drop(agg_rx);

        let mut sorted_runs = Vec::new();
        while let Some(result) = tasks.join_next().await {
            let runs = match result {
                Ok(Ok(runs)) => runs,
                Ok(Err(err)) => {
                    drop(sorted_runs);
                    abort_and_drain(&mut tasks).await;
                    return Err(err);
                }
                Err(err) if err.is_panic() => {
                    drop(sorted_runs);
                    abort_and_drain(&mut tasks).await;
                    std::panic::resume_unwind(err.into_panic())
                }
                Err(err) => {
                    drop(sorted_runs);
                    abort_and_drain(&mut tasks).await;
                    return Err(exec_datafusion_err!(
                        "Morsel post-sort task failed: {err}"
                    ));
                }
            };
            sorted_runs.extend(runs);
            if tx.is_closed() {
                drop(sorted_runs);
                abort_and_drain(&mut tasks).await;
                return Ok(());
            }
        }

        let fin = SortFinalizeCtx {
            batch_size,
            context,
            metrics,
            measurement: measurement.map(Arc::as_ref),
        };
        match post_sort {
            PostSort::TopN {
                sink,
                result_operators,
            } => {
                Self::finalize_direct_topn(sink, result_operators, sorted_runs, fin, tx)
                    .await
            }
            PostSort::Sort {
                sink,
                result_operators,
            } => {
                Self::finalize_direct_sort(sink, result_operators, sorted_runs, fin, tx)
                    .await
            }
        }
    }

    async fn run_count_result(
        finalized: FinalizedCount,
        operators: &[Operator],
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        if !tx.is_closed() {
            let batch = apply_operators(operators, finalized.into_batch())?;
            if batch.num_rows() > 0 {
                if let Some(run) = measurement {
                    metrics::send(run, None, &tx, batch).await;
                } else {
                    let _ = tx.send(batch).await;
                }
            }
        }
        Ok(())
    }

    #[expect(clippy::too_many_arguments)]
    pub(super) async fn run_pipeline_to_tx(
        self: Arc<Self>,
        label_prefix: Option<&str>,
        stage_inputs: &Arc<HashMap<StageId, Arc<MaterializedStage>>>,
        context: &Arc<TaskContext>,
        metrics: &ExecutionPlanMetricsSet,
        measurement: Option<&Arc<RunMetrics>>,
        tx: mpsc::Sender<RecordBatch>,
        cancel_tx: mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if let Sink::Limit {
            skip,
            fetch,
            result_operators,
        } = &self.sink
        {
            if *fetch == Some(0) || cancel_tx.is_closed() || tx.is_closed() {
                return Ok(());
            }
            let (limit_tx, mut limit_rx) = mpsc::channel(result_capacity(self.workers));
            let mut inner_tasks: JoinSet<Result<()>> = JoinSet::new();
            {
                let pipeline = Arc::clone(&self);
                let stage_inputs = Arc::clone(stage_inputs);
                let ctx = Arc::clone(context);
                let metrics = metrics.clone();
                let meas = measurement.cloned();
                let label = label_prefix.map(str::to_owned);
                let cancel = cancel_tx.clone();
                inner_tasks.spawn(async move {
                    pipeline
                        .run_pipeline_body_to_tx(
                            label.as_deref(),
                            &stage_inputs,
                            &ctx,
                            &metrics,
                            meas.as_ref(),
                            limit_tx,
                            cancel,
                        )
                        .await
                });
            }

            let mut remaining_skip = *skip;
            let mut remaining_fetch = *fetch;
            let mut inner_done = false;

            loop {
                let maybe_batch = if !inner_done {
                    tokio::select! {
                        biased;
                        _ = cancel_tx.closed() => {
                            drop(limit_rx);
                            abort_and_drain(&mut inner_tasks).await;
                            return Ok(());
                        }
                        result = inner_tasks.join_next() => {
                            inner_done = true;
                            match result {
                                Some(Ok(Ok(()))) | None => continue,
                                Some(Ok(Err(err))) => {
                                    drop(limit_rx);
                                    return Err(err);
                                }
                                Some(Err(err)) if err.is_panic() => {
                                    drop(limit_rx);
                                    std::panic::resume_unwind(err.into_panic())
                                }
                                Some(Err(err)) => {
                                    drop(limit_rx);
                                    return Err(exec_datafusion_err!(
                                        "Morsel limit task failed: {err}"
                                    ));
                                }
                            }
                        }
                        batch = limit_rx.recv() => batch,
                    }
                } else {
                    limit_rx.recv().await
                };

                let Some(mut batch) = maybe_batch else {
                    break;
                };
                if cancel_tx.is_closed() || tx.is_closed() {
                    drop(limit_rx);
                    abort_and_drain(&mut inner_tasks).await;
                    return Ok(());
                }
                if batch.num_rows() == 0 {
                    continue;
                }
                if remaining_skip >= batch.num_rows() {
                    remaining_skip -= batch.num_rows();
                    continue;
                }
                if remaining_skip > 0 {
                    batch =
                        batch.slice(remaining_skip, batch.num_rows() - remaining_skip);
                    remaining_skip = 0;
                }
                if let Some(rem) = remaining_fetch.as_mut() {
                    if *rem == 0 {
                        drop(limit_rx);
                        abort_and_drain(&mut inner_tasks).await;
                        return Ok(());
                    }
                    if batch.num_rows() > *rem {
                        batch = batch.slice(0, *rem);
                    }
                    *rem -= batch.num_rows();
                }
                let out = apply_operators(result_operators, batch)?;
                if out.num_rows() > 0 {
                    let sent = if let Some(run) = measurement {
                        metrics::send(run, None, &tx, out).await
                    } else {
                        tx.send(out).await.is_ok()
                    };
                    if !sent {
                        drop(limit_rx);
                        abort_and_drain(&mut inner_tasks).await;
                        return Ok(());
                    }
                }
                if remaining_fetch == Some(0) {
                    drop(limit_rx);
                    abort_and_drain(&mut inner_tasks).await;
                    return Ok(());
                }
            }

            if !inner_done {
                while let Some(result) = inner_tasks.join_next().await {
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => return Err(err),
                        Err(err) if err.is_panic() => {
                            std::panic::resume_unwind(err.into_panic())
                        }
                        Err(err) => {
                            return Err(exec_datafusion_err!(
                                "Morsel limit task failed: {err}"
                            ));
                        }
                    }
                }
            }
            return Ok(());
        }

        self.run_pipeline_body_to_tx(
            label_prefix,
            stage_inputs,
            context,
            metrics,
            measurement,
            tx,
            cancel_tx,
        )
        .await
    }

    #[expect(clippy::too_many_arguments)]
    async fn run_pipeline_body_to_tx(
        self: Arc<Self>,
        label_prefix: Option<&str>,
        stage_inputs: &Arc<HashMap<StageId, Arc<MaterializedStage>>>,
        context: &Arc<TaskContext>,
        metrics: &ExecutionPlanMetricsSet,
        measurement: Option<&Arc<RunMetrics>>,
        tx: mpsc::Sender<RecordBatch>,
        cancel_tx: mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        let output = WorkerOutput {
            tx,
            batch_size: context.session_config().batch_size(),
            tails: Arc::default(),
        };
        let has_join = self.has_join();

        if cancel_tx.is_closed() || output.tx.is_closed() {
            return Ok(());
        }

        // Phase 1: execute all join build pipelines and finalize hash tables
        // strictly before any probe work begins.
        let (finalized_tables, short_circuit_op, skip_all) = if has_join {
            match self
                .run_build_pipelines(
                    label_prefix,
                    stage_inputs,
                    context,
                    metrics,
                    measurement,
                    &cancel_tx,
                )
                .await?
            {
                BuildRunOutcome::Ready {
                    tables,
                    short_circuit_op,
                } => (Some(tables), short_circuit_op, false),
                BuildRunOutcome::ShortCircuitAll => (None, None, true),
                BuildRunOutcome::Cancelled => return Ok(()),
            }
        } else {
            (None, None, false)
        };

        let sink_reservation = match &self.sink {
            Sink::GroupCount { .. } => Some(
                MemoryConsumer::new("MorselGroupCount").register(context.memory_pool()),
            ),
            Sink::Aggregate { .. } => Some(
                MemoryConsumer::new("MorselAggregate").register(context.memory_pool()),
            ),
            Sink::TopN { .. } => {
                Some(MemoryConsumer::new("MorselTopN").register(context.memory_pool()))
            }
            Sink::Sort { .. } => None,
            Sink::Result
            | Sink::Count { .. }
            | Sink::Limit { .. }
            | Sink::HashJoinBuild(_) => None,
        };
        let shared_sort_memory = matches!(&self.sink, Sink::Sort { .. }).then(|| {
            SharedSortMemory::new(
                MemoryConsumer::new("MorselSort").register(context.memory_pool()),
            )
        });
        let mut combined_group = match (&self.sink, &sink_reservation) {
            (Sink::GroupCount { sink, .. }, Some(reservation)) => {
                Some(sink.new_state(reservation.new_empty()))
            }
            _ => None,
        };
        let mut combined_agg = match (&self.sink, &sink_reservation) {
            (Sink::Aggregate { sink, .. }, Some(reservation)) => {
                Some(sink.new_state(reservation.new_empty())?)
            }
            _ => None,
        };
        let mut combined_count = CountState::default();
        let mut sorted_runs: Vec<SortedRun> = Vec::new();

        // Phase 2: execute probe/scan pipeline unless short-circuited by an
        // empty or null-aware build table.
        let skip_probe = skip_all || short_circuit_op.is_some();
        if !skip_probe {
            let probe_reservation = finalized_tables.as_ref().map(|_| {
                MemoryConsumer::new("MorselHashJoinProbe").register(context.memory_pool())
            });
            let mut tasks = JoinSet::new();
            let make_worker_metrics = |id: usize| match (label_prefix, has_join) {
                (Some(prefix), true) => WorkerMetrics::for_pipeline(
                    measurement.cloned(),
                    id,
                    format!("{prefix}_probe"),
                ),
                (Some(prefix), false) => {
                    WorkerMetrics::for_pipeline(measurement.cloned(), id, prefix)
                }
                (None, true) => {
                    WorkerMetrics::for_pipeline(measurement.cloned(), id, "probe")
                }
                (None, false) => WorkerMetrics::new(measurement.cloned(), id),
            };
            match &self.source {
                Source::Memory(source) => {
                    let next = Arc::new(AtomicUsize::new(0));
                    let morsels = source.batches.len().div_ceil(BATCHES_PER_MORSEL);
                    for id in 0..self.workers.get().min(morsels) {
                        let (pipeline, next, output) =
                            (Arc::clone(&self), Arc::clone(&next), output.clone());
                        let sink_memory = if let Some(shared) = &shared_sort_memory {
                            Some(WorkerSinkMemory::SharedSort(Arc::clone(shared)))
                        } else {
                            sink_reservation
                                .as_ref()
                                .map(|r| WorkerSinkMemory::Reservation(r.new_empty()))
                        };
                        let join_ctx = finalized_tables.as_ref().map(|t| {
                            (
                                Arc::clone(t),
                                probe_reservation
                                    .as_ref()
                                    .expect("probe reservation")
                                    .new_empty(),
                            )
                        });
                        let worker = make_worker_metrics(id);
                        tasks.spawn(async move {
                            let source = match &pipeline.source {
                                Source::Memory(source) => source,
                                #[cfg(feature = "parquet")]
                                Source::Parquet(_) => unreachable!(),
                                Source::Stage(_) => unreachable!(),
                            };
                            pipeline
                                .run_memory_worker(
                                    source,
                                    &next,
                                    output,
                                    sink_memory,
                                    join_ctx,
                                    worker,
                                )
                                .await
                        });
                    }
                }
                Source::Stage(stage_src) => {
                    let stage_data = Arc::clone(
                        stage_inputs
                            .get(&stage_src.stage_id)
                            .expect("bound upstream stage artifact"),
                    );
                    let sequential = stage_src.ordered
                        && matches!(self.sink, Sink::Result | Sink::Limit { .. });
                    let next = Arc::new(AtomicUsize::new(0));
                    let worker_count = if sequential {
                        usize::from(!stage_data.batches.is_empty())
                    } else {
                        let morsels =
                            stage_data.batches.len().div_ceil(BATCHES_PER_MORSEL);
                        self.workers.get().min(morsels)
                    };
                    for id in 0..worker_count {
                        let (pipeline, stage_data, next, output) = (
                            Arc::clone(&self),
                            Arc::clone(&stage_data),
                            Arc::clone(&next),
                            output.clone(),
                        );
                        let sink_memory = if let Some(shared) = &shared_sort_memory {
                            Some(WorkerSinkMemory::SharedSort(Arc::clone(shared)))
                        } else {
                            sink_reservation
                                .as_ref()
                                .map(|r| WorkerSinkMemory::Reservation(r.new_empty()))
                        };
                        let join_ctx = finalized_tables.as_ref().map(|t| {
                            (
                                Arc::clone(t),
                                probe_reservation
                                    .as_ref()
                                    .expect("probe reservation")
                                    .new_empty(),
                            )
                        });
                        let worker = make_worker_metrics(id);
                        tasks.spawn(async move {
                            pipeline
                                .run_stage_worker(
                                    &stage_data.batches,
                                    &next,
                                    sequential,
                                    output,
                                    sink_memory,
                                    join_ctx,
                                    worker,
                                )
                                .await
                        });
                    }
                }
                #[cfg(feature = "parquet")]
                Source::Parquet(source) => {
                    let ready =
                        source.start(context, &mut tasks, self.workers, measurement)?;
                    for id in 0..self.workers.get() {
                        let (pipeline, ready, output) =
                            (Arc::clone(&self), ready.clone(), output.clone());
                        let sink_memory = if let Some(shared) = &shared_sort_memory {
                            Some(WorkerSinkMemory::SharedSort(Arc::clone(shared)))
                        } else {
                            sink_reservation
                                .as_ref()
                                .map(|r| WorkerSinkMemory::Reservation(r.new_empty()))
                        };
                        let join_ctx = finalized_tables.as_ref().map(|t| {
                            (
                                Arc::clone(t),
                                probe_reservation
                                    .as_ref()
                                    .expect("probe reservation")
                                    .new_empty(),
                            )
                        });
                        let worker = make_worker_metrics(id);
                        tasks.spawn(async move {
                            pipeline
                                .run_parquet_worker(
                                    ready,
                                    output,
                                    sink_memory,
                                    join_ctx,
                                    worker,
                                )
                                .await
                        });
                    }
                }
            }
            drop(probe_reservation);
            while let Some(result) = tasks.join_next().await {
                let local = match result {
                    Ok(Ok(local)) => local,
                    Ok(Err(err)) => {
                        drop(combined_group);
                        drop(combined_agg);
                        drop(sorted_runs);
                        abort_and_drain(&mut tasks).await;
                        drop(sink_reservation);
                        drop(shared_sort_memory);
                        drop(finalized_tables);
                        return Err(err);
                    }
                    Err(err) if err.is_panic() => {
                        drop(combined_group);
                        drop(combined_agg);
                        drop(sorted_runs);
                        abort_and_drain(&mut tasks).await;
                        drop(sink_reservation);
                        drop(shared_sort_memory);
                        drop(finalized_tables);
                        std::panic::resume_unwind(err.into_panic())
                    }
                    Err(err) => {
                        drop(combined_group);
                        drop(combined_agg);
                        drop(sorted_runs);
                        abort_and_drain(&mut tasks).await;
                        drop(sink_reservation);
                        drop(shared_sort_memory);
                        drop(finalized_tables);
                        return Err(exec_datafusion_err!("Morsel task failed: {err}"));
                    }
                };
                match local {
                    WorkerState::None | WorkerState::HashJoinBuild(_) => {}
                    WorkerState::Count(local) => {
                        if let Err(err) = combined_count.combine(local) {
                            drop(combined_group);
                            drop(combined_agg);
                            drop(sorted_runs);
                            abort_and_drain(&mut tasks).await;
                            drop(sink_reservation);
                            drop(shared_sort_memory);
                            drop(finalized_tables);
                            return Err(err);
                        }
                    }
                    WorkerState::GroupCount(local) => {
                        if let Some(combined) = combined_group.as_mut() {
                            match combined.merge_from(local, &cancel_tx).await {
                                Ok(true) => {}
                                Ok(false) => {
                                    drop(combined_group);
                                    drop(combined_agg);
                                    drop(sorted_runs);
                                    abort_and_drain(&mut tasks).await;
                                    drop(sink_reservation);
                                    drop(shared_sort_memory);
                                    drop(finalized_tables);
                                    return Ok(());
                                }
                                Err(err) => {
                                    drop(combined_group);
                                    drop(combined_agg);
                                    drop(sorted_runs);
                                    abort_and_drain(&mut tasks).await;
                                    drop(sink_reservation);
                                    drop(shared_sort_memory);
                                    drop(finalized_tables);
                                    return Err(err);
                                }
                            }
                        }
                    }
                    WorkerState::Aggregate(local) => {
                        if let Some(combined) = combined_agg.as_mut() {
                            match combined.merge_from(local, &cancel_tx).await {
                                Ok(true) => {}
                                Ok(false) => {
                                    drop(combined_group);
                                    drop(combined_agg);
                                    drop(sorted_runs);
                                    abort_and_drain(&mut tasks).await;
                                    drop(sink_reservation);
                                    drop(shared_sort_memory);
                                    drop(finalized_tables);
                                    return Ok(());
                                }
                                Err(err) => {
                                    drop(combined_group);
                                    drop(combined_agg);
                                    drop(sorted_runs);
                                    abort_and_drain(&mut tasks).await;
                                    drop(sink_reservation);
                                    drop(shared_sort_memory);
                                    drop(finalized_tables);
                                    return Err(err);
                                }
                            }
                        }
                    }
                    WorkerState::SortedRuns(runs) => {
                        sorted_runs.extend(runs);
                        if cancel_tx.is_closed() || output.tx.is_closed() {
                            drop(sorted_runs);
                            abort_and_drain(&mut tasks).await;
                            drop(sink_reservation);
                            drop(shared_sort_memory);
                            drop(finalized_tables);
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Phase 3: run dependent build-side emission child pipelines (RIGHT/FULL
        // OUTER unmatched build rows, RIGHT/LEFT SEMI matched build rows, ANTI
        // unmatched build rows) strictly after all probe workers complete.
        if let Some(tables) = &finalized_tables {
            let mut targets = BuildEmissionTargets {
                combined_count: Some(&mut combined_count),
                combined_group: Some(&mut combined_group),
                combined_agg: Some(&mut combined_agg),
                sorted_runs: Some(&mut sorted_runs),
                combined_build: None,
                sink_reservation: sink_reservation.as_ref(),
                shared_sort_memory: shared_sort_memory.as_ref(),
            };
            match self
                .run_build_emission_pipelines(
                    tables,
                    short_circuit_op,
                    label_prefix,
                    &output,
                    context,
                    metrics,
                    measurement,
                    &mut targets,
                )
                .await
            {
                Ok(true) => {}
                Ok(false) => {
                    drop(combined_group);
                    drop(combined_agg);
                    drop(sorted_runs);
                    drop(sink_reservation);
                    drop(shared_sort_memory);
                    drop(finalized_tables);
                    return Ok(());
                }
                Err(err) => {
                    drop(combined_group);
                    drop(combined_agg);
                    drop(sorted_runs);
                    drop(sink_reservation);
                    drop(shared_sort_memory);
                    drop(finalized_tables);
                    return Err(err);
                }
            }
        }
        drop(sink_reservation);
        drop(shared_sort_memory);
        drop(finalized_tables);

        // A worker's EOF is not necessarily a clean pipeline EOF. Publish its
        // buffered tail only after all workers and any scan producers succeed.
        let tails = std::mem::take(&mut *output.tails.lock());
        let tx = output.tx;
        for (batches, mut worker) in tails {
            for batch in batches {
                if cancel_tx.is_closed() || !worker.send(&tx, batch).await {
                    return Ok(());
                }
            }
        }
        if cancel_tx.is_closed() {
            return Ok(());
        }
        match &self.sink {
            Sink::Result | Sink::Limit { .. } | Sink::HashJoinBuild(_) => {}
            Sink::Count {
                sink,
                result_operators,
                post_sort,
            } => {
                if !tx.is_closed() && !cancel_tx.is_closed() {
                    #[cfg(test)]
                    if let Some(gate) = &sink.finalize_gate {
                        gate.wait().await;
                    }
                    if !tx.is_closed() && !cancel_tx.is_closed() {
                        let finalized = sink.finalize(combined_count)?;
                        MetricBuilder::new(metrics)
                            .counter("count_finalizations", 0)
                            .add(1);
                        if post_sort.is_some() {
                            self.run_post_sort_stage(
                                FinalizedAggOutput::Count(finalized),
                                output.batch_size,
                                context,
                                metrics,
                                tx,
                                measurement,
                            )
                            .await?;
                        } else {
                            Self::run_count_result(
                                finalized,
                                result_operators,
                                tx,
                                measurement.map(Arc::as_ref),
                            )
                            .await?;
                        }
                    }
                }
            }
            Sink::GroupCount {
                sink,
                result_operators,
                post_sort,
            } => {
                if !tx.is_closed() && !cancel_tx.is_closed() {
                    #[cfg(test)]
                    if let Some(gate) = &sink.finalize_gate {
                        gate.wait().await;
                    }
                    if !tx.is_closed()
                        && !cancel_tx.is_closed()
                        && let Some(combined) = combined_group.take()
                    {
                        let finalized = sink.finalize(combined);
                        MetricBuilder::new(metrics)
                            .counter("group_count_finalizations", 0)
                            .add(1);
                        if post_sort.is_some() {
                            self.run_post_sort_stage(
                                FinalizedAggOutput::GroupCount(finalized),
                                output.batch_size,
                                context,
                                metrics,
                                tx,
                                measurement,
                            )
                            .await?;
                        } else {
                            finalized
                                .emit(
                                    result_operators,
                                    output.batch_size,
                                    tx,
                                    measurement.map(Arc::as_ref),
                                )
                                .await?;
                        }
                    }
                }
            }
            Sink::Aggregate {
                sink,
                result_operators,
                post_sort,
            } => {
                if !tx.is_closed() && !cancel_tx.is_closed() {
                    #[cfg(test)]
                    if let Some(gate) = &sink.finalize_gate {
                        gate.wait().await;
                    }
                    if !tx.is_closed()
                        && !cancel_tx.is_closed()
                        && let Some(combined) = combined_agg.take()
                    {
                        let finalized = sink.finalize(combined);
                        MetricBuilder::new(metrics)
                            .counter("aggregate_finalizations", 0)
                            .add(1);
                        if post_sort.is_some() {
                            self.run_post_sort_stage(
                                FinalizedAggOutput::Aggregate(finalized),
                                output.batch_size,
                                context,
                                metrics,
                                tx,
                                measurement,
                            )
                            .await?;
                        } else {
                            finalized
                                .emit(
                                    result_operators,
                                    output.batch_size,
                                    tx,
                                    measurement.map(Arc::as_ref),
                                )
                                .await?;
                        }
                    }
                }
            }
            Sink::TopN {
                sink,
                result_operators,
            } => {
                Self::finalize_direct_topn(
                    sink,
                    result_operators,
                    sorted_runs,
                    SortFinalizeCtx {
                        batch_size: output.batch_size,
                        context,
                        metrics,
                        measurement: measurement.map(Arc::as_ref),
                    },
                    tx,
                )
                .await?;
            }
            Sink::Sort {
                sink,
                result_operators,
            } => {
                Self::finalize_direct_sort(
                    sink,
                    result_operators,
                    sorted_runs,
                    SortFinalizeCtx {
                        batch_size: output.batch_size,
                        context,
                        metrics,
                        measurement: measurement.map(Arc::as_ref),
                    },
                    tx,
                )
                .await?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn start(
        self: Arc<Self>,
        context: Arc<TaskContext>,
        metrics: ExecutionPlanMetricsSet,
        measurement: Option<Arc<RunMetrics>>,
    ) -> PipelineRun {
        self.start_dag(
            Arc::new(Vec::new()),
            Vec::new(),
            None,
            context,
            metrics,
            measurement,
        )
    }

    fn start_dag(
        self: Arc<Self>,
        stages: Arc<Vec<StageSpec>>,
        consumer_counts: Vec<usize>,
        scalar_subqueries: Option<Arc<ScalarSubqueryState>>,
        context: Arc<TaskContext>,
        metrics: ExecutionPlanMetricsSet,
        measurement: Option<Arc<RunMetrics>>,
    ) -> PipelineRun {
        let (tx, rx) = mpsc::channel(self.sink_capacity);
        let worker_measurement = measurement.clone();
        let supervisor = SpawnedTask::spawn(async move {
            let measurement = worker_measurement;
            let _sq_guard = scalar_subqueries
                .as_ref()
                .map(ScalarSubqueryState::acquire_run);
            let mut dag = DagRuntime::new(stages, consumer_counts);

            if let Some(sq_state) = &scalar_subqueries {
                let cont = sq_state
                    .populate(&mut dag, &context, &metrics, measurement.as_ref(), &tx)
                    .await?;
                if !cont {
                    return Ok(());
                }
            }

            let root_deps = collect_pipeline_stage_deps(&self);
            let bound_deps = if root_deps.is_empty() {
                Arc::new(HashMap::new())
            } else {
                let Some(deps) = bind_stage_dependencies(
                    &root_deps,
                    &mut dag,
                    &context,
                    &metrics,
                    measurement.as_ref(),
                    &tx,
                )
                .await?
                else {
                    return Ok(());
                };
                deps
            };
            drop(dag);

            let cancel_tx = tx.clone();
            self.run_pipeline_to_tx(
                None,
                &bound_deps,
                &context,
                &metrics,
                measurement.as_ref(),
                tx,
                cancel_tx,
            )
            .await
        });
        PipelineRun {
            rx,
            supervisor: Some(supervisor),
            measurement,
        }
    }
}

struct SortFinalizeCtx<'a> {
    batch_size: usize,
    context: &'a Arc<TaskContext>,
    metrics: &'a ExecutionPlanMetricsSet,
    measurement: Option<&'a RunMetrics>,
}

enum FinalizedAggOutput {
    Count(FinalizedCount),
    GroupCount(FinalizedGroupCounts),
    Aggregate(FinalizedAggregate),
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
    stages: Arc<Vec<StageSpec>>,
    consumer_counts: Vec<usize>,
    scalar_subqueries: Option<Arc<ScalarSubqueryState>>,
    pipeline: Arc<Pipeline>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl MorselExec {
    #[cfg(test)]
    fn new(pipeline: Pipeline, schema: SchemaRef) -> Self {
        Self::try_new(pipeline, schema).expect("valid test MorselExec")
    }

    #[cfg_attr(not(test), expect(dead_code))]
    fn try_new(pipeline: Pipeline, schema: SchemaRef) -> Result<Self> {
        Self::try_new_dag(Vec::new(), None, pipeline, schema, None)
    }

    fn try_new_dag(
        stages: Vec<StageSpec>,
        scalar_subqueries: Option<Arc<ScalarSubqueryState>>,
        pipeline: Pipeline,
        schema: SchemaRef,
        root_ordering: Option<LexOrdering>,
    ) -> Result<Self> {
        let consumer_counts =
            validate_dag(&stages, scalar_subqueries.as_deref(), &pipeline)?;
        let has_blocking_prefix = !stages.is_empty() || scalar_subqueries.is_some();
        let emission = if has_blocking_prefix {
            EmissionType::Final
        } else {
            match pipeline.sink {
                Sink::Result | Sink::Limit { .. } => EmissionType::Incremental,
                Sink::Count { .. }
                | Sink::GroupCount { .. }
                | Sink::Aggregate { .. }
                | Sink::TopN { .. }
                | Sink::Sort { .. }
                | Sink::HashJoinBuild(_) => EmissionType::Final,
            }
        };
        let eq_properties = if let Some((sort_schema, ordering, result_ops)) =
            pipeline.sink.root_sort_ordering()
        {
            let mut eq = EquivalenceProperties::new(Arc::clone(sort_schema));
            eq.add_ordering(ordering.clone());
            for op in result_ops {
                if let Operator::Projection(projector) = op {
                    let mapping =
                        projector.projection().projection_mapping(eq.schema())?;
                    eq = eq.project(&mapping, Arc::clone(projector.output_schema()));
                }
            }
            eq
        } else if let Some(ordering) = root_ordering {
            let mut eq = EquivalenceProperties::new(schema);
            eq.add_ordering(ordering);
            eq
        } else {
            EquivalenceProperties::new(schema)
        };
        let metrics = match &pipeline.source {
            Source::Memory(_) | Source::Stage(_) => ExecutionPlanMetricsSet::new(),
            #[cfg(feature = "parquet")]
            Source::Parquet(source) => source.metrics.clone(),
        };
        Ok(Self {
            stages: Arc::new(stages),
            consumer_counts,
            scalar_subqueries,
            metrics,
            pipeline: Arc::new(pipeline),
            properties: Arc::new(
                PlanProperties::new(
                    eq_properties,
                    Partitioning::UnknownPartitioning(1),
                    emission,
                    Boundedness::Bounded,
                )
                .with_scheduling_type(SchedulingType::Cooperative),
            ),
        })
    }

    fn is_inline_memory_stream(&self) -> bool {
        self.stages.is_empty()
            && self.scalar_subqueries.is_none()
            && self.pipeline.is_inline_memory_stream()
    }
}

fn display_source(source: &Source, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match source {
        Source::Memory(source) => write!(
            f,
            "MemTable[{}, projection={:?}]",
            source.table, source.projection
        ),
        #[cfg(feature = "parquet")]
        Source::Parquet(source) => match source.pushed_predicate() {
            Some(predicate) => {
                write!(f, "NativeParquet[{}, predicate={predicate}]", source.table)
            }
            None => write!(f, "NativeParquet[{}]", source.table),
        },
        Source::Stage(source) => {
            if source.ordered {
                write!(f, "StageSource[stage_{}, ordered]", source.stage_id.0)
            } else {
                write!(f, "StageSource[stage_{}]", source.stage_id.0)
            }
        }
    }
}

fn display_pipeline_body(pipeline: &Pipeline, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    display_build_pipelines(&pipeline.operators, f)?;
    display_source(&pipeline.source, f)?;
    display_operators(&pipeline.operators, f)?;
    match &pipeline.sink {
        Sink::Result => {}
        Sink::Count {
            result_operators,
            post_sort,
            ..
        } => {
            write!(f, " -> CountSink; FinalizedCountSource")?;
            display_operators(result_operators, f)?;
            if let Some(ps) = post_sort {
                ps.display_summary(f)?;
            }
        }
        Sink::GroupCount {
            sink,
            result_operators,
            post_sort,
        } => {
            write!(
                f,
                " -> GroupCountSink[key={}@{}]; FinalizedGroupCountSource",
                sink.key_name, sink.key_index
            )?;
            display_operators(result_operators, f)?;
            if let Some(ps) = post_sort {
                ps.display_summary(f)?;
            }
        }
        Sink::Aggregate {
            sink,
            result_operators,
            post_sort,
        } => {
            write!(f, " -> ")?;
            sink.display_summary(f)?;
            write!(f, "; FinalizedAggregateSource")?;
            display_operators(result_operators, f)?;
            if let Some(ps) = post_sort {
                ps.display_summary(f)?;
            }
        }
        Sink::TopN {
            sink,
            result_operators,
        } => {
            write!(f, " -> ")?;
            sink.display_summary(f)?;
            write!(f, "; FinalizedTopNSource")?;
            display_operators(result_operators, f)?;
        }
        Sink::Sort {
            sink,
            result_operators,
        } => {
            write!(f, " -> ")?;
            sink.display_summary(f)?;
            write!(f, "; FinalizedSortSource")?;
            display_operators(result_operators, f)?;
        }
        Sink::Limit {
            skip,
            fetch,
            result_operators,
        } => {
            write!(f, " -> LimitSink[skip={skip}, fetch={fetch:?}]")?;
            display_operators(result_operators, f)?;
        }
        Sink::HashJoinBuild(sink) => {
            write!(f, " -> ")?;
            sink.display_summary(f)?;
        }
    }
    Ok(())
}

fn display_build_pipelines(
    operators: &[Operator],
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    for operator in operators {
        if let Operator::HashProbe(probe) = operator {
            let bp = &probe.build_pipeline;
            display_build_pipelines(&bp.operators, f)?;
            write!(f, "[Build: ")?;
            display_source(&bp.source, f)?;
            display_operators(&bp.operators, f)?;
            if let Sink::HashJoinBuild(sink) = &bp.sink {
                write!(f, " -> ")?;
                sink.display_summary(f)?;
            }
            write!(f, "]; ")?;
        }
    }
    Ok(())
}

impl DisplayAs for MorselExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pipeline = &self.pipeline;
        write!(f, "MorselExec: workers={}, ", pipeline.workers)?;
        for stage in self.stages.iter() {
            write!(f, "[Stage {}: ", stage.label)?;
            display_pipeline_body(&stage.pipeline, f)?;
            write!(f, "]; ")?;
        }
        display_pipeline_body(pipeline, f)?;
        if self.is_inline_memory_stream() {
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
            Operator::HashProbe(probe) => {
                write!(f, " -> ")?;
                probe.display_summary(f)?;
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
        #[cfg(feature = "parquet")]
        {
            let mut combined = self.metrics.clone_inner();
            let mut build_sets = Vec::new();
            for stage in self.stages.iter() {
                if let Source::Parquet(source) = &stage.pipeline.source {
                    build_sets.push(&source.metrics);
                }
                stage
                    .pipeline
                    .collect_build_parquet_metrics(&mut build_sets);
            }
            self.pipeline.collect_build_parquet_metrics(&mut build_sets);
            for set in build_sets {
                for metric in set.clone_inner().iter() {
                    combined.push(Arc::clone(metric));
                }
            }
            Some(combined)
        }
        #[cfg(not(feature = "parquet"))]
        {
            Some(self.metrics.clone_inner())
        }
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        for stage in self.stages.iter() {
            if stage.pipeline.apply_expressions(f)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        self.pipeline.apply_expressions(f)
    }

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(sq) = &self.scalar_subqueries {
            sq.results.clear();
        }
        Ok(self)
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
        if self.is_inline_memory_stream() {
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
        let stages = Arc::clone(&self.stages);
        let consumer_counts = self.consumer_counts.clone();
        let scalar_subqueries = self.scalar_subqueries.clone();
        let metrics = self.metrics.clone();
        let stream = stream::once(async move {
            let measurement = collector.map(RunMetrics::start);
            stream::try_unfold(
                pipeline.start_dag(
                    stages,
                    consumer_counts,
                    scalar_subqueries,
                    context,
                    metrics,
                    measurement,
                ),
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
