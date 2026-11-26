// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Multi-stage pipeline DAG representation, validation, and runtime scheduler
//! for the experimental morsel engine.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{ArrayData, RecordBatch};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, exec_err, internal_err,
};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_expr::physical_planning_context::{ScalarSubqueryResults, SubqueryIndex};
use datafusion_expr::{Expr, LogicalPlan, Subquery};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use indexmap::IndexSet;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::join_filter::DynamicJoinFilterData;
use super::metrics::RunMetrics;
use super::{BoxFuture, Operator, Pipeline, SideEstimate, Source, abort_and_drain};

/// Identifier of an upstream materialized stage in a morsel DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct StageId(pub(super) usize);

/// Source reading finalized, pool-accounted batches from an upstream stage.
#[derive(Debug)]
pub(super) struct StageSource {
    pub(super) stage_id: StageId,
    pub(super) schema: SchemaRef,
    pub(super) ordered: bool,
}

/// Immutable specification of one intermediate stage in a morsel DAG.
#[derive(Debug)]
pub(super) struct StageSpec {
    pub(super) id: StageId,
    pub(super) label: String,
    pub(super) pipeline: Arc<Pipeline>,
    pub(super) output_schema: SchemaRef,
    pub(super) estimate: SideEstimate,
    /// Deduplicated direct upstream stages referenced by `pipeline` (including
    /// its nested hash join build pipelines).
    pub(super) dependencies: Vec<StageId>,
}

/// Binding between an uncorrelated scalar subquery stage and its result slot.
#[derive(Debug)]
pub(super) struct ScalarSubquerySpec {
    pub(super) stage_id: StageId,
    pub(super) index: SubqueryIndex,
    pub(super) data_type: DataType,
}

/// Shared execution coordinator for uncorrelated scalar subqueries owned by a
/// `MorselExec`. Clears results once all concurrent runs finish so repeated
/// executions always compute fresh values without racing.
#[derive(Debug)]
pub(super) struct ScalarSubqueryState {
    pub(super) specs: Vec<ScalarSubquerySpec>,
    pub(super) results: ScalarSubqueryResults,
    active_runs: Mutex<usize>,
    init_lock: Arc<tokio::sync::Mutex<()>>,
}

impl ScalarSubqueryState {
    pub(super) fn new(
        specs: Vec<ScalarSubquerySpec>,
        results: ScalarSubqueryResults,
    ) -> Self {
        Self {
            specs,
            results,
            active_runs: Mutex::new(0),
            init_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub(super) fn acquire_run(self: &Arc<Self>) -> ScalarSubqueryRunGuard {
        let mut active = self.active_runs.lock();
        *active += 1;
        ScalarSubqueryRunGuard {
            state: Arc::clone(self),
        }
    }

    pub(super) async fn populate(
        &self,
        dag: &mut DagRuntime,
        context: &Arc<TaskContext>,
        plan_metrics: &ExecutionPlanMetricsSet,
        measurement: Option<&Arc<RunMetrics>>,
        cancel_tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        let _guard = self.init_lock.lock().await;
        for spec in &self.specs {
            if cancel_tx.is_closed() {
                return Ok(false);
            }
            if self.results.get(spec.index).is_some() {
                dag.release_consumer(spec.stage_id);
                continue;
            }
            let empty_filters = HashMap::new();
            let Some(stage) = ensure_stage(
                spec.stage_id,
                dag,
                &empty_filters,
                context,
                plan_metrics,
                measurement,
                cancel_tx,
            )
            .await?
            else {
                return Ok(false);
            };
            let value = extract_scalar_value(&stage, &spec.data_type)?;
            drop(stage);
            dag.release_consumer(spec.stage_id);
            self.results.set(spec.index, value)?;
        }
        Ok(true)
    }
}

pub(super) struct ScalarSubqueryRunGuard {
    state: Arc<ScalarSubqueryState>,
}

impl Drop for ScalarSubqueryRunGuard {
    fn drop(&mut self) {
        let mut active = self.state.active_runs.lock();
        *active = active.saturating_sub(1);
        if *active == 0 {
            self.state.results.clear();
        }
    }
}

fn extract_scalar_value(
    stage: &MaterializedStage,
    data_type: &DataType,
) -> Result<ScalarValue> {
    let mut value: Option<ScalarValue> = None;
    for batch in &stage.batches {
        if batch.num_rows() == 0 {
            continue;
        }
        if value.is_some() || batch.num_rows() > 1 {
            return exec_err!("Scalar subquery returned more than one row");
        }
        value = Some(ScalarValue::try_from_array(batch.column(0), 0)?);
    }
    match value {
        Some(v) => Ok(v),
        None => ScalarValue::try_from(data_type),
    }
}

/// Collect all pre-existing `MemTable` source buffer addresses across `stages`
/// once at plan construction time. Returns `None` when `stages` is empty or
/// contains no pre-existing `MemTable` buffers.
pub(super) fn collect_stage_seed_buffers(
    stages: &[StageSpec],
) -> Option<Arc<HashSet<usize>>> {
    if stages.is_empty() {
        return None;
    }
    let mut seeds = HashSet::new();
    for stage in stages {
        collect_memory_source_buffers(&stage.pipeline, &mut seeds);
    }
    (!seeds.is_empty()).then(|| Arc::new(seeds))
}

fn collect_memory_source_buffers(pipeline: &Pipeline, out: &mut HashSet<usize>) {
    if let Source::Memory(mem) = &pipeline.source {
        for batch in &mem.batches {
            for col in batch.columns() {
                collect_array_data_buffers(&col.to_data(), out);
            }
        }
    }
    for op in &pipeline.operators {
        if let Operator::HashProbe(probe) = op {
            collect_memory_source_buffers(&probe.build_pipeline, out);
        }
    }
}

fn collect_array_data_buffers(data: &ArrayData, out: &mut HashSet<usize>) {
    if let Some(nulls) = data.nulls() {
        out.insert(nulls.buffer().data_ptr().addr().get());
    }
    for buf in data.buffers() {
        out.insert(buf.data_ptr().addr().get());
    }
    for child in data.child_data() {
        collect_array_data_buffers(child, out);
    }
}

fn visit_batch_buffers(batch: &RecordBatch, f: &mut impl FnMut(usize, usize)) {
    fn visit_data(data: &ArrayData, f: &mut impl FnMut(usize, usize)) {
        if let Some(nulls) = data.nulls() {
            let buf = nulls.buffer();
            if buf.capacity() > 0 {
                f(buf.data_ptr().addr().get(), buf.capacity());
            }
        }
        for buf in data.buffers() {
            if buf.capacity() > 0 {
                f(buf.data_ptr().addr().get(), buf.capacity());
            }
        }
        for child in data.child_data() {
            visit_data(child, f);
        }
    }
    for col in batch.columns() {
        visit_data(&col.to_data(), f);
    }
}

/// Run-wide buffer-deduplicating memory accounting for materialized intermediate
/// stages. Pre-existing `MemTable` source buffers (computed once at planning
/// time) are skipped without charging the pool, and execution-allocated buffers
/// shared across stages are charged once and released when the last referencing
/// stage drops.
#[derive(Debug)]
pub(super) struct SharedStageMemory {
    seed_buffers: Option<Arc<HashSet<usize>>>,
    inner: Mutex<SharedStageMemoryInner>,
}

#[derive(Debug)]
struct SharedStageMemoryInner {
    /// Active execution-allocated buffers: `addr -> (capacity, refcount)`.
    counted_buffers: HashMap<usize, (usize, usize)>,
    reservation: MemoryReservation,
}

impl SharedStageMemory {
    pub(super) fn new(
        seed_buffers: Option<Arc<HashSet<usize>>>,
        pool: &Arc<dyn MemoryPool>,
    ) -> Arc<Self> {
        let reservation = MemoryConsumer::new("MorselStageOutput").register(pool);
        Arc::new(Self {
            seed_buffers,
            inner: Mutex::new(SharedStageMemoryInner {
                counted_buffers: HashMap::new(),
                reservation,
            }),
        })
    }

    fn is_seeded(&self, addr: usize) -> bool {
        self.seed_buffers
            .as_ref()
            .is_some_and(|seeds| seeds.contains(&addr))
    }

    fn try_add_batch(&self, batch: &RecordBatch) -> Result<()> {
        let mut inner = self.inner.lock();
        let mut delta = 0usize;
        visit_batch_buffers(batch, &mut |addr, cap| {
            if self.is_seeded(addr) {
                return;
            }
            let entry = inner.counted_buffers.entry(addr).or_insert((cap, 0));
            if entry.1 == 0 {
                delta = delta.saturating_add(entry.0);
            }
            entry.1 += 1;
        });
        if delta > 0
            && let Err(err) = inner.reservation.try_grow(delta)
        {
            visit_batch_buffers(batch, &mut |addr, _| {
                if self.is_seeded(addr) {
                    return;
                }
                if let Some((_, count)) = inner.counted_buffers.get_mut(&addr) {
                    if *count <= 1 {
                        inner.counted_buffers.remove(&addr);
                    } else {
                        *count -= 1;
                    }
                }
            });
            return Err(err);
        }
        Ok(())
    }

    fn release_batches(&self, batches: &[RecordBatch]) {
        let mut inner = self.inner.lock();
        let mut freed = 0usize;
        for batch in batches {
            visit_batch_buffers(batch, &mut |addr, _| {
                if self.is_seeded(addr) {
                    return;
                }
                if let Some((cap, count)) = inner.counted_buffers.get_mut(&addr) {
                    if *count <= 1 {
                        let cap = *cap;
                        inner.counted_buffers.remove(&addr);
                        freed = freed.saturating_add(cap);
                    } else {
                        *count -= 1;
                    }
                }
            });
        }
        if freed > 0 {
            inner.reservation.shrink(freed);
        }
    }
}

/// Finalized, pool-accounted output batches of an intermediate DAG stage.
#[derive(Debug)]
pub(super) struct MaterializedStage {
    pub(super) batches: Vec<RecordBatch>,
    memory: Arc<SharedStageMemory>,
}

impl MaterializedStage {
    fn new(memory: Arc<SharedStageMemory>) -> Self {
        Self {
            batches: Vec::new(),
            memory,
        }
    }

    fn try_push(&mut self, batch: RecordBatch) -> Result<()> {
        self.memory.try_add_batch(&batch)?;
        self.batches.push(batch);
        Ok(())
    }
}

impl Drop for MaterializedStage {
    fn drop(&mut self) {
        self.memory.release_batches(&self.batches);
    }
}

/// Mutable per-run DAG execution state owned exclusively by the run supervisor.
pub(super) struct DagRuntime {
    stages: Arc<Vec<StageSpec>>,
    outputs: Vec<Option<Arc<MaterializedStage>>>,
    executed: Vec<bool>,
    remaining_consumers: Vec<usize>,
    memory: Option<Arc<SharedStageMemory>>,
}

impl DagRuntime {
    pub(super) fn new(
        stages: Arc<Vec<StageSpec>>,
        consumer_counts: Vec<usize>,
        seed_buffers: Option<Arc<HashSet<usize>>>,
        pool: &Arc<dyn MemoryPool>,
    ) -> Self {
        let n = stages.len();
        let memory = (n > 0).then(|| SharedStageMemory::new(seed_buffers, pool));
        Self {
            stages,
            outputs: vec![None; n],
            executed: vec![false; n],
            remaining_consumers: consumer_counts,
            memory,
        }
    }

    pub(super) fn release_consumer(&mut self, stage_id: StageId) {
        let rem = &mut self.remaining_consumers[stage_id.0];
        *rem = rem.saturating_sub(1);
        if *rem == 0 {
            self.outputs[stage_id.0] = None;
            if !self.executed[stage_id.0] {
                let mut ref_ids = Vec::new();
                collect_pipeline_stage_ref_ids(
                    &self.stages[stage_id.0].pipeline,
                    &mut ref_ids,
                );
                for dep in ref_ids {
                    self.release_consumer(dep);
                }
            }
        }
    }

    pub(super) fn release_source_ref(&mut self, source: &Source) {
        if let Source::Stage(src) = source {
            self.release_consumer(src.stage_id);
        }
    }

    pub(super) fn release_pipeline_refs(&mut self, pipeline: &Pipeline) {
        let mut ref_ids = Vec::new();
        collect_pipeline_stage_ref_ids(pipeline, &mut ref_ids);
        for dep in ref_ids {
            self.release_consumer(dep);
        }
    }
}

/// Collect all `StageSource` references inside `pipeline` and its nested join
/// build pipelines (preserving duplicates for exact per-reference counting).
pub(super) fn collect_pipeline_stage_refs(
    pipeline: &Pipeline,
    out: &mut Vec<(StageId, SchemaRef)>,
) {
    if let Source::Stage(src) = &pipeline.source {
        out.push((src.stage_id, Arc::clone(&src.schema)));
    }
    for op in &pipeline.operators {
        if let Operator::HashProbe(probe) = op {
            collect_pipeline_stage_refs(&probe.build_pipeline, out);
        }
    }
}

/// Collect all referenced `StageId`s inside `pipeline` and its nested join
/// build pipelines (preserving duplicates for exact per-reference counting).
pub(super) fn collect_pipeline_stage_ref_ids(
    pipeline: &Pipeline,
    out: &mut Vec<StageId>,
) {
    if let Source::Stage(src) = &pipeline.source {
        out.push(src.stage_id);
    }
    for op in &pipeline.operators {
        if let Operator::HashProbe(probe) = op {
            collect_pipeline_stage_ref_ids(&probe.build_pipeline, out);
        }
    }
}

/// Collect deduplicated sorted upstream `StageId`s referenced by `pipeline`.
pub(super) fn collect_pipeline_stage_deps(pipeline: &Pipeline) -> Vec<StageId> {
    let mut deps = Vec::new();
    collect_pipeline_stage_ref_ids(pipeline, &mut deps);
    deps.sort_unstable();
    deps.dedup();
    deps
}

/// Count exact (non-deduplicated) `StageSource` and scalar-subquery references
/// per stage across the entire DAG. A count of `1` proves the stage has a single
/// consumer and cannot create a dependency cycle with its consumer's join builds.
#[cfg(feature = "parquet")]
pub(super) fn count_dag_stage_refs(
    stages: &[StageSpec],
    scalar_subqueries: Option<&ScalarSubqueryState>,
    root_pipeline: &Pipeline,
) -> Vec<usize> {
    let mut counts = vec![0usize; stages.len()];
    let mut ids = Vec::new();
    for stage in stages {
        ids.clear();
        collect_pipeline_stage_ref_ids(&stage.pipeline, &mut ids);
        for id in &ids {
            if id.0 < counts.len() {
                counts[id.0] += 1;
            }
        }
    }
    ids.clear();
    collect_pipeline_stage_ref_ids(root_pipeline, &mut ids);
    for id in &ids {
        if id.0 < counts.len() {
            counts[id.0] += 1;
        }
    }
    if let Some(sq_state) = scalar_subqueries {
        for spec in &sq_state.specs {
            if spec.stage_id.0 < counts.len() {
                counts[spec.stage_id.0] += 1;
            }
        }
    }
    counts
}

/// Validate DAG topology (strict topological indices, zero cycles, schema
/// compatibility) and compute per-stage reference counts.
pub(super) fn validate_dag(
    stages: &[StageSpec],
    scalar_subqueries: Option<&ScalarSubqueryState>,
    root_pipeline: &Pipeline,
) -> Result<Vec<usize>> {
    let mut consumer_counts = vec![0usize; stages.len()];

    for (idx, stage) in stages.iter().enumerate() {
        if stage.id != StageId(idx) {
            return internal_err!(
                "Morsel DAG stage id mismatch: expected {:?}, got {:?}",
                StageId(idx),
                stage.id
            );
        }
        let mut refs = Vec::new();
        collect_pipeline_stage_refs(&stage.pipeline, &mut refs);
        for (dep, schema) in &refs {
            if dep.0 >= idx {
                return internal_err!(
                    "Morsel DAG cycle or out-of-order dependency: stage {idx} depends on stage {}",
                    dep.0
                );
            }
            if schema.as_ref() != stages[dep.0].output_schema.as_ref() {
                return internal_err!(
                    "Morsel DAG schema mismatch between stage {} output and stage {idx} input",
                    dep.0
                );
            }
            consumer_counts[dep.0] += 1;
        }
        for dep in &stage.dependencies {
            if dep.0 >= idx {
                return internal_err!(
                    "Morsel DAG invalid dependency {dep:?} on stage {idx}"
                );
            }
        }
    }

    let mut root_refs = Vec::new();
    collect_pipeline_stage_refs(root_pipeline, &mut root_refs);
    for (dep, schema) in &root_refs {
        if dep.0 >= stages.len() {
            return internal_err!(
                "Morsel DAG root references out-of-bounds stage {}",
                dep.0
            );
        }
        if schema.as_ref() != stages[dep.0].output_schema.as_ref() {
            return internal_err!(
                "Morsel DAG schema mismatch between stage {} output and root input",
                dep.0
            );
        }
        consumer_counts[dep.0] += 1;
    }

    if let Some(sq_state) = scalar_subqueries {
        for spec in &sq_state.specs {
            if spec.stage_id.0 >= stages.len() {
                return internal_err!(
                    "Morsel DAG scalar subquery references out-of-bounds stage {}",
                    spec.stage_id.0
                );
            }
            if stages[spec.stage_id.0].output_schema.fields().len() != 1 {
                return internal_err!(
                    "Morsel DAG scalar subquery stage {} must output 1 column, got {}",
                    spec.stage_id.0,
                    stages[spec.stage_id.0].output_schema.fields().len()
                );
            }
            consumer_counts[spec.stage_id.0] += 1;
        }
    }

    Ok(consumer_counts)
}

/// Collect all unique uncorrelated `Expr::ScalarSubquery` subqueries in
/// post-order (nested subqueries before their enclosing subqueries).
#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
pub(super) fn collect_scalar_subqueries_post_order(plan: &LogicalPlan) -> Vec<Subquery> {
    let mut seen: IndexSet<Subquery> = IndexSet::new();
    let mut ordered = Vec::new();
    collect_scalar_subqueries_rec(plan, &mut seen, &mut ordered);
    ordered
}

#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
fn collect_scalar_subqueries_rec(
    plan: &LogicalPlan,
    seen: &mut IndexSet<Subquery>,
    ordered: &mut Vec<Subquery>,
) {
    plan.apply(|node| {
        for expr in node.expressions() {
            expr.apply(|e| {
                if let Expr::ScalarSubquery(sq) = e
                    && sq.outer_ref_columns.is_empty()
                    && !seen.contains(sq)
                {
                    collect_scalar_subqueries_rec(&sq.subquery, seen, ordered);
                    if seen.insert(sq.clone()) {
                        ordered.push(sq.clone());
                    }
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .expect("infallible");
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .expect("infallible");
}

/// Ensure `stage_id` has been executed and materialized into `dag.outputs`,
/// propagating any already-published dynamic join filters from the consumer
/// and tracking materialized batch bytes in the run's `MemoryPool`.
pub(super) fn ensure_stage<'a>(
    stage_id: StageId,
    dag: &'a mut DagRuntime,
    published_filters: &'a HashMap<usize, Arc<DynamicJoinFilterData>>,
    context: &'a Arc<TaskContext>,
    plan_metrics: &'a ExecutionPlanMetricsSet,
    measurement: Option<&'a Arc<RunMetrics>>,
    cancel_tx: &'a mpsc::Sender<RecordBatch>,
) -> BoxFuture<'a, Result<Option<Arc<MaterializedStage>>>> {
    Box::pin(async move {
        if cancel_tx.is_closed() {
            return Ok(None);
        }
        if let Some(existing) = &dag.outputs[stage_id.0] {
            return Ok(Some(Arc::clone(existing)));
        }

        dag.executed[stage_id.0] = true;
        let stage_pipeline = Arc::clone(&dag.stages[stage_id.0].pipeline);
        let stage_label = dag.stages[stage_id.0].label.clone();

        // Phase 1: run any join build pipelines and ensure upstream stage
        // dependencies before spawning the streaming producer task.
        let mut stage_filters = published_filters.clone();
        let Some(prepared) = stage_pipeline
            .prepare_execution(
                Some(&stage_label),
                dag,
                &mut stage_filters,
                context,
                plan_metrics,
                measurement,
                cancel_tx,
            )
            .await?
        else {
            return Ok(None);
        };

        let channel_cap = stage_pipeline.sink_capacity.max(4);
        let (stage_tx, mut stage_rx) = mpsc::channel(channel_cap);

        let mut stage_tasks: JoinSet<Result<()>> = JoinSet::new();
        {
            let ctx = Arc::clone(context);
            let metrics = plan_metrics.clone();
            let meas = measurement.cloned();
            let cancel = cancel_tx.clone();
            stage_tasks.spawn(async move {
                stage_pipeline
                    .run_prepared_to_tx(
                        Some(&stage_label),
                        prepared,
                        &ctx,
                        &metrics,
                        meas.as_ref(),
                        stage_tx,
                        cancel,
                    )
                    .await
            });
        }

        let memory = Arc::clone(
            dag.memory
                .as_ref()
                .expect("SharedStageMemory present when stages are non-empty"),
        );
        let mut stage = MaterializedStage::new(memory);
        let mut task_done = false;

        loop {
            if !task_done {
                tokio::select! {
                    biased;
                    _ = cancel_tx.closed() => {
                        drop(stage_rx);
                        abort_and_drain(&mut stage_tasks).await;
                        drop(stage);
                        return Ok(None);
                    }
                    result = stage_tasks.join_next() => {
                        task_done = true;
                        match result {
                            Some(Ok(Ok(()))) | None => {}
                            Some(Ok(Err(err))) => {
                                drop(stage_rx);
                                drop(stage);
                                return Err(err);
                            }
                            Some(Err(err)) if err.is_panic() => {
                                drop(stage_rx);
                                drop(stage);
                                std::panic::resume_unwind(err.into_panic())
                            }
                            Some(Err(err)) => {
                                drop(stage_rx);
                                drop(stage);
                                return Err(exec_datafusion_err!(
                                    "Morsel stage task failed: {err}"
                                ));
                            }
                        }
                    }
                    maybe_batch = stage_rx.recv() => {
                        let Some(batch) = maybe_batch else {
                            break;
                        };
                        if batch.num_rows() == 0 {
                            continue;
                        }
                        if let Err(err) = stage.try_push(batch) {
                            drop(stage_rx);
                            abort_and_drain(&mut stage_tasks).await;
                            drop(stage);
                            return Err(err);
                        }
                    }
                }
            } else {
                match stage_rx.recv().await {
                    Some(batch) => {
                        if batch.num_rows() == 0 {
                            continue;
                        }
                        if cancel_tx.is_closed() {
                            drop(stage_rx);
                            drop(stage);
                            return Ok(None);
                        }
                        if let Err(err) = stage.try_push(batch) {
                            drop(stage_rx);
                            drop(stage);
                            return Err(err);
                        }
                    }
                    None => break,
                }
            }
        }

        if !task_done {
            while let Some(result) = stage_tasks.join_next().await {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        drop(stage);
                        return Err(err);
                    }
                    Err(err) if err.is_panic() => {
                        drop(stage);
                        std::panic::resume_unwind(err.into_panic())
                    }
                    Err(err) => {
                        drop(stage);
                        return Err(exec_datafusion_err!(
                            "Morsel stage task failed: {err}"
                        ));
                    }
                }
            }
        }

        if cancel_tx.is_closed() {
            drop(stage);
            return Ok(None);
        }

        let artifact = Arc::new(stage);
        dag.outputs[stage_id.0] = Some(Arc::clone(&artifact));
        Ok(Some(artifact))
    })
}
