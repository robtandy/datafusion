// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalMaterializedCte`]: the CTEs of `WITH x AS MATERIALIZED (...)`
//! (`MaterializedCteExec` and its `MaterializedCteScanExec`s).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_plan::materialized_cte::{
    MaterializedCteExec, MaterializedCteScanExec,
};
use datafusion_physical_plan::recursive_query::RecursiveQueryExec;
use datafusion_physical_plan::scalar_subquery::ScalarSubqueryExec;
use datafusion_physical_plan::{ExecutionPlan, displayable};
use parking_lot::Mutex;

use crate::bridge::bridged_plans;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkResult, SourceResult, downcast_state,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sources::batches::{batch_overhead, chunk_memory};

/// The bodies of the materialized CTEs whose continuations are being lowered,
/// by the id of their buffer, which their scans read (see
/// [`PhysicalMaterializedCte::try_lower`]).
#[derive(Debug, Default)]
pub(crate) struct MaterializedCtes {
    bodies: RefCell<HashMap<u64, Arc<MaterializedCteBody>>>,
}

/// `plan` (a `ScalarSubqueryExec`) below the materialized CTEs of its input
/// that its subqueries scan, `None` if there are none.
///
/// DataFusion plans the subqueries of a scalar subquery apart from its input,
/// in which their CTEs are: their scans run the body of their CTE on their
/// own, before its `MaterializedCteExec` does (DataFusion runs the subqueries
/// of a scalar subquery before its input). The pipeline engine runs the body
/// of a CTE before the pipelines of its continuation: so that every scan of
/// the CTE reads its body (computed once), the CTE (with the CTEs that its
/// body scans) becomes the root of the scalar subquery, whose input reads its
/// continuation instead.
pub(crate) fn hoist_materialized_ctes(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(exec) = plan.downcast_ref::<ScalarSubqueryExec>() else {
        return Ok(None);
    };
    // The CTEs of the input, from the outermost.
    let mut ctes = vec![];
    exec.input().apply(|node| {
        if node.is::<MaterializedCteExec>() {
            ctes.push(Arc::clone(node));
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let id = |cte: &Arc<dyn ExecutionPlan>| {
        cte.downcast_ref::<MaterializedCteExec>()
            .map(|cte| cte.buffer().id())
    };
    let mut scanned = HashSet::new();
    for link in exec.subqueries() {
        scanned.extend(scanned_ctes(&link.plan)?);
    }
    // And the CTEs that the bodies of the hoisted CTEs scan.
    let mut hoisted = HashSet::new();
    while let Some(next) = scanned.iter().find(|id| !hoisted.contains(*id)).copied() {
        hoisted.insert(next);
        if let Some(cte) = ctes.iter().find(|cte| id(cte) == Some(next)) {
            let body = &cte.children()[0];
            scanned.extend(scanned_ctes(body)?);
        }
    }
    let ctes = ctes
        .into_iter()
        .filter(|cte| id(cte).is_some_and(|id| hoisted.contains(&id)))
        .collect::<Vec<_>>();
    if ctes.is_empty() {
        return Ok(None);
    }
    let without_ctes = Arc::clone(plan)
        .transform_up(|node| {
            match id(&node) {
                Some(id) if hoisted.contains(&id) => {
                    // Its continuation.
                    Ok(Transformed::yes(Arc::clone(node.children()[1])))
                }
                _ => Ok(Transformed::no(node)),
            }
        })?
        .data;
    let mut hoisted_plan = without_ctes;
    for cte in ctes.iter().rev() {
        let Some(exec) = cte.downcast_ref::<MaterializedCteExec>() else {
            continue;
        };
        hoisted_plan = Arc::new(MaterializedCteExec::new(
            Arc::clone(exec.body()),
            hoisted_plan,
            Arc::clone(exec.buffer()),
        ));
    }
    Ok(Some(hoisted_plan))
}

/// The ids of the CTEs that the scans of `plan` read.
fn scanned_ctes(plan: &Arc<dyn ExecutionPlan>) -> Result<HashSet<u64>> {
    let mut ids = HashSet::new();
    plan.apply(|node| {
        if let Some(scan) = node.downcast_ref::<MaterializedCteScanExec>() {
            ids.insert(scan.id());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(ids)
}

/// Whether the recursive term of a recursive query in `plan` scans the
/// materialized CTE `id`.
fn scanned_by_recursive_terms(plan: &Arc<dyn ExecutionPlan>, id: u64) -> Result<bool> {
    let mut scanned = false;
    plan.apply(|node| {
        if let Some(query) = node.downcast_ref::<RecursiveQueryExec>() {
            scanned |= scanned_ctes(query.recursive_term())?.contains(&id);
        }
        Ok(if scanned {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    Ok(scanned)
}

/// Computes the body of a materialized CTE once, and then runs the rest of
/// the query (the continuation), whose scans of the CTE
/// ([`MaterializedCteScan`]s) read the rows of the body, as
/// `MaterializedCteExec` does.
///
/// The body is built into a child meta pipeline, whose sink (a
/// [`MaterializedCteBody`]) collects its rows, and the continuation continues
/// the pipeline of the operator: all pipelines of the continuation (also
/// those of the pipeline breakers in it) wait for the body to be complete. So
/// the body runs once, before any scan reads a row, and every scan reads all
/// of its rows. The operator itself takes part in no pipeline.
///
/// The rows of the body are kept in memory (reserved in the memory pool of
/// the session) until the execution completes; DataFusion spills them once
/// the pool is exhausted, the native operator fails with
/// `ResourcesExhausted`.
#[derive(Debug)]
pub struct PhysicalMaterializedCte {
    /// The continuation, followed by the sink of the body.
    children: Vec<Arc<dyn PhysicalOperator>>,
    description: String,
}

impl PhysicalMaterializedCte {
    /// Lower a `MaterializedCteExec` to a [`PhysicalMaterializedCte`]: its
    /// body with `lower_body` (the body reads all of its input), and its
    /// continuation with `lower_continuation`, while the scans of the CTE are
    /// lowered to scans of the body (with [`MaterializedCteScan::try_lower`]
    /// and `ctes`).
    ///
    /// A scan of the CTE that is left to DataFusion (e.g. below a
    /// user-defined node) runs the body itself: such a CTE is not lowered
    /// (`Ok(None)`), so that DataFusion runs its body once. So is a CTE that
    /// the recursive term of a recursive query in its continuation scans:
    /// every iteration of a native recursive query lowers its recursive term
    /// again, apart from the CTEs around it (see
    /// [`PhysicalRecursiveCte`](crate::operators::recursive_cte::PhysicalRecursiveCte)),
    /// and leaves the scans of these CTEs to DataFusion. Returns `Ok(None)`
    /// for any other node, without lowering anything.
    pub(crate) fn try_lower<B, C>(
        plan: &Arc<dyn ExecutionPlan>,
        ctes: &MaterializedCtes,
        lower_body: B,
        lower_continuation: C,
    ) -> Result<Option<Self>>
    where
        B: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
        C: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(cte) = plan.downcast_ref::<MaterializedCteExec>() else {
            return Ok(None);
        };
        let [_, continuation] = plan.children()[..] else {
            return internal_err!("MaterializedCteExec has a body and a continuation");
        };
        let id = cte.buffer().id();
        if scanned_by_recursive_terms(continuation, id)? {
            return Ok(None);
        }
        let body = Arc::new(MaterializedCteBody::new(lower_body(cte.body())?));
        let outer = ctes.bodies.borrow_mut().insert(id, Arc::clone(&body));
        let continuation = lower_continuation(continuation);
        match outer {
            Some(outer) => ctes.bodies.borrow_mut().insert(id, outer),
            None => ctes.bodies.borrow_mut().remove(&id),
        };
        let continuation = continuation?;
        for bridged in bridged_plans(&continuation) {
            let mut scans = false;
            bridged.apply(|node| {
                scans |= node
                    .downcast_ref::<MaterializedCteScanExec>()
                    .is_some_and(|scan| scan.id() == id);
                Ok(if scans {
                    TreeNodeRecursion::Stop
                } else {
                    TreeNodeRecursion::Continue
                })
            })?;
            if scans {
                return Ok(None);
            }
        }
        let node = displayable(plan.as_ref()).one_line().to_string();
        let node = node.trim_end();
        let description = format!(
            "PhysicalMaterializedCte: {}",
            node.strip_prefix("MaterializedCteExec: ").unwrap_or(node)
        );
        Ok(Some(Self {
            children: vec![continuation, body],
            description,
        }))
    }

    /// The continuation, whose rows the operator produces.
    pub fn continuation(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The sink of the body; its child is the plan of the body.
    pub fn body(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[1]
    }
}

impl PhysicalOperator for PhysicalMaterializedCte {
    fn name(&self) -> &str {
        "PhysicalMaterializedCte"
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn schema(&self) -> SchemaRef {
        self.continuation().schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        let body = Arc::clone(self.body());
        let [plan] = body.children() else {
            return internal_err!("The body of a materialized CTE has one child");
        };
        let dependency = {
            let (mut child, mut pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(&body))?;
            child.build(&mut pipeline, plan)?;
            child.id()
        };
        meta.build_with_dependency(current, self.continuation(), dependency)
    }
}

/// The body of a [`PhysicalMaterializedCte`]: the sink of the pipelines of
/// its plan (its only child), which collects its rows for the scans of the
/// CTE ([`MaterializedCteScan`]), reserving their memory in the memory pool.
#[derive(Debug)]
pub struct MaterializedCteBody {
    children: Vec<Arc<dyn PhysicalOperator>>,
}

impl MaterializedCteBody {
    fn new(plan: Arc<dyn PhysicalOperator>) -> Self {
        Self {
            children: vec![plan],
        }
    }
}

/// The global sink state of a [`MaterializedCteBody`]: the rows of the body.
struct BodyRows {
    batches: Mutex<Vec<RecordBatch>>,
    reservation: MemoryReservation,
}

impl GlobalSinkState for BodyRows {}

impl PhysicalOperator for MaterializedCteBody {
    fn name(&self) -> &str {
        "MaterializedCteBody"
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

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let reservation = MemoryConsumer::new("MaterializedCteBody")
            .register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(BodyRows {
            batches: Mutex::new(vec![]),
            reservation,
        }))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        _lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<BodyRows>(gstate.as_any(), self.name())?;
        // At most the memory of the chunk, before it is compacted.
        let estimate = chunk_memory(input);
        state.reservation.try_grow(estimate)?;
        let batch = input.compact()?;
        // The arrays of the batch, and the batch in the list of batches.
        let size = batch.get_array_memory_size() + batch_overhead(&batch);
        match size.checked_sub(estimate) {
            Some(more) => state.reservation.try_grow(more)?,
            None => state.reservation.shrink(estimate - size),
        }
        state.batches.lock().push(batch);
        Ok(SinkResult::NeedMoreInput)
    }

    fn build_pipelines(
        &self,
        _current: &mut PipelineBuilder,
        _meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        internal_err!(
            "The body of a materialized CTE is built by PhysicalMaterializedCte"
        )
    }
}

/// A scan of a materialized CTE (a `MaterializedCteScanExec`): it reads all
/// rows of its body, which the [`PhysicalMaterializedCte`] computed before
/// the pipeline of the scan runs, on all threads of the pipeline (which claim
/// chunks of at most [`DEFAULT_VECTOR_SIZE`] rows of them).
#[derive(Debug)]
pub struct MaterializedCteScan {
    body: Arc<MaterializedCteBody>,
    schema: SchemaRef,
}

impl MaterializedCteScan {
    /// Lower `plan` to a scan of the body of its CTE if it is a
    /// `MaterializedCteScanExec` of a CTE of `ctes` (whose continuation is
    /// being lowered).
    pub(crate) fn try_lower(
        plan: &Arc<dyn ExecutionPlan>,
        ctes: &MaterializedCtes,
    ) -> Option<Self> {
        let scan = plan.downcast_ref::<MaterializedCteScanExec>()?;
        let body = Arc::clone(ctes.bodies.borrow().get(&scan.id())?);
        Some(Self {
            body,
            schema: plan.schema(),
        })
    }
}

/// The global source state of a [`MaterializedCteScan`]: the rows of the
/// body, and the next chunk of them to claim.
struct ScanRows {
    batches: Vec<RecordBatch>,
    /// For every chunk: its batch and the offset of its first row.
    chunks: Vec<(usize, usize)>,
    next: AtomicUsize,
    /// The list of the batches and the chunks (the body holds the arrays).
    _reservation: MemoryReservation,
}

impl GlobalSourceState for ScanRows {
    fn max_threads(&self) -> usize {
        self.chunks.len().max(1)
    }
}

impl PhysicalOperator for MaterializedCteScan {
    fn name(&self) -> &str {
        "MaterializedCteScan"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let body = self.body.as_ref() as &dyn PhysicalOperator;
        let Some(state) = ctx.global_sink_state(body) else {
            return internal_err!("A materialized CTE is scanned before its body ran");
        };
        let rows = downcast_state::<BodyRows>(state.as_any(), self.name())?;
        let batches = rows.batches.lock();
        // The list of the batches and the chunks of the scan, before they
        // are allocated.
        let num_chunks = batches
            .iter()
            .map(|batch| batch.num_rows().div_ceil(DEFAULT_VECTOR_SIZE))
            .sum::<usize>();
        let reservation = MemoryConsumer::new("MaterializedCteScan")
            .register(ctx.task_ctx.memory_pool());
        reservation.try_grow(
            batches.iter().map(batch_overhead).sum::<usize>()
                + num_chunks * size_of::<(usize, usize)>(),
        )?;
        let batches = batches.clone();
        let chunks = batches
            .iter()
            .enumerate()
            .flat_map(|(index, batch)| {
                (0..batch.num_rows())
                    .step_by(DEFAULT_VECTOR_SIZE)
                    .map(move |offset| (index, offset))
            })
            .collect();
        Ok(Arc::new(ScanRows {
            batches,
            chunks,
            next: AtomicUsize::new(0),
            _reservation: reservation,
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<ScanRows>(gstate.as_any(), self.name())?;
        let next = state.next.fetch_add(1, Ordering::Relaxed);
        let Some(&(index, offset)) = state.chunks.get(next) else {
            return Ok(SourceResult::Finished);
        };
        let batch = &state.batches[index];
        let len = (batch.num_rows() - offset).min(DEFAULT_VECTOR_SIZE);
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(
            batch.slice(offset, len),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::prelude::{SessionConfig, SessionContext};

    use crate::operators::lowering_tests::{
        assert_exhausted, assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit, task_context_with_memory_limit,
    };
    use crate::test_utils::{register_test_tables, test_session_config};

    /// The configuration of the tests, with the dialect that parses
    /// `MATERIALIZED`.
    fn config() -> SessionConfig {
        let mut config = test_session_config();
        config.options_mut().sql_parser.dialect =
            datafusion_common::config::Dialect::PostgreSQL;
        config
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn materialized_ctes_are_native() {
        for sql in [
            "WITH c AS MATERIALIZED (SELECT a, b FROM t1 WHERE a > 0) \
             SELECT a, COUNT(*) AS n FROM c GROUP BY a",
            "WITH c AS MATERIALIZED (SELECT k, w FROM t2) \
             SELECT c1.k, c2.w FROM c c1 JOIN c c2 ON c1.k = c2.k",
            "WITH c AS MATERIALIZED (SELECT id, d FROM t1) \
             SELECT id FROM c WHERE d = 'alpha' UNION ALL SELECT id + 10000 FROM c",
            "WITH c AS MATERIALIZED (SELECT a, SUM(b) AS s FROM t1 GROUP BY a) \
             SELECT a, s FROM c WHERE s > (SELECT AVG(s) FROM c) ORDER BY a",
            // Nested CTEs, and a CTE of no rows.
            "WITH c AS MATERIALIZED (SELECT id, a FROM t1), \
             d AS MATERIALIZED (SELECT a, COUNT(*) AS n FROM c GROUP BY a) \
             SELECT c.id, d.n FROM c JOIN d ON c.a = d.a",
            "WITH c AS MATERIALIZED (SELECT * FROM t1 WHERE a > 1000000) \
             SELECT COUNT(*) FROM c, c AS c2",
            "WITH c AS MATERIALIZED (SELECT id FROM t1 ORDER BY id DESC LIMIT 3) \
             SELECT id FROM c ORDER BY id",
        ] {
            for target_partitions in [1, 4] {
                let config = config().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, async |ctx: &SessionContext| {
                    register_test_tables(ctx)
                })
                .await;
                assert!(count::<PhysicalMaterializedCte>(&root) > 0, "{sql}");
                assert!(count::<MaterializedCteScan>(&root) > 0, "{sql}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ctes_whose_scans_datafusion_runs_are_left_to_datafusion() -> Result<()> {
        // A scan of the CTE that DataFusion runs (here below a sort-merge
        // join, until it is native) runs the body itself: the CTE is then
        // left to DataFusion, which runs the body once, for all scans.
        let mut config = config();
        config.options_mut().optimizer.prefer_hash_join = false;
        let sql = "WITH c AS MATERIALIZED (SELECT k, w FROM t2) \
                   SELECT c1.k, c2.w FROM c c1 JOIN c c2 ON c1.k = c2.k";
        let (expected, actual) =
            crate::test_utils::run_engines(sql, config, async |ctx: &SessionContext| {
                register_test_tables(ctx)
            })
            .await?;
        if let Err(message) = crate::test_utils::compare_outputs(&expected, &actual) {
            panic!("{message}");
        }
        let root = actual
            .plan
            .downcast_ref::<crate::bridge::PipelineRootExec>()
            .unwrap()
            .operator();
        let bridged_scans = bridged_plans(root).iter().any(|plan| {
            let mut scans = false;
            plan.apply(|node| {
                scans |= node.is::<MaterializedCteScanExec>();
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
            scans
        });
        assert!(!bridged_scans || count::<PhysicalMaterializedCte>(root) == 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_body_runs_once() -> Result<()> {
        // A volatile body is evaluated once: both scans see the same value.
        let ctx = crate::test_utils::pipeline_context(config());
        let sql = "WITH r AS MATERIALIZED (SELECT random() AS x) \
                   SELECT COUNT(DISTINCT x) AS n FROM \
                   (SELECT x FROM r UNION ALL SELECT x FROM r)";
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalMaterializedCte>(&root), 1);
        assert_eq!(count::<MaterializedCteScan>(&root), 2);
        for _ in 0..3 {
            let batches = ctx.sql(sql).await?.collect().await?;
            let n = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap()
                .value(0);
            assert_eq!(n, 1);
        }
        Ok(())
    }

    #[test]
    fn the_batches_of_the_body_reserve_memory_also_without_columns() -> Result<()> {
        // The batches of a body without columns take memory of their own.
        let schema = Arc::new(arrow::datatypes::Schema::empty());
        let source = crate::sources::EmptyScanSource::new(Arc::clone(&schema));
        let body = MaterializedCteBody::new(Arc::new(source));
        let ctx = ExecutionContext::new(task_context_with_memory_limit(1), 0, 1);
        let state = body.get_global_sink_state(&ctx)?;
        let mut local = body.get_local_sink_state(&ctx, state.as_ref())?;
        let options = arrow::array::RecordBatchOptions::new().with_row_count(Some(10));
        let batch = RecordBatch::try_new_with_options(schema, vec![], &options)?;
        let chunk = DataChunk::from_record_batch(batch);
        assert_exhausted(body.sink(&ctx, &chunk, state.as_ref(), local.as_mut()));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_of_the_body_reserve_memory() -> Result<()> {
        let ctx = pipeline_context_with_memory_limit(10_000);
        let state =
            datafusion::execution::SessionStateBuilder::new_from_existing(ctx.state())
                .with_config(config())
                .build();
        let ctx = SessionContext::new_with_state(state);
        register_test_tables(&ctx)?;
        let sql = "WITH c AS MATERIALIZED (SELECT * FROM t1) \
                   SELECT COUNT(*) FROM c JOIN c AS c2 ON c.id = c2.id";
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalMaterializedCte>(&root), 1);
        assert_resources_exhausted(&ctx, sql).await;
        Ok(())
    }
}
