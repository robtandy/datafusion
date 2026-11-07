// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Interoperability between DataFusion [`ExecutionPlan`]s and the pipeline
//! engine.
//!
//! * [`BridgeOperator`] runs any [`ExecutionPlan`] as a parallel source of
//!   the pipeline engine. The planner uses it as a fallback for every node it
//!   does not lower to a native operator.
//! * [`PipelineRootExec`] is an [`ExecutionPlan`] that runs a
//!   [`PhysicalOperator`] tree with the [`Executor`] and streams its output,
//!   so that the pipeline engine plugs into DataFusion's `SessionContext`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, internal_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PhysicalExpr, PlanProperties, ReplaceChildrenOptions, displayable,
};
use futures::StreamExt;
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::executor::{ExecutionMetrics, Executor};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state, downcast_state_mut,
};

/// A source that executes a DataFusion [`ExecutionPlan`] and emits its
/// output as [`DataChunk`]s of at most [`DEFAULT_VECTOR_SIZE`] rows.
///
/// All output partitions of the plan are distributed over the threads of the
/// pipeline through a shared partition counter: each thread claims the next
/// unclaimed partition, executes it and drains its stream, slicing every
/// batch (without copying) into chunks, before it claims the next
/// partition.
#[derive(Debug)]
pub struct BridgeOperator {
    plan: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
}

impl BridgeOperator {
    /// Create a bridge that runs `plan`.
    pub fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        let schema = plan.schema();
        Self { plan, schema }
    }

    /// The bridged plan.
    pub fn plan(&self) -> &Arc<dyn ExecutionPlan> {
        &self.plan
    }
}

/// Global source state of a [`BridgeOperator`]: the next partition to claim.
#[derive(Debug)]
struct BridgeGlobalState {
    next_partition: AtomicUsize,
    num_partitions: usize,
}

impl GlobalSourceState for BridgeGlobalState {
    fn max_threads(&self) -> usize {
        self.num_partitions
    }
}

/// Local source state of a [`BridgeOperator`]: the stream of the partition
/// the thread is draining and the remainder of the current batch.
#[derive(Default)]
struct BridgeLocalState {
    stream: Option<SendableRecordBatchStream>,
    batch: Option<RecordBatch>,
    offset: usize,
}

impl LocalSourceState for BridgeLocalState {}

impl PhysicalOperator for BridgeOperator {
    fn name(&self) -> &str {
        "BridgeOperator"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        format!(
            "BridgeOperator: plan={}",
            displayable(self.plan.as_ref())
                .one_line()
                .to_string()
                .trim_end()
        )
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(BridgeGlobalState {
            next_partition: AtomicUsize::new(0),
            num_partitions: self.plan.output_partitioning().partition_count(),
        }))
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(BridgeLocalState::default()))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let gstate = downcast_state::<BridgeGlobalState>(gstate.as_any(), self.name())?;
        let lstate =
            downcast_state_mut::<BridgeLocalState>(lstate.as_any_mut(), self.name())?;
        loop {
            // Emit the next slice of the current batch.
            if let Some(batch) = &lstate.batch {
                let remaining = batch.num_rows() - lstate.offset;
                if remaining > 0 {
                    let len = remaining.min(DEFAULT_VECTOR_SIZE);
                    let chunk = if lstate.offset == 0 && len == batch.num_rows() {
                        batch.clone()
                    } else {
                        batch.slice(lstate.offset, len)
                    };
                    lstate.offset += len;
                    return Ok(SourceResult::Chunk(DataChunk::from_record_batch(chunk)));
                }
                lstate.batch = None;
            }

            // Claim the next partition once the current one is drained.
            if lstate.stream.is_none() {
                let partition = gstate.next_partition.fetch_add(1, Ordering::Relaxed);
                if partition >= gstate.num_partitions {
                    return Ok(SourceResult::Finished);
                }
                lstate.stream =
                    Some(self.plan.execute(partition, Arc::clone(&ctx.task_ctx))?);
            }

            let stream = lstate.stream.as_mut().expect("stream was just set");
            let mut cx = Context::from_waker(ctx.waker());
            match stream.poll_next_unpin(&mut cx) {
                Poll::Ready(Some(batch)) => {
                    let batch = batch?;
                    if batch.num_rows() == 0 {
                        // Return to the executor (which skips empty chunks)
                        // so that the call counts against its budget.
                        return Ok(SourceResult::Chunk(DataChunk::from_record_batch(
                            batch,
                        )));
                    }
                    lstate.batch = Some(batch);
                    lstate.offset = 0;
                }
                Poll::Ready(None) => lstate.stream = None,
                Poll::Pending => {
                    return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
                }
            }
        }
    }
}

/// Returns the plans of all [`BridgeOperator`]s in the operator tree below
/// (and including) `operator`, e.g. to check how much of a query runs on
/// native pipeline operators.
pub fn bridged_plans(
    operator: &Arc<dyn PhysicalOperator>,
) -> Vec<Arc<dyn ExecutionPlan>> {
    let mut plans = vec![];
    collect_bridged_plans(operator, &mut plans);
    plans
}

fn collect_bridged_plans(
    operator: &Arc<dyn PhysicalOperator>,
    plans: &mut Vec<Arc<dyn ExecutionPlan>>,
) {
    if let Some(bridge) = operator.downcast_ref::<BridgeOperator>() {
        plans.push(Arc::clone(bridge.plan()));
    }
    for child in operator.children() {
        collect_bridged_plans(child, plans);
    }
}

/// An [`ExecutionPlan`] that executes a [`PhysicalOperator`] tree with the
/// pipeline [`Executor`].
///
/// It has a single output partition: all threads of the root pipeline feed
/// the same output stream. Its schema is the schema of the DataFusion plan
/// the operator tree was lowered from (`input`), which must equal the schema
/// of the operator tree.
///
/// From DataFusion's point of view this is a leaf node: the lowered plan is
/// executed by the pipeline engine, not by DataFusion. Its `DisplayAs`
/// implementation and metrics live in [`crate::explain`].
///
/// Because the DataFusion plans executed by [`BridgeOperator`]s are hidden
/// from DataFusion's plan traversals, [`ExecutionPlan::reset_state`] resets
/// the whole `input` plan (with [`reset_plan_states`]) and lowers it again
/// with [`lower_execution_plan`](crate::planner::lower_execution_plan), so
/// that the plan can be executed again like a regular DataFusion plan.
#[derive(Debug)]
pub struct PipelineRootExec {
    root: Arc<dyn PhysicalOperator>,
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    execution_metrics: Mutex<Option<Arc<ExecutionMetrics>>>,
}

impl PipelineRootExec {
    /// Create a plan that executes `root`, the lowered form of `input`.
    pub fn try_new(
        root: Arc<dyn PhysicalOperator>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        let schema = input.schema();
        if root.schema() != schema {
            return internal_err!(
                "The schema of the lowered operator {} does not match the schema of the plan: {:?} vs {:?}",
                root.name(),
                root.schema(),
                schema
            );
        }
        // With a single input partition the operator tree is fed by a single
        // thread (the bridge cannot run more threads than partitions), which
        // preserves the order of the input.
        let eq_properties = if input.output_partitioning().partition_count() == 1 {
            input.equivalence_properties().clone()
        } else {
            EquivalenceProperties::new(Arc::clone(&schema))
        };
        let properties = PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            root,
            input,
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
            execution_metrics: Mutex::new(None),
        })
    }

    /// The root of the operator tree.
    pub fn operator(&self) -> &Arc<dyn PhysicalOperator> {
        &self.root
    }

    /// The DataFusion plan the operator tree was lowered from.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// The operator metrics of the most recent execution, if any.
    pub fn execution_metrics(&self) -> Option<Arc<ExecutionMetrics>> {
        self.execution_metrics.lock().clone()
    }

    pub(crate) fn metrics_set(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }
}

impl ExecutionPlan for PipelineRootExec {
    fn name(&self) -> &str {
        "PipelineRootExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
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
            return internal_err!("PipelineRootExec does not have children");
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

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        // `input` is not a child of this node, so the traversal that calls
        // `reset_state` does not reach it: reset it as a whole here, and
        // lower it again so that the operators refer to the reset plans.
        let input = reset_plan_states(Arc::clone(&self.input))?;
        let plan: Arc<dyn ExecutionPlan> = crate::planner::create_pipeline_plan(input)?;
        Ok(plan)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!(
                "PipelineRootExec has a single partition, got partition {partition}"
            );
        }
        let num_threads = context.session_config().target_partitions();
        let stream =
            Executor::execute_stream(Arc::clone(&self.root), context, num_threads)?;
        *self.execution_metrics.lock() = Some(Arc::clone(stream.metrics()));
        let stream = crate::explain::instrument_output_stream(self, stream);
        Ok(stream)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(crate::explain::root_metrics(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{AsArray, Int32Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_plan::collect;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]))
    }

    fn batch(values: std::ops::Range<i32>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from_iter_values(values))],
        )
        .unwrap()
    }

    /// A plan with three skewed partitions: 5000 rows, 10 rows, nothing.
    fn plan() -> Arc<dyn ExecutionPlan> {
        MemorySourceConfig::try_new_exec(
            &[vec![batch(0..5000)], vec![batch(5000..5010)], vec![]],
            schema(),
            None,
        )
        .unwrap()
    }

    fn sorted_values(batches: &[RecordBatch]) -> Vec<i32> {
        let mut values = batches
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
            .collect::<Vec<_>>();
        values.sort_unstable();
        values
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bridge_runs_all_partitions_in_chunks() -> Result<()> {
        let bridge: Arc<dyn PhysicalOperator> = Arc::new(BridgeOperator::new(plan()));
        for threads in [1, 2, 8] {
            let output = Executor::collect(
                Arc::clone(&bridge),
                Arc::new(TaskContext::default()),
                threads,
            )
            .await?;
            assert_eq!(sorted_values(&output), (0..5010).collect::<Vec<_>>());
            assert!(
                output
                    .iter()
                    .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
            );
            assert!(output.iter().all(|batch| batch.schema() == schema()));
        }
        assert_eq!(bridged_plans(&bridge).len(), 1);
        assert!(
            bridge
                .description()
                .starts_with("BridgeOperator: plan=DataSourceExec")
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bridge_threads_are_limited_by_partitions() -> Result<()> {
        let bridge = BridgeOperator::new(plan());
        let ctx = ExecutionContext::new(Arc::new(TaskContext::default()), 0, 1);
        assert_eq!(bridge.get_global_source_state(&ctx)?.max_threads(), 3);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipeline_root_exec_streams_the_result() -> Result<()> {
        let input = plan();
        let root = Arc::new(PipelineRootExec::try_new(
            Arc::new(BridgeOperator::new(Arc::clone(&input))),
            Arc::clone(&input),
        )?);
        assert_eq!(root.properties().output_partitioning().partition_count(), 1);
        assert_eq!(root.schema(), input.schema());
        assert!(root.children().is_empty());
        assert!(root.execution_metrics().is_none());
        let err = root
            .execute(1, Arc::new(TaskContext::default()))
            .err()
            .unwrap();
        assert!(err.to_string().contains("single partition"), "{err}");

        let output = collect(
            Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            Arc::new(TaskContext::default()),
        )
        .await?;
        assert_eq!(sorted_values(&output), (0..5010).collect::<Vec<_>>());

        let metrics = root.metrics().unwrap();
        assert_eq!(metrics.output_rows(), Some(5010));
        let execution = root.execution_metrics().unwrap();
        let (bridge, bridge_metrics) = execution.iter().next().unwrap();
        assert_eq!(bridge.name(), "BridgeOperator");
        assert_eq!(bridge_metrics.output_rows(), 5010);
        Ok(())
    }

    #[test]
    fn pipeline_root_exec_keeps_single_partition_ordering() -> Result<()> {
        use datafusion_physical_expr::expressions::col;
        use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};

        let ordering =
            LexOrdering::new(vec![PhysicalSortExpr::new_default(col("a", &schema())?)])
                .unwrap();
        let source = MemorySourceConfig::try_new(&[vec![batch(0..10)]], schema(), None)?
            .try_with_sort_information(vec![ordering.clone()])?;
        let input: Arc<dyn ExecutionPlan> =
            datafusion_datasource::source::DataSourceExec::from_data_source(source);
        let root = PipelineRootExec::try_new(
            Arc::new(BridgeOperator::new(Arc::clone(&input))),
            input,
        )?;
        assert_eq!(root.properties().output_ordering(), Some(&ordering));

        // With several input partitions there is no order.
        let root =
            PipelineRootExec::try_new(Arc::new(BridgeOperator::new(plan())), plan())?;
        assert_eq!(root.properties().output_ordering(), None);
        Ok(())
    }

    #[test]
    fn pipeline_root_exec_rejects_schema_mismatch() {
        let other = MemorySourceConfig::try_new_exec(
            &[vec![]],
            Arc::new(Schema::new(vec![Field::new("b", DataType::Int64, true)])),
            None,
        )
        .unwrap();
        let err = PipelineRootExec::try_new(Arc::new(BridgeOperator::new(plan())), other)
            .unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reset_plan_states_resets_the_bridged_plans() -> Result<()> {
        use datafusion_physical_plan::execution_plan::reset_plan_states;

        use crate::test_utils::{
            canonical_ipc_bytes, datafusion_context, pipeline_context,
            register_test_tables, test_session_config,
        };

        let sql = "SELECT a, COUNT(*) FROM t1 GROUP BY a";
        let expected = {
            let ctx = datafusion_context(test_session_config());
            register_test_tables(&ctx)?;
            ctx.sql(sql).await?.collect().await?
        };
        let expected_bytes =
            canonical_ipc_bytes(&expected[0].schema(), &expected, false)?;

        let ctx = pipeline_context(test_session_config());
        register_test_tables(&ctx)?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let mut plan = Arc::clone(&plan);
        for _ in 0..3 {
            let output = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
            assert_eq!(
                canonical_ipc_bytes(&expected[0].schema(), &output, false)?,
                expected_bytes
            );
            // Like default DataFusion, the plan can be executed again once
            // its state is reset.
            plan = reset_plan_states(plan)?;
            assert!(plan.is::<PipelineRootExec>());
        }
        Ok(())
    }
}
