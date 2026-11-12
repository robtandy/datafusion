// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalStreamingMerge`]: the merge of sorted partitions that never end
//! (a `SortPreservingMergeExec` of unbounded inputs).

use std::collections::HashMap;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, RecordBatch};
use arrow::compute::interleave;
use arrow::datatypes::SchemaRef;
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::Result;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use futures::StreamExt;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::executor::{Executor, PipelineStream};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state_mut,
};
use crate::operators::materialized_cte::OuterCte;
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::planner::lower_execution_plan_with_cte_rows;
use crate::sort::memory_size;
use crate::sources::batches::batch_overhead;

/// Merges the rows of its inputs, every one of them sorted by `expr`, into
/// rows sorted by `expr`, as they arrive, as `SortPreservingMergeExec` does:
/// it emits a row once every input that did not end has a row, the smallest
/// of their rows (in the Arrow row format of `expr`), so that it streams
/// over inputs that never end, which the native sort, which reads all of its
/// input first, does not. With a fetch, it emits the first `fetch` rows, and
/// then finishes.
///
/// Every input (the native operators of a partition of the merge, see
/// `Lowering::try_lower_streaming_merge` in `planner.rs`) runs in an
/// execution of its own ([`Executor::execute_stream`]), whose rows the merge,
/// the source of its pipeline, polls with the waker of its task. It holds the
/// batch of every input whose rows it merges, and their keys, which are
/// reserved in the memory pool of the session.
#[derive(Debug)]
pub struct PhysicalStreamingMerge {
    /// The inputs: they are not part of the pipelines of the merge.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    expr: LexOrdering,
    fetch: Option<usize>,
    /// The DataFusion plans of the inputs, if they scan materialized CTEs
    /// around the merge, `outer_ctes` (see [`Self::with_outer_ctes`]).
    plans: Vec<Arc<dyn ExecutionPlan>>,
    outer_ctes: Vec<(u64, OuterCte)>,
}

impl PhysicalStreamingMerge {
    /// Merge `inputs`, every one sorted by `expr`, into rows with the schema
    /// `schema`, keeping the first `fetch` rows (if any).
    pub fn try_new(
        inputs: Vec<Arc<dyn PhysicalOperator>>,
        schema: SchemaRef,
        expr: LexOrdering,
        fetch: Option<usize>,
    ) -> Result<Self> {
        // The keys must be in the row format.
        Self::converter(&expr, &schema)?;
        Ok(Self {
            children: inputs,
            schema,
            expr,
            fetch,
            plans: vec![],
            outer_ctes: vec![],
        })
    }

    /// This merge, whose inputs (the native operators of `plans`) scan the
    /// materialized CTEs around it, `outer_ctes` (by the ids of their
    /// buffers): an execution of an input, apart from the execution of the
    /// merge, does not see the rows of their bodies, so every execution of
    /// the merge lowers `plans` again, with scans of the rows of the bodies
    /// (see [`OuterCte`]), which the execution of the merge computed before.
    pub(crate) fn with_outer_ctes(
        mut self,
        plans: Vec<Arc<dyn ExecutionPlan>>,
        outer_ctes: Vec<(u64, OuterCte)>,
    ) -> Self {
        self.plans = plans;
        self.outer_ctes = outer_ctes;
        self
    }

    /// The inputs that an execution of the merge executes, in the execution
    /// of `ctx`.
    fn inputs(&self, ctx: &ExecutionContext) -> Result<Vec<Arc<dyn PhysicalOperator>>> {
        if self.outer_ctes.is_empty() {
            return Ok(self.children.clone());
        }
        let rows = self
            .outer_ctes
            .iter()
            .map(|(id, cte)| Ok((*id, cte.rows(ctx)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        let options = ctx.task_ctx.session_config().options();
        self.plans
            .iter()
            .map(|plan| {
                let plan = reset_plan_states(Arc::clone(plan))?;
                lower_execution_plan_with_cte_rows(
                    &plan,
                    options.as_ref(),
                    ctx.task_ctx.runtime_env().as_ref(),
                    &rows,
                )
            })
            .collect()
    }

    /// The row converter of the keys `expr` of rows with the schema `schema`.
    fn converter(expr: &LexOrdering, schema: &SchemaRef) -> Result<RowConverter> {
        let fields = expr
            .iter()
            .map(|key| {
                Ok(SortField::new_with_options(
                    key.expr.data_type(schema)?,
                    key.options,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RowConverter::new(fields)?)
    }
}

/// An input of a [`PhysicalStreamingMerge`]: its stream, and the batch whose
/// rows the merge merges, their keys and the next of them.
struct Input {
    stream: PipelineStream,
    batch: Option<(RecordBatch, Rows)>,
    next: usize,
    ended: bool,
}

/// The state of the thread of a [`PhysicalStreamingMerge`].
struct MergeState {
    converter: RowConverter,
    inputs: Vec<Input>,
    /// The rows that may still be emitted.
    remaining: Option<usize>,
    reservation: MemoryReservation,
}

impl LocalSourceState for MergeState {}

impl MergeState {
    /// Reserve the memory of the batches of the inputs and of their keys.
    fn reserve(&self) -> Result<()> {
        let memory = self
            .inputs
            .iter()
            .filter_map(|input| input.batch.as_ref())
            .map(|(batch, keys)| memory_size(batch) + batch_overhead(batch) + keys.size())
            .sum();
        self.reservation.try_resize(memory)
    }
}

impl PhysicalOperator for PhysicalStreamingMerge {
    fn name(&self) -> &str {
        "PhysicalStreamingMerge"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let fetch = self
            .fetch
            .map(|fetch| format!(", fetch={fetch}"))
            .unwrap_or_default();
        format!(
            "PhysicalStreamingMerge: expr=[{}], inputs={}{fetch}",
            self.expr,
            self.children.len()
        )
    }

    fn is_source(&self) -> bool {
        true
    }

    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        // The source is not parallel: a single thread merges the inputs.
        let inputs = self
            .inputs(ctx)?
            .iter()
            .map(|input| {
                let stream = Executor::execute_stream(
                    Arc::clone(input),
                    Arc::clone(&ctx.task_ctx),
                    ctx.num_threads,
                )?;
                Ok(Input {
                    stream,
                    batch: None,
                    next: 0,
                    ended: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let reservation = MemoryConsumer::new("PhysicalStreamingMerge")
            .register(ctx.task_ctx.memory_pool());
        Ok(Box::new(MergeState {
            converter: Self::converter(&self.expr, &self.schema)?,
            inputs,
            remaining: self.fetch,
            reservation,
        }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state_mut::<MergeState>(lstate.as_any_mut(), self.name())?;
        if state.remaining == Some(0) {
            return Ok(SourceResult::Finished);
        }
        // Every input that did not end needs a row.
        let mut cx = Context::from_waker(ctx.waker());
        for input in &mut state.inputs {
            while !input.ended && input.batch.is_none() {
                match input.stream.poll_next_unpin(&mut cx) {
                    Poll::Ready(Some(batch)) => {
                        let batch = batch?;
                        if batch.num_rows() == 0 {
                            continue;
                        }
                        let keys = self
                            .expr
                            .iter()
                            .map(|key| {
                                key.expr.evaluate(&batch)?.into_array(batch.num_rows())
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let keys = state.converter.convert_columns(&keys)?;
                        input.batch = Some((batch, keys));
                        input.next = 0;
                    }
                    Poll::Ready(None) => input.ended = true,
                    Poll::Pending => {
                        state.reserve()?;
                        return Ok(SourceResult::Blocked(
                            TaskInterrupt::waker_registered(),
                        ));
                    }
                }
            }
        }
        state.reserve()?;
        if state.inputs.iter().all(|input| input.batch.is_none()) {
            return Ok(SourceResult::Finished);
        }
        // The smallest rows, until the batch of an input is consumed (its
        // next batch may have smaller rows than the others).
        let limit = state
            .remaining
            .unwrap_or(usize::MAX)
            .min(DEFAULT_VECTOR_SIZE);
        let mut rows = Vec::with_capacity(limit);
        let mut consumed = false;
        while rows.len() < limit && !consumed {
            let (smallest, _) = state
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(index, input)| {
                    let (_, keys) = input.batch.as_ref()?;
                    Some((index, keys.row(input.next)))
                })
                .min_by(|(_, a), (_, b)| a.cmp(b))
                .expect("an input has a row");
            let input = &mut state.inputs[smallest];
            rows.push((smallest, input.next));
            input.next += 1;
            let len = input
                .batch
                .as_ref()
                .map_or(0, |(batch, _)| batch.num_rows());
            consumed = input.next == len;
        }
        // The batches of the inputs, in the order of the inputs.
        let batches = state
            .inputs
            .iter()
            .map(|input| input.batch.as_ref().map(|(batch, _)| batch))
            .collect::<Vec<_>>();
        let present = batches
            .iter()
            .enumerate()
            .filter_map(|(index, batch)| batch.map(|batch| (index, batch)))
            .collect::<Vec<_>>();
        let position = |input: usize| {
            present
                .iter()
                .position(|&(index, _)| index == input)
                .expect("the input of a row has a batch")
        };
        let indices = rows
            .iter()
            .map(|&(input, row)| (position(input), row))
            .collect::<Vec<_>>();
        let columns = (0..self.schema.fields().len())
            .map(|column| {
                let arrays = present
                    .iter()
                    .map(|(_, batch)| batch.column(column).as_ref())
                    .collect::<Vec<&dyn Array>>();
                Ok(interleave(&arrays, &indices)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let output = RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &arrow::array::RecordBatchOptions::new().with_row_count(Some(indices.len())),
        )?;
        // The batches that are consumed are released.
        for input in &mut state.inputs {
            if input
                .batch
                .as_ref()
                .is_some_and(|(batch, _)| input.next == batch.num_rows())
            {
                input.batch = None;
            }
        }
        if let Some(remaining) = &mut state.remaining {
            *remaining -= output.num_rows();
        }
        state.reserve()?;
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(output)))
    }

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        // The merge is the source of the pipeline; its inputs run in
        // executions of their own.
        meta.build_source(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::LazyLock;
    use std::time::Duration;

    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use datafusion_execution::{SendableRecordBatchStream, TaskContext};
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
    use datafusion_physical_plan::{ExecutionPlan, Partitioning};

    use crate::bridge::bridged_plans;
    use crate::operators::lowering_tests::{assert_plan_lowered, count};
    use crate::test_utils::{datafusion_context, test_session_config};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]))
    }

    fn ordering() -> LexOrdering {
        LexOrdering::new(vec![PhysicalSortExpr::new_default(
            col("a", &schema()).unwrap(),
        )])
        .unwrap()
    }

    /// A partition of batches whose stream stays pending once it produced
    /// them: it never ends.
    #[derive(Debug)]
    struct Endless(Vec<RecordBatch>);

    impl PartitionStream for Endless {
        fn schema(&self) -> &SchemaRef {
            static SCHEMA: LazyLock<SchemaRef> = LazyLock::new(schema);
            &SCHEMA
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            let batches = futures::stream::iter(self.0.clone().into_iter().map(Ok))
                .chain(futures::stream::pending());
            Box::pin(RecordBatchStreamAdapter::new(schema(), batches))
        }
    }

    /// A streaming table of `partitions` partitions that never end, sorted
    /// by `a`: partition `p` has the values `p, p + partitions, ...` below
    /// 1000, in batches of 7 rows.
    fn endless_table(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let streams = (0..partitions)
            .map(|partition| {
                let values = (partition as i64..1000)
                    .step_by(partitions)
                    .collect::<Vec<_>>();
                let batches = values
                    .chunks(7)
                    .map(|values| {
                        RecordBatch::try_new(
                            schema(),
                            vec![Arc::new(Int64Array::from(values.to_vec()))],
                        )
                        .unwrap()
                    })
                    .collect();
                Arc::new(Endless(batches)) as Arc<dyn PartitionStream>
            })
            .collect();
        let table = StreamingTableExec::try_new(
            schema(),
            streams,
            None,
            [ordering()],
            true,
            None,
        )
        .unwrap();
        Arc::new(table)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_partitions_that_never_end_are_native() {
        for (partitions, fetch) in [(2, 5), (3, 100), (4, 1), (2, 900)] {
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    SortPreservingMergeExec::new(ordering(), endless_table(partitions))
                        .with_fetch(Some(fetch)),
                )
            };
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert_eq!(count::<PhysicalStreamingMerge>(&root), 1, "{partitions}");
        }
        // A single partition, which a repartitioning that keeps the order of
        // every partition distributes, is merged already.
        let make_plan = || -> Arc<dyn ExecutionPlan> {
            let repartition = RepartitionExec::try_new(
                endless_table(1),
                Partitioning::RoundRobinBatch(4),
            )
            .unwrap()
            .with_preserve_order();
            Arc::new(
                SortPreservingMergeExec::new(ordering(), Arc::new(repartition))
                    .with_fetch(Some(10)),
            )
        };
        let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
        assert_eq!(count::<PhysicalStreamingMerge>(&root), 0);
    }

    fn pairs_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]))
    }

    /// A partition of batches of `pairs_schema` that never ends (unless
    /// `ends`, in a table that DataFusion considers unbounded all the same).
    #[derive(Debug)]
    struct EndlessPairs(Vec<RecordBatch>, bool);

    impl PartitionStream for EndlessPairs {
        fn schema(&self) -> &SchemaRef {
            static SCHEMA: LazyLock<SchemaRef> = LazyLock::new(pairs_schema);
            &SCHEMA
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            let batches = futures::stream::iter(self.0.clone().into_iter().map(Ok));
            let batches = match self.1 {
                true => batches.boxed(),
                false => batches.chain(futures::stream::pending()).boxed(),
            };
            Box::pin(RecordBatchStreamAdapter::new(pairs_schema(), batches))
        }
    }

    /// The merge, with a fetch of `merge_fetch`, of the partial sorts by `a`
    /// (the prefix) and `b`, with a fetch of `partial_fetch`, of the two
    /// partitions of a streaming table that never end: partition `p` has the
    /// rows of the values `v = p, p + 2, ...` below 1000, `a = v / 10` (sorted)
    /// and `b = v * 7 % 10`, in batches of 7 rows.
    fn merged_partial_sorts(
        partial_fetch: Option<usize>,
        merge_fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = pairs_schema();
        let streams = (0..2)
            .map(|partition| {
                let values = (partition..1000).step_by(2).collect::<Vec<i64>>();
                let batches = values
                    .chunks(7)
                    .map(|values| {
                        let a = values.iter().map(|v| v / 10).collect::<Vec<_>>();
                        let b = values.iter().map(|v| v * 7 % 10).collect::<Vec<_>>();
                        RecordBatch::try_new(
                            pairs_schema(),
                            vec![
                                Arc::new(Int64Array::from(a)),
                                Arc::new(Int64Array::from(b)),
                            ],
                        )
                        .unwrap()
                    })
                    .collect();
                Arc::new(EndlessPairs(batches, false)) as Arc<dyn PartitionStream>
            })
            .collect();
        let a = PhysicalSortExpr::new_default(col("a", &schema).unwrap());
        let b = PhysicalSortExpr::new_default(col("b", &schema).unwrap());
        let table = StreamingTableExec::try_new(
            Arc::clone(&schema),
            streams,
            None,
            [LexOrdering::new(vec![a.clone()]).unwrap()],
            true,
            None,
        )
        .unwrap();
        let ordering = LexOrdering::new(vec![a, b]).unwrap();
        let sort = datafusion_physical_plan::sorts::partial_sort::PartialSortExec::new(
            ordering.clone(),
            Arc::new(table),
            1,
        )
        .with_preserve_partitioning(true)
        .with_fetch(partial_fetch);
        Arc::new(
            SortPreservingMergeExec::new(ordering, Arc::new(sort))
                .with_fetch(merge_fetch),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_partial_sorts_of_partitions_that_never_end_are_native()
    -> Result<()> {
        for (partial_fetch, merge_fetch) in
            [(None, 2), (Some(1), 2), (Some(3), 5), (None, 500)]
        {
            let make_plan = || merged_partial_sorts(partial_fetch, Some(merge_fetch));
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert_eq!(
                count::<PhysicalStreamingMerge>(&root),
                1,
                "{partial_fetch:?}"
            );
            assert_eq!(
                count::<crate::operators::partial_sort::PhysicalPartialSort>(&root),
                2,
                "{partial_fetch:?}"
            );
        }
        // Without a fetch, the merge emits the rows of the groups that the
        // partial sorts completed while their inputs never end.
        let ctx = datafusion_context(test_session_config());
        let state = ctx.state();
        let root = crate::planner::create_pipeline_plan_with_runtime(
            merged_partial_sorts(None, None),
            state.config_options(),
            state.runtime_env(),
        )?;
        assert!(bridged_plans(root.operator()).is_empty());
        let mut stream = root.execute(0, ctx.task_ctx())?;
        let mut rows = 0;
        while rows < 900 {
            let batch = tokio::time::timeout(Duration::from_secs(10), stream.next())
                .await
                .expect("the merge emits rows")
                .expect("the merge does not end")?;
            rows += batch.num_rows();
        }
        Ok(())
    }

    /// A generator of the values `next, next + step, ...` (sorted), in
    /// batches of 7 rows, that never ends.
    #[derive(Debug, Clone, Copy)]
    struct Counter {
        next: i64,
        step: i64,
    }

    impl std::fmt::Display for Counter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "counter: next={}, step={}", self.next, self.step)
        }
    }

    impl datafusion_physical_plan::memory::LazyBatchGenerator for Counter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn boundedness(&self) -> datafusion_physical_plan::execution_plan::Boundedness {
            datafusion_physical_plan::execution_plan::Boundedness::Unbounded {
                requires_infinite_memory: false,
            }
        }

        fn generate_next_batch(&mut self) -> Result<Option<RecordBatch>> {
            let values = (0..7)
                .map(|i| self.next + i * self.step)
                .collect::<Vec<_>>();
            self.next += 7 * self.step;
            Ok(Some(RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from(values))],
            )?))
        }

        fn reset_state(
            &self,
        ) -> Arc<
            parking_lot::RwLock<dyn datafusion_physical_plan::memory::LazyBatchGenerator>,
        > {
            Arc::new(parking_lot::RwLock::new(*self))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_generators_that_never_end_are_native() {
        // Two generators of sorted values that never end, merged.
        for fetch in [4, 100] {
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                let generators = [(0, 2), (1, 2)]
                    .map(|(next, step)| {
                        Arc::new(parking_lot::RwLock::new(Counter { next, step }))
                            as Arc<
                                parking_lot::RwLock<
                                    dyn datafusion_physical_plan::memory::LazyBatchGenerator,
                                >,
                            >
                    })
                    .to_vec();
                let mut exec = datafusion_physical_plan::memory::LazyMemoryExec::try_new(
                    schema(),
                    generators,
                )
                .unwrap();
                exec.add_ordering(ordering().iter().cloned());
                Arc::new(
                    SortPreservingMergeExec::new(ordering(), Arc::new(exec))
                        .with_fetch(Some(fetch)),
                )
            };
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert_eq!(count::<PhysicalStreamingMerge>(&root), 1, "{fetch}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_inputs_that_scan_the_materialized_ctes_around_them_are_native() {
        use datafusion_physical_plan::materialized_cte::{
            MaterializedCteBuffer, MaterializedCteExec, MaterializedCteScanExec,
        };
        // A stream that never ends, merged with the scan of a materialized
        // CTE around the merge (which an execution of its own scans).
        let make_plan = || -> Arc<dyn ExecutionPlan> {
            let body = datafusion_datasource::memory::MemorySourceConfig::try_new_exec(
                &[vec![
                    RecordBatch::try_new(
                        schema(),
                        vec![Arc::new(Int64Array::from(vec![1, 5, 9]))],
                    )
                    .unwrap(),
                ]],
                schema(),
                None,
            )
            .unwrap();
            let buffer = Arc::new(MaterializedCteBuffer::new(7, "c"));
            let scan = MaterializedCteScanExec::new(7, "c", schema(), 1)
                .bind(Arc::clone(&buffer));
            let union = datafusion_physical_plan::union::UnionExec::try_new(vec![
                endless_table(1),
                Arc::new(scan),
            ])
            .unwrap();
            let merge =
                SortPreservingMergeExec::new(ordering(), union).with_fetch(Some(5));
            Arc::new(MaterializedCteExec::new(body, Arc::new(merge), buffer))
        };
        let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
        assert_eq!(count::<PhysicalStreamingMerge>(&root), 1);
        assert_eq!(
            count::<crate::operators::materialized_cte::PhysicalMaterializedCte>(&root),
            1
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partial_sorts_of_repartitionings_of_unbounded_streams() -> Result<()> {
        // Two streams (of an unbounded table), of the prefixes 100..103 and
        // 0..3, every group with `b = 0..100`; repartitioned by a hash of
        // `b` into four partitions (keeping their order), whose partial
        // sorts keep their first row; merged with a fetch of 4: the rows of
        // the prefix 0. (The streams end: DataFusion's merges of the
        // partitions of a repartitioning wait for whole batches of streams
        // that stay pending, the native merges do not.)
        let make_plan = |ends: bool| -> Arc<dyn ExecutionPlan> {
            let schema = pairs_schema();
            let stream = |prefixes: std::ops::Range<i64>| -> Arc<dyn PartitionStream> {
                let batches = prefixes
                    .map(|a| {
                        RecordBatch::try_new(
                            pairs_schema(),
                            vec![
                                Arc::new(Int64Array::from(vec![a; 100])),
                                Arc::new(Int64Array::from_iter_values(0..100)),
                            ],
                        )
                        .unwrap()
                    })
                    .collect();
                Arc::new(EndlessPairs(batches, ends))
            };
            let a = PhysicalSortExpr::new_default(col("a", &schema).unwrap());
            let b = PhysicalSortExpr::new_default(col("b", &schema).unwrap());
            let table = StreamingTableExec::try_new(
                Arc::clone(&schema),
                vec![stream(100..103), stream(0..3)],
                None,
                [LexOrdering::new(vec![a.clone()]).unwrap()],
                true,
                None,
            )
            .unwrap();
            let hash = Partitioning::Hash(vec![col("b", &schema).unwrap()], 4);
            let repartition = RepartitionExec::try_new(Arc::new(table), hash)
                .unwrap()
                .with_preserve_order();
            let ordering = LexOrdering::new(vec![a, b]).unwrap();
            let sort =
                datafusion_physical_plan::sorts::partial_sort::PartialSortExec::new(
                    ordering.clone(),
                    Arc::new(repartition),
                    1,
                )
                .with_preserve_partitioning(true)
                .with_fetch(Some(1));
            Arc::new(
                SortPreservingMergeExec::new(ordering, Arc::new(sort))
                    .with_fetch(Some(4)),
            )
        };
        let lowered =
            assert_plan_lowered(|| make_plan(true), test_session_config(), true);
        let root = tokio::time::timeout(Duration::from_secs(30), lowered)
            .await
            .expect("the merge ends once every partition has its row");
        assert_eq!(count::<PhysicalStreamingMerge>(&root), 1);
        // Of streams that never end, the native merge emits the rows of the
        // groups as they complete: the partial sorts end with their rows.
        let ctx = datafusion_context(test_session_config());
        let state = ctx.state();
        let root = crate::planner::create_pipeline_plan_with_runtime(
            make_plan(false),
            state.config_options(),
            state.runtime_env(),
        )?;
        assert!(bridged_plans(root.operator()).is_empty());
        let run = datafusion_physical_plan::collect(root, ctx.task_ctx());
        let output = tokio::time::timeout(Duration::from_secs(30), run)
            .await
            .expect("the partial sorts end with their rows")?;
        let prefixes = output
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(prefixes, vec![0; 4]);
        Ok(())
    }

    /// [`endless_table`], whose streams end (in a table that DataFusion
    /// considers unbounded all the same).
    fn ending_table(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let streams = (0..partitions)
            .map(|partition| {
                let values = (partition as i64..1000).step_by(partitions);
                let a = values.clone().collect::<Vec<_>>();
                let b = values.map(|value| value % 3).collect::<Vec<_>>();
                let batches = a
                    .chunks(7)
                    .zip(b.chunks(7))
                    .map(|(a, b)| {
                        RecordBatch::try_new(
                            pairs_schema(),
                            vec![
                                Arc::new(Int64Array::from(a.to_vec())),
                                Arc::new(Int64Array::from(b.to_vec())),
                            ],
                        )
                        .unwrap()
                    })
                    .collect();
                Arc::new(EndlessPairs(batches, true)) as Arc<dyn PartitionStream>
            })
            .collect();
        let table = StreamingTableExec::try_new(
            pairs_schema(),
            streams,
            None,
            [ordering()],
            true,
            None,
        )
        .unwrap();
        Arc::new(table)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_through_repartitionings_below_projections_and_unions() {
        // Repartitionings that keep the order of their partitions, below a
        // projection and below a union, of streams that never end.
        let repartitioned = |table: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
            Arc::new(
                RepartitionExec::try_new(table, Partitioning::RoundRobinBatch(4))
                    .unwrap()
                    .with_preserve_order(),
            )
        };
        let projected = |input: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
            let schema = input.schema();
            let a = vec![(col("a", &schema).unwrap(), "a".to_owned())];
            Arc::new(
                datafusion_physical_plan::projection::ProjectionExec::try_new(a, input)
                    .unwrap(),
            )
        };
        for (union, fetch) in [(false, 10), (true, 10), (true, 300)] {
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                let input = match union {
                    false => projected(repartitioned(ending_table(1))),
                    true => datafusion_physical_plan::union::UnionExec::try_new(vec![
                        projected(repartitioned(ending_table(1))),
                        projected(repartitioned(ending_table(2))),
                    ])
                    .unwrap(),
                };
                Arc::new(
                    SortPreservingMergeExec::new(ordering(), input)
                        .with_fetch(Some(fetch)),
                )
            };
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert_eq!(
                count::<PhysicalStreamingMerge>(&root),
                usize::from(union),
                "{union}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_partitions_of_a_merge_read_one_materialization() -> Result<()> {
        use datafusion_physical_plan::materialized_cte::{
            MaterializedCteBuffer, MaterializedCteExec, MaterializedCteScanExec,
        };
        // Two scans of a materialized `random()` around the merge, and a
        // stream of `0, 1, ...` that never ends: both scans read one value.
        let ctx = datafusion_context(test_session_config());
        let body = ctx
            .sql("SELECT CAST(random() * 500 AS BIGINT) AS a")
            .await?
            .create_physical_plan()
            .await?;
        let buffer = Arc::new(MaterializedCteBuffer::new(8, "c"));
        let scan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                MaterializedCteScanExec::new(8, "c", schema(), 1)
                    .bind(Arc::clone(&buffer)),
            )
        };
        let union = datafusion_physical_plan::union::UnionExec::try_new(vec![
            scan(),
            endless_table(1),
            scan(),
        ])?;
        let merge = SortPreservingMergeExec::new(ordering(), union).with_fetch(Some(600));
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(MaterializedCteExec::new(body, Arc::new(merge), buffer));
        let state = ctx.state();
        let root = crate::planner::create_pipeline_plan_with_runtime(
            plan,
            state.config_options(),
            state.runtime_env(),
        )?;
        assert!(bridged_plans(root.operator()).is_empty());
        assert_eq!(count::<PhysicalStreamingMerge>(root.operator()), 1);
        let output = datafusion_physical_plan::collect(root, ctx.task_ctx()).await?;
        let values = output
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        // The values `0..598` of the stream, and the value of the CTE (below
        // 500) twice: one value three times, the others once.
        assert_eq!(values.len(), 600);
        let mut counts = std::collections::BTreeMap::<i64, usize>::new();
        for value in values {
            *counts.entry(value).or_default() += 1;
        }
        let repeated = counts
            .values()
            .filter(|&&count| count > 1)
            .collect::<Vec<_>>();
        assert_eq!(repeated, vec![&3], "{counts:?}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_emit_their_rows_before_their_inputs_end() -> Result<()> {
        // Without a fetch, the merge of inputs that never end (and so never
        // ends either) emits their rows as they arrive.
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(SortPreservingMergeExec::new(ordering(), endless_table(3)));
        let ctx = datafusion_context(test_session_config());
        let state = ctx.state();
        let root = crate::planner::create_pipeline_plan_with_runtime(
            plan,
            state.config_options(),
            state.runtime_env(),
        )?;
        let operator = Arc::clone(root.operator());
        assert_eq!(count::<PhysicalStreamingMerge>(&operator), 1);
        assert!(bridged_plans(&operator).is_empty());
        let mut stream = root.execute(0, ctx.task_ctx())?;
        let mut values = vec![];
        while values.len() < 900 {
            let batch = tokio::time::timeout(Duration::from_secs(10), stream.next())
                .await
                .expect("the merge emits rows")
                .expect("the merge does not end")?;
            values
                .extend_from_slice(batch.column(0).as_primitive::<Int64Type>().values());
        }
        assert_eq!(values[..900], (0..900).collect::<Vec<_>>());
        Ok(())
    }
}
