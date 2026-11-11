// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`StreamingTableSource`]: the streams of the partitions of a
//! `StreamingTableExec` (e.g. of `information_schema` tables, and of
//! `StreamTable`s, such as FIFO files).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
use futures::StreamExt;

use super::batches::BatchChunks;
use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state, downcast_state_mut,
};

/// A source that reads the streams of the partitions of a
/// `StreamingTableExec` ([`PartitionStream`]s) on the threads of a pipeline.
///
/// Like the partitions of a [`BridgeOperator`](crate::bridge::BridgeOperator),
/// the threads claim the partitions through a shared counter, and a thread
/// polls the streams of the partitions it claimed in turn with the waker of
/// its task, claiming another partition (up to its share,
/// `ceil(partitions / threads)`) when all of its streams are pending; for an
/// infinite table, it claims its share right away. So a partition that is
/// idle or never ends does not hold back the others, and a single partition
/// (whose rows may be ordered) is read in order by a single thread.
///
/// The batches of every partition are projected to the columns of the scan
/// and limited to its `fetch` rows, as `StreamingTableExec` does. The memory
/// of the batch that a thread emits is reserved in the memory pool of the
/// session.
#[derive(Debug)]
pub struct StreamingTableSource {
    partitions: Vec<Arc<dyn PartitionStream>>,
    projection: Option<Arc<[usize]>>,
    schema: SchemaRef,
    /// The rows that every partition produces at most.
    limit: Option<usize>,
    infinite: bool,
}

impl StreamingTableSource {
    /// The source of `exec`.
    pub fn new(exec: &StreamingTableExec) -> Self {
        Self {
            partitions: exec.partitions().clone(),
            projection: exec.projection().clone(),
            schema: exec.schema(),
            limit: exec.limit(),
            infinite: exec.is_infinite(),
        }
    }

    /// Lower `plan` to a [`StreamingTableSource`] if it is a
    /// `StreamingTableExec`.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Option<Self> {
        plan.downcast_ref::<StreamingTableExec>().map(Self::new)
    }
}

/// Global source state of a [`StreamingTableSource`]: the next partition to
/// claim.
struct StreamingGlobalState {
    next: AtomicUsize,
    partitions: usize,
}

impl GlobalSourceState for StreamingGlobalState {
    fn max_threads(&self) -> usize {
        self.partitions.max(1)
    }
}

/// The stream of a partition that a thread claimed.
struct PartitionRows {
    stream: SendableRecordBatchStream,
    /// The rows that the partition may still produce.
    remaining: Option<usize>,
}

/// Local source state of a [`StreamingTableSource`]: the streams of the
/// partitions that the thread claimed and the batch it emits.
struct StreamingLocalState {
    streams: Vec<PartitionRows>,
    /// The stream to poll first, for round robin polling.
    next: usize,
    chunks: BatchChunks,
}

impl LocalSourceState for StreamingLocalState {}

impl StreamingTableSource {
    /// Claim the next partition that no thread claimed yet and add its
    /// stream to `lstate`. Returns `false` if there is none.
    fn claim(
        &self,
        ctx: &ExecutionContext,
        gstate: &StreamingGlobalState,
        lstate: &mut StreamingLocalState,
    ) -> bool {
        let partition = gstate.next.fetch_add(1, Ordering::Relaxed);
        let Some(stream) = self.partitions.get(partition) else {
            return false;
        };
        if self.limit != Some(0) {
            lstate.streams.push(PartitionRows {
                stream: stream.execute(Arc::clone(&ctx.task_ctx)),
                remaining: self.limit,
            });
        }
        true
    }
}

impl PhysicalOperator for StreamingTableSource {
    fn name(&self) -> &str {
        "StreamingTableSource"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        let infinite = if self.infinite {
            ", infinite_source=true"
        } else {
            ""
        };
        let limit = self
            .limit
            .map(|limit| format!(", fetch={limit}"))
            .unwrap_or_default();
        format!(
            "StreamingTableSource: partitions={}{infinite}{limit}",
            self.partitions.len()
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
        Ok(Arc::new(StreamingGlobalState {
            next: AtomicUsize::new(0),
            partitions: self.partitions.len(),
        }))
    }

    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        let reservation = MemoryConsumer::new("StreamingTableSource")
            .register(ctx.task_ctx.memory_pool());
        Ok(Box::new(StreamingLocalState {
            streams: vec![],
            next: 0,
            chunks: BatchChunks::new(reservation),
        }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let gstate =
            downcast_state::<StreamingGlobalState>(gstate.as_any(), self.name())?;
        let lstate =
            downcast_state_mut::<StreamingLocalState>(lstate.as_any_mut(), self.name())?;
        loop {
            if let Some(chunk) = lstate.chunks.next_chunk() {
                return Ok(SourceResult::Chunk(chunk));
            }
            // Claim a partition if this thread has none, and its share of
            // them for an infinite table.
            let share = gstate.partitions.div_ceil(ctx.num_threads.max(1));
            while lstate.streams.is_empty()
                || (self.infinite && lstate.streams.len() < share)
            {
                if !self.claim(ctx, gstate, lstate) {
                    break;
                }
            }
            if lstate.streams.is_empty() {
                return Ok(SourceResult::Finished);
            }

            // Poll the streams in turn, starting after the stream that
            // produced the last batch. Every pending stream registers the
            // waker of the task.
            let mut cx = Context::from_waker(ctx.waker());
            let streams = lstate.streams.len();
            let first = lstate.next % streams;
            let mut polled = None;
            for i in 0..streams {
                let index = (first + i) % streams;
                match lstate.streams[index].stream.poll_next_unpin(&mut cx) {
                    Poll::Ready(Some(batch)) => {
                        polled = Some((index, Some(batch?)));
                        break;
                    }
                    Poll::Ready(None) => {
                        polled = Some((index, None));
                        break;
                    }
                    Poll::Pending => {}
                }
            }
            match polled {
                Some((index, Some(batch))) => {
                    lstate.next = index + 1;
                    let batch = match &self.projection {
                        Some(projection) => batch.project(projection)?,
                        None => batch,
                    };
                    // Like the `LimitStream` of every partition of
                    // `StreamingTableExec`.
                    let partition = &mut lstate.streams[index];
                    let batch = match &mut partition.remaining {
                        Some(remaining) => {
                            let rows = batch.num_rows().min(*remaining);
                            *remaining -= rows;
                            if *remaining == 0 {
                                drop(lstate.streams.remove(index));
                                lstate.next = index;
                            }
                            batch.slice(0, rows)
                        }
                        None => batch,
                    };
                    if batch.num_rows() == 0 {
                        // Return to the executor (which skips empty chunks),
                        // so that the call counts against its budget.
                        return Ok(SourceResult::Chunk(DataChunk::from_record_batch(
                            batch,
                        )));
                    }
                    lstate.chunks.set(batch)?;
                }
                Some((index, None)) => {
                    // The next iteration may claim another partition instead.
                    drop(lstate.streams.remove(index));
                    lstate.next = index;
                }
                None => {
                    // All streams are pending: claim another partition (up
                    // to the share of the thread), so that idle partitions do
                    // not hold back the others, and poll it first.
                    if lstate.streams.len() < share && self.claim(ctx, gstate, lstate) {
                        lstate.next = lstate.streams.len().saturating_sub(1);
                        continue;
                    }
                    return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;

    use arrow::array::{AsArray, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use datafusion::catalog::streaming::StreamingTable;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::DataFusionError;
    use datafusion_execution::TaskContext;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;

    use crate::executor::Executor;
    use crate::operators::lowering_tests::{assert_lowered, count};
    use crate::test_utils::{register_test_tables, test_session_config};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
        ]))
    }

    fn batch(values: std::ops::Range<i64>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(values.clone())),
                Arc::new(StringArray::from_iter(
                    values.map(|v| (v % 3 != 0).then(|| format!("s{v}"))),
                )),
            ],
        )
        .unwrap()
    }

    /// A partition of `batches`; if `pending`, its stream stays pending
    /// once it produced them, and never ends.
    #[derive(Debug)]
    struct Partition {
        batches: Vec<RecordBatch>,
        pending: bool,
        /// How many times the partition was executed.
        executions: Mutex<usize>,
    }

    impl Partition {
        fn new(batches: Vec<RecordBatch>, pending: bool) -> Arc<Self> {
            Arc::new(Self {
                batches,
                pending,
                executions: Mutex::new(0),
            })
        }
    }

    impl PartitionStream for Partition {
        fn schema(&self) -> &SchemaRef {
            static SCHEMA: std::sync::LazyLock<SchemaRef> =
                std::sync::LazyLock::new(schema);
            &SCHEMA
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            *self.executions.lock().unwrap() += 1;
            let batches = futures::stream::iter(self.batches.clone().into_iter().map(Ok));
            let stream = if self.pending {
                batches.chain(futures::stream::pending()).boxed()
            } else {
                batches.boxed()
            };
            Box::pin(RecordBatchStreamAdapter::new(schema(), stream))
        }
    }

    fn values(batches: &[RecordBatch]) -> Vec<i64> {
        let mut values = batches
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect::<Vec<_>>();
        values.sort_unstable();
        values
    }

    fn exec(
        partitions: Vec<Arc<Partition>>,
        projection: Option<&[usize]>,
        infinite: bool,
        limit: Option<usize>,
    ) -> StreamingTableExec {
        let partitions = partitions
            .into_iter()
            .map(|partition| partition as Arc<dyn PartitionStream>)
            .collect();
        StreamingTableExec::try_new(schema(), partitions, projection, [], infinite, limit)
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partitions_are_projected_and_limited() -> Result<()> {
        let partitions = || {
            vec![
                Partition::new(vec![batch(0..5000), batch(5000..5100)], false),
                Partition::new(vec![], false),
                Partition::new(vec![batch(10_000..10_010)], false),
            ]
        };
        for threads in [1, 2, 8] {
            let source: Arc<dyn PhysicalOperator> = Arc::new(StreamingTableSource::new(
                &exec(partitions(), Some(&[1, 0]), false, None),
            ));
            assert_eq!(source.schema().field(0).name(), "s");
            let output =
                Executor::collect(source, Arc::new(TaskContext::default()), threads)
                    .await?;
            assert!(output.iter().all(|batch| batch.num_rows() <= 2048));
            let mut generated = output
                .iter()
                .flat_map(|b| b.column(1).as_primitive::<Int64Type>().values().to_vec())
                .collect::<Vec<_>>();
            generated.sort_unstable();
            assert_eq!(
                generated,
                (0..5100).chain(10_000..10_010).collect::<Vec<_>>()
            );

            // The limit applies to every partition.
            let source = Arc::new(StreamingTableSource::new(&exec(
                partitions(),
                None,
                false,
                Some(4999),
            )));
            assert!(source.description().ends_with("fetch=4999"));
            let output =
                Executor::collect(source, Arc::new(TaskContext::default()), threads)
                    .await?;
            assert_eq!(
                values(&output),
                (0..4999).chain(10_000..10_010).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idle_partitions_do_not_hold_back_the_others() -> Result<()> {
        // Three partitions that stay pending once they produced their rows,
        // read through a limit of all their rows on a single thread: the
        // thread claims the next partition while the others are pending.
        for infinite in [false, true] {
            let partitions = vec![
                Partition::new(vec![batch(0..10)], true),
                Partition::new(vec![batch(100..110)], true),
                Partition::new(vec![batch(200..210)], true),
            ];
            let source: Arc<dyn PhysicalOperator> = Arc::new(StreamingTableSource::new(
                &exec(partitions, None, infinite, None),
            ));
            let limit: Arc<dyn PhysicalOperator> = Arc::new(
                crate::operators::limit::PhysicalLimit::new(source, 0, Some(30)),
            );
            let output =
                Executor::collect(limit, Arc::new(TaskContext::default()), 1).await?;
            assert_eq!(
                values(&output),
                (0..10).chain(100..110).chain(200..210).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_batches_of_a_thread_reserve_memory() -> Result<()> {
        let source = Arc::new(StreamingTableSource::new(&exec(
            vec![Partition::new(vec![batch(0..100_000)], false)],
            None,
            false,
            None,
        )));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(100_000)))
            .build_arc()?;
        let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), runtime);
        let error = Executor::collect(source, ctx.task_ctx(), 1)
            .await
            .unwrap_err();
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
        Ok(())
    }

    /// Register `s`, a `StreamingTable` of three partitions.
    fn register_streaming_table(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)?;
        let partitions = vec![
            Partition::new(vec![batch(0..3000), batch(3000..3500)], false)
                as Arc<dyn PartitionStream>,
            Partition::new(vec![], false),
            Partition::new(vec![batch(5000..5100)], false),
        ];
        let table = StreamingTable::try_new(schema(), partitions)?;
        ctx.register_table("s", Arc::new(table))?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streaming_tables_are_native() {
        for sql in [
            "SELECT * FROM s",
            "SELECT s, v FROM s WHERE v % 7 = 3",
            "SELECT s, COUNT(*) AS n, SUM(v) AS sv FROM s GROUP BY s",
            "SELECT v, s FROM s ORDER BY v DESC LIMIT 17",
            "SELECT s.v, t1.a FROM s JOIN t1 ON s.v = t1.id",
            "SELECT COUNT(*) FROM s",
        ] {
            let root = assert_lowered(
                sql,
                test_session_config(),
                async |ctx: &SessionContext| register_streaming_table(ctx),
            )
            .await;
            assert_eq!(count::<StreamingTableSource>(&root), 1, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn information_schema_tables_are_native() {
        let config = test_session_config().with_information_schema(true);
        for sql in [
            "SELECT table_catalog, table_schema, table_name, table_type \
             FROM information_schema.tables",
            "SELECT table_name, column_name, data_type, is_nullable \
             FROM information_schema.columns WHERE table_name = 't1'",
            "SELECT COUNT(*) FROM information_schema.df_settings",
            "SELECT * FROM information_schema.schemata",
        ] {
            let root =
                assert_lowered(sql, config.clone(), async |ctx: &SessionContext| {
                    register_test_tables(ctx)
                })
                .await;
            assert_eq!(count::<StreamingTableSource>(&root), 1, "{sql}");
        }
    }
}
