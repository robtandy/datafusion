// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalDataSink`]: the writes of `INSERT` and `COPY TO` into a
//! `DataSink` (`DataSinkExec`).

use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{RecordBatch, UInt64Array};
use arrow::datatypes::SchemaRef;
use datafusion_common::{DataFusionError, Result, internal_err};
use datafusion_common_runtime::SpawnedTask;
use datafusion_datasource::sink::DataSinkExec;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_plan::execution_plan::check_not_null_constraints;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, displayable};
use futures::channel::mpsc;
use futures::{FutureExt, StreamExt};
use parking_lot::Mutex;

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
    TaskInterrupt, downcast_state, downcast_state_mut,
};

/// The number of batches that may wait for the writer of a
/// [`PhysicalDataSink`], besides one for every thread.
const QUEUED_BATCHES: usize = 2;

/// Writes the rows of its input into the `DataSink` of a `DataSinkExec` (e.g.
/// of the `INSERT` into a table, or of a `COPY TO` files), and then emits the
/// number of rows that the sink wrote, as `DataSinkExec` does.
///
/// The sink writes its rows with `DataSink::write_all`, in a task that the
/// operator starts when its execution starts: the threads of the pipelines
/// of its input send their rows to that task through a bounded queue (they
/// wait while it is full), as the stream that `DataSinkExec` hands to the
/// sink (with the schema of the sink), and once all of them are done, the
/// operator, as the source of the pipeline above it, waits for the task to
/// complete and emits its count (or its error). The rows of a column that the
/// sink requires to be not null, but that the input may have nulls in, are
/// checked first (`check_not_null_constraints`), as `DataSinkExec` checks
/// them.
///
/// The threads of its input send their rows concurrently, unless the input
/// is ordered: the rows are then sent in their order, by a single thread (so
/// that, e.g., the files of a `COPY` of an ordered query are ordered). The
/// memory of the rows that wait in the queue is reserved in the memory pool
/// of the session.
#[derive(Debug)]
pub struct PhysicalDataSink {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    /// The `DataSinkExec`, whose sink writes the rows.
    plan: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    /// The schema of the stream of rows that the sink writes.
    sink_schema: SchemaRef,
    /// The columns of the input with nulls that the sink does not accept.
    not_null_columns: Vec<usize>,
    /// Whether the rows are written in the order of the input.
    preserve_order: bool,
}

impl PhysicalDataSink {
    /// Lower a `DataSinkExec` to a [`PhysicalDataSink`], whose input is
    /// `lower_input` of the input of the node. Returns `Ok(None)` for any
    /// other node, without calling `lower_input`.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(exec) = plan.downcast_ref::<DataSinkExec>() else {
            return Ok(None);
        };
        let sink_schema = Arc::clone(exec.sink().schema());
        let input_schema = exec.input().schema();
        if sink_schema.fields().len() != input_schema.fields().len() {
            return internal_err!(
                "The sink of a DataSinkExec writes {} columns, its input has {}",
                sink_schema.fields().len(),
                input_schema.fields().len()
            );
        }
        // Like `execute_input_stream`, which `DataSinkExec` reads its input
        // with.
        let not_null_columns = sink_schema
            .fields()
            .iter()
            .zip(input_schema.fields())
            .enumerate()
            .filter(|(_, (sink, input))| !sink.is_nullable() && input.is_nullable())
            .map(|(index, _)| index)
            .collect();
        let preserve_order = exec.input().output_ordering().is_some();
        let input = lower_input(exec.input())?;
        Ok(Some(Self {
            children: vec![input],
            plan: Arc::clone(plan),
            schema: plan.schema(),
            sink_schema,
            not_null_columns,
            preserve_order,
        }))
    }

    fn sink_exec(plan: &Arc<dyn ExecutionPlan>) -> Result<&DataSinkExec> {
        match plan.downcast_ref::<DataSinkExec>() {
            Some(exec) => Ok(exec),
            None => internal_err!("PhysicalDataSink writes into a DataSinkExec"),
        }
    }
}

/// A batch of rows for the writer, with its memory.
type Queued = (RecordBatch, usize);

/// The global sink state of a [`PhysicalDataSink`]: the queue of the rows
/// for the writer, and the task of the writer.
struct DataSinkState {
    /// The sending end of the queue, of which every thread has a clone,
    /// closed once all threads are done (when the sink is finalized).
    sender: Mutex<Option<mpsc::Sender<Queued>>>,
    /// The task that writes the rows, and returns their count.
    writer: Mutex<Option<SpawnedTask<Result<u64>>>>,
    /// The memory of the rows in the queue.
    reservation: Arc<MemoryReservation>,
}

impl GlobalSinkState for DataSinkState {}

/// The local sink state of a [`PhysicalDataSink`]: the sending end of the
/// queue of the thread, and the rows that it could not send yet.
struct DataSinkLocalState {
    /// `None` once the writer stopped reading.
    sender: Option<mpsc::Sender<Queued>>,
    pending: Option<Queued>,
}

impl LocalSinkState for DataSinkLocalState {}

/// The local source state of a [`PhysicalDataSink`]: whether the count was
/// emitted.
struct DataSinkSourceState {
    emitted: bool,
}

impl LocalSourceState for DataSinkSourceState {}

impl PhysicalOperator for PhysicalDataSink {
    fn name(&self) -> &str {
        "PhysicalDataSink"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let node = displayable(self.plan.as_ref()).one_line().to_string();
        let sink = node.trim_end();
        let sink = sink.strip_prefix("DataSinkExec: ").unwrap_or(sink);
        let order = if self.preserve_order {
            ", preserve_order=true"
        } else {
            ""
        };
        format!("PhysicalDataSink: {sink}{order}")
    }

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        !self.preserve_order
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let (sender, receiver) = mpsc::channel::<Queued>(QUEUED_BATCHES);
        let reservation = Arc::new(
            MemoryConsumer::new("PhysicalDataSink").register(ctx.task_ctx.memory_pool()),
        );
        let written = Arc::clone(&reservation);
        let rows = receiver.map(move |(batch, size)| {
            written.shrink(size);
            Ok(batch)
        });
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.sink_schema),
            rows,
        ));
        let plan = Arc::clone(&self.plan);
        let task_ctx = Arc::clone(&ctx.task_ctx);
        let writer = SpawnedTask::spawn(async move {
            let exec = Self::sink_exec(&plan)?;
            exec.sink().write_all(stream, &task_ctx).await
        });
        Ok(Arc::new(DataSinkState {
            sender: Mutex::new(Some(sender)),
            writer: Mutex::new(Some(writer)),
            reservation,
        }))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        let state = downcast_state::<DataSinkState>(gstate.as_any(), self.name())?;
        let sender = state.sender.lock().clone();
        Ok(Box::new(DataSinkLocalState {
            sender,
            pending: None,
        }))
    }

    fn sink(
        &self,
        ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<DataSinkState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<DataSinkLocalState>(lstate.as_any_mut(), self.name())?;
        let queued = match local.pending.take() {
            Some(queued) => queued,
            None => {
                let batch = input.compact()?;
                let batch = if self.not_null_columns.is_empty() {
                    batch
                } else {
                    check_not_null_constraints(batch, &self.not_null_columns)?
                };
                let size = batch.get_array_memory_size();
                state.reservation.try_grow(size)?;
                (batch, size)
            }
        };
        let Some(sender) = local.sender.as_mut() else {
            state.reservation.shrink(queued.1);
            return Ok(SinkResult::Finished);
        };
        let mut cx = Context::from_waker(ctx.waker());
        match sender.poll_ready(&mut cx) {
            Poll::Ready(Ok(())) => {
                let size = queued.1;
                if sender.start_send(queued).is_ok() {
                    return Ok(SinkResult::NeedMoreInput);
                }
                state.reservation.shrink(size);
            }
            Poll::Ready(Err(_)) => state.reservation.shrink(queued.1),
            Poll::Pending => {
                local.pending = Some(queued);
                return Ok(SinkResult::Blocked(TaskInterrupt::waker_registered()));
            }
        }
        // The writer stopped reading (it failed, or returned early): no more
        // rows are needed, its outcome is the outcome of the operator.
        local.sender = None;
        Ok(SinkResult::Finished)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let state = downcast_state::<DataSinkState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<DataSinkLocalState>(lstate.as_any_mut(), self.name())?;
        if let Some((_, size)) = local.pending.take() {
            state.reservation.shrink(size);
        }
        local.sender = None;
        Ok(())
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<DataSinkState>(gstate.as_any(), self.name())?;
        // All threads are done: the stream of the writer ends.
        if let Some(mut sender) = state.sender.lock().take() {
            sender.close_channel();
        }
        Ok(SinkFinalize::Ready)
    }

    fn is_source(&self) -> bool {
        true
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(DataSinkSourceState { emitted: false }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let local =
            downcast_state_mut::<DataSinkSourceState>(lstate.as_any_mut(), self.name())?;
        if local.emitted {
            return Ok(SourceResult::Finished);
        }
        let Some(gstate) = ctx.global_sink_state(self) else {
            return internal_err!("PhysicalDataSink emits its count before it sinks");
        };
        let state = downcast_state::<DataSinkState>(gstate.as_any(), self.name())?;
        let mut writer = state.writer.lock();
        let Some(task) = writer.as_mut() else {
            return internal_err!("PhysicalDataSink emits its count once");
        };
        let mut cx = Context::from_waker(ctx.waker());
        let count = match task.poll_unpin(&mut cx) {
            Poll::Pending => {
                return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
            }
            Poll::Ready(Ok(count)) => count,
            Poll::Ready(Err(error)) if error.is_panic() => {
                std::panic::resume_unwind(error.into_panic())
            }
            Poll::Ready(Err(error)) => {
                Err(DataFusionError::ExecutionJoin(Box::new(error)))
            }
        };
        *writer = None;
        local.emitted = true;
        let batch = RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![Arc::new(UInt64Array::from(vec![count?]))],
        )?;
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::Int64Type;
    use datafusion::prelude::{ParquetReadOptions, SessionContext};
    use datafusion_physical_plan::collect;

    use crate::bridge::{PipelineRootExec, bridged_plans};
    use crate::operators::lowering_tests::{
        assert_resources_exhausted, count, lowered, pipeline_context_with_memory_limit,
    };
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, pipeline_context,
        register_test_tables, run_sql, test_session_config,
    };

    /// The count of the output of a `DataSinkExec`.
    fn count_of(batches: &[RecordBatch]) -> u64 {
        assert_eq!(batches.len(), 1);
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0)
    }

    /// Run the statement `sql` (e.g. an `INSERT` or a `COPY`) with the
    /// pipeline engine of `ctx`: assert that it is native, and return the
    /// count it emits.
    async fn run_native(ctx: &SessionContext, sql: &str) -> Result<u64> {
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        assert!(bridged_plans(root.operator()).is_empty(), "{sql}");
        assert_eq!(count::<PhysicalDataSink>(root.operator()), 1, "{sql}");
        let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert_eq!(batches[0].schema(), plan.schema());
        Ok(count_of(&batches))
    }

    /// Run `insert` into the table that `create` creates, on DataFusion and
    /// natively on the pipeline engine, and compare the counts they return,
    /// and then the rows of `check` (which reads the table).
    async fn assert_same_inserts(create: &str, insert: &str, check: &str) {
        let datafusion = datafusion_context(test_session_config());
        let pipeline = pipeline_context(test_session_config());
        let mut outputs: Vec<EngineOutput> = vec![];
        let mut counts = vec![];
        for (ctx, native) in [(&datafusion, false), (&pipeline, true)] {
            register_test_tables(ctx).unwrap();
            ctx.sql(create).await.unwrap().collect().await.unwrap();
            counts.push(if native {
                run_native(ctx, insert).await.unwrap()
            } else {
                count_of(&run_sql(ctx, insert).await.unwrap().batches)
            });
            outputs.push(run_sql(ctx, check).await.unwrap());
        }
        assert_eq!(counts[0], counts[1], "{insert}");
        if let Err(message) = compare_outputs(&outputs[0], &outputs[1]) {
            panic!("{insert}: {message}");
        }
    }

    /// Run `copy` into the directory `$DIR` of each engine, on DataFusion and
    /// natively on the pipeline engine, and compare the counts they return,
    /// and then the rows of `check` (which reads what they wrote, with
    /// DataFusion).
    async fn assert_same_copies(copy: &str, check: &str, dir: &Path) {
        let mut outputs: Vec<EngineOutput> = vec![];
        let mut counts = vec![];
        for (ctx, engine) in [
            (datafusion_context(test_session_config()), "datafusion"),
            (pipeline_context(test_session_config()), "pipeline"),
        ] {
            register_test_tables(&ctx).unwrap();
            let target = dir.join(engine);
            let sql = |sql: &str| sql.replace("$DIR", target.to_str().unwrap());
            counts.push(if engine == "pipeline" {
                run_native(&ctx, &sql(copy)).await.unwrap()
            } else {
                count_of(&run_sql(&ctx, &sql(copy)).await.unwrap().batches)
            });
            let reader = datafusion_context(test_session_config()).enable_url_table();
            outputs.push(run_sql(&reader, &sql(check)).await.unwrap());
        }
        assert_eq!(counts[0], counts[1], "{copy}");
        if let Err(message) = compare_outputs(&outputs[0], &outputs[1]) {
            panic!("{copy}: {message}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inserts_into_memory_tables_are_native() {
        let create = "CREATE TABLE target (id BIGINT NOT NULL, a INT, d VARCHAR)";
        for insert in [
            "INSERT INTO target SELECT id, a, d FROM t1 WHERE a > 10",
            "INSERT INTO target SELECT id, a, d FROM t1 ORDER BY id DESC",
            "INSERT INTO target VALUES (1, 2, 'x'), (3, NULL, NULL)",
            "INSERT INTO target SELECT id, a, d FROM t_empty",
            "INSERT INTO target SELECT t1.id, t2.k, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
            "INSERT INTO target SELECT k, COUNT(*), MIN(name) FROM t2 \
             WHERE k IS NOT NULL GROUP BY k",
        ] {
            assert_same_inserts(create, insert, "SELECT * FROM target").await;
        }
        // An insert into a table with rows, and an overwrite.
        let create = "CREATE TABLE target AS SELECT id, a FROM t1 WHERE id < 10";
        assert_same_inserts(
            create,
            "INSERT INTO target SELECT id, a FROM t1 WHERE id >= 4990",
            "SELECT * FROM target",
        )
        .await;
        assert_same_inserts(
            create,
            "INSERT OVERWRITE target SELECT id, a FROM t1 WHERE id >= 4990",
            "SELECT * FROM target",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn copies_to_files_are_native() {
        let dir = tempfile::tempdir().unwrap();
        for format in ["PARQUET", "CSV", "JSON", "ARROW"] {
            let name = format.to_lowercase();
            assert_same_copies(
                &format!(
                    "COPY (SELECT id, a, b, d, e FROM t1 WHERE id % 3 = 0) \
                     TO '$DIR/{name}/' STORED AS {format}"
                ),
                &format!("SELECT * FROM '$DIR/{name}/'"),
                dir.path(),
            )
            .await;
            // Partitioned by a column (without nulls, which DataFusion does
            // not write into partitions).
            assert_same_copies(
                &format!(
                    "COPY (SELECT id, a, e FROM t1 WHERE e IS NOT NULL) TO '$DIR/{name}_by_e/' \
                     STORED AS {format} PARTITIONED BY (e)"
                ),
                &format!(
                    "SELECT COUNT(*) AS n, SUM(id) AS s, SUM(a) AS sa \
                     FROM '$DIR/{name}_by_e/'"
                ),
                dir.path(),
            )
            .await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn copies_of_ordered_rows_keep_their_order() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let copy = "COPY (SELECT id, a FROM t1 ORDER BY a DESC NULLS LAST, id) \
                    TO '$DIR/ordered.parquet' STORED AS PARQUET";
        let ctx = pipeline_context(test_session_config());
        register_test_tables(&ctx)?;
        let path = dir.path().join("ordered.parquet");
        let sql = copy.replace("$DIR", dir.path().to_str().unwrap());
        assert_eq!(run_native(&ctx, &sql).await?, 5000);
        let reader = SessionContext::new();
        reader
            .register_parquet("f", path.to_str().unwrap(), ParquetReadOptions::default())
            .await?;
        let written = reader.sql("SELECT id FROM f").await?.collect().await?;
        let ids = written
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect::<Vec<_>>();
        let expected = ctx
            .sql("SELECT id FROM t1 ORDER BY a DESC NULLS LAST, id")
            .await?
            .collect()
            .await?;
        let expected = expected
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(ids, expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn nulls_in_columns_that_are_not_null_fail() -> Result<()> {
        let ctx = pipeline_context(test_session_config());
        register_test_tables(&ctx)?;
        ctx.sql("CREATE TABLE target (a INT NOT NULL)")
            .await?
            .collect()
            .await?;
        let sql = "INSERT INTO target SELECT a FROM t1";
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalDataSink>(&root), 1);
        let error = ctx.sql(sql).await?.collect().await.unwrap_err();
        assert!(error.to_string().contains("non-nullable"), "{error}");
        // Without nulls the rows are written.
        let sql = "INSERT INTO target SELECT a FROM t1 WHERE a IS NOT NULL";
        let written = run_native(&ctx, sql).await?;
        let rows = ctx
            .sql("SELECT COUNT(*) FROM target")
            .await?
            .collect()
            .await?;
        let rows = rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(written, rows.value(0) as u64);
        assert!(written > 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn queued_rows_reserve_memory() -> Result<()> {
        let ctx = pipeline_context_with_memory_limit(1000);
        register_test_tables(&ctx)?;
        ctx.sql("CREATE TABLE target (id BIGINT NOT NULL, d VARCHAR)")
            .await?
            .collect()
            .await?;
        let sql = "INSERT INTO target SELECT id, d FROM t1";
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalDataSink>(&root), 1);
        assert_resources_exhausted(&ctx, sql).await;
        Ok(())
    }
}
