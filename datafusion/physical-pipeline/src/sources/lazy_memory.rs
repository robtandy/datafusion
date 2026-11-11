// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`LazyMemorySource`]: the batches that the generators of a
//! `LazyMemoryExec` (e.g. of `generate_series` and `range`) generate.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::memory::{LazyBatchGenerator, LazyMemoryExec};
use parking_lot::RwLock;

use super::batches::BatchChunks;
use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state, downcast_state_mut,
};

/// A generator of batches of a `LazyMemoryExec`.
type Generator = Arc<RwLock<dyn LazyBatchGenerator>>;

/// A source that calls the batch generators of a `LazyMemoryExec` (e.g. of
/// `generate_series` and `range`), one generator per partition of the plan,
/// on the threads of a pipeline.
///
/// Every execution generates the batches of fresh generators
/// (`LazyBatchGenerator::reset_state`), as `LazyMemoryExec` does. A thread
/// claims a generator, emits its batches in chunks of at most
/// [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows until it
/// is exhausted, and claims the next one, so the source runs on at most as
/// many threads as it has generators, and a single generator (whose rows may
/// be ordered, like those of `generate_series`) is read in order by a single
/// thread. If a generator is unbounded, every thread claims its share of the
/// generators right away and calls them in turn, so that a generator that
/// never ends does not hold back the others.
///
/// The batches are projected to the columns of the plan, as `LazyMemoryExec`
/// does, and the memory of the batch that a thread emits is reserved in the
/// memory pool of the session.
#[derive(Debug)]
pub struct LazyMemorySource {
    schema: SchemaRef,
    generators: Vec<Generator>,
    /// Whether a generator may never end.
    unbounded: bool,
}

impl LazyMemorySource {
    /// The source of `exec`.
    pub fn new(exec: &LazyMemoryExec) -> Self {
        Self {
            schema: exec.schema(),
            generators: exec.generators().clone(),
            unbounded: exec.properties().boundedness.is_unbounded(),
        }
    }

    /// Lower `plan` to a [`LazyMemorySource`] if it is a `LazyMemoryExec`.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Option<Self> {
        plan.downcast_ref::<LazyMemoryExec>().map(Self::new)
    }
}

/// Global source state of a [`LazyMemorySource`]: the fresh generators of the
/// execution, and the next one to claim.
struct LazyMemoryGlobalState {
    generators: Vec<Generator>,
    next: AtomicUsize,
}

impl GlobalSourceState for LazyMemoryGlobalState {
    fn max_threads(&self) -> usize {
        self.generators.len().max(1)
    }
}

/// Local source state of a [`LazyMemorySource`]: the generators that the
/// thread claimed, and the batch it emits.
struct LazyMemoryLocalState {
    claimed: Vec<Generator>,
    /// The claimed generator to call next, for round robin calls.
    next: usize,
    chunks: BatchChunks,
    /// The projection of the batches of the last schema the generators
    /// produced, see [`projection_of`].
    projection: Option<(SchemaRef, Option<Vec<usize>>)>,
}

impl LocalSourceState for LazyMemoryLocalState {}

impl LazyMemorySource {
    /// Claim the next generator that no thread claimed yet. Returns `false`
    /// if there is none.
    fn claim(
        &self,
        gstate: &LazyMemoryGlobalState,
        lstate: &mut LazyMemoryLocalState,
    ) -> bool {
        let index = gstate.next.fetch_add(1, Ordering::Relaxed);
        match gstate.generators.get(index) {
            Some(generator) => {
                lstate.claimed.push(Arc::clone(generator));
                true
            }
            None => false,
        }
    }

    /// `batch`, projected to the columns of the source.
    fn project(
        &self,
        batch: RecordBatch,
        lstate: &mut LazyMemoryLocalState,
    ) -> Result<RecordBatch> {
        let schema = batch.schema();
        let projection = match &lstate.projection {
            Some((projected, projection)) if Arc::ptr_eq(projected, &schema) => {
                projection
            }
            _ => {
                let projection = projection_of(&schema, &self.schema)?;
                &lstate.projection.insert((schema, projection)).1
            }
        };
        Ok(match projection {
            Some(columns) => batch.project(columns)?,
            None => batch,
        })
    }
}

/// The columns of the batches of `generated` (the schema of the batches that
/// a generator produced) that a `LazyMemoryExec` with the schema `schema`
/// emits, `None` for all of them.
///
/// `LazyMemoryExec` projects the batches of its generators to the columns of
/// its scan, whose schema is the projection of the schema of the generated
/// batches: the columns are found by their names, which are unique in the
/// schema of a table.
fn projection_of(generated: &Schema, schema: &Schema) -> Result<Option<Vec<usize>>> {
    if generated.fields() == schema.fields() {
        return Ok(None);
    }
    schema
        .fields()
        .iter()
        .map(|field| {
            let mut matches = generated
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, generated)| generated.name() == field.name());
            match (matches.next(), matches.next()) {
                (Some((index, _)), None) => Ok(index),
                _ => internal_err!(
                    "LazyMemorySource cannot find the column {} in the batches of its generators: {generated}",
                    field.name()
                ),
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

impl PhysicalOperator for LazyMemorySource {
    fn name(&self) -> &str {
        "LazyMemorySource"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        let generators = self
            .generators
            .iter()
            .map(|generator| generator.read().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "LazyMemorySource: partitions={}, batch_generators=[{generators}]",
            self.generators.len()
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
        // Like `LazyMemoryExec::execute`, which generates the batches of a
        // fresh generator in every execution.
        let generators = self
            .generators
            .iter()
            .map(|generator| generator.read().reset_state())
            .collect();
        Ok(Arc::new(LazyMemoryGlobalState {
            generators,
            next: AtomicUsize::new(0),
        }))
    }

    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        let reservation =
            MemoryConsumer::new("LazyMemorySource").register(ctx.task_ctx.memory_pool());
        Ok(Box::new(LazyMemoryLocalState {
            claimed: vec![],
            next: 0,
            chunks: BatchChunks::new(reservation),
            projection: None,
        }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let gstate =
            downcast_state::<LazyMemoryGlobalState>(gstate.as_any(), self.name())?;
        let lstate =
            downcast_state_mut::<LazyMemoryLocalState>(lstate.as_any_mut(), self.name())?;
        loop {
            if let Some(chunk) = lstate.chunks.next_chunk() {
                return Ok(SourceResult::Chunk(chunk));
            }
            // Claim a generator if the thread has none, and its share of
            // them if a generator may never end.
            let share = gstate.generators.len().div_ceil(ctx.num_threads.max(1));
            while lstate.claimed.is_empty()
                || (self.unbounded && lstate.claimed.len() < share)
            {
                if !self.claim(gstate, lstate) {
                    break;
                }
            }
            if lstate.claimed.is_empty() {
                return Ok(SourceResult::Finished);
            }
            let index = lstate.next % lstate.claimed.len();
            let batch = lstate.claimed[index].write().generate_next_batch()?;
            match batch {
                Some(batch) => {
                    lstate.next = index + 1;
                    let batch = self.project(batch, lstate)?;
                    if batch.num_rows() == 0 {
                        // Return to the executor (which skips empty chunks),
                        // so that the call counts against its budget.
                        return Ok(SourceResult::Chunk(DataChunk::from_record_batch(
                            batch,
                        )));
                    }
                    lstate.chunks.set(batch)?;
                }
                None => {
                    drop(lstate.claimed.remove(index));
                    lstate.next = index;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt;

    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type};
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::DataFusionError;
    use datafusion_execution::TaskContext;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_physical_plan::execution_plan::Boundedness;

    use crate::executor::Executor;
    use crate::operators::lowering_tests::{assert_lowered, count};
    use crate::test_utils::{register_test_tables, test_session_config};

    /// A generator of `batches` batches of `rows` consecutive values of `v`
    /// (and `w = v * 2`), starting at `start`; unbounded if `batches` is
    /// `None`.
    #[derive(Debug, Clone)]
    struct Counter {
        start: i64,
        next: i64,
        rows: i64,
        batches: Option<i64>,
    }

    impl Counter {
        fn generator(start: i64, rows: i64, batches: Option<i64>) -> Generator {
            Arc::new(RwLock::new(Self {
                start,
                next: start,
                rows,
                batches,
            }))
        }

        fn schema() -> SchemaRef {
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Int64, false),
                Field::new("w", DataType::Int64, false),
            ]))
        }
    }

    impl fmt::Display for Counter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "counter: start={}", self.start)
        }
    }

    impl LazyBatchGenerator for Counter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn boundedness(&self) -> Boundedness {
            match self.batches {
                Some(_) => Boundedness::Bounded,
                None => Boundedness::Unbounded {
                    requires_infinite_memory: false,
                },
            }
        }

        fn generate_next_batch(&mut self) -> Result<Option<RecordBatch>> {
            let generated = (self.next - self.start) / self.rows.max(1);
            if self.batches.is_some_and(|batches| generated >= batches) {
                return Ok(None);
            }
            let values = self.next..self.next + self.rows;
            self.next += self.rows.max(1);
            let batch = RecordBatch::try_new(
                Self::schema(),
                vec![
                    Arc::new(Int64Array::from_iter_values(values.clone())),
                    Arc::new(Int64Array::from_iter_values(values.map(|v| v * 2))),
                ],
            )?;
            Ok(Some(batch))
        }

        fn reset_state(&self) -> Arc<RwLock<dyn LazyBatchGenerator>> {
            Arc::new(RwLock::new(Self {
                next: self.start,
                ..self.clone()
            }))
        }
    }

    fn values(batches: &[RecordBatch], column: usize) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(column)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn generators_are_read_by_the_threads_with_their_projection() -> Result<()> {
        let generators = vec![
            Counter::generator(0, 3000, Some(2)),
            Counter::generator(10_000, 100, Some(3)),
            Counter::generator(20_000, 0, Some(5)),
            Counter::generator(30_000, 1, Some(0)),
        ];
        let exec = LazyMemoryExec::try_new(Counter::schema(), generators)?
            .with_projection(Some(vec![1]));
        let source: Arc<dyn PhysicalOperator> = Arc::new(LazyMemorySource::new(&exec));
        assert_eq!(source.schema(), exec.schema());
        assert_eq!(
            source.description(),
            "LazyMemorySource: partitions=4, batch_generators=[counter: start=0, \
             counter: start=10000, counter: start=20000, counter: start=30000]"
        );
        let mut expected = (0..6000)
            .chain(10_000..10_300)
            .map(|v| v * 2)
            .collect::<Vec<_>>();
        expected.sort_unstable();
        for threads in [1, 3, 8] {
            // Every execution generates all batches again.
            for _ in 0..2 {
                let output = Executor::collect(
                    Arc::clone(&source),
                    Arc::new(TaskContext::default()),
                    threads,
                )
                .await?;
                assert!(output.iter().all(|batch| batch.schema() == exec.schema()));
                assert!(output.iter().all(|batch| batch.num_rows() <= 2048));
                let mut generated = values(&output, 0);
                generated.sort_unstable();
                assert_eq!(generated, expected, "{threads} threads");
            }
        }
        // A single generator is read in order.
        let exec = LazyMemoryExec::try_new(
            Counter::schema(),
            vec![Counter::generator(0, 1000, Some(9))],
        )?;
        let source = Arc::new(LazyMemorySource::new(&exec));
        let output =
            Executor::collect(source, Arc::new(TaskContext::default()), 4).await?;
        assert_eq!(values(&output, 0), (0..9000).collect::<Vec<_>>());
        assert_eq!(
            values(&output, 1),
            (0..9000).map(|v| v * 2).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn generators_that_never_end_do_not_hold_back_the_others() -> Result<()> {
        // Two generators that never end and one that does, read through a
        // limit on a single thread: the rows of all of them are emitted.
        let generators = vec![
            Counter::generator(0, 10, None),
            Counter::generator(1_000_000, 10, None),
            Counter::generator(-100, 10, Some(1)),
        ];
        let exec = LazyMemoryExec::try_new(Counter::schema(), generators)?;
        assert!(exec.properties().boundedness.is_unbounded());
        let source: Arc<dyn PhysicalOperator> = Arc::new(LazyMemorySource::new(&exec));
        let limit: Arc<dyn PhysicalOperator> = Arc::new(
            crate::operators::limit::PhysicalLimit::new(source, 0, Some(100)),
        );
        let output =
            Executor::collect(limit, Arc::new(TaskContext::default()), 1).await?;
        let generated = values(&output, 0);
        assert_eq!(generated.len(), 100);
        assert!(generated.iter().any(|v| *v < 0));
        assert!(generated.iter().any(|v| (0..1000).contains(v)));
        assert!(generated.iter().any(|v| *v >= 1_000_000));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_batches_of_a_thread_reserve_memory() -> Result<()> {
        let exec = LazyMemoryExec::try_new(
            Counter::schema(),
            vec![Counter::generator(0, 100_000, Some(1))],
        )?;
        let source = Arc::new(LazyMemorySource::new(&exec));
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

    #[test]
    fn projections_are_found_by_name() -> Result<()> {
        let schema = Counter::schema();
        assert_eq!(projection_of(&schema, &schema)?, None);
        assert_eq!(
            projection_of(&schema, &schema.project(&[1])?)?,
            Some(vec![1])
        );
        assert_eq!(
            projection_of(&schema, &schema.project(&[1, 0])?)?,
            Some(vec![1, 0])
        );
        assert_eq!(projection_of(&schema, &schema.project(&[])?)?, Some(vec![]));
        let other = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        assert!(projection_of(&schema, &other).is_err());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn generate_series_and_range_are_native() {
        for sql in [
            "SELECT * FROM generate_series(1, 10)",
            "SELECT * FROM generate_series(1, 10000, 3) ORDER BY value DESC",
            "SELECT value % 7 AS k, COUNT(*) AS n, SUM(value) AS s \
             FROM generate_series(1, 100000) GROUP BY value % 7",
            "SELECT COUNT(*) FROM generate_series(1, 50000)",
            "SELECT * FROM range(0, 20, 4)",
            "SELECT * FROM range(10, 0, -3) ORDER BY value",
            "SELECT * FROM generate_series(5, 1)",
            "SELECT * FROM generate_series(NULL, 10)",
            "SELECT t1.id, g.value FROM t1 JOIN generate_series(1, 100) AS g \
             ON t1.id = g.value",
            "SELECT * FROM generate_series(TIMESTAMP '2024-01-01', \
             TIMESTAMP '2024-01-03', INTERVAL '7 hours')",
            "SELECT * FROM generate_series(DATE '2024-01-01', DATE '2024-03-01', \
             INTERVAL '1 week')",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, async |ctx: &SessionContext| {
                    register_test_tables(ctx)
                })
                .await;
                assert!(count::<LazyMemorySource>(&root) > 0, "{sql}");
            }
        }
    }
}
