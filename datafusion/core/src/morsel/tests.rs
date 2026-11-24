// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;

use std::sync::Weak;
use std::time::Duration;

use arrow::array::Int32Array;
use arrow::datatypes::{Field, FieldRef, Schema};
use datafusion_common::cast::{as_int32_array, as_int64_array};
use datafusion_common::{ScalarValue, exec_err};
use datafusion_expr::logical_plan::{Filter, TableScanBuilder};
use datafusion_expr::{ColumnarValue, Volatility, create_udf, lit};
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::{Column, Literal};
use rstest::rstest;
use tokio::sync::Semaphore;
use tokio::task::Id;
use tokio::time::timeout;

use crate::datasource::provider_as_source;
use crate::execution::SessionStateBuilder;
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;

const INPUT_BATCHES: usize = 4 * BATCHES_PER_MORSEL + 1;
const WORKERS: usize = 2;
const DEADLINE: Duration = Duration::from_secs(10);
type Observations = mpsc::UnboundedReceiver<(i32, Option<Id>)>;

fn task_with_batch_size(size: usize) -> Arc<TaskContext> {
    let config = crate::prelude::SessionConfig::new().with_batch_size(size);
    Arc::new(TaskContext::default().with_session_config(config))
}

fn observed_pipeline(fail_on: Option<usize>) -> Result<(Pipeline, Observations)> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batches = (0..INPUT_BATCHES)
        .map(|i| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(vec![i as i32]))],
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let (observed, observations) = mpsc::unbounded_channel();
    let evaluated = AtomicUsize::new(0);
    let observe = create_udf(
        "observe",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(move |args| {
            let ColumnarValue::Array(ids) = &args[0] else {
                unreachable!()
            };
            let id = as_int32_array(ids.as_ref())?.value(0);
            let count = evaluated.fetch_add(1, Ordering::Relaxed) + 1;
            observed.send((id, tokio::task::try_id())).unwrap();
            if fail_on == Some(count) {
                return exec_err!("test morsel failure");
            }
            Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))))
        }),
    );
    let predicate = ScalarFunctionExpr::try_new(
        Arc::new(observe),
        vec![Arc::new(Column::new("id", 0))],
        &schema,
        Arc::default(),
    )?;
    Ok((
        Pipeline {
            source: Source::Memory(MemorySource {
                table: "test".to_owned(),
                batches,
                projection: None,
                schema,
            }),
            operators: vec![Operator::Filter {
                predicate: Arc::new(predicate),
                projection: None,
            }],
            workers: NonZeroUsize::new(WORKERS).unwrap(),
            sink: Sink::Result,
            sink_capacity: result_capacity(NonZeroUsize::new(WORKERS).unwrap()),
        },
        observations,
    ))
}

fn observed_plan(fail_on: Option<usize>) -> Result<(MorselExec, Observations)> {
    let (pipeline, observations) = observed_pipeline(fail_on)?;
    let schema = Arc::clone(pipeline.source.schema());
    Ok((MorselExec::new(pipeline, schema), observations))
}

fn inline_plan(fail_on: Option<usize>) -> Result<(MorselExec, Observations)> {
    let (mut pipeline, observations) = observed_pipeline(fail_on)?;
    pipeline.workers = NonZeroUsize::MIN;
    let schema = Arc::clone(pipeline.source.schema());
    Ok((MorselExec::new(pipeline, schema), observations))
}

#[derive(Debug)]
pub(super) struct FinalizeGate {
    entered: mpsc::UnboundedSender<()>,
    resume: Arc<Semaphore>,
}

impl FinalizeGate {
    pub(super) async fn wait(&self) {
        self.entered.send(()).unwrap();
        self.resume.acquire().await.unwrap().forget();
    }
}

struct ObservedCount {
    plan: MorselExec,
    scans: Observations,
    finalizing: mpsc::UnboundedReceiver<()>,
    resume: Arc<Semaphore>,
    results: mpsc::UnboundedReceiver<i64>,
}

fn observed_count(fail_on: Option<usize>) -> Result<ObservedCount> {
    let (mut pipeline, scans) = observed_pipeline(fail_on)?;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::Int64,
        false,
    )]));
    let (entered, finalizing) = mpsc::unbounded_channel();
    let resume = Arc::new(Semaphore::new(0));
    let mut sink = CountSink::new(Arc::clone(&schema));
    sink.finalize_gate = Some(FinalizeGate {
        entered,
        resume: Arc::clone(&resume),
    });
    let (started, results) = mpsc::unbounded_channel();
    let observe = create_udf(
        "observe_count",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Volatile,
        Arc::new(move |args| {
            let ColumnarValue::Array(count) = &args[0] else {
                unreachable!()
            };
            started.send(as_int64_array(count)?.value(0)).unwrap();
            Ok(args[0].clone())
        }),
    );
    let expr = ScalarFunctionExpr::try_new(
        Arc::new(observe),
        vec![Arc::new(Column::new("count", 0))],
        &schema,
        Arc::default(),
    )?;
    let projector = ProjectionExprs::new([ProjectionExpr::new(Arc::new(expr), "count")])
        .make_projector(&schema)?;
    let output_schema = Arc::clone(projector.output_schema());
    pipeline.sink = Sink::Count {
        sink,
        result_operators: vec![Operator::Projection(projector)],
    };
    Ok(ObservedCount {
        plan: MorselExec::new(pipeline, output_schema),
        scans,
        finalizing,
        resume,
        results,
    })
}

async fn observe_n(observations: &mut Observations, n: usize) -> Vec<(i32, Option<Id>)> {
    timeout(DEADLINE, async {
        let mut seen = Vec::new();
        for _ in 0..n {
            seen.push(observations.recv().await.unwrap());
        }
        seen
    })
    .await
    .expect("workers did not evaluate the expected number of batches")
}

async fn assert_workers_stopped(pipeline: &Weak<Pipeline>) {
    timeout(DEADLINE, async {
        while pipeline.strong_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pipeline still held by worker tasks");
}

#[test]
fn execute_does_not_need_a_runtime_before_poll() -> Result<()> {
    let (plan, mut observations) = observed_plan(None)?;
    let stream = plan.execute(0, Arc::default())?;
    drop(plan);
    assert!(matches!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    drop(stream);
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn single_worker_stream_is_lazy_and_does_not_run_ahead() -> Result<()> {
    let (plan, mut observations) = inline_plan(None)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, task_with_batch_size(1))?;
    drop(plan);
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    );

    let std::task::Poll::Ready(Some(batch)) = futures::poll!(stream.next()) else {
        panic!("the consumer's first poll should produce an in-memory batch inline");
    };
    assert_eq!(as_int32_array(batch?.column(0))?.values().as_ref(), &[0]);
    assert_eq!(observations.try_recv().unwrap().0, 0);
    tokio::task::yield_now().await;
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    );
    drop(stream);
    assert_eq!(pipeline.strong_count(), 0);
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn detailed_metrics_are_opt_in_and_isolated_on_reused_plans() -> Result<()> {
    let (plan, _scans) = inline_plan(None)?;
    let collector = Arc::new(MorselMetrics::default());
    let cfg = crate::prelude::SessionConfig::new()
        .with_batch_size(1)
        .with_extension(Arc::clone(&collector));
    let ctx = Arc::new(TaskContext::default().with_session_config(cfg));

    let mut cancelled = plan.execute(0, Arc::clone(&ctx))?;
    let mut complete = plan.execute(0, Arc::clone(&ctx))?;
    let never_polled = plan.execute(0, ctx)?;
    drop(never_polled);
    assert!(collector.take_runs().is_empty());
    assert_eq!(cancelled.next().await.unwrap()?.num_rows(), 1);
    assert_eq!(complete.next().await.unwrap()?.num_rows(), 1);
    drop(cancelled);
    let cancelled = collector.take_runs();
    assert_eq!(cancelled.len(), 1);
    let value = |report: &MetricsSet, name| {
        report
            .iter()
            .find(|m| m.value().name() == name)
            .unwrap()
            .value()
            .as_usize()
    };
    assert_eq!(value(&cancelled[0], "morsels_claimed"), 1);
    assert_eq!(value(&cancelled[0], "morsels_completed"), 0);
    assert_eq!(value(&cancelled[0], "morsel_peak_output_depth"), 0);
    assert_eq!(value(&cancelled[0], "morsel_output_wait"), 0);

    let mut rows = 1;
    while let Some(batch) = complete.next().await {
        rows += batch?.num_rows();
    }
    assert_eq!(rows, INPUT_BATCHES);
    let complete = collector.take_runs();
    assert_eq!(complete.len(), 1);
    let expected = INPUT_BATCHES.div_ceil(BATCHES_PER_MORSEL);
    assert_eq!(value(&complete[0], "morsels_claimed"), expected);
    assert_eq!(value(&complete[0], "morsels_completed"), expected);
    assert_eq!(value(&complete[0], "morsel_ready_wait"), 0);
    assert_eq!(value(&complete[0], "morsel_output_wait"), 0);
    assert_eq!(value(&complete[0], "morsel_decode_work"), 0);
    assert!(value(&complete[0], "morsel_process_work") > 0);

    let mut disabled = plan.execute(0, Arc::default())?;
    while let Some(batch) = disabled.next().await {
        batch?;
    }
    assert!(collector.take_runs().is_empty());
    assert!(
        plan.metrics()
            .unwrap()
            .sum_by_name("morsels_claimed")
            .is_none()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_worker_runs_report_separate_queue_and_work_totals() -> Result<()> {
    use datafusion_physical_plan::common::collect;

    let (plan, _scans) = observed_plan(None)?;
    let collector = Arc::new(MorselMetrics::default());
    let cfg = crate::prelude::SessionConfig::new()
        .with_batch_size(2)
        .with_extension(Arc::clone(&collector));
    let ctx = Arc::new(TaskContext::default().with_session_config(cfg));
    let first = plan.execute(0, Arc::clone(&ctx))?;
    let second = plan.execute(0, ctx)?;
    let (first, second) = timeout(DEADLINE, async {
        tokio::try_join!(collect(first), collect(second))
    })
    .await
    .expect("concurrent runs did not finish")?;
    for batches in [first, second] {
        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            INPUT_BATCHES
        );
    }
    let reports = collector.take_runs();
    assert_eq!(reports.len(), 2);
    for report in reports {
        let metric = |name| report.sum_by_name(name).unwrap().as_usize();
        let expected = INPUT_BATCHES.div_ceil(BATCHES_PER_MORSEL);
        assert_eq!(metric("morsels_claimed"), expected);
        assert_eq!(metric("morsels_completed"), expected);
        assert!(
            (1..=plan.pipeline.sink_capacity)
                .contains(&metric("morsel_peak_output_depth"))
        );
    }
    Ok(())
}

#[tokio::test]
async fn single_worker_stream_allows_other_query_progress() -> Result<()> {
    // Exercise both repeated Ready polls and an all-rejected scan whose first
    // call to next() would otherwise consume the whole input.
    for reject_all in [false, true] {
        let (mut long, mut scans) = observed_pipeline(None)?;
        long.workers = NonZeroUsize::MIN;
        if reject_all {
            long.operators.push(Operator::Filter {
                predicate: Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))),
                projection: None,
            });
        }
        let schema = Arc::clone(long.source.schema());
        let long = MorselExec::new(long, schema);
        let (short, _scans) = inline_plan(None)?;
        let mut long_stream =
            long.execute(0, task_with_batch_size(2 * BATCHES_PER_MORSEL))?;
        let mut short_stream = short.execute(0, task_with_batch_size(1))?;
        let (long_result, short_result) = timeout(DEADLINE, async {
            tokio::join!(
                biased;
                async {
                    let mut rows = 0;
                    while let Some(batch) = long_stream.next().await {
                        rows += batch?.num_rows();
                    }
                    Ok::<_, datafusion_common::DataFusionError>(rows)
                },
                async {
                    let batch = short_stream.next().await.unwrap()?;
                    let scanned = std::iter::from_fn(|| scans.try_recv().ok()).count();
                    assert_eq!(scanned, BATCHES_PER_MORSEL);
                    Ok::<_, datafusion_common::DataFusionError>(batch)
                },
            )
        })
        .await
        .expect("queries did not make progress");
        assert_eq!(long_result?, if reject_all { 0 } else { INPUT_BATCHES });
        assert_eq!(short_result?.num_rows(), 1);
    }
    Ok(())
}

#[rstest]
#[case::unmeasured(false)]
#[case::measured(true)]
#[tokio::test]
async fn single_worker_stream_error_terminates_without_reading_more_input(
    #[case] measure: bool,
) -> Result<()> {
    let (plan, mut observations) = inline_plan(Some(2))?;
    let collector = Arc::new(MorselMetrics::default());
    let ctx = if measure {
        let cfg = crate::prelude::SessionConfig::new()
            .with_batch_size(1)
            .with_extension(Arc::clone(&collector));
        Arc::new(TaskContext::default().with_session_config(cfg))
    } else {
        task_with_batch_size(1)
    };
    let mut stream = plan.execute(0, ctx)?;
    assert_eq!(stream.next().await.unwrap()?.num_rows(), 1);
    let error = stream.next().await.unwrap().unwrap_err();
    assert!(error.to_string().contains("test morsel failure"), "{error}");
    assert!(stream.next().await.is_none());
    assert_eq!(observations.try_recv().unwrap().0, 0);
    assert_eq!(observations.try_recv().unwrap().0, 1);
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    );
    let reports = collector.take_runs();
    assert_eq!(reports.len(), usize::from(measure));
    if let [report] = reports.as_slice() {
        let count = |name| report.sum_by_name(name).unwrap().as_usize();
        assert_eq!(count("morsels_claimed"), 1);
        assert_eq!(count("morsels_completed"), 0);
    }
    Ok(())
}

#[test]
fn result_channel_capacity_is_capped() {
    assert_eq!(result_capacity(NonZeroUsize::MIN), 2);
    assert_eq!(result_capacity(NonZeroUsize::new(8).unwrap()), 16);
    assert_eq!(result_capacity(NonZeroUsize::MAX), MAX_RESULT_CAPACITY);
}

#[test]
fn large_filtered_batch_bypasses_the_coalescer() -> Result<()> {
    let batch = RecordBatch::try_from_iter([(
        "id",
        Arc::new(Int32Array::from_iter_values(0..6)) as _,
    )])?;
    let operators = [Operator::Filter {
        predicate: Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
        projection: None,
    }];
    let mut buffer = ResultBuffer::new(&operators, 10, false);
    buffer.push_from(&operators, 0, batch.clone())?;
    let output = buffer.next(&operators, false)?.unwrap();
    assert!(Arc::ptr_eq(batch.column(0), output.column(0)));
    assert!(buffer.next(&operators, true)?.is_none());
    Ok(())
}

#[tokio::test]
async fn dropping_inline_stream_discards_partial_batch() -> Result<()> {
    let (plan, mut observations) = inline_plan(None)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, task_with_batch_size(2 * BATCHES_PER_MORSEL))?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    observe_n(&mut observations, BATCHES_PER_MORSEL).await;
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    );
    drop(stream);
    assert_eq!(pipeline.strong_count(), 0);
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn workers_coalesce_disjoint_ranges_and_backpressure_bounds_the_queue() -> Result<()>
{
    let (plan, mut observations) = observed_plan(None)?;
    let target = 2;
    let capacity = plan.pipeline.sink_capacity;
    let mut run =
        plan.pipeline
            .start(task_with_batch_size(target), plan.metrics.clone(), None);
    // Each output consumes two single-row inputs. Without consumption, workers
    // fill the channel and then each park at their next send, without reading ahead.
    let mut seen = observe_n(&mut observations, (capacity + WORKERS) * target).await;
    assert_eq!(run.rx.len(), capacity);
    assert_eq!(run.rx.capacity(), 0);
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    );

    let mut output = Vec::new();
    timeout(DEADLINE, async {
        while let Some(batch) = run.next().await? {
            output.push(as_int32_array(batch.column(0))?.values().to_vec());
        }
        Ok::<_, datafusion_common::DataFusionError>(())
    })
    .await
    .expect("pipeline did not drain")?;
    assert_eq!(output.len(), INPUT_BATCHES.div_ceil(target));
    assert!(
        output[..output.len() - 1]
            .iter()
            .all(|ids| ids.len() == target)
    );
    assert_eq!(output.last().unwrap().len(), 1);
    let mut rows: Vec<_> = output.iter().flatten().copied().collect();
    rows.sort_unstable();
    assert_eq!(rows, (0..INPUT_BATCHES as i32).collect::<Vec<_>>());

    while let Ok(claim) = observations.try_recv() {
        seen.push(claim);
    }
    seen.sort_unstable_by_key(|&(id, _)| id);
    assert_eq!(
        seen.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
        (0..INPUT_BATCHES as i32).collect::<Vec<_>>()
    );
    assert_ne!(seen[0].1, seen[BATCHES_PER_MORSEL].1);
    for range in seen.chunks(BATCHES_PER_MORSEL) {
        assert!(range.iter().all(|&(_, worker)| worker == range[0].1));
    }
    for batch in output {
        assert!(
            batch
                .iter()
                .all(|&id| seen[id as usize].1 == seen[batch[0] as usize].1)
        );
    }
    Ok(())
}

#[tokio::test]
async fn dropping_stream_cancels_workers_parked_on_the_full_sink() -> Result<()> {
    let (plan, mut observations) = observed_plan(None)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let capacity = plan.pipeline.sink_capacity;
    let mut stream = plan.execute(0, task_with_batch_size(2))?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    observe_n(&mut observations, (capacity + WORKERS) * 2).await;
    drop(stream);
    assert_workers_stopped(&pipeline).await;
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn worker_error_cancels_peers_without_polling_the_full_sink() -> Result<()> {
    let workers = NonZeroUsize::new(WORKERS).unwrap();
    let fail_on = (result_capacity(workers) + WORKERS) * 2;
    let (plan, mut observations) = observed_plan(Some(fail_on))?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, task_with_batch_size(2))?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    observe_n(&mut observations, fail_on).await;
    assert_workers_stopped(&pipeline).await;
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    let error = stream.next().await.unwrap().unwrap_err();
    assert!(error.to_string().contains("test morsel failure"), "{error}");
    assert!(stream.next().await.is_none());
    Ok(())
}

#[tokio::test]
async fn worker_error_discards_other_workers_eof_tails() -> Result<()> {
    let (mut pipeline, mut observations) = observed_pipeline(None)?;
    let schema = Arc::clone(pipeline.source.schema());
    let projected = Arc::new(AtomicUsize::new(0));
    let observe = create_udf(
        "observe_eof_tail",
        vec![DataType::Int32],
        DataType::Int32,
        Volatility::Volatile,
        Arc::new({
            let projected = Arc::clone(&projected);
            move |args| {
                let ColumnarValue::Array(ids) = &args[0] else {
                    unreachable!()
                };
                assert!(ids.len() >= BATCHES_PER_MORSEL);
                if projected.fetch_add(1, Ordering::Relaxed) > 0 {
                    return exec_err!("test EOF projection failure");
                }
                Ok(args[0].clone())
            }
        }),
    );
    let expr = ScalarFunctionExpr::try_new(
        Arc::new(observe),
        vec![Arc::new(Column::new("id", 0))],
        &schema,
        Arc::default(),
    )?;
    let projector = ProjectionExprs::new([ProjectionExpr::new(Arc::new(expr), "id")])
        .make_projector(&schema)?;
    pipeline.operators.push(Operator::Projection(projector));
    let pipeline = Arc::new(pipeline);
    let weak = Arc::downgrade(&pipeline);
    // Every worker has fewer rows than this target, so projection runs only at EOF.
    let mut run = pipeline.start(
        task_with_batch_size(2 * INPUT_BATCHES),
        Default::default(),
        None,
    );
    observe_n(&mut observations, INPUT_BATCHES).await;
    assert_workers_stopped(&weak).await;
    assert_eq!(projected.load(Ordering::Relaxed), WORKERS);
    assert!(
        run.rx.is_empty(),
        "an EOF tail was queued despite a worker error"
    );
    let error = run.next().await.unwrap_err().to_string();
    assert!(error.contains("test EOF projection failure"), "{error}");
    Ok(())
}

#[cfg(feature = "sql")]
#[tokio::test]
async fn memory_count_mask_is_selected_for_lowered_sql() -> Result<()> {
    let ctx = crate::prelude::SessionContext::new();
    ctx.register_batch(
        "t",
        RecordBatch::try_from_iter([(
            "id",
            Arc::new(Int32Array::from(vec![1, -1])) as _,
        )])?,
    )?;
    let logical = ctx
        .sql("SELECT COUNT(*) FROM t WHERE id > 0")
        .await?
        .into_optimized_plan()?;
    let planner = MorselQueryPlanner::new(NonZeroUsize::MIN);
    let Lowering::Native(mut plan) = planner.try_lower(&logical, &ctx.state()).await?
    else {
        panic!("expected native count plan for {logical:?}");
    };
    let pipeline = Arc::get_mut(&mut plan.pipeline).unwrap();
    assert!(matches!(pipeline.sink, Sink::Count { .. }));
    let [Operator::Filter { .. }, Operator::Projection(projector)] =
        pipeline.operators.as_slice()
    else {
        panic!(
            "expected Filter followed by Projection: {:?}",
            pipeline.operators
        );
    };
    assert!(projector.projection().as_ref().is_empty());
    assert!(pipeline.memory_count_filter().is_some());

    pipeline.operators.insert(
        1,
        Operator::Filter {
            predicate: Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))),
            projection: None,
        },
    );
    assert!(pipeline.memory_count_filter().is_none());
    Ok(())
}

// ScalarFunctionExpr rejects the wrong length before the runtime sees the mask.
#[derive(Debug, PartialEq, Eq, Hash)]
struct OversizedMask;

impl fmt::Display for OversizedMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OversizedMask")
    }
}

impl PhysicalExpr for OversizedMask {
    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }

    fn return_field(&self, _schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::new(Field::new("mask", DataType::Boolean, false)))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(vec![
            true;
            batch.num_rows() + 1
        ]))))
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

fn oversized_mask_pipeline(projection: Option<Vec<usize>>) -> Result<Pipeline> {
    let (mut pipeline, _observations) = observed_pipeline(None)?;
    pipeline.workers = NonZeroUsize::MIN;
    pipeline.operators = vec![Operator::Filter {
        predicate: Arc::new(OversizedMask),
        projection,
    }];
    Ok(pipeline)
}

async fn assert_oversized_mask_error(
    pipeline: Pipeline,
    schema: SchemaRef,
) -> Result<()> {
    let plan = MorselExec::new(pipeline, schema);
    let mut stream = plan.execute(0, Arc::default())?;
    let error = timeout(DEADLINE, stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(
        error.to_string().contains(
            "Filter predicate of length 2 is larger than target array of length 1"
        ),
        "{error}"
    );
    assert!(timeout(DEADLINE, stream.next()).await.unwrap().is_none());
    Ok(())
}

#[tokio::test]
async fn projected_filter_rejects_oversized_physical_mask() -> Result<()> {
    let mut pipeline = oversized_mask_pipeline(Some(vec![]))?;
    let projector = ProjectionExprs::new([ProjectionExpr::new(
        Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
        "constant",
    )])
    .make_projector(&Schema::empty())?;
    let schema = Arc::clone(projector.output_schema());
    pipeline.operators.push(Operator::Projection(projector));
    assert_oversized_mask_error(pipeline, schema).await
}

#[tokio::test]
async fn count_mask_rejects_oversized_physical_mask() -> Result<()> {
    let mut pipeline = oversized_mask_pipeline(None)?;
    assert!(pipeline.memory_count_filter().is_some());
    let schema = Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::Int64,
        false,
    )]));
    pipeline.sink = Sink::Count {
        sink: CountSink::new(Arc::clone(&schema)),
        result_operators: vec![],
    };
    assert_oversized_mask_error(pipeline, schema).await
}

#[tokio::test]
async fn count_result_waits_for_finalization_after_all_workers_finish() -> Result<()> {
    let ObservedCount {
        plan,
        mut scans,
        mut finalizing,
        resume,
        mut results,
    } = observed_count(None)?;
    let mut stream = plan.execute(0, Arc::default())?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    assert_eq!(
        timeout(DEADLINE, finalizing.recv()).await.unwrap(),
        Some(())
    );
    let seen = observe_n(&mut scans, INPUT_BATCHES).await;
    assert!(seen.iter().any(|&(_, worker)| worker != seen[0].1));
    assert_eq!(results.try_recv(), Err(mpsc::error::TryRecvError::Empty));
    assert!(futures::poll!(stream.next()).is_pending());

    resume.add_permits(1);
    let batch = timeout(DEADLINE, stream.next()).await.unwrap().unwrap()?;
    assert_eq!(
        as_int64_array(batch.column(0))?.values().as_ref(),
        &[INPUT_BATCHES as i64]
    );
    assert_eq!(results.try_recv(), Ok(INPUT_BATCHES as i64));
    assert!(timeout(DEADLINE, stream.next()).await.unwrap().is_none());
    assert_eq!(
        finalizing.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    assert_eq!(
        results.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn dropping_count_before_finalization_never_starts_result() -> Result<()> {
    let ObservedCount {
        plan,
        mut finalizing,
        mut results,
        scans: _scans,
        ..
    } = observed_count(None)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, Arc::default())?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    assert_eq!(
        timeout(DEADLINE, finalizing.recv()).await.unwrap(),
        Some(())
    );
    drop(stream);
    assert_workers_stopped(&pipeline).await;
    assert_eq!(
        finalizing.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    assert_eq!(
        results.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn count_worker_error_skips_finalization_and_result() -> Result<()> {
    let ObservedCount {
        plan,
        mut finalizing,
        mut results,
        scans: _scans,
        ..
    } = observed_count(Some(3))?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, Arc::default())?;
    drop(plan);
    let error = timeout(DEADLINE, stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("test morsel failure"), "{error}");
    assert!(timeout(DEADLINE, stream.next()).await.unwrap().is_none());
    assert_workers_stopped(&pipeline).await;
    assert_eq!(
        finalizing.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    assert_eq!(
        results.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn admitted_plan_reports_lowering_errors() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let table = Arc::new(MemTable::try_new(schema, vec![vec![]])?);
    let scan = TableScanBuilder::new("items", provider_as_source(table)).build()?;
    let mut filter = Filter::try_new(lit(true), Arc::new(LogicalPlan::TableScan(scan)))?;
    filter.predicate = lit(42);
    let planner = MorselQueryPlanner::new(NonZeroUsize::MIN);
    let mut state = SessionStateBuilder::new_with_default_features().build();
    for mode in [MorselMode::Off, MorselMode::Prefer, MorselMode::Required] {
        state.config_mut().options_mut().execution.morsel_mode = mode;
        let planner: &dyn QueryPlanner = if mode == MorselMode::Off {
            &planner
        } else {
            state.query_planner().as_ref()
        };
        let error = planner
            .create_physical_plan(&LogicalPlan::Filter(filter.clone()), &state)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Morsel filter predicate must return Boolean"),
            "{error}"
        );
    }
    Ok(())
}

struct ObservedGroupCount {
    plan: MorselExec,
    scans: Observations,
    finalizing: mpsc::UnboundedReceiver<()>,
    resume: Arc<Semaphore>,
    results: mpsc::UnboundedReceiver<usize>,
}

fn task_with_pool_and_batch_size(
    pool: Arc<dyn MemoryPool>,
    batch_size: usize,
) -> Result<Arc<TaskContext>> {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .build_arc()?;
    let config = crate::prelude::SessionConfig::new().with_batch_size(batch_size);
    Ok(Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime),
    ))
}

fn observed_group_count(fail_on: Option<usize>) -> Result<ObservedGroupCount> {
    let (mut pipeline, scans) = observed_pipeline(fail_on)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("count", DataType::Int64, false),
    ]));
    let (entered, finalizing) = mpsc::unbounded_channel();
    let resume = Arc::new(Semaphore::new(0));
    let mut sink =
        GroupCountSink::new(Arc::clone(&schema), 0, IntKeyType::Int32, "id".into());
    sink.finalize_gate = Some(FinalizeGate {
        entered,
        resume: Arc::clone(&resume),
    });
    let (started, results) = mpsc::unbounded_channel();
    let observe = create_udf(
        "observe_group_count",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Volatile,
        Arc::new(move |args| {
            let ColumnarValue::Array(counts) = &args[0] else {
                unreachable!()
            };
            started.send(counts.len()).unwrap();
            Ok(args[0].clone())
        }),
    );
    let expr = ScalarFunctionExpr::try_new(
        Arc::new(observe),
        vec![Arc::new(Column::new("count", 1))],
        &schema,
        Arc::default(),
    )?;
    let projector = ProjectionExprs::new([
        ProjectionExpr::new(Arc::new(Column::new("id", 0)), "id"),
        ProjectionExpr::new(Arc::new(expr), "count"),
    ])
    .make_projector(&schema)?;
    let output_schema = Arc::clone(projector.output_schema());
    pipeline.sink = Sink::GroupCount {
        sink,
        result_operators: vec![Operator::Projection(projector)],
    };
    pipeline.sink_capacity = GROUP_COUNT_RESULT_CAPACITY;
    Ok(ObservedGroupCount {
        plan: MorselExec::new(pipeline, output_schema),
        scans,
        finalizing,
        resume,
        results,
    })
}

#[tokio::test]
async fn group_count_result_waits_for_finalization_emits_bounded_batches_and_cleans_up()
-> Result<()> {
    let ObservedGroupCount {
        plan,
        mut scans,
        mut finalizing,
        resume,
        mut results,
    } = observed_group_count(None)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
    let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 4)?;
    let mut stream = plan.execute(0, ctx)?;
    assert!(futures::poll!(stream.next()).is_pending());
    assert_eq!(
        timeout(DEADLINE, finalizing.recv()).await.unwrap(),
        Some(())
    );
    let seen = observe_n(&mut scans, INPUT_BATCHES).await;
    assert!(seen.iter().any(|&(_, worker)| worker != seen[0].1));
    assert_eq!(results.try_recv(), Err(mpsc::error::TryRecvError::Empty));
    assert!(
        pool.reserved() > 0,
        "expected combined group table memory reservation while parked before finalize"
    );
    assert!(futures::poll!(stream.next()).is_pending());

    resume.add_permits(1);
    let mut keys = Vec::new();
    let mut batch_sizes = Vec::new();
    while let Some(batch) = timeout(DEADLINE, stream.next()).await.unwrap() {
        let batch = batch?;
        assert!(batch.num_rows() <= 4);
        batch_sizes.push(batch.num_rows());
        let ids = as_int32_array(batch.column(0))?;
        let counts = as_int64_array(batch.column(1))?;
        assert!(counts.values().iter().all(|&c| c == 1));
        keys.extend(ids.values().iter().copied());
    }
    keys.sort_unstable();
    assert_eq!(keys, (0..INPUT_BATCHES as i32).collect::<Vec<_>>());
    assert_eq!(batch_sizes.len(), INPUT_BATCHES.div_ceil(4));
    assert_eq!(
        plan.metrics()
            .unwrap()
            .sum_by_name("group_count_finalizations")
            .unwrap()
            .as_usize(),
        1
    );
    drop(plan);
    assert_eq!(pool.reserved(), 0);
    let mut observed_result_rows = 0;
    while let Ok(rows) = results.try_recv() {
        observed_result_rows += rows;
    }
    assert_eq!(observed_result_rows, INPUT_BATCHES);
    assert_eq!(
        finalizing.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn dropping_group_count_before_finalization_never_starts_result_and_frees_memory()
-> Result<()> {
    let ObservedGroupCount {
        plan,
        mut finalizing,
        mut results,
        scans: _scans,
        ..
    } = observed_group_count(None)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
    let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 4)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, ctx)?;
    assert!(futures::poll!(stream.next()).is_pending());
    assert_eq!(
        timeout(DEADLINE, finalizing.recv()).await.unwrap(),
        Some(())
    );
    assert!(pool.reserved() > 0);
    drop(stream);
    assert_eq!(
        plan.metrics()
            .and_then(|m| m.sum_by_name("group_count_finalizations"))
            .map_or(0, |v| v.as_usize()),
        0
    );
    drop(plan);
    assert_workers_stopped(&pipeline).await;
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        finalizing.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    assert_eq!(
        results.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn dropping_group_count_mid_emission_frees_all_memory() -> Result<()> {
    let ObservedGroupCount {
        plan,
        mut finalizing,
        resume,
        scans: _scans,
        results: _results,
    } = observed_group_count(None)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
    let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 2)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, ctx)?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    assert_eq!(
        timeout(DEADLINE, finalizing.recv()).await.unwrap(),
        Some(())
    );
    resume.add_permits(1);
    let first = timeout(DEADLINE, stream.next()).await.unwrap().unwrap()?;
    assert_eq!(first.num_rows(), 2);
    drop(stream);
    assert_workers_stopped(&pipeline).await;
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn group_count_worker_error_skips_finalization_and_frees_memory() -> Result<()> {
    let ObservedGroupCount {
        plan,
        mut finalizing,
        mut results,
        scans: _scans,
        ..
    } = observed_group_count(Some(3))?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
    let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 4)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, ctx)?;
    let error = timeout(DEADLINE, stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("test morsel failure"), "{error}");
    assert!(timeout(DEADLINE, stream.next()).await.unwrap().is_none());
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        plan.metrics()
            .and_then(|m| m.sum_by_name("group_count_finalizations"))
            .map_or(0, |v| v.as_usize()),
        0
    );
    drop(stream);
    drop(plan);
    assert_workers_stopped(&pipeline).await;
    assert_eq!(
        finalizing.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    assert_eq!(
        results.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    );
    Ok(())
}

#[tokio::test]
async fn group_count_mask_rejects_oversized_physical_mask() -> Result<()> {
    let mut pipeline = oversized_mask_pipeline(None)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("count", DataType::Int64, false),
    ]));
    pipeline.sink = Sink::GroupCount {
        sink: GroupCountSink::new(Arc::clone(&schema), 0, IntKeyType::Int32, "id".into()),
        result_operators: vec![],
    };
    assert!(matches!(
        pipeline.group_count_strategy(0),
        GroupCountStrategy::DirectMask { key_index: 0, .. }
    ));
    assert_oversized_mask_error(pipeline, schema).await
}

#[cfg(feature = "sql")]
#[tokio::test]
async fn group_count_strategy_selection_for_lowered_sql() -> Result<()> {
    let ctx = crate::prelude::SessionContext::new();
    ctx.register_batch(
        "t",
        RecordBatch::try_from_iter([
            ("k", Arc::new(Int32Array::from(vec![1, 2])) as _),
            ("v", Arc::new(Int32Array::from(vec![10, -5])) as _),
        ])?,
    )?;
    let planner = MorselQueryPlanner::new(NonZeroUsize::new(2).unwrap());

    let unfiltered = ctx
        .sql("SELECT k, COUNT(*) FROM t GROUP BY k")
        .await?
        .into_optimized_plan()?;
    let Lowering::Native(plan) = planner.try_lower(&unfiltered, &ctx.state()).await?
    else {
        panic!("expected native group count plan");
    };
    assert!(matches!(
        plan.pipeline.group_count_strategy(0),
        GroupCountStrategy::DirectColumn(0)
    ));

    let filtered = ctx
        .sql("SELECT k, COUNT(*) FROM t WHERE v > 0 GROUP BY k")
        .await?
        .into_optimized_plan()?;
    let Lowering::Native(mut plan) = planner.try_lower(&filtered, &ctx.state()).await?
    else {
        panic!("expected native filtered group count plan");
    };
    assert!(matches!(
        plan.pipeline.group_count_strategy(0),
        GroupCountStrategy::DirectMask { key_index: 0, .. }
    ));
    let pipeline = Arc::get_mut(&mut plan.pipeline).unwrap();
    pipeline.operators.insert(
        1,
        Operator::Filter {
            predicate: Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
            projection: None,
        },
    );
    assert!(matches!(
        pipeline.group_count_strategy(0),
        GroupCountStrategy::General(0)
    ));
    Ok(())
}

#[cfg(feature = "sql")]
struct ObservedAggregate {
    plan: MorselExec,
    scans: Observations,
    finalizing: mpsc::UnboundedReceiver<()>,
    resume: Arc<Semaphore>,
    results: mpsc::UnboundedReceiver<usize>,
}

#[cfg(feature = "sql")]
async fn observed_aggregate(
    grouped: bool,
    fail_on: Option<usize>,
) -> Result<ObservedAggregate> {
    let (mut pipeline, scans) = observed_pipeline(fail_on)?;
    let input_schema = Arc::clone(pipeline.source.schema());
    let ctx = crate::prelude::SessionContext::new();
    let mem = Arc::new(MemTable::try_new(Arc::clone(&input_schema), vec![vec![]])?);
    ctx.register_table("t", mem)?;
    let sql = if grouped {
        "SELECT id, SUM(id) AS s FROM t GROUP BY id"
    } else {
        "SELECT SUM(id) AS s, MIN(id) AS lo, MAX(id) AS hi FROM t"
    };
    let logical = ctx.sql(sql).await?.into_optimized_plan()?;
    let agg_node = match &logical {
        LogicalPlan::Projection(p) => match p.input.as_ref() {
            LogicalPlan::Aggregate(a) => a,
            other => panic!("unexpected plan: {other:?}"),
        },
        LogicalPlan::Aggregate(a) => a,
        other => panic!("unexpected plan: {other:?}"),
    };
    let mut sink = AggregateSink::try_new(agg_node, &input_schema, &ctx.state())?
        .expect("valid aggregate sink");
    let agg_schema = Arc::clone(sink.schema());
    let (entered, finalizing) = mpsc::unbounded_channel();
    let resume = Arc::new(Semaphore::new(0));
    sink.finalize_gate = Some(FinalizeGate {
        entered,
        resume: Arc::clone(&resume),
    });

    let sum_col_index = usize::from(grouped);
    let (started, results) = mpsc::unbounded_channel();
    let observe = create_udf(
        "observe_agg",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Volatile,
        Arc::new(move |args| {
            let ColumnarValue::Array(sums) = &args[0] else {
                unreachable!()
            };
            started.send(sums.len()).unwrap();
            Ok(ColumnarValue::Array(Arc::clone(sums)))
        }),
    );
    let observe_expr = Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(observe),
        vec![Arc::new(Column::new(
            agg_schema.field(sum_col_index).name(),
            sum_col_index,
        ))],
        &agg_schema,
        Arc::default(),
    )?);
    let projector = ProjectionExprs::new([ProjectionExpr::new(observe_expr, "s")])
        .make_projector(&agg_schema)?;
    let result_schema = Arc::clone(projector.output_schema());
    pipeline.sink = Sink::Aggregate {
        sink,
        result_operators: vec![Operator::Projection(projector)],
    };
    pipeline.sink_capacity = AGGREGATE_RESULT_CAPACITY;
    Ok(ObservedAggregate {
        plan: MorselExec::new(pipeline, result_schema),
        scans,
        finalizing,
        resume,
        results,
    })
}

#[cfg(feature = "sql")]
#[tokio::test]
async fn aggregate_result_waits_for_finalization_emits_bounded_batches_and_cleans_up()
-> Result<()> {
    for grouped in [false, true] {
        let ObservedAggregate {
            plan,
            mut scans,
            mut finalizing,
            resume,
            mut results,
        } = observed_aggregate(grouped, None).await?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 4)?;
        let pipeline = Arc::downgrade(&plan.pipeline);
        let mut stream = plan.execute(0, ctx)?;
        assert!(futures::poll!(stream.next()).is_pending());
        assert_eq!(
            timeout(DEADLINE, finalizing.recv()).await.unwrap(),
            Some(())
        );
        let seen = observe_n(&mut scans, INPUT_BATCHES).await;
        assert_eq!(seen.len(), INPUT_BATCHES);
        assert_eq!(
            results.try_recv(),
            Err(mpsc::error::TryRecvError::Empty),
            "grouped={grouped}: result pipeline started before finalization"
        );
        if grouped {
            assert!(pool.reserved() > 0);
        }

        resume.add_permits(1);
        let mut total_rows = 0;
        let mut total_sum = 0i64;
        let mut batches = 0;
        while let Some(batch) = timeout(DEADLINE, stream.next()).await.unwrap() {
            let batch = batch?;
            assert!(batch.num_rows() <= 4);
            total_rows += batch.num_rows();
            total_sum += as_int64_array(batch.column(0))?
                .values()
                .iter()
                .sum::<i64>();
            batches += 1;
        }
        let expected_rows = if grouped { INPUT_BATCHES } else { 1 };
        let expected_sum: i64 = (0..INPUT_BATCHES as i64).sum();
        assert_eq!(total_rows, expected_rows, "grouped={grouped}");
        assert_eq!(total_sum, expected_sum, "grouped={grouped}");
        if grouped {
            assert_eq!(batches, INPUT_BATCHES.div_ceil(4));
        } else {
            assert_eq!(batches, 1);
        }
        assert_eq!(
            plan.metrics()
                .and_then(|m| m.sum_by_name("aggregate_finalizations"))
                .map_or(0, |v| v.as_usize()),
            1
        );
        drop(stream);
        drop(plan);
        assert_workers_stopped(&pipeline).await;
        assert_eq!(pool.reserved(), 0);
    }
    Ok(())
}

#[cfg(feature = "sql")]
#[tokio::test]
async fn dropping_aggregate_before_finalization_never_starts_result_and_frees_memory()
-> Result<()> {
    for grouped in [false, true] {
        let ObservedAggregate {
            plan,
            mut finalizing,
            mut results,
            scans: _scans,
            ..
        } = observed_aggregate(grouped, None).await?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 4)?;
        let pipeline = Arc::downgrade(&plan.pipeline);
        let mut stream = plan.execute(0, ctx)?;
        assert!(futures::poll!(stream.next()).is_pending());
        assert_eq!(
            timeout(DEADLINE, finalizing.recv()).await.unwrap(),
            Some(())
        );
        drop(stream);
        assert_eq!(
            plan.metrics()
                .and_then(|m| m.sum_by_name("aggregate_finalizations"))
                .map_or(0, |v| v.as_usize()),
            0
        );
        drop(plan);
        assert_workers_stopped(&pipeline).await;
        assert_eq!(pool.reserved(), 0);
        assert_eq!(
            finalizing.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        );
        assert_eq!(
            results.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        );
    }
    Ok(())
}

#[cfg(feature = "sql")]
#[tokio::test]
async fn dropping_grouped_aggregate_mid_emission_frees_all_memory() -> Result<()> {
    let ObservedAggregate {
        plan,
        mut finalizing,
        resume,
        scans: _scans,
        results: _results,
    } = observed_aggregate(true, None).await?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
    let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 2)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, ctx)?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    assert_eq!(
        timeout(DEADLINE, finalizing.recv()).await.unwrap(),
        Some(())
    );
    resume.add_permits(1);
    let first = timeout(DEADLINE, stream.next()).await.unwrap().unwrap()?;
    assert_eq!(first.num_rows(), 2);
    drop(stream);
    assert_workers_stopped(&pipeline).await;
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[cfg(feature = "sql")]
#[tokio::test]
async fn aggregate_worker_error_skips_finalization_and_frees_memory() -> Result<()> {
    for grouped in [false, true] {
        let ObservedAggregate {
            plan,
            mut finalizing,
            mut results,
            scans: _scans,
            ..
        } = observed_aggregate(grouped, Some(3)).await?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let ctx = task_with_pool_and_batch_size(Arc::clone(&pool), 4)?;
        let pipeline = Arc::downgrade(&plan.pipeline);
        let mut stream = plan.execute(0, ctx)?;
        let error = timeout(DEADLINE, stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("test morsel failure"), "{error}");
        assert!(timeout(DEADLINE, stream.next()).await.unwrap().is_none());
        assert_eq!(pool.reserved(), 0);
        assert_eq!(
            plan.metrics()
                .and_then(|m| m.sum_by_name("aggregate_finalizations"))
                .map_or(0, |v| v.as_usize()),
            0
        );
        drop(stream);
        drop(plan);
        assert_workers_stopped(&pipeline).await;
        assert_eq!(
            finalizing.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        );
        assert_eq!(
            results.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        );
    }
    Ok(())
}
