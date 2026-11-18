// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;

use std::sync::Weak;
use std::time::Duration;

use arrow::array::Int32Array;
use arrow::datatypes::{Field, Schema};
use datafusion_common::cast::{as_int32_array, as_int64_array};
use datafusion_common::{ScalarValue, exec_err};
use datafusion_expr::logical_plan::{Filter, TableScanBuilder};
use datafusion_expr::{ColumnarValue, Volatility, create_udf, lit};
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::Column;
use tokio::sync::Semaphore;
use tokio::task::Id;
use tokio::time::timeout;

use crate::datasource::provider_as_source;
use crate::execution::SessionStateBuilder;

const INPUT_BATCHES: usize = 4 * BATCHES_PER_MORSEL + 1;
const WORKERS: usize = 2;
const DEADLINE: Duration = Duration::from_secs(10);
type Observations = mpsc::UnboundedReceiver<(i32, Id)>;

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
            observed.send((id, tokio::task::id())).unwrap();
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
            operators: vec![Operator::Filter(Arc::new(predicate))],
            workers: NonZeroUsize::new(WORKERS).unwrap(),
            sink: Sink::Result,
            sink_capacity: RESULT_CAPACITY,
        },
        observations,
    ))
}

fn observed_plan(fail_on: Option<usize>) -> Result<(MorselExec, Observations)> {
    let (pipeline, observations) = observed_pipeline(fail_on)?;
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

async fn observe_n(observations: &mut Observations, n: usize) -> Vec<(i32, Id)> {
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
async fn workers_claim_disjoint_ranges_and_backpressure_bounds_the_queue() -> Result<()> {
    let (plan, mut observations) = observed_plan(None)?;
    let mut run = plan.pipeline.start(Arc::default(), plan.metrics.clone());
    // Without consumption, the only evaluated batches can be the queued ones
    // and the one parked at each worker's send.
    let mut seen = observe_n(&mut observations, RESULT_CAPACITY + WORKERS).await;
    assert_eq!(run.rx.len(), RESULT_CAPACITY);
    assert_eq!(run.rx.capacity(), 0);
    assert_eq!(
        observations.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    );

    let mut output = [0; INPUT_BATCHES];
    timeout(DEADLINE, async {
        while let Some(batch) = run.next().await? {
            let id = as_int32_array(batch.column(0))?.value(0) as usize;
            output[id] += 1;
        }
        Ok::<_, datafusion_common::DataFusionError>(())
    })
    .await
    .expect("pipeline did not drain")?;
    assert_eq!(output, [1; INPUT_BATCHES]);

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
    Ok(())
}

#[tokio::test]
async fn dropping_stream_cancels_workers_parked_on_the_full_sink() -> Result<()> {
    let (plan, mut observations) = observed_plan(None)?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, Arc::default())?;
    drop(plan);
    assert!(futures::poll!(stream.next()).is_pending());
    observe_n(&mut observations, RESULT_CAPACITY + WORKERS).await;
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
    let fail_on = RESULT_CAPACITY + WORKERS;
    let (plan, mut observations) = observed_plan(Some(fail_on))?;
    let pipeline = Arc::downgrade(&plan.pipeline);
    let mut stream = plan.execute(0, Arc::default())?;
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
