// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::VecDeque;
use std::future::poll_fn;
use std::pin::pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use arrow::array::{ArrayData, RecordBatch};
use datafusion_common::instant::Instant;
#[cfg(feature = "parquet")]
use datafusion_physical_plan::metrics::Time;
use datafusion_physical_plan::metrics::{
    ExecutionPlanMetricsSet, Label, MetricBuilder, MetricsSet,
};
use parking_lot::Mutex;
use tokio::sync::mpsc;

/// Opt-in collector for detailed metrics from the experimental morsel runtime.
/// Attach it as a [`SessionConfig`](crate::prelude::SessionConfig) extension.
/// It does not enable morsel execution by itself and does not instrument fallback
/// plans. Detailed metrics are separate from the existing plan-lifetime metrics.
///
/// ```
/// # #[cfg(feature = "sql")]
/// # async fn example() -> datafusion::error::Result<()> {
/// use std::sync::Arc;
/// use datafusion::arrow::array::{Int32Array, RecordBatch};
/// use datafusion::common::config::MorselMode;
/// use datafusion::morsel::MorselMetrics;
/// use datafusion::prelude::{SessionConfig, SessionContext};
///
/// let metrics = Arc::new(MorselMetrics::default());
/// let mut config = SessionConfig::new().with_extension(Arc::clone(&metrics));
/// config.options_mut().execution.morsel_mode = MorselMode::Required;
/// let ctx = SessionContext::new_with_config(config);
/// ctx.register_batch("fixture", RecordBatch::try_from_iter([(
///     "id", Arc::new(Int32Array::from(vec![1, 2, 3])) as _,
/// )])?)?;
/// ctx.sql("SELECT id FROM fixture").await?.collect().await?;
/// let runs = metrics.take_runs();
/// assert_eq!(runs.len(), 1);
/// assert_eq!(runs[0].sum_by_name("morsels_completed").unwrap().as_usize(), 1);
/// # Ok(())
/// # }
/// # #[cfg(feature = "sql")]
/// # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(example()).unwrap();
/// ```
#[derive(Debug, Default)]
pub struct MorselMetrics {
    finished: Mutex<Vec<MetricsSet>>,
}

impl MorselMetrics {
    /// Drain reports in completion order, one per started native execution.
    /// A report is available after the stream has finished (or been dropped)
    /// and its tasks have stopped. Failed or cancelled runs report partial totals.
    /// Use a separate collector per task context to correlate concurrent runs.
    pub fn take_runs(&self) -> Vec<MetricsSet> {
        std::mem::take(&mut *self.finished.lock())
    }
}

pub(super) struct RunMetrics {
    collector: Arc<MorselMetrics>,
    metrics: ExecutionPlanMetricsSet,
    pub(super) output: QueueStats,
    #[cfg(feature = "parquet")]
    pub(super) ready: QueueStats,
    #[cfg(feature = "parquet")]
    pending_io: Time,
}

impl RunMetrics {
    pub(super) fn start(collector: Arc<MorselMetrics>) -> Arc<Self> {
        let metrics = ExecutionPlanMetricsSet::new();
        Arc::new(Self {
            collector,
            #[cfg(feature = "parquet")]
            pending_io: MetricBuilder::new(&metrics).subset_time("morsel_pending_io", 0),
            metrics,
            output: QueueStats::default(),
            #[cfg(feature = "parquet")]
            ready: QueueStats::default(),
        })
    }

    pub(super) fn record_pipeline_time(
        &self,
        pipeline: &str,
        name: &'static str,
        duration: Duration,
    ) {
        MetricBuilder::new(&self.metrics)
            .with_label(Label::new("pipeline", pipeline.to_owned()))
            .subset_time(name, 0)
            .add_duration_exact(duration);
    }

    #[cfg(feature = "parquet")]
    pub(super) async fn io<T>(self: Arc<Self>, future: impl Future<Output = T>) -> T {
        struct IoWait {
            run: Arc<RunMetrics>,
            duration: Duration,
        }

        impl Drop for IoWait {
            fn drop(&mut self) {
                self.run.pending_io.add_duration_exact(self.duration);
            }
        }

        let mut io = IoWait {
            run: self,
            duration: Duration::ZERO,
        };
        wait(Some(&mut io.duration), future).await
    }
}

impl Drop for RunMetrics {
    fn drop(&mut self) {
        self.output.report(
            &self.metrics,
            "morsel_peak_output_depth",
            "morsel_peak_output_arrow_bytes",
        );
        #[cfg(feature = "parquet")]
        self.ready.report(
            &self.metrics,
            "morsel_peak_ready_depth",
            "morsel_peak_ready_compressed_bytes_estimate",
        );
        self.collector
            .finished
            .lock()
            .push(self.metrics.clone_inner());
    }
}

#[derive(Default)]
pub(super) struct WorkerMetrics {
    run: Option<Arc<RunMetrics>>,
    id: usize,
    pipeline: Option<String>,
    claimed: usize,
    completed: usize,
    decode: Duration,
    process: Duration,
    ready_wait: Duration,
    output_wait: Duration,
}

impl WorkerMetrics {
    pub(super) fn new(run: Option<Arc<RunMetrics>>, id: usize) -> Self {
        Self {
            run,
            id,
            pipeline: None,
            claimed: 0,
            completed: 0,
            decode: Duration::ZERO,
            process: Duration::ZERO,
            ready_wait: Duration::ZERO,
            output_wait: Duration::ZERO,
        }
    }

    pub(super) fn for_pipeline(
        run: Option<Arc<RunMetrics>>,
        id: usize,
        pipeline: impl Into<String>,
    ) -> Self {
        Self {
            run,
            id,
            pipeline: Some(pipeline.into()),
            claimed: 0,
            completed: 0,
            decode: Duration::ZERO,
            process: Duration::ZERO,
            ready_wait: Duration::ZERO,
            output_wait: Duration::ZERO,
        }
    }

    pub(super) fn claim(&mut self) {
        if self.run.is_some() {
            self.claimed += 1;
        }
    }

    pub(super) fn complete(&mut self) {
        if self.run.is_some() {
            self.completed += 1;
        }
    }

    pub(super) fn process<T>(&mut self, f: impl FnOnce() -> T) -> T {
        time(self.run.as_ref().map(|_| &mut self.process), f)
    }

    #[cfg(feature = "parquet")]
    pub(super) fn decode<T>(&mut self, f: impl FnOnce() -> T) -> T {
        time(self.run.as_ref().map(|_| &mut self.decode), f)
    }

    #[cfg(feature = "parquet")]
    pub(super) async fn ready<T>(&mut self, future: impl Future<Output = T>) -> T {
        wait(self.run.as_ref().map(|_| &mut self.ready_wait), future).await
    }

    pub(super) async fn send(
        &mut self,
        tx: &mpsc::Sender<RecordBatch>,
        batch: RecordBatch,
    ) -> bool {
        if let Some(run) = &self.run {
            send(run, Some(&mut self.output_wait), tx, batch).await
        } else {
            tx.send(batch).await.is_ok()
        }
    }
}

impl Drop for WorkerMetrics {
    fn drop(&mut self) {
        let Some(run) = &self.run else { return };
        let builder = || {
            let mut b = MetricBuilder::new(&run.metrics)
                .with_label(Label::new("worker", self.id.to_string()));
            if let Some(pipeline) = &self.pipeline {
                b = b.with_label(Label::new("pipeline", pipeline.clone()));
            }
            b
        };
        builder().counter("morsels_claimed", 0).add(self.claimed);
        builder()
            .counter("morsels_completed", 0)
            .add(self.completed);
        for (name, duration) in [
            ("morsel_decode_work", self.decode),
            ("morsel_process_work", self.process),
            ("morsel_ready_wait", self.ready_wait),
            ("morsel_output_wait", self.output_wait),
        ] {
            builder().subset_time(name, 0).add_duration_exact(duration);
        }
    }
}

fn time<T>(total: Option<&mut Duration>, f: impl FnOnce() -> T) -> T {
    let start = total.as_ref().map(|_| Instant::now());
    let result = f();
    if let Some((total, start)) = total.zip(start) {
        *total += start.elapsed();
    }
    result
}

#[cfg(feature = "parquet")]
pub(super) async fn decode<T>(
    worker: &mut WorkerMetrics,
    future: impl Future<Output = T>,
) -> T {
    let mut future = pin!(future);
    poll_fn(|cx| worker.decode(|| future.as_mut().poll(cx))).await
}

async fn wait<T>(total: Option<&mut Duration>, future: impl Future<Output = T>) -> T {
    let Some(total) = total else {
        return future.await;
    };
    struct Suspended<'a> {
        total: &'a mut Duration,
        since: Option<Instant>,
    }
    impl Suspended<'_> {
        fn resume(&mut self) {
            if let Some(start) = self.since.take() {
                *self.total += start.elapsed();
            }
        }
    }
    impl Drop for Suspended<'_> {
        fn drop(&mut self) {
            self.resume();
        }
    }
    let mut suspended = Suspended { total, since: None };
    let mut future = pin!(future);
    poll_fn(|cx| {
        suspended.resume();
        let result = future.as_mut().poll(cx);
        if result.is_pending() {
            suspended.since = Some(Instant::now());
        }
        result
    })
    .await
}

pub(super) async fn send(
    run: &RunMetrics,
    wait_time: Option<&mut Duration>,
    tx: &mpsc::Sender<RecordBatch>,
    batch: RecordBatch,
) -> bool {
    let Ok(permit) = wait(wait_time, tx.reserve()).await else {
        return false;
    };
    run.output.publish(permit, batch, Footprint::batch);
    true
}

pub(super) async fn recv<T>(
    stats: Option<&QueueStats>,
    rx: &mut mpsc::Receiver<T>,
) -> Option<T> {
    match stats {
        Some(stats) => {
            poll_fn(|cx| {
                let mut state = stats.0.lock();
                let item = rx.poll_recv(cx);
                if let Poll::Ready(Some(_)) = &item {
                    state.pop();
                }
                item
            })
            .await
        }
        None => rx.recv().await,
    }
}

#[derive(Default)]
pub(super) struct Footprint {
    estimate: usize,
    buffers: Vec<(usize, usize)>,
}

impl Footprint {
    fn batch(batch: &RecordBatch) -> Self {
        fn collect(data: &ArrayData, buffers: &mut Vec<(usize, usize)>) {
            for buffer in data
                .buffers()
                .iter()
                .chain(data.nulls().map(|n| n.buffer()))
            {
                if buffer.capacity() > 0 {
                    let start = buffer.data_ptr().as_ptr() as usize;
                    buffers.push((start, start + buffer.capacity()));
                }
            }
            for child in data.child_data() {
                collect(child, buffers);
            }
        }
        let mut footprint = Self::default();
        for col in batch.columns() {
            collect(&col.to_data(), &mut footprint.buffers);
        }
        footprint
    }
}

#[derive(Default)]
pub(super) struct QueueStats(Mutex<QueueState>);

impl QueueStats {
    // Serialize the actual channel send and poll_recv with accounting. A receiver
    // can release capacity while another sender already holds a permit.
    pub(super) fn publish<T>(
        &self,
        permit: mpsc::Permit<'_, T>,
        item: T,
        footprint: impl FnOnce(&T) -> Footprint,
    ) {
        let footprint = footprint(&item);
        let mut state = self.0.lock();
        state.push(footprint);
        permit.send(item);
    }

    // The Parquet queue calls these while holding its queue lock, keeping
    // accounting in the same order as the corresponding queue operations.
    #[cfg(feature = "parquet")]
    pub(super) fn push_estimate(&self, bytes: usize) {
        self.0.lock().push(Footprint {
            estimate: bytes,
            buffers: vec![],
        });
    }

    #[cfg(feature = "parquet")]
    pub(super) fn pop_estimate(&self) {
        self.0.lock().pop();
    }

    fn report(
        &self,
        metrics: &ExecutionPlanMetricsSet,
        depth: &'static str,
        bytes: &'static str,
    ) {
        let state = self.0.lock();
        MetricBuilder::new(metrics)
            .global_gauge(depth)
            .set(state.peak_depth);
        MetricBuilder::new(metrics)
            .bytes_gauge(bytes, 0)
            .set(state.peak_bytes);
    }
}

#[derive(Default)]
struct QueueState {
    queued: VecDeque<Footprint>,
    ranges: Vec<(usize, usize)>,
    peak_depth: usize,
    peak_bytes: usize,
}

impl QueueState {
    fn push(&mut self, footprint: Footprint) {
        self.queued.push_back(footprint);
        self.peak_depth = self.peak_depth.max(self.queued.len());
        // Count the union, including slices, repeated columns and shared child
        // or dictionary buffers, even across simultaneously queued batches.
        self.ranges.clear();
        let mut bytes = 0;
        for item in &self.queued {
            bytes += item.estimate;
            self.ranges.extend_from_slice(&item.buffers);
        }
        self.ranges.sort_unstable();
        let mut end = 0;
        for &(lo, hi) in &self.ranges {
            bytes += hi.saturating_sub(lo.max(end));
            end = end.max(hi);
        }
        self.peak_bytes = self.peak_bytes.max(bytes);
    }

    fn pop(&mut self) {
        self.queued.pop_front().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int32Array, StructArray};
    use arrow::datatypes::Field;

    #[tokio::test]
    async fn queue_counts_shared_arrow_storage_once() {
        let data = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let capacity =
            data.values().inner().capacity() + data.nulls().unwrap().buffer().capacity();
        let batch = |values: ArrayRef| {
            let field = Arc::new(Field::new("child", values.data_type().clone(), true));
            let nested = StructArray::from(vec![(field, Arc::clone(&values))]);
            RecordBatch::try_from_iter([("values", values), ("nested", Arc::new(nested))])
                .unwrap()
        };
        let (tx, mut rx) = mpsc::channel(2);
        let stats = QueueStats::default();
        for offset in [0, 2] {
            stats.publish(
                tx.reserve().await.unwrap(),
                batch(Arc::new(data.slice(offset, 2))),
                Footprint::batch,
            );
        }
        assert_eq!(stats.0.lock().peak_depth, 2);
        assert_eq!(stats.0.lock().peak_bytes, capacity);
        for _ in 0..2 {
            assert!(recv(Some(&stats), &mut rx).await.is_some());
        }
        let unrelated = Int32Array::from(vec![20, 30]);
        stats.publish(
            tx.reserve().await.unwrap(),
            batch(Arc::new(unrelated)),
            Footprint::batch,
        );
        assert_eq!(stats.0.lock().queued.len(), 1);
        assert_eq!(stats.0.lock().peak_bytes, capacity);
    }
}
