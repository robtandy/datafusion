// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};
use datafusion::common::runtime::SpawnedTask;
#[cfg(feature = "morsel-metrics")]
use datafusion::physical_plan::metrics::MetricsSet;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep_until};

use crate::Args;
use crate::workload::{Activity, Digest, Progress, Query};

const MAX_SCANS_PER_LANE: usize = 10_000;

#[derive(Serialize)]
pub(crate) struct Report {
    pub(crate) valid: bool,
    mode: &'static str,
    repetition: u32,
    offered: usize,
    queued: usize,
    dropped: usize,
    failed: usize,
    overdue: usize,
    late_offers: usize,
    completed: usize,
    overlap_requests: usize,
    peak_background_at_request_start: usize,
    offer_window_ns: u64,
    foreground_elapsed_ns: u64,
    offered_per_second: f64,
    completed_per_second: f64,
    completion_rate: f64,
    drop_rate: f64,
    latency: Distribution,
    offer_lag: Distribution,
    queue_wait: Distribution,
    service_time: Distribution,
    planning: Distribution,
    background: BackgroundReport,
    #[cfg(feature = "morsel-metrics")]
    #[serde(skip_serializing_if = "Option::is_none")]
    morsel_metrics: Option<MorselMetricsReport>,
    requests: Vec<Request>,
}

#[cfg(feature = "morsel-metrics")]
#[derive(Serialize)]
struct MorselMetricsReport {
    foreground: Vec<MorselExecutionMetrics>,
    background: Vec<MorselExecutionMetrics>,
}

#[cfg(feature = "morsel-metrics")]
#[derive(Serialize)]
struct MorselExecutionMetrics {
    pending_io_ns: usize,
    ready_wait_ns: usize,
    decode_work_ns: usize,
    process_work_ns: usize,
    output_wait_ns: usize,
    morsels_claimed: usize,
    morsels_completed: usize,
    peak_ready_depth: usize,
    peak_ready_compressed_bytes_estimate: usize,
    peak_output_depth: usize,
    peak_output_arrow_bytes: usize,
}

#[cfg(feature = "morsel-metrics")]
impl MorselExecutionMetrics {
    fn from_metrics_set(set: &MetricsSet) -> Self {
        let sum = |name: &str| {
            set.sum_by_name(name)
                .map(|value| value.as_usize())
                .unwrap_or(0)
        };
        Self {
            pending_io_ns: sum("morsel_pending_io"),
            ready_wait_ns: sum("morsel_ready_wait"),
            decode_work_ns: sum("morsel_decode_work"),
            process_work_ns: sum("morsel_process_work"),
            output_wait_ns: sum("morsel_output_wait"),
            morsels_claimed: sum("morsels_claimed"),
            morsels_completed: sum("morsels_completed"),
            peak_ready_depth: sum("morsel_peak_ready_depth"),
            peak_ready_compressed_bytes_estimate: sum(
                "morsel_peak_ready_compressed_bytes_estimate",
            ),
            peak_output_depth: sum("morsel_peak_output_depth"),
            peak_output_arrow_bytes: sum("morsel_peak_output_arrow_bytes"),
        }
    }
}

#[derive(Default, Serialize)]
struct BackgroundReport {
    lanes_polled: usize,
    exhausted_lanes: usize,
    completed: usize,
    completed_in_window: usize,
    failed: usize,
    first_error: Option<String>,
    completed_in_window_per_second: f64,
}

#[derive(Serialize)]
struct Request {
    id: u32,
    scheduled_ns: u64,
    offered_ns: u64,
    started_ns: Option<u64>,
    plan_ready_ns: Option<u64>,
    execution_started_ns: Option<u64>,
    finished_ns: Option<u64>,
    queued: bool,
    overdue: bool,
    status: Status,
    active_scans_at_execution_start: usize,
    error: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Completed,
    Failed,
    Dropped,
}

struct Offer {
    id: u32,
    scheduled: Instant,
    offered: Instant,
    queued: bool,
}

struct Service {
    offer: Offer,
    started: Instant,
    finished: Instant,
    progress: Progress,
    result: Result<Digest>,
}

#[derive(Default)]
struct Load {
    active: AtomicUsize,
    end: OnceLock<Instant>,
}

struct Lane {
    scans: Vec<(Instant, Result<Digest>)>,
    exhausted: bool,
}

pub(crate) async fn measure(
    args: &Args,
    repetition: u32,
    small: &Query,
    large: Option<&Query>,
) -> Result<Report> {
    let load = Arc::new(Load::default());
    let timeout = Duration::from_millis(args.timeout_ms);
    let interval = Duration::from_millis(args.interval_ms);
    let overdue = Duration::from_millis(args.overdue_ms);
    let window = interval * args.requests;

    let (tx, mut rx) = mpsc::channel::<Offer>(args.queue_capacity as usize);
    let outstanding = Arc::new(AtomicUsize::new(0));
    let (ready, small_ready) = oneshot::channel();
    let worker = {
        let small = small.clone();
        let load = Arc::clone(&load);
        let outstanding = Arc::clone(&outstanding);
        SpawnedTask::spawn(async move {
            let _ = ready.send(());
            let mut results = Vec::new();
            while let Some(offer) = rx.recv().await {
                let started = Instant::now();
                let mut progress = Progress::default();
                let result = small
                    .run(
                        offer.scheduled + timeout,
                        Activity::Foreground(&load.active),
                        &mut progress,
                    )
                    .await;
                let finished = Instant::now();
                outstanding.fetch_sub(1, Ordering::SeqCst);
                results.push(Service {
                    offer,
                    started,
                    finished,
                    progress,
                    result,
                });
            }
            results
        })
    };
    small_ready
        .await
        .context("foreground worker did not start")?;

    let mut lanes = JoinSet::new();
    let mut lane_ready = Vec::new();
    if let Some(large) = large {
        for _ in 0..args.background_lanes {
            let (ready, polled) = oneshot::channel();
            lane_ready.push(polled);
            lanes.spawn(scan_lane(large.clone(), Arc::clone(&load), timeout, ready));
        }
    }
    let mut background = BackgroundReport::default();
    for polled in lane_ready {
        // A lane that fails before polling also unblocks startup; its error is reported after drain.
        background.lanes_polled += usize::from(polled.await.is_ok());
    }

    let epoch = Instant::now();
    let end = epoch + window;
    load.end.set(end).expect("new measurement window");
    let mut requests = Vec::with_capacity(args.requests as usize);
    for id in 0..args.requests {
        let scheduled = epoch + interval * id;
        // This is the block_on caller; offer pacing does not need a Tokio timer.
        let wait = scheduled.saturating_duration_since(Instant::now());
        if !wait.is_zero() {
            std::thread::sleep(wait);
        }
        let offered = Instant::now();
        match tx.try_reserve() {
            Ok(permit) => {
                let queued = outstanding.fetch_add(1, Ordering::SeqCst) > 0;
                permit.send(Offer {
                    id,
                    scheduled,
                    offered,
                    queued,
                });
            }
            Err(mpsc::error::TrySendError::Full(_)) => requests.push(Request {
                id,
                scheduled_ns: ns(scheduled - epoch),
                offered_ns: ns(offered - epoch),
                started_ns: None,
                plan_ready_ns: None,
                execution_started_ns: None,
                finished_ns: None,
                queued: false,
                overdue: offered - scheduled > overdue,
                status: Status::Dropped,
                active_scans_at_execution_start: 0,
                error: None,
            }),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                anyhow::bail!("foreground worker exited early")
            }
        }
    }
    drop(tx);
    sleep_until(end).await;
    let services = worker.await.context("joining foreground worker")?;
    let mut drained_lanes = Vec::new();
    while let Some(lane) = lanes.join_next().await {
        drained_lanes.push(lane.context("joining background lane")?);
    }
    #[cfg(feature = "morsel-metrics")]
    let morsel_metrics = if args.morsel_metrics {
        tokio::task::yield_now().await;
        let foreground = small
            .take_metrics()
            .unwrap_or_default()
            .iter()
            .map(MorselExecutionMetrics::from_metrics_set)
            .collect();
        let background = large
            .and_then(Query::take_metrics)
            .unwrap_or_default()
            .iter()
            .map(MorselExecutionMetrics::from_metrics_set)
            .collect();
        Some(MorselMetricsReport {
            foreground,
            background,
        })
    } else {
        None
    };

    let mut foreground_end = end;
    for Service {
        offer,
        started,
        finished,
        progress,
        result,
    } in services
    {
        foreground_end = foreground_end.max(finished);
        let error = small
            .validate(result)
            .err()
            .map(|error| format!("{error:#}"));
        requests.push(Request {
            id: offer.id,
            scheduled_ns: ns(offer.scheduled - epoch),
            offered_ns: ns(offer.offered - epoch),
            started_ns: Some(ns(started - epoch)),
            plan_ready_ns: progress.plan_ready.map(|t| ns(t - epoch)),
            execution_started_ns: progress.execution_started.map(|t| ns(t - epoch)),
            finished_ns: Some(ns(finished - epoch)),
            queued: offer.queued,
            overdue: finished - offer.scheduled > overdue,
            status: if error.is_none() {
                Status::Completed
            } else {
                Status::Failed
            },
            active_scans_at_execution_start: progress.active_scans,
            error,
        });
    }
    if let Some(large) = large {
        for lane in drained_lanes {
            background.exhausted_lanes += usize::from(lane.exhausted);
            for (finished, result) in lane.scans {
                match large.validate(result) {
                    Ok(()) => {
                        background.completed += 1;
                        background.completed_in_window +=
                            usize::from(finished >= epoch && finished < end);
                    }
                    Err(error) => {
                        background.failed += 1;
                        background
                            .first_error
                            .get_or_insert_with(|| format!("{error:#}"));
                    }
                }
            }
        }
        background.completed_in_window_per_second =
            background.completed_in_window as f64 / window.as_secs_f64();
    }

    requests.sort_unstable_by_key(|request| request.id);
    let successful = || requests.iter().filter(|r| r.status == Status::Completed);
    let completed = successful().count();
    let failed = requests
        .iter()
        .filter(|r| r.status == Status::Failed)
        .count();
    let dropped = requests
        .iter()
        .filter(|r| r.status == Status::Dropped)
        .count();
    let offered = requests.len();
    let overlap_requests = successful()
        .filter(|r| r.active_scans_at_execution_start > 0)
        .count();
    let foreground_elapsed = foreground_end - epoch;
    Ok(Report {
        valid: failed == 0
            && completed > 0
            && large.is_none_or(|_| {
                background.failed == 0
                    && background.exhausted_lanes == 0
                    && background.lanes_polled == args.background_lanes as usize
                    && overlap_requests > 0
            }),
        mode: if large.is_some() { "mixed" } else { "isolated" },
        repetition,
        offered,
        queued: requests.iter().filter(|r| r.queued).count(),
        dropped,
        failed,
        overdue: requests.iter().filter(|r| r.overdue).count(),
        late_offers: requests
            .iter()
            .filter(|r| r.offered_ns.saturating_sub(r.scheduled_ns) >= ns(interval))
            .count(),
        completed,
        overlap_requests,
        peak_background_at_request_start: successful()
            .map(|r| r.active_scans_at_execution_start)
            .max()
            .unwrap_or(0),
        offer_window_ns: ns(window),
        foreground_elapsed_ns: ns(foreground_elapsed),
        offered_per_second: offered as f64 / window.as_secs_f64(),
        completed_per_second: completed as f64 / foreground_elapsed.as_secs_f64(),
        completion_rate: completed as f64 / offered as f64,
        drop_rate: dropped as f64 / offered as f64,
        latency: Distribution::new(
            successful().filter_map(|r| Some(r.finished_ns? - r.scheduled_ns)),
        ),
        offer_lag: Distribution::new(
            successful().map(|r| r.offered_ns.saturating_sub(r.scheduled_ns)),
        ),
        queue_wait: Distribution::new(
            successful().filter_map(|r| Some(r.started_ns? - r.offered_ns)),
        ),
        service_time: Distribution::new(
            successful().filter_map(|r| Some(r.finished_ns? - r.started_ns?)),
        ),
        planning: Distribution::new(
            successful().filter_map(|r| Some(r.plan_ready_ns? - r.started_ns?)),
        ),
        background,
        #[cfg(feature = "morsel-metrics")]
        morsel_metrics,
        requests,
    })
}

async fn scan_lane(
    query: Query,
    load: Arc<Load>,
    timeout: Duration,
    ready: oneshot::Sender<()>,
) -> Lane {
    let window_open = || load.end.get().is_none_or(|end| Instant::now() < *end);
    let mut scans = Vec::new();
    let mut ready = Some(ready);
    while scans.len() < MAX_SCANS_PER_LANE && window_open() {
        let result = query
            .run(
                Instant::now() + timeout,
                Activity::Background(&load.active, ready.take()),
                &mut Progress::default(),
            )
            .await;
        let stop = result.is_err();
        scans.push((Instant::now(), result));
        if stop {
            break;
        }
        tokio::task::yield_now().await;
    }
    Lane {
        exhausted: scans.len() == MAX_SCANS_PER_LANE && window_open(),
        scans,
    }
}

fn ns(duration: Duration) -> u64 {
    duration.as_nanos() as u64
}

#[derive(Serialize)]
struct Distribution {
    samples: usize,
    p50_ns: Option<u64>,
    p95_ns: Option<u64>,
    p99_ns: Option<u64>,
    sample_variance_ns2: Option<f64>,
}

impl Distribution {
    fn new(samples: impl Iterator<Item = u64>) -> Self {
        let mut samples: Vec<_> = samples.collect();
        samples.sort_unstable();
        let n = samples.len();
        let quantile =
            |percent: usize| (n > 0).then(|| samples[(n * percent).div_ceil(100) - 1]);
        let variance = (n > 1).then(|| {
            let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            samples
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64
        });
        Self {
            samples: n,
            p50_ns: quantile(50),
            p95_ns: quantile(95),
            p99_ns: quantile(99),
            sample_variance_ns2: variance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_uses_nearest_rank_and_sample_variance() {
        let result = Distribution::new([4, 1, 3, 2].into_iter());
        assert_eq!(result.samples, 4);
        assert_eq!(result.p50_ns, Some(2));
        assert_eq!(result.p95_ns, Some(4));
        assert_eq!(result.p99_ns, Some(4));
        assert_eq!(result.sample_variance_ns2, Some(5.0 / 3.0));
        assert_eq!(Distribution::new([1].into_iter()).sample_variance_ns2, None);
        assert_eq!(Distribution::new(std::iter::empty()).p99_ns, None);
    }
}
