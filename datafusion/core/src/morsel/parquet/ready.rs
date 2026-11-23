// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::VecDeque;
use std::sync::Arc;

use datafusion_datasource::morsel::{Morsel, MorselPlanner, PendingMorselPlanner};
use datafusion_physical_plan::metrics::Count;
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use super::RunMetrics;

pub(super) fn channel(
    capacity: usize,
    measurement: Option<Arc<RunMetrics>>,
    claimed: Count,
) -> (Publisher, Phase2Queue, ReadyReaders) {
    let queue = Arc::new(Queue {
        inner: Mutex::new(QueueInner {
            readers: VecDeque::with_capacity(capacity),
            phase2: VecDeque::with_capacity(capacity),
            producers_open: true,
            pending_filter_rgs: 0,
        }),
        slots: Semaphore::new(capacity),
        available: Semaphore::new(0),
        phase2_available: Semaphore::new(0),
        measurement,
        claimed,
    });
    (
        Publisher(Arc::clone(&queue)),
        Phase2Queue(Arc::clone(&queue)),
        ReadyReaders(queue),
    )
}

struct QueueInner {
    // Queued entries never hold an `Arc<Queue>` so aborting a run with
    // unconsumed items cannot create a reference cycle that pins `RunMetrics`.
    readers: VecDeque<QueuedWork>,
    phase2: VecDeque<PendingMorselPlanner>,
    producers_open: bool,
    pending_filter_rgs: usize,
}

struct Queue {
    inner: Mutex<QueueInner>,
    slots: Semaphore,
    available: Semaphore,
    phase2_available: Semaphore,
    measurement: Option<Arc<RunMetrics>>,
    claimed: Count,
}

impl Queue {
    fn maybe_close(&self, inner: &QueueInner) {
        if !inner.producers_open && inner.pending_filter_rgs == 0 {
            self.available.close();
            self.phase2_available.close();
        }
    }

    fn push_ready(&self, inner: &mut QueueInner, item: QueuedWork, bytes: usize) {
        if let Some(run) = &self.measurement {
            run.ready.push_estimate(bytes);
        }
        inner.readers.push_back(item);
        self.available.add_permits(1);
    }
}

/// RAII token for one slot held outside [`Queue`] by a worker or Phase-2 task.
pub(in crate::morsel) struct SlotPermit(Option<Arc<Queue>>);

impl SlotPermit {
    fn new(queue: Arc<Queue>) -> Self {
        Self(Some(queue))
    }

    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for SlotPermit {
    fn drop(&mut self) {
        if let Some(queue) = self.0.take() {
            queue.slots.add_permits(1);
        }
    }
}

/// RAII token tracking a filtered row group held outside [`Queue`] whose
/// `RowFilter` or Phase-2 I/O has not yet published its final decode morsel.
pub(in crate::morsel) struct FilterRgGuard(Option<Arc<Queue>>);

impl FilterRgGuard {
    fn new(queue: Arc<Queue>) -> Self {
        Self(Some(queue))
    }

    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for FilterRgGuard {
    fn drop(&mut self) {
        if let Some(queue) = self.0.take() {
            let mut inner = queue.inner.lock();
            inner.pending_filter_rgs -= 1;
            queue.maybe_close(&inner);
        }
    }
}

enum QueuedWork {
    Morsel {
        morsel: Box<dyn Morsel>,
        count_claim: bool,
    },
    FilterStep {
        planner: Box<dyn MorselPlanner>,
        count_claim: bool,
    },
}

pub(in crate::morsel) enum ClaimedWork {
    Morsel {
        morsel: Box<dyn Morsel>,
        count_claim: bool,
    },
    FilterStep {
        planner: Box<dyn MorselPlanner>,
        permit: SlotPermit,
        filter_guard: FilterRgGuard,
        count_claim: bool,
    },
}

pub(super) struct Phase2Task {
    pub(super) io: PendingMorselPlanner,
    pub(super) permit: SlotPermit,
    pub(super) filter_guard: FilterRgGuard,
}

// The file lanes share one publisher, so dropping it marks producers closed
// only after all lanes have exited. Workers and Phase-2 I/O tasks still drain
// any in-flight work before channels close.
pub(super) struct Publisher(Arc<Queue>);

impl Publisher {
    pub(super) async fn send(&self, morsel: Box<dyn Morsel>) {
        let slot = self.0.slots.acquire().await.expect("slots never close");
        let bytes = morsel.buffered_bytes_estimate();
        let mut inner = self.0.inner.lock();
        self.0.push_ready(
            &mut inner,
            QueuedWork::Morsel {
                morsel,
                count_claim: true,
            },
            bytes,
        );
        slot.forget();
    }

    pub(super) async fn send_filter(&self, planner: Box<dyn MorselPlanner>) {
        let slot = self.0.slots.acquire().await.expect("slots never close");
        let bytes = planner.buffered_bytes_estimate();
        let mut inner = self.0.inner.lock();
        inner.pending_filter_rgs += 1;
        self.0.push_ready(
            &mut inner,
            QueuedWork::FilterStep {
                planner,
                count_claim: true,
            },
            bytes,
        );
        slot.forget();
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        let mut inner = self.0.inner.lock();
        inner.producers_open = false;
        self.0.maybe_close(&inner);
    }
}

#[derive(Clone)]
pub(super) struct Phase2Queue(Arc<Queue>);

impl Phase2Queue {
    pub(super) async fn claim(&self) -> Option<Phase2Task> {
        if let Ok(item) = self.0.phase2_available.acquire().await {
            item.forget();
        }
        let io = self.0.inner.lock().phase2.pop_front()?;
        Some(Phase2Task {
            io,
            permit: SlotPermit::new(Arc::clone(&self.0)),
            filter_guard: FilterRgGuard::new(Arc::clone(&self.0)),
        })
    }

    pub(super) fn publish_morsel(
        &self,
        morsel: Box<dyn Morsel>,
        permit: SlotPermit,
        filter_guard: FilterRgGuard,
    ) {
        let bytes = morsel.buffered_bytes_estimate();
        {
            let mut inner = self.0.inner.lock();
            self.0.push_ready(
                &mut inner,
                QueuedWork::Morsel {
                    morsel,
                    count_claim: false,
                },
                bytes,
            );
            permit.disarm();
        }
        // Drop guard after enqueuing so closing `available` happens only once
        // the decode-only morsel is visible to workers.
        drop(filter_guard);
    }

    pub(super) fn publish_filter(
        &self,
        planner: Box<dyn MorselPlanner>,
        permit: SlotPermit,
        filter_guard: FilterRgGuard,
    ) {
        let bytes = planner.buffered_bytes_estimate();
        let mut inner = self.0.inner.lock();
        self.0.push_ready(
            &mut inner,
            QueuedWork::FilterStep {
                planner,
                count_claim: false,
            },
            bytes,
        );
        permit.disarm();
        filter_guard.disarm();
    }
}

#[derive(Clone)]
pub(in crate::morsel) struct ReadyReaders(Arc<Queue>);

impl ReadyReaders {
    pub(in crate::morsel) async fn claim(&self) -> Option<ClaimedWork> {
        if let Ok(item) = self.0.available.acquire().await {
            item.forget();
        }
        // On close there can be no more publications. Drain whatever remains
        // without permits, including racing any peers that already took one.
        let work = {
            let mut inner = self.0.inner.lock();
            let work = inner.readers.pop_front()?;
            if let Some(run) = &self.0.measurement {
                run.ready.pop_estimate();
            }
            work
        };
        match work {
            QueuedWork::Morsel {
                morsel,
                count_claim,
            } => {
                self.0.slots.add_permits(1);
                if count_claim {
                    self.0.claimed.add(1);
                }
                Some(ClaimedWork::Morsel {
                    morsel,
                    count_claim,
                })
            }
            QueuedWork::FilterStep {
                planner,
                count_claim,
            } => {
                if count_claim {
                    self.0.claimed.add(1);
                }
                Some(ClaimedWork::FilterStep {
                    planner,
                    permit: SlotPermit::new(Arc::clone(&self.0)),
                    filter_guard: FilterRgGuard::new(Arc::clone(&self.0)),
                    count_claim,
                })
            }
        }
    }

    pub(in crate::morsel) fn submit_phase2(
        &self,
        io: PendingMorselPlanner,
        permit: SlotPermit,
        filter_guard: FilterRgGuard,
    ) {
        let mut inner = self.0.inner.lock();
        inner.phase2.push_back(io);
        permit.disarm();
        filter_guard.disarm();
        self.0.phase2_available.add_permits(1);
    }
}
