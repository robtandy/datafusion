// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::VecDeque;
use std::sync::Arc;

use datafusion_datasource::morsel::Morsel;
use datafusion_physical_plan::metrics::Count;
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use super::RunMetrics;

pub(super) fn channel(
    capacity: usize,
    measurement: Option<Arc<RunMetrics>>,
    claimed: Count,
) -> (Publisher, ReadyReaders) {
    let queue = Arc::new(Queue {
        readers: Mutex::new(VecDeque::with_capacity(capacity)),
        slots: Semaphore::new(capacity),
        available: Semaphore::new(0),
        measurement,
        claimed,
    });
    (Publisher(Arc::clone(&queue)), ReadyReaders(queue))
}

struct Queue {
    readers: Mutex<VecDeque<Box<dyn Morsel>>>,
    slots: Semaphore,
    available: Semaphore,
    measurement: Option<Arc<RunMetrics>>,
    claimed: Count,
}

// The file lanes share one publisher, so dropping it closes the queue only
// after all lanes have exited. Readers can still drain any published work.
pub(super) struct Publisher(Arc<Queue>);

impl Publisher {
    pub(super) async fn send(&self, morsel: Box<dyn Morsel>) {
        let slot = self.0.slots.acquire().await.expect("slots never close");
        {
            let mut readers = self.0.readers.lock();
            if let Some(run) = &self.0.measurement {
                run.ready.push_estimate(morsel.buffered_bytes_estimate());
            }
            readers.push_back(morsel);
            slot.forget();
        }
        self.0.available.add_permits(1);
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        self.0.available.close();
    }
}

#[derive(Clone)]
pub(in crate::morsel) struct ReadyReaders(Arc<Queue>);

impl ReadyReaders {
    pub(in crate::morsel) async fn claim(&self) -> Option<Box<dyn Morsel>> {
        if let Ok(item) = self.0.available.acquire().await {
            item.forget();
        }
        // On close there can be no more publications. Drain whatever remains
        // without permits, including racing any peers that already took one.
        let morsel = {
            let mut readers = self.0.readers.lock();
            let morsel = readers.pop_front()?;
            if let Some(run) = &self.0.measurement {
                run.ready.pop_estimate();
            }
            morsel
        };
        self.0.slots.add_permits(1);
        self.0.claimed.add(1);
        Some(morsel)
    }
}
