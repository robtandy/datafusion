// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Shared `SpillManager` helpers and cancellation-safe spill I/O wrappers for
//! morsel `GROUP BY`, `HashJoin`, and `Sort`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::spill_file::SpillFile;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
use datafusion_physical_plan::spill::SpillManager;
use tokio::sync::mpsc;

/// Cap for the minimum memory-pool viability check before enabling spilling.
/// Respected alongside `execution.sort_spill_reservation_bytes` so pools that
/// cannot even hold minimal IPC framing + one radix partition fail fast with
/// `ResourcesExhausted` instead of thrashing.
const MIN_SPILL_WORKING_SET_CAP: usize = 256 * 1024;

#[derive(Clone, Debug)]
pub(super) struct SpillContext {
    pub(super) manager: SpillManager,
    pub(super) min_headroom_bytes: usize,
    spilled: Arc<AtomicBool>,
}

impl SpillContext {
    pub(super) fn try_new(
        context: &TaskContext,
        plan_metrics: &ExecutionPlanMetricsSet,
        schema: SchemaRef,
    ) -> Option<Self> {
        let env = context.runtime_env();
        if !env.disk_manager.tmp_files_enabled() {
            return None;
        }
        let opts = &context.session_config().options().execution;
        let min_headroom_bytes = opts
            .sort_spill_reservation_bytes
            .min(MIN_SPILL_WORKING_SET_CAP);
        if min_headroom_bytes > 0 {
            let probe = MemoryConsumer::new("MorselSpillViability")
                .with_can_spill(true)
                .register(context.memory_pool());
            let viable = probe.try_grow(min_headroom_bytes).is_ok();
            drop(probe);
            if !viable {
                return None;
            }
        }
        let manager = SpillManager::new(env, SpillMetrics::new(plan_metrics, 0), schema)
            .with_compression_type(opts.spill_compression);
        Some(Self {
            manager,
            min_headroom_bytes,
            spilled: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(super) fn mark_spilled(&self) {
        self.spilled.store(true, Ordering::Release);
    }

    pub(super) fn is_spilled(&self) -> bool {
        self.spilled.load(Ordering::Acquire)
    }
}

/// Verify that `reservation` can temporarily admit `min_headroom_bytes` when no
/// other consumers hold memory.
#[allow(dead_code)]
pub(super) fn check_spill_headroom(
    reservation: &MemoryReservation,
    min_headroom_bytes: usize,
) -> Result<()> {
    if min_headroom_bytes == 0 {
        return Ok(());
    }
    let probe = reservation.new_empty();
    probe.try_grow(min_headroom_bytes)?;
    drop(probe);
    Ok(())
}

/// Spill non-empty `batches` asynchronously into a single finished `SpillFile`,
/// returning `(file, max_record_batch_memory)`. Aborts and discards the
/// in-progress file on error or cancellation (`tx.is_closed()`).
pub(super) async fn spill_batches_async(
    manager: &SpillManager,
    batches: &[RecordBatch],
    label: &str,
    tx: &mpsc::Sender<RecordBatch>,
) -> Result<Option<(Arc<dyn SpillFile>, usize)>> {
    if batches.iter().all(|b| b.num_rows() == 0) {
        return Ok(None);
    }
    let mut file = manager.create_in_progress_file(label)?;
    let mut max_batch_mem = 0usize;
    let res: Result<Option<Arc<dyn SpillFile>>> = async {
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            if tx.is_closed() {
                return Ok(None);
            }
            let batch_mem = file.append_batch_async(batch).await?;
            max_batch_mem = max_batch_mem.max(batch_mem);
        }
        if tx.is_closed() {
            return Ok(None);
        }
        file.finish_async().await
    }
    .await;

    match res {
        Ok(Some(finished)) => Ok(Some((finished, max_batch_mem))),
        Ok(None) => {
            let _ = file.abort_async().await;
            Ok(None)
        }
        Err(err) => {
            let _ = file.abort_async().await;
            Err(err)
        }
    }
}

/// Synchronous spill helper for `Operator` paths (e.g. `LocalProbeSpill`).
/// Returns `(file, max_record_batch_memory)`.
pub(super) fn spill_batches_sync(
    manager: &SpillManager,
    batches: &[RecordBatch],
    label: &str,
) -> Result<Option<(Arc<dyn SpillFile>, usize)>> {
    if batches.iter().all(|b| b.num_rows() == 0) {
        return Ok(None);
    }
    let mut file = manager.create_in_progress_file(label)?;
    let mut max_batch_mem = 0usize;
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let batch_mem = file.append_batch(batch)?;
        max_batch_mem = max_batch_mem.max(batch_mem);
    }
    Ok(file.finish()?.map(|f| (f, max_batch_mem)))
}
