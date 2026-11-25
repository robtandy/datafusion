// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Shared `SpillManager` helpers, reusable per-partition appendable writers
//! (bounding open file descriptors to `O(partitions)` instead of
//! `partitions × workers × spill_events`), and cancellation-safe spill I/O
//! wrappers for morsel `GROUP BY`, `HashJoin`, and `Sort`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_execution::spill_file::SpillFile;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
use datafusion_physical_plan::spill::SpillManager;
use parking_lot::Mutex;
use tokio::sync::mpsc;

/// Cap for the minimum memory-pool viability check before enabling spilling.
/// Respected alongside `execution.sort_spill_reservation_bytes` so pools that
/// cannot even hold minimal IPC framing + one radix partition fail fast with
/// `ResourcesExhausted` instead of thrashing.
const MIN_SPILL_WORKING_SET_CAP: usize = 256 * 1024;

enum SyncSpillCmd<'a> {
    Append(&'a RecordBatch),
    Finish,
}

enum SyncSpillOut {
    Appended(usize),
    Finished(Option<Arc<dyn SpillFile>>),
}

/// Type-erased reusable appendable wrapper around `InProgressSpillFile` (whose
/// module is `pub(crate)` in `datafusion-physical-plan`). Dropping this struct
/// without calling `finish()` synchronously closes the IPC writer and unlinks
/// the underlying `RefCountedTempFile`.
pub(super) struct SyncSpillWriter {
    op: Box<dyn FnMut(SyncSpillCmd<'_>) -> Result<SyncSpillOut> + Send>,
    max_batch_mem: usize,
}

impl std::fmt::Debug for SyncSpillWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncSpillWriter")
            .field("max_batch_mem", &self.max_batch_mem)
            .finish()
    }
}

impl SyncSpillWriter {
    pub(super) fn try_new(manager: &SpillManager, label: &str) -> Result<Self> {
        let mut file = manager.create_in_progress_file(label)?;
        Ok(Self {
            op: Box::new(move |cmd| match cmd {
                SyncSpillCmd::Append(batch) => {
                    let mem = file.append_batch(batch)?;
                    Ok(SyncSpillOut::Appended(mem))
                }
                SyncSpillCmd::Finish => Ok(SyncSpillOut::Finished(file.finish()?)),
            }),
            max_batch_mem: 0,
        })
    }

    pub(super) fn append_batch(&mut self, batch: &RecordBatch) -> Result<usize> {
        if batch.num_rows() == 0 {
            return Ok(0);
        }
        match (self.op)(SyncSpillCmd::Append(batch))? {
            SyncSpillOut::Appended(mem) => {
                self.max_batch_mem = self.max_batch_mem.max(mem);
                Ok(mem)
            }
            SyncSpillOut::Finished(_) => unreachable!(),
        }
    }

    pub(super) fn append_batches(&mut self, batches: &[RecordBatch]) -> Result<()> {
        for batch in batches {
            if batch.num_rows() > 0 {
                self.append_batch(batch)?;
            }
        }
        Ok(())
    }

    pub(super) fn finish(&mut self) -> Result<Option<(Arc<dyn SpillFile>, usize)>> {
        match (self.op)(SyncSpillCmd::Finish)? {
            SyncSpillOut::Finished(opt) => Ok(opt.map(|f| (f, self.max_batch_mem))),
            SyncSpillOut::Appended(_) => unreachable!(),
        }
    }
}

/// Shared pool of at most `num_parts` reusable appendable writers across all
/// workers and spill events of one radix stage. Each partition has its own
/// `Mutex`, avoiding cross-partition contention while guaranteeing peak open
/// spill files never exceed `num_parts`.
pub(super) struct SharedPartitionWriters {
    manager: SpillManager,
    label: &'static str,
    writers: Vec<Mutex<Option<SyncSpillWriter>>>,
}

impl std::fmt::Debug for SharedPartitionWriters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedPartitionWriters")
            .field("label", &self.label)
            .field("num_parts", &self.writers.len())
            .finish()
    }
}

impl SharedPartitionWriters {
    pub(super) fn new(
        manager: SpillManager,
        num_parts: usize,
        label: &'static str,
    ) -> Self {
        Self {
            manager,
            label,
            writers: (0..num_parts).map(|_| Mutex::new(None)).collect(),
        }
    }

    pub(super) fn append_batches(
        &self,
        part: usize,
        batches: &[RecordBatch],
    ) -> Result<()> {
        if batches.iter().all(|b| b.num_rows() == 0) {
            return Ok(());
        }
        let mut guard = self.writers[part].lock();
        if guard.is_none() {
            *guard = Some(SyncSpillWriter::try_new(&self.manager, self.label)?);
        }
        guard.as_mut().unwrap().append_batches(batches)
    }

    pub(super) fn finish_all(&self) -> Result<PartitionSpillFiles> {
        let mut out = Vec::with_capacity(self.writers.len());
        for slot in &self.writers {
            let finished = match slot.lock().take() {
                Some(mut writer) => writer.finish()?.map(|(f, _)| f),
                None => None,
            };
            out.push(finished);
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
pub(super) struct SpillContext {
    pub(super) manager: SpillManager,
    pub(super) memory_pool: Arc<dyn MemoryPool>,
    pub(super) min_headroom_bytes: usize,
    spilled: Arc<AtomicBool>,
    partition_writers: Arc<Mutex<Option<Arc<SharedPartitionWriters>>>>,
    single_writer: Arc<Mutex<Option<SyncSpillWriter>>>,
}

pub(super) type PartitionSpillFiles = Vec<Option<Arc<dyn SpillFile>>>;

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
            memory_pool: Arc::clone(context.memory_pool()),
            min_headroom_bytes,
            spilled: Arc::new(AtomicBool::new(false)),
            partition_writers: Arc::new(Mutex::new(None)),
            single_writer: Arc::new(Mutex::new(None)),
        })
    }

    pub(super) fn mark_spilled(&self) {
        self.spilled.store(true, Ordering::Release);
    }

    pub(super) fn is_spilled(&self) -> bool {
        self.spilled.load(Ordering::Acquire)
    }

    /// Return the stage-shared `SharedPartitionWriters` (lazily created once
    /// for `num_parts` partitions so all workers append to at most 1 spill
    /// file per partition).
    pub(super) fn get_or_init_partition_writers(
        &self,
        num_parts: usize,
        label: &'static str,
    ) -> Arc<SharedPartitionWriters> {
        let mut guard = self.partition_writers.lock();
        if let Some(existing) = guard.as_ref() {
            debug_assert_eq!(existing.writers.len(), num_parts);
            return Arc::clone(existing);
        }
        let created = Arc::new(SharedPartitionWriters::new(
            self.manager.clone(),
            num_parts,
            label,
        ));
        *guard = Some(Arc::clone(&created));
        created
    }

    /// Finish all stage-shared partition writers, closing write FDs and
    /// returning at most one finished `SpillFile` per partition.
    pub(super) fn finish_partition_writers(&self) -> Result<Option<PartitionSpillFiles>> {
        let writers = self.partition_writers.lock().take();
        match writers {
            Some(w) => Ok(Some(w.finish_all()?)),
            None => Ok(None),
        }
    }

    /// Append batches to the stage-shared single spill writer (e.g. NULL-key
    /// build rows), keeping at most 1 open spill file across all workers.
    pub(super) fn append_single(
        &self,
        label: &'static str,
        batches: &[RecordBatch],
    ) -> Result<()> {
        if batches.iter().all(|b| b.num_rows() == 0) {
            return Ok(());
        }
        let mut guard = self.single_writer.lock();
        if guard.is_none() {
            *guard = Some(SyncSpillWriter::try_new(&self.manager, label)?);
        }
        guard.as_mut().unwrap().append_batches(batches)
    }

    /// Finish the stage-shared single spill writer, returning at most one
    /// finished `SpillFile`.
    pub(super) fn finish_single(&self) -> Result<Option<Arc<dyn SpillFile>>> {
        let writer = self.single_writer.lock().take();
        match writer {
            Some(mut w) => Ok(w.finish()?.map(|(f, _)| f)),
            None => Ok(None),
        }
    }
}

/// Verify that `reservation` can temporarily admit `min_headroom_bytes` when no
/// other consumers hold memory.
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
