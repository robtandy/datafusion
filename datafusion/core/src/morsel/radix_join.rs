// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Radix-partitioned build buffer, `LocalProbeSpill`, and spilled partition
//! manifests for morsel out-of-core `HashJoin`.

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow::compute::take_record_batch;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_common::config::SpillCompression;
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_execution::spill_file::SpillFile;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
use datafusion_physical_plan::spill::SpillManager;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::radix::{DEFAULT_RADIX_BITS, RadixChunk, level0_shift, radix_permute_indices};
use super::spill::{SharedPartitionWriters, SpillContext};

/// Switch from `R = 0` single-partition build to `R = 6` (64-way) radix build
/// once non-integer (`Utf8`/`Rows`) build rows exceed this threshold.
pub(super) const RADIX_JOIN_THRESHOLD_ROWS: usize = 16_384;

/// Compact 2-key integer builds (`Int32Pair`/`Int64Pair`) stay on the contiguous
/// `R = 0` `FlatIntTable` path up to this row count (L3-resident).
pub(super) const RADIX_JOIN_INT_PAIR_THRESHOLD_ROWS: usize = 262_144;

/// Compact single-column integer builds stay on the contiguous `R = 0` path up
/// to this row count so dense array-map tables (e.g. 750K `customer` @ 3 MiB)
/// avoid 64-way radix overhead while wide builds still switch via bytes budget.
pub(super) const RADIX_JOIN_SINGLE_INT_THRESHOLD_ROWS: usize = 1_048_576;

/// Memory ceiling above which even a smaller-row-count build transitions to
/// 64-way radix partitioning so wide-payload builds parallelize across workers.
pub(super) const RADIX_JOIN_THRESHOLD_BYTES: usize = 16 * 1024 * 1024;

/// Flush threshold for `LocalProbeSpill` per-worker buffered probe batches
/// (falls back immediately if `MemoryReservation::try_grow` fails earlier).
const PROBE_SPILL_FLUSH_BYTES: usize = 2 * 1024 * 1024;

/// Build canonical schema for radix-partitioned and spilled build batches:
/// `[payload_cols..., key_cols..., __morsel_hash: UInt64]`.
pub(super) fn build_join_spill_schema(
    payload_schema: &SchemaRef,
    key_types: &[DataType],
) -> SchemaRef {
    let mut fields: Vec<Arc<Field>> =
        Vec::with_capacity(payload_schema.fields().len() + key_types.len() + 1);
    for (idx, f) in payload_schema.fields().iter().enumerate() {
        fields.push(Arc::new(Field::new(
            format!("__p_{idx}"),
            f.data_type().clone(),
            true,
        )));
    }
    for (idx, dt) in key_types.iter().enumerate() {
        fields.push(Arc::new(Field::new(format!("__k_{idx}"), dt.clone(), true)));
    }
    fields.push(Arc::new(Field::new(
        "__morsel_hash",
        DataType::UInt64,
        false,
    )));
    Arc::new(Schema::new(fields))
}

/// Pack `(payload_opt, keys, hashes)` into one `RecordBatch` matching
/// `build_spill_schema`.
pub(super) fn pack_build_spill_batch(
    spill_schema: &SchemaRef,
    payload: Option<&RecordBatch>,
    keys: &[ArrayRef],
    hashes: Vec<u64>,
) -> Result<RecordBatch> {
    let num_rows = hashes.len();
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spill_schema.fields().len());
    if let Some(p) = payload {
        cols.extend(p.columns().iter().cloned());
    }
    cols.extend(keys.iter().cloned());
    cols.push(Arc::new(UInt64Array::from(hashes)) as ArrayRef);
    let opts = RecordBatchOptions::new().with_row_count(Some(num_rows));
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(spill_schema),
        cols,
        &opts,
    )?)
}

/// Unpack a `build_spill_schema` batch into `(payload_batch_opt, key_arrays, hash_slice)`.
pub(super) fn unpack_build_spill_batch(
    batch: &RecordBatch,
    payload_schema: &SchemaRef,
    num_payload_cols: usize,
    num_key_cols: usize,
) -> Result<(Option<RecordBatch>, Vec<ArrayRef>, Vec<u64>)> {
    let num_rows = batch.num_rows();
    let payload = if num_payload_cols == 0 {
        None
    } else {
        let p_cols = batch.columns()[..num_payload_cols].to_vec();
        let opts = RecordBatchOptions::new().with_row_count(Some(num_rows));
        Some(RecordBatch::try_new_with_options(
            Arc::clone(payload_schema),
            p_cols,
            &opts,
        )?)
    };
    let keys =
        batch.columns()[num_payload_cols..num_payload_cols + num_key_cols].to_vec();
    let hash_col = batch
        .column(num_payload_cols + num_key_cols)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            datafusion_common::internal_datafusion_err!(
                "Expected UInt64 __morsel_hash column in build spill batch"
            )
        })?;
    let hashes = hash_col.values().to_vec();
    Ok((payload, keys, hashes))
}

/// Worker-local (and later combined) radix-partitioned store for HashJoin build
/// batches and spill files.
pub(super) struct RadixBuildStore {
    pub(super) radix_bits: u32,
    pub(super) shift: u32,
    pub(super) mem_chunks: Vec<RadixChunk>,
    pub(super) mem_part_bytes: Vec<usize>,
    pub(super) part_rows: Vec<usize>,
    pub(super) part_est_bytes: Vec<usize>,
    pub(super) total_mem_bytes: usize,
    pub(super) spill_files: Vec<Vec<Arc<dyn SpillFile>>>,
    pub(super) spilled: bool,
}

impl RadixBuildStore {
    pub(super) fn new_default() -> Self {
        Self::new(DEFAULT_RADIX_BITS, level0_shift(DEFAULT_RADIX_BITS))
    }

    pub(super) fn new(radix_bits: u32, shift: u32) -> Self {
        let num_parts = 1usize << radix_bits;
        Self {
            radix_bits,
            shift,
            mem_chunks: Vec::new(),
            mem_part_bytes: vec![0; num_parts],
            part_rows: vec![0; num_parts],
            part_est_bytes: vec![0; num_parts],
            total_mem_bytes: 0,
            spill_files: vec![Vec::new(); num_parts],
            spilled: false,
        }
    }

    pub(super) fn num_partitions(&self) -> usize {
        self.part_rows.len()
    }

    /// Permute a packed build batch into `2^R` radix partition ranges, retaining
    /// one [`RadixChunk`] in memory when budget allows or spilling to disk on OOM.
    pub(super) fn push_packed_batch(
        &mut self,
        packed: RecordBatch,
        hashes: &[u64],
        extra_bytes: usize,
        reservation: &MemoryReservation,
        spill: Option<&SpillContext>,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if packed.num_rows() == 0 {
            return Ok(());
        }
        let (perm, ranges) = radix_permute_indices(hashes, self.shift, self.radix_bits);
        let permuted = take_record_batch(&packed, &perm)?;
        drop(packed);

        let batch_bytes = get_record_batch_memory_size(&permuted);
        let total_rows = permuted.num_rows().max(1);
        let target_bytes = self
            .total_mem_bytes
            .saturating_add(batch_bytes)
            .saturating_add(extra_bytes);

        // Buffer in memory until `try_resize` hits the pool ceiling (or
        // buffered bytes reach 16 MiB while in spill mode) so spilled builds
        // write large coalesced partition files instead of 64 tiny files/batch.
        const SPILLED_FLUSH_CAP_BYTES: usize = 16 * 1024 * 1024;
        let in_spill_mode = self.spilled || spill.is_some_and(SpillContext::is_spilled);
        let mut should_spill = in_spill_mode
            && self.total_mem_bytes.saturating_add(batch_bytes)
                >= SPILLED_FLUSH_CAP_BYTES;
        if !should_spill && let Err(err) = reservation.try_resize(target_bytes) {
            if spill.is_none() {
                return Err(err);
            }
            should_spill = true;
        }

        if should_spill {
            let Some(spill_ctx) = spill else {
                return Err(datafusion_common::resources_datafusion_err!(
                    "Morsel radix HashJoin spill requested without an active SpillContext"
                ));
            };
            self.spilled = true;
            spill_ctx.mark_spilled();
            let num_parts = self.num_partitions();
            let writers = spill_ctx
                .get_or_init_partition_writers(num_parts, "MorselHashJoinBuildSpill");
            let prev_chunks = std::mem::take(&mut self.mem_chunks);
            for (p, &(offset, len)) in ranges.iter().enumerate() {
                if tx.is_closed() {
                    self.clear_mem_batches();
                    reservation.resize(extra_bytes);
                    return Ok(());
                }
                let mut to_spill = Vec::new();
                if self.mem_part_bytes[p] > 0 || self.part_rows[p] > 0 {
                    for c in &prev_chunks {
                        if let Some(s) = c.part_slice(p) {
                            to_spill.push(s);
                        }
                    }
                }
                if len > 0 {
                    to_spill.push(permuted.slice(offset, len));
                    self.part_rows[p] = self.part_rows[p].saturating_add(len);
                    let share = batch_bytes.saturating_mul(len) / total_rows;
                    self.part_est_bytes[p] = self.part_est_bytes[p].saturating_add(share);
                }
                self.mem_part_bytes[p] = 0;
                if !to_spill.is_empty() {
                    writers.append_batches(p, &to_spill)?;
                }
            }
            drop(prev_chunks);
            self.total_mem_bytes = 0;
            drop(permuted);
            reservation.resize(extra_bytes);
            return Ok(());
        }

        for (p, &(_, len)) in ranges.iter().enumerate() {
            if len > 0 {
                let share = batch_bytes.saturating_mul(len) / total_rows;
                self.mem_part_bytes[p] = self.mem_part_bytes[p].saturating_add(share);
                self.part_rows[p] = self.part_rows[p].saturating_add(len);
                self.part_est_bytes[p] = self.part_est_bytes[p].saturating_add(share);
            }
        }
        self.mem_chunks
            .push(RadixChunk::from_permuted(permuted, &ranges));
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(batch_bytes);
        Ok(())
    }

    pub(super) fn spill_all_mem(
        &mut self,
        extra_bytes: usize,
        reservation: &MemoryReservation,
        spill: &SpillContext,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        self.spilled = true;
        spill.mark_spilled();
        let num_parts = self.num_partitions();
        let writers =
            spill.get_or_init_partition_writers(num_parts, "MorselHashJoinBuildSpill");
        let chunks = std::mem::take(&mut self.mem_chunks);
        for p in 0..num_parts {
            if tx.is_closed() {
                self.clear_mem_batches();
                reservation.resize(extra_bytes);
                return Ok(());
            }
            if self.part_rows[p] > 0 {
                let batches: Vec<RecordBatch> =
                    chunks.iter().filter_map(|c| c.part_slice(p)).collect();
                if !batches.is_empty() {
                    writers.append_batches(p, &batches)?;
                }
            }
            self.mem_part_bytes[p] = 0;
        }
        drop(chunks);
        self.total_mem_bytes = 0;
        reservation.resize(extra_bytes);
        Ok(())
    }

    fn clear_mem_batches(&mut self) {
        self.mem_chunks.clear();
        self.mem_part_bytes.fill(0);
        self.total_mem_bytes = 0;
    }

    /// O(`2^R`) lock-free combine across build workers.
    pub(super) fn combine_from(&mut self, mut other: Self) {
        debug_assert_eq!(self.radix_bits, other.radix_bits);
        debug_assert_eq!(self.shift, other.shift);
        self.spilled |= other.spilled;
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(other.total_mem_bytes);
        self.mem_chunks.append(&mut other.mem_chunks);
        for p in 0..self.num_partitions() {
            self.mem_part_bytes[p] =
                self.mem_part_bytes[p].saturating_add(other.mem_part_bytes[p]);
            self.part_rows[p] = self.part_rows[p].saturating_add(other.part_rows[p]);
            self.part_est_bytes[p] =
                self.part_est_bytes[p].saturating_add(other.part_est_bytes[p]);
            self.spill_files[p].append(&mut other.spill_files[p]);
        }
    }
}

/// Take-once spilled build partition slot.
pub(super) struct SpilledBuildPartition {
    pub(super) files: Vec<Arc<dyn SpillFile>>,
}

/// Retained NULL-key build rows for `BuildOuter` / `FullOuter` / `BuildAnti`
/// (in-memory or spilled to disk).
pub(super) struct NullKeyBuildRows {
    pub(super) mem_batches: Vec<RecordBatch>,
    pub(super) spill_files: Vec<Arc<dyn SpillFile>>,
    pub(super) _reservation: MemoryReservation,
}

/// Lightweight factory passed to probe workers so `LocalProbeSpill` can lazily
/// create a `SpillManager` with the exact probe input `SchemaRef`.
#[derive(Clone)]
pub(super) struct ProbeSpillEnv {
    env: Arc<RuntimeEnv>,
    plan_metrics: ExecutionPlanMetricsSet,
    compression: SpillCompression,
}

impl ProbeSpillEnv {
    pub(super) fn new(
        context: &TaskContext,
        plan_metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        Self {
            env: context.runtime_env(),
            plan_metrics: plan_metrics.clone(),
            compression: context
                .session_config()
                .options()
                .execution
                .spill_compression,
        }
    }

    pub(super) fn create_manager(&self, schema: SchemaRef) -> SpillManager {
        SpillManager::new(
            Arc::clone(&self.env),
            SpillMetrics::new(&self.plan_metrics, 0),
            schema,
        )
        .with_compression_type(self.compression)
    }
}

pub(super) type FinishedProbeSpills =
    (Option<SpillManager>, Vec<Vec<Arc<dyn SpillFile>>>);

/// Shared manifest holding at most one reusable appendable probe spill writer
/// per radix partition across all `P1` probe workers (and any upstream `P2`
/// workers feeding a downstream spilled join).
pub(super) struct SpilledProbeManifest {
    num_parts: usize,
    manager: Option<SpillManager>,
    writers: Option<Arc<SharedPartitionWriters>>,
}

impl SpilledProbeManifest {
    pub(super) fn new(num_parts: usize) -> Self {
        Self {
            num_parts,
            manager: None,
            writers: None,
        }
    }

    fn get_or_init_writers(
        &mut self,
        env: &ProbeSpillEnv,
        schema: SchemaRef,
    ) -> Arc<SharedPartitionWriters> {
        if let Some(existing) = &self.writers {
            return Arc::clone(existing);
        }
        let mgr = env.create_manager(schema);
        let writers = Arc::new(SharedPartitionWriters::new(
            mgr.clone(),
            self.num_parts,
            "MorselHashJoinProbeSpill",
        ));
        self.manager = Some(mgr);
        self.writers = Some(Arc::clone(&writers));
        writers
    }

    /// Finish all open probe partition writers, closing write FDs and
    /// returning `(manager, files_by_part)` with at most 1 file per partition.
    pub(super) fn finish_all(&mut self) -> Result<FinishedProbeSpills> {
        let mgr = self.manager.take();
        let mut files_by_part = vec![Vec::new(); self.num_parts];
        if let Some(writers) = self.writers.take() {
            for (p, sf_opt) in writers.finish_all()?.into_iter().enumerate() {
                if let Some(sf) = sf_opt {
                    files_by_part[p].push(sf);
                }
            }
        }
        Ok((mgr, files_by_part))
    }
}

/// Worker-local synchronous probe spill buffer (DuckDB `ProbeSpill` pattern).
/// Flushes into the join's shared per-partition writers so open spill files
/// stay bounded by `num_parts` regardless of worker count or flush count.
pub(super) struct LocalProbeSpill {
    writers: Option<Arc<SharedPartitionWriters>>,
    buffers: Vec<Vec<RecordBatch>>,
    buffered_bytes: usize,
}

impl LocalProbeSpill {
    pub(super) fn new(num_parts: usize) -> Self {
        Self {
            writers: None,
            buffers: vec![Vec::new(); num_parts],
            buffered_bytes: 0,
        }
    }

    pub(super) fn push_batch(
        &mut self,
        part: usize,
        batch: RecordBatch,
        env: &ProbeSpillEnv,
        manifest: &Mutex<SpilledProbeManifest>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if self.writers.is_none() {
            let w = manifest.lock().get_or_init_writers(env, batch.schema());
            self.writers = Some(w);
        }
        let bytes = get_record_batch_memory_size(&batch);
        if self.buffered_bytes.saturating_add(bytes) >= PROBE_SPILL_FLUSH_BYTES
            || reservation.try_grow(bytes).is_err()
        {
            self.flush_buffers(reservation)?;
            // If this single batch still exceeds remaining memory, append it directly.
            if reservation.try_grow(bytes).is_err() {
                let writers = self.writers.as_ref().expect("probe spill writers");
                writers.append_batches(part, std::slice::from_ref(&batch))?;
                return Ok(());
            }
        }
        self.buffers[part].push(batch);
        self.buffered_bytes = self.buffered_bytes.saturating_add(bytes);
        Ok(())
    }

    pub(super) fn flush_buffers(
        &mut self,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        if self.buffered_bytes == 0 {
            return Ok(());
        }
        let Some(writers) = self.writers.as_ref() else {
            return Ok(());
        };
        for p in 0..self.buffers.len() {
            let batches = std::mem::take(&mut self.buffers[p]);
            if !batches.is_empty() {
                writers.append_batches(p, &batches)?;
            }
        }
        reservation.shrink(self.buffered_bytes);
        self.buffered_bytes = 0;
        Ok(())
    }

    pub(super) fn finish_into(mut self, reservation: &MemoryReservation) -> Result<()> {
        self.flush_buffers(reservation)?;
        self.writers = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::DataFusionError;
    use datafusion_execution::memory_pool::{
        MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
    };
    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Debug)]
    struct FailFirstTryGrowPool {
        failed_once: AtomicBool,
        reserved: AtomicUsize,
    }

    impl FailFirstTryGrowPool {
        fn new() -> Self {
            Self {
                failed_once: AtomicBool::new(false),
                reserved: AtomicUsize::new(0),
            }
        }
    }

    impl fmt::Display for FailFirstTryGrowPool {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "FailFirstTryGrowPool")
        }
    }

    impl MemoryPool for FailFirstTryGrowPool {
        fn name(&self) -> &str {
            "FailFirstTryGrowPool"
        }

        fn grow(&self, _reservation: &MemoryReservation, additional: usize) {
            self.reserved.fetch_add(additional, Ordering::Relaxed);
        }

        fn shrink(&self, _reservation: &MemoryReservation, shrink: usize) {
            self.reserved.fetch_sub(shrink, Ordering::Relaxed);
        }

        fn try_grow(
            &self,
            _reservation: &MemoryReservation,
            additional: usize,
        ) -> Result<()> {
            if !self.failed_once.swap(true, Ordering::SeqCst) {
                return Err(datafusion_common::resources_datafusion_err!(
                    "injected first try_grow OOM"
                ));
            }
            self.reserved.fetch_add(additional, Ordering::Relaxed);
            Ok(())
        }

        fn reserved(&self) -> usize {
            self.reserved.load(Ordering::Relaxed)
        }

        fn memory_limit(&self) -> MemoryLimit {
            MemoryLimit::Finite(usize::MAX)
        }
    }

    #[test]
    fn push_packed_batch_without_spill_returns_oom_without_panicking_when_retry_would_succeed()
     {
        let pool: Arc<dyn MemoryPool> = Arc::new(FailFirstTryGrowPool::new());
        let reservation = MemoryConsumer::new("TestRadixJoin")
            .with_can_spill(true)
            .register(&pool);
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("__morsel_hash", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(UInt64Array::from(vec![10u64, 20, 30, 40])),
            ],
        )
        .unwrap();
        let hashes = vec![10u64, 20, 30, 40];
        let (tx, _rx) = mpsc::channel(1);
        let mut store = RadixBuildStore::new_default();

        let err = store
            .push_packed_batch(batch, &hashes, 0, &reservation, None, &tx)
            .expect_err("expected ResourcesExhausted without panicking");
        assert!(
            matches!(err, DataFusionError::ResourcesExhausted(_)),
            "unexpected error: {err:?}"
        );
        assert_eq!(reservation.size(), 0);
        assert_eq!(pool.reserved(), 0);
    }
}
