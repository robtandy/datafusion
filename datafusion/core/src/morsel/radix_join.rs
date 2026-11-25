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
use tokio::sync::mpsc;

use super::radix::{DEFAULT_RADIX_BITS, level0_shift, radix_permute_indices};
use super::spill::{SpillContext, spill_batches_async, spill_batches_sync};

/// Switch from `R = 0` single-partition build to `R = 6` (64-way) radix build
/// once build rows exceed this threshold (keeps tiny dimension builds fast).
pub(super) const RADIX_JOIN_THRESHOLD_ROWS: usize = 16_384;

/// Flush threshold for `LocalProbeSpill` per-worker buffered probe batches.
const PROBE_SPILL_FLUSH_BYTES: usize = 256 * 1024;

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
    fields.push(Arc::new(Field::new("__morsel_hash", DataType::UInt64, false)));
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
    let keys = batch.columns()[num_payload_cols..num_payload_cols + num_key_cols].to_vec();
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
    pub(super) mem_batches: Vec<Vec<RecordBatch>>,
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
            mem_batches: vec![Vec::new(); num_parts],
            mem_part_bytes: vec![0; num_parts],
            part_rows: vec![0; num_parts],
            part_est_bytes: vec![0; num_parts],
            total_mem_bytes: 0,
            spill_files: vec![Vec::new(); num_parts],
            spilled: false,
        }
    }

    pub(super) fn num_partitions(&self) -> usize {
        self.mem_batches.len()
    }

    /// Permute a packed build batch into `2^R` radix slices, retaining slices
    /// in memory when budget allows or spilling to disk on OOM.
    #[expect(clippy::too_many_arguments)]
    pub(super) async fn push_packed_batch(
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

        let should_spill = self.spilled
            || spill.is_some_and(SpillContext::is_spilled)
            || reservation.try_resize(target_bytes).is_err();

        if should_spill {
            let Some(spill_ctx) = spill else {
                reservation.try_resize(target_bytes)?;
                unreachable!()
            };
            self.spilled = true;
            spill_ctx.mark_spilled();
            for (p, &(offset, len)) in ranges.iter().enumerate() {
                if tx.is_closed() {
                    self.clear_mem_batches();
                    reservation.resize(extra_bytes);
                    return Ok(());
                }
                let mut to_spill = std::mem::take(&mut self.mem_batches[p]);
                if len > 0 {
                    to_spill.push(permuted.slice(offset, len));
                    self.part_rows[p] = self.part_rows[p].saturating_add(len);
                    let share = batch_bytes.saturating_mul(len) / total_rows;
                    self.part_est_bytes[p] =
                        self.part_est_bytes[p].saturating_add(share);
                }
                self.mem_part_bytes[p] = 0;
                if !to_spill.is_empty()
                    && let Some((sf, _)) = spill_batches_async(
                        &spill_ctx.manager,
                        &to_spill,
                        "MorselHashJoinBuildSpill",
                        tx,
                    )
                    .await?
                {
                    self.spill_files[p].push(sf);
                }
            }
            self.total_mem_bytes = 0;
            drop(permuted);
            reservation.resize(extra_bytes);
            return Ok(());
        }

        for (p, &(offset, len)) in ranges.iter().enumerate() {
            if len > 0 {
                self.mem_batches[p].push(permuted.slice(offset, len));
                let share = batch_bytes.saturating_mul(len) / total_rows;
                self.mem_part_bytes[p] = self.mem_part_bytes[p].saturating_add(share);
                self.part_rows[p] = self.part_rows[p].saturating_add(len);
                self.part_est_bytes[p] = self.part_est_bytes[p].saturating_add(share);
            }
        }
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(batch_bytes);
        Ok(())
    }

    pub(super) async fn spill_all_mem(
        &mut self,
        extra_bytes: usize,
        reservation: &MemoryReservation,
        spill: &SpillContext,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        self.spilled = true;
        spill.mark_spilled();
        for p in 0..self.num_partitions() {
            if tx.is_closed() {
                self.clear_mem_batches();
                reservation.resize(extra_bytes);
                return Ok(());
            }
            let batches = std::mem::take(&mut self.mem_batches[p]);
            self.mem_part_bytes[p] = 0;
            if !batches.is_empty()
                && let Some((sf, _)) = spill_batches_async(
                    &spill.manager,
                    &batches,
                    "MorselHashJoinBuildSpill",
                    tx,
                )
                .await?
            {
                self.spill_files[p].push(sf);
            }
        }
        self.total_mem_bytes = 0;
        reservation.resize(extra_bytes);
        Ok(())
    }

    fn clear_mem_batches(&mut self) {
        for v in &mut self.mem_batches {
            v.clear();
        }
        self.mem_part_bytes.fill(0);
        self.total_mem_bytes = 0;
    }

    /// O(`2^R`) lock-free combine across build workers.
    pub(super) fn combine_from(&mut self, mut other: Self) {
        debug_assert_eq!(self.radix_bits, other.radix_bits);
        debug_assert_eq!(self.shift, other.shift);
        self.spilled |= other.spilled;
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(other.total_mem_bytes);
        for p in 0..self.num_partitions() {
            self.mem_batches[p].append(&mut other.mem_batches[p]);
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

/// Shared manifest populated by `P1` probe workers on EOF (and any upstream
/// `P2` build-emission/spill workers feeding a downstream spilled join).
pub(super) struct SpilledProbeManifest {
    pub(super) manager: Option<SpillManager>,
    pub(super) files_by_part: Vec<Vec<Arc<dyn SpillFile>>>,
}

impl SpilledProbeManifest {
    pub(super) fn new(num_parts: usize) -> Self {
        Self {
            manager: None,
            files_by_part: vec![Vec::new(); num_parts],
        }
    }

    pub(super) fn merge_from(
        &mut self,
        manager: Option<SpillManager>,
        mut files: Vec<Vec<Arc<dyn SpillFile>>>,
    ) {
        if self.manager.is_none() {
            self.manager = manager;
        }
        for (dst, src) in self.files_by_part.iter_mut().zip(files.iter_mut()) {
            dst.append(src);
        }
    }
}

/// Worker-local synchronous probe spill buffer (DuckDB `ProbeSpill` pattern).
pub(super) struct LocalProbeSpill {
    manager: Option<SpillManager>,
    buffers: Vec<Vec<RecordBatch>>,
    buffered_bytes: usize,
    files: Vec<Vec<Arc<dyn SpillFile>>>,
}

impl LocalProbeSpill {
    pub(super) fn new(num_parts: usize) -> Self {
        Self {
            manager: None,
            buffers: vec![Vec::new(); num_parts],
            buffered_bytes: 0,
            files: vec![Vec::new(); num_parts],
        }
    }

    pub(super) fn push_batch(
        &mut self,
        part: usize,
        batch: RecordBatch,
        env: &ProbeSpillEnv,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if self.manager.is_none() {
            self.manager = Some(env.create_manager(batch.schema()));
        }
        let bytes = get_record_batch_memory_size(&batch);
        if self.buffered_bytes.saturating_add(bytes) >= PROBE_SPILL_FLUSH_BYTES
            || reservation.try_grow(bytes).is_err()
        {
            self.flush_buffers(reservation)?;
            // If this single batch still exceeds remaining memory, spill it directly.
            if reservation.try_grow(bytes).is_err() {
                let mgr = self.manager.as_ref().expect("probe spill manager");
                if let Some((sf, _)) = spill_batches_sync(
                    mgr,
                    std::slice::from_ref(&batch),
                    "MorselHashJoinProbeSpill",
                )? {
                    self.files[part].push(sf);
                }
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
        let Some(mgr) = self.manager.as_ref() else {
            return Ok(());
        };
        for p in 0..self.buffers.len() {
            let batches = std::mem::take(&mut self.buffers[p]);
            if !batches.is_empty()
                && let Some((sf, _)) =
                    spill_batches_sync(mgr, &batches, "MorselHashJoinProbeSpill")?
            {
                self.files[p].push(sf);
            }
        }
        reservation.shrink(self.buffered_bytes);
        self.buffered_bytes = 0;
        Ok(())
    }

    pub(super) fn finish_into(
        mut self,
        manifest: &mut SpilledProbeManifest,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        self.flush_buffers(reservation)?;
        manifest.merge_from(self.manager.take(), std::mem::take(&mut self.files));
        Ok(())
    }
}
