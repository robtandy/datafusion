// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Shared radix-partitioned `GROUP BY` buffer, parallel partition finalize,
//! out-of-core partition spilling, recursive radix splitting, and sorted-run
//! skew fallback for `AggregateSink` and `GroupCountSink`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::{SortOptions, lexsort_to_indices, take_record_batch};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_common::{DataFusionError, Result, exec_datafusion_err, float_semantics};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_execution::spill_file::SpillFile;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::sorts::streaming_merge::{
    SortedSpillFile, StreamingMergeBuilder,
};
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::metrics::{self, RunMetrics};
use super::radix::{
    DEFAULT_RADIX_BITS, MAX_RADIX_DEPTH, RECURSIVE_RADIX_BITS, RadixChunk, level0_shift,
    next_level_shift, radix_permute_indices,
};
use super::spill::{SpillContext, SyncSpillWriter, spill_batches_async};
use super::{Operator, abort_and_drain, apply_operators};

/// Stay on the unpartitioned `R = 0` fast path while worker-local distinct
/// group count stays at or below this threshold (e.g., TPC-H Q1/Q9/Q12).
pub(super) const RADIX_TRANSITION_GROUPS: usize = 4_096;

/// Flush the worker-local L2 pre-aggregation cache into radix partitions once
/// it reaches this many groups (or `PREAGG_FLUSH_BYTES`).
pub(super) const PREAGG_FLUSH_GROUPS: usize = 16_384;
pub(super) const PREAGG_FLUSH_BYTES: usize = 4 * 1024 * 1024;

/// Fixed seed shared across all workers and recursive levels for `GROUP BY`
/// radix hashing.
const GROUP_HASH_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// Deterministic hash used for `NULL` single-integer group keys.
pub(super) const NULL_GROUP_HASH: u64 = 0x9e37_79b9_7f4a_7c15;

pub(super) fn group_random_state() -> RandomState {
    RandomState::with_seed(GROUP_HASH_SEED)
}

pub(super) fn has_float_group_type(schema: &SchemaRef) -> bool {
    schema.fields().iter().any(|f| {
        matches!(
            f.data_type(),
            DataType::Float16 | DataType::Float32 | DataType::Float64
        )
    })
}

/// Hash group key columns with zero-initialized buffer and `-0.0 -> +0.0` float
/// normalization so equal keys (including `NULL`s and `-0.0`/`+0.0`) always
/// route to the exact same radix partition.
pub(super) fn hash_group_columns(
    cols: &[ArrayRef],
    has_floats: bool,
    hashes: &mut Vec<u64>,
) -> Result<()> {
    let num_rows = cols.first().map_or(0, |c| c.len());
    hashes.resize(num_rows, 0);
    hashes.fill(0);
    if num_rows == 0 {
        return Ok(());
    }
    let state = group_random_state();
    if has_floats {
        let normalized: Vec<ArrayRef> = cols
            .iter()
            .map(|c| {
                if matches!(
                    c.data_type(),
                    DataType::Float16 | DataType::Float32 | DataType::Float64
                ) {
                    float_semantics::normalize(Arc::clone(c))
                } else {
                    Arc::clone(c)
                }
            })
            .collect();
        create_hashes(&normalized, &state, hashes)?;
    } else {
        create_hashes(cols, &state, hashes)?;
    }
    Ok(())
}

/// Worker-local (and later combined) radix-partitioned store of partial
/// aggregate state batches and spill files.
pub(super) struct RadixPartitionStore {
    pub(super) radix_bits: u32,
    pub(super) shift: u32,
    pub(super) mem_chunks: Vec<RadixChunk>,
    pub(super) part_non_empty: Vec<bool>,
    pub(super) mem_part_bytes: Vec<usize>,
    pub(super) total_mem_bytes: usize,
    pub(super) spill_files: Vec<Vec<Arc<dyn SpillFile>>>,
    pub(super) spilled: bool,
}

impl RadixPartitionStore {
    pub(super) fn new_default() -> Self {
        Self::new(DEFAULT_RADIX_BITS, level0_shift(DEFAULT_RADIX_BITS))
    }

    pub(super) fn new(radix_bits: u32, shift: u32) -> Self {
        let num_parts = 1usize << radix_bits;
        Self {
            radix_bits,
            shift,
            mem_chunks: Vec::new(),
            part_non_empty: vec![false; num_parts],
            mem_part_bytes: vec![0; num_parts],
            total_mem_bytes: 0,
            spill_files: vec![Vec::new(); num_parts],
            spilled: false,
        }
    }

    pub(super) fn num_partitions(&self) -> usize {
        self.mem_part_bytes.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        !self.spilled
            && self.mem_chunks.is_empty()
            && self.spill_files.iter().all(Vec::is_empty)
    }

    /// Permute a partial-state `RecordBatch` by radix partition in one O(N)
    /// pass and either retain a single [`RadixChunk`] in memory or spill on OOM.
    ///
    /// `extra_bytes` is the caller's other live reservation bytes (e.g. empty
    /// pre-agg table overhead) so `reservation.try_resize` tracks total bytes.
    pub(super) fn push_partial_batch(
        &mut self,
        partial_batch: RecordBatch,
        hashes: &[u64],
        extra_bytes: usize,
        reservation: &MemoryReservation,
        spill: Option<&SpillContext>,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if partial_batch.num_rows() == 0 {
            return Ok(());
        }
        let (perm, ranges) = radix_permute_indices(hashes, self.shift, self.radix_bits);
        let permuted = take_record_batch(&partial_batch, &perm)?;
        drop(partial_batch);

        // If we are already spilling (or trying to grow fails), spill cleanly.
        let batch_bytes = get_record_batch_memory_size(&permuted);
        let target_bytes = self
            .total_mem_bytes
            .saturating_add(batch_bytes)
            .saturating_add(extra_bytes);

        // Buffer in memory until `try_resize` actually hits the pool ceiling
        // (or buffered bytes reach 16 MiB while in spill mode) so spilled runs
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
                    "Morsel radix GROUP BY spill requested without an active SpillContext"
                ));
            };
            self.spilled = true;
            spill_ctx.mark_spilled();
            // Append previously buffered chunks together with the new batch's
            // per-partition slices into the stage-shared partition writers
            // (at most 1 open spill file per partition across all workers!).
            let num_parts = self.num_partitions();
            let writers =
                spill_ctx.get_or_init_partition_writers(num_parts, "MorselGroupBySpill");
            let prev_chunks = std::mem::take(&mut self.mem_chunks);
            for (p, &(offset, len)) in ranges.iter().enumerate() {
                if tx.is_closed() {
                    self.clear_mem_batches();
                    reservation.resize(extra_bytes);
                    return Ok(());
                }
                let mut to_spill = Vec::new();
                if self.part_non_empty[p] {
                    for c in &prev_chunks {
                        if let Some(s) = c.part_slice(p) {
                            to_spill.push(s);
                        }
                    }
                }
                if len > 0 {
                    to_spill.push(permuted.slice(offset, len));
                }
                if !to_spill.is_empty() {
                    writers.append_batches(p, &to_spill)?;
                }
                self.part_non_empty[p] = false;
                self.mem_part_bytes[p] = 0;
            }
            drop(prev_chunks);
            self.total_mem_bytes = 0;
            drop(permuted);
            reservation.resize(extra_bytes);
            return Ok(());
        }

        let total_rows = permuted.num_rows().max(1);
        for (p, &(_, len)) in ranges.iter().enumerate() {
            if len > 0 {
                self.part_non_empty[p] = true;
                // Proportional share so releasing individual partitions shrinks
                // the combined reservation smoothly without double-counting.
                let share = batch_bytes.saturating_mul(len) / total_rows;
                self.mem_part_bytes[p] = self.mem_part_bytes[p].saturating_add(share);
            }
        }
        self.mem_chunks
            .push(RadixChunk::from_permuted(permuted, &ranges));
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(batch_bytes);
        Ok(())
    }

    /// Spill all currently buffered in-memory chunks in this store and reset
    /// `total_mem_bytes` to 0.
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
            spill.get_or_init_partition_writers(num_parts, "MorselGroupBySpill");
        let chunks = std::mem::take(&mut self.mem_chunks);
        for p in 0..num_parts {
            if tx.is_closed() {
                self.clear_mem_batches();
                reservation.resize(extra_bytes);
                return Ok(());
            }
            if self.part_non_empty[p] {
                let batches: Vec<RecordBatch> =
                    chunks.iter().filter_map(|c| c.part_slice(p)).collect();
                if !batches.is_empty() {
                    writers.append_batches(p, &batches)?;
                }
            }
            self.part_non_empty[p] = false;
            self.mem_part_bytes[p] = 0;
        }
        drop(chunks);
        self.total_mem_bytes = 0;
        reservation.resize(extra_bytes);
        Ok(())
    }

    fn clear_mem_batches(&mut self) {
        self.mem_chunks.clear();
        self.part_non_empty.fill(false);
        self.mem_part_bytes.fill(0);
        self.total_mem_bytes = 0;
    }

    /// O(`2^R`) lock-free combine: move chunk vectors without hashing or
    /// key comparisons.
    pub(super) fn combine_from(&mut self, mut other: Self) {
        debug_assert_eq!(self.radix_bits, other.radix_bits);
        debug_assert_eq!(self.shift, other.shift);
        self.spilled |= other.spilled;
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(other.total_mem_bytes);
        self.mem_chunks.append(&mut other.mem_chunks);
        for p in 0..self.num_partitions() {
            self.part_non_empty[p] |= other.part_non_empty[p];
            self.mem_part_bytes[p] =
                self.mem_part_bytes[p].saturating_add(other.mem_part_bytes[p]);
            self.spill_files[p].append(&mut other.spill_files[p]);
        }
    }
}

/// Trait implemented by `Grouped`, `DirectGrouped`, and `GroupCount` to plug
/// into the shared parallel radix-finalize and out-of-core spill replay engine.
pub(super) trait RadixGroupMerger: Send + Sync + 'static {
    type Table: Send;

    fn partial_schema(&self) -> &SchemaRef;
    fn num_group_cols(&self) -> usize;
    fn new_table(&self, reservation: MemoryReservation) -> Result<Self::Table>;
    fn merge_partial_batch(
        &self,
        table: &mut Self::Table,
        batch: &RecordBatch,
    ) -> Result<()>;
    fn merge_partial_slice(
        &self,
        table: &mut Self::Table,
        batch: &RecordBatch,
        offset: usize,
        len: usize,
    ) -> Result<()> {
        self.merge_partial_batch(table, &batch.slice(offset, len))
    }
    fn hash_partial_batch(
        &self,
        batch: &RecordBatch,
        hashes: &mut Vec<u64>,
    ) -> Result<()>;
    fn table_is_empty(&self, table: &Self::Table) -> bool;
    /// Drain `table` into finalized output `RecordBatch`es of at most
    /// `batch_size` rows (applying `result_operators` and streaming to `tx`).
    fn emit_table<'a>(
        &'a self,
        table: Self::Table,
        result_operators: &'a [Operator],
        batch_size: usize,
        tx: &'a mpsc::Sender<RecordBatch>,
        measurement: Option<&'a RunMetrics>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

#[derive(Clone)]
pub(super) enum ResultOpsRef {
    Pipeline(Arc<super::Pipeline>),
    #[cfg(test)]
    Empty,
}

impl ResultOpsRef {
    pub(super) fn ops(&self) -> &[Operator] {
        match self {
            Self::Pipeline(pipeline) => match &pipeline.sink {
                super::Sink::Aggregate {
                    result_operators, ..
                }
                | super::Sink::GroupCount {
                    result_operators, ..
                } => result_operators.as_slice(),
                _ => &[],
            },
            #[cfg(test)]
            Self::Empty => &[],
        }
    }
}

pub(super) struct RadixFinalizeConfig<'a> {
    pub(super) workers: usize,
    pub(super) batch_size: usize,
    pub(super) memory_pool: &'a Arc<dyn MemoryPool>,
    pub(super) spill: Option<SpillContext>,
    pub(super) consumer_name: &'static str,
    pub(super) result_operators: ResultOpsRef,
    pub(super) measurement: Option<Arc<RunMetrics>>,
}

/// Parallelize partition finalization when in-memory, or stream-replay spilled
/// partitions one at a time in bounded memory (with recursive radix splitting
/// and sorted-run skew fallback).
pub(super) async fn finalize_and_emit_radix<M: RadixGroupMerger>(
    merger: Arc<M>,
    mut store: RadixPartitionStore,
    input_reservation: MemoryReservation,
    cfg: RadixFinalizeConfig<'_>,
    tx: mpsc::Sender<RecordBatch>,
) -> Result<()> {
    if tx.is_closed() || store.is_empty() {
        drop(store);
        drop(input_reservation);
        return Ok(());
    }

    let num_parts = store.num_partitions();
    let shift = store.shift;
    let any_spilled = store.spilled;
    let total_mem_bytes = store.total_mem_bytes;

    let mut mem_chunks = std::mem::take(&mut store.mem_chunks);
    let part_non_empty = std::mem::take(&mut store.part_non_empty);
    let mem_part_bytes = std::mem::take(&mut store.mem_part_bytes);
    let mut spill_files = std::mem::take(&mut store.spill_files);
    drop(store);

    let non_empty: usize = (0..num_parts)
        .filter(|&p| part_non_empty[p] || !spill_files[p].is_empty())
        .count();

    if !any_spilled && non_empty == 0 {
        drop(mem_chunks);
        drop(input_reservation);
        return Ok(());
    }

    // Decide whether we can run parallel in-memory finalize or must run
    // sequential spilled replay.
    let must_spill_sequential = if any_spilled {
        true
    } else if cfg.spill.is_some() {
        // Check if we have enough headroom to hold parallel finalize working
        // sets alongside the in-memory input chunks. If not, spill input
        // partitions up front and replay 1-at-a-time in bounded RAM.
        let parallel_workers = cfg.workers.min(non_empty).max(1);
        let probe_bytes = (total_mem_bytes / non_empty.max(1))
            .saturating_mul(parallel_workers)
            .max(64 * 1024);
        let probe = MemoryConsumer::new(cfg.consumer_name)
            .with_can_spill(true)
            .register(cfg.memory_pool);
        let ok = probe.try_grow(probe_bytes).is_ok();
        drop(probe);
        !ok
    } else {
        false
    };

    if must_spill_sequential {
        let Some(spill_ctx) = cfg.spill else {
            return datafusion_common::internal_err!(
                "Spilled GROUP BY partitions require SpillContext"
            );
        };
        // Flush any remaining in-memory chunks into the stage-shared partition
        // writers and finish them so every partition has at most 1 spill file
        // and 100% of sink memory is freed before replaying partition 0.
        if !mem_chunks.is_empty() {
            let writers =
                spill_ctx.get_or_init_partition_writers(num_parts, "MorselGroupBySpill");
            for (p, &has_mem) in part_non_empty.iter().enumerate() {
                if tx.is_closed() {
                    drop(mem_chunks);
                    drop(input_reservation);
                    return Ok(());
                }
                if has_mem {
                    let batches: Vec<RecordBatch> =
                        mem_chunks.iter().filter_map(|c| c.part_slice(p)).collect();
                    if !batches.is_empty() {
                        writers.append_batches(p, &batches)?;
                    }
                }
            }
            mem_chunks.clear();
        }
        drop(mem_chunks);
        if let Some(finished) = spill_ctx.finish_partition_writers()? {
            for (p, sf_opt) in finished.into_iter().enumerate() {
                if let Some(sf) = sf_opt {
                    spill_files[p].push(sf);
                }
            }
        }
        drop(input_reservation);

        for files in spill_files {
            if tx.is_closed() {
                break;
            }
            if files.is_empty() {
                continue;
            }
            finalize_spilled_partition(
                &merger,
                files,
                shift,
                0,
                cfg.memory_pool,
                &spill_ctx,
                cfg.consumer_name,
                cfg.result_operators.ops(),
                cfg.batch_size,
                &tx,
                cfg.measurement.as_deref(),
            )
            .await?;
        }
        return Ok(());
    }

    // Parallel lock-free in-memory partition finalization across `workers` tasks.
    let mem_chunks = Arc::new(mem_chunks);
    let part_non_empty = Arc::new(part_non_empty);
    let mem_part_bytes = Arc::new(mem_part_bytes);
    let input_reservation = Arc::new(Mutex::new(input_reservation));
    let next_part = Arc::new(AtomicUsize::new(0));
    let worker_count = cfg.workers.min(non_empty).max(1);
    let mut tasks: JoinSet<Result<()>> = JoinSet::new();

    for _ in 0..worker_count {
        let merger = Arc::clone(&merger);
        let mem_chunks = Arc::clone(&mem_chunks);
        let part_non_empty = Arc::clone(&part_non_empty);
        let mem_part_bytes = Arc::clone(&mem_part_bytes);
        let input_res = Arc::clone(&input_reservation);
        let next_part = Arc::clone(&next_part);
        let pool = Arc::clone(cfg.memory_pool);
        let spill = cfg.spill.clone();
        let consumer_name = cfg.consumer_name;
        let result_ops = cfg.result_operators.clone();
        let batch_size = cfg.batch_size;
        let tx = tx.clone();
        let measurement = cfg.measurement.clone();

        tasks.spawn(async move {
            loop {
                if tx.is_closed() {
                    return Ok(());
                }
                let p = next_part.fetch_add(1, Ordering::Relaxed);
                if p >= num_parts {
                    return Ok(());
                }
                if !part_non_empty[p] {
                    continue;
                }
                let est = mem_part_bytes[p];
                let part_res = MemoryConsumer::new(consumer_name)
                    .with_can_spill(spill.is_some())
                    .register(&pool);
                let mut table = merger.new_table(part_res)?;
                let mut merge_err = None;
                for chunk in mem_chunks.iter() {
                    let (off, len) = chunk.part_range(p);
                    if len == 0 {
                        continue;
                    }
                    if tx.is_closed() {
                        return Ok(());
                    }
                    if let Err(err) =
                        merger.merge_partial_slice(&mut table, &chunk.batch, off, len)
                    {
                        merge_err = Some(err);
                        break;
                    }
                }
                if let Some(err) = merge_err {
                    drop(table);
                    if matches!(err, DataFusionError::ResourcesExhausted(_))
                        && let Some(spill_ctx) = spill.as_ref()
                    {
                        // Partition overflowed during parallel finalize: spill
                        // this partition's slices and replay via recursive
                        // radix split without duplicating any output rows.
                        let part_batches: Vec<RecordBatch> =
                            mem_chunks.iter().filter_map(|c| c.part_slice(p)).collect();
                        let spilled = if let Some((sf, _)) = spill_batches_async(
                            &spill_ctx.manager,
                            &part_batches,
                            "MorselGroupBySpill",
                            &tx,
                        )
                        .await?
                        {
                            vec![sf]
                        } else {
                            Vec::new()
                        };
                        drop(part_batches);
                        {
                            let guard = input_res.lock();
                            let cur = guard.size();
                            guard.resize(cur.saturating_sub(est));
                        }
                        finalize_spilled_partition(
                            &merger,
                            spilled,
                            shift,
                            0,
                            &pool,
                            spill_ctx,
                            consumer_name,
                            result_ops.ops(),
                            batch_size,
                            &tx,
                            measurement.as_deref(),
                        )
                        .await?;
                        continue;
                    }
                    return Err(err);
                }

                // Shrink static lease for this partition before emit.
                {
                    let guard = input_res.lock();
                    let cur = guard.size();
                    guard.resize(cur.saturating_sub(est));
                }

                if !merger.table_is_empty(&table) {
                    merger
                        .emit_table(
                            table,
                            result_ops.ops(),
                            batch_size,
                            &tx,
                            measurement.as_deref(),
                        )
                        .await?;
                }
            }
        });
    }

    while let Some(res) = tasks.join_next().await {
        match res {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                abort_and_drain(&mut tasks).await;
                return Err(err);
            }
            Err(err) if err.is_panic() => {
                abort_and_drain(&mut tasks).await;
                std::panic::resume_unwind(err.into_panic())
            }
            Err(err) => {
                abort_and_drain(&mut tasks).await;
                return Err(exec_datafusion_err!(
                    "Morsel radix GROUP BY finalize task failed: {err}"
                ));
            }
        }
    }
    Ok(())
}

/// Replay a single spilled partition in bounded memory, recursively splitting
/// on the next disjoint radix window if it overflows, and falling back to
/// sorted-run external merge at `MAX_RADIX_DEPTH` or on hash-collision skew.
#[expect(clippy::too_many_arguments)]
fn finalize_spilled_partition<'a, M: RadixGroupMerger>(
    merger: &'a Arc<M>,
    spill_files: Vec<Arc<dyn SpillFile>>,
    shift: u32,
    depth: usize,
    memory_pool: &'a Arc<dyn MemoryPool>,
    spill: &'a SpillContext,
    consumer_name: &'static str,
    result_operators: &'a [Operator],
    batch_size: usize,
    tx: &'a mpsc::Sender<RecordBatch>,
    measurement: Option<&'a RunMetrics>,
) -> std::pin::Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(async move {
        if tx.is_closed() || spill_files.is_empty() {
            return Ok(());
        }

        // Fast attempt: merge all spill files of this partition into one table.
        let part_res = MemoryConsumer::new(consumer_name)
            .with_can_spill(true)
            .register(memory_pool);
        let mut table = merger.new_table(part_res)?;
        let mut overflowed = false;

        'files: for sf in &spill_files {
            if tx.is_closed() {
                return Ok(());
            }
            let mut stream = spill
                .manager
                .read_spill_as_stream_unbuffered(Arc::clone(sf), None)?;
            while let Some(batch_res) = stream.next().await {
                if tx.is_closed() {
                    return Ok(());
                }
                let batch = batch_res?;
                if batch.num_rows() == 0 {
                    continue;
                }
                match merger.merge_partial_batch(&mut table, &batch) {
                    Ok(()) => {}
                    Err(DataFusionError::ResourcesExhausted(_)) => {
                        overflowed = true;
                        break 'files;
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        if !overflowed {
            // Unlink partition spill files immediately before emitting.
            drop(spill_files);
            if !merger.table_is_empty(&table) {
                merger
                    .emit_table(table, result_operators, batch_size, tx, measurement)
                    .await?;
            }
            return Ok(());
        }

        // Drop partial table so 100% of MemoryPool is free for re-partitioning.
        drop(table);

        // Try recursive radix re-partitioning on the next lower disjoint window,
        // reusing one appendable writer per sub-partition (`<= 16` files total).
        if depth < MAX_RADIX_DEPTH
            && let Some(next_shift) = next_level_shift(shift, RECURSIVE_RADIX_BITS)
        {
            let sub_parts = 1usize << RECURSIVE_RADIX_BITS;
            let mut sub_writers: Vec<Option<SyncSpillWriter>> =
                (0..sub_parts).map(|_| None).collect();
            let mut sub_rows = vec![0usize; sub_parts];
            let mut total_rows = 0usize;
            let mut hashes = Vec::new();

            for sf in &spill_files {
                if tx.is_closed() {
                    return Ok(());
                }
                let mut stream = spill
                    .manager
                    .read_spill_as_stream_unbuffered(Arc::clone(sf), None)?;
                while let Some(batch_res) = stream.next().await {
                    if tx.is_closed() {
                        return Ok(());
                    }
                    let batch = batch_res?;
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    merger.hash_partial_batch(&batch, &mut hashes)?;
                    let (perm, ranges) =
                        radix_permute_indices(&hashes, next_shift, RECURSIVE_RADIX_BITS);
                    let permuted = take_record_batch(&batch, &perm)?;
                    drop(batch);
                    for (sp, &(offset, len)) in ranges.iter().enumerate() {
                        if len == 0 {
                            continue;
                        }
                        let slice = permuted.slice(offset, len);
                        if sub_writers[sp].is_none() {
                            sub_writers[sp] = Some(SyncSpillWriter::try_new(
                                &spill.manager,
                                "MorselGroupByRecursiveSpill",
                            )?);
                        }
                        sub_writers[sp].as_mut().unwrap().append_batch(&slice)?;
                        sub_rows[sp] += len;
                        total_rows += len;
                    }
                }
            }

            let mut sub_files: Vec<Vec<Arc<dyn SpillFile>>> = vec![Vec::new(); sub_parts];
            for (sp, writer_opt) in sub_writers.iter_mut().enumerate() {
                if let Some(writer) = writer_opt.as_mut()
                    && let Some((out_sf, _)) = writer.finish()?
                {
                    sub_files[sp].push(out_sf);
                }
            }
            drop(sub_writers);

            // Unlink parent partition files immediately!
            drop(spill_files);

            // Check if radix split actually made progress (not 95%+ in one bucket).
            let max_sub = sub_rows.iter().copied().max().unwrap_or(0);
            if total_rows > 0 && max_sub * 20 < total_rows * 19 {
                for files in sub_files {
                    if tx.is_closed() {
                        return Ok(());
                    }
                    if !files.is_empty() {
                        finalize_spilled_partition(
                            merger,
                            files,
                            next_shift,
                            depth + 1,
                            memory_pool,
                            spill,
                            consumer_name,
                            result_operators,
                            batch_size,
                            tx,
                            measurement,
                        )
                        .await?;
                    }
                }
                return Ok(());
            }

            // Non-progressing hash collision skew: merge all sub-files via sorted fallback.
            let flat: Vec<Arc<dyn SpillFile>> = sub_files.into_iter().flatten().collect();
            return finalize_sorted_spill_fallback(
                merger,
                flat,
                memory_pool,
                spill,
                consumer_name,
                result_operators,
                batch_size,
                tx,
                measurement,
            )
            .await;
        }

        // At MAX_RADIX_DEPTH: fall back to sorted-run spill-merge on group keys.
        finalize_sorted_spill_fallback(
            merger,
            spill_files,
            memory_pool,
            spill,
            consumer_name,
            result_operators,
            batch_size,
            tx,
            measurement,
        )
        .await
    })
}

/// Sorted-run external-merge fallback when a single partition still exceeds
/// `MemoryPool` after `MAX_RADIX_DEPTH` recursive splits (e.g. extreme hash
/// collisions or huge group counts in a tiny pool).
#[expect(clippy::too_many_arguments)]
async fn finalize_sorted_spill_fallback<M: RadixGroupMerger>(
    merger: &Arc<M>,
    spill_files: Vec<Arc<dyn SpillFile>>,
    memory_pool: &Arc<dyn MemoryPool>,
    spill: &SpillContext,
    consumer_name: &'static str,
    result_operators: &[Operator],
    batch_size: usize,
    tx: &mpsc::Sender<RecordBatch>,
    measurement: Option<&RunMetrics>,
) -> Result<()> {
    if tx.is_closed() || spill_files.is_empty() {
        return Ok(());
    }
    let partial_schema = Arc::clone(merger.partial_schema());
    let num_group_cols = merger.num_group_cols();
    let ordering = LexOrdering::new((0..num_group_cols).map(|i| {
        PhysicalSortExpr::new(
            Arc::new(Column::new(partial_schema.field(i).name(), i)),
            SortOptions::default(),
        )
    }))
    .ok_or_else(|| {
        datafusion_common::internal_datafusion_err!(
            "GROUP BY sorted fallback requires at least one group column"
        )
    })?;

    // Coalesce batches into bounded runs (~128 KiB) before sorting & spilling
    // so sorted fallback keeps open file count small instead of 1 file/batch.
    const SORTED_FALLBACK_RUN_BYTES: usize = 128 * 1024;
    let mut sorted_runs = Vec::new();
    let mut run_batches: Vec<RecordBatch> = Vec::new();
    let mut run_bytes = 0usize;

    let flush_sorted_run = async |batches: Vec<RecordBatch>,
                                  sorted_runs: &mut Vec<SortedSpillFile>|
           -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        let merged_batch = if batches.len() == 1 {
            batches.into_iter().next().unwrap()
        } else {
            arrow::compute::concat_batches(&partial_schema, &batches)?
        };
        if merged_batch.num_rows() == 0 {
            return Ok(());
        }
        let sort_cols = ordering
            .iter()
            .map(|e| e.evaluate_to_sort_column(&merged_batch))
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(&sort_cols, None)?;
        drop(sort_cols);
        let sorted = take_record_batch(&merged_batch, &indices)?;
        drop(merged_batch);
        // Write in `batch_size` slices so `max_record_batch_memory` stays small
        // for `MultiLevelMergeBuilder`'s tight-pool fan-in sizing!
        let chunk_size = batch_size.max(1);
        let mut chunks = Vec::with_capacity(sorted.num_rows().div_ceil(chunk_size));
        let mut off = 0usize;
        while off < sorted.num_rows() {
            let len = (sorted.num_rows() - off).min(chunk_size);
            chunks.push(sorted.slice(off, len));
            off += len;
        }
        if let Some((run_sf, max_mem)) = spill_batches_async(
            &spill.manager,
            &chunks,
            "MorselGroupBySortedFallback",
            tx,
        )
        .await?
        {
            sorted_runs.push(SortedSpillFile {
                file: run_sf,
                max_record_batch_memory: max_mem,
            });
        }
        Ok(())
    };

    for sf in spill_files {
        if tx.is_closed() {
            return Ok(());
        }
        let mut stream = spill.manager.read_spill_as_stream_unbuffered(sf, None)?;
        while let Some(batch_res) = stream.next().await {
            if tx.is_closed() {
                return Ok(());
            }
            let batch = batch_res?;
            if batch.num_rows() == 0 {
                continue;
            }
            let b_bytes = get_record_batch_memory_size(&batch);
            if !run_batches.is_empty()
                && run_bytes.saturating_add(b_bytes) >= SORTED_FALLBACK_RUN_BYTES
            {
                let to_flush = std::mem::take(&mut run_batches);
                run_bytes = 0;
                flush_sorted_run(to_flush, &mut sorted_runs).await?;
            }
            run_bytes = run_bytes.saturating_add(b_bytes);
            run_batches.push(batch);
        }
    }
    if !run_batches.is_empty() {
        flush_sorted_run(run_batches, &mut sorted_runs).await?;
    }
    if tx.is_closed() || sorted_runs.is_empty() {
        return Ok(());
    }

    // Hold dedicated replay headroom across every `merged.next().await` call
    // so `StreamingMergeBuilder` / `MultiLevelMergeBuilder` can never consume
    // the entire pool and starve `cur_table`.
    let headroom_res = MemoryConsumer::new(consumer_name)
        .with_can_spill(false)
        .register(memory_pool);
    let mut headroom_target = 0usize;
    for target in [64 * 1024usize, 32 * 1024] {
        if headroom_res.try_grow(target).is_ok() {
            headroom_target = target;
            break;
        }
    }

    let merge_metrics = ExecutionPlanMetricsSet::new();
    let merge_res = MemoryConsumer::new(consumer_name)
        .with_can_spill(true)
        .register(memory_pool);
    let mut merged = StreamingMergeBuilder::new()
        .with_sorted_spill_files(sorted_runs)
        .with_spill_manager(spill.manager.clone())
        .with_schema(Arc::clone(&partial_schema))
        .with_expressions(&ordering)
        .with_metrics(BaselineMetrics::new(&merge_metrics, 0))
        .with_batch_size(batch_size.max(1))
        .with_reservation(merge_res)
        .build()?;

    // Stream sorted partial rows: because identical group keys are contiguous,
    // we flush and emit whenever a batch boundary transitions to a new key,
    // holding at most one small batch of groups in RAM at a time.
    let mut cur_table = merger.new_table(
        MemoryConsumer::new(consumer_name)
            .with_can_spill(false)
            .register(memory_pool),
    )?;
    let mut last_key_batch: Option<RecordBatch> = None;

    while let Some(batch_res) = merged.next().await {
        // Temporarily release headroom while `cur_table` ingests/emits this
        // batch, then re-acquire before the next `merged.next().await`.
        headroom_res.resize(0);
        if tx.is_closed() {
            return Ok(());
        }
        let batch = batch_res?;
        if batch.num_rows() == 0 {
            let _ = headroom_res.try_resize(headroom_target);
            continue;
        }
        // Find the run boundary at the start of `batch` that matches the last
        // key of the previous batch (if any), so no group is ever split across
        // two emissions.
        let mut split_at = 0usize;
        if let Some(prev_last) = &last_key_batch {
            while split_at < batch.num_rows()
                && rows_equal_on_keys(prev_last, 0, &batch, split_at, num_group_cols)
            {
                split_at += 1;
            }
            if split_at > 0 {
                merger.merge_partial_batch(&mut cur_table, &batch.slice(0, split_at))?;
            }
            if split_at < batch.num_rows() {
                // Previous group(s) are 100% complete — emit and reset!
                let finished = std::mem::replace(
                    &mut cur_table,
                    merger.new_table(
                        MemoryConsumer::new(consumer_name)
                            .with_can_spill(false)
                            .register(memory_pool),
                    )?,
                );
                if !merger.table_is_empty(&finished) {
                    merger
                        .emit_table(
                            finished,
                            result_operators,
                            batch_size,
                            tx,
                            measurement,
                        )
                        .await?;
                }
            }
        }
        if split_at < batch.num_rows() {
            let tail = batch.slice(split_at, batch.num_rows() - split_at);
            let last_row = tail.num_rows() - 1;
            last_key_batch = Some(tail.slice(last_row, 1));
            merger.merge_partial_batch(&mut cur_table, &tail)?;
        }
        let _ = headroom_res.try_resize(headroom_target);
    }
    drop(headroom_res);

    if !tx.is_closed() && !merger.table_is_empty(&cur_table) {
        merger
            .emit_table(cur_table, result_operators, batch_size, tx, measurement)
            .await?;
    }
    Ok(())
}

fn rows_equal_on_keys(
    a: &RecordBatch,
    row_a: usize,
    b: &RecordBatch,
    row_b: usize,
    num_group_cols: usize,
) -> bool {
    for col_idx in 0..num_group_cols {
        let ca = a.column(col_idx);
        let cb = b.column(col_idx);
        let na = ca.is_null(row_a);
        let nb = cb.is_null(row_b);
        if na != nb {
            return false;
        }
        if na {
            continue;
        }
        let sa = ca.slice(row_a, 1);
        let sb = cb.slice(row_b, 1);
        if sa.as_ref() != sb.as_ref() {
            return false;
        }
    }
    true
}

/// Helper for streaming finalized batches from a partition table through
/// `result_operators` to `tx`.
pub(super) async fn emit_finalized_batch(
    batch: RecordBatch,
    result_operators: &[Operator],
    batch_size: usize,
    tx: &mpsc::Sender<RecordBatch>,
    measurement: Option<&RunMetrics>,
) -> Result<()> {
    let total_rows = batch.num_rows();
    if total_rows == 0 || tx.is_closed() {
        return Ok(());
    }
    let chunk_size = batch_size.max(1);
    let mut offset = 0usize;
    while offset < total_rows && !tx.is_closed() {
        let len = (total_rows - offset).min(chunk_size);
        let slice = if offset == 0 && len == total_rows {
            batch.clone()
        } else {
            batch.slice(offset, len)
        };
        let out = apply_operators(result_operators, slice)?;
        if out.num_rows() > 0 {
            let sent = if let Some(run) = measurement {
                metrics::send(run, None, tx, out).await
            } else {
                tx.send(out).await.is_ok()
            };
            if !sent {
                break;
            }
        }
        offset += len;
        if offset < total_rows {
            tokio::task::yield_now().await;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
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
    fn push_partial_batch_without_spill_returns_oom_without_panicking_when_retry_would_succeed()
     {
        let pool: Arc<dyn MemoryPool> = Arc::new(FailFirstTryGrowPool::new());
        let reservation = MemoryConsumer::new("TestRadixAgg")
            .with_can_spill(true)
            .register(&pool);
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4]))],
        )
        .unwrap();
        let hashes = vec![10u64, 20, 30, 40];
        let (tx, _rx) = mpsc::channel(1);
        let mut store = RadixPartitionStore::new_default();

        let err = store
            .push_partial_batch(batch, &hashes, 0, &reservation, None, &tx)
            .expect_err("expected ResourcesExhausted without panicking");
        assert!(
            matches!(err, DataFusionError::ResourcesExhausted(_)),
            "unexpected error: {err:?}"
        );
        assert_eq!(reservation.size(), 0);
        assert_eq!(pool.reserved(), 0);
    }
}
