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
    DEFAULT_RADIX_BITS, MAX_RADIX_DEPTH, RECURSIVE_RADIX_BITS, level0_shift,
    next_level_shift, radix_permute_indices,
};
use super::spill::{SpillContext, spill_batches_async};
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
    pub(super) mem_batches: Vec<Vec<RecordBatch>>,
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
            mem_batches: vec![Vec::new(); num_parts],
            mem_part_bytes: vec![0; num_parts],
            total_mem_bytes: 0,
            spill_files: vec![Vec::new(); num_parts],
            spilled: false,
        }
    }

    pub(super) fn num_partitions(&self) -> usize {
        self.mem_batches.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.mem_batches.iter().all(Vec::is_empty)
            && self.spill_files.iter().all(Vec::is_empty)
    }

    /// Permute a partial-state `RecordBatch` by radix partition in one O(N)
    /// pass and either retain zero-copy slices in memory or spill on OOM.
    ///
    /// `extra_bytes` is the caller's other live reservation bytes (e.g. empty
    /// pre-agg table overhead) so `reservation.try_resize` tracks total bytes.
    #[expect(clippy::too_many_arguments)]
    pub(super) async fn push_partial_batch(
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

        let should_spill = self.spilled
            || spill.is_some_and(SpillContext::is_spilled)
            || reservation.try_resize(target_bytes).is_err();
        if should_spill {
            let Some(spill_ctx) = spill else {
                // Propagate the exact DataFusion ResourcesExhausted error.
                reservation.try_resize(target_bytes)?;
                unreachable!()
            };
            self.spilled = true;
            spill_ctx.mark_spilled();
            // Spill previously buffered in-memory slices together with the new
            // batch's per-partition slices so all backing buffers drop at once.
            let num_parts = self.num_partitions();
            for (p, &(offset, len)) in ranges.iter().enumerate() {
                if tx.is_closed() {
                    self.clear_mem_batches();
                    reservation.resize(extra_bytes);
                    return Ok(());
                }
                let mut to_spill = std::mem::take(&mut self.mem_batches[p]);
                if len > 0 {
                    to_spill.push(permuted.slice(offset, len));
                }
                if !to_spill.is_empty() {
                    if let Some((sf, _)) = spill_batches_async(
                        &spill_ctx.manager,
                        &to_spill,
                        "MorselGroupBySpill",
                        tx,
                    )
                    .await?
                    {
                        self.spill_files[p].push(sf);
                    }
                }
                self.mem_part_bytes[p] = 0;
            }
            debug_assert_eq!(num_parts, self.mem_batches.len());
            self.total_mem_bytes = 0;
            drop(permuted);
            reservation.resize(extra_bytes);
            return Ok(());
        }

        let total_rows = permuted.num_rows().max(1);
        for (p, &(offset, len)) in ranges.iter().enumerate() {
            if len > 0 {
                self.mem_batches[p].push(permuted.slice(offset, len));
                // Proportional share so releasing individual partitions shrinks
                // the combined reservation smoothly without double-counting.
                let share = batch_bytes.saturating_mul(len) / total_rows;
                self.mem_part_bytes[p] = self.mem_part_bytes[p].saturating_add(share);
            }
        }
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(batch_bytes);
        Ok(())
    }

    /// Spill all currently buffered in-memory batches in this store and reset
    /// `total_mem_bytes` to 0.
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
                    "MorselGroupBySpill",
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

    /// O(`2^R`) lock-free combine: move partition vectors without hashing or
    /// key comparisons.
    pub(super) fn combine_from(&mut self, mut other: Self) {
        debug_assert_eq!(self.radix_bits, other.radix_bits);
        debug_assert_eq!(self.shift, other.shift);
        self.spilled |= other.spilled;
        self.total_mem_bytes = self.total_mem_bytes.saturating_add(other.total_mem_bytes);
        for p in 0..self.num_partitions() {
            self.mem_batches[p].append(&mut other.mem_batches[p]);
            self.mem_part_bytes[p] =
                self.mem_part_bytes[p].saturating_add(other.mem_part_bytes[p]);
            self.spill_files[p].append(&mut other.spill_files[p]);
        }
    }
}

/// Take-once per-partition input moved out of the shared finalize array as soon
/// as a finalize worker claims partition `p`.
struct PartitionInput {
    mem_batches: Vec<RecordBatch>,
    spill_files: Vec<Arc<dyn SpillFile>>,
    est_bytes: usize,
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
    store: RadixPartitionStore,
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

    let mut slots: Vec<Mutex<Option<PartitionInput>>> = Vec::with_capacity(num_parts);
    let mut non_empty = 0usize;
    for p in 0..num_parts {
        let mem = store.mem_batches[p].clone();
        let files = store.spill_files[p].clone();
        if !mem.is_empty() || !files.is_empty() {
            non_empty += 1;
            slots.push(Mutex::new(Some(PartitionInput {
                mem_batches: mem,
                spill_files: files,
                est_bytes: store.mem_part_bytes[p],
            })));
        } else {
            slots.push(Mutex::new(None));
        }
    }
    drop(store);

    if non_empty == 0 {
        drop(input_reservation);
        return Ok(());
    }

    // Decide whether we can run parallel in-memory finalize or must run
    // sequential spilled replay.
    let must_spill_sequential = if any_spilled {
        true
    } else if let Some(spill_ctx) = &cfg.spill {
        // Check if we have enough headroom to hold parallel finalize working
        // sets alongside the in-memory input slices. If not, spill input
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
        if !ok {
            // Spill all in-memory partitions to disk up front so 100% of
            // MemoryPool is freed for sequential 1-partition-at-a-time replay.
            for slot in &slots {
                if tx.is_closed() {
                    return Ok(());
                }
                let batches = {
                    let mut guard = slot.lock();
                    guard
                        .as_mut()
                        .map(|input| std::mem::take(&mut input.mem_batches))
                        .unwrap_or_default()
                };
                if !batches.is_empty()
                    && let Some((sf, _)) = spill_batches_async(
                        &spill_ctx.manager,
                        &batches,
                        "MorselGroupBySpill",
                        &tx,
                    )
                    .await?
                    && let Some(input) = slot.lock().as_mut()
                {
                    input.spill_files.push(sf);
                    input.est_bytes = 0;
                }
            }
            input_reservation.resize(0);
            true
        } else {
            false
        }
    } else {
        false
    };

    if must_spill_sequential {
        let Some(spill_ctx) = cfg.spill else {
            return datafusion_common::internal_err!(
                "Spilled GROUP BY partitions require SpillContext"
            );
        };
        // Flush any remaining in-memory batches across all partitions first so
        // 100% of sink memory is freed before replaying partition 0.
        for slot in &slots {
            if tx.is_closed() {
                drop(slots);
                drop(input_reservation);
                return Ok(());
            }
            let batches = {
                let mut guard = slot.lock();
                guard
                    .as_mut()
                    .map(|i| std::mem::take(&mut i.mem_batches))
                    .unwrap_or_default()
            };
            if !batches.is_empty()
                && let Some((sf, _)) = spill_batches_async(
                    &spill_ctx.manager,
                    &batches,
                    "MorselGroupBySpill",
                    &tx,
                )
                .await?
                && let Some(input) = slot.lock().as_mut()
            {
                input.spill_files.push(sf);
                input.est_bytes = 0;
            }
        }
        drop(input_reservation);

        for slot in &slots {
            if tx.is_closed() {
                break;
            }
            let Some(input) = slot.lock().take() else {
                continue;
            };
            finalize_spilled_partition(
                &merger,
                input.spill_files,
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
    let slots = Arc::new(slots);
    let input_reservation = Arc::new(Mutex::new(input_reservation));
    let next_part = Arc::new(AtomicUsize::new(0));
    let worker_count = cfg.workers.min(non_empty).max(1);
    let mut tasks: JoinSet<Result<()>> = JoinSet::new();

    for _ in 0..worker_count {
        let merger = Arc::clone(&merger);
        let slots = Arc::clone(&slots);
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
                if p >= slots.len() {
                    return Ok(());
                }
                let Some(input) = slots[p].lock().take() else {
                    continue;
                };
                let part_res = MemoryConsumer::new(consumer_name)
                    .with_can_spill(spill.is_some())
                    .register(&pool);
                let mut table = merger.new_table(part_res)?;
                let mut merge_err = None;
                for batch in &input.mem_batches {
                    if tx.is_closed() {
                        return Ok(());
                    }
                    if let Err(err) = merger.merge_partial_batch(&mut table, batch) {
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
                        // this partition's batches and replay via recursive
                        // radix split without duplicating any output rows.
                        let spilled = if let Some((sf, _)) = spill_batches_async(
                            &spill_ctx.manager,
                            &input.mem_batches,
                            "MorselGroupBySpill",
                            &tx,
                        )
                        .await?
                        {
                            vec![sf]
                        } else {
                            Vec::new()
                        };
                        let est = input.est_bytes;
                        drop(input);
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

                // Release input partition chunks & shrink static lease before emit.
                let est = input.est_bytes;
                drop(input);
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

        // Try recursive radix re-partitioning on the next lower disjoint window.
        if depth < MAX_RADIX_DEPTH
            && let Some(next_shift) = next_level_shift(shift, RECURSIVE_RADIX_BITS)
        {
            let sub_parts = 1usize << RECURSIVE_RADIX_BITS;
            let mut sub_files: Vec<Vec<Arc<dyn SpillFile>>> = vec![Vec::new(); sub_parts];
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
                        if let Some((out_sf, _)) = spill_batches_async(
                            &spill.manager,
                            std::slice::from_ref(&slice),
                            "MorselGroupByRecursiveSpill",
                            tx,
                        )
                        .await?
                        {
                            sub_files[sp].push(out_sf);
                            sub_rows[sp] += len;
                            total_rows += len;
                        }
                    }
                }
            }

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

    // Sort each batch of each spill file into sorted spill runs.
    let mut sorted_runs = Vec::new();
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
            let sort_cols = ordering
                .iter()
                .map(|e| e.evaluate_to_sort_column(&batch))
                .collect::<Result<Vec<_>>>()?;
            let indices = lexsort_to_indices(&sort_cols, None)?;
            drop(sort_cols);
            let sorted = take_record_batch(&batch, &indices)?;
            drop(batch);
            if let Some((run_sf, max_mem)) = spill_batches_async(
                &spill.manager,
                std::slice::from_ref(&sorted),
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
        }
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
    for target in [256 * 1024usize, 128 * 1024, 64 * 1024] {
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
