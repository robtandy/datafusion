// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native parallel TopN (`ORDER BY ... LIMIT K [OFFSET O]`) and full parallel
//! Sort (`ORDER BY ...`) sinks for the experimental morsel engine.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::task::{Context, Poll};

use arrow::array::{
    Array, ArrayDataBuilder, ArrayRef, ArrowPrimitiveType, BinaryViewArray, ByteView,
    GenericByteArray, GenericByteViewArray, LargeStringArray, NullBufferBuilder,
    PrimitiveArray, RecordBatch, RecordBatchOptions, StringArray, StringViewArray,
    UInt32Array, make_array,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::compute::{
    SortOptions, concat, concat_batches, interleave_record_batch, lexsort_to_indices,
    take_arrays, take_record_batch,
};
use arrow::datatypes::{
    ArrowNativeTypeOp, BinaryViewType, ByteViewType, DataType, Date32Type, Date64Type,
    Float32Type, Float64Type, GenericStringType, Int8Type, Int16Type, Int32Type,
    Int64Type, Schema, SchemaRef, StringViewType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow::row::{RowConverter, SortField};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::utils::memory::{
    RecordBatchMemoryCounter, RecordBatchMemoryTracker,
};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr};
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::sorts::streaming_merge::{
    SortedSpillFile, StreamingMergeBuilder,
};
use datafusion_physical_plan::spill::SpillManager;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::batch::{BatchView, Selection};
use super::metrics::{self, RunMetrics};
use super::spill::{SpillContext, check_spill_headroom};
use super::{Operator, apply_operators};

/// Maximum incremental buffered bytes (after the first batch's shared buffers)
/// per worker-local sort run before flushing to a sorted run. Bounds peak
/// temporary memory on huge single-worker sorts while keeping ~1 run per worker
/// even when batches inherit large shared view buffers (e.g. from a hash join).
const MAX_LOCAL_SORT_RUN_BYTES: usize = 64 * 1024 * 1024;
/// Smaller flush threshold once any sort worker has transitioned into spill mode
/// so workers stream sorted runs to disk instead of competing for tight RAM.
const MAX_SPILLED_LOCAL_SORT_RUN_BYTES: usize = 256 * 1024;
const MAX_INLINE_VIEW_LEN: u32 = 12;

pub(super) fn is_supported_sort_key_type(dt: &DataType) -> bool {
    match dt {
        DataType::Null
        | DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _)
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => true,
        DataType::Dictionary(_, value) => is_supported_sort_key_type(value),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _) => {
            is_supported_sort_key_type(field.data_type())
        }
        DataType::Struct(fields) => fields
            .iter()
            .all(|f| is_supported_sort_key_type(f.data_type())),
        _ => false,
    }
}

pub(super) fn validate_sort_ordering(
    ordering: &LexOrdering,
    schema: &Schema,
) -> Result<std::result::Result<Vec<SortField>, String>> {
    let mut sort_fields = Vec::with_capacity(ordering.len());
    for sort_expr in ordering.iter() {
        let dt = sort_expr.expr.data_type(schema)?;
        if !is_supported_sort_key_type(&dt) {
            return Ok(Err(format!(
                "unsupported sort key type {dt} in native morsel sort"
            )));
        }
        sort_fields.push(SortField::new_with_options(dt, sort_expr.options));
    }
    if let Err(err) = RowConverter::new(sort_fields.clone()) {
        return Ok(Err(format!(
            "unsupported sort key in native morsel sort: {err}"
        )));
    }
    Ok(Ok(sort_fields))
}

// ---------------------------------------------------------------------------
// View array compaction & deduplicating concat (Utf8View / BinaryView)
// ---------------------------------------------------------------------------

fn data_type_has_view(dt: &DataType) -> bool {
    match dt {
        DataType::Utf8View | DataType::BinaryView => true,
        DataType::Dictionary(_, value) => data_type_has_view(value),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => data_type_has_view(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|f| data_type_has_view(f.data_type()))
        }
        _ => false,
    }
}

fn schema_has_view(schema: &Schema) -> bool {
    schema
        .fields()
        .iter()
        .any(|f| data_type_has_view(f.data_type()))
}

fn dedup_buffer_entries(buffers: &[Buffer]) -> (Vec<u32>, Vec<Buffer>) {
    let mut first_index: HashMap<(*const u8, usize), u32> =
        HashMap::with_capacity(buffers.len());
    let mut unique: Vec<Buffer> = Vec::with_capacity(buffers.len());
    let remap = buffers
        .iter()
        .map(|buffer| {
            *first_index
                .entry((buffer.as_ptr(), buffer.len()))
                .or_insert_with(|| {
                    unique.push(buffer.clone());
                    (unique.len() - 1) as u32
                })
        })
        .collect();
    (remap, unique)
}

/// Compact or deduplicate a `GenericByteViewArray` so taking a small subset of
/// rows does not pin multi-megabyte source data buffers, while avoiding copying
/// shared bytes when many rows reference a smaller deduplicated buffer set.
fn compact_view_array<T: ByteViewType>(
    array: &GenericByteViewArray<T>,
) -> Option<GenericByteViewArray<T>> {
    let buffers = array.data_buffers();
    if buffers.is_empty() {
        return None;
    }

    let (remap, unique) = dedup_buffer_entries(buffers);
    let unique_size: usize = unique.iter().map(|b| b.len()).sum();
    let used_bytes = array.total_buffer_bytes_used();
    if used_bytes < unique_size || (unique.len() > 1 && used_bytes <= unique_size) {
        return Some(array.gc());
    }
    if unique.len() == buffers.len() {
        return None;
    }

    let views: Vec<u128> = array
        .views()
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            if array.is_null(i) {
                0
            } else if (raw as u32) <= MAX_INLINE_VIEW_LEN {
                raw
            } else {
                let view = ByteView::from(raw);
                ByteView {
                    buffer_index: remap[view.buffer_index as usize],
                    ..view
                }
                .as_u128()
            }
        })
        .collect();

    // SAFETY: every non-null, non-inline view points at the same bytes as
    // before, in an entry of `unique` that holds the same buffer, and inline
    // views are unchanged.
    Some(unsafe {
        GenericByteViewArray::new_unchecked(
            ScalarBuffer::from(views),
            unique.into(),
            array.nulls().cloned(),
        )
    })
}

fn gc_view_array(array: &ArrayRef) -> Result<(ArrayRef, bool)> {
    match array.data_type() {
        DataType::Utf8View => {
            let sv = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("Utf8View array");
            Ok(match compact_view_array(sv) {
                Some(compacted) => (Arc::new(compacted) as ArrayRef, true),
                None => (Arc::clone(array), false),
            })
        }
        DataType::BinaryView => {
            let bv = array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .expect("BinaryView array");
            Ok(match compact_view_array(bv) {
                Some(compacted) => (Arc::new(compacted) as ArrayRef, true),
                None => (Arc::clone(array), false),
            })
        }
        dt if data_type_has_view(dt) => {
            let data = array.to_data();
            if data.child_data().is_empty() {
                return Ok((Arc::clone(array), false));
            }
            let mut mutated = false;
            let mut child_data = Vec::with_capacity(data.child_data().len());
            for child in data.child_data() {
                let child_array = make_array(child.clone());
                let (gc_child, child_mutated) = gc_view_array(&child_array)?;
                mutated |= child_mutated;
                child_data.push(gc_child.to_data());
            }
            if !mutated {
                return Ok((Arc::clone(array), false));
            }
            let rebuilt = ArrayDataBuilder::new(data.data_type().clone())
                .len(data.len())
                .offset(data.offset())
                .nulls(data.nulls().cloned())
                .buffers(data.buffers().to_vec())
                .child_data(child_data)
                .build()?;
            Ok((make_array(rebuilt), true))
        }
        _ => Ok((Arc::clone(array), false)),
    }
}

pub(super) fn gc_view_batch(batch: &RecordBatch) -> Result<RecordBatch> {
    if !schema_has_view(batch.schema().as_ref()) {
        return Ok(batch.clone());
    }
    let mut mutated = false;
    let mut new_columns = Vec::with_capacity(batch.num_columns());
    for col in batch.columns() {
        let (gc_col, col_mutated) = gc_view_array(col)?;
        mutated |= col_mutated;
        new_columns.push(gc_col);
    }
    if mutated {
        let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
        Ok(RecordBatch::try_new_with_options(
            batch.schema(),
            new_columns,
            &options,
        )?)
    } else {
        Ok(batch.clone())
    }
}

/// Single-pass deduplicating concat for `GenericByteViewArray` columns.
/// Avoids Arrow `concat`'s naive `data_buffers` concatenation (which duplicates
/// shared buffers once per input batch and explodes buffer-reference counts on
/// small-batch join probes).
fn concat_view_arrays<T: ByteViewType>(
    arrays: &[&dyn Array],
    total_rows: usize,
) -> GenericByteViewArray<T> {
    let mut first_index: HashMap<(*const u8, usize), u32> = HashMap::new();
    let mut unique_buffers: Vec<Buffer> = Vec::new();
    let mut views: Vec<u128> = Vec::with_capacity(total_rows);
    let any_nulls = arrays.iter().any(|a| a.null_count() > 0);
    let mut null_builder = any_nulls.then(|| NullBufferBuilder::new(total_rows));

    for &raw_arr in arrays {
        let arr = raw_arr
            .as_any()
            .downcast_ref::<GenericByteViewArray<T>>()
            .expect("view array type mismatch");
        if arr.is_empty() {
            continue;
        }
        let mut is_identity = true;
        let mut remap = Vec::with_capacity(arr.data_buffers().len());
        for (local_idx, buffer) in arr.data_buffers().iter().enumerate() {
            let global_idx = *first_index
                .entry((buffer.as_ptr(), buffer.len()))
                .or_insert_with(|| {
                    unique_buffers.push(buffer.clone());
                    (unique_buffers.len() - 1) as u32
                });
            if global_idx as usize != local_idx {
                is_identity = false;
            }
            remap.push(global_idx);
        }

        if let Some(nb) = null_builder.as_mut() {
            if let Some(nulls) = arr.nulls() {
                nb.append_buffer(nulls);
            } else {
                nb.append_n_non_nulls(arr.len());
            }
        }

        if is_identity {
            views.extend_from_slice(arr.views());
        } else if arr.null_count() == 0 {
            views.extend(arr.views().iter().map(|&raw| {
                if (raw as u32) <= MAX_INLINE_VIEW_LEN {
                    raw
                } else {
                    let view = ByteView::from(raw);
                    ByteView {
                        buffer_index: remap[view.buffer_index as usize],
                        ..view
                    }
                    .as_u128()
                }
            }));
        } else {
            views.extend(arr.views().iter().enumerate().map(|(i, &raw)| {
                if arr.is_null(i) {
                    0
                } else if (raw as u32) <= MAX_INLINE_VIEW_LEN {
                    raw
                } else {
                    let view = ByteView::from(raw);
                    ByteView {
                        buffer_index: remap[view.buffer_index as usize],
                        ..view
                    }
                    .as_u128()
                }
            }));
        }
    }

    let nulls = null_builder.and_then(|mut nb| nb.finish());
    // SAFETY: every non-null, non-inline view points into `unique_buffers` at
    // the exact byte range of its original buffer; inline views are unchanged.
    let result = unsafe {
        GenericByteViewArray::new_unchecked(
            ScalarBuffer::from(views),
            unique_buffers.into(),
            nulls,
        )
    };
    let unique_size: usize = result.data_buffers().iter().map(|b| b.len()).sum();
    let used_bytes = result.total_buffer_bytes_used();
    if used_bytes < unique_size
        || (result.data_buffers().len() > 1 && used_bytes <= unique_size)
    {
        result.gc()
    } else {
        result
    }
}

pub(super) fn concat_batches_compact_views(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<RecordBatch> {
    if !schema_has_view(schema.as_ref()) {
        return Ok(concat_batches(schema, batches)?);
    }
    let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if total_rows == 0 {
        return Ok(RecordBatch::new_empty(Arc::clone(schema)));
    }
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for (col_idx, field) in schema.fields().iter().enumerate() {
        match field.data_type() {
            DataType::Utf8View => {
                let col_refs: Vec<&dyn Array> =
                    batches.iter().map(|b| b.column(col_idx).as_ref()).collect();
                columns.push(Arc::new(concat_view_arrays::<StringViewType>(
                    &col_refs, total_rows,
                )));
            }
            DataType::BinaryView => {
                let col_refs: Vec<&dyn Array> =
                    batches.iter().map(|b| b.column(col_idx).as_ref()).collect();
                columns.push(Arc::new(concat_view_arrays::<BinaryViewType>(
                    &col_refs, total_rows,
                )));
            }
            dt if data_type_has_view(dt) => {
                let compacted: Vec<ArrayRef> = batches
                    .iter()
                    .map(|b| gc_view_array(b.column(col_idx)).map(|(a, _)| a))
                    .collect::<Result<_>>()?;
                let col_refs: Vec<&dyn Array> =
                    compacted.iter().map(Arc::as_ref).collect();
                columns.push(concat(&col_refs)?);
            }
            _ => {
                let col_refs: Vec<&dyn Array> =
                    batches.iter().map(|b| b.column(col_idx).as_ref()).collect();
                columns.push(concat(&col_refs)?);
            }
        }
    }
    let options = RecordBatchOptions::new().with_row_count(Some(total_rows));
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &options,
    )?)
}

// ---------------------------------------------------------------------------
// Shared deduplicating memory tracker for parallel Sort workers & runs
// ---------------------------------------------------------------------------

struct SharedSortMemoryInner {
    tracker: RecordBatchMemoryTracker,
    extra_bytes: usize,
    reservation: MemoryReservation,
}

/// Deduplicates shared Arrow buffers across all workers and sorted runs of one
/// `SortSink` execution so inherited buffers (e.g. hash-join build `Utf8View`
/// buffers or zero-copy `MemTable`/aggregate slices) are reserved once instead
/// of once per batch or per worker.
pub(super) struct SharedSortMemory {
    inner: Mutex<SharedSortMemoryInner>,
    spilled: AtomicBool,
}

impl SharedSortMemory {
    pub(super) fn new(reservation: MemoryReservation) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(SharedSortMemoryInner {
                tracker: RecordBatchMemoryTracker::new(),
                extra_bytes: 0,
                reservation,
            }),
            spilled: AtomicBool::new(false),
        })
    }

    pub(super) fn mark_spilled(&self) {
        self.spilled.store(true, AtomicOrdering::Release);
    }

    fn is_spilled(&self) -> bool {
        self.spilled.load(AtomicOrdering::Acquire)
    }

    fn total_reserved(&self) -> usize {
        self.inner.lock().reservation.size()
    }

    fn check_spill_headroom(&self, min_headroom_bytes: usize) -> Result<()> {
        let probe = self.inner.lock().reservation.new_empty();
        check_spill_headroom(&probe, min_headroom_bytes)
    }

    fn count_batch(&self, batch: &RecordBatch) -> Result<()> {
        let mut inner = self.inner.lock();
        inner.tracker.count_batch(batch);
        let needed = inner
            .tracker
            .memory_usage()
            .saturating_add(inner.extra_bytes);
        if let Err(err) = inner.reservation.try_resize(needed) {
            inner.tracker.uncount_batch(batch);
            return Err(err);
        }
        Ok(())
    }

    fn uncount_batch(&self, batch: &RecordBatch) {
        let mut inner = self.inner.lock();
        inner.tracker.uncount_batch(batch);
        let needed = inner
            .tracker
            .memory_usage()
            .saturating_add(inner.extra_bytes);
        inner.reservation.resize(needed);
    }

    fn uncount_batches(&self, batches: &[RecordBatch]) {
        if batches.is_empty() {
            return;
        }
        let mut inner = self.inner.lock();
        for batch in batches {
            inner.tracker.uncount_batch(batch);
        }
        let needed = inner
            .tracker
            .memory_usage()
            .saturating_add(inner.extra_bytes);
        inner.reservation.resize(needed);
    }

    fn reserve_extra(&self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut inner = self.inner.lock();
        let new_extra = inner.extra_bytes.saturating_add(bytes);
        let needed = inner.tracker.memory_usage().saturating_add(new_extra);
        inner.reservation.try_resize(needed)?;
        inner.extra_bytes = new_extra;
        Ok(())
    }

    fn adjust_extra(&self, old_bytes: usize, new_bytes: usize) -> Result<()> {
        if old_bytes == new_bytes {
            return Ok(());
        }
        let mut inner = self.inner.lock();
        let new_extra = inner
            .extra_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        let needed = inner.tracker.memory_usage().saturating_add(new_extra);
        inner.reservation.try_resize(needed)?;
        inner.extra_bytes = new_extra;
        Ok(())
    }

    fn release_extra(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let mut inner = self.inner.lock();
        inner.extra_bytes = inner.extra_bytes.saturating_sub(bytes);
        let needed = inner
            .tracker
            .memory_usage()
            .saturating_add(inner.extra_bytes);
        inner.reservation.resize(needed);
    }
}

/// RAII guard ensuring `batches` are uncounted from `shared` *before* they are
/// dropped (satisfying `RecordBatchMemoryTracker`'s contract on every error/unwind).
struct TrackedBatches {
    batches: Vec<RecordBatch>,
    shared: Arc<SharedSortMemory>,
}

impl TrackedBatches {
    #[cfg(test)]
    fn take_single(mut self) -> RecordBatch {
        debug_assert_eq!(self.batches.len(), 1);
        self.batches.pop().unwrap()
    }
}

impl Drop for TrackedBatches {
    fn drop(&mut self) {
        self.shared.uncount_batches(&self.batches);
    }
}

struct TrackedBatch {
    batch: Option<RecordBatch>,
    shared: Arc<SharedSortMemory>,
}

impl TrackedBatch {
    fn into_inner(mut self) -> RecordBatch {
        self.batch.take().expect("TrackedBatch already taken")
    }
}

impl Drop for TrackedBatch {
    fn drop(&mut self) {
        if let Some(batch) = self.batch.as_ref() {
            self.shared.uncount_batch(batch);
        }
    }
}

struct TrackedExtra {
    bytes: usize,
    shared: Arc<SharedSortMemory>,
}

impl TrackedExtra {
    fn try_new(shared: Arc<SharedSortMemory>, bytes: usize) -> Result<Self> {
        shared.reserve_extra(bytes)?;
        Ok(Self { bytes, shared })
    }

    fn try_resize(&mut self, new_bytes: usize) -> Result<()> {
        self.shared.adjust_extra(self.bytes, new_bytes)?;
        self.bytes = new_bytes;
        Ok(())
    }

    fn into_bytes(mut self) -> usize {
        std::mem::take(&mut self.bytes)
    }
}

impl Drop for TrackedExtra {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.shared.release_extra(self.bytes);
            self.bytes = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Sorted runs & k-way merge emission (shared by TopN and Sort)
// ---------------------------------------------------------------------------

/// A worker-local sorted run ready for k-way merge or direct emission.
pub(super) enum SortedRun {
    /// Lazily slices `indices` and gathers `batch_size` chunks on demand.
    Indexed(IndexedRun),
    /// Already-materialized sorted chunks (used by TopN).
    Batches(Box<BatchedRun>),
    /// Sorted run spilled to disk via `SpillManager`.
    Spilled(SortedSpillFile),
}

async fn spill_sorted_indexed_run(
    batch: &RecordBatch,
    indices: &UInt32Array,
    chunk_size: usize,
    spill_manager: &SpillManager,
    tx: &mpsc::Sender<RecordBatch>,
) -> Result<Option<SortedSpillFile>> {
    if indices.is_empty() {
        return Ok(None);
    }
    let chunk_size = chunk_size.max(1);
    let mut file = spill_manager.create_in_progress_file("MorselSortSpill")?;
    let mut max_record_batch_memory = 0usize;
    let res: Result<Option<Arc<dyn datafusion_execution::spill_file::SpillFile>>> =
        async {
            let mut offset = 0usize;
            while offset < indices.len() {
                if tx.is_closed() {
                    return Ok(None);
                }
                let take_len = (indices.len() - offset).min(chunk_size);
                let slice = indices.slice(offset, take_len);
                offset += take_len;
                let columns = take_arrays(batch.columns(), &slice, None)?;
                let options = RecordBatchOptions::new().with_row_count(Some(take_len));
                let chunk =
                    RecordBatch::try_new_with_options(batch.schema(), columns, &options)?;
                let mem = file.append_batch_async(&chunk).await?;
                max_record_batch_memory = max_record_batch_memory.max(mem);
            }
            if tx.is_closed() {
                return Ok(None);
            }
            file.finish_async().await
        }
        .await;

    match res {
        Ok(Some(finished)) => Ok(Some(SortedSpillFile {
            file: finished,
            max_record_batch_memory,
        })),
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

async fn spill_batched_run(
    mut run: Box<BatchedRun>,
    spill_manager: &SpillManager,
    tx: &mpsc::Sender<RecordBatch>,
) -> Result<Option<SortedSpillFile>> {
    let mut file = spill_manager.create_in_progress_file("MorselSortSpill")?;
    let mut max_record_batch_memory = 0usize;
    let res: Result<Option<Arc<dyn datafusion_execution::spill_file::SpillFile>>> =
        async {
            while let Some(batch) = run.pop_front() {
                if batch.num_rows() == 0 {
                    continue;
                }
                if tx.is_closed() {
                    return Ok(None);
                }
                let mem = file.append_batch_async(&batch).await?;
                max_record_batch_memory = max_record_batch_memory.max(mem);
            }
            if tx.is_closed() {
                return Ok(None);
            }
            file.finish_async().await
        }
        .await;
    run.release();

    match res {
        Ok(Some(finished)) => Ok(Some(SortedSpillFile {
            file: finished,
            max_record_batch_memory,
        })),
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

async fn spill_batches_sorted(
    schema: &SchemaRef,
    ordering: &LexOrdering,
    batches: &[RecordBatch],
    chunk_size: usize,
    spill_manager: &SpillManager,
    tx: &mpsc::Sender<RecordBatch>,
) -> Result<Option<SortedSpillFile>> {
    if batches.is_empty() || batches.iter().all(|b| b.num_rows() == 0) {
        return Ok(None);
    }
    let merged = if batches.len() == 1 {
        batches[0].clone()
    } else {
        concat_batches_compact_views(schema, batches)?
    };
    let sort_columns = ordering
        .iter()
        .map(|e| e.evaluate_to_sort_column(&merged))
        .collect::<Result<Vec<_>>>()?;
    let indices = lexsort_to_indices(&sort_columns, None)?;
    drop(sort_columns);
    spill_sorted_indexed_run(&merged, &indices, chunk_size, spill_manager, tx).await
}

pub(super) struct IndexedRun {
    batch: Option<RecordBatch>,
    indices: UInt32Array,
    indices_bytes: usize,
    shared: Arc<SharedSortMemory>,
}

impl IndexedRun {
    fn release(&mut self) {
        if let Some(batch) = self.batch.as_ref() {
            self.shared.uncount_batch(batch);
        }
        self.batch = None;
        if self.indices_bytes > 0 {
            self.shared.release_extra(self.indices_bytes);
            self.indices_bytes = 0;
        }
    }
}

impl Drop for IndexedRun {
    fn drop(&mut self) {
        self.release();
    }
}

pub(super) struct BatchedRun {
    batches: VecDeque<RecordBatch>,
    tracker: RecordBatchMemoryTracker,
    reservation: MemoryReservation,
}

impl BatchedRun {
    fn pop_front(&mut self) -> Option<RecordBatch> {
        let next = self.batches.pop_front()?;
        self.tracker.uncount_batch(&next);
        self.reservation.resize(self.tracker.memory_usage());
        Some(next)
    }

    fn release(&mut self) {
        for batch in &self.batches {
            self.tracker.uncount_batch(batch);
        }
        self.batches.clear();
        self.reservation.free();
    }
}

impl Drop for BatchedRun {
    fn drop(&mut self) {
        self.release();
    }
}

struct SortedRunStream {
    schema: SchemaRef,
    state: SortedRunStreamState,
}

enum SortedRunStreamState {
    Indexed {
        run: IndexedRun,
        offset: usize,
        batch_size: usize,
    },
    Batches {
        run: Box<BatchedRun>,
    },
}

impl SortedRunStream {
    fn from_run(run: SortedRun, schema: SchemaRef, batch_size: usize) -> Self {
        let state = match run {
            SortedRun::Indexed(run) => SortedRunStreamState::Indexed {
                run,
                offset: 0,
                batch_size: batch_size.max(1),
            },
            SortedRun::Batches(run) => SortedRunStreamState::Batches { run },
            SortedRun::Spilled(_) => {
                unreachable!("spilled runs must merge through StreamingMergeBuilder")
            }
        };
        Self { schema, state }
    }
}

impl Stream for SortedRunStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match &mut self.state {
            SortedRunStreamState::Indexed {
                run,
                offset,
                batch_size,
            } => {
                if *offset >= run.indices.len() {
                    run.release();
                    return Poll::Ready(None);
                }
                let Some(backing) = run.batch.as_ref() else {
                    run.release();
                    return Poll::Ready(None);
                };
                let len = (run.indices.len() - *offset).min(*batch_size);
                let slice = run.indices.slice(*offset, len);
                *offset += len;
                let result = take_arrays(backing.columns(), &slice, None)
                    .and_then(|columns| {
                        let options = RecordBatchOptions::new().with_row_count(Some(len));
                        RecordBatch::try_new_with_options(
                            backing.schema(),
                            columns,
                            &options,
                        )
                    })
                    .map_err(Into::into)
                    .and_then(|chunk| {
                        // Compact view columns when taking a chunk from a larger run
                        // so StreamingMergeBuilder's BatchBuilder and FieldCursorStream
                        // do not charge every chunk for the entire run's data buffers.
                        if backing.num_rows() > len {
                            gc_view_batch(&chunk)
                        } else {
                            Ok(chunk)
                        }
                    });
                if *offset >= run.indices.len() {
                    run.release();
                }
                Poll::Ready(Some(result))
            }
            SortedRunStreamState::Batches { run } => match run.pop_front() {
                Some(next) => Poll::Ready(Some(Ok(next))),
                None => {
                    run.release();
                    Poll::Ready(None)
                }
            },
        }
    }
}

impl RecordBatchStream for SortedRunStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

struct EmitSortedRunsConfig<'a> {
    schema: SchemaRef,
    ordering: &'a LexOrdering,
    skip: usize,
    fetch: Option<usize>,
    result_operators: &'a [Operator],
    batch_size: usize,
    memory_pool: &'a Arc<dyn MemoryPool>,
    consumer_name: &'a str,
    spill: Option<SpillContext>,
}

async fn emit_sorted_runs(
    runs: Vec<SortedRun>,
    cfg: EmitSortedRunsConfig<'_>,
    tx: mpsc::Sender<RecordBatch>,
    measurement: Option<&RunMetrics>,
) -> Result<()> {
    let EmitSortedRunsConfig {
        schema,
        ordering,
        skip,
        fetch,
        result_operators,
        batch_size,
        memory_pool,
        consumer_name,
        spill,
    } = cfg;
    if tx.is_closed() || runs.is_empty() || fetch == Some(0) {
        return Ok(());
    }

    // Fast path: a single sorted run does not need k-way merge.
    if runs.len() == 1 {
        let run = runs.into_iter().next().unwrap();
        match run {
            SortedRun::Indexed(mut run) => {
                let total = run.indices.len();
                let start = skip.min(total);
                let remaining = total - start;
                let len = fetch.map_or(remaining, |f| f.min(remaining));
                if len == 0 {
                    run.release();
                    return Ok(());
                }
                let sliced = run.indices.slice(start, len);
                let chunk_size = batch_size.max(1);
                let mut offset = 0;
                while offset < sliced.len() {
                    if tx.is_closed() {
                        run.release();
                        return Ok(());
                    }
                    let take_len = (sliced.len() - offset).min(chunk_size);
                    let chunk_idx = sliced.slice(offset, take_len);
                    offset += take_len;
                    let backing = run.batch.as_ref().expect("indexed run batch");
                    let columns = take_arrays(backing.columns(), &chunk_idx, None)?;
                    let options =
                        RecordBatchOptions::new().with_row_count(Some(take_len));
                    let out = RecordBatch::try_new_with_options(
                        backing.schema(),
                        columns,
                        &options,
                    )?;
                    let out = if backing.num_rows() > take_len {
                        gc_view_batch(&out)?
                    } else {
                        out
                    };
                    let projected = apply_operators(result_operators, out)?;
                    if projected.num_rows() > 0
                        && !send_batch(&tx, projected, measurement).await
                    {
                        run.release();
                        return Ok(());
                    }
                }
                run.release();
                return Ok(());
            }
            SortedRun::Batches(mut run) => {
                let mut remaining_skip = skip;
                let mut remaining_fetch = fetch;
                while let Some(batch) = run.pop_front() {
                    if tx.is_closed() || remaining_fetch == Some(0) {
                        break;
                    }
                    let Some(sliced) = slice_skip_fetch(
                        batch,
                        &mut remaining_skip,
                        &mut remaining_fetch,
                    ) else {
                        continue;
                    };
                    let projected = apply_operators(result_operators, sliced)?;
                    if projected.num_rows() > 0
                        && !send_batch(&tx, projected, measurement).await
                    {
                        break;
                    }
                }
                run.release();
                return Ok(());
            }
            SortedRun::Spilled(sf) => {
                let Some(spill_ctx) = spill else {
                    return datafusion_common::internal_err!(
                        "Spilled sorted run requires SpillContext"
                    );
                };
                let mut stream = spill_ctx.manager.read_spill_as_stream_unbuffered(
                    sf.file,
                    Some(sf.max_record_batch_memory),
                )?;
                let mut remaining_skip = skip;
                let mut remaining_fetch = fetch;
                while let Some(batch) = stream.next().await {
                    if tx.is_closed() || remaining_fetch == Some(0) {
                        break;
                    }
                    let batch = batch?;
                    let Some(sliced) = slice_skip_fetch(
                        batch,
                        &mut remaining_skip,
                        &mut remaining_fetch,
                    ) else {
                        continue;
                    };
                    let projected = apply_operators(result_operators, sliced)?;
                    if projected.num_rows() > 0
                        && !send_batch(&tx, projected, measurement).await
                    {
                        break;
                    }
                }
                return Ok(());
            }
        }
    }

    let merge_fetch = fetch.and_then(|f| skip.checked_add(f));
    let merge_metrics = ExecutionPlanMetricsSet::new();
    let any_spilled = runs.iter().any(|r| matches!(r, SortedRun::Spilled(_)));

    let mut merged = if any_spilled {
        let Some(spill_ctx) = spill else {
            return datafusion_common::internal_err!(
                "Spilled sorted runs require SpillContext"
            );
        };
        // Spill any straggler in-memory runs so 100% of MemoryPool is available
        // for MultiLevelMergeBuilder's bounded-fan-in external merge.
        let mut spilled_files = Vec::with_capacity(runs.len());
        for run in runs {
            if tx.is_closed() {
                return Ok(());
            }
            match run {
                SortedRun::Spilled(sf) => spilled_files.push(sf),
                SortedRun::Indexed(mut ir) => {
                    let sf = if let Some(backing) = ir.batch.as_ref() {
                        spill_sorted_indexed_run(
                            backing,
                            &ir.indices,
                            batch_size,
                            &spill_ctx.manager,
                            &tx,
                        )
                        .await?
                    } else {
                        None
                    };
                    ir.release();
                    if let Some(sf) = sf {
                        spilled_files.push(sf);
                    }
                }
                SortedRun::Batches(br) => {
                    if let Some(sf) =
                        spill_batched_run(br, &spill_ctx.manager, &tx).await?
                    {
                        spilled_files.push(sf);
                    }
                }
            }
        }
        if tx.is_closed() || spilled_files.is_empty() {
            return Ok(());
        }
        let merge_reservation = MemoryConsumer::new(consumer_name)
            .with_can_spill(true)
            .register(memory_pool);
        StreamingMergeBuilder::new()
            .with_sorted_spill_files(spilled_files)
            .with_spill_manager(spill_ctx.manager)
            .with_schema(schema)
            .with_expressions(ordering)
            .with_metrics(BaselineMetrics::new(&merge_metrics, 0))
            .with_batch_size(batch_size.max(1))
            .with_fetch(merge_fetch)
            .with_reservation(merge_reservation)
            .build()?
    } else {
        let streams: Vec<SendableRecordBatchStream> = runs
            .into_iter()
            .map(|run| {
                Box::pin(SortedRunStream::from_run(
                    run,
                    Arc::clone(&schema),
                    batch_size,
                )) as SendableRecordBatchStream
            })
            .collect();
        let merge_reservation = MemoryConsumer::new(consumer_name)
            .with_can_spill(spill.is_some())
            .register(memory_pool);
        StreamingMergeBuilder::new()
            .with_streams(streams)
            .with_schema(schema)
            .with_expressions(ordering)
            .with_metrics(BaselineMetrics::new(&merge_metrics, 0))
            .with_batch_size(batch_size.max(1))
            .with_fetch(merge_fetch)
            .with_reservation(merge_reservation)
            .build()?
    };

    let mut remaining_skip = skip;
    let mut remaining_fetch = fetch;
    while let Some(batch) = merged.next().await {
        if tx.is_closed() {
            return Ok(());
        }
        let batch = batch?;
        let Some(sliced) =
            slice_skip_fetch(batch, &mut remaining_skip, &mut remaining_fetch)
        else {
            if remaining_fetch == Some(0) {
                break;
            }
            continue;
        };
        let projected = apply_operators(result_operators, sliced)?;
        if projected.num_rows() > 0 && !send_batch(&tx, projected, measurement).await {
            return Ok(());
        }
        if remaining_fetch == Some(0) {
            break;
        }
    }
    drop(merged);
    if any_spilled {
        tokio::task::yield_now().await;
    }
    Ok(())
}

fn slice_skip_fetch(
    mut batch: RecordBatch,
    remaining_skip: &mut usize,
    remaining_fetch: &mut Option<usize>,
) -> Option<RecordBatch> {
    if *remaining_fetch == Some(0) || batch.num_rows() == 0 {
        return None;
    }
    if *remaining_skip > 0 {
        if batch.num_rows() <= *remaining_skip {
            *remaining_skip -= batch.num_rows();
            return None;
        }
        let keep = batch.num_rows() - *remaining_skip;
        batch = batch.slice(*remaining_skip, keep);
        *remaining_skip = 0;
    }
    if let Some(fetch) = remaining_fetch.as_mut() {
        if *fetch == 0 {
            return None;
        }
        if batch.num_rows() > *fetch {
            batch = batch.slice(0, *fetch);
            *fetch = 0;
        } else {
            *fetch -= batch.num_rows();
        }
    }
    (batch.num_rows() > 0).then_some(batch)
}

async fn send_batch(
    tx: &mpsc::Sender<RecordBatch>,
    batch: RecordBatch,
    measurement: Option<&RunMetrics>,
) -> bool {
    if let Some(run) = measurement {
        metrics::send(run, None, tx, batch).await
    } else {
        tx.send(batch).await.is_ok()
    }
}

// ---------------------------------------------------------------------------
// Native TopN (ORDER BY ... LIMIT K [OFFSET O])
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct TopNSink {
    schema: SchemaRef,
    ordering: LexOrdering,
    sort_fields: Vec<SortField>,
    skip: usize,
    fetch: usize,
    heap_limit: usize,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
    #[cfg(test)]
    pub(super) peak_retained_rows: Arc<AtomicUsize>,
    #[cfg(test)]
    pub(super) compaction_count: Arc<AtomicUsize>,
    #[cfg(test)]
    pub(super) heap_rebuild_count: Arc<AtomicUsize>,
}

impl TopNSink {
    pub(super) fn try_new(
        schema: SchemaRef,
        ordering: LexOrdering,
        skip: usize,
        fetch: usize,
    ) -> Result<std::result::Result<Self, String>> {
        let sort_fields = match validate_sort_ordering(&ordering, schema.as_ref())? {
            Ok(fields) => fields,
            Err(reason) => return Ok(Err(reason)),
        };
        let heap_limit = if fetch == 0 {
            0
        } else {
            skip.saturating_add(fetch)
        };
        Ok(Ok(Self {
            schema,
            ordering,
            sort_fields,
            skip,
            fetch,
            heap_limit,
            #[cfg(test)]
            finalize_gate: None,
            #[cfg(test)]
            peak_retained_rows: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            compaction_count: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            heap_rebuild_count: Arc::new(AtomicUsize::new(0)),
        }))
    }

    pub(super) fn input_schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub(super) fn ordering(&self) -> &LexOrdering {
        &self.ordering
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(self.ordering.iter().map(|s| &s.expr), f)
    }

    pub(super) fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TopNSink[order=[{}], limit={}",
            self.ordering, self.fetch
        )?;
        if self.skip > 0 {
            write!(f, ", offset={}", self.skip)?;
        }
        write!(f, "]")
    }

    #[cfg_attr(not(test), expect(dead_code))]
    pub(super) fn new_state(&self, reservation: MemoryReservation) -> Result<TopNState> {
        self.new_state_with_spill(reservation, None, 8192)
    }

    pub(super) fn new_state_with_spill(
        &self,
        reservation: MemoryReservation,
        spill: Option<SpillContext>,
        batch_size: usize,
    ) -> Result<TopNState> {
        let row_converter = RowConverter::new(self.sort_fields.clone())?;
        let first_col_fast = FirstColFastPath::try_new(&self.ordering, &self.schema);
        Ok(TopNState {
            schema: Arc::clone(&self.schema),
            ordering: self.ordering.clone(),
            row_converter,
            limit: self.heap_limit,
            heap: BinaryHeap::new(),
            batches: Vec::new(),
            retained_rows: 0,
            batch_memory: RecordBatchMemoryTracker::new(),
            row_map_bytes: 0,
            key_bytes: 0,
            first_col_fast,
            spilled_runs: Vec::new(),
            spill,
            batch_size: batch_size.max(1),
            reservation,
            #[cfg(test)]
            peak_retained_rows: Arc::clone(&self.peak_retained_rows),
            #[cfg(test)]
            compaction_count: Arc::clone(&self.compaction_count),
            #[cfg(test)]
            heap_rebuild_count: Arc::clone(&self.heap_rebuild_count),
        })
    }

    #[expect(clippy::too_many_arguments)]
    pub(super) async fn finalize_and_emit(
        &self,
        runs: Vec<SortedRun>,
        result_operators: &[Operator],
        batch_size: usize,
        memory_pool: &Arc<dyn MemoryPool>,
        spill: Option<SpillContext>,
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        emit_sorted_runs(
            runs,
            EmitSortedRunsConfig {
                schema: Arc::clone(&self.schema),
                ordering: &self.ordering,
                skip: self.skip,
                fetch: Some(self.fetch),
                result_operators,
                batch_size,
                memory_pool,
                consumer_name: "MorselTopNMerge",
                spill,
            },
            tx,
            measurement,
        )
        .await
    }
}

#[derive(Eq)]
struct TopNHeapEntry {
    key: Box<[u8]>,
    batch_idx: u32,
    row_idx: u32,
}

impl PartialEq for TopNHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.batch_idx == other.batch_idx
            && self.row_idx == other.row_idx
    }
}

impl Ord for TopNHeapEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.key
            .cmp(&other.key)
            .then_with(|| self.batch_idx.cmp(&other.batch_idx))
            .then_with(|| self.row_idx.cmp(&other.row_idx))
    }
}

impl PartialOrd for TopNHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

enum FirstColThreshold {
    Int8(Option<i8>),
    Int16(Option<i16>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    UInt8(Option<u8>),
    UInt16(Option<u16>),
    UInt32(Option<u32>),
    UInt64(Option<u64>),
    Float32(Option<f32>),
    Float64(Option<f64>),
    Str(Option<Box<str>>),
}

struct FirstColFastPath {
    col_idx: usize,
    options: SortOptions,
    is_single_key: bool,
    threshold: Option<FirstColThreshold>,
}

#[inline]
fn compare_opt_primitive<T: ArrowNativeTypeOp>(
    candidate: Option<T>,
    threshold: Option<T>,
    options: SortOptions,
) -> CmpOrdering {
    match (candidate, threshold) {
        (None, None) => CmpOrdering::Equal,
        (None, Some(_)) => {
            if options.nulls_first {
                CmpOrdering::Less
            } else {
                CmpOrdering::Greater
            }
        }
        (Some(_), None) => {
            if options.nulls_first {
                CmpOrdering::Greater
            } else {
                CmpOrdering::Less
            }
        }
        (Some(a), Some(b)) => {
            let ord = a.compare(b);
            if options.descending {
                ord.reverse()
            } else {
                ord
            }
        }
    }
}

#[inline]
fn compare_opt_str(
    candidate: Option<&str>,
    threshold: Option<&str>,
    options: SortOptions,
) -> CmpOrdering {
    match (candidate, threshold) {
        (None, None) => CmpOrdering::Equal,
        (None, Some(_)) => {
            if options.nulls_first {
                CmpOrdering::Less
            } else {
                CmpOrdering::Greater
            }
        }
        (Some(_), None) => {
            if options.nulls_first {
                CmpOrdering::Greater
            } else {
                CmpOrdering::Less
            }
        }
        (Some(a), Some(b)) => {
            let ord = a.cmp(b);
            if options.descending {
                ord.reverse()
            } else {
                ord
            }
        }
    }
}

#[inline]
fn passes_ord(ord: CmpOrdering, is_single_key: bool) -> bool {
    if is_single_key {
        ord == CmpOrdering::Less
    } else {
        ord != CmpOrdering::Greater
    }
}

fn scan_primitive_array<P>(
    array: &PrimitiveArray<P>,
    selection: &Selection,
    threshold: Option<P::Native>,
    options: SortOptions,
    is_single_key: bool,
    out: &mut Vec<u32>,
) where
    P: ArrowPrimitiveType,
    P::Native: ArrowNativeTypeOp,
{
    let values = array.values();
    let has_nulls = array.null_count() > 0;
    if !has_nulls {
        let Some(t) = threshold else {
            if !options.nulls_first {
                match selection {
                    Selection::All => out.extend(0..(values.len() as u32)),
                    Selection::Mask { bits, .. } => {
                        out.extend(bits.set_indices().map(|i| i as u32));
                    }
                    Selection::Indices(indices) => {
                        out.extend_from_slice(indices.values());
                    }
                }
            }
            return;
        };
        let keep = |v: P::Native| -> bool {
            let ord = if options.descending {
                t.compare(v)
            } else {
                v.compare(t)
            };
            passes_ord(ord, is_single_key)
        };
        match selection {
            Selection::All => {
                for (i, &v) in values.iter().enumerate() {
                    if keep(v) {
                        out.push(i as u32);
                    }
                }
            }
            Selection::Mask { bits, .. } => {
                for i in bits.set_indices() {
                    if keep(values[i]) {
                        out.push(i as u32);
                    }
                }
            }
            Selection::Indices(indices) => {
                for &idx in indices.values() {
                    if keep(values[idx as usize]) {
                        out.push(idx);
                    }
                }
            }
        }
    } else {
        let check = |i: usize| -> bool {
            let v = if array.is_null(i) {
                None
            } else {
                Some(values[i])
            };
            passes_ord(compare_opt_primitive(v, threshold, options), is_single_key)
        };
        match selection {
            Selection::All => {
                for i in 0..array.len() {
                    if check(i) {
                        out.push(i as u32);
                    }
                }
            }
            Selection::Mask { bits, .. } => {
                for i in bits.set_indices() {
                    if check(i) {
                        out.push(i as u32);
                    }
                }
            }
            Selection::Indices(indices) => {
                for &idx in indices.values() {
                    if check(idx as usize) {
                        out.push(idx);
                    }
                }
            }
        }
    }
}

impl FirstColFastPath {
    fn scan_str_accessor<'a, F>(
        &self,
        len: usize,
        has_nulls: bool,
        get_opt: F,
        selection: &Selection,
        threshold: Option<&str>,
        out: &mut Vec<u32>,
    ) where
        F: Fn(usize) -> Option<&'a str>,
    {
        if !has_nulls && threshold.is_none() {
            if !self.options.nulls_first {
                match selection {
                    Selection::All => out.extend(0..(len as u32)),
                    Selection::Mask { bits, .. } => {
                        out.extend(bits.set_indices().map(|i| i as u32));
                    }
                    Selection::Indices(indices) => {
                        out.extend_from_slice(indices.values());
                    }
                }
            }
            return;
        }
        let check = |i: usize| -> bool {
            passes_ord(
                compare_opt_str(get_opt(i), threshold, self.options),
                self.is_single_key,
            )
        };
        match selection {
            Selection::All => {
                for i in 0..len {
                    if check(i) {
                        out.push(i as u32);
                    }
                }
            }
            Selection::Mask { bits, .. } => {
                for i in bits.set_indices() {
                    if check(i) {
                        out.push(i as u32);
                    }
                }
            }
            Selection::Indices(indices) => {
                for &idx in indices.values() {
                    if check(idx as usize) {
                        out.push(idx);
                    }
                }
            }
        }
    }

    fn try_new(ordering: &LexOrdering, schema: &Schema) -> Option<Self> {
        let first = ordering.first();
        let col = first.expr.downcast_ref::<Column>()?;
        let dt = schema.field(col.index()).data_type();
        let supported = matches!(
            dt,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
                | DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(_, _)
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
        );
        supported.then_some(Self {
            col_idx: col.index(),
            options: first.options,
            is_single_key: ordering.len() == 1,
            threshold: None,
        })
    }

    fn update_from_batch(&mut self, batch: &RecordBatch, row_idx: usize) {
        let col = batch.column(self.col_idx);
        let threshold = match col.data_type() {
            DataType::Int8 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int8Type>>()
                    .unwrap();
                FirstColThreshold::Int8((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Int16 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int16Type>>()
                    .unwrap();
                FirstColThreshold::Int16((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Int32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int32Type>>()
                    .unwrap();
                FirstColThreshold::Int32((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Date32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Date32Type>>()
                    .unwrap();
                FirstColThreshold::Int32((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Int64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int64Type>>()
                    .unwrap();
                FirstColThreshold::Int64((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Date64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Date64Type>>()
                    .unwrap();
                FirstColThreshold::Int64((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Timestamp(unit, _) => {
                let val = match unit {
                    arrow::datatypes::TimeUnit::Second => {
                        let a = col
                            .as_any()
                            .downcast_ref::<PrimitiveArray<TimestampSecondType>>()
                            .unwrap();
                        (!a.is_null(row_idx)).then(|| a.value(row_idx))
                    }
                    arrow::datatypes::TimeUnit::Millisecond => {
                        let a = col
                            .as_any()
                            .downcast_ref::<PrimitiveArray<TimestampMillisecondType>>()
                            .unwrap();
                        (!a.is_null(row_idx)).then(|| a.value(row_idx))
                    }
                    arrow::datatypes::TimeUnit::Microsecond => {
                        let a = col
                            .as_any()
                            .downcast_ref::<PrimitiveArray<TimestampMicrosecondType>>()
                            .unwrap();
                        (!a.is_null(row_idx)).then(|| a.value(row_idx))
                    }
                    arrow::datatypes::TimeUnit::Nanosecond => {
                        let a = col
                            .as_any()
                            .downcast_ref::<PrimitiveArray<TimestampNanosecondType>>()
                            .unwrap();
                        (!a.is_null(row_idx)).then(|| a.value(row_idx))
                    }
                };
                FirstColThreshold::Int64(val)
            }
            DataType::UInt8 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt8Type>>()
                    .unwrap();
                FirstColThreshold::UInt8((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::UInt16 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt16Type>>()
                    .unwrap();
                FirstColThreshold::UInt16((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::UInt32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt32Type>>()
                    .unwrap();
                FirstColThreshold::UInt32((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::UInt64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt64Type>>()
                    .unwrap();
                FirstColThreshold::UInt64((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Float32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Float32Type>>()
                    .unwrap();
                FirstColThreshold::Float32(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx)),
                )
            }
            DataType::Float64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Float64Type>>()
                    .unwrap();
                FirstColThreshold::Float64(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx)),
                )
            }
            DataType::Utf8 => {
                let a = col.as_any().downcast_ref::<StringArray>().unwrap();
                FirstColThreshold::Str(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx).into()),
                )
            }
            DataType::LargeUtf8 => {
                let a = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
                FirstColThreshold::Str(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx).into()),
                )
            }
            DataType::Utf8View => {
                let a = col.as_any().downcast_ref::<StringViewArray>().unwrap();
                FirstColThreshold::Str(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx).into()),
                )
            }
            _ => return,
        };
        self.threshold = Some(threshold);
    }

    fn filter_candidates(&self, view: &BatchView) -> Option<Vec<u32>> {
        let threshold = self.threshold.as_ref()?;
        let batch = view.batch();
        if batch.num_rows() > u32::MAX as usize {
            return None;
        }
        let col = batch.column(self.col_idx);
        let sel = view.selection();
        let mut out = Vec::new();
        match (col.data_type(), threshold) {
            (DataType::Int8, FirstColThreshold::Int8(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int8Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Int16, FirstColThreshold::Int16(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int16Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Int32, FirstColThreshold::Int32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int32Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Date32, FirstColThreshold::Int32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Date32Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Int64, FirstColThreshold::Int64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int64Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Date64, FirstColThreshold::Int64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Date64Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Timestamp(unit, _), FirstColThreshold::Int64(t)) => match unit {
                arrow::datatypes::TimeUnit::Second => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampSecondType>>()?,
                    sel,
                    *t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
                arrow::datatypes::TimeUnit::Millisecond => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampMillisecondType>>()?,
                    sel,
                    *t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
                arrow::datatypes::TimeUnit::Microsecond => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampMicrosecondType>>()?,
                    sel,
                    *t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
                arrow::datatypes::TimeUnit::Nanosecond => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampNanosecondType>>()?,
                    sel,
                    *t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
            },
            (DataType::UInt8, FirstColThreshold::UInt8(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt8Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::UInt16, FirstColThreshold::UInt16(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt16Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::UInt32, FirstColThreshold::UInt32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::UInt64, FirstColThreshold::UInt64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Float32, FirstColThreshold::Float32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Float32Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Float64, FirstColThreshold::Float64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Float64Type>>()?,
                sel,
                *t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Utf8, FirstColThreshold::Str(t)) => {
                let a = col
                    .as_any()
                    .downcast_ref::<GenericByteArray<GenericStringType<i32>>>()?;
                self.scan_str_accessor(
                    a.len(),
                    a.null_count() > 0,
                    |i| (!a.is_null(i)).then(|| a.value(i)),
                    sel,
                    t.as_deref(),
                    &mut out,
                );
            }
            (DataType::LargeUtf8, FirstColThreshold::Str(t)) => {
                let a = col
                    .as_any()
                    .downcast_ref::<GenericByteArray<GenericStringType<i64>>>()?;
                self.scan_str_accessor(
                    a.len(),
                    a.null_count() > 0,
                    |i| (!a.is_null(i)).then(|| a.value(i)),
                    sel,
                    t.as_deref(),
                    &mut out,
                );
            }
            (DataType::Utf8View, FirstColThreshold::Str(t)) => {
                let a = col.as_any().downcast_ref::<StringViewArray>()?;
                self.scan_str_accessor(
                    a.len(),
                    a.null_count() > 0,
                    |i| (!a.is_null(i)).then(|| a.value(i)),
                    sel,
                    t.as_deref(),
                    &mut out,
                );
            }
            _ => return None,
        }
        Some(out)
    }
}

struct RetainedBatch {
    batch: RecordBatch,
    /// Strictly ascending original `row_idx` values gathered into `batch`, or
    /// `None` when `batch` is dense/1-to-1 (`row_idx` maps directly to `batch`).
    row_map: Option<Vec<u32>>,
}

impl RetainedBatch {
    #[inline]
    fn resolve_row(&self, raw_row_idx: u32) -> usize {
        match &self.row_map {
            None => raw_row_idx as usize,
            Some(map) => map
                .binary_search(&raw_row_idx)
                .expect("live heap row_idx must exist in RetainedBatch::row_map"),
        }
    }

    fn row_map_bytes(&self) -> usize {
        self.row_map
            .as_ref()
            .map_or(0, |m| m.capacity().saturating_mul(size_of::<u32>()))
    }
}

pub(super) struct TopNState {
    schema: SchemaRef,
    ordering: LexOrdering,
    row_converter: RowConverter,
    limit: usize,
    heap: BinaryHeap<TopNHeapEntry>,
    batches: Vec<RetainedBatch>,
    retained_rows: usize,
    batch_memory: RecordBatchMemoryTracker,
    row_map_bytes: usize,
    key_bytes: usize,
    first_col_fast: Option<FirstColFastPath>,
    spilled_runs: Vec<SortedRun>,
    spill: Option<SpillContext>,
    batch_size: usize,
    reservation: MemoryReservation,
    #[cfg(test)]
    peak_retained_rows: Arc<AtomicUsize>,
    #[cfg(test)]
    compaction_count: Arc<AtomicUsize>,
    #[cfg(test)]
    heap_rebuild_count: Arc<AtomicUsize>,
}

impl TopNState {
    fn current_memory_bytes(&self) -> usize {
        self.batch_memory
            .memory_usage()
            .saturating_add(self.row_map_bytes)
            .saturating_add(self.key_bytes)
            .saturating_add(
                self.heap
                    .capacity()
                    .saturating_mul(size_of::<TopNHeapEntry>()),
            )
    }

    fn update_reservation(&mut self) -> Result<()> {
        let bytes = self.current_memory_bytes();
        self.reservation.try_resize(bytes)
    }

    fn refresh_threshold(&mut self) {
        if self.heap.len() == self.limit
            && let (Some(fast), Some(top)) = (&mut self.first_col_fast, self.heap.peek())
            && let Some(retained) = self.batches.get(top.batch_idx as usize)
        {
            let dense_row = retained.resolve_row(top.row_idx);
            fast.update_from_batch(&retained.batch, dense_row);
        }
    }

    pub(super) fn add_view(&mut self, view: &BatchView) -> Result<()> {
        if self.limit == 0 || view.is_empty() {
            return Ok(());
        }
        if self.heap.len() == self.limit
            && let Some(fast) = &self.first_col_fast
            && let Some(candidates) = fast.filter_candidates(view)
        {
            if candidates.is_empty() {
                return Ok(());
            }
            if view.is_all() && candidates.len() == view.batch().num_rows() {
                return self.ingest_dense_batch(view.batch());
            }
            let sub = take_record_batch(view.batch(), &UInt32Array::from(candidates))?;
            return self.ingest_dense_batch(&sub);
        }
        if view.is_all() {
            self.ingest_dense_batch(view.batch())
        } else {
            let materialized = view.clone().materialize()?;
            self.ingest_dense_batch(&materialized)
        }
    }

    async fn spill_heap_to_disk(
        &mut self,
        spill_manager: &SpillManager,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if self.heap.is_empty() {
            for old in &self.batches {
                self.batch_memory.uncount_batch(&old.batch);
            }
            self.batches.clear();
            self.retained_rows = 0;
            self.row_map_bytes = 0;
            self.key_bytes = 0;
            self.reservation.resize(0);
            return Ok(());
        }
        let sorted_entries = std::mem::take(&mut self.heap).into_sorted_vec();
        let indices: Vec<(usize, usize)> = sorted_entries
            .iter()
            .map(|e| {
                let b_idx = e.batch_idx as usize;
                let dense_row = self.batches[b_idx].resolve_row(e.row_idx);
                (b_idx, dense_row)
            })
            .collect();
        drop(sorted_entries);
        let batch_refs: Vec<&RecordBatch> =
            self.batches.iter().map(|r| &r.batch).collect();
        let chunk_size = self.batch_size.max(1);
        let mut file = spill_manager.create_in_progress_file("MorselTopNSpill")?;
        let mut max_record_batch_memory = 0usize;
        let res: Result<Option<Arc<dyn datafusion_execution::spill_file::SpillFile>>> =
            async {
                for slab in indices.chunks(chunk_size) {
                    if tx.is_closed() {
                        return Ok(None);
                    }
                    let out = interleave_record_batch(&batch_refs, slab)?;
                    let mem = file.append_batch_async(&out).await?;
                    max_record_batch_memory = max_record_batch_memory.max(mem);
                }
                if tx.is_closed() {
                    return Ok(None);
                }
                file.finish_async().await
            }
            .await;

        for old in &self.batches {
            self.batch_memory.uncount_batch(&old.batch);
        }
        self.batches.clear();
        self.retained_rows = 0;
        self.row_map_bytes = 0;
        self.key_bytes = 0;
        self.reservation.resize(0);
        if let Some(fast) = &mut self.first_col_fast {
            fast.threshold = None;
        }

        match res {
            Ok(Some(finished)) => {
                self.spilled_runs.push(SortedRun::Spilled(SortedSpillFile {
                    file: finished,
                    max_record_batch_memory,
                }));
                Ok(())
            }
            Ok(None) => {
                let _ = file.abort_async().await;
                Ok(())
            }
            Err(err) => {
                let _ = file.abort_async().await;
                Err(err)
            }
        }
    }

    pub(super) async fn add_view_async(
        &mut self,
        view: &BatchView,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if self.limit == 0 || view.is_empty() {
            return Ok(());
        }
        if self.spill.is_none() {
            return self.add_view(view);
        }
        let spill_ctx = self.spill.clone().unwrap();
        let batch = if view.is_all() {
            view.batch().clone()
        } else {
            view.clone().materialize()?
        };
        if spill_ctx.is_spilled() {
            self.spill_heap_to_disk(&spill_ctx.manager, tx).await?;
            if let Some(sf) = spill_batches_sorted(
                &self.schema,
                &self.ordering,
                std::slice::from_ref(&batch),
                self.batch_size,
                &spill_ctx.manager,
                tx,
            )
            .await?
            {
                self.spilled_runs.push(SortedRun::Spilled(sf));
            }
            return Ok(());
        }

        // Pre-check headroom BEFORE mutating `self.heap` so `self.heap` and
        // `self.batches` remain 100% consistent if we transition to spill mode.
        let est = datafusion_common::utils::memory::get_record_batch_memory_size(&batch)
            .saturating_mul(2)
            .saturating_add(batch.num_rows().saturating_mul(64));
        let probe = self.reservation.new_empty();
        let fits = probe.try_grow(est).is_ok();
        drop(probe);
        if !fits {
            spill_ctx.mark_spilled();
            self.spill_heap_to_disk(&spill_ctx.manager, tx).await?;
            if let Some(sf) = spill_batches_sorted(
                &self.schema,
                &self.ordering,
                std::slice::from_ref(&batch),
                self.batch_size,
                &spill_ctx.manager,
                tx,
            )
            .await?
            {
                self.spilled_runs.push(SortedRun::Spilled(sf));
            }
            return Ok(());
        }

        if let Err(err) = self.add_view(&BatchView::all(batch.clone())) {
            if matches!(err, DataFusionError::ResourcesExhausted(_)) {
                spill_ctx.mark_spilled();
                // If `ingest_dense_batch` failed during `compact()` (after
                // pushing `retained`), `self.heap` already includes `batch`!
                // If it failed on `retained` before push, skip out-of-range
                // batch_idx entries and spill `batch` directly.
                let pending_pushed = self
                    .heap
                    .iter()
                    .all(|e| (e.batch_idx as usize) < self.batches.len());
                if pending_pushed {
                    self.spill_heap_to_disk(&spill_ctx.manager, tx).await?;
                } else {
                    // Retain only entries pointing to valid earlier batches,
                    // spill them, then spill `batch` in full.
                    let max_idx = self.batches.len() as u32;
                    let valid: Vec<TopNHeapEntry> = std::mem::take(&mut self.heap)
                        .into_vec()
                        .into_iter()
                        .filter(|e| e.batch_idx < max_idx)
                        .collect();
                    self.heap = BinaryHeap::from(valid);
                    self.spill_heap_to_disk(&spill_ctx.manager, tx).await?;
                    if let Some(sf) = spill_batches_sorted(
                        &self.schema,
                        &self.ordering,
                        std::slice::from_ref(&batch),
                        self.batch_size,
                        &spill_ctx.manager,
                        tx,
                    )
                    .await?
                    {
                        self.spilled_runs.push(SortedRun::Spilled(sf));
                    }
                }
                return Ok(());
            }
            return Err(err);
        }
        Ok(())
    }

    pub(super) async fn finish_async(
        mut self,
        batch_size: usize,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Vec<SortedRun>> {
        if self.spilled_runs.is_empty() {
            let run = self.finish(batch_size)?;
            return Ok(run.into_iter().collect());
        }
        if let Some(spill_ctx) = self.spill.clone() {
            self.spill_heap_to_disk(&spill_ctx.manager, tx).await?;
        }
        Ok(std::mem::take(&mut self.spilled_runs))
    }

    fn ingest_dense_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.limit == 0 || batch.num_rows() == 0 {
            return Ok(());
        }
        let sort_arrays: Vec<ArrayRef> = self
            .ordering
            .iter()
            .map(|sort_expr| {
                Ok(match sort_expr.expr.evaluate(batch)? {
                    ColumnarValue::Array(arr) => arr,
                    ColumnarValue::Scalar(scalar) => {
                        scalar.to_array_of_size(batch.num_rows())?
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let rows = self.row_converter.convert_columns(&sort_arrays)?;
        let pending_batch_idx = u32::try_from(self.batches.len()).map_err(|_| {
            datafusion_common::exec_datafusion_err!("Morsel TopN batch index overflow")
        })?;
        let mut entered_rows: Vec<u32> = Vec::new();
        let mut evicted_pending: Vec<u32> = Vec::new();
        let mut heap_changed = false;

        for row_idx in 0..batch.num_rows() {
            let row = rows.row(row_idx);
            let row_bytes = row.as_ref();
            if self.heap.len() < self.limit {
                let key: Box<[u8]> = row_bytes.into();
                self.key_bytes = self.key_bytes.saturating_add(key.len());
                self.heap.push(TopNHeapEntry {
                    key,
                    batch_idx: pending_batch_idx,
                    row_idx: row_idx as u32,
                });
                entered_rows.push(row_idx as u32);
                heap_changed = true;
            } else if let Some(worst) = self.heap.peek()
                && row_bytes < worst.key.as_ref()
            {
                let evicted = self.heap.pop().unwrap();
                self.key_bytes = self.key_bytes.saturating_sub(evicted.key.len());
                if evicted.batch_idx == pending_batch_idx {
                    evicted_pending.push(evicted.row_idx);
                }
                let key: Box<[u8]> = row_bytes.into();
                self.key_bytes = self.key_bytes.saturating_add(key.len());
                self.heap.push(TopNHeapEntry {
                    key,
                    batch_idx: pending_batch_idx,
                    row_idx: row_idx as u32,
                });
                entered_rows.push(row_idx as u32);
                heap_changed = true;
            }
        }

        let pending_wins = entered_rows.len() - evicted_pending.len();
        if pending_wins == 0 {
            if heap_changed {
                self.update_reservation()?;
                self.refresh_threshold();
            }
            return Ok(());
        }

        // Compute strictly sorted surviving row indices from this batch in O(winners)
        // without touching or rebuilding `self.heap`.
        let take_indices = if evicted_pending.is_empty() {
            entered_rows
        } else {
            evicted_pending.sort_unstable();
            let mut surviving = Vec::with_capacity(pending_wins);
            let mut ev_idx = 0usize;
            for r in entered_rows {
                while ev_idx < evicted_pending.len() && evicted_pending[ev_idx] < r {
                    ev_idx += 1;
                }
                if ev_idx < evicted_pending.len() && evicted_pending[ev_idx] == r {
                    ev_idx += 1;
                } else {
                    surviving.push(r);
                }
            }
            surviving
        };

        let (compacted, row_map) = if take_indices.len() == batch.num_rows() {
            // Every row of `batch` survived (typical while heap is filling): keep
            // 1-to-1 coordinates without a `take_record_batch` copy or `row_map`.
            (gc_view_batch(batch)?, None)
        } else {
            let raw_compacted =
                take_record_batch(batch, &UInt32Array::from(take_indices.clone()))?;
            let compacted = gc_view_batch(&raw_compacted)?;
            drop(raw_compacted);
            (compacted, Some(take_indices))
        };

        let retained = RetainedBatch {
            batch: compacted,
            row_map,
        };
        let compacted_rows = retained.batch.num_rows();
        let added_map_bytes = retained.row_map_bytes();
        self.batch_memory.count_batch(&retained.batch);
        self.row_map_bytes = self.row_map_bytes.saturating_add(added_map_bytes);
        if let Err(err) = self.update_reservation() {
            self.batch_memory.uncount_batch(&retained.batch);
            self.row_map_bytes = self.row_map_bytes.saturating_sub(added_map_bytes);
            return Err(err);
        }
        self.batches.push(retained);
        self.retained_rows = self.retained_rows.saturating_add(compacted_rows);
        #[cfg(test)]
        self.peak_retained_rows
            .fetch_max(self.retained_rows, AtomicOrdering::Relaxed);

        if self.retained_rows > self.limit.saturating_mul(2) && self.batches.len() > 1 {
            self.compact()?;
        }
        self.refresh_threshold();
        Ok(())
    }

    fn compact(&mut self) -> Result<()> {
        #[cfg(test)]
        self.compaction_count.fetch_add(1, AtomicOrdering::Relaxed);
        if self.heap.is_empty() {
            for old in &self.batches {
                self.batch_memory.uncount_batch(&old.batch);
            }
            self.batches.clear();
            self.retained_rows = 0;
            self.row_map_bytes = 0;
            self.update_reservation()?;
            return Ok(());
        }
        #[cfg(test)]
        self.heap_rebuild_count
            .fetch_add(1, AtomicOrdering::Relaxed);
        let mut entries = std::mem::take(&mut self.heap).into_vec();
        let mut interleave_indices = Vec::with_capacity(entries.len());
        for (new_idx, entry) in entries.iter_mut().enumerate() {
            let b_idx = entry.batch_idx as usize;
            let dense_row = self.batches[b_idx].resolve_row(entry.row_idx);
            interleave_indices.push((b_idx, dense_row));
            entry.batch_idx = 0;
            entry.row_idx = new_idx as u32;
        }
        let batch_refs: Vec<&RecordBatch> =
            self.batches.iter().map(|r| &r.batch).collect();
        let raw_compacted = interleave_record_batch(&batch_refs, &interleave_indices)?;
        let compacted = gc_view_batch(&raw_compacted)?;
        drop(raw_compacted);
        let compacted_rows = compacted.num_rows();

        // Account for peak memory while both old and compacted batches are live.
        self.batch_memory.count_batch(&compacted);
        if let Err(err) = self.update_reservation() {
            self.batch_memory.uncount_batch(&compacted);
            return Err(err);
        }

        // Uncount old batches *before* dropping them.
        for old in &self.batches {
            self.batch_memory.uncount_batch(&old.batch);
        }
        self.batches.clear();
        self.batches.push(RetainedBatch {
            batch: compacted,
            row_map: None,
        });
        self.retained_rows = compacted_rows;
        self.row_map_bytes = 0;
        self.heap = BinaryHeap::from(entries);
        self.update_reservation()?;
        Ok(())
    }

    pub(super) fn finish(mut self, batch_size: usize) -> Result<Option<SortedRun>> {
        if self.heap.is_empty() {
            self.reservation.free();
            return Ok(None);
        }
        // `BinaryHeap::into_sorted_vec` produces ascending order = exact SQL sort order.
        let sorted_entries = self.heap.into_sorted_vec();
        let indices: Vec<(usize, usize)> = sorted_entries
            .iter()
            .map(|e| {
                let b_idx = e.batch_idx as usize;
                let dense_row = self.batches[b_idx].resolve_row(e.row_idx);
                (b_idx, dense_row)
            })
            .collect();
        drop(sorted_entries);
        self.key_bytes = 0;
        self.row_map_bytes = 0;

        let batch_refs: Vec<&RecordBatch> =
            self.batches.iter().map(|r| &r.batch).collect();
        let chunk_size = batch_size.max(1);
        // Interleave in slabs of at least 8,192 rows when many small batches are
        // retained so Arrow's per-call interleave table setup is amortized.
        let slab_size = if batch_refs.len() > 16 {
            chunk_size.max(8192)
        } else {
            chunk_size
        };
        let has_view = schema_has_view(batch_refs[0].schema().as_ref());
        let mut out_batches = VecDeque::with_capacity(indices.len().div_ceil(chunk_size));
        let mut out_tracker = RecordBatchMemoryTracker::new();
        for slab_indices in indices.chunks(slab_size) {
            let raw_slab = interleave_record_batch(&batch_refs, slab_indices)?;
            let slab = if has_view {
                gc_view_batch(&raw_slab)?
            } else {
                raw_slab
            };
            let mut offset = 0;
            while offset < slab.num_rows() {
                let len = (slab.num_rows() - offset).min(chunk_size);
                let out = if offset == 0 && len == slab.num_rows() {
                    slab.clone()
                } else {
                    let sliced = slab.slice(offset, len);
                    if has_view {
                        gc_view_batch(&sliced)?
                    } else {
                        sliced
                    }
                };
                offset += len;
                self.batch_memory.count_batch(&out);
                out_tracker.count_batch(&out);
                let peak = self.batch_memory.memory_usage();
                if let Err(err) = self.reservation.try_resize(peak) {
                    self.batch_memory.uncount_batch(&out);
                    out_tracker.uncount_batch(&out);
                    for b in &out_batches {
                        self.batch_memory.uncount_batch(b);
                        out_tracker.uncount_batch(b);
                    }
                    return Err(err);
                }
                out_batches.push_back(out);
            }
        }
        for old in &self.batches {
            self.batch_memory.uncount_batch(&old.batch);
        }
        drop(self.batches);
        self.reservation.resize(out_tracker.memory_usage());
        Ok(Some(SortedRun::Batches(Box::new(BatchedRun {
            batches: out_batches,
            tracker: out_tracker,
            reservation: self.reservation,
        }))))
    }
}

// ---------------------------------------------------------------------------
// Native Parallel Sort (ORDER BY ... without limit, or OFFSET-only)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct SortSink {
    schema: SchemaRef,
    ordering: LexOrdering,
    skip: usize,
    fetch: Option<usize>,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
    #[cfg(test)]
    pub(super) completed_run_count: Arc<AtomicUsize>,
}

impl SortSink {
    pub(super) fn try_new(
        schema: SchemaRef,
        ordering: LexOrdering,
        skip: usize,
        fetch: Option<usize>,
    ) -> Result<std::result::Result<Self, String>> {
        if let Err(reason) = validate_sort_ordering(&ordering, schema.as_ref())? {
            return Ok(Err(reason));
        }
        Ok(Ok(Self {
            schema,
            ordering,
            skip,
            fetch,
            #[cfg(test)]
            finalize_gate: None,
            #[cfg(test)]
            completed_run_count: Arc::new(AtomicUsize::new(0)),
        }))
    }

    pub(super) fn input_schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub(super) fn ordering(&self) -> &LexOrdering {
        &self.ordering
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(self.ordering.iter().map(|s| &s.expr), f)
    }

    pub(super) fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortSink[order=[{}]", self.ordering)?;
        if let Some(fetch) = self.fetch {
            write!(f, ", limit={fetch}")?;
        }
        if self.skip > 0 {
            write!(f, ", offset={}", self.skip)?;
        }
        write!(f, "]")
    }

    #[cfg_attr(not(test), expect(dead_code))]
    pub(super) fn new_state(&self, shared_memory: Arc<SharedSortMemory>) -> SortState {
        self.new_state_with_spill(shared_memory, None, 8192)
    }

    pub(super) fn new_state_with_spill(
        &self,
        shared_memory: Arc<SharedSortMemory>,
        spill: Option<SpillContext>,
        batch_size: usize,
    ) -> SortState {
        SortState {
            schema: Arc::clone(&self.schema),
            ordering: self.ordering.clone(),
            batches: Vec::new(),
            buffered_rows: 0,
            local_counter: RecordBatchMemoryCounter::new(),
            first_batch_bytes: None,
            completed_runs: Vec::new(),
            shared_memory,
            spill,
            batch_size: batch_size.max(1),
            #[cfg(test)]
            completed_run_count: Arc::clone(&self.completed_run_count),
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub(super) async fn finalize_and_emit(
        &self,
        runs: Vec<SortedRun>,
        result_operators: &[Operator],
        batch_size: usize,
        memory_pool: &Arc<dyn MemoryPool>,
        spill: Option<SpillContext>,
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        emit_sorted_runs(
            runs,
            EmitSortedRunsConfig {
                schema: Arc::clone(&self.schema),
                ordering: &self.ordering,
                skip: self.skip,
                fetch: self.fetch,
                result_operators,
                batch_size,
                memory_pool,
                consumer_name: "MorselSortMerge",
                spill,
            },
            tx,
            measurement,
        )
        .await
    }
}

pub(super) struct SortState {
    schema: SchemaRef,
    ordering: LexOrdering,
    batches: Vec<RecordBatch>,
    buffered_rows: usize,
    local_counter: RecordBatchMemoryCounter,
    first_batch_bytes: Option<usize>,
    completed_runs: Vec<SortedRun>,
    shared_memory: Arc<SharedSortMemory>,
    spill: Option<SpillContext>,
    batch_size: usize,
    #[cfg(test)]
    completed_run_count: Arc<AtomicUsize>,
}

impl Drop for SortState {
    fn drop(&mut self) {
        self.shared_memory.uncount_batches(&self.batches);
    }
}

impl SortState {
    #[cfg(test)]
    pub(super) fn add_view(&mut self, view: BatchView) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        let batch = view.materialize()?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let incremental_bytes = self
            .local_counter
            .memory_usage()
            .saturating_sub(self.first_batch_bytes.unwrap_or(0));
        if !self.batches.is_empty()
            && (incremental_bytes >= MAX_LOCAL_SORT_RUN_BYTES
                || self
                    .buffered_rows
                    .checked_add(batch.num_rows())
                    .is_none_or(|n| n > u32::MAX as usize))
        {
            self.flush_current_batches()?;
        }
        self.shared_memory.count_batch(&batch)?;
        self.local_counter.count_batch(&batch);
        if self.first_batch_bytes.is_none() {
            self.first_batch_bytes = Some(self.local_counter.memory_usage());
        }
        self.buffered_rows += batch.num_rows();
        self.batches.push(batch);
        Ok(())
    }

    pub(super) async fn add_view_async(
        &mut self,
        view: BatchView,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        let batch = view.materialize()?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let incremental_bytes = self
            .local_counter
            .memory_usage()
            .saturating_sub(self.first_batch_bytes.unwrap_or(0));
        let flush_limit = if self.shared_memory.is_spilled() {
            MAX_SPILLED_LOCAL_SORT_RUN_BYTES
        } else {
            MAX_LOCAL_SORT_RUN_BYTES
        };
        if !self.batches.is_empty()
            && (incremental_bytes >= flush_limit
                || self
                    .buffered_rows
                    .checked_add(batch.num_rows())
                    .is_none_or(|n| n > u32::MAX as usize))
        {
            self.flush_current_batches_async(tx).await?;
        }

        if let Err(err) = self.shared_memory.count_batch(&batch) {
            if matches!(err, DataFusionError::ResourcesExhausted(_))
                && let Some(spill_ctx) = self.spill.clone()
            {
                self.shared_memory.mark_spilled();
                self.spill_in_memory_completed_runs(&spill_ctx.manager, tx)
                    .await?;
                self.spill_current_batches(&spill_ctx.manager, tx).await?;
                if tx.is_closed() {
                    return Ok(());
                }
                if self.shared_memory.count_batch(&batch).is_ok() {
                    self.local_counter.count_batch(&batch);
                    if self.first_batch_bytes.is_none() {
                        self.first_batch_bytes = Some(self.local_counter.memory_usage());
                    }
                    self.buffered_rows += batch.num_rows();
                    self.batches.push(batch);
                    return Ok(());
                }
                // If no sort memory is reserved at all and even the minimum spill
                // headroom doesn't fit in the pool, fail fast with ResourcesExhausted.
                if self.shared_memory.total_reserved() == 0
                    && let Err(headroom_err) = self
                        .shared_memory
                        .check_spill_headroom(spill_ctx.min_headroom_bytes)
                {
                    return Err(headroom_err);
                }
                // Otherwise another worker currently holds memory or this single batch
                // exceeds remaining RAM: sort and spill this batch directly.
                if let Some(sf) = spill_batches_sorted(
                    &self.schema,
                    &self.ordering,
                    std::slice::from_ref(&batch),
                    self.batch_size,
                    &spill_ctx.manager,
                    tx,
                )
                .await?
                {
                    #[cfg(test)]
                    self.completed_run_count
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    self.completed_runs.push(SortedRun::Spilled(sf));
                }
                return Ok(());
            }
            return Err(err);
        }

        self.local_counter.count_batch(&batch);
        if self.first_batch_bytes.is_none() {
            self.first_batch_bytes = Some(self.local_counter.memory_usage());
        }
        self.buffered_rows += batch.num_rows();
        self.batches.push(batch);
        Ok(())
    }

    async fn spill_in_memory_completed_runs(
        &mut self,
        spill_manager: &SpillManager,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        let existing = std::mem::take(&mut self.completed_runs);
        for run in existing {
            if tx.is_closed() {
                return Ok(());
            }
            match run {
                SortedRun::Spilled(sf) => {
                    self.completed_runs.push(SortedRun::Spilled(sf))
                }
                SortedRun::Indexed(mut ir) => {
                    let sf = if let Some(backing) = ir.batch.as_ref() {
                        spill_sorted_indexed_run(
                            backing,
                            &ir.indices,
                            self.batch_size,
                            spill_manager,
                            tx,
                        )
                        .await?
                    } else {
                        None
                    };
                    ir.release();
                    if let Some(sf) = sf {
                        self.completed_runs.push(SortedRun::Spilled(sf));
                    }
                }
                SortedRun::Batches(br) => {
                    if let Some(sf) = spill_batched_run(br, spill_manager, tx).await? {
                        self.completed_runs.push(SortedRun::Spilled(sf));
                    }
                }
            }
        }
        Ok(())
    }

    async fn spill_current_batches(
        &mut self,
        spill_manager: &SpillManager,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if self.batches.is_empty() {
            return Ok(());
        }
        let tracked = TrackedBatches {
            batches: std::mem::take(&mut self.batches),
            shared: Arc::clone(&self.shared_memory),
        };
        self.buffered_rows = 0;
        self.local_counter = RecordBatchMemoryCounter::new();
        self.first_batch_bytes = None;

        let sf = spill_batches_sorted(
            &self.schema,
            &self.ordering,
            &tracked.batches,
            self.batch_size,
            spill_manager,
            tx,
        )
        .await?;
        drop(tracked);
        if let Some(sf) = sf {
            #[cfg(test)]
            self.completed_run_count
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.completed_runs.push(SortedRun::Spilled(sf));
        }
        Ok(())
    }

    async fn flush_current_batches_async(
        &mut self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if self.batches.is_empty() {
            return Ok(());
        }
        if self.shared_memory.is_spilled()
            && let Some(spill_ctx) = self.spill.clone()
        {
            self.spill_in_memory_completed_runs(&spill_ctx.manager, tx)
                .await?;
            return self.spill_current_batches(&spill_ctx.manager, tx).await;
        }
        // Try in-memory flush; if OOM occurs during concat/lexsort and spilling is
        // enabled, spill cleanly instead of failing.
        let tracked = TrackedBatches {
            batches: std::mem::take(&mut self.batches),
            shared: Arc::clone(&self.shared_memory),
        };
        self.buffered_rows = 0;
        self.local_counter = RecordBatchMemoryCounter::new();
        self.first_batch_bytes = None;

        match self.try_flush_tracked_in_memory(&tracked) {
            Ok(run) => {
                drop(tracked);
                #[cfg(test)]
                self.completed_run_count
                    .fetch_add(1, AtomicOrdering::Relaxed);
                self.completed_runs.push(run);
                Ok(())
            }
            Err(err)
                if matches!(err, DataFusionError::ResourcesExhausted(_))
                    && self.spill.is_some() =>
            {
                let spill_ctx = self.spill.clone().unwrap();
                self.shared_memory.mark_spilled();
                self.spill_in_memory_completed_runs(&spill_ctx.manager, tx)
                    .await?;
                let sf = spill_batches_sorted(
                    &self.schema,
                    &self.ordering,
                    &tracked.batches,
                    self.batch_size,
                    &spill_ctx.manager,
                    tx,
                )
                .await?;
                drop(tracked);
                if let Some(sf) = sf {
                    #[cfg(test)]
                    self.completed_run_count
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    self.completed_runs.push(SortedRun::Spilled(sf));
                }
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn try_flush_tracked_in_memory(&self, tracked: &TrackedBatches) -> Result<SortedRun> {
        let merged_guard = if tracked.batches.len() == 1 {
            let single = tracked.batches[0].clone();
            self.shared_memory.count_batch(&single)?;
            TrackedBatch {
                batch: Some(single),
                shared: Arc::clone(&self.shared_memory),
            }
        } else {
            let merged = concat_batches_compact_views(&self.schema, &tracked.batches)?;
            self.shared_memory.count_batch(&merged)?;
            TrackedBatch {
                batch: Some(merged),
                shared: Arc::clone(&self.shared_memory),
            }
        };

        let merged_ref = merged_guard.batch.as_ref().expect("merged batch");
        let sort_columns = self
            .ordering
            .iter()
            .map(|e| e.evaluate_to_sort_column(merged_ref))
            .collect::<Result<Vec<_>>>()?;
        let est_indices_bytes = merged_ref.num_rows().saturating_mul(size_of::<u32>());
        let mut indices_guard =
            TrackedExtra::try_new(Arc::clone(&self.shared_memory), est_indices_bytes)?;
        let indices = lexsort_to_indices(&sort_columns, None)?;
        drop(sort_columns);
        indices_guard.try_resize(indices.get_array_memory_size())?;

        let merged = merged_guard.into_inner();
        let indices_bytes = indices_guard.into_bytes();
        Ok(SortedRun::Indexed(IndexedRun {
            batch: Some(merged),
            indices,
            indices_bytes,
            shared: Arc::clone(&self.shared_memory),
        }))
    }

    #[cfg(test)]
    fn flush_current_batches(&mut self) -> Result<()> {
        if self.batches.is_empty() {
            return Ok(());
        }
        let tracked = TrackedBatches {
            batches: std::mem::take(&mut self.batches),
            shared: Arc::clone(&self.shared_memory),
        };
        self.buffered_rows = 0;
        self.local_counter = RecordBatchMemoryCounter::new();
        self.first_batch_bytes = None;

        let merged_guard = if tracked.batches.len() == 1 {
            TrackedBatch {
                batch: Some(tracked.take_single()),
                shared: Arc::clone(&self.shared_memory),
            }
        } else {
            let merged = concat_batches_compact_views(&self.schema, &tracked.batches)?;
            self.shared_memory.count_batch(&merged)?;
            let guard = TrackedBatch {
                batch: Some(merged),
                shared: Arc::clone(&self.shared_memory),
            };
            drop(tracked);
            guard
        };

        let merged_ref = merged_guard.batch.as_ref().expect("merged batch");
        let sort_columns = self
            .ordering
            .iter()
            .map(|e| e.evaluate_to_sort_column(merged_ref))
            .collect::<Result<Vec<_>>>()?;
        let est_indices_bytes = merged_ref.num_rows().saturating_mul(size_of::<u32>());
        let mut indices_guard =
            TrackedExtra::try_new(Arc::clone(&self.shared_memory), est_indices_bytes)?;
        let indices = lexsort_to_indices(&sort_columns, None)?;
        drop(sort_columns);
        indices_guard.try_resize(indices.get_array_memory_size())?;

        let merged = merged_guard.into_inner();
        let indices_bytes = indices_guard.into_bytes();
        #[cfg(test)]
        self.completed_run_count
            .fetch_add(1, AtomicOrdering::Relaxed);

        self.completed_runs.push(SortedRun::Indexed(IndexedRun {
            batch: Some(merged),
            indices,
            indices_bytes,
            shared: Arc::clone(&self.shared_memory),
        }));
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn finish(mut self) -> Result<Vec<SortedRun>> {
        self.flush_current_batches()?;
        Ok(std::mem::take(&mut self.completed_runs))
    }

    pub(super) async fn finish_async(
        mut self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Vec<SortedRun>> {
        self.flush_current_batches_async(tx).await?;
        if self.shared_memory.is_spilled()
            && let Some(spill_ctx) = self.spill.clone()
        {
            self.spill_in_memory_completed_runs(&spill_ctx.manager, tx)
                .await?;
        }
        Ok(std::mem::take(&mut self.completed_runs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Float64Array, Int32Array, StringArray, StringViewBuilder};
    use arrow::datatypes::Field;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, UnboundedMemoryPool,
    };
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::Column;

    fn test_pool(limit: Option<usize>) -> Arc<dyn MemoryPool> {
        match limit {
            Some(bytes) => Arc::new(GreedyMemoryPool::new(bytes)),
            None => Arc::new(UnboundedMemoryPool::default()),
        }
    }

    #[test]
    fn topn_bounds_retained_rows_under_adversarial_reverse_input() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let ordering = LexOrdering::from([PhysicalSortExpr::new(
            Arc::new(Column::new("k", 0)),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )]);
        let k = 10;
        let offset = 5;
        let limit = k + offset;
        let sink = TopNSink::try_new(Arc::clone(&schema), ordering, offset, k)?
            .expect("supported");
        let pool = test_pool(None);
        let res = MemoryConsumer::new("test_topn").register(&pool);
        let mut state = sink.new_state(res)?;

        for b in (0..50).rev() {
            let start = b * 1000;
            let ks: Vec<i32> = (start..start + 1000).rev().collect();
            let payloads: Vec<String> = ks
                .iter()
                .map(|v| format!("row-{v:06}-padding-payload"))
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(ks)),
                    Arc::new(StringArray::from(payloads)),
                ],
            )?;
            state.add_view(&BatchView::all(batch))?;
            assert!(
                state.retained_rows <= 2 * limit,
                "retained_rows {} exceeded 2 * limit ({})",
                state.retained_rows,
                2 * limit
            );
        }

        let peak = sink.peak_retained_rows.load(AtomicOrdering::Relaxed);
        assert!(
            peak <= 3 * limit,
            "peak_retained_rows {peak} exceeded 3 * limit ({})",
            3 * limit
        );

        let run = state.finish(1024)?.expect("non-empty run");
        let SortedRun::Batches(run) = run else {
            unreachable!()
        };
        let total_rows: usize = run.batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, limit);
        let ks = run.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let expected: Vec<i32> = (0..limit as i32).collect();
        assert_eq!(ks.values().as_ref(), expected.as_slice());

        drop(run);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn topn_utf8view_zero_copy_slices_bounds_memory_in_bytes() -> Result<()> {
        // Repro from review: 6.6 MB Utf8View backing buffer stored as zero-copy slices,
        // ORDER BY id DESC LIMIT 10 under a tight memory pool (much smaller than slices * 6.6 MB).
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("v", DataType::Utf8View, false),
        ]));
        let total_rows = 100_000usize;
        let ids: Vec<i32> = (0..total_rows as i32).collect();
        let mut vb = StringViewBuilder::with_capacity(total_rows);
        for i in 0..total_rows {
            vb.append_value(format!(
                "view-row-{i:06}-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            ));
        }
        let big_batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(ids)), Arc::new(vb.finish())],
        )?;

        let ordering = LexOrdering::from([PhysicalSortExpr::new(
            Arc::new(Column::new("id", 0)),
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        )]);
        let sink = TopNSink::try_new(Arc::clone(&schema), ordering, 0, 10)?.unwrap();
        // 512 KiB pool is << 6.6 MB single backing buffer and << 1.3 GB uncompacted slices!
        let pool = test_pool(Some(512 * 1024));
        let mut state =
            sink.new_state(MemoryConsumer::new("test_view_topn").register(&pool))?;

        // Feed 200 zero-copy slices in ascending id order so EVERY slice beats the heap!
        for start in (0..total_rows).step_by(500) {
            let slice = big_batch.slice(start, 500);
            state.add_view(&BatchView::all(slice))?;
            assert!(
                pool.reserved() < 128 * 1024,
                "TopN reserved {} bytes on Utf8View slice (expected << 128 KiB)",
                pool.reserved()
            );
        }

        let run = state.finish(16)?.unwrap();
        let SortedRun::Batches(run) = run else {
            unreachable!()
        };
        let got = run.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let expected: Vec<i32> = ((total_rows - 10) as i32..total_rows as i32)
            .rev()
            .collect();
        assert_eq!(got.values().as_ref(), expected.as_slice());
        drop(run);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn sort_state_shared_utf8view_buffers_produce_one_run_with_deduped_buffers()
    -> Result<()> {
        // Simulate 500 small probe batches (batch_size=256) all sharing the same
        // multi-MB Utf8View data buffers from a hash-join build table.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8View, false),
        ]));
        let dim_rows = 50_000usize;
        let mut vb = StringViewBuilder::with_capacity(dim_rows);
        for i in 0..dim_rows {
            vb.append_value(format!(
                "dim-product-name-{i:06}-padding-to-exceed-inline-threshold"
            ));
        }
        let dim_views = Arc::new(vb.finish()) as ArrayRef;
        let dim_buf_bytes: usize = dim_views
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .data_buffers()
            .iter()
            .map(|b| b.len())
            .sum();
        assert!(dim_buf_bytes > 2 * 1024 * 1024);

        let ordering = LexOrdering::from([
            PhysicalSortExpr::new(
                Arc::new(Column::new("name", 1)),
                SortOptions::default(),
            ),
            PhysicalSortExpr::new(Arc::new(Column::new("id", 0)), SortOptions::default()),
        ]);
        let sink = SortSink::try_new(Arc::clone(&schema), ordering, 0, None)?.unwrap();
        // Pool far smaller than 200 * dim_buf_bytes (~600 MB), but easily holds 1x shared buffers + rows.
        let pool = test_pool(Some(32 * 1024 * 1024));
        let shared = SharedSortMemory::new(
            MemoryConsumer::new("test_sort_shared").register(&pool),
        );
        let mut state = sink.new_state(Arc::clone(&shared));

        for b in 0..200 {
            let take_idx = UInt32Array::from_iter_values(
                (0..256).map(|i| ((b * 256 + i) % dim_rows) as u32),
            );
            let taken_names = arrow::compute::take(dim_views.as_ref(), &take_idx, None)?;
            let ids = Arc::new(Int32Array::from_iter_values(
                (0..256).map(|i| (b * 256 + i) as i32),
            ));
            let batch =
                RecordBatch::try_new(Arc::clone(&schema), vec![ids, taken_names])?;
            state.add_view(BatchView::all(batch))?;
        }

        let runs = state.finish()?;
        assert_eq!(
            runs.len(),
            1,
            "expected 1 sorted run, not per-batch flushes"
        );
        assert_eq!(sink.completed_run_count.load(AtomicOrdering::Relaxed), 1);
        let SortedRun::Indexed(run) = &runs[0] else {
            unreachable!()
        };
        let merged_sv = run
            .batch
            .as_ref()
            .unwrap()
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        let dim_buf_count = dim_views
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .data_buffers()
            .len();
        assert!(
            merged_sv.data_buffers().len() <= dim_buf_count,
            "expected deduplicated/compacted view buffers <= {dim_buf_count}, got {}",
            merged_sv.data_buffers().len()
        );

        drop(runs);
        drop(shared);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn topn_first_col_fast_path_handles_floats_strings_and_nulls() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float64, true),
            Field::new("s", DataType::Utf8View, true),
            Field::new("id", DataType::Int32, false),
        ]));
        let values = vec![
            Some(f64::NAN),
            Some(-f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(0.0),
            Some(-0.0),
            Some(1.5),
            Some(-1.5),
            None,
            None,
        ];
        let strings = vec![
            Some("zebra-long-string-value-001"),
            Some("alpha-long-string-value-002"),
            None,
            Some("alpha-long-string-value-002"),
            Some("short"),
            Some(""),
            Some("middle-long-string-value-003"),
            None,
            Some("beta-long-string-value-004"),
            Some("alpha"),
        ];
        let ids: Vec<i32> = (0..values.len() as i32).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float64Array::from(values)),
                Arc::new(StringViewArray::from(strings)),
                Arc::new(Int32Array::from(ids)),
            ],
        )?;

        for col_idx in [0usize, 1usize] {
            let col_name = schema.field(col_idx).name();
            for descending in [false, true] {
                for nulls_first in [false, true] {
                    let options = SortOptions {
                        descending,
                        nulls_first,
                    };
                    let ordering = LexOrdering::from([
                        PhysicalSortExpr::new(
                            Arc::new(Column::new(col_name, col_idx)),
                            options,
                        ),
                        PhysicalSortExpr::new(
                            Arc::new(Column::new("id", 2)),
                            SortOptions::default(),
                        ),
                    ]);
                    let sink =
                        TopNSink::try_new(Arc::clone(&schema), ordering.clone(), 0, 4)?
                            .unwrap();
                    let pool = test_pool(None);
                    let mut state = sink.new_state(
                        MemoryConsumer::new("test_fast_path").register(&pool),
                    )?;
                    for i in 0..batch.num_rows() {
                        state.add_view(&BatchView::all(batch.slice(i, 1)))?;
                    }
                    let run = state.finish(16)?.unwrap();
                    let SortedRun::Batches(run) = run else {
                        unreachable!()
                    };
                    let got_ids = run.batches[0]
                        .column(2)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap();

                    let expected = datafusion_physical_plan::sorts::sort::sort_batch(
                        &batch,
                        &ordering,
                        Some(4),
                    )?;
                    let exp_ids = expected
                        .column(2)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap();
                    assert_eq!(
                        got_ids.values().as_ref(),
                        exp_ids.values().as_ref(),
                        "mismatch for col={col_name}, descending={descending}, nulls_first={nulls_first}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn sort_state_oom_cleans_up_reservation() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let ordering = LexOrdering::from([PhysicalSortExpr::new(
            Arc::new(Column::new("k", 0)),
            SortOptions::default(),
        )]);
        let sink = SortSink::try_new(Arc::clone(&schema), ordering, 0, None)?.unwrap();
        let pool = test_pool(Some(128));
        let shared =
            SharedSortMemory::new(MemoryConsumer::new("test_sort").register(&pool));
        let mut state = sink.new_state(Arc::clone(&shared));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(0..1024))],
        )?;
        let err = state.add_view(BatchView::all(batch)).unwrap_err();
        assert!(
            matches!(err, DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        drop(state);
        drop(shared);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn topn_heap_rebuilds_track_compactions_not_input_batches() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
        ]));
        let ordering = LexOrdering::from([PhysicalSortExpr::new(
            Arc::new(Column::new("k", 0)),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )]);

        // Case 1: LIMIT > total_rows across 400 small 256-row batches -> 0 compactions, 0 rebuilds.
        {
            let sink =
                TopNSink::try_new(Arc::clone(&schema), ordering.clone(), 0, 200_000)?
                    .unwrap();
            let pool = test_pool(None);
            let mut state = sink.new_state(
                MemoryConsumer::new("test_large_k_unfilled").register(&pool),
            )?;
            for b in (0..400).rev() {
                let start = b * 256;
                let ks: Vec<i32> = (start..start + 256).rev().collect();
                let vs: Vec<i32> = ks.iter().map(|&x| x * 2).collect();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int32Array::from(ks)),
                        Arc::new(Int32Array::from(vs)),
                    ],
                )?;
                state.add_view(&BatchView::all(batch))?;
            }
            assert_eq!(sink.compaction_count.load(AtomicOrdering::Relaxed), 0);
            assert_eq!(sink.heap_rebuild_count.load(AtomicOrdering::Relaxed), 0);
            let run = state.finish(256)?.unwrap();
            let SortedRun::Batches(run) = run else {
                unreachable!()
            };
            let mut got = Vec::new();
            for b in &run.batches {
                let ks = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                got.extend_from_slice(ks.values());
            }
            let expected: Vec<i32> = (0..400 * 256).collect();
            assert_eq!(got, expected);
            drop(run);
            assert_eq!(pool.reserved(), 0);
        }

        // Case 2: LIMIT < total_rows across 400 small 256-row batches (reverse order
        // plus partial-batch winners + within-batch self-evictions).
        // Full-heap rebuilds must equal compactions (NOT 400 input batches), and
        // peak_retained_rows remains <= 3 * (K+O).
        {
            let limit = 10_000usize;
            let sink =
                TopNSink::try_new(Arc::clone(&schema), ordering, 0, limit)?.unwrap();
            let pool = test_pool(None);
            let mut state = sink
                .new_state(MemoryConsumer::new("test_large_k_filled").register(&pool))?;
            for b in (0..400).rev() {
                let start = b * 256;
                // Mix low keys and high keys inside each batch to exercise partial
                // winners and `row_map` binary search + self-eviction.
                let ks: Vec<i32> = (0..256)
                    .map(|i| {
                        if i % 2 == 0 {
                            start + (255 - i)
                        } else {
                            1_000_000 + start + i
                        }
                    })
                    .collect();
                let vs: Vec<i32> = ks.iter().map(|&x| x + 7).collect();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int32Array::from(ks)),
                        Arc::new(Int32Array::from(vs)),
                    ],
                )?;
                state.add_view(&BatchView::all(batch))?;
            }
            let compactions = sink.compaction_count.load(AtomicOrdering::Relaxed);
            let rebuilds = sink.heap_rebuild_count.load(AtomicOrdering::Relaxed);
            assert_eq!(
                rebuilds, compactions,
                "full-heap rebuilds ({rebuilds}) must track compactions ({compactions}), not input batches"
            );
            assert!(
                rebuilds <= 10,
                "expected few compactions/rebuilds (<= 10) across 400 batches, got {rebuilds}"
            );
            let peak = sink.peak_retained_rows.load(AtomicOrdering::Relaxed);
            assert!(
                peak <= 3 * limit,
                "peak_retained_rows {peak} exceeded 3 * limit ({})",
                3 * limit
            );

            let run = state.finish(256)?.unwrap();
            let SortedRun::Batches(run) = run else {
                unreachable!()
            };
            let mut got_k = Vec::new();
            let mut got_v = Vec::new();
            for b in &run.batches {
                got_k.extend_from_slice(
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values(),
                );
                got_v.extend_from_slice(
                    b.column(1)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values(),
                );
            }
            assert_eq!(got_k.len(), limit);
            // Verify sorted order and exact value pairing (proving row_map resolution!).
            for i in 0..limit {
                assert_eq!(got_v[i], got_k[i] + 7);
                if i > 0 {
                    assert!(got_k[i - 1] <= got_k[i]);
                }
            }
            drop(run);
            assert_eq!(pool.reserved(), 0);
        }
        Ok(())
    }
}
