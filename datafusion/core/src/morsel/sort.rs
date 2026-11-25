// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native parallel TopN (`ORDER BY ... LIMIT K [OFFSET O]`) and full parallel
//! Sort (`ORDER BY ...`) sinks for the experimental morsel engine.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, VecDeque};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::task::{Context, Poll};

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray, RecordBatch, RecordBatchOptions,
    UInt32Array,
};
use arrow::compute::{
    SortOptions, concat_batches, interleave_record_batch, lexsort_to_indices,
    take_arrays, take_record_batch,
};
use arrow::datatypes::{
    ArrowNativeTypeOp, DataType, Date32Type, Date64Type, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, Schema, SchemaRef,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow::row::{RowConverter, SortField};
use datafusion_common::Result;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr};
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::sorts::streaming_merge::StreamingMergeBuilder;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use super::batch::{BatchView, Selection};
use super::metrics::{self, RunMetrics};
use super::{Operator, apply_operators};

/// Maximum buffered bytes per worker-local sort run before flushing to a sorted run.
/// Bounds peak temporary memory during `concat_batches` while keeping merge fan-in tiny.
const MAX_LOCAL_SORT_RUN_BYTES: usize = 64 * 1024 * 1024;

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

/// A worker-local sorted run ready for k-way merge or direct emission.
pub(super) enum SortedRun {
    /// Lazily slices `indices` and gathers `batch_size` chunks on demand.
    Indexed {
        batch: RecordBatch,
        indices: UInt32Array,
        reservation: MemoryReservation,
    },
    /// Already-materialized sorted chunks (used by TopN).
    Batches {
        batches: Vec<RecordBatch>,
        reservation: MemoryReservation,
    },
}

struct SortedRunStream {
    schema: SchemaRef,
    state: SortedRunStreamState,
}

enum SortedRunStreamState {
    Indexed {
        batch: Option<RecordBatch>,
        indices: UInt32Array,
        offset: usize,
        batch_size: usize,
        reservation: MemoryReservation,
    },
    Batches {
        batches: VecDeque<RecordBatch>,
        reservation: MemoryReservation,
    },
}

impl SortedRunStream {
    fn from_run(run: SortedRun, schema: SchemaRef, batch_size: usize) -> Self {
        let state = match run {
            SortedRun::Indexed {
                batch,
                indices,
                reservation,
            } => SortedRunStreamState::Indexed {
                batch: Some(batch),
                indices,
                offset: 0,
                batch_size: batch_size.max(1),
                reservation,
            },
            SortedRun::Batches {
                batches,
                reservation,
            } => SortedRunStreamState::Batches {
                batches: batches.into(),
                reservation,
            },
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
                batch,
                indices,
                offset,
                batch_size,
                reservation,
            } => {
                if *offset >= indices.len() {
                    *batch = None;
                    reservation.free();
                    return Poll::Ready(None);
                }
                let Some(backing) = batch.as_ref() else {
                    reservation.free();
                    return Poll::Ready(None);
                };
                let len = (indices.len() - *offset).min(*batch_size);
                let slice = indices.slice(*offset, len);
                *offset += len;
                let result =
                    take_arrays(backing.columns(), &slice, None).and_then(|columns| {
                        let options = RecordBatchOptions::new().with_row_count(Some(len));
                        RecordBatch::try_new_with_options(
                            backing.schema(),
                            columns,
                            &options,
                        )
                    });
                if *offset >= indices.len() {
                    *batch = None;
                    reservation.free();
                }
                Poll::Ready(Some(result.map_err(Into::into)))
            }
            SortedRunStreamState::Batches {
                batches,
                reservation,
            } => {
                let Some(next) = batches.pop_front() else {
                    reservation.free();
                    return Poll::Ready(None);
                };
                let bytes = get_record_batch_memory_size(&next);
                let _ = reservation.try_shrink(bytes);
                if batches.is_empty() {
                    reservation.free();
                }
                Poll::Ready(Some(Ok(next)))
            }
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
    } = cfg;
    if tx.is_closed() || runs.is_empty() || fetch == Some(0) {
        return Ok(());
    }

    // Fast path: a single sorted run does not need k-way merge.
    if runs.len() == 1 {
        let run = runs.into_iter().next().unwrap();
        match run {
            SortedRun::Indexed {
                batch,
                indices,
                reservation,
            } => {
                let total = indices.len();
                let start = skip.min(total);
                let remaining = total - start;
                let len = fetch.map_or(remaining, |f| f.min(remaining));
                if len == 0 {
                    reservation.free();
                    return Ok(());
                }
                let sliced = indices.slice(start, len);
                let chunk_size = batch_size.max(1);
                let mut offset = 0;
                while offset < sliced.len() {
                    if tx.is_closed() {
                        reservation.free();
                        return Ok(());
                    }
                    let take_len = (sliced.len() - offset).min(chunk_size);
                    let chunk_idx = sliced.slice(offset, take_len);
                    offset += take_len;
                    let columns = take_arrays(batch.columns(), &chunk_idx, None)?;
                    let options =
                        RecordBatchOptions::new().with_row_count(Some(take_len));
                    let out = RecordBatch::try_new_with_options(
                        batch.schema(),
                        columns,
                        &options,
                    )?;
                    let projected = apply_operators(result_operators, out)?;
                    if projected.num_rows() > 0
                        && !send_batch(&tx, projected, measurement).await
                    {
                        reservation.free();
                        return Ok(());
                    }
                }
                reservation.free();
                return Ok(());
            }
            SortedRun::Batches {
                batches,
                reservation,
            } => {
                let mut remaining_skip = skip;
                let mut remaining_fetch = fetch;
                for batch in batches {
                    if tx.is_closed() || remaining_fetch == Some(0) {
                        break;
                    }
                    let bytes = get_record_batch_memory_size(&batch);
                    let _ = reservation.try_shrink(bytes);
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
                reservation.free();
                return Ok(());
            }
        }
    }

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

    let merge_fetch = fetch.and_then(|f| skip.checked_add(f));
    let merge_metrics = ExecutionPlanMetricsSet::new();
    let merge_reservation = MemoryConsumer::new(consumer_name).register(memory_pool);
    let mut merged = StreamingMergeBuilder::new()
        .with_streams(streams)
        .with_schema(schema)
        .with_expressions(ordering)
        .with_metrics(BaselineMetrics::new(&merge_metrics, 0))
        .with_batch_size(batch_size.max(1))
        .with_fetch(merge_fetch)
        .with_reservation(merge_reservation)
        .build()?;

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

    pub(super) fn new_state(&self, reservation: MemoryReservation) -> Result<TopNState> {
        let row_converter = RowConverter::new(self.sort_fields.clone())?;
        let first_col_fast = FirstColFastPath::try_new(&self.ordering, &self.schema);
        Ok(TopNState {
            ordering: self.ordering.clone(),
            row_converter,
            limit: self.heap_limit,
            heap: BinaryHeap::new(),
            batches: Vec::new(),
            retained_rows: 0,
            retained_batch_bytes: 0,
            key_bytes: 0,
            first_col_fast,
            reservation,
            #[cfg(test)]
            peak_retained_rows: Arc::clone(&self.peak_retained_rows),
        })
    }

    pub(super) async fn finalize_and_emit(
        &self,
        runs: Vec<SortedRun>,
        result_operators: &[Operator],
        batch_size: usize,
        memory_pool: &Arc<dyn MemoryPool>,
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

#[derive(Clone, Copy)]
enum PrimitiveThreshold {
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
}

struct FirstColFastPath {
    col_idx: usize,
    options: SortOptions,
    is_single_key: bool,
    threshold: Option<PrimitiveThreshold>,
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
fn passes_threshold<T: ArrowNativeTypeOp>(
    candidate: Option<T>,
    threshold: Option<T>,
    options: SortOptions,
    is_single_key: bool,
) -> bool {
    let ord = compare_opt_primitive(candidate, threshold, options);
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
            // Threshold is NULL and array has no NULLs:
            // - nulls_first => every non-null value is strictly greater => 0 candidates.
            // - nulls_last  => every non-null value is strictly less => all selected rows.
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
            if is_single_key {
                ord == CmpOrdering::Less
            } else {
                ord != CmpOrdering::Greater
            }
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
            passes_threshold(v, threshold, options, is_single_key)
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
                PrimitiveThreshold::Int8((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Int16 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int16Type>>()
                    .unwrap();
                PrimitiveThreshold::Int16((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Int32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int32Type>>()
                    .unwrap();
                PrimitiveThreshold::Int32((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Date32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Date32Type>>()
                    .unwrap();
                PrimitiveThreshold::Int32((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Int64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int64Type>>()
                    .unwrap();
                PrimitiveThreshold::Int64((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::Date64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Date64Type>>()
                    .unwrap();
                PrimitiveThreshold::Int64((!a.is_null(row_idx)).then(|| a.value(row_idx)))
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
                PrimitiveThreshold::Int64(val)
            }
            DataType::UInt8 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt8Type>>()
                    .unwrap();
                PrimitiveThreshold::UInt8((!a.is_null(row_idx)).then(|| a.value(row_idx)))
            }
            DataType::UInt16 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt16Type>>()
                    .unwrap();
                PrimitiveThreshold::UInt16(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx)),
                )
            }
            DataType::UInt32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt32Type>>()
                    .unwrap();
                PrimitiveThreshold::UInt32(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx)),
                )
            }
            DataType::UInt64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt64Type>>()
                    .unwrap();
                PrimitiveThreshold::UInt64(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx)),
                )
            }
            DataType::Float32 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Float32Type>>()
                    .unwrap();
                PrimitiveThreshold::Float32(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx)),
                )
            }
            DataType::Float64 => {
                let a = col
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Float64Type>>()
                    .unwrap();
                PrimitiveThreshold::Float64(
                    (!a.is_null(row_idx)).then(|| a.value(row_idx)),
                )
            }
            _ => return,
        };
        self.threshold = Some(threshold);
    }

    fn filter_candidates(&self, view: &BatchView) -> Option<Vec<u32>> {
        let threshold = self.threshold?;
        let batch = view.batch();
        if batch.num_rows() > u32::MAX as usize {
            return None;
        }
        let col = batch.column(self.col_idx);
        let sel = view.selection();
        let mut out = Vec::new();
        match (col.data_type(), threshold) {
            (DataType::Int8, PrimitiveThreshold::Int8(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int8Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Int16, PrimitiveThreshold::Int16(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int16Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Int32, PrimitiveThreshold::Int32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int32Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Date32, PrimitiveThreshold::Int32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Date32Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Int64, PrimitiveThreshold::Int64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Int64Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Date64, PrimitiveThreshold::Int64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Date64Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Timestamp(unit, _), PrimitiveThreshold::Int64(t)) => match unit {
                arrow::datatypes::TimeUnit::Second => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampSecondType>>()?,
                    sel,
                    t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
                arrow::datatypes::TimeUnit::Millisecond => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampMillisecondType>>()?,
                    sel,
                    t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
                arrow::datatypes::TimeUnit::Microsecond => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampMicrosecondType>>()?,
                    sel,
                    t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
                arrow::datatypes::TimeUnit::Nanosecond => scan_primitive_array(
                    col.as_any()
                        .downcast_ref::<PrimitiveArray<TimestampNanosecondType>>()?,
                    sel,
                    t,
                    self.options,
                    self.is_single_key,
                    &mut out,
                ),
            },
            (DataType::UInt8, PrimitiveThreshold::UInt8(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt8Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::UInt16, PrimitiveThreshold::UInt16(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt16Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::UInt32, PrimitiveThreshold::UInt32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::UInt64, PrimitiveThreshold::UInt64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Float32, PrimitiveThreshold::Float32(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Float32Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            (DataType::Float64, PrimitiveThreshold::Float64(t)) => scan_primitive_array(
                col.as_any().downcast_ref::<PrimitiveArray<Float64Type>>()?,
                sel,
                t,
                self.options,
                self.is_single_key,
                &mut out,
            ),
            _ => return None,
        }
        Some(out)
    }
}

pub(super) struct TopNState {
    ordering: LexOrdering,
    row_converter: RowConverter,
    limit: usize,
    heap: BinaryHeap<TopNHeapEntry>,
    batches: Vec<RecordBatch>,
    retained_rows: usize,
    retained_batch_bytes: usize,
    key_bytes: usize,
    first_col_fast: Option<FirstColFastPath>,
    reservation: MemoryReservation,
    #[cfg(test)]
    peak_retained_rows: Arc<AtomicUsize>,
}

impl TopNState {
    fn current_memory_bytes(&self) -> usize {
        self.retained_batch_bytes
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
            && let Some(batch) = self.batches.get(top.batch_idx as usize)
        {
            fast.update_from_batch(batch, top.row_idx as usize);
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
        let mut pending_wins = 0usize;
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
                pending_wins += 1;
                heap_changed = true;
            } else if let Some(worst) = self.heap.peek()
                && row_bytes < worst.key.as_ref()
            {
                let evicted = self.heap.pop().unwrap();
                self.key_bytes = self.key_bytes.saturating_sub(evicted.key.len());
                if evicted.batch_idx == pending_batch_idx {
                    pending_wins -= 1;
                }
                let key: Box<[u8]> = row_bytes.into();
                self.key_bytes = self.key_bytes.saturating_add(key.len());
                self.heap.push(TopNHeapEntry {
                    key,
                    batch_idx: pending_batch_idx,
                    row_idx: row_idx as u32,
                });
                pending_wins += 1;
                heap_changed = true;
            }
        }

        if pending_wins == 0 {
            if heap_changed {
                self.update_reservation()?;
                self.refresh_threshold();
            }
            return Ok(());
        }

        // Gather only the surviving rows from this batch, rewriting their
        // `row_idx` to the compacted batch coordinates.
        let mut entries = std::mem::take(&mut self.heap).into_vec();
        let mut take_indices = Vec::with_capacity(pending_wins);
        for entry in &mut entries {
            if entry.batch_idx == pending_batch_idx {
                let new_row_idx = take_indices.len() as u32;
                take_indices.push(entry.row_idx);
                entry.row_idx = new_row_idx;
            }
        }
        self.heap = BinaryHeap::from(entries);

        let compacted = take_record_batch(batch, &UInt32Array::from(take_indices))?;
        let compacted_rows = compacted.num_rows();
        let compacted_bytes = get_record_batch_memory_size(&compacted);
        self.batches.push(compacted);
        self.retained_rows = self.retained_rows.saturating_add(compacted_rows);
        self.retained_batch_bytes =
            self.retained_batch_bytes.saturating_add(compacted_bytes);
        #[cfg(test)]
        self.peak_retained_rows
            .fetch_max(self.retained_rows, AtomicOrdering::Relaxed);

        self.update_reservation()?;

        if self.retained_rows > self.limit.saturating_mul(2) && self.batches.len() > 1 {
            self.compact()?;
        }
        self.refresh_threshold();
        Ok(())
    }

    fn compact(&mut self) -> Result<()> {
        if self.heap.is_empty() {
            self.batches.clear();
            self.retained_rows = 0;
            self.retained_batch_bytes = 0;
            self.update_reservation()?;
            return Ok(());
        }
        let mut entries = std::mem::take(&mut self.heap).into_vec();
        let mut interleave_indices = Vec::with_capacity(entries.len());
        for (new_idx, entry) in entries.iter_mut().enumerate() {
            interleave_indices.push((entry.batch_idx as usize, entry.row_idx as usize));
            entry.batch_idx = 0;
            entry.row_idx = new_idx as u32;
        }
        let batch_refs: Vec<&RecordBatch> = self.batches.iter().collect();
        let compacted = interleave_record_batch(&batch_refs, &interleave_indices)?;
        let compacted_bytes = get_record_batch_memory_size(&compacted);
        let compacted_rows = compacted.num_rows();
        // Account for peak memory while both old and compacted batches are live.
        self.reservation.try_grow(compacted_bytes)?;

        self.batches.clear();
        self.batches.push(compacted);
        self.retained_rows = compacted_rows;
        self.retained_batch_bytes = compacted_bytes;
        self.heap = BinaryHeap::from(entries);
        self.update_reservation()?;
        Ok(())
    }

    pub(super) fn finish(self, batch_size: usize) -> Result<Option<SortedRun>> {
        if self.heap.is_empty() {
            self.reservation.free();
            return Ok(None);
        }
        // `BinaryHeap::into_sorted_vec` produces ascending order = exact SQL sort order.
        let sorted_entries = self.heap.into_sorted_vec();
        let indices: Vec<(usize, usize)> = sorted_entries
            .iter()
            .map(|e| (e.batch_idx as usize, e.row_idx as usize))
            .collect();
        drop(sorted_entries);

        let batch_refs: Vec<&RecordBatch> = self.batches.iter().collect();
        let chunk_size = batch_size.max(1);
        let mut out_batches = Vec::with_capacity(indices.len().div_ceil(chunk_size));
        let mut out_bytes = 0usize;
        for chunk in indices.chunks(chunk_size) {
            let out = interleave_record_batch(&batch_refs, chunk)?;
            let b = get_record_batch_memory_size(&out);
            self.reservation.try_grow(b)?;
            out_bytes = out_bytes.saturating_add(b);
            out_batches.push(out);
        }
        drop(self.batches);
        self.reservation.try_resize(out_bytes)?;
        Ok(Some(SortedRun::Batches {
            batches: out_batches,
            reservation: self.reservation,
        }))
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

    pub(super) fn new_state(&self, reservation: MemoryReservation) -> SortState {
        SortState {
            schema: Arc::clone(&self.schema),
            ordering: self.ordering.clone(),
            batches: Vec::new(),
            buffered_rows: 0,
            buffered_bytes: 0,
            completed_runs: Vec::new(),
            reservation,
        }
    }

    pub(super) async fn finalize_and_emit(
        &self,
        runs: Vec<SortedRun>,
        result_operators: &[Operator],
        batch_size: usize,
        memory_pool: &Arc<dyn MemoryPool>,
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
    buffered_bytes: usize,
    completed_runs: Vec<SortedRun>,
    reservation: MemoryReservation,
}

impl SortState {
    pub(super) fn add_view(&mut self, view: BatchView) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        let batch = view.materialize()?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if !self.batches.is_empty()
            && (self.buffered_bytes >= MAX_LOCAL_SORT_RUN_BYTES
                || self
                    .buffered_rows
                    .checked_add(batch.num_rows())
                    .is_none_or(|n| n > u32::MAX as usize))
        {
            self.flush_current_batches()?;
        }
        let bytes = get_record_batch_memory_size(&batch);
        self.reservation.try_grow(bytes)?;
        self.buffered_rows += batch.num_rows();
        self.buffered_bytes += bytes;
        self.batches.push(batch);
        Ok(())
    }

    fn flush_current_batches(&mut self) -> Result<()> {
        if self.batches.is_empty() {
            return Ok(());
        }
        let batches = std::mem::take(&mut self.batches);
        let chunk_bytes = std::mem::take(&mut self.buffered_bytes);
        self.buffered_rows = 0;
        let run_res = self.reservation.split(chunk_bytes);

        let merged = if batches.len() == 1 {
            batches.into_iter().next().unwrap()
        } else {
            run_res.try_grow(chunk_bytes)?;
            let merged = concat_batches(&self.schema, &batches)?;
            drop(batches);
            let merged_bytes = get_record_batch_memory_size(&merged);
            run_res.try_resize(merged_bytes)?;
            merged
        };

        let sort_columns = self
            .ordering
            .iter()
            .map(|e| e.evaluate_to_sort_column(&merged))
            .collect::<Result<Vec<_>>>()?;
        let indices_bytes = merged.num_rows().saturating_mul(size_of::<u32>());
        run_res.try_grow(indices_bytes)?;
        let indices = lexsort_to_indices(&sort_columns, None)?;
        drop(sort_columns);

        let final_bytes = get_record_batch_memory_size(&merged)
            .saturating_add(indices.get_array_memory_size());
        run_res.try_resize(final_bytes)?;

        self.completed_runs.push(SortedRun::Indexed {
            batch: merged,
            indices,
            reservation: run_res,
        });
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Vec<SortedRun>> {
        self.flush_current_batches()?;
        self.reservation.free();
        Ok(self.completed_runs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Float64Array, Int32Array, StringArray};
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

        // Feed 50,000 rows in strictly descending order across 50 batches so every
        // incoming batch beats every previously seen row (worst-case for TopN!).
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
        let SortedRun::Batches {
            batches,
            reservation,
        } = run
        else {
            unreachable!()
        };
        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, limit);
        let ks = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let expected: Vec<i32> = (0..limit as i32).collect();
        assert_eq!(ks.values().as_ref(), expected.as_slice());

        drop(batches);
        drop(reservation);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn topn_first_col_fast_path_handles_floats_and_nulls() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float64, true),
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
        let ids: Vec<i32> = (0..values.len() as i32).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float64Array::from(values)),
                Arc::new(Int32Array::from(ids)),
            ],
        )?;

        for descending in [false, true] {
            for nulls_first in [false, true] {
                let options = SortOptions {
                    descending,
                    nulls_first,
                };
                let ordering = LexOrdering::from([
                    PhysicalSortExpr::new(Arc::new(Column::new("f", 0)), options),
                    PhysicalSortExpr::new(
                        Arc::new(Column::new("id", 1)),
                        SortOptions::default(),
                    ),
                ]);
                let sink =
                    TopNSink::try_new(Arc::clone(&schema), ordering.clone(), 0, 4)?
                        .unwrap();
                let pool = test_pool(None);
                let mut state =
                    sink.new_state(MemoryConsumer::new("test_float").register(&pool))?;
                // Feed row-by-row in slices so the heap fills and exercises threshold rejection.
                for i in 0..batch.num_rows() {
                    state.add_view(&BatchView::all(batch.slice(i, 1)))?;
                }
                let run = state.finish(16)?.unwrap();
                let SortedRun::Batches { batches, .. } = run else {
                    unreachable!()
                };
                let got_ids = batches[0]
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();

                let expected = datafusion_physical_plan::sorts::sort::sort_batch(
                    &batch,
                    &ordering,
                    Some(4),
                )?;
                let exp_ids = expected
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                assert_eq!(
                    got_ids.values().as_ref(),
                    exp_ids.values().as_ref(),
                    "mismatch for descending={descending}, nulls_first={nulls_first}"
                );
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
        let mut state = sink.new_state(MemoryConsumer::new("test_sort").register(&pool));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(0..1024))],
        )?;
        let err = state.add_view(BatchView::all(batch)).unwrap_err();
        assert!(
            matches!(
                err,
                datafusion_common::DataFusionError::ResourcesExhausted(_)
            ),
            "{err}"
        );
        drop(state);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }
}
