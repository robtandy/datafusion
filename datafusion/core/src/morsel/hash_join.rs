// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    RecordBatchOptions, StringArray, StringViewArray, UInt32Array, UInt64Array,
};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
use arrow::compute::{concat, concat_batches, filter, filter_record_batch, take};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::float_semantics;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_common::{Result, exec_err, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use tokio::sync::mpsc;

use super::Pipeline;
use super::batch::{
    BatchView, Selection, collect_expr_column_indices, remap_expr_columns,
};

const FINALIZE_YIELD_CHUNK: usize = 4096;

// Fixed hash seed shared between build and probe workers within a run.
const JOIN_HASH_SEED: u64 = 0x517c_c1b7_2722_0a95;

fn join_random_state() -> RandomState {
    RandomState::with_seed(JOIN_HASH_SEED)
}

#[inline]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub(super) fn is_supported_join_key_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Boolean
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
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BuildSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum JoinOutputCol {
    Build(usize),
    Probe(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KeyKind {
    Int32,
    Int64,
    UInt32,
    UInt64,
    Utf8,
    LargeUtf8,
    Utf8View,
    Rows,
}

impl KeyKind {
    pub(super) fn from_types(types: &[DataType]) -> Self {
        match types {
            [DataType::Int32] => Self::Int32,
            [DataType::Int64] => Self::Int64,
            [DataType::UInt32] => Self::UInt32,
            [DataType::UInt64] => Self::UInt64,
            [DataType::Utf8] => Self::Utf8,
            [DataType::LargeUtf8] => Self::LargeUtf8,
            [DataType::Utf8View] => Self::Utf8View,
            _ => Self::Rows,
        }
    }
}

fn has_float_type(types: &[DataType]) -> bool {
    types.iter().any(|dt| {
        matches!(
            dt,
            DataType::Float16 | DataType::Float32 | DataType::Float64
        )
    })
}

fn normalize_key_arrays(keys: Vec<ArrayRef>, has_floats: bool) -> Vec<ArrayRef> {
    if !has_floats {
        return keys;
    }
    keys.into_iter().map(float_semantics::normalize).collect()
}

fn valid_key_mask(keys: &[ArrayRef]) -> Option<BooleanBuffer> {
    let mut combined: Option<BooleanBuffer> = None;
    for key in keys {
        if key.null_count() == 0 {
            continue;
        }
        let Some(nulls) = key.nulls() else {
            continue;
        };
        let buf = nulls.inner();
        combined = Some(match combined {
            Some(prev) => &prev & buf,
            None => buf.clone(),
        });
    }
    combined
}

fn compute_build_hashes(
    key_kind: KeyKind,
    keys: &[ArrayRef],
    num_rows: usize,
) -> Result<Vec<u64>> {
    let mut hashes = vec![0u64; num_rows];
    match key_kind {
        KeyKind::Int32 => {
            let arr = keys[0]
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    datafusion_common::internal_datafusion_err!(
                        "Expected Int32 join key array"
                    )
                })?;
            for (h, &v) in hashes.iter_mut().zip(arr.values().iter()) {
                *h = mix_u64(v as u32 as u64);
            }
        }
        KeyKind::Int64 => {
            let arr = keys[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    datafusion_common::internal_datafusion_err!(
                        "Expected Int64 join key array"
                    )
                })?;
            for (h, &v) in hashes.iter_mut().zip(arr.values().iter()) {
                *h = mix_u64(v as u64);
            }
        }
        KeyKind::UInt32 => {
            let arr =
                keys[0]
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected UInt32 join key array"
                        )
                    })?;
            for (h, &v) in hashes.iter_mut().zip(arr.values().iter()) {
                *h = mix_u64(v as u64);
            }
        }
        KeyKind::UInt64 => {
            let arr =
                keys[0]
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected UInt64 join key array"
                        )
                    })?;
            for (h, &v) in hashes.iter_mut().zip(arr.values().iter()) {
                *h = mix_u64(v);
            }
        }
        KeyKind::Utf8 | KeyKind::LargeUtf8 | KeyKind::Utf8View | KeyKind::Rows => {
            let state = join_random_state();
            create_hashes(keys, &state, &mut hashes)?;
        }
    }
    Ok(hashes)
}

fn chunk_memory_bytes(
    payload: Option<&RecordBatch>,
    keys: &[ArrayRef],
    hashes_capacity: usize,
) -> usize {
    let payload_bytes = payload.map_or(0, get_record_batch_memory_size);
    let keys_bytes = keys
        .iter()
        .map(|k| {
            // Avoid double-counting when a key array shares the exact Arc with a payload column.
            let shared_with_payload = payload
                .is_some_and(|p| p.columns().iter().any(|col| Arc::ptr_eq(col, k)));
            if shared_with_payload {
                0
            } else {
                k.get_array_memory_size()
            }
        })
        .fold(0usize, usize::saturating_add);
    let hashes_bytes = hashes_capacity.saturating_mul(size_of::<u64>());
    payload_bytes
        .saturating_add(keys_bytes)
        .saturating_add(hashes_bytes)
}

#[derive(Debug)]
pub(super) struct HashJoinBuildSink {
    key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    key_names: Vec<String>,
    key_types: Vec<DataType>,
    key_kind: KeyKind,
    has_floats: bool,
    payload_columns: Vec<usize>,
    payload_schema: SchemaRef,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
}

impl HashJoinBuildSink {
    pub(super) fn try_new(
        key_exprs: Vec<Arc<dyn PhysicalExpr>>,
        key_names: Vec<String>,
        key_types: Vec<DataType>,
        payload_columns: Vec<usize>,
        payload_schema: SchemaRef,
    ) -> Result<std::result::Result<Self, String>> {
        let key_kind = KeyKind::from_types(&key_types);
        if key_kind == KeyKind::Rows {
            let fields = key_types
                .iter()
                .map(|dt| SortField::new(dt.clone()))
                .collect();
            if let Err(err) = RowConverter::new(fields) {
                return Ok(Err(format!("unsupported join key type: {err}")));
            }
        }
        let has_floats = has_float_type(&key_types);
        Ok(Ok(Self {
            key_exprs,
            key_names,
            key_types,
            key_kind,
            has_floats,
            payload_columns,
            payload_schema,
            #[cfg(test)]
            finalize_gate: None,
        }))
    }

    pub(super) fn new_state(&self, reservation: MemoryReservation) -> HashJoinBuildState {
        HashJoinBuildState {
            key_exprs: self.key_exprs.clone(),
            key_kind: self.key_kind,
            has_floats: self.has_floats,
            payload_columns: self.payload_columns.clone(),
            batches: Vec::new(),
            key_batches: Vec::new(),
            hash_chunks: Vec::new(),
            num_rows: 0,
            reservation,
        }
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(self.key_exprs.iter(), f)
    }

    pub(super) fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashJoinBuildSink[keys=[{}]]", self.key_names.join(", "))
    }

    pub(super) async fn finalize(
        &self,
        combined: HashJoinBuildState,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<Arc<FinalizedHashTable>>> {
        let HashJoinBuildState {
            mut batches,
            mut key_batches,
            hash_chunks,
            num_rows,
            reservation,
            ..
        } = combined;

        if num_rows == 0 {
            reservation.resize(0);
            return Ok(Some(Arc::new(FinalizedHashTable::empty(
                Arc::clone(&self.payload_schema),
                reservation,
            ))));
        }

        if num_rows >= u32::MAX as usize {
            return exec_err!(
                "Morsel HashJoin build side exceeds u32::MAX rows ({num_rows})"
            );
        }

        if tx.is_closed() {
            return Ok(None);
        }

        let bucket_count = num_rows.saturating_mul(2).next_power_of_two().max(16);
        let payload_concat_bytes =
            if self.payload_columns.is_empty() || batches.len() <= 1 {
                0
            } else {
                batches
                    .iter()
                    .map(get_record_batch_memory_size)
                    .fold(0usize, usize::saturating_add)
            };
        let is_flat_int = matches!(
            self.key_kind,
            KeyKind::Int32 | KeyKind::Int64 | KeyKind::UInt32 | KeyKind::UInt64
        );
        let (dir_bytes, keys_peak_bytes) = match self.key_kind {
            KeyKind::Int32 | KeyKind::UInt32 => (
                0usize,
                bucket_count
                    .saturating_mul(size_of::<IntSlot<u32>>())
                    .saturating_add(num_rows.saturating_mul(size_of::<u32>())),
            ),
            KeyKind::Int64 | KeyKind::UInt64 => (
                0usize,
                bucket_count
                    .saturating_mul(size_of::<IntSlot<u64>>())
                    .saturating_add(num_rows.saturating_mul(size_of::<u32>())),
            ),
            KeyKind::Utf8 | KeyKind::LargeUtf8 | KeyKind::Utf8View => {
                let dir = bucket_count
                    .saturating_add(num_rows)
                    .saturating_mul(size_of::<u32>());
                let concat_bytes = if key_batches.len() <= 1 {
                    0
                } else {
                    key_batches
                        .iter()
                        .map(|kb| kb[0].get_array_memory_size())
                        .fold(0usize, usize::saturating_add)
                };
                (
                    dir,
                    concat_bytes
                        .saturating_add(num_rows.saturating_mul(size_of::<u64>())),
                )
            }
            KeyKind::Rows => {
                let dir = bucket_count
                    .saturating_add(num_rows)
                    .saturating_mul(size_of::<u32>());
                let raw_key_bytes = key_batches
                    .iter()
                    .flat_map(|kb| kb.iter())
                    .map(|a| a.get_array_memory_size())
                    .fold(0usize, usize::saturating_add);
                (
                    dir,
                    raw_key_bytes
                        .saturating_add(num_rows.saturating_mul(16))
                        .saturating_add(num_rows.saturating_mul(size_of::<u64>())),
                )
            }
        };

        let peak_extra = dir_bytes
            .saturating_add(payload_concat_bytes)
            .saturating_add(keys_peak_bytes);
        reservation.try_grow(peak_extra)?;

        if tx.is_closed() {
            return Ok(None);
        }

        let payload = if self.payload_columns.is_empty() {
            RecordBatch::try_new_with_options(
                Arc::clone(&self.payload_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(num_rows)),
            )?
        } else if batches.len() == 1 {
            batches.pop().unwrap()
        } else {
            concat_batches(&self.payload_schema, &batches)?
        };
        drop(batches);

        let mask = bucket_count - 1;
        let (build_keys, buckets, next) = if is_flat_int {
            let Some(keys) = self
                .build_flat_int_keys(
                    &mut key_batches,
                    &hash_chunks,
                    num_rows,
                    bucket_count,
                    tx,
                )
                .await?
            else {
                return Ok(None);
            };
            drop(key_batches);
            drop(hash_chunks);
            (keys, Vec::new(), Vec::new())
        } else {
            let build_keys =
                self.build_chained_keys(&mut key_batches, &hash_chunks, num_rows)?;
            drop(key_batches);

            let mut buckets = vec![0u32; bucket_count];
            let mut next = vec![0u32; num_rows];
            let mut row_idx = 0usize;
            let mut since_yield = 0usize;
            for chunk in &hash_chunks {
                for &h in chunk {
                    let bucket = (h as usize) & mask;
                    let entry = (row_idx + 1) as u32;
                    next[row_idx] = buckets[bucket];
                    buckets[bucket] = entry;
                    row_idx += 1;
                    since_yield += 1;
                    if since_yield >= FINALIZE_YIELD_CHUNK {
                        since_yield = 0;
                        if tx.is_closed() {
                            return Ok(None);
                        }
                        tokio::task::yield_now().await;
                    }
                }
            }
            drop(hash_chunks);
            (build_keys, buckets, next)
        };

        let final_bytes = get_record_batch_memory_size(&payload)
            .saturating_add(build_keys.memory_bytes())
            .saturating_add(dir_bytes);
        reservation.try_resize(final_bytes)?;

        Ok(Some(Arc::new(FinalizedHashTable {
            num_rows,
            payload,
            keys: build_keys,
            buckets,
            mask,
            next,
            _reservation: reservation,
        })))
    }

    async fn build_flat_int_keys(
        &self,
        key_batches: &mut Vec<Vec<ArrayRef>>,
        hash_chunks: &[Vec<u64>],
        num_rows: usize,
        slot_count: usize,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<BuildKeys>> {
        macro_rules! build_flat {
            ($Arr:ty, $K:ty, $Variant:ident) => {{
                let mask = slot_count - 1;
                let mut slots = vec![IntSlot::<$K>::empty(); slot_count];
                let mut dup_next: Vec<u32> = Vec::new();
                let mut row_idx = 0usize;
                let mut since_yield = 0usize;
                for (kb, hashes) in key_batches.drain(..).zip(hash_chunks.iter()) {
                    let arr = kb[0].as_any().downcast_ref::<$Arr>().ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected integer join key array"
                        )
                    })?;
                    for (&key, &h) in arr.values().iter().zip(hashes.iter()) {
                        let mut pos = (h as usize) & mask;
                        loop {
                            let slot = &mut slots[pos];
                            if slot.first_row == 0 {
                                slot.key = key;
                                slot.first_row = (row_idx + 1) as u32;
                                break;
                            }
                            if slot.key == key {
                                if dup_next.is_empty() {
                                    dup_next.resize(num_rows, 0);
                                }
                                dup_next[row_idx] = slot.dup_head;
                                slot.dup_head = (row_idx + 1) as u32;
                                break;
                            }
                            pos = (pos + 1) & mask;
                        }
                        row_idx += 1;
                        since_yield += 1;
                        if since_yield >= FINALIZE_YIELD_CHUNK {
                            since_yield = 0;
                            if tx.is_closed() {
                                return Ok(None);
                            }
                            tokio::task::yield_now().await;
                        }
                    }
                }
                Ok(Some(BuildKeys::$Variant(FlatIntTable {
                    slots,
                    mask,
                    dup_next,
                })))
            }};
        }

        match self.key_kind {
            KeyKind::Int32 => build_flat!(Int32Array, i32, Int32),
            KeyKind::Int64 => build_flat!(Int64Array, i64, Int64),
            KeyKind::UInt32 => build_flat!(UInt32Array, u32, UInt32),
            KeyKind::UInt64 => build_flat!(UInt64Array, u64, UInt64),
            _ => unreachable!(),
        }
    }

    fn build_chained_keys(
        &self,
        key_batches: &mut Vec<Vec<ArrayRef>>,
        hash_chunks: &[Vec<u64>],
        num_rows: usize,
    ) -> Result<BuildKeys> {
        let flatten_hashes = || {
            let mut hashes = Vec::with_capacity(num_rows);
            for chunk in hash_chunks {
                hashes.extend_from_slice(chunk);
            }
            hashes
        };

        match self.key_kind {
            KeyKind::Int32 | KeyKind::Int64 | KeyKind::UInt32 | KeyKind::UInt64 => {
                unreachable!()
            }
            KeyKind::Utf8 => {
                let arr_ref = concat_single_key(key_batches)?;
                let array = arr_ref
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8 join key array"
                        )
                    })?
                    .clone();
                Ok(BuildKeys::Utf8 {
                    array,
                    hashes: flatten_hashes(),
                })
            }
            KeyKind::LargeUtf8 => {
                let arr_ref = concat_single_key(key_batches)?;
                let array = arr_ref
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected LargeUtf8 join key array"
                        )
                    })?
                    .clone();
                Ok(BuildKeys::LargeUtf8 {
                    array,
                    hashes: flatten_hashes(),
                })
            }
            KeyKind::Utf8View => {
                let arr_ref = concat_single_key(key_batches)?;
                let array = arr_ref
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8View join key array"
                        )
                    })?
                    .clone();
                Ok(BuildKeys::Utf8View {
                    array,
                    hashes: flatten_hashes(),
                })
            }
            KeyKind::Rows => {
                let fields = self
                    .key_types
                    .iter()
                    .map(|dt| SortField::new(dt.clone()))
                    .collect();
                let converter = RowConverter::new(fields)?;
                let num_keys = self.key_types.len();
                let mut concat_cols = Vec::with_capacity(num_keys);
                for col_idx in 0..num_keys {
                    let col = if key_batches.len() == 1 {
                        Arc::clone(&key_batches[0][col_idx])
                    } else {
                        let refs: Vec<&dyn Array> =
                            key_batches.iter().map(|kb| kb[col_idx].as_ref()).collect();
                        concat(&refs)?
                    };
                    concat_cols.push(col);
                }
                let rows = converter.convert_columns(&concat_cols)?;
                Ok(BuildKeys::Rows {
                    converter,
                    rows,
                    hashes: flatten_hashes(),
                })
            }
        }
    }
}

fn concat_single_key(key_batches: &mut Vec<Vec<ArrayRef>>) -> Result<ArrayRef> {
    if key_batches.len() == 1 {
        Ok(Arc::clone(&key_batches[0][0]))
    } else {
        let refs: Vec<&dyn Array> = key_batches.iter().map(|kb| kb[0].as_ref()).collect();
        Ok(concat(&refs)?)
    }
}

pub(super) struct HashJoinBuildState {
    key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    key_kind: KeyKind,
    has_floats: bool,
    payload_columns: Vec<usize>,
    batches: Vec<RecordBatch>,
    key_batches: Vec<Vec<ArrayRef>>,
    hash_chunks: Vec<Vec<u64>>,
    num_rows: usize,
    reservation: MemoryReservation,
}

impl HashJoinBuildState {
    pub(super) fn add_view(&mut self, view: &BatchView) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        if view.is_all() {
            return self.add_batch(view.batch());
        }
        let batch = view.batch();
        let selected = view.num_rows();
        let key_needed_cols = collect_expr_column_indices(&self.key_exprs);
        let compact_key_input = view.materialize_columns(&key_needed_cols)?;
        let mut keys = Vec::with_capacity(self.key_exprs.len());
        for expr in &self.key_exprs {
            let rebound = remap_expr_columns(Arc::clone(expr), &key_needed_cols)?;
            keys.push(rebound.evaluate(&compact_key_input)?.into_array(selected)?);
        }

        let valid_mask = valid_key_mask(&keys);
        let (filtered_keys, payload_selection, valid_rows) = match valid_mask {
            Some(bits) => {
                let valid_rows = bits.count_set_bits();
                if valid_rows == 0 {
                    return Ok(());
                }
                let predicate = BooleanArray::new(bits, None);
                let filtered_keys = keys
                    .iter()
                    .map(|k| Ok(filter(k.as_ref(), &predicate)?))
                    .collect::<Result<Vec<ArrayRef>>>()?;
                let combined_sel = view
                    .selection()
                    .compose_compact_mask(batch.num_rows(), &predicate)?;
                (filtered_keys, Some(combined_sel), valid_rows)
            }
            None => (keys, None, selected),
        };

        let payload_batch = if self.payload_columns.is_empty() {
            None
        } else {
            match payload_selection {
                Some(sel) => {
                    let narrowed_view =
                        BatchView::try_new(batch.project(&self.payload_columns)?, sel)?;
                    Some(narrowed_view.materialize()?)
                }
                None => Some(view.materialize_columns(&self.payload_columns)?),
            }
        };

        let normalized_keys = normalize_key_arrays(filtered_keys, self.has_floats);
        let hashes = compute_build_hashes(self.key_kind, &normalized_keys, valid_rows)?;

        let delta = chunk_memory_bytes(
            payload_batch.as_ref(),
            &normalized_keys,
            hashes.capacity(),
        );
        self.reservation.try_grow(delta)?;

        if let Some(payload) = payload_batch {
            self.batches.push(payload);
        }
        self.key_batches.push(normalized_keys);
        self.hash_chunks.push(hashes);
        self.num_rows = self.num_rows.checked_add(valid_rows).ok_or_else(|| {
            datafusion_common::exec_datafusion_err!(
                "Morsel HashJoin build row count overflow"
            )
        })?;
        Ok(())
    }

    pub(super) fn add_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let mut keys = Vec::with_capacity(self.key_exprs.len());
        for expr in &self.key_exprs {
            keys.push(expr.evaluate(batch)?.into_array(batch.num_rows())?);
        }

        let valid_mask = valid_key_mask(&keys);
        let (filtered_keys, payload_batch, valid_rows) = match valid_mask {
            Some(bits) => {
                let valid_rows = bits.count_set_bits();
                if valid_rows == 0 {
                    return Ok(());
                }
                let predicate = BooleanArray::new(bits, None);
                let filtered_keys = keys
                    .iter()
                    .map(|k| Ok(filter(k.as_ref(), &predicate)?))
                    .collect::<Result<Vec<ArrayRef>>>()?;
                let payload_batch = if self.payload_columns.is_empty() {
                    None
                } else {
                    let projected = batch.project(&self.payload_columns)?;
                    Some(filter_record_batch(&projected, &predicate)?)
                };
                (filtered_keys, payload_batch, valid_rows)
            }
            None => {
                let payload_batch = if self.payload_columns.is_empty() {
                    None
                } else {
                    Some(batch.project(&self.payload_columns)?)
                };
                (keys, payload_batch, batch.num_rows())
            }
        };

        let normalized_keys = normalize_key_arrays(filtered_keys, self.has_floats);
        let hashes = compute_build_hashes(self.key_kind, &normalized_keys, valid_rows)?;

        let delta = chunk_memory_bytes(
            payload_batch.as_ref(),
            &normalized_keys,
            hashes.capacity(),
        );
        self.reservation.try_grow(delta)?;

        if let Some(payload) = payload_batch {
            self.batches.push(payload);
        }
        self.key_batches.push(normalized_keys);
        self.hash_chunks.push(hashes);
        self.num_rows = self.num_rows.checked_add(valid_rows).ok_or_else(|| {
            datafusion_common::exec_datafusion_err!(
                "Morsel HashJoin build row count overflow"
            )
        })?;
        Ok(())
    }

    pub(super) fn combine(&mut self, mut other: Self) {
        if other.num_rows == 0 {
            return;
        }
        if self.num_rows == 0 {
            std::mem::swap(self, &mut other);
            return;
        }
        self.batches.append(&mut other.batches);
        self.key_batches.append(&mut other.key_batches);
        self.hash_chunks.append(&mut other.hash_chunks);
        self.num_rows += other.num_rows;
        let transferred = other.reservation.free();
        self.reservation.grow(transferred);
    }
}

#[derive(Clone, Copy)]
struct IntSlot<K: Default + Copy> {
    key: K,
    first_row: u32,
    dup_head: u32,
}

impl<K: Default + Copy> IntSlot<K> {
    fn empty() -> Self {
        Self {
            key: K::default(),
            first_row: 0,
            dup_head: 0,
        }
    }
}

struct FlatIntTable<K: Default + Copy> {
    slots: Vec<IntSlot<K>>,
    mask: usize,
    dup_next: Vec<u32>,
}

impl<K: Default + Copy> FlatIntTable<K> {
    fn memory_bytes(&self) -> usize {
        self.slots
            .capacity()
            .saturating_mul(size_of::<IntSlot<K>>())
            .saturating_add(self.dup_next.capacity().saturating_mul(size_of::<u32>()))
    }
}

enum BuildKeys {
    Empty,
    Int32(FlatIntTable<i32>),
    Int64(FlatIntTable<i64>),
    UInt32(FlatIntTable<u32>),
    UInt64(FlatIntTable<u64>),
    Utf8 {
        array: StringArray,
        hashes: Vec<u64>,
    },
    LargeUtf8 {
        array: LargeStringArray,
        hashes: Vec<u64>,
    },
    Utf8View {
        array: StringViewArray,
        hashes: Vec<u64>,
    },
    Rows {
        converter: RowConverter,
        rows: Rows,
        hashes: Vec<u64>,
    },
}

impl BuildKeys {
    fn memory_bytes(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Int32(t) => t.memory_bytes(),
            Self::Int64(t) => t.memory_bytes(),
            Self::UInt32(t) => t.memory_bytes(),
            Self::UInt64(t) => t.memory_bytes(),
            Self::Utf8 { array, hashes } => array
                .get_array_memory_size()
                .saturating_add(hashes.capacity().saturating_mul(size_of::<u64>())),
            Self::LargeUtf8 { array, hashes } => array
                .get_array_memory_size()
                .saturating_add(hashes.capacity().saturating_mul(size_of::<u64>())),
            Self::Utf8View { array, hashes } => array
                .get_array_memory_size()
                .saturating_add(hashes.capacity().saturating_mul(size_of::<u64>())),
            Self::Rows { rows, hashes, .. } => rows
                .size()
                .saturating_add(hashes.capacity().saturating_mul(size_of::<u64>())),
        }
    }
}

pub(super) struct FinalizedHashTable {
    num_rows: usize,
    payload: RecordBatch,
    keys: BuildKeys,
    buckets: Vec<u32>,
    mask: usize,
    next: Vec<u32>,
    _reservation: MemoryReservation,
}

impl FinalizedHashTable {
    fn empty(payload_schema: SchemaRef, reservation: MemoryReservation) -> Self {
        Self {
            num_rows: 0,
            payload: RecordBatch::new_empty(payload_schema),
            keys: BuildKeys::Empty,
            buckets: Vec::new(),
            mask: 0,
            next: Vec::new(),
            _reservation: reservation,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.num_rows == 0
    }
}

pub(super) enum OperatorStep {
    NeedMoreInput(Option<RecordBatch>),
    HaveMoreOutput(RecordBatch),
}

#[derive(Default)]
pub(super) struct ProbeScratch {
    build_indices: Vec<u32>,
    probe_indices: Vec<u32>,
}

enum ProbeKeys {
    Int32(Int32Array),
    Int64(Int64Array),
    UInt32(UInt32Array),
    UInt64(UInt64Array),
    Utf8 {
        array: StringArray,
        hashes: Vec<u64>,
    },
    LargeUtf8 {
        array: LargeStringArray,
        hashes: Vec<u64>,
    },
    Utf8View {
        array: StringViewArray,
        hashes: Vec<u64>,
    },
    Rows {
        rows: Rows,
        hashes: Vec<u64>,
    },
}

enum ProbeRows {
    /// All rows `0..num_rows` in `cursor.batch` (key index == backing row).
    All { num_rows: usize },
    /// Backing-coordinate bitmask (key array is uncompacted; key index == backing row).
    Mask {
        bits: BooleanBuffer,
        num_rows: usize,
    },
    /// Backing-coordinate indices (key array is uncompacted; backing row = `indices[pos]`).
    BackingIndices(UInt32Array),
    /// Compacted key arrays of length `backing_rows.len()`; payload is gathered from `backing_rows[pos]`.
    CompactedKeys { backing_rows: UInt32Array },
}

pub(super) struct ProbeCursor {
    batch: RecordBatch,
    keys: ProbeKeys,
    valid_mask: Option<BooleanBuffer>,
    rows: ProbeRows,
    pos: usize,
    next_entry: u32,
    _reservation: MemoryReservation,
}

impl ProbeCursor {
    fn is_exhausted(&self) -> bool {
        match &self.rows {
            ProbeRows::All { num_rows } | ProbeRows::Mask { num_rows, .. } => {
                self.pos >= *num_rows
            }
            ProbeRows::BackingIndices(indices) => self.pos >= indices.len(),
            ProbeRows::CompactedKeys { backing_rows } => self.pos >= backing_rows.len(),
        }
    }
}

#[derive(Debug)]
pub(super) struct HashJoinProbe {
    pub(super) build_pipeline: Arc<Pipeline>,
    pub(super) build_side: BuildSide,
    probe_key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    probe_key_names: Vec<String>,
    plain_single_key_col: Option<usize>,
    key_kind: KeyKind,
    has_floats: bool,
    pub(super) output_projection: Vec<JoinOutputCol>,
    pub(super) output_schema: SchemaRef,
    pub(super) build_id: usize,
}

#[derive(Debug)]
pub(super) struct HashJoinProbeConfig {
    pub(super) build_pipeline: Arc<Pipeline>,
    pub(super) build_side: BuildSide,
    pub(super) probe_key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(super) probe_key_names: Vec<String>,
    pub(super) key_types: Vec<DataType>,
    pub(super) output_projection: Vec<JoinOutputCol>,
    pub(super) output_schema: SchemaRef,
    pub(super) build_id: usize,
}

impl HashJoinProbe {
    pub(super) fn new(config: HashJoinProbeConfig) -> Self {
        let key_kind = KeyKind::from_types(&config.key_types);
        let has_floats = has_float_type(&config.key_types);
        let plain_single_key_col = match config.probe_key_exprs.as_slice() {
            [expr] => expr.downcast_ref::<Column>().map(Column::index),
            _ => None,
        };
        Self {
            build_pipeline: config.build_pipeline,
            build_side: config.build_side,
            probe_key_exprs: config.probe_key_exprs,
            probe_key_names: config.probe_key_names,
            plain_single_key_col,
            key_kind,
            has_floats,
            output_projection: config.output_projection,
            output_schema: config.output_schema,
            build_id: config.build_id,
        }
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        if apply_expression_roots(self.probe_key_exprs.iter(), f)?
            == TreeNodeRecursion::Stop
        {
            return Ok(TreeNodeRecursion::Stop);
        }
        self.build_pipeline.apply_expressions(f)
    }

    pub(super) fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HashJoinProbe[keys=[{}], build={:?}]",
            self.probe_key_names.join(", "),
            self.build_side
        )
    }

    pub(super) fn start_probe_view(
        &self,
        view: BatchView,
        table: &FinalizedHashTable,
        probe_reservation: &MemoryReservation,
    ) -> Result<Option<ProbeCursor>> {
        if view.is_empty() || table.is_empty() {
            return Ok(None);
        }

        // Apply the same retention accounting rule to unfiltered and filtered cursors:
        // charge only pipeline-allocated memory (the selection buffer or any newly
        // compacted copy), never preexisting backing storage.
        let mut cursor_reservation = probe_reservation.new_empty();
        let view = view.into_reserved_for_retention(&mut cursor_reservation)?;
        if view.is_all() {
            return self.start_probe_all(view.into_parts().0, table, cursor_reservation);
        }

        // Zero-copy fast path for single plain integer column keys:
        // read directly from backing array at selected row positions without copying keys.
        let is_flat_int = matches!(
            self.key_kind,
            KeyKind::Int32 | KeyKind::Int64 | KeyKind::UInt32 | KeyKind::UInt64
        );
        if is_flat_int && let Some(key_col) = self.plain_single_key_col {
            let (batch, selection) = view.into_parts();
            let num_rows = batch.num_rows();
            let key_arr = batch.column(key_col);
            let (rows, valid_mask) = match selection {
                Selection::All => unreachable!(),
                Selection::Mask { bits, .. } => {
                    let effective_bits = if key_arr.null_count() > 0 {
                        let nulls = key_arr.nulls().unwrap().inner();
                        let combined = &bits & nulls;
                        if combined.count_set_bits() == 0 {
                            return Ok(None);
                        }
                        combined
                    } else {
                        bits
                    };
                    (
                        ProbeRows::Mask {
                            bits: effective_bits,
                            num_rows,
                        },
                        None,
                    )
                }
                Selection::Indices(indices) => {
                    let valid_mask = if key_arr.null_count() > 0 {
                        Some(key_arr.nulls().unwrap().inner().clone())
                    } else {
                        None
                    };
                    (ProbeRows::BackingIndices(indices), valid_mask)
                }
            };
            let keys =
                self.build_probe_keys(std::slice::from_ref(key_arr), num_rows, table)?;
            return Ok(Some(ProbeCursor {
                batch,
                keys,
                valid_mask,
                rows,
                pos: 0,
                next_entry: 0,
                _reservation: cursor_reservation,
            }));
        }

        // General selection-aware probe path (strings, composite keys, or expression keys):
        // evaluate key expressions and hash ONLY selected rows, while keeping the
        // backing batch uncopied so payload columns are gathered only for matches.
        let Some(backing_rows) = view.selection().to_indices(view.batch().num_rows())
        else {
            // Fallback for batches exceeding u32::MAX rows.
            return self.start_probe_all(view.materialize()?, table, cursor_reservation);
        };
        let selected = backing_rows.len();
        let key_needed_cols = collect_expr_column_indices(&self.probe_key_exprs);
        let compact_key_input = view.materialize_columns(&key_needed_cols)?;
        let mut key_arrays = Vec::with_capacity(self.probe_key_exprs.len());
        for expr in &self.probe_key_exprs {
            let rebound = remap_expr_columns(Arc::clone(expr), &key_needed_cols)?;
            key_arrays.push(rebound.evaluate(&compact_key_input)?.into_array(selected)?);
        }
        let valid_mask = valid_key_mask(&key_arrays);
        if let Some(bits) = &valid_mask
            && bits.count_set_bits() == 0
        {
            return Ok(None);
        }
        let keys = self.build_probe_keys(&key_arrays, selected, table)?;
        let (batch, _) = view.into_parts();
        Ok(Some(ProbeCursor {
            batch,
            keys,
            valid_mask,
            rows: ProbeRows::CompactedKeys { backing_rows },
            pos: 0,
            next_entry: 0,
            _reservation: cursor_reservation,
        }))
    }

    fn start_probe_all(
        &self,
        batch: RecordBatch,
        table: &FinalizedHashTable,
        reservation: MemoryReservation,
    ) -> Result<Option<ProbeCursor>> {
        let num_rows = batch.num_rows();
        if num_rows == 0 || table.is_empty() {
            return Ok(None);
        }
        let mut key_arrays = Vec::with_capacity(self.probe_key_exprs.len());
        for expr in &self.probe_key_exprs {
            key_arrays.push(expr.evaluate(&batch)?.into_array(num_rows)?);
        }
        let valid_mask = valid_key_mask(&key_arrays);
        if let Some(bits) = &valid_mask
            && bits.count_set_bits() == 0
        {
            return Ok(None);
        }
        let keys = self.build_probe_keys(&key_arrays, num_rows, table)?;

        Ok(Some(ProbeCursor {
            batch,
            keys,
            valid_mask,
            rows: ProbeRows::All { num_rows },
            pos: 0,
            next_entry: 0,
            _reservation: reservation,
        }))
    }

    fn build_probe_keys(
        &self,
        key_arrays: &[ArrayRef],
        num_rows: usize,
        table: &FinalizedHashTable,
    ) -> Result<ProbeKeys> {
        match (&self.key_kind, &table.keys) {
            (KeyKind::Int32, BuildKeys::Int32(_)) => {
                let arr = key_arrays[0]
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int32 probe key array"
                        )
                    })?
                    .clone();
                Ok(ProbeKeys::Int32(arr))
            }
            (KeyKind::Int64, BuildKeys::Int64(_)) => {
                let arr = key_arrays[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 probe key array"
                        )
                    })?
                    .clone();
                Ok(ProbeKeys::Int64(arr))
            }
            (KeyKind::UInt32, BuildKeys::UInt32(_)) => {
                let arr = key_arrays[0]
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected UInt32 probe key array"
                        )
                    })?
                    .clone();
                Ok(ProbeKeys::UInt32(arr))
            }
            (KeyKind::UInt64, BuildKeys::UInt64(_)) => {
                let arr = key_arrays[0]
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected UInt64 probe key array"
                        )
                    })?
                    .clone();
                Ok(ProbeKeys::UInt64(arr))
            }
            (KeyKind::Utf8, BuildKeys::Utf8 { .. }) => {
                let mut hashes = vec![0u64; num_rows];
                create_hashes(key_arrays, &join_random_state(), &mut hashes)?;
                let array = key_arrays[0]
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8 probe key array"
                        )
                    })?
                    .clone();
                Ok(ProbeKeys::Utf8 { array, hashes })
            }
            (KeyKind::LargeUtf8, BuildKeys::LargeUtf8 { .. }) => {
                let mut hashes = vec![0u64; num_rows];
                create_hashes(key_arrays, &join_random_state(), &mut hashes)?;
                let array = key_arrays[0]
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected LargeUtf8 probe key array"
                        )
                    })?
                    .clone();
                Ok(ProbeKeys::LargeUtf8 { array, hashes })
            }
            (KeyKind::Utf8View, BuildKeys::Utf8View { .. }) => {
                let mut hashes = vec![0u64; num_rows];
                create_hashes(key_arrays, &join_random_state(), &mut hashes)?;
                let array = key_arrays[0]
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8View probe key array"
                        )
                    })?
                    .clone();
                Ok(ProbeKeys::Utf8View { array, hashes })
            }
            (KeyKind::Rows, BuildKeys::Rows { converter, .. }) => {
                let normalized =
                    normalize_key_arrays(key_arrays.to_vec(), self.has_floats);
                let mut hashes = vec![0u64; num_rows];
                create_hashes(&normalized, &join_random_state(), &mut hashes)?;
                let rows = converter.convert_columns(&normalized)?;
                Ok(ProbeKeys::Rows { rows, hashes })
            }
            _ => internal_err!("Morsel HashJoin build/probe key representation mismatch"),
        }
    }

    pub(super) fn next_chunk(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
        batch_size: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let limit = batch_size.max(1);

        if self.output_projection.is_empty() {
            let count = probe_count_matches(cursor, table, limit)?;
            if count == 0 {
                return Ok(OperatorStep::NeedMoreInput(None));
            }
            let out = RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(count)),
            )?;
            return Ok(if cursor.is_exhausted() {
                OperatorStep::NeedMoreInput(Some(out))
            } else {
                OperatorStep::HaveMoreOutput(out)
            });
        }

        scratch.build_indices.clear();
        scratch.probe_indices.clear();
        if scratch.build_indices.capacity() < limit {
            scratch
                .build_indices
                .try_reserve(limit - scratch.build_indices.len())
                .map_err(|e| {
                    datafusion_common::exec_datafusion_err!(
                        "Morsel HashJoin probe buffer allocation failed: {e}"
                    )
                })?;
            scratch
                .probe_indices
                .try_reserve(limit - scratch.probe_indices.len())
                .map_err(|e| {
                    datafusion_common::exec_datafusion_err!(
                        "Morsel HashJoin probe buffer allocation failed: {e}"
                    )
                })?;
        }

        probe_collect_matches(cursor, table, limit, scratch)?;

        if scratch.build_indices.is_empty() {
            return Ok(OperatorStep::NeedMoreInput(None));
        }

        let build_idx = UInt32Array::new(
            ScalarBuffer::from(std::mem::take(&mut scratch.build_indices)),
            None,
        );
        let probe_idx = UInt32Array::new(
            ScalarBuffer::from(std::mem::take(&mut scratch.probe_indices)),
            None,
        );

        let mut columns = Vec::with_capacity(self.output_projection.len());
        for col in &self.output_projection {
            let taken = match col {
                JoinOutputCol::Build(idx) => {
                    take(table.payload.column(*idx).as_ref(), &build_idx, None)?
                }
                JoinOutputCol::Probe(idx) => {
                    take(cursor.batch.column(*idx).as_ref(), &probe_idx, None)?
                }
            };
            columns.push(taken);
        }

        // Return index buffers to worker scratch for reuse without reallocation.
        if let Ok(mut v) = build_idx.into_parts().1.into_inner().into_vec::<u32>() {
            v.clear();
            scratch.build_indices = v;
        }
        if let Ok(mut v) = probe_idx.into_parts().1.into_inner().into_vec::<u32>() {
            v.clear();
            scratch.probe_indices = v;
        }

        let out = RecordBatch::try_new(Arc::clone(&self.output_schema), columns)?;
        Ok(if cursor.is_exhausted() {
            OperatorStep::NeedMoreInput(Some(out))
        } else {
            OperatorStep::HaveMoreOutput(out)
        })
    }
}

macro_rules! probe_flat_int_step {
    ($cursor:expr, $flat:expr, $p_vals:expr, $limit:expr, $key_idx:expr, $backing_row:expr, $matched:expr, $hash_key:expr, $on_match:expr) => {{
        let slots = $flat.slots.as_slice();
        let dup_next = $flat.dup_next.as_slice();
        let mask = $flat.mask;
        let key_idx: usize = $key_idx;
        let backing_row: u32 = $backing_row;

        if $cursor.next_entry == 0 {
            if let Some(valid) = &$cursor.valid_mask
                && !valid.value(key_idx)
            {
                $cursor.pos += 1;
                continue;
            }
            let pk = $p_vals[key_idx];
            let mut slot_pos = ($hash_key(pk) as usize) & mask;
            let dup_head = loop {
                let slot = &slots[slot_pos];
                if slot.first_row == 0 {
                    break None;
                }
                if slot.key == pk {
                    $on_match(slot.first_row - 1, backing_row);
                    $matched += 1;
                    break Some(slot.dup_head);
                }
                slot_pos = (slot_pos + 1) & mask;
            };
            match dup_head {
                None | Some(0) => {
                    $cursor.pos += 1;
                    continue;
                }
                Some(head) => {
                    $cursor.next_entry = head;
                    if $matched == $limit {
                        break;
                    }
                }
            }
        }

        let mut entry = $cursor.next_entry;
        while entry != 0 {
            let build_row = (entry - 1) as usize;
            entry = dup_next[build_row];
            $on_match(build_row as u32, backing_row);
            $matched += 1;
            if $matched == $limit {
                $cursor.next_entry = entry;
                if entry == 0 {
                    $cursor.pos += 1;
                }
                break;
            }
        }
        if $matched < $limit {
            $cursor.next_entry = 0;
            $cursor.pos += 1;
        }
    }};
}

macro_rules! probe_flat_int_core {
    ($cursor:expr, $flat:expr, $p_vals:expr, $limit:expr, $hash_key:expr, $on_match:expr) => {{
        let mut matched = 0usize;
        match &$cursor.rows {
            ProbeRows::All { num_rows } => {
                let num_rows = *num_rows;
                while $cursor.pos < num_rows && matched < $limit {
                    let probe_row = $cursor.pos;
                    probe_flat_int_step!(
                        $cursor,
                        $flat,
                        $p_vals,
                        $limit,
                        probe_row,
                        probe_row as u32,
                        matched,
                        $hash_key,
                        $on_match
                    );
                }
            }
            ProbeRows::Mask { bits, num_rows } => {
                let slots = $flat.slots.as_slice();
                let dup_next = $flat.dup_next.as_slice();
                let mask = $flat.mask;
                let num_rows = *num_rows;

                // Resume pending duplicate chain for `$cursor.pos` first.
                if $cursor.next_entry != 0 && $cursor.pos < num_rows {
                    let probe_row = $cursor.pos;
                    let mut entry = $cursor.next_entry;
                    while entry != 0 && matched < $limit {
                        let build_row = (entry - 1) as usize;
                        entry = dup_next[build_row];
                        $on_match(build_row as u32, probe_row as u32);
                        matched += 1;
                    }
                    if entry == 0 {
                        $cursor.next_entry = 0;
                        $cursor.pos = probe_row + 1;
                    } else {
                        $cursor.next_entry = entry;
                    }
                }

                if matched < $limit && $cursor.pos < num_rows {
                    let start = $cursor.pos;
                    let mut hit_limit = false;
                    for rel in bits.slice(start, num_rows - start).set_indices() {
                        let probe_row = start + rel;
                        let pk = $p_vals[probe_row];
                        let mut slot_pos = ($hash_key(pk) as usize) & mask;
                        let dup_head = loop {
                            let slot = &slots[slot_pos];
                            if slot.first_row == 0 {
                                break None;
                            }
                            if slot.key == pk {
                                $on_match(slot.first_row - 1, probe_row as u32);
                                matched += 1;
                                break Some(slot.dup_head);
                            }
                            slot_pos = (slot_pos + 1) & mask;
                        };
                        match dup_head {
                            None | Some(0) => {
                                if matched == $limit {
                                    $cursor.pos = probe_row + 1;
                                    $cursor.next_entry = 0;
                                    hit_limit = true;
                                    break;
                                }
                            }
                            Some(head) => {
                                if matched == $limit {
                                    $cursor.pos = probe_row;
                                    $cursor.next_entry = head;
                                    hit_limit = true;
                                    break;
                                }
                                let mut entry = head;
                                while entry != 0 {
                                    let build_row = (entry - 1) as usize;
                                    entry = dup_next[build_row];
                                    $on_match(build_row as u32, probe_row as u32);
                                    matched += 1;
                                    if matched == $limit {
                                        $cursor.next_entry = entry;
                                        $cursor.pos = if entry == 0 {
                                            probe_row + 1
                                        } else {
                                            probe_row
                                        };
                                        hit_limit = true;
                                        break;
                                    }
                                }
                                if hit_limit {
                                    break;
                                }
                            }
                        }
                    }
                    if !hit_limit {
                        $cursor.pos = num_rows;
                        $cursor.next_entry = 0;
                    }
                }
            }
            ProbeRows::BackingIndices(indices) => {
                let sel = indices.values();
                let len = sel.len();
                while $cursor.pos < len && matched < $limit {
                    let backing_row = sel[$cursor.pos];
                    probe_flat_int_step!(
                        $cursor,
                        $flat,
                        $p_vals,
                        $limit,
                        backing_row as usize,
                        backing_row,
                        matched,
                        $hash_key,
                        $on_match
                    );
                }
            }
            ProbeRows::CompactedKeys { backing_rows } => {
                let backing = backing_rows.values();
                let len = backing.len();
                while $cursor.pos < len && matched < $limit {
                    let key_idx = $cursor.pos;
                    let backing_row = backing[key_idx];
                    probe_flat_int_step!(
                        $cursor,
                        $flat,
                        $p_vals,
                        $limit,
                        key_idx,
                        backing_row,
                        matched,
                        $hash_key,
                        $on_match
                    );
                }
            }
        }
        matched
    }};
}

macro_rules! probe_loop_step {
    ($cursor:expr, $table:expr, $limit:expr, $key_idx:expr, $backing_row:expr, $matched:expr, $get_hash:expr, $is_match:expr, $on_match:expr) => {{
        let buckets = $table.buckets.as_slice();
        let next = $table.next.as_slice();
        let mask = $table.mask;
        let key_idx: usize = $key_idx;
        let backing_row: u32 = $backing_row;

        if $cursor.next_entry == 0 {
            if let Some(valid) = &$cursor.valid_mask
                && !valid.value(key_idx)
            {
                $cursor.pos += 1;
                continue;
            }
            let h: u64 = $get_hash(key_idx);
            let entry = buckets[(h as usize) & mask];
            if entry == 0 {
                $cursor.pos += 1;
                continue;
            }
            $cursor.next_entry = entry;
        }

        let h: u64 = $get_hash(key_idx);
        let mut entry = $cursor.next_entry;
        while entry != 0 {
            let build_row = (entry - 1) as usize;
            entry = next[build_row];
            if $is_match(build_row, key_idx, h) {
                $on_match(build_row as u32, backing_row);
                $matched += 1;
                if $matched == $limit {
                    $cursor.next_entry = entry;
                    if entry == 0 {
                        $cursor.pos += 1;
                    }
                    break;
                }
            }
        }
        if $matched < $limit {
            $cursor.next_entry = 0;
            $cursor.pos += 1;
        }
    }};
}

macro_rules! probe_loop_core {
    ($cursor:expr, $table:expr, $limit:expr, $get_hash:expr, $is_match:expr, $on_match:expr) => {{
        let mut matched = 0usize;
        match &$cursor.rows {
            ProbeRows::All { num_rows } => {
                let num_rows = *num_rows;
                while $cursor.pos < num_rows && matched < $limit {
                    let probe_row = $cursor.pos;
                    probe_loop_step!(
                        $cursor,
                        $table,
                        $limit,
                        probe_row,
                        probe_row as u32,
                        matched,
                        $get_hash,
                        $is_match,
                        $on_match
                    );
                }
            }
            ProbeRows::CompactedKeys { backing_rows } => {
                let backing = backing_rows.values();
                let len = backing.len();
                while $cursor.pos < len && matched < $limit {
                    let key_idx = $cursor.pos;
                    let backing_row = backing[key_idx];
                    probe_loop_step!(
                        $cursor,
                        $table,
                        $limit,
                        key_idx,
                        backing_row,
                        matched,
                        $get_hash,
                        $is_match,
                        $on_match
                    );
                }
            }
            ProbeRows::Mask { .. } | ProbeRows::BackingIndices(_) => {
                unreachable!("non-integer probe keys always use All or CompactedKeys")
            }
        }
        matched
    }};
}

fn probe_collect_matches(
    cursor: &mut ProbeCursor,
    table: &FinalizedHashTable,
    limit: usize,
    scratch: &mut ProbeScratch,
) -> Result<()> {
    let build_out = &mut scratch.build_indices;
    let probe_out = &mut scratch.probe_indices;

    match (&cursor.keys, &table.keys) {
        (ProbeKeys::Int32(p_arr), BuildKeys::Int32(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: i32| mix_u64(v as u32 as u64),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (ProbeKeys::Int64(p_arr), BuildKeys::Int64(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: i64| mix_u64(v as u64),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (ProbeKeys::UInt32(p_arr), BuildKeys::UInt32(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: u32| mix_u64(v as u64),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (ProbeKeys::UInt64(p_arr), BuildKeys::UInt64(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: u64| mix_u64(v),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (
            ProbeKeys::Utf8 {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::Utf8 {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_arr.value(b) == p_arr.value(p),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (
            ProbeKeys::LargeUtf8 {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::LargeUtf8 {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_arr.value(b) == p_arr.value(p),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (
            ProbeKeys::Utf8View {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::Utf8View {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_arr.value(b) == p_arr.value(p),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (
            ProbeKeys::Rows {
                rows: p_rows,
                hashes: p_hashes,
            },
            BuildKeys::Rows {
                rows: b_rows,
                hashes: b_hashes,
                ..
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_rows.row(b).as_ref() == p_rows.row(p).as_ref(),
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        _ => {
            return internal_err!(
                "Morsel HashJoin probe/build key representation mismatch"
            );
        }
    }
    Ok(())
}

fn probe_count_matches(
    cursor: &mut ProbeCursor,
    table: &FinalizedHashTable,
    limit: usize,
) -> Result<usize> {
    let matched = match (&cursor.keys, &table.keys) {
        (ProbeKeys::Int32(p_arr), BuildKeys::Int32(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: i32| mix_u64(v as u32 as u64),
                |_b: u32, _p: u32| {}
            )
        }
        (ProbeKeys::Int64(p_arr), BuildKeys::Int64(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: i64| mix_u64(v as u64),
                |_b: u32, _p: u32| {}
            )
        }
        (ProbeKeys::UInt32(p_arr), BuildKeys::UInt32(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: u32| mix_u64(v as u64),
                |_b: u32, _p: u32| {}
            )
        }
        (ProbeKeys::UInt64(p_arr), BuildKeys::UInt64(flat)) => {
            let p_vals = p_arr.values();
            probe_flat_int_core!(
                cursor,
                flat,
                p_vals,
                limit,
                |v: u64| mix_u64(v),
                |_b: u32, _p: u32| {}
            )
        }
        (
            ProbeKeys::Utf8 {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::Utf8 {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_arr.value(b) == p_arr.value(p),
                |_b: u32, _p: u32| {}
            )
        }
        (
            ProbeKeys::LargeUtf8 {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::LargeUtf8 {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_arr.value(b) == p_arr.value(p),
                |_b: u32, _p: u32| {}
            )
        }
        (
            ProbeKeys::Utf8View {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::Utf8View {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_arr.value(b) == p_arr.value(p),
                |_b: u32, _p: u32| {}
            )
        }
        (
            ProbeKeys::Rows {
                rows: p_rows,
                hashes: p_hashes,
            },
            BuildKeys::Rows {
                rows: b_rows,
                hashes: b_hashes,
                ..
            },
        ) => {
            probe_loop_core!(
                cursor,
                table,
                limit,
                |p: usize| p_hashes[p],
                |b: usize, p: usize, h: u64| b_hashes[b] == h
                    && b_rows.row(b).as_ref() == p_rows.row(p).as_ref(),
                |_b: u32, _p: u32| {}
            )
        }
        _ => {
            return internal_err!(
                "Morsel HashJoin probe/build key representation mismatch"
            );
        }
    };
    Ok(matched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use arrow::datatypes::{Field, Schema};
    use datafusion_execution::memory_pool::{
        MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };

    use super::super::{MemorySource, Sink, Source};

    fn dummy_build_pipeline(
        schema: SchemaRef,
        sink: Arc<HashJoinBuildSink>,
    ) -> Arc<Pipeline> {
        Arc::new(Pipeline {
            source: Source::Memory(MemorySource {
                table: "build".to_owned(),
                batches: vec![],
                projection: None,
                schema,
            }),
            operators: vec![],
            workers: NonZeroUsize::MIN,
            sink: Sink::HashJoinBuild(sink),
            sink_capacity: 1,
        })
    }

    #[tokio::test]
    async fn probe_view_all_mask_indices_fanout_and_sliced_batches() -> Result<()> {
        // Build side: duplicate key 10 (3 rows), key 20 (2 rows), key NULL (ignored)
        let build_schema = Arc::new(Schema::new(vec![
            Field::new("bk", DataType::Int64, true),
            Field::new("bval", DataType::Int32, false),
        ]));
        let build_batch = RecordBatch::try_new(
            Arc::clone(&build_schema),
            vec![
                Arc::new(Int64Array::from(vec![
                    Some(10),
                    Some(10),
                    Some(20),
                    None,
                    Some(10),
                    Some(20),
                ])),
                Arc::new(Int32Array::from(vec![100, 101, 200, 999, 102, 201])),
            ],
        )?;
        let payload_schema = Arc::new(build_schema.project(&[1])?);
        let build_sink = Arc::new(
            HashJoinBuildSink::try_new(
                vec![Arc::new(Column::new("bk", 0))],
                vec!["bk".to_owned()],
                vec![DataType::Int64],
                vec![1],
                Arc::clone(&payload_schema),
            )?
            .expect("valid build sink"),
        );

        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let res = MemoryConsumer::new("test_hj_view").register(&pool);
        let mut build_state = build_sink.new_state(res);
        // Feed build side through a BatchView Mask that excludes row 5 (20, 201)
        let build_mask = BooleanArray::from(vec![true, true, true, true, true, false]);
        let build_view = BatchView::from_boolean_array(build_batch, &build_mask)?;
        build_state.add_view(&build_view)?;
        let (tx, _rx) = mpsc::channel(2);
        let table = build_sink.finalize(build_state, &tx).await?.unwrap();

        // Probe side (sliced): keys [10, NULL, 20, 99, 10, 20]
        let probe_schema = Arc::new(Schema::new(vec![
            Field::new("pk", DataType::Int64, true),
            Field::new("pid", DataType::Int32, false),
        ]));
        let probe_base = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![
                Arc::new(Int64Array::from(vec![
                    Some(10),
                    Some(10),
                    None,
                    Some(20),
                    Some(99),
                    Some(10),
                    Some(20),
                    Some(10),
                ])),
                Arc::new(Int32Array::from(vec![-1, 0, 1, 2, 3, 4, 5, -2])),
            ],
        )?;
        let probe_sliced = probe_base.slice(1, 6);
        let probe_mask = BooleanArray::from(vec![
            Some(true),  // pid=0, pk=10 -> 3 matches
            Some(true),  // pid=1, pk=NULL -> 0 matches
            Some(true),  // pid=2, pk=20 -> 1 match
            Some(true),  // pid=3, pk=99 -> 0 matches
            Some(false), // pid=4 rejected
            Some(true),  // pid=5, pk=20 -> 1 match
        ]);
        let view_mask = BatchView::from_boolean_array(probe_sliced.clone(), &probe_mask)?;
        let view_indices = view_mask.clone().with_indices_selection();
        let view_all = BatchView::all(view_mask.materialize_ref()?);

        let out_schema = Arc::new(Schema::new(vec![
            Field::new("bval", DataType::Int32, false),
            Field::new("pid", DataType::Int32, false),
        ]));
        let probe = HashJoinProbe::new(HashJoinProbeConfig {
            build_pipeline: dummy_build_pipeline(build_schema, Arc::clone(&build_sink)),
            build_side: BuildSide::Left,
            probe_key_exprs: vec![Arc::new(Column::new("pk", 0))],
            probe_key_names: vec!["pk".to_owned()],
            key_types: vec![DataType::Int64],
            output_projection: vec![JoinOutputCol::Build(0), JoinOutputCol::Probe(1)],
            output_schema: out_schema,
            build_id: 0,
        });

        let probe_res = MemoryConsumer::new("test_hj_probe").register(&pool);
        let drain = |view: BatchView| -> Result<Vec<(i32, i32)>> {
            let mut scratch = ProbeScratch::default();
            let mut pairs = Vec::new();
            let Some(mut cursor) = probe.start_probe_view(view, &table, &probe_res)?
            else {
                return Ok(pairs);
            };
            loop {
                // Tiny batch_size=2 forces fanout resumption across duplicate chains!
                match probe.next_chunk(&mut cursor, &table, 2, &mut scratch)? {
                    OperatorStep::NeedMoreInput(opt) => {
                        if let Some(batch) = opt {
                            assert!(batch.num_rows() <= 2);
                            let b = batch
                                .column(0)
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .unwrap();
                            let p = batch
                                .column(1)
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .unwrap();
                            for i in 0..batch.num_rows() {
                                pairs.push((b.value(i), p.value(i)));
                            }
                        }
                        break;
                    }
                    OperatorStep::HaveMoreOutput(batch) => {
                        assert_eq!(batch.num_rows(), 2);
                        let b = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        let p = batch
                            .column(1)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        for i in 0..batch.num_rows() {
                            pairs.push((b.value(i), p.value(i)));
                        }
                    }
                }
            }
            pairs.sort_unstable();
            Ok(pairs)
        };

        let expected = vec![(100, 0), (101, 0), (102, 0), (200, 2), (200, 5)];
        assert_eq!(drain(view_all)?, expected);
        assert_eq!(drain(view_mask.clone())?, expected);
        assert_eq!(drain(view_indices.clone())?, expected);

        // Count-only (zero-column output) across Mask and Indices with tiny batch_size=2
        let count_probe = HashJoinProbe::new(HashJoinProbeConfig {
            build_pipeline: dummy_build_pipeline(
                Arc::new(Schema::empty()),
                Arc::clone(&build_sink),
            ),
            build_side: BuildSide::Left,
            probe_key_exprs: vec![Arc::new(Column::new("pk", 0))],
            probe_key_names: vec!["pk".to_owned()],
            key_types: vec![DataType::Int64],
            output_projection: vec![],
            output_schema: Arc::new(Schema::empty()),
            build_id: 0,
        });
        for view in [view_mask, view_indices] {
            let mut scratch = ProbeScratch::default();
            let mut cursor = count_probe
                .start_probe_view(view, &table, &probe_res)?
                .unwrap();
            let mut total = 0usize;
            loop {
                match count_probe.next_chunk(&mut cursor, &table, 2, &mut scratch)? {
                    OperatorStep::NeedMoreInput(opt) => {
                        total += opt.map_or(0, |b| b.num_rows());
                        break;
                    }
                    OperatorStep::HaveMoreOutput(b) => {
                        assert_eq!(b.num_columns(), 0);
                        total += b.num_rows();
                    }
                }
            }
            assert_eq!(total, 5);
        }

        drop(table);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn probe_cursor_reservation_accounts_for_selection_and_compaction_only()
    -> Result<()> {
        use datafusion_execution::memory_pool::GreedyMemoryPool;

        let build_schema = Arc::new(Schema::new(vec![
            Field::new("bk", DataType::Int64, false),
            Field::new("bval", DataType::Int64, false),
        ]));
        let build_batch = RecordBatch::try_new(
            Arc::clone(&build_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..16)),
                Arc::new(Int64Array::from_iter_values((0..16).map(|i| i * 10))),
            ],
        )?;
        let payload_schema = Arc::new(build_schema.project(&[1])?);
        let build_sink = Arc::new(
            HashJoinBuildSink::try_new(
                vec![Arc::new(Column::new("bk", 0))],
                vec!["bk".to_owned()],
                vec![DataType::Int64],
                vec![1],
                payload_schema,
            )?
            .expect("valid build sink"),
        );

        let build_pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut build_state =
            build_sink.new_state(MemoryConsumer::new("test_build").register(&build_pool));
        build_state.add_batch(&build_batch)?;
        let (tx, _rx) = mpsc::channel(2);
        let table = build_sink.finalize(build_state, &tx).await?.unwrap();

        let probe_schema = Arc::new(Schema::new(vec![
            Field::new("pk", DataType::Int64, false),
            Field::new("wide", DataType::Utf8, false),
        ]));
        let n = 4096usize;
        let wide_str = "w".repeat(256);
        let probe_batch = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..n as i64)),
                Arc::new(StringArray::from(vec![wide_str.as_str(); n])),
            ],
        )?;
        assert!(get_record_batch_memory_size(&probe_batch) > 1024 * 1024);

        let out_schema = Arc::new(Schema::new(vec![
            Field::new("pk", DataType::Int64, false),
            Field::new("wide", DataType::Utf8, false),
            Field::new("bval", DataType::Int64, false),
        ]));
        let probe = HashJoinProbe::new(HashJoinProbeConfig {
            build_pipeline: dummy_build_pipeline(build_schema, Arc::clone(&build_sink)),
            build_side: BuildSide::Right,
            probe_key_exprs: vec![Arc::new(Column::new("pk", 0))],
            probe_key_names: vec!["pk".to_owned()],
            key_types: vec![DataType::Int64],
            output_projection: vec![
                JoinOutputCol::Probe(0),
                JoinOutputCol::Probe(1),
                JoinOutputCol::Build(0),
            ],
            output_schema: out_schema,
            build_id: 0,
        });

        // 1. Unfiltered Selection::All charges 0 bytes (never charges preexisting storage).
        let pool_512k: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(512 * 1024));
        let res_512k = MemoryConsumer::new("test_probe_512k").register(&pool_512k);
        {
            let cursor = probe
                .start_probe_view(BatchView::all(probe_batch.clone()), &table, &res_512k)?
                .unwrap();
            assert_eq!(cursor.batch.num_rows(), n);
            assert_eq!(pool_512k.reserved(), 0);
        }
        assert_eq!(pool_512k.reserved(), 0);

        // 2. 75% filtered view (`gate <> 0`): below 16x compaction threshold, so cursor
        //    keeps the backing batch without charging it — reserves only the selection
        //    bitmap (~512 B) and succeeds under 512 KiB pool.
        let mask_75 = BooleanArray::from((0..n).map(|i| i % 4 != 0).collect::<Vec<_>>());
        let view_75 = BatchView::from_boolean_array(probe_batch.clone(), &mask_75)?;
        let sel_75_bytes = view_75.selection_memory_size();
        assert!(sel_75_bytes > 0 && sel_75_bytes < 4096);
        {
            let cursor = probe.start_probe_view(view_75, &table, &res_512k)?.unwrap();
            assert_eq!(cursor.batch.num_rows(), n);
            assert_eq!(pool_512k.reserved(), sel_75_bytes);
        }
        assert_eq!(pool_512k.reserved(), 0);

        // 3. Tiny selection (16/4096 rows): triggers proactive threshold compaction
        //    (wasted > 512 KiB and ratio > 16x), reserves the newly compacted copy
        //    (~4.5 KiB), and drops the 1 MiB backing batch from the cursor.
        let mask_tiny =
            BooleanArray::from((0..n).map(|i| i % 256 == 0).collect::<Vec<_>>());
        let view_tiny = BatchView::from_boolean_array(probe_batch.clone(), &mask_tiny)?;
        let pool_64k: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024));
        let res_64k = MemoryConsumer::new("test_probe_64k").register(&pool_64k);
        {
            let cursor = probe
                .start_probe_view(view_tiny.clone(), &table, &res_64k)?
                .unwrap();
            assert_eq!(cursor.batch.num_rows(), 16);
            let compacted_bytes = get_record_batch_memory_size(&cursor.batch);
            assert!(compacted_bytes > 0 && compacted_bytes < 64 * 1024);
            assert_eq!(pool_64k.reserved(), compacted_bytes);
        }
        assert_eq!(pool_64k.reserved(), 0);

        // 4. Tiny selection under a 2 KiB pool (smaller than the compacted copy):
        //    gracefully skips compaction and charges only the selection buffer instead
        //    of failing with ResourcesExhausted.
        let pool_2k: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(2048));
        let res_2k = MemoryConsumer::new("test_probe_2k").register(&pool_2k);
        {
            let cursor = probe
                .start_probe_view(view_tiny.clone(), &table, &res_2k)?
                .unwrap();
            assert_eq!(cursor.batch.num_rows(), n);
            assert_eq!(pool_2k.reserved(), view_tiny.selection_memory_size());
        }
        assert_eq!(pool_2k.reserved(), 0);

        drop(table);
        assert_eq!(build_pool.reserved(), 0);
        Ok(())
    }
}
