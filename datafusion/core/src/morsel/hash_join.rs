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
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use tokio::sync::mpsc;

use super::Pipeline;

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
        let dir_bytes = bucket_count
            .saturating_add(num_rows)
            .saturating_mul(size_of::<u32>());
        let payload_concat_bytes =
            if self.payload_columns.is_empty() || batches.len() <= 1 {
                0
            } else {
                batches
                    .iter()
                    .map(get_record_batch_memory_size)
                    .fold(0usize, usize::saturating_add)
            };
        let keys_peak_bytes = match self.key_kind {
            KeyKind::Int32 | KeyKind::UInt32 => num_rows.saturating_mul(size_of::<u32>()),
            KeyKind::Int64 | KeyKind::UInt64 => num_rows.saturating_mul(size_of::<u64>()),
            KeyKind::Utf8 | KeyKind::LargeUtf8 | KeyKind::Utf8View => {
                let concat_bytes = if key_batches.len() <= 1 {
                    0
                } else {
                    key_batches
                        .iter()
                        .map(|kb| kb[0].get_array_memory_size())
                        .fold(0usize, usize::saturating_add)
                };
                concat_bytes.saturating_add(num_rows.saturating_mul(size_of::<u64>()))
            }
            KeyKind::Rows => {
                let raw_key_bytes = key_batches
                    .iter()
                    .flat_map(|kb| kb.iter())
                    .map(|a| a.get_array_memory_size())
                    .fold(0usize, usize::saturating_add);
                raw_key_bytes
                    .saturating_add(num_rows.saturating_mul(16))
                    .saturating_add(num_rows.saturating_mul(size_of::<u64>()))
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

        let build_keys = self.build_keys(&mut key_batches, &hash_chunks, num_rows)?;
        drop(key_batches);

        let mask = bucket_count - 1;
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

    fn build_keys(
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
            KeyKind::Int32 => {
                let mut vals = Vec::with_capacity(num_rows);
                for kb in key_batches.drain(..) {
                    let arr =
                        kb[0].as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int32 join key array"
                            )
                        })?;
                    vals.extend_from_slice(arr.values());
                }
                Ok(BuildKeys::Int32(vals))
            }
            KeyKind::Int64 => {
                let mut vals = Vec::with_capacity(num_rows);
                for kb in key_batches.drain(..) {
                    let arr =
                        kb[0].as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 join key array"
                            )
                        })?;
                    vals.extend_from_slice(arr.values());
                }
                Ok(BuildKeys::Int64(vals))
            }
            KeyKind::UInt32 => {
                let mut vals = Vec::with_capacity(num_rows);
                for kb in key_batches.drain(..) {
                    let arr = kb[0].as_any().downcast_ref::<UInt32Array>().ok_or_else(
                        || {
                            datafusion_common::internal_datafusion_err!(
                                "Expected UInt32 join key array"
                            )
                        },
                    )?;
                    vals.extend_from_slice(arr.values());
                }
                Ok(BuildKeys::UInt32(vals))
            }
            KeyKind::UInt64 => {
                let mut vals = Vec::with_capacity(num_rows);
                for kb in key_batches.drain(..) {
                    let arr = kb[0].as_any().downcast_ref::<UInt64Array>().ok_or_else(
                        || {
                            datafusion_common::internal_datafusion_err!(
                                "Expected UInt64 join key array"
                            )
                        },
                    )?;
                    vals.extend_from_slice(arr.values());
                }
                Ok(BuildKeys::UInt64(vals))
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

enum BuildKeys {
    Empty,
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
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
            Self::Int32(v) => v.capacity().saturating_mul(size_of::<i32>()),
            Self::Int64(v) => v.capacity().saturating_mul(size_of::<i64>()),
            Self::UInt32(v) => v.capacity().saturating_mul(size_of::<u32>()),
            Self::UInt64(v) => v.capacity().saturating_mul(size_of::<u64>()),
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

pub(super) struct ProbeCursor {
    batch: RecordBatch,
    keys: ProbeKeys,
    valid_mask: Option<BooleanBuffer>,
    probe_row: usize,
    next_entry: u32,
}

impl ProbeCursor {
    fn is_exhausted(&self) -> bool {
        self.probe_row >= self.batch.num_rows()
    }
}

#[derive(Debug)]
pub(super) struct HashJoinProbe {
    pub(super) build_pipeline: Arc<Pipeline>,
    pub(super) build_side: BuildSide,
    probe_key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    probe_key_names: Vec<String>,
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
        Self {
            build_pipeline: config.build_pipeline,
            build_side: config.build_side,
            probe_key_exprs: config.probe_key_exprs,
            probe_key_names: config.probe_key_names,
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

    pub(super) fn start_probe(
        &self,
        batch: RecordBatch,
        table: &FinalizedHashTable,
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
        let keys = match (&self.key_kind, &table.keys) {
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
                ProbeKeys::Int32(arr)
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
                ProbeKeys::Int64(arr)
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
                ProbeKeys::UInt32(arr)
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
                ProbeKeys::UInt64(arr)
            }
            (KeyKind::Utf8, BuildKeys::Utf8 { .. }) => {
                let mut hashes = vec![0u64; num_rows];
                create_hashes(&key_arrays, &join_random_state(), &mut hashes)?;
                let array = key_arrays[0]
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8 probe key array"
                        )
                    })?
                    .clone();
                ProbeKeys::Utf8 { array, hashes }
            }
            (KeyKind::LargeUtf8, BuildKeys::LargeUtf8 { .. }) => {
                let mut hashes = vec![0u64; num_rows];
                create_hashes(&key_arrays, &join_random_state(), &mut hashes)?;
                let array = key_arrays[0]
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected LargeUtf8 probe key array"
                        )
                    })?
                    .clone();
                ProbeKeys::LargeUtf8 { array, hashes }
            }
            (KeyKind::Utf8View, BuildKeys::Utf8View { .. }) => {
                let mut hashes = vec![0u64; num_rows];
                create_hashes(&key_arrays, &join_random_state(), &mut hashes)?;
                let array = key_arrays[0]
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8View probe key array"
                        )
                    })?
                    .clone();
                ProbeKeys::Utf8View { array, hashes }
            }
            (KeyKind::Rows, BuildKeys::Rows { converter, .. }) => {
                let normalized = normalize_key_arrays(key_arrays, self.has_floats);
                let mut hashes = vec![0u64; num_rows];
                create_hashes(&normalized, &join_random_state(), &mut hashes)?;
                let rows = converter.convert_columns(&normalized)?;
                ProbeKeys::Rows { rows, hashes }
            }
            _ => {
                return internal_err!(
                    "Morsel HashJoin build/probe key representation mismatch"
                );
            }
        };

        Ok(Some(ProbeCursor {
            batch,
            keys,
            valid_mask,
            probe_row: 0,
            next_entry: 0,
        }))
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

macro_rules! probe_loop_core {
    ($cursor:expr, $table:expr, $limit:expr, $num_rows:expr, $get_hash:expr, $is_match:expr, $on_match:expr) => {{
        let buckets = $table.buckets.as_slice();
        let next = $table.next.as_slice();
        let mask = $table.mask;
        let mut matched = 0usize;

        while $cursor.probe_row < $num_rows && matched < $limit {
            let probe_row = $cursor.probe_row;
            if $cursor.next_entry == 0 {
                if let Some(valid) = &$cursor.valid_mask
                    && !valid.value(probe_row)
                {
                    $cursor.probe_row += 1;
                    continue;
                }
                let h: u64 = $get_hash(probe_row);
                let entry = buckets[(h as usize) & mask];
                if entry == 0 {
                    $cursor.probe_row += 1;
                    continue;
                }
                $cursor.next_entry = entry;
            }

            let h: u64 = $get_hash(probe_row);
            let mut entry = $cursor.next_entry;
            while entry != 0 {
                let build_row = (entry - 1) as usize;
                entry = next[build_row];
                if $is_match(build_row, probe_row, h) {
                    $on_match(build_row as u32, probe_row as u32);
                    matched += 1;
                    if matched == $limit {
                        $cursor.next_entry = entry;
                        if entry == 0 {
                            $cursor.probe_row += 1;
                        }
                        break;
                    }
                }
            }
            if matched < $limit {
                $cursor.next_entry = 0;
                $cursor.probe_row += 1;
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
    let num_rows = cursor.batch.num_rows();
    let build_out = &mut scratch.build_indices;
    let probe_out = &mut scratch.probe_indices;

    match (&cursor.keys, &table.keys) {
        (ProbeKeys::Int32(p_arr), BuildKeys::Int32(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p] as u32 as u64),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (ProbeKeys::Int64(p_arr), BuildKeys::Int64(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p] as u64),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (ProbeKeys::UInt32(p_arr), BuildKeys::UInt32(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p] as u64),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
                |b: u32, p: u32| {
                    build_out.push(b);
                    probe_out.push(p);
                }
            );
        }
        (ProbeKeys::UInt64(p_arr), BuildKeys::UInt64(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p]),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
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
                num_rows,
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
                num_rows,
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
                num_rows,
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
                num_rows,
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
    let num_rows = cursor.batch.num_rows();
    let matched = match (&cursor.keys, &table.keys) {
        (ProbeKeys::Int32(p_arr), BuildKeys::Int32(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p] as u32 as u64),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
                |_b: u32, _p: u32| {}
            )
        }
        (ProbeKeys::Int64(p_arr), BuildKeys::Int64(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p] as u64),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
                |_b: u32, _p: u32| {}
            )
        }
        (ProbeKeys::UInt32(p_arr), BuildKeys::UInt32(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p] as u64),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
                |_b: u32, _p: u32| {}
            )
        }
        (ProbeKeys::UInt64(p_arr), BuildKeys::UInt64(b_vals)) => {
            let p_vals = p_arr.values();
            probe_loop_core!(
                cursor,
                table,
                limit,
                num_rows,
                |p: usize| mix_u64(p_vals[p]),
                |b: usize, p: usize, _h: u64| b_vals[b] == p_vals[p],
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
                num_rows,
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
                num_rows,
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
                num_rows,
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
                num_rows,
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
