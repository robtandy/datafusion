// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray,
    NullBufferBuilder, RecordBatch, RecordBatchOptions, StringArray, StringViewArray,
    UInt32Array, UInt64Array, new_null_array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer, ScalarBuffer};
use arrow::compute::{concat, concat_batches, filter, filter_record_batch, take};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_common::{
    JoinType, NullEquality, Result, ScalarValue, exec_err, float_semantics, internal_err,
};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use tokio::sync::mpsc;

use super::Pipeline;
use super::batch::{
    BatchView, Selection, collect_expr_column_indices, remap_expr_columns,
};
use super::join_filter::{
    DynamicJoinFilterConfig, DynamicJoinFilterData, JoinMembershipFilter,
    LocalJoinKeySummary, SplitBlockBloomFilter, try_build_small_inlist_from_chunks,
};

const FINALIZE_YIELD_CHUNK: usize = 4096;

// Fixed hash seed shared between build and probe workers within a run.
const JOIN_HASH_SEED: u64 = 0x517c_c1b7_2722_0a95;

pub(super) fn join_random_state() -> RandomState {
    RandomState::with_seed(JOIN_HASH_SEED)
}

#[inline]
pub(super) fn mix_u64(mut x: u64) -> u64 {
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
pub(super) enum ProbeJoinKind {
    Inner,
    ProbeOuter,
    BuildOuter,
    FullOuter,
    ProbeSemi,
    BuildSemi,
    ProbeAnti { null_aware: bool },
    BuildAnti,
    Cross,
}

impl ProbeJoinKind {
    pub(super) fn from_logical(
        join_type: JoinType,
        build_side: BuildSide,
        null_aware: bool,
        is_cross: bool,
    ) -> Result<Self> {
        if is_cross {
            return match join_type {
                JoinType::Inner if !null_aware => Ok(Self::Cross),
                _ => internal_err!("Cross join only supported for non-null-aware Inner"),
            };
        }
        Ok(match (join_type, build_side) {
            (JoinType::Inner, _) => Self::Inner,
            (JoinType::Left, BuildSide::Right) | (JoinType::Right, BuildSide::Left) => {
                Self::ProbeOuter
            }
            (JoinType::Left, BuildSide::Left) | (JoinType::Right, BuildSide::Right) => {
                Self::BuildOuter
            }
            (JoinType::Full, _) => Self::FullOuter,
            (JoinType::LeftSemi, BuildSide::Right)
            | (JoinType::RightSemi, BuildSide::Left) => Self::ProbeSemi,
            (JoinType::LeftSemi, BuildSide::Left)
            | (JoinType::RightSemi, BuildSide::Right) => Self::BuildSemi,
            (JoinType::LeftAnti, BuildSide::Right)
            | (JoinType::RightAnti, BuildSide::Left) => Self::ProbeAnti { null_aware },
            (JoinType::LeftAnti, BuildSide::Left)
            | (JoinType::RightAnti, BuildSide::Right) => {
                if null_aware {
                    return internal_err!(
                        "null-aware anti join must place subquery side on build"
                    );
                }
                Self::BuildAnti
            }
            (JoinType::LeftMark | JoinType::RightMark, _) => {
                return internal_err!("Mark join is not supported in morsel hash join");
            }
        })
    }

    pub(super) fn needs_matched_bitset(self) -> bool {
        matches!(
            self,
            Self::BuildOuter | Self::FullOuter | Self::BuildSemi | Self::BuildAnti
        )
    }

    pub(super) fn needs_build_emission(self) -> bool {
        self.needs_matched_bitset()
    }

    pub(super) fn emit_matched_build(self) -> bool {
        matches!(self, Self::BuildSemi)
    }

    pub(super) fn keep_null_key_build_rows(self) -> bool {
        matches!(self, Self::BuildOuter | Self::FullOuter | Self::BuildAnti)
    }

    pub(super) fn empty_build_short_circuits(self) -> bool {
        matches!(
            self,
            Self::Inner
                | Self::BuildOuter
                | Self::ProbeSemi
                | Self::BuildSemi
                | Self::BuildAnti
                | Self::Cross
        )
    }

    pub(super) fn emits_unmatched_probe(self) -> bool {
        matches!(
            self,
            Self::ProbeOuter | Self::FullOuter | Self::ProbeAnti { .. }
        )
    }

    pub(super) fn keeps_null_probe_keys(self) -> bool {
        matches!(
            self,
            Self::ProbeOuter | Self::FullOuter | Self::ProbeAnti { null_aware: false }
        )
    }

    pub(super) fn is_null_aware_anti(self) -> bool {
        matches!(self, Self::ProbeAnti { null_aware: true })
    }

    pub(super) fn allows_dynamic_filter_pushdown(self) -> bool {
        matches!(
            self,
            Self::Inner
                | Self::BuildOuter
                | Self::ProbeSemi
                | Self::BuildSemi
                | Self::BuildAnti
        )
    }
}

#[derive(Debug)]
pub(super) struct CompiledJoinFilter {
    pub(super) expr: Arc<dyn PhysicalExpr>,
    pub(super) schema: SchemaRef,
    pub(super) columns: Vec<JoinOutputCol>,
}

impl CompiledJoinFilter {
    fn evaluate(
        &self,
        build_payload: &RecordBatch,
        probe_batch: &RecordBatch,
        build_idx: &UInt32Array,
        probe_idx: &UInt32Array,
    ) -> Result<BooleanArray> {
        let len = build_idx.len();
        let batch = if self.columns.is_empty() {
            RecordBatch::try_new_with_options(
                Arc::clone(&self.schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?
        } else {
            let mut cols = Vec::with_capacity(self.columns.len());
            for col in &self.columns {
                let taken = match col {
                    JoinOutputCol::Build(idx) => {
                        take(build_payload.column(*idx).as_ref(), build_idx, None)?
                    }
                    JoinOutputCol::Probe(idx) => {
                        take(probe_batch.column(*idx).as_ref(), probe_idx, None)?
                    }
                };
                cols.push(taken);
            }
            RecordBatch::try_new(Arc::clone(&self.schema), cols)?
        };
        let arr = self.expr.evaluate(&batch)?.into_array(len)?;
        arr.as_any()
            .downcast_ref::<BooleanArray>()
            .cloned()
            .ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "Morsel join filter predicate did not evaluate to BooleanArray"
                )
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum JoinOutputCol {
    Build(usize),
    Probe(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum KeyKind {
    Cross,
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
    pub(super) fn from_types(types: &[DataType], null_equality: NullEquality) -> Self {
        if types.is_empty() {
            return Self::Cross;
        }
        if null_equality == NullEquality::NullEqualsNull {
            return Self::Rows;
        }
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

pub(super) fn valid_key_mask(keys: &[ArrayRef]) -> Option<BooleanBuffer> {
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
    if key_kind == KeyKind::Cross {
        return Ok(Vec::new());
    }
    let mut hashes = vec![0u64; num_rows];
    match key_kind {
        KeyKind::Cross => unreachable!(),
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
    valid_mask: Option<&BooleanBuffer>,
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
    let mask_bytes = valid_mask.map_or(0, |b| b.inner().capacity());
    payload_bytes
        .saturating_add(keys_bytes)
        .saturating_add(hashes_bytes)
        .saturating_add(mask_bytes)
}

pub(super) struct HashJoinBuildConfig {
    pub(super) key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(super) key_names: Vec<String>,
    pub(super) key_types: Vec<DataType>,
    pub(super) null_equality: NullEquality,
    pub(super) keep_null_key_rows: bool,
    pub(super) track_null_keys: bool,
    pub(super) needs_matched_bitset: bool,
    pub(super) payload_columns: Vec<usize>,
    pub(super) payload_schema: SchemaRef,
    pub(super) dynamic_filter: Option<DynamicJoinFilterConfig>,
}

#[derive(Debug)]
pub(super) struct HashJoinBuildSink {
    key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    key_names: Vec<String>,
    key_types: Vec<DataType>,
    key_kind: KeyKind,
    null_equality: NullEquality,
    keep_null_key_rows: bool,
    track_null_keys: bool,
    needs_matched_bitset: bool,
    has_floats: bool,
    payload_columns: Vec<usize>,
    payload_schema: SchemaRef,
    dynamic_filter: Option<DynamicJoinFilterConfig>,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
}

impl HashJoinBuildSink {
    pub(super) fn try_new(
        config: HashJoinBuildConfig,
    ) -> Result<std::result::Result<Self, String>> {
        let key_kind = KeyKind::from_types(&config.key_types, config.null_equality);
        if key_kind == KeyKind::Rows {
            let fields = config
                .key_types
                .iter()
                .map(|dt| SortField::new(dt.clone()))
                .collect();
            if let Err(err) = RowConverter::new(fields) {
                return Ok(Err(format!("unsupported join key type: {err}")));
            }
        }
        let has_floats = has_float_type(&config.key_types);
        Ok(Ok(Self {
            key_exprs: config.key_exprs,
            key_names: config.key_names,
            key_types: config.key_types,
            key_kind,
            null_equality: config.null_equality,
            keep_null_key_rows: config.keep_null_key_rows,
            track_null_keys: config.track_null_keys,
            needs_matched_bitset: config.needs_matched_bitset,
            has_floats,
            payload_columns: config.payload_columns,
            payload_schema: config.payload_schema,
            dynamic_filter: config.dynamic_filter,
            #[cfg(test)]
            finalize_gate: None,
        }))
    }

    pub(super) fn new_state(&self, reservation: MemoryReservation) -> HashJoinBuildState {
        let key_summary = self
            .dynamic_filter
            .map(|_| LocalJoinKeySummary::new(self.key_types.len(), self.has_floats));
        HashJoinBuildState {
            key_exprs: self.key_exprs.clone(),
            key_kind: self.key_kind,
            null_equality: self.null_equality,
            keep_null_key_rows: self.keep_null_key_rows,
            track_null_keys: self.track_null_keys,
            has_floats: self.has_floats,
            payload_columns: self.payload_columns.clone(),
            batches: Vec::new(),
            key_batches: Vec::new(),
            hash_chunks: Vec::new(),
            valid_masks: Vec::new(),
            key_summary,
            num_rows: 0,
            has_null_key: false,
            reservation,
        }
    }

    #[cfg(feature = "parquet")]
    pub(super) fn payload_columns(&self) -> &[usize] {
        &self.payload_columns
    }

    #[cfg(feature = "parquet")]
    pub(super) fn enable_dynamic_filter(&mut self, cfg: DynamicJoinFilterConfig) {
        match &mut self.dynamic_filter {
            Some(existing) => {
                existing.build_bloom |= cfg.build_bloom;
            }
            None => {
                self.dynamic_filter = Some(cfg);
            }
        }
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(self.key_exprs.iter(), f)
    }

    pub(super) fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.key_kind == KeyKind::Cross {
            write!(f, "CrossJoinBuildSink")
        } else {
            write!(f, "HashJoinBuildSink[keys=[{}]]", self.key_names.join(", "))
        }
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
            valid_masks,
            key_summary,
            num_rows,
            has_null_key,
            reservation,
            ..
        } = combined;

        if num_rows == 0 {
            reservation.resize(0);
            let dynamic_filter = self.dynamic_filter.map(|_| {
                Arc::new(DynamicJoinFilterData::empty(
                    self.key_kind,
                    self.null_equality,
                    self.has_floats,
                    self.key_types.len(),
                ))
            });
            return Ok(Some(Arc::new(FinalizedHashTable::empty(
                Arc::clone(&self.payload_schema),
                has_null_key,
                dynamic_filter,
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

        let payload_concat_bytes =
            if self.payload_columns.is_empty() || batches.len() <= 1 {
                0
            } else {
                batches
                    .iter()
                    .map(get_record_batch_memory_size)
                    .fold(0usize, usize::saturating_add)
            };
        let matched_words = if self.needs_matched_bitset {
            num_rows.div_ceil(64)
        } else {
            0
        };
        let matched_bytes = matched_words.saturating_mul(size_of::<AtomicU64>());

        // Cross join: materialize payload only, no hash table.
        if self.key_kind == KeyKind::Cross {
            reservation.try_grow(payload_concat_bytes.saturating_add(matched_bytes))?;
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
            let final_bytes = get_record_batch_memory_size(&payload);
            reservation.try_resize(final_bytes)?;
            return Ok(Some(Arc::new(FinalizedHashTable {
                num_rows,
                hash_entries: 0,
                has_null_key: false,
                payload,
                keys: BuildKeys::Cross,
                buckets: Vec::new(),
                mask: 0,
                next: Vec::new(),
                matched: Vec::new(),
                dynamic_filter: None,
                _reservation: reservation,
            })));
        }

        let hash_entries: usize = if self.keep_null_key_rows && !valid_masks.is_empty() {
            hash_chunks
                .iter()
                .zip(valid_masks.iter())
                .map(|(chunk, mask_opt)| match mask_opt {
                    Some(bits) => bits.count_set_bits(),
                    None => chunk.len(),
                })
                .sum()
        } else {
            num_rows
        };

        // All build rows had NULL join keys (retained for BuildOuter / FullOuter / BuildAnti).
        if hash_entries == 0 {
            reservation.try_grow(payload_concat_bytes.saturating_add(matched_bytes))?;
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
            drop(key_batches);
            drop(hash_chunks);
            drop(valid_masks);
            let matched: Vec<AtomicU64> =
                (0..matched_words).map(|_| AtomicU64::new(0)).collect();
            let final_bytes =
                get_record_batch_memory_size(&payload).saturating_add(matched_bytes);
            reservation.try_resize(final_bytes)?;
            let dynamic_filter = self.dynamic_filter.map(|_| {
                Arc::new(DynamicJoinFilterData::empty(
                    self.key_kind,
                    self.null_equality,
                    self.has_floats,
                    self.key_types.len(),
                ))
            });
            return Ok(Some(Arc::new(FinalizedHashTable {
                num_rows,
                hash_entries: 0,
                has_null_key,
                payload,
                keys: BuildKeys::Empty,
                buckets: Vec::new(),
                mask: 0,
                next: Vec::new(),
                matched,
                dynamic_filter,
                _reservation: reservation,
            })));
        }

        let bucket_count = hash_entries.saturating_mul(2).next_power_of_two().max(16);
        let is_flat_int = matches!(
            self.key_kind,
            KeyKind::Int32 | KeyKind::Int64 | KeyKind::UInt32 | KeyKind::UInt64
        );
        let (dir_bytes, keys_peak_bytes) = match self.key_kind {
            KeyKind::Cross => unreachable!(),
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
            .saturating_add(keys_peak_bytes)
            .saturating_add(matched_bytes);
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
        let (build_keys, buckets, next, dynamic_filter) = if is_flat_int {
            let Some((keys, distinct_count)) = self
                .build_flat_int_keys(
                    &mut key_batches,
                    &hash_chunks,
                    &valid_masks,
                    num_rows,
                    bucket_count,
                    tx,
                )
                .await?
            else {
                return Ok(None);
            };
            drop(key_batches);
            let dynamic_filter = match (self.dynamic_filter, key_summary) {
                (Some(cfg), Some(summary)) => {
                    let membership = Self::build_flat_int_membership(
                        &keys,
                        &hash_chunks,
                        &valid_masks,
                        distinct_count,
                        cfg,
                        &reservation,
                    );
                    Some(Arc::new(DynamicJoinFilterData {
                        key_kind: self.key_kind,
                        null_equality: self.null_equality,
                        has_floats: self.has_floats,
                        bounds: summary.into_bounds(),
                        membership,
                    }))
                }
                _ => None,
            };
            drop(hash_chunks);
            drop(valid_masks);
            (keys, Vec::new(), Vec::new(), dynamic_filter)
        } else {
            let small_inlist = if let Some(cfg) = self.dynamic_filter
                && self.key_types.len() == 1
                && !self.has_floats
                && hash_entries > 1
                && hash_entries <= cfg.max_inlist_values
            {
                try_build_small_inlist_from_chunks(
                    &key_batches,
                    &valid_masks,
                    self.null_equality,
                    cfg.max_inlist_values,
                    cfg.max_inlist_bytes,
                )?
            } else {
                None
            };
            let build_keys =
                self.build_chained_keys(&mut key_batches, &hash_chunks, num_rows)?;
            drop(key_batches);

            let mut buckets = vec![0u32; bucket_count];
            let mut next = vec![0u32; num_rows];
            let mut row_idx = 0usize;
            let mut since_yield = 0usize;
            for (chunk_idx, chunk) in hash_chunks.iter().enumerate() {
                let valid_opt = valid_masks.get(chunk_idx).and_then(Option::as_ref);
                for (i, &h) in chunk.iter().enumerate() {
                    if valid_opt.is_none_or(|bits| bits.value(i)) {
                        let bucket = (h as usize) & mask;
                        let entry = (row_idx + 1) as u32;
                        next[row_idx] = buckets[bucket];
                        buckets[bucket] = entry;
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
            let dynamic_filter = match (self.dynamic_filter, key_summary) {
                (Some(cfg), Some(summary)) => {
                    let membership = if let Some(arr) = small_inlist {
                        let bytes = arr.get_array_memory_size();
                        if reservation.try_grow(bytes).is_ok() {
                            JoinMembershipFilter::InList(arr)
                        } else {
                            JoinMembershipFilter::None
                        }
                    } else if cfg.build_bloom
                        && (hash_entries > 1
                            || self.key_types.len() > 1
                            || self.has_floats)
                    {
                        Self::build_bloom_membership(
                            &hash_chunks,
                            &valid_masks,
                            hash_entries,
                            &reservation,
                        )
                    } else {
                        JoinMembershipFilter::None
                    };
                    Some(Arc::new(DynamicJoinFilterData {
                        key_kind: self.key_kind,
                        null_equality: self.null_equality,
                        has_floats: self.has_floats,
                        bounds: summary.into_bounds(),
                        membership,
                    }))
                }
                _ => None,
            };
            drop(hash_chunks);
            drop(valid_masks);
            (build_keys, buckets, next, dynamic_filter)
        };

        let matched: Vec<AtomicU64> =
            (0..matched_words).map(|_| AtomicU64::new(0)).collect();
        let filter_bytes = dynamic_filter.as_ref().map_or(0, |f| f.memory_bytes());
        let final_bytes = get_record_batch_memory_size(&payload)
            .saturating_add(build_keys.memory_bytes())
            .saturating_add(dir_bytes)
            .saturating_add(matched_bytes)
            .saturating_add(filter_bytes);
        reservation.try_resize(final_bytes)?;

        Ok(Some(Arc::new(FinalizedHashTable {
            num_rows,
            hash_entries,
            has_null_key,
            payload,
            keys: build_keys,
            buckets,
            mask,
            next,
            matched,
            dynamic_filter,
            _reservation: reservation,
        })))
    }

    fn build_flat_int_membership(
        keys: &BuildKeys,
        hash_chunks: &[Vec<u64>],
        valid_masks: &[Option<BooleanBuffer>],
        distinct_count: usize,
        cfg: DynamicJoinFilterConfig,
        reservation: &MemoryReservation,
    ) -> JoinMembershipFilter {
        if distinct_count <= 1 {
            return JoinMembershipFilter::None;
        }
        macro_rules! try_int_inlist {
            ($table:expr, $Arr:ty, $K:ty) => {{
                let inlist_bytes = distinct_count.saturating_mul(size_of::<$K>());
                if distinct_count <= cfg.max_inlist_values
                    && inlist_bytes <= cfg.max_inlist_bytes
                {
                    let mut vals: Vec<$K> = Vec::with_capacity(distinct_count);
                    for slot in &$table.slots {
                        if slot.first_row != 0 {
                            vals.push(slot.key);
                        }
                    }
                    vals.sort_unstable();
                    let arr: ArrayRef = Arc::new(<$Arr>::from(vals));
                    let arr_bytes = arr.get_array_memory_size();
                    if reservation.try_grow(arr_bytes).is_ok() {
                        return JoinMembershipFilter::InList(arr);
                    }
                    return JoinMembershipFilter::None;
                }
            }};
        }
        match keys {
            BuildKeys::Int32(t) => try_int_inlist!(t, Int32Array, i32),
            BuildKeys::Int64(t) => try_int_inlist!(t, Int64Array, i64),
            BuildKeys::UInt32(t) => try_int_inlist!(t, UInt32Array, u32),
            BuildKeys::UInt64(t) => try_int_inlist!(t, UInt64Array, u64),
            _ => return JoinMembershipFilter::None,
        }
        if !cfg.build_bloom {
            return JoinMembershipFilter::None;
        }
        Self::build_bloom_membership(
            hash_chunks,
            valid_masks,
            distinct_count,
            reservation,
        )
    }

    fn build_bloom_membership(
        hash_chunks: &[Vec<u64>],
        valid_masks: &[Option<BooleanBuffer>],
        distinct_entries: usize,
        reservation: &MemoryReservation,
    ) -> JoinMembershipFilter {
        let Some(num_blocks) =
            SplitBlockBloomFilter::num_blocks_for_entries(distinct_entries)
        else {
            return JoinMembershipFilter::None;
        };
        let est_bytes = SplitBlockBloomFilter::estimated_bytes(num_blocks);
        if reservation.try_grow(est_bytes).is_err() {
            return JoinMembershipFilter::None;
        }
        let mut bloom = SplitBlockBloomFilter::with_blocks(num_blocks);
        for (chunk_idx, chunk) in hash_chunks.iter().enumerate() {
            let valid_opt = valid_masks.get(chunk_idx).and_then(Option::as_ref);
            for (i, &h) in chunk.iter().enumerate() {
                if valid_opt.is_none_or(|bits| bits.value(i)) {
                    bloom.insert(h);
                }
            }
        }
        JoinMembershipFilter::Bloom(Arc::new(bloom))
    }

    async fn build_flat_int_keys(
        &self,
        key_batches: &mut Vec<Vec<ArrayRef>>,
        hash_chunks: &[Vec<u64>],
        valid_masks: &[Option<BooleanBuffer>],
        num_rows: usize,
        slot_count: usize,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<(BuildKeys, usize)>> {
        macro_rules! build_flat {
            ($Arr:ty, $K:ty, $Variant:ident) => {{
                let mask = slot_count - 1;
                let mut slots = vec![IntSlot::<$K>::empty(); slot_count];
                let mut dup_next: Vec<u32> = Vec::new();
                let mut distinct_count = 0usize;
                let mut row_idx = 0usize;
                let mut since_yield = 0usize;
                for (chunk_idx, (kb, hashes)) in
                    key_batches.drain(..).zip(hash_chunks.iter()).enumerate()
                {
                    let arr = kb[0].as_any().downcast_ref::<$Arr>().ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected integer join key array"
                        )
                    })?;
                    let valid_opt = valid_masks.get(chunk_idx).and_then(Option::as_ref);
                    for (i, (&key, &h)) in
                        arr.values().iter().zip(hashes.iter()).enumerate()
                    {
                        if valid_opt.is_none_or(|bits| bits.value(i)) {
                            let mut pos = (h as usize) & mask;
                            loop {
                                let slot = &mut slots[pos];
                                if slot.first_row == 0 {
                                    slot.key = key;
                                    slot.first_row = (row_idx + 1) as u32;
                                    distinct_count += 1;
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
                Ok(Some((
                    BuildKeys::$Variant(FlatIntTable {
                        slots,
                        mask,
                        dup_next,
                    }),
                    distinct_count,
                )))
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
            KeyKind::Cross
            | KeyKind::Int32
            | KeyKind::Int64
            | KeyKind::UInt32
            | KeyKind::UInt64 => {
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
    null_equality: NullEquality,
    keep_null_key_rows: bool,
    track_null_keys: bool,
    has_floats: bool,
    payload_columns: Vec<usize>,
    batches: Vec<RecordBatch>,
    key_batches: Vec<Vec<ArrayRef>>,
    hash_chunks: Vec<Vec<u64>>,
    valid_masks: Vec<Option<BooleanBuffer>>,
    key_summary: Option<LocalJoinKeySummary>,
    num_rows: usize,
    has_null_key: bool,
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
        if self.key_kind == KeyKind::Cross {
            let payload_batch = if self.payload_columns.is_empty() {
                None
            } else {
                Some(view.materialize_columns(&self.payload_columns)?)
            };
            let delta = payload_batch
                .as_ref()
                .map_or(0, get_record_batch_memory_size);
            self.reservation.try_grow(delta)?;
            if let Some(payload) = payload_batch {
                self.batches.push(payload);
            }
            self.num_rows = self.num_rows.checked_add(selected).ok_or_else(|| {
                datafusion_common::exec_datafusion_err!(
                    "Morsel CrossJoin build row count overflow"
                )
            })?;
            return Ok(());
        }

        let key_needed_cols = collect_expr_column_indices(&self.key_exprs);
        let compact_key_input = view.materialize_columns(&key_needed_cols)?;
        let mut keys = Vec::with_capacity(self.key_exprs.len());
        for expr in &self.key_exprs {
            let rebound = remap_expr_columns(Arc::clone(expr), &key_needed_cols)?;
            keys.push(rebound.evaluate(&compact_key_input)?.into_array(selected)?);
        }

        let valid_mask = if self.null_equality == NullEquality::NullEqualsNull {
            None
        } else {
            valid_key_mask(&keys)
        };
        if self.track_null_keys && valid_mask.is_some() {
            self.has_null_key = true;
        }

        if self.keep_null_key_rows {
            let payload_batch = if self.payload_columns.is_empty() {
                None
            } else {
                Some(view.materialize_columns(&self.payload_columns)?)
            };
            let normalized_keys = normalize_key_arrays(keys, self.has_floats);
            let hashes = compute_build_hashes(self.key_kind, &normalized_keys, selected)?;
            let delta = chunk_memory_bytes(
                payload_batch.as_ref(),
                &normalized_keys,
                hashes.capacity(),
                valid_mask.as_ref(),
            );
            self.reservation.try_grow(delta)?;
            if let Some(summary) = &mut self.key_summary {
                summary.update(
                    &normalized_keys,
                    valid_mask.as_ref(),
                    self.null_equality,
                )?;
            }
            if let Some(payload) = payload_batch {
                self.batches.push(payload);
            }
            self.key_batches.push(normalized_keys);
            self.hash_chunks.push(hashes);
            self.valid_masks.push(valid_mask);
            self.num_rows = self.num_rows.checked_add(selected).ok_or_else(|| {
                datafusion_common::exec_datafusion_err!(
                    "Morsel HashJoin build row count overflow"
                )
            })?;
            return Ok(());
        }

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
            None,
        );
        self.reservation.try_grow(delta)?;
        if let Some(summary) = &mut self.key_summary {
            summary.update(&normalized_keys, None, self.null_equality)?;
        }

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
        if self.key_kind == KeyKind::Cross {
            let payload_batch = if self.payload_columns.is_empty() {
                None
            } else {
                Some(batch.project(&self.payload_columns)?)
            };
            let delta = payload_batch
                .as_ref()
                .map_or(0, get_record_batch_memory_size);
            self.reservation.try_grow(delta)?;
            if let Some(payload) = payload_batch {
                self.batches.push(payload);
            }
            self.num_rows =
                self.num_rows.checked_add(batch.num_rows()).ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!(
                        "Morsel CrossJoin build row count overflow"
                    )
                })?;
            return Ok(());
        }

        let mut keys = Vec::with_capacity(self.key_exprs.len());
        for expr in &self.key_exprs {
            keys.push(expr.evaluate(batch)?.into_array(batch.num_rows())?);
        }

        let valid_mask = if self.null_equality == NullEquality::NullEqualsNull {
            None
        } else {
            valid_key_mask(&keys)
        };
        if self.track_null_keys && valid_mask.is_some() {
            self.has_null_key = true;
        }

        if self.keep_null_key_rows {
            let payload_batch = if self.payload_columns.is_empty() {
                None
            } else {
                Some(batch.project(&self.payload_columns)?)
            };
            let normalized_keys = normalize_key_arrays(keys, self.has_floats);
            let hashes =
                compute_build_hashes(self.key_kind, &normalized_keys, batch.num_rows())?;
            let delta = chunk_memory_bytes(
                payload_batch.as_ref(),
                &normalized_keys,
                hashes.capacity(),
                valid_mask.as_ref(),
            );
            self.reservation.try_grow(delta)?;
            if let Some(summary) = &mut self.key_summary {
                summary.update(
                    &normalized_keys,
                    valid_mask.as_ref(),
                    self.null_equality,
                )?;
            }
            if let Some(payload) = payload_batch {
                self.batches.push(payload);
            }
            self.key_batches.push(normalized_keys);
            self.hash_chunks.push(hashes);
            self.valid_masks.push(valid_mask);
            self.num_rows =
                self.num_rows.checked_add(batch.num_rows()).ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!(
                        "Morsel HashJoin build row count overflow"
                    )
                })?;
            return Ok(());
        }

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
            None,
        );
        self.reservation.try_grow(delta)?;
        if let Some(summary) = &mut self.key_summary {
            summary.update(&normalized_keys, None, self.null_equality)?;
        }

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
        let combined_null = self.has_null_key | other.has_null_key;
        self.has_null_key = combined_null;
        if other.num_rows == 0 {
            return;
        }
        if self.num_rows == 0 {
            other.has_null_key = combined_null;
            std::mem::swap(self, &mut other);
            return;
        }
        if let (Some(mine), Some(theirs)) = (&mut self.key_summary, &other.key_summary) {
            mine.merge(theirs);
        }
        self.batches.append(&mut other.batches);
        self.key_batches.append(&mut other.key_batches);
        self.hash_chunks.append(&mut other.hash_chunks);
        self.valid_masks.append(&mut other.valid_masks);
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
    Cross,
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
            Self::Empty | Self::Cross => 0,
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
    hash_entries: usize,
    has_null_key: bool,
    payload: RecordBatch,
    keys: BuildKeys,
    buckets: Vec<u32>,
    mask: usize,
    next: Vec<u32>,
    matched: Vec<AtomicU64>,
    dynamic_filter: Option<Arc<DynamicJoinFilterData>>,
    _reservation: MemoryReservation,
}

impl FinalizedHashTable {
    fn empty(
        payload_schema: SchemaRef,
        has_null_key: bool,
        dynamic_filter: Option<Arc<DynamicJoinFilterData>>,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            num_rows: 0,
            hash_entries: 0,
            has_null_key,
            payload: RecordBatch::new_empty(payload_schema),
            keys: BuildKeys::Empty,
            buckets: Vec::new(),
            mask: 0,
            next: Vec::new(),
            matched: Vec::new(),
            dynamic_filter,
            _reservation: reservation,
        }
    }

    #[cfg(any(feature = "parquet", test))]
    pub(super) fn dynamic_filter(&self) -> Option<&Arc<DynamicJoinFilterData>> {
        self.dynamic_filter.as_ref()
    }

    pub(super) fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub(super) fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    pub(super) fn has_hash_entries(&self) -> bool {
        self.hash_entries > 0
    }

    pub(super) fn has_null_key(&self) -> bool {
        self.has_null_key
    }

    #[inline]
    fn mark_matched(&self, row: u32) {
        let idx = row as usize;
        let word = &self.matched[idx >> 6];
        let bit = 1u64 << (idx & 63);
        if (word.load(Ordering::Relaxed) & bit) == 0 {
            word.fetch_or(bit, Ordering::Relaxed);
        }
    }

    #[inline]
    fn is_matched(&self, row: u32) -> bool {
        let idx = row as usize;
        (self.matched[idx >> 6].load(Ordering::Acquire) & (1u64 << (idx & 63))) != 0
    }

    pub(super) fn collect_build_emission_indices(
        &self,
        start: usize,
        end: usize,
        want_matched: bool,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        let end = end.min(self.num_rows);
        if start >= end {
            return;
        }
        let first_word = start >> 6;
        let last_word = (end - 1) >> 6;
        for word_idx in first_word..=last_word {
            let mut bits = self.matched[word_idx].load(Ordering::Acquire);
            if !want_matched {
                bits = !bits;
            }
            let word_base = word_idx << 6;
            if word_idx == first_word {
                let low_skip = start - word_base;
                bits &= !0u64 << low_skip;
            }
            if word_idx == last_word {
                let high_keep = end - word_base;
                if high_keep < 64 {
                    bits &= (1u64 << high_keep) - 1;
                }
            }
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                out.push((word_base + tz) as u32);
                bits &= bits - 1;
            }
        }
    }
}

pub(super) enum OperatorStep {
    NeedMoreInput(Option<RecordBatch>),
    HaveMoreOutput(RecordBatch),
}

#[derive(Clone, Copy)]
struct CandGroup {
    backing_row: u32,
    cand_start: usize,
    cand_end: usize,
    chain_done: bool,
    continued_from_prev: bool,
}

#[derive(Default)]
pub(super) struct ProbeScratch {
    build_indices: Vec<u32>,
    probe_indices: Vec<u32>,
    cand_build: Vec<u32>,
    cand_probe: Vec<u32>,
    cand_groups: Vec<CandGroup>,
}

enum ProbeKeys {
    Cross,
    EmptyBuild,
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
    row_matched: bool,
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
    pub(super) kind: ProbeJoinKind,
    pub(super) join_type: JoinType,
    null_equality: NullEquality,
    probe_key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    probe_key_names: Vec<String>,
    plain_single_key_col: Option<usize>,
    key_kind: KeyKind,
    has_floats: bool,
    pub(super) filter: Option<CompiledJoinFilter>,
    pub(super) output_projection: Vec<JoinOutputCol>,
    pub(super) output_schema: SchemaRef,
    pub(super) build_id: usize,
    pub(super) filter_id: usize,
    #[cfg(test)]
    pub(super) build_emit_gate: Option<super::tests::FinalizeGate>,
}

#[derive(Debug)]
pub(super) struct HashJoinProbeConfig {
    pub(super) build_pipeline: Arc<Pipeline>,
    pub(super) build_side: BuildSide,
    pub(super) kind: ProbeJoinKind,
    pub(super) join_type: JoinType,
    pub(super) null_equality: NullEquality,
    pub(super) probe_key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(super) probe_key_names: Vec<String>,
    pub(super) key_types: Vec<DataType>,
    pub(super) filter: Option<CompiledJoinFilter>,
    pub(super) output_projection: Vec<JoinOutputCol>,
    pub(super) output_schema: SchemaRef,
    pub(super) build_id: usize,
}

impl HashJoinProbe {
    pub(super) fn new(config: HashJoinProbeConfig) -> Self {
        let key_kind = KeyKind::from_types(&config.key_types, config.null_equality);
        let has_floats = has_float_type(&config.key_types);
        let plain_single_key_col = match config.probe_key_exprs.as_slice() {
            [expr] => expr.downcast_ref::<Column>().map(Column::index),
            _ => None,
        };
        Self {
            build_pipeline: config.build_pipeline,
            build_side: config.build_side,
            kind: config.kind,
            join_type: config.join_type,
            null_equality: config.null_equality,
            probe_key_exprs: config.probe_key_exprs,
            probe_key_names: config.probe_key_names,
            plain_single_key_col,
            key_kind,
            has_floats,
            filter: config.filter,
            output_projection: config.output_projection,
            output_schema: config.output_schema,
            build_id: config.build_id,
            filter_id: 0,
            #[cfg(test)]
            build_emit_gate: None,
        }
    }

    pub(super) fn needs_build_emission(&self) -> bool {
        self.kind.needs_build_emission()
    }

    #[cfg(any(feature = "parquet", test))]
    pub(super) fn probe_key_exprs(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.probe_key_exprs
    }

    pub(super) fn empty_build_short_circuits(&self) -> bool {
        self.kind.empty_build_short_circuits()
    }

    pub(super) fn short_circuits_on_table(&self, table: &FinalizedHashTable) -> bool {
        (table.is_empty() && self.kind.empty_build_short_circuits())
            || (self.kind.is_null_aware_anti() && table.has_null_key())
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
        if let Some(filter) = &self.filter
            && apply_expression_roots(std::iter::once(&filter.expr), f)?
                == TreeNodeRecursion::Stop
        {
            return Ok(TreeNodeRecursion::Stop);
        }
        self.build_pipeline.apply_expressions(f)
    }

    pub(super) fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.kind == ProbeJoinKind::Cross {
            write!(f, "CrossJoinProbe[build={:?}", self.build_side)?;
        } else {
            write!(
                f,
                "HashJoinProbe[type={:?}, keys=[{}], build={:?}",
                self.join_type,
                self.probe_key_names.join(", "),
                self.build_side
            )?;
            if self.kind.is_null_aware_anti() {
                write!(f, ", null_aware")?;
            }
            if self.null_equality == NullEquality::NullEqualsNull {
                write!(f, ", null_equals_null")?;
            }
        }
        if let Some(filter) = &self.filter {
            write!(f, ", filter={}", filter.expr)?;
        }
        write!(f, "]")
    }

    pub(super) fn materialize_build_emission_batch(
        &self,
        table: &FinalizedHashTable,
        build_indices: &[u32],
    ) -> Result<RecordBatch> {
        let len = build_indices.len();
        if self.output_projection.is_empty() {
            return Ok(RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?);
        }
        let idx_arr = UInt32Array::from(build_indices.to_vec());
        let mut columns = Vec::with_capacity(self.output_projection.len());
        for (out_idx, col) in self.output_projection.iter().enumerate() {
            let arr = match col {
                JoinOutputCol::Build(b_idx) => {
                    take(table.payload.column(*b_idx).as_ref(), &idx_arr, None)?
                }
                JoinOutputCol::Probe(_) => {
                    new_null_array(self.output_schema.field(out_idx).data_type(), len)
                }
            };
            columns.push(arr);
        }
        Ok(RecordBatch::try_new(
            Arc::clone(&self.output_schema),
            columns,
        )?)
    }

    fn start_unkeyed_view(
        view: BatchView,
        keys: ProbeKeys,
        cursor_reservation: MemoryReservation,
    ) -> ProbeCursor {
        let (batch, selection) = view.into_parts();
        let num_rows = batch.num_rows();
        let rows = match selection {
            Selection::All => ProbeRows::All { num_rows },
            Selection::Mask { bits, .. } => ProbeRows::Mask { bits, num_rows },
            Selection::Indices(indices) => ProbeRows::BackingIndices(indices),
        };
        ProbeCursor {
            batch,
            keys,
            valid_mask: None,
            rows,
            pos: 0,
            next_entry: 0,
            row_matched: false,
            _reservation: cursor_reservation,
        }
    }

    pub(super) fn start_probe_view(
        &self,
        view: BatchView,
        table: &FinalizedHashTable,
        probe_reservation: &MemoryReservation,
    ) -> Result<Option<ProbeCursor>> {
        if view.is_empty() || self.short_circuits_on_table(table) {
            return Ok(None);
        }

        // Apply the same retention accounting rule to unfiltered and filtered cursors:
        // charge only pipeline-allocated memory (the selection buffer or any newly
        // compacted copy), never preexisting backing storage.
        let mut cursor_reservation = probe_reservation.new_empty();
        let view = view.into_reserved_for_retention(&mut cursor_reservation)?;

        if self.kind == ProbeJoinKind::Cross {
            return Ok(Some(Self::start_unkeyed_view(
                view,
                ProbeKeys::Cross,
                cursor_reservation,
            )));
        }

        if !table.has_hash_entries() {
            return if self.kind.emits_unmatched_probe() {
                Ok(Some(Self::start_unkeyed_view(
                    view,
                    ProbeKeys::EmptyBuild,
                    cursor_reservation,
                )))
            } else {
                Ok(None)
            };
        }

        if view.is_all() {
            return self.start_probe_all(view.into_parts().0, table, cursor_reservation);
        }

        let keep_nulls = self.kind.keeps_null_probe_keys();
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
                    if key_arr.null_count() > 0 {
                        let nulls = key_arr.nulls().unwrap().inner();
                        if keep_nulls {
                            (ProbeRows::Mask { bits, num_rows }, Some(nulls.clone()))
                        } else {
                            let combined = &bits & nulls;
                            if combined.count_set_bits() == 0 {
                                return Ok(None);
                            }
                            (
                                ProbeRows::Mask {
                                    bits: combined,
                                    num_rows,
                                },
                                None,
                            )
                        }
                    } else {
                        (ProbeRows::Mask { bits, num_rows }, None)
                    }
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
                row_matched: false,
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
        let valid_mask = if self.null_equality == NullEquality::NullEqualsNull {
            None
        } else {
            valid_key_mask(&key_arrays)
        };
        if !keep_nulls
            && let Some(bits) = &valid_mask
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
            row_matched: false,
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
        if num_rows == 0 || !table.has_hash_entries() {
            return Ok(None);
        }
        let mut key_arrays = Vec::with_capacity(self.probe_key_exprs.len());
        for expr in &self.probe_key_exprs {
            key_arrays.push(expr.evaluate(&batch)?.into_array(num_rows)?);
        }
        let valid_mask = if self.null_equality == NullEquality::NullEqualsNull {
            None
        } else {
            valid_key_mask(&key_arrays)
        };
        if !self.kind.keeps_null_probe_keys()
            && let Some(bits) = &valid_mask
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
            row_matched: false,
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

        if matches!(cursor.keys, ProbeKeys::Cross) {
            return self.next_chunk_cross(cursor, table, limit, scratch);
        }
        if matches!(cursor.keys, ProbeKeys::EmptyBuild) {
            return self.next_chunk_empty_build(cursor, limit, scratch);
        }

        if let Some(filter) = &self.filter {
            return self.next_chunk_filtered(cursor, table, filter, limit, scratch);
        }

        match self.kind {
            ProbeJoinKind::Inner => {
                self.next_chunk_inner_no_filter(cursor, table, limit, scratch)
            }
            ProbeJoinKind::BuildOuter => {
                self.next_chunk_build_outer_no_filter(cursor, table, limit, scratch)
            }
            ProbeJoinKind::ProbeOuter | ProbeJoinKind::FullOuter => self
                .next_chunk_probe_or_full_outer_no_filter(cursor, table, limit, scratch),
            ProbeJoinKind::ProbeSemi | ProbeJoinKind::ProbeAnti { .. } => self
                .next_chunk_probe_semi_or_anti_no_filter(cursor, table, limit, scratch),
            ProbeJoinKind::BuildSemi | ProbeJoinKind::BuildAnti => {
                self.next_chunk_build_semi_or_anti_no_filter(cursor, table)
            }
            ProbeJoinKind::Cross => unreachable!(),
        }
    }

    fn next_chunk_inner_no_filter(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
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
        reserve_indices(&mut scratch.build_indices, limit)?;
        reserve_indices(&mut scratch.probe_indices, limit)?;

        probe_collect_matches(cursor, table, limit, scratch)?;

        self.finish_non_null_output(cursor, table, scratch)
    }

    fn next_chunk_build_outer_no_filter(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        scratch.build_indices.clear();
        scratch.probe_indices.clear();
        reserve_indices(&mut scratch.build_indices, limit)?;
        reserve_indices(&mut scratch.probe_indices, limit)?;

        probe_collect_matches(cursor, table, limit, scratch)?;
        for &b in &scratch.build_indices {
            table.mark_matched(b);
        }

        self.finish_non_null_output(cursor, table, scratch)
    }

    fn next_chunk_probe_or_full_outer_no_filter(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let mark_build = matches!(self.kind, ProbeJoinKind::FullOuter);
        scratch.build_indices.clear();
        scratch.probe_indices.clear();
        reserve_indices(&mut scratch.build_indices, limit)?;
        reserve_indices(&mut scratch.probe_indices, limit)?;
        let mut build_nulls = NullBufferBuilder::new(limit);

        while !cursor.is_exhausted() && scratch.probe_indices.len() < limit {
            let Some((key_idx, backing_row)) = probe_row_at(&cursor.rows, cursor.pos)
            else {
                cursor.pos += 1;
                continue;
            };
            let rem = limit - scratch.probe_indices.len();
            if cursor.next_entry == 0
                && cursor
                    .valid_mask
                    .as_ref()
                    .is_some_and(|v| !v.value(key_idx))
            {
                scratch.build_indices.push(0);
                build_nulls.append_null();
                scratch.probe_indices.push(backing_row);
                cursor.pos += 1;
                continue;
            }
            let continued = cursor.next_entry != 0;
            let start_len = scratch.build_indices.len();
            let chain_done = probe_step_candidates(
                cursor,
                table,
                key_idx,
                backing_row,
                rem,
                false,
                &mut scratch.build_indices,
                &mut scratch.probe_indices,
            )?;
            let added = scratch.build_indices.len() - start_len;
            if added > 0 {
                if mark_build {
                    for &b in &scratch.build_indices[start_len..] {
                        table.mark_matched(b);
                    }
                }
                build_nulls.append_n_non_nulls(added);
                if chain_done {
                    cursor.pos += 1;
                    cursor.next_entry = 0;
                    cursor.row_matched = false;
                } else {
                    cursor.row_matched = true;
                    break;
                }
            } else {
                if !continued || !cursor.row_matched {
                    scratch.build_indices.push(0);
                    build_nulls.append_null();
                    scratch.probe_indices.push(backing_row);
                }
                cursor.pos += 1;
                cursor.next_entry = 0;
                cursor.row_matched = false;
            }
        }

        let nulls = build_nulls.finish();
        self.finish_nullable_build_output(cursor, table, scratch, nulls)
    }

    fn next_chunk_probe_semi_or_anti_no_filter(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let is_semi = matches!(self.kind, ProbeJoinKind::ProbeSemi);
        let keep_null_probe = self.kind.keeps_null_probe_keys();
        scratch.probe_indices.clear();
        reserve_indices(&mut scratch.probe_indices, limit)?;

        while !cursor.is_exhausted() && scratch.probe_indices.len() < limit {
            let Some((key_idx, backing_row)) = probe_row_at(&cursor.rows, cursor.pos)
            else {
                cursor.pos += 1;
                continue;
            };
            cursor.pos += 1;
            if cursor
                .valid_mask
                .as_ref()
                .is_some_and(|v| !v.value(key_idx))
            {
                if keep_null_probe {
                    scratch.probe_indices.push(backing_row);
                }
                continue;
            }
            let matched = probe_has_any_match(&cursor.keys, table, key_idx)?;
            if matched == is_semi {
                scratch.probe_indices.push(backing_row);
            }
        }

        self.finish_probe_only_output(cursor, scratch)
    }

    fn next_chunk_build_semi_or_anti_no_filter(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
    ) -> Result<OperatorStep> {
        while !cursor.is_exhausted() {
            let Some((key_idx, _)) = probe_row_at(&cursor.rows, cursor.pos) else {
                cursor.pos += 1;
                continue;
            };
            cursor.pos += 1;
            if cursor
                .valid_mask
                .as_ref()
                .is_some_and(|v| !v.value(key_idx))
            {
                continue;
            }
            probe_mark_all_matches(&cursor.keys, table, key_idx)?;
        }
        Ok(OperatorStep::NeedMoreInput(None))
    }

    fn next_chunk_empty_build(
        &self,
        cursor: &mut ProbeCursor,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        scratch.probe_indices.clear();
        reserve_indices(&mut scratch.probe_indices, limit)?;

        while !cursor.is_exhausted() && scratch.probe_indices.len() < limit {
            let Some((_, backing_row)) = probe_row_at(&cursor.rows, cursor.pos) else {
                cursor.pos += 1;
                continue;
            };
            cursor.pos += 1;
            scratch.probe_indices.push(backing_row);
        }

        let len = scratch.probe_indices.len();
        if len == 0 {
            return Ok(OperatorStep::NeedMoreInput(None));
        }
        if self.output_projection.is_empty() {
            scratch.probe_indices.clear();
            let out = RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?;
            return Ok(if cursor.is_exhausted() {
                OperatorStep::NeedMoreInput(Some(out))
            } else {
                OperatorStep::HaveMoreOutput(out)
            });
        }

        let probe_idx = UInt32Array::new(
            ScalarBuffer::from(std::mem::take(&mut scratch.probe_indices)),
            None,
        );
        let mut columns = Vec::with_capacity(self.output_projection.len());
        for (out_idx, col) in self.output_projection.iter().enumerate() {
            let arr = match col {
                JoinOutputCol::Build(_) => {
                    new_null_array(self.output_schema.field(out_idx).data_type(), len)
                }
                JoinOutputCol::Probe(idx) => {
                    take(cursor.batch.column(*idx).as_ref(), &probe_idx, None)?
                }
            };
            columns.push(arr);
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

    fn next_chunk_cross(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let build_rows = table.num_rows;
        if build_rows == 0 {
            return Ok(OperatorStep::NeedMoreInput(None));
        }

        // Fast path for single-row build (e.g. decorrelated scalar subqueries) with no filter.
        if build_rows == 1
            && self.filter.is_none()
            && cursor.pos == 0
            && cursor.next_entry == 0
            && let ProbeRows::All { num_rows } = cursor.rows
            && num_rows <= limit
        {
            cursor.pos = num_rows;
            if self.output_projection.is_empty() {
                let out = RecordBatch::try_new_with_options(
                    Arc::clone(&self.output_schema),
                    vec![],
                    &RecordBatchOptions::new().with_row_count(Some(num_rows)),
                )?;
                return Ok(OperatorStep::NeedMoreInput(Some(out)));
            }
            let mut columns = Vec::with_capacity(self.output_projection.len());
            for col in &self.output_projection {
                let arr = match col {
                    JoinOutputCol::Probe(idx) => Arc::clone(cursor.batch.column(*idx)),
                    JoinOutputCol::Build(idx) => {
                        let scalar = ScalarValue::try_from_array(
                            table.payload.column(*idx).as_ref(),
                            0,
                        )?;
                        scalar.to_array_of_size(num_rows)?
                    }
                };
                columns.push(arr);
            }
            let out = RecordBatch::try_new(Arc::clone(&self.output_schema), columns)?;
            return Ok(OperatorStep::NeedMoreInput(Some(out)));
        }

        scratch.build_indices.clear();
        scratch.probe_indices.clear();
        reserve_indices(&mut scratch.build_indices, limit)?;
        reserve_indices(&mut scratch.probe_indices, limit)?;

        if let Some(filter) = &self.filter {
            while !cursor.is_exhausted() && scratch.build_indices.len() < limit {
                let budget = limit - scratch.build_indices.len();
                scratch.cand_build.clear();
                scratch.cand_probe.clear();
                reserve_indices(&mut scratch.cand_build, budget)?;
                reserve_indices(&mut scratch.cand_probe, budget)?;
                collect_cross_pairs(
                    cursor,
                    build_rows,
                    budget,
                    &mut scratch.cand_build,
                    &mut scratch.cand_probe,
                );
                if scratch.cand_build.is_empty() {
                    break;
                }
                let b_arr = UInt32Array::new(
                    ScalarBuffer::from(std::mem::take(&mut scratch.cand_build)),
                    None,
                );
                let p_arr = UInt32Array::new(
                    ScalarBuffer::from(std::mem::take(&mut scratch.cand_probe)),
                    None,
                );
                let bools =
                    filter.evaluate(&table.payload, &cursor.batch, &b_arr, &p_arr)?;
                let b_vals = b_arr.values();
                let p_vals = p_arr.values();
                for i in 0..bools.len() {
                    if bools.is_valid(i) && bools.value(i) {
                        scratch.build_indices.push(b_vals[i]);
                        scratch.probe_indices.push(p_vals[i]);
                    }
                }
                if let Ok(mut v) = b_arr.into_parts().1.into_inner().into_vec::<u32>() {
                    v.clear();
                    scratch.cand_build = v;
                }
                if let Ok(mut v) = p_arr.into_parts().1.into_inner().into_vec::<u32>() {
                    v.clear();
                    scratch.cand_probe = v;
                }
            }
        } else {
            collect_cross_pairs(
                cursor,
                build_rows,
                limit,
                &mut scratch.build_indices,
                &mut scratch.probe_indices,
            );
        }

        self.finish_non_null_output(cursor, table, scratch)
    }

    fn next_chunk_filtered(
        &self,
        cursor: &mut ProbeCursor,
        table: &FinalizedHashTable,
        filter: &CompiledJoinFilter,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let is_outer = matches!(
            self.kind,
            ProbeJoinKind::ProbeOuter | ProbeJoinKind::FullOuter
        );
        let is_probe_only = matches!(
            self.kind,
            ProbeJoinKind::ProbeSemi | ProbeJoinKind::ProbeAnti { .. }
        );
        let is_build_mark_only = matches!(
            self.kind,
            ProbeJoinKind::BuildSemi | ProbeJoinKind::BuildAnti
        );
        let emit_unmatched_probe = self.kind.emits_unmatched_probe();
        let keep_null_probe = self.kind.keeps_null_probe_keys();

        scratch.build_indices.clear();
        scratch.probe_indices.clear();
        if !is_build_mark_only {
            reserve_indices(&mut scratch.probe_indices, limit)?;
            if !is_probe_only {
                reserve_indices(&mut scratch.build_indices, limit)?;
            }
        }
        let mut build_nulls = is_outer.then(|| NullBufferBuilder::new(limit));

        while !cursor.is_exhausted() && scratch.probe_indices.len() < limit {
            let budget = if is_build_mark_only {
                limit
            } else {
                limit - scratch.probe_indices.len()
            };
            scratch.cand_build.clear();
            scratch.cand_probe.clear();
            scratch.cand_groups.clear();
            let mut budget_used = 0usize;

            while !cursor.is_exhausted() && budget_used < budget {
                let Some((key_idx, backing_row)) = probe_row_at(&cursor.rows, cursor.pos)
                else {
                    cursor.pos += 1;
                    continue;
                };

                if cursor.next_entry == 0
                    && cursor
                        .valid_mask
                        .as_ref()
                        .is_some_and(|v| !v.value(key_idx))
                {
                    cursor.pos += 1;
                    if keep_null_probe {
                        let c = scratch.cand_build.len();
                        scratch.cand_groups.push(CandGroup {
                            backing_row,
                            cand_start: c,
                            cand_end: c,
                            chain_done: true,
                            continued_from_prev: false,
                        });
                        budget_used += 1;
                    }
                    continue;
                }

                let continued = cursor.next_entry != 0;
                let cand_start = scratch.cand_build.len();
                let rem = budget - budget_used;
                let chain_done = probe_step_candidates(
                    cursor,
                    table,
                    key_idx,
                    backing_row,
                    rem,
                    is_build_mark_only,
                    &mut scratch.cand_build,
                    &mut scratch.cand_probe,
                )?;
                let cand_end = scratch.cand_build.len();
                let added = cand_end - cand_start;

                if added > 0 {
                    scratch.cand_groups.push(CandGroup {
                        backing_row,
                        cand_start,
                        cand_end,
                        chain_done,
                        continued_from_prev: continued,
                    });
                    budget_used += added;
                    if chain_done {
                        cursor.pos += 1;
                        cursor.next_entry = 0;
                    } else {
                        break;
                    }
                } else {
                    // `added == 0` implies `chain_done == true`.
                    cursor.pos += 1;
                    cursor.next_entry = 0;
                    if emit_unmatched_probe {
                        scratch.cand_groups.push(CandGroup {
                            backing_row,
                            cand_start,
                            cand_end,
                            chain_done: true,
                            continued_from_prev: continued,
                        });
                        budget_used += 1;
                    } else if continued {
                        cursor.row_matched = false;
                    }
                }
            }

            if scratch.cand_groups.is_empty() {
                continue;
            }

            let (b_arr, p_arr, filter_bools) = if !scratch.cand_build.is_empty() {
                let b_arr = UInt32Array::new(
                    ScalarBuffer::from(std::mem::take(&mut scratch.cand_build)),
                    None,
                );
                let p_arr = UInt32Array::new(
                    ScalarBuffer::from(std::mem::take(&mut scratch.cand_probe)),
                    None,
                );
                let bools =
                    filter.evaluate(&table.payload, &cursor.batch, &b_arr, &p_arr)?;
                (Some(b_arr), Some(p_arr), Some(bools))
            } else {
                (None, None, None)
            };

            let b_vals = b_arr.as_ref().map(|a| a.values().as_ref());
            let pass = |i: usize| -> bool {
                filter_bools
                    .as_ref()
                    .is_some_and(|ba| ba.is_valid(i) && ba.value(i))
            };

            for g in &scratch.cand_groups {
                let was_matched = if g.continued_from_prev {
                    cursor.row_matched
                } else {
                    false
                };
                match self.kind {
                    ProbeJoinKind::Inner => {
                        let b_slice = b_vals.unwrap();
                        for (i, &b) in b_slice
                            .iter()
                            .enumerate()
                            .take(g.cand_end)
                            .skip(g.cand_start)
                        {
                            if pass(i) {
                                scratch.build_indices.push(b);
                                scratch.probe_indices.push(g.backing_row);
                            }
                        }
                    }
                    ProbeJoinKind::BuildOuter => {
                        let b_slice = b_vals.unwrap();
                        for (i, &b) in b_slice
                            .iter()
                            .enumerate()
                            .take(g.cand_end)
                            .skip(g.cand_start)
                        {
                            if pass(i) {
                                table.mark_matched(b);
                                scratch.build_indices.push(b);
                                scratch.probe_indices.push(g.backing_row);
                            }
                        }
                    }
                    ProbeJoinKind::ProbeOuter | ProbeJoinKind::FullOuter => {
                        let mark_build = matches!(self.kind, ProbeJoinKind::FullOuter);
                        let nulls = build_nulls.as_mut().unwrap();
                        let mut matched_now = false;
                        if g.cand_start < g.cand_end {
                            let b_slice = b_vals.unwrap();
                            for (i, &b) in b_slice
                                .iter()
                                .enumerate()
                                .take(g.cand_end)
                                .skip(g.cand_start)
                            {
                                if pass(i) {
                                    if mark_build {
                                        table.mark_matched(b);
                                    }
                                    scratch.build_indices.push(b);
                                    nulls.append_non_null();
                                    scratch.probe_indices.push(g.backing_row);
                                    matched_now = true;
                                }
                            }
                        }
                        if g.chain_done {
                            if !was_matched && !matched_now {
                                scratch.build_indices.push(0);
                                nulls.append_null();
                                scratch.probe_indices.push(g.backing_row);
                            }
                            cursor.row_matched = false;
                        } else {
                            cursor.row_matched = was_matched || matched_now;
                        }
                    }
                    ProbeJoinKind::ProbeSemi => {
                        let mut matched_now = false;
                        for i in g.cand_start..g.cand_end {
                            if pass(i) {
                                matched_now = true;
                                break;
                            }
                        }
                        if matched_now && !was_matched {
                            scratch.probe_indices.push(g.backing_row);
                        }
                        if !g.chain_done && (was_matched || matched_now) {
                            // Fast-skip the rest of this probe row's hash chain.
                            cursor.next_entry = 0;
                            cursor.pos += 1;
                        }
                        cursor.row_matched = false;
                    }
                    ProbeJoinKind::ProbeAnti { .. } => {
                        let mut matched_now = false;
                        for i in g.cand_start..g.cand_end {
                            if pass(i) {
                                matched_now = true;
                                break;
                            }
                        }
                        if !g.chain_done {
                            if was_matched || matched_now {
                                // Fast-skip the rest of this probe row's hash chain.
                                cursor.next_entry = 0;
                                cursor.pos += 1;
                            }
                        } else if !was_matched && !matched_now {
                            scratch.probe_indices.push(g.backing_row);
                        }
                        cursor.row_matched = false;
                    }
                    ProbeJoinKind::BuildSemi | ProbeJoinKind::BuildAnti => {
                        let b_slice = b_vals.unwrap();
                        for (i, &b) in b_slice
                            .iter()
                            .enumerate()
                            .take(g.cand_end)
                            .skip(g.cand_start)
                        {
                            if pass(i) {
                                table.mark_matched(b);
                            }
                        }
                    }
                    ProbeJoinKind::Cross => unreachable!(),
                }
            }

            if let Some(arr) = b_arr
                && let Ok(mut v) = arr.into_parts().1.into_inner().into_vec::<u32>()
            {
                v.clear();
                scratch.cand_build = v;
            }
            if let Some(arr) = p_arr
                && let Ok(mut v) = arr.into_parts().1.into_inner().into_vec::<u32>()
            {
                v.clear();
                scratch.cand_probe = v;
            }
        }

        if is_build_mark_only {
            Ok(OperatorStep::NeedMoreInput(None))
        } else if is_probe_only {
            self.finish_probe_only_output(cursor, scratch)
        } else if let Some(mut nulls) = build_nulls {
            self.finish_nullable_build_output(cursor, table, scratch, nulls.finish())
        } else {
            self.finish_non_null_output(cursor, table, scratch)
        }
    }

    fn finish_non_null_output(
        &self,
        cursor: &ProbeCursor,
        table: &FinalizedHashTable,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let len = scratch.build_indices.len();
        if len == 0 {
            return Ok(OperatorStep::NeedMoreInput(None));
        }
        if self.output_projection.is_empty() {
            scratch.build_indices.clear();
            scratch.probe_indices.clear();
            let out = RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?;
            return Ok(if cursor.is_exhausted() {
                OperatorStep::NeedMoreInput(Some(out))
            } else {
                OperatorStep::HaveMoreOutput(out)
            });
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

    fn finish_nullable_build_output(
        &self,
        cursor: &ProbeCursor,
        table: &FinalizedHashTable,
        scratch: &mut ProbeScratch,
        build_nulls: Option<NullBuffer>,
    ) -> Result<OperatorStep> {
        let len = scratch.probe_indices.len();
        if len == 0 {
            scratch.build_indices.clear();
            return Ok(OperatorStep::NeedMoreInput(None));
        }
        if self.output_projection.is_empty() {
            scratch.build_indices.clear();
            scratch.probe_indices.clear();
            let out = RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?;
            return Ok(if cursor.is_exhausted() {
                OperatorStep::NeedMoreInput(Some(out))
            } else {
                OperatorStep::HaveMoreOutput(out)
            });
        }

        let all_build_null = build_nulls
            .as_ref()
            .is_some_and(|nb| nb.null_count() == len);
        let build_idx = if all_build_null {
            scratch.build_indices.clear();
            None
        } else {
            Some(UInt32Array::new(
                ScalarBuffer::from(std::mem::take(&mut scratch.build_indices)),
                build_nulls,
            ))
        };
        let probe_idx = UInt32Array::new(
            ScalarBuffer::from(std::mem::take(&mut scratch.probe_indices)),
            None,
        );

        let mut columns = Vec::with_capacity(self.output_projection.len());
        for (out_idx, col) in self.output_projection.iter().enumerate() {
            let taken = match col {
                JoinOutputCol::Build(idx) => match &build_idx {
                    Some(b_idx) => {
                        take(table.payload.column(*idx).as_ref(), b_idx, None)?
                    }
                    None => {
                        new_null_array(self.output_schema.field(out_idx).data_type(), len)
                    }
                },
                JoinOutputCol::Probe(idx) => {
                    take(cursor.batch.column(*idx).as_ref(), &probe_idx, None)?
                }
            };
            columns.push(taken);
        }

        if let Some(b_idx) = build_idx
            && let Ok(mut v) = b_idx.into_parts().1.into_inner().into_vec::<u32>()
        {
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

    fn finish_probe_only_output(
        &self,
        cursor: &ProbeCursor,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let len = scratch.probe_indices.len();
        if len == 0 {
            return Ok(OperatorStep::NeedMoreInput(None));
        }
        if self.output_projection.is_empty() {
            scratch.probe_indices.clear();
            let out = RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?;
            return Ok(if cursor.is_exhausted() {
                OperatorStep::NeedMoreInput(Some(out))
            } else {
                OperatorStep::HaveMoreOutput(out)
            });
        }

        let probe_idx = UInt32Array::new(
            ScalarBuffer::from(std::mem::take(&mut scratch.probe_indices)),
            None,
        );
        let mut columns = Vec::with_capacity(self.output_projection.len());
        for col in &self.output_projection {
            let JoinOutputCol::Probe(idx) = col else {
                return internal_err!(
                    "Probe-only join output projection unexpectedly references build column"
                );
            };
            columns.push(take(cursor.batch.column(*idx).as_ref(), &probe_idx, None)?);
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

fn reserve_indices(buf: &mut Vec<u32>, limit: usize) -> Result<()> {
    if buf.capacity() < limit {
        buf.try_reserve(limit - buf.len()).map_err(|e| {
            datafusion_common::exec_datafusion_err!(
                "Morsel HashJoin probe buffer allocation failed: {e}"
            )
        })?;
    }
    Ok(())
}

#[inline]
fn probe_row_at(rows: &ProbeRows, pos: usize) -> Option<(usize, u32)> {
    match rows {
        ProbeRows::All { .. } => Some((pos, pos as u32)),
        ProbeRows::Mask { bits, .. } => bits.value(pos).then_some((pos, pos as u32)),
        ProbeRows::BackingIndices(indices) => {
            let backing = indices.value(pos);
            Some((backing as usize, backing))
        }
        ProbeRows::CompactedKeys { backing_rows } => Some((pos, backing_rows.value(pos))),
    }
}

fn collect_cross_pairs(
    cursor: &mut ProbeCursor,
    build_rows: usize,
    limit: usize,
    build_out: &mut Vec<u32>,
    probe_out: &mut Vec<u32>,
) {
    let mut produced = 0usize;
    while !cursor.is_exhausted() && produced < limit {
        let Some((_, backing_row)) = probe_row_at(&cursor.rows, cursor.pos) else {
            cursor.pos += 1;
            continue;
        };
        let start_b = cursor.next_entry as usize;
        let take_n = (build_rows - start_b).min(limit - produced);
        for b in start_b..(start_b + take_n) {
            build_out.push(b as u32);
            probe_out.push(backing_row);
        }
        produced += take_n;
        let next_b = start_b + take_n;
        if next_b >= build_rows {
            cursor.next_entry = 0;
            cursor.pos += 1;
        } else {
            cursor.next_entry = next_b as u32;
            break;
        }
    }
}

fn probe_has_any_match(
    keys: &ProbeKeys,
    table: &FinalizedHashTable,
    key_idx: usize,
) -> Result<bool> {
    macro_rules! flat_exists {
        ($p_arr:expr, $flat:expr, $hash_key:expr) => {{
            let pk = $p_arr.value(key_idx);
            let slots = $flat.slots.as_slice();
            let mask = $flat.mask;
            let mut slot_pos = ($hash_key(pk) as usize) & mask;
            loop {
                let slot = &slots[slot_pos];
                if slot.first_row == 0 {
                    return Ok(false);
                }
                if slot.key == pk {
                    return Ok(true);
                }
                slot_pos = (slot_pos + 1) & mask;
            }
        }};
    }
    macro_rules! chained_exists {
        ($h:expr, $eq:expr) => {{
            let h = $h;
            let mut entry = table.buckets[(h as usize) & table.mask];
            while entry != 0 {
                let b = (entry - 1) as usize;
                entry = table.next[b];
                if $eq(b, h) {
                    return Ok(true);
                }
            }
            return Ok(false);
        }};
    }
    match (keys, &table.keys) {
        (ProbeKeys::Int32(p), BuildKeys::Int32(f)) => {
            flat_exists!(p, f, |v: i32| mix_u64(v as u32 as u64))
        }
        (ProbeKeys::Int64(p), BuildKeys::Int64(f)) => {
            flat_exists!(p, f, |v: i64| mix_u64(v as u64))
        }
        (ProbeKeys::UInt32(p), BuildKeys::UInt32(f)) => {
            flat_exists!(p, f, |v: u32| mix_u64(v as u64))
        }
        (ProbeKeys::UInt64(p), BuildKeys::UInt64(f)) => {
            flat_exists!(p, f, |v: u64| mix_u64(v))
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
        ) => chained_exists!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
        (
            ProbeKeys::LargeUtf8 {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::LargeUtf8 {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => chained_exists!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
        (
            ProbeKeys::Utf8View {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::Utf8View {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => chained_exists!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
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
        ) => chained_exists!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_rows.row(b).as_ref() == p_rows.row(key_idx).as_ref()
        }),
        _ => internal_err!("Morsel HashJoin probe/build key representation mismatch"),
    }
}

fn probe_mark_all_matches(
    keys: &ProbeKeys,
    table: &FinalizedHashTable,
    key_idx: usize,
) -> Result<()> {
    macro_rules! flat_mark {
        ($p_arr:expr, $flat:expr, $hash_key:expr) => {{
            let pk = $p_arr.value(key_idx);
            let slots = $flat.slots.as_slice();
            let dup_next = $flat.dup_next.as_slice();
            let mask = $flat.mask;
            let mut slot_pos = ($hash_key(pk) as usize) & mask;
            loop {
                let slot = &slots[slot_pos];
                if slot.first_row == 0 {
                    return Ok(());
                }
                if slot.key == pk {
                    table.mark_matched(slot.first_row - 1);
                    let mut entry = slot.dup_head;
                    while entry != 0 {
                        let b = entry - 1;
                        entry = dup_next[b as usize];
                        table.mark_matched(b);
                    }
                    return Ok(());
                }
                slot_pos = (slot_pos + 1) & mask;
            }
        }};
    }
    macro_rules! chained_mark {
        ($h:expr, $eq:expr) => {{
            let h = $h;
            let mut entry = table.buckets[(h as usize) & table.mask];
            while entry != 0 {
                let b = (entry - 1) as usize;
                entry = table.next[b];
                if $eq(b, h) {
                    table.mark_matched(b as u32);
                }
            }
            return Ok(());
        }};
    }
    match (keys, &table.keys) {
        (ProbeKeys::Int32(p), BuildKeys::Int32(f)) => {
            flat_mark!(p, f, |v: i32| mix_u64(v as u32 as u64))
        }
        (ProbeKeys::Int64(p), BuildKeys::Int64(f)) => {
            flat_mark!(p, f, |v: i64| mix_u64(v as u64))
        }
        (ProbeKeys::UInt32(p), BuildKeys::UInt32(f)) => {
            flat_mark!(p, f, |v: u32| mix_u64(v as u64))
        }
        (ProbeKeys::UInt64(p), BuildKeys::UInt64(f)) => {
            flat_mark!(p, f, |v: u64| mix_u64(v))
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
        ) => chained_mark!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
        (
            ProbeKeys::LargeUtf8 {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::LargeUtf8 {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => chained_mark!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
        (
            ProbeKeys::Utf8View {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::Utf8View {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => chained_mark!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
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
        ) => chained_mark!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_rows.row(b).as_ref() == p_rows.row(key_idx).as_ref()
        }),
        _ => internal_err!("Morsel HashJoin probe/build key representation mismatch"),
    }
}

#[expect(clippy::too_many_arguments)]
fn probe_step_candidates(
    cursor: &mut ProbeCursor,
    table: &FinalizedHashTable,
    key_idx: usize,
    backing_row: u32,
    max_cands: usize,
    skip_already_matched: bool,
    cand_build: &mut Vec<u32>,
    cand_probe: &mut Vec<u32>,
) -> Result<bool> {
    macro_rules! flat_step {
        ($p_arr:expr, $flat:expr, $hash_key:expr) => {{
            let dup_next = $flat.dup_next.as_slice();
            let mut added = 0usize;
            let mut entry = if cursor.next_entry == 0 {
                let pk = $p_arr.value(key_idx);
                let slots = $flat.slots.as_slice();
                let mask = $flat.mask;
                let mut slot_pos = ($hash_key(pk) as usize) & mask;
                let found = loop {
                    let slot = &slots[slot_pos];
                    if slot.first_row == 0 {
                        break None;
                    }
                    if slot.key == pk {
                        break Some((slot.first_row - 1, slot.dup_head));
                    }
                    slot_pos = (slot_pos + 1) & mask;
                };
                let Some((first_row, dup_head)) = found else {
                    cursor.next_entry = 0;
                    return Ok(true);
                };
                if !skip_already_matched || !table.is_matched(first_row) {
                    cand_build.push(first_row);
                    cand_probe.push(backing_row);
                    added += 1;
                }
                dup_head
            } else {
                cursor.next_entry
            };
            while entry != 0 && added < max_cands {
                let b = entry - 1;
                entry = dup_next[b as usize];
                if !skip_already_matched || !table.is_matched(b) {
                    cand_build.push(b);
                    cand_probe.push(backing_row);
                    added += 1;
                }
            }
            cursor.next_entry = entry;
            return Ok(entry == 0);
        }};
    }
    macro_rules! chained_step {
        ($h:expr, $eq:expr) => {{
            let h = $h;
            let mut added = 0usize;
            let mut entry = if cursor.next_entry == 0 {
                table.buckets[(h as usize) & table.mask]
            } else {
                cursor.next_entry
            };
            while entry != 0 && added < max_cands {
                let b = (entry - 1) as usize;
                entry = table.next[b];
                if $eq(b, h) && (!skip_already_matched || !table.is_matched(b as u32)) {
                    cand_build.push(b as u32);
                    cand_probe.push(backing_row);
                    added += 1;
                }
            }
            cursor.next_entry = entry;
            return Ok(entry == 0);
        }};
    }
    match (&cursor.keys, &table.keys) {
        (ProbeKeys::Int32(p), BuildKeys::Int32(f)) => {
            flat_step!(p, f, |v: i32| mix_u64(v as u32 as u64))
        }
        (ProbeKeys::Int64(p), BuildKeys::Int64(f)) => {
            flat_step!(p, f, |v: i64| mix_u64(v as u64))
        }
        (ProbeKeys::UInt32(p), BuildKeys::UInt32(f)) => {
            flat_step!(p, f, |v: u32| mix_u64(v as u64))
        }
        (ProbeKeys::UInt64(p), BuildKeys::UInt64(f)) => {
            flat_step!(p, f, |v: u64| mix_u64(v))
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
        ) => chained_step!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
        (
            ProbeKeys::LargeUtf8 {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::LargeUtf8 {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => chained_step!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
        (
            ProbeKeys::Utf8View {
                array: p_arr,
                hashes: p_hashes,
            },
            BuildKeys::Utf8View {
                array: b_arr,
                hashes: b_hashes,
            },
        ) => chained_step!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_arr.value(b) == p_arr.value(key_idx)
        }),
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
        ) => chained_step!(p_hashes[key_idx], |b: usize, h: u64| {
            b_hashes[b] == h && b_rows.row(b).as_ref() == p_rows.row(key_idx).as_ref()
        }),
        _ => internal_err!("Morsel HashJoin probe/build key representation mismatch"),
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
            HashJoinBuildSink::try_new(HashJoinBuildConfig {
                key_exprs: vec![Arc::new(Column::new("bk", 0))],
                key_names: vec!["bk".to_owned()],
                key_types: vec![DataType::Int64],
                null_equality: NullEquality::NullEqualsNothing,
                keep_null_key_rows: false,
                track_null_keys: false,
                needs_matched_bitset: false,
                payload_columns: vec![1],
                payload_schema: Arc::clone(&payload_schema),
                dynamic_filter: None,
            })?
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
            kind: ProbeJoinKind::Inner,
            join_type: JoinType::Inner,
            null_equality: NullEquality::NullEqualsNothing,
            probe_key_exprs: vec![Arc::new(Column::new("pk", 0))],
            probe_key_names: vec!["pk".to_owned()],
            key_types: vec![DataType::Int64],
            filter: None,
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
            kind: ProbeJoinKind::Inner,
            join_type: JoinType::Inner,
            null_equality: NullEquality::NullEqualsNothing,
            probe_key_exprs: vec![Arc::new(Column::new("pk", 0))],
            probe_key_names: vec!["pk".to_owned()],
            key_types: vec![DataType::Int64],
            filter: None,
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
            HashJoinBuildSink::try_new(HashJoinBuildConfig {
                key_exprs: vec![Arc::new(Column::new("bk", 0))],
                key_names: vec!["bk".to_owned()],
                key_types: vec![DataType::Int64],
                null_equality: NullEquality::NullEqualsNothing,
                keep_null_key_rows: false,
                track_null_keys: false,
                needs_matched_bitset: false,
                payload_columns: vec![1],
                payload_schema,
                dynamic_filter: None,
            })?
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
            kind: ProbeJoinKind::Inner,
            join_type: JoinType::Inner,
            null_equality: NullEquality::NullEqualsNothing,
            probe_key_exprs: vec![Arc::new(Column::new("pk", 0))],
            probe_key_names: vec!["pk".to_owned()],
            key_types: vec![DataType::Int64],
            filter: None,
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
