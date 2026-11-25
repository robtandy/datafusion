// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray,
    NullBufferBuilder, RecordBatch, RecordBatchOptions, StringArray, StringViewArray,
    UInt32Array, UInt64Array, new_empty_array, new_null_array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer, ScalarBuffer};
use arrow::compute::{
    concat, concat_batches, filter, filter_record_batch, interleave, take,
    take_record_batch,
};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_common::{
    DataFusionError, JoinType, NullEquality, Result, ScalarValue, exec_datafusion_err,
    exec_err, float_semantics, internal_err,
};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_execution::spill_file::SpillFile;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use datafusion_physical_plan::spill::SpillManager;
use futures::StreamExt;
use parking_lot::{Mutex, RwLock};
use tokio::sync::mpsc;

use super::batch::{
    BatchView, Selection, collect_expr_column_indices, remap_expr_columns,
};
use super::join_filter::{
    AtomicSplitBlockBloomFilter, DynamicJoinFilterConfig, DynamicJoinFilterData,
    JoinMembershipFilter, LocalJoinKeySummary, SplitBlockBloomFilter,
    try_build_small_inlist_from_chunks,
};
use super::radix::{
    MAX_RADIX_DEPTH, RECURSIVE_RADIX_BITS, next_level_shift, radix_partition,
    radix_permute_indices,
};
use super::radix_join::{
    LocalProbeSpill, NullKeyBuildRows, ProbeSpillEnv, RADIX_JOIN_THRESHOLD_ROWS,
    RadixBuildStore, SpilledBuildPartition, SpilledProbeManifest,
    build_join_spill_schema, pack_build_spill_batch, unpack_build_spill_batch,
};
use super::spill::{SpillContext, spill_batches_async};
use super::{Pipeline, abort_and_drain};

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

    #[cfg(any(feature = "parquet", test))]
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
    build_spill_schema: SchemaRef,
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
        let build_spill_schema =
            build_join_spill_schema(&config.payload_schema, &config.key_types);
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
            build_spill_schema,
            dynamic_filter: config.dynamic_filter,
            #[cfg(test)]
            finalize_gate: None,
        }))
    }

    pub(super) fn build_spill_schema(&self) -> &SchemaRef {
        &self.build_spill_schema
    }

    pub(super) fn payload_schema(&self) -> &SchemaRef {
        &self.payload_schema
    }

    pub(super) fn can_radix_spill(&self) -> bool {
        self.key_kind != KeyKind::Cross && !self.track_null_keys
    }

    #[cfg_attr(not(test), expect(dead_code))]
    pub(super) fn new_state(&self, reservation: MemoryReservation) -> HashJoinBuildState {
        self.new_state_with_spill(reservation, None, None)
    }

    pub(super) fn new_state_with_spill(
        &self,
        reservation: MemoryReservation,
        build_spill: Option<SpillContext>,
        null_key_spill: Option<SpillContext>,
    ) -> HashJoinBuildState {
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
            payload_schema: Arc::clone(&self.payload_schema),
            build_spill_schema: Arc::clone(&self.build_spill_schema),
            can_radix: self.can_radix_spill(),
            batches: Vec::new(),
            key_batches: Vec::new(),
            hash_chunks: Vec::new(),
            valid_masks: Vec::new(),
            r0_mem_bytes: 0,
            radix: None,
            pending_radix_batches: Vec::new(),
            pending_radix_hashes: Vec::new(),
            pending_radix_rows: 0,
            pending_radix_bytes: 0,
            null_key_payloads: Vec::new(),
            null_key_spill_files: Vec::new(),
            null_key_rows: 0,
            null_key_mem_bytes: 0,
            build_spill,
            null_key_spill,
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

    #[cfg(test)]
    pub(super) async fn finalize(
        &self,
        combined: HashJoinBuildState,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<Arc<FinalizedHashTable>>> {
        let pool: Arc<dyn MemoryPool> =
            Arc::new(datafusion_execution::memory_pool::UnboundedMemoryPool::default());
        self.finalize_with_spill(combined, 1, &pool, None, None, tx)
            .await
    }

    pub(super) async fn finalize_with_spill(
        &self,
        mut combined: HashJoinBuildState,
        workers: usize,
        memory_pool: &Arc<dyn MemoryPool>,
        build_spill: Option<SpillContext>,
        null_key_spill: Option<SpillContext>,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<Arc<FinalizedHashTable>>> {
        if combined.radix.is_some() || !combined.pending_radix_batches.is_empty() {
            combined.flush_r0_to_radix(tx).await?;
            combined.flush_pending_radix(tx).await?;
            return self
                .finalize_radix(
                    combined,
                    workers,
                    memory_pool,
                    build_spill,
                    null_key_spill,
                    tx,
                )
                .await;
        }
        // If total rows in R=0 exceed RADIX_JOIN_THRESHOLD_ROWS (e.g. multiple
        // workers each under threshold combined above threshold), transition to
        // parallel radix finalize for L2 cache locality!
        if self.can_radix_spill() && combined.num_rows > RADIX_JOIN_THRESHOLD_ROWS {
            combined.flush_r0_to_radix(tx).await?;
            combined.flush_pending_radix(tx).await?;
            return self
                .finalize_radix(
                    combined,
                    workers,
                    memory_pool,
                    build_spill,
                    null_key_spill,
                    tx,
                )
                .await;
        }

        // Single-partition R = 0 fast path (small builds, Cross, null-aware).
        // If finalizing in memory hits OOM and spilling is available on a
        // spillable join, fall back seamlessly to radix spill finalize!
        let can_fallback_spill = self.can_radix_spill() && build_spill.is_some();
        if can_fallback_spill {
            let est_peak = combined
                .r0_mem_bytes
                .saturating_mul(2)
                .saturating_add(combined.num_rows.saturating_mul(24));
            let probe = MemoryConsumer::new("MorselHashJoinBuildFinalize")
                .with_can_spill(true)
                .register(memory_pool);
            let fits = probe.try_grow(est_peak).is_ok();
            drop(probe);
            if !fits {
                combined.flush_r0_to_radix(tx).await?;
                return self
                    .finalize_radix(
                        combined,
                        workers,
                        memory_pool,
                        build_spill,
                        null_key_spill,
                        tx,
                    )
                    .await;
            }
        }

        self.finalize_single(combined, tx).await
    }

    async fn finalize_single(
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
            return Ok(Some(Arc::new(FinalizedHashTable::from_single(
                SingleHashTable {
                    num_rows,
                    hash_entries: 0,
                    has_null_key: false,
                    payload,
                    keys: BuildKeys::Cross,
                    buckets: Vec::new(),
                    mask: 0,
                    next: Vec::new(),
                    matched: Vec::new(),
                    _reservation: reservation,
                },
                None,
            ))));
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
            return Ok(Some(Arc::new(FinalizedHashTable::from_single(
                SingleHashTable {
                    num_rows,
                    hash_entries: 0,
                    has_null_key,
                    payload,
                    keys: BuildKeys::Empty,
                    buckets: Vec::new(),
                    mask: 0,
                    next: Vec::new(),
                    matched,
                    _reservation: reservation,
                },
                dynamic_filter,
            ))));
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

        Ok(Some(Arc::new(FinalizedHashTable::from_single(
            SingleHashTable {
                num_rows,
                hash_entries,
                has_null_key,
                payload,
                keys: build_keys,
                buckets,
                mask,
                next,
                matched,
                _reservation: reservation,
            },
            dynamic_filter,
        ))))
    }

    async fn finalize_radix(
        &self,
        combined: HashJoinBuildState,
        workers: usize,
        memory_pool: &Arc<dyn MemoryPool>,
        build_spill: Option<SpillContext>,
        null_key_spill: Option<SpillContext>,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<Arc<FinalizedHashTable>>> {
        let HashJoinBuildState {
            radix,
            mut null_key_payloads,
            mut null_key_spill_files,
            null_key_rows,
            null_key_mem_bytes,
            key_summary,
            num_rows,
            has_null_key,
            reservation,
            ..
        } = combined;

        let mut store = radix.unwrap_or_else(RadixBuildStore::new_default);
        let num_parts = store.num_partitions();
        let shift = store.shift;
        let radix_bits = store.radix_bits;
        let total_hash_entries: usize = store.part_rows.iter().sum();

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

        if tx.is_closed() {
            return Ok(None);
        }

        let meta = Arc::new(RadixBuildMetadata {
            key_kind: self.key_kind,
            key_types: self.key_types.clone(),
            needs_matched_bitset: self.needs_matched_bitset,
            num_payload_cols: self.payload_columns.len(),
            payload_schema: Arc::clone(&self.payload_schema),
            row_converter: if self.key_kind == KeyKind::Rows {
                let fields = self
                    .key_types
                    .iter()
                    .map(|dt| SortField::new(dt.clone()))
                    .collect();
                Some(RowConverter::new(fields)?)
            } else {
                None
            },
        });

        // Estimate memory for building all non-empty partitions in parallel.
        let est_table_bytes = |p: usize| -> usize {
            let rows = store.part_rows[p];
            if rows == 0 {
                return 0;
            }
            let buckets = rows.saturating_mul(2).next_power_of_two().max(16);
            let table_overhead = buckets
                .saturating_mul(16)
                .saturating_add(rows.saturating_mul(16));
            store.part_est_bytes[p].saturating_add(table_overhead)
        };
        let total_est_final: usize = (0..num_parts).map(est_table_bytes).sum();

        let try_all_in_memory = if store.spilled {
            false
        } else if build_spill.is_some() {
            let probe = MemoryConsumer::new("MorselHashJoinBuildFinalize")
                .with_can_spill(true)
                .register(memory_pool);
            // Leave comfortable headroom above finalized tables when spilling is available.
            let fits = probe.try_grow(total_est_final).is_ok();
            drop(probe);
            fits
        } else {
            true
        };

        if try_all_in_memory {
            match self
                .finalize_radix_all_in_memory(
                    &mut store,
                    &meta,
                    workers,
                    memory_pool,
                    total_hash_entries,
                    key_summary.as_ref(),
                    &reservation,
                    tx,
                )
                .await
            {
                Ok(Some((resident_parts, resident_mask, dynamic_filter))) => {
                    let nk_res = reservation.new_empty();
                    let nk_mem = null_key_mem_bytes.min(reservation.size());
                    if nk_mem > 0 {
                        reservation.shrink(nk_mem);
                        nk_res.grow(nk_mem);
                    }
                    drop(reservation);
                    let null_key_build = if null_key_rows > 0 {
                        Some(NullKeyBuildRows {
                            mem_batches: null_key_payloads,
                            spill_files: null_key_spill_files,
                            _reservation: nk_res,
                        })
                    } else {
                        None
                    };
                    let spilled_build =
                        (0..num_parts).map(|_| Mutex::new(None)).collect();
                    let radix_table = RadixHashTable {
                        shift,
                        radix_bits,
                        resident: RwLock::new(Some(Arc::new(resident_parts))),
                        resident_mask,
                        spilled_build_mask: 0,
                        spilled_build,
                        null_key_build: Mutex::new(null_key_build),
                        probe_spill_manifest: Mutex::new(SpilledProbeManifest::new(
                            num_parts,
                        )),
                        build_spill_manager: build_spill.map(|s| s.manager),
                        null_key_spill_manager: null_key_spill.map(|s| s.manager),
                        meta,
                    };
                    return Ok(Some(Arc::new(FinalizedHashTable {
                        num_rows,
                        has_null_key,
                        kind: FinalizedHashTableKind::Radix(Box::new(radix_table)),
                        dynamic_filter,
                    })));
                }
                Ok(None) => return Ok(None),
                Err(err)
                    if matches!(err, DataFusionError::ResourcesExhausted(_))
                        && build_spill.is_some() =>
                {
                    // Fall through to hybrid resident M + spilled S below.
                }
                Err(err) => return Err(err),
            }
        }

        // Hybrid Resident M + Spilled S path (bounded memory out-of-core join).
        let Some(build_spill_ctx) = build_spill else {
            return Err(DataFusionError::ResourcesExhausted(
                "Morsel HashJoin build exceeded memory pool and spilling is disabled"
                    .into(),
            ));
        };

        // Spill any remaining in-memory NULL-key rows so RAM is dedicated to M / ProbeSpill.
        if !null_key_payloads.is_empty() {
            if let Some(nk_spill) = null_key_spill.as_ref() {
                if let Some((sf, _)) = spill_batches_async(
                    &nk_spill.manager,
                    &null_key_payloads,
                    "MorselHashJoinNullKeySpill",
                    tx,
                )
                .await?
                {
                    null_key_spill_files.push(sf);
                }
                null_key_payloads.clear();
            }
        }

        // Spill all in-memory chunks in `store` so 100% of sink memory is freed
        // before deciding resident subset M vs spilled subset S.
        // Keeping all non-empty spilled partitions in S when tight guarantees
        // maximum headroom for LocalProbeSpill and single-partition P2 replay!
        store
            .spill_all_mem(0, &reservation, &build_spill_ctx, tx)
            .await?;
        drop(reservation);

        if tx.is_closed() {
            return Ok(None);
        }

        // Greedily load a small resident subset M only if it fits comfortably
        // within a bounded fraction of available pool (leaving ample room for
        // LocalProbeSpill); remaining partitions stay in S!
        let mut resident_vec: Vec<Option<SingleHashTable>> =
            (0..num_parts).map(|_| None).collect();
        let mut resident_mask = 0u64;
        let mut spilled_build_mask = 0u64;
        let spilled_build: Vec<Mutex<Option<SpilledBuildPartition>>> =
            (0..num_parts).map(|_| Mutex::new(None)).collect();

        // Reserve probe-spill safety headroom while choosing resident M.
        let headroom_guard = MemoryConsumer::new("MorselHashJoinProbeHeadroom")
            .with_can_spill(true)
            .register(memory_pool);
        let has_headroom = headroom_guard
            .try_grow(build_spill_ctx.min_headroom_bytes.saturating_mul(2))
            .is_ok();

        for p in 0..num_parts {
            if tx.is_closed() {
                return Ok(None);
            }
            let files = std::mem::take(&mut store.spill_files[p]);
            let rows = store.part_rows[p];
            if files.is_empty() || rows == 0 {
                continue;
            }
            if has_headroom && resident_mask.count_ones() < 16 {
                // Attempt to keep this partition resident in M if it fits cleanly
                // alongside the reserved probe-spill headroom.
                let part_res = MemoryConsumer::new("MorselHashJoinResidentPart")
                    .with_can_spill(true)
                    .register(memory_pool);
                match load_spilled_build_partition_sync_or_async(
                    &files,
                    &build_spill_ctx.manager,
                    &meta,
                    part_res,
                    tx,
                )
                .await
                {
                    Ok(Some(pt)) => {
                        drop(files);
                        resident_vec[p] = Some(pt);
                        resident_mask |= 1u64 << p;
                        continue;
                    }
                    Ok(None) => return Ok(None),
                    Err(DataFusionError::ResourcesExhausted(_)) => {
                        // Doesn't fit alongside headroom -> keep in S!
                    }
                    Err(err) => return Err(err),
                }
            }
            spilled_build_mask |= 1u64 << p;
            *spilled_build[p].lock() = Some(SpilledBuildPartition { files });
        }
        drop(headroom_guard);

        // Sound dynamic join filter over M ∪ S: bounds cover 100% of M ∪ S;
        // membership is None whenever S is non-empty (zero false negatives!).
        let dynamic_filter = match (self.dynamic_filter, key_summary) {
            (Some(_), Some(summary)) => Some(Arc::new(DynamicJoinFilterData {
                key_kind: self.key_kind,
                null_equality: self.null_equality,
                has_floats: self.has_floats,
                bounds: summary.into_bounds(),
                membership: JoinMembershipFilter::None,
            })),
            _ => None,
        };

        let null_key_build = if null_key_rows > 0 {
            Some(NullKeyBuildRows {
                mem_batches: null_key_payloads,
                spill_files: null_key_spill_files,
                _reservation: MemoryConsumer::new("MorselHashJoinNullKey")
                    .register(memory_pool),
            })
        } else {
            None
        };

        let resident_parts = ResidentRadixPartitions {
            partitions: resident_vec,
            _reservation: MemoryConsumer::new("MorselHashJoinResident")
                .register(memory_pool),
        };
        let radix_table = RadixHashTable {
            shift,
            radix_bits,
            resident: RwLock::new(Some(Arc::new(resident_parts))),
            resident_mask,
            spilled_build_mask,
            spilled_build,
            null_key_build: Mutex::new(null_key_build),
            probe_spill_manifest: Mutex::new(SpilledProbeManifest::new(num_parts)),
            build_spill_manager: Some(build_spill_ctx.manager),
            null_key_spill_manager: null_key_spill.map(|s| s.manager),
            meta,
        };

        Ok(Some(Arc::new(FinalizedHashTable {
            num_rows,
            has_null_key,
            kind: FinalizedHashTableKind::Radix(Box::new(radix_table)),
            dynamic_filter,
        })))
    }

    #[expect(clippy::too_many_arguments)]
    async fn finalize_radix_all_in_memory(
        &self,
        store: &mut RadixBuildStore,
        meta: &Arc<RadixBuildMetadata>,
        workers: usize,
        memory_pool: &Arc<dyn MemoryPool>,
        total_hash_entries: usize,
        key_summary: Option<&LocalJoinKeySummary>,
        input_reservation: &MemoryReservation,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<
        Option<(
            ResidentRadixPartitions,
            u64,
            Option<Arc<DynamicJoinFilterData>>,
        )>,
    > {
        let num_parts = store.num_partitions();
        let non_empty: usize = store.mem_batches.iter().filter(|v| !v.is_empty()).count();
        if non_empty == 0 {
            return Ok(Some((
                ResidentRadixPartitions {
                    partitions: (0..num_parts).map(|_| None).collect(),
                    _reservation: MemoryConsumer::new("MorselHashJoinResident")
                        .register(memory_pool),
                },
                0,
                None,
            )));
        }

        // Optional parallel lock-free split-block Bloom filter populated across
        // all finalize workers.
        let bloom_reservation =
            MemoryConsumer::new("MorselHashJoinBloom").register(memory_pool);
        let atomic_bloom = if let Some(cfg) = self.dynamic_filter
            && cfg.build_bloom
            && (total_hash_entries > 1 || self.key_types.len() > 1 || self.has_floats)
            && let Some(num_blocks) =
                SplitBlockBloomFilter::num_blocks_for_entries(total_hash_entries)
        {
            let est = SplitBlockBloomFilter::estimated_bytes(num_blocks);
            if bloom_reservation.try_grow(est).is_ok() {
                Some(Arc::new(AtomicSplitBlockBloomFilter::with_blocks(
                    num_blocks,
                )))
            } else {
                None
            }
        } else {
            None
        };

        // Share per-partition input slices in Arc<Vec<Vec<RecordBatch>>> (keep
        // `store.mem_batches` intact until parallel build succeeds so OOM fallback
        // can cleanly spill all partitions!).
        let part_inputs: Arc<Vec<Vec<RecordBatch>>> = Arc::new(store.mem_batches.clone());
        let results: Arc<Vec<Mutex<Option<SingleHashTable>>>> =
            Arc::new((0..num_parts).map(|_| Mutex::new(None)).collect());
        let next_part = Arc::new(AtomicUsize::new(0));
        let worker_count = workers.min(non_empty).max(1);

        let mut tasks: JoinSet<Result<()>> = JoinSet::new();
        for _ in 0..worker_count {
            let part_inputs = Arc::clone(&part_inputs);
            let results = Arc::clone(&results);
            let next_part = Arc::clone(&next_part);
            let meta = Arc::clone(meta);
            let pool = Arc::clone(memory_pool);
            let bloom = atomic_bloom.clone();
            let tx = tx.clone();

            tasks.spawn(async move {
                loop {
                    if tx.is_closed() {
                        return Ok(());
                    }
                    let p = next_part.fetch_add(1, Ordering::Relaxed);
                    if p >= part_inputs.len() {
                        return Ok(());
                    }
                    let batches = &part_inputs[p];
                    if batches.is_empty() {
                        continue;
                    }
                    let part_res = MemoryConsumer::new("MorselHashJoinBuildPart")
                        .with_can_spill(true)
                        .register(&pool);
                    let table = build_single_table_from_packed_batches(
                        batches,
                        &meta,
                        bloom.as_deref(),
                        part_res,
                    )?;
                    *results[p].lock() = Some(table);
                    tokio::task::yield_now().await;
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
                        "Morsel radix HashJoin finalize task failed: {err}"
                    ));
                }
            }
        }
        drop(part_inputs);
        if tx.is_closed() {
            return Ok(None);
        }

        // Release raw input slices now that all 64 partition tables are built.
        for v in &mut store.mem_batches {
            v.clear();
        }
        store.total_mem_bytes = 0;
        let _ = input_reservation;

        let mut partitions = Vec::with_capacity(num_parts);
        let mut resident_mask = 0u64;
        for (p, slot) in results.iter().enumerate() {
            let pt = slot.lock().take();
            if pt.is_some() {
                resident_mask |= 1u64 << p;
            }
            partitions.push(pt);
        }

        let dynamic_filter = match (self.dynamic_filter, key_summary) {
            (Some(_cfg), Some(summary)) => {
                let membership = if let Some(ab) = atomic_bloom {
                    let bloom = Arc::try_unwrap(ab)
                        .ok()
                        .expect("all finalize tasks joined")
                        .into_bloom();
                    JoinMembershipFilter::Bloom(Arc::new(bloom))
                } else {
                    JoinMembershipFilter::None
                };
                Some(Arc::new(DynamicJoinFilterData {
                    key_kind: self.key_kind,
                    null_equality: self.null_equality,
                    has_floats: self.has_floats,
                    bounds: summary.clone().into_bounds(),
                    membership,
                }))
            }
            _ => None,
        };

        Ok(Some((
            ResidentRadixPartitions {
                partitions,
                _reservation: bloom_reservation,
            },
            resident_mask,
            dynamic_filter,
        )))
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
    payload_schema: SchemaRef,
    build_spill_schema: SchemaRef,
    can_radix: bool,
    batches: Vec<RecordBatch>,
    key_batches: Vec<Vec<ArrayRef>>,
    hash_chunks: Vec<Vec<u64>>,
    valid_masks: Vec<Option<BooleanBuffer>>,
    r0_mem_bytes: usize,
    radix: Option<RadixBuildStore>,
    pending_radix_batches: Vec<RecordBatch>,
    pending_radix_hashes: Vec<u64>,
    pending_radix_rows: usize,
    pending_radix_bytes: usize,
    null_key_payloads: Vec<RecordBatch>,
    null_key_spill_files: Vec<Arc<dyn SpillFile>>,
    null_key_rows: usize,
    null_key_mem_bytes: usize,
    build_spill: Option<SpillContext>,
    null_key_spill: Option<SpillContext>,
    key_summary: Option<LocalJoinKeySummary>,
    num_rows: usize,
    has_null_key: bool,
    reservation: MemoryReservation,
}

impl HashJoinBuildState {
    /// Flush any R=0 buffered chunks into `self.radix`, separating NULL-key
    /// rows (when `keep_null_key_rows == true`) into `null_key_payloads` /
    /// `null_key_spill_files` so every row in `radix` is matchable.
    async fn flush_r0_to_radix(&mut self, tx: &mpsc::Sender<RecordBatch>) -> Result<()> {
        let batches = std::mem::take(&mut self.batches);
        let key_batches = std::mem::take(&mut self.key_batches);
        let hash_chunks = std::mem::take(&mut self.hash_chunks);
        let valid_masks = std::mem::take(&mut self.valid_masks);
        if self.r0_mem_bytes > 0 {
            let shrink = self.r0_mem_bytes.min(self.reservation.size());
            self.reservation.shrink(shrink);
            self.r0_mem_bytes = 0;
        }
        let _ = self.radix.get_or_insert_with(RadixBuildStore::new_default);
        if hash_chunks.is_empty() {
            return Ok(());
        }

        for idx in 0..hash_chunks.len() {
            let payload_opt = if self.payload_columns.is_empty() {
                None
            } else {
                batches.get(idx).cloned()
            };
            let keys = &key_batches[idx];
            let hashes = &hash_chunks[idx];
            let valid_opt = valid_masks.get(idx).and_then(Option::as_ref);

            if let Some(bits) = valid_opt {
                let valid_count = bits.count_set_bits();
                let invalid_count = bits.len() - valid_count;
                if invalid_count > 0 && self.keep_null_key_rows {
                    let inv_mask = BooleanArray::new(!bits, None);
                    let nk_batch = match &payload_opt {
                        Some(p) => filter_record_batch(p, &inv_mask)?,
                        None => RecordBatch::try_new_with_options(
                            Arc::clone(&self.payload_schema),
                            vec![],
                            &RecordBatchOptions::new()
                                .with_row_count(Some(invalid_count)),
                        )?,
                    };
                    self.push_null_key_batch(nk_batch, tx).await?;
                }
                if valid_count > 0 {
                    let v_mask = BooleanArray::new(bits.clone(), None);
                    let v_keys: Vec<ArrayRef> = keys
                        .iter()
                        .map(|k| Ok(filter(k.as_ref(), &v_mask)?))
                        .collect::<Result<_>>()?;
                    let v_payload = match &payload_opt {
                        Some(p) => Some(filter_record_batch(p, &v_mask)?),
                        None => None,
                    };
                    let v_hashes: Vec<u64> =
                        bits.set_indices().map(|i| hashes[i]).collect();
                    let packed = pack_build_spill_batch(
                        &self.build_spill_schema,
                        v_payload.as_ref(),
                        &v_keys,
                        v_hashes.clone(),
                    )?;
                    let extra = self.null_key_mem_bytes;
                    let radix =
                        self.radix.get_or_insert_with(RadixBuildStore::new_default);
                    radix
                        .push_packed_batch(
                            packed,
                            &v_hashes,
                            extra,
                            &self.reservation,
                            self.build_spill.as_ref(),
                            tx,
                        )
                        .await?;
                }
            } else {
                let packed = pack_build_spill_batch(
                    &self.build_spill_schema,
                    payload_opt.as_ref(),
                    keys,
                    hashes.clone(),
                )?;
                let extra = self.null_key_mem_bytes;
                let radix = self.radix.get_or_insert_with(RadixBuildStore::new_default);
                radix
                    .push_packed_batch(
                        packed,
                        hashes,
                        extra,
                        &self.reservation,
                        self.build_spill.as_ref(),
                        tx,
                    )
                    .await?;
            }
        }
        Ok(())
    }

    async fn push_null_key_batch(
        &mut self,
        nk_batch: RecordBatch,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        let rows = nk_batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        self.null_key_rows = self.null_key_rows.saturating_add(rows);
        let bytes = get_record_batch_memory_size(&nk_batch);
        let already_spilled = self
            .build_spill
            .as_ref()
            .is_some_and(SpillContext::is_spilled)
            || !self.null_key_spill_files.is_empty();
        if !already_spilled && self.reservation.try_grow(bytes).is_ok() {
            self.null_key_payloads.push(nk_batch);
            self.null_key_mem_bytes = self.null_key_mem_bytes.saturating_add(bytes);
            return Ok(());
        }
        if let Some(nk_spill) = self.null_key_spill.clone() {
            let mut to_spill = std::mem::take(&mut self.null_key_payloads);
            if self.null_key_mem_bytes > 0 {
                let shrink = self.null_key_mem_bytes.min(self.reservation.size());
                self.reservation.shrink(shrink);
                self.null_key_mem_bytes = 0;
            }
            to_spill.push(nk_batch);
            if let Some((sf, _)) = spill_batches_async(
                &nk_spill.manager,
                &to_spill,
                "MorselHashJoinNullKeySpill",
                tx,
            )
            .await?
            {
                self.null_key_spill_files.push(sf);
            }
            return Ok(());
        }
        // Spilling disabled -> propagate ResourcesExhausted.
        self.reservation.try_grow(bytes)?;
        self.null_key_payloads.push(nk_batch);
        self.null_key_mem_bytes = self.null_key_mem_bytes.saturating_add(bytes);
        Ok(())
    }

    async fn flush_pending_radix(
        &mut self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if self.pending_radix_batches.is_empty() {
            return Ok(());
        }
        let batches = std::mem::take(&mut self.pending_radix_batches);
        let hashes = std::mem::take(&mut self.pending_radix_hashes);
        self.pending_radix_rows = 0;
        if self.pending_radix_bytes > 0 {
            let shrink = self.pending_radix_bytes.min(self.reservation.size());
            self.reservation.shrink(shrink);
            self.pending_radix_bytes = 0;
        }
        let merged = if batches.len() == 1 {
            batches.into_iter().next().unwrap()
        } else {
            super::sort::concat_batches_compact_views(&self.build_spill_schema, &batches)?
        };
        let extra = self.null_key_mem_bytes;
        let radix = self.radix.get_or_insert_with(RadixBuildStore::new_default);
        radix
            .push_packed_batch(
                merged,
                &hashes,
                extra,
                &self.reservation,
                self.build_spill.as_ref(),
                tx,
            )
            .await
    }

    async fn push_radix_packed(
        &mut self,
        packed: RecordBatch,
        hashes: Vec<u64>,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        const RADIX_COALESCE_TARGET_ROWS: usize = 8_192;
        let rows = packed.num_rows();
        if rows == 0 {
            return Ok(());
        }
        let spilled = self.radix.as_ref().is_some_and(|r| r.spilled)
            || self
                .build_spill
                .as_ref()
                .is_some_and(SpillContext::is_spilled);
        // Fast path: full-sized batch with nothing pending goes straight into
        // radix permutation without an extra concat copy.
        if !spilled
            && self.pending_radix_batches.is_empty()
            && rows >= RADIX_COALESCE_TARGET_ROWS
        {
            let extra = self.null_key_mem_bytes;
            let radix = self.radix.get_or_insert_with(RadixBuildStore::new_default);
            return radix
                .push_packed_batch(
                    packed,
                    &hashes,
                    extra,
                    &self.reservation,
                    self.build_spill.as_ref(),
                    tx,
                )
                .await;
        }
        let bytes = get_record_batch_memory_size(&packed);
        if spilled || self.reservation.try_grow(bytes).is_err() {
            self.flush_pending_radix(tx).await?;
            let extra = self.null_key_mem_bytes;
            let radix = self.radix.get_or_insert_with(RadixBuildStore::new_default);
            return radix
                .push_packed_batch(
                    packed,
                    &hashes,
                    extra,
                    &self.reservation,
                    self.build_spill.as_ref(),
                    tx,
                )
                .await;
        }
        self.pending_radix_bytes = self.pending_radix_bytes.saturating_add(bytes);
        self.pending_radix_rows = self.pending_radix_rows.saturating_add(rows);
        self.pending_radix_batches.push(packed);
        self.pending_radix_hashes.extend_from_slice(&hashes);
        if self.pending_radix_rows >= RADIX_COALESCE_TARGET_ROWS {
            self.flush_pending_radix(tx).await?;
        }
        Ok(())
    }

    pub(super) async fn add_view_async(
        &mut self,
        view: &BatchView,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        if !self.can_radix {
            return self.add_view(view);
        }

        let use_radix = self.radix.is_some()
            || self.num_rows.saturating_add(view.num_rows()) > RADIX_JOIN_THRESHOLD_ROWS
            || self
                .build_spill
                .as_ref()
                .is_some_and(SpillContext::is_spilled);
        if !use_radix {
            match self.add_view(view) {
                Ok(()) => return Ok(()),
                Err(err)
                    if matches!(err, DataFusionError::ResourcesExhausted(_))
                        && self.build_spill.is_some() =>
                {
                    if let Some(bs) = &self.build_spill {
                        bs.mark_spilled();
                    }
                    self.flush_r0_to_radix(tx).await?;
                    // Fall through to ingest `view` via radix/spill path below!
                }
                Err(err) => return Err(err),
            }
        } else if !self.hash_chunks.is_empty() {
            self.flush_r0_to_radix(tx).await?;
        }

        // Radix build path: evaluate keys, update dynamic filter bounds BEFORE
        // any eviction, split NULL-key rows if `keep_null_key_rows`, and scatter
        // matchable rows into 64 radix partitions (spilling cleanly on OOM).
        let batch = view.batch();
        let selected = view.num_rows();
        let keys = if view.is_all() {
            let mut ks = Vec::with_capacity(self.key_exprs.len());
            for expr in &self.key_exprs {
                ks.push(expr.evaluate(batch)?.into_array(selected)?);
            }
            ks
        } else {
            let key_needed_cols = collect_expr_column_indices(&self.key_exprs);
            let compact_key_input = view.materialize_columns(&key_needed_cols)?;
            let mut ks = Vec::with_capacity(self.key_exprs.len());
            for expr in &self.key_exprs {
                let rebound = remap_expr_columns(Arc::clone(expr), &key_needed_cols)?;
                ks.push(rebound.evaluate(&compact_key_input)?.into_array(selected)?);
            }
            ks
        };

        let valid_mask = if self.null_equality == NullEquality::NullEqualsNull {
            None
        } else {
            valid_key_mask(&keys)
        };
        if self.track_null_keys && valid_mask.is_some() {
            self.has_null_key = true;
        }

        let (valid_keys, valid_payload, valid_rows) = match valid_mask {
            Some(bits) => {
                let valid_rows = bits.count_set_bits();
                let invalid_rows = bits.len() - valid_rows;
                if invalid_rows > 0 && self.keep_null_key_rows {
                    let inv_pred = BooleanArray::new(!&bits, None);
                    let nk_batch = if self.payload_columns.is_empty() {
                        RecordBatch::try_new_with_options(
                            Arc::clone(&self.payload_schema),
                            vec![],
                            &RecordBatchOptions::new().with_row_count(Some(invalid_rows)),
                        )?
                    } else if view.is_all() {
                        let proj = batch.project(&self.payload_columns)?;
                        filter_record_batch(&proj, &inv_pred)?
                    } else {
                        let combined_sel = view
                            .selection()
                            .compose_compact_mask(batch.num_rows(), &inv_pred)?;
                        let nv = BatchView::try_new(
                            batch.project(&self.payload_columns)?,
                            combined_sel,
                        )?;
                        nv.materialize()?
                    };
                    self.push_null_key_batch(nk_batch, tx).await?;
                    self.num_rows =
                        self.num_rows.checked_add(invalid_rows).ok_or_else(|| {
                            datafusion_common::exec_datafusion_err!(
                                "Morsel HashJoin build row count overflow"
                            )
                        })?;
                }
                if valid_rows == 0 {
                    return Ok(());
                }
                let v_pred = BooleanArray::new(bits, None);
                let v_keys: Vec<ArrayRef> = keys
                    .iter()
                    .map(|k| Ok(filter(k.as_ref(), &v_pred)?))
                    .collect::<Result<_>>()?;
                let v_payload = if self.payload_columns.is_empty() {
                    None
                } else if view.is_all() {
                    let proj = batch.project(&self.payload_columns)?;
                    Some(filter_record_batch(&proj, &v_pred)?)
                } else {
                    let combined_sel = view
                        .selection()
                        .compose_compact_mask(batch.num_rows(), &v_pred)?;
                    let nv = BatchView::try_new(
                        batch.project(&self.payload_columns)?,
                        combined_sel,
                    )?;
                    Some(nv.materialize()?)
                };
                (v_keys, v_payload, valid_rows)
            }
            None => {
                let payload = if self.payload_columns.is_empty() {
                    None
                } else if view.is_all() {
                    Some(batch.project(&self.payload_columns)?)
                } else {
                    Some(view.materialize_columns(&self.payload_columns)?)
                };
                (keys, payload, selected)
            }
        };

        let normalized_keys = normalize_key_arrays(valid_keys, self.has_floats);
        if let Some(summary) = &mut self.key_summary {
            summary.update(&normalized_keys, None, self.null_equality)?;
        }
        let hashes = compute_build_hashes(self.key_kind, &normalized_keys, valid_rows)?;
        let packed = pack_build_spill_batch(
            &self.build_spill_schema,
            valid_payload.as_ref(),
            &normalized_keys,
            hashes.clone(),
        )?;
        self.push_radix_packed(packed, hashes, tx).await?;
        self.num_rows = self.num_rows.checked_add(valid_rows).ok_or_else(|| {
            datafusion_common::exec_datafusion_err!(
                "Morsel HashJoin build row count overflow"
            )
        })?;
        Ok(())
    }

    pub(super) async fn combine_async(
        &mut self,
        mut other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        let combined_null = self.has_null_key | other.has_null_key;
        self.has_null_key = combined_null;
        if other.num_rows == 0 {
            return Ok(());
        }
        if self.num_rows == 0 {
            other.has_null_key = combined_null;
            std::mem::swap(self, &mut other);
            return Ok(());
        }
        let use_radix = self.radix.is_some()
            || other.radix.is_some()
            || !self.pending_radix_batches.is_empty()
            || !other.pending_radix_batches.is_empty()
            || (self.can_radix
                && self.num_rows.saturating_add(other.num_rows)
                    > RADIX_JOIN_THRESHOLD_ROWS);
        if !use_radix {
            self.combine(other);
            return Ok(());
        }
        self.flush_r0_to_radix(tx).await?;
        self.flush_pending_radix(tx).await?;
        other.flush_r0_to_radix(tx).await?;
        other.flush_pending_radix(tx).await?;
        if let (Some(mine), Some(theirs)) = (&mut self.key_summary, &other.key_summary) {
            mine.merge(theirs);
        }
        if let Some(other_radix) = other.radix.take() {
            let radix = self.radix.get_or_insert_with(RadixBuildStore::new_default);
            radix.combine_from(other_radix);
        }
        self.null_key_payloads.append(&mut other.null_key_payloads);
        self.null_key_spill_files
            .append(&mut other.null_key_spill_files);
        self.null_key_rows = self.null_key_rows.saturating_add(other.null_key_rows);
        self.null_key_mem_bytes = self
            .null_key_mem_bytes
            .saturating_add(other.null_key_mem_bytes);
        self.num_rows = self.num_rows.saturating_add(other.num_rows);
        let transferred = other.reservation.free();
        self.reservation.grow(transferred);
        Ok(())
    }

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
            self.r0_mem_bytes = self.r0_mem_bytes.saturating_add(delta);
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
            self.r0_mem_bytes = self.r0_mem_bytes.saturating_add(delta);
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
        self.r0_mem_bytes = self.r0_mem_bytes.saturating_add(delta);
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
            self.r0_mem_bytes = self.r0_mem_bytes.saturating_add(delta);
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
            self.r0_mem_bytes = self.r0_mem_bytes.saturating_add(delta);
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
        self.r0_mem_bytes = self.r0_mem_bytes.saturating_add(delta);
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
        self.r0_mem_bytes = self.r0_mem_bytes.saturating_add(other.r0_mem_bytes);
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

pub(super) struct SingleHashTable {
    num_rows: usize,
    hash_entries: usize,
    has_null_key: bool,
    payload: RecordBatch,
    keys: BuildKeys,
    buckets: Vec<u32>,
    mask: usize,
    next: Vec<u32>,
    matched: Vec<AtomicU64>,
    _reservation: MemoryReservation,
}

impl SingleHashTable {
    fn empty(
        payload_schema: SchemaRef,
        has_null_key: bool,
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
            _reservation: reservation,
        }
    }

    pub(super) fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub(super) fn has_hash_entries(&self) -> bool {
        self.hash_entries > 0
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

pub(super) struct RadixBuildMetadata {
    pub(super) key_kind: KeyKind,
    pub(super) key_types: Vec<DataType>,
    pub(super) needs_matched_bitset: bool,
    pub(super) num_payload_cols: usize,
    pub(super) payload_schema: SchemaRef,
    pub(super) row_converter: Option<RowConverter>,
}

pub(super) struct ResidentRadixPartitions {
    pub(super) partitions: Vec<Option<SingleHashTable>>,
    _reservation: MemoryReservation,
}

pub(super) struct RadixHashTable {
    pub(super) shift: u32,
    pub(super) radix_bits: u32,
    pub(super) resident: RwLock<Option<Arc<ResidentRadixPartitions>>>,
    pub(super) resident_mask: u64,
    pub(super) spilled_build_mask: u64,
    pub(super) spilled_build: Vec<Mutex<Option<SpilledBuildPartition>>>,
    pub(super) null_key_build: Mutex<Option<NullKeyBuildRows>>,
    pub(super) probe_spill_manifest: Mutex<SpilledProbeManifest>,
    pub(super) build_spill_manager: Option<SpillManager>,
    pub(super) null_key_spill_manager: Option<SpillManager>,
    pub(super) meta: Arc<RadixBuildMetadata>,
}

pub(super) enum FinalizedHashTableKind {
    Single(Box<SingleHashTable>),
    Radix(Box<RadixHashTable>),
}

pub(super) struct FinalizedHashTable {
    num_rows: usize,
    has_null_key: bool,
    pub(super) kind: FinalizedHashTableKind,
    #[cfg_attr(not(any(feature = "parquet", test)), expect(dead_code))]
    dynamic_filter: Option<Arc<DynamicJoinFilterData>>,
}

impl FinalizedHashTable {
    fn from_single(
        single: SingleHashTable,
        dynamic_filter: Option<Arc<DynamicJoinFilterData>>,
    ) -> Self {
        Self {
            num_rows: single.num_rows,
            has_null_key: single.has_null_key,
            kind: FinalizedHashTableKind::Single(Box::new(single)),
            dynamic_filter,
        }
    }

    fn empty(
        payload_schema: SchemaRef,
        has_null_key: bool,
        dynamic_filter: Option<Arc<DynamicJoinFilterData>>,
        reservation: MemoryReservation,
    ) -> Self {
        Self::from_single(
            SingleHashTable::empty(payload_schema, has_null_key, reservation),
            dynamic_filter,
        )
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

    pub(super) fn has_null_key(&self) -> bool {
        self.has_null_key
    }

    pub(super) fn has_spilled_partitions(&self) -> bool {
        matches!(
            &self.kind,
            FinalizedHashTableKind::Radix(r) if r.spilled_build_mask != 0
        )
    }

    pub(super) fn clone_resident(&self) -> Option<Arc<ResidentRadixPartitions>> {
        match &self.kind {
            FinalizedHashTableKind::Single(_) => None,
            FinalizedHashTableKind::Radix(r) => r.resident.read().clone(),
        }
    }

    pub(super) fn take_resident(&self) -> Option<Arc<ResidentRadixPartitions>> {
        match &self.kind {
            FinalizedHashTableKind::Single(_) => None,
            FinalizedHashTableKind::Radix(r) => r.resident.write().take(),
        }
    }

    pub(super) fn num_radix_partitions(&self) -> usize {
        match &self.kind {
            FinalizedHashTableKind::Single(_) => 0,
            FinalizedHashTableKind::Radix(r) => 1usize << r.radix_bits,
        }
    }
}

/// Build a single-partition [`SingleHashTable`] synchronously from packed
/// `build_spill_schema` batches, populating an optional parallel Bloom filter.
pub(super) fn build_single_table_from_packed_batches(
    packed_batches: &[RecordBatch],
    meta: &RadixBuildMetadata,
    bloom: Option<&AtomicSplitBlockBloomFilter>,
    reservation: MemoryReservation,
) -> Result<SingleHashTable> {
    let num_rows: usize = packed_batches.iter().map(RecordBatch::num_rows).sum();
    if num_rows == 0 {
        reservation.resize(0);
        return Ok(SingleHashTable::empty(
            Arc::clone(&meta.payload_schema),
            false,
            reservation,
        ));
    }
    if num_rows >= u32::MAX as usize {
        return exec_err!("Morsel HashJoin partition exceeds u32::MAX rows ({num_rows})");
    }

    let num_key_cols = meta.key_types.len();
    let mut payload_batches = Vec::with_capacity(packed_batches.len());
    let mut key_batches: Vec<Vec<ArrayRef>> = Vec::with_capacity(packed_batches.len());
    let mut hash_chunks: Vec<Vec<u64>> = Vec::with_capacity(packed_batches.len());
    for b in packed_batches {
        if b.num_rows() == 0 {
            continue;
        }
        let (p_opt, ks, hs) = unpack_build_spill_batch(
            b,
            &meta.payload_schema,
            meta.num_payload_cols,
            num_key_cols,
        )?;
        if let Some(p) = p_opt {
            payload_batches.push(p);
        }
        key_batches.push(ks);
        hash_chunks.push(hs);
    }

    let payload_concat_bytes = if meta.num_payload_cols == 0 || payload_batches.len() <= 1
    {
        payload_batches
            .first()
            .map_or(0, get_record_batch_memory_size)
    } else {
        payload_batches
            .iter()
            .map(get_record_batch_memory_size)
            .fold(0usize, usize::saturating_add)
    };
    let matched_words = if meta.needs_matched_bitset {
        num_rows.div_ceil(64)
    } else {
        0
    };
    let matched_bytes = matched_words.saturating_mul(size_of::<AtomicU64>());
    let bucket_count = num_rows.saturating_mul(2).next_power_of_two().max(16);
    let is_flat_int = matches!(
        meta.key_kind,
        KeyKind::Int32 | KeyKind::Int64 | KeyKind::UInt32 | KeyKind::UInt64
    );
    let (dir_bytes, keys_peak_bytes) = match meta.key_kind {
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
            let concat_bytes = key_batches
                .iter()
                .map(|kb| kb[0].get_array_memory_size())
                .fold(0usize, usize::saturating_add);
            (
                dir,
                concat_bytes.saturating_add(num_rows.saturating_mul(size_of::<u64>())),
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

    let peak_bytes = dir_bytes
        .saturating_add(payload_concat_bytes)
        .saturating_add(keys_peak_bytes)
        .saturating_add(matched_bytes);
    reservation.try_grow(peak_bytes)?;

    let payload = if meta.num_payload_cols == 0 {
        RecordBatch::try_new_with_options(
            Arc::clone(&meta.payload_schema),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )?
    } else if payload_batches.len() == 1 {
        super::sort::gc_view_batch(&payload_batches.pop().unwrap())?
    } else {
        super::sort::concat_batches_compact_views(&meta.payload_schema, &payload_batches)?
    };
    drop(payload_batches);

    if let Some(b) = bloom {
        for chunk in &hash_chunks {
            for &h in chunk {
                b.insert(h);
            }
        }
    }

    let mask = bucket_count - 1;
    let (build_keys, buckets, next) = if is_flat_int {
        macro_rules! build_flat_sync {
            ($Arr:ty, $K:ty, $Variant:ident) => {{
                let mut slots = vec![IntSlot::<$K>::empty(); bucket_count];
                let mut dup_next: Vec<u32> = Vec::new();
                let mut row_idx = 0usize;
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
                    }
                }
                BuildKeys::$Variant(FlatIntTable {
                    slots,
                    mask,
                    dup_next,
                })
            }};
        }
        let keys = match meta.key_kind {
            KeyKind::Int32 => build_flat_sync!(Int32Array, i32, Int32),
            KeyKind::Int64 => build_flat_sync!(Int64Array, i64, Int64),
            KeyKind::UInt32 => build_flat_sync!(UInt32Array, u32, UInt32),
            KeyKind::UInt64 => build_flat_sync!(UInt64Array, u64, UInt64),
            _ => unreachable!(),
        };
        (keys, Vec::new(), Vec::new())
    } else {
        let mut flat_hashes = Vec::with_capacity(num_rows);
        for chunk in &hash_chunks {
            flat_hashes.extend_from_slice(chunk);
        }
        let build_keys = match meta.key_kind {
            KeyKind::Utf8 => {
                let arr_ref = concat_single_key(&mut key_batches)?;
                let array = arr_ref
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8 join key array"
                        )
                    })?
                    .clone();
                BuildKeys::Utf8 {
                    array,
                    hashes: flat_hashes,
                }
            }
            KeyKind::LargeUtf8 => {
                let arr_ref = concat_single_key(&mut key_batches)?;
                let array = arr_ref
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected LargeUtf8 join key array"
                        )
                    })?
                    .clone();
                BuildKeys::LargeUtf8 {
                    array,
                    hashes: flat_hashes,
                }
            }
            KeyKind::Utf8View => {
                let arr_ref = concat_single_key(&mut key_batches)?;
                let array = arr_ref
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Utf8View join key array"
                        )
                    })?
                    .clone();
                BuildKeys::Utf8View {
                    array,
                    hashes: flat_hashes,
                }
            }
            KeyKind::Rows => {
                let fields = meta
                    .key_types
                    .iter()
                    .map(|dt| SortField::new(dt.clone()))
                    .collect();
                let converter = RowConverter::new(fields)?;
                let mut concat_cols = Vec::with_capacity(num_key_cols);
                for col_idx in 0..num_key_cols {
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
                BuildKeys::Rows {
                    converter,
                    rows,
                    hashes: flat_hashes,
                }
            }
            _ => unreachable!(),
        };
        drop(key_batches);
        let mut buckets = vec![0u32; bucket_count];
        let mut next = vec![0u32; num_rows];
        let mut row_idx = 0usize;
        for chunk in &hash_chunks {
            for &h in chunk {
                let bucket = (h as usize) & mask;
                let entry = (row_idx + 1) as u32;
                next[row_idx] = buckets[bucket];
                buckets[bucket] = entry;
                row_idx += 1;
            }
        }
        (build_keys, buckets, next)
    };
    drop(hash_chunks);

    let matched: Vec<AtomicU64> = (0..matched_words).map(|_| AtomicU64::new(0)).collect();
    let final_bytes = get_record_batch_memory_size(&payload)
        .saturating_add(build_keys.memory_bytes())
        .saturating_add(dir_bytes)
        .saturating_add(matched_bytes);
    reservation.try_resize(final_bytes)?;

    Ok(SingleHashTable {
        num_rows,
        hash_entries: num_rows,
        has_null_key: false,
        payload,
        keys: build_keys,
        buckets,
        mask,
        next,
        matched,
        _reservation: reservation,
    })
}

/// Read all `SpillFile`s for one spilled build partition and construct its
/// [`SingleHashTable`], returning `Err(ResourcesExhausted)` cleanly if the
/// partition exceeds `MemoryPool`.
pub(super) async fn load_spilled_build_partition_sync_or_async(
    files: &[Arc<dyn SpillFile>],
    spill_manager: &SpillManager,
    meta: &RadixBuildMetadata,
    reservation: MemoryReservation,
    tx: &mpsc::Sender<RecordBatch>,
) -> Result<Option<SingleHashTable>> {
    let mut batches = Vec::new();
    let mut tracked_bytes = 0usize;
    for sf in files {
        if tx.is_closed() {
            return Ok(None);
        }
        let mut stream =
            spill_manager.read_spill_as_stream_unbuffered(Arc::clone(sf), None)?;
        while let Some(batch_res) = stream.next().await {
            if tx.is_closed() {
                return Ok(None);
            }
            let batch = batch_res?;
            if batch.num_rows() == 0 {
                continue;
            }
            let bytes = get_record_batch_memory_size(&batch);
            reservation.try_grow(bytes)?;
            tracked_bytes = tracked_bytes.saturating_add(bytes);
            batches.push(batch);
        }
    }
    if tx.is_closed() {
        return Ok(None);
    }
    let table =
        build_single_table_from_packed_batches(&batches, meta, None, reservation)?;
    drop(batches);
    let _ = tracked_bytes;
    Ok(Some(table))
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
    interleave_build: Vec<(usize, usize)>,
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

impl ProbeKeys {
    #[inline]
    fn hash_at(&self, key_idx: usize) -> u64 {
        match self {
            Self::Int32(a) => mix_u64(a.value(key_idx) as u32 as u64),
            Self::Int64(a) => mix_u64(a.value(key_idx) as u64),
            Self::UInt32(a) => mix_u64(a.value(key_idx) as u64),
            Self::UInt64(a) => mix_u64(a.value(key_idx)),
            Self::Utf8 { hashes, .. }
            | Self::LargeUtf8 { hashes, .. }
            | Self::Utf8View { hashes, .. }
            | Self::Rows { hashes, .. } => hashes[key_idx],
            Self::Cross | Self::EmptyBuild => 0,
        }
    }

    fn clone_cheap(&self) -> Self {
        match self {
            Self::Cross => Self::Cross,
            Self::EmptyBuild => Self::EmptyBuild,
            Self::Int32(a) => Self::Int32(a.clone()),
            Self::Int64(a) => Self::Int64(a.clone()),
            Self::UInt32(a) => Self::UInt32(a.clone()),
            Self::UInt64(a) => Self::UInt64(a.clone()),
            Self::Utf8 { array, hashes } => Self::Utf8 {
                array: array.clone(),
                hashes: hashes.clone(),
            },
            Self::LargeUtf8 { array, hashes } => Self::LargeUtf8 {
                array: array.clone(),
                hashes: hashes.clone(),
            },
            Self::Utf8View { array, hashes } => Self::Utf8View {
                array: array.clone(),
                hashes: hashes.clone(),
            },
            Self::Rows { .. } => {
                unreachable!("ProbeKeys::Rows uses direct partition probing")
            }
        }
    }
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

pub(super) struct SingleProbeCursor {
    batch: RecordBatch,
    keys: ProbeKeys,
    valid_mask: Option<BooleanBuffer>,
    rows: ProbeRows,
    pos: usize,
    next_entry: u32,
    row_matched: bool,
    _reservation: MemoryReservation,
}

impl SingleProbeCursor {
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

pub(super) struct RadixProbeCursor {
    batch: RecordBatch,
    keys: ProbeKeys,
    valid_mask: Option<BooleanBuffer>,
    rows: ProbeRows,
    resident: Arc<ResidentRadixPartitions>,
    shift: u32,
    radix_bits: u32,
    resident_mask: u64,
    spilled_build_mask: u64,
    pos: usize,
    next_entry: u32,
    row_matched: bool,
    /// Populated only when `probe.filter.is_some()` to reuse `SingleProbeCursor`.
    filtered_buckets: Option<Vec<Vec<u32>>>,
    filtered_unmatched: Vec<u32>,
    filtered_unmatched_pos: usize,
    filtered_part: usize,
    filtered_part_cursor: Option<SingleProbeCursor>,
    _reservation: MemoryReservation,
}

impl RadixProbeCursor {
    fn is_exhausted(&self) -> bool {
        if let Some(buckets) = &self.filtered_buckets {
            self.filtered_unmatched_pos >= self.filtered_unmatched.len()
                && self.filtered_part >= buckets.len()
                && self.filtered_part_cursor.is_none()
        } else {
            match &self.rows {
                ProbeRows::All { num_rows } | ProbeRows::Mask { num_rows, .. } => {
                    self.pos >= *num_rows
                }
                ProbeRows::BackingIndices(indices) => self.pos >= indices.len(),
                ProbeRows::CompactedKeys { backing_rows } => {
                    self.pos >= backing_rows.len()
                }
            }
        }
    }
}

pub(super) enum ProbeCursor {
    Single(SingleProbeCursor),
    Radix(Box<RadixProbeCursor>),
}

#[cfg(test)]
impl ProbeCursor {
    fn batch(&self) -> &RecordBatch {
        match self {
            Self::Single(c) => &c.batch,
            Self::Radix(c) => &c.batch,
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

    pub(super) fn materialize_single_build_emission_batch(
        &self,
        table: &SingleHashTable,
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

    /// Materialize an entire payload batch as unmatched build rows (used for
    /// NULL-key build rows and probe-empty spilled partitions in `BuildOuter` /
    /// `FullOuter` / `BuildAnti`).
    pub(super) fn materialize_unmatched_build_payload_batch(
        &self,
        payload: &RecordBatch,
    ) -> Result<RecordBatch> {
        let len = payload.num_rows();
        if self.output_projection.is_empty() {
            return Ok(RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?);
        }
        let mut columns = Vec::with_capacity(self.output_projection.len());
        for (out_idx, col) in self.output_projection.iter().enumerate() {
            let arr = match col {
                JoinOutputCol::Build(b_idx) => Arc::clone(payload.column(*b_idx)),
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

    /// Materialize unmatched probe rows at `probe_indices` (NULL-padded build
    /// columns for `ProbeOuter`/`FullOuter`, probe-only for `ProbeAnti`).
    pub(super) fn materialize_unmatched_probe_indices_batch(
        &self,
        probe_batch: &RecordBatch,
        probe_indices: &[u32],
    ) -> Result<RecordBatch> {
        let len = probe_indices.len();
        if self.output_projection.is_empty() {
            return Ok(RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?);
        }
        let idx_arr = UInt32Array::from(probe_indices.to_vec());
        let mut columns = Vec::with_capacity(self.output_projection.len());
        for (out_idx, col) in self.output_projection.iter().enumerate() {
            let arr = match col {
                JoinOutputCol::Build(_) => {
                    new_null_array(self.output_schema.field(out_idx).data_type(), len)
                }
                JoinOutputCol::Probe(p_idx) => {
                    take(probe_batch.column(*p_idx).as_ref(), &idx_arr, None)?
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
    ) -> SingleProbeCursor {
        let (batch, selection) = view.into_parts();
        let num_rows = batch.num_rows();
        let rows = match selection {
            Selection::All => ProbeRows::All { num_rows },
            Selection::Mask { bits, .. } => ProbeRows::Mask { bits, num_rows },
            Selection::Indices(indices) => ProbeRows::BackingIndices(indices),
        };
        SingleProbeCursor {
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

    #[cfg(test)]
    pub(super) fn start_probe_view(
        &self,
        view: BatchView,
        table: &FinalizedHashTable,
        probe_reservation: &MemoryReservation,
    ) -> Result<Option<ProbeCursor>> {
        let resident = table.clone_resident();
        self.start_probe_view_with_spill(
            view,
            table,
            resident.as_ref(),
            None,
            None,
            probe_reservation,
        )
    }

    pub(super) fn start_probe_view_with_spill(
        &self,
        view: BatchView,
        table: &FinalizedHashTable,
        resident: Option<&Arc<ResidentRadixPartitions>>,
        local_probe_spill: Option<&mut LocalProbeSpill>,
        probe_spill_env: Option<&ProbeSpillEnv>,
        probe_reservation: &MemoryReservation,
    ) -> Result<Option<ProbeCursor>> {
        if view.is_empty() || self.short_circuits_on_table(table) {
            return Ok(None);
        }

        match &table.kind {
            FinalizedHashTableKind::Single(single) => self
                .start_single_table_probe(view, single, probe_reservation)
                .map(|opt| opt.map(ProbeCursor::Single)),
            FinalizedHashTableKind::Radix(radix) => self
                .start_radix_probe_view(
                    view,
                    radix,
                    resident.expect("radix resident partitions"),
                    local_probe_spill,
                    probe_spill_env,
                    probe_reservation,
                )
                .map(|opt| opt.map(|c| ProbeCursor::Radix(Box::new(c)))),
        }
    }

    pub(super) fn start_single_table_probe(
        &self,
        view: BatchView,
        table: &SingleHashTable,
        probe_reservation: &MemoryReservation,
    ) -> Result<Option<SingleProbeCursor>> {
        if view.is_empty() {
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
            return Ok(Some(SingleProbeCursor {
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
        Ok(Some(SingleProbeCursor {
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

    fn start_radix_probe_view(
        &self,
        view: BatchView,
        radix: &RadixHashTable,
        resident: &Arc<ResidentRadixPartitions>,
        mut local_probe_spill: Option<&mut LocalProbeSpill>,
        probe_spill_env: Option<&ProbeSpillEnv>,
        probe_reservation: &MemoryReservation,
    ) -> Result<Option<RadixProbeCursor>> {
        let mut cursor_reservation = probe_reservation.new_empty();
        let num_parts = 1usize << radix.radix_bits;
        let shift = radix.shift;
        let radix_bits = radix.radix_bits;
        let resident_mask = radix.resident_mask;
        let spilled_build_mask = radix.spilled_build_mask;
        let keep_null_probe = self.kind.keeps_null_probe_keys();
        let emit_unmatched_probe = self.kind.emits_unmatched_probe();

        // Build zero-copy `(batch, keys, valid_mask, rows)` matching `SingleProbeCursor`.
        let is_flat_int = matches!(
            self.key_kind,
            KeyKind::Int32 | KeyKind::Int64 | KeyKind::UInt32 | KeyKind::UInt64
        );
        let (batch, keys, valid_mask, rows) = if view.is_all() {
            let batch = view.into_parts().0;
            let num_rows = batch.num_rows();
            if num_rows == 0 {
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
            let keys =
                self.build_probe_keys_for_meta(&key_arrays, num_rows, &radix.meta)?;
            (batch, keys, valid_mask, ProbeRows::All { num_rows })
        } else if is_flat_int && let Some(key_col) = self.plain_single_key_col {
            let view = view.into_reserved_for_retention(&mut cursor_reservation)?;
            let (batch, selection) = view.into_parts();
            let num_rows = batch.num_rows();
            let key_arr = batch.column(key_col);
            let keys = self.build_probe_keys_for_meta(
                std::slice::from_ref(key_arr),
                num_rows,
                &radix.meta,
            )?;
            let (rows, valid_mask) = match selection {
                Selection::All => unreachable!(),
                Selection::Mask { bits, .. } => {
                    if key_arr.null_count() > 0 {
                        let nulls = key_arr.nulls().unwrap().inner();
                        if keep_null_probe {
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
            (batch, keys, valid_mask, rows)
        } else {
            let mat = view.materialize()?;
            let num_rows = mat.num_rows();
            if num_rows == 0 {
                return Ok(None);
            }
            let bytes = get_record_batch_memory_size(&mat);
            let _ = cursor_reservation.try_grow(bytes);
            let mut key_arrays = Vec::with_capacity(self.probe_key_exprs.len());
            for expr in &self.probe_key_exprs {
                key_arrays.push(expr.evaluate(&mat)?.into_array(num_rows)?);
            }
            let valid_mask = if self.null_equality == NullEquality::NullEqualsNull {
                None
            } else {
                valid_key_mask(&key_arrays)
            };
            let keys =
                self.build_probe_keys_for_meta(&key_arrays, num_rows, &radix.meta)?;
            (mat, keys, valid_mask, ProbeRows::All { num_rows })
        };

        // Route probe rows belonging to spilled build partitions `p ∈ S` into
        // worker-local synchronous `LocalProbeSpill`.
        if spilled_build_mask != 0 {
            let mut spill_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_parts];
            let mut pos = 0usize;
            let row_len = match &rows {
                ProbeRows::All { num_rows } | ProbeRows::Mask { num_rows, .. } => {
                    *num_rows
                }
                ProbeRows::BackingIndices(idx) => idx.len(),
                ProbeRows::CompactedKeys { backing_rows } => backing_rows.len(),
            };
            while pos < row_len {
                if let Some((key_idx, backing_row)) = probe_row_at(&rows, pos)
                    && valid_mask.as_ref().is_none_or(|v| v.value(key_idx))
                {
                    let h = keys.hash_at(key_idx);
                    let p = radix_partition(h, shift, radix_bits);
                    if ((spilled_build_mask >> p) & 1) != 0 {
                        spill_buckets[p].push(backing_row);
                    }
                }
                pos += 1;
            }
            let env = probe_spill_env.expect("probe spill env for spilled join");
            let lps = local_probe_spill
                .as_deref_mut()
                .expect("local probe spill for spilled join");
            for (p, p_rows) in spill_buckets.into_iter().enumerate() {
                if p_rows.is_empty() {
                    continue;
                }
                let idx_arr = UInt32Array::from(p_rows);
                let sub = take_record_batch(&batch, &idx_arr)?;
                lps.push_batch(p, sub, env, probe_reservation)?;
            }
        }

        // Fast path: equi-joins without residual filter probe directly over
        // `rows` with ZERO 64-bucket Vec allocations!
        if self.filter.is_none() {
            if resident_mask == 0 && !emit_unmatched_probe {
                return Ok(None);
            }
            return Ok(Some(RadixProbeCursor {
                batch,
                keys,
                valid_mask,
                rows,
                resident: Arc::clone(resident),
                shift,
                radix_bits,
                resident_mask,
                spilled_build_mask,
                pos: 0,
                next_entry: 0,
                row_matched: false,
                filtered_buckets: None,
                filtered_unmatched: Vec::new(),
                filtered_unmatched_pos: 0,
                filtered_part: 0,
                filtered_part_cursor: None,
                _reservation: cursor_reservation,
            }));
        }

        // Residual non-equi filter path: bucket resident rows by partition so
        // `SingleProbeCursor::next_chunk_filtered` handles arbitrary filters.
        let mut filtered_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_parts];
        let mut filtered_unmatched: Vec<u32> = Vec::new();
        let row_len = match &rows {
            ProbeRows::All { num_rows } | ProbeRows::Mask { num_rows, .. } => *num_rows,
            ProbeRows::BackingIndices(idx) => idx.len(),
            ProbeRows::CompactedKeys { backing_rows } => backing_rows.len(),
        };
        for pos in 0..row_len {
            let Some((key_idx, backing_row)) = probe_row_at(&rows, pos) else {
                continue;
            };
            if valid_mask.as_ref().is_some_and(|v| !v.value(key_idx)) {
                if keep_null_probe {
                    filtered_unmatched.push(backing_row);
                }
                continue;
            }
            let h = keys.hash_at(key_idx);
            let p = radix_partition(h, shift, radix_bits);
            if ((resident_mask >> p) & 1) != 0 {
                filtered_buckets[p].push(backing_row);
            } else if ((spilled_build_mask >> p) & 1) == 0 && emit_unmatched_probe {
                filtered_unmatched.push(backing_row);
            }
        }
        let has_work = filtered_buckets.iter().any(|b| !b.is_empty())
            || !filtered_unmatched.is_empty();
        if !has_work {
            return Ok(None);
        }

        Ok(Some(RadixProbeCursor {
            batch,
            keys,
            valid_mask,
            rows,
            resident: Arc::clone(resident),
            shift,
            radix_bits,
            resident_mask,
            spilled_build_mask,
            pos: 0,
            next_entry: 0,
            row_matched: false,
            filtered_buckets: Some(filtered_buckets),
            filtered_unmatched,
            filtered_unmatched_pos: 0,
            filtered_part: 0,
            filtered_part_cursor: None,
            _reservation: cursor_reservation,
        }))
    }

    fn build_probe_keys_for_meta(
        &self,
        key_arrays: &[ArrayRef],
        num_rows: usize,
        meta: &RadixBuildMetadata,
    ) -> Result<ProbeKeys> {
        match self.key_kind {
            KeyKind::Int32 => {
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
            KeyKind::Int64 => {
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
            KeyKind::UInt32 => {
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
            KeyKind::UInt64 => {
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
            KeyKind::Utf8 => {
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
            KeyKind::LargeUtf8 => {
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
            KeyKind::Utf8View => {
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
            KeyKind::Rows => {
                let converter = meta.row_converter.as_ref().ok_or_else(|| {
                    datafusion_common::internal_datafusion_err!(
                        "Missing RowConverter on RadixBuildMetadata"
                    )
                })?;
                let normalized =
                    normalize_key_arrays(key_arrays.to_vec(), self.has_floats);
                let mut hashes = vec![0u64; num_rows];
                create_hashes(&normalized, &join_random_state(), &mut hashes)?;
                let rows = converter.convert_columns(&normalized)?;
                Ok(ProbeKeys::Rows { rows, hashes })
            }
            KeyKind::Cross => {
                internal_err!("Cross join cannot use radix probe")
            }
        }
    }

    fn start_probe_all(
        &self,
        batch: RecordBatch,
        table: &SingleHashTable,
        reservation: MemoryReservation,
    ) -> Result<Option<SingleProbeCursor>> {
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

        Ok(Some(SingleProbeCursor {
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
        table: &SingleHashTable,
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
        match (cursor, &table.kind) {
            (ProbeCursor::Single(sc), FinalizedHashTableKind::Single(st)) => {
                self.next_chunk_single(sc, st, batch_size, scratch)
            }
            (ProbeCursor::Radix(rc), FinalizedHashTableKind::Radix(_)) => {
                self.next_chunk_radix(rc, batch_size, scratch)
            }
            _ => internal_err!("Morsel HashJoin cursor/table kind mismatch"),
        }
    }

    pub(super) fn next_chunk_single(
        &self,
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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

    fn next_chunk_radix(
        &self,
        cursor: &mut RadixProbeCursor,
        batch_size: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let limit = batch_size.max(1);
        let num_parts = cursor.resident.partitions.len();

        // Residual non-equi filter path: step through each non-empty resident
        // partition using SingleProbeCursor and coalesce up to `limit` rows.
        if self.filter.is_some() {
            return self.next_chunk_radix_filtered(cursor, limit, scratch);
        }

        let shift = cursor.shift;
        let radix_bits = cursor.radix_bits;
        let resident_mask = cursor.resident_mask;
        let spilled_build_mask = cursor.spilled_build_mask;
        let keep_null_probe = self.kind.keeps_null_probe_keys();
        let emit_unmatched_probe = self.kind.emits_unmatched_probe();

        // BuildSemi / BuildAnti: mark build-side matched bits directly with
        // ZERO bucket allocations and ZERO output during P1.
        if matches!(
            self.kind,
            ProbeJoinKind::BuildSemi | ProbeJoinKind::BuildAnti
        ) {
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
                let h = cursor.keys.hash_at(key_idx);
                let p = radix_partition(h, shift, radix_bits);
                if ((resident_mask >> p) & 1) != 0
                    && let Some(part_table) = cursor.resident.partitions[p].as_ref()
                {
                    probe_mark_all_matches(&cursor.keys, part_table, key_idx)?;
                }
            }
            return Ok(OperatorStep::NeedMoreInput(None));
        }

        // ProbeSemi / ProbeAnti: direct radix lookup, probe-only output.
        if matches!(
            self.kind,
            ProbeJoinKind::ProbeSemi | ProbeJoinKind::ProbeAnti { .. }
        ) {
            let is_semi = matches!(self.kind, ProbeJoinKind::ProbeSemi);
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
                let h = cursor.keys.hash_at(key_idx);
                let p = radix_partition(h, shift, radix_bits);
                if ((spilled_build_mask >> p) & 1) != 0 {
                    continue;
                }
                let matched = if ((resident_mask >> p) & 1) != 0
                    && let Some(part_table) = cursor.resident.partitions[p].as_ref()
                {
                    probe_has_any_match(&cursor.keys, part_table, key_idx)?
                } else {
                    false
                };
                if matched == is_semi {
                    scratch.probe_indices.push(backing_row);
                }
            }

            let len = scratch.probe_indices.len();
            if len == 0 {
                return Ok(OperatorStep::NeedMoreInput(None));
            }
            let out = if self.output_projection.is_empty() {
                scratch.probe_indices.clear();
                RecordBatch::try_new_with_options(
                    Arc::clone(&self.output_schema),
                    vec![],
                    &RecordBatchOptions::new().with_row_count(Some(len)),
                )?
            } else {
                let probe_idx = UInt32Array::new(
                    ScalarBuffer::from(std::mem::take(&mut scratch.probe_indices)),
                    None,
                );
                let mut cols = Vec::with_capacity(self.output_projection.len());
                for col in &self.output_projection {
                    let JoinOutputCol::Probe(idx) = col else {
                        return internal_err!(
                            "ProbeSemi/ProbeAnti cannot project build columns"
                        );
                    };
                    cols.push(take(
                        cursor.batch.column(*idx).as_ref(),
                        &probe_idx,
                        None,
                    )?);
                }
                if let Ok(mut v) = probe_idx.into_parts().1.into_inner().into_vec::<u32>()
                {
                    v.clear();
                    scratch.probe_indices = v;
                }
                RecordBatch::try_new(Arc::clone(&self.output_schema), cols)?
            };
            return Ok(if cursor.is_exhausted() {
                OperatorStep::NeedMoreInput(Some(out))
            } else {
                OperatorStep::HaveMoreOutput(out)
            });
        }

        // Inner / BuildOuter / ProbeOuter / FullOuter equi-join fast path:
        // Direct radix probe with ZERO bucket allocations!
        let mark_build = matches!(
            self.kind,
            ProbeJoinKind::BuildOuter | ProbeJoinKind::FullOuter
        );
        scratch.interleave_build.clear();
        scratch.probe_indices.clear();
        scratch.interleave_build.reserve(limit);
        reserve_indices(&mut scratch.probe_indices, limit)?;
        let mut has_null_build = false;
        let mut single_part_hit: Option<usize> = None;
        let mut multi_part_hit = false;

        while !cursor.is_exhausted() && scratch.probe_indices.len() < limit {
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
                if keep_null_probe {
                    scratch.interleave_build.push((num_parts, 0));
                    scratch.probe_indices.push(backing_row);
                    has_null_build = true;
                }
                cursor.pos += 1;
                continue;
            }
            let h = cursor.keys.hash_at(key_idx);
            let p = radix_partition(h, shift, radix_bits);
            if ((spilled_build_mask >> p) & 1) != 0 {
                cursor.pos += 1;
                cursor.next_entry = 0;
                cursor.row_matched = false;
                continue;
            }
            if ((resident_mask >> p) & 1) == 0 {
                if emit_unmatched_probe {
                    scratch.interleave_build.push((num_parts, 0));
                    scratch.probe_indices.push(backing_row);
                    has_null_build = true;
                }
                cursor.pos += 1;
                cursor.next_entry = 0;
                cursor.row_matched = false;
                continue;
            }
            let part_table = cursor.resident.partitions[p]
                .as_ref()
                .expect("resident partition present");
            let rem = limit - scratch.probe_indices.len();
            let continued = cursor.next_entry != 0;
            scratch.build_indices.clear();
            scratch.cand_probe.clear();
            let chain_done = probe_step_key_candidates(
                &cursor.keys,
                part_table,
                key_idx,
                backing_row,
                &mut cursor.next_entry,
                rem,
                false,
                &mut scratch.build_indices,
                &mut scratch.cand_probe,
            )?;
            let added = scratch.build_indices.len();
            if added > 0 {
                match single_part_hit {
                    None => single_part_hit = Some(p),
                    Some(prev) if prev != p => multi_part_hit = true,
                    _ => {}
                }
                for &b in &scratch.build_indices {
                    if mark_build {
                        part_table.mark_matched(b);
                    }
                    scratch.interleave_build.push((p, b as usize));
                    scratch.probe_indices.push(backing_row);
                }
                if chain_done {
                    cursor.pos += 1;
                    cursor.next_entry = 0;
                    cursor.row_matched = false;
                } else {
                    cursor.row_matched = true;
                    break;
                }
            } else {
                if emit_unmatched_probe && (!continued || !cursor.row_matched) {
                    scratch.interleave_build.push((num_parts, 0));
                    scratch.probe_indices.push(backing_row);
                    has_null_build = true;
                }
                cursor.pos += 1;
                cursor.next_entry = 0;
                cursor.row_matched = false;
            }
        }

        let len = scratch.probe_indices.len();
        if len == 0 {
            return Ok(OperatorStep::NeedMoreInput(None));
        }
        let out = if self.output_projection.is_empty() {
            scratch.interleave_build.clear();
            scratch.probe_indices.clear();
            RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?
        } else {
            let probe_idx = UInt32Array::new(
                ScalarBuffer::from(std::mem::take(&mut scratch.probe_indices)),
                None,
            );
            let mut cols = Vec::with_capacity(self.output_projection.len());
            for (out_idx, col) in self.output_projection.iter().enumerate() {
                let arr = match col {
                    JoinOutputCol::Probe(idx) => {
                        take(cursor.batch.column(*idx).as_ref(), &probe_idx, None)?
                    }
                    JoinOutputCol::Build(b_idx) => {
                        let dt = self.output_schema.field(out_idx).data_type();
                        if !has_null_build && !multi_part_hit {
                            if let Some(p) = single_part_hit {
                                let pt = cursor.resident.partitions[p]
                                    .as_ref()
                                    .expect("hit partition exists");
                                let b_arr = UInt32Array::from_iter_values(
                                    scratch
                                        .interleave_build
                                        .iter()
                                        .map(|&(_, r)| r as u32),
                                );
                                take(pt.payload.column(*b_idx).as_ref(), &b_arr, None)?
                            } else {
                                new_null_array(dt, len)
                            }
                        } else {
                            let empty_holder = new_empty_array(dt);
                            let null_holder = new_null_array(dt, 1);
                            let mut arr_refs: Vec<&dyn Array> =
                                Vec::with_capacity(num_parts + 1);
                            for p in 0..num_parts {
                                if let Some(pt) = cursor.resident.partitions[p].as_ref() {
                                    arr_refs.push(pt.payload.column(*b_idx).as_ref());
                                } else {
                                    arr_refs.push(empty_holder.as_ref());
                                }
                            }
                            arr_refs.push(null_holder.as_ref());
                            let raw = interleave(&arr_refs, &scratch.interleave_build)?;
                            match raw.data_type() {
                                DataType::Utf8View => {
                                    let sv = raw
                                        .as_any()
                                        .downcast_ref::<StringViewArray>()
                                        .unwrap();
                                    if sv.data_buffers().len() > 2 {
                                        Arc::new(sv.gc()) as ArrayRef
                                    } else {
                                        raw
                                    }
                                }
                                DataType::BinaryView => {
                                    let bv = raw
                                        .as_any()
                                        .downcast_ref::<arrow::array::BinaryViewArray>()
                                        .unwrap();
                                    if bv.data_buffers().len() > 2 {
                                        Arc::new(bv.gc()) as ArrayRef
                                    } else {
                                        raw
                                    }
                                }
                                _ => raw,
                            }
                        }
                    }
                };
                cols.push(arr);
            }
            if let Ok(mut v) = probe_idx.into_parts().1.into_inner().into_vec::<u32>() {
                v.clear();
                scratch.probe_indices = v;
            }
            scratch.interleave_build.clear();
            RecordBatch::try_new(Arc::clone(&self.output_schema), cols)?
        };

        Ok(if cursor.is_exhausted() {
            OperatorStep::NeedMoreInput(Some(out))
        } else {
            OperatorStep::HaveMoreOutput(out)
        })
    }

    fn next_chunk_radix_filtered(
        &self,
        cursor: &mut RadixProbeCursor,
        limit: usize,
        scratch: &mut ProbeScratch,
    ) -> Result<OperatorStep> {
        let num_parts = cursor.resident.partitions.len();
        let mut chunks: Vec<RecordBatch> = Vec::new();
        let mut total_rows = 0usize;

        if cursor.filtered_unmatched_pos < cursor.filtered_unmatched.len()
            && self.kind.emits_unmatched_probe()
        {
            let take_n = (cursor.filtered_unmatched.len()
                - cursor.filtered_unmatched_pos)
                .min(limit - total_rows);
            let slice = &cursor.filtered_unmatched
                [cursor.filtered_unmatched_pos..cursor.filtered_unmatched_pos + take_n];
            cursor.filtered_unmatched_pos += take_n;
            let b =
                self.materialize_unmatched_probe_indices_batch(&cursor.batch, slice)?;
            total_rows += b.num_rows();
            chunks.push(b);
        } else {
            cursor.filtered_unmatched_pos = cursor.filtered_unmatched.len();
        }

        while total_rows < limit {
            if let Some(mut sc) = cursor.filtered_part_cursor.take() {
                let p = cursor.filtered_part;
                let part_table = cursor.resident.partitions[p]
                    .as_ref()
                    .expect("active filtered partition exists");
                let step = self.next_chunk_single(
                    &mut sc,
                    part_table,
                    limit - total_rows,
                    scratch,
                )?;
                match step {
                    OperatorStep::NeedMoreInput(opt) => {
                        if let Some(b) = opt
                            && b.num_rows() > 0
                        {
                            total_rows += b.num_rows();
                            chunks.push(b);
                        }
                        cursor.filtered_part += 1;
                    }
                    OperatorStep::HaveMoreOutput(b) => {
                        if b.num_rows() > 0 {
                            chunks.push(b);
                        }
                        cursor.filtered_part_cursor = Some(sc);
                        break;
                    }
                }
                continue;
            }
            if cursor.filtered_part >= num_parts {
                break;
            }
            let p = cursor.filtered_part;
            let rows = cursor
                .filtered_buckets
                .as_ref()
                .map(|b| b[p].as_slice())
                .unwrap_or(&[]);
            if rows.is_empty() || cursor.resident.partitions[p].is_none() {
                cursor.filtered_part += 1;
                continue;
            }
            let part_table = cursor.resident.partitions[p].as_ref().unwrap();
            let idx_arr = UInt32Array::from(rows.to_vec());
            let sub_keys = match &cursor.keys {
                ProbeKeys::Rows { .. } => {
                    let sub_batch = take_record_batch(&cursor.batch, &idx_arr)?;
                    let mut ks = Vec::with_capacity(self.probe_key_exprs.len());
                    for expr in &self.probe_key_exprs {
                        ks.push(
                            expr.evaluate(&sub_batch)?
                                .into_array(sub_batch.num_rows())?,
                        );
                    }
                    let pk =
                        self.build_probe_keys(&ks, sub_batch.num_rows(), part_table)?;
                    let sc = SingleProbeCursor {
                        batch: sub_batch,
                        keys: pk,
                        valid_mask: None,
                        rows: ProbeRows::All {
                            num_rows: rows.len(),
                        },
                        pos: 0,
                        next_entry: 0,
                        row_matched: false,
                        _reservation: cursor._reservation.new_empty(),
                    };
                    cursor.filtered_part_cursor = Some(sc);
                    continue;
                }
                other => other.clone_cheap(),
            };
            let sc = SingleProbeCursor {
                batch: cursor.batch.clone(),
                keys: sub_keys,
                valid_mask: None,
                rows: ProbeRows::BackingIndices(idx_arr),
                pos: 0,
                next_entry: 0,
                row_matched: false,
                _reservation: cursor._reservation.new_empty(),
            };
            cursor.filtered_part_cursor = Some(sc);
        }

        if chunks.is_empty() {
            return Ok(OperatorStep::NeedMoreInput(None));
        }
        let out = if chunks.len() == 1 {
            chunks.pop().unwrap()
        } else {
            super::sort::concat_batches_compact_views(&self.output_schema, &chunks)?
        };
        Ok(if cursor.is_exhausted() {
            OperatorStep::NeedMoreInput(Some(out))
        } else {
            OperatorStep::HaveMoreOutput(out)
        })
    }

    /// Recursively re-partition one overflowing spilled build+probe partition
    /// pair on `next_shift` (`RECURSIVE_RADIX_BITS = 4`, 16 sub-partitions).
    /// Returns `Ok(Some(sub_pairs))` if splitting made progress, or `Ok(None)`
    /// on non-progressing duplicate-key skew (caller runs BNL sub-pass).
    #[expect(clippy::too_many_arguments)]
    pub(super) async fn try_repartition_spilled_pair(
        &self,
        build_files: &[Arc<dyn SpillFile>],
        probe_files: &[Arc<dyn SpillFile>],
        build_mgr: &SpillManager,
        probe_mgr: &SpillManager,
        meta: &RadixBuildMetadata,
        shift: u32,
        depth: usize,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<Option<(u32, Vec<(Vec<Arc<dyn SpillFile>>, Vec<Arc<dyn SpillFile>>)>)>>
    {
        if depth >= MAX_RADIX_DEPTH {
            return Ok(None);
        }
        let Some(next_shift) = next_level_shift(shift, RECURSIVE_RADIX_BITS) else {
            return Ok(None);
        };
        let sub_parts = 1usize << RECURSIVE_RADIX_BITS;
        let mut sub_build: Vec<Vec<Arc<dyn SpillFile>>> = vec![Vec::new(); sub_parts];
        let mut sub_build_rows = vec![0usize; sub_parts];
        let mut total_build_rows = 0usize;
        let hash_col_idx = meta.num_payload_cols + meta.key_types.len();

        for sf in build_files {
            if tx.is_closed() {
                return Ok(None);
            }
            let mut stream =
                build_mgr.read_spill_as_stream_unbuffered(Arc::clone(sf), None)?;
            while let Some(batch_res) = stream.next().await {
                if tx.is_closed() {
                    return Ok(None);
                }
                let batch = batch_res?;
                if batch.num_rows() == 0 {
                    continue;
                }
                let hash_arr = batch
                    .column(hash_col_idx)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected UInt64 __morsel_hash column"
                        )
                    })?;
                let (perm, ranges) = radix_permute_indices(
                    hash_arr.values(),
                    next_shift,
                    RECURSIVE_RADIX_BITS,
                );
                let permuted = take_record_batch(&batch, &perm)?;
                drop(batch);
                for (sp, &(offset, len)) in ranges.iter().enumerate() {
                    if len == 0 {
                        continue;
                    }
                    let slice = permuted.slice(offset, len);
                    if let Some((out_sf, _)) = spill_batches_async(
                        build_mgr,
                        std::slice::from_ref(&slice),
                        "MorselHashJoinRecursiveBuildSpill",
                        tx,
                    )
                    .await?
                    {
                        sub_build[sp].push(out_sf);
                        sub_build_rows[sp] += len;
                        total_build_rows += len;
                    }
                }
            }
        }

        // Check if recursive split made meaningful progress (not >=95% in 1 bucket).
        let max_sub = sub_build_rows.iter().copied().max().unwrap_or(0);
        if total_build_rows == 0 || max_sub * 20 >= total_build_rows * 19 {
            return Ok(None);
        }

        // Re-partition probe files on the exact same `next_shift`.
        let mut sub_probe: Vec<Vec<Arc<dyn SpillFile>>> = vec![Vec::new(); sub_parts];
        for sf in probe_files {
            if tx.is_closed() {
                return Ok(None);
            }
            let mut stream =
                probe_mgr.read_spill_as_stream_unbuffered(Arc::clone(sf), None)?;
            while let Some(batch_res) = stream.next().await {
                if tx.is_closed() {
                    return Ok(None);
                }
                let batch = batch_res?;
                if batch.num_rows() == 0 {
                    continue;
                }
                let n = batch.num_rows();
                let mut key_arrays = Vec::with_capacity(self.probe_key_exprs.len());
                for expr in &self.probe_key_exprs {
                    key_arrays.push(expr.evaluate(&batch)?.into_array(n)?);
                }
                let keys = self.build_probe_keys_for_meta(&key_arrays, n, meta)?;
                let hashes: Vec<u64> = (0..n).map(|i| keys.hash_at(i)).collect();
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
                        probe_mgr,
                        std::slice::from_ref(&slice),
                        "MorselHashJoinRecursiveProbeSpill",
                        tx,
                    )
                    .await?
                    {
                        sub_probe[sp].push(out_sf);
                    }
                }
            }
        }

        let pairs = sub_build.into_iter().zip(sub_probe).collect();
        Ok(Some((next_shift, pairs)))
    }

    /// Probe one `probe_batch` against one build slice during a block-nested-loop
    /// (BNL) hash sub-pass on an unsplittable skewed partition.
    ///
    /// Updates `probe_matched` (1 bit per row across the partition's probe stream
    /// starting at `probe_row_offset`) for `ProbeOuter`/`FullOuter`/`ProbeSemi`/`ProbeAnti`,
    /// emits matches (deduplicating `ProbeSemi` on first `0 -> 1` bit transition),
    /// and never emits unmatched probe rows until all build slices finish.
    pub(super) fn probe_bnl_slice_batch(
        &self,
        probe_batch: &RecordBatch,
        slice_table: &SingleHashTable,
        probe_row_offset: usize,
        probe_matched: &mut Vec<u64>,
        batch_size: usize,
        probe_reservation: &MemoryReservation,
        scratch: &mut ProbeScratch,
    ) -> Result<Vec<RecordBatch>> {
        let n = probe_batch.num_rows();
        if n == 0 || !slice_table.has_hash_entries() {
            return Ok(Vec::new());
        }
        let needed_words = (probe_row_offset + n).div_ceil(64);
        if probe_matched.len() < needed_words {
            probe_matched.resize(needed_words, 0);
        }

        let set_probe_bit = |pm: &mut [u64], local_row: usize| -> bool {
            let g = probe_row_offset + local_row;
            let word = &mut pm[g >> 6];
            let bit = 1u64 << (g & 63);
            let was_set = (*word & bit) != 0;
            *word |= bit;
            was_set
        };

        let mut key_arrays = Vec::with_capacity(self.probe_key_exprs.len());
        for expr in &self.probe_key_exprs {
            key_arrays.push(expr.evaluate(probe_batch)?.into_array(n)?);
        }
        let valid_mask = if self.null_equality == NullEquality::NullEqualsNull {
            None
        } else {
            valid_key_mask(&key_arrays)
        };
        let keys = self.build_probe_keys(&key_arrays, n, slice_table)?;
        let limit = batch_size.max(1);
        let mark_build = self.kind.needs_matched_bitset();

        let mut out_batches = Vec::new();

        // BuildSemi / BuildAnti: mark build-side matched bits only.
        if matches!(
            self.kind,
            ProbeJoinKind::BuildSemi | ProbeJoinKind::BuildAnti
        ) {
            if let Some(mut sc) = self.start_single_table_probe(
                BatchView::all(probe_batch.clone()),
                slice_table,
                probe_reservation,
            )? {
                loop {
                    match self.next_chunk_single(&mut sc, slice_table, limit, scratch)? {
                        OperatorStep::NeedMoreInput(_) => break,
                        OperatorStep::HaveMoreOutput(_) => {}
                    }
                }
            }
            return Ok(out_batches);
        }

        // ProbeSemi / ProbeAnti: determine matching probe rows (applying residual
        // filter if present), update `probe_matched`, emit `ProbeSemi` ONLY on
        // first `0 -> 1` bit transition.
        if matches!(
            self.kind,
            ProbeJoinKind::ProbeSemi | ProbeJoinKind::ProbeAnti { .. }
        ) {
            let is_semi = matches!(self.kind, ProbeJoinKind::ProbeSemi);
            let mut emit_rows = Vec::new();
            for r in 0..n {
                if valid_mask.as_ref().is_some_and(|v| !v.value(r)) {
                    continue;
                }
                let matched = if let Some(filter) = &self.filter {
                    let mut next_entry = 0u32;
                    let mut found = false;
                    loop {
                        scratch.cand_build.clear();
                        scratch.cand_probe.clear();
                        let done = probe_step_key_candidates(
                            &keys,
                            slice_table,
                            r,
                            r as u32,
                            &mut next_entry,
                            limit,
                            false,
                            &mut scratch.cand_build,
                            &mut scratch.cand_probe,
                        )?;
                        if !scratch.cand_build.is_empty() {
                            let b_idx = UInt32Array::from(scratch.cand_build.clone());
                            let p_idx = UInt32Array::from(scratch.cand_probe.clone());
                            let mask = filter.evaluate(
                                &slice_table.payload,
                                probe_batch,
                                &b_idx,
                                &p_idx,
                            )?;
                            if mask.true_count() > 0 {
                                found = true;
                                break;
                            }
                        }
                        if done {
                            break;
                        }
                    }
                    found
                } else {
                    probe_has_any_match(&keys, slice_table, r)?
                };
                if matched {
                    let was_set = set_probe_bit(probe_matched, r);
                    if is_semi && !was_set {
                        emit_rows.push(r as u32);
                    }
                }
            }
            if !emit_rows.is_empty() {
                let idx_arr = UInt32Array::from(emit_rows);
                let mut cols = Vec::with_capacity(self.output_projection.len());
                for col in &self.output_projection {
                    let JoinOutputCol::Probe(idx) = col else {
                        return internal_err!("ProbeSemi cannot project build columns");
                    };
                    cols.push(take(probe_batch.column(*idx).as_ref(), &idx_arr, None)?);
                }
                out_batches
                    .push(RecordBatch::try_new(Arc::clone(&self.output_schema), cols)?);
            }
            return Ok(out_batches);
        }

        // Inner / BuildOuter / ProbeOuter / FullOuter: collect ALL matching
        // (build_row, probe_row) pairs (evaluating residual filter if present),
        // mark build & probe matched bits, and emit matches in `limit` chunks.
        let mut build_rows = Vec::new();
        let mut probe_rows = Vec::new();
        for r in 0..n {
            if valid_mask.as_ref().is_some_and(|v| !v.value(r)) {
                continue;
            }
            let mut next_entry = 0u32;
            loop {
                scratch.cand_build.clear();
                scratch.cand_probe.clear();
                let done = probe_step_key_candidates(
                    &keys,
                    slice_table,
                    r,
                    r as u32,
                    &mut next_entry,
                    limit,
                    false,
                    &mut scratch.cand_build,
                    &mut scratch.cand_probe,
                )?;
                if !scratch.cand_build.is_empty() {
                    if let Some(filter) = &self.filter {
                        let b_idx = UInt32Array::from(scratch.cand_build.clone());
                        let p_idx = UInt32Array::from(scratch.cand_probe.clone());
                        let mask = filter.evaluate(
                            &slice_table.payload,
                            probe_batch,
                            &b_idx,
                            &p_idx,
                        )?;
                        for (i, (&b, &p)) in scratch
                            .cand_build
                            .iter()
                            .zip(scratch.cand_probe.iter())
                            .enumerate()
                        {
                            if mask.is_valid(i) && mask.value(i) {
                                if mark_build {
                                    slice_table.mark_matched(b);
                                }
                                set_probe_bit(probe_matched, p as usize);
                                build_rows.push(b);
                                probe_rows.push(p);
                            }
                        }
                    } else {
                        for (&b, &p) in
                            scratch.cand_build.iter().zip(scratch.cand_probe.iter())
                        {
                            if mark_build {
                                slice_table.mark_matched(b);
                            }
                            set_probe_bit(probe_matched, p as usize);
                            build_rows.push(b);
                            probe_rows.push(p);
                        }
                    }
                }
                if build_rows.len() >= limit {
                    out_batches.push(self.materialize_pair_batch(
                        &slice_table.payload,
                        probe_batch,
                        &build_rows,
                        &probe_rows,
                    )?);
                    build_rows.clear();
                    probe_rows.clear();
                }
                if done {
                    break;
                }
            }
        }
        if !build_rows.is_empty() {
            out_batches.push(self.materialize_pair_batch(
                &slice_table.payload,
                probe_batch,
                &build_rows,
                &probe_rows,
            )?);
        }
        Ok(out_batches)
    }

    fn materialize_pair_batch(
        &self,
        build_payload: &RecordBatch,
        probe_batch: &RecordBatch,
        build_rows: &[u32],
        probe_rows: &[u32],
    ) -> Result<RecordBatch> {
        let len = build_rows.len();
        if self.output_projection.is_empty() {
            return Ok(RecordBatch::try_new_with_options(
                Arc::clone(&self.output_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )?);
        }
        let b_idx = UInt32Array::from(build_rows.to_vec());
        let p_idx = UInt32Array::from(probe_rows.to_vec());
        let mut cols = Vec::with_capacity(self.output_projection.len());
        for col in &self.output_projection {
            let arr = match col {
                JoinOutputCol::Build(idx) => {
                    take(build_payload.column(*idx).as_ref(), &b_idx, None)?
                }
                JoinOutputCol::Probe(idx) => {
                    take(probe_batch.column(*idx).as_ref(), &p_idx, None)?
                }
            };
            cols.push(arr);
        }
        Ok(RecordBatch::try_new(Arc::clone(&self.output_schema), cols)?)
    }

    fn next_chunk_inner_no_filter(
        &self,
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &mut SingleProbeCursor,
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
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &mut SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &SingleProbeCursor,
        table: &SingleHashTable,
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
        cursor: &SingleProbeCursor,
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
    cursor: &mut SingleProbeCursor,
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
    table: &SingleHashTable,
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
    table: &SingleHashTable,
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
    cursor: &mut SingleProbeCursor,
    table: &SingleHashTable,
    key_idx: usize,
    backing_row: u32,
    max_cands: usize,
    skip_already_matched: bool,
    cand_build: &mut Vec<u32>,
    cand_probe: &mut Vec<u32>,
) -> Result<bool> {
    probe_step_key_candidates(
        &cursor.keys,
        table,
        key_idx,
        backing_row,
        &mut cursor.next_entry,
        max_cands,
        skip_already_matched,
        cand_build,
        cand_probe,
    )
}

#[expect(clippy::too_many_arguments)]
fn probe_step_key_candidates(
    keys: &ProbeKeys,
    table: &SingleHashTable,
    key_idx: usize,
    backing_row: u32,
    next_entry: &mut u32,
    max_cands: usize,
    skip_already_matched: bool,
    cand_build: &mut Vec<u32>,
    cand_probe: &mut Vec<u32>,
) -> Result<bool> {
    macro_rules! flat_step {
        ($p_arr:expr, $flat:expr, $hash_key:expr) => {{
            let dup_next = $flat.dup_next.as_slice();
            let mut added = 0usize;
            let mut entry = if *next_entry == 0 {
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
                    *next_entry = 0;
                    return Ok(true);
                };
                if !skip_already_matched || !table.is_matched(first_row) {
                    cand_build.push(first_row);
                    cand_probe.push(backing_row);
                    added += 1;
                }
                dup_head
            } else {
                *next_entry
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
            *next_entry = entry;
            return Ok(entry == 0);
        }};
    }
    macro_rules! chained_step {
        ($h:expr, $eq:expr) => {{
            let h = $h;
            let mut added = 0usize;
            let mut entry = if *next_entry == 0 {
                table.buckets[(h as usize) & table.mask]
            } else {
                *next_entry
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
            *next_entry = entry;
            return Ok(entry == 0);
        }};
    }
    match (keys, &table.keys) {
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
    cursor: &mut SingleProbeCursor,
    table: &SingleHashTable,
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
    cursor: &mut SingleProbeCursor,
    table: &SingleHashTable,
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
            assert_eq!(cursor.batch().num_rows(), n);
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
            assert_eq!(cursor.batch().num_rows(), n);
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
            assert_eq!(cursor.batch().num_rows(), 16);
            let compacted_bytes = get_record_batch_memory_size(cursor.batch());
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
            assert_eq!(cursor.batch().num_rows(), n);
            assert_eq!(pool_2k.reserved(), view_tiny.selection_memory_size());
        }
        assert_eq!(pool_2k.reserved(), 0);

        drop(table);
        assert_eq!(build_pool.reserved(), 0);
        Ok(())
    }
}
