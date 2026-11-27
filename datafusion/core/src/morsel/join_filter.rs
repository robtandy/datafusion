// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Dynamic join filter collection (per-key min/max bounds, small `InList`,
//! and cache-resident split-block Bloom filters) for native morsel hash joins.

#[cfg(any(feature = "parquet", test))]
use std::fmt::{self, Display, Formatter};
#[cfg(any(feature = "parquet", test))]
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::Arc;

#[cfg(any(feature = "parquet", test))]
use arrow::array::RecordBatch;
use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array,
    Date64Array, Decimal128Array, Decimal256Array, DurationMicrosecondArray,
    DurationMillisecondArray, DurationNanosecondArray, DurationSecondArray,
    FixedSizeBinaryArray, Float16Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, StringArray,
    StringViewArray, Time32MillisecondArray, Time32SecondArray, Time64MicrosecondArray,
    Time64NanosecondArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::buffer::BooleanBuffer;
use arrow::compute::{
    filter, max, max_binary, max_binary_view, max_boolean, max_string, max_string_view,
    min, min_binary, min_binary_view, min_boolean, min_string, min_string_view,
};
#[cfg(any(feature = "parquet", test))]
use arrow::datatypes::Schema;
use arrow::datatypes::{DataType, TimeUnit};
#[cfg(any(feature = "parquet", test))]
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::{NullEquality, Result, ScalarValue};
#[cfg(any(feature = "parquet", test))]
use datafusion_common::{float_semantics, internal_datafusion_err};
#[cfg(any(feature = "parquet", test))]
use datafusion_expr::{ColumnarValue, Operator as ExprOperator};
#[cfg(any(feature = "parquet", test))]
use datafusion_physical_expr::PhysicalExpr;
#[cfg(any(feature = "parquet", test))]
use datafusion_physical_expr::expressions::{
    BinaryExpr, InListExpr, IsNullExpr, Literal,
};

use super::hash_join::KeyKind;
#[cfg(any(feature = "parquet", test))]
use super::hash_join::{join_random_state, mix_u64, mix_u64_pair, valid_key_mask};
use super::radix::{DEFAULT_RADIX_BITS, level0_shift, radix_partition};

// Putze et al. / Parquet split-block Bloom filter odd multiplicative salts.
const SBBF_SALTS: [u32; 8] = [
    0x47b6_137b,
    0x4497_4d91,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// Target at least 12 bits per distinct key (~0.9% FPP before power-of-two rounding).
const MIN_BLOOM_BITS_PER_KEY: usize = 12;

/// Hard memory ceiling for a single dynamic join Bloom filter (8 MiB = 262,144 blocks).
const MAX_BLOOM_BYTES: usize = 8 * 1024 * 1024;

/// Skip Bloom construction above 5M build keys where an 8 MiB filter would lose selectivity.
const MAX_BLOOM_KEYS: usize = 5_000_000;

/// Skip Bloom construction on unfiltered base-table builds larger than this row
/// count (they typically span the full key domain and prune ~0% of probe rows).
const UNFILTERED_BLOOM_MAX_ROWS: usize = 16_384;

/// Cap for per-column distinct tracking on multi-key joins (enables exact
/// `InList` pushdown on low-cardinality keys such as `s_nationkey` in TPC-H Q5
/// when only one key of a composite join traces back to a scan).
const MAX_PER_KEY_INLIST: usize = 32;

/// Configuration carried onto [`HashJoinBuildSink`](super::hash_join::HashJoinBuildSink)
/// when its probe side can consume a dynamic join filter.
#[derive(Debug, Clone, Copy)]
pub(super) struct DynamicJoinFilterConfig {
    pub(super) max_inlist_values: usize,
    pub(super) max_inlist_bytes: usize,
    /// Build a split-block Bloom filter only when the probe Parquet scan runs
    /// row-level `RowFilter` (`pushdown_filters = true`); when false, only
    /// min/max bounds and small `InList` are needed for file-open pruning.
    pub(super) build_bloom: bool,
    /// True when the build pipeline applies a selective filter or reads from a
    /// materialized upstream stage.
    pub(super) build_is_filtered: bool,
}

impl DynamicJoinFilterConfig {
    pub(super) fn should_build_bloom(&self, entries: usize) -> bool {
        self.build_bloom
            && (self.build_is_filtered || entries <= UNFILTERED_BLOOM_MAX_ROWS)
    }
}

/// Cache-line-friendly 256-bit (8 x `u32`) split-block Bloom filter.
///
/// When `num_blocks >= 64`, the top 6 bits of the block index use the Level-0
/// radix partition bits (`[51..57)`) so every Level-0 radix partition maps to a
/// disjoint contiguous slice of `blocks` and can be built lock-free without
/// atomics or cross-core cache-line contention.
#[derive(Debug)]
pub(super) struct SplitBlockBloomFilter {
    blocks: Vec<[u32; 8]>,
    block_mask: usize,
    radix_low_bits: Option<u32>,
}

impl SplitBlockBloomFilter {
    /// Returns the power-of-two block count for `entries`, or `None` when
    /// `entries == 0` or exceeds [`MAX_BLOOM_KEYS`].
    pub(super) fn num_blocks_for_entries(entries: usize) -> Option<usize> {
        if entries == 0 || entries > MAX_BLOOM_KEYS {
            return None;
        }
        let needed_bits = entries.saturating_mul(MIN_BLOOM_BITS_PER_KEY);
        let needed_blocks = needed_bits.div_ceil(256).max(1).next_power_of_two();
        let max_blocks = MAX_BLOOM_BYTES / size_of::<[u32; 8]>();
        Some(needed_blocks.min(max_blocks))
    }

    pub(super) fn estimated_bytes(num_blocks: usize) -> usize {
        num_blocks.saturating_mul(size_of::<[u32; 8]>())
    }

    #[inline]
    pub(super) fn radix_low_bits_for_blocks(num_blocks: usize) -> Option<u32> {
        let min_parts = 1usize << DEFAULT_RADIX_BITS;
        if num_blocks >= min_parts {
            Some(num_blocks.trailing_zeros() - DEFAULT_RADIX_BITS)
        } else {
            None
        }
    }

    pub(super) fn with_blocks(num_blocks: usize) -> Self {
        debug_assert!(num_blocks.is_power_of_two());
        Self {
            blocks: vec![[0u32; 8]; num_blocks],
            block_mask: num_blocks - 1,
            radix_low_bits: Self::radix_low_bits_for_blocks(num_blocks),
        }
    }

    pub(super) fn from_radix_partition_blocks(
        num_blocks: usize,
        parts: Vec<Option<Vec<[u32; 8]>>>,
    ) -> Self {
        let low_bits = Self::radix_low_bits_for_blocks(num_blocks)
            .expect("radix partition Bloom requires num_blocks >= 64");
        let stride = 1usize << low_bits;
        debug_assert_eq!(parts.len(), 1usize << DEFAULT_RADIX_BITS);
        let mut blocks = Vec::with_capacity(num_blocks);
        for part in parts {
            match part {
                Some(slice) => {
                    debug_assert_eq!(slice.len(), stride);
                    blocks.extend_from_slice(&slice);
                }
                None => blocks.resize(blocks.len() + stride, [0u32; 8]),
            }
        }
        Self {
            blocks,
            block_mask: num_blocks - 1,
            radix_low_bits: Some(low_bits),
        }
    }

    pub(super) fn memory_bytes(&self) -> usize {
        self.blocks.capacity().saturating_mul(size_of::<[u32; 8]>())
    }

    #[inline]
    fn block_index(&self, h: u64) -> usize {
        if let Some(low_bits) = self.radix_low_bits {
            let part =
                radix_partition(h, level0_shift(DEFAULT_RADIX_BITS), DEFAULT_RADIX_BITS);
            let low_mask = (1usize << low_bits) - 1;
            (part << low_bits) | ((h as usize) & low_mask)
        } else {
            (h as usize) & self.block_mask
        }
    }

    #[inline]
    fn intra_block_mask(h: u64) -> [u32; 8] {
        let x = (h >> 32) as u32;
        [
            1u32 << (x.wrapping_mul(SBBF_SALTS[0]) >> 27),
            1u32 << (x.wrapping_mul(SBBF_SALTS[1]) >> 27),
            1u32 << (x.wrapping_mul(SBBF_SALTS[2]) >> 27),
            1u32 << (x.wrapping_mul(SBBF_SALTS[3]) >> 27),
            1u32 << (x.wrapping_mul(SBBF_SALTS[4]) >> 27),
            1u32 << (x.wrapping_mul(SBBF_SALTS[5]) >> 27),
            1u32 << (x.wrapping_mul(SBBF_SALTS[6]) >> 27),
            1u32 << (x.wrapping_mul(SBBF_SALTS[7]) >> 27),
        ]
    }

    #[inline]
    pub(super) fn insert_into_partition_slice(
        slice: &mut [[u32; 8]],
        low_mask: usize,
        h: u64,
    ) {
        let block = &mut slice[(h as usize) & low_mask];
        let mask = Self::intra_block_mask(h);
        for (word, bit) in block.iter_mut().zip(mask) {
            *word |= bit;
        }
    }

    #[inline]
    pub(super) fn insert(&mut self, h: u64) {
        let idx = self.block_index(h);
        let block = &mut self.blocks[idx];
        let mask = Self::intra_block_mask(h);
        for (word, bit) in block.iter_mut().zip(mask) {
            *word |= bit;
        }
    }

    pub(super) fn merge_from(&mut self, other: &Self) {
        debug_assert_eq!(self.blocks.len(), other.blocks.len());
        for (dst, src) in self.blocks.iter_mut().zip(other.blocks.iter()) {
            for (w_dst, &w_src) in dst.iter_mut().zip(src.iter()) {
                *w_dst |= w_src;
            }
        }
    }

    #[cfg(any(feature = "parquet", test))]
    #[inline]
    pub(super) fn might_contain(&self, h: u64) -> bool {
        let idx = self.block_index(h);
        let block = &self.blocks[idx];
        let mask = Self::intra_block_mask(h);
        let mut all = true;
        for (&word, bit) in block.iter().zip(mask) {
            all &= (word & bit) != 0;
        }
        all
    }
}

#[derive(Debug, Clone, Default)]
enum SmallDistinctValues {
    #[default]
    Disabled,
    Open(Vec<ScalarValue>),
    Overflow,
}

impl SmallDistinctValues {
    fn update_from_array(&mut self, array: &dyn Array) {
        let Self::Open(seen) = self else {
            return;
        };
        for i in 0..array.len() {
            if array.is_null(i) {
                continue;
            }
            let Ok(sv) = ScalarValue::try_from_array(array, i) else {
                *self = Self::Overflow;
                return;
            };
            if !seen.contains(&sv) {
                seen.push(sv);
                if seen.len() > MAX_PER_KEY_INLIST {
                    *self = Self::Overflow;
                    return;
                }
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        match (self, other) {
            (Self::Disabled, _) | (_, Self::Disabled) => {}
            (s @ Self::Open(_), Self::Overflow) => *s = Self::Overflow,
            (Self::Open(mine), Self::Open(theirs)) => {
                for sv in theirs {
                    if !mine.contains(sv) {
                        mine.push(sv.clone());
                        if mine.len() > MAX_PER_KEY_INLIST {
                            *mine = Vec::new();
                            break;
                        }
                    }
                }
            }
            (Self::Overflow, _) => {}
        }
    }

    fn into_inlist_array(self) -> Option<ArrayRef> {
        let Self::Open(seen) = self else {
            return None;
        };
        if seen.len() <= 1 || seen.len() > MAX_PER_KEY_INLIST {
            return None;
        }
        ScalarValue::iter_to_array(seen).ok()
    }
}

/// Per-key-column min/max range and NULL/NaN summary accumulated across build workers.
#[derive(Debug, Clone, Default)]
pub(super) struct ColumnKeyBounds {
    pub(super) min: Option<ScalarValue>,
    pub(super) max: Option<ScalarValue>,
    pub(super) has_null: bool,
    pub(super) has_nan: bool,
    small_distinct: SmallDistinctValues,
    /// Optional small per-key `InList` (<= 32 distinct values) finalized for
    /// multi-key joins so partially-bound keys can push exact `IN (...)`.
    pub(super) per_key_inlist: Option<ArrayRef>,
}

impl ColumnKeyBounds {
    fn new(track_small_distinct: bool) -> Self {
        Self {
            small_distinct: if track_small_distinct {
                SmallDistinctValues::Open(Vec::new())
            } else {
                SmallDistinctValues::Disabled
            },
            ..Self::default()
        }
    }

    fn update_from_array(&mut self, array: &dyn Array) {
        if array.null_count() > 0 {
            self.has_null = true;
        }
        if array.null_count() == array.len() {
            return;
        }
        self.small_distinct.update_from_array(array);
        if self.has_nan {
            return;
        }
        match array_min_max(array) {
            MinMaxResult::HasNaN => {
                self.has_nan = true;
                self.min = None;
                self.max = None;
                self.small_distinct = SmallDistinctValues::Overflow;
            }
            MinMaxResult::Bounds(lo, hi) => {
                match &mut self.min {
                    Some(cur) if lo < *cur => *cur = lo,
                    None => self.min = Some(lo),
                    _ => {}
                }
                match &mut self.max {
                    Some(cur) if hi > *cur => *cur = hi,
                    None => self.max = Some(hi),
                    _ => {}
                }
            }
            MinMaxResult::None => {}
        }
    }

    fn merge(&mut self, other: &Self) {
        self.has_null |= other.has_null;
        self.has_nan |= other.has_nan;
        if self.has_nan {
            self.min = None;
            self.max = None;
            self.small_distinct = SmallDistinctValues::Overflow;
            return;
        }
        let was_open = matches!(self.small_distinct, SmallDistinctValues::Open(_));
        self.small_distinct.merge(&other.small_distinct);
        if was_open
            && let SmallDistinctValues::Open(ref v) = self.small_distinct
            && v.is_empty()
            && !matches!(other.small_distinct, SmallDistinctValues::Open(ref ov) if ov.is_empty())
        {
            self.small_distinct = SmallDistinctValues::Overflow;
        }
        if let Some(other_min) = &other.min {
            match &mut self.min {
                Some(cur) if other_min < cur => *cur = other_min.clone(),
                None => self.min = Some(other_min.clone()),
                _ => {}
            }
        }
        if let Some(other_max) = &other.max {
            match &mut self.max {
                Some(cur) if other_max > cur => *cur = other_max.clone(),
                None => self.max = Some(other_max.clone()),
                _ => {}
            }
        }
    }

    fn finalize_per_key_inlist(&mut self) {
        let distinct = std::mem::take(&mut self.small_distinct);
        self.per_key_inlist = distinct.into_inlist_array();
    }
}

/// Worker-local dynamic join key bounds accumulated during build `sink` and merged in `combine`.
#[derive(Debug, Clone)]
pub(super) struct LocalJoinKeySummary {
    columns: Vec<ColumnKeyBounds>,
}

impl LocalJoinKeySummary {
    pub(super) fn new(num_keys: usize, has_floats: bool) -> Self {
        let track_small_distinct = num_keys > 1 && !has_floats;
        Self {
            columns: (0..num_keys)
                .map(|_| ColumnKeyBounds::new(track_small_distinct))
                .collect(),
        }
    }

    pub(super) fn update(
        &mut self,
        keys: &[ArrayRef],
        valid_mask: Option<&BooleanBuffer>,
        null_equality: NullEquality,
    ) -> Result<()> {
        debug_assert_eq!(self.columns.len(), keys.len());
        if keys.is_empty() || keys[0].is_empty() {
            return Ok(());
        }
        if null_equality == NullEquality::NullEqualsNothing
            && let Some(bits) = valid_mask
        {
            let valid_rows = bits.count_set_bits();
            if valid_rows == 0 {
                return Ok(());
            }
            if valid_rows < keys[0].len() {
                let pred = BooleanArray::new(bits.clone(), None);
                for (bounds, key) in self.columns.iter_mut().zip(keys.iter()) {
                    let filtered = filter(key.as_ref(), &pred)?;
                    bounds.update_from_array(filtered.as_ref());
                }
                return Ok(());
            }
        }
        for (bounds, key) in self.columns.iter_mut().zip(keys.iter()) {
            bounds.update_from_array(key.as_ref());
        }
        Ok(())
    }

    pub(super) fn merge(&mut self, other: &Self) {
        for (a, b) in self.columns.iter_mut().zip(other.columns.iter()) {
            a.merge(b);
        }
    }

    pub(super) fn is_saturated_int_domain(&self, distinct_count: usize) -> bool {
        if self.columns.len() != 1 {
            return false;
        }
        let col = &self.columns[0];
        if col.has_nan {
            return false;
        }
        let (Some(lo), Some(hi)) = (&col.min, &col.max) else {
            return false;
        };
        let span: Option<u128> = match (lo, hi) {
            (ScalarValue::Int32(Some(a)), ScalarValue::Int32(Some(b))) if a <= b => {
                Some((*b as i128 - *a as i128) as u128 + 1)
            }
            (ScalarValue::Int64(Some(a)), ScalarValue::Int64(Some(b))) if a <= b => {
                Some((*b as i128 - *a as i128) as u128 + 1)
            }
            (ScalarValue::UInt32(Some(a)), ScalarValue::UInt32(Some(b))) if a <= b => {
                Some((*b as u128 - *a as u128) + 1)
            }
            (ScalarValue::UInt64(Some(a)), ScalarValue::UInt64(Some(b))) if a <= b => {
                Some((*b as u128 - *a as u128) + 1)
            }
            _ => None,
        };
        span.is_some_and(|s| {
            (distinct_count as u128).saturating_mul(4) >= s.saturating_mul(3)
        })
    }

    pub(super) fn into_bounds(mut self) -> Vec<ColumnKeyBounds> {
        for col in &mut self.columns {
            col.finalize_per_key_inlist();
        }
        self.columns
    }
}

/// Finalized membership filter published on [`FinalizedHashTable`](super::hash_join::FinalizedHashTable).
#[derive(Debug)]
pub(super) enum JoinMembershipFilter {
    /// Zero matchable build keys (empty build, or all-NULL build under `NullEqualsNothing`).
    Empty,
    /// Small single-column build: deduplicated non-null build keys for exact `InListExpr`.
    InList(ArrayRef),
    /// Cache-resident split-block Bloom filter built over the join's 64-bit hashes.
    Bloom(Arc<SplitBlockBloomFilter>),
    /// Membership filter skipped (e.g. single exact bound, tight memory pool, or oversized build).
    None,
}

impl JoinMembershipFilter {
    pub(super) fn memory_bytes(&self) -> usize {
        match self {
            Self::Empty | Self::None => 0,
            Self::InList(arr) => arr.get_array_memory_size(),
            Self::Bloom(filter) => filter.memory_bytes(),
        }
    }
}

/// Immutable dynamic join filter summary produced during build `finalize` and
/// consumed by probe-side `ParquetScan::start` before any probe row group is fetched.
#[derive(Debug)]
#[cfg_attr(not(any(feature = "parquet", test)), expect(dead_code))]
pub(super) struct DynamicJoinFilterData {
    pub(super) key_kind: KeyKind,
    pub(super) null_equality: NullEquality,
    pub(super) has_floats: bool,
    pub(super) bounds: Vec<ColumnKeyBounds>,
    pub(super) membership: JoinMembershipFilter,
}

impl DynamicJoinFilterData {
    pub(super) fn empty(
        key_kind: KeyKind,
        null_equality: NullEquality,
        has_floats: bool,
        num_keys: usize,
    ) -> Self {
        Self {
            key_kind,
            null_equality,
            has_floats,
            bounds: vec![ColumnKeyBounds::default(); num_keys],
            membership: JoinMembershipFilter::Empty,
        }
    }

    pub(super) fn memory_bytes(&self) -> usize {
        let per_key_bytes: usize = self
            .bounds
            .iter()
            .filter_map(|b| b.per_key_inlist.as_ref())
            .map(|a| a.get_array_memory_size())
            .fold(0usize, usize::saturating_add);
        self.membership.memory_bytes().saturating_add(per_key_bytes)
    }

    #[cfg(any(feature = "parquet", test))]
    pub(super) fn is_empty_matches(&self) -> bool {
        matches!(self.membership, JoinMembershipFilter::Empty)
    }
}

/// Plan-time binding from one downstream probe-filtering `HashJoinProbe` to
/// expressions in the probe `ParquetScan`'s projected output schema coordinates.
#[cfg(any(feature = "parquet", test))]
#[derive(Debug, Clone)]
pub(super) struct JoinFilterBinding {
    pub(super) filter_id: usize,
    /// Per-join-key expression in `ParquetScan.schema` coordinates (`None` when
    /// a key cannot be traced back to the target scan).
    pub(super) probe_key_exprs: Vec<Option<Arc<dyn PhysicalExpr>>>,
    pub(super) all_keys_bound: bool,
}

#[cfg(any(feature = "parquet", test))]
pub(super) enum BoundJoinPredicates {
    /// Build side has zero matchable rows -> probe scan can short-circuit to 0 rows.
    ShortCircuitEmpty,
    /// Pushable filter conjuncts in `ParquetScan.schema` coordinates.
    Predicates(Vec<Arc<dyn PhysicalExpr>>),
}

#[cfg(any(feature = "parquet", test))]
impl JoinFilterBinding {
    pub(super) fn build_predicates(
        &self,
        data: &DynamicJoinFilterData,
        include_bloom: bool,
        scan_schema: &Schema,
    ) -> Result<BoundJoinPredicates> {
        if data.is_empty_matches() {
            return Ok(BoundJoinPredicates::ShortCircuitEmpty);
        }

        let mut preds: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
        let mut single_exact_bound = false;

        for (idx, expr_opt) in self.probe_key_exprs.iter().enumerate() {
            let Some(probe_expr) = expr_opt else {
                continue;
            };
            let Some(col_bounds) = data.bounds.get(idx) else {
                continue;
            };
            if col_bounds.has_nan {
                continue;
            }

            let range_pred = match (&col_bounds.min, &col_bounds.max) {
                (Some(lo), Some(hi)) if lo == hi => {
                    if self.probe_key_exprs.len() == 1
                        && !(data.null_equality == NullEquality::NullEqualsNull
                            && col_bounds.has_null)
                    {
                        single_exact_bound = true;
                    }
                    Some(vec![Arc::new(BinaryExpr::new(
                        Arc::clone(probe_expr),
                        ExprOperator::Eq,
                        Arc::new(Literal::new(lo.clone())),
                    )) as Arc<dyn PhysicalExpr>])
                }
                (Some(lo), Some(hi)) => Some(vec![
                    Arc::new(BinaryExpr::new(
                        Arc::clone(probe_expr),
                        ExprOperator::GtEq,
                        Arc::new(Literal::new(lo.clone())),
                    )) as Arc<dyn PhysicalExpr>,
                    Arc::new(BinaryExpr::new(
                        Arc::clone(probe_expr),
                        ExprOperator::LtEq,
                        Arc::new(Literal::new(hi.clone())),
                    )) as Arc<dyn PhysicalExpr>,
                ]),
                _ => None,
            };

            if data.null_equality == NullEquality::NullEqualsNull && col_bounds.has_null {
                let is_null: Arc<dyn PhysicalExpr> =
                    Arc::new(IsNullExpr::new(Arc::clone(probe_expr)));
                match range_pred {
                    Some(range_conjuncts) => {
                        let range =
                            datafusion_physical_expr::conjunction(range_conjuncts);
                        preds.push(Arc::new(BinaryExpr::new(
                            range,
                            ExprOperator::Or,
                            is_null,
                        )));
                    }
                    None => {
                        preds.push(is_null);
                    }
                }
            } else if let Some(range_conjuncts) = range_pred {
                preds.extend(range_conjuncts);
            }

            // For partially-bound multi-key joins (e.g. `c_nationkey` from
            // `supplier(s_suppkey, s_nationkey)` in TPC-H Q5), push any small
            // per-key `InList` directly onto this key column.
            if !self.all_keys_bound
                && let Some(values) = &col_bounds.per_key_inlist
            {
                let in_list: Arc<dyn PhysicalExpr> =
                    Arc::new(InListExpr::try_new_from_array(
                        Arc::clone(probe_expr),
                        Arc::clone(values),
                        false,
                        scan_schema,
                    )?);
                if data.null_equality == NullEquality::NullEqualsNull
                    && col_bounds.has_null
                {
                    let is_null: Arc<dyn PhysicalExpr> =
                        Arc::new(IsNullExpr::new(Arc::clone(probe_expr)));
                    preds.push(Arc::new(BinaryExpr::new(
                        in_list,
                        ExprOperator::Or,
                        is_null,
                    )));
                } else {
                    preds.push(in_list);
                }
            }
        }

        if self.all_keys_bound && !single_exact_bound {
            let bound_keys: Vec<Arc<dyn PhysicalExpr>> = self
                .probe_key_exprs
                .iter()
                .map(|e| Arc::clone(e.as_ref().expect("all_keys_bound")))
                .collect();
            match &data.membership {
                JoinMembershipFilter::Empty | JoinMembershipFilter::None => {}
                JoinMembershipFilter::InList(values) if bound_keys.len() == 1 => {
                    let in_list: Arc<dyn PhysicalExpr> =
                        Arc::new(InListExpr::try_new_from_array(
                            Arc::clone(&bound_keys[0]),
                            Arc::clone(values),
                            false,
                            scan_schema,
                        )?);
                    if data.null_equality == NullEquality::NullEqualsNull
                        && data.bounds.first().is_some_and(|b| b.has_null)
                    {
                        let is_null: Arc<dyn PhysicalExpr> =
                            Arc::new(IsNullExpr::new(Arc::clone(&bound_keys[0])));
                        preds.push(Arc::new(BinaryExpr::new(
                            in_list,
                            ExprOperator::Or,
                            is_null,
                        )));
                    } else {
                        preds.push(in_list);
                    }
                }
                JoinMembershipFilter::InList(_) => {}
                JoinMembershipFilter::Bloom(filter) if include_bloom => {
                    preds.push(Arc::new(JoinBloomFilterExpr {
                        key_exprs: bound_keys,
                        key_kind: data.key_kind,
                        null_equality: data.null_equality,
                        has_floats: data.has_floats,
                        filter: Arc::clone(filter),
                    }));
                }
                JoinMembershipFilter::Bloom(_) => {}
            }
        }

        Ok(BoundJoinPredicates::Predicates(preds))
    }
}

/// Advisory `PhysicalExpr` evaluating a [`SplitBlockBloomFilter`] over join
/// key column(s) inside Parquet `RowFilter`.
#[cfg(any(feature = "parquet", test))]
#[derive(Debug)]
pub(super) struct JoinBloomFilterExpr {
    key_exprs: Vec<Arc<dyn PhysicalExpr>>,
    key_kind: KeyKind,
    null_equality: NullEquality,
    has_floats: bool,
    filter: Arc<SplitBlockBloomFilter>,
}

#[cfg(any(feature = "parquet", test))]
impl JoinBloomFilterExpr {
    pub(super) fn filter_memory_bytes(&self) -> usize {
        self.filter.memory_bytes()
    }
}

#[cfg(any(feature = "parquet", test))]
impl PartialEq for JoinBloomFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.key_kind == other.key_kind
            && self.null_equality == other.null_equality
            && self.has_floats == other.has_floats
            && Arc::ptr_eq(&self.filter, &other.filter)
            && self.key_exprs.len() == other.key_exprs.len()
            && self
                .key_exprs
                .iter()
                .zip(other.key_exprs.iter())
                .all(|(a, b)| a.as_ref() == b.as_ref())
    }
}

#[cfg(any(feature = "parquet", test))]
impl Eq for JoinBloomFilterExpr {}

#[cfg(any(feature = "parquet", test))]
impl Hash for JoinBloomFilterExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.filter).hash(state);
        self.key_kind.hash(state);
        self.null_equality.hash(state);
        self.has_floats.hash(state);
        for expr in &self.key_exprs {
            expr.hash(state);
        }
    }
}

#[cfg(any(feature = "parquet", test))]
impl Display for JoinBloomFilterExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "morsel_bloom_filter(")?;
        for (i, expr) in self.key_exprs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{expr}")?;
        }
        write!(f, ")")
    }
}

#[cfg(any(feature = "parquet", test))]
impl PhysicalExpr for JoinBloomFilterExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(Vec::<
                bool,
            >::new(
            )))));
        }

        macro_rules! eval_flat_int {
            ($Arr:ty, $to_u64:expr) => {{
                let arr_ref = self.key_exprs[0].evaluate(batch)?.into_array(num_rows)?;
                let arr = arr_ref.as_any().downcast_ref::<$Arr>().ok_or_else(|| {
                    internal_datafusion_err!(
                        "Expected {} join key array in JoinBloomFilterExpr, got {}",
                        stringify!($Arr),
                        arr_ref.data_type()
                    )
                })?;
                let values = arr.values();
                let filter = self.filter.as_ref();
                let to_u64: fn(_) -> u64 = $to_u64;
                let buf = if arr.null_count() == 0 {
                    BooleanBuffer::collect_bool(num_rows, |i| {
                        filter.might_contain(mix_u64(to_u64(values[i])))
                    })
                } else {
                    let nulls = arr.nulls().unwrap();
                    BooleanBuffer::collect_bool(num_rows, |i| {
                        nulls.is_valid(i)
                            && filter.might_contain(mix_u64(to_u64(values[i])))
                    })
                };
                return Ok(ColumnarValue::Array(Arc::new(BooleanArray::new(buf, None))));
            }};
        }

        macro_rules! eval_flat_int_pair {
            ($Arr:ty, $to_u64:expr) => {{
                let arr0_ref = self.key_exprs[0].evaluate(batch)?.into_array(num_rows)?;
                let arr1_ref = self.key_exprs[1].evaluate(batch)?.into_array(num_rows)?;
                let arr0 = arr0_ref.as_any().downcast_ref::<$Arr>().ok_or_else(|| {
                    internal_datafusion_err!(
                        "Expected {} join key array in JoinBloomFilterExpr, got {}",
                        stringify!($Arr),
                        arr0_ref.data_type()
                    )
                })?;
                let arr1 = arr1_ref.as_any().downcast_ref::<$Arr>().ok_or_else(|| {
                    internal_datafusion_err!(
                        "Expected {} join key array in JoinBloomFilterExpr, got {}",
                        stringify!($Arr),
                        arr1_ref.data_type()
                    )
                })?;
                let v0 = arr0.values();
                let v1 = arr1.values();
                let filter = self.filter.as_ref();
                let to_u64: fn(_) -> u64 = $to_u64;
                let buf = if arr0.null_count() == 0 && arr1.null_count() == 0 {
                    BooleanBuffer::collect_bool(num_rows, |i| {
                        filter.might_contain(mix_u64_pair(to_u64(v0[i]), to_u64(v1[i])))
                    })
                } else {
                    let n0 = arr0.nulls();
                    let n1 = arr1.nulls();
                    BooleanBuffer::collect_bool(num_rows, |i| {
                        n0.is_none_or(|n| n.is_valid(i))
                            && n1.is_none_or(|n| n.is_valid(i))
                            && filter.might_contain(mix_u64_pair(
                                to_u64(v0[i]),
                                to_u64(v1[i]),
                            ))
                    })
                };
                return Ok(ColumnarValue::Array(Arc::new(BooleanArray::new(buf, None))));
            }};
        }

        match self.key_kind {
            KeyKind::Int32 => eval_flat_int!(Int32Array, |v: i32| v as u32 as u64),
            KeyKind::Int64 => eval_flat_int!(Int64Array, |v: i64| v as u64),
            KeyKind::UInt32 => eval_flat_int!(UInt32Array, |v: u32| v as u64),
            KeyKind::UInt64 => eval_flat_int!(UInt64Array, |v: u64| v),
            KeyKind::Int32Pair => {
                eval_flat_int_pair!(Int32Array, |v: i32| v as u32 as u64)
            }
            KeyKind::Int64Pair => eval_flat_int_pair!(Int64Array, |v: i64| v as u64),
            KeyKind::Cross => {
                return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))));
            }
            KeyKind::Utf8 | KeyKind::LargeUtf8 | KeyKind::Utf8View | KeyKind::Rows => {}
        }

        let mut keys = Vec::with_capacity(self.key_exprs.len());
        for expr in &self.key_exprs {
            keys.push(expr.evaluate(batch)?.into_array(num_rows)?);
        }
        if self.has_floats {
            for key in &mut keys {
                *key = float_semantics::normalize(Arc::clone(key));
            }
        }
        let valid_mask = if self.null_equality == NullEquality::NullEqualsNothing {
            valid_key_mask(&keys)
        } else {
            None
        };
        // Zero-fill so NULL slots under NullEqualsNull hash identically to build.
        let mut hashes = vec![0u64; num_rows];
        create_hashes(&keys, &join_random_state(), &mut hashes)?;
        let filter = self.filter.as_ref();
        let buf = match valid_mask {
            Some(valid) => BooleanBuffer::collect_bool(num_rows, |i| {
                valid.value(i) && filter.might_contain(hashes[i])
            }),
            None => {
                BooleanBuffer::collect_bool(num_rows, |i| filter.might_contain(hashes[i]))
            }
        };
        Ok(ColumnarValue::Array(Arc::new(BooleanArray::new(buf, None))))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.key_exprs.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self {
            key_exprs: children,
            key_kind: self.key_kind,
            null_equality: self.null_equality,
            has_floats: self.has_floats,
            filter: Arc::clone(&self.filter),
        }))
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

enum MinMaxResult {
    Bounds(ScalarValue, ScalarValue),
    HasNaN,
    None,
}

fn array_min_max(array: &dyn Array) -> MinMaxResult {
    macro_rules! prim_min_max {
        ($Arr:ty, $wrap:expr) => {{
            let arr = array.as_any().downcast_ref::<$Arr>().unwrap();
            match (min(arr), max(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds($wrap(lo), $wrap(hi)),
                _ => MinMaxResult::None,
            }
        }};
    }

    match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            match (min_boolean(arr), max_boolean(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::Boolean(Some(lo)),
                    ScalarValue::Boolean(Some(hi)),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::Int8 => prim_min_max!(Int8Array, |v| ScalarValue::Int8(Some(v))),
        DataType::Int16 => prim_min_max!(Int16Array, |v| ScalarValue::Int16(Some(v))),
        DataType::Int32 => prim_min_max!(Int32Array, |v| ScalarValue::Int32(Some(v))),
        DataType::Int64 => prim_min_max!(Int64Array, |v| ScalarValue::Int64(Some(v))),
        DataType::UInt8 => prim_min_max!(UInt8Array, |v| ScalarValue::UInt8(Some(v))),
        DataType::UInt16 => prim_min_max!(UInt16Array, |v| ScalarValue::UInt16(Some(v))),
        DataType::UInt32 => prim_min_max!(UInt32Array, |v| ScalarValue::UInt32(Some(v))),
        DataType::UInt64 => prim_min_max!(UInt64Array, |v| ScalarValue::UInt64(Some(v))),
        DataType::Date32 => prim_min_max!(Date32Array, |v| ScalarValue::Date32(Some(v))),
        DataType::Date64 => prim_min_max!(Date64Array, |v| ScalarValue::Date64(Some(v))),
        DataType::Time32(TimeUnit::Second) => {
            prim_min_max!(Time32SecondArray, |v| ScalarValue::Time32Second(Some(v)))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            prim_min_max!(Time32MillisecondArray, |v| {
                ScalarValue::Time32Millisecond(Some(v))
            })
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            prim_min_max!(Time64MicrosecondArray, |v| {
                ScalarValue::Time64Microsecond(Some(v))
            })
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            prim_min_max!(Time64NanosecondArray, |v| {
                ScalarValue::Time64Nanosecond(Some(v))
            })
        }
        DataType::Timestamp(TimeUnit::Second, tz) => {
            prim_min_max!(TimestampSecondArray, |v| {
                ScalarValue::TimestampSecond(Some(v), tz.clone())
            })
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            prim_min_max!(TimestampMillisecondArray, |v| {
                ScalarValue::TimestampMillisecond(Some(v), tz.clone())
            })
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            prim_min_max!(TimestampMicrosecondArray, |v| {
                ScalarValue::TimestampMicrosecond(Some(v), tz.clone())
            })
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            prim_min_max!(TimestampNanosecondArray, |v| {
                ScalarValue::TimestampNanosecond(Some(v), tz.clone())
            })
        }
        DataType::Duration(TimeUnit::Second) => {
            prim_min_max!(DurationSecondArray, |v| ScalarValue::DurationSecond(Some(
                v
            )))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            prim_min_max!(DurationMillisecondArray, |v| {
                ScalarValue::DurationMillisecond(Some(v))
            })
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            prim_min_max!(DurationMicrosecondArray, |v| {
                ScalarValue::DurationMicrosecond(Some(v))
            })
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            prim_min_max!(DurationNanosecondArray, |v| {
                ScalarValue::DurationNanosecond(Some(v))
            })
        }
        DataType::Decimal128(p, s) => {
            prim_min_max!(Decimal128Array, |v| {
                ScalarValue::Decimal128(Some(v), *p, *s)
            })
        }
        DataType::Decimal256(p, s) => {
            prim_min_max!(Decimal256Array, |v| {
                ScalarValue::Decimal256(Some(v), *p, *s)
            })
        }
        DataType::Float16 => {
            let arr = array.as_any().downcast_ref::<Float16Array>().unwrap();
            let mut lo = None;
            let mut hi = None;
            for v in arr.iter().flatten() {
                if v.is_nan() {
                    return MinMaxResult::HasNaN;
                }
                lo = Some(lo.map_or(v, |cur| if v < cur { v } else { cur }));
                hi = Some(hi.map_or(v, |cur| if v > cur { v } else { cur }));
            }
            match (lo, hi) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::Float16(Some(lo)),
                    ScalarValue::Float16(Some(hi)),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut lo: Option<f32> = None;
            let mut hi: Option<f32> = None;
            for v in arr.iter().flatten() {
                if v.is_nan() {
                    return MinMaxResult::HasNaN;
                }
                lo = Some(lo.map_or(v, |cur| cur.min(v)));
                hi = Some(hi.map_or(v, |cur| cur.max(v)));
            }
            match (lo, hi) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::Float32(Some(lo)),
                    ScalarValue::Float32(Some(hi)),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let mut lo: Option<f64> = None;
            let mut hi: Option<f64> = None;
            for v in arr.iter().flatten() {
                if v.is_nan() {
                    return MinMaxResult::HasNaN;
                }
                lo = Some(lo.map_or(v, |cur| cur.min(v)));
                hi = Some(hi.map_or(v, |cur| cur.max(v)));
            }
            match (lo, hi) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::Float64(Some(lo)),
                    ScalarValue::Float64(Some(hi)),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            match (min_string(arr), max_string(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::Utf8(Some(lo.to_owned())),
                    ScalarValue::Utf8(Some(hi.to_owned())),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            match (min_string(arr), max_string(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::LargeUtf8(Some(lo.to_owned())),
                    ScalarValue::LargeUtf8(Some(hi.to_owned())),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::Utf8View => {
            let arr = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            match (min_string_view(arr), max_string_view(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::Utf8View(Some(lo.to_owned())),
                    ScalarValue::Utf8View(Some(hi.to_owned())),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            match (min_binary(arr), max_binary(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::Binary(Some(lo.to_vec())),
                    ScalarValue::Binary(Some(hi.to_vec())),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::LargeBinary => {
            let arr = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            match (min_binary(arr), max_binary(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::LargeBinary(Some(lo.to_vec())),
                    ScalarValue::LargeBinary(Some(hi.to_vec())),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::BinaryView => {
            let arr = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            match (min_binary_view(arr), max_binary_view(arr)) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::BinaryView(Some(lo.to_vec())),
                    ScalarValue::BinaryView(Some(hi.to_vec())),
                ),
                _ => MinMaxResult::None,
            }
        }
        DataType::FixedSizeBinary(len) => {
            let arr = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            let mut lo: Option<&[u8]> = None;
            let mut hi: Option<&[u8]> = None;
            for v in arr.iter().flatten() {
                lo = Some(lo.map_or(v, |cur| cur.min(v)));
                hi = Some(hi.map_or(v, |cur| cur.max(v)));
            }
            match (lo, hi) {
                (Some(lo), Some(hi)) => MinMaxResult::Bounds(
                    ScalarValue::FixedSizeBinary(*len, Some(lo.to_vec())),
                    ScalarValue::FixedSizeBinary(*len, Some(hi.to_vec())),
                ),
                _ => MinMaxResult::None,
            }
        }
        _ => MinMaxResult::None,
    }
}

/// Deduplicates a small single-column key across build chunks into a non-null
/// `ArrayRef` suitable for `InListExpr`, returning `None` if there is <= 1
/// distinct value (already covered by `col = min`) or it exceeds `max_values` / `max_bytes`.
pub(super) fn try_build_small_inlist_from_chunks(
    key_batches: &[Vec<ArrayRef>],
    valid_masks: &[Option<BooleanBuffer>],
    null_equality: NullEquality,
    max_values: usize,
    max_bytes: usize,
) -> Result<Option<ArrayRef>> {
    if max_values == 0 || max_bytes == 0 || key_batches.is_empty() {
        return Ok(None);
    }
    let mut seen: Vec<ScalarValue> = Vec::new();
    for (chunk_idx, kb) in key_batches.iter().enumerate() {
        let array = &kb[0];
        let valid_opt = if null_equality == NullEquality::NullEqualsNothing {
            valid_masks.get(chunk_idx).and_then(Option::as_ref)
        } else {
            None
        };
        for i in 0..array.len() {
            if valid_opt.is_some_and(|bits| !bits.value(i)) || array.is_null(i) {
                continue;
            }
            let sv = ScalarValue::try_from_array(array.as_ref(), i)?;
            if !seen.contains(&sv) {
                seen.push(sv);
                if seen.len() > max_values {
                    return Ok(None);
                }
            }
        }
    }
    if seen.len() <= 1 {
        return Ok(None);
    }
    let out = ScalarValue::iter_to_array(seen)?;
    if out.get_array_memory_size() > max_bytes {
        return Ok(None);
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use datafusion_physical_expr::expressions::Column;

    #[test]
    fn split_block_bloom_zero_false_negatives_and_low_fpp() {
        let n = 10_000usize;
        let blocks = SplitBlockBloomFilter::num_blocks_for_entries(n).unwrap();
        let mut bloom = SplitBlockBloomFilter::with_blocks(blocks);
        for k in 0..n as i64 {
            bloom.insert(mix_u64(k as u64));
        }
        for k in 0..n as i64 {
            assert!(
                bloom.might_contain(mix_u64(k as u64)),
                "false negative for key {k}"
            );
        }
        let probes = 100_000usize;
        let fps = (0..probes as i64)
            .filter(|&k| bloom.might_contain(mix_u64((k + 1_000_000) as u64)))
            .count();
        let fpp = fps as f64 / probes as f64;
        assert!(fpp < 0.02, "expected FPP < 2%, got {:.4}%", fpp * 100.0);
    }

    #[test]
    fn bloom_expr_handles_int64_nulls_and_nan_floats() -> Result<()> {
        let n = 500i64;
        let blocks = SplitBlockBloomFilter::num_blocks_for_entries(n as usize).unwrap();
        let mut bloom = SplitBlockBloomFilter::with_blocks(blocks);
        for k in 0..n {
            bloom.insert(mix_u64(k as u64));
        }
        let expr = JoinBloomFilterExpr {
            key_exprs: vec![Arc::new(Column::new("k", 0))],
            key_kind: KeyKind::Int64,
            null_equality: NullEquality::NullEqualsNothing,
            has_floats: false,
            filter: Arc::new(bloom),
        };
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![
                Some(0),
                Some(42),
                None,
                Some(999_999),
                Some(499),
            ]))],
        )?;
        let out = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
        let mask = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(mask.value(0));
        assert!(mask.value(1));
        assert!(
            !mask.value(2),
            "NULL must evaluate to false under NullEqualsNothing"
        );
        assert!(mask.value(4));
        Ok(())
    }

    #[test]
    fn bounds_drop_on_nan_and_merge_nulls_safely() -> Result<()> {
        let mut s1 = LocalJoinKeySummary::new(1, true);
        let mut s2 = LocalJoinKeySummary::new(1, true);
        let a1: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.5), Some(3.5)]));
        let a2: ArrayRef = Arc::new(Float64Array::from(vec![Some(f64::NAN), Some(2.0)]));
        s1.update(&[a1], None, NullEquality::NullEqualsNothing)?;
        s2.update(&[a2], None, NullEquality::NullEqualsNothing)?;
        assert_eq!(s1.columns[0].min, Some(ScalarValue::Float64(Some(1.5))));
        assert_eq!(s1.columns[0].max, Some(ScalarValue::Float64(Some(3.5))));
        assert!(!s1.columns[0].has_nan);
        assert!(s2.columns[0].has_nan);
        s1.merge(&s2);
        assert!(s1.columns[0].has_nan);
        assert!(s1.columns[0].min.is_none());
        assert!(s1.columns[0].max.is_none());
        Ok(())
    }

    #[test]
    fn radix_partitioned_bloom_matches_sequential_with_zero_false_negatives() {
        let n = 20_000usize;
        let num_blocks = SplitBlockBloomFilter::num_blocks_for_entries(n).unwrap();
        let low_bits = SplitBlockBloomFilter::radix_low_bits_for_blocks(num_blocks)
            .expect("20K keys must use radix-partitioned blocks");
        let stride = 1usize << low_bits;
        let low_mask = stride - 1;
        let num_parts = 1usize << DEFAULT_RADIX_BITS;
        let shift = level0_shift(DEFAULT_RADIX_BITS);

        let mut seq = SplitBlockBloomFilter::with_blocks(num_blocks);
        let mut parts: Vec<Option<Vec<[u32; 8]>>> =
            (0..num_parts).map(|_| Some(vec![[0u32; 8]; stride])).collect();

        for k in 0..n as u64 {
            let h = mix_u64(k);
            seq.insert(h);
            let p = radix_partition(h, shift, DEFAULT_RADIX_BITS);
            SplitBlockBloomFilter::insert_into_partition_slice(
                parts[p].as_mut().unwrap(),
                low_mask,
                h,
            );
        }

        let par = SplitBlockBloomFilter::from_radix_partition_blocks(num_blocks, parts);
        assert_eq!(seq.blocks, par.blocks, "partitioned Bloom must match bit-for-bit");
        for k in 0..n as u64 {
            assert!(par.might_contain(mix_u64(k)));
        }
        let fps = (0..50_000u64)
            .filter(|&k| par.might_contain(mix_u64(k + 10_000_000)))
            .count();
        assert!(fps < 1_000, "unexpected FPP: {fps}/50000");
    }

    #[test]
    fn saturated_int_domain_and_unfiltered_bloom_gating() -> Result<()> {
        let cfg_unfiltered = DynamicJoinFilterConfig {
            max_inlist_values: 150,
            max_inlist_bytes: 8192,
            build_bloom: true,
            build_is_filtered: false,
        };
        assert!(cfg_unfiltered.should_build_bloom(500));
        assert!(cfg_unfiltered.should_build_bloom(UNFILTERED_BLOOM_MAX_ROWS));
        assert!(!cfg_unfiltered.should_build_bloom(UNFILTERED_BLOOM_MAX_ROWS + 1));

        let cfg_filtered = DynamicJoinFilterConfig {
            build_is_filtered: true,
            ..cfg_unfiltered
        };
        assert!(cfg_filtered.should_build_bloom(500_000));

        // 100% dense [1..=1000] -> saturated (skip Bloom).
        let mut dense = LocalJoinKeySummary::new(1, false);
        let dense_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(1..=1000));
        dense.update(&[dense_arr], None, NullEquality::NullEqualsNothing)?;
        assert!(dense.is_saturated_int_domain(1000));
        assert!(!dense.is_saturated_int_domain(100));

        // Extreme i64::MIN..i64::MAX must never overflow span arithmetic.
        let mut extreme = LocalJoinKeySummary::new(1, false);
        let ext_arr: ArrayRef =
            Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX]));
        extreme.update(&[ext_arr], None, NullEquality::NullEqualsNothing)?;
        assert!(!extreme.is_saturated_int_domain(3));
        Ok(())
    }
}
