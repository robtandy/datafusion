// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! In-pipeline deferred materialization via selection vectors (`BatchView`).

use std::sync::Arc;

#[cfg(test)]
use arrow::array::new_empty_array;
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBufferBuilder, RecordBatch, RecordBatchOptions,
    UInt32Array,
};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
#[cfg(test)]
use arrow::compute::filter;
use arrow::compute::{FilterBuilder, filter_record_batch, take};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::projection::Projector;

/// Default minimum wasted backing bytes before compacting a `BatchView` for retention.
pub(super) const DEFAULT_COMPACT_MIN_WASTED_BYTES: usize = 512 * 1024;
/// Default minimum `retained / estimated_compacted` ratio before compacting for retention.
pub(super) const DEFAULT_COMPACT_MAX_RETENTION_RATIO: usize = 16;
/// Convert sparse `Mask` selections (`selected * 8 <= physical_rows`) to `Indices`
/// when gathering multiple columns or composing stacked filters.
pub(super) const SPARSE_INDICES_DENOMINATOR: usize = 8;

/// Row selection over a backing Arrow [`RecordBatch`].
///
/// Selection coordinates are always relative to the backing batch (including when
/// the backing batch is an Arrow slice) and represent SQL `TRUE` rows only (never
/// nulls).
#[derive(Debug, Clone)]
pub(super) enum Selection {
    All,
    Mask {
        bits: BooleanBuffer,
        selected: usize,
    },
    Indices(UInt32Array),
}

impl Selection {
    pub(super) fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    pub(super) fn selected_rows(&self, physical_rows: usize) -> usize {
        match self {
            Self::All => physical_rows,
            Self::Mask { selected, .. } => *selected,
            Self::Indices(indices) => indices.len(),
        }
    }

    pub(super) fn is_empty(&self, physical_rows: usize) -> bool {
        self.selected_rows(physical_rows) == 0
    }

    pub(super) fn memory_size(&self) -> usize {
        match self {
            Self::All => 0,
            Self::Mask { bits, .. } => {
                if bits.is_empty() {
                    0
                } else {
                    bits.inner().capacity()
                }
            }
            Self::Indices(indices) => indices.get_array_memory_size(),
        }
    }

    pub(super) fn validate(&self, physical_rows: usize) -> Result<()> {
        match self {
            Self::All => Ok(()),
            Self::Mask { bits, selected } => {
                if bits.len() != physical_rows {
                    return internal_err!(
                        "BatchView Mask length {} does not match backing batch row count {}",
                        bits.len(),
                        physical_rows
                    );
                }
                let actual = bits.count_set_bits();
                if *selected != actual {
                    return internal_err!(
                        "BatchView Mask selected count {} does not match set bit count {}",
                        selected,
                        actual
                    );
                }
                Ok(())
            }
            Self::Indices(indices) => {
                if indices.null_count() != 0 {
                    return internal_err!(
                        "BatchView Indices selection must not contain nulls (found {})",
                        indices.null_count()
                    );
                }
                let mut prev: Option<u32> = None;
                for &idx in indices.values() {
                    if (idx as usize) >= physical_rows {
                        return internal_err!(
                            "BatchView selection index {} out of bounds for batch with {} rows",
                            idx,
                            physical_rows
                        );
                    }
                    if let Some(p) = prev
                        && idx <= p
                    {
                        return internal_err!(
                            "BatchView selection indices must be strictly increasing (got {} after {})",
                            idx,
                            p
                        );
                    }
                    prev = Some(idx);
                }
                Ok(())
            }
        }
    }

    /// Convert this selection to strictly increasing `u32` backing-row indices.
    /// Returns `None` if `physical_rows > u32::MAX as usize` for `All` or `Mask`.
    pub(super) fn to_indices(&self, physical_rows: usize) -> Option<UInt32Array> {
        match self {
            Self::Indices(indices) => Some(indices.clone()),
            Self::All => {
                let len = u32::try_from(physical_rows).ok()?;
                let values: Vec<u32> = (0..len).collect();
                Some(UInt32Array::new(ScalarBuffer::from(values), None))
            }
            Self::Mask { bits, selected } => {
                if physical_rows > u32::MAX as usize {
                    return None;
                }
                let mut values = Vec::with_capacity(*selected);
                values.extend(bits.set_indices().map(|i| i as u32));
                Some(UInt32Array::new(ScalarBuffer::from(values), None))
            }
        }
    }

    /// Convert this selection to a backing-coordinate `BooleanBuffer`.
    #[cfg(test)]
    pub(super) fn to_mask(&self, physical_rows: usize) -> BooleanBuffer {
        match self {
            Self::All => BooleanBuffer::new_set(physical_rows),
            Self::Mask { bits, .. } => bits.clone(),
            Self::Indices(indices) => {
                let mut builder = BooleanBufferBuilder::new(physical_rows);
                builder.append_n(physical_rows, false);
                for &idx in indices.values() {
                    builder.set_bit(idx as usize, true);
                }
                builder.finish()
            }
        }
    }

    /// Intersect this selection with a `BooleanArray` evaluated in backing coordinates.
    #[cfg(test)]
    pub(super) fn intersect_backing_mask(
        &self,
        physical_rows: usize,
        mask: &BooleanArray,
    ) -> Result<Self> {
        if mask.len() != physical_rows {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Filter predicate of length {} does not match backing batch length {}",
                mask.len(),
                physical_rows
            ))
            .into());
        }
        let mask_bits = effective_boolean_buffer(mask);
        Ok(match self {
            Self::All => {
                let selected = mask_bits.count_set_bits();
                if selected == physical_rows {
                    Self::All
                } else {
                    Self::Mask {
                        bits: mask_bits,
                        selected,
                    }
                }
            }
            Self::Mask { bits, .. } => {
                let combined = bits & &mask_bits;
                let selected = combined.count_set_bits();
                if selected == physical_rows {
                    Self::All
                } else {
                    Self::Mask {
                        bits: combined,
                        selected,
                    }
                }
            }
            Self::Indices(indices) => {
                let mut kept = Vec::with_capacity(indices.len());
                for &idx in indices.values() {
                    if mask_bits.value(idx as usize) {
                        kept.push(idx);
                    }
                }
                Self::Indices(UInt32Array::new(ScalarBuffer::from(kept), None))
            }
        })
    }

    /// Compose a `BooleanArray` evaluated on compacted rows (`0..prev_selected`)
    /// back through this selection into backing coordinates.
    pub(super) fn compose_compact_mask(
        &self,
        physical_rows: usize,
        compact_mask: &BooleanArray,
    ) -> Result<Self> {
        let prev_selected = self.selected_rows(physical_rows);
        if compact_mask.len() > prev_selected {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Filter predicate of length {} is larger than target array of length {}",
                compact_mask.len(),
                prev_selected
            ))
            .into());
        }
        let compact_bits = effective_boolean_buffer(compact_mask);
        let new_selected = compact_bits.count_set_bits();
        if new_selected == 0 {
            return Ok(Self::Indices(UInt32Array::new(
                ScalarBuffer::from(Vec::<u32>::new()),
                None,
            )));
        }
        if new_selected == prev_selected && compact_mask.len() == prev_selected {
            return Ok(self.clone());
        }
        Ok(match self {
            Self::All => {
                if compact_mask.len() == physical_rows {
                    Self::Mask {
                        bits: compact_bits,
                        selected: new_selected,
                    }
                } else {
                    let mut builder = BooleanBufferBuilder::new(physical_rows);
                    builder.append_buffer(&compact_bits);
                    builder.append_n(physical_rows - compact_bits.len(), false);
                    Self::Mask {
                        bits: builder.finish(),
                        selected: new_selected,
                    }
                }
            }
            Self::Mask { bits, .. } => {
                let limit = compact_bits.len();
                let use_indices = u32::try_from(physical_rows).is_ok()
                    && new_selected.saturating_mul(SPARSE_INDICES_DENOMINATOR)
                        <= physical_rows;
                if use_indices {
                    let mut out = Vec::with_capacity(new_selected);
                    for (compact_idx, backing_idx) in bits.set_indices().enumerate() {
                        if compact_idx >= limit {
                            break;
                        }
                        if compact_bits.value(compact_idx) {
                            out.push(backing_idx as u32);
                        }
                    }
                    Self::Indices(UInt32Array::new(ScalarBuffer::from(out), None))
                } else {
                    let mut builder = BooleanBufferBuilder::new(physical_rows);
                    builder.append_n(physical_rows, false);
                    for (compact_idx, backing_idx) in bits.set_indices().enumerate() {
                        if compact_idx >= limit {
                            break;
                        }
                        if compact_bits.value(compact_idx) {
                            builder.set_bit(backing_idx, true);
                        }
                    }
                    Self::Mask {
                        bits: builder.finish(),
                        selected: new_selected,
                    }
                }
            }
            Self::Indices(old_indices) => {
                let mut out = Vec::with_capacity(new_selected);
                for compact_idx in compact_bits.set_indices() {
                    out.push(old_indices.value(compact_idx));
                }
                Self::Indices(UInt32Array::new(ScalarBuffer::from(out), None))
            }
        })
    }
}

/// Extract a null-free `BooleanBuffer` whose set bits correspond strictly to
/// valid SQL `TRUE` entries (clearing any undefined value bits at null positions).
pub(super) fn effective_boolean_buffer(mask: &BooleanArray) -> BooleanBuffer {
    match mask.nulls() {
        Some(nulls) => mask.values() & nulls.inner(),
        None => mask.values().clone(),
    }
}

/// A backing Arrow [`RecordBatch`] paired with an in-pipeline row [`Selection`].
#[derive(Debug, Clone)]
pub(super) struct BatchView {
    batch: RecordBatch,
    selection: Selection,
}

impl BatchView {
    pub(super) fn all(batch: RecordBatch) -> Self {
        Self {
            batch,
            selection: Selection::All,
        }
    }

    pub(super) fn empty(schema: SchemaRef) -> Self {
        Self {
            batch: RecordBatch::new_empty(schema),
            selection: Selection::All,
        }
    }

    pub(super) fn try_new(batch: RecordBatch, selection: Selection) -> Result<Self> {
        selection.validate(batch.num_rows())?;
        Ok(Self { batch, selection })
    }

    /// Construct a `BatchView` from a backing `RecordBatch` and a predicate `BooleanArray`.
    ///
    /// Matches Arrow's `filter_record_batch` length validation: rejects masks longer
    /// than `batch.num_rows()` and slices `batch` to `mask.len()` for undersized masks.
    pub(super) fn from_boolean_array(
        mut batch: RecordBatch,
        mask: &BooleanArray,
    ) -> Result<Self> {
        let physical_rows = batch.num_rows();
        if mask.len() > physical_rows {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Filter predicate of length {} is larger than target array of length {}",
                mask.len(),
                physical_rows
            ))
            .into());
        }
        if mask.len() < physical_rows {
            batch = batch.slice(0, mask.len());
        }
        let bits = effective_boolean_buffer(mask);
        let selected = bits.count_set_bits();
        if selected == 0 {
            return Ok(Self::empty(batch.schema()));
        }
        if selected == batch.num_rows() {
            return Ok(Self::all(batch));
        }
        Ok(Self {
            batch,
            selection: Selection::Mask { bits, selected },
        })
    }

    pub(super) fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub(super) fn selection(&self) -> &Selection {
        &self.selection
    }

    pub(super) fn into_parts(self) -> (RecordBatch, Selection) {
        (self.batch, self.selection)
    }

    pub(super) fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }

    /// Number of logical (selected) rows in this view.
    pub(super) fn num_rows(&self) -> usize {
        self.selection.selected_rows(self.batch.num_rows())
    }

    /// Number of physical rows in the backing `RecordBatch`.
    #[cfg(test)]
    pub(super) fn physical_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }

    pub(super) fn is_all(&self) -> bool {
        self.selection.is_all()
    }

    /// Convert a `Mask` selection into `Indices` when row indices fit in `u32`.
    #[cfg(test)]
    pub(super) fn with_indices_selection(self) -> Self {
        match &self.selection {
            Selection::All | Selection::Indices(_) => self,
            Selection::Mask { .. } => {
                match self.selection.to_indices(self.batch.num_rows()) {
                    Some(indices) => Self {
                        batch: self.batch,
                        selection: Selection::Indices(indices),
                    },
                    None => self,
                }
            }
        }
    }

    /// Zero-copy column narrowing preserving the active selection using a precomputed schema.
    pub(super) fn project_columns_with_schema(
        self,
        columns: &[usize],
        schema: SchemaRef,
    ) -> Result<Self> {
        if columns.is_empty() {
            let batch = RecordBatch::try_new_with_options(
                schema,
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(self.num_rows())),
            )?;
            return Ok(Self::all(batch));
        }
        let mut cols = Vec::with_capacity(columns.len());
        for &idx in columns {
            cols.push(Arc::clone(self.batch.column(idx)));
        }
        let batch = RecordBatch::try_new_with_options(
            schema,
            cols,
            &RecordBatchOptions::new().with_row_count(Some(self.batch.num_rows())),
        )?;
        Ok(Self {
            batch,
            selection: self.selection,
        })
    }

    /// Zero-copy column narrowing preserving the active selection.
    pub(super) fn project_columns(self, columns: &[usize]) -> Result<Self> {
        if columns.is_empty() {
            let empty_schema = Arc::new(self.batch.schema().project(columns)?);
            let batch = RecordBatch::try_new_with_options(
                empty_schema,
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(self.num_rows())),
            )?;
            return Ok(Self::all(batch));
        }
        let projected = self.batch.project(columns)?;
        Ok(Self {
            batch: projected,
            selection: self.selection,
        })
    }

    /// Evaluate a filter predicate over this view without copying unused columns
    /// and without evaluating stacked predicates on already-rejected rows.
    pub(super) fn filter(self, predicate: &Arc<dyn PhysicalExpr>) -> Result<Self> {
        if self.is_empty() {
            return Ok(Self::empty(self.schema()));
        }
        if self.is_all() {
            return match predicate.evaluate(&self.batch)? {
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => Ok(self),
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(false) | None)) => {
                    Ok(Self::empty(self.schema()))
                }
                ColumnarValue::Scalar(_) => {
                    internal_err!(
                        "Cannot create filter_array from non-boolean predicates"
                    )
                }
                ColumnarValue::Array(array) => {
                    let Ok(mask) = as_boolean_array(&array) else {
                        return internal_err!(
                            "Cannot create filter_array from non-boolean predicates"
                        );
                    };
                    Self::from_boolean_array(self.batch, mask)
                }
            };
        }

        // Stacked filter: compact only the columns referenced by `predicate`,
        // evaluate on selected rows only, and compose positions back into
        // backing coordinates.
        let needed_cols = collect_expr_column_indices(std::slice::from_ref(predicate));
        let compact = self.materialize_columns(&needed_cols)?;
        let rebound = remap_expr_columns(Arc::clone(predicate), &needed_cols)?;
        match rebound.evaluate(&compact)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => Ok(self),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false) | None)) => {
                Ok(Self::empty(self.schema()))
            }
            ColumnarValue::Scalar(_) => {
                internal_err!("Cannot create filter_array from non-boolean predicates")
            }
            ColumnarValue::Array(array) => {
                let Ok(mask) = as_boolean_array(&array) else {
                    return internal_err!(
                        "Cannot create filter_array from non-boolean predicates"
                    );
                };
                let new_selection = self
                    .selection
                    .compose_compact_mask(self.batch.num_rows(), mask)?;
                if new_selection.is_empty(self.batch.num_rows()) {
                    Ok(Self::empty(self.schema()))
                } else {
                    Ok(Self {
                        batch: self.batch,
                        selection: new_selection,
                    })
                }
            }
        }
    }

    /// Selection-aware projection:
    /// - Pure column projections share/reorder columns zero-copy while keeping `Selection`.
    /// - General expression projections compact only required inputs once and never
    ///   evaluate expressions on rejected rows.
    pub(super) fn apply_projection(self, projector: &Projector) -> Result<Self> {
        let out_schema = Arc::clone(projector.output_schema());
        if self.is_empty() {
            return Ok(Self::empty(out_schema));
        }
        if self.is_all() {
            return Ok(Self::all(projector.project_batch(&self.batch)?));
        }

        let exprs = projector.projection().as_ref();
        if exprs.is_empty() {
            let batch = RecordBatch::try_new_with_options(
                out_schema,
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(self.num_rows())),
            )?;
            return Ok(Self::all(batch));
        }

        if is_pure_column_projection(projector) {
            let mut cols = Vec::with_capacity(exprs.len());
            for proj in exprs {
                let idx = proj.expr.downcast_ref::<Column>().unwrap().index();
                cols.push(Arc::clone(self.batch.column(idx)));
            }
            let batch = RecordBatch::try_new_with_options(
                out_schema,
                cols,
                &RecordBatchOptions::new().with_row_count(Some(self.batch.num_rows())),
            )?;
            return Ok(Self {
                batch,
                selection: self.selection,
            });
        }

        // Mixed or general expressions: compact only referenced columns once,
        // share compacted pass-through columns, and evaluate expressions on
        // compacted rows only.
        let needed_cols = projector.projection().column_indices();
        let is_identity = needed_cols.len() == self.batch.num_columns()
            && needed_cols.iter().enumerate().all(|(i, &c)| i == c);
        if is_identity {
            let compact = self.materialize()?;
            return Ok(Self::all(projector.project_batch(&compact)?));
        }
        let compact = self.materialize_columns(&needed_cols)?;
        let selected_rows = compact.num_rows();
        let mut out_cols = Vec::with_capacity(exprs.len());
        for proj in exprs {
            if let Some(col) = proj.expr.downcast_ref::<Column>() {
                let pos = needed_cols.binary_search(&col.index()).map_err(|_| {
                    datafusion_common::internal_datafusion_err!(
                        "Projection column index {} missing from collected indices",
                        col.index()
                    )
                })?;
                out_cols.push(Arc::clone(compact.column(pos)));
            } else {
                let rebound = remap_expr_columns(Arc::clone(&proj.expr), &needed_cols)?;
                out_cols.push(rebound.evaluate(&compact)?.into_array(selected_rows)?);
            }
        }
        let batch = RecordBatch::try_new_with_options(
            out_schema,
            out_cols,
            &RecordBatchOptions::new().with_row_count(Some(selected_rows)),
        )?;
        Ok(Self::all(batch))
    }

    /// Materialize this view into a standard dense Arrow [`RecordBatch`].
    pub(super) fn materialize(self) -> Result<RecordBatch> {
        let physical_rows = self.batch.num_rows();
        let selected = self.num_rows();
        if self.is_all() {
            return Ok(self.batch);
        }
        if selected == 0 {
            return Ok(RecordBatch::new_empty(self.batch.schema()));
        }
        if self.batch.num_columns() == 0 {
            return Ok(RecordBatch::try_new_with_options(
                self.batch.schema(),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(selected)),
            )?);
        }
        if selected == physical_rows {
            return Ok(self.batch);
        }
        materialize_batch_with_selection(&self.batch, &self.selection)
    }

    /// Materialize this view into a standard dense Arrow [`RecordBatch`] without consuming it.
    #[cfg(test)]
    pub(super) fn materialize_ref(&self) -> Result<RecordBatch> {
        self.clone().materialize()
    }

    /// Materialize only the specified backing column indices for the selected rows.
    pub(super) fn materialize_columns(&self, columns: &[usize]) -> Result<RecordBatch> {
        let selected = self.num_rows();
        if columns.is_empty() {
            let empty_schema = Arc::new(self.batch.schema().project(columns)?);
            return Ok(RecordBatch::try_new_with_options(
                empty_schema,
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(selected)),
            )?);
        }
        let is_identity = columns.len() == self.batch.num_columns()
            && columns.iter().enumerate().all(|(i, &c)| i == c);
        if self.is_all() {
            return if is_identity {
                Ok(self.batch.clone())
            } else {
                Ok(self.batch.project(columns)?)
            };
        }
        let narrowed = if is_identity {
            self.batch.clone()
        } else {
            self.batch.project(columns)?
        };
        if selected == 0 {
            return Ok(RecordBatch::new_empty(narrowed.schema()));
        }
        if selected == narrowed.num_rows() {
            return Ok(narrowed);
        }
        materialize_batch_with_selection(&narrowed, &self.selection)
    }

    /// Materialize a single backing column for the selected rows.
    #[cfg(test)]
    pub(super) fn materialize_column(&self, col_idx: usize) -> Result<ArrayRef> {
        let col = self.batch.column(col_idx);
        let physical_rows = self.batch.num_rows();
        match &self.selection {
            Selection::All => Ok(Arc::clone(col)),
            Selection::Mask { bits, selected } => {
                if *selected == 0 {
                    Ok(new_empty_array(col.data_type()))
                } else if *selected == physical_rows {
                    Ok(Arc::clone(col))
                } else {
                    let predicate = BooleanArray::new(bits.clone(), None);
                    Ok(filter(col.as_ref(), &predicate)?)
                }
            }
            Selection::Indices(indices) => {
                if indices.is_empty() {
                    Ok(new_empty_array(col.data_type()))
                } else {
                    Ok(take(col.as_ref(), indices, None)?)
                }
            }
        }
    }

    /// Unique memory bytes pinned by the backing `RecordBatch` buffers.
    pub(super) fn backing_memory_size(&self) -> usize {
        get_record_batch_memory_size(&self.batch)
    }

    /// Memory bytes held by the `Selection` representation itself.
    pub(super) fn selection_memory_size(&self) -> usize {
        self.selection.memory_size()
    }

    /// Total retained memory bytes (`backing + selection`).
    pub(super) fn retained_memory_size(&self) -> usize {
        self.backing_memory_size()
            .saturating_add(self.selection_memory_size())
    }

    /// Estimated memory bytes if this view were compacted to its selected rows.
    pub(super) fn estimated_compacted_bytes(&self) -> usize {
        let physical = self.batch.num_rows();
        let selected = self.num_rows();
        if selected == 0 || physical == 0 || self.batch.num_columns() == 0 {
            return 0;
        }
        let backing = self.backing_memory_size();
        if selected >= physical {
            return backing;
        }
        ((backing as u128 * selected as u128) / physical as u128) as usize
    }

    /// Decide whether keeping this view across a queue or blocking sink would pin
    /// disproportionately large dead backing buffers.
    ///
    /// Selectivity alone is insufficient: tiny batches are not worth copying, while
    /// a small selection pinning a large buffer should be compacted.
    pub(super) fn should_compact_for_retention(
        &self,
        min_wasted_bytes: usize,
        max_retention_ratio: usize,
    ) -> bool {
        if self.is_empty() {
            return self.backing_memory_size() > 0 || self.selection_memory_size() > 0;
        }
        if self.is_all() {
            return false;
        }
        // Fast O(1) pre-check: avoid walking Arrow buffers when selectivity cannot
        // meet `max_retention_ratio` (with 2x slack for selection buffer overhead).
        let physical = self.batch.num_rows();
        let selected = self.num_rows();
        if selected.saturating_mul(max_retention_ratio) > physical.saturating_mul(2) {
            return false;
        }
        let retained = self.retained_memory_size();
        let estimated = self.estimated_compacted_bytes();
        let wasted = retained.saturating_sub(estimated);
        wasted >= min_wasted_bytes
            && retained >= estimated.saturating_mul(max_retention_ratio)
    }

    /// Compact to `Selection::All` when retention policy says keeping the backing
    /// buffers would waste too much memory.
    pub(super) fn into_compacted_for_retention(
        self,
        min_wasted_bytes: usize,
        max_retention_ratio: usize,
    ) -> Result<Self> {
        if self.should_compact_for_retention(min_wasted_bytes, max_retention_ratio) {
            Ok(Self::all(self.materialize()?))
        } else {
            Ok(self)
        }
    }

    /// Reserve memory for retaining this view, proactively compacting oversized
    /// backing buffers and falling back to compaction if the reservation limit
    /// cannot accommodate the uncompacted backing buffers.
    pub(super) fn into_reserved_for_retention(
        self,
        reservation: &mut MemoryReservation,
    ) -> Result<Self> {
        let view = self.into_compacted_for_retention(
            DEFAULT_COMPACT_MIN_WASTED_BYTES,
            DEFAULT_COMPACT_MAX_RETENTION_RATIO,
        )?;
        let retained = view.retained_memory_size();
        if reservation.try_grow(retained).is_ok() {
            return Ok(view);
        }
        if !view.is_all() && view.estimated_compacted_bytes() < retained {
            let compacted = Self::all(view.materialize()?);
            let compacted_bytes = compacted.retained_memory_size();
            reservation.try_grow(compacted_bytes)?;
            return Ok(compacted);
        }
        reservation.try_grow(retained)?;
        Ok(view)
    }
}

fn materialize_batch_with_selection(
    batch: &RecordBatch,
    selection: &Selection,
) -> Result<RecordBatch> {
    let physical_rows = batch.num_rows();
    match selection {
        Selection::All => Ok(batch.clone()),
        Selection::Mask { bits, selected } => {
            if *selected == 0 {
                return Ok(RecordBatch::new_empty(batch.schema()));
            }
            if *selected == physical_rows {
                return Ok(batch.clone());
            }
            if batch.num_columns() >= 2
                && u32::try_from(physical_rows).is_ok()
                && selected.saturating_mul(SPARSE_INDICES_DENOMINATOR) <= physical_rows
                && let Some(indices) = selection.to_indices(physical_rows)
            {
                let cols = batch
                    .columns()
                    .iter()
                    .map(|col| Ok(take(col.as_ref(), &indices, None)?))
                    .collect::<Result<Vec<ArrayRef>>>()?;
                return Ok(RecordBatch::try_new(batch.schema(), cols)?);
            }
            let predicate = BooleanArray::new(bits.clone(), None);
            if batch.num_columns() >= 2 {
                let filter_pred = FilterBuilder::new(&predicate).optimize().build();
                let cols = batch
                    .columns()
                    .iter()
                    .map(|col| Ok(filter_pred.filter(col.as_ref())?))
                    .collect::<Result<Vec<ArrayRef>>>()?;
                Ok(RecordBatch::try_new(batch.schema(), cols)?)
            } else {
                Ok(filter_record_batch(batch, &predicate)?)
            }
        }
        Selection::Indices(indices) => {
            if indices.is_empty() {
                return Ok(RecordBatch::new_empty(batch.schema()));
            }
            let cols = batch
                .columns()
                .iter()
                .map(|col| Ok(take(col.as_ref(), indices, None)?))
                .collect::<Result<Vec<ArrayRef>>>()?;
            Ok(RecordBatch::try_new(batch.schema(), cols)?)
        }
    }
}

pub(super) fn is_pure_column_projection(projector: &Projector) -> bool {
    projector
        .projection()
        .as_ref()
        .iter()
        .all(|p| p.expr.is::<Column>())
}

/// Collect sorted, deduplicated column indices referenced by `exprs`.
pub(super) fn collect_expr_column_indices(exprs: &[Arc<dyn PhysicalExpr>]) -> Vec<usize> {
    let mut indices = Vec::new();
    for expr in exprs {
        let _ = expr.apply(|node| {
            if let Some(col) = node.downcast_ref::<Column>() {
                indices.push(col.index());
            }
            Ok(datafusion_common::tree_node::TreeNodeRecursion::Continue)
        });
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Remap `Column` indices inside `expr` from original indices into positions
/// within sorted `needed_cols`.
pub(super) fn remap_expr_columns(
    expr: Arc<dyn PhysicalExpr>,
    needed_cols: &[usize],
) -> Result<Arc<dyn PhysicalExpr>> {
    let transformed = expr.transform_down(|node| {
        if let Some(col) = node.downcast_ref::<Column>() {
            let new_idx = needed_cols.binary_search(&col.index()).map_err(|_| {
                datafusion_common::internal_datafusion_err!(
                    "Column index {} not found in narrowed column map {:?}",
                    col.index(),
                    needed_cols
                )
            })?;
            if new_idx != col.index() {
                return Ok(Transformed::yes(Arc::new(Column::new(col.name(), new_idx))));
            }
        }
        Ok(Transformed::no(node))
    })?;
    Ok(transformed.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::fmt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{Int32Array, Int64Array, StringArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, FieldRef, Schema};
    use datafusion_common::exec_err;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };
    use datafusion_expr::Operator as LogicalOp;
    use datafusion_physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs};

    fn sample_batch(rows: usize) -> RecordBatch {
        let mut metadata = HashMap::new();
        metadata.insert("origin".to_owned(), "morsel-test".to_owned());
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("id", DataType::Int32, false),
                Field::new("val", DataType::Int64, true),
                Field::new("tag", DataType::Utf8, true),
            ],
            metadata,
        ));
        let ids: Vec<i32> = (0..rows as i32).collect();
        let vals: Vec<Option<i64>> = (0..rows)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else {
                    Some(i as i64 * 10)
                }
            })
            .collect();
        let tags: Vec<Option<String>> = (0..rows)
            .map(|i| {
                if i % 7 == 0 {
                    None
                } else {
                    Some(format!("row-{i:04}"))
                }
            })
            .collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int64Array::from(vals)),
                Arc::new(StringArray::from(tags)),
            ],
        )
        .unwrap()
    }

    fn zero_col_batch(rows: usize) -> RecordBatch {
        let mut metadata = HashMap::new();
        metadata.insert("zero".to_owned(), "true".to_owned());
        let schema = Arc::new(Schema::new_with_metadata(Vec::<Field>::new(), metadata));
        RecordBatch::try_new_with_options(
            schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap()
    }

    #[test]
    fn selection_invariants_reject_invalid_masks_and_indices() -> Result<()> {
        let batch = sample_batch(5);

        // Wrong Mask length
        let err = BatchView::try_new(
            batch.clone(),
            Selection::Mask {
                bits: BooleanBuffer::new_set(4),
                selected: 4,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("Mask length 4"), "{err}");

        // Wrong Mask selected count
        let err = BatchView::try_new(
            batch.clone(),
            Selection::Mask {
                bits: BooleanBuffer::from(vec![true, false, true, false, false]),
                selected: 3,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("selected count 3"), "{err}");

        // Null in Indices
        let null_idx = UInt32Array::from(vec![Some(0), None, Some(2)]);
        let err =
            BatchView::try_new(batch.clone(), Selection::Indices(null_idx)).unwrap_err();
        assert!(err.to_string().contains("must not contain nulls"), "{err}");

        // Out of bounds index
        let oob = UInt32Array::from(vec![0, 5]);
        let err = BatchView::try_new(batch.clone(), Selection::Indices(oob)).unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");

        // Duplicate index (not strictly increasing)
        let dup = UInt32Array::from(vec![1, 1, 3]);
        let err = BatchView::try_new(batch.clone(), Selection::Indices(dup)).unwrap_err();
        assert!(err.to_string().contains("strictly increasing"), "{err}");

        // Unsorted index
        let unsorted = UInt32Array::from(vec![3, 1]);
        let err = BatchView::try_new(batch, Selection::Indices(unsorted)).unwrap_err();
        assert!(err.to_string().contains("strictly increasing"), "{err}");

        Ok(())
    }

    #[test]
    fn from_boolean_array_enforces_sql_true_and_mask_bounds() -> Result<()> {
        let batch = sample_batch(4);

        // Value bits are [true, true, false, true], validity is [true, false, true, false].
        // Only index 0 is SQL TRUE (index 1 and 3 have set value bits under NULL).
        let mask = BooleanArray::new(
            BooleanBuffer::from(vec![true, true, false, true]),
            Some(NullBuffer::from(vec![true, false, true, false])),
        );
        let view = BatchView::from_boolean_array(batch.clone(), &mask)?;
        assert_eq!(view.num_rows(), 1);
        assert_eq!(view.physical_rows(), 4);
        let mat = view.materialize()?;
        assert_eq!(mat, filter_record_batch(&batch, &mask)?);
        assert_eq!(mat.schema(), batch.schema());

        // All-selected normalizes to Selection::All
        let all_true = BooleanArray::from(vec![true; 4]);
        let view = BatchView::from_boolean_array(batch.clone(), &all_true)?;
        assert!(view.is_all());
        assert_eq!(view.num_rows(), 4);

        // All-filtered (false + null) normalizes to empty and drops the backing batch
        let all_false_or_null = BooleanArray::from(vec![Some(false), None, None, None]);
        let view = BatchView::from_boolean_array(batch.clone(), &all_false_or_null)?;
        assert!(view.is_empty());
        assert_eq!(view.physical_rows(), 0);
        assert!(view.backing_memory_size() < get_record_batch_memory_size(&batch));
        assert_eq!(view.materialize()?.schema(), batch.schema());

        // Oversized mask rejected
        let oversized = BooleanArray::from(vec![true; 5]);
        let err = BatchView::from_boolean_array(batch.clone(), &oversized).unwrap_err();
        assert!(
            err.to_string().contains(
                "Filter predicate of length 5 is larger than target array of length 4"
            ),
            "{err}"
        );

        // Undersized mask matches Arrow filter_record_batch
        let undersized = BooleanArray::from(vec![true, false]);
        let view = BatchView::from_boolean_array(batch.clone(), &undersized)?;
        assert_eq!(
            view.materialize()?,
            filter_record_batch(&batch, &undersized)?
        );

        Ok(())
    }

    #[test]
    fn sliced_record_batch_all_mask_indices_and_stacked_filters() -> Result<()> {
        let base = sample_batch(20);
        let sliced = base.slice(5, 10); // ids 5..15

        // All
        let v_all = BatchView::all(sliced.clone());
        assert_eq!(v_all.num_rows(), 10);
        assert_eq!(v_all.materialize_ref()?, sliced);

        // Mask with sliced BooleanArray (non-zero bit offset + nulls)
        let base_mask = BooleanArray::new(
            BooleanBuffer::from(vec![
                false, true, true, true, false, true, true, false, true, false, true,
                true, false, true,
            ]),
            Some(NullBuffer::from(vec![
                true, true, false, true, true, true, true, true, false, true, true, true,
                true, true,
            ])),
        );
        let sliced_mask = base_mask.slice(1, 10);
        let v_mask = BatchView::from_boolean_array(sliced.clone(), &sliced_mask)?;
        let expected = filter_record_batch(&sliced, &sliced_mask)?;
        assert_eq!(v_mask.materialize_ref()?, expected);

        // Indices
        let v_indices = v_mask.clone().with_indices_selection();
        assert!(matches!(v_indices.selection(), Selection::Indices(_)));
        assert_eq!(v_indices.materialize_ref()?, expected);

        // Stacked filter: id >= 8 AND id % 2 == 0 (evaluated via two separate filters)
        let pred1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            LogicalOp::GtEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(8)))),
        ));
        let pred2: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            LogicalOp::Lt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(13)))),
        ));
        let stacked = BatchView::all(sliced.clone())
            .filter(&pred1)?
            .filter(&pred2)?;
        assert_eq!(stacked.num_rows(), 5);
        let mat = stacked.materialize()?;
        let ids = mat.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(ids.values(), &[8, 9, 10, 11, 12]);

        // Stacked filter on Indices selection composes in backing coordinates
        let start_indices = BatchView::try_new(
            sliced,
            Selection::Indices(UInt32Array::from(vec![1, 3, 5, 7, 9])), // ids 6,8,10,12,14
        )?;
        let composed = start_indices.filter(&pred1)?.filter(&pred2)?;
        assert!(matches!(composed.selection(), Selection::Indices(_)));
        let mat = composed.materialize()?;
        let ids = mat.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(ids.values(), &[8, 10, 12]);

        Ok(())
    }

    #[test]
    fn zero_column_batches_preserve_row_counts_and_metadata() -> Result<()> {
        let batch = zero_col_batch(6);

        let mask = BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            Some(true),
            None,
        ]);
        let view = BatchView::from_boolean_array(batch.clone(), &mask)?;
        assert_eq!(view.num_rows(), 3);
        let mat = view.clone().materialize()?;
        assert_eq!(mat.num_columns(), 0);
        assert_eq!(mat.num_rows(), 3);
        assert_eq!(mat.schema(), batch.schema());

        // Indices on zero-column batch
        let idx_view = view.with_indices_selection();
        let mat_idx = idx_view.materialize()?;
        assert_eq!(mat_idx.num_columns(), 0);
        assert_eq!(mat_idx.num_rows(), 3);
        assert_eq!(mat_idx.schema(), batch.schema());

        // Undersized mask on zero-column batch
        let undersized = BooleanArray::from(vec![true, false, true]);
        let under_view = BatchView::from_boolean_array(batch.clone(), &undersized)?;
        assert_eq!(under_view.num_rows(), 2);
        assert_eq!(under_view.materialize()?.num_rows(), 2);

        // Oversized mask on zero-column batch rejected
        let oversized = BooleanArray::from(vec![true; 7]);
        assert!(BatchView::from_boolean_array(batch, &oversized).is_err());

        Ok(())
    }

    #[derive(Debug)]
    struct CountingFallibleExpr {
        col: Arc<dyn PhysicalExpr>,
        rows_seen: Arc<AtomicUsize>,
        fail_on_value: i32,
    }

    impl PartialEq for CountingFallibleExpr {
        fn eq(&self, other: &Self) -> bool {
            self.col.eq(&other.col)
                && self.fail_on_value == other.fail_on_value
                && Arc::ptr_eq(&self.rows_seen, &other.rows_seen)
        }
    }

    impl Eq for CountingFallibleExpr {}

    impl std::hash::Hash for CountingFallibleExpr {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.col.hash(state);
            self.fail_on_value.hash(state);
            Arc::as_ptr(&self.rows_seen).hash(state);
        }
    }

    impl fmt::Display for CountingFallibleExpr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "CountingFallibleExpr({})", self.col)
        }
    }

    impl PhysicalExpr for CountingFallibleExpr {
        fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self, f)
        }

        fn return_field(&self, _schema: &Schema) -> Result<FieldRef> {
            Ok(Arc::new(Field::new("out", DataType::Int32, false)))
        }

        fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
            let arr = self.col.evaluate(batch)?.into_array(batch.num_rows())?;
            let ints = arr.as_any().downcast_ref::<Int32Array>().unwrap();
            self.rows_seen.fetch_add(ints.len(), Ordering::Relaxed);
            for &v in ints.values() {
                if v == self.fail_on_value {
                    return exec_err!("evaluated rejected row {v}");
                }
            }
            let doubled: Int32Array = ints.values().iter().map(|&v| v * 2).collect();
            Ok(ColumnarValue::Array(Arc::new(doubled)))
        }

        fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
            vec![&self.col]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn PhysicalExpr>>,
        ) -> Result<Arc<dyn PhysicalExpr>> {
            assert_eq!(children.len(), 1);
            Ok(Arc::new(Self {
                col: Arc::clone(&children[0]),
                rows_seen: Arc::clone(&self.rows_seen),
                fail_on_value: self.fail_on_value,
            }))
        }
    }

    impl CountingFallibleExpr {
        fn into_binary_predicate(
            self: Arc<Self>,
            threshold: i32,
        ) -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(
                self,
                LogicalOp::Gt,
                Arc::new(Literal::new(ScalarValue::Int32(Some(threshold)))),
            ))
        }
    }

    #[test]
    fn stacked_filter_and_projection_never_evaluate_rejected_rows() -> Result<()> {
        // Put `id` at column index 2 (not 0) to verify column index remapping.
        let batch = sample_batch(10).project(&[2, 1, 0])?; // [tag, val, id]
        let seen_filter = Arc::new(AtomicUsize::new(0));
        let seen_proj = Arc::new(AtomicUsize::new(0));

        // First filter keeps only ids >= 7 (rows 7, 8, 9) — excludes 0 (which would error!).
        let pred1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 2)),
            LogicalOp::GtEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(7)))),
        ));

        // Second filter has a custom expr over Column(2) that errors if it sees id == 0.
        let expr2 = Arc::new(CountingFallibleExpr {
            col: Arc::new(Column::new("id", 2)),
            rows_seen: Arc::clone(&seen_filter),
            fail_on_value: 0,
        });
        let pred2 = expr2.into_binary_predicate(14); // 2 * id > 14 => id > 7 => [8, 9]

        let view = BatchView::all(batch.clone())
            .filter(&pred1)?
            .filter(&pred2)?;
        assert_eq!(seen_filter.load(Ordering::Relaxed), 3);
        assert_eq!(view.num_rows(), 2);

        // Pure column projection shares backing arrays zero-copy and preserves selection
        let pure_proj = ProjectionExprs::new([
            ProjectionExpr::new(Arc::new(Column::new("tag", 0)), "tag_alias"),
            ProjectionExpr::new(Arc::new(Column::new("id", 2)), "id"),
        ])
        .make_projector(&batch.schema())?;
        let projected_pure = view.clone().apply_projection(&pure_proj)?;
        assert!(!projected_pure.is_all());
        assert_eq!(projected_pure.num_rows(), 2);
        assert!(Arc::ptr_eq(
            projected_pure.batch().column(0),
            batch.column(0)
        ));
        assert!(Arc::ptr_eq(
            projected_pure.batch().column(1),
            batch.column(2)
        ));

        // Mixed projection with expression evaluates only on the 2 surviving rows
        let proj_expr: Arc<dyn PhysicalExpr> = Arc::new(CountingFallibleExpr {
            col: Arc::new(Column::new("id", 2)),
            rows_seen: Arc::clone(&seen_proj),
            fail_on_value: 0,
        });
        let mixed_proj = ProjectionExprs::new([
            ProjectionExpr::new(Arc::new(Column::new("tag", 0)), "tag"),
            ProjectionExpr::new(proj_expr, "doubled"),
        ])
        .make_projector(&batch.schema())?;
        let projected_mixed = view.apply_projection(&mixed_proj)?;
        assert_eq!(seen_proj.load(Ordering::Relaxed), 2);
        assert!(projected_mixed.is_all());
        let out = projected_mixed.materialize()?;
        assert_eq!(out.num_rows(), 2);
        let doubled = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(doubled.values(), &[16, 18]);

        Ok(())
    }

    #[test]
    fn backing_buffer_compaction_policy_and_memory_limits() -> Result<()> {
        // Large backing batch: 8192 rows with wide strings (~1 MiB backing buffer)
        let rows = 8192usize;
        let payload = "x".repeat(128);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..rows as i64)),
                Arc::new(StringArray::from(vec![payload.as_str(); rows])),
            ],
        )?;
        let full_bytes = get_record_batch_memory_size(&batch);
        assert!(full_bytes > 1024 * 1024);

        // Select just 16 rows (~0.2%)
        let mut mask_vals = vec![false; rows];
        for i in (0..rows).step_by(512) {
            mask_vals[i] = true;
        }
        let mask = BooleanArray::from(mask_vals);
        let view = BatchView::from_boolean_array(batch.clone(), &mask)?;
        assert_eq!(view.num_rows(), 16);
        assert_eq!(view.backing_memory_size(), full_bytes);
        assert!(view.should_compact_for_retention(
            DEFAULT_COMPACT_MIN_WASTED_BYTES,
            DEFAULT_COMPACT_MAX_RETENTION_RATIO
        ));

        // Tight pool (64 KiB) cannot hold the 1 MiB backing buffer, but succeeds
        // by compacting the 16-row selection (~3 KiB).
        let tight_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024));
        let mut res = MemoryConsumer::new("test_batch_view").register(&tight_pool);
        let reserved_view = view.into_reserved_for_retention(&mut res)?;
        assert!(reserved_view.is_all());
        assert_eq!(reserved_view.num_rows(), 16);
        assert!(res.size() < 64 * 1024);
        assert_eq!(res.size(), reserved_view.retained_memory_size());
        drop(reserved_view);
        drop(res);
        assert_eq!(tight_pool.reserved(), 0);

        // Small batch with low selectivity does NOT compact when wasted bytes < threshold.
        let small = sample_batch(16);
        let mut small_mask = vec![false; 16];
        small_mask[0] = true;
        let small_view =
            BatchView::from_boolean_array(small, &BooleanArray::from(small_mask))?;
        assert!(!small_view.should_compact_for_retention(
            DEFAULT_COMPACT_MIN_WASTED_BYTES,
            DEFAULT_COMPACT_MAX_RETENTION_RATIO
        ));

        // Moderate selectivity (40%) fails proactive 4x heuristic, yet still compacts
        // under a tight memory pool that fits compacted bytes but not full backing bytes.
        let mod_mask =
            BooleanArray::from((0..rows).map(|i| i % 5 < 2).collect::<Vec<_>>());
        let mod_view = BatchView::from_boolean_array(batch, &mod_mask)?;
        assert!(!mod_view.should_compact_for_retention(
            DEFAULT_COMPACT_MIN_WASTED_BYTES,
            DEFAULT_COMPACT_MAX_RETENTION_RATIO
        ));
        let mod_est = mod_view.estimated_compacted_bytes();
        assert!(mod_est < full_bytes);
        let mid_pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(mod_est + 128 * 1024));
        assert_eq!(mid_pool.reserved(), 0);
        let mut mid_res = MemoryConsumer::new("test_mid_pool").register(&mid_pool);
        let compacted = mod_view.into_reserved_for_retention(&mut mid_res)?;
        assert!(compacted.is_all());
        let expected_mod_rows = (0..rows).filter(|i| i % 5 < 2).count();
        assert_eq!(compacted.num_rows(), expected_mod_rows);
        assert!(mid_res.size() < full_bytes);
        drop(compacted);
        drop(mid_res);
        assert_eq!(mid_pool.reserved(), 0);

        // Unbounded pool still cleans up properly
        let unbounded: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        assert_eq!(unbounded.reserved(), 0);

        Ok(())
    }

    #[test]
    fn selection_conversions_intersections_and_scalar_predicates() -> Result<()> {
        let batch = sample_batch(8);
        let mask1 = BooleanArray::from(vec![
            Some(true),
            Some(false),
            Some(true),
            None,
            Some(true),
            Some(true),
            Some(false),
            Some(true),
        ]);
        let mask2 = BooleanArray::from(vec![
            Some(false),
            Some(true),
            Some(true),
            Some(true),
            None,
            Some(true),
            Some(true),
            Some(true),
        ]);

        // Intersect backing masks across All, Mask, and Indices
        let sel_from_all = Selection::All.intersect_backing_mask(8, &mask1)?;
        let sel_mask = sel_from_all.intersect_backing_mask(8, &mask2)?;
        assert_eq!(sel_mask.selected_rows(8), 3); // rows 2, 5, 7

        let idx_init = Selection::Indices(UInt32Array::from(vec![0, 2, 4, 5, 7]));
        let sel_idx = idx_init.intersect_backing_mask(8, &mask2)?;
        assert_eq!(sel_idx.selected_rows(8), 3);
        assert_eq!(sel_idx.to_mask(8), sel_mask.to_mask(8));
        assert_eq!(sel_idx.to_indices(8), sel_mask.to_indices(8));

        // materialize_column matches across Mask and Indices
        let view_mask = BatchView::try_new(batch.clone(), sel_mask)?;
        let view_idx = BatchView::try_new(batch.clone(), sel_idx)?;
        for col in 0..batch.num_columns() {
            assert_eq!(
                view_mask.materialize_column(col)?.as_ref(),
                view_idx.materialize_column(col)?.as_ref()
            );
        }

        // Scalar predicates (true, false, NULL, non-boolean) on All and Mask views
        let lit_true: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true))));
        let lit_false: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false))));
        let lit_null: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(None)));
        let lit_int: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Int32(Some(1))));

        for base in [BatchView::all(batch.clone()), view_mask.clone(), view_idx] {
            let expected_rows = base.num_rows();
            assert_eq!(base.clone().filter(&lit_true)?.num_rows(), expected_rows);
            assert!(base.clone().filter(&lit_false)?.is_empty());
            assert!(base.clone().filter(&lit_null)?.is_empty());
            let err = base.filter(&lit_int).unwrap_err();
            assert!(err.to_string().contains("non-boolean predicates"), "{err}");
        }

        // Empty batch passes through cleanly
        let empty_view = BatchView::empty(batch.schema());
        assert!(empty_view.clone().filter(&lit_true)?.is_empty());
        assert_eq!(empty_view.materialize()?.num_rows(), 0);

        Ok(())
    }
}
