// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`DataChunk`] and [`SelectionVector`]: the unit of data that flows through
//! a [`Pipeline`](crate::pipeline::Pipeline).
//!
//! A [`DataChunk`] is a set of Arrow columns plus an optional
//! [`SelectionVector`]. The selection vector lists the *physical* row indices
//! of the columns that are logically part of the chunk, which lets selective
//! operators (such as filters) discard rows without copying any column data.
//! The selection is only materialized (with Arrow's `filter` or `take`
//! kernels, see [`Gather`]) when a chunk leaves the pipeline engine or an
//! operator needs contiguous arrays, see [`DataChunk::compact`].

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    new_empty_array,
};
use arrow::datatypes::{DataType, SchemaRef};
use arrow_buffer::{BooleanBuffer, Buffer, ScalarBuffer};
use arrow_select::filter::{FilterBuilder, FilterPredicate};
use arrow_select::take::take;
use datafusion_common::{Result, exec_err, internal_datafusion_err};

/// The target number of rows in a [`DataChunk`] (DuckDB's `STANDARD_VECTOR_SIZE`).
///
/// Sources split their output into chunks of at most this many rows, so that
/// the working set of a pipeline stays cache resident.
pub const DEFAULT_VECTOR_SIZE: usize = 2048;

/// A list of physical row indices selected from the columns of a [`DataChunk`].
///
/// The indices are stored in an immutable, reference-counted `[u32]` (an
/// Arrow [`ScalarBuffer<u32>`]), so cloning and slicing a selection is O(1)
/// and it can be handed to Arrow's `take` kernel without copying.
///
/// Every index must be smaller than the number of physical rows of the
/// columns the selection is applied to. Indices are usually, but not
/// necessarily, strictly increasing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionVector {
    indices: ScalarBuffer<u32>,
}

impl SelectionVector {
    /// Create a selection from a list of row indices.
    pub fn new(indices: impl Into<ScalarBuffer<u32>>) -> Self {
        Self {
            indices: indices.into(),
        }
    }

    /// Create a selection of the rows `0..len`.
    pub fn identity(len: usize) -> Result<Self> {
        let len = u32::try_from(len).map_err(|_| {
            internal_datafusion_err!("SelectionVector length {len} exceeds u32::MAX")
        })?;
        Ok(Self::new((0..len).collect::<Vec<_>>()))
    }

    /// Create a selection of the rows for which `mask` is `true`.
    ///
    /// Following SQL `WHERE` semantics, `NULL` mask entries are not selected.
    pub fn from_boolean_mask(mask: &BooleanArray) -> Self {
        let selected = match mask.nulls() {
            Some(nulls) if nulls.null_count() > 0 => mask.values() & nulls.inner(),
            _ => mask.values().clone(),
        };
        Self::new(selected.set_indices_u32().collect::<Vec<_>>())
    }

    /// Number of selected rows.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if no row is selected.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The selected physical row indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Iterate over the selected physical row indices.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices.iter().map(|&i| i as usize)
    }

    /// The largest selected index, or `None` if the selection is empty.
    pub fn max_index(&self) -> Option<usize> {
        self.indices.iter().max().map(|&i| i as usize)
    }

    /// Return the entries `offset..offset + len` of this selection (O(1)).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len()`.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset.saturating_add(len) <= self.len(),
            "SelectionVector slice out of bounds: offset {offset} + len {len} > {}",
            self.len()
        );
        Self {
            indices: self.indices.slice(offset, len),
        }
    }

    /// Compose two selections: apply `other`, whose entries index the rows
    /// selected by `self`, on top of `self`.
    ///
    /// The `i`-th entry of the result is `self[other[i]]`. This is how a
    /// selective operator refines the selection of a chunk that is already
    /// filtered: it computes `other` relative to the logical rows of its input
    /// and composes it with the input's selection.
    pub fn slice_sel(&self, other: &SelectionVector) -> Result<Self> {
        let indices = other
            .indices
            .iter()
            .map(|&i| {
                self.indices.get(i as usize).copied().ok_or_else(|| {
                    internal_datafusion_err!(
                        "SelectionVector index {i} out of bounds for selection of length {}",
                        self.len()
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(indices))
    }

    /// If the selection is the contiguous range `start..start + len`, return
    /// `Some((start, len))`.
    pub fn as_contiguous_range(&self) -> Option<(usize, usize)> {
        let first = *self.indices.first()?;
        let contiguous = self
            .indices
            .iter()
            .zip(first..)
            .all(|(&actual, expected)| actual == expected);
        contiguous.then_some((first as usize, self.len()))
    }

    /// Returns `true` if the selected indices are strictly increasing (also if
    /// the selection is empty or contiguous), like those of
    /// [`Self::from_boolean_mask`]: the selection keeps the order of the rows,
    /// and selects each at most once.
    pub fn is_strictly_increasing(&self) -> bool {
        !matches!(self.order(), Order::Unordered)
    }

    /// The order of the selected indices, found in one pass over them
    /// (without branches, so that it vectorizes).
    fn order(&self) -> Order {
        let indices = self.indices.as_ref();
        let Some(&first) = indices.first() else {
            return Order::Empty;
        };
        // The pairs of consecutive indices that do not increase, and that do
        // not increase by one.
        let (descents, gaps) = indices[1..].iter().zip(indices).fold(
            (0_usize, 0_usize),
            |(descents, gaps), (&next, &previous)| {
                (
                    descents + usize::from(next <= previous),
                    gaps + usize::from(next.wrapping_sub(previous) != 1),
                )
            },
        );
        match (descents, gaps) {
            (0, 0) => Order::Contiguous(first as usize),
            (0, _) => Order::Increasing,
            _ => Order::Unordered,
        }
    }

    /// The selection as an Arrow [`UInt32Array`], e.g. for the `take` kernel.
    ///
    /// This does not copy the indices.
    pub fn to_uint32_array(&self) -> UInt32Array {
        UInt32Array::new(self.indices.clone(), None)
    }

    /// Return the underlying index buffer.
    pub fn into_inner(self) -> ScalarBuffer<u32> {
        self.indices
    }
}

impl From<Vec<u32>> for SelectionVector {
    fn from(indices: Vec<u32>) -> Self {
        Self::new(indices)
    }
}

impl From<ScalarBuffer<u32>> for SelectionVector {
    fn from(indices: ScalarBuffer<u32>) -> Self {
        Self::new(indices)
    }
}

/// A batch of rows flowing through a pipeline: Arrow columns plus an optional
/// [`SelectionVector`].
///
/// * `columns` hold the physical data; all columns have the same length.
/// * `selection`, if present, lists the physical rows that are logically part
///   of the chunk, in output order.
/// * `len` is the logical number of rows: the length of the selection if one
///   is present, otherwise the number of physical rows. It is stored
///   explicitly so that chunks without columns (e.g. the input of
///   `COUNT(*)`) still carry a row count.
///
/// A chunk can also be *cleared* (see [`DataChunk::empty`] and
/// [`DataChunk::reset`]): it then has no rows and no columns, but keeps its
/// schema. This is the state of an output chunk before an operator writes to
/// it, which avoids allocating empty arrays on every call.
#[derive(Debug, Clone)]
pub struct DataChunk {
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    selection: Option<SelectionVector>,
    len: usize,
}

impl DataChunk {
    /// Create a chunk from columns and an optional selection, validating that
    /// the columns match `schema` and the selection is in bounds.
    pub fn try_new(
        schema: SchemaRef,
        columns: Vec<ArrayRef>,
        selection: Option<SelectionVector>,
    ) -> Result<Self> {
        if columns.len() != schema.fields().len() {
            return exec_err!(
                "DataChunk has {} columns but its schema has {} fields",
                columns.len(),
                schema.fields().len()
            );
        }
        let physical_len = columns.first().map_or(0, |c| c.len());
        for (column, field) in columns.iter().zip(schema.fields()) {
            if column.len() != physical_len {
                return exec_err!(
                    "DataChunk columns have different lengths: {} vs {physical_len}",
                    column.len()
                );
            }
            if column.data_type() != field.data_type() {
                return exec_err!(
                    "DataChunk column '{}' has type {} but the schema expects {}",
                    field.name(),
                    column.data_type(),
                    field.data_type()
                );
            }
        }
        let len = match &selection {
            Some(selection) => {
                check_selection_bounds(selection, physical_len)?;
                selection.len()
            }
            None => physical_len,
        };
        Ok(Self {
            schema,
            columns,
            selection,
            len,
        })
    }

    /// Create a chunk without columns that has `len` rows.
    ///
    /// `schema` must not have any fields.
    pub fn try_new_without_columns(schema: SchemaRef, len: usize) -> Result<Self> {
        if !schema.fields().is_empty() {
            return exec_err!(
                "DataChunk::try_new_without_columns requires a schema without fields"
            );
        }
        Ok(Self {
            schema,
            columns: vec![],
            selection: None,
            len,
        })
    }

    /// Create a cleared chunk (no rows, no columns) with the given schema.
    pub fn empty(schema: SchemaRef) -> Self {
        Self {
            schema,
            columns: vec![],
            selection: None,
            len: 0,
        }
    }

    /// Create a chunk from a [`RecordBatch`] without copying any data.
    pub fn from_record_batch(batch: RecordBatch) -> Self {
        let (schema, columns, len) = batch.into_parts();
        Self {
            schema,
            columns,
            selection: None,
            len,
        }
    }

    /// Restrict this chunk to the rows in `selection`.
    ///
    /// The entries of `selection` index the *logical* rows of this chunk
    /// (`0..self.len()`). If the chunk already has a selection, the two are
    /// composed (see [`SelectionVector::slice_sel`]), so no data is copied.
    pub fn with_selection(mut self, selection: SelectionVector) -> Result<Self> {
        check_selection_bounds(&selection, self.len)?;
        if self.is_cleared() {
            // `selection` is empty because the chunk has no rows.
            return Ok(self);
        }
        let selection = match self.selection.take() {
            Some(existing) => existing.slice_sel(&selection)?,
            None => selection,
        };
        self.len = selection.len();
        self.selection = Some(selection);
        Ok(self)
    }

    /// The schema of this chunk.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The physical columns of this chunk.
    ///
    /// If the chunk has a selection, only the rows in the selection are
    /// logically part of the chunk. Cleared chunks have no columns.
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns
    }

    /// The physical column at index `i`. See [`Self::columns`].
    pub fn column(&self, i: usize) -> &ArrayRef {
        &self.columns[i]
    }

    /// The number of columns in the schema of this chunk.
    pub fn num_columns(&self) -> usize {
        self.schema.fields().len()
    }

    /// The selection of this chunk, if any.
    pub fn selection(&self) -> Option<&SelectionVector> {
        self.selection.as_ref()
    }

    /// The logical number of rows in this chunk.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this chunk has no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of rows of the physical columns of this chunk.
    pub fn physical_len(&self) -> usize {
        self.columns.first().map_or(self.len, |c| c.len())
    }

    /// Clear this chunk: remove all rows and columns but keep the schema.
    pub fn reset(&mut self) {
        self.columns.clear();
        self.selection = None;
        self.len = 0;
    }

    /// Return the logical rows `offset..offset + length` of this chunk.
    ///
    /// This never copies column data: a chunk with a selection is sliced by
    /// slicing its selection, otherwise the columns are sliced.
    ///
    /// # Panics
    ///
    /// Panics if `offset + length > self.len()`.
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        assert!(
            offset.saturating_add(length) <= self.len,
            "DataChunk slice out of bounds: offset {offset} + length {length} > {}",
            self.len
        );
        if self.is_cleared() {
            return self.clone();
        }
        match &self.selection {
            Some(selection) => Self {
                schema: Arc::clone(&self.schema),
                columns: self.columns.clone(),
                selection: Some(selection.slice(offset, length)),
                len: length,
            },
            None => Self {
                schema: Arc::clone(&self.schema),
                columns: self
                    .columns
                    .iter()
                    .map(|c| c.slice(offset, length))
                    .collect(),
                selection: None,
                len: length,
            },
        }
    }

    /// Return a chunk with only the columns at `indices` (O(1) per column).
    ///
    /// The selection, if any, is preserved.
    pub fn project(&self, indices: &[usize]) -> Result<Self> {
        let schema = Arc::new(self.schema.project(indices)?);
        let columns = if self.is_cleared() {
            vec![]
        } else {
            indices
                .iter()
                .map(|&i| Arc::clone(&self.columns[i]))
                .collect()
        };
        Ok(Self {
            schema,
            columns,
            selection: self.selection.clone(),
            len: self.len,
        })
    }

    /// Return the logical rows of column `i` as a contiguous array.
    ///
    /// This copies data only if the chunk has a selection that is not a
    /// contiguous range (see [`Gather`]).
    pub fn compacted_column(&self, i: usize) -> Result<ArrayRef> {
        if self.is_cleared() {
            return Ok(new_empty_array(self.schema.field(i).data_type()));
        }
        let column = &self.columns[i];
        match &self.selection {
            Some(selection) => {
                Gather::new(selection, column.len(), [column.data_type()]).column(column)
            }
            None => Ok(Arc::clone(column)),
        }
    }

    /// Materialize the logical rows of this chunk as a [`RecordBatch`].
    ///
    /// If the chunk has a selection, the selected rows are zero-copy sliced if
    /// the selection is a contiguous range, copied with Arrow's `filter` kernel
    /// if they are many and in order, and gathered with its `take` kernel
    /// otherwise, with a strategy decided once for all columns (see
    /// [`Gather`]). Otherwise the columns are returned as is.
    pub fn compact(&self) -> Result<RecordBatch> {
        self.clone().into_record_batch()
    }

    /// Like [`Self::compact`], but consumes the chunk, avoiding a clone of
    /// the column list.
    pub fn into_record_batch(self) -> Result<RecordBatch> {
        if self.is_cleared() {
            return Ok(RecordBatch::new_empty(self.schema));
        }
        let columns = match &self.selection {
            Some(selection) => {
                let gather = Gather::new(
                    selection,
                    self.physical_len(),
                    self.schema.fields().iter().map(|field| field.data_type()),
                );
                self.columns
                    .iter()
                    .map(|column| gather.column(column))
                    .collect::<Result<Vec<_>>>()?
            }
            None => self.columns,
        };
        let options = RecordBatchOptions::new().with_row_count(Some(self.len));
        Ok(RecordBatch::try_new_with_options(
            self.schema,
            columns,
            &options,
        )?)
    }

    /// Returns `true` if this chunk is cleared, i.e. has neither rows nor
    /// columns although its schema has fields.
    fn is_cleared(&self) -> bool {
        self.columns.is_empty() && !self.schema.fields().is_empty()
    }
}

impl From<RecordBatch> for DataChunk {
    fn from(batch: RecordBatch) -> Self {
        Self::from_record_batch(batch)
    }
}

fn check_selection_bounds(selection: &SelectionVector, len: usize) -> Result<()> {
    match selection.max_index() {
        Some(max) if max >= len => exec_err!(
            "SelectionVector index {max} out of bounds for a chunk with {len} rows"
        ),
        _ => Ok(()),
    }
}

/// The order of the indices of a [`SelectionVector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    /// No index.
    Empty,
    /// The contiguous range of indices from the given start.
    Contiguous(usize),
    /// Strictly increasing indices, but not contiguous.
    Increasing,
    /// Any other order, or repeated indices.
    Unordered,
}

/// How the rows of a [`SelectionVector`] are gathered from the columns of a
/// chunk (all of the same physical length): decided once, for all of them.
///
/// * A contiguous range of rows (or no row) is sliced, without copying any
///   data.
/// * Strictly increasing rows, if more than half of the physical rows are
///   selected, are copied with Arrow's `filter` kernel, with a predicate
///   built once for all columns, as DataFusion's `FilterExec` does, if that
///   pays off (see `filter_pays_off`): `filter` copies runs of rows at
///   once, which is faster than gathering them one by one. Only flat columns
///   (primitive values, booleans, strings, binaries and their views) are
///   filtered: for these, `filter` and `take` produce the same arrays (but for
///   the bytes beneath the nulls of strings and binaries, which `filter`
///   keeps, as `FilterExec` does), others are gathered with `take`.
/// * Other rows are gathered with Arrow's `take` kernel.
///
/// On a chunk of 2048 rows with 10 columns (4 `Int64`, 2 `Float64`, 2 `Utf8`
/// and 2 `Utf8View`), this is 1.1x to 1.3x faster than gathering every column
/// with `take` if 60% to 75% of the rows are selected, 1.5x if 85% are, 1.6x
/// to 1.8x if 92% are and 1.9x if 99% are, within 5% of `filter_record_batch`
/// with a ready mask; and 1.4x to 1.7x faster on a single column of strings
/// if more than 85% of its rows are selected.
#[derive(Debug)]
pub struct Gather {
    strategy: GatherStrategy,
}

#[derive(Debug)]
enum GatherStrategy {
    /// The rows `offset..offset + len`.
    Slice { offset: usize, len: usize },
    /// Dense, strictly increasing rows of columns of `len` rows: `predicate`
    /// selects them for `filter`, `indices` for `take`.
    Filter {
        len: usize,
        predicate: FilterPredicate,
        indices: UInt32Array,
    },
    /// Any rows, for `take`.
    Take(UInt32Array),
}

impl Gather {
    /// How to gather the rows of `selection` from columns of `physical_len`
    /// rows (more than any index of `selection`), of the types `data_types`.
    pub fn new<'a>(
        selection: &SelectionVector,
        physical_len: usize,
        data_types: impl IntoIterator<Item = &'a DataType>,
    ) -> Self {
        // Only dense selections of columns that `filter` pays off for may be
        // filtered: others are only checked for a contiguous range, which
        // stops at their first gap.
        let selected = selection.len();
        let filterable = selected.saturating_mul(2) > physical_len
            && filter_pays_off(selected, physical_len, data_types);
        let order = if filterable {
            selection.order()
        } else {
            match selection.as_contiguous_range() {
                Some((offset, _)) => Order::Contiguous(offset),
                None if selection.is_empty() => Order::Empty,
                None => Order::Unordered,
            }
        };
        let strategy = match order {
            Order::Empty => GatherStrategy::Slice { offset: 0, len: 0 },
            Order::Contiguous(offset) => GatherStrategy::Slice {
                offset,
                len: selected,
            },
            // The last index of increasing indices is the largest one.
            Order::Increasing
                if filterable
                    && selection
                        .indices()
                        .last()
                        .is_some_and(|&last| (last as usize) < physical_len) =>
            {
                let mask =
                    BooleanArray::new(increasing_mask(selection, physical_len), None);
                GatherStrategy::Filter {
                    len: physical_len,
                    predicate: FilterBuilder::new(&mask).optimize().build(),
                    indices: selection.to_uint32_array(),
                }
            }
            Order::Increasing | Order::Unordered => {
                GatherStrategy::Take(selection.to_uint32_array())
            }
        };
        Self { strategy }
    }

    /// The name of the strategy, for tests.
    #[cfg(test)]
    fn strategy(&self) -> &'static str {
        match self.strategy {
            GatherStrategy::Slice { .. } => "slice",
            GatherStrategy::Filter { .. } => "filter",
            GatherStrategy::Take(_) => "take",
        }
    }

    /// The selected rows of `column`, a contiguous array.
    pub fn column(&self, column: &ArrayRef) -> Result<ArrayRef> {
        match &self.strategy {
            GatherStrategy::Slice { offset, len } => {
                if *offset == 0 && *len == column.len() {
                    Ok(Arc::clone(column))
                } else {
                    Ok(column.slice(*offset, *len))
                }
            }
            GatherStrategy::Filter {
                len,
                predicate,
                indices,
            } => {
                if *len == column.len() && filters_like_take(column.data_type()) {
                    Ok(predicate.filter(column.as_ref())?)
                } else {
                    Ok(take(column.as_ref(), indices, None)?)
                }
            }
            GatherStrategy::Take(indices) => Ok(take(column.as_ref(), indices, None)?),
        }
    }
}

/// Whether copying `selected` of `physical_len` rows (more than half of them,
/// strictly increasing) of columns of `data_types` with Arrow's `filter`
/// kernel is faster than gathering them with its `take` kernel, as measured
/// on chunks of 2048 rows.
///
/// Building the predicate costs (about 3.5 µs) as much as `filter` saves on 4
/// flat columns (1 to 3 µs each: e.g. 1 µs of `Int64` values if 60% of the
/// rows are selected, 2.5 µs if 95% are), but on a single column of strings
/// or binaries if more than 80% of the rows are selected (8 to 12 µs): then
/// the predicate copies runs of rows, and `filter` copies the bytes of a run
/// at once, where `take` copies the bytes of each value.
fn filter_pays_off<'a>(
    selected: usize,
    physical_len: usize,
    data_types: impl IntoIterator<Item = &'a DataType>,
) -> bool {
    let runs = selected.saturating_mul(5) > physical_len.saturating_mul(4);
    let savings = data_types
        .into_iter()
        .map(|data_type| match data_type {
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
                if runs =>
            {
                4
            }
            data_type if filters_like_take(data_type) => 1,
            _ => 0,
        })
        .sum::<usize>();
    savings >= 4
}

/// The bitmap of `len` bits whose set bits are the strictly increasing
/// indices of `selection` (all smaller than `len`).
fn increasing_mask(selection: &SelectionVector, len: usize) -> BooleanBuffer {
    let mut words = vec![0_u64; len.div_ceil(64)];
    // The bits of a word are collected in a register, and stored once: the
    // indices are increasing, so the bits of a word are consecutive.
    let mut index = 0;
    let mut word = 0_u64;
    for &row in selection.indices() {
        let row = row as usize;
        if row / 64 != index {
            words[index] = word;
            index = row / 64;
            word = 0;
        }
        word |= 1 << (row % 64);
    }
    if let Some(last) = words.get_mut(index) {
        *last |= word;
    }
    BooleanBuffer::new(Buffer::from_vec(words), 0, len)
}

/// Whether Arrow's `filter` kernel copies the selected rows of arrays of
/// `data_type` exactly like its `take` kernel gathers them, with the same
/// values, nulls and type, from the same buffers for views: flat arrays of
/// primitive values (numbers, decimals, temporal values), booleans, strings
/// and binaries (also of fixed sizes) and their views. (`filter` keeps the
/// bytes beneath the nulls of strings and binaries, which `take` drops, as
/// DataFusion's `FilterExec` does.) Arrow's `filter` rebuilds other arrays
/// differently, e.g. run-end encoded arrays with its own child fields.
fn filters_like_take(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{AsArray, Int32Array, StringArray, StringViewArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};

    fn batch(n: i32) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        let a = Int32Array::from_iter((0..n).map(|i| (i % 7 != 3).then_some(i)));
        let b = StringArray::from_iter((0..n).map(|i| Some(format!("s{i}"))));
        RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
    }

    fn int_values(batch: &RecordBatch) -> Vec<Option<i32>> {
        batch
            .column(0)
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>()
    }

    #[test]
    fn selection_from_boolean_mask_skips_nulls_and_false() {
        let mask = BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(true),
            None,
        ]);
        let selection = SelectionVector::from_boolean_mask(&mask);
        assert_eq!(selection.indices(), &[0, 3, 4]);
        assert_eq!(selection.len(), 3);
        assert!(!selection.is_empty());

        // A sliced mask must respect the slice offset.
        let sliced = mask.slice(2, 4);
        let selection = SelectionVector::from_boolean_mask(&sliced);
        assert_eq!(selection.indices(), &[1, 2]);

        let none =
            SelectionVector::from_boolean_mask(&BooleanArray::from(vec![false, false]));
        assert!(none.is_empty());
        assert_eq!(none.max_index(), None);
    }

    #[test]
    fn selection_compose_and_slice() {
        let outer = SelectionVector::from(vec![1, 3, 5, 7, 9]);
        let inner = SelectionVector::from(vec![0, 2, 4]);
        let composed = outer.slice_sel(&inner).unwrap();
        assert_eq!(composed.indices(), &[1, 5, 9]);

        let err = outer
            .slice_sel(&SelectionVector::from(vec![5]))
            .unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");

        let sliced = outer.slice(1, 3);
        assert_eq!(sliced.indices(), &[3, 5, 7]);
        assert_eq!(sliced.iter().collect::<Vec<_>>(), vec![3, 5, 7]);
        assert_eq!(sliced.to_uint32_array().values().as_ref(), &[3, 5, 7]);
    }

    #[test]
    fn selection_identity_and_contiguous_ranges() {
        let identity = SelectionVector::identity(4).unwrap();
        assert_eq!(identity.indices(), &[0, 1, 2, 3]);
        assert_eq!(identity.as_contiguous_range(), Some((0, 4)));
        assert_eq!(
            SelectionVector::from(vec![3, 4, 5]).as_contiguous_range(),
            Some((3, 3))
        );
        assert_eq!(
            SelectionVector::from(vec![3, 5]).as_contiguous_range(),
            None
        );
        assert_eq!(
            SelectionVector::from(Vec::<u32>::new()).as_contiguous_range(),
            None
        );
    }

    #[test]
    fn chunk_round_trips_record_batch_without_copy() {
        let batch = batch(10);
        let chunk = DataChunk::from_record_batch(batch.clone());
        assert_eq!(chunk.len(), 10);
        assert_eq!(chunk.physical_len(), 10);
        assert_eq!(chunk.num_columns(), 2);
        assert!(chunk.selection().is_none());

        let compacted = chunk.compact().unwrap();
        assert_eq!(compacted, batch);
        // No selection: the very same buffers are returned.
        assert_eq!(
            compacted.column(0).to_data().buffers()[0].as_ptr(),
            batch.column(0).to_data().buffers()[0].as_ptr()
        );
        assert_eq!(chunk.into_record_batch().unwrap(), batch);
    }

    #[test]
    fn chunk_with_selection_compacts_with_take() {
        let batch = batch(10);
        let chunk = DataChunk::from_record_batch(batch.clone())
            .with_selection(SelectionVector::from(vec![1, 3, 4, 8]))
            .unwrap();
        assert_eq!(chunk.len(), 4);
        assert_eq!(chunk.physical_len(), 10);
        assert!(!chunk.is_empty());

        let compacted = chunk.compact().unwrap();
        assert_eq!(compacted.num_rows(), 4);
        assert_eq!(compacted.schema(), batch.schema());
        assert_eq!(
            int_values(&compacted),
            vec![Some(1), None, Some(4), Some(8)]
        );
        assert_eq!(
            compacted.column(1).as_string::<i32>().value(3),
            "s8".to_string()
        );

        // A second selection is relative to the logical rows and is composed.
        let refined = chunk
            .with_selection(SelectionVector::from(vec![0, 3]))
            .unwrap();
        assert_eq!(refined.selection().unwrap().indices(), &[1, 8]);
        assert_eq!(
            int_values(&refined.into_record_batch().unwrap()),
            vec![Some(1), Some(8)]
        );
    }

    #[test]
    fn chunk_selection_out_of_bounds_is_an_error() {
        let chunk = DataChunk::from_record_batch(batch(3));
        let err = chunk
            .clone()
            .with_selection(SelectionVector::from(vec![0, 3]))
            .unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");

        let err = DataChunk::try_new(
            Arc::clone(chunk.schema()),
            chunk.columns().to_vec(),
            Some(SelectionVector::from(vec![7])),
        )
        .unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");
    }

    #[test]
    fn chunk_contiguous_selection_is_zero_copy() {
        let batch = batch(10);
        let chunk = DataChunk::from_record_batch(batch.clone())
            .with_selection(SelectionVector::from(vec![2, 3, 4]))
            .unwrap();
        let compacted = chunk.compact().unwrap();
        let expected = batch.slice(2, 3);
        assert_eq!(compacted, expected);
        assert_eq!(
            compacted.column(0).to_data().buffers()[0].as_ptr(),
            expected.column(0).to_data().buffers()[0].as_ptr()
        );

        let empty = DataChunk::from_record_batch(batch.clone())
            .with_selection(SelectionVector::from(Vec::<u32>::new()))
            .unwrap();
        assert!(empty.is_empty());
        let compacted = empty.compact().unwrap();
        assert_eq!(compacted.num_rows(), 0);
        assert_eq!(compacted.schema(), batch.schema());
    }

    #[test]
    fn chunk_slice_with_and_without_selection() {
        let batch = batch(10);
        let chunk = DataChunk::from_record_batch(batch.clone());
        let sliced = chunk.slice(3, 4);
        assert_eq!(sliced.len(), 4);
        assert_eq!(sliced.compact().unwrap(), batch.slice(3, 4));

        let selected = chunk
            .with_selection(SelectionVector::from(vec![0, 2, 4, 6, 8]))
            .unwrap();
        let sliced = selected.slice(1, 3);
        assert_eq!(sliced.len(), 3);
        // Slicing a selected chunk slices the selection, not the columns.
        assert_eq!(sliced.physical_len(), 10);
        assert_eq!(sliced.selection().unwrap().indices(), &[2, 4, 6]);
        assert_eq!(
            int_values(&sliced.compact().unwrap()),
            vec![Some(2), Some(4), Some(6)]
        );
        assert!(selected.slice(5, 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "DataChunk slice out of bounds")]
    fn chunk_slice_out_of_bounds_panics() {
        let _ = DataChunk::from_record_batch(batch(4)).slice(3, 2);
    }

    #[test]
    fn chunk_project_keeps_selection() {
        let batch = batch(6);
        let chunk = DataChunk::from_record_batch(batch)
            .with_selection(SelectionVector::from(vec![5, 1]))
            .unwrap();
        let projected = chunk.project(&[1]).unwrap();
        assert_eq!(projected.num_columns(), 1);
        assert_eq!(projected.schema().field(0).name(), "b");
        let compacted = projected.compact().unwrap();
        let values = compacted.column(0).as_string::<i32>();
        assert_eq!(values.value(0), "s5");
        assert_eq!(values.value(1), "s1");
        assert_eq!(
            chunk
                .compacted_column(0)
                .unwrap()
                .as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(5), Some(1)])
        );
    }

    #[test]
    fn cleared_and_columnless_chunks() {
        let batch = batch(3);
        let mut chunk = DataChunk::from_record_batch(batch.clone());
        chunk.reset();
        assert!(chunk.is_empty());
        assert!(chunk.columns().is_empty());
        assert_eq!(chunk.num_columns(), 2);
        let compacted = chunk.compact().unwrap();
        assert_eq!(compacted.num_rows(), 0);
        assert_eq!(compacted.schema(), batch.schema());
        assert_eq!(
            chunk.compacted_column(1).unwrap().data_type(),
            &DataType::Utf8
        );
        assert!(DataChunk::empty(batch.schema()).slice(0, 0).is_empty());

        // Chunks without columns still carry a row count.
        let schema = Arc::new(Schema::empty());
        let chunk = DataChunk::try_new_without_columns(Arc::clone(&schema), 5).unwrap();
        assert_eq!(chunk.len(), 5);
        assert_eq!(chunk.physical_len(), 5);
        assert_eq!(chunk.compact().unwrap().num_rows(), 5);
        let chunk = DataChunk::from_record_batch(
            RecordBatch::try_new_with_options(
                schema,
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(7)),
            )
            .unwrap(),
        )
        .with_selection(SelectionVector::from(vec![1, 2]))
        .unwrap();
        assert_eq!(chunk.len(), 2);
        assert_eq!(chunk.compact().unwrap().num_rows(), 2);
        assert_eq!(chunk.slice(1, 1).compact().unwrap().num_rows(), 1);
        assert!(
            DataChunk::try_new_without_columns(batch.schema(), 1)
                .unwrap_err()
                .to_string()
                .contains("without fields")
        );
    }

    #[test]
    fn try_new_validates_columns() {
        let batch = batch(3);
        let err =
            DataChunk::try_new(batch.schema(), vec![Arc::clone(batch.column(0))], None)
                .unwrap_err();
        assert!(err.to_string().contains("2 fields"), "{err}");

        let err = DataChunk::try_new(
            batch.schema(),
            vec![Arc::clone(batch.column(0)), batch.column(1).slice(0, 2)],
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("different lengths"), "{err}");

        let view: ArrayRef = Arc::new(StringViewArray::from(vec!["x", "y", "z"]));
        let err = DataChunk::try_new(
            batch.schema(),
            vec![Arc::clone(batch.column(0)), view],
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("expects Utf8"), "{err}");

        let chunk = DataChunk::try_new(
            batch.schema(),
            batch.columns().to_vec(),
            Some(SelectionVector::from(vec![2, 0])),
        )
        .unwrap();
        assert_eq!(
            int_values(&chunk.compact().unwrap()),
            vec![Some(2), Some(0)]
        );
    }

    /// A batch of `n` rows of columns of many types, with nulls: flat ones,
    /// which `Gather` may filter, and others, which it takes.
    fn wide_batch(n: usize) -> RecordBatch {
        use arrow::array::{
            BinaryArray, Decimal128Array, DictionaryArray, FixedSizeBinaryArray,
            Float64Array, ListArray, StructArray,
        };
        use arrow::datatypes::Int64Type;
        let rows = 0..n as i64;
        let valid = |i: i64, k: i64| i % k != 3;
        let ints =
            Int32Array::from_iter(rows.clone().map(|i| valid(i, 7).then_some(i as i32)));
        let floats = Float64Array::from_iter(
            rows.clone().map(|i| valid(i, 5).then_some(i as f64 / 3.0)),
        );
        let strings = StringArray::from_iter(
            rows.clone()
                .map(|i| valid(i, 4).then(|| format!("string {i}"))),
        );
        let views = StringViewArray::from_iter(rows.clone().map(|i| {
            valid(i, 6).then(|| {
                if i % 2 == 0 {
                    format!("v{i}")
                } else {
                    format!("a view longer than twelve bytes {i}")
                }
            })
        }));
        let binaries = BinaryArray::from_iter(
            rows.clone()
                .map(|i| valid(i, 9).then(|| i.to_le_bytes().to_vec())),
        );
        let fixed = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            rows.clone().map(|i| valid(i, 8).then(|| i.to_be_bytes())),
            8,
        )
        .unwrap();
        let booleans = BooleanArray::from_iter(
            rows.clone().map(|i| valid(i, 10).then_some(i % 3 == 0)),
        );
        let decimals = Decimal128Array::from_iter(
            rows.clone()
                .map(|i| valid(i, 11).then_some(i128::from(i) * 1001)),
        )
        .with_precision_and_scale(15, 2)
        .unwrap();
        let dictionary = rows
            .clone()
            .map(|i| valid(i, 12).then(|| format!("key {}", i % 5)))
            .collect::<Vec<_>>();
        let dictionary = dictionary
            .iter()
            .map(Option::as_deref)
            .collect::<DictionaryArray<Int32Type>>();
        let lists = ListArray::from_iter_primitive::<Int64Type, _, _>(
            rows.clone()
                .map(|i| valid(i, 13).then(|| (0..i % 4).map(Some).collect::<Vec<_>>())),
        );
        let structs = StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Int32, true)),
            Arc::new(ints.clone()) as ArrayRef,
        )]);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(ints),
            Arc::new(floats),
            Arc::new(strings),
            Arc::new(views),
            Arc::new(binaries),
            Arc::new(fixed),
            Arc::new(booleans),
            Arc::new(decimals),
            Arc::new(dictionary),
            Arc::new(lists),
            Arc::new(structs),
        ];
        RecordBatch::try_from_iter(
            columns
                .into_iter()
                .enumerate()
                .map(|(i, column)| (format!("c{i}"), column)),
        )
        .unwrap()
    }

    /// The rows `indices` of `batch`, gathered with Arrow's `take`.
    fn taken(batch: &RecordBatch, indices: &[u32]) -> RecordBatch {
        let indices = UInt32Array::from(indices.to_vec());
        arrow_select::take::take_record_batch(batch, &indices).unwrap()
    }

    #[test]
    fn compact_gathers_like_take_whatever_the_selection() {
        let n = 2048;
        let batch = wide_batch(n);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut selections = vec![
            (vec![], "slice"),
            ((0..n as u32).collect(), "slice"),
            ((100..1900).collect(), "slice"),
            ((0..n as u32).step_by(3).collect(), "take"),
            ((0..n as u32).rev().collect(), "take"),
            (vec![5, 5, 7, 1], "take"),
            // Dense but not increasing, or with a repeated row.
            (
                (0..n as u32).filter(|i| i % 10 != 0).rev().collect(),
                "take",
            ),
            (
                (0..n as u32).flat_map(|i| [i, i]).take(1500).collect(),
                "take",
            ),
        ];
        for percent in [51, 60, 75, 85, 92, 99] {
            let rows = (0..n as u32)
                .filter(|_| random() % 100 < percent)
                .collect::<Vec<_>>();
            selections.push((rows, "filter"));
        }
        for (rows, strategy) in selections {
            let selection = SelectionVector::from(rows.clone());
            let data_types = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.data_type().clone())
                .collect::<Vec<_>>();
            let gather = Gather::new(&selection, n, data_types.iter());
            assert_eq!(gather.strategy(), strategy, "{} rows", rows.len());
            let chunk = DataChunk::from_record_batch(batch.clone())
                .with_selection(selection)
                .unwrap();
            let compacted = chunk.compact().unwrap();
            let expected = taken(&batch, &rows);
            assert_eq!(compacted, expected, "{strategy} of {} rows", rows.len());
            assert_eq!(compacted.schema(), batch.schema());
            for i in 0..batch.num_columns() {
                assert_eq!(&chunk.compacted_column(i).unwrap(), expected.column(i));
            }
            // The bound expressions of the native operators gather the same
            // rows.
            assert_eq!(
                Gather::new(chunk.selection().unwrap(), n, data_types.iter())
                    .column(batch.column(2))
                    .unwrap()
                    .as_ref(),
                expected.column(2).as_ref()
            );
        }
    }

    #[test]
    fn dense_selections_are_filtered_if_that_pays_off() {
        let n = 1000;
        let strings = DataType::Utf8;
        let ints = DataType::Int64;
        let selection = |percent: u32| {
            SelectionVector::from(
                (0..n as u32)
                    .filter(|i| i % 100 < percent)
                    .collect::<Vec<_>>(),
            )
        };
        for (percent, data_types, strategy) in [
            // Building the predicate costs as much as filtering saves on 4
            // flat columns, or on a column of strings or binaries if more than
            // 80% of the rows are selected.
            (60, vec![&ints; 4], "filter"),
            (60, vec![&ints; 3], "take"),
            (99, vec![&ints], "take"),
            (85, vec![&strings], "filter"),
            (75, vec![&strings], "take"),
            (75, vec![&strings, &ints, &ints, &ints], "filter"),
            // Not more than half of the rows.
            (50, vec![&ints; 10], "take"),
            // Lists are not filtered.
            (
                90,
                vec![&DataType::new_list(DataType::Int64, true); 10],
                "take",
            ),
        ] {
            let gather = Gather::new(&selection(percent), n, data_types.iter().copied());
            assert_eq!(gather.strategy(), strategy, "{percent}% of {data_types:?}");
        }
    }

    #[test]
    fn filtered_strings_keep_the_bytes_beneath_nulls() {
        use std::fmt::Write;
        // Like DataFusion's `FilterExec`, which filters its input: `take` drops
        // the bytes of null strings, `filter` keeps them. Both have the same
        // values.
        let n = 100;
        let values = (0..n).fold(String::new(), |mut values, i| {
            let _ = write!(values, "v{i:03}");
            values
        });
        let strings: ArrayRef = Arc::new(
            StringArray::try_new(
                arrow::buffer::OffsetBuffer::from_lengths(std::iter::repeat_n(4, n)),
                values.into_bytes().into(),
                Some(arrow::buffer::NullBuffer::from_iter(
                    (0..n).map(|i| i % 2 == 0),
                )),
            )
            .unwrap(),
        );
        let selection = SelectionVector::from(
            (0..n as u32).filter(|i| i % 10 != 9).collect::<Vec<_>>(),
        );
        let gather = Gather::new(&selection, n, [strings.data_type()]);
        assert_eq!(gather.strategy(), "filter");
        let filtered = gather.column(&strings).unwrap();
        let taken = take(strings.as_ref(), &selection.to_uint32_array(), None).unwrap();
        assert_eq!(filtered.as_ref(), taken.as_ref());
        let bytes = |array: &ArrayRef| array.as_string::<i32>().values().len();
        assert_eq!(bytes(&filtered), 4 * selection.len());
        // The even rows are valid: all 50 of them are selected.
        assert_eq!(bytes(&taken), 4 * 50);
        let mask = BooleanArray::from_iter((0..n).map(|i| Some(i % 10 != 9)));
        let expected = arrow_select::filter::filter(strings.as_ref(), &mask).unwrap();
        assert_eq!(bytes(&expected), bytes(&filtered));
    }

    #[test]
    fn selection_orders() {
        for (indices, order) in [
            (vec![], Order::Empty),
            (vec![4], Order::Contiguous(4)),
            (vec![4, 5, 6], Order::Contiguous(4)),
            (vec![4, 6, 7], Order::Increasing),
            (vec![4, 4, 5], Order::Unordered),
            (vec![4, 3], Order::Unordered),
            (vec![0, u32::MAX], Order::Increasing),
            (vec![u32::MAX, 0], Order::Unordered),
        ] {
            let selection = SelectionVector::from(indices.clone());
            assert_eq!(selection.order(), order, "{indices:?}");
            assert_eq!(
                selection.is_strictly_increasing(),
                order != Order::Unordered,
                "{indices:?}"
            );
        }
    }

    #[test]
    fn increasing_masks_set_the_bits_of_the_selection() {
        for (len, indices) in [
            (1, vec![0]),
            (64, vec![0, 63]),
            (65, vec![63, 64]),
            (200, vec![1, 2, 3, 70, 128, 199]),
            (200, vec![150]),
        ] {
            let selection = SelectionVector::from(indices.clone());
            let mask = increasing_mask(&selection, len);
            assert_eq!(mask.len(), len);
            assert_eq!(
                mask.set_indices().collect::<Vec<_>>(),
                indices.iter().map(|&i| i as usize).collect::<Vec<_>>()
            );
        }
    }
}
