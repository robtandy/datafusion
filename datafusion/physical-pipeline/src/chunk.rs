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
//! The selection is only materialized (with Arrow's `take` kernel) when a
//! chunk leaves the pipeline engine or an operator needs contiguous arrays,
//! see [`DataChunk::compact`].

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    new_empty_array,
};
use arrow::datatypes::SchemaRef;
use arrow_buffer::ScalarBuffer;
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
    /// This copies data only if the chunk has a selection.
    pub fn compacted_column(&self, i: usize) -> Result<ArrayRef> {
        if self.is_cleared() {
            return Ok(new_empty_array(self.schema.field(i).data_type()));
        }
        let column = &self.columns[i];
        match &self.selection {
            Some(selection) => apply_selection(column, selection),
            None => Ok(Arc::clone(column)),
        }
    }

    /// Materialize the logical rows of this chunk as a [`RecordBatch`].
    ///
    /// If the chunk has a selection, the selected rows are gathered with
    /// Arrow's `take` kernel (or zero-copy sliced if the selection is a
    /// contiguous range). Otherwise the columns are returned as is.
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
            Some(selection) => self
                .columns
                .iter()
                .map(|column| apply_selection(column, selection))
                .collect::<Result<Vec<_>>>()?,
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

fn apply_selection(column: &ArrayRef, selection: &SelectionVector) -> Result<ArrayRef> {
    if let Some((offset, len)) = selection.as_contiguous_range() {
        if offset == 0 && len == column.len() {
            return Ok(Arc::clone(column));
        }
        return Ok(column.slice(offset, len));
    }
    if selection.is_empty() {
        return Ok(column.slice(0, 0));
    }
    Ok(take(column.as_ref(), &selection.to_uint32_array(), None)?)
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
}
