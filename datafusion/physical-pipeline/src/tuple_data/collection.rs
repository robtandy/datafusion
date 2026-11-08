// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`TupleDataCollection`]: block-allocated, row-major storage of tuples,
//! with vectorized scatter ([`TupleDataCollection::append_chunk`]) and
//! gather ([`TupleDataCollection::gather_columns`],
//! [`TupleDataCollection::scan_chunk`]).
//!
//! A collection is a list of *blocks*. A block holds up to
//! [`TupleDataLayout::rows_per_block`] fixed-width rows (see
//! [`TupleDataLayout`]) and the *heap* with the variable-length values of
//! these rows that are not stored inline. Rows are appended to the last
//! block; a row never moves once it is stored, so it is addressed by a
//! [`RowLocation`]: the index of its block and its index in the block.
//!
//! Heap references are offsets into the heap of the block that holds the
//! row, so blocks are self-contained: [`TupleDataCollection::combine`] moves
//! the blocks of another collection without copying or rewriting any row.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayDataBuilder, ArrayRef, AsArray, BooleanArray, GenericByteArray,
    GenericByteViewArray, RecordBatch, RecordBatchOptions, UInt64Array, make_array,
    make_view, new_null_array,
};
use arrow::buffer::{
    BooleanBuffer, Buffer, MutableBuffer, NullBuffer, OffsetBuffer, ScalarBuffer,
};
use arrow::compute::take;
use arrow::datatypes::{
    ArrowNativeType, BinaryType, BinaryViewType, ByteArrayType, ByteViewType, DataType,
    LargeBinaryType, LargeUtf8Type, SchemaRef, StringViewType, Utf8Type,
};
use arrow_row::Rows;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::{
    DataFusionError, Result, exec_datafusion_err, exec_err, internal_datafusion_err,
    internal_err, resources_datafusion_err,
};

use super::layout::{
    ColumnLayout, ColumnStorage, HASH_WIDTH, MAX_INLINE_LEN, MAX_ROWS_PER_BLOCK,
    TupleDataLayout, VAR_LEN_SLOT_WIDTH, new_dictionary,
};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};

/// The location of a row in a [`TupleDataCollection`]: the index of its
/// block and the index of the row in the block.
///
/// A location is packed into the lower [`RowLocation::BITS`] (48) bits of a
/// `u64` ([`RowLocation::as_u64`]), so that it can be stored together with
/// 16 bits of other data (e.g. a hash salt) in a single 64-bit entry of a
/// hash table.
///
/// Locations stay valid until the collection is dropped. When a collection
/// is [combined](TupleDataCollection::combine) into another one, the
/// locations of its rows are shifted by the number of blocks of the other
/// collection (see [`RowLocation::with_block_offset`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowLocation(u64);

impl RowLocation {
    /// The number of bits of the row index.
    pub const ROW_BITS: u32 = 16;
    /// The number of bits of the block index.
    pub const BLOCK_BITS: u32 = 32;
    /// The number of bits of a packed location.
    pub const BITS: u32 = Self::ROW_BITS + Self::BLOCK_BITS;
    /// The maximum number of blocks of a collection.
    pub const MAX_BLOCKS: u64 = 1 << Self::BLOCK_BITS;

    const ROW_MASK: u64 = (1 << Self::ROW_BITS) - 1;
    const MASK: u64 = (1 << Self::BITS) - 1;

    /// The location of row `row` of block `block`.
    ///
    /// `row` must be smaller than [`MAX_ROWS_PER_BLOCK`] and `block` smaller
    /// than [`RowLocation::MAX_BLOCKS`].
    #[inline]
    pub fn new(block: usize, row: usize) -> Self {
        debug_assert!(row < MAX_ROWS_PER_BLOCK, "row index {row} out of range");
        debug_assert!(
            (block as u64) < Self::MAX_BLOCKS,
            "block {block} out of range"
        );
        Self(((block as u64) << Self::ROW_BITS) | (row as u64 & Self::ROW_MASK))
    }

    /// The index of the block of the row.
    #[inline]
    pub fn block(self) -> usize {
        (self.0 >> Self::ROW_BITS) as usize
    }

    /// The index of the row in its block.
    #[inline]
    pub fn row(self) -> usize {
        (self.0 & Self::ROW_MASK) as usize
    }

    /// The packed location. Only the lower [`RowLocation::BITS`] bits are
    /// used.
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Unpack a location packed with [`RowLocation::as_u64`]. Bits above
    /// [`RowLocation::BITS`] are ignored.
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        Self(value & Self::MASK)
    }

    /// The location of the same row after the blocks of its collection were
    /// shifted by `offset` blocks, see [`TupleDataCollection::combine`].
    #[inline]
    pub fn with_block_offset(self, offset: usize) -> Self {
        Self::new(self.block() + offset, self.row())
    }
}

/// A block of rows and the heap with their variable-length values.
#[derive(Default)]
struct TupleDataBlock {
    /// `count * row_width` bytes.
    rows: Vec<u8>,
    /// Variable-length values that are not stored inline in `rows`.
    heap: Vec<u8>,
    /// The number of rows in the block.
    count: usize,
}

impl TupleDataBlock {
    /// Add `additional` zeroed rows.
    ///
    /// The row buffer of the first block of a collection grows geometrically
    /// up to the capacity of a full block, so that small collections (e.g.
    /// the partitions of a thread-local [`PartitionedTupleData`]) stay small.
    /// The other blocks are allocated with their full capacity, so that the
    /// rows of large collections are not copied when a block grows.
    ///
    /// [`PartitionedTupleData`]: super::PartitionedTupleData
    fn grow_rows(
        &mut self,
        additional: usize,
        row_width: usize,
        rows_per_block: usize,
        first_block: bool,
    ) -> Result<()> {
        let len = (self.count + additional) * row_width;
        if len > self.rows.capacity() {
            let full = rows_per_block * row_width;
            let target = if first_block {
                self.rows.capacity().saturating_mul(2).min(full).max(len)
            } else {
                full.max(len)
            };
            self.rows
                .try_reserve_exact(target - self.rows.len())
                .map_err(|e| {
                    resources_datafusion_err!(
                        "Failed to allocate {target} bytes for a TupleDataCollection block: {e}"
                    )
                })?;
        }
        self.rows.resize(len, 0);
        Ok(())
    }
}

/// A reference to a stored row and to the heap of its block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowRef<'a> {
    pub(crate) row: &'a [u8],
    pub(crate) heap: &'a [u8],
}

impl<'a> RowRef<'a> {
    /// Returns `true` if the value of column `column` is valid.
    #[inline]
    pub(crate) fn is_valid(&self, column: usize) -> bool {
        self.row[column >> 3] & (1 << (column & 7)) != 0
    }

    /// The length of the variable-length value whose slot is at `offset`.
    #[inline]
    pub(crate) fn var_len_size(&self, offset: usize) -> usize {
        read_u32(self.row, offset) as usize
    }

    /// The bytes of the variable-length value whose slot is at `offset`.
    #[inline]
    pub(crate) fn var_len(&self, offset: usize) -> &'a [u8] {
        let len = self.var_len_size(offset);
        if len <= MAX_INLINE_LEN {
            &self.row[offset + 4..offset + 4 + len]
        } else {
            let start = read_u64(self.row, offset + 8) as usize;
            &self.heap[start..start + len]
        }
    }

    /// The hash stored at `offset`.
    #[inline]
    pub(crate) fn hash(&self, offset: usize) -> u64 {
        read_u64(self.row, offset)
    }
}

#[inline]
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0; 4];
    value.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(value)
}

#[inline]
fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut value = [0; 8];
    value.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(value)
}

/// Block-allocated, row-major storage of tuples with a [`TupleDataLayout`].
///
/// * [`Self::append_chunk`] scatters the (selected) rows of a
///   [`DataChunk`] into newly allocated rows and returns their
///   [`RowLocation`]s.
/// * [`Self::gather_columns`] / [`Self::gather`] reconstruct Arrow arrays
///   from a list of row locations, with exactly the data types, and
///   [`Self::scan_chunk`] reads the whole collection in chunks.
/// * [`Self::combine`] moves all blocks of another collection into this one
///   without copying rows, which is how thread-local collections are merged
///   in `Sink::combine`.
///
/// See the [module documentation](self) for the storage format.
pub struct TupleDataCollection {
    layout: Arc<TupleDataLayout>,
    blocks: Vec<TupleDataBlock>,
    /// `block_starts[b]` is the number of rows in the blocks before block
    /// `b`. Only the last block grows, so the entries never change.
    block_starts: Vec<usize>,
    count: usize,
}

impl fmt::Debug for TupleDataCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TupleDataCollection")
            .field("schema", self.layout.schema())
            .field("rows", &self.count)
            .field("blocks", &self.blocks.len())
            .field("memory_size", &self.memory_size())
            .finish()
    }
}

impl TupleDataCollection {
    /// Create an empty collection of rows with the given layout.
    pub fn new(layout: Arc<TupleDataLayout>) -> Self {
        Self {
            layout,
            blocks: vec![],
            block_starts: vec![],
            count: 0,
        }
    }

    /// The layout of the rows.
    pub fn layout(&self) -> &Arc<TupleDataLayout> {
        &self.layout
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the collection has no rows.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The number of rows in block `block`.
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.num_blocks()`.
    pub fn block_len(&self, block: usize) -> usize {
        self.blocks[block].count
    }

    /// The number of bytes allocated by the collection (row buffers and
    /// heaps, including unused capacity), for memory accounting.
    pub fn memory_size(&self) -> usize {
        size_of::<Self>()
            + self.blocks.capacity() * size_of::<TupleDataBlock>()
            + self.block_starts.capacity() * size_of::<usize>()
            + self
                .blocks
                .iter()
                .map(|b| b.rows.capacity() + b.heap.capacity())
                .sum::<usize>()
    }

    /// The locations of all rows, in insertion order (for collections that
    /// were combined: the rows of this collection, then the rows of the
    /// other collection).
    pub fn row_locations(&self) -> impl Iterator<Item = RowLocation> + '_ {
        self.blocks.iter().enumerate().flat_map(|(block, b)| {
            (0..b.count).map(move |row| RowLocation::new(block, row))
        })
    }

    /// The index of the row at `location` in [`Self::row_locations`], a
    /// dense index in `0..self.len()` (e.g. for bitmaps of matched rows).
    ///
    /// # Panics
    ///
    /// Panics if the block of `location` does not exist.
    #[inline]
    pub fn row_index(&self, location: RowLocation) -> usize {
        self.block_starts[location.block()] + location.row()
    }

    /// Scatter the rows of `chunk` into newly allocated rows, and return
    /// the locations of the new rows, one for every row of the chunk.
    ///
    /// If the chunk has a selection, only the selected rows are stored, in
    /// the order of the selection; the chunk is not compacted. `hashes`
    /// must be given if and only if the layout has a hash slot, and then
    /// holds one hash for every (logical) row of the chunk.
    ///
    /// The columns of the chunk must have the data types of the layout.
    pub fn append_chunk(
        &mut self,
        chunk: &DataChunk,
        hashes: Option<&[u64]>,
    ) -> Result<Vec<RowLocation>> {
        check_chunk(&self.layout, chunk)?;
        check_hashes(&self.layout, chunk, hashes)?;
        let mut locations = Vec::new();
        if chunk.is_empty() {
            return Ok(locations);
        }
        let source = ScatterSource::try_new(&self.layout, chunk)?;
        let rows = physical_rows(chunk)?;
        self.append_rows(&source, &rows, hashes, Some(&mut locations))?;
        Ok(locations)
    }

    /// Append the physical rows `rows` of `source` (with the hashes
    /// `hashes`, one for every entry of `rows`), and push their locations
    /// to `locations`.
    pub(crate) fn append_rows(
        &mut self,
        source: &ScatterSource,
        rows: &[u32],
        hashes: Option<&[u64]>,
        mut locations: Option<&mut Vec<RowLocation>>,
    ) -> Result<()> {
        if let Some(locations) = locations.as_deref_mut() {
            locations.try_reserve(rows.len()).map_err(|e| {
                resources_datafusion_err!("Failed to allocate row locations: {e}")
            })?;
        }
        let layout = Arc::clone(&self.layout);
        let row_width = layout.row_width();
        let rows_per_block = layout.rows_per_block();
        let mut appended = 0;
        while appended < rows.len() {
            if self
                .blocks
                .last()
                .is_none_or(|block| block.count >= rows_per_block)
            {
                self.push_block()?;
            }
            let block_index = self.blocks.len() - 1;
            let block = &mut self.blocks[block_index];
            let n = (rows.len() - appended).min(rows_per_block - block.count);
            let segment = appended..appended + n;
            let start = block.count;
            block.grow_rows(n, row_width, rows_per_block, block_index == 0)?;
            let target = &mut block.rows[start * row_width..];
            let result = scatter_rows(
                &layout,
                source,
                &rows[segment.clone()],
                hashes.map(|hashes| &hashes[segment]),
                target,
                &mut block.heap,
            );
            if let Err(e) = result {
                block.rows.truncate(start * row_width);
                return Err(e);
            }
            block.count += n;
            self.count += n;
            if let Some(locations) = locations.as_deref_mut() {
                locations.extend(
                    (start..start + n).map(|row| RowLocation::new(block_index, row)),
                );
            }
            appended += n;
        }
        Ok(())
    }

    fn push_block(&mut self) -> Result<()> {
        if self.blocks.len() as u64 >= RowLocation::MAX_BLOCKS {
            return exec_err!(
                "TupleDataCollection exceeds the maximum of {} blocks",
                RowLocation::MAX_BLOCKS
            );
        }
        self.blocks
            .try_reserve(1)
            .and_then(|()| self.block_starts.try_reserve(1))
            .map_err(|e| {
                resources_datafusion_err!(
                    "Failed to allocate a TupleDataCollection block: {e}"
                )
            })?;
        self.block_starts.push(self.count);
        self.blocks.push(TupleDataBlock::default());
        Ok(())
    }

    /// Move all rows of `other` into this collection, without copying them:
    /// the blocks of `other` are appended to the blocks of this collection.
    ///
    /// The rows of `other` keep their row index, and their block index is
    /// shifted by the number of blocks this collection had before the call
    /// (see [`RowLocation::with_block_offset`]). The layouts must be
    /// [compatible](TupleDataLayout::is_compatible).
    pub fn combine(&mut self, mut other: TupleDataCollection) -> Result<()> {
        if !self.layout.is_compatible(&other.layout) {
            return internal_err!(
                "Cannot combine TupleDataCollections with different layouts: {:?} and {:?}",
                self.layout.schema(),
                other.layout.schema()
            );
        }
        if other.blocks.is_empty() {
            return Ok(());
        }
        if (self.blocks.len() + other.blocks.len()) as u64 > RowLocation::MAX_BLOCKS {
            return exec_err!(
                "TupleDataCollection exceeds the maximum of {} blocks",
                RowLocation::MAX_BLOCKS
            );
        }
        self.blocks
            .try_reserve(other.blocks.len())
            .and_then(|()| self.block_starts.try_reserve(other.blocks.len()))
            .map_err(|e| {
                resources_datafusion_err!("Failed to combine TupleDataCollections: {e}")
            })?;
        let offset = self.count;
        self.block_starts
            .extend(other.block_starts.iter().map(|start| start + offset));
        self.blocks.append(&mut other.blocks);
        self.count += other.count;
        Ok(())
    }

    /// Resolve `location` (of a row with `row_width` bytes) to its row, `None`
    /// if the location is invalid.
    #[inline(always)]
    fn try_row_ref(&self, location: RowLocation, row_width: usize) -> Option<RowRef<'_>> {
        let block = self.blocks.get(location.block())?;
        if location.row() >= block.count {
            return None;
        }
        let start = location.row() * row_width;
        Some(RowRef {
            row: block.rows.get(start..start + row_width)?,
            heap: &block.heap,
        })
    }

    #[cold]
    #[inline(never)]
    fn invalid_location(&self, location: RowLocation) -> DataFusionError {
        internal_datafusion_err!(
            "Invalid row location (block {}, row {}) for a TupleDataCollection with {} blocks",
            location.block(),
            location.row(),
            self.blocks.len()
        )
    }

    /// Resolve `locations` to their rows.
    pub(crate) fn row_refs(&self, locations: &[RowLocation]) -> Result<Vec<RowRef<'_>>> {
        self.row_refs_of(locations.iter().copied(), locations.len())
    }

    /// Resolve the `len` locations of `locations` to their rows.
    #[inline]
    pub(crate) fn row_refs_of(
        &self,
        locations: impl Iterator<Item = RowLocation>,
        len: usize,
    ) -> Result<Vec<RowRef<'_>>> {
        let row_width = self.layout.row_width();
        let mut rows = Vec::with_capacity(len);
        for location in locations {
            match self.try_row_ref(location, row_width) {
                Some(row) => rows.push(row),
                None => return Err(self.invalid_location(location)),
            }
        }
        Ok(rows)
    }

    /// Gather the values of the columns `column_ids` of the rows at
    /// `locations` into Arrow arrays (one array per column id, with one
    /// value per location, in order).
    ///
    /// The arrays have exactly the data types of the columns of the layout,
    /// and the values that were appended (bit for bit, including `-0.0` and
    /// `NaN` payloads). Dictionary columns are dictionary encoded again: the
    /// dictionary of the result holds the distinct values of the gathered
    /// rows, in order of first occurrence.
    pub fn gather_columns(
        &self,
        locations: &[RowLocation],
        column_ids: &[usize],
    ) -> Result<Vec<ArrayRef>> {
        let num_columns = self.layout.num_columns();
        if let Some(&id) = column_ids.iter().find(|&&id| id >= num_columns) {
            return internal_err!(
                "Column {id} out of bounds for a TupleDataLayout with {num_columns} columns"
            );
        }
        let rows = self.row_refs(locations)?;
        column_ids
            .iter()
            .map(|&id| gather_column(self.layout.column(id), id, &rows))
            .collect()
    }

    /// Gather all columns of the rows at `locations` into a [`RecordBatch`]
    /// with the schema of the layout.
    pub fn gather(&self, locations: &[RowLocation]) -> Result<RecordBatch> {
        let column_ids = (0..self.layout.num_columns()).collect::<Vec<_>>();
        let columns = self.gather_columns(locations, &column_ids)?;
        record_batch(Arc::clone(self.layout.schema()), columns, locations.len())
    }

    /// Gather the columns `column_ids` of the rows at `locations` into a
    /// [`RecordBatch`] with the schema of the layout projected to
    /// `column_ids`.
    pub fn gather_projected(
        &self,
        locations: &[RowLocation],
        column_ids: &[usize],
    ) -> Result<RecordBatch> {
        let columns = self.gather_columns(locations, column_ids)?;
        let schema = Arc::new(self.layout.schema().project(column_ids)?);
        record_batch(schema, columns, locations.len())
    }

    /// The hashes of the rows at `locations`. The layout must have a hash
    /// slot.
    pub fn gather_hashes(&self, locations: &[RowLocation]) -> Result<Vec<u64>> {
        let Some(offset) = self.layout.hash_offset() else {
            return internal_err!("The TupleDataLayout has no hash slot");
        };
        let row_width = self.layout.row_width();
        let mut hashes = Vec::with_capacity(locations.len());
        for &location in locations {
            match self.try_row_ref(location, row_width) {
                Some(row) => hashes.push(row.hash(offset)),
                None => return Err(self.invalid_location(location)),
            }
        }
        Ok(hashes)
    }

    /// Return the next chunk of at most [`TupleDataScanState::chunk_size`]
    /// rows of the scan described by `state`, or `None` once all rows were
    /// returned. The locations of the rows of the returned chunk are
    /// available in [`TupleDataScanState::locations`].
    ///
    /// Rows are returned in the order of [`Self::row_locations`].
    pub fn scan_chunk(
        &self,
        state: &mut TupleDataScanState,
    ) -> Result<Option<DataChunk>> {
        state.locations.clear();
        let end = state.blocks.end.min(self.blocks.len());
        while state.locations.len() < state.chunk_size && state.block < end {
            let block = &self.blocks[state.block];
            let n =
                (block.count - state.row).min(state.chunk_size - state.locations.len());
            let block_index = state.block;
            state.locations.extend(
                (state.row..state.row + n).map(|row| RowLocation::new(block_index, row)),
            );
            state.row += n;
            if state.row >= block.count {
                state.block += 1;
                state.row = 0;
            }
        }
        if state.locations.is_empty() {
            return Ok(None);
        }
        let schema = match &state.schema {
            Some(schema) => Arc::clone(schema),
            None => {
                let schema = match &state.column_ids {
                    Some(column_ids) => {
                        Arc::new(self.layout.schema().project(column_ids)?)
                    }
                    None => Arc::clone(self.layout.schema()),
                };
                state.schema = Some(Arc::clone(&schema));
                schema
            }
        };
        let columns = match &state.column_ids {
            Some(column_ids) => self.gather_columns(&state.locations, column_ids)?,
            None => {
                let column_ids = (0..self.layout.num_columns()).collect::<Vec<_>>();
                self.gather_columns(&state.locations, &column_ids)?
            }
        };
        let batch = record_batch(schema, columns, state.locations.len())?;
        Ok(Some(DataChunk::from_record_batch(batch)))
    }
}

fn record_batch(
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    num_rows: usize,
) -> Result<RecordBatch> {
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    Ok(RecordBatch::try_new_with_options(
        schema, columns, &options,
    )?)
}

/// The state of a sequential scan of a [`TupleDataCollection`] with
/// [`TupleDataCollection::scan_chunk`].
///
/// By default a scan reads all columns of all blocks in chunks of
/// [`DEFAULT_VECTOR_SIZE`] rows. A scan can be restricted to a range of
/// blocks, e.g. to split the scan of a collection between threads.
#[derive(Debug, Clone)]
pub struct TupleDataScanState {
    column_ids: Option<Vec<usize>>,
    blocks: Range<usize>,
    block: usize,
    row: usize,
    chunk_size: usize,
    schema: Option<SchemaRef>,
    locations: Vec<RowLocation>,
}

impl Default for TupleDataScanState {
    fn default() -> Self {
        Self::new()
    }
}

impl TupleDataScanState {
    /// A scan of all columns of all rows.
    pub fn new() -> Self {
        Self {
            column_ids: None,
            blocks: 0..usize::MAX,
            block: 0,
            row: 0,
            chunk_size: DEFAULT_VECTOR_SIZE,
            schema: None,
            locations: vec![],
        }
    }

    /// Only read the columns `column_ids`, in this order.
    pub fn with_column_ids(mut self, column_ids: Vec<usize>) -> Self {
        self.column_ids = Some(column_ids);
        self.schema = None;
        self
    }

    /// Only read the rows of the blocks in `blocks`.
    pub fn with_block_range(mut self, blocks: Range<usize>) -> Self {
        self.block = blocks.start;
        self.row = 0;
        self.blocks = blocks;
        self
    }

    /// Return chunks of at most `chunk_size` rows (at least one).
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// The maximum number of rows of a chunk.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// The locations of the rows of the chunk last returned by
    /// [`TupleDataCollection::scan_chunk`].
    pub fn locations(&self) -> &[RowLocation] {
        &self.locations
    }
}

/// Check that `hashes` match the layout and the chunk.
pub(crate) fn check_hashes(
    layout: &TupleDataLayout,
    chunk: &DataChunk,
    hashes: Option<&[u64]>,
) -> Result<()> {
    match (layout.has_hash(), hashes) {
        (true, None) => {
            internal_err!("The TupleDataLayout has a hash slot, but no hashes were given")
        }
        (false, Some(_)) => {
            internal_err!("Hashes were given, but the TupleDataLayout has no hash slot")
        }
        (_, Some(hashes)) if hashes.len() != chunk.len() => internal_err!(
            "Got {} hashes for a DataChunk with {} rows",
            hashes.len(),
            chunk.len()
        ),
        _ => Ok(()),
    }
}

/// The physical indices of the logical rows of `chunk`.
pub(crate) fn physical_rows(chunk: &DataChunk) -> Result<Cow<'_, [u32]>> {
    match chunk.selection() {
        Some(selection) => Ok(Cow::Borrowed(selection.indices())),
        None => {
            let len = u32::try_from(chunk.len()).map_err(|_| {
                exec_datafusion_err!("DataChunk with {} rows is too large", chunk.len())
            })?;
            Ok(Cow::Owned((0..len).collect()))
        }
    }
}

/// The columns of a [`DataChunk`], prepared for scattering any of its rows
/// into collections with a given layout (e.g. into all partitions of a
/// [`PartitionedTupleData`](super::PartitionedTupleData)): nested values are
/// encoded with the row format once per chunk (the values of dictionaries
/// once per value, see [`DictionaryValues`]).
///
/// All columns are indexed by *physical* row.
pub(crate) struct ScatterSource {
    columns: Vec<SourceColumn>,
}

struct SourceColumn {
    /// The values of the column (the values of the dictionary, for
    /// dictionaries).
    values: ArrayRef,
    /// For dictionaries, the index in `values` of the value of every row,
    /// see [`DictionaryValues`].
    indices: Option<Vec<usize>>,
    /// The logical nulls of the column, `None` if all values are valid.
    nulls: Option<NullBuffer>,
    /// The null keys of a dictionary column, `None` if all keys are valid.
    key_nulls: Option<NullBuffer>,
    data: SourceData,
}

enum SourceData {
    /// Nothing to store besides the validity.
    Null,
    Boolean(BooleanBuffer),
    /// The values bytes (`width` bytes per row, starting at row 0).
    Fixed(Buffer),
    /// A string or binary array, see [`SourceColumn::values`].
    VarLen,
    /// The values encoded with the row format.
    Encoded(Rows),
}

impl ScatterSource {
    /// Prepare the columns of `chunk`, which must not be empty, for
    /// scattering into collections with layout `layout`.
    pub(crate) fn try_new(layout: &TupleDataLayout, chunk: &DataChunk) -> Result<Self> {
        check_chunk(layout, chunk)?;
        let columns = layout
            .columns()
            .iter()
            .zip(chunk.columns())
            .map(|(column, array)| SourceColumn::try_new(column, array))
            .collect::<Result<_>>()?;
        Ok(Self { columns })
    }
}

impl SourceColumn {
    fn try_new(column: &ColumnLayout, array: &ArrayRef) -> Result<Self> {
        let (values, indices, key_nulls) = if column.is_dictionary() {
            // The nulls of a dictionary array are the nulls of its keys.
            let key_nulls = array.nulls().filter(|n| n.null_count() > 0).cloned();
            // Encode only the values that the rows refer to.
            let used_only = matches!(column.storage(), ColumnStorage::Encoded);
            let DictionaryValues { values, indices } =
                DictionaryValues::try_new(array, used_only)?;
            (values, Some(indices), key_nulls)
        } else {
            (Arc::clone(array), None, None)
        };
        // The logical nulls of a dictionary are the null keys and the keys of
        // null values (the value of a null key is any value).
        let nulls = array.logical_nulls().filter(|n| n.null_count() > 0);
        let data = match column.storage() {
            ColumnStorage::Null => SourceData::Null,
            ColumnStorage::Boolean => {
                SourceData::Boolean(values.as_boolean().values().clone())
            }
            ColumnStorage::Fixed(width) => {
                SourceData::Fixed(fixed_values(&values, width))
            }
            ColumnStorage::VarLen => SourceData::VarLen,
            ColumnStorage::Encoded => {
                let codec = column.codec().ok_or_else(|| {
                    internal_datafusion_err!("Encoded column without a row codec")
                })?;
                SourceData::Encoded(
                    codec.converter.convert_columns(&[Arc::clone(&values)])?,
                )
            }
        };
        Ok(Self {
            values,
            indices,
            nulls,
            key_nulls,
            data,
        })
    }
}

/// Check that the columns of `chunk` have the data types of `layout`.
pub(crate) fn check_chunk(layout: &TupleDataLayout, chunk: &DataChunk) -> Result<()> {
    if chunk.num_columns() != layout.num_columns() {
        return internal_err!(
            "DataChunk has {} columns, but the TupleDataLayout has {}",
            chunk.num_columns(),
            layout.num_columns()
        );
    }
    if chunk.columns().len() != layout.num_columns() && !chunk.is_empty() {
        return internal_err!("A DataChunk with rows must have columns");
    }
    for (i, (field, column)) in chunk
        .schema()
        .fields()
        .iter()
        .zip(layout.columns())
        .enumerate()
    {
        if field.data_type() != column.data_type() {
            return internal_err!(
                "DataChunk column {i} has type {}, but the TupleDataLayout expects {}",
                field.data_type(),
                column.data_type()
            );
        }
    }
    Ok(())
}

/// The values of the rows of a dictionary array, without expanding them to
/// one value per row, which could make them too large for their type (e.g.
/// the run-end encoded children of nested values, or the offsets of
/// strings): the value of every row `i` with a valid key is
/// `values[indices[i]]`, and the rows with a null key (which are null in
/// [`Array::logical_nulls`] of the dictionary) have the index of any value.
pub(crate) struct DictionaryValues {
    pub(crate) values: ArrayRef,
    pub(crate) indices: Vec<usize>,
}

impl DictionaryValues {
    /// The values of the rows of the dictionary array `array`. With
    /// `used_only`, if the dictionary has more values than rows, `values`
    /// holds only the values that rows with a valid key refer to (e.g. to
    /// encode them).
    pub(crate) fn try_new(array: &ArrayRef, used_only: bool) -> Result<Self> {
        let Some(dictionary) = array.as_any_dictionary_opt() else {
            return internal_err!(
                "Expected a dictionary array, got {}",
                array.data_type()
            );
        };
        let values = dictionary.values();
        if values.is_empty() {
            // All keys are null: they refer to a null value.
            return Ok(Self {
                values: new_null_array(values.data_type(), 1),
                indices: vec![0; array.len()],
            });
        }
        // In bounds also for the null keys, whose key values can be any.
        let mut indices = dictionary.normalized_keys();
        if !used_only || values.len() <= indices.len() {
            return Ok(Self {
                values: Arc::clone(values),
                indices,
            });
        }
        let keys = dictionary.keys();
        let mut used = Vec::new();
        let mut used_indices =
            HashMap::with_capacity_and_hasher(indices.len(), RandomState::default());
        for (row, index) in indices.iter_mut().enumerate() {
            let value = *index;
            *index = if keys.is_null(row) {
                0
            } else {
                *used_indices.entry(value).or_insert_with(|| {
                    used.push(value as u64);
                    used.len() - 1
                })
            };
        }
        if used.is_empty() {
            used.push(0);
        }
        Ok(Self {
            values: take(values.as_ref(), &UInt64Array::from(used), None)?,
            indices,
        })
    }
}

/// The bytes of the values of a fixed-width array, starting at its first
/// row.
pub(crate) fn fixed_values(array: &ArrayRef, width: usize) -> Buffer {
    let data = array.to_data();
    match data.buffers().first() {
        Some(buffer) => {
            buffer.slice_with_length(data.offset() * width, data.len() * width)
        }
        None => Buffer::from_vec(Vec::<u8>::new()),
    }
}

/// Write the rows `rows` of `source` into `target`, the bytes of
/// `rows.len()` new, zeroed rows.
fn scatter_rows(
    layout: &TupleDataLayout,
    source: &ScatterSource,
    rows: &[u32],
    hashes: Option<&[u64]>,
    target: &mut [u8],
    heap: &mut Vec<u8>,
) -> Result<()> {
    let row_width = layout.row_width();
    if row_width == 0 {
        return Ok(());
    }
    // Initialize the validity bytes: all values are valid.
    match *layout.all_valid() {
        [] => {}
        // Avoid a `memcpy` call per row for the common case of up to 8
        // columns.
        [all_valid] => target
            .chunks_exact_mut(row_width)
            .for_each(|row| row[0] = all_valid),
        ref all_valid => target
            .chunks_exact_mut(row_width)
            .for_each(|row| row[..all_valid.len()].copy_from_slice(all_valid)),
    }
    for (index, (column, source)) in
        layout.columns().iter().zip(&source.columns).enumerate()
    {
        let mut target = ScatterTarget {
            rows: &mut *target,
            row_width,
            offset: column.offset(),
        };
        if let Some(nulls) = &source.nulls {
            target.clear_validity(index, nulls, rows);
        }
        if let (Some(bit), Some(key_nulls)) =
            (column.key_validity_bit(), &source.key_nulls)
        {
            target.clear_validity(bit, key_nulls, rows);
        }
        match source.indices.as_deref() {
            None => scatter_values(&mut target, column, source, rows, heap, |i| i)?,
            Some(indices) => {
                scatter_values(&mut target, column, source, rows, heap, |i| indices[i])?
            }
        }
    }
    if let (Some(offset), Some(hashes)) = (layout.hash_offset(), hashes) {
        for (row, hash) in target.chunks_exact_mut(row_width).zip(hashes) {
            row[offset..offset + HASH_WIDTH].copy_from_slice(&hash.to_le_bytes());
        }
    }
    Ok(())
}

/// Write the values of the (physical) rows `rows` of `source` (with layout
/// `column`) into `target`: the value of row `i` is value `index(i)` of
/// `source`.
fn scatter_values(
    target: &mut ScatterTarget<'_>,
    column: &ColumnLayout,
    source: &SourceColumn,
    rows: &[u32],
    heap: &mut Vec<u8>,
    index: impl Fn(usize) -> usize + Copy,
) -> Result<()> {
    let nulls = source.nulls.as_ref();
    match &source.data {
        SourceData::Null => {}
        SourceData::Boolean(values) => target.scatter_bool(values, nulls, rows, index),
        SourceData::Fixed(values) => {
            target.scatter_fixed(values, column.width(), nulls, rows, index)
        }
        SourceData::VarLen => {
            scatter_var_len(target, &source.values, nulls, rows, index, heap)?
        }
        SourceData::Encoded(encoded) => {
            // The encoding of null values is stored too, it is part of the
            // value (e.g. the variant of a null union value).
            target.scatter_bytes(|i| encoded.row(index(i)).data(), None, rows, heap)?
        }
    }
    Ok(())
}

fn scatter_var_len(
    target: &mut ScatterTarget<'_>,
    values: &ArrayRef,
    nulls: Option<&NullBuffer>,
    rows: &[u32],
    index: impl Fn(usize) -> usize + Copy,
    heap: &mut Vec<u8>,
) -> Result<()> {
    match values.data_type() {
        DataType::Utf8 => {
            let values = values.as_string::<i32>();
            let value = |i| values.value(index(i)).as_bytes();
            target.scatter_bytes(value, nulls, rows, heap)
        }
        DataType::LargeUtf8 => {
            let values = values.as_string::<i64>();
            let value = |i| values.value(index(i)).as_bytes();
            target.scatter_bytes(value, nulls, rows, heap)
        }
        DataType::Utf8View => {
            let values = values.as_string_view();
            let value = |i| values.value(index(i)).as_bytes();
            target.scatter_bytes(value, nulls, rows, heap)
        }
        DataType::Binary => {
            let values = values.as_binary::<i32>();
            target.scatter_bytes(|i| values.value(index(i)), nulls, rows, heap)
        }
        DataType::LargeBinary => {
            let values = values.as_binary::<i64>();
            target.scatter_bytes(|i| values.value(index(i)), nulls, rows, heap)
        }
        DataType::BinaryView => {
            let values = values.as_binary_view();
            target.scatter_bytes(|i| values.value(index(i)), nulls, rows, heap)
        }
        data_type => internal_err!("Unexpected variable-length data type {data_type}"),
    }
}

/// The slots of one column in a range of new rows.
struct ScatterTarget<'a> {
    rows: &'a mut [u8],
    row_width: usize,
    /// The offset of the slot of the column in a row.
    offset: usize,
}

impl ScatterTarget<'_> {
    fn clear_validity(&mut self, column: usize, nulls: &NullBuffer, rows: &[u32]) {
        let (byte, mask) = (column >> 3, !(1_u8 << (column & 7)));
        for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
            if nulls.is_null(i as usize) {
                row[byte] &= mask;
            }
        }
    }

    /// Store the booleans of the rows `rows`: the value of row `i` is value
    /// `index(i)` of `values`.
    fn scatter_bool(
        &mut self,
        values: &BooleanBuffer,
        nulls: Option<&NullBuffer>,
        rows: &[u32],
        index: impl Fn(usize) -> usize,
    ) {
        let offset = self.offset;
        for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
            let i = i as usize;
            let valid = nulls.is_none_or(|n| n.is_valid(i));
            row[offset] = u8::from(valid && values.value(index(i)));
        }
    }

    /// Store the fixed-width values of the rows `rows`: the value of row `i`
    /// is value `index(i)` of `values`.
    fn scatter_fixed(
        &mut self,
        values: &[u8],
        width: usize,
        nulls: Option<&NullBuffer>,
        rows: &[u32],
        index: impl Fn(usize) -> usize + Copy,
    ) {
        match width {
            0 => {}
            1 => self.scatter_fixed_width::<1>(values, nulls, rows, index),
            2 => self.scatter_fixed_width::<2>(values, nulls, rows, index),
            4 => self.scatter_fixed_width::<4>(values, nulls, rows, index),
            8 => self.scatter_fixed_width::<8>(values, nulls, rows, index),
            16 => self.scatter_fixed_width::<16>(values, nulls, rows, index),
            32 => self.scatter_fixed_width::<32>(values, nulls, rows, index),
            width => {
                let offset = self.offset;
                for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
                    let i = i as usize;
                    if nulls.is_none_or(|n| n.is_valid(i)) {
                        let v = index(i);
                        row[offset..offset + width]
                            .copy_from_slice(&values[v * width..(v + 1) * width]);
                    }
                }
            }
        }
    }

    #[inline]
    fn scatter_fixed_width<const W: usize>(
        &mut self,
        values: &[u8],
        nulls: Option<&NullBuffer>,
        rows: &[u32],
        index: impl Fn(usize) -> usize,
    ) {
        let offset = self.offset;
        let rows_and_indices = self.rows.chunks_exact_mut(self.row_width).zip(rows);
        match nulls {
            None => {
                for (row, &i) in rows_and_indices {
                    let v = index(i as usize);
                    row[offset..offset + W].copy_from_slice(&values[v * W..(v + 1) * W]);
                }
            }
            Some(nulls) => {
                for (row, &i) in rows_and_indices {
                    let i = i as usize;
                    if nulls.is_valid(i) {
                        let v = index(i);
                        row[offset..offset + W]
                            .copy_from_slice(&values[v * W..(v + 1) * W]);
                    }
                }
            }
        }
    }

    /// Store the variable-length values `value(i)` of the rows `rows`,
    /// except for null rows (if `nulls` is given), which keep length `0`.
    fn scatter_bytes<'v>(
        &mut self,
        value: impl Fn(usize) -> &'v [u8],
        nulls: Option<&NullBuffer>,
        rows: &[u32],
        heap: &mut Vec<u8>,
    ) -> Result<()> {
        let is_valid = |i: usize| nulls.is_none_or(|n| n.is_valid(i));
        let heap_bytes = rows
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| is_valid(i))
            .map(|i| value(i).len())
            .filter(|&len| len > MAX_INLINE_LEN)
            .sum::<usize>();
        heap.try_reserve(heap_bytes).map_err(|e| {
            resources_datafusion_err!(
                "Failed to allocate {heap_bytes} bytes for a TupleDataCollection heap: {e}"
            )
        })?;
        let offset = self.offset;
        for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
            let i = i as usize;
            if is_valid(i) {
                write_var_len(
                    &mut row[offset..offset + VAR_LEN_SLOT_WIDTH],
                    value(i),
                    heap,
                )?;
            }
        }
        Ok(())
    }
}

/// Write `value` into the (zeroed) variable-length `slot`, see
/// [`TupleDataLayout`].
#[inline]
fn write_var_len(slot: &mut [u8], value: &[u8], heap: &mut Vec<u8>) -> Result<()> {
    let len = u32::try_from(value.len()).map_err(|_| {
        exec_datafusion_err!(
            "A value of {} bytes is too large for a TupleDataCollection",
            value.len()
        )
    })?;
    slot[..4].copy_from_slice(&len.to_le_bytes());
    if value.len() <= MAX_INLINE_LEN {
        slot[4..4 + value.len()].copy_from_slice(value);
    } else {
        slot[4..8].copy_from_slice(&value[..4]);
        slot[8..16].copy_from_slice(&(heap.len() as u64).to_le_bytes());
        heap.extend_from_slice(value);
    }
    Ok(())
}

/// Gather the values of column `index` (with layout `column`) of `rows`.
pub(crate) fn gather_column(
    column: &ColumnLayout,
    index: usize,
    rows: &[RowRef<'_>],
) -> Result<ArrayRef> {
    match column.data_type() {
        DataType::Dictionary(key_type, _) => {
            gather_dictionary(column, index, key_type, rows)
        }
        _ => gather_values(column, index, rows),
    }
}

/// Gather the dictionary column `index` (with layout `column` and keys of
/// type `key_type`) of `rows`. The dictionary holds the distinct values of
/// the rows with a valid key (including null values) in order of first
/// occurrence, distinct by their stored bytes (so `-0.0` and `0.0` stay
/// distinct, and so do the null values of different union variants). Each
/// is decoded once: the values of all rows could be too large for their
/// type (e.g. nested run-end encoded values, or strings with offsets).
fn gather_dictionary<'r>(
    column: &ColumnLayout,
    index: usize,
    key_type: &DataType,
    rows: &[RowRef<'r>],
) -> Result<ArrayRef> {
    let Some(key_bit) = column.key_validity_bit() else {
        return internal_err!("Dictionary column {index} without a key validity bit");
    };
    let (offset, width) = (column.offset(), column.width());
    // The bytes of the slots of null values are zero, but an encoded null
    // value is stored with its encoding, which is part of the value.
    let stored = |row: RowRef<'r>| -> (bool, &'r [u8]) {
        let bytes = match column.storage() {
            ColumnStorage::VarLen | ColumnStorage::Encoded => row.var_len(offset),
            _ => &row.row[offset..offset + width],
        };
        (row.is_valid(index), bytes)
    };
    let mut keys_of_values =
        HashMap::with_capacity_and_hasher(rows.len(), RandomState::default());
    let mut distinct = Vec::new();
    let keys = rows
        .iter()
        .map(|&row| {
            if !row.is_valid(key_bit) {
                return 0;
            }
            *keys_of_values.entry(stored(row)).or_insert_with(|| {
                distinct.push(row);
                distinct.len() as u64 - 1
            })
        })
        .collect::<Vec<u64>>();
    let dictionary = gather_values(column, index, &distinct)?;
    new_dictionary(
        keys,
        gather_nulls(key_bit, rows),
        &dictionary,
        key_type,
        column.data_type(),
    )
}

/// Gather the values of column `index` of `rows`, with the value type of
/// the column (i.e. not dictionary encoded).
pub(crate) fn gather_values(
    column: &ColumnLayout,
    index: usize,
    rows: &[RowRef<'_>],
) -> Result<ArrayRef> {
    let offset = column.offset();
    let value_type = column.value_type();
    match column.storage() {
        ColumnStorage::Null => Ok(new_null_array(value_type, rows.len())),
        ColumnStorage::Boolean => {
            let values =
                BooleanBuffer::collect_bool(rows.len(), |i| rows[i].row[offset] != 0);
            Ok(Arc::new(BooleanArray::new(
                values,
                gather_nulls(index, rows),
            )))
        }
        ColumnStorage::Fixed(width) => {
            let values = gather_fixed(offset, width, rows);
            let data = ArrayDataBuilder::new(value_type.clone())
                .len(rows.len())
                .add_buffer(values.into())
                .nulls(gather_nulls(index, rows))
                .build()?;
            Ok(make_array(data))
        }
        ColumnStorage::VarLen => match value_type {
            DataType::Utf8 => gather_bytes::<Utf8Type>(offset, index, rows),
            DataType::LargeUtf8 => gather_bytes::<LargeUtf8Type>(offset, index, rows),
            DataType::Binary => gather_bytes::<BinaryType>(offset, index, rows),
            DataType::LargeBinary => gather_bytes::<LargeBinaryType>(offset, index, rows),
            DataType::Utf8View => gather_views::<StringViewType>(offset, index, rows),
            DataType::BinaryView => gather_views::<BinaryViewType>(offset, index, rows),
            data_type => {
                internal_err!("Unexpected variable-length data type {data_type}")
            }
        },
        ColumnStorage::Encoded => {
            let codec = column.codec().ok_or_else(|| {
                internal_datafusion_err!("Encoded column without a row codec")
            })?;
            codec.decode(rows.iter().map(|row| row.var_len(offset)))
        }
    }
}

/// The validity of column `index` of `rows`, `None` if all are valid.
fn gather_nulls(index: usize, rows: &[RowRef<'_>]) -> Option<NullBuffer> {
    let nulls = NullBuffer::new(BooleanBuffer::collect_bool(rows.len(), |i| {
        rows[i].is_valid(index)
    }));
    (nulls.null_count() > 0).then_some(nulls)
}

fn gather_fixed(offset: usize, width: usize, rows: &[RowRef<'_>]) -> MutableBuffer {
    fn gather<const W: usize>(offset: usize, rows: &[RowRef<'_>]) -> MutableBuffer {
        let mut values = MutableBuffer::from_len_zeroed(rows.len() * W);
        let (chunks, _) = values.as_slice_mut().as_chunks_mut::<W>();
        for (value, row) in chunks.iter_mut().zip(rows) {
            value.copy_from_slice(&row.row[offset..offset + W]);
        }
        values
    }
    match width {
        0 => MutableBuffer::new(0),
        1 => gather::<1>(offset, rows),
        2 => gather::<2>(offset, rows),
        4 => gather::<4>(offset, rows),
        8 => gather::<8>(offset, rows),
        16 => gather::<16>(offset, rows),
        32 => gather::<32>(offset, rows),
        width => {
            let mut values = MutableBuffer::new(rows.len() * width);
            for row in rows {
                values.extend_from_slice(&row.row[offset..offset + width]);
            }
            values
        }
    }
}

fn gather_bytes<T: ByteArrayType>(
    offset: usize,
    index: usize,
    rows: &[RowRef<'_>],
) -> Result<ArrayRef> {
    let len = rows
        .iter()
        .map(|row| row.var_len_size(offset))
        .sum::<usize>();
    if T::Offset::from_usize(len).is_none() {
        return exec_err!(
            "Cannot gather {len} bytes into an array of type {}",
            T::DATA_TYPE
        );
    }
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(T::Offset::usize_as(0));
    let mut values = MutableBuffer::new(len);
    for row in rows {
        values.extend_from_slice(row.var_len(offset));
        offsets.push(T::Offset::usize_as(values.len()));
    }
    let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
    Ok(Arc::new(GenericByteArray::<T>::try_new(
        offsets,
        values.into(),
        gather_nulls(index, rows),
    )?))
}

fn gather_views<T: ByteViewType + ?Sized>(
    offset: usize,
    index: usize,
    rows: &[RowRef<'_>],
) -> Result<ArrayRef> {
    /// The maximum size of a data buffer.
    const MAX_BUFFER_LEN: usize = i32::MAX as usize;
    let out_of_line = rows
        .iter()
        .map(|row| row.var_len_size(offset))
        .filter(|&len| len > MAX_INLINE_LEN)
        .sum::<usize>();
    let mut views = Vec::with_capacity(rows.len());
    let mut buffers = vec![];
    let mut buffer = MutableBuffer::new(out_of_line.min(MAX_BUFFER_LEN));
    for row in rows {
        let value = row.var_len(offset);
        if value.len() <= MAX_INLINE_LEN {
            views.push(make_view(value, 0, 0));
            continue;
        }
        if !buffer.is_empty() && buffer.len() + value.len() > MAX_BUFFER_LEN {
            let remaining = out_of_line.saturating_sub(buffer.len()).min(MAX_BUFFER_LEN);
            buffers.push(
                std::mem::replace(&mut buffer, MutableBuffer::new(remaining)).into(),
            );
        }
        let buffer_index = u32::try_from(buffers.len())
            .map_err(|_| exec_datafusion_err!("Too many view buffers"))?;
        let buffer_offset = u32::try_from(buffer.len())
            .map_err(|_| exec_datafusion_err!("View buffer too large"))?;
        views.push(make_view(value, buffer_index, buffer_offset));
        buffer.extend_from_slice(value);
    }
    if !buffer.is_empty() {
        buffers.push(buffer.into());
    }
    Ok(Arc::new(GenericByteViewArray::<T>::try_new(
        ScalarBuffer::from(views),
        buffers,
        gather_nulls(index, rows),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Int32Array, Int64Array, StringArray};
    use arrow::compute::cast;
    use arrow::datatypes::{Field, Fields, Schema};

    use crate::chunk::SelectionVector;
    use crate::tuple_data::test_data::{
        all_types_batch, assert_identical, chunk_with_selection, custom_run_end_encoded,
    };

    fn collection_for(batch: &RecordBatch, with_hash: bool) -> TupleDataCollection {
        let layout = TupleDataLayout::try_new(batch.schema(), with_hash).unwrap();
        TupleDataCollection::new(Arc::new(layout))
    }

    #[test]
    fn row_location_packing() {
        let location = RowLocation::new(123_456, 65_535);
        assert_eq!(location.block(), 123_456);
        assert_eq!(location.row(), 65_535);
        assert_eq!(location.as_u64(), (123_456 << 16) | 65_535);
        assert!(location.as_u64() < 1 << RowLocation::BITS);
        let salted = location.as_u64() | (0xABCD << RowLocation::BITS);
        assert_eq!(RowLocation::from_u64(salted), location);
        assert_eq!(
            location.with_block_offset(4),
            RowLocation::new(123_460, 65_535)
        );
        let max = RowLocation::new((RowLocation::MAX_BLOCKS - 1) as usize, 0);
        assert_eq!(max.block() as u64, RowLocation::MAX_BLOCKS - 1);
        assert!(RowLocation::new(0, 1) < RowLocation::new(1, 0));
    }

    #[test]
    fn round_trip_all_types() {
        let batch = all_types_batch(100);
        let mut collection = collection_for(&batch, false);
        let chunk = DataChunk::from_record_batch(batch.clone());
        let locations = collection.append_chunk(&chunk, None).unwrap();
        assert_eq!(locations.len(), 100);
        assert_eq!(collection.len(), 100);
        assert_identical(&collection.gather(&locations).unwrap(), &batch);
    }

    #[test]
    fn round_trip_with_selection_and_many_blocks() {
        let batch = all_types_batch(64);
        let layout = TupleDataLayout::try_new(batch.schema(), false)
            .unwrap()
            .with_rows_per_block(5)
            .unwrap();
        let mut collection = TupleDataCollection::new(Arc::new(layout));
        // Reversed, with gaps and repetitions.
        let selection = (0..64_u32).rev().filter(|i| i % 3 != 1).chain([7, 7, 0]);
        let chunk = chunk_with_selection(&batch, selection.collect());
        let expected = chunk.compact().unwrap();
        let locations = collection.append_chunk(&chunk, None).unwrap();
        assert_eq!(locations.len(), chunk.len());
        assert_eq!(collection.num_blocks(), chunk.len().div_ceil(5));
        assert!((0..collection.num_blocks()).all(|b| collection.block_len(b) <= 5));
        assert_identical(&collection.gather(&locations).unwrap(), &expected);

        // Gather the rows in another order.
        let mut shuffled = locations.clone();
        shuffled.reverse();
        shuffled.rotate_left(7);
        let indices = shuffled
            .iter()
            .map(|l| locations.iter().position(|x| x == l).unwrap() as u32)
            .collect::<Vec<_>>();
        let expected = chunk_with_selection(&expected, indices).compact().unwrap();
        assert_identical(&collection.gather(&shuffled).unwrap(), &expected);
    }

    #[test]
    fn round_trip_many_chunks() {
        let batch = all_types_batch(300);
        let layout = TupleDataLayout::try_new(batch.schema(), false)
            .unwrap()
            .with_rows_per_block(64)
            .unwrap();
        let mut collection = TupleDataCollection::new(Arc::new(layout));
        let mut locations = vec![];
        for offset in (0..300).step_by(37) {
            let len = 37.min(300 - offset);
            let chunk = DataChunk::from_record_batch(batch.slice(offset, len));
            locations.extend(collection.append_chunk(&chunk, None).unwrap());
        }
        assert_eq!(collection.len(), 300);
        assert_eq!(collection.num_blocks(), 5);
        assert_eq!(
            collection.row_locations().collect::<Vec<_>>(),
            locations,
            "rows are stored in insertion order"
        );
        assert!(
            locations
                .iter()
                .enumerate()
                .all(|(i, &l)| collection.row_index(l) == i)
        );
        assert_identical(&collection.gather(&locations).unwrap(), &batch);
    }

    #[test]
    fn gather_projected_and_scan() {
        let batch = all_types_batch(5000);
        let layout = TupleDataLayout::try_new(batch.schema(), false)
            .unwrap()
            .with_rows_per_block(1000)
            .unwrap();
        let mut collection = TupleDataCollection::new(Arc::new(layout));
        for offset in (0..5000).step_by(DEFAULT_VECTOR_SIZE) {
            let len = DEFAULT_VECTOR_SIZE.min(5000 - offset);
            collection
                .append_chunk(
                    &DataChunk::from_record_batch(batch.slice(offset, len)),
                    None,
                )
                .unwrap();
        }

        // A full scan returns all rows in chunks of at most 2048 rows.
        let mut state = TupleDataScanState::new();
        let mut chunks = vec![];
        let mut locations = vec![];
        while let Some(chunk) = collection.scan_chunk(&mut state).unwrap() {
            assert!(chunk.len() <= DEFAULT_VECTOR_SIZE);
            assert_eq!(state.locations().len(), chunk.len());
            locations.extend_from_slice(state.locations());
            chunks.push(chunk.into_record_batch().unwrap());
        }
        assert_eq!(chunks.len(), 3);
        assert_eq!(locations, collection.row_locations().collect::<Vec<_>>());
        let scanned = arrow::compute::concat_batches(&batch.schema(), &chunks).unwrap();
        assert_identical(&scanned, &batch);
        assert!(collection.scan_chunk(&mut state).unwrap().is_none());

        // A projected scan of some blocks.
        let column_ids = vec![3, 0, 3];
        let mut state = TupleDataScanState::new()
            .with_column_ids(column_ids.clone())
            .with_block_range(1..3)
            .with_chunk_size(700);
        let mut chunks = vec![];
        while let Some(chunk) = collection.scan_chunk(&mut state).unwrap() {
            assert!(chunk.len() <= 700);
            chunks.push(chunk.into_record_batch().unwrap());
        }
        assert_eq!(
            chunks.iter().map(RecordBatch::num_rows).sum::<usize>(),
            2000
        );
        let projected = batch.slice(1000, 2000).project(&column_ids).unwrap();
        let scanned =
            arrow::compute::concat_batches(&projected.schema(), &chunks).unwrap();
        assert_identical(&scanned, &projected);

        let locations = locations.into_iter().step_by(7).collect::<Vec<_>>();
        let gathered = collection
            .gather_projected(&locations, &column_ids)
            .unwrap();
        let indices = (0..5000).step_by(7).map(|i| i as u32).collect::<Vec<_>>();
        let expected = chunk_with_selection(&projected_all(&batch, &column_ids), indices)
            .compact()
            .unwrap();
        assert_identical(&gathered, &expected);
    }

    fn projected_all(batch: &RecordBatch, column_ids: &[usize]) -> RecordBatch {
        batch.project(column_ids).unwrap()
    }

    #[test]
    fn empty_chunks_and_empty_gathers() {
        let batch = all_types_batch(10);
        let mut collection = collection_for(&batch, true);

        // A cleared chunk, a chunk without rows and a chunk with an empty
        // selection add nothing.
        let empty = DataChunk::empty(batch.schema());
        assert!(
            collection
                .append_chunk(&empty, Some(&[]))
                .unwrap()
                .is_empty()
        );
        let no_rows = DataChunk::from_record_batch(batch.slice(0, 0));
        assert!(
            collection
                .append_chunk(&no_rows, Some(&[]))
                .unwrap()
                .is_empty()
        );
        let none_selected = chunk_with_selection(&batch, vec![]);
        assert!(
            collection
                .append_chunk(&none_selected, Some(&[]))
                .unwrap()
                .is_empty()
        );
        assert!(collection.is_empty());
        assert_eq!(collection.num_blocks(), 0);

        let gathered = collection.gather(&[]).unwrap();
        assert_identical(&gathered, &batch.slice(0, 0));
        let mut state = TupleDataScanState::new();
        assert!(collection.scan_chunk(&mut state).unwrap().is_none());
        assert!(collection.gather_hashes(&[]).unwrap().is_empty());
    }

    #[test]
    fn layout_without_columns() {
        let schema = Arc::new(Schema::empty());
        let layout =
            Arc::new(TupleDataLayout::try_new(Arc::clone(&schema), false).unwrap());
        let mut collection = TupleDataCollection::new(Arc::clone(&layout));
        let chunk = DataChunk::try_new_without_columns(Arc::clone(&schema), 5).unwrap();
        let locations = collection.append_chunk(&chunk, None).unwrap();
        assert_eq!(locations.len(), 5);
        let batch = collection.gather(&locations[1..]).unwrap();
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 0);

        // Only hashes.
        let layout =
            Arc::new(TupleDataLayout::try_new(Arc::clone(&schema), true).unwrap());
        let mut collection = TupleDataCollection::new(layout);
        let locations = collection
            .append_chunk(&chunk, Some(&[1, 2, 3, 4, 5]))
            .unwrap();
        assert_eq!(
            collection.gather_hashes(&locations[2..]).unwrap(),
            vec![3, 4, 5]
        );
    }

    #[test]
    fn hashes_are_stored_per_logical_row() {
        let batch = all_types_batch(20);
        let mut collection = collection_for(&batch, true);
        let chunk = chunk_with_selection(&batch, vec![19, 3, 4, 11]);
        let hashes = [u64::MAX, 0, 42, 1 << 63];
        let locations = collection.append_chunk(&chunk, Some(&hashes)).unwrap();
        assert_eq!(collection.gather_hashes(&locations).unwrap(), hashes);
        assert_identical(
            &collection.gather(&locations).unwrap(),
            &chunk.compact().unwrap(),
        );

        // Wrong number of hashes, missing and unexpected hashes.
        let err = collection.append_chunk(&chunk, Some(&[1, 2])).unwrap_err();
        assert!(err.to_string().contains("2 hashes"), "{err}");
        let err = collection.append_chunk(&chunk, None).unwrap_err();
        assert!(err.to_string().contains("no hashes"), "{err}");
        let mut without_hash = collection_for(&batch, false);
        let err = without_hash
            .append_chunk(&chunk, Some(&hashes))
            .unwrap_err();
        assert!(err.to_string().contains("no hash slot"), "{err}");
        assert!(without_hash.gather_hashes(&[]).is_err());
        assert_eq!(collection.len(), 4);
    }

    #[test]
    fn var_len_values_inline_and_in_heap() {
        let values = vec![
            Some(""),
            Some("a"),
            Some("exactly 12 b"),
            Some("thirteen byte"),
            None,
            Some("a much longer string that is stored in the heap"),
            Some("ünïcödé ✓ strings of all kinds"),
            Some("exactly 12 b"),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Utf8View, true),
            Field::new("b", DataType::LargeBinary, true),
        ]));
        let strings = StringArray::from(values.clone());
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(strings.clone()),
                cast(&strings, &DataType::Utf8View).unwrap(),
                cast(&strings, &DataType::LargeBinary).unwrap(),
            ],
        )
        .unwrap();
        let layout = TupleDataLayout::try_new(schema, false)
            .unwrap()
            .with_rows_per_block(3)
            .unwrap();
        let mut collection = TupleDataCollection::new(Arc::new(layout));
        let locations = collection
            .append_chunk(&DataChunk::from_record_batch(batch.clone()), None)
            .unwrap();
        assert_identical(&collection.gather(&locations).unwrap(), &batch);
        // Only values longer than 12 bytes use the heap, once per column.
        let heap_bytes = values
            .iter()
            .flatten()
            .filter(|v| v.len() > MAX_INLINE_LEN)
            .map(|v| v.len())
            .sum::<usize>();
        let heaps = collection
            .blocks
            .iter()
            .map(|b| b.heap.len())
            .sum::<usize>();
        assert_eq!(heaps, 3 * heap_bytes);
    }

    #[test]
    fn combine_moves_blocks_without_copying() {
        let batch = all_types_batch(50);
        let layout = Arc::new(
            TupleDataLayout::try_new(batch.schema(), true)
                .unwrap()
                .with_rows_per_block(8)
                .unwrap(),
        );
        let hashes = (0..50).collect::<Vec<u64>>();
        let mut first = TupleDataCollection::new(Arc::clone(&layout));
        let first_locations = first
            .append_chunk(
                &DataChunk::from_record_batch(batch.slice(0, 20)),
                Some(&hashes[..20]),
            )
            .unwrap();
        let mut second = TupleDataCollection::new(Arc::clone(&layout));
        let second_locations = second
            .append_chunk(
                &DataChunk::from_record_batch(batch.slice(20, 30)),
                Some(&hashes[20..]),
            )
            .unwrap();
        let second_blocks = second.num_blocks();
        let row_buffers = second
            .blocks
            .iter()
            .map(|b| b.rows.as_ptr())
            .collect::<Vec<_>>();
        let memory = first.memory_size() + second.memory_size();

        let offset = first.num_blocks();
        first.combine(second).unwrap();
        assert_eq!(first.len(), 50);
        assert_eq!(first.num_blocks(), offset + second_blocks);
        // The blocks were moved, not copied.
        assert_eq!(
            first.blocks[offset..]
                .iter()
                .map(|b| b.rows.as_ptr())
                .collect::<Vec<_>>(),
            row_buffers
        );
        assert!(first.memory_size() <= memory);

        let locations = first_locations
            .into_iter()
            .chain(
                second_locations
                    .into_iter()
                    .map(|l| l.with_block_offset(offset)),
            )
            .collect::<Vec<_>>();
        assert_eq!(locations, first.row_locations().collect::<Vec<_>>());
        assert_identical(&first.gather(&locations).unwrap(), &batch);
        assert_eq!(first.gather_hashes(&locations).unwrap(), hashes);
        assert!(
            locations
                .iter()
                .enumerate()
                .all(|(i, &l)| first.row_index(l) == i)
        );

        // The combined collection can still be appended to.
        let more = first
            .append_chunk(
                &DataChunk::from_record_batch(batch.slice(0, 10)),
                Some(&hashes[..10]),
            )
            .unwrap();
        assert_identical(&first.gather(&more).unwrap(), &batch.slice(0, 10));
        assert_eq!(first.len(), 60);

        // Combining an empty collection is a no-op, combining collections
        // with different layouts is an error.
        first
            .combine(TupleDataCollection::new(Arc::clone(&layout)))
            .unwrap();
        assert_eq!(first.len(), 60);
        let other = collection_for(&batch, false);
        assert!(first.combine(other).is_err());
    }

    #[test]
    fn memory_size_grows_with_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
        ]));
        let layout =
            Arc::new(TupleDataLayout::try_new(Arc::clone(&schema), false).unwrap());
        let mut collection = TupleDataCollection::new(Arc::clone(&layout));
        let empty_size = collection.memory_size();
        assert!(empty_size > 0);
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..1000)),
                Arc::new(StringArray::from_iter_values(
                    (0..1000).map(|i| format!("a string longer than 12 bytes {i}")),
                )),
            ],
        )
        .unwrap();
        collection
            .append_chunk(&DataChunk::from_record_batch(batch), None)
            .unwrap();
        // At least the rows and the heap.
        let at_least = 1000 * layout.row_width() + 1000 * 30;
        assert!(collection.memory_size() >= empty_size + at_least);
        // Blocks grow geometrically, so the waste is bounded.
        assert!(collection.memory_size() <= empty_size + 3 * at_least);
    }

    #[test]
    fn invalid_inputs_are_errors() {
        let batch = all_types_batch(4);
        let mut collection = collection_for(&batch, false);
        let locations = collection
            .append_chunk(&DataChunk::from_record_batch(batch.clone()), None)
            .unwrap();

        // Invalid locations.
        let err = collection.gather(&[RowLocation::new(0, 4)]).unwrap_err();
        assert!(err.to_string().contains("Invalid row location"), "{err}");
        let err = collection.gather(&[RowLocation::new(1, 0)]).unwrap_err();
        assert!(err.to_string().contains("Invalid row location"), "{err}");
        // Invalid column ids.
        let err = collection
            .gather_columns(&locations, &[batch.num_columns()])
            .unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");

        // Chunks that do not match the layout.
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let err = collection
            .append_chunk(&DataChunk::from_record_batch(other.clone()), None)
            .unwrap_err();
        assert!(err.to_string().contains("columns"), "{err}");
        let layout =
            TupleDataLayout::try_new_from_types([DataType::Int64], false).unwrap();
        let mut collection = TupleDataCollection::new(Arc::new(layout));
        let err = collection
            .append_chunk(&DataChunk::from_record_batch(other), None)
            .unwrap_err();
        assert!(err.to_string().contains("expects Int64"), "{err}");
    }

    #[test]
    fn dictionary_columns_are_re_encoded() {
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let schema =
            Arc::new(Schema::new(vec![Field::new("d", dictionary.clone(), true)]));
        let layout =
            Arc::new(TupleDataLayout::try_new(Arc::clone(&schema), false).unwrap());
        let mut collection = TupleDataCollection::new(layout);
        // Two chunks with different dictionaries, which together have more
        // values than an `Int8` key can address.
        let mut locations = vec![];
        for chunk in 0..2 {
            let values = (0..100)
                .map(|i| Some(format!("value {}", chunk * 100 + i)))
                .chain([None])
                .collect::<Vec<_>>();
            let array = values
                .iter()
                .map(Option::as_deref)
                .collect::<arrow::array::DictionaryArray<arrow::datatypes::Int8Type>>();
            let batch =
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();
            locations.extend(
                collection
                    .append_chunk(&DataChunk::from_record_batch(batch), None)
                    .unwrap(),
            );
        }
        // Gathering rows of both dictionaries re-encodes their values.
        let some = [0, 150, 1, 100, 151, 200, 0].map(|i| locations[i]);
        let gathered = collection.gather(&some).unwrap();
        let column = gathered
            .column(0)
            .as_dictionary::<arrow::datatypes::Int8Type>();
        assert_eq!(column.data_type(), &dictionary);
        assert_eq!(column.values().len(), 5);
        assert_eq!(
            column.keys(),
            &arrow::array::Int8Array::from(vec![
                Some(0),
                Some(1),
                Some(2),
                None,
                Some(3),
                Some(4),
                Some(0),
            ])
        );
        let values = cast(column, &DataType::Utf8).unwrap();
        assert_eq!(
            values.as_string::<i32>().iter().collect::<Vec<_>>(),
            vec![
                Some("value 0"),
                Some("value 149"),
                Some("value 1"),
                None,
                Some("value 150"),
                Some("value 199"),
                Some("value 0"),
            ]
        );
        // All 200 values do not fit `Int8` keys.
        let err = collection.gather(&locations).unwrap_err();
        assert!(err.to_string().contains("dictionary encode"), "{err}");
    }

    #[test]
    fn selection_vector_is_respected_without_compacting() {
        // A selection that excludes a row with a value that the chunk can not
        // even hold (the excluded row is never read).
        let batch = all_types_batch(8);
        let chunk = DataChunk::from_record_batch(batch.clone())
            .with_selection(SelectionVector::from(vec![6, 1]))
            .unwrap();
        let mut collection = collection_for(&batch, false);
        let locations = collection.append_chunk(&chunk, None).unwrap();
        assert_eq!(locations.len(), 2);
        assert_identical(
            &collection.gather(&locations).unwrap(),
            &chunk.compact().unwrap(),
        );
        // The chunk itself is untouched.
        assert_eq!(chunk.physical_len(), 8);
        assert_eq!(chunk.selection().unwrap().indices(), &[6, 1]);
    }

    #[test]
    fn nested_dictionaries_are_restored() {
        use arrow::array::{
            FixedSizeListBuilder, LargeListBuilder, MapBuilder, StringBuilder,
            StringDictionaryBuilder, StructArray,
        };
        use arrow::datatypes::{Int8Type, Int16Type, Int32Type};

        let n = 40;
        let value = |i: usize| format!("value {}", i % 7);
        let mut fixed_list =
            FixedSizeListBuilder::new(StringDictionaryBuilder::<Int32Type>::new(), 2);
        let mut large_list =
            LargeListBuilder::new(StringDictionaryBuilder::<Int8Type>::new());
        let mut map = MapBuilder::new(
            None,
            StringBuilder::new(),
            StringDictionaryBuilder::<Int16Type>::new(),
        );
        for i in 0..n {
            fixed_list.values().append_value(value(i));
            fixed_list.values().append_value(value(i + 1));
            fixed_list.append(i % 5 != 1);
            for j in 0..i % 3 {
                large_list.values().append_value(value(i + j));
                map.keys().append_value(format!("k{j}"));
                map.values().append_value(value(i * j));
            }
            large_list.append(i % 4 != 2);
            map.append(i % 6 != 3).unwrap();
        }
        let dictionary = (0..n)
            .map(|i| (i % 3 != 0).then(|| value(i)))
            .collect::<Vec<_>>();
        let dictionary = dictionary
            .iter()
            .map(Option::as_deref)
            .collect::<arrow::array::DictionaryArray<Int16Type>>();
        let structs = StructArray::try_from(vec![
            ("d", Arc::new(dictionary) as ArrayRef),
            (
                "i",
                Arc::new(Int32Array::from_iter_values(0..n as i32)) as ArrayRef,
            ),
        ])
        .unwrap();
        let batch = RecordBatch::try_from_iter(vec![
            ("fixed_list", Arc::new(fixed_list.finish()) as ArrayRef),
            ("large_list", Arc::new(large_list.finish()) as ArrayRef),
            ("map", Arc::new(map.finish()) as ArrayRef),
            ("struct", Arc::new(structs) as ArrayRef),
        ])
        .unwrap();
        let mut collection = collection_for(&batch, false);
        for column in collection.layout().columns() {
            assert_eq!(column.storage(), ColumnStorage::Encoded);
        }
        let chunk = chunk_with_selection(&batch, (0..n as u32).rev().collect());
        let locations = collection.append_chunk(&chunk, None).unwrap();
        assert_identical(
            &collection.gather(&locations).unwrap(),
            &chunk.compact().unwrap(),
        );
    }

    #[test]
    fn run_end_encoded_fields_are_restored() {
        use arrow::array::{ListArray, StructArray};

        // Top level: 8 rows.
        let top = custom_run_end_encoded(
            Arc::new(StringArray::from(vec![
                Some("a"),
                None,
                Some("b"),
                Some("a"),
            ])),
            &[2, 3, 6, 8],
            true,
        );
        // In a list: 8 lists of 12 run-end encoded values.
        let values = custom_run_end_encoded(
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            &[3, 4, 8, 9, 12],
            false,
        );
        let item = Arc::new(Field::new("element", values.data_type().clone(), false));
        let lists: ArrayRef = Arc::new(ListArray::new(
            item,
            OffsetBuffer::from_lengths([2, 0, 3, 1, 2, 1, 0, 3]),
            values,
            Some(NullBuffer::from(vec![
                true, true, false, true, true, true, true, true,
            ])),
        ));
        // In a struct, next to another field.
        let samples = custom_run_end_encoded(
            Arc::new(Int32Array::from(vec![Some(7), None])),
            &[5, 8],
            true,
        );
        let structs: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![
                    Field::new("r", samples.data_type().clone(), true),
                    Field::new("i", DataType::Int32, true),
                ]
                .into(),
                vec![samples, Arc::new(Int32Array::from_iter_values(0..8))],
                Some(NullBuffer::from(vec![
                    true, false, true, true, true, true, true, true,
                ])),
            )
            .unwrap(),
        );
        // As the values of a map.
        let map_values = custom_run_end_encoded(
            Arc::new(StringArray::from(vec![Some("x"), Some("y"), None])),
            &[4, 6, 9],
            true,
        );
        let entry_fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", map_values.data_type().clone(), true),
        ]);
        let entries = StructArray::try_new(
            entry_fields,
            vec![
                Arc::new(StringArray::from_iter_values(
                    (0..9).map(|i| format!("k{i}")),
                )),
                map_values,
            ],
            None,
        )
        .unwrap();
        let maps: ArrayRef = Arc::new(
            arrow::array::MapArray::try_new(
                Arc::new(Field::new("entries", entries.data_type().clone(), false)),
                OffsetBuffer::from_lengths([1, 2, 0, 1, 1, 3, 0, 1]),
                entries,
                Some(NullBuffer::from(vec![
                    true, true, true, false, true, true, true, true,
                ])),
                false,
            )
            .unwrap(),
        );
        // With dictionary values, which are cast back after decoding.
        let dictionary = [Some("p"), None, Some("q")]
            .into_iter()
            .collect::<arrow::array::DictionaryArray<arrow::datatypes::Int8Type>>(
        );
        let dictionaries = custom_run_end_encoded(Arc::new(dictionary), &[1, 5, 8], true);
        let batch = RecordBatch::try_from_iter(vec![
            ("top", top),
            ("lists", lists),
            ("structs", structs),
            ("maps", maps),
            ("dictionaries", dictionaries),
        ])
        .unwrap();

        let mut collection = collection_for(&batch, false);
        let chunk = chunk_with_selection(&batch, vec![7, 0, 3, 3, 5, 1, 2, 6, 4]);
        let locations = collection.append_chunk(&chunk, None).unwrap();
        // The exact schema, with the custom fields, is restored, also when no
        // rows are gathered.
        let gathered = collection.gather(&locations).unwrap();
        assert_eq!(gathered.schema(), batch.schema());
        assert_identical(&gathered, &chunk.compact().unwrap());
        assert_identical(&collection.gather(&[]).unwrap(), &batch.slice(0, 0));

        let mut state = TupleDataScanState::new().with_column_ids(vec![2, 4, 0, 3, 1]);
        let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
        let expected = chunk.compact().unwrap().project(&[2, 4, 0, 3, 1]).unwrap();
        assert_eq!(scanned.schema(), &expected.schema());
        assert_identical(&scanned.into_record_batch().unwrap(), &expected);
    }

    #[test]
    fn run_end_encoded_fields_in_unions_are_restored() {
        use arrow::array::{ListArray, UnionArray};
        use arrow::datatypes::UnionFields;

        let samples = custom_run_end_encoded(
            Arc::new(StringArray::from(vec![Some("a"), None, Some("b")])),
            &[2, 3, 6],
            true,
        );
        let fields = UnionFields::try_new(
            [0, 1],
            [
                Field::new("r", samples.data_type().clone(), true),
                Field::new("i", DataType::Int32, true),
            ],
        )
        .unwrap();
        let type_ids = [0, 1, 0, 0, 1, 0].into_iter().collect::<ScalarBuffer<i8>>();
        let sparse: ArrayRef = Arc::new(
            UnionArray::try_new(
                fields.clone(),
                type_ids.clone(),
                None,
                vec![
                    Arc::clone(&samples),
                    Arc::new(Int32Array::from(vec![
                        Some(1),
                        Some(2),
                        None,
                        Some(3),
                        None,
                        Some(4),
                    ])),
                ],
            )
            .unwrap(),
        );
        let dense: ArrayRef = Arc::new(
            UnionArray::try_new(
                fields,
                type_ids,
                Some(vec![0, 0, 1, 2, 1, 3].into()),
                vec![
                    custom_run_end_encoded(
                        Arc::new(StringArray::from(vec![Some("c"), None])),
                        &[1, 4],
                        true,
                    ),
                    Arc::new(Int32Array::from(vec![Some(5), None])),
                ],
            )
            .unwrap(),
        );
        let lists: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(sparse.data_type().clone(), true)),
            OffsetBuffer::from_lengths([2, 0, 1, 2, 1, 0]),
            Arc::clone(&sparse),
            Some(NullBuffer::from(vec![true, true, false, true, true, true])),
        ));
        let batch = RecordBatch::try_from_iter(vec![
            ("sparse", sparse),
            ("dense", dense),
            ("lists", lists),
        ])
        .unwrap();
        let mut collection = collection_for(&batch, false);
        let chunk = chunk_with_selection(&batch, vec![5, 0, 2, 2, 4, 1, 3]);
        let locations = collection.append_chunk(&chunk, None).unwrap();
        assert_identical(
            &collection.gather(&locations).unwrap(),
            &chunk.compact().unwrap(),
        );
    }

    #[test]
    fn dictionary_keys_of_null_values_are_restored() {
        use arrow::array::{DictionaryArray, Int8Array};
        use arrow::datatypes::Int8Type;

        // Valid keys of a null value: the field is not nullable, as the
        // dictionary has no null keys.
        let dictionary = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![0, 1]),
            Arc::new(StringArray::from(vec![Some("x"), None])),
        )
        .unwrap();
        let batch =
            RecordBatch::try_from_iter(vec![("d", Arc::new(dictionary) as ArrayRef)])
                .unwrap();
        assert!(!batch.schema().field(0).is_nullable());
        let mut collection = collection_for(&batch, false);
        let locations = collection
            .append_chunk(&DataChunk::from_record_batch(batch.clone()), None)
            .unwrap();
        let gathered = collection.gather(&locations).unwrap();
        assert_identical(&gathered, &batch);
        assert_eq!(gathered.column(0).to_data(), batch.column(0).to_data());
        let mut state = TupleDataScanState::new();
        let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
        assert_eq!(scanned.schema(), &batch.schema());
        assert_identical(&scanned.into_record_batch().unwrap(), &batch);

        // In a nullable field, null keys and valid keys of null values are
        // both restored as they were.
        let dictionary = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![Some(1), None, Some(0), Some(2), None, Some(1)]),
            Arc::new(StringArray::from(vec![Some("x"), None, Some("y")])),
        )
        .unwrap();
        let batch =
            RecordBatch::try_from_iter(vec![("d", Arc::new(dictionary) as ArrayRef)])
                .unwrap();
        assert!(batch.schema().field(0).is_nullable());
        let mut collection = collection_for(&batch, false);
        let chunk = chunk_with_selection(&batch, vec![5, 4, 3, 2, 1, 0, 0]);
        let locations = collection.append_chunk(&chunk, None).unwrap();
        let gathered = collection.gather(&locations).unwrap();
        let expected = chunk.compact().unwrap();
        assert_identical(&gathered, &expected);
        let keys = gathered
            .column(0)
            .as_any_dictionary()
            .keys()
            .nulls()
            .cloned();
        assert_eq!(
            keys,
            expected
                .column(0)
                .as_any_dictionary()
                .keys()
                .nulls()
                .cloned()
        );
        assert_eq!(gathered.column(0).to_data(), expected.column(0).to_data());
    }

    #[test]
    fn nested_dictionary_keys_of_null_values_are_restored() {
        use arrow::array::{
            DictionaryArray, FixedSizeListArray, Int8Array, StructArray, UnionArray,
        };
        use arrow::datatypes::{Int8Type, UnionFields};

        // Dictionaries with valid keys of a null value in non-nullable
        // fields, where Arrow allows them: in the values of run-end encoded
        // arrays and in the children of unions (which do not check the
        // nullability of their fields), and under null parents (structs and
        // fixed size lists check the logical nulls of their children).
        let dictionary = |keys: Vec<i8>| -> ArrayRef {
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(
                    Int8Array::from(keys),
                    Arc::new(StringArray::from(vec![Some("x"), None, Some("y")])),
                )
                .unwrap(),
            )
        };
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let field = |name: &str| Field::new(name, dictionary_type.clone(), false);
        let runs = ArrayDataBuilder::new(DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(field("values")),
        ))
        .len(5)
        .add_child_data(Int32Array::from(vec![1, 3, 5]).to_data())
        .add_child_data(dictionary(vec![1, 0, 1]).to_data())
        .build()
        .unwrap();
        let unions: ArrayRef = Arc::new(
            UnionArray::try_new(
                UnionFields::try_new(
                    [0, 1],
                    [field("d"), Field::new("i", DataType::Int32, true)],
                )
                .unwrap(),
                vec![0, 0, 1, 0, 0].into(),
                None,
                vec![
                    dictionary(vec![1, 0, 0, 2, 1]),
                    Arc::new(Int32Array::from(vec![None, None, Some(3), None, None])),
                ],
            )
            .unwrap(),
        );
        let structs: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![field("d")].into(),
                vec![dictionary(vec![1, 0, 1, 2, 0])],
                Some(NullBuffer::from(vec![false, true, false, true, true])),
            )
            .unwrap(),
        );
        let fixed: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(field("item")),
                2,
                dictionary(vec![1, 1, 0, 2, 1, 0, 0, 0, 2, 1]),
                Some(NullBuffer::from(vec![false, true, false, true, false])),
            )
            .unwrap(),
        );
        let batch = RecordBatch::try_from_iter(vec![
            ("runs", make_array(runs)),
            ("unions", unions),
            ("structs", structs),
            ("fixed", fixed),
        ])
        .unwrap();
        let mut collection = collection_for(&batch, false);
        let chunk = chunk_with_selection(&batch, vec![4, 3, 2, 1, 0, 2]);
        let locations = collection.append_chunk(&chunk, None).unwrap();
        let expected = chunk.compact().unwrap();
        let gathered = collection.gather(&locations).unwrap();
        assert_identical(&gathered, &expected);
        // The dictionaries in non-nullable fields have no null keys.
        let runs = gathered.column(0).to_data();
        let unions = gathered.column(1).to_data();
        let structs = gathered.column(2).to_data();
        let fixed = gathered.column(3).to_data();
        for dictionary in [
            &runs.child_data()[1],
            &unions.child_data()[0],
            &structs.child_data()[0],
            &fixed.child_data()[0],
        ] {
            assert!(matches!(dictionary.data_type(), DataType::Dictionary(..)));
            assert_eq!(dictionary.null_count(), 0);
        }
        let mut state = TupleDataScanState::new().with_column_ids(vec![3, 0, 2, 1]);
        let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
        assert_identical(
            &scanned.into_record_batch().unwrap(),
            &expected.project(&[3, 0, 2, 1]).unwrap(),
        );
    }

    #[test]
    fn exotic_types_round_trip() {
        for (name, array, nullable) in crate::tuple_data::test_data::exotic_arrays() {
            let schema = Arc::new(Schema::new(vec![Field::new(
                name,
                array.data_type().clone(),
                nullable,
            )]));
            let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
            let mut collection = collection_for(&batch, false);
            let locations = collection
                .append_chunk(&DataChunk::from_record_batch(batch.clone()), None)
                .unwrap();
            assert_identical(&collection.gather(&locations).unwrap(), &batch);
            let reversed = locations.iter().rev().copied().collect::<Vec<_>>();
            let indices = (0..batch.num_rows() as u32).rev().collect::<Vec<_>>();
            assert_identical(
                &collection.gather(&reversed).unwrap(),
                &chunk_with_selection(&batch, indices).compact().unwrap(),
            );
            assert_identical(&collection.gather(&[]).unwrap(), &batch.slice(0, 0));
        }
    }

    #[test]
    fn full_dictionaries_under_masked_slots() {
        use arrow::array::{
            DictionaryArray, FixedSizeListArray, Int8Array, StructArray, UnionArray,
        };
        use arrow::datatypes::{Int8Type, UnionFields};

        // Non-nullable dictionaries that use all 128 Int8 keys, with one
        // more row in a slot that is masked by a null parent or that belongs
        // to an inactive union variant: the masked value must not need a
        // dictionary entry of its own.
        let dictionary = || -> ArrayRef {
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(
                    Int8Array::from_iter_values((0..=127).chain([0])),
                    Arc::new(Int32Array::from_iter_values(0..=127)),
                )
                .unwrap(),
            )
        };
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32));
        let field = |name: &str| Field::new(name, dictionary_type.clone(), false);
        let valid_but_last =
            || NullBuffer::from((0..129).map(|i| i < 128).collect::<Vec<_>>());
        let structs: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![field("d")].into(),
                vec![dictionary()],
                Some(valid_but_last()),
            )
            .unwrap(),
        );
        let fixed: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(field("item")),
                1,
                dictionary(),
                Some(valid_but_last()),
            )
            .unwrap(),
        );
        let unions: ArrayRef = Arc::new(
            UnionArray::try_new(
                UnionFields::try_new(
                    [0, 1],
                    [field("d"), Field::new("i", DataType::Int32, true)],
                )
                .unwrap(),
                (0..129).map(|i| i8::from(i >= 128)).collect(),
                None,
                vec![dictionary(), Arc::new(Int32Array::from(vec![7; 129]))],
            )
            .unwrap(),
        );
        // The same in a nullable field (the masked slots get null keys), and
        // a struct in a null struct.
        let nullable_structs: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![Field::new("d", dictionary_type.clone(), true)].into(),
                vec![dictionary()],
                Some(valid_but_last()),
            )
            .unwrap(),
        );
        let inner =
            StructArray::try_new(vec![field("d")].into(), vec![dictionary()], None)
                .unwrap();
        let nested_structs: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![Field::new("inner", inner.data_type().clone(), false)].into(),
                vec![Arc::new(inner)],
                Some(valid_but_last()),
            )
            .unwrap(),
        );
        for (name, array) in [
            ("structs", structs),
            ("fixed", fixed),
            ("unions", unions),
            ("nullable structs", nullable_structs),
            ("nested structs", nested_structs),
        ] {
            let batch = RecordBatch::try_from_iter(vec![(name, array)]).unwrap();
            let mut collection = collection_for(&batch, false);
            let locations = collection
                .append_chunk(&DataChunk::from_record_batch(batch.clone()), None)
                .unwrap();
            assert_identical(&collection.gather(&locations).unwrap(), &batch);
            let mut state = TupleDataScanState::new();
            let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
            assert_eq!(scanned.schema(), &batch.schema());
            assert_identical(&scanned.into_record_batch().unwrap(), &batch);
        }
    }

    /// Randomized round trips of non-nullable and nullable nested
    /// dictionaries that use (almost) all their `Int8` keys, with more rows
    /// in slots that are masked by null parents or belong to inactive union
    /// variants, and with null values where Arrow allows them.
    #[test]
    fn randomized_nested_dictionaries_round_trip() {
        use arrow::array::{
            DictionaryArray, FixedSizeListArray, Int8Array, ListArray, StructArray,
            UnionArray,
        };
        use arrow::datatypes::{Int8Type, UnionFields};
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};

        for seed in 0..200 {
            let mut rng = StdRng::seed_from_u64(seed);
            let nullable = rng.random_bool(0.5);
            // Value 3 of the dictionary is null if `null_value`.
            let null_value = rng.random_bool(0.5);
            let size = if rng.random_bool(0.5) {
                128
            } else {
                rng.random_range(120..128)
            };
            // The first `size` rows matter (they are valid and of the variant
            // of the dictionary) and use every value once, the other rows are
            // random, and often masked or of the other variant.
            let n = size + rng.random_range(1..40);
            let matters = |rng: &mut StdRng, i: usize| i < size || rng.random_bool(0.4);
            let parent_nulls = NullBuffer::from(
                (0..n).map(|i| matters(&mut rng, i)).collect::<Vec<_>>(),
            );
            let type_ids = (0..n)
                .map(|i| i8::from(!matters(&mut rng, i)))
                .collect::<Vec<_>>();
            let random_keys = (0..n)
                .map(|i| {
                    if i < size {
                        i
                    } else {
                        rng.random_range(0..size)
                    }
                })
                .collect::<Vec<_>>();
            // Structs and fixed size lists reject the null values of
            // non-nullable children that are not masked by a null parent.
            let masked_keys = random_keys
                .iter()
                .enumerate()
                .map(|(i, &key)| {
                    let unmasked_null =
                        key == 3 && null_value && parent_nulls.is_valid(i);
                    if unmasked_null && !nullable {
                        4
                    } else {
                        key as i8
                    }
                })
                .collect::<Vec<_>>();
            let all_keys = random_keys.iter().map(|&key| key as i8).collect::<Vec<_>>();
            let dictionary = |keys: Vec<i8>| -> ArrayRef {
                let values =
                    (0..size as i32).map(|i| (i != 3 || !null_value).then_some(i));
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(keys),
                        Arc::new(Int32Array::from_iter(values)),
                    )
                    .unwrap(),
                )
            };
            let data_type =
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32));
            let field = |name: &str| Field::new(name, data_type.clone(), nullable);
            let union_fields = || {
                UnionFields::try_new(
                    [0, 1],
                    [field("d"), Field::new("i", DataType::Int32, true)],
                )
                .unwrap()
            };
            let case = rng.random_range(0..6);
            let array: ArrayRef = match case {
                0 => Arc::new(
                    StructArray::try_new(
                        vec![field("d")].into(),
                        vec![dictionary(masked_keys)],
                        Some(parent_nulls),
                    )
                    .unwrap(),
                ),
                1 => Arc::new(
                    FixedSizeListArray::try_new(
                        Arc::new(field("item")),
                        1,
                        dictionary(masked_keys),
                        Some(parent_nulls),
                    )
                    .unwrap(),
                ),
                2 => Arc::new(
                    UnionArray::try_new(
                        union_fields(),
                        type_ids.into(),
                        None,
                        vec![
                            dictionary(all_keys),
                            Arc::new(Int32Array::from(vec![1; n])),
                        ],
                    )
                    .unwrap(),
                ),
                3 => {
                    // A dense union, with the keys of the active rows.
                    let mut counts = [0_usize; 2];
                    let offsets = type_ids
                        .iter()
                        .map(|&id| {
                            counts[id as usize] += 1;
                            (counts[id as usize] - 1) as i32
                        })
                        .collect::<Vec<_>>();
                    let keys = (0..n)
                        .filter(|&i| type_ids[i] == 0)
                        .map(|i| all_keys[i])
                        .collect();
                    Arc::new(
                        UnionArray::try_new(
                            union_fields(),
                            type_ids.into(),
                            Some(offsets.into()),
                            vec![
                                dictionary(keys),
                                Arc::new(Int32Array::from(vec![1; counts[1]])),
                            ],
                        )
                        .unwrap(),
                    )
                }
                4 => {
                    // Run-end encoded values (a run per row) in a struct.
                    let runs = make_array(
                        ArrayDataBuilder::new(DataType::RunEndEncoded(
                            Arc::new(Field::new("run_ends", DataType::Int32, false)),
                            Arc::new(field("values")),
                        ))
                        .len(n)
                        .add_child_data(
                            Int32Array::from_iter_values(1..=n as i32).to_data(),
                        )
                        .add_child_data(dictionary(all_keys).to_data())
                        .build()
                        .unwrap(),
                    );
                    Arc::new(
                        StructArray::try_new(
                            vec![Field::new("r", runs.data_type().clone(), true)].into(),
                            vec![runs],
                            Some(parent_nulls),
                        )
                        .unwrap(),
                    )
                }
                _ => {
                    // A struct in a list in a struct.
                    let inner = StructArray::try_new(
                        vec![field("d")].into(),
                        vec![dictionary(masked_keys)],
                        Some(parent_nulls),
                    )
                    .unwrap();
                    let lists = ListArray::new(
                        Arc::new(Field::new("item", inner.data_type().clone(), true)),
                        OffsetBuffer::from_lengths(std::iter::repeat_n(1, n)),
                        Arc::new(inner),
                        None,
                    );
                    let outer_nulls = NullBuffer::from(
                        (0..n).map(|i| matters(&mut rng, i)).collect::<Vec<_>>(),
                    );
                    Arc::new(
                        StructArray::try_new(
                            vec![Field::new("l", lists.data_type().clone(), true)].into(),
                            vec![Arc::new(lists)],
                            Some(outer_nulls),
                        )
                        .unwrap(),
                    )
                }
            };
            let batch = RecordBatch::try_from_iter(vec![("c", array)]).unwrap();
            // All rows that matter, and some others, in random order.
            let mut selection = (0..n as u32)
                .filter(|&i| (i as usize) < size || rng.random_bool(0.7))
                .collect::<Vec<_>>();
            selection.shuffle(&mut rng);
            let rows_per_block = rng.random_range(1..50);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let layout = TupleDataLayout::try_new(batch.schema(), false)
                    .unwrap()
                    .with_rows_per_block(rows_per_block)
                    .unwrap();
                let mut collection = TupleDataCollection::new(Arc::new(layout));
                let chunk = chunk_with_selection(&batch, selection);
                let expected = chunk.compact().unwrap();
                let locations = collection.append_chunk(&chunk, None).unwrap();
                assert_identical(&collection.gather(&locations).unwrap(), &expected);
                let mut state = TupleDataScanState::new();
                let mut scanned = vec![];
                while let Some(chunk) = collection.scan_chunk(&mut state).unwrap() {
                    scanned.push(chunk.into_record_batch().unwrap());
                }
                let scanned =
                    arrow::compute::concat_batches(&batch.schema(), &scanned).unwrap();
                assert_identical(&scanned, &expected);
            }));
            assert!(
                result.is_ok(),
                "seed {seed}, case {case}, nullable {nullable}, size {size}"
            );
        }
    }

    #[test]
    fn dictionaries_with_null_keys_of_values_without_nulls() {
        use crate::tuple_data::PartitionedTupleData;

        for dictionary in
            crate::tuple_data::test_data::dictionaries_of_values_without_nulls()
        {
            let batch = RecordBatch::try_from_iter(vec![("d", dictionary)]).unwrap();
            for selection in [vec![0, 1, 2], vec![1, 2], vec![2, 0, 0]] {
                let chunk = chunk_with_selection(&batch, selection);
                let expected = chunk.compact().unwrap();
                let mut collection = collection_for(&batch, false);
                let locations = collection.append_chunk(&chunk, None).unwrap();
                assert_identical(&collection.gather(&locations).unwrap(), &expected);
                let mut state = TupleDataScanState::new();
                let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
                assert_eq!(scanned.schema(), &batch.schema());
                assert_identical(&scanned.into_record_batch().unwrap(), &expected);

                let layout =
                    Arc::new(TupleDataLayout::try_new(batch.schema(), true).unwrap());
                let mut partitioned = PartitionedTupleData::try_new(layout, 1).unwrap();
                let hashes = (0..chunk.len() as u64).map(|i| i << 63).collect::<Vec<_>>();
                let locations = partitioned
                    .append_chunk_with_locations(&chunk, &hashes)
                    .unwrap();
                for (row, location) in locations.iter().enumerate() {
                    let gathered = partitioned
                        .partition(location.partition)
                        .gather(&[location.location])
                        .unwrap();
                    assert_identical(&gathered, &expected.slice(row, 1));
                }
            }
        }
    }

    #[test]
    fn dictionaries_of_run_end_encoded_lists_are_not_expanded() {
        use crate::tuple_data::PartitionedTupleData;

        for dictionary in
            crate::tuple_data::test_data::dictionaries_of_run_end_encoded_lists()
        {
            let batch = RecordBatch::try_from_iter(vec![("d", dictionary)]).unwrap();
            let rows = batch.num_rows() as u32;
            for selection in [
                (0..rows).collect(),
                vec![1, 0],
                (0..rows).rev().step_by(3).collect::<Vec<_>>(),
            ] {
                let chunk = chunk_with_selection(&batch, selection);
                let expected = chunk.compact().unwrap();
                let mut collection = collection_for(&batch, false);
                let locations = collection.append_chunk(&chunk, None).unwrap();
                assert_identical(&collection.gather(&locations).unwrap(), &expected);
                let mut state = TupleDataScanState::new();
                let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
                assert_eq!(scanned.schema(), &batch.schema());
                assert_identical(&scanned.into_record_batch().unwrap(), &expected);

                // Into 2 partitions, by the lowest bit of the row index.
                let layout =
                    Arc::new(TupleDataLayout::try_new(batch.schema(), true).unwrap());
                let mut partitioned = PartitionedTupleData::try_new(layout, 1).unwrap();
                let hashes = (0..chunk.len() as u64).map(|i| i << 63).collect::<Vec<_>>();
                let locations = partitioned
                    .append_chunk_with_locations(&chunk, &hashes)
                    .unwrap();
                for partition in 0..2 {
                    let (rows, locations): (Vec<u32>, Vec<RowLocation>) = locations
                        .iter()
                        .enumerate()
                        .filter(|(_, location)| location.partition == partition)
                        .map(|(row, location)| (row as u32, location.location))
                        .unzip();
                    let gathered =
                        partitioned.partition(partition).gather(&locations).unwrap();
                    let expected =
                        chunk_with_selection(&expected, rows).compact().unwrap();
                    assert_identical(&gathered, &expected);
                }
            }

            // Slices with fewer rows than dictionary values (some keys null),
            // so that only the values that they refer to are encoded.
            for (offset, len) in [(0, 1), (1, 2), (5, 2)] {
                let slice = batch.slice(offset, len);
                let mut collection = collection_for(&batch, false);
                let chunk = DataChunk::from_record_batch(slice.clone());
                let locations = collection.append_chunk(&chunk, None).unwrap();
                assert_identical(&collection.gather(&locations).unwrap(), &slice);
            }

            // Appended in chunks of 128 rows, gathered and scanned at once.
            let mut collection = collection_for(&batch, false);
            let mut locations = vec![];
            for start in (0..batch.num_rows()).step_by(128) {
                let chunk = DataChunk::from_record_batch(batch.slice(start, 128));
                locations.extend(collection.append_chunk(&chunk, None).unwrap());
            }
            assert_identical(&collection.gather(&locations).unwrap(), &batch);
            let mut state = TupleDataScanState::new();
            let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
            assert_identical(&scanned.into_record_batch().unwrap(), &batch);
        }
    }

    #[test]
    fn dictionaries_of_nested_values() {
        use arrow::array::{
            DictionaryArray, Int8Array, ListArray, StructArray, UnionArray,
        };
        use arrow::datatypes::{Int8Type, Int32Type, UnionFields};

        let union = UnionArray::try_new(
            UnionFields::try_new(
                [0, 1],
                [
                    Field::new("i", DataType::Int32, true),
                    Field::new("s", DataType::Utf8, true),
                ],
            )
            .unwrap(),
            vec![0, 1, 0].into(),
            None,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, None])),
                Arc::new(StringArray::from(vec![None, Some("x"), None])),
            ],
        )
        .unwrap();
        let values: [(&str, ArrayRef); 4] = [
            (
                "run-end encoded values",
                custom_run_end_encoded(
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                    &[1, 2, 3],
                    true,
                ),
            ),
            ("union values", Arc::new(union)),
            (
                "struct values",
                Arc::new(
                    StructArray::try_from(vec![(
                        "a",
                        Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))
                            as ArrayRef,
                    )])
                    .unwrap(),
                ),
            ),
            (
                "list values",
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1)]),
                    None,
                    Some(vec![]),
                ])),
            ),
        ];
        for (name, values) in values {
            let keys =
                Int8Array::from(vec![Some(0), None, Some(1), Some(0), Some(2), Some(1)]);
            let dictionary = DictionaryArray::<Int8Type>::try_new(keys, values).unwrap();
            let batch = RecordBatch::try_from_iter(vec![(
                name,
                Arc::new(dictionary) as ArrayRef,
            )])
            .unwrap();
            let mut collection = collection_for(&batch, false);
            let chunk = chunk_with_selection(&batch, vec![4, 0, 5, 1, 1, 3, 2]);
            let locations = collection.append_chunk(&chunk, None).unwrap();
            assert_identical(
                &collection.gather(&locations).unwrap(),
                &chunk.compact().unwrap(),
            );
        }
    }

    #[test]
    fn unions_of_dictionaries() {
        use arrow::array::{DictionaryArray, Int8Array, UnionArray};
        use arrow::datatypes::{Int8Type, UnionFields};

        // The row format decodes the dictionaries of the children of unions to
        // their values but keeps the type of the union: the dictionaries are
        // restored.
        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![Some(0), None, Some(1), Some(1), Some(2)]),
                Arc::new(StringArray::from(vec![Some("x"), None, Some("y")])),
            )
            .unwrap(),
        );
        let fields = UnionFields::try_new(
            [0, 1],
            [
                Field::new("d", dictionary.data_type().clone(), true),
                Field::new("i", DataType::Int32, true),
            ],
        )
        .unwrap();
        let ints: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(2),
            None,
            Some(3),
        ]));
        let type_ids = [0, 1, 0, 0, 1].into_iter().collect::<ScalarBuffer<i8>>();
        let sparse = UnionArray::try_new(
            fields.clone(),
            type_ids.clone(),
            None,
            vec![Arc::clone(&dictionary), Arc::clone(&ints)],
        )
        .unwrap();
        let dense = UnionArray::try_new(
            fields,
            type_ids,
            Some(vec![0, 0, 1, 2, 1].into()),
            vec![dictionary.slice(0, 3), ints.slice(0, 2)],
        )
        .unwrap();
        let batch = RecordBatch::try_from_iter(vec![
            ("sparse", Arc::new(sparse) as ArrayRef),
            ("dense", Arc::new(dense) as ArrayRef),
        ])
        .unwrap();
        let mut collection = collection_for(&batch, false);
        let chunk = chunk_with_selection(&batch, vec![4, 0, 3, 2, 1, 1]);
        let locations = collection.append_chunk(&chunk, None).unwrap();
        assert_identical(
            &collection.gather(&locations).unwrap(),
            &chunk.compact().unwrap(),
        );
    }

    #[test]
    fn zero_width_and_wide_fixed_size_binaries() {
        use arrow::array::FixedSizeBinaryArray;

        let empty = FixedSizeBinaryArray::try_new(
            0,
            Buffer::from_vec(Vec::<u8>::new()),
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap();
        let wide =
            FixedSizeBinaryArray::try_from_iter((0..3_u8).map(|i| vec![i; 100_000]))
                .unwrap();
        let batch = RecordBatch::try_from_iter(vec![
            ("empty", Arc::new(empty) as ArrayRef),
            ("wide", Arc::new(wide) as ArrayRef),
        ])
        .unwrap();
        let mut collection = collection_for(&batch, false);
        assert_eq!(collection.layout().rows_per_block(), 2);
        let locations = collection
            .append_chunk(&DataChunk::from_record_batch(batch.clone()), None)
            .unwrap();
        assert_eq!(collection.num_blocks(), 2);
        assert_identical(&collection.gather(&locations).unwrap(), &batch);
    }

    #[test]
    fn collections_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TupleDataLayout>();
        assert_send_sync::<TupleDataCollection>();
        assert_send_sync::<TupleDataScanState>();
    }
}
