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
//!
//! # Fast paths
//!
//! * A gather resolves its locations to the addresses of their rows once
//!   (`RowPtr`), for all gathered columns: every column is then read in a
//!   tight loop without block lookups or bounds checks. Fixed-width values are
//!   read with their native type, in the same pass as their validity bits,
//!   and a validity buffer is only built if a gathered value is null. The
//!   slot of a variable-length value of at most [`MAX_INLINE_LEN`] bytes is
//!   its Arrow view, so `Utf8View` and `BinaryView` values are gathered by
//!   copying their slots, and the gathered string and binary arrays are not
//!   validated again (they hold copies of the appended values).
//! * A scatter clears the validity bits of null values in the same pass as it
//!   stores the values, without branching on their validity, and builds the
//!   slots of variable-length values from the views of view arrays and from
//!   fixed size reads of the bytes of arrays with offsets.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayDataBuilder, ArrayRef, AsArray, BooleanArray, GenericByteArray,
    GenericByteViewArray, RecordBatch, RecordBatchOptions, make_array, new_null_array,
};
use arrow::buffer::{
    BooleanBuffer, Buffer, MutableBuffer, NullBuffer, OffsetBuffer, ScalarBuffer,
};
use arrow::datatypes::{
    ArrowNativeType, BinaryType, BinaryViewType, ByteArrayType, ByteViewType, DataType,
    LargeBinaryType, LargeUtf8Type, SchemaRef, StringViewType, Utf8Type, i256,
};
use arrow_row::Rows;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::{
    DataFusionError, Result, exec_datafusion_err, exec_err, internal_datafusion_err,
    internal_err, resources_datafusion_err,
};

use super::layout::{
    ColumnLayout, ColumnStorage, HASH_WIDTH, MAX_INLINE_LEN, MAX_ROWS_PER_BLOCK,
    RowCodec, TupleDataLayout, VAR_LEN_SLOT_WIDTH, has_run_end_encoded, new_dictionary,
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

/// The address of a stored row, resolved once from its [`RowLocation`] by
/// [`TupleDataCollection::resolve_rows`] (DuckDB's row pointers): the values
/// of the row are then read in tight loops, without looking up its block or
/// checking bounds for every value.
///
/// A `RowPtr<'a>` points to the first byte of a row of
/// [`TupleDataLayout::row_width`] bytes of a collection that is borrowed for
/// `'a`, so that the row can neither be changed nor freed while the pointer
/// exists. Its accessors are `unsafe`: their callers guarantee that they read
/// within the row, i.e. within the slots and the validity bits of the columns
/// of the layout of the row, see [`check_column`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowPtr<'a> {
    ptr: *const u8,
    row: PhantomData<&'a [u8]>,
}

impl<'a> RowPtr<'a> {
    /// The `W` bytes at `offset`.
    ///
    /// # Safety
    ///
    /// `offset + W` must not exceed the width of the row.
    #[inline(always)]
    pub(crate) unsafe fn read<const W: usize>(self, offset: usize) -> [u8; W] {
        // SAFETY: the bytes are in the row (guaranteed by the caller), which
        // lives and does not change for `'a`; `[u8; W]` has an alignment of 1.
        unsafe { self.ptr.add(offset).cast::<[u8; W]>().read() }
    }

    /// The byte at `offset`.
    ///
    /// # Safety
    ///
    /// `offset` must be less than the width of the row.
    #[inline(always)]
    pub(crate) unsafe fn byte(self, offset: usize) -> u8 {
        // SAFETY: guaranteed by the caller.
        unsafe { *self.ptr.add(offset) }
    }

    /// The value of type `T` stored at `offset`, in its Arrow (native)
    /// representation.
    ///
    /// # Safety
    ///
    /// `offset + size_of::<T>()` must not exceed the width of the row.
    #[inline(always)]
    unsafe fn read_native<T: ArrowNativeType>(self, offset: usize) -> T {
        // SAFETY: the bytes are in the row (guaranteed by the caller), and
        // any bytes are a valid value of the (sealed) native types, which are
        // integers, floats and structs of them.
        unsafe { self.ptr.add(offset).cast::<T>().read_unaligned() }
    }

    /// The `len` bytes at `offset`.
    ///
    /// # Safety
    ///
    /// `offset + len` must not exceed the width of the row.
    #[inline(always)]
    pub(crate) unsafe fn slice(self, offset: usize, len: usize) -> &'a [u8] {
        // SAFETY: the bytes are in the row (guaranteed by the caller), which
        // lives and does not change for `'a`.
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) }
    }

    /// Returns `true` if bit `bit` of the validity bytes of the row is set,
    /// e.g. if the value of column `bit` is valid.
    ///
    /// # Safety
    ///
    /// `bit / 8` must be less than the number of validity bytes of the row.
    #[inline(always)]
    pub(crate) unsafe fn is_valid(self, bit: usize) -> bool {
        // SAFETY: guaranteed by the caller.
        unsafe { self.byte(bit >> 3) & (1 << (bit & 7)) != 0 }
    }

    /// The 16 bytes of the slot at `offset` of a variable-length value, see
    /// [`TupleDataLayout`]: the length in the lower 32 bits, followed by the
    /// zero padded bytes of a value of at most [`MAX_INLINE_LEN`] bytes, or
    /// by the prefix and the heap offset of a longer value.
    ///
    /// # Safety
    ///
    /// `offset + VAR_LEN_SLOT_WIDTH` must not exceed the width of the row.
    #[inline(always)]
    pub(crate) unsafe fn var_len_slot(self, offset: usize) -> u128 {
        // SAFETY: guaranteed by the caller.
        u128::from_le_bytes(unsafe { self.read::<VAR_LEN_SLOT_WIDTH>(offset) })
    }

    /// The bytes of the variable-length value whose slot is at `offset`: in
    /// the slot, or in `heap()`, the heap of the block of the row.
    ///
    /// # Safety
    ///
    /// `offset + VAR_LEN_SLOT_WIDTH` must not exceed the width of the row.
    #[inline(always)]
    pub(crate) unsafe fn var_len(
        self,
        offset: usize,
        heap: impl FnOnce() -> &'a [u8],
    ) -> &'a [u8] {
        // SAFETY: guaranteed by the caller.
        let slot = unsafe { self.var_len_slot(offset) };
        let len = slot as u32 as usize;
        if len <= MAX_INLINE_LEN {
            // SAFETY: the inline bytes are in the slot, in the row.
            unsafe { self.slice(offset + 4, len) }
        } else {
            let start = heap_offset(slot);
            &heap()[start..start + len]
        }
    }
}

/// The heap offset in the slot `slot` of a variable-length value of more than
/// [`MAX_INLINE_LEN`] bytes, see [`RowPtr::var_len_slot`].
#[inline(always)]
pub(crate) fn heap_offset(slot: u128) -> usize {
    (slot >> 64) as u64 as usize
}

/// Check that the slot and the validity bits of `column`, the column `index`
/// of a layout, are in the rows of `layout`, so that they can be read from the
/// [`RowPtr`]s of rows with this layout. This is always the case for the
/// columns of `layout` and of [compatible](TupleDataLayout::is_compatible)
/// layouts, which have the same slots.
pub(crate) fn check_column(
    layout: &TupleDataLayout,
    column: &ColumnLayout,
    index: usize,
) -> Result<()> {
    let validity_bits = 8 * layout.validity_bytes();
    let in_rows = column
        .offset()
        .checked_add(column.width())
        .is_some_and(|end| end <= layout.row_width())
        && index < validity_bits
        && column
            .key_validity_bit()
            .is_none_or(|bit| bit < validity_bits);
    if !in_rows {
        return internal_err!("Column {index} is not in the rows of the TupleDataLayout");
    }
    Ok(())
}

/// The validity bits of a [`NullBuffer`], to test the validity of many rows
/// in a loop: unlike with [`NullBuffer::is_valid`], the bytes and the bit
/// offset stay in registers.
#[derive(Clone, Copy)]
pub(crate) struct ValidityBits<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ValidityBits<'a> {
    pub(crate) fn new(nulls: &'a NullBuffer) -> Self {
        Self {
            bytes: nulls.inner().values(),
            offset: nulls.offset(),
        }
    }

    /// `1` if row `i` is valid, `0` if it is null.
    #[inline(always)]
    pub(crate) fn bit(self, i: usize) -> u8 {
        let i = i + self.offset;
        (self.bytes[i >> 3] >> (i & 7)) & 1
    }

    /// Returns `true` if row `i` is valid.
    #[inline(always)]
    pub(crate) fn is_valid(self, i: usize) -> bool {
        self.bit(i) != 0
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
    /// the order of the selection; the chunk is not compacted, except for
    /// the columns with run-end encoded values, which are taken like
    /// [`DataChunk::compact`] takes them (Arrow's `take` merges consecutive
    /// runs of equal values, e.g. of nulls of different union variants). `hashes`
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
        let logical = source.logical_rows(rows.len())?;
        self.append_rows(
            &source,
            &rows,
            logical.as_deref(),
            hashes,
            Some(&mut locations),
        )?;
        Ok(locations)
    }

    /// Append the physical rows `rows` of `source` (with the hashes
    /// `hashes`, one for every entry of `rows`), and push their locations
    /// to `locations`. `logical` holds the logical rows of the entries of
    /// `rows`, which are only needed if
    /// [`ScatterSource::has_logical_columns`].
    pub(crate) fn append_rows(
        &mut self,
        source: &ScatterSource,
        rows: &[u32],
        logical: Option<&[u32]>,
        hashes: Option<&[u64]>,
        mut locations: Option<&mut Vec<RowLocation>>,
    ) -> Result<()> {
        if source.has_logical_columns()
            && logical.is_none_or(|logical| logical.len() != rows.len())
        {
            return internal_err!("The logical rows of the scattered rows are missing");
        }
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
                logical.map(|logical| &logical[segment.clone()]),
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

    /// The bytes of the row at `location` (of a row with `row_width` bytes),
    /// `None` if the location is invalid.
    #[inline(always)]
    fn try_row(&self, location: RowLocation, row_width: usize) -> Option<&[u8]> {
        let block = self.blocks.get(location.block())?;
        if location.row() >= block.count {
            return None;
        }
        let start = location.row() * row_width;
        block.rows.get(start..start + row_width)
    }

    /// Resolve `location` (of a row with `row_width` bytes) to its row, `None`
    /// if the location is invalid.
    #[inline(always)]
    fn try_row_ref(&self, location: RowLocation, row_width: usize) -> Option<RowRef<'_>> {
        Some(RowRef {
            row: self.try_row(location, row_width)?,
            heap: &self.blocks[location.block()].heap,
        })
    }

    /// The heap of the block of the row at `location`, which must be a valid
    /// location (e.g. of a row resolved by [`Self::resolve_rows`]).
    #[inline]
    pub(crate) fn heap(&self, location: RowLocation) -> &[u8] {
        &self.blocks[location.block()].heap
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

    /// Resolve `locations` to the addresses of their rows, checking every
    /// location once: the values of the rows can then be read without
    /// looking up their blocks again, see [`RowPtr`].
    ///
    /// The locations are resolved in a single tight pass, which checks all of
    /// them before the first invalid one is reported.
    #[inline]
    pub(crate) fn resolve_rows(
        &self,
        locations: impl Iterator<Item = RowLocation>,
    ) -> Result<Vec<RowPtr<'_>>> {
        let row_width = self.layout.row_width();
        debug_assert!(
            self.blocks
                .iter()
                .all(|block| block.rows.len() == block.count * row_width)
        );
        let blocks = self.blocks.as_slice();
        let mut invalid = None;
        let rows = locations
            .map(|location| {
                let row = location.row();
                match blocks.get(location.block()) {
                    // The rows of a block are `count * row_width` bytes, so
                    // that the row is in the rows of the block.
                    Some(block) if row < block.count => RowPtr {
                        ptr: block.rows.as_ptr().wrapping_add(row * row_width),
                        row: PhantomData,
                    },
                    _ => {
                        invalid.get_or_insert(location);
                        RowPtr {
                            ptr: std::ptr::null(),
                            row: PhantomData,
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        match invalid {
            Some(location) => Err(self.invalid_location(location)),
            None => Ok(rows),
        }
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
        // One block lookup per row, for all columns.
        let rows = self.resolve_rows(locations.iter().copied())?;
        let rows = GatherRows {
            collection: self,
            locations,
            rows: &rows,
        };
        column_ids
            .iter()
            .map(|&id| gather_column(self.layout.column(id), id, rows))
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
    /// Rows are returned in the order of [`Self::row_locations`]. A chunk has
    /// fewer rows if the values of more rows do not fit their types together
    /// (and so do the following chunks of the scan), e.g. if they would need
    /// more dictionary keys than the key type has, more run-end encoded values
    /// than the run end type counts, or larger offsets than the offset type
    /// holds.
    pub fn scan_chunk(
        &self,
        state: &mut TupleDataScanState,
    ) -> Result<Option<DataChunk>> {
        state.locations.clear();
        let end = state.blocks.end.min(self.blocks.len());
        while state.locations.len() < state.max_rows && state.block < end {
            let block = &self.blocks[state.block];
            let n = (block.count - state.row).min(state.max_rows - state.locations.len());
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
        let column_ids = match &state.column_ids {
            Some(column_ids) => Cow::Borrowed(column_ids.as_slice()),
            None => Cow::Owned((0..self.layout.num_columns()).collect()),
        };
        // Halve the chunk until its values can be gathered together, and
        // the following chunks too.
        let mut len = state.locations.len();
        let columns = loop {
            match self.gather_columns(&state.locations[..len], &column_ids) {
                Ok(columns) => break columns,
                Err(e) if len == 1 => return Err(e),
                Err(_) => {
                    len /= 2;
                    state.max_rows = len;
                }
            }
        };
        if len < state.locations.len() {
            // Continue the scan after the returned rows.
            let next = state.locations[len];
            (state.block, state.row) = (next.block(), next.row());
            state.locations.truncate(len);
        }
        let batch = record_batch(schema, columns, len)?;
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
    /// The maximum number of rows of the next chunk: `chunk_size`, or fewer
    /// once the values of more rows did not fit their types together.
    max_rows: usize,
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
            max_rows: DEFAULT_VECTOR_SIZE,
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
        self.max_rows = self.chunk_size;
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
/// The columns are indexed by *physical* row, except the columns with
/// run-end encoded values of chunks with a selection, which are indexed by
/// *logical* row (see [`SourceColumn::logical`]).
pub(crate) struct ScatterSource {
    columns: Vec<SourceColumn>,
}

struct SourceColumn {
    /// Whether the column holds the values of the logical rows of the chunk
    /// (indexed by logical row): Arrow's `take` merges consecutive runs of
    /// equal values of run-end encoded arrays (e.g. nulls of different union
    /// variants) into one run with the value of its last row, so the values
    /// of selected rows are taken like the chunk is compacted.
    logical: bool,
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
            .enumerate()
            .map(|(i, (column, array))| {
                if chunk.selection().is_some() && has_run_end_encoded(column.data_type())
                {
                    let mut source =
                        SourceColumn::try_new(column, &chunk.compacted_column(i)?)?;
                    source.logical = true;
                    Ok(source)
                } else {
                    SourceColumn::try_new(column, array)
                }
            })
            .collect::<Result<_>>()?;
        Ok(Self { columns })
    }

    /// Returns `true` if some columns are indexed by logical row, see
    /// [`SourceColumn::logical`].
    pub(crate) fn has_logical_columns(&self) -> bool {
        self.columns.iter().any(|column| column.logical)
    }

    /// The logical rows `0..len` if some columns are indexed by logical row.
    pub(crate) fn logical_rows(&self, len: usize) -> Result<Option<Vec<u32>>> {
        if !self.has_logical_columns() {
            return Ok(None);
        }
        let len = u32::try_from(len).map_err(|_| {
            exec_datafusion_err!("DataChunk with {len} rows is too large")
        })?;
        Ok(Some((0..len).collect()))
    }
}

impl SourceColumn {
    fn try_new(column: &ColumnLayout, array: &ArrayRef) -> Result<Self> {
        let (values, mut dictionary, key_nulls) = if column.is_dictionary() {
            // The nulls of a dictionary array are the nulls of its keys.
            let key_nulls = array.nulls().filter(|n| n.null_count() > 0).cloned();
            let dictionary = DictionaryValues::try_new(array)?;
            (Arc::clone(&dictionary.values), Some(dictionary), key_nulls)
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
                SourceData::Encoded(match &mut dictionary {
                    // Only the values that the rows refer to.
                    Some(dictionary) => dictionary.encode(codec, array)?,
                    None => codec.converter.convert_columns(&[Arc::clone(&values)])?,
                })
            }
        };
        let indices = dictionary.map(|dictionary| dictionary.indices);
        Ok(Self {
            logical: false,
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
    /// The values of the rows of the dictionary array `array`.
    pub(crate) fn try_new(array: &ArrayRef) -> Result<Self> {
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
        Ok(Self {
            values: Arc::clone(values),
            // In bounds also for the null keys, whose key values can be any.
            indices: dictionary.normalized_keys(),
        })
    }

    /// Encode the values with `codec`, and make `indices` the indices of the
    /// encodings of the values of the rows (of the dictionary array `array`).
    ///
    /// If the dictionary has more values than rows, only the values that rows
    /// with a valid key refer to are encoded, in order, slice by slice of
    /// consecutive values: an array of these values (from Arrow's `take`, or
    /// concatenated slices) could differ from them (`take` merges the runs
    /// of equal run-end encoded values into runs of the value of their last
    /// row, see `SourceColumn::logical`, which changes equal values that are
    /// not identical, e.g. nulls of different union variants) or be too large
    /// for their type (concatenated slices of run-end encoded values have a
    /// value for every slice, e.g. a list each, whose run ends can overflow).
    pub(crate) fn encode(&mut self, codec: &RowCodec, array: &ArrayRef) -> Result<Rows> {
        let Self { values, indices } = self;
        if values.len() <= indices.len() {
            return Ok(codec.converter.convert_columns(&[Arc::clone(values)])?);
        }
        let keys = array.as_any_dictionary().keys();
        let mut used = (0..indices.len())
            .filter(|&row| keys.is_valid(row))
            .map(|row| indices[row])
            .collect::<Vec<_>>();
        used.sort_unstable();
        used.dedup();
        if used.is_empty() {
            used.push(0);
        }
        let mut encoded = codec.converter.empty_rows(used.len(), 0);
        let mut start = 0;
        for end in 1..=used.len() {
            if end == used.len() || used[end] != used[end - 1] + 1 {
                let first = used[start];
                let slice = values.slice(first, used[end - 1] + 1 - first);
                codec.converter.append(&mut encoded, &[slice])?;
                start = end;
            }
        }
        for (row, index) in indices.iter_mut().enumerate() {
            *index = match keys.is_valid(row) {
                // The position of the value among the used values.
                true => used.binary_search(index).map_err(|_| {
                    internal_datafusion_err!("Dictionary value {index} is not used")
                })?,
                false => 0,
            };
        }
        Ok(encoded)
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

/// Write the (physical) rows `rows` of `source` into `target`, the bytes of
/// `rows.len()` new, zeroed rows. `logical` holds the logical rows of the
/// entries of `rows`, for the columns indexed by logical row (see
/// [`SourceColumn::logical`]).
fn scatter_rows(
    layout: &TupleDataLayout,
    source: &ScatterSource,
    rows: &[u32],
    logical: Option<&[u32]>,
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
            bit: index,
        };
        let rows = match (source.logical, logical) {
            (false, _) => rows,
            (true, Some(logical)) => logical,
            (true, None) => {
                return internal_err!(
                    "The logical rows of the scattered rows are missing"
                );
            }
        };
        // The scatters of booleans, fixed-width and variable-length values
        // clear the validity bits of null values themselves.
        let scatter_validity = matches!(
            source.data,
            SourceData::Boolean(_) | SourceData::Fixed(_) | SourceData::VarLen
        );
        if let (Some(nulls), false) = (&source.nulls, scatter_validity) {
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
            let (offsets, data) = (values.value_offsets(), values.value_data());
            target.scatter_offsets(offsets, data, nulls, rows, index, heap)
        }
        DataType::LargeUtf8 => {
            let values = values.as_string::<i64>();
            let (offsets, data) = (values.value_offsets(), values.value_data());
            target.scatter_offsets(offsets, data, nulls, rows, index, heap)
        }
        DataType::Binary => {
            let values = values.as_binary::<i32>();
            let (offsets, data) = (values.value_offsets(), values.value_data());
            target.scatter_offsets(offsets, data, nulls, rows, index, heap)
        }
        DataType::LargeBinary => {
            let values = values.as_binary::<i64>();
            let (offsets, data) = (values.value_offsets(), values.value_data());
            target.scatter_offsets(offsets, data, nulls, rows, index, heap)
        }
        DataType::Utf8View => {
            let values = values.as_string_view();
            let (views, buffers) = (values.views(), values.data_buffers());
            target.scatter_views(views, buffers, nulls, rows, index, heap)
        }
        DataType::BinaryView => {
            let values = values.as_binary_view();
            let (views, buffers) = (values.views(), values.data_buffers());
            target.scatter_views(views, buffers, nulls, rows, index, heap)
        }
        data_type => internal_err!("Unexpected variable-length data type {data_type}"),
    }
}

/// The slot of a variable-length value of at most [`MAX_INLINE_LEN`] bytes:
/// its length `len` and its bytes, zero padded, where `bytes` holds the bytes
/// (and possibly other bytes after them) in its bytes 4 to 15. This is also
/// the view of the value in Arrow's view arrays.
#[inline(always)]
fn inline_slot(len: usize, bytes: u128) -> u128 {
    debug_assert!(len <= MAX_INLINE_LEN);
    // The bits of the length and of the bytes of the value.
    let mask = u128::MAX >> (8 * (MAX_INLINE_LEN - len));
    (bytes & mask & !u128::from(u32::MAX)) | len as u128
}

/// The slot of a variable-length value of more than [`MAX_INLINE_LEN`] bytes
/// whose view (or length and prefix, in the lower 64 bits of `view`) is
/// `view`, stored at `heap_offset` in the heap.
#[inline(always)]
fn heap_slot(view: u128, heap_offset: usize) -> u128 {
    (view & u128::from(u64::MAX)) | ((heap_offset as u128) << 64)
}

/// The view of the value at `range` of `data` (the bytes of the values of a
/// string or binary array with offsets), laid out like the view of an Arrow
/// view array and like the slot of the value: the length and the zero padded
/// bytes of a value of at most [`MAX_INLINE_LEN`] bytes (from a fixed size
/// read of its bytes unless they are at the end of `data`, instead of a copy
/// of their length), and the length (truncated to 32 bits) and the prefix of
/// a longer value, with a buffer index and offset of `0`.
#[inline(always)]
pub(crate) fn offset_value_view(data: &[u8], range: Range<usize>) -> u128 {
    let len = range.len();
    if len <= MAX_INLINE_LEN {
        let bytes = match data.get(range.start..range.start + MAX_INLINE_LEN) {
            Some(inline) => {
                // Two loads, not a copy into a buffer that is read back.
                let (low, high) = inline.split_at(8);
                let low = u64::from_le_bytes(low.try_into().unwrap());
                let high = u32::from_le_bytes(high.try_into().unwrap());
                (u128::from(low) << 32) | (u128::from(high) << 96)
            }
            None => {
                let mut bytes = [0; VAR_LEN_SLOT_WIDTH];
                bytes[4..4 + len].copy_from_slice(&data[range]);
                u128::from_le_bytes(bytes)
            }
        };
        inline_slot(len, bytes)
    } else {
        let mut prefix = [0; 4];
        prefix.copy_from_slice(&data[range.start..range.start + 4]);
        u128::from(len as u32) | (u128::from(u32::from_le_bytes(prefix)) << 32)
    }
}

/// The length of a value of `len` bytes, if it fits in a slot.
#[inline]
fn slot_len(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        exec_datafusion_err!(
            "A value of {len} bytes is too large for a TupleDataCollection"
        )
    })
}

/// The slots of one column in a range of new rows.
struct ScatterTarget<'a> {
    rows: &'a mut [u8],
    row_width: usize,
    /// The offset of the slot of the column in a row.
    offset: usize,
    /// The validity bit of the column, which the scatters of booleans,
    /// fixed-width and variable-length values clear for null values, in the
    /// same pass as they store the values.
    bit: usize,
}

/// Clear the validity bit `bit` of `row` if `valid` is `0` (it is `0` or
/// `1`), without branching on the validity, which is unpredictable.
#[inline(always)]
fn set_validity(row: &mut [u8], bit: usize, valid: u8) {
    row[bit >> 3] &= !(1 << (bit & 7)) | (valid << (bit & 7));
}

impl ScatterTarget<'_> {
    fn clear_validity(&mut self, column: usize, nulls: &NullBuffer, rows: &[u32]) {
        let nulls = ValidityBits::new(nulls);
        for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
            set_validity(row, column, nulls.bit(i as usize));
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
        let (offset, bit) = (self.offset, self.bit);
        let rows_and_indices = self.rows.chunks_exact_mut(self.row_width).zip(rows);
        match nulls.map(ValidityBits::new) {
            None => {
                for (row, &i) in rows_and_indices {
                    row[offset] = u8::from(values.value(index(i as usize)));
                }
            }
            Some(nulls) => {
                for (row, &i) in rows_and_indices {
                    let i = i as usize;
                    let valid = nulls.bit(i);
                    set_validity(row, bit, valid);
                    row[offset] = valid & u8::from(values.value(index(i)));
                }
            }
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
            0 => {
                if let Some(nulls) = nulls {
                    self.clear_validity(self.bit, nulls, rows);
                }
            }
            1 => self.scatter_fixed_width::<1>(values, nulls, rows, index),
            2 => self.scatter_fixed_width::<2>(values, nulls, rows, index),
            4 => self.scatter_fixed_width::<4>(values, nulls, rows, index),
            8 => self.scatter_fixed_width::<8>(values, nulls, rows, index),
            16 => self.scatter_fixed_width::<16>(values, nulls, rows, index),
            32 => self.scatter_fixed_width::<32>(values, nulls, rows, index),
            width => {
                let (offset, bit) = (self.offset, self.bit);
                let nulls = nulls.map(ValidityBits::new);
                for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
                    let i = i as usize;
                    if nulls.is_none_or(|n| n.is_valid(i)) {
                        let v = index(i);
                        row[offset..offset + width]
                            .copy_from_slice(&values[v * width..(v + 1) * width]);
                    } else {
                        set_validity(row, bit, 0);
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
        let (values, _) = values.as_chunks::<W>();
        let rows_and_indices = self.rows.chunks_exact_mut(self.row_width).zip(rows);
        match nulls {
            None => {
                for (row, &i) in rows_and_indices {
                    row[offset..offset + W].copy_from_slice(&values[index(i as usize)]);
                }
            }
            Some(nulls) => {
                let (bit, nulls) = (self.bit, ValidityBits::new(nulls));
                for (row, &i) in rows_and_indices {
                    let i = i as usize;
                    let valid = nulls.bit(i);
                    set_validity(row, bit, valid);
                    // The bytes of null values are zero. A select instead of a
                    // branch on the validity, which is unpredictable (`index`
                    // is in bounds for null rows too).
                    let value = values[index(i)];
                    let value = if valid != 0 { value } else { [0; W] };
                    row[offset..offset + W].copy_from_slice(&value);
                }
            }
        }
    }

    /// Store the strings or binaries with offsets `offsets` into `data` of
    /// the rows `rows` (the value of row `i` is value `index(i)`), except for
    /// null rows, which keep length `0`.
    ///
    /// The slots are the views of the values (see [`offset_value_view`]).
    fn scatter_offsets<O: ArrowNativeType>(
        &mut self,
        offsets: &[O],
        data: &[u8],
        nulls: Option<&NullBuffer>,
        rows: &[u32],
        index: impl Fn(usize) -> usize + Copy,
        heap: &mut Vec<u8>,
    ) -> Result<()> {
        let nulls = nulls.map(ValidityBits::new);
        let is_valid = |i: usize| nulls.is_none_or(|n| n.is_valid(i));
        let range = |i: usize| {
            let v = index(i);
            offsets[v].as_usize()..offsets[v + 1].as_usize()
        };
        let heap_bytes = rows
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| is_valid(i))
            .map(|i| range(i).len())
            .filter(|&len| len > MAX_INLINE_LEN)
            .sum::<usize>();
        reserve_heap(heap, heap_bytes)?;
        let offset = self.offset;
        let bit = self.bit;
        for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
            let i = i as usize;
            if !is_valid(i) {
                set_validity(row, bit, 0);
                continue;
            }
            let range = range(i);
            let len = range.len();
            let view = offset_value_view(data, range.clone());
            let slot = if len <= MAX_INLINE_LEN {
                view
            } else {
                slot_len(len)?;
                let slot = heap_slot(view, heap.len());
                heap.extend_from_slice(&data[range]);
                slot
            };
            row[offset..offset + VAR_LEN_SLOT_WIDTH].copy_from_slice(&slot.to_le_bytes());
        }
        Ok(())
    }

    /// Store the values of the view array with the views `views` and the
    /// data buffers `buffers` of the rows `rows` (the value of row `i` is
    /// value `index(i)`), except for null rows, which keep length `0`.
    ///
    /// The slot of a value of at most [`MAX_INLINE_LEN`] bytes is its view
    /// (with its padding cleared), the slot of a longer value has its length
    /// and its prefix, which are read from its bytes as they are copied (not
    /// from its view).
    fn scatter_views(
        &mut self,
        views: &[u128],
        buffers: &[Buffer],
        nulls: Option<&NullBuffer>,
        rows: &[u32],
        index: impl Fn(usize) -> usize + Copy,
        heap: &mut Vec<u8>,
    ) -> Result<()> {
        let nulls = nulls.map(ValidityBits::new);
        let is_valid = |i: usize| nulls.is_none_or(|n| n.is_valid(i));
        let heap_bytes = rows
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| is_valid(i))
            .map(|i| views[index(i)] as u32 as usize)
            .filter(|&len| len > MAX_INLINE_LEN)
            .sum::<usize>();
        reserve_heap(heap, heap_bytes)?;
        let offset = self.offset;
        let bit = self.bit;
        for (row, &i) in self.rows.chunks_exact_mut(self.row_width).zip(rows) {
            let i = i as usize;
            if !is_valid(i) {
                set_validity(row, bit, 0);
                continue;
            }
            let view = views[index(i)];
            let len = view as u32 as usize;
            let slot = if len <= MAX_INLINE_LEN {
                inline_slot(len, view)
            } else {
                let start = (view >> 96) as usize;
                let buffer = &buffers[(view >> 64) as u32 as usize];
                let value = &buffer[start..start + len];
                let slot = heap_slot(offset_value_view(value, 0..len), heap.len());
                heap.extend_from_slice(value);
                slot
            };
            row[offset..offset + VAR_LEN_SLOT_WIDTH].copy_from_slice(&slot.to_le_bytes());
        }
        Ok(())
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
        let nulls = nulls.map(ValidityBits::new);
        let is_valid = |i: usize| nulls.is_none_or(|n| n.is_valid(i));
        let heap_bytes = rows
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| is_valid(i))
            .map(|i| value(i).len())
            .filter(|&len| len > MAX_INLINE_LEN)
            .sum::<usize>();
        reserve_heap(heap, heap_bytes)?;
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

/// Reserve `heap_bytes` more bytes in `heap`.
fn reserve_heap(heap: &mut Vec<u8>, heap_bytes: usize) -> Result<()> {
    heap.try_reserve(heap_bytes).map_err(|e| {
        resources_datafusion_err!(
            "Failed to allocate {heap_bytes} bytes for a TupleDataCollection heap: {e}"
        )
    })
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

/// The rows of a gather: the rows at `locations` of `collection`, resolved to
/// `rows` by [`TupleDataCollection::resolve_rows`].
#[derive(Clone, Copy)]
pub(crate) struct GatherRows<'a> {
    collection: &'a TupleDataCollection,
    locations: &'a [RowLocation],
    rows: &'a [RowPtr<'a>],
}

impl<'a> GatherRows<'a> {
    fn len(&self) -> usize {
        self.rows.len()
    }

    /// The heap of the block of row `i`.
    #[inline]
    fn heap(&self, i: usize) -> &'a [u8] {
        self.collection.heap(self.locations[i])
    }
}

/// Gather the values of column `index` (with layout `column`) of `rows`.
fn gather_column(
    column: &ColumnLayout,
    index: usize,
    rows: GatherRows<'_>,
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
    rows: GatherRows<'r>,
) -> Result<ArrayRef> {
    let Some(key_bit) = column.key_validity_bit() else {
        return internal_err!("Dictionary column {index} without a key validity bit");
    };
    check_column(rows.collection.layout(), column, index)?;
    let (offset, width) = (column.offset(), column.width());
    let var_len = column.storage().uses_heap();
    // The bytes of the slots of null values are zero, but an encoded null
    // value is stored with its encoding, which is part of the value.
    // SAFETY (of all reads): the slot and the validity bits of the column are
    // in the rows (checked above).
    let stored = |i: usize| -> (bool, &'r [u8]) {
        let row = rows.rows[i];
        let bytes = if var_len {
            unsafe { row.var_len(offset, || rows.heap(i)) }
        } else {
            unsafe { row.slice(offset, width) }
        };
        (unsafe { row.is_valid(index) }, bytes)
    };
    let mut keys_of_values =
        HashMap::with_capacity_and_hasher(rows.len(), RandomState::default());
    let mut distinct = Vec::new();
    let keys = (0..rows.len())
        .map(|i| {
            if !unsafe { rows.rows[i].is_valid(key_bit) } {
                return 0;
            }
            *keys_of_values.entry(stored(i)).or_insert_with(|| {
                distinct.push(i);
                distinct.len() as u64 - 1
            })
        })
        .collect::<Vec<u64>>();
    let distinct_rows = distinct.iter().map(|&i| rows.rows[i]).collect::<Vec<_>>();
    let distinct_locations = distinct
        .iter()
        .map(|&i| rows.locations[i])
        .collect::<Vec<_>>();
    let distinct = GatherRows {
        collection: rows.collection,
        locations: &distinct_locations,
        rows: &distinct_rows,
    };
    let dictionary = gather_values(column, index, distinct)?;
    new_dictionary(
        keys,
        unsafe { gather_nulls(key_bit, rows.rows) },
        &dictionary,
        key_type,
        column.data_type(),
    )
}

/// Gather the values of column `index` of `rows`, with the value type of
/// the column (i.e. not dictionary encoded).
fn gather_values(
    column: &ColumnLayout,
    index: usize,
    rows: GatherRows<'_>,
) -> Result<ArrayRef> {
    check_column(rows.collection.layout(), column, index)?;
    let offset = column.offset();
    let value_type = column.value_type();
    let len = rows.len();
    // SAFETY (of all reads): the slot and the validity bit of the column are
    // in the rows (checked above). The values of a column are values of its
    // value type: the columns of the chunks that are appended must have the
    // data types of the layout, and so must the rows that are combined.
    match column.storage() {
        ColumnStorage::Null => Ok(new_null_array(value_type, len)),
        ColumnStorage::Boolean => {
            let values = BooleanBuffer::collect_bool(len, |i| unsafe {
                rows.rows[i].byte(offset) != 0
            });
            let nulls = unsafe { gather_nulls(index, rows.rows) };
            Ok(Arc::new(BooleanArray::new(values, nulls)))
        }
        ColumnStorage::Fixed(width) => {
            let (values, all_valid) =
                unsafe { gather_fixed(rows.rows, offset, width, index) };
            // Only build a validity buffer if a value is null.
            let nulls = match all_valid {
                true => None,
                false => unsafe { gather_nulls(index, rows.rows) },
            };
            let data = ArrayDataBuilder::new(value_type.clone())
                .len(len)
                .add_buffer(values)
                .nulls(nulls)
                .build()?;
            Ok(make_array(data))
        }
        ColumnStorage::VarLen => unsafe {
            match value_type {
                DataType::Utf8 => gather_bytes::<Utf8Type>(offset, index, rows),
                DataType::LargeUtf8 => gather_bytes::<LargeUtf8Type>(offset, index, rows),
                DataType::Binary => gather_bytes::<BinaryType>(offset, index, rows),
                DataType::LargeBinary => {
                    gather_bytes::<LargeBinaryType>(offset, index, rows)
                }
                DataType::Utf8View => gather_views::<StringViewType>(offset, index, rows),
                DataType::BinaryView => {
                    gather_views::<BinaryViewType>(offset, index, rows)
                }
                data_type => {
                    internal_err!("Unexpected variable-length data type {data_type}")
                }
            }
        },
        ColumnStorage::Encoded => {
            let codec = column.codec().ok_or_else(|| {
                internal_datafusion_err!("Encoded column without a row codec")
            })?;
            codec.decode(
                (0..len)
                    .map(|i| unsafe { rows.rows[i].var_len(offset, || rows.heap(i)) }),
            )
        }
    }
}

/// The validity bit `bit` of `rows`, `None` if it is set in all rows: the
/// validity buffer is only built if a row has the bit cleared.
///
/// # Safety
///
/// `bit / 8` must be less than the number of validity bytes of the rows.
unsafe fn gather_nulls(bit: usize, rows: &[RowPtr<'_>]) -> Option<NullBuffer> {
    let (byte, shift) = (bit >> 3, bit & 7);
    // SAFETY: guaranteed by the caller.
    let valid = |row: &RowPtr<'_>| u64::from((unsafe { row.byte(byte) } >> shift) & 1);
    if rows.iter().all(|row| valid(row) != 0) {
        return None;
    }
    // The bits of 64 rows at a time.
    let mut bits = MutableBuffer::new(rows.len().div_ceil(64) * 8);
    for rows in rows.chunks(64) {
        let word = rows
            .iter()
            .enumerate()
            .fold(0_u64, |word, (i, row)| word | (valid(row) << i));
        bits.push(word);
    }
    let bits = BooleanBuffer::new(bits.into(), 0, rows.len());
    Some(NullBuffer::new(bits))
}

/// The bytes of the values of `width` bytes at `offset` of `rows`, and
/// whether the validity bit `bit` is set in all rows.
///
/// # Safety
///
/// `offset + width` must not exceed the width of the rows, and `bit / 8` must
/// be less than the number of their validity bytes.
unsafe fn gather_fixed(
    rows: &[RowPtr<'_>],
    offset: usize,
    width: usize,
    bit: usize,
) -> (Buffer, bool) {
    /// The values of type `T` at `offset` of `rows`, read in a tight loop that
    /// also checks the validity bit `bit` of the rows (in the same pass, so
    /// that the validity buffer is only built if a value is null).
    ///
    /// # Safety
    ///
    /// As for `gather_fixed`, with `width == size_of::<T>()`.
    #[inline(always)]
    unsafe fn gather<T: ArrowNativeType>(
        rows: &[RowPtr<'_>],
        offset: usize,
        bit: usize,
    ) -> (Buffer, bool) {
        let (byte, mask) = (bit >> 3, 1_u8 << (bit & 7));
        let mut valid = mask;
        let values = rows
            .iter()
            .map(|row| {
                // SAFETY: guaranteed by the caller.
                unsafe {
                    valid &= row.byte(byte);
                    row.read_native::<T>(offset)
                }
            })
            .collect::<Vec<T>>();
        (Buffer::from_vec(values), valid != 0)
    }
    // SAFETY: guaranteed by the caller.
    unsafe {
        match width {
            1 => gather::<u8>(rows, offset, bit),
            2 => gather::<u16>(rows, offset, bit),
            4 => gather::<u32>(rows, offset, bit),
            8 => gather::<u64>(rows, offset, bit),
            16 => gather::<i128>(rows, offset, bit),
            32 => gather::<i256>(rows, offset, bit),
            width => {
                let mut values = MutableBuffer::new(rows.len() * width);
                for row in rows {
                    values.extend_from_slice(row.slice(offset, width));
                }
                let all_valid = rows.iter().all(|row| row.is_valid(bit));
                (values.into(), all_valid)
            }
        }
    }
}

/// The length of the variable-length values whose slots are at `offset` of
/// `rows`.
///
/// # Safety
///
/// `offset + VAR_LEN_SLOT_WIDTH` must not exceed the width of the rows.
#[inline(always)]
unsafe fn var_len_sizes<'a>(
    rows: &'a [RowPtr<'_>],
    offset: usize,
) -> impl Iterator<Item = usize> + 'a {
    // SAFETY: guaranteed by the caller.
    rows.iter()
        .map(move |row| unsafe { row.var_len_slot(offset) } as u32 as usize)
}

/// Gather the variable-length values at `offset` of `rows` (whose validity
/// bit is `bit`) into an array of type `T`.
///
/// # Safety
///
/// The slot and the validity bit must be in the rows, and the values must be
/// values of `T` (e.g. valid UTF-8 for strings), i.e. values that were
/// appended from arrays of type `T`.
unsafe fn gather_bytes<T: ByteArrayType>(
    offset: usize,
    bit: usize,
    rows: GatherRows<'_>,
) -> Result<ArrayRef> {
    // SAFETY: guaranteed by the caller.
    let len = unsafe { var_len_sizes(rows.rows, offset) }.sum::<usize>();
    if T::Offset::from_usize(len).is_none() {
        return exec_err!(
            "Cannot gather {len} bytes into an array of type {}",
            T::DATA_TYPE
        );
    }
    let allocation_error = |e| {
        resources_datafusion_err!(
            "Failed to allocate {len} bytes of {}: {e}",
            T::DATA_TYPE
        )
    };
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(rows.len() + 1)
        .map_err(allocation_error)?;
    // Room to copy the whole inline part of a slot after the last value.
    let mut values = Vec::new();
    values
        .try_reserve_exact(len + MAX_INLINE_LEN)
        .map_err(allocation_error)?;
    offsets.push(T::Offset::usize_as(0));
    offsets.extend(rows.rows.iter().enumerate().map(|(i, row)| {
        // SAFETY: guaranteed by the caller.
        let slot = unsafe { row.var_len_slot(offset) };
        let value_len = slot as u32 as usize;
        if value_len <= MAX_INLINE_LEN {
            // Copy all inline bytes (a fixed size copy, unlike the copy of
            // the value alone) and drop the padding.
            values.extend_from_slice(&slot.to_le_bytes()[4..]);
            values.truncate(values.len() - (MAX_INLINE_LEN - value_len));
        } else {
            let start = heap_offset(slot);
            values.extend_from_slice(&rows.heap(i)[start..start + value_len]);
        }
        T::Offset::usize_as(values.len())
    }));
    // SAFETY: the validity bit is in the rows (guaranteed by the caller).
    let nulls = unsafe { gather_nulls(bit, rows.rows) };
    // SAFETY: the offsets start at 0 and grow up to the length of the values
    // (which fits the offsets, checked above), and the values are values of
    // `T` (guaranteed by the caller).
    let array = unsafe {
        let offsets = OffsetBuffer::new_unchecked(ScalarBuffer::from(offsets));
        GenericByteArray::<T>::new_unchecked(offsets, Buffer::from_vec(values), nulls)
    };
    debug_assert!(array.to_data().validate_full().is_ok());
    Ok(Arc::new(array))
}

/// Gather the variable-length values at `offset` of `rows` (whose validity
/// bit is `bit`) into a view array of type `T`.
///
/// The slot of a value of at most [`MAX_INLINE_LEN`] bytes is its view: its
/// length and its zero padded bytes (slots are zeroed before they are
/// written). The view of a longer value is the length and the prefix of its
/// slot and the location of its bytes in the data buffers.
///
/// # Safety
///
/// As for [`gather_bytes`].
unsafe fn gather_views<T: ByteViewType + ?Sized>(
    offset: usize,
    bit: usize,
    rows: GatherRows<'_>,
) -> Result<ArrayRef> {
    /// The maximum size of a data buffer.
    const MAX_BUFFER_LEN: usize = i32::MAX as usize;
    // SAFETY: guaranteed by the caller.
    let mut remaining = unsafe { var_len_sizes(rows.rows, offset) }
        .filter(|&len| len > MAX_INLINE_LEN)
        .sum::<usize>();
    let allocation_error = |e| {
        resources_datafusion_err!(
            "Failed to allocate the values of {}: {e}",
            T::DATA_TYPE
        )
    };
    let new_buffer = |len: usize| -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(len.min(MAX_BUFFER_LEN))
            .map_err(allocation_error)?;
        Ok(buffer)
    };
    let mut views = Vec::new();
    views
        .try_reserve_exact(rows.len())
        .map_err(allocation_error)?;
    let mut buffers = vec![];
    let mut buffer = new_buffer(remaining)?;
    for (i, row) in rows.rows.iter().enumerate() {
        // SAFETY: guaranteed by the caller.
        let slot = unsafe { row.var_len_slot(offset) };
        let len = slot as u32 as usize;
        if len <= MAX_INLINE_LEN {
            views.push(slot);
            continue;
        }
        if !buffer.is_empty() && buffer.len() + len > MAX_BUFFER_LEN {
            remaining -= buffer.len();
            let full = std::mem::replace(&mut buffer, new_buffer(remaining)?);
            buffers.push(Buffer::from_vec(full));
        }
        let buffer_index = u32::try_from(buffers.len())
            .map_err(|_| exec_datafusion_err!("Too many view buffers"))?;
        let buffer_offset = u32::try_from(buffer.len())
            .map_err(|_| exec_datafusion_err!("View buffer too large"))?;
        let start = heap_offset(slot);
        buffer.extend_from_slice(&rows.heap(i)[start..start + len]);
        // The length and the prefix, and where the bytes are.
        views.push(
            (slot & u128::from(u64::MAX))
                | (u128::from(buffer_index) << 64)
                | (u128::from(buffer_offset) << 96),
        );
    }
    if !buffer.is_empty() {
        buffers.push(Buffer::from_vec(buffer));
    }
    // SAFETY: the validity bit is in the rows (guaranteed by the caller).
    let nulls = unsafe { gather_nulls(bit, rows.rows) };
    // SAFETY: the inline views have zero padding, the other views have the
    // prefix of their bytes in the buffers, and the values are values of `T`
    // (guaranteed by the caller).
    let array = unsafe {
        GenericByteViewArray::<T>::new_unchecked(
            ScalarBuffer::from(views),
            buffers.into(),
            nulls,
        )
    };
    debug_assert!(array.to_data().validate_full().is_ok());
    Ok(Arc::new(array))
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
        // The values of dense unions can be shared by several rows, which
        // the row format decodes once per row: the run ends could overflow.
        assert!(!TupleDataLayout::supports_data_type(dense.data_type()));
        let batch =
            RecordBatch::try_from_iter(vec![("sparse", sparse), ("lists", lists)])
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
    fn scans_return_fewer_rows_whose_values_do_not_fit_together() {
        use arrow::array::{
            DictionaryArray, FixedSizeListArray, Int8Array, Int16Array, RunArray,
        };
        use arrow::datatypes::{Int8Type, Int16Type};

        // 8 chunks of 128 rows: lists of 32 run-end encoded values (more than
        // `Int16` run ends count in 1,024 rows), and dictionaries of 128
        // distinct strings per chunk (more than `Int8` keys count in 256
        // rows of different chunks).
        let lists = |chunk: i32| -> ArrayRef {
            let runs = RunArray::<Int16Type>::try_new(
                &Int16Array::from_iter_values((1..=128).map(|i| i * 32)),
                &Int32Array::from_iter_values((0..128).map(|i| chunk * 128 + i)),
            )
            .unwrap();
            let item = Arc::new(Field::new("item", runs.data_type().clone(), true));
            Arc::new(FixedSizeListArray::try_new(item, 32, Arc::new(runs), None).unwrap())
        };
        let strings = |chunk: i32| -> ArrayRef {
            let values = (0..128).map(|i| format!("{chunk}-{i}")).collect::<Vec<_>>();
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(
                    Int8Array::from_iter_values(0..=127),
                    Arc::new(StringArray::from(values)),
                )
                .unwrap(),
            )
        };
        let columns: [&dyn Fn(i32) -> ArrayRef; 2] = [&lists, &strings];
        for column in columns {
            let batches = (0..8)
                .map(|chunk| {
                    RecordBatch::try_from_iter(vec![("c", column(chunk))]).unwrap()
                })
                .collect::<Vec<_>>();
            let mut collection = collection_for(&batches[0], false);
            for batch in &batches {
                let chunk = DataChunk::from_record_batch(batch.clone());
                collection.append_chunk(&chunk, None).unwrap();
            }
            let locations = collection.row_locations().collect::<Vec<_>>();
            let mut state = TupleDataScanState::new();
            let mut row = 0;
            let mut sizes = vec![];
            while let Some(chunk) = collection.scan_chunk(&mut state).unwrap() {
                sizes.push(chunk.len());
                assert_eq!(state.locations(), &locations[row..row + chunk.len()]);
                let scanned = chunk.into_record_batch().unwrap();
                // Compare with the appended rows, by appended chunk.
                let mut start = 0;
                while start < scanned.num_rows() {
                    let (batch, offset) = ((row + start) / 128, (row + start) % 128);
                    let len = (128 - offset).min(scanned.num_rows() - start);
                    assert_identical(
                        &scanned.slice(start, len),
                        &batches[batch].slice(offset, len),
                    );
                    start += len;
                }
                row += scanned.num_rows();
            }
            assert_eq!(row, 1024);
            // Halved until the values fit, also for the following chunks.
            let size = sizes[0];
            assert!(size < 1024 && sizes.iter().all(|&s| s == size), "{sizes:?}");
        }
    }

    #[test]
    fn selected_run_end_encoded_values_are_taken_like_arrow() {
        use arrow::array::{Int16Array, ListArray, RunArray, StructArray, UnionArray};
        use arrow::datatypes::{Int16Type, UnionFields};

        use crate::tuple_data::PartitionedTupleData;

        // A null of each variant of a dense union, in runs [1, 2]: Arrow's
        // `take` merges equal values (all nulls are equal) into one run,
        // whose value is the value of its last row.
        let union = UnionArray::try_new(
            UnionFields::try_new(
                [0, 1],
                [
                    Field::new("i", DataType::Int32, true),
                    Field::new("s", DataType::Utf8, true),
                ],
            )
            .unwrap(),
            vec![0, 1].into(),
            Some(vec![0, 0].into()),
            vec![
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(StringArray::from(vec![None::<&str>])),
            ],
        )
        .unwrap();
        let runs: ArrayRef = Arc::new(
            RunArray::<Int16Type>::try_new(&Int16Array::from(vec![1, 2]), &union)
                .unwrap(),
        );
        let structs: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("r", runs.data_type().clone(), true)),
            Arc::clone(&runs),
        )]));
        let lists: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", runs.data_type().clone(), true)),
            OffsetBuffer::from_lengths([1, 1]),
            Arc::clone(&runs),
            None,
        ));
        for column in [runs, structs, lists] {
            let batch = RecordBatch::try_from_iter(vec![("c", column)]).unwrap();
            for selection in [vec![1, 0, 0], vec![0, 1, 1, 0], vec![1, 1, 0]] {
                let chunk = chunk_with_selection(&batch, selection);
                let expected = chunk.compact().unwrap();
                let mut collection = collection_for(&batch, false);
                let locations = collection.append_chunk(&chunk, None).unwrap();
                assert_identical(&collection.gather(&locations).unwrap(), &expected);
                let mut state = TupleDataScanState::new();
                let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
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
        }
    }

    #[test]
    fn unused_dictionary_values_of_run_end_encoded_unions() {
        use arrow::array::{
            DictionaryArray, Int8Array, Int16Array, RunArray, UnionArray,
        };
        use arrow::datatypes::{Int8Type, Int16Type, UnionFields};

        // Four run-end encoded values (two of them unused): a null of each
        // variant of a dense union, which Arrow's `take` would merge into a
        // run of the second one, then two valid values.
        let union = UnionArray::try_new(
            UnionFields::try_new(
                [0, 1],
                [
                    Field::new("i", DataType::Int32, true),
                    Field::new("s", DataType::Utf8, true),
                ],
            )
            .unwrap(),
            vec![0, 1, 0, 1].into(),
            Some(vec![0, 0, 1, 1].into()),
            vec![
                Arc::new(Int32Array::from(vec![None, Some(5)])),
                Arc::new(StringArray::from(vec![None, Some("x")])),
            ],
        )
        .unwrap();
        let values =
            RunArray::<Int16Type>::try_new(&Int16Array::from(vec![1, 2, 3, 4]), &union)
                .unwrap();
        for keys in [vec![0, 1], vec![1, 0], vec![0, 1, 3]] {
            let dictionary = DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(keys),
                Arc::new(values.clone()),
            )
            .unwrap();
            let batch =
                RecordBatch::try_from_iter(vec![("d", Arc::new(dictionary) as ArrayRef)])
                    .unwrap();
            let mut collection = collection_for(&batch, false);
            let chunk = DataChunk::from_record_batch(batch.clone());
            let locations = collection.append_chunk(&chunk, None).unwrap();
            assert_identical(&collection.gather(&locations).unwrap(), &batch);
            let mut state = TupleDataScanState::new();
            let scanned = collection.scan_chunk(&mut state).unwrap().unwrap();
            assert_identical(&scanned.into_record_batch().unwrap(), &batch);
        }
    }

    #[test]
    fn too_many_run_end_encoded_values_to_gather_are_an_error() {
        use arrow::array::{Int16Array, ListArray, RunArray};
        use arrow::datatypes::Int16Type;

        // 30,000 equal values (one run): one such array fits `Int16` run
        // ends, three do not (and their run ends would wrap to valid ones).
        let runs = || -> ArrayRef {
            Arc::new(
                RunArray::<Int16Type>::try_new(
                    &Int16Array::from(vec![30_000]),
                    &Int32Array::from(vec![7]),
                )
                .unwrap(),
            )
        };
        let list = |values: ArrayRef| -> ArrayRef {
            let item = Arc::new(Field::new("item", values.data_type().clone(), true));
            Arc::new(ListArray::new(
                item,
                OffsetBuffer::from_lengths([values.len()]),
                values,
                None,
            ))
        };
        for column in [runs(), list(runs())] {
            let batch = RecordBatch::try_from_iter(vec![("c", column)]).unwrap();
            let mut collection = collection_for(&batch, false);
            let mut locations = vec![];
            for _ in 0..3 {
                let chunk = DataChunk::from_record_batch(batch.clone());
                locations.extend(collection.append_chunk(&chunk, None).unwrap());
            }
            let rows = batch.num_rows();
            assert_identical(
                &collection.gather(&locations[rows..2 * rows]).unwrap(),
                &batch,
            );
            let err = collection.gather(&locations).unwrap_err();
            assert!(err.to_string().contains("run ends"), "{err}");
        }
    }

    #[test]
    fn dictionaries_of_shared_run_end_encoded_lists_are_not_expanded() {
        use crate::tuple_data::PartitionedTupleData;

        for dictionary in
            crate::tuple_data::test_data::dictionaries_of_shared_run_end_encoded_lists()
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

    /// Columns with the fast gather paths, with nulls at the rows `nulls`.
    fn fast_path_columns(len: usize, nulls: &[usize]) -> RecordBatch {
        use arrow::array::{
            BooleanArray, Date32Array, Decimal128Array, Float64Array, StringViewArray,
            UInt64Array,
        };
        let valid = |i: usize| !nulls.contains(&i);
        let opt = |i: usize| valid(i).then_some(i);
        let string = |i: usize| "x".repeat(i % 30);
        RecordBatch::try_from_iter(vec![
            (
                "i32",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(|i| i as i32))
                        .collect::<Int32Array>(),
                ) as ArrayRef,
            ),
            (
                "u64",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(|i| u64::MAX - i as u64))
                        .collect::<UInt64Array>(),
                ),
            ),
            (
                "f64",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(|i| -(i as f64)))
                        .collect::<Float64Array>(),
                ),
            ),
            (
                "dec128",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(|i| i128::MAX - i as i128))
                        .collect::<Decimal128Array>()
                        .with_precision_and_scale(38, 0)
                        .unwrap(),
                ),
            ),
            (
                "date32",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(|i| i as i32))
                        .collect::<Date32Array>(),
                ),
            ),
            (
                "bool",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(|i| i % 3 == 0))
                        .collect::<BooleanArray>(),
                ),
            ),
            (
                "utf8",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(string))
                        .collect::<StringArray>(),
                ),
            ),
            (
                "utf8_view",
                Arc::new(
                    (0..len)
                        .map(|i| opt(i).map(string))
                        .collect::<StringViewArray>(),
                ),
            ),
        ])
        .unwrap()
    }

    /// A validity buffer is only gathered if a gathered row is null, also
    /// when the null rows are at the boundaries of the 64 bit words of the
    /// validity buffer.
    #[test]
    fn validity_buffers_are_only_gathered_with_nulls() {
        for len in [1, 63, 64, 65, 130] {
            for nulls in [
                vec![],
                vec![0],
                vec![len - 1],
                vec![0, 63, 64, 127, len - 1],
            ] {
                let mut nulls =
                    nulls.into_iter().filter(|&i| i < len).collect::<Vec<_>>();
                nulls.dedup();
                let batch = fast_path_columns(len, &nulls);
                let mut collection = collection_for(&batch, false);
                let locations = collection
                    .append_chunk(&DataChunk::from_record_batch(batch.clone()), None)
                    .unwrap();
                let gathered = collection.gather(&locations).unwrap();
                assert_identical(&gathered, &batch);
                for column in gathered.columns() {
                    match column.nulls() {
                        None => assert!(nulls.is_empty(), "{len} rows, nulls {nulls:?}"),
                        Some(gathered) => {
                            let null_rows = (0..len)
                                .filter(|&i| gathered.is_null(i))
                                .collect::<Vec<_>>();
                            assert_eq!(null_rows, nulls, "{}", column.data_type());
                        }
                    }
                }
                // Gathering only the valid rows gathers no validity buffer.
                let valid = (0..len)
                    .filter(|i| !nulls.contains(i))
                    .map(|i| locations[i])
                    .rev()
                    .collect::<Vec<_>>();
                let gathered = collection.gather(&valid).unwrap();
                assert!(gathered.columns().iter().all(|c| c.nulls().is_none()));
            }
        }
    }

    /// Strings and binaries of all lengths up to 40 bytes (inline and in the
    /// heap, around 12 bytes, with zero bytes and multi-byte UTF-8), and with
    /// nulls, in slices of arrays (whose offsets, views and validity do not
    /// start at the first row), in view arrays with several data buffers, at
    /// the end of the data of their arrays and in dictionaries.
    #[test]
    fn var_len_values_of_all_lengths_round_trip() {
        use arrow::array::{DictionaryArray, Int16Array, StringViewBuilder};
        use arrow::datatypes::Int16Type;

        let value = |i: usize| -> String {
            let len = i % 41;
            match i % 3 {
                0 => "z".repeat(len),
                1 => (0..len)
                    .map(|j| char::from(b'a' + (j % 26) as u8))
                    .collect(),
                _ => "\0é".repeat(len / 3) + &"\0".repeat(len % 3),
            }
        };
        let values = (0..130)
            .map(|i| (i % 7 != 5).then(|| value(i)))
            .collect::<Vec<_>>();
        // The last values are short: their bytes end the data of the array.
        let strings = values.iter().map(Option::as_deref).collect::<StringArray>();
        let mut views = StringViewBuilder::new().with_fixed_block_size(64);
        views.extend(values.iter().map(Option::as_deref));
        let views = Arc::new(views.finish()) as ArrayRef;
        assert!(views.as_string_view().data_buffers().len() > 1);
        let mut arrays = vec![Arc::new(strings.clone()) as ArrayRef, views];
        for data_type in [
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::BinaryView,
        ] {
            arrays.push(cast(&strings, &data_type).unwrap());
        }
        let keys = (0..130)
            .map(|i| (i % 11 != 4).then_some((i * 7 % 50) as i16))
            .collect::<Int16Array>();
        let dictionary_values = arrays.clone();
        for values in dictionary_values {
            arrays.push(Arc::new(
                DictionaryArray::<Int16Type>::try_new(keys.clone(), values.slice(0, 50))
                    .unwrap(),
            ));
        }
        for array in arrays {
            for array in [Arc::clone(&array), array.slice(3, 120), array.slice(77, 53)] {
                let batch = RecordBatch::try_from_iter(vec![("v", array)]).unwrap();
                let mut collection = collection_for(&batch, false);
                let rows = batch.num_rows() as u32;
                let selection = (0..rows).rev().chain(0..rows / 2).collect::<Vec<_>>();
                let chunk = chunk_with_selection(&batch, selection);
                let locations = collection.append_chunk(&chunk, None).unwrap();
                let expected = chunk.compact().unwrap();
                let gathered = collection.gather(&locations).unwrap();
                assert_identical(&gathered, &expected);
                // Every view of a gathered view array is valid.
                gathered.column(0).to_data().validate_full().unwrap();
                let mut shuffled = (0..locations.len()).collect::<Vec<_>>();
                shuffled.sort_by_key(|&i| (i * 37) % locations.len());
                let gathered = collection
                    .gather(&shuffled.iter().map(|&i| locations[i]).collect::<Vec<_>>())
                    .unwrap();
                let indices = shuffled.iter().map(|&i| i as u32).collect();
                assert_identical(
                    &gathered,
                    &chunk_with_selection(&expected, indices).compact().unwrap(),
                );
            }
        }
    }

    /// Fixed-width and boolean values (and their validity) of slices of
    /// arrays, whose values and validity bits do not start at the first row.
    #[test]
    fn fixed_width_values_of_slices_round_trip() {
        let nulls = [0, 2, 3, 8, 13, 40, 63, 64, 99];
        let batch = fast_path_columns(100, &nulls);
        for (offset, len) in [(0, 100), (5, 90), (13, 60), (63, 37)] {
            let slice = batch.slice(offset, len);
            for selection in [None, Some((0..len as u32).rev().step_by(3).collect())] {
                let chunk = match selection {
                    Some(selection) => chunk_with_selection(&slice, selection),
                    None => DataChunk::from_record_batch(slice.clone()),
                };
                let mut collection = collection_for(&slice, false);
                let locations = collection.append_chunk(&chunk, None).unwrap();
                assert_identical(
                    &collection.gather(&locations).unwrap(),
                    &chunk.compact().unwrap(),
                );
            }
        }
    }

    /// The views of values with offsets are the views that Arrow creates for
    /// them (with the length and the prefix of longer values), also for values
    /// at the end of their data.
    #[test]
    fn views_of_values_with_offsets() {
        let data = (0..60_u8).map(|i| i.wrapping_mul(37)).collect::<Vec<_>>();
        for start in [0, 1, 20, 40] {
            for end in start..data.len() {
                let view = offset_value_view(&data, start..end);
                let expected = arrow::array::make_view(&data[start..end], 0, 0);
                if end - start <= MAX_INLINE_LEN {
                    assert_eq!(view, expected, "{start}..{end}");
                } else {
                    assert_eq!(view as u64, expected as u64, "{start}..{end}");
                }
            }
        }
        // The bytes after the value are cleared.
        for len in 0..=MAX_INLINE_LEN {
            let slot = inline_slot(len, u128::MAX);
            assert_eq!(slot as u32 as usize, len);
            let bytes = slot.to_le_bytes();
            assert!(bytes[4..4 + len].iter().all(|&b| b == 0xFF), "{len}");
            assert!(bytes[4 + len..].iter().all(|&b| b == 0), "{len}");
        }
    }

    #[test]
    fn collections_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TupleDataLayout>();
        assert_send_sync::<TupleDataCollection>();
        assert_send_sync::<TupleDataScanState>();
    }
}

/// Micro-benchmarks of the scatter ([`TupleDataCollection::append_chunk`]),
/// the gather ([`TupleDataCollection::gather_columns`]) and the key
/// comparison ([`RowMatcher::match_rows`](super::RowMatcher::match_rows)) of
/// the column types that are common join keys and payloads, reported in
/// nanoseconds per row. Run them in an optimized build:
///
/// ```text
/// cargo test --profile release-nonlto -p datafusion-physical-pipeline --lib \
///     tuple_data::collection::bench -- --ignored --nocapture
/// ```
///
/// `TUPLE_DATA_BENCH_ROWS` sets the number of stored rows (default 65,536),
/// and `TUPLE_DATA_BENCH_FILTER` restricts the benchmark to the cases whose
/// name contains it.
#[cfg(test)]
mod bench {
    use super::*;

    use std::fmt::Write as _;
    use std::time::Duration;

    use arrow::array::{
        Date32Array, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray,
        StringViewArray, UInt32Array, UInt64Array,
    };
    use arrow::compute::take;
    use arrow::datatypes::{Field, Schema};
    use datafusion_common::NullEquality;
    use datafusion_common::instant::Instant;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;

    use crate::tuple_data::RowMatcher;

    /// A well distributed value of row `i`.
    fn mix(i: usize) -> u64 {
        (i as u64 ^ 0x5DEE_CE66).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// `Some(value(i))`, or `None` for about `null_percent` percent of the
    /// rows.
    fn nullable<T>(i: usize, null_percent: u64, value: impl Fn(usize) -> T) -> Option<T> {
        (mix(i + 7) % 100 >= null_percent).then(|| value(i))
    }

    /// The benchmarked columns: `(name, array)`.
    fn columns(rows: usize) -> Vec<(&'static str, ArrayRef)> {
        let ints = |nulls| (0..rows).map(move |i| nullable(i, nulls, mix));
        let short = |i: usize| format!("{:016x}", mix(i))[6..].to_string();
        let long = |i: usize| format!("Customer#{:016x}", mix(i));
        let strings = |value: fn(usize) -> String, nulls| {
            (0..rows)
                .map(move |i| nullable(i, nulls, value))
                .collect::<Vec<_>>()
        };
        vec![
            (
                "Int32",
                Arc::new(ints(0).map(|v| v.map(|v| v as i32)).collect::<Int32Array>())
                    as ArrayRef,
            ),
            (
                "Int64",
                Arc::new(ints(0).map(|v| v.map(|v| v as i64)).collect::<Int64Array>()),
            ),
            (
                "Int64 (10% nulls)",
                Arc::new(
                    ints(10)
                        .map(|v| v.map(|v| v as i64))
                        .collect::<Int64Array>(),
                ),
            ),
            (
                "UInt32",
                Arc::new(
                    ints(0)
                        .map(|v| v.map(|v| v as u32))
                        .collect::<UInt32Array>(),
                ),
            ),
            ("UInt64", Arc::new(ints(0).collect::<UInt64Array>())),
            (
                "Date32",
                Arc::new(
                    ints(0)
                        .map(|v| v.map(|v| (v % 20_000) as i32))
                        .collect::<Date32Array>(),
                ),
            ),
            (
                "Decimal128(38, 10)",
                Arc::new(
                    ints(0)
                        .map(|v| v.map(|v| i128::from(v) << 20))
                        .collect::<Decimal128Array>()
                        .with_precision_and_scale(38, 10)
                        .unwrap(),
                ),
            ),
            (
                "Float64",
                Arc::new(
                    ints(0)
                        .map(|v| v.map(|v| v as f64 / 7.0))
                        .collect::<Float64Array>(),
                ),
            ),
            (
                "Utf8View (10 bytes)",
                Arc::new(StringViewArray::from(strings(short, 0))),
            ),
            (
                "Utf8View (25 bytes)",
                Arc::new(StringViewArray::from(strings(long, 0))),
            ),
            (
                "Utf8View (25 bytes, 10% nulls)",
                Arc::new(StringViewArray::from(strings(long, 10))),
            ),
            (
                "Utf8 (10 bytes)",
                Arc::new(StringArray::from(strings(short, 0))),
            ),
            (
                "Utf8 (25 bytes)",
                Arc::new(StringArray::from(strings(long, 0))),
            ),
        ]
    }

    /// The best time per row of `run` (which processes `rows` rows), in
    /// nanoseconds.
    fn ns_per_row(rows: usize, mut run: impl FnMut()) -> f64 {
        run();
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let start = Instant::now();
            let mut iterations = 0;
            while iterations == 0 || start.elapsed() < Duration::from_millis(60) {
                run();
                iterations += 1;
            }
            let elapsed = start.elapsed().as_nanos() as f64;
            best = best.min(elapsed / (iterations * rows) as f64);
        }
        best
    }

    struct Measurement {
        append: f64,
        gather_sequential: f64,
        gather_random: f64,
        match_rows: f64,
        match_prepared: f64,
    }

    /// Measure the batches `batches` (all with the same schema) whose first
    /// column is the key compared by `RowMatcher`.
    fn measure(batches: &[RecordBatch], rng: &mut StdRng) -> Measurement {
        let schema = batches[0].schema();
        let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        let layout =
            Arc::new(TupleDataLayout::try_new(Arc::clone(&schema), false).unwrap());
        let chunks = batches
            .iter()
            .map(|batch| DataChunk::from_record_batch(batch.clone()))
            .collect::<Vec<_>>();
        let append = ns_per_row(rows, || {
            let mut collection = TupleDataCollection::new(Arc::clone(&layout));
            for chunk in &chunks {
                std::hint::black_box(collection.append_chunk(chunk, None).unwrap());
            }
            std::hint::black_box(collection);
        });

        let mut collection = TupleDataCollection::new(Arc::clone(&layout));
        let mut locations = vec![];
        for chunk in &chunks {
            locations.extend(collection.append_chunk(chunk, None).unwrap());
        }
        let column_ids = (0..schema.fields().len()).collect::<Vec<_>>();
        let gather = |locations: &[RowLocation]| {
            for chunk in locations.chunks(DEFAULT_VECTOR_SIZE) {
                std::hint::black_box(
                    collection.gather_columns(chunk, &column_ids).unwrap(),
                );
            }
        };
        let gather_sequential = ns_per_row(rows, || gather(&locations));
        let mut shuffled = (0..rows).collect::<Vec<_>>();
        shuffled.shuffle(rng);
        let random_locations = shuffled.iter().map(|&i| locations[i]).collect::<Vec<_>>();
        let gather_random = ns_per_row(rows, || gather(&random_locations));

        // Probe chunks whose keys are equal to the keys of random candidates.
        let keys = arrow::compute::concat(
            &batches
                .iter()
                .map(|b| b.column(0).as_ref())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let matcher = RowMatcher::try_new(
            Arc::clone(&layout),
            vec![0],
            NullEquality::NullEqualsNull,
        )
        .unwrap();
        let probes = shuffled
            .chunks(DEFAULT_VECTOR_SIZE)
            .map(|rows| {
                let indices = rows.iter().map(|&i| i as u32).collect::<UInt32Array>();
                let probe_keys = take(keys.as_ref(), &indices, None).unwrap();
                let mut columns = vec![probe_keys];
                // The other columns are not compared.
                columns.extend(
                    schema.fields()[1..]
                        .iter()
                        .map(|f| new_null_array(f.data_type(), rows.len())),
                );
                let chunk =
                    DataChunk::try_new(Arc::clone(&schema), columns, None).unwrap();
                let candidates = rows.iter().map(|&i| locations[i]).collect::<Vec<_>>();
                (chunk, candidates)
            })
            .collect::<Vec<_>>();
        let mut sel = Vec::with_capacity(DEFAULT_VECTOR_SIZE);
        let match_rows = ns_per_row(rows, || {
            for (chunk, candidates) in &probes {
                sel.clear();
                sel.extend(0..chunk.len() as u32);
                let matched = matcher
                    .match_rows(chunk, &mut sel, candidates, &collection, None)
                    .unwrap();
                assert_eq!(matched, chunk.len());
            }
        });
        let prepared = probes
            .iter()
            .map(|(chunk, _)| matcher.prepare(chunk).unwrap())
            .collect::<Vec<_>>();
        let match_prepared = ns_per_row(rows, || {
            for ((chunk, candidates), prepared) in probes.iter().zip(&prepared) {
                sel.clear();
                sel.extend(0..chunk.len() as u32);
                let matched = matcher
                    .match_prepared(prepared, &mut sel, candidates, &collection, None)
                    .unwrap();
                assert_eq!(matched, chunk.len());
            }
        });
        Measurement {
            append,
            gather_sequential,
            gather_random,
            match_rows,
            match_prepared,
        }
    }

    fn batches(schema: &SchemaRef, columns: &[ArrayRef]) -> Vec<RecordBatch> {
        let rows = columns[0].len();
        (0..rows)
            .step_by(DEFAULT_VECTOR_SIZE)
            .map(|offset| {
                let len = DEFAULT_VECTOR_SIZE.min(rows - offset);
                RecordBatch::try_new(
                    Arc::clone(schema),
                    columns.iter().map(|c| c.slice(offset, len)).collect(),
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    #[ignore = "benchmark: run with --ignored in an optimized build"]
    fn tuple_data_fast_paths() {
        let rows = std::env::var("TUPLE_DATA_BENCH_ROWS")
            .ok()
            .and_then(|rows| rows.parse().ok())
            .unwrap_or(1 << 16);
        let filter = std::env::var("TUPLE_DATA_BENCH_FILTER").unwrap_or_default();
        let mut rng = StdRng::seed_from_u64(42);
        let columns = columns(rows);
        let mut report = format!(
            "\n{rows} rows, ns/row (best of 7)\n\n\
             | column | append_chunk | gather_columns (sequential) | gather_columns (random) | match_rows | match_prepared |\n\
             | --- | ---: | ---: | ---: | ---: | ---: |\n"
        );
        let mut line = |name: &str, m: Measurement| {
            writeln!(
                report,
                "| {name} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                m.append,
                m.gather_sequential,
                m.gather_random,
                m.match_rows,
                m.match_prepared
            )
            .unwrap();
        };
        for (name, array) in columns.iter().filter(|(name, _)| name.contains(&filter)) {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "k",
                array.data_type().clone(),
                true,
            )]));
            let measurement = measure(&batches(&schema, &[Arc::clone(array)]), &mut rng);
            line(name, measurement);
        }
        // A join build side: an `Int64` key and payload columns of all types
        // (the key comparison only compares the key).
        let payload = [
            "Int64",
            "Int32",
            "Float64",
            "Decimal128(38, 10)",
            "Date32",
            "Utf8View (25 bytes)",
            "Utf8 (10 bytes)",
        ];
        let (fields, arrays): (Vec<_>, Vec<_>) = payload
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let array = &columns.iter().find(|(n, _)| n == name).unwrap().1;
                (
                    Field::new(format!("c{i}"), array.data_type().clone(), true),
                    Arc::clone(array),
                )
            })
            .unzip();
        let schema = Arc::new(Schema::new(fields));
        let name = "7 columns (Int64 key, all of the above)";
        if name.contains(&filter) {
            line(name, measure(&batches(&schema, &arrays), &mut rng));
        }
        println!("{report}");
    }
}
