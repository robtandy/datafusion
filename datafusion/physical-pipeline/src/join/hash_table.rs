// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`JoinHashTable`]: the shared, read-only build side of a hash join,
//! modelled after DuckDB's `JoinHashTable`.
//!
//! # Build
//!
//! Every thread of the build pipeline hashes the join keys of its rows
//! ([`hash_join_keys`]) and scatters the rows (the keys first, then the
//! payload columns, plus the 64-bit hash) into a thread-local
//! [`PartitionedTupleData`], radix partitioned by the upper bits of the
//! hash. The thread-local partitions are merged by moving their blocks
//! (O(1) per block, no row is copied).
//!
//! A [`JoinHashTableBuilder`] then puts all partitions into a single
//! [`TupleDataCollection`], partition after partition, and builds one
//! pointer table per radix partition. Rows with equal keys have equal hashes
//! and therefore end up in the same partition, so the tables of the
//! partitions are independent: threads build them in parallel without any
//! synchronization ([`JoinHashTableBuilder::build_next_partition`]), and
//! [`JoinHashTableBuilder::finish`] assembles the [`JoinHashTable`]. A small
//! build side gets a single, unpartitioned table
//! ([`JoinHashTableBuilder::single_partition_up_to`]), and every thread that
//! builds tables gets thousands of rows to insert
//! ([`JoinHashTableBuilder::max_threads`]): tiny builds (e.g. the 25 nations of
//! TPC-H) are not spread over dozens of empty partitions and threads.
//!
//! # Pointer tables
//!
//! The table of a partition is an open-addressed, linear-probing array of
//! 64-bit entries; an entry packs a 16-bit salt and a 48-bit
//! [`RowLocation`]:
//!
//! ```text
//! +-----------------+--------------------------------------------+
//! | salt (16 bits)  | location of the first row of a chain + 1   |
//! +-----------------+--------------------------------------------+
//!   bits 63..48       bits 47..0 (0 = empty entry)
//! ```
//!
//! The rows with equal keys form a *chain* (a `next` array with one entry
//! per row), and every chain has one entry, which points to its first row.
//! So all duplicates of a key share an entry, and a table has as many
//! entries as distinct keys. Keys are equal with DataFusion's join key
//! equality (see [`RowMatcher`]: e.g. `-0.0 = +0.0`, and NULLs are equal
//! with [`NullEquality::NullEqualsNull`]). Of the 64 bits of a hash, the
//! upper [`PartitionedTupleData::radix_bits`] select the partition, the
//! lower bits the first slot of the probe sequence, and bits 32..48 are the
//! salt: the keys of a row are only compared with the keys of the first row
//! of a chain if the salt of its entry matches, so almost all lookups of
//! keys that are not in the table fail on the salt comparison without
//! touching any row. Hash collisions, even of all 64 bits, only cost a key
//! comparison: like the probe rows, the build rows are inserted by comparing
//! their keys with the first row of the chains whose salt matches.
//!
//! # Integer keys: dense index and key filter
//!
//! A join on a single key of an integer type (or `Date32`) whose build keys
//! are known to lie in a [`KeyRange`] (computed as the build rows are
//! collected, see [`JoinHashTableBuilder::with_key_range`]) does without
//! hashing (DuckDB's perfect hash join, DataFusion's `ArrayMap`):
//!
//! * If the range spans at most [`MAX_DENSE_SPAN`] keys (and is not much
//!   sparser than the build side), the table is a *dense index*: one entry
//!   per key of the range, indexed by `key - min`, with the packed location of
//!   the first row of the chain of that key. A probe row checks that its key is
//!   in the range and reads its entry: neither its hash nor a key comparison
//!   is needed (equal integers are equal keys).
//! * Otherwise the table is hashed, with a *key filter* that rejects the
//!   probe keys outside of the range before they are looked up, and, for a
//!   build side with few keys for its range, the keys outside of the buckets
//!   of a bitmap of at most 64 KiB (DataFusion's `KeyRangeBitmap`) in which no
//!   build key falls.
//!
//! # Probe
//!
//! [`ChainScan`] looks up the chain of every row of a probe chunk (comparing
//! the keys of the probe rows with the first rows of the chains whose salt
//! matches, in vectorized rounds, like DuckDB's `ScanStructure`) and walks
//! the matches in probe order: all build rows of the chain of the first
//! probe row, in chain order, then those of the second probe row, and so
//! on, in batches of a bounded size.
//!
//! The table can track which build rows found a match
//! ([`JoinHashTable::mark_matched`]), in an atomic bitset shared by all
//! probe threads, for the joins that emit build rows once the probe side is
//! exhausted (e.g. the unmatched rows of a `LEFT` join).

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use arrow::array::{Array, ArrayRef, ArrowPrimitiveType, AsArray};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{
    DataType, Date32Type, Int8Type, Int16Type, Int32Type, Int64Type, SchemaRef,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::{
    NullEquality, Result, exec_err, internal_err, resources_datafusion_err,
};
use datafusion_execution::memory_pool::MemoryReservation;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::tuple_data::{
    ColumnStorage, PartitionedTupleData, RowLocation, RowMatcher, TupleDataCollection,
    TupleDataLayout,
};

/// The seed of the hash of the join keys. Any constant works: the hashes
/// never leave the join, they only have to be the same on the build and on
/// the probe side.
const JOIN_HASH_SEED: u64 = 0x2f4b_6c3d_9e1a_8507;

/// The hash state of [`hash_join_keys`].
const JOIN_RANDOM_STATE: RandomState = RandomState::with_seed(JOIN_HASH_SEED);

/// The position of the salt in an entry and in a hash.
const SALT_SHIFT: u32 = RowLocation::BITS;
/// The bits of a hash that are the salt of its entry: bits 32..48, above the
/// bits that select a slot (unless a partition has more than 2^32 slots) and
/// below the bits that select a partition (at most
/// [`PartitionedTupleData::MAX_RADIX_BITS`]).
const HASH_SALT_SHIFT: u32 = 32;
const SALT_MASK: u64 = 0xFFFF;
/// The bits of an entry that hold the location of a row, plus one.
const LOCATION_MASK: u64 = (1 << RowLocation::BITS) - 1;
/// The smallest pointer table of a partition with rows.
const MIN_CAPACITY: usize = 8;

/// The widest [`KeyRange`] (`max - min`) of a [`DenseIndex`]: 262,145 entries
/// of 8 bytes (2 MiB).
pub const MAX_DENSE_SPAN: u64 = 262_144;
/// The key ranges of at most this many keys always get a [`DenseIndex`] (of at
/// most 512 KiB), however few build rows they have.
const ALWAYS_DENSE_KEYS: u64 = 1 << 16;
/// Wider key ranges get a [`DenseIndex`] if they have at most this many keys
/// per build row: a handful of keys spread over a wide range is better served
/// by a small hash table than by megabytes of mostly empty entries.
const DENSE_KEYS_PER_ROW: u64 = 8;
/// The most buckets of the bitmap of a [`KeyFilter`]: 2^19 bits, 64 KiB.
const MAX_FILTER_BUCKETS: u64 = 1 << 19;
/// The build sides of at most this many rows get a single, unpartitioned
/// table (see [`JoinHashTableBuilder::single_partition_up_to`]).
pub const SMALL_BUILD_ROWS: usize = 4096;
/// The number of rows every thread that builds pointer tables in parallel
/// gets at least (see [`JoinHashTableBuilder::max_threads`]).
const ROWS_PER_BUILD_THREAD: usize = 4096;

/// The number of radix bits of a hash join built by `threads` threads: about
/// four partitions per thread, so that the threads that build the pointer
/// tables in parallel share the work evenly.
pub fn radix_bits_for_threads(threads: usize) -> u32 {
    let threads_bits = usize::BITS - threads.max(1).saturating_sub(1).leading_zeros();
    (threads_bits + 2).clamp(2, 8)
}

/// Compute the hash of every row of the join key columns `keys` (with
/// `num_rows` rows) into `hashes`.
///
/// The build side and the probe side of a join must hash their keys with
/// this function: equal keys (with DataFusion's join key equality, e.g.
/// `-0.0 = +0.0`) then have equal hashes. The hashes of DataFusion's
/// `create_hashes` are mixed once more (with the finalizer of MurmurHash3)
/// so that all their bits, which select the partition, the slot and the
/// salt, are well distributed, also for multi-column keys.
pub fn hash_join_keys(
    keys: &[ArrayRef],
    num_rows: usize,
    hashes: &mut Vec<u64>,
) -> Result<()> {
    hashes.clear();
    hashes
        .try_reserve(num_rows)
        .map_err(|e| resources_datafusion_err!("Failed to allocate join hashes: {e}"))?;
    hashes.resize(num_rows, 0);
    create_hashes(keys, &JOIN_RANDOM_STATE, hashes)?;
    for hash in hashes.iter_mut() {
        *hash = mix(*hash);
    }
    Ok(())
}

/// Whether [`hash_join_keys`] gives all values the same hash, as with the
/// `datafusion-common/force_hash_collisions` feature, with which the tests run
/// too (in the merge queue).
#[cfg(test)]
pub(crate) fn hashes_are_forced_to_collide() -> bool {
    let keys: Vec<ArrayRef> = vec![Arc::new(arrow::array::Int64Array::from(vec![0, 1]))];
    let mut hashes = vec![];
    hash_join_keys(&keys, 2, &mut hashes).unwrap();
    hashes[0] == hashes[1]
}

/// The finalizer of MurmurHash3 (`fmix64`).
#[inline]
fn mix(mut hash: u64) -> u64 {
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

/// The salt of a row with hash `hash`.
#[inline]
fn salt(hash: u64) -> u64 {
    (hash >> HASH_SALT_SHIFT) & SALT_MASK
}

/// `location` packed into the lower 48 bits of a `u64`, plus one, so that
/// `0` means "no row".
#[inline]
fn pack(location: RowLocation) -> u64 {
    location.as_u64() + 1
}

/// The location packed by [`pack`]; `packed` must not be `0`.
#[inline]
fn unpack(packed: u64) -> RowLocation {
    RowLocation::from_u64(packed - 1)
}

/// The entry for the chain whose first row is at `location`, with salt
/// `salt`.
#[inline]
fn entry(salt: u64, location: RowLocation) -> u64 {
    (salt << SALT_SHIFT) | pack(location)
}

#[inline]
fn entry_salt(entry: u64) -> u64 {
    entry >> SALT_SHIFT
}

#[inline]
fn entry_location(entry: u64) -> RowLocation {
    unpack(entry & LOCATION_MASK)
}

/// The smallest and the largest value of the single join key of the build
/// rows of a join on a key of an integer type or `Date32` (see
/// [`KeyRange::supports`]), which lets a [`JoinHashTable`] look the probe keys
/// up without hashing them (see the [module documentation](self)).
///
/// Keys are compared as their *key bits*: the value as a `u64`, sign extended
/// for signed types. A key is in the range if and only if its bits minus the
/// bits of the minimum, with wrapping arithmetic, are at most `max - min`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRange {
    min: i128,
    max: i128,
}

impl KeyRange {
    /// The range `min..=max` of keys; `None` if `min > max`, or if they are not
    /// both values of a type that [`Self::supports`] (from `i64::MIN` to
    /// `i64::MAX`, or from `0` to `u64::MAX`).
    pub fn new(min: i128, max: i128) -> Option<Self> {
        let signed = i128::from(i64::MIN)..=i128::from(i64::MAX);
        let unsigned = 0..=i128::from(u64::MAX);
        let values_of = |values: &std::ops::RangeInclusive<i128>| {
            values.contains(&min) && values.contains(&max)
        };
        (min <= max && (values_of(&signed) || values_of(&unsigned)))
            .then_some(Self { min, max })
    }

    /// Whether a key of `data_type` can have a range: `Int8`, `Int16`,
    /// `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64` and `Date32`.
    pub fn supports(data_type: &DataType) -> bool {
        matches!(
            data_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Date32
        )
    }

    /// The range of the valid values of `array`, `None` if it has none or if
    /// its type has no range (see [`Self::supports`]).
    pub fn of_array(array: &dyn Array) -> Option<Self> {
        fn range<T: ArrowPrimitiveType>(array: &dyn Array) -> Option<(i128, i128)>
        where
            T::Native: Into<i128>,
        {
            let array = array.as_primitive::<T>();
            let min = arrow::compute::min(array)?;
            let max = arrow::compute::max(array)?;
            Some((min.into(), max.into()))
        }
        let (min, max) = match array.data_type() {
            DataType::Int8 => range::<Int8Type>(array),
            DataType::Int16 => range::<Int16Type>(array),
            DataType::Int32 => range::<Int32Type>(array),
            DataType::Int64 => range::<Int64Type>(array),
            DataType::UInt8 => range::<UInt8Type>(array),
            DataType::UInt16 => range::<UInt16Type>(array),
            DataType::UInt32 => range::<UInt32Type>(array),
            DataType::UInt64 => range::<UInt64Type>(array),
            DataType::Date32 => range::<Date32Type>(array),
            _ => None,
        }?;
        Self::new(min, max)
    }

    /// The smallest range that holds both ranges.
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// The smallest key.
    pub fn min(&self) -> i128 {
        self.min
    }

    /// The largest key.
    pub fn max(&self) -> i128 {
        self.max
    }

    /// The key bits of the smallest key.
    fn offset(&self) -> u64 {
        // The two's complement of the value, its key bits.
        self.min as u64
    }

    /// `max - min`, which a `u64` holds for keys of the supported types.
    fn span(&self) -> u64 {
        (self.max - self.min) as u64
    }
}

/// Write the key bits (see [`KeyRange`]) of every row of `array` to `bits`
/// (any bits for the NULL rows). Returns `false` (and writes nothing) if
/// `array` has a type that [`KeyRange::supports`] not.
fn key_bits(array: &dyn Array, bits: &mut Vec<u64>) -> bool {
    fn signed<T: ArrowPrimitiveType>(array: &dyn Array, bits: &mut Vec<u64>)
    where
        T::Native: Into<i64>,
    {
        let values = array.as_primitive::<T>().values();
        bits.extend(values.iter().map(|&value| value.into() as u64));
    }
    fn unsigned<T: ArrowPrimitiveType>(array: &dyn Array, bits: &mut Vec<u64>)
    where
        T::Native: Into<u64>,
    {
        let values = array.as_primitive::<T>().values();
        bits.extend(values.iter().map(|&value| value.into()));
    }
    bits.clear();
    match array.data_type() {
        DataType::Int8 => signed::<Int8Type>(array, bits),
        DataType::Int16 => signed::<Int16Type>(array, bits),
        DataType::Int32 => signed::<Int32Type>(array, bits),
        DataType::Int64 => signed::<Int64Type>(array, bits),
        DataType::Date32 => signed::<Date32Type>(array, bits),
        DataType::UInt8 => unsigned::<UInt8Type>(array, bits),
        DataType::UInt16 => unsigned::<UInt16Type>(array, bits),
        DataType::UInt32 => unsigned::<UInt32Type>(array, bits),
        DataType::UInt64 => unsigned::<UInt64Type>(array, bits),
        _ => return false,
    }
    true
}

/// Reads the key bits (see [`KeyRange`]) of the single key of stored rows:
/// the first column of their layout, an integer of 1, 2, 4 or 8 bytes stored
/// as its Arrow (native) bytes.
#[derive(Debug, Clone, Copy)]
struct KeyReader {
    offset: usize,
    width: usize,
    signed: bool,
}

impl KeyReader {
    /// The reader of the first column of `layout`, `None` if it is not a key
    /// of a type that [`KeyRange::supports`].
    fn try_new(layout: &TupleDataLayout) -> Option<Self> {
        let column = layout.columns().first()?;
        let data_type = column.data_type();
        if !KeyRange::supports(data_type) || column.is_dictionary() {
            return None;
        }
        let ColumnStorage::Fixed(width) = column.storage() else {
            return None;
        };
        let in_rows = column
            .offset()
            .checked_add(width)
            .is_some_and(|end| end <= layout.row_width());
        (matches!(width, 1 | 2 | 4 | 8) && in_rows).then(|| Self {
            offset: column.offset(),
            width,
            signed: data_type.is_signed_integer() || *data_type == DataType::Date32,
        })
    }

    /// The key bits of the (valid) key of the row `row`.
    #[inline]
    fn read(self, row: &[u8]) -> u64 {
        fn bytes<const W: usize>(row: &[u8], offset: usize) -> [u8; W] {
            let mut bytes = [0; W];
            bytes.copy_from_slice(&row[offset..offset + W]);
            bytes
        }
        let offset = self.offset;
        match (self.width, self.signed) {
            (1, true) => i64::from(i8::from_ne_bytes(bytes(row, offset))) as u64,
            (1, false) => u64::from(u8::from_ne_bytes(bytes(row, offset))),
            (2, true) => i64::from(i16::from_ne_bytes(bytes(row, offset))) as u64,
            (2, false) => u64::from(u16::from_ne_bytes(bytes(row, offset))),
            (4, true) => i64::from(i32::from_ne_bytes(bytes(row, offset))) as u64,
            (4, false) => u64::from(u32::from_ne_bytes(bytes(row, offset))),
            // The bits of an `Int64` are its two's complement.
            _ => u64::from_ne_bytes(bytes(row, offset)),
        }
    }
}

/// The index of a table whose single integer key has a [`KeyRange`] of at most
/// [`MAX_DENSE_SPAN`] keys (see the [module documentation](self)): one entry
/// per key of the range.
struct DenseIndex {
    /// The key bits of the smallest key.
    offset: u64,
    /// `max - min`: `entries` has `span + 1` entries.
    span: u64,
    /// The packed location of the first row of the chain of the key `min + i`
    /// (`0`: none) in entry `i`.
    entries: Vec<u64>,
    /// For every row (by dense row index): the packed location of the next row
    /// of its chain, `0` for the last row.
    next: Vec<u64>,
    /// The chain of the rows with a NULL key, which match the probe rows with
    /// a NULL key under [`NullEquality::NullEqualsNull`] (`0` otherwise).
    null_chain: u64,
}

impl DenseIndex {
    fn memory_size(&self) -> usize {
        (self.entries.capacity() + self.next.capacity()) * size_of::<u64>()
    }

    /// The chain of the key with key bits `bits`, `0` if it has none.
    #[inline]
    fn chain(&self, bits: u64) -> u64 {
        let offset = bits.wrapping_sub(self.offset);
        if offset <= self.span {
            self.entries[offset as usize]
        } else {
            0
        }
    }
}

/// Rejects the probe keys that no build key equals before they are looked up
/// in a hashed table: those outside of the [`KeyRange`] of the build keys, and
/// those in a bucket of keys of the range in which no build key falls.
struct KeyFilter {
    /// The key bits of the smallest build key.
    offset: u64,
    /// `max - min`.
    span: u64,
    /// One bit per bucket of `2^shift` keys of the range (from `min`), set if a
    /// build key falls in it; `None` to only check the range.
    buckets: Option<KeyBuckets>,
}

struct KeyBuckets {
    shift: u32,
    words: Vec<u64>,
}

impl KeyFilter {
    /// Whether a build key may equal the key with key bits `bits`.
    #[inline]
    fn accepts(&self, bits: u64) -> bool {
        let offset = bits.wrapping_sub(self.offset);
        offset <= self.span
            && self.buckets.as_ref().is_none_or(|buckets| {
                let bucket = (offset >> buckets.shift) as usize;
                buckets.words[bucket / 64] & (1 << (bucket % 64)) != 0
            })
    }

    fn memory_size(&self) -> usize {
        self.buckets
            .as_ref()
            .map_or(0, |buckets| buckets.words.capacity() * size_of::<u64>())
    }
}

/// How a [`JoinHashTableBuilder`] indexes the rows of a join on a single
/// integer key with a [`KeyRange`] (see [`JoinHashTableBuilder::with_key_range`]).
struct KeyIndexPlan {
    range: KeyRange,
    reader: KeyReader,
    /// The [`DenseIndex`] that the partitions build, if the table is one
    /// (otherwise, hashed tables with a [`KeyFilter`]).
    dense: Option<DenseBuild>,
    /// The number of keys per bucket of the [`KeyFilter`] of hashed tables
    /// (`2^shift`), and of buckets, if it has buckets (otherwise, it only
    /// checks the range).
    buckets: Option<(u32, usize)>,
}

/// The entries and the chains of a [`DenseIndex`], which the partitions build
/// in parallel: all rows of a key are in the same partition, which writes the
/// entry of the key, and the chains of its own rows (a range of dense row
/// indices).
struct DenseBuild {
    entries: Vec<AtomicU64>,
    next: Vec<AtomicU64>,
    null_chain: AtomicU64,
}

impl DenseBuild {
    /// The entries of `span + 1` keys and the chains of `num_rows` rows, whose
    /// memory is reserved in `reservation` before it is allocated.
    fn try_new(
        span: u64,
        num_rows: usize,
        reservation: &MemoryReservation,
    ) -> Result<Self> {
        let too_large = || {
            resources_datafusion_err!(
                "The dense JoinHashTable of {num_rows} rows and {span} keys is too large"
            )
        };
        let num_keys = usize::try_from(span)
            .ok()
            .and_then(|span| span.checked_add(1))
            .ok_or_else(too_large)?;
        let bytes = num_keys
            .checked_add(num_rows)
            .and_then(|entries| entries.checked_mul(size_of::<u64>()))
            .ok_or_else(too_large)?;
        reservation.try_grow(bytes)?;
        let atomics = |len: usize, what: &str| -> Result<Vec<AtomicU64>> {
            let mut values = Vec::new();
            values.try_reserve_exact(len).map_err(|e| {
                resources_datafusion_err!("Failed to allocate {what}: {e}")
            })?;
            values.extend((0..len).map(|_| AtomicU64::new(0)));
            Ok(values)
        };
        Ok(Self {
            entries: atomics(num_keys, "a dense JoinHashTable")?,
            next: atomics(num_rows, "the chains of a dense JoinHashTable")?,
            null_chain: AtomicU64::new(0),
        })
    }

    /// The dense index, once all partitions are built.
    fn finish(self, range: KeyRange) -> DenseIndex {
        let plain = |values: Vec<AtomicU64>| {
            values
                .into_iter()
                .map(AtomicU64::into_inner)
                .collect::<Vec<_>>()
        };
        DenseIndex {
            offset: range.offset(),
            span: range.span(),
            entries: plain(self.entries),
            next: plain(self.next),
            null_chain: self.null_chain.into_inner(),
        }
    }
}

impl KeyIndexPlan {
    /// The offset in the range of the build key with key bits `bits`, which
    /// must be in the range.
    #[inline]
    fn offset(&self, bits: u64) -> Result<u64> {
        let offset = bits.wrapping_sub(self.range.offset());
        if offset > self.range.span() {
            return internal_err!(
                "A build key of the JoinHashTable is outside of its range {:?}",
                self.range
            );
        }
        Ok(offset)
    }
}

/// The blocks and the rows (dense row indices, see
/// [`TupleDataCollection::row_index`]) of a partition in the collection of
/// a [`JoinHashTableBuilder`].
#[derive(Debug, Clone)]
struct PartitionRange {
    blocks: Range<usize>,
    rows: Range<usize>,
}

/// The pointer table of a radix partition.
#[derive(Default)]
struct PartitionTable {
    /// The entries, see the [module documentation](self). Empty if the
    /// partition has no rows to look up, otherwise a power of two of at
    /// least twice the number of rows, so that every probe sequence ends at
    /// an empty entry.
    entries: Vec<u64>,
    /// `entries.len() - 1`.
    mask: usize,
    /// For every row of the partition (by dense row index, from
    /// `first_row`): the packed location of the next row of its chain, `0`
    /// for the last row.
    next: Vec<u64>,
    /// The dense row index of the first row of the partition.
    first_row: usize,
    /// The number of rows in the chains of the table.
    num_entries: usize,
    /// The buckets of the key filter in which the keys of the rows fall (see
    /// [`KeyIndexPlan::buckets`]), until the filter is built.
    buckets: Vec<u32>,
}

impl PartitionTable {
    fn memory_size(&self) -> usize {
        (self.entries.capacity() + self.next.capacity()) * size_of::<u64>()
            + self.buckets.capacity() * size_of::<u32>()
    }
}

impl fmt::Debug for PartitionTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartitionTable")
            .field("capacity", &self.entries.len())
            .field("rows", &self.next.len())
            .field("num_entries", &self.num_entries)
            .finish()
    }
}

/// Allocate a zeroed vector of `len` entries, reporting allocation
/// failures.
fn zeroed(len: usize, what: &str) -> Result<Vec<u64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| resources_datafusion_err!("Failed to allocate {what}: {e}"))?;
    values.resize(len, 0);
    Ok(values)
}

/// Build the pointer table of the partition `range` of `collection`, whose
/// keys `matcher` compares (`key_schema` is the schema of the keys). The
/// memory of the table and of its chains is reserved in `reservation`
/// before they are allocated; the scratch space of the build is bounded,
/// it holds [`DEFAULT_VECTOR_SIZE`] rows at most.
///
/// The rows are inserted in reverse order and prepended to their chain, so
/// that the rows of a chain are in the order of the collection. A row whose
/// salt matches the salt of an entry is compared with the first row of the
/// chain of that entry. The comparisons are vectorized: the rows are
/// inserted in batches, in rounds that stop every row at its next empty
/// entry, which it claims, or at its next entry with its salt, whose chain
/// it joins if its keys are equal (otherwise it continues after that entry
/// in the next round).
///
/// With `key_index` (a join on a single integer key with a [`KeyRange`]), the
/// buckets of the [`KeyFilter`] in which the keys fall are recorded (if it has
/// buckets), and the keys are checked to be in the range (also in debug
/// builds).
fn build_partition(
    collection: &TupleDataCollection,
    range: &PartitionRange,
    matcher: &RowMatcher,
    key_schema: &SchemaRef,
    reservation: &MemoryReservation,
    key_index: Option<&KeyIndexPlan>,
) -> Result<PartitionTable> {
    let num_rows = range.rows.len();
    if num_rows == 0 {
        return Ok(PartitionTable {
            first_row: range.rows.start,
            ..Default::default()
        });
    }
    let Some(hash_offset) = collection.layout().hash_offset() else {
        return internal_err!("The rows of a JoinHashTable must store their hash");
    };
    let num_keys = matcher.key_columns().len();
    let null_equality = matcher.null_equality();
    // Room for every row: the rows with a NULL key that cannot match are only
    // skipped while they are inserted.
    let too_large = || {
        resources_datafusion_err!(
            "JoinHashTable partition with {num_rows} rows is too large"
        )
    };
    let capacity = num_rows
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .map(|capacity| capacity.max(MIN_CAPACITY))
        .ok_or_else(too_large)?;
    // The table and the chains (and the buckets of the key filter, until it
    // is built), reserved before they are allocated.
    let bucket_shift = key_index
        .and_then(|plan| plan.buckets)
        .map(|(shift, _)| shift);
    let table_bytes = capacity
        .checked_add(num_rows)
        .and_then(|entries| entries.checked_mul(size_of::<u64>()))
        .and_then(|bytes| {
            let buckets = if bucket_shift.is_some() { num_rows } else { 0 };
            bytes.checked_add(buckets.checked_mul(size_of::<u32>())?)
        })
        .ok_or_else(too_large)?;
    reservation.try_grow(table_bytes)?;
    let mut entries = zeroed(capacity, "a JoinHashTable")?;
    let mut next = zeroed(num_rows, "the chains of a JoinHashTable")?;
    let mut buckets = Vec::new();
    if bucket_shift.is_some() {
        buckets.try_reserve_exact(num_rows).map_err(|e| {
            resources_datafusion_err!(
                "Failed to allocate the key filter of a JoinHashTable: {e}"
            )
        })?;
    }
    let mask = capacity - 1;
    let key_columns = (0..num_keys).collect::<Vec<_>>();
    let mut num_entries = 0;
    // The rows of the current batch, their hashes, and the rows of the
    // batch to insert in the current round, with the slot at which they
    // continue.
    let mut locations = Vec::with_capacity(DEFAULT_VECTOR_SIZE);
    let mut hashes = Vec::with_capacity(DEFAULT_VECTOR_SIZE);
    let mut pending = Vec::with_capacity(DEFAULT_VECTOR_SIZE);
    // The rows that stopped at an entry with their salt, with that entry,
    // and the first row of its chain.
    let mut compared = Vec::new();
    let mut heads = Vec::new();
    let mut no_match = Vec::new();
    // The rows in reverse order: the blocks in reverse order, and the rows of
    // every block in reverse order, in batches.
    for block in range.blocks.clone().rev() {
        // The index in the partition of the first row of the block.
        let first_row =
            collection.row_index(RowLocation::new(block, 0)) - range.rows.start;
        let mut end = collection.block_len(block);
        while end > 0 {
            let start = end.saturating_sub(DEFAULT_VECTOR_SIZE);
            // The row `i` of the batch is the row `end - 1 - i` of the block.
            locations.clear();
            locations.extend((start..end).rev().map(|row| RowLocation::new(block, row)));
            hashes.clear();
            pending.clear();
            for (i, row) in collection.row_refs(&locations)?.iter().enumerate() {
                let hash = row.hash(hash_offset);
                hashes.push(hash);
                // Under `NullEqualsNothing`, rows with a NULL key never match:
                // they are kept (e.g. for the unmatched rows of an outer
                // join), but not inserted.
                if null_equality == NullEquality::NullEqualsNull
                    || (0..num_keys).all(|key| row.is_valid(key))
                {
                    pending.push((i, hash as usize & mask));
                }
                // The keys are checked (in debug builds, or as their buckets
                // are recorded anyway): reading the key of every row of a
                // large build side costs as much as the range saves.
                if let Some(plan) = key_index
                    && (bucket_shift.is_some() || cfg!(debug_assertions))
                    && row.is_valid(0)
                {
                    let key = plan.offset(plan.reader.read(row.row))?;
                    if let Some(shift) = bucket_shift {
                        buckets.push((key >> shift) as u32);
                    }
                }
            }
            num_entries += pending.len();
            while !pending.is_empty() {
                compared.clear();
                heads.clear();
                for &(i, mut slot) in &pending {
                    let salt = salt(hashes[i]);
                    loop {
                        let current = entries[slot];
                        if current == 0 {
                            // The first row of a new chain.
                            entries[slot] = entry(salt, locations[i]);
                            break;
                        }
                        if entry_salt(current) == salt {
                            compared.push((i, slot));
                            heads.push(entry_location(current));
                            break;
                        }
                        slot = (slot + 1) & mask;
                    }
                }
                if compared.is_empty() {
                    break;
                }
                let rows = compared
                    .iter()
                    .map(|&(i, _)| locations[i])
                    .collect::<Vec<_>>();
                let keys = collection.gather_columns(&rows, &key_columns)?;
                let keys = DataChunk::try_new(Arc::clone(key_schema), keys, None)?;
                let prepared = matcher.prepare(&keys)?;
                let mut matched = (0..compared.len() as u32).collect::<Vec<_>>();
                no_match.clear();
                matcher.match_prepared(
                    &prepared,
                    &mut matched,
                    &heads,
                    collection,
                    Some(&mut no_match),
                )?;
                for &pair in &matched {
                    // Prepend the row to the chain of the entry, whose first
                    // row may have changed in this round, to a row with equal
                    // keys.
                    let (i, slot) = compared[pair as usize];
                    next[first_row + end - 1 - i] = pack(entry_location(entries[slot]));
                    entries[slot] = entry(salt(hashes[i]), locations[i]);
                }
                pending.clear();
                pending.extend(no_match.iter().map(|&pair| {
                    let (i, slot) = compared[pair as usize];
                    (i, (slot + 1) & mask)
                }));
            }
            end = start;
        }
    }
    Ok(PartitionTable {
        entries,
        mask,
        next,
        first_row: range.rows.start,
        num_entries,
        buckets,
    })
}

/// The keys of a partition that is added to a [`DenseIndex`], with the first
/// row of their chains: an open-addressed table of the offsets of the keys in
/// their range (plus one: `0` is an empty slot), which grows with the keys, and
/// whose memory is reserved before it is allocated.
///
/// The table only keeps the threads that build the partitions in parallel from
/// writing the shared cache lines of the entries of the index for every row,
/// rather than once per key: it is optional. Without room for it (or for a
/// larger one) in the memory pool, the first rows of the chains are kept in
/// the entries of the index themselves (those of the keys in the table are
/// moved there), which only the partition of a key writes (all rows of a key
/// are in one partition). So a dense index needs no memory beyond its entries
/// and its chains, which [`DenseBuild::try_new`] reserves before the build:
/// the dense index of a build side whose hashed tables would fit in the pool
/// fits too if its entries and chains do.
struct PartitionKeys<'a> {
    /// Empty if the first rows of the chains are kept in `entries`.
    keys: Vec<u64>,
    heads: Vec<u64>,
    len: usize,
    /// The bits of the hash of a key that select its first slot.
    shift: u32,
    /// The entries of the index.
    entries: &'a [AtomicU64],
    reservation: &'a MemoryReservation,
}

impl<'a> PartitionKeys<'a> {
    /// The smallest table.
    const MIN_CAPACITY: usize = 64;

    /// The keys of a partition of the index of `entries`, in a table if the
    /// memory pool of `reservation` has room for it, otherwise in `entries`.
    fn new(entries: &'a [AtomicU64], reservation: &'a MemoryReservation) -> Self {
        let mut keys = Self {
            keys: vec![],
            heads: vec![],
            len: 0,
            shift: 0,
            entries,
            reservation,
        };
        // Without room for the table, the entries keep the chains.
        let _ = keys.resize(Self::MIN_CAPACITY);
        keys
    }

    fn memory_size(&self) -> usize {
        (self.keys.capacity() + self.heads.capacity()) * size_of::<u64>()
    }

    /// Reallocate the table with `capacity` slots (a power of two).
    fn resize(&mut self, capacity: usize) -> Result<()> {
        let bytes = 2 * capacity * size_of::<u64>();
        self.reservation.try_grow(bytes)?;
        let what = "the keys of a dense JoinHashTable";
        let (keys, heads) = match (zeroed(capacity, what), zeroed(capacity, what)) {
            (Ok(keys), Ok(heads)) => (keys, heads),
            (Err(e), _) | (_, Err(e)) => {
                self.reservation.shrink(bytes);
                return Err(e);
            }
        };
        let keys = std::mem::replace(&mut self.keys, keys);
        let heads = std::mem::replace(&mut self.heads, heads);
        self.reservation
            .shrink((keys.capacity() + heads.capacity()) * size_of::<u64>());
        self.shift = u64::BITS - capacity.trailing_zeros();
        for (key, head) in keys.into_iter().zip(heads) {
            if key != 0 {
                let slot = self.slot(key);
                self.keys[slot] = key;
                self.heads[slot] = head;
            }
        }
        Ok(())
    }

    /// The slot of the key `key` (an offset plus one): its slot, or the empty
    /// slot where it belongs.
    #[inline]
    fn slot(&self, key: u64) -> usize {
        let mask = self.keys.len() - 1;
        // Fibonacci hashing of the key, then linear probing.
        let mut slot = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
        while self.keys[slot] != 0 && self.keys[slot] != key {
            slot = (slot + 1) & mask;
        }
        slot
    }

    /// Make `row` (a packed location) the first row of the chain of the key
    /// with offset `offset`, and return the previous first row (packed, `0`
    /// for a new key).
    #[inline]
    fn prepend(&mut self, offset: u64, row: u64) -> u64 {
        if !self.keys.is_empty() {
            let key = offset + 1;
            let mut slot = self.slot(key);
            if self.keys[slot] != 0 {
                return std::mem::replace(&mut self.heads[slot], row);
            }
            // At most half of the slots are used. Without room for a larger
            // table, the entries keep the chains from now on.
            let full = 2 * (self.len + 1) > self.keys.len();
            if !full || self.resize(2 * self.keys.len()).is_ok() {
                if full {
                    slot = self.slot(key);
                }
                self.keys[slot] = key;
                self.heads[slot] = row;
                self.len += 1;
                return 0;
            }
            self.flush();
        }
        // Only the partition of the key writes its entry.
        let entry = &self.entries[offset as usize];
        let previous = entry.load(Ordering::Relaxed);
        entry.store(row, Ordering::Relaxed);
        previous
    }

    /// Write the first rows of the chains of the keys in the table to their
    /// entries, and release the table: the entries keep the chains from now
    /// on.
    fn flush(&mut self) {
        for (&key, &head) in self.keys.iter().zip(&self.heads) {
            if key != 0 {
                self.entries[(key - 1) as usize].store(head, Ordering::Relaxed);
            }
        }
        self.reservation.shrink(self.memory_size());
        self.keys = vec![];
        self.heads = vec![];
        self.len = 0;
    }
}

impl Drop for PartitionKeys<'_> {
    fn drop(&mut self) {
        self.reservation.shrink(self.memory_size());
    }
}

/// Build the chains of the rows of the partition `range` of `collection` into
/// the [`DenseIndex`] `dense` of the keys of `plan` (compared with
/// `null_equality`), and return the number of rows in the chains.
///
/// Like [`build_partition`], the rows are inserted in reverse order and
/// prepended to their chain, so that the rows of a chain are in the order of
/// the collection; no key is hashed or compared with the keys of other rows.
/// The first row of the chain of every key of the partition is kept in a table
/// of the keys of the partition ([`PartitionKeys`]), and written to the entry
/// of the key once the partition is built (or in the entry itself, if the
/// memory pool has no room for the table): the threads that build other
/// partitions write the entries of other keys, and the chains of other rows.
fn build_dense_partition(
    collection: &TupleDataCollection,
    range: &PartitionRange,
    plan: &KeyIndexPlan,
    dense: &DenseBuild,
    null_equality: NullEquality,
    reservation: &MemoryReservation,
) -> Result<usize> {
    if range.rows.is_empty() {
        return Ok(0);
    }
    let mut keys = PartitionKeys::new(&dense.entries, reservation);
    let null_equals_null = null_equality == NullEquality::NullEqualsNull;
    let mut null_chain = 0;
    let mut num_entries = 0;
    let mut locations = Vec::with_capacity(DEFAULT_VECTOR_SIZE);
    for block in range.blocks.clone().rev() {
        let first_row = collection.row_index(RowLocation::new(block, 0));
        let mut end = collection.block_len(block);
        while end > 0 {
            let start = end.saturating_sub(DEFAULT_VECTOR_SIZE);
            locations.clear();
            locations.extend((start..end).rev().map(|row| RowLocation::new(block, row)));
            let rows = collection.row_refs(&locations)?;
            for (&location, row) in locations.iter().zip(&rows) {
                let next = if row.is_valid(0) {
                    keys.prepend(plan.offset(plan.reader.read(row.row))?, pack(location))
                } else if null_equals_null {
                    std::mem::replace(&mut null_chain, pack(location))
                } else {
                    // A NULL key matches nothing.
                    continue;
                };
                dense.next[first_row + location.row()].store(next, Ordering::Relaxed);
                num_entries += 1;
            }
            end = start;
        }
    }
    keys.flush();
    if null_chain != 0 {
        dense.null_chain.store(null_chain, Ordering::Relaxed);
    }
    Ok(num_entries)
}

/// Builds a [`JoinHashTable`] from the combined build rows of all threads.
///
/// The pointer tables of the radix partitions are independent, so several
/// threads can build them in parallel: every call of
/// [`Self::build_next_partition`] (from any thread) builds the table of a
/// partition that no other call claimed. [`Self::finish`] builds the tables
/// that were not built yet and returns the [`JoinHashTable`].
///
/// Before the build starts (and the builder is shared), a build side of few
/// rows can be put in a single partition ([`Self::single_partition_up_to`]),
/// and a join on a single integer key whose build keys have a known
/// [`KeyRange`] can be indexed without hashing ([`Self::with_key_range`]).
pub struct JoinHashTableBuilder {
    collection: TupleDataCollection,
    /// Compares the keys of the rows, the first columns of the layout.
    matcher: RowMatcher,
    /// The schema of the keys.
    key_schema: SchemaRef,
    radix_bits: u32,
    partitions: Vec<PartitionRange>,
    /// The number of partitions claimed by [`Self::build_next_partition`].
    claimed: AtomicUsize,
    built: Vec<OnceLock<PartitionTable>>,
    /// How the rows of a join on a single integer key with a [`KeyRange`] are
    /// indexed, see [`Self::with_key_range`].
    key_index: Option<KeyIndexPlan>,
    /// The memory of the pointer tables, the chains and the matched rows,
    /// reserved before they are allocated.
    reservation: MemoryReservation,
}

impl fmt::Debug for JoinHashTableBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JoinHashTableBuilder")
            .field("rows", &self.collection.len())
            .field("radix_bits", &self.radix_bits)
            .field("num_keys", &self.matcher.key_columns().len())
            .field("key_range", &self.key_index.as_ref().map(|plan| plan.range))
            .field("dense", &self.has_dense_index())
            .finish()
    }
}

impl JoinHashTableBuilder {
    /// A builder of the table of the rows of `data`, whose first `num_keys`
    /// columns are the join keys, compared with `null_equality`. The layout
    /// of the rows must have a hash slot holding the hash of the keys (see
    /// [`hash_join_keys`]), by which the rows were partitioned.
    ///
    /// The memory of the pointer tables, of the chains and of the matched
    /// rows is reserved in `reservation` before it is allocated (the memory
    /// of the rows is not: the caller accounts for it). The table keeps the
    /// reservation as long as it lives.
    pub fn try_new(
        data: PartitionedTupleData,
        num_keys: usize,
        null_equality: NullEquality,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        let layout = data.layout();
        if !layout.has_hash() {
            return internal_err!("The rows of a JoinHashTable must store their hash");
        }
        if num_keys == 0 || num_keys > layout.num_columns() {
            return internal_err!(
                "A JoinHashTable needs between 1 and {} key columns, got {num_keys}",
                layout.num_columns()
            );
        }
        let key_columns = (0..num_keys).collect::<Vec<_>>();
        let matcher =
            RowMatcher::try_new(Arc::clone(layout), key_columns.clone(), null_equality)?;
        let key_schema = Arc::new(layout.schema().project(&key_columns)?);
        let radix_bits = data.radix_bits();
        let (collection, blocks) = data.into_collection()?;
        // The location of every row, plus one, must fit in 48 bits.
        if collection.num_blocks() as u64 >= RowLocation::MAX_BLOCKS {
            return exec_err!(
                "A JoinHashTable supports at most {} blocks of rows",
                RowLocation::MAX_BLOCKS - 1
            );
        }
        let mut next_row = 0;
        let partitions = blocks
            .into_iter()
            .map(|blocks| {
                let start = next_row;
                next_row += blocks
                    .clone()
                    .map(|block| collection.block_len(block))
                    .sum::<usize>();
                PartitionRange {
                    blocks,
                    rows: start..next_row,
                }
            })
            .collect::<Vec<_>>();
        let built = partitions.iter().map(|_| OnceLock::new()).collect();
        Ok(Self {
            collection,
            matcher,
            key_schema,
            radix_bits,
            partitions,
            claimed: AtomicUsize::new(0),
            built,
            key_index: None,
            reservation,
        })
    }

    /// Build a single, unpartitioned table (with `0` radix bits) if there are
    /// at most `max_rows` rows: the rows of all partitions are consecutive in
    /// the collection, so they form one partition, of which a single thread
    /// builds the table (e.g. of the 25 nations of TPC-H, which would otherwise
    /// get tables in up to dozens of partitions, built by as many threads).
    pub fn single_partition_up_to(mut self, max_rows: usize) -> Self {
        let num_rows = self.collection.len();
        if num_rows <= max_rows && self.partitions.len() > 1 {
            self.partitions = vec![PartitionRange {
                blocks: 0..self.collection.num_blocks(),
                rows: 0..num_rows,
            }];
            self.built = vec![OnceLock::new()];
            self.radix_bits = 0;
        }
        self
    }

    /// Index the rows of a join on a single key of an integer type (see
    /// [`KeyRange::supports`]) all of whose valid build keys are in `range`
    /// without hashing the probe keys (see the [module documentation](self)).
    /// The build checks that the keys are in the range as it reads them (for a
    /// dense index and for the buckets of a key filter, and in debug builds):
    /// a probe key outside of the range is rejected, so a range that misses a
    /// build key would lose its matches.
    ///
    /// * with a [`DenseIndex`] if the range spans at most [`MAX_DENSE_SPAN`]
    ///   keys, and at most 65,536 keys or 8 keys per build row (and if the
    ///   memory pool has room for its entries and its chains, all the memory
    ///   that building it needs, see `PartitionKeys`);
    /// * otherwise with the hashed tables of the partitions and a
    ///   [`KeyFilter`] of the range, and of the buckets of keys in which build
    ///   keys fall (of at most 2^19 buckets, which the partitions record as
    ///   they are built) if the build side has few keys for its range: at most
    ///   one row for two buckets, and at most half of the buckets of the range
    ///   with a key.
    ///
    /// The partitions of a dense index are built in parallel too: all rows of
    /// a key are in the same partition, which writes its entry.
    pub fn with_key_range(mut self, range: KeyRange) -> Result<Self> {
        let layout = self.matcher.layout();
        let reader = match self.matcher.key_columns() {
            [0] => KeyReader::try_new(layout),
            _ => None,
        };
        let Some(reader) = reader else {
            return internal_err!(
                "A JoinHashTable on {} keys of types {:?} has no key range",
                self.matcher.key_columns().len(),
                self.key_schema
                    .fields()
                    .iter()
                    .map(|field| field.data_type())
                    .collect::<Vec<_>>()
            );
        };
        let span = range.span();
        let num_rows = self.collection.len() as u64;
        // A dense index is optional too: without room for it in the memory
        // pool, the rows get hashed tables.
        let dense = (span <= MAX_DENSE_SPAN
            && (span < ALWAYS_DENSE_KEYS
                || span < num_rows.saturating_mul(DENSE_KEYS_PER_ROW)))
        .then(|| DenseBuild::try_new(span, self.collection.len(), &self.reservation).ok())
        .flatten();
        // The buckets of keys are only worth it if many of them have no key:
        // if there are at most half as many rows as buckets (see also
        // `key_filter`).
        let num_buckets = span
            .saturating_add(1)
            .min(MAX_FILTER_BUCKETS)
            .next_power_of_two();
        let buckets = (dense.is_none() && num_rows.saturating_mul(2) <= num_buckets)
            .then(|| {
                let shift = (0..u64::BITS)
                    .find(|&shift| (span >> shift) < num_buckets)
                    .unwrap_or(u64::BITS - 1);
                (shift, num_buckets as usize)
            });
        self.key_index = Some(KeyIndexPlan {
            range,
            reader,
            dense,
            buckets,
        });
        Ok(self)
    }

    /// Whether the table gets a [`DenseIndex`] (see [`Self::with_key_range`]).
    pub fn has_dense_index(&self) -> bool {
        self.key_index
            .as_ref()
            .is_some_and(|plan| plan.dense.is_some())
    }

    /// The number of build rows.
    pub fn num_rows(&self) -> usize {
        self.collection.len()
    }

    /// The number of radix partitions.
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// The number of partitions that have rows, i.e. a table to build.
    pub fn num_partitions_with_rows(&self) -> usize {
        self.partitions
            .iter()
            .filter(|partition| !partition.rows.is_empty())
            .count()
    }

    /// The number of threads that build the table in parallel with
    /// [`Self::build_next_partition`] (at least one): one per partition with
    /// rows, but at most one per [`ROWS_PER_BUILD_THREAD`] rows, so that a
    /// small build side is not spread over many threads that each insert a
    /// handful of rows.
    pub fn max_threads(&self) -> usize {
        self.num_partitions_with_rows()
            .min(self.collection.len().div_ceil(ROWS_PER_BUILD_THREAD))
            .max(1)
    }

    /// Build the table of a partition with rows that no other call claimed
    /// yet (or its part of the dense index, see [`Self::with_key_range`]).
    /// Returns `false` once every partition is claimed.
    pub fn build_next_partition(&self) -> Result<bool> {
        loop {
            let partition = self.claimed.fetch_add(1, Ordering::Relaxed);
            let Some(range) = self.partitions.get(partition) else {
                return Ok(false);
            };
            if range.rows.is_empty() {
                continue;
            }
            let table = self.build_partition(range)?;
            if self.built[partition].set(table).is_err() {
                return internal_err!(
                    "The table of partition {partition} was built twice"
                );
            }
            return Ok(true);
        }
    }

    /// Build the table of the partition `range`: its pointer table, or, with
    /// a dense index, the chains of its rows in the index (and a table without
    /// entries, of their number).
    fn build_partition(&self, range: &PartitionRange) -> Result<PartitionTable> {
        match self.key_index.as_ref() {
            Some(
                plan @ KeyIndexPlan {
                    dense: Some(dense), ..
                },
            ) => {
                let num_entries = build_dense_partition(
                    &self.collection,
                    range,
                    plan,
                    dense,
                    self.matcher.null_equality(),
                    &self.reservation,
                )?;
                Ok(PartitionTable {
                    first_row: range.rows.start,
                    num_entries,
                    ..Default::default()
                })
            }
            key_index => build_partition(
                &self.collection,
                range,
                &self.matcher,
                &self.key_schema,
                &self.reservation,
                key_index,
            ),
        }
    }

    /// Build the tables that were not built yet and return the
    /// [`JoinHashTable`]. If `track_matches`, the table tracks which rows
    /// found a match (see [`JoinHashTable::mark_matched`]).
    pub fn finish(self, track_matches: bool) -> Result<JoinHashTable> {
        let tables = self
            .built
            .iter()
            .zip(&self.partitions)
            .map(|(built, range)| match built.get() {
                Some(_) => Ok(None),
                None => self.build_partition(range).map(Some),
            })
            .collect::<Result<Vec<_>>>()?;
        let Self {
            collection,
            matcher,
            key_schema: _,
            radix_bits,
            partitions: _,
            built,
            claimed: _,
            key_index,
            reservation,
        } = self;
        let tables = built
            .into_iter()
            .zip(tables)
            .map(|(built, table)| match (built.into_inner(), table) {
                (Some(table), _) | (None, Some(table)) => Ok(table),
                (None, None) => internal_err!("A partition of the table was not built"),
            })
            .collect::<Result<Vec<_>>>()?;
        let num_entries = tables.iter().map(|table| table.num_entries).sum();
        let (dense, key_filter, tables) = match key_index {
            Some(KeyIndexPlan {
                range,
                dense: Some(dense),
                ..
            }) => (Some(dense.finish(range)), None, vec![]),
            Some(plan) => {
                let mut tables = tables;
                let filter = key_filter(&plan, &mut tables, &reservation)?;
                (None, Some(filter), tables)
            }
            None => (None, None, tables),
        };
        let matched = track_matches
            .then(|| MatchedRows::try_new(collection.len(), &reservation))
            .transpose()?;
        Ok(JoinHashTable {
            collection,
            matcher,
            radix_bits,
            partitions: tables,
            dense,
            key_filter,
            num_entries,
            matched,
            reservation,
        })
    }
}

/// The [`KeyFilter`] of the hashed tables `tables` of `plan`, with the buckets
/// in which the build keys fall if `plan` has buckets: those that the tables
/// recorded as they were built (the tables then release them, and their memory
/// in `reservation`). The buckets are optional: they are left out (and the
/// filter only checks the range) if the memory pool has no room for them, and
/// if more than half of the buckets of the range hold a key (they would reject
/// few keys that the range does not, and cost a lookup per probe row).
fn key_filter(
    plan: &KeyIndexPlan,
    tables: &mut [PartitionTable],
    reservation: &MemoryReservation,
) -> Result<KeyFilter> {
    let (offset, span) = (plan.range.offset(), plan.range.span());
    let mut filter = KeyFilter {
        offset,
        span,
        buckets: None,
    };
    let Some((shift, num_buckets)) = plan.buckets else {
        return Ok(filter);
    };
    let bytes = num_buckets.div_ceil(64) * size_of::<u64>();
    let mut words = match reservation.try_grow(bytes) {
        Ok(()) => Some(zeroed(
            num_buckets.div_ceil(64),
            "the key filter of a JoinHashTable",
        )?),
        Err(_) => None,
    };
    for table in tables {
        let buckets = std::mem::take(&mut table.buckets);
        if let Some(words) = &mut words {
            for &bucket in &buckets {
                let bucket = bucket as usize;
                words[bucket / 64] |= 1 << (bucket % 64);
            }
        }
        reservation.shrink(buckets.capacity() * size_of::<u32>());
    }
    let Some(words) = words else {
        return Ok(filter);
    };
    // The buckets of the range, of which no other bit is set.
    let reachable = (span >> shift) as usize + 1;
    let set = words
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum::<usize>();
    if set * 2 > reachable {
        drop(words);
        reservation.shrink(bytes);
    } else {
        filter.buckets = Some(KeyBuckets { shift, words });
    }
    Ok(filter)
}

/// One bit per build row (by dense row index), set by any thread once the
/// row found a match.
struct MatchedRows {
    words: Vec<AtomicU64>,
}

impl MatchedRows {
    /// The bitset of `num_rows` rows, whose memory is reserved in
    /// `reservation` before it is allocated.
    fn try_new(num_rows: usize, reservation: &MemoryReservation) -> Result<Self> {
        let num_words = num_rows.div_ceil(64);
        reservation.try_grow(num_words * size_of::<AtomicU64>())?;
        let mut words = Vec::new();
        words.try_reserve_exact(num_words).map_err(|e| {
            resources_datafusion_err!(
                "Failed to allocate the matched rows of a join: {e}"
            )
        })?;
        words.extend((0..num_words).map(|_| AtomicU64::new(0)));
        Ok(Self { words })
    }

    #[inline]
    fn set(&self, row: usize) {
        let word = &self.words[row / 64];
        let bit = 1 << (row % 64);
        // Avoid writing the cache line of a row that already matched (e.g.
        // a key that many probe rows match).
        if word.load(Ordering::Relaxed) & bit == 0 {
            word.fetch_or(bit, Ordering::Relaxed);
        }
    }

    #[inline]
    fn get(&self, row: usize) -> bool {
        self.words[row / 64].load(Ordering::Relaxed) & (1 << (row % 64)) != 0
    }

    fn memory_size(&self) -> usize {
        self.words.capacity() * size_of::<AtomicU64>()
    }
}

/// The build side of a hash join: the build rows in a
/// [`TupleDataCollection`] and a pointer table per radix partition, shared
/// read-only by all probe threads. See the [module documentation](self).
pub struct JoinHashTable {
    /// The build rows: the key columns, then the payload columns.
    collection: TupleDataCollection,
    /// Compares the keys of probe rows with the keys of build rows.
    matcher: RowMatcher,
    radix_bits: u32,
    /// The hashed tables of the partitions (none with a dense index).
    partitions: Vec<PartitionTable>,
    /// The index of a join on a single integer key with a narrow
    /// [`KeyRange`], instead of hashed tables.
    dense: Option<DenseIndex>,
    /// The filter of the probe keys of the hashed tables of a join on a single
    /// integer key with a [`KeyRange`].
    key_filter: Option<KeyFilter>,
    /// The number of rows in the chains (the rows that can match).
    num_entries: usize,
    /// The rows that found a match, if tracked.
    matched: Option<MatchedRows>,
    /// The memory of the pointer tables, the chains and the matched rows.
    reservation: MemoryReservation,
}

impl fmt::Debug for JoinHashTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let index = match (&self.dense, &self.key_filter) {
            (Some(_), _) => "dense",
            (None, Some(filter)) if filter.buckets.is_some() => {
                "hashed, key range and buckets"
            }
            (None, Some(_)) => "hashed, key range",
            (None, None) => "hashed",
        };
        f.debug_struct("JoinHashTable")
            .field("rows", &self.collection.len())
            .field("entries", &self.num_entries)
            .field("radix_bits", &self.radix_bits)
            .field("index", &index)
            .field("tracks_matches", &self.matched.is_some())
            .finish()
    }
}

impl JoinHashTable {
    /// Build the table of the rows of `data` on the calling thread, see
    /// [`JoinHashTableBuilder`].
    pub fn try_new(
        data: PartitionedTupleData,
        num_keys: usize,
        null_equality: NullEquality,
        track_matches: bool,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        JoinHashTableBuilder::try_new(data, num_keys, null_equality, reservation)?
            .finish(track_matches)
    }

    /// The build rows.
    pub fn collection(&self) -> &TupleDataCollection {
        &self.collection
    }

    /// The layout of the build rows.
    pub fn layout(&self) -> &Arc<TupleDataLayout> {
        self.collection.layout()
    }

    /// The number of key columns (the first columns of the layout).
    pub fn num_keys(&self) -> usize {
        self.matcher.key_columns().len()
    }

    /// How keys are compared.
    pub fn null_equality(&self) -> NullEquality {
        self.matcher.null_equality()
    }

    /// The number of build rows, including the rows that cannot match
    /// (with a NULL key under [`NullEquality::NullEqualsNothing`]).
    pub fn num_rows(&self) -> usize {
        self.collection.len()
    }

    /// The number of build rows that can match a probe row.
    pub fn num_entries(&self) -> usize {
        self.num_entries
    }

    /// Returns `true` if no probe row can match any build row.
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// The number of radix bits of the partitions.
    pub fn radix_bits(&self) -> u32 {
        self.radix_bits
    }

    /// Whether the table is a dense index of the keys of its [`KeyRange`]
    /// (see [`JoinHashTableBuilder::with_key_range`]), which looks probe keys
    /// up without their hashes.
    pub fn has_dense_index(&self) -> bool {
        self.dense.is_some()
    }

    /// Whether the probe keys are filtered by the [`KeyRange`] of the build
    /// keys before they are looked up in the hashed tables, and, if
    /// `with_buckets`, also by the buckets of keys of the range in which a
    /// build key falls (see [`JoinHashTableBuilder::with_key_range`]).
    pub fn filters_keys(&self, with_buckets: bool) -> bool {
        self.key_filter
            .as_ref()
            .is_some_and(|filter| !with_buckets || filter.buckets.is_some())
    }

    /// Whether [`ChainScan::try_new`] needs the hashes of the probe keys (see
    /// [`hash_join_keys`]): not for a dense index.
    pub fn needs_hashes(&self) -> bool {
        self.dense.is_none()
    }

    /// Whether the table tracks the rows that found a match.
    pub fn tracks_matches(&self) -> bool {
        self.matched.is_some()
    }

    /// The memory reserved for the pointer tables, the chains and the
    /// matched rows, i.e. for the table without its rows (see
    /// [`TupleDataCollection::memory_size`]).
    pub fn reserved_memory(&self) -> usize {
        self.reservation.size()
    }

    /// The memory used by the pointer tables, the chains and the matched
    /// rows, i.e. by the table without its rows (see
    /// [`TupleDataCollection::memory_size`]).
    pub fn index_memory_size(&self) -> usize {
        self.partitions
            .iter()
            .map(PartitionTable::memory_size)
            .sum::<usize>()
            + self.dense.as_ref().map_or(0, DenseIndex::memory_size)
            + self.key_filter.as_ref().map_or(0, KeyFilter::memory_size)
            + self.matched.as_ref().map_or(0, MatchedRows::memory_size)
    }

    #[inline]
    fn partition(&self, hash: u64) -> &PartitionTable {
        &self.partitions[PartitionedTupleData::partition_index(hash, self.radix_bits)]
    }

    /// Look up the chain of every probe row into `scan.chains`: the packed
    /// location of the first build row of the chain whose keys are equal to
    /// the keys of probe row `i` (the row `i` of `probe_keys`, whose first
    /// [`Self::num_keys`] columns are the keys, and whose hash is
    /// `scan.hashes[i]`), or `0` if there is none or if the row is null in
    /// `valid` (a row with a NULL key that cannot match).
    ///
    /// Like the build rows when they are inserted, every probe row stops at
    /// its next entry with its salt, and the keys of all these rows are
    /// compared with the first rows of their chains at once; the rows whose
    /// keys differ (a collision of the salt or of the hash) continue after
    /// that entry in the next round. The rows whose key the key filter rejects
    /// are not looked up.
    ///
    /// A dense index reads the entry of the key of every probe row instead:
    /// no hash (`scan.hashes` is not read), no key comparison.
    fn find_chains(
        &self,
        probe_keys: &DataChunk,
        valid: Option<&NullBuffer>,
        scan: &mut ChainScan,
    ) -> Result<()> {
        let ChainScan {
            hashes,
            chains,
            slots,
            candidates,
            compared,
            no_match,
            bits,
            ..
        } = scan;
        let num_rows = probe_keys.len();
        chains.clear();
        chains.resize(num_rows, 0);
        if self.is_empty() {
            return Ok(());
        }
        if probe_keys.selection().is_some() || probe_keys.num_columns() < self.num_keys()
        {
            return internal_err!(
                "The probe keys of a join must be {num_rows} rows without a selection"
            );
        }
        // The key bits of the single integer key of a dense index or a key
        // filter.
        let key = probe_keys.column(0);
        if (self.dense.is_some() || self.key_filter.is_some())
            && (key.data_type() != self.layout().column(0).data_type()
                || !key_bits(key.as_ref(), bits))
        {
            return internal_err!(
                "The probe key of type {} cannot be looked up in a JoinHashTable on {}",
                key.data_type(),
                self.layout().column(0).data_type()
            );
        }
        if let Some(dense) = &self.dense {
            // The entry of the key of every row, without hashing it nor
            // comparing it with the key of a build row.
            for (chain, &bits) in chains.iter_mut().zip(bits.iter()) {
                *chain = dense.chain(bits);
            }
            // The rows with a NULL key match the rows of the NULL chain (under
            // `NullEqualsNull`, otherwise there is none), and those excluded
            // by `valid` nothing.
            if let Some(nulls) = key.logical_nulls().filter(|n| n.null_count() > 0) {
                for (row, chain) in chains.iter_mut().enumerate() {
                    if nulls.is_null(row) {
                        *chain = dense.null_chain;
                    }
                }
            }
            if let Some(valid) = valid.filter(|valid| valid.null_count() > 0) {
                for (row, chain) in chains.iter_mut().enumerate() {
                    if valid.is_null(row) {
                        *chain = 0;
                    }
                }
            }
            return Ok(());
        }
        if hashes.len() != num_rows {
            return internal_err!(
                "Got {} hashes for {num_rows} probe rows of a join",
                hashes.len()
            );
        }
        // The rows that the key filter rejects are not looked up (but those
        // with a NULL key, which a NULL key of a build row may match).
        let key_nulls = key.logical_nulls();
        let rejected = |row: usize| {
            self.key_filter.as_ref().is_some_and(|filter| {
                key_nulls.as_ref().is_none_or(|nulls| nulls.is_valid(row))
                    && !filter.accepts(bits[row])
            })
        };
        // The slot of the entry with its salt at which every row stopped,
        // and the first row of the chain of that entry.
        slots.clear();
        slots.resize(num_rows, 0);
        candidates.clear();
        candidates.resize(num_rows, RowLocation::default());
        // Stop the probe row `row` with hash `hash` at its next entry with
        // its salt, from `slot` on.
        let stop = |row: usize,
                    hash: u64,
                    mut slot: usize,
                    slots: &mut [usize],
                    candidates: &mut [RowLocation],
                    compared: &mut Vec<u32>| {
            let salt = salt(hash);
            let table = self.partition(hash);
            loop {
                let current = table.entries[slot];
                if current == 0 {
                    // No build row has the keys of the row.
                    return;
                }
                if entry_salt(current) == salt {
                    slots[row] = slot;
                    candidates[row] = entry_location(current);
                    compared.push(row as u32);
                    return;
                }
                slot = (slot + 1) & table.mask;
            }
        };
        compared.clear();
        for (row, &hash) in hashes.iter().enumerate() {
            if valid.is_some_and(|valid| valid.is_null(row)) || rejected(row) {
                continue;
            }
            let table = self.partition(hash);
            if !table.entries.is_empty() {
                let slot = hash as usize & table.mask;
                stop(row, hash, slot, slots, candidates, compared);
            }
        }
        let mut prepared = None;
        while !compared.is_empty() {
            let prepared = match &prepared {
                Some(prepared) => prepared,
                None => prepared.insert(self.matcher.prepare(probe_keys)?),
            };
            no_match.clear();
            self.matcher.match_prepared(
                prepared,
                compared,
                candidates,
                &self.collection,
                Some(no_match),
            )?;
            for &row in compared.iter() {
                chains[row as usize] = pack(candidates[row as usize]);
            }
            compared.clear();
            for &row in no_match.iter() {
                let row = row as usize;
                let hash = hashes[row];
                let slot = (slots[row] + 1) & self.partition(hash).mask;
                stop(row, hash, slot, slots, candidates, compared);
            }
        }
        Ok(())
    }

    /// Record that the build rows at `locations` found a match. Does nothing
    /// if the table does not track matches.
    pub fn mark_matched(&self, locations: &[RowLocation]) {
        if let Some(matched) = &self.matched {
            for &location in locations {
                matched.set(self.collection.row_index(location));
            }
        }
    }

    /// Whether the build row at `location` found a match. Always `false` if
    /// the table does not track matches.
    pub fn is_matched_at(&self, location: RowLocation) -> bool {
        self.is_matched(self.collection.row_index(location))
    }

    /// Whether a key of each of the build rows at `locations` is NULL.
    pub fn null_keys(&self, locations: &[RowLocation]) -> Result<Vec<bool>> {
        let num_keys = self.num_keys();
        Ok(self
            .collection
            .row_refs(locations)?
            .iter()
            .map(|row| (0..num_keys).any(|key| !row.is_valid(key)))
            .collect())
    }

    /// Whether the build row with dense row index `row` (see
    /// [`TupleDataCollection::row_index`]) found a match. Always `false` if
    /// the table does not track matches.
    pub fn is_matched(&self, row: usize) -> bool {
        self.matched
            .as_ref()
            .is_some_and(|matched| matched.get(row))
    }
}

/// The matches of the rows of a probe chunk, walked in probe order: all
/// build rows that match probe row 0 (in the order of their chain), then all
/// build rows that match probe row 1, and so on (DuckDB's `ScanStructure`,
/// but in probe order, so that the output of a join keeps the order of its
/// probe side).
///
/// The matches of a probe row are the rows of the chain whose keys are equal
/// to its keys (see [`JoinHashTable`]).
///
/// A scan can be [reset](Self::reset) to scan the matches of another probe
/// chunk, reusing its buffers.
#[derive(Debug, Default)]
pub struct ChainScan {
    hashes: Vec<u64>,
    /// The first row of the chain of every probe row, packed (`0`: none).
    chains: Vec<u64>,
    /// The next probe row whose matches are returned.
    row: usize,
    /// Whether the matches of `row` are being returned.
    started: bool,
    /// The next match of `row`, packed (`0`: none), if `started`.
    next: u64,
    // Buffers of `JoinHashTable::find_chains`.
    slots: Vec<usize>,
    candidates: Vec<RowLocation>,
    compared: Vec<u32>,
    no_match: Vec<u32>,
    /// The key bits of the probe keys (see [`KeyRange`]).
    bits: Vec<u64>,
}

/// The probe rows covered by a batch of [`ChainScan::next_batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanBatch {
    /// The first probe row of the batch.
    pub start: usize,
    /// All matches of the probe rows `start..end` were returned (in this
    /// batch or, for `start`, possibly in previous batches too).
    pub end: usize,
    /// The probe row `end` has more matches, returned by the next batches;
    /// the batch may have returned some of them.
    pub in_progress: bool,
}

impl ChainScan {
    /// Start a scan of the matches of the probe rows whose keys are the
    /// first [`JoinHashTable::num_keys`] columns of `probe_keys` (without a
    /// selection), with the hashes `hashes` (see [`hash_join_keys`]; none if
    /// the table does not [need them](JoinHashTable::needs_hashes)). The rows
    /// that are null in `valid` have no matches.
    pub fn try_new(
        table: &JoinHashTable,
        probe_keys: &DataChunk,
        hashes: &[u64],
        valid: Option<&NullBuffer>,
    ) -> Result<Self> {
        let mut scan = Self::default();
        scan.reset(table, probe_keys, hashes, valid)?;
        Ok(scan)
    }

    /// Start a scan of other probe rows, see [`Self::try_new`].
    pub fn reset(
        &mut self,
        table: &JoinHashTable,
        probe_keys: &DataChunk,
        hashes: &[u64],
        valid: Option<&NullBuffer>,
    ) -> Result<()> {
        self.hashes.clear();
        self.hashes.extend_from_slice(hashes);
        self.row = 0;
        self.started = false;
        self.next = 0;
        table.find_chains(probe_keys, valid, self)
    }

    /// A scan of `num_rows` probe rows without any match (e.g. for an empty
    /// build side).
    pub fn without_matches(num_rows: usize) -> Self {
        let mut scan = Self::default();
        scan.reset_without_matches(num_rows);
        scan
    }

    /// Start a scan of `num_rows` probe rows without any match.
    pub fn reset_without_matches(&mut self, num_rows: usize) {
        self.hashes.clear();
        self.hashes.resize(num_rows, 0);
        self.chains.clear();
        self.chains.resize(num_rows, 0);
        self.row = 0;
        self.started = false;
        self.next = 0;
    }

    /// The number of probe rows.
    pub fn num_rows(&self) -> usize {
        self.chains.len()
    }

    /// Drop the matches of the probe rows whose chain starts with a row for
    /// which `skip` is true, before the scan starts.
    pub fn skip_chains(&mut self, skip: impl Fn(RowLocation) -> bool) {
        for chain in &mut self.chains {
            if *chain != 0 && skip(unpack(*chain)) {
                *chain = 0;
            }
        }
    }

    /// Whether the probe row `row` has matches.
    pub fn has_matches(&self, row: usize) -> bool {
        self.chains[row] != 0
    }

    /// Skip the next probe rows (at most `max_rows` of them) without returning
    /// their matches, e.g. for a join that only needs to know whether they have
    /// matches (see [`Self::has_matches`]), and return them. Must not be called
    /// while the matches of a probe row are being returned.
    pub fn skip_rows(&mut self, max_rows: usize) -> Range<usize> {
        let start = self.row;
        self.row = start.saturating_add(max_rows).min(self.chains.len());
        start..self.row
    }

    /// Returns `true` once the matches of all probe rows were returned.
    pub fn is_done(&self) -> bool {
        self.row >= self.chains.len()
    }

    /// Return the next matches in `probe_rows` (the probe row of every
    /// match) and `matches` (the location of its build row), both cleared
    /// first.
    ///
    /// The batch ends once `max_cost` is reached: every match costs `1`, and
    /// every probe row without any match costs `cost_without_matches` (e.g.
    /// `1` for a join that emits a row for it, `0` for one that does not). So
    /// a join that emits at most one row per match, plus one row per probe
    /// row without any match, emits at most `max_cost` rows per batch.
    pub fn next_batch(
        &mut self,
        table: &JoinHashTable,
        max_cost: usize,
        cost_without_matches: usize,
        probe_rows: &mut Vec<u32>,
        matches: &mut Vec<RowLocation>,
    ) -> ScanBatch {
        probe_rows.clear();
        matches.clear();
        let max_cost = max_cost.max(1);
        // Reserving can only fail if the memory is exhausted, then pushing
        // fails as well.
        let _ = probe_rows.try_reserve(max_cost);
        let _ = matches.try_reserve(max_cost);
        let start = self.row;
        let (mut row, mut next, mut started) = (self.row, self.next, self.started);
        let num_rows = self.chains.len();
        let mut cost = 0;
        while row < num_rows && cost < max_cost {
            if !started {
                next = self.chains[row];
                if next == 0 {
                    cost += cost_without_matches;
                    row += 1;
                    continue;
                }
                started = true;
            }
            // The chains of a dense index, or of the table of the partition of
            // the row (all rows of a chain are in the same partition).
            let (links, first_row) = match &table.dense {
                Some(dense) => (dense.next.as_slice(), 0),
                None => {
                    let partition = table.partition(self.hashes[row]);
                    (partition.next.as_slice(), partition.first_row)
                }
            };
            let probe_row = row as u32;
            loop {
                let location = unpack(next);
                probe_rows.push(probe_row);
                matches.push(location);
                cost += 1;
                next = links[table.collection.row_index(location) - first_row];
                if next == 0 || cost >= max_cost {
                    break;
                }
            }
            if next == 0 {
                started = false;
                row += 1;
            }
        }
        self.row = row;
        self.next = next;
        self.started = started;
        ScanBatch {
            start,
            end: self.row,
            in_progress: self.started,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        Array, AsArray, Date32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        Int64Array, RecordBatch, StringViewArray, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    };
    use arrow::datatypes::{DataType, Field, Int32Type, Int64Type, Schema, SchemaRef};
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };

    /// A reservation of a pool without limit.
    fn reservation() -> MemoryReservation {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        MemoryConsumer::new("JoinHashTable").register(&pool)
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int64, false),
        ]))
    }

    /// A batch of keys `keys`, with the values `first_value..` in `v`.
    fn batch_with_values(keys: Vec<Option<i32>>, first_value: i64) -> RecordBatch {
        let values =
            Int64Array::from_iter_values(first_value..first_value + keys.len() as i64);
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from(keys)), Arc::new(values)],
        )
        .unwrap()
    }

    fn batch(keys: Vec<Option<i32>>) -> RecordBatch {
        batch_with_values(keys, 0)
    }

    fn layout(schema: SchemaRef, rows_per_block: usize) -> Arc<TupleDataLayout> {
        Arc::new(
            TupleDataLayout::try_new(schema, true)
                .unwrap()
                .with_rows_per_block(rows_per_block)
                .unwrap(),
        )
    }

    /// The hashes of the first `num_keys` columns of `batch`.
    fn hashes(batch: &RecordBatch, num_keys: usize) -> Vec<u64> {
        let mut hashes = vec![];
        hash_join_keys(&batch.columns()[..num_keys], batch.num_rows(), &mut hashes)
            .unwrap();
        hashes
    }

    /// The rows of `batch` whose first `num_keys` columns have no NULL, `None`
    /// if all of them.
    fn valid_keys(batch: &RecordBatch, num_keys: usize) -> Option<NullBuffer> {
        let nulls = batch.columns()[..num_keys]
            .iter()
            .map(|column| column.logical_nulls())
            .collect::<Vec<_>>();
        NullBuffer::union_many(nulls.iter().map(Option::as_ref))
    }

    /// Store `batches` in `threads` thread-local partitions (with the hashes
    /// computed by `hash`), combine them and build the table.
    fn table(
        batches: &[RecordBatch],
        num_keys: usize,
        null_equality: NullEquality,
        radix_bits: u32,
        threads: usize,
        hash: impl Fn(&RecordBatch) -> Vec<u64>,
    ) -> JoinHashTable {
        configured_table(
            batches,
            num_keys,
            null_equality,
            radix_bits,
            threads,
            hash,
            Ok,
        )
        .unwrap()
    }

    /// [`table`], with the builder configured by `configure` before the build.
    fn configured_table(
        batches: &[RecordBatch],
        num_keys: usize,
        null_equality: NullEquality,
        radix_bits: u32,
        threads: usize,
        hash: impl Fn(&RecordBatch) -> Vec<u64>,
        configure: impl FnOnce(JoinHashTableBuilder) -> Result<JoinHashTableBuilder>,
    ) -> Result<JoinHashTable> {
        let layout = layout(batches[0].schema(), 7);
        let mut locals = (0..threads)
            .map(|_| {
                PartitionedTupleData::try_new(Arc::clone(&layout), radix_bits).unwrap()
            })
            .collect::<Vec<_>>();
        for (i, batch) in batches.iter().enumerate() {
            let chunk = DataChunk::from_record_batch(batch.clone());
            locals[i % threads]
                .append_chunk(&chunk, &hash(batch))
                .unwrap();
        }
        let mut global = PartitionedTupleData::try_new(layout, radix_bits).unwrap();
        for local in locals {
            global.combine(local).unwrap();
        }
        let builder = configure(JoinHashTableBuilder::try_new(
            global,
            num_keys,
            null_equality,
            reservation(),
        )?)?;
        // Build a partition "in parallel", the others in `finish`.
        builder.build_next_partition()?;
        builder.finish(true)
    }

    /// All (probe row, `v` of the build row) pairs with equal keys, in the
    /// order of the scan, which returns batches of at most `max_cost`
    /// matches. The keys are the first `num_keys` columns of `probe`, and `v`
    /// is the column after them.
    fn probe(
        table: &JoinHashTable,
        probe: &RecordBatch,
        num_keys: usize,
        hashes: Vec<u64>,
        max_cost: usize,
    ) -> Vec<(u32, i64)> {
        let keys = probe.project(&(0..num_keys).collect::<Vec<_>>()).unwrap();
        let keys = DataChunk::from_record_batch(keys);
        let valid = match table.null_equality() {
            NullEquality::NullEqualsNothing => valid_keys(probe, num_keys),
            NullEquality::NullEqualsNull => None,
        };
        // A dense index does not need the hashes.
        let hashes = if table.needs_hashes() { hashes } else { vec![] };
        let mut scan = ChainScan::try_new(table, &keys, &hashes, valid.as_ref()).unwrap();
        let (mut rows, mut matches) = (vec![], vec![]);
        let mut pairs = vec![];
        let mut previous_end = 0;
        while !scan.is_done() {
            let batch = scan.next_batch(table, max_cost, 0, &mut rows, &mut matches);
            assert!(rows.len() <= max_cost);
            assert_eq!(batch.start, previous_end);
            assert!(batch.end >= batch.start);
            previous_end = batch.end;
            let values = table
                .collection()
                .gather_columns(&matches, &[num_keys])
                .unwrap();
            let values = values[0].as_primitive::<Int64Type>();
            pairs.extend(rows.iter().copied().zip(values.values().iter().copied()));
        }
        pairs
    }

    fn probe_one_key(
        table: &JoinHashTable,
        probe_batch: &RecordBatch,
        max_cost: usize,
    ) -> Vec<(u32, i64)> {
        probe(table, probe_batch, 1, hashes(probe_batch, 1), max_cost)
    }

    /// Every probe row with every build row with an equal key `k`: the pairs
    /// (probe row, `v` of the build row), in probe order and then in build
    /// order.
    fn expected(
        build: &[RecordBatch],
        probe: &RecordBatch,
        null_equality: NullEquality,
    ) -> Vec<(u32, i64)> {
        let build_rows = build
            .iter()
            .flat_map(|batch| {
                let keys = batch.column(0).as_primitive::<Int32Type>().clone();
                let values = batch.column(1).as_primitive::<Int64Type>().clone();
                (0..batch.num_rows()).map(move |i| {
                    (keys.is_valid(i).then(|| keys.value(i)), values.value(i))
                })
            })
            .collect::<Vec<_>>();
        let keys = probe.column(0).as_primitive::<Int32Type>();
        let mut pairs = vec![];
        for row in 0..probe.num_rows() {
            let key = keys.is_valid(row).then(|| keys.value(row));
            for &(build_key, value) in &build_rows {
                let equal = match (key, build_key) {
                    (Some(a), Some(b)) => a == b,
                    (None, None) => null_equality == NullEquality::NullEqualsNull,
                    _ => false,
                };
                if equal {
                    pairs.push((row as u32, value));
                }
            }
        }
        pairs
    }

    fn sorted(mut pairs: Vec<(u32, i64)>) -> Vec<(u32, i64)> {
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn hashes_select_partition_salt_and_slot_independently() {
        assert_eq!(radix_bits_for_threads(0), 2);
        assert_eq!(radix_bits_for_threads(1), 2);
        assert_eq!(radix_bits_for_threads(2), 3);
        assert_eq!(radix_bits_for_threads(4), 4);
        assert_eq!(radix_bits_for_threads(16), 6);
        assert_eq!(radix_bits_for_threads(1 << 20), 8);

        let location = RowLocation::new(123_456, 4_321);
        let packed = entry(0xBEEF, location);
        assert_eq!(entry_salt(packed), 0xBEEF);
        assert_eq!(entry_location(packed), location);
        assert_ne!(entry(0, RowLocation::new(0, 0)), 0);
        assert_eq!(salt(0x0000_ABCD_0000_0000), 0xABCD);
        assert_eq!(salt(u64::MAX), SALT_MASK);

        // Equal keys have equal hashes, also `-0.0` and `+0.0`.
        let keys: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![0.0, -0.0, f64::NAN, 1.0])),
            Arc::new(StringViewArray::from(vec!["a", "a", "b", "a"])),
        ];
        let mut hashes = vec![];
        hash_join_keys(&keys, 4, &mut hashes).unwrap();
        assert_eq!(hashes[0], hashes[1]);
        // Different keys have different hashes, unless every value hashes to
        // the same value (as with `datafusion-common/force_hash_collisions`,
        // which `hash_collisions_only_cost_key_comparisons` simulates).
        if !hashes_are_forced_to_collide() {
            assert_ne!(hashes[0], hashes[2]);
            assert_ne!(hashes[0], hashes[3]);
        }
    }

    #[test]
    fn probe_finds_all_matches_in_probe_order() {
        // Duplicates, NULLs and keys on one side only, in several batches
        // and blocks.
        let build = (0..5)
            .map(|b| {
                let keys = (0..40)
                    .map(|i| ((i + b) % 9 != 4).then_some((i * 3 + b) % 17 - 3))
                    .collect();
                batch_with_values(keys, i64::from(b) * 1000)
            })
            .collect::<Vec<_>>();
        let probe_batch = batch(
            (0..100)
                .map(|i| (i % 11 != 0).then_some(i % 23 - 5))
                .collect(),
        );
        let nulls = build
            .iter()
            .map(|batch| batch.column(0).null_count())
            .sum::<usize>();
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            let expected = expected(&build, &probe_batch, null_equality);
            assert!(expected.len() > 100);
            for (radix_bits, threads) in [(0, 1), (2, 3), (5, 2)] {
                let table = table(&build, 1, null_equality, radix_bits, threads, |b| {
                    hashes(b, 1)
                });
                assert_eq!(table.num_rows(), 200);
                assert_eq!(table.radix_bits(), radix_bits);
                let entries = match null_equality {
                    NullEquality::NullEqualsNothing => 200 - nulls,
                    NullEquality::NullEqualsNull => 200,
                };
                assert_eq!(table.num_entries(), entries);
                for max_cost in [1, 7, 2048] {
                    let actual = probe_one_key(&table, &probe_batch, max_cost);
                    let case = format!(
                        "{null_equality:?}, {radix_bits} bits, {threads} threads, max cost {max_cost}"
                    );
                    assert!(actual.windows(2).all(|w| w[0].0 <= w[1].0), "{case}");
                    if threads == 1 {
                        // The rows of a chain are in build order.
                        assert_eq!(actual, expected, "{case}");
                    } else {
                        assert_eq!(sorted(actual), sorted(expected.clone()), "{case}");
                    }
                }
            }
        }
    }

    #[test]
    fn hash_collisions_only_cost_key_comparisons() {
        let build = vec![batch_with_values(
            (0..50).map(|i| Some(i % 5)).collect(),
            100,
        )];
        let expected = |probe_batch: &RecordBatch| {
            expected(&build, probe_batch, NullEquality::NullEqualsNothing)
        };
        // All rows have the same hash: one chain per key, in the same probe
        // sequence, which the key comparisons tell apart.
        let same_hash = |b: &RecordBatch| vec![42; b.num_rows()];
        let collisions =
            table(&build, 1, NullEquality::NullEqualsNothing, 2, 2, same_hash);
        let probe_batch = batch(vec![Some(3), Some(7), None, Some(0)]);
        let actual = probe(&collisions, &probe_batch, 1, same_hash(&probe_batch), 16);
        assert_eq!(sorted(actual), sorted(expected(&probe_batch)));

        // Equal salts and slots, different hashes: the search continues
        // after the entries of other hashes.
        let salted = |b: &RecordBatch| {
            let keys = b.column(0).as_primitive::<Int32Type>();
            (0..b.num_rows())
                .map(|i| (0xAB_u64 << HASH_SALT_SHIFT) | (keys.value(i) as u64 % 3 * 64))
                .collect::<Vec<_>>()
        };
        let table = table(&build, 1, NullEquality::NullEqualsNothing, 0, 1, salted);
        let probe_batch = batch(vec![Some(3), Some(4), Some(9), Some(1)]);
        let actual = probe(&table, &probe_batch, 1, salted(&probe_batch), 5);
        assert_eq!(actual, expected(&probe_batch));
    }

    #[test]
    fn collisions_of_string_keys() {
        // String keys, all with the same hash: one chain per key.
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8View, true),
            Field::new("v", DataType::Int64, false),
        ]));
        let words = ["a", "b", "a string longer than twelve bytes", "", "c"];
        let batch = |keys: Vec<Option<&str>>| {
            let values = Int64Array::from_iter_values(0..keys.len() as i64);
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(StringViewArray::from(keys)), Arc::new(values)],
            )
            .unwrap()
        };
        let build = vec![batch(
            (0..40)
                .map(|i| (i % 7 != 6).then_some(words[i % 5]))
                .collect(),
        )];
        let probe_batch = batch(vec![Some("a"), None, Some("x"), Some(""), Some("c")]);
        let same_hash = |b: &RecordBatch| vec![7; b.num_rows()];
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            let table = table(&build, 1, null_equality, 1, 3, same_hash);
            let actual = probe(&table, &probe_batch, 1, same_hash(&probe_batch), 9);
            let build_keys = build[0].column(0).as_string_view();
            let probe_keys = probe_batch.column(0).as_string_view();
            let mut expected = vec![];
            for row in 0..probe_batch.num_rows() {
                for i in 0..build[0].num_rows() {
                    let equal = match (probe_keys.is_valid(row), build_keys.is_valid(i)) {
                        (true, true) => probe_keys.value(row) == build_keys.value(i),
                        (false, false) => null_equality == NullEquality::NullEqualsNull,
                        _ => false,
                    };
                    if equal {
                        expected.push((row as u32, i as i64));
                    }
                }
            }
            assert!(expected.len() > 10);
            assert_eq!(sorted(actual), sorted(expected), "{null_equality:?}");
        }
    }

    #[test]
    fn multi_column_keys_and_matched_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float64, true),
            Field::new("s", DataType::Utf8View, true),
            Field::new("v", DataType::Int64, false),
        ]));
        let build = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float64Array::from(vec![
                    Some(0.0),
                    Some(-0.0),
                    Some(f64::NAN),
                    None,
                    Some(1.5),
                ])),
                Arc::new(StringViewArray::from(vec![
                    Some("a fairly long string value"),
                    Some("a fairly long string value"),
                    Some("n"),
                    Some("x"),
                    None,
                ])),
                Arc::new(Int64Array::from(vec![10, 11, 12, 13, 14])),
            ],
        )
        .unwrap();
        let probe_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![
                    Some(-0.0),
                    Some(f64::NAN),
                    None,
                    Some(1.5),
                    Some(0.0),
                ])),
                Arc::new(StringViewArray::from(vec![
                    Some("a fairly long string value"),
                    Some("n"),
                    Some("x"),
                    None,
                    Some("a fairly long string valuE"),
                ])),
                Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4])),
            ],
        )
        .unwrap();
        for (null_equality, expected) in [
            (
                NullEquality::NullEqualsNothing,
                vec![(0, 10), (0, 11), (1, 12)],
            ),
            (
                NullEquality::NullEqualsNull,
                vec![(0, 10), (0, 11), (1, 12), (2, 13), (3, 14)],
            ),
        ] {
            let table =
                table(std::slice::from_ref(&build), 2, null_equality, 1, 1, |b| {
                    hashes(b, 2)
                });
            assert_eq!(table.num_keys(), 2);
            let actual = probe(&table, &probe_batch, 2, hashes(&probe_batch, 2), 100);
            assert_eq!(actual, expected, "{null_equality:?}");

            // The matched rows are tracked by dense row index.
            assert!(table.tracks_matches());
            let locations = table.collection().row_locations().collect::<Vec<_>>();
            table.mark_matched(&locations[1..3]);
            table.mark_matched(&locations[2..3]);
            let matched = (0..table.num_rows())
                .filter(|&row| table.is_matched(row))
                .collect::<Vec<_>>();
            assert_eq!(matched, vec![1, 2]);
        }
    }

    #[test]
    fn batches_stop_at_the_cost_and_resume_within_a_chain() {
        // Key 0 with 10 rows, key 1 with 1 row.
        let build = vec![batch((0..11).map(|i| Some(i32::from(i == 10))).collect())];
        let probe_batch = batch(vec![Some(5), Some(0), Some(1), Some(0)]);
        let table = table(&build, 1, NullEquality::NullEqualsNothing, 1, 1, |b| {
            hashes(b, 1)
        });
        let keys = DataChunk::from_record_batch(probe_batch.project(&[0]).unwrap());
        let mut scan =
            ChainScan::try_new(&table, &keys, &hashes(&probe_batch, 1), None).unwrap();
        assert_eq!(scan.num_rows(), 4);
        assert!(!scan.has_matches(0));
        assert!(scan.has_matches(1));
        let (mut rows, mut matches) = (vec![], vec![]);
        let mut batches = vec![];
        while !scan.is_done() {
            let batch = scan.next_batch(&table, 4, 1, &mut rows, &mut matches);
            assert_eq!(rows.len(), matches.len());
            batches.push((batch, rows.clone()));
        }
        let batch = |start, end, in_progress| ScanBatch {
            start,
            end,
            in_progress,
        };
        // Probe row 0 has no match and costs 1.
        let expected = vec![
            (batch(0, 1, true), vec![1, 1, 1]),
            (batch(1, 1, true), vec![1, 1, 1, 1]),
            (batch(1, 3, false), vec![1, 1, 1, 2]),
            (batch(3, 3, true), vec![3, 3, 3, 3]),
            (batch(3, 3, true), vec![3, 3, 3, 3]),
            (batch(3, 4, false), vec![3, 3]),
        ];
        assert_eq!(batches, expected);

        // Skipped chains.
        let mut scan =
            ChainScan::try_new(&table, &keys, &hashes(&probe_batch, 1), None).unwrap();
        let first = table.collection().row_locations().next().unwrap();
        // Key 0's chain starts with the first row (rows are in build order).
        scan.skip_chains(|head| head != first);
        assert!(scan.has_matches(1) && scan.has_matches(3));
        assert!(!scan.has_matches(2));

        // Probe rows whose matches are not walked.
        let mut scan =
            ChainScan::try_new(&table, &keys, &hashes(&probe_batch, 1), None).unwrap();
        assert_eq!(scan.skip_rows(3), 0..3);
        assert!(!scan.is_done());
        assert_eq!(scan.skip_rows(3), 3..4);
        assert!(scan.is_done());
        assert_eq!(scan.skip_rows(3), 4..4);

        // Probe rows without matches that cost nothing.
        let mut scan = ChainScan::without_matches(3);
        let next = scan.next_batch(&table, 2, 1, &mut rows, &mut matches);
        assert_eq!(next, batch(0, 2, false));
        assert!(rows.is_empty());
        let next = scan.next_batch(&table, 2, 0, &mut rows, &mut matches);
        assert_eq!(next, batch(2, 3, false));
        assert!(scan.is_done());
    }

    #[test]
    fn the_index_memory_is_reserved_before_it_is_allocated() {
        // 20,000 rows in 8 partitions, NULL keys included.
        let batches = (0..10)
            .map(|b| {
                let keys = (0..2000)
                    .map(|i| (i % 13 != 0).then_some(b * 2000 + i))
                    .collect();
                batch_with_values(keys, i64::from(b) * 2000)
            })
            .collect::<Vec<_>>();
        // Hashes that spread the keys over partitions, salts and slots without
        // `create_hashes`, whose hashes all collide with
        // `force_hash_collisions`, which would make the builds quadratic.
        let spread = |batch: &RecordBatch| {
            let keys = batch.column(0).as_primitive::<Int32Type>();
            (0..batch.num_rows())
                .map(|i| (keys.value(i) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .collect::<Vec<_>>()
        };
        let data = || {
            let layout = layout(schema(), 512);
            let mut data = PartitionedTupleData::try_new(layout, 3).unwrap();
            for batch in &batches {
                let chunk = DataChunk::from_record_batch(batch.clone());
                data.append_chunk(&chunk, &spread(batch)).unwrap();
            }
            data
        };
        let builder = |pool: &Arc<dyn MemoryPool>| {
            let reservation = MemoryConsumer::new("JoinHashTable").register(pool);
            JoinHashTableBuilder::try_new(
                data(),
                1,
                NullEquality::NullEqualsNothing,
                reservation,
            )
            .unwrap()
        };

        // Without limit, the reservation holds exactly the memory of the
        // pointer tables, the chains and the matched rows, until the table
        // is dropped.
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let unlimited = builder(&pool);
        assert!(unlimited.build_next_partition().unwrap());
        let table = unlimited.finish(true).unwrap();
        let index = table.index_memory_size();
        // Tables of at least twice the rows, chains and matched rows.
        assert!(index >= 20_000 * 3 * 8 + 20_000 / 8, "{index}");
        assert_eq!(table.reserved_memory(), index);
        assert_eq!(pool.reserved(), index);
        drop(table);
        assert_eq!(pool.reserved(), 0);

        // With a limit, the build fails before it allocates more than the
        // limit allows, while building the partitions in parallel or when
        // finishing the table.
        for limit in [0, 1000, index / 2, index - 1] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
            let limited = builder(&pool);
            let mut result = Ok(true);
            while matches!(result, Ok(true)) {
                result = limited.build_next_partition();
                assert!(pool.reserved() <= limit);
            }
            let err = match result {
                Err(err) => err,
                Ok(_) => limited.finish(true).unwrap_err(),
            };
            assert!(
                matches!(
                    err,
                    datafusion_common::DataFusionError::ResourcesExhausted(_)
                ),
                "{err}"
            );
            assert!(err.to_string().contains("JoinHashTable"), "{err}");
            assert!(pool.reserved() <= limit);
        }
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(index));
        assert_eq!(
            builder(&pool).finish(true).unwrap().reserved_memory(),
            index
        );
    }

    #[test]
    fn empty_tables_and_invalid_builders() {
        let layout = layout(schema(), 16);
        let data = PartitionedTupleData::try_new(Arc::clone(&layout), 3).unwrap();
        let table = JoinHashTable::try_new(
            data,
            1,
            NullEquality::NullEqualsNothing,
            false,
            reservation(),
        )
        .unwrap();
        assert!(table.is_empty());
        assert_eq!(table.num_rows(), 0);
        assert!(!table.tracks_matches());
        assert!(!table.is_matched(0));
        let keys = DataChunk::from_record_batch(batch(vec![Some(1), Some(2), None]));
        let mut scan = ChainScan::try_new(&table, &keys, &[1, 2, 3], None).unwrap();
        let (mut rows, mut matches) = (vec![], vec![]);
        let next = scan.next_batch(&table, 10, 1, &mut rows, &mut matches);
        assert_eq!(next.end, 3);
        assert!(rows.is_empty());

        // Only rows with NULL keys: nothing to look up.
        let mut data = PartitionedTupleData::try_new(Arc::clone(&layout), 2).unwrap();
        let nulls = batch(vec![None, None, None]);
        let chunk = DataChunk::from_record_batch(nulls.clone());
        data.append_chunk(&chunk, &hashes(&nulls, 1)).unwrap();
        let builder = JoinHashTableBuilder::try_new(
            data,
            1,
            NullEquality::NullEqualsNothing,
            reservation(),
        )
        .unwrap();
        assert_eq!(builder.num_rows(), 3);
        assert_eq!(builder.num_partitions(), 4);
        assert_eq!(builder.num_partitions_with_rows(), 1);
        assert!(builder.build_next_partition().unwrap());
        assert!(!builder.build_next_partition().unwrap());
        let table = builder.finish(true).unwrap();
        assert!(table.is_empty());
        assert_eq!(table.num_rows(), 3);
        assert!(table.index_memory_size() > 0);
        assert!(format!("{table:?}").contains("entries: 0"), "{table:?}");

        let without_hash = Arc::new(TupleDataLayout::try_new(schema(), false).unwrap());
        let data = PartitionedTupleData::try_new(without_hash, 0).unwrap();
        assert!(
            JoinHashTableBuilder::try_new(
                data,
                1,
                NullEquality::NullEqualsNothing,
                reservation()
            )
            .is_err()
        );
        for num_keys in [0, 3] {
            let data = PartitionedTupleData::try_new(Arc::clone(&layout), 0).unwrap();
            let builder = JoinHashTableBuilder::try_new(
                data,
                num_keys,
                NullEquality::NullEqualsNothing,
                reservation(),
            );
            assert!(builder.is_err(), "{num_keys} keys");
        }
    }

    /// The build and probe batches of `probe_finds_all_matches_in_probe_order`:
    /// keys from -3 to 13 and from -5 to 17, with duplicates and NULLs.
    fn build_and_probe() -> (Vec<RecordBatch>, RecordBatch) {
        let build = (0..5)
            .map(|b| {
                let keys = (0..40)
                    .map(|i| ((i + b) % 9 != 4).then_some((i * 3 + b) % 17 - 3))
                    .collect();
                batch_with_values(keys, i64::from(b) * 1000)
            })
            .collect::<Vec<_>>();
        let probe_batch = batch(
            (0..100)
                .map(|i| (i % 11 != 0).then_some(i % 23 - 5))
                .collect(),
        );
        (build, probe_batch)
    }

    /// `batch` (of [`schema`]) with its keys `k` mapped to keys of `data_type`
    /// by `map`, an injective map: the keys are equal if and only if the keys
    /// of `batch` are.
    fn with_keys(
        batch: &RecordBatch,
        data_type: &DataType,
        map: impl Fn(i32) -> i128,
    ) -> RecordBatch {
        let keys = batch
            .column(0)
            .as_primitive::<Int32Type>()
            .iter()
            .map(|key| key.map(&map))
            .collect::<Vec<_>>();
        fn array<T>(keys: &[Option<i128>]) -> Vec<Option<T>>
        where
            T: TryFrom<i128>,
            T::Error: fmt::Debug,
        {
            keys.iter()
                .map(|key| key.map(|key| T::try_from(key).unwrap()))
                .collect()
        }
        let keys: ArrayRef = match data_type {
            DataType::Int8 => Arc::new(Int8Array::from(array::<i8>(&keys))),
            DataType::Int16 => Arc::new(Int16Array::from(array::<i16>(&keys))),
            DataType::Int32 => Arc::new(Int32Array::from(array::<i32>(&keys))),
            DataType::Int64 => Arc::new(Int64Array::from(array::<i64>(&keys))),
            DataType::UInt8 => Arc::new(UInt8Array::from(array::<u8>(&keys))),
            DataType::UInt16 => Arc::new(UInt16Array::from(array::<u16>(&keys))),
            DataType::UInt32 => Arc::new(UInt32Array::from(array::<u32>(&keys))),
            DataType::UInt64 => Arc::new(UInt64Array::from(array::<u64>(&keys))),
            DataType::Date32 => Arc::new(Date32Array::from(array::<i32>(&keys))),
            other => panic!("no keys of type {other}"),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", data_type.clone(), true),
            Field::new("v", DataType::Int64, false),
        ]));
        RecordBatch::try_new(schema, vec![keys, Arc::clone(batch.column(1))]).unwrap()
    }

    /// The [`KeyRange`] of the keys of `batches`.
    fn key_range(batches: &[RecordBatch]) -> KeyRange {
        batches
            .iter()
            .filter_map(|batch| KeyRange::of_array(batch.column(0).as_ref()))
            .reduce(KeyRange::union)
            .unwrap()
    }

    #[test]
    fn key_ranges_of_integer_keys() {
        assert_eq!(KeyRange::new(3, 2), None);
        assert_eq!(KeyRange::new(i128::from(i64::MIN) - 1, 0), None);
        assert_eq!(KeyRange::new(0, i128::from(u64::MAX) + 1), None);
        // The values of no type.
        assert_eq!(KeyRange::new(-1, i128::from(u64::MAX)), None);
        let range = KeyRange::new(-5, 17).unwrap();
        assert_eq!((range.min(), range.max()), (-5, 17));
        assert_eq!((range.offset(), range.span()), ((-5_i64) as u64, 22));
        for (min, max) in [
            (i128::from(i64::MIN), i128::from(i64::MAX)),
            (0, i128::from(u64::MAX)),
        ] {
            let full = KeyRange::new(min, max).unwrap();
            assert_eq!(full.span(), u64::MAX);
        }
        assert_eq!(
            range.union(KeyRange::new(20, 30).unwrap()),
            KeyRange::new(-5, 30).unwrap()
        );
        for data_type in [DataType::Float64, DataType::Utf8, DataType::Date64] {
            assert!(!KeyRange::supports(&data_type));
        }
        let keys = Int32Array::from(vec![Some(4), None, Some(-7), Some(2)]);
        assert_eq!(KeyRange::of_array(&keys), KeyRange::new(-7, 4));
        assert_eq!(KeyRange::of_array(&Int32Array::new_null(3)), None);
        assert_eq!(
            KeyRange::of_array(&UInt64Array::from(vec![u64::MAX, 1])),
            KeyRange::new(1, i128::from(u64::MAX))
        );
        assert_eq!(KeyRange::of_array(&Float64Array::from(vec![1.0])), None);
        // The key bits are the values sign extended, as the build stores them.
        let mut bits = vec![];
        assert!(key_bits(&keys, &mut bits));
        assert_eq!(bits[0], 4);
        assert_eq!(bits[2], (-7_i64) as u64);
        assert!(!key_bits(&Float64Array::from(vec![1.0]), &mut bits));
    }

    #[test]
    fn stored_keys_are_read_with_their_key_bits() {
        let (build, _) = build_and_probe();
        for (data_type, map) in integer_key_types() {
            let batch = with_keys(&build[1], &data_type, map);
            let layout = layout(batch.schema(), 16);
            let mut data = PartitionedTupleData::try_new(layout, 0).unwrap();
            let chunk = DataChunk::from_record_batch(batch.clone());
            data.append_chunk(&chunk, &hashes(&batch, 1)).unwrap();
            let (collection, _) = data.into_collection().unwrap();
            let reader = KeyReader::try_new(collection.layout()).unwrap();
            let locations = collection.row_locations().collect::<Vec<_>>();
            let rows = collection.row_refs(&locations).unwrap();
            let mut bits = vec![];
            assert!(key_bits(batch.column(0).as_ref(), &mut bits));
            let keys = batch.column(0);
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(row.is_valid(0), keys.is_valid(i));
                if keys.is_valid(i) {
                    assert_eq!(reader.read(row.row), bits[i], "{data_type}, row {i}");
                }
            }
        }
        // Only integer keys are read.
        let floats =
            Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, true)]));
        assert!(KeyReader::try_new(&layout(floats, 16)).is_none());
    }

    type KeyMap = fn(i32) -> i128;

    /// Injective maps of the keys of [`build_and_probe`] to keys of every
    /// integer type, some at the ends of the values of their types.
    fn integer_key_types() -> Vec<(DataType, KeyMap)> {
        vec![
            (DataType::Int8, |key| i128::from(key)),
            (DataType::Int16, |key| i128::from(key) * 300),
            (DataType::Int32, |key| i128::from(key) - 7),
            (DataType::Int64, |key| {
                i128::from(i64::MIN) + 10 + i128::from(key)
            }),
            (DataType::UInt8, |key| i128::from(key + 5)),
            (DataType::UInt16, |key| i128::from(key + 5) * 1000),
            (DataType::UInt32, |key| {
                i128::from(u32::MAX) - 30 + i128::from(key)
            }),
            (DataType::UInt64, |key| {
                i128::from(u64::MAX) - 30 + i128::from(key)
            }),
            (DataType::Date32, |key| i128::from(key) - 1000),
        ]
    }

    #[test]
    fn integer_keys_are_looked_up_without_hashes() {
        let (build, probe_batch) = build_and_probe();
        // Narrow ranges get dense indexes; wide ones hashed tables with a key
        // filter, whose buckets have room for the 17 distinct keys.
        let mut cases = integer_key_types()
            .into_iter()
            .map(|(data_type, map)| (data_type, map, "dense"))
            .collect::<Vec<_>>();
        let wide: [(DataType, KeyMap); 3] = [
            (DataType::Int32, |key| i128::from(key) * 100_000),
            (DataType::Int64, |key| i128::from(key) << 50),
            (DataType::UInt64, |key| i128::from(key + 5) << 58),
        ];
        cases.extend(
            wide.into_iter()
                .map(|(data_type, map)| (data_type, map, "buckets")),
        );
        for (data_type, map, index) in cases {
            let typed_build = build
                .iter()
                .map(|batch| with_keys(batch, &data_type, map))
                .collect::<Vec<_>>();
            let typed_probe = with_keys(&probe_batch, &data_type, map);
            let range = key_range(&typed_build);
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                let expected = expected(&build, &probe_batch, null_equality);
                assert!(expected.len() > 100);
                for (radix_bits, threads) in [(0, 1), (2, 3), (5, 2)] {
                    let table = configured_table(
                        &typed_build,
                        1,
                        null_equality,
                        radix_bits,
                        threads,
                        |b| hashes(b, 1),
                        |builder| builder.with_key_range(range),
                    )
                    .unwrap();
                    let case = format!(
                        "{data_type}, {index}, {null_equality:?}, {radix_bits} bits, {threads} threads"
                    );
                    assert_eq!(table.has_dense_index(), index == "dense", "{case}");
                    assert_eq!(!table.needs_hashes(), index == "dense", "{case}");
                    assert_eq!(table.filters_keys(true), index == "buckets", "{case}");
                    assert_eq!(table.num_rows(), 200);
                    let entries = table.num_entries();
                    assert_eq!(
                        entries,
                        table_entries(&typed_build, null_equality),
                        "{case}"
                    );
                    for max_cost in [1, 7, 2048] {
                        let actual = probe_one_key(&table, &typed_probe, max_cost);
                        assert!(actual.windows(2).all(|w| w[0].0 <= w[1].0), "{case}");
                        if threads == 1 {
                            // The rows of a chain are in build order.
                            assert_eq!(actual, expected, "{case}");
                        } else {
                            assert_eq!(
                                sorted(actual),
                                sorted(expected.clone()),
                                "{case}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The number of rows of `batches` that a probe row can match: all of
    /// them, but those with a NULL key under `NullEqualsNothing`.
    fn table_entries(batches: &[RecordBatch], null_equality: NullEquality) -> usize {
        batches
            .iter()
            .map(|batch| match null_equality {
                NullEquality::NullEqualsNothing => {
                    batch.num_rows() - batch.column(0).null_count()
                }
                NullEquality::NullEqualsNull => batch.num_rows(),
            })
            .sum()
    }

    #[test]
    fn key_filters_reject_the_keys_of_empty_buckets() {
        // 10 keys spread over a range of 9 billion keys: the buckets of the
        // filter have about 17,000 keys each, and hold at most one build key.
        let build = batch_with_values((0..10).map(Some).collect(), 0);
        let build = vec![with_keys(&build, &DataType::Int64, |key| {
            i128::from(key) * 1_000_000_000
        })];
        let range = key_range(&build);
        assert_eq!((range.min(), range.max()), (0, 9_000_000_000));
        let table = configured_table(
            &build,
            1,
            NullEquality::NullEqualsNothing,
            2,
            1,
            |b| hashes(b, 1),
            |builder| builder.with_key_range(range),
        )
        .unwrap();
        assert!(table.filters_keys(true));
        assert!(
            format!("{table:?}").contains("key range and buckets"),
            "{table:?}"
        );
        let filter = table.key_filter.as_ref().unwrap();
        // The keys of the build rows, the keys of their buckets, and other
        // keys of the range or outside of it.
        let bucket = 1 << filter.buckets.as_ref().unwrap().shift;
        for (key, accepted) in [
            (0, true),
            (3_000_000_000, true),
            (9_000_000_000, true),
            (bucket - 1, true),
            (bucket, false),
            (5_000_000_001 + bucket, false),
            (-1, false),
            (9_000_000_001 + bucket, false),
            (i64::MIN, false),
            (i64::MAX, false),
        ] {
            assert_eq!(filter.accepts(key as u64), accepted, "{key}");
        }
        // Only the keys of the build rows match.
        let probe_keys = Int64Array::from(vec![
            Some(4_000_000_000),
            Some(bucket),
            None,
            Some(-1),
            Some(9_000_000_000),
        ]);
        let probe_batch = RecordBatch::try_new(
            build[0].schema(),
            vec![Arc::new(probe_keys), Arc::new(Int64Array::from(vec![0; 5]))],
        )
        .unwrap();
        let actual = probe_one_key(&table, &probe_batch, 100);
        assert_eq!(actual, vec![(0, 4), (4, 9)]);

        // The buckets are optional: without room for them in the memory pool,
        // the filter only checks the range.
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1000));
        let mut data =
            PartitionedTupleData::try_new(layout(build[0].schema(), 7), 2).unwrap();
        let chunk = DataChunk::from_record_batch(build[0].clone());
        data.append_chunk(&chunk, &hashes(&build[0], 1)).unwrap();
        let table = JoinHashTableBuilder::try_new(
            data,
            1,
            NullEquality::NullEqualsNothing,
            MemoryConsumer::new("JoinHashTable").register(&pool),
        )
        .unwrap()
        .with_key_range(range)
        .unwrap()
        .finish(false)
        .unwrap();
        assert!(table.filters_keys(false) && !table.filters_keys(true));
        assert!(!table.has_dense_index());
        assert!(
            format!("{table:?}").contains("\"hashed, key range\""),
            "{table:?}"
        );
        let actual = probe_one_key(&table, &probe_batch, 100);
        assert_eq!(actual, vec![(0, 4), (4, 9)]);
    }

    #[test]
    fn buckets_with_keys_in_most_of_them_are_dropped() {
        // The keys 0 to 15, in 4 buckets of 4 keys, recorded by two tables.
        let filter = |buckets: [Vec<u32>; 2]| {
            let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
            let reservation = MemoryConsumer::new("JoinHashTable").register(&pool);
            let mut tables = buckets.map(|buckets| {
                reservation.grow(buckets.capacity() * size_of::<u32>());
                PartitionTable {
                    buckets,
                    ..Default::default()
                }
            });
            let reader = KeyReader {
                offset: 0,
                width: 4,
                signed: true,
            };
            let plan = KeyIndexPlan {
                range: KeyRange::new(0, 15).unwrap(),
                reader,
                dense: None,
                buckets: Some((2, 4)),
            };
            let filter = key_filter(&plan, &mut tables, &reservation).unwrap();
            assert!(tables.iter().all(|table| table.buckets.is_empty()));
            (filter, reservation.size())
        };
        // Two buckets of four hold a key.
        let (half, reserved) = filter([vec![0, 3], vec![0]]);
        assert_eq!(reserved, size_of::<u64>());
        let accepted = (0..20).filter(|&key| half.accepts(key)).collect::<Vec<_>>();
        assert_eq!(accepted, vec![0, 1, 2, 3, 12, 13, 14, 15]);
        let (most, reserved) = filter([vec![0], vec![1, 3]]);
        assert!(most.buckets.is_none());
        assert_eq!(reserved, 0);
        let accepted = (0..20).filter(|&key| most.accepts(key)).count();
        assert_eq!(accepted, 16);
    }

    #[test]
    fn build_keys_outside_of_their_range_are_internal_errors() {
        let (build, _) = build_and_probe();
        for (map, range) in [
            // A dense index of the keys from 0 (the build keys start at -3).
            (
                (|key| i128::from(key)) as KeyMap,
                KeyRange::new(0, 13).unwrap(),
            ),
            // Hashed tables of keys up to 100,000 (they go up to 1,300,000).
            (
                |key| i128::from(key) * 100_000,
                KeyRange::new(-300_000, 100_000).unwrap(),
            ),
        ] {
            let typed = build
                .iter()
                .map(|batch| with_keys(batch, &DataType::Int64, map))
                .collect::<Vec<_>>();
            let err = configured_table(
                &typed,
                1,
                NullEquality::NullEqualsNothing,
                2,
                2,
                |b| hashes(b, 1),
                |builder| builder.with_key_range(range),
            )
            .unwrap_err();
            assert!(err.to_string().contains("outside of its range"), "{err}");
        }
        // Only a single key of an integer type has a range.
        let strings = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8View, true),
            Field::new("v", DataType::Int64, false),
        ]));
        let data = PartitionedTupleData::try_new(layout(strings, 16), 0).unwrap();
        let builder = JoinHashTableBuilder::try_new(
            data,
            1,
            NullEquality::NullEqualsNothing,
            reservation(),
        )
        .unwrap();
        assert!(
            builder
                .with_key_range(KeyRange::new(0, 1).unwrap())
                .is_err()
        );
        let data = PartitionedTupleData::try_new(layout(schema(), 16), 0).unwrap();
        let builder = JoinHashTableBuilder::try_new(
            data,
            2,
            NullEquality::NullEqualsNothing,
            reservation(),
        )
        .unwrap();
        assert!(
            builder
                .with_key_range(KeyRange::new(0, 1).unwrap())
                .is_err()
        );
    }

    #[test]
    fn small_builds_get_a_single_table() {
        let (build, probe_batch) = build_and_probe();
        let expected = expected(&build, &probe_batch, NullEquality::NullEqualsNothing);
        let data = |radix_bits: u32, batches: &[RecordBatch]| {
            let mut data =
                PartitionedTupleData::try_new(layout(schema(), 7), radix_bits).unwrap();
            for batch in batches {
                let chunk = DataChunk::from_record_batch(batch.clone());
                data.append_chunk(&chunk, &hashes(batch, 1)).unwrap();
            }
            data
        };
        let builder = |radix_bits: u32, batches: &[RecordBatch]| {
            JoinHashTableBuilder::try_new(
                data(radix_bits, batches),
                1,
                NullEquality::NullEqualsNothing,
                reservation(),
            )
            .unwrap()
        };
        // All rows are in one partition if all hashes collide (see
        // `hashes_are_forced_to_collide`).
        let collide = hashes_are_forced_to_collide();
        // 200 rows in up to 32 partitions get a single table, which one
        // thread builds.
        let partitioned = builder(5, &build);
        assert_eq!(partitioned.num_partitions(), 32);
        assert!(collide || partitioned.num_partitions_with_rows() > 1);
        assert_eq!(partitioned.max_threads(), 1);
        let single = builder(5, &build).single_partition_up_to(SMALL_BUILD_ROWS);
        assert_eq!(single.num_partitions(), 1);
        assert_eq!(single.max_threads(), 1);
        assert!(single.build_next_partition().unwrap());
        assert!(!single.build_next_partition().unwrap());
        let table = single.finish(false).unwrap();
        assert_eq!(table.radix_bits(), 0);
        assert_eq!(probe_one_key(&table, &probe_batch, 2048), expected);
        // Larger builds keep their partitions, and get a thread per 4096 rows
        // at most.
        let large = (0..250).map(|b| build[b % 5].clone()).collect::<Vec<_>>();
        let partitioned = builder(3, &large).single_partition_up_to(SMALL_BUILD_ROWS);
        assert_eq!(partitioned.num_rows(), 10_000);
        assert_eq!(partitioned.num_partitions(), 8);
        assert_eq!(partitioned.max_threads(), if collide { 1 } else { 3 });
        // A dense index is built by partition, in parallel.
        let dense = builder(3, &large)
            .with_key_range(key_range(&build))
            .unwrap();
        assert!(dense.has_dense_index());
        assert_eq!(dense.max_threads(), if collide { 1 } else { 3 });
        let mut built = 0;
        while dense.build_next_partition().unwrap() {
            built += 1;
        }
        assert_eq!(built, dense.num_partitions_with_rows());
        let table = dense.finish(false).unwrap();
        assert!(table.has_dense_index());
        // The build rows 50 times: every match 50 times.
        let large_expected = expected
            .iter()
            .flat_map(|&(row, value)| (0..50).map(move |_| (row, value)))
            .collect::<Vec<_>>();
        assert_eq!(
            sorted(probe_one_key(&table, &probe_batch, 2048))
                .into_iter()
                .map(|(row, _)| row)
                .collect::<Vec<_>>(),
            sorted(large_expected)
                .into_iter()
                .map(|(row, _)| row)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_dense_index_memory_is_reserved_before_it_is_allocated() {
        // 200 rows, of which 17 keys from -3 to 13: 17 entries and 200 chains.
        let (build, _) = build_and_probe();
        let data = || {
            let mut data = PartitionedTupleData::try_new(layout(schema(), 7), 2).unwrap();
            for batch in &build {
                let chunk = DataChunk::from_record_batch(batch.clone());
                data.append_chunk(&chunk, &hashes(batch, 1)).unwrap();
            }
            data
        };
        let index = (17 + 200) * size_of::<u64>();
        // The index, and the keys of a partition while it is built (of at most
        // 512 slots of 16 bytes for the 200 rows) if there is room for them
        // (see `PartitionKeys`). Without room for the index, the rows get
        // hashed tables, which need more memory here.
        for (limit, dense, fits) in [
            (0, false, false),
            (index - 1, false, false),
            (index, true, true),
            (index + 8192, true, true),
        ] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
            let reservation = MemoryConsumer::new("JoinHashTable").register(&pool);
            let builder = JoinHashTableBuilder::try_new(
                data(),
                1,
                NullEquality::NullEqualsNothing,
                reservation,
            )
            .unwrap()
            .with_key_range(key_range(&build))
            .unwrap();
            assert_eq!(builder.has_dense_index(), dense, "{limit}");
            let result = builder.finish(false);
            assert!(pool.reserved() <= limit);
            match result {
                Ok(table) => {
                    assert!(fits, "{limit}");
                    assert!(table.has_dense_index());
                    assert_eq!(table.reserved_memory(), index);
                    assert_eq!(table.index_memory_size(), index);
                    drop(table);
                    assert_eq!(pool.reserved(), 0);
                }
                Err(err) => {
                    assert!(!fits, "{limit}: {err}");
                    assert!(err.to_string().contains("JoinHashTable"), "{err}");
                }
            }
        }
        // Hashed tables of 200 rows fit in 16 KiB, a dense index of 50,000 keys
        // does not.
        let wide = build
            .iter()
            .map(|batch| with_keys(batch, &DataType::Int32, |key| i128::from(key) * 3000))
            .collect::<Vec<_>>();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 * 1024));
        let mut data =
            PartitionedTupleData::try_new(layout(wide[0].schema(), 7), 2).unwrap();
        for batch in &wide {
            let chunk = DataChunk::from_record_batch(batch.clone());
            data.append_chunk(&chunk, &hashes(batch, 1)).unwrap();
        }
        let builder = JoinHashTableBuilder::try_new(
            data,
            1,
            NullEquality::NullEqualsNothing,
            MemoryConsumer::new("JoinHashTable").register(&pool),
        )
        .unwrap()
        .with_key_range(key_range(&wide))
        .unwrap();
        assert!(!builder.has_dense_index());
        let table = builder.finish(false).unwrap();
        assert!(!table.has_dense_index() && table.filters_keys(true));
    }

    #[test]
    fn dense_indexes_fit_where_their_entries_and_chains_fit() {
        // A build side of one row: a dense index of one entry and one chain
        // (16 bytes), or a hashed table of 8 slots and one chain (72 bytes).
        // Both fit in 100 bytes: the dense index is built without the table of
        // the keys of its partition (of at least 1,024 bytes, see
        // `PartitionKeys`), for which there is no room.
        let build = batch(vec![Some(7)]);
        let probe_batch = batch(vec![Some(7), Some(8), None, Some(7)]);
        for (dense, reserved) in [(false, 72), (true, 16)] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(100));
            let mut data = PartitionedTupleData::try_new(layout(schema(), 7), 2).unwrap();
            let chunk = DataChunk::from_record_batch(build.clone());
            data.append_chunk(&chunk, &hashes(&build, 1)).unwrap();
            let mut builder = JoinHashTableBuilder::try_new(
                data,
                1,
                NullEquality::NullEqualsNothing,
                MemoryConsumer::new("JoinHashTable").register(&pool),
            )
            .unwrap()
            .single_partition_up_to(SMALL_BUILD_ROWS);
            if dense {
                builder = builder
                    .with_key_range(key_range(std::slice::from_ref(&build)))
                    .unwrap();
            }
            assert_eq!(builder.has_dense_index(), dense);
            let table = builder.finish(false).unwrap();
            assert_eq!(table.has_dense_index(), dense);
            assert_eq!(table.reserved_memory(), reserved, "dense: {dense}");
            assert_eq!(pool.reserved(), reserved, "dense: {dense}");
            assert_eq!(
                probe_one_key(&table, &probe_batch, 100),
                vec![(0, 0), (3, 0)],
                "dense: {dense}"
            );
        }
    }

    #[test]
    fn dense_partitions_without_room_for_their_keys_build_the_index_directly() {
        // 1,000 build rows of the 250 keys `0..250`, every key once in each of
        // 4 batches, probed with the keys from -10 to 259 and a NULL.
        let build = (0..4)
            .map(|b| {
                let keys = (0..250).map(|i| Some((i * 7 + b * 13) % 250)).collect();
                batch_with_values(keys, i64::from(b) * 1000)
            })
            .collect::<Vec<_>>();
        let probe_batch = batch((-10..260).map(Some).chain([None]).collect());
        let expected = expected(&build, &probe_batch, NullEquality::NullEqualsNothing);
        let index = (250 + 1000) * size_of::<u64>();
        // The smallest table of the keys of a partition: 64 slots of 16 bytes,
        // for up to 32 keys.
        let smallest_keys = PartitionKeys::MIN_CAPACITY * 2 * size_of::<u64>();
        // In a single partition (with 250 keys), or in 4 partitions built by 4
        // threads at once (in one with `force_hash_collisions`). The pool has
        // room for the index only: the partitions keep the first rows of their
        // chains in the entries of the index from the start. Or for the index
        // and the smallest table of the keys of a partition, but not for a
        // larger one: a partition with more than 32 keys moves the first rows
        // in its table to the entries once its table is full. Or for all
        // tables.
        for radix_bits in [0, 2] {
            for limit in [index, index + smallest_keys + 100, 100 * index] {
                let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
                let mut data =
                    PartitionedTupleData::try_new(layout(schema(), 64), radix_bits)
                        .unwrap();
                for batch in &build {
                    let chunk = DataChunk::from_record_batch(batch.clone());
                    data.append_chunk(&chunk, &hashes(batch, 1)).unwrap();
                }
                let builder = JoinHashTableBuilder::try_new(
                    data,
                    1,
                    NullEquality::NullEqualsNothing,
                    MemoryConsumer::new("JoinHashTable").register(&pool),
                )
                .unwrap()
                .with_key_range(key_range(&build))
                .unwrap();
                assert!(builder.has_dense_index(), "{limit}");
                std::thread::scope(|scope| {
                    for _ in 0..4 {
                        scope.spawn(|| while builder.build_next_partition().unwrap() {});
                    }
                });
                let table = builder.finish(false).unwrap();
                assert!(table.has_dense_index());
                // The tables of the keys of the partitions are released.
                assert_eq!(table.reserved_memory(), index, "{radix_bits}, {limit}");
                assert_eq!(pool.reserved(), index, "{radix_bits}, {limit}");
                // All matches, in the order of the build rows.
                assert_eq!(
                    probe_one_key(&table, &probe_batch, 300),
                    expected,
                    "{radix_bits}, {limit}"
                );
            }
        }
    }
}
