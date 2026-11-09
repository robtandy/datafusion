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
//! [`JoinHashTableBuilder::finish`] assembles the [`JoinHashTable`].
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

use arrow::array::ArrayRef;
use arrow::buffer::NullBuffer;
use arrow::datatypes::SchemaRef;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::{
    NullEquality, Result, exec_err, internal_err, resources_datafusion_err,
};

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::tuple_data::{
    PartitionedTupleData, RowLocation, RowMatcher, TupleDataCollection, TupleDataLayout,
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
}

impl PartitionTable {
    fn memory_size(&self) -> usize {
        (self.entries.capacity() + self.next.capacity()) * size_of::<u64>()
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
/// keys `matcher` compares (`key_schema` is the schema of the keys).
///
/// The rows are inserted in reverse order and prepended to their chain, so
/// that the rows of a chain are in the order of the collection. A row whose
/// salt matches the salt of an entry is compared with the first row of the
/// chain of that entry. The comparisons are vectorized: the rows are
/// inserted in batches, in rounds that stop every row at its next empty
/// entry, which it claims, or at its next entry with its salt, whose chain
/// it joins if its keys are equal (otherwise it continues after that entry
/// in the next round).
fn build_partition(
    collection: &TupleDataCollection,
    range: &PartitionRange,
    matcher: &RowMatcher,
    key_schema: &SchemaRef,
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
    let mut locations = Vec::new();
    let mut hashes = Vec::new();
    // The rows to insert (by index in the partition).
    let mut inserted = Vec::new();
    locations
        .try_reserve_exact(num_rows)
        .and_then(|()| hashes.try_reserve_exact(num_rows))
        .and_then(|()| inserted.try_reserve_exact(num_rows))
        .map_err(|e| {
            resources_datafusion_err!("Failed to allocate a JoinHashTable partition: {e}")
        })?;
    for block in range.blocks.clone() {
        let start = locations.len();
        locations.extend(
            (0..collection.block_len(block)).map(|row| RowLocation::new(block, row)),
        );
        for (i, row) in collection.row_refs(&locations[start..])?.iter().enumerate() {
            hashes.push(row.hash(hash_offset));
            // Under `NullEqualsNothing`, rows with a NULL key never match:
            // they are kept (e.g. for the unmatched rows of an outer join),
            // but not looked up.
            if null_equality == NullEquality::NullEqualsNull
                || (0..num_keys).all(|key| row.is_valid(key))
            {
                inserted.push(start + i);
            }
        }
    }
    if locations.len() != num_rows {
        return internal_err!(
            "JoinHashTable partition has {} rows, expected {num_rows}",
            locations.len()
        );
    }
    let num_entries = inserted.len();
    let capacity = if num_entries == 0 {
        0
    } else {
        num_entries
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| {
                resources_datafusion_err!(
                    "JoinHashTable partition with {num_entries} rows is too large"
                )
            })?
            .max(MIN_CAPACITY)
    };
    let mut entries = zeroed(capacity, "a JoinHashTable")?;
    let mut next = zeroed(num_rows, "the chains of a JoinHashTable")?;
    let mask = capacity.wrapping_sub(1);
    let key_columns = (0..num_keys).collect::<Vec<_>>();
    inserted.reverse();
    // The rows of the current round, with the slot at which they continue.
    let mut pending = Vec::new();
    // The rows that stopped at an entry with their salt, with that entry,
    // and the first row of its chain.
    let mut compared = Vec::new();
    let mut heads = Vec::new();
    let mut no_match = Vec::new();
    for batch in inserted.chunks(DEFAULT_VECTOR_SIZE) {
        pending.clear();
        pending.extend(batch.iter().map(|&row| (row, hashes[row] as usize & mask)));
        while !pending.is_empty() {
            compared.clear();
            heads.clear();
            for &(row, mut slot) in &pending {
                let salt = salt(hashes[row]);
                loop {
                    let current = entries[slot];
                    if current == 0 {
                        // The first row of a new chain.
                        entries[slot] = entry(salt, locations[row]);
                        break;
                    }
                    if entry_salt(current) == salt {
                        compared.push((row, slot));
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
                .map(|&(row, _)| locations[row])
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
                // Prepend the row to the chain of the entry, whose first row
                // may have changed in this round, to a row with equal keys.
                let (row, slot) = compared[pair as usize];
                next[row] = pack(entry_location(entries[slot]));
                entries[slot] = entry(salt(hashes[row]), locations[row]);
            }
            pending.clear();
            pending.extend(no_match.iter().map(|&pair| {
                let (row, slot) = compared[pair as usize];
                (row, (slot + 1) & mask)
            }));
        }
    }
    Ok(PartitionTable {
        entries,
        mask,
        next,
        first_row: range.rows.start,
        num_entries,
    })
}

/// Builds a [`JoinHashTable`] from the combined build rows of all threads.
///
/// The pointer tables of the radix partitions are independent, so several
/// threads can build them in parallel: every call of
/// [`Self::build_next_partition`] (from any thread) builds the table of a
/// partition that no other call claimed. [`Self::finish`] builds the tables
/// that were not built yet and returns the [`JoinHashTable`].
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
}

impl fmt::Debug for JoinHashTableBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JoinHashTableBuilder")
            .field("rows", &self.collection.len())
            .field("radix_bits", &self.radix_bits)
            .field("num_keys", &self.matcher.key_columns().len())
            .finish()
    }
}

impl JoinHashTableBuilder {
    /// A builder of the table of the rows of `data`, whose first `num_keys`
    /// columns are the join keys, compared with `null_equality`. The layout
    /// of the rows must have a hash slot holding the hash of the keys (see
    /// [`hash_join_keys`]), by which the rows were partitioned.
    pub fn try_new(
        data: PartitionedTupleData,
        num_keys: usize,
        null_equality: NullEquality,
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
        })
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

    /// Build the table of a partition with rows that no other call claimed
    /// yet. Returns `false` once every partition is claimed.
    pub fn build_next_partition(&self) -> Result<bool> {
        loop {
            let partition = self.claimed.fetch_add(1, Ordering::Relaxed);
            let Some(range) = self.partitions.get(partition) else {
                return Ok(false);
            };
            if range.rows.is_empty() {
                continue;
            }
            let table = build_partition(
                &self.collection,
                range,
                &self.matcher,
                &self.key_schema,
            )?;
            if self.built[partition].set(table).is_err() {
                return internal_err!(
                    "The table of partition {partition} was built twice"
                );
            }
            return Ok(true);
        }
    }

    /// Build the tables that were not built yet and return the
    /// [`JoinHashTable`]. If `track_matches`, the table tracks which rows
    /// found a match (see [`JoinHashTable::mark_matched`]).
    pub fn finish(self, track_matches: bool) -> Result<JoinHashTable> {
        let Self {
            collection,
            matcher,
            key_schema,
            radix_bits,
            partitions,
            built,
            claimed: _,
        } = self;
        let tables = built
            .into_iter()
            .zip(&partitions)
            .map(|(built, range)| match built.into_inner() {
                Some(table) => Ok(table),
                None => build_partition(&collection, range, &matcher, &key_schema),
            })
            .collect::<Result<Vec<_>>>()?;
        let num_entries = tables.iter().map(|table| table.num_entries).sum();
        let matched = track_matches
            .then(|| MatchedRows::try_new(collection.len()))
            .transpose()?;
        Ok(JoinHashTable {
            collection,
            matcher,
            radix_bits,
            partitions: tables,
            num_entries,
            matched,
        })
    }
}

/// One bit per build row (by dense row index), set by any thread once the
/// row found a match.
struct MatchedRows {
    words: Vec<AtomicU64>,
}

impl MatchedRows {
    fn try_new(num_rows: usize) -> Result<Self> {
        let num_words = num_rows.div_ceil(64);
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
    partitions: Vec<PartitionTable>,
    /// The number of rows in the chains (the rows that can match).
    num_entries: usize,
    /// The rows that found a match, if tracked.
    matched: Option<MatchedRows>,
}

impl fmt::Debug for JoinHashTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JoinHashTable")
            .field("rows", &self.collection.len())
            .field("entries", &self.num_entries)
            .field("radix_bits", &self.radix_bits)
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
    ) -> Result<Self> {
        JoinHashTableBuilder::try_new(data, num_keys, null_equality)?
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

    /// Whether the table tracks the rows that found a match.
    pub fn tracks_matches(&self) -> bool {
        self.matched.is_some()
    }

    /// The memory used by the pointer tables, the chains and the matched
    /// rows, i.e. by the table without its rows (see
    /// [`TupleDataCollection::memory_size`]).
    pub fn index_memory_size(&self) -> usize {
        self.partitions
            .iter()
            .map(PartitionTable::memory_size)
            .sum::<usize>()
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
    /// that entry in the next round.
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
            ..
        } = scan;
        let num_rows = hashes.len();
        chains.clear();
        chains.resize(num_rows, 0);
        if self.is_empty() {
            return Ok(());
        }
        if probe_keys.len() != num_rows || probe_keys.selection().is_some() {
            return internal_err!(
                "The probe keys of a join must be {num_rows} rows without a selection"
            );
        }
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
            if valid.is_some_and(|valid| valid.is_null(row)) {
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
    /// selection), with the hashes `hashes` (see [`hash_join_keys`]). The
    /// rows that are null in `valid` have no matches.
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
            // All rows of a chain are in the table of the same partition.
            let partition = table.partition(self.hashes[row]);
            let probe_row = row as u32;
            loop {
                let location = unpack(next);
                probe_rows.push(probe_row);
                matches.push(location);
                cost += 1;
                next = partition.next
                    [table.collection.row_index(location) - partition.first_row];
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
        Array, AsArray, Float64Array, Int32Array, Int64Array, RecordBatch,
        StringViewArray,
    };
    use arrow::datatypes::{DataType, Field, Int32Type, Int64Type, Schema, SchemaRef};

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
        let builder =
            JoinHashTableBuilder::try_new(global, num_keys, null_equality).unwrap();
        // Build a partition "in parallel", the others in `finish`.
        builder.build_next_partition().unwrap();
        builder.finish(true).unwrap()
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
        assert_ne!(hashes[0], hashes[2]);
        assert_ne!(hashes[0], hashes[3]);
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
    fn empty_tables_and_invalid_builders() {
        let layout = layout(schema(), 16);
        let data = PartitionedTupleData::try_new(Arc::clone(&layout), 3).unwrap();
        let table =
            JoinHashTable::try_new(data, 1, NullEquality::NullEqualsNothing, false)
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
        let builder =
            JoinHashTableBuilder::try_new(data, 1, NullEquality::NullEqualsNothing)
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
            JoinHashTableBuilder::try_new(data, 1, NullEquality::NullEqualsNothing)
                .is_err()
        );
        for num_keys in [0, 3] {
            let data = PartitionedTupleData::try_new(Arc::clone(&layout), 0).unwrap();
            let builder = JoinHashTableBuilder::try_new(
                data,
                num_keys,
                NullEquality::NullEqualsNothing,
            );
            assert!(builder.is_err(), "{num_keys} keys");
        }
    }
}
