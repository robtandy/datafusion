// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PartitionedTupleData`]: rows radix-partitioned by their hash into
//! `2^radix_bits` [`TupleDataCollection`]s (DuckDB's
//! `RadixPartitionedTupleData`).
//!
//! The partition of a row is given by the upper `radix_bits` bits of its
//! 64-bit hash ([`PartitionedTupleData::partition_index`]). Rows with equal
//! keys have equal hashes and therefore end up in the same partition, so
//! that the partitions can be processed independently, e.g. by different
//! threads.
//!
//! The typical use in a sink:
//!
//! 1. every thread scatters its input into a thread-local
//!    `PartitionedTupleData` ([`PartitionedTupleData::append_chunk`]);
//! 2. in `Sink::combine`, the thread-local partitions are merged into the
//!    global partitions with [`PartitionedTupleData::combine`], which moves
//!    the blocks of each partition (O(1) per block, no row is copied);
//! 3. in `Sink::finalize` or in the source, every partition is processed
//!    on its own (e.g. a hash table is built per partition).

use std::ops::Range;
use std::sync::Arc;

use datafusion_common::{Result, internal_err, resources_datafusion_err};

use super::collection::{
    RowLocation, ScatterSource, TupleDataCollection, check_chunk, check_hashes,
    physical_rows,
};
use super::layout::TupleDataLayout;
use crate::chunk::DataChunk;

/// The location of a row in a [`PartitionedTupleData`]: its partition and
/// its location in the collection of the partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionedRowLocation {
    /// The index of the partition.
    pub partition: usize,
    /// The location of the row in the collection of the partition.
    pub location: RowLocation,
}

/// Rows radix-partitioned into `2^radix_bits` [`TupleDataCollection`]s by
/// the upper `radix_bits` bits of their hash. See the
/// [module documentation](self).
#[derive(Debug)]
pub struct PartitionedTupleData {
    layout: Arc<TupleDataLayout>,
    radix_bits: u32,
    partitions: Vec<TupleDataCollection>,
    /// Buffers reused by [`Self::append_chunk`].
    scratch: Scratch,
}

#[derive(Debug, Default)]
struct Scratch {
    /// The partition of every logical row of the chunk.
    partition_of: Vec<u32>,
    /// The first entry of every partition in `rows` (and one more entry,
    /// the number of rows).
    starts: Vec<usize>,
    /// The next free entry of every partition in `rows`.
    next: Vec<usize>,
    /// The physical rows of the chunk, grouped by partition.
    rows: Vec<u32>,
    /// The hashes of `rows`.
    hashes: Vec<u64>,
    /// The logical rows of `rows`.
    logical: Vec<u32>,
    /// The locations of the rows appended to one partition.
    locations: Vec<RowLocation>,
}

impl PartitionedTupleData {
    /// The maximum number of radix bits (4096 partitions).
    pub const MAX_RADIX_BITS: u32 = 12;

    /// Create `2^radix_bits` empty partitions of rows with layout `layout`.
    ///
    /// `radix_bits` must be at most [`Self::MAX_RADIX_BITS`]; with `0` radix
    /// bits there is a single partition.
    pub fn try_new(layout: Arc<TupleDataLayout>, radix_bits: u32) -> Result<Self> {
        if radix_bits > Self::MAX_RADIX_BITS {
            return internal_err!(
                "PartitionedTupleData supports at most {} radix bits, got {radix_bits}",
                Self::MAX_RADIX_BITS
            );
        }
        let partitions = (0..1_usize << radix_bits)
            .map(|_| TupleDataCollection::new(Arc::clone(&layout)))
            .collect();
        Ok(Self {
            layout,
            radix_bits,
            partitions,
            scratch: Scratch::default(),
        })
    }

    /// The partition of a row with hash `hash`, for `2^radix_bits`
    /// partitions: the upper `radix_bits` bits of the hash,
    /// `hash >> (64 - radix_bits)`.
    #[inline]
    pub fn partition_index(hash: u64, radix_bits: u32) -> usize {
        debug_assert!(radix_bits <= Self::MAX_RADIX_BITS);
        if radix_bits == 0 {
            0
        } else {
            (hash >> (u64::BITS - radix_bits)) as usize
        }
    }

    /// The layout of the rows.
    pub fn layout(&self) -> &Arc<TupleDataLayout> {
        &self.layout
    }

    /// The number of radix bits.
    pub fn radix_bits(&self) -> u32 {
        self.radix_bits
    }

    /// The number of partitions, `2^radix_bits`.
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// The partitions.
    pub fn partitions(&self) -> &[TupleDataCollection] {
        &self.partitions
    }

    /// The partition `partition`.
    ///
    /// # Panics
    ///
    /// Panics if `partition >= self.num_partitions()`.
    pub fn partition(&self, partition: usize) -> &TupleDataCollection {
        &self.partitions[partition]
    }

    /// The partitions, mutably (e.g. to take a partition out with
    /// [`std::mem::replace`] when it is finalized).
    pub fn partitions_mut(&mut self) -> &mut [TupleDataCollection] {
        &mut self.partitions
    }

    /// The partitions, by value.
    pub fn into_partitions(self) -> Vec<TupleDataCollection> {
        self.partitions
    }

    /// Move the rows of all partitions into a single collection, partition
    /// after partition, with [`TupleDataCollection::combine`] (O(1) per
    /// block, no row is copied). Also returns the range of blocks of every
    /// partition in the collection, e.g. to scan the partitions in parallel
    /// with [`TupleDataScanState::with_block_range`].
    ///
    /// This allows building a hash table per partition whose entries refer
    /// to rows of a single collection, which a [`RowMatcher`] can then
    /// compare the probe rows of all partitions with.
    ///
    /// [`TupleDataScanState::with_block_range`]: super::TupleDataScanState::with_block_range
    /// [`RowMatcher`]: super::RowMatcher
    pub fn into_collection(self) -> Result<(TupleDataCollection, Vec<Range<usize>>)> {
        let mut collection = TupleDataCollection::new(self.layout);
        let mut blocks = Vec::with_capacity(self.partitions.len());
        for partition in self.partitions {
            let start = collection.num_blocks();
            collection.combine(partition)?;
            blocks.push(start..collection.num_blocks());
        }
        Ok((collection, blocks))
    }

    /// The number of rows in all partitions.
    pub fn len(&self) -> usize {
        self.partitions.iter().map(TupleDataCollection::len).sum()
    }

    /// Returns `true` if no partition has rows.
    pub fn is_empty(&self) -> bool {
        self.partitions.iter().all(TupleDataCollection::is_empty)
    }

    /// The number of bytes allocated by all partitions, for memory
    /// accounting.
    pub fn memory_size(&self) -> usize {
        size_of::<Self>()
            + self.partitions.capacity() * size_of::<TupleDataCollection>()
            + self
                .partitions
                .iter()
                .map(|p| p.memory_size() - size_of::<TupleDataCollection>())
                .sum::<usize>()
            + self.scratch.memory_size()
    }

    /// Scatter the rows of `chunk` into their partitions. `hashes` holds the
    /// hash of every (logical) row of the chunk; the hashes are stored in
    /// the rows if the layout has a hash slot.
    ///
    /// If the chunk has a selection, only the selected rows are stored; the
    /// chunk is not compacted. Within a partition, rows are stored in the
    /// order of the chunk.
    pub fn append_chunk(&mut self, chunk: &DataChunk, hashes: &[u64]) -> Result<()> {
        self.append(chunk, hashes, None)
    }

    /// Like [`Self::append_chunk`], but also returns the locations of the
    /// new rows, one for every (logical) row of the chunk.
    pub fn append_chunk_with_locations(
        &mut self,
        chunk: &DataChunk,
        hashes: &[u64],
    ) -> Result<Vec<PartitionedRowLocation>> {
        let mut locations = vec![];
        self.append(chunk, hashes, Some(&mut locations))?;
        Ok(locations)
    }

    fn append(
        &mut self,
        chunk: &DataChunk,
        hashes: &[u64],
        mut locations: Option<&mut Vec<PartitionedRowLocation>>,
    ) -> Result<()> {
        if hashes.len() != chunk.len() {
            return internal_err!(
                "Got {} hashes for a DataChunk with {} rows",
                hashes.len(),
                chunk.len()
            );
        }
        let stored_hashes = self.layout.has_hash().then_some(hashes);
        check_chunk(&self.layout, chunk)?;
        check_hashes(&self.layout, chunk, stored_hashes)?;
        if chunk.is_empty() {
            return Ok(());
        }
        let source = ScatterSource::try_new(&self.layout, chunk)?;
        let physical = physical_rows(chunk)?;
        if let Some(locations) = locations.as_deref_mut() {
            locations.clear();
            locations.try_reserve(chunk.len()).map_err(|e| {
                resources_datafusion_err!("Failed to allocate row locations: {e}")
            })?;
            locations.resize(chunk.len(), PartitionedRowLocation::default());
        }

        let scratch = &mut self.scratch;
        scratch.locations.clear();
        if self.radix_bits == 0 {
            let partition_locations =
                locations.is_some().then_some(&mut scratch.locations);
            let result = self.partitions[0].append_rows(
                &source,
                &physical,
                stored_hashes,
                partition_locations,
            );
            if let Some(locations) = locations {
                for (location, row) in
                    locations.iter_mut().zip(scratch.locations.drain(..))
                {
                    *location = PartitionedRowLocation {
                        partition: 0,
                        location: row,
                    };
                }
            }
            return result;
        }

        // Counting sort of the rows by partition (stable, so that the rows of
        // a partition keep the order of the chunk).
        let num_partitions = self.partitions.len();
        scratch.partition_of.clear();
        scratch.partition_of.extend(
            hashes
                .iter()
                .map(|&hash| Self::partition_index(hash, self.radix_bits) as u32),
        );
        scratch.starts.clear();
        scratch.starts.resize(num_partitions + 1, 0);
        for &partition in &scratch.partition_of {
            scratch.starts[partition as usize + 1] += 1;
        }
        for partition in 0..num_partitions {
            scratch.starts[partition + 1] += scratch.starts[partition];
        }
        let n = chunk.len();
        scratch.rows.clear();
        scratch.rows.resize(n, 0);
        scratch.hashes.clear();
        scratch.hashes.resize(n, 0);
        scratch.logical.clear();
        scratch.logical.resize(n, 0);
        scratch.next.clear();
        scratch
            .next
            .extend_from_slice(&scratch.starts[..num_partitions]);
        for (logical, &partition) in scratch.partition_of.iter().enumerate() {
            let entry = &mut scratch.next[partition as usize];
            scratch.rows[*entry] = physical[logical];
            scratch.hashes[*entry] = hashes[logical];
            scratch.logical[*entry] = logical as u32;
            *entry += 1;
        }

        for partition in 0..num_partitions {
            let range = scratch.starts[partition]..scratch.starts[partition + 1];
            if range.is_empty() {
                continue;
            }
            scratch.locations.clear();
            let partition_locations =
                locations.is_some().then_some(&mut scratch.locations);
            self.partitions[partition].append_rows(
                &source,
                &scratch.rows[range.clone()],
                stored_hashes.map(|_| &scratch.hashes[range.clone()]),
                partition_locations,
            )?;
            if let Some(locations) = locations.as_deref_mut() {
                for (&logical, &location) in
                    scratch.logical[range].iter().zip(&scratch.locations)
                {
                    locations[logical as usize] = PartitionedRowLocation {
                        partition,
                        location,
                    };
                }
            }
        }
        scratch.locations.clear();
        Ok(())
    }

    /// Move all rows of `other` into this partitioned data, partition by
    /// partition, with [`TupleDataCollection::combine`]: the blocks of
    /// every partition of `other` are moved, no row is copied.
    ///
    /// Both must have the same number of radix bits and
    /// [compatible](TupleDataLayout::is_compatible) layouts.
    pub fn combine(&mut self, other: PartitionedTupleData) -> Result<()> {
        if self.radix_bits != other.radix_bits {
            return internal_err!(
                "Cannot combine PartitionedTupleData with {} and {} radix bits",
                self.radix_bits,
                other.radix_bits
            );
        }
        if !self.layout.is_compatible(&other.layout) {
            return internal_err!(
                "Cannot combine PartitionedTupleData with different layouts"
            );
        }
        for (partition, other) in self.partitions.iter_mut().zip(other.partitions) {
            partition.combine(other)?;
        }
        Ok(())
    }
}

impl Scratch {
    fn memory_size(&self) -> usize {
        self.partition_of.capacity() * size_of::<u32>()
            + (self.starts.capacity() + self.next.capacity()) * size_of::<usize>()
            + self.rows.capacity() * size_of::<u32>()
            + self.hashes.capacity() * size_of::<u64>()
            + self.logical.capacity() * size_of::<u32>()
            + self.locations.capacity() * size_of::<RowLocation>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{RecordBatch, UInt32Array};
    use arrow::compute::{concat_batches, take_record_batch};
    use datafusion_common::hash_utils::{RandomState, create_hashes};

    use crate::tuple_data::TupleDataScanState;
    use crate::tuple_data::test_data::{
        all_types_batch, assert_identical, chunk_with_selection,
    };

    fn hashes_of(batch: &RecordBatch, columns: &[usize]) -> Vec<u64> {
        let arrays = columns
            .iter()
            .map(|&c| Arc::clone(batch.column(c)))
            .collect::<Vec<_>>();
        let mut hashes = vec![0; batch.num_rows()];
        create_hashes(&arrays, &RandomState::with_seed(7), &mut hashes).unwrap();
        hashes
    }

    fn layout(
        batch: &RecordBatch,
        with_hash: bool,
        rows_per_block: usize,
    ) -> Arc<TupleDataLayout> {
        Arc::new(
            TupleDataLayout::try_new(batch.schema(), with_hash)
                .unwrap()
                .with_rows_per_block(rows_per_block)
                .unwrap(),
        )
    }

    /// The rows of `batch` (with hashes `hashes`) that belong to `partition`.
    fn expected_partition(
        batch: &RecordBatch,
        hashes: &[u64],
        radix_bits: u32,
        partition: usize,
    ) -> RecordBatch {
        let indices = hashes
            .iter()
            .enumerate()
            .filter(|(_, h)| {
                PartitionedTupleData::partition_index(**h, radix_bits) == partition
            })
            .map(|(i, _)| i as u32)
            .collect::<UInt32Array>();
        take_record_batch(batch, &indices).unwrap()
    }

    fn partition_batch(partition: &TupleDataCollection) -> RecordBatch {
        let locations = partition.row_locations().collect::<Vec<_>>();
        partition.gather(&locations).unwrap()
    }

    #[test]
    fn partition_index_uses_the_upper_hash_bits() {
        let hash = 0xF0E1_D2C3_B4A5_9687_u64;
        assert_eq!(PartitionedTupleData::partition_index(hash, 0), 0);
        assert_eq!(PartitionedTupleData::partition_index(hash, 1), 1);
        assert_eq!(PartitionedTupleData::partition_index(hash, 4), 0xF);
        assert_eq!(PartitionedTupleData::partition_index(hash, 8), 0xF0);
        assert_eq!(PartitionedTupleData::partition_index(hash, 12), 0xF0E);
        assert_eq!(PartitionedTupleData::partition_index(u64::MAX, 12), 4095);
        assert_eq!(PartitionedTupleData::partition_index(0, 12), 0);
        assert_eq!(PartitionedTupleData::partition_index(1 << 63, 3), 4);

        let batch = all_types_batch(1);
        assert!(PartitionedTupleData::try_new(layout(&batch, false, 10), 13).is_err());
        let data = PartitionedTupleData::try_new(layout(&batch, false, 10), 12).unwrap();
        assert_eq!(data.num_partitions(), 4096);
        assert_eq!(data.radix_bits(), 12);
        assert!(data.is_empty());
    }

    #[test]
    fn rows_are_scattered_to_their_partitions() {
        let batch = all_types_batch(1000);
        let hashes = hashes_of(&batch, &[2, 34]);
        for radix_bits in [0, 1, 3, 6] {
            let mut data =
                PartitionedTupleData::try_new(layout(&batch, true, 50), radix_bits)
                    .unwrap();
            for offset in (0..1000).step_by(300) {
                let len = 300.min(1000 - offset);
                let chunk = DataChunk::from_record_batch(batch.slice(offset, len));
                data.append_chunk(&chunk, &hashes[offset..offset + len])
                    .unwrap();
            }
            assert_eq!(data.len(), 1000);
            assert_eq!(data.num_partitions(), 1 << radix_bits);
            for (i, partition) in data.partitions().iter().enumerate() {
                let expected = expected_partition(&batch, &hashes, radix_bits, i);
                assert_eq!(partition.len(), expected.num_rows());
                // Every row is in the partition of its hash, in the order of
                // the input, with its hash.
                let locations = partition.row_locations().collect::<Vec<_>>();
                for hash in partition.gather_hashes(&locations).unwrap() {
                    assert_eq!(
                        PartitionedTupleData::partition_index(hash, radix_bits),
                        i
                    );
                }
                assert_identical(&partition_batch(partition), &expected);
            }
        }
    }

    #[test]
    fn selection_is_respected() {
        let batch = all_types_batch(500);
        let hashes = hashes_of(&batch, &[3]);
        let selection = (0..500_u32)
            .rev()
            .filter(|i| i % 4 != 0)
            .collect::<Vec<_>>();
        let chunk = chunk_with_selection(&batch, selection.clone());
        let compacted = chunk.compact().unwrap();
        let selected_hashes = selection
            .iter()
            .map(|&i| hashes[i as usize])
            .collect::<Vec<_>>();

        let mut data =
            PartitionedTupleData::try_new(layout(&batch, false, 64), 4).unwrap();
        let locations = data
            .append_chunk_with_locations(&chunk, &selected_hashes)
            .unwrap();
        assert_eq!(locations.len(), selection.len());
        for (i, partition) in data.partitions().iter().enumerate() {
            let expected = expected_partition(&compacted, &selected_hashes, 4, i);
            assert_identical(&partition_batch(partition), &expected);
        }
        // Every location points at its row.
        for (row, location) in locations.iter().enumerate() {
            assert_eq!(
                location.partition,
                PartitionedTupleData::partition_index(selected_hashes[row], 4)
            );
            let gathered = data
                .partition(location.partition)
                .gather(&[location.location])
                .unwrap();
            assert_identical(&gathered, &compacted.slice(row, 1));
        }

        // With a single partition.
        let mut data =
            PartitionedTupleData::try_new(layout(&batch, true, 64), 0).unwrap();
        let locations = data
            .append_chunk_with_locations(&chunk, &selected_hashes)
            .unwrap();
        assert!(locations.iter().all(|l| l.partition == 0));
        let locations = locations.iter().map(|l| l.location).collect::<Vec<_>>();
        assert_identical(&data.partition(0).gather(&locations).unwrap(), &compacted);
        assert_eq!(
            data.partition(0).gather_hashes(&locations).unwrap(),
            selected_hashes
        );
    }

    #[test]
    fn thread_local_partitions_are_combined_without_copying() {
        let batch = all_types_batch(3000);
        let hashes = hashes_of(&batch, &[4, 5]);
        let radix_bits = 3;
        let layout = layout(&batch, true, 32);

        // Four "threads" scatter disjoint parts of the input.
        let mut locals = (0..4)
            .map(|_| {
                PartitionedTupleData::try_new(Arc::clone(&layout), radix_bits).unwrap()
            })
            .collect::<Vec<_>>();
        for (chunk, offset) in (0..3000).step_by(250).enumerate() {
            let local = &mut locals[chunk % 4];
            local
                .append_chunk(
                    &DataChunk::from_record_batch(batch.slice(offset, 250)),
                    &hashes[offset..offset + 250],
                )
                .unwrap();
        }
        let mut global =
            PartitionedTupleData::try_new(Arc::clone(&layout), radix_bits).unwrap();
        let mut expected_blocks = vec![0; global.num_partitions()];
        let mut expected_batches = vec![vec![]; global.num_partitions()];
        for local in locals {
            for (i, partition) in local.partitions().iter().enumerate() {
                expected_blocks[i] += partition.num_blocks();
                expected_batches[i].push(partition_batch(partition));
            }
            global.combine(local).unwrap();
        }
        assert_eq!(global.len(), 3000);
        for (i, partition) in global.partitions().iter().enumerate() {
            // Blocks are moved, so the number of blocks adds up.
            assert_eq!(partition.num_blocks(), expected_blocks[i]);
            let expected = concat_batches(&batch.schema(), &expected_batches[i]).unwrap();
            assert_identical(&partition_batch(partition), &expected);
            let locations = partition.row_locations().collect::<Vec<_>>();
            assert!(
                partition
                    .gather_hashes(&locations)
                    .unwrap()
                    .iter()
                    .all(|&h| PartitionedTupleData::partition_index(h, radix_bits) == i)
            );
        }
        assert!(global.memory_size() > 0);

        let other =
            PartitionedTupleData::try_new(Arc::clone(&layout), radix_bits + 1).unwrap();
        assert!(global.combine(other).is_err());

        // All partitions in a single collection, each in a range of blocks.
        let partition_batches = global
            .partitions()
            .iter()
            .map(partition_batch)
            .collect::<Vec<_>>();
        let blocks_per_partition = global
            .partitions()
            .iter()
            .map(TupleDataCollection::num_blocks)
            .collect::<Vec<_>>();
        let (collection, blocks) = global.into_collection().unwrap();
        assert_eq!(collection.len(), 3000);
        assert_eq!(blocks.len(), 8);
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[7].end, collection.num_blocks());
        for (i, range) in blocks.iter().enumerate() {
            assert_eq!(range.len(), blocks_per_partition[i]);
            let mut state = TupleDataScanState::new().with_block_range(range.clone());
            let mut scanned = vec![];
            while let Some(chunk) = collection.scan_chunk(&mut state).unwrap() {
                scanned.push(chunk.into_record_batch().unwrap());
            }
            let scanned = concat_batches(&batch.schema(), &scanned).unwrap();
            assert_identical(&scanned, &partition_batches[i]);
        }

        let data = PartitionedTupleData::try_new(Arc::clone(&layout), 2).unwrap();
        let partitions = data.into_partitions();
        assert_eq!(partitions.len(), 4);
        assert!(partitions.iter().all(TupleDataCollection::is_empty));
    }

    #[test]
    fn invalid_hashes_and_empty_chunks() {
        let batch = all_types_batch(10);
        let mut data =
            PartitionedTupleData::try_new(layout(&batch, false, 10), 2).unwrap();
        let chunk = DataChunk::from_record_batch(batch.clone());
        let err = data.append_chunk(&chunk, &[1, 2, 3]).unwrap_err();
        assert!(err.to_string().contains("3 hashes"), "{err}");
        data.append_chunk(&DataChunk::empty(batch.schema()), &[])
            .unwrap();
        data.append_chunk(&chunk_with_selection(&batch, vec![]), &[])
            .unwrap();
        assert!(data.is_empty());
        assert!(
            data.append_chunk_with_locations(&chunk_with_selection(&batch, vec![]), &[])
                .unwrap()
                .is_empty()
        );
        assert!(
            data.partitions_mut()
                .iter()
                .all(TupleDataCollection::is_empty)
        );
        assert_eq!(data.layout().num_columns(), batch.num_columns());
    }
}
