// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`RadixHashAggregate`]: the grouped aggregation of a
//! [`PhysicalHashAggregate`](super::PhysicalHashAggregate), with
//! thread-local pre-aggregation and radix-partitioned parallel finalization
//! (DuckDB's `RadixPartitionedHashTable`).
//!
//! ```text
//!  sink (every thread)               combine (every thread)    get_data (every thread)
//!
//!  chunks -> GroupedHashTable -----> the same flush of the  -> claim partition p,
//!            (thread-local)    |     remaining groups          merge the states of
//!                              |                               its groups, and
//!           when full: flush   |                               aggregate its rows,
//!           = hash the keys of +-->  64 radix partitions       from all threads,
//!           the groups once,   +-->  (hash >> 58)              into a new
//!           scatter the groups |                               GroupedHashTable,
//!                              |                               evaluate, emit chunks
//!  chunks of a thread whose    |
//!  groups hardly repeat -------+
//!  = hash the keys of the rows once,
//!    scatter the rows (bypass)
//! ```
//!
//! * `sink`: every thread aggregates the rows of its chunks into its own
//!   [`GroupedHashTable`] (DataFusion's partial aggregation), without any
//!   synchronization. The table has a bounded capacity, so that it stays
//!   cache resident: when it is full, it is *flushed*, as DuckDB abandons its
//!   thread-local tables (or it grows, if the groups of the thread keep
//!   coming back, see `RadixLocalState::aggregate`).
//! * A thread whose full table has almost as many groups as it aggregated
//!   rows (e.g. for a `GROUP BY` of an id) gains nothing from aggregating
//!   them, it only builds a hash table for them in vain: it stops, like
//!   DataFusion skips its partial aggregation. Its rows *bypass* the table:
//!   the keys of their groups are hashed once, and their keys and aggregate
//!   arguments are scattered to the radix partitions directly, in batches of
//!   many chunks, so that every row is inserted into a single hash table, the
//!   one of the final aggregation of its partition. If the groups of the
//!   thread come back after all, it aggregates its rows again (see
//!   `RadixLocalState::aggregate`).
//! * `combine`: every thread flushes the groups that remain in its table,
//!   and scatters the rows that bypassed it.
//!   A flush emits the keys and partial states of the groups of the table,
//!   hashes the keys of every group once and scatters the groups to
//!   [`NUM_PARTITIONS`] radix partitions by the upper [`RADIX_BITS`] bits of
//!   their hash, as the rows that bypass the tables are scattered. The
//!   partial states (and the rows) of a group, from all threads and flushes,
//!   all end up in the same partition. This replaces the
//!   `RepartitionExec(Hash)` between DataFusion's partial and final
//!   aggregations: the states are moved by reference, not sent through
//!   channels.
//! * `finalize`: records the partitions that have groups. If there is none,
//!   the aggregation has no output.
//! * `get_data`: the threads of the pipeline that reads the aggregation claim
//!   the partitions one by one. A thread merges the states of the groups of
//!   its partition, and aggregates the rows of the partition that bypassed
//!   the thread-local tables, into a new [`GroupedHashTable`] (DataFusion's
//!   final aggregation) and evaluates it, and emits the result in chunks of
//!   at most [`DEFAULT_VECTOR_SIZE`] rows. Partitions have disjoint groups, so
//!   the threads finalize them in parallel, without sharing anything.
//!
//! A top-k aggregation (see [`TopK`]) only needs the groups that the sort
//! above it may keep: every flush of a thread-local table only scatters
//! these groups of the table, and the final top groups of every partition
//! are merged into the top groups of the aggregation, which the thread that
//! finalizes the last partition emits.
//!
//! The partition hash uses another seed than the hash tables
//! ([`GroupValues`](datafusion_physical_plan::aggregates::group_values::GroupValues)),
//! so that the groups of a partition, which share the upper bits of their
//! partition hash, still spread over the hash table of the partition.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use std::collections::HashMap;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, BooleanBufferBuilder,
    ByteView, GenericByteViewArray, PrimitiveArray, RecordBatch, UInt32Array,
    downcast_primitive,
};
use arrow::buffer::NullBuffer;
use arrow::compute::{concat_batches, interleave, take};
use arrow::datatypes::{BinaryViewType, ByteViewType, DataType, StringViewType};
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::{Result, internal_err};
use parking_lot::Mutex;

use super::grouped_ht::{GroupedHashTable, GroupedStates, evaluate_rows};
use super::hash_aggregate::{AggregateSpec, InputBuffer, TopK, TopKOrder};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    GlobalSinkState, GlobalSourceState, LocalSinkState, LocalSourceState, SinkFinalize,
    SourceResult, downcast_state, downcast_state_mut,
};

/// The name of the operator, for error messages.
const OPERATOR: &str = "PhysicalHashAggregate";

/// The number of bits of the hash of a group that select its partition.
pub(crate) const RADIX_BITS: u32 = 6;

/// The number of radix partitions.
pub(crate) const NUM_PARTITIONS: usize = 1 << RADIX_BITS;

/// The number of bits of the hash of a group, after its [`RADIX_BITS`],
/// that select its sub-partition in its partition: a partition with many
/// groups is finalized in sub-partitions (see [`finalize_partition`]), so
/// that their hash tables stay cache resident.
const SPLIT_BITS: u32 = 2;

/// The number of sub-partitions of a partition.
const NUM_SPLITS: usize = 1 << SPLIT_BITS;

/// The number of rows of a partition (partial states and input rows) above
/// which it is finalized in sub-partitions.
const SPLIT_ROWS: usize = 1 << 15;

// The sub-partitions of all partitions are numbered by a `u8`.
const _: () = assert!(RADIX_BITS + SPLIT_BITS <= u8::BITS);

/// The hash state of the partition hash: another seed than the one of the
/// hash tables of DataFusion's aggregation and repartitioning.
const PARTITION_HASH: RandomState = RandomState::with_seed(0x8f1b_c3a5_d2e4_7a61);

/// The number of groups per aggregated row of a full thread-local table
/// above which the rows of the thread bypass its table (see
/// [`RadixLocalState::aggregate`]).
const BYPASS_RATIO: f64 = 0.75;

/// The bytes of the strings and binaries with 32-bit offsets of the buffered
/// rows that bypassed a thread-local table at which they are scattered: the
/// scatter gathers the rows of all buffered chunks into a single array first.
const BYPASS_BYTES: usize = 64 << 20;

/// The sink and source roles of a grouped aggregation, see the
/// [module documentation](self).
pub(crate) struct RadixHashAggregate;

/// The data of a radix partition, from all threads.
#[derive(Debug, Default)]
struct Partition {
    /// The partial states of groups of the partition: one entry per flush of
    /// a thread-local table that had groups in the partition.
    states: Vec<GroupedStates>,
    /// Input rows of the partition that bypassed the thread-local tables
    /// (their columns, see [`evaluate_rows`]), by sub-partition: one entry
    /// per scatter of rows of a thread that had rows in the sub-partition.
    rows: [Vec<Vec<ArrayRef>>; NUM_SPLITS],
}

impl Partition {
    fn is_empty(&self) -> bool {
        self.states.is_empty() && self.rows.iter().all(Vec::is_empty)
    }

    /// The number of partial states and input rows of the partition.
    fn num_rows(&self) -> usize {
        let rows = self
            .rows
            .iter()
            .flatten()
            .map(|columns| columns.first().map_or(0, |column| column.len()));
        self.states
            .iter()
            .map(GroupedStates::num_groups)
            .chain(rows)
            .sum()
    }
}

/// The global sink state: the radix partitions.
#[derive(Debug)]
struct RadixSinkState {
    partitions: Vec<Mutex<Partition>>,
    /// The partitions that have groups, set by `finalize`.
    non_empty: Mutex<Option<Vec<usize>>>,
}

impl GlobalSinkState for RadixSinkState {}

impl RadixSinkState {
    fn new() -> Self {
        Self {
            partitions: (0..NUM_PARTITIONS)
                .map(|_| Mutex::new(Partition::default()))
                .collect(),
            non_empty: Mutex::new(None),
        }
    }

    /// Scatter `states`, the groups of a thread-local table, to their
    /// partitions `partitions` (see [`hash_keys`]).
    fn add_states(&self, states: GroupedStates, partitions: &[u8]) -> Result<()> {
        for (partition, states) in scatter_states(states, partitions, NUM_PARTITIONS)? {
            self.partitions[partition].lock().states.push(states);
        }
        Ok(())
    }

    /// Scatter input rows, the columns of the rows of several chunks (see
    /// [`evaluate_rows`]), to their sub-partitions `sub_partitions` (see
    /// [`sub_partition_of`]), in the order of the chunks.
    fn add_rows(&self, chunks: Vec<Vec<ArrayRef>>, sub_partitions: &[u8]) -> Result<()> {
        for (sub_partition, columns) in
            scatter(chunks, sub_partitions, NUM_PARTITIONS * NUM_SPLITS)?
        {
            let (partition, split) =
                (sub_partition >> SPLIT_BITS, sub_partition % NUM_SPLITS);
            self.partitions[partition].lock().rows[split].push(columns);
        }
        Ok(())
    }
}

/// The thread-local sink state.
#[derive(Debug)]
struct RadixLocalState {
    table: GroupedHashTable,
    /// The number of groups at which the table is flushed or grown.
    capacity: usize,
    /// The capacity the table can grow to.
    maximum_capacity: usize,
    /// The number of input rows aggregated into the table since it was last
    /// flushed.
    table_rows: usize,
    /// The number of groups flushed so far, and of rows that bypassed the
    /// table.
    flushed: usize,
    /// The number of distinct groups among the flushed groups and the rows
    /// that bypassed the table.
    distinct_flushed: DistinctEstimate,
    /// The input rows, if they must be sorted before they are aggregated.
    buffer: InputBuffer,
    /// The rows that bypass the table, while the thread does not aggregate
    /// its rows (see [`Self::aggregate`]).
    bypass: Option<BypassBuffer>,
    /// The number of buffered rows that bypassed the table at which they are
    /// scattered.
    bypass_rows: usize,
    /// The partition hashes of the groups being flushed, and their
    /// partitions.
    hashes: Vec<u64>,
    partitions: Vec<u8>,
}

impl LocalSinkState for RadixLocalState {}

impl RadixLocalState {
    /// Aggregate the input rows of `batch` into the thread-local table, and
    /// flush the table into the partitions of `global` if it reached its
    /// capacity.
    ///
    /// Like DuckDB's thread-local hash tables, the table has a bounded
    /// capacity, so that it stays cache resident: with many groups, a thread
    /// would otherwise aggregate into a huge table (with a cache miss for
    /// most rows) that hardly reduces its rows. Once it reaches its capacity,
    /// it is flushed (DuckDB's `Abandon`): its groups are scattered to the
    /// radix partitions, which merge them with the other groups of the same
    /// keys anyway, and it starts over. If the flushes emitted the same
    /// groups over and over instead (at least half of the flushed groups,
    /// and more than the table holds, were flushed before), because the
    /// groups of the thread just do not fit, the table grows to hold all of
    /// them, if they fit into its maximum capacity, like DuckDB's
    /// `TryGrow`: the number of distinct flushed groups is estimated with a
    /// HyperLogLog sketch of their partition hashes.
    ///
    /// If the full table has more than [`BYPASS_RATIO`] groups per row it
    /// aggregated (and its groups were not flushed over and over), the rows
    /// of the thread hardly repeat their groups: aggregating them builds a
    /// hash table for nothing, and the final aggregation of the partitions
    /// interns their groups once more (DataFusion skips its partial
    /// aggregation then). So the thread flushes its table and stops
    /// aggregating its rows (for the aggregations of
    /// [`AggregateSpec::bypass`]): the rows *bypass* the table, they are
    /// scattered to the radix partitions directly (see [`BypassBuffer`]),
    /// whose final aggregation aggregates them. Their hashes are added to the
    /// sketch of the flushed groups, and once it shows that the groups of
    /// the thread came back after all (by the same rule), the thread
    /// aggregates its rows again, in a table that holds its groups.
    fn aggregate(
        &mut self,
        spec: &AggregateSpec,
        global: &RadixSinkState,
        batch: &RecordBatch,
    ) -> Result<()> {
        if let Some(bypass) = &mut self.bypass {
            bypass.push(spec, batch, &mut self.distinct_flushed)?;
            self.flushed += batch.num_rows();
            if bypass.is_full(self.bypass_rows) {
                bypass.scatter(global)?;
                if let Some(capacity) = self.grown_capacity() {
                    self.capacity = capacity;
                    self.bypass = None;
                }
            }
            return Ok(());
        }
        self.table.update(spec, batch)?;
        self.table_rows += batch.num_rows();
        let num_groups = self.table.num_groups();
        if num_groups < self.capacity {
            return Ok(());
        }
        if let Some(capacity) = self.grown_capacity() {
            self.capacity = capacity;
            return Ok(());
        }
        let bypass =
            spec.bypass && num_groups as f64 > BYPASS_RATIO * self.table_rows as f64;
        self.flush(spec, global)?;
        if bypass {
            self.bypass = Some(BypassBuffer::default());
        }
        Ok(())
    }

    /// The capacity that the table grows to if the flushed groups came back
    /// over and over, and all groups fit into its maximum capacity (with some
    /// slack for the estimate), see [`Self::aggregate`].
    fn grown_capacity(&self) -> Option<usize> {
        let distinct = self.distinct_flushed.estimate();
        let repeated = self.flushed.saturating_sub(distinct);
        // Room for all groups, with some slack for the estimate.
        let needed = (distinct + distinct / 4).max(self.capacity * 2);
        (repeated >= self.capacity
            && repeated * 2 >= self.flushed
            && needed <= self.maximum_capacity)
            .then(|| needed.next_power_of_two().min(self.maximum_capacity))
    }

    /// Flush the groups of the table into the partitions of `global`: for a
    /// top-k aggregation, only the groups that may be among its top groups.
    fn flush(&mut self, spec: &AggregateSpec, global: &RadixSinkState) -> Result<()> {
        self.table_rows = 0;
        let Some(mut states) = self.table.emit_states()? else {
            return Ok(());
        };
        if let Some(top_k) = &spec.top_k
            && let Some(indices) = top_k.select(top_k_values(top_k, &states)?)?
        {
            states = states.take(&indices)?;
        }
        hash_keys(&states.keys, &mut self.hashes, &mut self.partitions)?;
        self.flushed += states.num_groups();
        self.distinct_flushed.add(&self.hashes);
        global.add_states(states, &self.partitions)
    }
}

/// The values that order the groups of `states`, partial states of a top-k
/// aggregation: their keys, or the partial states of their aggregate (its
/// value).
fn top_k_values<'a>(top_k: &TopK, states: &'a GroupedStates) -> Result<&'a dyn Array> {
    let values = match top_k.order {
        TopKOrder::GroupKey => states.keys.first(),
        TopKOrder::Aggregate => states.states.first().and_then(|state| state.first()),
    };
    match values {
        Some(values) => Ok(values.as_ref()),
        None => internal_err!("{OPERATOR} found no value to order the groups by"),
    }
}

/// The rows of a thread that bypass its table (see
/// [`RadixLocalState::aggregate`]), until they are scattered to the radix
/// partitions: the columns of the rows of their chunks (see
/// [`evaluate_rows`]), and the sub-partition of every row (see
/// [`sub_partition_of`]). The rows of many chunks are scattered at once, so
/// that the final aggregation of a partition aggregates large slices of its
/// rows.
#[derive(Debug, Default)]
struct BypassBuffer {
    chunks: Vec<Vec<ArrayRef>>,
    /// The sub-partition of every row, in the order of the chunks.
    partitions: Vec<u8>,
    /// The bytes of the values of the buffered columns with 32-bit offsets.
    offset_bytes: usize,
    hashes: Vec<u64>,
}

impl BypassBuffer {
    /// Buffer the input rows of `batch`: evaluate their group keys and the
    /// arguments of the aggregates, hash their keys, and add their hashes to
    /// `distinct`.
    fn push(
        &mut self,
        spec: &AggregateSpec,
        batch: &RecordBatch,
        distinct: &mut DistinctEstimate,
    ) -> Result<()> {
        let columns = evaluate_rows(spec, batch)?;
        let keys = &columns[..spec.group_by.expr().len()];
        self.hashes.clear();
        self.hashes.resize(batch.num_rows(), 0);
        create_hashes(keys, &PARTITION_HASH, &mut self.hashes)?;
        self.partitions
            .extend(self.hashes.iter().map(|&hash| sub_partition_of(hash)));
        distinct.add(&self.hashes);
        self.offset_bytes += columns
            .iter()
            .map(|column| offset_bytes(column.as_ref()))
            .sum::<usize>();
        self.chunks.push(columns);
        Ok(())
    }

    /// Whether the buffered rows are to be scattered: once there are `rows`
    /// of them, or they have [`BYPASS_BYTES`] bytes of values with 32-bit
    /// offsets.
    fn is_full(&self, rows: usize) -> bool {
        self.partitions.len() >= rows || self.offset_bytes >= BYPASS_BYTES
    }

    /// Scatter the buffered rows to the partitions of `global`.
    fn scatter(&mut self, global: &RadixSinkState) -> Result<()> {
        if self.chunks.is_empty() {
            return Ok(());
        }
        global.add_rows(std::mem::take(&mut self.chunks), &self.partitions)?;
        self.partitions.clear();
        self.offset_bytes = 0;
        Ok(())
    }
}

/// The bytes of the values of `array` if it has 32-bit offsets (strings and
/// binaries), which [`scatter`] adds up for the rows of several chunks.
fn offset_bytes(array: &dyn Array) -> usize {
    let offsets = match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value_offsets(),
        DataType::Binary => array.as_binary::<i32>().value_offsets(),
        _ => return 0,
    };
    match (offsets.first(), offsets.last()) {
        (Some(&first), Some(&last)) => (last - first) as usize,
        _ => 0,
    }
}

/// The number of groups a thread-local table holds before it is flushed or
/// grown, see [`RadixLocalState::aggregate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableCapacity {
    /// The initial capacity. The rows that bypass the table are scattered in
    /// batches of four times as many rows.
    pub(crate) initial: usize,
    /// The capacity a table can grow to.
    pub(crate) maximum: usize,
}

impl Default for TableCapacity {
    /// 32,768 groups (DuckDB's `SinkCapacity` for more than two threads),
    /// up to 262,144 groups.
    fn default() -> Self {
        Self {
            initial: 1 << 15,
            maximum: 1 << 18,
        }
    }
}

/// The global source state: the partitions to finalize.
#[derive(Debug)]
struct RadixSourceState {
    sink: Arc<RadixSinkState>,
    /// The partitions that have groups.
    partitions: Vec<usize>,
    /// The next entry of `partitions` to claim.
    next: AtomicUsize,
    /// The top groups of a top-k aggregation.
    top_k: Option<TopKGroups>,
}

impl GlobalSourceState for RadixSourceState {
    fn max_threads(&self) -> usize {
        self.partitions.len().max(1)
    }
}

/// The top groups (see [`TopK`]) of the partitions of a top-k aggregation
/// that were finalized so far. The thread that finalizes the last partition
/// emits them, so that the aggregation produces the top groups of all of its
/// groups, as DataFusion's top-k aggregation of a single partition does
/// (rather than the top groups of every partition).
#[derive(Debug)]
struct TopKGroups {
    top_k: TopK,
    groups: Mutex<Option<RecordBatch>>,
    /// The number of partitions that were not finalized yet.
    remaining: AtomicUsize,
}

impl TopKGroups {
    /// Add `outputs`, the groups of a finalized partition. Returns the top
    /// groups of all partitions once all of them were added.
    fn add(&self, outputs: Vec<RecordBatch>) -> Result<Option<RecordBatch>> {
        for output in outputs {
            let output = self.top_k.select_groups(output)?;
            let mut groups = self.groups.lock();
            let merged = match groups.take() {
                Some(groups) => self.top_k.select_groups(concat_batches(
                    &output.schema(),
                    [&groups, &output],
                )?)?,
                None => output,
            };
            *groups = Some(merged);
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            return Ok(self.groups.lock().take());
        }
        Ok(None)
    }
}

/// The thread-local source state: the evaluated groups of the partition the
/// thread finalized last that were not emitted yet.
#[derive(Debug, Default)]
struct RadixLocalSource {
    /// The batches of groups to emit, the next one last.
    outputs: Vec<RecordBatch>,
    /// The number of rows of the next batch that were emitted.
    offset: usize,
}

impl LocalSourceState for RadixLocalSource {}

impl RadixHashAggregate {
    pub(crate) fn global_sink_state() -> Result<Arc<dyn GlobalSinkState>> {
        Ok(Arc::new(RadixSinkState::new()))
    }

    pub(crate) fn local_sink_state(
        spec: &AggregateSpec,
        capacity: TableCapacity,
    ) -> Result<Box<dyn LocalSinkState>> {
        let initial = capacity.initial.max(1);
        Ok(Box::new(RadixLocalState {
            table: GroupedHashTable::try_new_update(spec)?,
            capacity: initial,
            maximum_capacity: capacity.maximum,
            table_rows: 0,
            flushed: 0,
            distinct_flushed: DistinctEstimate::default(),
            buffer: InputBuffer::default(),
            bypass: None,
            bypass_rows: initial.saturating_mul(4),
            hashes: vec![],
            partitions: vec![],
        }))
    }

    pub(crate) fn sink(
        spec: &AggregateSpec,
        batch: RecordBatch,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let local = downcast_state_mut::<RadixLocalState>(lstate.as_any_mut(), OPERATOR)?;
        if spec.input_order.is_some() {
            local.buffer.push(batch);
            return Ok(());
        }
        let global = downcast_state::<RadixSinkState>(gstate.as_any(), OPERATOR)?;
        local.aggregate(spec, global, &batch)
    }

    pub(crate) fn combine(
        spec: &AggregateSpec,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let global = downcast_state::<RadixSinkState>(gstate.as_any(), OPERATOR)?;
        let local = downcast_state_mut::<RadixLocalState>(lstate.as_any_mut(), OPERATOR)?;
        if let Some(order) = &spec.input_order {
            for batch in local.buffer.take_sorted(&spec.input_schema, order)? {
                local.aggregate(spec, global, &batch)?;
            }
        }
        if let Some(bypass) = &mut local.bypass {
            bypass.scatter(global)?;
        }
        // Like DataFusion's partial aggregation of a partition without rows,
        // a thread without groups contributes the grouping sets without
        // group columns.
        if local.flushed == 0 {
            local.table.add_empty_grouping_sets(spec)?;
        }
        local.flush(spec, global)
    }

    pub(crate) fn finalize(gstate: &dyn GlobalSinkState) -> Result<SinkFinalize> {
        let global = downcast_state::<RadixSinkState>(gstate.as_any(), OPERATOR)?;
        let non_empty = global
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, partition)| !partition.lock().is_empty())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let result = if non_empty.is_empty() {
            SinkFinalize::NoOutputPossible
        } else {
            SinkFinalize::Ready
        };
        *global.non_empty.lock() = Some(non_empty);
        Ok(result)
    }

    pub(crate) fn global_source_state(
        spec: &AggregateSpec,
        sink_state: Arc<dyn GlobalSinkState>,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let Some(sink) = sink_state.downcast_arc::<RadixSinkState>() else {
            return internal_err!(
                "{OPERATOR} received a sink state of an unexpected type"
            );
        };
        let Some(partitions) = sink.non_empty.lock().clone() else {
            return internal_err!("{OPERATOR} is a source before it was finalized");
        };
        let top_k = spec.top_k.map(|top_k| TopKGroups {
            top_k,
            groups: Mutex::new(None),
            remaining: AtomicUsize::new(partitions.len()),
        });
        Ok(Arc::new(RadixSourceState {
            sink,
            partitions,
            next: AtomicUsize::new(0),
            top_k,
        }))
    }

    pub(crate) fn local_source_state() -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(RadixLocalSource::default()))
    }

    pub(crate) fn get_data(
        spec: &AggregateSpec,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<RadixSourceState>(gstate.as_any(), OPERATOR)?;
        let local =
            downcast_state_mut::<RadixLocalSource>(lstate.as_any_mut(), OPERATOR)?;
        loop {
            if let Some(output) = local.outputs.last() {
                let remaining = output.num_rows() - local.offset;
                if remaining > 0 {
                    let len = remaining.min(DEFAULT_VECTOR_SIZE);
                    let chunk = output.slice(local.offset, len);
                    local.offset += len;
                    return Ok(SourceResult::Chunk(DataChunk::from_record_batch(chunk)));
                }
                local.outputs.pop();
                local.offset = 0;
                continue;
            }
            let next = state.next.fetch_add(1, Ordering::Relaxed);
            let Some(&partition) = state.partitions.get(next) else {
                return Ok(SourceResult::Finished);
            };
            let data = std::mem::take(&mut *state.sink.partitions[partition].lock());
            let outputs = finalize_partition(spec, data)?;
            local.outputs = match &state.top_k {
                Some(top_k) => top_k.add(outputs)?.into_iter().collect(),
                None => outputs.into_iter().rev().collect(),
            };
        }
    }
}

/// Hash `keys`, the group keys of a set of groups or rows, into `hashes`
/// (their partition hashes), and their partitions into `partitions`.
fn hash_keys(
    keys: &[ArrayRef],
    hashes: &mut Vec<u64>,
    partitions: &mut Vec<u8>,
) -> Result<()> {
    let num_rows = keys.first().map_or(0, |keys| keys.len());
    hashes.clear();
    hashes.resize(num_rows, 0);
    create_hashes(keys, &PARTITION_HASH, hashes)?;
    partitions.clear();
    partitions.extend(hashes.iter().map(|&hash| partition_of(hash)));
    Ok(())
}

/// The partition of a group with partition hash `hash`: its upper
/// [`RADIX_BITS`] bits.
fn partition_of(hash: u64) -> u8 {
    (hash >> (u64::BITS - RADIX_BITS)) as u8
}

/// The sub-partition of a group with partition hash `hash`, among the
/// sub-partitions of all partitions: its upper [`RADIX_BITS`] and
/// [`SPLIT_BITS`] bits, its partition followed by its sub-partition in its
/// partition.
fn sub_partition_of(hash: u64) -> u8 {
    (hash >> (u64::BITS - RADIX_BITS - SPLIT_BITS)) as u8
}

/// Scatter the groups of `states` to their partitions `partitions` (one per
/// group, among `num_partitions`), see [`scatter`].
fn scatter_states(
    states: GroupedStates,
    partitions: &[u8],
    num_partitions: usize,
) -> Result<Vec<(usize, GroupedStates)>> {
    let num_keys = states.keys.len();
    let widths = states.states.iter().map(Vec::len).collect::<Vec<_>>();
    let mut columns = states.keys;
    columns.extend(states.states.into_iter().flatten());
    Ok(scatter(vec![columns], partitions, num_partitions)?
        .into_iter()
        .map(|(partition, columns)| {
            let mut columns = columns.into_iter();
            let keys = columns.by_ref().take(num_keys).collect();
            let states = widths
                .iter()
                .map(|&width| columns.by_ref().take(width).collect())
                .collect();
            (partition, GroupedStates { keys, states })
        })
        .collect())
}

/// Scatter the rows of `chunks` (the columns of the rows of every chunk, of
/// the same types in every chunk) to their partitions `partitions` (one per
/// row, in the order of the chunks, among `num_partitions`). Returns the rows
/// of every partition that has some, in the order of their partitions (and of
/// their chunks).
///
/// A counting sort of the rows by partition gives every row its position in
/// the order of the partitions, every column is reordered once (see
/// [`scatter_column`]), and every partition gets a zero-copy slice of the
/// result. If all rows are in the same partition, the chunks are kept as
/// they are.
fn scatter(
    chunks: Vec<Vec<ArrayRef>>,
    partitions: &[u8],
    num_partitions: usize,
) -> Result<Vec<(usize, Vec<ArrayRef>)>> {
    let num_rows = partitions.len();
    let chunk_rows = chunks
        .iter()
        .map(|columns| columns.first().map_or(0, |column| column.len()))
        .sum::<usize>();
    if chunk_rows != num_rows {
        return internal_err!(
            "{OPERATOR} cannot scatter {chunk_rows} rows to the partitions of {num_rows} rows"
        );
    }
    if u32::try_from(num_rows).is_err() {
        return internal_err!("{OPERATOR} cannot partition {num_rows} rows at once");
    }

    if partitions
        .iter()
        .any(|&partition| usize::from(partition) >= num_partitions)
    {
        return internal_err!(
            "{OPERATOR} cannot scatter rows to {num_partitions} partitions"
        );
    }
    // A counting sort of the rows by partition.
    let mut offsets = vec![0usize; num_partitions + 1];
    for &partition in partitions {
        offsets[usize::from(partition) + 1] += 1;
    }
    for i in 0..num_partitions {
        offsets[i + 1] += offsets[i];
    }
    if let Some(partition) =
        (0..num_partitions).find(|&p| offsets[p + 1] - offsets[p] == num_rows)
    {
        return Ok(chunks
            .into_iter()
            .filter(|columns| columns.first().is_some_and(|column| !column.is_empty()))
            .map(|columns| (partition, columns))
            .collect());
    }
    // The position of every row in the order of the partitions.
    let mut next = offsets.clone();
    let positions = partitions
        .iter()
        .map(|&partition| {
            let position = &mut next[usize::from(partition)];
            *position += 1;
            (*position - 1) as u32
        })
        .collect::<Vec<_>>();
    let num_columns = chunks.first().map_or(0, Vec::len);
    let sorted = (0..num_columns)
        .map(|column| {
            let arrays = chunks
                .iter()
                .map(|columns| columns[column].as_ref())
                .collect::<Vec<_>>();
            scatter_column(&arrays, &positions)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(offsets
        .windows(2)
        .enumerate()
        .filter(|(_, range)| range[1] > range[0])
        .map(|(partition, range)| {
            let (offset, len) = (range[0], range[1] - range[0]);
            let columns = sorted
                .iter()
                .map(|column| column.slice(offset, len))
                .collect();
            (partition, columns)
        })
        .collect())
}

/// The rows of `arrays` (of the same type, the rows of consecutive chunks)
/// at their `positions` (a permutation of their rows, in the order of the
/// arrays).
///
/// Primitive values, booleans and views (of `Utf8View` and `BinaryView`
/// strings, whose bytes are not copied) are *scattered*: every row is read
/// once, in order, and written to its position, which advances
/// sequentially in every partition, so that the writes of every partition
/// stay in cache. Arrays of other types are gathered by Arrow's `take` (a
/// single array) or `interleave` (several arrays).
fn scatter_column(arrays: &[&dyn Array], positions: &[u32]) -> Result<ArrayRef> {
    let Some(first) = arrays.first() else {
        return internal_err!("{OPERATOR} cannot scatter the rows of no array");
    };
    let data_type = first.data_type().clone();
    if arrays.iter().any(|array| array.data_type() != &data_type) {
        return internal_err!("{OPERATOR} cannot scatter arrays of different types");
    }
    macro_rules! primitive {
        ($t:ty) => {
            scatter_primitive::<$t>(arrays, positions, &data_type)
        };
    }
    let scattered = downcast_primitive! {
        data_type => (primitive),
        DataType::Boolean => scatter_boolean(arrays, positions),
        DataType::Utf8View => scatter_views::<StringViewType>(arrays, positions),
        DataType::BinaryView => scatter_views::<BinaryViewType>(arrays, positions),
        _ => {
            // The array and the row of every position.
            let mut rows = vec![(0, 0); positions.len()];
            for (index, (_, positions)) in positions_of_arrays(arrays, positions).enumerate()
            {
                for (row, &position) in positions.iter().enumerate() {
                    rows[position as usize] = (index, row);
                }
            }
            if let [array] = arrays {
                let indices = UInt32Array::from_iter_values(
                    rows.iter().map(|&(_, row)| row as u32),
                );
                take(*array, &indices, None)?
            } else {
                interleave(arrays, &rows)?
            }
        }
    };
    Ok(scattered)
}

/// The positions of the rows of every array of `arrays` (consecutive slices
/// of `positions`).
fn positions_of_arrays<'a>(
    arrays: &'a [&'a dyn Array],
    positions: &'a [u32],
) -> impl Iterator<Item = (&'a dyn Array, &'a [u32])> {
    arrays.iter().scan(positions, |positions, &array| {
        let (positions_of_array, rest) = positions.split_at(array.len());
        *positions = rest;
        Some((array, positions_of_array))
    })
}

/// The validity of the rows of `arrays` at their `positions`, see
/// [`scatter_column`].
fn scatter_nulls(arrays: &[&dyn Array], positions: &[u32]) -> Option<NullBuffer> {
    if arrays.iter().all(|array| array.null_count() == 0) {
        return None;
    }
    let mut valid = BooleanBufferBuilder::new(positions.len());
    valid.append_n(positions.len(), false);
    for (array, positions) in positions_of_arrays(arrays, positions) {
        match array.nulls() {
            Some(nulls) => {
                for (row, &position) in positions.iter().enumerate() {
                    if nulls.is_valid(row) {
                        valid.set_bit(position as usize, true);
                    }
                }
            }
            None => {
                for &position in positions {
                    valid.set_bit(position as usize, true);
                }
            }
        }
    }
    Some(NullBuffer::new(valid.finish()))
}

/// [`scatter_column`] of primitive arrays of type `data_type`.
fn scatter_primitive<T: ArrowPrimitiveType>(
    arrays: &[&dyn Array],
    positions: &[u32],
    data_type: &DataType,
) -> ArrayRef {
    let mut values = vec![T::Native::default(); positions.len()];
    for (array, positions) in positions_of_arrays(arrays, positions) {
        for (&value, &position) in
            array.as_primitive::<T>().values().iter().zip(positions)
        {
            values[position as usize] = value;
        }
    }
    let nulls = scatter_nulls(arrays, positions);
    Arc::new(
        PrimitiveArray::<T>::new(values.into(), nulls).with_data_type(data_type.clone()),
    )
}

/// [`scatter_column`] of boolean arrays.
fn scatter_boolean(arrays: &[&dyn Array], positions: &[u32]) -> ArrayRef {
    let mut values = BooleanBufferBuilder::new(positions.len());
    values.append_n(positions.len(), false);
    for (array, positions) in positions_of_arrays(arrays, positions) {
        for (value, &position) in array.as_boolean().values().iter().zip(positions) {
            if value {
                values.set_bit(position as usize, true);
            }
        }
    }
    let nulls = scatter_nulls(arrays, positions);
    Arc::new(BooleanArray::new(values.finish(), nulls))
}

/// [`scatter_column`] of view arrays: the views are scattered, and refer to
/// the data buffers of all arrays (each once: the arrays are often slices of
/// the same arrays, which share their buffers).
fn scatter_views<T: ByteViewType>(arrays: &[&dyn Array], positions: &[u32]) -> ArrayRef {
    let arrays = arrays
        .iter()
        .map(|array| array.as_byte_view::<T>())
        .collect::<Vec<_>>();
    let shared = arrays.windows(2).all(|pair| {
        let (first, second) = (pair[0].data_buffers(), pair[1].data_buffers());
        Arc::ptr_eq(first, second)
            || (first.len() == second.len()
                && first.iter().zip(second.iter()).all(|(first, second)| {
                    first.as_ptr() == second.as_ptr() && first.len() == second.len()
                }))
    });
    // The buffers of the result, and the indices of the buffers of every
    // array among them (if the arrays do not share their buffers).
    let mut buffers = vec![];
    let mut indices = vec![];
    if !shared {
        let mut known = HashMap::new();
        for array in &arrays {
            indices.push(
                array
                    .data_buffers()
                    .iter()
                    .map(|buffer| {
                        *known.entry((buffer.as_ptr(), buffer.len())).or_insert_with(
                            || {
                                buffers.push(buffer.clone());
                                (buffers.len() - 1) as u32
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }
    }
    let mut views = vec![0u128; positions.len()];
    let mut offset = 0;
    for (index, array) in arrays.iter().enumerate() {
        let positions = &positions[offset..offset + array.len()];
        offset += array.len();
        if shared {
            for (&view, &position) in array.views().iter().zip(positions) {
                views[position as usize] = view;
            }
            continue;
        }
        let indices = &indices[index];
        for (&view, &position) in array.views().iter().zip(positions) {
            // Views of at most 12 bytes hold their bytes inline.
            views[position as usize] = if (view as u32) <= 12 {
                view
            } else {
                let mut view = ByteView::from(view);
                view.buffer_index = indices[view.buffer_index as usize];
                view.as_u128()
            };
        }
    }
    let nulls = scatter_nulls(
        &arrays
            .iter()
            .map(|array| *array as &dyn Array)
            .collect::<Vec<_>>(),
        positions,
    );
    let buffers = match arrays.first() {
        Some(first) if shared => Arc::clone(first.data_buffers()),
        _ => Arc::from(buffers),
    };
    // SAFETY: every view is a view of one of the arrays, and refers to the
    // same bytes, of the same buffer, as in that (valid) array.
    Arc::new(unsafe {
        GenericByteViewArray::<T>::new_unchecked(views.into(), buffers, nulls)
    })
}

/// An estimate of the number of distinct values of a set of hashes: a
/// HyperLogLog sketch with [`DistinctEstimate::REGISTERS`] registers (a
/// standard error of about 3%).
#[derive(Debug, Clone)]
struct DistinctEstimate {
    registers: Box<[u8; Self::REGISTERS]>,
}

impl Default for DistinctEstimate {
    fn default() -> Self {
        Self {
            registers: Box::new([0; Self::REGISTERS]),
        }
    }
}

impl DistinctEstimate {
    /// The number of bits of a hash that select its register.
    const INDEX_BITS: u32 = 10;
    const REGISTERS: usize = 1 << Self::INDEX_BITS;

    fn add(&mut self, hashes: &[u64]) {
        for &hash in hashes {
            // The fast hash of DataFusion spreads e.g. consecutive integers
            // more evenly than random values over its upper bits, which the
            // estimate assumes: mix it first (MurmurHash3's finalizer).
            let mut hash = hash;
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            hash ^= hash >> 33;
            let index = (hash >> (u64::BITS - Self::INDEX_BITS)) as usize;
            // The position of the first set bit of the other bits (with a
            // sentinel bit, so that it is at most `65 - INDEX_BITS`).
            let rest = (hash << Self::INDEX_BITS) | (1 << (Self::INDEX_BITS - 1));
            let rank = rest.leading_zeros() as u8 + 1;
            let register = &mut self.registers[index];
            *register = (*register).max(rank);
        }
    }

    fn estimate(&self) -> usize {
        let m = Self::REGISTERS as f64;
        let (sum, zeros) =
            self.registers
                .iter()
                .fold((0f64, 0usize), |(sum, zeros), &rank| {
                    (
                        sum + 2f64.powi(-i32::from(rank)),
                        zeros + usize::from(rank == 0),
                    )
                });
        let estimate = 0.7213 / (1.0 + 1.079 / m) * m * m / sum;
        if estimate <= 2.5 * m && zeros > 0 {
            // Linear counting for small cardinalities.
            (m * (m / zeros as f64).ln()) as usize
        } else {
            estimate as usize
        }
    }
}

/// Finalize a partition: merge the partial states of its groups and
/// aggregate its rows that bypassed the thread-local tables, from all
/// threads, and evaluate the groups. Returns the output rows of the
/// partition.
///
/// A partition of more than [`SPLIT_ROWS`] rows is finalized in its
/// [`NUM_SPLITS`] sub-partitions, one after the other, so that their hash
/// tables are smaller and stay cache resident: its input rows are scattered
/// to them already, its partial states are scattered (by the next
/// [`SPLIT_BITS`] bits of the partition hashes of their keys) first.
fn finalize_partition(
    spec: &AggregateSpec,
    partition: Partition,
) -> Result<Vec<RecordBatch>> {
    if partition.num_rows() <= SPLIT_ROWS {
        let output =
            finalize_groups(spec, &partition.states, partition.rows.iter().flatten())?;
        return Ok(output.into_iter().collect());
    }
    let mut states: [Vec<GroupedStates>; NUM_SPLITS] = Default::default();
    let (mut hashes, mut splits) = (vec![], vec![]);
    for groups in partition.states {
        hash_keys(&groups.keys, &mut hashes, &mut splits)?;
        splits.clear();
        splits.extend(
            hashes
                .iter()
                .map(|&hash| sub_partition_of(hash) % NUM_SPLITS as u8),
        );
        for (split, groups) in scatter_states(groups, &splits, NUM_SPLITS)? {
            states[split].push(groups);
        }
    }
    let mut outputs = vec![];
    for (states, rows) in states.iter().zip(&partition.rows) {
        outputs.extend(finalize_groups(spec, states, rows)?);
    }
    Ok(outputs)
}

/// Merge the partial states `states` and aggregate the input rows `rows` of
/// a set of groups into a new table, and evaluate the groups. Returns `None`
/// if there are no groups.
fn finalize_groups<'a>(
    spec: &AggregateSpec,
    states: &[GroupedStates],
    rows: impl IntoIterator<Item = &'a Vec<ArrayRef>>,
) -> Result<Option<RecordBatch>> {
    let mut table = GroupedHashTable::try_new_merge(spec)?;
    for states in states {
        table.merge(states)?;
    }
    for rows in rows {
        table.update_rows(spec, rows)?;
    }
    let Some(columns) = table.evaluate()? else {
        return Ok(None);
    };
    Ok(Some(RecordBatch::try_new(
        Arc::clone(&spec.schema),
        columns,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use arrow::array::{
        BooleanArray, Decimal128Array, Int64Array, StringArray, StringViewArray,
    };
    use arrow::compute::SortOptions;
    use arrow::datatypes::{Field, Int64Type, Schema, SchemaRef};
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{col, lit};
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::aggregates::{
        AggregateExec, AggregateMode, PhysicalGroupBy,
    };

    use super::super::PhysicalHashAggregate;
    use super::super::test_support::{assert_aggregate_parity_with_setup, lower};
    use crate::executor::Executor;
    use crate::test_utils::{canonical_batch, datafusion_context, test_session_config};

    /// The partition of the group of the keys of `row`.
    fn partition_of_row(keys: &[ArrayRef], row: usize) -> usize {
        let keys = keys.iter().map(|key| key.slice(row, 1)).collect::<Vec<_>>();
        let mut hash = [0];
        create_hashes(&keys, &PARTITION_HASH, &mut hash).unwrap();
        usize::from(partition_of(hash[0]))
    }

    /// The partitions of the groups of `keys`.
    fn partitions_of(keys: &[ArrayRef]) -> Result<Vec<u8>> {
        let (mut hashes, mut partitions) = (vec![], vec![]);
        hash_keys(keys, &mut hashes, &mut partitions)?;
        Ok(partitions)
    }

    /// Groups `(i, "s{i % 100}")` with nulls, whose states are `i * 10`.
    fn groups(n: i64) -> GroupedStates {
        let keys: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter(
                (0..n).map(|i| (i % 9 != 4).then_some(i)),
            )),
            Arc::new(StringArray::from_iter(
                (0..n).map(|i| (i % 7 != 1).then(|| format!("s{}", i % 100))),
            )),
        ];
        let states: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(
            (0..n).map(|i| i * 10),
        ))];
        GroupedStates {
            keys,
            states: vec![states],
        }
    }

    #[test]
    fn groups_are_scattered_to_the_partitions_of_their_hashes() -> Result<()> {
        for n in [1, 2, 63, 64, 1000, 5000] {
            let states = groups(n);
            let global = RadixSinkState::new();
            global.add_states(states.clone(), &partitions_of(&states.keys)?)?;
            let mut seen = vec![false; n as usize];
            for (partition, data) in global.partitions.iter().enumerate() {
                let data = data.lock();
                assert!(data.rows.iter().all(Vec::is_empty));
                // A single flush puts at most one set of groups into every
                // partition.
                assert!(data.states.len() <= 1, "{n}");
                for groups in &data.states {
                    assert!(groups.num_groups() > 0);
                    for row in 0..groups.num_groups() {
                        assert_eq!(partition_of_row(&groups.keys, row), partition);
                        // The keys and the states of a group stay together.
                        let state =
                            groups.states[0][0].as_primitive::<Int64Type>().value(row);
                        let group = (state / 10) as usize;
                        assert!(!std::mem::replace(&mut seen[group], true));
                        let strings = groups.keys[1].as_string::<i32>();
                        assert_eq!(
                            strings.is_valid(row),
                            states.keys[1].is_valid(group),
                            "{n}"
                        );
                        if strings.is_valid(row) {
                            assert_eq!(strings.value(row), format!("s{}", group % 100));
                        }
                    }
                }
            }
            assert!(seen.iter().all(|&seen| seen));
        }
        // 5000 groups are spread over all partitions.
        let states = groups(5000);
        let global = RadixSinkState::new();
        global.add_states(states.clone(), &partitions_of(&states.keys)?)?;
        assert!(global.partitions.iter().all(|data| !data.lock().is_empty()));
        Ok(())
    }

    /// The columns of a chunk of rows `rows` (with nulls): the row numbers,
    /// an integer and a string view key, and string, decimal and boolean
    /// arguments.
    fn chunk(rows: std::ops::Range<i64>) -> Result<Vec<ArrayRef>> {
        Ok(vec![
            Arc::new(Int64Array::from_iter_values(rows.clone())),
            Arc::new(Int64Array::from_iter(
                rows.clone().map(|i| (i % 11 != 3).then_some(i * 7)),
            )),
            Arc::new(StringViewArray::from_iter(rows.clone().map(|i| {
                (i % 5 != 2).then(|| format!("a key that is not inlined {}", i % 300))
            }))),
            Arc::new(StringArray::from_iter(
                rows.clone().map(|i| (i % 3 != 1).then(|| format!("s{i}"))),
            )),
            Arc::new(
                Decimal128Array::from_iter(
                    rows.clone().map(|i| (i % 4 != 0).then_some(i128::from(i))),
                )
                .with_precision_and_scale(10, 2)?,
            ),
            Arc::new(BooleanArray::from_iter(
                rows.map(|i| (i % 6 != 5).then_some(i % 2 == 0)),
            )),
        ])
    }

    #[test]
    fn rows_of_several_chunks_are_scattered_to_their_partitions() -> Result<()> {
        // A single chunk, a single row, and several chunks of other sizes.
        for sizes in [&[1000][..], &[1], &[1, 700, 1, 1299, 63]] {
            let mut start = 0;
            let mut chunks = vec![];
            let mut partitions = vec![];
            for &size in sizes {
                let columns = chunk(start..start + size)?;
                partitions.extend(partitions_of(&columns[1..3])?);
                chunks.push(columns);
                start += size;
            }
            let expected = chunk(0..start)?;
            let mut seen = vec![false; start as usize];
            let scattered = scatter(chunks, &partitions, NUM_PARTITIONS)?;
            assert!(scattered.windows(2).all(|w| w[0].0 < w[1].0), "{sizes:?}");
            for (partition, columns) in &scattered {
                let rows = columns[0].as_primitive::<Int64Type>();
                assert!(!rows.is_empty());
                // In the order of their chunks and rows.
                assert!(rows.values().windows(2).all(|w| w[0] < w[1]));
                for (position, &row) in rows.values().iter().enumerate() {
                    assert!(!std::mem::replace(&mut seen[row as usize], true));
                    assert_eq!(usize::from(partitions[row as usize]), *partition);
                    for (column, expected) in columns.iter().zip(&expected) {
                        assert_eq!(column.data_type(), expected.data_type());
                        assert_eq!(
                            column.slice(position, 1).to_data(),
                            expected.slice(row as usize, 1).to_data(),
                            "{sizes:?}"
                        );
                    }
                }
            }
            assert!(seen.iter().all(|&seen| seen), "{sizes:?}");
        }
        // The rows of a single partition keep their chunks.
        let chunks = vec![chunk(0..10)?, chunk(10..15)?];
        let scattered = scatter(chunks.clone(), &[7; 15], NUM_PARTITIONS)?;
        assert_eq!(scattered.len(), 2);
        for ((partition, columns), chunk) in scattered.iter().zip(&chunks) {
            assert_eq!(*partition, 7);
            assert!(
                columns
                    .iter()
                    .zip(chunk)
                    .all(|(column, chunk)| Arc::ptr_eq(column, chunk))
            );
        }
        // The partitions are those of all rows, among the partitions.
        assert!(scatter(chunks.clone(), &[7; 14], NUM_PARTITIONS).is_err());
        assert!(scatter(chunks, &[7; 15], 7).is_err());
        Ok(())
    }

    #[test]
    fn columns_of_every_type_are_scattered() -> Result<()> {
        use arrow::array::{
            BinaryViewArray, DictionaryArray, FixedSizeBinaryArray, Int32Array,
            NullArray, TimestampMicrosecondArray,
        };
        use arrow::datatypes::Int8Type;
        use datafusion_common::ScalarValue;

        let rows = 100;
        // Arrays that `scatter_column` scatters itself (primitives, booleans
        // and views) or gathers with Arrow's kernels, with nulls.
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter(
                (0..rows).map(|i| (i % 5 != 1).then_some(i)),
            )),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values((0..rows).map(i64::from))
                    .with_timezone("+01:00"),
            ),
            Arc::new(
                Decimal128Array::from_iter(
                    (0..rows).map(|i| (i % 9 != 2).then_some(i.into())),
                )
                .with_precision_and_scale(9, 3)?,
            ),
            Arc::new(BooleanArray::from_iter(
                (0..rows).map(|i| (i % 3 != 0).then_some(i % 2 == 0)),
            )),
            Arc::new(StringViewArray::from_iter((0..rows).map(|i| {
                (i % 7 != 3).then(|| format!("a view of more than 12 bytes {i}"))
            }))),
            Arc::new(BinaryViewArray::from_iter((0..rows).map(|i| {
                (i % 4 != 0).then(|| format!("binary {}", "x".repeat(i as usize % 20)))
            }))),
            Arc::new(StringArray::from_iter(
                (0..rows).map(|i| (i % 6 != 5).then(|| format!("s{i}"))),
            )),
            Arc::new(FixedSizeBinaryArray::try_from_iter(
                (0..rows).map(|i| [i as u8; 3]),
            )?),
            Arc::new(
                (0..rows)
                    .map(|i| ["x", "y", "z"][i as usize % 3])
                    .collect::<DictionaryArray<Int8Type>>(),
            ),
            Arc::new(NullArray::new(rows as usize)),
        ];
        // Row `r` goes to position `positions[r]`.
        let positions = (0..rows)
            .map(|i| (i * 37 % rows) as u32)
            .collect::<Vec<_>>();
        let check = |column: &ArrayRef, scattered: &ArrayRef| -> Result<()> {
            assert_eq!(scattered.data_type(), column.data_type());
            assert_eq!(scattered.len(), column.len());
            assert_eq!(scattered.null_count(), column.null_count());
            for (row, &position) in positions.iter().enumerate() {
                assert_eq!(
                    ScalarValue::try_from_array(scattered, position as usize)?,
                    ScalarValue::try_from_array(column, row)?,
                    "{}",
                    column.data_type()
                );
            }
            Ok(())
        };
        // Slices of the columns (which share their buffers) as chunks of
        // various sizes.
        for sizes in [&[100][..], &[1, 49, 50], &[30, 30, 39, 1]] {
            for column in &columns {
                let mut offset = 0;
                let chunks = sizes
                    .iter()
                    .map(|&size| {
                        offset += size;
                        column.slice(offset - size, size)
                    })
                    .collect::<Vec<_>>();
                let arrays = chunks
                    .iter()
                    .map(|chunk| chunk.as_ref())
                    .collect::<Vec<_>>();
                check(column, &scatter_column(&arrays, &positions)?)?;
            }
        }
        // Views of arrays with buffers of their own.
        let chunks = (0..4)
            .map(|chunk| {
                Arc::new(StringViewArray::from_iter((0..25).map(|i| {
                    (i % 6 != 1).then(|| format!("a view of chunk {chunk}, row {i:>10}"))
                }))) as ArrayRef
            })
            .collect::<Vec<_>>();
        let column = arrow::compute::concat(
            &chunks
                .iter()
                .map(|chunk| chunk.as_ref())
                .collect::<Vec<_>>(),
        )?;
        let arrays = chunks
            .iter()
            .map(|chunk| chunk.as_ref())
            .collect::<Vec<_>>();
        let scattered = scatter_column(&arrays, &positions)?;
        check(&column, &scattered)?;
        // Every buffer of the chunks once.
        assert_eq!(
            scattered.as_string_view().data_buffers().len(),
            chunks
                .iter()
                .map(|chunk| chunk.as_string_view().data_buffers().len())
                .sum::<usize>()
        );
        // Arrays of different types are not scattered together.
        let arrays = [columns[0].as_ref(), columns[1].as_ref()];
        assert!(scatter_column(&arrays, &positions).is_err());
        Ok(())
    }

    /// The spec, a local and the global sink state of a thread of
    /// `COUNT(*) GROUP BY k`.
    type Sink = (
        Arc<AggregateSpec>,
        Box<dyn LocalSinkState>,
        Arc<dyn GlobalSinkState>,
    );

    fn key_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]))
    }

    /// The sink of `COUNT(*) GROUP BY k`, with tables of capacity `capacity`,
    /// whose rows may bypass the tables if `bypass`.
    fn count_by_key_with(capacity: TableCapacity, bypass: bool) -> Result<Sink> {
        let schema = key_schema();
        let count = AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
            .schema(Arc::clone(&schema))
            .alias("count(*)")
            .build()?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![Arc::new(count)],
            vec![None],
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?,
            schema,
        )?;
        let input = Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(
            aggregate.input(),
        )));
        let mut spec =
            PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?.spec();
        assert!(spec.bypass);
        Arc::get_mut(&mut spec).expect("a single owner").bypass = bypass;
        let local = RadixHashAggregate::local_sink_state(&spec, capacity)?;
        Ok((spec, local, RadixHashAggregate::global_sink_state()?))
    }

    /// The sink of `COUNT(*) GROUP BY k`, with tables of capacity `capacity`.
    fn count_by_key(capacity: TableCapacity) -> Result<Sink> {
        count_by_key_with(capacity, true)
    }

    fn key_batch(keys: impl IntoIterator<Item = i64>) -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            key_schema(),
            vec![Arc::new(Int64Array::from_iter_values(keys))],
        )?)
    }

    /// Combine the local sink state of `sink`, finalize the aggregation and
    /// return the counts of its keys.
    fn finish(sink: Sink) -> Result<BTreeMap<i64, i64>> {
        let (spec, mut local, global) = sink;
        RadixHashAggregate::combine(&spec, global.as_ref(), local.as_mut())?;
        let mut counts = BTreeMap::new();
        if RadixHashAggregate::finalize(global.as_ref())?
            == SinkFinalize::NoOutputPossible
        {
            return Ok(counts);
        }
        let source = RadixHashAggregate::global_source_state(&spec, global)?;
        let mut local = RadixHashAggregate::local_source_state()?;
        while let SourceResult::Chunk(chunk) =
            RadixHashAggregate::get_data(&spec, source.as_ref(), local.as_mut())?
        {
            let batch = chunk.compact()?;
            let keys = batch.column(0).as_primitive::<Int64Type>();
            let values = batch.column(1).as_primitive::<Int64Type>();
            for (key, count) in keys.values().iter().zip(values.values()) {
                assert!(counts.insert(*key, *count).is_none(), "{key} twice");
            }
        }
        Ok(counts)
    }

    fn sink_batches(sink: &mut Sink, batches: impl IntoIterator<Item = Vec<i64>>) {
        let (spec, local, global) = sink;
        for keys in batches {
            let batch = key_batch(keys).unwrap();
            RadixHashAggregate::sink(spec, batch, global.as_ref(), local.as_mut())
                .unwrap();
        }
    }

    fn local_state(sink: &Sink) -> &RadixLocalState {
        sink.1.downcast_ref::<RadixLocalState>().unwrap()
    }

    /// The numbers of entries of states and of rows of the partitions of
    /// `sink`.
    fn partition_entries(sink: &Sink) -> (usize, usize) {
        let global = sink.2.downcast_ref::<RadixSinkState>().unwrap();
        global
            .partitions
            .iter()
            .fold((0, 0), |(states, rows), data| {
                let data = data.lock();
                let rows = rows + data.rows.iter().map(Vec::len).sum::<usize>();
                (states + data.states.len(), rows)
            })
    }

    const SMALL_TABLES: TableCapacity = TableCapacity {
        initial: 16,
        maximum: 64,
    };

    #[test]
    fn full_tables_are_flushed_or_grown() -> Result<()> {
        // Distinct keys: the table is flushed whenever it is full (if its rows
        // cannot bypass it).
        let mut sink = count_by_key_with(SMALL_TABLES, false)?;
        sink_batches(&mut sink, (0..20).map(|b| (b * 10..(b + 1) * 10).collect()));
        let state = local_state(&sink);
        assert_eq!(state.capacity, 16);
        assert!(state.bypass.is_none());
        assert!(state.flushed >= 180, "{}", state.flushed);
        assert_eq!(finish(sink)?, (0..200).map(|k| (k, 1)).collect());

        // The same 40 keys over and over: once the flushes emitted many of
        // them repeatedly, the table grows to hold all of them.
        let cycle = |keys: i64| {
            (0..40).map(move |b| (0..10).map(|i| (b * 10 + i) % keys).collect())
        };
        for bypass in [false, true] {
            let mut sink = count_by_key_with(SMALL_TABLES, bypass)?;
            sink_batches(&mut sink, cycle(40));
            let state = local_state(&sink);
            assert_eq!(state.capacity, 64);
            assert!(state.bypass.is_none());
            let flushed = state.flushed;
            assert!(flushed < 400, "{flushed}");
            sink_batches(&mut sink, cycle(40));
            let state = local_state(&sink);
            assert_eq!(state.flushed, flushed);
            assert_eq!(state.table.num_groups(), 40);
            assert_eq!(finish(sink)?, (0..40).map(|k| (k, 20)).collect());
        }

        // Unless they do not fit.
        for bypass in [false, true] {
            let mut sink = count_by_key_with(SMALL_TABLES, bypass)?;
            sink_batches(&mut sink, cycle(100));
            let state = local_state(&sink);
            assert_eq!(state.capacity, 16);
            assert_eq!(state.bypass.is_some(), bypass);
            assert_eq!(finish(sink)?, (0..100).map(|k| (k, 4)).collect());
        }
        Ok(())
    }

    #[test]
    fn rows_of_threads_whose_groups_hardly_repeat_bypass_their_tables() -> Result<()> {
        // Distinct keys: once the table is full, the rows bypass it, and they
        // are scattered in batches of four times the capacity of the table.
        let mut sink = count_by_key(SMALL_TABLES)?;
        sink_batches(&mut sink, [(0..10).collect(), (10..20).collect()]);
        let state = local_state(&sink);
        assert!(state.bypass.is_some());
        assert!(state.table.is_empty());
        assert_eq!(state.flushed, 20);
        let (states, rows) = partition_entries(&sink);
        assert!(states > 0 && rows == 0);
        sink_batches(&mut sink, (2..30).map(|b| (b * 10..(b + 1) * 10).collect()));
        let state = local_state(&sink);
        assert!(state.bypass.is_some());
        assert!(state.table.is_empty());
        assert_eq!(state.flushed, 300);
        assert!(state.bypass.as_ref().unwrap().partitions.len() < 64);
        let (_, rows) = partition_entries(&sink);
        assert!(rows > 0);
        assert_eq!(finish(sink)?, (0..300).map(|k| (k, 1)).collect());

        // Mostly distinct keys, with some repeated in later batches: their
        // rows end up in the same partition as their groups.
        let mut sink = count_by_key(SMALL_TABLES)?;
        let batches = (0..30).map(|b| {
            (0..10)
                .map(|i| if i == 0 { b % 7 } else { b * 10 + i })
                .collect()
        });
        sink_batches(&mut sink, batches);
        assert!(local_state(&sink).bypass.is_some());
        let mut expected = (0..30)
            .flat_map(|b| (1..10).map(move |i| (b * 10 + i, 1)))
            .collect::<BTreeMap<_, _>>();
        for b in 0..30 {
            *expected.entry(b % 7).or_default() += 1;
        }
        assert_eq!(finish(sink)?, expected);

        // A table that hardly reduces its rows does not stop aggregating
        // them if the rows of the aggregation cannot bypass it.
        let mut sink = count_by_key_with(SMALL_TABLES, false)?;
        sink_batches(&mut sink, (0..30).map(|b| (b * 10..(b + 1) * 10).collect()));
        assert!(local_state(&sink).bypass.is_none());
        assert_eq!(partition_entries(&sink).1, 0);
        Ok(())
    }

    #[test]
    fn rows_of_threads_whose_groups_repeat_do_not_bypass_their_tables() -> Result<()> {
        // Every key twice per batch: a full table has half as many groups as
        // it aggregated rows.
        let mut sink = count_by_key(SMALL_TABLES)?;
        let batches = (0..30).map(|b| (0..10).map(|i| b * 5 + i % 5).collect());
        sink_batches(&mut sink, batches);
        let state = local_state(&sink);
        assert!(state.bypass.is_none());
        assert!(state.flushed > 0);
        assert_eq!(partition_entries(&sink).1, 0);
        assert_eq!(finish(sink)?, (0..150).map(|k| (k, 2)).collect());

        // A thread whose rows bypass its table aggregates them again, in a
        // table that holds all of its groups, once their groups come back.
        let mut sink = count_by_key(SMALL_TABLES)?;
        let cycle = |batches: std::ops::Range<i64>| {
            batches.map(|b| (0..10).map(|i| (b * 10 + i) % 50).collect())
        };
        sink_batches(&mut sink, cycle(0..3));
        assert!(local_state(&sink).bypass.is_some());
        sink_batches(&mut sink, cycle(3..100));
        let state = local_state(&sink);
        assert!(state.bypass.is_none());
        assert_eq!(state.capacity, 64);
        assert_eq!(state.table.num_groups(), 50);
        assert_eq!(finish(sink)?, (0..50).map(|k| (k, 20)).collect());
        Ok(())
    }

    #[test]
    fn large_partitions_are_finalized_in_their_sub_partitions() -> Result<()> {
        // 70,000 keys of partition 9, as the partial states of two flushes and
        // as input rows (in their sub-partitions), of `COUNT(*) GROUP BY k`.
        let candidates: ArrayRef = Arc::new(Int64Array::from_iter_values(0..6_000_000));
        let mut hashes = vec![0; candidates.len()];
        create_hashes(&[candidates], &PARTITION_HASH, &mut hashes)?;
        let keys = hashes
            .iter()
            .enumerate()
            .filter(|(_, hash)| partition_of(**hash) == 9)
            .map(|(key, _)| key as i64)
            .take(70_000)
            .collect::<Vec<_>>();
        assert_eq!(keys.len(), 70_000);
        let states = |keys: &[i64], count: i64| GroupedStates {
            keys: vec![Arc::new(Int64Array::from(keys.to_vec()))],
            states: vec![vec![Arc::new(Int64Array::from(vec![count; keys.len()]))]],
        };
        let (spec, ..) = count_by_key(SMALL_TABLES)?;
        for (num_keys, split) in [(70_000, true), (20_000, false)] {
            let keys = &keys[..num_keys];
            let mut partition = Partition::default();
            partition.states.push(states(&keys[..num_keys * 4 / 7], 2));
            partition.states.push(states(&keys[num_keys * 3 / 7..], 3));
            let rows: ArrayRef =
                Arc::new(Int64Array::from(keys[..num_keys / 7].to_vec()));
            let (mut hashes, mut partitions) = (vec![], vec![]);
            hash_keys(&[Arc::clone(&rows)], &mut hashes, &mut partitions)?;
            let splits = hashes
                .iter()
                .map(|&hash| sub_partition_of(hash) % NUM_SPLITS as u8)
                .collect::<Vec<_>>();
            for (split, columns) in scatter(vec![vec![rows]], &splits, NUM_SPLITS)? {
                partition.rows[split].push(columns);
            }
            assert_eq!(partition.num_rows() > SPLIT_ROWS, split);
            let outputs = finalize_partition(&spec, partition)?;
            assert_eq!(outputs.len(), if split { NUM_SPLITS } else { 1 });
            let mut counts = BTreeMap::new();
            for output in &outputs {
                let keys = output.column(0).as_primitive::<Int64Type>();
                let values = output.column(1).as_primitive::<Int64Type>();
                let mut splits = BTreeMap::<u8, usize>::new();
                for (&key, &count) in keys.values().iter().zip(values.values()) {
                    assert!(counts.insert(key, count).is_none(), "{key} twice");
                    let key: ArrayRef = Arc::new(Int64Array::from(vec![key]));
                    let (mut hashes, mut partitions) = (vec![], vec![]);
                    hash_keys(&[key], &mut hashes, &mut partitions)?;
                    *splits.entry(sub_partition_of(hashes[0])).or_default() += 1;
                }
                // The groups of every output are those of a sub-partition.
                assert!(!split || splits.len() == 1, "{splits:?}");
            }
            let expected = keys
                .iter()
                .enumerate()
                .map(|(i, &key)| {
                    let count = 2 * i64::from(i < num_keys * 4 / 7)
                        + 3 * i64::from(i >= num_keys * 3 / 7)
                        + i64::from(i < num_keys / 7);
                    (key, count)
                })
                .collect::<BTreeMap<_, _>>();
            assert_eq!(counts, expected);
        }
        Ok(())
    }

    #[test]
    fn distinct_estimates_are_close() -> Result<()> {
        for distinct in [0, 1, 10, 1000, 30_000, 200_000] {
            let mut estimate = DistinctEstimate::default();
            // Every value three times, in batches.
            for _ in 0..3 {
                let values: ArrayRef =
                    Arc::new(Int64Array::from_iter_values(0..distinct));
                let mut hashes = vec![0; distinct as usize];
                create_hashes(&[values], &PARTITION_HASH, &mut hashes)?;
                for batch in hashes.chunks(1000) {
                    estimate.add(batch);
                }
            }
            let error = estimate.estimate().abs_diff(distinct as usize) as f64;
            assert!(error <= 2.0 + distinct as f64 * 0.1, "{distinct}: {error}");
        }
        Ok(())
    }

    /// `n` values whose groups all fall into partition `target`.
    fn keys_of_partition(target: usize, n: usize) -> Vec<i64> {
        (0..)
            .filter(|&i| {
                let key: ArrayRef = Arc::new(Int64Array::from(vec![i]));
                partition_of_row(&[key], 0) == target
            })
            .take(n)
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn large_partitions_are_emitted_in_chunks() -> Result<()> {
        // 5000 groups in a single partition: one thread finalizes them and
        // emits them in chunks of at most `DEFAULT_VECTOR_SIZE` rows.
        let keys = keys_of_partition(5, 5000);
        let schema = key_schema();
        let batch = |keys: &[i64]| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(keys.to_vec()))],
            )
        };
        // Every key twice, in different batches of the input.
        let partitions = vec![vec![
            batch(&keys[..3000])?,
            batch(&keys)?,
            batch(&keys[3000..])?,
        ]];
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let input =
                MemorySourceConfig::try_new_exec(&partitions, Arc::clone(&schema), None)?;
            let count = AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
                .schema(Arc::clone(&schema))
                .alias("count(*)")
                .build()?;
            Ok(Arc::new(AggregateExec::try_new(
                AggregateMode::Single,
                PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
                vec![Arc::new(count)],
                vec![None],
                input,
                Arc::clone(&schema),
            )?))
        };
        let plan = make()?;
        let root = lower(&plan)?;
        assert!(root.is::<PhysicalHashAggregate>());
        let output = Executor::collect(root, Arc::new(TaskContext::default()), 4).await?;
        assert!(output.len() >= 3);
        assert!(
            output
                .iter()
                .all(|batch| (1..=DEFAULT_VECTOR_SIZE).contains(&batch.num_rows()))
        );
        let expected =
            datafusion_physical_plan::collect(make()?, Arc::new(TaskContext::default()))
                .await?;
        assert_eq!(
            canonical_batch(&plan.schema(), &output, false)?,
            canonical_batch(&plan.schema(), &expected, false)?
        );
        assert_eq!(
            output.iter().map(RecordBatch::num_rows).sum::<usize>(),
            5000
        );
        Ok(())
    }

    /// Register table `u`: 3,000 rows in 3 partitions of batches of 50 rows
    /// (and a few of other sizes), with a high-cardinality key `k` (with
    /// nulls), strings `s` and `sv` (a string view), values `v`, decimals `d`
    /// and booleans `b`, all with nulls.
    fn register_u(ctx: &SessionContext) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("sv", DataType::Utf8View, true),
            Field::new("v", DataType::Int64, true),
            Field::new("d", DataType::Decimal128(10, 2), true),
            Field::new("b", DataType::Boolean, true),
        ]));
        let batch = |rows: std::ops::Range<i64>| -> Result<RecordBatch> {
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter(
                    rows.clone()
                        .map(|i| (i % 97 != 13).then_some((i * 7_919) % 2_999)),
                )),
                Arc::new(StringArray::from_iter(rows.clone().map(|i| {
                    (i % 31 != 7).then(|| format!("s{}", (i * 13) % 1_777))
                }))),
                Arc::new(StringViewArray::from_iter(rows.clone().map(|i| {
                    (i % 29 != 5).then(|| format!("a string view of {}", i % 2_003))
                }))),
                Arc::new(Int64Array::from_iter(
                    rows.clone()
                        .map(|i| (i % 13 != 4).then_some((i * 37) % 1_009 - 400)),
                )),
                Arc::new(
                    Decimal128Array::from_iter(rows.clone().map(|i| {
                        (i % 17 != 9).then_some(i128::from((i * 101) % 10_007))
                    }))
                    .with_precision_and_scale(10, 2)?,
                ),
                Arc::new(BooleanArray::from_iter(
                    rows.map(|i| (i % 11 != 2).then_some(i % 3 == 0)),
                )),
            ];
            Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
        };
        let batches = |rows: std::ops::Range<i64>| -> Result<Vec<RecordBatch>> {
            rows.clone()
                .step_by(50)
                .map(|start| batch(start..(start + 50).min(rows.end)))
                .collect()
        };
        let mut first = batches(0..1_400)?;
        first.push(batch(1_400..1_401)?);
        first.extend(batches(1_401..1_800)?);
        let partitions = vec![first, batches(1_800..2_950)?, batches(2_950..3_000)?];
        ctx.register_table(
            "u",
            Arc::new(datafusion::datasource::MemTable::try_new(
                schema, partitions,
            )?),
        )?;
        Ok(())
    }

    /// The aggregation at the top of `plan` (below projections or sorts),
    /// the plan that [`PhysicalHashAggregate::try_lower`] lowers.
    fn aggregation(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
            && !matches!(
                aggregate.mode(),
                AggregateMode::Partial | AggregateMode::PartialReduce
            )
        {
            return Some(Arc::clone(plan));
        }
        plan.children().into_iter().find_map(aggregation)
    }

    /// Assert that the aggregation of `sql` (see [`aggregation`]) produces
    /// DataFusion's results with thread-local tables of capacity `capacity`
    /// on `threads` threads, and that its rows may bypass the tables.
    async fn assert_bypassed_parity(
        ctx: &SessionContext,
        sql: &str,
        capacity: TableCapacity,
        threads: usize,
    ) -> Result<()> {
        let plan = async || -> Result<Arc<dyn ExecutionPlan>> {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            Ok(aggregation(&plan).expect("an aggregation"))
        };
        let expected_plan = plan().await?;
        let expected =
            datafusion_physical_plan::collect(Arc::clone(&expected_plan), ctx.task_ctx())
                .await?;
        let aggregate = PhysicalHashAggregate::try_lower(&plan().await?, lower)?
            .expect("a lowered aggregation")
            .with_table_capacity(capacity);
        assert!(aggregate.spec().bypass, "{sql}");
        let actual =
            Executor::collect(Arc::new(aggregate), ctx.task_ctx(), threads).await?;
        let schema = expected_plan.schema();
        assert_eq!(
            canonical_batch(&schema, &actual, false)?,
            canonical_batch(&schema, &expected, false)?,
            "{sql} on {threads} threads with tables of {capacity:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bypassed_rows_are_aggregated_like_datafusion() -> Result<()> {
        let queries = [
            "SELECT k, COUNT(*), SUM(v), MIN(s), MAX(sv), AVG(d), BOOL_OR(b) FROM u \
             GROUP BY k",
            "SELECT s, sv, COUNT(v), MAX(k), MIN(d) FROM u GROUP BY s, sv",
            "SELECT k, d, b, COUNT(*) FROM u WHERE v > -300 GROUP BY k, d, b",
            // Filters, which may select all rows of a chunk or none, and whose
            // rows are the only ones that the arguments are evaluated on.
            "SELECT k, COUNT(*) FILTER (WHERE v > 0), SUM(v) FILTER (WHERE s IS NULL), \
             MIN(d) FILTER (WHERE false), MAX(s) FILTER (WHERE true), \
             SUM(1000 / v) FILTER (WHERE v <> 0) FROM u GROUP BY k",
            // Groups whose rows bypass the tables of several threads.
            "SELECT k % 97 AS m, COUNT(DISTINCT v), MEDIAN(v), APPROX_DISTINCT(s), \
             SUM(DISTINCT d) FROM u GROUP BY k % 97",
            "SELECT k % 50 AS m, ARRAY_AGG(k ORDER BY k), FIRST_VALUE(sv ORDER BY v, k) \
             FROM u GROUP BY k % 50",
            "SELECT sv, COUNT(*), MAX(s), BIT_XOR(v) FROM u GROUP BY sv",
        ];
        let tables = [
            SMALL_TABLES,
            TableCapacity {
                initial: 64,
                maximum: 256,
            },
        ];
        for target_partitions in [1, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = datafusion_context(config.clone());
            register_u(&ctx)?;
            for sql in queries {
                for capacity in tables {
                    for threads in [1, 4] {
                        assert_bypassed_parity(&ctx, sql, capacity, threads).await?;
                    }
                }
                // And through the planner, with the default and with tiny
                // tables.
                let setup = async |ctx: &SessionContext| register_u(ctx);
                assert_aggregate_parity_with_setup(sql, config.clone(), setup).await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn flushes_of_top_k_aggregations_keep_their_top_groups() -> Result<()> {
        // `SELECT k FROM t GROUP BY k ORDER BY k DESC LIMIT 3` over distinct
        // keys: every flush of a table scatters 3 groups only.
        let schema = key_schema();
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![],
            vec![],
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?,
            Arc::clone(&schema),
        )?
        .try_optimize_topk(3, "k", SortOptions::new(true, false))
        .expect("a top-k aggregation")
        .data;
        let input = Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(
            aggregate.input(),
        )));
        let spec = PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?.spec();
        assert!(spec.top_k.is_some() && !spec.bypass);
        let global = RadixHashAggregate::global_sink_state()?;
        // Two threads.
        let mut locals = (0..2)
            .map(|_| RadixHashAggregate::local_sink_state(&spec, SMALL_TABLES))
            .collect::<Result<Vec<_>>>()?;
        for batch in 0..30 {
            let keys = (0..10).map(|i| (batch * 10 + i) * 7 % 300);
            let local = &mut locals[(batch % 2) as usize];
            RadixHashAggregate::sink(
                &spec,
                key_batch(keys)?,
                global.as_ref(),
                local.as_mut(),
            )?;
        }
        for local in &mut locals {
            RadixHashAggregate::combine(&spec, global.as_ref(), local.as_mut())?;
        }
        let sink = global.downcast_ref::<RadixSinkState>().unwrap();
        let (flushes, groups) =
            sink.partitions
                .iter()
                .fold((0, 0), |(flushes, groups), data| {
                    let data = data.lock();
                    let groups = groups
                        + data
                            .states
                            .iter()
                            .map(GroupedStates::num_groups)
                            .sum::<usize>();
                    (flushes + data.states.len(), groups)
                });
        assert!(flushes > 0);
        // At most 3 groups per flush of a table, out of 300.
        assert!(groups <= 3 * 300 / 16 + 6, "{groups}");
        assert!(matches!(
            RadixHashAggregate::finalize(global.as_ref())?,
            SinkFinalize::Ready
        ));
        // The threads that finalize the partitions emit the top 3 groups once.
        let source = RadixHashAggregate::global_source_state(&spec, Arc::clone(&global))?;
        let mut keys: Vec<i64> = vec![];
        let mut sources = (0..2)
            .map(|_| RadixHashAggregate::local_source_state())
            .collect::<Result<Vec<_>>>()?;
        let mut finished = [false; 2];
        while !finished.iter().all(|&finished| finished) {
            for (thread, local) in sources.iter_mut().enumerate() {
                if finished[thread] {
                    continue;
                }
                match RadixHashAggregate::get_data(
                    &spec,
                    source.as_ref(),
                    local.as_mut(),
                )? {
                    SourceResult::Chunk(chunk) => keys.extend(
                        chunk
                            .compact()?
                            .column(0)
                            .as_primitive::<Int64Type>()
                            .values(),
                    ),
                    _ => finished[thread] = true,
                }
            }
        }
        keys.sort_unstable();
        assert_eq!(keys, [297, 298, 299]);
        Ok(())
    }
}
