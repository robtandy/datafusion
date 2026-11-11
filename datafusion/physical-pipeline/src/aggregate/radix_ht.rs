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
//! these groups of the table, the rows that cannot make their groups among
//! them are not aggregated, and the final top groups of every partition are
//! merged into the top groups of the aggregation, which the thread that
//! finalizes the last partition emits. An aggregation with a soft limit of
//! `n` groups (see `AggregateSpec::soft_limit`) only needs any `n` groups:
//! a thread whose table holds `n` groups finishes the sink, and the threads
//! of its pipelines stop.
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
use arrow::compute::{filter, filter_record_batch, interleave, take};
use arrow::datatypes::{BinaryViewType, ByteViewType, DataType, StringViewType};
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::{Result, internal_err};
use datafusion_physical_expr::PhysicalExpr;
use parking_lot::Mutex;

use super::grouped_ht::{GroupedHashTable, GroupedStates, evaluate_rows};
use super::hash_aggregate::{
    AggregateSpec, InputBuffer, TopK, TopKOrder, rebuilt_literal,
};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    GlobalSinkState, GlobalSourceState, LocalSinkState, LocalSourceState, SinkFinalize,
    SinkResult, SourceResult, downcast_state, downcast_state_mut,
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

/// The most bytes of strings and binaries with 32-bit offsets of the rows
/// that bypassed a thread-local table that are scattered together (see
/// [`BypassBuffer`]): the scatter gathers the rows of all buffered chunks into
/// a single array, whose offsets must not overflow.
const BYPASS_BYTES: usize = 64 << 20;

/// The bytes of strings and binaries with 32-bit offsets of the rows that
/// bypassed a thread-local table that are scattered together per row of the
/// initial capacity of the table, up to [`BYPASS_BYTES`] (which the default
/// capacity reaches).
const BYPASS_BYTES_PER_ROW: usize = 2 << 10;

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
    /// Whether the rows of the thread may bypass its table (see
    /// [`AggregateSpec::bypass`]).
    may_bypass: bool,
    /// Whether the aggregation has a limit (a top-k or a soft limit), whose
    /// rows take the path of [`Self::aggregate_bypassed_or_limited`].
    limited: bool,
    /// Whether the table of the thread holds the groups of the soft limit of
    /// the aggregation (see [`AggregateSpec::soft_limit`]): the thread
    /// finishes the sink.
    finished: bool,
    /// The rows that bypass the table, while the thread does not aggregate
    /// its rows (see [`Self::aggregate`]).
    bypass: Option<BypassBuffer>,
    /// The number of buffered rows that bypassed the table at which they are
    /// scattered.
    bypass_rows: usize,
    /// The most bytes of strings and binaries with 32-bit offsets of the
    /// rows that bypassed the table that are scattered together.
    bypass_bytes: usize,
    /// The partition hashes of the groups being flushed, and their
    /// partitions.
    hashes: Vec<u64>,
    partitions: Vec<u8>,
    /// For a top-k aggregation, the best cutoff of the groups that the thread
    /// flushed so far (see [`TopK::cutoff`]): the input rows whose value is
    /// worse are not aggregated.
    cutoff: Option<ArrayRef>,
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
    /// aggregating its rows (if they may, see [`AggregateSpec::bypass`]): the
    /// rows *bypass* the table, they are scattered to the radix partitions
    /// directly (see [`BypassBuffer`]),
    /// whose final aggregation aggregates them. Their hashes are added to the
    /// sketch of the flushed groups, and once it shows that the groups of
    /// the thread came back after all (by the same rule), the thread
    /// aggregates its rows again, in a table that holds its groups.
    ///
    /// The rows of a top-k aggregation that its cutoff rules out (see
    /// [`TopK::candidates`]) are not aggregated (the keys and arguments of its
    /// rows are evaluated once, to compare them with the cutoff and to
    /// aggregate them), and its table is also flushed once it aggregated as
    /// many rows as it can hold groups, even if it is not full: the flush of a
    /// table keeps its top groups, whose values are better than those of the
    /// other groups of the table. The thread of an
    /// aggregation with a soft limit finishes once its table holds the groups
    /// of the limit.
    fn aggregate(
        &mut self,
        spec: &AggregateSpec,
        global: &RadixSinkState,
        batch: &RecordBatch,
    ) -> Result<()> {
        // The common case, as small as it can be: the rows of the thread are
        // aggregated into its table.
        if self.bypass.is_some() || self.limited {
            return self.aggregate_bypassed_or_limited(spec, global, batch);
        }
        self.table.update(spec, batch)?;
        self.table_rows += batch.num_rows();
        if self.table.num_groups() < self.capacity {
            return Ok(());
        }
        self.flush_or_grow(spec, global)
    }

    /// [`Self::aggregate`] of a thread whose rows bypass its table, or of a
    /// limited aggregation.
    #[inline(never)]
    fn aggregate_bypassed_or_limited(
        &mut self,
        spec: &AggregateSpec,
        global: &RadixSinkState,
        batch: &RecordBatch,
    ) -> Result<()> {
        if let Some(bypass) = &mut self.bypass {
            let scattered =
                bypass.push(spec, global, batch, &mut self.distinct_flushed)?;
            self.flushed += batch.num_rows();
            let full = bypass.is_full(self.bypass_rows);
            if full {
                bypass.scatter(global)?;
            }
            // Once rows were scattered (because there were enough of them, or
            // of their bytes), the thread aggregates its rows again if its
            // groups came back: the rows that it buffered are scattered first.
            if (full || scattered)
                && let Some(capacity) = self.grown_capacity()
            {
                if let Some(mut bypass) = self.bypass.take() {
                    bypass.scatter(global)?;
                }
                self.capacity = capacity;
            }
            return Ok(());
        }
        // The rows of a top-k aggregation that its cutoff does not rule out,
        // whose keys and arguments are evaluated once (see
        // `top_k_candidates`).
        let candidates = match (&spec.top_k, &self.cutoff) {
            (Some(top_k), Some(cutoff)) => {
                Some(top_k_candidates(spec, top_k, cutoff, batch)?)
            }
            _ => None,
        };
        let num_rows = match &candidates {
            Some(columns) => columns.first().map_or(0, |column| column.len()),
            None => batch.num_rows(),
        };
        if num_rows == 0 {
            return Ok(());
        }
        match &candidates {
            Some(columns) => self.table.update_rows(spec, columns)?,
            None => self.table.update(spec, batch)?,
        }
        self.table_rows += num_rows;
        let num_groups = self.table.num_groups();
        if spec.soft_limit.is_some_and(|limit| num_groups >= limit) {
            self.finished = true;
        }
        if num_groups < self.capacity {
            // A table of a top-k aggregation that aggregated as many rows as
            // it can hold groups is flushed even if it is not full (if its
            // flush keeps a cutoff), so that the flushes of a thread whose
            // groups fit into its table set its cutoff too.
            if let Some(top_k) = &spec.top_k
                && self.table_rows >= self.capacity
                && num_groups >= top_k.limit
            {
                self.flush(spec, global)?;
            }
            return Ok(());
        }
        self.flush_or_grow(spec, global)
    }

    /// Grow the full table of the thread if its flushed groups came back,
    /// flush it otherwise (and bypass it from then on if its rows hardly
    /// repeat their groups), see [`Self::aggregate`].
    #[inline(never)]
    fn flush_or_grow(
        &mut self,
        spec: &AggregateSpec,
        global: &RadixSinkState,
    ) -> Result<()> {
        if let Some(capacity) = self.grown_capacity() {
            self.capacity = capacity;
            return Ok(());
        }
        let bypass = self.may_bypass
            && self.table.num_groups() as f64 > BYPASS_RATIO * self.table_rows as f64;
        self.flush(spec, global)?;
        if bypass {
            self.bypass = Some(BypassBuffer::new(self.bypass_bytes));
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
        if let Some(top_k) = &spec.top_k {
            if let Some(indices) = top_k.select(top_k_values(top_k, &states)?)? {
                states = states.take(&indices)?;
            }
            if let Some(cutoff) = top_k.cutoff(top_k_values(top_k, &states)?)? {
                let better = match &self.cutoff {
                    Some(current) => {
                        top_k.is_better(cutoff.as_ref(), current.as_ref())?
                    }
                    None => true,
                };
                if better {
                    self.cutoff = Some(cutoff);
                }
            }
        }
        hash_keys(&states.keys, &mut self.hashes, &mut self.partitions)?;
        self.flushed += states.num_groups();
        self.distinct_flushed.add(&self.hashes);
        global.add_states(states, &self.partitions)
    }
}

/// The values of the groups of `states`, partial states of the top-k
/// aggregation `top_k`, which order them: their keys, or the partial states
/// of their aggregate (its value).
fn top_k_values<'a>(top_k: &TopK, states: &'a GroupedStates) -> Result<&'a dyn Array> {
    let values = match top_k.order {
        TopKOrder::Key => states.keys.first(),
        TopKOrder::Aggregate => states.states.first().and_then(|state| state.first()),
    };
    match values {
        Some(values) => Ok(values.as_ref()),
        None => internal_err!("{OPERATOR} found no value to order the groups by"),
    }
}

/// The columns (see [`evaluate_rows`]) of the input rows of `batch` of the
/// top-k aggregation `top_k` of `spec` that its cutoff `cutoff` does not rule
/// out (see [`TopK::candidates`]): no columns if it rules out all rows.
///
/// The values that order the groups of the rows (their keys, or the argument
/// of the aggregate) are evaluated once for all rows, to compare them with the
/// cutoff, and those of the candidates are aggregated (they may be volatile);
/// the key of an aggregation by its aggregate is evaluated for the candidates
/// only.
fn top_k_candidates(
    spec: &AggregateSpec,
    top_k: &TopK,
    cutoff: &ArrayRef,
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    // A top-k aggregation has a single key, without grouping sets, and no
    // aggregates if it is ordered by its key, or a single `MIN` / `MAX` of a
    // single argument, without a filter or an order, if it is ordered by it.
    let ([(key, _)], true) = (spec.group_by.expr(), spec.group_by.is_single()) else {
        return internal_err!("{OPERATOR} found no single key of a top-k aggregation");
    };
    let argument = match (top_k.order, spec.update.as_slice()) {
        (TopKOrder::Key, []) => None,
        (TopKOrder::Aggregate, [update])
            if update.arguments.len() == 1 && update.filter.is_none() =>
        {
            Some(&update.arguments[0])
        }
        _ => {
            return internal_err!(
                "{OPERATOR} found other aggregates than those of a top-k aggregation"
            );
        }
    };
    let num_rows = batch.num_rows();
    let evaluate = |expr: &Arc<dyn PhysicalExpr>, batch: &RecordBatch| {
        expr.evaluate(batch)?.into_array(batch.num_rows())
    };
    // A literal argument is rebuilt, not evaluated (nor stored).
    let literal = argument.and_then(|argument| rebuilt_literal(argument));
    let values = match (argument, literal) {
        (None, _) => evaluate(key, batch)?,
        (Some(_), Some(literal)) => literal.to_array_of_size(num_rows)?,
        (Some(argument), None) => evaluate(argument, batch)?,
    };
    let selected = top_k.candidates(&values, cutoff)?;
    if selected
        .as_ref()
        .is_some_and(|selected| selected.true_count() == 0)
    {
        return Ok(vec![]);
    }
    let values = match &selected {
        Some(selected) => filter(values.as_ref(), selected)?,
        None => values,
    };
    if argument.is_none() {
        return Ok(vec![values]);
    }
    let candidates = match &selected {
        Some(selected) => &filter_record_batch(batch, selected)?,
        None => batch,
    };
    let mut columns = vec![evaluate(key, candidates)?];
    if literal.is_none() {
        columns.push(values);
    }
    Ok(columns)
}

/// The rows of a thread that bypass its table (see
/// [`RadixLocalState::aggregate`]), until they are scattered to the radix
/// partitions: the columns of the rows of their chunks (see
/// [`evaluate_rows`]), and the sub-partition of every row (see
/// [`sub_partition_of`]). The rows of many chunks are scattered at once, so
/// that the final aggregation of a partition aggregates large slices of its
/// rows; but the rows of several chunks have at most `max_bytes` bytes of
/// strings and binaries with 32-bit offsets, which the scatter gathers into
/// a single array, so that its offsets cannot overflow (a single chunk, whose
/// arrays hold its bytes already, may have more).
#[derive(Debug)]
struct BypassBuffer {
    chunks: Vec<Vec<ArrayRef>>,
    /// The sub-partition of every row, in the order of the chunks.
    partitions: Vec<u8>,
    /// The bytes of the values of the buffered columns with 32-bit offsets.
    offset_bytes: usize,
    /// The most bytes of the values of the buffered columns with 32-bit
    /// offsets of several chunks.
    max_bytes: usize,
    hashes: Vec<u64>,
}

impl BypassBuffer {
    fn new(max_bytes: usize) -> Self {
        Self {
            chunks: vec![],
            partitions: vec![],
            offset_bytes: 0,
            max_bytes,
            hashes: vec![],
        }
    }

    /// Buffer the input rows of `batch`: evaluate their group keys and the
    /// arguments of the aggregates, hash their keys, and add their hashes to
    /// `distinct`. If the buffered rows would have more than `max_bytes`
    /// bytes of values with 32-bit offsets with them, the buffered rows are
    /// scattered to the partitions of `global` first. Returns whether they
    /// were.
    fn push(
        &mut self,
        spec: &AggregateSpec,
        global: &RadixSinkState,
        batch: &RecordBatch,
        distinct: &mut DistinctEstimate,
    ) -> Result<bool> {
        let columns = evaluate_rows(spec, batch)?;
        let bytes = columns
            .iter()
            .map(|column| offset_bytes(column.as_ref()))
            .sum::<usize>();
        let scattered =
            !self.chunks.is_empty() && self.offset_bytes + bytes > self.max_bytes;
        if scattered {
            self.scatter(global)?;
        }
        let keys = &columns[..spec.group_by.expr().len()];
        self.hashes.clear();
        self.hashes.resize(batch.num_rows(), 0);
        create_hashes(keys, &PARTITION_HASH, &mut self.hashes)?;
        self.partitions
            .extend(self.hashes.iter().map(|&hash| sub_partition_of(hash)));
        distinct.add(&self.hashes);
        self.offset_bytes += bytes;
        self.chunks.push(columns);
        Ok(scattered)
    }

    /// Whether the buffered rows are to be scattered: once there are `rows`
    /// of them, or they have `max_bytes` bytes of values with 32-bit offsets.
    fn is_full(&self, rows: usize) -> bool {
        self.partitions.len() >= rows || self.offset_bytes >= self.max_bytes
    }

    /// Scatter the buffered rows to the partitions of `global`.
    fn scatter(&mut self, global: &RadixSinkState) -> Result<()> {
        if self.chunks.is_empty() {
            return Ok(());
        }
        debug_assert!(
            self.chunks.len() == 1 || self.offset_bytes <= self.max_bytes,
            "{} chunks with {} bytes of values with 32-bit offsets",
            self.chunks.len(),
            self.offset_bytes
        );
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
    /// batches of four times as many rows, and of at most
    /// [`BYPASS_BYTES_PER_ROW`] bytes of strings and binaries with 32-bit
    /// offsets per row of it (up to [`BYPASS_BYTES`]).
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
/// groups (rather than the top groups of every partition).
#[derive(Debug)]
struct TopKGroups {
    top_k: TopK,
    /// The top groups of the partitions that were finalized so far, sorted
    /// (see [`TopK::sort_groups`]).
    groups: Mutex<Option<RecordBatch>>,
    /// The cutoff of `groups` (see [`TopK::sorted_cutoff`]): the groups of
    /// the other partitions whose values are worse are not among the top
    /// groups.
    cutoff: Mutex<Option<ArrayRef>>,
    /// The number of partitions that were not finalized yet.
    remaining: AtomicUsize,
}

impl TopKGroups {
    fn new(top_k: TopK, partitions: usize) -> Self {
        Self {
            top_k,
            groups: Mutex::new(None),
            cutoff: Mutex::new(None),
            remaining: AtomicUsize::new(partitions),
        }
    }

    /// Add `outputs`, the groups of a finalized partition. Returns the top
    /// groups of all partitions once all of them were added.
    ///
    /// The groups of a partition that may be among the top groups (not worse
    /// than the cutoff of the groups added so far) are selected and sorted
    /// first, by the thread that finalized the partition, so that merging
    /// them with the groups added so far, one partition after the other,
    /// gathers the top groups only.
    fn add(&self, outputs: Vec<RecordBatch>) -> Result<Option<RecordBatch>> {
        let column = self.top_k.output_column();
        for output in outputs {
            let cutoff = self.cutoff.lock().clone();
            let output = match &cutoff {
                Some(cutoff) => {
                    match self.top_k.candidates(output.column(column), cutoff)? {
                        Some(selected) => filter_record_batch(&output, &selected)?,
                        None => output,
                    }
                }
                None => output,
            };
            if output.num_rows() == 0 {
                continue;
            }
            let output = self.top_k.select_sorted_groups(&output)?;
            let mut groups = self.groups.lock();
            let merged = match groups.take() {
                Some(groups) => self.top_k.merge_sorted_groups(&groups, &output)?,
                None => output,
            };
            if let Some(cutoff) = self.top_k.sorted_cutoff(&merged) {
                *self.cutoff.lock() = Some(cutoff);
            }
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

    /// The local sink state of a thread, with a table of capacity `capacity`,
    /// whose rows may bypass the table if `may_bypass` (see
    /// [`AggregateSpec::bypass`]).
    ///
    /// The table of a top-k aggregation holds twice its limit at least (up
    /// to the maximum capacity), so that every flush keeps the best half of
    /// its groups and sets a cutoff (see [`TopK::cutoff`]), which rules out
    /// most of the rows after it: a table that holds fewer groups than the
    /// limit would keep all of them, and never set one.
    pub(crate) fn local_sink_state(
        spec: &AggregateSpec,
        capacity: TableCapacity,
        may_bypass: bool,
    ) -> Result<Box<dyn LocalSinkState>> {
        let mut initial = capacity.initial.max(1);
        if let Some(top_k) = &spec.top_k {
            let room = top_k.limit.saturating_mul(2).min(capacity.maximum);
            initial = initial.max(room);
        }
        Ok(Box::new(RadixLocalState {
            table: GroupedHashTable::try_new_update(spec)?,
            capacity: initial,
            maximum_capacity: capacity.maximum,
            table_rows: 0,
            flushed: 0,
            distinct_flushed: DistinctEstimate::default(),
            buffer: InputBuffer::default(),
            may_bypass,
            limited: spec.top_k.is_some() || spec.soft_limit.is_some(),
            finished: false,
            bypass: None,
            bypass_rows: initial.saturating_mul(4),
            bypass_bytes: initial
                .saturating_mul(BYPASS_BYTES_PER_ROW)
                .min(BYPASS_BYTES),
            hashes: vec![],
            partitions: vec![],
            cutoff: None,
        }))
    }

    /// Aggregate the input rows of `batch`. Finishes the sink once the table
    /// of the thread holds the groups of the soft limit of the aggregation.
    pub(crate) fn sink(
        spec: &AggregateSpec,
        batch: RecordBatch,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let local = downcast_state_mut::<RadixLocalState>(lstate.as_any_mut(), OPERATOR)?;
        if spec.input_order.is_some() {
            local.buffer.push(batch);
            return Ok(SinkResult::NeedMoreInput);
        }
        let global = downcast_state::<RadixSinkState>(gstate.as_any(), OPERATOR)?;
        local.aggregate(spec, global, &batch)?;
        Ok(match local.finished {
            true => SinkResult::Finished,
            false => SinkResult::NeedMoreInput,
        })
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
        let top_k = spec
            .top_k
            .map(|top_k| TopKGroups::new(top_k, partitions.len()));
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
    let mut next = offsets.clone();
    let sorted = if let [columns] = chunks.as_slice() {
        // The rows of a single chunk (e.g. the groups of a flushed table) are
        // gathered in the order of the partitions by Arrow's `take`, whose
        // kernels write every value once (and its validity by words).
        let mut indices = vec![0u32; num_rows];
        for (row, &partition) in partitions.iter().enumerate() {
            let position = &mut next[usize::from(partition)];
            indices[*position] = row as u32;
            *position += 1;
        }
        let indices = UInt32Array::from(indices);
        columns
            .iter()
            .map(|column| Ok(take(column.as_ref(), &indices, None)?))
            .collect::<Result<Vec<_>>>()?
    } else {
        scatter_chunks(&chunks, partitions, &mut next)?
    };
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

/// The columns of the rows of `chunks` (several chunks) in the order of their
/// partitions `partitions`, `next` the first position of every partition:
/// every column is reordered once (see [`scatter_column`]).
fn scatter_chunks(
    chunks: &[Vec<ArrayRef>],
    partitions: &[u8],
    next: &mut [usize],
) -> Result<Vec<ArrayRef>> {
    // The position of every row in the order of the partitions.
    let positions = partitions
        .iter()
        .map(|&partition| {
            let position = &mut next[usize::from(partition)];
            *position += 1;
            (*position - 1) as u32
        })
        .collect::<Vec<_>>();
    let num_columns = chunks.first().map_or(0, Vec::len);
    (0..num_columns)
        .map(|column| {
            let arrays = chunks
                .iter()
                .map(|columns| columns[column].as_ref())
                .collect::<Vec<_>>();
            scatter_column(&arrays, &positions)
        })
        .collect()
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
/// A partition of more than [`SPLIT_ROWS`] rows, some of which bypassed the
/// thread-local tables, is finalized in its [`NUM_SPLITS`] sub-partitions,
/// one after the other, so that their hash tables are smaller and stay cache
/// resident: its input rows are scattered to them already, its partial states
/// are scattered (by the next [`SPLIT_BITS`] bits of the partition hashes of
/// their keys) first. A partition of partial states only is finalized at
/// once: hashing and scattering all of its states again costs more than the
/// smaller tables save.
fn finalize_partition(
    spec: &AggregateSpec,
    partition: Partition,
) -> Result<Vec<RecordBatch>> {
    if partition.num_rows() <= SPLIT_ROWS || partition.rows.iter().all(Vec::is_empty) {
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
    use datafusion::prelude::{SessionConfig, SessionContext};
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
    use crate::test_utils::{
        assert_byte_eq_with_datafusion_with_config, canonical_batch, datafusion_context,
        test_session_config,
    };

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

    /// The `Partial` and `Final` stages of `aggregate`, a `Single`
    /// aggregation.
    fn two_stages(
        aggregate: &AggregateExec,
    ) -> Result<(Arc<AggregateExec>, AggregateExec)> {
        let partial = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            aggregate.group_expr().clone(),
            aggregate.aggr_expr().to_vec(),
            aggregate.filter_expr().to_vec(),
            Arc::clone(aggregate.input()),
            aggregate.input_schema(),
        )?);
        let last = AggregateExec::try_new(
            AggregateMode::Final,
            aggregate.group_expr().as_final(),
            aggregate.aggr_expr().to_vec(),
            vec![None; aggregate.aggr_expr().len()],
            Arc::clone(&partial) as Arc<dyn ExecutionPlan>,
            aggregate.input_schema(),
        )?;
        Ok((partial, last))
    }

    /// The spec of the two stages of `aggregate`, a `Single` aggregation (see
    /// [`two_stages`]).
    fn two_stage_spec(aggregate: &AggregateExec) -> Result<Arc<AggregateSpec>> {
        let (partial, last) = two_stages(aggregate)?;
        let input = Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(
            aggregate.input(),
        )));
        Ok(PhysicalHashAggregate::try_new_two_stage(&partial, &last, input)?.spec())
    }

    /// The sink of `COUNT(*) GROUP BY k` (of two stages), with tables of
    /// capacity `capacity`, whose rows may bypass the tables if `bypass` (see
    /// `AggregateSpec::bypass`), with counts to check.
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
        let spec = two_stage_spec(&aggregate)?;
        assert!(spec.bypass);
        let local = RadixHashAggregate::local_sink_state(&spec, capacity, bypass)?;
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

    /// The schema of keys `k` and binaries `b`.
    fn binary_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("b", DataType::Binary, false),
        ]))
    }

    /// The sink of `COUNT(*), MAX(b) GROUP BY k` (of two stages), with tables
    /// of capacity `capacity`, whose rows may bypass the tables (see
    /// [`count_by_key_with`]): its bypassed rows hold the binaries.
    fn max_binaries_by_key(capacity: TableCapacity) -> Result<Sink> {
        use datafusion::functions_aggregate::min_max::max_udaf;
        let schema = binary_schema();
        let count = AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
            .schema(Arc::clone(&schema))
            .alias("count(*)")
            .build()?;
        let max = AggregateExprBuilder::new(max_udaf(), vec![col("b", &schema)?])
            .schema(Arc::clone(&schema))
            .alias("max(b)")
            .build()?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![Arc::new(count), Arc::new(max)],
            vec![None, None],
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?,
            schema,
        )?;
        let mut spec = two_stage_spec(&aggregate)?;
        Arc::get_mut(&mut spec).expect("a single owner").bypass = true;
        let local = RadixHashAggregate::local_sink_state(&spec, capacity, spec.bypass)?;
        Ok((spec, local, RadixHashAggregate::global_sink_state()?))
    }

    /// A batch of keys `keys`, with binaries of `len` bytes.
    fn binary_batch(keys: std::ops::Range<i64>, len: usize) -> Result<RecordBatch> {
        let values = keys.clone().map(|key| vec![key as u8; len]);
        Ok(RecordBatch::try_new(
            binary_schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(keys)),
                Arc::new(arrow::array::BinaryArray::from_iter_values(values)),
            ],
        )?)
    }

    #[test]
    fn bypassed_rows_are_scattered_before_chunks_whose_bytes_they_cannot_take()
    -> Result<()> {
        // Tables of 16 groups: the binaries of several chunks of rows that
        // bypass them are scattered together while they have at most 32 KiB.
        let mut sink = max_binaries_by_key(SMALL_TABLES)?;
        let max_bytes = 16 * BYPASS_BYTES_PER_ROW;
        let sink_batch = |sink: &mut Sink, batch: RecordBatch| -> Result<()> {
            let (spec, local, global) = sink;
            RadixHashAggregate::sink(spec, batch, global.as_ref(), local.as_mut())?;
            Ok(())
        };
        let buffered = |sink: &Sink| {
            let bypass = local_state(sink).bypass.as_ref().expect("bypassed rows");
            assert_eq!(bypass.max_bytes, max_bytes);
            (bypass.chunks.len(), bypass.offset_bytes)
        };
        // Distinct keys fill the table: the next rows bypass it.
        sink_batch(&mut sink, binary_batch(0..20, 1)?)?;
        assert_eq!(buffered(&sink), (0, 0));
        sink_batch(&mut sink, binary_batch(100..110, 1_000)?)?;
        assert_eq!(buffered(&sink), (1, 10_000));
        // Together with the buffered rows, more than 32 KiB: the buffered rows
        // are scattered first.
        sink_batch(&mut sink, binary_batch(200..202, 12_000)?)?;
        assert_eq!(buffered(&sink), (1, 24_000));
        // More than 32 KiB on their own: scattered on their own.
        sink_batch(&mut sink, binary_batch(300..301, 40_000)?)?;
        assert_eq!(buffered(&sink), (0, 0));
        sink_batch(&mut sink, binary_batch(400..410, 100)?)?;
        assert_eq!(buffered(&sink), (1, 1_000));
        // Every scattered set of rows has the rows of a single chunk here.
        let global = sink.2.downcast_ref::<RadixSinkState>().unwrap();
        let mut scattered = 0;
        for partition in &global.partitions {
            for columns in partition.lock().rows.iter().flatten() {
                let keys = columns[0].as_primitive::<Int64Type>().values();
                let chunk = |key: i64| key / 100;
                assert!(keys.iter().all(|&key| chunk(key) == chunk(keys[0])));
                scattered += keys.len();
            }
        }
        assert_eq!(scattered, 13);
        let expected = (0..20)
            .chain(100..110)
            .chain(200..202)
            .chain(300..301)
            .chain(400..410)
            .map(|key| (key, 1))
            .collect();
        assert_eq!(finish(sink)?, expected);
        Ok(())
    }

    #[test]
    fn threads_whose_groups_come_back_aggregate_again_after_scatters_of_bytes()
    -> Result<()> {
        // Tables of 16 groups, up to 64: 40 keys cycling in batches of ten rows
        // of 1,700-byte binaries. Two batches have more than 32 KiB together,
        // so every batch scatters the rows before it, and the buffered rows
        // never get to 64: the thread aggregates its rows again once its groups
        // came back (after the scatters of their bytes), in a table of 64
        // groups, which holds them all.
        let mut sink = max_binaries_by_key(SMALL_TABLES)?;
        let sink_batch = |sink: &mut Sink, batch: RecordBatch| -> Result<()> {
            let (spec, local, global) = sink;
            RadixHashAggregate::sink(spec, batch, global.as_ref(), local.as_mut())?;
            Ok(())
        };
        let batch = |b: i64| binary_batch((b * 10) % 40..(b * 10) % 40 + 10, 1_700);
        sink_batch(&mut sink, batch(0)?)?;
        sink_batch(&mut sink, batch(1)?)?;
        assert!(local_state(&sink).bypass.is_some());
        for b in 2..20 {
            sink_batch(&mut sink, batch(b)?)?;
        }
        let state = local_state(&sink);
        assert!(state.bypass.is_none());
        assert_eq!(state.capacity, 64);
        assert_eq!(state.table.num_groups(), 40);
        assert_eq!(finish(sink)?, (0..40).map(|key| (key, 5)).collect());
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

    /// `COUNT(<1,100,000-byte string>) FILTER (WHERE p) GROUP BY k`, see
    /// [`count_of_literal`]. As many copies of the literal as there are rows of
    /// a chunk would overflow the offsets of an array of strings: DataFusion
    /// builds them for the selected rows only.
    fn count_of_long_literal(
        selected: fn(i64) -> bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let literal = datafusion_common::ScalarValue::Utf8(Some("x".repeat(1_100_000)));
        count_of_literal(literal, selected)
    }

    /// `COUNT(<literal>) FILTER (WHERE p) GROUP BY k` over a single partition:
    /// 32,768 distinct keys (which fill the thread-local table of a thread,
    /// whose rows then bypass it, if they may), followed by 2,048 rows of a new
    /// key; `p` is true for the rows of `selected`.
    fn count_of_literal(
        literal: datafusion_common::ScalarValue,
        selected: fn(i64) -> bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("p", DataType::Boolean, false),
        ]));
        let batch = |keys: Vec<i64>, rows: std::ops::Range<i64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(BooleanArray::from_iter(
                        rows.map(|row| Some(selected(row))),
                    )),
                ],
            )
        };
        let partitions = vec![vec![
            batch((0..32_768).collect(), 0..32_768)?,
            batch(vec![100_000; 2_048], 32_768..34_816)?,
        ]];
        let input =
            MemorySourceConfig::try_new_exec(&partitions, Arc::clone(&schema), None)?;
        let count = AggregateExprBuilder::new(count_udaf(), vec![lit(literal)])
            .schema(Arc::clone(&schema))
            .alias("count(literal)")
            .build()?;
        let single = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![Arc::new(count)],
            vec![Some(col("p", &schema)?)],
            input,
            schema,
        )?;
        // Of two stages, whose rows may bypass the tables.
        Ok(Arc::new(two_stages(&single)?.1))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn literal_arguments_of_filtered_aggregates_are_built_for_their_rows()
    -> Result<()> {
        // No row passes the filter, and a few do; on 1, 4 and 16 threads; whose
        // rows do not bypass the tables (the regular path evaluates the
        // arguments for the rows that pass the filter only, like DataFusion),
        // and whose rows bypass them (see `AggregateSpec::bypass`, and
        // `evaluate_row_aligned`).
        let selections: [fn(i64) -> bool; 2] = [|_| false, |row| row % 1_000 == 7];
        let runs = selections.into_iter().flat_map(|selected| {
            [1, 4, 16]
                .into_iter()
                .flat_map(|threads| [(threads, false), (threads, true)])
                .map(move |(threads, bypass)| (selected, threads, bypass))
        });
        for (selected, threads, bypass) in runs {
            let plan = count_of_long_literal(selected)?;
            let aggregate = PhysicalHashAggregate::try_lower(&plan, lower)?
                .expect("a native aggregation");
            assert!(aggregate.spec().bypass);
            let aggregate = if bypass {
                aggregate
            } else {
                aggregate.without_bypass()
            };
            assert_eq!(aggregate.spec().bypass, bypass);
            let task_ctx = Arc::new(TaskContext::default());
            let actual =
                Executor::collect(Arc::new(aggregate), Arc::clone(&task_ctx), threads)
                    .await?;
            let expected = datafusion_physical_plan::collect(
                count_of_long_literal(selected)?,
                task_ctx,
            )
            .await?;
            let schema = plan.schema();
            assert_eq!(
                canonical_batch(&schema, &actual, false)?,
                canonical_batch(&schema, &expected, false)?
            );
            assert_eq!(
                actual.iter().map(RecordBatch::num_rows).sum::<usize>(),
                32_769
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wide_fixed_size_binary_arguments_do_not_bypass_the_tables() -> Result<()> {
        // `COUNT(<a NULL fixed-size binary of i32::MAX bytes>) FILTER (WHERE p)`
        // with `p` false: aligned with the rows of a chunk, the argument would
        // hold a `NULL` as wide as a value for each of its 2,048 rows (4 TiB),
        // where DataFusion (and the regular path) evaluate it for no row.
        let literal = datafusion_common::ScalarValue::FixedSizeBinary(i32::MAX, None);
        let plan = count_of_literal(literal.clone(), |_| false)?;
        let root = lower(&plan)?;
        let aggregate = root
            .downcast_ref::<PhysicalHashAggregate>()
            .expect("a native aggregation");
        // Checked before the execution, which would fail to allocate otherwise
        // (see `is_flat_type`, and `AggregateSpec::bypass`).
        assert!(!aggregate.spec().bypass);
        // Like DataFusion: 32,769 groups that count no row.
        let task_ctx = Arc::new(TaskContext::default());
        let actual =
            Executor::collect(Arc::clone(&root), Arc::clone(&task_ctx), 1).await?;
        let expected = datafusion_physical_plan::collect(
            count_of_literal(literal.clone(), |_| false)?,
            task_ctx,
        )
        .await?;
        let schema = plan.schema();
        assert_eq!(
            canonical_batch(&schema, &actual, false)?,
            canonical_batch(&schema, &expected, false)?
        );
        let counts = actual
            .iter()
            .map(|batch| {
                batch
                    .column(1)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>()
            .concat();
        assert_eq!(counts.len(), 32_769);
        assert!(counts.iter().all(|&count| count == 0));
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

    /// `config`, trusting the built-in aggregate functions of DataFusion (see
    /// `TrustedAggregateFunctions`), which changes nothing.
    fn trusting(config: SessionConfig) -> SessionConfig {
        let functions =
            datafusion::functions_aggregate::all_default_aggregate_functions();
        config.with_option_extension(PhysicalHashAggregate::trusted_aggregate_functions(
            &functions,
        ))
    }

    /// Assert that the aggregation of `sql` (see [`aggregation`]) produces
    /// DataFusion's results with thread-local tables of capacity `capacity`
    /// on `threads` threads, and that the rows of threads whose groups hardly
    /// repeat may bypass the tables if `bypass` (see `AggregateSpec::bypass`):
    /// then also when they do not.
    async fn assert_bypassed_parity(
        ctx: &SessionContext,
        sql: &str,
        capacity: TableCapacity,
        threads: usize,
        bypass: bool,
    ) -> Result<()> {
        let plan = async || -> Result<Arc<dyn ExecutionPlan>> {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            Ok(aggregation(&plan).expect("an aggregation"))
        };
        let expected_plan = plan().await?;
        let expected =
            datafusion_physical_plan::collect(Arc::clone(&expected_plan), ctx.task_ctx())
                .await?;
        let aggregate = || async {
            Ok::<_, datafusion_common::DataFusionError>(
                PhysicalHashAggregate::try_lower(&plan().await?, lower)?
                    .expect("a lowered aggregation")
                    .with_table_capacity(capacity),
            )
        };
        assert_eq!(aggregate().await?.spec().bypass, bypass, "{sql}");
        let mut aggregates = vec![aggregate().await?];
        if bypass {
            aggregates.push(aggregate().await?.without_bypass());
        }
        for aggregate in aggregates {
            let bypassed = aggregate.spec().bypass;
            let actual =
                Executor::collect(Arc::new(aggregate), ctx.task_ctx(), threads).await?;
            let schema = expected_plan.schema();
            assert_eq!(
                canonical_batch(&schema, &actual, false)?,
                canonical_batch(&schema, &expected, false)?,
                "{sql} on {threads} threads with tables of {capacity:?} (bypass: \
                 {bypassed})"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregates_ordered_by_values_without_a_total_order_are_left_to_datafusion()
    -> Result<()> {
        use arrow::array::StructArray;
        use arrow::datatypes::Fields;
        // DataFusion sorts the rows of `FIRST_VALUE(v ORDER BY v)` by Arrow's
        // order, which puts the NULLs of a field last, and takes the first; the
        // native accumulator (see `without_input_order`) compared their
        // `ScalarValue`s, which do not order structs totally (see
        // `compares_totally`): of one partition of 8,192 rows of a single
        // group, `{a: NULL}` in the first 2,048 rows and `{a: 1}` after,
        // DataFusion's first value was `{a: 1}`, the native one `{a: NULL}`,
        // also on one thread. Aggregates ordered by such values are left to
        // DataFusion; those ordered by other values stay native.
        let fields = Fields::from(vec![Field::new("a", DataType::Int64, true)]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("i", DataType::Int64, false),
            Field::new("v", DataType::Struct(fields.clone()), true),
        ]));
        let a = Int64Array::from_iter((0..8_192).map(|i| (i >= 2_048).then_some(1)));
        let v = StructArray::new(fields, vec![Arc::new(a) as ArrayRef], None);
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1; 8_192])),
                Arc::new(Int64Array::from_iter_values((0..8_192).rev())),
                Arc::new(v),
            ],
        )?;
        let setup = mem_table_setup(Arc::clone(&schema), vec![vec![batch]]);
        for target_partitions in [1, 4, 16] {
            let config = test_session_config()
                .with_target_partitions(target_partitions)
                .with_repartition_aggregations(false);
            for sql in [
                "SELECT k, FIRST_VALUE(v ORDER BY v) FROM t GROUP BY k",
                "SELECT k, LAST_VALUE(v ORDER BY v DESC) FROM t GROUP BY k",
                "SELECT FIRST_VALUE(v ORDER BY v), LAST_VALUE(v ORDER BY v) FROM t",
                "SELECT k, NTH_VALUE(v, 2 ORDER BY v) FROM t GROUP BY k",
                "SELECT k, ARRAY_AGG(v ORDER BY v) FROM t GROUP BY k",
            ] {
                assert_left_to_datafusion(sql, config.clone(), &setup).await?;
            }
            // Ordered by a column without ties.
            let sql = "SELECT k, FIRST_VALUE(v ORDER BY i), MAX(i) FROM t GROUP BY k";
            assert!(
                planned_aggregations(sql, &config, &setup).await? > 0,
                "{sql}"
            );
            assert_aggregate_parity_with_setup(sql, config.clone(), &setup).await;
        }
        Ok(())
    }

    /// A `MemTable` `t` of the partitions `partitions` (of the schema of their
    /// first batch, or of `schema`).
    fn mem_table_setup(
        schema: SchemaRef,
        partitions: Vec<Vec<RecordBatch>>,
    ) -> impl AsyncFn(&SessionContext) -> Result<()> {
        async move |ctx: &SessionContext| -> Result<()> {
            let table = datafusion::datasource::MemTable::try_new(
                Arc::clone(&schema),
                partitions.clone(),
            )?;
            ctx.register_table("t", Arc::new(table))?;
            Ok(())
        }
    }

    /// Assert that the aggregation of `sql` (see [`aggregation`]) in a session
    /// of configuration `config`, set up by `setup`, is left to DataFusion, by
    /// [`PhysicalHashAggregate::try_lower`] and by the planner, and that the
    /// pipeline engine returns DataFusion's results.
    async fn assert_left_to_datafusion<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Result<()>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        use super::super::test_support::assert_lowered_parity_with_setup;
        assert_not_lowered(sql, &config, &setup).await?;
        assert_lowered_parity_with_setup(sql, config, &setup).await;
        Ok(())
    }

    /// Assert that neither [`PhysicalHashAggregate::try_lower`] nor the
    /// planner lowers the aggregation of `sql`, in a session of configuration
    /// `config` set up by `setup`.
    async fn assert_not_lowered<F>(
        sql: &str,
        config: &SessionConfig,
        setup: F,
    ) -> Result<()>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = datafusion_context(config.clone());
        setup(&ctx).await?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let plan = aggregation(&plan).expect("an aggregation");
        assert!(
            PhysicalHashAggregate::try_lower(&plan, lower)?.is_none(),
            "{sql}"
        );
        assert_eq!(planned_aggregations(sql, config, &setup).await?, 0, "{sql}");
        Ok(())
    }

    /// The number of native aggregations of the plan of `sql` that the planner
    /// lowers in a session of configuration `config` set up by `setup`.
    async fn planned_aggregations<F>(
        sql: &str,
        config: &SessionConfig,
        setup: F,
    ) -> Result<usize>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = crate::test_utils::pipeline_context(config.clone());
        setup(&ctx).await?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let Some(root) = plan.downcast_ref::<crate::bridge::PipelineRootExec>() else {
            return internal_err!("The pipeline engine did not plan {sql}");
        };
        Ok(crate::operators::union::test_support::count::<
            PhysicalHashAggregate,
        >(root.operator()))
    }

    /// Table `hc`: 34,816 distinct keys `k`, values `v`, strings `s` (views)
    /// and booleans `b` (with nulls), in a single partition of batches of
    /// 2,048 rows: a thread that aggregates them fills its table (of 32,768
    /// groups) with the first 16 batches, and the rows of the last one then
    /// bypass it, if they may.
    fn hc_batches() -> Result<Vec<RecordBatch>> {
        const ROWS: i64 = 17 * DEFAULT_VECTOR_SIZE as i64;
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
            Field::new("s", DataType::Utf8View, true),
            Field::new("b", DataType::Boolean, true),
        ]));
        (0..ROWS)
            .step_by(DEFAULT_VECTOR_SIZE)
            .map(|start| {
                let rows = start..start + DEFAULT_VECTOR_SIZE as i64;
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from_iter_values(
                        rows.clone().map(|i| (i * 7_919) % 1_000_003),
                    )),
                    Arc::new(Int64Array::from_iter(
                        rows.clone()
                            .map(|i| (i % 13 != 4).then_some(i % 1_000 - 500)),
                    )),
                    Arc::new(StringViewArray::from_iter(rows.clone().map(|i| {
                        (i % 17 != 3).then(|| format!("a string that is not inlined {i}"))
                    }))),
                    Arc::new(BooleanArray::from_iter(
                        rows.map(|i| (i % 11 != 5).then_some(i % 3 == 0)),
                    )),
                ];
                Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
            })
            .collect()
    }

    /// Whether the rows of `batches` (of a table of the query) bypass the
    /// table of a thread of the native aggregation that consumes them, the
    /// deepest of the plan of `sql` that the planner lowers in the session of
    /// `ctx`, when its sink consumes them on a thread of an execution in that
    /// session, as the executor runs it (see
    /// `PhysicalHashAggregate::get_local_sink_state`).
    async fn planned_aggregation_bypasses(
        ctx: &SessionContext,
        sql: &str,
        batches: &[RecordBatch],
    ) -> Result<bool> {
        use crate::operator::{ExecutionContext, PhysicalOperator};
        fn deepest(
            operator: &Arc<dyn PhysicalOperator>,
        ) -> Option<Arc<dyn PhysicalOperator>> {
            operator.children().iter().find_map(deepest).or_else(|| {
                operator
                    .is::<PhysicalHashAggregate>()
                    .then(|| Arc::clone(operator))
            })
        }
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let Some(root) = plan.downcast_ref::<crate::bridge::PipelineRootExec>() else {
            return internal_err!("The pipeline engine did not plan {sql}");
        };
        let Some(aggregate) = deepest(root.operator()) else {
            return internal_err!("No native aggregation in the plan of {sql}");
        };
        let schema = aggregate.children()[0].schema();
        let execution = ExecutionContext::new(ctx.task_ctx(), 0, 1);
        let global = aggregate.get_global_sink_state(&execution)?;
        let mut local = aggregate.get_local_sink_state(&execution, global.as_ref())?;
        let mut bypassed = false;
        for batch in batches {
            let columns = schema
                .fields()
                .iter()
                .map(|field| {
                    Arc::clone(batch.column_by_name(field.name()).expect("a column"))
                })
                .collect();
            let chunk = DataChunk::from_record_batch(RecordBatch::try_new(
                Arc::clone(&schema),
                columns,
            )?);
            aggregate.sink(&execution, &chunk, global.as_ref(), local.as_mut())?;
            let Some(state) = local.downcast_ref::<RadixLocalState>() else {
                return internal_err!("The aggregation of {sql} has no radix sink state");
            };
            bypassed |= state.bypass.is_some();
        }
        Ok(bypassed)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rows_of_planned_aggregations_bypass_the_tables() -> Result<()> {
        use super::super::test_support::{lookalike_count, lookalike_sum};
        // The native aggregations that the planner lowers, their sinks consuming
        // 34,816 distinct keys on a thread of an execution of the session, as
        // the executor runs them: once the table of the thread is full, its
        // rows bypass it, whatever the aggregates (built-in or user-defined,
        // `DISTINCT` and ordered ones included), in every session, also in
        // the sessions that trust DataFusion's built-in aggregate functions
        // (which changes nothing). Nothing forces `AggregateSpec::bypass` here.
        // (On a single target partition, on which the planner lowers all of
        // these aggregations: on several, it leaves those whose partial states
        // are lists or binaries to DataFusion, see the module documentation of
        // `planner`.)
        let batches = hc_batches()?;
        let queries = [
            // The query, and whether its rows bypass the tables.
            (
                "SELECT k, COUNT(*), SUM(v), MIN(s), MAX(v), AVG(v) FROM hc GROUP BY k",
                true,
            ),
            (
                "SELECT k, BOOL_AND(b), BOOL_OR(b), BIT_AND(v), BIT_OR(v), BIT_XOR(v), \
                 APPROX_DISTINCT(s), MEDIAN(v), ARRAY_AGG(v), FIRST_VALUE(v), \
                 LAST_VALUE(s), COUNT(v) FILTER (WHERE b) FROM hc GROUP BY k",
                true,
            ),
            ("SELECT k, s FROM hc GROUP BY k, s", true),
            ("SELECT DISTINCT k, v FROM hc", true),
            // `DISTINCT` and ordered aggregates, and other built-in functions.
            (
                "SELECT k, COUNT(DISTINCT v), COUNT(DISTINCT s) FROM hc GROUP BY k",
                true,
            ),
            (
                "SELECT k, SUM(DISTINCT v), COUNT(*) FROM hc GROUP BY k",
                true,
            ),
            (
                "SELECT k, ARRAY_AGG(v ORDER BY s), COUNT(*) FROM hc GROUP BY k",
                true,
            ),
            (
                "SELECT k, LAST_VALUE(v ORDER BY s DESC), MAX(v) FROM hc GROUP BY k",
                true,
            ),
            ("SELECT k, STDDEV(v), COUNT(*) FROM hc GROUP BY k", true),
            // An aggregate that requires its rows in an order, which the thread
            // sorts once it saw all of them.
            (
                "SELECT k, NTH_VALUE(v, 2 ORDER BY s), COUNT(*) FROM hc GROUP BY k",
                false,
            ),
        ];
        for trusted in [false, true] {
            let mut config = test_session_config().with_target_partitions(1);
            if trusted {
                config = trusting(config);
            }
            let setup = async |ctx: &SessionContext| -> Result<()> {
                let table = datafusion::datasource::MemTable::try_new(
                    batches[0].schema(),
                    vec![batches.clone()],
                )?;
                ctx.register_table("hc", Arc::new(table))?;
                Ok(())
            };
            let ctx = crate::test_utils::pipeline_context(config.clone());
            setup(&ctx).await?;
            for (sql, bypass) in queries {
                assert_eq!(
                    planned_aggregation_bypasses(&ctx, sql, &batches).await?,
                    bypass,
                    "{sql} (trusted: {trusted})"
                );
                // The results of the rows that bypass the table of the thread
                // (those of the other paths are checked by the other tests).
                if bypass {
                    assert_byte_eq_with_datafusion_with_config(
                        sql,
                        config.clone(),
                        &setup,
                    )
                    .await;
                }
            }
            // User-defined functions that look like DataFusion's `count` and
            // `sum`, and replace them (see `lookalike_count`): their rows bypass
            // the tables too.
            let setup_lookalikes = async |ctx: &SessionContext| -> Result<()> {
                setup(ctx).await?;
                ctx.register_udaf(lookalike_count());
                ctx.register_udaf(lookalike_sum());
                Ok(())
            };
            let ctx = crate::test_utils::pipeline_context(config.clone());
            setup_lookalikes(&ctx).await?;
            for sql in [
                "SELECT k, count(v), sum(v) FROM hc GROUP BY k",
                "SELECT k, count(*), MAX(v) FROM hc GROUP BY k",
            ] {
                assert!(planned_aggregation_bypasses(&ctx, sql, &batches).await?);
                assert_byte_eq_with_datafusion_with_config(
                    sql,
                    config.clone(),
                    &setup_lookalikes,
                )
                .await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn arguments_of_bypassed_rows_fail() -> Result<()> {
        use super::super::test_support::assert_aggregate_fails_with_setup;
        // 40,000 distinct keys (the rows of a thread bypass its table once it is
        // full), and a last row on which an argument of an aggregate fails:
        // DataFusion and the native aggregation fail on it, on 1, 4 and 16
        // threads, with the default and with tiny tables, through the planner
        // and the lowering of the tests. The bypassed rows evaluate the
        // arguments of the aggregates when they are consumed, like the tables.
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("d", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batches = (0..40_000)
            .step_by(DEFAULT_VECTOR_SIZE)
            .map(|start| {
                let rows = start..(start + DEFAULT_VECTOR_SIZE as i64).min(40_000);
                let last = |i: i64| i == 39_999;
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from_iter_values(
                        rows.clone().map(|i| (i * 7_919) % 40_009),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        rows.clone().map(|i| if last(i) { 1_000 } else { i % 100 }),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        rows.clone().map(|i| i64::from(!last(i))),
                    )),
                    Arc::new(StringArray::from_iter_values(rows.map(|i| {
                        if last(i) {
                            "not an integer".to_string()
                        } else {
                            format!("{i}")
                        }
                    }))),
                ];
                Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let setup = async |ctx: &SessionContext| -> Result<()> {
            let table = datafusion::datasource::MemTable::try_new(
                Arc::clone(&schema),
                vec![batches.clone()],
            )?;
            ctx.register_table("t", Arc::new(table))?;
            Ok(())
        };
        // The queries, their errors, and the numbers of target partitions on
        // which the planner lowers their aggregations: all of them, whatever
        // their arguments may fail with (a cast of a string, a division, which
        // may fail with a division by zero or an overflow, or a cast of an
        // integer into a decimal, which may overflow), see
        // `docs/compatibility.md`.
        let queries: [(&str, &str, &[usize]); 3] = [
            (
                "SELECT k, COUNT(*), SUM(CAST(s AS BIGINT)) FROM t GROUP BY k",
                "Cannot cast string 'not an integer' to value of Int64 type",
                &[1, 4, 16],
            ),
            (
                "SELECT k, MAX(v), SUM(v / d) FROM t GROUP BY k",
                "Divide by zero error",
                &[1, 4, 16],
            ),
            (
                "SELECT k, MAX(v), SUM(CAST(v AS DECIMAL(3, 0))) FROM t GROUP BY k",
                "1000",
                &[1, 4, 16],
            ),
        ];
        for target_partitions in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for (sql, message, planned) in queries {
                let ctx = datafusion_context(config.clone());
                setup(&ctx).await?;
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                let aggregate = PhysicalHashAggregate::try_lower(
                    &aggregation(&plan).expect("an aggregation"),
                    lower,
                )?
                .expect("a lowered aggregation");
                assert!(aggregate.spec().bypass, "{sql}");
                assert_eq!(
                    planned_aggregations(sql, &config, &setup).await?,
                    usize::from(planned.contains(&target_partitions)),
                    "{sql} on {target_partitions} target partitions"
                );
                assert_aggregate_fails_with_setup(
                    sql,
                    config.clone(),
                    &setup,
                    &[message],
                )
                .await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bypassed_rows_are_aggregated_like_datafusion() -> Result<()> {
        assert_bypassed_rows_are_aggregated_like_datafusion(&[1, 4], &[1, 4]).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bypassed_rows_are_aggregated_like_datafusion_on_16_threads() -> Result<()> {
        assert_bypassed_rows_are_aggregated_like_datafusion(&[16], &[16]).await
    }

    /// Assert that the queries over table `u` (see [`register_u`]) produce
    /// DataFusion's results, with `target_partitions` of `partitions`, on
    /// `threads` native threads (and through the planner), with tables small
    /// enough for their rows to bypass them.
    async fn assert_bypassed_rows_are_aggregated_like_datafusion(
        partitions: &[usize],
        threads: &[usize],
    ) -> Result<()> {
        // The queries, whose rows bypass the tables whatever their aggregates
        // (`DISTINCT` and ordered ones included).
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
            "SELECT k % 97 AS m, MEDIAN(v), APPROX_DISTINCT(s), BOOL_AND(b), \
             BIT_AND(v), BIT_OR(v), SUM(d), AVG(v) FROM u GROUP BY k % 97",
            "SELECT k % 50 AS m, ARRAY_AGG(k ORDER BY k), FIRST_VALUE(sv ORDER BY v, k) \
             FROM u GROUP BY k % 50",
            "SELECT sv, COUNT(*), MAX(s), BIT_XOR(v) FROM u GROUP BY sv",
            // Without aggregates.
            "SELECT k, s, sv FROM u GROUP BY k, s, sv",
            "SELECT DISTINCT d, b, v FROM u",
            "SELECT k % 97 AS m FROM u GROUP BY k % 97",
            // Counts of the arguments that the bypassed rows hold: filtered, of
            // several arguments, and distinct ones.
            "SELECT k, COUNT(s), COUNT(sv) FILTER (WHERE v > 0), COUNT(d, s), \
             COUNT(b) FILTER (WHERE false), COUNT(v) FILTER (WHERE true), \
             COUNT(DISTINCT s), COUNT(NULL) FROM u GROUP BY k",
        ];
        let tables = [
            SMALL_TABLES,
            TableCapacity {
                initial: 64,
                maximum: 256,
            },
        ];
        for &target_partitions in partitions {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = datafusion_context(config.clone());
            register_u(&ctx)?;
            for sql in queries {
                for capacity in tables {
                    for &threads in threads {
                        assert_bypassed_parity(&ctx, sql, capacity, threads, true)
                            .await?;
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

    /// A function `count` of a type named `Count` too, which replaces
    /// DataFusion's `COUNT` and counts the `true` values of its argument.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct Count {
        signature: datafusion_expr::Signature,
    }

    impl datafusion_expr::AggregateUDFImpl for Count {
        fn name(&self) -> &str {
            "count"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn return_type(&self, _: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn accumulator(
            &self,
            _: datafusion_expr::function::AccumulatorArgs,
        ) -> Result<Box<dyn datafusion_expr::Accumulator>> {
            Ok(Box::new(TrueCount(0)))
        }
    }

    /// The accumulator of [`Count`].
    #[derive(Debug)]
    struct TrueCount(i64);

    impl datafusion_expr::Accumulator for TrueCount {
        fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
            self.0 += values[0].as_boolean().true_count() as i64;
            Ok(())
        }

        fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
            let counts = states[0].as_primitive::<Int64Type>();
            self.0 += counts.iter().flatten().sum::<i64>();
            Ok(())
        }

        fn state(&mut self) -> Result<Vec<datafusion_common::ScalarValue>> {
            Ok(vec![datafusion_common::ScalarValue::Int64(Some(self.0))])
        }

        fn evaluate(&mut self) -> Result<datafusion_common::ScalarValue> {
            Ok(datafusion_common::ScalarValue::Int64(Some(self.0)))
        }

        fn size(&self) -> usize {
            size_of::<Self>()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn user_defined_counts_are_aggregated_like_datafusion() -> Result<()> {
        // `count` of [`Count`] over 40,000 distinct keys of a single
        // partition, in batches of 2,048 rows, whose arguments are all `true`:
        // the rows after the first 32,768 fill the table of the thread, and
        // bypass it, with the values of their arguments, which the
        // accumulators of the final aggregation read.
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("b", DataType::Boolean, false),
        ]));
        let batches = (0..40_000)
            .step_by(2_048)
            .map(|start| {
                let keys = start..(start + 2_048).min(40_000);
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from_iter_values(keys.clone())),
                        Arc::new(BooleanArray::from(vec![
                            true;
                            (keys.end - keys.start)
                                as usize
                        ])),
                    ],
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let setup = async |ctx: &SessionContext| -> Result<()> {
            let table = datafusion::datasource::MemTable::try_new(
                Arc::clone(&schema),
                vec![batches.clone()],
            )?;
            ctx.register_table("trues", Arc::new(table))?;
            let signature = datafusion_expr::Signature::exact(
                vec![DataType::Boolean],
                datafusion_expr::Volatility::Immutable,
            );
            ctx.register_udaf(datafusion_expr::AggregateUDF::from(Count { signature }));
            Ok(())
        };
        let sql = "SELECT k, count(b) AS c FROM trues GROUP BY k";
        let config = test_session_config().with_target_partitions(4);
        let ctx = datafusion_context(config.clone());
        setup(&ctx).await?;
        // DataFusion counts 1 per group.
        let batches = ctx.sql(sql).await?.collect().await?;
        let counts = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(1)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(counts.len(), 40_000);
        assert!(counts.iter().all(|&count| count == 1));
        // So does the native aggregation, whose rows bypass the tables or not,
        // on 1, 4 and 16 threads.
        for target_partitions in [4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = datafusion_context(config.clone());
            setup(&ctx).await?;
            for threads in [1, 4, 16] {
                assert_bypassed_parity(
                    &ctx,
                    sql,
                    TableCapacity::default(),
                    threads,
                    true,
                )
                .await?;
            }
            assert_aggregate_parity_with_setup(sql, config, &setup).await;
        }
        Ok(())
    }

    /// A function `non_negative_sum`, whose accumulator rejects negative
    /// values.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct NonNegativeSum {
        signature: datafusion_expr::Signature,
    }

    impl datafusion_expr::AggregateUDFImpl for NonNegativeSum {
        fn name(&self) -> &str {
            "non_negative_sum"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn return_type(&self, _: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn accumulator(
            &self,
            _: datafusion_expr::function::AccumulatorArgs,
        ) -> Result<Box<dyn datafusion_expr::Accumulator>> {
            Ok(Box::new(NonNegativeSumAccumulator(0)))
        }
    }

    /// The accumulator of [`NonNegativeSum`].
    #[derive(Debug)]
    struct NonNegativeSumAccumulator(i64);

    impl datafusion_expr::Accumulator for NonNegativeSumAccumulator {
        fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
            for value in values[0].as_primitive::<Int64Type>().iter().flatten() {
                if value < 0 {
                    return Err(datafusion_common::DataFusionError::Execution(format!(
                        "non_negative_sum rejects the negative value {value}"
                    )));
                }
                self.0 += value;
            }
            Ok(())
        }

        fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
            let sums = states[0].as_primitive::<Int64Type>();
            self.0 += sums.iter().flatten().sum::<i64>();
            Ok(())
        }

        fn state(&mut self) -> Result<Vec<datafusion_common::ScalarValue>> {
            Ok(vec![datafusion_common::ScalarValue::Int64(Some(self.0))])
        }

        fn evaluate(&mut self) -> Result<datafusion_common::ScalarValue> {
            Ok(datafusion_common::ScalarValue::Int64(Some(self.0)))
        }

        fn size(&self) -> usize {
            size_of::<Self>()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn accumulators_fail_on_the_rows_after_full_tables() -> Result<()> {
        use super::super::test_support::assert_aggregate_fails_with_setup;
        // 32,768 distinct keys, which fill the table of a thread, then a
        // negative value, which `non_negative_sum` rejects, then a string that
        // is not an integer, whose cast fails: DataFusion fails on the negative
        // value, the native aggregation (on a single thread, of a single stage,
        // of a partition, and of two, of two partitions, one of them empty) on
        // either. Its rows bypass the table, whose accumulators aggregate the
        // negative value after the cast of the next row failed (which of the
        // errors a query raises is the engine's own).
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batch = |keys: Vec<i64>, value: i64, string: &str| {
            let rows = keys.len();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(Int64Array::from(vec![value; rows])),
                    Arc::new(StringArray::from(vec![string; rows])),
                ],
            )
        };
        let batches = vec![
            batch((0..32_768).collect(), 1, "1")?,
            batch(vec![50_000], -5, "2")?,
            batch(vec![50_001], 3, "not an integer")?,
        ];
        let sql =
            "SELECT k, non_negative_sum(v), SUM(CAST(s AS BIGINT)) FROM t GROUP BY k";
        for partitions in [vec![batches.clone()], vec![batches.clone(), vec![]]] {
            let setup = async |ctx: &SessionContext| -> Result<()> {
                let table = datafusion::datasource::MemTable::try_new(
                    Arc::clone(&schema),
                    partitions.clone(),
                )?;
                ctx.register_table("t", Arc::new(table))?;
                let signature = datafusion_expr::Signature::exact(
                    vec![DataType::Int64],
                    datafusion_expr::Volatility::Immutable,
                );
                let udaf = NonNegativeSum { signature };
                ctx.register_udaf(datafusion_expr::AggregateUDF::from(udaf));
                Ok(())
            };
            let config = test_session_config().with_target_partitions(1);
            let ctx = datafusion_context(config.clone());
            setup(&ctx).await?;
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let aggregate = PhysicalHashAggregate::try_lower(
                &aggregation(&plan).expect("an aggregation"),
                lower,
            )?
            .expect("a lowered aggregation");
            assert!(aggregate.spec().bypass, "{} partitions", partitions.len());
            let messages = [
                "non_negative_sum rejects the negative value -5",
                "Cannot cast string 'not an integer'",
            ];
            assert_aggregate_fails_with_setup(sql, config, setup, &messages).await;
        }
        Ok(())
    }

    /// A function `running_sum`, whose accumulator rejects the rows that make
    /// its running total negative.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct RunningSum {
        signature: datafusion_expr::Signature,
    }

    impl datafusion_expr::AggregateUDFImpl for RunningSum {
        fn name(&self) -> &str {
            "running_sum"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn return_type(&self, _: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn accumulator(
            &self,
            _: datafusion_expr::function::AccumulatorArgs,
        ) -> Result<Box<dyn datafusion_expr::Accumulator>> {
            Ok(Box::new(RunningSumAccumulator(0)))
        }
    }

    /// The accumulator of [`RunningSum`].
    #[derive(Debug)]
    struct RunningSumAccumulator(i64);

    impl datafusion_expr::Accumulator for RunningSumAccumulator {
        fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
            for value in values[0].as_primitive::<Int64Type>().iter().flatten() {
                self.0 += value;
                if self.0 < 0 {
                    return Err(datafusion_common::DataFusionError::Execution(format!(
                        "negative running total: {}",
                        self.0
                    )));
                }
            }
            Ok(())
        }

        fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
            let sums = states[0].as_primitive::<Int64Type>();
            self.0 += sums.iter().flatten().sum::<i64>();
            Ok(())
        }

        fn state(&mut self) -> Result<Vec<datafusion_common::ScalarValue>> {
            Ok(vec![datafusion_common::ScalarValue::Int64(Some(self.0))])
        }

        fn evaluate(&mut self) -> Result<datafusion_common::ScalarValue> {
            Ok(datafusion_common::ScalarValue::Int64(Some(self.0)))
        }

        fn size(&self) -> usize {
            size_of::<Self>()
        }
    }

    /// A function `sum` of a type named `Sum` too, which replaces DataFusion's
    /// `SUM`: its accumulator rejects the rows that make its running total
    /// negative, like the one of [`RunningSum`].
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct Sum {
        signature: datafusion_expr::Signature,
    }

    impl datafusion_expr::AggregateUDFImpl for Sum {
        fn name(&self) -> &str {
            "sum"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn return_type(&self, _: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn accumulator(
            &self,
            _: datafusion_expr::function::AccumulatorArgs,
        ) -> Result<Box<dyn datafusion_expr::Accumulator>> {
            Ok(Box::new(RunningSumAccumulator(0)))
        }
    }

    /// Register table `t` of `partitions`, and the function of a running total
    /// named `function`: [`RunningSum`] (`running_sum`) or [`Sum`] (`sum`, which
    /// replaces DataFusion's).
    fn register_running_totals(
        ctx: &SessionContext,
        partitions: &[Vec<RecordBatch>],
        function: &str,
    ) -> Result<()> {
        let schema = partitions
            .iter()
            .flatten()
            .next()
            .map(RecordBatch::schema)
            .expect("rows");
        let table =
            datafusion::datasource::MemTable::try_new(schema, partitions.to_vec())?;
        ctx.register_table("t", Arc::new(table))?;
        let signature = datafusion_expr::Signature::exact(
            vec![DataType::Int64],
            datafusion_expr::Volatility::Immutable,
        );
        ctx.register_udaf(match function {
            "sum" => datafusion_expr::AggregateUDF::from(Sum { signature }),
            _ => datafusion_expr::AggregateUDF::from(RunningSum { signature }),
        });
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn running_totals_of_user_defined_aggregates_are_aggregated_like_datafusion()
    -> Result<()> {
        use super::super::test_support::assert_aggregate_parity_with_setup;
        // 32,768 distinct keys of value 1, which fill the table of a thread,
        // then the rows `(50000, 1)` and `(50000, -1)` of a group, in a batch or
        // in two: the running total of the group gets to 1, then 0, as in
        // DataFusion, and never to -1 (which it rejects), also if the
        // user-defined function is named `sum`, of a type named `Sum`. The rows
        // of a thread that bypass its table (those after the first 32,768 on a
        // thread) are aggregated in their order, after the partial states of
        // their groups (whether the running total of a group of the rows of
        // several threads gets negative depends on the threads, as in
        // DataFusion, on several partitions).
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = |keys: Vec<i64>, values: Vec<i64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(Int64Array::from(values)),
                ],
            )
        };
        let distinct = batch((0..32_768).collect(), vec![1; 32_768])?;
        for (last, thread_counts) in [
            (
                vec![batch(vec![50_000, 50_000], vec![1, -1])?],
                &[1, 4, 16][..],
            ),
            // On several threads, DataFusion's partial aggregations of the
            // partitions of its round-robin repartitioning aggregate the two
            // batches apart (and reject -1), the native threads the chunks of
            // their morsels: on one thread only.
            (
                vec![
                    batch(vec![50_000], vec![1])?,
                    batch(vec![50_000], vec![-1])?,
                ],
                &[1],
            ),
        ] {
            let mut rows = vec![distinct.clone()];
            rows.extend(last);
            // Two partitions (the second empty): the aggregation has two
            // stages, also on one target partition.
            let partitions = vec![rows, vec![]];
            for function in ["running_sum", "sum"] {
                let setup = async |ctx: &SessionContext| {
                    register_running_totals(ctx, &partitions, function)
                };
                let sql = format!("SELECT k, {function}(v) AS s FROM t GROUP BY k");
                for &threads in thread_counts {
                    let config = test_session_config().with_target_partitions(threads);
                    let ctx = datafusion_context(config.clone());
                    setup(&ctx).await?;
                    let groups = ctx.sql(&sql).await?.collect().await?;
                    assert_eq!(
                        groups.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        32_769
                    );
                    assert_bypassed_parity(
                        &ctx,
                        &sql,
                        TableCapacity::default(),
                        threads,
                        true,
                    )
                    .await?;
                    assert_aggregate_parity_with_setup(&sql, config, &setup).await;
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn user_defined_sums_fail_on_the_rows_after_full_tables() -> Result<()> {
        use super::super::test_support::assert_aggregate_fails_with_setup;
        // 32,768 distinct keys, which fill the table of a thread, then a row
        // whose value makes the running total of its group negative, which
        // [`Sum`] (a function `sum` of a type named `Sum`, which replaces
        // DataFusion's) rejects, then a string that is not an integer, whose
        // cast fails: DataFusion fails on the negative running total, the
        // native aggregation (of one stage and of two) on either. Its rows
        // bypass the table: the negative value is aggregated later, by the final
        // aggregation, after the cast of the next row failed.
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batch = |keys: Vec<i64>, value: i64, string: &str| {
            let rows = keys.len();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(Int64Array::from(vec![value; rows])),
                    Arc::new(StringArray::from(vec![string; rows])),
                ],
            )
        };
        let batches = vec![
            batch((0..32_768).collect(), 1, "1")?,
            batch(vec![50_000], -5, "2")?,
            batch(vec![50_001], 3, "not an integer")?,
        ];
        let sql = "SELECT k, sum(v), MAX(CAST(s AS BIGINT)) FROM t GROUP BY k";
        let messages = ["negative running total: -5", "not an integer"];
        for partitions in [vec![batches.clone()], vec![batches.clone(), vec![]]] {
            let setup = async |ctx: &SessionContext| {
                register_running_totals(ctx, &partitions, "sum")
            };
            let config = test_session_config().with_target_partitions(1);
            let ctx = datafusion_context(config.clone());
            setup(&ctx).await?;
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let aggregate = PhysicalHashAggregate::try_lower(
                &aggregation(&plan).expect("an aggregation"),
                lower,
            )?
            .expect("a lowered aggregation");
            let spec = aggregate.spec();
            assert!(spec.bypass);
            assert_aggregate_fails_with_setup(sql, config.clone(), &setup, &messages)
                .await;
            // A thread whose rows bypass its table fails on the cast first.
            let local = RadixHashAggregate::local_sink_state(
                &spec,
                TableCapacity::default(),
                true,
            )?;
            let mut sink = (spec, local, RadixHashAggregate::global_sink_state()?);
            let (spec, local, global) = &mut sink;
            let error = batches
                .iter()
                .flat_map(|batch| {
                    (0..batch.num_rows()).step_by(DEFAULT_VECTOR_SIZE).map(
                        move |offset| {
                            batch.slice(
                                offset,
                                DEFAULT_VECTOR_SIZE.min(batch.num_rows() - offset),
                            )
                        },
                    )
                })
                .try_for_each(|chunk| {
                    RadixHashAggregate::sink(spec, chunk, global.as_ref(), local.as_mut())
                        .map(|_| ())
                })
                .expect_err("a cast error");
            assert!(error.to_string().contains("not an integer"), "{error}");
        }
        Ok(())
    }

    /// Register table `long`, of a single partition: keys `k`, binaries `b`
    /// and strings `s` of 1 to 40,000 bytes, and values `v`, in batches of
    /// distinct keys that fill a thread-local table of 16 groups first, and of
    /// binaries and strings of more than 32 KiB together (or on their own).
    fn register_long(ctx: &SessionContext) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("b", DataType::Binary, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Int64, true),
        ]));
        let batch = |keys: std::ops::Range<i64>, len: usize| -> Result<RecordBatch> {
            let valid = |key: i64| key % 7 != 3;
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter(
                    keys.clone().map(|key| (key % 11 != 5).then_some(key)),
                )),
                Arc::new(arrow::array::BinaryArray::from_iter(
                    keys.clone()
                        .map(|key| valid(key).then(|| vec![(key % 251) as u8; len])),
                )),
                Arc::new(StringArray::from_iter(keys.clone().map(|key| {
                    valid(key + 1).then(|| format!("{key:05}{}", "s".repeat(len)))
                }))),
                Arc::new(Int64Array::from_iter_values(keys.map(|key| key * 3 - 100))),
            ];
            Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
        };
        let batches = vec![
            batch(0..40, 1)?,
            batch(100..110, 1_000)?,
            batch(200..202, 12_000)?,
            batch(300..301, 40_000)?,
            batch(400..450, 10)?,
            batch(0..12, 3_000)?,
            batch(500..520, 2_000)?,
            batch(600..603, 20_000)?,
        ];
        let table = datafusion::datasource::MemTable::try_new(schema, vec![batches])?;
        ctx.register_table("long", Arc::new(table))?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bypassed_rows_of_long_values_are_aggregated_like_datafusion() -> Result<()> {
        assert_bypassed_rows_of_long_values_are_aggregated_like_datafusion(
            &[1, 4],
            &[1, 4],
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bypassed_rows_of_long_values_are_aggregated_like_datafusion_on_16_threads()
    -> Result<()> {
        assert_bypassed_rows_of_long_values_are_aggregated_like_datafusion(&[16], &[16])
            .await
    }

    /// Assert that the queries over table `long` (see [`register_long`])
    /// produce DataFusion's results, like
    /// [`assert_bypassed_rows_are_aggregated_like_datafusion`].
    async fn assert_bypassed_rows_of_long_values_are_aggregated_like_datafusion(
        partitions: &[usize],
        threads: &[usize],
    ) -> Result<()> {
        // The queries, whose rows (binaries and strings of up to 40 KB
        // included) bypass the tables.
        let queries = [
            "SELECT k, COUNT(b), MAX(s), SUM(v) FROM long GROUP BY k",
            "SELECT k, MIN(b), MAX(b), COUNT(*) FROM long GROUP BY k",
            "SELECT s, COUNT(*), MIN(k), COUNT(b) FROM long GROUP BY s",
            "SELECT k, b FROM long GROUP BY k, b",
            "SELECT DISTINCT s, v FROM long",
        ];
        let tables = [
            SMALL_TABLES,
            TableCapacity {
                initial: 64,
                maximum: 256,
            },
        ];
        for &target_partitions in partitions {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = datafusion_context(config.clone());
            register_long(&ctx)?;
            for sql in queries {
                for capacity in tables {
                    for &threads in threads {
                        assert_bypassed_parity(&ctx, sql, capacity, threads, true)
                            .await?;
                    }
                }
                let setup = async |ctx: &SessionContext| register_long(ctx);
                assert_aggregate_parity_with_setup(sql, config.clone(), setup).await;
            }
        }
        Ok(())
    }

    /// Register table `wide`, of a single partition: 32,768 distinct keys
    /// `k` with short binaries `b`, which fill the thread-local table of a
    /// thread (whose rows then bypass it), followed by a chunk of a binary of
    /// 50,000,000 bytes and one of 2,147,000,000 bytes: more than a `Binary`
    /// array can hold together.
    fn huge_binaries() -> Result<(SchemaRef, Vec<Vec<RecordBatch>>)> {
        use arrow::array::BinaryArray;
        use arrow::buffer::{Buffer, OffsetBuffer};
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("b", DataType::Binary, false),
        ]));
        let binary = |len: usize, byte: u8| {
            let offsets = OffsetBuffer::new(vec![0, len as i32].into());
            BinaryArray::new(offsets, Buffer::from_vec(vec![byte; len]), None)
        };
        let batch = |key: i64, len: usize| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![key])),
                    Arc::new(binary(len, key as u8)),
                ],
            )
        };
        let short = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..32_768)),
                Arc::new(BinaryArray::from_iter_values((0..32_768).map(|_| b"x"))),
            ],
        )?;
        let partitions = vec![vec![
            short,
            batch(40_000, 50_000_000)?,
            batch(40_001, 2_147_000_000)?,
        ]];
        Ok((schema, partitions))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "allocates more than 4 GiB"]
    async fn bypassed_rows_of_huge_binaries_are_aggregated_like_datafusion() -> Result<()>
    {
        let (schema, partitions) = huge_binaries()?;
        let setup = async |ctx: &SessionContext| {
            let table = datafusion::datasource::MemTable::try_new(
                Arc::clone(&schema),
                partitions.clone(),
            )?;
            ctx.register_table("wide", Arc::new(table))?;
            Ok(())
        };
        let sql = "SELECT k, COUNT(b) FROM wide GROUP BY k";
        // The rows of the binaries bypass the table of the thread (the
        // reviewer's case), and are scattered before a chunk whose bytes they
        // cannot take (see `BypassBuffer`).
        let config = test_session_config().with_target_partitions(1);
        let ctx = crate::test_utils::pipeline_context(config.clone());
        setup(&ctx).await?;
        let batches = partitions.concat();
        assert!(planned_aggregation_bypasses(&ctx, sql, &batches).await?);
        let ctx = datafusion_context(config.clone());
        setup(&ctx).await?;
        assert_bypassed_parity(&ctx, sql, TableCapacity::default(), 1, true).await?;
        assert_aggregate_parity_with_setup(sql, config, &setup).await;
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
            .map(|_| {
                RadixHashAggregate::local_sink_state(&spec, SMALL_TABLES, spec.bypass)
            })
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

    /// The sink of the top-k aggregation `aggregate` (of a single stage),
    /// with tables of [`SMALL_TABLES`].
    fn top_k_sink(aggregate: AggregateExec) -> Result<Sink> {
        top_k_sink_with(aggregate, SMALL_TABLES)
    }

    /// The sink of the top-k aggregation `aggregate` (of a single stage),
    /// with tables of capacity `capacity`.
    fn top_k_sink_with(
        aggregate: AggregateExec,
        capacity: TableCapacity,
    ) -> Result<Sink> {
        let input = Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(
            aggregate.input(),
        )));
        let spec = PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?.spec();
        assert!(spec.top_k.is_some() && !spec.bypass);
        let local = RadixHashAggregate::local_sink_state(&spec, capacity, spec.bypass)?;
        Ok((spec, local, RadixHashAggregate::global_sink_state()?))
    }

    /// Combine the local sink state of `sink`, finalize the aggregation and
    /// return its output, from a single source thread.
    fn top_k_output(sink: Sink) -> Result<RecordBatch> {
        let (spec, mut local, global) = sink;
        RadixHashAggregate::combine(&spec, global.as_ref(), local.as_mut())?;
        let mut batches = vec![];
        if RadixHashAggregate::finalize(global.as_ref())? == SinkFinalize::Ready {
            let source = RadixHashAggregate::global_source_state(&spec, global)?;
            let mut local = RadixHashAggregate::local_source_state()?;
            while let SourceResult::Chunk(chunk) =
                RadixHashAggregate::get_data(&spec, source.as_ref(), local.as_mut())?
            {
                batches.push(chunk.compact()?);
            }
        }
        let schema = batches.first().map(RecordBatch::schema);
        let schema = schema.unwrap_or_else(|| Arc::new(Schema::empty()));
        Ok(arrow::compute::concat_batches(&schema, &batches)?)
    }

    /// The cutoff of the thread of `sink`, an `Int64`.
    fn cutoff(sink: &Sink) -> Option<i64> {
        let cutoff = local_state(sink).cutoff.as_ref()?;
        Some(cutoff.as_primitive::<Int64Type>().value(0))
    }

    #[test]
    fn cutoffs_of_top_k_aggregations_rule_out_rows_before_they_are_aggregated()
    -> Result<()> {
        let sink_batch = |sink: &mut Sink, batch: RecordBatch| -> Result<()> {
            let (spec, local, global) = sink;
            RadixHashAggregate::sink(spec, batch, global.as_ref(), local.as_mut())?;
            Ok(())
        };
        // `SELECT k FROM t GROUP BY k ORDER BY k DESC LIMIT 3`, of nullable keys.
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let aggregate_by_key = |schema: &SchemaRef| -> Result<AggregateExec> {
            Ok(AggregateExec::try_new(
                AggregateMode::Single,
                PhysicalGroupBy::new_single(vec![(col("k", schema)?, "k".into())]),
                vec![],
                vec![],
                MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(schema), None)?,
                Arc::clone(schema),
            )?
            .try_optimize_topk(3, "k", SortOptions::new(true, false))
            .expect("a top-k aggregation")
            .data)
        };
        let mut sink = top_k_sink(aggregate_by_key(&schema)?)?;
        let key_rows = |keys: Vec<Option<i64>>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(keys))],
            )
        };
        assert_eq!(cutoff(&sink), None);
        // 16 groups fill the table, whose flush keeps 15, 14 and 13: at least
        // 3 groups are 13 or more from then on.
        sink_batch(&mut sink, key_rows((0..16).map(Some).collect())?)?;
        assert_eq!(cutoff(&sink), Some(13));
        assert_eq!(local_state(&sink).table.num_groups(), 0);
        // Only the rows of keys that are not worse are aggregated: 13 (not
        // worse), 20, and the `NULL` key (whose group is always kept).
        let batch = key_rows(vec![Some(5), Some(13), Some(20), None, Some(1), Some(-4)])?;
        sink_batch(&mut sink, batch)?;
        assert_eq!(local_state(&sink).table.num_groups(), 3);
        assert_eq!(local_state(&sink).table_rows, 3);
        sink_batch(&mut sink, key_rows((0..13).map(Some).collect())?)?;
        assert_eq!(local_state(&sink).table.num_groups(), 3);
        let output = top_k_output(sink)?;
        let mut keys = output
            .column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, [None, Some(14), Some(15), Some(20)]);

        // A table that aggregated as many rows as it can hold groups (16) is
        // flushed even if it is not full, once it holds 3 groups at least.
        let mut sink = top_k_sink(aggregate_by_key(&schema)?)?;
        sink_batch(&mut sink, key_rows((0..15).map(|i| Some(i % 2)).collect())?)?;
        sink_batch(&mut sink, key_rows(vec![Some(0), Some(1)])?)?;
        assert_eq!(local_state(&sink).table.num_groups(), 2);
        assert_eq!(cutoff(&sink), None);
        // Its flush keeps 7, 1 and 0.
        sink_batch(&mut sink, key_rows(vec![Some(7); 9])?)?;
        assert_eq!(cutoff(&sink), Some(0));
        assert_eq!(local_state(&sink).table.num_groups(), 0);
        sink_batch(&mut sink, key_rows(vec![Some(4); 8])?)?;
        sink_batch(&mut sink, key_rows(vec![Some(-1), Some(3), Some(9)])?)?;
        assert_eq!(local_state(&sink).table.num_groups(), 3);
        assert_eq!(local_state(&sink).table_rows, 10);
        let output = top_k_output(sink)?;
        let mut keys = output
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec();
        keys.sort_unstable();
        assert_eq!(keys, [4, 7, 9]);

        // `SELECT k FROM t GROUP BY k ORDER BY k LIMIT 2`: the least keys.
        let ascending = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![],
            vec![],
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?,
            Arc::clone(&schema),
        )?
        .try_optimize_topk(2, "k", SortOptions::new(false, false))
        .expect("a top-k aggregation")
        .data;
        let mut sink = top_k_sink(ascending)?;
        sink_batch(&mut sink, key_rows((0..16).rev().map(Some).collect())?)?;
        assert_eq!(cutoff(&sink), Some(1));
        sink_batch(&mut sink, key_rows(vec![Some(5), Some(1), Some(-3), None])?)?;
        assert_eq!(local_state(&sink).table.num_groups(), 3);
        let output = top_k_output(sink)?;
        let mut keys = output
            .column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, [None, Some(-3), Some(0)]);
        Ok(())
    }

    /// `SELECT k, MAX(v) FROM t GROUP BY k ORDER BY MAX(v) DESC LIMIT limit`
    /// (`descending`) or `SELECT k, MIN(v) ... ORDER BY MIN(v) LIMIT limit`, of
    /// a single stage, of nullable values `v`.
    fn top_k_by_aggregate(limit: usize, descending: bool) -> Result<AggregateExec> {
        use datafusion::functions_aggregate::min_max::{max_udaf, min_udaf};
        let schema = key_value_schema();
        let (function, name) = match descending {
            true => (max_udaf(), "max(v)"),
            false => (min_udaf(), "min(v)"),
        };
        let aggregate = AggregateExprBuilder::new(function, vec![col("v", &schema)?])
            .schema(Arc::clone(&schema))
            .alias(name)
            .build()?;
        Ok(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![Arc::new(aggregate)],
            vec![None],
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?,
            Arc::clone(&schema),
        )?
        .try_optimize_topk(limit, name, SortOptions::new(descending, false))
        .expect("a top-k aggregation")
        .data)
    }

    /// The schema of keys `k` and nullable values `v`.
    fn key_value_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
        ]))
    }

    /// A batch of keys and values of [`key_value_schema`].
    fn key_value_batch(rows: &[(i64, Option<i64>)]) -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            key_value_schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Int64Array::from_iter(rows.iter().map(|row| row.1))),
            ],
        )?)
    }

    /// The groups of `output`, of a key and a nullable value, sorted.
    fn key_values(output: &RecordBatch) -> Vec<(i64, Option<i64>)> {
        let keys = output.column(0).as_primitive::<Int64Type>();
        let values = output.column(1).as_primitive::<Int64Type>();
        let mut groups = keys
            .values()
            .iter()
            .zip(values.iter())
            .map(|(&key, value)| (key, value))
            .collect::<Vec<_>>();
        groups.sort_unstable();
        groups
    }

    #[test]
    fn cutoffs_of_top_k_aggregations_by_aggregates_rule_out_rows_before_they_are_aggregated()
    -> Result<()> {
        let sink_batch = |sink: &mut Sink, rows: &[(i64, Option<i64>)]| -> Result<()> {
            let (spec, local, global) = sink;
            let batch = key_value_batch(rows)?;
            RadixHashAggregate::sink(spec, batch, global.as_ref(), local.as_mut())?;
            Ok(())
        };
        // `MAX(v) DESC LIMIT 3`: 16 groups fill the table, whose flush keeps
        // the groups of 150, 140 and 130: at least 3 groups are 130 or more
        // from then on.
        let mut sink = top_k_sink(top_k_by_aggregate(3, true)?)?;
        let rows = (0..16).map(|k| (k, Some(k * 10))).collect::<Vec<_>>();
        sink_batch(&mut sink, &rows)?;
        assert_eq!(cutoff(&sink), Some(130));
        assert_eq!(local_state(&sink).table.num_groups(), 0);
        // Only the rows whose values are not worse are aggregated: a group
        // that comes back with a better value (1), and values that tie (20) or
        // beat the cutoff (13). `NULL`s change no group.
        sink_batch(
            &mut sink,
            &[
                (1, Some(200)),
                (2, Some(5)),
                (20, Some(130)),
                (21, None),
                (13, Some(131)),
                (3, Some(129)),
            ],
        )?;
        assert_eq!(local_state(&sink).table.num_groups(), 3);
        assert_eq!(local_state(&sink).table_rows, 3);
        // The 3 best groups, with their values.
        let output = top_k_output(sink)?;
        assert_eq!(
            key_values(&output),
            [(1, Some(200)), (14, Some(140)), (15, Some(150))]
        );

        // `MIN(v) LIMIT 2`, the least values.
        let mut sink = top_k_sink(top_k_by_aggregate(2, false)?)?;
        let rows = (0..16).map(|k| (k, Some(100 - k))).collect::<Vec<_>>();
        sink_batch(&mut sink, &rows)?;
        assert_eq!(cutoff(&sink), Some(86));
        sink_batch(&mut sink, &[(0, Some(3)), (1, Some(87)), (2, Some(86))])?;
        assert_eq!(local_state(&sink).table.num_groups(), 2);
        let output = top_k_output(sink)?;
        assert_eq!(key_values(&output), [(0, Some(3)), (15, Some(85))]);

        // Groups without values come last: as many as there are fewer groups
        // with values than the limit.
        let mut sink = top_k_sink(top_k_by_aggregate(3, true)?)?;
        sink_batch(&mut sink, &[(0, None), (1, None), (2, Some(5)), (3, None)])?;
        assert_eq!(cutoff(&sink), None);
        let output = top_k_output(sink)?;
        let groups = key_values(&output);
        assert_eq!(groups.len(), 3);
        assert!(groups.contains(&(2, Some(5))));
        assert!(
            groups
                .iter()
                .all(|&(key, value)| (key, value) == (2, Some(5)) || value.is_none())
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn flushes_of_top_k_aggregations_by_aggregates_keep_their_top_groups()
    -> Result<()> {
        // `SELECT k, MAX(v) ... ORDER BY MAX(v) DESC LIMIT 5` over 3,000 groups
        // of 3 rows each, whose rows come in batches of 50, on two threads: the
        // flushes of the tables (of 16 groups) keep 5 groups each, and the
        // groups of the best values are produced, with their values.
        let aggregate = top_k_by_aggregate(5, true)?;
        let input = Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(
            aggregate.input(),
        )));
        let spec = PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?.spec();
        let global = RadixHashAggregate::global_sink_state()?;
        let mut locals = (0..2)
            .map(|_| {
                RadixHashAggregate::local_sink_state(&spec, SMALL_TABLES, spec.bypass)
            })
            .collect::<Result<Vec<_>>>()?;
        let value = |row: i64| (row * 7_919) % 9_001;
        let rows = (0..9_000)
            .map(|row| ((row * 31) % 3_000, Some(value(row))))
            .collect::<Vec<_>>();
        for (batch, rows) in rows.chunks(50).enumerate() {
            let local = &mut locals[batch % 2];
            let batch = key_value_batch(rows)?;
            RadixHashAggregate::sink(&spec, batch, global.as_ref(), local.as_mut())?;
        }
        for local in &mut locals {
            RadixHashAggregate::combine(&spec, global.as_ref(), local.as_mut())?;
        }
        let sink = global.downcast_ref::<RadixSinkState>().unwrap();
        let groups = sink
            .partitions
            .iter()
            .map(|data| {
                let data = data.lock();
                data.states
                    .iter()
                    .map(GroupedStates::num_groups)
                    .sum::<usize>()
            })
            .sum::<usize>();
        assert!(groups < 9_000 / 16 * 5 + 10, "{groups}");
        assert_eq!(
            RadixHashAggregate::finalize(global.as_ref())?,
            SinkFinalize::Ready
        );
        let source = RadixHashAggregate::global_source_state(&spec, Arc::clone(&global))?;
        let mut local = RadixHashAggregate::local_source_state()?;
        let mut output = vec![];
        while let SourceResult::Chunk(chunk) =
            RadixHashAggregate::get_data(&spec, source.as_ref(), local.as_mut())?
        {
            output.push(chunk.compact()?);
        }
        let output = arrow::compute::concat_batches(&output[0].schema(), &output)?;
        let mut expected = BTreeMap::<i64, i64>::new();
        for &(key, value) in &rows {
            let max = expected.entry(key).or_insert(i64::MIN);
            *max = (*max).max(value.expect("a value"));
        }
        let mut expected = expected
            .into_iter()
            .map(|(key, max)| (key, Some(max)))
            .collect::<Vec<_>>();
        expected.sort_unstable_by_key(|&(key, max)| (std::cmp::Reverse(max), key));
        expected.truncate(5);
        expected.sort_unstable();
        assert_eq!(key_values(&output), expected);
        Ok(())
    }

    /// `tick(v)`: a volatile function that numbers the rows it is evaluated
    /// for, from 1, and counts them in `ticks` (its argument only makes it a
    /// function of the rows).
    #[derive(Debug)]
    struct Tick {
        ticks: Arc<AtomicUsize>,
        signature: datafusion_expr::Signature,
    }

    impl PartialEq for Tick {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.ticks, &other.ticks)
        }
    }

    impl Eq for Tick {}

    impl std::hash::Hash for Tick {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            Arc::as_ptr(&self.ticks).hash(state);
        }
    }

    impl datafusion_expr::ScalarUDFImpl for Tick {
        fn name(&self) -> &str {
            "tick"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn return_type(&self, _: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn invoke_with_args(
            &self,
            args: datafusion_expr::ScalarFunctionArgs,
        ) -> Result<datafusion_expr::ColumnarValue> {
            let rows = args.number_rows;
            let first = self.ticks.fetch_add(rows, Ordering::SeqCst) as i64;
            let ticks = Int64Array::from_iter_values(first + 1..=first + rows as i64);
            Ok(datafusion_expr::ColumnarValue::Array(Arc::new(ticks)))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn volatile_values_of_top_k_aggregations_are_evaluated_once() -> Result<()> {
        use super::super::test_support::lower_with_tiny_tables;
        use crate::bridge::PipelineRootExec;
        // 40,000 distinct keys `k` of a single partition, in batches of 2,048
        // rows: the first 32,768 rows of a thread fill its table, whose flush
        // sets a cutoff, which the rows after it are compared with (all of them
        // with tiny tables). `tick(v)` numbers the rows it is evaluated for:
        // evaluated once for every row, its values are 1 to 40,000, the
        // greatest 40,000. The native top-k aggregation evaluates its keys and
        // arguments (`GROUP BY tick(v)`, `MAX(tick(v))`) once, both to compare
        // them with the cutoff and to aggregate them (evaluated again for the
        // rows that passed the cutoff, the greatest value was 47,232).
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batches = (0..40_000)
            .step_by(2_048)
            .map(|start| {
                let rows = start..(start + 2_048).min(40_000);
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from_iter_values(
                            rows.clone().map(|i| (i * 7_919) % 40_009),
                        )),
                        Arc::new(Int64Array::from_iter_values(rows)),
                    ],
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let setup = |ctx: &SessionContext| -> Result<Arc<AtomicUsize>> {
            let table = datafusion::datasource::MemTable::try_new(
                Arc::clone(&schema),
                vec![batches.clone()],
            )?;
            ctx.register_table("t", Arc::new(table))?;
            let ticks = Arc::new(AtomicUsize::new(0));
            ctx.register_udf(datafusion_expr::ScalarUDF::new_from_impl(Tick {
                ticks: Arc::clone(&ticks),
                signature: datafusion_expr::Signature::exact(
                    vec![DataType::Int64],
                    datafusion_expr::Volatility::Volatile,
                ),
            }));
            Ok(ticks)
        };
        // The greatest value of the output column `column` of `batches`.
        let greatest = |batches: &[RecordBatch], column: usize| {
            let values = batches
                .iter()
                .flat_map(|batch| batch.column(column).as_primitive::<Int64Type>().iter())
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 1, "{values:?}");
            values[0]
        };
        for (sql, column) in [
            (
                "SELECT k, MAX(tick(v)) AS m FROM t GROUP BY k \
                 ORDER BY m DESC NULLS LAST LIMIT 1",
                1,
            ),
            // (A key of the aggregation, which DataFusion evaluates in it: of a
            // `SELECT DISTINCT tick(v)`, in a projection below it.)
            (
                "SELECT tick(v) AS x FROM t GROUP BY tick(v) ORDER BY x DESC LIMIT 1",
                0,
            ),
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let context = format!("{sql} on {target_partitions} partitions");
                // DataFusion.
                let ctx = datafusion_context(config.clone());
                let ticks = setup(&ctx)?;
                let output = ctx.sql(sql).await?.collect().await?;
                assert_eq!(greatest(&output, column), Some(40_000), "{context}");
                assert_eq!(ticks.load(Ordering::SeqCst), 40_000, "{context}");
                // The native top-k aggregation, with tables of the default
                // capacity and tiny ones.
                for tiny in [false, true] {
                    let ctx = datafusion_context(config.clone());
                    let ticks = setup(&ctx)?;
                    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                    let root = if tiny {
                        lower_with_tiny_tables(&plan)?
                    } else {
                        lower(&plan)?
                    };
                    assert_eq!(native_top_k(&root), 1, "{context}");
                    let exec: Arc<dyn ExecutionPlan> =
                        Arc::new(PipelineRootExec::try_new(root, plan)?);
                    let output =
                        datafusion_physical_plan::collect(exec, ctx.task_ctx()).await?;
                    let context = format!("{context}, tiny tables: {tiny}");
                    assert_eq!(greatest(&output, column), Some(40_000), "{context}");
                    assert_eq!(ticks.load(Ordering::SeqCst), 40_000, "{context}");
                }
            }
        }
        Ok(())
    }

    /// The number of native top-k aggregations of `root`.
    fn native_top_k(root: &Arc<dyn crate::PhysicalOperator>) -> usize {
        let top_k = root
            .downcast_ref::<PhysicalHashAggregate>()
            .is_some_and(|aggregate| aggregate.spec().top_k.is_some());
        usize::from(top_k) + root.children().iter().map(native_top_k).sum::<usize>()
    }

    #[test]
    fn threads_of_soft_limited_aggregations_finish_once_they_have_their_groups()
    -> Result<()> {
        // `SELECT DISTINCT k FROM t LIMIT 5`: a thread finishes the sink once
        // its table holds 5 groups, and keeps finishing it.
        let schema = key_schema();
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![],
            vec![],
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?,
            Arc::clone(&schema),
        )?
        .try_optimize_distinct_soft_limit(5)
        .expect("a soft limit")
        .data;
        let input = Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(
            aggregate.input(),
        )));
        let spec = PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?.spec();
        assert_eq!(spec.soft_limit, Some(5));
        assert!(spec.top_k.is_none() && !spec.bypass);
        let global = RadixHashAggregate::global_sink_state()?;
        let mut local = RadixHashAggregate::local_sink_state(&spec, SMALL_TABLES, false)?;
        let mut sink = |keys: Vec<i64>| {
            RadixHashAggregate::sink(
                &spec,
                key_batch(keys)?,
                global.as_ref(),
                local.as_mut(),
            )
        };
        assert!(matches!(sink(vec![0, 0, 1, 1])?, SinkResult::NeedMoreInput));
        assert!(matches!(sink(vec![2, 3])?, SinkResult::NeedMoreInput));
        assert!(matches!(sink(vec![4, 4, 5])?, SinkResult::Finished));
        assert!(matches!(sink(vec![6])?, SinkResult::Finished));
        RadixHashAggregate::combine(&spec, global.as_ref(), local.as_mut())?;
        let partitions = global.downcast_ref::<RadixSinkState>().unwrap();
        let groups = partitions
            .partitions
            .iter()
            .map(|data| {
                let data = data.lock();
                data.states
                    .iter()
                    .map(GroupedStates::num_groups)
                    .sum::<usize>()
            })
            .sum::<usize>();
        assert_eq!(groups, 7);
        Ok(())
    }

    #[test]
    fn tables_of_top_k_aggregations_hold_twice_their_limit() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let aggregate = |limit: usize| -> Result<AggregateExec> {
            Ok(AggregateExec::try_new(
                AggregateMode::Single,
                PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
                vec![],
                vec![],
                MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?,
                Arc::clone(&schema),
            )?
            .try_optimize_topk(limit, "k", SortOptions::new(false, false))
            .expect("a top-k aggregation")
            .data)
        };
        let capacity = |limit: usize, tables: TableCapacity| -> Result<usize> {
            let input = Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(
                aggregate(limit)?.input(),
            )));
            let spec =
                PhysicalHashAggregate::try_new_single_stage(&aggregate(limit)?, input)?
                    .spec();
            let local = RadixHashAggregate::local_sink_state(&spec, tables, spec.bypass)?;
            Ok(local.downcast_ref::<RadixLocalState>().unwrap().capacity)
        };
        // A table of more groups than the limit would set no cutoff (40,000
        // groups out of every 32,768).
        let tables = TableCapacity::default();
        assert_eq!(capacity(10, tables)?, 1 << 15);
        assert_eq!(capacity(32_768, tables)?, 65_536);
        assert_eq!(capacity(32_769, tables)?, 65_538);
        assert_eq!(capacity(40_000, tables)?, 80_000);
        // Up to the maximum capacity.
        assert_eq!(capacity(1_000, SMALL_TABLES)?, 64);
        assert_eq!(capacity(3, SMALL_TABLES)?, 16);
        // Its first flush keeps the best half of its groups, and sets a
        // cutoff.
        let tables = TableCapacity {
            initial: 16,
            maximum: 1_024,
        };
        let mut sink = top_k_sink_with(aggregate(40)?, tables)?;
        assert_eq!(local_state(&sink).capacity, 80);
        let (spec, local, global) = &mut sink;
        let keys = (0..80).map(|i| Some((i * 37) % 80)).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(keys))],
        )?;
        RadixHashAggregate::sink(spec, batch, global.as_ref(), local.as_mut())?;
        assert_eq!(cutoff(&sink), Some(39));
        Ok(())
    }
}
