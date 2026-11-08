// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`RadixHashAggregate`]: the grouped aggregation of a
//! [`PhysicalHashAggregate`](super::PhysicalHashAggregate), with
//! thread-local pre-aggregation and radix-partitioned parallel finalization
//! (DuckDB's `RadixPartitionedHashTable`).
//!
//! ```text
//!  sink (every thread)         combine (every thread)       get_data (every thread)
//!
//!  chunks -> GroupedHashTable -> hash the keys of the   ->  claim partition p,
//!            (thread-local)      groups once, scatter       merge the states of
//!                                the groups to radix        its groups from all
//!                                partition (hash >> 58)     threads into a new
//!                                of 64 partitions           GroupedHashTable,
//!                                                           evaluate, emit chunks
//! ```
//!
//! * `sink`: every thread aggregates the rows of its chunks into its own
//!   [`GroupedHashTable`] (DataFusion's partial aggregation), without any
//!   synchronization.
//! * `combine`: every thread emits the keys and partial states of the groups
//!   of its table, hashes the keys of every group once and scatters the
//!   groups to [`NUM_PARTITIONS`] radix partitions by the upper
//!   [`RADIX_BITS`] bits of their hash. The partial states of a group, from
//!   all threads, all end up in the same partition. This replaces the
//!   `RepartitionExec(Hash)` between DataFusion's partial and final
//!   aggregations: the states are moved by reference, not sent through
//!   channels.
//! * `finalize`: records the partitions that have groups. If there is none,
//!   the aggregation has no output.
//! * `get_data`: the threads of the pipeline that reads the aggregation claim
//!   the partitions one by one. A thread merges the states of the groups of
//!   its partition into a new [`GroupedHashTable`] (DataFusion's final
//!   aggregation) and evaluates it, and emits the result in chunks of at most
//!   [`DEFAULT_VECTOR_SIZE`] rows. Partitions have disjoint groups, so the
//!   threads finalize them in parallel, without sharing anything.
//!
//! The partition hash uses another seed than the hash tables
//! ([`GroupValues`](datafusion_physical_plan::aggregates::group_values::GroupValues)),
//! so that the groups of a partition, which share the upper bits of their
//! partition hash, still spread over the hash table of the partition.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute::take;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::{Result, internal_err};
use parking_lot::Mutex;

use super::grouped_ht::{GroupedHashTable, GroupedStates};
use super::hash_aggregate::{AggregateSpec, InputBuffer};
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

/// The hash state of the partition hash: another seed than the one of the
/// hash tables of DataFusion's aggregation and repartitioning.
const PARTITION_HASH: RandomState = RandomState::with_seed(0x8f1b_c3a5_d2e4_7a61);

/// The sink and source roles of a grouped aggregation, see the
/// [module documentation](self).
pub(crate) struct RadixHashAggregate;

/// The global sink state: the radix partitions.
#[derive(Debug)]
struct RadixSinkState {
    /// For every partition, the partial states of its groups, from all
    /// threads (one entry per thread that had groups in the partition).
    partitions: Vec<Mutex<Vec<GroupedStates>>>,
    /// The partitions that have groups, set by `finalize`.
    non_empty: Mutex<Option<Vec<usize>>>,
}

impl GlobalSinkState for RadixSinkState {}

impl RadixSinkState {
    /// Scatter `states`, the groups of a thread-local table, to their
    /// partitions; `hashes` are the partition hashes of the groups.
    fn add(&self, states: GroupedStates, hashes: &[u64]) -> Result<()> {
        for (partition, states) in partition(states, hashes)? {
            self.partitions[partition].lock().push(states);
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
    /// The number of groups flushed so far.
    flushed: usize,
    /// The number of distinct groups among the flushed groups.
    distinct_flushed: DistinctEstimate,
    /// The input rows, if they must be sorted before they are aggregated.
    buffer: InputBuffer,
}

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
    fn aggregate(
        &mut self,
        spec: &AggregateSpec,
        global: &RadixSinkState,
        batch: &RecordBatch,
    ) -> Result<()> {
        self.table.update(spec, batch)?;
        if self.table.num_groups() < self.capacity {
            return Ok(());
        }
        let distinct = self.distinct_flushed.estimate();
        let repeated = self.flushed.saturating_sub(distinct);
        // Room for all groups, with some slack for the estimate.
        let needed = (distinct + distinct / 4).max(self.capacity * 2);
        if repeated >= self.capacity
            && repeated * 2 >= self.flushed
            && needed <= self.maximum_capacity
        {
            self.capacity = needed.next_power_of_two().min(self.maximum_capacity);
            return Ok(());
        }
        self.flush(global)
    }

    /// Flush the groups of the table into the partitions of `global`.
    fn flush(&mut self, global: &RadixSinkState) -> Result<()> {
        let Some(states) = self.table.emit_states()? else {
            return Ok(());
        };
        let hashes = hash_groups(&states)?;
        self.flushed += hashes.len();
        self.distinct_flushed.add(&hashes);
        global.add(states, &hashes)
    }
}

/// The number of groups a thread-local table holds before it is flushed or
/// grown, see [`RadixLocalState::aggregate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableCapacity {
    /// The initial capacity.
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

impl LocalSinkState for RadixLocalState {}

/// The global source state: the partitions to finalize.
#[derive(Debug)]
struct RadixSourceState {
    sink: Arc<RadixSinkState>,
    /// The partitions that have groups.
    partitions: Vec<usize>,
    /// The next entry of `partitions` to claim.
    next: AtomicUsize,
}

impl GlobalSourceState for RadixSourceState {
    fn max_threads(&self) -> usize {
        self.partitions.len().max(1)
    }
}

/// The thread-local source state: the evaluated groups of the partition the
/// thread finalized last, and how many of them were emitted.
#[derive(Debug, Default)]
struct RadixLocalSource {
    output: Option<RecordBatch>,
    offset: usize,
}

impl LocalSourceState for RadixLocalSource {}

impl RadixHashAggregate {
    pub(crate) fn global_sink_state() -> Result<Arc<dyn GlobalSinkState>> {
        Ok(Arc::new(RadixSinkState {
            partitions: (0..NUM_PARTITIONS).map(|_| Mutex::new(vec![])).collect(),
            non_empty: Mutex::new(None),
        }))
    }

    pub(crate) fn local_sink_state(
        spec: &AggregateSpec,
        capacity: TableCapacity,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(RadixLocalState {
            table: GroupedHashTable::try_new_update(spec)?,
            capacity: capacity.initial.max(1),
            maximum_capacity: capacity.maximum,
            flushed: 0,
            distinct_flushed: DistinctEstimate::default(),
            buffer: InputBuffer::default(),
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
        // Like DataFusion's partial aggregation of a partition without rows,
        // a thread without groups contributes the grouping sets without
        // group columns.
        if local.flushed == 0 {
            local.table.add_empty_grouping_sets(spec)?;
        }
        local.flush(global)
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
        Ok(Arc::new(RadixSourceState {
            sink,
            partitions,
            next: AtomicUsize::new(0),
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
            if let Some(output) = &local.output {
                let remaining = output.num_rows() - local.offset;
                if remaining > 0 {
                    let len = remaining.min(DEFAULT_VECTOR_SIZE);
                    let chunk = output.slice(local.offset, len);
                    local.offset += len;
                    return Ok(SourceResult::Chunk(DataChunk::from_record_batch(chunk)));
                }
                local.output = None;
            }
            let next = state.next.fetch_add(1, Ordering::Relaxed);
            let Some(&partition) = state.partitions.get(next) else {
                return Ok(SourceResult::Finished);
            };
            let states = std::mem::take(&mut *state.sink.partitions[partition].lock());
            local.output = finalize_partition(spec, &states)?;
            local.offset = 0;
        }
    }
}

/// The partition hashes of the groups of `states`: the hashes of their keys.
fn hash_groups(states: &GroupedStates) -> Result<Vec<u64>> {
    let mut hashes = vec![0; states.num_groups()];
    create_hashes(&states.keys, &PARTITION_HASH, &mut hashes)?;
    Ok(hashes)
}

/// The partition of a group with partition hash `hash`: its upper
/// [`RADIX_BITS`] bits.
fn partition_of(hash: u64) -> usize {
    (hash >> (u64::BITS - RADIX_BITS)) as usize
}

/// Scatter the groups of `states`, with partition hashes `hashes`, to their
/// radix partitions. Returns the groups of every partition that has some, in
/// the order of their partitions.
///
/// The groups are ordered by partition with a single `take` of every column,
/// and every partition gets a zero-copy slice of the result.
fn partition(
    states: GroupedStates,
    hashes: &[u64],
) -> Result<Vec<(usize, GroupedStates)>> {
    let num_groups = states.num_groups();

    if u32::try_from(num_groups).is_err() {
        return internal_err!("{OPERATOR} cannot partition {num_groups} groups at once");
    }

    // A counting sort of the groups by partition.
    let mut offsets = vec![0usize; NUM_PARTITIONS + 1];
    for &hash in hashes {
        offsets[partition_of(hash) + 1] += 1;
    }
    for i in 0..NUM_PARTITIONS {
        offsets[i + 1] += offsets[i];
    }
    let mut next = offsets.clone();
    let mut indices = vec![0u32; num_groups];
    for (group, &hash) in hashes.iter().enumerate() {
        let position = &mut next[partition_of(hash)];
        indices[*position] = group as u32;
        *position += 1;
    }

    let sorted = if offsets.windows(2).any(|w| w[1] - w[0] == num_groups) {
        // All groups are in the same partition.
        states
    } else {
        let indices = UInt32Array::from(indices);
        let take_all = |arrays: &[arrow::array::ArrayRef]| {
            arrays
                .iter()
                .map(|array| Ok(take(array.as_ref(), &indices, None)?))
                .collect::<Result<Vec<_>>>()
        };
        GroupedStates {
            keys: take_all(&states.keys)?,
            states: states
                .states
                .iter()
                .map(|state| take_all(state))
                .collect::<Result<_>>()?,
        }
    };
    Ok(offsets
        .windows(2)
        .enumerate()
        .filter(|(_, range)| range[1] > range[0])
        .map(|(partition, range)| {
            let (offset, len) = (range[0], range[1] - range[0]);
            let slice = |arrays: &[arrow::array::ArrayRef]| {
                arrays
                    .iter()
                    .map(|array| array.slice(offset, len))
                    .collect::<Vec<_>>()
            };
            let states = GroupedStates {
                keys: slice(&sorted.keys),
                states: sorted.states.iter().map(|state| slice(state)).collect(),
            };
            (partition, states)
        })
        .collect())
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

/// Finalize a partition: merge the partial states of its groups, from all
/// threads, and evaluate the groups. Returns the output rows of the
/// partition, `None` if it has no groups.
fn finalize_partition(
    spec: &AggregateSpec,
    states: &[GroupedStates],
) -> Result<Option<RecordBatch>> {
    let mut table = GroupedHashTable::try_new_merge(spec)?;
    for states in states {
        table.merge(states)?;
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

    use arrow::array::{Array, ArrayRef, AsArray, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{col, lit};
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::aggregates::{
        AggregateExec, AggregateMode, PhysicalGroupBy,
    };

    use super::super::PhysicalHashAggregate;
    use super::super::hash_aggregate::AggregateSpec;
    use super::super::test_support::lower;
    use crate::executor::Executor;
    use crate::test_utils::canonical_batch;

    fn partition_of(keys: &[ArrayRef], row: usize) -> usize {
        let keys = keys.iter().map(|key| key.slice(row, 1)).collect::<Vec<_>>();
        let mut hash = [0];
        create_hashes(&keys, &PARTITION_HASH, &mut hash).unwrap();
        (hash[0] >> (u64::BITS - RADIX_BITS)) as usize
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
            let partitions = partition(states.clone(), &hash_groups(&states)?)?;
            assert!(partitions.windows(2).all(|w| w[0].0 < w[1].0), "{n}");
            assert_eq!(
                partitions
                    .iter()
                    .map(|(_, p)| p.num_groups())
                    .sum::<usize>(),
                n as usize
            );
            let mut seen = vec![false; n as usize];
            for (partition, groups) in &partitions {
                assert!(*partition < NUM_PARTITIONS);
                assert!(groups.num_groups() > 0);
                for row in 0..groups.num_groups() {
                    assert_eq!(partition_of(&groups.keys, row), *partition);
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
            assert!(seen.iter().all(|&seen| seen));
        }
        // 5000 groups are spread over all partitions.
        let states = groups(5000);
        let hashes = hash_groups(&states)?;
        assert_eq!(partition(states, &hashes)?.len(), NUM_PARTITIONS);
        Ok(())
    }

    /// The spec, a local and the global sink state of a thread of
    /// `COUNT(*) GROUP BY k`.
    type Sink = (
        Arc<AggregateSpec>,
        Box<dyn LocalSinkState>,
        Arc<dyn GlobalSinkState>,
    );

    /// The sink of `COUNT(*) GROUP BY k`, with tables of capacity `capacity`.
    fn count_by_key(capacity: TableCapacity) -> Result<Sink> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
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
        let spec = PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?.spec();
        let local = RadixHashAggregate::local_sink_state(&spec, capacity)?;
        Ok((spec, local, RadixHashAggregate::global_sink_state()?))
    }

    fn key_batch(keys: impl IntoIterator<Item = i64>) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        Ok(RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from_iter_values(keys))],
        )?)
    }

    #[test]
    fn full_tables_are_flushed_or_grown() -> Result<()> {
        let capacity = TableCapacity {
            initial: 16,
            maximum: 64,
        };
        // Distinct keys: the table is flushed whenever it is full.
        let (spec, mut local, global) = count_by_key(capacity)?;
        for batch in 0..20 {
            let batch = key_batch(batch * 10..(batch + 1) * 10)?;
            RadixHashAggregate::sink(&spec, batch, global.as_ref(), local.as_mut())?;
        }
        let state = local.downcast_ref::<RadixLocalState>().unwrap();
        assert_eq!(state.capacity, 16);
        assert!(state.flushed >= 180, "{}", state.flushed);

        // The same 40 keys over and over: once the flushes emitted many of
        // them repeatedly, the table grows to hold all of them.
        let (spec, mut local, global) = count_by_key(capacity)?;
        for batch in 0..40 {
            let batch = key_batch((0..10).map(|i| (batch * 10 + i) % 40))?;
            RadixHashAggregate::sink(&spec, batch, global.as_ref(), local.as_mut())?;
        }
        let state = local.downcast_ref::<RadixLocalState>().unwrap();
        assert_eq!(state.capacity, 64);
        let flushed = state.flushed;
        assert!(flushed < 400, "{flushed}");
        for batch in 0..40 {
            let batch = key_batch((0..10).map(|i| (batch * 10 + i) % 40))?;
            RadixHashAggregate::sink(&spec, batch, global.as_ref(), local.as_mut())?;
        }
        let state = local.downcast_ref::<RadixLocalState>().unwrap();
        assert_eq!(state.flushed, flushed);
        assert_eq!(state.table.num_groups(), 40);

        // Unless they do not fit.
        let (spec, mut local, global) = count_by_key(capacity)?;
        for batch in 0..40 {
            let batch = key_batch((0..10).map(|i| (batch * 10 + i) % 100))?;
            RadixHashAggregate::sink(&spec, batch, global.as_ref(), local.as_mut())?;
        }
        let state = local.downcast_ref::<RadixLocalState>().unwrap();
        assert_eq!(state.capacity, 16);
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
                partition_of(&[key], 0) == target
            })
            .take(n)
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn large_partitions_are_emitted_in_chunks() -> Result<()> {
        // 5000 groups in a single partition: one thread finalizes them and
        // emits them in chunks of at most `DEFAULT_VECTOR_SIZE` rows.
        let keys = keys_of_partition(5, 5000);
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
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
}
