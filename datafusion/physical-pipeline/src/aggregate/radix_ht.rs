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

/// The thread-local sink state.
#[derive(Debug)]
struct RadixLocalState {
    table: GroupedHashTable,
    /// The input rows, if they must be sorted before they are aggregated.
    buffer: InputBuffer,
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
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(RadixLocalState {
            table: GroupedHashTable::try_new_update(spec)?,
            buffer: InputBuffer::default(),
        }))
    }

    pub(crate) fn sink(
        spec: &AggregateSpec,
        batch: RecordBatch,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let local = downcast_state_mut::<RadixLocalState>(lstate.as_any_mut(), OPERATOR)?;
        if spec.input_order.is_some() {
            local.buffer.push(batch);
            return Ok(());
        }
        local.table.update(spec, &batch)
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
                local.table.update(spec, &batch)?;
            }
        }
        local.table.add_empty_grouping_sets(spec)?;
        let Some(states) = local.table.emit_states()? else {
            return Ok(());
        };
        for (partition, states) in partition(states)? {
            global.partitions[partition].lock().push(states);
        }
        Ok(())
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

/// Scatter the groups of `states` to their radix partitions. Returns the
/// groups of every partition that has some, in the order of their partitions.
///
/// The groups are ordered by partition with a single `take` of every column,
/// and every partition gets a zero-copy slice of the result.
fn partition(states: GroupedStates) -> Result<Vec<(usize, GroupedStates)>> {
    let num_groups = states.num_groups();
    let mut hashes = vec![0; num_groups];
    create_hashes(&states.keys, &PARTITION_HASH, &mut hashes)?;
    let partition_of = |hash: u64| (hash >> (u64::BITS - RADIX_BITS)) as usize;

    // A counting sort of the groups by partition.
    let mut offsets = vec![0u32; NUM_PARTITIONS + 1];
    for &hash in &hashes {
        offsets[partition_of(hash) + 1] += 1;
    }
    for i in 0..NUM_PARTITIONS {
        offsets[i + 1] += offsets[i];
    }
    let mut next = offsets.clone();
    let mut indices = vec![0u32; num_groups];
    for (group, &hash) in hashes.iter().enumerate() {
        let position = &mut next[partition_of(hash)];
        indices[*position as usize] = group as u32;
        *position += 1;
    }

    let sorted = if offsets.windows(2).any(|w| w[1] - w[0] == num_groups as u32) {
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
            let (offset, len) = (range[0] as usize, (range[1] - range[0]) as usize);
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
