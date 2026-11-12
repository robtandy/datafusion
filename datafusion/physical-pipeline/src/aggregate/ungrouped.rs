// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The aggregation without `GROUP BY` of a
//! [`PhysicalHashAggregate`](super::PhysicalHashAggregate), e.g.
//! `SELECT SUM(x), COUNT(*), MIN(y) FROM t`: always exactly one row, also for
//! an empty input.
//!
//! Like DataFusion's `AggregateStream`, it keeps one [`Accumulator`] per
//! aggregate:
//!
//! * `sink`: every thread updates its own accumulators with the rows of its
//!   chunks (the rows that pass the `FILTER` of an aggregate, for which its
//!   arguments are evaluated);
//! * `combine`: the accumulators of every thread are merged into the global
//!   accumulators, through their states (`Accumulator::state` and
//!   `Accumulator::merge_batch`) like the states of the partitions of a
//!   `Partial` aggregation are merged by the `Final` one. The global
//!   accumulators are those of the final stage; for a single stage
//!   aggregation (`Single`), the first thread to combine contributes its
//!   accumulators instead, so that a single thread updates and evaluates
//!   exactly like DataFusion's single stage;
//! * `finalize`: the global accumulators are evaluated into the single
//!   output row, which the source role emits once. Without any input row,
//!   the accumulators of the final stage are evaluated as they are created,
//!   e.g. `COUNT` is `0` and `SUM` is `NULL`.
//!
//! An aggregation of `MIN` and `MAX` aggregates with a dynamic filter (see
//! [`super::dynamic_filter`]) shares the bounds of the accumulators of all
//! threads in its global state: a thread whose accumulators have better
//! bounds than those it knows updates them, and the filter, at most once per
//! batch of rows, as the partitions of DataFusion's `AggregateStream` do after
//! every batch.
//!
//! The memory of the accumulators (e.g. the values of an `ARRAY_AGG`) is
//! reserved in the memory pool of the session as they grow, after every
//! batch, as DataFusion's `AggregateStream` reserves it, and so are the
//! buffered rows of an aggregation that sorts its input: the aggregation fails
//! with `ResourcesExhausted` when the pool cannot hold them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::RecordBatch;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::Accumulator;
use datafusion_physical_plan::filter::batch_filter;
use parking_lot::Mutex;

use super::dynamic_filter::{SharedBounds, ThreadBounds};
use super::hash_aggregate::{AggregateSpec, InputBuffer};
use crate::chunk::DataChunk;
use crate::operator::{
    GlobalSinkState, GlobalSourceState, LocalSinkState, LocalSourceState, SinkFinalize,
    SourceResult, downcast_state, downcast_state_mut,
};
use crate::sort::{ReservedMemory, buffered_memory, resize_share};

/// The name of the operator, for error messages.
const OPERATOR: &str = "PhysicalHashAggregate";

/// The sink and source roles of an aggregation without `GROUP BY`, see the
/// [module documentation](self).
pub(crate) struct UngroupedAggregate;

/// The global sink state: the accumulators the threads combine into, and
/// the output row once finalized.
#[derive(Debug)]
struct UngroupedSinkState {
    accumulators: Mutex<Combined>,
    output: Mutex<Option<RecordBatch>>,
    /// The bounds of the `MIN` and `MAX` aggregates of all threads, if the
    /// threads update the dynamic filter of the aggregation.
    bounds: Option<SharedBounds>,
    /// The memory of the accumulators and the buffered rows of all threads,
    /// of the accumulators they combine into, and of the output row.
    reservation: MemoryReservation,
}

impl GlobalSinkState for UngroupedSinkState {}

/// The accumulators that the threads combine into.
#[derive(Debug, Default)]
struct Combined {
    accumulators: Option<Vec<Box<dyn Accumulator>>>,
    /// The memory of `accumulators`, which is reserved.
    memory: usize,
}

/// The thread-local sink state.
#[derive(Debug)]
struct UngroupedLocalState {
    accumulators: Vec<Box<dyn Accumulator>>,
    /// The memory of `accumulators`, which is reserved.
    memory: usize,
    /// The input rows, if they must be sorted before they are aggregated.
    buffer: InputBuffer,
    /// What the thread knows of the bounds of all threads, if the threads
    /// update the dynamic filter of the aggregation (see
    /// `AggregateFilter::update_bounds`).
    bounds: Option<ThreadBounds>,
}

impl LocalSinkState for UngroupedLocalState {}

impl UngroupedLocalState {
    /// Update the accumulators of the thread with the rows of `batch`, and
    /// reserve the memory that they hold in `reservation`.
    fn update(
        &mut self,
        spec: &AggregateSpec,
        batch: &RecordBatch,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        update(spec, &mut self.accumulators, batch)?;
        let memory = accumulators_memory(&self.accumulators);
        resize_share(reservation, &mut self.memory, memory)
    }

    /// Update the bounds of all threads and the dynamic filter of the
    /// aggregation `spec`, if its threads update it, with the bounds of the
    /// accumulators of the thread, which aggregated `rows` more rows (see
    /// `AggregateFilter::update_bounds`), and are `done` if the thread
    /// aggregated all of its rows.
    fn update_bounds(
        &mut self,
        spec: &AggregateSpec,
        global: &UngroupedSinkState,
        rows: usize,
        done: bool,
    ) -> Result<()> {
        if let (Some(filter), Some(shared), Some(thread)) =
            (&spec.dynamic_filter, &global.bounds, &mut self.bounds)
        {
            filter.update_bounds(&mut self.accumulators, thread, shared, rows, done)?;
        }
        Ok(())
    }
}

/// The global source state: the output row and whether it was emitted.
#[derive(Debug)]
struct UngroupedSourceState {
    output: RecordBatch,
    emitted: AtomicBool,
}

impl GlobalSourceState for UngroupedSourceState {
    fn max_threads(&self) -> usize {
        1
    }
}

impl UngroupedAggregate {
    /// The global sink state of an execution of `spec`, whose threads update
    /// its dynamic filter, if it has one, if `filters` (see
    /// `filters_enabled`), at most once per `batch_size` rows, and reserve
    /// their memory in `reservation`.
    pub(crate) fn global_sink_state(
        spec: &AggregateSpec,
        filters: bool,
        batch_size: usize,
        reservation: MemoryReservation,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let bounds = match (&spec.dynamic_filter, filters) {
            (Some(filter), true) => filter.shared_bounds(batch_size),
            _ => None,
        };
        Ok(Arc::new(UngroupedSinkState {
            accumulators: Mutex::default(),
            output: Mutex::default(),
            bounds,
            reservation,
        }))
    }

    pub(crate) fn local_sink_state(
        spec: &AggregateSpec,
        gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        let global = downcast_state::<UngroupedSinkState>(gstate.as_any(), OPERATOR)?;
        let accumulators = update_accumulators(spec)?;
        let mut memory = 0;
        resize_share(
            &global.reservation,
            &mut memory,
            accumulators_memory(&accumulators),
        )?;
        Ok(Box::new(UngroupedLocalState {
            accumulators,
            memory,
            buffer: InputBuffer::default(),
            bounds: global.bounds.as_ref().map(SharedBounds::thread_bounds),
        }))
    }

    pub(crate) fn sink(
        spec: &AggregateSpec,
        batch: RecordBatch,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let local =
            downcast_state_mut::<UngroupedLocalState>(lstate.as_any_mut(), OPERATOR)?;
        let global = downcast_state::<UngroupedSinkState>(gstate.as_any(), OPERATOR)?;
        if spec.input_order.is_some() {
            return local.buffer.push(batch, &global.reservation);
        }
        local.update(spec, &batch, &global.reservation)?;
        local.update_bounds(spec, global, batch.num_rows(), false)
    }

    pub(crate) fn combine(
        spec: &AggregateSpec,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let global = downcast_state::<UngroupedSinkState>(gstate.as_any(), OPERATOR)?;
        let local =
            downcast_state_mut::<UngroupedLocalState>(lstate.as_any_mut(), OPERATOR)?;
        if let Some(order) = &spec.input_order {
            // The buffered and the sorted rows, freed once they are aggregated.
            let mut sorting = ReservedMemory::new(&global.reservation);
            let sorted =
                local
                    .buffer
                    .take_sorted(&spec.input_schema, order, &mut sorting)?;
            for batch in sorted {
                local.update(spec, &batch, &global.reservation)?;
            }
        }
        // The last bounds of the thread, for the threads that are still
        // aggregating.
        local.update_bounds(spec, global, 0, true)?;
        let mut combined = global.accumulators.lock();
        let Combined {
            accumulators,
            memory,
        } = &mut *combined;
        if accumulators.is_none() {
            if spec.single_stage {
                // The accumulators of the thread, with their memory.
                *accumulators = Some(std::mem::take(&mut local.accumulators));
                *memory = std::mem::take(&mut local.memory);
                return Ok(());
            }
            let merged = merge_accumulators(spec)?;
            resize_share(&global.reservation, memory, accumulators_memory(&merged))?;
            *accumulators = Some(merged);
        }
        let Some(accumulators) = accumulators.as_mut() else {
            return internal_err!("{OPERATOR} has no accumulators to combine into");
        };
        merge(accumulators, &mut local.accumulators)?;
        resize_share(
            &global.reservation,
            memory,
            accumulators_memory(accumulators),
        )?;
        // The accumulators of the thread are dropped with its state.
        resize_share(&global.reservation, &mut local.memory, 0)
    }

    pub(crate) fn finalize(
        spec: &AggregateSpec,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let global = downcast_state::<UngroupedSinkState>(gstate.as_any(), OPERATOR)?;
        let Combined {
            accumulators,
            mut memory,
        } = std::mem::take(&mut *global.accumulators.lock());
        let mut accumulators = match accumulators {
            Some(accumulators) => accumulators,
            None => merge_accumulators(spec)?,
        };
        let columns = accumulators
            .iter_mut()
            .map(|accumulator| accumulator.evaluate()?.to_array())
            .collect::<Result<Vec<_>>>()?;
        let output = RecordBatch::try_new(Arc::clone(&spec.schema), columns)?;
        // The output row replaces the accumulators, which are dropped.
        drop(accumulators);
        resize_share(&global.reservation, &mut memory, buffered_memory(&output))?;
        *global.output.lock() = Some(output);
        Ok(SinkFinalize::Ready)
    }

    pub(crate) fn global_source_state(
        sink_state: &dyn GlobalSinkState,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let global = downcast_state::<UngroupedSinkState>(sink_state.as_any(), OPERATOR)?;
        let Some(output) = global.output.lock().clone() else {
            return internal_err!("{OPERATOR} is a source before it was finalized");
        };
        Ok(Arc::new(UngroupedSourceState {
            output,
            emitted: AtomicBool::new(false),
        }))
    }

    pub(crate) fn local_source_state() -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(crate::operator::EmptyState))
    }

    pub(crate) fn get_data(gstate: &dyn GlobalSourceState) -> Result<SourceResult> {
        let state = downcast_state::<UngroupedSourceState>(gstate.as_any(), OPERATOR)?;
        if state.emitted.swap(true, Ordering::Relaxed) {
            return Ok(SourceResult::Finished);
        }
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(
            state.output.clone(),
        )))
    }
}

/// The accumulators that consume the input rows.
fn update_accumulators(spec: &AggregateSpec) -> Result<Vec<Box<dyn Accumulator>>> {
    spec.update
        .iter()
        .map(|update| update.expr.create_accumulator())
        .collect()
}

/// The accumulators of the final stage, which merge the states of the
/// threads.
fn merge_accumulators(spec: &AggregateSpec) -> Result<Vec<Box<dyn Accumulator>>> {
    spec.merge
        .iter()
        .map(|expr| expr.create_accumulator())
        .collect()
}

/// Update `accumulators` with the rows of `batch`, like DataFusion's
/// `AggregateStream`: every aggregate is updated with its arguments for the
/// rows that pass its filter.
fn update(
    spec: &AggregateSpec,
    accumulators: &mut [Box<dyn Accumulator>],
    batch: &RecordBatch,
) -> Result<()> {
    for (accumulator, update) in accumulators.iter_mut().zip(&spec.update) {
        let values = match &update.filter {
            Some(filter) => update.evaluate_arguments(&batch_filter(batch, filter)?)?,
            None => update.evaluate_arguments(batch)?,
        };
        accumulator.update_batch(&values)?;
    }
    Ok(())
}

/// The memory that `accumulators` hold.
fn accumulators_memory(accumulators: &[Box<dyn Accumulator>]) -> usize {
    accumulators
        .iter()
        .map(|accumulator| accumulator.size())
        .sum()
}

/// Merge the states of the accumulators `local` into `global`.
fn merge(
    global: &mut [Box<dyn Accumulator>],
    local: &mut [Box<dyn Accumulator>],
) -> Result<()> {
    for (global, local) in global.iter_mut().zip(local.iter_mut()) {
        let state = local
            .state()?
            .iter()
            .map(|value| value.to_array())
            .collect::<Result<Vec<_>>>()?;
        global.merge_batch(&state)?;
    }
    Ok(())
}
