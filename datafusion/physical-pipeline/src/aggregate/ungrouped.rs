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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::RecordBatch;
use datafusion_common::{Result, internal_err};
use datafusion_expr::Accumulator;
use datafusion_physical_plan::filter::batch_filter;
use parking_lot::Mutex;

use super::hash_aggregate::{AggregateSpec, InputBuffer};
use crate::chunk::DataChunk;
use crate::operator::{
    GlobalSinkState, GlobalSourceState, LocalSinkState, LocalSourceState, SinkFinalize,
    SourceResult, downcast_state, downcast_state_mut,
};

/// The name of the operator, for error messages.
const OPERATOR: &str = "PhysicalHashAggregate";

/// The sink and source roles of an aggregation without `GROUP BY`, see the
/// [module documentation](self).
pub(crate) struct UngroupedAggregate;

/// The global sink state: the accumulators the threads combine into, and
/// the output row once finalized.
#[derive(Debug, Default)]
struct UngroupedSinkState {
    accumulators: Mutex<Option<Vec<Box<dyn Accumulator>>>>,
    output: Mutex<Option<RecordBatch>>,
}

impl GlobalSinkState for UngroupedSinkState {}

/// The thread-local sink state.
#[derive(Debug)]
struct UngroupedLocalState {
    accumulators: Vec<Box<dyn Accumulator>>,
    /// The input rows, if they must be sorted before they are aggregated.
    buffer: InputBuffer,
}

impl LocalSinkState for UngroupedLocalState {}

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
    pub(crate) fn global_sink_state() -> Result<Arc<dyn GlobalSinkState>> {
        Ok(Arc::new(UngroupedSinkState::default()))
    }

    pub(crate) fn local_sink_state(
        spec: &AggregateSpec,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(UngroupedLocalState {
            accumulators: update_accumulators(spec)?,
            buffer: InputBuffer::default(),
        }))
    }

    pub(crate) fn sink(
        spec: &AggregateSpec,
        batch: RecordBatch,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let local =
            downcast_state_mut::<UngroupedLocalState>(lstate.as_any_mut(), OPERATOR)?;
        if spec.input_order.is_some() {
            local.buffer.push(batch);
            return Ok(());
        }
        update(spec, &mut local.accumulators, &batch)
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
            for batch in local.buffer.take_sorted(&spec.input_schema, order)? {
                update(spec, &mut local.accumulators, &batch)?;
            }
        }
        let mut accumulators = global.accumulators.lock();
        if accumulators.is_none() {
            if spec.single_stage {
                *accumulators = Some(std::mem::take(&mut local.accumulators));
                return Ok(());
            }
            *accumulators = Some(merge_accumulators(spec)?);
        }
        match accumulators.as_mut() {
            Some(accumulators) => merge(accumulators, &mut local.accumulators),
            None => internal_err!("{OPERATOR} has no accumulators to combine into"),
        }
    }

    pub(crate) fn finalize(
        spec: &AggregateSpec,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let global = downcast_state::<UngroupedSinkState>(gstate.as_any(), OPERATOR)?;
        let mut accumulators = match global.accumulators.lock().take() {
            Some(accumulators) => accumulators,
            None => merge_accumulators(spec)?,
        };
        let columns = accumulators
            .iter_mut()
            .map(|accumulator| accumulator.evaluate()?.to_array())
            .collect::<Result<Vec<_>>>()?;
        let output = RecordBatch::try_new(Arc::clone(&spec.schema), columns)?;
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
