// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalTopK`]: a parallel top-k, the pipeline breaker of
//! `ORDER BY ... LIMIT k`.
//!
//! Like DuckDB's `PhysicalTopN`, every thread keeps a bounded top-k of the
//! rows it sees, and the top-ks of all threads are merged once they are done:
//!
//! * [`sink`](PhysicalOperator::sink): every thread keeps the best `k` rows
//!   it saw so far as a [`SortedRun`] in its local sink state. Once it has
//!   `k` rows, the key of its `k`-th row is a *threshold*: a row that does not
//!   come before it cannot be in the result, as the thread has `k` rows that
//!   come before or equal it. Every chunk is first filtered with the
//!   threshold predicate (`key < threshold`, as DataFusion's TopK builds it),
//!   so rows behind the threshold are discarded right away. The rows that
//!   pass are buffered; once the buffer holds at least `k` rows (and
//!   [`MIN_TOPK_BUFFER_ROWS`]), or at least [`TOPK_BUFFER_BYTES`] and as many
//!   bytes as the run, the thread sorts its run and the buffer and keeps the
//!   first `k` rows, which gives a tighter threshold. The cost per buffered
//!   row is `O(log k)`, also for large `k`.
//! * The tightest threshold of any thread is shared with the other threads
//!   through the global sink state, so that every thread filters with the
//!   tightest threshold known. If the `SortExec` that the operator replaces
//!   has a [`DynamicFilterPhysicalExpr`] (which DataFusion pushes into the
//!   scans below the TopK), the operator updates it with the predicate of
//!   every tighter threshold, so that the scans skip files, row groups and
//!   rows that cannot be in the result, as for DataFusion's TopK.
//! * [`combine`](PhysicalOperator::combine): the thread reduces its buffer
//!   into its run and hands the run to the global sink state.
//! * [`finalize`](PhysicalOperator::finalize): a [`RunMerger`] is set up over
//!   the runs of the threads (at most `k` rows per thread), keeping the first
//!   `k` rows; the dynamic filter is marked complete.
//! * [`get_data`](PhysicalOperator::get_data): the merger emits the result,
//!   in order, on a single thread.
//!
//! The rows that the threshold discards are never sorted, so their sort keys
//! are only evaluated by the threshold predicate, and the rows that the
//! dynamic filter prunes are not even read: which rows the TopK (and the
//! scans below it) evaluate depends on when its threshold tightens.
//!
//! The TopK keeps the first `k` rows in the order of its sort keys, in which
//! floats compare in their total order (`-0.0` before `0.0`, `NaN` last).
//! Which of the rows that tie with its last row it keeps, and in which order
//! it emits rows with equal keys, is its own.
//!
//! The buffered rows and the top-ks of the threads are reserved in the memory
//! pool of the session, and what reducing them allocates before it allocates
//! it (see [`SortKeys::sort`]): the TopK fails with `ResourcesExhausted` when
//! the pool cannot hold them.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::RecordBatch;
use arrow::compute::{SortOptions, filter_record_batch};
use arrow::datatypes::SchemaRef;
use arrow_row::OwnedRow;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::{ColumnarValue, Operator};
use datafusion_physical_expr::expressions::{
    BinaryExpr, DynamicFilterPhysicalExpr, is_not_null, is_null, lit,
};
use datafusion_physical_expr::{LexOrdering, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlan;
use parking_lot::Mutex;

use super::merge::{RunMerger, SortedRun};
use super::{
    SortKeys, SortPlan, SortedOutput, buffered_memory, memory_size, next_sorted_chunk,
};
use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
    downcast_state, downcast_state_mut,
};

/// A thread reduces its buffered rows to its top `k` rows once it buffered
/// `k` rows, but not before it buffered this many rows.
pub const MIN_TOPK_BUFFER_ROWS: usize = 1024;

/// A thread also reduces its buffered rows once they occupy about this many
/// bytes (and as many as its top `k` rows), so that large rows do not pile
/// up: e.g. 1024 rows of 2 MiB hold more `Utf8` values than an Arrow array
/// addresses.
pub const TOPK_BUFFER_BYTES: usize = 64 * 1024 * 1024;

/// The [`DynamicFilterPhysicalExpr`] of a `SortExec` with a fetch, which a
/// [`PhysicalTopK`] updates, and the expressions of the filter's input (the
/// sort keys of the `SortExec`) that the leading sort keys of the TopK
/// correspond to.
#[derive(Debug, Clone)]
pub struct TopKDynamicFilter {
    filter: Arc<DynamicFilterPhysicalExpr>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
}

impl TopKDynamicFilter {
    /// Update `filter` with predicates on `keys`, the expressions that the
    /// first `keys.len()` sort keys of the TopK correspond to in the input of
    /// the filter. They must be children of the filter.
    pub fn new(
        filter: Arc<DynamicFilterPhysicalExpr>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self { filter, keys }
    }

    /// The dynamic filter.
    pub fn filter(&self) -> &Arc<DynamicFilterPhysicalExpr> {
        &self.filter
    }

    /// The expressions the leading sort keys of the TopK correspond to.
    pub fn keys(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.keys
    }
}

/// Formerly DataFusion's TopK, which a [`PhysicalTopK`] by float keys
/// executed again instead of emitting its own result, if the last row of its
/// result had a float zero (DataFusion's threshold may keep other rows among
/// the zeros, which tie in SQL).
///
/// The native TopK no longer does: the rows that tie with its last row, also
/// zeros of either sign, are its own to choose. The type has no values, and
/// [`PhysicalTopK::float_key_fallback`] is always `None`; both remain only
/// for the callers that still ask for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatKeyFallback {}

/// Keeps the first `k` rows of its input in the order of its sort keys, see
/// the [module documentation](self).
#[derive(Debug)]
pub struct PhysicalTopK {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    keys: SortKeys,
    k: usize,
    dynamic_filter: Option<TopKDynamicFilter>,
}

impl PhysicalTopK {
    /// Keep the first `k` rows of `input` in the order `ordering`. `k` must
    /// not be zero.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        ordering: LexOrdering,
        k: usize,
    ) -> Result<Self> {
        if k == 0 {
            return internal_err!("PhysicalTopK requires k > 0");
        }
        let schema = input.schema();
        let keys = SortKeys::try_new(ordering, &schema)?;
        Ok(Self {
            children: vec![input],
            schema,
            keys,
            k,
            dynamic_filter: None,
        })
    }

    /// Update `dynamic_filter` with the threshold of the TopK as it gets
    /// tighter. It must not have more keys than the TopK.
    pub fn with_dynamic_filter(
        mut self,
        dynamic_filter: TopKDynamicFilter,
    ) -> Result<Self> {
        if dynamic_filter.keys.len() > self.keys.ordering().len() {
            return internal_err!(
                "The dynamic filter of a PhysicalTopK with {} sort keys has {} keys",
                self.keys.ordering().len(),
                dynamic_filter.keys.len()
            );
        }
        self.dynamic_filter = Some(dynamic_filter);
        Ok(self)
    }

    /// Lower a sort of a DataFusion plan that keeps only its first rows: see
    /// [`try_lower`](super::try_lower), which lowers the other sorts to a
    /// [`PhysicalSort`](super::PhysicalSort). Returns `Ok(None)` for those and
    /// any other node, without calling `lower_input`.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        options: &ConfigOptions,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        match SortPlan::try_new(plan, options)? {
            Some(sort) if sort.is_topk() => sort.into_topk(plan, lower_input).map(Some),
            _ => Ok(None),
        }
    }

    /// The input of the TopK.
    pub fn input(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The sort keys.
    pub fn ordering(&self) -> &LexOrdering {
        self.keys.ordering()
    }

    /// The number of rows the TopK emits.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The dynamic filter the TopK updates, if any.
    pub fn dynamic_filter(&self) -> Option<&TopKDynamicFilter> {
        self.dynamic_filter.as_ref()
    }

    /// Always `None`: the TopK never executes DataFusion's TopK instead of
    /// emitting its own result (see [`FloatKeyFallback`]).
    pub fn float_key_fallback(&self) -> Option<&FloatKeyFallback> {
        None
    }

    /// The number of buffered rows at which a thread reduces its rows.
    fn reduce_rows(&self) -> usize {
        self.k.max(MIN_TOPK_BUFFER_ROWS)
    }

    /// Adopt the threshold of the global state if it is tighter than the
    /// threshold of the thread.
    fn refresh_threshold(global: &TopKSinkState, state: &mut TopKLocalState) {
        let generation = global.generation.load(Ordering::Acquire);
        if generation == state.generation {
            return;
        }
        state.generation = generation;
        let current = global.threshold.lock().as_ref().map(Arc::clone);
        if let Some(current) = current
            && state
                .threshold
                .as_ref()
                .is_none_or(|threshold| current.key < threshold.key)
        {
            state.threshold = Some(current);
        }
    }

    /// Sort the runs and the buffered rows of `state` and keep the first `k`
    /// rows. If that gives a tighter threshold, publish it. The memory of the
    /// rows that are kept is reserved in `global` instead of the memory of
    /// the runs and the buffered rows.
    fn reduce(&self, global: &TopKSinkState, state: &mut TopKLocalState) -> Result<()> {
        if state.buffer.is_empty() {
            return Ok(());
        }
        let mut batches = Vec::with_capacity(state.buffer.len() + state.top.len());
        let reserved = state.buffered_memory
            + state.top.iter().map(SortedRun::memory_size).sum::<usize>();
        batches.extend(state.top.drain(..).map(SortedRun::into_batch));
        batches.append(&mut state.buffer);
        state.buffered_rows = 0;
        state.buffered_bytes = 0;
        state.buffered_memory = 0;
        let top =
            self.keys
                .sort(&self.schema, &batches, Some(self.k), &global.reservation)?;
        drop(batches);
        global.reservation.shrink(reserved);
        state.top = top;
        state.top_bytes = state.top.iter().map(|run| memory_size(run.batch())).sum();

        // The `k`-th row is the last row of the last run.
        let rows = state.top.iter().map(SortedRun::num_rows).sum::<usize>();
        let Some(last) = state.top.last().filter(|_| rows == self.k) else {
            return Ok(());
        };
        let row = last.num_rows() - 1;
        let key = self.keys.encoded_key(last, row)?;
        if state
            .threshold
            .as_ref()
            .is_some_and(|threshold| threshold.key <= key)
        {
            return Ok(());
        }
        let threshold = Arc::new(self.threshold(last, row, key)?);
        state.threshold = Some(Arc::clone(&threshold));
        self.publish(global, &threshold)
    }

    /// The threshold of a top-k whose `k`-th row is row `row` of `run`, whose
    /// encoded sort key is `key`.
    fn threshold(&self, run: &SortedRun, row: usize, key: OwnedRow) -> Result<Threshold> {
        let batch = run.batch().slice(row, 1);
        let values = self
            .ordering()
            .iter()
            .map(|sort_expr| match sort_expr.expr.evaluate(&batch)? {
                ColumnarValue::Scalar(value) => Ok(value),
                ColumnarValue::Array(array) => ScalarValue::try_from_array(&array, 0),
            })
            .collect::<Result<Vec<_>>>()?;
        let keys = self
            .ordering()
            .iter()
            .map(|sort_expr| (Arc::clone(&sort_expr.expr), sort_expr.options))
            .collect::<Vec<_>>();
        let predicate = threshold_predicate(&keys, &values)?;
        Ok(Threshold {
            key,
            values,
            predicate,
        })
    }

    /// Make `threshold` the threshold of all threads if it is tighter than the
    /// current one, and update the dynamic filter with it.
    fn publish(&self, global: &TopKSinkState, threshold: &Arc<Threshold>) -> Result<()> {
        let mut current = global.threshold.lock();
        if current
            .as_ref()
            .is_some_and(|current| current.key <= threshold.key)
        {
            return Ok(());
        }
        if let Some(dynamic_filter) = &self.dynamic_filter {
            let keys = dynamic_filter
                .keys
                .iter()
                .zip(self.ordering().iter())
                .map(|(expr, sort_expr)| (Arc::clone(expr), sort_expr.options))
                .collect::<Vec<_>>();
            if let Some(predicate) = threshold_predicate(&keys, &threshold.values)? {
                dynamic_filter.filter.update(predicate)?;
            }
        }
        *current = Some(Arc::clone(threshold));
        global.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

/// The predicate that selects the rows that come before a threshold row whose
/// sort keys have the values `values`, like DataFusion's TopK builds it: for
/// keys `a`, `b` with values `x`, `y`, `a < x OR (a = x AND b < y)` (with `>`
/// for descending keys, and nulls ordered as the keys order them).
///
/// `keys` are the expressions and options of the leading keys whose values
/// the predicate compares. A key without an expression, or with a value of a
/// nested type (whose comparison orders nulls differently than the sort), is
/// not compared: the predicate then also selects the rows whose earlier keys
/// equal the values, so that it never discards a row that comes before the
/// threshold row. Returns `None` if it cannot discard any row.
///
/// The sort orders floats in their total order, `-0.0` before `0.0`, while
/// comparisons may consider both zeros equal. A float key whose value is a
/// zero is therefore compared with `<=` (`>=`), which selects both zeros, so
/// that the predicate never discards a `-0.0` that comes before a `0.0`
/// threshold row. (The predicate selects all rows that come before the
/// threshold row, and some others, which the TopK sorts out.)
pub(crate) fn threshold_predicate(
    keys: &[(Arc<dyn PhysicalExpr>, SortOptions)],
    values: &[ScalarValue],
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    let mut filters: Vec<Arc<dyn PhysicalExpr>> = vec![];
    // Rows whose keys so far equal the values.
    let mut equal: Option<Arc<dyn PhysicalExpr>> = None;
    for (index, value) in values.iter().enumerate() {
        let Some((expr, options)) =
            keys.get(index).filter(|_| !value.data_type().is_nested())
        else {
            filters.extend(equal.take());
            break;
        };
        let op = match (options.descending, is_float_zero(value)) {
            (false, false) => Operator::Lt,
            (false, true) => Operator::LtEq,
            (true, false) => Operator::Gt,
            (true, true) => Operator::GtEq,
        };
        let comparison: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(Arc::clone(expr), op, lit(value.clone())));
        let before: Arc<dyn PhysicalExpr> = match (options.nulls_first, value.is_null()) {
            // Nothing comes before a null that comes first.
            (true, true) => lit(false),
            (true, false) => Arc::new(BinaryExpr::new(
                is_null(Arc::clone(expr))?,
                Operator::Or,
                comparison,
            )),
            // Every value comes before a null that comes last.
            (false, true) => is_not_null(Arc::clone(expr))?,
            (false, false) => comparison,
        };
        let mut equal_value: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(expr),
            Operator::Eq,
            lit(value.clone()),
        ));
        if value.is_null() {
            equal_value = Arc::new(BinaryExpr::new(
                is_null(Arc::clone(expr))?,
                Operator::Or,
                equal_value,
            ));
        }
        match equal.take() {
            None => {
                filters.push(before);
                equal = Some(equal_value);
            }
            Some(previous) => {
                filters.push(Arc::new(BinaryExpr::new(
                    Arc::clone(&previous),
                    Operator::And,
                    before,
                )));
                equal = Some(Arc::new(BinaryExpr::new(
                    previous,
                    Operator::And,
                    equal_value,
                )));
            }
        }
    }
    Ok(filters.into_iter().reduce(|a, b| -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(a, Operator::Or, b))
    }))
}

/// Whether `value` is a float `0.0` or `-0.0` (possibly dictionary or run-end
/// encoded).
fn is_float_zero(value: &ScalarValue) -> bool {
    match value {
        ScalarValue::Float16(Some(v)) => v.to_f32() == 0.0,
        ScalarValue::Float32(Some(v)) => *v == 0.0,
        ScalarValue::Float64(Some(v)) => *v == 0.0,
        ScalarValue::Dictionary(_, value) | ScalarValue::RunEndEncoded(_, _, value) => {
            is_float_zero(value)
        }
        _ => false,
    }
}

/// The `k`-th row of a top-k: rows that do not come before it cannot be in
/// the result.
#[derive(Debug)]
struct Threshold {
    /// The encoded sort key of the row.
    key: OwnedRow,
    /// The values of the sort keys of the row.
    values: Vec<ScalarValue>,
    /// Selects the rows of an input chunk that come before the row, `None` if
    /// it cannot discard any row.
    predicate: Option<Arc<dyn PhysicalExpr>>,
}

/// The global sink state of a [`PhysicalTopK`].
#[derive(Debug)]
struct TopKSinkState {
    /// The top-ks of the threads that completed.
    runs: Mutex<Vec<SortedRun>>,
    /// The tightest threshold of any thread.
    threshold: Mutex<Option<Arc<Threshold>>>,
    /// Incremented whenever `threshold` gets tighter, so that the threads
    /// notice a new threshold with an atomic load.
    generation: AtomicUsize,
    output: SortedOutput,
    /// The memory of the rows that all threads buffered and of their
    /// top-ks.
    reservation: MemoryReservation,
}

impl GlobalSinkState for TopKSinkState {}

/// The local sink state of a [`PhysicalTopK`]: the top-k of the thread.
#[derive(Debug, Default)]
struct TopKLocalState {
    /// The first (at most) `k` rows of the thread, in order: consecutive runs
    /// (usually a single one, see [`SortKeys::sort`]).
    top: Vec<SortedRun>,
    /// The estimated memory size of `top`.
    top_bytes: usize,
    /// Compacted rows that came before the threshold and were not reduced
    /// into `top` yet.
    buffer: Vec<RecordBatch>,
    /// The number of rows of `buffer`.
    buffered_rows: usize,
    /// The estimated memory size of `buffer`.
    buffered_bytes: usize,
    /// The memory of `buffer` (see [`buffered_memory`]), which is reserved.
    buffered_memory: usize,
    /// The tightest threshold the thread knows.
    threshold: Option<Arc<Threshold>>,
    /// The generation of the global threshold the thread knows.
    generation: usize,
}

impl LocalSinkState for TopKLocalState {}

impl PhysicalOperator for PhysicalTopK {
    fn name(&self) -> &str {
        "PhysicalTopK"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let mut description =
            format!("PhysicalTopK: fetch={}, expr=[{}]", self.k, self.ordering());
        if self.dynamic_filter.is_some() {
            description.push_str(", dynamic_filter=true");
        }
        description
    }

    // ---------------------------------------------------------------------
    // Sink
    // ---------------------------------------------------------------------

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let reservation =
            MemoryConsumer::new(self.name()).register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(TopKSinkState {
            runs: Mutex::default(),
            threshold: Mutex::default(),
            generation: AtomicUsize::new(0),
            output: SortedOutput::default(),
            reservation,
        }))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(TopKLocalState::default()))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let global = downcast_state::<TopKSinkState>(gstate.as_any(), self.name())?;
        let state =
            downcast_state_mut::<TopKLocalState>(lstate.as_any_mut(), self.name())?;
        Self::refresh_threshold(global, state);
        let mut batch = input.compact()?;
        if let Some(predicate) = state
            .threshold
            .as_ref()
            .and_then(|threshold| threshold.predicate.as_ref())
        {
            let selected = predicate.evaluate(&batch)?.into_array(batch.num_rows())?;
            let selected = as_boolean_array(&selected)?;
            match selected.true_count() {
                0 => return Ok(SinkResult::NeedMoreInput),
                count if count < batch.num_rows() => {
                    batch = filter_record_batch(&batch, selected)?;
                }
                _ => {}
            }
        }
        let memory = buffered_memory(&batch);
        global.reservation.try_grow(memory)?;
        state.buffered_memory += memory;
        state.buffered_rows += batch.num_rows();
        state.buffered_bytes += memory_size(&batch);
        state.buffer.push(batch);
        if state.buffered_rows >= self.reduce_rows()
            || state.buffered_bytes >= TOPK_BUFFER_BYTES.max(state.top_bytes)
        {
            self.reduce(global, state)?;
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let global = downcast_state::<TopKSinkState>(gstate.as_any(), self.name())?;
        let state =
            downcast_state_mut::<TopKLocalState>(lstate.as_any_mut(), self.name())?;
        self.reduce(global, state)?;
        global.runs.lock().append(&mut state.top);
        Ok(())
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let global = downcast_state::<TopKSinkState>(gstate.as_any(), self.name())?;
        let runs = std::mem::take(&mut *global.runs.lock());
        let merger = RunMerger::new(Arc::clone(&self.schema), runs, Some(self.k));
        if let Some(dynamic_filter) = &self.dynamic_filter {
            dynamic_filter.filter.mark_complete();
        }
        let rows = merger.remaining_rows();
        global.output.set(merger, &global.reservation);
        Ok(if rows == 0 {
            SinkFinalize::NoOutputPossible
        } else {
            SinkFinalize::Ready
        })
    }

    // ---------------------------------------------------------------------
    // Source
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        // A single thread emits the rows, in order.
        false
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let Some(sink) = ctx.global_sink_state(self) else {
            return internal_err!(
                "{} is read before it collected its input",
                self.name()
            );
        };
        let global = downcast_state::<TopKSinkState>(sink.as_any(), self.name())?;
        global.output.take_source(self.name())
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        next_sorted_chunk(gstate, self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cmp::Ordering as CmpOrdering;

    use arrow::array::{
        Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int32Array, Int64Array,
        StringArray,
    };
    use arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
    use arrow_row::{RowConverter, SortField};
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::col;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::chunk::{DEFAULT_VECTOR_SIZE, SelectionVector};
    use crate::explain::format_pipelines;
    use crate::operators::union::test_support::{
        TestSource, run, values, values_chunk, values_chunks, values_schema,
    };

    fn ordering(schema: &SchemaRef, descending: bool) -> LexOrdering {
        let mut sort_expr = PhysicalSortExpr::new_default(col("v", schema).unwrap());
        if descending {
            sort_expr = sort_expr.desc();
        }
        LexOrdering::new([sort_expr]).unwrap()
    }

    fn topk(
        input: Arc<dyn PhysicalOperator>,
        descending: bool,
        k: usize,
    ) -> PhysicalTopK {
        let ordering = ordering(&input.schema(), descending);
        PhysicalTopK::try_new(input, ordering, k).unwrap()
    }

    /// Chunks of the values `0..n` in a scrambled order, of `size` rows each.
    fn scrambled_chunks(n: i64, size: usize) -> Vec<DataChunk> {
        let scrambled = (0..n).map(|i| (i * 7919) % n).collect::<Vec<_>>();
        scrambled
            .chunks(size)
            .map(|values| values_chunk(values.iter().copied()))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_reserve_their_rows_in_the_memory_pool() -> Result<()> {
        use crate::executor::Executor;
        use crate::operators::lowering_tests::{
            assert_exhausted, task_context_with_memory_limit,
        };
        use crate::sort::test_support::assert_memory_freed;
        // The first 50,000 of 100,000 rows of an `Int64` column: 400 KB.
        let n = 100_000;
        let root = || {
            let input = Arc::new(TestSource::new(scrambled_chunks(n, 2048)));
            Arc::new(topk(input, false, 50_000))
        };
        // 256 KiB do not hold the rows that the TopK keeps: it fails.
        let ctx = task_context_with_memory_limit(256 << 10);
        let result = Executor::collect(root(), Arc::clone(&ctx), 4).await;
        let error = result.as_ref().map(|_| ()).unwrap_err().to_string();
        assert!(error.contains("PhysicalTopK"), "{error}");
        assert_exhausted(result);
        assert_memory_freed(ctx.memory_pool().as_ref()).await;
        // 16 MiB do: the TopK returns its rows, and frees their memory.
        let ctx = task_context_with_memory_limit(16 << 20);
        let output = Executor::collect(root(), Arc::clone(&ctx), 4).await?;
        assert_eq!(values(&output), (0..50_000).collect::<Vec<_>>());
        assert_eq!(ctx.memory_pool().reserved(), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topk_keeps_the_first_k_rows() -> Result<()> {
        let n = 10_007;
        for k in [
            1,
            2,
            7,
            1023,
            1024,
            2048,
            2049,
            5000,
            10_006,
            10_007,
            10_008,
            usize::MAX,
        ] {
            for (threads, descending) in [(1, false), (4, false), (4, true)] {
                let input = Arc::new(TestSource::new(scrambled_chunks(n, 1000)));
                let output = run(Arc::new(topk(input, descending, k)), threads).await?;
                let mut expected = (0..n).collect::<Vec<_>>();
                if descending {
                    expected.reverse();
                }
                expected.truncate(k);
                assert_eq!(values(&output), expected, "k = {k}, {threads} threads");
                assert!(
                    output
                        .iter()
                        .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
                );
                assert!(output.iter().all(|batch| batch.schema() == values_schema()));
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topk_of_selected_chunks_empty_and_single_row_inputs() -> Result<()> {
        // The even values of 0..100 and the values 100..200 in reverse order.
        let evens = values_chunk(0..100).with_selection(SelectionVector::from(
            (0..100).step_by(2).collect::<Vec<u32>>(),
        ))?;
        let reversed = values_chunk(100..200).with_selection(SelectionVector::from(
            (0..100).rev().collect::<Vec<u32>>(),
        ))?;
        let input = Arc::new(TestSource::new(vec![reversed, evens]));
        let output = run(Arc::new(topk(input, false, 60)), 4).await?;
        let expected = (0..50).map(|i| i * 2).chain(100..110).collect::<Vec<_>>();
        assert_eq!(values(&output), expected);

        let empty = Arc::new(TestSource::with_schema(values_schema(), vec![]));
        assert!(run(Arc::new(topk(empty, false, 3)), 4).await?.is_empty());
        let single = Arc::new(TestSource::new(vec![values_chunk([42])]));
        let output = run(Arc::new(topk(single, true, 3)), 4).await?;
        assert_eq!(values(&output), vec![42]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topk_of_several_keys_with_nulls_and_ties() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]));
        let chunk = |values: std::ops::Range<i64>| {
            let k = Int64Array::from_iter(
                values.clone().map(|v| (v % 5 != 0).then_some(v % 3)),
            );
            let s = StringArray::from_iter(
                values
                    .clone()
                    .map(|v| (v % 7 != 0).then(|| format!("s{}", v % 4))),
            );
            let v = Int64Array::from_iter_values(values);
            DataChunk::from_record_batch(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(k), Arc::new(s), Arc::new(v)],
                )
                .unwrap(),
            )
        };
        let rows = (0..3000_i64)
            .map(|v| {
                (
                    (v % 5 != 0).then_some(v % 3),
                    (v % 7 != 0).then(|| format!("s{}", v % 4)),
                    v,
                )
            })
            .collect::<Vec<_>>();
        // k DESC NULLS FIRST, s ASC NULLS LAST, with ties: only the keys are
        // compared.
        let compare = |a: &(Option<i64>, Option<String>, i64),
                       b: &(Option<i64>, Option<String>, i64)| {
            let k = match (a.0, b.0) {
                (None, None) => CmpOrdering::Equal,
                (None, Some(_)) => CmpOrdering::Less,
                (Some(_), None) => CmpOrdering::Greater,
                (Some(x), Some(y)) => y.cmp(&x),
            };
            let s = match (&a.1, &b.1) {
                (None, None) => CmpOrdering::Equal,
                (None, Some(_)) => CmpOrdering::Greater,
                (Some(_), None) => CmpOrdering::Less,
                (Some(x), Some(y)) => x.cmp(y),
            };
            k.then(s)
        };
        let mut expected = rows.clone();
        expected.sort_by(compare);
        for k in [1, 10, 333, 1500, 2999, 3000, 4000] {
            for threads in [1, 4] {
                let chunks = (0..30).map(|i| chunk(i * 100..i * 100 + 100)).collect();
                let input: Arc<dyn PhysicalOperator> =
                    Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks));
                let ordering = LexOrdering::new([
                    PhysicalSortExpr::new_default(col("k", &schema)?)
                        .desc()
                        .nulls_first(),
                    PhysicalSortExpr::new_default(col("s", &schema)?)
                        .asc()
                        .nulls_last(),
                ])
                .unwrap();
                let root = PhysicalTopK::try_new(input, ordering, k)?;
                let output = run(Arc::new(root), threads).await?;
                let actual = output
                    .iter()
                    .flat_map(|batch| {
                        let k = batch.column(0).as_primitive::<Int64Type>();
                        let s = batch.column(1).as_string::<i32>();
                        let v = batch.column(2).as_primitive::<Int64Type>();
                        (0..batch.num_rows())
                            .map(|i| {
                                (
                                    k.is_valid(i).then(|| k.value(i)),
                                    s.is_valid(i).then(|| s.value(i).to_string()),
                                    v.value(i),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let len = k.min(rows.len());
                assert_eq!(actual.len(), len, "k = {k}");
                // The keys are those of the first `k` rows, and every row is
                // a row of the input.
                let keys = |rows: &[(Option<i64>, Option<String>, i64)]| {
                    rows.iter()
                        .map(|row| (row.0, row.1.clone()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(keys(&actual), keys(&expected[..len]), "k = {k}");
                assert!(actual.iter().all(|row| rows[row.2 as usize] == *row));
                let mut ids = actual.iter().map(|row| row.2).collect::<Vec<_>>();
                ids.sort_unstable();
                ids.dedup();
                assert_eq!(ids.len(), len, "k = {k}: duplicate rows");
            }
        }
        Ok(())
    }

    /// The order of the rows of `columns` by `keys` (with options), in the
    /// row format.
    fn encoded(columns: &[ArrayRef], options: &[SortOptions]) -> arrow_row::Rows {
        let fields = columns
            .iter()
            .zip(options)
            .map(|(column, options)| {
                SortField::new_with_options(column.data_type().clone(), *options)
            })
            .collect();
        RowConverter::new(fields)
            .unwrap()
            .convert_columns(columns)
            .unwrap()
    }

    #[test]
    fn threshold_predicate_selects_the_rows_before_the_threshold() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 300;
        let int = |rng: &mut StdRng| -> ArrayRef {
            Arc::new(Int32Array::from_iter((0..n).map(|_| {
                (rng.random_range(0..8) != 0).then(|| rng.random_range(-3..3))
            })))
        };
        let float = |rng: &mut StdRng| -> ArrayRef {
            Arc::new(Float64Array::from_iter((0..n).map(
                |_| match rng.random_range(0..9) {
                    0 => None,
                    1 => Some(f64::NAN),
                    2 => Some(-0.0),
                    3 => Some(0.0),
                    4 => Some(f64::INFINITY),
                    _ => Some(f64::from(rng.random_range(-2..2))),
                },
            )))
        };
        let string = |rng: &mut StdRng| -> ArrayRef {
            Arc::new(StringArray::from_iter((0..n).map(|_| {
                (rng.random_range(0..6) != 0)
                    .then(|| ["", "a", "ab", "b"][rng.random_range(0..4)])
            })))
        };
        let makers: [&dyn Fn(&mut StdRng) -> ArrayRef; 3] = [&int, &float, &string];
        for case in 0..200 {
            let num_keys = rng.random_range(1..4);
            let columns = (0..num_keys)
                .map(|_| makers[rng.random_range(0..3)](&mut rng))
                .collect::<Vec<_>>();
            let options = (0..num_keys)
                .map(|_| SortOptions::new(rng.random_bool(0.5), rng.random_bool(0.5)))
                .collect::<Vec<_>>();
            let fields = columns
                .iter()
                .enumerate()
                .map(|(i, column)| {
                    Field::new(format!("c{i}"), column.data_type().clone(), true)
                })
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns.clone())?;
            let keys = (0..num_keys)
                .map(|i| (col(&format!("c{i}"), &schema).unwrap(), options[i]))
                .collect::<Vec<_>>();
            let rows = encoded(&columns, &options);
            let threshold = rng.random_range(0..n);
            let values = columns
                .iter()
                .map(|column| ScalarValue::try_from_array(column, threshold))
                .collect::<Result<Vec<_>>>()?;
            // All keys: the rows before the threshold row, and only these if
            // no value is a float zero (both zeros are selected then).
            let exact = !values.iter().any(is_float_zero);
            let predicate = threshold_predicate(&keys, &values)?.expect("a predicate");
            let selected = predicate.evaluate(&batch)?.into_array(n)?;
            let selected = selected.as_boolean();
            for row in 0..n {
                let before = rows.row(row) < rows.row(threshold);
                let is_selected = selected.is_valid(row) && selected.value(row);
                assert!(
                    if exact {
                        is_selected == before
                    } else {
                        is_selected || !before
                    },
                    "case {case}, row {row}: {predicate}, selected: {is_selected}\n{:?}\nthreshold: {values:?}",
                    batch.slice(row, 1),
                );
            }
            // A prefix of the keys: the rows before or equal to the
            // threshold row in that prefix.
            let prefix = rng.random_range(1..=num_keys);
            if prefix < num_keys {
                let predicate =
                    threshold_predicate(&keys[..prefix], &values)?.expect("a predicate");
                let selected = predicate.evaluate(&batch)?.into_array(n)?;
                let selected = selected.as_boolean();
                let prefix_rows = encoded(&columns[..prefix], &options[..prefix]);
                let exact = !values[..prefix].iter().any(is_float_zero);
                for row in 0..n {
                    let before = prefix_rows.row(row) <= prefix_rows.row(threshold);
                    let is_selected = selected.is_valid(row) && selected.value(row);
                    assert!(
                        if exact {
                            is_selected == before
                        } else {
                            is_selected || !before
                        },
                        "case {case}, prefix {prefix}, row {row}: {predicate}"
                    );
                }
            }
        }
        // No key to compare: no predicate.
        assert!(threshold_predicate(&[], &[ScalarValue::Int32(Some(1))])?.is_none());
        Ok(())
    }

    #[test]
    fn float_zeros_are_recognized_in_all_encodings() {
        let float16 = |v: f64| {
            let array =
                arrow::compute::cast(&Float64Array::from(vec![v]), &DataType::Float16);
            ScalarValue::try_from_array(&array.unwrap(), 0).unwrap()
        };
        let dictionary = |value: ScalarValue| {
            ScalarValue::Dictionary(Box::new(DataType::Int32), Box::new(value))
        };
        let run_end_encoded = |value: ScalarValue| {
            ScalarValue::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int32, false)),
                Arc::new(Field::new("values", value.data_type(), true)),
                Box::new(value),
            )
        };
        for zero in [
            ScalarValue::Float64(Some(0.0)),
            ScalarValue::Float64(Some(-0.0)),
            ScalarValue::Float32(Some(-0.0)),
            float16(0.0),
            float16(-0.0),
        ] {
            assert!(is_float_zero(&zero), "{zero:?}");
            assert!(is_float_zero(&dictionary(zero.clone())), "{zero:?}");
            assert!(is_float_zero(&run_end_encoded(zero.clone())), "{zero:?}");
            assert!(
                is_float_zero(&run_end_encoded(dictionary(zero.clone()))),
                "{zero:?}"
            );
        }
        for other in [
            ScalarValue::Float64(Some(1.0)),
            ScalarValue::Float64(Some(f64::NAN)),
            ScalarValue::Float64(None),
            ScalarValue::Int64(Some(0)),
            ScalarValue::Utf8(Some("0".to_string())),
        ] {
            assert!(!is_float_zero(&other), "{other:?}");
            assert!(!is_float_zero(&run_end_encoded(other.clone())), "{other:?}");
        }
    }

    /// Run the sink of `topk` on `chunks` on a single thread with the states
    /// `global` and `local`.
    fn sink_all(
        topk: &PhysicalTopK,
        global: &dyn GlobalSinkState,
        local: &mut dyn LocalSinkState,
        chunks: &[DataChunk],
    ) -> Result<()> {
        let ctx = ExecutionContext::new(Arc::new(TaskContext::default()), 0, 1);
        for chunk in chunks {
            assert!(matches!(
                topk.sink(&ctx, chunk, global, local)?,
                SinkResult::NeedMoreInput
            ));
        }
        Ok(())
    }

    fn buffered_rows(local: &dyn LocalSinkState) -> usize {
        local
            .downcast_ref::<TopKLocalState>()
            .unwrap()
            .buffered_rows
    }

    #[test]
    fn buffers_of_large_rows_are_reduced_before_they_hold_k_rows() -> Result<()> {
        // Rows of 2 MiB (sharing one buffer): the thread reduces its buffer
        // once it holds `TOPK_BUFFER_BYTES` bytes (32 rows), long before it
        // holds `MIN_TOPK_BUFFER_ROWS` rows (2 GiB, more than the 32-bit
        // offsets of a `Utf8` array of all of them address).
        let large = 2 << 20;
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec!["b".repeat(large)]))],
        )?;
        let chunks = vec![DataChunk::from_record_batch(batch); 100];
        let input: Arc<dyn PhysicalOperator> = Arc::new(TestSource::new(chunks.clone()));
        let topk = topk(input, false, 1);
        let ctx = ExecutionContext::new(Arc::new(TaskContext::default()), 0, 1);
        let global = topk.get_global_sink_state(&ctx)?;
        let mut local = topk.get_local_sink_state(&ctx, global.as_ref())?;
        let limit = TOPK_BUFFER_BYTES.div_ceil(large);
        for chunk in chunks.chunks(10) {
            sink_all(&topk, global.as_ref(), local.as_mut(), chunk)?;
            assert!(buffered_rows(local.as_ref()) < limit);
        }
        let state = local.downcast_ref::<TopKLocalState>().unwrap();
        assert_eq!(state.top.len(), 1);
        assert_eq!(state.top[0].num_rows(), 1);
        assert!(state.threshold.is_some());
        Ok(())
    }

    #[test]
    fn rows_behind_the_threshold_are_discarded_while_sinking() -> Result<()> {
        let input: Arc<dyn PhysicalOperator> = Arc::new(TestSource::new(vec![]));
        let topk = topk(input, false, 10);
        let ctx = ExecutionContext::new(Arc::new(TaskContext::default()), 0, 1);
        let global = topk.get_global_sink_state(&ctx)?;
        let mut local = topk.get_local_sink_state(&ctx, global.as_ref())?;
        // The first 2048 rows are reduced to the top 10: the threshold is 9.
        sink_all(
            &topk,
            global.as_ref(),
            local.as_mut(),
            &values_chunks(0, &[2048]),
        )?;
        assert_eq!(buffered_rows(local.as_ref()), 0);
        let state = local.downcast_ref::<TopKLocalState>().unwrap();
        assert_eq!(state.top.len(), 1);
        assert_eq!(state.top[0].num_rows(), 10);
        let threshold = state.threshold.as_ref().unwrap();
        assert_eq!(threshold.values, vec![ScalarValue::Int64(Some(9))]);
        assert_eq!(
            threshold.predicate.as_ref().unwrap().to_string(),
            "v@0 IS NULL OR v@0 < 9"
        );
        // No later row comes before 9, except these two.
        let chunks = values_chunks(2048, &[2048, 100]);
        sink_all(&topk, global.as_ref(), local.as_mut(), &chunks)?;
        assert_eq!(buffered_rows(local.as_ref()), 0);
        sink_all(
            &topk,
            global.as_ref(),
            local.as_mut(),
            &[values_chunk([3, 9, 100, -5])],
        )?;
        assert_eq!(buffered_rows(local.as_ref()), 2);

        // Another thread uses the threshold of the first one.
        let mut other = topk.get_local_sink_state(&ctx, global.as_ref())?;
        sink_all(
            &topk,
            global.as_ref(),
            other.as_mut(),
            &[values_chunk([50, 8, 9, 1])],
        )?;
        assert_eq!(buffered_rows(other.as_ref()), 2);

        // The result: the top 10 of all rows.
        topk.combine(&ctx, global.as_ref(), local.as_mut())?;
        topk.combine(&ctx, global.as_ref(), other.as_mut())?;
        assert_eq!(topk.finalize(&ctx, global.as_ref())?, SinkFinalize::Ready);
        let source = topk.get_global_source_state(&ctx);
        // (The context of the test has no registry with the sink state.)
        assert!(source.is_err());
        let global = global.downcast_ref::<TopKSinkState>().unwrap();
        let source = global.output.take_source(topk.name())?;
        let mut output = vec![];
        let mut lstate = topk.get_local_source_state(&ctx, source.as_ref())?;
        while let SourceResult::Chunk(chunk) =
            topk.get_data(&ctx, source.as_ref(), lstate.as_mut())?
        {
            output.push(chunk.into_record_batch()?);
        }
        assert_eq!(values(&output), vec![-5, 0, 1, 1, 2, 3, 3, 4, 5, 6]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topk_updates_the_dynamic_filter() -> Result<()> {
        let schema = values_schema();
        let v = col("v", &schema)?;
        let cases = [
            (false, 10, "v@0 IS NULL OR v@0 < 9"),
            (true, 3, "v@0 IS NULL OR v@0 > 9997"),
        ];
        for (descending, k, expected_filter) in cases {
            for threads in [1, 4] {
                let filter = Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![Arc::clone(&v)],
                    lit(true),
                ));
                let input = Arc::new(TestSource::new(scrambled_chunks(10_000, 2048)));
                let root = topk(input, descending, k).with_dynamic_filter(
                    TopKDynamicFilter::new(Arc::clone(&filter), vec![Arc::clone(&v)]),
                )?;
                assert!(root.description().ends_with(", dynamic_filter=true"));
                let output = run(Arc::new(root), threads).await?;
                let result = values(&output);
                assert_eq!(result.len(), k);
                assert!(filter.is_complete());
                if threads == 1 {
                    // The predicate of the threshold of the only thread.
                    assert_eq!(filter.current()?.to_string(), expected_filter);
                }
                // The filter selects the rows of the result before the last.
                let batch = values_chunk(result[..k - 1].iter().copied()).compact()?;
                let selected = filter.evaluate(&batch)?.into_array(k - 1)?;
                assert_eq!(
                    selected.as_boolean().true_count(),
                    k - 1,
                    "{threads} threads"
                );
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topk_keeps_negative_zeros_before_zeros() -> Result<()> {
        // The sort orders `-0.0` before `0.0`: once the threshold is a zero,
        // the zeros of the other sign must still pass its predicate, so that
        // the TopK keeps the first rows of its sort.
        let schema =
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let chunk = |values: Vec<f64>| {
            DataChunk::from_record_batch(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Float64Array::from(values))],
                )
                .unwrap(),
            )
        };
        for (descending, first, then) in [(false, 0.0, -0.0), (true, -0.0, 0.0)] {
            let chunks = vec![chunk(vec![first; 2048]), chunk(vec![then; 5])];
            let input: Arc<dyn PhysicalOperator> =
                Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks));
            let ordering = ordering(&schema, descending);
            let root = PhysicalTopK::try_new(input, ordering, 10)?;
            let output = run(Arc::new(root), 1).await?;
            let bits = output
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(0)
                        .as_primitive::<Float64Type>()
                        .values()
                        .to_vec()
                })
                .map(f64::to_bits)
                .collect::<Vec<_>>();
            let expected = [vec![then.to_bits(); 5], vec![first.to_bits(); 5]].concat();
            assert_eq!(bits, expected, "descending: {descending}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_float_keys_keep_the_first_rows_of_their_total_order() -> Result<()>
    {
        // Every `k`, also where the last row of the result is a zero of either
        // sign: the TopK returns the first `k` rows of its sort, in which floats
        // are in their total order, and never anything else (it used to execute
        // DataFusion's TopK instead where the last row was a zero).
        let schema =
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let values = vec![3.0, -0.0, 1.0, f64::NAN, 0.0, 2.0, -0.0, 0.0];
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        for descending in [false, true] {
            let expected = match descending {
                false => sorted.clone(),
                true => sorted.iter().rev().copied().collect(),
            };
            for k in 1..=values.len() + 1 {
                for threads in [1, 4] {
                    // A chunk per value, so that the threshold tightens as they
                    // come.
                    let chunks = values
                        .iter()
                        .map(|&value| {
                            let batch = RecordBatch::try_new(
                                Arc::clone(&schema),
                                vec![Arc::new(Float64Array::from(vec![value]))],
                            )?;
                            Ok(DataChunk::from_record_batch(batch))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let input =
                        Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks));
                    let topk =
                        PhysicalTopK::try_new(input, ordering(&schema, descending), k)?;
                    assert!(topk.float_key_fallback().is_none());
                    assert!(!topk.description().contains("fallback"));
                    let output = run(Arc::new(topk), threads).await?;
                    let bits = output
                        .iter()
                        .flat_map(|batch| {
                            let values = batch.column(0).as_primitive::<Float64Type>();
                            values.values().to_vec()
                        })
                        .map(f64::to_bits)
                        .collect::<Vec<_>>();
                    let expected = expected
                        .iter()
                        .take(k)
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>();
                    assert_eq!(
                        bits, expected,
                        "descending: {descending}, k = {k}, {threads} threads"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn the_dynamic_filter_of_a_prefix_of_the_keys_keeps_ties() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let (a, b) = (col("a", &schema)?, col("b", &schema)?);
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&a)],
            lit(true),
        ));
        let batch = |a: Vec<i64>, b: Vec<i64>| {
            DataChunk::from_record_batch(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from(a)), Arc::new(Int64Array::from(b))],
                )
                .unwrap(),
            )
        };
        let input: Arc<dyn PhysicalOperator> =
            Arc::new(TestSource::with_schema(Arc::clone(&schema), vec![]));
        let ordering = LexOrdering::new([
            PhysicalSortExpr::new_default(Arc::clone(&a)),
            PhysicalSortExpr::new_default(Arc::clone(&b)),
        ])
        .unwrap();
        // Only the first key of the TopK is a key of the filter.
        let topk = PhysicalTopK::try_new(input, ordering, 2)?
            .with_dynamic_filter(TopKDynamicFilter::new(Arc::clone(&filter), vec![a]))?;
        let ctx = ExecutionContext::new(Arc::new(TaskContext::default()), 0, 1);
        let global = topk.get_global_sink_state(&ctx)?;
        let mut local = topk.get_local_sink_state(&ctx, global.as_ref())?;
        let rows = (0..1024).map(|i| i + 10).collect::<Vec<_>>();
        let chunks = [
            batch(vec![1, 1, 1], vec![5, 3, 4]),
            batch(rows.clone(), rows),
        ];
        sink_all(&topk, global.as_ref(), local.as_mut(), &chunks)?;
        // The threshold is (1, 4): rows with a = 1 and b < 4 come before it.
        assert_eq!(
            filter.current()?.to_string(),
            "a@0 IS NULL OR a@0 < 1 OR a@0 = 1",
            "the filter keeps the rows with a = 1"
        );
        let evaluated = filter
            .evaluate(&batch(vec![0, 1, 2], vec![9, 0, 0]).into_record_batch()?)?
            .into_array(3)?;
        assert_eq!(
            evaluated.as_boolean(),
            &BooleanArray::from(vec![true, true, false])
        );
        // The TopK itself compares both keys.
        let state = local.downcast_ref::<TopKLocalState>().unwrap();
        let predicate = state
            .threshold
            .as_ref()
            .unwrap()
            .predicate
            .as_ref()
            .unwrap();
        assert_eq!(
            predicate.to_string(),
            "a@0 IS NULL OR a@0 < 1 OR a@0 = 1 AND (b@1 IS NULL OR b@1 < 4)"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topk_of_chunks_without_columns() -> Result<()> {
        let schema = Arc::new(Schema::empty());
        let chunks = (0..3)
            .map(|_| DataChunk::try_new_without_columns(Arc::clone(&schema), 2000))
            .collect::<Result<Vec<_>>>()?;
        let ordering =
            LexOrdering::new([PhysicalSortExpr::new_default(lit(1_i32))]).unwrap();
        for k in [1, 1500, 6000, 7000] {
            let input =
                Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks.clone()));
            let root = PhysicalTopK::try_new(input, ordering.clone(), k)?;
            let output = run(Arc::new(root), 4).await?;
            let rows = output.iter().map(RecordBatch::num_rows).sum::<usize>();
            assert_eq!(rows, k.min(6000), "k = {k}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_can_be_executed_again() -> Result<()> {
        let input = Arc::new(TestSource::new(scrambled_chunks(5000, 600)));
        let root: Arc<dyn PhysicalOperator> = Arc::new(topk(input, true, 5));
        for _ in 0..3 {
            let output = run(Arc::clone(&root), 4).await?;
            assert_eq!(values(&output), vec![4999, 4998, 4997, 4996, 4995]);
        }
        Ok(())
    }

    #[test]
    fn topk_properties_and_pipelines() -> Result<()> {
        let input: Arc<dyn PhysicalOperator> =
            Arc::new(TestSource::new(vec![values_chunk(0..3)]));
        assert!(
            PhysicalTopK::try_new(
                Arc::clone(&input),
                ordering(&input.schema(), false),
                0
            )
            .is_err()
        );
        let root = topk(Arc::clone(&input), false, 5);
        assert!(root.is_sink() && root.parallel_sink());
        assert!(root.is_source() && !root.parallel_source());
        assert_eq!(root.k(), 5);
        assert!(root.dynamic_filter().is_none());
        assert!(Arc::ptr_eq(root.input(), &input));
        assert_eq!(root.schema(), input.schema());
        assert_eq!(root.description(), "PhysicalTopK: fetch=5, expr=[v@0 ASC]");
        // A dynamic filter cannot have more keys than the TopK.
        let v = col("v", &input.schema())?;
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&v)],
            lit(true),
        ));
        let err = topk(Arc::clone(&input), false, 5)
            .with_dynamic_filter(TopKDynamicFilter::new(filter, vec![Arc::clone(&v), v]))
            .unwrap_err();
        assert!(err.to_string().contains("has 2 keys"), "{err}");
        let root: Arc<dyn PhysicalOperator> = Arc::new(root);
        assert_eq!(
            format_pipelines(&root)?,
            "MetaPipeline 0 (sink: ResultSink) children: [1]\n  \
             Pipeline 0: PhysicalTopK -> ResultSink depends on meta pipelines [1]\n\
             MetaPipeline 1 (sink: PhysicalTopK)\n  \
             Pipeline 1: TestSource -> PhysicalTopK\n"
        );
        Ok(())
    }
}
