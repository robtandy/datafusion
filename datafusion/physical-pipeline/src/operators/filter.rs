// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalFilter`]: selection-vector based filtering.
//!
//! The native counterpart of DataFusion's `FilterExec`. Instead of copying
//! the selected rows of every column (`filter_record_batch`), the filter
//! attaches a [`SelectionVector`] to its input chunk:
//!
//! 1. The predicate is split into its conjuncts by an [`AdaptiveFilter`],
//!    which also decides the order in which they are evaluated.
//! 2. The conjuncts are evaluated one after the other, each on the columns
//!    it references. A row is selected if every conjunct is `true` for it: a
//!    conjunct rejects the rows for which it is `false` or `NULL`, and the
//!    following conjuncts are not evaluated on them.
//! 3. As soon as every row is rejected, the remaining conjuncts are skipped
//!    and no output is produced.
//! 4. The output chunk shares the columns of the input (`Arc` clones, after
//!    applying the projection embedded in the `FilterExec`, if any) and
//!    carries the selected rows as its selection. No column is copied: the
//!    columns that are not referenced by the predicate are never touched.
//!
//! # The rows a conjunct is evaluated on
//!
//! A conjunct that can fail (see
//! [`fallibility`](super::adaptive_filter::fallibility)), e.g.
//! `10 / x > 1`, is only evaluated on the rows that the conjuncts before it
//! did not reject, and the [`AdaptiveFilter`] never moves a conjunct across
//! a fallible one: `x <> 0 AND 10 / x > 1` never divides by zero, however
//! many of the rows of a chunk have `x = 0`. (DataFusion evaluates such a
//! conjunct on all rows of a batch unless at most 20% of them are left, see
//! `BinaryExpr::evaluate_conjunction`, so it may fail where the native
//! filter does not: which rows an engine evaluates an expression on is its
//! own, see `docs/compatibility.md`.)
//!
//! A conjunct that cannot fail is evaluated on the rows that were not
//! rejected, or, if more than a quarter of the physical rows are left, on
//! all physical rows, which avoids copying its columns; its result for the
//! other rows is ignored.
//!
//! Only the columns a conjunct references are compacted (`take`) to the rows
//! it is evaluated on. Since `p AND q` is `true` exactly when both `p` and
//! `q` are `true`, the selected rows are the rows for which the predicate is
//! `true`, as in DataFusion.
//!
//! # Lowering a `FilterExec`
//!
//! [`PhysicalFilter::try_from_filter_exec`] lowers every `FilterExec`, whatever
//! its predicate, but those whose input the native operators cannot gather
//! the rows of, and those that limit every partition of an input of several
//! partitions to their fetch, see [`PhysicalFilter::supports_filter_exec`].

use std::borrow::Cow;
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, SchemaRef};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use datafusion_common::instant::Instant;
use datafusion_common::{Result, ScalarValue, internal_err, not_impl_err, plan_err};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use itertools::Itertools;

use super::adaptive_filter::{AdaptiveFilter, AdaptiveFilterState};
use super::projection::{BoundExprs, can_gather, check_lowered_input, logical_rows};
use crate::chunk::{DataChunk, SelectionVector};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, downcast_state, downcast_state_mut,
};

/// An infallible conjunct is evaluated on all physical rows of a chunk,
/// instead of only on the rows that are not rejected, if more than one in
/// this many rows are left: gathering the columns of the rows (`take`)
/// costs more per row than evaluating a cheap conjunct.
const DENSE_SELECTION: usize = 4;

/// Filters the [`DataChunk`]s of its input with a predicate by attaching a
/// [`SelectionVector`], like DataFusion's `FilterExec`.
///
/// See the [module documentation](self).
#[derive(Debug)]
pub struct PhysicalFilter {
    children: Vec<Arc<dyn PhysicalOperator>>,
    predicate: Arc<dyn PhysicalExpr>,
    /// The conjuncts of `predicate` and their evaluation orders.
    filter: AdaptiveFilter,
    /// The conjuncts of `filter`, bound to the input columns they reference.
    conjuncts: Vec<BoundExprs>,
    /// The input columns of the output, if not all of them.
    projection: Option<Vec<usize>>,
    /// The maximum number of rows to produce.
    fetch: Option<usize>,
    schema: SchemaRef,
}

impl PhysicalFilter {
    /// Create a filter that selects the rows of the chunks of `input` for
    /// which `predicate` is `true`, and outputs the input columns
    /// `projection` (all of them if `None`).
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        predicate: Arc<dyn PhysicalExpr>,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let schema = match &projection {
            Some(projection) => Arc::new(input_schema.project(projection)?),
            None => Arc::clone(&input_schema),
        };
        Self::try_new_with_schema(input, predicate, projection, schema)
    }

    /// Whether [`Self::try_from_filter_exec`] can lower `exec`. It lowers
    /// every predicate, whatever it may fail with (e.g. `x <> 0 AND 10 / x >
    /// 1`, which DataFusion may evaluate on rows with `x = 0`, see the
    /// [module documentation](self)), also with a fetch, but not if
    ///
    /// * the input of `exec` has lists of values that take less than a byte
    ///   each (e.g. fixed-size lists of `1 << 30` nulls, booleans, or empty
    ///   lists or binaries), which the native operators cannot gather with
    ///   `take`, or run-end encoded arrays whose child fields are not Arrow's
    ///   (see `can_gather`);
    /// * `exec` has a `fetch` and its input has several partitions: the
    ///   fetch limits every partition, but the chunks of a pipeline do not
    ///   belong to partitions. (A limit above such a filter that keeps at
    ///   most as many rows of all partitions makes the fetch redundant: see
    ///   [`PhysicalLimit::try_lower`](super::limit::PhysicalLimit::try_lower).)
    pub fn supports_filter_exec(exec: &FilterExec) -> bool {
        unsupported_filter_exec(exec).is_none()
    }

    /// Lower a DataFusion [`FilterExec`]: apply its predicate, projection and
    /// fetch to the chunks of `input` (the lowered input of `exec`) and
    /// produce exactly its output schema.
    ///
    /// Fails if `exec` is not supported, see [`Self::supports_filter_exec`].
    pub fn try_from_filter_exec(
        exec: &FilterExec,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        if let Some(reason) = unsupported_filter_exec(exec) {
            return not_impl_err!(
                "PhysicalFilter cannot lower this FilterExec: {reason}"
            );
        }
        check_lowered_input(exec.input().schema().as_ref(), input.as_ref())?;
        let projection = exec
            .projection()
            .as_ref()
            .map(|projection| projection.to_vec());
        Ok(Self::try_new_with_schema(
            input,
            Arc::clone(exec.predicate()),
            projection,
            exec.schema(),
        )?
        .with_fetch(exec.fetch()))
    }

    fn try_new_with_schema(
        input: Arc<dyn PhysicalOperator>,
        predicate: Arc<dyn PhysicalExpr>,
        projection: Option<Vec<usize>>,
        schema: SchemaRef,
    ) -> Result<Self> {
        let input_schema = input.schema();
        match predicate.data_type(input_schema.as_ref())? {
            DataType::Boolean => {}
            other => {
                return plan_err!(
                    "Filter predicate must return BOOLEAN values, got {other:?}"
                );
            }
        }
        let projected = match &projection {
            Some(projection) => input_schema.project(projection)?,
            None => input_schema.as_ref().clone(),
        };
        let same_types = projected.fields().len() == schema.fields().len()
            && projected
                .fields()
                .iter()
                .zip(schema.fields())
                .all(|(projected, field)| projected.data_type() == field.data_type());
        if !same_types {
            return internal_err!(
                "The schema of PhysicalFilter does not match its projected input: {schema:?} vs {projected:?}"
            );
        }
        let filter = AdaptiveFilter::try_new(&predicate, input_schema.as_ref())?;
        let conjuncts = filter
            .conjuncts()
            .iter()
            .map(|conjunct| {
                BoundExprs::try_new([Arc::clone(conjunct.expr())], input_schema.as_ref())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            children: vec![input],
            predicate,
            filter,
            conjuncts,
            projection,
            fetch: None,
            schema,
        })
    }

    /// Produce at most `fetch` rows: the first `fetch` rows of the input that
    /// pass the filter, then return [`OperatorResult::Finished`]. Like a
    /// `FilterExec` with a single input partition, a filter with a fetch is
    /// executed by a single thread (see [`PhysicalOperator::parallel_operator`]),
    /// so that it filters its input in order. It stops as soon as the fetch
    /// is reached.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// The filter predicate.
    pub fn predicate(&self) -> &Arc<dyn PhysicalExpr> {
        &self.predicate
    }

    /// The conjuncts of the predicate.
    pub fn adaptive_filter(&self) -> &AdaptiveFilter {
        &self.filter
    }

    /// The input columns of the output, if not all of them.
    pub fn projection(&self) -> Option<&[usize]> {
        self.projection.as_deref()
    }

    /// The maximum number of rows the filter produces.
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }
}

/// Why `exec` cannot be lowered, see [`PhysicalFilter::supports_filter_exec`].
fn unsupported_filter_exec(exec: &FilterExec) -> Option<String> {
    if let Some(field) = exec
        .input()
        .schema()
        .fields()
        .iter()
        .find(|field| !can_gather(field.data_type()))
    {
        return Some(format!(
            "its input column '{}' has lists of values that take less than a byte \
             each, or run-end encoded arrays with other child fields than Arrow's, \
             which the native operators cannot gather",
            field.name()
        ));
    }
    exec.fetch()?;
    let partitions = exec.input().output_partitioning().partition_count();
    if partitions > 1 {
        return Some(format!(
            "it limits each of its {partitions} input partitions to its fetch"
        ));
    }
    None
}

/// The global state of a [`PhysicalFilter`]: the rows it may still produce.
#[derive(Debug)]
struct FilterGlobalState {
    /// `None` without a fetch.
    remaining: Option<AtomicUsize>,
}

impl GlobalOperatorState for FilterGlobalState {}

impl FilterGlobalState {
    /// Whether the filter produced all the rows of its fetch.
    fn is_exhausted(&self) -> bool {
        self.remaining
            .as_ref()
            .is_some_and(|remaining| remaining.load(Ordering::Acquire) == 0)
    }

    /// Claim up to `rows` rows of the fetch. Returns the number of rows that
    /// may be produced and whether the fetch is exhausted.
    fn claim(&self, rows: usize) -> (usize, bool) {
        let Some(remaining) = &self.remaining else {
            return (rows, false);
        };
        let mut available = remaining.load(Ordering::Acquire);
        loop {
            let claimed = rows.min(available);
            match remaining.compare_exchange_weak(
                available,
                available - claimed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return (claimed, claimed == available),
                Err(actual) => available = actual,
            }
        }
    }
}

/// The state of a [`PhysicalFilter`] in one thread.
#[derive(Debug)]
struct FilterState {
    adaptive: AdaptiveFilterState,
}

impl OperatorState for FilterState {}

/// The rows of an input chunk that pass the filter.
enum Selected {
    /// No row passes.
    Nothing,
    /// Every (logical) row passes.
    All,
    /// The physical rows that pass, a subset of the logical rows.
    Rows(SelectionVector),
}

/// The result of a conjunct for the rows it was evaluated on.
enum Mask {
    /// Every row is `true`.
    True,
    /// No row is `true`: every row is `false` or `NULL`.
    False,
    /// The rows that are `true`.
    Some(BooleanBuffer),
}

/// The rows of a [`Mask`]: how the rows the conjunct was evaluated on relate
/// to the logical rows of the chunk.
enum MaskRows<'a> {
    /// The rows that are not rejected, in order.
    Undecided,
    /// All physical rows; the logical rows are `incoming` (all physical rows
    /// if `None`).
    Physical(Option<&'a SelectionVector>),
}

/// The logical rows of a chunk that the conjuncts evaluated so far did not
/// reject: those for which all of them are `true`, as a bitmap over the
/// logical rows. A row for which a conjunct is `NULL` is never selected, so
/// it is rejected at once, like a `false` row: the next conjuncts are not
/// evaluated on it.
struct Undecided {
    /// The number of logical rows of the chunk.
    num_rows: usize,
    /// The rows that are not rejected (`None` for all logical rows).
    rows: Option<BooleanBuffer>,
    /// The number of rows of `rows`.
    count: usize,
}

impl Undecided {
    fn all(num_rows: usize) -> Self {
        Self {
            num_rows,
            rows: None,
            count: num_rows,
        }
    }

    /// The number of rows that are not rejected.
    fn len(&self) -> usize {
        self.count
    }

    /// The physical rows that are not rejected (`None` for all physical
    /// rows), given the physical rows of the logical rows, `incoming`.
    fn physical_rows<'a>(
        &self,
        incoming: Option<&'a SelectionVector>,
    ) -> Option<Cow<'a, SelectionVector>> {
        let Some(rows) = &self.rows else {
            return incoming.map(Cow::Borrowed);
        };
        let mut physical = Vec::with_capacity(self.count);
        match incoming {
            Some(incoming) => {
                let incoming = incoming.indices();
                physical.extend(rows.set_indices().map(|row| incoming[row]));
            }
            None => physical.extend(rows.set_indices_u32()),
        }
        Some(Cow::Owned(SelectionVector::from(physical)))
    }

    /// Reject the rows for which the conjunct with result `mask` is not
    /// `true`.
    fn update(&mut self, mask: Mask, mask_rows: &MaskRows<'_>) {
        let selected = match mask {
            Mask::True => return,
            Mask::False => {
                self.rows = Some(BooleanBuffer::new_unset(self.num_rows));
                self.count = 0;
                return;
            }
            Mask::Some(selected) => selected,
        };
        // The mask, over the logical rows.
        let selected = match mask_rows {
            MaskRows::Physical(None) => selected,
            MaskRows::Undecided => match &self.rows {
                None => selected,
                // Scatter the results to the rows they were computed for.
                Some(rows) => scatter(&selected, rows, self.num_rows),
            },
            MaskRows::Physical(Some(incoming)) => gather(&selected, incoming),
        };
        let rows = match &self.rows {
            Some(rows) => rows & &selected,
            None => selected,
        };
        self.count = rows.count_set_bits();
        self.rows = Some(rows);
    }

    /// The selected rows: those for which every conjunct was `true`.
    fn into_selected(self, incoming: Option<&SelectionVector>) -> Selected {
        if self.count == 0 {
            return Selected::Nothing;
        }
        let Some(selected) = self.rows else {
            return Selected::All;
        };
        let mut physical = Vec::with_capacity(self.count);
        match incoming {
            Some(incoming) => {
                let incoming = incoming.indices();
                physical.extend(selected.set_indices().map(|row| incoming[row]));
            }
            None => physical.extend(selected.set_indices_u32()),
        }
        Selected::Rows(SelectionVector::from(physical))
    }
}

/// The bitmap over `num_rows` rows whose bit for the `i`-th set bit of
/// `rows` is the `i`-th bit of `values`.
fn scatter(
    values: &BooleanBuffer,
    rows: &BooleanBuffer,
    num_rows: usize,
) -> BooleanBuffer {
    let mut builder = BooleanBufferBuilder::new(num_rows);
    builder.append_n(num_rows, false);
    for (value, row) in values.iter().zip(rows.set_indices()) {
        if value {
            builder.set_bit(row, true);
        }
    }
    builder.finish()
}

/// The bitmap whose `i`-th bit is the bit `rows[i]` of `values`.
fn gather(values: &BooleanBuffer, rows: &SelectionVector) -> BooleanBuffer {
    BooleanBuffer::from_iter(rows.iter().map(|row| values.value(row)))
}

impl PhysicalOperator for PhysicalFilter {
    fn name(&self) -> &str {
        "PhysicalFilter"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let mut description = format!("PhysicalFilter: {}", self.predicate);
        if let Some(projection) = &self.projection {
            let input_schema = self.children[0].schema();
            let _ = write!(
                description,
                ", projection=[{}]",
                projection
                    .iter()
                    .map(|&index| format!("{}@{index}", input_schema.field(index).name()))
                    .join(", ")
            );
        }
        if let Some(fetch) = self.fetch {
            let _ = write!(description, ", fetch={fetch}");
        }
        description
    }

    /// A filter with a fetch is executed by a single thread, which filters
    /// its input in order and produces its first `fetch` rows, as
    /// `FilterExec` does for a single partition.
    fn parallel_operator(&self) -> bool {
        self.fetch.is_none()
    }

    fn get_global_operator_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalOperatorState>> {
        Ok(Arc::new(FilterGlobalState {
            remaining: self.fetch.map(AtomicUsize::new),
        }))
    }

    fn get_operator_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(FilterState {
            adaptive: self.filter.create_state(),
        }))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let gstate = downcast_state::<FilterGlobalState>(gstate.as_any(), self.name())?;
        let state = downcast_state_mut::<FilterState>(lstate.as_any_mut(), self.name())?;
        if gstate.is_exhausted() {
            return Ok(OperatorResult::Finished);
        }
        let (selection, num_rows) = match self.select(input, &mut state.adaptive)? {
            // Short circuit: the output stays empty.
            Selected::Nothing => return Ok(OperatorResult::NeedMoreInput),
            Selected::All => (logical_rows(input).cloned(), input.len()),
            Selected::Rows(rows) => {
                let num_rows = rows.len();
                (Some(rows), num_rows)
            }
        };
        let (num_rows, exhausted) = gstate.claim(num_rows);
        if num_rows > 0 {
            *output = self.output_chunk(input, selection, num_rows)?;
        }
        Ok(if exhausted {
            OperatorResult::Finished
        } else {
            OperatorResult::NeedMoreInput
        })
    }
}

impl PhysicalFilter {
    /// Evaluate the conjuncts on the (non-empty) chunk `input`, in the order
    /// of `state`, and record their statistics.
    fn select(
        &self,
        input: &DataChunk,
        state: &mut AdaptiveFilterState,
    ) -> Result<Selected> {
        let incoming = logical_rows(input);
        let mut undecided = Undecided::all(input.len());
        let timed = state.is_adaptive();
        for position in 0..self.conjuncts.len() {
            if undecided.len() == 0 {
                // Every row is rejected: the remaining conjuncts are not
                // evaluated.
                break;
            }
            let conjunct = state.permutation()[position];
            let input_rows = undecided.len();
            let start = timed.then(Instant::now);
            self.evaluate_conjunct(conjunct, input, incoming, &mut undecided)?;
            if let Some(start) = start {
                state.record(conjunct, input_rows, undecided.len(), start.elapsed());
            }
        }
        state.end_chunk();
        Ok(undecided.into_selected(incoming))
    }

    /// Evaluate conjunct `conjunct` on `input` (whose logical rows are
    /// `incoming`, all physical rows if `None`), and reject the rows of
    /// `undecided` for which it is `false`.
    fn evaluate_conjunct(
        &self,
        conjunct: usize,
        input: &DataChunk,
        incoming: Option<&SelectionVector>,
        undecided: &mut Undecided,
    ) -> Result<()> {
        let bound = &self.conjuncts[conjunct];
        let expr = &bound.exprs()[0];
        // A conjunct that may fail is only evaluated on the rows that are left,
        // so that it does not fail on the rows that the conjuncts before it
        // rejected (e.g. `x <> 0 AND 10 / x > 1`).
        if !self.filter.conjuncts()[conjunct].is_fallible()
            && undecided.len() * DENSE_SELECTION > input.physical_len()
        {
            // Many rows are left and the conjunct cannot fail on the other
            // ones: evaluating it on all physical rows is cheaper than
            // gathering its columns.
            let batch = bound.batch(input, None)?;
            undecided.update(evaluate_mask(expr, &batch)?, &MaskRows::Physical(incoming));
            return Ok(());
        }
        let rows = undecided.physical_rows(incoming);
        let batch = bound.batch(input, rows.as_deref())?;
        undecided.update(evaluate_mask(expr, &batch)?, &MaskRows::Undecided);
        Ok(())
    }

    /// The output chunk: the (projected) columns of `input` with the
    /// selection `selection`, limited to `num_rows` rows.
    fn output_chunk(
        &self,
        input: &DataChunk,
        selection: Option<SelectionVector>,
        num_rows: usize,
    ) -> Result<DataChunk> {
        if self.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(
                Arc::clone(&self.schema),
                num_rows,
            );
        }
        let columns = match &self.projection {
            Some(projection) => projection
                .iter()
                .map(|&index| Arc::clone(input.column(index)))
                .collect(),
            None => input.columns().to_vec(),
        };
        let chunk = DataChunk::try_new(Arc::clone(&self.schema), columns, selection)?;
        Ok(if num_rows < chunk.len() {
            chunk.slice(0, num_rows)
        } else {
            chunk
        })
    }
}

/// Evaluate a conjunct on `batch`, with the semantics of `FilterExec`: the
/// result must be a boolean, and the rows for which it is `true` pass.
fn evaluate_mask(expr: &Arc<dyn PhysicalExpr>, batch: &RecordBatch) -> Result<Mask> {
    let num_rows = batch.num_rows();
    let array = match expr.evaluate(batch)? {
        ColumnarValue::Scalar(ScalarValue::Boolean(value)) => {
            return Ok(match value {
                Some(true) => Mask::True,
                Some(false) | None => Mask::False,
            });
        }
        value => value.into_array(num_rows)?,
    };
    let Some(mask) = array.as_boolean_opt() else {
        return internal_err!("Cannot create filter_array from non-boolean predicates");
    };
    if mask.len() != num_rows {
        return internal_err!(
            "Filter predicate returned {} values for {num_rows} rows",
            mask.len()
        );
    }
    // A `NULL` is not `true`.
    let selected = match mask.nulls().filter(|nulls| nulls.null_count() > 0) {
        Some(nulls) => mask.values() & nulls.inner(),
        None => mask.values().clone(),
    };
    Ok(match selected.count_set_bits() {
        0 => Mask::False,
        count if count == num_rows => Mask::True,
        _ => Mask::Some(selected),
    })
}

/// Test support shared by the tests of the native streaming operators.
///
/// The planner does not lower any node natively yet, so the
/// differential tests of these operators lower the plans themselves:
/// [`lower`] replaces every `FilterExec` with a [`PhysicalFilter`] wrapped
/// in a [`CachingOperator`](super::caching::CachingOperator) and every
/// `ProjectionExec` with a [`PhysicalProjection`](super::projection::PhysicalProjection),
/// and executes all other nodes with DataFusion.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, DictionaryArray, Int32Array, Int64Array, RecordBatch, StringArray,
        StringViewArray,
    };
    use arrow::datatypes::{DataType, Field, Int8Type, Int32Type, Schema, SchemaRef};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::{DFSchema, DataFusionError, Result};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        Volatility,
    };
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::limit::LocalLimitExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::{
        ChildrenPropertiesMode, Distribution, ExecutionPlan, ExecutionPlanProperties,
        Partitioning, ReplaceChildrenOptions, collect,
    };

    use super::PhysicalFilter;
    use crate::bridge::{BridgeOperator, PipelineRootExec};
    use crate::chunk::DataChunk;
    use crate::operator::{
        ExecutionContext, GlobalOperatorState, OperatorFinalizeResult, OperatorResult,
        OperatorState, PhysicalOperator,
    };
    use crate::operators::adaptive_filter::TrustedFunctions;
    use crate::operators::caching::CachingOperator;
    use crate::operators::projection::PhysicalProjection;
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, has_top_level_order_by,
        register_test_tables, run_engines, run_sql, test_session_config,
    };

    /// Lower `plan` into an operator tree that executes the `FilterExec`s
    /// and `ProjectionExec`s natively, and the other nodes with DataFusion:
    ///
    /// * `FilterExec` becomes `CachingOperator(PhysicalFilter)` and
    ///   `ProjectionExec` becomes `PhysicalProjection`, if supported (see
    ///   [`PhysicalFilter::supports_filter_exec`] and
    ///   [`PhysicalProjection::supports_projection_exec`]), whatever their
    ///   expressions may fail with, also below limits.
    /// * Scans become the native sources of
    ///   [`try_create_scan_source`](crate::sources::try_create_scan_source).
    /// * `CoalescePartitionsExec` (without fetch) and round robin
    ///   `RepartitionExec`s above native operators are removed: they only
    ///   change the partitioning, which the pipeline engine does not need.
    /// * Every other node is executed by DataFusion (in a `BridgeOperator`),
    ///   see [`bridged_plan`].
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            if PhysicalFilter::supports_filter_exec(filter) {
                let input = lower(filter.input())?;
                let filter: Arc<dyn PhysicalOperator> =
                    Arc::new(PhysicalFilter::try_from_filter_exec(filter, input)?);
                return Ok(Arc::new(CachingOperator::new(filter)));
            }
            return Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))));
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            if PhysicalProjection::supports_projection_exec(projection) {
                let input = lower(projection.input())?;
                return Ok(Arc::new(PhysicalProjection::try_from_projection_exec(
                    projection, input,
                )?));
            }
            return Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))));
        }
        if let Some(scan) = crate::sources::try_create_scan_source(plan)? {
            return Ok(scan);
        }
        if only_repartitions(plan) {
            let input = lower(plan.children()[0])?;
            if !input.is::<BridgeOperator>() {
                return Ok(input);
            }
        }
        Ok(Arc::new(BridgeOperator::new(bridged_plan(plan)?)))
    }

    /// `plan`, to be executed by DataFusion, with its children lowered: a
    /// child that contains native operators is executed by a
    /// [`PipelineRootExec`], unless `plan` needs the partitioning or the
    /// order of the child (or limits every partition of the child) and the
    /// child has several partitions, which a `PipelineRootExec` (a single
    /// partition) does not keep. DataFusion then executes the child too, with
    /// its own children lowered.
    fn bridged_plan(plan: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let children = plan.children();
        let new_children = children
            .iter()
            .enumerate()
            .map(|(index, child)| child_plan(plan, index, child))
            .collect::<Result<Vec<_>>>()?;
        let unchanged = children
            .iter()
            .zip(&new_children)
            .all(|(child, new_child)| Arc::ptr_eq(child, new_child));
        if unchanged {
            return Ok(Arc::clone(plan));
        }
        Arc::clone(plan).replace_children(
            new_children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    /// Whether `plan` only changes the partitioning of its input.
    fn only_repartitions(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
            return coalesce.fetch().is_none();
        }
        plan.downcast_ref::<RepartitionExec>()
            .is_some_and(|repartition| {
                matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
                    && !repartition.preserve_order()
            })
    }

    /// The plan that DataFusion executes as child `index` of `parent`.
    fn child_plan(
        parent: &Arc<dyn ExecutionPlan>,
        index: usize,
        child: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let lowered = lower(child)?;
        if let Some(bridge) = lowered.downcast_ref::<BridgeOperator>() {
            return Ok(Arc::clone(bridge.plan()));
        }
        let single_partition = child.output_partitioning().partition_count() == 1;
        // A `FilterExec` with a fetch and a `LocalLimitExec` limit every
        // partition of their input.
        let limits_partitions = (parent.is::<FilterExec>() && parent.fetch().is_some())
            || parent.is::<LocalLimitExec>();
        let unconstrained = matches!(
            parent
                .input_distribution_requirements()
                .child_distribution(index),
            None | Some(
                Distribution::UnspecifiedDistribution | Distribution::SinglePartition
            )
        ) && child.output_ordering().is_none()
            && !limits_partitions;
        if !single_partition && !unconstrained {
            return bridged_plan(child);
        }
        Ok(Arc::new(PipelineRootExec::try_new(
            lowered,
            Arc::clone(child),
        )?))
    }

    /// The native operators of a lowered plan (see [`lower`]).
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct NativeOperators {
        pub(crate) filters: usize,
        pub(crate) projections: usize,
        pub(crate) caching: usize,
        /// Native scan sources.
        pub(crate) scans: usize,
    }

    impl NativeOperators {
        /// Count the native operators of an operator tree, including those
        /// of the `PipelineRootExec`s inside bridged plans.
        pub(crate) fn of(operator: &Arc<dyn PhysicalOperator>) -> Self {
            let mut counts = Self::default();
            counts.add_operator(operator);
            counts
        }

        fn add_operator(&mut self, operator: &Arc<dyn PhysicalOperator>) {
            self.filters += usize::from(operator.is::<PhysicalFilter>());
            self.projections += usize::from(operator.is::<PhysicalProjection>());
            self.caching += usize::from(operator.is::<CachingOperator>());
            self.scans += usize::from(
                operator.children().is_empty() && !operator.is::<BridgeOperator>(),
            );
            if let Some(bridge) = operator.downcast_ref::<BridgeOperator>() {
                self.add_plan(bridge.plan());
            }
            for child in operator.children() {
                self.add_operator(child);
            }
        }

        fn add_plan(&mut self, plan: &Arc<dyn ExecutionPlan>) {
            if let Some(root) = plan.downcast_ref::<PipelineRootExec>() {
                self.add_operator(root.operator());
            }
            for child in plan.children() {
                self.add_plan(child);
            }
        }
    }

    /// The result of running a query with DataFusion and with the native
    /// operators.
    pub(crate) struct NativeRun {
        pub(crate) expected: EngineOutput,
        pub(crate) actual: EngineOutput,
        /// The plan executed for `actual`.
        pub(crate) root: Arc<PipelineRootExec>,
        pub(crate) operators: NativeOperators,
    }

    /// Run `sql` with DataFusion and with the plan of DataFusion lowered by
    /// [`lower`], on sessions configured with `config` and set up with
    /// `setup`.
    pub(crate) async fn run_native<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Result<NativeRun>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let config = trusting(config);
        let datafusion = datafusion_context(config.clone());
        setup(&datafusion).await?;
        let native = datafusion_context(config);
        setup(&native).await?;

        let expected = run_sql(&datafusion, sql).await?;
        let df = native.sql(sql).await?;
        let ordered = has_top_level_order_by(df.logical_plan());
        let plan = df.create_physical_plan().await?;
        let operator = lower(&plan)?;
        let operators = NativeOperators::of(&operator);
        let root = Arc::new(PipelineRootExec::try_new(operator, plan)?);
        let batches = collect(
            Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            native.task_ctx(),
        )
        .await?;
        let actual = EngineOutput {
            plan: Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            schema: root.schema(),
            batches,
            ordered,
        };
        Ok(NativeRun {
            expected,
            actual,
            root,
            operators,
        })
    }

    /// Assert that `sql` produces byte-identical results with DataFusion and
    /// with its native operators (see [`lower`]), like
    /// [`assert_byte_eq_with_datafusion_with_setup`](crate::test_utils::assert_byte_eq_with_datafusion_with_setup).
    /// `setup` registers the tables used by the query, e.g. [`all_tables`].
    pub(crate) async fn assert_native_byte_eq_with_setup<F>(
        sql: &str,
        setup: F,
    ) -> NativeRun
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_native_byte_eq_with_config(sql, test_session_config(), setup).await
    }

    /// Like [`assert_native_byte_eq_with_setup`], with a custom session
    /// configuration.
    pub(crate) async fn assert_native_byte_eq_with_config<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> NativeRun
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let run = match run_native(sql, config, setup).await {
            Ok(run) => run,
            Err(e) => panic!("Failed to run query on both engines: {e}\nquery: {sql}"),
        };
        if let Err(message) = compare_outputs(&run.expected, &run.actual) {
            panic!("Results differ: {message}\nquery: {sql}");
        }
        run
    }

    /// Run `sql` with DataFusion and with the
    /// [`PipelineQueryPlanner`](crate::PipelineQueryPlanner), in sessions
    /// configured with `config` (as is: unlike [`run_native`], this does not
    /// make them trust the built-in functions, see [`trusting`]) and set up
    /// with `setup`, assert that the results are byte-identical, and return
    /// the plan of DataFusion and the operator tree that the planner lowered
    /// it to.
    pub(crate) async fn assert_planned_byte_eq<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> (Arc<dyn ExecutionPlan>, Arc<dyn PhysicalOperator>)
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let (expected, actual) = match run_engines(sql, config, setup).await {
            Ok(outputs) => outputs,
            Err(e) => panic!("Failed to run query on both engines: {e}\nquery: {sql}"),
        };
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\nquery: {sql}");
        }
        let root = actual
            .plan
            .downcast_ref::<PipelineRootExec>()
            .expect("the pipeline engine planned the query");
        (expected.plan, Arc::clone(root.operator()))
    }

    /// Execute `plan`, created by `make_plan`, with DataFusion and with its
    /// native operators (see [`lower`]), and assert that the results are
    /// byte-identical (in the same order if `ordered`).
    pub(crate) async fn assert_native_plan_byte_eq(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        ordered: bool,
    ) -> NativeRun {
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_target_partitions(4)),
        );
        let plan = make_plan();
        let expected = EngineOutput {
            schema: plan.schema(),
            batches: collect(Arc::clone(&plan), Arc::clone(&task_ctx))
                .await
                .unwrap(),
            plan,
            ordered,
        };
        let plan = make_plan();
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        let root = Arc::new(PipelineRootExec::try_new(operator, plan).unwrap());
        let batches = collect(Arc::clone(&root) as Arc<dyn ExecutionPlan>, task_ctx)
            .await
            .unwrap();
        let actual = EngineOutput {
            plan: Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            schema: root.schema(),
            batches,
            ordered,
        };
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}");
        }
        NativeRun {
            expected,
            actual,
            root,
            operators,
        }
    }

    /// [`assert_native_plan_parity`] with the session configuration `config`
    /// (whose target partitions are the threads of the pipelines).
    pub(crate) async fn assert_native_plan_parity_with_config(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        ordered: bool,
        config: SessionConfig,
    ) -> (Option<String>, NativeOperators) {
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let plan = make_plan();
        let expected =
            collect(Arc::clone(&plan), Arc::clone(&task_ctx))
                .await
                .map(|batches| EngineOutput {
                    schema: plan.schema(),
                    batches,
                    plan,
                    ordered,
                });
        let plan = make_plan();
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        let root = executable(operator, plan).unwrap();
        let actual =
            collect(Arc::clone(&root), task_ctx)
                .await
                .map(|batches| EngineOutput {
                    plan: Arc::clone(&root),
                    schema: root.schema(),
                    batches,
                    ordered,
                });
        let error = match (expected, actual) {
            (Ok(expected), Ok(actual)) => {
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}");
                }
                None
            }
            (Err(expected), Err(actual)) => {
                assert_eq!(
                    error_kind(&expected),
                    error_kind(&actual),
                    "Errors differ: {expected} vs {actual}"
                );
                Some(expected.find_root().strip_backtrace())
            }
            (Ok(_), Err(actual)) => panic!("Only the native operators failed: {actual}"),
            (Err(expected), Ok(_)) => panic!("Only DataFusion failed: {expected}"),
        };
        (error, operators)
    }

    /// The kind of an error, e.g. `DivideByZero` or `CastError`, without the
    /// data it reports: with several partitions, which row fails first is
    /// not deterministic, neither in DataFusion nor in the pipeline.
    fn error_kind(error: &DataFusionError) -> String {
        let debug = match error.find_root() {
            DataFusionError::ArrowError(error, _) => format!("ArrowError::{error:?}"),
            error => format!("{error:?}"),
        };
        debug.split('(').next().unwrap_or_default().to_owned()
    }

    /// Parse the SQL expression `sql` on `schema` into a physical expression,
    /// with DataFusion's type coercion.
    pub(crate) fn physical_expr(sql: &str, schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        physical_expr_in(
            &SessionContext::new_with_config(trusting(SessionConfig::new())),
            sql,
            schema,
        )
    }

    /// Like [`physical_expr`], in the session of `ctx`, which may not trust
    /// the built-in functions of DataFusion (see [`trusting`]).
    pub(crate) fn physical_expr_in(
        ctx: &SessionContext,
        sql: &str,
        schema: &SchemaRef,
    ) -> Arc<dyn PhysicalExpr> {
        let df_schema = DFSchema::try_from(schema.as_ref().clone()).unwrap();
        let expr = ctx.parse_sql_expr(sql, &df_schema).unwrap();
        // The state of a query, whose configuration the calls of functions hold.
        ctx.state().create_physical_expr(expr, &df_schema).unwrap()
    }

    /// `config`, trusting the built-in functions of DataFusion, which the
    /// sessions of the tests register, by their types (see
    /// [`TrustedFunctions`]: by default, sessions trust them by their names).
    pub(crate) fn trusting(config: SessionConfig) -> SessionConfig {
        config.with_option_extension(TrustedFunctions::new(
            &datafusion::functions::all_default_functions(),
        ))
    }

    /// A batch of the columns `x` and `y` (`Int32`) and `s` (`Utf8`, `"a"` on
    /// all rows), with the runs `((x, y), rows)`.
    pub(crate) fn xys(runs: &[((i32, i32), usize)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let rows = || {
            runs.iter()
                .flat_map(|&(xy, rows)| std::iter::repeat_n(xy, rows))
        };
        let x = rows().map(|(x, _)| x).collect::<Int32Array>();
        let y = rows().map(|(_, y)| y).collect::<Int32Array>();
        let s = StringArray::from_iter_values(rows().map(|_| "a"));
        RecordBatch::try_new(schema, vec![Arc::new(x), Arc::new(y), Arc::new(s)]).unwrap()
    }

    /// A batch of the columns `x` (`Int32`) and `s` (`Utf8`), with the runs
    /// `((x, s), rows)`.
    pub(crate) fn xss(runs: &[((i32, &str), usize)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let rows = || {
            runs.iter()
                .flat_map(|&(xs, rows)| std::iter::repeat_n(xs, rows))
        };
        let x = rows().map(|(x, _)| x).collect::<Int32Array>();
        let s = StringArray::from_iter_values(rows().map(|(_, s)| s));
        RecordBatch::try_new(schema, vec![Arc::new(x), Arc::new(s)]).unwrap()
    }

    /// A column that is not a child of the expression (like the `CustomColumn`
    /// of the tests of DataFusion's `CaseExpr`): the input columns it reads
    /// are opaque.
    #[derive(Debug, Hash, PartialEq, Eq)]
    pub(crate) struct OpaqueColumn(Column);

    /// The [`OpaqueColumn`] of the column `name` of `schema`.
    pub(crate) fn opaque_column(name: &str, schema: &Schema) -> Arc<dyn PhysicalExpr> {
        Arc::new(OpaqueColumn(Column::new_with_schema(name, schema).unwrap()))
    }

    impl std::fmt::Display for OpaqueColumn {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "opaque({})", self.0)
        }
    }

    impl PhysicalExpr for OpaqueColumn {
        fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
            self.0.data_type(input_schema)
        }

        fn nullable(&self, input_schema: &Schema) -> Result<bool> {
            self.0.nullable(input_schema)
        }

        fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
            self.0.evaluate(batch)
        }

        fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn PhysicalExpr>>,
        ) -> Result<Arc<dyn PhysicalExpr>> {
            Ok(self)
        }

        fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "opaque({})", self.0.name())
        }
    }

    /// A volatile function that counts its calls: it returns the number of
    /// times it was called, as a string, on all rows.
    #[derive(Debug)]
    pub(crate) struct CountCalls {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        signature: Signature,
    }

    impl PartialEq for CountCalls {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.calls, &other.calls)
        }
    }

    impl Eq for CountCalls {}

    impl std::hash::Hash for CountCalls {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            Arc::as_ptr(&self.calls).hash(state);
        }
    }

    impl ScalarUDFImpl for CountCalls {
        fn name(&self) -> &str {
            "count_calls"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Utf8)
        }

        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let calls = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            let values = std::iter::repeat_n(calls.to_string(), args.number_rows);
            Ok(ColumnarValue::Array(Arc::new(
                StringArray::from_iter_values(values),
            )))
        }
    }

    /// `count_calls()` (see [`CountCalls`]), counting in `calls`.
    pub(crate) fn count_calls(
        calls: &Arc<std::sync::atomic::AtomicUsize>,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        let udf = ScalarUDF::new_from_impl(CountCalls {
            calls: Arc::clone(calls),
            signature: Signature::nullary(Volatility::Volatile),
        });
        Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(udf),
                vec![],
                schema,
                Arc::new(datafusion_common::config::ConfigOptions::default()),
            )
            .unwrap(),
        )
    }

    /// A volatile function that returns whether it is called for the first
    /// time, on all rows.
    #[derive(Debug)]
    pub(crate) struct FirstCall {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        signature: Signature,
    }

    impl PartialEq for FirstCall {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.calls, &other.calls)
        }
    }

    impl Eq for FirstCall {}

    impl std::hash::Hash for FirstCall {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            Arc::as_ptr(&self.calls).hash(state);
        }
    }

    impl ScalarUDFImpl for FirstCall {
        fn name(&self) -> &str {
            "first_call"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Boolean)
        }

        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let first = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
            Ok(ColumnarValue::Array(Arc::new(
                arrow::array::BooleanArray::from(vec![first; args.number_rows]),
            )))
        }
    }

    /// `CAST(dynamic(first_call()) AS Utf8) LIKE 'true'`, with `first_call()`
    /// (see [`FirstCall`]) counting in `calls` in a `DynamicFilterPhysicalExpr`
    /// (without children): `true` if evaluated once, `false` after.
    pub(crate) fn dynamic_first_call(
        calls: &Arc<std::sync::atomic::AtomicUsize>,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        let udf = ScalarUDF::new_from_impl(FirstCall {
            calls: Arc::clone(calls),
            signature: Signature::nullary(Volatility::Volatile),
        });
        let first_call = Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(udf),
                vec![],
                schema,
                Arc::new(datafusion_common::config::ConfigOptions::default()),
            )
            .unwrap(),
        );
        let dynamic = Arc::new(
            datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr::new(
                vec![],
                first_call,
            ),
        );
        Arc::new(datafusion_physical_expr::expressions::LikeExpr::new(
            false,
            false,
            Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
                dynamic,
                DataType::Utf8,
                None,
            )),
            Arc::new(datafusion_physical_expr::expressions::Literal::new(
                datafusion_common::ScalarValue::Utf8(Some("true".to_owned())),
            )),
        ))
    }

    /// An immutable function of strings `s` that returns the lengths of
    /// `s || s`, an intermediate of strings with 32-bit offsets: it returns
    /// integers, but fails on more than 1 GiB of strings in a batch.
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub(crate) struct ConcatLength {
        signature: Signature,
    }

    impl ScalarUDFImpl for ConcatLength {
        fn name(&self) -> &str {
            "concat_length"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int32)
        }

        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let strings = args.args[0].to_array(args.number_rows)?;
            let strings = strings
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("strings");
            let twice = arrow::compute::kernels::concat_elements::concat_elements_utf8(
                strings, strings,
            )?;
            Ok(ColumnarValue::Array(
                arrow::compute::kernels::length::length(&twice)?,
            ))
        }
    }

    /// `concat_length(s)` (see [`ConcatLength`]) of the column `s` of
    /// `schema`.
    pub(crate) fn concat_length(schema: &Schema) -> Arc<dyn PhysicalExpr> {
        let udf = ScalarUDF::new_from_impl(ConcatLength {
            signature: Signature::exact(vec![DataType::Utf8], Volatility::Immutable),
        });
        Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(udf),
                vec![Arc::new(Column::new_with_schema("s", schema).unwrap())],
                schema,
                Arc::new(datafusion_common::config::ConfigOptions::default()),
            )
            .unwrap(),
        )
    }

    /// A batch of 2048 rows of `x` (`Int32`), alternately `0` and `1`, and of
    /// `s` (`Utf8`), `"a"` but on the first row, which is null, above the
    /// bytes of `"é"`, which are not ASCII.
    pub(crate) fn non_ascii_beneath_a_null() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let x = Int32Array::from_iter_values((0..2048).map(|row| row % 2));
        let values = std::iter::once("é")
            .chain(std::iter::repeat_n("a", 2047))
            .collect::<String>();
        let offsets = arrow::buffer::OffsetBuffer::from_lengths(
            std::iter::once("é".len()).chain(std::iter::repeat_n(1, 2047)),
        );
        let nulls = arrow::buffer::NullBuffer::from_iter((0..2048).map(|row| row != 0));
        let s = StringArray::try_new(offsets, values.into_bytes().into(), Some(nulls))
            .unwrap();
        RecordBatch::try_new(schema, vec![Arc::new(x), Arc::new(s)]).unwrap()
    }

    /// A batch of two columns `a` and `b` of dictionaries of dictionaries
    /// (`Dictionary(Int32, Dictionary(Int32, Utf8))`), which Arrow cannot
    /// compare.
    pub(crate) fn nested_dictionaries() -> RecordBatch {
        let column = || -> ArrayRef {
            let inner = DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![0, 1]),
                Arc::new(StringArray::from(vec!["p", "q"])),
            )
            .unwrap();
            Arc::new(
                DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(vec![0, 1, 0]),
                    Arc::new(inner),
                )
                .unwrap(),
            )
        };
        let (a, b) = (column(), column());
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", a.data_type().clone(), true),
            Field::new("b", b.data_type().clone(), true),
        ]));
        RecordBatch::try_new(schema, vec![a, b]).unwrap()
    }

    /// `x = 0` packed into a dictionary of booleans, on `schema`: Arrow cannot
    /// pack booleans into a dictionary, so this fails whenever it is
    /// evaluated.
    pub(crate) fn packed_booleans(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
            physical_expr("x = 0", schema),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean)),
            None,
        ))
    }

    /// `x = 0 OR (y = 0 OR packed_booleans IS NOT NULL)`, on `schema`.
    pub(crate) fn nested_packed_booleans(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        use datafusion_expr::Operator::Or;
        use datafusion_physical_expr::expressions::{BinaryExpr, IsNotNullExpr};
        let packed = Arc::new(IsNotNullExpr::new(packed_booleans(schema)));
        let inner = Arc::new(BinaryExpr::new(physical_expr("y = 0", schema), Or, packed));
        Arc::new(BinaryExpr::new(physical_expr("x = 0", schema), Or, inner))
    }

    /// Two batches of 512 rows of `x` (`Int32`), `1` on the first 256 rows and
    /// `0` on the others, and of `l`, lists of nulls, empty but on the first
    /// row, a list of `1 << 30` nulls (which take no memory).
    pub(crate) fn long_lists_of_nulls() -> Vec<RecordBatch> {
        let field = Arc::new(Field::new_list_field(DataType::Null, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("l", DataType::List(Arc::clone(&field)), true),
        ]));
        (0..2)
            .map(|_| {
                let x = Int32Array::from_iter_values(
                    (0..512).map(|row| i32::from(row < 256)),
                );
                let offsets = arrow::buffer::OffsetBuffer::from_lengths(
                    std::iter::once(1 << 30).chain(std::iter::repeat_n(0, 511)),
                );
                let nulls = Arc::new(arrow::array::NullArray::new(1 << 30));
                let l = arrow::array::ListArray::try_new(
                    Arc::clone(&field),
                    offsets,
                    nulls,
                    None,
                )
                .unwrap();
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(x), Arc::new(l)])
                    .unwrap()
            })
            .collect()
    }

    /// The plan that executes `operator`, the lowering of `plan` (see
    /// [`lower`]): a `PipelineRootExec`, or if `operator` executes everything
    /// in DataFusion (a `BridgeOperator`), the plan it executes, as a planner
    /// would not run a pipeline for it (and a `PipelineRootExec` checks that
    /// the batches have its schema exactly, which DataFusion's do not always:
    /// Arrow's `filter` renames the fields of run-end encoded arrays).
    pub(crate) fn executable(
        operator: Arc<dyn PhysicalOperator>,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(bridge) = operator.downcast_ref::<BridgeOperator>() {
            return Ok(Arc::clone(bridge.plan()));
        }
        Ok(Arc::new(PipelineRootExec::try_new(operator, plan)?))
    }

    /// The number of rows of `plan`, executed on a task, or its error, or the
    /// message of its panic.
    pub(crate) async fn outcome(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
    ) -> std::result::Result<usize, String> {
        match datafusion_common_runtime::SpawnedTask::spawn(collect(plan, task_ctx))
            .join()
            .await
        {
            Ok(Ok(batches)) => Ok(batches.iter().map(RecordBatch::num_rows).sum()),
            Ok(Err(error)) => Err(error.strip_backtrace()),
            Err(join) if join.is_panic() => {
                let panic = join.into_panic();
                Err(panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default())
            }
            Err(join) => Err(join.to_string()),
        }
    }

    /// Execute `plan` with DataFusion and lowered (see [`lower`]), each on a
    /// task, and assert that it has native filters or projections, and that
    /// their outcome is allowed: the same number of rows as DataFusion, or,
    /// if either engine fails, an error that contains one of `errors`.
    /// Returns the outcome of DataFusion.
    ///
    /// The native operators evaluate the expressions of DataFusion on their
    /// chunks, not on DataFusion's batches: an expression whose errors depend
    /// on the rows it is evaluated on together may fail in one engine and not
    /// in the other, which `docs/compatibility.md` allows (behaviour that
    /// depends on batch sizes); `errors` are the errors that the test expects
    /// in either engine.
    pub(crate) async fn assert_allowed_native_outcome(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
        errors: &[&str],
    ) -> std::result::Result<usize, String> {
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        assert!(
            operators.filters + operators.projections > 0,
            "no native filter or projection"
        );
        let lowered = executable(operator, Arc::clone(&plan)).unwrap();
        let expected = outcome(plan, Arc::clone(&task_ctx)).await;
        let actual = outcome(lowered, task_ctx).await;
        assert_allowed_outcomes(
            expected.as_ref().map(|_| ()).map_err(String::as_str),
            actual.as_ref().map(|_| ()).map_err(String::as_str),
            errors,
        );
        if let (Ok(expected), Ok(actual)) = (&expected, &actual) {
            assert_eq!(actual, expected, "the numbers of rows differ");
        }
        expected
    }

    /// Assert that the outcomes of DataFusion (`expected`) and of the native
    /// operators (`actual`) are allowed: both succeed, or the errors of those
    /// that fail contain one of `errors`.
    fn assert_allowed_outcomes(
        expected: std::result::Result<(), &str>,
        actual: std::result::Result<(), &str>,
        errors: &[&str],
    ) {
        let allowed = |error: &str| errors.iter().any(|message| error.contains(message));
        if let Err(error) = expected {
            assert!(
                allowed(error),
                "DataFusion fails with {error}, expected one of {errors:?}"
            );
        }
        if let Err(error) = actual {
            assert!(
                allowed(error),
                "The native operators fail with {error}, expected one of {errors:?}"
            );
        }
    }

    /// The outcomes of a query or a plan with DataFusion and with its native
    /// operators (see [`lower`]).
    #[derive(Debug)]
    pub(crate) struct NativeOutcome {
        /// The error of DataFusion, if it fails.
        pub(crate) datafusion: Option<String>,
        /// The error of the native operators, if they fail.
        pub(crate) native: Option<String>,
        /// The native operators that were used.
        pub(crate) operators: NativeOperators,
    }

    /// Execute `plan`, created by `make_plan`, with DataFusion and with its
    /// native operators (see [`lower`]), in a session configured with
    /// `config`, and assert that their outcomes are allowed (see
    /// [`assert_allowed_native_outcome`]): byte-identical results (in the
    /// same order if `ordered`) if both succeed, and otherwise errors that
    /// contain one of `errors`.
    pub(crate) async fn assert_allowed_native_plan_outcome_with_config(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        ordered: bool,
        config: SessionConfig,
        errors: &[&str],
    ) -> NativeOutcome {
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let plan = make_plan();
        let expected =
            collect(Arc::clone(&plan), Arc::clone(&task_ctx))
                .await
                .map(|batches| EngineOutput {
                    schema: plan.schema(),
                    batches,
                    plan,
                    ordered,
                });
        let plan = make_plan();
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        let root = executable(operator, plan).unwrap();
        let actual =
            collect(Arc::clone(&root), task_ctx)
                .await
                .map(|batches| EngineOutput {
                    plan: Arc::clone(&root),
                    schema: root.schema(),
                    batches,
                    ordered,
                });
        let message = |error: &DataFusionError| error.find_root().strip_backtrace();
        let (expected_error, actual_error) = (
            expected.as_ref().err().map(message),
            actual.as_ref().err().map(message),
        );
        assert_allowed_outcomes(
            expected_error.as_deref().map_or(Ok(()), Err),
            actual_error.as_deref().map_or(Ok(()), Err),
            errors,
        );
        if let (Ok(expected), Ok(actual)) = (&expected, &actual)
            && let Err(message) = compare_outputs(expected, actual)
        {
            panic!("Results differ: {message}");
        }
        NativeOutcome {
            datafusion: expected_error,
            native: actual_error,
            operators,
        }
    }

    /// [`assert_allowed_native_plan_outcome_with_config`] on 4 threads.
    pub(crate) async fn assert_allowed_native_plan_outcome(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        ordered: bool,
        errors: &[&str],
    ) -> NativeOutcome {
        let config = SessionConfig::new().with_target_partitions(4);
        assert_allowed_native_plan_outcome_with_config(make_plan, ordered, config, errors)
            .await
    }

    /// [`assert_allowed_native_outcome_with_config`] with the configuration
    /// of the tests.
    pub(crate) async fn assert_allowed_native_outcome_with_setup<F>(
        sql: &str,
        setup: F,
        errors: &[&str],
    ) -> NativeOutcome
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_allowed_native_outcome_with_config(
            sql,
            test_session_config(),
            setup,
            errors,
        )
        .await
    }

    /// Run `sql` with DataFusion and with its native operators (see
    /// [`lower`]), in sessions configured with `config` and set up with
    /// `setup`, and assert that their outcomes are allowed (see
    /// [`assert_allowed_native_plan_outcome_with_config`]).
    pub(crate) async fn assert_allowed_native_outcome_with_config<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        errors: &[&str],
    ) -> NativeOutcome
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let config = trusting(config);
        let datafusion = datafusion_context(config.clone());
        setup(&datafusion).await.unwrap();
        let native = datafusion_context(config);
        setup(&native).await.unwrap();
        let expected = run_sql(&datafusion, sql).await;
        let df = native.sql(sql).await.unwrap();
        let ordered = has_top_level_order_by(df.logical_plan());
        let plan = df.create_physical_plan().await.unwrap();
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        let root = executable(operator, plan).unwrap();
        let actual = collect(Arc::clone(&root), native.task_ctx())
            .await
            .map(|batches| EngineOutput {
                plan: Arc::clone(&root) as Arc<dyn ExecutionPlan>,
                schema: root.schema(),
                batches,
                ordered,
            });
        let message = |error: &DataFusionError| error.find_root().strip_backtrace();
        let (expected_error, actual_error) = (
            expected.as_ref().err().map(message),
            actual.as_ref().err().map(message),
        );
        assert_allowed_outcomes(
            expected_error.as_deref().map_or(Ok(()), Err),
            actual_error.as_deref().map_or(Ok(()), Err),
            errors,
        );
        if let (Ok(expected), Ok(actual)) = (&expected, &actual)
            && let Err(message) = compare_outputs(expected, actual)
        {
            panic!("Results differ: {message}\nquery: {sql}");
        }
        NativeOutcome {
            datafusion: expected_error,
            native: actual_error,
            operators,
        }
    }

    /// Execute `plan` with DataFusion and lowered (see [`lower`]), each on a
    /// task, and assert that the outcomes are the same, with no native filter
    /// or projection.
    pub(crate) async fn assert_same_outcome_in_datafusion(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
    ) -> std::result::Result<usize, String> {
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        assert_eq!((operators.filters, operators.projections), (0, 0));
        let lowered = executable(operator, Arc::clone(&plan)).unwrap();
        let expected = outcome(plan, Arc::clone(&task_ctx)).await;
        assert_eq!(outcome(lowered, task_ctx).await, expected);
        expected
    }

    /// A batch of 4096 rows of `x` (`Boolean`), `true` on the first 2048 rows,
    /// and of `a` and `b`, lists of nulls (which take no memory), empty but a
    /// list of `1 << 30` nulls on the first row (`a`) and on the 2049th row
    /// (`b`).
    pub(crate) fn lists_to_merge() -> RecordBatch {
        let field = Arc::new(Field::new_list_field(DataType::Null, true));
        let list = |row: usize| -> ArrayRef {
            let offsets = arrow::buffer::OffsetBuffer::from_lengths(
                (0..4096).map(|other| if other == row { 1 << 30 } else { 0 }),
            );
            let nulls = Arc::new(arrow::array::NullArray::new(1 << 30));
            Arc::new(
                arrow::array::ListArray::try_new(
                    Arc::clone(&field),
                    offsets,
                    nulls,
                    None,
                )
                .unwrap(),
            )
        };
        let x =
            arrow::array::BooleanArray::from_iter((0..4096).map(|row| Some(row < 2048)));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Boolean, true),
            Field::new("a", DataType::List(Arc::clone(&field)), true),
            Field::new("b", DataType::List(Arc::clone(&field)), true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(x), list(0), list(2048)]).unwrap()
    }

    /// Values that take less than a byte each.
    #[derive(Debug, Clone, Copy)]
    pub(crate) enum TinyValues {
        /// Nulls.
        Nulls,
        /// One run of `42` (`RunEndEncoded<Int64, Int32>`).
        Runs,
        /// Fixed-size lists of no `Int32`.
        EmptyLists,
        /// Fixed-size binaries of no byte.
        EmptyBinaries,
        /// `false`, in bits.
        Booleans,
    }

    /// A batch of the rows `x` (`Int32`) and of `f`, fixed-size lists of
    /// `1 << 30` `values`, which take less than a byte each (no memory, but
    /// for booleans).
    pub(crate) fn wide_lists(x: &[i32], values: TinyValues) -> RecordBatch {
        let len = x.len() << 30;
        let child: ArrayRef = match values {
            TinyValues::Nulls => Arc::new(arrow::array::NullArray::new(len)),
            TinyValues::Runs => Arc::new(
                arrow::array::RunArray::<arrow::datatypes::Int64Type>::try_new(
                    &Int64Array::from(vec![len as i64]),
                    &Int32Array::from(vec![42]),
                )
                .unwrap(),
            ),
            TinyValues::EmptyLists => Arc::new(
                arrow::array::FixedSizeListArray::try_new_with_length(
                    Arc::new(Field::new_list_field(DataType::Int32, true)),
                    0,
                    Arc::new(Int32Array::from(Vec::<i32>::new())),
                    None,
                    len,
                )
                .unwrap(),
            ),
            TinyValues::EmptyBinaries => Arc::new(
                arrow::array::FixedSizeBinaryArray::try_new_with_len(
                    0,
                    arrow::buffer::Buffer::from(Vec::<u8>::new()),
                    None,
                    len,
                )
                .unwrap(),
            ),
            TinyValues::Booleans => Arc::new(arrow::array::BooleanArray::new(
                arrow::buffer::BooleanBuffer::new_unset(len),
                None,
            )),
        };
        let field = Arc::new(Field::new_list_field(child.data_type().clone(), true));
        let f = arrow::array::FixedSizeListArray::try_new(
            Arc::clone(&field),
            1 << 30,
            child,
            None,
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("f", DataType::FixedSizeList(field, 1 << 30), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(x.to_vec())), Arc::new(f)],
        )
        .unwrap()
    }

    /// A batch of 8 rows of `x` (`Int32`), alternately `1` and `0`, and of `r`,
    /// run-end encoded values whose child fields have other names than
    /// Arrow's (`ends` and `samples`).
    pub(crate) fn runs_with_named_fields() -> RecordBatch {
        let ends = Int32Array::from(vec![2, 4, 6, 8]);
        let samples = Int32Array::from(vec![1, 2, 3, 4]);
        let data_type = DataType::RunEndEncoded(
            Arc::new(Field::new("ends", DataType::Int32, false)),
            Arc::new(Field::new("samples", DataType::Int32, true)),
        );
        let data = arrow::array::ArrayData::try_new(
            data_type.clone(),
            8,
            None,
            0,
            vec![],
            vec![
                arrow::array::Array::to_data(&ends),
                arrow::array::Array::to_data(&samples),
            ],
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("r", data_type, true),
        ]));
        let x = Int32Array::from_iter_values((0..8).map(|row| i32::from(row % 2 == 0)));
        RecordBatch::try_new(schema, vec![Arc::new(x), arrow::array::make_array(data)])
            .unwrap()
    }

    /// A constant fixed-size list of 8 values, one run of `42` with 16-bit run
    /// ends (`RunEndEncoded<Int16, Int32>`): broadcast to more than 4095 rows,
    /// its values overflow the run ends.
    pub(crate) fn constant_list_of_short_runs() -> Arc<dyn PhysicalExpr> {
        let runs = arrow::array::RunArray::<arrow::datatypes::Int16Type>::try_new(
            &arrow::array::Int16Array::from(vec![8]),
            &Int32Array::from(vec![42]),
        )
        .unwrap();
        let data_type = arrow::array::Array::data_type(&runs).clone();
        let field = Arc::new(Field::new_list_field(data_type, true));
        let list =
            arrow::array::FixedSizeListArray::try_new(field, 8, Arc::new(runs), None)
                .unwrap();
        Arc::new(datafusion_physical_expr::expressions::Literal::new(
            datafusion_common::ScalarValue::FixedSizeList(Arc::new(list)),
        ))
    }

    /// A constant list of `1 << 19` nulls.
    pub(crate) fn constant_list_of_nulls() -> Arc<dyn PhysicalExpr> {
        let field = Arc::new(Field::new_list_field(DataType::Null, true));
        let list = arrow::array::ListArray::try_new(
            field,
            arrow::buffer::OffsetBuffer::from_lengths([1 << 19]),
            Arc::new(arrow::array::NullArray::new(1 << 19)),
            None,
        )
        .unwrap();
        Arc::new(datafusion_physical_expr::expressions::Literal::new(
            datafusion_common::ScalarValue::List(Arc::new(list)),
        ))
    }

    /// Two batches of a dictionary column `d` (`Int32` keys) with two rows,
    /// `"1"` and `"x"`, then `"2"` and `"y"`, with distinct dictionaries.
    pub(crate) fn dictionary_batches() -> Vec<RecordBatch> {
        let data_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![Field::new("d", data_type, true)]));
        [["1", "x"], ["2", "y"]]
            .into_iter()
            .map(|values| {
                let d = DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(vec![0, 1]),
                    Arc::new(StringArray::from(values.to_vec())),
                )
                .unwrap();
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(d)]).unwrap()
            })
            .collect()
    }

    /// A batch of 4096 rows of `id` (`Int32`, the row number) and of `u`, a
    /// dense union of a string of 2 MiB and of integers: 600 rows of each half
    /// of the batch (2048 rows) point to the string.
    pub(crate) fn dense_union_of_a_large_string() -> RecordBatch {
        let rows = 4096;
        let big = |row: i32| row % 2048 < 600;
        let type_ids = (0..rows).map(|row| i8::from(!big(row))).collect();
        let mut integers = 0;
        let offsets = (0..rows)
            .map(|row| {
                if big(row) {
                    0
                } else {
                    integers += 1;
                    integers - 1
                }
            })
            .collect();
        let fields = arrow::datatypes::UnionFields::try_new(
            [0, 1],
            [
                Field::new("s", DataType::Utf8, true),
                Field::new("i", DataType::Int32, true),
            ],
        )
        .unwrap();
        let u = arrow::array::UnionArray::try_new(
            fields,
            type_ids,
            Some(offsets),
            vec![
                Arc::new(StringArray::from(vec!["x".repeat(2 << 20)])),
                Arc::new(Int32Array::from_iter_values(0..rows)),
            ],
        )
        .unwrap();
        RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(0..rows)) as ArrayRef,
            ),
            ("u", Arc::new(u) as ArrayRef),
        ])
        .unwrap()
    }

    /// A guard that each test taking gigabytes of memory holds while it runs,
    /// so that only one runs at a time: the tests that run in parallel do not
    /// take the sum.
    pub(crate) async fn one_large_test_at_a_time() -> tokio::sync::MutexGuard<'static, ()>
    {
        static LARGE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        LARGE.lock().await
    }

    /// A batch of 4096 rows of `b` (`Boolean`), `true` on the first 2048 rows
    /// and `false` on the others, and of `u`, a dense union of a single field
    /// of integers that is not nullable: a union without nulls.
    pub(crate) fn unions_without_nulls() -> RecordBatch {
        let b =
            arrow::array::BooleanArray::from_iter((0..4096).map(|row| Some(row < 2048)));
        let fields = arrow::datatypes::UnionFields::try_new(
            [0],
            [Field::new("i", DataType::Int32, false)],
        )
        .unwrap();
        let u = arrow::array::UnionArray::try_new(
            fields,
            vec![0_i8; 4096].into(),
            Some((0..4096).collect()),
            vec![Arc::new(Int32Array::from_iter_values(0..4096))],
        )
        .unwrap();
        RecordBatch::try_from_iter([
            ("b", Arc::new(b) as ArrayRef),
            ("u", Arc::new(u) as ArrayRef),
        ])
        .unwrap()
    }

    /// `CASE b WHEN true THEN u END` (see [`unions_without_nulls`]): `CASE`
    /// merges `u` with nulls for the other rows, which Arrow cannot do with a
    /// union without nullable fields.
    pub(crate) fn case_of_unions(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        Arc::new(
            datafusion_physical_expr::expressions::CaseExpr::try_new(
                Some(physical_expr("b", schema)),
                vec![(physical_expr("true", schema), physical_expr("u", schema))],
                None,
            )
            .unwrap(),
        )
    }

    /// One partition of two batches of 2048 rows, `(x, s) = (0, "1")` then
    /// `(0, "invalid")`, projected by `x, CAST(s AS BIGINT) AS c, length(s) AS
    /// l`, and filtered by `x >= 0` (whose `FilterExec` coalesces batches of
    /// 8192 rows).
    pub(crate) fn filtered_casts() -> Arc<dyn ExecutionPlan> {
        let batches = [xss(&[((0, "1"), 2048)]), xss(&[((0, "invalid"), 2048)])];
        let schema = batches[0].schema();
        let projection = Arc::new(
            ProjectionExec::try_new(
                [("x", "x"), ("CAST(s AS BIGINT)", "c"), ("length(s)", "l")].map(
                    |(expr, name)| {
                        datafusion_physical_expr::projection::ProjectionExpr::new(
                            physical_expr(expr, &schema),
                            name,
                        )
                    },
                ),
                scan_batches(&schema, &batches),
            )
            .unwrap(),
        );
        let predicate = physical_expr("x >= 0", &projection.schema());
        Arc::new(FilterExec::try_new(predicate, projection).unwrap())
    }

    /// A batch of 4096 rows of `x` (`Boolean`), `true` on the first 2048 rows
    /// and `false` on the others, and of `s` and `p` (`Utf8View`), `"a"`.
    pub(crate) fn guarded_patterns() -> RecordBatch {
        let x =
            arrow::array::BooleanArray::from_iter((0..4096).map(|row| Some(row < 2048)));
        let a = || Arc::new(StringViewArray::from(vec!["a"; 4096])) as ArrayRef;
        RecordBatch::try_from_iter([
            ("x", Arc::new(x) as ArrayRef),
            ("s", a()),
            ("p", a()),
        ])
        .unwrap()
    }

    /// `s ILIKE CASE WHEN x THEN 'sss…' ELSE p END`, with 100000 `s` (see
    /// [`guarded_patterns`]): `CASE` returns the pattern as is (a scalar) if
    /// all rows take its branch, which Arrow matches with the ASCII strings
    /// without compiling it, but compiles it in an array of patterns, into a
    /// regular expression that is too big.
    pub(crate) fn guarded_pattern_match(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        let pattern = datafusion_physical_expr::expressions::CaseExpr::try_new(
            None,
            vec![(
                physical_expr("x", schema),
                Arc::new(datafusion_physical_expr::expressions::Literal::new(
                    datafusion_common::ScalarValue::Utf8View(Some("s".repeat(100_000))),
                )),
            )],
            Some(physical_expr("p", schema)),
        )
        .unwrap();
        Arc::new(datafusion_physical_expr::expressions::LikeExpr::new(
            false,
            true,
            physical_expr("s", schema),
            Arc::new(pattern),
        ))
    }

    /// A batch of 4096 rows of `x` (`Boolean`), `false` on the first 2048
    /// rows and `true` on the others, and of `y` (`Int64`), `1`.
    pub(crate) fn guarded_minimums() -> RecordBatch {
        let x =
            arrow::array::BooleanArray::from_iter((0..4096).map(|row| Some(row >= 2048)));
        RecordBatch::try_from_iter([
            ("x", Arc::new(x) as ArrayRef),
            ("y", Arc::new(Int64Array::from(vec![1; 4096])) as ArrayRef),
        ])
        .unwrap()
    }

    /// `-(CASE WHEN x THEN y ELSE -9223372036854775808 END)` (see
    /// [`guarded_minimums`]): `CASE` returns its `ELSE` value as is if all
    /// rows take it, a scalar, which `-` negates with overflow checks, but an
    /// array, which `-` negates wrapping, otherwise.
    pub(crate) fn negated_case(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        let case = datafusion_physical_expr::expressions::CaseExpr::try_new(
            None,
            vec![(physical_expr("x", schema), physical_expr("y", schema))],
            Some(Arc::new(
                datafusion_physical_expr::expressions::Literal::new(
                    datafusion_common::ScalarValue::Int64(Some(i64::MIN)),
                ),
            )),
        )
        .unwrap();
        Arc::new(datafusion_physical_expr::expressions::NegativeExpr::new(
            Arc::new(case),
        ))
    }

    /// A batch of 8192 rows of `id` (`Int32`) and of `s` (`Utf8`), strings of
    /// 128 KiB: 1 GiB of strings, and 256 MiB on a chunk of 2048 rows.
    pub(crate) fn large_strings() -> RecordBatch {
        let string = "x".repeat(128 << 10);
        RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(0..8192)) as ArrayRef,
            ),
            (
                "s",
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    &string, 8192,
                ))) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    /// A batch of 8192 rows of `s` (`LargeUtf8`), `"x"`, and of `n` (`Int64`),
    /// `1 << 19`: `repeat(s, n)` builds 4 GiB of strings on the batch, and
    /// 1 GiB on a chunk of 2048 rows.
    pub(crate) fn repeated_strings() -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "s",
                Arc::new(arrow::array::LargeStringArray::from(vec!["x"; 8192]))
                    as ArrayRef,
            ),
            (
                "n",
                Arc::new(Int64Array::from(vec![1 << 19; 8192])) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    /// `predicate` (on `schema`) in a `DynamicFilterPhysicalExpr`, whose
    /// children are the columns that `predicate` reads (as DataFusion builds
    /// them), not `predicate`.
    pub(crate) fn dynamic_filter(
        predicate: &str,
        schema: &SchemaRef,
    ) -> Arc<dyn PhysicalExpr> {
        let inner = physical_expr(predicate, schema);
        let children = datafusion_physical_expr::utils::collect_columns(&inner)
            .into_iter()
            .map(|column| Arc::new(column) as Arc<dyn PhysicalExpr>)
            .collect();
        Arc::new(
            datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr::new(
                children, inner,
            ),
        )
    }

    /// The types of times that Arrow has no arrays of: of microseconds or
    /// nanoseconds in 32 bits, of seconds or milliseconds in 64 bits.
    pub(crate) fn times_without_arrays() -> Vec<DataType> {
        use arrow::datatypes::TimeUnit;
        vec![
            DataType::Time32(TimeUnit::Microsecond),
            DataType::Time32(TimeUnit::Nanosecond),
            DataType::Time64(TimeUnit::Second),
            DataType::Time64(TimeUnit::Millisecond),
        ]
    }

    /// A batch of 4096 rows of `x` (`Boolean`): `true` on the first 2048 rows,
    /// `false` on the others.
    pub(crate) fn half_true() -> RecordBatch {
        let x =
            arrow::array::BooleanArray::from_iter((0..4096).map(|row| Some(row < 2048)));
        RecordBatch::try_from_iter([("x", Arc::new(x) as ArrayRef)]).unwrap()
    }

    /// `(CASE WHEN x THEN '…' END) IS NULL`, with a string of 2 MiB (see
    /// [`half_true`]).
    pub(crate) fn case_of_a_large_constant(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        let constant = Arc::new(datafusion_physical_expr::expressions::Literal::new(
            datafusion_common::ScalarValue::Utf8(Some("x".repeat(2 << 20))),
        ));
        let case = datafusion_physical_expr::expressions::CaseExpr::try_new(
            None,
            vec![(physical_expr("x", schema), constant)],
            None,
        )
        .unwrap();
        Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(
            Arc::new(case),
        ))
    }

    /// A batch of `rows` rows of `id` (`Int32`, the row number) and of `d`, a
    /// dictionary (`Int32` keys) of a single string of 2 MiB: the keys of the
    /// first `nulls` rows are null, the others `0`.
    pub(crate) fn repeated_large_string(rows: i32, nulls: i32) -> RecordBatch {
        let keys =
            Int32Array::from_iter((0..rows).map(|row| (row >= nulls).then_some(0)));
        let d = DictionaryArray::<Int32Type>::try_new(
            keys,
            Arc::new(StringArray::from(vec!["x".repeat(2 << 20)])),
        )
        .unwrap();
        RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(0..rows)) as ArrayRef,
            ),
            ("d", Arc::new(d) as ArrayRef),
        ])
        .unwrap()
    }

    /// `id < bound AND CAST(d AS Utf8) IS NULL` (see [`repeated_large_string`]).
    pub(crate) fn guarded_unpacking(
        schema: &SchemaRef,
        bound: i32,
    ) -> Arc<dyn PhysicalExpr> {
        let cast = Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
            physical_expr("d", schema),
            DataType::Utf8,
            None,
        ));
        Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
            physical_expr(&format!("id < {bound}"), schema),
            datafusion_expr::Operator::And,
            Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(cast)),
        ))
    }

    /// A batch of 4096 rows of `x` (`Int32`) and of `d`, a dictionary of
    /// `(1 << 31) + 1` nulls (which take no memory) with `UInt64` keys: the
    /// first 2049 rows are `0` and the key `0`, the others `1` and the key
    /// `1 << 31`, which `Int32` keys cannot hold.
    pub(crate) fn large_dictionary_keys() -> RecordBatch {
        let rows = 4096;
        let guarded = 2049;
        let keys = arrow::array::UInt64Array::from_iter_values(
            (0..rows).map(|row| if row < guarded { 0 } else { 1 << 31 }),
        );
        let d = DictionaryArray::<arrow::datatypes::UInt64Type>::try_new(
            keys,
            Arc::new(arrow::array::NullArray::new((1 << 31) + 1)),
        )
        .unwrap();
        let x =
            Int32Array::from_iter_values((0..rows).map(|row| i32::from(row >= guarded)));
        RecordBatch::try_from_iter([
            ("x", Arc::new(x) as ArrayRef),
            ("d", Arc::new(d) as ArrayRef),
        ])
        .unwrap()
    }

    /// `x = 0 AND CAST(d AS Dictionary(Int32, Null)) IS NULL` (see
    /// [`large_dictionary_keys`]).
    pub(crate) fn guarded_key_cast(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        let cast = Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
            physical_expr("d", schema),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Null)),
            None,
        ));
        Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
            physical_expr("x = 0", schema),
            datafusion_expr::Operator::And,
            Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(cast)),
        ))
    }

    /// Two batches of 100 rows of a dictionary column `d`, with `keys`, and of
    /// the row number `i`: the 100 values of the first dictionary (`a0`, `a1`,
    /// …) are all distinct from those of the second one (`b0`, …).
    pub(crate) fn distinct_dictionary_batches(keys: &DataType) -> Vec<RecordBatch> {
        let data_type =
            DataType::Dictionary(Box::new(keys.clone()), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", data_type.clone(), true),
            Field::new("i", DataType::Int32, false),
        ]));
        ["a", "b"]
            .into_iter()
            .zip([0, 100])
            .map(|(prefix, first_row)| {
                let values = StringArray::from_iter_values(
                    (0..100).map(|v| format!("{prefix}{v}")),
                );
                let d = DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from_iter_values(0..100),
                    Arc::new(values),
                )
                .unwrap();
                let d = arrow::compute::cast(&d, &data_type).unwrap();
                let i = Int32Array::from_iter_values(first_row..first_row + 100);
                RecordBatch::try_new(Arc::clone(&schema), vec![d, Arc::new(i)]).unwrap()
            })
            .collect()
    }

    /// A scan of `batches` (of `schema`, possibly empty or none), in a single
    /// partition.
    pub(crate) fn scan_batches(
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Arc<dyn ExecutionPlan> {
        MemorySourceConfig::try_new_exec(&[batches.to_vec()], Arc::clone(schema), None)
            .unwrap()
    }

    /// Register `batches` as table `name`, with one partition per entry.
    pub(crate) fn register_partitions(
        ctx: &SessionContext,
        name: &str,
        partitions: Vec<Vec<RecordBatch>>,
    ) -> Result<()> {
        let schema = partitions
            .iter()
            .flatten()
            .next()
            .expect("at least one batch")
            .schema();
        ctx.register_table(name, Arc::new(MemTable::try_new(schema, partitions)?))?;
        Ok(())
    }

    /// Register the default test tables and:
    ///
    /// * `fallible`: 6000 rows in 3 partitions with `id BIGINT NOT NULL`,
    ///   `x INT` (90% of the rows are `0`, 2% are `NULL`), `s VARCHAR` (10%
    ///   of the rows are numbers) and `big BIGINT` (values that overflow an
    ///   `INT`, except on 10% of the rows). Guards like `x <> 0` select few
    ///   enough rows (at most 20%) for DataFusion to evaluate the conjuncts
    ///   after them only on the selected rows, so that DataFusion does not
    ///   fail on such queries either.
    /// * `vd`: 5000 rows in 3 partitions with `i BIGINT NOT NULL`, `sv`
    ///   (`Utf8View`), `d32` (`Dictionary(Int32, Utf8)`) and `d8`
    ///   (`Dictionary(Int8, Utf8)`), with nulls and different dictionaries
    ///   in every batch.
    pub(crate) fn register_all_tables(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)?;
        register_partitions(ctx, "fallible", fallible_table()?)?;
        register_partitions(ctx, "vd", view_dictionary_table()?)
    }

    /// [`register_all_tables`], as the setup of the differential tests.
    pub(crate) async fn all_tables(ctx: &SessionContext) -> Result<()> {
        register_all_tables(ctx)
    }

    /// The partitions of table `fallible` of [`register_all_tables`].
    pub(crate) fn fallible_table() -> Result<Vec<Vec<RecordBatch>>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("big", DataType::Int64, true),
        ]));
        let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let x = Int32Array::from_iter(ids.clone().map(|i| match i {
                _ if i % 50 == 7 => None,
                _ if i % 10 == 3 => Some(match (i % 23) as i32 - 11 {
                    0 => 7,
                    v => v,
                }),
                _ => Some(0),
            }));
            let s = StringArray::from_iter(ids.clone().map(|i| match i {
                _ if i % 31 == 5 => None,
                _ if i % 10 == 0 => Some(format!("{}", i % 1000)),
                _ => Some(format!("n/a {i}")),
            }));
            let big = Int64Array::from_iter(ids.clone().map(|i| {
                (i % 37 != 2).then_some(if i % 10 == 0 { i } else { i * 10_000_000_000 })
            }));
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids)),
                    Arc::new(x),
                    Arc::new(s),
                    Arc::new(big),
                ],
            )?)
        };
        Ok(vec![
            vec![batch(0..2000)?],
            vec![batch(2000..2500)?, batch(2500..5000)?],
            vec![batch(5000..6000)?],
        ])
    }

    /// The partitions of table `vd` of [`register_all_tables`].
    pub(crate) fn view_dictionary_table() -> Result<Vec<Vec<RecordBatch>>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, false),
            Field::new("sv", DataType::Utf8View, true),
            Field::new(
                "d32",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new(
                "d8",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let sv = StringViewArray::from_iter(ids.clone().map(|i| {
                (i % 17 != 4).then(|| {
                    if i % 2 == 0 {
                        format!("short{}", i % 9)
                    } else {
                        format!("a string that is longer than twelve bytes {}", i % 300)
                    }
                })
            }));
            // Every batch has its own dictionaries.
            let d32 = ids
                .clone()
                .map(|i| (i % 13 != 6).then(|| format!("k{}", i % 40)))
                .collect::<Vec<_>>();
            let d32 = d32
                .iter()
                .map(Option::as_deref)
                .collect::<DictionaryArray<Int32Type>>();
            let d8 = ids
                .clone()
                .map(|i| (i % 19 != 8).then(|| format!("x{}", (i * 7) % 90)))
                .collect::<Vec<_>>();
            let d8 = d8
                .iter()
                .map(Option::as_deref)
                .collect::<DictionaryArray<Int8Type>>();
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids)),
                    Arc::new(sv),
                    Arc::new(d32),
                    Arc::new(d8),
                ],
            )?)
        };
        Ok(vec![
            vec![batch(0..2100)?, batch(2100..2200)?],
            vec![batch(2200..4000)?],
            vec![batch(4000..5000)?],
        ])
    }

    /// A table `name` of 3 partitions with 4 batches of 100 rows each, with
    /// the columns `id` and `x`, `y`, `a`, `b` (`INT`, nullable) given by
    /// `row(id)`. The batches are not split into several chunks, so the
    /// filter decides like DataFusion for the same rows.
    pub(crate) fn register_guard_table(
        ctx: &SessionContext,
        name: &str,
        row: impl Fn(i64) -> [Option<i32>; 4],
    ) -> Result<()> {
        let schema = Arc::new(Schema::new(
            std::iter::once(Field::new("id", DataType::Int64, false))
                .chain(
                    ["x", "y", "a", "b"]
                        .into_iter()
                        .map(|name| Field::new(name, DataType::Int32, true)),
                )
                .collect::<Vec<_>>(),
        ));
        let batch = |batch: i64| {
            let ids = batch * 100..(batch + 1) * 100;
            let rows = ids.clone().map(&row).collect::<Vec<_>>();
            let mut columns: Vec<ArrayRef> =
                vec![Arc::new(Int64Array::from_iter_values(ids))];
            for column in 0..4 {
                columns.push(Arc::new(Int32Array::from_iter(
                    rows.iter().map(|row| row[column]),
                )));
            }
            RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
        };
        register_partitions(
            ctx,
            name,
            vec![vec![batch(0), batch(1)], vec![batch(2)], vec![batch(3)]],
        )
    }

    /// Register the default test tables and three tables of TPC-H's schema
    /// (`DECIMAL(15, 2)` prices, `DATE` dates, `Utf8View` strings, as in
    /// DataFusion's SQL tables and Parquet scans), with nulls, in skewed
    /// partitions (one of them empty) of batches of very different sizes:
    ///
    /// * `lineitem`: 6000 rows of `l_orderkey`, `l_quantity`,
    ///   `l_extendedprice`, `l_discount`, `l_tax`, `l_returnflag`,
    ///   `l_linestatus`, `l_shipdate` and `l_shipmode`;
    /// * `orders`: 3000 rows of `o_orderkey`, `o_custkey`, `o_orderdate`,
    ///   `o_orderpriority`, `o_totalprice` and `o_comment` (`Utf8`);
    /// * `customer`: 1500 rows of `c_custkey`, `c_phone` (`NN-NNN-NNN-NNNN`,
    ///   whose country codes are 10 to 34), `c_acctbal` (also negative) and
    ///   `c_mktsegment`.
    pub(crate) async fn tpch_tables(ctx: &SessionContext) -> Result<()> {
        register_all_tables(ctx)?;
        use arrow::array::{Date32Array, Decimal128Array};
        let decimal = DataType::Decimal128(15, 2);
        let decimals = |values: Vec<Option<i64>>| -> Result<ArrayRef> {
            Ok(Arc::new(
                Decimal128Array::from_iter(values.into_iter().map(|v| v.map(i128::from)))
                    .with_precision_and_scale(15, 2)?,
            ))
        };
        let views = |values: Vec<Option<String>>| -> ArrayRef {
            Arc::new(StringViewArray::from_iter(values))
        };
        // 1992-01-01 plus up to 7 years.
        let date = |i: i64| 8035 + ((i * 7919) % 2557) as i32;
        let skewed = |batch: &dyn Fn(std::ops::Range<i64>) -> Result<RecordBatch>,
                      n: i64| {
            Ok::<_, DataFusionError>(vec![
                vec![batch(0..n / 2)?, batch(n / 2..n / 2 + 7)?],
                vec![],
                vec![batch(n / 2 + 7..n / 2 + 8)?, batch(n / 2 + 8..n)?],
            ])
        };

        let lineitem = Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("l_quantity", decimal.clone(), false),
            Field::new("l_extendedprice", decimal.clone(), false),
            Field::new("l_discount", decimal.clone(), true),
            Field::new("l_tax", decimal.clone(), true),
            Field::new("l_returnflag", DataType::Utf8View, false),
            Field::new("l_linestatus", DataType::Utf8View, false),
            Field::new("l_shipdate", DataType::Date32, true),
            Field::new("l_shipmode", DataType::Utf8View, true),
        ]));
        let lineitem_batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let modes = ["AIR", "MAIL", "SHIP", "TRUCK", "REG AIR", "RAIL", "FOB"];
            let flags = ["A", "N", "R"];
            let columns = vec![
                Arc::new(Int64Array::from_iter_values(ids.clone().map(|i| i / 4)))
                    as ArrayRef,
                decimals(ids.clone().map(|i| Some(100 + (i * 37) % 5000)).collect())?,
                decimals(
                    ids.clone()
                        .map(|i| Some(90_000 + (i * 7_919) % 10_000_000))
                        .collect(),
                )?,
                decimals(
                    ids.clone()
                        .map(|i| (i % 23 != 5).then_some(i % 11))
                        .collect(),
                )?,
                decimals(
                    ids.clone()
                        .map(|i| (i % 29 != 8).then_some(i % 9))
                        .collect(),
                )?,
                views(
                    ids.clone()
                        .map(|i| Some(flags[(i % 3) as usize].to_owned()))
                        .collect(),
                ),
                views(
                    ids.clone()
                        .map(|i| Some(if i % 5 < 2 { "O" } else { "F" }.to_owned()))
                        .collect(),
                ),
                Arc::new(Date32Array::from_iter(
                    ids.clone().map(|i| (i % 31 != 3).then(|| date(i))),
                )),
                views(
                    ids.map(|i| {
                        (i % 41 != 6).then(|| modes[(i % 7) as usize].to_owned())
                    })
                    .collect(),
                ),
            ];
            Ok(RecordBatch::try_new(Arc::clone(&lineitem), columns)?)
        };
        register_partitions(ctx, "lineitem", skewed(&lineitem_batch, 6000)?)?;

        let orders = Arc::new(Schema::new(vec![
            Field::new("o_orderkey", DataType::Int64, false),
            Field::new("o_custkey", DataType::Int64, true),
            Field::new("o_orderdate", DataType::Date32, false),
            Field::new("o_orderpriority", DataType::Utf8View, false),
            Field::new("o_totalprice", decimal.clone(), true),
            Field::new("o_comment", DataType::Utf8, true),
        ]));
        let orders_batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let priorities =
                ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"];
            let columns = vec![
                Arc::new(Int64Array::from_iter_values(ids.clone())) as ArrayRef,
                Arc::new(Int64Array::from_iter(
                    ids.clone()
                        .map(|i| (i % 37 != 2).then_some((i * 13) % 1500)),
                )),
                Arc::new(Date32Array::from_iter_values(ids.clone().map(date))),
                views(
                    ids.clone()
                        .map(|i| Some(priorities[((i * 3) % 5) as usize].to_owned()))
                        .collect(),
                ),
                decimals(
                    ids.clone()
                        .map(|i| (i % 43 != 1).then_some((i * 104_729) % 50_000_000))
                        .collect(),
                )?,
                Arc::new(StringArray::from_iter(ids.map(|i| {
                    (i % 19 != 4).then(|| format!("comment {} special requests", i % 97))
                }))),
            ];
            Ok(RecordBatch::try_new(Arc::clone(&orders), columns)?)
        };
        register_partitions(ctx, "orders", skewed(&orders_batch, 3000)?)?;

        let customer = Arc::new(Schema::new(vec![
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("c_phone", DataType::Utf8View, true),
            Field::new("c_acctbal", decimal, true),
            Field::new("c_mktsegment", DataType::Utf8View, false),
        ]));
        let customer_batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let segments = [
                "AUTOMOBILE",
                "BUILDING",
                "FURNITURE",
                "MACHINERY",
                "HOUSEHOLD",
            ];
            let columns = vec![
                Arc::new(Int64Array::from_iter_values(ids.clone())) as ArrayRef,
                views(
                    ids.clone()
                        .map(|i| {
                            (i % 17 != 9).then(|| {
                                format!(
                                    "{}-{:03}-{:03}-{:04}",
                                    10 + i % 25,
                                    (i * 7) % 1000,
                                    (i * 11) % 1000,
                                    (i * 13) % 10_000
                                )
                            })
                        })
                        .collect(),
                ),
                decimals(
                    ids.clone()
                        .map(|i| {
                            (i % 13 != 7).then_some((i * 7_907) % 1_100_000 - 100_000)
                        })
                        .collect(),
                )?,
                views(
                    ids.map(|i| Some(segments[(i % 5) as usize].to_owned()))
                        .collect(),
                ),
            ];
            Ok(RecordBatch::try_new(Arc::clone(&customer), columns)?)
        };
        register_partitions(ctx, "customer", skewed(&customer_batch, 1500)?)
    }

    /// Assert that every `ProjectionExec` and `FilterExec` of `plan` (at any
    /// depth) can be lowered (see [`PhysicalProjection::supports_projection_exec`]
    /// and [`PhysicalFilter::supports_filter_exec`]), and return their numbers.
    pub(crate) fn assert_all_lowerable(plan: &Arc<dyn ExecutionPlan>) -> (usize, usize) {
        let (mut projections, mut filters) = (0, 0);
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            assert!(
                PhysicalProjection::supports_projection_exec(projection),
                "not supported: {}",
                datafusion_physical_plan::displayable(projection).one_line()
            );
            projections += 1;
        }
        if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            assert!(
                PhysicalFilter::supports_filter_exec(filter),
                "not supported: {}",
                datafusion_physical_plan::displayable(filter).one_line()
            );
            filters += 1;
        }
        for child in plan.children() {
            let (child_projections, child_filters) = assert_all_lowerable(child);
            projections += child_projections;
            filters += child_filters;
        }
        (projections, filters)
    }

    /// The tables of the error parity tests: [`register_all_tables`] and the
    /// guard tables `dense`, `sparse`, `nulls` and `nulls_ok`.
    pub(crate) async fn guard_tables(ctx: &SessionContext) -> Result<()> {
        register_all_tables(ctx)?;
        // Half of the rows have `x = 0`; `y = 1` on 2% of the rows, and only
        // where `x <> 0`.
        register_guard_table(ctx, "dense", |i| {
            let x = if i % 2 == 0 { 0 } else { 1 + (i % 7) as i32 };
            [Some(x), Some(i32::from(i % 50 == 1)), None, None]
        })?;
        // 90% of the rows have `x = 0`.
        register_guard_table(ctx, "sparse", |i| {
            let x = if i % 10 == 3 { 1 + (i % 4) as i32 } else { 0 };
            [Some(x), None, None, None]
        })?;
        // `a > 0` is `NULL` on 5% of the rows and `false` on the others; `b
        // = 0` on the `NULL` rows of `nulls`, but not of `nulls_ok`.
        register_guard_table(ctx, "nulls", |i| {
            let null = i % 20 == 7;
            [None, None, (!null).then_some(-1), Some(i32::from(!null))]
        })?;
        register_guard_table(ctx, "nulls_ok", |i| {
            let null = i % 20 == 7;
            [None, None, (!null).then_some(-1), Some(2)]
        })
    }

    /// A source that emits `batches` (in one partition, so in order).
    pub(crate) fn source(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Arc<dyn PhysicalOperator> {
        let plan = MemorySourceConfig::try_new_exec(&[batches], schema, None)
            .expect("valid memory source");
        Arc::new(BridgeOperator::new(plan))
    }

    /// A context to call operators directly.
    pub(crate) fn context() -> ExecutionContext {
        ExecutionContext::new(Arc::new(TaskContext::default()), 0, 1)
    }

    /// The states of `operator` for one thread.
    pub(crate) fn states(
        operator: &dyn PhysicalOperator,
    ) -> (Arc<dyn GlobalOperatorState>, Box<dyn OperatorState>) {
        let ctx = context();
        let gstate = operator.get_global_operator_state(&ctx).unwrap();
        let lstate = operator.get_operator_state(&ctx, gstate.as_ref()).unwrap();
        (gstate, lstate)
    }

    /// Push `inputs` through `operator` like a pipeline executor with a
    /// single thread does (including `HaveMoreOutput` and the final flush),
    /// and return the non-empty output chunks and whether the operator
    /// finished.
    pub(crate) fn drive(
        operator: &dyn PhysicalOperator,
        inputs: impl IntoIterator<Item = DataChunk>,
    ) -> Result<(Vec<DataChunk>, bool)> {
        let (gstate, mut lstate) = states(operator);
        drive_with_states(operator, inputs, gstate.as_ref(), lstate.as_mut())
    }

    /// Like [`drive`], with given states.
    pub(crate) fn drive_with_states(
        operator: &dyn PhysicalOperator,
        inputs: impl IntoIterator<Item = DataChunk>,
        gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<(Vec<DataChunk>, bool)> {
        let ctx = context();
        let mut outputs = vec![];
        let mut output = DataChunk::empty(operator.schema());
        for input in inputs {
            loop {
                output.reset();
                let result =
                    operator.execute(&ctx, &input, &mut output, gstate, lstate)?;
                if !output.is_empty() {
                    outputs.push(output.clone());
                }
                match result {
                    OperatorResult::NeedMoreInput => break,
                    OperatorResult::HaveMoreOutput => {}
                    OperatorResult::Finished => return Ok((outputs, true)),
                    OperatorResult::Blocked(interrupt) => {
                        panic!("unexpected {interrupt:?}")
                    }
                }
            }
        }
        if operator.requires_final_execute() {
            loop {
                output.reset();
                let result = operator.final_execute(&ctx, &mut output, gstate, lstate)?;
                if !output.is_empty() {
                    outputs.push(output.clone());
                }
                if result == OperatorFinalizeResult::Finished {
                    break;
                }
            }
        }
        Ok((outputs, false))
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray, StringViewArray};
    use arrow::compute::filter_record_batch;
    use arrow::datatypes::{Field, Schema};
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::filter::batch_filter;
    use datafusion_physical_plan::sorts::sort::SortExec;

    use crate::test_utils::test_session_config;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Utf8View, true),
        ]))
    }

    /// `rows` rows starting at `start`: `a` (nulls on every 7th row), `b`,
    /// `s` and `v`.
    fn batch(start: i64, rows: usize) -> RecordBatch {
        let ids = start..start + rows as i64;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter(
                ids.clone()
                    .map(|i| (i % 7 != 3).then_some((i % 100) as i32)),
            )),
            Arc::new(Int64Array::from_iter_values(ids.clone())),
            Arc::new(StringArray::from_iter(
                ids.clone()
                    .map(|i| (i % 5 != 1).then(|| format!("s{}", i % 11))),
            )),
            Arc::new(StringViewArray::from_iter(ids.map(|i| {
                (i % 9 != 2).then(|| format!("a string view longer than 12 bytes {i}"))
            }))),
        ];
        RecordBatch::try_new(schema(), columns).unwrap()
    }

    fn new_filter(predicate: &str, projection: Option<Vec<usize>>) -> PhysicalFilter {
        let schema = schema();
        PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr(predicate, &schema),
            projection,
        )
        .unwrap()
    }

    /// Filter `input` with `filter` in a fresh thread state.
    fn execute(
        filter: &PhysicalFilter,
        input: &DataChunk,
    ) -> (DataChunk, OperatorResult) {
        let (gstate, mut lstate) = states(filter);
        let mut output = DataChunk::empty(filter.schema());
        let result = filter
            .execute(
                &context(),
                input,
                &mut output,
                gstate.as_ref(),
                lstate.as_mut(),
            )
            .unwrap();
        (output, result)
    }

    /// What `FilterExec` produces for `predicate` on the logical rows of
    /// `input`.
    fn expected(predicate: &str, input: &DataChunk) -> RecordBatch {
        batch_filter(
            &input.compact().unwrap(),
            &physical_expr(predicate, &schema()),
        )
        .unwrap()
    }

    fn same_buffers(left: &ArrayRef, right: &ArrayRef) -> bool {
        left.to_data().buffers()[0].as_ptr() == right.to_data().buffers()[0].as_ptr()
    }

    #[test]
    fn selects_rows_with_a_selection_vector_without_copying() {
        let input = DataChunk::from_record_batch(batch(0, 100));
        let filter = new_filter("a > 50", None);
        let (output, result) = execute(&filter, &input);
        assert!(matches!(result, OperatorResult::NeedMoreInput));
        assert_eq!(output.schema(), &schema());
        // The rows with `a > 50`: `NULL` is not selected.
        let indices = (0..100u32)
            .filter(|i| i % 7 != 3 && *i > 50)
            .collect::<Vec<_>>();
        assert_eq!(output.selection().unwrap().indices(), indices.as_slice());
        // The columns are the input columns.
        for (output, input) in output.columns().iter().zip(input.columns()) {
            assert!(Arc::ptr_eq(output, input));
        }
        assert_eq!(output.compact().unwrap(), expected("a > 50", &input));
    }

    #[test]
    fn all_true_keeps_the_input_and_all_false_produces_nothing() {
        let input = DataChunk::from_record_batch(batch(0, 100));
        let (output, result) = execute(&new_filter("b >= 0", None), &input);
        assert!(matches!(result, OperatorResult::NeedMoreInput));
        assert!(output.selection().is_none());
        assert_eq!(output.len(), 100);
        assert!(Arc::ptr_eq(output.column(2), input.column(2)));

        for predicate in [
            "b < 0",
            "a > 50 AND b < 10",
            "false",
            "CAST(NULL AS BOOLEAN)",
        ] {
            let (output, result) = execute(&new_filter(predicate, None), &input);
            assert!(
                matches!(result, OperatorResult::NeedMoreInput),
                "{predicate}"
            );
            assert!(output.is_empty(), "{predicate}");
        }
        let (output, _) = execute(&new_filter("true", None), &input);
        assert_eq!(output.len(), 100);
        assert!(output.selection().is_none());
    }

    #[test]
    fn conjuncts_refine_the_selection_of_the_input() {
        // The input already has a selection, e.g. from a previous filter.
        let input = DataChunk::from_record_batch(batch(0, 2000));
        for (incoming, predicate) in [
            // Dense incoming selections (more than a quarter of the rows):
            // infallible conjuncts are evaluated on all physical rows.
            (
                (0..2000).filter(|i| i % 3 != 0).collect::<Vec<u32>>(),
                "a > 10 AND s <> 's3'",
            ),
            // Sparse: the referenced columns are gathered.
            (
                (0..2000).filter(|i| i % 9 == 0).collect(),
                "a > 10 AND s <> 's3' AND v LIKE '%1%'",
            ),
            // Not increasing, dense and sparse.
            (
                (0..2000).rev().step_by(3).collect(),
                "a % 2 = 0 AND b > 100",
            ),
            (
                (0..2000).rev().step_by(9).collect(),
                "a > 10 AND s <> 's3' AND b > 100",
            ),
            ((500..1500).collect(), "a < 90 OR s IS NULL"),
        ] {
            let input = input
                .clone()
                .with_selection(SelectionVector::from(incoming))
                .unwrap();
            let (output, _) = execute(&new_filter(predicate, None), &input);
            assert_eq!(
                output.compact().unwrap(),
                expected(predicate, &input),
                "{predicate}"
            );
            // Still the physical columns of the input.
            assert!(Arc::ptr_eq(output.column(3), input.column(3)));
        }
    }

    #[test]
    fn short_circuits_once_no_row_is_left() {
        // `10 / z` fails on every row: it is never evaluated, because `a < 0`
        // selects no row.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("z", DataType::Int32, false),
        ]));
        let input = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(0, 100))),
            ],
        )
        .unwrap();
        let filter = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("a < 0 AND 10 / z > 1", &schema),
            None,
        )
        .unwrap();
        let (outputs, finished) = drive(
            &filter,
            std::iter::repeat_n(DataChunk::from(input.clone()), 50),
        )
        .unwrap();
        assert!(outputs.is_empty());
        assert!(!finished);
        // Without the guard, the division fails.
        let filter = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("10 / z > 1 AND a < 0", &schema),
            None,
        )
        .unwrap();
        let err = drive(&filter, [DataChunk::from(input)]).unwrap_err();
        assert!(err.to_string().contains("Divide by zero"), "{err}");
    }

    /// A filter for `predicate` on the columns of `batch`.
    fn filter_for(predicate: &str, schema: &SchemaRef) -> PhysicalFilter {
        PhysicalFilter::try_new(
            source(Arc::clone(schema), vec![]),
            physical_expr(predicate, schema),
            None,
        )
        .unwrap()
    }

    /// The rows of `batch` that pass `predicate`, filtered conjunct by
    /// conjunct with DataFusion's `FilterExec` kernel: every conjunct is
    /// evaluated on the rows that the conjuncts before it select, as the
    /// native filter evaluates the conjuncts that may fail (and the order of
    /// the others makes no difference). Or the error of the first conjunct
    /// that fails on these rows.
    fn filtered_conjunct_by_conjunct(
        predicate: &Arc<dyn PhysicalExpr>,
        batch: &RecordBatch,
    ) -> Result<RecordBatch> {
        let mut batch = batch.clone();
        for conjunct in datafusion_physical_expr::split_conjunction(predicate) {
            batch = batch_filter(&batch, conjunct)?;
        }
        Ok(batch)
    }

    /// Filter `input` natively (in the thread state `lstate`) and conjunct by
    /// conjunct with DataFusion's `FilterExec` kernel (see
    /// [`filtered_conjunct_by_conjunct`]), and assert that both select the
    /// same rows or both fail with the same error. Returns the error, if any.
    fn assert_filters_conjunct_by_conjunct(
        filter: &PhysicalFilter,
        input: &DataChunk,
        gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Option<String> {
        let batch = input.compact().unwrap();
        let expected = filtered_conjunct_by_conjunct(filter.predicate(), &batch);
        let actual = drive_with_states(filter, [input.clone()], gstate, lstate).map(
            |(outputs, _)| {
                let outputs = outputs
                    .iter()
                    .map(|chunk| chunk.compact().unwrap())
                    .collect::<Vec<_>>();
                arrow::compute::concat_batches(&filter.schema(), &outputs).unwrap()
            },
        );
        match (expected, actual) {
            (Ok(expected), Ok(actual)) => {
                assert_eq!(actual, expected, "{}", filter.predicate());
                None
            }
            (Err(expected), Err(actual)) => {
                let expected = expected.find_root().strip_backtrace();
                assert_eq!(
                    actual.find_root().strip_backtrace(),
                    expected,
                    "{}",
                    filter.predicate()
                );
                Some(expected)
            }
            (expected, actual) => panic!(
                "{}: filtering conjunct by conjunct returned {expected:?}, the native \
                 filter {actual:?}",
                filter.predicate()
            ),
        }
    }

    /// A batch with the columns `x` and `y` (`Int32`, nullable).
    fn xy(x: Vec<Option<i32>>, y: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(x)), Arc::new(Int32Array::from(y))],
        )
        .unwrap()
    }

    /// Whether filtering `batch` with `predicate` fails, natively and conjunct
    /// by conjunct (both must agree, see
    /// [`assert_filters_conjunct_by_conjunct`]).
    fn fails(predicate: &str, batch: &RecordBatch) -> bool {
        let filter = filter_for(predicate, &batch.schema());
        let (gstate, mut lstate) = states(&filter);
        assert_filters_conjunct_by_conjunct(
            &filter,
            &DataChunk::from(batch.clone()),
            gstate.as_ref(),
            lstate.as_mut(),
        )
        .is_some()
    }

    #[test]
    fn guards_keep_conjuncts_that_may_fail_from_the_rows_they_reject() {
        let some = |values: &[i32]| values.iter().map(|&v| Some(v)).collect::<Vec<_>>();
        // Whatever the share of the rows that `x <> 0` leaves (DataFusion
        // evaluates `10 / x` on all rows of a batch unless at most 20% of
        // them are left), the native filter only divides the rows it leaves.
        for batch in [
            xy(some(&[0, 1, 1, 1]), some(&[1, 1, 1, 1])),
            xy(some(&[0, 0, 0, 0, 2]), some(&[1; 5])),
            xy(some(&[0, 0, 0, 0, 0, 0, 0, 2, 3, 0]), some(&[1; 10])),
            xy(some(&[0, 0, 0, 0, 0, 0, 2, 3, 4, 0]), some(&[1; 10])),
            // Everything rejected.
            xy(some(&[0, 0, 0]), some(&[1, 1, 1])),
        ] {
            assert!(!fails("x <> 0 AND 10 / x > 1", &batch));
        }
        // A guard on another column.
        let batch = xy(some(&[0, 5, 0, 5, 5]), some(&[0, 1, 0, 1, 1]));
        assert!(!fails("y = 1 AND 10 / x > 1", &batch));
        // The first conjunct is evaluated on all rows.
        let batch = xy(some(&[0, 0, 0]), some(&[1, 1, 1]));
        assert!(fails("10 / x > 1 AND x <> 0", &batch));
    }

    #[test]
    fn null_rows_are_rejected_at_once() {
        // `x > 0` is `NULL` for the first row, which can never be selected: it
        // is rejected at once, and `10 / y` is not evaluated on it.
        let mut x = vec![None];
        x.extend([Some(1); 9]);
        let mut y = vec![Some(0)];
        y.extend([Some(5); 9]);
        assert!(!fails("x > 0 AND 10 / y > 1", &xy(x.clone(), y.clone())));
        // Only `NULL` rows.
        assert!(!fails(
            "x > 0 AND 10 / y > 1",
            &xy(vec![None; 10], vec![Some(0); 10])
        ));
        // A `true` row is evaluated.
        x[0] = Some(1);
        assert!(fails("x > 0 AND 10 / y > 1", &xy(x, y)));
    }

    #[test]
    fn chunks_with_a_selection_are_filtered_on_their_rows() {
        // The filter evaluates the predicate on the rows of the chunk: those
        // of its selection, which the guard `x <> 0` keeps from the division.
        let mut x = vec![Some(0); 100];
        x[90] = Some(1);
        x[95] = Some(2);
        let batch = xy(x, vec![Some(1); 100]);
        let filter = filter_for("x <> 0 AND 10 / x > 1", &batch.schema());
        let (gstate, mut lstate) = states(&filter);
        for selection in [
            (0..100).collect::<Vec<u32>>(),
            vec![89, 90, 94, 95],
            (86..96).collect(),
            // Not increasing.
            vec![95, 0, 90],
        ] {
            let input = DataChunk::from(batch.clone())
                .with_selection(SelectionVector::from(selection))
                .unwrap();
            let error = assert_filters_conjunct_by_conjunct(
                &filter,
                &input,
                gstate.as_ref(),
                lstate.as_mut(),
            );
            assert!(error.is_none(), "{input:?}");
        }
        // Without the guard, the division fails on a row of the selection
        // only.
        let filter = filter_for("10 / x > 1", &batch.schema());
        let (gstate, mut lstate) = states(&filter);
        for (selection, fail) in [(vec![90, 95], false), (vec![95, 0, 90], true)] {
            let input = DataChunk::from(batch.clone())
                .with_selection(SelectionVector::from(selection))
                .unwrap();
            let error = assert_filters_conjunct_by_conjunct(
                &filter,
                &input,
                gstate.as_ref(),
                lstate.as_mut(),
            );
            assert_eq!(error.is_some(), fail, "{input:?}");
        }
    }

    #[test]
    fn adaptation_does_not_change_errors() {
        // The selective `y = 1` must not move before the division: after
        // warm-up chunks without `x = 0`, a chunk with `x = 0` and `y = 0`
        // fails, as the division is the first conjunct.
        let batch = |zeros: bool| {
            xy(
                (0..2048)
                    .map(|i| Some(if zeros && i % 2 == 0 { 0 } else { 1 + i % 5 }))
                    .collect(),
                (0..2048).map(|i| Some(i32::from(i % 50 == 1))).collect(),
            )
        };
        let filter = filter_for("10 / x > 1 AND y = 1 AND x > 0", &batch(false).schema());
        assert_eq!(
            filter
                .adaptive_filter()
                .segments()
                .iter()
                .map(|s| (s.start, s.end))
                .collect::<Vec<_>>(),
            vec![(1, 3)]
        );
        let (gstate, mut lstate) = states(&filter);
        let chunks = super::super::adaptive_filter::WARMUP_CHUNKS
            + super::super::adaptive_filter::ADAPT_INTERVAL;
        for _ in 0..chunks {
            let error = assert_filters_conjunct_by_conjunct(
                &filter,
                &DataChunk::from(batch(false)),
                gstate.as_ref(),
                lstate.as_mut(),
            );
            assert!(error.is_none());
        }
        let permutation = lstate
            .downcast_ref::<FilterState>()
            .unwrap()
            .adaptive
            .permutation()
            .to_vec();
        // The division stays first; `y = 1` moved before `x > 0`.
        assert_eq!(permutation, vec![0, 1, 2]);
        let error = assert_filters_conjunct_by_conjunct(
            &filter,
            &DataChunk::from(batch(true)),
            gstate.as_ref(),
            lstate.as_mut(),
        );
        assert!(error.unwrap().contains("Divide by zero"));
    }

    /// A pseudo-random number generator (a linear congruential generator).
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n
        }
    }

    #[test]
    fn results_and_errors_are_those_of_filtering_conjunct_by_conjunct() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
            Field::new("z", DataType::Int32, false),
        ]));
        let mut random = Lcg(42);
        // Chunks of up to 300 rows in which a varying share of the rows has
        // `x = 0`, `y = 0` or `NULL`s: guards leave below and above 20% of the
        // rows.
        let chunks = (0..400)
            .map(|_| {
                let rows = 1 + random.below(300) as usize;
                let zeros = random.below(100);
                let nulls = random.below(30);
                let value = |random: &mut Lcg| match random.below(100) {
                    p if p < nulls => None,
                    p if p < nulls + zeros => Some(0),
                    _ => Some(1 + random.below(6) as i32),
                };
                let x = (0..rows)
                    .map(|_| value(&mut random))
                    .collect::<Int32Array>();
                let y = (0..rows)
                    .map(|_| value(&mut random))
                    .collect::<Int32Array>();
                let z = Int32Array::from_iter_values(
                    (0..rows).map(|_| random.below(10) as i32),
                );
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(x), Arc::new(y), Arc::new(z)],
                )
                .unwrap();
                let chunk = DataChunk::from(batch);
                if random.below(4) == 0 {
                    // A chunk with a selection.
                    let selection = (0..rows as u32)
                        .filter(|_| random.below(3) != 0)
                        .collect::<Vec<_>>();
                    if !selection.is_empty() {
                        return chunk
                            .with_selection(SelectionVector::from(selection))
                            .unwrap();
                    }
                }
                chunk
            })
            .collect::<Vec<_>>();
        // The predicates, and whether they fail on some chunks: guarded
        // divisions never do.
        for (predicate, can_fail) in [
            ("x <> 0 AND 10 / x > 1", false),
            ("y > 2 AND 10 / x > 1", true),
            ("10 / x > 1 AND y = 1", true),
            ("y = 1 AND z > 3 AND 10 / x > 1 AND y < 5", true),
            ("x IS NOT NULL AND y > 0 AND 100 % x = 1 AND z <> 4", true),
            ("(x = 0 OR 10 / x > 2) AND y > 1", true),
            ("y <> 0 AND 10 / y > 1 AND x <> 0 AND 10 / x > 1", false),
            ("x > 0 AND z > 2 AND 10 / (x - 1) > 1", true),
            (
                "CASE WHEN x <> 0 THEN 10 / x > 1 ELSE false END AND y > 0",
                false,
            ),
            ("x > 0 AND y > 0 AND z > 0", false),
        ] {
            let filter = filter_for(predicate, &schema);
            let (gstate, mut lstate) = states(&filter);
            let mut errors = 0;
            let mut rows = 0;
            for chunk in &chunks {
                match assert_filters_conjunct_by_conjunct(
                    &filter,
                    chunk,
                    gstate.as_ref(),
                    lstate.as_mut(),
                ) {
                    Some(_) => errors += 1,
                    None => rows += 1,
                }
            }
            assert!(rows > 0, "{predicate}");
            assert_eq!(errors > 0, can_fail, "{predicate}");
        }
    }

    #[test]
    fn adapts_the_order_of_the_conjuncts() {
        // `b >= 0` selects everything, `upper(s) = 'S7'` 9% of the rows.
        let filter = new_filter("b >= 0 AND s IS NOT NULL AND upper(s) = 'S7'", None);
        assert!(filter.adaptive_filter().is_adaptive());
        let (gstate, mut lstate) = states(&filter);
        let permutation = |lstate: &dyn OperatorState| {
            lstate
                .downcast_ref::<FilterState>()
                .unwrap()
                .adaptive
                .permutation()
                .to_vec()
        };
        // `upper(s) = 'S7'` (a function) is fallible: it stays after the
        // others.
        for b in 0..20 {
            let input = DataChunk::from_record_batch(batch(b * 2048, 2048));
            let (outputs, _) = drive_with_states(
                &filter,
                [input.clone()],
                gstate.as_ref(),
                lstate.as_mut(),
            )
            .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(
                outputs[0].compact().unwrap(),
                expected("b >= 0 AND s IS NOT NULL AND upper(s) = 'S7'", &input)
            );
        }
        // `s IS NOT NULL` removes some rows, `b >= 0` none: it moved first.
        assert_eq!(permutation(lstate.as_ref()), vec![1, 0, 2]);

        // Infallible conjuncts are reordered freely. (The conjuncts that
        // select every row have a rank of 0, so this order does not depend
        // on the measured costs.)
        let filter = new_filter("b >= 0 AND b < 1000000000 AND a = 7", None);
        let (gstate, mut lstate) = states(&filter);
        for b in 0..(super::super::adaptive_filter::WARMUP_CHUNKS as i64) {
            let input = DataChunk::from_record_batch(batch(b * 2048, 2048));
            drive_with_states(&filter, [input], gstate.as_ref(), lstate.as_mut())
                .unwrap();
        }
        assert_eq!(permutation(lstate.as_ref()), vec![2, 0, 1]);
    }

    #[test]
    fn embedded_projection_shares_the_input_columns() {
        let input = DataChunk::from_record_batch(batch(0, 100));
        let filter = new_filter("a > 50", Some(vec![3, 1]));
        assert_eq!(
            filter
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["v", "b"]
        );
        let (output, _) = execute(&filter, &input);
        assert!(Arc::ptr_eq(output.column(0), input.column(3)));
        assert!(Arc::ptr_eq(output.column(1), input.column(1)));
        assert!(same_buffers(output.column(1), input.column(1)));
        assert_eq!(
            output.compact().unwrap(),
            expected("a > 50", &input).project(&[3, 1]).unwrap()
        );

        // An empty projection only keeps the number of rows.
        let (output, _) = execute(&new_filter("a > 50", Some(vec![])), &input);
        assert_eq!(output.num_columns(), 0);
        assert_eq!(output.len(), expected("a > 50", &input).num_rows());
    }

    #[test]
    fn chunks_without_columns_keep_their_row_count() {
        let schema = Arc::new(Schema::empty());
        let filter = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("true", &schema),
            None,
        )
        .unwrap();
        let input = DataChunk::try_new_without_columns(Arc::clone(&schema), 10).unwrap();
        let (output, _) = execute(&filter, &input);
        assert_eq!(output.len(), 10);
        // A selection of a chunk without columns is only a row count.
        let input = input
            .with_selection(SelectionVector::from(vec![2, 5, 9]))
            .unwrap();
        let (output, _) = execute(&filter, &input);
        assert_eq!(output.len(), 3);
        assert_eq!(output.compact().unwrap().num_rows(), 3);
    }

    #[test]
    fn fetch_limits_the_rows_of_all_states() {
        let filter = new_filter("b % 10 = 1", None).with_fetch(Some(25));
        let ctx = context();
        let gstate = filter.get_global_operator_state(&ctx).unwrap();
        // Two local states sharing the global state, as for the pipelines of
        // the inputs of a union below the filter (which run one after the
        // other, as the filter is not parallel): 10 rows per chunk.
        let mut threads = (0..2)
            .map(|_| filter.get_operator_state(&ctx, gstate.as_ref()).unwrap())
            .collect::<Vec<_>>();
        let mut rows = vec![];
        let mut results = vec![];
        for i in 0..4 {
            let lstate = &mut threads[i % 2];
            let input = DataChunk::from_record_batch(batch(i as i64 * 100, 100));
            let mut output = DataChunk::empty(filter.schema());
            let result = filter
                .execute(&ctx, &input, &mut output, gstate.as_ref(), lstate.as_mut())
                .unwrap();
            rows.push(output.len());
            results.push(matches!(result, OperatorResult::Finished));
        }
        assert_eq!(rows, vec![10, 10, 5, 0]);
        assert_eq!(results, vec![false, false, true, true]);

        // `fetch = 0` produces nothing.
        let filter = new_filter("true", None).with_fetch(Some(0));
        let (outputs, finished) =
            drive(&filter, [DataChunk::from_record_batch(batch(0, 10))]).unwrap();
        assert!(outputs.is_empty());
        assert!(finished);
    }

    #[test]
    fn rejects_non_boolean_predicates() {
        let schema = schema();
        let err = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("a + 1", &schema),
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Filter predicate must return BOOLEAN values"),
            "{err}"
        );
    }

    #[test]
    fn description_shows_predicate_projection_and_fetch() {
        let filter = new_filter("b > 50", Some(vec![3, 1])).with_fetch(Some(5));
        assert_eq!(
            filter.description(),
            "PhysicalFilter: b@1 > 50, projection=[v@3, b@1], fetch=5"
        );
        assert_eq!(filter.fetch(), Some(5));
        assert_eq!(filter.projection(), Some([3, 1].as_slice()));
    }

    #[test]
    fn results_match_filter_record_batch_over_many_chunks() {
        // Whatever the order of the conjuncts, the result is the same.
        let predicates = [
            "a > 20 AND s <> 's4' AND v LIKE '%7%' AND b % 3 = 0",
            "s IS NOT NULL AND a IS NULL",
            "(a > 90 OR s = 's1') AND v IS NOT NULL AND b < 50000",
        ];
        for predicate in predicates {
            let filter = new_filter(predicate, Some(vec![1, 2]));
            let (gstate, mut lstate) = states(&filter);
            for b in 0..40 {
                let batch = batch(b * 1500, 1500);
                let input = DataChunk::from(batch.clone());
                let (outputs, _) =
                    drive_with_states(&filter, [input], gstate.as_ref(), lstate.as_mut())
                        .unwrap();
                let mask = physical_expr(predicate, &schema())
                    .evaluate(&batch)
                    .unwrap()
                    .into_array(batch.num_rows())
                    .unwrap();
                let reference = filter_record_batch(
                    &batch.project(&[1, 2]).unwrap(),
                    mask.as_boolean(),
                )
                .unwrap();
                let actual = outputs
                    .iter()
                    .map(|chunk| chunk.compact().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    arrow::compute::concat_batches(&filter.schema(), &actual).unwrap(),
                    reference,
                    "{predicate}"
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // Differential tests
    // ---------------------------------------------------------------------

    async fn assert_all_native_byte_eq(queries: &[&str]) {
        for sql in queries {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert!(run.operators.filters > 0, "no native filter: {sql}");
            assert!(run.operators.scans > 0, "no native scan: {sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_selective_filters() {
        assert_all_native_byte_eq(&[
            "SELECT * EXCLUDE (d) FROM t1 WHERE a > 10",
            "SELECT * EXCLUDE (d) FROM t1 WHERE id % 1000 = 7",
            "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
            "SELECT * EXCLUDE (d) FROM t1 WHERE id < 0",
            "SELECT * EXCLUDE (d) FROM t_empty WHERE a > 0",
            "SELECT * EXCLUDE (d) FROM t1 WHERE a IS NULL",
            "SELECT id, c FROM t1 WHERE c > -0.0",
            "SELECT id, f, ts FROM t1 WHERE f > DATE '1970-01-02' AND ts < TIMESTAMP '2024-01-10 00:00:00'",
            "SELECT id, g FROM t1 WHERE g > 0 AND g < 100.5",
            "SELECT id FROM t1 WHERE a IN (1, 2, 3, -5) AND d IS NOT NULL",
            "SELECT id, e FROM t1 WHERE e",
            "SELECT id, e FROM t1 WHERE NOT e",
            "SELECT * EXCLUDE (name) FROM t2 WHERE k > 0",
        ])
        .await;
        // Functions, e.g. `isnan`, on the rows that DataFusion decides per
        // batch to evaluate the right side of `OR` on, and strings with 32-bit
        // offsets, which `FilterExec` may fail to concatenate as it coalesces
        // its output (`d` and `name` are `Utf8`).
        assert_all_native_byte_eq(&[
            "SELECT id, c FROM t1 WHERE c IS NULL OR c = 0.0 OR isnan(c)",
            "SELECT * FROM t1 WHERE a > 10",
            "SELECT id, d FROM t1 WHERE a > 40 OR d = 'gamma'",
            "SELECT * FROM t2 WHERE k > 0",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_multi_conjunct_and_or_filters() {
        assert_all_native_byte_eq(&[
            "SELECT id, a, b FROM t1 WHERE a > 0 AND b < 5000 AND c > 0.0 AND e",
            "SELECT id, h FROM t1 WHERE a > 40 OR d = 'gamma'",
            "SELECT id FROM t1 WHERE (a > 0 OR b IS NULL) AND (d LIKE 'a%' OR h LIKE '%01%') AND NOT e",
            "SELECT id, h FROM t1 WHERE h > 'a fairly long view string 0150' AND id % 2 = 0 AND a < 30",
            "SELECT id FROM t1 WHERE b % 7 = 1 AND a + b > 100 AND c * 2.0 > 1.0",
            "SELECT id FROM t1 WHERE ((a > 0 AND b > 0) OR (a < 0 AND b < 100)) AND c IS NOT NULL",
            "SELECT id FROM t1 WHERE a BETWEEN -10 AND 10 AND d IN ('alpha', 'beta', '') AND f IS NOT NULL",
            "SELECT id, a FROM t1 WHERE CASE WHEN a > 0 THEN b > 100 ELSE c < 0.0 END AND g IS NOT NULL",
        ])
        .await;
        // A function after the first conjunct (DataFusion's optimizer moves
        // function calls after the cheap conjuncts).
        assert_all_native_byte_eq(&[
            "SELECT id FROM t1 WHERE b % 7 = 1 AND a + b > 100 AND upper(d) = 'ALPHA'",
            "SELECT id FROM t1 WHERE upper(d) = 'ALPHA' AND b % 7 = 1 AND a + b > 100",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_fallible_conjuncts() {
        // A fallible first conjunct is evaluated on all rows, whatever the
        // batches: these filters are native.
        assert_all_native_byte_eq(&[
            "SELECT id, x FROM fallible WHERE CAST(id AS INT) % 10 = 0 AND s IS NOT NULL",
            "SELECT id, x FROM fallible WHERE 10 / (x + 100) < 1 AND x <> 0",
        ])
        .await;
        // Functions, whatever they may fail with (e.g. DataFusion limits the
        // sizes of the strings of `repeat` per batch).
        assert_all_native_byte_eq(&[
            "SELECT id, x FROM fallible WHERE length(s) <> 3",
            "SELECT id, x FROM fallible WHERE length(s) <> 3 AND s LIKE '%1%'",
            "SELECT id, x FROM fallible WHERE s LIKE '%1%' AND length(s) <> 3",
            "SELECT id, x FROM fallible WHERE upper(arrow_cast(s, 'Utf8View')) <> 'N/A 7'",
            "SELECT id, x FROM fallible WHERE upper(s) <> 'X'",
            "SELECT id, x FROM fallible WHERE upper(s) <> 'X' AND s LIKE '%1%'",
            "SELECT id, x FROM fallible WHERE repeat(s, 2) <> 'X'",
        ])
        .await;
        // Guards: DataFusion divides only the rows with `x <> 0` because, in
        // each batch, at most 20% of the rows are left; the native filter
        // only divides them in any case.
        assert_all_native_byte_eq(&[
            "SELECT id, x FROM fallible WHERE x <> 0 AND 10 / x > 1",
            "SELECT id, x FROM fallible WHERE x <> 0 AND 10 / x > 1 AND id % 2 = 0",
            "SELECT id FROM fallible WHERE x <> 0 AND 100 % x = 0",
            // (DataFusion's optimizer moves cheap conjuncts such as casts
            // before regular expressions, so the guard must be cheap too.)
            "SELECT id, s FROM fallible WHERE id % 10 = 0 AND CAST(s AS INT) > 500",
            "SELECT id, big FROM fallible WHERE big < 100000 AND CAST(big AS INT) % 2 = 0",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_errors_of_native_filters() {
        // Whether DataFusion fails, and whether the native filter fails: it
        // only evaluates a conjunct that may fail on the rows that the
        // conjuncts before it select, whatever the share of them that are
        // left, and rejects the rows for which a conjunct is `NULL` at once.
        // DataFusion evaluates the conjuncts after the first one on all rows
        // of a batch unless at most 20% of them are left, and does not reject
        // `NULL` rows. All filters are native.
        for (sql, datafusion_fails, native_fails) in [
            // Dense guards: DataFusion divides all rows, including `x = 0`.
            (
                "SELECT * FROM dense WHERE x <> 0 AND 10 / x > 1",
                true,
                false,
            ),
            (
                "SELECT id FROM dense WHERE x > 0 AND x < 100 AND 100 % x = 1",
                true,
                false,
            ),
            // A selective conjunct after the division does not guard it, even
            // once the conjuncts are reordered at runtime.
            (
                "SELECT * FROM dense WHERE 10 / x > 1 AND y = 1 AND id > 0",
                true,
                true,
            ),
            // Sparse guards: only the rows that are left are divided.
            (
                "SELECT * FROM sparse WHERE x <> 0 AND 10 / x > 1",
                false,
                false,
            ),
            (
                "SELECT id, 10 / x AS q FROM sparse WHERE x <> 0",
                false,
                false,
            ),
            (
                "SELECT id, 10 / x AS q FROM dense WHERE x <> 0",
                false,
                false,
            ),
            // `NULL` rows: DataFusion divides them, the native filter rejects
            // them first.
            (
                "SELECT * FROM nulls WHERE a > 0 AND 10 / b > 1",
                true,
                false,
            ),
            (
                "SELECT * FROM nulls_ok WHERE a > 0 AND 10 / b > 1",
                false,
                false,
            ),
            (
                "SELECT * FROM nulls WHERE a IS NULL AND 10 / b > 1",
                true,
                true,
            ),
            // No guard.
            ("SELECT id FROM fallible WHERE 10 / x > 1", true, true),
            (
                "SELECT id FROM fallible WHERE CAST(s AS INT) > 5",
                true,
                true,
            ),
            (
                "SELECT id FROM fallible WHERE 10 / x > 1 OR x = 0",
                true,
                true,
            ),
            (
                "SELECT id FROM fallible WHERE x <> 0 AND 10 / x > 1",
                false,
                false,
            ),
        ] {
            let outcome = assert_allowed_native_outcome_with_config(
                sql,
                test_session_config(),
                guard_tables,
                &["Divide by zero", "Cannot cast string"],
            )
            .await;
            assert!(outcome.operators.filters > 0, "{sql}");
            assert_eq!(
                outcome.datafusion.is_some(),
                datafusion_fails,
                "{sql}: {outcome:?}"
            );
            assert_eq!(outcome.native.is_some(), native_fails, "{sql}: {outcome:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_string_view_and_dictionary_filters() {
        assert_all_native_byte_eq(&[
            "SELECT i FROM vd WHERE d32 = 'k3'",
            "SELECT i, sv FROM vd WHERE sv LIKE '%7%' AND d8 <> 'x2'",
            // Patterns that compile cannot fail, after other conjuncts too.
            "SELECT i, sv FROM vd WHERE i % 3 = 0 AND sv LIKE '%7%'",
            "SELECT i, sv FROM vd WHERE i % 2 = 0 AND (sv ILIKE 'A%' OR sv ~ '7$')",
            "SELECT i FROM vd WHERE d32 IN ('k1', 'k2', 'k39') OR d8 IS NULL",
            "SELECT i, sv FROM vd WHERE sv > 'short4' AND i % 3 = 0",
            "SELECT i, sv FROM vd WHERE d8 = 'x5' AND d32 IS NOT NULL",
        ])
        .await;
        // Also with dictionaries of narrow keys (`d8`, `Int8` keys) and strings
        // in the output, which `FilterExec` merges and concatenates as it
        // coalesces its output (which may fail).
        assert_all_native_byte_eq(&[
            "SELECT * FROM vd WHERE d32 = 'k3'",
            "SELECT d32, i FROM vd WHERE d32 = 'k3'",
            "SELECT i, d8, sv FROM vd WHERE sv LIKE '%7%' AND d8 <> 'x2'",
            "SELECT * FROM vd WHERE sv IS NULL",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_filters_below_other_operators() {
        assert_all_native_byte_eq(&[
            "SELECT COUNT(*) FROM t1 WHERE a > 0",
            "SELECT h, COUNT(*), SUM(a) FROM t1 WHERE a > 0 AND e GROUP BY h",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k WHERE t1.b > 10 AND t2.w < 100",
            "SELECT t1.id, t2.k FROM t1 LEFT JOIN t2 ON t1.a = t2.k WHERE t1.id % 10 = 3",
            "SELECT id, a FROM t1 WHERE a > 0 ORDER BY id",
            "SELECT id, h FROM t1 WHERE h LIKE '%1%' ORDER BY h DESC, id LIMIT 20",
            "SELECT * FROM (SELECT id, a, h FROM t1 ORDER BY id LIMIT 100) WHERE a > 0",
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2 WHERE k < 0",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY h ORDER BY id) AS rn FROM t1 WHERE a > 0",
            "SELECT e, MAX(h) FROM t1 WHERE id % 3 = 0 GROUP BY e HAVING COUNT(*) > 100",
        ])
        .await;
        // The limit is pushed into the filter (`fetch = 7`), which is lowered:
        // with a fetch, it runs on one thread. Below it, the merge of the
        // sorted partitions (which `lower` does not lower) is executed by
        // DataFusion, with its input lowered.
        let sql = "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0 LIMIT 7";
        let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
        assert_eq!(
            (run.operators.filters, run.operators.scans),
            (1, 1),
            "{sql}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_tpch_filters_are_native() {
        for sql in [
            // Decimal arithmetic chains, whose parts may only fail with
            // arithmetic overflows.
            "SELECT l_orderkey, l_quantity FROM lineitem \
             WHERE l_extendedprice * (1 - l_discount) > 50000.00",
            "SELECT l_orderkey FROM lineitem \
             WHERE l_extendedprice * (1 - l_discount) * (1 + l_tax) > l_extendedprice",
            // `extract(year from ...)` and `substring(... from 1 for 2)`, also
            // after other conjuncts.
            "SELECT o_orderkey, o_orderdate FROM orders WHERE extract(year from o_orderdate) = 1995",
            "SELECT c_custkey, c_acctbal FROM customer \
             WHERE substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17') \
             AND c_acctbal > 0.00",
            "SELECT c_custkey FROM customer WHERE c_acctbal > 0.00 \
             AND substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17')",
            "SELECT o_orderkey FROM orders WHERE o_custkey > 100 \
             AND date_part('month', o_orderdate) = 3 AND strpos(o_orderpriority, 'HIGH') > 0",
            // Constant strings of bounded sizes.
            "SELECT o_orderkey FROM orders WHERE CASE WHEN o_orderpriority = '1-URGENT' \
             OR o_orderpriority = '2-HIGH' THEN '1-URGENT' ELSE '2-HIGH' END = '1-URGENT'",
            // Widening casts of decimals.
            "SELECT c_custkey FROM customer WHERE CAST(c_acctbal AS DECIMAL(19, 6)) > 1000.50",
        ] {
            let run = assert_native_byte_eq_with_setup(sql, tpch_tables).await;
            let (_, filters) = assert_all_lowerable(&run.expected.plan);
            assert!(filters > 0, "no filter: {sql}");
            assert!(run.operators.filters > 0, "no native filter: {sql}");
            assert!(run.operators.scans > 0, "no native scan: {sql}");
        }
        // Also with a cast that may fail with another error than an overflow:
        // `CAST(l_quantity * 1000 AS DECIMAL(38, 6))` of a `DECIMAL(36, 2)`
        // may overflow its precision.
        let sql = "SELECT l_orderkey FROM lineitem \
                   WHERE l_extendedprice * (1 - l_discount) * (1 + l_tax) < l_quantity * 1000";
        let run = assert_native_byte_eq_with_setup(sql, tpch_tables).await;
        assert_eq!(run.operators.filters, 1, "{sql}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn builtin_functions_of_tpch_filters_are_native_by_default() {
        // The sessions of `test_session_config` (4 threads) do not register the
        // functions they trust (see `TrustedFunctions`): by default, they trust
        // the built-in functions of DataFusion, so that the
        // `PipelineQueryPlanner` lowers the filters of TPC-H natively, with
        // every node of their plans.
        let config = test_session_config();
        assert!(
            config
                .options()
                .extensions
                .get::<crate::operators::adaptive_filter::TrustedFunctions>()
                .is_none()
        );
        for sql in [
            // `extract(year from ...)` (Q7 to Q9).
            "SELECT o_orderkey, o_orderdate FROM orders \
             WHERE extract(year from o_orderdate) = 1995",
            // `substring(c_phone from 1 for 2)` (Q22), also after other
            // conjuncts, and Q22 itself (with its scalar subquery and its anti
            // join).
            "SELECT c_custkey, c_acctbal FROM customer \
             WHERE substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17') \
             AND c_acctbal > 0.00",
            "SELECT c_custkey FROM customer WHERE c_acctbal > 0.00 \
             AND substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17')",
            "SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal FROM (\
             SELECT substring(c_phone from 1 for 2) AS cntrycode, c_acctbal FROM customer \
             WHERE substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17') \
             AND c_acctbal > (SELECT avg(c_acctbal) FROM customer WHERE c_acctbal > 0.00 \
             AND substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17')) \
             AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)) AS custsale \
             GROUP BY cntrycode ORDER BY cntrycode",
            // Constant strings of bounded sizes.
            "SELECT o_orderkey FROM orders WHERE CASE WHEN o_orderpriority = '1-URGENT' \
             OR o_orderpriority = '2-HIGH' THEN '1-URGENT' ELSE '2-HIGH' END = '1-URGENT'",
            // A decimal division by a `NULLIF` of `0` (a built-in function),
            // which can only fail with arithmetic overflows.
            "SELECT l_orderkey FROM lineitem \
             WHERE l_extendedprice / NULLIF(l_discount, 0) > 100000.00",
        ] {
            let (plan, root) =
                assert_planned_byte_eq(sql, config.clone(), tpch_tables).await;
            let (_, filters) = assert_all_lowerable(&plan);
            assert!(filters > 0, "no filter: {sql}");
            assert!(
                crate::bridge::bridged_plans(&root).is_empty(),
                "{sql}\n{}",
                crate::explain::format_operator_tree(&root)
            );
            assert_eq!(NativeOperators::of(&root).filters, filters, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operators_that_may_fail_share_pipelines() {
        // Parts of a projection and of a filter below it that may fail, with
        // arithmetic overflows or with other errors (a division), are native.
        for sql in [
            "SELECT l_extendedprice * (1 - l_discount) AS v FROM lineitem \
             WHERE l_extendedprice * l_tax > 10.00",
            "SELECT 10 / (l_orderkey + 1) AS q FROM lineitem \
             WHERE l_extendedprice * l_tax > 10.00",
        ] {
            let run = assert_native_byte_eq_with_setup(sql, tpch_tables).await;
            assert_eq!((run.operators.projections, run.operators.filters), (1, 1));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_filters_on_any_number_of_threads() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
                "SELECT id FROM t1 WHERE id % 1000 = 7",
                "SELECT id, a FROM t1 WHERE a > 0 ORDER BY id",
                "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0",
                "SELECT id, x FROM fallible WHERE CAST(id AS INT) % 10 = 0 AND s IS NOT NULL",
            ] {
                let run =
                    assert_native_byte_eq_with_config(sql, config.clone(), all_tables)
                        .await;
                assert!(run.operators.filters > 0, "{sql}");
            }
        }
    }

    /// A plan `FilterExec(a % 2 = 0, fetch = 3)` over `partitions`
    /// partitions of 20 rows each.
    fn filter_with_fetch(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = |start: i64| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from_iter_values(start..start + 20))],
            )
            .unwrap()
        };
        let partitions = (0..partitions as i64)
            .map(|p| vec![batch(p * 100)])
            .collect::<Vec<_>>();
        let input = datafusion_datasource::memory::MemorySourceConfig::try_new_exec(
            &partitions,
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let predicate = physical_expr("a % 2 = 0", &schema);
        Arc::new(
            datafusion_physical_plan::filter::FilterExecBuilder::new(predicate, input)
                .with_fetch(Some(3))
                .build()
                .unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn per_partition_fetch_is_kept() {
        // DataFusion produces 3 rows per partition: 6 rows for 2 partitions.
        // The chunks of a pipeline do not belong to partitions, so such a
        // `FilterExec` stays in DataFusion.
        let plan = filter_with_fetch(2);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let err =
            PhysicalFilter::try_from_filter_exec(exec, lower(exec.input()).unwrap())
                .unwrap_err();
        assert!(
            err.to_string().contains("each of its 2 input partitions"),
            "{err}"
        );
        let run = assert_native_plan_byte_eq(|| filter_with_fetch(2), false).await;
        assert_eq!(run.operators.filters, 0);
        let rows = run
            .actual
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(rows, 6);

        // With a single partition, the native filter produces the first 3
        // rows, in order, on a single thread.
        let plan = filter_with_fetch(1);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let filter =
            PhysicalFilter::try_from_filter_exec(exec, lower(exec.input()).unwrap())
                .unwrap();
        assert!(!filter.parallel_operator());
        assert!(new_filter("a > 0", None).parallel_operator());
        let run = assert_native_plan_byte_eq(|| filter_with_fetch(1), true).await;
        assert_eq!(run.operators.filters, 1);
        let rows = run
            .actual
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(rows, 3);

        // In SQL, a limit above the filter enforces the fetch over all
        // partitions: with the lowering of the tests, which lowers no limit,
        // the filter stays in DataFusion, with the same result. (The
        // `PipelineQueryPlanner` lowers the limit, and the filter without its
        // fetch, see `PhysicalLimit::try_lower`.)
        for sql in [
            "SELECT id FROM t1 WHERE id < 5 LIMIT 100",
            "SELECT id, a FROM t1 WHERE a > 1000 LIMIT 10",
        ] {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert_eq!(run.operators.filters, 0, "{sql}");
        }
    }

    /// A batch of a column `x` with the runs `(value, rows)`.
    fn runs(runs: &[(i32, usize)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let x = runs
            .iter()
            .flat_map(|&(value, rows)| std::iter::repeat_n(value, rows))
            .collect::<Int32Array>();
        RecordBatch::try_new(schema, vec![Arc::new(x)]).unwrap()
    }

    /// A scan of `batch`, in a single partition.
    fn scan(batch: &RecordBatch) -> Arc<dyn ExecutionPlan> {
        datafusion_datasource::memory::MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()]],
            batch.schema(),
            None,
        )
        .unwrap()
    }

    fn filter_exec(
        predicate: &str,
        input: Arc<dyn ExecutionPlan>,
        fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let predicate = physical_expr(predicate, &input.schema());
        Arc::new(
            datafusion_physical_plan::filter::FilterExecBuilder::new(predicate, input)
                .with_fetch(fetch)
                .build()
                .unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cases_of_unions_are_native() {
        // On the batch of `FilterExec`, `CASE` merges the unions of the rows
        // with `b` with nulls for the others, and fails: their union has no
        // nullable field. On the chunks of the native filter, all rows take
        // the branch, or none does: `CASE` merges nothing.
        let batch = unions_without_nulls();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let predicate =
                Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(
                    case_of_unions(&schema),
                ));
            Arc::new(FilterExec::try_new(predicate, scan(&batch)).unwrap())
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Cannot extend a union with no nullable fields with nulls"],
        )
        .await;
        assert!(
            outcome.as_ref().is_err_and(|error| error
                .contains("Cannot extend a union with no nullable fields with nulls")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_between_failures_are_native() {
        // `FilterExec(x >= 0)` coalesces its output: it pulls both batches of
        // its input, and the cast of the second one fails, before the filter
        // above it divides by zero the rows of the first one. The native
        // filter passes the rows of the first chunk on at once, and the native
        // filter above it may divide them by zero first: which of several
        // errors is raised is the engine's own.
        let plan = || filter_exec("10 / x + c + l > 0", filtered_casts(), None);
        let config =
            || datafusion::prelude::SessionConfig::new().with_target_partitions(1);
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome_with_config(
            plan,
            false,
            config(),
            &["Cannot cast string 'invalid'", "Divide by zero"],
        )
        .await;
        let error = error.unwrap();
        assert!(error.contains("Cannot cast string 'invalid'"), "{error}");
        // Both filters are native.
        assert_eq!(operators.filters, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn matches_of_patterns_that_may_be_scalars_are_native() {
        // On the batch of `FilterExec`, `CASE` returns an array of patterns,
        // which Arrow compiles, and the pattern of 100000 `s` (each `[sSſ]`)
        // is too big. On the first chunk of the native filter, all rows take
        // the first branch, whose pattern `CASE` returns as is, a scalar, which
        // Arrow matches with the ASCII strings without compiling it.
        let batch = guarded_patterns();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                FilterExec::try_new(guarded_pattern_match(&schema), scan(&batch))
                    .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Unable to build regex"])
            .await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.filters, 1);
        // A pattern that is a column is always an array.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec("s ILIKE p", scan(&batch), None),
            false,
            &["Unable to build regex"],
        )
        .await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn negations_of_values_that_may_be_scalars_are_native() {
        // On the batch of `FilterExec`, `CASE` returns an array, which `-`
        // negates wrapping: `-(-9223372036854775808)` is
        // `-9223372036854775808`. On the first chunk of the native filter,
        // all rows take the `ELSE` branch, whose value `CASE` returns as is, a
        // scalar, which `-` negates with overflow checks, and fails.
        let batch = guarded_minimums();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let predicate =
                Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
                    negated_case(&schema),
                    datafusion_expr::Operator::Eq,
                    Arc::new(datafusion_physical_expr::expressions::Literal::new(
                        ScalarValue::Int64(Some(i64::MIN)),
                    )),
                ));
            Arc::new(FilterExec::try_new(predicate, scan(&batch)).unwrap())
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Arithmetic overflow"])
            .await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.filters, 1);
        // The negation of a column is always that of an array.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec("-y < 0", scan(&batch), None),
            false,
            &["Arithmetic overflow"],
        )
        .await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn functions_are_native() {
        let _large = one_large_test_at_a_time().await;
        // A function is opaque: whatever it returns, its implementation may
        // fail depending on the rows it is evaluated on together. Here
        // `concat_length(s)` builds `s || s`, 2 GiB of strings on the batch of
        // 8192 rows of `FilterExec`, whose 32-bit offsets overflow, but
        // 512 MiB on the chunks of the native filter (whose output is `id`).
        let batch = large_strings();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let predicate =
                Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
                    concat_length(&schema),
                    datafusion_expr::Operator::Gt,
                    Arc::new(datafusion_physical_expr::expressions::Literal::new(
                        ScalarValue::Int32(Some(0)),
                    )),
                ));
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    predicate,
                    scan(&batch),
                )
                .apply_projection(Some(vec![0]))
                .unwrap()
                .build()
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Offset overflow error"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Offset overflow error: 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concatenations_of_strings_are_native() {
        let _large = one_large_test_at_a_time().await;
        // `s || s` builds 2 GiB of strings on the batch of 8192 rows of
        // `FilterExec`, whose 32-bit offsets overflow, but 512 MiB on the
        // chunks of the native filter (whose output drops `s`).
        let batch = large_strings();
        let plan = || filter_exec("(s || s) IS NOT NULL", scan(&batch), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Offset overflow error"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Offset overflow error: 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn functions_building_values_of_variable_lengths_are_native() {
        // DataFusion checks the size of the strings that `repeat` builds on a
        // batch (also with 64-bit offsets): 4 GiB on the batch of 8192 rows of
        // `FilterExec`, more than it allows, but 1 GiB on the chunks of the
        // native filter. Functions are opaque: whether one that builds values
        // of variable lengths fails depends on the rows it is evaluated on
        // together.
        let batch = repeated_strings();
        let plan = || filter_exec("repeat(s, n) IS NOT NULL", scan(&batch), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["string size overflow on repeat"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("string size overflow on repeat")),
            "{outcome:?}"
        );
        // So is one that returns values of fixed widths, e.g. `abs`.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec("abs(n) > 0", scan(&batch), None),
            false,
            &["string size overflow on repeat"],
        )
        .await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dynamic_predicates_are_native() {
        // The children of a `DynamicFilterPhysicalExpr` are the columns it
        // reads, not the predicate it evaluates, which may change during the
        // execution: here `x <> 0 AND 10 / x > 1`, whose errors depend on the
        // batches (see `batch_dependent_predicates_are_native`),
        // alone, or in another predicate.
        let batch = runs(&[(0, 1848), (1, 2248)]);
        let schema = batch.schema();
        let dynamic = dynamic_filter("x <> 0 AND 10 / x > 1", &schema);
        let nested: Arc<dyn PhysicalExpr> =
            Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
                physical_expr("x >= 0", &schema),
                datafusion_expr::Operator::And,
                Arc::clone(&dynamic),
            ));
        for predicate in [dynamic, nested] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    FilterExec::try_new(Arc::clone(&predicate), scan(&batch)).unwrap(),
                )
            };
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(plan, false, &["Divide by zero"])
                .await;
            assert!(
                error
                    .as_ref()
                    .is_some_and(|error| error.contains("Divide by zero")),
                "{predicate}: {error:?}"
            );
            assert_eq!(operators.filters, 1, "{predicate}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn batch_dependent_predicates_are_native() {
        // One batch of 4096 rows, 1848 zeros then 2248 ones. Split into two
        // chunks, 90% of the rows of the first chunk are zeros, but only 45%
        // of the batch: DataFusion divides all rows by `x` and fails.
        let batch = runs(&[(0, 1848), (1, 2248)]);
        for predicate in ["x <> 0 AND 10 / x > 1", "x = 0 OR 10 / x > 1"] {
            let plan = filter_exec(predicate, scan(&batch), None);
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec), "{predicate}");

            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter_exec(predicate, scan(&batch), None),
                false,
                &["Divide by zero"],
            )
            .await;
            assert!(error.unwrap().contains("Divide by zero"), "{predicate}");
            assert_eq!(operators.filters, 1, "{predicate}");

            // On the chunks of the pipeline, the predicate does not fail.
            let filter = filter_for(predicate, &batch.schema());
            let chunks =
                [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
            assert!(drive(&filter, chunks).is_ok(), "{predicate}");
        }
        // Predicates whose errors do not depend on the batches are lowered.
        for (predicate, fails) in [("10 / x > 1 AND x <> 0", true), ("x <> 0", false)] {
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter_exec(predicate, scan(&batch), None),
                false,
                &["Divide by zero"],
            )
            .await;
            assert_eq!(error.is_some(), fails, "{predicate}");
            assert_eq!(operators.filters, 1, "{predicate}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn opaque_columns_see_all_input_columns() {
        // `opaque(x)` reads `x` without a `Column` child: bound to the columns
        // that the predicate references (`y`, or none), it would read `y` as
        // `x`, or no column at all. An opaque expression may also fail
        // depending on the rows it is evaluated on together, so it stays in
        // DataFusion (the native operators bind all columns for it, see
        // `BoundExprs`).
        let batch = xys(&[((0, 1), 100), ((2, 2), 100), ((3, 0), 100)]);
        let schema = batch.schema();
        let one: Arc<dyn PhysicalExpr> =
            Arc::new(datafusion_physical_expr::expressions::Literal::new(
                ScalarValue::Int32(Some(1)),
            ));
        for (left, op, right) in [
            (
                physical_expr("y", &schema),
                datafusion_expr::Operator::Eq,
                opaque_column("x", &schema),
            ),
            (
                opaque_column("x", &schema),
                datafusion_expr::Operator::Gt,
                one,
            ),
        ] {
            let predicate: Arc<dyn PhysicalExpr> = Arc::new(
                datafusion_physical_expr::expressions::BinaryExpr::new(left, op, right),
            );
            // (Without the strings of `s` in the output, see
            // `may_fail_to_coalesce`.)
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    datafusion_physical_plan::filter::FilterExecBuilder::new(
                        Arc::clone(&predicate),
                        scan(&batch),
                    )
                    .apply_projection(Some(vec![0, 1]))
                    .unwrap()
                    .build()
                    .unwrap(),
                )
            };
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(plan, false, &[]).await;
            assert!(error.is_none(), "{predicate}: {error:?}");
            assert_eq!(operators.filters, 1, "{predicate}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_that_fail_on_unicode_are_native() {
        // One batch of 4096 rows, `(x, s) = (1, "a")` then `(0, "é")`. On
        // strings that are all ASCII, Arrow matches an ASCII `ILIKE` pattern
        // without wildcards without a regular expression; on others, it
        // compiles one, here of 100000 `s` (each `[sSſ]`), which is too big.
        // DataFusion evaluates the `ILIKE` on all rows of the batch and fails;
        // the native filter only on the first chunk, which is all ASCII.
        let pattern = "s".repeat(100_000);
        let batch = xss(&[((1, "a"), 2048), ((0, "é"), 2048)]);
        let predicate = format!("x > 0 AND s ILIKE '{pattern}'");
        let plan = || filter_exec(&predicate, scan(&batch), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Unable to build regex"])
            .await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.filters, 1);
        // On the chunks of the pipeline, the predicate does not fail.
        let filter = filter_for(&predicate, &batch.schema());
        let chunks = [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
        assert!(drive(&filter, chunks).is_ok());
        // Patterns that compile are lowered (without strings in the output).
        for predicate in ["x > 0 AND s ILIKE 'A%'", "x > 0 AND s NOT ILIKE '%É'"] {
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter_exec(predicate, scan(&batch), None),
                false,
                &["Unable to build regex"],
            )
            .await;
            assert!(error.is_none(), "{predicate}");
            assert_eq!(operators.filters, 1, "{predicate}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn volatile_predicates_are_not_evaluated_before_execution() {
        use datafusion_physical_expr::expressions::{BinaryExpr, LikeExpr, lit};
        // `count_calls()` returns how often it was called: DataFusion calls it
        // once for the batch of 3 rows, and selects them all. Evaluating it
        // before (on an empty batch, to find whether it fails without rows, or
        // to compile the pattern of a `LIKE`) would change its results.
        let batch = runs(&[(1, 3)]);
        let schema = batch.schema();
        /// A predicate of `count_calls()`.
        type Predicate = fn(Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr>;
        let predicates: [Predicate; 3] = [
            |calls| {
                Arc::new(BinaryExpr::new(
                    calls,
                    datafusion_expr::Operator::Eq,
                    lit("1"),
                ))
            },
            |calls| Arc::new(LikeExpr::new(false, false, calls, lit("1%"))),
            |calls| Arc::new(LikeExpr::new(false, true, calls, lit("1"))),
        ];
        let filter = |predicate: Arc<dyn PhysicalExpr>| -> Arc<dyn ExecutionPlan> {
            Arc::new(FilterExec::try_new(predicate, scan(&batch)).unwrap())
        };
        for predicate in predicates {
            let calls = Arc::new(AtomicUsize::new(0));
            let expr = predicate(count_calls(&calls, &schema));
            let plan = filter(Arc::clone(&expr));
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec), "{expr}");
            let _ = crate::operators::adaptive_filter::fallibility(&expr, &schema);
            AdaptiveFilter::try_new(&expr, &schema).unwrap();
            lower(&plan).unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 0, "{expr}");

            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter(predicate(count_calls(&Arc::default(), &schema))),
                false,
                &[],
            )
            .await;
            assert!(error.is_none(), "{expr}");
            assert_eq!(operators.filters, 1, "{expr}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dynamic_predicates_are_not_evaluated_before_execution() {
        // `first_call()` is volatile, but hidden in a dynamic expression (whose
        // children are not the expression it evaluates): evaluating it before
        // the execution (e.g. on an empty batch, to find whether the `LIKE`
        // fails without rows) would make it return `false` during it.
        let batch = runs(&[(1, 3)]);
        let schema = batch.schema();
        let calls = Arc::new(AtomicUsize::new(0));
        let expr = dynamic_first_call(&calls, &schema);
        let filter = |predicate: Arc<dyn PhysicalExpr>| -> Arc<dyn ExecutionPlan> {
            Arc::new(FilterExec::try_new(predicate, scan(&batch)).unwrap())
        };
        let plan = filter(Arc::clone(&expr));
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let _ = crate::operators::adaptive_filter::fallibility(&expr, &schema);
        lower(&plan).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        // DataFusion selects all rows, natively too.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter(dynamic_first_call(&Arc::default(), &schema)),
            false,
            &[],
        )
        .await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.filters, 1);
        let outcome = outcome(
            filter(dynamic_first_call(&Arc::default(), &schema)),
            Arc::new(TaskContext::default()),
        )
        .await;
        assert_eq!(outcome, Ok(3));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_on_bytes_beneath_nulls_are_native() {
        // `x = 0` selects every other row, the first one too, whose string is
        // null above bytes that are not ASCII. DataFusion's `filter` keeps
        // these bytes: Arrow finds that the strings are not all ASCII and
        // compiles the pattern of the `ILIKE`, which is too big. The native
        // filter gathers the selected rows with `take`, which drops them: the
        // strings are ASCII, and the pattern is not compiled.
        let batch = non_ascii_beneath_a_null();
        let schema = batch.schema();
        let predicate = format!("s ILIKE '{}'", "s".repeat(100_000));
        let plan =
            || filter_exec(&predicate, filter_exec("x = 0", scan(&batch), None), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Unable to build regex"])
            .await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.filters, 2);
        let selected = DataChunk::from(batch.clone())
            .with_selection(SelectionVector::from(
                (0..2048).step_by(2).collect::<Vec<_>>(),
            ))
            .unwrap();
        assert!(drive(&filter_for(&predicate, &schema), [selected]).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_with_comparisons_that_arrow_cannot_evaluate_is_native() {
        // Arrow cannot compare dictionaries of dictionaries. With `fetch = 0`,
        // `FilterExec` evaluates its predicate on the first batch, and fails.
        let batch = nested_dictionaries();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let predicate =
                Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
                    physical_expr("a", &schema),
                    datafusion_expr::Operator::Eq,
                    physical_expr("b", &schema),
                ));
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    predicate,
                    scan(&batch),
                )
                .with_fetch(Some(0))
                .build()
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            plan,
            true,
            &["Invalid comparison operation"],
        )
        .await;
        let error = error.unwrap();
        assert!(error.contains("Invalid comparison operation"), "{error}");
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_with_case_insensitive_patterns_is_native() {
        // With `fetch = 0`, `FilterExec` evaluates its predicate on the first
        // batch, whose strings are not all ASCII: Arrow compiles the pattern of
        // `ILIKE`, which is too big, and fails (see
        // `case_insensitive_patterns_that_fail_on_unicode_are_native`).
        let batch = xss(&[((1, "a"), 2048), ((0, "é"), 2048)]);
        let predicate = format!("s ILIKE '{}'", "s".repeat(100_000));
        let plan = || filter_exec(&predicate, scan(&batch), Some(0));
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, true, &["Unable to build regex"])
            .await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_strings_in_shared_buffers_are_native() {
        let _large = one_large_test_at_a_time().await;
        // `FilterExec` concatenates the strings of the batches it coalesces
        // into batches of 8192 rows, whose 32-bit offsets overflow beyond
        // 2 GiB of bytes, which batches that share their buffers reach without
        // memory: here 32 copies of a batch of 256 rows with a string of 64 MiB
        // (and 255 empty strings). The native filter does not coalesce its
        // output, and the cache only up to 2048 rows.
        let mut strings = vec![String::new(); 256];
        strings[0] = "x".repeat((1 << 26) + 1);
        let batch = RecordBatch::try_from_iter([
            ("x", Arc::new(Int32Array::from(vec![1; 256])) as ArrayRef),
            ("s", Arc::new(StringArray::from(strings)) as ArrayRef),
        ])
        .unwrap();
        let batches = vec![batch; 32];
        let schema = batches[0].schema();
        let plan = || filter_exec("x > 0", scan_batches(&schema, &batches), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Offset overflow error"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Offset overflow error: 2147483680")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_dictionaries_of_values_that_arrow_does_not_merge_are_native() {
        let _large = one_large_test_at_a_time().await;
        // `FilterExec` concatenates the dictionaries of the batches it
        // coalesces: Arrow merges those of strings, binaries and primitive
        // values, but adds up the values of others, here `(1 << 31) + 1`
        // fixed-size binaries, more than `Int32` keys can index, though they
        // share their storage (1 GiB). The native filter does not coalesce.
        let values = arrow::array::FixedSizeBinaryArray::try_new(
            1,
            arrow::buffer::Buffer::from_vec(vec![0_u8; (1 << 30) + 1]),
            None,
        )
        .unwrap();
        let batch = |len: usize| {
            let d =
                arrow::array::DictionaryArray::<arrow::datatypes::Int32Type>::try_new(
                    Int32Array::from(vec![0]),
                    Arc::new(values.slice(0, len)),
                )
                .unwrap();
            RecordBatch::try_from_iter([
                ("x", Arc::new(Int32Array::from(vec![1])) as ArrayRef),
                ("d", Arc::new(d) as ArrayRef),
            ])
            .unwrap()
        };
        let batches = [batch(1 << 30), batch((1 << 30) + 1)];
        let schema = batches[0].schema();
        let plan = || filter_exec("x > 0", scan_batches(&schema, &batches), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Dictionary key bigger than the key type"],
        )
        .await;
        assert!(
            outcome.as_ref().is_err_and(
                |error| error.contains("Dictionary key bigger than the key type")
            ),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_dictionaries_of_values_without_memory_are_native() {
        // `FilterExec` concatenates the dictionaries of the batches it
        // coalesces: Arrow cannot merge dictionaries of nulls, which take no
        // memory, so it adds up their values, here `(1 << 31) + 1`, more than
        // `Int32` keys can index. The native filter does not coalesce.
        let batch = |values: usize| {
            let d =
                arrow::array::DictionaryArray::<arrow::datatypes::Int32Type>::try_new(
                    Int32Array::from(vec![0]),
                    Arc::new(arrow::array::NullArray::new(values)),
                )
                .unwrap();
            RecordBatch::try_from_iter([
                ("x", Arc::new(Int32Array::from(vec![1])) as ArrayRef),
                ("d", Arc::new(d) as ArrayRef),
            ])
            .unwrap()
        };
        let batches = [batch(1 << 30), batch((1 << 30) + 1)];
        let schema = batches[0].schema();
        let plan = || filter_exec("x > 0", scan_batches(&schema, &batches), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Dictionary key bigger than the key type"],
        )
        .await;
        assert!(
            outcome.as_ref().is_err_and(
                |error| error.contains("Dictionary key bigger than the key type")
            ),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_whose_output_dictionaries_may_overflow_are_native() {
        // One partition of two batches of 100 rows, with 100 distinct values
        // each, in dictionaries with `Int8` keys. `FilterExec` coalesces them
        // into one batch, and merging their 200 values overflows the keys.
        // The native filter does not coalesce its output: it selected the 200
        // rows.
        let batches = distinct_dictionary_batches(&DataType::Int8);
        let schema = batches[0].schema();
        let predicate = "d <> 'missing'";
        let plan = || filter_exec(predicate, scan_batches(&schema, &batches), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let overflow = "Dictionary key bigger than the key type";
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            plan,
            false,
            &["Dictionary key bigger than the key type"],
        )
        .await;
        assert!(error.unwrap().contains(overflow));
        assert_eq!(operators.filters, 1);
        let filter = filter_for(predicate, &schema);
        assert!(drive(&filter, batches.iter().cloned().map(DataChunk::from)).is_ok());

        // The same in SQL, on one target partition (with more, DataFusion
        // distributes the batches to several filters).
        let setup = async |ctx: &datafusion::prelude::SessionContext| {
            register_partitions(ctx, "narrow", vec![batches.clone()])
        };
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_outcome_with_config(
            "SELECT * FROM narrow WHERE d <> 'missing'",
            test_session_config().with_target_partitions(1),
            setup,
            &["Dictionary key bigger than the key type"],
        )
        .await;
        assert!(error.unwrap().contains(overflow));
        assert_eq!(operators.filters, 1);

        // Without the dictionaries in its output, or with keys of 32 bits,
        // which can index the values of any batch (of strings with 64-bit
        // offsets: Arrow may concatenate the values), the filter is lowered.
        let projected = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    physical_expr(predicate, &schema),
                    scan_batches(&schema, &batches),
                )
                .apply_projection(Some(vec![1]))
                .unwrap()
                .build()
                .unwrap(),
            )
        };
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            projected,
            false,
            &["Dictionary key bigger than the key type"],
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
        let large = DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::LargeUtf8),
        );
        let wide = distinct_dictionary_batches(&DataType::Int32)
            .into_iter()
            .map(|batch| {
                let d = arrow::compute::cast(batch.column(0), &large).unwrap();
                RecordBatch::try_from_iter([("d", d), ("i", Arc::clone(batch.column(1)))])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec(predicate, scan_batches(&wide[0].schema(), &wide), None),
            false,
            &["Dictionary key bigger than the key type"],
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_lists_with_fallible_values_are_native() {
        // One batch of 4096 rows, `(x, y) = (1, 0)` then `(2, 1)`. `x = 1` is
        // not true on all rows of the batch, so DataFusion evaluates `10 / y`
        // on all of them, and fails. But it is true on all rows of the first
        // chunk of the pipeline, and `y = 1` on all rows of the second one.
        let batch = xys(&[((1, 0), 2048), ((2, 1), 2048)]);
        for predicate in ["x IN (1, 10 / y)", "x NOT IN (1, 10 / y)"] {
            let plan = filter_exec(predicate, scan(&batch), None);
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter_exec(predicate, scan(&batch), None),
                false,
                &["Divide by zero"],
            )
            .await;
            assert!(error.unwrap().contains("Divide by zero"), "{predicate}");
            assert_eq!(operators.filters, 1, "{predicate}");

            // On the chunks of the pipeline, the predicate does not fail.
            let filter = filter_for(predicate, &batch.schema());
            let chunks =
                [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
            assert!(drive(&filter, chunks).is_ok(), "{predicate}");
        }
        // The first value is evaluated on all rows.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec("x IN (10 / y, 1)", scan(&batch), None),
            false,
            &["Divide by zero"],
        )
        .await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn patterns_that_fail_behind_short_circuits_are_native() {
        // One batch of 4096 rows, `(x, y) = (0, 1)` then `(1, 0)`. Neither
        // `x = 0` nor `y = 0` is true on all rows of the batch, so DataFusion
        // evaluates `s ~ 'a{1000000}'`, whose pattern is too big to compile.
        // But `x = 0` is true on all rows of the first chunk of the pipeline,
        // and `y = 0` on all rows of the second one.
        let batch = xys(&[((0, 1), 2048), ((1, 0), 2048)]);
        let predicate = "x = 0 OR (y = 0 OR s ~ 'a{1000000}')";
        let plan = filter_exec(predicate, scan(&batch), None);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec(predicate, scan(&batch), None),
            false,
            &["did not compile"],
        )
        .await;
        assert!(error.unwrap().contains("did not compile"));
        assert_eq!(operators.filters, 1);

        // On the chunks of the pipeline, the predicate does not fail.
        let filter = filter_for(predicate, &batch.schema());
        let chunks = [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
        assert!(drive(&filter, chunks).is_ok());

        // A pattern that compiles cannot fail.
        let predicate = "x = 0 OR (y = 0 OR s ~ 'a+')";
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec(predicate, scan(&batch), None),
            false,
            &["did not compile"],
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn predicates_that_fail_on_empty_batches_are_native() {
        // DataFusion evaluates the predicate on every batch, empty ones
        // included, on which `OR` evaluates its right side: after an empty
        // batch, `x = 0 OR s ~ 'a{1000000}'` fails even if `x = 0` on all
        // rows. Without batches, it is not evaluated at all.
        let schema = xys(&[]).schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let zeros = xys(&[((0, 0), 2048)]);
        for (predicate, batches, fails) in [
            (
                "x = 0 OR s ~ 'a{1000000}'",
                vec![empty.clone(), zeros.clone()],
                true,
            ),
            ("x = 0 OR s ~ 'a{1000000}'", vec![zeros.clone()], false),
            ("x = 0 OR s ~ 'a{1000000}'", vec![], false),
            // Functions may fail on empty batches too.
            ("regexp_like(s, 'a{1000000}')", vec![empty.clone()], true),
            ("regexp_like(s, 'a{1000000}')", vec![], false),
            // (`regexp_count` only compiles its pattern if there are rows, but
            // functions are opaque, see `functions_are_native`.)
            (
                "regexp_count(s, 'a{1000000}') > 0",
                vec![empty.clone()],
                false,
            ),
            (
                "regexp_count(s, 'a{1000000}') > 0",
                vec![empty.clone(), zeros.clone()],
                true,
            ),
        ] {
            let plan = || filter_exec(predicate, scan_batches(&schema, &batches), None);
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(plan, false, &["did not compile"])
                .await;
            let input = format!("{predicate} on {} batches", batches.len());
            assert_eq!(error.is_some(), fails, "{input}: {error:?}");
            assert_eq!(operators.filters, 1, "{input}");
        }
        // A pattern that compiles cannot fail, with or without empty batches.
        for (predicate, batches, fails) in [
            ("x = 0 OR s ~ 'a+'", vec![empty, zeros.clone()], false),
            ("x = 0 OR s ~ 'a+'", vec![zeros], false),
            ("x = 0 OR s ~ 'a+'", vec![], false),
        ] {
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter_exec(predicate, scan_batches(&schema, &batches), None),
                false,
                &["did not compile"],
            )
            .await;
            let input = format!("{predicate} on {} batches", batches.len());
            assert_eq!(error.is_some(), fails, "{input}: {error:?}");
            assert_eq!(operators.filters, 1, "{input}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_dense_unions_of_shared_strings_stay_in_datafusion() {
        let _large = one_large_test_at_a_time().await;
        // Rows of a dense union may point to the same value of a child, which
        // Arrow copies for every row when it extracts the child (casting the
        // union) or filters the union: the 32-bit offsets of 1200 copies of a
        // string of 2 MiB overflow on the batch of `FilterExec`, but not those
        // of 600 copies on the chunks of the native filter.
        let batch = dense_union_of_a_large_string();
        let schema = batch.schema();
        let cast = Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(
            Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
                physical_expr("u", &schema),
                DataType::Utf8,
                None,
            )),
        )) as Arc<dyn PhysicalExpr>;
        for predicate in [cast, physical_expr("id > 0", &schema)] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    FilterExec::try_new(
                        Arc::clone(&predicate),
                        scan_batches(&schema, std::slice::from_ref(&batch)),
                    )
                    .unwrap(),
                )
            };
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let outcome = assert_same_outcome_in_datafusion(
                plan(),
                Arc::new(TaskContext::default()),
            )
            .await;
            assert!(
                outcome
                    .as_ref()
                    .is_err_and(|error| error.to_lowercase().contains("overflow")),
                "{predicate}: {outcome:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn broadcast_constants_are_native() {
        // `CASE WHEN x THEN '…' END` copies a string of 2 MiB for every row
        // with `x`: the 32-bit offsets of 2048 copies overflow. `FilterExec`
        // evaluates it on the batch, with 2048 such rows, and panics, but the
        // native filter on chunks of 2048 rows, on which `CASE` returns the
        // constant itself, or nulls.
        let batch = half_true();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                FilterExec::try_new(
                    case_of_a_large_constant(&schema),
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["offset overflow"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.to_lowercase().contains("overflow")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unpacking_repeated_values_is_native() {
        // Unpacking a dictionary copies its values for every row that has
        // them: the 32-bit offsets of 1536 strings of 2 MiB overflow.
        // `FilterExec` unpacks all rows of the batch (a quarter of them pass
        // `id < 512`), and fails, but the native filter only the rows that
        // pass `id < 512`, whose keys are null.
        let batch = repeated_large_string(2048, 512);
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                FilterExec::try_new(
                    guarded_unpacking(&schema, 512),
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Offset overflow error"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.to_lowercase().contains("overflow")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_of_dictionary_keys_are_native() {
        // Casting a dictionary into another casts its keys, which index its
        // values, however many (nulls take no memory): `1 << 31` does not fit
        // in `Int32` keys. `FilterExec` evaluates the cast on all rows of the
        // batch (half of them pass `x = 0`), and fails, but the native filter
        // only on the rows that pass `x = 0`.
        let batch = large_dictionary_keys();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                FilterExec::try_new(
                    guarded_key_cast(&schema),
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let outcome = assert_allowed_native_outcome(
            plan(),
            Arc::new(TaskContext::default()),
            &["Can't cast value 2147483648 to type Int32"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Can't cast value 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn predicates_on_dictionary_values_are_native() {
        // The first `FilterExec` coalesces the rows `"1"` and `"2"` into one
        // batch, merging the dictionaries of the two batches without the
        // values that no row references, `"x"` and `"y"`, so DataFusion casts
        // `"1"` and `"2"`. The chunks of a pipeline keep their dictionaries:
        // the native filters may fail to cast `"x"` (a value of no row), which
        // is the engine's own.
        let batches = dictionary_batches();
        let schema = batches[0].schema();
        let plan = || {
            let input = filter_exec(
                "d <> 'x' AND d <> 'y'",
                scan_batches(&schema, &batches),
                None,
            );
            filter_exec("CAST(d AS INT) > 0", input, None)
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Cannot cast string"])
            .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 2);
        let chunk = DataChunk::from(batches[0].clone())
            .with_selection(SelectionVector::from(vec![0]))
            .unwrap();
        assert!(drive(&filter_for("CAST(d AS INT) > 0", &schema), [chunk]).is_err());
    }

    /// `GlobalLimitExec(1)` above `FilterExec(q >= 0)`, coalescing its output
    /// into batches of `filter_batch_size` rows, above
    /// `ProjectionExec(10 / x AS q)`, over a batch of `rows` ones and a batch
    /// of `rows` zeros.
    fn limited_filter(rows: usize, filter_batch_size: usize) -> Arc<dyn ExecutionPlan> {
        let (ones, zeros) = (runs(&[(1, rows)]), runs(&[(0, rows)]));
        let schema = ones.schema();
        let projection = Arc::new(
            datafusion_physical_plan::projection::ProjectionExec::try_new(
                [datafusion_physical_expr::projection::ProjectionExpr::new(
                    physical_expr("10 / x", &schema),
                    "q",
                )],
                scan_batches(&schema, &[ones, zeros]),
            )
            .unwrap(),
        );
        let predicate = physical_expr("q >= 0", &projection.schema());
        let filter = datafusion_physical_plan::filter::FilterExecBuilder::new(
            predicate, projection,
        )
        .with_batch_size(filter_batch_size)
        .build()
        .unwrap();
        Arc::new(datafusion_physical_plan::limit::GlobalLimitExec::new(
            Arc::new(filter),
            0,
            Some(1),
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn plans_below_limits_are_native() {
        // `FilterExec` pulls its input until it has a batch of its batch
        // size, or its input ends: with batches of 8192 rows, it pulls the
        // zeros, which the projection divides by, before the limit is reached.
        // A native filter, which does not coalesce its output, reaches it with
        // the ones.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || limited_filter(2048, 8192),
            true,
            &["Divide by zero"],
        )
        .await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!((operators.filters, operators.projections), (1, 1));
        // With batches of 32 rows, `FilterExec` returns the ones, and the limit
        // is reached before it pulls the zeros. The native cache keeps the 32
        // rows of a chunk, and pulls the zeros.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || limited_filter(32, 32),
            true,
            &["Divide by zero"],
        )
        .await;
        assert!(error.is_none());
        assert_eq!((operators.filters, operators.projections), (1, 1));
    }

    /// `FilterExec(q >= 0)`, coalescing its output into batches of 32 rows,
    /// above `ProjectionExec(id, 10 / x AS q)`, over a batch of 32 rows with
    /// `x = 1` then a batch of 32 rows with `x = 0`, ordered by `id` (`0` to
    /// `63`), which the scan declares if `ordered`.
    fn filtered_divisions(ordered: bool) -> Arc<dyn ExecutionPlan> {
        use datafusion_physical_expr::projection::ProjectionExpr;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Int32, false),
        ]));
        let batch = |first: i64, x: i32| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(first..first + 32)),
                    Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(x, 32))),
                ],
            )
            .unwrap()
        };
        let mut source = datafusion_datasource::memory::MemorySourceConfig::try_new(
            &[vec![batch(0, 1), batch(32, 0)]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        if ordered {
            let id = LexOrdering::new([PhysicalSortExpr::new_default(physical_expr(
                "id", &schema,
            ))])
            .unwrap();
            source = source.try_with_sort_information(vec![id]).unwrap();
        }
        let projection = Arc::new(
            datafusion_physical_plan::projection::ProjectionExec::try_new(
                [
                    ProjectionExpr::new(physical_expr("id", &schema), "id"),
                    ProjectionExpr::new(physical_expr("10 / x", &schema), "q"),
                ],
                datafusion_datasource::source::DataSourceExec::from_data_source(source),
            )
            .unwrap(),
        );
        let predicate = physical_expr("q >= 0", &projection.schema());
        Arc::new(
            datafusion_physical_plan::filter::FilterExecBuilder::new(
                predicate, projection,
            )
            .with_batch_size(32)
            .build()
            .unwrap(),
        )
    }

    /// A sort of [`filtered_divisions`] by `keys`, with `fetch`.
    fn sorted_divisions(
        keys: &[&str],
        fetch: Option<usize>,
        ordered: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let input = filtered_divisions(ordered);
        let ordering = LexOrdering::new(keys.iter().map(|key| {
            PhysicalSortExpr::new_default(physical_expr(key, &input.schema()))
        }))
        .unwrap();
        Arc::new(SortExec::new(ordering, input).with_fetch(fetch))
    }

    /// `GlobalLimitExec(1)` above `input`.
    fn limit_one(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(datafusion_physical_plan::limit::GlobalLimitExec::new(
            input,
            0,
            Some(1),
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_that_may_stop_early_are_native() {
        // The input of the sort is ordered by `id`. A sort by `id` streams it,
        // and a sort by `(id, q)` with a fetch (a TopK) stops after a batch
        // once it has its rows and the next `id`s are greater: both stop
        // before `FilterExec` pulls the zeros, which the projection divides
        // by. A native cache, which keeps the 32 rows of the first chunk,
        // would pull them.
        let stopping: [fn() -> Arc<dyn ExecutionPlan>; 2] = [
            || limit_one(sorted_divisions(&["id"], None, true)),
            || sorted_divisions(&["id", "q"], Some(1), true),
        ];
        for plan in stopping {
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(plan, true, &["Divide by zero"]).await;
            assert!(error.is_none(), "{error:?}");
            assert_eq!((operators.filters, operators.projections), (1, 1));
        }
        // A sort whose order its input does not satisfy reads all of its
        // input, in DataFusion as natively: what is below it is lowered.
        let reading: [fn() -> Arc<dyn ExecutionPlan>; 2] = [
            || limit_one(sorted_divisions(&["id"], None, false)),
            || sorted_divisions(&["id", "q"], Some(1), false),
        ];
        for plan in reading {
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(plan, true, &["Divide by zero"]).await;
            assert!(error.unwrap().contains("Divide by zero"));
            assert_eq!(operators.filters, 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_that_arrow_cannot_evaluate_are_native() {
        // On one batch of `(x, y) = (0, 1)` then `(1, 0)` rows, DataFusion
        // evaluates the cast, as neither `x = 0` nor `y = 0` is true on all of
        // its rows, and fails; on the chunks of a pipeline, `x = 0` and then
        // `y = 0` are true on all rows.
        let mixed = xys(&[((0, 1), 2048), ((1, 0), 2048)]);
        let schema = mixed.schema();
        // DataFusion also evaluates it on an empty batch, which a pipeline
        // skips.
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let nested = || nested_packed_booleans(&schema);
        let alone = || -> Arc<dyn PhysicalExpr> {
            Arc::new(datafusion_physical_expr::expressions::IsNotNullExpr::new(
                packed_booleans(&schema),
            ))
        };
        for (predicate, batch) in [
            (&nested as &dyn Fn() -> Arc<dyn PhysicalExpr>, &mixed),
            (&alone, &empty),
        ] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    FilterExec::try_new(
                        predicate(),
                        scan_batches(&schema, std::slice::from_ref(batch)),
                    )
                    .unwrap(),
                )
            };
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec));
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                plan,
                false,
                &["Unsupported output type for dictionary packing"],
            )
            .await;
            let error = error.unwrap();
            assert!(error.contains("dictionary packing"), "{error}");
            assert_eq!(operators.filters, 1);
        }
        // In SQL, the `OR`s are reassociated: the filter and the projection
        // are native too.
        let setup = async |ctx: &datafusion::prelude::SessionContext| {
            register_partitions(ctx, "t", vec![vec![mixed.clone()]])
        };
        let predicate = "x = 0 OR (y = 0 OR \
            arrow_cast(x = 0, 'Dictionary(Int32, Boolean)') IS NOT NULL)";
        for sql in [
            format!("SELECT * FROM t WHERE {predicate}"),
            format!("SELECT {predicate} AS m FROM t"),
        ] {
            let NativeOutcome { operators, .. } =
                assert_allowed_native_outcome_with_setup(
                    &sql,
                    setup,
                    &["Unsupported output type for dictionary packing"],
                )
                .await;
            assert!(operators.filters + operators.projections > 0, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_lists_of_nulls_stay_in_datafusion() {
        // Lists of `1 << 30` nulls, which take no memory: the native operators
        // cannot gather the rows of such lists (see `can_gather`), so the
        // filter stays in DataFusion. (`FilterExec` coalesces the first 256
        // rows of both batches into one, whose lists have `2 << 30` nulls: more
        // than their 32-bit offsets can count, and Arrow's `concat` panics.)
        let batches = long_lists_of_nulls();
        let schema = batches[0].schema();
        let plan = filter_exec("x > 0", scan_batches(&schema, &batches), None);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let overflow = assert_same_outcome_in_datafusion(plan, task_ctx).await;
        assert!(overflow.unwrap_err().contains("offset overflow"));

        // The same in SQL, on one target partition.
        let config = test_session_config().with_target_partitions(1);
        let ctx = crate::test_utils::datafusion_context(config);
        register_partitions(&ctx, "lists", vec![batches]).unwrap();
        let df = ctx.sql("SELECT * FROM lists WHERE x > 0").await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let overflow = assert_same_outcome_in_datafusion(plan, ctx.task_ctx()).await;
        assert!(overflow.unwrap_err().contains("offset overflow"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lists_built_per_batch_stay_in_datafusion() {
        // `CASE WHEN x THEN a ELSE b END` merges the lists of `a` on the first
        // 2048 rows (`1 << 30` nulls) and of `b` on the others (`1 << 30` nulls
        // too): on the batch of 4096 rows, their 32-bit offsets overflow, on
        // each chunk of 2048 rows of a pipeline, they do not. But the native
        // operators cannot gather the rows of lists of nulls, which take no
        // memory (see `can_gather`): the filter stays in DataFusion.
        let batch = lists_to_merge();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let case = physical_expr("CASE WHEN x THEN a ELSE b END", &schema);
            let predicate = Arc::new(
                datafusion_physical_expr::expressions::IsNotNullExpr::new(case),
            );
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    predicate,
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .apply_projection(Some(vec![0]))
                .unwrap()
                .build()
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let outcome =
            assert_same_outcome_in_datafusion(plan(), Arc::new(TaskContext::default()))
                .await;
        assert!(outcome.unwrap_err().contains("overflow"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lists_of_tiny_values_stay_in_datafusion() {
        // Arrow's `take`, with which the native operators gather rows (e.g. the
        // cache, to copy the rows it keeps), computes the indices of the values
        // of fixed-size lists in 32 bits: with lists of `1 << 30` nulls, of
        // runs, of empty lists (of any type) or binaries, which take no
        // memory, or of booleans (128 MiB per row), they overflow from the
        // fourth row on. DataFusion's `filter` does not.
        let cases: [(&[i32], TinyValues); 6] = [
            (&[0, 0, 0, 0, 0, 1], TinyValues::Nulls),
            (&[1, 0, 1, 0, 1, 1], TinyValues::Nulls),
            (&[0, 0, 0, 0, 1, 0, 1, 0], TinyValues::Runs),
            (&[0, 0, 0, 0, 0, 1, 0, 1], TinyValues::EmptyLists),
            (&[0, 0, 0, 0, 0, 1, 0, 1], TinyValues::EmptyBinaries),
            (&[0, 0, 0, 0, 1], TinyValues::Booleans),
        ];
        for (x, values) in cases {
            let batch = wide_lists(x, values);
            let schema = batch.schema();
            let plan = || {
                filter_exec(
                    "x > 0",
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                    None,
                )
            };
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec));
            let outcome = assert_same_outcome_in_datafusion(
                plan(),
                Arc::new(TaskContext::default()),
            )
            .await;
            assert_eq!(outcome, Ok(x.iter().filter(|&&x| x > 0).count()));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn predicates_with_several_failures_are_native() {
        // `FilterExec` evaluates `CAST(s AS INT)` on all rows of the batch
        // before `10 / x`, and fails to cast `"invalid"`; on the first chunk of
        // a pipeline, `"1"` casts, and `10 / x` divides by zero.
        let batch = xss(&[((0, "1"), 2048), ((1, "invalid"), 2048)]);
        let predicate = "CAST(s AS INT) + 10 / x > 0";
        let plan = || filter_exec(predicate, scan(&batch), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        // On one thread, a pipeline evaluates the first chunk first.
        let config = datafusion::prelude::SessionConfig::new().with_target_partitions(1);
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome_with_config(
            plan,
            false,
            config,
            &["Cannot cast string 'invalid'", "Divide by zero"],
        )
        .await;
        assert!(error.unwrap().contains("Cannot cast string 'invalid'"));
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn chains_with_several_failures_are_native() {
        // `ProjectionExec(x, CAST(s AS INT))` casts all rows of the batch, and
        // fails on `"invalid"`, before `FilterExec(10 / x > 1)` sees the batch;
        // a pipeline evaluates both operators on its first chunk first, and
        // divides by zero.
        let batch = xss(&[((0, "1"), 2048), ((1, "invalid"), 2048)]);
        let schema = batch.schema();
        let plan = |cast: &str| {
            let projection = Arc::new(
                datafusion_physical_plan::projection::ProjectionExec::try_new(
                    [
                        datafusion_physical_expr::projection::ProjectionExpr::new(
                            physical_expr("x", &schema),
                            "x",
                        ),
                        datafusion_physical_expr::projection::ProjectionExpr::new(
                            physical_expr(cast, &schema),
                            "c",
                        ),
                    ],
                    scan(&batch),
                )
                .unwrap(),
            );
            filter_exec("10 / x > 1", projection, None)
        };
        // Each operator alone could be lowered.
        let filter_plan = plan("CAST(s AS INT)");
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let projection = exec
            .input()
            .downcast_ref::<datafusion_physical_plan::projection::ProjectionExec>()
            .unwrap();
        assert!(
            crate::operators::projection::PhysicalProjection::supports_projection_exec(
                projection
            )
        );
        // On one thread, a pipeline evaluates the first chunk first.
        let config =
            || datafusion::prelude::SessionConfig::new().with_target_partitions(1);
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome_with_config(
            || plan("CAST(s AS INT)"),
            false,
            config(),
            &["Cannot cast string 'invalid'", "Divide by zero"],
        )
        .await;
        assert!(error.unwrap().contains("Cannot cast string 'invalid'"));
        assert_eq!((operators.filters, operators.projections), (1, 1));
        // Without a part that may fail in the projection, both are lowered.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome_with_config(
            || plan("x + 1"),
            false,
            config(),
            &["Cannot cast string 'invalid'", "Divide by zero"],
        )
        .await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!((operators.filters, operators.projections), (1, 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_runs_with_named_fields_stay_in_datafusion() {
        // Arrow's `filter` rebuilds run-end encoded arrays with its own names
        // of fields, so the output of `FilterExec` does not have its type any
        // more (and its coalescer fails); a native filter keeps the type.
        let batch = runs_with_named_fields();
        let schema = batch.schema();
        let plan = || {
            filter_exec(
                "x > 0",
                scan_batches(&schema, std::slice::from_ref(&batch)),
                None,
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let datafusion = outcome(plan(), Arc::clone(&task_ctx)).await;
        assert_eq!(
            assert_same_outcome_in_datafusion(plan(), task_ctx).await,
            datafusion
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operations_on_types_without_nulls_are_lowered() {
        // The classification probes expressions on columns of nulls of the
        // types of their operands, or on empty batches of their input, which
        // Arrow cannot build for every type: an empty union has no value, so
        // no null; a fixed-size list of 32768 run-end encoded values is one run
        // of 32768 nulls, which `Int16` run ends cannot count; and no array
        // has a fixed-size binary type of negative size, a dictionary type
        // with string keys, or a type of times of microseconds in 32 bits (but
        // a schema can).
        let empty_union = DataType::Union(
            arrow::datatypes::UnionFields::try_new(Vec::<i8>::new(), Vec::<Field>::new())
                .unwrap(),
            arrow::datatypes::UnionMode::Sparse,
        );
        let short_runs = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int16, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        );
        let lists_of_runs = DataType::FixedSizeList(
            Arc::new(Field::new_list_field(short_runs, true)),
            32768,
        );
        let string_keys =
            DataType::Dictionary(Box::new(DataType::Utf8), Box::new(DataType::Utf8));
        let mut types = vec![
            (empty_union, true),
            (lists_of_runs, true),
            (DataType::FixedSizeBinary(-1), false),
            (string_keys, false),
        ];
        // Nor times of these units in these widths.
        types.extend(times_without_arrays().into_iter().map(|time| (time, false)));
        for (data_type, has_arrays) in types {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Int32, false),
                Field::new("u", data_type.clone(), true),
            ]));
            // An empty batch, or none if there is no array of the type.
            let batches = if has_arrays {
                vec![RecordBatch::new_empty(Arc::clone(&schema))]
            } else {
                vec![]
            };
            // `u IS NULL` is probed on a null of the type of `u`, and
            // `10 / x > 1`, which can fail, on an empty batch, with a column
            // of the type of `u`.
            let is_null: Arc<dyn PhysicalExpr> =
                Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(
                    physical_expr("u", &schema),
                ));
            let mut predicates = vec![is_null];
            if !has_arrays {
                predicates.push(physical_expr("10 / x > 1", &schema));
            }
            for predicate in predicates {
                let plan = || -> Arc<dyn ExecutionPlan> {
                    Arc::new(
                        FilterExec::try_new(
                            Arc::clone(&predicate),
                            scan_batches(&schema, &batches),
                        )
                        .unwrap(),
                    )
                };
                let filter_plan = plan();
                let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
                assert_eq!(
                    PhysicalFilter::supports_filter_exec(exec),
                    can_gather(&data_type),
                    "{predicate} on {data_type}"
                );
                // (`assert_native_plan_parity` cannot encode the rows of types
                // without arrays.)
                let task_ctx = Arc::new(TaskContext::default());
                let lowered = executable(lower(&plan()).unwrap(), plan()).unwrap();
                assert_eq!(
                    outcome(lowered, Arc::clone(&task_ctx)).await,
                    Ok(0),
                    "{predicate} on {data_type}"
                );
                assert_eq!(outcome(plan(), task_ctx).await, Ok(0));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operations_on_unsupported_types_are_native() {
        // `NOT` of an integer fails whenever it is evaluated, on an empty batch
        // too, which a pipeline skips.
        let schema = runs(&[]).schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let plan = || -> Arc<dyn ExecutionPlan> {
            let not = Arc::new(datafusion_physical_expr::expressions::NotExpr::new(
                physical_expr("x", &schema),
            ));
            Arc::new(
                FilterExec::try_new(
                    not,
                    scan_batches(&schema, std::slice::from_ref(&empty)),
                )
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            plan,
            false,
            &["could not cast array of type Int32"],
        )
        .await;
        assert!(error.is_some());
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_zero_is_native() {
        // With `fetch = 0`, `FilterExec` evaluates its predicate on the first
        // batch of its input, even an empty one, and stops: it never pulls the
        // zeros, which its input divides by. Pipelines skip empty chunks: the
        // native filter pulls them, and may divide by zero, which is the
        // engine's own (an error for rows beyond a limit, which DataFusion
        // never evaluates). (The planner lowers a filter with a fetch of 0 to
        // a source of no rows.)
        let schema = runs(&[]).schema();
        let plan = || {
            let batches = [
                RecordBatch::new_empty(Arc::clone(&schema)),
                runs(&[(0, 10)]),
            ];
            let projection = Arc::new(
                datafusion_physical_plan::projection::ProjectionExec::try_new(
                    [datafusion_physical_expr::projection::ProjectionExpr::new(
                        physical_expr("10 / x", &schema),
                        "q",
                    )],
                    scan_batches(&schema, &batches),
                )
                .unwrap(),
            );
            filter_exec("q >= 0", projection, Some(0))
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, true, &["Divide by zero"]).await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_with_fallible_predicates_is_native() {
        // `FilterExec` evaluates its predicate on whole batches until the fetch
        // is reached, on the first batch even with `fetch = 0`.
        for (batch, predicate, fetch) in [
            (runs(&[(0, 10)]), "10 / x > 1", 0),
            (runs(&[(1, 2048), (0, 2048)]), "10 / x > 1", 1),
            (runs(&[(1, 2048), (0, 2048)]), "x >= 0 AND 100 % x = 1", 1),
        ] {
            let plan = filter_exec(predicate, scan(&batch), Some(fetch));
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter_exec(predicate, scan(&batch), Some(fetch)),
                true,
                &["Divide by zero", "did not compile"],
            )
            .await;
            assert_eq!(operators.filters, 1, "{predicate}");
            if predicate.contains('/') {
                assert!(error.unwrap().contains("Divide by zero"), "{predicate}");
            }
        }
        // A predicate that fails whenever it is evaluated, and a pattern that
        // compiles.
        // (Views, which `FilterExec` coalesces without 32-bit offsets.)
        let strings = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)])),
            vec![Arc::new(StringViewArray::from(vec!["a", "b"]))],
        )
        .unwrap();
        for (predicate, fails) in [("s ~ 'a{1000000}'", true), ("s LIKE 'a%'", false)] {
            let plan = filter_exec(predicate, scan(&strings), Some(1));
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || filter_exec(predicate, scan(&strings), Some(1)),
                true,
                &["Divide by zero", "did not compile"],
            )
            .await;
            assert_eq!(error.is_some(), fails, "{predicate}");
            assert_eq!(operators.filters, 1, "{predicate}");
        }
        // A predicate that cannot fail is lowered with its fetch.
        let batch = runs(&[(1, 2048), (0, 2048)]);
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || filter_exec("x >= 0", scan(&batch), Some(1)),
            true,
            &["Divide by zero", "did not compile"],
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fallible_operators_below_limits_are_native() {
        // A limit stops the execution after the batch in which it is
        // reached: DataFusion evaluates `10 / x` on all 4096 rows and fails,
        // a native projection only on the first chunk of 2048 ones.
        let batch = runs(&[(1, 2048), (0, 2048)]);
        let ones = runs(&[(1, 4096)]);
        let limit = |expr: &str, filter: bool, batch: &RecordBatch| {
            let input = scan(batch);
            let input: Arc<dyn ExecutionPlan> = if filter {
                filter_exec(expr, input, None)
            } else {
                let expr = physical_expr(expr, &input.schema());
                Arc::new(
                    datafusion_physical_plan::projection::ProjectionExec::try_new(
                        [datafusion_physical_expr::projection::ProjectionExpr::new(
                            expr, "q",
                        )],
                        input,
                    )
                    .unwrap(),
                )
            };
            Arc::new(datafusion_physical_plan::limit::GlobalLimitExec::new(
                input,
                0,
                Some(1),
            )) as Arc<dyn ExecutionPlan>
        };
        for (expr, filter) in [("10 / x", false), ("10 / x > 1", true)] {
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || limit(expr, filter, &batch),
                true,
                &["Divide by zero"],
            )
            .await;
            assert!(error.unwrap().contains("Divide by zero"), "{expr}");
            assert!(operators.filters + operators.projections > 0, "{expr}");
        }
        // So are infallible operators below the limit (see
        // `plans_below_limits_are_native`).
        for (expr, filter) in [("x + 1", false), ("x >= 0", true)] {
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || limit(expr, filter, &ones),
                true,
                &["Divide by zero"],
            )
            .await;
            assert!(error.is_none(), "{expr}");
            assert!(operators.filters + operators.projections > 0, "{expr}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    // `with_limit_options` is the only public way to build a soft limit.
    #[expect(deprecated)]
    async fn fallible_operators_below_soft_limits_are_native() {
        use datafusion_physical_plan::aggregates::{
            AggregateExec, AggregateMode, LimitOptions, PhysicalGroupBy,
        };
        // `SELECT DISTINCT x ... LIMIT 1`: the aggregation stops reading its
        // input after the batch in which it has one group. DataFusion divides
        // all 4096 rows of the batch, a native filter only the first chunk.
        let batch = runs(&[(1, 2048), (0, 2048)]);
        let plan = || {
            let filter = filter_exec("10 / x > 0", scan(&batch), None);
            let schema = filter.schema();
            let group_by = PhysicalGroupBy::new_single(vec![(
                physical_expr("x", &schema),
                "x".to_owned(),
            )]);
            let aggregate = AggregateExec::try_new(
                AggregateMode::Partial,
                group_by,
                vec![],
                vec![],
                filter,
                schema,
            )
            .unwrap()
            .with_limit_options(Some(LimitOptions::new(1)));
            assert!(aggregate.limit_options().is_some());
            Arc::new(aggregate) as Arc<dyn ExecutionPlan>
        };
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Divide by zero"]).await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filter_with_fetch_is_lowered_with_its_limit() {
        let sql = "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0 LIMIT 7";
        let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
        let rows = run
            .actual
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(rows, 7);
        let description = crate::explain::format_operator_tree(run.root.operator());
        assert!(description.contains("fetch=7"), "{description}");
    }
}
