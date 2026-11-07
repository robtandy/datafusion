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
//! 2. The first conjunct is evaluated on the columns it references. The
//!    `true` rows of the resulting `BooleanArray` (`NULL` is not selected)
//!    become the selection.
//! 3. Every following conjunct is only evaluated on the rows selected so far
//!    (see below) and refines the selection. As soon as no row is left, the
//!    remaining conjuncts are skipped and no output is produced.
//! 4. The output chunk shares the columns of the input (`Arc` clones, after
//!    applying the projection embedded in the `FilterExec`, if any) and
//!    carries the selection. No column is copied: the columns that are
//!    not referenced by the predicate are never touched.
//!
//! A conjunct is evaluated on the selected rows by compacting (`take`) only
//! the columns it references, so it never sees a row that a previous
//! conjunct removed: `x <> 0 AND 10 / x > 1` never divides by zero. If the
//! conjunct cannot fail (see
//! [`is_infallible`](super::adaptive_filter::is_infallible)) and most rows
//! are still selected, it is evaluated on all physical rows instead, which
//! avoids the copy; its result for the rows that are not selected is ignored.
//!
//! Since `p AND q` is `true` exactly when both `p` and `q` are `true`, the
//! selected rows do not depend on the order of the conjuncts: they are the
//! rows for which the predicate is `true`, as in DataFusion.

use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, AsArray, BooleanArray, RecordBatch};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::instant::Instant;
use datafusion_common::{Result, ScalarValue, internal_err, plan_err};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::filter::FilterExec;
use itertools::Itertools;

use super::adaptive_filter::{AdaptiveFilter, AdaptiveFilterState};
use super::projection::{BoundExprs, check_lowered_input, logical_rows};
use crate::chunk::{DataChunk, SelectionVector};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, downcast_state, downcast_state_mut,
};

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
    /// The maximum number of rows to produce, over all threads.
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

    /// Lower a DataFusion [`FilterExec`]: apply its predicate, projection and
    /// fetch to the chunks of `input` (the lowered input of `exec`) and
    /// produce exactly its output schema.
    ///
    /// The `fetch` of a `FilterExec` limits the rows of each of its output
    /// partitions; the `fetch` of a `PhysicalFilter` limits the rows over all
    /// threads, which is what the limit above such a `FilterExec` enforces.
    pub fn try_from_filter_exec(
        exec: &FilterExec,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
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

    /// Produce at most `fetch` rows in total (over all threads), then
    /// return [`OperatorResult::Finished`].
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

/// The result of evaluating a conjunct.
enum Mask {
    /// All rows are `true`.
    All,
    /// No row is `true`.
    Nothing,
    /// Some rows are `true`.
    Some(BooleanArray),
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
        let physical_rows = input.physical_len();
        let incoming = logical_rows(input);
        // The rows selected so far, `None` if they are the rows of `incoming`.
        let mut selected: Option<SelectionVector> = None;
        let timed = state.is_adaptive();
        for position in 0..self.conjuncts.len() {
            let conjunct = state.permutation()[position];
            let rows = selected.as_ref().or(incoming);
            let input_rows = rows.map_or(physical_rows, SelectionVector::len);
            let start = timed.then(Instant::now);
            if let Some(refined) =
                self.evaluate_conjunct(conjunct, input, rows, physical_rows)?
            {
                selected = Some(refined);
            }
            let selected_rows = selected
                .as_ref()
                .or(incoming)
                .map_or(physical_rows, SelectionVector::len);
            if let Some(start) = start {
                state.record(conjunct, input_rows, selected_rows, start.elapsed());
            }
            if selected_rows == 0 {
                state.end_chunk();
                return Ok(Selected::Nothing);
            }
        }
        state.end_chunk();
        Ok(match selected {
            Some(rows) => Selected::Rows(rows),
            None => Selected::All,
        })
    }

    /// Evaluate conjunct `conjunct` on the physical rows `rows` of `input`
    /// (all physical rows if `None`). Returns the rows of `rows` for which it
    /// is `true`, or `None` if it is `true` for all of them.
    fn evaluate_conjunct(
        &self,
        conjunct: usize,
        input: &DataChunk,
        rows: Option<&SelectionVector>,
        physical_rows: usize,
    ) -> Result<Option<SelectionVector>> {
        let bound = &self.conjuncts[conjunct];
        let expr = &bound.exprs()[0];
        let Some(rows) = rows else {
            let batch = bound.batch(input, None)?;
            return Ok(match evaluate_mask(expr, &batch)? {
                Mask::All => None,
                Mask::Nothing => Some(SelectionVector::new(Vec::<u32>::new())),
                Mask::Some(mask) => Some(SelectionVector::from_boolean_mask(&mask)),
            });
        };
        let infallible = !self.filter.conjuncts()[conjunct].is_fallible();
        if infallible && rows.len() * 2 > physical_rows {
            // Most rows are selected and the conjunct cannot fail on the
            // other ones: evaluating it on all physical rows is cheaper than
            // gathering its columns.
            let batch = bound.batch(input, None)?;
            return Ok(match evaluate_mask(expr, &batch)? {
                Mask::All => None,
                Mask::Nothing => Some(SelectionVector::new(Vec::<u32>::new())),
                Mask::Some(mask) => Some(refine(rows, &mask)),
            });
        }
        // Only evaluate the conjunct on the selected rows.
        let batch = bound.batch(input, Some(rows))?;
        Ok(match evaluate_mask(expr, &batch)? {
            Mask::All => None,
            Mask::Nothing => Some(SelectionVector::new(Vec::<u32>::new())),
            Mask::Some(mask) => {
                Some(rows.slice_sel(&SelectionVector::from_boolean_mask(&mask))?)
            }
        })
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
/// result must be a boolean, and only `true` rows are selected.
fn evaluate_mask(expr: &Arc<dyn PhysicalExpr>, batch: &RecordBatch) -> Result<Mask> {
    let num_rows = batch.num_rows();
    let array = match expr.evaluate(batch)? {
        ColumnarValue::Scalar(ScalarValue::Boolean(value)) => {
            return Ok(if value == Some(true) {
                Mask::All
            } else {
                Mask::Nothing
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
    Ok(match mask.true_count() {
        0 => Mask::Nothing,
        selected if selected == num_rows => Mask::All,
        _ => Mask::Some(mask.clone()),
    })
}

/// The rows of `rows` for which `mask`, which covers all physical rows, is
/// `true`.
fn refine(rows: &SelectionVector, mask: &BooleanArray) -> SelectionVector {
    let selected = match mask.nulls() {
        Some(nulls) if nulls.null_count() > 0 => mask.values() & nulls.inner(),
        _ => mask.values().clone(),
    };
    SelectionVector::new(
        rows.indices()
            .iter()
            .copied()
            .filter(|&row| selected.value(row as usize))
            .collect::<Vec<_>>(),
    )
}
