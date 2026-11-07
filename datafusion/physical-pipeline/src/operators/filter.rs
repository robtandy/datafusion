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
//! [`is_infallible`](super::adaptive_filter::is_infallible)) and enough rows
//! are still selected, it is evaluated on all physical rows instead, which
//! avoids the copy; its result for the rows that are not selected is ignored.
//!
//! Since `p AND q` is `true` exactly when both `p` and `q` are `true`, the
//! selected rows do not depend on the order of the conjuncts: they are the
//! rows for which the predicate is `true`, as in DataFusion.

use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, SchemaRef};
use arrow_buffer::BooleanBuffer;
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

/// An infallible conjunct is evaluated on all physical rows of a chunk,
/// instead of only on the selected ones, if more than one in this many rows
/// are selected: gathering the columns of the selected rows (`take`) costs
/// more per row than evaluating a cheap conjunct.
const DENSE_SELECTION: usize = 4;

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
    /// The rows that are `true` (`NULL` is not `true`), and their number.
    Some(BooleanBuffer, usize),
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
                Mask::Some(mask, count) => {
                    let mut selected = Vec::with_capacity(count);
                    selected.extend(mask.set_indices_u32());
                    Some(SelectionVector::from(selected))
                }
            });
        };
        let infallible = !self.filter.conjuncts()[conjunct].is_fallible();
        if infallible && rows.len() * DENSE_SELECTION > physical_rows {
            // Many rows are selected and the conjunct cannot fail on the
            // other ones: evaluating it on all physical rows is cheaper than
            // gathering its columns.
            let batch = bound.batch(input, None)?;
            return Ok(match evaluate_mask(expr, &batch)? {
                Mask::All => None,
                Mask::Nothing => Some(SelectionVector::new(Vec::<u32>::new())),
                Mask::Some(mask, _) => {
                    let mut selected = Vec::with_capacity(rows.len());
                    selected.extend(
                        rows.indices()
                            .iter()
                            .filter(|&&row| mask.value(row as usize)),
                    );
                    Some(SelectionVector::from(selected))
                }
            });
        }
        // Only evaluate the conjunct on the selected rows.
        let batch = bound.batch(input, Some(rows))?;
        Ok(match evaluate_mask(expr, &batch)? {
            Mask::All => None,
            Mask::Nothing => Some(SelectionVector::new(Vec::<u32>::new())),
            Mask::Some(mask, count) => {
                // `mask` has an entry for every row of `rows`.
                let rows = rows.indices();
                let mut selected = Vec::with_capacity(count);
                selected.extend(mask.set_indices().map(|row| rows[row]));
                Some(SelectionVector::from(selected))
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
    // `NULL` is not selected.
    let selected = match mask.nulls() {
        Some(nulls) if nulls.null_count() > 0 => mask.values() & nulls.inner(),
        _ => mask.values().clone(),
    };
    Ok(match selected.count_set_bits() {
        0 => Mask::Nothing,
        count if count == num_rows => Mask::All,
        count => Mask::Some(selected, count),
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
        DictionaryArray, Int32Array, Int64Array, RecordBatch, StringArray,
        StringViewArray,
    };
    use arrow::datatypes::{DataType, Field, Int8Type, Int32Type, Schema, SchemaRef};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::{DFSchema, Result};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::filter::FilterExec;
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
    use crate::operators::caching::CachingOperator;
    use crate::operators::projection::PhysicalProjection;
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, has_top_level_order_by,
        register_test_tables, run_sql, test_session_config,
    };

    /// Lower `plan` into an operator tree that executes every `FilterExec`
    /// and `ProjectionExec` natively.
    ///
    /// * `FilterExec` becomes `CachingOperator(PhysicalFilter)`, and
    ///   `ProjectionExec` becomes `PhysicalProjection`.
    /// * `CoalescePartitionsExec` (without fetch) and round robin
    ///   `RepartitionExec`s above native operators are removed: they only
    ///   change the partitioning, which the pipeline engine does not need.
    /// * Every other node is executed by DataFusion (in a `BridgeOperator`),
    ///   see [`bridged_plan`].
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            let input = lower(filter.input())?;
            let filter: Arc<dyn PhysicalOperator> =
                Arc::new(PhysicalFilter::try_from_filter_exec(filter, input)?);
            return Ok(Arc::new(CachingOperator::new(filter)));
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            let input = lower(projection.input())?;
            return Ok(Arc::new(PhysicalProjection::try_from_projection_exec(
                projection, input,
            )?));
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
    /// order of the child and the child has several partitions, which a
    /// `PipelineRootExec` (a single partition) does not keep. DataFusion then
    /// executes the child too, with its own children lowered.
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
        let unconstrained = matches!(
            parent
                .input_distribution_requirements()
                .child_distribution(index),
            None | Some(
                Distribution::UnspecifiedDistribution | Distribution::SinglePartition
            )
        ) && child.output_ordering().is_none();
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

    /// Parse the SQL expression `sql` on `schema` into a physical expression,
    /// with DataFusion's type coercion.
    pub(crate) fn physical_expr(sql: &str, schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        let ctx = SessionContext::new();
        let df_schema = DFSchema::try_from(schema.as_ref().clone()).unwrap();
        let expr = ctx.parse_sql_expr(sql, &df_schema).unwrap();
        ctx.create_physical_expr(expr, &df_schema).unwrap()
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
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::filter::batch_filter;

    use crate::bridge::PipelineRootExec;
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

    #[test]
    fn fallible_conjuncts_only_see_the_rows_selected_before_them() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, false),
        ]));
        // More than half of the rows have `x = 0`. (`FilterExec` only removes
        // the rows of a previous conjunct before evaluating a fallible one if
        // at most 20% of the rows are left, so DataFusion would divide by
        // zero here. The pipeline never evaluates a conjunct on a row that a
        // previous one removed.)
        let batches = (0..30)
            .map(|b| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int32Array::from_iter(
                            (0..2048).map(|i| (i % 13 != 0).then_some(i % 2 * (i % 7))),
                        )),
                        Arc::new(Int32Array::from_iter_values(
                            (0..2048).map(|i| (b * 2048 + i) % 101),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let predicate = "x <> 0 AND 10 / x > 1 AND y = 5";
        let filter = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr(predicate, &schema),
            None,
        )
        .unwrap();
        assert_eq!(
            filter
                .adaptive_filter()
                .conjuncts()
                .iter()
                .map(|c| c.is_fallible())
                .collect::<Vec<_>>(),
            vec![false, true, false]
        );
        let (gstate, mut lstate) = states(&filter);
        for batch in &batches {
            let input = DataChunk::from(batch.clone());
            let (outputs, _) = drive_with_states(
                &filter,
                [input.clone()],
                gstate.as_ref(),
                lstate.as_mut(),
            )
            .unwrap();
            let rows = outputs
                .iter()
                .map(|chunk| chunk.compact().unwrap())
                .collect::<Vec<_>>();
            // DataFusion evaluates the predicate on the rows with `x <> 0`.
            let guarded = batch_filter(batch, &physical_expr("x <> 0", &schema)).unwrap();
            let reference =
                batch_filter(&guarded, &physical_expr(predicate, &schema)).unwrap();
            assert!(reference.num_rows() > 0);
            assert_eq!(
                arrow::compute::concat_batches(&schema, &rows).unwrap(),
                reference
            );
            // The division is never evaluated before `x <> 0`.
            let permutation = lstate
                .downcast_ref::<FilterState>()
                .unwrap()
                .adaptive
                .permutation()
                .to_vec();
            let position = |c| permutation.iter().position(|&p| p == c).unwrap();
            assert!(position(0) < position(1), "{permutation:?}");
        }
        // (Whether `y = 5` moves before `x <> 0` depends on the measured
        // costs; see `adapts_the_order_of_the_conjuncts`.)
        let state = lstate.downcast_ref::<FilterState>().unwrap();
        assert!(state.adaptive.is_adaptive());
    }

    #[test]
    fn adapts_the_order_of_the_conjuncts() {
        // `b >= 0` selects everything, `b % 100 = 7` 1% of the rows.
        let filter = new_filter("b >= 0 AND s IS NOT NULL AND b % 100 = 7", None);
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
        // `b % 100 = 7` (arithmetic) is fallible: it stays after the others.
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
                expected("b >= 0 AND s IS NOT NULL AND b % 100 = 7", &input)
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
    fn fetch_limits_the_rows_over_all_threads() {
        let filter = new_filter("b % 10 = 1", None).with_fetch(Some(25));
        let ctx = context();
        let gstate = filter.get_global_operator_state(&ctx).unwrap();
        // Two threads, sharing the global state: 10 rows per chunk.
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
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_selective_filters() {
        assert_all_native_byte_eq(&[
            "SELECT * FROM t1 WHERE a > 10",
            "SELECT * FROM t1 WHERE id % 1000 = 7",
            "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
            "SELECT * FROM t1 WHERE id < 0",
            "SELECT * FROM t_empty WHERE a > 0",
            "SELECT * FROM t1 WHERE a IS NULL",
            "SELECT id, c FROM t1 WHERE c IS NULL OR c = 0.0 OR isnan(c)",
            "SELECT id, c FROM t1 WHERE c > -0.0",
            "SELECT id, f, ts FROM t1 WHERE f > DATE '1970-01-02' AND ts < TIMESTAMP '2024-01-10 00:00:00'",
            "SELECT id, g FROM t1 WHERE g > 0 AND g < 100.5",
            "SELECT id FROM t1 WHERE a IN (1, 2, 3, -5) AND d IS NOT NULL",
            "SELECT id, e FROM t1 WHERE e",
            "SELECT id, e FROM t1 WHERE NOT e",
            "SELECT * FROM t2 WHERE k > 0",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_multi_conjunct_and_or_filters() {
        assert_all_native_byte_eq(&[
            "SELECT id, a, b FROM t1 WHERE a > 0 AND b < 5000 AND c > 0.0 AND e",
            "SELECT id, d FROM t1 WHERE a > 40 OR d = 'gamma'",
            "SELECT id FROM t1 WHERE (a > 0 OR b IS NULL) AND (d LIKE 'a%' OR h LIKE '%01%') AND NOT e",
            "SELECT id, h FROM t1 WHERE h > 'a fairly long view string 0150' AND id % 2 = 0 AND a < 30",
            "SELECT id FROM t1 WHERE b % 7 = 1 AND a + b > 100 AND upper(d) = 'ALPHA'",
            "SELECT id FROM t1 WHERE ((a > 0 AND b > 0) OR (a < 0 AND b < 100)) AND c IS NOT NULL",
            "SELECT id FROM t1 WHERE a BETWEEN -10 AND 10 AND d IN ('alpha', 'beta', '') AND f IS NOT NULL",
            "SELECT id, a FROM t1 WHERE CASE WHEN a > 0 THEN b > 100 ELSE c < 0.0 END AND g IS NOT NULL",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_fallible_conjuncts() {
        assert_all_native_byte_eq(&[
            // DataFusion only divides the rows with `x <> 0` (90% of the rows
            // are 0); the pipeline only evaluates the division on them too.
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
    async fn errors_match_datafusion() {
        for sql in [
            "SELECT id FROM fallible WHERE 10 / x > 1",
            "SELECT id FROM fallible WHERE CAST(s AS INT) > 5",
        ] {
            let err = run_native(sql, test_session_config(), all_tables)
                .await
                .err()
                .unwrap_or_else(|| panic!("{sql} succeeded"));
            assert!(
                err.to_string().contains("Divide by zero")
                    || err.to_string().contains("Cannot cast"),
                "{sql}: {err}"
            );
            // The pipeline fails too.
            let ctx = SessionContext::new();
            register_all_tables(&ctx).unwrap();
            let plan = ctx
                .sql(sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            let root =
                Arc::new(PipelineRootExec::try_new(lower(&plan).unwrap(), plan).unwrap());
            let result = datafusion_physical_plan::collect(root, ctx.task_ctx()).await;
            assert!(result.is_err(), "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_string_view_and_dictionary_filters() {
        assert_all_native_byte_eq(&[
            "SELECT * FROM vd WHERE d32 = 'k3'",
            "SELECT i, d8, sv FROM vd WHERE sv LIKE '%7%' AND d8 <> 'x2'",
            "SELECT i, d32 FROM vd WHERE d32 IN ('k1', 'k2', 'k39') OR d8 IS NULL",
            "SELECT i, sv FROM vd WHERE sv > 'short4' AND i % 3 = 0",
            "SELECT * FROM vd WHERE d8 = 'x5' AND d32 IS NOT NULL",
            "SELECT * FROM vd WHERE sv IS NULL",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_filters_below_other_operators() {
        assert_all_native_byte_eq(&[
            "SELECT COUNT(*) FROM t1 WHERE a > 0",
            "SELECT d, COUNT(*), SUM(a) FROM t1 WHERE a > 0 AND e GROUP BY d",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k WHERE t1.b > 10 AND t2.w < 100",
            "SELECT t1.id, t2.k FROM t1 LEFT JOIN t2 ON t1.a = t2.k WHERE t1.id % 10 = 3",
            "SELECT id, a FROM t1 WHERE a > 0 ORDER BY id",
            "SELECT id, h FROM t1 WHERE h LIKE '%1%' ORDER BY h DESC, id LIMIT 20",
            "SELECT * FROM (SELECT * FROM t1 ORDER BY id LIMIT 100) WHERE a > 0",
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2 WHERE k < 0",
            "SELECT id FROM t1 WHERE id < 5 LIMIT 100",
            "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0 LIMIT 7",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1 WHERE a > 0",
            "SELECT d, MAX(h) FROM t1 WHERE id % 3 = 0 GROUP BY d HAVING COUNT(*) > 100",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_filters_on_any_number_of_threads() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
                "SELECT id FROM t1 WHERE id % 1000 = 7",
                "SELECT id, a FROM t1 WHERE a > 0 ORDER BY id",
                "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0 LIMIT 7",
                "SELECT id, x FROM fallible WHERE x <> 0 AND 10 / x > 1",
            ] {
                let run =
                    assert_native_byte_eq_with_config(sql, config.clone(), all_tables)
                        .await;
                assert!(run.operators.filters > 0, "{sql}");
            }
        }
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
