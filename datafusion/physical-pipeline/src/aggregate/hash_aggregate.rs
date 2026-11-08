// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalHashAggregate`]: the pipeline breaker that replaces DataFusion's
//! `AggregateExec`s, see the [module documentation](super).

use std::fmt;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, AsArray, BooleanArray, RecordBatch, RecordBatchOptions, new_empty_array,
};
use arrow::compute::{
    SortColumn, concat, filter_record_batch, lexsort_to_indices, take_record_batch,
};
use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, GroupClusteringMode, PhysicalGroupBy,
};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use super::radix_ht::RadixHashAggregate;
use super::ungrouped::UngroupedAggregate;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
};

/// A hash aggregation: a pipeline breaker that consumes the rows of its input
/// in parallel and produces one row per group (DuckDB's
/// `PhysicalHashAggregate`).
///
/// It computes exactly what DataFusion computes with a single
/// `AggregateExec` (`Single` / `SinglePartitioned`) or with the two-stage
/// `AggregateExec(Partial) -> RepartitionExec(Hash) / CoalescePartitionsExec
/// -> AggregateExec(Final / FinalPartitioned)` chain, using the expressions,
/// accumulators and schema of these nodes, without shuffling the partial
/// states through channels. See the [module documentation](super) for how
/// the work is split between the threads.
#[derive(Debug)]
pub struct PhysicalHashAggregate {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    spec: Arc<AggregateSpec>,
}

/// The aggregation that a [`PhysicalHashAggregate`] computes.
#[derive(Debug)]
pub(crate) struct AggregateSpec {
    /// The schema of the input rows.
    pub(crate) input_schema: SchemaRef,
    /// The `GROUP BY` of the stage that consumes the input rows, with its
    /// grouping sets.
    pub(crate) group_by: PhysicalGroupBy,
    /// The schema of the group keys: the group columns, followed by
    /// `__grouping_id` for grouping sets.
    pub(crate) group_schema: SchemaRef,
    /// The aggregates of the stage that consumes the input rows, which
    /// update the thread-local states.
    pub(crate) update: Vec<AggregateUpdate>,
    /// The aggregates of the final stage, which merge the partial states
    /// and produce the final values.
    pub(crate) merge: Vec<Arc<AggregateFunctionExpr>>,
    /// The output schema: the group columns followed by the aggregates.
    pub(crate) schema: SchemaRef,
    /// The order in which the rows of every thread must be fed to the
    /// aggregates, if an aggregate requires its input in an order.
    pub(crate) input_order: Option<LexOrdering>,
    /// Whether the aggregation has a single stage (`Single`,
    /// `SinglePartitioned`), whose accumulators update and evaluate.
    pub(crate) single_stage: bool,
}

impl AggregateSpec {
    /// Whether there is no `GROUP BY` (and no grouping sets): the
    /// aggregation produces exactly one row.
    pub(crate) fn is_ungrouped(&self) -> bool {
        self.group_by.is_true_no_grouping()
    }
}

/// An aggregate of the stage that consumes the input rows: the accumulator
/// expression, its arguments and its `FILTER`.
#[derive(Debug)]
pub(crate) struct AggregateUpdate {
    pub(crate) expr: Arc<AggregateFunctionExpr>,
    /// The arguments passed to the accumulator: the arguments of the
    /// aggregate, followed by its `ORDER BY` expressions (like
    /// `aggregate_expressions` of DataFusion).
    pub(crate) arguments: Vec<Arc<dyn PhysicalExpr>>,
    /// `FILTER (WHERE ...)`.
    pub(crate) filter: Option<Arc<dyn PhysicalExpr>>,
}

/// The arguments of an aggregate for the rows of a batch that pass its
/// filter.
pub(crate) struct FilteredArguments {
    /// The arguments for the selected rows only.
    pub(crate) values: Vec<ArrayRef>,
    /// The filter result, one value per row of the batch; `None` without a
    /// filter.
    pub(crate) selection: Option<BooleanArray>,
}

impl AggregateUpdate {
    fn try_new(
        expr: Arc<AggregateFunctionExpr>,
        filter: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        let mut arguments = expr.expressions();
        arguments.extend(expr.order_bys().iter().map(|sort| Arc::clone(&sort.expr)));
        Self {
            expr,
            arguments,
            filter,
        }
    }

    /// Evaluate the arguments of the aggregate for the rows of `batch` that
    /// pass its filter, like the hash aggregation of DataFusion: the
    /// arguments are evaluated only for the selected rows, so that the rows
    /// that do not pass the filter cannot cause errors.
    pub(crate) fn evaluate_filtered(
        &self,
        batch: &RecordBatch,
    ) -> Result<FilteredArguments> {
        let selection = self
            .filter
            .as_ref()
            .map(|filter| {
                let mask = filter.evaluate(batch)?.into_array(batch.num_rows())?;
                Ok::<_, datafusion_common::DataFusionError>(mask.as_boolean().clone())
            })
            .transpose()?;
        let selected = selection.as_ref().map(BooleanArray::true_count);
        let values = match selected {
            None => self.evaluate_arguments(batch)?,
            Some(0) => self
                .arguments
                .iter()
                .map(|argument| {
                    Ok(new_empty_array(&argument.data_type(batch.schema_ref())?))
                })
                .collect::<Result<_>>()?,
            Some(selected) if selected == batch.num_rows() => {
                self.evaluate_arguments(batch)?
            }
            Some(_) => {
                let mask = selection.as_ref().expect("a selection");
                self.evaluate_arguments(&filter_record_batch(batch, mask)?)?
            }
        };
        Ok(FilteredArguments { values, selection })
    }

    /// Evaluate the arguments of the aggregate for all rows of `batch`.
    pub(crate) fn evaluate_arguments(
        &self,
        batch: &RecordBatch,
    ) -> Result<Vec<ArrayRef>> {
        self.arguments
            .iter()
            .map(|argument| argument.evaluate(batch)?.into_array(batch.num_rows()))
            .collect()
    }
}

/// The input rows of a thread, buffered so that they can be aggregated in the
/// order [`AggregateSpec::input_order`] once the thread saw all of them.
#[derive(Debug, Default)]
pub(crate) struct InputBuffer {
    batches: Vec<RecordBatch>,
}

impl InputBuffer {
    pub(crate) fn push(&mut self, batch: RecordBatch) {
        if batch.num_rows() > 0 {
            self.batches.push(batch);
        }
    }

    /// Take the buffered rows (with schema `schema`), sorted by `order`, in
    /// batches of at most [`DEFAULT_VECTOR_SIZE`] rows.
    pub(crate) fn take_sorted(
        &mut self,
        schema: &SchemaRef,
        order: &LexOrdering,
    ) -> Result<Vec<RecordBatch>> {
        let batches = std::mem::take(&mut self.batches);
        if batches.is_empty() {
            return Ok(vec![]);
        }
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum();
        let columns = (0..schema.fields().len())
            .map(|i| {
                let arrays = batches
                    .iter()
                    .map(|batch| batch.column(i).as_ref())
                    .collect::<Vec<_>>();
                Ok(concat(&arrays)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
        let batch =
            RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)?;
        let sort_columns = order
            .iter()
            .map(|sort| {
                Ok(SortColumn {
                    values: sort.expr.evaluate(&batch)?.into_array(num_rows)?,
                    options: Some(sort.options),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(&sort_columns, None)?;
        let sorted = take_record_batch(&batch, &indices)?;
        Ok((0..num_rows)
            .step_by(DEFAULT_VECTOR_SIZE)
            .map(|offset| {
                sorted.slice(offset, DEFAULT_VECTOR_SIZE.min(num_rows - offset))
            })
            .collect())
    }
}

impl PhysicalHashAggregate {
    /// Lower a DataFusion aggregation to a [`PhysicalHashAggregate`]:
    ///
    /// * an `AggregateExec` in `Single` or `SinglePartitioned` mode;
    /// * an `AggregateExec` in `Final` or `FinalPartitioned` mode over the
    ///   `AggregateExec` in `Partial` mode that computes its partial states,
    ///   possibly through the `RepartitionExec` or `CoalescePartitionsExec`
    ///   (without a fetch) that distributes them: the whole chain becomes a
    ///   single operator.
    ///
    /// `lower_input` lowers the input of the stage that consumes the input
    /// rows. Returns `Ok(None)` for any other node, and for the aggregations
    /// whose result the native operator cannot reproduce exactly, so that
    /// DataFusion executes them:
    ///
    /// * aggregations of input that is clustered by the group keys
    ///   (`GroupClusteringMode` other than `None`): they keep the order of
    ///   their input, which the plan above them may rely on;
    /// * aggregations of input without partitions, which DataFusion
    ///   aggregates differently from an empty input (e.g. it produces no
    ///   grand total for a `ROLLUP`, and fails to coalesce the partial
    ///   states of an ungrouped aggregation).
    ///
    /// The limits that DataFusion pushes into an aggregation (`lim=[n]`,
    /// for a `DISTINCT ... LIMIT` or a top-k `ORDER BY ... LIMIT`) are hints
    /// that let it stop early or keep fewer groups; the limit or sort above
    /// the aggregation applies them, so the native operator computes all
    /// groups instead.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(top) = plan.downcast_ref::<AggregateExec>() else {
            return Ok(None);
        };
        let (raw, last) = match top.mode() {
            AggregateMode::Single | AggregateMode::SinglePartitioned => (top, None),
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                match partial_aggregate(top.input()) {
                    Some(partial) => (partial, Some(top)),
                    None => return Ok(None),
                }
            }
            AggregateMode::Partial | AggregateMode::PartialReduce => return Ok(None),
        };
        if [Some(raw), last].into_iter().flatten().any(|aggregate| {
            aggregate.group_clustering_mode() != &GroupClusteringMode::None
        }) {
            return Ok(None);
        }
        if raw.input().output_partitioning().partition_count() == 0 {
            return Ok(None);
        }
        if let Some(last) = last
            && !is_final_of(last, raw)
        {
            return Ok(None);
        }
        let input = lower_input(raw.input())?;
        let aggregate = match last {
            Some(last) => Self::try_new_two_stage(raw, last, input)?,
            None => Self::try_new_single_stage(raw, input)?,
        };
        Ok(Some(aggregate))
    }

    /// The aggregation of `aggregate`, an `AggregateExec` in `Single` or
    /// `SinglePartitioned` mode, over `input`, the lowered input of
    /// `aggregate`.
    pub fn try_new_single_stage(
        aggregate: &AggregateExec,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        if !matches!(
            aggregate.mode(),
            AggregateMode::Single | AggregateMode::SinglePartitioned
        ) {
            return internal_err!(
                "PhysicalHashAggregate::try_new_single_stage requires a Single or \
                 SinglePartitioned AggregateExec, got {:?}",
                aggregate.mode()
            );
        }
        let merge = without_input_order(aggregate.aggr_expr())?;
        Self::try_new(aggregate, merge, aggregate.schema(), true, input)
    }

    /// The aggregation of the chain `partial -> ... -> last`, where
    /// `partial` is an `AggregateExec` in `Partial` mode and `last` the
    /// `AggregateExec` in `Final` or `FinalPartitioned` mode that merges its
    /// states, over `input`, the lowered input of `partial`.
    pub fn try_new_two_stage(
        partial: &AggregateExec,
        last: &AggregateExec,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        if partial.mode() != &AggregateMode::Partial
            || !matches!(
                last.mode(),
                AggregateMode::Final | AggregateMode::FinalPartitioned
            )
        {
            return internal_err!(
                "PhysicalHashAggregate::try_new_two_stage requires a Partial and a \
                 Final or FinalPartitioned AggregateExec, got {:?} and {:?}",
                partial.mode(),
                last.mode()
            );
        }
        if !is_final_of(last, partial) {
            return internal_err!(
                "The {:?} AggregateExec does not merge the states of the Partial \
                 AggregateExec",
                last.mode()
            );
        }
        let merge = without_input_order(last.aggr_expr())?;
        Self::try_new(partial, merge, last.schema(), false, input)
    }

    /// `raw` is the `AggregateExec` that consumes the input rows, `merge` the
    /// aggregates that merge the partial states and evaluate, `schema` the
    /// output schema.
    fn try_new(
        raw: &AggregateExec,
        merge: Vec<Arc<AggregateFunctionExpr>>,
        schema: SchemaRef,
        single_stage: bool,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        let input_schema = raw.input().schema();
        if input.schema() != input_schema {
            return internal_err!(
                "The lowered input {} of the AggregateExec has schema {:?}, expected {:?}",
                input.name(),
                input.schema(),
                input_schema
            );
        }
        let group_by = raw.group_expr().clone();
        let group_schema = group_by.group_schema(&input_schema)?;
        let update_exprs = without_input_order(raw.aggr_expr())?;
        if update_exprs.len() != merge.len()
            || raw.filter_expr().len() != update_exprs.len()
        {
            return internal_err!(
                "PhysicalHashAggregate requires one filter and one final aggregate per \
                 aggregate"
            );
        }
        let input_order = hard_input_order(raw)?;
        let update = update_exprs
            .into_iter()
            .zip(raw.filter_expr())
            .map(|(expr, filter)| AggregateUpdate::try_new(expr, filter.clone()))
            .collect();
        let num_fields = group_schema.fields().len() + merge.len();
        if schema.fields().len() != num_fields {
            return internal_err!(
                "The schema of the AggregateExec has {} fields, expected {num_fields}",
                schema.fields().len()
            );
        }
        Ok(Self {
            children: vec![input],
            spec: Arc::new(AggregateSpec {
                input_schema,
                group_by,
                group_schema,
                update,
                merge,
                schema,
                input_order,
                single_stage,
            }),
        })
    }

    /// The input of the aggregation.
    pub fn input(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The `GROUP BY` of the aggregation (of its stage that consumes the
    /// input rows), with its grouping sets.
    pub fn group_by(&self) -> &PhysicalGroupBy {
        &self.spec.group_by
    }

    /// The aggregates that consume the input rows.
    pub fn aggregates(&self) -> impl Iterator<Item = &Arc<AggregateFunctionExpr>> {
        self.spec.update.iter().map(|update| &update.expr)
    }

    /// Whether the aggregation has no `GROUP BY` (and no grouping sets), and
    /// thus produces exactly one row.
    pub fn is_ungrouped(&self) -> bool {
        self.spec.is_ungrouped()
    }
}

/// The `AggregateExec` in `Partial` mode below `plan`, the input of an
/// aggregation in `Final` or `FinalPartitioned` mode, looking through the
/// nodes that distribute its states without changing them.
fn partial_aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<&AggregateExec> {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return (aggregate.mode() == &AggregateMode::Partial).then_some(aggregate);
    }
    if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
        && !repartition.preserve_order()
    {
        return partial_aggregate(repartition.input());
    }
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
        && coalesce.fetch().is_none()
    {
        return partial_aggregate(coalesce.input());
    }
    None
}

/// Whether `last` merges the states produced by `partial`: it groups by the
/// group columns of `partial` and merges its aggregates.
fn is_final_of(last: &AggregateExec, partial: &AggregateExec) -> bool {
    last.group_expr() == &partial.group_expr().as_final()
        && last.aggr_expr().len() == partial.aggr_expr().len()
        && last
            .aggr_expr()
            .iter()
            .zip(partial.aggr_expr())
            .all(
                |(last, partial)| match (last.state_fields(), partial.state_fields()) {
                    (Ok(last), Ok(partial)) => {
                        last.len() == partial.len()
                            && last
                                .iter()
                                .zip(&partial)
                                .all(|(l, p)| l.data_type() == p.data_type())
                    }
                    _ => false,
                },
            )
}

/// The aggregates `exprs`, with the aggregates whose accumulators may rely
/// on the order of their input (`with_beneficial_ordering(true)`, set by
/// DataFusion when the input of an aggregation is sorted as an ordered
/// aggregate such as `ARRAY_AGG(x ORDER BY y)` or `FIRST_VALUE` prefers)
/// replaced by aggregates that do not: the threads of a pipeline see the
/// rows in no particular order, so these accumulators order the values
/// themselves.
fn without_input_order(
    exprs: &[Arc<AggregateFunctionExpr>],
) -> Result<Vec<Arc<AggregateFunctionExpr>>> {
    exprs
        .iter()
        .map(|expr| {
            if expr.order_sensitivity().is_beneficial()
                && let Some(expr) = Arc::clone(expr).with_beneficial_ordering(false)?
            {
                return Ok(Arc::new(expr));
            }
            Ok(Arc::clone(expr))
        })
        .collect()
}

/// The order in which the aggregates of `raw`, the `AggregateExec` that
/// consumes the input rows, require their input, if one of them has a hard
/// ordering requirement (e.g. `NTH_VALUE(x, 2 ORDER BY y)`): its
/// accumulator relies on the order of the rows it is updated with, which
/// DataFusion establishes by sorting every input partition by the input
/// ordering the `AggregateExec` requires. The threads of the native operator
/// sort their rows by it instead.
///
/// Returns `None` if no aggregate has a hard requirement, or if the
/// requirement is trivially satisfied (e.g. it only orders by group
/// columns).
fn hard_input_order(raw: &AggregateExec) -> Result<Option<LexOrdering>> {
    if !raw
        .aggr_expr()
        .iter()
        .any(|expr| expr.order_sensitivity().hard_requires())
    {
        return Ok(None);
    }
    let Some(requirement) = raw.required_input_ordering().swap_remove(0) else {
        return Ok(None);
    };
    let sort_exprs = requirement.into_single().into_iter().map(|requirement| {
        PhysicalSortExpr::new(requirement.expr, requirement.options.unwrap_or_default())
    });
    Ok(LexOrdering::new(sort_exprs))
}

impl fmt::Display for AggregateSpec {
    /// Like `AggregateExec`: `gby=[...], aggr=[...]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let alias = |(expr, alias): &(Arc<dyn PhysicalExpr>, String)| {
            let expr = expr.to_string();
            if &expr == alias {
                expr
            } else {
                format!("{expr} as {alias}")
            }
        };
        let group_by = if self.group_by.is_single() {
            self.group_by.expr().iter().map(alias).collect::<Vec<_>>()
        } else {
            self.group_by
                .groups()
                .iter()
                .map(|group| {
                    let terms = group
                        .iter()
                        .enumerate()
                        .map(|(i, &is_null)| {
                            if is_null {
                                alias(&self.group_by.null_expr()[i])
                            } else {
                                alias(&self.group_by.expr()[i])
                            }
                        })
                        .collect::<Vec<_>>();
                    format!("({})", terms.join(", "))
                })
                .collect()
        };
        let aggregates = self
            .update
            .iter()
            .map(|update| update.expr.name())
            .collect::<Vec<_>>();
        write!(
            f,
            "gby=[{}], aggr=[{}]",
            group_by.join(", "),
            aggregates.join(", ")
        )
    }
}

impl PhysicalOperator for PhysicalHashAggregate {
    fn name(&self) -> &str {
        "PhysicalHashAggregate"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.spec.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!("PhysicalHashAggregate: {}", self.spec)
    }

    // ---------------------------------------------------------------------
    // Sink role: thread-local aggregation of the input rows
    // ---------------------------------------------------------------------

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        if self.spec.is_ungrouped() {
            UngroupedAggregate::global_sink_state()
        } else {
            RadixHashAggregate::global_sink_state()
        }
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        if self.spec.is_ungrouped() {
            UngroupedAggregate::local_sink_state(&self.spec)
        } else {
            RadixHashAggregate::local_sink_state(&self.spec)
        }
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        _gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let batch = input.compact()?;
        if self.spec.is_ungrouped() {
            UngroupedAggregate::sink(&self.spec, batch, lstate)?;
        } else {
            RadixHashAggregate::sink(&self.spec, batch, lstate)?;
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        if self.spec.is_ungrouped() {
            UngroupedAggregate::combine(&self.spec, gstate, lstate)
        } else {
            RadixHashAggregate::combine(&self.spec, gstate, lstate)
        }
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        if self.spec.is_ungrouped() {
            UngroupedAggregate::finalize(&self.spec, gstate)
        } else {
            RadixHashAggregate::finalize(gstate)
        }
    }

    // ---------------------------------------------------------------------
    // Source role: the groups, finalized in parallel
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let Some(sink_state) = ctx.global_sink_state(self) else {
            return internal_err!(
                "PhysicalHashAggregate is a source before it was a sink"
            );
        };
        if self.spec.is_ungrouped() {
            UngroupedAggregate::global_source_state(sink_state.as_ref())
        } else {
            RadixHashAggregate::global_source_state(sink_state)
        }
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        if self.spec.is_ungrouped() {
            UngroupedAggregate::local_source_state()
        } else {
            RadixHashAggregate::local_source_state()
        }
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        if self.spec.is_ungrouped() {
            UngroupedAggregate::get_data(gstate)
        } else {
            RadixHashAggregate::get_data(&self.spec, gstate, lstate)
        }
    }
}
