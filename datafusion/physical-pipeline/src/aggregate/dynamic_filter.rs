// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The dynamic filters of a [`PhysicalHashAggregate`](super::PhysicalHashAggregate):
//! predicates of its input rows that it tightens while it aggregates them,
//! with which the nodes below it that read them (e.g. scans of Parquet
//! files) skip the files, row groups and rows that cannot change its result.
//!
//! # DataFusion's filters
//!
//! DataFusion's partial `AggregateExec` of an aggregation without `GROUP BY`
//! whose aggregates are all `MIN` or `MAX` of columns produces a dynamic
//! filter (see `ExecutionPlan::dynamic_expressions_produced`), which its
//! `FilterPushdown` rule pushes into the scans below it. Its partitions
//! share the *bounds* of the aggregates, the least value of every `MIN` and
//! the greatest value of every `MAX` so far, and after every batch they set
//! the filter to the rows that may change them: `a < min(a) OR b > max(b) OR
//! ...` (`true` while an aggregate has no value yet). An aggregation lowered
//! from such an `AggregateExec` updates the same filter the same way, with the
//! bounds of the accumulators of all of its threads after every chunk (see
//! [`AggregateFilter::update_bounds`]), and marks it complete once it
//! aggregated its input.
//!
//! # Published filters
//!
//! The native aggregation also publishes filters of its own, for the
//! aggregations whose state can prune their input but for which DataFusion
//! produces no filter:
//!
//! * the single stage aggregations without `GROUP BY` of `MIN` and `MAX`
//!   aggregates (DataFusion plans a single stage, without a filter, for an
//!   input of a single partition): their bounds, as above;
//! * the top-k aggregations (see [`TopK`]), which drop the rows whose value is
//!   worse than their *cutoff* (see [`TopK::candidates`]): `key <= cutoff OR
//!   key IS NULL` for a top-k aggregation by its group key (`>=` for a
//!   descending one), `x <= cutoff` for a top-k aggregation by a `MIN(x)`
//!   (`x >= cutoff` for a `MAX(x)`), with the best cutoff of all threads (see
//!   [`SharedCutoff`]).
//!
//! It creates the filter when it is lowered, and pushes it into its input
//! like DataFusion's `FilterPushdown` rule pushes the dynamic filters of the
//! nodes that produce them (see [`AggregateFilter::try_publish`]): through
//! the nodes that let the filters of their parents through (projections,
//! filters, repartitionings, coalesces, unions, joins on the side of the
//! column, ...), into the nodes that accept them (scans of Parquet files).
//! The filter is published only if a node of the input then reads it.
//!
//! # Which values are bounded
//!
//! A filter compares the values of a column with a value of the aggregation,
//! and the scans prune with the statistics of the column, so the native
//! aggregation only sets the filters of DataFusion's built-in `MIN` and `MAX`
//! (identified by the types that implement them: DataFusion goes by their
//! names, so that a user-defined function named `max` gets a filter too), of
//! columns whose values the comparisons and the statistics order like the
//! aggregates do (see [`has_exact_bounds`]). Not floating point numbers: the
//! statistics of Parquet files leave their `NaN`s out, so that a scan would
//! skip a row group whose `NaN` is the `MAX` of the aggregation (a `NaN` is
//! greater than any other value, and a `-NaN` less: Arrow orders floats
//! totally). The filters of other aggregations stay `true`, which prunes
//! nothing.
//!
//! As DataFusion's, the filters are only updated in sessions that push the
//! dynamic filters of aggregations down
//! (`datafusion.optimizer.enable_aggregate_dynamic_filter_pushdown`).
//!
//! A filter is shared by all executions of the aggregation, like the dynamic
//! filters of DataFusion's plans: an execution after another one starts from
//! the filter that the first one left, so the plan must be reset before it is
//! executed again (see `PipelineRootExec::reset_state`, which lowers the
//! reset plan again).

use std::any::TypeId;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicUsize};

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{Accumulator, Operator};
use datafusion_functions_aggregate::min_max;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, is_null, lit,
};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::execution_plan::{
    plan_contains_expression_id, replace_children_if_necessary,
};
use datafusion_physical_plan::filter_pushdown::{
    ChildFilterPushdownResult, ChildPushdownResult, FilterPushdownPhase, PushedDown,
};
use parking_lot::Mutex;

use super::hash_aggregate::{TopK, TopKOrder, implementation_type};
use crate::operator::ExecutionContext;

/// A dynamic filter that an aggregation updates while it aggregates its
/// input, see the [module documentation](self).
#[derive(Debug)]
pub(crate) struct AggregateFilter {
    /// The filters, which nodes of the input read: those that DataFusion's
    /// `AggregateExec` produces, or the one that the aggregation published.
    /// All of them are set to the same predicate.
    filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
    kind: FilterKind,
}

/// The rows that an [`AggregateFilter`] selects.
#[derive(Debug)]
pub(crate) enum FilterKind {
    /// The rows that may change the bounds of the `MIN` and `MAX` aggregates
    /// of an aggregation without `GROUP BY`, all of its aggregates: `a <
    /// min(a) OR b > max(b) OR ...` (see [`AggregateFilter::update_bounds`]).
    Bounds(Vec<BoundedAggregate>),
    /// The rows that may make their groups among the top groups of a top-k
    /// aggregation: those whose `value` (a column, the group key or the
    /// argument of the aggregate) is not worse than the best cutoff, see
    /// [`AggregateFilter::update_cutoff`].
    Cutoff {
        top_k: TopK,
        value: Arc<dyn PhysicalExpr>,
    },
}

/// A `MIN` or `MAX` aggregate whose bound a [`FilterKind::Bounds`] filter
/// compares its argument with.
#[derive(Debug)]
pub(crate) struct BoundedAggregate {
    /// The index of the aggregate among the aggregates of the aggregation.
    aggregate: usize,
    /// Whether the aggregate is a `MAX` (otherwise a `MIN`).
    max: bool,
    /// The argument of the aggregate: a column of the input rows.
    argument: Arc<dyn PhysicalExpr>,
}

impl BoundedAggregate {
    /// Whether `value` is a better bound than `bound` (`NULL`: none yet): a
    /// smaller value for a `MIN`, a greater one for a `MAX`.
    fn is_better(&self, value: &ScalarValue, bound: &ScalarValue) -> Result<bool> {
        if value.is_null() {
            return Ok(false);
        }
        if bound.is_null() {
            return Ok(true);
        }
        let expected = match self.max {
            true => Ordering::Greater,
            false => Ordering::Less,
        };
        match value.partial_cmp(bound) {
            Some(ordering) => Ok(ordering == expected),
            None => internal_err!(
                "Cannot compare the bounds {value:?} and {bound:?} of an aggregate"
            ),
        }
    }
}

impl AggregateFilter {
    /// The dynamic filters that `raw`, the `AggregateExec` that consumes the
    /// input rows of an aggregation, produces, if they are DataFusion's
    /// filters of the bounds of its `MIN` and `MAX` aggregates that the
    /// native aggregation sets (see [`bounded_aggregates`]).
    pub(crate) fn of_aggregate_exec(raw: &AggregateExec) -> Option<Self> {
        let filters = dynamic_filters(raw)?;
        if filters.is_empty() {
            return None;
        }
        Some(Self {
            filters,
            kind: FilterKind::Bounds(bounded_aggregates(raw)?),
        })
    }

    /// The filter that the aggregation whose stage that consumes the input
    /// rows is `raw` (and whose final stage is `last`, if it has two), a
    /// top-k aggregation if `top_k` is set, publishes into its input, if it
    /// publishes one (see the [module documentation](self)), and the input
    /// of `raw` with the nodes that read it. `None` if DataFusion's
    /// `AggregateExec`s produce dynamic filters, or if no node of the input
    /// would read the filter.
    ///
    /// The filter is pushed into the input with the default options of the
    /// session (the lowering does not know the session; the session decides
    /// whether the filter is updated, see [`filters_enabled`]).
    pub(crate) fn try_publish(
        raw: &AggregateExec,
        last: Option<&AggregateExec>,
        top_k: Option<TopK>,
    ) -> Result<Option<(Self, Arc<dyn ExecutionPlan>)>> {
        // DataFusion's filters (or their absence: DataFusion keeps them only
        // if a node of the input reads them) take precedence.
        if [Some(raw), last]
            .into_iter()
            .flatten()
            .any(|stage| !stage.dynamic_expressions_produced().is_empty())
        {
            return Ok(None);
        }
        let input = raw.input();
        let schema = input.schema();
        let (kind, children) = match top_k {
            Some(top_k) => {
                // The group key, or the argument of the aggregate.
                let value = match top_k.order {
                    TopKOrder::Key => raw
                        .group_expr()
                        .expr()
                        .first()
                        .map(|(key, _)| Arc::clone(key)),
                    TopKOrder::Aggregate => raw
                        .aggr_expr()
                        .first()
                        .and_then(|aggregate| aggregate.expressions().first().cloned()),
                };
                let Some(value) = value.filter(|value| is_bounded_column(value, &schema))
                else {
                    return Ok(None);
                };
                let children = vec![Arc::clone(&value)];
                (FilterKind::Cutoff { top_k, value }, children)
            }
            // DataFusion produces the filters of the bounds of partial
            // aggregations (unless it does not push dynamic filters of
            // aggregations down, or no node of the input would read them),
            // but not of single stage ones.
            None if matches!(
                raw.mode(),
                AggregateMode::Single | AggregateMode::SinglePartitioned
            ) =>
            {
                let Some(bounded) = bounded_aggregates(raw) else {
                    return Ok(None);
                };
                let children = bounded
                    .iter()
                    .map(|bounded| Arc::clone(&bounded.argument))
                    .collect();
                (FilterKind::Bounds(bounded), children)
            }
            None => return Ok(None),
        };
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(children, lit(true)));
        let config = ConfigOptions::default();
        let Some(input) = push_down(input, vec![Arc::clone(&filter) as _], &config)?
        else {
            return Ok(None);
        };
        let Some(id) = filter.expression_id() else {
            return internal_err!("A dynamic filter has no expression id");
        };
        if !plan_contains_expression_id(&input, id)? {
            return Ok(None);
        }
        Ok(Some((
            Self {
                filters: vec![filter],
                kind,
            },
            input,
        )))
    }

    /// The filters that the aggregation updates.
    #[cfg(test)]
    pub(crate) fn filters(&self) -> &[Arc<DynamicFilterPhysicalExpr>] {
        &self.filters
    }

    /// The bounds of all threads of an execution of the aggregation, if the
    /// filter is a [`FilterKind::Bounds`] filter: `NULL` (no bound yet).
    pub(crate) fn shared_bounds(&self) -> Option<SharedBounds> {
        match &self.kind {
            FilterKind::Bounds(bounded) => Some(SharedBounds {
                bounds: Mutex::new(vec![ScalarValue::Null; bounded.len()]),
            }),
            FilterKind::Cutoff { .. } => None,
        }
    }

    /// Update `shared`, the bounds of all threads, with the values of
    /// `accumulators`, the accumulators of the aggregates of a thread that
    /// aggregated a chunk, and set the filter to the rows that may change
    /// them if they changed, like the partitions of DataFusion's
    /// `AggregateExec` do after every batch: `a < min(a) OR b > max(b) OR
    /// ...`, or `true` while an aggregate has no bound (a row that does not
    /// change the other bounds may give it its first value).
    ///
    /// `known` are the bounds of all threads that the thread saw last: only
    /// if the value of one of its accumulators is better, the thread locks
    /// the bounds of all threads (and updates `known`), so that the threads
    /// rarely wait for each other once the bounds are tight.
    pub(crate) fn update_bounds(
        &self,
        accumulators: &mut [Box<dyn Accumulator>],
        known: &mut [ScalarValue],
        shared: &SharedBounds,
    ) -> Result<()> {
        let FilterKind::Bounds(bounded) = &self.kind else {
            return Ok(());
        };
        let mut values = Vec::with_capacity(bounded.len());
        let mut improves = false;
        for (aggregate, known) in bounded.iter().zip(known.iter()) {
            let Some(accumulator) = accumulators.get_mut(aggregate.aggregate) else {
                return internal_err!(
                    "No accumulator for the bounded aggregate {}",
                    aggregate.aggregate
                );
            };
            let value = accumulator.evaluate()?;
            improves |= aggregate.is_better(&value, known)?;
            values.push(value);
        }
        if !improves {
            return Ok(());
        }
        let mut bounds = shared.bounds.lock();
        let mut changed = false;
        for ((aggregate, value), bound) in
            bounded.iter().zip(values).zip(bounds.iter_mut())
        {
            if aggregate.is_better(&value, bound)? {
                *bound = value;
                changed = true;
            }
        }
        known.clone_from_slice(&bounds);
        if changed {
            // Updated under the lock, so that the filter is set to the
            // tightest bounds last.
            let predicate = bounds_predicate(bounded, &bounds);
            for filter in &self.filters {
                filter.update(Arc::clone(&predicate))?;
            }
        }
        Ok(())
    }

    /// Set the filter of a top-k aggregation to the rows that may make their
    /// groups among its top groups given the cutoff `cutoff` (see
    /// [`TopK::candidates`]): those whose value is not worse than the cutoff,
    /// and, for a top-k aggregation by its group key, those of the `NULL` key,
    /// whose group is always produced.
    pub(crate) fn update_cutoff(&self, cutoff: &ArrayRef) -> Result<()> {
        let FilterKind::Cutoff { top_k, value } = &self.kind else {
            return Ok(());
        };
        if cutoff.len() != 1 || cutoff.is_null(0) {
            return internal_err!(
                "A cutoff of a top-k aggregation must be a single value"
            );
        }
        let op = match top_k.descending {
            true => Operator::GtEq,
            false => Operator::LtEq,
        };
        let cutoff = ScalarValue::try_from_array(cutoff.as_ref(), 0)?;
        let mut predicate: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(Arc::clone(value), op, lit(cutoff)));
        if top_k.order == TopKOrder::Key {
            predicate = Arc::new(BinaryExpr::new(
                predicate,
                Operator::Or,
                is_null(Arc::clone(value))?,
            ));
        }
        for filter in &self.filters {
            filter.update(Arc::clone(&predicate))?;
        }
        Ok(())
    }

    /// Mark the filters complete: the aggregation will not change them any
    /// more.
    pub(crate) fn mark_complete(&self) {
        for filter in &self.filters {
            filter.mark_complete();
        }
    }
}

/// `a < min(a) OR b > max(b) OR ...` for the `bounds` of the aggregates
/// `bounded`, or `true` if an aggregate has no bound yet, like DataFusion's
/// `AggregateExec` builds it.
fn bounds_predicate(
    bounded: &[BoundedAggregate],
    bounds: &[ScalarValue],
) -> Arc<dyn PhysicalExpr> {
    if bounds.iter().any(ScalarValue::is_null) {
        return lit(true);
    }
    bounded
        .iter()
        .zip(bounds)
        .map(|(aggregate, bound)| {
            let op = match aggregate.max {
                true => Operator::Gt,
                false => Operator::Lt,
            };
            Arc::new(BinaryExpr::new(
                Arc::clone(&aggregate.argument),
                op,
                lit(bound.clone()),
            )) as Arc<dyn PhysicalExpr>
        })
        .reduce(|left, right| Arc::new(BinaryExpr::new(left, Operator::Or, right)))
        .unwrap_or_else(|| lit(true))
}

/// The bounds of the `MIN` and `MAX` aggregates of all threads of an
/// execution of an aggregation (see [`AggregateFilter::update_bounds`]):
/// DataFusion's `shared_bound`s of its `AggregateExec`.
#[derive(Debug)]
pub(crate) struct SharedBounds {
    /// The bound of every bounded aggregate, `NULL` if it has none yet.
    bounds: Mutex<Vec<ScalarValue>>,
}

impl SharedBounds {
    /// No bounds, as a thread knows them before it aggregated a row.
    pub(crate) fn unknown(&self) -> Vec<ScalarValue> {
        vec![ScalarValue::Null; self.bounds.lock().len()]
    }
}

/// The best cutoff (see [`TopK::cutoff`]) that the threads of an execution of
/// a top-k aggregation found so far: every thread drops the rows that it
/// rules out (see [`TopK::candidates`]), and it sets the dynamic filter of
/// the aggregation, if it has one.
#[derive(Debug, Default)]
pub(crate) struct SharedCutoff {
    cutoff: Mutex<Option<ArrayRef>>,
    /// Incremented whenever `cutoff` gets better, so that the threads notice
    /// a better cutoff with an atomic load.
    generation: AtomicUsize,
}

impl SharedCutoff {
    /// Make `cutoff`, a cutoff of the top-k aggregation `top_k` that a thread
    /// found, the best cutoff if it is better, and then set `filter` to it.
    pub(crate) fn offer(
        &self,
        top_k: &TopK,
        cutoff: &ArrayRef,
        filter: Option<&AggregateFilter>,
    ) -> Result<()> {
        let mut best = self.cutoff.lock();
        if let Some(best) = best.as_ref()
            && !top_k.is_better(cutoff.as_ref(), best.as_ref())?
        {
            return Ok(());
        }
        // Set under the lock, so that the filter is set to the best cutoff
        // last.
        if let Some(filter) = filter {
            filter.update_cutoff(cutoff)?;
        }
        *best = Some(Arc::clone(cutoff));
        self.generation.fetch_add(1, atomic::Ordering::Release);
        Ok(())
    }

    /// The best cutoff, if it got better since the thread that saw the
    /// generation `seen` last looked (`seen` is then updated).
    pub(crate) fn changed_since(&self, seen: &mut usize) -> Option<ArrayRef> {
        let generation = self.generation.load(atomic::Ordering::Acquire);
        if generation == *seen {
            return None;
        }
        *seen = generation;
        self.cutoff.lock().clone()
    }
}

/// Whether the session of `ctx` updates the dynamic filters of aggregations,
/// as DataFusion's `AggregateExec` does only if
/// `datafusion.optimizer.enable_aggregate_dynamic_filter_pushdown` is set.
pub(crate) fn filters_enabled(ctx: &ExecutionContext) -> bool {
    ctx.task_ctx
        .session_config()
        .options()
        .optimizer
        .enable_aggregate_dynamic_filter_pushdown
}

/// The dynamic filters that `plan` produces (see
/// `ExecutionPlan::dynamic_expressions_produced`), `None` if one of them is
/// not a [`DynamicFilterPhysicalExpr`].
fn dynamic_filters(
    plan: &dyn ExecutionPlan,
) -> Option<Vec<Arc<DynamicFilterPhysicalExpr>>> {
    plan.dynamic_expressions_produced()
        .into_iter()
        .map(|filter| {
            let filter: Arc<dyn std::any::Any + Send + Sync> = filter;
            filter.downcast::<DynamicFilterPhysicalExpr>().ok()
        })
        .collect()
}

/// The aggregates of `raw`, an `AggregateExec` that consumes input rows, whose
/// bounds a filter compares their arguments with (see
/// [`FilterKind::Bounds`]), if it has no `GROUP BY` and all of its aggregates
/// are DataFusion's built-in `MIN` or `MAX` (identified by the types that
/// implement them) of a column whose values have exact bounds (see
/// [`is_bounded_column`]), as DataFusion's `AggregateExec` produces filters
/// for its aggregates named `min` and `max` of columns.
fn bounded_aggregates(raw: &AggregateExec) -> Option<Vec<BoundedAggregate>> {
    if !raw.group_expr().is_true_no_grouping() || raw.aggr_expr().is_empty() {
        return None;
    }
    let schema = raw.input().schema();
    raw.aggr_expr()
        .iter()
        .enumerate()
        .map(|(index, aggregate)| {
            let implementation = implementation_type(aggregate.fun());
            let max = if implementation == TypeId::of::<min_max::Max>() {
                true
            } else if implementation == TypeId::of::<min_max::Min>() {
                false
            } else {
                return None;
            };
            let arguments = aggregate.expressions();
            let [argument] = arguments.as_slice() else {
                return None;
            };
            is_bounded_column(argument, &schema).then(|| BoundedAggregate {
                aggregate: index,
                max,
                argument: Arc::clone(argument),
            })
        })
        .collect()
}

/// Whether `expr` is a column of `schema` whose values have exact bounds
/// (see [`has_exact_bounds`]): a filter that compares it with a value cannot
/// fail, wherever a scan evaluates it, and prunes like it compares.
fn is_bounded_column(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> bool {
    expr.is::<Column>()
        && expr
            .data_type(schema)
            .is_ok_and(|data_type| has_exact_bounds(&data_type))
}

/// Whether the values of `data_type` are ordered by Arrow's comparisons (with
/// which a filter compares them, and the aggregates order them) like the
/// statistics of Parquet files bound them: integers, decimals, dates, times,
/// timestamps, durations, booleans, strings and binaries. Not floating point
/// numbers, whose `NaN`s (greater, or for `-NaN` less, than any other value
/// in Arrow's total order) the statistics leave out, nor dictionaries,
/// intervals and nested values.
pub(crate) fn has_exact_bounds(data_type: &DataType) -> bool {
    data_type.is_integer()
        || (data_type.is_temporal() && !matches!(data_type, DataType::Interval(_)))
        || matches!(
            data_type,
            DataType::Decimal32(_, _)
                | DataType::Decimal64(_, _)
                | DataType::Decimal128(_, _)
                | DataType::Decimal256(_, _)
                | DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
}

/// Push `filters`, predicates of the output rows of `plan`, down into `plan`,
/// like DataFusion's `FilterPushdown` rule pushes the filters of the parents
/// of a node in its post-optimization phase, in which it pushes the dynamic
/// filters of the nodes that produce them: every node tells which of the
/// filters can be pushed into which of its children (rewritten for the
/// columns of the child, see `ExecutionPlan::gather_filters_for_pushdown`),
/// and the leaves take those that they can use (see
/// `ExecutionPlan::handle_child_pushdown_result`, e.g. a scan of Parquet
/// files adds them to its predicate). Returns `plan` with the nodes that
/// took filters, `None` if no node did.
///
/// Unlike the rule, only `filters` are pushed down, not the filters of the
/// nodes themselves, which are already where the rule pushed them, and the
/// nodes above the leaves only get their new children: they drop the filters
/// that their children do not take (DataFusion's nodes handle the filters of
/// their parents in the post-optimization phase only by pushing them into
/// their children), which is correct for a dynamic filter, which only prunes
/// rows that are not needed.
fn push_down(
    plan: &Arc<dyn ExecutionPlan>,
    filters: Vec<Arc<dyn PhysicalExpr>>,
    config: &ConfigOptions,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let children = plan.children();
    if children.is_empty() {
        let result = ChildPushdownResult {
            parent_filters: filters
                .into_iter()
                .map(|filter| ChildFilterPushdownResult {
                    filter,
                    child_results: vec![],
                })
                .collect(),
            self_filters: vec![],
        };
        return Ok(plan
            .handle_child_pushdown_result(FilterPushdownPhase::Post, result, config)?
            .updated_node);
    }
    let description =
        plan.gather_filters_for_pushdown(FilterPushdownPhase::Post, filters, config)?;
    let parent_filters = description.parent_filters();
    if parent_filters.len() != children.len() {
        return Ok(None);
    }
    let mut pushed = false;
    let mut new_children = Vec::with_capacity(children.len());
    for (child, filters) in children.into_iter().zip(parent_filters) {
        let filters = filters
            .into_iter()
            .filter(|filter| matches!(filter.discriminant, PushedDown::Yes))
            .map(|filter| filter.predicate)
            .collect::<Vec<_>>();
        let new_child = match filters.is_empty() {
            true => None,
            false => push_down(child, filters, config)?,
        };
        pushed |= new_child.is_some();
        new_children.push(new_child.unwrap_or_else(|| Arc::clone(child)));
    }
    if !pushed {
        return Ok(None);
    }
    Ok(Some(replace_children_if_necessary(
        Arc::clone(plan),
        new_children,
    )?))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use arrow::array::{
        ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array,
        Int64Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{Field, Schema};
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_datasource::file_scan_config::FileScanConfig;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::metrics::MetricValue;

    use super::super::PhysicalHashAggregate;
    use super::super::test_support::{
        assert_aggregate_parity_with_setup, assert_lowered_results_with_setup,
        lookalike_max, lower_with_tiny_tables, without_limited_aggregations,
    };
    use super::*;
    use crate::bridge::{PipelineRootExec, bridged_plans};
    use crate::explain::format_operator_tree;
    use crate::operator::PhysicalOperator;
    use crate::sources::ParquetScanSource;
    use crate::sources::test_harness::write_parquet;
    use crate::test_utils::{datafusion_context, pipeline_context, test_session_config};

    /// The number of files of the Parquet table `m`, and of rows of every file.
    const M_FILES: i64 = 16;
    const M_ROWS: i64 = 4096;

    /// Write the Parquet table `m` into `dir`: [`M_FILES`] files of
    /// [`M_ROWS`] rows, in two row groups each (of chunks large enough that
    /// the filters of the queries do not cache them, see `CachingOperator`),
    /// whose values are bounded by the values of every file before them: file
    /// `i` has values `x` from `i` to `1000 - i` (both of them, in its first
    /// two rows, followed by `i + 1` and `999 - i`), and `s`,
    /// `d`, `dec` and `f` of the same order (`f` with a `NaN` in its last
    /// file). Once a thread aggregated the `MIN` or the `MAX` of a file, a
    /// filter of its bounds prunes all files after it. `id` is unique and
    /// ascending, `e` is true for two of three rows, `n` is always `NULL` and
    /// `g` has 7 values.
    fn write_table_m(dir: &Path) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("d", DataType::Date32, true),
            Field::new("dec", DataType::Decimal128(10, 2), true),
            Field::new("f", DataType::Float64, true),
            Field::new("e", DataType::Boolean, true),
            Field::new("n", DataType::Int64, true),
            Field::new("g", DataType::Int32, true),
        ]));
        let properties = ::parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(M_ROWS as usize / 2))
            .build();
        for file in 0..M_FILES {
            let x = (0..M_ROWS)
                .map(|row| match row {
                    0 => file,
                    1 => 1000 - file,
                    2 => file + 1,
                    3 => 999 - file,
                    _ => file + 1 + (row * 37) % (999 - 2 * file),
                })
                .collect::<Vec<_>>();
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..M_ROWS).map(|row| file * M_ROWS + row),
                )),
                Arc::new(Int64Array::from(x.clone())),
                Arc::new(StringArray::from_iter_values(
                    x.iter().map(|x| format!("s{x:04}")),
                )),
                Arc::new(Date32Array::from_iter_values(
                    x.iter().map(|&x| 19_000 + x as i32),
                )),
                Arc::new(
                    Decimal128Array::from_iter_values(
                        x.iter().map(|&x| i128::from(x) * 100 + 5),
                    )
                    .with_precision_and_scale(10, 2)?,
                ),
                Arc::new(Float64Array::from_iter_values(x.iter().enumerate().map(
                    |(row, &x)| match (file, row) {
                        (15, 4) => f64::NAN,
                        _ => x as f64,
                    },
                ))),
                Arc::new(BooleanArray::from_iter(
                    (0..M_ROWS).map(|row| Some(row % 3 != 2)),
                )),
                Arc::new(Int64Array::from(vec![None; M_ROWS as usize])),
                Arc::new(Int32Array::from_iter_values(
                    (0..M_ROWS).map(|row| (row % 7) as i32),
                )),
            ];
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
            write_parquet(
                &dir.join(format!("m/{file:02}.parquet")),
                Arc::clone(&schema),
                &[batch],
                Some(properties.clone()),
            )?;
        }
        Ok(())
    }

    /// The number of files of the Parquet table `t`, and of rows of every file.
    const T_FILES: i64 = 8;
    const T_ROWS: i64 = 40_000;

    /// Write the Parquet table `t` into `dir`: [`T_FILES`] files of
    /// [`T_ROWS`] rows (more than the thread-local tables of a top-k
    /// aggregation hold before they are flushed, which sets their cutoff),
    /// with keys `k` that are unique and ascending, `hi = -k`, and groups `g`
    /// of 1,000 values in every file.
    fn write_table_t(dir: &Path) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("hi", DataType::Int64, false),
            Field::new("g", DataType::Int64, false),
        ]));
        for file in 0..T_FILES {
            let k = (0..T_ROWS).map(|row| file * T_ROWS + row);
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter_values(k.clone())),
                Arc::new(Int64Array::from_iter_values(k.map(|k| -k))),
                Arc::new(Int64Array::from_iter_values(
                    (0..T_ROWS).map(|row| row % 1_000),
                )),
            ];
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
            write_parquet(
                &dir.join(format!("t/{file:02}.parquet")),
                Arc::clone(&schema),
                &[batch],
                None,
            )?;
        }
        Ok(())
    }

    /// Register the Parquet tables `m` and `t` of `dir` (see
    /// [`write_table_m`] and [`write_table_t`]).
    async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<()> {
        for table in ["m", "t"] {
            let path = dir.join(table);
            let path = path.to_str().expect("a UTF-8 path");
            ctx.register_parquet(table, path, ParquetReadOptions::default())
                .await?;
        }
        Ok(())
    }

    /// A directory with the Parquet tables `m` and `t`.
    fn write_tables() -> Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        write_table_m(dir.path())?;
        write_table_t(dir.path())?;
        Ok(dir)
    }

    /// The first scan of files of `plan`.
    fn find_file_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan
            .downcast_ref::<DataSourceExec>()
            .is_some_and(|scan| scan.data_source().is::<FileScanConfig>())
        {
            return Some(Arc::clone(plan));
        }
        plan.children().into_iter().find_map(find_file_scan)
    }

    /// The number of items pruned by the scan of files `scan` (native or not:
    /// the native scans record the metrics of the scans of DataFusion that
    /// they replace), by its pruning metric `name`.
    fn pruned(scan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        match scan.metrics().and_then(|metrics| metrics.sum_by_name(name)) {
            Some(MetricValue::PruningMetrics {
                pruning_metrics, ..
            }) => pruning_metrics.pruned(),
            Some(other) => other.as_usize(),
            None => 0,
        }
    }

    /// The files and row groups that the scan of files `scan` pruned with
    /// statistics.
    fn pruned_by_statistics(scan: &Arc<dyn ExecutionPlan>) -> usize {
        pruned(scan, "files_ranges_pruned_statistics")
            + pruned(scan, "row_groups_pruned_statistics")
    }

    /// The dynamic filters that the `AggregateExec`s of `plan` produce.
    fn aggregate_exec_filters(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Vec<Arc<DynamicFilterPhysicalExpr>> {
        let mut filters = vec![];
        plan.apply(|node| {
            if node.is::<AggregateExec>() {
                filters.extend(dynamic_filters(node.as_ref()).expect("dynamic filters"));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("a traversal");
        filters
    }

    /// The dynamic filters that the native aggregations of `root` update.
    fn native_filters(
        root: &Arc<dyn PhysicalOperator>,
    ) -> Vec<Arc<DynamicFilterPhysicalExpr>> {
        let mut filters = vec![];
        if let Some(aggregate) = root.downcast_ref::<PhysicalHashAggregate>()
            && let Some(filter) = &aggregate.spec().dynamic_filter
        {
            filters.extend(filter.filters().iter().cloned());
        }
        for child in root.children() {
            filters.extend(native_filters(child));
        }
        filters
    }

    /// The outcome of an execution of a query by the pipeline engine.
    struct Execution {
        /// The dynamic filters of the `AggregateExec`s of DataFusion's plan.
        datafusion_filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
        /// The dynamic filters that the native aggregations update.
        native_filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
        /// The (first) scan of files of DataFusion's plan.
        scan: Arc<dyn ExecutionPlan>,
        /// The rows that the native scans produced.
        scanned_rows: usize,
        /// The lowered operator tree.
        tree: String,
    }

    /// Execute `root`, lowered from `plan`, in `ctx`.
    async fn execute_lowered(
        ctx: &SessionContext,
        plan: Arc<dyn ExecutionPlan>,
        root: Arc<dyn PhysicalOperator>,
    ) -> Result<Execution> {
        let tree = format_operator_tree(&root);
        assert!(bridged_plans(&root).is_empty(), "{tree}");
        let execution = Execution {
            datafusion_filters: aggregate_exec_filters(&plan),
            native_filters: native_filters(&root),
            scan: find_file_scan(&plan).expect("a scan of files"),
            scanned_rows: 0,
            tree,
        };
        let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
        collect(Arc::clone(&exec) as _, ctx.task_ctx()).await?;
        let scanned_rows = exec
            .execution_metrics()
            .expect("the metrics of the execution")
            .iter()
            .filter(|(operator, _)| operator.is::<ParquetScanSource>())
            .map(|(_, metrics)| metrics.output_rows())
            .sum();
        Ok(Execution {
            scanned_rows,
            ..execution
        })
    }

    /// Execute `sql` with the `PipelineQueryPlanner`, in a session of
    /// configuration `config` with the tables of `dir`. The plan is fully
    /// native.
    async fn execute(sql: &str, config: SessionConfig, dir: &Path) -> Result<Execution> {
        let ctx = pipeline_context(config);
        register_tables(&ctx, dir).await?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let Some(root) = plan.downcast_ref::<PipelineRootExec>() else {
            panic!("The pipeline engine did not plan {sql}");
        };
        execute_lowered(&ctx, Arc::clone(root.input()), Arc::clone(root.operator())).await
    }

    /// `config` with `pushdown_filters`.
    fn with_pushdown_filters(
        config: SessionConfig,
        pushdown_filters: bool,
    ) -> SessionConfig {
        let mut config = config;
        config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
        config
    }

    /// The queries of `MIN` and `MAX` aggregates of `m` for which DataFusion
    /// produces dynamic filters, and the predicates that the filters are set
    /// to.
    const BOUNDED_QUERIES: [(&str, &str); 4] = [
        (
            "SELECT MIN(x), MAX(x) FROM m WHERE e",
            "x@0 < 0 OR x@0 > 1000",
        ),
        ("SELECT MAX(x) FROM m WHERE g > 0", "x@0 > 1000"),
        // (Without the rows of `x` 0 and 1000, the first two of the first
        // file.)
        (
            "SELECT MIN(s), MAX(d), MIN(dec) FROM m WHERE id > 1",
            "s@0 < s0001 OR d@1 > 2024-10-03 OR dec@2 < 1.05",
        ),
        // With a filter of an aggregate: the rows that do not change the
        // bounds are pruned, whether they pass it or not.
        (
            "SELECT MIN(x) FILTER (WHERE g = 3), MAX(x) FROM m WHERE e",
            "x@0 < 1 OR x@0 > 1000",
        ),
    ];

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregations_update_the_dynamic_filters_of_datafusion() -> Result<()> {
        // DataFusion's partial aggregations of `MIN` and `MAX` without `GROUP
        // BY` produce dynamic filters, which the scans of `m` read. The native
        // aggregations set them to the bounds of their aggregates of all
        // threads, as DataFusion does, so the scans prune the files whose
        // values are within the bounds: every file but the first one of every
        // thread (of 4).
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        for pushdown_filters in [false, true] {
            let config = with_pushdown_filters(test_session_config(), pushdown_filters);
            for (sql, predicate) in BOUNDED_QUERIES {
                let context = format!("{sql} (pushdown_filters: {pushdown_filters})");
                let execution = execute(sql, config.clone(), dir.path()).await?;
                let [filter] = execution.datafusion_filters.as_slice() else {
                    panic!("DataFusion produces one filter: {context}");
                };
                // The native aggregation updates the filter of DataFusion.
                let [native] = execution.native_filters.as_slice() else {
                    panic!("No filter is updated: {context}\n{}", execution.tree);
                };
                assert!(Arc::ptr_eq(native, filter), "{context}");
                assert!(
                    execution.tree.contains("dynamic_filter=true"),
                    "{context}\n{}",
                    execution.tree
                );
                assert!(filter.is_complete(), "{context}");
                assert_eq!(filter.current()?.to_string(), predicate, "{context}");
                // At most one file per thread is read.
                let pruned = pruned_by_statistics(&execution.scan);
                assert!(
                    pruned >= (M_FILES - 4) as usize,
                    "{context}: {pruned} pruned"
                );
                assert!(
                    execution.scanned_rows <= 4 * M_ROWS as usize,
                    "{context}: {} rows scanned",
                    execution.scanned_rows
                );
                assert_aggregate_parity_with_setup(sql, config.clone(), &setup).await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_stage_aggregations_publish_filters_of_their_bounds() -> Result<()> {
        // On a single partition, DataFusion plans a single stage aggregation,
        // which produces no dynamic filter. The native aggregation publishes
        // one into the scan, with which the scan prunes every file after the
        // first one.
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        for pushdown_filters in [false, true] {
            let config = with_pushdown_filters(
                test_session_config().with_target_partitions(1),
                pushdown_filters,
            );
            for (sql, predicate) in BOUNDED_QUERIES {
                let context = format!("{sql} (pushdown_filters: {pushdown_filters})");
                let execution = execute(sql, config.clone(), dir.path()).await?;
                assert!(execution.datafusion_filters.is_empty(), "{context}");
                let [filter] = execution.native_filters.as_slice() else {
                    panic!("No filter is published: {context}\n{}", execution.tree);
                };
                assert!(filter.is_complete(), "{context}");
                assert_eq!(filter.current()?.to_string(), predicate, "{context}");
                let pruned = pruned_by_statistics(&execution.scan);
                assert!(
                    pruned >= (M_FILES - 1) as usize,
                    "{context}: {pruned} pruned"
                );
                // At most the first file is read (the scan prunes the row
                // groups of a file that it did not read yet too).
                assert!(
                    execution.scanned_rows <= M_ROWS as usize,
                    "{context}: {} rows scanned",
                    execution.scanned_rows
                );
                assert_aggregate_parity_with_setup(sql, config.clone(), &setup).await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_publish_filters_of_their_cutoffs() -> Result<()> {
        // The top-k aggregations by the group key or by a `MIN` / `MAX` of
        // `t` publish a dynamic filter into the scan, which the best cutoff of
        // their threads sets: once a thread flushed its table (after 32,768
        // rows of its first file), the scan prunes the files whose values are
        // all worse, every file after the first one of every thread (of 4).
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        let queries = [
            (
                "SELECT DISTINCT k FROM t ORDER BY k LIMIT 3",
                "k@0 <= 2 OR k@0 IS NULL",
            ),
            (
                "SELECT k FROM t GROUP BY k ORDER BY k NULLS FIRST LIMIT 10",
                "k@0 <= 9 OR k@0 IS NULL",
            ),
            (
                "SELECT g, MAX(hi) FROM t GROUP BY g ORDER BY MAX(hi) DESC LIMIT 3",
                "hi@0 >= -2",
            ),
            (
                "SELECT g, MIN(k) AS m FROM t GROUP BY g ORDER BY m LIMIT 5",
                "k@0 <= 4",
            ),
        ];
        for pushdown_filters in [false, true] {
            let config = with_pushdown_filters(test_session_config(), pushdown_filters);
            let reference = without_limited_aggregations(config.clone());
            for (sql, predicate) in queries {
                let context = format!("{sql} (pushdown_filters: {pushdown_filters})");
                let execution = execute(sql, config.clone(), dir.path()).await?;
                let [filter] = execution.native_filters.as_slice() else {
                    panic!("No filter is published: {context}\n{}", execution.tree);
                };
                assert!(filter.is_complete(), "{context}");
                assert_eq!(filter.current()?.to_string(), predicate, "{context}");
                let pruned = pruned_by_statistics(&execution.scan);
                assert!(
                    pruned >= (T_FILES - 4) as usize,
                    "{context}: {pruned} pruned"
                );
                assert!(
                    execution.scanned_rows <= 4 * T_ROWS as usize,
                    "{context}: {} rows scanned",
                    execution.scanned_rows
                );
                assert_lowered_results_with_setup(
                    sql,
                    config.clone(),
                    reference.clone(),
                    &setup,
                )
                .await;
            }
        }
        // Descending, the best keys come last, so the cutoffs of the first
        // files prune hardly anything.
        let sql = "SELECT DISTINCT k FROM t ORDER BY k DESC LIMIT 3";
        let config = test_session_config();
        let execution = execute(sql, config.clone(), dir.path()).await?;
        assert_eq!(execution.native_filters.len(), 1, "{sql}");
        let reference = without_limited_aggregations(config.clone());
        assert_lowered_results_with_setup(sql, config, reference, &setup).await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_of_small_tables_prune_with_their_cutoffs() -> Result<()> {
        // With tiny thread-local tables, which are flushed after a few rows,
        // the cutoffs of the top-k aggregations of `m` prune all files but
        // the first one of every thread on any number of threads.
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        let queries = [
            (
                "SELECT DISTINCT id FROM m ORDER BY id LIMIT 4",
                "id@0 <= 3 OR id@0 IS NULL",
            ),
            (
                "SELECT g, MIN(id) FROM m GROUP BY g ORDER BY MIN(id) LIMIT 2",
                "id@0 <= 1",
            ),
        ];
        for threads in [1, 4] {
            let config = test_session_config().with_target_partitions(threads);
            let ctx = datafusion_context(config.clone());
            register_tables(&ctx, dir.path()).await?;
            for (sql, predicate) in queries {
                let context = format!("{sql} on {threads} threads");
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                let root = lower_with_tiny_tables(&plan)?;
                let execution = execute_lowered(&ctx, plan, root).await?;
                let [filter] = execution.native_filters.as_slice() else {
                    panic!("No filter is published: {context}\n{}", execution.tree);
                };
                assert_eq!(filter.current()?.to_string(), predicate, "{context}");
                assert!(
                    execution.scanned_rows <= threads * M_ROWS as usize,
                    "{context}: {} rows scanned",
                    execution.scanned_rows
                );
                let reference = without_limited_aggregations(config.clone());
                assert_lowered_results_with_setup(sql, config.clone(), reference, &setup)
                    .await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_other_aggregations_stay_true() -> Result<()> {
        // The native aggregations do not set the filters of `MIN` and `MAX`
        // of floating point values, whose `NaN`s the statistics of Parquet
        // files leave out (a `NaN` of the last file of `m` is the `MAX` of
        // `f`, which the bounds of the files before would prune), nor of
        // functions that are only named `min` or `max`: DataFusion's filters
        // stay `true`, and the scans read every row. The results are those of
        // DataFusion without its filters. A filter of an aggregate without
        // values (`n` is always `NULL`) is set to `true`.
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        let config = test_session_config();
        let mut reference = config.clone();
        reference
            .options_mut()
            .optimizer
            .enable_aggregate_dynamic_filter_pushdown = false;
        let all_rows = (M_FILES * M_ROWS) as usize;
        for sql in [
            "SELECT MIN(f), MAX(f) FROM m WHERE e",
            "SELECT MAX(f), MAX(x) FROM m WHERE g > 0",
        ] {
            let execution = execute(sql, config.clone(), dir.path()).await?;
            assert_eq!(execution.datafusion_filters.len(), 1, "{sql}");
            assert!(execution.native_filters.is_empty(), "{sql}");
            assert_eq!(
                execution.datafusion_filters[0].current()?.to_string(),
                "true",
                "{sql}"
            );
            assert_eq!(execution.scanned_rows, all_rows, "{sql}");
            assert_lowered_results_with_setup(
                sql,
                config.clone(),
                reference.clone(),
                &setup,
            )
            .await;
        }
        // A `MAX` that is not DataFusion's.
        let lookalike = async |ctx: &SessionContext| {
            register_tables(ctx, dir.path()).await?;
            ctx.register_udaf(lookalike_max());
            Ok(())
        };
        let sql = "SELECT MAX(x) FROM m WHERE e";
        let ctx = pipeline_context(config.clone());
        lookalike(&ctx).await?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        let execution =
            execute_lowered(&ctx, Arc::clone(root.input()), Arc::clone(root.operator()))
                .await?;
        assert_eq!(execution.datafusion_filters.len(), 1, "{sql}");
        assert!(execution.native_filters.is_empty(), "{sql}");
        assert_eq!(execution.scanned_rows, all_rows, "{sql}");
        assert_lowered_results_with_setup(sql, config.clone(), reference, &lookalike)
            .await;
        // An aggregate without values.
        let sql = "SELECT MIN(x), MAX(n) FROM m WHERE e";
        let execution = execute(sql, config.clone(), dir.path()).await?;
        let [filter] = execution.native_filters.as_slice() else {
            panic!("No filter is updated: {sql}\n{}", execution.tree);
        };
        assert_eq!(filter.current()?.to_string(), "true", "{sql}");
        assert_eq!(execution.scanned_rows, all_rows, "{sql}");
        assert_aggregate_parity_with_setup(sql, config, &setup).await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_are_not_updated_if_the_session_does_not_push_them_down() -> Result<()>
    {
        // Without dynamic filters of aggregations, DataFusion produces none,
        // and the filters that the native aggregations publish (which the
        // lowering, unaware of the session, pushes into the scans) stay
        // `true`: the scans read every row.
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        for threads in [1, 4] {
            let mut config = test_session_config().with_target_partitions(threads);
            config
                .options_mut()
                .optimizer
                .enable_aggregate_dynamic_filter_pushdown = false;
            for (sql, rows) in [
                ("SELECT MIN(x), MAX(x) FROM m WHERE e", M_FILES * M_ROWS),
                (
                    "SELECT DISTINCT k FROM t ORDER BY k LIMIT 3",
                    T_FILES * T_ROWS,
                ),
            ] {
                let context = format!("{sql} on {threads} threads");
                let execution = execute(sql, config.clone(), dir.path()).await?;
                assert!(execution.datafusion_filters.is_empty(), "{context}");
                for filter in &execution.native_filters {
                    assert_eq!(filter.current()?.to_string(), "true", "{context}");
                }
                assert_eq!(execution.scanned_rows, rows as usize, "{context}");
                let reference = without_limited_aggregations(config.clone());
                assert_lowered_results_with_setup(sql, config.clone(), reference, &setup)
                    .await;
            }
        }
        Ok(())
    }

    /// A benchmark of aggregations whose dynamic filters prune Parquet scans:
    /// `MIN` / `MAX` without `GROUP BY` and top-k aggregations of a table `t`
    /// of `DYNAMIC_FILTER_ROWS` rows (10,000,000) in `DYNAMIC_FILTER_FILES`
    /// files (16) of row groups of 131,072 rows (as the ClickBench and TPC-H
    /// benchmark writes them), with ascending ids `id` and timestamps `ts`,
    /// dates `d` of 31 days, values `v` and advertising engines `a` (zero for
    /// 95% of the rows) at random, and 1,000 groups `g`. Prints the median
    /// time of `DYNAMIC_FILTER_ROUNDS` executions (5, without planning) on
    /// each engine on `DYNAMIC_FILTER_THREADS` threads (all), and the rows
    /// that the native scans produced. Run it with `cargo test --profile
    /// release-nonlto -p datafusion-physical-pipeline --lib
    /// aggregate_dynamic_filters_benchmark -- --ignored --nocapture`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "a benchmark"]
    async fn aggregate_dynamic_filters_benchmark() -> Result<()> {
        use std::sync::Arc;
        use std::time::Duration;

        use arrow::array::{
            ArrayRef, Date32Array, Int16Array, Int32Array, Int64Array, RecordBatch,
        };
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion::prelude::{ParquetReadOptions, SessionContext};
        use datafusion_common::DataFusionError;
        use datafusion_common::instant::Instant;
        use datafusion_physical_plan::collect;

        use crate::bridge::PipelineRootExec;
        use crate::sources::ParquetScanSource;
        use crate::sources::test_harness::write_parquet;
        use crate::test_utils::{
            canonical_batch, datafusion_context, pipeline_context, test_session_config,
        };

        let env = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        };
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let rows = env("DYNAMIC_FILTER_ROWS", 10_000_000);
        let files = env("DYNAMIC_FILTER_FILES", 16);
        let rounds = env("DYNAMIC_FILTER_ROUNDS", 5);
        let threads = env("DYNAMIC_FILTER_THREADS", threads);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("ts", DataType::Int64, false),
            Field::new("d", DataType::Date32, false),
            Field::new("a", DataType::Int16, false),
            Field::new("g", DataType::Int32, false),
            Field::new("v", DataType::Int64, false),
        ]));
        // A SplitMix64 hash of `i`, as random values.
        let random = |i: i64, stream: u64| {
            let mut z = (i as u64 ^ (stream << 56)).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let dir = tempfile::tempdir()?;
        let properties = ::parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(128 * 1024))
            .build();
        let per_file = rows.div_ceil(files);
        for file in 0..files {
            let start = (file * per_file) as i64;
            let end = ((file + 1) * per_file).min(rows) as i64;
            let mut batches = vec![];
            for first in (start..end).step_by(8192) {
                let ids = first..(first + 8192).min(end);
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from_iter_values(ids.clone())),
                    Arc::new(Int64Array::from_iter_values(
                        ids.clone()
                            .map(|i| 1_372_636_800 + i * 3 + (random(i, 0) % 3) as i64),
                    )),
                    Arc::new(Date32Array::from_iter_values(
                        ids.clone().map(|i| 15_887 + (random(i, 1) % 31) as i32),
                    )),
                    Arc::new(Int16Array::from_iter_values(ids.clone().map(
                        |i| match random(i, 2) % 20 {
                            0 => 1 + (random(i, 3) % 30) as i16,
                            _ => 0,
                        },
                    ))),
                    Arc::new(Int32Array::from_iter_values(
                        ids.clone().map(|i| (i % 1000) as i32),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        ids.map(|i| (random(i, 4) >> 1) as i64),
                    )),
                ];
                batches.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
            }
            write_parquet(
                &dir.path().join(format!("t/part-{file:02}.parquet")),
                Arc::clone(&schema),
                &batches,
                Some(properties.clone()),
            )?;
        }
        let path = dir.path().join("t");
        let path = path.to_str().expect("a UTF-8 path");
        let config = test_session_config().with_target_partitions(threads);
        let engines = [
            ("datafusion", datafusion_context(config.clone())),
            ("pipeline", pipeline_context(config)),
        ];
        for (_, ctx) in &engines {
            ctx.register_parquet("t", path, ParquetReadOptions::default())
                .await?;
        }
        let queries = [
            "SELECT MIN(d), MAX(d) FROM t WHERE a <> 0",
            "SELECT MAX(v) FROM t WHERE a <> 0",
            "SELECT MIN(ts), MAX(ts) FROM t WHERE a <> 0",
            "SELECT DISTINCT id FROM t ORDER BY id LIMIT 10",
            "SELECT g, MIN(ts) FROM t GROUP BY g ORDER BY MIN(ts) LIMIT 10",
            "SELECT g, MAX(ts) FROM t GROUP BY g ORDER BY MAX(ts) DESC LIMIT 10",
        ];
        // Plan `sql` (anew, as an execution changes the dynamic filters of a
        // plan) and execute it: the batches, the time of the execution, and
        // the rows that the native scans produced.
        let execute = async |ctx: &SessionContext, sql: &str| {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let start = Instant::now();
            let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
            let elapsed = start.elapsed();
            let scanned = plan
                .downcast_ref::<PipelineRootExec>()
                .and_then(PipelineRootExec::execution_metrics)
                .map(|metrics| {
                    metrics
                        .iter()
                        .filter(|(operator, _)| operator.is::<ParquetScanSource>())
                        .map(|(_, metrics)| metrics.output_rows())
                        .sum::<usize>()
                });
            Ok::<_, DataFusionError>((batches, elapsed, scanned, plan.schema()))
        };
        let median = |times: &mut Vec<Duration>| {
            times.sort_unstable();
            times[times.len() / 2].as_secs_f64() * 1e3
        };
        println!(
            "\n{rows} rows in {files} files, {threads} threads, median of {rounds} \
             executions in ms\n"
        );
        println!("| Query | DataFusion | Pipeline | Rows scanned by the pipeline |");
        println!("| ----- | ---------: | -------: | ---------------------------: |");
        for sql in queries {
            let mut times = vec![vec![]; engines.len()];
            let mut outputs = vec![];
            let mut scanned = None;
            for round in 0..rounds {
                for (engine, (_, ctx)) in engines.iter().enumerate() {
                    let (batches, elapsed, rows, schema) = execute(ctx, sql).await?;
                    times[engine].push(elapsed);
                    scanned = scanned.or(rows);
                    if round == 0 {
                        outputs.push(canonical_batch(&schema, &batches, false)?);
                    }
                }
            }
            assert_eq!(outputs[0], outputs[1], "{sql}");
            let [datafusion, pipeline] = [median(&mut times[0]), median(&mut times[1])];
            println!(
                "| `{sql}` | {datafusion:.2} | {pipeline:.2} | {} |",
                scanned.unwrap_or_default()
            );
        }
        Ok(())
    }
}
