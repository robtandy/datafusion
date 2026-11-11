// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalHashAggregate`]: the pipeline breaker that replaces DataFusion's
//! `AggregateExec`s, see the [module documentation](super).

use std::any::{Any, TypeId};
use std::fmt;
use std::sync::{Arc, LazyLock};

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, Scalar, UInt32Array,
    make_comparator, new_empty_array, new_null_array,
};
use arrow::compute::kernels::cmp::{gt_eq, lt_eq};
use arrow::compute::{
    SortColumn, SortOptions, concat, filter_record_batch, interleave, lexsort_to_indices,
    sort_to_indices, take, take_record_batch,
};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::config::{
    ConfigEntry, ConfigExtension, ConfigOptions, ExtensionOptions,
};
use datafusion_common::stats::Precision;
use datafusion_common::{Result, ScalarValue, config_err, internal_err};
use datafusion_expr::{AggregateUDF, Operator};
use datafusion_functions_aggregate::{
    approx_distinct, array_agg, average, bit_and_or_xor, bool_and_or, count, first_last,
    median, min_max, sum,
};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, CastExpr, Column, Literal};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_expr_common::utils::scatter;
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, GroupClusteringMode, PhysicalGroupBy,
};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::statistics::{StatisticsArgs, StatisticsContext};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use super::radix_ht::{RadixHashAggregate, TableCapacity};
use super::ungrouped::UngroupedAggregate;
use crate::bridge::BridgeOperator;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::join::PhysicalHashJoin;
use crate::join::nested_loop::{PhysicalCrossJoin, PhysicalNestedLoopJoin};
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
};
use crate::operators::adaptive_filter::{failures, is_infallible};
use crate::operators::caching::CachingOperator;
use crate::operators::filter::PhysicalFilter;
use crate::operators::limit::PhysicalLimit;
use crate::operators::placeholder_row::PhysicalPlaceholderRow;
use crate::operators::projection::PhysicalProjection;
use crate::planner::PhysicalScalarSubquery;
use crate::sources::csv::CsvScanSource;
use crate::sources::empty::EmptyScanSource;
use crate::sources::memory::MemoryScanSource;
use crate::sources::parquet::ParquetScanSource;

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
    /// The capacity of the thread-local tables of a grouped aggregation.
    table_capacity: TableCapacity,
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
    /// The limit of a top-k aggregation: only the groups that the sort above
    /// it may keep are produced.
    pub(crate) top_k: Option<TopK>,
    /// Whether the input rows of a thread whose groups hardly repeat may
    /// bypass its thread-local table, and be aggregated by the final
    /// aggregation of their radix partitions directly (see
    /// [`rows_can_bypass_tables`]), if the session trusts the accumulators of
    /// all aggregates (see [`Self::rows_bypass_tables`]).
    pub(crate) bypass: bool,
    /// Whether all aggregates (of both stages) have the same values however
    /// the rows of a group are split into partial states (see
    /// [`splits_exactly`]).
    pub(crate) splits_exactly: bool,
    /// The number of partitions of the input of the stage that consumes the
    /// input rows, which DataFusion aggregates on their own.
    pub(crate) input_partitions: usize,
    /// Whether the aggregation of a single partition of DataFusion, or a
    /// native operator of its pipeline, may fail (see `own_parts` and
    /// `pipeline_parts`): its pipelines then run on a single thread, which
    /// evaluates the rows in order, and so fails on the first of them that
    /// fails in DataFusion (the threads of a pipeline evaluate the chunks of a
    /// partition in parallel, and any of them may fail first). (DataFusion
    /// evaluates several partitions in parallel, and fails on the first of
    /// them that fails.)
    pub(crate) evaluates_in_order: bool,
}

impl AggregateSpec {
    /// Whether the aggregation must aggregate the rows of its input in order,
    /// on a single thread, into a single state per group: if DataFusion
    /// aggregates a single partition, in order, and the values of some
    /// aggregate depend on how the rows of a group are split into partial
    /// states (see [`splits_exactly`]). (The threads of a pipeline evaluate the
    /// chunks of a partition of DataFusion in parallel. The aggregations of
    /// such aggregates of several partitions, and those whose input comes in
    /// another order than DataFusion's, are left to DataFusion, see
    /// [`PhysicalHashAggregate::try_lower`].)
    pub(crate) fn aggregates_in_order(&self) -> bool {
        self.input_partitions == 1 && !self.splits_exactly
    }

    /// Whether there is no `GROUP BY` (and no grouping sets): the
    /// aggregation produces exactly one row.
    pub(crate) fn is_ungrouped(&self) -> bool {
        self.group_by.is_true_no_grouping()
    }

    /// Whether the input rows of a thread whose groups hardly repeat bypass
    /// its thread-local table in an execution with the configuration
    /// `options`: if they may (see [`Self::bypass`]), and the aggregation has
    /// no aggregates, or the session trusts the accumulators of all of them
    /// (see [`TrustedAggregateFunctions`]). (But the single thread of an
    /// aggregation with aggregates aggregates all of its rows in its table,
    /// see `PhysicalHashAggregate::get_local_sink_state`.)
    pub(crate) fn rows_bypass_tables(&self, options: &ConfigOptions) -> bool {
        if !self.bypass {
            return false;
        }
        if self.update.is_empty() && self.merge.is_empty() {
            return true;
        }
        options
            .extensions
            .get::<TrustedAggregateFunctions>()
            .is_some_and(|trusted| {
                self.update
                    .iter()
                    .all(|update| trusted.trusts(&update.expr))
                    && self.merge.iter().all(|merge| trusted.trusts(merge))
            })
    }
}

/// The limit of a top-k aggregation by its group key, which DataFusion's
/// `TopKAggregation` rule sets on the stages of an aggregation without
/// aggregates (`lim=[n]`, DataFusion's `TopKDistinct`, e.g. `SELECT DISTINCT
/// k FROM t ORDER BY k LIMIT 10`) below a sort with a fetch of `limit` rows
/// whose only sort key is its only group key.
///
/// The sort keeps the first `limit` groups by their key, so the aggregation
/// only produces the groups that can be among them, exactly like DataFusion's
/// `GroupedTopKAggregateStream` of a single partition: the `limit` groups of
/// the best non-null keys, and the group of the `NULL` key, which DataFusion
/// keeps whatever the order of the `NULL`s. Distinct groups have distinct
/// keys, so these groups do not depend on the order of the rows. The keys are
/// compared as the sort compares them, as DataFusion's top-k aggregation does
/// for the types of [`orders_like_datafusion`]. So the operators between the
/// aggregation and the sort above it see exactly the groups they see in
/// DataFusion, also the operators that see all of them (e.g. a window
/// `COUNT(*) OVER ()`: DataFusion's `TopKAggregation` rule passes the limit
/// through the operators that keep the number of rows, windows included).
///
/// DataFusion's top-k aggregations by a `MIN` or `MAX` (`TopKMinMax`) are
/// left to it: the values of distinct groups may tie, and of the groups that
/// tie with the worst of its top groups, its heap keeps the one whose row came
/// last (a new group that ties with its worst group replaces it), which
/// depends on the order of the rows (and decides, e.g., the groups that a
/// projection `1000 / (a - 1)` above it is evaluated for).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TopK {
    /// The number of groups the sort keeps.
    pub(crate) limit: usize,
    /// Whether the groups with the greatest keys come first.
    pub(crate) descending: bool,
}

impl TopK {
    /// The rows of `keys`, the keys of a set of distinct groups (one per
    /// group), whose groups may be among the groups the sort keeps (see
    /// [`TopK`]): the `limit` best non-null keys, followed by the `NULL` key.
    /// Returns `None` if these are all rows. A group that is not selected
    /// (`limit` other groups beat it) is not among the groups the sort keeps.
    pub(crate) fn select(&self, keys: &dyn Array) -> Result<Option<UInt32Array>> {
        if keys.len() - keys.null_count() <= self.limit {
            return Ok(None);
        }
        // With the nulls last, the first `limit` rows are non-null.
        let options = SortOptions::new(self.descending, false);
        let mut indices = sort_to_indices(keys, Some(options), Some(self.limit))?
            .values()
            .to_vec();
        indices.extend(
            (0..keys.len())
                .filter(|&row| keys.is_null(row))
                .map(|row| row as u32),
        );
        Ok(Some(UInt32Array::from(indices)))
    }

    /// The *cutoff* of `keys`, the keys of a set of distinct groups (one per
    /// group): the `limit`-th best of them, if `limit` of them are not
    /// `NULL`. From then on, at least `limit` groups have a key at least as
    /// good as the cutoff, so a row whose key is worse cannot be of one of the
    /// groups the sort keeps (see [`Self::candidates`]).
    pub(crate) fn cutoff(&self, keys: &dyn Array) -> Result<Option<ArrayRef>> {
        let valid = keys.len() - keys.null_count();
        if self.limit == 0 || valid < self.limit {
            return Ok(None);
        }
        // With the nulls last, the first `limit` rows are non-null.
        let options = SortOptions::new(self.descending, false);
        let best = sort_to_indices(keys, Some(options), Some(self.limit))?;
        Ok(Some(take(keys, &best.slice(self.limit - 1, 1), None)?))
    }

    /// Whether `cutoff` is a better cutoff than `other` (see [`Self::cutoff`]),
    /// which rules out more rows: whether it comes before it in the order of
    /// the groups.
    pub(crate) fn is_better(
        &self,
        cutoff: &dyn Array,
        other: &dyn Array,
    ) -> Result<bool> {
        let options = SortOptions::new(self.descending, false);
        Ok(make_comparator(cutoff, other, options)?(0, 0).is_lt())
    }

    /// The rows of `keys`, the group keys of input rows, whose groups may be
    /// among the groups the sort keeps given the cutoff `cutoff` (see
    /// [`Self::cutoff`]): the rows whose key is not worse than the cutoff, and
    /// those whose key is `NULL` (whose group is always kept). Returns `None`
    /// if these are all rows.
    pub(crate) fn candidates(
        &self,
        keys: &ArrayRef,
        cutoff: &ArrayRef,
    ) -> Result<Option<BooleanArray>> {
        let cutoff = Scalar::new(Arc::clone(cutoff));
        let selected = match self.descending {
            true => gt_eq(keys, &cutoff)?,
            false => lt_eq(keys, &cutoff)?,
        };
        let selected = match selected.nulls() {
            Some(nulls) => BooleanArray::new(selected.values() | &!nulls.inner(), None),
            None => selected,
        };
        Ok((selected.true_count() < selected.len()).then_some(selected))
    }

    /// The groups of `batch`, the output of the aggregation for a set of
    /// distinct groups, that may be among the groups the sort keeps (see
    /// [`Self::select`]).
    pub(crate) fn select_groups(&self, batch: RecordBatch) -> Result<RecordBatch> {
        match self.select(batch.column(0).as_ref())? {
            Some(indices) => Ok(take_record_batch(&batch, &indices)?),
            None => Ok(batch),
        }
    }

    /// The groups of `batch`, the top groups of the aggregation, in the order
    /// in which DataFusion's top-k aggregation produces them: it drains its
    /// heap from the best group to the worst, and the group of the `NULL` key
    /// comes last. The operators between the aggregation and the sort above
    /// it see the groups in that order, e.g. a projection that fails on
    /// several of them fails on the first one, whose value its error shows.
    pub(crate) fn sort_groups(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let options = SortOptions::new(self.descending, false);
        let indices = sort_to_indices(batch.column(0), Some(options), None)?;
        Ok(take_record_batch(batch, &indices)?)
    }

    /// The groups of `left` and `right`, outputs of the aggregation for two
    /// sets of distinct groups (selected by [`Self::select_groups`]), that may
    /// be among the groups the sort keeps: [`Self::select_groups`] of all of
    /// them, without concatenating them. Only the selected groups are
    /// gathered, which DataFusion's top-k aggregation outputs as well, so
    /// that their strings fit into an array (of 32-bit offsets) whenever its
    /// do.
    pub(crate) fn merge_groups(
        &self,
        left: &RecordBatch,
        right: &RecordBatch,
    ) -> Result<RecordBatch> {
        let (left_values, right_values) = (left.column(0), right.column(0));
        let options = SortOptions::new(self.descending, false);
        // The non-null values of either side, the best first.
        let sorted = |values: &ArrayRef| -> Result<Vec<u32>> {
            let valid = values.len() - values.null_count();
            if valid == 0 {
                return Ok(vec![]);
            }
            Ok(sort_to_indices(values, Some(options), Some(valid))?
                .values()
                .to_vec())
        };
        let (left_sorted, right_sorted) = (sorted(left_values)?, sorted(right_values)?);
        let compare =
            make_comparator(left_values.as_ref(), right_values.as_ref(), options)?;
        // A merge of the sorted sides, up to `limit` rows.
        let mut rows = vec![];
        let (mut l, mut r) = (0, 0);
        while rows.len() < self.limit && (l < left_sorted.len() || r < right_sorted.len())
        {
            let from_left = r == right_sorted.len()
                || (l < left_sorted.len()
                    && compare(left_sorted[l] as usize, right_sorted[r] as usize)
                        .is_le());
            if from_left {
                rows.push((0, left_sorted[l] as usize));
                l += 1;
            } else {
                rows.push((1, right_sorted[r] as usize));
                r += 1;
            }
        }
        let nulls = |side: usize, values: &ArrayRef| {
            (0..values.len())
                .filter(|&row| values.is_null(row))
                .map(move |row| (side, row))
                .collect::<Vec<_>>()
        };
        rows.extend(
            nulls(0, left_values)
                .into_iter()
                .chain(nulls(1, right_values)),
        );
        let columns = left
            .columns()
            .iter()
            .zip(right.columns())
            .map(|(left, right)| Ok(interleave(&[left.as_ref(), right.as_ref()], &rows)?))
            .collect::<Result<Vec<_>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(rows.len()));
        Ok(RecordBatch::try_new_with_options(
            left.schema(),
            columns,
            &options,
        )?)
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
    fn new(
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
        let Some(filter) = &self.filter else {
            return Ok(FilteredArguments {
                values: self.evaluate_arguments(batch)?,
                selection: None,
            });
        };
        let mask = filter.evaluate(batch)?.into_array(batch.num_rows())?;
        let mask = as_boolean_array(&mask)?.clone();
        let values = match mask.true_count() {
            0 => self
                .arguments
                .iter()
                .map(|argument| {
                    Ok(new_empty_array(&argument.data_type(batch.schema_ref())?))
                })
                .collect::<Result<_>>()?,
            selected if selected == batch.num_rows() => self.evaluate_arguments(batch)?,
            _ => self.evaluate_arguments(&filter_record_batch(batch, &mask)?)?,
        };
        Ok(FilteredArguments {
            values,
            selection: Some(mask),
        })
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

    /// Evaluate the arguments of the aggregate that are not rebuilt (see
    /// [`rebuilt_literal`]) for the rows of `batch` that pass its filter, like
    /// [`Self::evaluate_filtered`], aligned with the rows of `batch`: the
    /// arguments of the rows that do not pass the filter are `NULL` (they are
    /// not evaluated, so they cannot cause errors). Returns the arguments and
    /// the filter result, `None` without a filter.
    pub(crate) fn evaluate_row_aligned(
        &self,
        batch: &RecordBatch,
    ) -> Result<(Vec<ArrayRef>, Option<BooleanArray>)> {
        let num_rows = batch.num_rows();
        let arguments = self
            .arguments
            .iter()
            .filter(|argument| rebuilt_literal(argument).is_none());
        let evaluate = |argument: &Arc<dyn PhysicalExpr>, batch: &RecordBatch| {
            argument.evaluate(batch)?.into_array(batch.num_rows())
        };
        let Some(filter) = &self.filter else {
            let values = arguments
                .map(|argument| evaluate(argument, batch))
                .collect::<Result<_>>()?;
            return Ok((values, None));
        };
        let mask = filter.evaluate(batch)?.into_array(num_rows)?;
        let mask = as_boolean_array(&mask)?.clone();
        let values = match mask.true_count() {
            0 => arguments
                .map(|argument| {
                    Ok(new_null_array(
                        &argument.data_type(batch.schema_ref())?,
                        num_rows,
                    ))
                })
                .collect::<Result<_>>()?,
            selected if selected == num_rows => arguments
                .map(|argument| evaluate(argument, batch))
                .collect::<Result<_>>()?,
            _ => {
                let selected = filter_record_batch(batch, &mask)?;
                arguments
                    .map(|argument| {
                        scatter(&mask, evaluate(argument, &selected)?.as_ref())
                    })
                    .collect::<Result<_>>()?
            }
        };
        Ok((values, Some(mask)))
    }
}

/// The value of `argument`, an argument of an aggregate, if it is a literal
/// of a fixed width (e.g. the `1` of `COUNT(*)`, `count(Int64(1))`), which
/// the input rows that bypass the thread-local tables do not store: it is
/// rebuilt for the rows that the aggregate aggregates (see
/// `GroupedHashTable::update_rows`). Literals of variable length (e.g.
/// strings) are evaluated like the other arguments, for the rows that pass
/// the filter of the aggregate only, so that no more of their bytes are built
/// at once than the regular path, and DataFusion, build.
pub(crate) fn rebuilt_literal(argument: &Arc<dyn PhysicalExpr>) -> Option<&ScalarValue> {
    let value = argument.downcast_ref::<Literal>()?.value();
    let data_type = value.data_type();
    (data_type.is_primitive() || matches!(data_type, DataType::Boolean | DataType::Null))
        .then_some(value)
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
    ///   states of an ungrouped aggregation);
    /// * aggregations whose groups may span several partitions of the input
    ///   of the final (or single) `AggregateExec`, which aggregates every
    ///   partition on its own (the native operator aggregates all rows
    ///   together, which is the same if every partition has its own groups:
    ///   if there is a single partition, or the partitions are partitioned
    ///   by the group keys);
    /// * aggregations that advertise another number of partitions than their
    ///   input has (after their input was replaced without recomputing their
    ///   properties): DataFusion then aggregates some partitions of their
    ///   input only, or fails;
    /// * aggregations that may fail with several parts that may fail (group
    ///   keys, arguments and filters of aggregates, and the expressions of the
    ///   native operators of their pipeline), whose first error depends on
    ///   DataFusion's batches (see `fail_in_the_order_of_the_batches`);
    /// * aggregations with an aggregate whose accumulators may get other
    ///   values from the chunks of the native operator than from DataFusion's
    ///   batches, or fail with other errors (see `updates_like_datafusion`):
    ///   user-defined aggregates, the percentile functions of percentiles of
    ///   columns, and some built-in aggregates of a single partition;
    /// * aggregations with an aggregate whose value may depend on how the rows
    ///   of a group are split into partial states (see `splits_exactly`), e.g.
    ///   a sum of floating point values, unless the native operator aggregates
    ///   the rows in DataFusion's order (see
    ///   `AggregateSpec::aggregates_in_order`): of several partitions, which
    ///   DataFusion aggregates on their own first (the native sources read
    ///   them as a single queue of chunks, so that the native operator cannot
    ///   aggregate every partition on its own), or of a single partition whose
    ///   lowered input produces its rows in another order than DataFusion's
    ///   (see `keeps_datafusion_order`), e.g. an aggregation of the distinct
    ///   values of a `SUM(DISTINCT x)`, or whose rows are sorted for ordered
    ///   aggregates (e.g. `ARRAY_AGG(x ORDER BY y)`), by DataFusion or by the
    ///   native operator, which order the rows that tie in the sort otherwise,
    ///   or of a single partition grouped by a single grouping set in two
    ///   stages, whose partial aggregation DataFusion may skip, so that a group
    ///   gets several states that depend on DataFusion's batches. (Ordered
    ///   aggregates order their values themselves: but for rows that tie in
    ///   their order, their values do not depend on the order of their input.);
    /// * aggregations with a soft limit on their groups (`lim=[n]` without
    ///   an order), from DataFusion's optimization of `SELECT DISTINCT ...
    ///   LIMIT n`: each stage stops reading its input once it has `n` groups,
    ///   so DataFusion neither evaluates the rest of its input (which may
    ///   fail, e.g. with a division by zero) nor returns its groups, and the
    ///   limit above returns groups of the rows read first, while the native
    ///   operator reads all of its input, on several threads;
    /// * top-k aggregations (`lim=[n]` with an order, of an `ORDER BY ...
    ///   LIMIT n` by the group key or a `MIN` / `MAX` aggregate): DataFusion
    ///   keeps only the top `n` groups in every stage, so the operators
    ///   between the aggregation and the sort above it (e.g. a projection,
    ///   which may fail on the other groups, or a window, which sees all of
    ///   them) only see these, and so does the native operator, if it
    ///   reproduces them exactly (see [`TopK`]). It does for the top-k
    ///   aggregations by their group key, without aggregates, whose stages
    ///   have the same limit (of at most [`MAX_TOP_K_LIMIT`] groups), of a
    ///   single key of a type that DataFusion's top-k aggregation orders like
    ///   its sorts (see [`orders_like_datafusion`]) but strings with 32-bit
    ///   offsets (see [`holds_like_datafusion`]), which cannot fail
    ///   (conservatively), and whose final stage has a single partition.
    ///   DataFusion's top-k aggregation of several partitions keeps the top
    ///   groups of every partition, which depend on how its rows are
    ///   partitioned (e.g. the number of groups that a window `COUNT(*) OVER
    ///   ()` above it counts); and the groups of its top-k aggregations by a
    ///   `MIN` or `MAX` depend on the order of its rows when their values tie
    ///   (see [`TopK`]).
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
        if matches!(group_limit(raw, last), GroupLimit::Unsupported) {
            return Ok(None);
        }
        if raw.input().output_partitioning().partition_count() == 0
            || !advertises_its_input_partitions(raw)
            || !groups_are_partition_local(last.unwrap_or(raw))
        {
            return Ok(None);
        }
        if let Some(last) = last
            && (!is_final_of(last, raw) || !advertises_its_input_partitions(last))
        {
            return Ok(None);
        }
        let single_partition = raw.input().output_partitioning().partition_count() == 1;
        let grouped = !raw.group_expr().is_true_no_grouping();
        // (Computed from DataFusion's statistics if an aggregate needs it.)
        let bound = std::cell::OnceCell::new();
        let max_rows = || *bound.get_or_init(|| max_rows(raw.input()));
        let aggregates = || {
            [Some(raw), last]
                .into_iter()
                .flatten()
                .flat_map(|aggregate| aggregate.aggr_expr())
        };
        if aggregates().any(|aggregate| {
            !updates_like_datafusion(
                aggregate,
                grouped,
                single_partition,
                raw.input(),
                &max_rows,
            )
        }) {
            return Ok(None);
        }
        // Whether the values of the aggregates do not depend on the order of
        // the rows, and on how they are split into partial states: they split
        // exactly, or they are ordered (whose values the native operator
        // orders itself, see `without_input_order`), or the aggregation fails
        // before it aggregates any row, natively as in DataFusion, as one of its
        // accumulators cannot be created when the streams are (see
        // `fails_when_created`, e.g. of a percentile out of range).
        let exact = aggregates().all(|aggregate| {
            splits_exactly(aggregate, raw.input(), &max_rows)
                || !aggregate.order_bys().is_empty()
        }) || aggregates()
            .any(|aggregate| fails_when_created(aggregate, grouped));
        if !exact && !single_partition {
            return Ok(None);
        }
        // DataFusion's partial aggregation of a single grouping set skips
        // aggregating the rows of its partition once most of them make new
        // groups (see the session's `skip_partial_aggregation_probe_rows_threshold`
        // and `skip_partial_aggregation_probe_ratio_threshold`, which are not
        // known here), and passes every further row on as a state of its own:
        // its final aggregation merges the state of the rows of a group before
        // with those of the rows after, which depend on DataFusion's batches.
        if !exact && last.is_some() && grouped && raw.group_expr().is_single() {
            return Ok(None);
        }
        // DataFusion aggregates the rows in the order of its input: sorted, if
        // it sorts it for ordered aggregates (a sort that the planner removes,
        // see `without_ordering_sort`). The native operator sorts the rows of
        // its thread if an aggregate requires them in an order (see
        // `hard_input_order`), by the same order, but not the rows that tie in
        // it as DataFusion does (which sorts every batch, by an unstable sort,
        // and merges them), so that the other aggregates may see another
        // sequence. It aggregates the rows in the order of its input otherwise,
        // which must then be DataFusion's.
        if !exact && (raw.input().is::<SortExec>() || hard_input_order(raw)?.is_some()) {
            return Ok(None);
        }
        let input = lower_input(raw.input())?;
        if !exact && !keeps_datafusion_order(&input) {
            return Ok(None);
        }
        if fail_in_the_order_of_the_batches(raw, &input) {
            return Ok(None);
        }
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
        Self::try_new(aggregate, None, merge, aggregate.schema(), input)
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
        Self::try_new(partial, Some(last), merge, last.schema(), input)
    }

    /// `raw` is the `AggregateExec` that consumes the input rows, `merge` the
    /// aggregates that merge the partial states and evaluate, `schema` the
    /// output schema.
    fn try_new(
        raw: &AggregateExec,
        last: Option<&AggregateExec>,
        merge: Vec<Arc<AggregateFunctionExpr>>,
        schema: SchemaRef,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        let single_stage = last.is_none();
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
            .map(|(expr, filter)| AggregateUpdate::new(expr, filter.clone()))
            .collect::<Vec<_>>();
        let num_fields = group_schema.fields().len() + merge.len();
        if schema.fields().len() != num_fields {
            return internal_err!(
                "The schema of the AggregateExec has {} fields, expected {num_fields}",
                schema.fields().len()
            );
        }
        // The constructors compute all groups of the limits that `try_lower`
        // leaves to DataFusion, which is all that the sort above needs.
        let top_k = match group_limit(raw, last) {
            GroupLimit::TopK(top_k) => Some(top_k),
            GroupLimit::None | GroupLimit::Unsupported => None,
        };
        let bypass = top_k.is_none()
            && input_order.is_none()
            && rows_can_bypass_tables(&group_by, &update, &merge, &input_schema);
        let bound = std::cell::OnceCell::new();
        let max_rows = || *bound.get_or_init(|| max_rows(raw.input()));
        let splits_exactly = update
            .iter()
            .map(|update| update.expr.as_ref())
            .chain(merge.iter().map(Arc::as_ref))
            .all(|aggregate| splits_exactly(aggregate, raw.input(), &max_rows));
        let input_partitions = raw.input().output_partitioning().partition_count();
        let evaluates_in_order =
            input_partitions == 1 && own_parts(raw) + pipeline_parts(&input) > 0;
        Ok(Self {
            children: vec![input],
            table_capacity: TableCapacity::default(),
            spec: Arc::new(AggregateSpec {
                input_schema,
                group_by,
                group_schema,
                update,
                merge,
                schema,
                input_order,
                single_stage,
                top_k,
                bypass,
                splits_exactly,
                input_partitions,
                evaluates_in_order,
            }),
        })
    }

    /// The configuration extension of a session that trusts `functions`, the
    /// built-in aggregate functions of DataFusion (e.g.
    /// `datafusion::functions_aggregate::all_default_aggregate_functions()`),
    /// whose accumulators cannot fail: `count`, `sum`, `min`, `max`, `avg`,
    /// `first_value`, `last_value`, `bool_and`, `bool_or`, `bit_and`,
    /// `bit_or`, `bit_xor`, `approx_distinct`, `median` and `array_agg`, if
    /// they are implemented by DataFusion's types. In the sessions that trust
    /// the functions of all aggregates of an aggregation (that are neither
    /// `DISTINCT` nor ordered), the input rows of the threads whose groups
    /// hardly repeat bypass their thread-local tables, when the aggregation
    /// runs on several threads.
    ///
    /// ```
    /// use datafusion::prelude::{SessionConfig, SessionContext};
    /// use datafusion_physical_pipeline::aggregate::PhysicalHashAggregate;
    ///
    /// // The built-in aggregate functions of DataFusion, which a session
    /// // registers by default.
    /// let functions = datafusion::functions_aggregate::all_default_aggregate_functions();
    /// let trusted = PhysicalHashAggregate::trusted_aggregate_functions(&functions);
    /// assert!(trusted.names().contains(&"count"));
    /// let config = SessionConfig::new().with_option_extension(trusted);
    /// let ctx = SessionContext::new_with_config(config);
    /// ```
    pub fn trusted_aggregate_functions(
        functions: &[Arc<AggregateUDF>],
    ) -> TrustedAggregateFunctions {
        TrustedAggregateFunctions::new(functions)
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

    /// Use thread-local tables of capacity `capacity` (in groups) for a
    /// grouped aggregation, e.g. tiny ones to test how they are flushed.
    #[cfg(test)]
    pub(crate) fn with_table_capacity(mut self, capacity: TableCapacity) -> Self {
        self.table_capacity = capacity;
        self
    }

    #[cfg(test)]
    pub(crate) fn spec(&self) -> Arc<AggregateSpec> {
        Arc::clone(&self.spec)
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

/// Whether `aggregate` advertises the partitions of its input: DataFusion
/// aggregates the input partitions of the partitions an aggregation
/// advertises.
fn advertises_its_input_partitions(aggregate: &AggregateExec) -> bool {
    aggregate
        .properties()
        .output_partitioning()
        .partition_count()
        == aggregate.input().output_partitioning().partition_count()
}

/// Whether no group of `aggregate`, the `AggregateExec` that produces the
/// final values, spans several partitions of its input. DataFusion
/// aggregates every partition on its own, which is the same as aggregating
/// all rows together exactly then: if there is a single partition, or the
/// partitions are partitioned by (a subset of) the group keys. (With grouping
/// sets, a single stage aggregation makes groups of the rows of all
/// partitions, e.g. the grand total of a `ROLLUP`.)
fn groups_are_partition_local(aggregate: &AggregateExec) -> bool {
    let input = aggregate.input();
    let partitioning = input.output_partitioning();
    if partitioning.partition_count() == 1 {
        return true;
    }
    let group_by = aggregate.group_expr();
    let keys = group_by.input_exprs();
    !group_by.has_grouping_set()
        && !keys.is_empty()
        && partitioning.keeps_keys_local(&keys, input.equivalence_properties(), true)
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

/// The limit on the groups of an aggregation, see
/// [`PhysicalHashAggregate::try_lower`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupLimit {
    /// All groups are produced.
    None,
    /// A top-k aggregation that the native operator reproduces.
    TopK(TopK),
    /// A limit that the native operator does not reproduce.
    Unsupported,
}

/// The largest limit of a top-k aggregation that the native operator
/// reproduces: it merges the top groups of all partitions into a single
/// batch, which must stay small.
const MAX_TOP_K_LIMIT: usize = 1 << 16;

/// The limit on the groups of the aggregation whose stage that consumes the
/// input rows is `raw`, and whose final stage is `last` (if it has two
/// stages), see [`PhysicalHashAggregate::try_lower`] and [`TopK`].
fn group_limit(raw: &AggregateExec, last: Option<&AggregateExec>) -> GroupLimit {
    let options = raw.limit_options();
    if last.is_some_and(|last| last.limit_options() != options) {
        return GroupLimit::Unsupported;
    }
    let Some(options) = options else {
        return GroupLimit::None;
    };
    // A soft limit has no order.
    let Some(descending) = options.descending() else {
        return GroupLimit::Unsupported;
    };
    let limit = options.limit();
    let group_by = raw.group_expr();
    let schema = raw.input().schema();
    let orders = |expr: &Arc<dyn PhysicalExpr>| {
        is_infallible(expr, &schema)
            && expr.data_type(&schema).is_ok_and(|data_type| {
                orders_like_datafusion(&data_type) && holds_like_datafusion(&data_type)
            })
    };
    let [(key, _)] = group_by.expr() else {
        return GroupLimit::Unsupported;
    };
    // The top groups of every partition depend on the partitioning.
    let partitions = last
        .unwrap_or(raw)
        .properties()
        .output_partitioning()
        .partition_count();
    if group_by.has_grouping_set()
        || partitions != 1
        || !(1..=MAX_TOP_K_LIMIT).contains(&limit)
        || !orders(key)
        || raw.filter_expr().iter().any(Option::is_some)
    {
        return GroupLimit::Unsupported;
    }
    // The groups of a top-k aggregation by a `MIN` or `MAX` depend on the
    // order of its rows when their values tie (see `TopK`).
    if !raw.aggr_expr().is_empty() {
        return GroupLimit::Unsupported;
    }
    GroupLimit::TopK(TopK { limit, descending })
}

/// Whether the native operator holds the values of `data_type` of the groups
/// of a top-k aggregation wherever DataFusion's top-k aggregation does. The
/// native operator holds all groups of a thread-local table (and the top
/// groups of every flush of a partition) before it keeps their top groups,
/// DataFusion's top-k aggregation its top groups only: strings with 32-bit
/// offsets (`Utf8`) of more groups than DataFusion holds may overflow the
/// offsets of the arrays of the native tables (e.g. two distinct keys of 1.1
/// GB), so their aggregations are left to DataFusion. Fixed-width values,
/// `LargeUtf8` (64-bit offsets) and `Utf8View` (views of several buffers) do
/// not overflow.
fn holds_like_datafusion(data_type: &DataType) -> bool {
    data_type != &DataType::Utf8
}

/// Whether DataFusion's top-k aggregation compares the values of
/// `data_type` as its sorts do, so that it keeps the groups that the sort
/// above it keeps: integers, decimals, temporal values (but intervals) and
/// strings. Its heap compares floating point values both by the IEEE
/// comparisons and by their total order, which disagree on `-0.0`, `0.0` and
/// `NaN`s, so that it may drop a group that the sort would keep.
fn orders_like_datafusion(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
    )
}

/// Whether the input rows of an aggregation without an input order or a
/// limit may bypass the thread-local tables of a thread whose groups hardly
/// repeat (see `RadixLocalState::aggregate`), and be aggregated by the final
/// aggregation of their radix partitions directly:
///
/// * the aggregation has group keys and no grouping sets;
/// * the aggregates of the final aggregation (`merge`), which aggregate the
///   bypassed rows (see `GroupedHashTable::update_rows`), are those of the
///   stage that consumes the input rows (`update`), as in DataFusion's plans;
/// * the keys and the arguments (of the rows of a thread, which the radix
///   partitioning gathers from several chunks) are of flat types, which are
///   gathered without merging dictionaries or nested offsets.
///
/// The rows of an aggregation with aggregates bypass the tables only in the
/// sessions that trust the accumulators of all of its aggregates (see
/// [`AggregateSpec::rows_bypass_tables`]). The accumulators of aggregates
/// aggregate every row in the tables, when it is consumed, with the rows of
/// its group that the table aggregated so far, like DataFusion's: an
/// accumulator may fail depending on the rows that it aggregates together
/// (e.g. a user-defined sum that rejects negative running totals), so that
/// deferring its rows to the final aggregation fails later than DataFusion
/// (after the rows that follow them), and converting them to partial states
/// one by one (like DataFusion's partial stage when it skips its
/// aggregation, which it decides by its own counts) fails where DataFusion
/// does not. The accumulators of the built-in aggregate functions of
/// [`TrustedAggregateFunctions`] cannot fail on the rows they aggregate, nor
/// on the partial states they merge: the final aggregation of a partition
/// aggregates the bypassed rows into exactly the states that the tables
/// would have merged, so that the rows of their aggregations bypass the
/// tables where the session trusts them.
fn rows_can_bypass_tables(
    group_by: &PhysicalGroupBy,
    update: &[AggregateUpdate],
    merge: &[Arc<AggregateFunctionExpr>],
    schema: &SchemaRef,
) -> bool {
    if !group_by.is_single() || group_by.expr().is_empty() || update.len() != merge.len()
    {
        return false;
    }
    let aggregates_rows = update.iter().zip(merge).all(|(update, merge)| {
        let update = &update.expr;
        Arc::ptr_eq(update, merge)
            || (update.as_ref() == merge.as_ref()
                && update.is_distinct() == merge.is_distinct()
                && update.ignore_nulls() == merge.ignore_nulls()
                && update.is_reversed() == merge.is_reversed()
                && update.order_bys() == merge.order_bys())
    });
    let is_flat = |expr: &Arc<dyn PhysicalExpr>| {
        expr.data_type(schema)
            .is_ok_and(|data_type| is_flat_type(&data_type))
    };
    aggregates_rows
        && group_by.expr().iter().all(|(key, _)| is_flat(key))
        && update
            .iter()
            .all(|update| update.arguments.iter().all(is_flat))
}

/// The built-in aggregate functions of DataFusion: their names, and the types
/// that implement them (see [`implementation_type`]).
static DATAFUSION_AGGREGATES: LazyLock<Vec<(String, TypeId)>> = LazyLock::new(|| {
    datafusion_functions_aggregate::all_default_aggregate_functions()
        .iter()
        .map(|function| (function.name().to_string(), implementation_type(function)))
        .collect()
});

/// The name of `function` if it is one of DataFusion's built-in aggregate
/// functions: if it is implemented by the type that implements the built-in
/// function of its name (as [`TrustedAggregateFunctions`] identifies them).
fn builtin_name(function: &AggregateUDF) -> Option<&'static str> {
    let implementation = implementation_type(function);
    DATAFUSION_AGGREGATES
        .iter()
        .find(|(name, builtin)| name == function.name() && *builtin == implementation)
        .map(|(name, _)| name.as_str())
}

/// Whether the accumulators of `aggregate`, of an aggregation (`grouped`, or
/// without `GROUP BY`) of `input` (of at most `max_rows()` rows, if known), get
/// the values and the errors of DataFusion's accumulators, which
/// DataFusion updates with its batches, every aggregate on a whole batch
/// before the next aggregate, and the native operator with its chunks of at
/// most [`DEFAULT_VECTOR_SIZE`] rows.
///
/// * A user-defined aggregate may compute anything from the rows of a call of
///   its accumulator (e.g. add up the values of a batch first, then add that
///   sum to its total): its aggregations are left to DataFusion.
/// * A `DISTINCT` aggregate collects its distinct values in a hash set, which
///   reserves room for every batch it extends it with: the order of its
///   values depends on the batches, and so does the value of an aggregate
///   that does not split its rows exactly (see [`splits_exactly`]), e.g. a
///   `SUM(DISTINCT f)` of floating point values (that DataFusion does not
///   rewrite, next to a `COUNT(*)`), or an `ARRAY_AGG(DISTINCT x)`: their
///   aggregations are left to DataFusion.
/// * The accumulators of DataFusion's `percentile_cont`,
///   `approx_percentile_cont` and `approx_percentile_cont_with_weight` of a
///   percentile of a column validate it on every call, the range of its first
///   value and that it is constant: on every batch (without `GROUP BY`), or on
///   every row of a group. Which of the aggregates of an aggregation fails
///   first, and with which values, depends on the calls: their aggregations
///   are left to DataFusion. (Literal percentiles are validated when the
///   accumulators are created.)
/// * DataFusion's built-in accumulators update their states row by row, or
///   collect or combine their values exactly, so that their values do not
///   depend on the batches of their rows, but the t-digests of
///   `approx_percentile_cont`, `approx_percentile_cont_with_weight` and
///   `approx_median`, which merge every batch into a digest, and the sums
///   (and averages) of floating point values without `GROUP BY`, whose
///   accumulators add up every batch first (their grouped accumulators add up
///   their rows one by one). Their aggregations of a single partition
///   (`single_partition`), whose results are deterministic, are left to
///   DataFusion. (DataFusion's results of several partitions depend on the
///   order in which their partial states arrive.)
fn updates_like_datafusion(
    aggregate: &AggregateFunctionExpr,
    grouped: bool,
    single_partition: bool,
    input: &Arc<dyn ExecutionPlan>,
    max_rows: &dyn Fn() -> Option<usize>,
) -> bool {
    let Some(name) = builtin_name(aggregate.fun()) else {
        return false;
    };
    if aggregate.is_distinct() && !splits_exactly(aggregate, input, max_rows) {
        return false;
    }
    if matches!(
        name,
        "percentile_cont"
            | "approx_percentile_cont"
            | "approx_percentile_cont_with_weight"
    ) && aggregate
        .expressions()
        .iter()
        .skip(1)
        .any(|argument| !collect_columns(argument).is_empty())
    {
        return false;
    }
    if !single_partition {
        return true;
    }
    match name {
        "approx_percentile_cont"
        | "approx_percentile_cont_with_weight"
        | "approx_median" => false,
        "sum" | "avg" if !grouped => adds_up_exactly(aggregate, input, max_rows),
        _ => true,
    }
}

/// Whether the arguments of `aggregate`, of the rows of `input` (at most
/// `max_rows()`, if known), are values that a `sum` or an `avg` adds up in any
/// order to the same values (see [`splits_exactly`]): integers or decimals,
/// which they add up with (wrapping) integer arithmetic, or the casts into
/// `Float64` of integers whose sums are exact (see [`sums_exactly_as_floats`]),
/// e.g. the `avg` of a `SMALLINT`.
fn adds_up_exactly(
    aggregate: &AggregateFunctionExpr,
    input: &Arc<dyn ExecutionPlan>,
    max_rows: &dyn Fn() -> Option<usize>,
) -> bool {
    let schema = input.schema();
    aggregate.expressions().iter().all(|argument| {
        sums_exactly_as_floats(argument, input, max_rows)
            || argument.data_type(&schema).is_ok_and(|data_type| {
                data_type.is_integer()
                    || matches!(
                        data_type,
                        DataType::Decimal32(_, _)
                            | DataType::Decimal64(_, _)
                            | DataType::Decimal128(_, _)
                            | DataType::Decimal256(_, _)
                    )
            })
    })
}

/// Whether `argument`, of the rows of `input` (at most `max_rows()`, which is
/// only computed for such casts), is a cast into `Float64` of integers whose
/// sums are exact, in any order: of integers of at most `m` in magnitude,
/// every sum of at most `n` of them is an integer of at most `n * m`, which
/// `Float64` represents exactly if it is at most 2^53, so that every addition
/// of partial sums is exact. E.g. DataFusion's `avg` of a `SMALLINT` (`m` =
/// 2^15), which it adds up as `Float64` values, of up to 2^38 rows. The cast
/// may be computed by a projection below the aggregation, into which
/// DataFusion moves the expressions that several aggregates share
/// (`__common_expr_1`).
fn sums_exactly_as_floats(
    argument: &Arc<dyn PhysicalExpr>,
    input: &Arc<dyn ExecutionPlan>,
    max_rows: &dyn Fn() -> Option<usize>,
) -> bool {
    if let Some(column) = argument.downcast_ref::<Column>()
        && let Some(projection) = input.downcast_ref::<ProjectionExec>()
        && let Some(expr) = projection.expr().get(column.index())
    {
        return sums_exactly_as_floats(&expr.expr, projection.input(), max_rows);
    }
    let Some(cast) = argument.downcast_ref::<CastExpr>() else {
        return false;
    };
    let magnitude: u128 = match cast.expr().data_type(&input.schema()) {
        Ok(DataType::Int8) => 1 << 7,
        Ok(DataType::Int16) => 1 << 15,
        Ok(DataType::Int32) => 1 << 31,
        Ok(DataType::UInt8) => u8::MAX.into(),
        Ok(DataType::UInt16) => u16::MAX.into(),
        Ok(DataType::UInt32) => u32::MAX.into(),
        _ => return false,
    };
    cast.cast_type() == &DataType::Float64
        && max_rows().is_some_and(|rows| (rows as u128) * magnitude <= 1 << 53)
}

/// Whether the first error of `raw`, the `AggregateExec` that consumes the input
/// rows, which `input` (its lowered input) evaluates, may depend on the
/// batches of its input: if `raw` may fail, and several of the parts that it
/// and the native operators of its pipeline (see [`pipeline_parts`]) evaluate
/// may fail (see [`refined_parts`]). DataFusion evaluates the parts of an
/// operator (the group keys of an aggregation, then the arguments of its
/// aggregates, then their filters), each on all rows of a batch before the
/// next one, and an operator on a batch before the operator above it; the
/// native operators evaluate them on their chunks, so that another of them
/// may fail first, also with an error of the same kind (an accumulator that
/// fails to be created for every group counts as a part that may fail, see
/// [`fails_for_each_group`]): e.g. of `SUM(CAST(x
/// AS DECIMAL(3, 0)))` and `SUM(CAST(y AS DECIMAL(38, 25)))` (or of a filter
/// `CAST(x AS DECIMAL(3, 0)) + 0.5 > 1` below `SUM(CAST(y AS DECIMAL(38,
/// 25)))`), of a batch in which `x` overflows on its last row, and `y` on its
/// first, DataFusion raises the invalid argument of `x`, the native operators
/// the arithmetic overflow of `y`; and of `SUM(p * 10)` and `SUM(q * 10)` of
/// `DECIMAL(38, 0)`s, the arithmetic overflow of `p` and of `q`. (The planner
/// accepts several casts into decimals and several arithmetic overflows, see
/// `Failures::order_matters`, whichever of them comes first.)
fn fail_in_the_order_of_the_batches(
    raw: &AggregateExec,
    input: &Arc<dyn PhysicalOperator>,
) -> bool {
    let own = own_parts(raw);
    own > 0 && own + pipeline_parts(input) > 1
}

/// The number of the parts that `raw`, the `AggregateExec` that consumes the
/// input rows, evaluates on every batch that may fail (see [`refined_parts`]):
/// its group keys, the arguments and the `ORDER BY` expressions of its
/// aggregates (which DataFusion evaluates with their arguments) and their
/// filters, and the accumulators that fail to be created for every group (see
/// [`fails_for_each_group`]).
fn own_parts(raw: &AggregateExec) -> usize {
    let schema = raw.input().schema();
    let grouped = !raw.group_expr().is_true_no_grouping();
    let expressions = raw
        .group_expr()
        .expr()
        .iter()
        .map(|(expr, _)| Arc::clone(expr))
        .chain(raw.aggr_expr().iter().flat_map(|aggregate| {
            aggregate.expressions().into_iter().chain(
                aggregate
                    .order_bys()
                    .iter()
                    .map(|sort| Arc::clone(&sort.expr))
                    .collect::<Vec<_>>(),
            )
        }))
        .chain(raw.filter_expr().iter().flatten().cloned())
        .map(|expr| refined_parts(&expr, &schema))
        .sum::<usize>();
    let accumulators = raw
        .aggr_expr()
        .iter()
        .filter(|aggregate| fails_for_each_group(aggregate, grouped))
        .count();
    expressions + accumulators
}

/// Whether the accumulator of `aggregate` (of an aggregation `grouped` or
/// not) cannot be created, and DataFusion and the native operator fail when
/// they create their streams and states, before they evaluate any row: its
/// accumulator, without `GROUP BY`, or its groups accumulator, if its function
/// supports them. (Otherwise, they create an accumulator for every group, as
/// its first row comes, see [`fails_for_each_group`].)
fn fails_when_created(aggregate: &AggregateFunctionExpr, grouped: bool) -> bool {
    if grouped {
        aggregate.groups_accumulator_supported()
            && aggregate.create_groups_accumulator().is_err()
    } else {
        aggregate.create_accumulator().is_err()
    }
}

/// Whether the accumulator of `aggregate`, of an aggregation with `GROUP BY`
/// (`grouped`) whose function does not support groups accumulators, cannot be
/// created: DataFusion and the native operator create it for every group, as
/// its first row comes (see `GroupsAccumulatorAdapter`), and fail then, after
/// DataFusion evaluated the arguments of all aggregates on all rows of the
/// batch, the native operator on those of a chunk, e.g. of `NTH_VALUE(f, 0)`
/// (of an invalid index). It is a part that may fail (see
/// [`fail_in_the_order_of_the_batches`]).
fn fails_for_each_group(aggregate: &AggregateFunctionExpr, grouped: bool) -> bool {
    grouped
        && !aggregate.groups_accumulator_supported()
        && aggregate.create_accumulator().is_err()
}

/// The number of the parts of `expr` (over `schema`) that may fail (see
/// [`failures`]), but the decimal additions, subtractions and
/// multiplications that cannot overflow (see [`cannot_overflow`]), which
/// [`failures`] counts as parts that may overflow: e.g. of TPC-H's
/// `l_extendedprice * (1 - l_discount)`, of `DECIMAL(15, 2)`s.
fn refined_parts(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> usize {
    failures(expr, schema)
        .parts()
        .saturating_sub(overflows_ruled_out(expr, schema))
}

/// The number of the operations of `expr` that [`failures`] counts as parts
/// that may fail, but that cannot overflow (see [`cannot_overflow`]).
fn overflows_ruled_out(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> usize {
    let children = expr.children();
    let own = if cannot_overflow(expr, schema) {
        let below = children
            .iter()
            .map(|child| failures(child, schema).parts())
            .sum::<usize>();
        failures(expr, schema).parts().saturating_sub(below).min(1)
    } else {
        0
    };
    children
        .iter()
        .map(|child| overflows_ruled_out(child, schema))
        .fold(own, |ruled_out, child| ruled_out + child)
}

/// Whether `expr` (over `schema`) is an addition, a subtraction or a
/// multiplication of `Decimal128`s that cannot overflow, whatever the values
/// of its operands (within the precisions of their types, see
/// [`decimal_digits`]). Arrow rescales the operands of an addition (and a
/// subtraction) to their larger scale `s` and adds them up as `i128`s, which
/// overflow only above `2^127`: the unscaled values of operands of at most
/// `d1` and `d2` digits and of scales `s1` and `s2` add up to less than
/// `2 * 10^e`, with `e = max(d1 - s1, d2 - s2) + s`, which fits for `e <= 37`.
/// It multiplies them as `i128`s, to less than `10^(d1 + d2)`, which fits for
/// `d1 + d2 <= 38` (if the scale `s1 + s2` of the product is valid). Arrow does
/// not validate the precisions of the results.
fn cannot_overflow(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
        return false;
    };
    let (Some((d1, s1)), Some((d2, s2))) = (
        decimal_digits(binary.left(), schema),
        decimal_digits(binary.right(), schema),
    ) else {
        return false;
    };
    match binary.op() {
        Operator::Plus | Operator::Minus => {
            let scale = s1.max(s2);
            (d1 - s1).max(d2 - s2) + scale <= 37
        }
        Operator::Multiply => d1 + d2 <= 38 && s1 + s2 <= 38,
        _ => false,
    }
}

/// An upper bound of the number of digits of the unscaled values of `expr`,
/// of type `Decimal128` (over `schema`), and its scale, if known: the
/// precision of its type, for a column (of a precision below the maximum of
/// 38, which the type of a result may only cap), a literal or a cast into the
/// decimal (which fails on the values of more digits); or that of the
/// operands of an addition, a subtraction or a multiplication that cannot
/// overflow (see [`cannot_overflow`]): `max(d1 - s1, d2 - s2) + max(s1, s2) +
/// 1` and `d1 + d2` digits.
fn decimal_digits(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Option<(i32, i32)> {
    let DataType::Decimal128(precision, scale) = expr.data_type(schema).ok()? else {
        return None;
    };
    let (precision, scale) = (i32::from(precision), i32::from(scale));
    if expr.is::<Literal>()
        || expr.is::<CastExpr>()
        || (expr.is::<Column>() && precision < 38)
    {
        return Some((precision, scale));
    }
    let binary = expr.downcast_ref::<BinaryExpr>()?;
    let (d1, s1) = decimal_digits(binary.left(), schema)?;
    let (d2, s2) = decimal_digits(binary.right(), schema)?;
    if !cannot_overflow(expr, schema) {
        return None;
    }
    match binary.op() {
        Operator::Plus | Operator::Minus => {
            Some(((d1 - s1).max(d2 - s2) + s1.max(s2) + 1, s1.max(s2)))
        }
        Operator::Multiply => Some((d1 + d2, s1 + s2)),
        _ => None,
    }
}

/// The number of the parts that may fail (see [`refined_parts`]) of the
/// native operators that evaluate the chunks of the pipeline of `operator`,
/// from `operator` down to the source of the pipeline, as the planner's
/// `native_failures` counts them: the filters, the projections and the probe
/// keys and filters of hash joins, through the operators that pass on the
/// chunks of their inputs, down to the sources and pipeline breakers, which
/// belong to other pipelines (and bridged plans, which DataFusion evaluates on
/// whole batches).
fn pipeline_parts(operator: &Arc<dyn PhysicalOperator>) -> usize {
    if let Some(join) = operator.downcast_ref::<PhysicalHashJoin>() {
        let probe_schema = join.probe_side().schema();
        let keys = join
            .on()
            .iter()
            .map(|(_, key)| refined_parts(key, &probe_schema))
            .sum::<usize>();
        let filter = join.filter().map_or(0, |filter| {
            refined_parts(filter.expression(), filter.schema())
        });
        return keys + filter + pipeline_parts(join.probe_side());
    }
    if let Some(join) = operator.downcast_ref::<PhysicalNestedLoopJoin>() {
        return pipeline_parts(join.probe_side());
    }
    if let Some(join) = operator.downcast_ref::<PhysicalCrossJoin>() {
        return pipeline_parts(join.probe_side());
    }
    if let Some(subquery) = operator.downcast_ref::<PhysicalScalarSubquery>() {
        return pipeline_parts(subquery.input());
    }
    if operator.is_source() || operator.is_sink() {
        return 0;
    }
    let input_schema = || operator.children()[0].schema();
    let own = if let Some(filter) = operator.downcast_ref::<PhysicalFilter>() {
        refined_parts(filter.predicate(), &input_schema())
    } else if let Some(projection) = operator.downcast_ref::<PhysicalProjection>() {
        let input_schema = input_schema();
        projection
            .exprs()
            .iter()
            .map(|(expr, _)| refined_parts(expr, &input_schema))
            .sum()
    } else {
        test_operator_parts(operator)
    };
    operator
        .children()
        .iter()
        .map(pipeline_parts)
        .fold(own, |parts, child| parts + child)
}

/// The number of the parts that may fail of `operator`, a stand-in of the
/// tests for the native filter or projection (see [`pipeline_parts`]).
#[cfg(test)]
fn test_operator_parts(operator: &Arc<dyn PhysicalOperator>) -> usize {
    use super::test_support::{TestFilter, TestProjection};
    let input_schema = || operator.children()[0].schema();
    if let Some(filter) = operator.downcast_ref::<TestFilter>() {
        return refined_parts(filter.predicate(), &input_schema());
    }
    if let Some(projection) = operator.downcast_ref::<TestProjection>() {
        let input_schema = input_schema();
        return projection
            .exprs()
            .iter()
            .map(|expr| refined_parts(expr, &input_schema))
            .sum();
    }
    0
}

#[cfg(not(test))]
fn test_operator_parts(_operator: &Arc<dyn PhysicalOperator>) -> usize {
    0
}

/// An upper bound of the number of rows of `plan`, from DataFusion's
/// statistics: its number of rows if it is exact (e.g. of a `MemTable`, or of
/// the metadata of Parquet files), or that of the input of a node that
/// produces at most the rows of its input (a filter, a projection, a sort, a
/// limit, or a node that coalesces or repartitions its partitions).
fn max_rows(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
    if let Ok(statistics) =
        StatisticsContext::new().compute_arc(plan, &StatisticsArgs::new())
        && let Precision::Exact(rows) = statistics.num_rows
    {
        return Some(rows);
    }
    let at_most_its_input = plan.is::<FilterExec>()
        || plan.is::<ProjectionExec>()
        || plan.is::<RepartitionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<SortExec>()
        || plan.is::<SortPreservingMergeExec>()
        || plan.is::<GlobalLimitExec>()
        || plan.is::<LocalLimitExec>();
    match plan.children().as_slice() {
        [input] if at_most_its_input => max_rows(input),
        _ => None,
    }
}

/// Whether `input`, the lowered input of an aggregation of a single partition
/// of DataFusion, produces the rows of that partition in DataFusion's order,
/// on a single thread (see `AggregateSpec::aggregates_in_order`): if it
/// consists of the native sources and of DataFusion's plans (in
/// [`BridgeOperator`]s), which read a partition in order, and of filters,
/// projections, limits and caches of chunks, which keep the order of their
/// rows. The other native operators produce their rows in other orders than
/// DataFusion's (e.g. an aggregation, whose groups DataFusion produces in the
/// order of their first rows, or a join or a sort, of rows that tie).
fn keeps_datafusion_order(input: &Arc<dyn PhysicalOperator>) -> bool {
    let keeps_order = input.is::<PhysicalFilter>()
        || input.is::<PhysicalProjection>()
        || input.is::<CachingOperator>()
        || input.is::<PhysicalLimit>()
        || input.is::<PhysicalPlaceholderRow>()
        || input.is::<BridgeOperator>()
        || input.is::<MemoryScanSource>()
        || input.is::<ParquetScanSource>()
        || input.is::<CsvScanSource>()
        || input.is::<EmptyScanSource>()
        || is_test_operator_that_keeps_order(input);
    keeps_order && input.children().iter().all(keeps_datafusion_order)
}

/// Whether `input` is a stand-in of the tests for the native filter or
/// projection, which keep the order of their rows too.
#[cfg(test)]
fn is_test_operator_that_keeps_order(input: &Arc<dyn PhysicalOperator>) -> bool {
    input.is::<super::test_support::TestFilter>()
        || input.is::<super::test_support::TestProjection>()
}

#[cfg(not(test))]
fn is_test_operator_that_keeps_order(_input: &Arc<dyn PhysicalOperator>) -> bool {
    false
}

/// The built-in aggregate functions of DataFusion whose accumulators cannot
/// fail on the rows that they aggregate, nor on the partial states that they
/// merge (see [`TrustedAggregateFunctions`]): their name, and the function of
/// `datafusion-functions-aggregate` that returns them.
///
/// * `count`, `sum`, `avg`, `bit_and`, `bit_or` and `bit_xor` add up (with
///   wrapping arithmetic) or combine fixed-width values, `min`, `max`,
///   `bool_and` and `bool_or` compare them, `first_value` and `last_value`
///   keep one of them, `approx_distinct` adds their hashes to a
///   HyperLogLog sketch, and `median` and `array_agg` collect them. None of
///   them fails on the values of its rows, or on how its rows are grouped
///   into batches and partial states.
/// * Evaluating the final value of a group may fail (e.g. the average of
///   decimals that does not fit the output precision), but on the final state
///   of the group only, which is the same whether its rows were aggregated
///   in the thread-local tables or bypassed them: the states are sums,
///   counts, extremes or collections of the same values.
///
/// `DISTINCT` aggregates and aggregates with an `ORDER BY` are not trusted
/// (see [`TrustedAggregateFunctions::trusts`]): their accumulators keep sets
/// or ordered lists of values, of other states.
const BUILTIN_AGGREGATES: [(&str, BuiltinAggregate); 15] = [
    ("count", count::count_udaf),
    ("sum", sum::sum_udaf),
    ("min", min_max::min_udaf),
    ("max", min_max::max_udaf),
    ("avg", average::avg_udaf),
    ("first_value", first_last::first_value_udaf),
    ("last_value", first_last::last_value_udaf),
    ("bool_and", bool_and_or::bool_and_udaf),
    ("bool_or", bool_and_or::bool_or_udaf),
    ("bit_and", bit_and_or_xor::bit_and_udaf),
    ("bit_or", bit_and_or_xor::bit_or_udaf),
    ("bit_xor", bit_and_or_xor::bit_xor_udaf),
    ("approx_distinct", approx_distinct::approx_distinct_udaf),
    ("median", median::median_udaf),
    ("array_agg", array_agg::array_agg_udaf),
];

/// The function of `datafusion-functions-aggregate` that returns a built-in
/// aggregate function (see [`BUILTIN_AGGREGATES`]).
type BuiltinAggregate = fn() -> Arc<AggregateUDF>;

/// The type that implements `udaf`.
fn implementation_type(udaf: &AggregateUDF) -> TypeId {
    let implementation: &dyn Any = udaf.inner().as_ref();
    implementation.type_id()
}

/// Whether the value of `aggregate` (of the rows of `input`) is the same
/// however the rows of a group are split into partial states, which
/// are merged in any order: whether it is DataFusion's built-in (see
/// [`BUILTIN_AGGREGATES`], by its implementation type) `count`, `min`, `max`,
/// `bit_and`, `bit_or`, `bit_xor`, `bool_and`, `bool_or`, `approx_distinct`
/// or `median`, which count, compare (floating point values by their total
/// order), combine the bits of, sketch or collect the values of their rows,
/// or its `sum` or `avg` of values that they add up exactly (see
/// [`adds_up_exactly`], of an input of at most `max_rows()` rows, if known):
/// integers or decimals, which they add up with (wrapping) integer
/// arithmetic, and the casts into `Float64` of small integers; or its
/// `array_agg`, `first_value`, `last_value` or `nth_value` ordered by their
/// own arguments (e.g. `ARRAY_AGG(x ORDER BY x)`, see
/// [`orders_its_arguments`]), whose values do not depend on the order of the
/// rows either. The values of the other aggregates may depend on it: e.g. a
/// sum of floating point values, whose rounding depends on how its additions
/// are grouped (`(s + a) + b` is not always `s + (a + b)`), a first or last
/// value, an array, or a user-defined aggregate.
fn splits_exactly(
    aggregate: &AggregateFunctionExpr,
    input: &Arc<dyn ExecutionPlan>,
    max_rows: &dyn Fn() -> Option<usize>,
) -> bool {
    let Some(name) = builtin_name(aggregate.fun()) else {
        return false;
    };
    match name {
        "count" | "min" | "max" | "bit_and" | "bit_or" | "bit_xor" | "bool_and"
        | "bool_or" | "approx_distinct" | "median" => true,
        "sum" | "avg" => adds_up_exactly(aggregate, input, max_rows),
        "array_agg" | "first_value" | "last_value" | "nth_value" => {
            orders_its_arguments(aggregate)
        }
        _ => false,
    }
}

/// Whether `aggregate` is ordered by all of its arguments but literals (e.g.
/// `ARRAY_AGG(x ORDER BY x)` or `NTH_VALUE(x, 2 ORDER BY y, x)`): the rows
/// that tie in its order have the same values of its arguments, so that its
/// array, or its first, last or n-th value, is the same in whatever order its
/// rows come, and however they are split into partial states.
fn orders_its_arguments(aggregate: &AggregateFunctionExpr) -> bool {
    let order = aggregate.order_bys();
    !order.is_empty()
        && aggregate.expressions().iter().all(|argument| {
            argument.downcast_ref::<Literal>().is_some()
                || order
                    .iter()
                    .any(|sort| sort.expr.as_ref() == argument.as_ref())
        })
}

/// The built-in aggregate functions of DataFusion whose accumulators a
/// session trusts the native aggregation not to fail (see
/// [`BUILTIN_AGGREGATES`]): a configuration extension of the session, created
/// with [`PhysicalHashAggregate::trusted_aggregate_functions`].
///
/// In the sessions that trust the aggregate functions of all aggregates of an
/// aggregation, the input rows of a thread (of several threads) whose groups
/// hardly repeat bypass its thread-local table, like the rows of aggregations
/// without aggregates (see `rows_can_bypass_tables`): they are aggregated
/// once, by the final aggregation of their radix partition, rather than in the
/// thread-local table first (DataFusion skips its partial aggregation then).
///
/// A function is trusted if it is implemented by the type of
/// `datafusion-functions-aggregate` that implements the built-in function of
/// its name (the type of `count_udaf()` for `count`, ...): a user-defined
/// function that shadows a built-in one under its name, also with a type of
/// the same name, is not trusted, and neither is a built-in function of
/// another version of DataFusion than the one of this crate.
#[derive(Debug, Clone, Default)]
pub struct TrustedAggregateFunctions {
    /// The trusted functions: their names, and the types that implement them.
    implementations: Vec<(&'static str, TypeId)>,
}

impl TrustedAggregateFunctions {
    /// Trust `functions`, the built-in aggregate functions of DataFusion
    /// (e.g. `datafusion::functions_aggregate::all_default_aggregate_functions()`):
    /// those whose accumulators cannot fail (see [`BUILTIN_AGGREGATES`]), if
    /// they are implemented by DataFusion's types.
    fn new(functions: &[Arc<AggregateUDF>]) -> Self {
        let mut implementations: Vec<(&'static str, TypeId)> = vec![];
        for function in functions {
            let Some(&(name, builtin)) = BUILTIN_AGGREGATES
                .iter()
                .find(|(name, _)| *name == function.name())
            else {
                continue;
            };
            let implementation = (name, implementation_type(&builtin()));
            if implementation_type(function) == implementation.1
                && !implementations.contains(&implementation)
            {
                implementations.push(implementation);
            }
        }
        Self { implementations }
    }

    /// The names of the trusted functions.
    pub fn names(&self) -> Vec<&'static str> {
        self.implementations.iter().map(|(name, _)| *name).collect()
    }

    /// Whether the accumulators of `aggregate` cannot fail: whether it is
    /// neither `DISTINCT` nor ordered (`ORDER BY`), and its function is a
    /// trusted one.
    pub(crate) fn trusts(&self, aggregate: &AggregateFunctionExpr) -> bool {
        let function = aggregate.fun();
        !aggregate.is_distinct()
            && aggregate.order_bys().is_empty()
            && self
                .implementations
                .contains(&(function.name(), implementation_type(function)))
    }
}

impl ExtensionOptions for TrustedAggregateFunctions {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, _value: &str) -> Result<()> {
        config_err!(
            "{key} cannot be set: aggregate functions are trusted with \
             `PhysicalHashAggregate::trusted_aggregate_functions`"
        )
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![ConfigEntry {
            key: format!("{}.functions", Self::PREFIX),
            value: Some(self.names().join(",")),
            description: "The built-in aggregate functions whose accumulators the native aggregation of the pipeline engine trusts not to fail",
        }]
    }
}

impl ConfigExtension for TrustedAggregateFunctions {
    const PREFIX: &'static str = "pipeline_trusted_aggregate_functions";
}

/// The widest fixed-size binaries of the keys and arguments of rows that may
/// bypass the thread-local tables (see [`is_flat_type`]).
const MAX_FIXED_SIZE_BINARY: i32 = 256;

/// Whether keys and arguments of `data_type` may bypass the thread-local
/// tables: arrays without children or dictionaries (which gathering the rows
/// of several chunks could fail to merge), whose `NULL`s are narrow. The
/// bypassed rows hold the arguments of an aggregate aligned with their rows,
/// with a `NULL` for every row that does not pass its filter, where
/// DataFusion (and the regular path) evaluate the arguments of the rows that
/// pass it only; a `NULL` of a fixed-size binary is as wide as its values, so
/// these are at most [`MAX_FIXED_SIZE_BINARY`] bytes wide.
fn is_flat_type(data_type: &DataType) -> bool {
    if let DataType::FixedSizeBinary(width) = data_type {
        return *width <= MAX_FIXED_SIZE_BINARY;
    }
    matches!(
        data_type,
        DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
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
        )?;
        if let Some(top_k) = &self.top_k {
            let order = if top_k.descending { "DESC" } else { "ASC" };
            write!(f, ", lim=[{}], order={order}", top_k.limit)?;
        }
        Ok(())
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

    /// Not if the aggregation aggregates its input in order (see
    /// `AggregateSpec::aggregates_in_order`), or if it or its pipeline may fail
    /// (see `AggregateSpec::evaluates_in_order`): the pipelines that end in it
    /// then run on a single thread.
    fn parallel_sink(&self) -> bool {
        !self.spec.aggregates_in_order() && !self.spec.evaluates_in_order
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
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        if self.spec.is_ungrouped() {
            return UngroupedAggregate::local_sink_state(&self.spec);
        }
        // The single thread of an aggregation that aggregates its input in
        // order (see `AggregateSpec::aggregates_in_order` and
        // `parallel_sink`) sees the rows in the order of the single partition
        // of DataFusion. Its table holds all of its groups, so that the
        // accumulators aggregate all rows of a group into a single state, as
        // DataFusion's do (see `TableCapacity::UNBOUNDED`).
        if self.spec.aggregates_in_order() {
            return RadixHashAggregate::local_sink_state(
                &self.spec,
                TableCapacity::UNBOUNDED,
                false,
            );
        }
        let options = ctx.task_ctx.session_config().options();
        let bypass = self.spec.rows_bypass_tables(options);
        RadixHashAggregate::local_sink_state(&self.spec, self.table_capacity, bypass)
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let batch = input.compact()?;
        if self.spec.is_ungrouped() {
            UngroupedAggregate::sink(&self.spec, batch, lstate)?;
        } else {
            RadixHashAggregate::sink(&self.spec, batch, gstate, lstate)?;
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
            RadixHashAggregate::finalize(&self.spec, gstate)
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
            RadixHashAggregate::global_source_state(&self.spec, sink_state)
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

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Int64Array, StringArray};
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion::functions_aggregate::min_max::max_udaf;
    use datafusion::functions_aggregate::sum::sum_udaf;
    use datafusion_common::ScalarValue;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{col, lit};
    use datafusion_physical_plan::{
        ChildrenPropertiesMode, Partitioning, ReplaceChildrenOptions,
    };

    use super::super::test_support::lower;
    use crate::bridge::BridgeOperator;
    use crate::executor::Executor;
    use crate::operators::union::test_support::assert_lowered_plan_byte_eq;
    use crate::test_utils::{canonical_batch, test_session_config};

    type Plan = Arc<dyn ExecutionPlan>;

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, true),
            Field::new("v", DataType::Int64, true),
        ]))
    }

    /// `partitions` partitions of rows `(k, v)`, optionally sorted by `k`.
    fn input(partitions: usize, sorted: bool) -> Result<Plan> {
        let schema = input_schema();
        let batch = |p: i64| {
            let rows = (0..1000).map(move |i| p * 1000 + i);
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from_iter(
                        rows.clone()
                            .map(|i| (i % 11 != 3).then(|| format!("key {}", i % 37))),
                    )),
                    Arc::new(Int64Array::from_iter(
                        rows.map(|i| (i % 7 != 2).then_some(i)),
                    )),
                ],
            )
        };
        let partitions = (0..partitions as i64)
            .map(|p| Ok(vec![batch(p)?]))
            .collect::<Result<Vec<_>>>()?;
        let source = MemorySourceConfig::try_new(&partitions, Arc::clone(&schema), None)?;
        let source = if sorted {
            let ordering =
                LexOrdering::new([PhysicalSortExpr::new_default(col("k", &schema)?)])
                    .expect("an ordering");
            source.try_with_sort_information(vec![ordering])?
        } else {
            source
        };
        Ok(Arc::new(
            datafusion_datasource::source::DataSourceExec::new(Arc::new(source)),
        ))
    }

    fn aggregates(schema: &SchemaRef) -> Result<Vec<Arc<AggregateFunctionExpr>>> {
        Ok(vec![
            Arc::new(
                AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
                    .schema(Arc::clone(schema))
                    .alias("count(*)")
                    .build()?,
            ),
            Arc::new(
                AggregateExprBuilder::new(sum_udaf(), vec![col("v", schema)?])
                    .schema(Arc::clone(schema))
                    .alias("sum(v)")
                    .build()?,
            ),
        ])
    }

    fn group_by(grouped: bool) -> Result<PhysicalGroupBy> {
        Ok(if grouped {
            PhysicalGroupBy::new_single(vec![(col("k", &input_schema())?, "k".into())])
        } else {
            PhysicalGroupBy::default()
        })
    }

    /// An aggregation in mode `mode` over `input`.
    fn single(mode: AggregateMode, grouped: bool, input: Plan) -> Result<Plan> {
        let schema = input.schema();
        Ok(Arc::new(AggregateExec::try_new(
            mode,
            group_by(grouped)?,
            aggregates(&schema)?,
            vec![None, None],
            input,
            schema,
        )?))
    }

    /// The partial aggregation of `input`.
    fn partial(grouped: bool, input: Plan) -> Result<Plan> {
        single(AggregateMode::Partial, grouped, input)
    }

    /// The final aggregation in mode `mode` of `states`, the partial states
    /// of an aggregation of `input_schema`, possibly redistributed.
    fn last(mode: AggregateMode, grouped: bool, states: Plan) -> Result<Plan> {
        let input_schema = input_schema();
        Ok(Arc::new(AggregateExec::try_new(
            mode,
            group_by(grouped)?.as_final(),
            aggregates(&input_schema)?,
            vec![None, None],
            states,
            input_schema,
        )?))
    }

    fn hash_repartition(states: Plan) -> Result<Plan> {
        let key = col("k", &states.schema())?;
        Ok(Arc::new(RepartitionExec::try_new(
            states,
            Partitioning::Hash(vec![key], 3),
        )?))
    }

    fn coalesce(states: Plan) -> Plan {
        Arc::new(CoalescePartitionsExec::new(states))
    }

    /// Lower `plan` with [`PhysicalHashAggregate::try_lower`], bridging the
    /// input.
    fn try_lower(plan: &Plan) -> Result<Option<PhysicalHashAggregate>> {
        PhysicalHashAggregate::try_lower(plan, |input| {
            Ok(Arc::new(BridgeOperator::new(Arc::clone(input))))
        })
    }

    /// The plans of all the aggregations that are lowered: a single stage,
    /// and both stages through a repartition, a coalesce or directly.
    /// A named function that creates a plan.
    type MakePlan = (&'static str, Box<dyn Fn() -> Result<Plan>>);

    fn lowered_plans() -> Vec<MakePlan> {
        vec![
            (
                "single",
                Box::new(|| single(AggregateMode::Single, true, input(1, false)?)),
            ),
            (
                "single ungrouped",
                Box::new(|| single(AggregateMode::Single, false, input(1, false)?)),
            ),
            (
                "single partitioned",
                Box::new(|| {
                    let input = hash_repartition(input(2, false)?)?;
                    single(AggregateMode::SinglePartitioned, true, input)
                }),
            ),
            (
                "final partitioned",
                Box::new(|| {
                    let states = hash_repartition(partial(true, input(4, false)?)?)?;
                    last(AggregateMode::FinalPartitioned, true, states)
                }),
            ),
            (
                "final",
                Box::new(|| {
                    let states = coalesce(partial(true, input(4, false)?)?);
                    last(AggregateMode::Final, true, states)
                }),
            ),
            (
                "final ungrouped",
                Box::new(|| {
                    let states = coalesce(partial(false, input(4, false)?)?);
                    last(AggregateMode::Final, false, states)
                }),
            ),
            (
                "final of one partition",
                Box::new(|| {
                    let states = partial(true, input(1, false)?)?;
                    last(AggregateMode::Final, true, states)
                }),
            ),
        ]
    }

    #[test]
    fn lowering_of_aggregations() -> Result<()> {
        for (name, make) in lowered_plans() {
            let plan = make()?;
            let aggregate = try_lower(&plan)?.unwrap_or_else(|| panic!("{name}"));
            assert_eq!(aggregate.schema(), plan.schema(), "{name}");
            assert_eq!(aggregate.is_ungrouped(), name.ends_with("ungrouped"));
            assert_eq!(aggregate.aggregates().count(), 2);
            assert!(aggregate.is_sink() && aggregate.parallel_sink());
            assert!(aggregate.is_source() && aggregate.parallel_source());
            // The input of the stage that consumes the rows is lowered.
            let bridge = aggregate.input().downcast_ref::<BridgeOperator>().unwrap();
            let raw = if name.starts_with("single") {
                plan.children()[0]
            } else {
                partial_aggregate(plan.children()[0]).unwrap().input()
            };
            assert!(Arc::ptr_eq(bridge.plan(), raw), "{name}");
        }
        let description = try_lower(&lowered_plans()[0].1()?)?.unwrap().description();
        assert_eq!(
            description,
            "PhysicalHashAggregate: gby=[k@0 as k], aggr=[count(*), sum(v)]"
        );
        Ok(())
    }

    #[test]
    fn aggregations_that_are_not_lowered() -> Result<()> {
        let not_lowered: Vec<(&str, Plan)> = vec![
            ("not an aggregation", input(1, false)?),
            ("partial", partial(true, input(4, false)?)?),
            (
                "final without partial",
                last(AggregateMode::Final, true, coalesce(input(4, false)?))?,
            ),
            (
                "clustered input",
                single(AggregateMode::Single, true, input(1, true)?)?,
            ),
            (
                "input without partitions",
                single(AggregateMode::Single, false, input(0, false)?)?,
            ),
            (
                "partial input without partitions",
                last(
                    AggregateMode::Final,
                    false,
                    coalesce(partial(false, input(0, false)?)?),
                )?,
            ),
            (
                "single stage of several partitions",
                single(AggregateMode::Single, true, input(2, false)?)?,
            ),
            (
                "ungrouped single stage of several partitions",
                single(AggregateMode::Single, false, input(2, false)?)?,
            ),
            (
                "final stage of states that are not partitioned by key",
                last(
                    AggregateMode::FinalPartitioned,
                    true,
                    Arc::new(RepartitionExec::try_new(
                        partial(true, input(2, false)?)?,
                        Partitioning::RoundRobinBatch(3),
                    )?),
                )?,
            ),
            (
                "single stage with grouping sets of several partitions",
                Arc::new(AggregateExec::try_new(
                    AggregateMode::SinglePartitioned,
                    PhysicalGroupBy::new(
                        vec![(col("k", &input_schema())?, "k".into())],
                        vec![(lit(ScalarValue::Utf8(None)), "k".into())],
                        vec![vec![false], vec![true]],
                        true,
                    ),
                    aggregates(&input_schema())?,
                    vec![None, None],
                    hash_repartition(input(2, false)?)?,
                    input_schema(),
                )?),
            ),
            (
                "stale partitions",
                lowered_plans()[0].1()?.replace_children(
                    vec![input(3, false)?],
                    ReplaceChildrenOptions::new(ChildrenPropertiesMode::Keep),
                )?,
            ),
            (
                "limited partial states",
                last(
                    AggregateMode::Final,
                    true,
                    Arc::new(
                        CoalescePartitionsExec::new(partial(true, input(4, false)?)?)
                            .with_fetch(Some(3)),
                    ),
                )?,
            ),
        ];
        for (name, plan) in &not_lowered {
            assert!(try_lower(plan)?.is_none(), "{name}");
        }
        let clustered = not_lowered
            .iter()
            .find(|(name, _)| *name == "clustered input")
            .and_then(|(_, plan)| plan.downcast_ref::<AggregateExec>())
            .unwrap();
        assert_eq!(
            clustered.group_clustering_mode(),
            &GroupClusteringMode::Full
        );

        // A final aggregation of the states of another aggregation.
        let states = coalesce(partial(true, input(4, false)?)?);
        let other = Arc::new(AggregateExec::try_new(
            AggregateMode::Final,
            group_by(false)?.as_final(),
            aggregates(&input_schema())?,
            vec![None, None],
            states,
            input_schema(),
        )?);
        assert!(!is_final_of(
            &other,
            partial_aggregate(other.input()).unwrap()
        ));
        assert!(try_lower(&(other as Plan))?.is_none());
        Ok(())
    }

    /// A limit of an aggregation with the group key `k`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Limit {
        None,
        /// DataFusion's soft limit of a `SELECT DISTINCT k ... LIMIT 3`: the
        /// stream stops reading its input once it has 3 groups.
        Soft,
        /// The top-k limit of a `SELECT DISTINCT k ... ORDER BY k DESC
        /// LIMIT 3`: the stream keeps the top 3 groups.
        TopKDistinct,
        /// The top-k limit of a `SELECT k, MAX(v) ... ORDER BY MAX(v) DESC
        /// LIMIT 3`.
        TopKMax,
    }

    /// An aggregation with the group key `k` (and `MAX(v)` for
    /// [`Limit::TopKMax`]) in mode `mode` over `input`, with limit `limit` if
    /// `apply`.
    fn limited(
        mode: AggregateMode,
        limit: Limit,
        apply: bool,
        input: Plan,
    ) -> Result<AggregateExec> {
        let schema = input_schema();
        let group_by =
            if matches!(mode, AggregateMode::Final | AggregateMode::FinalPartitioned) {
                group_by(true)?.as_final()
            } else {
                group_by(true)?
            };
        let aggregates = if limit == Limit::TopKMax {
            vec![Arc::new(
                AggregateExprBuilder::new(max_udaf(), vec![col("v", &schema)?])
                    .schema(Arc::clone(&schema))
                    .alias("max(v)")
                    .build()?,
            )]
        } else {
            vec![]
        };
        let filters = vec![None; aggregates.len()];
        let aggregate =
            AggregateExec::try_new(mode, group_by, aggregates, filters, input, schema)?;
        if !apply {
            return Ok(aggregate);
        }
        let descending = SortOptions::new(true, false);
        let limited = match limit {
            Limit::None => return Ok(aggregate),
            Limit::Soft => aggregate.try_optimize_distinct_soft_limit(3),
            Limit::TopKDistinct => aggregate.try_optimize_topk(3, "k", descending),
            Limit::TopKMax => aggregate.try_optimize_topk(3, "max(v)", descending),
        };
        let Some(limited) = limited else {
            panic!("no {limit:?} limit for mode {mode:?}");
        };
        let limited = limited.data;
        let order = limited.limit_options().expect("a limit").descending();
        assert_eq!(order.is_some(), limit != Limit::Soft, "{limit:?}");
        Ok(limited)
    }

    #[test]
    fn soft_limited_aggregations_are_not_lowered_and_top_k_ones_are() -> Result<()> {
        let plan = |aggregate: AggregateExec| Arc::new(aggregate) as Plan;
        // The top-k limit of the lowered aggregation, if it is lowered.
        let lowered_limit = |plan: &Plan| -> Result<Option<Option<TopK>>> {
            Ok(try_lower(plan)?.map(|aggregate| aggregate.spec.top_k))
        };
        for limit in [
            Limit::None,
            Limit::Soft,
            Limit::TopKDistinct,
            Limit::TopKMax,
        ] {
            // The key `k` is a `Utf8` string (see `holds_like_datafusion`), the
            // argument `v` of `MAX(v)` nullable: only the aggregations without
            // a limit are lowered.
            let expected = |limited: bool| {
                if limited && limit != Limit::None {
                    None
                } else {
                    Some(None)
                }
            };
            for apply in [false, true] {
                let single =
                    limited(AggregateMode::Single, limit, apply, input(1, false)?)?;
                assert_eq!(lowered_limit(&plan(single))?, expected(apply), "{limit:?}");
            }
            // A limit of the partial or of the final stage, or of both.
            for (mode, single_partition) in [
                (AggregateMode::FinalPartitioned, false),
                (AggregateMode::Final, true),
            ] {
                for (limit_partial, limit_final) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    let partial = limited(
                        AggregateMode::Partial,
                        limit,
                        limit_partial,
                        input(4, false)?,
                    )?;
                    let states = if single_partition {
                        coalesce(plan(partial))
                    } else {
                        hash_repartition(plan(partial))?
                    };
                    let last = limited(mode, limit, limit_final, states)?;
                    assert_eq!(
                        lowered_limit(&plan(last))?,
                        expected(limit_partial || limit_final),
                        "{limit:?} {mode:?}"
                    );
                }
            }
        }
        // A top-k aggregation by the integer key `v` is lowered: of a single
        // stage, or of two stages with the same limit and a single final
        // partition.
        let schema = input_schema();
        let by_v = |mode: AggregateMode, input: Plan, top_k: bool| -> Result<Plan> {
            let key = col("v", &input.schema())?;
            let aggregate = AggregateExec::try_new(
                mode,
                PhysicalGroupBy::new_single(vec![(key, "v".into())]),
                vec![],
                vec![],
                input,
                Arc::clone(&schema),
            )?;
            if !top_k {
                return Ok(Arc::new(aggregate));
            }
            let limited = aggregate
                .try_optimize_topk(3, "v", SortOptions::new(false, false))
                .expect("a top-k aggregation")
                .data;
            Ok(Arc::new(limited))
        };
        let top_k = Some(Some(TopK {
            limit: 3,
            descending: false,
        }));
        let single = by_v(AggregateMode::Single, input(1, false)?, true)?;
        assert_eq!(lowered_limit(&single)?, top_k);
        let partial = || by_v(AggregateMode::Partial, input(4, false)?, true);
        let last = by_v(AggregateMode::Final, coalesce(partial()?), true)?;
        assert_eq!(lowered_limit(&last)?, top_k);
        let last = by_v(AggregateMode::Final, coalesce(partial()?), false)?;
        assert_eq!(lowered_limit(&last)?, None);
        let states = partial()?;
        let key = col("v", &states.schema())?;
        let states = Arc::new(RepartitionExec::try_new(
            states,
            Partitioning::Hash(vec![key], 3),
        )?);
        let last = by_v(AggregateMode::FinalPartitioned, states, true)?;
        assert_eq!(lowered_limit(&last)?, None);
        Ok(())
    }

    #[test]
    fn lowered_input_must_have_the_input_schema() -> Result<()> {
        let plan = lowered_plans()[0].1()?;
        let err = PhysicalHashAggregate::try_lower(&plan, |_| {
            Ok(Arc::new(BridgeOperator::new(Arc::clone(&plan))))
        })
        .unwrap_err();
        assert!(err.to_string().contains("has schema"), "{err}");

        // The constructors check the modes of the aggregations.
        let aggregate = plan.downcast_ref::<AggregateExec>().unwrap();
        let bridge = Arc::new(BridgeOperator::new(input(1, false)?));
        let err = PhysicalHashAggregate::try_new_two_stage(
            aggregate,
            aggregate,
            Arc::clone(&bridge) as _,
        )
        .unwrap_err();
        assert!(err.to_string().contains("requires a Partial"), "{err}");
        let states = partial(true, input(1, false)?)?;
        let partial = states.downcast_ref::<AggregateExec>().unwrap();
        let err = PhysicalHashAggregate::try_new_single_stage(partial, bridge as _)
            .unwrap_err();
        assert!(err.to_string().contains("requires a Single"), "{err}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lowered_aggregations_produce_the_results_of_datafusion() -> Result<()> {
        for threads in [1, 2, 4, 16] {
            let task_ctx = Arc::new(TaskContext::default().with_session_config(
                test_session_config().with_target_partitions(threads),
            ));
            for (name, make) in lowered_plans() {
                let root = assert_lowered_plan_byte_eq(
                    &make,
                    lower,
                    Arc::clone(&task_ctx),
                    false,
                )
                .await;
                assert!(root.is::<PhysicalHashAggregate>(), "{name}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregations_can_be_executed_again() -> Result<()> {
        for (name, make) in lowered_plans() {
            let plan = make()?;
            let expected = datafusion_physical_plan::collect(
                make()?,
                Arc::new(TaskContext::default()),
            )
            .await?;
            let root = lower(&plan)?;
            let canonical = |batches: &[RecordBatch]| {
                canonical_batch(&plan.schema(), batches, false).unwrap()
            };
            for threads in [4, 1, 4] {
                let actual = Executor::collect(
                    Arc::clone(&root),
                    Arc::new(TaskContext::default()),
                    threads,
                )
                .await?;
                assert_eq!(canonical(&actual), canonical(&expected), "{name}");
            }
        }
        Ok(())
    }

    #[test]
    fn top_k_selects_the_best_values_and_the_nulls() -> Result<()> {
        let values = Int64Array::from(vec![
            Some(5),
            None,
            Some(9),
            Some(1),
            None,
            Some(7),
            Some(3),
            None,
        ]);
        let select = |limit, descending| -> Result<Option<Vec<u32>>> {
            let top_k = TopK { limit, descending };
            Ok(top_k.select(&values)?.map(|indices| {
                let mut indices = indices.values().to_vec();
                indices.sort_unstable();
                indices
            }))
        };
        // The 2 greatest keys (9, 7), the 2 least (1, 3), and the `NULL` keys.
        assert_eq!(select(2, true)?, Some(vec![1, 2, 4, 5, 7]));
        assert_eq!(select(2, false)?, Some(vec![1, 3, 4, 6, 7]));
        assert_eq!(select(1, false)?, Some(vec![1, 3, 4, 7]));
        // All rows, if there are not more of them.
        assert_eq!(select(5, true)?, None);
        // Strings, by their bytes.
        let strings = StringArray::from(vec![Some("b"), Some("ab"), None, Some("B")]);
        let top_k = TopK {
            limit: 1,
            descending: false,
        };
        let indices = top_k.select(&strings)?.expect("a selection");
        assert_eq!(indices.values().as_ref(), &[3, 2]);
        Ok(())
    }

    #[test]
    fn top_k_merges_the_best_groups_of_two_sets_without_concatenating_them() -> Result<()>
    {
        use arrow::array::{AsArray, StringViewArray};
        use arrow::compute::concat_batches;
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8View, true),
            Field::new("m", DataType::Int64, true),
        ]));
        let batch = |keys: &[Option<&str>], values: &[Option<i64>]| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringViewArray::from(keys.to_vec())),
                    Arc::new(Int64Array::from(values.to_vec())),
                ],
            )
        };
        let left = batch(
            &[
                Some("a long key 3"),
                None,
                Some("a long key 9"),
                Some("a long key 5"),
            ],
            &[Some(30), Some(1), None, Some(50)],
        )?;
        let right = batch(
            &[
                Some("a long key 4"),
                Some("a long key 1"),
                Some("a long key 8"),
            ],
            &[Some(40), None, Some(80)],
        )?;
        // The rows of a batch, sorted.
        let rows = |batch: &RecordBatch| {
            let mut rows = (0..batch.num_rows())
                .map(|row| {
                    let key = batch.column(0).as_string_view();
                    let value = batch
                        .column(1)
                        .as_primitive::<arrow::datatypes::Int64Type>();
                    (
                        key.is_valid(row).then(|| key.value(row).to_string()),
                        value.is_valid(row).then(|| value.value(row)),
                    )
                })
                .collect::<Vec<_>>();
            rows.sort();
            rows
        };
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        for descending in [false, true] {
            for limit in [1, 2, 3, 10] {
                let top_k = TopK { limit, descending };
                for (left, right) in [(&left, &right), (&right, &left), (&left, &empty)] {
                    let merged = top_k.merge_groups(left, right)?;
                    let expected =
                        top_k.select_groups(concat_batches(&schema, [left, right])?)?;
                    assert_eq!(rows(&merged), rows(&expected), "{top_k:?}");
                    assert_eq!(merged.schema(), schema);
                }
            }
        }
        Ok(())
    }

    /// The aggregations of `plan` that [`PhysicalHashAggregate::try_lower`]
    /// lowers (with their inputs bridged), from the top: an aggregation of two
    /// stages once.
    fn lowered_aggregations(plan: &Plan) -> Result<Vec<PhysicalHashAggregate>> {
        if let Some(aggregate) = try_lower(plan)? {
            return Ok(vec![aggregate]);
        }
        let mut aggregates = vec![];
        for child in plan.children() {
            aggregates.extend(lowered_aggregations(child)?);
        }
        Ok(aggregates)
    }

    /// The first aggregation of `plan` from the top, as the constructors build
    /// it over its bridged input, also if [`PhysicalHashAggregate::try_lower`]
    /// leaves it to DataFusion.
    fn built_aggregation(plan: &Plan) -> Result<Option<PhysicalHashAggregate>> {
        let bridge = |input: &Plan| -> Arc<dyn PhysicalOperator> {
            Arc::new(crate::bridge::BridgeOperator::new(Arc::clone(input)))
        };
        if let Some(top) = plan.downcast_ref::<AggregateExec>() {
            match top.mode() {
                AggregateMode::Single | AggregateMode::SinglePartitioned => {
                    let input = bridge(top.input());
                    return PhysicalHashAggregate::try_new_single_stage(top, input)
                        .map(Some);
                }
                AggregateMode::Final | AggregateMode::FinalPartitioned => {
                    if let Some(partial) = partial_aggregate(top.input()) {
                        let input = bridge(partial.input());
                        return PhysicalHashAggregate::try_new_two_stage(
                            partial, top, input,
                        )
                        .map(Some);
                    }
                }
                AggregateMode::Partial | AggregateMode::PartialReduce => {}
            }
        }
        for child in plan.children() {
            if let Some(aggregate) = built_aggregation(child)? {
                return Ok(Some(aggregate));
            }
        }
        Ok(None)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inputs_have_at_most_the_rows_of_their_exact_statistics() -> Result<()> {
        // The 5,000 rows of `t1`, from the exact statistics of its `MemTable`,
        // through the nodes that produce at most the rows of their input (see
        // `max_rows`): a projection, a filter, a sort (also with a limit, a
        // bound above its 10 rows); but no bound of a join or an aggregation.
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        crate::test_utils::register_test_tables(&ctx)?;
        for (sql, rows) in [
            ("SELECT a, b FROM t1", Some(5_000)),
            (
                "SELECT a + 1 AS x FROM t1 WHERE b > 0 ORDER BY x LIMIT 10",
                Some(5_000),
            ),
            (
                "SELECT a + 1 AS x FROM t1 WHERE b > 0 ORDER BY x",
                Some(5_000),
            ),
            ("SELECT t1.a FROM t1 JOIN t2 ON t1.a = t2.k", None),
            ("SELECT a FROM t1 GROUP BY a", None),
        ] {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            assert_eq!(max_rows(&plan), rows, "{sql}");
        }
        Ok(())
    }

    /// The aggregation with a top-k limit at the top of `plan`.
    fn top_k_aggregation(plan: &Plan) -> Option<Plan> {
        if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
            && aggregate.limit_options().is_some()
        {
            return Some(Arc::clone(plan));
        }
        plan.children().into_iter().find_map(top_k_aggregation)
    }

    /// The lowered aggregations of `root` that are top-k aggregations.
    fn native_top_k_aggregations(root: &Arc<dyn PhysicalOperator>) -> Vec<TopK> {
        let mut top_k = vec![];
        if let Some(aggregate) = root.downcast_ref::<PhysicalHashAggregate>() {
            top_k.extend(aggregate.spec.top_k);
        }
        for child in root.children() {
            top_k.extend(native_top_k_aggregations(child));
        }
        top_k
    }

    /// Register table `tk`: 3,000 rows in 4 partitions (one of them empty),
    /// with keys `g` (211 values and `NULL`), `u` (`UInt8`), `dt` (dates) and
    /// strings `s` (`Utf8`), `sv` (`Utf8View`) and `ls` (`LargeUtf8`) of
    /// distinct values (but for nulls), and distinct values `v` (`NULL` for
    /// all rows of some groups of `g`), `dec` (decimals) and `t` (timestamps
    /// with a time zone), and distinct values that are not nullable: `id`,
    /// `nsv` (`Utf8View`), `ns` (`Utf8`), `ndec` (decimals) and `nt`
    /// (timestamps).
    fn register_tk(ctx: &datafusion::prelude::SessionContext) -> Result<()> {
        use arrow::array::{
            Date32Array, Decimal128Array, Int32Array, LargeStringArray, StringViewArray,
            TimestampMicrosecondArray, UInt8Array,
        };
        use arrow::datatypes::TimeUnit;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int32, true),
            Field::new("u", DataType::UInt8, true),
            Field::new("dt", DataType::Date32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("sv", DataType::Utf8View, true),
            Field::new("ls", DataType::LargeUtf8, true),
            Field::new("v", DataType::Int64, true),
            Field::new("id", DataType::Int64, false),
            Field::new("dec", DataType::Decimal128(20, 4), true),
            Field::new(
                "t",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
            Field::new("nsv", DataType::Utf8View, false),
            Field::new("ns", DataType::Utf8, false),
            Field::new("ndec", DataType::Decimal128(20, 4), false),
            Field::new(
                "nt",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]));
        let batch = |rows: std::ops::Range<i64>| -> Result<RecordBatch> {
            let g = |i: i64| (i % 53 != 7).then_some((i % 211) as i32);
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int32Array::from_iter(rows.clone().map(g))),
                Arc::new(UInt8Array::from_iter(
                    rows.clone()
                        .map(|i| (i % 41 != 2).then_some((i % 200) as u8)),
                )),
                Arc::new(Date32Array::from_iter(
                    rows.clone()
                        .map(|i| (i % 43 != 3).then_some(19_000 + (i % 700) as i32)),
                )),
                Arc::new(StringArray::from_iter(rows.clone().map(|i| {
                    (i % 37 != 11).then(|| format!("s{:05}", (i * 7) % 3_001))
                }))),
                Arc::new(StringViewArray::from_iter(rows.clone().map(|i| {
                    (i % 31 != 6).then(|| format!("a string that is not inlined {i:05}"))
                }))),
                Arc::new(LargeStringArray::from_iter(rows.clone().map(|i| {
                    (i % 29 != 8).then(|| format!("large {:05}", (i * 11) % 3_001))
                }))),
                Arc::new(Int64Array::from_iter(rows.clone().map(|i| {
                    let all_null_group = g(i).is_some_and(|g| g % 17 == 3);
                    (!all_null_group && i % 23 != 5)
                        .then_some((i * 1_000_003) % 1_000_000_007 - 500_000_000)
                }))),
                Arc::new(Int64Array::from_iter_values(rows.clone().map(|i| i * 3))),
                Arc::new(
                    Decimal128Array::from_iter(rows.clone().map(|i| {
                        (i % 19 != 4).then_some(i128::from((i * 37) % 100_003) - 50_000)
                    }))
                    .with_precision_and_scale(20, 4)?,
                ),
                Arc::new(
                    TimestampMicrosecondArray::from_iter(rows.clone().map(|i| {
                        (i % 47 != 9).then_some(1_600_000_000_000_000 + i * 1_000_003)
                    }))
                    .with_timezone("UTC"),
                ),
                Arc::new(StringViewArray::from_iter_values(rows.clone().map(|i| {
                    format!("a string that is not nullable {:05}", (i * 13) % 3_001)
                }))),
                Arc::new(StringArray::from_iter_values(
                    rows.clone().map(|i| format!("s{:05}", (i * 17) % 3_001)),
                )),
                Arc::new(
                    Decimal128Array::from_iter_values(
                        rows.clone()
                            .map(|i| i128::from((i * 41) % 100_003) - 50_000),
                    )
                    .with_precision_and_scale(20, 4)?,
                ),
                Arc::new(TimestampMicrosecondArray::from_iter_values(
                    rows.map(|i| 1_600_000_000_000_000 - i * 999_983),
                )),
            ];
            Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
        };
        let batches =
            |rows: std::ops::Range<i64>, size: usize| -> Result<Vec<RecordBatch>> {
                rows.clone()
                    .step_by(size)
                    .map(|start| batch(start..(start + size as i64).min(rows.end)))
                    .collect()
            };
        let partitions = vec![
            batches(0..1_500, 700)?,
            vec![],
            batches(1_500..2_990, 90)?,
            batches(2_990..3_000, 3)?,
        ];
        ctx.register_table(
            "tk",
            Arc::new(datafusion::datasource::MemTable::try_new(
                schema, partitions,
            )?),
        )?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_produce_the_results_of_datafusion() -> Result<()> {
        use super::super::test_support::assert_lowered_parity_with_setup;
        let setup = async |ctx: &datafusion::prelude::SessionContext| register_tk(ctx);
        // The queries, and whether their top-k aggregation is native when its
        // final stage has a single partition (on a single thread).
        let queries = [
            // By the group key, of every supported type, in both directions,
            // with the `NULL` key first or last.
            ("SELECT DISTINCT g FROM tk ORDER BY g LIMIT 5", true),
            ("SELECT DISTINCT g FROM tk ORDER BY g DESC LIMIT 5", true),
            (
                "SELECT DISTINCT g FROM tk ORDER BY g NULLS FIRST LIMIT 3",
                true,
            ),
            (
                "SELECT DISTINCT g FROM tk ORDER BY g DESC NULLS LAST LIMIT 7",
                true,
            ),
            ("SELECT g FROM tk GROUP BY g ORDER BY g LIMIT 1", true),
            ("SELECT DISTINCT u FROM tk ORDER BY u DESC LIMIT 4", true),
            ("SELECT DISTINCT dt FROM tk ORDER BY dt LIMIT 6", true),
            // Strings with 32-bit offsets are left to DataFusion.
            (
                "SELECT DISTINCT s FROM tk ORDER BY s DESC NULLS LAST LIMIT 3",
                false,
            ),
            ("SELECT DISTINCT sv FROM tk ORDER BY sv LIMIT 8", true),
            ("SELECT DISTINCT ls FROM tk ORDER BY ls DESC LIMIT 2", true),
            ("SELECT DISTINCT dec FROM tk ORDER BY dec LIMIT 5", true),
            (
                "SELECT DISTINCT t FROM tk ORDER BY t DESC NULLS LAST LIMIT 5",
                true,
            ),
            // More groups than there are.
            ("SELECT DISTINCT g FROM tk ORDER BY g DESC LIMIT 1000", true),
            (
                "SELECT DISTINCT g FROM tk WHERE g > 1000 ORDER BY g LIMIT 3",
                true,
            ),
            // By a `MAX` descending or a `MIN` ascending: left to DataFusion,
            // whose groups depend on the order of the rows when values tie.
            (
                "SELECT g, MAX(id) FROM tk GROUP BY g ORDER BY MAX(id) DESC LIMIT 4",
                false,
            ),
            (
                "SELECT g, MIN(id) FROM tk GROUP BY g ORDER BY MIN(id) LIMIT 5",
                false,
            ),
            (
                "SELECT u, MIN(ndec) FROM tk GROUP BY u ORDER BY MIN(ndec) LIMIT 3",
                false,
            ),
            (
                "SELECT s, MAX(nt) FROM tk GROUP BY s ORDER BY MAX(nt) DESC NULLS LAST \
                 LIMIT 6",
                false,
            ),
            (
                "SELECT ls, MAX(nt) FROM tk GROUP BY ls ORDER BY MAX(nt) DESC LIMIT 6",
                false,
            ),
            (
                "SELECT dt, MIN(nsv) FROM tk GROUP BY dt ORDER BY MIN(nsv) LIMIT 4",
                false,
            ),
            (
                "SELECT g, MAX(nsv) FROM tk GROUP BY g ORDER BY MAX(nsv) DESC LIMIT 1000",
                false,
            ),
            // DataFusion keeps groups whose aggregate is `NULL` depending on
            // the order of its rows: left to it (in no particular order).
            (
                "SELECT g, MAX(m) FROM (SELECT g, MAX(v) AS m FROM tk GROUP BY g \
                 ORDER BY m DESC NULLS LAST LIMIT 1000) GROUP BY g",
                false,
            ),
            (
                "SELECT g, MAX(v) FROM tk GROUP BY g ORDER BY MAX(v) DESC NULLS LAST LIMIT 5",
                false,
            ),
            // With projections, aggregated again, and a window, which sees all
            // groups of the aggregation.
            (
                "SELECT g + 1 AS g1, MAX(id) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 5",
                false,
            ),
            (
                "SELECT COUNT(*), SUM(m), MIN(g) FROM \
                 (SELECT g, MAX(id) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 5)",
                false,
            ),
            (
                "SELECT COUNT(*), MIN(s), MAX(s) FROM \
                 (SELECT DISTINCT s FROM tk ORDER BY s LIMIT 10)",
                false,
            ),
            (
                "SELECT COUNT(*), MIN(sv), MAX(sv) FROM \
                 (SELECT DISTINCT sv FROM tk ORDER BY sv LIMIT 10)",
                true,
            ),
            (
                "SELECT g, MAX(id) AS m, COUNT(*) OVER () AS c FROM tk GROUP BY g \
                 ORDER BY m DESC LIMIT 3",
                false,
            ),
            (
                "SELECT g, COUNT(*) OVER () AS c FROM tk GROUP BY g ORDER BY g DESC LIMIT 3",
                true,
            ),
        ];
        for target_partitions in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = crate::test_utils::datafusion_context(config.clone());
            register_tk(&ctx)?;
            let pipeline = crate::test_utils::pipeline_context(config.clone());
            register_tk(&pipeline)?;
            for (sql, native) in queries {
                // DataFusion plans a top-k aggregation.
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                assert!(top_k_aggregation(&plan).is_some(), "{sql}");
                // On several threads, its final stage has several partitions.
                let native = native && target_partitions == 1;
                // The lowering of the tests leaves windows (with their input)
                // to DataFusion.
                let window = sql.contains(" OVER ");
                let root =
                    assert_lowered_parity_with_setup(sql, config.clone(), &setup).await;
                let top_k = native_top_k_aggregations(&root);
                assert_eq!(top_k.len(), usize::from(native && !window), "{sql}");
                // The planner lowers it natively too (but the windows here).
                let plan = pipeline.sql(sql).await?.create_physical_plan().await?;
                let root = plan
                    .downcast_ref::<crate::bridge::PipelineRootExec>()
                    .expect("a pipeline plan")
                    .operator();
                let planned = native_top_k_aggregations(root);
                if !window {
                    assert_eq!(planned, top_k, "{sql}");
                }
                assert!(planned.is_empty() || planned == top_k || window, "{sql}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_produce_their_groups_in_the_order_of_datafusion()
    -> Result<()> {
        use super::super::test_support::{
            assert_aggregate_fails_like_datafusion_with_setup, lower,
        };
        use arrow::array::StringViewArray;
        // DataFusion's top-k aggregation produces its groups from the best one
        // to the worst (and the `NULL` key last): a projection above it that
        // fails on several of them fails on the first one, whose value its
        // error shows. So does the native one, also when it keeps all of its
        // groups (without selecting or merging them).
        let schema =
            Arc::new(Schema::new(vec![Field::new("k", DataType::Utf8View, true)]));
        let cases = [
            (
                vec!["invalid0021", "invalid0011"],
                "LIMIT 2",
                "'invalid0011'",
            ),
            (
                vec!["invalid0011", "invalid0021"],
                "DESC LIMIT 2",
                "'invalid0021'",
            ),
            (
                vec![
                    "invalid0050",
                    "invalid0030",
                    "invalid0010",
                    "invalid0040",
                    "invalid0020",
                ],
                "LIMIT 10",
                "'invalid0010'",
            ),
            (
                vec![
                    "invalid0010",
                    "invalid0030",
                    "invalid0050",
                    "invalid0020",
                    "invalid0040",
                ],
                "DESC LIMIT 3",
                "'invalid0050'",
            ),
        ];
        for (keys, order, message) in cases {
            let sql = format!(
                "SELECT CAST(k AS BIGINT) AS i, k FROM t GROUP BY k ORDER BY k {order}"
            );
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(StringViewArray::from(keys.clone()))],
            )?;
            let setup = async |ctx: &datafusion::prelude::SessionContext| -> Result<()> {
                let table = datafusion::datasource::MemTable::try_new(
                    Arc::clone(&schema),
                    vec![vec![batch.clone()]],
                )?;
                ctx.register_table("t", Arc::new(table))?;
                Ok(())
            };
            let config = test_session_config().with_target_partitions(1);
            // The top-k aggregation is native.
            let ctx = crate::test_utils::datafusion_context(config.clone());
            setup(&ctx).await?;
            let plan = ctx.sql(&sql).await?.create_physical_plan().await?;
            assert_eq!(native_top_k_aggregations(&lower(&plan)?).len(), 1, "{sql}");
            assert_aggregate_fails_like_datafusion_with_setup(
                &sql,
                config.clone(),
                &setup,
                message,
            )
            .await;
            // And through the planner.
            let pipeline = crate::test_utils::pipeline_context(config);
            setup(&pipeline).await?;
            let error = pipeline
                .sql(&sql)
                .await?
                .collect()
                .await
                .expect_err("a cast error");
            assert!(error.to_string().contains(message), "{sql}: {error}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_of_tied_values_are_left_to_datafusion() -> Result<()> {
        use super::super::test_support::assert_lowered_parity_with_setup;
        // Of the groups that tie with the worst of its top groups, the heap of
        // DataFusion's top-k aggregation by a `MIN` / `MAX` keeps the one whose
        // row came last, which decides the groups that the projection above it
        // evaluates `1000 / ...` for (and the groups the sort returns).
        let setup = async |_: &datafusion::prelude::SessionContext| Ok(());
        let queries = [
            // DataFusion keeps `a = 3`, not `a = 1`.
            "SELECT 1000 / (a - 1) AS q, a, MAX(b) AS m \
             FROM (VALUES (1, 1), (2, 1), (3, 1)) AS t(a, b) \
             GROUP BY a ORDER BY m DESC LIMIT 1",
            "SELECT 1000 / (a - 1) AS q, a, MIN(b) AS m \
             FROM (VALUES (1, 1), (2, 1), (3, 1)) AS t(a, b) \
             GROUP BY a ORDER BY m LIMIT 1",
            // Ties behind a better group.
            "SELECT 1000 / (a - 3) AS q, a, MAX(b) AS m \
             FROM (VALUES (1, 5), (2, 3), (3, 3), (4, 3), (5, 1)) AS t(a, b) \
             GROUP BY a ORDER BY m DESC LIMIT 2",
            // Groups that drop out and come back with a value that ties.
            "SELECT 1000 / (a - 2) AS q, a, MAX(b) AS m \
             FROM (VALUES (1, 1), (2, 2), (3, 1), (2, 1), (1, 2), (3, 2)) AS t(a, b) \
             GROUP BY a ORDER BY m DESC LIMIT 1",
            // Without a projection, the sort returns the group it keeps (`a = 3`,
            // and `a = 2`). (The order in which the sort of the pipeline engine
            // returns several rows that tie differs from DataFusion's, on the
            // base too: such queries are not used here.)
            "SELECT a, MAX(b) AS m FROM (VALUES (1, 1), (2, 1), (3, 1)) AS t(a, b) \
             GROUP BY a ORDER BY m DESC LIMIT 1",
            "SELECT a, MIN(b) AS m FROM (VALUES (3, 1), (1, 1), (2, 1)) AS t(a, b) \
             GROUP BY a ORDER BY m LIMIT 1",
        ];
        for target_partitions in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = crate::test_utils::datafusion_context(config.clone());
            for sql in queries {
                if target_partitions == 1 {
                    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                    assert!(top_k_aggregation(&plan).is_some(), "{sql}");
                }
                let root =
                    assert_lowered_parity_with_setup(sql, config.clone(), &setup).await;
                assert!(native_top_k_aggregations(&root).is_empty(), "{sql}");
            }
        }
        Ok(())
    }

    /// Register table `tkn`: 60,000 rows of a single partition, in batches of
    /// 1,000 rows, with distinct keys in a shuffled order `k` (and `NULL`),
    /// in ascending and descending order (`asc`, `desc`) and as strings
    /// (`sv`), 40,000 groups `g` of 1 or 2 rows, 100 groups `few`, and
    /// distinct values `v`: more groups than a thread-local table holds, and
    /// more rows, whose flushes set the cutoffs of top-k aggregations.
    fn register_tkn(ctx: &datafusion::prelude::SessionContext) -> Result<()> {
        use arrow::array::StringViewArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("asc", DataType::Int64, false),
            Field::new("desc", DataType::Int64, false),
            Field::new("sv", DataType::Utf8View, false),
            Field::new("g", DataType::Int64, false),
            Field::new("few", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let shuffled = |i: i64| (i * 7_919 + 13) % 60_011;
        let batches = (0..60_000)
            .step_by(1_000)
            .map(|start| {
                let rows = start..start + 1_000;
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from_iter(
                        rows.clone().map(|i| (i % 997 != 5).then_some(shuffled(i))),
                    )),
                    Arc::new(Int64Array::from_iter_values(rows.clone())),
                    Arc::new(Int64Array::from_iter_values(
                        rows.clone().map(|i| 60_000 - i),
                    )),
                    Arc::new(StringViewArray::from_iter_values(rows.clone().map(|i| {
                        format!("a key that is not inlined {:06}", shuffled(i))
                    }))),
                    Arc::new(Int64Array::from_iter_values(
                        rows.clone().map(|i| (i * 7_919) % 40_000),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        rows.clone().map(|i| (i * 7_919) % 100),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        rows.map(|i| (i * 104_729) % 1_000_003 - 500_000),
                    )),
                ];
                Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let table = datafusion::datasource::MemTable::try_new(schema, vec![batches])?;
        ctx.register_table("tkn", Arc::new(table))?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_of_many_groups_produce_the_results_of_datafusion()
    -> Result<()> {
        use super::super::test_support::assert_lowered_parity_with_setup;
        let setup = async |ctx: &datafusion::prelude::SessionContext| register_tkn(ctx);
        let queries = [
            // Shuffled keys, with the `NULL` key first or last.
            "SELECT DISTINCT k FROM tkn ORDER BY k LIMIT 10",
            "SELECT DISTINCT k FROM tkn ORDER BY k DESC NULLS LAST LIMIT 10",
            "SELECT DISTINCT k FROM tkn ORDER BY k NULLS FIRST LIMIT 5",
            "SELECT DISTINCT sv FROM tkn ORDER BY sv DESC LIMIT 6",
            // The best keys first, and last.
            "SELECT DISTINCT asc FROM tkn ORDER BY asc LIMIT 7",
            "SELECT DISTINCT asc FROM tkn ORDER BY asc DESC LIMIT 7",
            "SELECT DISTINCT desc FROM tkn ORDER BY desc LIMIT 7",
            // By a `MAX` or a `MIN` (left to DataFusion), of groups of 1 or 2
            // rows, whose values are shuffled, or come best (or worst) first.
            "SELECT g, MAX(v) FROM tkn GROUP BY g ORDER BY MAX(v) DESC LIMIT 10",
            "SELECT g, MIN(v) FROM tkn GROUP BY g ORDER BY MIN(v) LIMIT 10",
            "SELECT k, MIN(asc) FROM tkn GROUP BY k ORDER BY MIN(asc) LIMIT 5",
            "SELECT k, MAX(asc) FROM tkn GROUP BY k ORDER BY MAX(asc) DESC LIMIT 5",
            "SELECT sv, MIN(v) FROM tkn GROUP BY sv ORDER BY MIN(v) LIMIT 4",
            "SELECT g, MAX(sv) FROM tkn GROUP BY g ORDER BY MAX(sv) DESC LIMIT 3",
            // Fewer groups than a table holds (by a `MIN` or `MAX`, left to
            // DataFusion, too).
            "SELECT few, MAX(v) FROM tkn GROUP BY few ORDER BY MAX(v) DESC LIMIT 10",
            "SELECT few, MIN(asc) FROM tkn GROUP BY few ORDER BY MIN(asc) LIMIT 3",
            "SELECT few, MAX(asc) FROM tkn GROUP BY few ORDER BY MAX(asc) DESC LIMIT 3",
            "SELECT DISTINCT few FROM tkn ORDER BY few DESC LIMIT 5",
            // A limit of more groups than a default table holds: its table
            // holds twice as many (40,000), whose flush sets a cutoff.
            "SELECT DISTINCT k FROM tkn ORDER BY k LIMIT 20000",
            "SELECT DISTINCT asc FROM tkn ORDER BY asc DESC LIMIT 20000",
            "SELECT DISTINCT k FROM tkn ORDER BY k DESC LIMIT 32768",
            "SELECT DISTINCT k FROM tkn ORDER BY k LIMIT 32769",
        ];
        for target_partitions in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = crate::test_utils::datafusion_context(config.clone());
            register_tkn(&ctx)?;
            for sql in queries {
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                assert!(top_k_aggregation(&plan).is_some(), "{sql}");
                let root =
                    assert_lowered_parity_with_setup(sql, config.clone(), &setup).await;
                let native = native_top_k_aggregations(&root).len();
                let by_key = !sql.contains("MIN(") && !sql.contains("MAX(");
                assert_eq!(
                    native,
                    usize::from(target_partitions == 1 && by_key),
                    "{sql}"
                );
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "allocates more than 8 GiB"]
    async fn top_k_aggregations_of_huge_strings_produce_the_results_of_datafusion()
    -> Result<()> {
        use arrow::array::{LargeStringArray, StringArray};
        use arrow::buffer::{Buffer, OffsetBuffer};

        use super::super::test_support::assert_lowered_parity_with_setup;
        // Two distinct keys of 1,100,000,000 bytes, in batches of their own of
        // a single partition: together, more than an array of strings with
        // 32-bit offsets can hold. DataFusion's top-k aggregation holds the
        // best of them only.
        let len = 1_100_000_000;
        for data_type in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
            let schema =
                Arc::new(Schema::new(vec![Field::new("k", data_type.clone(), false)]));
            let key = |byte: u8| -> Result<RecordBatch> {
                let bytes = Buffer::from_vec(vec![byte; len]);
                let key: ArrayRef = match data_type {
                    DataType::Utf8 => Arc::new(StringArray::new(
                        OffsetBuffer::new(vec![0, len as i32].into()),
                        bytes,
                        None,
                    )),
                    _ => {
                        let key = LargeStringArray::new(
                            OffsetBuffer::new(vec![0, len as i64].into()),
                            bytes,
                            None,
                        );
                        arrow::compute::cast(&key, &data_type)?
                    }
                };
                Ok(RecordBatch::try_new(Arc::clone(&schema), vec![key])?)
            };
            let partitions = vec![vec![key(b'b')?, key(b'a')?]];
            let setup = async |ctx: &datafusion::prelude::SessionContext| {
                let table = datafusion::datasource::MemTable::try_new(
                    Arc::clone(&schema),
                    partitions.clone(),
                )?;
                ctx.register_table("wide", Arc::new(table))?;
                Ok(())
            };
            let config = test_session_config().with_target_partitions(1);
            let sql =
                "SELECT COUNT(*) FROM (SELECT DISTINCT k FROM wide ORDER BY k LIMIT 1)";
            assert_lowered_parity_with_setup(sql, config, setup).await;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_of_other_values_are_left_to_datafusion() -> Result<()> {
        // On a single thread, the final stages have a single partition.
        let config = test_session_config().with_target_partitions(1);
        let ctx = crate::test_utils::datafusion_context(config);
        crate::test_utils::register_test_tables(&ctx)?;
        register_tk(&ctx)?;
        let lowered = async |ctx: &datafusion::prelude::SessionContext, sql: &str| {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let aggregate = top_k_aggregation(&plan).expect("a limited aggregation");
            Ok::<_, datafusion_common::DataFusionError>(try_lower(&aggregate)?.is_some())
        };
        assert!(lowered(&ctx, "SELECT DISTINCT g FROM tk ORDER BY g LIMIT 3").await?);
        for sql in [
            // Floating point values, which DataFusion's top-k aggregation
            // does not compare like its sorts.
            "SELECT DISTINCT c FROM t1 ORDER BY c LIMIT 3",
            "SELECT a, MAX(c) FROM t1 GROUP BY a ORDER BY MAX(c) DESC NULLS LAST LIMIT 3",
            // Expressions that may fail (of a key that DataFusion computes in
            // the aggregation, not in a projection below it).
            "SELECT 100 / a AS q FROM t1 GROUP BY 100 / a ORDER BY q LIMIT 3",
            "SELECT g, MAX(1000 / id) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 3",
            // A nullable argument of a `MIN` / `MAX`.
            "SELECT g, MAX(v) AS m FROM tk GROUP BY g ORDER BY m DESC NULLS LAST LIMIT 3",
            // Strings with 32-bit offsets, as keys and as values.
            "SELECT DISTINCT s FROM tk ORDER BY s LIMIT 3",
            "SELECT g, MAX(ns) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 3",
            // A soft limit.
            "SELECT DISTINCT d FROM t1 LIMIT 3",
        ] {
            assert!(!lowered(&ctx, sql).await?, "{sql}");
        }
        // A final stage of several partitions.
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        register_tk(&ctx)?;
        assert!(!lowered(&ctx, "SELECT DISTINCT g FROM tk ORDER BY g LIMIT 3").await?);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rows_of_simple_aggregations_may_bypass_the_tables() -> Result<()> {
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        crate::test_utils::register_test_tables(&ctx)?;
        let untrusting = ConfigOptions::default();
        let with_extension = |trusted: TrustedAggregateFunctions| {
            let mut options = ConfigOptions::default();
            options.extensions.insert(trusted);
            options
        };
        let functions =
            datafusion::functions_aggregate::all_default_aggregate_functions();
        let trusting = with_extension(
            PhysicalHashAggregate::trusted_aggregate_functions(&functions),
        );
        // Sessions whose extension trusts no function behave like the sessions
        // without it.
        let untrusting_sessions = [
            untrusting,
            with_extension(TrustedAggregateFunctions::default()),
            with_extension(PhysicalHashAggregate::trusted_aggregate_functions(&[
                Arc::new(super::super::test_support::lookalike_count()),
                datafusion::functions_aggregate::stddev::stddev_udaf(),
            ])),
        ];
        // The queries, and whether their rows bypass the tables in a session
        // that does not trust DataFusion's built-in aggregate functions, and
        // in a session that does.
        let queries = [
            // Without aggregates.
            ("SELECT DISTINCT id, c, f, ts FROM t1", true, true),
            ("SELECT a, d FROM t1 GROUP BY a, d", true, true),
            // The accumulators of aggregates aggregate every row in the tables,
            // but in the sessions that trust their functions (see
            // `rows_can_bypass_tables`).
            (
                "SELECT a, COUNT(*), SUM(b), MIN(d), AVG(g) FROM t1 GROUP BY a",
                false,
                true,
            ),
            (
                "SELECT h, MAX(c), BOOL_AND(e), BOOL_OR(e), BIT_AND(b), BIT_OR(a), \
                 BIT_XOR(b), APPROX_DISTINCT(d), MEDIAN(b) FILTER (WHERE e), \
                 ARRAY_AGG(a), FIRST_VALUE(d), LAST_VALUE(f), COUNT(a, b) \
                 FROM t1 GROUP BY h",
                false,
                true,
            ),
            // `DISTINCT` and ordered aggregates, whose accumulators are not
            // trusted.
            (
                "SELECT d, h, COUNT(DISTINCT a), MEDIAN(b) FILTER (WHERE e) FROM t1 \
                 GROUP BY d, h",
                false,
                false,
            ),
            (
                "SELECT d, ARRAY_AGG(id ORDER BY id) FROM t1 GROUP BY d",
                false,
                false,
            ),
            (
                "SELECT d, FIRST_VALUE(a ORDER BY id) FROM t1 GROUP BY d",
                false,
                false,
            ),
            // (DataFusion plans a single `DISTINCT` aggregate, or several of the
            // same argument with `SUM`, `MIN` or `MAX`, as an aggregation of
            // the aggregation by the group keys and the argument, without
            // `DISTINCT`: these queries keep their `DISTINCT` aggregates.)
            (
                "SELECT d, SUM(DISTINCT b), COUNT(*) FROM t1 GROUP BY d",
                false,
                false,
            ),
            (
                "SELECT d, COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t1 GROUP BY d",
                false,
                false,
            ),
            (
                "SELECT h, ARRAY_AGG(DISTINCT a), COUNT(*) FROM t1 GROUP BY h",
                false,
                false,
            ),
            (
                "SELECT h, AVG(DISTINCT a), MEDIAN(DISTINCT b), MAX(c) FROM t1 GROUP BY h",
                false,
                false,
            ),
            (
                "SELECT d, LAST_VALUE(a ORDER BY id), COUNT(*) FROM t1 GROUP BY d",
                false,
                false,
            ),
            (
                "SELECT d, ARRAY_AGG(a ORDER BY b DESC) FILTER (WHERE e), MAX(a) FROM t1 \
                 GROUP BY d",
                false,
                false,
            ),
            (
                "SELECT d, FIRST_VALUE(a ORDER BY id DESC), SUM(b) FROM t1 GROUP BY d",
                false,
                false,
            ),
            // Functions that are not trusted.
            ("SELECT d, STDDEV(b) FROM t1 GROUP BY d", false, false),
            (
                "SELECT d, COUNT(*), STRING_AGG(h, ',') FROM t1 GROUP BY d",
                false,
                false,
            ),
            // Grouping sets.
            ("SELECT d, e FROM t1 GROUP BY ROLLUP (d, e)", false, false),
            (
                "SELECT d, e, COUNT(*) FROM t1 GROUP BY ROLLUP (d, e)",
                false,
                false,
            ),
            // Keys and arguments of nested types and dictionaries.
            ("SELECT [a, b] AS ab FROM t1 GROUP BY [a, b]", false, false),
            (
                "SELECT arrow_cast(d, 'Dictionary(Int32, Utf8)') AS dd FROM t1 GROUP BY dd",
                false,
                false,
            ),
            ("SELECT d, MIN([a, b]) FROM t1 GROUP BY d", false, false),
        ];
        // (The specs of the aggregations, also of those that `try_lower` leaves
        // to DataFusion for other reasons.)
        for (sql, untrusted, trusted) in queries {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let aggregate = built_aggregation(&plan)?
                .unwrap_or_else(|| panic!("no aggregation: {sql}"));
            let spec = &aggregate.spec;
            for options in &untrusting_sessions {
                assert_eq!(spec.rows_bypass_tables(options), untrusted, "{sql}");
                // As on the base: the rows of the aggregations without aggregates
                // (of the aggregations that may bypass the tables) only.
                assert_eq!(untrusted, spec.bypass && spec.update.is_empty(), "{sql}");
            }
            assert_eq!(spec.rows_bypass_tables(&trusting), trusted, "{sql}");
        }
        // The aggregations of user-defined functions that look like
        // DataFusion's `count` and `sum` (see `lookalike_count`), and replace
        // them, are left to DataFusion (see `updates_like_datafusion`): no
        // aggregation of their plans is lowered. (That their functions are not
        // trusted either, see
        // `trusted_aggregate_functions_are_identified_by_their_implementation_types`.)
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        crate::test_utils::register_test_tables(&ctx)?;
        ctx.register_udaf(super::super::test_support::lookalike_count());
        ctx.register_udaf(super::super::test_support::lookalike_sum());
        for sql in [
            "SELECT a, count(b) FROM t1 GROUP BY a",
            "SELECT a, count(*) FROM t1 GROUP BY a",
            "SELECT a, MAX(b), sum(b) FROM t1 GROUP BY a",
        ] {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            assert!(lowered_aggregations(&plan)?.is_empty(), "{sql}");
        }
        // Narrow fixed-size binaries only.
        assert!(is_flat_type(&DataType::FixedSizeBinary(16)));
        assert!(is_flat_type(&DataType::FixedSizeBinary(
            MAX_FIXED_SIZE_BINARY
        )));
        assert!(!is_flat_type(&DataType::FixedSizeBinary(1_100_000)));
        // Nor do the rows of a top-k aggregation (of a single partition).
        let ctx = crate::test_utils::datafusion_context(
            test_session_config().with_target_partitions(1),
        );
        crate::test_utils::register_test_tables(&ctx)?;
        let sql = "SELECT DISTINCT a FROM t1 ORDER BY a LIMIT 3";
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let aggregate =
            try_lower(&top_k_aggregation(&plan).expect("a top-k aggregation"))?
                .expect("a lowered aggregation");
        assert!(aggregate.spec.top_k.is_some() && !aggregate.spec.bypass);
        Ok(())
    }

    #[test]
    fn trusted_aggregate_functions_are_identified_by_their_implementation_types()
    -> Result<()> {
        use datafusion::functions_aggregate::{
            approx_distinct::ApproxDistinct, array_agg::ArrayAgg, average::Avg,
            bool_and_or::BoolAnd, bool_and_or::BoolOr, count::Count,
            first_last::FirstValue, first_last::LastValue, median::Median, min_max::Max,
            min_max::Min, sum::Sum,
        };

        use super::super::test_support::{lookalike_count, lookalike_sum};
        // Of the built-in aggregate functions that a session registers by
        // default, exactly those of `BUILTIN_AGGREGATES` are trusted.
        let functions =
            datafusion::functions_aggregate::all_default_aggregate_functions();
        let trusted = PhysicalHashAggregate::trusted_aggregate_functions(&functions);
        let mut names = trusted.names();
        names.sort_unstable();
        let mut builtins = BUILTIN_AGGREGATES.map(|(name, _)| name);
        builtins.sort_unstable();
        assert_eq!(names, builtins);
        // By the types of `datafusion-functions-aggregate` that implement them.
        let function = |name: &str| {
            functions
                .iter()
                .find(|function| function.name() == name)
                .unwrap_or_else(|| panic!("no function {name}"))
        };
        for (name, implementation) in [
            ("count", TypeId::of::<Count>()),
            ("sum", TypeId::of::<Sum>()),
            ("min", TypeId::of::<Min>()),
            ("max", TypeId::of::<Max>()),
            ("avg", TypeId::of::<Avg>()),
            ("first_value", TypeId::of::<FirstValue>()),
            ("last_value", TypeId::of::<LastValue>()),
            ("bool_and", TypeId::of::<BoolAnd>()),
            ("bool_or", TypeId::of::<BoolOr>()),
            ("approx_distinct", TypeId::of::<ApproxDistinct>()),
            ("median", TypeId::of::<Median>()),
            ("array_agg", TypeId::of::<ArrayAgg>()),
        ] {
            assert_eq!(
                implementation_type(function(name)),
                implementation,
                "{name}"
            );
            assert!(trusted.implementations.contains(&(name, implementation)));
        }
        // `bit_and`, `bit_or` and `bit_xor` share a type (private to the crate),
        // which no other function has: their names tell them apart.
        let bitwise = ["bit_and", "bit_or", "bit_xor"].map(|name| {
            let implementation = implementation_type(function(name));
            assert!(trusted.implementations.contains(&(name, implementation)));
            implementation
        });
        assert!(
            bitwise
                .iter()
                .all(|&implementation| implementation == bitwise[0])
        );
        let types = trusted
            .implementations
            .iter()
            .map(|&(_, implementation)| implementation)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(types.len(), BUILTIN_AGGREGATES.len() - 2);

        // Functions that look like DataFusion's `count` and `sum` in every way
        // but their types: of the same names, implemented by types of the same
        // names, whose `Debug` output is the built-in functions'.
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let aggregate = |function: Arc<AggregateUDF>| {
            AggregateExprBuilder::new(function, vec![col("v", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("aggregate")
                .build()
        };
        let lookalikes = [lookalike_count(), lookalike_sum()].map(Arc::new);
        for (lookalike, builtin) in lookalikes.iter().zip([count_udaf(), sum_udaf()]) {
            assert_eq!(lookalike.name(), builtin.name());
            assert_eq!(
                format!("{:?}", lookalike.inner()),
                format!("{:?}", builtin.inner())
            );
            assert_ne!(
                implementation_type(lookalike),
                implementation_type(&builtin)
            );
            // In a session that trusts the built-in functions, the aggregates of
            // the built-in function are trusted, those of the lookalike are not.
            assert!(trusted.trusts(&aggregate(Arc::clone(&builtin))?));
            assert!(!trusted.trusts(&aggregate(Arc::clone(lookalike))?));
        }
        // Nor are they trusted if they are given as built-in functions, on
        // their own or with the built-in functions.
        let given = PhysicalHashAggregate::trusted_aggregate_functions(&lookalikes);
        assert!(given.names().is_empty());
        for aggregate in lookalikes
            .iter()
            .map(|lookalike| aggregate(Arc::clone(lookalike)))
        {
            assert!(!given.trusts(&aggregate?));
        }
        let mut all = lookalikes.to_vec();
        all.extend(functions.iter().cloned());
        let given = PhysicalHashAggregate::trusted_aggregate_functions(&all);
        let mut names = given.names();
        names.sort_unstable();
        assert_eq!(names, builtins);
        for aggregate in lookalikes
            .iter()
            .map(|lookalike| aggregate(Arc::clone(lookalike)))
        {
            assert!(!given.trusts(&aggregate?));
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregates_split_exactly_if_their_values_do_not_depend_on_it() -> Result<()>
    {
        use super::super::test_support::{lookalike_count, lookalike_sum};
        // Whether the values of the aggregates of an aggregation are the same
        // however the rows of a group are split into partial states (see
        // `splits_exactly`): `c` is a `DOUBLE`, `g` a decimal, `a` an `INT`, `b`
        // and `id` `BIGINT`s.
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        crate::test_utils::register_test_tables(&ctx)?;
        let lookalikes = crate::test_utils::datafusion_context(test_session_config());
        crate::test_utils::register_test_tables(&lookalikes)?;
        lookalikes.register_udaf(lookalike_count());
        lookalikes.register_udaf(lookalike_sum());
        for (ctx, sql, exact) in [
            // Counts, extremes (floating point values by their total order),
            // bits, sketches, medians, and integer and decimal sums.
            (
                &ctx,
                "SELECT d, COUNT(*), COUNT(DISTINCT b), MIN(c), MAX(c), MIN(h), \
                 BIT_XOR(a), BOOL_AND(e), APPROX_DISTINCT(c), MEDIAN(c), SUM(a), \
                 SUM(g), AVG(g), SUM(DISTINCT b), COUNT(c) FILTER (WHERE e) \
                 FROM t1 GROUP BY d",
                true,
            ),
            (&ctx, "SELECT DISTINCT a, d FROM t1", true),
            // The averages of the 5,000 `INT`s of `t1`, whose `Float64` sums are
            // exact (see `sums_exactly_as_floats`), also when DataFusion casts
            // them in a projection below the aggregation (`__common_expr_1`).
            (&ctx, "SELECT d, AVG(a), SUM(a) FROM t1 GROUP BY d", true),
            (
                &ctx,
                "SELECT d, SUM(a), AVG(a), MEDIAN(a), APPROX_DISTINCT(a) FROM t1 GROUP BY d",
                true,
            ),
            // Ordered aggregates of their own order (see
            // `orders_its_arguments`).
            (
                &ctx,
                "SELECT d, ARRAY_AGG(id ORDER BY id), FIRST_VALUE(a ORDER BY a), \
                 LAST_VALUE(h ORDER BY h DESC), NTH_VALUE(id, 2 ORDER BY b, id), \
                 ARRAY_AGG(DISTINCT a ORDER BY a) FROM t1 GROUP BY d",
                true,
            ),
            // Sums of floating point values (also the averages of integers,
            // which DataFusion adds up as floating point values), first and
            // last values, arrays, and other functions.
            (&ctx, "SELECT d, COUNT(*), SUM(c) FROM t1 GROUP BY d", false),
            (&ctx, "SELECT d, AVG(c) FROM t1 GROUP BY d", false),
            (&ctx, "SELECT d, MAX(a), AVG(b) FROM t1 GROUP BY d", false),
            (&ctx, "SELECT d, FIRST_VALUE(a) FROM t1 GROUP BY d", false),
            (
                &ctx,
                "SELECT d, LAST_VALUE(a ORDER BY id) FROM t1 GROUP BY d",
                false,
            ),
            (&ctx, "SELECT d, ARRAY_AGG(a) FROM t1 GROUP BY d", false),
            (&ctx, "SELECT d, STDDEV(b) FROM t1 GROUP BY d", false),
            (
                &ctx,
                "SELECT d, STRING_AGG(h, ',') FROM t1 GROUP BY d",
                false,
            ),
        ] {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let aggregate = built_aggregation(&plan)?.unwrap_or_else(|| panic!("{sql}"));
            assert_eq!(aggregate.spec.splits_exactly, exact, "{sql}");
            // `try_lower` lowers the aggregations of the 4 partitions of `t1` if
            // they split exactly, or are ordered (whose values the native
            // operator orders itself, see `try_lower`).
            let lowered = exact || sql.contains("ORDER BY");
            assert_eq!(
                lowered_aggregations(&plan)?.len(),
                usize::from(lowered),
                "{sql}"
            );
        }
        // The bounds of the exact sums of casts into `Float64`: of up to 2^38
        // `SMALLINT`s and 2^22 `INT`s, of no `BIGINT`s, and of a known number of
        // rows only; also through a projection.
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Int16, true),
            Field::new("i", DataType::Int32, true),
            Field::new("l", DataType::Int64, true),
            Field::new("u", DataType::UInt8, true),
            Field::new("f", DataType::Float64, true),
        ]));
        let input: Arc<dyn ExecutionPlan> = Arc::new(
            datafusion_physical_plan::empty::EmptyExec::new(Arc::clone(&schema)),
        );
        let cast = |name: &str| -> Result<Arc<dyn PhysicalExpr>> {
            Ok(Arc::new(CastExpr::new(
                col(name, &schema)?,
                DataType::Float64,
                None,
            )))
        };
        for (argument, rows, exact) in [
            (cast("s")?, Some(1 << 38), true),
            (cast("s")?, Some((1 << 38) + 1), false),
            (cast("s")?, None, false),
            (cast("i")?, Some(1 << 22), true),
            (cast("i")?, Some((1 << 22) + 1), false),
            (cast("u")?, Some((1 << 53) / 255), true),
            (cast("u")?, Some((1 << 53) / 255 + 1), false),
            (cast("l")?, Some(1), false),
            (col("f", &schema)?, Some(1), false),
        ] {
            assert_eq!(
                sums_exactly_as_floats(&argument, &input, &|| rows),
                exact,
                "{argument} of {rows:?} rows"
            );
        }
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (cast("s")?, "c".to_string()),
                (col("f", &schema)?, "f".to_string()),
            ],
            Arc::clone(&input),
        )?);
        let projected = projection.schema();
        let one = || Some(1);
        assert!(sums_exactly_as_floats(
            &col("c", &projected)?,
            &projection,
            &one
        ));
        assert!(!sums_exactly_as_floats(
            &col("f", &projected)?,
            &projection,
            &one
        ));
        // The lookalikes of DataFusion's `count` and `sum` (see
        // `lookalike_count`), which DataFusion's types do not implement, do not
        // split their rows exactly; their aggregations are left to DataFusion
        // (see `updates_like_datafusion`).
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let input: Arc<dyn ExecutionPlan> = Arc::new(
            datafusion_physical_plan::empty::EmptyExec::new(Arc::clone(&schema)),
        );
        for lookalike in [lookalike_count(), lookalike_sum()] {
            let aggregate =
                AggregateExprBuilder::new(Arc::new(lookalike), vec![col("v", &schema)?])
                    .schema(Arc::clone(&schema))
                    .alias("aggregate")
                    .build()?;
            assert!(!splits_exactly(&aggregate, &input, &|| Some(1)));
        }
        for sql in [
            "SELECT d, count(a) FROM t1 GROUP BY d",
            "SELECT d, sum(a) FROM t1 GROUP BY d",
        ] {
            let plan = lookalikes.sql(sql).await?.create_physical_plan().await?;
            assert!(lowered_aggregations(&plan)?.is_empty(), "{sql}");
        }
        Ok(())
    }

    #[test]
    fn distinct_and_ordered_aggregates_are_not_trusted() -> Result<()> {
        // Every built-in function that a session may trust, plain (trusted),
        // `DISTINCT` and ordered (not trusted). DataFusion drops the `ORDER BY`
        // of the aggregates of the functions whose values do not depend on the
        // order of their rows: their ordered aggregates are not ordered (and
        // are trusted), those of `FIRST_VALUE`, `LAST_VALUE` and `ARRAY_AGG`
        // are.
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, true),
            Field::new("b", DataType::Boolean, true),
        ]));
        let functions =
            datafusion::functions_aggregate::all_default_aggregate_functions();
        let trusted = PhysicalHashAggregate::trusted_aggregate_functions(&functions);
        for (name, _) in BUILTIN_AGGREGATES {
            let function = functions
                .iter()
                .find(|function| function.name() == name)
                .expect("a built-in function");
            let argument = if name.starts_with("bool_") { "b" } else { "v" };
            let aggregate = |distinct: bool, order_by: Vec<PhysicalSortExpr>| {
                AggregateExprBuilder::new(
                    Arc::clone(function),
                    vec![col(argument, &schema)?],
                )
                .schema(Arc::clone(&schema))
                .alias(name)
                .with_distinct(distinct)
                .order_by(order_by)
                .build()
            };
            let by_v = || vec![PhysicalSortExpr::new_default(col("v", &schema).unwrap())];
            assert!(trusted.trusts(&aggregate(false, vec![])?), "{name}");
            assert!(!trusted.trusts(&aggregate(true, vec![])?), "{name}");
            assert!(!trusted.trusts(&aggregate(true, by_v())?), "{name}");
            let ordered = aggregate(false, by_v())?;
            let sensitive = ["first_value", "last_value", "array_agg"].contains(&name);
            assert_eq!(ordered.order_bys().is_empty(), !sensitive, "{name}");
            assert_eq!(trusted.trusts(&ordered), !sensitive, "{name}");
        }
        Ok(())
    }

    /// A benchmark of top-k aggregations, not a test: the pipeline engine and
    /// DataFusion, interleaved, over `TOP_K_ROWS` (10,500,000) rows of a
    /// single partition of 8,192-row batches, `TOP_K_ROUNDS` (5) times, on 1
    /// target partition: `shuffled` keys `k` and values `v` of linear
    /// congruential generators (`(i * 1664525 + 1013904223) & 0xffffffff`),
    /// and `sorted` ones (`i`), whose best rows come first for ascending
    /// orders (and last for descending ones), with 1,000 groups `g` of
    /// `k % 1000`. Prints the median and best times of each engine, and
    /// whether the pipeline engine lowered the top-k aggregation. Run it with
    /// `cargo test --profile release-nonlto -p datafusion-physical-pipeline
    /// --lib top_k_aggregations_benchmark -- --ignored --nocapture`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "a benchmark"]
    async fn top_k_aggregations_benchmark() -> Result<()> {
        use crate::test_utils::{
            canonical_batch, datafusion_context, pipeline_context, test_session_config,
        };
        use arrow::array::{ArrayRef, Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion_common::instant::Instant;
        use datafusion_physical_plan::displayable;
        use std::sync::Arc;
        use std::time::Duration;
        let env = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        };
        let (rows, rounds) = (env("TOP_K_ROWS", 10_500_000), env("TOP_K_ROUNDS", 5));
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("g", DataType::Int64, false),
        ]));
        let table = |key: fn(i64) -> i64,
                     value: fn(i64) -> i64|
         -> Result<Arc<MemTable>> {
            let batches = (0..rows as i64)
                .step_by(8_192)
                .map(|start| {
                    let rows = start..(start + 8_192).min(rows as i64);
                    let columns: Vec<ArrayRef> = vec![
                        Arc::new(Int64Array::from_iter_values(rows.clone().map(key))),
                        Arc::new(Int64Array::from_iter_values(rows.clone().map(value))),
                        Arc::new(Int64Array::from_iter_values(
                            rows.map(|i| key(i) % 1_000),
                        )),
                    ];
                    RecordBatch::try_new(Arc::clone(&schema), columns)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![batches],
            )?))
        };
        let shuffled = table(
            |i| (i.wrapping_mul(1_664_525) + 1_013_904_223) & 0xffff_ffff,
            |i| (i.wrapping_mul(22_695_477) + 1) & 0xffff_ffff,
        )?;
        let sorted = table(|i| i, |i| i)?;
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(8_192);
        let engines = [
            ("pipeline", pipeline_context(config.clone())),
            ("datafusion", datafusion_context(config)),
        ];
        for (_, ctx) in &engines {
            ctx.register_table("shuffled", Arc::clone(&shuffled) as _)?;
            ctx.register_table("sorted", Arc::clone(&sorted) as _)?;
        }
        let queries = [
            ("shuffled", "SELECT DISTINCT k FROM {t} ORDER BY k LIMIT 10"),
            (
                "shuffled",
                "SELECT DISTINCT k FROM {t} ORDER BY k DESC LIMIT 10",
            ),
            (
                "shuffled",
                "SELECT DISTINCT k FROM {t} ORDER BY k LIMIT 1000",
            ),
            // Limits near and above the capacity of a table (32,768 groups).
            (
                "shuffled",
                "SELECT DISTINCT k FROM {t} ORDER BY k LIMIT 32768",
            ),
            (
                "shuffled",
                "SELECT DISTINCT k FROM {t} ORDER BY k LIMIT 32769",
            ),
            (
                "shuffled",
                "SELECT DISTINCT k FROM {t} ORDER BY k LIMIT 40000",
            ),
            (
                "shuffled",
                "SELECT DISTINCT k FROM {t} ORDER BY k DESC LIMIT 65536",
            ),
            (
                "shuffled",
                "SELECT k, MIN(v) FROM {t} GROUP BY k ORDER BY MIN(v) LIMIT 10",
            ),
            (
                "shuffled",
                "SELECT k, MAX(v) FROM {t} GROUP BY k ORDER BY MAX(v) DESC LIMIT 10",
            ),
            (
                "shuffled",
                "SELECT g, MAX(v) FROM {t} GROUP BY g ORDER BY MAX(v) DESC LIMIT 10",
            ),
            ("sorted", "SELECT DISTINCT k FROM {t} ORDER BY k LIMIT 10"),
            (
                "sorted",
                "SELECT DISTINCT k FROM {t} ORDER BY k DESC LIMIT 10",
            ),
            (
                "sorted",
                "SELECT k, MIN(v) FROM {t} GROUP BY k ORDER BY MIN(v) LIMIT 10",
            ),
            (
                "sorted",
                "SELECT k, MAX(v) FROM {t} GROUP BY k ORDER BY MAX(v) DESC LIMIT 10",
            ),
        ];
        let median = |times: &mut Vec<Duration>| {
            times.sort_unstable();
            (times[times.len() / 2], times[0])
        };
        for (table, query) in queries {
            let sql = query.replace("{t}", table);
            let plan = engines[0].1.sql(&sql).await?.create_physical_plan().await?;
            let native = displayable(plan.as_ref()).indent(true).to_string();
            // DataFusion's top-k aggregations show their limit too.
            let native = native.contains("], order=");
            let mut times = vec![vec![]; engines.len()];
            let mut outputs = vec![];
            for round in 0..rounds {
                for (engine, (_, ctx)) in engines.iter().enumerate() {
                    let start = Instant::now();
                    let batches = ctx.sql(&sql).await?.collect().await?;
                    times[engine].push(start.elapsed());
                    if round == 0 {
                        let schema = batches[0].schema();
                        outputs.push(canonical_batch(&schema, &batches, true)?);
                    }
                }
            }
            assert_eq!(outputs[0], outputs[1], "{sql}");
            let [(pipeline, pipeline_best), (datafusion, datafusion_best)] =
                [median(&mut times[0]), median(&mut times[1])];
            println!(
                "top-k {sql:<72} native={native:<5} pipeline {:>8.1} ms (best {:>8.1}) \
                 datafusion {:>8.1} ms (best {:>8.1})",
                pipeline.as_secs_f64() * 1e3,
                pipeline_best.as_secs_f64() * 1e3,
                datafusion.as_secs_f64() * 1e3,
                datafusion_best.as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }
}
