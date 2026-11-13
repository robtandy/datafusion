// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalHashAggregate`]: the pipeline breaker that replaces DataFusion's
//! `AggregateExec`s, see the [module documentation](super).

use std::any::{Any, TypeId};
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, Scalar, UInt32Array,
    make_comparator, new_empty_array, new_null_array,
};
use arrow::compute::kernels::cmp::{gt_eq, lt_eq};
use arrow::compute::{
    SortColumn, SortOptions, concat, filter_record_batch, interleave, lexsort_to_indices,
    sort_to_indices, take, take_record_batch,
};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion_common::{Result, ScalarValue, config_err, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::AggregateUDF;
use datafusion_functions_aggregate::{
    approx_distinct, array_agg, average, bit_and_or_xor, bool_and_or, count, first_last,
    median, min_max, sum,
};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_expr_common::utils::scatter;
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, GroupClusteringMode, PhysicalGroupBy,
};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use super::dynamic_filter::{AggregateFilter, filters_enabled};
use super::radix_ht::{RadixHashAggregate, TableCapacity};
use super::ungrouped::UngroupedAggregate;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
};
use crate::sort::{
    KERNEL_SORT_BYTES_PER_ROW, ReservedMemory, buffered_memory, compact_to_keep,
    memory_size,
};

/// A hash aggregation: a pipeline breaker that consumes the rows of its input
/// in parallel and produces one row per group (DuckDB's
/// `PhysicalHashAggregate`).
///
/// It computes what DataFusion computes with a single `AggregateExec`
/// (`Single` / `SinglePartitioned`) or with the two-stage
/// `AggregateExec(Partial) -> RepartitionExec(Hash) / CoalescePartitionsExec
/// -> AggregateExec(Final / FinalPartitioned)` chain, using the expressions,
/// accumulators and schema of these nodes, without shuffling the partial
/// states through channels. See the [module documentation](super) for how
/// the work is split between the threads, and for the dynamic filters that
/// it updates while it aggregates its input.
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
    /// The soft limit of an aggregation without aggregates (DataFusion's
    /// `lim=[n]` without an order, of `SELECT DISTINCT ... LIMIT n`): the
    /// limit above it returns any `n` of its groups, so a thread stops
    /// consuming its input once its table holds `n` groups.
    pub(crate) soft_limit: Option<usize>,
    /// Whether the input rows of a thread whose groups hardly repeat bypass
    /// its thread-local table, and are aggregated by the final aggregation of
    /// their radix partitions directly (see [`rows_can_bypass_tables`]).
    pub(crate) bypass: bool,
    /// The dynamic filter that the aggregation updates while it aggregates
    /// its input, which nodes of its input read (see [`super::dynamic_filter`]):
    /// the filter of DataFusion's `AggregateExec` of the bounds of its `MIN` and
    /// `MAX` aggregates, or a filter that the aggregation published.
    pub(crate) dynamic_filter: Option<AggregateFilter>,
}

impl AggregateSpec {
    /// Whether there is no `GROUP BY` (and no grouping sets): the
    /// aggregation produces exactly one row.
    pub(crate) fn is_ungrouped(&self) -> bool {
        self.group_by.is_true_no_grouping()
    }
}

/// The limit of a *top-k aggregation*, which DataFusion's `TopKAggregation`
/// rule sets on the stages of an aggregation of a single group key below a
/// sort with a fetch of `limit` rows (`lim=[n]` with an order), by its only
/// sort key:
///
/// * the group key of an aggregation without aggregates (DataFusion's
///   `TopKDistinct`, e.g. `SELECT DISTINCT k FROM t ORDER BY k LIMIT 10`), see
///   [`TopKOrder::Key`];
/// * its only aggregate, a `MIN` ascending or a `MAX` descending (DataFusion's
///   `TopKMinMax`, e.g. `SELECT k, MAX(v) FROM t GROUP BY k ORDER BY MAX(v)
///   DESC LIMIT 10`), see [`TopKOrder::Aggregate`].
///
/// The sort keeps the `limit` best groups by their *value* (their key, or
/// their aggregate), so the aggregation only needs to produce groups among
/// which are those, with their exact values. The value of a group only gets
/// better as more of its rows come: a key does not change, and the `MIN`
/// (`MAX`) of more rows is smaller (greater). So:
///
/// * of a set of partial states of distinct groups (e.g. the groups of a
///   flush of a thread-local table), only the states of the `limit` best
///   values are needed (see [`Self::select`]): every other group is beaten by
///   `limit` groups, whose final values are at least as good, so that it is
///   not among the best groups (unless it ties with them);
/// * once such a set has `limit` values, the rows whose value is worse than
///   the worst of them, its *cutoff*, cannot make their groups among the best
///   ones (see [`Self::cutoff`] and [`Self::candidates`]).
///
/// The partial state of the best value of a group among the best groups is
/// never dropped, so its final value is exact. Which of the groups that tie
/// with the last of them the aggregation produces is its own.
///
/// The values are compared as the sort compares them: by Arrow's kernels, in
/// the order of `descending`, floating point values by their total order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TopK {
    /// The number of groups the sort keeps.
    pub(crate) limit: usize,
    /// Whether the groups with the greatest values come first.
    pub(crate) descending: bool,
    /// The value that orders the groups.
    pub(crate) order: TopKOrder,
}

/// The value that orders the groups of a top-k aggregation (see [`TopK`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopKOrder {
    /// The group key. The group of the `NULL` key is always produced: the
    /// limit does not tell whether the sort puts it first or last.
    Key,
    /// The only aggregate, a `MIN` (ascending) or `MAX` (descending) of an
    /// argument of the type of the aggregate. A group without values (`NULL`)
    /// comes last: DataFusion's `TopKAggregation` sets such limits with an
    /// order that puts `NULL` first only for arguments that cannot be `NULL`,
    /// whose groups always have a value.
    Aggregate,
}

impl TopK {
    /// The column of the output of the aggregation that holds the values of
    /// its groups: its only group column, or the aggregate after it.
    pub(crate) fn output_column(&self) -> usize {
        match self.order {
            TopKOrder::Key => 0,
            TopKOrder::Aggregate => 1,
        }
    }

    /// The options of the sort of the values: the best first, the `NULL`s
    /// last.
    fn sort_options(&self) -> SortOptions {
        SortOptions::new(self.descending, false)
    }

    /// The rows of `values`, the values of a set of distinct groups (one per
    /// group), whose groups may be among the groups the sort keeps (see
    /// [`TopK`]): the `limit` best non-null values (and, for
    /// [`TopKOrder::Key`], the group of the `NULL` key, or, for
    /// [`TopKOrder::Aggregate`], as many groups without values as the best
    /// values are fewer than `limit`). Returns `None` if these are all rows.
    pub(crate) fn select(&self, values: &dyn Array) -> Result<Option<UInt32Array>> {
        let selected = match self.order {
            TopKOrder::Key => values.len() - values.null_count(),
            TopKOrder::Aggregate => values.len(),
        };
        if selected <= self.limit {
            return Ok(None);
        }
        // With the nulls last, the first `limit` rows are the best values,
        // followed by `NULL`s if there are fewer of them.
        let best = sort_to_indices(values, Some(self.sort_options()), Some(self.limit))?;
        if self.order == TopKOrder::Aggregate {
            return Ok(Some(best));
        }
        let mut indices = best.values().to_vec();
        indices.extend(
            (0..values.len())
                .filter(|&row| values.is_null(row))
                .map(|row| row as u32),
        );
        Ok(Some(UInt32Array::from(indices)))
    }

    /// The *cutoff* of `values`, the values of a set of distinct groups (one
    /// per group): the `limit`-th best of them, if `limit` of them are not
    /// `NULL`. From then on, at least `limit` groups have a value at least as
    /// good as the cutoff, so a row whose value is worse cannot make its group
    /// one of the groups the sort keeps (see [`Self::candidates`]).
    pub(crate) fn cutoff(&self, values: &dyn Array) -> Result<Option<ArrayRef>> {
        let valid = values.len() - values.null_count();
        if self.limit == 0 || valid < self.limit {
            return Ok(None);
        }
        // With the nulls last, the first `limit` rows are non-null.
        let best = sort_to_indices(values, Some(self.sort_options()), Some(self.limit))?;
        Ok(Some(take(values, &best.slice(self.limit - 1, 1), None)?))
    }

    /// Whether `cutoff` is a better cutoff than `other` (see [`Self::cutoff`]),
    /// which rules out more rows: whether it comes before it in the order of
    /// the groups.
    pub(crate) fn is_better(
        &self,
        cutoff: &dyn Array,
        other: &dyn Array,
    ) -> Result<bool> {
        Ok(make_comparator(cutoff, other, self.sort_options())?(0, 0).is_lt())
    }

    /// The rows of `values`, the values of input rows (their group keys, or
    /// the arguments of the aggregate), that may make their groups among the
    /// groups the sort keeps given the cutoff `cutoff` (see [`Self::cutoff`]):
    /// the rows whose value is not worse than the cutoff, and, for
    /// [`TopKOrder::Key`], the rows of the `NULL` key (whose group is always
    /// kept). (The `NULL` argument of an aggregate changes no value.) Returns
    /// `None` if these are all rows.
    pub(crate) fn candidates(
        &self,
        values: &ArrayRef,
        cutoff: &ArrayRef,
    ) -> Result<Option<BooleanArray>> {
        let cutoff = Scalar::new(Arc::clone(cutoff));
        let selected = match self.descending {
            true => gt_eq(values, &cutoff)?,
            false => lt_eq(values, &cutoff)?,
        };
        let selected = match (self.order, selected.nulls()) {
            (TopKOrder::Key, Some(nulls)) => {
                BooleanArray::new(selected.values() | &!nulls.inner(), None)
            }
            (TopKOrder::Aggregate, Some(nulls)) => {
                BooleanArray::new(selected.values() & nulls.inner(), None)
            }
            (_, None) => selected,
        };
        Ok((selected.true_count() < selected.len()).then_some(selected))
    }

    /// The groups of `batch`, the output of the aggregation for a set of
    /// distinct groups, sorted by their values: the best first, those of
    /// `NULL` values last.
    pub(crate) fn sort_groups(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let values = batch.column(self.output_column());
        let indices = sort_to_indices(values, Some(self.sort_options()), None)?;
        Ok(take_record_batch(batch, &indices)?)
    }

    /// The groups of `batch`, the output of the aggregation for a set of
    /// distinct groups, that may be among the groups the sort keeps (see
    /// [`Self::select`]), sorted (see [`Self::sort_groups`]).
    pub(crate) fn select_sorted_groups(
        &self,
        batch: &RecordBatch,
    ) -> Result<RecordBatch> {
        match self.select(batch.column(self.output_column()).as_ref())? {
            // The best values first, then `NULL`s.
            Some(indices) => Ok(take_record_batch(batch, &indices)?),
            None => self.sort_groups(batch),
        }
    }

    /// The cutoff (see [`Self::cutoff`]) of `groups`, a set of distinct
    /// groups sorted by [`Self::sort_groups`]: its `limit`-th value, if it is
    /// not `NULL`.
    pub(crate) fn sorted_cutoff(&self, groups: &RecordBatch) -> Option<ArrayRef> {
        let values = groups.column(self.output_column());
        let valid = values.len() - values.null_count();
        (self.limit > 0 && valid >= self.limit).then(|| values.slice(self.limit - 1, 1))
    }

    /// The groups of `left` and `right`, outputs of the aggregation for two
    /// sets of distinct groups selected and sorted by
    /// [`Self::select_sorted_groups`], that may be among the groups the sort
    /// keeps: [`Self::select_sorted_groups`] of all of them, without
    /// concatenating them or sorting them again. Only the selected groups are
    /// gathered, so that their strings fit into an array (of 32-bit offsets)
    /// whenever the selected groups of either side and the groups of the
    /// result do.
    pub(crate) fn merge_sorted_groups(
        &self,
        left: &RecordBatch,
        right: &RecordBatch,
    ) -> Result<RecordBatch> {
        let column = self.output_column();
        let (left_values, right_values) = (left.column(column), right.column(column));
        // The non-null values of either side come first, the best first, then
        // its `NULL`s.
        let valid = |values: &ArrayRef| values.len() - values.null_count();
        let (left_valid, right_valid) = (valid(left_values), valid(right_values));
        let compare = make_comparator(
            left_values.as_ref(),
            right_values.as_ref(),
            self.sort_options(),
        )?;
        // A merge of the non-null values of both sides, up to `limit` rows.
        let mut rows = vec![];
        let (mut l, mut r) = (0, 0);
        while rows.len() < self.limit && (l < left_valid || r < right_valid) {
            if r == right_valid || (l < left_valid && compare(l, r).is_le()) {
                rows.push((0, l));
                l += 1;
            } else {
                rows.push((1, r));
                r += 1;
            }
        }
        let nulls = (left_valid..left_values.len())
            .map(|row| (0, row))
            .chain((right_valid..right_values.len()).map(|row| (1, row)));
        match self.order {
            TopKOrder::Key => rows.extend(nulls),
            TopKOrder::Aggregate => {
                let room = self.limit.saturating_sub(rows.len());
                rows.extend(nulls.take(room));
            }
        }
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
    /// The memory of `batches` (see [`buffered_memory`]), which is reserved.
    memory: usize,
}

impl InputBuffer {
    /// Buffer `batch`, reserving its memory in `reservation` (and keeping a
    /// copy of its rows if they are slices of larger arrays, see
    /// [`compact_to_keep`]).
    pub(crate) fn push(
        &mut self,
        batch: RecordBatch,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        if batch.num_rows() > 0 {
            let batch = compact_to_keep(batch)?;
            let memory = buffered_memory(&batch);
            reservation.try_grow(memory)?;
            self.memory += memory;
            self.batches.push(batch);
        }
        Ok(())
    }

    /// Take the buffered rows (with schema `schema`), sorted by `order`, in
    /// batches of at most [`DEFAULT_VECTOR_SIZE`] rows.
    ///
    /// `working` takes over the memory of the buffered rows, and reserves what
    /// sorting them allocates before it is allocated: their concatenation, the
    /// sort keys and their order, and the sorted rows, which the returned
    /// batches hold until they are dropped.
    pub(crate) fn take_sorted(
        &mut self,
        schema: &SchemaRef,
        order: &LexOrdering,
        working: &mut ReservedMemory<'_>,
    ) -> Result<Vec<RecordBatch>> {
        let batches = std::mem::take(&mut self.batches);
        working.adopt(std::mem::take(&mut self.memory));
        if batches.is_empty() {
            return Ok(vec![]);
        }
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        let rows = batches.iter().map(memory_size).sum::<usize>();
        // The concatenation and the sorted copy of the rows, the sort keys
        // (the keys of a single value as large as the value for every row, the
        // other ones as large as the rows), and what sorting them allocates.
        let keys = order
            .iter()
            .map(|sort| match sort.expr.downcast_ref::<Literal>() {
                Some(literal) => literal.value().size().saturating_mul(num_rows),
                None => 0,
            })
            .sum::<usize>();
        working.try_grow(3 * rows + keys + num_rows * KERNEL_SORT_BYTES_PER_ROW)?;
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
    /// rows. Every such aggregation is lowered, whatever its aggregates
    /// (user-defined ones included), its expressions (also those that may
    /// fail) and its limits, but for those whose result the native operator
    /// does not compute, which are left to DataFusion (`Ok(None)`, as for any
    /// other node):
    ///
    /// * aggregations of input that is clustered by the group keys
    ///   (`GroupClusteringMode` other than `None`): they produce their groups
    ///   in the order of their input, which the plan above them may rely on
    ///   (e.g. instead of a sort), the native operator in no particular order;
    /// * aggregations whose groups may span several partitions of the input
    ///   of the final (or single) `AggregateExec`, which aggregates every
    ///   partition on its own (the native operator aggregates all rows
    ///   together, which is the same if every partition has its own groups:
    ///   if there is at most one partition, or the partitions are partitioned
    ///   by the group keys);
    /// * aggregations with an aggregate ordered by values that its
    ///   accumulators do not order like the sorts (nested values, see
    ///   `compares_totally`): they would keep other rows than the first ones
    ///   of their order.
    ///
    /// The limits of DataFusion's aggregations only allow them to drop groups
    /// that the limit or the sort above them drops anyway:
    ///
    /// * with a soft limit on its groups (`lim=[n]` without an order, from
    ///   `SELECT DISTINCT ... LIMIT n`), the limit above returns any `n`
    ///   groups: the threads stop consuming their input once one of them has
    ///   `n` groups;
    /// * with a top-k limit (`lim=[n]` with an order, from `ORDER BY ...
    ///   LIMIT n` by the group key, or by a `MIN` / `MAX` aggregate), the
    ///   aggregation produces the top `n` groups of all of its groups, on any
    ///   number of partitions, and the groups that tie with them as it
    ///   chooses (see `TopK`), or all of its groups, among which the sort
    ///   above selects them (e.g. for limits of more than `MAX_TOP_K_LIMIT`
    ///   groups).
    ///
    /// The aggregation updates the dynamic filter of the `MIN` and `MAX`
    /// aggregates that DataFusion's partial `AggregateExec` produces, and it
    /// publishes a dynamic filter of its own into its input (before it is
    /// lowered) if its state can prune its input but DataFusion produces no
    /// filter: for a single stage aggregation of `MIN` and `MAX` aggregates
    /// without `GROUP BY`, and for a top-k aggregation (see the module
    /// `dynamic_filter`).
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
        if !groups_are_partition_local(last.unwrap_or(raw)) {
            return Ok(None);
        }
        if let Some(last) = last
            && !is_final_of(last, raw)
        {
            return Ok(None);
        }
        // DataFusion sorts the rows of ordered aggregates by Arrow's order (the
        // input of a `FIRST_VALUE`, which takes the first row then), the native
        // operator lets their accumulators compare them as `ScalarValue`s (see
        // `without_input_order`), as they do when they merge their states:
        // which compare nested values by another order, and not by a total one
        // (see `compares_totally`, e.g. `{a: NULL}` is equal to `{a: 1}`, and
        // `[NULL]` is after `[1]` whatever the order of the NULLs), so that they
        // would keep a row that is not the first in the order of the aggregate
        // (a wrong result, on any number of threads). Aggregates ordered by such
        // values are left to DataFusion.
        let input_schema = raw.input().schema();
        if raw.aggr_expr().iter().any(|aggregate| {
            aggregate.order_bys().iter().any(|sort| {
                !sort
                    .expr
                    .data_type(&input_schema)
                    .is_ok_and(|data_type| compares_totally(&data_type))
            })
        }) {
            return Ok(None);
        }
        let top_k = match group_limit(raw, last) {
            GroupLimit::TopK(top_k) => Some(top_k),
            GroupLimit::None | GroupLimit::Soft(_) => None,
        };
        let published = AggregateFilter::try_publish(raw, last, top_k)?;
        let input = match &published {
            // The input with the nodes that read the published filter.
            Some((_, input)) => lower_input(input)?,
            None => lower_input(raw.input())?,
        };
        let mut aggregate = match last {
            Some(last) => Self::try_new_two_stage(raw, last, input)?,
            None => Self::try_new_single_stage(raw, input)?,
        };
        if let Some((filter, _)) = published {
            aggregate = aggregate.with_published_filter(filter)?;
        }
        Ok(Some(aggregate))
    }

    /// The aggregation, which updates `filter`, a dynamic filter that it
    /// published into its input (see [`AggregateFilter::try_publish`]).
    fn with_published_filter(mut self, filter: AggregateFilter) -> Result<Self> {
        let Some(spec) = Arc::get_mut(&mut self.spec) else {
            return internal_err!(
                "The aggregation is shared: it cannot publish a filter"
            );
        };
        if spec.dynamic_filter.is_some() {
            return internal_err!("The aggregation updates the filters of DataFusion");
        }
        spec.dynamic_filter = Some(filter);
        Ok(self)
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
        let (top_k, soft_limit) = match group_limit(raw, last) {
            GroupLimit::None => (None, None),
            GroupLimit::Soft(limit) => (None, Some(limit)),
            GroupLimit::TopK(top_k) => (Some(top_k), None),
        };
        // The tables of limited aggregations hold their groups (the groups
        // that a soft limit counts, the best groups of a top-k aggregation).
        let bypass = top_k.is_none()
            && soft_limit.is_none()
            && input_order.is_none()
            && rows_can_bypass_tables(&group_by, &update, &merge, &input_schema);
        let dynamic_filter = AggregateFilter::of_aggregate_exec(raw)?;
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
                soft_limit,
                bypass,
                dynamic_filter,
            }),
        })
    }

    /// The configuration extension of a session that trusts `functions`, the
    /// built-in aggregate functions of DataFusion (e.g.
    /// `datafusion::functions_aggregate::all_default_aggregate_functions()`):
    /// `count`, `sum`, `min`, `max`, `avg`, `first_value`, `last_value`,
    /// `bool_and`, `bool_or`, `bit_and`, `bit_or`, `bit_xor`,
    /// `approx_distinct`, `median` and `array_agg`, if they are implemented by
    /// DataFusion's types.
    ///
    /// The extension no longer changes how aggregations are executed: the
    /// input rows of the threads whose groups hardly repeat bypass their
    /// thread-local tables whatever the aggregate functions (user-defined ones
    /// included), in every session. It is kept so that the sessions that
    /// register it keep working.
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

    /// Never let the input rows of the threads bypass their thread-local
    /// tables, e.g. to compare the results of both paths.
    #[cfg(test)]
    pub(crate) fn without_bypass(mut self) -> Self {
        Arc::get_mut(&mut self.spec)
            .expect("the spec of a new aggregation")
            .bypass = false;
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

/// Whether no group of `aggregate`, the `AggregateExec` that produces the
/// final values, spans several partitions of its input. DataFusion
/// aggregates every partition on its own, which is the same as aggregating
/// all rows together exactly then: if there is at most a single partition, or
/// the partitions are partitioned by (a subset of) the group keys. (With
/// grouping sets, a single stage aggregation makes groups of the rows of all
/// partitions, e.g. the grand total of a `ROLLUP`.)
fn groups_are_partition_local(aggregate: &AggregateExec) -> bool {
    let input = aggregate.input();
    let partitioning = input.output_partitioning();
    if partitioning.partition_count() <= 1 {
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
    /// Any `n` groups will do, see [`AggregateSpec::soft_limit`].
    Soft(usize),
    /// A top-k aggregation, see [`TopK`].
    TopK(TopK),
}

/// The largest limit of a top-k aggregation that the native operator keeps
/// the top groups of: it merges the top groups of all partitions into a
/// single batch, which must stay small. The aggregations of larger limits
/// produce all of their groups, among which the sort above selects the top
/// groups.
const MAX_TOP_K_LIMIT: usize = 1 << 16;

/// The limit on the groups of the aggregation whose stage that consumes the
/// input rows is `raw`, and whose final stage is `last` (if it has two
/// stages), see [`PhysicalHashAggregate::try_lower`].
///
/// A stage with a limit drops the groups that the limit (or the sort) above
/// the aggregation drops anyway, so may the native operator, which fuses the
/// stages, if their limits agree (the greatest of them, then): its other
/// limits are dropped, and the aggregation produces all of its groups, which
/// is always correct.
fn group_limit(raw: &AggregateExec, last: Option<&AggregateExec>) -> GroupLimit {
    let stages = || [Some(raw), last].into_iter().flatten();
    let mut limits = stages().filter_map(AggregateExec::limit_options);
    let Some(first) = limits.next() else {
        return GroupLimit::None;
    };
    let mut limit = first.limit();
    for other in limits {
        if other.descending() != first.descending() {
            return GroupLimit::None;
        }
        limit = limit.max(other.limit());
    }
    match first.descending() {
        // A soft limit has no order. Only aggregations without aggregates
        // (whose groups are complete with their first row) may stop early.
        None => {
            let distinct = !raw.group_expr().is_true_no_grouping()
                && stages().all(|stage| stage.aggr_expr().is_empty());
            if distinct && limit > 0 {
                GroupLimit::Soft(limit)
            } else {
                GroupLimit::None
            }
        }
        Some(descending) => {
            top_k(raw, last, limit, descending).map_or(GroupLimit::None, GroupLimit::TopK)
        }
    }
}

/// The top-k aggregation of `limit` groups in the order of `descending` of
/// the aggregation whose stages are `raw` and `last` (see [`group_limit`]),
/// if the native operator keeps only its top groups: an aggregation of a
/// single group key (without grouping sets or filters), by that key if it
/// has no aggregates, or by its only aggregate if it is a `MIN` ascending or
/// a `MAX` descending (see [`is_min_max`]), of values whose order is
/// supported (see [`is_top_k_type`]).
fn top_k(
    raw: &AggregateExec,
    last: Option<&AggregateExec>,
    limit: usize,
    descending: bool,
) -> Option<TopK> {
    let group_by = raw.group_expr();
    let [(key, _)] = group_by.expr() else {
        return None;
    };
    if group_by.has_grouping_set()
        || !(1..=MAX_TOP_K_LIMIT).contains(&limit)
        || raw.filter_expr().iter().any(Option::is_some)
    {
        return None;
    }
    let schema = raw.input().schema();
    let (order, value_type) = match raw.aggr_expr() {
        [] => (TopKOrder::Key, key.data_type(&schema).ok()?),
        [aggregate] => {
            let min_max = [Some(raw), last].into_iter().flatten().all(|stage| {
                matches!(stage.aggr_expr(), [aggregate] if is_min_max(aggregate, descending, &schema))
            });
            if !min_max {
                return None;
            }
            (TopKOrder::Aggregate, aggregate.field().data_type().clone())
        }
        _ => return None,
    };
    is_top_k_type(&value_type).then_some(TopK {
        limit,
        descending,
        order,
    })
}

/// Whether `aggregate` (of input rows of schema `schema`) is DataFusion's
/// built-in `MAX` (if `descending`) or `MIN` (identified by the type that
/// implements it), neither `DISTINCT` nor ordered, of a single argument of
/// its own type, whose partial state is its value: an aggregate whose value
/// only gets better in the order `descending` as more rows come, which a
/// top-k aggregation may order its groups by (see [`TopK`]).
fn is_min_max(
    aggregate: &AggregateFunctionExpr,
    descending: bool,
    schema: &SchemaRef,
) -> bool {
    let function = match descending {
        true => TypeId::of::<min_max::Max>(),
        false => TypeId::of::<min_max::Min>(),
    };
    let output = aggregate.field();
    let value_type = output.data_type();
    let arguments = aggregate.expressions();
    let [argument] = arguments.as_slice() else {
        return false;
    };
    implementation_type(aggregate.fun()) == function
        && !aggregate.is_distinct()
        && aggregate.order_bys().is_empty()
        && argument
            .data_type(schema)
            .is_ok_and(|data_type| &data_type == value_type)
        && aggregate.state_fields().is_ok_and(|fields| {
            matches!(fields.as_slice(), [state] if state.data_type() == value_type)
        })
}

/// Whether the native top-k aggregation orders values of `data_type`, by the
/// Arrow kernels that the sort above it orders them by: the types of
/// DataFusion's top-k aggregations, primitive values (floating point values
/// by their total order) and strings.
fn is_top_k_type(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
}

/// Whether the input rows of an aggregation without an input order or a
/// limit bypass the thread-local tables of a thread whose groups hardly
/// repeat (see `RadixLocalState::aggregate`), and are aggregated by the final
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
/// Whatever the aggregates, user-defined, `DISTINCT` and ordered ones
/// included: the final aggregation of a partition updates the accumulators
/// of its groups with the bypassed rows, as the tables would have, and merges
/// them with the partial states of the tables. A bypassed row is aggregated
/// once, rather than in a table first, and its group interned once (as
/// DataFusion skips its partial aggregation). That an accumulator gets its
/// rows in other batches, or fails on other rows first (e.g. a user-defined
/// sum that rejects negative running totals), than in DataFusion depends on
/// the batches and the threads, which are the engine's own.
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

/// The built-in aggregate functions of DataFusion that
/// [`TrustedAggregateFunctions`] names: their name, and the function of
/// `datafusion-functions-aggregate` that returns them.
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
pub(crate) fn implementation_type(udaf: &AggregateUDF) -> TypeId {
    let implementation: &dyn Any = udaf.inner().as_ref();
    implementation.type_id()
}

/// The built-in aggregate functions of DataFusion that a session trusts (see
/// [`BUILTIN_AGGREGATES`]): a configuration extension of the session, created
/// with [`PhysicalHashAggregate::trusted_aggregate_functions`].
///
/// The native aggregation no longer reads it: the input rows of a thread
/// whose groups hardly repeat bypass its thread-local table whatever the
/// aggregate functions (see `rows_can_bypass_tables`). The extension is kept
/// so that the sessions that register it keep working; it has no effect.
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
    /// those of [`BUILTIN_AGGREGATES`], if they are implemented by DataFusion's
    /// types.
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
            description: "The built-in aggregate functions that the session trusts (no longer read by the native aggregation of the pipeline engine)",
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

/// Whether DataFusion compares the `ScalarValue`s of `data_type` by a total
/// order consistent with Arrow's, by which it sorts them: values that are
/// not nested (floating point values by their total order), and dictionaries
/// of them. Nested values are compared field by field, skipping NULLs, or
/// element by element, NULLs last: e.g. `{a: NULL}` compares equal to
/// `{a: 1}` and to `{a: 2}`, which Arrow sorts after both.
fn compares_totally(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(_, values) => compares_totally(values),
        DataType::RunEndEncoded(_, _) => false,
        data_type => !data_type.is_nested(),
    }
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
        if let Some(limit) = self.soft_limit {
            write!(f, ", lim=[{limit}]")?;
        }
        if self.dynamic_filter.is_some() {
            write!(f, ", dynamic_filter=true")?;
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

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        // Whether the threads update the dynamic filter of the aggregation.
        let filters = filters_enabled(ctx);
        let reservation =
            MemoryConsumer::new(self.name()).register(ctx.task_ctx.memory_pool());
        if self.spec.is_ungrouped() {
            let batch_size = ctx.task_ctx.session_config().batch_size();
            UngroupedAggregate::global_sink_state(
                &self.spec,
                filters,
                batch_size,
                reservation,
            )
        } else {
            RadixHashAggregate::global_sink_state(filters, reservation)
        }
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        if self.spec.is_ungrouped() {
            return UngroupedAggregate::local_sink_state(&self.spec, gstate);
        }
        RadixHashAggregate::local_sink_state(
            &self.spec,
            self.table_capacity,
            self.spec.bypass,
        )
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
            UngroupedAggregate::sink(&self.spec, batch, gstate, lstate)?;
            return Ok(SinkResult::NeedMoreInput);
        }
        RadixHashAggregate::sink(&self.spec, batch, gstate, lstate)
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
        // The input is aggregated: the dynamic filter will not change any
        // more.
        if let Some(filter) = &self.spec.dynamic_filter {
            filter.mark_complete();
        }
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
            RadixHashAggregate::global_source_state(&self.spec, sink_state)
        }
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        if self.spec.is_ungrouped() {
            UngroupedAggregate::local_source_state()
        } else {
            RadixHashAggregate::local_source_state(gstate)
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

    use super::super::test_support::{lower, rows_of};
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregations_of_inputs_without_partitions_are_lowered() -> Result<()> {
        // An input without partitions has no rows: an ungrouped aggregation
        // produces its row, a grouping set without group columns its group,
        // as for an empty input. (DataFusion produces no row, or fails to
        // coalesce the partial states of no partition.)
        let rollup = || -> Result<PhysicalGroupBy> {
            Ok(PhysicalGroupBy::new(
                vec![(col("k", &input_schema())?, "k".into())],
                vec![(lit(ScalarValue::Utf8(None)), "k".into())],
                vec![vec![false], vec![true]],
                true,
            ))
        };
        let rolled_up = |mode: AggregateMode, input: Plan| -> Result<Plan> {
            Ok(Arc::new(AggregateExec::try_new(
                mode,
                rollup()?,
                aggregates(&input_schema())?,
                vec![None, None],
                input,
                input_schema(),
            )?))
        };
        let nothing = || vec![ScalarValue::Int64(Some(0)), ScalarValue::Int64(None)];
        let total = || {
            let mut row = vec![ScalarValue::Utf8(None), ScalarValue::UInt8(Some(1))];
            row.extend(nothing());
            row
        };
        let plans: [(&str, Plan, Vec<Vec<ScalarValue>>); 5] = [
            (
                "ungrouped",
                single(AggregateMode::Single, false, input(0, false)?)?,
                vec![nothing()],
            ),
            (
                "ungrouped of two stages",
                last(
                    AggregateMode::Final,
                    false,
                    coalesce(partial(false, input(0, false)?)?),
                )?,
                vec![nothing()],
            ),
            (
                "grouped",
                single(AggregateMode::Single, true, input(0, false)?)?,
                vec![],
            ),
            (
                "grouping sets",
                rolled_up(AggregateMode::Single, input(0, false)?)?,
                vec![total()],
            ),
            (
                "grouping sets of two stages",
                Arc::new(AggregateExec::try_new(
                    AggregateMode::Final,
                    rollup()?.as_final(),
                    aggregates(&input_schema())?,
                    vec![None, None],
                    coalesce(rolled_up(AggregateMode::Partial, input(0, false)?)?),
                    input_schema(),
                )?),
                vec![total()],
            ),
        ];
        for (name, plan, expected) in plans {
            let root = lower(&plan)?;
            assert!(root.is::<PhysicalHashAggregate>(), "{name}");
            assert_eq!(root.schema(), plan.schema(), "{name}");
            for threads in [1, 4] {
                let output = Executor::collect(
                    Arc::clone(&root),
                    Arc::new(TaskContext::default()),
                    threads,
                )
                .await?;
                assert_eq!(rows_of(&output), expected, "{name} on {threads} threads");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregations_with_stale_partitions_aggregate_their_input() -> Result<()> {
        // An aggregation whose input was replaced without recomputing its
        // properties advertises 3 partitions of an input of 1: DataFusion would
        // execute input partitions that do not exist. The native operator
        // aggregates its input.
        let stale = lowered_plans()[2].1()?.replace_children(
            vec![input(1, false)?],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Keep),
        )?;
        assert_eq!(
            stale.properties().output_partitioning().partition_count(),
            3
        );
        let root = lower(&stale)?;
        assert!(root.is::<PhysicalHashAggregate>());
        let task_ctx = Arc::new(TaskContext::default());
        let actual = Executor::collect(root, Arc::clone(&task_ctx), 4).await?;
        let expected = datafusion_physical_plan::collect(
            single(AggregateMode::Single, true, input(1, false)?)?,
            task_ctx,
        )
        .await?;
        let schema = stale.schema();
        assert_eq!(
            canonical_batch(&schema, &actual, false)?,
            canonical_batch(&schema, &expected, false)?
        );
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
    fn limited_aggregations_are_lowered() -> Result<()> {
        let plan = |aggregate: AggregateExec| Arc::new(aggregate) as Plan;
        // The top-k and soft limits of the lowered aggregation.
        let lowered_limits = |plan: &Plan| -> Result<(Option<TopK>, Option<usize>)> {
            let aggregate = try_lower(plan)?.expect("a lowered aggregation");
            Ok((aggregate.spec.top_k, aggregate.spec.soft_limit))
        };
        let top_k = |order: TopKOrder| TopK {
            limit: 3,
            descending: true,
            order,
        };
        for limit in [
            Limit::None,
            Limit::Soft,
            Limit::TopKDistinct,
            Limit::TopKMax,
        ] {
            // The key `k` is a `Utf8` string, the argument `v` of `MAX(v)`
            // nullable.
            let expected = |limited: bool| match (limited, limit) {
                (false, _) | (true, Limit::None) => (None, None),
                (true, Limit::Soft) => (None, Some(3)),
                (true, Limit::TopKDistinct) => (Some(top_k(TopKOrder::Key)), None),
                (true, Limit::TopKMax) => (Some(top_k(TopKOrder::Aggregate)), None),
            };
            for apply in [false, true] {
                let single =
                    limited(AggregateMode::Single, limit, apply, input(1, false)?)?;
                assert_eq!(lowered_limits(&plan(single))?, expected(apply), "{limit:?}");
            }
            // A limit of the partial or of the final stage, or of both, of a
            // final stage of several partitions or of one.
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
                        lowered_limits(&plan(last))?,
                        expected(limit_partial || limit_final),
                        "{limit:?} {mode:?}"
                    );
                }
            }
        }
        // Stages of different limits produce all groups: of other orders, and
        // a soft and a top-k limit. Stages of the same order the greatest
        // limit.
        let by_v = |mode: AggregateMode, input: Plan, limit: Option<(usize, bool)>| {
            let key = col("v", &input.schema())?;
            let aggregate = AggregateExec::try_new(
                mode,
                PhysicalGroupBy::new_single(vec![(key, "v".into())]),
                vec![],
                vec![],
                input,
                input_schema(),
            )?;
            let Some((limit, descending)) = limit else {
                return Ok(Arc::new(aggregate) as Plan);
            };
            let limited = aggregate
                .try_optimize_topk(limit, "v", SortOptions::new(descending, false))
                .expect("a top-k aggregation")
                .data;
            Ok::<_, datafusion_common::DataFusionError>(Arc::new(limited))
        };
        let two_stages = |partial: Option<(usize, bool)>, last: Option<(usize, bool)>| {
            let states = by_v(AggregateMode::Partial, input(4, false)?, partial)?;
            let states = Arc::new(RepartitionExec::try_new(
                Arc::clone(&states),
                Partitioning::Hash(vec![col("v", &states.schema())?], 3),
            )?);
            let plan = by_v(AggregateMode::FinalPartitioned, states, last)?;
            Ok::<_, datafusion_common::DataFusionError>(lowered_limits(&plan)?.0)
        };
        let ascending = |limit: usize| TopK {
            limit,
            descending: false,
            order: TopKOrder::Key,
        };
        assert_eq!(
            two_stages(Some((3, false)), Some((3, false)))?,
            Some(ascending(3))
        );
        assert_eq!(
            two_stages(Some((3, false)), Some((5, false)))?,
            Some(ascending(5))
        );
        assert_eq!(two_stages(Some((3, false)), None)?, Some(ascending(3)));
        assert_eq!(two_stages(Some((3, false)), Some((3, true)))?, None);
        let soft = limited(AggregateMode::Partial, Limit::Soft, true, input(4, false)?)?;
        let topk = limited(
            AggregateMode::Final,
            Limit::TopKDistinct,
            true,
            coalesce(plan(soft)),
        )?;
        assert_eq!(lowered_limits(&plan(topk))?, (None, None));
        Ok(())
    }

    #[test]
    fn top_k_aggregations_of_large_limits_or_other_aggregates_produce_all_groups()
    -> Result<()> {
        // DataFusion's top-k aggregations of more groups than the native
        // operator keeps the top groups of (`MAX_TOP_K_LIMIT`), or by an
        // aggregate that is not DataFusion's `MIN` / `MAX` (see `is_min_max`),
        // are lowered without their limit: they produce all of their groups.
        let schema = input_schema();
        let by_k = |aggregates: Vec<Arc<AggregateFunctionExpr>>,
                    limit: usize,
                    order_by: &str|
         -> Result<Plan> {
            let filters = vec![None; aggregates.len()];
            let aggregate = AggregateExec::try_new(
                AggregateMode::Single,
                group_by(true)?,
                aggregates,
                filters,
                input(1, false)?,
                Arc::clone(&schema),
            )?
            .try_optimize_topk(limit, order_by, SortOptions::new(true, false))
            .expect("a top-k aggregation")
            .data;
            assert!(aggregate.limit_options().is_some());
            Ok(Arc::new(aggregate))
        };
        let max = |function: Arc<AggregateUDF>| -> Result<Arc<AggregateFunctionExpr>> {
            Ok(Arc::new(
                AggregateExprBuilder::new(function, vec![col("v", &schema)?])
                    .schema(Arc::clone(&schema))
                    .alias("max(v)")
                    .build()?,
            ))
        };
        let top_k = |plan: &Plan| -> Result<Option<TopK>> {
            Ok(try_lower(plan)?.expect("a lowered aggregation").spec.top_k)
        };
        assert_eq!(
            top_k(&by_k(vec![max(max_udaf())?], 3, "max(v)")?)?,
            Some(TopK {
                limit: 3,
                descending: true,
                order: TopKOrder::Aggregate,
            })
        );
        // More groups than a top-k aggregation keeps.
        let large = MAX_TOP_K_LIMIT + 1;
        assert_eq!(top_k(&by_k(vec![], large, "k")?)?, None);
        assert_eq!(
            top_k(&by_k(vec![max(max_udaf())?], large, "max(v)")?)?,
            None
        );
        // A user-defined `max` of DataFusion's name, type name and order.
        let lookalike = Arc::new(super::super::test_support::lookalike_max());
        assert_eq!(top_k(&by_k(vec![max(lookalike)?], 3, "max(v)")?)?, None);
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
        let top_k = |limit, descending, order| TopK {
            limit,
            descending,
            order,
        };
        let select = |top_k: TopK| -> Result<Option<Vec<u32>>> {
            Ok(top_k.select(&values)?.map(|indices| {
                let mut indices = indices.values().to_vec();
                indices.sort_unstable();
                indices
            }))
        };
        // By key: the 2 greatest keys (9, 7), the 2 least (1, 3), and the
        // `NULL` keys.
        let key = |limit, descending| top_k(limit, descending, TopKOrder::Key);
        assert_eq!(select(key(2, true))?, Some(vec![1, 2, 4, 5, 7]));
        assert_eq!(select(key(2, false))?, Some(vec![1, 3, 4, 6, 7]));
        assert_eq!(select(key(1, false))?, Some(vec![1, 3, 4, 7]));
        // All rows, if there are not more of them.
        assert_eq!(select(key(5, true))?, None);
        // By aggregate: the best values, followed by groups without values
        // (`NULL`) if there are fewer of them.
        let aggregate =
            |limit, descending| top_k(limit, descending, TopKOrder::Aggregate);
        assert_eq!(select(aggregate(2, true))?, Some(vec![2, 5]));
        assert_eq!(select(aggregate(3, false))?, Some(vec![0, 3, 6]));
        assert_eq!(select(aggregate(5, true))?, Some(vec![0, 2, 3, 5, 6]));
        let selected = select(aggregate(6, true))?.expect("a selection");
        assert_eq!(selected.len(), 6);
        assert!([0, 2, 3, 5, 6].iter().all(|row| selected.contains(row)));
        assert_eq!(select(aggregate(8, true))?, None);
        // Strings, by their bytes.
        let strings = StringArray::from(vec![Some("b"), Some("ab"), None, Some("B")]);
        let indices = key(1, false).select(&strings)?.expect("a selection");
        assert_eq!(indices.values().as_ref(), &[3, 2]);
        let indices = aggregate(1, true).select(&strings)?.expect("a selection");
        assert_eq!(indices.values().as_ref(), &[0]);
        // Floating point values by their total order: `NaN` is the greatest,
        // `-0.0` less than `0.0`.
        let floats = arrow::array::Float64Array::from(vec![
            Some(0.0),
            Some(f64::NAN),
            None,
            Some(-0.0),
            Some(1.5),
        ]);
        let indices = aggregate(2, true).select(&floats)?.expect("a selection");
        assert_eq!(indices.values().as_ref(), &[1, 4]);
        let indices = aggregate(2, false).select(&floats)?.expect("a selection");
        assert_eq!(indices.values().as_ref(), &[3, 0]);
        Ok(())
    }

    #[test]
    fn top_k_candidates_are_the_rows_not_worse_than_the_cutoff() -> Result<()> {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(5),
            None,
            Some(9),
            Some(1),
            Some(7),
        ]));
        let candidates = |descending, order| -> Result<Option<Vec<bool>>> {
            let top_k = TopK {
                limit: 2,
                descending,
                order,
            };
            let cutoff = top_k.cutoff(values.as_ref())?.expect("a cutoff");
            Ok(top_k
                .candidates(&values, &cutoff)?
                .map(|selected| selected.iter().map(|row| row == Some(true)).collect()))
        };
        // The cutoff of the 2 greatest values is 7, of the 2 least 5. The rows of
        // the `NULL` key are always candidates, those of `NULL` arguments never.
        assert_eq!(
            candidates(true, TopKOrder::Key)?,
            Some(vec![false, true, true, false, true])
        );
        assert_eq!(
            candidates(true, TopKOrder::Aggregate)?,
            Some(vec![false, false, true, false, true])
        );
        assert_eq!(
            candidates(false, TopKOrder::Aggregate)?,
            Some(vec![true, false, false, true, false])
        );
        // Fewer values than the limit have no cutoff.
        let top_k = TopK {
            limit: 5,
            descending: true,
            order: TopKOrder::Aggregate,
        };
        assert!(top_k.cutoff(values.as_ref())?.is_none());
        Ok(())
    }

    #[test]
    fn top_k_merges_the_best_groups_of_two_sorted_sets_without_concatenating_them()
    -> Result<()> {
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
        // By the keys `k`, and by the values `m` (as if they were the values of
        // an aggregate).
        let orders = [TopKOrder::Key, TopKOrder::Aggregate];
        for (descending, order) in [false, true].into_iter().flat_map(|descending| {
            orders.into_iter().map(move |order| (descending, order))
        }) {
            for limit in [1, 2, 3, 10] {
                let top_k = TopK {
                    limit,
                    descending,
                    order,
                };
                for (left, right) in [(&left, &right), (&right, &left), (&left, &empty)] {
                    let (left, right) = (
                        top_k.select_sorted_groups(left)?,
                        top_k.select_sorted_groups(right)?,
                    );
                    assert_eq!(top_k.sort_groups(&left)?, left, "{top_k:?}");
                    let merged = top_k.merge_sorted_groups(&left, &right)?;
                    let expected = top_k.select_sorted_groups(&concat_batches(
                        &schema,
                        [&left, &right],
                    )?)?;
                    assert_eq!(rows(&merged), rows(&expected), "{top_k:?}");
                    assert_eq!(merged.schema(), schema);
                    // Sorted, with their `limit`-th value as their cutoff.
                    assert_eq!(top_k.sort_groups(&merged)?, merged, "{top_k:?}");
                    let values = merged.column(top_k.output_column());
                    assert_eq!(
                        top_k.sorted_cutoff(&merged),
                        top_k.cutoff(values.as_ref())?,
                        "{top_k:?}"
                    );
                }
            }
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
    async fn top_k_aggregations_produce_the_results_of_their_queries() -> Result<()> {
        use super::super::test_support::{
            assert_lowered_results_with_setup, without_limited_aggregations,
        };
        let setup = async |ctx: &datafusion::prelude::SessionContext| register_tk(ctx);
        // The queries, and what their native top-k aggregation orders its
        // groups by (their results are unambiguous: there are no ties).
        let (key, aggregate) = (TopKOrder::Key, TopKOrder::Aggregate);
        let queries = [
            // By the group key, of every supported type, in both directions,
            // with the `NULL` key first or last.
            ("SELECT DISTINCT g FROM tk ORDER BY g LIMIT 5", key),
            ("SELECT DISTINCT g FROM tk ORDER BY g DESC LIMIT 5", key),
            (
                "SELECT DISTINCT g FROM tk ORDER BY g NULLS FIRST LIMIT 3",
                key,
            ),
            (
                "SELECT DISTINCT g FROM tk ORDER BY g DESC NULLS LAST LIMIT 7",
                key,
            ),
            ("SELECT g FROM tk GROUP BY g ORDER BY g LIMIT 1", key),
            ("SELECT DISTINCT u FROM tk ORDER BY u DESC LIMIT 4", key),
            ("SELECT DISTINCT dt FROM tk ORDER BY dt LIMIT 6", key),
            (
                "SELECT DISTINCT s FROM tk ORDER BY s DESC NULLS LAST LIMIT 3",
                key,
            ),
            ("SELECT DISTINCT sv FROM tk ORDER BY sv LIMIT 8", key),
            ("SELECT DISTINCT ls FROM tk ORDER BY ls DESC LIMIT 2", key),
            ("SELECT DISTINCT dec FROM tk ORDER BY dec LIMIT 5", key),
            (
                "SELECT DISTINCT t FROM tk ORDER BY t DESC NULLS LAST LIMIT 5",
                key,
            ),
            // More groups than there are.
            ("SELECT DISTINCT g FROM tk ORDER BY g DESC LIMIT 1000", key),
            (
                "SELECT DISTINCT g FROM tk WHERE g > 1000 ORDER BY g LIMIT 3",
                key,
            ),
            // By a `MAX` descending or a `MIN` ascending, of values of every
            // supported type.
            (
                "SELECT g, MAX(id) FROM tk GROUP BY g ORDER BY MAX(id) DESC LIMIT 4",
                aggregate,
            ),
            (
                "SELECT g, MIN(id) FROM tk GROUP BY g ORDER BY MIN(id) LIMIT 5",
                aggregate,
            ),
            (
                "SELECT u, MIN(ndec) FROM tk GROUP BY u ORDER BY MIN(ndec) LIMIT 3",
                aggregate,
            ),
            (
                "SELECT s, MAX(nt) FROM tk GROUP BY s ORDER BY MAX(nt) DESC NULLS LAST \
                 LIMIT 6",
                aggregate,
            ),
            (
                "SELECT ls, MAX(nt) FROM tk GROUP BY ls ORDER BY MAX(nt) DESC LIMIT 6",
                aggregate,
            ),
            (
                "SELECT dt, MIN(nsv) FROM tk GROUP BY dt ORDER BY MIN(nsv) LIMIT 4",
                aggregate,
            ),
            (
                "SELECT g, MAX(ns) FROM tk GROUP BY g ORDER BY MAX(ns) DESC LIMIT 3",
                aggregate,
            ),
            (
                "SELECT g, MAX(nsv) FROM tk GROUP BY g ORDER BY MAX(nsv) DESC LIMIT 1000",
                aggregate,
            ),
            // Of a nullable argument, its groups without values last.
            (
                "SELECT g, MAX(v) FROM tk GROUP BY g ORDER BY MAX(v) DESC NULLS LAST LIMIT 5",
                aggregate,
            ),
            (
                "SELECT g, MIN(dec) FROM tk GROUP BY g ORDER BY MIN(dec) LIMIT 5",
                aggregate,
            ),
            (
                "SELECT g, MAX(m) FROM (SELECT g, MAX(v) AS m FROM tk GROUP BY g \
                 ORDER BY m DESC NULLS LAST LIMIT 1000) GROUP BY g",
                aggregate,
            ),
            // With projections, and aggregated again.
            (
                "SELECT g + 1 AS g1, MAX(id) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 5",
                aggregate,
            ),
            (
                "SELECT COUNT(*), SUM(m), MIN(g) FROM \
                 (SELECT g, MAX(id) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 5)",
                aggregate,
            ),
            (
                "SELECT COUNT(*), MIN(s), MAX(s) FROM \
                 (SELECT DISTINCT s FROM tk ORDER BY s LIMIT 10)",
                key,
            ),
            (
                "SELECT COUNT(*), MIN(sv), MAX(sv) FROM \
                 (SELECT DISTINCT sv FROM tk ORDER BY sv LIMIT 10)",
                key,
            ),
        ];
        for target_partitions in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let reference = without_limited_aggregations(config.clone());
            let ctx = crate::test_utils::datafusion_context(config.clone());
            register_tk(&ctx)?;
            let pipeline = crate::test_utils::pipeline_context(config.clone());
            register_tk(&pipeline)?;
            for (sql, order) in queries {
                // DataFusion plans a top-k aggregation.
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                assert!(top_k_aggregation(&plan).is_some(), "{sql}");
                let root = assert_lowered_results_with_setup(
                    sql,
                    config.clone(),
                    reference.clone(),
                    &setup,
                )
                .await;
                let top_k = native_top_k_aggregations(&root);
                let orders = top_k.iter().map(|top_k| top_k.order).collect::<Vec<_>>();
                assert_eq!(orders, [order], "{sql} on {target_partitions} partitions");
                // So does the planner, but where it leaves the aggregation to
                // DataFusion for reasons of its own (e.g. the coalescing of
                // partial states of strings with 32-bit offsets).
                let plan = pipeline.sql(sql).await?.create_physical_plan().await?;
                let root = plan
                    .downcast_ref::<crate::bridge::PipelineRootExec>()
                    .expect("a pipeline plan")
                    .operator();
                let planned = native_top_k_aggregations(root);
                assert!(
                    planned == top_k || planned.is_empty(),
                    "{sql} on {target_partitions} partitions: {planned:?}"
                );
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_produce_only_their_top_groups() -> Result<()> {
        use super::super::test_support::test_lowering_outputs;
        // The operators above a native top-k aggregation only see its top
        // groups (of all partitions): the keys that do not cast to integers
        // are never among them, on 1 and 4 threads, of a single stage and of
        // two. (Which rows an error is raised for is the engine's own: on
        // several partitions, those of DataFusion's top groups of every
        // partition may not cast.)
        let schema =
            Arc::new(Schema::new(vec![Field::new("k", DataType::Utf8View, true)]));
        let keys = (0..10_000)
            .map(|i| Some(format!("{}{i:05}", if i < 10 { "" } else { "invalid" })))
            .chain([None])
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(arrow::array::StringViewArray::from(keys))],
        )?;
        let setup = async |ctx: &datafusion::prelude::SessionContext| -> Result<()> {
            let table = datafusion::datasource::MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![batch.clone()]],
            )?;
            ctx.register_table("t", Arc::new(table))?;
            Ok(())
        };
        let row = |i: Option<i64>| {
            vec![
                ScalarValue::Int64(i),
                ScalarValue::Utf8View(i.map(|i| format!("{i:05}"))),
            ]
        };
        let queries = [
            (
                "SELECT CAST(k AS BIGINT) AS i, k FROM t GROUP BY k ORDER BY k LIMIT 3",
                vec![row(Some(0)), row(Some(1)), row(Some(2))],
            ),
            (
                "SELECT CAST(k AS BIGINT) AS i, k FROM t GROUP BY k \
                 ORDER BY k NULLS FIRST LIMIT 3",
                vec![row(None), row(Some(0)), row(Some(1))],
            ),
            (
                "SELECT CAST(MIN(k) AS BIGINT) AS i, k FROM t GROUP BY k \
                 ORDER BY MIN(k) LIMIT 3",
                vec![row(Some(0)), row(Some(1)), row(Some(2))],
            ),
        ];
        for threads in [1, 4] {
            let config = test_session_config().with_target_partitions(threads);
            for (sql, expected) in &queries {
                // Through the lowering of the tests, whose native projection
                // evaluates the cast of the output of the native aggregation.
                for (engine, output, root) in
                    test_lowering_outputs(sql, config.clone(), &setup).await
                {
                    assert_eq!(native_top_k_aggregations(&root).len(), 1, "{sql}");
                    assert_eq!(
                        &rows_of(&output.batches),
                        expected,
                        "{engine} on {threads} threads: {sql}"
                    );
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_break_ties_as_they_choose() -> Result<()> {
        use super::super::test_support::{assert_top_rows, lowered_outputs};
        // Of the groups that tie with the last of the top groups, a top-k
        // aggregation by a `MIN` / `MAX` keeps any: its query returns the best
        // values, of groups of its rows.
        let setup = async |ctx: &datafusion::prelude::SessionContext| register_tkn(ctx);
        let queries = [
            // The query, the query of all of its rows, the column of its value,
            // whether its order is descending, and its limit.
            (
                "SELECT a, MAX(b) AS m FROM (VALUES (1, 1), (2, 1), (3, 1)) AS t(a, b) \
                 GROUP BY a ORDER BY m DESC LIMIT 1",
                "SELECT a, MAX(b) AS m FROM (VALUES (1, 1), (2, 1), (3, 1)) AS t(a, b) \
                 GROUP BY a",
                true,
                1,
            ),
            (
                "SELECT a, MIN(b) AS m FROM (VALUES (3, 1), (1, 1), (2, 1)) AS t(a, b) \
                 GROUP BY a ORDER BY m LIMIT 1",
                "SELECT a, MIN(b) AS m FROM (VALUES (3, 1), (1, 1), (2, 1)) AS t(a, b) \
                 GROUP BY a",
                false,
                1,
            ),
            // Ties behind a better group.
            (
                "SELECT a, MAX(b) AS m \
                 FROM (VALUES (1, 5), (2, 3), (3, 3), (4, 3), (5, 1)) AS t(a, b) \
                 GROUP BY a ORDER BY m DESC LIMIT 2",
                "SELECT a, MAX(b) AS m \
                 FROM (VALUES (1, 5), (2, 3), (3, 3), (4, 3), (5, 1)) AS t(a, b) \
                 GROUP BY a",
                true,
                2,
            ),
            // Groups that drop out and come back with a value that ties.
            (
                "SELECT a, MAX(b) AS m \
                 FROM (VALUES (1, 1), (2, 2), (3, 1), (2, 1), (1, 2), (3, 2)) AS t(a, b) \
                 GROUP BY a ORDER BY m DESC LIMIT 1",
                "SELECT a, MAX(b) AS m \
                 FROM (VALUES (1, 1), (2, 2), (3, 1), (2, 1), (1, 2), (3, 2)) AS t(a, b) \
                 GROUP BY a",
                true,
                1,
            ),
            // 40,000 groups of 1 or 2 rows, and 13 values: most groups tie.
            (
                "SELECT g, MAX(v % 7) AS m FROM tkn GROUP BY g ORDER BY m DESC LIMIT 10",
                "SELECT g, MAX(v % 7) AS m FROM tkn GROUP BY g",
                true,
                10,
            ),
            (
                "SELECT g, MIN(v % 7) AS m FROM tkn GROUP BY g ORDER BY m LIMIT 1000",
                "SELECT g, MIN(v % 7) AS m FROM tkn GROUP BY g",
                false,
                1000,
            ),
            // Groups without values, which tie as the last ones: there are only
            // 4 groups of `g < 5` with values.
            (
                "SELECT g, MAX(k) AS m FROM tkn WHERE g < 5 OR k IS NULL GROUP BY g \
                 ORDER BY m DESC NULLS LAST LIMIT 8",
                "SELECT g, MAX(k) AS m FROM tkn WHERE g < 5 OR k IS NULL GROUP BY g",
                true,
                8,
            ),
        ];
        for target_partitions in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = crate::test_utils::datafusion_context(config.clone());
            register_tkn(&ctx)?;
            for (sql, all, descending, limit) in queries {
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                assert!(top_k_aggregation(&plan).is_some(), "{sql}");
                let all = ctx.sql(all).await?.collect().await?;
                for (engine, output, root) in
                    lowered_outputs(sql, config.clone(), &setup).await
                {
                    let context =
                        format!("{engine} on {target_partitions} partitions: {sql}");
                    if engine != "planner" {
                        let top_k = native_top_k_aggregations(&root);
                        assert_eq!(top_k.len(), 1, "{context}");
                        assert_eq!(top_k[0].order, TopKOrder::Aggregate, "{context}");
                    }
                    assert_top_rows(
                        &output.batches,
                        &all,
                        1,
                        descending,
                        limit,
                        &context,
                    );
                }
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
    async fn top_k_aggregations_of_many_groups_produce_the_results_of_their_queries()
    -> Result<()> {
        use super::super::test_support::{
            assert_lowered_results_with_setup, without_limited_aggregations,
        };
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
            // By a `MAX` or a `MIN`, of groups of 1 or 2 rows, whose values are
            // shuffled, or come best (or worst) first.
            "SELECT g, MAX(v) FROM tkn GROUP BY g ORDER BY MAX(v) DESC LIMIT 10",
            "SELECT g, MIN(v) FROM tkn GROUP BY g ORDER BY MIN(v) LIMIT 10",
            "SELECT k, MIN(asc) FROM tkn GROUP BY k ORDER BY MIN(asc) LIMIT 5",
            "SELECT k, MAX(asc) FROM tkn GROUP BY k ORDER BY MAX(asc) DESC LIMIT 5",
            "SELECT sv, MIN(v) FROM tkn GROUP BY sv ORDER BY MIN(v) LIMIT 4",
            "SELECT g, MAX(sv) FROM tkn GROUP BY g ORDER BY MAX(sv) DESC LIMIT 3",
            // Fewer groups than a table holds.
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
            "SELECT k, MIN(v) FROM tkn GROUP BY k ORDER BY MIN(v) LIMIT 30000",
        ];
        for target_partitions in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = crate::test_utils::datafusion_context(config.clone());
            register_tkn(&ctx)?;
            for sql in queries {
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                assert!(top_k_aggregation(&plan).is_some(), "{sql}");
                let root = assert_lowered_results_with_setup(
                    sql,
                    config.clone(),
                    without_limited_aggregations(config.clone()),
                    &setup,
                )
                .await;
                let top_k = native_top_k_aggregations(&root);
                let by_key = !sql.contains("MIN(") && !sql.contains("MAX(");
                let order = if by_key {
                    TopKOrder::Key
                } else {
                    TopKOrder::Aggregate
                };
                let orders = top_k.iter().map(|top_k| top_k.order).collect::<Vec<_>>();
                assert_eq!(orders, [order], "{sql} on {target_partitions} partitions");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "allocates more than 8 GiB"]
    async fn top_k_aggregations_of_huge_strings_produce_the_results_of_their_queries()
    -> Result<()> {
        use arrow::array::LargeStringArray;
        use arrow::buffer::{Buffer, OffsetBuffer};

        use super::super::test_support::assert_lowered_parity_with_setup;
        // Two distinct keys of 1,100,000,000 bytes, in batches of their own of
        // a single partition: together, more than an array of strings with
        // 32-bit offsets can hold, which the table of a thread holds of the
        // keys of `LargeUtf8` (64-bit offsets) and `Utf8View` (views of several
        // buffers). (Those of `Utf8` do not fit into the `GroupValues` of a
        // table, whatever its limit, nor into DataFusion's: a capacity limit of
        // the tables of both engines, which DataFusion's top-k aggregation, of
        // the best group only, avoids.)
        let len = 1_100_000_000;
        for data_type in [DataType::LargeUtf8, DataType::Utf8View] {
            let schema =
                Arc::new(Schema::new(vec![Field::new("k", data_type.clone(), false)]));
            let key = |byte: u8| -> Result<RecordBatch> {
                let bytes = Buffer::from_vec(vec![byte; len]);
                let key = LargeStringArray::new(
                    OffsetBuffer::new(vec![0, len as i64].into()),
                    bytes,
                    None,
                );
                let key = arrow::compute::cast(&key, &data_type)?;
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
    async fn limited_aggregations_of_any_values_are_native() -> Result<()> {
        use super::super::test_support::{
            assert_aggregate_fails_with_setup, assert_any_rows,
            assert_lowered_results_with_setup, assert_top_rows, lowered_outputs,
            without_limited_aggregations,
        };
        let setup = async |ctx: &datafusion::prelude::SessionContext| {
            crate::test_utils::register_test_tables(ctx)?;
            register_tk(ctx)
        };
        let (key, aggregate) = (TopKOrder::Key, TopKOrder::Aggregate);
        for target_partitions in [1, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            let ctx = crate::test_utils::datafusion_context(config.clone());
            setup(&ctx).await?;
            // The limits of the native aggregation of `sql`.
            let lowered =
                async |sql: &str| -> Result<(Option<TopKOrder>, Option<usize>)> {
                    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                    let plan = top_k_aggregation(&plan).expect("a limited aggregation");
                    let aggregate = try_lower(&plan)?.expect("a lowered aggregation");
                    let spec = &aggregate.spec;
                    Ok((spec.top_k.map(|top_k| top_k.order), spec.soft_limit))
                };
            // Floating point values, by their total order, as the sort orders
            // them (DataFusion's top-k aggregation compares them otherwise), of
            // nullable arguments, strings with 32-bit offsets, and final stages
            // of several partitions.
            for (sql, order) in [
                ("SELECT DISTINCT c FROM t1 ORDER BY c LIMIT 3", key),
                ("SELECT DISTINCT c FROM t1 ORDER BY c DESC LIMIT 3", key),
                (
                    "SELECT g, MAX(v) AS m FROM tk GROUP BY g ORDER BY m DESC NULLS LAST \
                     LIMIT 3",
                    aggregate,
                ),
                ("SELECT DISTINCT s FROM tk ORDER BY s LIMIT 3", key),
                (
                    "SELECT g, MAX(ns) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 3",
                    aggregate,
                ),
                ("SELECT DISTINCT g FROM tk ORDER BY g LIMIT 3", key),
            ] {
                assert_eq!(lowered(sql).await?, (Some(order), None), "{sql}");
                assert_lowered_results_with_setup(
                    sql,
                    config.clone(),
                    without_limited_aggregations(config.clone()),
                    &setup,
                )
                .await;
            }
            // Minima and maxima of floating point values, of which many groups
            // tie (`NaN`s are the greatest values).
            for (sql, descending) in [
                (
                    "SELECT a, MAX(c) AS m FROM t1 GROUP BY a ORDER BY m DESC NULLS LAST \
                     LIMIT 3",
                    true,
                ),
                (
                    "SELECT a, MIN(c) AS m FROM t1 GROUP BY a ORDER BY m LIMIT 3",
                    false,
                ),
            ] {
                assert_eq!(lowered(sql).await?, (Some(aggregate), None), "{sql}");
                let function = if descending { "MAX" } else { "MIN" };
                let all = format!("SELECT a, {function}(c) AS m FROM t1 GROUP BY a");
                let all = ctx.sql(&all).await?.collect().await?;
                for (engine, output, _) in
                    lowered_outputs(sql, config.clone(), &setup).await
                {
                    assert_top_rows(&output.batches, &all, 1, descending, 3, engine);
                }
            }
            // Keys and arguments that may fail, which both engines evaluate for
            // all rows.
            for (sql, order) in [
                (
                    "SELECT 100 / a AS q FROM t1 GROUP BY 100 / a ORDER BY q LIMIT 3",
                    key,
                ),
                (
                    "SELECT g, MAX(1000 / id) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 3",
                    aggregate,
                ),
            ] {
                assert_eq!(lowered(sql).await?, (Some(order), None), "{sql}");
                assert_aggregate_fails_with_setup(
                    sql,
                    config.clone(),
                    &setup,
                    &["Divide by zero"],
                )
                .await;
            }
            // A soft limit: any 3 groups.
            let sql = "SELECT DISTINCT d FROM t1 LIMIT 3";
            assert_eq!(lowered(sql).await?, (None, Some(3)), "{sql}");
            let all = ctx
                .sql("SELECT DISTINCT d FROM t1")
                .await?
                .collect()
                .await?;
            for (engine, output, _) in lowered_outputs(sql, config.clone(), &setup).await
            {
                assert_any_rows(&output.batches, &all, 3, engine);
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rows_of_aggregations_of_any_aggregates_may_bypass_the_tables() -> Result<()>
    {
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        crate::test_utils::register_test_tables(&ctx)?;
        ctx.register_udaf(super::super::test_support::lookalike_count());
        ctx.register_udaf(super::super::test_support::lookalike_sum());
        // The queries, and whether their rows may bypass the tables, whatever
        // their aggregate functions (built-in or user-defined, `DISTINCT` and
        // ordered ones included).
        let queries = [
            // Without aggregates.
            ("SELECT DISTINCT id, c, f, ts FROM t1", true),
            ("SELECT a, d FROM t1 GROUP BY a, d", true),
            (
                "SELECT a, COUNT(*), SUM(b), MIN(d), AVG(g) FROM t1 GROUP BY a",
                true,
            ),
            (
                "SELECT h, MAX(c), BOOL_AND(e), BOOL_OR(e), BIT_AND(b), BIT_OR(a), \
                 BIT_XOR(b), APPROX_DISTINCT(d), MEDIAN(b) FILTER (WHERE e), \
                 ARRAY_AGG(a), FIRST_VALUE(d), LAST_VALUE(f), COUNT(a, b) \
                 FROM t1 GROUP BY h",
                true,
            ),
            // `DISTINCT` aggregates. (DataFusion plans a single `DISTINCT`
            // aggregate, or several of the same argument with `SUM`, `MIN` or
            // `MAX`, as an aggregation of the aggregation by the group keys and
            // the argument, without `DISTINCT`: these queries keep their
            // `DISTINCT` aggregates.)
            (
                "SELECT d, h, COUNT(DISTINCT a), MEDIAN(b) FILTER (WHERE e) FROM t1 \
                 GROUP BY d, h",
                true,
            ),
            (
                "SELECT d, SUM(DISTINCT b), COUNT(*) FROM t1 GROUP BY d",
                true,
            ),
            (
                "SELECT h, ARRAY_AGG(DISTINCT a), COUNT(*) FROM t1 GROUP BY h",
                true,
            ),
            (
                "SELECT h, AVG(DISTINCT a), MEDIAN(DISTINCT b), MAX(c) FROM t1 GROUP BY h",
                true,
            ),
            // Ordered aggregates, whose accumulators order their values.
            (
                "SELECT d, ARRAY_AGG(id ORDER BY id) FROM t1 GROUP BY d",
                true,
            ),
            (
                "SELECT d, LAST_VALUE(a ORDER BY id), COUNT(*) FROM t1 GROUP BY d",
                true,
            ),
            (
                "SELECT d, ARRAY_AGG(a ORDER BY b DESC) FILTER (WHERE e), MAX(a) FROM t1 \
                 GROUP BY d",
                true,
            ),
            // Other built-in functions, and user-defined functions that look like
            // DataFusion's `count` and `sum` (see `lookalike_count`).
            ("SELECT d, STDDEV(b) FROM t1 GROUP BY d", true),
            (
                "SELECT d, COUNT(*), STRING_AGG(h, ',') FROM t1 GROUP BY d",
                true,
            ),
            ("SELECT a, count(b), sum(b) FROM t1 GROUP BY a", true),
            // Aggregates that require their rows in an order, which every
            // thread sorts before it aggregates them.
            (
                "SELECT d, NTH_VALUE(a, 2 ORDER BY id) FROM t1 GROUP BY d",
                false,
            ),
            // Grouping sets.
            ("SELECT d, e FROM t1 GROUP BY ROLLUP (d, e)", false),
            (
                "SELECT d, e, COUNT(*) FROM t1 GROUP BY ROLLUP (d, e)",
                false,
            ),
            // Keys and arguments of nested types and dictionaries.
            ("SELECT [a, b] AS ab FROM t1 GROUP BY [a, b]", false),
            (
                "SELECT arrow_cast(d, 'Dictionary(Int32, Utf8)') AS dd FROM t1 GROUP BY dd",
                false,
            ),
            ("SELECT d, MIN([a, b]) FROM t1 GROUP BY d", false),
            // Limited aggregations, whose tables hold the groups of their limit.
            ("SELECT DISTINCT d FROM t1 LIMIT 3", false),
            ("SELECT DISTINCT a FROM t1 ORDER BY a LIMIT 3", false),
        ];
        /// The first aggregation of the plan of `sql` that is lowered, from the
        /// top.
        async fn lowered_aggregation(
            ctx: &datafusion::prelude::SessionContext,
            sql: &str,
        ) -> Result<PhysicalHashAggregate> {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let mut aggregate = None;
            let mut node = Arc::clone(&plan);
            while aggregate.is_none() {
                aggregate = try_lower(&node)?;
                let Some(child) = node.children().first().map(|&child| Arc::clone(child))
                else {
                    break;
                };
                node = child;
            }
            Ok(aggregate.unwrap_or_else(|| panic!("no aggregation: {sql}")))
        }
        for (sql, bypass) in queries {
            let aggregate = lowered_aggregation(&ctx, sql).await?;
            assert_eq!(aggregate.spec.bypass, bypass, "{sql}");
        }
        // Narrow fixed-size binaries only.
        assert!(is_flat_type(&DataType::FixedSizeBinary(16)));
        assert!(is_flat_type(&DataType::FixedSizeBinary(
            MAX_FIXED_SIZE_BINARY
        )));
        assert!(!is_flat_type(&DataType::FixedSizeBinary(1_100_000)));
        Ok(())
    }

    #[test]
    fn built_in_aggregate_functions_are_identified_by_their_implementation_types()
    -> Result<()> {
        use datafusion::functions_aggregate::{
            approx_distinct::ApproxDistinct, array_agg::ArrayAgg, average::Avg,
            bool_and_or::BoolAnd, bool_and_or::BoolOr, count::Count,
            first_last::FirstValue, first_last::LastValue, median::Median, min_max::Max,
            min_max::Min, sum::Sum,
        };

        use super::super::test_support::{lookalike_count, lookalike_max, lookalike_sum};
        // Of the built-in aggregate functions that a session registers by
        // default, exactly those of `BUILTIN_AGGREGATES` are trusted (by the
        // extension, which the native aggregation no longer reads).
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

        // Functions that look like DataFusion's `count`, `sum` and `max` in
        // every way but their types: of the same names, implemented by types of
        // the same names, whose `Debug` output is the built-in functions'.
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let aggregate = |function: Arc<AggregateUDF>| {
            AggregateExprBuilder::new(function, vec![col("v", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("aggregate")
                .build()
        };
        let lookalikes =
            [lookalike_count(), lookalike_sum(), lookalike_max()].map(Arc::new);
        for (lookalike, builtin) in
            lookalikes
                .iter()
                .zip([count_udaf(), sum_udaf(), max_udaf()])
        {
            assert_eq!(lookalike.name(), builtin.name());
            assert_eq!(
                format!("{:?}", lookalike.inner()),
                format!("{:?}", builtin.inner())
            );
            assert_ne!(
                implementation_type(lookalike),
                implementation_type(&builtin)
            );
        }
        // They are not trusted, on their own or with the built-in functions.
        let given = PhysicalHashAggregate::trusted_aggregate_functions(&lookalikes);
        assert!(given.names().is_empty());
        let mut all = lookalikes.to_vec();
        all.extend(functions.iter().cloned());
        let given = PhysicalHashAggregate::trusted_aggregate_functions(&all);
        let mut names = given.names();
        names.sort_unstable();
        assert_eq!(names, builtins);
        // Only DataFusion's `max`, descending, and `min`, ascending, order the
        // groups of top-k aggregations (of the same order, see `is_min_max`).
        let (max, lookalike) = (
            aggregate(max_udaf())?,
            aggregate(Arc::clone(&lookalikes[2]))?,
        );
        assert_eq!(max.fun().inner().is_descending(), Some(true));
        assert_eq!(lookalike.fun().inner().is_descending(), Some(true));
        assert!(is_min_max(&max, true, &schema));
        assert!(!is_min_max(&max, false, &schema));
        assert!(!is_min_max(&lookalike, true, &schema));
        let min = aggregate(min_max::min_udaf())?;
        assert!(is_min_max(&min, false, &schema));
        assert!(!is_min_max(&min, true, &schema));
        assert!(!is_min_max(&aggregate(sum_udaf())?, true, &schema));
        Ok(())
    }

    /// A benchmark of top-k aggregations, not a test: the pipeline engine and
    /// DataFusion, interleaved, over `TOP_K_ROWS` (10,500,000) rows of a
    /// single partition of 8,192-row batches, `TOP_K_ROUNDS` (5) times, on
    /// `TOP_K_PARTITIONS` (1) target partitions: `shuffled` keys `k` and
    /// values `v` of linear congruential generators (`(i * 1664525 +
    /// 1013904223) & 0xffffffff`), and `sorted` ones (`i`), whose best rows
    /// come first for ascending orders (and last for descending ones), with
    /// 1,000 groups `g` of `k % 1000`. Prints the median and best times of
    /// each engine, and whether the pipeline engine lowered the top-k
    /// aggregation. Run it with `cargo test --profile release-nonlto -p
    /// datafusion-physical-pipeline --lib top_k_aggregations_benchmark --
    /// --ignored --nocapture`.
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
        use std::sync::Arc;
        use std::time::Duration;
        let env = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        };
        let (rows, rounds) = (env("TOP_K_ROWS", 10_500_000), env("TOP_K_ROUNDS", 5));
        let partitions = env("TOP_K_PARTITIONS", 1);
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
            .with_target_partitions(partitions)
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
            let native = plan
                .downcast_ref::<crate::bridge::PipelineRootExec>()
                .is_some_and(|root| {
                    !native_top_k_aggregations(root.operator()).is_empty()
                });
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
