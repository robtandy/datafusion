// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalHashAggregate`]: the pipeline breaker that replaces DataFusion's
//! `AggregateExec`s, see the [module documentation](super).

use std::fmt;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    make_comparator, new_empty_array, new_null_array,
};
use arrow::compute::{
    SortColumn, SortOptions, concat, filter_record_batch, interleave, lexsort_to_indices,
    sort_to_indices, take_record_batch,
};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{Result, ScalarValue, internal_err};
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

use super::radix_ht::{RadixHashAggregate, TableCapacity};
use super::ungrouped::UngroupedAggregate;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
};
use crate::operators::adaptive_filter::is_infallible;

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
    /// [`rows_can_bypass_tables`]).
    pub(crate) bypass: bool,
}

impl AggregateSpec {
    /// Whether there is no `GROUP BY` (and no grouping sets): the
    /// aggregation produces exactly one row.
    pub(crate) fn is_ungrouped(&self) -> bool {
        self.group_by.is_true_no_grouping()
    }
}

/// The limit of a top-k aggregation, which DataFusion's `TopKAggregation`
/// rule sets on the stages of an aggregation (`lim=[n]`) below a sort with a
/// fetch of `limit` rows whose only sort key is
///
/// * the only group key of an aggregation without aggregates (DataFusion's
///   `TopKDistinct`, e.g. `SELECT DISTINCT k FROM t ORDER BY k LIMIT 10`), or
/// * the only aggregate of an aggregation by a single group key, a `MAX`
///   ordered descending or a `MIN` ordered ascending (`TopKMinMax`, e.g.
///   `SELECT k, MAX(v) FROM t GROUP BY k ORDER BY MAX(v) DESC LIMIT 10`).
///
/// The sort keeps the first `limit` groups by that value, so the aggregation
/// only produces the groups that can be among them, exactly like DataFusion's
/// `GroupedTopKAggregateStream` of a single partition: the `limit` groups
/// with the best non-null values (any of those that tie with the last of
/// them), and the group of the `NULL` key, which DataFusion keeps whatever the
/// order of the `NULL`s (the aggregate of a top-k aggregation that the native
/// operator reproduces is never `NULL`, see
/// [`PhysicalHashAggregate::try_lower`]). The groups are compared by their
/// values as the sort compares them, as DataFusion's top-k aggregation does
/// for the types of [`orders_like_datafusion`]. So the operators between the
/// aggregation and the sort above it see exactly the groups they see in
/// DataFusion (but for ties), also the operators that see all of them (e.g. a
/// window `COUNT(*) OVER ()`: DataFusion's `TopKAggregation` rule passes the
/// limit through the operators that keep the number of rows, windows
/// included).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TopK {
    /// The number of groups the sort keeps.
    pub(crate) limit: usize,
    /// Whether the groups with the greatest values come first.
    pub(crate) descending: bool,
    /// What orders the groups.
    pub(crate) order: TopKOrder,
}

/// What orders the groups of a top-k aggregation, see [`TopK`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopKOrder {
    /// The group key (the aggregation has no aggregates).
    GroupKey,
    /// The value of the only aggregate, a `MIN` or `MAX`, whose partial state
    /// is that value too.
    Aggregate,
}

impl TopK {
    /// The rows of `values`, the values that order a set of distinct groups
    /// (one per group), whose groups may be among the groups the sort keeps
    /// (see [`TopK`]): the `limit` best non-null values, followed by the
    /// `NULL`s (of the group key, which is `NULL` for one group at most; at
    /// most `limit` of an aggregate). Returns `None` if these are all rows.
    ///
    /// The values of a group never get worse when more of its rows are
    /// aggregated (a `MAX` only grows, a `MIN` only shrinks), so a group that
    /// is not selected from the partial states of some rows (other groups
    /// beat it already) is not among the groups the sort keeps either, unless
    /// a better value of it comes from other rows.
    pub(crate) fn select(&self, values: &dyn Array) -> Result<Option<UInt32Array>> {
        let nulls = values.null_count();
        let valid = values.len() - nulls;
        let null_limit = self.null_limit();
        if valid <= self.limit && nulls <= null_limit {
            return Ok(None);
        }
        let mut indices = if valid <= self.limit {
            (0..values.len())
                .filter(|&row| values.is_valid(row))
                .map(|row| row as u32)
                .collect::<Vec<_>>()
        } else {
            // With the nulls last, the first `limit` rows are non-null.
            let options = SortOptions::new(self.descending, false);
            sort_to_indices(values, Some(options), Some(self.limit))?
                .values()
                .to_vec()
        };
        indices.extend(
            (0..values.len())
                .filter(|&row| values.is_null(row))
                .take(null_limit)
                .map(|row| row as u32),
        );
        Ok(Some(UInt32Array::from(indices)))
    }

    /// The groups of `batch`, the output of the aggregation for a set of
    /// distinct groups, that may be among the groups the sort keeps (see
    /// [`Self::select`]).
    pub(crate) fn select_groups(&self, batch: RecordBatch) -> Result<RecordBatch> {
        match self.select(batch.column(self.column()).as_ref())? {
            Some(indices) => Ok(take_record_batch(&batch, &indices)?),
            None => Ok(batch),
        }
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
        let (left_values, right_values) =
            (left.column(self.column()), right.column(self.column()));
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
                .chain(nulls(1, right_values))
                .take(self.null_limit()),
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

    /// The column of the output of the aggregation that orders the groups:
    /// the group key, followed by the aggregate.
    fn column(&self) -> usize {
        match self.order {
            TopKOrder::GroupKey => 0,
            TopKOrder::Aggregate => 1,
        }
    }

    /// The most groups whose value is `NULL` that may be among the groups
    /// the sort keeps, see [`Self::select`].
    fn null_limit(&self) -> usize {
        match self.order {
            // At most one group has a `NULL` key.
            TopKOrder::GroupKey => usize::MAX,
            TopKOrder::Aggregate => self.limit,
        }
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
    ///   aggregations whose stages have the same limit (of at most
    ///   [`MAX_TOP_K_LIMIT`] groups), that group by a single key and order by
    ///   it (without aggregates) or by their only aggregate, a `MIN` or `MAX`
    ///   (without a filter or an order of its own) of an argument that is not
    ///   nullable, whose partial state is its value, of types that
    ///   DataFusion's top-k aggregation orders like its sorts (see
    ///   [`orders_like_datafusion`]) but strings with 32-bit offsets (see
    ///   [`holds_like_datafusion`]), whose key and argument cannot fail
    ///   (conservatively), and whose final stage has a single partition.
    ///   DataFusion's top-k aggregation of several partitions keeps the top
    ///   groups of every partition, which depend on how its rows are
    ///   partitioned (e.g. the number of groups that a window `COUNT(*) OVER
    ///   ()` above it counts), and it keeps (some of) the groups whose
    ///   aggregate is `NULL`, which depend on the order of its rows.
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
    let order = match raw.aggr_expr() {
        [] => TopKOrder::GroupKey,
        [aggregate] => {
            let value_type = aggregate.field().data_type().clone();
            // A `MAX` ordered descending, or a `MIN` ordered ascending, whose
            // partial state is its value.
            let ordered = matches!(aggregate.fun().name(), "min" | "max")
                && aggregate
                    .get_minmax_desc()
                    .is_some_and(|(_, desc)| desc == descending)
                && !aggregate.is_distinct()
                && aggregate.order_bys().is_empty()
                && aggregate.state_fields().is_ok_and(|fields| {
                    matches!(fields.as_slice(), [state] if state.data_type() == &value_type)
                });
            // DataFusion keeps groups whose aggregate is `NULL` depending on
            // the order of the rows: with a value that loses to its top groups
            // first, and a `NULL` once the value is gone, a group is kept with
            // a `NULL` aggregate.
            match aggregate.expressions().as_slice() {
                [argument]
                    if ordered
                        && orders(argument)
                        && argument.nullable(&schema).is_ok_and(|nullable| !nullable)
                        && argument
                            .data_type(&schema)
                            .is_ok_and(|data_type| data_type == value_type) =>
                {
                    TopKOrder::Aggregate
                }
                _ => return GroupLimit::Unsupported,
            }
        }
        _ => return GroupLimit::Unsupported,
    };
    GroupLimit::TopK(TopK {
        limit,
        descending,
        order,
    })
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
/// * the aggregation has group keys, and no grouping sets;
/// * the aggregates of the final aggregation (`merge`), which aggregate the
///   rows that bypass the tables, are those of the stage that consumes the
///   input rows (`update`), as in DataFusion's plans;
/// * the keys and the arguments (of the rows of a thread, which the radix
///   partitioning gathers from several chunks) are of flat types, which are
///   gathered without merging dictionaries or nested offsets.
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

/// Whether arrays of `data_type` have no children and no dictionary.
fn is_flat_type(data_type: &DataType) -> bool {
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
            | DataType::FixedSizeBinary(_)
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
            RadixHashAggregate::local_sink_state(&self.spec, self.table_capacity)
        }
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
            order: TopKOrder::GroupKey,
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
        let select = |limit, descending, order| -> Result<Option<Vec<u32>>> {
            let top_k = TopK {
                limit,
                descending,
                order,
            };
            Ok(top_k.select(&values)?.map(|indices| {
                let mut indices = indices.values().to_vec();
                indices.sort_unstable();
                indices
            }))
        };
        // The 2 greatest values (9, 7), the 2 least (1, 3), and the nulls:
        // all of them for a group key, at most `limit` for an aggregate.
        assert_eq!(
            select(2, true, TopKOrder::GroupKey)?,
            Some(vec![1, 2, 4, 5, 7])
        );
        assert_eq!(
            select(2, false, TopKOrder::GroupKey)?,
            Some(vec![1, 3, 4, 6, 7])
        );
        assert_eq!(
            select(2, true, TopKOrder::Aggregate)?,
            Some(vec![1, 2, 4, 5])
        );
        assert_eq!(select(1, false, TopKOrder::Aggregate)?, Some(vec![1, 3]));
        assert_eq!(
            select(3, true, TopKOrder::Aggregate)?,
            Some(vec![0, 1, 2, 4, 5, 7])
        );
        // All rows, if there are not more of them.
        assert_eq!(select(5, true, TopKOrder::GroupKey)?, None);
        assert_eq!(select(5, false, TopKOrder::Aggregate)?, None);
        // All values, but not all nulls.
        let few = Int64Array::from(vec![None, Some(4), None, None]);
        let top_k = TopK {
            limit: 2,
            descending: false,
            order: TopKOrder::Aggregate,
        };
        let indices = top_k.select(&few)?.expect("a selection");
        assert_eq!(indices.values().as_ref(), &[1, 0, 2]);
        // Strings, by their bytes.
        let strings = StringArray::from(vec![Some("b"), Some("ab"), None, Some("B")]);
        let top_k = TopK {
            limit: 1,
            descending: false,
            order: TopKOrder::GroupKey,
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
        for (order, descending) in [
            (TopKOrder::GroupKey, false),
            (TopKOrder::GroupKey, true),
            (TopKOrder::Aggregate, false),
            (TopKOrder::Aggregate, true),
        ] {
            for limit in [1, 2, 3, 10] {
                let top_k = TopK {
                    limit,
                    descending,
                    order,
                };
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
            // By a `MAX` descending or a `MIN` ascending of an argument that is
            // not nullable, of every supported type.
            (
                "SELECT g, MAX(id) FROM tk GROUP BY g ORDER BY MAX(id) DESC LIMIT 4",
                true,
            ),
            (
                "SELECT g, MIN(id) FROM tk GROUP BY g ORDER BY MIN(id) LIMIT 5",
                true,
            ),
            (
                "SELECT u, MIN(ndec) FROM tk GROUP BY u ORDER BY MIN(ndec) LIMIT 3",
                true,
            ),
            (
                "SELECT s, MAX(nt) FROM tk GROUP BY s ORDER BY MAX(nt) DESC NULLS LAST \
                 LIMIT 6",
                false,
            ),
            (
                "SELECT ls, MAX(nt) FROM tk GROUP BY ls ORDER BY MAX(nt) DESC LIMIT 6",
                true,
            ),
            (
                "SELECT dt, MIN(nsv) FROM tk GROUP BY dt ORDER BY MIN(nsv) LIMIT 4",
                true,
            ),
            (
                "SELECT g, MAX(nsv) FROM tk GROUP BY g ORDER BY MAX(nsv) DESC LIMIT 1000",
                true,
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
                true,
            ),
            (
                "SELECT COUNT(*), SUM(m), MIN(g) FROM \
                 (SELECT g, MAX(id) AS m FROM tk GROUP BY g ORDER BY m DESC LIMIT 5)",
                true,
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
                true,
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
        for (sql, bypass) in [
            (
                "SELECT a, COUNT(*), SUM(b), MIN(d), AVG(g) FROM t1 GROUP BY a",
                true,
            ),
            (
                "SELECT d, h, COUNT(DISTINCT a), MEDIAN(b) FILTER (WHERE e) FROM t1 \
                 GROUP BY d, h",
                true,
            ),
            ("SELECT DISTINCT id, c, f, ts FROM t1", true),
            (
                "SELECT d, ARRAY_AGG(id ORDER BY id) FROM t1 GROUP BY d",
                true,
            ),
            // Grouping sets.
            (
                "SELECT d, e, COUNT(*) FROM t1 GROUP BY ROLLUP (d, e)",
                false,
            ),
            // The rows of an aggregate with a hard order are sorted.
            (
                "SELECT e, NTH_VALUE(a, 3 ORDER BY id) FROM t1 GROUP BY e",
                false,
            ),
            // Keys and arguments of nested types and dictionaries.
            (
                "SELECT [a, b] AS ab, COUNT(*) FROM t1 GROUP BY [a, b]",
                false,
            ),
            (
                "SELECT arrow_cast(d, 'Dictionary(Int32, Utf8)') AS dd, COUNT(*) FROM t1 \
                 GROUP BY dd",
                false,
            ),
            ("SELECT d, ARRAY_AGG([a, b]) FROM t1 GROUP BY d", false),
        ] {
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
            let aggregate = aggregate.unwrap_or_else(|| panic!("no aggregation: {sql}"));
            assert_eq!(aggregate.spec.bypass, bypass, "{sql}");
        }
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
}
