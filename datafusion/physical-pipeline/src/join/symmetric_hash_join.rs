// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalSymmetricHashJoin`], and the native lowering of DataFusion's
//! `SymmetricHashJoinExec`.
//!
//! DataFusion's `JoinSelection` replaces a hash join of two inputs that never
//! end (unbounded inputs, e.g. streams) by a `SymmetricHashJoinExec`: a hash
//! join cannot read its build side first, as it never ends. The symmetric
//! join reads both inputs at the same time and keeps a hash table of the rows
//! of each of them: every row is joined with the rows of the other input that
//! arrived before it, and then inserted into the table of its own input.
//!
//! # Lowering
//!
//! [`try_lower`] lowers every `SymmetricHashJoinExec` natively, whatever its
//! join type, keys, filter and null equality:
//!
//! * if an input ends (it is bounded), as a [`PhysicalHashJoin`] that builds
//!   its table on that input (on the smaller one if both end, by their
//!   statistics), and streams the other one through its probe side;
//! * if both inputs never end, as a [`PhysicalSymmetricHashJoin`].
//!
//! The nodes below the join that only distribute its rows over partitions
//! (its partition mode) are removed: the native joins read all rows of both
//! inputs, on all threads. A symmetric join whose inputs are sorted by
//! expressions of its filter (see below) reads each input in its order
//! instead: a `SortPreservingMergeExec` merges the sorted partitions of an
//! input that has several.
//!
//! # `PhysicalSymmetricHashJoin`
//!
//! The join is the source of its pipeline. Its inputs are executed by
//! executors of their own (see `NestedInput`), on the threads of the
//! session, and the join reads both of them at the same time, as their rows
//! come, whatever the operators above it (also those that are not parallel,
//! e.g. a filter with a fetch, whose pipelines the executor runs one after the
//! other: pipelines of the two inputs that shared it would wait for each
//! other forever):
//!
//! ```text
//! MetaPipeline 0 (sink: ...)
//!   Pipeline 0: PhysicalSymmetricHashJoin -> ...
//! (the left and the right input: executions of their own)
//! ```
//!
//! * A single thread reads the next chunk of either input (alternately, of the
//!   inputs that have one), appends its rows to the table of its input, and
//!   joins them with the rows of the other input that are in its table (with
//!   the residual filter). So every pair of rows is joined exactly once, when
//!   the later of the two arrives, whatever the order in which the rows of the
//!   two inputs arrive. A chunk is joined in parts of at most
//!   [`DEFAULT_VECTOR_SIZE`] candidate pairs and about a MiB of the rows that
//!   the join gathers for them, for its filter and its output (fewer pairs of
//!   wide rows): the join reserves the bytes of these rows (those of their
//!   strings, too) in the memory pool before it gathers them, and emits every
//!   part at once (the pairs that pass the filter, and the rows of a semi
//!   join when they match for the first time): the join never buffers more
//!   than a part of candidates, however many rows a row matches, and a limit
//!   above it stops it after the rows it needs.
//! * A row is *determined* once no row of the other input still to come can
//!   match it: the join emits its final row then (for `Left`, `Right`, `Full`,
//!   anti and mark joins: the row without a match, with NULLs for the other
//!   input, or the row with its mark), and drops it from its table, releasing
//!   its memory. The rows of an input are determined once the other input
//!   ended, and, if both inputs are sorted by expressions of the filter (e.g.
//!   by times, with a filter `l.time > r.time`), as soon as the bounds of the
//!   values still to come rule them out: like DataFusion's interval pruning,
//!   the join derives from the last sorted value of either input (the values
//!   of its rows still to come are not before it) the interval of the sorted
//!   values of the other input that can still match, with the interval
//!   arithmetic of the filter (see [`ExprIntervalGraph`]; it supports
//!   comparisons, `AND`, and the arithmetic that cannot overflow: not that of
//!   integers, which wraps). The rows of the other input before that interval
//!   (they arrived sorted) are determined. Interval arithmetic bounds real
//!   numbers, which the arithmetic of floats is not (its NaN, of a NaN still
//!   to come or of `inf - inf`, compares after or before every number): a
//!   filter that computes with floats determines no rows by the orders.
//!   The join does so after every chunk of either input (DataFusion's join
//!   after the chunks of the other input only), so that the rows of an input
//!   whose chunks come after those of the other input are determined, too,
//!   and it reads such inputs alternately (it reads no more than
//!   `MAX_CHUNKS_AHEAD` chunks of either ahead of the other, as DataFusion's
//!   join), so that the rows that wait for the other input are few.
//!   So a limit above an outer, anti or mark join of streams that never end
//!   ends the query, and a join of sorted streams keeps the rows that can
//!   still match only.
//!
//! The tables reserve their memory in the memory pool of the session: the
//! join fails with `ResourcesExhausted` when the pool refuses them. Join keys
//! are equal with DataFusion's join key equality: `-0.0 = +0.0`, NaNs with the
//! same bits are equal, and NULLs are equal only with
//! [`NullEquality::NullEqualsNull`].
//!
//! [`PhysicalHashJoin`]: crate::join::PhysicalHashJoin

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Arc;
use std::task::Poll;

use arrow::array::{ArrayRef, BooleanArray, RecordBatch, UInt32Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_row::{OwnedRow, Row, RowConverter, Rows, SortField};
use datafusion_common::float_semantics::normalize_floats;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{
    DataFusionError, JoinSide, JoinType, NullEquality, Result, ScalarValue, internal_err,
    plan_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::Operator;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal, NegativeExpr};
use datafusion_physical_expr::intervals::cp_solver::{
    ExprIntervalGraph, PropagationResult,
};
use datafusion_physical_expr::intervals::utils::check_support;
use datafusion_physical_expr::{LexOrdering, PhysicalExprRef, PhysicalSortExpr};
use datafusion_physical_plan::joins::SymmetricHashJoinExec;
use datafusion_physical_plan::joins::utils::JoinFilter;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use hashbrown::HashMap;
use parking_lot::Mutex;

use super::merge_join::{
    EquiJoin, FILTER, GatherWidths, InputRole, JoinOutput, JoinRows, NestedInput, OUTPUT,
    PAIR_INDEX_BYTES, PART_BYTES, StoredRows, compact_to_keep, comparable_type,
    evaluate_keys, filter_mask, is_smaller, key_memory, matchable_rows, probe_role,
    without_reordering,
};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::memory_size;

/// Lower a DataFusion `SymmetricHashJoinExec` natively, see the [module
/// documentation](self): to a [`PhysicalHashJoin`] that builds its table on
/// an input that ends, or to a [`PhysicalSymmetricHashJoin`] if both inputs
/// never end.
///
/// `lower_input` lowers an input of the join, which the join reads as its
/// [`InputRole`] says.
///
/// Returns `Ok(None)` for any other node, and for the joins that no native
/// join implements: joins whose keys have different types on the two sides,
/// or whose keys the row format does not support.
///
/// [`PhysicalHashJoin`]: crate::join::PhysicalHashJoin
pub fn try_lower<F>(
    plan: &Arc<dyn ExecutionPlan>,
    mut lower_input: F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, InputRole) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(join) = plan.downcast_ref::<SymmetricHashJoinExec>() else {
        return Ok(None);
    };
    let left = without_reordering(join.left())?;
    let right = without_reordering(join.right())?;
    let equi_join = EquiJoin {
        plan,
        left: &left,
        right: &right,
        on: join.on(),
        filter: join.filter(),
        join_type: *join.join_type(),
        projection: None,
        null_equality: join.null_equality(),
        null_aware: false,
    };
    let unbounded =
        [join.left(), join.right()].map(|input| input.boundedness().is_unbounded());
    if unbounded == [true, true] {
        let sorted = match (join.left_sort_exprs(), join.right_sort_exprs()) {
            (Some(left), Some(right)) => Some(SortedInputs {
                inputs: [join.left(), join.right()],
                orders: [left, right],
            }),
            _ => None,
        };
        return lower_symmetric(&equi_join, sorted, &mut lower_input);
    }
    // A hash join builds its table on an input that ends, on the smaller one
    // if both do.
    let build_right = unbounded[0] || (!unbounded[1] && is_smaller(&right, &left));
    let (build, probe) = match build_right {
        true => (&right, &left),
        false => (&left, &right),
    };
    let build = lower_input(build, InputRole::Collected)?;
    let probe = lower_input(probe, probe_role(probe))?;
    if let Some(hash_join) = equi_join.hash_join(build_right, &build, &probe)? {
        return Ok(Some(Arc::new(hash_join)));
    }
    // A join that the hash join does not implement: a symmetric hash join of
    // the same inputs.
    let (left, right) = match build_right {
        true => (probe, build),
        false => (build, probe),
    };
    let Some(spec) = SymmetricJoinSpec::of(&equi_join)? else {
        return Ok(None);
    };
    Ok(Some(Arc::new(PhysicalSymmetricHashJoin {
        children: vec![left, right],
        spec,
    })))
}

/// The inputs of a symmetric join (as its plan has them), whose rows are
/// sorted, and their orders.
pub(crate) struct SortedInputs<'a> {
    pub(crate) inputs: [&'a Arc<dyn ExecutionPlan>; 2],
    pub(crate) orders: [&'a LexOrdering; 2],
}

/// Lower `join`, a join of two inputs that never end, to a
/// [`PhysicalSymmetricHashJoin`] of its inputs, lowered with `lower_input`
/// as streams. If the inputs are `sorted` by expressions of the filter of the
/// join, the join reads them in their orders, and determines their rows by
/// the bounds of the values still to come (see the [module
/// documentation](self)). `Ok(None)` if the symmetric join does not implement
/// it (see [`try_lower`]).
pub(crate) fn lower_symmetric<F>(
    join: &EquiJoin<'_>,
    sorted: Option<SortedInputs<'_>>,
    lower_input: &mut F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, InputRole) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(mut spec) = SymmetricJoinSpec::of(join)? else {
        return Ok(None);
    };
    let mut inputs = [Arc::clone(join.left), Arc::clone(join.right)];
    if let Some(sorted) = sorted
        && let Some(sorted_filter) = SortedFilter::try_new(
            join.filter,
            [&join.left.schema(), &join.right.schema()],
            sorted.orders,
        )?
        && let Some(left) = sorted_plan(sorted.inputs[0], sorted.orders[0])?
        && let Some(right) = sorted_plan(sorted.inputs[1], sorted.orders[1])?
    {
        inputs = [left, right];
        spec.sorted_filter = Some(sorted_filter);
    }
    let mut children = Vec::with_capacity(2);
    for input in &inputs {
        let lowered = lower_input(input, InputRole::Streamed)?;
        if lowered.schema() != input.schema() {
            return internal_err!(
                "The lowered input {} of a symmetric hash join has schema {:?}, expected {:?}",
                lowered.name(),
                lowered.schema(),
                input.schema()
            );
        }
        children.push(lowered);
    }
    Ok(Some(Arc::new(PhysicalSymmetricHashJoin { children, spec })))
}

/// The plan that produces the rows of `input`, an input of a symmetric join
/// whose plan reads it sorted by `order`, in that order on a single
/// partition: `input` without the nodes that only sort or distribute its rows,
/// whose partitions a `SortPreservingMergeExec` merges by `order` if it has
/// several. `Ok(None)` if `input` declares another order. An input that
/// declares no order at all is sorted by `order` as the plan says, as
/// DataFusion's join takes it: the native join checks it as the rows come
/// (see [`Pruning::sort_rows`]).
fn sorted_plan(
    input: &Arc<dyn ExecutionPlan>,
    order: &LexOrdering,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let stripped = without_reordering(input)?;
    let declared = stripped
        .equivalence_properties()
        .ordering_satisfy(order.clone())?;
    let undeclared =
        input.output_ordering().is_none() && stripped.output_ordering().is_none();
    if !declared && !undeclared {
        return Ok(None);
    }
    Ok(Some(
        match stripped.output_partitioning().partition_count() > 1 {
            true => Arc::new(SortPreservingMergeExec::new(order.clone(), stripped)),
            false => stripped,
        },
    ))
}

/// A hash join that streams both of its inputs, with the semantics of
/// DataFusion's `SymmetricHashJoinExec`: its join types, null equality and
/// residual filter, and exactly its output schema (and an embedded projection,
/// for the sort-merge joins of inputs that never end).
///
/// See the [module documentation](self) for how it reads its inputs.
#[derive(Debug)]
pub struct PhysicalSymmetricHashJoin {
    /// The left and the right input.
    children: Vec<Arc<dyn PhysicalOperator>>,
    spec: SymmetricJoinSpec,
}

/// Everything about a symmetric hash join that does not depend on its
/// inputs, only on their schemas.
#[derive(Debug)]
struct SymmetricJoinSpec {
    output: JoinOutput,
    /// The join keys: `(left key, right key)`.
    on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    filter: Option<JoinFilter>,
    null_equality: NullEquality,
    /// The fields of the keys in the row format, in which equal keys have
    /// equal bytes.
    key_fields: Vec<SortField>,
    /// The bytes that gathering the rows of the left and of the right input
    /// takes, for the filter and the output.
    widths: [GatherWidths; 2],
    /// If the inputs are read sorted by expressions of the filter: how the
    /// join determines their rows by the bounds of the values still to come.
    sorted_filter: Option<SortedFilter>,
}

impl SymmetricJoinSpec {
    fn try_new(
        left: &Schema,
        right: &Schema,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
        null_equality: NullEquality,
    ) -> Result<Self> {
        if on.is_empty() {
            return plan_err!("PhysicalSymmetricHashJoin requires at least one join key");
        }
        let mut key_fields = Vec::with_capacity(on.len());
        for (i, (left_key, right_key)) in on.iter().enumerate() {
            let left_type = left_key.data_type(left)?;
            let right_type = right_key.data_type(right)?;
            if left_type != right_type {
                return plan_err!(
                    "PhysicalSymmetricHashJoin requires join keys of the same type, key {i} has types {left_type} and {right_type}"
                );
            }
            // Unions as DataFusion compares them (see `comparable_unions`).
            key_fields.push(SortField::new(comparable_type(&left_type)));
        }
        if !RowConverter::supports_fields(&key_fields) {
            return plan_err!("The row format does not support the keys of the join");
        }
        if let Some(filter) = &filter
            && let Some(column) =
                filter
                    .column_indices()
                    .iter()
                    .find(|column| match column.side {
                        JoinSide::Left => column.index >= left.fields().len(),
                        JoinSide::Right => column.index >= right.fields().len(),
                        JoinSide::None => true,
                    })
        {
            return plan_err!(
                "The filter of a PhysicalSymmetricHashJoin reads column {} of side {:?}, which its input does not have",
                column.index,
                column.side
            );
        }
        let output = JoinOutput::try_new(left, right, join_type, projection)?;
        let widths =
            [(JoinSide::Left, left), (JoinSide::Right, right)].map(|(side, schema)| {
                let filter_columns = filter
                    .iter()
                    .flat_map(|filter| filter.column_indices())
                    .filter(|column| column.side == side)
                    .map(|column| column.index);
                let output_columns = output
                    .column_indices
                    .iter()
                    .filter(|column| column.side == side)
                    .map(|column| column.index);
                GatherWidths::new(schema, filter_columns, output_columns)
            });
        Ok(Self {
            output,
            on,
            filter,
            null_equality,
            key_fields,
            widths,
            sorted_filter: None,
        })
    }

    /// The specification of the symmetric join of `join`, `None` if the
    /// symmetric join does not implement it.
    fn of(join: &EquiJoin<'_>) -> Result<Option<Self>> {
        if join.null_aware {
            return Ok(None);
        }
        let spec = match Self::try_new(
            &join.left.schema(),
            &join.right.schema(),
            join.on.to_vec(),
            join.filter.cloned(),
            join.join_type,
            join.projection.map(<[usize]>::to_vec),
            join.null_equality,
        ) {
            Ok(spec) => spec,
            Err(DataFusionError::NotImplemented(_) | DataFusionError::Plan(_)) => {
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        if spec.output.schema != join.plan.schema() {
            return internal_err!(
                "PhysicalSymmetricHashJoin would have schema {:?}, but {} has schema {:?}",
                spec.output.schema,
                join.plan.name(),
                join.plan.schema()
            );
        }
        Ok(Some(spec))
    }

    fn join_type(&self) -> JoinType {
        self.output.join_type
    }

    /// The keys of the input `side` (0: left, 1: right).
    fn keys(&self, side: usize) -> impl Iterator<Item = PhysicalExprRef> + '_ {
        self.on.iter().map(move |(left, right)| {
            Arc::clone(match side {
                0 => left,
                _ => right,
            })
        })
    }

    /// Whether the join emits the rows of input `side` once they are
    /// determined: those without a match (outer and anti joins), or all of
    /// them with their mark (mark joins).
    fn emits_final_rows(&self, side: usize) -> bool {
        matches!(
            (self.join_type(), side),
            (JoinType::Full, _)
                | (JoinType::Left | JoinType::LeftAnti | JoinType::LeftMark, 0)
                | (
                    JoinType::Right | JoinType::RightAnti | JoinType::RightMark,
                    1
                )
        )
    }

    /// Whether the join emits the rows of input `side` when they match for
    /// the first time (semi joins).
    fn emits_first_matches(&self, side: usize) -> bool {
        matches!(
            (self.join_type(), side),
            (JoinType::LeftSemi, 0) | (JoinType::RightSemi, 1)
        )
    }

    /// Whether the join emits pairs of rows.
    fn emits_pairs(&self) -> bool {
        matches!(
            self.join_type(),
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
        )
    }

    fn key_converter(&self) -> Result<RowConverter> {
        Ok(RowConverter::new(self.key_fields.clone())?)
    }
}

/// How a symmetric join of inputs sorted by expressions of its filter
/// determines their rows (see the [module documentation](self)), as
/// DataFusion's `SymmetricHashJoinExec` prunes them: the interval graph of
/// the filter, in which the first expression of the order of either input is
/// a leaf (as built: every determination solves a copy of it).
#[derive(Debug, Clone)]
struct SortedFilter {
    graph: ExprIntervalGraph,
    /// The first expression of the order of the left and of the right input.
    sides: [SortedExpr; 2],
}

/// The first expression of the order of an input of a symmetric join, see
/// [`SortedFilter`].
#[derive(Debug, Clone)]
struct SortedExpr {
    /// The expression, over the columns of the input, and its order.
    order: PhysicalSortExpr,
    /// The node of the expression, over the columns of the filter, in the
    /// interval graph of the filter.
    node: usize,
    /// The type of its values.
    data_type: DataType,
}

impl SortedFilter {
    /// The determination of the rows of a join with the filter `filter`, of
    /// inputs with the schemas `schemas` sorted by `orders`: `None` without a
    /// filter, if interval arithmetic does not support the filter (see
    /// [`check_support`]), or if the first expression of either order is not
    /// an expression of the filter.
    fn try_new(
        filter: Option<&JoinFilter>,
        schemas: [&SchemaRef; 2],
        orders: [&LexOrdering; 2],
    ) -> Result<Option<Self>> {
        let Some(filter) = filter else {
            return Ok(None);
        };
        if !check_support(filter.expression(), filter.schema()) {
            return Ok(None);
        }
        // Interval arithmetic bounds real numbers, but the arithmetic of floats
        // yields NaN (of a NaN still to come, of `inf - inf`, `0 * inf`, ...),
        // which the comparisons of the filter order after (or, negative,
        // before) every number: arithmetic of floats, also of floats that the
        // filter computes (e.g. a cast of integers), would make it rule out
        // rows that the filter matches. The comparisons of floats themselves
        // order NaN as the order of the inputs does. The values are compared
        // with `-0.0` as `+0.0` (see `Pruning`), but interval arithmetic orders
        // `-0.0` before `+0.0`: the literal `-0.0` of floats would do so too.
        let floats = has_float_values(filter.expression(), filter.schema());
        if floats
            && (has_float_arithmetic(filter.expression(), filter.schema())
                || may_negate_zeros(filter.expression()))
        {
            return Ok(None);
        }
        let mut exprs = Vec::with_capacity(2);
        for (side, (schema, order)) in [JoinSide::Left, JoinSide::Right]
            .into_iter()
            .zip(schemas.into_iter().zip(orders))
        {
            let order = order.first();
            let Some(expr) = filter_expression(filter, side, &order.expr)? else {
                return Ok(None);
            };
            exprs.push((order.clone(), order.expr.data_type(schema)?, expr));
        }
        let Ok(mut graph) =
            ExprIntervalGraph::try_new(Arc::clone(filter.expression()), filter.schema())
        else {
            return Ok(None);
        };
        let filter_exprs = exprs
            .iter()
            .map(|(_, _, expr)| Arc::clone(expr))
            .collect::<Vec<_>>();
        let nodes = graph.gather_node_indices(&filter_exprs);
        if nodes.iter().any(|&(_, node)| node == usize::MAX) {
            return Ok(None);
        }
        let mut sides =
            exprs
                .into_iter()
                .zip(nodes)
                .map(|((order, data_type, _), (_, node))| SortedExpr {
                    order,
                    node,
                    data_type,
                });
        let (Some(left), Some(right)) = (sides.next(), sides.next()) else {
            return internal_err!("A symmetric hash join without two inputs");
        };
        Ok(Some(Self {
            graph,
            sides: [left, right],
        }))
    }
}

/// `expr`, an expression over the columns of the input `side` of a join, over
/// the columns of its filter `filter`, if the filter has all its columns and
/// the filter expression has it as a subexpression (as DataFusion's
/// `convert_sort_expr_with_filter_schema`).
fn filter_expression(
    filter: &JoinFilter,
    side: JoinSide,
    expr: &PhysicalExprRef,
) -> Result<Option<PhysicalExprRef>> {
    let filter_columns = filter
        .column_indices()
        .iter()
        .enumerate()
        .filter(|(_, column)| column.side == side)
        .map(|(index, column)| {
            let name = filter.schema().field(index).name();
            (column.index, Column::new(name, index))
        })
        .collect::<HashMap<_, _>>();
    let mut complete = true;
    let converted = Arc::clone(expr)
        .transform_up(|node| {
            let Some(column) = node.downcast_ref::<Column>() else {
                return Ok(Transformed::no(node));
            };
            Ok(match filter_columns.get(&column.index()) {
                Some(filter_column) => {
                    Transformed::yes(Arc::new(filter_column.clone()) as PhysicalExprRef)
                }
                None => {
                    complete = false;
                    Transformed::no(node)
                }
            })
        })
        .data()?;
    let contains = |expr: &PhysicalExprRef| has_subexpression(expr, &converted);
    Ok((complete && contains(filter.expression())).then_some(converted))
}

/// Whether `expr`, or an expression below it, evaluates to floats (of the
/// columns of `schema`).
fn has_float_values(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    expr.data_type(schema)
        .is_ok_and(|data_type| data_type.is_floating())
        || expr
            .children()
            .into_iter()
            .any(|child| has_float_values(child, schema))
}

/// Whether `expr`, or an expression below it, is arithmetic of floats (of the
/// columns of `schema`): a negation, or a sum, difference, product, quotient
/// or remainder of floats or into floats.
fn has_float_arithmetic(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    let floating = |expr: &PhysicalExprRef| {
        expr.data_type(schema)
            .is_ok_and(|data_type| data_type.is_floating())
    };
    let arithmetic = match expr.downcast_ref::<BinaryExpr>() {
        Some(binary) => {
            binary.op().is_numerical_operators()
                && (floating(expr) || floating(binary.left()) || floating(binary.right()))
        }
        None => expr.is::<NegativeExpr>() && floating(expr),
    };
    arithmetic
        || expr
            .children()
            .into_iter()
            .any(|child| has_float_arithmetic(child, schema))
}

/// Whether `expr` may evaluate to `-0.0` for operands that are `+0.0`: it has a
/// negation, a product, a quotient or a literal `-0.0`.
fn may_negate_zeros(expr: &PhysicalExprRef) -> bool {
    let negates = expr.is::<NegativeExpr>()
        || expr.downcast_ref::<BinaryExpr>().is_some_and(|binary| {
            matches!(binary.op(), Operator::Multiply | Operator::Divide)
        })
        || expr
            .downcast_ref::<Literal>()
            .is_some_and(|literal| match literal.value() {
                ScalarValue::Float16(Some(value)) => value.is_sign_negative(),
                ScalarValue::Float32(Some(value)) => value.is_sign_negative(),
                ScalarValue::Float64(Some(value)) => value.is_sign_negative(),
                _ => false,
            });
    negates || expr.children().into_iter().any(may_negate_zeros)
}

/// Whether `expr` is `subexpression`, or has it below it.
fn has_subexpression(expr: &PhysicalExprRef, subexpression: &PhysicalExprRef) -> bool {
    expr.eq(subexpression)
        || expr
            .children()
            .into_iter()
            .any(|child| has_subexpression(child, subexpression))
}

/// The rows of one input of a symmetric hash join, and their hash table. The
/// rows have positions in the order in which they arrived; the rows before
/// `determined` are determined (no row of the other input still to come can
/// match them), and those before `emitted` emitted (their final rows, if
/// any). The table drops the batches of emitted rows: it has the rows from
/// `base` on.
#[derive(Debug)]
struct SideTable {
    /// The rows at the positions from `base` on (at `position - base` in
    /// `rows`).
    rows: StoredRows,
    /// The position of the first row of `rows`.
    base: usize,
    /// The first row that is not determined.
    determined: usize,
    /// The first row that the join did not emit the final row of (or that
    /// it did not skip, if it has none).
    emitted: usize,
    /// The last row of every key (whose keys can match).
    heads: HashMap<Box<[u8]>, usize>,
    /// For every row, the row before it with the same key ([`NO_ROW`] for
    /// none).
    previous: VecDeque<usize>,
    /// For every row, whether it matched a row of the other input.
    matched: VecDeque<bool>,
    /// For every row, the bytes of its values of their own, for the filter
    /// and the output (see [`GatherWidths::variable_bytes`]).
    variable_bytes: VecDeque<[u32; 2]>,
    /// For every batch of `rows`, if the join determines rows by their order:
    /// the values of the order of its rows, in the row format.
    sort_rows: VecDeque<Rows>,
    /// For every batch of `rows`: the memory of the batch, and of the values
    /// of the order of its rows.
    batch_bytes: VecDeque<usize>,
    /// The sum of `batch_bytes`.
    batches_bytes: usize,
    /// The bytes of the keys of `heads` (each a copy of its own).
    key_bytes: usize,
}

impl SideTable {
    fn new(schema: SchemaRef) -> Self {
        Self {
            rows: StoredRows::new(schema),
            base: 0,
            determined: 0,
            emitted: 0,
            heads: HashMap::new(),
            previous: VecDeque::new(),
            matched: VecDeque::new(),
            variable_bytes: VecDeque::new(),
            sort_rows: VecDeque::new(),
            batch_bytes: VecDeque::new(),
            batches_bytes: 0,
            key_bytes: 0,
        }
    }

    /// The position after the last row.
    fn end(&self) -> usize {
        self.base + self.rows.len()
    }

    /// The memory of the table, as allocated: of its batches (and of the
    /// values of the order of their rows), of its hash table (its buckets and
    /// its keys) and of the structures of its rows (their capacities, which
    /// dropping rows does not free, see [`Self::shrink`]).
    fn memory(&self) -> usize {
        self.batches_bytes
            + self.key_bytes
            + self.heads.allocation_size()
            + self.previous.capacity() * size_of::<usize>()
            + self.matched.capacity() * size_of::<bool>()
            + self.variable_bytes.capacity() * size_of::<[u32; 2]>()
            + self.sort_rows.capacity() * size_of::<Rows>()
            + self.batch_bytes.capacity() * size_of::<usize>()
            + self.rows.structure_bytes(0)
    }

    /// An upper bound of the memory of the table while it appends (see
    /// [`Self::append`]) a batch of `rows` rows whose batch (and the values of
    /// the order of its rows) take `bytes` bytes, with keys of `key_bytes`
    /// bytes at most: its structures grow to hold them, with the old ones
    /// while they grow.
    fn memory_appending(&self, rows: usize, bytes: usize, key_bytes: usize) -> usize {
        let heads = self.heads.allocation_size();
        let heads = match self.heads.len() + rows > self.heads.capacity() {
            // The buckets double (at least) as they grow: the old ones are at
            // most half the new ones as they move.
            true => {
                let grown =
                    (2 * heads).max(hash_map_bytes_holding(self.heads.len() + rows));
                grown + grown / 2
            }
            false => heads,
        };
        let rows_of = |len: usize, capacity: usize, size: usize| {
            deque_bytes_appending(len, capacity, rows, size)
        };
        self.batches_bytes
            + bytes
            + self.key_bytes
            + key_bytes
            + heads
            + rows_of(
                self.previous.len(),
                self.previous.capacity(),
                size_of::<usize>(),
            )
            + rows_of(
                self.matched.len(),
                self.matched.capacity(),
                size_of::<bool>(),
            )
            + rows_of(
                self.variable_bytes.len(),
                self.variable_bytes.capacity(),
                size_of::<[u32; 2]>(),
            )
            + deque_bytes_appending(
                self.sort_rows.len(),
                self.sort_rows.capacity(),
                1,
                size_of::<Rows>(),
            )
            + deque_bytes_appending(
                self.batch_bytes.len(),
                self.batch_bytes.capacity(),
                1,
                size_of::<usize>(),
            )
            + self.rows.structure_bytes(1)
    }

    /// Append `batch`, whose keys are `keys` (of which those of the rows of
    /// `matchable` can match, all if `None`), the bytes of the values of its
    /// rows of their own `variable_bytes`, and the values of the order of its
    /// rows `sort_rows` (if the join determines rows by them).
    fn append(
        &mut self,
        batch: RecordBatch,
        keys: &Rows,
        matchable: Option<&NullBuffer>,
        variable_bytes: Vec<[u32; 2]>,
        sort_rows: Option<Rows>,
    ) {
        let rows = batch.num_rows();
        let first = self.end();
        self.previous.reserve(rows);
        self.matched.reserve(rows);
        self.variable_bytes.reserve(rows);
        for row in 0..rows {
            let position = first + row;
            let previous = match matchable.is_none_or(|valid| valid.is_valid(row)) {
                true => {
                    let key = keys.row(row);
                    match self.heads.get_mut(key.as_ref()) {
                        Some(head) => std::mem::replace(head, position),
                        None => {
                            // Only the keys that are new to the table are
                            // copied into it.
                            self.key_bytes += key.as_ref().len();
                            self.heads.insert(key.as_ref().into(), position);
                            NO_ROW
                        }
                    }
                }
                false => NO_ROW,
            };
            self.previous.push_back(previous);
        }
        self.matched.extend(std::iter::repeat_n(false, rows));
        self.variable_bytes.extend(variable_bytes);
        let bytes = memory_size(&batch) + sort_rows.as_ref().map_or(0, Rows::size);
        self.batches_bytes += bytes;
        self.batch_bytes.push_back(bytes);
        self.sort_rows.extend(sort_rows);
        self.rows.push(batch);
    }

    /// Drop the first `batches` batches, with the keys of the hash table whose
    /// rows they all are: `keys` encodes the keys of the rows of a batch (and
    /// reserves them while they are looked up).
    fn drop_batches(
        &mut self,
        batches: usize,
        mut keys: impl FnMut(&RecordBatch) -> Result<(Rows, MemoryReservation)>,
    ) -> Result<()> {
        let starts = self.rows.batch_starts();
        let rows = starts.get(batches).copied().unwrap_or(self.rows.len());
        for (batch, &start) in self.rows.batches()[..batches].iter().zip(starts) {
            let (keys, _reservation) = keys(batch)?;
            for row in 0..batch.num_rows() {
                let key = keys.row(row);
                let position = self.base + start + row;
                if self.heads.get(key.as_ref()) == Some(&position) {
                    self.heads.remove(key.as_ref());
                    self.key_bytes -= key.as_ref().len();
                }
            }
        }
        self.rows = self.rows.without_first_batches(batches);
        self.base += rows;
        self.previous.drain(..rows);
        self.matched.drain(..rows);
        self.variable_bytes.drain(..rows);
        let sort_rows = batches.min(self.sort_rows.len());
        self.sort_rows.drain(..sort_rows);
        self.batches_bytes -= self.batch_bytes.drain(..batches).sum::<usize>();
        Ok(())
    }

    /// About the buckets of the hash table (each with an entry and a control
    /// byte).
    fn heads_buckets(&self) -> usize {
        self.heads.allocation_size() / (HEAD_BYTES + 1)
    }

    /// The memory that [`Self::shrink`] allocates: of the smaller structures,
    /// while they take the entries of the larger ones.
    fn shrink_bytes(&self) -> usize {
        let deque = |len: usize, capacity: usize, size: usize| match is_oversized(
            len, capacity,
        ) {
            true => 2 * len * size,
            false => 0,
        };
        let heads = match is_oversized(self.heads.len(), self.heads_buckets()) {
            true => hash_map_bytes_holding(2 * self.heads.len()),
            false => 0,
        };
        heads
            + deque(
                self.previous.len(),
                self.previous.capacity(),
                size_of::<usize>(),
            )
            + deque(
                self.matched.len(),
                self.matched.capacity(),
                size_of::<bool>(),
            )
            + deque(
                self.variable_bytes.len(),
                self.variable_bytes.capacity(),
                size_of::<[u32; 2]>(),
            )
            + deque(
                self.sort_rows.len(),
                self.sort_rows.capacity(),
                size_of::<Rows>(),
            )
            + deque(
                self.batch_bytes.len(),
                self.batch_bytes.capacity(),
                size_of::<usize>(),
            )
    }

    /// Free the capacity of the structures that hold far fewer entries than
    /// they can (once rows were dropped, see [`is_oversized`]): they shrink to
    /// twice their entries.
    fn shrink(&mut self) {
        if is_oversized(self.heads.len(), self.heads_buckets()) {
            self.heads.shrink_to(2 * self.heads.len());
        }
        fn shrink_deque<T>(deque: &mut VecDeque<T>) {
            if is_oversized(deque.len(), deque.capacity()) {
                deque.shrink_to(2 * deque.len());
            }
        }
        shrink_deque(&mut self.previous);
        shrink_deque(&mut self.matched);
        shrink_deque(&mut self.variable_bytes);
        shrink_deque(&mut self.sort_rows);
        shrink_deque(&mut self.batch_bytes);
    }

    /// The first row at or after the position `from` whose value of the order
    /// (in `sort_rows`) does not come before `bound`, as the rows are sorted.
    fn first_not_before(&self, from: usize, bound: Row<'_>) -> usize {
        let starts = self.rows.batch_starts();
        let from = from - self.base;
        let mut batch = starts
            .partition_point(|&start| start <= from)
            .saturating_sub(1);
        while let (Some(rows), Some(&start)) =
            (self.sort_rows.get(batch), starts.get(batch))
        {
            let len = rows.num_rows();
            if len > 0 && rows.row(len - 1) < bound {
                batch += 1;
                continue;
            }
            let (mut low, mut high) = (from.saturating_sub(start).min(len), len);
            while low < high {
                let mid = low + (high - low) / 2;
                if rows.row(mid) < bound {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            return self.base + start + low;
        }
        self.end()
    }
}

/// The marker of the end of a chain of rows of a [`SideTable`].
const NO_ROW: usize = usize::MAX;

/// The join reads at most this many chunks of its inputs that produce no
/// output in a call of `get_data` before it lets the other tasks run.
const CHUNKS_PER_CALL: usize = 16;

/// The join joins at most this many parts of the candidates of its chunks
/// that produce no output (each of at most [`DEFAULT_VECTOR_SIZE`] candidate
/// pairs, see [`PhysicalSymmetricHashJoin::next_chunk_output`]) in a call of
/// `get_data` before it lets the other tasks run.
const PARTS_PER_CALL: usize = 16;

/// The bytes of the index of a final row, and of its mark.
const FINAL_ROW_INDEX_BYTES: usize = size_of::<Option<usize>>() + 1;

/// A join that determines rows by the order of its inputs reads at most this
/// many chunks of an input more than of the other one (unless the other one
/// ended), as DataFusion's join reads them alternately: the rows of an input
/// are determined as the other input advances, so that one that is read ahead
/// of the other keeps its rows in its table.
const MAX_CHUNKS_AHEAD: usize = 2;

/// The global source state of a [`PhysicalSymmetricHashJoin`]: everything
/// about its execution, read by a single thread.
struct SymmetricJoinSource {
    run: Mutex<SymmetricRun>,
}

impl GlobalSourceState for SymmetricJoinSource {}

/// The execution of a [`PhysicalSymmetricHashJoin`].
struct SymmetricRun {
    /// The left and the right input, read as streams.
    inputs: [NestedInput; 2],
    /// The rows of the left and of the right input read so far.
    tables: [SideTable; 2],
    /// Encodes the keys of both inputs.
    converter: RowConverter,
    /// The chunk being joined.
    job: Option<ChunkJoin>,
    /// The input to read the next chunk of first (the inputs alternate).
    next_side: usize,
    /// The chunks read of the left and of the right input.
    chunks_read: [usize; 2],
    /// If the join determines rows by the order of its inputs: the interval
    /// graph of its filter, and the orders of the rows read so far.
    pruning: Option<Pruning>,
    /// The memory of the tables.
    reservation: MemoryReservation,
}

/// A chunk of rows of an input being joined with the rows of the other input
/// read before it (its rows are in the table of their input already).
struct ChunkJoin {
    /// The input of the chunk: 0 (left) or 1 (right).
    side: usize,
    batch: RecordBatch,
    /// The keys of the rows, encoded.
    keys: Rows,
    /// The rows whose keys can match, `None` if all.
    matchable: Option<NullBuffer>,
    /// The position of the first row of the chunk in the table of its input.
    first: usize,
    /// The next row to join.
    next_row: usize,
    /// The row whose candidates did not all fit into the last output, and its
    /// next candidate (a row of the other input).
    chain: Option<(usize, usize)>,
    /// The memory of `keys` and `matchable`, held while the chunk is joined.
    _reservation: MemoryReservation,
}

impl ChunkJoin {
    fn is_matchable(&self, row: usize) -> bool {
        self.matchable
            .as_ref()
            .is_none_or(|matchable| matchable.is_valid(row))
    }

    /// Whether all rows of the chunk were joined.
    fn is_done(&self) -> bool {
        self.chain.is_none() && self.next_row >= self.batch.num_rows()
    }
}

/// The determination of the rows of a symmetric join by the order of its
/// inputs, in an execution (see [`SortedFilter`]). The values of the orders
/// and the bounds are compared with `-0.0` as `+0.0`, as the comparisons of
/// the filter do (see `normalize_floats`).
struct Pruning {
    filter: SortedFilter,
    /// Encode the values of the order of the left and of the right input
    /// (in their order).
    converters: [RowConverter; 2],
    /// The value of the order of the last row of the left and of the right
    /// input, as a value and in the row format.
    last: [Option<(ScalarValue, OwnedRow)>; 2],
}

impl Pruning {
    fn try_new(filter: SortedFilter) -> Result<Self> {
        let converter = |sorted: &SortedExpr| {
            RowConverter::new(vec![SortField::new_with_options(
                sorted.data_type.clone(),
                sorted.order.options,
            )])
        };
        let converters = [converter(&filter.sides[0])?, converter(&filter.sides[1])?];
        Ok(Self {
            filter,
            converters,
            last: [None, None],
        })
    }

    /// The values of the order of the input `side` of the rows of `batch`,
    /// in the row format, after checking that they are sorted (also after the
    /// rows before them); remembers the value of the last row.
    fn sort_rows(&mut self, side: usize, batch: &RecordBatch) -> Result<Rows> {
        let order = &self.filter.sides[side].order;
        // `-0.0` is `+0.0`, as the comparisons of the filter take it.
        let values =
            normalize_floats(&order.expr.evaluate(batch)?.into_array(batch.num_rows())?);
        let rows = self.converters[side].convert_columns(&[Arc::clone(&values)])?;
        let after_last = match (&self.last[side], rows.num_rows()) {
            (Some((_, last)), 1..) => last.row() <= rows.row(0),
            _ => true,
        };
        if !after_last
            || (1..rows.num_rows()).any(|row| rows.row(row) < rows.row(row - 1))
        {
            let input = ["left", "right"][side];
            return internal_err!(
                "The {input} input of PhysicalSymmetricHashJoin is not sorted by {order}"
            );
        }
        if let Some(last) = rows.num_rows().checked_sub(1) {
            let value = ScalarValue::try_from_array(&values, last)?;
            self.last[side] = Some((value, rows.row(last).owned()));
        }
        Ok(rows)
    }

    /// The first row of `table`, the table of input `side`, that the rows of
    /// the other input still to come may match: the interval of the values of
    /// the order of the rows of `side` that can match, with the filter, the
    /// rows of the other input whose values of its order are not before the
    /// value of its last row (DataFusion's interval pruning). The rows of
    /// `side` before that interval are determined.
    fn first_undetermined(&mut self, side: usize, table: &SideTable) -> Result<usize> {
        let from = table.determined;
        let [own, other] = match side {
            0 => [&self.filter.sides[0], &self.filter.sides[1]],
            _ => [&self.filter.sides[1], &self.filter.sides[0]],
        };
        let first = self.value_at(side, table, from)?;
        let own_values = values_from(first, own)?;
        let other_values = match &self.last[1 - side] {
            Some((last, _)) => values_from(last.clone(), other)?,
            None => Interval::make_unbounded(&other.data_type)?,
        };
        let mut bounds = [(own.node, own_values), (other.node, other_values)];
        // Every step is solved from the graph as built: propagating the
        // constraints narrows the intervals of all its nodes, also of the other
        // columns of the filter, which hold for these bounds only (they would
        // rule out rows that later bounds let match).
        let mut graph = self.filter.graph.clone();
        match graph.update_ranges(&mut bounds, Interval::TRUE)? {
            // No row of `side` can match a row still to come.
            PropagationResult::Infeasible => return Ok(table.end()),
            PropagationResult::CannotPropagate => return Ok(from),
            PropagationResult::Success => {}
        }
        let [(_, matchable), _] = bounds;
        let bound = match own.order.options.descending {
            true => matchable.upper(),
            false => matchable.lower(),
        };
        if bound.is_null() || is_nan(bound) {
            return Ok(from);
        }
        let bound = normalize_floats(&bound.to_array()?);
        let bound = self.converters[side].convert_columns(&[bound])?;
        Ok(table.first_not_before(from, bound.row(0)))
    }

    /// The value of the order of the row at `position` of `table`, the table
    /// of input `side`.
    fn value_at(
        &self,
        side: usize,
        table: &SideTable,
        position: usize,
    ) -> Result<ScalarValue> {
        let index = position - table.base;
        let starts = table.rows.batch_starts();
        let batch = starts
            .partition_point(|&start| start <= index)
            .saturating_sub(1);
        let (Some(rows), Some(&start)) = (table.sort_rows.get(batch), starts.get(batch))
        else {
            return internal_err!("A row of PhysicalSymmetricHashJoin without its order");
        };
        let values = self.converters[side].convert_rows([rows.row(index - start)])?;
        ScalarValue::try_from_array(&values[0], 0)
    }
}

/// The interval of the values, in the order `sorted`, of the rows from a row
/// whose value is `value` on: not before `value`.
fn values_from(value: ScalarValue, sorted: &SortedExpr) -> Result<Interval> {
    if is_nan(&value) {
        return Interval::make_unbounded(&sorted.data_type);
    }
    let unbounded = ScalarValue::try_from(&sorted.data_type)?;
    match sorted.order.options.descending {
        false => Interval::try_new(value, unbounded),
        true => Interval::try_new(unbounded, value),
    }
}

/// Whether `value` is a floating point NaN, which intervals do not bound.
fn is_nan(value: &ScalarValue) -> bool {
    match value {
        ScalarValue::Float16(Some(value)) => value.is_nan(),
        ScalarValue::Float32(Some(value)) => value.is_nan(),
        ScalarValue::Float64(Some(value)) => value.is_nan(),
        _ => false,
    }
}

impl PhysicalSymmetricHashJoin {
    /// The left input.
    pub fn left(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The right input.
    pub fn right(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[1]
    }

    /// The join type.
    pub fn join_type(&self) -> JoinType {
        self.spec.join_type()
    }

    /// Whether the join reads its inputs sorted by expressions of its filter,
    /// and determines their rows by the bounds of the values still to come
    /// (see the [module documentation](self)).
    pub fn determines_rows_by_order(&self) -> bool {
        self.spec.sorted_filter.is_some()
    }

    /// Append the rows of `batch`, a chunk of the input `side`, to the table of
    /// their input, reserving their memory, and start joining them.
    fn start_chunk(
        &self,
        run: &mut SymmetricRun,
        side: usize,
        batch: RecordBatch,
    ) -> Result<ChunkJoin> {
        // The table of the side keeps the rows of the chunk.
        let batch = compact_to_keep(batch)?;
        // The computed keys, and their encoding, are reserved before they are
        // encoded (see `key_memory`), and the encoded keys while the chunk is
        // joined.
        let reservation = run.reservation.new_empty();
        let arrays = evaluate_keys(self.spec.keys(side), &batch, &reservation)?;
        reservation.try_resize(key_memory(&arrays, &batch))?;
        let matchable = matchable_rows(&arrays, self.spec.null_equality);
        let keys = run.converter.convert_columns(&arrays)?;
        drop(arrays);
        let matchable_bytes = matchable.as_ref().map_or(0, |nulls| nulls.buffer().len());
        reservation.try_resize(keys.size() + matchable_bytes)?;
        let num_rows = batch.num_rows();
        // The bytes of the rows are counted with their scratch reserved.
        let scratch = run.reservation.new_empty();
        scratch.try_grow(self.spec.widths[side].scratch_bytes(&batch))?;
        let variable_bytes = self.spec.widths[side].variable_bytes(&batch);
        drop(scratch);
        let sort_rows = match &mut run.pruning {
            Some(pruning) => Some(pruning.sort_rows(side, &batch)?),
            None => None,
        };
        // The tables are reserved as they are (see `SideTable::memory`): as the
        // table grows to hold the rows, and then as it holds them.
        let other = run.tables[1 - side].memory();
        let table = &mut run.tables[side];
        let first = table.end();
        let bytes = memory_size(&batch) + sort_rows.as_ref().map_or(0, Rows::size);
        run.reservation
            .try_resize(other + table.memory_appending(num_rows, bytes, keys.size()))?;
        table.append(
            batch.clone(),
            &keys,
            matchable.as_ref(),
            variable_bytes,
            sort_rows,
        );
        run.reservation.try_resize(other + table.memory())?;
        Ok(ChunkJoin {
            side,
            batch,
            keys,
            matchable,
            first,
            next_row: 0,
            chain: None,
            _reservation: reservation,
        })
    }

    /// The next output of the join of the rows of `job` with the rows of the
    /// other input in `tables` that are not determined: of at most
    /// [`DEFAULT_VECTOR_SIZE`] candidate pairs (a row without candidates
    /// counts as one) and about [`PART_BYTES`] bytes of their rows, with the
    /// filter evaluated on all of them at once. The rows that it gathers are
    /// reserved while it gathers them. `None` if they produce no output.
    fn next_chunk_output(
        &self,
        job: &mut ChunkJoin,
        tables: &mut [SideTable; 2],
        reservation: &MemoryReservation,
    ) -> Result<Option<RecordBatch>> {
        let spec = &self.spec;
        let side = job.side;
        let [left, right] = tables;
        let (own, other) = match side {
            0 => (left, right),
            _ => (right, left),
        };
        let (own_widths, other_widths) = (&spec.widths[side], &spec.widths[1 - side]);
        // The bytes that a candidate pair gathers, for the filter and for the
        // output (at most all candidates pass the filter), and its indices.
        let pair_bytes = |own_index: usize, other_index: usize| {
            let (own_bytes, other_bytes) = (
                own.variable_bytes[own_index],
                other.variable_bytes[other_index],
            );
            let filter = match spec.filter {
                Some(_) => {
                    own_widths.row_bytes(own_bytes, FILTER)
                        + other_widths.row_bytes(other_bytes, FILTER)
                }
                None => 0,
            };
            filter
                + own_widths.row_bytes(own_bytes, OUTPUT)
                + other_widths.row_bytes(other_bytes, OUTPUT)
                + PAIR_INDEX_BYTES
        };
        let num_rows = job.batch.num_rows();
        let own_first = job.first - own.base;
        let mut pairs_own = Vec::new();
        let mut pairs_other = Vec::new();
        let mut cost = 0;
        let mut bytes = 0;
        while cost < DEFAULT_VECTOR_SIZE && bytes < PART_BYTES {
            let (row, mut position) = match job.chain.take() {
                Some(chain) => chain,
                None => {
                    let row = job.next_row;
                    if row >= num_rows {
                        break;
                    }
                    job.next_row += 1;
                    cost += 1;
                    if !job.is_matchable(row) {
                        continue;
                    }
                    let head = other.heads.get(job.keys.row(row).as_ref());
                    (row, head.copied().unwrap_or(NO_ROW))
                }
            };
            // The determined rows of the other input (before its first row
            // that is not, as the chain goes back) match no row any more.
            let candidate =
                |position: usize| position != NO_ROW && position >= other.determined;
            while candidate(position) && cost < DEFAULT_VECTOR_SIZE && bytes < PART_BYTES
            {
                let index = position - other.base;
                pairs_own.push(row as u32);
                pairs_other.push(Some(index));
                bytes += pair_bytes(own_first + row, index);
                position = other.previous[index];
                cost += 1;
            }
            if candidate(position) {
                job.chain = Some((row, position));
                break;
            }
        }
        if pairs_other.is_empty() {
            return Ok(None);
        }
        // The rows gathered for the filter and the output are reserved while
        // they are joined.
        let transient = reservation.new_empty();
        transient.try_grow(bytes)?;
        let own_rows = UInt32Array::from(pairs_own);
        let mask = match &spec.filter {
            Some(filter) => {
                let (left, right) = sides(
                    side,
                    JoinRows::Batch(&job.batch, &own_rows),
                    JoinRows::Stored(&other.rows, &pairs_other),
                );
                Some(filter_mask(filter, &left, &right, pairs_other.len())?)
            }
            None => None,
        };

        // The matches, and the rows that match for the first time.
        let mut out_own = Vec::new();
        let mut out_other = Vec::new();
        let mut first_own = Vec::new();
        let mut first_other = Vec::new();
        for (pair, (&row, &index)) in
            own_rows.values().iter().zip(&pairs_other).enumerate()
        {
            if mask.as_ref().is_some_and(|mask| !mask.value(pair)) {
                continue;
            }
            let Some(index) = index else {
                continue;
            };
            let own_index = own_first + row as usize;
            if !own.matched[own_index] {
                own.matched[own_index] = true;
                first_own.push(row);
            }
            if !other.matched[index] {
                other.matched[index] = true;
                first_other.push(Some(index));
            }
            if spec.emits_pairs() {
                out_own.push(Some(row));
                out_other.push(Some(index));
            }
        }

        // The output: the pairs, or the rows of a semi join that matched for
        // the first time (of the input of the chunk, or of the other one).
        let other_schema = self.children[1 - side].schema();
        let own_schema = self.children[side].schema();
        if spec.emits_pairs() && !out_own.is_empty() {
            let own_rows = UInt32Array::from(out_own);
            let (left, right) = sides(
                side,
                JoinRows::Batch(&job.batch, &own_rows),
                JoinRows::Stored(&other.rows, &out_other),
            );
            return Ok(Some(spec.output.batch(
                &left,
                &right,
                None,
                own_rows.len(),
            )?));
        }
        if spec.emits_first_matches(side) && !first_own.is_empty() {
            let num_rows = first_own.len();
            let own_rows = UInt32Array::from(first_own);
            let (left, right) = sides(
                side,
                JoinRows::Batch(&job.batch, &own_rows),
                JoinRows::Nulls(&other_schema, num_rows),
            );
            return Ok(Some(spec.output.batch(&left, &right, None, num_rows)?));
        }
        if spec.emits_first_matches(1 - side) && !first_other.is_empty() {
            let num_rows = first_other.len();
            let (left, right) = sides(
                1 - side,
                JoinRows::Stored(&other.rows, &first_other),
                JoinRows::Nulls(&own_schema, num_rows),
            );
            return Ok(Some(spec.output.batch(&left, &right, None, num_rows)?));
        }
        Ok(None)
    }

    /// Determine more rows of input `side`: all of them once the other input
    /// ended, or those that the bounds of the values of the other input still
    /// to come rule out (see [`Pruning::first_undetermined`]).
    fn determine(&self, run: &mut SymmetricRun, side: usize) {
        let table = &run.tables[side];
        let (from, end) = (table.determined, table.end());
        if from == end {
            return;
        }
        let determined = match &mut run.pruning {
            _ if run.inputs[1 - side].is_done() => Ok(end),
            Some(pruning) => pruning.first_undetermined(side, table),
            None => Ok(from),
        };
        run.tables[side].determined = determined.unwrap_or_else(|_| {
            // The bounds cannot be computed: the rows are determined once the
            // other input ends.
            run.pruning = None;
            from
        });
    }

    /// The next output of the determined rows that the join did not emit yet:
    /// of at most [`DEFAULT_VECTOR_SIZE`] rows and about [`PART_BYTES`] bytes,
    /// the rows without a match (with NULLs for the other input) of the outer
    /// and anti joins, or every row with its mark for the mark joins. Drops
    /// the batches of the rows that it emitted (or skipped, without a final
    /// row) from their table. `None` if there are none.
    fn next_determined_output(
        &self,
        run: &mut SymmetricRun,
    ) -> Result<Option<RecordBatch>> {
        let mark = self.spec.output.is_mark_join();
        for side in 0..2 {
            let table = &mut run.tables[side];
            if !self.spec.emits_final_rows(side) {
                table.emitted = table.determined;
            }
            let (widths, other_widths) =
                (&self.spec.widths[side], &self.spec.widths[1 - side]);
            while table.emitted < table.determined {
                let mut positions = Vec::new();
                let mut marks = Vec::new();
                let mut bytes = 0;
                let mut position = table.emitted;
                while position < table.determined
                    && positions.len() < DEFAULT_VECTOR_SIZE
                    && bytes < PART_BYTES
                {
                    let index = position - table.base;
                    let matched = table.matched[index];
                    if mark || !matched {
                        positions.push(Some(index));
                        marks.push(matched);
                        bytes += widths.row_bytes(table.variable_bytes[index], OUTPUT)
                            + other_widths.null_row_bytes(OUTPUT)
                            + FINAL_ROW_INDEX_BYTES;
                    }
                    position += 1;
                }
                table.emitted = position;
                if positions.is_empty() {
                    continue;
                }
                let transient = run.reservation.new_empty();
                transient.try_grow(bytes)?;
                let other_schema = self.children[1 - side].schema();
                let marks = mark.then(|| Arc::new(BooleanArray::from(marks)) as ArrayRef);
                let (left, right) = sides(
                    side,
                    JoinRows::Stored(&table.rows, &positions),
                    JoinRows::Nulls(&other_schema, positions.len()),
                );
                let batch = self.spec.output.batch(
                    &left,
                    &right,
                    marks.as_ref(),
                    positions.len(),
                )?;
                return Ok(Some(batch));
            }
            self.drop_emitted_rows(run, side)?;
        }
        Ok(None)
    }

    /// Drop the batches of the table of input `side` whose rows the join
    /// emitted (or skipped) all, with the keys of the hash table whose rows
    /// they all are, releasing their memory.
    fn drop_emitted_rows(&self, run: &mut SymmetricRun, side: usize) -> Result<()> {
        let SymmetricRun {
            tables,
            converter,
            reservation,
            ..
        } = run;
        let table = &mut tables[side];
        let emitted = table.emitted - table.base;
        let dropped = table
            .rows
            .batches()
            .iter()
            .zip(table.rows.batch_starts())
            .take_while(|(batch, start)| *start + batch.num_rows() <= emitted)
            .count();
        if dropped == 0 {
            return Ok(());
        }
        // The keys whose last row is dropped: their rows are all dropped. The
        // keys, evaluated again and encoded, are reserved while they are
        // looked up.
        table.drop_batches(dropped, |batch| {
            let evaluated = reservation.new_empty();
            let arrays = evaluate_keys(self.spec.keys(side), batch, &evaluated)?;
            evaluated.try_resize(key_memory(&arrays, batch))?;
            Ok((converter.convert_columns(&arrays)?, evaluated))
        })?;
        // The structures that the dropped rows leave mostly empty shrink, if
        // the pool holds the smaller ones while they take the entries; the
        // tables are reserved as they are then.
        if reservation.try_grow(table.shrink_bytes()).is_ok() {
            table.shrink();
        }
        reservation.try_resize(tables.iter().map(SideTable::memory).sum())?;
        debug_assert_eq!(
            reservation.size(),
            tables.iter().map(SideTable::memory).sum::<usize>()
        );
        Ok(())
    }

    /// The next output of the join: joins the chunk being joined, emits the
    /// final rows of the rows that are determined, and reads the next chunk of
    /// an input (alternately, of the inputs that have one). Once both inputs
    /// ended, all rows are determined. `Blocked` while neither input has a
    /// chunk (the waker of the task of `ctx` is registered with both).
    fn next_output(
        &self,
        ctx: &ExecutionContext,
        run: &mut SymmetricRun,
    ) -> Result<SourceResult> {
        let mut chunks = 0;
        let mut parts = 0;
        loop {
            if let Some(job) = &mut run.job {
                // The parts of the candidates of a chunk without output (e.g.
                // whose pairs the filter rejects all) count towards the call
                // too: the chunk is joined further in the next call.
                if parts == PARTS_PER_CALL {
                    return Ok(SourceResult::Blocked(TaskInterrupt::yield_now()));
                }
                let output =
                    self.next_chunk_output(job, &mut run.tables, &run.reservation)?;
                if job.is_done() {
                    run.job = None;
                    self.determine(run, 0);
                    self.determine(run, 1);
                }
                if let Some(batch) = output {
                    return Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)));
                }
                parts += 1;
                continue;
            }
            if let Some(batch) = self.next_determined_output(run)? {
                return Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)));
            }
            if run.inputs.iter().all(NestedInput::is_done) {
                if run.tables.iter().all(|table| table.emitted == table.end()) {
                    return Ok(SourceResult::Finished);
                }
                self.determine(run, 0);
                self.determine(run, 1);
                continue;
            }
            if chunks == CHUNKS_PER_CALL {
                return Ok(SourceResult::Blocked(TaskInterrupt::yield_now()));
            }
            let mut read = false;
            for side in [run.next_side, 1 - run.next_side] {
                if run.inputs[side].is_done() {
                    continue;
                }
                let ahead =
                    run.chunks_read[side].saturating_sub(run.chunks_read[1 - side]);
                if run.pruning.is_some()
                    && ahead >= MAX_CHUNKS_AHEAD
                    && !run.inputs[1 - side].is_done()
                {
                    continue;
                }
                match run.inputs[side].poll_next(ctx) {
                    Poll::Ready(Ok(Some(batch))) => {
                        if batch.num_rows() > 0 {
                            run.job = Some(self.start_chunk(run, side, batch)?);
                            run.chunks_read[side] += 1;
                        }
                        run.next_side = 1 - side;
                        chunks += 1;
                        read = true;
                        break;
                    }
                    // No rows of the input are to come: the rows of the other
                    // input are determined.
                    Poll::Ready(Ok(None)) => {
                        self.determine(run, 1 - side);
                        read = true;
                    }
                    Poll::Ready(Err(e)) => return Err(e),
                    Poll::Pending => {}
                }
            }
            if !read {
                return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
            }
        }
    }
}

/// The bytes of an entry of the hash table of a [`SideTable`] (the key in it
/// takes bytes of its own, see [`SideTable::key_bytes`]).
const HEAD_BYTES: usize = size_of::<(Box<[u8]>, usize)>();

/// A structure of a [`SideTable`] whose capacity is more than this many
/// entries, and more than four times its entries, shrinks (see
/// [`SideTable::shrink`]).
const SHRUNK_CAPACITY: usize = 1024;

/// Whether a structure of `len` entries of `capacity` shrinks.
fn is_oversized(len: usize, capacity: usize) -> bool {
    capacity > SHRUNK_CAPACITY && capacity > 4 * len
}

/// The bytes that the hash table of a [`SideTable`] allocates once it grew to
/// hold `entries` entries: its buckets (7 of every 8 of them hold an entry
/// at most), with a control byte each, and a group of control bytes more (and
/// their alignment).
fn hash_map_bytes_holding(entries: usize) -> usize {
    let buckets = match entries {
        0 => return 0,
        1..4 => 4,
        4..8 => 8,
        _ => (entries * 8 / 7).next_power_of_two(),
    };
    buckets * (HEAD_BYTES + 1) + 32
}

/// The bytes that a deque of `len` entries of `size` bytes whose capacity is
/// `capacity` allocates while it grows to hold `added` entries more: it
/// doubles its capacity (at least), with the old entries while they move.
fn deque_bytes_appending(
    len: usize,
    capacity: usize,
    added: usize,
    size: usize,
) -> usize {
    let capacity = match len + added > capacity {
        true => (2 * capacity).max(len + added).max(8) + capacity,
        false => capacity,
    };
    capacity * size
}

/// The left and the right rows of pairs of rows of the input `side` and of
/// the other input.
fn sides<'a>(
    side: usize,
    own: JoinRows<'a>,
    other: JoinRows<'a>,
) -> (JoinRows<'a>, JoinRows<'a>) {
    match side {
        0 => (own, other),
        _ => (other, own),
    }
}

impl PhysicalOperator for PhysicalSymmetricHashJoin {
    fn name(&self) -> &str {
        "PhysicalSymmetricHashJoin"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.spec.output.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let spec = &self.spec;
        let mut description = spec.output.describe(
            self.name(),
            &spec.on,
            spec.filter.as_ref(),
            spec.null_equality,
        );
        if let Some(sorted) = &spec.sorted_filter {
            let [left, right] = &sorted.sides;
            let _ = write!(
                description,
                ", determined_by=[{}, {}]",
                left.order, right.order
            );
        }
        description
    }

    fn is_source(&self) -> bool {
        true
    }

    /// A single thread reads both inputs and joins their rows.
    fn parallel_source(&self) -> bool {
        false
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let reservation = MemoryConsumer::new("PhysicalSymmetricHashJoin")
            .register(ctx.task_ctx.memory_pool());
        let pruning = match &self.spec.sorted_filter {
            Some(filter) => Some(Pruning::try_new(filter.clone())?),
            None => None,
        };
        let run = SymmetricRun {
            inputs: [
                NestedInput::start(self.left(), ctx)?,
                NestedInput::start(self.right(), ctx)?,
            ],
            tables: [
                SideTable::new(self.left().schema()),
                SideTable::new(self.right().schema()),
            ],
            converter: self.spec.key_converter()?,
            job: None,
            next_side: 0,
            chunks_read: [0, 0],
            pruning,
            reservation,
        };
        Ok(Arc::new(SymmetricJoinSource {
            run: Mutex::new(run),
        }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<SymmetricJoinSource>(gstate.as_any(), self.name())?;
        self.next_output(ctx, &mut state.run.lock())
    }

    /// The join is the source of its pipeline: its inputs are executions of
    /// their own, not pipelines of this execution.
    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        meta.build_source(current)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use arrow::array::TimestampNanosecondArray;
    use arrow::array::UnionArray;
    use arrow::array::{
        Array, AsArray, Float64Array, Int32Array, Int64Array, ListArray, StringArray,
        StringViewArray,
    };
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, TimeUnit, UnionFields, UnionMode};
    use datafusion::catalog::streaming::StreamingTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::{DataFusionError, internal_datafusion_err};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::memory_pool::MemoryPool;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_execution::{SendableRecordBatchStream, TaskContext};
    use datafusion_expr::ColumnarValue;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::expressions::{
        BinaryExpr, CastExpr, NegativeExpr, lit,
    };
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::Partitioning;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::joins::SortMergeJoinExec;
    use datafusion_physical_plan::joins::utils::ColumnIndex;
    use datafusion_physical_plan::joins::{
        HashJoinExec, PartitionMode, StreamJoinPartitionMode,
    };
    use datafusion_physical_plan::limit::GlobalLimitExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::sorts::sort::SortExec;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
    use futures::StreamExt;

    use crate::bridge::{PipelineRootExec, bridged_plans};
    use crate::explain::{format_operator_tree, format_pipelines};
    use crate::join::PhysicalHashJoin;
    use crate::join::merge_join::allocations;
    use crate::join::merge_join::tests::{
        ALL_JOIN_TYPES, BuiltPartition, PeakPool, SlicedRows, count_native_rows,
        data_batch, data_schema, dictionary_probe_rows, dictionary_schema,
        dictionary_view_rows, empty_dictionary_cases, fanout_rows, fanout_schema,
        gather_cases, join_filter, keyed_rows, limited_single_thread_ctx,
        limited_task_ctx, num_rows, payload_schema, reference_join_of_keys,
        reference_join_of_rows, rejecting_filter, run_native, time_beside,
        union_key_layouts, union_key_matches, with_timeout,
    };
    use crate::operators::filter::PhysicalFilter;
    use crate::operators::union::test_support::count;
    use crate::test_utils::{
        EngineOutput, compare_outputs, run_engines, test_table_t1, test_table_t2,
    };

    /// A stream of `batches`, which then ends, or never produces anything
    /// again if `forever`.
    #[derive(Debug)]
    struct BatchesPartition {
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        forever: bool,
    }

    impl PartitionStream for BatchesPartition {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            let batches = futures::stream::iter(self.batches.clone().into_iter().map(Ok));
            let stream = match self.forever {
                true => batches.chain(futures::stream::pending()).boxed(),
                false => batches.boxed(),
            };
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.schema),
                stream,
            ))
        }
    }

    /// The partitions of a stream of the rows `0..rows` of the `side` input
    /// (see [`data_batch`]), in 3 partitions (one of them empty).
    fn partitions(
        side: &str,
        rows: i64,
        skew: i64,
        forever: bool,
    ) -> Vec<Arc<dyn PartitionStream>> {
        let part = |ids: Range<i64>| -> Arc<dyn PartitionStream> {
            let batches = match ids.is_empty() {
                true => vec![],
                false => vec![data_batch(side, ids, skew)],
            };
            Arc::new(BatchesPartition {
                schema: data_schema(side),
                batches,
                forever,
            })
        };
        vec![part(0..rows / 2), part(0..0), part(rows / 2..rows)]
    }

    /// A scan of a stream (that never ends, for DataFusion) of the `side`
    /// input, whose partitions end unless `forever`.
    fn stream_scan(
        side: &str,
        rows: i64,
        skew: i64,
        forever: bool,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(
            StreamingTableExec::try_new(
                data_schema(side),
                partitions(side, rows, skew, forever),
                None,
                vec![],
                true,
                None,
            )
            .unwrap(),
        )
    }

    /// A bounded scan of the `side` input.
    fn bounded_scan(side: &str, rows: i64, skew: i64) -> Arc<dyn ExecutionPlan> {
        let batch = data_batch(side, 0..rows, skew);
        MemorySourceConfig::try_new_exec(&[vec![batch]], data_schema(side), None).unwrap()
    }

    /// A `SymmetricHashJoinExec` of `left` and `right` on `lk = rk`, in
    /// single partition mode (of the merged partitions of its inputs).
    fn symmetric_join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
        null_equality: NullEquality,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let left = Arc::new(CoalescePartitionsExec::new(left));
        let right = Arc::new(CoalescePartitionsExec::new(right));
        Ok(Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            filter,
            &join_type,
            null_equality,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?))
    }

    fn task_ctx() -> Arc<TaskContext> {
        Arc::new(
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_target_partitions(4)),
        )
    }

    fn lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>> {
        try_lower(plan, |input, _| crate::planner::lower_execution_plan(input))?
            .ok_or_else(|| internal_datafusion_err!("The join was not lowered"))
    }

    /// Assert that the plan of `make_plan` returns the same rows on DataFusion
    /// and lowered with [`try_lower`]. Returns the lowered plan.
    async fn assert_same_rows(
        make_plan: impl Fn() -> Result<Arc<dyn ExecutionPlan>>,
        context: &str,
    ) -> Arc<dyn PhysicalOperator> {
        let plan = make_plan().unwrap();
        let batches = with_timeout(collect(Arc::clone(&plan), task_ctx()))
            .await
            .unwrap_or_else(|e| panic!("DataFusion fails: {e}\n{context}"));
        let expected = EngineOutput {
            schema: plan.schema(),
            plan: Arc::clone(&plan),
            batches,
            ordered: false,
        };
        let plan = make_plan().unwrap();
        let root = lower(&plan).unwrap();
        let tree = format_operator_tree(&root);
        let actual = run_native(&root, plan, task_ctx())
            .await
            .unwrap_or_else(|e| panic!("The native join fails: {e}\n{context}\n{tree}"));
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\n{context}\n{tree}");
        }
        assert!(
            bridged_plans(&root)
                .iter()
                .all(|plan| plan.is::<StreamingTableExec>())
        );
        root
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_of_unbounded_inputs_match_datafusion() {
        // The reference is DataFusion's hash join of the same rows (in bounded
        // inputs): its `SymmetricHashJoinExec` misses some matches of NULL keys
        // with `NullEqualsNull`, depending on the order in which the rows of
        // its inputs arrive.
        for join_type in ALL_JOIN_TYPES {
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                for filter in [None, Some(("id", Operator::Lt, "id"))] {
                    let context = format!("{join_type} {null_equality:?} {filter:?}");
                    let filter = || filter.map(|(l, op, r)| join_filter(l, op, r));
                    let reference = hash_join(
                        bounded_scan("l", 300, 0),
                        bounded_scan("r", 500, 3),
                        join_type,
                        filter(),
                        null_equality,
                    )
                    .unwrap();
                    let batches =
                        with_timeout(collect(Arc::clone(&reference), task_ctx()))
                            .await
                            .unwrap();
                    let expected = EngineOutput {
                        schema: reference.schema(),
                        plan: reference,
                        batches,
                        ordered: false,
                    };
                    let plan = symmetric_join(
                        stream_scan("l", 300, 0, false),
                        stream_scan("r", 500, 3, false),
                        join_type,
                        filter(),
                        null_equality,
                    )
                    .unwrap();
                    let root = lower(&plan).unwrap();
                    let tree = format_operator_tree(&root);
                    assert!(root.is::<PhysicalSymmetricHashJoin>(), "{context}");
                    let actual = run_native(&root, plan, task_ctx())
                        .await
                        .unwrap_or_else(|e| panic!("{context}: {e}\n{tree}"));
                    if let Err(message) = compare_outputs(&expected, &actual) {
                        panic!("Results differ: {message}\n{context}\n{tree}");
                    }
                    // And DataFusion's symmetric hash join, without NULL keys
                    // that match.
                    if null_equality == NullEquality::NullEqualsNothing {
                        let make_plan = || {
                            symmetric_join(
                                stream_scan("l", 300, 0, false),
                                stream_scan("r", 500, 3, false),
                                join_type,
                                filter(),
                                null_equality,
                            )
                        };
                        assert_same_rows(make_plan, &context).await;
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_of_bounded_inputs_are_hash_joins() {
        // An input that ends is the build side of a hash join: both, the left
        // one or the right one.
        for (left_bounded, right_bounded) in [(true, true), (true, false), (false, true)]
        {
            for join_type in ALL_JOIN_TYPES {
                let context = format!("{join_type} {left_bounded} {right_bounded}");
                let make_plan = || {
                    let scan = |side: &str, rows, skew, bounded: bool| match bounded {
                        true => bounded_scan(side, rows, skew),
                        false => stream_scan(side, rows, skew, false),
                    };
                    symmetric_join(
                        scan("l", 300, 0, left_bounded),
                        scan("r", 500, 3, right_bounded),
                        join_type,
                        Some(join_filter("d", Operator::NotEq, "d")),
                        NullEquality::NullEqualsNothing,
                    )
                };
                let root = assert_same_rows(make_plan, &context).await;
                let join = root.downcast_ref::<PhysicalHashJoin>().expect(&context);
                // The hash join builds on the right input if the left one never
                // ends (swapped), or if it is smaller.
                let swapped = !left_bounded;
                assert_eq!(
                    join.join_type(),
                    match swapped {
                        true => join_type.swap(),
                        false => join_type,
                    },
                    "{context}"
                );
            }
        }
    }

    /// The ids of the rows of `batches`: of the columns `lid` and `rid` that
    /// they have.
    fn ids(batches: &[RecordBatch]) -> Vec<(Option<i64>, Option<i64>)> {
        let mut ids = vec![];
        for batch in batches {
            let column = |name: &str| {
                batch.schema().index_of(name).ok().map(|index| {
                    batch
                        .column(index)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .clone()
                })
            };
            let (left, right) = (column("lid"), column("rid"));
            let value = |column: &Option<Int64Array>, row: usize| {
                column
                    .as_ref()
                    .and_then(|column| column.is_valid(row).then(|| column.value(row)))
            };
            ids.extend(
                (0..batch.num_rows()).map(|row| (value(&left, row), value(&right, row))),
            );
        }
        ids.sort_unstable();
        ids
    }

    /// A `HashJoinExec` of the bounded `left` and `right` inputs on `lk = rk`:
    /// the reference for the results of a join.
    fn hash_join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
        null_equality: NullEquality,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            left,
            right,
            on,
            filter,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            null_equality,
            false,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_of_inputs_that_never_end_emit_their_matches() -> Result<()> {
        // Both inputs produce their rows and then nothing, forever: the matches
        // (and the rows of a semi join) are emitted as the rows arrive, so that
        // a limit above the join ends the query, with matches of the join.
        // (DataFusion's symmetric hash join emits the rows of a semi join only
        // once its inputs end, or are pruned, so that it waits forever here.)
        for (join_type, limit) in [
            (JoinType::Inner, 50),
            (JoinType::Left, 50),
            (JoinType::Full, 50),
            (JoinType::LeftSemi, 20),
            (JoinType::RightSemi, 20),
        ] {
            let plan = || -> Result<Arc<dyn ExecutionPlan>> {
                let join = symmetric_join(
                    stream_scan("l", 300, 0, true),
                    stream_scan("r", 500, 3, true),
                    join_type,
                    None,
                    NullEquality::NullEqualsNothing,
                )?;
                Ok(Arc::new(GlobalLimitExec::new(join, 0, Some(limit))))
            };
            if !matches!(join_type, JoinType::LeftSemi | JoinType::RightSemi) {
                let expected = with_timeout(collect(plan()?, task_ctx())).await?;
                assert_eq!(num_rows(&expected), limit, "{join_type}");
            }
            let all = hash_join(
                bounded_scan("l", 300, 0),
                bounded_scan("r", 500, 3),
                join_type,
                None,
                NullEquality::NullEqualsNothing,
            )?;
            let all = ids(&collect(all, task_ctx()).await?);
            let plan = plan()?;
            let root = crate::planner::lower_execution_plan(&plan)?;
            assert_eq!(count::<PhysicalSymmetricHashJoin>(&root), 1, "{join_type}");
            let output = run_native(&root, plan, task_ctx()).await?;
            let emitted = ids(&output.batches);
            assert_eq!(emitted.len(), limit, "{join_type}");
            for row in emitted {
                assert!(all.binary_search(&row).is_ok(), "{join_type}: {row:?}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_tables_are_reserved_in_the_memory_pool() -> Result<()> {
        let plan = || {
            symmetric_join(
                stream_scan("l", 20_000, 0, false),
                stream_scan("r", 20_000, 3, false),
                JoinType::LeftAnti,
                None,
                NullEquality::NullEqualsNothing,
            )
        };
        let limited = |limit: usize| -> Result<Arc<TaskContext>> {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(limit, 1.0)
                .build_arc()?;
            Ok(Arc::new(TaskContext::default().with_runtime(runtime)))
        };
        let root = lower(&plan()?)?;
        let err = run_native(&root, plan()?, limited(256 * 1024)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        assert!(
            err.to_string().contains("PhysicalSymmetricHashJoin"),
            "{err}"
        );
        let output = run_native(&root, plan()?, limited(256 << 20)?).await?;
        assert!(num_rows(&output.batches) > 0);
        Ok(())
    }

    #[test]
    fn pipelines_of_a_symmetric_hash_join() -> Result<()> {
        // The join is the source of its pipeline; its inputs are executions of
        // their own.
        for join_type in [JoinType::Inner, JoinType::LeftAnti] {
            let plan = symmetric_join(
                stream_scan("l", 10, 0, false),
                stream_scan("r", 10, 0, false),
                join_type,
                None,
                NullEquality::NullEqualsNothing,
            )?;
            assert_eq!(
                format_pipelines(&lower(&plan)?)?,
                "MetaPipeline 0 (sink: ResultSink)\n  \
                 Pipeline 0: PhysicalSymmetricHashJoin -> ResultSink\n",
                "{join_type}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inputs_that_never_end_are_read_together_below_operators_that_are_not_parallel()
    -> Result<()> {
        // A filter with a fetch is not a parallel operator: the executor runs
        // the pipelines that contain it one after the other. The inputs of the
        // join, which produce matching rows and then nothing forever, are read
        // together all the same, so that the filter gets its row.
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let join = symmetric_join(
                stream_scan("l", 300, 0, true),
                stream_scan("r", 500, 3, true),
                JoinType::Inner,
                None,
                NullEquality::NullEqualsNothing,
            )?;
            let predicate = Arc::new(BinaryExpr::new(
                col("lid", &join.schema())?,
                Operator::GtEq,
                lit(0_i64),
            ));
            let filter = FilterExec::try_new(predicate, join)?;
            Ok(filter.with_fetch(Some(1)).expect("a filter with a fetch"))
        };
        let expected = with_timeout(collect(plan()?, task_ctx())).await?;
        assert_eq!(num_rows(&expected), 1);
        let plan = plan()?;
        let root = crate::planner::lower_execution_plan(&plan)?;
        let tree = format_operator_tree(&root);
        let filter = root.children()[0].downcast_ref::<PhysicalFilter>();
        assert!(
            filter.is_some_and(|filter| !filter.parallel_operator()),
            "{tree}"
        );
        assert_eq!(count::<PhysicalSymmetricHashJoin>(&root), 1, "{tree}");
        let output = run_native(&root, plan, task_ctx()).await?;
        assert_eq!(num_rows(&output.batches), 1);
        Ok(())
    }

    /// A join filter that is true for every pair of rows, and counts the
    /// pairs it is evaluated on.
    #[derive(Debug)]
    struct CountingFilter {
        pairs: Arc<AtomicUsize>,
    }

    impl PartialEq for CountingFilter {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.pairs, &other.pairs)
        }
    }

    impl Eq for CountingFilter {}

    impl std::hash::Hash for CountingFilter {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            Arc::as_ptr(&self.pairs).hash(state);
        }
    }

    impl std::fmt::Display for CountingFilter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "counting_filter")
        }
    }

    impl PhysicalExpr for CountingFilter {
        fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
            Ok(DataType::Boolean)
        }

        fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
            Ok(false)
        }

        fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
            self.pairs.fetch_add(batch.num_rows(), Ordering::SeqCst);
            Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(vec![
                true;
                batch.num_rows()
            ]))))
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
            write!(f, "counting_filter")
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rows_of_many_matches_are_joined_in_chunks() -> Result<()> {
        // 500 rows on either side with the same key: 250,000 matches, which
        // the join emits in chunks of at most 2,048 rows, one chunk of
        // candidates at a time. With a limit of a row, it stops after its first
        // chunk of candidates, within a memory pool that holds its tables, but
        // not the matches; without the limit, it emits all of them, in chunks,
        // within the same pool. The candidates are reserved while they are
        // joined: a pool of 64 KiB holds the tables (of about 20 KB), but not
        // a chunk of candidates.
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Int32, false),
            ]))
        };
        let stream = |side: &str| -> Result<Arc<dyn ExecutionPlan>> {
            let batch = RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int64Array::from_iter_values(0..500)),
                    Arc::new(Int32Array::from(vec![7; 500])),
                ],
            )?;
            let partition = Arc::new(BatchesPartition {
                schema: schema(side),
                batches: (0..500).step_by(100).map(|i| batch.slice(i, 100)).collect(),
                forever: false,
            });
            Ok(Arc::new(StreamingTableExec::try_new(
                schema(side),
                vec![partition],
                None,
                vec![],
                true,
                None,
            )?))
        };
        let pairs = Arc::new(AtomicUsize::new(0));
        let plan = |limit: Option<usize>| -> Result<Arc<dyn ExecutionPlan>> {
            let filter_schema = Arc::new(Schema::new(vec![
                Field::new("lid", DataType::Int64, false),
                Field::new("rid", DataType::Int64, false),
            ]));
            let filter = JoinFilter::new(
                Arc::new(CountingFilter {
                    pairs: Arc::clone(&pairs),
                }),
                vec![
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Right,
                    },
                ],
                filter_schema,
            );
            let join = symmetric_join(
                stream("l")?,
                stream("r")?,
                JoinType::Inner,
                Some(filter),
                NullEquality::NullEqualsNothing,
            )?;
            Ok(match limit {
                Some(limit) => Arc::new(GlobalLimitExec::new(join, 0, Some(limit))),
                None => join,
            })
        };
        for limit in [Some(1), None] {
            let plan = plan(limit)?;
            let root = crate::planner::lower_execution_plan(&plan)?;
            let err = run_native(&root, plan, limited_task_ctx(64 * 1024)?)
                .await
                .unwrap_err();
            assert!(
                matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
                "{err}"
            );
            assert!(
                err.to_string().contains("PhysicalSymmetricHashJoin"),
                "{err}"
            );
        }
        let pool = 512 * 1024;
        for (limit, rows) in [(Some(1), 1), (None, 250_000)] {
            pairs.store(0, Ordering::SeqCst);
            let plan = plan(limit)?;
            let root = crate::planner::lower_execution_plan(&plan)?;
            assert_eq!(count::<PhysicalSymmetricHashJoin>(&root), 1);
            let output = run_native(&root, plan, limited_task_ctx(pool)?).await?;
            assert_eq!(num_rows(&output.batches), rows);
            assert!(
                output
                    .batches
                    .iter()
                    .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
            );
            let evaluated = pairs.load(Ordering::SeqCst);
            match limit {
                // The candidates of a chunk, and maybe of the next one.
                Some(_) => assert!(evaluated <= 2 * DEFAULT_VECTOR_SIZE, "{evaluated}"),
                None => assert_eq!(evaluated, 250_000),
            }
        }
        Ok(())
    }

    /// Register the test tables `t1` and `t2` (see `register_test_tables`) as
    /// streams that never end (for DataFusion), `s1` and `s2`.
    fn register_streams(ctx: &SessionContext) -> Result<()> {
        for (name, partitions) in [("s1", test_table_t1()?), ("s2", test_table_t2()?)] {
            let schema = partitions[0][0].schema();
            let streams = partitions
                .into_iter()
                .map(|batches| {
                    Arc::new(BatchesPartition {
                        schema: Arc::clone(&schema),
                        batches,
                        forever: false,
                    }) as Arc<dyn PartitionStream>
                })
                .collect();
            let table =
                StreamingTable::try_new(schema, streams)?.with_infinite_table(true);
            ctx.register_table(name, Arc::new(table))?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn symmetric_hash_joins_of_queries_are_native() {
        for sql in [
            "SELECT s1.id, s2.name FROM s1 JOIN s2 ON s1.a = s2.k",
            "SELECT s1.id, s2.name FROM s1 LEFT JOIN s2 ON s1.a = s2.k",
            "SELECT s1.id, s2.name FROM s1 RIGHT JOIN s2 ON s1.a = s2.k",
            "SELECT s1.id, s2.name FROM s1 FULL JOIN s2 ON s1.a = s2.k AND s1.b > s2.w",
            "SELECT s1.id FROM s1 WHERE EXISTS (SELECT 1 FROM s2 WHERE s2.k = s1.a)",
            "SELECT s1.id FROM s1 WHERE NOT EXISTS (SELECT 1 FROM s2 WHERE s2.k = s1.a AND s2.w > s1.b)",
            "SELECT s1.id FROM s1 WHERE EXISTS (SELECT 1 FROM s2 WHERE s2.k = s1.a) OR s1.id < 10",
            "SELECT s1.id, s2.w FROM s1 JOIN s2 ON s1.a = s2.k AND s1.d = s2.name",
        ] {
            for partitions in [4, 1] {
                let config = SessionConfig::new().with_target_partitions(partitions);
                let setup = async |ctx: &SessionContext| register_streams(ctx);
                let (expected, actual) = with_timeout(run_engines(sql, config, setup))
                    .await
                    .unwrap_or_else(|e| panic!("{sql}: {e}"));
                let plan = expected.plan.as_ref();
                let symmetric = datafusion_physical_plan::displayable(plan)
                    .indent(true)
                    .to_string()
                    .contains("SymmetricHashJoinExec");
                assert!(symmetric, "{sql}");
                let root = actual
                    .plan
                    .downcast_ref::<PipelineRootExec>()
                    .unwrap()
                    .operator();
                let tree = format_operator_tree(root);
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}\nquery: {sql}\nlowered: {tree}");
                }
                assert_eq!(count::<PhysicalSymmetricHashJoin>(root), 1, "{sql}\n{tree}");
                assert!(
                    bridged_plans(root)
                        .iter()
                        .all(|plan| plan.is::<StreamingTableExec>()),
                    "{sql}\n{tree}"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sort_merge_joins_of_unbounded_inputs_are_native() -> Result<()> {
        // A sort-merge join (of several partitions, whose order no node above
        // relies on) of two inputs that never end is a symmetric hash join; of
        // one that ends, a hash join that builds on it.
        for (left_unbounded, right_unbounded) in
            [(true, true), (true, false), (false, true)]
        {
            for join_type in ALL_JOIN_TYPES {
                let context = format!("{join_type} {left_unbounded} {right_unbounded}");
                let make_plan = || -> Result<Arc<dyn ExecutionPlan>> {
                    let input = |side: &str, rows, skew, unbounded| -> Result<_> {
                        let scan = match unbounded {
                            true => stream_scan(side, rows, skew, false),
                            false => bounded_scan(side, rows, skew),
                        };
                        let key = col(&format!("{side}k"), &scan.schema())?;
                        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(
                            Arc::clone(&key),
                        )])
                        .unwrap();
                        let partitioning = Partitioning::Hash(vec![Arc::clone(&key)], 4);
                        let repartition = RepartitionExec::try_new(scan, partitioning)?;
                        let sort = SortExec::new(ordering, Arc::new(repartition))
                            .with_preserve_partitioning(true);
                        Ok((Arc::new(sort) as Arc<dyn ExecutionPlan>, key))
                    };
                    let (left, left_key) = input("l", 300, 0, left_unbounded)?;
                    let (right, right_key) = input("r", 500, 3, right_unbounded)?;
                    Ok(Arc::new(SortMergeJoinExec::try_new(
                        left,
                        right,
                        vec![(left_key, right_key)],
                        Some(join_filter("id", Operator::Gt, "id")),
                        join_type,
                        vec![SortOptions::default()],
                        NullEquality::NullEqualsNothing,
                    )?))
                };
                let plan = make_plan()?;
                let expected = with_timeout(collect(make_plan()?, task_ctx())).await?;
                let expected = EngineOutput {
                    schema: plan.schema(),
                    plan: Arc::clone(&plan),
                    batches: expected,
                    ordered: false,
                };
                let root =
                    crate::join::merge_join::try_lower(&plan, false, |input, _| {
                        crate::planner::lower_execution_plan(input)
                    })?
                    .expect("a native join");
                let tree = format_operator_tree(&root);
                match left_unbounded && right_unbounded {
                    true => assert!(root.is::<PhysicalSymmetricHashJoin>(), "{context}"),
                    false => assert!(root.is::<PhysicalHashJoin>(), "{context}"),
                }
                let actual = run_native(&root, plan, task_ctx()).await?;
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}\n{context}\n{tree}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn keys_of_different_types_are_not_lowered() -> Result<()> {
        let left = stream_scan("l", 10, 0, false);
        let right = stream_scan("r", 10, 0, false);
        let on = vec![(col("lk", &left.schema())?, col("rid", &right.schema())?)];
        assert_eq!(left.schema().field(1).data_type(), &DataType::Int32);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?);
        assert!(
            try_lower(&plan, |input, _| crate::planner::lower_execution_plan(
                input
            ))?
            .is_none()
        );
        Ok(())
    }

    /// The schema of the rows `(id, k, t, s)` of the `side` input of the joins
    /// of sorted streams: an id, a key, a time and a string.
    fn timed_schema(side: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(format!("{side}id"), DataType::Int64, false),
            Field::new(format!("{side}k"), DataType::Int32, true),
            Field::new(format!("{side}t"), DataType::Int64, true),
            Field::new(format!("{side}s"), DataType::Utf8, true),
        ]))
    }

    /// The rows `ids` of the `side` input, with the keys `key(id)`, the times
    /// `time(id)` and the strings `text(id)`, in batches of `batch_rows` rows.
    fn timed_batches(
        side: &str,
        ids: Range<i64>,
        key: impl Fn(i64) -> Option<i32>,
        time: impl Fn(i64) -> Option<i64>,
        text: impl Fn(i64) -> String,
        batch_rows: usize,
    ) -> Vec<RecordBatch> {
        let ids = ids.collect::<Vec<_>>();
        ids.chunks(batch_rows)
            .map(|ids| {
                let column = |values: ArrayRef| values;
                RecordBatch::try_new(
                    timed_schema(side),
                    vec![
                        column(Arc::new(Int64Array::from(ids.to_vec()))),
                        column(Arc::new(Int32Array::from_iter(
                            ids.iter().map(|&id| key(id)),
                        ))),
                        column(Arc::new(Int64Array::from_iter(
                            ids.iter().map(|&id| time(id)),
                        ))),
                        column(Arc::new(StringArray::from_iter_values(
                            ids.iter().map(|&id| text(id)),
                        ))),
                    ],
                )
                .unwrap()
            })
            .collect()
    }

    /// The order of the timed streams of the `side` input: by `<side>t`, NULLs
    /// first.
    fn time_order(side: &str) -> LexOrdering {
        let time = col(&format!("{side}t"), &timed_schema(side)).unwrap();
        LexOrdering::new([PhysicalSortExpr::new(time, SortOptions::new(false, true))])
            .unwrap()
    }

    /// A stream (that never ends, for DataFusion) of the `batches` of the
    /// `side` input, of one partition, sorted by its times if `sorted`. It
    /// ends unless `forever`.
    fn timed_stream(
        side: &str,
        batches: Vec<RecordBatch>,
        sorted: bool,
        forever: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let partition = BatchesPartition {
            schema: timed_schema(side),
            batches,
            forever,
        };
        let orders = match sorted {
            true => vec![time_order(side)],
            false => vec![],
        };
        Arc::new(
            StreamingTableExec::try_new(
                timed_schema(side),
                vec![Arc::new(partition)],
                None,
                orders,
                true,
                None,
            )
            .unwrap(),
        )
    }

    /// A filter of a join of timed inputs, `expr` of their times `lt` and
    /// `rt`.
    fn time_filter(
        expr: impl Fn(PhysicalExprRef, PhysicalExprRef) -> PhysicalExprRef,
    ) -> JoinFilter {
        let schema = Arc::new(Schema::new(vec![
            Field::new("lt", DataType::Int64, true),
            Field::new("rt", DataType::Int64, true),
        ]));
        let (lt, rt) = (col("lt", &schema).unwrap(), col("rt", &schema).unwrap());
        JoinFilter::new(
            expr(lt, rt),
            vec![
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Right,
                },
            ],
            schema,
        )
    }

    /// `left op right`.
    fn binary(
        left: PhysicalExprRef,
        op: Operator,
        right: PhysicalExprRef,
    ) -> PhysicalExprRef {
        Arc::new(BinaryExpr::new(left, op, right))
    }

    /// `left op right`, arithmetic that fails on overflow (which interval
    /// arithmetic bounds, unlike arithmetic that wraps).
    fn checked(
        left: PhysicalExprRef,
        op: Operator,
        right: PhysicalExprRef,
    ) -> PhysicalExprRef {
        Arc::new(BinaryExpr::new(left, op, right).with_fail_on_overflow(true))
    }

    /// The filters of the joins of timed inputs: `lt > rt` (no right row still
    /// to come matches a left row before the last right time), `lt < rt`, and
    /// a band `rt - 7 < lt AND lt < rt + 5` (both). The arithmetic of the band
    /// fails on overflow: interval arithmetic does not bound the integer
    /// arithmetic that wraps (nor that of floats, see
    /// [`nan_of_arithmetic_determines_no_rows`]).
    fn time_filters() -> Vec<(&'static str, JoinFilter)> {
        vec![
            (
                "lt > rt",
                time_filter(|lt, rt| binary(lt, Operator::Gt, rt)),
            ),
            (
                "lt < rt",
                time_filter(|lt, rt| binary(lt, Operator::Lt, rt)),
            ),
            (
                "rt - 7 < lt AND lt < rt + 5",
                time_filter(|lt, rt| {
                    let low = checked(Arc::clone(&rt), Operator::Minus, lit(7i64));
                    let high = checked(rt, Operator::Plus, lit(5i64));
                    binary(
                        binary(low, Operator::Lt, Arc::clone(&lt)),
                        Operator::And,
                        binary(lt, Operator::Lt, high),
                    )
                }),
            ),
        ]
    }

    /// A `SymmetricHashJoinExec` of the timed inputs `left` and `right` on
    /// `lk = rk`, which DataFusion prunes by their times if `sorted`.
    fn timed_join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
        null_equality: NullEquality,
        sorted: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let (left_order, right_order) = match sorted {
            true => (Some(time_order("l")), Some(time_order("r"))),
            false => (None, None),
        };
        Ok(Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            filter,
            &join_type,
            null_equality,
            left_order,
            right_order,
            StreamJoinPartitionMode::SinglePartition,
        )?))
    }

    /// The rows of the timed `side` input of [`joins_of_sorted_streams_match_datafusion`]:
    /// a few rows without a time first (the NULLs come first), then ascending
    /// times, with keys of which some are NULL, in batches of 50 rows.
    fn sorted_rows(side: &str, rows: i64, every: i64) -> Vec<RecordBatch> {
        timed_batches(
            side,
            0..rows,
            |id| (id % 13 != 4).then_some((id % 7) as i32),
            |id| (id >= 3).then_some(id / every),
            |id| format!("{side}{id}"),
            50,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_of_sorted_streams_match_datafusion() {
        // Streams sorted by their times, joined with filters of their times:
        // the join determines the rows that no row still to come can match, as
        // the rows come (also the rows of the input whose rows come later),
        // emits their final rows and drops them. The reference is DataFusion's
        // hash join of the same rows (in bounded inputs).
        for join_type in ALL_JOIN_TYPES {
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                for (name, filter) in time_filters() {
                    let context = format!("{join_type} {null_equality:?} {name}");
                    let bounded = |side: &str, rows: i64, every: i64| {
                        MemorySourceConfig::try_new_exec(
                            &[sorted_rows(side, rows, every)],
                            timed_schema(side),
                            None,
                        )
                        .unwrap() as Arc<dyn ExecutionPlan>
                    };
                    let reference = hash_join(
                        bounded("l", 600, 2),
                        bounded("r", 800, 3),
                        join_type,
                        Some(filter.clone()),
                        null_equality,
                    )
                    .unwrap();
                    let batches =
                        with_timeout(collect(Arc::clone(&reference), task_ctx()))
                            .await
                            .unwrap();
                    let expected = EngineOutput {
                        schema: reference.schema(),
                        plan: reference,
                        batches,
                        ordered: false,
                    };
                    let stream = |side: &str, rows: i64, every: i64| {
                        timed_stream(side, sorted_rows(side, rows, every), true, false)
                    };
                    let plan = timed_join(
                        stream("l", 600, 2),
                        stream("r", 800, 3),
                        join_type,
                        Some(filter),
                        null_equality,
                        true,
                    )
                    .unwrap();
                    let root = lower(&plan).unwrap();
                    let tree = format_operator_tree(&root);
                    let join = root
                        .downcast_ref::<PhysicalSymmetricHashJoin>()
                        .unwrap_or_else(|| panic!("{context}\n{tree}"));
                    assert!(join.determines_rows_by_order(), "{context}\n{tree}");
                    let actual = run_native(&root, plan, task_ctx())
                        .await
                        .unwrap_or_else(|e| panic!("{context}: {e}\n{tree}"));
                    if let Err(message) = compare_outputs(&expected, &actual) {
                        panic!("Results differ: {message}\n{context}\n{tree}");
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn determined_rows_of_streams_that_never_end_are_emitted() -> Result<()> {
        // Streams sorted by their times, with keys that never match (even keys
        // on the left, odd ones on the right), which produce their rows and
        // then nothing, forever. With `lt > rt`, no right row still to come
        // matches a left row whose time is not after the last right time, and
        // with `lt < rt`, the other way round: the join emits these rows
        // (without a match, or with their mark) as soon as they are
        // determined, so that a limit above it ends the query, as with
        // DataFusion's join, which prunes them.
        let filters = [
            ("lt > rt", Operator::Gt, JoinSide::Left),
            ("lt < rt", Operator::Lt, JoinSide::Right),
        ];
        for (name, op, side) in filters {
            let join_types = match side {
                JoinSide::Left => {
                    [JoinType::Left, JoinType::LeftAnti, JoinType::LeftMark]
                }
                _ => [JoinType::Right, JoinType::RightAnti, JoinType::RightMark],
            };
            for join_type in join_types.into_iter().chain([JoinType::Full]) {
                let context = format!("{join_type} {name}");
                let plan = || -> Result<Arc<dyn ExecutionPlan>> {
                    let stream = |side: &str, key: fn(i64) -> Option<i32>| {
                        let batches = timed_batches(
                            side,
                            0..100,
                            key,
                            Some,
                            |id| id.to_string(),
                            20,
                        );
                        timed_stream(side, batches, true, true)
                    };
                    let join = timed_join(
                        stream("l", |id| Some((id * 2) as i32)),
                        stream("r", |id| Some((id * 2 + 1) as i32)),
                        join_type,
                        Some(time_filter(|lt, rt| binary(lt, op, rt))),
                        NullEquality::NullEqualsNothing,
                        true,
                    )?;
                    Ok(Arc::new(GlobalLimitExec::new(join, 0, Some(1))))
                };
                let expected = with_timeout(collect(plan()?, task_ctx())).await?;
                assert_eq!(num_rows(&expected), 1, "{context}");
                let plan = plan()?;
                let root = crate::planner::lower_execution_plan(&plan)?;
                let join = root.children()[0]
                    .downcast_ref::<PhysicalSymmetricHashJoin>()
                    .unwrap_or_else(|| {
                        panic!("{context}\n{}", format_operator_tree(&root))
                    });
                assert!(join.determines_rows_by_order(), "{context}");
                let actual = run_native(&root, plan, task_ctx()).await?;
                assert_eq!(num_rows(&actual.batches), 1, "{context}");
                // A row of the input whose rows the filter determines, without
                // a match.
                let batch = &actual.batches[0];
                let schema = batch.schema();
                let preserved = match side {
                    JoinSide::Left => "lid",
                    _ => "rid",
                };
                assert!(schema.index_of(preserved).is_ok(), "{context}: {schema:?}");
                for (field, column) in schema.fields().iter().zip(batch.columns()) {
                    let other = match side {
                        JoinSide::Left => field.name().starts_with('r'),
                        _ => field.name().starts_with('l'),
                    };
                    if other {
                        assert_eq!(column.null_count(), 1, "{context}: {}", field.name());
                    }
                    if field.name() == "mark" {
                        let marks =
                            column.as_any().downcast_ref::<BooleanArray>().unwrap();
                        assert!(!marks.value(0), "{context}");
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_of_sorted_streams_keep_the_rows_that_can_still_match() -> Result<()> {
        // A band join of long streams sorted by their times: a row matches the
        // rows of the other input whose times are close to its own only, so
        // that the join drops the rows of either input once the times of the
        // other one passed them, and holds a few batches of rows at a time,
        // within a memory pool that the rows of the inputs exceed. Without the
        // orders of its inputs, it keeps them all, and fails.
        let rows = 20_000;
        let plan = |sorted: bool| -> Result<Arc<dyn ExecutionPlan>> {
            let stream = |side: &str| {
                let batches = timed_batches(
                    side,
                    0..rows,
                    |id| Some((id % 10) as i32),
                    Some,
                    |id| format!("{side}{id}"),
                    100,
                );
                timed_stream(side, batches, sorted, false)
            };
            let filter = time_filter(|lt, rt| {
                let low = checked(Arc::clone(&rt), Operator::Minus, lit(5i64));
                let high = checked(rt, Operator::Plus, lit(5i64));
                binary(
                    binary(low, Operator::Lt, Arc::clone(&lt)),
                    Operator::And,
                    binary(lt, Operator::Lt, high),
                )
            });
            timed_join(
                stream("l"),
                stream("r"),
                JoinType::Left,
                Some(filter),
                NullEquality::NullEqualsNothing,
                sorted,
            )
        };
        // The tables of the join hold about 150 KB of rows at most, a part of
        // its candidates 260 KB; the rows of the inputs have about 2 MB.
        let pool = 1 << 20;
        // Every left row matches the right row with its id only.
        let sorted = plan(true)?;
        let root = lower(&sorted)?;
        let actual = run_native(&root, sorted, limited_task_ctx(pool)?).await?;
        assert_eq!(num_rows(&actual.batches), rows as usize);
        let unsorted = plan(false)?;
        let root = lower(&unsorted)?;
        let err = run_native(&root, unsorted, limited_task_ctx(pool)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wide_rows_are_reserved_and_joined_in_fewer_pairs() -> Result<()> {
        // 100 rows on either side with the same key and strings of 1 KiB: the
        // 10,000 matches have 2 KiB of strings each, which the join reserves
        // before it gathers them (also those that a limit does not need), in
        // parts of about a MiB, not of 2,048 pairs. A pool of 512 KiB holds
        // the tables, but not a part; one of 4 MiB holds both.
        let plan = |limit: Option<usize>| -> Result<Arc<dyn ExecutionPlan>> {
            let stream = |side: &str| {
                let batches = timed_batches(
                    side,
                    0..100,
                    |_| Some(1),
                    Some,
                    |id| format!("{id:>1024}"),
                    25,
                );
                timed_stream(side, batches, false, limit.is_some())
            };
            let join = symmetric_join(
                stream("l"),
                stream("r"),
                JoinType::Inner,
                None,
                NullEquality::NullEqualsNothing,
            )?;
            Ok(match limit {
                Some(limit) => Arc::new(GlobalLimitExec::new(join, 0, Some(limit))),
                None => join,
            })
        };
        let plan_limit = plan(Some(1))?;
        let root = crate::planner::lower_execution_plan(&plan_limit)?;
        let err = run_native(&root, plan_limit, limited_task_ctx(512 * 1024)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        assert!(
            err.to_string().contains("PhysicalSymmetricHashJoin"),
            "{err}"
        );
        let plan_limit = plan(Some(1))?;
        let root = crate::planner::lower_execution_plan(&plan_limit)?;
        let output = run_native(&root, plan_limit, limited_task_ctx(4 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 1);
        let plan = plan(None)?;
        let root = lower(&plan)?;
        let output = run_native(&root, plan, limited_task_ctx(4 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 10_000);
        for batch in &output.batches {
            assert!(
                batch.num_rows() <= PART_BYTES / 2048 + 1,
                "{}",
                batch.num_rows()
            );
        }
        Ok(())
    }

    /// Register the timed streams `l` and `r` (see [`timed_stream`]) of the
    /// rows `left` and `right`, sorted by their times, that never end for
    /// DataFusion, and end unless `forever`.
    fn register_timed_streams(
        ctx: &SessionContext,
        left: Vec<RecordBatch>,
        right: Vec<RecordBatch>,
        forever: bool,
    ) -> Result<()> {
        for (side, batches) in [("l", left), ("r", right)] {
            let partition = BatchesPartition {
                schema: timed_schema(side),
                batches,
                forever,
            };
            let table =
                StreamingTable::try_new(timed_schema(side), vec![Arc::new(partition)])?
                    .with_infinite_table(true)
                    .with_sort_order(vec![
                        datafusion_expr::col(format!("{side}t")).sort(true, true),
                    ]);
            ctx.register_table(side, Arc::new(table))?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn symmetric_hash_joins_of_sorted_streams_of_queries_determine_their_rows() {
        // DataFusion plans symmetric hash joins of the streams with the orders
        // of their times, which their filters compare: the native joins read
        // them in their orders (without the repartitionings of DataFusion's
        // plans), and determine their rows by their times.
        // The bands compute with floats (of the integer times), whose NaN
        // interval arithmetic does not bound: their joins determine no rows
        // (see `nan_of_arithmetic_determines_no_rows`), which DataFusion's do.
        for (sql, determined) in [
            (
                "SELECT l.lid, r.rid FROM l LEFT JOIN r ON l.lk = r.rk AND l.lt > r.rt",
                true,
            ),
            (
                "SELECT l.lid, r.rid FROM l RIGHT JOIN r ON l.lk = r.rk AND l.lt < r.rt",
                true,
            ),
            (
                "SELECT l.lid, r.rid FROM l FULL JOIN r ON l.lk = r.rk AND l.lt > r.rt - 7.0 AND l.lt < r.rt + 5.0",
                false,
            ),
            (
                "SELECT l.lid, r.rid FROM l JOIN r ON l.lk = r.rk AND l.lt > r.rt - 7.0 AND l.lt < r.rt + 5.0",
                false,
            ),
            (
                "SELECT l.lid FROM l WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.rk = l.lk AND r.rt < l.lt)",
                true,
            ),
        ] {
            for partitions in [4, 1] {
                let config = SessionConfig::new().with_target_partitions(partitions);
                let setup = async |ctx: &SessionContext| {
                    register_timed_streams(
                        ctx,
                        sorted_rows("l", 300, 2),
                        sorted_rows("r", 400, 3),
                        false,
                    )
                };
                let (expected, actual) = with_timeout(run_engines(sql, config, setup))
                    .await
                    .unwrap_or_else(|e| panic!("{sql}: {e}"));
                let root = actual
                    .plan
                    .downcast_ref::<PipelineRootExec>()
                    .unwrap()
                    .operator();
                let tree = format_operator_tree(root);
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}\nquery: {sql}\nlowered: {tree}");
                }
                assert_eq!(
                    tree.contains("determined_by=["),
                    determined,
                    "{sql}\n{tree}"
                );
                assert!(
                    bridged_plans(root)
                        .iter()
                        .all(|plan| plan.is::<StreamingTableExec>()),
                    "{sql}\n{tree}"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn determined_rows_of_streams_of_timestamps_are_emitted() -> Result<()> {
        // As `determined_rows_of_streams_that_never_end_are_emitted`, with
        // ascending timestamps (in nanoseconds, of a time zone), as the rows
        // of streams of events have: of streams that declare their order, or
        // that declare none, whose order the plan of the join says (the
        // native join takes it, as DataFusion's join does).
        let schema = |side: &str| -> SchemaRef {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Int32, false),
                Field::new(
                    format!("{side}time"),
                    DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                    false,
                ),
            ]))
        };
        let time_order = |side: &str| {
            let time = col(&format!("{side}time"), &schema(side)).unwrap();
            LexOrdering::new([PhysicalSortExpr::new(
                time,
                SortOptions::new(false, false),
            )])
            .unwrap()
        };
        let stream = |side: &str, key: fn(i64) -> i32, declared: bool| {
            let batches = (0..5)
                .map(|batch| {
                    let ids = (batch * 20..batch * 20 + 20).collect::<Vec<i64>>();
                    let times = TimestampNanosecondArray::from_iter_values(
                        ids.iter()
                            .map(|id| 1_700_000_000_000_000_000 + id * 1_000_000_000),
                    )
                    .with_timezone("UTC");
                    RecordBatch::try_new(
                        schema(side),
                        vec![
                            Arc::new(Int64Array::from(ids.clone())),
                            Arc::new(Int32Array::from_iter_values(
                                ids.iter().map(|&id| key(id)),
                            )),
                            Arc::new(times),
                        ],
                    )
                    .unwrap()
                })
                .collect();
            let partition = BatchesPartition {
                schema: schema(side),
                batches,
                forever: true,
            };
            let orders = match declared {
                true => vec![time_order(side)],
                false => vec![],
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema(side),
                    vec![Arc::new(partition)],
                    None,
                    orders,
                    true,
                    None,
                )
                .unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        for join_type in [JoinType::Left, JoinType::LeftAnti, JoinType::LeftMark] {
            for declared in [true, false] {
                let context = format!("{join_type}, declared order: {declared}");
                let plan = || -> Result<Arc<dyn ExecutionPlan>> {
                    let left = stream("l", |id| (id * 2) as i32, declared);
                    let right = stream("r", |id| (id * 2 + 1) as i32, declared);
                    let on =
                        vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
                    let filter_schema = Arc::new(Schema::new(vec![
                        left.schema().field(2).clone(),
                        right.schema().field(2).clone(),
                    ]));
                    let filter = JoinFilter::new(
                        binary(
                            col("ltime", &filter_schema)?,
                            Operator::Gt,
                            col("rtime", &filter_schema)?,
                        ),
                        vec![
                            ColumnIndex {
                                index: 2,
                                side: JoinSide::Left,
                            },
                            ColumnIndex {
                                index: 2,
                                side: JoinSide::Right,
                            },
                        ],
                        filter_schema,
                    );
                    let join = Arc::new(SymmetricHashJoinExec::try_new(
                        left,
                        right,
                        on,
                        Some(filter),
                        &join_type,
                        NullEquality::NullEqualsNothing,
                        Some(time_order("l")),
                        Some(time_order("r")),
                        StreamJoinPartitionMode::SinglePartition,
                    )?);
                    Ok(Arc::new(GlobalLimitExec::new(join, 0, Some(1))))
                };
                let expected = with_timeout(collect(plan()?, task_ctx())).await?;
                assert_eq!(num_rows(&expected), 1, "{context}");
                let plan = plan()?;
                let root = crate::planner::lower_execution_plan(&plan)?;
                let tree = format_operator_tree(&root);
                let join = root.children()[0]
                    .downcast_ref::<PhysicalSymmetricHashJoin>()
                    .unwrap_or_else(|| panic!("{context}\n{tree}"));
                assert!(join.determines_rows_by_order(), "{context}\n{tree}");
                let actual = run_native(&root, plan, task_ctx()).await?;
                assert_eq!(num_rows(&actual.batches), 1, "{context}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_of_streams_sorted_descending_determine_their_rows() -> Result<()> {
        // Streams sorted by descending times, NULLs first: the times still to
        // come are not after the last one of their stream, so that with
        // `lt < rt` the left rows whose times are not before the last right
        // time are determined (and the other way round with `lt > rt`). The
        // reference is DataFusion's hash join of the same rows.
        let order = |side: &str| {
            let time = col(&format!("{side}t"), &timed_schema(side)).unwrap();
            LexOrdering::new([PhysicalSortExpr::new(time, SortOptions::new(true, true))])
                .unwrap()
        };
        let rows = |side: &str, rows: i64, every: i64, key: fn(i64) -> Option<i32>| {
            timed_batches(
                side,
                0..rows,
                key,
                |id| (id >= 3).then_some((rows - id) / every),
                |id| format!("{side}{id}"),
                50,
            )
        };
        let stream = |side: &str, batches: Vec<RecordBatch>, forever: bool| {
            let partition = BatchesPartition {
                schema: timed_schema(side),
                batches,
                forever,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    timed_schema(side),
                    vec![Arc::new(partition)],
                    None,
                    [order(side)],
                    true,
                    None,
                )
                .unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        let join = |left: Arc<dyn ExecutionPlan>,
                    right: Arc<dyn ExecutionPlan>,
                    join_type: JoinType,
                    filter: JoinFilter|
         -> Result<Arc<dyn ExecutionPlan>> {
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            Ok(Arc::new(SymmetricHashJoinExec::try_new(
                left,
                right,
                on,
                Some(filter),
                &join_type,
                NullEquality::NullEqualsNothing,
                Some(order("l")),
                Some(order("r")),
                StreamJoinPartitionMode::SinglePartition,
            )?))
        };
        let key = |id: i64| (id % 13 != 4).then_some((id % 7) as i32);
        for join_type in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::LeftAnti,
            JoinType::RightAnti,
            JoinType::LeftMark,
            JoinType::RightSemi,
        ] {
            for (name, filter) in time_filters() {
                let context = format!("{join_type} {name}");
                let bounded = |side: &str, n: i64, every: i64| {
                    MemorySourceConfig::try_new_exec(
                        &[rows(side, n, every, key)],
                        timed_schema(side),
                        None,
                    )
                    .unwrap() as Arc<dyn ExecutionPlan>
                };
                let reference = hash_join(
                    bounded("l", 600, 2),
                    bounded("r", 800, 3),
                    join_type,
                    Some(filter.clone()),
                    NullEquality::NullEqualsNothing,
                )?;
                let expected = EngineOutput {
                    schema: reference.schema(),
                    batches: with_timeout(collect(Arc::clone(&reference), task_ctx()))
                        .await?,
                    plan: reference,
                    ordered: false,
                };
                let plan = join(
                    stream("l", rows("l", 600, 2, key), false),
                    stream("r", rows("r", 800, 3, key), false),
                    join_type,
                    filter,
                )?;
                let root = lower(&plan)?;
                let tree = format_operator_tree(&root);
                let symmetric = root
                    .downcast_ref::<PhysicalSymmetricHashJoin>()
                    .unwrap_or_else(|| panic!("{context}\n{tree}"));
                assert!(symmetric.determines_rows_by_order(), "{context}\n{tree}");
                let actual = run_native(&root, plan, task_ctx()).await?;
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}\n{context}\n{tree}");
                }
            }
        }
        // Streams that never end, with keys that never match: under a limit,
        // the anti join emits a determined left row.
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let left = stream("l", rows("l", 100, 1, |id| Some((id * 2) as i32)), true);
            let right =
                stream("r", rows("r", 100, 1, |id| Some((id * 2 + 1) as i32)), true);
            let filter = time_filter(|lt, rt| binary(lt, Operator::Lt, rt));
            let join = join(left, right, JoinType::LeftAnti, filter)?;
            Ok(Arc::new(GlobalLimitExec::new(join, 0, Some(1))))
        };
        let expected = with_timeout(collect(plan()?, task_ctx())).await?;
        assert_eq!(num_rows(&expected), 1);
        let plan = plan()?;
        let root = crate::planner::lower_execution_plan(&plan)?;
        let actual = run_native(&root, plan, task_ctx()).await?;
        assert_eq!(num_rows(&actual.batches), 1);
        Ok(())
    }

    /// The schema of the rows `(id, k, t)` of the `side` input of the joins of
    /// float times.
    fn float_timed_schema(side: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(format!("{side}id"), DataType::Int64, false),
            Field::new(format!("{side}k"), DataType::Int32, false),
            Field::new(format!("{side}t"), DataType::Float64, false),
        ]))
    }

    /// The rows of the `side` input of the joins of float times: the times
    /// `times`, all of the key 1, in batches of one row.
    fn float_timed_batches(side: &str, times: &[f64]) -> Vec<RecordBatch> {
        times
            .iter()
            .zip(0..)
            .map(|(&time, id)| {
                RecordBatch::try_new(
                    float_timed_schema(side),
                    vec![
                        Arc::new(Int64Array::from(vec![id])),
                        Arc::new(Int32Array::from(vec![1])),
                        Arc::new(Float64Array::from(vec![time])),
                    ],
                )
                .unwrap()
            })
            .collect()
    }

    /// A filter of a join of float times, `expr` of their times `lt` and `rt`.
    fn float_time_filter(
        expr: impl Fn(PhysicalExprRef, PhysicalExprRef) -> PhysicalExprRef,
    ) -> JoinFilter {
        let schema = Arc::new(Schema::new(vec![
            Field::new("lt", DataType::Float64, false),
            Field::new("rt", DataType::Float64, false),
        ]));
        let (lt, rt) = (col("lt", &schema).unwrap(), col("rt", &schema).unwrap());
        JoinFilter::new(
            expr(lt, rt),
            vec![
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Right,
                },
            ],
            schema,
        )
    }

    /// Join the float times `left` and `right` (streams that end, of batches
    /// of one row, sorted by their times, descending if `descending`) on their
    /// keys with `filter`, and check the rows against DataFusion's hash join
    /// of the same rows, and whether the join determines its rows by the
    /// orders of its inputs against `determined`. Returns the number of rows.
    async fn assert_float_time_join(
        [left, right]: [&[f64]; 2],
        descending: bool,
        join_type: JoinType,
        filter: JoinFilter,
        determined: bool,
        context: &str,
    ) -> Result<usize> {
        let order = |side: &str| {
            let time = col(&format!("{side}t"), &float_timed_schema(side)).unwrap();
            LexOrdering::new([PhysicalSortExpr::new(
                time,
                SortOptions::new(descending, true),
            )])
            .unwrap()
        };
        let bounded = |side: &str, times: &[f64]| {
            MemorySourceConfig::try_new_exec(
                &[float_timed_batches(side, times)],
                float_timed_schema(side),
                None,
            )
            .unwrap() as Arc<dyn ExecutionPlan>
        };
        let reference = hash_join(
            bounded("l", left),
            bounded("r", right),
            join_type,
            Some(filter.clone()),
            NullEquality::NullEqualsNothing,
        )?;
        let expected = EngineOutput {
            schema: reference.schema(),
            batches: with_timeout(collect(Arc::clone(&reference), task_ctx())).await?,
            plan: reference,
            ordered: false,
        };
        let stream = |side: &str, times: &[f64]| {
            let partition = BatchesPartition {
                schema: float_timed_schema(side),
                batches: float_timed_batches(side, times),
                forever: false,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    float_timed_schema(side),
                    vec![Arc::new(partition)],
                    None,
                    [order(side)],
                    true,
                    None,
                )
                .unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        let (left, right) = (stream("l", left), stream("r", right));
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            Some(filter),
            &join_type,
            NullEquality::NullEqualsNothing,
            Some(order("l")),
            Some(order("r")),
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let root = lower(&plan)?;
        let tree = format_operator_tree(&root);
        let join = root
            .downcast_ref::<PhysicalSymmetricHashJoin>()
            .unwrap_or_else(|| panic!("{context}\n{tree}"));
        let actual = run_native(&root, plan, task_ctx()).await?;
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\n{context}\n{tree}");
        }
        assert_eq!(
            join.determines_rows_by_order(),
            determined,
            "{context}\n{tree}"
        );
        Ok(num_rows(&expected.batches))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn signed_zeros_are_equal_times() -> Result<()> {
        // The filter compares `-0.0` as `+0.0`: a left time `-0.0` matches the
        // right times `+0.0` that come after it, in batches of their own, also
        // once the bounds of the right times determine rows (DataFusion's hash
        // join is the reference).
        let filters = [
            (
                "lt = rt",
                float_time_filter(|lt, rt| binary(lt, Operator::Eq, rt)),
            ),
            (
                "lt >= rt AND lt <= rt",
                float_time_filter(|lt, rt| {
                    binary(
                        binary(Arc::clone(&lt), Operator::GtEq, Arc::clone(&rt)),
                        Operator::And,
                        binary(lt, Operator::LtEq, rt),
                    )
                }),
            ),
        ];
        for (name, filter) in filters {
            for join_type in [JoinType::Inner, JoinType::Left, JoinType::Full] {
                let context = format!("{join_type} {name}");
                let times: [&[f64]; 2] = [&[-0.0], &[0.0; 3]];
                let rows = assert_float_time_join(
                    times,
                    false,
                    join_type,
                    filter.clone(),
                    true,
                    &context,
                )
                .await?;
                assert_eq!(rows, 3, "{context}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn nan_of_arithmetic_determines_no_rows() -> Result<()> {
        // Both inputs have the times `0.0, 1.0, NaN` (in batches of one row, of
        // the same key), and the filters compute with them: `0.0 - NaN` is NaN,
        // which the comparisons order after every number, so that with
        // `(lt - rt) > 0.0` the left `0.0` matches the right NaN, which
        // interval arithmetic rules out once the right times reached `1.0` (it
        // bounds real numbers). The joins determine no rows by the orders of
        // their inputs (ascending or descending): their rows are those of
        // DataFusion's hash join, also the rows without a match.
        let filters = [
            (
                "(lt - rt) > 0.0",
                float_time_filter(|lt, rt| {
                    binary(binary(lt, Operator::Minus, rt), Operator::Gt, lit(0.0))
                }),
            ),
            (
                "(rt - lt) > 0.0",
                float_time_filter(|lt, rt| {
                    binary(binary(rt, Operator::Minus, lt), Operator::Gt, lit(0.0))
                }),
            ),
        ];
        for descending in [false, true] {
            let times = match descending {
                false => [0.0, 1.0, f64::NAN],
                true => [f64::NAN, 1.0, 0.0],
            };
            for (name, filter) in &filters {
                for join_type in [
                    JoinType::Inner,
                    JoinType::Left,
                    JoinType::Right,
                    JoinType::Full,
                    JoinType::LeftSemi,
                    JoinType::LeftAnti,
                    JoinType::RightSemi,
                    JoinType::RightAnti,
                ] {
                    let context = format!("{join_type} {name} descending={descending}");
                    let rows = assert_float_time_join(
                        [&times, &times],
                        descending,
                        join_type,
                        filter.clone(),
                        false,
                        &context,
                    )
                    .await?;
                    // `(0, NaN)`, `(1, 0)`, `(1, NaN)` and the NaN with every
                    // time, by the left times (and the other way round).
                    if join_type == JoinType::Inner {
                        assert_eq!(rows, 6, "{context}");
                    }
                }
            }
        }
        Ok(())
    }

    /// The rows of the `side` input of the joins of skewed lists: 100 rows of
    /// the key 1 (`<side>k`) whose lists (`<side>l`) are empty, but for the
    /// first one, of 128 Ki values (1 MiB).
    pub(crate) fn skewed_list_batch(side: &str) -> RecordBatch {
        let item = Arc::new(Field::new_list_field(DataType::Int64, false));
        let schema = Arc::new(Schema::new(vec![
            Field::new(format!("{side}id"), DataType::Int64, false),
            Field::new(format!("{side}k"), DataType::Int32, false),
            Field::new(format!("{side}l"), DataType::List(Arc::clone(&item)), false),
        ]));
        let big = 128 * 1024;
        let offsets = std::iter::once(0)
            .chain(std::iter::repeat_n(big, 100))
            .collect::<Vec<i32>>();
        let values = Int64Array::from_iter_values(0..i64::from(big));
        let lists = ListArray::new(
            item,
            arrow::buffer::OffsetBuffer::new(offsets.into()),
            Arc::new(values),
            None,
        );
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from(vec![1; 100])),
                Arc::new(lists),
            ],
        )
        .unwrap()
    }

    /// The big lists of the output of a join of skewed lists (of either input),
    /// in every batch of `batches`.
    pub(crate) fn big_lists(batches: &[RecordBatch]) -> Vec<usize> {
        batches
            .iter()
            .map(|batch| {
                ["ll", "rl"]
                    .into_iter()
                    .map(|name| {
                        let lists = batch.column_by_name(name).unwrap().as_list::<i32>();
                        (0..lists.len())
                            .filter(|&row| lists.value_length(row) > 0)
                            .count()
                    })
                    .sum()
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn skewed_lists_are_reserved_by_the_row() -> Result<()> {
        // The first of the 100 rows of either input has a list of 1 MiB, and
        // all rows match the 100 rows of the other input: the 199 pairs with
        // a big list have 200 MiB of lists, whichever input's rows come first
        // (and are probed by the rows of the other one). The join reserves the
        // bytes of every row that it gathers (with the positions of the items
        // of its lists that the gathering collects, 2 MiB for a big list), so
        // that a part has one pair with a big list at most: a pool of 2.5 MiB
        // holds the tables (2 MiB), but not a part with a big list; one of
        // 10 MiB holds both (6 MiB for the pair of the two big lists), also
        // without the limit.
        let plan = |limit: Option<usize>| -> Result<Arc<dyn ExecutionPlan>> {
            let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
                let batch = skewed_list_batch(side);
                let partition = BatchesPartition {
                    schema: batch.schema(),
                    batches: vec![batch.clone()],
                    forever: limit.is_some(),
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        batch.schema(),
                        vec![Arc::new(partition)],
                        None,
                        vec![],
                        true,
                        None,
                    )
                    .unwrap(),
                )
            };
            let join = symmetric_join(
                stream("l"),
                stream("r"),
                JoinType::Inner,
                None,
                NullEquality::NullEqualsNothing,
            )?;
            Ok(match limit {
                Some(limit) => Arc::new(GlobalLimitExec::new(join, 0, Some(limit))),
                None => join,
            })
        };
        let limited = plan(Some(1))?;
        let root = crate::planner::lower_execution_plan(&limited)?;
        let err = run_native(&root, limited, limited_task_ctx(5 << 19)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let limited = plan(Some(1))?;
        let root = crate::planner::lower_execution_plan(&limited)?;
        let output = run_native(&root, limited, limited_task_ctx(10 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 1);
        let all = plan(None)?;
        let root = lower(&all)?;
        let output = run_native(&root, all, limited_task_ctx(10 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 10_000);
        let big = big_lists(&output.batches);
        assert_eq!(big.iter().sum::<usize>(), 200);
        assert!(big.iter().all(|&big| big <= 2), "{big:?}");
        Ok(())
    }

    /// The rows of the `side` input of the joins of sparse unions: 100 rows
    /// of the key 1 (`<side>k`) whose sparse unions (`<side>u`) are integers,
    /// of a union whose other child has strings of 1 KiB.
    pub(crate) fn sparse_union_batch(side: &str) -> RecordBatch {
        let fields = UnionFields::try_new(
            vec![0, 1],
            vec![
                Field::new("i", DataType::Int32, false),
                Field::new("s", DataType::Utf8, false),
            ],
        )
        .unwrap();
        let union = UnionArray::try_new(
            fields.clone(),
            vec![0_i8; 100].into(),
            None,
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|id| format!("{id:>1024}")),
                )),
            ],
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new(format!("{side}id"), DataType::Int64, false),
            Field::new(format!("{side}k"), DataType::Int32, false),
            Field::new(
                format!("{side}u"),
                DataType::Union(fields, UnionMode::Sparse),
                false,
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from(vec![1; 100])),
                Arc::new(union),
            ],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sparse_unions_are_reserved_with_all_their_children() -> Result<()> {
        // 100 rows on either side with the same key and sparse unions whose
        // values are integers, of a union whose other child has strings of 1
        // KiB: the join gathers the strings of both children of every row (as
        // the kernels gather sparse unions), 2 KiB a pair, which it reserves,
        // in parts of about a MiB. A pool of 512 KiB holds the tables, but not
        // a part; one of 4 MiB holds both.
        let plan = |limit: Option<usize>| -> Result<Arc<dyn ExecutionPlan>> {
            let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
                let batch = sparse_union_batch(side);
                let batches = (0..4).map(|batch_| batch.slice(batch_ * 25, 25)).collect();
                let partition = BatchesPartition {
                    schema: batch.schema(),
                    batches,
                    forever: limit.is_some(),
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        batch.schema(),
                        vec![Arc::new(partition)],
                        None,
                        vec![],
                        true,
                        None,
                    )
                    .unwrap(),
                )
            };
            let join = symmetric_join(
                stream("l"),
                stream("r"),
                JoinType::Inner,
                None,
                NullEquality::NullEqualsNothing,
            )?;
            Ok(match limit {
                Some(limit) => Arc::new(GlobalLimitExec::new(join, 0, Some(limit))),
                None => join,
            })
        };
        let limited = plan(Some(1))?;
        let root = crate::planner::lower_execution_plan(&limited)?;
        let err = run_native(&root, limited, limited_task_ctx(512 * 1024)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let all = plan(None)?;
        let root = lower(&all)?;
        let output = run_native(&root, all, limited_task_ctx(4 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 10_000);
        for batch in &output.batches {
            assert!(
                batch.num_rows() <= PART_BYTES / 2048 + 1,
                "{}",
                batch.num_rows()
            );
        }
        Ok(())
    }

    /// The schema of the right input of the outer joins whose NULLs are big:
    /// its rows have a fixed size list of a million integers (8 MB).
    pub(crate) fn big_null_schema() -> SchemaRef {
        let item = Arc::new(Field::new_list_field(DataType::Int64, true));
        Arc::new(Schema::new(vec![
            Field::new("rid", DataType::Int64, false),
            Field::new("rk", DataType::Int32, false),
            Field::new("rl", DataType::FixedSizeList(item, 1_000_000), true),
        ]))
    }

    /// The left input of the outer joins whose NULLs are big: a row of the key
    /// 1.
    pub(crate) fn one_left_row() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("lid", DataType::Int64, false),
                Field::new("lk", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![1])),
            ],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn nulls_of_outer_joins_are_reserved() -> Result<()> {
        // The left row of a `Left` join of an empty right input has NULLs for
        // the right columns, of a fixed size list of a million integers: the
        // NULLs take 8 MB, which the join reserves before it builds them, so
        // that it fails in a pool of 64 KiB, and runs in one of 64 MiB.
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let stream = |batches: Vec<RecordBatch>, schema: SchemaRef| {
                let partition = BatchesPartition {
                    schema: Arc::clone(&schema),
                    batches,
                    forever: false,
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        schema,
                        vec![Arc::new(partition)],
                        None,
                        vec![],
                        true,
                        None,
                    )
                    .unwrap(),
                ) as Arc<dyn ExecutionPlan>
            };
            let left = one_left_row();
            symmetric_join(
                stream(vec![left.clone()], left.schema()),
                stream(vec![], big_null_schema()),
                JoinType::Left,
                None,
                NullEquality::NullEqualsNothing,
            )
        };
        let join = plan()?;
        let root = lower(&join)?;
        assert!(root.is::<PhysicalSymmetricHashJoin>());
        let err = run_native(&root, join, limited_task_ctx(64 * 1024)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let join = plan()?;
        let root = lower(&join)?;
        let output = run_native(&root, join, limited_task_ctx(64 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn every_determination_starts_from_the_filter_as_built() -> Result<()> {
        // A `Full` join of rows of the same key, with the filter
        // `lt < lx AND lx < rt` of the ascending left times `lt`, the left
        // values `lx`, and the descending right times `rt`: the bounds that a
        // determination derives for `lx` (a column of the filter that no input
        // is sorted by) hold for its bounds only, not for the next ones. The
        // rows `(26, 27)` and `29` match, as with DataFusion's hash join.
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("lid", DataType::Int64, false),
            Field::new("lk", DataType::Int32, false),
            Field::new("lt", DataType::Int64, false),
            Field::new("lx", DataType::Int64, false),
        ]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("rid", DataType::Int64, false),
            Field::new("rk", DataType::Int32, false),
            Field::new("rt", DataType::Int64, false),
        ]));
        let left_rows = [(5, 22), (24, 24), (26, 27)];
        let left_batch = |rows: &[(i64, i64)], first: i64| {
            RecordBatch::try_new(
                Arc::clone(&left_schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        first..first + rows.len() as i64,
                    )),
                    Arc::new(Int32Array::from(vec![1; rows.len()])),
                    Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
                    Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1))),
                ],
            )
            .unwrap()
        };
        let right_batch = RecordBatch::try_new(
            Arc::clone(&right_schema),
            vec![
                Arc::new(Int64Array::from(vec![0, 1])),
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int64Array::from(vec![29, 23])),
            ],
        )?;
        let filter = || {
            let schema = Arc::new(Schema::new(vec![
                Field::new("lt", DataType::Int64, false),
                Field::new("lx", DataType::Int64, false),
                Field::new("rt", DataType::Int64, false),
            ]));
            let [lt, lx, rt] = ["lt", "lx", "rt"].map(|name| col(name, &schema).unwrap());
            JoinFilter::new(
                binary(
                    binary(lt, Operator::Lt, Arc::clone(&lx)),
                    Operator::And,
                    binary(lx, Operator::Lt, rt),
                ),
                vec![
                    ColumnIndex {
                        index: 2,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 3,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 2,
                        side: JoinSide::Right,
                    },
                ],
                schema,
            )
        };
        let order = |schema: &SchemaRef, name: &str, descending: bool| {
            LexOrdering::new([PhysicalSortExpr::new(
                col(name, schema).unwrap(),
                SortOptions::new(descending, false),
            )])
            .unwrap()
        };
        let on = |left: &SchemaRef, right: &SchemaRef| {
            vec![(col("lk", left).unwrap(), col("rk", right).unwrap())]
        };
        let reference = hash_join(
            MemorySourceConfig::try_new_exec(
                &[vec![left_batch(&left_rows, 0)]],
                Arc::clone(&left_schema),
                None,
            )?,
            MemorySourceConfig::try_new_exec(
                &[vec![right_batch.clone()]],
                Arc::clone(&right_schema),
                None,
            )?,
            JoinType::Full,
            Some(filter()),
            NullEquality::NullEqualsNothing,
        )?;
        let expected = EngineOutput {
            schema: reference.schema(),
            batches: with_timeout(collect(Arc::clone(&reference), task_ctx())).await?,
            plan: reference,
            ordered: false,
        };
        assert_eq!(num_rows(&expected.batches), 4);
        let stream =
            |schema: &SchemaRef, batches: Vec<RecordBatch>, order: LexOrdering| {
                let partition = BatchesPartition {
                    schema: Arc::clone(schema),
                    batches,
                    forever: false,
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        Arc::clone(schema),
                        vec![Arc::new(partition)],
                        None,
                        [order],
                        true,
                        None,
                    )
                    .unwrap(),
                ) as Arc<dyn ExecutionPlan>
            };
        let left_batches = left_rows
            .iter()
            .enumerate()
            .map(|(id, row)| left_batch(std::slice::from_ref(row), id as i64))
            .collect();
        let left = stream(&left_schema, left_batches, order(&left_schema, "lt", false));
        let right = stream(
            &right_schema,
            vec![right_batch],
            order(&right_schema, "rt", true),
        );
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on(&left_schema, &right_schema),
            Some(filter()),
            &JoinType::Full,
            NullEquality::NullEqualsNothing,
            Some(order(&left_schema, "lt", false)),
            Some(order(&right_schema, "rt", true)),
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let root = lower(&plan)?;
        let join = root.downcast_ref::<PhysicalSymmetricHashJoin>().unwrap();
        assert!(join.determines_rows_by_order());
        let actual = run_native(&root, plan, task_ctx()).await?;
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}");
        }
        Ok(())
    }

    /// The key of the joins of wide computed keys: a string of 16 KiB.
    pub(crate) fn wide_key() -> PhysicalExprRef {
        lit("k".repeat(16 * 1024))
    }

    /// The rows `(id)` of the `side` input of the joins of wide computed keys.
    pub(crate) fn id_rows(side: &str, rows: i64) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                format!("{side}id"),
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from_iter_values(0..rows))],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wide_computed_keys_are_reserved() -> Result<()> {
        // A join of 1,024 rows with a row on a computed key, a string of 16 KiB:
        // the keys of a chunk take 16 MB as values and 16 MB encoded, which the
        // join reserves while it encodes and joins them, so that it fails in a
        // pool of 256 KiB, and runs in one of 64 MiB.
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let stream = |batch: RecordBatch| {
                let partition = BatchesPartition {
                    schema: batch.schema(),
                    batches: vec![batch.clone()],
                    forever: false,
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        batch.schema(),
                        vec![Arc::new(partition)],
                        None,
                        vec![],
                        true,
                        None,
                    )
                    .unwrap(),
                ) as Arc<dyn ExecutionPlan>
            };
            Ok(Arc::new(SymmetricHashJoinExec::try_new(
                stream(id_rows("l", 1024)),
                stream(id_rows("r", 1)),
                vec![(wide_key(), wide_key())],
                None,
                &JoinType::Inner,
                NullEquality::NullEqualsNothing,
                None,
                None,
                StreamJoinPartitionMode::SinglePartition,
            )?))
        };
        let join = plan()?;
        let root = lower(&join)?;
        assert!(root.is::<PhysicalSymmetricHashJoin>());
        let err = run_native(&root, join, limited_task_ctx(256 * 1024)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let join = plan()?;
        let root = lower(&join)?;
        let output = run_native(&root, join, limited_task_ctx(64 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 1024);
        Ok(())
    }

    /// The rows `(id, k)` of the `side` input of the joins of shared views:
    /// `rows` rows whose keys are the same string of 1 MiB, views of a single
    /// buffer.
    pub(crate) fn shared_view_rows(side: &str, rows: usize) -> RecordBatch {
        let one = StringViewArray::from(vec!["v".repeat(1 << 20)]);
        let keys =
            arrow::compute::take(&one, &UInt32Array::from(vec![0; rows]), None).unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Utf8View, false),
            ])),
            vec![Arc::new(Int64Array::from_iter_values(0..rows as i64)), keys],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn encoded_keys_of_shared_views_are_reserved() -> Result<()> {
        // 128 keys that are views of the same string of 1 MiB take 1 MiB as
        // values, but 128 MiB encoded (the encoding copies the bytes of every
        // view): the join reserves the encoding before it encodes them, so
        // that it fails in a pool of 8 MiB, also against an empty input; 8
        // such keys join in a pool of 64 MiB.
        let plan = |rows: usize| -> Result<Arc<dyn ExecutionPlan>> {
            let stream = |batch: RecordBatch, empty: bool| {
                let partition = BatchesPartition {
                    schema: batch.schema(),
                    batches: match empty {
                        true => vec![],
                        false => vec![batch.clone()],
                    },
                    forever: false,
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        batch.schema(),
                        vec![Arc::new(partition)],
                        None,
                        vec![],
                        true,
                        None,
                    )
                    .unwrap(),
                ) as Arc<dyn ExecutionPlan>
            };
            symmetric_join(
                stream(shared_view_rows("l", rows), false),
                stream(shared_view_rows("r", 1), rows > 8),
                JoinType::Inner,
                None,
                NullEquality::NullEqualsNothing,
            )
        };
        let join = plan(128)?;
        let root = lower(&join)?;
        assert!(root.is::<PhysicalSymmetricHashJoin>());
        let err = run_native(&root, join, limited_task_ctx(8 << 20)?)
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let join = plan(8)?;
        let root = lower(&join)?;
        let output = run_native(&root, join, limited_task_ctx(64 << 20)?).await?;
        assert_eq!(num_rows(&output.batches), 8);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn computed_signed_zeros_determine_no_rows() -> Result<()> {
        // Integer times, all zero, by which the inputs are sorted as doubles,
        // and the filter `-CAST(lt AS DOUBLE) = CAST(rt AS DOUBLE)`, which
        // computes `-0.0` of the left times: the filter compares it as `+0.0`,
        // interval arithmetic before it, so that the join does not determine
        // rows by the orders, and the left row matches the three right rows (in
        // batches of their own), as with DataFusion's hash join.
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Int32, false),
                Field::new(format!("{side}t"), DataType::Int64, false),
            ])) as SchemaRef
        };
        let rows = |side: &str, ids: Range<i64>| {
            let n = (ids.end - ids.start) as usize;
            RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids)),
                    Arc::new(Int32Array::from(vec![1; n])),
                    Arc::new(Int64Array::from(vec![0; n])),
                ],
            )
            .unwrap()
        };
        let double = |expr: PhysicalExprRef| -> PhysicalExprRef {
            Arc::new(CastExpr::new(expr, DataType::Float64, None))
        };
        let filter = || {
            let filter_schema = Arc::new(Schema::new(vec![
                Field::new("lt", DataType::Int64, false),
                Field::new("rt", DataType::Int64, false),
            ]));
            let (lt, rt) = (
                col("lt", &filter_schema).unwrap(),
                col("rt", &filter_schema).unwrap(),
            );
            let negated: PhysicalExprRef = Arc::new(NegativeExpr::new(double(lt)));
            JoinFilter::new(
                binary(negated, Operator::Eq, double(rt)),
                vec![
                    ColumnIndex {
                        index: 2,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 2,
                        side: JoinSide::Right,
                    },
                ],
                filter_schema,
            )
        };
        let order = |side: &str| {
            let time = col(&format!("{side}t"), &schema(side)).unwrap();
            LexOrdering::new([PhysicalSortExpr::new(
                double(time),
                SortOptions::new(false, false),
            )])
            .unwrap()
        };
        let stream = |side: &str, batches: Vec<RecordBatch>| {
            let partition = BatchesPartition {
                schema: schema(side),
                batches,
                forever: false,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema(side),
                    vec![Arc::new(partition)],
                    None,
                    [order(side)],
                    true,
                    None,
                )
                .unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        let reference = hash_join(
            MemorySourceConfig::try_new_exec(
                &[vec![rows("l", 0..1)]],
                schema("l"),
                None,
            )?,
            MemorySourceConfig::try_new_exec(
                &[vec![rows("r", 0..3)]],
                schema("r"),
                None,
            )?,
            JoinType::Inner,
            Some(filter()),
            NullEquality::NullEqualsNothing,
        )?;
        let expected = EngineOutput {
            schema: reference.schema(),
            batches: with_timeout(collect(Arc::clone(&reference), task_ctx())).await?,
            plan: reference,
            ordered: false,
        };
        assert_eq!(num_rows(&expected.batches), 3);
        let left = stream("l", vec![rows("l", 0..1)]);
        let right = stream("r", (0..3).map(|id| rows("r", id..id + 1)).collect());
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            Some(filter()),
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            Some(order("l")),
            Some(order("r")),
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let root = lower(&plan)?;
        let join = root.downcast_ref::<PhysicalSymmetricHashJoin>().unwrap();
        assert!(!join.determines_rows_by_order());
        let actual = run_native(&root, plan, task_ctx()).await?;
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn keys_of_dictionaries_of_shared_views_are_reserved_before_they_are_encoded()
    -> Result<()> {
        // A symmetric join of streams of the rows of `dictionary_view_rows`, in
        // a pool of 8 MiB: the join reserves the encoding of their keys (128
        // MiB) before it encodes them, which the pool refuses. The query runs
        // on the thread of the test (a runtime of the current thread runs all
        // its tasks there), which allocates far fewer bytes meanwhile than the
        // encoding takes.
        let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
            let batch = dictionary_view_rows(side);
            let schema = batch.schema();
            let partition = BatchesPartition {
                schema: Arc::clone(&schema),
                batches: vec![batch],
                forever: false,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema,
                    vec![Arc::new(partition)],
                    None,
                    [],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let (left, right) = (stream("l"), stream("r"));
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let root = lower(&plan)?;
        assert!(root.is::<PhysicalSymmetricHashJoin>());
        allocations::reset_peak();
        let before = allocations::held();
        let err = run_native(&root, plan, limited_task_ctx(8 << 20)?)
            .await
            .unwrap_err();
        let allocated = allocations::peak() - before;
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        assert!(
            err.to_string().contains("PhysicalSymmetricHashJoin"),
            "{err}"
        );
        assert!(
            allocated < 16 << 20,
            "the query allocated {allocated} bytes"
        );
        Ok(())
    }

    #[tokio::test]
    async fn tables_pruned_to_a_row_free_their_capacity() -> Result<()> {
        // Streams of 100,000 rows (a batch) and then a row, of distinct keys
        // that never match, sorted by their times, joined with `lt = rt`, which
        // then produce nothing, forever. The left times are the ids, the right
        // ones 0, but for the last right row (100,000): the join keeps the
        // left rows until that row, a window of 100,000 rows, and drops all
        // of them then but the last one (a batch of its own). The structures
        // of the table (the hash table of 100,000 keys, ...) shrink, and the
        // join reserves what they allocate: while it waits for more rows, the
        // bytes that the query holds (on the thread of the test, as a runtime
        // of the current thread runs all its tasks there) are about those
        // that it reserves, and few (and while it held the window, it reserved
        // it).
        let stream = |side: &str,
                      key: fn(i64) -> Option<i32>,
                      time: fn(i64) -> Option<i64>| {
            let text = |_| String::new();
            let mut batches = timed_batches(side, 0..100_000, key, time, text, 100_000);
            batches.extend(timed_batches(side, 100_000..100_001, key, Some, text, 1));
            timed_stream(side, batches, true, true)
        };
        let plan = timed_join(
            stream("l", |id| Some(id as i32), Some),
            stream("r", |id| Some(id as i32 + 1_000_000), |_| Some(0)),
            JoinType::Inner,
            Some(time_filter(|lt, rt| binary(lt, Operator::Eq, rt))),
            NullEquality::NullEqualsNothing,
            true,
        )?;
        let root = lower(&plan)?;
        let join = root.downcast_ref::<PhysicalSymmetricHashJoin>().unwrap();
        assert!(join.determines_rows_by_order());
        let pool = Arc::new(PeakPool::new(1 << 30));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
            .build_arc()?;
        let ctx = Arc::new(TaskContext::default().with_runtime(runtime));
        let exec = PipelineRootExec::try_new(root, plan)?;
        allocations::reset_peak();
        let before = allocations::held();
        let mut output = exec.execute(0, ctx)?;
        // The join reads all rows, and waits for more.
        let next = tokio::time::timeout(Duration::from_secs(2), output.next()).await;
        assert!(next.is_err(), "the join produced {next:?}");
        let held = allocations::held() - before;
        let reserved = pool.reserved() as isize;
        let most_held = allocations::peak() - before;
        let most_reserved = pool.peak.load(Ordering::Relaxed) as isize;
        // The window of 100,000 rows was reserved as long as it was held.
        assert!(
            most_held < most_reserved + (1 << 20),
            "the query held {most_held} bytes at most, and reserved {most_reserved}"
        );
        assert!(
            held < reserved + (1 << 20),
            "the query holds {held} bytes, and reserves {reserved}"
        );
        assert!(reserved < 1 << 20, "the join reserves {reserved} bytes");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn union_keys_are_joined_as_datafusion_compares_them() -> Result<()> {
        // Streams of union keys, with a NULL of the Int32 child on the left
        // and of the Utf8 child on the right: DataFusion compares them as equal
        // NULLs (also nested in lists and structs, where they match either
        // way), which the row format distinguishes by their types. The
        // symmetric join matches the rows of DataFusion's sort-merge join.
        for (name, keys) in union_key_layouts() {
            for null_equality in [
                NullEquality::NullEqualsNull,
                NullEquality::NullEqualsNothing,
            ] {
                for join_type in [JoinType::Inner, JoinType::Full] {
                    let context = format!("{name} {null_equality:?} {join_type}");
                    let expected =
                        reference_join_of_keys(&keys, join_type, null_equality).await?;
                    if join_type == JoinType::Inner {
                        assert_eq!(
                            num_rows(&expected.batches),
                            union_key_matches(name, null_equality),
                            "{context}"
                        );
                    }
                    let stream =
                        |side: &str, keys: &ArrayRef| -> Arc<dyn ExecutionPlan> {
                            let batch = keyed_rows(side, keys);
                            let schema = batch.schema();
                            let partition = BatchesPartition {
                                schema: Arc::clone(&schema),
                                batches: vec![batch],
                                forever: false,
                            };
                            Arc::new(
                                StreamingTableExec::try_new(
                                    schema,
                                    vec![Arc::new(partition)],
                                    None,
                                    [],
                                    true,
                                    None,
                                )
                                .unwrap(),
                            )
                        };
                    let (left, right) = (stream("l", &keys[0]), stream("r", &keys[1]));
                    let on =
                        vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
                    let plan: Arc<dyn ExecutionPlan> =
                        Arc::new(SymmetricHashJoinExec::try_new(
                            left,
                            right,
                            on,
                            None,
                            &join_type,
                            null_equality,
                            None,
                            None,
                            StreamJoinPartitionMode::SinglePartition,
                        )?);
                    let root = lower(&plan)?;
                    assert!(root.is::<PhysicalSymmetricHashJoin>(), "{context}");
                    let actual = run_native(&root, plan, task_ctx())
                        .await
                        .unwrap_or_else(|e| panic!("{context}: {e}"));
                    if let Err(message) = compare_outputs(&expected, &actual) {
                        panic!("Results differ: {message}\n{context}");
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn dictionaries_are_gathered_with_the_values_that_their_rows_use() -> Result<()>
    {
        // The right stream produces two batches of 50 rows of the key 1 whose
        // payloads have dictionaries of the values `"a"` and a string of 1 MiB
        // (2 MiB in all, built as the stream is read), whose rows all have
        // `"a"` (in every layout of `dictionary_layouts`), and the left stream,
        // a little later, two rows of the key 1: the symmetric join gathers the
        // right rows of the 200 pairs from both batches of its table, with the
        // values of their dictionaries that they use only (`interleave` copies
        // all values of all dictionaries), so that it allocates no more than it
        // reserves (on the thread of the test, as a runtime of the current
        // thread runs all tasks of the query there), also in a pool of 3 MiB. So
        // do the NULL keys of dictionaries without values of fixed size lists of
        // 1,000,000 integers, in a pool of 64 KiB: the join gathers no NULL value
        // of 8 MB for them (see `gather_cases`).
        #[derive(Debug)]
        struct LatePartition;
        impl PartitionStream for LatePartition {
            fn schema(&self) -> &SchemaRef {
                static SCHEMA: std::sync::OnceLock<SchemaRef> =
                    std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| dictionary_schema("l"))
            }
            fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
                let rows = futures::stream::once(async {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok(dictionary_probe_rows())
                });
                Box::pin(RecordBatchStreamAdapter::new(dictionary_schema("l"), rows))
            }
        }
        for case in gather_cases() {
            for &pool_size in &case.pools {
                let context = format!("{} in a pool of {pool_size} bytes", case.name);
                let stream = |schema: SchemaRef, partition: Arc<dyn PartitionStream>| {
                    Arc::new(
                        StreamingTableExec::try_new(
                            schema,
                            vec![partition],
                            None,
                            [],
                            true,
                            None,
                        )
                        .unwrap(),
                    ) as Arc<dyn ExecutionPlan>
                };
                let left = stream(dictionary_schema("l"), Arc::new(LatePartition));
                let right_schema = payload_schema("r", &case.payload);
                let right = stream(
                    Arc::clone(&right_schema),
                    Arc::new(BuiltPartition {
                        schema: right_schema,
                        rows: Arc::clone(&case.rows),
                    }),
                );
                let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
                let plan: Arc<dyn ExecutionPlan> =
                    Arc::new(SymmetricHashJoinExec::try_new(
                        left,
                        right,
                        on,
                        None,
                        &JoinType::Inner,
                        NullEquality::NullEqualsNothing,
                        None,
                        None,
                        StreamJoinPartitionMode::SinglePartition,
                    )?);
                let root = lower(&plan)?;
                assert!(root.is::<PhysicalSymmetricHashJoin>(), "{context}");
                let pool = Arc::new(PeakPool::new(pool_size));
                let runtime = RuntimeEnvBuilder::new()
                    .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
                    .build_arc()?;
                let ctx = Arc::new(TaskContext::default().with_runtime(runtime));
                allocations::reset_peak();
                let before = allocations::held();
                let output = run_native(&root, plan, ctx).await;
                let allocated = allocations::peak() - before;
                let reserved = pool.peak.load(Ordering::Relaxed) as isize;
                match case.matches {
                    Some(matches) => {
                        let output = output.unwrap_or_else(|e| panic!("{context}: {e}"));
                        assert_eq!(num_rows(&output.batches), matches, "{context}");
                        assert!(
                            allocated < reserved + (512 << 10),
                            "{context}: the query allocated {allocated} bytes, and reserved {reserved}"
                        );
                    }
                    // Refused before the query allocates more than the pool.
                    None => {
                        let err = output.expect_err(&context);
                        assert!(
                            matches!(
                                err.find_root(),
                                DataFusionError::ResourcesExhausted(_)
                            ),
                            "{context}: {err}"
                        );
                        assert!(
                            allocated < pool_size as isize,
                            "{context}: the query allocated {allocated} bytes"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dictionaries_without_values_are_joined() -> Result<()> {
        // Dictionaries without values (whose keys are all NULL, which is
        // valid; `normalized_keys` panics for them), as the payloads of the
        // rows and as the keys of streams: the symmetric join returns the rows
        // of DataFusion's join.
        for (name, rows, null_equality, matches) in empty_dictionary_cases() {
            let context = format!("{name} {null_equality:?}");
            let expected = reference_join_of_rows(&rows, null_equality).await?;
            assert_eq!(num_rows(&expected.batches), matches, "{context}");
            let stream = |batch: &RecordBatch| -> Arc<dyn ExecutionPlan> {
                let partition = BatchesPartition {
                    schema: batch.schema(),
                    batches: vec![batch.clone()],
                    forever: false,
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        batch.schema(),
                        vec![Arc::new(partition)],
                        None,
                        [],
                        true,
                        None,
                    )
                    .unwrap(),
                )
            };
            let (left, right) = (stream(&rows[0]), stream(&rows[1]));
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            let plan: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
                left,
                right,
                on,
                None,
                &JoinType::Inner,
                null_equality,
                None,
                None,
                StreamJoinPartitionMode::SinglePartition,
            )?);
            let root = lower(&plan)?;
            assert!(root.is::<PhysicalSymmetricHashJoin>(), "{context}");
            let actual = run_native(&root, plan, task_ctx())
                .await
                .unwrap_or_else(|e| panic!("{context}: {e}"));
            if let Err(message) = compare_outputs(&expected, &actual) {
                panic!("Results differ: {message}\n{context}");
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn joins_without_output_let_other_tasks_run() -> Result<()> {
        // A symmetric join of streams of 4,000 rows of one key each, with a
        // filter that rejects all of their 16,000,000 pairs (joined in parts of
        // 2,048 of them): the join yields after a few parts without output,
        // so that the other tasks of the runtime of the current thread run (a
        // timer of 10 ms fires about in time), and the join is cancelled when
        // they abort it.
        let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
            let partition = BatchesPartition {
                schema: fanout_schema(side),
                batches: vec![fanout_rows(side, 4000)],
                forever: false,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    fanout_schema(side),
                    vec![Arc::new(partition)],
                    None,
                    [],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let (left, right) = (stream("l"), stream("r"));
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            Some(rejecting_filter()),
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let root = lower(&plan)?;
        assert!(root.is::<PhysicalSymmetricHashJoin>());
        let (waited, cancelled) = time_beside(root, plan).await?;
        assert!(
            waited < Duration::from_secs(1),
            "a timer of 10 ms fired after {waited:?}"
        );
        assert!(cancelled);
        Ok(())
    }

    #[tokio::test]
    async fn literal_keys_are_reserved_before_their_rows_are() -> Result<()> {
        // A symmetric join of 1,024 rows with a row on a literal key, a string
        // of 16 KiB: the keys of all rows (16 MB as values, and as much
        // encoded) are reserved before they are arrays (the literal is a
        // single value), so that in a pool of 256 KiB the join fails with
        // `ResourcesExhausted` before it allocates them (on the thread of the
        // test, as a runtime of the current thread runs all tasks of the query
        // there).
        let stream = |batch: RecordBatch| {
            let partition = BatchesPartition {
                schema: batch.schema(),
                batches: vec![batch.clone()],
                forever: false,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    batch.schema(),
                    vec![Arc::new(partition)],
                    None,
                    vec![],
                    true,
                    None,
                )
                .unwrap(),
            ) as Arc<dyn ExecutionPlan>
        };
        let join: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            stream(id_rows("l", 1024)),
            stream(id_rows("r", 1)),
            vec![(wide_key(), wide_key())],
            None,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let root = lower(&join)?;
        assert!(root.is::<PhysicalSymmetricHashJoin>());
        allocations::reset_peak();
        let before = allocations::held();
        let err = run_native(&root, join, limited_task_ctx(256 << 10)?)
            .await
            .unwrap_err();
        let allocated = allocations::peak() - before;
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        assert!(
            err.to_string().contains("PhysicalSymmetricHashJoin"),
            "{err}"
        );
        assert!(allocated < 1 << 20, "the query allocated {allocated} bytes");
        Ok(())
    }

    #[tokio::test]
    async fn sliced_rows_are_kept_in_copies_of_their_rows() -> Result<()> {
        // A symmetric join of streams that never end of 32 rows, every row a
        // zero-copy slice of a row of an array of 1 MiB of its own (see
        // `SlicedRows`): the tables keep copies of the rows only, not the
        // arrays that they are slices of (which the bytes of the rows, which
        // they reserve, do not count). In a pool of 4 MiB, the join joins all
        // rows, allocating less than 16 MiB (on the thread of the test, as a
        // runtime of the current thread runs all tasks of the query there),
        // while the arrays of all rows take 64 MiB. (On a thread, the nested
        // executions of the inputs queue at most two rows for the join; the
        // rows of the output are dropped as they are read.)
        let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
            let partition = SlicedRows::new(side, 32);
            let schema = Arc::clone(&partition.schema);
            Arc::new(
                StreamingTableExec::try_new(
                    schema,
                    vec![Arc::new(partition)],
                    None,
                    vec![],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let (left, right) = (stream("l"), stream("r"));
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let join: Arc<dyn ExecutionPlan> = Arc::new(SymmetricHashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            None,
            None,
            StreamJoinPartitionMode::SinglePartition,
        )?);
        let root = lower(&join)?;
        assert!(root.is::<PhysicalSymmetricHashJoin>());
        allocations::reset_peak();
        let before = allocations::held();
        let rows =
            count_native_rows(&root, join, limited_single_thread_ctx(4 << 20)?).await?;
        let allocated = allocations::peak() - before;
        assert_eq!(rows, 32);
        assert!(
            allocated < 16 << 20,
            "the query allocated {allocated} bytes"
        );
        Ok(())
    }
}
