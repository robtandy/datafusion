// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalMergeJoin`], and the native lowering of DataFusion's
//! `SortMergeJoinExec`.
//!
//! DataFusion plans a `SortMergeJoinExec` for an equi-join instead of a
//! `HashJoinExec` when hash joins are not preferred
//! (`datafusion.optimizer.prefer_hash_join = false`), and for the null-aware
//! `NOT IN` of a single key without filter (`ps_suppkey NOT IN (SELECT
//! s_suppkey ...)`). It sorts every partition of both inputs by the join keys
//! (or reads inputs sorted by them) and merges them, for every join type. Its
//! output keeps the order of its *streamed* input, the left one, or the right
//! one for the `Right`, `RightSemi`, `RightAnti` and `RightMark` joins
//! (`SortMergeJoinExec::probe_side`), with the matches of every streamed row
//! in the order of the other, *buffered*, input; a `Full` join has no order.
//!
//! # Lowering
//!
//! [`try_lower`] lowers every `SortMergeJoinExec` natively, whatever its
//! join type, keys, filter, null equality and embedded projection:
//!
//! * as a [`PhysicalHashJoin`] if no node above it relies on its order:
//!   joining needs no sorted inputs, so the sorts and the repartitionings
//!   below the join are removed, and the join builds its table on the smaller
//!   input (by the statistics of the inputs, as DataFusion's `JoinSelection`
//!   chooses the build side of a hash join), probing it with the other one;
//! * as a [`PhysicalMergeJoin`] if the order of its output is required: by a
//!   `SortPreservingMergeExec` above it that merges its sorted partitions
//!   (e.g. of `ORDER BY key NULLS FIRST`, through projections and filters,
//!   see [`ordered_join_below`]), which the merge join then replaces, or
//!   because the join has a single partition with an order, which the nodes
//!   above it (a limit, or the result) may rely on, like the null-aware
//!   `NOT IN` join (see [`promises_order`]);
//! * a join whose inputs never end (unbounded inputs) is a
//!   [`PhysicalHashJoin`] that builds on its bounded input, or a
//!   [`PhysicalSymmetricHashJoin`] if both are unbounded (see
//!   [`super::symmetric_hash_join`]), unless its order is required: the
//!   native joins read all of an input before they emit rows in its order.
//!
//! A join that the hash join does not implement (e.g. of keys or columns
//! that its hash table cannot store) is a merge join too.
//!
//! # `PhysicalMergeJoin`
//!
//! Like DuckDB's sort-merge joins (`PhysicalPiecewiseMergeJoin`), the merge
//! join sorts its buffered input once, in a pipeline breaker, and merges the
//! sorted rows of its streamed input against it in its streaming pipeline:
//!
//! ```text
//! MetaPipeline 0 (sink: ...) children: [1, 2]
//!   Pipeline 0: PhysicalSort -> PhysicalMergeJoin -> ... depends on meta pipelines [1, 2]
//!   Pipeline 4: PhysicalMergeJoin -> ... depends on pipelines [0]          (Full joins)
//! MetaPipeline 1 (sink: PhysicalMergeJoin) children: [3]
//!   Pipeline 1: PhysicalSort -> PhysicalMergeJoin depends on meta pipelines [3]
//! MetaPipeline 2 (sink: PhysicalSort)
//!   Pipeline 2: streamed input -> PhysicalSort
//! MetaPipeline 3 (sink: PhysicalSort)
//!   Pipeline 3: buffered input -> PhysicalSort
//! ```
//!
//! * Both inputs are sorted by parallel [`PhysicalSort`]s, which replace the
//!   sorts and the repartitionings below the `SortMergeJoinExec`: by the
//!   order of the input of the `SortMergeJoinExec` (its sort, or the order of
//!   a sorted input), which the join keeps. Floating point keys are sorted
//!   with `-0.0` and `+0.0` as one value, as the join compares them.
//! * **Sink** (the buffered input): a single thread collects the sorted rows
//!   of the buffered input, and their keys encoded in the Arrow row format
//!   (in the order of the join keys), in which comparing two keys is a
//!   `memcmp`. The rows are reserved in the memory pool of the session: the
//!   join fails with `ResourcesExhausted` if the pool refuses them.
//! * **Operator** (the streamed input): a single thread reads the sorted
//!   streamed rows and merges them with the buffered rows: a cursor moves
//!   forward through the buffered keys (by galloping, so that runs of buffered
//!   keys without a streamed match are skipped in logarithmic time), and the
//!   streamed rows of a key are joined with the buffered rows of that key
//!   (with the residual filter evaluated on the pairs, many streamed rows at
//!   once). The output keeps the order of the streamed rows, with the matches
//!   of every streamed row in the order of the buffered rows: a pipeline that
//!   reads it in order on a single thread emits the rows in the order of the
//!   `SortMergeJoinExec`.
//! * **Source** (`Full` joins only): once the streamed input ends, the threads
//!   of a last pipeline emit the buffered rows that found no match.
//!
//! Join keys are equal with DataFusion's join key equality: `-0.0 = +0.0`,
//! NaNs with the same bits are equal, and NULLs are equal only with
//! [`NullEquality::NullEqualsNull`]. The null-aware `LeftAnti` join (`NOT
//! IN`) emits no row if a buffered key is NULL, and the streamed rows with a
//! NULL key only if the buffered input is empty.
//!
//! The module also holds the parts that the merge join shares with the
//! [`PhysicalSymmetricHashJoin`]: the output of a join ([`JoinOutput`]), the
//! rows that it gathers ([`JoinRows`], [`StoredRows`]) and the evaluation of
//! its residual filter ([`filter_mask`]).
//!
//! [`PhysicalSymmetricHashJoin`]: super::symmetric_hash_join::PhysicalSymmetricHashJoin

use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    new_empty_array, new_null_array,
};
use arrow::buffer::NullBuffer;
use arrow::compute::{SortOptions, interleave, prep_null_mask_filter, take};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_row::{OwnedRow, Row, RowConverter, Rows, SortField};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::float_semantics::{has_float_leaf, normalize_floats};
use datafusion_common::{
    DataFusionError, JoinSide, JoinType, NullEquality, Result, internal_datafusion_err,
    internal_err, not_impl_err, plan_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::{
    LexOrdering, PhysicalExpr, PhysicalExprRef, PhysicalSortExpr,
};
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::SortMergeJoinExec;
use datafusion_physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, build_join_schema, swap_join_projection,
};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::statistics::{StatisticsArgs, StatisticsContext};
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties,
    ReplaceChildrenOptions,
};
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::join::PhysicalHashJoin;
use crate::join::symmetric_hash_join;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorResult, OperatorState, PhysicalOperator,
    SinkFinalize, SinkResult, SourceResult, downcast_state, downcast_state_mut,
};
use crate::operators::caching::CachingOperator;
use crate::operators::filter::PhysicalFilter;
use crate::operators::projection::PhysicalProjection;
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::{PhysicalSort, SortKeys, memory_size};

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

/// Lower a DataFusion `SortMergeJoinExec` natively, see the [module
/// documentation](self): to a [`PhysicalMergeJoin`] if `ordered` (a node above
/// it relies on its order, see [`ordered_join_below`]) or if it promises its
/// order to the nodes above it (see [`promises_order`]), and to a
/// [`PhysicalHashJoin`] (or, for inputs that never end, a
/// [`PhysicalSymmetricHashJoin`](super::symmetric_hash_join::PhysicalSymmetricHashJoin))
/// otherwise.
///
/// `lower_input` lowers an input of the join; its second argument tells
/// whether the input runs in the pipeline of the join (the probe side of a
/// hash join, the inputs of a symmetric hash join), which stops when the join
/// stops, or in a pipeline that reads all of its input (the build side of a
/// hash join, the sorted inputs of a merge join).
///
/// Returns `Ok(None)` for any other node, and for the joins that no native
/// join executes: joins whose keys have different types on the two sides,
/// whose keys or columns the native joins cannot store or sort (e.g. keys of
/// unions), and joins of inputs that never end whose order is required (or
/// that are null-aware), which the native joins would wait for forever.
pub fn try_lower<F>(
    plan: &Arc<dyn ExecutionPlan>,
    ordered: bool,
    mut lower_input: F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, bool) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(join) = plan.downcast_ref::<SortMergeJoinExec>() else {
        return Ok(None);
    };
    let unbounded =
        [join.left(), join.right()].map(|input| input.boundedness().is_unbounded());
    if ordered || promises_order(plan) {
        // The merge join reads all of both inputs before it emits a row.
        if unbounded.contains(&true) {
            return Ok(None);
        }
        let Some(spec) = MergeJoinSpec::of(join)? else {
            return Ok(None);
        };
        let join = lower_merge_join(join, spec, &mut lower_input)?;
        return Ok(join.map(|join| Arc::new(join) as Arc<dyn PhysicalOperator>));
    }
    let left = without_reordering(join.left())?;
    let right = without_reordering(join.right())?;
    let equi_join = EquiJoin {
        plan,
        left: &left,
        right: &right,
        on: join.on(),
        filter: join.filter().as_ref(),
        join_type: join.join_type(),
        projection: join.projection.as_deref(),
        null_equality: join.null_equality(),
        null_aware: join.null_aware(),
    };
    if unbounded == [true, true] {
        // Two inputs that never end: neither can be built into a table first.
        if join.null_aware() {
            return Ok(None);
        }
        return symmetric_hash_join::lower_symmetric(&equi_join, &mut lower_input);
    }
    // A hash join builds its table on an input that ends, on the smaller one
    // if both do.
    let build_right = unbounded[0] || (!unbounded[1] && is_smaller(&right, &left));
    let (build, probe) = match build_right {
        true => (&right, &left),
        false => (&left, &right),
    };
    let build = lower_input(build, false)?;
    let probe = lower_input(probe, true)?;
    if let Some(hash_join) = equi_join.hash_join(build_right, &build, &probe)? {
        return Ok(Some(Arc::new(hash_join)));
    }
    // A join that the hash join does not implement: a merge join, of the same
    // inputs, unless an input never ends.
    if unbounded.contains(&true) {
        return Ok(None);
    }
    let Some(spec) = MergeJoinSpec::of(join)? else {
        return Ok(None);
    };
    let (left, right) = match build_right {
        true => (probe, build),
        false => (build, probe),
    };
    let join = merge_join_of_inputs(join, spec, left, right)?;
    Ok(join.map(|join| Arc::new(join) as Arc<dyn PhysicalOperator>))
}

/// Whether `plan` promises the order of its output to the nodes above it: it
/// has a single output partition with an output ordering. The nodes above it
/// (e.g. a limit, or the result of the query, see
/// [`PipelineRootExec`](crate::PipelineRootExec)) may rely on that order, as
/// DataFusion removes the sorts that it satisfies (e.g. of the null-aware
/// `NOT IN` join, whose inputs DataFusion merges into one partition). (The
/// order of a join of several partitions holds within every partition only,
/// which no native operator relies on: the merges of these partitions are
/// lowered by [`ordered_join_below`].)
pub fn promises_order(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.output_partitioning().partition_count() == 1 && plan.output_ordering().is_some()
}

/// The `SortMergeJoinExec` whose order the `SortPreservingMergeExec` `plan`
/// relies on, if it merges the sorted partitions of such a join: its input is
/// ordered by its keys, and is a `SortMergeJoinExec` with an output ordering,
/// possibly below `ProjectionExec`s, `FilterExec`s, `CoalesceBatchesExec`s
/// and `RepartitionExec`s that preserve the order, which keep the order of
/// the rows of every partition. (DataFusion plans such merges for an `ORDER
/// BY` of the join keys, e.g. `ORDER BY a.k NULLS FIRST` of `a JOIN b ON a.k =
/// b.k`.)
///
/// A [`PhysicalMergeJoin`] emits all rows of the join in its order, on a
/// single thread, which the native operators of the nodes above it keep: the
/// planner lowers the join below such a merge to a merge join (see
/// [`try_lower`]), and the merge to its input (with a limit for its fetch),
/// if its input is lowered to the rows of a merge join (see
/// [`emits_merge_join_order`]).
pub fn ordered_join_below(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<&Arc<dyn ExecutionPlan>>> {
    let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() else {
        return Ok(None);
    };
    let input = merge.input();
    if input.output_partitioning().partition_count() == 0
        || !input
            .equivalence_properties()
            .ordering_satisfy(merge.expr().clone())?
    {
        return Ok(None);
    }
    let mut node = input;
    loop {
        if node.is::<SortMergeJoinExec>() {
            return Ok(node.output_ordering().is_some().then_some(node));
        }
        node = if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            projection.input()
        } else if let Some(filter) = node.downcast_ref::<FilterExec>() {
            filter.input()
        } else if is_coalesce_batches(node) {
            node.children()[0]
        } else if let Some(repartition) = node.downcast_ref::<RepartitionExec>()
            && repartition.preserve_order()
        {
            repartition.input()
        } else {
            return Ok(None);
        };
    }
}

/// Whether `operator` emits the rows of a [`PhysicalMergeJoin`] in the order
/// of the join: it is the merge join, or a projection or a filter (with its
/// cache) above it, which keep the order of the single thread that executes
/// the pipeline of the merge join.
pub fn emits_merge_join_order(operator: &Arc<dyn PhysicalOperator>) -> bool {
    let mut operator = operator;
    loop {
        if operator.is::<PhysicalMergeJoin>() {
            return true;
        }
        let order_preserving = operator.is::<PhysicalProjection>()
            || operator.is::<PhysicalFilter>()
            || operator.is::<CachingOperator>();
        match operator.children() {
            [input] if order_preserving => operator = input,
            _ => return false,
        }
    }
}

/// `plan`, an input of a join, without the nodes that only sort its rows or
/// distribute them over partitions: `SortExec`s and `SortPreservingMergeExec`s
/// without a fetch, `CoalescePartitionsExec`s without a fetch,
/// `RepartitionExec`s and `CoalesceBatchesExec`s, directly above it or below
/// the `ProjectionExec`s and `FilterExec`s (without a fetch) above it, which
/// are rebuilt over their inputs without them (e.g. of the sort that
/// DataFusion pushes below the projection of a sort-merge join of an
/// aggregation). The native joins read all rows of both inputs, on all
/// threads (which partitions the rows are distributed over, or what a
/// repartitioning evaluates to distribute them, does not change the result of
/// a join, nor of the projections and filters of their rows), and a merge
/// join sorts its inputs itself.
pub(crate) fn without_reordering(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let plan = without_reordering_nodes(plan);
    let streams_rows = plan.is::<ProjectionExec>()
        || (plan.is::<FilterExec>() && plan.fetch().is_none());
    let children = plan.children();
    let [input] = children.as_slice() else {
        return Ok(Arc::clone(plan));
    };
    if !streams_rows {
        return Ok(Arc::clone(plan));
    }
    let stripped = without_reordering(input)?;
    if Arc::ptr_eq(&stripped, input) {
        return Ok(Arc::clone(plan));
    }
    Arc::clone(plan).replace_children(
        vec![stripped],
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
    )
}

/// `plan` without the nodes directly above it that only sort its rows or
/// distribute them over partitions, see [`without_reordering`].
fn without_reordering_nodes(plan: &Arc<dyn ExecutionPlan>) -> &Arc<dyn ExecutionPlan> {
    let mut plan = plan;
    loop {
        plan = if let Some(sort) = plan.downcast_ref::<SortExec>()
            && sort.fetch().is_none()
        {
            sort.input()
        } else if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>()
            && merge.fetch().is_none()
        {
            merge.input()
        } else if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
        {
            coalesce.input()
        } else if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
            repartition.input()
        } else if is_coalesce_batches(plan) && plan.fetch().is_none() {
            plan.children()[0]
        } else {
            return plan;
        };
    }
}

/// Whether `plan` is a `CoalesceBatchesExec` (which DataFusion does not plan
/// any more), which only changes the sizes of the batches of its input.
#[expect(deprecated)]
fn is_coalesce_batches(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec>()
}

/// Whether `plan` is smaller than `other` by their statistics: by their sizes
/// in bytes, or by their numbers of rows if a size is unknown, like
/// DataFusion's `JoinSelection` compares the inputs of a hash join to put the
/// smaller one on its build side. `false` if neither statistic is known.
pub(crate) fn is_smaller(
    plan: &Arc<dyn ExecutionPlan>,
    other: &Arc<dyn ExecutionPlan>,
) -> bool {
    let context = StatisticsContext::new();
    let args = StatisticsArgs::new();
    let (Ok(plan), Ok(other)) = (
        context.compute_arc(plan, &args),
        context.compute_arc(other, &args),
    ) else {
        return false;
    };
    match (
        plan.total_byte_size.get_value(),
        other.total_byte_size.get_value(),
    ) {
        (Some(size), Some(other)) => size < other,
        _ => match (plan.num_rows.get_value(), other.num_rows.get_value()) {
            (Some(rows), Some(other)) => rows < other,
            _ => false,
        },
    }
}

/// An equi-join of DataFusion (a `SortMergeJoinExec` or a
/// `SymmetricHashJoinExec`) to lower to a native join.
pub(crate) struct EquiJoin<'a> {
    /// The join.
    pub(crate) plan: &'a Arc<dyn ExecutionPlan>,
    /// Its inputs, without the nodes that only sort or distribute their rows
    /// (see [`without_reordering`]).
    pub(crate) left: &'a Arc<dyn ExecutionPlan>,
    pub(crate) right: &'a Arc<dyn ExecutionPlan>,
    /// The join keys: `(left key, right key)`.
    pub(crate) on: &'a [(PhysicalExprRef, PhysicalExprRef)],
    pub(crate) filter: Option<&'a JoinFilter>,
    pub(crate) join_type: JoinType,
    /// The output columns, as indices of the columns of the join.
    pub(crate) projection: Option<&'a [usize]>,
    pub(crate) null_equality: NullEquality,
    pub(crate) null_aware: bool,
}

impl EquiJoin<'_> {
    /// The [`PhysicalHashJoin`] of the join that builds its table on `build`,
    /// the lowered right input if `build_right` (and the left one otherwise),
    /// probing it with `probe`, the other one. With the build side on the
    /// right, the join is swapped: its keys, its filter and its join type, and
    /// a projection restores the order of the columns of the join.
    ///
    /// `Ok(None)` for the joins that the hash join does not implement (e.g.
    /// of keys or columns that its hash table cannot store), or if the swapped
    /// join would not have exactly the schema of the plan.
    pub(crate) fn hash_join(
        &self,
        build_right: bool,
        build: &Arc<dyn PhysicalOperator>,
        probe: &Arc<dyn PhysicalOperator>,
    ) -> Result<Option<PhysicalHashJoin>> {
        let left_columns = self.left.schema().fields().len();
        let right_columns = self.right.schema().fields().len();
        let (on, filter, join_type, projection) = if build_right {
            let on = self
                .on
                .iter()
                .map(|(left, right)| (Arc::clone(right), Arc::clone(left)))
                .collect::<Vec<_>>();
            let join_type = self.join_type.swap();
            let projection = match self.projection {
                Some(projection) => swap_join_projection(
                    left_columns,
                    right_columns,
                    Some(projection),
                    &self.join_type,
                ),
                // The swapped join emits the right columns first.
                None if emits_both_inputs(self.join_type) => Some(
                    (right_columns..right_columns + left_columns)
                        .chain(0..right_columns)
                        .collect(),
                ),
                None => None,
            };
            (on, self.filter.map(JoinFilter::swap), join_type, projection)
        } else {
            (
                self.on.to_vec(),
                self.filter.cloned(),
                self.join_type,
                self.projection.map(<[usize]>::to_vec),
            )
        };
        let output = JoinOutput::try_new(
            &build.schema(),
            &probe.schema(),
            join_type,
            projection.clone(),
        )?;
        if output.schema != self.plan.schema() {
            if build_right {
                // E.g. the metadata of the inputs, merged in another order.
                return Ok(None);
            }
            return internal_err!(
                "The hash join of {} would have schema {:?}, expected {:?}",
                self.plan.name(),
                output.schema,
                self.plan.schema()
            );
        }
        match PhysicalHashJoin::try_new(
            Arc::clone(build),
            Arc::clone(probe),
            on,
            filter,
            join_type,
            projection,
            self.null_equality,
            self.null_aware,
        ) {
            Ok(join) => Ok(Some(join)),
            // Unsupported null-aware joins, keys of different types, or columns
            // that the hash table cannot store.
            Err(DataFusionError::NotImplemented(_) | DataFusionError::Plan(_)) => {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

/// Whether the rows of a join of type `join_type` have the columns of both
/// inputs.
fn emits_both_inputs(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
    )
}

/// Lower `join` to a [`PhysicalMergeJoin`] with the specification `spec`,
/// sorting both of its inputs (lowered with `lower_input`, without the sorts
/// and repartitionings below the join) with [`PhysicalSort`]s.
fn lower_merge_join<F>(
    join: &SortMergeJoinExec,
    spec: MergeJoinSpec,
    lower_input: &mut F,
) -> Result<Option<PhysicalMergeJoin>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, bool) -> Result<Arc<dyn PhysicalOperator>>,
{
    // The sorts read all of their input.
    let left = lower_input(&without_reordering(join.left())?, false)?;
    let right = lower_input(&without_reordering(join.right())?, false)?;
    merge_join_of_inputs(join, spec, left, right)
}

/// The [`PhysicalMergeJoin`] of `join` with the specification `spec`, over
/// its lowered inputs `left` and `right` (without the sorts and
/// repartitionings below the join), which it sorts with [`PhysicalSort`]s by
/// the order of the inputs of the `SortMergeJoinExec` (see [`sort_order`]).
/// `Ok(None)` if they cannot be sorted natively.
fn merge_join_of_inputs(
    join: &SortMergeJoinExec,
    spec: MergeJoinSpec,
    left: Arc<dyn PhysicalOperator>,
    right: Arc<dyn PhysicalOperator>,
) -> Result<Option<PhysicalMergeJoin>> {
    let mut children = Vec::with_capacity(2);
    for (input, lowered) in [(join.left(), left), (join.right(), right)] {
        if lowered.schema() != input.schema() {
            return internal_err!(
                "The lowered input {} of a SortMergeJoinExec has schema {:?}, expected {:?}",
                lowered.name(),
                lowered.schema(),
                input.schema()
            );
        }
        let keys = join
            .on()
            .iter()
            .map(|(left, right)| match children.is_empty() {
                true => left,
                false => right,
            });
        let Some(ordering) = sort_order(input, keys, join.sort_options())? else {
            return Ok(None);
        };
        let sort: Arc<dyn PhysicalOperator> =
            Arc::new(PhysicalSort::try_new(lowered, ordering, None)?);
        children.push(sort);
    }
    let right = children.pop();
    let left = children.pop();
    let (Some(left), Some(right)) = (left, right) else {
        return internal_err!("A merge join without inputs");
    };
    Ok(Some(PhysicalMergeJoin {
        children: vec![left, right],
        spec,
    }))
}

/// The order in which a [`PhysicalMergeJoin`] sorts `input`, an input of a
/// `SortMergeJoinExec` whose join keys on that side are `keys` (sorted with
/// `options`): the output ordering of `input` (which satisfies the order of
/// the keys, and which the output of the `SortMergeJoinExec` keeps), or the
/// order of the keys if it has none (e.g. constant keys).
///
/// If a key has floating point values, the keys come first, with `-0.0`
/// sorted as `+0.0` (see [`NormalizedFloats`]): the merge compares the keys so,
/// and the order of the input would separate them if a key follows a key with
/// both zeros. `Ok(None)` if the order cannot be sorted natively (e.g. keys of
/// unions, which the row format orders by their type first).
fn sort_order<'a>(
    input: &Arc<dyn ExecutionPlan>,
    keys: impl Iterator<Item = &'a PhysicalExprRef>,
    options: &[SortOptions],
) -> Result<Option<LexOrdering>> {
    let schema = input.schema();
    let keys = keys.collect::<Vec<_>>();
    let mut normalized = false;
    let mut key_order = Vec::with_capacity(keys.len());
    for (&key, &options) in keys.iter().zip(options) {
        let data_type = key.data_type(&schema)?;
        let key: PhysicalExprRef = if has_float_leaf(&data_type) {
            normalized = true;
            Arc::new(NormalizedFloats::new(Arc::clone(key)))
        } else {
            Arc::clone(key)
        };
        key_order.push(PhysicalSortExpr::new(key, options));
    }
    let input_order = input.output_ordering().cloned();
    let ordering = match (input_order, normalized) {
        (Some(order), false) => Some(order),
        (Some(order), true) => LexOrdering::new(key_order.into_iter().chain(order)),
        (None, _) => LexOrdering::new(key_order),
    };
    let Some(ordering) = ordering else {
        return internal_err!("A SortMergeJoinExec without join keys");
    };
    let sortable = SortKeys::supports(&ordering, &schema)?
        && ordering
            .iter()
            .map(|sort_expr| sort_expr.expr.data_type(&schema))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .all(|data_type| !has_union(data_type));
    Ok(sortable.then_some(ordering))
}

/// Whether values of `data_type` are unions, or have unions inside them.
fn has_union(data_type: &DataType) -> bool {
    match data_type {
        DataType::Union(_, _) => true,
        DataType::Dictionary(_, values) => has_union(values),
        DataType::RunEndEncoded(_, values) => has_union(values.data_type()),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_union(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|field| has_union(field.data_type()))
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// The output of joins, shared with the symmetric hash join
// ---------------------------------------------------------------------------

/// The output of a join: its schema (without and with its projection), and
/// the input column of every column (or the mark of a mark join).
#[derive(Debug, Clone)]
pub(crate) struct JoinOutput {
    pub(crate) join_type: JoinType,
    /// The schema of the join without its projection.
    pub(crate) join_schema: SchemaRef,
    /// The output schema.
    pub(crate) schema: SchemaRef,
    /// The output columns, as indices of the columns of `join_schema`.
    pub(crate) projection: Option<Vec<usize>>,
    /// The input column of every output column.
    pub(crate) column_indices: Vec<ColumnIndex>,
}

impl JoinOutput {
    /// The output of a join of type `join_type` of inputs with the schemas
    /// `left` and `right`, with the output columns `projection` (all if
    /// `None`): the schema of DataFusion's joins.
    pub(crate) fn try_new(
        left: &Schema,
        right: &Schema,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let (join_schema, columns) = build_join_schema(left, right, &join_type);
        let join_schema = Arc::new(join_schema);
        let (schema, column_indices) = match &projection {
            Some(projection) => {
                let column_indices = projection
                    .iter()
                    .map(|&i| {
                        columns.get(i).cloned().ok_or_else(|| {
                            internal_datafusion_err!(
                                "Projection index {i} out of bounds for a join with {} columns",
                                columns.len()
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                (Arc::new(join_schema.project(projection)?), column_indices)
            }
            None => (Arc::clone(&join_schema), columns),
        };
        Ok(Self {
            join_type,
            join_schema,
            schema,
            projection,
            column_indices,
        })
    }

    /// Whether the join has a mark column.
    pub(crate) fn is_mark_join(&self) -> bool {
        matches!(self.join_type, JoinType::LeftMark | JoinType::RightMark)
    }

    /// The output rows of `num_rows` pairs of `left` and `right` rows, with
    /// the marks `marks` of a mark join.
    pub(crate) fn batch(
        &self,
        left: &JoinRows<'_>,
        right: &JoinRows<'_>,
        marks: Option<&ArrayRef>,
        num_rows: usize,
    ) -> Result<RecordBatch> {
        let columns = self
            .column_indices
            .iter()
            .map(|column| match column.side {
                JoinSide::Left => left.column(column.index),
                JoinSide::Right => right.column(column.index),
                JoinSide::None => marks.map(Arc::clone).ok_or_else(|| {
                    internal_datafusion_err!("The marks of a mark join are missing")
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        record_batch(Arc::clone(&self.schema), columns, num_rows)
    }

    /// A description of the join for `EXPLAIN`: its join type, keys, filter
    /// and projection, as DataFusion displays its joins.
    pub(crate) fn describe(
        &self,
        name: &str,
        on: &[(PhysicalExprRef, PhysicalExprRef)],
        filter: Option<&JoinFilter>,
        null_equality: NullEquality,
    ) -> String {
        let on = on
            .iter()
            .map(|(left, right)| format!("({left}, {right})"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut description = format!("{name}: join_type={}, on=[{on}]", self.join_type);
        if let Some(filter) = filter {
            let _ = write!(description, ", filter={}", filter.expression());
        }
        if let Some(projection) = &self.projection {
            let columns = projection
                .iter()
                .map(|&i| format!("{}@{i}", self.join_schema.field(i).name()))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(description, ", projection=[{columns}]");
        }
        if null_equality == NullEquality::NullEqualsNull {
            description.push_str(", NullsEqual: true");
        }
        description
    }
}

/// Rows of one input of a join, gathered for its output or its filter.
pub(crate) enum JoinRows<'a> {
    /// The rows `rows` of a batch (a row of NULLs for a NULL index).
    Batch(&'a RecordBatch, &'a UInt32Array),
    /// The rows at `positions` of a [`StoredRows`] (a row of NULLs for
    /// `None`).
    Stored(&'a StoredRows, &'a [Option<usize>]),
    /// Rows of NULLs of a schema.
    Nulls(&'a SchemaRef, usize),
}

impl JoinRows<'_> {
    /// The values of column `index` of the rows.
    fn column(&self, index: usize) -> Result<ArrayRef> {
        match self {
            Self::Batch(batch, rows) => {
                let column = batch.columns().get(index).ok_or_else(|| {
                    internal_datafusion_err!("Missing column {index} of a join input")
                })?;
                take_rows(column, rows)
            }
            Self::Stored(stored, positions) => stored.gather(index, positions),
            Self::Nulls(schema, num_rows) => {
                let field = schema.fields().get(index).ok_or_else(|| {
                    internal_datafusion_err!("Missing column {index} of a join input")
                })?;
                Ok(new_null_array(field.data_type(), *num_rows))
            }
        }
    }
}

/// The rows `rows` of `array`: zero copy if they are a contiguous range
/// without NULL indices.
fn take_rows(array: &ArrayRef, rows: &UInt32Array) -> Result<ArrayRef> {
    let values = rows.values();
    if let (Some(&first), Some(&last)) = (values.first(), values.last())
        && rows.null_count() == 0
        && last.checked_sub(first).map(|span| span as usize + 1) == Some(values.len())
        && values.windows(2).all(|pair| pair[1] == pair[0] + 1)
    {
        let (first, len) = (first as usize, values.len());
        if first == 0 && len == array.len() {
            return Ok(Arc::clone(array));
        }
        return Ok(array.slice(first, len));
    }
    Ok(take(array.as_ref(), rows, None)?)
}

/// The rows that a join keeps of one of its inputs, in batches (e.g. as they
/// arrived), addressed by their positions: the rows of the batches in order.
#[derive(Debug)]
pub(crate) struct StoredRows {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    /// The position of the first row of every batch.
    starts: Vec<usize>,
    num_rows: usize,
}

impl StoredRows {
    /// No rows of schema `schema`.
    pub(crate) fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            batches: vec![],
            starts: vec![],
            num_rows: 0,
        }
    }

    /// The number of rows.
    pub(crate) fn len(&self) -> usize {
        self.num_rows
    }

    /// Whether there are no rows.
    pub(crate) fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Append the rows of `batch`, at the positions after the rows so far.
    pub(crate) fn push(&mut self, batch: RecordBatch) {
        if batch.num_rows() == 0 {
            return;
        }
        self.starts.push(self.num_rows);
        self.num_rows += batch.num_rows();
        self.batches.push(batch);
    }

    /// The batch and the row in it of every position of `positions` (the
    /// next batch after the last one for `None`, the batch of NULLs of
    /// [`Self::gather`]).
    fn locate(&self, positions: &[Option<usize>]) -> Result<Vec<(usize, usize)>> {
        let mut located = Vec::with_capacity(positions.len());
        // The batch of the previous position: positions are often close.
        let mut batch = 0;
        for position in positions {
            let Some(position) = *position else {
                located.push((self.batches.len(), 0));
                continue;
            };
            if position >= self.num_rows {
                return internal_err!(
                    "Row {position} out of bounds for {} stored rows",
                    self.num_rows
                );
            }
            let start = self.starts[batch];
            let end = start + self.batches[batch].num_rows();
            if position < start || position >= end {
                batch = self.starts.partition_point(|&start| start <= position) - 1;
            }
            located.push((batch, position - self.starts[batch]));
        }
        Ok(located)
    }

    /// The values of column `column` of the rows at `positions` (NULL for
    /// `None`).
    pub(crate) fn gather(
        &self,
        column: usize,
        positions: &[Option<usize>],
    ) -> Result<ArrayRef> {
        let Some(field) = self.schema.fields().get(column) else {
            return internal_err!("Missing column {column} of the stored rows");
        };
        if positions.is_empty() {
            return Ok(new_empty_array(field.data_type()));
        }
        let located = self.locate(positions)?;
        // The rows of a single batch are taken from it.
        let first = located[0].0;
        if first < self.batches.len() && located.iter().all(|&(batch, _)| batch == first)
        {
            let rows =
                UInt32Array::from_iter_values(located.iter().map(|&(_, row)| row as u32));
            return take_rows(self.batches[first].column(column), &rows);
        }
        let nulls = new_null_array(field.data_type(), 1);
        let arrays = self
            .batches
            .iter()
            .map(|batch| batch.column(column).as_ref())
            .chain(std::iter::once(nulls.as_ref()))
            .collect::<Vec<_>>();
        Ok(interleave(&arrays, &located)?)
    }
}

/// Whether the residual filter `filter` holds for the `num_rows` pairs of
/// `left` and `right` rows: `true`, or `false` (also for NULL), for every
/// pair.
pub(crate) fn filter_mask(
    filter: &JoinFilter,
    left: &JoinRows<'_>,
    right: &JoinRows<'_>,
    num_rows: usize,
) -> Result<BooleanArray> {
    let columns = filter
        .column_indices()
        .iter()
        .map(|column| match column.side {
            JoinSide::Left => left.column(column.index),
            JoinSide::Right => right.column(column.index),
            JoinSide::None => {
                internal_err!("The filter of a join reads a column of neither input")
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let batch = record_batch(Arc::clone(filter.schema()), columns, num_rows)?;
    let mask = filter.expression().evaluate(&batch)?.into_array(num_rows)?;
    let mask = as_boolean_array(&mask)?;
    Ok(match mask.null_count() {
        0 => mask.clone(),
        _ => prep_null_mask_filter(mask),
    })
}

/// The values of the join keys `keys` for the rows of `batch`, with `-0.0`
/// replaced by `+0.0` (also nested), as DataFusion compares them as join
/// keys.
pub(crate) fn evaluate_keys(
    keys: impl IntoIterator<Item = PhysicalExprRef>,
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    let keys = keys.into_iter().collect::<Vec<_>>();
    Ok(evaluate_expressions_to_arrays(&keys, batch)?
        .iter()
        .map(normalize_floats)
        .collect())
}

/// The rows whose keys `keys` can match a key of the other input: `None` if
/// all of them can, otherwise the rows without a NULL key (with
/// [`NullEquality::NullEqualsNothing`]).
pub(crate) fn matchable_rows(
    keys: &[ArrayRef],
    null_equality: NullEquality,
) -> Option<NullBuffer> {
    match null_equality {
        NullEquality::NullEqualsNothing => {
            let nulls = keys
                .iter()
                .map(|key| key.logical_nulls())
                .collect::<Vec<_>>();
            NullBuffer::union_many(nulls.iter().map(Option::as_ref))
                .filter(|nulls| nulls.null_count() > 0)
        }
        NullEquality::NullEqualsNull => None,
    }
}

/// A [`RecordBatch`] of `num_rows` rows (also without columns).
pub(crate) fn record_batch(
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    num_rows: usize,
) -> Result<RecordBatch> {
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    Ok(RecordBatch::try_new_with_options(
        schema, columns, &options,
    )?)
}

/// `-0.0` replaced by `+0.0` in the values of `expr` (also in nested values),
/// as DataFusion compares floats as join keys (see `normalize_floats`): a
/// [`PhysicalMergeJoin`] sorts its inputs by the normalized keys, so that the
/// keys that are equal for the join are adjacent in its sorted inputs.
#[derive(Debug, Eq)]
pub(crate) struct NormalizedFloats {
    expr: PhysicalExprRef,
}

impl NormalizedFloats {
    pub(crate) fn new(expr: PhysicalExprRef) -> Self {
        Self { expr }
    }
}

impl PartialEq for NormalizedFloats {
    fn eq(&self, other: &Self) -> bool {
        self.expr.eq(&other.expr)
    }
}

impl Hash for NormalizedFloats {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.hash(state);
    }
}

impl fmt::Display for NormalizedFloats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "normalize_floats({})", self.expr)
    }
}

impl PhysicalExpr for NormalizedFloats {
    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        self.expr.data_type(input_schema)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.expr.nullable(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let values = self.expr.evaluate(batch)?.into_array(batch.num_rows())?;
        Ok(ColumnarValue::Array(normalize_floats(&values)))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.expr]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        match (children.pop(), children.is_empty()) {
            (Some(expr), true) => Ok(Arc::new(Self { expr })),
            _ => internal_err!("NormalizedFloats has exactly one child"),
        }
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "normalize_floats(")?;
        self.expr.fmt_sql(f)?;
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// PhysicalMergeJoin
// ---------------------------------------------------------------------------

/// A sort-merge join of two inputs sorted by the join keys, with the
/// semantics of DataFusion's `SortMergeJoinExec`: its join types, null
/// equality, residual filter, `NOT IN` null awareness and embedded
/// projection, and exactly its output schema and its order.
///
/// See the [module documentation](self) for its pipelines.
#[derive(Debug)]
pub struct PhysicalMergeJoin {
    /// The left and the right input, sorted by the join keys.
    children: Vec<Arc<dyn PhysicalOperator>>,
    spec: MergeJoinSpec,
}

/// Everything about a merge join that does not depend on its inputs, only on
/// their schemas.
#[derive(Debug)]
struct MergeJoinSpec {
    output: JoinOutput,
    /// The join keys: `(left key, right key)`.
    on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    filter: Option<JoinFilter>,
    null_equality: NullEquality,
    null_aware: bool,
    /// The input that the join streams: its rows are joined in its pipeline,
    /// in their order. The other input is buffered.
    streamed: JoinSide,
    /// The fields of the keys in the row format: their types, and their
    /// orders in the sorted inputs.
    key_fields: Vec<SortField>,
}

impl MergeJoinSpec {
    #[expect(clippy::too_many_arguments)]
    fn try_new(
        left: &Schema,
        right: &Schema,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        sort_options: &[SortOptions],
        null_equality: NullEquality,
        null_aware: bool,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        if on.is_empty() {
            return plan_err!("PhysicalMergeJoin requires at least one join key");
        }
        if sort_options.len() != on.len() {
            return plan_err!(
                "PhysicalMergeJoin requires one sort order per join key, got {} for {} keys",
                sort_options.len(),
                on.len()
            );
        }
        if null_aware
            && (join_type != JoinType::LeftAnti
                || on.len() != 1
                || filter.is_some()
                || null_equality != NullEquality::NullEqualsNothing)
        {
            return not_impl_err!(
                "A null-aware PhysicalMergeJoin is a LeftAnti join of a single key, without filter and with NullEqualsNothing"
            );
        }
        let mut key_fields = Vec::with_capacity(on.len());
        for (i, ((left_key, right_key), options)) in
            on.iter().zip(sort_options).enumerate()
        {
            let left_type = left_key.data_type(left)?;
            let right_type = right_key.data_type(right)?;
            if left_type != right_type {
                return plan_err!(
                    "PhysicalMergeJoin requires join keys of the same type, key {i} has types {left_type} and {right_type}"
                );
            }
            key_fields.push(SortField::new_with_options(left_type, *options));
        }
        if !RowConverter::supports_fields(&key_fields) {
            return not_impl_err!("The row format does not support the keys of the join");
        }
        if let Some(filter) = &filter {
            let columns = filter.column_indices();
            if columns.len() != filter.schema().fields().len() {
                return plan_err!(
                    "The filter of a PhysicalMergeJoin reads {} columns, but its schema has {} fields",
                    columns.len(),
                    filter.schema().fields().len()
                );
            }
            if let Some(column) = columns.iter().find(|column| match column.side {
                JoinSide::Left => column.index >= left.fields().len(),
                JoinSide::Right => column.index >= right.fields().len(),
                JoinSide::None => true,
            }) {
                return plan_err!(
                    "The filter of a PhysicalMergeJoin reads column {} of side {:?}, which its input does not have",
                    column.index,
                    column.side
                );
            }
        }
        Ok(Self {
            output: JoinOutput::try_new(left, right, join_type, projection)?,
            on,
            filter,
            null_equality,
            null_aware,
            streamed: SortMergeJoinExec::probe_side(&join_type),
            key_fields,
        })
    }

    /// The specification of the merge join of `join`, `None` if the merge
    /// join does not implement it (see [`try_lower`]).
    fn of(join: &SortMergeJoinExec) -> Result<Option<Self>> {
        let spec = match Self::try_new(
            &join.left().schema(),
            &join.right().schema(),
            join.on().to_vec(),
            join.filter().clone(),
            join.join_type(),
            join.sort_options(),
            join.null_equality(),
            join.null_aware(),
            join.projection.clone(),
        ) {
            Ok(spec) => spec,
            Err(DataFusionError::NotImplemented(_) | DataFusionError::Plan(_)) => {
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        if spec.output.schema != join.schema() {
            return internal_err!(
                "PhysicalMergeJoin would have schema {:?}, but the SortMergeJoinExec has schema {:?}",
                spec.output.schema,
                join.schema()
            );
        }
        Ok(Some(spec))
    }

    fn join_type(&self) -> JoinType {
        self.output.join_type
    }

    /// The keys of the streamed input.
    fn streamed_keys(&self) -> impl Iterator<Item = PhysicalExprRef> + '_ {
        let left = self.streamed == JoinSide::Left;
        self.on
            .iter()
            .map(move |(l, r)| Arc::clone(if left { l } else { r }))
    }

    /// The keys of the buffered input.
    fn buffered_keys(&self) -> impl Iterator<Item = PhysicalExprRef> + '_ {
        let left = self.streamed == JoinSide::Left;
        self.on
            .iter()
            .map(move |(l, r)| Arc::clone(if left { r } else { l }))
    }

    /// The left and the right rows of pairs of `streamed` and `buffered`
    /// rows.
    fn sides<'a>(
        &self,
        streamed: JoinRows<'a>,
        buffered: JoinRows<'a>,
    ) -> (JoinRows<'a>, JoinRows<'a>) {
        match self.streamed {
            JoinSide::Left => (streamed, buffered),
            _ => (buffered, streamed),
        }
    }

    /// Whether the join emits pairs of a streamed and a buffered row.
    fn emits_pairs(&self) -> bool {
        emits_both_inputs(self.join_type())
    }

    /// Whether the join emits the buffered rows without a match (`Full`).
    fn emits_unmatched_buffered_rows(&self) -> bool {
        self.join_type() == JoinType::Full
    }

    /// Whether the join can produce output with an empty buffered input.
    fn produces_output_without_buffered_rows(&self) -> bool {
        !matches!(
            self.join_type(),
            JoinType::Inner | JoinType::LeftSemi | JoinType::RightSemi
        )
    }

    fn key_converter(&self) -> Result<RowConverter> {
        Ok(RowConverter::new(self.key_fields.clone())?)
    }
}

impl PhysicalMergeJoin {
    /// A merge join of `left` and `right` on the keys `on` (`(left key, right
    /// key)`), with the semantics of a `SortMergeJoinExec` with the same
    /// arguments, and its output schema and order. Both inputs must emit their
    /// rows sorted by their keys, with `sort_options`, on a single thread
    /// (e.g. [`PhysicalSort`]s): the join fails if they do not.
    #[expect(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn PhysicalOperator>,
        right: Arc<dyn PhysicalOperator>,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        sort_options: &[SortOptions],
        null_equality: NullEquality,
        null_aware: bool,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let spec = MergeJoinSpec::try_new(
            &left.schema(),
            &right.schema(),
            on,
            filter,
            join_type,
            sort_options,
            null_equality,
            null_aware,
            projection,
        )?;
        Ok(Self {
            children: vec![left, right],
            spec,
        })
    }

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

    /// The input whose rows the join streams, in their order (DataFusion's
    /// `SortMergeJoinExec::probe_side`); the other one is buffered.
    pub fn streamed_side(&self) -> JoinSide {
        self.spec.streamed
    }

    fn streamed_input(&self) -> &Arc<dyn PhysicalOperator> {
        match self.spec.streamed {
            JoinSide::Left => self.left(),
            _ => self.right(),
        }
    }

    fn buffered_input(&self) -> &Arc<dyn PhysicalOperator> {
        match self.spec.streamed {
            JoinSide::Left => self.right(),
            _ => self.left(),
        }
    }

    /// The buffered rows of the execution of `ctx`, once the buffered input
    /// is finalized.
    fn buffered_rows(&self, ctx: &ExecutionContext) -> Result<Arc<BufferedRows>> {
        ctx.global_sink_state(self)
            .and_then(|state| state.downcast_arc::<MergeJoinState>())
            .and_then(|state| state.rows.get().map(Arc::clone))
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "The buffered input of PhysicalMergeJoin was not finalized"
                )
            })
    }

    /// Start joining the streamed rows of `input`.
    fn start_chunk(
        &self,
        state: &StreamState,
        input: &DataChunk,
    ) -> Result<StreamedChunk> {
        let batch = input.compact()?;
        let keys = evaluate_keys(self.spec.streamed_keys(), &batch)?;
        let matchable = matchable_rows(&keys, self.spec.null_equality);
        let keys = state.converter.convert_columns(&keys)?;
        Ok(StreamedChunk {
            batch,
            keys,
            matchable,
            next_row: 0,
            in_progress: None,
        })
    }

    /// The next output of the streamed rows of the current chunk of `state`,
    /// of at most [`DEFAULT_VECTOR_SIZE`] rows: `None` if the rows joined so
    /// far produce no output.
    ///
    /// The candidate pairs of a batch of streamed rows (with the buffered rows
    /// of their keys) are collected first, at most [`DEFAULT_VECTOR_SIZE`] of
    /// them (a streamed row without candidates counts as one), then the filter
    /// is evaluated on all of them at once, and the rows are emitted in the
    /// order of the streamed rows. A streamed row whose candidates do not all
    /// fit is continued by the next call.
    fn next_output(&self, state: &mut StreamState) -> Result<Option<RecordBatch>> {
        let spec = &self.spec;
        let join_type = spec.join_type();
        let StreamState {
            buffered,
            cursor,
            group,
            chunk,
            ..
        } = state;
        let Some(chunk) = chunk.as_mut() else {
            return internal_err!("PhysicalMergeJoin has no streamed rows to join");
        };
        // The pairs are needed to evaluate the filter, or for the output.
        let enumerates_pairs = spec.filter.is_some() || spec.emits_pairs();
        // Whether a streamed row is decided by its first match (semi, anti and
        // mark joins).
        let decided_by_match = !spec.emits_pairs();

        let mut pairs_streamed = Vec::new();
        let mut pairs_buffered = Vec::new();
        let mut parts = Vec::new();
        let mut cost = 0;
        while cost < DEFAULT_VECTOR_SIZE {
            let mut progress = match chunk.in_progress.take() {
                Some(progress) => progress,
                None => {
                    let row = chunk.next_row;
                    if row >= chunk.batch.num_rows() {
                        break;
                    }
                    chunk.next_row += 1;
                    let rows = match chunk.is_matchable(row) {
                        true => {
                            buffered.rows_of_key(cursor, group, chunk.keys.row(row))?
                        }
                        false => 0..0,
                    };
                    RowProgress {
                        row,
                        next: rows.start,
                        end: rows.end,
                        matched: false,
                        first: true,
                    }
                }
            };
            // A semi, anti or mark join is decided once a row matched.
            if decided_by_match && progress.matched {
                progress.next = progress.end;
            }
            let remaining = progress.end - progress.next;
            if !enumerates_pairs || remaining == 0 {
                parts.push(Part {
                    row: progress.row,
                    pairs: pairs_buffered.len()..pairs_buffered.len(),
                    candidates: remaining > 0,
                    first: progress.first,
                    last: true,
                    matched_before: progress.matched,
                });
                cost += 1;
                continue;
            }
            let n = remaining.min(DEFAULT_VECTOR_SIZE - cost);
            let start = pairs_buffered.len();
            pairs_streamed.extend(std::iter::repeat_n(progress.row as u32, n));
            pairs_buffered.extend((progress.next..progress.next + n).map(Some));
            cost += n;
            progress.next += n;
            let last = progress.next == progress.end;
            parts.push(Part {
                row: progress.row,
                pairs: start..start + n,
                candidates: true,
                first: progress.first,
                last,
                matched_before: progress.matched,
            });
            if !last {
                progress.first = false;
                chunk.in_progress = Some(progress);
                break;
            }
        }

        // The filter, on all pairs at once.
        let mask = match &spec.filter {
            Some(filter) if !pairs_buffered.is_empty() => {
                let streamed_rows = UInt32Array::from(pairs_streamed);
                let (left, right) = spec.sides(
                    JoinRows::Batch(&chunk.batch, &streamed_rows),
                    JoinRows::Stored(&buffered.rows, &pairs_buffered),
                );
                Some(filter_mask(filter, &left, &right, pairs_buffered.len())?)
            }
            _ => None,
        };
        let passes = |pair: usize| mask.as_ref().is_none_or(|mask| mask.value(pair));

        // The output, in the order of the streamed rows.
        let mut out_streamed: Vec<Option<u32>> = Vec::new();
        let mut out_buffered: Vec<Option<usize>> = Vec::new();
        let mut marks = Vec::new();
        let mut matched = false;
        // The null-aware `NOT IN` of a non-empty buffered input emits no row
        // with a NULL key (and none at all if a buffered key is NULL).
        let excludes_null_keys = spec.null_aware && !buffered.rows.is_empty();
        let excludes_all = spec.null_aware && buffered.has_null_key;
        for part in &parts {
            // A row continued from the last output carries whether it matched.
            matched = !part.first && part.matched_before;
            if spec.emits_pairs() {
                for pair in part.pairs.clone() {
                    if passes(pair) {
                        let position = pairs_buffered[pair];
                        out_streamed.push(Some(part.row as u32));
                        out_buffered.push(position);
                        matched = true;
                        if let Some(position) = position {
                            buffered.set_matched(position);
                        }
                    }
                }
            } else if enumerates_pairs {
                matched |= part.pairs.clone().any(passes);
            } else {
                matched |= part.candidates;
            }
            if !part.last {
                continue;
            }
            let row = Some(part.row as u32);
            match join_type {
                JoinType::Left | JoinType::Right | JoinType::Full if !matched => {
                    out_streamed.push(row);
                    out_buffered.push(None);
                }
                JoinType::LeftSemi | JoinType::RightSemi if matched => {
                    out_streamed.push(row);
                }
                JoinType::LeftAnti | JoinType::RightAnti
                    if !matched
                        && !excludes_all
                        && (!excludes_null_keys || chunk.is_matchable(part.row)) =>
                {
                    out_streamed.push(row);
                }
                JoinType::LeftMark | JoinType::RightMark => {
                    out_streamed.push(row);
                    marks.push(matched);
                }
                _ => {}
            }
        }
        if let Some(progress) = &mut chunk.in_progress {
            progress.matched = matched;
        }

        let num_rows = out_streamed.len();
        if num_rows == 0 {
            return Ok(None);
        }
        let streamed_rows = UInt32Array::from(out_streamed);
        let streamed = JoinRows::Batch(&chunk.batch, &streamed_rows);
        let buffered_rows = JoinRows::Stored(&buffered.rows, &out_buffered);
        let (left, right) = spec.sides(streamed, buffered_rows);
        let marks = spec
            .output
            .is_mark_join()
            .then(|| Arc::new(BooleanArray::from(marks)) as ArrayRef);
        Ok(Some(spec.output.batch(
            &left,
            &right,
            marks.as_ref(),
            num_rows,
        )?))
    }
}

/// The global state of a merge join, its global sink state: the buffered
/// rows.
struct MergeJoinState {
    /// The buffered rows, until the buffered input is finalized.
    building: Mutex<Option<BufferedBuild>>,
    /// The buffered rows, once the buffered input is finalized.
    rows: OnceLock<Arc<BufferedRows>>,
    /// The memory of the buffered rows.
    reservation: MemoryReservation,
}

impl GlobalSinkState for MergeJoinState {}

/// The buffered rows, as the buffered input produces them.
struct BufferedBuild {
    converter: RowConverter,
    rows: StoredRows,
    keys: Rows,
    has_null_key: bool,
}

/// The sorted buffered rows of a merge join.
struct BufferedRows {
    rows: StoredRows,
    /// The keys of the rows, in the row format, in their order.
    keys: Rows,
    /// Whether a key is NULL (for the null-aware joins).
    has_null_key: bool,
    /// For `Full` joins: the rows that matched a streamed row, one bit per
    /// row.
    matched: Vec<AtomicU64>,
}

impl BufferedRows {
    /// The positions of the buffered rows whose key is `key`, a key not below
    /// the keys before it. `cursor` is the first row whose key is not below
    /// these keys, and `group` the rows of the last key (both are updated).
    fn rows_of_key(
        &self,
        cursor: &mut usize,
        group: &mut Option<KeyGroup>,
        key: Row<'_>,
    ) -> Result<Range<usize>> {
        if let Some(group) = group.as_ref() {
            match key.cmp(&group.key.row()) {
                std::cmp::Ordering::Equal => return Ok(group.rows.clone()),
                std::cmp::Ordering::Less => {
                    return internal_err!(
                        "The streamed input of PhysicalMergeJoin is not sorted by the join keys"
                    );
                }
                // All rows of the last key are below this key.
                std::cmp::Ordering::Greater => *cursor = group.rows.end,
            }
        }
        let start = partition_point_from(&self.keys, *cursor, |row| row < key);
        let end = partition_point_from(&self.keys, start, |row| row <= key);
        *cursor = start;
        *group = Some(KeyGroup {
            key: key.owned(),
            rows: start..end,
        });
        Ok(start..end)
    }

    fn set_matched(&self, position: usize) {
        if let Some(word) = self.matched.get(position / 64) {
            word.fetch_or(1 << (position % 64), Ordering::Relaxed);
        }
    }

    fn is_matched(&self, position: usize) -> bool {
        self.matched.get(position / 64).is_some_and(|word| {
            word.load(Ordering::Relaxed) & (1 << (position % 64)) != 0
        })
    }
}

/// The first position `p >= from` in `keys` (sorted) for which `before` is
/// false, for a predicate that is true for the first rows only: found by
/// galloping from `from`, so that it takes time logarithmic in the distance.
fn partition_point_from(
    keys: &Rows,
    from: usize,
    before: impl Fn(Row<'_>) -> bool,
) -> usize {
    let len = keys.num_rows();
    if from >= len || !before(keys.row(from)) {
        return from;
    }
    // `before(keys.row(low))` holds, `high` is the end or a row after.
    let mut low = from;
    let mut step = 1;
    let mut high = loop {
        let probe = low.saturating_add(step);
        if probe >= len {
            break len;
        }
        if !before(keys.row(probe)) {
            break probe;
        }
        low = probe;
        step = step.saturating_mul(2);
    };
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if before(keys.row(mid)) {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

/// The buffered rows of the key of the last streamed row.
struct KeyGroup {
    key: OwnedRow,
    rows: Range<usize>,
}

/// The state of the thread that streams the streamed input.
struct StreamState {
    buffered: Arc<BufferedRows>,
    converter: RowConverter,
    /// The first buffered row whose key is not below the key of the last
    /// streamed row.
    cursor: usize,
    /// The buffered rows of the key of the last streamed row with a key that
    /// can match.
    group: Option<KeyGroup>,
    /// The streamed rows being joined.
    chunk: Option<StreamedChunk>,
}

impl OperatorState for StreamState {}

/// A chunk of streamed rows being joined.
struct StreamedChunk {
    batch: RecordBatch,
    keys: Rows,
    /// The rows whose keys can match, `None` if all.
    matchable: Option<NullBuffer>,
    /// The next row to join.
    next_row: usize,
    /// The row whose candidates did not all fit into the last output.
    in_progress: Option<RowProgress>,
}

impl StreamedChunk {
    fn is_matchable(&self, row: usize) -> bool {
        self.matchable
            .as_ref()
            .is_none_or(|matchable| matchable.is_valid(row))
    }

    /// Whether all rows of the chunk were joined.
    fn is_done(&self) -> bool {
        self.in_progress.is_none() && self.next_row >= self.batch.num_rows()
    }
}

/// How far a streamed row was joined.
struct RowProgress {
    row: usize,
    /// The next buffered row of the key of the row, and the end of them.
    next: usize,
    end: usize,
    /// Whether the row matched so far.
    matched: bool,
    /// Whether no candidate of the row was joined yet.
    first: bool,
}

/// The candidates of a streamed row in an output batch.
struct Part {
    row: usize,
    /// The candidate pairs of the row (indices of the pairs of the batch).
    pairs: Range<usize>,
    /// Whether the row has candidates (also if they were not enumerated).
    candidates: bool,
    /// Whether these are the first, and the last, candidates of the row.
    first: bool,
    last: bool,
    /// Whether the row matched before these candidates.
    matched_before: bool,
}

/// The global state of the scan of the buffered rows without a match (of a
/// `Full` join).
struct UnmatchedScanState {
    buffered: Arc<BufferedRows>,
    /// The next position to scan.
    next: AtomicUsize,
}

impl GlobalSourceState for UnmatchedScanState {
    fn max_threads(&self) -> usize {
        self.buffered
            .rows
            .len()
            .div_ceil(DEFAULT_VECTOR_SIZE)
            .max(1)
    }
}

impl PhysicalOperator for PhysicalMergeJoin {
    fn name(&self) -> &str {
        "PhysicalMergeJoin"
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
        if spec.null_aware {
            description.push_str(", null_aware");
        }
        description
    }

    // ---------------------------------------------------------------------
    // Sink: the buffered input
    // ---------------------------------------------------------------------

    fn is_sink(&self) -> bool {
        true
    }

    /// The buffered rows arrive in order, from a single thread.
    fn parallel_sink(&self) -> bool {
        false
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let converter = self.spec.key_converter()?;
        let keys = converter.empty_rows(0, 0);
        let reservation =
            MemoryConsumer::new("PhysicalMergeJoin").register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(MergeJoinState {
            building: Mutex::new(Some(BufferedBuild {
                converter,
                rows: StoredRows::new(self.buffered_input().schema()),
                keys,
                has_null_key: false,
            })),
            rows: OnceLock::new(),
            reservation,
        }))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        _lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<MergeJoinState>(gstate.as_any(), self.name())?;
        let batch = input.compact()?;
        if batch.num_rows() == 0 {
            return Ok(SinkResult::NeedMoreInput);
        }
        let keys = evaluate_keys(self.spec.buffered_keys(), &batch)?;
        let mut building = state.building.lock();
        let Some(build) = building.as_mut() else {
            return internal_err!(
                "PhysicalMergeJoin received buffered rows after its build"
            );
        };
        let key_bytes = build.keys.size();
        let first = build.keys.num_rows();
        build.converter.append(&mut build.keys, &keys)?;
        // The rows must arrive sorted by their keys.
        if (first.max(1)..build.keys.num_rows())
            .any(|row| build.keys.row(row) < build.keys.row(row - 1))
        {
            return internal_err!(
                "The buffered input of PhysicalMergeJoin is not sorted by the join keys"
            );
        }
        if self.spec.null_aware {
            build.has_null_key |= keys.iter().any(|key| key.logical_null_count() > 0);
        }
        let grown = memory_size(&batch) + build.keys.size().saturating_sub(key_bytes);
        state.reservation.try_grow(grown)?;
        build.rows.push(batch);
        Ok(SinkResult::NeedMoreInput)
    }

    fn finalize(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<MergeJoinState>(gstate.as_any(), self.name())?;
        let Some(build) = state.building.lock().take() else {
            return internal_err!("PhysicalMergeJoin was finalized twice");
        };
        // The output of a canceled join is not needed any more.
        if ctx.registry().is_sink_finished(self) {
            return Ok(SinkFinalize::NoOutputPossible);
        }
        let matched = match self.spec.emits_unmatched_buffered_rows() {
            true => {
                let words = build.rows.len().div_ceil(64);
                state.reservation.try_grow(words * size_of::<AtomicU64>())?;
                (0..words).map(|_| AtomicU64::new(0)).collect()
            }
            false => vec![],
        };
        let empty = build.rows.is_empty();
        let has_null_key = build.has_null_key;
        let rows = Arc::new(BufferedRows {
            rows: build.rows,
            keys: build.keys,
            has_null_key,
            matched,
        });
        if state.rows.set(rows).is_err() {
            return internal_err!("PhysicalMergeJoin was finalized twice");
        }
        // The streamed input is not read if the join cannot produce output:
        // e.g. an inner join without buffered rows, or a `NOT IN` of a NULL.
        let no_output = (empty && !self.spec.produces_output_without_buffered_rows())
            || (self.spec.null_aware && has_null_key);
        Ok(match no_output {
            true => SinkFinalize::NoOutputPossible,
            false => SinkFinalize::Ready,
        })
    }

    // ---------------------------------------------------------------------
    // Operator: the streamed input
    // ---------------------------------------------------------------------

    /// The streamed rows are merged in order, by a single thread.
    fn parallel_operator(&self) -> bool {
        false
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(StreamState {
            buffered: self.buffered_rows(ctx)?,
            converter: self.spec.key_converter()?,
            cursor: 0,
            group: None,
            chunk: None,
        }))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<StreamState>(lstate.as_any_mut(), self.name())?;
        if state.chunk.is_none() {
            state.chunk = Some(self.start_chunk(state, input)?);
        }
        if let Some(batch) = self.next_output(state)? {
            *output = DataChunk::from_record_batch(batch);
        }
        if state.chunk.as_ref().is_some_and(StreamedChunk::is_done) {
            state.chunk = None;
            return Ok(OperatorResult::NeedMoreInput);
        }
        Ok(OperatorResult::HaveMoreOutput)
    }

    // ---------------------------------------------------------------------
    // Source: the buffered rows without a match (`Full` joins)
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        self.spec.emits_unmatched_buffered_rows()
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(UnmatchedScanState {
            buffered: self.buffered_rows(ctx)?,
            next: AtomicUsize::new(0),
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<UnmatchedScanState>(gstate.as_any(), self.name())?;
        let buffered = &state.buffered;
        let num_rows = buffered.rows.len();
        loop {
            let start = state.next.fetch_add(DEFAULT_VECTOR_SIZE, Ordering::Relaxed);
            if start >= num_rows {
                return Ok(SourceResult::Finished);
            }
            let end = (start + DEFAULT_VECTOR_SIZE).min(num_rows);
            let positions = (start..end)
                .filter(|&position| !buffered.is_matched(position))
                .map(Some)
                .collect::<Vec<_>>();
            if positions.is_empty() {
                continue;
            }
            let streamed_schema = self.streamed_input().schema();
            let (left, right) = self.spec.sides(
                JoinRows::Nulls(&streamed_schema, positions.len()),
                JoinRows::Stored(&buffered.rows, &positions),
            );
            let batch = self
                .spec
                .output
                .batch(&left, &right, None, positions.len())?;
            return Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)));
        }
    }

    // ---------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        let this = meta.current_operator()?;
        // The join merges the streamed rows in the pipeline of the streamed
        // input.
        meta.add_operator(current, Arc::clone(&this))?;
        let Some(last_pipeline) = meta.pipelines().last().copied() else {
            return internal_err!("A meta pipeline without pipelines");
        };
        // The buffered input sinks into the join, in a child meta pipeline.
        {
            let (mut build, mut build_pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
            build.build(&mut build_pipeline, self.buffered_input())?;
        }
        // The streamed input continues the current pipeline. (Its sort runs
        // concurrently with the sort of the buffered input.)
        meta.build(current, self.streamed_input())?;
        // The buffered rows without a match are emitted once the streamed
        // input ended.
        if self.spec.emits_unmatched_buffered_rows() {
            meta.create_child_pipeline(current, this, &last_pipeline)?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use std::time::Duration;

    use arrow::array::{
        Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray,
        StringViewArray, TimestampNanosecondArray,
    };
    use arrow::compute::concat_batches;
    use arrow::datatypes::{Field, TimeUnit};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, col};
    use datafusion_physical_plan::Partitioning;
    use datafusion_physical_plan::collect;

    use crate::bridge::{PipelineRootExec, bridged_plans};
    use crate::executor::Executor;
    use crate::explain::{format_operator_tree, format_pipelines};
    use crate::operators::union::test_support::{TestSource, count};
    use crate::test_utils::{
        EngineOutput, compare_outputs, register_test_tables, run_engines,
    };

    pub(crate) const ALL_JOIN_TYPES: [JoinType; 10] = [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::RightSemi,
        JoinType::LeftAnti,
        JoinType::RightAnti,
        JoinType::LeftMark,
        JoinType::RightMark,
    ];

    pub(crate) async fn with_timeout<T>(future: impl Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(120), future)
            .await
            .expect("the execution did not complete")
    }

    /// A configuration with which DataFusion plans sort-merge joins, on
    /// `partitions` partitions.
    fn sort_merge_config(partitions: usize) -> SessionConfig {
        let mut config = SessionConfig::new().with_target_partitions(partitions);
        config.options_mut().optimizer.prefer_hash_join = false;
        config
    }

    fn task_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default().with_session_config(sort_merge_config(4)))
    }

    // ---------------------------------------------------------------------
    // Test data
    // ---------------------------------------------------------------------

    /// The schema of the `side` (`"l"` or `"r"`) input of the plans of the
    /// tests: keys of many types (with duplicates, NULLs, `-0.0` and `NaN`)
    /// and a unique `id`.
    pub(crate) fn data_schema(side: &str) -> SchemaRef {
        let field = |name: &str, data_type: DataType, nullable: bool| {
            Field::new(format!("{side}{name}"), data_type, nullable)
        };
        Arc::new(Schema::new(vec![
            field("id", DataType::Int64, false),
            field("k", DataType::Int32, true),
            field("f", DataType::Float64, true),
            field("s", DataType::Utf8, true),
            field("v", DataType::Utf8View, true),
            field("d", DataType::Decimal128(12, 2), true),
            field(
                "t",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                true,
            ),
        ]))
    }

    /// The rows `ids` of the `side` input. `skew` changes the distribution
    /// of the keys, so that the two sides do not match one to one.
    pub(crate) fn data_batch(side: &str, ids: Range<i64>, skew: i64) -> RecordBatch {
        let words = ["alpha", "beta", "a string longer than twelve bytes", ""];
        let key = |i: i64| ((i * 7 + skew) % 23) as i32 - 5;
        let k = Int32Array::from_iter(ids.clone().map(|i| (i % 11 != 3).then(|| key(i))));
        let f = Float64Array::from_iter(ids.clone().map(|i| match (i + skew) % 13 {
            0 => None,
            1 => Some(-0.0),
            2 => Some(0.0),
            3 => Some(f64::NAN),
            r => Some(f64::from(r as i32 % 5) / 4.0),
        }));
        let s = StringArray::from_iter(
            ids.clone()
                .map(|i| (i % 9 != 5).then(|| words[((i + skew) % 4) as usize])),
        );
        let v = StringViewArray::from_iter(ids.clone().map(|i| {
            (i % 10 != 7).then(|| format!("a fairly long view string {}", (i + skew) % 6))
        }));
        let d = Decimal128Array::from_iter(
            ids.clone()
                .map(|i| (i % 8 != 6).then(|| i128::from((i * 3 + skew) % 7) * 125)),
        )
        .with_precision_and_scale(12, 2)
        .unwrap();
        let t = TimestampNanosecondArray::from_iter(ids.clone().map(|i| {
            (i % 12 != 2)
                .then(|| 1_700_000_000_000_000_000 + (i + skew) % 5 * 1_000_000_007)
        }))
        .with_timezone("UTC");
        let id = Int64Array::from_iter_values(ids);
        RecordBatch::try_new(
            data_schema(side),
            vec![
                Arc::new(id),
                Arc::new(k),
                Arc::new(f),
                Arc::new(s),
                Arc::new(v),
                Arc::new(d),
                Arc::new(t),
            ],
        )
        .unwrap()
    }

    /// A scan of the `side` input with `rows` rows in skewed partitions (one
    /// of them empty), or of no rows.
    fn data_scan(side: &str, rows: i64, skew: i64) -> Arc<dyn ExecutionPlan> {
        let partitions = if rows == 0 {
            vec![vec![], vec![]]
        } else {
            let a = rows / 2;
            let b = rows * 3 / 4;
            vec![
                vec![data_batch(side, 0..a, skew)],
                vec![],
                vec![
                    data_batch(side, a..b, skew),
                    data_batch(side, b..rows, skew),
                ],
            ]
        };
        MemorySourceConfig::try_new_exec(&partitions, data_schema(side), None).unwrap()
    }

    /// A residual filter `l<left> <op> r<right>`.
    pub(crate) fn join_filter(left: &str, op: Operator, right: &str) -> JoinFilter {
        let left_schema = data_schema("l");
        let right_schema = data_schema("r");
        let (l, r) = (
            left_schema.index_of(&format!("l{left}")).unwrap(),
            right_schema.index_of(&format!("r{right}")).unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            left_schema.field(l).clone(),
            right_schema.field(r).clone(),
        ]));
        let expression = Arc::new(BinaryExpr::new(
            col(left_schema.field(l).name(), &schema).unwrap(),
            op,
            col(right_schema.field(r).name(), &schema).unwrap(),
        ));
        JoinFilter::new(
            expression,
            vec![
                ColumnIndex {
                    index: l,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: r,
                    side: JoinSide::Right,
                },
            ],
            schema,
        )
    }

    /// The arguments of a sort-merge join of the test inputs.
    #[derive(Debug, Clone)]
    struct JoinCase {
        join_type: JoinType,
        /// The key columns, e.g. `["k", "s"]` for `lk = rk AND ls = rs`.
        keys: Vec<&'static str>,
        filter: Option<(&'static str, Operator, &'static str)>,
        projection: Option<Vec<usize>>,
        sort_options: SortOptions,
        null_equality: NullEquality,
        left_rows: i64,
        right_rows: i64,
    }

    impl JoinCase {
        fn new(join_type: JoinType, keys: Vec<&'static str>) -> Self {
            Self {
                join_type,
                keys,
                filter: None,
                projection: None,
                sort_options: SortOptions::default(),
                null_equality: NullEquality::NullEqualsNothing,
                left_rows: 300,
                right_rows: 500,
            }
        }

        /// The `SortMergeJoinExec` of the case, over hash partitioned and
        /// sorted inputs, as DataFusion plans it.
        fn plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
            let sorted = |side: &str, rows: i64, skew: i64| {
                let scan = data_scan(side, rows, skew);
                let keys = self
                    .keys
                    .iter()
                    .map(|key| col(&format!("{side}{key}"), &scan.schema()))
                    .collect::<Result<Vec<_>>>()?;
                let repartition: Arc<dyn ExecutionPlan> = Arc::new(
                    RepartitionExec::try_new(scan, Partitioning::Hash(keys.clone(), 4))?,
                );
                let ordering = LexOrdering::new(keys.iter().map(|key| {
                    PhysicalSortExpr::new(Arc::clone(key), self.sort_options)
                }))
                .unwrap();
                let sort: Arc<dyn ExecutionPlan> = Arc::new(
                    SortExec::new(ordering, repartition).with_preserve_partitioning(true),
                );
                Ok::<_, DataFusionError>((sort, keys))
            };
            let (left, left_keys) = sorted("l", self.left_rows, 0)?;
            let (right, right_keys) = sorted("r", self.right_rows, 3)?;
            let filter = self
                .filter
                .map(|(left, op, right)| join_filter(left, op, right));
            let join = SortMergeJoinExec::try_new(
                left,
                right,
                left_keys.into_iter().zip(right_keys).collect(),
                filter,
                self.join_type,
                vec![self.sort_options; self.keys.len()],
                self.null_equality,
            )?
            .with_projection(self.projection.clone())?;
            Ok(Arc::new(join))
        }
    }

    /// Lower the sort-merge join `plan` with [`try_lower`], ordered or not,
    /// with its inputs lowered by the planner.
    fn lower_join(
        plan: &Arc<dyn ExecutionPlan>,
        ordered: bool,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        try_lower(plan, ordered, |input, _| {
            crate::planner::lower_execution_plan(input)
        })?
        .ok_or_else(|| internal_datafusion_err!("The join was not lowered"))
    }

    /// Execute `root`, lowered from `plan`, with the pipeline engine.
    pub(crate) async fn run_native(
        root: &Arc<dyn PhysicalOperator>,
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
    ) -> Result<EngineOutput> {
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(PipelineRootExec::try_new(Arc::clone(root), plan)?);
        let batches = with_timeout(collect(Arc::clone(&exec), task_ctx)).await?;
        Ok(EngineOutput {
            schema: exec.schema(),
            plan: exec,
            batches,
            ordered: false,
        })
    }

    /// Assert that the rows of `batches` are sorted by `ordering` (with
    /// `-0.0` and `+0.0` as one value, as the merge join sorts them).
    fn assert_sorted(batches: &[RecordBatch], ordering: &LexOrdering, context: &str) {
        let Some(first) = batches.first() else {
            return;
        };
        let batch = concat_batches(&first.schema(), batches).unwrap();
        let schema = batch.schema();
        let fields = ordering
            .iter()
            .map(|sort_expr| {
                SortField::new_with_options(
                    sort_expr.expr.data_type(&schema).unwrap(),
                    sort_expr.options,
                )
            })
            .collect::<Vec<_>>();
        let keys = ordering
            .iter()
            .map(|sort_expr| {
                let values = sort_expr
                    .expr
                    .evaluate(&batch)
                    .unwrap()
                    .into_array(batch.num_rows())
                    .unwrap();
                normalize_floats(&values)
            })
            .collect::<Vec<_>>();
        let rows = RowConverter::new(fields)
            .unwrap()
            .convert_columns(&keys)
            .unwrap();
        for row in 1..rows.num_rows() {
            assert!(
                rows.row(row - 1) <= rows.row(row),
                "row {row} is not sorted by {ordering}: {context}"
            );
        }
    }

    /// Assert that the join of `case`, lowered to a merge join if `ordered`
    /// (and to a hash join otherwise), produces the same rows as DataFusion's
    /// sort-merge join, and, for a merge join, in its order. Returns the
    /// lowered join.
    async fn assert_case(case: &JoinCase, ordered: bool) -> Arc<dyn PhysicalOperator> {
        let context = format!("{case:?}, ordered: {ordered}");
        let plan = case.plan().unwrap();
        let batches = with_timeout(collect(case.plan().unwrap(), task_ctx()))
            .await
            .unwrap_or_else(|e| panic!("DataFusion fails: {e}\n{context}"));
        let expected = EngineOutput {
            schema: plan.schema(),
            plan: Arc::clone(&plan),
            batches,
            ordered: false,
        };
        let root = lower_join(&plan, ordered).unwrap();
        let tree = format_operator_tree(&root);
        let actual = run_native(&root, Arc::clone(&plan), task_ctx())
            .await
            .unwrap_or_else(|e| panic!("The native join fails: {e}\n{context}\n{tree}"));
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\n{context}\n{tree}");
        }
        if ordered {
            assert!(root.is::<PhysicalMergeJoin>(), "{context}\n{tree}");
            // The order of the streamed input, the first part of the order of
            // the join. (DataFusion's order of a join of both inputs goes on
            // with the order of the buffered input, which the rows of NULLs of
            // the streamed rows without a match break.)
            if let Some(ordering) = plan.output_ordering().and_then(|ordering| {
                LexOrdering::new(ordering.iter().take(case.keys.len()).cloned())
            }) {
                assert_sorted(&actual.batches, &ordering, &context);
            }
        } else {
            assert!(root.is::<PhysicalHashJoin>(), "{context}\n{tree}");
        }
        assert!(bridged_plans(&root).is_empty(), "{context}\n{tree}");
        root
    }

    // ---------------------------------------------------------------------
    // Join semantics
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_join_types_match_datafusion() {
        for join_type in ALL_JOIN_TYPES {
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                for filter in [None, Some(("id", Operator::Lt, "id"))] {
                    for ordered in [true, false] {
                        let case = JoinCase {
                            filter,
                            null_equality,
                            ..JoinCase::new(join_type, vec!["k"])
                        };
                        assert_case(&case, ordered).await;
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn keys_of_many_types_and_several_columns() {
        // Not `f` before other keys: DataFusion sorts `-0.0` before `+0.0`
        // and merges the keys as equal, so that it misses the matches of the
        // keys after them (see `zeros_before_other_keys_match`).
        let keys: [Vec<&'static str>; 8] = [
            vec!["f"],
            vec!["s"],
            vec!["v"],
            vec!["d"],
            vec!["t"],
            vec!["k", "s"],
            vec!["d", "t", "f"],
            vec!["v", "k", "f"],
        ];
        for keys in keys {
            for join_type in [
                JoinType::Inner,
                JoinType::Full,
                JoinType::RightSemi,
                JoinType::LeftAnti,
                JoinType::LeftMark,
            ] {
                for ordered in [true, false] {
                    assert_case(&JoinCase::new(join_type, keys.clone()), ordered).await;
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sort_orders_filters_and_projections() {
        let orders = [
            SortOptions::new(false, false),
            SortOptions::new(true, true),
            SortOptions::new(true, false),
        ];
        for sort_options in orders {
            for join_type in [JoinType::Left, JoinType::Right, JoinType::RightAnti] {
                let case = JoinCase {
                    sort_options,
                    ..JoinCase::new(join_type, vec!["k", "d"])
                };
                assert_case(&case, true).await;
            }
        }
        for filter in [("s", Operator::NotEq, "s"), ("d", Operator::Gt, "d")] {
            for join_type in ALL_JOIN_TYPES {
                let case = JoinCase {
                    filter: Some(filter),
                    ..JoinCase::new(join_type, vec!["k"])
                };
                for ordered in [true, false] {
                    assert_case(&case, ordered).await;
                }
            }
        }
        // Projections of both inputs, of one of them and of no column.
        for (join_type, projection) in [
            (JoinType::Inner, vec![9, 0, 3]),
            (JoinType::Left, vec![1, 8]),
            (JoinType::Right, vec![]),
            (JoinType::Full, vec![13, 2, 7]),
            (JoinType::LeftSemi, vec![3, 0]),
            (JoinType::RightMark, vec![7, 1]),
        ] {
            let case = JoinCase {
                projection: Some(projection),
                filter: Some(("id", Operator::Gt, "id")),
                ..JoinCase::new(join_type, vec!["k"])
            };
            for ordered in [true, false] {
                assert_case(&case, ordered).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_inputs() {
        for join_type in ALL_JOIN_TYPES {
            for (left_rows, right_rows) in [(0, 500), (300, 0), (0, 0)] {
                let case = JoinCase {
                    left_rows,
                    right_rows,
                    ..JoinCase::new(join_type, vec!["k"])
                };
                for ordered in [true, false] {
                    assert_case(&case, ordered).await;
                }
            }
        }
    }

    /// The scan of the rows `(id, k)` of `rows`, in one partition.
    fn key_scan(name: &str, rows: &[(i64, Option<f64>, i32)]) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new(format!("{name}id"), DataType::Int64, false),
            Field::new(format!("{name}f"), DataType::Float64, true),
            Field::new(format!("{name}k"), DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Float64Array::from_iter(rows.iter().map(|row| row.1))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.2))),
            ],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn zeros_before_other_keys_match() -> Result<()> {
        // `-0.0 = +0.0`: `(+0.0, 1)` matches `(-0.0, 1)` and `(-0.0, 5)`
        // matches `(+0.0, 5)`, although sorting by the raw keys separates them
        // (DataFusion's sort-merge join, which sorts by them and merges them as
        // equal, misses the first match). The native joins find both, as
        // DataFusion's hash join does.
        let left = key_scan("l", &[(1, Some(-0.0), 5), (2, Some(0.0), 1)]);
        let right = key_scan("r", &[(3, Some(-0.0), 1), (4, Some(0.0), 5)]);
        let keys =
            |plan: &Arc<dyn ExecutionPlan>, side: &str| -> Result<Vec<PhysicalExprRef>> {
                Ok(vec![
                    col(&format!("{side}f"), &plan.schema())?,
                    col(&format!("{side}k"), &plan.schema())?,
                ])
            };
        let sorted = |plan: Arc<dyn ExecutionPlan>, keys: &[PhysicalExprRef]| {
            let ordering = LexOrdering::new(
                keys.iter()
                    .map(|key| PhysicalSortExpr::new_default(Arc::clone(key))),
            )
            .unwrap();
            Arc::new(SortExec::new(ordering, plan)) as Arc<dyn ExecutionPlan>
        };
        let (left_keys, right_keys) = (keys(&left, "l")?, keys(&right, "r")?);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortMergeJoinExec::try_new(
            sorted(left, &left_keys),
            sorted(right, &right_keys),
            left_keys.into_iter().zip(right_keys).collect(),
            None,
            JoinType::Inner,
            vec![SortOptions::default(); 2],
            NullEquality::NullEqualsNothing,
        )?);
        for ordered in [true, false] {
            let root = lower_join(&plan, ordered)?;
            let output = run_native(&root, Arc::clone(&plan), task_ctx()).await?;
            let batch = concat_batches(&output.schema, &output.batches)?;
            let mut pairs = (0..batch.num_rows())
                .map(|row| {
                    let id = |column: usize| {
                        batch
                            .column(column)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .value(row)
                    };
                    (id(0), id(3))
                })
                .collect::<Vec<_>>();
            pairs.sort_unstable();
            assert_eq!(pairs, vec![(1, 4), (2, 3)], "ordered: {ordered}");
        }
        Ok(())
    }

    /// A sort-merge join of `left` and `right` (two sorted inputs of
    /// [`TestSource`]s with the rows of `(id, k)`) on `k`.
    fn merge_join_of(
        left: Vec<(i64, i32)>,
        right: Vec<(i64, i32)>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let source = |side: &str,
                      rows: Vec<(i64, i32)>|
         -> Result<Arc<dyn PhysicalOperator>> {
            let schema = Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Int32, false),
            ]));
            let chunks = rows
                .chunks(1000)
                .map(|rows| {
                    let batch = RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![
                            Arc::new(Int64Array::from_iter_values(
                                rows.iter().map(|r| r.0),
                            )),
                            Arc::new(Int32Array::from_iter_values(
                                rows.iter().map(|r| r.1),
                            )),
                        ],
                    )?;
                    Ok(DataChunk::from_record_batch(batch))
                })
                .collect::<Result<Vec<_>>>()?;
            let source = Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks));
            let ordering = LexOrdering::new([PhysicalSortExpr::new_default(col(
                &format!("{side}k"),
                &schema,
            )?)])
            .unwrap();
            Ok(Arc::new(PhysicalSort::try_new(source, ordering, None)?))
        };
        let left = source("l", left)?;
        let right = source("r", right)?;
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        Ok(Arc::new(PhysicalMergeJoin::try_new(
            left,
            right,
            on,
            filter,
            join_type,
            &[SortOptions::default()],
            NullEquality::NullEqualsNothing,
            false,
            None,
        )?))
    }

    /// The number of rows of `batches`.
    pub(crate) fn num_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(RecordBatch::num_rows).sum()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn key_groups_span_several_chunks() -> Result<()> {
        // 3 keys with 1,000 rows on the left and 300 on the right: every left
        // row has 300 matches, more than fit into one chunk with the matches
        // of other rows, and some of them pass the filter.
        let left = (0..3000).map(|i| (i, (i % 3) as i32)).collect::<Vec<_>>();
        let right = (0..900).map(|i| (i, (i % 3) as i32)).collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(vec![
            Field::new("lid", DataType::Int64, false),
            Field::new("rid", DataType::Int64, false),
        ]));
        let expression = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                col("lid", &schema)?,
                Operator::Plus,
                col("rid", &schema)?,
            )),
            Operator::Modulo,
            datafusion_physical_expr::expressions::lit(7_i64),
        ));
        let filter = || {
            JoinFilter::new(
                Arc::new(BinaryExpr::new(
                    Arc::clone(&expression) as PhysicalExprRef,
                    Operator::Eq,
                    datafusion_physical_expr::expressions::lit(0_i64),
                )),
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
                Arc::clone(&schema),
            )
        };
        let expected_pairs = |filtered: bool| {
            left.iter()
                .map(|&(lid, lk)| {
                    right
                        .iter()
                        .filter(|&&(rid, rk)| {
                            lk == rk && (!filtered || (lid + rid) % 7 == 0)
                        })
                        .count()
                })
                .collect::<Vec<_>>()
        };
        let task_ctx = Arc::new(TaskContext::default());
        for (join_type, filtered) in [
            (JoinType::Inner, false),
            (JoinType::Inner, true),
            (JoinType::Left, true),
            (JoinType::LeftSemi, true),
            (JoinType::LeftAnti, true),
            (JoinType::LeftMark, true),
        ] {
            let join = merge_join_of(
                left.clone(),
                right.clone(),
                join_type,
                filtered.then(filter),
            )?;
            let batches =
                with_timeout(Executor::collect(join, Arc::clone(&task_ctx), 4)).await?;
            assert!(
                batches
                    .iter()
                    .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE),
                "{join_type}"
            );
            let matches = expected_pairs(filtered);
            let expected = match join_type {
                JoinType::Inner => matches.iter().sum(),
                JoinType::Left => matches.iter().map(|&n| n.max(1)).sum(),
                JoinType::LeftSemi => matches.iter().filter(|&&n| n > 0).count(),
                JoinType::LeftAnti => matches.iter().filter(|&&n| n == 0).count(),
                _ => left.len(),
            };
            assert_eq!(num_rows(&batches), expected, "{join_type}");
            // The rows come in the order of the left keys.
            let keys = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>();
            assert!(keys.is_sorted(), "{join_type}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unsorted_inputs_are_an_error() -> Result<()> {
        // Inputs that are not sorted by the keys (here, not sorted at all)
        // would be merged into a wrong result: the join fails.
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let source = |values: Vec<i32>| -> Arc<dyn PhysicalOperator> {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(values))],
            )
            .unwrap();
            Arc::new(TestSource::with_schema(
                Arc::clone(&schema),
                vec![DataChunk::from_record_batch(batch)],
            ))
        };
        for (left, right, side) in [
            (vec![1, 2, 3], vec![3, 1, 2], "buffered"),
            (vec![3, 1, 2], vec![1, 2, 3], "streamed"),
        ] {
            let join = PhysicalMergeJoin::try_new(
                source(left),
                source(right),
                vec![(col("k", &schema)?, col("k", &schema)?)],
                None,
                JoinType::Inner,
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
                false,
                None,
            )?;
            let err = with_timeout(Executor::collect(
                Arc::new(join),
                Arc::new(TaskContext::default()),
                2,
            ))
            .await
            .unwrap_err();
            assert!(
                err.to_string().contains(&format!(
                    "The {side} input of PhysicalMergeJoin is not sorted"
                )),
                "{err}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_buffered_rows_are_reserved_in_the_memory_pool() -> Result<()> {
        let left = (0..100).map(|i| (i, (i % 10) as i32)).collect::<Vec<_>>();
        let right = (0..20_000)
            .map(|i| (i, (i % 10) as i32))
            .collect::<Vec<_>>();
        let limited = |limit: usize| -> Result<Arc<TaskContext>> {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(limit, 1.0)
                .build_arc()?;
            Ok(Arc::new(TaskContext::default().with_runtime(runtime)))
        };
        let join =
            || merge_join_of(left.clone(), right.clone(), JoinType::LeftSemi, None);
        let err = with_timeout(Executor::collect(join()?, limited(64 * 1024)?, 4))
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        assert!(err.to_string().contains("PhysicalMergeJoin"), "{err}");
        let batches =
            with_timeout(Executor::collect(join()?, limited(64 << 20)?, 4)).await?;
        assert_eq!(num_rows(&batches), 100);
        Ok(())
    }

    #[test]
    fn pipelines_of_a_merge_join() -> Result<()> {
        let left = (0..10).map(|i| (i, i as i32)).collect::<Vec<_>>();
        for (join_type, expected) in [
            (
                JoinType::Inner,
                "MetaPipeline 0 (sink: ResultSink) children: [1, 3]\n  \
                 Pipeline 0: PhysicalSort -> PhysicalMergeJoin -> ResultSink depends on meta pipelines [1, 3]\n\
                 MetaPipeline 1 (sink: PhysicalMergeJoin) children: [2]\n  \
                 Pipeline 1: PhysicalSort -> PhysicalMergeJoin depends on meta pipelines [2]\n\
                 MetaPipeline 2 (sink: PhysicalSort)\n  \
                 Pipeline 2: TestSource -> PhysicalSort\n\
                 MetaPipeline 3 (sink: PhysicalSort)\n  \
                 Pipeline 3: TestSource -> PhysicalSort\n",
            ),
            (
                JoinType::Full,
                "MetaPipeline 0 (sink: ResultSink) children: [1, 3]\n  \
                 Pipeline 0: PhysicalSort -> PhysicalMergeJoin -> ResultSink depends on meta pipelines [1, 3]\n  \
                 Pipeline 4: PhysicalMergeJoin -> ResultSink depends on pipelines [0]\n\
                 MetaPipeline 1 (sink: PhysicalMergeJoin) children: [2]\n  \
                 Pipeline 1: PhysicalSort -> PhysicalMergeJoin depends on meta pipelines [2]\n\
                 MetaPipeline 2 (sink: PhysicalSort)\n  \
                 Pipeline 2: TestSource -> PhysicalSort\n\
                 MetaPipeline 3 (sink: PhysicalSort)\n  \
                 Pipeline 3: TestSource -> PhysicalSort\n",
            ),
        ] {
            let join = merge_join_of(left.clone(), left.clone(), join_type, None)?;
            assert_eq!(format_pipelines(&join)?, expected, "{join_type}");
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Lowering of SQL queries
    // ---------------------------------------------------------------------

    /// Run `sql` on both engines (with the test tables) configured with
    /// `config`, and assert that they return the same rows, compared in their
    /// canonical forms (as a multiset), and that the pipeline engine executes
    /// all of it natively. Returns the operator tree and the output of the
    /// pipeline engine.
    async fn assert_native_same_rows(
        sql: &str,
        config: SessionConfig,
    ) -> (Arc<dyn PhysicalOperator>, EngineOutput) {
        let setup = async |ctx: &SessionContext| register_test_tables(ctx);
        let (mut expected, mut actual) = with_timeout(run_engines(sql, config, setup))
            .await
            .unwrap_or_else(|e| {
                panic!("Failed to run the query on both engines: {e}\n{sql}")
            });
        let root = Arc::clone(
            actual
                .plan
                .downcast_ref::<PipelineRootExec>()
                .expect("a pipeline plan")
                .operator(),
        );
        let tree = format_operator_tree(&root);
        expected.ordered = false;
        actual.ordered = false;
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\nquery: {sql}\nlowered: {tree}");
        }
        assert!(
            bridged_plans(&root).is_empty(),
            "Parts of the plan are left to DataFusion\nquery: {sql}\nlowered: {tree}"
        );
        (root, actual)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sort_merge_joins_of_queries_are_native_hash_joins() {
        for sql in [
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t1.d, t2.name FROM t1 LEFT JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t2.name, t2.w FROM t1 RIGHT JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t2.name FROM t1 FULL JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.k = t1.a)",
            "SELECT t1.id FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.k = t1.a)",
            "SELECT t1.id FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.k = t1.a) OR t1.id < 10",
            "SELECT t1.id FROM t1 WHERE t1.a NOT IN (SELECT k FROM t2)",
            "SELECT t1.id FROM t1 WHERE t1.a NOT IN (SELECT k FROM t2 WHERE k IS NOT NULL)",
            "SELECT t1.id, t2.w FROM t1 JOIN t2 ON t1.a = t2.k AND t1.b > t2.w",
            "SELECT t1.id, t2.w FROM t1 LEFT JOIN t2 ON t1.a = t2.k AND t1.b < t2.w * 3",
            "SELECT t1.id, t2.w FROM t1 FULL JOIN t2 ON t1.a = t2.k AND t1.id % 3 = t2.w % 3",
            "SELECT t1.id, t2.w FROM t1 JOIN t2 ON t1.a IS NOT DISTINCT FROM t2.k",
            "SELECT t1.id, t2.w FROM t1 JOIN t2 ON t1.a = t2.k AND t1.d = t2.name",
            "SELECT t1.id, t2.w FROM t1 JOIN t2 ON CAST(t1.a AS BIGINT) = t2.w",
            "SELECT a.id, b.id FROM t1 a JOIN t1 b ON a.h = b.h AND a.id < b.id AND a.id < 300",
            "SELECT a.id, c.name FROM t1 a JOIN t1 b ON a.id = b.id + 1 JOIN t2 c ON b.a = c.k",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t1.a, t1.id, t2.name",
            // DataFusion sorts the aggregation below the projection of its
            // columns, which the join reads (as in TPC-H Q15).
            "SELECT t2.name, s.total FROM t2 JOIN (SELECT a AS k, sum(b) AS total FROM t1 GROUP BY a) s ON t2.k = s.k",
            "SELECT t2.name, s.total FROM t2 LEFT JOIN (SELECT a AS k, sum(b) AS total FROM t1 WHERE c > 0 GROUP BY a) s ON t2.k = s.k WHERE s.total > 0 OR s.total IS NULL",
        ] {
            for partitions in [4, 3] {
                let (root, _) =
                    assert_native_same_rows(sql, sort_merge_config(partitions)).await;
                let tree = format_operator_tree(&root);
                assert!(count::<PhysicalHashJoin>(&root) >= 1, "{sql}\n{tree}");
                assert_eq!(count::<PhysicalMergeJoin>(&root), 0, "{sql}\n{tree}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_sort_merge_joins_are_native_merge_joins() {
        // DataFusion merges the sorted partitions of the joins (the joins sort
        // by the keys with NULLs first), possibly with a limit, through
        // projections and filters: the merge joins emit their rows in order.
        for (sql, keys) in [
            (
                "SELECT t1.a, t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t1.a NULLS FIRST",
                1,
            ),
            (
                "SELECT t1.a, t1.id, t2.name FROM t1 LEFT JOIN t2 ON t1.a = t2.k ORDER BY t1.a NULLS FIRST, t1.id",
                2,
            ),
            (
                "SELECT t2.k, t2.w, t1.id FROM t1 RIGHT JOIN t2 ON t1.a = t2.k ORDER BY t2.k NULLS FIRST",
                1,
            ),
            (
                "SELECT t1.a, t1.id FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.k = t1.a) ORDER BY t1.a NULLS FIRST",
                1,
            ),
            (
                "SELECT t1.a, t1.id FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.k = t1.a) ORDER BY t1.a NULLS FIRST",
                1,
            ),
            (
                "SELECT t1.a, t1.id + 1 AS x, t2.name FROM t1 JOIN t2 ON t1.a = t2.k WHERE t1.id + t2.w > 100 ORDER BY t1.a NULLS FIRST",
                1,
            ),
            (
                "SELECT t1.a, t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k AND t1.b > t2.w ORDER BY t1.a NULLS FIRST",
                1,
            ),
            (
                "SELECT t1.a FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t1.a NULLS FIRST LIMIT 30",
                1,
            ),
            (
                "SELECT t2.k, s.total, t2.name FROM t2 JOIN (SELECT a AS k, sum(b) AS total FROM t1 GROUP BY a) s ON t2.k = s.k ORDER BY t2.k NULLS FIRST",
                1,
            ),
        ] {
            let (root, output) = assert_native_same_rows(sql, sort_merge_config(4)).await;
            let tree = format_operator_tree(&root);
            assert_eq!(count::<PhysicalMergeJoin>(&root), 1, "{sql}\n{tree}");
            assert_eq!(count::<PhysicalHashJoin>(&root), 0, "{sql}\n{tree}");
            // The rows are sorted by their first `keys` columns: the first
            // with NULLs first, the second with NULLs last.
            let ordering = LexOrdering::new((0..keys).map(|i| {
                let key = Column::new(output.schema.field(i).name(), i);
                PhysicalSortExpr::new(Arc::new(key), SortOptions::new(false, i == 0))
            }))
            .unwrap();
            assert_sorted(&output.batches, &ordering, sql);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn null_aware_not_in_is_a_merge_join_that_keeps_its_order() {
        // DataFusion merges both inputs of the null-aware join into one
        // partition, so that the limit above it reads its rows in order.
        let sql = "SELECT t1.a FROM t1 WHERE t1.a NOT IN (SELECT k FROM t2 WHERE k IS NOT NULL) ORDER BY t1.a NULLS FIRST LIMIT 7";
        let (root, output) = assert_native_same_rows(sql, sort_merge_config(4)).await;
        let tree = format_operator_tree(&root);
        assert_eq!(count::<PhysicalMergeJoin>(&root), 1, "{tree}");
        assert_eq!(num_rows(&output.batches), 7);
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(Column::new("a", 0)),
            SortOptions::new(false, true),
        )])
        .unwrap();
        assert_sorted(&output.batches, &ordering, sql);
        for sql in [
            "SELECT t1.id FROM t1 WHERE t1.a NOT IN (SELECT k FROM t2)",
            "SELECT t1.a, t1.id FROM t1 WHERE t1.a NOT IN (SELECT k FROM t2 WHERE k IS NOT NULL)",
            "SELECT t1.a, t1.id FROM t1 WHERE t1.a NOT IN (SELECT k FROM t2 WHERE k > 1000)",
        ] {
            assert_native_same_rows(sql, sort_merge_config(4)).await;
        }
    }

    #[test]
    fn hash_joins_build_on_the_smaller_input() -> Result<()> {
        // The right input is smaller: the hash join builds on it, and swaps
        // the join (a `Left` join builds on its left input otherwise).
        for (left_rows, right_rows, build_rows) in [(300, 20, 20), (20, 300, 20)] {
            let case = JoinCase {
                left_rows,
                right_rows,
                ..JoinCase::new(JoinType::Left, vec!["k"])
            };
            let plan = case.plan()?;
            let root = lower_join(&plan, false)?;
            let join = root
                .downcast_ref::<PhysicalHashJoin>()
                .expect("a hash join");
            let expected = match build_rows == right_rows {
                true => JoinType::Right,
                false => JoinType::Left,
            };
            assert_eq!(join.join_type(), expected);
            assert_eq!(root.schema(), plan.schema());
        }
        Ok(())
    }
}
