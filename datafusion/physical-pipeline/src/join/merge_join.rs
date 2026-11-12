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
//! * an unordered join of inputs that never end (unbounded inputs) is a
//!   [`PhysicalHashJoin`] that builds on its bounded input, or a
//!   [`PhysicalSymmetricHashJoin`] if both are unbounded (see
//!   [`super::symmetric_hash_join`]); an ordered one is a merge join that
//!   reads them in their own order, see below.
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
//! MetaPipeline 0 (sink: ...) children: [1, 3]
//!   Pipeline 0: MergeJoinSort -> PhysicalMergeJoin -> ... depends on meta pipelines [1, 3]
//!   Pipeline 4: PhysicalMergeJoin -> ... depends on pipelines [0]          (Full joins)
//! MetaPipeline 1 (sink: PhysicalMergeJoin) children: [2]
//!   Pipeline 1: MergeJoinSort -> PhysicalMergeJoin depends on meta pipelines [2]
//! MetaPipeline 2 (sink: MergeJoinSort)
//!   Pipeline 2: buffered input -> MergeJoinSort
//! MetaPipeline 3 (sink: MergeJoinSort)
//!   Pipeline 3: streamed input -> MergeJoinSort
//! ```
//!
//! * Both inputs are sorted by parallel [`MergeJoinSort`]s, which replace the
//!   sorts and the repartitionings below the `SortMergeJoinExec`: by the
//!   order of the input of the `SortMergeJoinExec` (its sort, or the order of
//!   a sorted input), which the join keeps. Floating point keys are sorted
//!   with `-0.0` and `+0.0` as one value, as the join compares them. The sorts
//!   reserve the memory of the rows they keep in the memory pool of the
//!   session.
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
//! An input that never ends is not sorted (it is sorted already: DataFusion
//! cannot sort it either), but read in its own order (merged by a
//! `SortPreservingMergeExec` if it has several sorted partitions). Such a
//! streamed input runs in the pipeline of the join; such a buffered input is
//! read as a stream (`NestedInput`, an execution of its own) as the
//! streamed rows need its rows: before it joins a streamed row, the join reads
//! the buffered rows until one has a later key (or the input ends), waiting
//! for them (`Blocked`) if the input has none yet, and it drops the buffered
//! rows that no later streamed row matches (but for a `Full` join, which emits
//! the buffered rows without a match once the streamed input ends). So the
//! join emits its rows, in order, as its inputs produce them. An input in its
//! own order puts `-0.0` before `+0.0`: where a float key (or a float in a
//! nested key) is a zero, the keys or values after it are sorted for either
//! sign, not for both (e.g. `(-0.0, 5)` comes before `(+0.0, 1)`). The join
//! sorts such runs of buffered rows by its keys as they come (it joins those
//! that no row still to come comes before: those up to a row without a zero of
//! the sign that comes first in the order of its key, so that a run of `+0.0`
//! of an ascending key, or of `-0.0` of a descending one, does not wait for
//! its end; the rows of a run of the other sign wait for a row after them),
//! and the streamed rows of such a run (which keep their order) find the
//! buffered rows of their keys from where the run started (see `ZeroRuns`).
//!
//! Join keys are equal with DataFusion's join key equality: `-0.0 = +0.0`,
//! NaNs with the same bits are equal, and NULLs are equal only with
//! [`NullEquality::NullEqualsNull`]. The null-aware `LeftAnti` join (`NOT
//! IN`) emits no row if a buffered key is NULL, and the streamed rows with a
//! NULL key only if the buffered input is empty.
//!
//! The module also holds the parts that the merge join shares with the
//! [`PhysicalSymmetricHashJoin`]: the output of a join (`JoinOutput`), the
//! rows that it gathers (`JoinRows`, `StoredRows`), the evaluation of its
//! residual filter (`filter_mask`) and the inputs that it reads as streams
//! (`NestedInput`).
//!
//! [`PhysicalSymmetricHashJoin`]: super::symmetric_hash_join::PhysicalSymmetricHashJoin

use std::collections::VecDeque;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    make_array, new_empty_array, new_null_array,
};
use arrow::buffer::NullBuffer;
use arrow::compute::{
    SortOptions, concat_batches, interleave, prep_null_mask_filter, take,
    take_record_batch,
};
use arrow::datatypes::{
    DataType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
    Schema, SchemaRef, UnionMode,
};
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
use futures::StreamExt;
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::executor::{Executor, PipelineStream};
use crate::join::PhysicalHashJoin;
use crate::join::symmetric_hash_join;
use crate::operator::{
    EmptyState, ExecutionContext, GlobalOperatorState, GlobalSinkState,
    GlobalSourceState, LocalSinkState, LocalSourceState, OperatorResult, OperatorState,
    PhysicalOperator, SinkFinalize, SinkResult, SourceResult, TaskInterrupt,
    downcast_state, downcast_state_mut,
};
use crate::operators::caching::CachingOperator;
use crate::operators::filter::PhysicalFilter;
use crate::operators::projection::PhysicalProjection;
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::sort_sink::{SORT_RUN_BYTES, SORT_RUN_ROWS};
use crate::sort::{RunKeys, RunMerger, SortKeys, SortedRun, memory_size};

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

/// How a join reads one of its inputs, which decides how the input is
/// lowered (see `Lowering::lower_in` of the planner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRole {
    /// The join reads all rows of the input before it emits any, in a
    /// pipeline that runs until the input ends: the build side of a hash
    /// join, the inputs that a merge join sorts.
    Collected,
    /// The input runs in the pipeline of the join, which may stop before the
    /// input ends when the pipeline does: the probe side of a hash join.
    Pipelined,
    /// The join reads the rows of the input as they come (an input that never
    /// ends): no operator of the input may hold its rows back (e.g. the cache
    /// of a filter, until it is full), as the input may never produce more.
    Streamed,
}

/// Lower a DataFusion `SortMergeJoinExec` natively, see the [module
/// documentation](self): to a [`PhysicalMergeJoin`] if `ordered` (a node above
/// it relies on its order, see [`ordered_join_below`]) or if it promises its
/// order to the nodes above it (see [`promises_order`]), and to a
/// [`PhysicalHashJoin`] (or, for inputs that never end, a
/// [`PhysicalSymmetricHashJoin`](super::symmetric_hash_join::PhysicalSymmetricHashJoin))
/// otherwise.
///
/// `lower_input` lowers an input of the join, which the join reads as its
/// [`InputRole`] says.
///
/// Returns `Ok(None)` for any other node, and for the joins that no native
/// join executes: joins whose keys have different types on the two sides,
/// whose keys or columns the native joins cannot store or sort (e.g. keys of
/// unions), and null-aware joins of two inputs that never end, which neither
/// of them can be collected for first.
pub fn try_lower<F>(
    plan: &Arc<dyn ExecutionPlan>,
    ordered: bool,
    mut lower_input: F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, InputRole) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(join) = plan.downcast_ref::<SortMergeJoinExec>() else {
        return Ok(None);
    };
    let unbounded =
        [join.left(), join.right()].map(|input| input.boundedness().is_unbounded());
    if ordered || promises_order(plan) {
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
        // A null-aware join (`NOT IN`), which the symmetric join does not
        // implement, merges them (its inputs are sorted), also if its output
        // has no order (e.g. it projects none of its keys).
        if join.null_aware() {
            let Some(spec) = MergeJoinSpec::of(join)? else {
                return Ok(None);
            };
            let join = lower_merge_join(join, spec, &mut lower_input)?;
            return Ok(join.map(|join| Arc::new(join) as Arc<dyn PhysicalOperator>));
        }
        return symmetric_hash_join::lower_symmetric(&equi_join, None, &mut lower_input);
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
    // A join that the hash join does not implement: a merge join, of the same
    // inputs if they end.
    let Some(spec) = MergeJoinSpec::of(join)? else {
        return Ok(None);
    };
    if unbounded.contains(&true) {
        let join = lower_merge_join(join, spec, &mut lower_input)?;
        return Ok(join.map(|join| Arc::new(join) as Arc<dyn PhysicalOperator>));
    }
    let (left, right) = match build_right {
        true => (probe, build),
        false => (build, probe),
    };
    let join = merge_join_of_inputs(join, spec, left, right)?;
    Ok(join.map(|join| Arc::new(join) as Arc<dyn PhysicalOperator>))
}

/// How a hash join reads its probe side `probe`: in its pipeline, as a stream
/// if it never ends.
pub(crate) fn probe_role(probe: &Arc<dyn ExecutionPlan>) -> InputRole {
    match probe.boundedness().is_unbounded() {
        true => InputRole::Streamed,
        false => InputRole::Pipelined,
    }
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

/// Lower `join` to a [`PhysicalMergeJoin`] with the specification `spec`, of
/// its inputs lowered with `lower_input` (without the sorts and
/// repartitionings below the join): sorted by [`MergeJoinSort`]s if they end,
/// and read in their own order if they never end (see [`ordered_input`]),
/// the buffered input then as a stream.
fn lower_merge_join<F>(
    join: &SortMergeJoinExec,
    spec: MergeJoinSpec,
    lower_input: &mut F,
) -> Result<Option<PhysicalMergeJoin>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, InputRole) -> Result<Arc<dyn PhysicalOperator>>,
{
    let mut children = Vec::with_capacity(2);
    for (side, input) in [
        (JoinSide::Left, join.left()),
        (JoinSide::Right, join.right()),
    ] {
        let child = match input.boundedness().is_unbounded() {
            false => {
                // The sorts read all of their input.
                let lowered =
                    lower_input(&without_reordering(input)?, InputRole::Collected)?;
                sorted_input(join, side, input, lowered)?
            }
            true => ordered_input(input, lower_input)?,
        };
        let Some(child) = child else {
            return Ok(None);
        };
        children.push(child);
    }
    let buffered = match spec.streamed {
        JoinSide::Left => join.right(),
        _ => join.left(),
    };
    let streams_buffered = buffered.boundedness().is_unbounded();
    let (Some(right), Some(left)) = (children.pop(), children.pop()) else {
        return internal_err!("A merge join without inputs");
    };
    Ok(Some(PhysicalMergeJoin {
        children: vec![left, right],
        spec,
        streams_buffered,
    }))
}

/// The [`PhysicalMergeJoin`] of `join` with the specification `spec`, over
/// its lowered inputs `left` and `right` (without the sorts and
/// repartitionings below the join), inputs that end, which it sorts (see
/// [`sorted_input`]). `Ok(None)` if they cannot be sorted natively.
fn merge_join_of_inputs(
    join: &SortMergeJoinExec,
    spec: MergeJoinSpec,
    left: Arc<dyn PhysicalOperator>,
    right: Arc<dyn PhysicalOperator>,
) -> Result<Option<PhysicalMergeJoin>> {
    let left = sorted_input(join, JoinSide::Left, join.left(), left)?;
    let right = sorted_input(join, JoinSide::Right, join.right(), right)?;
    let (Some(left), Some(right)) = (left, right) else {
        return Ok(None);
    };
    Ok(Some(PhysicalMergeJoin {
        children: vec![left, right],
        spec,
        streams_buffered: false,
    }))
}

/// `lowered`, the lowered input `input` (on the side `side`) of `join`, an
/// input that ends, sorted by a [`MergeJoinSort`] by the order of `input` (see
/// [`sort_order`]). `Ok(None)` if it cannot be sorted natively.
fn sorted_input(
    join: &SortMergeJoinExec,
    side: JoinSide,
    input: &Arc<dyn ExecutionPlan>,
    lowered: Arc<dyn PhysicalOperator>,
) -> Result<Option<Arc<dyn PhysicalOperator>>> {
    check_lowered_input(input, &lowered)?;
    let keys = join.on().iter().map(|(left, right)| match side {
        JoinSide::Left => left,
        _ => right,
    });
    let Some(ordering) = sort_order(input, keys, join.sort_options())? else {
        return Ok(None);
    };
    Ok(Some(Arc::new(MergeJoinSort::try_new(lowered, ordering)?)))
}

/// `input`, an input of a sort-merge join that never ends, lowered with
/// `lower_input` to the rows of `input` in its order, on a single thread: a
/// stream that never ends is sorted already (DataFusion cannot sort it, nor
/// can the merge join). The nodes that only sort or distribute its rows are
/// removed (see [`without_reordering`]); if they merged several sorted
/// partitions, the input is merged by a `SortPreservingMergeExec` (in order)
/// instead. `Ok(None)` if `input` has no order, or its partitions are not
/// sorted by it.
fn ordered_input<F>(
    input: &Arc<dyn ExecutionPlan>,
    lower_input: &mut F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, InputRole) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(ordered) = ordered_plan(input)? else {
        return Ok(None);
    };
    let lowered = lower_input(&ordered, InputRole::Streamed)?;
    check_lowered_input(input, &lowered)?;
    Ok(Some(lowered))
}

/// The plan that produces the rows of `input` in its order, on a single
/// partition, for a join that reads them so (see [`ordered_input`]): `input`
/// without the nodes that only sort or distribute its rows, whose partitions
/// a `SortPreservingMergeExec` merges if it has several. `Ok(None)` if `input`
/// has no order, or its partitions are not sorted by it.
pub(crate) fn ordered_plan(
    input: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(ordering) = input.output_ordering().cloned() else {
        return Ok(None);
    };
    let stripped = without_reordering(input)?;
    if !stripped
        .equivalence_properties()
        .ordering_satisfy(ordering.clone())?
    {
        return Ok(None);
    }
    Ok(Some(
        match stripped.output_partitioning().partition_count() > 1 {
            true => Arc::new(SortPreservingMergeExec::new(ordering, stripped)),
            false => stripped,
        },
    ))
}

/// Check that `lowered` has the schema of `input`, which it was lowered from.
fn check_lowered_input(
    input: &Arc<dyn ExecutionPlan>,
    lowered: &Arc<dyn PhysicalOperator>,
) -> Result<()> {
    if lowered.schema() != input.schema() {
        return internal_err!(
            "The lowered input {} of a join has schema {:?}, expected {:?}",
            lowered.name(),
            lowered.schema(),
            input.schema()
        );
    }
    Ok(())
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

    /// The position of the first row of every batch.
    pub(crate) fn batch_starts(&self) -> &[usize] {
        &self.starts
    }

    /// The batches of the rows, in order.
    pub(crate) fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// The rows without the first `batches` batches, at positions decreased
    /// by the rows of these batches.
    pub(crate) fn without_first_batches(&self, batches: usize) -> Self {
        let mut rows = Self::new(Arc::clone(&self.schema));
        for batch in self.batches.iter().skip(batches) {
            rows.push(batch.clone());
        }
        rows
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
        // A row of NULLs for the NULL positions, if any (it can be big, e.g.
        // of a fixed size list).
        let nulls = positions
            .iter()
            .any(Option::is_none)
            .then(|| new_null_array(field.data_type(), 1));
        let arrays = self
            .batches
            .iter()
            .map(|batch| batch.column(column).as_ref())
            .chain(nulls.as_deref())
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

/// The memory of the arrays `keys`, the keys of the rows of `batch` (see
/// [`evaluate_keys`]), that are not columns of `batch` (whose memory is that
/// of the batch): of computed keys, and of columns of floats with `-0.0`.
pub(crate) fn computed_key_bytes(keys: &[ArrayRef], batch: &RecordBatch) -> usize {
    keys.iter()
        .filter(|key| {
            !batch
                .columns()
                .iter()
                .any(|column| Arc::ptr_eq(column, key))
        })
        .map(|key| key.get_array_memory_size())
        .sum()
}

/// The memory that evaluating and encoding the keys `keys` of the rows of
/// `batch` takes: of the computed keys (see [`computed_key_bytes`]), and
/// about that of their encoding in the row format (see [`encoded_bytes`]),
/// to reserve before they are encoded.
pub(crate) fn key_memory(keys: &[ArrayRef], batch: &RecordBatch) -> usize {
    computed_key_bytes(keys, batch)
        + keys
            .iter()
            .map(|key| encoded_bytes(key.as_ref()))
            .sum::<usize>()
        + (batch.num_rows() + 1) * size_of::<usize>()
}

/// About (an upper bound of) the bytes that the row format takes to encode
/// the values of `array`: the bytes of its values, those that views refer to
/// too (which the encoding copies, even if the views share them), in blocks
/// of 32 bytes with a byte of their own, and a few bytes for every row.
fn encoded_bytes(array: &dyn Array) -> usize {
    let values = match array.data_type() {
        DataType::Utf8View => array
            .as_string_view()
            .lengths()
            .map(|length| length as usize)
            .sum(),
        DataType::BinaryView => array
            .as_binary_view()
            .lengths()
            .map(|length| length as usize)
            .sum(),
        DataType::Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .map(|column| encoded_bytes(column.as_ref()))
            .sum(),
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Map(_, _) => encoded_bytes(list_ranges(array).0),
        _ => nested_row_bytes(array).iter().sum(),
    };
    values + values / 32 + array.len() * size_of::<u64>()
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

/// The bytes that gathering rows of an input of a join takes, for the
/// columns that the join gathers into the batch of its filter and into its
/// output (see [`JoinRows`]): the bytes of every row that do not depend on its
/// values, and, for the columns whose values have bytes of their own (e.g.
/// strings), the bytes of the values of each row (see
/// [`Self::variable_bytes`]). A join reserves the bytes of the rows that it
/// gathers before it gathers them.
#[derive(Debug, Clone)]
pub(crate) struct GatherWidths {
    /// For the filter and for the output (see [`FILTER`] and [`OUTPUT`]): the
    /// bytes of every row that do not depend on its values.
    fixed: [usize; 2],
    /// For the filter and for the output: the columns whose values have bytes
    /// of their own, and how to count them.
    variable: [Vec<(usize, VariableWidth)>; 2],
    /// For the filter and for the output: the bytes of a row of NULLs (see
    /// [`null_value_bytes`]).
    nulls: [usize; 2],
}

/// A join gathers about this many bytes of rows for a batch of candidate
/// pairs (for its filter and its output, see [`GatherWidths`]), or of rows
/// that it emits at once: fewer than [`DEFAULT_VECTOR_SIZE`] rows if they are
/// wide. A batch has a pair (or a row) at least.
pub(crate) const PART_BYTES: usize = 1 << 20;

/// The bytes of the indices of a candidate pair of rows of a join, and of a
/// pair of its output.
pub(crate) const PAIR_INDEX_BYTES: usize =
    2 * (size_of::<u32>() + size_of::<Option<usize>>());

/// The batch of the filter of a join, see [`GatherWidths`].
pub(crate) const FILTER: usize = 0;

/// The output of a join, see [`GatherWidths`].
pub(crate) const OUTPUT: usize = 1;

/// How [`GatherWidths`] counts the bytes of the values of a column whose
/// values have bytes of their own.
#[derive(Debug, Clone, Copy)]
enum VariableWidth {
    /// Strings or binaries with offsets of 32 bits: the bytes of every value.
    Offsets32,
    /// Strings or binaries with offsets of 64 bits: the bytes of every value.
    Offsets64,
    /// Other values (e.g. lists or structs): the bytes of the values of every
    /// row, with those nested in them (see [`nested_row_bytes`]).
    Nested,
}

impl GatherWidths {
    /// The widths of the rows of an input with the schema `schema`, whose
    /// columns `filter` a join gathers for its filter and `output` for its
    /// output (a column as often as it gathers it).
    pub(crate) fn new(
        schema: &Schema,
        filter: impl IntoIterator<Item = usize>,
        output: impl IntoIterator<Item = usize>,
    ) -> Self {
        let mut fixed = [0; 2];
        let mut variable = [vec![], vec![]];
        let mut nulls = [0_usize; 2];
        let columns = [
            filter.into_iter().collect::<Vec<_>>(),
            output.into_iter().collect(),
        ];
        for (which, columns) in columns.into_iter().enumerate() {
            for column in columns {
                let Some(field) = schema.fields().get(column) else {
                    continue;
                };
                let (bytes, width) = value_width(field.data_type());
                fixed[which] += bytes;
                if let Some(width) = width {
                    variable[which].push((column, width));
                }
                nulls[which] =
                    nulls[which].saturating_add(null_value_bytes(field.data_type()));
            }
        }
        Self {
            fixed,
            variable,
            nulls,
        }
    }

    /// For every row of `batch`, a batch of the input: the bytes of its values
    /// that have bytes of their own, for the filter and for the output.
    pub(crate) fn variable_bytes(&self, batch: &RecordBatch) -> Vec<[u32; 2]> {
        let mut bytes = vec![[0; 2]; batch.num_rows()];
        for (which, columns) in self.variable.iter().enumerate() {
            for &(column, width) in columns {
                if let Some(array) = batch.columns().get(column) {
                    add_value_bytes(array, width, which, &mut bytes);
                }
            }
        }
        bytes
    }

    /// The bytes of a row in the batch `which` ([`FILTER`] or [`OUTPUT`]),
    /// whose values have the bytes `variable` of their own (see
    /// [`Self::variable_bytes`]).
    pub(crate) fn row_bytes(&self, variable: [u32; 2], which: usize) -> usize {
        self.fixed[which] + variable[which] as usize
    }

    /// The bytes of a row of NULLs in the batch `which` (e.g. for the other
    /// input of the rows without a match of an outer join), as
    /// `new_null_array` allocates them: of the values nested in the NULLs,
    /// too (see [`null_value_bytes`]).
    pub(crate) fn null_row_bytes(&self, which: usize) -> usize {
        self.nulls[which]
    }
}

/// The bytes that `new_null_array` allocates for a NULL of `data_type` (with
/// a byte for its validity): those of its offsets, and of the NULLs of its
/// children that it has (a fixed size list has its size of them, a struct one
/// of every field, a sparse union one of every child). Lists, maps and
/// strings have no values.
fn null_value_bytes(data_type: &DataType) -> usize {
    let validity = 1;
    match data_type {
        DataType::Null => 0,
        DataType::Boolean => 1 + validity,
        DataType::Utf8 | DataType::Binary | DataType::List(_) | DataType::Map(_, _) => {
            size_of::<i32>() + validity
        }
        DataType::LargeUtf8 | DataType::LargeBinary | DataType::LargeList(_) => {
            size_of::<i64>() + validity
        }
        DataType::ListView(_) => 2 * size_of::<i32>() + validity,
        DataType::LargeListView(_) => 2 * size_of::<i64>() + validity,
        DataType::Utf8View | DataType::BinaryView => size_of::<u128>() + validity,
        DataType::FixedSizeBinary(width) => {
            usize::try_from(*width).unwrap_or_default() + validity
        }
        DataType::FixedSizeList(field, size) => usize::try_from(*size)
            .unwrap_or_default()
            .saturating_mul(null_value_bytes(field.data_type()))
            .saturating_add(validity),
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| null_value_bytes(field.data_type()))
            .fold(validity, usize::saturating_add),
        DataType::Union(fields, mode) => {
            let offsets = match mode {
                UnionMode::Sparse => 0,
                UnionMode::Dense => size_of::<i32>(),
            };
            fields
                .iter()
                .map(|(_, field)| null_value_bytes(field.data_type()))
                .fold(size_of::<i8>() + offsets, usize::saturating_add)
        }
        DataType::Dictionary(key, _) => {
            key.primitive_width().unwrap_or(size_of::<u64>()) + validity
        }
        DataType::RunEndEncoded(run_ends, values) => {
            run_ends
                .data_type()
                .primitive_width()
                .unwrap_or(size_of::<u64>())
                + null_value_bytes(values.data_type())
        }
        data_type => data_type.primitive_width().unwrap_or(size_of::<u64>()) + validity,
    }
}

/// The bytes of a value of `data_type` that do not depend on the value (with
/// a byte for its validity), and how to count its bytes of its own, if it has
/// some.
fn value_width(data_type: &DataType) -> (usize, Option<VariableWidth>) {
    match data_type {
        DataType::Null => (0, None),
        DataType::Boolean => (1, None),
        DataType::Utf8 | DataType::Binary => {
            (size_of::<i32>() + 1, Some(VariableWidth::Offsets32))
        }
        DataType::LargeUtf8 | DataType::LargeBinary => {
            (size_of::<i64>() + 1, Some(VariableWidth::Offsets64))
        }
        // The gathered views share the buffers of the values.
        DataType::Utf8View | DataType::BinaryView => (size_of::<u128>() + 1, None),
        DataType::FixedSizeBinary(width) => {
            (usize::try_from(*width).unwrap_or_default() + 1, None)
        }
        data_type => match data_type.primitive_width() {
            Some(width) => (width + 1, None),
            None => (0, Some(VariableWidth::Nested)),
        },
    }
}

/// Add the bytes of the values of `array`, counted as `width` says, to the
/// bytes `bytes[row][which]` of its rows.
fn add_value_bytes(
    array: &ArrayRef,
    width: VariableWidth,
    which: usize,
    bytes: &mut [[u32; 2]],
) {
    let mut add = |row: usize, value: usize| {
        if let Some(bytes) = bytes.get_mut(row) {
            let value = u32::try_from(value).unwrap_or(u32::MAX);
            bytes[which] = bytes[which].saturating_add(value);
        }
    };
    match (width, array.data_type()) {
        (VariableWidth::Offsets32, DataType::Utf8) => {
            for (row, ends) in array
                .as_string::<i32>()
                .value_offsets()
                .windows(2)
                .enumerate()
            {
                add(row, usize::try_from(ends[1] - ends[0]).unwrap_or_default());
            }
        }
        (VariableWidth::Offsets32, _) => {
            for (row, ends) in array
                .as_binary::<i32>()
                .value_offsets()
                .windows(2)
                .enumerate()
            {
                add(row, usize::try_from(ends[1] - ends[0]).unwrap_or_default());
            }
        }
        (VariableWidth::Offsets64, DataType::LargeUtf8) => {
            for (row, ends) in array
                .as_string::<i64>()
                .value_offsets()
                .windows(2)
                .enumerate()
            {
                add(row, usize::try_from(ends[1] - ends[0]).unwrap_or_default());
            }
        }
        (VariableWidth::Offsets64, _) => {
            for (row, ends) in array
                .as_binary::<i64>()
                .value_offsets()
                .windows(2)
                .enumerate()
            {
                add(row, usize::try_from(ends[1] - ends[0]).unwrap_or_default());
            }
        }
        (VariableWidth::Nested, _) => {
            for (row, bytes) in nested_row_bytes(array.as_ref()).into_iter().enumerate() {
                add(row, bytes);
            }
        }
    }
}

/// The bytes that gathering every row of `array` takes: its values, with the
/// values nested in it (of the range of a list, the fields of a struct, the
/// value of a dictionary key, the values of all children of a sparse union,
/// which the gathering kernels copy, ...), its offsets and its validity.
/// Exact for the values of fixed width, strings, binaries, lists, maps,
/// structs, dictionaries and unions; for other types (e.g. run-end encoded
/// values), every row may take as many bytes as the whole array, an upper
/// bound.
fn nested_row_bytes(array: &dyn Array) -> Vec<usize> {
    let rows = array.len();
    // The bytes of the rows of a list whose ranges of `values` are delimited
    // by `offsets` (the ranges of `starts` and `ends`).
    fn ranges<O: Copy + TryInto<usize>>(
        starts: impl Iterator<Item = O>,
        ends: impl Iterator<Item = O>,
        values: &dyn Array,
        fixed: usize,
    ) -> Vec<usize> {
        let mut sums = vec![0];
        for bytes in nested_row_bytes(values) {
            sums.push(sums[sums.len() - 1] + bytes);
        }
        let at = |offset: O| {
            let offset = offset.try_into().unwrap_or(usize::MAX).min(sums.len() - 1);
            sums[offset]
        };
        starts
            .zip(ends)
            .map(|(start, end)| fixed + at(end).saturating_sub(at(start)))
            .collect()
    }
    let validity = 1;
    match array.data_type() {
        DataType::Utf8
        | DataType::Binary
        | DataType::LargeUtf8
        | DataType::LargeBinary => {
            let (fixed, width) = value_width(array.data_type());
            let mut bytes = vec![[0; 2]; rows];
            if let Some(width) = width {
                add_value_bytes(&make_array(array.to_data()), width, 0, &mut bytes);
            }
            bytes
                .iter()
                .map(|bytes| fixed + bytes[0] as usize)
                .collect()
        }
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let offsets = list.value_offsets();
            let fixed = size_of::<i32>() + validity;
            ranges(
                offsets.iter().copied(),
                offsets.iter().skip(1).copied(),
                list.values().as_ref(),
                fixed,
            )
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let offsets = list.value_offsets();
            let fixed = size_of::<i64>() + validity;
            ranges(
                offsets.iter().copied(),
                offsets.iter().skip(1).copied(),
                list.values().as_ref(),
                fixed,
            )
        }
        DataType::Map(_, _) => {
            let map = array.as_map();
            let offsets = map.value_offsets();
            let fixed = size_of::<i32>() + validity;
            ranges(
                offsets.iter().copied(),
                offsets.iter().skip(1).copied(),
                map.entries(),
                fixed,
            )
        }
        DataType::ListView(_) => {
            let list = array.as_list_view::<i32>();
            let ends = list
                .value_offsets()
                .iter()
                .zip(list.value_sizes())
                .map(|(offset, size)| offset + size);
            let fixed = 2 * size_of::<i32>() + validity;
            ranges(
                list.value_offsets().iter().copied(),
                ends,
                list.values().as_ref(),
                fixed,
            )
        }
        DataType::LargeListView(_) => {
            let list = array.as_list_view::<i64>();
            let ends = list
                .value_offsets()
                .iter()
                .zip(list.value_sizes())
                .map(|(offset, size)| offset + size);
            let fixed = 2 * size_of::<i64>() + validity;
            ranges(
                list.value_offsets().iter().copied(),
                ends,
                list.values().as_ref(),
                fixed,
            )
        }
        DataType::FixedSizeList(_, _) => {
            let list = array.as_fixed_size_list();
            let size = usize::try_from(list.value_length()).unwrap_or_default();
            let starts = (0..rows).map(|row| row * size);
            ranges(
                starts.clone(),
                starts.map(|start| start + size),
                list.values().as_ref(),
                validity,
            )
        }
        DataType::Struct(_) => {
            let mut bytes = vec![validity; rows];
            for column in array.as_struct().columns() {
                for (row, column_bytes) in
                    nested_row_bytes(column.as_ref()).into_iter().enumerate()
                {
                    if let Some(bytes) = bytes.get_mut(row) {
                        *bytes += column_bytes;
                    }
                }
            }
            bytes
        }
        DataType::Dictionary(key, _) => {
            let dictionary = array.as_any_dictionary();
            let values = nested_row_bytes(dictionary.values().as_ref());
            let key = key.primitive_width().unwrap_or(size_of::<u64>()) + validity;
            dictionary
                .normalized_keys()
                .into_iter()
                .map(|value| key + values.get(value).copied().unwrap_or_default())
                .collect()
        }
        DataType::Union(fields, mode) => {
            let union = array.as_union();
            let children = fields
                .iter()
                .map(|(type_id, _)| {
                    (type_id, nested_row_bytes(union.child(type_id).as_ref()))
                })
                .collect::<Vec<_>>();
            match mode {
                // A sparse union gathers the values of all its children for
                // every row, those of the children that are not its type, too.
                UnionMode::Sparse => (0..rows)
                    .map(|row| {
                        let values = children
                            .iter()
                            .map(|(_, bytes)| bytes.get(row).copied().unwrap_or_default())
                            .sum::<usize>();
                        size_of::<i8>() + values
                    })
                    .collect(),
                // A dense union gathers the value of its type only.
                UnionMode::Dense => (0..rows)
                    .map(|row| {
                        let type_id = union.type_id(row);
                        let child_row = union.value_offset(row);
                        let child = children
                            .iter()
                            .find(|(id, _)| *id == type_id)
                            .and_then(|(_, bytes)| bytes.get(child_row).copied())
                            .unwrap_or_default();
                        size_of::<i8>() + size_of::<i32>() + child
                    })
                    .collect(),
            }
        }
        data_type => match value_width(data_type) {
            (fixed, None) => vec![fixed; rows],
            (_, Some(_)) => vec![array.get_array_memory_size(); rows],
        },
    }
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
// The inputs of joins that are read as streams, shared with the symmetric
// hash join
// ---------------------------------------------------------------------------

/// An input of a join that the join reads as a stream: its operator tree is
/// executed by an [`Executor`] of its own (on the threads of the session, with
/// the memory pool of the session), and the join reads its rows as they come,
/// in their order.
///
/// The joins read so the inputs that never end whose rows they need at the
/// same time as the rows of another input: the pipelines of an execution read
/// the output of a pipeline breaker only once it is complete, and the
/// executor runs the pipelines that share an operator that is not parallel
/// (e.g. a limit that keeps the order of its input) one after the other, so
/// that two pipelines of inputs that never end would wait for each other
/// forever. The operators of the input are still part of the operator tree
/// (they are the children of the join), but not of the pipelines of the
/// join, nor of its metrics.
pub(crate) struct NestedInput {
    /// The output of the execution of the input, until it ended.
    stream: Option<PipelineStream>,
}

impl NestedInput {
    /// Execute `input`, in the execution of `ctx`: the execution starts when
    /// the input is first polled, and is cancelled when it is dropped.
    pub(crate) fn start(
        input: &Arc<dyn PhysicalOperator>,
        ctx: &ExecutionContext,
    ) -> Result<Self> {
        let stream = Executor::execute_stream(
            Arc::clone(input),
            Arc::clone(&ctx.task_ctx),
            ctx.num_threads.max(1),
        )?;
        Ok(Self {
            stream: Some(stream),
        })
    }

    /// The next batch of the input: `Ready(Ok(Some(batch)))`, `Ready(Ok(None))`
    /// once the input ended, or `Pending`, with the waker of the task of `ctx`
    /// registered with the input.
    pub(crate) fn poll_next(
        &mut self,
        ctx: &ExecutionContext,
    ) -> Poll<Result<Option<RecordBatch>>> {
        let Some(stream) = &mut self.stream else {
            return Poll::Ready(Ok(None));
        };
        let mut cx = Context::from_waker(ctx.waker());
        match stream.poll_next_unpin(&mut cx) {
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Ok(Some(batch))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Err(e)),
            Poll::Ready(None) => {
                self.stream = None;
                Poll::Ready(Ok(None))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    /// Whether the input ended.
    pub(crate) fn is_done(&self) -> bool {
        self.stream.is_none()
    }
}

impl fmt::Debug for NestedInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NestedInput")
            .field("done", &self.is_done())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MergeJoinSort
// ---------------------------------------------------------------------------

/// Sorts an input of a [`PhysicalMergeJoin`] that ends, like a
/// [`PhysicalSort`](crate::sort::PhysicalSort), and reserves the memory of
/// the rows that it keeps in the memory pool of the session.
///
/// * **Sink**: every thread of the pipelines of the input buffers the chunks
///   it pushes, and sorts them into runs once it buffered
///   [`SORT_RUN_ROWS`] rows (or about [`SORT_RUN_BYTES`] bytes), in
///   parallel. The buffered rows and the runs are reserved in the memory
///   pool before they are kept (the runs while the rows they copy are still
///   buffered): the sort fails with `ResourcesExhausted` when the pool
///   refuses them.
/// * **Source**: a single thread merges the runs of all threads (with a
///   [`RunMerger`]) and emits the rows in order. The memory of the runs stays
///   reserved until the source is dropped, once its pipeline completed.
#[derive(Debug)]
pub struct MergeJoinSort {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    keys: SortKeys,
}

impl MergeJoinSort {
    /// Sort the rows of `input` by `ordering`.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        ordering: LexOrdering,
    ) -> Result<Self> {
        let schema = input.schema();
        let keys = SortKeys::try_new(ordering, &schema)?;
        Ok(Self {
            children: vec![input],
            schema,
            keys,
        })
    }

    /// The sort keys.
    pub fn ordering(&self) -> &LexOrdering {
        self.keys.ordering()
    }

    /// Sort the rows buffered in `local` into runs, reserving their memory
    /// in `reservation` (and releasing the memory of the buffered rows).
    fn sort_buffered_rows(
        &self,
        local: &mut SortBuffer,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        if local.batches.is_empty() {
            return Ok(());
        }
        let batches = std::mem::take(&mut local.batches);
        // The working memory of the sort, reserved while it sorts: the keys
        // that it evaluates (if computed), their encoding and its sorted copy,
        // and the sorted copy of the rows (the run).
        let mut working = 0;
        for batch in &batches {
            let keys = self.keys.evaluate(batch)?;
            working += key_memory(&keys, batch)
                + keys
                    .iter()
                    .map(|key| encoded_bytes(key.as_ref()))
                    .sum::<usize>()
                + memory_size(batch);
        }
        let sorting = reservation.new_empty();
        sorting.try_grow(working)?;
        let runs = self.keys.sort(&self.schema, &batches, None)?;
        reservation.try_grow(runs.iter().map(run_memory_size).sum())?;
        drop(sorting);
        drop(batches);
        reservation.shrink(std::mem::take(&mut local.bytes));
        local.rows = 0;
        local.runs.extend(runs);
        Ok(())
    }
}

/// The bytes of a sorted run: its rows and its keys.
fn run_memory_size(run: &SortedRun) -> usize {
    let keys = match run.keys() {
        RunKeys::Rows(rows) => rows.size(),
        RunKeys::Column(column) => column.values().get_array_memory_size(),
    };
    memory_size(run.batch()) + keys
}

/// The global sink state of a [`MergeJoinSort`].
struct MergeJoinSortState {
    /// The sorted runs of the threads that completed.
    runs: Mutex<Vec<SortedRun>>,
    /// The merge of the runs, once the sink is finalized.
    merger: Mutex<Option<RunMerger>>,
    /// The memory of the buffered rows and of the runs.
    reservation: MemoryReservation,
}

impl GlobalSinkState for MergeJoinSortState {}

/// The rows of a thread of the input of a [`MergeJoinSort`].
#[derive(Default)]
struct SortBuffer {
    /// Compacted chunks that are not sorted yet.
    batches: Vec<RecordBatch>,
    /// The number of rows of `batches`.
    rows: usize,
    /// The memory size of `batches`, reserved.
    bytes: usize,
    /// The sorted runs of the thread.
    runs: Vec<SortedRun>,
}

impl LocalSinkState for SortBuffer {}

/// The global source state of a [`MergeJoinSort`]: the merge of its runs,
/// read by a single thread, and the reservation of their memory.
struct MergeJoinSortSource {
    merger: Mutex<RunMerger>,
    _reservation: MemoryReservation,
}

impl GlobalSourceState for MergeJoinSortSource {}

impl PhysicalOperator for MergeJoinSort {
    fn name(&self) -> &str {
        "MergeJoinSort"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!("MergeJoinSort: expr=[{}]", self.ordering())
    }

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
        let reservation =
            MemoryConsumer::new("MergeJoinSort").register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(MergeJoinSortState {
            runs: Mutex::new(vec![]),
            merger: Mutex::new(None),
            reservation,
        }))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(SortBuffer::default()))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<MergeJoinSortState>(gstate.as_any(), self.name())?;
        let local = downcast_state_mut::<SortBuffer>(lstate.as_any_mut(), self.name())?;
        let batch = input.compact()?;
        let bytes = memory_size(&batch);
        state.reservation.try_grow(bytes)?;
        local.rows += batch.num_rows();
        local.bytes += bytes;
        local.batches.push(batch);
        if local.rows >= SORT_RUN_ROWS || local.bytes >= SORT_RUN_BYTES {
            self.sort_buffered_rows(local, &state.reservation)?;
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let state = downcast_state::<MergeJoinSortState>(gstate.as_any(), self.name())?;
        let local = downcast_state_mut::<SortBuffer>(lstate.as_any_mut(), self.name())?;
        self.sort_buffered_rows(local, &state.reservation)?;
        state.runs.lock().append(&mut local.runs);
        Ok(())
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<MergeJoinSortState>(gstate.as_any(), self.name())?;
        let runs = std::mem::take(&mut *state.runs.lock());
        let merger = RunMerger::new(Arc::clone(&self.schema), runs, None);
        let empty = merger.remaining_rows() == 0;
        *state.merger.lock() = Some(merger);
        Ok(match empty {
            true => SinkFinalize::NoOutputPossible,
            false => SinkFinalize::Ready,
        })
    }

    fn is_source(&self) -> bool {
        true
    }

    /// A single thread emits the rows, in order.
    fn parallel_source(&self) -> bool {
        false
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let Some(sink) = ctx.global_sink_state(self) else {
            return internal_err!(
                "{} is read before it collected its input",
                self.name()
            );
        };
        let state = downcast_state::<MergeJoinSortState>(sink.as_any(), self.name())?;
        let Some(merger) = state.merger.lock().take() else {
            return internal_err!("{} is read before its sink is finalized", self.name());
        };
        // The memory of the runs is held by the merger from now on.
        let reservation = state.reservation.split(state.reservation.size());
        Ok(Arc::new(MergeJoinSortSource {
            merger: Mutex::new(merger),
            _reservation: reservation,
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<MergeJoinSortSource>(gstate.as_any(), self.name())?;
        Ok(match state.merger.lock().next_batch(DEFAULT_VECTOR_SIZE)? {
            Some(batch) => SourceResult::Chunk(DataChunk::from_record_batch(batch)),
            None => SourceResult::Finished,
        })
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
    /// The left and the right input, which emit their rows sorted by the
    /// join keys, on a single thread.
    children: Vec<Arc<dyn PhysicalOperator>>,
    spec: MergeJoinSpec,
    /// Whether the join reads its buffered input as a stream (an input that
    /// never ends, see [`NestedInput`]), as the streamed rows need its rows,
    /// rather than collecting all of its rows as a sink first.
    streams_buffered: bool,
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
    /// The bytes that gathering the rows of the left and of the right input
    /// takes, for the filter and the output.
    widths: [GatherWidths; 2],
    /// The components of the keys, in the order in which the row format
    /// compares them (see [`KeyComponent`]).
    components: Vec<KeyComponent>,
    /// The components with floats that other components follow (or that
    /// compare values of their own after their floats, as lists do): the rows
    /// of an input sorted by its keys (in which `-0.0` comes before `+0.0`)
    /// are not sorted by the keys of the join (in which `-0.0` is `+0.0`)
    /// where such a component is (or has) a zero (see [`ZeroRuns`]).
    zero_components: Vec<usize>,
    /// For every key, the order by which the inputs are sorted.
    key_options: Vec<SortOptions>,
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
        // The NULLs of the buffered input come first, so that a stream of it
        // tells with its first row whether it has a NULL key.
        if null_aware
            && (join_type != JoinType::LeftAnti
                || on.len() != 1
                || filter.is_some()
                || null_equality != NullEquality::NullEqualsNothing
                || !sort_options[0].nulls_first)
        {
            return not_impl_err!(
                "A null-aware PhysicalMergeJoin is a LeftAnti join of a single key sorted with NULLs first, without filter and with NullEqualsNothing"
            );
        }
        let mut key_fields = Vec::with_capacity(on.len());
        let mut components = vec![];
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
            KeyComponent::push(i, &left_type, vec![], &mut components);
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
        let zero_components = (0..components.len())
            .filter(|&index| {
                let followed = index + 1 < components.len();
                has_zero_order(&components[index].data_type, followed)
            })
            .collect();
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
            null_aware,
            streamed: SortMergeJoinExec::probe_side(&join_type),
            key_fields,
            widths,
            components,
            zero_components,
            key_options: sort_options.to_vec(),
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

    /// The bytes that the candidate pairs of the join gather of each row of
    /// `batch`, a batch of the streamed input (or of the buffered one if not
    /// `streamed`), for the filter (if any) and the output.
    fn gathered_bytes(&self, batch: &RecordBatch, streamed: bool) -> Vec<usize> {
        let side = match (self.streamed, streamed) {
            (JoinSide::Left, true) | (JoinSide::Right, false) => 0,
            _ => 1,
        };
        let widths = &self.widths[side];
        widths
            .variable_bytes(batch)
            .into_iter()
            .map(|variable| {
                let filter = match self.filter {
                    Some(_) => widths.row_bytes(variable, FILTER),
                    None => 0,
                };
                filter + widths.row_bytes(variable, OUTPUT)
            })
            .collect()
    }

    /// The bytes of a row of NULLs of the buffered input (or of the streamed
    /// one if not `buffered`) in the output: the other side of a row without a
    /// match.
    fn null_row_bytes(&self, buffered: bool) -> usize {
        let side = match (self.streamed, buffered) {
            (JoinSide::Left, false) | (JoinSide::Right, true) => 0,
            _ => 1,
        };
        self.widths[side].null_row_bytes(OUTPUT)
    }

    fn key_converter(&self) -> Result<RowConverter> {
        Ok(RowConverter::new(self.key_fields.clone())?)
    }

    /// No buffered rows yet, with the encoder of their keys.
    fn empty_buffered_rows(
        &self,
        schema: SchemaRef,
    ) -> Result<(RowConverter, BufferedRows)> {
        let converter = self.key_converter()?;
        let rows = BufferedRows {
            rows: StoredRows::new(schema),
            keys: converter.empty_rows(0, 0),
            has_null_key: false,
            matched: vec![],
            gathered: vec![0],
        };
        Ok((converter, rows))
    }
}

impl PhysicalMergeJoin {
    /// A merge join of `left` and `right` on the keys `on` (`(left key, right
    /// key)`), with the semantics of a `SortMergeJoinExec` with the same
    /// arguments, and its output schema and order. Both inputs must emit their
    /// rows sorted by their keys, with `sort_options`, on a single thread
    /// (e.g. [`MergeJoinSort`]s): the join fails if they do not.
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
            streams_buffered: false,
        })
    }

    /// Read the buffered input as a stream, as the streamed rows need its
    /// rows (e.g. an input that never ends), if `streams`: it is executed by an
    /// executor of its own, and the join keeps only the buffered rows that
    /// later streamed rows may still match (all of them for a `Full` join).
    pub fn with_buffered_stream(mut self, streams: bool) -> Self {
        self.streams_buffered = streams;
        self
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

    /// Whether the join reads its buffered input as a stream, see
    /// [`Self::with_buffered_stream`].
    pub fn streams_buffered_input(&self) -> bool {
        self.streams_buffered
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
    /// is finalized (for a join that collects them as a sink).
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

    /// The buffered rows of the execution of `ctx` (for a join that reads its
    /// buffered input as a stream): its global operator state.
    fn buffered_window(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<Mutex<BufferedWindow>>> {
        let state = ctx.registry().get_or_create_operator_state(self, ctx)?;
        let state = downcast_state::<StreamedBuffer>(state.as_any(), self.name())?;
        Ok(Arc::clone(&state.window))
    }

    /// Start joining the streamed rows of `input`.
    fn start_chunk(
        &self,
        progress: &MergeProgress,
        input: &DataChunk,
        reservation: &MemoryReservation,
    ) -> Result<StreamedChunk> {
        let batch = input.compact()?;
        // The computed keys are reserved while they are encoded, and the
        // encoded keys (with the other structures of the chunk) while the
        // chunk is joined.
        let arrays = evaluate_keys(self.spec.streamed_keys(), &batch)?;
        let chunk_reservation = reservation.new_empty();
        chunk_reservation.try_grow(key_memory(&arrays, &batch))?;
        let matchable = matchable_rows(&arrays, self.spec.null_equality);
        let zero_runs = ZeroRuns::of(&self.spec, &arrays)?;
        let keys = progress.converter.convert_columns(&arrays)?;
        drop(arrays);
        let gathered = self.spec.gathered_bytes(&batch, true);
        chunk_reservation.try_resize(
            keys.size()
                + zero_runs.as_ref().map_or(0, ZeroRuns::size)
                + gathered.capacity() * size_of::<usize>()
                + matchable.as_ref().map_or(0, |nulls| nulls.buffer().len()),
        )?;
        Ok(StreamedChunk {
            batch,
            keys,
            matchable,
            gathered,
            zero_runs,
            _reservation: chunk_reservation,
            next_row: 0,
            in_progress: None,
        })
    }

    /// The next output of the streamed rows of the current chunk of
    /// `progress`, joined with the buffered rows `buffered`, of at most
    /// [`DEFAULT_VECTOR_SIZE`] rows.
    ///
    /// The candidate pairs of a batch of streamed rows (with the buffered rows
    /// of their keys) are collected first, at most [`DEFAULT_VECTOR_SIZE`] of
    /// them (a streamed row without candidates counts as one) and about
    /// [`PART_BYTES`] of the rows that they gather for the filter and the
    /// output (fewer pairs of wide rows), whose bytes are reserved with
    /// `reservation` while they are gathered; then the filter is evaluated on
    /// all of them at once, and the rows are emitted in the order of the
    /// streamed rows. A streamed row whose candidates do not all fit is
    /// continued by the next call.
    ///
    /// `complete(key, row)` tells whether `buffered` has all the buffered rows
    /// that the streamed row `row`, with the key `key` (`None`: a key that
    /// cannot match), may match: a buffered input that is read as a stream may not have them yet.
    /// The batch stops before the first streamed row whose rows are not
    /// complete, and the output says so ([`MergeOutput::needs_buffered_rows`]).
    fn next_output(
        &self,
        progress: &mut MergeProgress,
        buffered: &BufferedRows,
        complete: &dyn Fn(Option<Row<'_>>, usize) -> bool,
        reservation: &MemoryReservation,
    ) -> Result<MergeOutput> {
        let spec = &self.spec;
        let join_type = spec.join_type();
        let MergeProgress {
            cursor,
            group,
            chunk,
            run,
            ..
        } = progress;
        let Some(chunk) = chunk.as_mut() else {
            return internal_err!("PhysicalMergeJoin has no streamed rows to join");
        };
        // The pairs are needed to evaluate the filter, or for the output.
        let enumerates_pairs = spec.filter.is_some() || spec.emits_pairs();
        // Whether a streamed row is decided by its first match (semi, anti and
        // mark joins).
        let decided_by_match = !spec.emits_pairs();

        let mut needs_buffered_rows = false;
        let mut pairs_streamed = Vec::new();
        let mut pairs_buffered = Vec::new();
        let mut parts = Vec::new();
        let mut cost = 0;
        // The bytes of the rows that the batch gathers.
        let mut bytes = 0;
        let null_buffered_bytes = spec.null_row_bytes(true);
        // The row of NULLs that the buffered rows of the streamed rows without a
        // match are taken from (see `StoredRows::gather`), reserved with the
        // rows (it does not bound them: a batch has a row at least).
        let null_template_bytes = match join_type {
            JoinType::Left | JoinType::Right | JoinType::Full => null_buffered_bytes,
            _ => 0,
        };
        while cost < DEFAULT_VECTOR_SIZE && bytes < PART_BYTES {
            let mut progress = match chunk.in_progress.take() {
                Some(progress) => progress,
                None => {
                    let row = chunk.next_row;
                    if row >= chunk.batch.num_rows() {
                        break;
                    }
                    let key = chunk.is_matchable(row).then(|| chunk.keys.row(row));
                    if !complete(key, row) {
                        needs_buffered_rows = true;
                        break;
                    }
                    chunk.next_row += 1;
                    let zero_run =
                        chunk.zero_runs.as_ref().and_then(|runs| runs.run(row));
                    let rows = match (key, zero_run) {
                        (Some(key), Some((position, prefix))) => {
                            let same = run.as_ref().is_some_and(|run| {
                                run.position == position && *run.prefix == *prefix
                            });
                            if !same {
                                // No buffered row of the keys of the rows of the
                                // run comes before the cursor (the keys of the run
                                // are not before those of the last streamed row).
                                *run = Some(StreamedRun {
                                    position,
                                    prefix: prefix.into(),
                                    start: *cursor,
                                });
                                *group = None;
                            }
                            let start = run.as_ref().map_or(*cursor, |run| run.start);
                            *cursor = (*cursor).min(start);
                            let first =
                                partition_point_from(&buffered.keys, start, |row| {
                                    row < key
                                });
                            let end =
                                partition_point_from(&buffered.keys, first, |row| {
                                    row <= key
                                });
                            first..end
                        }
                        (Some(key), None) => {
                            *run = None;
                            buffered.rows_of_key(cursor, group, key)?
                        }
                        (None, _) => 0..0,
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
            let row_bytes = chunk
                .gathered
                .get(progress.row)
                .copied()
                .unwrap_or_default();
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
                bytes += row_bytes + null_buffered_bytes + PAIR_INDEX_BYTES;
                continue;
            }
            // The pairs whose rows fit into the bytes of the batch, one at
            // least.
            let pairs_bytes = |n: usize| {
                n * (row_bytes + PAIR_INDEX_BYTES)
                    + buffered.gathered_bytes(progress.next..progress.next + n)
            };
            let mut n = remaining.min(DEFAULT_VECTOR_SIZE - cost);
            let budget = PART_BYTES.saturating_sub(bytes);
            if pairs_bytes(n) > budget {
                let (mut low, mut high) = (1, n);
                while low < high {
                    let mid = low + (high - low).div_ceil(2);
                    match pairs_bytes(mid) <= budget {
                        true => low = mid,
                        false => high = mid - 1,
                    }
                }
                n = low;
            }
            bytes += pairs_bytes(n);
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

        // The rows gathered for the filter and the output are reserved while
        // they are joined.
        let transient = reservation.new_empty();
        transient.try_grow(bytes + null_template_bytes)?;

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
            return Ok(MergeOutput {
                batch: None,
                needs_buffered_rows,
            });
        }
        let streamed_rows = UInt32Array::from(out_streamed);
        let streamed = JoinRows::Batch(&chunk.batch, &streamed_rows);
        let buffered_rows = JoinRows::Stored(&buffered.rows, &out_buffered);
        let (left, right) = spec.sides(streamed, buffered_rows);
        let marks = spec
            .output
            .is_mark_join()
            .then(|| Arc::new(BooleanArray::from(marks)) as ArrayRef);
        let batch = spec.output.batch(&left, &right, marks.as_ref(), num_rows)?;
        Ok(MergeOutput {
            batch: Some(batch),
            needs_buffered_rows,
        })
    }

    /// Join the streamed rows of the current chunk of `progress` with the
    /// buffered rows of a buffered input that the join reads as a stream (see
    /// [`Self::next_output`]): read the buffered rows as the streamed rows need
    /// them, waiting for the stream when it has none, or yielding after
    /// [`BUFFERED_BATCHES_PER_CALL`] batches of buffered rows without output.
    fn next_streaming_output(
        &self,
        ctx: &ExecutionContext,
        progress: &mut MergeProgress,
        window: &Mutex<BufferedWindow>,
        reservation: &MemoryReservation,
    ) -> Result<StreamingStep> {
        let mut window = window.lock();
        let mut batches = 0;
        loop {
            // A NULL buffered key (the NULLs come first) proves that a null-aware
            // join (`NOT IN`) has no rows at all: it ends at once, whatever the
            // rows of its inputs still to come.
            if self.spec.null_aware && window.rows.has_null_key {
                return Ok(StreamingStep::Finished);
            }
            // The rows that no later streamed row matches are dropped.
            if progress
                .chunk
                .as_ref()
                .is_none_or(|chunk| chunk.in_progress.is_none())
                && !self.spec.emits_unmatched_buffered_rows()
            {
                let dropped = window.drop_rows_before(progress.cursor)?;
                progress.rebase(dropped);
            }
            let output = {
                let window = &*window;
                let null_aware = self.spec.null_aware;
                // The streamed rows before the held run have all their
                // buffered rows, though the run did not end.
                let before = match &progress.chunk {
                    Some(chunk) => window.before_pending(chunk),
                    None => vec![],
                };
                self.next_output(
                    progress,
                    &window.rows,
                    &|key, row| {
                        window.is_complete_for(key, null_aware)
                            || before.get(row).copied().unwrap_or_default()
                    },
                    reservation,
                )?
            };
            if output.batch.is_some() || !output.needs_buffered_rows {
                return Ok(StreamingStep::Output(output.batch));
            }
            if batches == BUFFERED_BATCHES_PER_CALL {
                return Ok(StreamingStep::Yield);
            }
            if window.poll_more(ctx, &self.spec)?.is_pending() {
                return Ok(StreamingStep::Wait);
            }
            batches += 1;
        }
    }
}

/// What a merge join that reads its buffered input as a stream does next
/// (see [`PhysicalMergeJoin::next_streaming_output`]).
enum StreamingStep {
    /// Emit the output of the streamed rows joined so far (`None`: none).
    Output(Option<RecordBatch>),
    /// Wait for the buffered input, which has no rows yet: the waker of the
    /// task is registered with it.
    Wait,
    /// Let the other tasks run, after reading many buffered rows.
    Yield,
    /// The join has no more output: a null-aware join (`NOT IN`) of a
    /// buffered input with a NULL key.
    Finished,
}

/// The output of [`PhysicalMergeJoin::next_output`].
struct MergeOutput {
    batch: Option<RecordBatch>,
    /// Whether the next streamed row needs buffered rows that a buffered
    /// input read as a stream did not produce yet.
    needs_buffered_rows: bool,
}

/// The global sink state of a merge join that collects its buffered rows as
/// a sink.
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
    rows: BufferedRows,
}

/// The buffered rows of a merge join, in the order of their keys.
struct BufferedRows {
    rows: StoredRows,
    /// The keys of the rows, in the row format, in their order.
    keys: Rows,
    /// Whether a key is NULL (for the null-aware joins).
    has_null_key: bool,
    /// For `Full` joins: the rows that matched a streamed row, one bit per
    /// row.
    matched: Vec<AtomicU64>,
    /// The bytes that the candidate pairs gather of the rows before every
    /// position (from the first row, see [`MergeJoinSpec::gathered_bytes`]):
    /// those of the rows `a..b` are `gathered[b] - gathered[a]`.
    gathered: Vec<usize>,
}

impl BufferedRows {
    /// Append the buffered rows of `batch` (whose keys are `keys`, encoded by
    /// `converter`), which must not come before the rows so far. Returns the
    /// bytes of the rows and of their keys.
    fn append(
        &mut self,
        converter: &RowConverter,
        batch: RecordBatch,
        keys: &[ArrayRef],
        spec: &MergeJoinSpec,
    ) -> Result<usize> {
        let key_bytes = self.keys.size();
        let first = self.keys.num_rows();
        converter.append(&mut self.keys, keys)?;
        // The rows must arrive sorted by their keys.
        if (first.max(1)..self.keys.num_rows())
            .any(|row| self.keys.row(row) < self.keys.row(row - 1))
        {
            return internal_err!(
                "The buffered input of PhysicalMergeJoin is not sorted by the join keys"
            );
        }
        if spec.null_aware {
            self.has_null_key |= keys.iter().any(|key| key.logical_null_count() > 0);
        }
        let mut gathered = self.gathered.last().copied().unwrap_or_default();
        for row_bytes in spec.gathered_bytes(&batch, false) {
            gathered += row_bytes;
            self.gathered.push(gathered);
        }
        let mut bytes = memory_size(&batch)
            + self.keys.size().saturating_sub(key_bytes)
            + batch.num_rows() * size_of::<usize>();
        self.rows.push(batch);
        if spec.emits_unmatched_buffered_rows() {
            let words = self.rows.len().div_ceil(64);
            bytes += words.saturating_sub(self.matched.len()) * size_of::<AtomicU64>();
            self.matched.resize_with(words, || AtomicU64::new(0));
        }
        Ok(bytes)
    }

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

    /// The memory of the rows but for that of their batches: of their keys,
    /// of the bytes that the pairs gather of them, and of their matches, as
    /// allocated (with their capacities).
    fn shared_size(&self) -> usize {
        self.keys.size()
            + self.gathered.capacity() * size_of::<usize>()
            + self.matched.capacity() * size_of::<AtomicU64>()
    }

    /// The bytes that the candidate pairs gather of the rows `rows`.
    fn gathered_bytes(&self, rows: Range<usize>) -> usize {
        match (self.gathered.get(rows.start), self.gathered.get(rows.end)) {
            (Some(start), Some(end)) => end - start,
            _ => 0,
        }
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

/// The global operator state of a merge join that reads its buffered input
/// as a stream: the buffered rows read so far.
struct StreamedBuffer {
    window: Arc<Mutex<BufferedWindow>>,
}

impl GlobalOperatorState for StreamedBuffer {}

/// The buffered rows of a buffered input that the join reads as a stream:
/// those that the streamed rows still need, and the stream of the others.
struct BufferedWindow {
    input: NestedInput,
    rows: BufferedRows,
    converter: RowConverter,
    /// The key of the last buffered row read.
    last_key: Option<OwnedRow>,
    /// The memory of the window, exactly that of its rows and of the held run
    /// (see [`Self::memory`]).
    reservation: MemoryReservation,
    /// The memory of every batch of `rows`, and of all of them.
    batch_bytes: VecDeque<usize>,
    batches_bytes: usize,
    /// The rows of the run of signed zeros being read (see [`ZeroRuns`]):
    /// held back until a row after the run arrives, or the input ends, and
    /// then sorted by the keys of the join.
    pending: Option<PendingRun>,
}

/// The rows of a run of signed zeros of a buffered input read as a stream
/// (see [`ZeroRuns`]), sorted by the keys of the join once the run ends, or
/// as far as no row still to come of the run comes before them.
struct PendingRun {
    /// The position of the zero key of the run in
    /// [`MergeJoinSpec::zero_components`], and what comes before the zero of
    /// its rows (see [`ZeroRuns::run`]).
    position: usize,
    prefix: Box<[u8]>,
    /// The key of the first row of the run (with `-0.0` as `+0.0`), in the row
    /// format. The rows of the run and after it come after that row in the
    /// order of the input: each of them does not come before it once `-0.0`
    /// is `+0.0`, or differs from it first where one has `-0.0` and the other
    /// `+0.0`, so that it is of the run (see [`ZeroRuns`]). A streamed key
    /// before it, of another run or of none, equals none of them: it has all
    /// its buffered rows (see [`BufferedWindow::before_pending`]).
    first: OwnedRow,
    batches: Vec<RecordBatch>,
    /// The memory of `batches`.
    bytes: usize,
}

/// A join that reads its buffered input as a stream reads at most this many
/// batches of buffered rows without output in a call of `execute` before it
/// lets the other tasks run.
const BUFFERED_BATCHES_PER_CALL: usize = 16;

/// The join drops the buffered rows that no later streamed row matches once
/// there are this many of them, and at least as many as the rows it keeps.
const DROPPED_BUFFERED_ROWS: usize = 1 << 16;

impl BufferedWindow {
    /// Whether the buffered rows read so far are all the buffered rows that a
    /// streamed row with the key `key` (`None`: a key that cannot match) may
    /// match: the input ended, or a later key was read (the keys are
    /// sorted). A null-aware join (`NOT IN`) needs to know first whether the
    /// input is empty or has a NULL key (which come first).
    fn is_complete_for(&self, key: Option<Row<'_>>, null_aware: bool) -> bool {
        let ended = self.input.is_done();
        if null_aware && !ended && self.rows.rows.is_empty() {
            return false;
        }
        match key {
            None => true,
            Some(key) => {
                ended || self.last_key.as_ref().is_some_and(|last| last.row() > key)
            }
        }
    }

    /// For every streamed row of `chunk` (with `-0.0` as `+0.0`): whether it
    /// comes before the first row of the held run, of another run of signed
    /// zeros or of none, so that the buffered rows read so far are all those
    /// that it can match, though the held run did not end (see
    /// [`PendingRun::first`]). Empty without a held run.
    fn before_pending(&self, chunk: &StreamedChunk) -> Vec<bool> {
        let Some(pending) = &self.pending else {
            return vec![];
        };
        let run = Some((pending.position, &*pending.prefix));
        (0..chunk.keys.num_rows())
            .map(|row| {
                chunk.keys.row(row) < pending.first.row()
                    && chunk.zero_runs.as_ref().and_then(|runs| runs.run(row)) != run
            })
            .collect()
    }

    /// Read the next batch of buffered rows: `Ready(())` once a batch was read
    /// or the input ended, `Pending` while the input has nothing.
    fn poll_more(
        &mut self,
        ctx: &ExecutionContext,
        spec: &MergeJoinSpec,
    ) -> Result<Poll<()>> {
        let batch = match self.input.poll_next(ctx) {
            Poll::Ready(Ok(Some(batch))) => batch,
            Poll::Ready(Ok(None)) => {
                self.append_pending(spec)?;
                return Ok(Poll::Ready(()));
            }
            Poll::Ready(Err(e)) => return Err(e),
            Poll::Pending => return Ok(Poll::Pending),
        };
        if batch.num_rows() == 0 {
            return Ok(Poll::Ready(()));
        }
        let (keys, evaluated) = self.keys_of(spec, &batch)?;
        // A NULL key proves a null-aware join empty: the join learns of it now,
        // also if the row is held in a run.
        if spec.null_aware && keys.iter().any(|key| key.logical_null_count() > 0) {
            self.rows.has_null_key = true;
        }
        let Some(runs) = ZeroRuns::of(spec, &keys)? else {
            if self.pending.is_some() {
                self.append_pending(spec)?;
            }
            return self.append(batch, &keys, spec).map(|()| Poll::Ready(()));
        };
        // The runs are reserved with the keys (and their encoding, see
        // `key_memory`) while the rows of the batch are taken. The rows of runs
        // are held back, and sorted when their runs end.
        evaluated.try_grow(runs.size())?;
        let encoded = self.converter.convert_columns(&keys)?;
        let mut start = 0;
        let mut held = self.pending.is_some();
        for row in 0..batch.num_rows() {
            let run = runs.run(row);
            // A row of the held run, or a row of no run after rows of none.
            let same_rows = match (&self.pending, &run) {
                (Some(pending), Some((position, prefix))) if held => {
                    pending.position == *position && *pending.prefix == **prefix
                }
                (_, None) => !held,
                _ => false,
            };
            if same_rows {
                continue;
            }
            // The rows before `row` end the held run or are free rows.
            if row > start {
                self.take_rows(batch.slice(start, row - start), held, spec)?;
            }
            if held {
                self.append_pending(spec)?;
            }
            start = row;
            held = match run {
                Some((position, prefix)) => {
                    self.pending = Some(PendingRun {
                        position,
                        prefix: prefix.into(),
                        first: encoded.row(row).owned(),
                        batches: vec![],
                        bytes: 0,
                    });
                    true
                }
                None => false,
            };
        }
        self.take_rows(batch.slice(start, batch.num_rows() - start), held, spec)?;
        // The rows still to come do not come before the last row if it has no
        // zero of the sign that comes first in the order of its key (`-0.0` if
        // ascending, `+0.0` if descending), which a later row could have with
        // the other sign: they come after it in the order of the input, and
        // are equal or later once `-0.0` is `+0.0`. The held rows up to it are
        // joined now, so that a run of `+0.0` (if ascending) does not wait for
        // the rows after it.
        if held {
            let last = batch.slice(batch.num_rows() - 1, 1);
            let raw = evaluate_expressions_to_arrays(
                &spec.buffered_keys().collect::<Vec<_>>(),
                &last,
            )?;
            let flips = spec.zero_components.iter().any(|&index| {
                let component = &spec.components[index];
                let negative = !spec
                    .key_options
                    .get(component.key)
                    .is_some_and(|options| options.descending);
                zero_value_rows(component.array(&raw).as_ref(), Some(negative))
                    .first()
                    .copied()
                    .unwrap_or_default()
            });
            if !flips {
                let (keys, _evaluated) = self.keys_of(spec, &last)?;
                let bound = self.converter.convert_columns(&keys)?.row(0).owned();
                self.release_pending(bound.row(), spec)?;
            }
        }
        Ok(Poll::Ready(()))
    }

    /// Append the rows of the held run whose keys come before `bound` (keys
    /// of the join, in the row format) or are equal to it, sorted by the keys
    /// of the join, as no row still to come of the run comes before `bound`;
    /// the others stay held.
    fn release_pending(&mut self, bound: Row<'_>, spec: &MergeJoinSpec) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        let (position, prefix) = (pending.position, pending.prefix.clone());
        let first = pending.first.clone();
        let held = |batches: Vec<RecordBatch>, bytes: usize| PendingRun {
            position,
            prefix: prefix.clone(),
            first: first.clone(),
            batches,
            bytes,
        };
        let Some(first) = pending.batches.first() else {
            self.pending = Some(held(vec![], 0));
            return self.reserve(0);
        };
        let batch = concat_batches(&first.schema(), &pending.batches)?;
        // The rows of the run, and their copies, are reserved until they are
        // held or appended again.
        self.reserve(pending.bytes + memory_size(&batch))?;
        let (keys, _evaluated) = self.keys_of(spec, &batch)?;
        let rows = self.converter.convert_columns(&keys)?;
        let mut order = (0..batch.num_rows() as u32).collect::<Vec<_>>();
        order.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
        let released = order.partition_point(|&row| rows.row(row as usize) <= bound);
        let (now, later) = order.split_at(released);
        let later = take_record_batch(&batch, &UInt32Array::from(later.to_vec()))?;
        let now = take_record_batch(&batch, &UInt32Array::from(now.to_vec()))?;
        drop((pending, batch, rows));
        let bytes = memory_size(&later);
        let later = match later.num_rows() {
            0 => vec![],
            _ => vec![later],
        };
        self.pending = Some(held(later, bytes));
        match now.num_rows() {
            0 => self.reserve(0),
            _ => {
                let (keys, _evaluated) = self.keys_of(spec, &now)?;
                self.append(now, &keys, spec)
            }
        }
    }

    /// Append `batch`, rows of the buffered input after the rows so far, to the
    /// held run if `held`, to the rows otherwise.
    fn take_rows(
        &mut self,
        batch: RecordBatch,
        held: bool,
        spec: &MergeJoinSpec,
    ) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        match (held, &mut self.pending) {
            (true, Some(pending)) => {
                pending.bytes += memory_size(&batch);
                pending.batches.push(batch);
                self.reserve(0)
            }
            _ => {
                let (keys, _evaluated) = self.keys_of(spec, &batch)?;
                self.append(batch, &keys, spec)
            }
        }
    }

    /// Append the rows of the held run, if any, sorted by the keys of the join.
    fn append_pending(&mut self, spec: &MergeJoinSpec) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        let Some(first) = pending.batches.first() else {
            return self.reserve(0);
        };
        let batch = concat_batches(&first.schema(), &pending.batches)?;
        // The rows of the run, and their copies, are reserved until they are
        // appended.
        self.reserve(pending.bytes + memory_size(&batch))?;
        let (keys, _evaluated) = self.keys_of(spec, &batch)?;
        let rows = self.converter.convert_columns(&keys)?;
        let mut order = (0..batch.num_rows() as u32).collect::<Vec<_>>();
        order.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
        let sorted = take_record_batch(&batch, &UInt32Array::from(order))?;
        drop((pending, batch, rows));
        let (keys, _evaluated) = self.keys_of(spec, &sorted)?;
        self.append(sorted, &keys, spec)
    }

    /// Append `batch`, buffered rows whose keys are `keys`, to the rows.
    fn append(
        &mut self,
        batch: RecordBatch,
        keys: &[ArrayRef],
        spec: &MergeJoinSpec,
    ) -> Result<()> {
        let bytes = memory_size(&batch);
        self.rows.append(&self.converter, batch, keys, spec)?;
        self.batch_bytes.push_back(bytes);
        self.batches_bytes += bytes;
        self.reserve(0)?;
        let last = self.rows.keys.num_rows() - 1;
        self.last_key = Some(self.rows.keys.row(last).owned());
        Ok(())
    }

    /// The keys of the rows of `batch` (see [`evaluate_keys`]), and the
    /// reservation of their memory if they are computed and of their encoding
    /// (see [`key_memory`]), to hold while the keys are (until their encoding
    /// is reserved with the rows).
    fn keys_of(
        &self,
        spec: &MergeJoinSpec,
        batch: &RecordBatch,
    ) -> Result<(Vec<ArrayRef>, MemoryReservation)> {
        let keys = evaluate_keys(spec.buffered_keys(), batch)?;
        let reservation = self.reservation.new_empty();
        reservation.try_grow(key_memory(&keys, batch))?;
        Ok((keys, reservation))
    }

    /// The memory of the window: of the batches of its rows, of the structures
    /// of the rows (their keys, ..., see [`BufferedRows::shared_size`]), and
    /// of the held run.
    fn memory(&self) -> usize {
        self.batches_bytes
            + self.rows.shared_size()
            + self.pending.as_ref().map_or(0, |pending| {
                pending.bytes + pending.first.row().as_ref().len()
            })
    }

    /// Reserve the memory of the window (see [`Self::memory`]), with `extra`
    /// bytes more (of copies being made).
    fn reserve(&self, extra: usize) -> Result<()> {
        self.reservation.try_resize(self.memory() + extra)
    }

    /// Drop the buffered rows before `position` if there are enough of them
    /// (see [`DROPPED_BUFFERED_ROWS`]), in whole batches, releasing their
    /// memory: the number of rows dropped, by which the positions of the
    /// other rows decrease.
    fn drop_rows_before(&mut self, position: usize) -> Result<usize> {
        let starts = self.rows.rows.batch_starts();
        // The batches that end before `position`.
        let batches = starts
            .iter()
            .zip(
                starts
                    .iter()
                    .skip(1)
                    .chain(std::iter::once(&self.rows.rows.len())),
            )
            .take_while(|&(_, &end)| end <= position)
            .count();
        let dropped = starts.get(batches).copied().unwrap_or(self.rows.rows.len());
        if dropped < DROPPED_BUFFERED_ROWS || dropped * 2 < self.rows.rows.len() {
            return Ok(0);
        }
        let rows = self.rows.rows.without_first_batches(batches);
        // The keys of the rows that stay, in keys of the size that they take
        // (not of the capacity of the keys so far), reserved while both are
        // held.
        let retained = (dropped..self.rows.keys.num_rows())
            .map(|row| self.rows.keys.row(row).as_ref().len())
            .sum::<usize>();
        let rows_left = self.rows.keys.num_rows() - dropped;
        self.reserve(
            retained + (rows_left + 1) * size_of::<usize>() + size_of::<Rows>(),
        )?;
        let keys = retained_rows(&self.converter, &self.rows.keys, dropped, retained);
        self.rows.rows = rows;
        self.rows.keys = keys;
        let base = self.rows.gathered[dropped];
        self.rows.gathered = self.rows.gathered[dropped..]
            .iter()
            .map(|bytes| bytes - base)
            .collect();
        let released = self.batch_bytes.drain(..batches).sum::<usize>();
        self.batches_bytes -= released;
        self.reserve(0)?;
        debug_assert_eq!(self.reservation.size(), self.memory());
        Ok(dropped)
    }
}

/// The rows of `keys` from the row `from` on, whose data has `data` bytes, in
/// rows of the capacity that they take.
fn retained_rows(
    converter: &RowConverter,
    keys: &Rows,
    from: usize,
    data: usize,
) -> Rows {
    let mut retained = converter.empty_rows(keys.num_rows().saturating_sub(from), data);
    for row in from..keys.num_rows() {
        retained.push(keys.row(row));
    }
    retained
}

/// A component of the keys of a merge join, in the order in which the row
/// format compares them: a key, or, for a key that is a struct (also nested
/// in one), the validity of the struct and the components of its fields, in
/// their order.
#[derive(Debug, Clone)]
struct KeyComponent {
    /// The key that the component is part of.
    key: usize,
    /// The fields of the structs that lead from the key to the component.
    path: Vec<usize>,
    /// Whether the component is the validity of the struct at `path`, rather
    /// than its values.
    validity: bool,
    data_type: DataType,
}

impl KeyComponent {
    /// Append the components of `data_type`, of the key `key` at `path`.
    fn push(
        key: usize,
        data_type: &DataType,
        path: Vec<usize>,
        components: &mut Vec<Self>,
    ) {
        let DataType::Struct(fields) = data_type else {
            components.push(Self {
                key,
                path,
                validity: false,
                data_type: data_type.clone(),
            });
            return;
        };
        components.push(Self {
            key,
            path: path.clone(),
            validity: true,
            data_type: DataType::Boolean,
        });
        for (index, field) in fields.iter().enumerate() {
            let mut path = path.clone();
            path.push(index);
            Self::push(key, field.data_type(), path, components);
        }
    }

    /// The values of the component of the rows whose keys are `keys`: NULL
    /// where a struct above it is (its fields do not count then, as the row
    /// format compares them).
    fn array(&self, keys: &[ArrayRef]) -> ArrayRef {
        let mut array = Arc::clone(&keys[self.key]);
        let mut nulls = array.logical_nulls();
        for &field in &self.path {
            let child = Arc::clone(array.as_struct().column(field));
            nulls = NullBuffer::union(nulls.as_ref(), child.logical_nulls().as_ref());
            array = child;
        }
        match (self.validity, nulls) {
            (true, Some(nulls)) => Arc::new(BooleanArray::new(nulls.into_inner(), None)),
            (true, None) => Arc::new(BooleanArray::from(vec![true; array.len()])),
            (false, Some(nulls)) if !self.path.is_empty() => array
                .to_data()
                .into_builder()
                .nulls(Some(nulls))
                .build()
                .map_or(array, make_array),
            (false, _) => array,
        }
    }
}

/// Whether the values of `data_type`, a component of keys (see
/// [`KeyComponent`]), compare in another order when `-0.0` is `+0.0`: if they
/// are (or have) floats that other values follow, of other components if
/// `followed`, or of their own (the elements of lists after their first).
fn has_zero_order(data_type: &DataType, followed: bool) -> bool {
    match data_type {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => followed,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_float_leaf(field.data_type()),
        DataType::Dictionary(_, values) => has_zero_order(values, followed),
        DataType::RunEndEncoded(_, values) => {
            has_zero_order(values.data_type(), followed)
        }
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| has_zero_order(field.data_type(), followed)),
        data_type => has_float_leaf(data_type),
    }
}

/// The runs of signed zeros of a batch of rows of an input of a merge join:
/// the rows whose key of [`MergeJoinSpec::zero_components`] is (or has) a
/// zero, with what comes before the first zero of each of them.
///
/// An input sorted by its keys orders `-0.0` before `+0.0`, the join compares
/// its keys with `-0.0` as `+0.0`: where a float that other keys or values
/// follow is a zero, the rows (a *run*, e.g. `(-0.0, 5), (+0.0, 1)` of keys
/// `(a, b)` or of a key that is a struct of them, or `[1, -0.0, 5], [1, +0.0,
/// 1]` of a list key) come in the order of the signs of the zeros first, not
/// of the values after them. A run is the rows in a row that have their first
/// zero in the same place, with the same values before it: of the keys before
/// its key, and of its key before the zero (the elements of a list before
/// it, see [`zero_prefixes`]). Two rows that the join compares in another
/// order than the input orders them differ first where one has `-0.0` and
/// the other `+0.0`, so that they, and the rows between them in the order of
/// the input, are of one run: a row of another run ends a run. A buffered
/// input that the join reads as a stream sorts the rows of a run by the keys
/// of the join before it joins them (see [`BufferedWindow`]); the streamed
/// rows of a run keep their order (the order of the output), and find the
/// buffered rows of their keys from where the run started (none of them comes
/// before it).
struct ZeroRuns {
    /// For every row, its run: the position in
    /// [`MergeJoinSpec::zero_components`] of its first key that has a zero,
    /// and what comes before the zero (the keys before that key, in the row
    /// format, and the prefix of that key, see [`zero_prefixes`]).
    runs: Vec<Option<(usize, Box<[u8]>)>>,
}

impl ZeroRuns {
    /// The runs of the rows whose keys (with `-0.0` as `+0.0`) are `keys`,
    /// `None` if no row has a zero in a key of
    /// [`MergeJoinSpec::zero_components`].
    fn of(spec: &MergeJoinSpec, keys: &[ArrayRef]) -> Result<Option<Self>> {
        let Some(&last) = spec.zero_components.last() else {
            return Ok(None);
        };
        let components = spec.components[..=last]
            .iter()
            .map(|component| component.array(keys))
            .collect::<Vec<_>>();
        let rows = keys.first().map_or(0, |key| key.len());
        let mut runs: Vec<Option<(usize, Box<[u8]>)>> = vec![None; rows];
        for (position, &component) in spec.zero_components.iter().enumerate() {
            let prefixes = zero_prefixes(components[component].as_ref())?;
            let first = runs
                .iter()
                .zip(&prefixes)
                .any(|(run, prefix)| run.is_none() && prefix.is_some());
            if !first {
                continue;
            }
            // The components before the zero, as they compare (for equality
            // only).
            let before = match component {
                0 => None,
                _ => Some(encoded(&components[..component])?),
            };
            for (row, (run, prefix)) in runs.iter_mut().zip(prefixes).enumerate() {
                let Some(prefix) = prefix else {
                    continue;
                };
                if run.is_some() {
                    continue;
                }
                let mut identity = before
                    .as_ref()
                    .map_or_else(Vec::new, |before| before.row(row).as_ref().to_vec());
                identity.extend_from_slice(&prefix);
                *run = Some((position, identity.into()));
            }
        }
        Ok(runs.iter().any(Option::is_some).then_some(Self { runs }))
    }

    /// The memory of the runs.
    fn size(&self) -> usize {
        self.runs.capacity() * size_of::<Option<(usize, Box<[u8]>)>>()
            + self
                .runs
                .iter()
                .flatten()
                .map(|(_, prefix)| prefix.len())
                .sum::<usize>()
    }

    /// The run of `row`: the position of its first key with a zero in
    /// [`MergeJoinSpec::zero_components`], and what comes before the zero.
    fn run(&self, row: usize) -> Option<(usize, &[u8])> {
        let (position, prefix) = self.runs.get(row)?.as_ref()?;
        Some((*position, prefix))
    }
}

/// The values of `arrays` in the row format, as they compare (for equality).
fn encoded(arrays: &[ArrayRef]) -> Result<Rows> {
    let fields = arrays
        .iter()
        .map(|array| SortField::new(array.data_type().clone()))
        .collect();
    Ok(RowConverter::new(fields)?.convert_columns(arrays)?)
}

/// For every row of `array` (a component of keys, see [`KeyComponent`], with
/// `-0.0` as `+0.0`) that is a float zero, or has one nested in it: what comes
/// before its first zero in the order in which the row format compares it,
/// `None` for the other rows. Rows have the same prefix if they have their
/// first zero in the same place, with the same values before it: the field,
/// element or type of a union that has it (and the one in it that has it,
/// ...), and the values before it there (the fields of a struct before the
/// field, the elements of a list before the element). The values of NULL
/// structs and lists do not count. The layouts that [`zero_value_rows`] looks
/// into are looked into; any other with floats has the same prefix in every
/// row (one run of all its rows).
fn zero_prefixes(array: &dyn Array) -> Result<Vec<Option<Vec<u8>>>> {
    let rows = array.len();
    let prefixes = match array.data_type() {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            zero_value_rows(array, None)
                .into_iter()
                .map(|zero| zero.then(Vec::new))
                .collect()
        }
        DataType::Struct(_) => {
            let structs = array.as_struct();
            let mut prefixes = vec![None; rows];
            for (index, column) in structs.columns().iter().enumerate() {
                let fields = zero_prefixes(column.as_ref())?;
                if fields.iter().all(Option::is_none) {
                    continue;
                }
                // The fields before it, as they compare.
                let before = match index {
                    0 => None,
                    _ => Some(encoded(&structs.columns()[..index])?),
                };
                for (row, field) in fields.into_iter().enumerate() {
                    let Some(field) = field else {
                        continue;
                    };
                    if prefixes[row].is_some() || array.is_null(row) {
                        continue;
                    }
                    let mut prefix = (index as u64).to_be_bytes().to_vec();
                    if let Some(before) = &before {
                        prefix.extend_from_slice(before.row(row).as_ref());
                    }
                    prefix.extend_from_slice(&field);
                    prefixes[row] = Some(prefix);
                }
            }
            prefixes
        }
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Map(_, _) => {
            let (values, ranges) = list_ranges(array);
            let elements = zero_prefixes(values)?;
            if elements.iter().all(Option::is_none) {
                return Ok(vec![None; rows]);
            }
            // The elements, as they compare.
            let values = encoded(&[make_array(values.to_data())])?;
            ranges
                .into_iter()
                .enumerate()
                .map(|(row, (start, end))| {
                    if array.is_null(row) {
                        return None;
                    }
                    let (first, element) = (start..end).find_map(|element| {
                        let prefix = elements.get(element)?.as_ref()?;
                        Some((element, prefix))
                    })?;
                    let mut prefix = ((first - start) as u64).to_be_bytes().to_vec();
                    for element in start..first {
                        prefix.extend_from_slice(values.row(element).as_ref());
                    }
                    prefix.extend_from_slice(element);
                    Some(prefix)
                })
                .collect()
        }
        DataType::Dictionary(_, _) => {
            let dictionary = array.as_any_dictionary();
            let values = zero_prefixes(dictionary.values().as_ref())?;
            dictionary
                .normalized_keys()
                .into_iter()
                .enumerate()
                .map(|(row, key)| match array.is_valid(row) {
                    true => values.get(key).cloned().flatten(),
                    false => None,
                })
                .collect()
        }
        DataType::RunEndEncoded(run_ends, _) => {
            let (values, physical) = run_values(array, run_ends.data_type());
            let values = zero_prefixes(values.as_ref())?;
            physical
                .into_iter()
                .map(|index| values.get(index).cloned().flatten())
                .collect()
        }
        DataType::Union(fields, _) => {
            let union = array.as_union();
            let children = fields
                .iter()
                .map(|(type_id, _)| {
                    Ok((type_id, zero_prefixes(union.child(type_id).as_ref())?))
                })
                .collect::<Result<Vec<_>>>()?;
            (0..rows)
                .map(|row| {
                    let type_id = union.type_id(row);
                    let (_, prefixes) = children.iter().find(|(id, _)| *id == type_id)?;
                    let child = prefixes.get(union.value_offset(row))?.as_ref()?;
                    let mut prefix = type_id.to_be_bytes().to_vec();
                    prefix.extend_from_slice(child);
                    Some(prefix)
                })
                .collect()
        }
        data_type => vec![has_float_leaf(data_type).then(Vec::new); rows],
    };
    Ok(prefixes)
}

/// The values of a run-end encoded `array` whose run ends are of the type
/// `run_ends`, and the index of the value of every row.
fn run_values<'a>(
    array: &'a dyn Array,
    run_ends: &DataType,
) -> (&'a ArrayRef, Vec<usize>) {
    let rows = array.len();
    match run_ends {
        DataType::Int16 => {
            let run = array.as_run::<Int16Type>();
            let physical = (0..rows).map(|row| run.get_physical_index(row));
            (run.values(), physical.collect())
        }
        DataType::Int32 => {
            let run = array.as_run::<Int32Type>();
            let physical = (0..rows).map(|row| run.get_physical_index(row));
            (run.values(), physical.collect())
        }
        _ => {
            let run = array.as_run::<Int64Type>();
            let physical = (0..rows).map(|row| run.get_physical_index(row));
            (run.values(), physical.collect())
        }
    }
}

/// For every row of `array`, a key: whether it is a float zero, or has one
/// nested in it, of any sign if `sign` is `None` (a key with `-0.0` as
/// `+0.0`), of a negative sign if `Some(true)` (`-0.0`), of a positive sign
/// if `Some(false)` (`+0.0`, of a key as its input has it). The layouts that
/// the row format supports are looked into: structs, lists, list views,
/// fixed size lists, maps, dictionaries, run-end encoded values and unions;
/// any other with floats has a zero in every row (an upper bound).
fn zero_value_rows(array: &dyn Array, sign: Option<bool>) -> Vec<bool> {
    let rows = array.len();
    let leaf =
        |zero: bool, negative: bool| zero && sign.is_none_or(|sign| negative == sign);
    match array.data_type() {
        DataType::Float16 => array
            .as_primitive::<Float16Type>()
            .iter()
            .map(|value| {
                value.is_some_and(|v| leaf(v.to_f32() == 0.0, v.is_sign_negative()))
            })
            .collect(),
        DataType::Float32 => array
            .as_primitive::<Float32Type>()
            .iter()
            .map(|value| value.is_some_and(|v| leaf(v == 0.0, v.is_sign_negative())))
            .collect(),
        DataType::Float64 => array
            .as_primitive::<Float64Type>()
            .iter()
            .map(|value| value.is_some_and(|v| leaf(v == 0.0, v.is_sign_negative())))
            .collect(),
        DataType::Struct(_) => {
            let mut zeros = vec![false; rows];
            for column in array.as_struct().columns() {
                let column_zeros = zero_value_rows(column.as_ref(), sign);
                for (zero, column_zero) in zeros.iter_mut().zip(column_zeros) {
                    *zero |= column_zero;
                }
            }
            // The fields of a NULL struct do not count.
            for (row, zero) in zeros.iter_mut().enumerate() {
                *zero &= array.is_valid(row);
            }
            zeros
        }
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Map(_, _) => {
            let (values, ranges) = list_ranges(array);
            let values = zero_value_rows(values, sign);
            // The values of a NULL list do not count.
            ranges
                .into_iter()
                .enumerate()
                .map(|(row, (start, end))| {
                    array.is_valid(row)
                        && values
                            .get(start..end)
                            .is_some_and(|zeros| zeros.contains(&true))
                })
                .collect()
        }
        DataType::Dictionary(_, _) => {
            let dictionary = array.as_any_dictionary();
            let values = zero_value_rows(dictionary.values().as_ref(), sign);
            dictionary
                .normalized_keys()
                .into_iter()
                .enumerate()
                .map(|(row, key)| {
                    array.is_valid(row) && values.get(key).copied().unwrap_or_default()
                })
                .collect()
        }
        DataType::RunEndEncoded(run_ends, _) => {
            // The value of the run of every row.
            let (values, physical) = run_values(array, run_ends.data_type());
            let values = zero_value_rows(values.as_ref(), sign);
            physical
                .into_iter()
                .map(|index| values.get(index).copied().unwrap_or_default())
                .collect()
        }
        DataType::Union(fields, _) => {
            let union = array.as_union();
            let children = fields
                .iter()
                .map(|(type_id, _)| {
                    (
                        type_id,
                        zero_value_rows(union.child(type_id).as_ref(), sign),
                    )
                })
                .collect::<Vec<_>>();
            (0..rows)
                .map(|row| {
                    let type_id = union.type_id(row);
                    children
                        .iter()
                        .find(|(id, _)| *id == type_id)
                        .and_then(|(_, zeros)| {
                            zeros.get(union.value_offset(row)).copied()
                        })
                        .unwrap_or_default()
                })
                .collect()
        }
        data_type => vec![has_float_leaf(data_type); rows],
    }
}

/// The values of a list `array` (of type `List`, `LargeList`, `ListView`,
/// `LargeListView`, `FixedSizeList` or `Map`, whose values are its entries),
/// and the range of them of every row.
fn list_ranges(array: &dyn Array) -> (&dyn Array, Vec<(usize, usize)>) {
    let to_usize = |offset: i64| usize::try_from(offset).unwrap_or_default();
    match array.data_type() {
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let ranges = list
                .value_offsets()
                .windows(2)
                .map(|ends| (to_usize(ends[0].into()), to_usize(ends[1].into())))
                .collect();
            (list.values().as_ref(), ranges)
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let ranges = list
                .value_offsets()
                .windows(2)
                .map(|ends| (to_usize(ends[0]), to_usize(ends[1])))
                .collect();
            (list.values().as_ref(), ranges)
        }
        DataType::ListView(_) => {
            let list = array.as_list_view::<i32>();
            let ranges = list
                .value_offsets()
                .iter()
                .zip(list.value_sizes())
                .map(|(&offset, &size)| {
                    let start = to_usize(offset.into());
                    (start, start + to_usize(size.into()))
                })
                .collect();
            (list.values().as_ref(), ranges)
        }
        DataType::LargeListView(_) => {
            let list = array.as_list_view::<i64>();
            let ranges = list
                .value_offsets()
                .iter()
                .zip(list.value_sizes())
                .map(|(&offset, &size)| {
                    (to_usize(offset), to_usize(offset) + to_usize(size))
                })
                .collect();
            (list.values().as_ref(), ranges)
        }
        DataType::Map(_, _) => {
            let map = array.as_map();
            let ranges = map
                .value_offsets()
                .windows(2)
                .map(|ends| (to_usize(ends[0].into()), to_usize(ends[1].into())))
                .collect();
            (map.entries() as &dyn Array, ranges)
        }
        _ => {
            let list = array.as_fixed_size_list();
            let size = usize::try_from(list.value_length()).unwrap_or_default();
            let ranges = (0..array.len())
                .map(|row| (row * size, (row + 1) * size))
                .collect();
            (list.values().as_ref(), ranges)
        }
    }
}

/// The buffered rows of the key of the last streamed row.
struct KeyGroup {
    key: OwnedRow,
    rows: Range<usize>,
}

/// The buffered rows that the streamed rows are joined with.
enum Buffered {
    /// All buffered rows, collected by the join as a sink.
    Sorted(Arc<BufferedRows>),
    /// The buffered rows read so far from a buffered input that the join
    /// reads as a stream.
    Streamed(Arc<Mutex<BufferedWindow>>),
}

/// The state of the thread that streams the streamed input.
struct StreamState {
    buffered: Buffered,
    progress: MergeProgress,
    /// Reserves the rows that the join gathers for a batch of candidates.
    reservation: MemoryReservation,
}

impl OperatorState for StreamState {}

/// How far the streamed rows were joined.
struct MergeProgress {
    converter: RowConverter,
    /// The first buffered row whose key is not below the key of the last
    /// streamed row.
    cursor: usize,
    /// The buffered rows of the key of the last streamed row with a key that
    /// can match.
    group: Option<KeyGroup>,
    /// The streamed rows being joined.
    chunk: Option<StreamedChunk>,
    /// The run of signed zeros of the last streamed row, if any (see
    /// [`ZeroRuns`]).
    run: Option<StreamedRun>,
}

/// A run of signed zeros of streamed rows (see [`ZeroRuns`]): its keys are
/// not sorted by the keys of the join, so that the streamed rows of the run
/// find the buffered rows of their keys from where the cursor was when the run
/// started, where the cursor stays.
struct StreamedRun {
    /// The position of the zero key of the run in
    /// [`MergeJoinSpec::zero_components`], and what comes before the zero of
    /// its rows (see [`ZeroRuns::run`]).
    position: usize,
    prefix: Box<[u8]>,
    /// The cursor when the run started: no buffered row of the keys of the
    /// run comes before it.
    start: usize,
}

impl MergeProgress {
    /// The positions of the buffered rows decreased by `dropped`, the number
    /// of buffered rows that were dropped before them.
    fn rebase(&mut self, dropped: usize) {
        if dropped == 0 {
            return;
        }
        self.cursor -= dropped;
        if let Some(group) = &mut self.group {
            group.rows = group.rows.start - dropped..group.rows.end - dropped;
        }
        if let Some(run) = &mut self.run {
            run.start -= dropped;
        }
    }
}

/// A chunk of streamed rows being joined.
struct StreamedChunk {
    batch: RecordBatch,
    keys: Rows,
    /// The rows whose keys can match, `None` if all.
    matchable: Option<NullBuffer>,
    /// The bytes that the candidate pairs gather of every row (see
    /// [`MergeJoinSpec::gathered_bytes`]).
    gathered: Vec<usize>,
    /// The runs of signed zeros of the rows, if any.
    zero_runs: Option<ZeroRuns>,
    /// The memory of `keys`, `gathered` and `zero_runs`, held while the
    /// chunk is joined.
    _reservation: MemoryReservation,
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
enum UnmatchedScanState {
    /// All buffered rows, collected by the join as a sink, scanned in
    /// parallel: the next position to scan.
    Sorted {
        buffered: Arc<BufferedRows>,
        next: Mutex<usize>,
        reservation: MemoryReservation,
    },
    /// The buffered rows of a buffered input that the join reads as a stream,
    /// all of which (until the input ends) are scanned by a single thread.
    Streamed {
        window: Arc<Mutex<BufferedWindow>>,
        next: Mutex<usize>,
        reservation: MemoryReservation,
    },
}

impl GlobalSourceState for UnmatchedScanState {
    fn max_threads(&self) -> usize {
        match self {
            Self::Sorted { buffered, .. } => {
                buffered.rows.len().div_ceil(DEFAULT_VECTOR_SIZE).max(1)
            }
            Self::Streamed { .. } => 1,
        }
    }
}

impl PhysicalMergeJoin {
    /// The next chunk of the buffered rows without a match of `buffered`,
    /// from the position `next` on, `None` once all were scanned: of the rows
    /// of at most [`DEFAULT_VECTOR_SIZE`] positions and about [`PART_BYTES`]
    /// of their bytes, which are reserved with `reservation` while they are
    /// gathered.
    fn next_unmatched_rows(
        &self,
        buffered: &BufferedRows,
        next: &Mutex<usize>,
        reservation: &MemoryReservation,
    ) -> Result<Option<RecordBatch>> {
        let num_rows = buffered.rows.len();
        let null_streamed_bytes = self.spec.null_row_bytes(false);
        let rows_bytes = |rows: Range<usize>| {
            buffered.gathered_bytes(rows.clone())
                + rows.len() * (null_streamed_bytes + size_of::<Option<usize>>())
        };
        loop {
            let (start, end) = {
                let mut next = next.lock();
                let start = *next;
                if start >= num_rows {
                    return Ok(None);
                }
                // The rows that fit into the bytes of a chunk, one at least.
                let (mut low, mut high) =
                    (start + 1, (start + DEFAULT_VECTOR_SIZE).min(num_rows));
                while low < high {
                    let mid = low + (high - low).div_ceil(2);
                    match rows_bytes(start..mid) <= PART_BYTES {
                        true => low = mid,
                        false => high = mid - 1,
                    }
                }
                *next = low;
                (start, low)
            };
            let positions = (start..end)
                .filter(|&position| !buffered.is_matched(position))
                .map(Some)
                .collect::<Vec<_>>();
            if positions.is_empty() {
                continue;
            }
            let transient = reservation.new_empty();
            transient.try_grow(rows_bytes(start..end))?;
            let streamed_schema = self.streamed_input().schema();
            let (left, right) = self.spec.sides(
                JoinRows::Nulls(&streamed_schema, positions.len()),
                JoinRows::Stored(&buffered.rows, &positions),
            );
            let batch = self
                .spec
                .output
                .batch(&left, &right, None, positions.len())?;
            return Ok(Some(batch));
        }
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
        if self.streams_buffered {
            description.push_str(", buffered_stream=true");
        }
        description
    }

    // ---------------------------------------------------------------------
    // Sink: the buffered input (unless it is read as a stream)
    // ---------------------------------------------------------------------

    fn is_sink(&self) -> bool {
        !self.streams_buffered
    }

    /// The buffered rows arrive in order, from a single thread.
    fn parallel_sink(&self) -> bool {
        false
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let (converter, rows) = self
            .spec
            .empty_buffered_rows(self.buffered_input().schema())?;
        let reservation =
            MemoryConsumer::new("PhysicalMergeJoin").register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(MergeJoinState {
            building: Mutex::new(Some(BufferedBuild { converter, rows })),
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
        // The computed keys are reserved while they are encoded, and their
        // encoding until it is reserved with the rows.
        let evaluated = state.reservation.new_empty();
        evaluated.try_grow(key_memory(&keys, &batch))?;
        let mut building = state.building.lock();
        let Some(build) = building.as_mut() else {
            return internal_err!(
                "PhysicalMergeJoin received buffered rows after its build"
            );
        };
        let bytes = build
            .rows
            .append(&build.converter, batch, &keys, &self.spec)?;
        state.reservation.try_grow(bytes)?;
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
        let rows = build.rows;
        let empty = rows.rows.is_empty();
        let has_null_key = rows.has_null_key;
        if state.rows.set(Arc::new(rows)).is_err() {
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

    /// For a join that reads its buffered input as a stream: the stream, and
    /// the buffered rows read so far.
    fn get_global_operator_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalOperatorState>> {
        if !self.streams_buffered {
            return Ok(Arc::new(EmptyState));
        }
        let (converter, rows) = self
            .spec
            .empty_buffered_rows(self.buffered_input().schema())?;
        let reservation =
            MemoryConsumer::new("PhysicalMergeJoin").register(ctx.task_ctx.memory_pool());
        let window = BufferedWindow {
            input: NestedInput::start(self.buffered_input(), ctx)?,
            rows,
            converter,
            last_key: None,
            reservation,
            batch_bytes: VecDeque::new(),
            batches_bytes: 0,
            pending: None,
        };
        Ok(Arc::new(StreamedBuffer {
            window: Arc::new(Mutex::new(window)),
        }))
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        let buffered = match self.streams_buffered {
            false => Buffered::Sorted(self.buffered_rows(ctx)?),
            true => {
                let state =
                    downcast_state::<StreamedBuffer>(gstate.as_any(), self.name())?;
                Buffered::Streamed(Arc::clone(&state.window))
            }
        };
        let reservation =
            MemoryConsumer::new("PhysicalMergeJoin").register(ctx.task_ctx.memory_pool());
        Ok(Box::new(StreamState {
            buffered,
            reservation,
            progress: MergeProgress {
                converter: self.spec.key_converter()?,
                cursor: 0,
                group: None,
                chunk: None,
                run: None,
            },
        }))
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<StreamState>(lstate.as_any_mut(), self.name())?;
        if state.progress.chunk.is_none() {
            state.progress.chunk =
                Some(self.start_chunk(&state.progress, input, &state.reservation)?);
        }
        let batch = match &state.buffered {
            Buffered::Sorted(buffered) => {
                let buffered = Arc::clone(buffered);
                self.next_output(
                    &mut state.progress,
                    &buffered,
                    &|_, _| true,
                    &state.reservation,
                )?
                .batch
            }
            Buffered::Streamed(window) => {
                let window = Arc::clone(window);
                match self.next_streaming_output(
                    ctx,
                    &mut state.progress,
                    &window,
                    &state.reservation,
                )? {
                    StreamingStep::Output(batch) => batch,
                    StreamingStep::Yield => {
                        return Ok(OperatorResult::Blocked(TaskInterrupt::yield_now()));
                    }
                    StreamingStep::Finished => {
                        state.progress.chunk = None;
                        return Ok(OperatorResult::Finished);
                    }
                    StreamingStep::Wait => {
                        return Ok(OperatorResult::Blocked(
                            TaskInterrupt::waker_registered(),
                        ));
                    }
                }
            }
        };
        if let Some(batch) = batch {
            *output = DataChunk::from_record_batch(batch);
        }
        if state
            .progress
            .chunk
            .as_ref()
            .is_some_and(StreamedChunk::is_done)
        {
            state.progress.chunk = None;
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

    /// The rows of a buffered input read as a stream are read by a single
    /// thread.
    fn parallel_source(&self) -> bool {
        !self.streams_buffered
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let reservation =
            MemoryConsumer::new("PhysicalMergeJoin").register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(match self.streams_buffered {
            false => UnmatchedScanState::Sorted {
                buffered: self.buffered_rows(ctx)?,
                next: Mutex::new(0),
                reservation,
            },
            true => UnmatchedScanState::Streamed {
                window: self.buffered_window(ctx)?,
                next: Mutex::new(0),
                reservation,
            },
        }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<UnmatchedScanState>(gstate.as_any(), self.name())?;
        let batch = match state {
            UnmatchedScanState::Sorted {
                buffered,
                next,
                reservation,
            } => self.next_unmatched_rows(buffered, next, reservation)?,
            UnmatchedScanState::Streamed {
                window,
                next,
                reservation,
            } => {
                let mut window = window.lock();
                // The rows of the input that come after the last streamed row
                // match none: all of them are read first.
                while !window.input.is_done() {
                    if window.poll_more(ctx, &self.spec)?.is_pending() {
                        return Ok(SourceResult::Blocked(
                            TaskInterrupt::waker_registered(),
                        ));
                    }
                }
                self.next_unmatched_rows(&window.rows, next, reservation)?
            }
        };
        Ok(match batch {
            Some(batch) => SourceResult::Chunk(DataChunk::from_record_batch(batch)),
            None => SourceResult::Finished,
        })
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
        // The buffered input sinks into the join, in a child meta pipeline,
        // unless the join reads it as a stream (in an execution of its own).
        if !self.streams_buffered {
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

    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use arrow::array::{
        Decimal128Array, FixedSizeListArray, Float64Array, Float64Builder, Int32Array,
        Int32Builder, Int64Array, LargeListArray, LargeListViewArray, ListArray,
        ListViewArray, MapBuilder, RunArray, StringArray, StringViewArray, StructArray,
        TimestampNanosecondArray,
    };
    use arrow::compute::{SortColumn, lexsort_to_indices};
    use arrow::datatypes::{Field, Fields, TimeUnit};
    use datafusion::catalog::streaming::StreamingTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::SendableRecordBatchStream;
    use datafusion_execution::TaskContext;
    use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, col};
    use datafusion_physical_plan::Partitioning;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
    use datafusion_physical_plan::limit::GlobalLimitExec;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};

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

    /// The schema of the rows `(id, k)` of the `side` input of the joins of
    /// [`TestSource`]s.
    fn key_schema(side: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(format!("{side}id"), DataType::Int64, false),
            Field::new(format!("{side}k"), DataType::Int32, false),
        ]))
    }

    /// The rows `(id, k)` of `rows`, in chunks of 1,000 rows.
    fn key_chunks(side: &str, rows: &[(i64, i32)]) -> Result<Vec<DataChunk>> {
        rows.chunks(1000)
            .map(|rows| {
                let batch = RecordBatch::try_new(
                    key_schema(side),
                    vec![
                        Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
                        Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.1))),
                    ],
                )?;
                Ok(DataChunk::from_record_batch(batch))
            })
            .collect()
    }

    /// A sort-merge join of `left` and `right` (two inputs of [`TestSource`]s
    /// with the rows of `(id, k)`, sorted by [`MergeJoinSort`]s) on `k`.
    fn merge_join_of(
        left: Vec<(i64, i32)>,
        right: Vec<(i64, i32)>,
        join_type: JoinType,
        filter: Option<JoinFilter>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let source = |side: &str,
                      rows: Vec<(i64, i32)>|
         -> Result<Arc<dyn PhysicalOperator>> {
            let schema = key_schema(side);
            let chunks = key_chunks(side, &rows)?;
            let source = Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks));
            let ordering = LexOrdering::new([PhysicalSortExpr::new_default(col(
                &format!("{side}k"),
                &schema,
            )?)])
            .unwrap();
            Ok(Arc::new(MergeJoinSort::try_new(source, ordering)?))
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

    /// A task context whose memory pool has `limit` bytes.
    pub(crate) fn limited_task_ctx(limit: usize) -> Result<Arc<TaskContext>> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(limit, 1.0)
            .build_arc()?;
        Ok(Arc::new(TaskContext::default().with_runtime(runtime)))
    }

    /// A merge join of `left` and `right` on `k` like [`merge_join_of`], of
    /// inputs that emit their rows sorted (without sorting them).
    fn merge_join_of_sorted(
        left: Vec<(i64, i32)>,
        right: Vec<(i64, i32)>,
        join_type: JoinType,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let source =
            |side: &str, rows: Vec<(i64, i32)>| -> Result<Arc<dyn PhysicalOperator>> {
                let chunks = key_chunks(side, &rows)?;
                Ok(Arc::new(TestSource::with_schema(key_schema(side), chunks)))
            };
        let (left, right) = (source("l", left)?, source("r", right)?);
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        Ok(Arc::new(PhysicalMergeJoin::try_new(
            left,
            right,
            on,
            None,
            join_type,
            &[SortOptions::default()],
            NullEquality::NullEqualsNothing,
            false,
            None,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_of_merge_joins_are_reserved_in_the_memory_pool() -> Result<()> {
        let rows = |n: i64| (0..n).map(|i| (i, (i % 10) as i32)).collect::<Vec<_>>();
        let sorted = |n: i64| {
            let mut rows = rows(n);
            rows.sort_by_key(|row| row.1);
            rows
        };
        // The buffered rows of the join (of inputs that are sorted already).
        let join =
            || merge_join_of_sorted(sorted(100), sorted(20_000), JoinType::LeftSemi);
        // The rows of the sorts of the inputs: many streamed rows, and a
        // single buffered row.
        let sorts = || merge_join_of(rows(50_000), rows(1), JoinType::Inner, None);
        for (join, consumer, output) in [
            (join()?, "PhysicalMergeJoin", 100),
            (sorts()?, "MergeJoinSort", 5_000),
        ] {
            let err = with_timeout(Executor::collect(
                Arc::clone(&join),
                limited_task_ctx(64 * 1024)?,
                4,
            ))
            .await
            .unwrap_err();
            assert!(
                matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
                "{err}"
            );
            assert!(err.to_string().contains(consumer), "{err}");
            let batches =
                with_timeout(Executor::collect(join, limited_task_ctx(64 << 20)?, 4))
                    .await?;
            assert_eq!(num_rows(&batches), output, "{consumer}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wide_rows_are_reserved_and_merged_in_fewer_pairs() -> Result<()> {
        // Sorted inputs whose rows have strings of 1 KiB: the merge join
        // reserves the bytes of the rows that it gathers for a batch of pairs
        // (or of rows without a match) before it gathers them, in batches of
        // about a MiB of rows, not of 2,048 pairs.
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Int32, false),
                Field::new(format!("{side}s"), DataType::Utf8, false),
            ])) as SchemaRef
        };
        let source =
            |side: &str, rows: i64, key: i32| -> Result<Arc<dyn PhysicalOperator>> {
                let batch = RecordBatch::try_new(
                    schema(side),
                    vec![
                        Arc::new(Int64Array::from_iter_values(0..rows)),
                        Arc::new(Int32Array::from(vec![key; rows as usize])),
                        Arc::new(StringArray::from_iter_values(
                            (0..rows).map(|id| format!("{id:>1024}")),
                        )),
                    ],
                )?;
                let chunks = (0..batch.num_rows())
                    .step_by(100)
                    .map(|offset| {
                        DataChunk::from_record_batch(
                            batch.slice(offset, 100.min(batch.num_rows() - offset)),
                        )
                    })
                    .collect();
                Ok(Arc::new(TestSource::with_schema(schema(side), chunks)))
            };
        let join = |right_rows: i64,
                    right_key: i32,
                    join_type: JoinType|
         -> Result<Arc<dyn PhysicalOperator>> {
            let (left, right) =
                (source("l", 100, 1)?, source("r", right_rows, right_key)?);
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            Ok(Arc::new(PhysicalMergeJoin::try_new(
                left,
                right,
                on,
                None,
                join_type,
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
                false,
                None,
            )?))
        };
        // 10,000 pairs of 2 KiB of strings: a pool of 512 KiB holds the buffered
        // rows, but not a batch of pairs; one of 4 MiB holds both.
        let err = with_timeout(Executor::collect(
            join(100, 1, JoinType::Inner)?,
            limited_task_ctx(512 * 1024)?,
            4,
        ))
        .await
        .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        assert!(err.to_string().contains("PhysicalMergeJoin"), "{err}");
        let batches = with_timeout(Executor::collect(
            join(100, 1, JoinType::Inner)?,
            limited_task_ctx(4 << 20)?,
            4,
        ))
        .await?;
        assert_eq!(num_rows(&batches), 10_000);
        assert!(
            batches
                .iter()
                .all(|batch| batch.num_rows() <= PART_BYTES / 2048 + 1)
        );
        // The 2,000 buffered rows of a `Full` join match no streamed row: they
        // are emitted in chunks of about a MiB, too.
        let batches = with_timeout(Executor::collect(
            join(2_000, 2, JoinType::Full)?,
            limited_task_ctx(8 << 20)?,
            4,
        ))
        .await?;
        assert_eq!(num_rows(&batches), 2_100);
        assert!(
            batches
                .iter()
                .all(|batch| batch.num_rows() <= PART_BYTES / 1024 + 1)
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn skewed_lists_are_reserved_by_the_row() -> Result<()> {
        // As the symmetric join's: the first of the 100 rows of either input
        // has a list of 1 MiB, and all rows match the 100 rows of the other
        // input. The merge join reserves the bytes of every row that it
        // gathers, in batches with one pair with a big list at most: a pool of
        // 768 KiB does not hold a batch with one, one of 6 MiB does.
        use crate::join::symmetric_hash_join::tests::{big_lists, skewed_list_batch};
        let join = || -> Result<Arc<dyn PhysicalOperator>> {
            let source = |side: &str| -> Arc<dyn PhysicalOperator> {
                let batch = skewed_list_batch(side);
                Arc::new(TestSource::with_schema(
                    batch.schema(),
                    vec![DataChunk::from_record_batch(batch)],
                ))
            };
            let (left, right) = (source("l"), source("r"));
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            Ok(Arc::new(PhysicalMergeJoin::try_new(
                left,
                right,
                on,
                None,
                JoinType::Inner,
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
                false,
                None,
            )?))
        };
        let err = with_timeout(Executor::collect(join()?, limited_task_ctx(3 << 18)?, 4))
            .await
            .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let batches =
            with_timeout(Executor::collect(join()?, limited_task_ctx(6 << 20)?, 4))
                .await?;
        assert_eq!(num_rows(&batches), 10_000);
        let big = big_lists(&batches);
        assert_eq!(big.iter().sum::<usize>(), 200);
        assert!(big.iter().all(|&big| big <= 2), "{big:?}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sparse_unions_are_reserved_with_all_their_children() -> Result<()> {
        // As the symmetric join's: the 100 x 100 pairs of rows of a key whose
        // sparse unions are integers, of a union whose other child has strings
        // of 1 KiB, gather the strings of both children, 2 KiB a pair, which the
        // merge join reserves, in batches of about a MiB of rows.
        use crate::join::symmetric_hash_join::tests::sparse_union_batch;
        let join = || -> Result<Arc<dyn PhysicalOperator>> {
            let source = |side: &str| -> Arc<dyn PhysicalOperator> {
                let batch = sparse_union_batch(side);
                let chunks = (0..4)
                    .map(|chunk| {
                        DataChunk::from_record_batch(batch.slice(chunk * 25, 25))
                    })
                    .collect();
                Arc::new(TestSource::with_schema(batch.schema(), chunks))
            };
            let (left, right) = (source("l"), source("r"));
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            Ok(Arc::new(PhysicalMergeJoin::try_new(
                left,
                right,
                on,
                None,
                JoinType::Inner,
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
                false,
                None,
            )?))
        };
        let err =
            with_timeout(Executor::collect(join()?, limited_task_ctx(512 * 1024)?, 4))
                .await
                .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let batches =
            with_timeout(Executor::collect(join()?, limited_task_ctx(4 << 20)?, 4))
                .await?;
        assert_eq!(num_rows(&batches), 10_000);
        assert!(
            batches
                .iter()
                .all(|batch| batch.num_rows() <= PART_BYTES / 2048 + 1)
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn nulls_of_outer_joins_are_reserved() -> Result<()> {
        // As the symmetric join's: the left row of a `Left` join of an empty
        // right input has NULLs of a fixed size list of a million integers
        // (8 MB), which the merge join reserves before it builds them.
        use crate::join::symmetric_hash_join::tests::{big_null_schema, one_left_row};
        let join = || -> Result<Arc<dyn PhysicalOperator>> {
            let left = one_left_row();
            let left: Arc<dyn PhysicalOperator> = Arc::new(TestSource::with_schema(
                left.schema(),
                vec![DataChunk::from_record_batch(left)],
            ));
            let right: Arc<dyn PhysicalOperator> =
                Arc::new(TestSource::with_schema(big_null_schema(), vec![]));
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            Ok(Arc::new(PhysicalMergeJoin::try_new(
                left,
                right,
                on,
                None,
                JoinType::Left,
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
                false,
                None,
            )?))
        };
        let err =
            with_timeout(Executor::collect(join()?, limited_task_ctx(64 * 1024)?, 4))
                .await
                .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let batches =
            with_timeout(Executor::collect(join()?, limited_task_ctx(64 << 20)?, 4))
                .await?;
        assert_eq!(num_rows(&batches), 1);
        Ok(())
    }

    #[test]
    fn retained_keys_take_the_memory_of_their_rows() -> Result<()> {
        // The keys of the buffered rows that a merge join keeps when it drops
        // the rows before them are copied into keys of their size: dropping
        // rows again and again keeps the memory of the keys bounded, rather
        // than growing with the capacity of the keys before them (with which
        // 65,536 integer keys grew from 0.85 MB to 13.76 MB in 12 rounds).
        let strings = |round: usize, rows: usize| -> ArrayRef {
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|i| format!("{round:04}{i:060}")),
            ))
        };
        let integers = |round: usize, rows: usize| -> ArrayRef {
            Arc::new(Int32Array::from_iter_values(
                (0..rows).map(|i| (round * rows + i) as i32),
            ))
        };
        type Keys = fn(usize, usize) -> ArrayRef;
        let cases: [(DataType, Keys, usize, usize); 2] = [
            (DataType::Utf8, strings, 1000, 50),
            (DataType::Int32, integers, 65_536, 12),
        ];
        for (data_type, keys_of, kept, rounds) in cases {
            let converter = RowConverter::new(vec![SortField::new(data_type.clone())])?;
            let mut keys = converter.empty_rows(0, 0);
            let mut sizes = vec![];
            for round in 0..rounds {
                converter.append(&mut keys, &[keys_of(round, kept)])?;
                let from = keys.num_rows() - kept;
                let data = (from..keys.num_rows())
                    .map(|row| keys.row(row).as_ref().len())
                    .sum::<usize>();
                keys = retained_rows(&converter, &keys, from, data);
                assert_eq!(keys.num_rows(), kept);
                // The data of the rows, their offsets, and the rows.
                let size = data + (kept + 1) * size_of::<usize>() + size_of::<Rows>();
                assert!(
                    keys.size() <= size,
                    "{data_type} {round}: {} for {data} bytes",
                    keys.size()
                );
                sizes.push(keys.size());
            }
            assert!(
                sizes.iter().all(|&size| size == sizes[0]),
                "{data_type}: {sizes:?}"
            );
        }
        Ok(())
    }

    /// A memory pool of `limit` bytes that remembers the most bytes that were
    /// reserved at once.
    #[derive(Debug)]
    struct PeakPool {
        pool: GreedyMemoryPool,
        peak: AtomicUsize,
    }

    impl PeakPool {
        fn new(limit: usize) -> Self {
            Self {
                pool: GreedyMemoryPool::new(limit),
                peak: AtomicUsize::new(0),
            }
        }

        fn record(&self) {
            self.peak.fetch_max(self.pool.reserved(), Ordering::Relaxed);
        }
    }

    impl fmt::Display for PeakPool {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "PeakPool({})", self.pool)
        }
    }

    impl MemoryPool for PeakPool {
        fn name(&self) -> &str {
            "PeakPool"
        }

        fn grow(&self, reservation: &MemoryReservation, additional: usize) {
            self.pool.grow(reservation, additional);
            self.record();
        }

        fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
            self.pool.shrink(reservation, shrink);
        }

        fn try_grow(
            &self,
            reservation: &MemoryReservation,
            additional: usize,
        ) -> Result<()> {
            self.pool.try_grow(reservation, additional)?;
            self.record();
            Ok(())
        }

        fn reserved(&self) -> usize {
            self.pool.reserved()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_long_streams_keep_their_buffered_rows_bounded() -> Result<()>
    {
        // An ordered join of two streams of 300,000 rows with keys of 48
        // bytes: the join drops the buffered rows that no later streamed row
        // matches, again and again, in whole batches, and copies the keys of
        // the others into keys of their size, all reserved, so that it runs
        // in a pool that holds far fewer rows than the streams have (about 33
        // MB of rows), and reserves a bounded part of it at most.
        let rows = 300_000;
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Utf8, false),
            ])) as SchemaRef
        };
        let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
            let batches = (0..rows)
                .step_by(4096)
                .map(|start| {
                    let ids = (start..(start + 4096).min(rows)).collect::<Vec<i64>>();
                    RecordBatch::try_new(
                        schema(side),
                        vec![
                            Arc::new(Int64Array::from(ids.clone())),
                            Arc::new(StringArray::from_iter_values(
                                ids.iter().map(|id| format!("{id:048}")),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect();
            let key = col(&format!("{side}k"), &schema(side)).unwrap();
            let order = LexOrdering::new([PhysicalSortExpr::new(
                key,
                SortOptions::new(false, true),
            )])
            .unwrap();
            let partition = StreamPartition {
                schema: schema(side),
                batches,
                forever: false,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema(side),
                    vec![Arc::new(partition)],
                    None,
                    [order],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let (left, right) = (stream("l"), stream("r"));
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortMergeJoinExec::try_new(
            left,
            right,
            on,
            None,
            JoinType::Inner,
            vec![SortOptions::new(false, true)],
            NullEquality::NullEqualsNothing,
        )?);
        let root = lower_join(&plan, false)?;
        assert_streaming_merge_join(&root, "long streams");
        let pool = Arc::new(PeakPool::new(24 << 20));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
            .build_arc()?;
        let ctx = Arc::new(TaskContext::default().with_runtime(runtime));
        let output = run_native(&root, plan, ctx).await?;
        assert_eq!(num_rows(&output.batches), rows as usize);
        let peak = pool.peak.load(Ordering::Relaxed);
        assert!(peak < 20 << 20, "the join reserved {peak} bytes at most");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wide_computed_keys_are_reserved() -> Result<()> {
        // As the symmetric join's: a merge join of 1,024 streamed rows with a
        // buffered row on a computed key, a string of 16 KiB, reserves the
        // keys of the streamed chunk (16 MB as values, 16 MB encoded) while it
        // encodes and joins them.
        use crate::join::symmetric_hash_join::tests::{id_rows, wide_key};
        let join = || -> Result<Arc<dyn PhysicalOperator>> {
            let source = |batch: RecordBatch| -> Arc<dyn PhysicalOperator> {
                Arc::new(TestSource::with_schema(
                    batch.schema(),
                    vec![DataChunk::from_record_batch(batch)],
                ))
            };
            Ok(Arc::new(PhysicalMergeJoin::try_new(
                source(id_rows("l", 1024)),
                source(id_rows("r", 1)),
                vec![(wide_key(), wide_key())],
                None,
                JoinType::Inner,
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
                false,
                None,
            )?))
        };
        let err =
            with_timeout(Executor::collect(join()?, limited_task_ctx(256 * 1024)?, 4))
                .await
                .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let batches =
            with_timeout(Executor::collect(join()?, limited_task_ctx(64 << 20)?, 4))
                .await?;
        assert_eq!(num_rows(&batches), 1024);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn encoded_keys_of_shared_views_are_reserved() -> Result<()> {
        // As the symmetric join's: 128 streamed keys that are views of the same
        // string of 1 MiB take 128 MiB encoded, which the merge join reserves
        // before it encodes them.
        use crate::join::symmetric_hash_join::tests::shared_view_rows;
        let join = |rows: usize| -> Result<Arc<dyn PhysicalOperator>> {
            let source = |batch: RecordBatch| -> Arc<dyn PhysicalOperator> {
                Arc::new(TestSource::with_schema(
                    batch.schema(),
                    vec![DataChunk::from_record_batch(batch)],
                ))
            };
            let (left, right) = (
                source(shared_view_rows("l", rows)),
                source(shared_view_rows("r", 1)),
            );
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            Ok(Arc::new(PhysicalMergeJoin::try_new(
                left,
                right,
                on,
                None,
                JoinType::Inner,
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
                false,
                None,
            )?))
        };
        let err =
            with_timeout(Executor::collect(join(128)?, limited_task_ctx(8 << 20)?, 4))
                .await
                .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        let batches =
            with_timeout(Executor::collect(join(8)?, limited_task_ctx(64 << 20)?, 4))
                .await?;
        assert_eq!(num_rows(&batches), 8);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn null_aware_joins_of_streams_end_at_a_null_struct_key() -> Result<()> {
        // The `NOT IN` of streams that never end, of struct keys `(f, i)`: the
        // buffered (right) stream produces a NULL struct (whose fields are
        // `(-0.0, 5)`, which do not count) and then nothing, forever. The NULL
        // proves the result empty: the join ends at once (it neither takes the
        // fields for a zero, nor waits for a run to end), as DataFusion's does.
        let fields = Fields::from(vec![
            Field::new("f", DataType::Float64, false),
            Field::new("i", DataType::Int32, false),
        ]);
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Struct(fields.clone()), true),
            ])) as SchemaRef
        };
        let rows = |side: &str, keys: &[(f64, i32)], nulls: Option<NullBuffer>| {
            let key = StructArray::new(
                fields.clone(),
                vec![
                    Arc::new(Float64Array::from_iter_values(
                        keys.iter().map(|key| key.0),
                    )),
                    Arc::new(Int32Array::from_iter_values(keys.iter().map(|key| key.1))),
                ],
                nulls,
            );
            RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int64Array::from_iter_values(0..keys.len() as i64)),
                    Arc::new(key),
                ],
            )
            .unwrap()
        };
        let stream = |side: &str, batch: RecordBatch| -> Arc<dyn ExecutionPlan> {
            let key = col(&format!("{side}k"), &schema(side)).unwrap();
            let order = LexOrdering::new([PhysicalSortExpr::new(
                key,
                SortOptions::new(false, true),
            )])
            .unwrap();
            let partition = StreamPartition {
                schema: schema(side),
                batches: vec![batch],
                forever: true,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema(side),
                    vec![Arc::new(partition)],
                    None,
                    [order],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let left = stream("l", rows("l", &[(1.0, 1), (2.0, 2)], None));
            let right = stream(
                "r",
                rows("r", &[(-0.0, 5)], Some(NullBuffer::from(vec![false]))),
            );
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            let join = SortMergeJoinExec::try_new(
                left,
                right,
                on,
                None,
                JoinType::LeftAnti,
                vec![SortOptions::new(false, true)],
                NullEquality::NullEqualsNothing,
            )?
            .with_null_aware(true)?;
            Ok(Arc::new(GlobalLimitExec::new(Arc::new(join), 0, Some(1))))
        };
        let expected = with_timeout(collect(plan()?, task_ctx())).await?;
        assert_eq!(num_rows(&expected), 0);
        let plan = plan()?;
        let root = crate::planner::lower_execution_plan(&plan)?;
        assert_streaming_merge_join(&root.children()[0], "NOT IN");
        let actual = run_native(&root, plan, task_ctx()).await?;
        assert_eq!(num_rows(&actual.batches), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_working_memory_of_merge_join_sorts_is_reserved() -> Result<()> {
        // A sort of 1,024 rows by a computed key, a string of 16 KiB, and an
        // integer: while it sorts, it holds the computed keys (16 MB), their
        // encoding (16 MB) and its sorted copy (16 MB), which it reserves, so
        // that it fails in a pool of 20 MiB, and sorts in one of 128 MiB.
        use crate::join::symmetric_hash_join::tests::{id_rows, wide_key};
        let sort = || -> Result<Arc<dyn PhysicalOperator>> {
            let batch = id_rows("l", 1024);
            let id = col("lid", &batch.schema())?;
            let source: Arc<dyn PhysicalOperator> = Arc::new(TestSource::with_schema(
                batch.schema(),
                vec![DataChunk::from_record_batch(batch)],
            ));
            let ordering = LexOrdering::new([
                PhysicalSortExpr::new(wide_key(), SortOptions::default()),
                PhysicalSortExpr::new(id, SortOptions::default()),
            ])
            .unwrap();
            Ok(Arc::new(MergeJoinSort::try_new(source, ordering)?))
        };
        let err =
            with_timeout(Executor::collect(sort()?, limited_task_ctx(20 << 20)?, 4))
                .await
                .unwrap_err();
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{err}"
        );
        assert!(err.to_string().contains("MergeJoinSort"), "{err}");
        let batches =
            with_timeout(Executor::collect(sort()?, limited_task_ctx(128 << 20)?, 4))
                .await?;
        assert_eq!(num_rows(&batches), 1024);
        Ok(())
    }

    /// A partition of a stream that produces `batches`, and then ends, or
    /// never produces anything again if `forever`.
    #[derive(Debug)]
    pub(crate) struct StreamPartition {
        pub(crate) schema: SchemaRef,
        pub(crate) batches: Vec<RecordBatch>,
        pub(crate) forever: bool,
    }

    impl PartitionStream for StreamPartition {
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

    /// The order of the sorted streams: by `<side>k`, NULLs first.
    fn stream_order(side: &str) -> LexOrdering {
        let key = col(&format!("{side}k"), &data_schema(side)).unwrap();
        LexOrdering::new([PhysicalSortExpr::new(key, SortOptions::new(false, true))])
            .unwrap()
    }

    /// A stream (that never ends, for DataFusion) of the rows `0..rows` of the
    /// `side` input (see [`data_batch`]) in `partitions` partitions of batches
    /// of 50 rows, every partition sorted by `<side>k` (NULLs first). The
    /// partitions end, unless `forever` (then they produce nothing after their
    /// rows).
    fn sorted_stream(
        side: &str,
        rows: i64,
        skew: i64,
        partitions: i64,
        forever: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let batch = data_batch(side, 0..rows, skew);
        let partitions = (0..partitions)
            .map(|partition| {
                let rows = (0..rows)
                    .filter(|row| row % partitions == partition)
                    .map(|row| row as u32)
                    .collect::<Vec<_>>();
                let rows = take_record_batch(&batch, &UInt32Array::from(rows)).unwrap();
                let order = SortColumn {
                    values: Arc::clone(rows.column(1)),
                    options: Some(SortOptions::new(false, true)),
                };
                let sorted = lexsort_to_indices(&[order], None).unwrap();
                let sorted = take_record_batch(&rows, &sorted).unwrap();
                let batches = (0..sorted.num_rows())
                    .step_by(50)
                    .map(|offset| {
                        sorted.slice(offset, 50.min(sorted.num_rows() - offset))
                    })
                    .collect();
                Arc::new(StreamPartition {
                    schema: data_schema(side),
                    batches,
                    forever,
                }) as Arc<dyn PartitionStream>
            })
            .collect();
        Arc::new(
            StreamingTableExec::try_new(
                data_schema(side),
                partitions,
                None,
                [stream_order(side)],
                true,
                None,
            )
            .unwrap(),
        )
    }

    /// A sort-merge join of sorted streams (see [`sorted_stream`]) of the
    /// left and the right input on `lk = rk`, of a single partition: its
    /// order is the order of its streamed input. A stream of several
    /// partitions is merged by a `SortPreservingMergeExec` first.
    fn join_of_sorted_streams(
        join_type: JoinType,
        filter: Option<JoinFilter>,
        partitions: i64,
        forever: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input = |side: &str, rows: i64, skew: i64| -> Arc<dyn ExecutionPlan> {
            let stream = sorted_stream(side, rows, skew, partitions, forever);
            match partitions {
                1 => stream,
                _ => Arc::new(SortPreservingMergeExec::new(stream_order(side), stream)),
            }
        };
        let (left, right) = (input("l", 300, 0), input("r", 500, 3));
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        // The `NOT IN` of a single partition, without filter.
        let null_aware =
            join_type == JoinType::LeftAnti && filter.is_none() && partitions == 1;
        let join = SortMergeJoinExec::try_new(
            left,
            right,
            on,
            filter,
            join_type,
            vec![SortOptions::new(false, true)],
            NullEquality::NullEqualsNothing,
        )?
        .with_null_aware(null_aware)?;
        Ok(Arc::new(join))
    }

    /// Assert that `root`, a native merge join, reads its inputs as streams:
    /// it neither sorts them nor collects them, nor is it left to DataFusion,
    /// but the streams below it.
    fn assert_streaming_merge_join(root: &Arc<dyn PhysicalOperator>, context: &str) {
        let tree = format_operator_tree(root);
        let join = root
            .downcast_ref::<PhysicalMergeJoin>()
            .unwrap_or_else(|| panic!("not a merge join: {context}\n{tree}"));
        assert!(join.streams_buffered_input(), "{context}\n{tree}");
        assert_eq!(count::<MergeJoinSort>(root), 0, "{context}\n{tree}");
        for plan in bridged_plans(root) {
            let streams = plan
                .apply(|node| {
                    Ok(
                        match node.is::<StreamingTableExec>()
                            || node.is::<SortPreservingMergeExec>()
                        {
                            true => TreeNodeRecursion::Continue,
                            false => TreeNodeRecursion::Stop,
                        },
                    )
                })
                .is_ok_and(|recursion| recursion != TreeNodeRecursion::Stop);
            assert!(streams, "{context}: bridged {}\n{tree}", plan.name());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_sorted_inputs_that_never_end_are_native() -> Result<()> {
        // A sort-merge join of one partition promises its order: the merge join
        // reads its inputs, which never end, in their order (the buffered one
        // as a stream), instead of sorting them, and emits its rows in the
        // order of the streamed rows, as they come. (A `Full` join has no
        // order: it is a symmetric hash join.)
        for join_type in ALL_JOIN_TYPES
            .into_iter()
            .filter(|join_type| *join_type != JoinType::Full)
        {
            for filter in [None, Some(("id", Operator::Lt, "id"))] {
                for partitions in [1, 3] {
                    let context = format!("{join_type} {filter:?} {partitions}");
                    let make_plan = || {
                        let filter = filter.map(|(l, op, r)| join_filter(l, op, r));
                        join_of_sorted_streams(join_type, filter, partitions, false)
                    };
                    let plan = make_plan()?;
                    let expected = EngineOutput {
                        schema: plan.schema(),
                        plan: Arc::clone(&plan),
                        batches: with_timeout(collect(make_plan()?, task_ctx())).await?,
                        ordered: false,
                    };
                    let root = lower_join(&plan, false)?;
                    assert_streaming_merge_join(&root, &context);
                    let actual = run_native(&root, Arc::clone(&plan), task_ctx()).await?;
                    if let Err(message) = compare_outputs(&expected, &actual) {
                        panic!(
                            "Results differ: {message}\n{context}\n{}",
                            format_operator_tree(&root)
                        );
                    }
                    let ordering = plan.output_ordering().expect("an ordered join");
                    let ordering =
                        LexOrdering::new(ordering.iter().take(1).cloned()).unwrap();
                    assert_sorted(&actual.batches, &ordering, &context);
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_streams_that_never_end_emit_their_rows() -> Result<()> {
        // The streams produce their rows and then nothing, forever: the join
        // emits its rows in order as they come, so that a limit above it ends
        // the query, with the first rows of the join.
        for join_type in [JoinType::Inner, JoinType::Left, JoinType::RightSemi] {
            let limit = 40;
            let all = |forever| join_of_sorted_streams(join_type, None, 1, forever);
            let plan = all(true)?;
            let limited: Arc<dyn ExecutionPlan> =
                Arc::new(GlobalLimitExec::new(Arc::clone(&plan), 0, Some(limit)));
            let root = crate::planner::lower_execution_plan(&limited)?;
            let join = &root.children()[0];
            assert_streaming_merge_join(join, &join_type.to_string());
            let output = run_native(&root, limited, task_ctx()).await?;
            assert_eq!(num_rows(&output.batches), limit, "{join_type}");
            // The first rows of the join of the rows of the streams.
            let expected = collect(all(false)?, task_ctx()).await?;
            let expected = concat_batches(&plan.schema(), &expected)?;
            let ordering = plan.output_ordering().expect("an ordered join");
            let ordering = LexOrdering::new(ordering.iter().take(1).cloned()).unwrap();
            assert_sorted(&output.batches, &ordering, &join_type.to_string());
            let keys = |batch: &RecordBatch| {
                let key = ordering[0].expr.evaluate(batch).unwrap();
                key.into_array(batch.num_rows()).unwrap()
            };
            let actual = concat_batches(&plan.schema(), &output.batches)?;
            assert_eq!(
                keys(&actual).as_ref(),
                keys(&expected.slice(0, limit)).as_ref(),
                "{join_type}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn null_aware_joins_of_streams_end_at_a_null_buffered_key() -> Result<()> {
        // The `NOT IN` of streams that never end: the streamed (left) one
        // produces the keys 1, 2 and 3, the buffered (right) one a NULL key,
        // and then both produce nothing, forever. The NULL proves the result
        // empty: the join ends at once, as DataFusion's does, also under a
        // limit that it never reaches.
        let stream = |side: &str, keys: Vec<Option<i32>>| -> Arc<dyn ExecutionPlan> {
            let schema = Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Int32, true),
            ]));
            let ids = Int64Array::from_iter_values(0..keys.len() as i64);
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(ids), Arc::new(Int32Array::from(keys))],
            )
            .unwrap();
            let key = col(&format!("{side}k"), &schema).unwrap();
            let order = LexOrdering::new([PhysicalSortExpr::new(
                key,
                SortOptions::new(false, true),
            )])
            .unwrap();
            let partition = StreamPartition {
                schema: Arc::clone(&schema),
                batches: vec![batch],
                forever: true,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema,
                    vec![Arc::new(partition)],
                    None,
                    [order],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let left = stream("l", vec![Some(1), Some(2), Some(3)]);
            let right = stream("r", vec![None]);
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            let join = SortMergeJoinExec::try_new(
                left,
                right,
                on,
                None,
                JoinType::LeftAnti,
                vec![SortOptions::new(false, true)],
                NullEquality::NullEqualsNothing,
            )?
            .with_null_aware(true)?;
            Ok(Arc::new(GlobalLimitExec::new(Arc::new(join), 0, Some(1))))
        };
        let expected = with_timeout(collect(plan()?, task_ctx())).await?;
        assert_eq!(num_rows(&expected), 0);
        let plan = plan()?;
        let root = crate::planner::lower_execution_plan(&plan)?;
        assert_streaming_merge_join(&root.children()[0], "NOT IN");
        let actual = run_native(&root, plan, task_ctx()).await?;
        assert_eq!(num_rows(&actual.batches), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_streams_with_signed_zeros_match_them() -> Result<()> {
        // Streams sorted by `(a, b)`, a float and an integer, in which `-0.0`
        // comes before `+0.0`: their rows `(-0.0, 5)` and `(+0.0, 1)` are not
        // sorted once `-0.0` is `+0.0`, as the join compares its keys. The
        // ordered join (of a single partition) matches them, in the order of
        // the streamed rows, as DataFusion's hash join does.
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}a"), DataType::Float64, true),
                Field::new(format!("{side}b"), DataType::Int32, true),
            ])) as SchemaRef
        };
        // The rows `rows` of the `side` input, the first with the id `first`.
        let rows = |side: &str, rows: &[(f64, i32)], first: i64| {
            RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        first..first + rows.len() as i64,
                    )),
                    Arc::new(Float64Array::from_iter_values(
                        rows.iter().map(|row| row.0),
                    )),
                    Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.1))),
                ],
            )
            .unwrap()
        };
        type Keys = &'static [(f64, i32)];
        let cases: [(Keys, Keys); 3] = [
            (&[(-0.0, 5), (0.0, 1)], &[(-0.0, 5), (0.0, 1)]),
            (
                &[
                    (-1.0, 3),
                    (-0.0, 2),
                    (-0.0, 7),
                    (0.0, 1),
                    (0.0, 7),
                    (2.0, 0),
                ],
                &[(-0.0, 1), (-0.0, 7), (0.0, 2), (0.0, 3), (0.0, 7), (1.0, 4)],
            ),
            (&[(0.0, 1), (0.0, 5)], &[(-0.0, 5), (-0.0, 9), (0.0, 1)]),
        ];
        // The rows of a case in the order of their keys, ascending or descending
        // (in which `+0.0` comes before `-0.0`).
        let sorted = |rows: &[(f64, i32)], descending: bool| {
            let mut rows = rows.to_vec();
            rows.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            if descending {
                rows.reverse();
            }
            rows
        };
        for ((left_rows, right_rows), descending) in cases
            .into_iter()
            .flat_map(|case| [(case, false), (case, true)])
        {
            let (left_rows, right_rows) = (
                &sorted(left_rows, descending)[..],
                &sorted(right_rows, descending)[..],
            );
            let options = SortOptions::new(descending, true);
            for join_type in [
                JoinType::Inner,
                JoinType::Left,
                JoinType::LeftSemi,
                JoinType::LeftAnti,
                JoinType::Right,
                JoinType::RightSemi,
                JoinType::RightAnti,
            ] {
                let context =
                    format!("{join_type} {left_rows:?} {right_rows:?} {descending}");
                let order = |side: &str| {
                    LexOrdering::new([
                        PhysicalSortExpr::new(
                            col(&format!("{side}a"), &schema(side)).unwrap(),
                            options,
                        ),
                        PhysicalSortExpr::new(
                            col(&format!("{side}b"), &schema(side)).unwrap(),
                            options,
                        ),
                    ])
                    .unwrap()
                };
                let stream =
                    |side: &str, batches: Vec<RecordBatch>| -> Arc<dyn ExecutionPlan> {
                        let partition = StreamPartition {
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
                        )
                    };
                // One row a batch, and all in one; with a buffered input that
                // never ends (read as a stream), or one that ends (collected and
                // sorted), against the streamed rows of a stream.
                for (split, mixed) in [(true, false), (false, false), (true, true)] {
                    let batches = |side: &str, all: &[(f64, i32)]| match split {
                        true => all
                            .iter()
                            .enumerate()
                            .map(|(id, row)| {
                                rows(side, std::slice::from_ref(row), id as i64)
                            })
                            .collect(),
                        false => vec![rows(side, all, 0)],
                    };
                    let on = |left: &Arc<dyn ExecutionPlan>, right: &Arc<dyn ExecutionPlan>| -> Result<Vec<(PhysicalExprRef, PhysicalExprRef)>> {
                        Ok(vec![
                            (col("la", &left.schema())?, col("ra", &right.schema())?),
                            (col("lb", &left.schema())?, col("rb", &right.schema())?),
                        ])
                    };
                    let bounded = |side: &str, all: &[(f64, i32)]| {
                        MemorySourceConfig::try_new_exec(
                            &[vec![rows(side, all, 0)]],
                            schema(side),
                            None,
                        )
                        .unwrap() as Arc<dyn ExecutionPlan>
                    };
                    let (left, right) =
                        (bounded("l", left_rows), bounded("r", right_rows));
                    let reference = Arc::new(HashJoinExec::try_new(
                        Arc::clone(&left),
                        Arc::clone(&right),
                        on(&left, &right)?,
                        None,
                        &join_type,
                        None,
                        PartitionMode::CollectLeft,
                        NullEquality::NullEqualsNothing,
                        false,
                    )?) as Arc<dyn ExecutionPlan>;
                    let expected = EngineOutput {
                        schema: reference.schema(),
                        batches: with_timeout(collect(
                            Arc::clone(&reference),
                            task_ctx(),
                        ))
                        .await?,
                        plan: reference,
                        ordered: false,
                    };
                    let streamed_side = SortMergeJoinExec::probe_side(&join_type);
                    let input = |side: &str, all: &[(f64, i32)]| {
                        let buffered = match streamed_side {
                            JoinSide::Left => side == "r",
                            _ => side == "l",
                        };
                        match mixed && buffered {
                            true => bounded(side, all),
                            false => stream(side, batches(side, all)),
                        }
                    };
                    let (left, right) = (input("l", left_rows), input("r", right_rows));
                    let on = on(&left, &right)?;
                    let plan: Arc<dyn ExecutionPlan> =
                        Arc::new(SortMergeJoinExec::try_new(
                            left,
                            right,
                            on,
                            None,
                            join_type,
                            vec![options; 2],
                            NullEquality::NullEqualsNothing,
                        )?);
                    let root = lower_join(&plan, false)?;
                    let context = format!("{context}, mixed: {mixed}, split: {split}");
                    match mixed {
                        true => {
                            let join = root
                                .downcast_ref::<PhysicalMergeJoin>()
                                .unwrap_or_else(|| panic!("{context}"));
                            assert!(!join.streams_buffered_input(), "{context}");
                        }
                        false => assert_streaming_merge_join(&root, &context),
                    }
                    let actual = run_native(&root, Arc::clone(&plan), task_ctx())
                        .await
                        .unwrap_or_else(|e| panic!("{context}: {e}"));
                    if let Err(message) = compare_outputs(&expected, &actual) {
                        panic!(
                            "Results differ: {message}\n{context}\n{}",
                            format_operator_tree(&root)
                        );
                    }
                    // In the order of the streamed rows (by their ids).
                    let streamed = match streamed_side {
                        JoinSide::Left => "lid",
                        _ => "rid",
                    };
                    let ids = actual
                        .batches
                        .iter()
                        .flat_map(|batch| {
                            let ids = batch.column_by_name(streamed).unwrap();
                            let ids = ids.as_any().downcast_ref::<Int64Array>().unwrap();
                            ids.values().to_vec()
                        })
                        .collect::<Vec<_>>();
                    assert!(
                        ids.windows(2).all(|pair| pair[0] <= pair[1]),
                        "{context}: {ids:?}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The schema of the rows `(id, k)` of the `side` input whose key `k` is a
    /// struct of a float `f` and an integer `i`.
    fn struct_key_schema(side: &str) -> SchemaRef {
        let fields = Fields::from(vec![
            Field::new("f", DataType::Float64, false),
            Field::new("i", DataType::Int32, false),
        ]);
        Arc::new(Schema::new(vec![
            Field::new(format!("{side}id"), DataType::Int64, false),
            Field::new(format!("{side}k"), DataType::Struct(fields), false),
        ]))
    }

    /// The rows of the `side` input whose struct keys are `keys`, with the ids
    /// from `first` on.
    fn struct_key_rows(side: &str, keys: &[(f64, i32)], first: i64) -> RecordBatch {
        let schema = struct_key_schema(side);
        let DataType::Struct(fields) = schema.field(1).data_type().clone() else {
            unreachable!()
        };
        let key = StructArray::new(
            fields,
            vec![
                Arc::new(Float64Array::from_iter_values(keys.iter().map(|key| key.0))),
                Arc::new(Int32Array::from_iter_values(keys.iter().map(|key| key.1))),
            ],
            None,
        );
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(
                    first..first + keys.len() as i64,
                )),
                Arc::new(key),
            ],
        )
        .unwrap()
    }

    /// A stream of `batches` of the `side` input with struct keys, sorted by
    /// them with `options`, which never ends if `forever`.
    fn struct_key_stream(
        side: &str,
        batches: Vec<RecordBatch>,
        options: SortOptions,
        forever: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = struct_key_schema(side);
        let key = col(&format!("{side}k"), &schema).unwrap();
        let order = LexOrdering::new([PhysicalSortExpr::new(key, options)]).unwrap();
        let partition = StreamPartition {
            schema: Arc::clone(&schema),
            batches,
            forever,
        };
        Arc::new(
            StreamingTableExec::try_new(
                schema,
                vec![Arc::new(partition)],
                None,
                [order],
                true,
                None,
            )
            .unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_streams_with_signed_zeros_in_structs_match_them()
    -> Result<()> {
        // A join on one key, a struct of a float and an integer: streams sorted
        // by it have `(-0.0, 5)` before `(+0.0, 1)`, which the join compares as
        // `(0, 5)` and `(0, 1)`. The ordered join matches them as DataFusion's
        // hash join does, in the order of the streamed rows.
        type Keys = &'static [(f64, i32)];
        let cases: [(Keys, Keys); 3] = [
            (&[(-0.0, 5), (0.0, 1)], &[(-0.0, 5), (0.0, 1)]),
            (
                &[
                    (-1.0, 3),
                    (-0.0, 2),
                    (-0.0, 7),
                    (0.0, 1),
                    (0.0, 7),
                    (2.0, 0),
                ],
                &[(-0.0, 1), (-0.0, 7), (0.0, 2), (0.0, 3), (0.0, 7), (1.0, 4)],
            ),
            (&[(0.0, 1), (0.0, 5)], &[(-0.0, 5), (-0.0, 9), (0.0, 1)]),
        ];
        // The keys of a case in their order, ascending or descending (in
        // which `+0.0` comes before `-0.0`).
        let sorted = |keys: &[(f64, i32)], descending: bool| {
            let mut keys = keys.to_vec();
            keys.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            if descending {
                keys.reverse();
            }
            keys
        };
        for ((left_keys, right_keys), descending) in cases
            .into_iter()
            .flat_map(|case| [(case, false), (case, true)])
        {
            let (left_keys, right_keys) = (
                &sorted(left_keys, descending)[..],
                &sorted(right_keys, descending)[..],
            );
            let options = SortOptions::new(descending, true);
            for join_type in [
                JoinType::Inner,
                JoinType::Left,
                JoinType::LeftSemi,
                JoinType::LeftAnti,
                JoinType::Right,
                JoinType::RightSemi,
                JoinType::RightAnti,
            ] {
                for split in [true, false] {
                    let context = format!(
                        "{join_type} {left_keys:?} {right_keys:?} {split} {descending}"
                    );
                    let batches = |side: &str, keys: &[(f64, i32)]| match split {
                        true => keys
                            .iter()
                            .enumerate()
                            .map(|(id, key)| {
                                struct_key_rows(
                                    side,
                                    std::slice::from_ref(key),
                                    id as i64,
                                )
                            })
                            .collect(),
                        false => vec![struct_key_rows(side, keys, 0)],
                    };
                    let on =
                        |left: &Arc<dyn ExecutionPlan>,
                         right: &Arc<dyn ExecutionPlan>| {
                            vec![(
                                col("lk", &left.schema()).unwrap(),
                                col("rk", &right.schema()).unwrap(),
                            )]
                        };
                    let bounded = |side: &str, keys: &[(f64, i32)]| {
                        MemorySourceConfig::try_new_exec(
                            &[vec![struct_key_rows(side, keys, 0)]],
                            struct_key_schema(side),
                            None,
                        )
                        .unwrap() as Arc<dyn ExecutionPlan>
                    };
                    let (left, right) =
                        (bounded("l", left_keys), bounded("r", right_keys));
                    let reference = Arc::new(HashJoinExec::try_new(
                        Arc::clone(&left),
                        Arc::clone(&right),
                        on(&left, &right),
                        None,
                        &join_type,
                        None,
                        PartitionMode::CollectLeft,
                        NullEquality::NullEqualsNothing,
                        false,
                    )?) as Arc<dyn ExecutionPlan>;
                    let expected = EngineOutput {
                        schema: reference.schema(),
                        batches: with_timeout(collect(
                            Arc::clone(&reference),
                            task_ctx(),
                        ))
                        .await?,
                        plan: reference,
                        ordered: false,
                    };
                    let left =
                        struct_key_stream("l", batches("l", left_keys), options, false);
                    let right =
                        struct_key_stream("r", batches("r", right_keys), options, false);
                    let on = on(&left, &right);
                    let plan: Arc<dyn ExecutionPlan> =
                        Arc::new(SortMergeJoinExec::try_new(
                            left,
                            right,
                            on,
                            None,
                            join_type,
                            vec![options],
                            NullEquality::NullEqualsNothing,
                        )?);
                    let root = lower_join(&plan, false)?;
                    assert_streaming_merge_join(&root, &context);
                    let actual = run_native(&root, Arc::clone(&plan), task_ctx())
                        .await
                        .unwrap_or_else(|e| panic!("{context}: {e}"));
                    if let Err(message) = compare_outputs(&expected, &actual) {
                        panic!("Results differ: {message}\n{context}");
                    }
                    let streamed = match SortMergeJoinExec::probe_side(&join_type) {
                        JoinSide::Left => "lid",
                        _ => "rid",
                    };
                    let ids = actual
                        .batches
                        .iter()
                        .flat_map(|batch| {
                            let ids = batch.column_by_name(streamed).unwrap();
                            ids.as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .values()
                                .to_vec()
                        })
                        .collect::<Vec<_>>();
                    assert!(
                        ids.windows(2).all(|pair| pair[0] <= pair[1]),
                        "{context}: {ids:?}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The keys of the rows `rows` (a float and an integer) in the layout
    /// `layout`: a list (or list view, ...) of both as floats, a map of the
    /// float to the integer, or a run-end encoded float and the integer (two
    /// keys).
    fn nested_keys(layout: &str, rows: &[(f64, i32)]) -> Vec<ArrayRef> {
        let lists = || {
            rows.iter()
                .map(|&(f, i)| Some(vec![Some(f), Some(f64::from(i))]))
                .collect::<Vec<_>>()
        };
        match layout {
            "List" => vec![Arc::new(
                ListArray::from_iter_primitive::<Float64Type, _, _>(lists()),
            )],
            "LargeList" => vec![Arc::new(LargeListArray::from_iter_primitive::<
                Float64Type,
                _,
                _,
            >(lists()))],
            "ListView" => vec![Arc::new(ListViewArray::from(
                ListArray::from_iter_primitive::<Float64Type, _, _>(lists()),
            ))],
            "LargeListView" => vec![Arc::new(LargeListViewArray::from(
                LargeListArray::from_iter_primitive::<Float64Type, _, _>(lists()),
            ))],
            "FixedSizeList" => vec![Arc::new(FixedSizeListArray::from_iter_primitive::<
                Float64Type,
                _,
                _,
            >(lists(), 2))],
            "Map" => {
                let mut builder =
                    MapBuilder::new(None, Float64Builder::new(), Int32Builder::new());
                for &(f, i) in rows {
                    builder.keys().append_value(f);
                    builder.values().append_value(i);
                    builder.append(true).unwrap();
                }
                vec![Arc::new(builder.finish())]
            }
            _ => {
                let ends = Int32Array::from_iter_values(1..=rows.len() as i32);
                let floats = Float64Array::from_iter_values(rows.iter().map(|row| row.0));
                vec![
                    Arc::new(RunArray::<Int32Type>::try_new(&ends, &floats).unwrap()),
                    Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.1))),
                ]
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_streams_with_signed_zeros_in_nested_keys_match_them()
    -> Result<()> {
        // Keys of every nested layout that the row format supports, with the
        // rows `[-0.0, 5]` and `[+0.0, 1]` (in their order): the join compares
        // them as `[0, 5]` and `[0, 1]`. Streams of them (in one batch, and
        // in a batch a row) joined with themselves: every row matches itself,
        // as with DataFusion's hash join.
        let rows = [(-0.0, 5), (0.0, 1)];
        for layout in [
            "ListView",
            "LargeListView",
            "List",
            "LargeList",
            "FixedSizeList",
            "Map",
            "RunEndEncoded",
        ] {
            for split in [false, true] {
                let context = format!("{layout} split: {split}");
                let keys = nested_keys(layout, &rows);
                let schema = |side: &str| {
                    let mut fields =
                        vec![Field::new(format!("{side}id"), DataType::Int64, false)];
                    for (i, key) in keys.iter().enumerate() {
                        fields.push(Field::new(
                            format!("{side}k{i}"),
                            key.data_type().clone(),
                            true,
                        ));
                    }
                    Arc::new(Schema::new(fields)) as SchemaRef
                };
                let batch = |side: &str| {
                    let mut columns: Vec<ArrayRef> = vec![Arc::new(
                        Int64Array::from_iter_values(0..rows.len() as i64),
                    )];
                    columns.extend(nested_keys(layout, &rows));
                    RecordBatch::try_new(schema(side), columns).unwrap()
                };
                let batches = |side: &str| match split {
                    true => (0..rows.len())
                        .map(|row| batch(side).slice(row, 1))
                        .collect(),
                    false => vec![batch(side)],
                };
                let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
                    let order = LexOrdering::new((0..keys.len()).map(|i| {
                        PhysicalSortExpr::new(
                            col(&format!("{side}k{i}"), &schema(side)).unwrap(),
                            SortOptions::new(false, true),
                        )
                    }))
                    .unwrap();
                    let partition = StreamPartition {
                        schema: schema(side),
                        batches: batches(side),
                        forever: false,
                    };
                    Arc::new(
                        StreamingTableExec::try_new(
                            schema(side),
                            vec![Arc::new(partition)],
                            None,
                            [order],
                            true,
                            None,
                        )
                        .unwrap(),
                    )
                };
                let on = |left: &Arc<dyn ExecutionPlan>,
                          right: &Arc<dyn ExecutionPlan>| {
                    (0..keys.len())
                        .map(|i| {
                            (
                                col(&format!("lk{i}"), &left.schema()).unwrap(),
                                col(&format!("rk{i}"), &right.schema()).unwrap(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                let (left, right) = (stream("l"), stream("r"));
                let on_keys = on(&left, &right);
                let plan: Arc<dyn ExecutionPlan> = Arc::new(SortMergeJoinExec::try_new(
                    left,
                    right,
                    on_keys,
                    None,
                    JoinType::Inner,
                    vec![SortOptions::new(false, true); keys.len()],
                    NullEquality::NullEqualsNothing,
                )?);
                let root = lower_join(&plan, false)?;
                assert_streaming_merge_join(&root, &context);
                let actual = run_native(&root, Arc::clone(&plan), task_ctx())
                    .await
                    .unwrap_or_else(|e| panic!("{context}: {e}"));
                // Every row matches itself.
                let mut pairs = actual
                    .batches
                    .iter()
                    .flat_map(|batch| {
                        let id = |name: &str| {
                            let ids = batch.column_by_name(name).unwrap();
                            ids.as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .values()
                                .to_vec()
                        };
                        id("lid").into_iter().zip(id("rid"))
                    })
                    .collect::<Vec<_>>();
                pairs.sort_unstable();
                assert_eq!(pairs, vec![(0, 0), (1, 1)], "{context}");
                // And as DataFusion's hash join, if it joins such keys.
                let bounded = |side: &str| {
                    MemorySourceConfig::try_new_exec(
                        &[vec![batch(side)]],
                        schema(side),
                        None,
                    )
                    .unwrap() as Arc<dyn ExecutionPlan>
                };
                let (left, right) = (bounded("l"), bounded("r"));
                let on_keys = on(&left, &right);
                let reference = Arc::new(HashJoinExec::try_new(
                    left,
                    right,
                    on_keys,
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )?) as Arc<dyn ExecutionPlan>;
                if let Ok(batches) = collect(Arc::clone(&reference), task_ctx()).await {
                    let expected = EngineOutput {
                        schema: reference.schema(),
                        batches,
                        plan: reference,
                        ordered: false,
                    };
                    if let Err(message) = compare_outputs(&expected, &actual) {
                        panic!("Results differ: {message}\n{context}");
                    }
                } else {
                    assert_ne!(
                        layout, "ListView",
                        "DataFusion's hash join fails: {context}"
                    );
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_streams_of_struct_zeros_that_never_end_emit_their_rows()
    -> Result<()> {
        // Streams of struct keys with `-0.0` for every row, by ascending values
        // of the fields before it (a row a batch), which then produce nothing,
        // forever: a struct `(i, -0.0)` (whose float no value follows, as no key
        // follows it), a struct `(i, -0.0, j)`, and the keys `((i, -0.0), j)`.
        // A run of signed zeros ends where a value before its zero advances:
        // the join emits the matches of the rows of the runs before, and a
        // limit above it ends the query.
        let float_struct = |names: &[(&str, DataType)]| {
            Fields::from(
                names
                    .iter()
                    .map(|(name, data_type)| Field::new(*name, data_type.clone(), false))
                    .collect::<Vec<_>>(),
            )
        };
        let layouts: [(&str, Fields, bool); 3] = [
            (
                "(i, -0.0)",
                float_struct(&[("i", DataType::Int32), ("f", DataType::Float64)]),
                false,
            ),
            (
                "(i, -0.0, j)",
                float_struct(&[
                    ("i", DataType::Int32),
                    ("f", DataType::Float64),
                    ("j", DataType::Int32),
                ]),
                false,
            ),
            (
                "((i, -0.0), j)",
                float_struct(&[("i", DataType::Int32), ("f", DataType::Float64)]),
                true,
            ),
        ];
        for (name, fields, second_key) in layouts {
            let schema = |side: &str| {
                let mut columns = vec![
                    Field::new(format!("{side}id"), DataType::Int64, false),
                    Field::new(
                        format!("{side}k"),
                        DataType::Struct(fields.clone()),
                        false,
                    ),
                ];
                if second_key {
                    columns.push(Field::new(format!("{side}j"), DataType::Int32, false));
                }
                Arc::new(Schema::new(columns)) as SchemaRef
            };
            let row = |side: &str, i: i32| {
                let values = fields
                    .iter()
                    .map(|field| match field.data_type() {
                        DataType::Float64 => {
                            Arc::new(Float64Array::from(vec![-0.0])) as ArrayRef
                        }
                        _ if field.name() == "i" => Arc::new(Int32Array::from(vec![i])),
                        _ => Arc::new(Int32Array::from(vec![0])),
                    })
                    .collect();
                let mut columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from(vec![i64::from(i)])),
                    Arc::new(StructArray::new(fields.clone(), values, None)),
                ];
                if second_key {
                    columns.push(Arc::new(Int32Array::from(vec![0])));
                }
                RecordBatch::try_new(schema(side), columns).unwrap()
            };
            let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
                let mut keys = vec![PhysicalSortExpr::new(
                    col(&format!("{side}k"), &schema(side)).unwrap(),
                    SortOptions::new(false, true),
                )];
                if second_key {
                    keys.push(PhysicalSortExpr::new(
                        col(&format!("{side}j"), &schema(side)).unwrap(),
                        SortOptions::new(false, true),
                    ));
                }
                let partition = StreamPartition {
                    schema: schema(side),
                    batches: (0..3).map(|i| row(side, i)).collect(),
                    forever: true,
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        schema(side),
                        vec![Arc::new(partition)],
                        None,
                        [LexOrdering::new(keys).unwrap()],
                        true,
                        None,
                    )
                    .unwrap(),
                )
            };
            let (left, right) = (stream("l"), stream("r"));
            let mut on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            if second_key {
                on.push((col("lj", &left.schema())?, col("rj", &right.schema())?));
            }
            let keys = on.len();
            let join: Arc<dyn ExecutionPlan> = Arc::new(SortMergeJoinExec::try_new(
                left,
                right,
                on,
                None,
                JoinType::Inner,
                vec![SortOptions::new(false, true); keys],
                NullEquality::NullEqualsNothing,
            )?);
            let limited: Arc<dyn ExecutionPlan> =
                Arc::new(GlobalLimitExec::new(join, 0, Some(1)));
            let root = crate::planner::lower_execution_plan(&limited)?;
            assert_streaming_merge_join(&root.children()[0], name);
            let output = run_native(&root, limited, task_ctx()).await?;
            assert_eq!(num_rows(&output.batches), 1, "{name}");
            let batch = &output.batches[0];
            let id = |column: &str| {
                let ids = batch.column_by_name(column).unwrap();
                ids.as_any().downcast_ref::<Int64Array>().unwrap().value(0)
            };
            assert_eq!(id("lid"), id("rid"), "{name}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_streams_of_list_zeros_that_never_end_emit_their_rows()
    -> Result<()> {
        // Streams of list keys with `-0.0` in every row, by ascending values of
        // the elements before it (a row a batch), which then produce nothing,
        // forever: `[i, -0.0]`, `[i, -0.0, 5.0]` (a zero that other elements
        // follow), `[[i, -0.0]]` (a list of lists) and `[(i, -0.0)]` (a list of
        // structs). A run of signed zeros ends where a value before its zero
        // advances, also in a list: the join emits the matches of the rows of
        // the runs before, and a limit above it ends the query, as DataFusion's
        // join (of batches of one row).
        let floats = |values: &[f64]| -> ArrayRef {
            Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>([Some(
                values.iter().map(|&value| Some(value)),
            )]))
        };
        let lists = |values: &[f64]| -> ArrayRef {
            let inner = floats(values);
            let field = Arc::new(Field::new_list_field(inner.data_type().clone(), true));
            Arc::new(ListArray::new(
                field,
                arrow::buffer::OffsetBuffer::from_lengths([1]),
                inner,
                None,
            ))
        };
        let structs = |i: i32| -> ArrayRef {
            let fields = Fields::from(vec![
                Field::new("i", DataType::Int32, false),
                Field::new("f", DataType::Float64, false),
            ]);
            let values = StructArray::new(
                fields.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![i])),
                    Arc::new(Float64Array::from(vec![-0.0])),
                ],
                None,
            );
            let field = Arc::new(Field::new_list_field(DataType::Struct(fields), true));
            Arc::new(ListArray::new(
                field,
                arrow::buffer::OffsetBuffer::from_lengths([1]),
                Arc::new(values),
                None,
            ))
        };
        type Key<'a> = &'a dyn Fn(i32) -> ArrayRef;
        let layouts: [(&str, Key); 4] = [
            ("[i, -0.0]", &|i| floats(&[f64::from(i), -0.0])),
            ("[i, -0.0, 5.0]", &|i| floats(&[f64::from(i), -0.0, 5.0])),
            ("[[i, -0.0]]", &|i| lists(&[f64::from(i), -0.0])),
            ("[(i, -0.0)]", &structs),
        ];
        let schema = |side: &str, key: &ArrayRef| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), key.data_type().clone(), false),
            ])) as SchemaRef
        };
        // A stream of the rows of `keys` (of ids from 0 on), a row a batch,
        // sorted by them, which then produces nothing, forever.
        let stream = |side: &str, keys: Vec<ArrayRef>| -> Arc<dyn ExecutionPlan> {
            let schema = schema(side, &keys[0]);
            let batches = keys
                .into_iter()
                .zip(0..)
                .map(|(key, id)| {
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(Int64Array::from(vec![id])), key],
                    )
                    .unwrap()
                })
                .collect();
            let order = LexOrdering::new([PhysicalSortExpr::new(
                col(&format!("{side}k"), &schema).unwrap(),
                SortOptions::new(false, true),
            )])
            .unwrap();
            let partition = StreamPartition {
                schema: Arc::clone(&schema),
                batches,
                forever: true,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema,
                    vec![Arc::new(partition)],
                    None,
                    [order],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let limited = |left: Vec<ArrayRef>,
                       right: Vec<ArrayRef>,
                       join_type: JoinType|
         -> Result<Arc<dyn ExecutionPlan>> {
            let (left, right) = (stream("l", left), stream("r", right));
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            let join = SortMergeJoinExec::try_new(
                left,
                right,
                on,
                None,
                join_type,
                vec![SortOptions::new(false, true)],
                NullEquality::NullEqualsNothing,
            )?;
            Ok(Arc::new(GlobalLimitExec::new(Arc::new(join), 0, Some(1))))
        };
        let one_row_batches = || {
            Arc::new(
                TaskContext::default()
                    .with_session_config(SessionConfig::new().with_batch_size(1)),
            )
        };
        for (name, key) in layouts {
            let keys = || (1..=3).map(key).collect::<Vec<_>>();
            let plan = || limited(keys(), keys(), JoinType::Inner);
            let expected = with_timeout(collect(plan()?, one_row_batches())).await?;
            assert_eq!(num_rows(&expected), 1, "{name}");
            let plan = plan()?;
            let root = crate::planner::lower_execution_plan(&plan)?;
            assert_streaming_merge_join(&root.children()[0], name);
            let output = run_native(&root, plan, task_ctx()).await?;
            assert_eq!(num_rows(&output.batches), 1, "{name}");
            let batch = &output.batches[0];
            let id = |column: &str| {
                let ids = batch.column_by_name(column).unwrap();
                ids.as_any().downcast_ref::<Int64Array>().unwrap().value(0)
            };
            assert_eq!(id("lid"), id("rid"), "{name}");
        }
        // A `Left` join whose right stream produces `[3.0, -0.0, 1.0]` and
        // then nothing, forever: a run that never ends (`[3.0, +0.0, 0.5]`
        // could still come, before it), but the left keys `[2.0, 5.0]` and
        // `[3.0, -5.0]` come before its first row and have no zero, so that no
        // right row still to come equals them: they have all their right rows,
        // and are emitted without a match (DataFusion's join waits for more
        // right rows).
        for left in [[2.0, 5.0], [3.0, -5.0]] {
            let name = format!("{left:?}");
            let plan = limited(
                vec![floats(&left)],
                vec![floats(&[3.0, -0.0, 1.0])],
                JoinType::Left,
            )?;
            let root = crate::planner::lower_execution_plan(&plan)?;
            assert_streaming_merge_join(&root.children()[0], &name);
            let output = run_native(&root, plan, task_ctx()).await?;
            assert_eq!(num_rows(&output.batches), 1, "{name}");
            let rid = output.batches[0].column_by_name("rid").unwrap();
            assert_eq!(rid.null_count(), 1, "{name}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streamed_rows_before_a_held_run_do_not_wait_for_it() -> Result<()> {
        // A `Left` join of 20,000 left rows (-1.0, i) with the right rows
        // (-0.0, 5), (-0.0, 6), and then nothing, forever: the right rows are a
        // run of signed zeros that never ends (a row (+0.0, 1) could still
        // come, before them), but no right row still to come comes before
        // (0, ...), so that the left rows have all their right rows: they are
        // emitted without a match, and a limit above the join ends the query.
        // Of the keys (a, b), and of a struct key (a, b).
        let limit = 1;
        let order = SortOptions::new(false, true);
        // The keys (a, b).
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}a"), DataType::Float64, false),
                Field::new(format!("{side}b"), DataType::Int32, false),
            ])) as SchemaRef
        };
        let rows = |side: &str, keys: &[(f64, i32)], first: i64| {
            RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        first..first + keys.len() as i64,
                    )),
                    Arc::new(Float64Array::from_iter_values(
                        keys.iter().map(|key| key.0),
                    )),
                    Arc::new(Int32Array::from_iter_values(keys.iter().map(|key| key.1))),
                ],
            )
            .unwrap()
        };
        let stream = |side: &str, batches: Vec<RecordBatch>| -> Arc<dyn ExecutionPlan> {
            let keys = ["a", "b"].map(|key| {
                PhysicalSortExpr::new(
                    col(&format!("{side}{key}"), &schema(side)).unwrap(),
                    order,
                )
            });
            let partition = StreamPartition {
                schema: schema(side),
                batches,
                forever: true,
            };
            Arc::new(
                StreamingTableExec::try_new(
                    schema(side),
                    vec![Arc::new(partition)],
                    None,
                    [LexOrdering::new(keys).unwrap()],
                    true,
                    None,
                )
                .unwrap(),
            )
        };
        let left_keys = (0..20_000).map(|i| (-1.0, i)).collect::<Vec<_>>();
        let right_keys = [(-0.0, 5), (-0.0, 6)];
        let left = stream(
            "l",
            left_keys
                .chunks(4096)
                .enumerate()
                .map(|(batch, keys)| rows("l", keys, batch as i64 * 4096))
                .collect(),
        );
        let right = stream("r", vec![rows("r", &right_keys, 0)]);
        let on = vec![
            (col("la", &left.schema())?, col("ra", &right.schema())?),
            (col("lb", &left.schema())?, col("rb", &right.schema())?),
        ];
        let flat: Arc<dyn ExecutionPlan> = Arc::new(SortMergeJoinExec::try_new(
            left,
            right,
            on,
            None,
            JoinType::Left,
            vec![order; 2],
            NullEquality::NullEqualsNothing,
        )?);
        // The struct key (a, b).
        let left = struct_key_stream(
            "l",
            left_keys
                .chunks(4096)
                .enumerate()
                .map(|(batch, keys)| struct_key_rows("l", keys, batch as i64 * 4096))
                .collect(),
            order,
            true,
        );
        let right = struct_key_stream(
            "r",
            vec![struct_key_rows("r", &right_keys, 0)],
            order,
            true,
        );
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        let nested: Arc<dyn ExecutionPlan> = Arc::new(SortMergeJoinExec::try_new(
            left,
            right,
            on,
            None,
            JoinType::Left,
            vec![order],
            NullEquality::NullEqualsNothing,
        )?);
        for (name, join) in [("(a, b)", flat), ("struct (a, b)", nested)] {
            let limited: Arc<dyn ExecutionPlan> =
                Arc::new(GlobalLimitExec::new(join, 0, Some(limit)));
            let root = crate::planner::lower_execution_plan(&limited)?;
            assert_streaming_merge_join(&root.children()[0], name);
            let output = run_native(&root, limited, task_ctx()).await?;
            assert_eq!(num_rows(&output.batches), limit, "{name}");
            let batch = &output.batches[0];
            assert_eq!(
                batch.column_by_name("rid").unwrap().null_count(),
                1,
                "{name}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_streams_of_zeros_that_never_end_emit_their_rows()
    -> Result<()> {
        // Streams whose keys all have a `+0.0` that other values follow (a struct
        // `(0.0, i)`, or the keys `(0.0, i)`), which then produce nothing,
        // forever: no row still to come comes before the rows so far, so that
        // the join emits their matches as they come, and a limit above it ends
        // the query.
        let limit = 10;
        type MakeJoin = Box<dyn Fn() -> Result<Arc<dyn ExecutionPlan>>>;
        let joins: [MakeJoin; 2] = [
            Box::new(|| {
                let keys = (0..100).map(|i| (0.0, i)).collect::<Vec<_>>();
                let stream = |side: &str| {
                    let batches = keys
                        .chunks(10)
                        .enumerate()
                        .map(|(batch, keys)| {
                            struct_key_rows(side, keys, batch as i64 * 10)
                        })
                        .collect();
                    struct_key_stream(side, batches, SortOptions::new(false, true), true)
                };
                let (left, right) = (stream("l"), stream("r"));
                let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
                Ok(Arc::new(SortMergeJoinExec::try_new(
                    left,
                    right,
                    on,
                    None,
                    JoinType::Inner,
                    vec![SortOptions::new(false, true)],
                    NullEquality::NullEqualsNothing,
                )?))
            }),
            Box::new(|| {
                let schema = |side: &str| {
                    Arc::new(Schema::new(vec![
                        Field::new(format!("{side}id"), DataType::Int64, false),
                        Field::new(format!("{side}a"), DataType::Float64, false),
                        Field::new(format!("{side}b"), DataType::Int32, false),
                    ])) as SchemaRef
                };
                let stream = |side: &str| -> Arc<dyn ExecutionPlan> {
                    let batches = (0..10)
                        .map(|batch| {
                            let ids = (batch * 10..batch * 10 + 10).collect::<Vec<i64>>();
                            RecordBatch::try_new(
                                schema(side),
                                vec![
                                    Arc::new(Int64Array::from(ids.clone())),
                                    Arc::new(Float64Array::from(vec![0.0; 10])),
                                    Arc::new(Int32Array::from_iter_values(
                                        ids.iter().map(|&id| id as i32),
                                    )),
                                ],
                            )
                            .unwrap()
                        })
                        .collect();
                    let order = LexOrdering::new([
                        PhysicalSortExpr::new(
                            col(&format!("{side}a"), &schema(side)).unwrap(),
                            SortOptions::new(false, true),
                        ),
                        PhysicalSortExpr::new(
                            col(&format!("{side}b"), &schema(side)).unwrap(),
                            SortOptions::new(false, true),
                        ),
                    ])
                    .unwrap();
                    let partition = StreamPartition {
                        schema: schema(side),
                        batches,
                        forever: true,
                    };
                    Arc::new(
                        StreamingTableExec::try_new(
                            schema(side),
                            vec![Arc::new(partition)],
                            None,
                            [order],
                            true,
                            None,
                        )
                        .unwrap(),
                    )
                };
                let (left, right) = (stream("l"), stream("r"));
                let on = vec![
                    (col("la", &left.schema())?, col("ra", &right.schema())?),
                    (col("lb", &left.schema())?, col("rb", &right.schema())?),
                ];
                Ok(Arc::new(SortMergeJoinExec::try_new(
                    left,
                    right,
                    on,
                    None,
                    JoinType::Inner,
                    vec![SortOptions::new(false, true); 2],
                    NullEquality::NullEqualsNothing,
                )?))
            }),
        ];
        for (index, join) in joins.iter().enumerate() {
            let join = join()?;
            let limited: Arc<dyn ExecutionPlan> =
                Arc::new(GlobalLimitExec::new(join, 0, Some(limit)));
            let root = crate::planner::lower_execution_plan(&limited)?;
            assert_streaming_merge_join(&root.children()[0], &index.to_string());
            let output = run_native(&root, limited, task_ctx()).await?;
            assert_eq!(num_rows(&output.batches), limit, "{index}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn null_aware_joins_of_streams_without_order_are_native() -> Result<()> {
        // A `NOT IN` of streams that projects none of its keys has no order
        // (e.g. that of `SELECT count(*) ... WHERE k NOT IN (...)`): it merges
        // its sorted inputs natively all the same (the symmetric join does not
        // implement it), as DataFusion's hash join does.
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}id"), DataType::Int64, false),
                Field::new(format!("{side}k"), DataType::Int32, true),
            ])) as SchemaRef
        };
        let rows = |side: &str, keys: &[Option<i32>]| {
            RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int64Array::from_iter_values(0..keys.len() as i64)),
                    Arc::new(Int32Array::from(keys.to_vec())),
                ],
            )
            .unwrap()
        };
        let stream =
            |side: &str, keys: &[Option<i32>], forever: bool| -> Arc<dyn ExecutionPlan> {
                let key = col(&format!("{side}k"), &schema(side)).unwrap();
                let order = LexOrdering::new([PhysicalSortExpr::new(
                    key,
                    SortOptions::new(false, true),
                )])
                .unwrap();
                let partition = StreamPartition {
                    schema: schema(side),
                    batches: vec![rows(side, keys)],
                    forever,
                };
                Arc::new(
                    StreamingTableExec::try_new(
                        schema(side),
                        vec![Arc::new(partition)],
                        None,
                        [order],
                        true,
                        None,
                    )
                    .unwrap(),
                )
            };
        let left_keys = [None, Some(1), Some(2), Some(3), Some(4), Some(5)];
        for right_keys in [&[Some(2), Some(4)][..], &[None, Some(2)][..], &[][..]] {
            for projection in [Some(vec![]), Some(vec![0])] {
                let context = format!("{right_keys:?} {projection:?}");
                let bounded = |side: &str, keys: &[Option<i32>]| {
                    MemorySourceConfig::try_new_exec(
                        &[vec![rows(side, keys)]],
                        schema(side),
                        None,
                    )
                    .unwrap() as Arc<dyn ExecutionPlan>
                };
                let (left, right) = (bounded("l", &left_keys), bounded("r", right_keys));
                let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
                let reference = Arc::new(HashJoinExec::try_new(
                    left,
                    right,
                    on,
                    None,
                    &JoinType::LeftAnti,
                    projection.clone(),
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    true,
                )?) as Arc<dyn ExecutionPlan>;
                let expected = EngineOutput {
                    schema: reference.schema(),
                    batches: with_timeout(collect(Arc::clone(&reference), task_ctx()))
                        .await?,
                    plan: reference,
                    ordered: false,
                };
                let plan = |forever: bool| -> Result<Arc<dyn ExecutionPlan>> {
                    let (left, right) = (
                        stream("l", &left_keys, forever),
                        stream("r", right_keys, forever),
                    );
                    let on =
                        vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
                    let join = SortMergeJoinExec::try_new(
                        left,
                        right,
                        on,
                        None,
                        JoinType::LeftAnti,
                        vec![SortOptions::new(false, true)],
                        NullEquality::NullEqualsNothing,
                    )?
                    .with_null_aware(true)?
                    .with_projection(projection.clone())?;
                    Ok(Arc::new(join))
                };
                let join = plan(false)?;
                assert!(join.output_ordering().is_none(), "{context}");
                let root = lower_join(&join, false)?;
                assert_streaming_merge_join(&root, &context);
                let actual = run_native(&root, join, task_ctx()).await?;
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}\n{context}");
                }
                // Of streams that never end: a buffered NULL ends the join.
                if right_keys.contains(&None) {
                    let limited: Arc<dyn ExecutionPlan> =
                        Arc::new(GlobalLimitExec::new(plan(true)?, 0, Some(1)));
                    let root = crate::planner::lower_execution_plan(&limited)?;
                    let output = run_native(&root, limited, task_ctx()).await?;
                    assert_eq!(num_rows(&output.batches), 0, "{context}");
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_joins_of_sorted_streams_of_queries_are_native() {
        // Streams sorted by their keys (NULLs first) that never end, for
        // DataFusion: it plans a sort-merge join of their sort-preserving
        // repartitionings, whose partitions it merges for the `ORDER BY`.
        let setup = async |ctx: &SessionContext| {
            for (side, name, rows, skew) in [("l", "s1", 300, 0), ("r", "s2", 500, 3)] {
                let stream = sorted_stream(side, rows, skew, 1, false);
                let stream = stream.downcast_ref::<StreamingTableExec>().unwrap();
                let table = StreamingTable::try_new(
                    data_schema(side),
                    stream.partitions().clone(),
                )?
                .with_infinite_table(true)
                .with_sort_order(vec![
                    datafusion_expr::col(format!("{side}k")).sort(true, true),
                ]);
                ctx.register_table(name, Arc::new(table))?;
            }
            Ok(())
        };
        for sql in [
            "SELECT s1.lk, s1.lid, s2.rid FROM s1 JOIN s2 ON s1.lk = s2.rk ORDER BY s1.lk NULLS FIRST",
            "SELECT s1.lk, s1.lid, s2.rid FROM s1 LEFT JOIN s2 ON s1.lk = s2.rk AND s1.lid < s2.rid ORDER BY s1.lk NULLS FIRST",
            "SELECT s1.lk, s1.lid FROM s1 WHERE EXISTS (SELECT 1 FROM s2 WHERE s2.rk = s1.lk) ORDER BY s1.lk NULLS FIRST",
        ] {
            let (mut expected, mut actual) =
                with_timeout(run_engines(sql, sort_merge_config(4), setup))
                    .await
                    .unwrap_or_else(|e| panic!("{sql}: {e}"));
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
            let mut join = &root;
            while !join.is::<PhysicalMergeJoin>() {
                join = join
                    .children()
                    .first()
                    .unwrap_or_else(|| panic!("no merge join: {sql}\n{tree}"));
            }
            assert_streaming_merge_join(join, sql);
            let ordering = LexOrdering::new([PhysicalSortExpr::new(
                Arc::new(Column::new("lk", 0)),
                SortOptions::new(false, true),
            )])
            .unwrap();
            assert_sorted(&actual.batches, &ordering, sql);
        }
    }

    #[test]
    fn pipelines_of_a_merge_join() -> Result<()> {
        let left = (0..10).map(|i| (i, i as i32)).collect::<Vec<_>>();
        for (join_type, expected) in [
            (
                JoinType::Inner,
                "MetaPipeline 0 (sink: ResultSink) children: [1, 3]\n  \
                 Pipeline 0: MergeJoinSort -> PhysicalMergeJoin -> ResultSink depends on meta pipelines [1, 3]\n\
                 MetaPipeline 1 (sink: PhysicalMergeJoin) children: [2]\n  \
                 Pipeline 1: MergeJoinSort -> PhysicalMergeJoin depends on meta pipelines [2]\n\
                 MetaPipeline 2 (sink: MergeJoinSort)\n  \
                 Pipeline 2: TestSource -> MergeJoinSort\n\
                 MetaPipeline 3 (sink: MergeJoinSort)\n  \
                 Pipeline 3: TestSource -> MergeJoinSort\n",
            ),
            (
                JoinType::Full,
                "MetaPipeline 0 (sink: ResultSink) children: [1, 3]\n  \
                 Pipeline 0: MergeJoinSort -> PhysicalMergeJoin -> ResultSink depends on meta pipelines [1, 3]\n  \
                 Pipeline 4: PhysicalMergeJoin -> ResultSink depends on pipelines [0]\n\
                 MetaPipeline 1 (sink: PhysicalMergeJoin) children: [2]\n  \
                 Pipeline 1: MergeJoinSort -> PhysicalMergeJoin depends on meta pipelines [2]\n\
                 MetaPipeline 2 (sink: MergeJoinSort)\n  \
                 Pipeline 2: TestSource -> MergeJoinSort\n\
                 MetaPipeline 3 (sink: MergeJoinSort)\n  \
                 Pipeline 3: TestSource -> MergeJoinSort\n",
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
