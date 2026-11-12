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
//! inputs, on all threads.
//!
//! # `PhysicalSymmetricHashJoin`
//!
//! The join is an operator of two pipelines, which run at the same time: the
//! pipeline of its left input (in which the join itself is the operator) and
//! the pipeline of its right input (in which a [`SymmetricHashJoinRightInput`]
//! stands for the join), like the pipelines of the inputs of a union. Both
//! continue into the operators above the join and its sink:
//!
//! ```text
//! MetaPipeline 0 (sink: ...)
//!   Pipeline 0: left input -> PhysicalSymmetricHashJoin -> ...
//!   Pipeline 1: right input -> SymmetricHashJoinRightInput -> ...
//!   Pipeline 2: PhysicalSymmetricHashJoin -> ... depends on pipelines [0, 1]
//! ```
//!
//! * Every thread of both pipelines joins its chunks under the lock of the
//!   tables of the join (its global operator state): the rows of a chunk are
//!   joined with the rows of the other input that are in its table (with the
//!   residual filter), and then appended to the table of their own input. So
//!   every pair of rows is joined exactly once, by the thread of the later
//!   one, whatever the threads and the order in which the rows of the two
//!   inputs arrive. The pairs are emitted at once, and so are the rows of a
//!   semi join, when they match for the first time.
//! * Once both inputs ended (pipeline 2, for `Left`, `Right`, `Full`, anti
//!   and mark joins), the join emits the rows that never matched (with NULLs
//!   for the other input), or every row of its input with its mark.
//!
//! The tables keep all rows of both inputs, and reserve their memory in the
//! memory pool of the session: the join fails with `ResourcesExhausted` when
//! the pool refuses them. Unlike DataFusion's, the native join does not prune
//! the rows that the filter of a join of sorted inputs rules out for the rows
//! still to come (DataFusion's interval pruning): it emits the rows without a
//! match once its inputs end, not when they are pruned.
//!
//! The pipelines of the two inputs run at the same time as long as all
//! operators above the join (and its sink) are parallel, as they are for
//! every input whose order does not matter (the order of a symmetric hash
//! join is arbitrary). Join keys are equal with DataFusion's join key
//! equality: `-0.0 = +0.0`, NaNs with the same bits are equal, and NULLs are
//! equal only with [`NullEquality::NullEqualsNull`].
//!
//! [`PhysicalHashJoin`]: crate::join::PhysicalHashJoin

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{ArrayRef, BooleanArray, RecordBatch, UInt32Array};
use arrow::datatypes::{Schema, SchemaRef};
use arrow_row::{RowConverter, SortField};
use datafusion_common::{
    DataFusionError, JoinSide, JoinType, NullEquality, Result, internal_datafusion_err,
    internal_err, plan_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_plan::joins::SymmetricHashJoinExec;
use datafusion_physical_plan::joins::utils::JoinFilter;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use hashbrown::HashMap;
use parking_lot::Mutex;

use super::merge_join::{
    EquiJoin, JoinOutput, JoinRows, StoredRows, evaluate_keys, filter_mask, is_smaller,
    matchable_rows, without_reordering,
};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSourceState, LocalSourceState,
    OperatorResult, OperatorState, PhysicalOperator, SourceResult, downcast_state,
    downcast_state_mut,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::memory_size;

/// Lower a DataFusion `SymmetricHashJoinExec` natively, see the [module
/// documentation](self): to a [`PhysicalHashJoin`] that builds its table on
/// an input that ends, or to a [`PhysicalSymmetricHashJoin`] if both inputs
/// never end.
///
/// `lower_input` lowers an input of the join; its second argument tells
/// whether the input runs in the pipeline of the join (the probe side of a
/// hash join, the inputs of a symmetric hash join) or in a pipeline that
/// reads all of its input (the build side of a hash join).
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
    F: FnMut(&Arc<dyn ExecutionPlan>, bool) -> Result<Arc<dyn PhysicalOperator>>,
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
        return lower_symmetric(&equi_join, &mut lower_input);
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

/// Lower `join`, a join of two inputs that never end, to a
/// [`PhysicalSymmetricHashJoin`] of its inputs, lowered with `lower_input` in
/// the pipelines of the join. `Ok(None)` if the symmetric join does not
/// implement it (see [`try_lower`]).
pub(crate) fn lower_symmetric<F>(
    join: &EquiJoin<'_>,
    lower_input: &mut F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, bool) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(spec) = SymmetricJoinSpec::of(join)? else {
        return Ok(None);
    };
    let mut children = Vec::with_capacity(2);
    for input in [join.left, join.right] {
        let lowered = lower_input(input, true)?;
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

/// A hash join that streams both of its inputs, with the semantics of
/// DataFusion's `SymmetricHashJoinExec`: its join types, null equality and
/// residual filter, and exactly its output schema (and an embedded projection,
/// for the sort-merge joins of inputs that never end).
///
/// See the [module documentation](self) for its pipelines.
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
            key_fields.push(SortField::new(left_type));
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
        Ok(Self {
            output: JoinOutput::try_new(left, right, join_type, projection)?,
            on,
            filter,
            null_equality,
            key_fields,
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

    /// Whether the join emits the rows of input `side` once both inputs
    /// ended: those without a match (outer and anti joins), or all of them
    /// with their mark (mark joins).
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

/// The rows of one input of a symmetric hash join, and their hash table.
#[derive(Debug)]
struct SideTable {
    rows: StoredRows,
    /// The last row of every key (whose keys can match).
    heads: HashMap<Box<[u8]>, usize>,
    /// For every row, the row before it with the same key (`usize::MAX` for
    /// none).
    previous: Vec<usize>,
    /// For every row, whether it matched a row of the other input.
    matched: Vec<bool>,
}

impl SideTable {
    fn new(schema: SchemaRef) -> Self {
        Self {
            rows: StoredRows::new(schema),
            heads: HashMap::new(),
            previous: vec![],
            matched: vec![],
        }
    }
}

/// The global state of a symmetric hash join, shared by the threads of the
/// pipelines of both inputs and by its final source: the tables of both
/// inputs.
struct SymmetricJoinState {
    /// The tables of the left and the right input, until the final source
    /// takes them.
    tables: Mutex<Option<[SideTable; 2]>>,
    /// The memory of the tables.
    reservation: MemoryReservation,
}

impl GlobalOperatorState for SymmetricJoinState {}

/// The state of a thread of the pipeline of an input of the join.
struct SideState {
    /// The input: 0 (left) or 1 (right).
    side: usize,
    join: Arc<SymmetricJoinState>,
    converter: RowConverter,
    /// The output of the input chunk being joined, not emitted yet.
    pending: VecDeque<RecordBatch>,
    /// Whether the input chunk was joined (its output is `pending`).
    joined: bool,
}

impl OperatorState for SideState {}

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

    /// The shared state of the join in the execution of `ctx`.
    fn join_state(&self, ctx: &ExecutionContext) -> Result<Arc<SymmetricJoinState>> {
        let state = ctx.registry().get_or_create_operator_state(self, ctx)?;
        let state: Arc<dyn std::any::Any + Send + Sync> = state;
        state.downcast::<SymmetricJoinState>().map_err(|_| {
            internal_datafusion_err!(
                "PhysicalSymmetricHashJoin received a state of an unexpected type"
            )
        })
    }

    fn side_state(&self, ctx: &ExecutionContext, side: usize) -> Result<SideState> {
        Ok(SideState {
            side,
            join: self.join_state(ctx)?,
            converter: self.spec.key_converter()?,
            pending: VecDeque::new(),
            joined: false,
        })
    }

    /// Join the rows of `input`, a chunk of the input `state.side`, with the
    /// rows of the other input so far, and add them to the table of their
    /// input: the output of the join of these rows, in batches of at most
    /// [`DEFAULT_VECTOR_SIZE`] rows.
    fn join_chunk(
        &self,
        state: &SideState,
        input: &DataChunk,
    ) -> Result<Vec<RecordBatch>> {
        let spec = &self.spec;
        let side = state.side;
        let other = 1 - side;
        let batch = input.compact()?;
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(vec![]);
        }
        let keys = evaluate_keys(spec.keys(side), &batch)?;
        let matchable = matchable_rows(&keys, spec.null_equality);
        let encoded = state.converter.convert_columns(&keys)?;
        let is_matchable =
            |row: usize| matchable.as_ref().is_none_or(|valid| valid.is_valid(row));

        let mut guard = state.join.tables.lock();
        let Some(tables) = guard.as_mut() else {
            return internal_err!(
                "PhysicalSymmetricHashJoin received rows after its end"
            );
        };
        let (own, others) = match side {
            0 => {
                let [left, right] = tables;
                (left, right)
            }
            _ => {
                let [left, right] = tables;
                (right, left)
            }
        };

        // The candidates: the rows of the other input with the same keys.
        let mut pairs_own = Vec::new();
        let mut pairs_other = Vec::new();
        for row in (0..num_rows).filter(|&row| is_matchable(row)) {
            let mut position = others
                .heads
                .get(encoded.row(row).as_ref())
                .copied()
                .unwrap_or(usize::MAX);
            while position != usize::MAX {
                pairs_own.push(row as u32);
                pairs_other.push(Some(position));
                position = others.previous[position];
            }
        }
        let mask = match &spec.filter {
            Some(filter) if !pairs_other.is_empty() => {
                let own_rows = UInt32Array::from(pairs_own.clone());
                let (left, right) = sides(
                    side,
                    JoinRows::Batch(&batch, &own_rows),
                    JoinRows::Stored(&others.rows, &pairs_other),
                );
                Some(filter_mask(filter, &left, &right, pairs_other.len())?)
            }
            _ => None,
        };

        // The matches, and the rows of the other input that match for the
        // first time.
        let mut own_matched = vec![false; num_rows];
        let mut out_own = Vec::new();
        let mut out_other = Vec::new();
        let mut first_matches = Vec::new();
        for (pair, (&row, &position)) in pairs_own.iter().zip(&pairs_other).enumerate() {
            if mask.as_ref().is_some_and(|mask| !mask.value(pair)) {
                continue;
            }
            let Some(position) = position else {
                continue;
            };
            own_matched[row as usize] = true;
            if !others.matched[position] {
                others.matched[position] = true;
                first_matches.push(Some(position));
            }
            if spec.emits_pairs() {
                out_own.push(Some(row));
                out_other.push(Some(position));
            }
        }

        // The rows join the table of their input. (Only the keys that are new
        // to the table are copied into it.)
        let first = own.rows.len();
        let mut key_bytes = 0;
        for row in 0..num_rows {
            let position = first + row;
            let previous = match is_matchable(row) {
                true => {
                    let key = encoded.row(row);
                    match own.heads.get_mut(key.as_ref()) {
                        Some(head) => std::mem::replace(head, position),
                        None => {
                            key_bytes += key.as_ref().len();
                            own.heads.insert(key.as_ref().into(), position);
                            usize::MAX
                        }
                    }
                }
                false => usize::MAX,
            };
            own.previous.push(previous);
        }
        own.matched.extend_from_slice(&own_matched);
        let grown = memory_size(&batch)
            + key_bytes
            + num_rows * (size_of::<usize>() * 3 + size_of::<bool>());
        own.rows.push(batch.clone());
        state.join.reservation.try_grow(grown)?;

        // The output.
        let mut output = Vec::new();
        for (own_rows, other_rows) in out_own
            .chunks(DEFAULT_VECTOR_SIZE)
            .zip(out_other.chunks(DEFAULT_VECTOR_SIZE))
        {
            let own_rows = UInt32Array::from(own_rows.to_vec());
            let (left, right) = sides(
                side,
                JoinRows::Batch(&batch, &own_rows),
                JoinRows::Stored(&others.rows, other_rows),
            );
            output.push(spec.output.batch(&left, &right, None, own_rows.len())?);
        }
        if spec.emits_first_matches(side) {
            let rows = (0..num_rows)
                .filter(|&row| own_matched[row])
                .map(|row| row as u32)
                .collect::<Vec<_>>();
            for rows in rows.chunks(DEFAULT_VECTOR_SIZE) {
                let rows = UInt32Array::from(rows.to_vec());
                let nulls = JoinRows::Nulls(&self.spec.output.schema, 0);
                let (left, right) = sides(side, JoinRows::Batch(&batch, &rows), nulls);
                output.push(spec.output.batch(&left, &right, None, rows.len())?);
            }
            // The own rows that matched are in the table: they are emitted
            // now, and never again.
        }
        if spec.emits_first_matches(other) {
            for positions in first_matches.chunks(DEFAULT_VECTOR_SIZE) {
                let nulls = JoinRows::Nulls(&self.spec.output.schema, 0);
                let (left, right) =
                    sides(other, JoinRows::Stored(&others.rows, positions), nulls);
                output.push(spec.output.batch(&left, &right, None, positions.len())?);
            }
        }
        Ok(output)
    }

    /// The operator role of the join for the input `side`: join the chunks
    /// of that input, see [`Self::join_chunk`].
    fn execute_side(
        &self,
        input: &DataChunk,
        output: &mut DataChunk,
        state: &mut SideState,
    ) -> Result<OperatorResult> {
        if !state.joined {
            state.pending = self.join_chunk(state, input)?.into();
            state.joined = true;
        }
        if let Some(batch) = state.pending.pop_front() {
            *output = DataChunk::from_record_batch(batch);
        }
        if state.pending.is_empty() {
            state.joined = false;
            return Ok(OperatorResult::NeedMoreInput);
        }
        Ok(OperatorResult::HaveMoreOutput)
    }
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

/// The global state of the source of a symmetric hash join, which emits the
/// rows that the join emits once both inputs ended.
struct FinalRowsState {
    tables: [SideTable; 2],
    /// The rows to emit: of the left, then of the right input.
    rows: [Vec<Option<usize>>; 2],
    /// The next chunk of rows to emit, over both inputs.
    next: AtomicUsize,
}

impl GlobalSourceState for FinalRowsState {
    fn max_threads(&self) -> usize {
        self.rows
            .iter()
            .map(|rows| rows.len().div_ceil(DEFAULT_VECTOR_SIZE))
            .sum::<usize>()
            .max(1)
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
        spec.output.describe(
            self.name(),
            &spec.on,
            spec.filter.as_ref(),
            spec.null_equality,
        )
    }

    // ---------------------------------------------------------------------
    // Operator: the pipeline of the left input (the right input has its own
    // operator, `SymmetricHashJoinRightInput`)
    // ---------------------------------------------------------------------

    fn parallel_operator(&self) -> bool {
        true
    }

    fn get_global_operator_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalOperatorState>> {
        let reservation = MemoryConsumer::new("PhysicalSymmetricHashJoin")
            .register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(SymmetricJoinState {
            tables: Mutex::new(Some([
                SideTable::new(self.left().schema()),
                SideTable::new(self.right().schema()),
            ])),
            reservation,
        }))
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(self.side_state(ctx, 0)?))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<SideState>(lstate.as_any_mut(), self.name())?;
        self.execute_side(input, output, state)
    }

    // ---------------------------------------------------------------------
    // Source: the rows emitted once both inputs ended
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        self.spec.emits_final_rows(0) || self.spec.emits_final_rows(1)
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        // Both inputs ended: nothing changes the tables any more.
        let state = self.join_state(ctx)?;
        let Some(tables) = state.tables.lock().take() else {
            return internal_err!(
                "The final rows of PhysicalSymmetricHashJoin were emitted"
            );
        };
        let mark = self.spec.output.is_mark_join();
        let rows = [0, 1].map(|side| {
            if !self.spec.emits_final_rows(side) {
                return vec![];
            }
            let table = &tables[side];
            (0..table.rows.len())
                .filter(|&position| mark || !table.matched[position])
                .map(Some)
                .collect()
        });
        Ok(Arc::new(FinalRowsState {
            tables,
            rows,
            next: AtomicUsize::new(0),
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<FinalRowsState>(gstate.as_any(), self.name())?;
        let chunks = state
            .rows
            .each_ref()
            .map(|rows| rows.len().div_ceil(DEFAULT_VECTOR_SIZE));
        let chunk = state.next.fetch_add(1, Ordering::Relaxed);
        let (side, chunk) = match chunk < chunks[0] {
            true => (0, chunk),
            false if chunk - chunks[0] < chunks[1] => (1, chunk - chunks[0]),
            false => return Ok(SourceResult::Finished),
        };
        let rows = &state.rows[side];
        let start = chunk * DEFAULT_VECTOR_SIZE;
        let positions = &rows[start..(start + DEFAULT_VECTOR_SIZE).min(rows.len())];
        let table = &state.tables[side];
        let other_schema = self.children[1 - side].schema();
        let marks = self.spec.output.is_mark_join().then(|| {
            let marks = positions
                .iter()
                .map(|position| position.is_some_and(|position| table.matched[position]))
                .collect::<Vec<_>>();
            Arc::new(BooleanArray::from(marks)) as ArrayRef
        });
        let (left, right) = sides(
            side,
            JoinRows::Stored(&table.rows, positions),
            JoinRows::Nulls(&other_schema, positions.len()),
        );
        let batch =
            self.spec
                .output
                .batch(&left, &right, marks.as_ref(), positions.len())?;
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)))
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
        let Some(last_pipeline) = meta.pipelines().last().copied() else {
            return internal_err!("A meta pipeline without pipelines");
        };
        // The pipeline of the right input shares the operators above the join
        // and the sink, like the pipelines of the inputs of a union.
        let mut right = meta.create_union_pipeline(current, false)?;
        meta.add_operator(current, Arc::clone(&this))?;
        let right_input: Arc<dyn PhysicalOperator> =
            Arc::new(SymmetricHashJoinRightInput {
                join: Arc::clone(&this),
                schema: self.schema(),
            });
        meta.add_operator(&right, right_input)?;
        meta.build(current, self.left())?;
        meta.build(&mut right, self.right())?;
        // The final rows, once the pipelines of both inputs completed.
        if self.is_source() {
            meta.create_child_pipeline(current, this, &last_pipeline)?;
        }
        Ok(())
    }
}

/// The operator of a [`PhysicalSymmetricHashJoin`] in the pipeline of its
/// right input: it joins the chunks of the right input, in the shared state
/// of the join (the global operator state of the join).
#[derive(Debug)]
pub struct SymmetricHashJoinRightInput {
    /// The join (a [`PhysicalSymmetricHashJoin`]).
    join: Arc<dyn PhysicalOperator>,
    schema: SchemaRef,
}

impl SymmetricHashJoinRightInput {
    fn join(&self) -> Result<&PhysicalSymmetricHashJoin> {
        self.join
            .downcast_ref::<PhysicalSymmetricHashJoin>()
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "SymmetricHashJoinRightInput is the input of a PhysicalSymmetricHashJoin"
                )
            })
    }
}

impl PhysicalOperator for SymmetricHashJoinRightInput {
    fn name(&self) -> &str {
        "SymmetricHashJoinRightInput"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// The input is a child of the join, not of this operator.
    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn parallel_operator(&self) -> bool {
        true
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(self.join()?.side_state(ctx, 1)?))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<SideState>(lstate.as_any_mut(), self.name())?;
        self.join()?.execute_side(input, output, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ops::Range;

    use arrow::array::{Array, Int64Array};
    use arrow::compute::SortOptions;
    use arrow::datatypes::DataType;
    use datafusion::catalog::streaming::StreamingTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::DataFusionError;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_execution::{SendableRecordBatchStream, TaskContext};
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::Partitioning;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::joins::SortMergeJoinExec;
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
    use crate::join::merge_join::tests::{
        ALL_JOIN_TYPES, data_batch, data_schema, join_filter, num_rows, run_native,
        with_timeout,
    };
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
        for (join_type, expected) in [
            (
                JoinType::Inner,
                "MetaPipeline 0 (sink: ResultSink)\n  \
                 Pipeline 0: BridgeOperator -> PhysicalSymmetricHashJoin -> ResultSink\n  \
                 Pipeline 1: BridgeOperator -> SymmetricHashJoinRightInput -> ResultSink\n",
            ),
            (
                JoinType::LeftAnti,
                "MetaPipeline 0 (sink: ResultSink)\n  \
                 Pipeline 0: BridgeOperator -> PhysicalSymmetricHashJoin -> ResultSink\n  \
                 Pipeline 1: BridgeOperator -> SymmetricHashJoinRightInput -> ResultSink\n  \
                 Pipeline 2: PhysicalSymmetricHashJoin -> ResultSink depends on pipelines [0, 1]\n",
            ),
        ] {
            let plan = symmetric_join(
                stream_scan("l", 10, 0, false),
                stream_scan("r", 10, 0, false),
                join_type,
                None,
                NullEquality::NullEqualsNothing,
            )?;
            assert_eq!(format_pipelines(&lower(&plan)?)?, expected, "{join_type}");
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
}
