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
//! The join is the source of its pipeline. Its inputs are executed by
//! executors of their own (see [`NestedInput`]), on the threads of the
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
//!   [`DEFAULT_VECTOR_SIZE`] candidate pairs, each of which is emitted at once
//!   (the pairs that pass the filter, and the rows of a semi join when they
//!   match for the first time): the join never buffers more than a chunk of
//!   candidates, however many rows a row matches, and a limit above it stops
//!   it after the rows it needs.
//! * Once both inputs ended (for `Left`, `Right`, `Full`, anti and mark
//!   joins), the join emits the rows that never matched (with NULLs for the
//!   other input), or every row of its input with its mark.
//!
//! The tables keep all rows of both inputs, and reserve their memory in the
//! memory pool of the session: the join fails with `ResourcesExhausted` when
//! the pool refuses them. Unlike DataFusion's, the native join does not prune
//! the rows that the filter of a join of sorted inputs rules out for the rows
//! still to come (DataFusion's interval pruning): it emits the rows without a
//! match once its inputs end, not when they are pruned. Join keys are equal
//! with DataFusion's join key equality: `-0.0 = +0.0`, NaNs with the same bits
//! are equal, and NULLs are equal only with [`NullEquality::NullEqualsNull`].
//!
//! [`PhysicalHashJoin`]: crate::join::PhysicalHashJoin

use std::sync::Arc;
use std::task::Poll;

use arrow::array::{ArrayRef, BooleanArray, RecordBatch, UInt32Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{Schema, SchemaRef};
use arrow_row::{RowConverter, Rows, SortField};
use datafusion_common::{
    DataFusionError, JoinSide, JoinType, NullEquality, Result, internal_err, plan_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_plan::joins::SymmetricHashJoinExec;
use datafusion_physical_plan::joins::utils::JoinFilter;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use hashbrown::HashMap;
use parking_lot::Mutex;

use super::merge_join::{
    EquiJoin, InputRole, JoinOutput, JoinRows, NestedInput, StoredRows, evaluate_keys,
    filter_mask, is_smaller, matchable_rows, probe_role, without_reordering,
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
        return lower_symmetric(&equi_join, &mut lower_input);
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

/// Lower `join`, a join of two inputs that never end, to a
/// [`PhysicalSymmetricHashJoin`] of its inputs, lowered with `lower_input`
/// as streams. `Ok(None)` if the symmetric join does not implement it (see
/// [`try_lower`]).
pub(crate) fn lower_symmetric<F>(
    join: &EquiJoin<'_>,
    lower_input: &mut F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>, InputRole) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(spec) = SymmetricJoinSpec::of(join)? else {
        return Ok(None);
    };
    let mut children = Vec::with_capacity(2);
    for input in [join.left, join.right] {
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

/// The marker of the end of a chain of rows of a [`SideTable`].
const NO_ROW: usize = usize::MAX;

/// The join reads at most this many chunks of its inputs that produce no
/// output in a call of `get_data` before it lets the other tasks run.
const CHUNKS_PER_CALL: usize = 16;

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
    /// Once both inputs ended: the rows that the join emits then.
    final_rows: Option<FinalRows>,
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

/// The rows that a join emits once both inputs ended.
struct FinalRows {
    /// The rows to emit: of the left, then of the right input.
    rows: [Vec<Option<usize>>; 2],
    /// The next chunk of rows to emit, over both inputs.
    next_chunk: usize,
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

    /// Append the rows of `batch`, a chunk of the input `side`, to the table of
    /// their input, reserving their memory, and start joining them.
    fn start_chunk(
        &self,
        run: &mut SymmetricRun,
        side: usize,
        batch: RecordBatch,
    ) -> Result<ChunkJoin> {
        let keys = evaluate_keys(self.spec.keys(side), &batch)?;
        let matchable = matchable_rows(&keys, self.spec.null_equality);
        let keys = run.converter.convert_columns(&keys)?;
        let num_rows = batch.num_rows();
        let is_matchable =
            |row: usize| matchable.as_ref().is_none_or(|valid| valid.is_valid(row));
        let table = &mut run.tables[side];
        let first = table.rows.len();
        // Only the keys that are new to the table are copied into it.
        let mut key_bytes = 0;
        for row in 0..num_rows {
            let position = first + row;
            let previous = match is_matchable(row) {
                true => {
                    let key = keys.row(row);
                    match table.heads.get_mut(key.as_ref()) {
                        Some(head) => std::mem::replace(head, position),
                        None => {
                            key_bytes +=
                                key.as_ref().len() + size_of::<(Box<[u8]>, usize)>();
                            table.heads.insert(key.as_ref().into(), position);
                            NO_ROW
                        }
                    }
                }
                false => NO_ROW,
            };
            table.previous.push(previous);
        }
        table.matched.resize(first + num_rows, false);
        let bytes = memory_size(&batch)
            + key_bytes
            + num_rows * (size_of::<usize>() + size_of::<bool>());
        run.reservation.try_grow(bytes)?;
        table.rows.push(batch.clone());
        Ok(ChunkJoin {
            side,
            batch,
            keys,
            matchable,
            first,
            next_row: 0,
            chain: None,
        })
    }

    /// The next output of the join of the rows of `job` with the rows of the
    /// other input in `tables`: of at most [`DEFAULT_VECTOR_SIZE`] candidate
    /// pairs (a row without candidates counts as one), with the filter
    /// evaluated on all of them at once. `None` if they produce no output.
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
        let num_rows = job.batch.num_rows();
        let mut pairs_own = Vec::new();
        let mut pairs_other = Vec::new();
        let mut cost = 0;
        while cost < DEFAULT_VECTOR_SIZE {
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
            while position != NO_ROW && cost < DEFAULT_VECTOR_SIZE {
                pairs_own.push(row as u32);
                pairs_other.push(Some(position));
                position = other.previous[position];
                cost += 1;
            }
            if position != NO_ROW {
                job.chain = Some((row, position));
                break;
            }
        }
        if pairs_other.is_empty() {
            return Ok(None);
        }
        // The candidates, and the rows gathered for the filter and the output
        // (at most a chunk of them), are reserved while they are joined.
        let row_bytes = |schema: &SchemaRef| {
            schema
                .fields()
                .iter()
                .map(|field| field.data_type().primitive_width().unwrap_or(16))
                .sum::<usize>()
        };
        let transient = reservation.new_empty();
        transient.try_grow(
            pairs_other.len()
                * (size_of::<u32>()
                    + size_of::<Option<usize>>()
                    + row_bytes(&self.spec.output.schema)
                    + spec
                        .filter
                        .as_ref()
                        .map_or(0, |filter| row_bytes(filter.schema()))),
        )?;
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
        for (pair, (&row, &position)) in
            own_rows.values().iter().zip(&pairs_other).enumerate()
        {
            if mask.as_ref().is_some_and(|mask| !mask.value(pair)) {
                continue;
            }
            let Some(position) = position else {
                continue;
            };
            let own_position = job.first + row as usize;
            if !own.matched[own_position] {
                own.matched[own_position] = true;
                first_own.push(row);
            }
            if !other.matched[position] {
                other.matched[position] = true;
                first_other.push(Some(position));
            }
            if spec.emits_pairs() {
                out_own.push(Some(row));
                out_other.push(Some(position));
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

    /// The next chunk of the rows that the join emits once both inputs
    /// ended: the rows without a match (with NULLs for the other input) of
    /// the outer and anti joins, every row with its mark for the mark joins.
    fn next_final_rows(&self, run: &mut SymmetricRun) -> Result<SourceResult> {
        let mark = self.spec.output.is_mark_join();
        let final_rows = run.final_rows.get_or_insert_with(|| FinalRows {
            rows: [0, 1].map(|side| {
                if !self.spec.emits_final_rows(side) {
                    return vec![];
                }
                let table = &run.tables[side];
                (0..table.rows.len())
                    .filter(|&position| mark || !table.matched[position])
                    .map(Some)
                    .collect()
            }),
            next_chunk: 0,
        });
        let chunks = final_rows
            .rows
            .each_ref()
            .map(|rows| rows.len().div_ceil(DEFAULT_VECTOR_SIZE));
        let chunk = final_rows.next_chunk;
        let (side, chunk) = match chunk < chunks[0] {
            true => (0, chunk),
            false if chunk - chunks[0] < chunks[1] => (1, chunk - chunks[0]),
            false => return Ok(SourceResult::Finished),
        };
        final_rows.next_chunk += 1;
        let rows = &final_rows.rows[side];
        let start = chunk * DEFAULT_VECTOR_SIZE;
        let positions = &rows[start..(start + DEFAULT_VECTOR_SIZE).min(rows.len())];
        let table = &run.tables[side];
        let other_schema = self.children[1 - side].schema();
        let marks = mark.then(|| {
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

    /// The next output of the join: joins the chunk being joined, reads the
    /// next chunk of an input (alternately, of the inputs that have one), and
    /// emits the final rows once both inputs ended. `Blocked` while neither
    /// input has a chunk (the waker of the task of `ctx` is registered with
    /// both).
    fn next_output(
        &self,
        ctx: &ExecutionContext,
        run: &mut SymmetricRun,
    ) -> Result<SourceResult> {
        let mut chunks = 0;
        loop {
            if let Some(job) = &mut run.job {
                let output =
                    self.next_chunk_output(job, &mut run.tables, &run.reservation)?;
                if job.is_done() {
                    run.job = None;
                }
                if let Some(batch) = output {
                    return Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)));
                }
                continue;
            }
            if run.inputs.iter().all(NestedInput::is_done) {
                return self.next_final_rows(run);
            }
            if chunks == CHUNKS_PER_CALL {
                return Ok(SourceResult::Blocked(TaskInterrupt::yield_now()));
            }
            let mut read = false;
            for side in [run.next_side, 1 - run.next_side] {
                if run.inputs[side].is_done() {
                    continue;
                }
                match run.inputs[side].poll_next(ctx) {
                    Poll::Ready(Ok(Some(batch))) => {
                        if batch.num_rows() > 0 {
                            run.job = Some(self.start_chunk(run, side, batch)?);
                        }
                        run.next_side = 1 - side;
                        chunks += 1;
                        read = true;
                        break;
                    }
                    Poll::Ready(Ok(None)) => read = true,
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
        spec.output.describe(
            self.name(),
            &spec.on,
            spec.filter.as_ref(),
            spec.null_equality,
        )
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
            final_rows: None,
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
    /// their own (see [`NestedInput`]), not pipelines of this execution.
    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        meta.build_source(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{Array, Int64Array};
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field};
    use datafusion::catalog::streaming::StreamingTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::internal_datafusion_err;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_execution::{SendableRecordBatchStream, TaskContext};
    use datafusion_expr::ColumnarValue;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::expressions::{BinaryExpr, lit};
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
    use crate::join::merge_join::tests::{
        ALL_JOIN_TYPES, data_batch, data_schema, join_filter, limited_task_ctx, num_rows,
        run_native, with_timeout,
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
        // within the same pool.
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
                    Arc::new(arrow::array::Int32Array::from(vec![7; 500])),
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
}
