// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalRangeJoin`]: the joins by a single range predicate
//! (`PiecewiseMergeJoinExec`), which sort their build side and find the rows
//! that every probe row joins with by a binary search.

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    make_comparator, new_empty_array, new_null_array,
};
use arrow::compute::kernels::sort::SortOptions;
use arrow::compute::{concat, concat_batches, filter_record_batch, take};
use arrow::datatypes::SchemaRef;
use datafusion_common::float_semantics::normalize_floats;
use datafusion_common::{JoinType, Result, internal_datafusion_err, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::Operator;
use datafusion_physical_expr_common::physical_expr::PhysicalExprRef;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::joins::PiecewiseMergeJoinExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorResult, OperatorState, PhysicalOperator,
    SinkFinalize, SinkResult, SourceResult, downcast_state, downcast_state_mut,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::memory_size;
use crate::sources::batches::{batch_overhead, chunk_memory};

/// Joins the rows of its build input (the buffered input of a
/// `PiecewiseMergeJoinExec`, its left input) with the rows of its probe
/// input (the streamed input, its right input) for which `build_key <op>
/// probe_key` (e.g. `l.a < r.b`) is true, with the join type of the node, as
/// the `NestedLoopJoinExec` with that filter does.
///
/// The build input sinks into the join, which sorts the build rows with a
/// key that is not null by their keys (in the order of `make_comparator`,
/// with the floats of the keys of both sides, also nested ones, normalized:
/// `-0.0` is `0.0`, as DataFusion compares them). The
/// build rows that a probe row joins with are then a prefix of them (for `<`
/// and `<=`) or a suffix (for `>` and `>=`), which a binary search finds:
/// every thread of the probe side emits the pairs of the rows of its chunks
/// (at most [`DEFAULT_VECTOR_SIZE`] at a time), or their probe rows for the
/// semi, anti and mark joins of the probe side. The build rows that a join
/// emits once it probed all rows (of the outer, semi, anti and mark joins of
/// the build side) are a prefix or a suffix of the sorted rows too (the union
/// of the prefixes or suffixes of all probe rows), which a child pipeline
/// emits once all pipelines of the probe side completed. Rows with a null
/// key join with no row.
///
/// The rows of the build input, their keys, their order, and the build rows
/// that it emits at the end are reserved in the memory pool of the session.
#[derive(Debug)]
pub struct PhysicalRangeJoin {
    /// The probe input, then the build input.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    build_key: PhysicalExprRef,
    probe_key: PhysicalExprRef,
    op: Operator,
    join_type: JoinType,
}

impl PhysicalRangeJoin {
    /// Lower a `PiecewiseMergeJoinExec` to a [`PhysicalRangeJoin`], whose
    /// build input is `lower_build` of its buffered input (without the sort
    /// that DataFusion plans below it), and whose probe input is
    /// `lower_probe` of its streamed input. Returns `Ok(None)` for any other
    /// node, and for keys of different types or of types that Arrow does not
    /// compare (which the nested loop join of the node evaluates), without
    /// lowering anything.
    pub fn try_lower<B, P>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_build: B,
        lower_probe: P,
    ) -> Result<Option<Self>>
    where
        B: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
        P: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(join) = plan.downcast_ref::<PiecewiseMergeJoinExec>() else {
            return Ok(None);
        };
        if !matches!(
            join.operator,
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
        ) {
            return Ok(None);
        }
        let build = match join.buffered().downcast_ref::<SortExec>() {
            Some(sort) if sort.fetch().is_none() => sort.input(),
            _ => join.buffered(),
        };
        let (build_key, probe_key) = &join.on;
        let data_type = build_key.data_type(&build.schema())?;
        if probe_key.data_type(&join.streamed().schema())? != data_type {
            return Ok(None);
        }
        let empty = new_empty_array(&data_type);
        if make_comparator(empty.as_ref(), empty.as_ref(), SortOptions::default())
            .is_err()
        {
            return Ok(None);
        }
        let build = lower_build(build)?;
        let probe = lower_probe(join.streamed())?;
        Ok(Some(Self {
            children: vec![probe, build],
            schema: plan.schema(),
            build_key: Arc::clone(build_key),
            probe_key: Arc::clone(probe_key),
            op: join.operator,
            join_type: join.join_type(),
        }))
    }

    /// The probe input.
    pub fn probe_side(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The build input.
    pub fn build_side(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[1]
    }

    /// Whether the build rows that a probe row joins with are a prefix of
    /// the sorted build rows (`<` and `<=`), rather than a suffix.
    fn prefixes(&self) -> bool {
        matches!(self.op, Operator::Lt | Operator::LtEq)
    }

    /// Whether the join emits build rows once it probed all rows.
    fn emits_final_build_rows(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::Left
                | JoinType::Full
                | JoinType::LeftSemi
                | JoinType::LeftAnti
                | JoinType::LeftMark
        )
    }

    /// The sorted build rows, once the build input is complete.
    fn build_rows(&self, ctx: &ExecutionContext) -> Result<Arc<BuildRows>> {
        let state = ctx
            .global_sink_state(self)
            .and_then(|state| state.downcast_arc::<BuildInput>())
            .ok_or_else(|| {
                internal_datafusion_err!("The build input of {} did not run", self.name())
            })?;
        state.sorted.get().map(Arc::clone).ok_or_else(|| {
            internal_datafusion_err!("The build input of {} is not sorted", self.name())
        })
    }

    /// For every row of `keys` (probe keys), the range of the sorted build
    /// rows of `build` that it joins with (empty for a null key).
    fn ranges(&self, build: &BuildRows, keys: &ArrayRef) -> Result<Vec<(usize, usize)>> {
        let compare =
            make_comparator(build.keys.as_ref(), keys.as_ref(), SortOptions::default())?;
        let order = build.order.as_slice();
        let len = order.len();
        let nulls = keys.logical_nulls();
        Ok((0..keys.len())
            .map(|row| {
                if nulls.as_ref().is_some_and(|nulls| nulls.is_null(row)) {
                    return (0, 0);
                }
                let position = |strict: bool| {
                    order.partition_point(|&build_row| {
                        let ordering = compare(build_row as usize, row);
                        if strict {
                            ordering.is_lt()
                        } else {
                            ordering.is_le()
                        }
                    })
                };
                match self.op {
                    Operator::Lt => (0, position(true)),
                    Operator::LtEq => (0, position(false)),
                    Operator::Gt => (position(false), len),
                    _ => (position(true), len),
                }
            })
            .collect())
    }

    /// The output of the pairs `pairs` (of a build row, or none, and a probe
    /// row of `probe`, or none), with the columns of the build rows and then
    /// those of the probe rows.
    fn pairs(
        &self,
        build: &BuildRows,
        probe: &RecordBatch,
        build_rows: Vec<Option<u32>>,
        probe_rows: Vec<Option<u32>>,
    ) -> Result<RecordBatch> {
        let build_rows = UInt32Array::from(build_rows);
        let probe_rows = UInt32Array::from(probe_rows);
        let mut columns = Vec::with_capacity(self.schema.fields().len());
        for column in build.rows.columns() {
            columns.push(take(column.as_ref(), &build_rows, None)?);
        }
        for column in probe.columns() {
            columns.push(take(column.as_ref(), &probe_rows, None)?);
        }
        self.output(columns, build_rows.len())
    }

    /// The output batch of `columns`, of `rows` rows.
    fn output(&self, columns: Vec<ArrayRef>, rows: usize) -> Result<RecordBatch> {
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &options,
        )?)
    }
}

/// The global sink state of a [`PhysicalRangeJoin`]: the chunks of the build
/// input and their keys, their sorted rows once they are all there, and the
/// build rows that the probe rows joined with.
struct BuildInput {
    chunks: Mutex<Vec<(RecordBatch, ArrayRef)>>,
    sorted: OnceLock<Arc<BuildRows>>,
    reservation: MemoryReservation,
}

impl GlobalSinkState for BuildInput {}

/// The rows of the build input, and those with a key that is not null in
/// the order of their keys.
struct BuildRows {
    rows: RecordBatch,
    keys: ArrayRef,
    /// The indices of the rows of `rows` with a key that is not null, in the
    /// order of their keys.
    order: Vec<u32>,
    /// The indices of the rows of `rows` with a null key.
    null_keys: Vec<u32>,
    /// The end of the prefix (for `<` and `<=`) or the start of the suffix
    /// (for `>` and `>=`) of `order` that the probe rows joined with.
    matched: AtomicUsize,
}

impl BuildRows {
    /// The memory that the rows hold.
    fn memory(&self) -> usize {
        memory_size(&self.rows)
            + self.keys.get_array_memory_size()
            + (self.order.capacity() + self.null_keys.capacity()) * size_of::<u32>()
    }
}

/// The state of a thread of the probe side: the pairs of the chunk that it
/// did not emit yet.
#[derive(Default)]
struct ProbeState {
    pending: Option<PendingPairs>,
}

impl OperatorState for ProbeState {}

/// The pairs of a probe chunk: for every probe row, the range of the sorted
/// build rows that it joins with, and the next pair to emit.
struct PendingPairs {
    probe: RecordBatch,
    ranges: Vec<(usize, usize)>,
    /// The next probe row, and the next build row of its range.
    row: usize,
    offset: usize,
}

/// The global source state of the build rows that a [`PhysicalRangeJoin`]
/// emits once it probed all rows: their indices, and whether they matched
/// (for a mark join), with the next block of them to claim.
struct FinalRows {
    build: Arc<BuildRows>,
    rows: Vec<u32>,
    marks: Option<Vec<bool>>,
    next: AtomicUsize,
    _reservation: MemoryReservation,
}

impl GlobalSourceState for FinalRows {
    fn max_threads(&self) -> usize {
        self.rows.len().div_ceil(DEFAULT_VECTOR_SIZE).max(1)
    }
}

impl PhysicalOperator for PhysicalRangeJoin {
    fn name(&self) -> &str {
        "PhysicalRangeJoin"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!(
            "PhysicalRangeJoin: join_type={}, on=[{} {} {}]",
            self.join_type, self.build_key, self.op, self.probe_key
        )
    }

    // ---------------------------------------------------------------------
    // Sink: the build input
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
        let reservation =
            MemoryConsumer::new("PhysicalRangeJoin").register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(BuildInput {
            chunks: Mutex::new(vec![]),
            sorted: OnceLock::new(),
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
        let state = downcast_state::<BuildInput>(gstate.as_any(), self.name())?;
        // At most the memory of the chunk, before it is compacted.
        let estimate = chunk_memory(input);
        state.reservation.try_grow(estimate)?;
        let batch = input.compact()?;
        let size = memory_size(&batch) + batch_overhead(&batch);
        match size.checked_sub(estimate) {
            Some(more) => state.reservation.try_grow(more)?,
            None => state.reservation.shrink(estimate - size),
        }
        // The keys of the chunk, as large as the chunk at most (with `-0.0`
        // as `0.0`, which DataFusion compares as equal).
        let keys = normalize_floats(
            &self
                .build_key
                .evaluate(&batch)?
                .into_array(batch.num_rows())?,
        );
        if !batch
            .columns()
            .iter()
            .any(|column| Arc::ptr_eq(column, &keys))
        {
            state.reservation.try_grow(keys.get_array_memory_size())?;
        }
        state.chunks.lock().push((batch, keys));
        Ok(SinkResult::NeedMoreInput)
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<BuildInput>(gstate.as_any(), self.name())?;
        let chunks = std::mem::take(&mut *state.chunks.lock());
        let schema = self.build_side().schema();
        let num_rows = chunks
            .iter()
            .map(|(batch, _)| batch.num_rows())
            .sum::<usize>();
        if u32::try_from(num_rows).is_err() {
            return internal_err!(
                "PhysicalRangeJoin joins at most {} build rows, got {num_rows}",
                u32::MAX
            );
        }
        // The concatenations of the rows and of their keys, and their order,
        // before they are allocated (the chunks are released once they are
        // concatenated).
        let held = chunks
            .iter()
            .map(|(batch, keys)| memory_size(batch) + keys.get_array_memory_size())
            .sum::<usize>();
        state
            .reservation
            .try_grow(held + num_rows * size_of::<u32>())?;
        let rows = concat_batches(&schema, chunks.iter().map(|(batch, _)| batch))?;
        let keys = match chunks.as_slice() {
            [] => new_empty_array(&self.build_key.data_type(&schema)?),
            chunks => concat(
                &chunks
                    .iter()
                    .map(|(_, keys)| keys.as_ref())
                    .collect::<Vec<_>>(),
            )?,
        };
        drop(chunks);
        let nulls = keys.logical_nulls();
        let (mut order, null_keys): (Vec<u32>, Vec<u32>) = (0..num_rows as u32)
            .partition(|&row| {
                nulls
                    .as_ref()
                    .is_none_or(|nulls| nulls.is_valid(row as usize))
            });
        let compare =
            make_comparator(keys.as_ref(), keys.as_ref(), SortOptions::default())?;
        order.sort_unstable_by(|&a, &b| compare(a as usize, b as usize));
        let matched = match self.prefixes() {
            true => 0,
            false => order.len(),
        };
        let build = BuildRows {
            rows,
            keys,
            order,
            null_keys,
            matched: AtomicUsize::new(matched),
        };
        state.reservation.try_resize(build.memory())?;
        if state.sorted.set(Arc::new(build)).is_err() {
            return internal_err!("The build input of {} is sorted twice", self.name());
        }
        Ok(SinkFinalize::Ready)
    }

    // ---------------------------------------------------------------------
    // Operator: the probe input
    // ---------------------------------------------------------------------

    fn get_operator_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::<ProbeState>::default())
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<ProbeState>(lstate.as_any_mut(), self.name())?;
        let build = self.build_rows(ctx)?;
        let mut pending = match state.pending.take() {
            Some(pending) => pending,
            None => {
                let probe = input.compact()?;
                let keys = normalize_floats(
                    &self
                        .probe_key
                        .evaluate(&probe)?
                        .into_array(probe.num_rows())?,
                );
                let ranges = self.ranges(&build, &keys)?;
                // The build rows that the probe rows joined with.
                let joined = ranges.iter().filter(|(start, end)| start < end);
                if self.prefixes() {
                    if let Some(end) = joined.map(|&(_, end)| end).max() {
                        build.matched.fetch_max(end, AtomicOrdering::Relaxed);
                    }
                } else if let Some(start) = joined.map(|&(start, _)| start).min() {
                    build.matched.fetch_min(start, AtomicOrdering::Relaxed);
                }
                PendingPairs {
                    probe,
                    ranges,
                    row: 0,
                    offset: 0,
                }
            }
        };
        let probe = &pending.probe;
        match self.join_type {
            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
                // The build rows are emitted once all rows are probed.
                *output = DataChunk::from_record_batch(RecordBatch::new_empty(
                    Arc::clone(&self.schema),
                ));
                Ok(OperatorResult::NeedMoreInput)
            }
            JoinType::RightSemi | JoinType::RightAnti => {
                let semi = self.join_type == JoinType::RightSemi;
                let keep = pending
                    .ranges
                    .iter()
                    .map(|(start, end)| Some((start < end) == semi))
                    .collect::<BooleanArray>();
                let rows = filter_record_batch(probe, &keep)?;
                *output = DataChunk::from_record_batch(
                    self.output(rows.columns().to_vec(), rows.num_rows())?,
                );
                Ok(OperatorResult::NeedMoreInput)
            }
            JoinType::RightMark => {
                let marks = pending
                    .ranges
                    .iter()
                    .map(|(start, end)| Some(start < end))
                    .collect::<BooleanArray>();
                let mut columns = probe.columns().to_vec();
                columns.push(Arc::new(marks));
                *output =
                    DataChunk::from_record_batch(self.output(columns, probe.num_rows())?);
                Ok(OperatorResult::NeedMoreInput)
            }
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                // The probe rows without build rows are emitted with nulls in
                // the joins that keep them.
                let keep_unmatched =
                    matches!(self.join_type, JoinType::Right | JoinType::Full);
                let mut build_rows = Vec::with_capacity(DEFAULT_VECTOR_SIZE);
                let mut probe_rows = Vec::with_capacity(DEFAULT_VECTOR_SIZE);
                while pending.row < pending.ranges.len()
                    && build_rows.len() < DEFAULT_VECTOR_SIZE
                {
                    let (start, end) = pending.ranges[pending.row];
                    if start >= end {
                        if keep_unmatched {
                            build_rows.push(None);
                            probe_rows.push(Some(pending.row as u32));
                        }
                        pending.row += 1;
                        continue;
                    }
                    let from = start + pending.offset;
                    let to = end.min(from + DEFAULT_VECTOR_SIZE - build_rows.len());
                    build_rows.extend(build.order[from..to].iter().map(|&row| Some(row)));
                    probe_rows
                        .extend(std::iter::repeat_n(Some(pending.row as u32), to - from));
                    if to == end {
                        pending.row += 1;
                        pending.offset = 0;
                    } else {
                        pending.offset = to - start;
                    }
                }
                *output = DataChunk::from_record_batch(
                    self.pairs(&build, probe, build_rows, probe_rows)?,
                );
                if pending.row < pending.ranges.len() {
                    state.pending = Some(pending);
                    return Ok(OperatorResult::HaveMoreOutput);
                }
                Ok(OperatorResult::NeedMoreInput)
            }
        }
    }

    // ---------------------------------------------------------------------
    // Source: the build rows emitted once all rows are probed
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        self.emits_final_build_rows()
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let build = self.build_rows(ctx)?;
        let matched = build.matched.load(AtomicOrdering::Relaxed);
        let (joined, not_joined) = match self.prefixes() {
            true => build.order.split_at(matched),
            false => {
                let (not_joined, joined) = build.order.split_at(matched);
                (joined, not_joined)
            }
        };
        let reservation =
            MemoryConsumer::new("PhysicalRangeJoin").register(ctx.task_ctx.memory_pool());
        let (rows, marks) = match self.join_type {
            JoinType::LeftSemi => {
                reservation.try_grow(size_of_val(joined))?;
                (joined.to_vec(), None)
            }
            JoinType::LeftMark => {
                let len = build.rows.num_rows();
                reservation.try_grow(len * (size_of::<u32>() + size_of::<bool>()))?;
                let rows = joined
                    .iter()
                    .chain(not_joined)
                    .chain(&build.null_keys)
                    .copied()
                    .collect::<Vec<_>>();
                let marks = (0..len).map(|index| index < joined.len()).collect();
                (rows, Some(marks))
            }
            _ => {
                let len = not_joined.len() + build.null_keys.len();
                reservation.try_grow(len * size_of::<u32>())?;
                let rows = not_joined
                    .iter()
                    .chain(&build.null_keys)
                    .copied()
                    .collect::<Vec<_>>();
                (rows, None)
            }
        };
        Ok(Arc::new(FinalRows {
            build,
            rows,
            marks,
            next: AtomicUsize::new(0),
            _reservation: reservation,
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<FinalRows>(gstate.as_any(), self.name())?;
        let block = state.next.fetch_add(1, AtomicOrdering::Relaxed);
        let start = block * DEFAULT_VECTOR_SIZE;
        if start >= state.rows.len() {
            return Ok(SourceResult::Finished);
        }
        let end = state.rows.len().min(start + DEFAULT_VECTOR_SIZE);
        let indices = UInt32Array::from(state.rows[start..end].to_vec());
        let mut columns = state
            .build
            .rows
            .columns()
            .iter()
            .map(|column| Ok(take(column.as_ref(), &indices, None)?))
            .collect::<Result<Vec<_>>>()?;
        let len = end - start;
        match self.join_type {
            JoinType::Left | JoinType::Full => {
                // The columns of the probe side, null.
                let probe = self.probe_side().schema();
                columns.extend(
                    probe
                        .fields()
                        .iter()
                        .map(|field| new_null_array(field.data_type(), len)),
                );
            }
            JoinType::LeftMark => {
                let Some(marks) = &state.marks else {
                    return internal_err!("A mark join without marks");
                };
                columns.push(Arc::new(BooleanArray::from(marks[start..end].to_vec())));
            }
            _ => {}
        }
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(
            self.output(columns, len)?,
        )))
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
        // The join probes the build rows in the pipeline of its probe side.
        meta.add_operator(current, Arc::clone(&this))?;
        let Some(last_pipeline) = meta.pipelines().last().copied() else {
            return internal_err!("A meta pipeline without pipelines");
        };
        let build_meta = {
            // The build input sinks into the join, in a child meta pipeline.
            let (mut build, mut build_pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
            build.build(&mut build_pipeline, self.build_side())?;
            build.id()
        };
        // The probe input continues the current pipeline; the pipelines of
        // its pipeline breakers wait for the build input.
        meta.build_with_dependency(current, self.probe_side(), build_meta)?;
        // The build rows emitted once all pipelines of the probe side
        // completed.
        if self.emits_final_build_rows() {
            meta.create_child_pipeline(current, this, &last_pipeline)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::prelude::SessionContext;
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};

    use datafusion_physical_plan::joins::PiecewiseMergeJoinExec;

    use crate::operators::lowering_tests::{
        assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit,
    };
    use crate::test_utils::{register_test_tables, test_session_config};

    async fn setup(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)?;
        let rows = ctx.sql("SELECT id, a, b FROM t1 WHERE id < 400").await?;
        ctx.register_table("s", rows.cache().await?.into_view())?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn range_joins_are_native() {
        let mut config = test_session_config();
        config.options_mut().optimizer.enable_piecewise_merge_join = true;
        // `s`: the first 400 rows of `t1`.
        for sql in [
            "SELECT s.id, t2.k, t2.w FROM s JOIN t2 ON s.a < t2.k",
            "SELECT s.id, t2.k FROM s JOIN t2 ON s.b + 1 >= t2.w * 2",
            "SELECT t2.k, s.id FROM t2 LEFT JOIN s ON t2.w > s.b",
            "SELECT t2.k, s.id FROM t2 RIGHT JOIN s ON t2.w <= s.b",
            "SELECT t2.k, s.id FROM t2 FULL JOIN s ON t2.k > s.a",
            "SELECT k, name FROM t2 WHERE EXISTS (SELECT 1 FROM s WHERE s.a > t2.k * 10)",
            "SELECT k, name FROM t2 WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a > t2.k * 10)",
            "SELECT id FROM s WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.w < s.b)",
            "SELECT id FROM s WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.w < s.b)",
            // Mark joins.
            "SELECT k, name FROM t2 WHERE EXISTS (SELECT 1 FROM s WHERE s.a > t2.k * 10) \
             OR k < 3",
            "SELECT id FROM s WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.w < s.b) OR id < 5",
            // Keys with nulls on both sides, of floats and of strings.
            "SELECT s.id, t1.id FROM s JOIN t1 ON s.a >= t1.a WHERE t1.id < 300",
            "SELECT s.id, t1.id FROM s LEFT JOIN t1 ON s.a < t1.a WHERE s.id < 50",
            "SELECT s.id, t.id FROM s FULL JOIN (SELECT * FROM t1 WHERE id < 100) t \
             ON s.a < t.a - 50",
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.c > t2.w",
            "SELECT t2.k, t1.id FROM t2 JOIN t1 ON t2.name <= t1.d",
            // `-0.0` equals `0.0`, for every comparison, also of nested
            // floats.
            "SELECT l.i, r.j FROM (VALUES (1, CAST(-0.0 AS DOUBLE)), (2, 0.0), (3, 1.0)) \
             AS l(i, x) JOIN (VALUES (1, CAST(0.0 AS DOUBLE)), (2, CAST(-0.0 AS DOUBLE)), \
             (3, -1.0)) AS r(j, y) ON l.x < r.y",
            "SELECT l.i, r.j FROM (VALUES (1, CAST(-0.0 AS DOUBLE)), (2, 0.0), (3, 1.0)) \
             AS l(i, x) JOIN (VALUES (1, CAST(0.0 AS DOUBLE)), (2, CAST(-0.0 AS DOUBLE)), \
             (3, -1.0)) AS r(j, y) ON l.x <= r.y",
            "SELECT l.i, r.j FROM (VALUES (1, CAST(-0.0 AS DOUBLE)), (2, 0.0), (3, 1.0)) \
             AS l(i, x) LEFT JOIN (VALUES (1, CAST(0.0 AS DOUBLE)), \
             (2, CAST(-0.0 AS DOUBLE)), (3, -1.0)) AS r(j, y) ON l.x > r.y",
            "SELECT l.i, r.j FROM (VALUES (1, CAST(-0.0 AS DOUBLE)), (2, 0.0), (3, 1.0)) \
             AS l(i, x) FULL JOIN (VALUES (1, CAST(0.0 AS DOUBLE)), \
             (2, CAST(-0.0 AS DOUBLE)), (3, -1.0)) AS r(j, y) ON l.x >= r.y",
            "SELECT l.i, r.j FROM (VALUES (1, named_struct('f', CAST(-0.0 AS DOUBLE))), \
             (2, named_struct('f', 0.0))) AS l(i, x) JOIN (VALUES \
             (1, named_struct('f', CAST(0.0 AS DOUBLE))), \
             (2, named_struct('f', CAST(-0.0 AS DOUBLE)))) AS r(j, y) ON l.x < r.y",
        ] {
            for target_partitions in [1, 4] {
                let config = config.clone().with_target_partitions(target_partitions);
                let datafusion_plan = {
                    let ctx = crate::test_utils::datafusion_context(config.clone());
                    setup(&ctx).await.unwrap();
                    ctx.sql(sql)
                        .await
                        .unwrap()
                        .create_physical_plan()
                        .await
                        .unwrap()
                };
                let merge_joins = {
                    let mut joins = 0;
                    TreeNode::apply(&datafusion_plan, |node| {
                        joins += usize::from(node.is::<PiecewiseMergeJoinExec>());
                        Ok(TreeNodeRecursion::Continue)
                    })
                    .unwrap();
                    joins
                };
                let root = assert_lowered(sql, config, setup).await;
                assert_eq!(count::<PhysicalRangeJoin>(&root), merge_joins, "{sql}");
                assert!(merge_joins > 0, "{sql}");
            }
        }
    }

    /// A session of the pipeline engine with a memory pool of `limit` bytes,
    /// which plans piecewise merge joins.
    fn context_with_memory_limit(limit: usize) -> SessionContext {
        let ctx = pipeline_context_with_memory_limit(limit);
        let mut state = ctx.state();
        state
            .config_mut()
            .options_mut()
            .optimizer
            .enable_piecewise_merge_join = true;
        SessionContext::new_with_state(state)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn disjoint_range_joins_compare_no_pairs() -> Result<()> {
        // Every probe row finds the build rows it joins with by a binary
        // search: 50,000 × 50,000 rows that join with none (a nested loop
        // join compares 2.5 billion pairs).
        let ctx = context_with_memory_limit(usize::MAX);
        let sql = "SELECT count(*) AS n FROM \
                   (SELECT value AS a FROM generate_series(1, 50000)) l JOIN \
                   (SELECT value + 100000 AS b FROM generate_series(1, 50000)) r \
                   ON l.a > r.b";
        assert_eq!(count::<PhysicalRangeJoin>(&lowered(&ctx, sql).await?), 1);
        let run = async { ctx.sql(sql).await?.collect().await };
        let batches = tokio::time::timeout(std::time::Duration::from_secs(60), run)
            .await
            .expect("a range join of no pairs takes no time")?;
        let n = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(n, 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_of_the_build_input_reserve_memory() -> Result<()> {
        let ctx = context_with_memory_limit(10_000);
        setup(&ctx).await?;
        let sql = "SELECT s.id, t2.k FROM s JOIN t2 ON s.a < t2.k";
        assert_eq!(count::<PhysicalRangeJoin>(&lowered(&ctx, sql).await?), 1);
        assert_resources_exhausted(&ctx, sql).await;
        // In a pool that holds them, nothing stays reserved.
        let ctx = context_with_memory_limit(100 << 20);
        setup(&ctx).await?;
        ctx.sql(sql).await?.collect().await?;
        assert_eq!(ctx.runtime_env().memory_pool.reserved(), 0);
        Ok(())
    }
}
