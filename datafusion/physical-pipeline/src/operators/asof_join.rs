// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalAsOfJoin`]: the ASOF join (`AsOfJoinExec`), which joins every
//! left row with the closest eligible right row.

use std::cmp::Ordering;
use std::sync::{Arc, OnceLock};

use arrow::array::{
    Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array, make_comparator,
    new_empty_array,
};
use arrow::buffer::NullBuffer;
use arrow::compute::kernels::sort::SortOptions;
use arrow::compute::{concat, concat_batches, take};
use arrow::datatypes::SchemaRef;
use arrow::row::{Row, RowConverter, Rows, SortField};
use datafusion_common::float_semantics::normalize_floats;
use datafusion_common::{
    JoinSide, Result, internal_datafusion_err, internal_err, not_impl_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::Operator;
use datafusion_physical_expr_common::physical_expr::PhysicalExprRef;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::joins::AsOfJoinExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use parking_lot::Mutex;

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, LocalSinkState,
    OperatorResult, OperatorState, PhysicalOperator, SinkFinalize, SinkResult,
    downcast_state,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::memory_size;
use crate::sources::batches::chunk_memory;

/// Joins every row of its left input with the closest eligible row of its
/// right input, as `AsOfJoinExec` does (`ASOF JOIN ... MATCH_CONDITION (l.t
/// >= r.t) ON l.k = r.k`).
///
/// The right rows that are eligible for a left row have its equality keys
/// (all of them not null) and a match value (not null) that satisfies the
/// match condition with the left row's match value; the closest of them is
/// the one with the greatest match value for `>=` and `>`, and the smallest
/// for `<=` and `<`. A left row without an eligible right row, or with a null
/// key or match value, is joined with nulls. Match values compare as
/// DataFusion's ASOF join compares them (Arrow's order, `-0.0` equal to
/// `0.0`); which of the eligible right rows with the same closest match value
/// a left row is joined with is not defined, as in DataFusion.
///
/// The right input is the build side: its rows sink into the join, which
/// then sorts the eligible ones by their equality keys (in the Arrow row
/// format, with `-0.0` as `0.0`, as DataFusion compares them), and then by
/// their match values. The left input is the probe side: every thread finds
/// the closest right row of every left row of its chunks by a binary search,
/// and emits the left rows in the order of its chunks, with their closest
/// right rows. The order of the right
/// input, which DataFusion sorts for its merge, does not matter (it is
/// lowered without its sort). The order of a left input of a single
/// partition is kept, as in DataFusion; a left input of several partitions,
/// which DataFusion sorts every partition of for its merge, is lowered
/// without these sorts. The rows of the right input, their keys and match
/// values, and the order of the eligible ones, are reserved in the memory
/// pool of the session, as they arrive and before they are concatenated and
/// sorted.
#[derive(Debug)]
pub struct PhysicalAsOfJoin {
    /// The left input (the probe side), then the right input (the build
    /// side).
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    left_keys: Vec<PhysicalExprRef>,
    right_keys: Vec<PhysicalExprRef>,
    left_match: PhysicalExprRef,
    right_match: PhysicalExprRef,
    op: Operator,
    /// For every output column: its input and its index in that input.
    columns: Vec<(JoinSide, usize)>,
    /// The row converter of the equality keys, `None` without keys.
    converter: Option<RowConverter>,
}

impl PhysicalAsOfJoin {
    /// Lower an `AsOfJoinExec` to a [`PhysicalAsOfJoin`], whose left input is
    /// `lower_left` of the left input of the node, and whose right input is
    /// `lower_right` of the right input of the node, without the sorts and
    /// merges (without a fetch) at its top, which order it for DataFusion's
    /// merge. Returns `Ok(None)` for any other node, without lowering
    /// anything.
    pub fn try_lower<L, R>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_left: L,
        lower_right: R,
    ) -> Result<Option<Self>>
    where
        L: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
        R: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(join) = plan.downcast_ref::<AsOfJoinExec>() else {
            return Ok(None);
        };
        let condition = join.match_condition();
        if !matches!(
            condition.op,
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
        ) {
            return not_impl_err!(
                "PhysicalAsOfJoin does not compare with {}",
                condition.op
            );
        }
        // The columns of the join: the left columns, then the right columns.
        let left_columns = join.left().schema().fields().len();
        let right_columns = join.right().schema().fields().len();
        let join_columns = (0..left_columns)
            .map(|index| (JoinSide::Left, index))
            .chain((0..right_columns).map(|index| (JoinSide::Right, index)))
            .collect::<Vec<_>>();
        let columns = match join.projection() {
            Some(projection) => projection
                .iter()
                .map(|&index| {
                    join_columns.get(index).copied().ok_or_else(|| {
                        internal_datafusion_err!(
                            "AsOfJoinExec projects column {index} of {} columns",
                            join_columns.len()
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            None => join_columns,
        };
        let (left_keys, right_keys): (Vec<_>, Vec<_>) = join.on().iter().cloned().unzip();
        // The keys of both sides are compared in the row format of the
        // types of the right keys.
        let left_schema = join.left().schema();
        let right_schema = join.right().schema();
        let mut fields = Vec::with_capacity(right_keys.len());
        for (left, right) in left_keys.iter().zip(&right_keys) {
            let data_type = right.data_type(&right_schema)?;
            if left.data_type(&left_schema)? != data_type {
                return not_impl_err!(
                    "PhysicalAsOfJoin compares keys of the same type, got {left} and {right}"
                );
            }
            fields.push(SortField::new(data_type));
        }
        let converter = match fields.is_empty() {
            true => None,
            false => Some(RowConverter::new(fields)?),
        };
        // Of a single partition, the output is ordered as the left input. Of
        // several, DataFusion orders every partition for its merge, which no
        // native operator above depends on (they order or partition the rows
        // of several partitions themselves).
        let left = match join.left().output_partitioning().partition_count() {
            1 => join.left(),
            _ => unordered(join.left()),
        };
        let left = lower_left(left)?;
        let right = lower_right(unordered(join.right()))?;
        Ok(Some(Self {
            children: vec![left, right],
            schema: plan.schema(),
            left_keys,
            right_keys,
            left_match: Arc::clone(&condition.left),
            right_match: Arc::clone(&condition.right),
            op: condition.op,
            columns,
            converter,
        }))
    }

    /// The left input, the probe side.
    pub fn left(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The right input, the build side.
    pub fn right(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[1]
    }

    /// The sorted rows of the right input, once it is complete.
    fn right_rows(&self, ctx: &ExecutionContext) -> Result<Arc<SortedRightRows>> {
        let state = ctx
            .global_sink_state(self)
            .and_then(|state| state.downcast_arc::<RightInput>())
            .ok_or_else(|| {
                internal_datafusion_err!("The right input of {} did not run", self.name())
            })?;
        state.sorted.get().map(Arc::clone).ok_or_else(|| {
            internal_datafusion_err!("The right input of {} is not sorted", self.name())
        })
    }

    /// The equality keys `keys` and the match value `matches` over `batch`
    /// (with `-0.0` as `0.0`, as DataFusion compares them), the keys in the
    /// row format, and which rows have neither a null key nor a null match
    /// value (`None` if all of them).
    fn evaluate(
        &self,
        keys: &[PhysicalExprRef],
        matches: &PhysicalExprRef,
        batch: &RecordBatch,
    ) -> Result<(Option<Rows>, ArrayRef, Option<NullBuffer>)> {
        let mut arrays = keys
            .iter()
            .map(|key| {
                Ok(normalize_floats(
                    &key.evaluate(batch)?.into_array(batch.num_rows())?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let key_rows = match &self.converter {
            Some(converter) => Some(converter.convert_columns(&arrays)?),
            None => None,
        };
        let matches =
            normalize_floats(&matches.evaluate(batch)?.into_array(batch.num_rows())?);
        arrays.push(Arc::clone(&matches));
        let eligible = arrays.iter().fold(None, |validity, array| {
            NullBuffer::union(validity.as_ref(), array.logical_nulls().as_ref())
        });
        Ok((key_rows, matches, eligible))
    }
}

/// `plan` without the sorts and merges without a fetch at its top, and the
/// `CoalescePartitionsExec`s without a fetch between them: they only order
/// (or merge) its rows, which the native join, which sorts the right rows
/// itself, does not need.
fn unordered(plan: &Arc<dyn ExecutionPlan>) -> &Arc<dyn ExecutionPlan> {
    let mut node = plan;
    loop {
        if let Some(sort) = node.downcast_ref::<SortExec>()
            && sort.fetch().is_none()
        {
            node = sort.input();
        } else if let Some(merge) = node.downcast_ref::<SortPreservingMergeExec>()
            && merge.fetch().is_none()
        {
            node = merge.input();
        } else if let Some(coalesce) = node.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
        {
            node = coalesce.input();
        } else {
            return node;
        }
    }
}

/// The global sink state of a [`PhysicalAsOfJoin`]: the chunks of the right
/// input, and their sorted rows once they are all there.
struct RightInput {
    chunks: Mutex<Vec<RightChunk>>,
    sorted: OnceLock<Arc<SortedRightRows>>,
    reservation: MemoryReservation,
}

impl GlobalSinkState for RightInput {}

/// A chunk of the right input: its rows, their equality keys in the row
/// format, their match values, and which of them are eligible (have neither
/// a null key nor a null match value, `None` if all of them).
struct RightChunk {
    batch: RecordBatch,
    keys: Option<Rows>,
    matches: ArrayRef,
    eligible: Option<NullBuffer>,
}

impl RightChunk {
    /// The number of eligible rows.
    fn eligible_rows(&self) -> usize {
        match &self.eligible {
            Some(eligible) => eligible.len() - eligible.null_count(),
            None => self.batch.num_rows(),
        }
    }

    /// The memory that the chunk holds besides its rows: its key rows, its
    /// match values (if they are not a column of the chunk) and which rows
    /// are eligible.
    fn derived_memory(&self) -> usize {
        let matches = if self
            .batch
            .columns()
            .iter()
            .any(|column| Arc::ptr_eq(column, &self.matches))
        {
            0
        } else {
            self.matches.get_array_memory_size()
        };
        self.keys.as_ref().map_or(0, Rows::size)
            + matches
            + self
                .eligible
                .as_ref()
                .map_or(0, |eligible| eligible.buffer().capacity())
    }
}

/// The rows of the right input, and the eligible ones in the order of their
/// equality keys (in the row format), then of their match values.
struct SortedRightRows {
    rows: RecordBatch,
    /// The equality keys of `rows` in the row format, `None` without keys.
    keys: Option<Rows>,
    /// The match values of `rows`, with `-0.0` as `0.0`.
    matches: ArrayRef,
    /// The indices of the eligible rows of `rows`, sorted.
    order: Vec<u32>,
}

impl SortedRightRows {
    /// The memory that the rows hold.
    fn memory(&self) -> usize {
        memory_size(&self.rows)
            + self.keys.as_ref().map_or(0, Rows::size)
            + self.matches.get_array_memory_size()
            + self.order.capacity() * size_of::<u32>()
    }

    /// Compare the equality keys of the right row `row` with `key`.
    fn compare_keys(&self, row: u32, key: Option<Row<'_>>) -> Ordering {
        match (&self.keys, key) {
            (Some(keys), Some(key)) => keys.row(row as usize).cmp(&key),
            _ => Ordering::Equal,
        }
    }
}

impl PhysicalOperator for PhysicalAsOfJoin {
    fn name(&self) -> &str {
        "PhysicalAsOfJoin"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let on = self
            .left_keys
            .iter()
            .zip(&self.right_keys)
            .map(|(left, right)| format!("({left} = {right})"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "PhysicalAsOfJoin: on=[{on}], match=[{} {} {}]",
            self.left_match, self.op, self.right_match
        )
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
            MemoryConsumer::new("PhysicalAsOfJoin").register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(RightInput {
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
        let state = downcast_state::<RightInput>(gstate.as_any(), self.name())?;
        // At most the memory of the chunk, before it is compacted.
        let estimate = chunk_memory(input);
        state.reservation.try_grow(estimate)?;
        let batch = input.compact()?;
        let size = memory_size(&batch);
        match size.checked_sub(estimate) {
            Some(more) => state.reservation.try_grow(more)?,
            None => state.reservation.shrink(estimate - size),
        }
        // The keys, match values and eligible rows of the chunk, which are
        // as large as the chunk at most, and which it keeps.
        let (keys, matches, eligible) =
            self.evaluate(&self.right_keys, &self.right_match, &batch)?;
        let chunk = RightChunk {
            batch,
            keys,
            matches,
            eligible,
        };
        state.reservation.try_grow(chunk.derived_memory())?;
        state.chunks.lock().push(chunk);
        Ok(SinkResult::NeedMoreInput)
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<RightInput>(gstate.as_any(), self.name())?;
        let chunks = std::mem::take(&mut *state.chunks.lock());
        let schema = self.right().schema();
        let num_rows = chunks
            .iter()
            .map(|chunk| chunk.batch.num_rows())
            .sum::<usize>();
        if u32::try_from(num_rows).is_err() {
            return not_impl_err!(
                "PhysicalAsOfJoin joins at most {} right rows, got {num_rows}",
                u32::MAX
            );
        }
        let eligible_rows = chunks.iter().map(RightChunk::eligible_rows).sum::<usize>();
        // The concatenations of the rows, of their keys and of their match
        // values, and the indices of the eligible rows, before they are
        // allocated (the chunks are released once they are concatenated).
        let held = chunks
            .iter()
            .map(|chunk| memory_size(&chunk.batch) + chunk.derived_memory())
            .sum::<usize>();
        state
            .reservation
            .try_grow(held + eligible_rows * size_of::<u32>())?;
        let rows = concat_batches(&schema, chunks.iter().map(|chunk| &chunk.batch))?;
        let matches = match chunks.as_slice() {
            [] => new_empty_array(&self.right_match.data_type(&schema)?),
            chunks => concat(
                &chunks
                    .iter()
                    .map(|chunk| chunk.matches.as_ref())
                    .collect::<Vec<_>>(),
            )?,
        };
        let keys = self.converter.as_ref().map(|converter| {
            let data = chunks
                .iter()
                .filter_map(|chunk| chunk.keys.as_ref())
                .map(Rows::size)
                .sum::<usize>();
            let mut keys = converter.empty_rows(num_rows, data);
            for chunk in &chunks {
                for key in chunk.keys.iter().flat_map(|keys| keys.iter()) {
                    keys.push(key);
                }
            }
            keys
        });
        let mut order = Vec::with_capacity(eligible_rows);
        let mut offset = 0;
        for chunk in &chunks {
            let len = chunk.batch.num_rows();
            match &chunk.eligible {
                Some(eligible) => order
                    .extend(eligible.valid_indices().map(|row| (offset + row) as u32)),
                None => order.extend((offset..offset + len).map(|row| row as u32)),
            }
            offset += len;
        }
        drop(chunks);
        let compare =
            make_comparator(matches.as_ref(), matches.as_ref(), SortOptions::default())?;
        let mut sorted = SortedRightRows {
            rows,
            keys,
            matches,
            order: vec![],
        };
        // Sorted in place, by the keys and then by the match values.
        order.sort_unstable_by(|&a, &b| {
            let key = sorted.keys.as_ref().map(|keys| keys.row(b as usize));
            sorted
                .compare_keys(a, key)
                .then_with(|| compare(a as usize, b as usize))
        });
        sorted.order = order;
        state.reservation.try_resize(sorted.memory())?;
        if state.sorted.set(Arc::new(sorted)).is_err() {
            return internal_err!("The right input of {} is sorted twice", self.name());
        }
        Ok(SinkFinalize::Ready)
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let right = self.right_rows(ctx)?;
        let left = input.compact()?;
        let (keys, matches, eligible) =
            self.evaluate(&self.left_keys, &self.left_match, &left)?;
        // Compares a right match value with a left match value.
        let compare = make_comparator(
            right.matches.as_ref(),
            matches.as_ref(),
            SortOptions::default(),
        )?;
        let order = right.order.as_slice();
        let mut closest = Vec::with_capacity(left.num_rows());
        for row in 0..left.num_rows() {
            if eligible
                .as_ref()
                .is_some_and(|eligible| eligible.is_null(row))
            {
                closest.push(None);
                continue;
            }
            let key = keys.as_ref().map(|keys| keys.row(row));
            // The eligible right rows are sorted by their keys, then by their
            // match values: the rows with the key of the left row, and a
            // match value that satisfies the match condition, are the last
            // ones before (`>=` and `>`) or the first ones after (`<=` and
            // `<`) the position of the left row.
            let ordering = |&right_row: &u32| {
                right
                    .compare_keys(right_row, key)
                    .then_with(|| compare(right_row as usize, row))
            };
            let index = match self.op {
                Operator::GtEq => order
                    .partition_point(|right| ordering(right).is_le())
                    .checked_sub(1),
                Operator::Gt => order
                    .partition_point(|right| ordering(right).is_lt())
                    .checked_sub(1),
                Operator::LtEq => {
                    Some(order.partition_point(|right| ordering(right).is_lt()))
                }
                Operator::Lt => {
                    Some(order.partition_point(|right| ordering(right).is_le()))
                }
                op => {
                    return internal_err!("PhysicalAsOfJoin does not compare with {op}");
                }
            };
            closest.push(
                index
                    .and_then(|index| order.get(index).copied())
                    .filter(|&right_row| right.compare_keys(right_row, key).is_eq()),
            );
        }
        let closest = UInt32Array::from(closest);
        let columns = self
            .columns
            .iter()
            .map(|&(side, index)| match side {
                JoinSide::Left => Ok(Arc::clone(left.column(index))),
                JoinSide::Right => Ok(take(right.rows.column(index), &closest, None)?),
                JoinSide::None => {
                    internal_err!("PhysicalAsOfJoin does not emit a mark column")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(left.num_rows()));
        *output = DataChunk::from_record_batch(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &options,
        )?);
        Ok(OperatorResult::NeedMoreInput)
    }

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        let this = meta.current_operator()?;
        // The join is an operator of the pipeline of its left input.
        meta.add_operator(current, Arc::clone(&this))?;
        let right = {
            // The right input sinks into the join, in a child meta pipeline.
            let (mut right, mut pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
            right.build(&mut pipeline, self.right())?;
            right.id()
        };
        // The left input continues the current pipeline; the pipelines of its
        // pipeline breakers wait for the right input.
        meta.build_with_dependency(current, self.left(), right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::prelude::SessionContext;

    use crate::operators::lowering_tests::{
        assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit,
    };
    use crate::test_utils::test_session_config;

    /// `trades`: ids, symbols (with nulls) and timestamps (with nulls), in
    /// three partitions; `quotes`: symbols (with nulls), timestamps that are
    /// unique for every symbol (with nulls), and prices; `no_quotes`: the
    /// schema of `quotes` without rows.
    async fn register_tables(ctx: &SessionContext) -> Result<()> {
        use arrow::array::{Int64Array, StringArray, TimestampSecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use datafusion::datasource::MemTable;

        let symbol = |i: i64| (i % 7 != 0).then(|| format!("s{}", (i % 7) % 4));
        let ts = |i: i64, step: i64| (i % 11 != 3).then_some(1_700_000_000 + i * step);
        let timestamp = DataType::Timestamp(TimeUnit::Second, None);
        let trades_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("sym", DataType::Utf8, true),
            Field::new("ts", timestamp.clone(), true),
        ]));
        let trades = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            Ok(RecordBatch::try_new(
                Arc::clone(&trades_schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids.clone())),
                    Arc::new(StringArray::from_iter(ids.clone().map(symbol))),
                    Arc::new(TimestampSecondArray::from_iter(ids.map(|i| ts(i, 5)))),
                ],
            )?)
        };
        let table = MemTable::try_new(
            Arc::clone(&trades_schema),
            vec![vec![trades(0..250)?], vec![], vec![trades(250..600)?]],
        )?;
        ctx.register_table("trades", Arc::new(table))?;
        let quotes_schema = Arc::new(Schema::new(vec![
            Field::new("sym", DataType::Utf8, true),
            Field::new("ts", timestamp, true),
            Field::new("price", DataType::Int64, false),
        ]));
        let ids = 0..900;
        let quotes = RecordBatch::try_new(
            Arc::clone(&quotes_schema),
            vec![
                Arc::new(StringArray::from_iter(ids.clone().map(symbol))),
                Arc::new(TimestampSecondArray::from_iter(
                    ids.clone().map(|i| ts(i, 3)),
                )),
                Arc::new(Int64Array::from_iter_values(ids.map(|i| (i * 37) % 1000))),
            ],
        )?;
        let table = MemTable::try_new(
            Arc::clone(&quotes_schema),
            vec![vec![quotes.slice(0, 500)], vec![quotes.slice(500, 400)]],
        )?;
        ctx.register_table("quotes", Arc::new(table))?;
        ctx.register_table(
            "no_quotes",
            Arc::new(MemTable::try_new(quotes_schema, vec![vec![]])?),
        )?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn asof_joins_are_native() {
        for sql in [
            // Every operator, with equality keys.
            "SELECT t.id, t.ts, q.ts AS qts, q.price FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.ts >= q.ts) ON t.sym = q.sym",
            "SELECT t.id, q.ts AS qts, q.price FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.ts > q.ts) ON t.sym = q.sym",
            "SELECT t.id, q.ts AS qts, q.price FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.ts <= q.ts) ON t.sym = q.sym",
            "SELECT t.id, q.ts AS qts, q.price FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.ts < q.ts) ON t.sym = q.sym",
            // Without equality keys (over the rows of a single symbol, whose
            // timestamps are unique).
            "SELECT t.id, q.price FROM trades t ASOF JOIN \
             (SELECT * FROM quotes WHERE sym = 's1') q MATCH_CONDITION (t.ts >= q.ts)",
            // Ordered, and below an aggregation.
            "SELECT t.id, t.sym, q.price FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.ts >= q.ts) ON t.sym = q.sym ORDER BY t.id",
            "SELECT q.price IS NULL AS unmatched, COUNT(*) AS n FROM trades t \
             ASOF JOIN quotes q MATCH_CONDITION (t.ts <= q.ts) ON t.sym = q.sym \
             GROUP BY q.price IS NULL",
            // Of a right input of no rows.
            "SELECT t.id, q.price FROM trades t ASOF JOIN no_quotes q \
             MATCH_CONDITION (t.ts >= q.ts) ON t.sym = q.sym",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, register_tables).await;
                assert_eq!(count::<PhysicalAsOfJoin>(&root), 1, "{sql}");
            }
        }
    }

    /// The tables of DataFusion's `asof_join.slt`.
    async fn register_slt_tables(ctx: &SessionContext) -> Result<()> {
        for sql in [
            "CREATE TABLE asof_left(id INT, grp TEXT, ts TIMESTAMP) AS VALUES \
             (1, 'A', TIMESTAMP '2024-01-01 09:00:01'), \
             (2, 'A', TIMESTAMP '2024-01-01 09:00:04'), \
             (3, 'A', TIMESTAMP '2024-01-01 09:00:07'), \
             (4, 'B', TIMESTAMP '2024-01-01 09:00:02'), \
             (5, 'B', TIMESTAMP '2024-01-01 09:00:08'), \
             (6, NULL, TIMESTAMP '2024-01-01 09:00:03'), \
             (7, 'A', NULL)",
            "CREATE TABLE asof_right(grp TEXT, ts TIMESTAMP, val TEXT) AS VALUES \
             ('A', TIMESTAMP '2024-01-01 09:00:02', 'a2'), \
             ('A', TIMESTAMP '2024-01-01 09:00:04', 'a4'), \
             ('A', TIMESTAMP '2024-01-01 09:00:06', 'a6'), \
             ('B', TIMESTAMP '2024-01-01 09:00:01', 'b1'), \
             ('B', TIMESTAMP '2024-01-01 09:00:06', 'b6'), \
             (NULL, TIMESTAMP '2024-01-01 09:00:02', 'null-group'), \
             ('A', NULL, 'null-ts')",
        ] {
            ctx.sql(sql).await?.collect().await?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_asof_joins_of_datafusions_tests_are_native() {
        let multiple_keys = "SELECT l.id, r.val FROM (VALUES \
             (1, 'X', 'A', TIMESTAMP '2024-01-01 09:00:04'), \
             (2, 'Y', 'A', TIMESTAMP '2024-01-01 09:00:04')) AS l(id, venue, grp, ts) \
             ASOF JOIN (VALUES \
             ('X', 'A', TIMESTAMP '2024-01-01 09:00:02', 'x-a2'), \
             ('Y', 'A', TIMESTAMP '2024-01-01 09:00:03', 'y-a3'), \
             ('X', 'B', TIMESTAMP '2024-01-01 09:00:04', 'x-b4')) AS r(venue, grp, ts, val) \
             MATCH_CONDITION (l.ts >= r.ts) ON l.venue = r.venue AND l.grp = r.grp \
             ORDER BY l.id";
        for sql in [
            "SELECT l.id, l.ts, r.ts, r.val FROM asof_left l ASOF JOIN asof_right r \
             MATCH_CONDITION (l.ts >= r.ts) ON l.grp = r.grp ORDER BY l.id",
            "SELECT l.id, l.ts, r.ts, r.val FROM asof_left l ASOF JOIN asof_right r \
             MATCH_CONDITION (l.ts > r.ts) ON l.grp = r.grp ORDER BY l.id",
            "SELECT l.id, l.ts, r.ts, r.val FROM asof_left l ASOF JOIN asof_right r \
             MATCH_CONDITION (l.ts <= r.ts) ON l.grp = r.grp ORDER BY l.id",
            "SELECT l.id, l.ts, r.ts, r.val FROM asof_left l ASOF JOIN asof_right r \
             MATCH_CONDITION (l.ts < r.ts) ON l.grp = r.grp ORDER BY l.id",
            "SELECT grp, l.id, r.ts, r.val FROM asof_left l ASOF JOIN asof_right r \
             MATCH_CONDITION (l.ts >= r.ts) USING (grp) ORDER BY l.id",
            "SELECT * FROM asof_left x ASOF JOIN asof_right a \
             MATCH_CONDITION (x.ts >= a.ts) USING (grp) WHERE x.id IN (1, 2) ORDER BY x.id",
            "SELECT l.id, r.label FROM (VALUES (1, 1), (2, 5), (3, CAST(NULL AS INT))) \
             AS l(id, ts) ASOF JOIN (VALUES (2, 'r2'), (4, 'r4')) AS r(ts, label) \
             MATCH_CONDITION (l.ts >= r.ts) ORDER BY l.id",
            multiple_keys,
            "SELECT l.id, r.val FROM asof_left l ASOF JOIN \
             (SELECT * FROM asof_right WHERE val <> 'a6') r MATCH_CONDITION (l.ts >= r.ts) \
             ON l.grp = r.grp WHERE l.id IN (3, 5) ORDER BY l.id",
            "SELECT l.id, r.val FROM asof_left l ASOF JOIN asof_right r \
             MATCH_CONDITION (l.ts >= r.ts + INTERVAL '1 second') \
             ON lower(l.grp) = lower(r.grp) WHERE l.id IN (2, 3, 4) ORDER BY l.id",
            "SELECT l.id, r.val FROM (SELECT * FROM asof_left UNION ALL \
             SELECT * FROM asof_left) l ASOF JOIN asof_right r \
             MATCH_CONDITION (l.ts >= r.ts) ON l.grp = r.grp ORDER BY l.id, r.val",
            "SELECT a.id, b.id FROM asof_left a ASOF JOIN asof_left b \
             MATCH_CONDITION (a.ts > b.ts) ON a.grp = b.grp ORDER BY a.id",
            "SELECT l.id, r.payload FROM (VALUES (1, CAST(5 AS SMALLINT), CAST(10 AS INT))) \
             AS l(id, grp, ts) ASOF JOIN (VALUES (CAST(5 AS BIGINT), CAST(9 AS BIGINT), 90)) \
             AS r(grp, ts, payload) MATCH_CONDITION (l.ts >= r.ts) ON l.grp = r.grp",
            // `-0.0` equals `0.0` in the match values (DataFusion does not
            // join on floating-point keys).
            "SELECT l.id, r.payload FROM (VALUES (1, 1, CAST(-0.0 AS DOUBLE)), (2, 1, 0.0), \
             (3, 2, -1.0)) AS l(id, k, ts) ASOF JOIN (VALUES (1, CAST(0.0 AS DOUBLE), 10), \
             (1, 0.5, 20), (2, CAST(-0.0 AS DOUBLE), 30)) AS r(k, ts, payload) \
             MATCH_CONDITION (l.ts >= r.ts) ON l.k = r.k ORDER BY l.id",
            "SELECT l.id, r.payload FROM (VALUES (1, CAST(-0.0 AS DOUBLE)), (2, 0.0)) \
             AS l(id, ts) ASOF JOIN (VALUES (CAST(0.0 AS DOUBLE), 10), (1.0, 20)) \
             AS r(ts, payload) MATCH_CONDITION (l.ts > r.ts) ORDER BY l.id",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, register_slt_tables).await;
                assert!(count::<PhysicalAsOfJoin>(&root) > 0, "{sql}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_of_the_right_input_reserve_memory() -> Result<()> {
        let ctx = pipeline_context_with_memory_limit(10_000);
        register_tables(&ctx).await?;
        let sql = "SELECT t.id, q.price FROM trades t ASOF JOIN quotes q \
                   MATCH_CONDITION (t.ts >= q.ts) ON t.sym = q.sym";
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalAsOfJoin>(&root), 1);
        assert_resources_exhausted(&ctx, sql).await;
        // In a pool that holds them, nothing stays reserved.
        let ctx = pipeline_context_with_memory_limit(100 << 20);
        register_tables(&ctx).await?;
        let rows = ctx.sql(sql).await?.collect().await?;
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 600);
        assert_eq!(ctx.runtime_env().memory_pool.reserved(), 0);
        Ok(())
    }
}
