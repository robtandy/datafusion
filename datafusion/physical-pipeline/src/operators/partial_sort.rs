// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalPartialSort`]: the sort of an input that is sorted by a prefix
//! of the sort keys (`PartialSortExec`).

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::{concat_batches, lexsort_to_indices, take_record_batch};
use arrow::datatypes::SchemaRef;
use arrow_ord::ord::make_comparator;
use datafusion_common::utils::evaluate_partition_ranges;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::sorts::partial_sort::PartialSortExec;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorFinalizeResult, OperatorResult,
    OperatorState, PhysicalOperator, downcast_state_mut,
};

/// Sorts an input whose rows are sorted by a prefix of its sort keys (the
/// first `common_prefix_length` keys), as `PartialSortExec` does: it buffers
/// the rows of a group of rows with the same prefix until a row of another
/// prefix arrives, which completes the group (and the groups before it), and
/// then emits the rows of the completed groups, sorted by all keys. So it
/// streams, also over inputs that never end, which DataFusion plans it for.
/// With a fetch, it emits the first `fetch` rows, and then finishes.
///
/// It is not a parallel operator, so that the pipelines that contain it run
/// on a single thread, which sees the chunks of its input in their order
/// (the order of its single partition). Over several partitions (a
/// `PartialSortExec` that preserves its partitioning), whose rows the pipeline
/// engine does not keep apart, it sorts the runs of rows with the same prefix
/// as they arrive: no native operator depends on the order of the rows of
/// several partitions, which they partition, merge or sort themselves.
///
/// The memory of the rows that it buffers, and of the sorted rows that it
/// did not emit yet, is reserved in the memory pool of the session.
#[derive(Debug)]
pub struct PhysicalPartialSort {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    expr: LexOrdering,
    /// The number of keys by which the input is sorted.
    common_prefix_length: usize,
    fetch: Option<usize>,
}

impl PhysicalPartialSort {
    /// Sort `input`, sorted by the first `common_prefix_length` keys of
    /// `expr`, by `expr`, keeping the first `fetch` rows (if any).
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        expr: LexOrdering,
        common_prefix_length: usize,
        fetch: Option<usize>,
    ) -> Result<Self> {
        if common_prefix_length == 0 || common_prefix_length > expr.len() {
            return internal_err!(
                "PhysicalPartialSort sorts by {} keys, its input by {common_prefix_length}",
                expr.len()
            );
        }
        Ok(Self {
            children: vec![input],
            expr,
            common_prefix_length,
            fetch,
        })
    }

    /// Lower a `PartialSortExec` to a [`PhysicalPartialSort`], whose input
    /// is `lower_input` of the input of the node. Returns `Ok(None)` for any
    /// other node, without calling `lower_input`.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(sort) = plan.downcast_ref::<PartialSortExec>() else {
            return Ok(None);
        };
        let input = lower_input(sort.input())?;
        Self::try_new(
            input,
            sort.expr().clone(),
            sort.common_prefix_length(),
            sort.fetch(),
        )
        .map(Some)
    }

    /// The ranges of the rows of `batch` with the same prefix, in order.
    fn prefix_ranges(&self, batch: &RecordBatch) -> Result<Vec<std::ops::Range<usize>>> {
        let keys = self.expr[..self.common_prefix_length]
            .iter()
            .map(|expr| expr.evaluate_to_sort_column(batch))
            .collect::<Result<Vec<_>>>()?;
        evaluate_partition_ranges(batch.num_rows(), &keys)
    }

    /// Whether the first row of `next` has the prefix of the last row of
    /// `previous`.
    fn same_prefix(&self, previous: &RecordBatch, next: &RecordBatch) -> Result<bool> {
        let last = previous.slice(previous.num_rows() - 1, 1);
        let first = next.slice(0, 1);
        for expr in &self.expr[..self.common_prefix_length] {
            let last = expr.evaluate_to_sort_column(&last)?;
            let first = expr.evaluate_to_sort_column(&first)?;
            let comparator = make_comparator(
                last.values.as_ref(),
                first.values.as_ref(),
                expr.options,
            )?;
            if !comparator(0, 0).is_eq() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `rows` (of complete groups), sorted by all keys, at most `limit` of
    /// them.
    fn sort(&self, rows: &[RecordBatch], limit: Option<usize>) -> Result<RecordBatch> {
        let schema = self.children[0].schema();
        let batch = concat_batches(&schema, rows)?;
        let keys = self
            .expr
            .iter()
            .map(|expr| expr.evaluate_to_sort_column(&batch))
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(&keys, limit)?;
        Ok(take_record_batch(&batch, &indices)?)
    }
}

/// The state of the thread of a [`PhysicalPartialSort`].
struct PartialSortState {
    /// The rows of the last group of rows with the same prefix, which may
    /// continue in the next chunk.
    group: Vec<RecordBatch>,
    /// The sorted rows that are not emitted yet, and the first of them.
    sorted: Option<RecordBatch>,
    offset: usize,
    /// Whether the chunk that the operator is called with was consumed.
    consumed: bool,
    /// The rows that may still be emitted.
    remaining: Option<usize>,
    reservation: MemoryReservation,
}

impl OperatorState for PartialSortState {}

impl PartialSortState {
    /// Reserve the memory of the buffered and the sorted rows.
    fn reserve(&self) -> Result<()> {
        let size = self
            .group
            .iter()
            .chain(&self.sorted)
            .map(RecordBatch::get_array_memory_size)
            .sum();
        self.reservation.try_resize(size)
    }

    /// Emit the next chunk of the sorted rows into `output`. Returns whether
    /// sorted rows are left.
    fn emit(&mut self, output: &mut DataChunk) -> bool {
        let Some(sorted) = &self.sorted else {
            return false;
        };
        let len = (sorted.num_rows() - self.offset).min(DEFAULT_VECTOR_SIZE);
        *output = DataChunk::from_record_batch(sorted.slice(self.offset, len));
        self.offset += len;
        if self.offset < sorted.num_rows() {
            return true;
        }
        self.sorted = None;
        false
    }

    /// Make `sorted` the sorted rows to emit, keeping at most the rows that
    /// may still be emitted.
    fn set_sorted(&mut self, sorted: RecordBatch) {
        let sorted = match &mut self.remaining {
            Some(remaining) => {
                let rows = sorted.num_rows().min(*remaining);
                *remaining -= rows;
                sorted.slice(0, rows)
            }
            None => sorted,
        };
        self.offset = 0;
        self.sorted = (sorted.num_rows() > 0).then_some(sorted);
    }

    fn done(&self) -> bool {
        self.remaining == Some(0) && self.sorted.is_none()
    }
}

impl PhysicalOperator for PhysicalPartialSort {
    fn name(&self) -> &str {
        "PhysicalPartialSort"
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let fetch = self
            .fetch
            .map(|fetch| format!(", fetch={fetch}"))
            .unwrap_or_default();
        format!(
            "PhysicalPartialSort: expr=[{}], common_prefix_length={}{fetch}",
            self.expr, self.common_prefix_length
        )
    }

    fn parallel_operator(&self) -> bool {
        false
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        let reservation = MemoryConsumer::new("PhysicalPartialSort")
            .register(ctx.task_ctx.memory_pool());
        Ok(Box::new(PartialSortState {
            group: vec![],
            sorted: None,
            offset: 0,
            consumed: false,
            remaining: self.fetch,
            reservation,
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
        let state =
            downcast_state_mut::<PartialSortState>(lstate.as_any_mut(), self.name())?;
        if !state.consumed {
            state.consumed = true;
            let batch = input.compact()?;
            let ranges = self.prefix_ranges(&batch)?;
            let Some(last) = ranges.last() else {
                return internal_err!(
                    "PhysicalPartialSort received a chunk without rows"
                );
            };
            let continues = match state.group.last() {
                Some(previous) => self.same_prefix(previous, &batch)?,
                None => true,
            };
            if ranges.len() == 1 && continues {
                // The chunk continues the last group.
                state.group.push(batch);
            } else {
                // The groups before the last group of the chunk are complete.
                let mut complete = std::mem::take(&mut state.group);
                if last.start > 0 {
                    complete.push(batch.slice(0, last.start));
                }
                state.group.push(batch.slice(last.start, last.len()));
                if !complete.is_empty() {
                    let sorted = self.sort(&complete, state.remaining)?;
                    state.set_sorted(sorted);
                }
            }
            state.reserve()?;
        }
        if state.emit(output) {
            return Ok(OperatorResult::HaveMoreOutput);
        }
        state.consumed = false;
        state.reserve()?;
        if state.done() {
            return Ok(OperatorResult::Finished);
        }
        Ok(OperatorResult::NeedMoreInput)
    }

    fn requires_final_execute(&self) -> bool {
        true
    }

    fn final_execute(
        &self,
        _ctx: &ExecutionContext,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorFinalizeResult> {
        let state =
            downcast_state_mut::<PartialSortState>(lstate.as_any_mut(), self.name())?;
        // The last group is complete once the input ends.
        if !state.group.is_empty() && state.remaining != Some(0) {
            let group = std::mem::take(&mut state.group);
            let sorted = self.sort(&group, state.remaining)?;
            state.set_sorted(sorted);
            state.reserve()?;
        }
        state.group.clear();
        if state.emit(output) {
            return Ok(OperatorFinalizeResult::HaveMoreOutput);
        }
        state.reservation.free();
        Ok(OperatorFinalizeResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::Write;
    use std::path::Path;

    use arrow::array::{AsArray, Int32Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::col;

    use crate::executor::Executor;
    use crate::operators::lowering_tests::{
        assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit,
    };
    use crate::sources::MemoryScanSource;
    use crate::test_utils::{register_test_tables, test_session_config};

    /// Write `rows` rows sorted by `a` and `b` (in groups of about 37 rows
    /// with the same `a` and `b`), whose `c` and `d` are not sorted, to the
    /// CSV file `path`.
    fn write_sorted_csv(path: &Path, rows: i32) -> Result<()> {
        let mut csv = String::from("a,b,c,d\n");
        for i in 0..rows {
            let a = i / 1000;
            let b = (i % 1000) / 37;
            let c = (i * 7919) % 101;
            let d = if i % 13 == 0 {
                String::new()
            } else {
                ((i * 31) % 17).to_string()
            };
            writeln!(csv, "{a},{b},{c},{d}").unwrap();
        }
        std::fs::write(path, csv)?;
        Ok(())
    }

    /// Register `s`, an unbounded table of a sorted CSV file (a stream that
    /// ends), ordered by `a` and `b`.
    async fn register_sorted_stream(ctx: &SessionContext, path: &Path) -> Result<()> {
        register_test_tables(ctx)?;
        let sql = format!(
            "CREATE UNBOUNDED EXTERNAL TABLE s (a INT, b INT, c INT, d INT) \
             STORED AS CSV WITH ORDER (a ASC, b ASC) LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            path.display()
        );
        ctx.sql(&sql).await?.collect().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partial_sorts_are_native() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sorted.csv");
        write_sorted_csv(&path, 10_000)?;
        let setup = async |ctx: &SessionContext| register_sorted_stream(ctx, &path).await;
        for sql in [
            "SELECT * FROM s ORDER BY a, b, c, d",
            "SELECT a, b, c FROM s ORDER BY a, b, c DESC",
            "SELECT * FROM s ORDER BY a, b, d NULLS FIRST, c",
            "SELECT * FROM s ORDER BY a, b, c, d LIMIT 50",
            "SELECT * FROM s ORDER BY a, b, c, d LIMIT 5000",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, &setup).await;
                assert_eq!(count::<PhysicalPartialSort>(&root), 1, "{sql}");
            }
        }
        Ok(())
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    /// A partial sort by `a` (the prefix) and `b` of `batches` (sorted by
    /// `a`), keeping `fetch` rows.
    fn partial_sort(
        batches: Vec<RecordBatch>,
        fetch: Option<usize>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let source = MemoryScanSource::try_new(&[batches], schema(), None, None)?
            .with_preserve_order(true);
        let schema = schema();
        let expr = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("a", &schema)?),
            PhysicalSortExpr::new_default(col("b", &schema)?),
        ])
        .unwrap();
        Ok(Arc::new(PhysicalPartialSort::try_new(
            Arc::new(source),
            expr,
            1,
            fetch,
        )?))
    }

    fn batch(a: Vec<i32>, b: Vec<i32>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from(a)), Arc::new(Int32Array::from(b))],
        )
        .unwrap()
    }

    fn rows(batches: &[RecordBatch]) -> Vec<(i32, i32)> {
        batches
            .iter()
            .flat_map(|batch| {
                let a = batch.column(0).as_primitive::<Int32Type>();
                let b = batch.column(1).as_primitive::<Int32Type>();
                (0..batch.num_rows())
                    .map(|i| (a.value(i), b.value(i)))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn groups_that_span_chunks_are_sorted_once_complete() -> Result<()> {
        // A group of `a = 1` over three chunks, one of `a = 2` of a single
        // row, and a group of 5000 rows that the operator emits in chunks.
        let batches = vec![
            batch(vec![0, 0, 1, 1], vec![9, 3, 7, 5]),
            batch(vec![1], vec![1]),
            batch(vec![1, 1, 2, 3], vec![8, 2, 4, 6]),
            batch(vec![3; 5000], (0..5000).rev().collect()),
        ];
        // The rows, sorted by `a` and `b`.
        let mut input = rows(&batches);
        input.sort_unstable();
        let output = Executor::collect(
            partial_sort(batches.clone(), None)?,
            Arc::new(TaskContext::default()),
            4,
        )
        .await?;
        assert!(
            output
                .iter()
                .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
        );
        assert_eq!(rows(&output), input);
        // With a fetch, the first rows.
        for fetch in [0, 1, 3, 7, 9, 4000] {
            let output = Executor::collect(
                partial_sort(batches.clone(), Some(fetch))?,
                Arc::new(TaskContext::default()),
                4,
            )
            .await?;
            assert_eq!(rows(&output), input[..fetch.min(input.len())], "{fetch}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn buffered_rows_reserve_memory() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sorted.csv");
        write_sorted_csv(&path, 10_000)?;
        let ctx = pipeline_context_with_memory_limit(2000);
        register_sorted_stream(&ctx, &path).await?;
        let sql = "SELECT * FROM s ORDER BY a, b, c, d";
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalPartialSort>(&root), 1);
        assert_resources_exhausted(&ctx, sql).await;
        Ok(())
    }
}
