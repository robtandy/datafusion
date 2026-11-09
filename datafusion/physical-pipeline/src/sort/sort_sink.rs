// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalSort`]: a parallel sort, the pipeline breaker of `ORDER BY`.
//!
//! Like DuckDB's `PhysicalOrder`, the sort is the sink of the pipelines of
//! its input and the source of the pipeline that reads its output:
//!
//! * [`sink`](PhysicalOperator::sink): every thread compacts the chunks it
//!   pushes and buffers them in its local sink state. Once it buffered
//!   [`SORT_RUN_ROWS`] rows (or about [`SORT_RUN_BYTES`] bytes), it sorts them
//!   into a [`SortedRun`] (with `lexsort_to_indices` and `take` for a single
//!   sort key of a primitive, boolean, string or binary type, by their keys in
//!   the Arrow row format otherwise), so that the runs fit in Arrow arrays and
//!   the threads sort while they read their input;
//! * [`combine`](PhysicalOperator::combine): the thread sorts the rest of its
//!   rows and hands its runs to the global sink state;
//! * [`finalize`](PhysicalOperator::finalize): a [`RunMerger`] is set up over
//!   all runs of all threads;
//! * [`get_data`](PhysicalOperator::get_data): the merger emits the sorted
//!   rows in chunks of at most
//!   [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows, on a
//!   single thread (the source is not parallel), stopping after `fetch` rows.
//!
//! With a fetch, every run keeps only its first `fetch` rows: no other row
//! can be among the first `fetch` rows of the merge. (A sort with a fetch is
//! usually lowered to a [`PhysicalTopK`](super::PhysicalTopK), which keeps
//! even fewer rows.)

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, internal_err};
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::ExecutionPlan;
use parking_lot::Mutex;

use super::merge::{RunMerger, SortedRun};
use super::{SortKeys, SortPlan, SortedOutput, memory_size, next_sorted_chunk};
use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
    downcast_state, downcast_state_mut,
};

/// A thread sorts its buffered rows into a run once it buffered this many
/// rows.
pub const SORT_RUN_ROWS: usize = 1 << 18;

/// A thread sorts its buffered rows into a run once they occupy about this
/// many bytes, so that the columns of a run stay far below the 2 GiB that the
/// 32-bit offsets of string, binary and list arrays address.
pub const SORT_RUN_BYTES: usize = 256 * 1024 * 1024;

/// Sorts all rows of its input, see the [module documentation](self).
#[derive(Debug)]
pub struct PhysicalSort {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    keys: SortKeys,
    fetch: Option<usize>,
    run_rows: usize,
    run_bytes: usize,
}

impl PhysicalSort {
    /// Sort the rows of `input` by `ordering` and emit the first `fetch` of
    /// them (all of them if `None`).
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        ordering: LexOrdering,
        fetch: Option<usize>,
    ) -> Result<Self> {
        let schema = input.schema();
        let keys = SortKeys::try_new(ordering, &schema)?;
        Ok(Self {
            children: vec![input],
            schema,
            keys,
            fetch,
            run_rows: SORT_RUN_ROWS,
            run_bytes: SORT_RUN_BYTES,
        })
    }

    /// Lower a sort of a DataFusion plan that is not a TopK: see
    /// [`try_lower`](super::try_lower), which lowers a sort that keeps only
    /// its first rows to a [`PhysicalTopK`](super::PhysicalTopK), unless it is
    /// a merge with a fetch of a sort whose keys can fail to evaluate (a
    /// `PhysicalSort` with a fetch). Returns `Ok(None)` for TopKs and any
    /// other node, without calling `lower_input`.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        options: &ConfigOptions,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        match SortPlan::try_new(plan, options)? {
            Some(sort) if !sort.is_topk() => sort.into_sort(plan, lower_input).map(Some),
            _ => Ok(None),
        }
    }

    /// Sort runs of at most `rows` rows and about `bytes` bytes.
    #[cfg(test)]
    pub(crate) fn with_run_limits(mut self, rows: usize, bytes: usize) -> Self {
        self.run_rows = rows;
        self.run_bytes = bytes;
        self
    }

    /// The input of the sort.
    pub fn input(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The sort keys.
    pub fn ordering(&self) -> &LexOrdering {
        self.keys.ordering()
    }

    /// The number of rows the sort emits, `None` for all of them.
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Sort the rows buffered in `state` into a run.
    fn sort_buffered_rows(&self, state: &mut SortLocalState) -> Result<()> {
        let batches = std::mem::take(&mut state.batches);
        state.rows = 0;
        state.bytes = 0;
        state
            .runs
            .extend(self.keys.sort(&self.schema, &batches, self.fetch)?);
        Ok(())
    }
}

/// The global sink state of a [`PhysicalSort`].
#[derive(Debug, Default)]
struct SortSinkState {
    /// The sorted runs of the threads that completed.
    runs: Mutex<Vec<SortedRun>>,
    output: SortedOutput,
}

impl GlobalSinkState for SortSinkState {}

/// The local sink state of a [`PhysicalSort`]: the rows of the thread.
#[derive(Debug, Default)]
struct SortLocalState {
    /// Compacted chunks that are not sorted yet.
    batches: Vec<RecordBatch>,
    /// The number of rows of `batches`.
    rows: usize,
    /// The estimated memory size of `batches`.
    bytes: usize,
    /// The sorted runs of the thread.
    runs: Vec<SortedRun>,
}

impl LocalSinkState for SortLocalState {}

impl PhysicalOperator for PhysicalSort {
    fn name(&self) -> &str {
        "PhysicalSort"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        match self.fetch {
            Some(fetch) => {
                format!("PhysicalSort: expr=[{}], fetch={fetch}", self.ordering())
            }
            None => format!("PhysicalSort: expr=[{}]", self.ordering()),
        }
    }

    // ---------------------------------------------------------------------
    // Sink
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
        Ok(Arc::new(SortSinkState::default()))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(SortLocalState::default()))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        _gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        if self.fetch == Some(0) {
            return Ok(SinkResult::Finished);
        }
        let state =
            downcast_state_mut::<SortLocalState>(lstate.as_any_mut(), self.name())?;
        let batch = input.compact()?;
        state.rows += batch.num_rows();
        state.bytes += memory_size(&batch);
        state.batches.push(batch);
        if state.rows >= self.run_rows || state.bytes >= self.run_bytes {
            self.sort_buffered_rows(state)?;
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let global = downcast_state::<SortSinkState>(gstate.as_any(), self.name())?;
        let state =
            downcast_state_mut::<SortLocalState>(lstate.as_any_mut(), self.name())?;
        self.sort_buffered_rows(state)?;
        global.runs.lock().append(&mut state.runs);
        Ok(())
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let global = downcast_state::<SortSinkState>(gstate.as_any(), self.name())?;
        let runs = std::mem::take(&mut *global.runs.lock());
        let merger = RunMerger::new(Arc::clone(&self.schema), runs, self.fetch);
        let empty = merger.remaining_rows() == 0;
        global.output.set(merger);
        Ok(if empty {
            SinkFinalize::NoOutputPossible
        } else {
            SinkFinalize::Ready
        })
    }

    // ---------------------------------------------------------------------
    // Source
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        // A single thread emits the rows, in order.
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
        let global = downcast_state::<SortSinkState>(sink.as_any(), self.name())?;
        global.output.take_source(self.name())
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        next_sorted_chunk(gstate, self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::Array;

    use arrow::array::{AsArray, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::{col, lit};

    use crate::chunk::SelectionVector;
    use crate::executor::Executor;
    use crate::explain::format_pipelines;
    use crate::operators::union::test_support::{
        TestSource, run, values, values_chunk, values_chunks, values_schema,
    };

    fn sort(
        input: Arc<dyn PhysicalOperator>,
        descending: bool,
        fetch: Option<usize>,
    ) -> PhysicalSort {
        let v = col("v", &input.schema()).unwrap();
        let mut sort_expr = PhysicalSortExpr::new_default(v);
        if descending {
            sort_expr = sort_expr.desc();
        }
        PhysicalSort::try_new(input, LexOrdering::new([sort_expr]).unwrap(), fetch)
            .unwrap()
    }

    /// Chunks of the values `0..n` in a scrambled order, of `size` rows each.
    fn scrambled_chunks(n: i64, size: usize) -> Vec<DataChunk> {
        let scrambled = (0..n).map(|i| (i * 7919) % n).collect::<Vec<_>>();
        scrambled
            .chunks(size)
            .map(|values| values_chunk(values.iter().copied()))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_all_rows_on_any_number_of_threads() -> Result<()> {
        let n = 10_007;
        for threads in [1, 2, 4] {
            for (run_rows, descending) in [
                (SORT_RUN_ROWS, false),
                (SORT_RUN_ROWS, true),
                (1000, false),
                (1, true),
            ] {
                let input = Arc::new(TestSource::new(scrambled_chunks(n, 1000)));
                let root = sort(input, descending, None)
                    .with_run_limits(run_rows, SORT_RUN_BYTES);
                let output = run(Arc::new(root), threads).await?;
                let mut expected = (0..n).collect::<Vec<_>>();
                if descending {
                    expected.reverse();
                }
                assert_eq!(
                    values(&output),
                    expected,
                    "{threads} threads, runs of {run_rows} rows"
                );
                assert!(
                    output
                        .iter()
                        .all(|batch| batch.num_rows() <= crate::DEFAULT_VECTOR_SIZE)
                );
                assert!(output.iter().all(|batch| batch.schema() == values_schema()));
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_the_logical_rows_of_selected_chunks() -> Result<()> {
        // The even values of 0..100 and the values 100..200 in reverse order.
        let evens = values_chunk(0..100).with_selection(SelectionVector::from(
            (0..100).step_by(2).collect::<Vec<u32>>(),
        ))?;
        let reversed = values_chunk(100..200).with_selection(SelectionVector::from(
            (0..100).rev().collect::<Vec<u32>>(),
        ))?;
        let input = Arc::new(TestSource::new(vec![reversed, evens]));
        let output = run(Arc::new(sort(input, true, Some(110))), 4).await?;
        let expected = (100..200_i64)
            .rev()
            .chain((40..50).rev().map(|i| i * 2))
            .collect::<Vec<_>>();
        assert_eq!(values(&output), expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_keeps_the_first_rows() -> Result<()> {
        let n = 5000;
        for fetch in [0, 1, 7, 2048, 2049, 4999, 5000, 6000, usize::MAX] {
            for run_rows in [SORT_RUN_ROWS, 300] {
                let input = Arc::new(TestSource::new(scrambled_chunks(n, 700)));
                let root = sort(input, false, Some(fetch))
                    .with_run_limits(run_rows, SORT_RUN_BYTES);
                let output = run(Arc::new(root), 4).await?;
                let expected = (0..n).take(fetch).collect::<Vec<_>>();
                assert_eq!(
                    values(&output),
                    expected,
                    "fetch {fetch}, runs of {run_rows} rows"
                );
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_empty_single_row_and_skewed_inputs() -> Result<()> {
        let empty = Arc::new(TestSource::with_schema(values_schema(), vec![]));
        assert!(run(Arc::new(sort(empty, false, None)), 4).await?.is_empty());

        let single = Arc::new(TestSource::new(vec![values_chunk([42])]));
        let output = run(Arc::new(sort(single, false, None)), 4).await?;
        assert_eq!(values(&output), vec![42]);

        // One large chunk, many tiny ones and duplicates.
        let mut chunks = vec![values_chunk((0..2048).rev())];
        chunks.extend((0..300).map(|i| values_chunk([i % 7, 5000 + i])));
        let input = Arc::new(TestSource::new(chunks));
        let output = run(Arc::new(sort(input, false, None)), 4).await?;
        let mut expected = (0..2048).chain((0..300).map(|i| i % 7)).collect::<Vec<_>>();
        expected.extend(5000..5300);
        expected.sort_unstable();
        assert_eq!(values(&output), expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_by_several_keys_with_all_columns() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]));
        let chunk = |values: std::ops::Range<i64>| {
            let k = Int64Array::from_iter(
                values.clone().map(|v| (v % 5 != 0).then_some(v % 3)),
            );
            let s = StringArray::from_iter(
                values
                    .clone()
                    .map(|v| (v % 7 != 0).then(|| format!("s{}", v % 4))),
            );
            let v = Int64Array::from_iter_values(values);
            DataChunk::from_record_batch(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(k), Arc::new(s), Arc::new(v)],
                )
                .unwrap(),
            )
        };
        let chunks = (0..20).map(|i| chunk(i * 100..i * 100 + 100)).collect();
        let input: Arc<dyn PhysicalOperator> =
            Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks));
        // k DESC NULLS FIRST, s ASC NULLS LAST, v ASC.
        let ordering = LexOrdering::new([
            PhysicalSortExpr::new_default(col("k", &schema)?)
                .desc()
                .nulls_first(),
            PhysicalSortExpr::new_default(col("s", &schema)?)
                .asc()
                .nulls_last(),
            PhysicalSortExpr::new_default(col("v", &schema)?),
        ])
        .unwrap();
        let root = PhysicalSort::try_new(input, ordering, None)?
            .with_run_limits(250, SORT_RUN_BYTES);
        let output =
            Executor::collect(Arc::new(root), Arc::new(TaskContext::default()), 4)
                .await?;
        let mut expected = (0..2000_i64)
            .map(|v| {
                (
                    (v % 5 != 0).then_some(v % 3),
                    (v % 7 != 0).then(|| format!("s{}", v % 4)),
                    v,
                )
            })
            .collect::<Vec<_>>();
        expected.sort_by(|a, b| {
            // NULLS FIRST, DESC.
            let k = match (a.0, b.0) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(x), Some(y)) => y.cmp(&x),
            };
            // NULLS LAST, ASC.
            let s = match (&a.1, &b.1) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (Some(_), None) => std::cmp::Ordering::Less,
                (Some(x), Some(y)) => x.cmp(y),
            };
            k.then(s).then(a.2.cmp(&b.2))
        });
        let actual = output
            .iter()
            .flat_map(|batch| {
                let k = batch.column(0).as_primitive::<Int64Type>();
                let s = batch.column(1).as_string::<i32>();
                let v = batch.column(2).as_primitive::<Int64Type>();
                (0..batch.num_rows())
                    .map(|i| {
                        (
                            k.is_valid(i).then(|| k.value(i)),
                            s.is_valid(i).then(|| s.value(i).to_string()),
                            v.value(i),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_chunks_without_columns() -> Result<()> {
        let schema = Arc::new(Schema::empty());
        let chunks = (0..3)
            .map(|_| DataChunk::try_new_without_columns(Arc::clone(&schema), 2000))
            .collect::<Result<Vec<_>>>()?;
        let ordering =
            LexOrdering::new([PhysicalSortExpr::new_default(lit(1_i32))]).unwrap();
        for (fetch, run_rows) in [(None, SORT_RUN_ROWS), (None, 1500), (Some(10), 700)] {
            let input =
                Arc::new(TestSource::with_schema(Arc::clone(&schema), chunks.clone()));
            let root = PhysicalSort::try_new(input, ordering.clone(), fetch)?
                .with_run_limits(run_rows, SORT_RUN_BYTES);
            let output = run(Arc::new(root), 4).await?;
            let rows = output.iter().map(RecordBatch::num_rows).sum::<usize>();
            assert_eq!(rows, fetch.unwrap_or(6000));
            assert!(output.iter().all(|batch| batch.num_columns() == 0));
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_can_be_executed_again() -> Result<()> {
        let input = Arc::new(TestSource::new(values_chunks(0, &[10, 2048, 7])));
        let root: Arc<dyn PhysicalOperator> = Arc::new(sort(input, true, Some(100)));
        for _ in 0..3 {
            let output = run(Arc::clone(&root), 4).await?;
            assert_eq!(values(&output), (1965..2065).rev().collect::<Vec<_>>());
        }
        Ok(())
    }

    #[test]
    fn sort_properties_and_pipelines() -> Result<()> {
        let input: Arc<dyn PhysicalOperator> =
            Arc::new(TestSource::new(vec![values_chunk(0..3)]));
        let root = sort(Arc::clone(&input), true, Some(5));
        assert!(root.is_sink() && root.parallel_sink());
        assert!(root.is_source() && !root.parallel_source());
        assert_eq!(root.fetch(), Some(5));
        assert!(Arc::ptr_eq(root.input(), &input));
        assert_eq!(root.schema(), input.schema());
        assert_eq!(root.description(), "PhysicalSort: expr=[v@0 DESC], fetch=5");
        assert_eq!(
            sort(Arc::clone(&input), false, None).description(),
            "PhysicalSort: expr=[v@0 ASC]"
        );
        let root: Arc<dyn PhysicalOperator> = Arc::new(root);
        assert_eq!(
            format_pipelines(&root)?,
            "MetaPipeline 0 (sink: ResultSink) children: [1]\n  \
             Pipeline 0: PhysicalSort -> ResultSink depends on meta pipelines [1]\n\
             MetaPipeline 1 (sink: PhysicalSort)\n  \
             Pipeline 1: TestSource -> PhysicalSort\n"
        );
        Ok(())
    }
}
