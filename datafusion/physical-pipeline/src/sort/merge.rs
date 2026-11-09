// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The k-way merge of sorted runs with a tree of losers.
//!
//! The sinks of [`PhysicalSort`](super::PhysicalSort) and
//! [`PhysicalTopK`](super::PhysicalTopK) sort the rows that every thread
//! collects into [`SortedRun`]s. Once all threads are done, a [`RunMerger`]
//! merges the runs of all threads into the global order and emits the result
//! in chunks of at most
//! [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows, stopping
//! after `fetch` rows.
//!
//! The merger is a tournament tree of losers (Knuth, TAOCP vol. 3, 5.4.1):
//! every inner node holds the run that lost the match played at that node,
//! and the root holds the overall winner, the run whose current row comes
//! first. Emitting a row advances the winner and replays only the matches on
//! the path from its leaf to the root, so every row costs `log2(k)`
//! comparisons for `k` runs. Rows are compared in the Arrow row format, in
//! which comparing two rows is a `memcmp` of their bytes. Rows that compare
//! equal are emitted in the order of their runs, and within a run in the
//! order of the run.
//!
//! Materializing rows can need more bytes than a single Arrow array
//! addresses (e.g. 2 GiB of `Utf8` values, whose offsets are 32-bit), also
//! when the rows come from arrays that fit: the rows are then split into
//! several runs, and the merger emits smaller chunks.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::compute::take_arrays;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow_row::{Row, Rows};
use arrow_select::interleave::interleave;
use datafusion_common::{Result, internal_err};

/// Whether `error` says that rows do not fit into one array: its offsets
/// (32-bit for `Utf8`, `Binary`, `List`, ..., also in nested and run-end
/// encoded arrays) would overflow. The rows then fit into smaller arrays.
pub(crate) fn is_capacity_error(error: &ArrowError) -> bool {
    match error {
        ArrowError::OffsetOverflowError(_) => true,
        // `MutableArrayData`, which some kernels fall back to.
        ArrowError::InvalidArgumentError(message) => message.contains("offset overflow"),
        _ => false,
    }
}

/// The rows `positions` (`(batch, row)`) of `batches` as one batch with
/// schema `schema`: taken from the only batch, or interleaved.
pub(crate) fn gather_rows(
    schema: &SchemaRef,
    batches: &[&RecordBatch],
    positions: &[(usize, usize)],
) -> std::result::Result<RecordBatch, ArrowError> {
    let columns = match batches {
        [batch] => {
            let rows = positions
                .iter()
                .map(|&(_, row)| u32::try_from(row))
                .collect::<std::result::Result<UInt32Array, _>>()
                .map_err(|e| ArrowError::ComputeError(e.to_string()))?;
            take_arrays(batch.columns(), &rows, None)?
        }
        _ => (0..schema.fields().len())
            .map(|column| {
                let values = batches
                    .iter()
                    .map(|batch| batch.column(column).as_ref())
                    .collect::<Vec<&dyn Array>>();
                interleave(&values, positions)
            })
            .collect::<std::result::Result<Vec<ArrayRef>, _>>()?,
    };
    let options = RecordBatchOptions::new().with_row_count(Some(positions.len()));
    RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
}

/// The rows `positions` of `batches` (see [`gather_rows`]), in order, in as
/// few batches as their columns fit into: all of them in one batch, unless
/// that needs more bytes than an array addresses, in which case the rows
/// are split (see [`is_capacity_error`]).
pub(crate) fn gather_rows_in_batches(
    schema: &SchemaRef,
    batches: &[&RecordBatch],
    positions: &[(usize, usize)],
) -> Result<Vec<RecordBatch>> {
    let mut gathered = vec![];
    let mut start = 0;
    while start < positions.len() {
        let mut len = positions.len() - start;
        loop {
            match gather_rows(schema, batches, &positions[start..start + len]) {
                Ok(batch) => {
                    gathered.push(batch);
                    start += len;
                    break;
                }
                Err(error) if len > 1 && is_capacity_error(&error) => {
                    len = len.div_ceil(2);
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(gathered)
}

/// Rows in sort order: a batch, and the sort keys of its rows in the Arrow
/// row format (the same [`RowConverter`](arrow_row::RowConverter) for all
/// runs that are merged together).
#[derive(Debug)]
pub struct SortedRun {
    batch: RecordBatch,
    keys: Rows,
}

impl SortedRun {
    /// A run of the rows of `batch`, which must be sorted by `keys`, the
    /// encoded sort keys of its rows.
    pub fn try_new(batch: RecordBatch, keys: Rows) -> Result<Self> {
        if keys.num_rows() != batch.num_rows() {
            return internal_err!(
                "A sorted run of {} rows has sort keys for {} rows",
                batch.num_rows(),
                keys.num_rows()
            );
        }
        Ok(Self { batch, keys })
    }

    /// The rows of the run, in sort order.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// The encoded sort keys of the rows of the run.
    pub fn keys(&self) -> &Rows {
        &self.keys
    }

    /// The number of rows of the run.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// The encoded sort key of row `row`.
    pub fn key(&self, row: usize) -> Row<'_> {
        self.keys.row(row)
    }

    /// The batch of the run.
    pub fn into_batch(self) -> RecordBatch {
        self.batch
    }
}

/// Merges [`SortedRun`]s with a tree of losers, see the
/// [module documentation](self).
#[derive(Debug)]
pub struct RunMerger {
    schema: SchemaRef,
    /// The runs that still have rows to emit; exhausted runs are dropped
    /// (`None`) to release their memory.
    runs: Vec<Option<SortedRun>>,
    /// The next row of every run.
    positions: Vec<usize>,
    /// Rows `(run, row)` that the tree chose, but that were not emitted yet,
    /// in order: a chunk whose rows do not fit into one batch is emitted in
    /// parts.
    pending: Vec<(usize, usize)>,
    /// `tree[0]` is the winner, the run whose current row comes first;
    /// `tree[1..]` are the losers of the matches at the inner nodes. Leaf `i`
    /// (run `i`) is the child of node `(k + i) / 2`, and node `j` is the child of
    /// node `j / 2`.
    tree: Vec<usize>,
    /// The number of rows still to emit.
    remaining: usize,
}

impl RunMerger {
    /// Merge `runs`, whose rows all have schema `schema`, and emit at most
    /// `fetch` rows (all rows if `None`).
    ///
    /// Rows with equal sort keys are emitted in the order of `runs`.
    pub fn new(schema: SchemaRef, runs: Vec<SortedRun>, fetch: Option<usize>) -> Self {
        let runs = runs
            .into_iter()
            .filter(|run| run.num_rows() > 0)
            .map(Some)
            .collect::<Vec<_>>();
        let total = runs
            .iter()
            .flatten()
            .map(SortedRun::num_rows)
            .sum::<usize>();
        let mut merger = Self {
            schema,
            positions: vec![0; runs.len()],
            pending: vec![],
            tree: vec![],
            runs,
            remaining: fetch.map_or(total, |fetch| fetch.min(total)),
        };
        merger.init_tree();
        merger
    }

    /// The schema of the merged rows.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The number of rows the merger still emits.
    pub fn remaining_rows(&self) -> usize {
        self.remaining + self.pending.len()
    }

    /// Emit the next (at most) `max_rows` rows in sort order, or `None` once
    /// all rows (or `fetch` rows) were emitted.
    ///
    /// The batch has fewer rows if the rows do not fit into one batch (the
    /// offsets of a column would overflow); the other rows come with the next
    /// batches.
    pub fn next_batch(&mut self, max_rows: usize) -> Result<Option<RecordBatch>> {
        if self.pending.is_empty() {
            let len = self.remaining.min(max_rows);
            if len == 0 {
                return Ok(None);
            }
            for _ in 0..len {
                let winner = self.tree[0];
                let position = self.positions[winner];
                self.pending.push((winner, position));
                self.positions[winner] = position + 1;
                self.replay(winner);
            }
            self.remaining -= len;
        }
        let mut len = self.pending.len().min(max_rows);
        let batch = loop {
            match self.gather(&self.pending[..len])? {
                Ok(batch) => break batch,
                Err(error) if len > 1 && is_capacity_error(&error) => {
                    len = len.div_ceil(2);
                }
                Err(error) => return Err(error.into()),
            }
        };
        // Drop the runs that have no rows left to emit, neither in the tree
        // nor pending.
        let (emitted, pending) = self.pending.split_at(len);
        for &(run, _) in emitted {
            if self.runs[run]
                .as_ref()
                .is_some_and(|sorted| self.positions[run] >= sorted.num_rows())
                && !pending.iter().any(|&(other, _)| other == run)
            {
                self.runs[run] = None;
            }
        }
        self.pending.drain(..len);
        Ok(Some(batch))
    }

    /// Build the tree: every run plays from its leaf towards the root until
    /// it reaches a node without a loser, where it waits for the winner of
    /// the other subtree. The last run reaches the root.
    fn init_tree(&mut self) {
        let k = self.runs.len();
        if k == 0 {
            return;
        }
        self.tree = vec![usize::MAX; k];
        for run in 0..k {
            let mut winner = run;
            let mut node = usize::midpoint(k, run);
            while node != 0 && self.tree[node] != usize::MAX {
                let challenger = self.tree[node];
                if self.comes_after(winner, challenger) {
                    self.tree[node] = winner;
                    winner = challenger;
                }
                node /= 2;
            }
            self.tree[node] = winner;
        }
    }

    /// Replay the matches on the path from the leaf of `run`, which was the
    /// winner and advanced to its next row, to the root.
    fn replay(&mut self, run: usize) {
        let k = self.runs.len();
        let mut winner = run;
        let mut node = usize::midpoint(k, run);
        while node != 0 {
            let challenger = self.tree[node];
            if self.comes_after(winner, challenger) {
                self.tree[node] = winner;
                winner = challenger;
            }
            node /= 2;
        }
        self.tree[0] = winner;
    }

    /// Whether the current row of run `a` comes after the current row of run
    /// `b`. An exhausted run comes after every run that is not; rows with
    /// equal keys come in the order of their runs.
    fn comes_after(&self, a: usize, b: usize) -> bool {
        match (self.current_key(a), self.current_key(b)) {
            (Some(key_a), Some(key_b)) => match key_a.cmp(&key_b) {
                Ordering::Equal => a > b,
                ordering => ordering == Ordering::Greater,
            },
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => a > b,
        }
    }

    /// The key of the current row of `run`, `None` if the run is exhausted.
    fn current_key(&self, run: usize) -> Option<Row<'_>> {
        let position = self.positions[run];
        self.runs[run]
            .as_ref()
            .filter(|run| position < run.num_rows())
            .map(|run| run.key(position))
    }

    /// The rows `(run, row)` of `indices`, as a batch. The inner error says
    /// why the rows cannot be materialized, e.g. that they do not fit into one
    /// batch.
    fn gather(
        &self,
        indices: &[(usize, usize)],
    ) -> Result<std::result::Result<RecordBatch, ArrowError>> {
        let (first_run, first_row) = indices[0];
        let consecutive = indices
            .iter()
            .enumerate()
            .all(|(i, &(run, row))| run == first_run && row == first_row + i);
        if consecutive {
            // All rows come from one run, in order (e.g. a single run): no
            // copy.
            let batch = self.run(first_run)?.batch.slice(first_row, indices.len());
            let (_, columns, _) = batch.into_parts();
            let options = RecordBatchOptions::new().with_row_count(Some(indices.len()));
            return Ok(RecordBatch::try_new_with_options(
                SchemaRef::clone(&self.schema),
                columns,
                &options,
            ));
        }
        // Interleave the rows of the runs that occur in `indices`.
        let mut slots = vec![usize::MAX; self.runs.len()];
        let mut sources = vec![];
        let indices = indices
            .iter()
            .map(|&(run, row)| {
                if slots[run] == usize::MAX {
                    slots[run] = sources.len();
                    sources.push(run);
                }
                (slots[run], row)
            })
            .collect::<Vec<_>>();
        let batches = sources
            .iter()
            .map(|&run| Ok(&self.run(run)?.batch))
            .collect::<Result<Vec<_>>>()?;
        Ok(gather_rows(&self.schema, &batches, &indices))
    }

    fn run(&self, run: usize) -> Result<&SortedRun> {
        match &self.runs[run] {
            Some(run) => Ok(run),
            None => internal_err!("RunMerger: run {run} was already released"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::{AsArray, Int32Array, Int64Array, StringArray};
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Int32Type, Int64Type, Schema};
    use arrow_row::{RowConverter, SortField};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("id", DataType::Int64, false),
        ]))
    }

    fn converter(options: SortOptions) -> RowConverter {
        RowConverter::new(vec![SortField::new_with_options(DataType::Int32, options)])
            .unwrap()
    }

    /// A run of the rows `(k, id)`, sorted by `k` with `options`, with `s`
    /// the string of `id`.
    fn run(
        converter: &RowConverter,
        options: SortOptions,
        mut rows: Vec<(Option<i32>, i64)>,
    ) -> SortedRun {
        let compare = |a: &Option<i32>, b: &Option<i32>| match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) if options.nulls_first => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) if options.nulls_first => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) if options.descending => b.cmp(a),
            (Some(a), Some(b)) => a.cmp(b),
        };
        // A stable sort: the order of equal keys is the order of the ids.
        rows.sort_by(|a, b| compare(&a.0, &b.0));
        let k = Arc::new(Int32Array::from_iter(rows.iter().map(|row| row.0))) as ArrayRef;
        let s = Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.1.to_string()),
        ));
        let id = Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1)));
        let keys = converter.convert_columns(&[Arc::clone(&k)]).unwrap();
        let batch = RecordBatch::try_new(schema(), vec![k, s, id]).unwrap();
        SortedRun::try_new(batch, keys).unwrap()
    }

    /// Merge `runs` with `merger`, in batches of at most `max_rows` rows.
    fn merge_all(mut merger: RunMerger, max_rows: usize) -> Vec<RecordBatch> {
        let mut batches = vec![];
        while let Some(batch) = merger.next_batch(max_rows).unwrap() {
            assert!(batch.num_rows() <= max_rows && batch.num_rows() > 0);
            assert_eq!(batch.schema(), schema());
            batches.push(batch);
        }
        assert_eq!(merger.remaining_rows(), 0);
        assert!(merger.next_batch(max_rows).unwrap().is_none());
        batches
    }

    fn rows_of(batches: &[RecordBatch]) -> Vec<(Option<i32>, String, i64)> {
        batches
            .iter()
            .flat_map(|batch| {
                let k = batch.column(0).as_primitive::<Int32Type>();
                let s = batch.column(1).as_string::<i32>();
                let id = batch.column(2).as_primitive::<Int64Type>();
                (0..batch.num_rows())
                    .map(|i| {
                        (
                            k.is_valid(i).then(|| k.value(i)),
                            s.value(i).to_string(),
                            id.value(i),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn capacity_errors_are_recognized() {
        assert!(is_capacity_error(&ArrowError::OffsetOverflowError(1 << 31)));
        // The error of `MutableArrayData`.
        let message = "offset overflow: data exceeds the capacity of the offset type";
        assert!(is_capacity_error(&ArrowError::InvalidArgumentError(
            message.to_string()
        )));
        assert!(!is_capacity_error(&ArrowError::DivideByZero));
        assert!(!is_capacity_error(&ArrowError::InvalidArgumentError(
            "invalid".to_string()
        )));
    }

    #[test]
    fn merges_random_runs_in_sort_order() {
        let mut rng = StdRng::seed_from_u64(42);
        let all_options = [
            SortOptions::new(false, false),
            SortOptions::new(false, true),
            SortOptions::new(true, false),
            SortOptions::new(true, true),
        ];
        for k in [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33] {
            for options in all_options {
                let converter = converter(options);
                let mut next_id = 0;
                let mut inputs = vec![];
                let runs = (0..k)
                    .map(|_| {
                        // Some runs are empty, keys repeat, some are null.
                        let len = match rng.random_range(0..5) {
                            0 => 0,
                            1 => 1,
                            _ => rng.random_range(0..300),
                        };
                        let rows = (0..len)
                            .map(|_| {
                                next_id += 1;
                                let key = (rng.random_range(0..10) != 0)
                                    .then(|| rng.random_range(-50..50));
                                (key, next_id)
                            })
                            .collect::<Vec<_>>();
                        let run = run(&converter, options, rows);
                        inputs.push(rows_of(std::slice::from_ref(run.batch())));
                        run
                    })
                    .collect::<Vec<_>>();
                // The expected result: all rows, stably sorted by key, so that
                // the rows of the runs come in the order of the runs.
                let all = inputs.concat();
                let keys = converter
                    .convert_columns(&[Arc::new(Int32Array::from_iter(
                        all.iter().map(|row| row.0),
                    )) as ArrayRef])
                    .unwrap();
                let mut order = (0..all.len()).collect::<Vec<_>>();
                order.sort_by(|&a, &b| keys.row(a).cmp(&keys.row(b)));
                let expected = order.iter().map(|&i| all[i].clone()).collect::<Vec<_>>();
                for fetch in [None, Some(0), Some(1), Some(7), Some(expected.len())] {
                    for max_rows in [1, 3, 64, 2048] {
                        let runs = runs
                            .iter()
                            .map(|run| {
                                SortedRun::try_new(
                                    run.batch().clone(),
                                    converter
                                        .convert_columns(&[Arc::clone(
                                            run.batch().column(0),
                                        )])
                                        .unwrap(),
                                )
                                .unwrap()
                            })
                            .collect();
                        let merger = RunMerger::new(schema(), runs, fetch);
                        let len = fetch.map_or(expected.len(), |f| f.min(expected.len()));
                        assert_eq!(merger.remaining_rows(), len);
                        let actual = rows_of(&merge_all(merger, max_rows));
                        assert_eq!(
                            actual,
                            expected[..len],
                            "{k} runs, {options:?}, fetch {fetch:?}, {max_rows} rows per batch"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn merges_no_runs_and_empty_runs() {
        let options = SortOptions::default();
        let converter = converter(options);
        let merger = RunMerger::new(schema(), vec![], None);
        assert!(merge_all(merger, 10).is_empty());
        let empty = || run(&converter, options, vec![]);
        let merger = RunMerger::new(schema(), vec![empty(), empty()], Some(5));
        assert_eq!(merger.remaining_rows(), 0);
        assert!(merge_all(merger, 10).is_empty());
    }

    #[test]
    fn a_single_run_is_sliced_without_copying() {
        let options = SortOptions::default();
        let converter = converter(options);
        let rows = (0..10).map(|i| (Some(i), i64::from(i))).collect();
        let single = run(&converter, options, rows);
        let values = single.batch().column(2).to_data().buffers()[0].as_ptr();
        let merger = RunMerger::new(schema(), vec![single], Some(7));
        let batches = merge_all(merger, 3);
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![3, 3, 1]
        );
        assert_eq!(
            batches[0].column(2).to_data().buffers()[0].as_ptr(),
            values,
            "the batches are slices of the run"
        );
        let ids = rows_of(&batches)
            .iter()
            .map(|row| row.2)
            .collect::<Vec<_>>();
        assert_eq!(ids, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn exhausted_runs_are_released() {
        let options = SortOptions::default();
        let converter = converter(options);
        let first = run(&converter, options, vec![(Some(1), 1), (Some(2), 2)]);
        let second = run(&converter, options, vec![(Some(3), 3), (Some(4), 4)]);
        let mut merger = RunMerger::new(schema(), vec![first, second], None);
        let batch = merger.next_batch(3).unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert!(merger.runs[0].is_none(), "the first run is exhausted");
        assert!(merger.runs[1].is_some());
        let batch = merger.next_batch(3).unwrap().unwrap();
        assert_eq!(rows_of(&[batch]), vec![(Some(4), "4".to_string(), 4)]);
        assert!(merger.runs.iter().all(Option::is_none));
        assert!(merger.next_batch(3).unwrap().is_none());
    }

    #[test]
    fn runs_check_their_keys() {
        let options = SortOptions::default();
        let converter = converter(options);
        let sorted = run(&converter, options, vec![(Some(1), 1), (Some(2), 2)]);
        let keys = converter
            .convert_columns(&[Arc::new(Int32Array::from(vec![1])) as ArrayRef])
            .unwrap();
        let err = SortedRun::try_new(sorted.into_batch(), keys).unwrap_err();
        assert!(err.to_string().contains("sort keys for 1 rows"), "{err}");
    }
}
