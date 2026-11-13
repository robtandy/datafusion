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
//! comparisons for `k` runs. Rows that compare equal are emitted in the order
//! of their runs, and within a run in the order of the run.
//!
//! The keys of the runs ([`RunKeys`]) are either encoded in the Arrow row
//! format, in which comparing two rows is a `memcmp` of their bytes, or, for
//! a single sort key of a type that [`compares_directly`] (integers, floats,
//! decimals, temporal values, booleans, strings and binary values), the
//! sorted key values themselves ([`KeyColumn`]), which the tree compares
//! directly in the typed arrays, without encoding them: in the order of the
//! row format and of the sort kernel (floats in their total order, `-0.0`
//! before `0.0`). The merger dispatches on the type of the keys once per
//! chunk, so that its matches are monomorphized for every key type.
//!
//! Materializing rows can need more bytes than a single Arrow array
//! addresses (e.g. 2 GiB of `Utf8` values, whose offsets are 32-bit), also
//! when the rows come from arrays that fit: the rows are then split into
//! several runs, and the merger emits smaller chunks.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, GenericByteArray, GenericByteViewArray, PrimitiveArray,
    RecordBatch, RecordBatchOptions, UInt32Array,
};
use arrow::buffer::NullBuffer;
use arrow::compute::{SortOptions, take_arrays};
use arrow::datatypes::{
    ArrowNativeTypeOp, ArrowPrimitiveType, BinaryType, BinaryViewType, ByteArrayType,
    ByteViewType, DataType, LargeBinaryType, LargeUtf8Type, SchemaRef, StringViewType,
    Utf8Type,
};
use arrow::downcast_primitive_array;
use arrow::error::ArrowError;
use arrow_ord::ord::make_comparator;
use arrow_row::{Row, Rows};
use arrow_select::interleave::interleave;
use datafusion_common::{Result, internal_err};

/// Whether `error` says that rows do not fit into one array: its offsets
/// (32-bit for `Utf8`, `Binary`, `List`, ..., also in nested and run-end
/// encoded arrays) would overflow, the merged values of its dictionaries need
/// more keys than their key type holds (e.g. more than 128 values for `Int8`
/// keys), or its run ends more rows than their type holds (e.g. 32767 for
/// `Int16` run ends). The rows then fit into smaller arrays.
pub(crate) fn is_capacity_error(error: &ArrowError) -> bool {
    match error {
        ArrowError::OffsetOverflowError(_)
        | ArrowError::DictionaryKeyOverflowError
        | ArrowError::RunEndIndexOverflowError => true,
        // `MutableArrayData`, which some kernels fall back to.
        ArrowError::InvalidArgumentError(message) => message.contains("offset overflow"),
        // The interleaving of run-end encoded arrays.
        ArrowError::ComputeError(message) => {
            message.contains("does not fit run-end type")
        }
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
        [batch] => take_arrays(batch.columns(), &row_indices(positions)?, None)?,
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

/// The values `positions` (`(array, row)`) of `arrays`, as one array: taken
/// from the only array, or interleaved.
pub(crate) fn gather_values(
    arrays: &[&dyn Array],
    positions: &[(usize, usize)],
) -> std::result::Result<ArrayRef, ArrowError> {
    match arrays {
        [array] => arrow::compute::take(*array, &row_indices(positions)?, None),
        _ => interleave(arrays, positions),
    }
}

/// The rows of `positions` (`(batch, row)`), as `take` indices.
fn row_indices(
    positions: &[(usize, usize)],
) -> std::result::Result<UInt32Array, ArrowError> {
    positions
        .iter()
        .map(|&(_, row)| u32::try_from(row))
        .collect::<std::result::Result<UInt32Array, _>>()
        .map_err(|e| ArrowError::ComputeError(e.to_string()))
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

/// Whether the values of `data_type` can be compared directly, as the values
/// of a [`KeyColumn`], in the order in which the Arrow row format orders them:
/// the types without children that the sort kernel (`sort_to_indices`) sorts
/// exactly like the row format, i.e. primitive types (integers, floats in
/// their total order, decimals, dates, times, timestamps, durations and
/// intervals, compared with [`ArrowNativeTypeOp::compare`] as by the sort
/// kernel), booleans, strings and binary values (compared byte by byte).
pub fn compares_directly(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
}

/// The values of a single sort key, in sort order, compared directly (see
/// [`compares_directly`]) with the options of the sort key: descending keys
/// compare in reverse, and nulls come first or last as the options say.
#[derive(Debug, Clone)]
pub struct KeyColumn {
    values: ArrayRef,
    options: SortOptions,
}

impl KeyColumn {
    /// The values `values` of a sort key with the options `options`. Their
    /// type must be one that [`compares_directly`].
    pub fn try_new(values: ArrayRef, options: SortOptions) -> Result<Self> {
        if !compares_directly(values.data_type()) {
            return internal_err!(
                "Sort keys of type {} cannot be compared directly",
                values.data_type()
            );
        }
        Ok(Self { values, options })
    }

    /// The values of the sort key.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// The options of the sort key.
    pub fn options(&self) -> SortOptions {
        self.options
    }
}

/// The sort keys of the rows of a [`SortedRun`].
#[derive(Debug, Clone)]
pub enum RunKeys {
    /// The keys encoded in the Arrow row format, with the same
    /// [`RowConverter`](arrow_row::RowConverter) for all runs that are merged
    /// together: comparing two keys is a `memcmp`.
    Rows(Rows),
    /// The values of a single sort key, which are compared directly.
    Column(KeyColumn),
}

impl RunKeys {
    /// The number of keys.
    pub fn len(&self) -> usize {
        match self {
            Self::Rows(rows) => rows.num_rows(),
            Self::Column(column) => column.values.len(),
        }
    }

    /// Whether there are no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Rows in sort order: a batch, and the sort keys of its rows ([`RunKeys`]:
/// the same kind of keys for all runs that are merged together).
#[derive(Debug)]
pub struct SortedRun {
    batch: RecordBatch,
    keys: RunKeys,
}

impl SortedRun {
    /// A run of the rows of `batch`, which must be sorted by `keys`, the
    /// encoded sort keys of its rows.
    pub fn try_new(batch: RecordBatch, keys: Rows) -> Result<Self> {
        Self::try_new_with_keys(batch, RunKeys::Rows(keys))
    }

    /// A run of the rows of `batch`, which must be sorted by `keys`, the sort
    /// keys of its rows.
    pub fn try_new_with_keys(batch: RecordBatch, keys: RunKeys) -> Result<Self> {
        if keys.len() != batch.num_rows() {
            return internal_err!(
                "A sorted run of {} rows has sort keys for {} rows",
                batch.num_rows(),
                keys.len()
            );
        }
        Ok(Self { batch, keys })
    }

    /// The rows of the run, in sort order.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// The sort keys of the rows of the run.
    pub fn keys(&self) -> &RunKeys {
        &self.keys
    }

    /// The number of rows of the run.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// The memory that the run holds: its rows (see
    /// [`buffered_memory`](super::buffered_memory)) and their keys, unless
    /// the keys are a column of the rows.
    pub fn memory_size(&self) -> usize {
        let keys = match &self.keys {
            RunKeys::Rows(rows) => rows.size(),
            RunKeys::Column(column)
                if self
                    .batch
                    .columns()
                    .iter()
                    .any(|values| Arc::ptr_eq(values, &column.values)) =>
            {
                0
            }
            RunKeys::Column(column) => column.values.get_array_memory_size(),
        };
        super::buffered_memory(&self.batch) + keys
    }

    /// The sort key of row `row` encoded in the row format, if the keys of
    /// the run are encoded (see [`RunKeys::Rows`]).
    pub fn encoded_key(&self, row: usize) -> Option<Row<'_>> {
        match &self.keys {
            RunKeys::Rows(rows) => Some(rows.row(row)),
            RunKeys::Column(_) => None,
        }
    }

    /// The order of row `row` of the run and row `other_row` of `other`,
    /// whose keys must be of the same kind (and type).
    pub fn compare_rows(
        &self,
        row: usize,
        other: &SortedRun,
        other_row: usize,
    ) -> Result<Ordering> {
        match (&self.keys, &other.keys) {
            (RunKeys::Rows(rows), RunKeys::Rows(other_rows)) => {
                Ok(rows.row(row).cmp(&other_rows.row(other_row)))
            }
            (RunKeys::Column(column), RunKeys::Column(other_column)) => {
                let compare = make_comparator(
                    column.values.as_ref(),
                    other_column.values.as_ref(),
                    column.options,
                )?;
                Ok(compare(row, other_row))
            }
            _ => internal_err!("Sorted runs with different kinds of keys"),
        }
    }

    /// The batch of the run.
    pub fn into_batch(self) -> RecordBatch {
        self.batch
    }
}

/// How the keys of the runs of a [`RunMerger`] compare.
#[derive(Debug, Clone, PartialEq)]
enum KeyKind {
    /// Encoded rows ([`RunKeys::Rows`]).
    Rows,
    /// Columns of values of `data_type`, sorted with `options`
    /// ([`RunKeys::Column`]).
    Column {
        data_type: DataType,
        options: SortOptions,
    },
}

impl KeyKind {
    fn of(keys: &RunKeys) -> Self {
        match keys {
            RunKeys::Rows(_) => Self::Rows,
            RunKeys::Column(column) => Self::Column {
                data_type: column.values.data_type().clone(),
                options: column.options,
            },
        }
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
    /// The number of rows of every run.
    lens: Vec<usize>,
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
    /// The memory of the runs that are not dropped yet (see
    /// [`SortedRun::memory_size`]).
    memory: usize,
    /// How the keys of the runs compare, or why the runs cannot be merged:
    /// their keys are of different kinds or types.
    keys: std::result::Result<KeyKind, String>,
}

impl RunMerger {
    /// Merge `runs`, whose rows all have schema `schema`, and emit at most
    /// `fetch` rows (all rows if `None`).
    ///
    /// Rows with equal sort keys are emitted in the order of `runs`. The keys
    /// of all runs must be of the same kind (and type), otherwise
    /// [`Self::next_batch`] fails.
    pub fn new(schema: SchemaRef, runs: Vec<SortedRun>, fetch: Option<usize>) -> Self {
        let runs = runs
            .into_iter()
            .filter(|run| run.num_rows() > 0)
            .collect::<Vec<_>>();
        let lens = runs.iter().map(SortedRun::num_rows).collect::<Vec<_>>();
        let total = lens.iter().sum::<usize>();
        let memory = runs.iter().map(SortedRun::memory_size).sum();
        let kinds = runs.iter().map(|run| KeyKind::of(&run.keys));
        let mut keys = Ok(KeyKind::Rows);
        for (index, kind) in kinds.enumerate() {
            match &keys {
                Ok(_) if index == 0 => keys = Ok(kind),
                Ok(first) if *first != kind => {
                    keys = Err(format!(
                        "RunMerger: runs with keys of different kinds: {first:?} and {kind:?}"
                    ));
                }
                _ => {}
            }
        }
        let mut merger = Self {
            schema,
            positions: vec![0; runs.len()],
            pending: vec![],
            tree: vec![],
            runs: runs.into_iter().map(Some).collect(),
            lens,
            remaining: fetch.map_or(total, |fetch| fetch.min(total)),
            memory,
            keys,
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

    /// The memory of the runs that the merger still holds (see
    /// [`SortedRun::memory_size`]): it drops the runs whose rows it emitted,
    /// and all of them once it emitted its rows.
    pub fn memory_size(&self) -> usize {
        self.memory
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
            let Self {
                runs,
                lens,
                positions,
                pending,
                tree,
                keys,
                ..
            } = &mut *self;
            let keys = match keys {
                Ok(keys) => &*keys,
                Err(message) => return internal_err!("{message}"),
            };
            let choose = Choose {
                tree,
                positions,
                lens,
                pending,
                len,
            };
            play(runs, keys, choose)?;
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
            if self.positions[run] >= self.lens[run]
                && !pending.iter().any(|&(other, _)| other == run)
                && let Some(dropped) = self.runs[run].take()
            {
                self.memory -= dropped.memory_size();
            }
        }
        self.pending.drain(..len);
        // Once all rows are emitted (e.g. the `fetch` of a sort), no row of
        // any run is needed any more.
        if self.remaining_rows() == 0 {
            self.runs.iter_mut().for_each(|run| *run = None);
            self.memory = 0;
        }
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
        let Self {
            runs,
            lens,
            positions,
            tree,
            keys,
            ..
        } = &mut *self;
        // Runs whose keys cannot be compared are not merged: `next_batch`
        // fails. (Building the tree cannot fail otherwise.)
        if let Ok(kind) = keys {
            let init = InitTree {
                tree,
                positions,
                lens,
            };
            if let Err(error) = play(runs, kind, init) {
                *keys = Err(format!("RunMerger: {error}"));
            }
        }
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

/// Work on the tree of a [`RunMerger`] that compares the current rows of
/// runs, with a comparison `compare(run_a, row_a, run_b, row_b)` of row
/// `row_a` of run `run_a` and row `row_b` of run `run_b` that is
/// monomorphized for the type of the keys.
trait Tournament {
    fn play<C>(self, compare: C)
    where
        C: Fn(usize, usize, usize, usize) -> Ordering;
}

/// Whether the current row of run `a` comes after the current row of run `b`
/// (see [`Tournament`]). An exhausted run comes after every run that is not;
/// rows with equal keys come in the order of their runs.
#[inline]
fn comes_after<C>(
    compare: &C,
    positions: &[usize],
    lens: &[usize],
    a: usize,
    b: usize,
) -> bool
where
    C: Fn(usize, usize, usize, usize) -> Ordering,
{
    let (row_a, row_b) = (positions[a], positions[b]);
    match (row_a < lens[a], row_b < lens[b]) {
        (true, true) => match compare(a, row_a, b, row_b) {
            Ordering::Equal => a > b,
            ordering => ordering == Ordering::Greater,
        },
        (false, true) => true,
        (true, false) => false,
        (false, false) => a > b,
    }
}

/// Builds the tree, see [`RunMerger::init_tree`].
struct InitTree<'a> {
    tree: &'a mut [usize],
    positions: &'a [usize],
    lens: &'a [usize],
}

impl Tournament for InitTree<'_> {
    fn play<C>(self, compare: C)
    where
        C: Fn(usize, usize, usize, usize) -> Ordering,
    {
        let k = self.tree.len();
        for run in 0..k {
            let mut winner = run;
            let mut node = usize::midpoint(k, run);
            while node != 0 && self.tree[node] != usize::MAX {
                let challenger = self.tree[node];
                if comes_after(&compare, self.positions, self.lens, winner, challenger) {
                    self.tree[node] = winner;
                    winner = challenger;
                }
                node /= 2;
            }
            self.tree[node] = winner;
        }
    }
}

/// Chooses the next `len` rows: emits the winner and replays the matches on
/// the path from its leaf to the root, `len` times.
struct Choose<'a> {
    tree: &'a mut [usize],
    positions: &'a mut [usize],
    lens: &'a [usize],
    pending: &'a mut Vec<(usize, usize)>,
    len: usize,
}

impl Tournament for Choose<'_> {
    fn play<C>(self, compare: C)
    where
        C: Fn(usize, usize, usize, usize) -> Ordering,
    {
        let k = self.tree.len();
        for _ in 0..self.len {
            let run = self.tree[0];
            let position = self.positions[run];
            self.pending.push((run, position));
            self.positions[run] = position + 1;
            let mut winner = run;
            let mut node = usize::midpoint(k, run);
            while node != 0 {
                let challenger = self.tree[node];
                if comes_after(&compare, self.positions, self.lens, winner, challenger) {
                    self.tree[node] = winner;
                    winner = challenger;
                }
                node /= 2;
            }
            self.tree[0] = winner;
        }
    }
}

/// Play `tournament` with the comparison of the keys of `runs`, which are of
/// kind `keys` (exhausted runs, `None`, are never compared).
fn play<T: Tournament>(
    runs: &[Option<SortedRun>],
    keys: &KeyKind,
    tournament: T,
) -> Result<()> {
    let options = match keys {
        KeyKind::Rows => {
            let rows = runs
                .iter()
                .map(|run| match run.as_ref().map(SortedRun::keys) {
                    Some(RunKeys::Rows(rows)) => Some(rows),
                    _ => None,
                })
                .collect::<Vec<_>>();
            tournament.play(|a, row_a, b, row_b| match (rows[a], rows[b]) {
                (Some(rows_a), Some(rows_b)) => rows_a.row(row_a).cmp(&rows_b.row(row_b)),
                _ => Ordering::Equal,
            });
            return Ok(());
        }
        KeyKind::Column { options, .. } => *options,
    };
    let columns = runs
        .iter()
        .map(|run| match run.as_ref().map(SortedRun::keys) {
            Some(RunKeys::Column(column)) => Some(column.values.as_ref()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let Some(first) = columns.iter().flatten().next().copied() else {
        // No run left to compare.
        tournament.play(|_, _, _, _| Ordering::Equal);
        return Ok(());
    };
    downcast_primitive_array! {
        first => play_primitive(&columns, first, options, tournament),
        DataType::Boolean => {
            let arrays = typed(&columns, |array| array.as_boolean_opt());
            play_columns(&arrays, options, tournament, |a, i, b, j| {
                a.value(i).cmp(&b.value(j))
            });
        }
        DataType::Utf8 => play_bytes::<Utf8Type, T>(&columns, options, tournament),
        DataType::LargeUtf8 => play_bytes::<LargeUtf8Type, T>(&columns, options, tournament),
        DataType::Binary => play_bytes::<BinaryType, T>(&columns, options, tournament),
        DataType::LargeBinary => {
            play_bytes::<LargeBinaryType, T>(&columns, options, tournament)
        }
        DataType::Utf8View => play_views::<StringViewType, T>(&columns, options, tournament),
        DataType::BinaryView => play_views::<BinaryViewType, T>(&columns, options, tournament),
        DataType::FixedSizeBinary(_) => {
            let arrays = typed(&columns, |array| array.as_fixed_size_binary_opt());
            play_columns(&arrays, options, tournament, |a, i, b, j| {
                a.value(i).cmp(b.value(j))
            });
        }
        data_type => {
            return internal_err!("Sort keys of type {data_type} cannot be compared directly");
        }
    }
    Ok(())
}

/// The arrays `columns` (`None` for exhausted runs), downcast with `downcast`.
fn typed<'a, A: Array>(
    columns: &[Option<&'a dyn Array>],
    downcast: impl Fn(&'a dyn Array) -> Option<&'a A>,
) -> Vec<Option<&'a A>> {
    columns
        .iter()
        .map(|column| column.and_then(&downcast))
        .collect()
}

/// [`play`] with keys of the primitive type of `_first`, compared with
/// [`ArrowNativeTypeOp::compare`] (the comparison of the sort kernel, which
/// orders floats by their total order).
fn play_primitive<P: ArrowPrimitiveType, T: Tournament>(
    columns: &[Option<&dyn Array>],
    _first: &PrimitiveArray<P>,
    options: SortOptions,
    tournament: T,
) {
    let arrays = typed(columns, |array| array.as_primitive_opt::<P>());
    let values = arrays
        .iter()
        .map(|array| array.map_or(&[][..], |array| array.values().as_ref()))
        .collect::<Vec<&[P::Native]>>();
    let nulls = arrays
        .iter()
        .map(|array| array.and_then(|array| array.nulls()))
        .collect::<Vec<_>>();
    tournament.play(|a, i, b, j| {
        compare_nullable(nulls[a], i, nulls[b], j, options, || {
            values[a][i].compare(values[b][j])
        })
    });
}

/// [`play`] with keys of strings or binary values, compared byte by byte.
fn play_bytes<B: ByteArrayType, T: Tournament>(
    columns: &[Option<&dyn Array>],
    options: SortOptions,
    tournament: T,
) {
    let arrays = typed(columns, |array| array.as_bytes_opt::<B>());
    play_columns(
        &arrays,
        options,
        tournament,
        |a: &GenericByteArray<B>, i, b: &GenericByteArray<B>, j| {
            let (value_a, value_b): (&[u8], &[u8]) =
                (a.value(i).as_ref(), b.value(j).as_ref());
            value_a.cmp(value_b)
        },
    );
}

/// [`play`] with keys of string or binary views, compared byte by byte: by
/// their inlined bytes, if that decides (as the sort kernel compares views).
fn play_views<V: ByteViewType, T: Tournament>(
    columns: &[Option<&dyn Array>],
    options: SortOptions,
    tournament: T,
) {
    let arrays = typed(columns, |array| array.as_byte_view_opt::<V>());
    play_columns(&arrays, options, tournament, compare_views::<V>);
}

/// The order of view `i` of `a` and view `j` of `b`, byte by byte: values of
/// at most 12 bytes are inlined in their views, longer values compare by
/// their first 4 bytes (inlined in their views too) unless these are equal.
#[inline]
fn compare_views<V: ByteViewType>(
    a: &GenericByteViewArray<V>,
    i: usize,
    b: &GenericByteViewArray<V>,
    j: usize,
) -> Ordering {
    let (view_a, view_b) = (a.views()[i], b.views()[j]);
    let (len_a, len_b) = (view_a as u32, view_b as u32);
    if len_a <= 12 && len_b <= 12 {
        let key = GenericByteViewArray::<V>::inline_key_fast;
        return key(view_a).cmp(&key(view_b));
    }
    // The first 4 bytes, in big-endian order (zero padded for short values).
    let prefix = |view: u128| ((view >> 32) as u32).swap_bytes();
    match prefix(view_a).cmp(&prefix(view_b)) {
        Ordering::Equal => {
            let (value_a, value_b): (&[u8], &[u8]) =
                (a.value(i).as_ref(), b.value(j).as_ref());
            value_a.cmp(value_b)
        }
        ordering => ordering,
    }
}

/// [`play`] with the keys `arrays` (`None` for exhausted runs), whose
/// non-null values `compare_values(a, i, b, j)` compares.
fn play_columns<A: Array, T: Tournament>(
    arrays: &[Option<&A>],
    options: SortOptions,
    tournament: T,
    compare_values: impl Fn(&A, usize, &A, usize) -> Ordering,
) {
    let nulls = arrays
        .iter()
        .map(|array| array.and_then(|array| array.nulls()))
        .collect::<Vec<_>>();
    tournament.play(|a, i, b, j| match (arrays[a], arrays[b]) {
        (Some(array_a), Some(array_b)) => {
            compare_nullable(nulls[a], i, nulls[b], j, options, || {
                compare_values(array_a, i, array_b, j)
            })
        }
        _ => Ordering::Equal,
    });
}

/// The order of value `i` of a key whose nulls are `nulls_a` and value `j`
/// of a key whose nulls are `nulls_b`, sorted with `options`: nulls come
/// first or last, and `compare_values` compares the values if neither is
/// null (in ascending order).
#[inline]
fn compare_nullable(
    nulls_a: Option<&NullBuffer>,
    i: usize,
    nulls_b: Option<&NullBuffer>,
    j: usize,
    options: SortOptions,
    compare_values: impl FnOnce() -> Ordering,
) -> Ordering {
    let null_a = nulls_a.is_some_and(|nulls| nulls.is_null(i));
    let null_b = nulls_b.is_some_and(|nulls| nulls.is_null(j));
    match (null_a, null_b) {
        (false, false) => {
            let ordering = compare_values();
            if options.descending {
                ordering.reverse()
            } else {
                ordering
            }
        }
        (true, true) => Ordering::Equal,
        (true, false) if options.nulls_first => Ordering::Less,
        (true, false) => Ordering::Greater,
        (false, true) if options.nulls_first => Ordering::Greater,
        (false, true) => Ordering::Less,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::{
        AsArray, BooleanArray, Decimal128Array, FixedSizeBinaryArray, Float64Array,
        Int32Array, Int64Array, IntervalDayTimeArray, IntervalMonthDayNanoArray,
        IntervalYearMonthArray, ListArray, StringArray,
    };
    use arrow::compute::{SortOptions, cast};
    use arrow::datatypes::{
        DataType, Field, Int32Type, Int64Type, IntervalDayTime, IntervalMonthDayNano,
        IntervalUnit, Schema, TimeUnit,
    };
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
    fn chunks_whose_rows_do_not_fit_into_one_batch_are_split() {
        // 1100 runs of one row, whose strings share a 2 MiB buffer: a chunk of
        // all 1100 rows needs 2.2 GiB of string values, more than the 32-bit
        // offsets of `Utf8` address. The merger emits two chunks instead.
        let converter = converter(SortOptions::default());
        let s = StringArray::from(vec!["x".repeat(2 << 20)]);
        let runs = (0..1100)
            .map(|i| {
                let k = Arc::new(Int32Array::from(vec![i])) as ArrayRef;
                let id = Arc::new(Int64Array::from(vec![i64::from(i)]));
                let keys = converter.convert_columns(&[Arc::clone(&k)]).unwrap();
                let batch =
                    RecordBatch::try_new(schema(), vec![k, Arc::new(s.clone()), id])
                        .unwrap();
                SortedRun::try_new(batch, keys).unwrap()
            })
            .collect();
        let mut merger = RunMerger::new(schema(), runs, None);
        let mut sizes = vec![];
        let mut ids = vec![];
        while let Some(batch) = merger.next_batch(2048).unwrap() {
            sizes.push(batch.num_rows());
            let s = batch.column(1).as_string::<i32>();
            assert!((0..batch.num_rows()).all(|row| s.value(row).len() == 2 << 20));
            ids.extend_from_slice(batch.column(2).as_primitive::<Int64Type>().values());
        }
        assert_eq!(sizes, [550, 550]);
        assert_eq!(ids, (0..1100).collect::<Vec<_>>());
        assert_eq!(merger.remaining_rows(), 0);
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
        // Narrow dictionary keys and run ends.
        assert!(is_capacity_error(&ArrowError::DictionaryKeyOverflowError));
        assert!(is_capacity_error(&ArrowError::RunEndIndexOverflowError));
        let message = "interleave_run_end: output length 40000 does not fit run-end type";
        assert!(is_capacity_error(&ArrowError::ComputeError(
            message.to_string()
        )));
        assert!(!is_capacity_error(&ArrowError::ComputeError(
            "other".to_string()
        )));
    }

    #[test]
    fn merges_rows_whose_narrow_encodings_do_not_fit_into_one_array() {
        use arrow::array::{DictionaryArray, Int16Array, RunArray};
        use arrow::datatypes::{Int8Type, Int16Type};
        // Three runs of 64 rows with interleaving keys and 64 distinct
        // `Int8`-keyed dictionary values each: a chunk of their merge with all
        // 192 values needs more than the 128 values that `Int8` keys address,
        // so the merger emits smaller chunks.
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", dictionary, false),
        ]));
        let runs = (0..3_i64)
            .map(|run| {
                let k = (0..64).map(|i| i * 3 + run).collect::<Vec<_>>();
                let s = k.iter().map(|k| format!("value {k}")).collect::<Vec<_>>();
                let s = s
                    .iter()
                    .map(String::as_str)
                    .collect::<DictionaryArray<Int8Type>>();
                let keys = Int64Array::from(k);
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(keys.clone()), Arc::new(s)],
                )
                .unwrap();
                let options = SortOptions::default();
                let keys = KeyColumn::try_new(Arc::new(keys), options).unwrap();
                SortedRun::try_new_with_keys(batch, RunKeys::Column(keys)).unwrap()
            })
            .collect::<Vec<_>>();
        let mut merger = RunMerger::new(Arc::clone(&schema), runs, None);
        let mut keys = vec![];
        let mut chunks = 0;
        while let Some(batch) = merger.next_batch(2048).unwrap() {
            chunks += 1;
            keys.extend_from_slice(batch.column(0).as_primitive::<Int64Type>().values());
            let s = batch.column(1).as_dictionary::<Int8Type>();
            let s = s.downcast_dict::<StringArray>().unwrap();
            for (row, value) in s.into_iter().enumerate() {
                let k = batch.column(0).as_primitive::<Int64Type>().value(row);
                assert_eq!(value, Some(format!("value {k}").as_str()));
            }
        }
        assert!(chunks > 1, "{chunks} chunks");
        assert_eq!(keys, (0..192).collect::<Vec<_>>());

        // Two runs of 20000 rows of `Int16` run ends: all 40000 rows do not
        // fit into one array.
        let run_end_encoded = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int16, false)),
            Arc::new(Field::new("values", DataType::Int64, true)),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("r", run_end_encoded, false),
        ]));
        let runs = (0..2_i64)
            .map(|run| {
                let keys = Int64Array::from_iter_values((0..20_000).map(|i| i * 2 + run));
                let values = RunArray::<Int16Type>::try_new(
                    &Int16Array::from(vec![20_000]),
                    &Int64Array::from(vec![run]),
                )
                .unwrap();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(keys.clone()), Arc::new(values)],
                )
                .unwrap();
                let keys =
                    KeyColumn::try_new(Arc::new(keys), SortOptions::default()).unwrap();
                SortedRun::try_new_with_keys(batch, RunKeys::Column(keys)).unwrap()
            })
            .collect::<Vec<_>>();
        let mut merger = RunMerger::new(Arc::clone(&schema), runs, None);
        let mut sizes = vec![];
        let mut keys = vec![];
        while let Some(batch) = merger.next_batch(usize::MAX).unwrap() {
            sizes.push(batch.num_rows());
            keys.extend_from_slice(batch.column(0).as_primitive::<Int64Type>().values());
        }
        assert_eq!(sizes, [20_000, 20_000]);
        assert_eq!(keys, (0..40_000).collect::<Vec<_>>());
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
        let (first_memory, second_memory) = (first.memory_size(), second.memory_size());
        assert!(first_memory > 0 && second_memory > 0);
        let mut merger = RunMerger::new(schema(), vec![first, second], None);
        assert_eq!(merger.memory_size(), first_memory + second_memory);
        let batch = merger.next_batch(3).unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert!(merger.runs[0].is_none(), "the first run is exhausted");
        assert!(merger.runs[1].is_some());
        assert_eq!(merger.memory_size(), second_memory);
        let batch = merger.next_batch(3).unwrap().unwrap();
        assert_eq!(rows_of(&[batch]), vec![(Some(4), "4".to_string(), 4)]);
        assert!(merger.runs.iter().all(Option::is_none));
        assert_eq!(merger.memory_size(), 0);
        assert!(merger.next_batch(3).unwrap().is_none());
        // A merger with a fetch drops all runs once it emitted its rows.
        let first = run(&converter, options, vec![(Some(1), 1), (Some(2), 2)]);
        let second = run(&converter, options, vec![(Some(3), 3), (Some(4), 4)]);
        let mut merger = RunMerger::new(schema(), vec![first, second], Some(1));
        assert_eq!(merger.next_batch(3).unwrap().unwrap().num_rows(), 1);
        assert!(merger.runs.iter().all(Option::is_none));
        assert_eq!(merger.memory_size(), 0);
    }

    /// Random sort keys of `data_type`, with nulls and many ties: integers
    /// in `-4..4`, floats with NaNs, infinities and both zeros, and strings
    /// (also as binary values and views) with shared prefixes, longer and
    /// shorter than the 12 bytes that views inline.
    fn random_keys(rng: &mut StdRng, data_type: &DataType, len: usize) -> ArrayRef {
        let valid = |rng: &mut StdRng| rng.random_range(0..6) != 0;
        let words = [
            "",
            "a",
            "a\0",
            "ab",
            "b",
            "abcd",
            "abcdefghijkl",
            "abcdefghijklm",
            "abcdefghijklmnopq",
            "abcdzzzzzzzzzzzzzz",
            "abce",
            "\u{e9}t\u{e9}",
        ];
        match data_type {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                let values = (0..len).map(|_| {
                    valid(rng).then(|| match rng.random_range(0..8) {
                        0 => f64::NAN,
                        1 => -f64::NAN,
                        2 => -0.0,
                        3 => 0.0,
                        4 => f64::INFINITY,
                        5 => f64::NEG_INFINITY,
                        _ => f64::from(rng.random_range(-4..4)) / 2.0,
                    })
                });
                let floats: ArrayRef = Arc::new(Float64Array::from_iter(values));
                cast(&floats, data_type).unwrap()
            }
            DataType::Boolean => Arc::new(BooleanArray::from_iter(
                (0..len).map(|_| valid(rng).then(|| rng.random_bool(0.5))),
            )),
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView => {
                let strings: ArrayRef =
                    Arc::new(StringArray::from_iter((0..len).map(|_| {
                        valid(rng).then(|| words[rng.random_range(0..words.len())])
                    })));
                cast(&strings, data_type).unwrap()
            }
            DataType::FixedSizeBinary(size) => {
                let values = (0..len)
                    .map(|_| {
                        valid(rng).then(|| {
                            (0..*size)
                                .map(|_| rng.random_range(0..3))
                                .collect::<Vec<u8>>()
                        })
                    })
                    .collect::<Vec<_>>();
                Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        values.into_iter(),
                        *size,
                    )
                    .unwrap(),
                )
            }
            DataType::Interval(IntervalUnit::DayTime) => {
                Arc::new(IntervalDayTimeArray::from_iter((0..len).map(|_| {
                    valid(rng).then(|| {
                        IntervalDayTime::new(
                            rng.random_range(-2..2),
                            rng.random_range(-2..2),
                        )
                    })
                })))
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                Arc::new(IntervalMonthDayNanoArray::from_iter((0..len).map(|_| {
                    valid(rng).then(|| {
                        IntervalMonthDayNano::new(
                            rng.random_range(-2..2),
                            rng.random_range(-2..2),
                            rng.random_range(-2..2),
                        )
                    })
                })))
            }
            DataType::Decimal128(precision, scale) => Arc::new(
                Decimal128Array::from_iter(
                    (0..len)
                        .map(|_| valid(rng).then(|| i128::from(rng.random_range(-4..4)))),
                )
                .with_precision_and_scale(*precision, *scale)
                .unwrap(),
            ),
            DataType::Interval(IntervalUnit::YearMonth) => {
                Arc::new(IntervalYearMonthArray::from_iter(
                    (0..len).map(|_| valid(rng).then(|| rng.random_range(-4..4))),
                ))
            }
            DataType::Date32 | DataType::Time32(_) => {
                let ints: ArrayRef = Arc::new(Int32Array::from_iter(
                    (0..len).map(|_| valid(rng).then(|| rng.random_range(0..8))),
                ));
                cast(&ints, data_type).unwrap()
            }
            _ => {
                // Negative values too, if the type has them.
                let min = match data_type.is_unsigned_integer() {
                    true => 0,
                    false => -4,
                };
                let ints: ArrayRef = Arc::new(Int64Array::from_iter(
                    (0..len).map(|_| valid(rng).then(|| rng.random_range(min..min + 8))),
                ));
                cast(&ints, data_type).unwrap()
            }
        }
    }

    /// The runs `(k, id)` of the keys `keys` (one run per array), each sorted
    /// by the row format with `options`: with encoded keys, and with keys that
    /// are compared directly.
    fn runs_of_keys(
        keys: &[ArrayRef],
        options: SortOptions,
    ) -> (SchemaRef, Vec<SortedRun>, Vec<SortedRun>) {
        let data_type = keys[0].data_type().clone();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", data_type.clone(), true),
            Field::new("id", DataType::Int64, false),
        ]));
        let converter =
            RowConverter::new(vec![SortField::new_with_options(data_type, options)])
                .unwrap();
        let mut next_id = 0;
        let (mut encoded, mut direct) = (vec![], vec![]);
        for keys in keys {
            let rows = converter.convert_columns(&[Arc::clone(keys)]).unwrap();
            let mut order = (0..keys.len() as u32).collect::<Vec<_>>();
            order.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
            let sorted =
                arrow::compute::take(keys, &UInt32Array::from(order), None).unwrap();
            let ids = (next_id..next_id + keys.len() as i64).collect::<Vec<_>>();
            next_id += keys.len() as i64;
            let columns = vec![Arc::clone(&sorted), Arc::new(Int64Array::from(ids)) as _];
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            let rows = converter.convert_columns(&[Arc::clone(&sorted)]).unwrap();
            encoded.push(SortedRun::try_new(batch.clone(), rows).unwrap());
            let column = KeyColumn::try_new(sorted, options).unwrap();
            direct.push(
                SortedRun::try_new_with_keys(batch, RunKeys::Column(column)).unwrap(),
            );
        }
        (schema, encoded, direct)
    }

    /// The ids of the rows that `merger` emits, in batches of `max_rows`.
    fn merged_ids(mut merger: RunMerger, max_rows: usize) -> Vec<i64> {
        let mut ids = vec![];
        while let Some(batch) = merger.next_batch(max_rows).unwrap() {
            assert!(batch.num_rows() <= max_rows);
            ids.extend_from_slice(batch.column(1).as_primitive::<Int64Type>().values());
        }
        ids
    }

    #[test]
    fn keys_compared_directly_merge_like_encoded_keys() {
        // For every type that is compared directly and every sort option, the
        // merge emits the rows of runs of keys in the same order (ties in the
        // order of the runs) as with the keys encoded in the row format.
        let mut rng = StdRng::seed_from_u64(7);
        let data_types = [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal128(12, 2),
            DataType::Date32,
            DataType::Date64,
            DataType::Time32(TimeUnit::Second),
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("+01:00".into())),
            DataType::Duration(TimeUnit::Microsecond),
            DataType::Interval(IntervalUnit::YearMonth),
            DataType::Interval(IntervalUnit::DayTime),
            DataType::Interval(IntervalUnit::MonthDayNano),
            DataType::Boolean,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Utf8View,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::BinaryView,
            DataType::FixedSizeBinary(2),
        ];
        for data_type in &data_types {
            assert!(compares_directly(data_type), "{data_type}");
            for (descending, nulls_first) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                let options = SortOptions::new(descending, nulls_first);
                let num_runs = rng.random_range(1..12);
                let keys = (0..num_runs)
                    .map(|_| {
                        let len = match rng.random_range(0..4) {
                            0 => rng.random_range(0..3),
                            _ => rng.random_range(0..200),
                        };
                        random_keys(&mut rng, data_type, len)
                    })
                    .collect::<Vec<_>>();
                let (schema, encoded, direct) = runs_of_keys(&keys, options);
                let expected =
                    merged_ids(RunMerger::new(Arc::clone(&schema), encoded, None), 64);
                let total = keys.iter().map(|keys| keys.len()).sum::<usize>();
                assert_eq!(expected.len(), total);
                for (fetch, max_rows) in [(None, 2048), (None, 7), (Some(5), 3)] {
                    let direct = direct
                        .iter()
                        .map(|run| {
                            SortedRun::try_new_with_keys(
                                run.batch().clone(),
                                run.keys().clone(),
                            )
                            .unwrap()
                        })
                        .collect();
                    let merger = RunMerger::new(Arc::clone(&schema), direct, fetch);
                    let len = fetch.map_or(total, |fetch| fetch.min(total));
                    assert_eq!(
                        merged_ids(merger, max_rows),
                        expected[..len],
                        "{data_type}, {options:?}, fetch {fetch:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn runs_whose_keys_differ_in_kind_are_not_merged() {
        let options = SortOptions::default();
        let converter = converter(options);
        let encoded = run(&converter, options, vec![(Some(1), 1), (Some(2), 2)]);
        let direct = |data_type: &DataType| {
            let keys = random_keys(&mut StdRng::seed_from_u64(1), data_type, 2);
            let (_, _, mut runs) = runs_of_keys(&[keys], options);
            runs.pop().unwrap()
        };
        let merger =
            RunMerger::new(schema(), vec![encoded, direct(&DataType::Int32)], None);
        let mut merger = merger;
        let err = merger.next_batch(10).unwrap_err();
        assert!(err.to_string().contains("keys of different kinds"), "{err}");
        // Keys of different types.
        let (schema, _, mut runs) = runs_of_keys(
            &[random_keys(
                &mut StdRng::seed_from_u64(1),
                &DataType::Int32,
                2,
            )],
            options,
        );
        runs.push(direct(&DataType::Int64));
        let err = RunMerger::new(schema, runs, None)
            .next_batch(10)
            .unwrap_err();
        assert!(err.to_string().contains("keys of different kinds"), "{err}");
        // A key column of a type that is not compared directly.
        let list = Arc::new(ListArray::new_null(
            Arc::new(Field::new("item", DataType::Int32, true)),
            1,
        ));
        let err = KeyColumn::try_new(list, options).unwrap_err();
        assert!(
            err.to_string().contains("cannot be compared directly"),
            "{err}"
        );
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
