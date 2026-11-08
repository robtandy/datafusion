// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalSort` and `PhysicalTopK`: parallel sort and top-k.
//!
//! [`PhysicalSort`] (`ORDER BY`) and [`PhysicalTopK`] (`ORDER BY ... LIMIT
//! k`) replace DataFusion's `SortExec`, together with the
//! `SortPreservingMergeExec` or `CoalescePartitionsExec` that merges its
//! partitions. Like DuckDB's `PhysicalOrder` and `PhysicalTopN`, both are
//! pipeline breakers that sort in parallel and merge once:
//!
//! * **Sink**: every thread of the input pipeline collects the chunks it
//!   pushes in its local sink state (compacting their selection vectors).
//!   [`PhysicalSort`] sorts them into *runs* of bounded size
//!   (`lexsort_to_indices` and `take` for a single sort key of a primitive,
//!   boolean, string or binary type, the Arrow row format otherwise).
//!   [`PhysicalTopK`] keeps the best `k` rows of the thread instead and
//!   discards the rows that cannot make it into the result as they arrive,
//!   see [`topk_sink`].
//! * **Combine**: every thread hands its sorted runs to the global sink state.
//! * **Finalize**: a [`RunMerger`] is set up over the runs of all threads.
//! * **Source**: the sort is the source of the pipeline that consumes its
//!   output. The merger merges the runs with a tree of losers (see [`merge`])
//!   and emits the rows in order, in chunks of at most
//!   [`DEFAULT_VECTOR_SIZE`] rows, stopping after `fetch` rows. The source is
//!   not parallel: the pipeline that reads the sorted output runs on a single
//!   thread, which keeps the order, as DataFusion reads the single partition
//!   of the merge.
//!
//! The sort keys of the runs are encoded in the Arrow row format, in which
//! comparing two keys is a `memcmp`, so the merge works for every sort key
//! the row format supports.
//!
//! The native operators keep all rows in memory: they do not spill and do
//! not reserve memory in the memory pool. As in DataFusion, rows with equal
//! sort keys may come in any order.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::compute::{SortColumn, concat, lexsort_to_indices, take, take_arrays};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_row::{RowConverter, SortField};
use arrow_select::interleave::interleave;
use datafusion_common::{Result, exec_datafusion_err, internal_err};
use datafusion_physical_expr::LexOrdering;
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{GlobalSourceState, SourceResult, downcast_state};

pub mod merge;
pub mod sort_sink;
pub mod topk_sink;

pub use merge::{RunMerger, SortedRun};
pub use sort_sink::PhysicalSort;
pub use topk_sink::{PhysicalTopK, TopKDynamicFilter};

/// The sort keys of a native sort: evaluates the sort expressions and sorts
/// rows into [`SortedRun`]s, whose keys it encodes in the Arrow row format
/// (with the same [`RowConverter`] for all runs, so that they can be
/// merged).
#[derive(Debug)]
pub(crate) struct SortKeys {
    ordering: LexOrdering,
    converter: RowConverter,
    /// Sort with `lexsort_to_indices` rather than by the encoded keys: a
    /// single sort key of a type that `sort_to_indices` orders exactly like
    /// the row format (primitive, boolean, string and binary types).
    sort_with_kernel: bool,
}

impl SortKeys {
    /// The sort keys `ordering` of rows with schema `schema`.
    pub(crate) fn try_new(ordering: LexOrdering, schema: &Schema) -> Result<Self> {
        let types = data_types(&ordering, schema)?;
        let fields = ordering
            .iter()
            .zip(&types)
            .map(|(sort_expr, data_type)| {
                SortField::new_with_options(data_type.clone(), sort_expr.options)
            })
            .collect::<Vec<_>>();
        let converter = RowConverter::new(fields)?;
        let sort_with_kernel =
            matches!(types.as_slice(), [data_type] if sorts_like_rows(data_type));
        Ok(Self {
            ordering,
            converter,
            sort_with_kernel,
        })
    }

    /// The sort keys.
    pub(crate) fn ordering(&self) -> &LexOrdering {
        &self.ordering
    }

    /// The values of the sort keys of the rows of `batch`.
    pub(crate) fn evaluate(&self, batch: &RecordBatch) -> Result<Vec<ArrayRef>> {
        self.ordering
            .iter()
            .map(|sort_expr| sort_expr.expr.evaluate(batch)?.into_array(batch.num_rows()))
            .collect()
    }

    /// Sort the rows of `batches` into a run with schema `schema`, keeping
    /// only the first `limit` rows (all rows if `None`). Returns `None` if
    /// there are no rows to keep.
    pub(crate) fn sort(
        &self,
        schema: &SchemaRef,
        batches: &[RecordBatch],
        limit: Option<usize>,
    ) -> Result<Option<SortedRun>> {
        let batches = batches
            .iter()
            .filter(|batch| batch.num_rows() > 0)
            .collect::<Vec<_>>();
        let num_rows = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        let len = limit.map_or(num_rows, |limit| limit.min(num_rows));
        if len == 0 {
            return Ok(None);
        }
        let num_rows_u32 = u32::try_from(num_rows)
            .map_err(|_| exec_datafusion_err!("Cannot sort {num_rows} rows at once"))?;
        let limit = (len < num_rows).then_some(len);
        let keys = batches
            .iter()
            .map(|batch| self.evaluate(batch))
            .collect::<Result<Vec<_>>>()?;

        // The positions of the sorted rows in the concatenation of `batches`,
        // and the encoded keys of the sorted rows.
        let (indices, sorted_keys) = if self.sort_with_kernel {
            let key = match keys.as_slice() {
                [keys] => Arc::clone(&keys[0]),
                _ => {
                    concat(&keys.iter().map(|keys| keys[0].as_ref()).collect::<Vec<_>>())?
                }
            };
            let column = SortColumn {
                values: Arc::clone(&key),
                options: Some(self.ordering[0].options),
            };
            let indices = lexsort_to_indices(&[column], limit)?;
            let sorted = take(key.as_ref(), &indices, None)?;
            let sorted_keys = self.converter.convert_columns(&[sorted])?;
            (indices, sorted_keys)
        } else {
            let mut rows = self.converter.empty_rows(num_rows, 0);
            for keys in &keys {
                self.converter.append(&mut rows, keys)?;
            }
            let mut order = (0..num_rows_u32).collect::<Vec<_>>();
            let compare =
                |a: &u32, b: &u32| rows.row(*a as usize).cmp(&rows.row(*b as usize));
            if len < num_rows {
                order.select_nth_unstable_by(len - 1, compare);
                order.truncate(len);
            }
            order.sort_unstable_by(compare);
            let data_len = order.iter().map(|&i| rows.row_len(i as usize)).sum();
            let mut sorted_keys = self.converter.empty_rows(len, data_len);
            for &i in &order {
                sorted_keys.push(rows.row(i as usize));
            }
            (UInt32Array::from(order), sorted_keys)
        };

        let columns = match batches.as_slice() {
            [batch] => take_arrays(batch.columns(), &indices, None)?,
            _ => {
                // The batch and row of every sorted row.
                let mut starts = Vec::with_capacity(batches.len());
                let mut start = 0;
                for batch in &batches {
                    starts.push(start);
                    start += batch.num_rows();
                }
                let positions = indices
                    .values()
                    .iter()
                    .map(|&index| {
                        let index = index as usize;
                        let batch = starts.partition_point(|&start| start <= index) - 1;
                        (batch, index - starts[batch])
                    })
                    .collect::<Vec<_>>();
                (0..schema.fields().len())
                    .map(|column| {
                        let values = batches
                            .iter()
                            .map(|batch| batch.column(column).as_ref())
                            .collect::<Vec<&dyn Array>>();
                        Ok(interleave(&values, &positions)?)
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };
        let options = RecordBatchOptions::new().with_row_count(Some(len));
        let batch =
            RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)?;
        Ok(Some(SortedRun::try_new(batch, sorted_keys)?))
    }
}

/// The data types of the sort keys `ordering` of rows with schema `schema`.
fn data_types(ordering: &LexOrdering, schema: &Schema) -> Result<Vec<DataType>> {
    ordering
        .iter()
        .map(|sort_expr| sort_expr.expr.data_type(schema))
        .collect()
}

/// Whether `sort_to_indices` sorts values of `data_type` exactly like the row
/// format orders them (types without children and dictionaries).
fn sorts_like_rows(data_type: &DataType) -> bool {
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

/// The sorted output of a [`PhysicalSort`] or [`PhysicalTopK`], set up when
/// the sink is finalized and handed to the source.
#[derive(Debug, Default)]
pub(crate) struct SortedOutput {
    merger: Mutex<Option<RunMerger>>,
}

impl SortedOutput {
    /// Set the merger of the runs collected by the sink.
    pub(crate) fn set(&self, merger: RunMerger) {
        *self.merger.lock() = Some(merger);
    }

    /// The global source state of the operator `name`: the merger, which
    /// the source takes over.
    pub(crate) fn take_source(&self, name: &str) -> Result<Arc<dyn GlobalSourceState>> {
        match self.merger.lock().take() {
            Some(merger) => Ok(Arc::new(MergeSource {
                merger: Mutex::new(merger),
            })),
            None => internal_err!("{name} is read before its sink is finalized"),
        }
    }
}

/// The global source state of a [`PhysicalSort`] or [`PhysicalTopK`]. The
/// source is not parallel, so a single thread reads the merger.
#[derive(Debug)]
struct MergeSource {
    merger: Mutex<RunMerger>,
}

impl GlobalSourceState for MergeSource {}

/// The next chunk of the sorted output of the source state `gstate` of the
/// operator `name`.
pub(crate) fn next_sorted_chunk(
    gstate: &dyn GlobalSourceState,
    name: &str,
) -> Result<SourceResult> {
    let source = downcast_state::<MergeSource>(gstate.as_any(), name)?;
    let batch = source.merger.lock().next_batch(DEFAULT_VECTOR_SIZE)?;
    Ok(match batch {
        Some(batch) => SourceResult::Chunk(DataChunk::from_record_batch(batch)),
        None => SourceResult::Finished,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        BooleanArray, Date32Array, Decimal128Array, DictionaryArray, Float64Array,
        Int32Array, StringArray, StringViewArray, TimestampNanosecondArray,
    };
    use arrow::compute::SortOptions;
    use arrow::datatypes::{Field, Int32Type};
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::col;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // ---------------------------------------------------------------------
    // Sort keys
    // ---------------------------------------------------------------------

    /// The sort keys `columns` (with `options`) of a schema with these
    /// columns.
    fn sort_keys(schema: &SchemaRef, keys: &[(&str, SortOptions)]) -> SortKeys {
        let ordering = LexOrdering::new(keys.iter().map(|(name, options)| {
            PhysicalSortExpr::new(col(name, schema).unwrap(), *options)
        }))
        .unwrap();
        SortKeys::try_new(ordering, schema).unwrap()
    }

    /// Random columns of every type the sorts handle specially, with nulls.
    fn random_columns(rng: &mut StdRng, n: usize) -> Vec<ArrayRef> {
        let null = |rng: &mut StdRng| rng.random_range(0..8) == 0;
        let words = [
            "",
            "a",
            "ab",
            "b",
            "a fairly long string that is not inlined",
        ];
        let int = Int32Array::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_range(-5..5))),
        );
        let float = Float64Array::from_iter((0..n).map(|_| {
            (!null(rng)).then(|| match rng.random_range(0..6) {
                0 => f64::NAN,
                1 => -0.0,
                2 => 0.0,
                3 => f64::NEG_INFINITY,
                _ => f64::from(rng.random_range(-3..3)) / 2.0,
            })
        }));
        let string = StringArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| words[rng.random_range(0..words.len())])),
        );
        let view = StringViewArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| words[rng.random_range(0..words.len())])),
        );
        let boolean = BooleanArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_bool(0.5))),
        );
        let decimal = Decimal128Array::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| i128::from(rng.random_range(-300..300)))),
        )
        .with_precision_and_scale(12, 2)
        .unwrap();
        let date = Date32Array::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_range(0..9))),
        );
        let ts = TimestampNanosecondArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_range(-4..4))),
        );
        let dictionary = DictionaryArray::<Int32Type>::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| words[rng.random_range(0..words.len())])),
        );
        vec![
            Arc::new(int),
            Arc::new(float),
            Arc::new(string),
            Arc::new(view),
            Arc::new(boolean),
            Arc::new(decimal),
            Arc::new(date),
            Arc::new(ts),
            Arc::new(dictionary),
        ]
    }

    #[test]
    fn sorted_runs_are_sorted_by_their_keys() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(1);
        let all_options = [
            SortOptions::new(false, false),
            SortOptions::new(false, true),
            SortOptions::new(true, false),
            SortOptions::new(true, true),
        ];
        for case in 0..40 {
            let columns = random_columns(&mut rng, 500);
            let fields = columns
                .iter()
                .enumerate()
                .map(|(i, column)| {
                    Field::new(format!("c{i}"), column.data_type().clone(), true)
                })
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
            // Several batches of different sizes, one of them empty.
            let batches = [
                batch.slice(0, 120),
                batch.slice(120, 0),
                batch.slice(120, 380),
            ];
            let num_keys = rng.random_range(1..4);
            let names = (0..num_keys)
                .map(|_| format!("c{}", rng.random_range(0..schema.fields().len())))
                .collect::<Vec<_>>();
            let mut keys = vec![];
            for name in &names {
                if !keys.iter().any(|(key, _)| key == name) {
                    keys.push((name.as_str(), all_options[rng.random_range(0..4)]));
                }
            }
            let sort_keys = sort_keys(&schema, &keys);
            for limit in [None, Some(1), Some(77), Some(499), Some(500), Some(800)] {
                let run = sort_keys.sort(&schema, &batches, limit)?.expect("rows");
                assert_eq!(run.num_rows(), limit.unwrap_or(500).min(500));
                // The keys are the encoded keys of the rows of the run, in
                // order.
                let encoded = sort_keys
                    .converter
                    .convert_columns(&sort_keys.evaluate(run.batch())?)?;
                for row in 0..run.num_rows() {
                    assert_eq!(run.key(row), encoded.row(row), "case {case}");
                    if row > 0 {
                        assert!(
                            run.key(row - 1) <= run.key(row),
                            "case {case}: {keys:?}"
                        );
                    }
                }
                // The first rows: no row left out comes before the last row.
                let all = sort_keys
                    .converter
                    .convert_columns(&sort_keys.evaluate(&batch)?)?;
                if let Some(last) = run.num_rows().checked_sub(1) {
                    let before = (0..500).filter(|&i| all.row(i) < run.key(last)).count();
                    assert!(before <= last, "case {case}: {keys:?}, limit {limit:?}");
                }
                // A single key of a simple type is sorted with the sort
                // kernel, in the same order as by the row format.
                if sort_keys.sort_with_kernel {
                    let mut by_rows =
                        SortKeys::try_new(sort_keys.ordering.clone(), &schema)?;
                    by_rows.sort_with_kernel = false;
                    let other = by_rows.sort(&schema, &batches, limit)?.expect("rows");
                    for row in 0..run.num_rows() {
                        assert_eq!(run.key(row), other.key(row), "case {case}: {keys:?}");
                    }
                }
            }
            assert!(sort_keys.sort(&schema, &batches, Some(0))?.is_none());
            assert!(sort_keys.sort(&schema, &[], None)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn simple_single_keys_are_sorted_with_the_kernel() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("s", DataType::Utf8View, true),
            Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        let options = SortOptions::default();
        assert!(sort_keys(&schema, &[("i", options)]).sort_with_kernel);
        assert!(sort_keys(&schema, &[("s", options)]).sort_with_kernel);
        assert!(!sort_keys(&schema, &[("d", options)]).sort_with_kernel);
        assert!(!sort_keys(&schema, &[("i", options), ("s", options)]).sort_with_kernel);
    }
}
