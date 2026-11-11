// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`CachingOperator`]: coalesces the small chunks produced by selective
//! operators, like DuckDB's `CachingPhysicalOperator`.
//!
//! A selective operator such as a
//! [`PhysicalFilter`](super::filter::PhysicalFilter) can turn full input
//! chunks into tiny output chunks (a filter that keeps 1% of the rows emits
//! chunks of about 20 rows). Every chunk has a fixed processing cost in the
//! operators downstream (expression evaluation, hash table probes, ...), so
//! many tiny chunks are slow. A `CachingOperator` wraps such an operator (it
//! is placed directly after it in the pipeline): chunks with fewer than
//! [`CACHE_THRESHOLD`] rows are compacted and appended to a cache in the
//! thread-local [`OperatorState`], which is emitted as one chunk once it
//! holds [`DEFAULT_VECTOR_SIZE`] rows. The rest of the cache is emitted by
//! [`PhysicalOperator::final_execute`] once the input is exhausted. Larger
//! chunks are passed through unchanged, with their selection vectors.
//!
//! Rows are emitted in the order they arrive: when a large chunk arrives
//! while rows are cached, the cache is emitted first and the chunk on the
//! next call (`HaveMoreOutput`), like DuckDB's caching operators do when
//! the order matters. So a single-threaded pipeline keeps the order of its
//! input.
//!
//! The cache owns the rows it holds: they are copied out of their chunks,
//! so the cache does not keep the buffers of the chunks alive (a slice of a
//! column of 2048 strings keeps all of them). Columns whose copy would still
//! share buffers with the chunk (dictionaries, and view arrays nested in
//! other types) are not cached, see [`can_cache`].
//!
//! Since chunks are only emitted once the cache is full, caching delays the
//! output of a pipeline: it should not be used below a `LIMIT` that may stop
//! the pipeline early (DuckDB does not cache in such pipelines either), whose
//! input may never end, e.g. a stream that stays pending once it has produced
//! the rows of the limit. The [`PipelineQueryPlanner`](crate::PipelineQueryPlanner)
//! does not cache the filters of such pipelines.

use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::datatypes::{DataType, SchemaRef};
use arrow_select::coalesce::BatchCoalescer;
use arrow_select::take::take;
use datafusion_common::{Result, plan_err};

use super::projection::can_gather;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorFinalizeResult, OperatorResult,
    OperatorState, PhysicalOperator, downcast_state_mut,
};

/// Chunks with fewer rows than this are cached by a [`CachingOperator`] by
/// default.
///
/// DuckDB uses 64 rows for vectors of 2048 rows. Arrow chunks have a higher
/// fixed cost per chunk (an `ArrayRef` per column, and every kernel call
/// allocates its result), so chunks are cached up to an eighth of
/// [`DEFAULT_VECTOR_SIZE`].
pub const CACHE_THRESHOLD: usize = 256;

/// Coalesces the small chunks of its input into full chunks.
///
/// See the [module documentation](self).
#[derive(Debug)]
pub struct CachingOperator {
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    /// Chunks with fewer rows are cached.
    threshold: usize,
    /// Whether chunks of this schema can be cached, see [`can_cache`].
    caching_supported: bool,
}

impl CachingOperator {
    /// Cache the chunks of `input` that have fewer than [`CACHE_THRESHOLD`]
    /// rows.
    pub fn new(input: Arc<dyn PhysicalOperator>) -> Self {
        let schema = input.schema();
        let caching_supported = schema
            .fields()
            .iter()
            .all(|field| can_cache(field.data_type()));
        Self {
            children: vec![input],
            schema,
            threshold: CACHE_THRESHOLD,
            caching_supported,
        }
    }

    /// Cache the chunks that have fewer than `threshold` rows instead, at
    /// most [`DEFAULT_VECTOR_SIZE`]. A threshold of `0` or `1` disables
    /// caching.
    pub fn try_with_threshold(mut self, threshold: usize) -> Result<Self> {
        if threshold > DEFAULT_VECTOR_SIZE {
            return plan_err!(
                "The threshold of a CachingOperator must be at most {DEFAULT_VECTOR_SIZE}, got {threshold}"
            );
        }
        self.threshold = threshold;
        Ok(self)
    }

    /// Chunks with fewer rows than this are cached.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Whether the chunks of the input can be cached at all, see
    /// [`can_cache`]. If not, all chunks are passed through.
    pub fn caching_supported(&self) -> bool {
        self.caching_supported
    }
}

/// The state of a [`CachingOperator`] in one thread: the cached rows.
struct CachingState {
    /// Accumulates the cached rows into chunks of [`DEFAULT_VECTOR_SIZE`]
    /// rows. It owns them: they are copied before they are pushed (see
    /// [`owned_rows`]), and it compacts view arrays.
    cache: BatchCoalescer,
}

impl OperatorState for CachingState {}

impl CachingState {
    /// Cache the rows of `chunk`.
    fn push(&mut self, chunk: &DataChunk) -> Result<()> {
        self.cache.push_batch(owned_rows(chunk)?)?;
        Ok(())
    }

    /// Move the next full chunk of the cache, if any, to `output`.
    fn emit_completed(&mut self, output: &mut DataChunk) {
        if let Some(batch) = self.cache.next_completed_batch() {
            *output = DataChunk::from_record_batch(batch);
        }
    }

    /// Move all cached rows to `output`.
    fn emit_all(&mut self, output: &mut DataChunk) -> Result<()> {
        self.cache.finish_buffered_batch()?;
        self.emit_completed(output);
        Ok(())
    }
}

impl PhysicalOperator for CachingOperator {
    fn name(&self) -> &str {
        "CachingOperator"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        if self.caching_supported {
            format!("CachingOperator: threshold={}", self.threshold)
        } else {
            "CachingOperator: disabled".to_owned()
        }
    }

    fn get_operator_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(CachingState {
            cache: BatchCoalescer::new(Arc::clone(&self.schema), DEFAULT_VECTOR_SIZE),
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
        let state = downcast_state_mut::<CachingState>(lstate.as_any_mut(), self.name())?;
        if self.caching_supported && input.len() < self.threshold {
            // A small chunk: cache its rows, emit the cache once it is full.
            state.push(input)?;
            state.emit_completed(output);
            return Ok(OperatorResult::NeedMoreInput);
        }
        if state.cache.get_buffered_rows() > 0 {
            // Keep the order of the rows: emit the cached rows first, and
            // the input when called again with it.
            state.emit_all(output)?;
            return Ok(OperatorResult::HaveMoreOutput);
        }
        *output = input.clone();
        Ok(OperatorResult::NeedMoreInput)
    }

    fn requires_final_execute(&self) -> bool {
        self.caching_supported && self.threshold > 1
    }

    fn final_execute(
        &self,
        _ctx: &ExecutionContext,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorFinalizeResult> {
        let state = downcast_state_mut::<CachingState>(lstate.as_any_mut(), self.name())?;
        state.emit_all(output)?;
        Ok(OperatorFinalizeResult::Finished)
    }
}

/// The logical rows of `chunk`, copied into arrays that do not share any
/// buffer with the columns of `chunk`.
///
/// `DataChunk::compact` slices the columns for contiguous selections, and
/// the coalescer buffers slices of most arrays until it emits them: a cached
/// row would keep the buffers of its whole chunk alive. `take` copies the
/// values of all the types [`can_cache`] accepts (view arrays share their
/// data buffers, which the coalescer compacts).
fn owned_rows(chunk: &DataChunk) -> Result<RecordBatch> {
    let indices = match chunk.selection() {
        Some(selection) if chunk.num_columns() > 0 => selection.to_uint32_array(),
        _ => UInt32Array::from_iter_values(0..chunk.len() as u32),
    };
    let columns = chunk
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()?;
    let options = RecordBatchOptions::new().with_row_count(Some(chunk.len()));
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(chunk.schema()),
        columns,
        &options,
    )?)
}

/// Whether columns of `data_type` can be cached.
///
/// The cache must own its rows (they are copied with Arrow's `take` kernel
/// before they are cached), and concatenates the rows of different chunks.
///
/// * Dictionaries are not cached: copying their rows keeps their whole
///   dictionary, and concatenating the dictionaries of different chunks may
///   need more keys than the key type has, even if every chunk is valid
///   (DataFusion only concatenates the batches of a single partition, which a
///   pipeline does not know about).
/// * View arrays share their data buffers after a copy: the coalescer
///   compacts them, but not when they are nested in other types.
/// * Run-end encoded and list view arrays are not cached either.
/// * Lists of values that take less than a byte each (e.g. fixed-size lists
///   of nulls, booleans, or empty lists or binaries), which `take` cannot
///   copy, are not cached, nor run-end encoded arrays whose child fields are
///   not Arrow's.
/// * Strings, binaries, lists and maps with 32-bit offsets (`Utf8`,
///   `Binary`, `List`, `Map`, at any depth) are not cached: their offsets
///   overflow beyond 2 GiB of bytes (or `i32::MAX` values) in the cache,
///   which chunks that share their buffers reach without memory (e.g. 1024
///   rows that share a string of 2 MiB), while DataFusion may pass them on
///   as they are (e.g. a `FilterExec` with batches of one row). Their large
///   types are cached.
pub fn can_cache(data_type: &DataType) -> bool {
    can_gather(data_type) && can_cache_nested(data_type, false)
}

fn can_cache_nested(data_type: &DataType, nested: bool) -> bool {
    match data_type {
        DataType::Dictionary(_, _)
        | DataType::RunEndEncoded(_, _)
        | DataType::ListView(_)
        | DataType::LargeListView(_) => false,
        // Their 32-bit offsets overflow beyond 2 GiB of bytes (or `i32::MAX`
        // values) in the cache, which the chunks it concatenates reach without
        // memory if they share their buffers.
        DataType::Utf8 | DataType::Binary | DataType::List(_) | DataType::Map(..) => {
            false
        }
        DataType::Utf8View | DataType::BinaryView => !nested,
        DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            can_cache_nested(field.data_type(), true)
        }
        DataType::Struct(fields) => fields
            .iter()
            .all(|field| can_cache_nested(field.data_type(), true)),
        DataType::Union(fields, _) => fields
            .iter()
            .all(|(_, field)| can_cache_nested(field.data_type(), true)),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        Array, ArrayRef, AsArray, DictionaryArray, Int64Array, RecordBatch, StringArray,
        StringViewArray,
    };
    use arrow::datatypes::{Field, Fields, Int8Type, Int32Type, Int64Type, Schema};
    use datafusion::prelude::SessionContext;

    use crate::chunk::SelectionVector;
    use crate::operators::filter::test_support::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8View, true),
        ]))
    }

    /// A chunk of the rows `start..start + rows`, with a selection of every
    /// `step`-th row.
    fn chunk(start: i64, rows: usize, step: usize) -> DataChunk {
        let ids = start..start + rows as i64;
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(ids.clone())),
                Arc::new(StringViewArray::from_iter(ids.map(|i| {
                    (i % 3 != 0)
                        .then(|| format!("a string view longer than 12 bytes {i}"))
                }))),
            ],
        )
        .unwrap();
        let selection = (0..rows as u32).step_by(step).collect::<Vec<_>>();
        DataChunk::from_record_batch(batch)
            .with_selection(SelectionVector::from(selection))
            .unwrap()
    }

    fn caching() -> CachingOperator {
        CachingOperator::new(source(schema(), vec![]))
    }

    fn ids(chunks: &[DataChunk]) -> Vec<i64> {
        chunks
            .iter()
            .flat_map(|chunk| {
                chunk
                    .compact()
                    .unwrap()
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[test]
    fn coalesces_small_chunks_into_full_chunks() {
        // 100 chunks of 2048 rows of which 32 are selected.
        let inputs = (0..100)
            .map(|i| chunk(i * 2048, 2048, 64))
            .collect::<Vec<_>>();
        assert!(inputs.iter().all(|c| c.len() == 32));
        let (outputs, finished) = drive(&caching(), inputs.clone()).unwrap();
        assert!(!finished);
        // 3200 rows: one full chunk, and the rest when the input is exhausted.
        assert_eq!(
            outputs.iter().map(DataChunk::len).collect::<Vec<_>>(),
            vec![DEFAULT_VECTOR_SIZE, 3200 - DEFAULT_VECTOR_SIZE]
        );
        assert!(outputs.iter().all(|chunk| chunk.selection().is_none()));
        assert!(outputs.iter().all(|chunk| chunk.schema() == &schema()));
        // All rows, in order.
        assert_eq!(ids(&outputs), ids(&inputs));
        let expected = inputs
            .iter()
            .map(|chunk| chunk.compact().unwrap())
            .collect::<Vec<_>>();
        let actual = outputs
            .iter()
            .map(|chunk| chunk.compact().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            arrow::compute::concat_batches(&schema(), &actual).unwrap(),
            arrow::compute::concat_batches(&schema(), &expected).unwrap()
        );
    }

    #[test]
    fn large_chunks_pass_through_and_the_order_is_kept() {
        let small = [chunk(0, 100, 10), chunk(100, 100, 5)];
        let large = chunk(1000, 2048, 2);
        let last = chunk(5000, 10, 1);
        let inputs = [
            small[0].clone(),
            small[1].clone(),
            large.clone(),
            last.clone(),
        ];
        let (outputs, _) = drive(&caching(), inputs.clone()).unwrap();
        assert_eq!(
            outputs.iter().map(DataChunk::len).collect::<Vec<_>>(),
            vec![30, 1024, 10]
        );
        assert_eq!(ids(&outputs), ids(&inputs));
        // The large chunk is passed through as is, selection included.
        assert_eq!(outputs[1].selection(), large.selection());
        assert!(Arc::ptr_eq(outputs[1].column(1), large.column(1)));

        // The cached rows are emitted before a large chunk, which is passed
        // through when the operator is called again with it.
        let operator = caching();
        let (gstate, mut lstate) = states(&operator);
        let ctx = context();
        let mut output = DataChunk::empty(schema());
        let result = operator
            .execute(
                &ctx,
                &small[0],
                &mut output,
                gstate.as_ref(),
                lstate.as_mut(),
            )
            .unwrap();
        assert!(matches!(result, OperatorResult::NeedMoreInput));
        assert!(output.is_empty());
        let result = operator
            .execute(&ctx, &large, &mut output, gstate.as_ref(), lstate.as_mut())
            .unwrap();
        assert!(matches!(result, OperatorResult::HaveMoreOutput));
        assert_eq!(output.len(), 10);
        output.reset();
        let result = operator
            .execute(&ctx, &large, &mut output, gstate.as_ref(), lstate.as_mut())
            .unwrap();
        assert!(matches!(result, OperatorResult::NeedMoreInput));
        assert_eq!(output.len(), 1024);
        output.reset();
        let result = operator
            .final_execute(&ctx, &mut output, gstate.as_ref(), lstate.as_mut())
            .unwrap();
        assert_eq!(result, OperatorFinalizeResult::Finished);
        assert!(output.is_empty());
    }

    #[test]
    fn threshold_is_configurable() {
        let inputs = (0..10)
            .map(|i| chunk(i * 2048, 2048, 2))
            .collect::<Vec<_>>();
        // Chunks of 1024 rows are not cached by default ...
        let (outputs, _) = drive(&caching(), inputs.clone()).unwrap();
        assert_eq!(outputs.len(), 10);
        // ... but with a threshold of 2048.
        let operator = caching().try_with_threshold(DEFAULT_VECTOR_SIZE).unwrap();
        assert_eq!(operator.threshold(), DEFAULT_VECTOR_SIZE);
        let (outputs, _) = drive(&operator, inputs.clone()).unwrap();
        assert_eq!(
            outputs.iter().map(DataChunk::len).collect::<Vec<_>>(),
            vec![2048; 5]
        );
        assert_eq!(ids(&outputs), ids(&inputs));

        // A threshold of 0 disables caching.
        let operator = caching().try_with_threshold(0).unwrap();
        assert!(!operator.requires_final_execute());
        let inputs = vec![chunk(0, 100, 10), chunk(100, 100, 10)];
        let (outputs, _) = drive(&operator, inputs.clone()).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].selection(), inputs[0].selection());

        let err = caching()
            .try_with_threshold(DEFAULT_VECTOR_SIZE + 1)
            .unwrap_err();
        assert!(err.to_string().contains("at most 2048"), "{err}");
        assert_eq!(caching().description(), "CachingOperator: threshold=256");
    }

    #[test]
    fn cached_view_arrays_do_not_keep_the_input_buffers() {
        let inputs = (0..10)
            .map(|i| chunk(i * 2048, 2048, 100))
            .collect::<Vec<_>>();
        let input_bytes = inputs[0].column(1).as_string_view().data_buffers()[0].len();
        let (outputs, _) = drive(&caching(), inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let views = outputs[0].column(1).as_string_view();
        let output_bytes = views.data_buffers().iter().map(|b| b.len()).sum::<usize>();
        // Only the strings of the 210 cached rows are kept.
        assert!(
            output_bytes < input_bytes,
            "{output_bytes} vs {input_bytes}"
        );
    }

    #[test]
    fn chunks_without_columns_are_cached_as_row_counts() {
        let schema = Arc::new(Schema::empty());
        let operator = CachingOperator::new(source(Arc::clone(&schema), vec![]));
        let inputs = (0..100).map(|_| {
            DataChunk::try_new_without_columns(Arc::clone(&schema), 30).unwrap()
        });
        let (outputs, _) = drive(&operator, inputs).unwrap();
        assert_eq!(
            outputs.iter().map(DataChunk::len).collect::<Vec<_>>(),
            vec![2048, 3000 - 2048]
        );
    }

    #[test]
    fn dictionaries_and_nested_views_are_not_cached() {
        let dictionary =
            |key: DataType| DataType::Dictionary(Box::new(key), Box::new(DataType::Utf8));
        assert!(!can_cache(&dictionary(DataType::Int8)));
        assert!(!can_cache(&dictionary(DataType::Int32)));
        let nested = DataType::Struct(Fields::from(vec![Field::new(
            "d",
            DataType::new_list(dictionary(DataType::Int64), true),
            true,
        )]));
        assert!(!can_cache(&nested));
        // The coalescer compacts view arrays, but not nested ones.
        assert!(can_cache(&DataType::Utf8View));
        assert!(!can_cache(&DataType::new_list(DataType::Utf8View, true)));
        assert!(can_cache(&DataType::new_large_list(
            DataType::LargeUtf8,
            true
        )));
        assert!(can_cache(&DataType::Struct(Fields::from(vec![
            Field::new("s", DataType::LargeUtf8, true)
        ]))));
        // Nor strings, binaries, lists or maps with 32-bit offsets, which
        // concatenating the rows of many chunks may overflow.
        for data_type in [
            DataType::Utf8,
            DataType::Binary,
            DataType::new_list(DataType::Int32, true),
            DataType::new_large_list(DataType::Utf8, true),
            DataType::Struct(Fields::from(vec![Field::new("s", DataType::Binary, true)])),
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(Fields::from(vec![
                        Field::new("key", DataType::Int32, false),
                        Field::new("value", DataType::Int32, true),
                    ])),
                    false,
                )),
                false,
            ),
        ] {
            assert!(!can_cache(&data_type), "{data_type}");
        }
        assert!(can_cache(&DataType::LargeBinary));
        // Nor lists of values that take less than a byte each: copying them
        // with `take` overflows the 32-bit indices of the values of fixed-size
        // lists.
        let nulls = Arc::new(Field::new_list_field(DataType::Null, true));
        assert!(!can_cache(&DataType::FixedSizeList(
            Arc::clone(&nulls),
            1 << 30
        )));
        assert!(!can_cache(&DataType::List(nulls)));
        let empty_lists = Arc::new(Field::new_list_field(
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Int32, true)),
                0,
            ),
            true,
        ));
        assert!(!can_cache(&DataType::FixedSizeList(empty_lists, 1 << 30)));
        let empty_binaries =
            Arc::new(Field::new_list_field(DataType::FixedSizeBinary(0), true));
        assert!(!can_cache(&DataType::FixedSizeList(
            empty_binaries,
            1 << 30
        )));
        assert!(can_cache(&DataType::FixedSizeList(
            Arc::new(Field::new_list_field(DataType::Float32, true)),
            4
        )));

        // Chunks of dictionaries are passed through.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            dictionary(DataType::Int8),
            true,
        )]));
        let operator = CachingOperator::new(source(Arc::clone(&schema), vec![]));
        assert!(!operator.caching_supported());
        assert!(!operator.requires_final_execute());
        assert_eq!(operator.description(), "CachingOperator: disabled");
        let inputs = (0..3).map(|b| {
            let values = (0..100).map(|i| format!("{b}-{i}")).collect::<Vec<_>>();
            let array: ArrayRef = Arc::new(
                values
                    .iter()
                    .map(String::as_str)
                    .collect::<DictionaryArray<Int8Type>>(),
            );
            DataChunk::from_record_batch(
                RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap(),
            )
        });
        let (outputs, _) = drive(&operator, inputs).unwrap();
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn cached_rows_do_not_keep_the_buffers_of_their_chunks() {
        // Chunks of 2048 strings of 128 bytes (a 262144-byte buffer), of which
        // one row is selected: a contiguous selection, which `compact` would
        // slice without copying. (Strings and lists with 64-bit offsets: those
        // with 32-bit offsets are not cached.)
        let list = DataType::new_large_list(DataType::LargeUtf8, true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::LargeUtf8, false),
            Field::new("l", list.clone(), true),
        ]));
        let chunk = |i: usize| {
            let strings = (0..2048).map(|j| format!("{:0128}", i * 2048 + j));
            let s = arrow::array::LargeStringArray::from_iter_values(strings.clone());
            let l = arrow::array::ListArray::from_iter_primitive::<Int32Type, _, _>(
                (0..2048).map(|j: i32| Some(vec![Some(j)])),
            );
            // A list of strings instead of integers.
            let l = arrow::compute::cast(&l, &list).unwrap();
            let batch =
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(s), l]).unwrap();
            DataChunk::from_record_batch(batch)
                .with_selection(SelectionVector::from(vec![7]))
                .unwrap()
        };
        let input_bytes = chunk(0).column(0).get_array_memory_size();
        assert!(input_bytes >= 262_144, "{input_bytes}");

        let operator = CachingOperator::new(source(Arc::clone(&schema), vec![]));
        assert!(operator.caching_supported());
        let (gstate, mut lstate) = states(&operator);
        let ctx = context();
        for i in 0..100 {
            let mut output = DataChunk::empty(Arc::clone(&schema));
            operator
                .execute(
                    &ctx,
                    &chunk(i),
                    &mut output,
                    gstate.as_ref(),
                    lstate.as_mut(),
                )
                .unwrap();
            assert!(output.is_empty());
        }
        // Before the final flush, the cache only holds the 100 cached rows.
        let cached_bytes = lstate.downcast_ref::<CachingState>().unwrap().cache.size();
        assert!(
            cached_bytes < input_bytes,
            "{cached_bytes} vs {input_bytes}"
        );

        let mut output = DataChunk::empty(Arc::clone(&schema));
        operator
            .final_execute(&ctx, &mut output, gstate.as_ref(), lstate.as_mut())
            .unwrap();
        assert_eq!(output.len(), 100);
        let strings = output.column(0).as_string::<i64>();
        assert_eq!(strings.value(1), format!("{:0128}", 2048 + 7));
    }

    // ---------------------------------------------------------------------
    // Differential tests
    // ---------------------------------------------------------------------

    /// 100 000 rows in 2 partitions.
    fn large_table() -> Vec<Vec<RecordBatch>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8View, true),
        ]));
        let batch = |ids: std::ops::Range<i64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids.clone())),
                    Arc::new(StringViewArray::from_iter(ids.map(|i| {
                        (i % 7 != 0).then(|| format!("value number {}", i % 5000))
                    }))),
                ],
            )
            .unwrap()
        };
        vec![
            (0..6)
                .map(|b| batch(b * 10_000..(b + 1) * 10_000))
                .collect(),
            (6..10)
                .map(|b| batch(b * 10_000..(b + 1) * 10_000))
                .collect(),
        ]
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn strings_with_32_bit_offsets_are_not_cached() {
        let _large = one_large_test_at_a_time().await;
        // A `FilterExec` with batches of one row passes on the 1024 rows of
        // its input as they are, which share a string of 2 MiB: the cache would
        // concatenate them into 2 GiB of strings, whose 32-bit offsets
        // overflow.
        let batch = RecordBatch::try_from_iter([
            ("x", Arc::new(Int64Array::from(vec![1])) as ArrayRef),
            (
                "s",
                Arc::new(StringArray::from(vec!["x".repeat(2 << 20)])) as ArrayRef,
            ),
        ])
        .unwrap();
        let batches = vec![batch; 1024];
        let schema = batches[0].schema();
        let plan = || -> Arc<dyn datafusion_physical_plan::ExecutionPlan> {
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    physical_expr("x > 0", &schema),
                    scan_batches(&schema, &batches),
                )
                .with_batch_size(1)
                .build()
                .unwrap(),
            )
        };
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let expected = outcome(plan(), Arc::clone(&task_ctx)).await;
        assert_eq!(expected, Ok(1024));
        let cache =
            CachingOperator::new(Arc::new(crate::bridge::BridgeOperator::new(plan())));
        assert!(!cache.caching_supported());
        let root =
            crate::bridge::PipelineRootExec::try_new(Arc::new(cache), plan()).unwrap();
        assert_eq!(outcome(Arc::new(root), task_ctx).await, expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_small_batch_coalescing() {
        let setup = async |ctx: &SessionContext| {
            register_all_tables(ctx)?;
            register_partitions(ctx, "large", large_table())
        };
        for sql in [
            "SELECT * FROM large WHERE id % 100 = 7",
            "SELECT id, v FROM large WHERE id % 100 = 7 AND v IS NOT NULL",
            "SELECT id, upper(v) AS u FROM large WHERE id % 50 = 1",
            "SELECT v, COUNT(*) FROM large WHERE id % 100 < 3 GROUP BY v",
            "SELECT id FROM large WHERE id % 100 = 7 ORDER BY id",
        ] {
            let run = assert_native_byte_eq_with_setup(sql, &setup).await;
            assert_eq!(run.operators.caching, run.operators.filters, "{sql}");
            assert!(
                run.actual
                    .batches
                    .iter()
                    .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE),
                "{sql}"
            );
        }

        // The filter emits about 49 chunks of about 20 rows, the caching
        // operator one or two chunks per thread.
        let run = assert_native_byte_eq_with_setup(
            "SELECT * FROM large WHERE id % 100 = 7",
            &setup,
        )
        .await;
        let metrics = run.root.execution_metrics().unwrap();
        let chunks = |name: &str| {
            metrics
                .iter()
                .find(|(operator, _)| operator.name() == name)
                .map(|(_, metrics)| (metrics.output_chunks(), metrics.output_rows()))
                .unwrap()
        };
        let (filter_chunks, filter_rows) = chunks("PhysicalFilter");
        let (caching_chunks, caching_rows) = chunks("CachingOperator");
        assert_eq!(filter_rows, 1000);
        assert_eq!(caching_rows, 1000);
        assert!(filter_chunks >= 49, "{filter_chunks}");
        assert!(caching_chunks <= 8, "{caching_chunks}");
    }
}
