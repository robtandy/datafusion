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
//! Since chunks are only emitted once the cache is full, caching delays the
//! output of a pipeline: it should not be used below a `LIMIT` that may stop
//! the pipeline early (DuckDB does not cache in such pipelines either).

use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use arrow_select::coalesce::BatchCoalescer;
use datafusion_common::{Result, plan_err};

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
    /// rows. Buffered rows are copied, so the cache does not keep the
    /// buffers of its input chunks alive (view arrays are compacted too).
    cache: BatchCoalescer,
}

impl OperatorState for CachingState {}

impl CachingState {
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
            state.cache.push_batch(input.compact()?)?;
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

/// Whether columns of `data_type` can be cached.
///
/// Caching concatenates the rows of different chunks. For dictionary arrays,
/// this merges the dictionaries of the chunks, which may need more distinct
/// keys than a key type narrower than 32 bits can represent, even if every
/// chunk is valid (DataFusion only concatenates the batches of a single
/// partition, which a pipeline does not know about). Chunks with such
/// dictionaries, at any nesting level, are therefore not cached.
pub fn can_cache(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(key, value) => {
            matches!(
                key.as_ref(),
                DataType::Int32 | DataType::Int64 | DataType::UInt32 | DataType::UInt64
            ) && can_cache(value)
        }
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => can_cache(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().all(|field| can_cache(field.data_type()))
        }
        DataType::Union(fields, _) => {
            fields.iter().all(|(_, field)| can_cache(field.data_type()))
        }
        DataType::RunEndEncoded(run_ends, values) => {
            can_cache(run_ends.data_type()) && can_cache(values.data_type())
        }
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        Array, ArrayRef, AsArray, DictionaryArray, Int64Array, RecordBatch,
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
    fn narrow_dictionaries_are_not_cached() {
        let dictionary =
            |key: DataType| DataType::Dictionary(Box::new(key), Box::new(DataType::Utf8));
        assert!(can_cache(&dictionary(DataType::Int32)));
        assert!(can_cache(&dictionary(DataType::UInt64)));
        assert!(!can_cache(&dictionary(DataType::Int8)));
        assert!(!can_cache(&dictionary(DataType::UInt16)));
        let nested = DataType::Struct(Fields::from(vec![Field::new(
            "d",
            DataType::new_list(dictionary(DataType::Int8), true),
            true,
        )]));
        assert!(!can_cache(&nested));
        assert!(can_cache(&DataType::new_list(DataType::Utf8View, true)));

        // Chunks of narrow dictionaries are passed through.
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

        // Wider keys are cached.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            dictionary(DataType::Int32),
            true,
        )]));
        let operator = CachingOperator::new(source(Arc::clone(&schema), vec![]));
        let inputs = (0..3).map(|b| {
            let values = (0..100).map(|i| format!("{b}-{i}")).collect::<Vec<_>>();
            let array: ArrayRef = Arc::new(
                values
                    .iter()
                    .map(String::as_str)
                    .collect::<DictionaryArray<Int32Type>>(),
            );
            DataChunk::from_record_batch(
                RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap(),
            )
        });
        let (outputs, _) = drive(&operator, inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 300);
        assert_eq!(outputs[0].column(0).null_count(), 0);
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
