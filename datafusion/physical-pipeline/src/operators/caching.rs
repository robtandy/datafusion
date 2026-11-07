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
