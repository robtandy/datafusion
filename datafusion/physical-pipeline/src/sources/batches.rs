// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`BatchChunks`]: the chunks of the batch that a thread of a source emits,
//! with the memory of the batch reserved in the memory pool.

use arrow::array::RecordBatch;
use datafusion_common::Result;
use datafusion_execution::memory_pool::MemoryReservation;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};

/// The batch that a thread of a source emits, in zero-copy chunks of at most
/// [`DEFAULT_VECTOR_SIZE`] rows.
///
/// The memory of the batch is reserved in the memory pool of the session
/// while the batch is emitted (see [`Self::set`]), so that a source fails
/// with `ResourcesExhausted` when the pool cannot hold the batch it holds,
/// and is freed once its last chunk was emitted.
#[derive(Debug)]
pub(crate) struct BatchChunks {
    batch: Option<RecordBatch>,
    offset: usize,
    reservation: MemoryReservation,
}

impl BatchChunks {
    /// Emit the batches of a thread, reserving their memory in
    /// `reservation`.
    pub(crate) fn new(reservation: MemoryReservation) -> Self {
        Self {
            batch: None,
            offset: 0,
            reservation,
        }
    }

    /// Emit `batch` (after the batch that is emitted, if any, which is
    /// dropped), reserving its memory. Fails with `ResourcesExhausted` if the
    /// memory pool cannot hold it.
    pub(crate) fn set(&mut self, batch: RecordBatch) -> Result<()> {
        self.batch = None;
        self.reservation.try_resize(batch.get_array_memory_size())?;
        self.offset = 0;
        if batch.num_rows() > 0 {
            self.batch = Some(batch);
        } else {
            self.reservation.free();
        }
        Ok(())
    }

    /// The next chunk of the batch, if any rows are left. Frees the memory
    /// of the batch with its last chunk.
    pub(crate) fn next_chunk(&mut self) -> Option<DataChunk> {
        let batch = self.batch.as_ref()?;
        let rows = batch.num_rows();
        let len = (rows - self.offset).min(DEFAULT_VECTOR_SIZE);
        let chunk = if len == rows {
            batch.clone()
        } else {
            batch.slice(self.offset, len)
        };
        self.offset += len;
        if self.offset >= rows {
            self.batch = None;
            self.reservation.free();
        }
        Some(DataChunk::from_record_batch(chunk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use datafusion_common::DataFusionError;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };

    fn batch(rows: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from_iter_values(0..rows))],
        )
        .unwrap()
    }

    #[test]
    fn emits_chunks_of_at_most_vector_size_and_frees_the_batch() -> Result<()> {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut chunks = BatchChunks::new(MemoryConsumer::new("test").register(&pool));
        assert!(chunks.next_chunk().is_none());
        chunks.set(batch(5000))?;
        assert!(pool.reserved() >= 5000 * 8);
        let mut values: Vec<i64> = vec![];
        let mut lengths = vec![];
        while let Some(chunk) = chunks.next_chunk() {
            lengths.push(chunk.len());
            let batch = chunk.into_record_batch()?;
            values.extend(batch.column(0).as_primitive::<Int64Type>().values());
        }
        assert_eq!(lengths, vec![2048, 2048, 904]);
        assert_eq!(values, (0..5000).collect::<Vec<_>>());
        assert_eq!(pool.reserved(), 0);
        // Batches without rows emit nothing and hold no memory.
        chunks.set(batch(0))?;
        assert!(chunks.next_chunk().is_none());
        assert_eq!(pool.reserved(), 0);
        // A new batch replaces the batch that is emitted.
        chunks.set(batch(5000))?;
        assert_eq!(chunks.next_chunk().unwrap().len(), 2048);
        chunks.set(batch(10))?;
        assert_eq!(chunks.next_chunk().unwrap().len(), 10);
        assert!(chunks.next_chunk().is_none());
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn fails_when_the_pool_cannot_hold_the_batch() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1000));
        let mut chunks = BatchChunks::new(MemoryConsumer::new("test").register(&pool));
        let error = chunks.set(batch(5000)).unwrap_err();
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
        assert!(chunks.next_chunk().is_none());
        chunks.set(batch(10)).unwrap();
        assert_eq!(chunks.next_chunk().unwrap().len(), 10);
    }
}
