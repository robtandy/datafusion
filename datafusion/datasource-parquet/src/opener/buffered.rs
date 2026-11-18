// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::{collections::VecDeque, fmt, ops::Range, sync::Arc};

use arrow::array::RecordBatch;
use bytes::Bytes;
use datafusion_common::{DataFusionError, Result};
use datafusion_datasource::morsel::{Morsel, MorselPlan, MorselPlanner};
use datafusion_physical_plan::metrics::{BaselineMetrics, Time};
use futures::{StreamExt, stream};
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::ParquetPushDecoder;

use crate::decoder_projection::DecoderProjection;
use crate::metrics::ByteProgress;
use crate::push_decoder::RgPlanEntry;

pub(super) struct BufferedReaders {
    pub(super) decoder: ParquetPushDecoder,
    pub(super) reader: Box<dyn AsyncFileReader>,
    pub(super) projection: Arc<DecoderProjection>,
    pub(super) file_name: Arc<str>,
    pub(super) baseline: BaselineMetrics,
    pub(super) rg_plan: VecDeque<RgPlanEntry>,
    pub(super) byte_progress: ByteProgress,
    pub(super) ranges: Option<(Vec<Range<u64>>, Vec<Bytes>)>,
}

impl fmt::Debug for BufferedReaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedReaders").finish_non_exhaustive()
    }
}

impl MorselPlanner for BufferedReaders {
    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.baseline.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();
        if let Some((ranges, bytes)) = self.ranges.take() {
            self.decoder.push_ranges(ranges, bytes)?;
        }
        let plan = match self.decoder.try_next_reader()? {
            DecodeResult::NeedsData(ranges) => {
                MorselPlan::new().with_pending_planner(async move {
                    let bytes = self.reader.get_byte_ranges(ranges.clone()).await?;
                    self.ranges = Some((ranges, bytes));
                    Ok(self as Box<dyn MorselPlanner>)
                })
            }
            DecodeResult::Data(reader) => {
                // After extraction the decoder is at a boundary. It may have
                // skipped empty selections before arriving at this reader.
                let remaining = self.decoder.row_groups_remaining();
                while self.rg_plan.len() > remaining {
                    self.byte_progress
                        .credit(self.rg_plan.pop_front().unwrap().bytes);
                }
                let reader = BufferedReader {
                    reader,
                    projection: Arc::clone(&self.projection),
                    file_name: Arc::clone(&self.file_name),
                    elapsed_compute: elapsed_compute.clone(),
                };
                MorselPlan::new()
                    .with_morsels(vec![Box::new(reader)])
                    .with_planners(vec![self])
            }
            DecodeResult::Finished => return Ok(None),
        };
        Ok(Some(plan))
    }
}

struct BufferedReader {
    reader: ParquetRecordBatchReader,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    elapsed_compute: Time,
}

impl fmt::Debug for BufferedReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedReader").finish_non_exhaustive()
    }
}

impl Morsel for BufferedReader {
    fn into_stream(self: Box<Self>) -> stream::BoxStream<'static, Result<RecordBatch>> {
        let Self {
            mut reader,
            projection,
            file_name,
            elapsed_compute,
        } = *self;
        stream::iter(std::iter::from_fn(move || {
            let _timer = elapsed_compute.timer();
            reader.next().map(|batch| {
                batch
                    .map_err(DataFusionError::from)
                    .and_then(|batch| projection.map(&batch))
                    .map_err(|err| {
                        err.context(format!("Decoding Parquet morsel from {file_name}"))
                    })
            })
        }))
        .boxed()
    }
}
