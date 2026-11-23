// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::{collections::VecDeque, fmt, ops::Range, sync::Arc};

use arrow::array::RecordBatch;
use bytes::Bytes;
use datafusion_common::{DataFusionError, Result};
use datafusion_datasource::morsel::{Morsel, MorselPlan, MorselPlanner};
use datafusion_physical_plan::metrics::{BaselineMetrics, Gauge, Time};
use futures::{StreamExt, stream};
use object_store::{ObjectStore, path::Path};
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReader, RowSelectionPolicy,
};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::{
    ParquetPushDecoder, ParquetPushDecoderBuilder, RowGroupSelection,
};
use parquet::errors::ParquetError;
use tokio::sync::Mutex;

use crate::ParquetFileMetrics;
use crate::decoder_projection::DecoderProjection;
use crate::metrics::{ByteProgress, RowFilterSkippedFullyMatchedMetric};
use crate::push_decoder::{RgPlanEntry, RowFilterContext};

pub(super) enum BufferedReaders {
    Plain(PlainBufferedReaders),
    Filtered(FilteredBufferedReaders),
}

pub(super) struct PlainBufferedReaders {
    pub(super) decoder: ParquetPushDecoder,
    pub(super) reader: Box<dyn AsyncFileReader>,
    pub(super) projection: Arc<DecoderProjection>,
    pub(super) file_name: Arc<str>,
    pub(super) baseline: BaselineMetrics,
    pub(super) rg_plan: VecDeque<RgPlanEntry>,
    pub(super) byte_progress: ByteProgress,
    pub(super) ranges: Option<(Vec<Range<u64>>, Vec<Bytes>)>,
}

pub(super) struct FilteredBufferedReadersConfig {
    pub(super) pending: VecDeque<(RowGroupSelection, RgPlanEntry)>,
    pub(super) reader_metadata: ArrowReaderMetadata,
    pub(super) projection: Arc<DecoderProjection>,
    pub(super) row_filter_context: RowFilterContext,
    pub(super) row_filter_skipped_fully_matched: RowFilterSkippedFullyMatchedMetric,
    pub(super) batch_size: usize,
    pub(super) force_filter_selections: bool,
    pub(super) max_predicate_cache_size: Option<usize>,
    pub(super) arrow_reader_metrics: ArrowReaderMetrics,
    pub(super) file_metrics: ParquetFileMetrics,
    pub(super) file_name: Arc<str>,
    pub(super) file_location: Path,
    pub(super) file_size: u64,
    pub(super) baseline: BaselineMetrics,
    pub(super) byte_progress: ByteProgress,
    pub(super) async_file_reader: Box<dyn AsyncFileReader>,
    pub(super) direct_store: Option<Arc<dyn ObjectStore>>,
}

pub(super) struct FilteredBufferedReaders {
    pending: VecDeque<(RowGroupSelection, RgPlanEntry)>,
    reader_metadata: ArrowReaderMetadata,
    projection: Arc<DecoderProjection>,
    row_filter_context: RowFilterContext,
    row_filter_skipped_fully_matched: RowFilterSkippedFullyMatchedMetric,
    batch_size: usize,
    force_filter_selections: bool,
    max_predicate_cache_size: Option<usize>,
    arrow_reader_metrics: ArrowReaderMetrics,
    predicate_cache_inner_records: Gauge,
    predicate_cache_records: Gauge,
    file_name: Arc<str>,
    baseline: BaselineMetrics,
    byte_progress: ByteProgress,
    producer_reader: ProducerReader,
    worker_fetcher: RangeFetcher,
    in_flight: Option<InFlight>,
}

impl BufferedReaders {
    pub(super) fn filtered(config: FilteredBufferedReadersConfig) -> Self {
        let predicate_cache_inner_records =
            config.file_metrics.predicate_cache_inner_records.clone();
        let predicate_cache_records = config.file_metrics.predicate_cache_records.clone();
        let (producer_reader, worker_fetcher) = match config.direct_store {
            Some(store) => (
                ProducerReader::Owned(config.async_file_reader),
                RangeFetcher::Direct(Arc::new(DirectRangeFetcher {
                    store,
                    location: config.file_location,
                    file_size: config.file_size,
                    file_metrics: config.file_metrics,
                })),
            ),
            None => {
                let shared = Arc::new(Mutex::new(config.async_file_reader));
                (
                    ProducerReader::Shared(Arc::clone(&shared)),
                    RangeFetcher::Shared(shared),
                )
            }
        };
        Self::Filtered(FilteredBufferedReaders {
            pending: config.pending,
            reader_metadata: config.reader_metadata,
            projection: config.projection,
            row_filter_context: config.row_filter_context,
            row_filter_skipped_fully_matched: config.row_filter_skipped_fully_matched,
            batch_size: config.batch_size,
            force_filter_selections: config.force_filter_selections,
            max_predicate_cache_size: config.max_predicate_cache_size,
            arrow_reader_metrics: config.arrow_reader_metrics,
            predicate_cache_inner_records,
            predicate_cache_records,
            file_name: config.file_name,
            baseline: config.baseline,
            byte_progress: config.byte_progress,
            producer_reader,
            worker_fetcher,
            in_flight: None,
        })
    }
}

impl fmt::Debug for BufferedReaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedReaders").finish_non_exhaustive()
    }
}

impl MorselPlanner for BufferedReaders {
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        match *self {
            Self::Plain(plain) => plain.plan(),
            Self::Filtered(filtered) => filtered.plan(),
        }
    }
}

impl PlainBufferedReaders {
    fn plan(mut self) -> Result<Option<MorselPlan>> {
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
                    Ok(Box::new(BufferedReaders::Plain(self)) as Box<dyn MorselPlanner>)
                })
            }
            DecodeResult::Data(reader) => {
                // After extraction the decoder is at a boundary. It may have
                // skipped empty selections before arriving at this reader.
                let remaining = self.decoder.row_groups_remaining();
                let mut compressed_bytes = 0;
                while self.rg_plan.len() > remaining {
                    compressed_bytes = self.rg_plan.pop_front().unwrap().bytes;
                    self.byte_progress.credit(compressed_bytes);
                }
                let reader = BufferedReader {
                    reader,
                    compressed_bytes,
                    projection: Arc::clone(&self.projection),
                    file_name: Arc::clone(&self.file_name),
                    elapsed_compute: elapsed_compute.clone(),
                };
                MorselPlan::new()
                    .with_morsels(vec![Box::new(reader)])
                    .with_planners(vec![Box::new(BufferedReaders::Plain(self))])
            }
            DecodeResult::Finished => return Ok(None),
        };
        Ok(Some(plan))
    }
}

enum ProducerReader {
    Owned(Box<dyn AsyncFileReader>),
    Shared(Arc<Mutex<Box<dyn AsyncFileReader>>>),
}

impl ProducerReader {
    async fn get_byte_ranges(&mut self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        match self {
            Self::Owned(reader) => Ok(reader.get_byte_ranges(ranges).await?),
            Self::Shared(reader) => {
                let mut guard = reader.lock().await;
                Ok(guard.get_byte_ranges(ranges).await?)
            }
        }
    }
}

struct DirectRangeFetcher {
    store: Arc<dyn ObjectStore>,
    location: Path,
    file_size: u64,
    file_metrics: ParquetFileMetrics,
}

impl DirectRangeFetcher {
    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        self.store
            .get_ranges(&self.location, &ranges)
            .await
            .map_err(|err| DataFusionError::from(ParquetError::External(Box::new(err))))
    }
}

impl Drop for DirectRangeFetcher {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.file_metrics.bytes_scanned.value());
        self.file_metrics
            .scan_efficiency_ratio
            .set_total(self.file_size as usize);
    }
}

#[derive(Clone)]
enum RangeFetcher {
    Direct(Arc<DirectRangeFetcher>),
    Shared(Arc<Mutex<Box<dyn AsyncFileReader>>>),
}

impl RangeFetcher {
    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        match self {
            Self::Direct(fetcher) => fetcher.get_byte_ranges(ranges).await,
            Self::Shared(reader) => {
                let mut guard = reader.lock().await;
                Ok(guard.get_byte_ranges(ranges).await?)
            }
        }
    }
}

enum InFlight {
    Filtered {
        decoder: ParquetPushDecoder,
        entry: RgPlanEntry,
        ranges: Vec<Range<u64>>,
        bytes: Vec<Bytes>,
    },
    FullyMatched {
        decoder: ParquetPushDecoder,
        entry: RgPlanEntry,
        ranges: Vec<Range<u64>>,
        bytes: Vec<Bytes>,
    },
}

impl FilteredBufferedReaders {
    fn build_rg_decoder(
        &self,
        selection: RowGroupSelection,
        install_filter: bool,
    ) -> Result<ParquetPushDecoder> {
        let mut builder =
            ParquetPushDecoderBuilder::new_with_metadata(self.reader_metadata.clone())
                .with_projection(self.projection.projection_mask().clone())
                .with_batch_size(self.batch_size)
                .with_metrics(self.arrow_reader_metrics.clone())
                .with_row_group_selections(vec![selection]);
        if self.force_filter_selections {
            builder = builder.with_row_selection_policy(RowSelectionPolicy::Selectors);
        }
        if install_filter {
            builder = builder.with_row_filter(self.row_filter_context.build_row_filter());
            if let Some(cap) = self.max_predicate_cache_size {
                builder = builder.with_max_predicate_cache_size(cap);
            }
        }
        Ok(builder.build()?)
    }

    fn publish_morsel(self, morsel: Box<dyn Morsel>) -> MorselPlan {
        let mut plan = MorselPlan::new().with_morsels(vec![morsel]);
        if !self.pending.is_empty() {
            plan = plan.with_planners(vec![Box::new(BufferedReaders::Filtered(self))]);
        }
        plan
    }

    fn plan(mut self) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.baseline.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();

        if let Some(in_flight) = self.in_flight.take() {
            match in_flight {
                InFlight::Filtered {
                    mut decoder,
                    entry,
                    ranges,
                    bytes,
                } => {
                    // Push the fetched predicate column bytes without calling
                    // `try_next_reader` again so RowFilter decoding and
                    // evaluation run on the pipeline worker, not on the I/O lane.
                    let buffered_bytes = bytes.iter().map(Bytes::len).sum();
                    decoder.push_ranges(ranges, bytes)?;
                    self.byte_progress.credit(entry.bytes);
                    let morsel = Box::new(FilteredBufferedReader {
                        decoder,
                        buffered_bytes,
                        fetcher: self.worker_fetcher.clone(),
                        projection: Arc::clone(&self.projection),
                        file_name: Arc::clone(&self.file_name),
                        elapsed_compute: elapsed_compute.clone(),
                        arrow_reader_metrics: self.arrow_reader_metrics.clone(),
                        predicate_cache_inner_records: self
                            .predicate_cache_inner_records
                            .clone(),
                        predicate_cache_records: self.predicate_cache_records.clone(),
                    });
                    return Ok(Some(self.publish_morsel(morsel)));
                }
                InFlight::FullyMatched {
                    mut decoder,
                    entry,
                    ranges,
                    bytes,
                } => {
                    decoder.push_ranges(ranges, bytes)?;
                    match decoder.try_next_reader()? {
                        DecodeResult::NeedsData(more_ranges) => {
                            return Ok(Some(MorselPlan::new().with_pending_planner(
                                async move {
                                    let bytes = self
                                        .producer_reader
                                        .get_byte_ranges(more_ranges.clone())
                                        .await?;
                                    self.in_flight = Some(InFlight::FullyMatched {
                                        decoder,
                                        entry,
                                        ranges: more_ranges,
                                        bytes,
                                    });
                                    Ok(Box::new(BufferedReaders::Filtered(self))
                                        as Box<dyn MorselPlanner>)
                                },
                            )));
                        }
                        DecodeResult::Data(reader) => {
                            self.byte_progress.credit(entry.bytes);
                            let morsel = Box::new(BufferedReader {
                                reader,
                                compressed_bytes: entry.bytes,
                                projection: Arc::clone(&self.projection),
                                file_name: Arc::clone(&self.file_name),
                                elapsed_compute: elapsed_compute.clone(),
                            });
                            return Ok(Some(self.publish_morsel(morsel)));
                        }
                        DecodeResult::Finished => {
                            self.byte_progress.credit(entry.bytes);
                        }
                    }
                }
            }
        }

        while let Some((selection, entry)) = self.pending.pop_front() {
            let install_filter = !entry.fully_matched;
            if entry.fully_matched {
                self.row_filter_skipped_fully_matched.add_one();
            }
            let mut decoder = self.build_rg_decoder(selection, install_filter)?;
            match decoder.try_next_reader()? {
                DecodeResult::NeedsData(ranges) => {
                    return Ok(Some(MorselPlan::new().with_pending_planner(
                        async move {
                            let bytes = self
                                .producer_reader
                                .get_byte_ranges(ranges.clone())
                                .await?;
                            self.in_flight = Some(if install_filter {
                                InFlight::Filtered {
                                    decoder,
                                    entry,
                                    ranges,
                                    bytes,
                                }
                            } else {
                                InFlight::FullyMatched {
                                    decoder,
                                    entry,
                                    ranges,
                                    bytes,
                                }
                            });
                            Ok(Box::new(BufferedReaders::Filtered(self))
                                as Box<dyn MorselPlanner>)
                        },
                    )));
                }
                DecodeResult::Data(reader) => {
                    self.byte_progress.credit(entry.bytes);
                    let morsel = Box::new(BufferedReader {
                        reader,
                        compressed_bytes: 0,
                        projection: Arc::clone(&self.projection),
                        file_name: Arc::clone(&self.file_name),
                        elapsed_compute: elapsed_compute.clone(),
                    });
                    return Ok(Some(self.publish_morsel(morsel)));
                }
                DecodeResult::Finished => {
                    self.byte_progress.credit(entry.bytes);
                }
            }
        }

        Ok(None)
    }
}

struct BufferedReader {
    reader: ParquetRecordBatchReader,
    compressed_bytes: u64,
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
    fn buffered_bytes_estimate(&self) -> usize {
        self.compressed_bytes as usize
    }

    fn into_stream(self: Box<Self>) -> stream::BoxStream<'static, Result<RecordBatch>> {
        let Self {
            mut reader,
            projection,
            file_name,
            elapsed_compute,
            ..
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

struct FilteredBufferedReader {
    decoder: ParquetPushDecoder,
    buffered_bytes: usize,
    fetcher: RangeFetcher,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    elapsed_compute: Time,
    arrow_reader_metrics: ArrowReaderMetrics,
    predicate_cache_inner_records: Gauge,
    predicate_cache_records: Gauge,
}

impl fmt::Debug for FilteredBufferedReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredBufferedReader")
            .finish_non_exhaustive()
    }
}

struct FilteredStreamState {
    decoder: Option<ParquetPushDecoder>,
    active_reader: Option<ParquetRecordBatchReader>,
    fetcher: RangeFetcher,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    elapsed_compute: Time,
    arrow_reader_metrics: ArrowReaderMetrics,
    predicate_cache_inner_records: Gauge,
    predicate_cache_records: Gauge,
}

impl FilteredStreamState {
    fn copy_arrow_reader_metrics(&self) {
        if let Some(v) = self.arrow_reader_metrics.records_read_from_inner() {
            self.predicate_cache_inner_records.set(v);
        }
        if let Some(v) = self.arrow_reader_metrics.records_read_from_cache() {
            self.predicate_cache_records.set(v);
        }
    }

    fn decode_error(&self, err: DataFusionError) -> DataFusionError {
        err.context(format!("Decoding Parquet morsel from {}", self.file_name))
    }

    async fn next_batch(mut self) -> Option<(Result<RecordBatch>, Self)> {
        if self.decoder.is_none() && self.active_reader.is_none() {
            return None;
        }
        let elapsed_compute = self.elapsed_compute.clone();
        let mut timer = elapsed_compute.timer();
        loop {
            if let Some(reader) = self.active_reader.as_mut() {
                match reader.next() {
                    Some(Ok(batch)) => {
                        self.copy_arrow_reader_metrics();
                        let result = self
                            .projection
                            .map(&batch)
                            .map_err(|err| self.decode_error(err));
                        if result.is_err() {
                            self.active_reader = None;
                            self.decoder = None;
                        }
                        return Some((result, self));
                    }
                    Some(Err(err)) => {
                        self.active_reader = None;
                        self.decoder = None;
                        let err = self.decode_error(DataFusionError::from(err));
                        return Some((Err(err), self));
                    }
                    None => {
                        self.active_reader = None;
                    }
                }
            }

            let Some(decoder) = self.decoder.as_mut() else {
                return None;
            };
            match decoder.try_next_reader() {
                Ok(DecodeResult::NeedsData(ranges)) => {
                    timer.stop();
                    let fetched = self.fetcher.get_byte_ranges(ranges.clone()).await;
                    timer.restart();
                    match fetched {
                        Ok(bytes) => {
                            if let Err(err) = self
                                .decoder
                                .as_mut()
                                .expect("decoder present")
                                .push_ranges(ranges, bytes)
                            {
                                self.decoder = None;
                                let err = self.decode_error(DataFusionError::from(err));
                                return Some((Err(err), self));
                            }
                        }
                        Err(err) => {
                            self.decoder = None;
                            let err = self.decode_error(err);
                            return Some((Err(err), self));
                        }
                    }
                }
                Ok(DecodeResult::Data(reader)) => {
                    self.active_reader = Some(reader);
                }
                Ok(DecodeResult::Finished) => {
                    self.copy_arrow_reader_metrics();
                    self.decoder = None;
                    return None;
                }
                Err(err) => {
                    self.decoder = None;
                    let err = self.decode_error(DataFusionError::from(err));
                    return Some((Err(err), self));
                }
            }
        }
    }
}

impl Morsel for FilteredBufferedReader {
    fn buffered_bytes_estimate(&self) -> usize {
        self.buffered_bytes
    }

    fn into_stream(self: Box<Self>) -> stream::BoxStream<'static, Result<RecordBatch>> {
        let state = FilteredStreamState {
            decoder: Some(self.decoder),
            active_reader: None,
            fetcher: self.fetcher,
            projection: self.projection,
            file_name: self.file_name,
            elapsed_compute: self.elapsed_compute,
            arrow_reader_metrics: self.arrow_reader_metrics,
            predicate_cache_inner_records: self.predicate_cache_inner_records,
            predicate_cache_records: self.predicate_cache_records,
        };
        stream::unfold(state, |state| state.next_batch())
            .fuse()
            .boxed()
    }
}
