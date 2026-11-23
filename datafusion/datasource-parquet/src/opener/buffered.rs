// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::{collections::VecDeque, fmt, ops::Range, sync::Arc};

use arrow::array::RecordBatch;
use bytes::Bytes;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
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
use tokio::sync::{mpsc, oneshot};

use crate::ParquetFileMetrics;
use crate::decoder_projection::DecoderProjection;
use crate::metrics::{ByteProgress, RowFilterSkippedFullyMatchedMetric};
use crate::push_decoder::{RgPlanEntry, RowFilterContext};

pub(super) enum BufferedReaders {
    Plain(PlainBufferedReaders),
    Filtered(Box<FilteredBufferedReaders>),
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

type HandbackRequest = (Vec<Range<u64>>, oneshot::Sender<Result<Vec<Bytes>>>);

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
    /// Exclusively owned by this file's I/O lane (never shared behind a lock).
    reader: Box<dyn AsyncFileReader>,
    /// Lock-free object-store fetcher for Phase-2 I/O lanes when using a
    /// built-in reader factory; `None` for custom factories (which serve
    /// Phase-2 requests via `handback_rx` on this file's I/O lane).
    direct_fetcher: Option<Arc<DirectRangeFetcher>>,
    in_flight: Option<InFlight>,
    handback_rx: Option<mpsc::UnboundedReceiver<HandbackRequest>>,
}

impl BufferedReaders {
    pub(super) fn filtered(config: FilteredBufferedReadersConfig) -> Self {
        let predicate_cache_inner_records =
            config.file_metrics.predicate_cache_inner_records.clone();
        let predicate_cache_records = config.file_metrics.predicate_cache_records.clone();
        let direct_fetcher = config.direct_store.map(|store| {
            Arc::new(DirectRangeFetcher {
                store,
                location: config.file_location,
                file_size: config.file_size,
                file_metrics: config.file_metrics,
            })
        });
        Self::Filtered(Box::new(FilteredBufferedReaders {
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
            reader: config.async_file_reader,
            direct_fetcher,
            in_flight: None,
            handback_rx: None,
        }))
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
                    cache_metrics: None,
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
struct PredicateCacheMetrics {
    arrow_reader_metrics: ArrowReaderMetrics,
    predicate_cache_inner_records: Gauge,
    predicate_cache_records: Gauge,
}

impl PredicateCacheMetrics {
    fn copy_to_gauges(&self) {
        if let Some(v) = self.arrow_reader_metrics.records_read_from_inner() {
            self.predicate_cache_inner_records.set(v);
        }
        if let Some(v) = self.arrow_reader_metrics.records_read_from_cache() {
            self.predicate_cache_records.set(v);
        }
    }
}

#[derive(Clone)]
enum Phase2Transport {
    Direct(Arc<DirectRangeFetcher>),
    Handback(mpsc::UnboundedSender<HandbackRequest>),
}

impl Phase2Transport {
    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        match self {
            Self::Direct(fetcher) => fetcher.get_byte_ranges(ranges).await,
            Self::Handback(tx) => {
                let (reply_tx, reply_rx) = oneshot::channel();
                tx.send((ranges, reply_tx)).map_err(|_| {
                    internal_datafusion_err!(
                        "Parquet file I/O lane closed before serving Phase-2 range read"
                    )
                })?;
                reply_rx.await.map_err(|_| {
                    internal_datafusion_err!(
                        "Parquet file I/O lane dropped Phase-2 range response"
                    )
                })?
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
        fetched_bytes: usize,
    },
}

/// Sentinel file continuation emitted alongside a worker-bound
/// [`FilteredRowGroupPlanner`] when the file has no more row groups to open.
#[derive(Debug)]
struct EmptyDonePlanner;

impl MorselPlanner for EmptyDonePlanner {
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        Ok(None)
    }
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

    fn cache_metrics(&self) -> PredicateCacheMetrics {
        PredicateCacheMetrics {
            arrow_reader_metrics: self.arrow_reader_metrics.clone(),
            predicate_cache_inner_records: self.predicate_cache_inner_records.clone(),
            predicate_cache_records: self.predicate_cache_records.clone(),
        }
    }

    fn publish_morsel(self: Box<Self>, morsel: Box<dyn Morsel>) -> MorselPlan {
        let mut plan = MorselPlan::new().with_morsels(vec![morsel]);
        if !self.pending.is_empty() {
            plan = plan.with_planners(vec![Box::new(BufferedReaders::Filtered(self))]);
        }
        plan
    }

    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.baseline.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();

        // For custom reader factories, serve any Phase-2 range reads for the
        // previously published row group on this file lane until the row group
        // finishes or publishes its decode-only reader (dropping its sender).
        if let Some(mut rx) = self.handback_rx.take() {
            return Ok(Some(MorselPlan::new().with_pending_planner(async move {
                while let Some((ranges, reply_tx)) = rx.recv().await {
                    let result = self
                        .reader
                        .get_byte_ranges(ranges)
                        .await
                        .map_err(DataFusionError::from);
                    let _ = reply_tx.send(result);
                }
                Ok(Box::new(BufferedReaders::Filtered(self)) as Box<dyn MorselPlanner>)
            })));
        }

        if let Some(in_flight) = self.in_flight.take() {
            match in_flight {
                InFlight::Filtered {
                    mut decoder,
                    entry,
                    ranges,
                    bytes,
                } => {
                    // Push fetched predicate column bytes without calling
                    // `try_next_reader` so RowFilter decoding and evaluation run
                    // on a pipeline worker, not on this I/O lane.
                    let buffered_bytes = bytes.iter().map(Bytes::len).sum();
                    decoder.push_ranges(ranges, bytes)?;
                    self.byte_progress.credit(entry.bytes);
                    let transport = match &self.direct_fetcher {
                        Some(fetcher) => Phase2Transport::Direct(Arc::clone(fetcher)),
                        None => {
                            let (tx, rx) = mpsc::unbounded_channel();
                            self.handback_rx = Some(rx);
                            Phase2Transport::Handback(tx)
                        }
                    };
                    let rg_planner: Box<dyn MorselPlanner> =
                        Box::new(FilteredRowGroupPlanner {
                            decoder,
                            stage: RgStage::RunFilter,
                            pending_predicates: self.row_filter_context.num_predicates(),
                            buffered_bytes,
                            fetched: None,
                            transport,
                            projection: Arc::clone(&self.projection),
                            file_name: Arc::clone(&self.file_name),
                            elapsed_compute: elapsed_compute.clone(),
                            cache_metrics: self.cache_metrics(),
                        });
                    let file_continuation: Box<dyn MorselPlanner> =
                        if self.handback_rx.is_some() || !self.pending.is_empty() {
                            Box::new(BufferedReaders::Filtered(self))
                        } else {
                            Box::new(EmptyDonePlanner)
                        };
                    // Two ready planners signal `[file_continuation, worker_filter_step]`
                    // to the native Parquet driver without touching stock `FileStream`.
                    return Ok(Some(
                        MorselPlan::new()
                            .with_planners(vec![file_continuation, rg_planner]),
                    ));
                }
                InFlight::FullyMatched {
                    mut decoder,
                    entry,
                    ranges,
                    bytes,
                    mut fetched_bytes,
                } => {
                    fetched_bytes += bytes.iter().map(Bytes::len).sum::<usize>();
                    decoder.push_ranges(ranges, bytes)?;
                    match decoder.try_next_reader()? {
                        DecodeResult::NeedsData(more_ranges) => {
                            return Ok(Some(MorselPlan::new().with_pending_planner(
                                async move {
                                    let bytes = self
                                        .reader
                                        .get_byte_ranges(more_ranges.clone())
                                        .await?;
                                    self.in_flight = Some(InFlight::FullyMatched {
                                        decoder,
                                        entry,
                                        ranges: more_ranges,
                                        bytes,
                                        fetched_bytes,
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
                                compressed_bytes: fetched_bytes as u64,
                                projection: Arc::clone(&self.projection),
                                file_name: Arc::clone(&self.file_name),
                                elapsed_compute: elapsed_compute.clone(),
                                cache_metrics: Some(self.cache_metrics()),
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
                            let bytes =
                                self.reader.get_byte_ranges(ranges.clone()).await?;
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
                                    fetched_bytes: 0,
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
                        cache_metrics: Some(self.cache_metrics()),
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

enum RgStage {
    /// Ready to run `RowFilter` CPU evaluation synchronously on a worker.
    RunFilter,
    /// Returned from a Phase-2 `PendingMorselPlanner` on an I/O lane with newly
    /// fetched bytes in `fetched`.
    AfterIo,
}

/// Per-row-group state machine separating worker `RowFilter` CPU work from
/// I/O-lane Phase-2 range fetches.
///
/// - In [`RgStage::RunFilter`], [`Self::plan`] runs on a pipeline worker,
///   evaluating `RowFilter` predicates on already-buffered bytes. It never
///   performs I/O: if more column ranges are needed, it returns a
///   `PendingMorselPlanner` for the I/O lanes to await.
/// - In [`RgStage::AfterIo`], [`Self::plan`] runs on an I/O lane after the
///   fetched bytes arrive. It either publishes a decode-only [`BufferedReader`]
///   (if all predicates have run) or hands the row group back to workers for
///   the next `RowFilter` conjunct.
struct FilteredRowGroupPlanner {
    decoder: ParquetPushDecoder,
    stage: RgStage,
    pending_predicates: usize,
    buffered_bytes: usize,
    fetched: Option<(Vec<Range<u64>>, Vec<Bytes>)>,
    transport: Phase2Transport,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    elapsed_compute: Time,
    cache_metrics: PredicateCacheMetrics,
}

impl fmt::Debug for FilteredRowGroupPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredRowGroupPlanner")
            .finish_non_exhaustive()
    }
}

impl FilteredRowGroupPlanner {
    fn context_err(&self, err: DataFusionError) -> DataFusionError {
        err.context(format!("Decoding Parquet morsel from {}", self.file_name))
    }

    fn into_decode_morsel(self, reader: ParquetRecordBatchReader) -> Box<dyn Morsel> {
        Box::new(BufferedReader {
            reader,
            compressed_bytes: self.buffered_bytes as u64,
            projection: self.projection,
            file_name: self.file_name,
            elapsed_compute: self.elapsed_compute,
            cache_metrics: Some(self.cache_metrics),
        })
    }

    fn schedule_fetch(mut self: Box<Self>, ranges: Vec<Range<u64>>) -> MorselPlan {
        self.stage = RgStage::AfterIo;
        let transport = self.transport.clone();
        MorselPlan::new().with_pending_planner(async move {
            let bytes = transport.get_byte_ranges(ranges.clone()).await?;
            self.fetched = Some((ranges, bytes));
            Ok(self as Box<dyn MorselPlanner>)
        })
    }
}

impl MorselPlanner for FilteredRowGroupPlanner {
    fn buffered_bytes_estimate(&self) -> usize {
        self.buffered_bytes
    }

    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.elapsed_compute.clone();
        let _timer = elapsed_compute.timer();

        match self.stage {
            RgStage::RunFilter => {
                let result = self
                    .decoder
                    .try_next_reader()
                    .map_err(|err| self.context_err(DataFusionError::from(err)))?;
                self.pending_predicates = self.pending_predicates.saturating_sub(1);
                match result {
                    DecodeResult::Finished => {
                        self.cache_metrics.copy_to_gauges();
                        Ok(None)
                    }
                    DecodeResult::Data(reader) => {
                        let morsel = self.into_decode_morsel(reader);
                        Ok(Some(MorselPlan::new().with_morsels(vec![morsel])))
                    }
                    DecodeResult::NeedsData(ranges) => {
                        Ok(Some(self.schedule_fetch(ranges)))
                    }
                }
            }
            RgStage::AfterIo => {
                if let Some((ranges, bytes)) = self.fetched.take() {
                    self.buffered_bytes += bytes.iter().map(Bytes::len).sum::<usize>();
                    self.decoder
                        .push_ranges(ranges, bytes)
                        .map_err(|err| self.context_err(DataFusionError::from(err)))?;
                }
                if self.pending_predicates > 0 {
                    // More RowFilter conjuncts remain: send back to pipeline
                    // workers without decoding on the I/O lane.
                    self.stage = RgStage::RunFilter;
                    return Ok(Some(
                        MorselPlan::new()
                            .with_planners(vec![Box::new(EmptyDonePlanner), self]),
                    ));
                }
                // All predicates have run: build the decode-only reader from
                // the buffered chunks without decoding output rows here.
                let result = self
                    .decoder
                    .try_next_reader()
                    .map_err(|err| self.context_err(DataFusionError::from(err)))?;
                match result {
                    DecodeResult::Finished => {
                        self.cache_metrics.copy_to_gauges();
                        Ok(None)
                    }
                    DecodeResult::Data(reader) => {
                        let morsel = self.into_decode_morsel(reader);
                        Ok(Some(MorselPlan::new().with_morsels(vec![morsel])))
                    }
                    DecodeResult::NeedsData(more_ranges) => {
                        Ok(Some(self.schedule_fetch(more_ranges)))
                    }
                }
            }
        }
    }
}

struct BufferedReader {
    reader: ParquetRecordBatchReader,
    compressed_bytes: u64,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    elapsed_compute: Time,
    cache_metrics: Option<PredicateCacheMetrics>,
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
            cache_metrics,
            ..
        } = *self;
        stream::iter(std::iter::from_fn(move || {
            let _timer = elapsed_compute.timer();
            let next = reader.next();
            if let Some(metrics) = &cache_metrics {
                metrics.copy_to_gauges();
            }
            next.map(|batch| {
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
