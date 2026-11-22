// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::num::NonZeroUsize;
use std::ops::Range;
use std::time::Duration;

use arrow::array::{Int32Array, RecordBatch};
use bytes::Bytes;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource_parquet::{
    DefaultParquetFileReaderFactory, ParquetFileReaderFactory,
};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_plan::metrics::MetricsSet;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use object_store::{ObjectStore, ObjectStoreExt, memory::InMemory};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::properties::WriterProperties;
use tokio::sync::oneshot;
use tokio::time::timeout;

use super::*;
use crate::morsel::{
    CountSink, MorselExec, MorselMetrics, Pipeline, PipelineRun, Sink, Source,
};

const GROUPS: usize = 8;
const WORKERS: usize = 2;
const SINK_CAPACITY: usize = 1;
const DEADLINE: Duration = Duration::from_secs(10);
const QUIET: Duration = Duration::from_millis(50);

fn metric_total(report: &MetricsSet, name: &str) -> usize {
    report
        .iter()
        .filter(|m| m.value().name() == name)
        .map(|m| m.value().as_usize())
        .sum()
}

type Request = oneshot::Sender<ParquetResult<()>>;

#[derive(Debug)]
struct Gates {
    requests: mpsc::UnboundedSender<Request>,
    opened: AtomicUsize,
    active: AtomicUsize,
}

struct InFlight<'a>(&'a AtomicUsize);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct GatedFactory {
    inner: DefaultParquetFileReaderFactory,
    gates: Arc<Gates>,
}

impl ParquetFileReaderFactory for GatedFactory {
    fn create_reader(
        &self,
        partition: usize,
        file: PartitionedFile,
        hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Box<dyn AsyncFileReader + Send>> {
        self.gates.opened.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(GatedReader {
            reader: self.inner.create_reader(partition, file, hint, metrics)?,
            gates: Arc::clone(&self.gates),
        }))
    }
}

struct GatedReader {
    reader: Box<dyn AsyncFileReader + Send>,
    gates: Arc<Gates>,
}

impl AsyncFileReader for GatedReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        self.reader.get_bytes(range)
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        self.reader.get_metadata(options)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        async move {
            self.gates.active.fetch_add(1, Ordering::SeqCst);
            let _active = InFlight(&self.gates.active);
            let (release, wait) = oneshot::channel();
            self.gates.requests.send(release).unwrap();
            wait.await.unwrap()?;
            self.reader.get_byte_ranges(ranges).await
        }
        .boxed()
    }
}

struct ControlledScan {
    plan: MorselExec,
    context: Arc<TaskContext>,
    requests: mpsc::UnboundedReceiver<Request>,
    gates: Arc<Gates>,
    measurement: Arc<MorselMetrics>,
    group_bytes: usize,
}

impl ControlledScan {
    async fn new(num_files: usize, count: bool) -> Result<Self> {
        let batch = RecordBatch::try_from_iter([(
            "id",
            Arc::new(Int32Array::from_iter_values(0..GROUPS as i32)) as _,
        )])?;
        let mut bytes = Vec::new();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1))
            .build();
        let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        let metadata = writer.close()?;
        let group_bytes = metadata.row_group(0).compressed_size();
        assert!(
            metadata
                .row_groups()
                .iter()
                .all(|group| group.compressed_size() == group_bytes)
        );
        let bytes = Bytes::from(bytes);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let mut files = Vec::new();
        for file in 0..num_files {
            let file =
                PartitionedFile::new(format!("{file}.parquet"), bytes.len() as u64);
            store
                .put(&file.object_meta.location, bytes.clone().into())
                .await?;
            files.push(file);
        }
        let (send, requests) = mpsc::unbounded_channel();
        let gates = Arc::new(Gates {
            requests: send,
            opened: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
        });
        let parquet = ParquetSource::new(batch.schema())
            .with_parquet_file_reader_factory(Arc::new(GatedFactory {
                inner: DefaultParquetFileReaderFactory::new(Arc::clone(&store)),
                gates: Arc::clone(&gates),
            }));
        let metrics = parquet.metrics().clone();
        let context = Arc::new(TaskContext::default());
        let url = ObjectStoreUrl::parse("memory://morsel")?;
        context
            .runtime_env()
            .register_object_store(url.as_ref(), store);
        let config = FileScanConfigBuilder::new(url, Arc::new(parquet))
            .with_file_groups(vec![FileGroup::new(files)])
            .build();
        let files = config.file_groups[0].files().to_vec().into();
        let schema = config.projected_schema()?;
        let (sink, output_schema) = if count {
            let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new(
                    "n",
                    arrow::datatypes::DataType::Int64,
                    false,
                ),
            ]));
            (
                Sink::Count {
                    sink: CountSink::new(Arc::clone(&output_schema)),
                    result_operators: vec![],
                },
                output_schema,
            )
        } else {
            (Sink::Result, Arc::clone(&schema))
        };
        let plan = MorselExec::new(
            Pipeline {
                source: Source::Parquet(ParquetScan {
                    table: "test".into(),
                    schema,
                    config: Some(Box::new(config)),
                    files,
                    metrics,
                }),
                operators: vec![],
                workers: NonZeroUsize::new(WORKERS).unwrap(),
                sink,
                sink_capacity: SINK_CAPACITY,
            },
            output_schema,
        );
        Ok(Self {
            plan,
            context,
            requests,
            gates,
            measurement: Arc::default(),
            group_bytes: group_bytes as usize,
        })
    }

    fn start(&self) -> PipelineRun {
        Arc::clone(&self.plan.pipeline).start(
            Arc::clone(&self.context),
            self.plan.metrics.clone(),
            Some(RunMetrics::start(Arc::clone(&self.measurement))),
        )
    }

    async fn report(&self) -> MetricsSet {
        timeout(DEADLINE, async {
            loop {
                let mut reports = self.measurement.take_runs();
                if let Some(report) = reports.pop() {
                    assert!(reports.is_empty());
                    return report;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("metrics were not finalized")
    }

    async fn request(&mut self) -> Request {
        timeout(DEADLINE, self.requests.recv())
            .await
            .expect("data I/O never started")
            .unwrap()
    }

    fn metric(&self, name: &str) -> usize {
        self.plan
            .metrics
            .clone_inner()
            .sum_by_name(name)
            .map_or(0, |value| value.as_usize())
    }

    async fn wait_stopped(&self) {
        timeout(DEADLINE, async {
            while Arc::strong_count(&self.plan.pipeline) != 1
                || self.gates.active.load(Ordering::SeqCst) != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worker or data I/O task did not stop");
    }
}

#[tokio::test]
async fn shared_workers_claim_one_stock_groups_readers_with_bounded_backpressure()
-> Result<()> {
    let mut scan = ControlledScan::new(1, false).await?;
    let mut run = scan.start();
    let capacity = SINK_CAPACITY + WORKERS + READY_READERS;
    for _ in 0..capacity {
        scan.request().await.send(Ok(())).unwrap();
    }
    // One result is buffered, two workers each hold one reader blocked at the
    // sink, and both ready slots are occupied. The same file supplies all five.
    assert!(timeout(QUIET, scan.requests.recv()).await.is_err());
    assert_eq!(run.rx.len(), SINK_CAPACITY);
    assert_eq!(
        scan.metric("parquet_morsels_claimed"),
        SINK_CAPACITY + WORKERS
    );
    assert_eq!(scan.gates.opened.load(Ordering::SeqCst), 1);
    assert_eq!(scan.gates.active.load(Ordering::SeqCst), 0);

    let ((mut rows, peak_batch_bytes), ()) = timeout(DEADLINE, async {
        tokio::try_join!(
            async {
                let mut rows = Vec::new();
                let mut peak_batch_bytes = 0;
                while let Some(batch) = run.next().await? {
                    let array = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap();
                    rows.extend_from_slice(array.values());
                    peak_batch_bytes =
                        peak_batch_bytes.max(array.values().inner().capacity());
                }
                Ok::<_, datafusion_common::DataFusionError>((rows, peak_batch_bytes))
            },
            async {
                for _ in capacity..GROUPS {
                    scan.request().await.send(Ok(())).unwrap();
                }
                Ok(())
            }
        )
    })
    .await
    .expect("did not drain after resuming the sink")?;
    rows.sort_unstable();
    assert_eq!(rows, (0..GROUPS as i32).collect::<Vec<_>>());
    assert_eq!(scan.metric("parquet_morsels_claimed"), GROUPS);
    drop(run);
    let report = scan.report().await;
    let metric = |name| report.sum_by_name(name).unwrap().as_usize();
    assert_eq!(metric("morsels_claimed"), GROUPS);
    assert_eq!(metric("morsels_completed"), GROUPS);
    for worker in ["0", "1"] {
        let value = |name| {
            report
                .iter()
                .find(|m| {
                    m.value().name() == name
                        && m.labels()
                            .iter()
                            .any(|l| l.name() == "worker" && l.value() == worker)
                })
                .unwrap()
                .value()
                .as_usize()
        };
        let claimed = value("morsels_claimed");
        assert!(claimed > 0);
        assert_eq!(claimed, value("morsels_completed"));
        assert!(value("morsel_decode_work") > 0);
        assert!(value("morsel_process_work") > 0);
    }
    assert!(metric("morsel_output_wait") >= (QUIET / 2).as_nanos() as usize);
    assert_eq!(metric("morsel_peak_ready_depth"), READY_READERS);
    assert_eq!(
        metric("morsel_peak_ready_compressed_bytes_estimate"),
        READY_READERS * scan.group_bytes
    );
    assert_eq!(metric("morsel_peak_output_depth"), SINK_CAPACITY);
    assert_eq!(metric("morsel_peak_output_arrow_bytes"), peak_batch_bytes);
    Ok(())
}

#[tokio::test]
async fn pending_data_io_prevents_finalization_and_is_cancelled_on_drop() -> Result<()> {
    let mut scan = ControlledScan::new(1, true).await?;
    let mut run = scan.start();
    let first = scan.request().await;
    assert_eq!(scan.metric("parquet_morsels_claimed"), 0);
    first.send(Ok(())).unwrap();
    let mut second = scan.request().await;
    assert!(timeout(QUIET, run.next()).await.is_err());
    assert_eq!(scan.metric("parquet_morsels_claimed"), 1);
    assert_eq!(scan.metric("count_finalizations"), 0);
    drop(run);
    timeout(DEADLINE, second.closed())
        .await
        .expect("pending I/O not cancelled");
    scan.wait_stopped().await;
    assert_eq!(scan.metric("count_finalizations"), 0);
    let report = scan.report().await;
    let metric = |name| metric_total(&report, name);
    assert_eq!(metric("morsels_claimed"), 1);
    assert_eq!(metric("morsels_completed"), 1);
    assert!(metric("morsel_pending_io") >= (QUIET / 2).as_nanos() as usize);
    assert!(metric("morsel_ready_wait") >= (QUIET / 2).as_nanos() as usize);
    assert_eq!(metric("morsel_output_wait"), 0);
    assert_eq!(metric("morsel_peak_output_depth"), 0);
    Ok(())
}

#[tokio::test]
async fn bounded_pending_io_releases_peers_on_error_without_finalizing() -> Result<()> {
    let mut scan = ControlledScan::new(FILE_LANES + 2, true).await?;
    let mut run = scan.start();
    let first = scan.request().await;
    let mut second = scan.request().await;
    assert!(timeout(QUIET, scan.requests.recv()).await.is_err());
    assert_eq!(scan.gates.opened.load(Ordering::SeqCst), FILE_LANES);
    assert_eq!(scan.gates.active.load(Ordering::SeqCst), PENDING_IO);
    first
        .send(Err(ParquetError::General("gated data read failed".into())))
        .unwrap();
    let error = timeout(DEADLINE, run.next())
        .await
        .expect("I/O error was not reported")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("gated data read failed") && error.contains(".parquet"),
        "{error}"
    );
    timeout(DEADLINE, second.closed())
        .await
        .expect("peer I/O not cancelled");
    drop(run);
    scan.wait_stopped().await;
    assert_eq!(scan.metric("count_finalizations"), 0);
    let report = scan.report().await;
    let metric = |name| metric_total(&report, name);
    assert!(metric("morsel_pending_io") > 0);
    assert!(metric("morsel_ready_wait") > 0);
    for name in [
        "morsels_claimed",
        "morsels_completed",
        "morsel_decode_work",
        "morsel_process_work",
        "morsel_output_wait",
    ] {
        assert_eq!(metric(name), 0, "{name}");
    }
    Ok(())
}

#[tokio::test]
async fn io_error_aborts_peers_even_with_full_result_and_ready_queues() -> Result<()> {
    let mut scan = ControlledScan::new(2, false).await?;
    let mut run = scan.start();
    let held = scan.request().await;
    // One file is stalled in I/O and the other fills both queues.
    for _ in 0..SINK_CAPACITY + WORKERS + READY_READERS {
        scan.request().await.send(Ok(())).unwrap();
    }
    assert!(timeout(QUIET, scan.requests.recv()).await.is_err());
    assert_eq!(run.rx.len(), SINK_CAPACITY);
    held.send(Err(ParquetError::General(
        "test I/O error with full sink".into(),
    )))
    .unwrap();
    scan.wait_stopped().await;
    assert_eq!(run.rx.len(), SINK_CAPACITY);
    let error = run.next().await.unwrap_err().to_string();
    assert!(error.contains("test I/O error with full sink"), "{error}");
    drop(run);
    let report = scan.report().await;
    let metric = |name| metric_total(&report, name);
    assert_eq!(metric("morsels_claimed"), SINK_CAPACITY + WORKERS);
    assert_eq!(metric("morsels_completed"), SINK_CAPACITY);
    assert!(metric("morsel_output_wait") >= (QUIET / 2).as_nanos() as usize);
    Ok(())
}

#[tokio::test]
async fn stock_morsel_drives_data_io_when_polled() -> Result<()> {
    let mut scan = ControlledScan::new(1, false).await?;
    let Source::Parquet(source) = &scan.plan.pipeline.source else {
        unreachable!()
    };
    let config = source.config.as_ref().unwrap();
    let store = scan
        .context
        .runtime_env()
        .object_store(&config.object_store_url)?;
    let morselizer = config
        .file_source()
        .with_batch_size(1)
        .create_morselizer(store, config, 0)?;
    let mut planner = morselizer.plan_file(source.files[0].clone())?;
    let mut stream = timeout(DEADLINE, async {
        loop {
            let mut plan = planner
                .plan()?
                .expect("stock scan unexpectedly pruned the file");
            if let Some(morsel) = plan.take_morsels().pop() {
                return Ok::<_, datafusion_common::DataFusionError>(morsel.into_stream());
            }
            planner = if let Some(io) = plan.take_pending_planner() {
                io.await?
            } else {
                plan.take_ready_planners().pop().unwrap()
            };
        }
    })
    .await
    .expect("stock planning did not complete")?;
    assert!(scan.requests.try_recv().is_err());
    let mut batch = stream.next();
    assert!(futures::poll!(&mut batch).is_pending());
    scan.request().await.send(Ok(())).unwrap();
    assert_eq!(
        timeout(DEADLINE, batch).await.unwrap().unwrap()?.num_rows(),
        1
    );
    Ok(())
}
