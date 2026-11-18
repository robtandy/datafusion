// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::datatypes::SchemaRef;
use datafusion_catalog::{ScanArgs, TableProvider};
use datafusion_common::{Result, internal_datafusion_err, internal_err};
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::morsel::{Morsel, Morselizer};
use datafusion_datasource::source::{DataSource, DataSourceExec};
use datafusion_datasource_parquet::{file_format::ParquetFormat, source::ParquetSource};
use datafusion_execution::TaskContext;
use datafusion_expr::TableScan;
use datafusion_expr::expr_rewriter::unnormalize_cols;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_physical_expr::create_physical_expr;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
use datafusion_session::Session;
use tokio::sync::{Mutex, Semaphore, mpsc};

use crate::datasource::listing::ListingTable;

use super::{CountState, supported_expr};

#[cfg(test)]
mod tests;

const FILE_LANES: usize = 4;
const PENDING_IO: usize = 2;
const READY_READERS: usize = 2;

#[derive(Debug)]
pub(super) struct ParquetScan {
    pub(super) table: String,
    pub(super) schema: SchemaRef,
    // An admitted listing scan may have no files, without losing its projected schema.
    config: Option<Box<FileScanConfig>>,
    files: Arc<[PartitionedFile]>,
    pub(super) metrics: ExecutionPlanMetricsSet,
}

impl ParquetScan {
    pub(super) async fn lower(
        table: &ListingTable,
        scan: &TableScan,
        session: &dyn Session,
    ) -> Result<std::result::Result<Self, String>> {
        let unsupported = |feature| {
            Ok(Err(format!(
                "native Parquet morsels do not support {feature} on {}",
                scan.table_name
            )))
        };
        let options = table.options();
        let Some(format) = options.format.downcast_ref::<ParquetFormat>() else {
            return unsupported("non-Parquet listing scans");
        };
        if !options.table_partition_cols.is_empty() {
            return unsupported("partition columns");
        }
        if !options.file_sort_order.is_empty() || options.output_partitioning.is_some() {
            return unsupported("declared file ordering or distribution");
        }
        let crypto = &format.options().crypto;
        if crypto.file_decryption.is_some() || crypto.factory_id.is_some() {
            return unsupported("Parquet encryption");
        }
        let filters: Vec<_> = unnormalize_cols(scan.filters.iter().cloned())
            .into_iter()
            .collect();
        for filter in &filters {
            if !supported_expr(filter)? {
                return unsupported("a scan-context or asynchronous filter");
            }
        }
        let leaf = table
            .scan_with_args(
                session,
                ScanArgs::default()
                    .with_projection(scan.projection.as_deref())
                    .with_filters(Some(&filters)),
            )
            .await?;
        let plan = leaf.plan();
        let mut source = Self {
            table: scan.table_name.to_string(),
            schema: plan.schema(),
            config: None,
            files: Arc::default(),
            metrics: ExecutionPlanMetricsSet::new(),
        };
        if source.schema.as_ref() != scan.projected_schema.as_arrow() {
            return unsupported("a scan schema different from the logical projection");
        }
        if plan.is::<EmptyExec>() {
            return Ok(Ok(source));
        }
        let Some(config) = plan
            .downcast_ref::<DataSourceExec>()
            .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
        else {
            return unsupported("a wrapped Parquet scan");
        };
        let Some(parquet) = config.file_source().downcast_ref::<ParquetSource>() else {
            return unsupported("a non-Parquet file source");
        };
        if config.limit.is_some()
            || config.preserves_file_order()
            || config.output_partitioning.is_some()
        {
            return unsupported("scan limits, ordering or distribution");
        }
        if config.file_compression_type != FileCompressionType::UNCOMPRESSED
            || config
                .file_groups
                .iter()
                .flat_map(|group| group.iter())
                .any(|file| {
                    file.range.is_some()
                        || !file.extensions.is_empty()
                        || !file.partition_values.is_empty()
                })
        {
            return unsupported("a split, compressed or extended file layout");
        }
        if !config.table_partition_cols().is_empty()
            || !parquet.table_schema().virtual_columns().is_empty()
        {
            return unsupported("partition or virtual columns");
        }
        if let Some(feature) = parquet.unsupported_buffered_morsel_source() {
            return unsupported(feature);
        }

        let mut config = config.clone();
        config.file_source = Arc::new(parquet.clone().with_pushdown_filters(false));
        // The listing provider's file predicates are inexact. Attach pruning
        // hints in projected coordinates, without removing any logical Filter.
        let physical_hints = scan
            .filters
            .iter()
            .filter(|expr| {
                !expr.is_volatile()
                    && expr
                        .column_refs()
                        .iter()
                        .all(|col| scan.projected_schema.has_column(col))
            })
            .map(|expr| {
                create_physical_expr(
                    expr,
                    &scan.projected_schema,
                    session.execution_props(),
                    &PhysicalPlanningContext::default(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if !physical_hints.is_empty() {
            let mut prune_only = session.config_options().clone();
            prune_only.execution.parquet.pushdown_filters = false;
            let update = config.try_pushdown_filters(physical_hints, &prune_only)?;
            if update
                .filters
                .iter()
                .any(|pushed| matches!(pushed, PushedDown::Yes))
            {
                return internal_err!(
                    "Native Parquet scan received exact filter pushdown"
                );
            }
            if let Some(updated) = update.updated_node {
                config = updated
                    .downcast_ref::<FileScanConfig>()
                    .ok_or_else(|| {
                        internal_datafusion_err!(
                            "Expected FileScanConfig from Parquet filter pushdown"
                        )
                    })?
                    .clone();
            }
        }
        source.metrics = config.file_source().metrics().clone();
        source.files = std::mem::take(&mut config.file_groups)
            .into_iter()
            .flat_map(|group| group.into_inner())
            .collect();
        source.config = Some(Box::new(config));
        Ok(Ok(source))
    }

    pub(super) fn start(
        &self,
        context: &TaskContext,
        tasks: &mut JoinSet<Result<Option<CountState>>>,
    ) -> Result<ReadyReaders> {
        let (publish, ready) = mpsc::channel(READY_READERS);
        if let Some(config) = &self.config {
            let parquet = config
                .file_source()
                .downcast_ref::<ParquetSource>()
                .ok_or_else(|| {
                    internal_datafusion_err!("Expected the admitted ParquetSource")
                })?;
            let work = Arc::new(FileWork {
                morselizer: parquet.create_buffered_morselizer(config, context)?,
                files: Arc::clone(&self.files),
                next: AtomicUsize::new(0),
                io_slots: Semaphore::new(PENDING_IO),
            });
            for _ in 0..FILE_LANES.min(self.files.len()) {
                let work = Arc::clone(&work);
                let publish = publish.clone();
                tasks.spawn(async move {
                    work.produce(publish).await?;
                    Ok(None)
                });
            }
        }
        Ok(ReadyReaders {
            queue: Arc::new(Mutex::new(ready)),
            claimed: MetricBuilder::new(&self.metrics)
                .counter("parquet_morsels_claimed", 0),
        })
    }
}

#[derive(Clone)]
pub(super) struct ReadyReaders {
    queue: Arc<Mutex<mpsc::Receiver<Box<dyn Morsel>>>>,
    claimed: Count,
}

impl ReadyReaders {
    pub(super) async fn claim(&self) -> Option<Box<dyn Morsel>> {
        let reader = self.queue.lock().await.recv().await?;
        self.claimed.add(1);
        Some(reader)
    }
}

struct FileWork {
    morselizer: Box<dyn Morselizer>,
    files: Arc<[PartitionedFile]>,
    next: AtomicUsize,
    io_slots: Semaphore,
}

impl FileWork {
    async fn produce(&self, publish: mpsc::Sender<Box<dyn Morsel>>) -> Result<()> {
        while !publish.is_closed()
            && let Some(file) = self.files.get(self.next.fetch_add(1, Ordering::Relaxed))
        {
            self.produce_file(file, &publish).await.map_err(|error| {
                error.context(format!(
                    "Planning Parquet morsels for {}",
                    file.object_meta.location
                ))
            })?;
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    async fn produce_file(
        &self,
        file: &PartitionedFile,
        publish: &mpsc::Sender<Box<dyn Morsel>>,
    ) -> Result<()> {
        let mut planner = self.morselizer.plan_file(file.clone())?;
        loop {
            // Reserve before CPU preparation: no reader is parked outside the queue.
            let Ok(permit) = publish.reserve().await else {
                return Ok(());
            };
            let Some(mut step) = planner.plan()? else {
                return Ok(());
            };
            let mut morsels = step.take_morsels();
            let mut ready = step.take_ready_planners();
            let pending = step.take_pending_planner();
            if morsels.len() > 1 || ready.len() + usize::from(pending.is_some()) > 1 {
                return internal_err!(
                    "Expected at most one Parquet reader and one file continuation per step"
                );
            }
            if let Some(morsel) = morsels.pop() {
                permit.send(morsel);
            } else {
                drop(permit);
            }
            if let Some(io) = pending {
                // The I/O task only polls the pending phase. Its abort-on-drop
                // handle is owned by this lane, and there is one phase per lane.
                let _slot = self
                    .io_slots
                    .acquire()
                    .await
                    .expect("semaphore is never closed");
                planner = SpawnedTask::spawn(io).join_unwind().await.map_err(|e| {
                    datafusion_common::exec_datafusion_err!(
                        "Parquet morsel I/O task failed: {e}"
                    )
                })??;
            } else if let Some(next) = ready.pop() {
                planner = next;
            } else {
                return Ok(());
            }
            tokio::task::yield_now().await;
        }
    }
}
