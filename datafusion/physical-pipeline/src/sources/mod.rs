// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native scan sources that hand out chunks of at most
//! [`DEFAULT_VECTOR_SIZE`] rows to all the threads of a pipeline.
//!
//! * [`MemoryScanSource`] scans in-memory batches (`MemTable`, `VALUES`):
//!   all batches of all partitions are cut into slices of at most
//!   [`DEFAULT_VECTOR_SIZE`] rows, which the threads claim through a shared
//!   atomic cursor. One huge partition is therefore scanned by all threads.
//! * [`ParquetScanSource`] and [`CsvScanSource`] scan files. The files of
//!   all file groups (and byte ranges of large files) form one queue shared
//!   by all threads; every thread drives its own
//!   [`Morselizer`], created by
//!   the scan's `FileSource` exactly as DataFusion's `FileStream` does, so
//!   all pushdowns of the scan are kept (projections, predicates with
//!   statistics, page index and bloom filter pruning, row filters, dynamic
//!   filters, schema adaptation, partition columns and limits).
//! * [`EmptyScanSource`] replaces `EmptyExec`: it produces no rows.
//!
//! [`try_create_scan_source`] converts the scans of a DataFusion plan into
//! these sources.
//!
//! # Output
//!
//! A source produces exactly the rows that the union of all partitions of
//! the scan it replaces produces, and exactly DataFusion's batch schemas.
//! With a `LIMIT` (`fetch`) pushed into the scan:
//!
//! * a memory scan applies it to every partition, as DataFusion does, so it
//!   produces the same rows as DataFusion;
//! * a file scan that shares its files between the threads applies it to
//!   all threads together: it produces at most `fetch` rows in total. As in
//!   DataFusion with work stealing, which rows these are is not
//!   deterministic, unless a single file is scanned.
//!
//! The order of the rows is preserved only where DataFusion's plan promises
//! it to the plan above it, i.e. for scans with a single partition and an
//! output ordering (then a single thread reads the scan in order, like a
//! [`BridgeOperator`](crate::bridge::BridgeOperator) over a single
//! partition does). File scans that must preserve the order of their files
//! (`FileScanConfig::preserves_file_order`) read every file group, in order,
//! on a single thread, and apply a limit to every file group, as DataFusion
//! does.

use std::collections::VecDeque;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::morsel::{
    Morsel, MorselPlanner, Morselizer, PendingMorselPlanner,
};
use datafusion_datasource::source::{DataSource, DataSourceExec};
use datafusion_datasource_csv::source::CsvSource;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::{DisplayFormatType, ExecutionPlan};
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use object_store::ObjectStore;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state, downcast_state_mut,
};

pub mod csv;
pub mod empty;
pub mod memory;
pub mod parquet;

pub use csv::CsvScanSource;
pub use empty::EmptyScanSource;
pub use memory::MemoryScanSource;
pub use parquet::ParquetScanSource;

/// Convert a DataFusion scan into a native pipeline source.
///
/// Converts
///
/// * a `DataSourceExec` that scans a [`MemorySourceConfig`] into a
///   [`MemoryScanSource`],
/// * a `DataSourceExec` that scans Parquet files ([`ParquetSource`]) into a
///   [`ParquetScanSource`],
/// * a `DataSourceExec` that scans CSV files ([`CsvSource`]) into a
///   [`CsvScanSource`],
/// * an [`EmptyExec`] into an [`EmptyScanSource`].
///
/// The source has exactly the schema of `plan`. Returns `Ok(None)` for every
/// other plan, which the caller then runs with a
/// [`BridgeOperator`](crate::bridge::BridgeOperator). See the
/// [module documentation](self) for the rows the sources produce and in which
/// order.
pub fn try_create_scan_source(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn PhysicalOperator>>> {
    if let Some(empty) = plan.downcast_ref::<EmptyExec>() {
        return Ok(Some(Arc::new(EmptyScanSource::from_exec(empty))));
    }
    let Some(exec) = plan.downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let source: Arc<dyn PhysicalOperator> =
        if exec.data_source().is::<MemorySourceConfig>() {
            Arc::new(MemoryScanSource::try_from_exec(exec)?)
        } else if exec.downcast_to_file_source::<ParquetSource>().is_some() {
            Arc::new(ParquetScanSource::try_new(exec)?)
        } else if exec.downcast_to_file_source::<CsvSource>().is_some() {
            Arc::new(CsvScanSource::try_new(exec)?)
        } else {
            return Ok(None);
        };
    Ok(Some(source))
}

/// Alias of [`try_create_scan_source`].
pub use self::try_create_scan_source as try_from_execution_plan;

/// Displays a [`DataSource`] like `DataSourceExec` does, without the
/// `DataSourceExec: ` prefix.
struct DisplayDataSource<'a>(&'a dyn DataSource);

impl fmt::Display for DisplayDataSource<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_as(DisplayFormatType::Default, f)
    }
}

/// The smallest default size of the byte ranges that [`FileScan`] splits
/// large files into, see [`FileScan::range_bytes`]. Every range is planned
/// separately (a Parquet range loads the metadata of the file and reads the
/// row groups whose midpoint it contains), so ranges much smaller than a
/// typical row group would mostly be planned for nothing.
const DEFAULT_MIN_RANGE_BYTES: u64 = 16 * 1024 * 1024;

/// The maximum number of planning steps (planning a file, a CPU planning step
/// of a morsel planner, claiming a unit) one [`FileScan::get_data`] call makes
/// before it returns an empty chunk, so that the executor can yield when many
/// files are pruned without producing any rows.
const PLANNING_STEPS_PER_CALL: usize = 32;

/// How a [`FileScan`] distributes its files over the threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileScanMode {
    /// All files (and byte ranges of large files) are in a single queue,
    /// shared by all threads: a thread that needs work claims the next one.
    /// A limit applies to all threads together.
    Shared,
    /// Every file group is read, file after file, by a single thread, as the
    /// scan must preserve the order of its files. A limit applies to every
    /// file group, as in DataFusion.
    Ordered,
}

/// The morsel-driven scan of a [`FileScanConfig`], shared by
/// [`ParquetScanSource`] and [`CsvScanSource`].
///
/// The work of the scan is a list of *units*: a single file or byte range
/// (shared mode) or a file group (ordered mode), see [`FileScanMode`]. The
/// threads claim the units through an atomic cursor. Every thread creates
/// its own [`Morselizer`] with the file source of the scan (as DataFusion
/// does for every partition) and drives it like DataFusion's `FileStream`:
///
/// ```text
/// claimed unit -> files -> Morselizer::plan_file -> planners -> plan()
///   -> morsels -> into_stream() -> reader -> batches -> chunks
///                     |
///                     v
///              pending planner (I/O, polled with the task's waker)
/// ```
///
/// A thread has at most one planner I/O outstanding, as in `FileStream`.
/// Batches are sliced into chunks of at most [`DEFAULT_VECTOR_SIZE`] rows
/// without copying.
#[derive(Debug)]
struct FileScan {
    /// Name of the operator, e.g. `ParquetScanSource`.
    name: &'static str,
    config: FileScanConfig,
    schema: SchemaRef,
    mode: FileScanMode,
    /// The files of the scan, in the order they are claimed: in shared mode
    /// all files, reordered by the file source like DataFusion's shared work
    /// queue (e.g. by statistics for TopK queries); in ordered mode the files
    /// of the non-empty file groups, group after group.
    files: Vec<PartitionedFile>,
    /// In ordered mode, the files of every non-empty file group.
    groups: Vec<Range<usize>>,
    /// Size of the byte ranges large files are split into, `0` to never
    /// split; `None` to derive it from the session configuration.
    range_bytes: Option<u64>,
}

impl FileScan {
    /// Create the scan of `config`, whose output schema is `schema`.
    fn new(name: &'static str, config: FileScanConfig, schema: SchemaRef) -> Self {
        let mode = if config.preserves_file_order() {
            FileScanMode::Ordered
        } else {
            FileScanMode::Shared
        };
        let (files, groups) = match mode {
            FileScanMode::Shared => {
                let files = config
                    .file_groups
                    .iter()
                    .flat_map(FileGroup::iter)
                    .cloned()
                    .collect();
                (config.file_source.reorder_files(files), vec![])
            }
            FileScanMode::Ordered => {
                let mut files = vec![];
                let mut groups = vec![];
                for group in config.file_groups.iter().filter(|g| !g.is_empty()) {
                    let start = files.len();
                    files.extend(group.iter().cloned());
                    groups.push(start..files.len());
                }
                (files, groups)
            }
        };
        Self {
            name,
            config,
            schema,
            mode,
            files,
            groups,
            range_bytes: None,
        }
    }

    fn with_range_bytes(mut self, range_bytes: u64) -> Self {
        self.range_bytes = Some(range_bytes);
        self
    }

    fn description(&self) -> String {
        format!("{}: {}", self.name, DisplayDataSource(&self.config))
    }

    /// The size of the byte ranges that large files are split into in shared
    /// mode, or `None` if files are not split.
    ///
    /// Files are not split if the scan has a limit (so that a scan of a
    /// single file with a limit produces the first rows of the file, like
    /// DataFusion), if the file source cannot read byte ranges or the files
    /// are compressed, or if `datafusion.optimizer.repartition_file_scans` is
    /// disabled. By default the ranges are at least
    /// `datafusion.optimizer.repartition_file_min_size` and 16 MiB large.
    fn range_bytes(&self, ctx: &ExecutionContext) -> Option<u64> {
        let options = ctx.task_ctx.session_config().options();
        let splittable = self.mode == FileScanMode::Shared
            && self.config.limit.is_none()
            && self.config.file_source.supports_repartitioning()
            && !self.config.file_compression_type.is_compressed();
        if !splittable {
            return None;
        }
        let range_bytes = match self.range_bytes {
            Some(range_bytes) => range_bytes,
            None if options.optimizer.repartition_file_scans => {
                (options.optimizer.repartition_file_min_size as u64)
                    .max(DEFAULT_MIN_RANGE_BYTES)
            }
            None => 0,
        };
        (range_bytes > 0).then_some(range_bytes)
    }

    /// [`PhysicalOperator::get_global_source_state`].
    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let task_ctx = &ctx.task_ctx;
        let object_store = task_ctx
            .runtime_env()
            .object_store(&self.config.object_store_url)?;
        let batch_size = self
            .config
            .batch_size
            .unwrap_or_else(|| task_ctx.session_config().batch_size());
        let file_source = self.config.file_source.with_batch_size(batch_size);
        let (files, units) = match self.mode {
            FileScanMode::Shared => {
                let range_bytes = self.range_bytes(ctx);
                let mut files = Vec::with_capacity(self.files.len());
                for file in &self.files {
                    push_file_ranges(&mut files, file, range_bytes);
                }
                let units = (0..files.len()).map(|i| i..i + 1).collect();
                (files, units)
            }
            FileScanMode::Ordered => (self.files.clone(), self.groups.clone()),
        };
        let (remaining, unit_limit) = match self.mode {
            FileScanMode::Shared => (self.config.limit.map(AtomicUsize::new), None),
            FileScanMode::Ordered => (None, self.config.limit),
        };
        Ok(Arc::new(FileScanGlobalState {
            files,
            units,
            next_unit: AtomicUsize::new(0),
            remaining,
            unit_limit,
            object_store,
            file_source,
        }))
    }

    /// [`PhysicalOperator::get_local_source_state`].
    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        let gstate = downcast_state::<FileScanGlobalState>(gstate.as_any(), self.name)?;
        // Like DataFusion, which creates a morselizer for every partition.
        let morselizer = gstate.file_source.create_morselizer(
            Arc::clone(&gstate.object_store),
            &self.config,
            ctx.thread_id,
        )?;
        Ok(Box::new(FileScanLocalState {
            morselizer,
            files: VecDeque::new(),
            unit_remaining: None,
            ready_planners: VecDeque::new(),
            ready_morsels: VecDeque::new(),
            pending_planner: None,
            reader: None,
            batch: None,
            offset: 0,
        }))
    }

    /// [`PhysicalOperator::get_data`].
    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let gstate = downcast_state::<FileScanGlobalState>(gstate.as_any(), self.name)?;
        let lstate =
            downcast_state_mut::<FileScanLocalState>(lstate.as_any_mut(), self.name)?;
        let mut cx = Context::from_waker(ctx.waker());
        let mut planning_steps = 0;
        loop {
            // The next chunk of the current batch.
            if let Some(chunk) = lstate.next_chunk() {
                return Ok(SourceResult::Chunk(chunk));
            }
            // All threads together produced the limit of the scan.
            if gstate.limit_reached() {
                lstate.end_unit();
                return Ok(SourceResult::Finished);
            }
            if planning_steps == PLANNING_STEPS_PER_CALL {
                // Let the executor yield if it has to.
                return Ok(SourceResult::Chunk(DataChunk::empty(Arc::clone(
                    &self.schema,
                ))));
            }

            // Drive the outstanding planner I/O, if any.
            if let Some(pending) = lstate.pending_planner.as_mut()
                && let Poll::Ready(planner) = pending.poll_unpin(&mut cx)
            {
                lstate.pending_planner = None;
                lstate.ready_planners.push_back(planner?);
            }

            // Read the active morsel.
            if let Some(reader) = lstate.reader.as_mut() {
                match reader.poll_next_unpin(&mut cx) {
                    Poll::Ready(Some(batch)) => {
                        let batch = batch?;
                        if batch.num_rows() == 0 {
                            // Return to the executor (which skips empty
                            // chunks), so that the call counts against its
                            // budget.
                            return Ok(SourceResult::Chunk(
                                DataChunk::from_record_batch(batch),
                            ));
                        }
                        lstate.set_batch(batch, gstate);
                        continue;
                    }
                    Poll::Ready(None) => {
                        lstate.reader = None;
                        continue;
                    }
                    Poll::Pending => {
                        return Ok(SourceResult::Blocked(
                            TaskInterrupt::waker_registered(),
                        ));
                    }
                }
            }
            if let Some(morsel) = lstate.ready_morsels.pop_front() {
                lstate.reader = Some(morsel.into_stream());
                continue;
            }
            // Like `FileStream`, a thread does not plan more while planner
            // I/O is outstanding, as planning may need more I/O.
            if lstate.pending_planner.is_some() {
                return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
            }

            planning_steps += 1;
            if let Some(planner) = lstate.ready_planners.pop_front() {
                if let Some(mut plan) = planner.plan()? {
                    lstate.ready_morsels.extend(plan.take_morsels());
                    lstate.ready_planners.extend(plan.take_ready_planners());
                    lstate.pending_planner = plan.take_pending_planner();
                }
                continue;
            }
            if let Some(file) = lstate.files.pop_front() {
                let planner = lstate.morselizer.plan_file(file)?;
                lstate.ready_planners.push_back(planner);
                continue;
            }
            if !lstate.claim_unit(gstate) {
                return Ok(SourceResult::Finished);
            }
        }
    }
}

/// Push `file` to `files`, split into byte ranges of at least `range_bytes`
/// bytes if it is at least twice as large. The ranges cover the file (or
/// the range of the file) exactly; the file source reads every row once
/// (Parquet assigns every row group to the range of its midpoint, CSV every
/// line to the range of its first byte), as for the ranges DataFusion
/// creates when it repartitions file scans. Files with extensions (e.g. a
/// Parquet access plan) are never split.
fn push_file_ranges(
    files: &mut Vec<PartitionedFile>,
    file: &PartitionedFile,
    range_bytes: Option<u64>,
) {
    let (start, end) = file.range();
    let size = end.saturating_sub(start);
    let ranges = match range_bytes {
        Some(range_bytes) if range_bytes > 0 && file.extensions.is_empty() => {
            size / range_bytes
        }
        _ => 0,
    };
    if ranges < 2 {
        files.push(file.clone());
        return;
    }
    let step = size.div_ceil(ranges);
    let mut range_start = start;
    while range_start < end {
        let range_end = (range_start + step).min(end);
        files.push(
            file.clone()
                .with_range(range_start as i64, range_end as i64),
        );
        range_start = range_end;
    }
}

/// Global source state of a [`FileScan`]: the units of work and the cursor
/// that hands them out.
struct FileScanGlobalState {
    /// The files to scan (with the ranges large files are split into).
    files: Vec<PartitionedFile>,
    /// The units of work: ranges of `files`.
    units: Vec<Range<usize>>,
    /// The next unit to claim.
    next_unit: AtomicUsize,
    /// In shared mode with a limit: the rows that all threads together may
    /// still produce.
    remaining: Option<AtomicUsize>,
    /// In ordered mode: the rows every unit (file group) may produce.
    unit_limit: Option<usize>,
    object_store: Arc<dyn ObjectStore>,
    /// The file source of the scan, with the batch size of the execution.
    file_source: Arc<dyn FileSource>,
}

impl FileScanGlobalState {
    /// Whether the threads together produced the limit of the scan.
    fn limit_reached(&self) -> bool {
        self.remaining
            .as_ref()
            .is_some_and(|remaining| remaining.load(Ordering::Acquire) == 0)
    }

    /// Reserve up to `rows` of the rows that may still be produced. Returns
    /// the number of reserved rows.
    fn reserve(&self, rows: usize) -> usize {
        let Some(remaining) = &self.remaining else {
            return rows;
        };
        match remaining.try_update(Ordering::AcqRel, Ordering::Acquire, |left| {
            (left > 0).then(|| left - left.min(rows))
        }) {
            Ok(left) => left.min(rows),
            Err(_) => 0,
        }
    }
}

impl GlobalSourceState for FileScanGlobalState {
    fn max_threads(&self) -> usize {
        self.units.len().max(1)
    }
}

/// Local source state of a [`FileScan`]: the unit a thread claimed and the
/// state of its morselizer, as in DataFusion's `FileStream`.
struct FileScanLocalState {
    morselizer: Box<dyn Morselizer>,
    /// The files of the claimed unit that are not planned yet.
    files: VecDeque<PartitionedFile>,
    /// In ordered mode with a limit: the rows the claimed unit may still
    /// produce.
    unit_remaining: Option<usize>,
    ready_planners: VecDeque<Box<dyn MorselPlanner>>,
    ready_morsels: VecDeque<Box<dyn Morsel>>,
    /// The planner waiting for I/O.
    pending_planner: Option<PendingMorselPlanner>,
    /// The stream of the morsel that is read.
    reader: Option<BoxStream<'static, Result<RecordBatch>>>,
    /// The batch that is emitted in chunks, and the offset of the next chunk.
    batch: Option<RecordBatch>,
    offset: usize,
}

impl LocalSourceState for FileScanLocalState {}

impl FileScanLocalState {
    /// The next chunk of at most [`DEFAULT_VECTOR_SIZE`] rows of the current
    /// batch (zero-copy).
    fn next_chunk(&mut self) -> Option<DataChunk> {
        let batch = self.batch.as_ref()?;
        let rows = batch.num_rows();
        let len = rows.saturating_sub(self.offset).min(DEFAULT_VECTOR_SIZE);
        let chunk = if len == 0 {
            None
        } else if len == rows {
            Some(batch.clone())
        } else {
            Some(batch.slice(self.offset, len))
        };
        self.offset += len;
        if self.offset >= rows {
            self.batch = None;
        }
        chunk.map(DataChunk::from_record_batch)
    }

    /// Make `batch` the current batch, after applying the limits.
    fn set_batch(&mut self, batch: RecordBatch, gstate: &FileScanGlobalState) {
        let mut rows = batch.num_rows();
        // Like `FileStream`, which stops once it produced the limit.
        let mut unit_done = false;
        if let Some(remaining) = self.unit_remaining.as_mut() {
            if rows >= *remaining {
                rows = *remaining;
                unit_done = true;
            }
            *remaining -= rows;
        }
        let rows = gstate.reserve(rows);
        if unit_done {
            self.end_unit();
        }
        if rows > 0 {
            let batch = if rows == batch.num_rows() {
                batch
            } else {
                batch.slice(0, rows)
            };
            self.batch = Some(batch);
            self.offset = 0;
        }
    }

    /// Stop reading the claimed unit.
    fn end_unit(&mut self) {
        self.files.clear();
        self.ready_planners.clear();
        self.ready_morsels.clear();
        self.pending_planner = None;
        self.reader = None;
    }

    /// Claim the next unit of work. Returns `false` if there is none.
    fn claim_unit(&mut self, gstate: &FileScanGlobalState) -> bool {
        loop {
            let unit = gstate.next_unit.fetch_add(1, Ordering::Relaxed);
            let Some(files) = gstate.units.get(unit) else {
                return false;
            };
            if gstate.unit_limit == Some(0) {
                // The unit cannot produce any row.
                continue;
            }
            self.files
                .extend(gstate.files[files.clone()].iter().cloned());
            self.unit_remaining = gstate.unit_limit;
            return true;
        }
    }
}

/// Returns the file scan configuration of `plan` if its file source is a
/// `T`, otherwise an error naming `operator`.
fn file_scan_config<'a, T: FileSource>(
    plan: &'a DataSourceExec,
    operator: &str,
    file_type: &str,
) -> Result<&'a FileScanConfig> {
    match plan.downcast_to_file_source::<T>() {
        Some((config, _)) => Ok(config),
        None => internal_err!(
            "{operator} requires a DataSourceExec that scans {file_type} files"
        ),
    }
}

/// Differential tests of the native scan sources: like
/// [`crate::test_utils::assert_byte_eq_with_datafusion`], but the scans of the
/// plan are lowered to native sources.
///
/// [`PipelineQueryPlanner`](crate::planner::PipelineQueryPlanner) does not
/// lower scans natively yet (it bridges every plan), so the harness uses its
/// own query planner: it plans the query with DataFusion's
/// `DefaultPhysicalPlanner` and replaces every scan that
/// [`try_create_scan_source`] converts by a [`PipelineRootExec`] that runs
/// the native source with the pipeline executor (on `target_partitions`
/// threads). A query that is only a scan is thus executed exactly as the
/// pipeline engine executes it; DataFusion executes the operators above the
/// scans otherwise. The results are compared with
/// [`compare_outputs`](crate::test_utils::compare_outputs): exactly the same
/// schemas and Arrow data (modulo the row order, unless the query has an
/// `ORDER BY`).
///
/// [`PipelineRootExec`]: crate::bridge::PipelineRootExec
#[cfg(test)]
pub(crate) mod test_harness {
    use std::fmt;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{
        ArrayRef, Date32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
        StringArray, TimestampNanosecondArray,
    };
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
    use async_trait::async_trait;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
    use datafusion_common::{DataFusionError, Result};
    use datafusion_expr::LogicalPlan;
    use datafusion_physical_plan::{ExecutionPlan, displayable};
    use datafusion_session::{QueryPlanner, Session};

    use super::try_create_scan_source;
    use crate::bridge::PipelineRootExec;
    use crate::operator::PhysicalOperator;
    use crate::test_utils::{
        compare_outputs, datafusion_context, register_test_tables, run_sql,
        test_session_config,
    };

    /// Converts a scan into a native source, like [`try_create_scan_source`].
    pub(crate) type LowerScan =
        fn(&Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn PhysicalOperator>>>;

    /// Plans queries with DataFusion and runs the scans that `lower`
    /// converts with the pipeline engine.
    struct NativeScanPlanner {
        lower: LowerScan,
        /// The number of scans lowered by the last planned query.
        lowered: Arc<AtomicUsize>,
    }

    impl fmt::Debug for NativeScanPlanner {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("NativeScanPlanner").finish_non_exhaustive()
        }
    }

    #[async_trait]
    impl QueryPlanner for NativeScanPlanner {
        async fn create_physical_plan(
            &self,
            logical_plan: &LogicalPlan,
            session: &dyn Session,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            let plan = DefaultPhysicalPlanner::default()
                .create_physical_plan(logical_plan, session)
                .await?;
            lower_scans(plan, self.lower, &self.lowered)
        }
    }

    /// Replace every scan of `plan` that `lower` converts by a
    /// [`PipelineRootExec`] that runs the native source.
    pub(crate) fn lower_scans(
        plan: Arc<dyn ExecutionPlan>,
        lower: LowerScan,
        lowered: &AtomicUsize,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(source) = lower(&node)? else {
                return Ok(Transformed::no(node));
            };
            lowered.fetch_add(1, Ordering::Relaxed);
            let root: Arc<dyn ExecutionPlan> =
                Arc::new(PipelineRootExec::try_new(source, node)?);
            Ok(Transformed::yes(root))
        })
        .data()
    }

    /// A `SessionContext` whose scans run with native sources (converted by
    /// `lower`).
    fn native_scan_context(
        config: SessionConfig,
        lower: LowerScan,
        lowered: Arc<AtomicUsize>,
    ) -> SessionContext {
        let state = SessionStateBuilder::new()
            .with_config(config)
            .with_runtime_env(Arc::new(RuntimeEnv::default()))
            .with_default_features()
            .with_query_planner(Arc::new(NativeScanPlanner { lower, lowered }))
            .build();
        SessionContext::new_with_state(state)
    }

    /// Assert that `sql` produces byte-identical results on default
    /// DataFusion and with the scans lowered to native sources by `lower`,
    /// on sessions configured with `config` and set up with `setup`.
    ///
    /// Returns the number of lowered scans, which must not be zero.
    pub(crate) async fn assert_native_scans_byte_eq_with<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        lower: LowerScan,
    ) -> usize
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let lowered = Arc::new(AtomicUsize::new(0));
        let outputs = async {
            let datafusion = datafusion_context(config.clone());
            setup(&datafusion).await?;
            let native = native_scan_context(config, lower, Arc::clone(&lowered));
            setup(&native).await?;
            let expected = run_sql(&datafusion, sql).await?;
            lowered.store(0, Ordering::Relaxed);
            let actual = run_sql(&native, sql).await?;
            Ok::<_, DataFusionError>((expected, actual))
        }
        .await;
        let (expected, actual) = match outputs {
            Ok(outputs) => outputs,
            Err(e) => panic!("Failed to run query on both engines: {e}\nquery: {sql}"),
        };
        let plan = displayable(actual.plan.as_ref()).indent(true).to_string();
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\nquery: {sql}\nplan:\n{plan}");
        }
        let lowered = lowered.load(Ordering::Relaxed);
        assert!(
            lowered > 0,
            "No scan was lowered\nquery: {sql}\nplan:\n{plan}"
        );
        lowered
    }

    /// [`assert_native_scans_byte_eq_with`] with [`try_create_scan_source`].
    pub(crate) async fn assert_native_scans_byte_eq_with_config<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> usize
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_native_scans_byte_eq_with(sql, config, setup, try_create_scan_source).await
    }

    /// [`assert_native_scans_byte_eq_with_config`] with the harness's
    /// default configuration.
    pub(crate) async fn assert_native_scans_byte_eq_with_setup<F>(
        sql: &str,
        setup: F,
    ) -> usize
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_native_scans_byte_eq_with_config(sql, test_session_config(), setup).await
    }

    /// [`assert_native_scans_byte_eq_with_setup`] with the tables of
    /// [`register_test_tables`].
    pub(crate) async fn assert_native_scans_byte_eq(sql: &str) -> usize {
        assert_native_scans_byte_eq_with_setup(sql, async |ctx: &SessionContext| {
            register_test_tables(ctx)
        })
        .await
    }

    /// The schema of [`data_batch`].
    pub(crate) fn data_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("a", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("c", DataType::Float64, true),
            Field::new("d", DataType::Date32, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        ]))
    }

    /// A batch of the rows with the ids `ids`: `id` is unique, the other
    /// columns have nulls, `c` has `NaN` and `-0.0`.
    pub(crate) fn data_batch(ids: std::ops::Range<i64>) -> RecordBatch {
        let id = Int64Array::from_iter_values(ids.clone());
        let a = Int32Array::from_iter(
            ids.clone()
                .map(|i| (i % 11 != 3).then_some(((i * 7) % 101) as i32 - 50)),
        );
        let s = StringArray::from_iter(
            ids.clone()
                .map(|i| (i % 13 != 5).then(|| format!("value-{}", i % 37))),
        );
        let c = Float64Array::from_iter(ids.clone().map(|i| match i {
            _ if i % 17 == 2 => None,
            _ if i % 101 == 0 => Some(f64::NAN),
            _ if i % 89 == 7 => Some(-0.0),
            _ => Some(((i * 31) % 1000) as f64 / 8.0 - 60.0),
        }));
        let d = Date32Array::from_iter(
            ids.clone()
                .map(|i| (i % 19 != 4).then_some(19_000 + (i % 365) as i32)),
        );
        let ts = TimestampNanosecondArray::from_iter(ids.map(|i| {
            (i % 23 != 9).then_some(1_700_000_000_000_000_000 + i * 1_000_000_007)
        }));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(id),
            Arc::new(a),
            Arc::new(s),
            Arc::new(c),
            Arc::new(d),
            Arc::new(ts),
        ];
        RecordBatch::try_new(data_schema(), columns).unwrap()
    }

    /// Write `batches` (with schema `schema`) to the Parquet file `path`.
    pub(crate) fn write_parquet(
        path: &Path,
        schema: SchemaRef,
        batches: &[RecordBatch],
        properties: Option<::parquet::file::properties::WriterProperties>,
    ) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::File::create(path)?;
        let mut writer =
            ::parquet::arrow::ArrowWriter::try_new(file, schema, properties)?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
        Ok(())
    }

    /// Write `batches` to the CSV file `path`, with a header.
    pub(crate) fn write_csv(path: &Path, batches: &[RecordBatch]) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::File::create(path)?;
        let mut writer = arrow::csv::Writer::new(file);
        for batch in batches {
            writer.write(batch)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::{
        CsvReadOptions, JsonReadOptions, ParquetReadOptions, SessionContext,
    };
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;

    use super::test_harness::{data_batch, data_schema, write_csv, write_parquet};
    use crate::test_utils::{register_test_tables, test_session_config};

    /// The plan of `sql`, and the leaf of the plan (its scan).
    async fn plan_and_scan(
        ctx: &SessionContext,
        sql: &str,
    ) -> Result<(Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>)> {
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let mut scan = None;
        plan.apply(|node| {
            if node.children().is_empty() {
                scan = Some(Arc::clone(node));
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok((plan, scan.unwrap()))
    }

    #[tokio::test]
    async fn converts_the_supported_scans() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_parquet(
            &dir.path().join("p/data.parquet"),
            data_schema(),
            &[data_batch(0..10)],
            None,
        )?;
        write_csv(&dir.path().join("c/data.csv"), &[data_batch(0..10)])?;
        std::fs::create_dir_all(dir.path().join("j"))?;
        std::fs::write(dir.path().join("j/data.json"), "{\"x\": 1}\n{\"x\": 2}\n")?;

        let ctx = SessionContext::new_with_config(test_session_config());
        register_test_tables(&ctx)?;
        let p = dir.path().join("p");
        ctx.register_parquet("p", p.to_str().unwrap(), ParquetReadOptions::default())
            .await?;
        let c = dir.path().join("c");
        ctx.register_csv("c", c.to_str().unwrap(), CsvReadOptions::new())
            .await?;
        let j = dir.path().join("j");
        ctx.register_json("j", j.to_str().unwrap(), JsonReadOptions::default())
            .await?;

        for (sql, name) in [
            ("SELECT id, d FROM t1", "MemoryScanSource"),
            (
                "SELECT * FROM (VALUES (1), (2)) AS v(x)",
                "MemoryScanSource",
            ),
            ("SELECT s, id FROM p", "ParquetScanSource"),
            ("SELECT s, id FROM c", "CsvScanSource"),
            ("SELECT * FROM t1 WHERE false", "EmptyScanSource"),
        ] {
            let (_, scan) = plan_and_scan(&ctx, sql).await?;
            let source = try_create_scan_source(&scan)?
                .unwrap_or_else(|| panic!("{sql}: {} is not converted", scan.name()));
            assert_eq!(source.name(), name, "{sql}");
            assert_eq!(source.schema(), scan.schema(), "{sql}");
            assert!(source.is_source() && source.parallel_source(), "{sql}");
            assert!(source.children().is_empty(), "{sql}");
            assert!(source.description().starts_with(name), "{sql}");
            // The alias.
            let source = try_from_execution_plan(&scan)?.unwrap();
            assert_eq!(source.name(), name, "{sql}");
        }
        assert!(
            source_description(&ctx, "SELECT s, id FROM p")
                .await?
                .contains("projection=[s, id], file_type=parquet")
        );

        // Scans of other formats and other plans are not converted.
        for sql in ["SELECT x FROM j", "SELECT 1"] {
            let (plan, scan) = plan_and_scan(&ctx, sql).await?;
            assert!(try_create_scan_source(&scan)?.is_none(), "{sql}");
            assert!(try_create_scan_source(&plan)?.is_none(), "{sql}");
        }
        let (_, scan) = plan_and_scan(&ctx, "SELECT 1").await?;
        assert!(scan.is::<PlaceholderRowExec>());
        let (plan, _) = plan_and_scan(&ctx, "SELECT id + 1 FROM t1").await?;
        assert!(try_create_scan_source(&plan)?.is_none());
        Ok(())
    }

    async fn source_description(ctx: &SessionContext, sql: &str) -> Result<String> {
        let (_, scan) = plan_and_scan(ctx, sql).await?;
        Ok(try_create_scan_source(&scan)?.unwrap().description())
    }

    #[test]
    fn large_files_are_split_into_byte_ranges() {
        let file = PartitionedFile::new("file.parquet", 10_000);
        let ranges = |file: &PartitionedFile, range_bytes: Option<u64>| {
            let mut files = vec![];
            push_file_ranges(&mut files, file, range_bytes);
            files
                .iter()
                .map(|file| file.range.as_ref().map(|range| (range.start, range.end)))
                .collect::<Vec<_>>()
        };
        assert_eq!(ranges(&file, None), vec![None]);
        assert_eq!(ranges(&file, Some(5001)), vec![None]);
        assert_eq!(
            ranges(&file, Some(5000)),
            vec![Some((0, 5000)), Some((5000, 10_000))]
        );
        assert_eq!(
            ranges(&file, Some(3000)),
            vec![Some((0, 3334)), Some((3334, 6668)), Some((6668, 10_000))]
        );
        // The ranges of a range cover exactly that range.
        let range = file.clone().with_range(1000, 8001);
        assert_eq!(
            ranges(&range, Some(2000)),
            vec![Some((1000, 3334)), Some((3334, 5668)), Some((5668, 8001))]
        );
        assert_eq!(ranges(&range, Some(4000)), vec![Some((1000, 8001))]);
        // Files with extensions are never split.
        let extended = file.with_extension(42_u32);
        assert_eq!(ranges(&extended, Some(1000)), vec![None]);
    }

    #[test]
    fn display_data_source() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let config = MemorySourceConfig::try_new(
            &[vec![RecordBatch::new_empty(Arc::clone(&schema))]],
            schema,
            None,
        )?;
        assert_eq!(
            DisplayDataSource(&config).to_string(),
            "partitions=1, partition_sizes=[1]"
        );
        Ok(())
    }
}
