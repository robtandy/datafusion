// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`ParquetScanSource`]: a parallel, morsel-driven scan of Parquet files.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Waker;

use arrow::datatypes::SchemaRef;
use datafusion_common::parquet_config::RowGroupRangeAssignment;
use datafusion_common::{Result, internal_err};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_parquet::metadata::CachedParquetMetaData;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::cache::cache_manager::{
    CachedFileMetadataEntry, FileMetadataCache,
};
use datafusion_physical_plan::ExecutionPlan;
use parking_lot::Mutex;
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};

use super::{
    FileScan, FileScanGlobalState, FileScanMode, file_scan_config, push_file_ranges,
};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state, downcast_state_mut,
};

/// The smallest number of rows of a unit of work of row groups (see
/// [`ParquetScanSource`]): consecutive row groups of a file are claimed
/// together until they have at least this many rows, so that a file of many
/// small row groups is not opened (planned and pruned) once for every row
/// group.
const MIN_UNIT_ROWS: usize = 32 * DEFAULT_VECTOR_SIZE;

/// A source that scans Parquet files on all threads of a pipeline.
///
/// It replaces a `DataSourceExec` whose [`FileScanConfig`] scans Parquet
/// files ([`ParquetSource`]). Instead of reading the file group of a
/// partition on every thread, all files of all file groups form a single
/// queue of units of work that the threads share: a thread that needs work
/// claims the next unit.
///
/// The units are row groups, as in DuckDB's Parquet scans: the files whose
/// metadata DataFusion's file metadata cache holds (which it does once the
/// table was planned with statistics, or read once) are split at the
/// boundaries of their row groups, into byte ranges that each read one row
/// group, or consecutive row groups of at least 65,536 rows together. So
/// the threads share the work of a file by row groups: a thread that claimed
/// a file of several row groups does not read all of them while the other
/// threads wait for it at the end of the pipeline (the last of the threads
/// finished 40% to 50% of the time of a pipeline after the first one in
/// TPC-H SF 1 queries, whose files have three row groups), and the byte
/// ranges of DataFusion's file groups that contain no row group (e.g. the
/// last bytes of a file, which DataFusion assigns to the next file group)
/// are not opened. Files that are not cached, and all files if the size of
/// the byte ranges is set (see [`Self::with_range_bytes`]), are split into
/// byte ranges instead: files larger than twice the range size are split
/// into ranges of at least the range size, each of which reads the row
/// groups that DataFusion assigns to it, so a single large file is read by
/// many threads too. Once no unit is left, the threads that still read hand
/// their chunks to the threads that ran out of units, so that the threads
/// finish together.
///
/// Every thread reads its files with its own
/// [`Morselizer`](datafusion_datasource::morsel::Morselizer), created by
/// [`ParquetSource`] exactly as for a partition of the `DataSourceExec`. So
/// all of DataFusion's Parquet optimizations apply: projection pushdown,
/// file, row group and page pruning with statistics and the page index,
/// bloom filters, row filters (`pushdown_filters`), dynamic filters (e.g.
/// from a TopK or a hash join), schema adaptation and coercion, partition
/// column values, limits, and the TopK file order (`reorder_files`). Decoded
/// batches are sliced into chunks of at most
/// [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows without
/// copying.
///
/// A scan that must preserve the order of its files
/// ([`FileScanConfig::preserves_file_order`], e.g. a scan with an output
/// ordering) reads every file group, in order, on a single thread instead,
/// and applies the limit to every file group, like DataFusion. See the
/// [module documentation](super) for the rows the source produces.
#[derive(Debug)]
pub struct ParquetScanSource {
    scan: FileScan,
    /// See [`MIN_UNIT_ROWS`].
    min_unit_rows: usize,
}

impl ParquetScanSource {
    /// Create a source that scans the Parquet files of `plan`.
    ///
    /// Returns an error if `plan` does not scan Parquet files.
    pub fn try_new(plan: &DataSourceExec) -> Result<Self> {
        let config =
            file_scan_config::<ParquetSource>(plan, "ParquetScanSource", "Parquet")?;
        Ok(Self {
            scan: FileScan::new("ParquetScanSource", config.clone(), plan.schema()),
            min_unit_rows: MIN_UNIT_ROWS,
        })
    }

    /// Create a source that scans the Parquet files of `config`.
    ///
    /// Returns an error if the file source of `config` is not a
    /// [`ParquetSource`].
    pub fn try_from_config(config: FileScanConfig) -> Result<Self> {
        if !config.file_source().is::<ParquetSource>() {
            return internal_err!(
                "ParquetScanSource requires a FileScanConfig with a ParquetSource, got {}",
                config.file_source().file_type()
            );
        }
        let schema = config.projected_schema()?;
        Ok(Self {
            scan: FileScan::new("ParquetScanSource", config, schema),
            min_unit_rows: MIN_UNIT_ROWS,
        })
    }

    /// Set the size, in bytes, of the byte ranges that files are split into,
    /// instead of splitting them at the boundaries of their row groups: a
    /// file (or byte range of a file) of at least twice this size is split
    /// into ranges of at least this size. `0` disables splitting.
    ///
    /// By default, files whose metadata is cached are split at the
    /// boundaries of their row groups, and the other files into byte ranges
    /// of the maximum of `datafusion.optimizer.repartition_file_min_size` and
    /// 16 MiB, if `datafusion.optimizer.repartition_file_scans` is enabled.
    /// Files are never split if the scan has a limit, must preserve the order
    /// of its files, or a file has extensions (such as a
    /// `ParquetAccessPlan`).
    pub fn with_range_bytes(self, range_bytes: u64) -> Self {
        Self {
            scan: self.scan.with_range_bytes(range_bytes),
            ..self
        }
    }

    /// Set the smallest number of rows of the units of work that files are
    /// split into at the boundaries of their row groups (see
    /// [`MIN_UNIT_ROWS`]): `1` makes every row group a unit.
    #[cfg(test)]
    fn with_min_unit_rows(self, min_unit_rows: usize) -> Self {
        Self {
            min_unit_rows,
            ..self
        }
    }

    /// The configuration of the scan.
    pub fn config(&self) -> &FileScanConfig {
        &self.scan.config
    }

    /// The units of work of an execution in shared mode: the files of the
    /// scan, the files whose metadata `cache` holds split at the boundaries
    /// of their row groups (see [`push_row_group_units`]), and the other
    /// files into byte ranges of at least `range_bytes`, as
    /// [`FileScan`] splits them.
    fn units(&self, cache: &FileMetadataCache, range_bytes: u64) -> Vec<PartitionedFile> {
        let assignment = self
            .scan
            .config
            .file_source()
            .downcast_ref::<ParquetSource>()
            .map(|source| {
                source
                    .table_parquet_options()
                    .global
                    .row_group_range_assignment
            })
            .unwrap_or_default();
        let mut units = Vec::with_capacity(self.scan.files.len());
        for file in &self.scan.files {
            // A file with extensions (e.g. a `ParquetAccessPlan` of its row
            // groups) is read as it is.
            let row_groups = file
                .extensions
                .is_empty()
                .then(|| cached_row_groups(cache, file, assignment))
                .flatten();
            match row_groups {
                Some(row_groups) => push_row_group_units(
                    &mut units,
                    file,
                    &row_groups,
                    self.min_unit_rows,
                ),
                None => push_file_ranges(&mut units, file, Some(range_bytes)),
            }
        }
        units
    }

    /// The global state of an execution that shares the files between all
    /// threads (shared mode, without a limit) and splits them, by default:
    /// with the units of [`Self::units`]. `None` for the scans that
    /// [`FileScan`] splits into byte ranges of a set size, or does not split.
    fn row_group_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<FileScanGlobalState>> {
        let scan = &self.scan;
        let (None, Some(range_bytes)) = (scan.range_bytes, scan.range_bytes(ctx)) else {
            return Ok(None);
        };
        let task_ctx = &ctx.task_ctx;
        let runtime = task_ctx.runtime_env();
        let files = self.units(
            runtime.cache_manager.get_file_metadata_cache().as_ref(),
            range_bytes,
        );
        let object_store = runtime.object_store(&scan.config.object_store_url)?;
        let batch_size = scan
            .config
            .batch_size
            .unwrap_or_else(|| task_ctx.session_config().batch_size());
        Ok(Some(FileScanGlobalState {
            units: (0..files.len()).map(|i| i..i + 1).collect(),
            files,
            next_unit: AtomicUsize::new(0),
            remaining: scan.config.limit.map(AtomicUsize::new),
            unit_limit: None,
            object_store,
            file_source: scan.config.file_source.with_batch_size(batch_size),
        }))
    }
}

/// The row groups of `file` according to the metadata of `cache`, if it
/// holds the metadata of `file` (and it is up to date): for every row group,
/// the offset by which DataFusion assigns it to the byte ranges of `file`
/// (see [`row_group_offset`]) and its rows.
fn cached_row_groups(
    cache: &FileMetadataCache,
    file: &PartitionedFile,
    assignment: RowGroupRangeAssignment,
) -> Option<Vec<(i64, usize)>> {
    let entry = cache.get(&file.object_meta.location)?;
    let metadata = parquet_metadata(&entry, file)?;
    metadata
        .row_groups()
        .iter()
        .map(|row_group| {
            let rows = usize::try_from(row_group.num_rows()).ok()?;
            Some((row_group_offset(row_group, assignment)?, rows))
        })
        .collect()
}

/// The Parquet metadata of `entry`, the cache entry of `file`, if it is up
/// to date.
fn parquet_metadata<'a>(
    entry: &'a CachedFileMetadataEntry,
    file: &PartitionedFile,
) -> Option<&'a ParquetMetaData> {
    if !entry.is_valid_for(&file.object_meta) {
        return None;
    }
    let metadata = entry.file_metadata.as_any();
    match metadata.downcast_ref::<CachedParquetMetaData>() {
        Some(cached) => Some(cached.parquet_metadata()),
        None => metadata
            .downcast_ref::<datafusion_datasource_parquet::CachedParquetMetaData>()
            .map(|cached| cached.parquet_metadata().as_ref()),
    }
}

/// The offset of `row_group` by which DataFusion assigns it to the byte range
/// of a file that contains it (`row_group_in_range` of the Parquet opener):
/// the start of its first column chunk, or the middle of the row group with
/// [`RowGroupRangeAssignment::Midpoint`]. `None` for a row group without
/// columns.
fn row_group_offset(
    row_group: &RowGroupMetaData,
    assignment: RowGroupRangeAssignment,
) -> Option<i64> {
    let column = row_group.columns().first()?;
    let data_page_offset = column.data_page_offset();
    Some(match assignment {
        RowGroupRangeAssignment::StartOffset => {
            column.dictionary_page_offset().unwrap_or(data_page_offset)
        }
        RowGroupRangeAssignment::Midpoint => column
            .dictionary_page_offset()
            .map_or(data_page_offset, |offset| offset.min(data_page_offset))
            .saturating_add(row_group.compressed_size() / 2),
    })
}

/// Push the units of `file` (a file or a byte range of a file) to `units`:
/// the byte ranges of `file` that each contain the offsets (see
/// [`row_group_offset`]) of consecutive row groups of at least `min_rows`
/// rows (fewer at the end of `file`), of the row groups `row_groups` of the
/// file (offset and rows). The ranges cover `file` exactly, so DataFusion
/// reads every row group of `file` in exactly one of them, whatever the row
/// groups really are. A byte range that contains no row group (e.g. the last
/// bytes of a file, which DataFusion assigns to a file group of their own)
/// has no unit: it would read nothing.
fn push_row_group_units(
    units: &mut Vec<PartitionedFile>,
    file: &PartitionedFile,
    row_groups: &[(i64, usize)],
    min_rows: usize,
) {
    let (start, end) = file.range();
    let (Ok(start), Ok(end)) = (i64::try_from(start), i64::try_from(end)) else {
        units.push(file.clone());
        return;
    };
    let mut in_range = row_groups
        .iter()
        .copied()
        .filter(|(offset, _)| (start..end).contains(offset))
        .collect::<Vec<_>>();
    if in_range.is_empty() {
        return;
    }
    in_range.sort_unstable();
    // The offsets of the first row groups of the units after the first.
    let mut cuts = vec![];
    let mut rows = 0;
    for (offset, row_group_rows) in in_range {
        if rows >= min_rows.max(1) {
            cuts.push(offset);
            rows = 0;
        }
        rows += row_group_rows;
    }
    if cuts.is_empty() {
        units.push(file.clone());
        return;
    }
    let mut range_start = start;
    for range_end in cuts.into_iter().chain([end]) {
        units.push(file.clone().with_range(range_start, range_end));
        range_start = range_end;
    }
}

impl PhysicalOperator for ParquetScanSource {
    fn name(&self) -> &str {
        self.scan.name
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.scan.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        self.scan.description()
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let scan: Arc<dyn GlobalSourceState> = match self.row_group_state(ctx)? {
            Some(state) => Arc::new(state),
            None => self.scan.get_global_source_state(ctx)?,
        };
        // The threads of an ordered scan read their file groups in order, and
        // a limit may end a scan before its units do.
        let shares =
            self.scan.mode == FileScanMode::Shared && self.scan.config.limit.is_none();
        Ok(Arc::new(ParquetScanState {
            scan,
            sharing: shares.then(Arc::default),
        }))
    }

    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        let gstate = downcast_state::<ParquetScanState>(gstate.as_any(), self.name())?;
        let scan = self
            .scan
            .get_local_source_state(ctx, gstate.scan.as_ref())?;
        let sharing = gstate.sharing.as_ref().map(|sharing| {
            sharing.add_reader();
            Arc::clone(sharing)
        });
        Ok(Box::new(ParquetScanLocalState { scan, sharing }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let gstate = downcast_state::<ParquetScanState>(gstate.as_any(), self.name())?;
        let lstate = downcast_state_mut::<ParquetScanLocalState>(
            lstate.as_any_mut(),
            self.name(),
        )?;
        let Some(sharing) = gstate.sharing.as_deref() else {
            return self
                .scan
                .get_data(ctx, gstate.scan.as_ref(), lstate.scan.as_mut());
        };
        // A reader claims units, an idle thread takes the chunks of readers.
        while lstate.sharing.is_some() {
            match self
                .scan
                .get_data(ctx, gstate.scan.as_ref(), lstate.scan.as_mut())?
            {
                SourceResult::Chunk(chunk) if !chunk.is_empty() => {
                    // Hand the chunk to an idle thread if one waits, and
                    // read on.
                    if let Some(chunk) = sharing.offer(chunk) {
                        return Ok(SourceResult::Chunk(chunk));
                    }
                }
                SourceResult::Finished => {
                    // No unit is left: the thread becomes idle.
                    lstate.sharing = None;
                    sharing.remove_reader();
                }
                result => return Ok(result),
            }
        }
        Ok(sharing.take(ctx.thread_id, ctx.waker()))
    }
}

/// The global state of a [`ParquetScanSource`]: the state of its
/// [`FileScan`], whose units the threads claim, and the chunks that they share
/// at the end of the scan.
struct ParquetScanState {
    scan: Arc<dyn GlobalSourceState>,
    /// `None` for the scans that do not share their chunks: the ordered
    /// scans, and those with a limit.
    sharing: Option<Arc<ChunkSharing>>,
}

impl GlobalSourceState for ParquetScanState {
    fn max_threads(&self) -> usize {
        self.scan.max_threads()
    }
}

/// The local state of a [`ParquetScanSource`]: the state of its [`FileScan`].
struct ParquetScanLocalState {
    scan: Box<dyn LocalSourceState>,
    /// The chunk sharing of the scan while the thread is one of its readers,
    /// i.e. it may still claim units; `None` once it is idle, and for scans
    /// that do not share their chunks.
    sharing: Option<Arc<ChunkSharing>>,
}

impl LocalSourceState for ParquetScanLocalState {}

impl Drop for ParquetScanLocalState {
    fn drop(&mut self) {
        // A thread that stops reading (e.g. its pipeline was canceled) no
        // longer produces chunks for the idle threads.
        if let Some(sharing) = self.sharing.take() {
            sharing.remove_reader();
        }
    }
}

/// The decoded chunks that the threads of a scan share at its end, so that
/// they finish together.
///
/// A thread decodes the row groups of the units it claimed, and pushes their
/// chunks through the operators of its pipeline. Once no unit is left to
/// claim, a thread becomes idle: the threads that are still reading (the
/// readers) hand their next chunks to the idle threads, one for every thread
/// that waits, instead of processing all of them themselves, and the idle
/// threads push them through their own operators. So the work of the last
/// units, mostly that of the operators above the scan (joins, aggregations),
/// is shared by all threads: without sharing, threads that claimed the same
/// number of rows finished as much as a fifth of the time of a pipeline
/// apart in TPC-H SF 1 queries, as their reads of the files and the other
/// processes on the machine slow them down differently. An idle thread
/// finishes once all readers finished and the shared chunks are taken. The
/// shared chunks are slices of the batches that the readers decoded, at most
/// one for every idle thread.
#[derive(Debug, Default)]
struct ChunkSharing {
    /// The number of idle threads waiting for a chunk (the length of
    /// `SharedChunks::idle`), which the readers check without the lock.
    waiting: AtomicUsize,
    shared: Mutex<SharedChunks>,
}

#[derive(Debug, Default)]
struct SharedChunks {
    /// The threads that may still claim units.
    readers: usize,
    /// The chunks handed to idle threads that did not take them yet.
    chunks: VecDeque<DataChunk>,
    /// The idle threads waiting for a chunk: thread id and waker.
    idle: Vec<(usize, Waker)>,
}

impl ChunkSharing {
    /// Register a thread that may claim units.
    fn add_reader(&self) {
        self.shared.lock().readers += 1;
    }

    /// A reader stopped claiming units. Once the last one did, the idle
    /// threads finish.
    fn remove_reader(&self) {
        let idle = {
            let mut shared = self.shared.lock();
            shared.readers = shared.readers.saturating_sub(1);
            if shared.readers > 0 {
                return;
            }
            self.waiting.store(0, Ordering::Relaxed);
            std::mem::take(&mut shared.idle)
        };
        for (_, waker) in idle {
            waker.wake();
        }
    }

    /// Hand `chunk`, of a reader, to an idle thread that waits for one. Returns
    /// `chunk` if no thread waits.
    fn offer(&self, chunk: DataChunk) -> Option<DataChunk> {
        if self.waiting.load(Ordering::Relaxed) == 0 {
            return Some(chunk);
        }
        let waker = {
            let mut shared = self.shared.lock();
            let Some((_, waker)) = shared.idle.pop() else {
                return Some(chunk);
            };
            shared.chunks.push_back(chunk);
            self.waiting.store(shared.idle.len(), Ordering::Relaxed);
            waker
        };
        waker.wake();
        None
    }

    /// The next chunk of the idle thread `thread_id`: a shared chunk,
    /// `Finished` once no reader is left, or `Blocked` until a reader hands
    /// it a chunk or the last reader finishes (which wake `waker`).
    fn take(&self, thread_id: usize, waker: &Waker) -> SourceResult {
        let mut shared = self.shared.lock();
        let registered = shared.idle.iter().position(|(id, _)| *id == thread_id);
        if let Some(chunk) = shared.chunks.pop_front() {
            // The thread does not wait any more (if it was woken for another
            // reason than this chunk, the thread that was woken for it waits
            // again).
            if let Some(index) = registered {
                shared.idle.swap_remove(index);
                self.waiting.store(shared.idle.len(), Ordering::Relaxed);
            }
            return SourceResult::Chunk(chunk);
        }
        if shared.readers == 0 {
            return SourceResult::Finished;
        }
        match registered {
            Some(index) => shared.idle[index].1.clone_from(waker),
            None => shared.idle.push((thread_id, waker.clone())),
        }
        self.waiting.store(shared.idle.len(), Ordering::Relaxed);
        SourceResult::Blocked(TaskInterrupt::waker_registered())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;
    use std::time::Duration;

    use arrow::array::{AsArray, Int32Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Int64Type, Schema};
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext, col};
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_datasource::file_groups::FileGroup;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_datasource_csv::source::CsvSource;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_plan::metrics::MetricValue;
    use parquet::file::properties::WriterProperties;
    use parquet::schema::types::ColumnPath;
    use tempfile::TempDir;

    use crate::executor::Executor;
    use crate::sources::test_harness::{
        LowerScan, assert_native_scans_byte_eq_with,
        assert_native_scans_byte_eq_with_config, assert_native_scans_byte_eq_with_setup,
        data_batch, data_schema, write_parquet,
    };
    use crate::sources::try_create_scan_source;
    use crate::test_utils::{
        pipeline_context, register_test_tables, run_sql, test_session_config,
    };

    /// Write the Hive-partitioned table of [`register_table`] to `dir`:
    ///
    /// * `year=2020`: 5000 rows in row groups of 1000 rows, with data pages
    ///   of 100 rows and a bloom filter on `s`;
    /// * `year=2021`: 100 rows; `year=2022`: 7 rows;
    /// * `year=2023`: a file without rows.
    fn write_table(dir: &Path) -> Result<()> {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1000))
            .set_data_page_row_count_limit(100)
            .set_write_batch_size(100)
            .set_column_bloom_filter_enabled(ColumnPath::from("s"), true)
            .build();
        write_parquet(
            &dir.join("year=2020/data.parquet"),
            data_schema(),
            &[data_batch(0..2500), data_batch(2500..5000)],
            Some(properties),
        )?;
        write_parquet(
            &dir.join("year=2021/data.parquet"),
            data_schema(),
            &[data_batch(5000..5100)],
            None,
        )?;
        write_parquet(
            &dir.join("year=2022/data.parquet"),
            data_schema(),
            &[data_batch(5100..5107)],
            None,
        )?;
        write_parquet(
            &dir.join("year=2023/data.parquet"),
            data_schema(),
            &[],
            None,
        )
    }

    /// Register the table `p` of `dir` (see [`write_table`]), partitioned by
    /// `year`, and the tables of [`register_test_tables`].
    async fn register_table(ctx: &SessionContext, dir: &Path) -> Result<()> {
        register_test_tables(ctx)?;
        ctx.register_parquet(
            "p",
            dir.to_str().unwrap(),
            ParquetReadOptions::default()
                .table_partition_cols(vec![("year".to_string(), DataType::Int32)]),
        )
        .await
    }

    fn table() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_table(dir.path()).unwrap();
        dir
    }

    /// The first `DataSourceExec` of `plan`.
    fn find_scan(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let mut scan = None;
        plan.apply(|node| {
            if node.is::<DataSourceExec>() {
                scan = Some(Arc::clone(node));
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        scan.expect("the plan has a scan")
    }

    /// The scan of the plan of `sql`, lowered to a [`ParquetScanSource`].
    async fn scan_source(ctx: &SessionContext, sql: &str) -> Result<ParquetScanSource> {
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let scan = find_scan(&plan);
        ParquetScanSource::try_new(scan.downcast_ref::<DataSourceExec>().unwrap())
    }

    /// The ids and years of `batches` (whose first column is `id` and last
    /// column is `year`).
    fn ids_and_years(batches: &[RecordBatch]) -> Vec<(i64, i32)> {
        let mut rows = batches
            .iter()
            .flat_map(|batch| {
                let ids = batch.column(0).as_primitive::<Int64Type>().clone();
                let years = batch
                    .column(batch.num_columns() - 1)
                    .as_primitive::<Int32Type>()
                    .clone();
                ids.values()
                    .iter()
                    .copied()
                    .zip(years.values().iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    }

    /// The year of the file that holds `id`, see [`write_table`].
    fn year_of(id: i64) -> i32 {
        match id {
            ..5000 => 2020,
            5000..5100 => 2021,
            _ => 2022,
        }
    }

    const QUERIES: &[&str] = &[
        "SELECT * FROM p",
        "SELECT id, s, year FROM p",
        "SELECT year FROM p",
        "SELECT id, a FROM p WHERE a > 40",
        "SELECT * FROM p WHERE year = 2021",
        "SELECT id, year FROM p WHERE year > 2020 AND a < 0",
        "SELECT * FROM p WHERE s = 'value-3'",
        "SELECT id, c FROM p WHERE id BETWEEN 1510 AND 1720",
        "SELECT id, c, d, ts FROM p WHERE c < 0 OR c IS NULL OR isnan(c)",
        "SELECT year, COUNT(*), SUM(a), MIN(s), MAX(d) FROM p GROUP BY year",
        "SELECT COUNT(*), COUNT(a) FROM p WHERE a > 3",
        "SELECT p.id, t2.name FROM p JOIN t2 ON p.a = t2.k",
        "SELECT p.id, t2.w FROM t2 JOIN p ON p.a = t2.k AND p.year = 2020 WHERE t2.w > 100",
        "SELECT id, a, year FROM p ORDER BY a DESC NULLS LAST, id LIMIT 10",
        "SELECT * FROM p ORDER BY id LIMIT 25 OFFSET 4990",
        // A limit on a single file: the first rows of the file.
        "SELECT * FROM p WHERE year = 2021 LIMIT 50",
        "SELECT * FROM p WHERE year = 2023",
    ];

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_scans() {
        let dir = table();
        let setup = async |ctx: &SessionContext| register_table(ctx, dir.path()).await;
        for sql in QUERIES {
            assert_native_scans_byte_eq_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_scans_with_pushdown_filters() {
        let dir = table();
        let setup = async |ctx: &SessionContext| register_table(ctx, dir.path()).await;
        let mut config = test_session_config();
        let parquet = &mut config.options_mut().execution.parquet;
        parquet.pushdown_filters = true;
        parquet.reorder_filters = true;
        for sql in QUERIES {
            assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup).await;
        }
        // Without page index and bloom filters.
        let parquet = &mut config.options_mut().execution.parquet;
        parquet.enable_page_index = false;
        parquet.bloom_filter_on_read = false;
        parquet.pruning = false;
        for sql in &QUERIES[3..9] {
            assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_thread_counts() {
        let dir = table();
        let setup = async |ctx: &SessionContext| register_table(ctx, dir.path()).await;
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [QUERIES[0], QUERIES[3], QUERIES[9], QUERIES[11], QUERIES[13]] {
                assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup)
                    .await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_byte_ranges() {
        // Split the large file into byte ranges of about 4 KiB, so that its
        // five row groups are read by different threads, and some ranges
        // contain no row group at all.
        let dir = table();
        let setup = async |ctx: &SessionContext| register_table(ctx, dir.path()).await;
        let lower: LowerScan = |plan| match plan.downcast_ref::<DataSourceExec>() {
            Some(exec) if exec.downcast_to_file_source::<ParquetSource>().is_some() => {
                let source = ParquetScanSource::try_new(exec)?.with_range_bytes(4096);
                Ok(Some(Arc::new(source)))
            }
            _ => try_create_scan_source(plan),
        };
        for sql in QUERIES.iter().take(13) {
            assert_native_scans_byte_eq_with(sql, test_session_config(), &setup, lower)
                .await;
        }
        let mut config = test_session_config();
        config.options_mut().execution.parquet.pushdown_filters = true;
        for sql in &QUERIES[3..9] {
            assert_native_scans_byte_eq_with(sql, config.clone(), &setup, lower).await;
        }
    }

    /// The number of pruned items of the pruning metric `name` of the scans
    /// of `scan`.
    fn pruned(scan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        match scan.metrics().unwrap().sum_by_name(name) {
            Some(MetricValue::PruningMetrics {
                pruning_metrics, ..
            }) => pruning_metrics.pruned(),
            Some(other) => other.as_usize(),
            None => 0,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn native_scans_prune_like_datafusion() -> Result<()> {
        // Pruning does not change results, so check DataFusion's Parquet
        // metrics, recorded by the morselizers of the native source.
        let dir = table();
        for (pushdown_filters, sql, metric, rows) in [
            // `value-1x` is within the statistics of every row group, but in
            // no bloom filter.
            (
                false,
                "SELECT * FROM p WHERE s = 'value-1x'",
                "row_groups_pruned_bloom_filter",
                0,
            ),
            (
                false,
                "SELECT * FROM p WHERE id BETWEEN 1510 AND 1720",
                "row_groups_pruned_statistics",
                211,
            ),
            (
                false,
                "SELECT * FROM p WHERE id BETWEEN 1510 AND 1720",
                "page_index_rows_pruned",
                211,
            ),
            (
                false,
                "SELECT * FROM p WHERE a > 1000",
                "files_ranges_pruned_statistics",
                0,
            ),
            (
                true,
                "SELECT id FROM p WHERE a = 7",
                "pushdown_rows_pruned",
                46,
            ),
        ] {
            let mut config = test_session_config();
            config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
            let ctx = SessionContext::new_with_config(config);
            register_table(&ctx, dir.path()).await?;
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let scan = find_scan(&plan);
            let source: Arc<dyn PhysicalOperator> = Arc::new(ParquetScanSource::try_new(
                scan.downcast_ref::<DataSourceExec>().unwrap(),
            )?);
            let output = Executor::collect(source, ctx.task_ctx(), 4).await?;
            let output_rows = output.iter().map(RecordBatch::num_rows).sum::<usize>();
            if pushdown_filters {
                assert_eq!(output_rows, rows, "{sql}");
            } else {
                // Without filter pushdown, the scan only prunes: a filter
                // above it removes the other rows.
                assert!((rows..5107).contains(&output_rows), "{sql}: {output_rows}");
            }
            assert!(pruned(&scan, metric) > 0, "{sql}: {metric}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_many_pruned_files() {
        // Many files, most of which are pruned with their statistics: a
        // thread plans many files without producing rows.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..60 {
            write_parquet(
                &dir.path().join(format!("{i:02}.parquet")),
                data_schema(),
                &[data_batch(i * 100..i * 100 + 100)],
                None,
            )
            .unwrap();
        }
        let setup = async |ctx: &SessionContext| {
            let path = dir.path().to_str().unwrap();
            ctx.register_parquet("many", path, ParquetReadOptions::default())
                .await
        };
        let mut pushdown = test_session_config();
        pushdown.options_mut().execution.parquet.pushdown_filters = true;
        for config in [test_session_config(), pushdown] {
            for sql in [
                "SELECT * FROM many WHERE id = 4321",
                "SELECT id, s FROM many WHERE id > 5950 OR id < 3",
                "SELECT * FROM many WHERE id < 0",
                "SELECT COUNT(*), SUM(a) FROM many WHERE a > 0",
            ] {
                assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup)
                    .await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn files_are_split_into_byte_ranges() -> Result<()> {
        let dir = table();
        let ctx = SessionContext::new_with_config(test_session_config());
        register_table(&ctx, dir.path()).await?;
        let source = scan_source(&ctx, "SELECT * FROM p").await?;
        let exec_ctx = ExecutionContext::new(ctx.task_ctx(), 0, 4);
        // By default the files are split at their row groups (see
        // `files_are_split_at_row_groups`): the row groups of the large file
        // are too small to be units of their own, and the empty file has no
        // unit.
        let files = source
            .config()
            .file_groups
            .iter()
            .map(|g| g.len())
            .sum::<usize>();
        assert_eq!(files, 4);
        assert_eq!(source.get_global_source_state(&exec_ctx)?.max_threads(), 3);

        let file_size =
            std::fs::metadata(dir.path().join("year=2020/data.parquet"))?.len();
        let source = source.with_range_bytes(4096);
        let units = source.get_global_source_state(&exec_ctx)?.max_threads();
        assert_eq!(
            units,
            3 + (file_size / 4096) as usize,
            "file size {file_size}"
        );
        let output = Executor::collect(Arc::new(source), ctx.task_ctx(), 4).await?;
        let expected = (0..5107).map(|id| (id, year_of(id))).collect::<Vec<_>>();
        assert_eq!(ids_and_years(&output), expected);

        // Not with a limit.
        let source = scan_source(&ctx, "SELECT * FROM p LIMIT 7").await?;
        let source = source.with_range_bytes(4096);
        let units = source.get_global_source_state(&exec_ctx)?.max_threads();
        let files = source
            .config()
            .file_groups
            .iter()
            .map(|g| g.len())
            .sum::<usize>();
        assert_eq!(units, files);
        Ok(())
    }

    /// The number of units of work of an execution of `source` (the threads
    /// it can use).
    fn units(source: &ParquetScanSource, ctx: &SessionContext) -> Result<usize> {
        let exec_ctx = ExecutionContext::new(ctx.task_ctx(), 0, 4);
        Ok(source.get_global_source_state(&exec_ctx)?.max_threads())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn files_are_split_at_row_groups() -> Result<()> {
        // The threads share the work of a file by row groups, so that a
        // thread that claimed a file of several row groups does not read all
        // of them alone. The metadata of the files is cached once the table
        // was planned (with its statistics).
        let dir = table();
        let ctx = SessionContext::new_with_config(test_session_config());
        register_table(&ctx, dir.path()).await?;
        let source = scan_source(&ctx, "SELECT * FROM p").await?;
        // The large file has 5 row groups of 1000 rows; the files of 2021 and
        // 2022 one each, and the empty file of 2023 none: no unit.
        let expected = (0..5107).map(|id| (id, year_of(id))).collect::<Vec<_>>();
        for (min_unit_rows, large_file_units) in [
            (1, 5),
            (1000, 5),
            (1001, 3),
            (2000, 3),
            (2001, 2),
            (5000, 1),
        ] {
            let source = Arc::new(
                ParquetScanSource::try_from_config(source.config().clone())?
                    .with_min_unit_rows(min_unit_rows),
            );
            assert_eq!(
                units(&source, &ctx)?,
                2 + large_file_units,
                "{min_unit_rows}"
            );
            for threads in [1, 4] {
                let output = Executor::collect(
                    Arc::clone(&source) as Arc<dyn PhysicalOperator>,
                    ctx.task_ctx(),
                    threads,
                )
                .await?;
                assert_eq!(ids_and_years(&output), expected, "{min_unit_rows}");
            }
        }
        // By default, row groups of fewer than `MIN_UNIT_ROWS` rows are
        // claimed together.
        assert_eq!(units(&source, &ctx)?, 3);

        // The byte ranges of the file groups of DataFusion are split at the
        // row groups that they read, and those that contain none have no unit:
        // here the first range reads the first row group (whose first column
        // chunk starts at byte 4), the second range the other four, and the
        // last bytes of the file none.
        let large =
            |file: &PartitionedFile| file.object_meta.location.as_ref().contains("2020");
        let files = source
            .config()
            .file_groups
            .iter()
            .flat_map(|group| group.iter().cloned());
        let (large_file, others): (Vec<_>, Vec<_>) = files.partition(large);
        let [large_file] = large_file.try_into().unwrap();
        let size = large_file.object_meta.size as i64;
        let ranges = vec![
            large_file.clone().with_range(0, 10),
            large_file.clone().with_range(10, size - 10),
            large_file.with_range(size - 10, size),
        ];
        let config = FileScanConfigBuilder::from(source.config().clone())
            .with_file_groups(vec![FileGroup::new(ranges), FileGroup::new(others)])
            .build();
        for (min_unit_rows, large_file_units) in [(1, 5), (2000, 3), (MIN_UNIT_ROWS, 2)] {
            let source = ParquetScanSource::try_from_config(config.clone())?
                .with_min_unit_rows(min_unit_rows);
            assert_eq!(
                units(&source, &ctx)?,
                2 + large_file_units,
                "{min_unit_rows}"
            );
            let output = Executor::collect(Arc::new(source), ctx.task_ctx(), 4).await?;
            assert_eq!(ids_and_years(&output), expected, "{min_unit_rows}");
        }

        // Without cached metadata, the files are split into byte ranges
        // (here they are too small to be split).
        let uncached = SessionContext::new_with_config(test_session_config());
        assert_eq!(units(&source, &uncached)?, 4);
        let output = Executor::collect(Arc::new(source), uncached.task_ctx(), 4).await?;
        assert_eq!(ids_and_years(&output), expected);
        Ok(())
    }

    #[test]
    fn byte_ranges_are_split_at_row_groups() {
        let file = PartitionedFile::new("file.parquet", 10_000);
        // The offsets (by which DataFusion assigns row groups to byte ranges)
        // and rows of the row groups of `file`, in any order.
        let row_groups = [(4000, 300), (4, 100), (2000, 200), (8000, 400)];
        let units = |file: &PartitionedFile, min_rows| {
            let mut units = vec![];
            push_row_group_units(&mut units, file, &row_groups, min_rows);
            units
                .iter()
                .map(|unit| unit.range.as_ref().map(|range| (range.start, range.end)))
                .collect::<Vec<_>>()
        };
        let every_row_group = vec![
            Some((0, 2000)),
            Some((2000, 4000)),
            Some((4000, 8000)),
            Some((8000, 10_000)),
        ];
        assert_eq!(units(&file, 0), every_row_group);
        assert_eq!(units(&file, 1), every_row_group);
        assert_eq!(units(&file, 100), every_row_group);
        // Consecutive row groups of at least `min_rows` rows.
        let first_two_together =
            vec![Some((0, 4000)), Some((4000, 8000)), Some((8000, 10_000))];
        assert_eq!(units(&file, 101), first_two_together);
        assert_eq!(units(&file, 300), first_two_together);
        assert_eq!(
            units(&file, 301),
            vec![Some((0, 8000)), Some((8000, 10_000))]
        );
        // A single unit: the file as it is.
        assert_eq!(units(&file, 601), vec![None]);
        // A byte range is split at the row groups that it reads.
        let range = file.clone().with_range(1000, 9000);
        assert_eq!(
            units(&range, 1),
            vec![Some((1000, 4000)), Some((4000, 8000)), Some((8000, 9000))]
        );
        assert_eq!(
            units(&file.clone().with_range(2000, 4001), 1),
            vec![Some((2000, 4000)), Some((4000, 4001))]
        );
        assert_eq!(
            units(&file.clone().with_range(3000, 5000), 1),
            vec![Some((3000, 5000))]
        );
        // A byte range that reads no row group has no unit.
        assert_eq!(units(&file.clone().with_range(4001, 8000), 1), vec![]);
        assert_eq!(units(&file.clone().with_range(0, 4), 1), vec![]);
    }

    /// A waker that counts how often it is woken.
    #[derive(Debug, Default)]
    struct CountingWaker(AtomicUsize);

    impl std::task::Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn idle_threads_take_the_chunks_of_readers() {
        let sharing = ChunkSharing::default();
        let wakes = Arc::new(CountingWaker::default());
        let waker = Waker::from(Arc::clone(&wakes));
        let woken = || wakes.0.load(Ordering::SeqCst);
        let chunk = |ids| DataChunk::from_record_batch(data_batch(ids));
        let taken = |result: SourceResult| match result {
            SourceResult::Chunk(chunk) => Some(chunk.len()),
            SourceResult::Blocked(_) => None,
            SourceResult::Finished => panic!("finished"),
        };
        sharing.add_reader();
        sharing.add_reader();
        // No thread waits: a reader keeps its chunk.
        assert_eq!(
            sharing.offer(chunk(0..10)).map(|chunk| chunk.len()),
            Some(10)
        );
        // An idle thread waits until a reader hands it a chunk, once.
        assert_eq!(taken(sharing.take(3, &waker)), None);
        assert_eq!(taken(sharing.take(3, &waker)), None);
        assert!(sharing.offer(chunk(10..30)).is_none());
        assert_eq!(woken(), 1);
        assert!(sharing.offer(chunk(30..70)).is_some());
        assert_eq!(taken(sharing.take(3, &waker)), Some(20));
        // The idle thread finishes once the last reader finished.
        assert_eq!(taken(sharing.take(3, &waker)), None);
        sharing.remove_reader();
        assert_eq!(woken(), 1);
        assert_eq!(taken(sharing.take(3, &waker)), None);
        sharing.remove_reader();
        assert_eq!(woken(), 2);
        assert!(matches!(sharing.take(3, &waker), SourceResult::Finished));
        // Without readers, chunks are kept.
        assert!(sharing.offer(chunk(0..1)).is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limits_stop_the_threads_that_wait_for_chunks() -> Result<()> {
        // The scan of `p` (with the pipeline engine) has three units, of 5000,
        // 100 and 7 rows: the threads that read the small ones wait for the
        // chunks of the thread that reads the large one, until the native
        // limit above the filter stops the pipeline.
        let dir = table();
        let ctx = pipeline_context(test_session_config());
        register_table(&ctx, dir.path()).await?;
        for (sql, rows) in [
            ("SELECT id FROM p WHERE a > -100 LIMIT 3000", 3000),
            ("SELECT id, s FROM p WHERE a > 40 OR s IS NULL LIMIT 5", 5),
            ("SELECT id FROM p WHERE a > -100", 4643),
        ] {
            for _ in 0..3 {
                let output =
                    tokio::time::timeout(Duration::from_secs(60), run_sql(&ctx, sql))
                        .await
                        .unwrap_or_else(|_| panic!("{sql} hangs"))?;
                let output_rows = output
                    .batches
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>();
                assert_eq!(output_rows, rows, "{sql}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_row_group_units() {
        // Every row group is a unit of its own, and the threads that run out
        // of units take the chunks of the threads that still read.
        let dir = table();
        let setup = async |ctx: &SessionContext| register_table(ctx, dir.path()).await;
        let lower: LowerScan = |plan| match plan.downcast_ref::<DataSourceExec>() {
            Some(exec) if exec.downcast_to_file_source::<ParquetSource>().is_some() => {
                let source = ParquetScanSource::try_new(exec)?.with_min_unit_rows(1);
                Ok(Some(Arc::new(source)))
            }
            _ => try_create_scan_source(plan),
        };
        let mut pushdown = test_session_config();
        pushdown.options_mut().execution.parquet.pushdown_filters = true;
        // DataFusion splits the large file into byte ranges that do not
        // follow its row groups.
        let mut repartitioned = test_session_config();
        repartitioned
            .options_mut()
            .optimizer
            .repartition_file_min_size = 1024;
        let mut midpoint = repartitioned.clone();
        midpoint
            .options_mut()
            .execution
            .parquet
            .row_group_range_assignment = RowGroupRangeAssignment::Midpoint;
        for config in [
            test_session_config(),
            pushdown,
            repartitioned,
            midpoint,
            test_session_config().with_target_partitions(16),
        ] {
            for sql in QUERIES {
                assert_native_scans_byte_eq_with(sql, config.clone(), &setup, lower)
                    .await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limit_applies_to_all_threads() -> Result<()> {
        let dir = table();
        let ctx = SessionContext::new_with_config(test_session_config());
        register_table(&ctx, dir.path()).await?;
        let scan = scan_source(&ctx, "SELECT * FROM p").await?;
        for limit in [0, 1, 25, 2047, 2049, 5106, 5107, 9000] {
            // The scan of all files, with a limit.
            let config = FileScanConfigBuilder::from(scan.config().clone())
                .with_limit(Some(limit))
                .build();
            let source: Arc<dyn PhysicalOperator> = Arc::new(
                ParquetScanSource::try_from_config(config)?.with_range_bytes(4096),
            );
            let sql = format!("SELECT * FROM p LIMIT {limit}");
            for threads in [1, 4] {
                let output =
                    Executor::collect(Arc::clone(&source), ctx.task_ctx(), threads)
                        .await?;
                let rows = ids_and_years(&output);
                assert_eq!(rows.len(), limit.min(5107), "{sql} on {threads} threads");
                // Every row is a row of the table.
                for (id, year) in rows {
                    assert_eq!(year, year_of(id), "{id}");
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_ordered_scans() {
        // A scan with an output ordering reads every file group in order on
        // a single thread: DataFusion relies on the order to skip the sort.
        let dir = table();
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet(
                "sorted",
                dir.path().join("year=2020").to_str().unwrap(),
                ParquetReadOptions::default()
                    .file_sort_order(vec![vec![col("id").sort(true, false)]]),
            )
            .await
        };
        let config = test_session_config().with_target_partitions(1);
        for sql in [
            "SELECT * FROM sorted ORDER BY id",
            "SELECT id, a FROM sorted ORDER BY id LIMIT 20 OFFSET 1990",
            "SELECT id, s FROM sorted WHERE a > 0 ORDER BY id",
        ] {
            assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup).await;
        }

        let ctx = SessionContext::new_with_config(config);
        setup(&ctx).await.unwrap();
        let source = scan_source(&ctx, "SELECT * FROM sorted ORDER BY id")
            .await
            .unwrap()
            .with_range_bytes(4096);
        assert!(source.config().preserves_file_order());
        let exec_ctx = ExecutionContext::new(ctx.task_ctx(), 0, 4);
        let gstate = source.get_global_source_state(&exec_ctx).unwrap();
        // A single file group, not split.
        assert_eq!(gstate.max_threads(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_scans_limit_every_file_group() -> Result<()> {
        // With several file groups that each preserve the order of their
        // files, the limit applies to every file group, like in DataFusion.
        let dir = tempfile::tempdir()?;
        for (name, ids) in [("a", 0..3000), ("b", 3000..3010), ("c", 3010..6000)] {
            write_parquet(
                &dir.path().join(format!("{name}.parquet")),
                data_schema(),
                &[data_batch(ids)],
                None,
            )?;
        }
        let ctx = SessionContext::new_with_config(test_session_config());
        ctx.register_parquet(
            "sorted",
            dir.path().to_str().unwrap(),
            ParquetReadOptions::default()
                .file_sort_order(vec![vec![col("id").sort(true, false)]]),
        )
        .await?;
        let source = scan_source(&ctx, "SELECT * FROM sorted ORDER BY id").await?;
        assert!(source.config().preserves_file_order());
        let mut files = source
            .config()
            .file_groups
            .iter()
            .flat_map(|group| group.iter().cloned())
            .collect::<Vec<_>>();
        files.sort_by(|a, b| a.object_meta.location.cmp(&b.object_meta.location));
        let [a, b, c] = files.try_into().unwrap();
        // The first group reads `b` (10 rows), then `a`.
        let config = FileScanConfigBuilder::from(source.config().clone())
            .with_file_groups(vec![FileGroup::new(vec![b, a]), FileGroup::new(vec![c])])
            .with_limit(Some(15))
            .build();
        let expected = (3000..3010)
            .chain(0..5)
            .chain(3010..3025)
            .collect::<Vec<_>>();
        let sorted_ids = |batches: &[RecordBatch]| {
            let mut ids = batches
                .iter()
                .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids
        };
        let mut sorted_expected = expected.clone();
        sorted_expected.sort_unstable();

        let source: Arc<dyn PhysicalOperator> =
            Arc::new(ParquetScanSource::try_from_config(config.clone())?);
        let exec_ctx = ExecutionContext::new(ctx.task_ctx(), 0, 4);
        assert_eq!(source.get_global_source_state(&exec_ctx)?.max_threads(), 2);
        for threads in [1, 4] {
            let output =
                Executor::collect(Arc::clone(&source), ctx.task_ctx(), threads).await?;
            assert_eq!(sorted_ids(&output), sorted_expected);
        }
        // DataFusion produces the same rows.
        let exec = DataSourceExec::from_data_source(config.clone());
        let output = datafusion_physical_plan::collect(exec, ctx.task_ctx()).await?;
        assert_eq!(sorted_ids(&output), sorted_expected);

        // A single file group is read in order on one thread.
        let config = FileScanConfigBuilder::from(config)
            .with_file_groups(vec![FileGroup::new(
                source
                    .downcast_ref::<ParquetScanSource>()
                    .unwrap()
                    .scan
                    .files
                    .clone(),
            )])
            .with_limit(Some(3002))
            .build();
        let source: Arc<dyn PhysicalOperator> =
            Arc::new(ParquetScanSource::try_from_config(config)?);
        let output = Executor::collect(source, ctx.task_ctx(), 4).await?;
        let in_order = output
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect::<Vec<_>>();
        let expected = (3000..3010).chain(0..2992).collect::<Vec<_>>();
        assert_eq!(in_order, expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_schema_evolution() {
        // Files with different schemas: the table schema coerces `a` from
        // INT to BIGINT and fills the column missing from the first file
        // with nulls.
        let dir = tempfile::tempdir().unwrap();
        let old_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("a", DataType::Int32, true),
        ]));
        let old = RecordBatch::try_new(
            Arc::clone(&old_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..3000)),
                Arc::new(Int32Array::from_iter(
                    (0..3000).map(|i| (i % 5 != 0).then_some(i)),
                )),
            ],
        )
        .unwrap();
        let new_schema = Arc::new(Schema::new(vec![
            Field::new("extra", DataType::Utf8, true),
            Field::new("id", DataType::Int64, false),
            Field::new("a", DataType::Int64, true),
        ]));
        let new = RecordBatch::try_new(
            Arc::clone(&new_schema),
            vec![
                Arc::new(StringArray::from_iter(
                    (3000..3500).map(|i| Some(format!("x{i}"))),
                )),
                Arc::new(Int64Array::from_iter_values(3000..3500)),
                Arc::new(Int64Array::from_iter_values((3000..3500).map(|i| i * 1000))),
            ],
        )
        .unwrap();
        write_parquet(&dir.path().join("old.parquet"), old_schema, &[old], None).unwrap();
        write_parquet(&dir.path().join("new.parquet"), new_schema, &[new], None).unwrap();
        let table_schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("a", DataType::Int64, true),
            Field::new("extra", DataType::Utf8, true),
        ]);
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet(
                "evolved",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default().schema(&table_schema),
            )
            .await
        };
        let mut pushdown = test_session_config();
        pushdown.options_mut().execution.parquet.pushdown_filters = true;
        for config in [test_session_config(), pushdown] {
            for sql in [
                "SELECT * FROM evolved",
                "SELECT extra, a FROM evolved WHERE a > 2990",
                "SELECT id FROM evolved WHERE extra IS NULL AND a < 100",
                "SELECT COUNT(*), SUM(a) FROM evolved WHERE extra IS NOT NULL",
            ] {
                assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup)
                    .await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scans_run_again_and_report_errors() -> Result<()> {
        let dir = table();
        let ctx = SessionContext::new_with_config(SessionConfig::new());
        register_table(&ctx, dir.path()).await?;
        let source: Arc<dyn PhysicalOperator> =
            Arc::new(scan_source(&ctx, "SELECT * FROM p").await?);
        let expected = (0..5107).map(|id| (id, year_of(id))).collect::<Vec<_>>();
        for _ in 0..2 {
            let output =
                Executor::collect(Arc::clone(&source), ctx.task_ctx(), 4).await?;
            assert_eq!(ids_and_years(&output), expected);
            assert!(output.iter().all(|b| b.num_rows() <= 2048));
            assert!(output.iter().all(|b| b.schema() == source.schema()));
        }
        assert!(
            source
                .description()
                .starts_with("ParquetScanSource: file_groups={"),
            "{}",
            source.description()
        );

        // A file that disappeared is an error, as in DataFusion.
        std::fs::remove_file(dir.path().join("year=2021/data.parquet"))?;
        let err = Executor::collect(Arc::clone(&source), ctx.task_ctx(), 4)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("data.parquet"), "{err}");
        let plan = ctx
            .sql("SELECT * FROM p")
            .await?
            .create_physical_plan()
            .await?;
        assert!(
            datafusion_physical_plan::collect(plan, ctx.task_ctx())
                .await
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn requires_a_parquet_scan() {
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(CsvSource::new(data_schema())),
        )
        .build();
        let err = ParquetScanSource::try_from_config(config.clone()).unwrap_err();
        assert!(err.to_string().contains("ParquetSource"), "{err}");
        let err = ParquetScanSource::try_new(&DataSourceExec::new(Arc::new(config)))
            .unwrap_err();
        assert!(err.to_string().contains("Parquet files"), "{err}");
    }
}
