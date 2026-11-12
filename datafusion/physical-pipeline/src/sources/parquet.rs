// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`ParquetScanSource`]: a parallel, morsel-driven scan of Parquet files,
//! which filters its rows with adaptive row filters and late
//! materialization.

use std::collections::VecDeque;
use std::fmt::{self, Write as _};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::task::Waker;

use arrow::array::{Array, AsArray, BooleanArray, RecordBatch};
use arrow::datatypes::{FieldRef, Schema, SchemaRef};
use arrow_buffer::BooleanBuffer;
use datafusion_common::instant::Instant;
use datafusion_common::parquet_config::RowGroupRangeAssignment;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_parquet::can_expr_be_pushed_down_with_schemas;
use datafusion_datasource_parquet::metadata::CachedParquetMetaData;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::cache::cache_manager::{
    CachedFileMetadataEntry, FileMetadataCache,
};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::expressions::{Column, DynamicFilterPhysicalExpr};
use datafusion_physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion_physical_expr::{PhysicalExpr, conjunction, split_conjunction};
use datafusion_physical_expr_common::physical_expr::{is_volatile, snapshot_generation};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::metrics::{Count, Gauge, MetricBuilder};
use parking_lot::Mutex;
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};

use super::{
    EmptyScanSource, FileScan, FileScanGlobalState, FileScanLocalState, FileScanMode,
    file_scan_config, push_file_ranges,
};
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk, SelectionVector};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSourceState, LocalSourceState,
    OperatorState, PhysicalOperator, SourceResult, TaskInterrupt, downcast_state,
    downcast_state_mut,
};
use crate::operators::adaptive_filter::{Fallibility, fallibility};
use crate::operators::filter::PhysicalFilter;

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
/// bloom filters, dynamic filters (e.g. from a TopK or a hash join), schema
/// adaptation and coercion, partition column values, limits, and the TopK
/// file order (`reorder_files`). Decoded batches are sliced into chunks of
/// at most [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows
/// without copying.
///
/// A scan that must preserve the order of its files
/// ([`FileScanConfig::preserves_file_order`], e.g. a scan with an output
/// ordering) reads every file group, in order, on a single thread instead,
/// and applies the limit to every file group, like DataFusion. See the
/// [module documentation](super) for the rows the source produces.
///
/// # Row filters and late materialization
///
/// Like DuckDB's scans, which apply the filters of a query while they read,
/// the scan filters its rows with
///
/// * the predicate of the `FilterExec` directly above it, which the planner
///   folds into it (see [`Self::try_with_filter`]): then the scan produces
///   the output of the `FilterExec`;
/// * the conjuncts of the predicate of its [`ParquetSource`]: the dynamic
///   filters of the TopKs and hash joins above it, and the filters that
///   DataFusion pushed into it to prune its files, row groups and pages.
///   No row has to be filtered by them (the nodes above apply them, or
///   prune the rows that they reject), so the scan only applies them where
///   that pays off; but with `pushdown_filters`, DataFusion relies on the
///   scan to apply its static conjuncts, which it then always does.
///
/// whatever `datafusion.execution.parquet.pushdown_filters` says. A conjunct
/// is evaluated either by the row filter of the Parquet decoder, which
/// decodes the columns of its conjuncts first, one conjunct after the other
/// on the rows that the conjuncts before it selected, and the other columns
/// of a row group only for the rows that pass all of them (late
/// materialization), or natively, on the decoded chunks. Which one pays off
/// depends on the data: the decoder only saves work by skipping long runs
/// of rows (it decodes the rows of short runs, which it then filters out),
/// and every conjunct in the row filter costs the decoder a pass over its
/// row groups. So the scan measures, on all of its threads, the rows that
/// every conjunct rejects, how long the runs of rows that it selects or
/// rejects are, and how long it takes, and evaluates it where it pays off
/// (see [`ConjunctMeter`]):
///
/// * in the row filter, if it rejects many rows in long runs;
/// * natively, after decoding, otherwise: the conjuncts of the folded
///   filter, and the optional conjuncts that reject enough rows (like
///   [`JoinFilteredScan`](crate::planner) does for memory scans);
/// * not at all, for the other optional conjuncts.
///
/// The row filter of a unit evaluates its conjuncts in decreasing order of
/// rank, the fraction of rows that a conjunct rejects per nanosecond that it
/// costs on a row (its measured evaluation, and the estimated decoding of
/// its columns), like DuckDB's adaptive filters order their conjuncts. The
/// threads take a new order, and new placements, whenever they claim a unit.
///
/// The scan also measures how long its units take with and without row
/// filters (per row), reading one in eight units in the configuration that
/// it does not prefer while it has at least two units per thread, and
/// switches to filtering natively after decoding if the row filters make it
/// slower (see [`UnitTimings`]).
///
/// The folded filter is always evaluated natively on the decoded chunks, by
/// a [`PhysicalFilter`], also on the rows that the row filter selected: the
/// row filter only removes rows, which cannot change the result, and
/// conjuncts that may fail on some rows (e.g. `10 / x > 1`, which a
/// conjunct `x <> 0` may guard) are never evaluated by the row filter.
#[derive(Debug)]
pub struct ParquetScanSource {
    scan: FileScan,
    /// See [`MIN_UNIT_ROWS`].
    min_unit_rows: usize,
    /// The conjuncts that the scan may filter its rows with.
    predicate: Arc<ScanPredicate>,
    /// The filter folded into the scan, see [`Self::try_with_filter`].
    filter: Option<Arc<PhysicalFilter>>,
}

impl ParquetScanSource {
    /// Create a source that scans the Parquet files of `plan`.
    ///
    /// Returns an error if `plan` does not scan Parquet files.
    pub fn try_new(plan: &DataSourceExec) -> Result<Self> {
        let config =
            file_scan_config::<ParquetSource>(plan, "ParquetScanSource", "Parquet")?;
        let scan = FileScan::new("ParquetScanSource", config.clone(), plan.schema());
        Self::try_from_scan(scan)
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
        Self::try_from_scan(FileScan::new("ParquetScanSource", config, schema))
    }

    fn try_from_scan(scan: FileScan) -> Result<Self> {
        let predicate = ScanPredicate::try_new(&scan.config, None)?;
        Ok(Self {
            scan,
            min_unit_rows: MIN_UNIT_ROWS,
            predicate: Arc::new(predicate),
            filter: None,
        })
    }

    /// Whether the scan `plan` can produce the output of a `FilterExec` above
    /// it, see [`Self::try_with_filter`]: a `DataSourceExec` that scans
    /// Parquet files without a limit.
    pub fn can_fold_filter_into(plan: &Arc<dyn ExecutionPlan>) -> bool {
        plan.downcast_ref::<DataSourceExec>()
            .and_then(DataSourceExec::downcast_to_file_source::<ParquetSource>)
            .is_some_and(|(config, _)| config.limit.is_none())
    }

    /// This scan with the `FilterExec` `filter` folded into it: the scan
    /// produces the output of `filter`, whose input must produce the rows of
    /// this scan (e.g. this scan, or a repartitioning of it). The scan
    /// evaluates the predicate of `filter` with its row filters, and always
    /// natively on the decoded chunks, and applies the projection of
    /// `filter` (see the [type documentation](Self)).
    ///
    /// Fails if `filter` has a fetch, if a filter is folded into the scan
    /// already, if the scan has a limit (which applies to the rows of the
    /// scan, not to those of the filter), or if the input of `filter` does
    /// not have the schema of the scan.
    pub fn try_with_filter(&self, filter: &FilterExec) -> Result<Self> {
        if filter.fetch().is_some()
            || self.filter.is_some()
            || self.scan.config.limit.is_some()
        {
            return internal_err!(
                "ParquetScanSource cannot fold a filter with a fetch, a second filter, or a filter above a limit"
            );
        }
        let input: Arc<dyn PhysicalOperator> =
            Arc::new(EmptyScanSource::new(self.schema()));
        let native = PhysicalFilter::try_from_filter_exec(filter, input)?;
        let predicate =
            ScanPredicate::try_new(&self.scan.config, Some(filter.predicate()))?;
        let mut scan = FileScan::new(
            self.scan.name,
            self.scan.config.clone(),
            Arc::clone(&self.scan.schema),
        );
        scan.range_bytes = self.scan.range_bytes;
        Ok(Self {
            scan,
            min_unit_rows: self.min_unit_rows,
            predicate: Arc::new(predicate),
            filter: Some(Arc::new(native)),
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

    /// The filter folded into the scan, if any (see
    /// [`Self::try_with_filter`]).
    pub fn filter(&self) -> Option<&PhysicalFilter> {
        self.filter.as_deref()
    }

    /// The units of work of an execution in shared mode: the files of the
    /// scan, the files whose metadata `cache` holds split at the boundaries
    /// of their row groups (see [`push_row_group_units`]), and the other
    /// files into byte ranges of at least `range_bytes`, as
    /// [`FileScan`] splits them. With the rows of every unit, if they are
    /// known (for the units of row groups).
    fn units(
        &self,
        cache: &FileMetadataCache,
        range_bytes: u64,
    ) -> (Vec<PartitionedFile>, Vec<Option<usize>>) {
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
        let mut rows = Vec::with_capacity(self.scan.files.len());
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
                    &mut rows,
                    file,
                    &row_groups,
                    self.min_unit_rows,
                ),
                None => {
                    push_file_ranges(&mut units, file, Some(range_bytes));
                    rows.resize(units.len(), None);
                }
            }
        }
        (units, rows)
    }

    /// The global state of an execution that shares the files between all
    /// threads (shared mode, without a limit) and splits them, by default:
    /// with the units of [`Self::units`], and their rows. `None` for the
    /// scans that [`FileScan`] splits into byte ranges of a set size, or does
    /// not split.
    fn row_group_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<(FileScanGlobalState, Vec<Option<usize>>)>> {
        let scan = &self.scan;
        let (None, Some(range_bytes)) = (scan.range_bytes, scan.range_bytes(ctx)) else {
            return Ok(None);
        };
        let task_ctx = &ctx.task_ctx;
        let runtime = task_ctx.runtime_env();
        let (files, rows) = self.units(
            runtime.cache_manager.get_file_metadata_cache().as_ref(),
            range_bytes,
        );
        let object_store = runtime.object_store(&scan.config.object_store_url)?;
        let batch_size = scan
            .config
            .batch_size
            .unwrap_or_else(|| task_ctx.session_config().batch_size());
        let state = FileScanGlobalState {
            units: (0..files.len()).map(|i| i..i + 1).collect(),
            files,
            next_unit: AtomicUsize::new(0),
            remaining: scan.config.limit.map(AtomicUsize::new),
            unit_limit: None,
            object_store,
            file_source: scan.config.file_source.with_batch_size(batch_size),
        };
        Ok(Some((state, rows)))
    }

    /// The next chunk of the units of this thread (not filtered by the
    /// folded filter yet), and whether its unit was read with a row filter.
    ///
    /// A scan with row filters claims its units itself, so that it can pick
    /// the row filter of every unit (see [`RowFilterState::claim_unit`]), and
    /// lets the [`FileScan`] read one unit at a time.
    fn read(
        &self,
        ctx: &ExecutionContext,
        gstate: &ParquetScanState,
        lstate: &mut ParquetScanLocalState,
    ) -> Result<(SourceResult, bool)> {
        let Some(row_filters) = &gstate.row_filters else {
            let result =
                self.scan
                    .get_data(ctx, gstate.scan.as_ref(), lstate.scan.as_mut())?;
            return Ok((result, false));
        };
        let queue =
            downcast_state::<FileScanGlobalState>(gstate.scan.as_any(), self.name())?;
        loop {
            if lstate.unit.is_none() {
                let Some(unit) = row_filters.claim_unit(self, ctx, queue, lstate)? else {
                    return Ok((SourceResult::Finished, false));
                };
                lstate.unit = Some(unit);
            }
            let Some(unit) = lstate.unit.as_mut() else {
                return internal_err!("ParquetScanSource lost its unit");
            };
            let start = Instant::now();
            let result = self.scan.get_data(ctx, &unit.state, lstate.scan.as_mut());
            unit.nanos += elapsed_nanos(start);
            match result? {
                SourceResult::Finished => {
                    if let Some(unit) = lstate.unit.take() {
                        row_filters.finish_unit(&unit);
                    }
                }
                SourceResult::Chunk(chunk) => {
                    if unit.row_filter {
                        row_filters.metrics.row_filter_output_rows.add(chunk.len());
                    }
                    return Ok((SourceResult::Chunk(chunk), unit.row_filter));
                }
                result => return Ok((result, false)),
            }
        }
    }

    /// Filter `chunk`, a decoded chunk of the scan (which a row filter
    /// filtered already if `row_filtered`), with the optional conjuncts that
    /// the scan evaluates natively, and with the folded filter (which also
    /// projects it).
    fn filter_chunk(
        &self,
        ctx: &ExecutionContext,
        gstate: &ParquetScanState,
        lstate: &mut ParquetScanLocalState,
        chunk: DataChunk,
        row_filtered: bool,
    ) -> Result<DataChunk> {
        let chunk = match &gstate.row_filters {
            Some(row_filters) if !chunk.is_empty() => {
                row_filters.filter_natively(&self.predicate, chunk, row_filtered)?
            }
            _ => chunk,
        };
        let (Some(filter), Some(filter_gstate), Some(filter_lstate)) = (
            self.filter.as_deref(),
            gstate.filter.as_deref(),
            lstate.filter.as_deref_mut(),
        ) else {
            return Ok(chunk);
        };
        let mut output = DataChunk::empty(self.schema());
        if !chunk.is_empty() {
            filter.execute(ctx, &chunk, &mut output, filter_gstate, filter_lstate)?;
        }
        Ok(output)
    }

    /// The next chunk of a thread that reads units (see [`Self::read`]),
    /// filtered (see [`Self::filter_chunk`]), with the time spent filtering
    /// it added to its unit.
    fn next_filtered(
        &self,
        ctx: &ExecutionContext,
        gstate: &ParquetScanState,
        lstate: &mut ParquetScanLocalState,
    ) -> Result<SourceResult> {
        Ok(match self.read(ctx, gstate, lstate)? {
            (SourceResult::Chunk(chunk), row_filtered) => {
                if gstate.row_filters.is_none() && self.filter.is_none() {
                    return Ok(SourceResult::Chunk(chunk));
                }
                let start = Instant::now();
                let chunk =
                    self.filter_chunk(ctx, gstate, lstate, chunk, row_filtered)?;
                if let Some(unit) = lstate.unit.as_mut() {
                    unit.nanos += elapsed_nanos(start);
                }
                SourceResult::Chunk(chunk)
            }
            (result, _) => result,
        })
    }
}

/// The nanoseconds since `start`.
fn elapsed_nanos(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
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

/// Push the units of `file` (a file or a byte range of a file) to `units`,
/// and their rows to `unit_rows`: the byte ranges of `file` that each
/// contain the offsets (see [`row_group_offset`]) of consecutive row groups
/// of at least `min_rows` rows (fewer at the end of `file`), of the row
/// groups `row_groups` of the file (offset and rows). The ranges cover
/// `file` exactly, so DataFusion reads every row group of `file` in exactly
/// one of them, whatever the row groups really are. A byte range that
/// contains no row group (e.g. the last bytes of a file, which DataFusion
/// assigns to a file group of their own) has no unit: it would read nothing.
fn push_row_group_units(
    units: &mut Vec<PartitionedFile>,
    unit_rows: &mut Vec<Option<usize>>,
    file: &PartitionedFile,
    row_groups: &[(i64, usize)],
    min_rows: usize,
) {
    let (start, end) = file.range();
    let (Ok(start), Ok(end)) = (i64::try_from(start), i64::try_from(end)) else {
        units.push(file.clone());
        unit_rows.push(None);
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
    // The offsets of the first row groups of the units after the first, and
    // the rows of every unit.
    let mut cuts = vec![];
    let mut rows_of_units = vec![];
    let mut rows = 0;
    for (offset, row_group_rows) in in_range {
        if rows >= min_rows.max(1) {
            cuts.push(offset);
            rows_of_units.push(rows);
            rows = 0;
        }
        rows += row_group_rows;
    }
    rows_of_units.push(rows);
    if cuts.is_empty() {
        units.push(file.clone());
        unit_rows.push(Some(rows));
        return;
    }
    let mut range_start = start;
    for (range_end, rows) in cuts.into_iter().chain([end]).zip(rows_of_units) {
        units.push(file.clone().with_range(range_start, range_end));
        unit_rows.push(Some(rows));
        range_start = range_end;
    }
}

impl PhysicalOperator for ParquetScanSource {
    fn name(&self) -> &str {
        self.scan.name
    }

    fn schema(&self) -> SchemaRef {
        match &self.filter {
            Some(filter) => filter.schema(),
            None => Arc::clone(&self.scan.schema),
        }
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        let description = self.scan.description();
        let Some(filter) = &self.filter else {
            return description;
        };
        let mut description = format!("{description}, filter={}", filter.predicate());
        if let Some(projection) = filter.projection() {
            let columns = projection
                .iter()
                .map(|&index| format!("{}@{index}", self.scan.schema.field(index).name()))
                .collect::<Vec<_>>();
            let _ = write!(description, ", projection=[{}]", columns.join(", "));
        }
        description
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
        let (scan, unit_rows): (Arc<dyn GlobalSourceState>, Vec<Option<usize>>) =
            match self.row_group_state(ctx)? {
                Some((state, rows)) => (Arc::new(state), rows),
                None => (self.scan.get_global_source_state(ctx)?, vec![]),
            };
        // The threads of an ordered scan read their file groups in order, and
        // a limit may end a scan before its units do.
        let shares =
            self.scan.mode == FileScanMode::Shared && self.scan.config.limit.is_none();
        let row_filters = match self.predicate.candidates.is_empty() {
            true => None,
            false => {
                let queue =
                    downcast_state::<FileScanGlobalState>(scan.as_any(), self.name())?;
                Some(RowFilterState::try_new(self, ctx, queue, unit_rows)?)
            }
        };
        let filter = self
            .filter
            .as_ref()
            .map(|filter| filter.get_global_operator_state(ctx))
            .transpose()?;
        Ok(Arc::new(ParquetScanState {
            scan,
            sharing: shares.then(Arc::default),
            row_filters,
            filter,
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
        let filter = match (&self.filter, &gstate.filter) {
            (Some(filter), Some(filter_gstate)) => {
                Some(filter.get_operator_state(ctx, filter_gstate.as_ref())?)
            }
            _ => None,
        };
        Ok(Box::new(ParquetScanLocalState {
            scan,
            sharing,
            unit: None,
            morselizer: None,
            filter,
        }))
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
            return self.next_filtered(ctx, gstate, lstate);
        };
        // A reader claims units, an idle thread takes the chunks of readers.
        while lstate.sharing.is_some() {
            match self.next_filtered(ctx, gstate, lstate)? {
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
/// [`FileScan`], whose units the threads claim, the chunks that they share
/// at the end of the scan, and the states of its row filters and of the
/// filter folded into it.
struct ParquetScanState {
    scan: Arc<dyn GlobalSourceState>,
    /// `None` for the scans that do not share their chunks: the ordered
    /// scans, and those with a limit.
    sharing: Option<Arc<ChunkSharing>>,
    /// `None` for the scans without conjuncts to filter their rows with
    /// (but the folded filter).
    row_filters: Option<RowFilterState>,
    /// The global state of the folded filter, if any.
    filter: Option<Arc<dyn GlobalOperatorState>>,
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
    /// With row filters: the unit that the thread reads.
    unit: Option<UnitRead>,
    /// With row filters: the version of the row filters (see
    /// [`ScanVersion`]) that the morselizer of the thread reads with.
    morselizer: Option<u64>,
    /// The state of the folded filter in this thread, if any.
    filter: Option<Box<dyn OperatorState>>,
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
/// shared chunks are slices of the batches that the readers decoded (and
/// filtered), at most one for every idle thread.
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

// ---------------------------------------------------------------------------
// Row filters
// ---------------------------------------------------------------------------

/// The rows that a conjunct is evaluated on before the scan decides where to
/// evaluate it, and between two decisions (see [`ConjunctMeter`]).
const SAMPLE_ROWS: usize = 8 * DEFAULT_VECTOR_SIZE;

/// A conjunct stays in the row filter if it rejects at least one in this
/// many of the rows that it is evaluated on...
const ROW_FILTER_MIN_REJECTED: usize = 4;

/// ...in runs of consecutive rows that it selects or rejects of at least this
/// many rows on average. The Parquet decoder skips the runs of rows that a
/// row filter rejects if they are this long on average; it decodes all rows
/// of shorter runs, which it then filters out (see arrow's
/// `RowSelectionPolicy`), which saves no decoding.
const ROW_FILTER_MIN_RUN_ROWS: usize = 32;

/// An optional conjunct is evaluated natively if it rejects at least one in
/// this many rows (as the planner's `JoinFilteredScan` does).
const NATIVE_MIN_REJECTED: usize = 8;

/// A conjunct that is not evaluated by the row filter is sampled in the row
/// filter again once this many rows passed it (and an optional dynamic
/// filter as soon as it changes).
const RESAMPLE_ROWS: usize = 256 * DEFAULT_VECTOR_SIZE;

/// The estimated cost of decoding a byte of a column (uncompressed), in
/// nanoseconds, for the ranks of the conjuncts.
const DECODE_NANOS_PER_BYTE: f64 = 0.25;

/// The bytes per row of a column whose size is unknown, for its decoding
/// cost.
const UNKNOWN_COLUMN_BYTES: usize = 8;

/// The weight of the statistics of the past windows of a conjunct, relative
/// to those of its last window, in its rank.
const RANK_DECAY: f64 = 0.5;

/// The units that the scan reads with and without row filters before it
/// compares them (see [`UnitTimings`]).
const SWITCH_MIN_UNITS: usize = 2;

/// The scan filters natively after decoding if that takes at most this
/// fraction of the time per row that its row filters take.
const SWITCH_MARGIN: f64 = 0.9;

/// The weight of the timings of the past units, relative to the last one.
const TIMING_DECAY: f64 = 0.75;

/// One in this many units is read in the configuration that the scan does
/// not prefer (with or without row filters) until it was timed often enough,
/// and one in four times as many after that...
const EXPLORE_STRIDE: usize = 8;

/// ...if the scan has at least this many units per thread, so that the
/// other threads absorb the extra time of these units.
const EXPLORE_MIN_UNITS_PER_THREAD: usize = 2;

/// Why the scan may evaluate a conjunct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConjunctKind {
    /// A conjunct of the folded filter: the scan evaluates it natively after
    /// decoding (with the folded filter), and in the row filter if that pays
    /// off.
    Folded,
    /// A conjunct of a scan with `pushdown_filters`, which DataFusion relies
    /// on the scan to apply: always in the row filter.
    Exact,
    /// A conjunct that no row has to be filtered by: a dynamic filter, or a
    /// conjunct that DataFusion pushed into the scan to prune it, which a
    /// node above the scan evaluates. Evaluated in the row filter or
    /// natively if that pays off, otherwise not at all.
    Optional,
}

/// A conjunct that a [`ParquetScanSource`] may evaluate in its row filter.
#[derive(Debug, Clone)]
struct RowFilterConjunct {
    /// The conjunct, over the table schema of the scan (its file and
    /// partition columns).
    expr: Arc<dyn PhysicalExpr>,
    /// An optional conjunct over the columns of the decoded chunks, to
    /// evaluate it natively, if they have all of its columns.
    native: Option<Arc<dyn PhysicalExpr>>,
    kind: ConjunctKind,
    /// The columns of the table schema that it reads.
    columns: Vec<usize>,
}

/// The conjuncts that a [`ParquetScanSource`] may filter its rows with: those
/// of the predicate of its [`ParquetSource`], and of the filter folded into
/// it.
#[derive(Debug, Default)]
struct ScanPredicate {
    /// The conjuncts that the row filter may evaluate: those that it can
    /// evaluate (see `can_expr_be_pushed_down_with_schemas`), that are
    /// neither volatile nor may fail on some rows (but `Exact` ones, which
    /// DataFusion's row filter evaluates anyway), in the order of the
    /// predicate, followed by those of the folded filter that the predicate
    /// does not have.
    candidates: Vec<RowFilterConjunct>,
    /// The other conjuncts of the predicate, which the row filter never
    /// evaluates: they only prune.
    pruning_only: Vec<Arc<dyn PhysicalExpr>>,
    /// Whether the row filter may reorder the candidates: not if an `Exact`
    /// conjunct may fail, which DataFusion evaluates in the order of the
    /// predicate.
    reorder: bool,
    /// The columns of the table schema that the projection of the scan
    /// reads.
    projected_columns: Vec<usize>,
}

impl ScanPredicate {
    /// The conjuncts of the predicate of the scan of `config`, and of
    /// `folded`, the predicate of a filter folded into it (over the output
    /// schema of the scan). None if the scan has a limit (which the decoder
    /// would apply to the rows of the row filter), or if a conjunct reads a
    /// virtual column (which DataFusion cannot read in a row filter).
    fn try_new(
        config: &FileScanConfig,
        folded: Option<&Arc<dyn PhysicalExpr>>,
    ) -> Result<Self> {
        let Some(source) = config.file_source().downcast_ref::<ParquetSource>() else {
            return Ok(Self::default());
        };
        let table = source.table_schema();
        let table_schema = table.table_schema();
        let projected_columns = match source.projection() {
            Some(projection) => {
                let mut columns = projection
                    .expr_iter()
                    .flat_map(|expr| collect_columns(&expr))
                    .map(|column| column.index())
                    .collect::<Vec<_>>();
                columns.sort_unstable();
                columns.dedup();
                columns
            }
            None => (0..table_schema.fields().len()).collect(),
        };
        let mut predicate = Self {
            reorder: true,
            projected_columns,
            ..Self::default()
        };
        if config.limit.is_some() {
            return Ok(predicate);
        }
        let pushable_schema = table.schema_without_virtual_columns();
        let exact = source.table_parquet_options().global.pushdown_filters;
        // The output column of every table column that the projection of the
        // scan produces as it is.
        let output_columns = output_columns(source, table_schema.fields().len());
        let output_schema = config.projected_schema()?;
        let conjuncts = source.filter();
        for conjunct in conjuncts.iter().flat_map(split_conjunction) {
            let dynamic = conjunct.is::<DynamicFilterPhysicalExpr>();
            let kind = match exact && !dynamic {
                true => ConjunctKind::Exact,
                false => ConjunctKind::Optional,
            };
            let infallible = dynamic
                || (!is_dynamic(conjunct)
                    && fallibility(conjunct, table_schema) == Fallibility::Infallible);
            if !can_expr_be_pushed_down_with_schemas(conjunct, pushable_schema)
                || is_volatile(conjunct)
                || (kind == ConjunctKind::Optional && !infallible)
            {
                predicate.pruning_only.push(Arc::clone(conjunct));
                continue;
            }
            if kind == ConjunctKind::Exact && !infallible {
                predicate.reorder = false;
            }
            predicate.candidates.push(RowFilterConjunct {
                expr: Arc::clone(conjunct),
                native: native_form(conjunct, &output_columns, &output_schema),
                kind,
                columns: column_indices(conjunct),
            });
        }
        for conjunct in folded.iter().flat_map(|folded| split_conjunction(folded)) {
            let expr = unproject(source, conjunct)?;
            if let Some(candidate) = predicate
                .candidates
                .iter_mut()
                .find(|candidate| candidate.expr.eq(&expr))
            {
                if candidate.kind == ConjunctKind::Optional {
                    candidate.kind = ConjunctKind::Folded;
                    candidate.native = None;
                }
                continue;
            }
            if predicate.pruning_only.contains(&expr)
                || !can_expr_be_pushed_down_with_schemas(&expr, pushable_schema)
                || is_volatile(&expr)
                || is_dynamic(&expr)
                || fallibility(&expr, table_schema) != Fallibility::Infallible
            {
                continue;
            }
            predicate.candidates.push(RowFilterConjunct {
                columns: column_indices(&expr),
                expr,
                native: None,
                kind: ConjunctKind::Folded,
            });
        }
        let virtual_columns = table
            .virtual_columns()
            .iter()
            .filter_map(|field| table_schema.index_of(field.name()).ok())
            .collect::<Vec<_>>();
        let reads_virtual_columns = predicate
            .candidates
            .iter()
            .flat_map(|candidate| &candidate.columns)
            .copied()
            .chain(predicate.pruning_only.iter().flat_map(column_indices))
            .any(|column| virtual_columns.contains(&column));
        if reads_virtual_columns {
            predicate.candidates.clear();
        }
        Ok(predicate)
    }

    /// The columns of the table schema that the projection reads and that
    /// none of the `in_row_filter` candidates read: the columns that the row
    /// filter materializes late.
    fn late_materialized_columns(&self, in_row_filter: &[usize]) -> usize {
        self.projected_columns
            .iter()
            .filter(|column| {
                !in_row_filter
                    .iter()
                    .any(|&candidate| self.candidates[candidate].columns.contains(column))
            })
            .count()
    }
}

/// For every column of the table schema (of `columns` columns) of `source`,
/// the output column of the scan that its projection produces it in, as it
/// is, if any.
fn output_columns(source: &ParquetSource, columns: usize) -> Vec<Option<usize>> {
    let mut outputs = vec![None; columns];
    match source.projection() {
        Some(projection) => {
            for (output, expr) in projection.expr_iter().enumerate() {
                if let Some(column) = expr.downcast_ref::<Column>()
                    && let Some(slot) = outputs.get_mut(column.index())
                {
                    slot.get_or_insert(output);
                }
            }
        }
        None => outputs
            .iter_mut()
            .enumerate()
            .for_each(|(i, slot)| *slot = Some(i)),
    }
    outputs
}

/// `conjunct`, over the table schema, over the output columns of the scan
/// (`output_schema`), if they have all of its columns (see
/// [`output_columns`]).
fn native_form(
    conjunct: &Arc<dyn PhysicalExpr>,
    output_columns: &[Option<usize>],
    output_schema: &Schema,
) -> Option<Arc<dyn PhysicalExpr>> {
    Arc::clone(conjunct)
        .transform(|expr| {
            let Some(column) = expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            match output_columns.get(column.index()).copied().flatten() {
                Some(output) => {
                    let name = output_schema.field(output).name();
                    Ok(Transformed::yes(Arc::new(Column::new(name, output)) as _))
                }
                None => internal_err!("column {column} is not an output of the scan"),
            }
        })
        .data()
        .ok()
}

/// `conjunct`, a conjunct of a filter above the scan of `source` (over its
/// output schema), over the table schema of the scan, as DataFusion pushes
/// filters into the scan (see `FileScanConfig::try_pushdown_filters`).
fn unproject(
    source: &ParquetSource,
    conjunct: &Arc<dyn PhysicalExpr>,
) -> Result<Arc<dyn PhysicalExpr>> {
    let unprojected = match source.projection() {
        Some(projection) => projection.unproject_expr(conjunct)?,
        None => Arc::clone(conjunct),
    };
    reassign_expr_columns(unprojected, source.table_schema().table_schema())
}

/// The indices of the columns that `expr` reads.
fn column_indices(expr: &Arc<dyn PhysicalExpr>) -> Vec<usize> {
    let mut columns = collect_columns(expr)
        .iter()
        .map(Column::index)
        .collect::<Vec<_>>();
    columns.sort_unstable();
    columns
}

/// Whether a part of `expr` is dynamic: it has a snapshot (see
/// `PhysicalExpr::snapshot`), which may hide what it evaluates.
fn is_dynamic(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.exists(|expr| Ok(!matches!(expr.snapshot(), Ok(None))))
        .unwrap_or(true)
}

/// The conjunct is evaluated by the row filter (also to sample it).
const IN_ROW_FILTER: u8 = 0;
/// The conjunct is evaluated natively after decoding: an optional one by the
/// scan, a folded one by the folded filter.
const NATIVELY: u8 = 1;
/// An optional conjunct that is not evaluated.
const NOT_EVALUATED: u8 = 2;

/// Where a [`ParquetScanSource`] evaluates one of its candidate conjuncts
/// (see [`ScanPredicate::candidates`]), and the statistics it decides that
/// from, shared by all threads of an execution.
///
/// Every conjunct starts in the row filter. Once it was evaluated on
/// [`SAMPLE_ROWS`] rows (by any threads), the scan decides from the rows it
/// rejected and from the runs of consecutive rows that it selected or
/// rejected where it evaluates it (see [`Self::decide`]), and again after
/// every [`SAMPLE_ROWS`] rows that it evaluates. A conjunct that is not
/// evaluated by the row filter any more is sampled in the row filter again
/// once [`RESAMPLE_ROWS`] rows passed the scan, or, for a dynamic filter, as
/// soon as it changes (e.g. the threshold of a TopK).
#[derive(Debug)]
struct ConjunctMeter {
    kind: ConjunctKind,
    /// Whether the scan can evaluate the conjunct natively: a folded
    /// conjunct, and an optional conjunct with a native form.
    native: bool,
    /// The conjunct, over the table schema, for the generation of its
    /// dynamic filters.
    expr: Arc<dyn PhysicalExpr>,
    /// [`IN_ROW_FILTER`], [`NATIVELY`] or [`NOT_EVALUATED`].
    placement: AtomicU8,
    /// The epoch of the row filters of the scan, shared by its meters: bumped
    /// whenever a meter decides, so that the threads build their row filters
    /// again (with new placements and orders) for their next units.
    epoch: Arc<AtomicU64>,
    window: Mutex<MeterWindow>,
    /// The rows that passed the conjunct unevaluated since its last decision.
    passed: AtomicUsize,
    /// The generation of the dynamic filters of the conjunct at its last
    /// decision.
    generation: AtomicU64,
    /// The `row_filter_evaluated_rows` metric of the scan.
    row_filter_rows: Count,
}

/// The statistics of a [`ConjunctMeter`].
#[derive(Debug, Default)]
struct MeterWindow {
    /// The rows that the conjunct was evaluated on since its last decision.
    rows: usize,
    /// The rows it rejected (also those for which it is `NULL`).
    rejected: usize,
    /// The runs of consecutive rows that it selected or rejected.
    runs: usize,
    /// The time it took.
    nanos: u64,
    /// The rows, rejected rows and nanoseconds of the past windows, which
    /// decay (see [`RANK_DECAY`]), for its rank.
    past_rows: f64,
    past_rejected: f64,
    past_nanos: f64,
}

impl ConjunctMeter {
    fn new(
        candidate: &RowFilterConjunct,
        epoch: Arc<AtomicU64>,
        row_filter_rows: Count,
    ) -> Self {
        Self {
            kind: candidate.kind,
            native: candidate.kind == ConjunctKind::Folded || candidate.native.is_some(),
            expr: Arc::clone(&candidate.expr),
            placement: AtomicU8::new(IN_ROW_FILTER),
            epoch,
            window: Mutex::default(),
            passed: AtomicUsize::new(0),
            generation: AtomicU64::new(snapshot_generation(&candidate.expr)),
            row_filter_rows,
        }
    }

    /// [`IN_ROW_FILTER`], [`NATIVELY`] or [`NOT_EVALUATED`].
    fn placement(&self) -> u8 {
        self.placement.load(Ordering::Relaxed)
    }

    /// Record that the conjunct selected `selected` of `rows` rows, in `runs`
    /// runs of consecutive rows that it selected or rejected, in `nanos`.
    fn record(&self, rows: usize, selected: usize, runs: usize, nanos: u64) {
        let mut window = self.window.lock();
        window.rows += rows;
        window.rejected += rows.saturating_sub(selected);
        window.runs += runs;
        window.nanos += nanos;
        if window.rows >= SAMPLE_ROWS {
            self.decide(&mut window);
        }
    }

    /// Record that `rows` rows passed the conjunct unevaluated: sample it in
    /// the row filter again after [`RESAMPLE_ROWS`] rows, or as soon as its
    /// dynamic filters change.
    fn record_passed(&self, rows: usize) {
        let passed = self.passed.fetch_add(rows, Ordering::Relaxed) + rows;
        let changed = self.kind == ConjunctKind::Optional
            && snapshot_generation(&self.expr) != self.generation.load(Ordering::Relaxed);
        if passed >= RESAMPLE_ROWS || changed {
            self.passed.store(0, Ordering::Relaxed);
            self.generation
                .store(snapshot_generation(&self.expr), Ordering::Relaxed);
            self.set_placement(IN_ROW_FILTER);
        }
    }

    /// Decide where to evaluate the conjunct from the statistics of its last
    /// window:
    ///
    /// * in the row filter if it rejects at least one in
    ///   [`ROW_FILTER_MIN_REJECTED`] rows, in runs of at least
    ///   [`ROW_FILTER_MIN_RUN_ROWS`] rows on average (and always if the scan
    ///   relies on its row filter, see [`ConjunctKind::Exact`]);
    /// * otherwise natively, if it is a folded conjunct, or an optional
    ///   conjunct that rejects at least one in [`NATIVE_MIN_REJECTED`] rows
    ///   and that the scan can evaluate natively;
    /// * otherwise not at all.
    fn decide(&self, window: &mut MeterWindow) {
        let rows = window.rows.max(1);
        let rejects_enough = window.rejected * ROW_FILTER_MIN_REJECTED >= rows;
        let long_runs = rows >= ROW_FILTER_MIN_RUN_ROWS * window.runs.max(1);
        let placement = match self.kind {
            ConjunctKind::Exact => IN_ROW_FILTER,
            _ if rejects_enough && long_runs => IN_ROW_FILTER,
            ConjunctKind::Folded => NATIVELY,
            ConjunctKind::Optional
                if self.native && window.rejected * NATIVE_MIN_REJECTED >= rows =>
            {
                NATIVELY
            }
            ConjunctKind::Optional => NOT_EVALUATED,
        };
        window.past_rows = window.past_rows * RANK_DECAY + window.rows as f64;
        window.past_rejected = window.past_rejected * RANK_DECAY + window.rejected as f64;
        window.past_nanos = window.past_nanos * RANK_DECAY + window.nanos as f64;
        window.rows = 0;
        window.rejected = 0;
        window.runs = 0;
        window.nanos = 0;
        self.passed.store(0, Ordering::Relaxed);
        self.generation
            .store(snapshot_generation(&self.expr), Ordering::Relaxed);
        self.set_placement(placement);
    }

    fn set_placement(&self, placement: u8) {
        self.placement.store(placement, Ordering::Relaxed);
        // Rebuild the row filters with the new placement and ranks.
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// The rank of the conjunct: the fraction of the rows that it rejects per
    /// nanosecond that it costs on a row, its evaluation and
    /// `decode_nanos`, the estimated cost of decoding its columns. The row
    /// filter evaluates its conjuncts by decreasing rank, which minimizes
    /// the cost of independent conjuncts (see
    /// [`ConjunctStats::rank`](crate::operators::adaptive_filter::ConjunctStats::rank)).
    /// A conjunct that was not evaluated yet is assumed to reject half of
    /// its rows in a nanosecond per row.
    fn rank(&self, decode_nanos: f64) -> f64 {
        let window = self.window.lock();
        let rows = window.past_rows + window.rows as f64;
        let (rejected, nanos) = match rows > 0.0 {
            true => (
                (window.past_rejected + window.rejected as f64) / rows,
                (window.past_nanos + window.nanos as f64) / rows,
            ),
            false => (0.5, 1.0),
        };
        rejected / (nanos + decode_nanos).max(1e-3)
    }
}

/// The selected rows of the result `value` of a conjunct on `rows` rows (a
/// `NULL` does not select its row), and the number of runs of consecutive
/// rows that it selects or rejects.
fn selection_stats(value: &ColumnarValue, rows: usize) -> Result<(usize, usize)> {
    let array = match value {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => return Ok((rows, 1)),
        ColumnarValue::Scalar(_) => return Ok((0, 1)),
        ColumnarValue::Array(array) => array,
    };
    let Some(array) = array.as_boolean_opt() else {
        return internal_err!("a filter returned a {} array", array.data_type());
    };
    let selected = selected_rows(array);
    Ok((selected.count_set_bits(), runs(&selected)))
}

/// The rows that `array` selects: those that are `true` (not `NULL`).
fn selected_rows(array: &BooleanArray) -> BooleanBuffer {
    match array.nulls().filter(|nulls| nulls.null_count() > 0) {
        Some(nulls) => array.values() & nulls.inner(),
        None => array.values().clone(),
    }
}

/// The number of runs of consecutive equal bits of `bits`.
fn runs(bits: &BooleanBuffer) -> usize {
    let len = bits.len();
    if len == 0 {
        return 0;
    }
    let set_runs = bits.set_slices().count();
    if set_runs == 0 {
        return 1;
    }
    let starts_set = bits.value(0);
    let ends_set = bits.value(len - 1);
    let unset_runs = set_runs + 1 - usize::from(starts_set) - usize::from(ends_set);
    set_runs + unset_runs
}

/// A conjunct of the predicate of a [`ParquetScanSource`] in the predicate of
/// the [`ParquetSource`] that it reads a unit with: DataFusion evaluates it
/// in the row filter of the decoder, and prunes with it.
///
/// It evaluates its conjunct only where its [`ConjunctMeter`] places it in
/// the row filter (and records the statistics of the meter), and selects
/// every row otherwise (then the scan evaluates it natively, or not at all).
/// A conjunct without a meter only prunes: the row filter never evaluates
/// it. Its snapshot (see [`PhysicalExpr::snapshot`]) is its conjunct, which
/// DataFusion prunes files, row groups and pages with (its pruning
/// predicates are built from snapshots), as without the row filter.
///
/// It claims to be volatile, so that DataFusion's simplifier does not
/// replace it by its value when its conjunct is a constant (e.g. after a
/// missing column is replaced by `NULL`): then it would not prune with the
/// conjunct any more.
#[derive(Debug)]
struct RowFilterExpr {
    expr: Arc<dyn PhysicalExpr>,
    meter: Option<Arc<ConjunctMeter>>,
}

impl RowFilterExpr {
    fn new_arc(
        expr: Arc<dyn PhysicalExpr>,
        meter: Option<Arc<ConjunctMeter>>,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(Self { expr, meter })
    }
}

impl PartialEq for RowFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.expr.eq(&other.expr)
            && match (&self.meter, &other.meter) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for RowFilterExpr {}

impl Hash for RowFilterExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.hash(state);
        self.meter
            .as_ref()
            .map(|meter| Arc::as_ptr(meter).addr())
            .hash(state);
    }
}

impl fmt::Display for RowFilterExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

impl PhysicalExpr for RowFilterExpr {
    fn return_field(&self, input_schema: &Schema) -> Result<FieldRef> {
        self.expr.return_field(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let rows = batch.num_rows();
        let Some(meter) = self
            .meter
            .as_ref()
            .filter(|meter| meter.placement() == IN_ROW_FILTER)
        else {
            return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))));
        };
        let start = Instant::now();
        let value = self.expr.evaluate(batch)?;
        let nanos = elapsed_nanos(start);
        let (selected, runs) = selection_stats(&value, rows)?;
        meter.record(rows, selected, runs, nanos);
        meter.row_filter_rows.add(rows);
        Ok(value)
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.expr]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let [expr] = <[_; 1]>::try_from(children).map_err(|children| {
            datafusion_common::internal_datafusion_err!(
                "RowFilterExpr has one child, got {}",
                children.len()
            )
        })?;
        Ok(Self::new_arc(expr, self.meter.clone()))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expr.fmt_sql(f)
    }

    fn snapshot(&self) -> Result<Option<Arc<dyn PhysicalExpr>>> {
        Ok(Some(Arc::clone(&self.expr)))
    }

    fn is_volatile_node(&self) -> bool {
        true
    }
}

/// The metrics of the row filters of a [`ParquetScanSource`], in the metrics
/// of its [`ParquetSource`] (those of the `DataSourceExec` it replaces).
struct RowFilterMetrics {
    /// `row_filter_evaluated_rows`: the rows that the conjuncts of the row
    /// filters were evaluated on, summed over the conjuncts. The decoder
    /// decodes the columns of a conjunct for these rows.
    row_filter_evaluated_rows: Count,
    /// `row_filter_output_rows`: the rows that the decoders of the units
    /// with row filters produced, the rows that passed their row filters:
    /// the columns that they materialize late are decoded for these rows
    /// only.
    row_filter_output_rows: Count,
    /// `late_materialized_columns`: the columns that the last row filter of
    /// the scan materializes late: the columns of the projection of the scan
    /// that none of its conjuncts reads.
    late_materialized_columns: Gauge,
    /// `row_filtered_units`: the units read with row filters.
    row_filtered_units: Count,
    /// `natively_filtered_units`: the units read without row filters, which
    /// the scan filters natively.
    natively_filtered_units: Count,
}

impl RowFilterMetrics {
    fn new(source: &dyn FileSource) -> Self {
        let builder = || MetricBuilder::new(source.metrics());
        Self {
            row_filter_evaluated_rows: builder()
                .global_counter("row_filter_evaluated_rows"),
            row_filter_output_rows: builder().global_counter("row_filter_output_rows"),
            late_materialized_columns: builder()
                .global_gauge("late_materialized_columns"),
            row_filtered_units: builder().global_counter("row_filtered_units"),
            natively_filtered_units: builder().global_counter("natively_filtered_units"),
        }
    }
}

/// The predicate and the configuration of the decoder that a thread reads a
/// unit with: the [`ParquetSource`] that it creates its morselizer with.
struct ScanVersion {
    id: u64,
    file_source: Arc<dyn FileSource>,
    /// Whether the decoder has a row filter.
    row_filter: bool,
}

/// The versions of the row filters of a scan.
struct ScanVersions {
    next_id: u64,
    /// The version with row filters, and the epoch of the meters that it was
    /// built in.
    row_filter: Option<(u64, Arc<ScanVersion>)>,
    /// The version without row filters.
    native: Arc<ScanVersion>,
}

/// The time per row of the units that a scan read with and without row
/// filters, which the scan prefers the faster of.
#[derive(Debug, Default)]
struct UnitTimings {
    row_filter: UnitTiming,
    native: UnitTiming,
}

#[derive(Debug, Default)]
struct UnitTiming {
    units: usize,
    /// The rows and nanoseconds of the units, which decay (see
    /// [`TIMING_DECAY`]).
    rows: f64,
    nanos: f64,
}

impl UnitTiming {
    fn record(&mut self, rows: usize, nanos: u64) {
        self.units += 1;
        self.rows = self.rows * TIMING_DECAY + rows as f64;
        self.nanos = self.nanos * TIMING_DECAY + nanos as f64;
    }

    fn nanos_per_row(&self) -> f64 {
        self.nanos / self.rows.max(1.0)
    }
}

impl UnitTimings {
    /// Whether the scan prefers row filters: unless both configurations were
    /// timed on [`SWITCH_MIN_UNITS`] units, and the units without row filters
    /// took at most [`SWITCH_MARGIN`] of the time per row of those with.
    fn prefers_row_filter(&self) -> bool {
        self.row_filter.units < SWITCH_MIN_UNITS
            || self.native.units < SWITCH_MIN_UNITS
            || self.native.nanos_per_row()
                > self.row_filter.nanos_per_row() * SWITCH_MARGIN
    }
}

/// The unit that a thread of a scan with row filters reads.
struct UnitRead {
    /// The state that the [`FileScan`] reads the unit with: of a single unit.
    state: FileScanGlobalState,
    /// The index of the unit.
    index: usize,
    /// Whether the unit is read with a row filter.
    row_filter: bool,
    /// The time spent reading and filtering it.
    nanos: u64,
}

/// The state of the row filters of an execution of a [`ParquetScanSource`]
/// with candidate conjuncts (see [`ScanPredicate::candidates`]).
struct RowFilterState {
    /// The meter of every candidate.
    meters: Vec<Arc<ConjunctMeter>>,
    /// See [`ConjunctMeter::epoch`].
    epoch: Arc<AtomicU64>,
    /// The estimated cost of decoding the columns of every candidate, in
    /// nanoseconds per row.
    decode_nanos: Vec<f64>,
    versions: Mutex<ScanVersions>,
    timings: Mutex<UnitTimings>,
    /// The rows of every unit, if known.
    unit_rows: Vec<Option<usize>>,
    /// Whether the scan may read units without row filters: not if it has
    /// conjuncts that DataFusion relies on its row filter for.
    native_allowed: bool,
    /// Whether the scan reads some units in the configuration that it does
    /// not prefer (see [`EXPLORE_MIN_UNITS_PER_THREAD`]).
    explores: bool,
    /// The batch size of the decoders.
    batch_size: usize,
    metrics: RowFilterMetrics,
}

impl RowFilterState {
    fn try_new(
        scan: &ParquetScanSource,
        ctx: &ExecutionContext,
        queue: &FileScanGlobalState,
        unit_rows: Vec<Option<usize>>,
    ) -> Result<Self> {
        let config = &scan.scan.config;
        let Some(source) = config.file_source().downcast_ref::<ParquetSource>() else {
            return internal_err!("ParquetScanSource requires a ParquetSource");
        };
        let predicate = &scan.predicate;
        let metrics = RowFilterMetrics::new(source);
        let epoch = Arc::new(AtomicU64::new(0));
        let meters = predicate
            .candidates
            .iter()
            .map(|candidate| {
                Arc::new(ConjunctMeter::new(
                    candidate,
                    Arc::clone(&epoch),
                    metrics.row_filter_evaluated_rows.clone(),
                ))
            })
            .collect::<Vec<_>>();
        let runtime = ctx.task_ctx.runtime_env();
        let cache = runtime.cache_manager.get_file_metadata_cache();
        let decode_nanos = decode_nanos(predicate, source, cache.as_ref(), &queue.files);
        let batch_size = config
            .batch_size
            .unwrap_or_else(|| ctx.task_ctx.session_config().batch_size());
        // The version without row filters prunes with all conjuncts (also
        // those of the folded filter).
        let mut native = source.clone();
        if let Some(predicate) = conjunction_opt(
            predicate
                .candidates
                .iter()
                .map(|candidate| Arc::clone(&candidate.expr))
                .chain(predicate.pruning_only.iter().cloned()),
        ) {
            native = native.with_predicate(predicate);
        }
        let native_allowed = predicate
            .candidates
            .iter()
            .all(|candidate| candidate.kind != ConjunctKind::Exact);
        let units = queue.units.len();
        Ok(Self {
            meters,
            epoch,
            decode_nanos,
            versions: Mutex::new(ScanVersions {
                next_id: 1,
                row_filter: None,
                native: Arc::new(ScanVersion {
                    id: 0,
                    file_source: native
                        .with_pushdown_filters(false)
                        .with_batch_size(batch_size),
                    row_filter: false,
                }),
            }),
            timings: Mutex::default(),
            unit_rows,
            native_allowed,
            explores: native_allowed
                && units >= EXPLORE_MIN_UNITS_PER_THREAD * ctx.num_threads.max(1),
            batch_size,
            metrics,
        })
    }

    /// Claim the next unit of `queue`, the units of the scan `scan`, for the
    /// thread of `lstate`, with the version of the row filters that it reads
    /// it with: if that version is not the one that the morselizer of the
    /// thread was created with, create the morselizer again. `None` if no
    /// unit is left.
    fn claim_unit(
        &self,
        scan: &ParquetScanSource,
        ctx: &ExecutionContext,
        queue: &FileScanGlobalState,
        lstate: &mut ParquetScanLocalState,
    ) -> Result<Option<UnitRead>> {
        let index = queue.next_unit.fetch_add(1, Ordering::Relaxed);
        let Some(files) = queue.units.get(index) else {
            return Ok(None);
        };
        let version = self.version(scan, index)?;
        if lstate.morselizer != Some(version.id) {
            let scan_state = downcast_state_mut::<FileScanLocalState>(
                lstate.scan.as_any_mut(),
                scan.name(),
            )?;
            scan_state.morselizer = version.file_source.create_morselizer(
                Arc::clone(&queue.object_store),
                &scan.scan.config,
                ctx.thread_id,
            )?;
            lstate.morselizer = Some(version.id);
        }
        let unit = match version.row_filter {
            true => &self.metrics.row_filtered_units,
            false => &self.metrics.natively_filtered_units,
        };
        unit.add(1);
        Ok(Some(UnitRead {
            state: FileScanGlobalState {
                files: queue.files[files.clone()].to_vec(),
                units: std::iter::once(0..files.len()).collect(),
                next_unit: AtomicUsize::new(0),
                remaining: None,
                unit_limit: queue.unit_limit,
                object_store: Arc::clone(&queue.object_store),
                file_source: Arc::clone(&version.file_source),
            },
            index,
            row_filter: version.row_filter,
            nanos: 0,
        }))
    }

    /// Record the time a thread spent on `unit`, which it finished reading.
    fn finish_unit(&self, unit: &UnitRead) {
        let Some(rows) = self.unit_rows.get(unit.index).copied().flatten() else {
            return;
        };
        if rows == 0 {
            return;
        }
        let mut timings = self.timings.lock();
        match unit.row_filter {
            true => timings.row_filter.record(rows, unit.nanos),
            false => timings.native.record(rows, unit.nanos),
        }
    }

    /// The version of the row filters that the unit `unit` (an index into the
    /// units of the scan) is read with: with the row filters of the
    /// conjuncts that the meters place in the row filter (built again
    /// whenever their epoch changed), unless none is, or the scan reads
    /// the unit without row filters (see [`Self::reads_with_row_filter`]).
    fn version(&self, scan: &ParquetScanSource, unit: usize) -> Result<Arc<ScanVersion>> {
        // The epoch before the placements: a version is built again if they
        // change after.
        let epoch = self.epoch.load(Ordering::Relaxed);
        let in_row_filter = (0..self.meters.len())
            .filter(|&candidate| self.meters[candidate].placement() == IN_ROW_FILTER)
            .collect::<Vec<_>>();
        if in_row_filter.is_empty() || !self.reads_with_row_filter(unit) {
            return Ok(Arc::clone(&self.versions.lock().native));
        }
        let mut versions = self.versions.lock();
        if let Some((built, version)) = &versions.row_filter
            && *built == epoch
        {
            return Ok(Arc::clone(version));
        }
        let id = versions.next_id;
        versions.next_id += 1;
        let version = Arc::new(self.row_filter_version(scan, id, in_row_filter)?);
        versions.row_filter = Some((epoch, Arc::clone(&version)));
        Ok(version)
    }

    /// Whether the scan reads the unit `unit` with row filters: if it
    /// prefers them (see [`UnitTimings::prefers_row_filter`]), but for one in
    /// [`EXPLORE_STRIDE`] units (one in four times as many once the
    /// configuration it does not prefer was timed often enough) of a scan
    /// that explores, which it reads in the other configuration.
    fn reads_with_row_filter(&self, unit: usize) -> bool {
        if !self.native_allowed {
            return true;
        }
        let timings = self.timings.lock();
        let preferred = timings.prefers_row_filter();
        if !self.explores {
            return preferred;
        }
        let other = match preferred {
            true => &timings.native,
            false => &timings.row_filter,
        };
        let stride = match other.units < SWITCH_MIN_UNITS {
            true => EXPLORE_STRIDE,
            false => 4 * EXPLORE_STRIDE,
        };
        preferred != (unit % stride == stride - 1)
    }

    /// The version `id` of the row filters, whose row filter evaluates the
    /// candidates `in_row_filter`, by decreasing rank (unless the scan may
    /// not reorder them), followed by the other candidates (which select
    /// every row, unless they are sampled again) and the conjuncts that only
    /// prune.
    fn row_filter_version(
        &self,
        scan: &ParquetScanSource,
        id: u64,
        mut in_row_filter: Vec<usize>,
    ) -> Result<ScanVersion> {
        let Some(source) = scan.config().file_source().downcast_ref::<ParquetSource>()
        else {
            return internal_err!("ParquetScanSource requires a ParquetSource");
        };
        let predicate = &scan.predicate;
        if predicate.reorder {
            let ranks = (0..self.meters.len())
                .map(|candidate| {
                    self.meters[candidate].rank(self.decode_nanos[candidate])
                })
                .collect::<Vec<_>>();
            // A stable sort: ties keep the order of the predicate.
            in_row_filter.sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]));
        }
        self.metrics
            .late_materialized_columns
            .set(predicate.late_materialized_columns(&in_row_filter));
        let others =
            (0..self.meters.len()).filter(|candidate| !in_row_filter.contains(candidate));
        let conjuncts = in_row_filter
            .iter()
            .copied()
            .chain(others)
            .map(|candidate| {
                RowFilterExpr::new_arc(
                    Arc::clone(&predicate.candidates[candidate].expr),
                    Some(Arc::clone(&self.meters[candidate])),
                )
            })
            .chain(
                predicate
                    .pruning_only
                    .iter()
                    .map(|conjunct| RowFilterExpr::new_arc(Arc::clone(conjunct), None)),
            );
        let file_source = source
            .with_predicate(conjunction(conjuncts))
            .with_pushdown_filters(true)
            .with_reorder_filters(false)
            .with_batch_size(self.batch_size);
        Ok(ScanVersion {
            id,
            file_source,
            row_filter: true,
        })
    }

    /// Filter `chunk` natively with the optional candidates that the meters
    /// place there (also with those that they place in the row filter, if
    /// the chunk was not `row_filtered`), and record their statistics; record
    /// the rows that passed the other candidates unevaluated.
    fn filter_natively(
        &self,
        predicate: &ScanPredicate,
        chunk: DataChunk,
        row_filtered: bool,
    ) -> Result<DataChunk> {
        let rows = chunk.len();
        let mut batch = None;
        let mut selected: Option<BooleanBuffer> = None;
        for (candidate, meter) in predicate.candidates.iter().zip(&self.meters) {
            let placement = meter.placement();
            let evaluated = match (&candidate.native, placement) {
                (Some(_), NATIVELY) => true,
                (Some(_), IN_ROW_FILTER) => !row_filtered,
                _ => false,
            };
            if !evaluated {
                if placement != IN_ROW_FILTER {
                    meter.record_passed(rows);
                }
                continue;
            }
            let Some(native) = &candidate.native else {
                continue;
            };
            let batch = match &batch {
                Some(batch) => batch,
                None => batch.insert(chunk.compact()?),
            };
            let start = Instant::now();
            let value = native.evaluate(batch)?;
            let nanos = elapsed_nanos(start);
            let mask = match value {
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                    BooleanBuffer::new_set(rows)
                }
                ColumnarValue::Scalar(_) => BooleanBuffer::new_unset(rows),
                ColumnarValue::Array(array) => match array.as_boolean_opt() {
                    Some(array) => selected_rows(array),
                    None => {
                        return internal_err!(
                            "a filter returned a {} array",
                            array.data_type()
                        );
                    }
                },
            };
            meter.record(rows, mask.count_set_bits(), runs(&mask), nanos);
            selected = Some(match selected {
                Some(selected) => &selected & &mask,
                None => mask,
            });
        }
        let (Some(batch), Some(selected)) = (batch, selected) else {
            return Ok(chunk);
        };
        match selected.count_set_bits() {
            count if count == rows => Ok(chunk),
            0 => Ok(DataChunk::empty(Arc::clone(chunk.schema()))),
            _ => DataChunk::from_record_batch(batch).with_selection(
                SelectionVector::from_boolean_mask(&BooleanArray::new(selected, None)),
            ),
        }
    }
}

/// The estimated cost of decoding the columns of every candidate of
/// `predicate`, in nanoseconds per row: [`DECODE_NANOS_PER_BYTE`] for every
/// byte of the columns (uncompressed) in the first of `files` whose metadata
/// `cache` holds, or of [`UNKNOWN_COLUMN_BYTES`] for a column of another
/// size. Partition columns are not decoded.
fn decode_nanos(
    predicate: &ScanPredicate,
    source: &ParquetSource,
    cache: &FileMetadataCache,
    files: &[PartitionedFile],
) -> Vec<f64> {
    let table_schema = source.table_schema().table_schema();
    let file_columns = source.table_schema().file_schema().fields().len();
    let metadata = files.iter().find_map(|file| {
        let entry = cache.get(&file.object_meta.location)?;
        let metadata = parquet_metadata(&entry, file)?;
        // The uncompressed bytes per row of every root column of the file.
        let rows = metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group.num_rows().max(0) as f64)
            .sum::<f64>();
        let mut bytes = std::collections::HashMap::<String, f64>::new();
        for row_group in metadata.row_groups() {
            for column in row_group.columns() {
                let root = column.column_path().parts().first().cloned();
                if let Some(root) = root {
                    *bytes.entry(root).or_default() +=
                        column.uncompressed_size().max(0) as f64;
                }
            }
        }
        Some(
            bytes
                .into_iter()
                .map(|(root, bytes)| (root, bytes / rows.max(1.0)))
                .collect::<std::collections::HashMap<_, _>>(),
        )
    });
    predicate
        .candidates
        .iter()
        .map(|candidate| {
            let bytes = candidate
                .columns
                .iter()
                .filter(|&&column| column < file_columns)
                .map(|&column| {
                    let field = table_schema.field(column);
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get(field.name()).copied())
                        .unwrap_or_else(|| {
                            field
                                .data_type()
                                .primitive_width()
                                .unwrap_or(UNKNOWN_COLUMN_BYTES)
                                as f64
                        })
                })
                .sum::<f64>();
            bytes * DECODE_NANOS_PER_BYTE
        })
        .collect()
}

/// [`conjunction`] of `exprs`, `None` if there are none.
fn conjunction_opt(
    exprs: impl IntoIterator<Item = Arc<dyn PhysicalExpr>>,
) -> Option<Arc<dyn PhysicalExpr>> {
    let mut exprs = exprs.into_iter().peekable();
    exprs.peek()?;
    Some(conjunction(exprs))
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

    use crate::bridge::PipelineRootExec;
    use crate::executor::Executor;
    use crate::explain::format_operator_tree;
    use crate::sources::test_harness::{
        LowerScan, assert_native_scans_byte_eq_with,
        assert_native_scans_byte_eq_with_config, assert_native_scans_byte_eq_with_setup,
        data_batch, data_schema, write_parquet,
    };
    use crate::sources::try_create_scan_source;
    use crate::test_utils::{
        compare_outputs, pipeline_context, register_test_tables, run_engines, run_sql,
        test_session_config,
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
        let units_and_rows = |file: &PartitionedFile, min_rows| {
            let mut units = vec![];
            let mut rows = vec![];
            push_row_group_units(&mut units, &mut rows, file, &row_groups, min_rows);
            assert_eq!(units.len(), rows.len());
            let units = units
                .iter()
                .map(|unit| unit.range.as_ref().map(|range| (range.start, range.end)))
                .collect::<Vec<_>>();
            (units, rows)
        };
        let units = |file: &PartitionedFile, min_rows| units_and_rows(file, min_rows).0;
        // The rows of the units are those of their row groups.
        let rows = |file: &PartitionedFile, min_rows| units_and_rows(file, min_rows).1;
        assert_eq!(
            rows(&file, 1),
            vec![Some(100), Some(200), Some(300), Some(400)]
        );
        assert_eq!(rows(&file, 101), vec![Some(300), Some(300), Some(400)]);
        assert_eq!(rows(&file, 601), vec![Some(1000)]);
        assert_eq!(
            rows(&file.clone().with_range(1000, 9000), 1),
            vec![Some(200), Some(300), Some(400)]
        );
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

    // -----------------------------------------------------------------------
    // Row filters and late materialization
    // -----------------------------------------------------------------------

    /// The rows of each file of [`write_big_table`], and of its row groups and
    /// data pages.
    const BIG_FILE_ROWS: i64 = 32_768;
    const BIG_ROW_GROUP_ROWS: usize = 8192;
    const BIG_PAGE_ROWS: usize = 1024;

    /// The schema of [`big_batch`].
    fn big_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("id", DataType::Int64, false),
            Field::new("a", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("c", DataType::Float64, true),
        ]))
    }

    /// The rows `ks` of the table `big`: `k` numbers the rows (so it is
    /// sorted), the other columns do not follow it (and have nulls).
    fn big_batch(ks: std::ops::Range<i64>) -> RecordBatch {
        let rows = 4 * BIG_FILE_ROWS;
        let k = Int64Array::from_iter_values(ks.clone());
        let id = Int64Array::from_iter_values(ks.clone().map(|k| (k * 7919) % rows));
        let a = Int32Array::from_iter(
            ks.clone()
                .map(|k| (k % 97 != 0).then_some(((k * 31) % 1009) as i32 - 504)),
        );
        let s = StringArray::from_iter(
            ks.clone()
                .map(|k| (k % 89 != 0).then(|| format!("v{}", (k * 13) % 101))),
        );
        let c =
            arrow::array::Float64Array::from_iter(ks.map(|k| {
                (k % 53 != 0).then_some(((k * 17) % 1000) as f64 / 10.0 - 50.0)
            }));
        RecordBatch::try_new(
            big_schema(),
            vec![
                Arc::new(k),
                Arc::new(id),
                Arc::new(a),
                Arc::new(s),
                Arc::new(c),
            ],
        )
        .unwrap()
    }

    /// Write the table `big` to `dir`: 4 files of [`BIG_FILE_ROWS`] rows, in
    /// row groups of [`BIG_ROW_GROUP_ROWS`] rows and data pages of
    /// [`BIG_PAGE_ROWS`] rows.
    fn write_big_table(dir: &Path) -> Result<()> {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(BIG_ROW_GROUP_ROWS))
            .set_data_page_row_count_limit(BIG_PAGE_ROWS)
            .set_write_batch_size(BIG_PAGE_ROWS)
            .build();
        for file in 0..4 {
            let start = file * BIG_FILE_ROWS;
            write_parquet(
                &dir.join(format!("{file}.parquet")),
                big_schema(),
                &[big_batch(start..start + BIG_FILE_ROWS)],
                Some(properties.clone()),
            )?;
        }
        Ok(())
    }

    /// A directory with the tables of [`register_table`] (`p`) and of
    /// [`write_big_table`] (`big`).
    fn tables() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_table(&dir.path().join("p")).unwrap();
        write_big_table(&dir.path().join("big")).unwrap();
        dir
    }

    /// Register the tables of [`tables`] in `dir`, and those of
    /// [`register_test_tables`].
    async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<()> {
        register_table(ctx, &dir.join("p")).await?;
        ctx.register_parquet(
            "big",
            dir.join("big").to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
    }

    /// Queries with filters directly above Parquet scans, which the planner
    /// folds into the scans.
    const FILTER_QUERIES: &[&str] = &[
        "SELECT id, s FROM p WHERE a > 40",
        "SELECT * FROM p WHERE a > 40 AND s LIKE 'value-1%'",
        "SELECT id FROM p WHERE a % 7 = 3 OR c < 0",
        "SELECT id, year FROM p WHERE year = 2021 AND a < 0",
        "SELECT id, d FROM p WHERE d > DATE '2022-06-01' AND id % 100 < 30",
        // May fail on some rows (but it does not): only evaluated natively.
        "SELECT id FROM p WHERE (CASE WHEN a <> 0 THEN 100 / a END) > 2",
        "SELECT id, c FROM p WHERE c < 0 OR c IS NULL OR isnan(c)",
        "SELECT id FROM p WHERE a IS NULL AND id > 10",
        "SELECT s, COUNT(*) AS n FROM p WHERE id BETWEEN 1000 AND 4000 GROUP BY s",
        "SELECT * FROM p WHERE year = 2023 AND a > 0",
        // Volatile: only evaluated natively.
        "SELECT id FROM p WHERE random() < 2.0 AND a > 10",
        "SELECT p.id, t2.name FROM p JOIN t2 ON p.a = t2.k WHERE p.c > 0",
        "SELECT p.id, t2.w FROM t2 JOIN p ON p.a = t2.k AND p.year = 2020 \
         WHERE t2.w > 100 AND p.id % 3 = 0",
        "SELECT id, a FROM p WHERE s <> 'value-3' ORDER BY a DESC NULLS LAST, id LIMIT 10",
        "SELECT id, ts FROM p WHERE id % 1000 < 50 ORDER BY ts, id LIMIT 7",
        // Rejects most rows in long runs: in the row filter.
        "SELECT id, s FROM big WHERE k % 8192 < 256",
        // Rejects half of the rows in short runs: natively.
        "SELECT id, s FROM big WHERE a % 2 = 0",
        "SELECT id, c FROM big WHERE a % 2 = 0 AND k % 8192 < 256",
        "SELECT k FROM big WHERE s LIKE '%7%' AND c > 0",
        "SELECT COUNT(*), SUM(a) FROM big WHERE c IS NULL OR c > 10",
        "SELECT big.id, t2.name FROM big JOIN t2 ON big.a = t2.k WHERE big.c < 20",
        "SELECT id, k FROM big WHERE s <> 'v3' ORDER BY k DESC LIMIT 5",
        "SELECT k, id FROM big WHERE a > 400 ORDER BY id LIMIT 12",
    ];

    /// Assert that `sql` has the same results on the pipeline engine as on
    /// DataFusion (in their canonical forms, see `compare_outputs`), in
    /// sessions configured with `config` and set up with `setup`. Returns the
    /// operator tree of the pipeline engine.
    async fn assert_same_results<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let (expected, actual) =
            run_engines(sql, config, setup).await.unwrap_or_else(|e| {
                panic!("Failed to run query on both engines: {e}\n{sql}")
            });
        let root = Arc::clone(
            actual
                .plan
                .downcast_ref::<PipelineRootExec>()
                .expect("a pipeline plan")
                .operator(),
        );
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!(
                "Results differ: {message}\nquery: {sql}\nlowered:\n{}",
                format_operator_tree(&root)
            );
        }
        root
    }

    /// The operators of the tree `root`.
    fn operators(root: &Arc<dyn PhysicalOperator>) -> Vec<&Arc<dyn PhysicalOperator>> {
        let mut operators = vec![];
        let mut stack = vec![root];
        while let Some(operator) = stack.pop() {
            operators.push(operator);
            stack.extend(operator.children());
        }
        operators
    }

    /// The Parquet scans of the operator tree `root`.
    fn parquet_scans(root: &Arc<dyn PhysicalOperator>) -> Vec<&ParquetScanSource> {
        operators(root)
            .into_iter()
            .filter_map(|operator| operator.downcast_ref::<ParquetScanSource>())
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn folded_filters_have_the_results_of_datafusion() {
        let dir = tables();
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        let mut pushdown = test_session_config();
        let parquet = &mut pushdown.options_mut().execution.parquet;
        parquet.pushdown_filters = true;
        parquet.reorder_filters = true;
        for (config, folds) in [
            (test_session_config(), true),
            // Small batches: the scans decide where to evaluate the conjuncts
            // many times in a unit.
            (test_session_config().with_batch_size(100), true),
            (test_session_config().with_target_partitions(1), true),
            (test_session_config().with_target_partitions(16), true),
            // DataFusion applies the filters in its scans, and removes the
            // filters above them (but those it cannot apply there).
            (pushdown, false),
        ] {
            for sql in FILTER_QUERIES {
                let root = assert_same_results(sql, config.clone(), &setup).await;
                if !folds {
                    continue;
                }
                // No native filter is left directly above a Parquet scan:
                // the filters are folded into the scans.
                let folded = parquet_scans(&root)
                    .iter()
                    .filter(|scan| scan.filter().is_some())
                    .count();
                assert!(folded > 0, "{sql}\n{}", format_operator_tree(&root));
                for operator in operators(&root) {
                    if operator.is::<PhysicalFilter>() {
                        assert!(
                            !operator.children()[0].is::<ParquetScanSource>(),
                            "{sql}\n{}",
                            format_operator_tree(&root)
                        );
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_folded_filters() {
        // The scans with the filters above them folded into them, and the
        // other nodes in DataFusion: exactly the same rows as DataFusion.
        let dir = tables();
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        let fold: LowerScan = |plan| {
            let Some(filter) = plan.downcast_ref::<FilterExec>() else {
                return Ok(None);
            };
            let scan = scan_below(filter);
            if filter.fetch().is_some() || !ParquetScanSource::can_fold_filter_into(scan)
            {
                return Ok(None);
            }
            let exec = scan.downcast_ref::<DataSourceExec>().unwrap();
            // Every row group is a unit of its own.
            let source = ParquetScanSource::try_new(exec)?
                .with_min_unit_rows(1)
                .try_with_filter(filter)?;
            Ok(Some(Arc::new(source)))
        };
        for config in [
            test_session_config(),
            test_session_config().with_batch_size(100),
            test_session_config().with_target_partitions(16),
        ] {
            for sql in FILTER_QUERIES {
                assert_native_scans_byte_eq_with(sql, config.clone(), &setup, fold).await;
            }
        }
    }

    /// The value of the metric `name` of the [`ParquetSource`] of `scan`,
    /// summed over its executions.
    fn metric(scan: &ParquetScanSource, name: &str) -> usize {
        scan.config()
            .file_source()
            .metrics()
            .clone_inner()
            .sum_by_name(name)
            .map_or(0, |value| value.as_usize())
    }

    /// The value of the metric `name` of the `DataSourceExec`s of `plan`.
    fn plan_metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        let mut sum = 0;
        plan.apply(|node| {
            if node.is::<DataSourceExec>()
                && let Some(value) =
                    node.metrics().and_then(|metrics| metrics.sum_by_name(name))
            {
                sum += value.as_usize();
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        sum
    }

    /// Write a single file of `rows` rows of `big` to `dir`, in a single row
    /// group (so that the scan has a single unit), with data pages of
    /// [`BIG_PAGE_ROWS`] rows.
    fn write_single_unit_table(dir: &Path, rows: i64) -> Result<()> {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows as usize))
            .set_data_page_row_count_limit(BIG_PAGE_ROWS)
            .set_write_batch_size(BIG_PAGE_ROWS)
            .build();
        write_parquet(
            &dir.join("data.parquet"),
            big_schema(),
            &[big_batch(0..rows)],
            Some(properties),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn late_materialization_shows_in_the_metrics() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let rows = 65_536;
        write_single_unit_table(dir.path(), rows)?;
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet(
                "t",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await
        };
        // `k % 8192 < 256` rejects 31 in 32 rows, in long runs: the row filter
        // decodes `k` for every row, and the other columns of the projection
        // (`id` and `s`) only for the rows that pass it. DataFusion (without
        // `pushdown_filters`) decodes every column of every row, and filters
        // the rows above its scan.
        let sql = "SELECT id, s FROM t WHERE k % 8192 < 256";
        let (expected, actual) = run_engines(sql, test_session_config(), &setup).await?;
        compare_outputs(&expected, &actual).unwrap();
        let root = actual
            .plan
            .downcast_ref::<PipelineRootExec>()
            .unwrap()
            .operator();
        let [scan] = parquet_scans(root)[..] else {
            panic!("one scan\n{}", format_operator_tree(root));
        };
        assert!(scan.filter().is_some());
        let selected = (rows / 32) as usize;
        assert_eq!(metric(scan, "row_filter_evaluated_rows"), rows as usize);
        assert_eq!(metric(scan, "row_filter_output_rows"), selected);
        assert_eq!(metric(scan, "late_materialized_columns"), 2);
        assert_eq!(metric(scan, "row_filtered_units"), 1);
        assert_eq!(metric(scan, "natively_filtered_units"), 0);
        // DataFusion's metrics of the row filter.
        assert_eq!(metric(scan, "pushdown_rows_matched"), selected);
        assert_eq!(
            metric(scan, "pushdown_rows_pruned"),
            rows as usize - selected
        );
        // No more bytes than DataFusion (which reads every row of every
        // column).
        let bytes = metric(scan, "bytes_scanned");
        let datafusion_bytes = plan_metric(&expected.plan, "bytes_scanned");
        assert!(
            bytes <= datafusion_bytes,
            "{bytes} vs {datafusion_bytes} bytes scanned"
        );

        // `a % 2 = 0` rejects half of the rows in runs of about two rows: the
        // decoder would decode all rows of the other columns anyway. The scan
        // evaluates it in the row filter for a sample of rows, then natively
        // after decoding.
        let sql = "SELECT id, s FROM t WHERE a % 2 = 0";
        let (expected, actual) = run_engines(sql, test_session_config(), &setup).await?;
        compare_outputs(&expected, &actual).unwrap();
        let root = actual
            .plan
            .downcast_ref::<PipelineRootExec>()
            .unwrap()
            .operator();
        let [scan] = parquet_scans(root)[..] else {
            panic!("one scan\n{}", format_operator_tree(root));
        };
        let evaluated = metric(scan, "row_filter_evaluated_rows");
        assert!(
            (SAMPLE_ROWS..rows as usize / 2).contains(&evaluated),
            "{evaluated} rows evaluated by the row filter"
        );
        assert!(metric(scan, "row_filter_output_rows") > rows as usize / 2);
        Ok(())
    }

    /// A candidate conjunct of `kind` for the meters of the tests.
    fn candidate(kind: ConjunctKind, native: bool) -> RowFilterConjunct {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("k", 0));
        RowFilterConjunct {
            native: native.then(|| Arc::clone(&expr)),
            expr,
            kind,
            columns: vec![0],
        }
    }

    #[test]
    fn conjuncts_are_placed_where_they_pay_off() {
        let epoch = Arc::new(AtomicU64::new(0));
        let meter = |kind, native| {
            ConjunctMeter::new(&candidate(kind, native), Arc::clone(&epoch), Count::new())
        };
        let rows = SAMPLE_ROWS;
        // Rejects 31 in 32 rows, in 32 runs: in the row filter.
        let folded = meter(ConjunctKind::Folded, false);
        folded.record(rows, rows / 32, 32, 1000);
        assert_eq!(folded.placement(), IN_ROW_FILTER);
        // The same rows in short runs: natively.
        folded.record(rows, rows / 32, rows / 16, 1000);
        assert_eq!(folded.placement(), NATIVELY);
        // ...until enough rows passed it to sample it again.
        folded.record_passed(RESAMPLE_ROWS - 1);
        assert_eq!(folded.placement(), NATIVELY);
        folded.record_passed(1);
        assert_eq!(folded.placement(), IN_ROW_FILTER);
        // A decision needs a full window.
        folded.record(rows - 1, 0, 1, 1000);
        assert_eq!(folded.placement(), IN_ROW_FILTER);
        folded.record(1, 1, 1, 1);
        assert_eq!(folded.placement(), IN_ROW_FILTER);
        // Rejects few rows: natively.
        folded.record(rows, rows - 10, 20, 1000);
        assert_eq!(folded.placement(), NATIVELY);

        // An optional conjunct that rejects many rows in short runs is
        // evaluated natively, if it can be; it is not evaluated if it rejects
        // few rows.
        let optional = meter(ConjunctKind::Optional, true);
        optional.record(rows, rows / 4, rows / 2, 1000);
        assert_eq!(optional.placement(), NATIVELY);
        optional.record(rows, rows - rows / 16, rows / 8, 1000);
        assert_eq!(optional.placement(), NOT_EVALUATED);
        let without_native_form = meter(ConjunctKind::Optional, false);
        without_native_form.record(rows, rows / 4, rows / 2, 1000);
        assert_eq!(without_native_form.placement(), NOT_EVALUATED);
        // With long runs, also an optional conjunct is in the row filter.
        without_native_form.record_passed(RESAMPLE_ROWS);
        assert_eq!(without_native_form.placement(), IN_ROW_FILTER);
        without_native_form.record(rows, 0, 1, 1000);
        assert_eq!(without_native_form.placement(), IN_ROW_FILTER);

        // A conjunct that DataFusion relies on the row filter for stays there.
        let exact = meter(ConjunctKind::Exact, false);
        exact.record(rows, rows, 1, 1000);
        assert_eq!(exact.placement(), IN_ROW_FILTER);

        // Every decision bumps the epoch, so that the threads build their row
        // filters again.
        assert!(epoch.load(Ordering::Relaxed) >= 10);
    }

    #[test]
    fn selections_count_their_runs() {
        let bits = |values: &[bool]| BooleanBuffer::from(values.to_vec());
        assert_eq!(runs(&bits(&[])), 0);
        assert_eq!(runs(&bits(&[false, false])), 1);
        assert_eq!(runs(&bits(&[true, true, true])), 1);
        assert_eq!(runs(&bits(&[true, false, true])), 3);
        assert_eq!(runs(&bits(&[false, true, true, false])), 3);
        assert_eq!(runs(&bits(&[false, true, false, true])), 4);
        let array = BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]);
        let value = ColumnarValue::Array(Arc::new(array));
        // A `NULL` does not select its row.
        assert_eq!(selection_stats(&value, 4).unwrap(), (2, 4));
        let all = ColumnarValue::Scalar(ScalarValue::Boolean(Some(true)));
        assert_eq!(selection_stats(&all, 7).unwrap(), (7, 1));
        let none = ColumnarValue::Scalar(ScalarValue::Boolean(None));
        assert_eq!(selection_stats(&none, 7).unwrap(), (0, 1));
    }

    /// The global state of an execution of `source` on `threads` threads.
    fn execution_state(
        source: &ParquetScanSource,
        ctx: &SessionContext,
        threads: usize,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let exec_ctx = ExecutionContext::new(ctx.task_ctx(), 0, threads);
        source.get_global_source_state(&exec_ctx)
    }

    /// The scan of the plan of `sql`, with the filter of the plan folded
    /// into it.
    async fn folded_scan(ctx: &SessionContext, sql: &str) -> Result<ParquetScanSource> {
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let mut filter = None;
        plan.apply(|node| {
            if node.is::<FilterExec>() {
                filter = Some(Arc::clone(node));
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        let filter = filter.expect("a filter");
        let filter = filter.downcast_ref::<FilterExec>().unwrap();
        let exec = scan_below(filter).downcast_ref::<DataSourceExec>().unwrap();
        ParquetScanSource::try_new(exec)?.try_with_filter(filter)
    }

    /// The input of `filter`, below the repartitionings of its rows.
    fn scan_below(filter: &FilterExec) -> &Arc<dyn ExecutionPlan> {
        let mut input = filter.input();
        while let Some(repartition) =
            input.downcast_ref::<datafusion_physical_plan::repartition::RepartitionExec>()
        {
            input = repartition.input();
        }
        input
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn row_filters_order_their_conjuncts_by_rank() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_big_table(dir.path())?;
        let ctx = SessionContext::new_with_config(test_session_config());
        ctx.register_parquet("big", dir.path().to_str().unwrap(), Default::default())
            .await?;
        let scan = folded_scan(
            &ctx,
            "SELECT id FROM big WHERE a % 2 = 0 AND k % 8192 < 256 AND s <> 'x'",
        )
        .await?;
        let conjuncts = scan
            .predicate
            .candidates
            .iter()
            .map(|candidate| candidate.expr.to_string())
            .collect::<Vec<_>>();
        assert_eq!(conjuncts.len(), 3, "{conjuncts:?}");
        let gstate = execution_state(&scan, &ctx, 1)?;
        let gstate = downcast_state::<ParquetScanState>(gstate.as_any(), "test")?;
        let row_filters = gstate.row_filters.as_ref().unwrap();
        // The conjuncts of the row filter of a version, in order.
        let order = |version: &ScanVersion| {
            let predicate = version.file_source.filter().unwrap();
            split_conjunction(&predicate)
                .into_iter()
                .map(|conjunct| {
                    let conjunct = conjunct.downcast_ref::<RowFilterExpr>().unwrap();
                    conjuncts
                        .iter()
                        .position(|expr| *expr == conjunct.expr.to_string())
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        // Without statistics, by the estimated costs of decoding their
        // columns (from the sizes of the columns in the files).
        let version = row_filters.version(&scan, 0)?;
        assert!(version.row_filter);
        let mut by_cost = vec![0, 1, 2];
        by_cost.sort_by(|&a, &b| {
            row_filters.decode_nanos[a].total_cmp(&row_filters.decode_nanos[b])
        });
        assert_eq!(order(&version), by_cost, "{:?}", row_filters.decode_nanos);
        // Once the meters decided (which builds the row filters again):
        // `a % 2 = 0` rejects 3 in 10 of its rows, and `k % 8192 < 256` all but
        // one in 32, both in long runs: the row filter evaluates them by
        // decreasing rank. `s <> 'x'` rejects no row: it is evaluated natively,
        // after the conjuncts of the row filter (where it selects every row).
        let rows = SAMPLE_ROWS;
        row_filters.meters[0].record(rows, rows * 7 / 10, 16, 2000);
        row_filters.meters[1].record(rows, rows / 32, 16, 2000);
        row_filters.meters[2].record(rows, rows, 1, 8000);
        assert_eq!(row_filters.meters[2].placement(), NATIVELY);
        let rank = |candidate: usize| {
            row_filters.meters[candidate].rank(row_filters.decode_nanos[candidate])
        };
        let mut by_rank = vec![0, 1];
        by_rank.sort_by(|&a, &b| rank(b).total_cmp(&rank(a)));
        by_rank.push(2);
        let version = row_filters.version(&scan, 1)?;
        assert_eq!(order(&version), by_rank);
        // The rows rejected per nanosecond.
        assert!(rank(1) > rank(2));
        assert_eq!(rank(2), 0.0);
        // `k % 8192 < 256` rejects its rows in short runs: natively too.
        row_filters.meters[1].record(rows, rows / 32, rows / 4, 2000);
        assert_eq!(row_filters.meters[1].placement(), NATIVELY);
        let version = row_filters.version(&scan, 2)?;
        assert_eq!(order(&version), [0, 1, 2]);
        // Without a conjunct in the row filter, the units are read without.
        row_filters.meters[0].record(rows, rows / 2, rows / 2, 2000);
        assert_eq!(row_filters.meters[0].placement(), NATIVELY);
        assert!(!row_filters.version(&scan, 3)?.row_filter);
        Ok(())
    }

    #[test]
    fn ranks_favor_selective_and_cheap_conjuncts() {
        let epoch = Arc::new(AtomicU64::new(0));
        let meter = |rejected: usize, nanos: u64| {
            let meter = ConjunctMeter::new(
                &candidate(ConjunctKind::Folded, false),
                Arc::clone(&epoch),
                Count::new(),
            );
            meter.record(1000, 1000 - rejected, 1, nanos);
            meter
        };
        // Rejects 90% of its rows, in a nanosecond per row.
        let selective = meter(900, 1000);
        assert!(selective.rank(0.0) > meter(500, 1000).rank(0.0));
        assert!(selective.rank(0.0) > meter(900, 10_000).rank(0.0));
        // Decoding its columns costs too.
        assert!(selective.rank(0.0) > selective.rank(10.0));
        assert!((selective.rank(1.0) - 0.45).abs() < 1e-9);
        // Without statistics, half of the rows in a nanosecond per row.
        let unknown = ConjunctMeter::new(
            &candidate(ConjunctKind::Folded, false),
            Arc::clone(&epoch),
            Count::new(),
        );
        assert!((unknown.rank(1.0) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn scans_prefer_the_faster_configuration() {
        let mut timings = UnitTimings::default();
        assert!(timings.prefers_row_filter());
        timings.row_filter.record(1000, 10_000);
        timings.native.record(1000, 5000);
        // Not before both configurations were timed often enough.
        assert!(timings.prefers_row_filter());
        timings.row_filter.record(1000, 10_000);
        timings.native.record(1000, 5000);
        assert!(!timings.prefers_row_filter());
        // Natively only if that is faster by a margin.
        let mut close = UnitTimings::default();
        for _ in 0..SWITCH_MIN_UNITS {
            close.row_filter.record(1000, 10_000);
            close.native.record(1000, 9500);
        }
        assert!(close.prefers_row_filter());
        // The timings of the last units weigh more.
        for _ in 0..8 {
            timings.native.record(1000, 20_000);
        }
        assert!(timings.prefers_row_filter());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scans_read_some_units_without_row_filters() -> Result<()> {
        // With 16 units (every row group of `big`) for 4 threads, the scan
        // times units with and without row filters: one in eight is read in
        // the configuration that it does not prefer.
        let dir = tempfile::tempdir()?;
        write_big_table(dir.path())?;
        let config = test_session_config();
        let ctx = SessionContext::new_with_config(config.clone());
        ctx.register_parquet("big", dir.path().to_str().unwrap(), Default::default())
            .await?;
        let sql = "SELECT k, id FROM big WHERE k % 8192 < 256";
        let scan = Arc::new(folded_scan(&ctx, sql).await?.with_min_unit_rows(1));
        let gstate = execution_state(&scan, &ctx, 4)?;
        let gstate = downcast_state::<ParquetScanState>(gstate.as_any(), "test")?;
        let row_filters = gstate.row_filters.as_ref().unwrap();
        assert!(row_filters.explores);
        let schedule = (0..16)
            .map(|unit| row_filters.reads_with_row_filter(unit))
            .collect::<Vec<_>>();
        assert_eq!(
            schedule.iter().filter(|row_filter| !**row_filter).count(),
            2,
            "{schedule:?}"
        );
        let output = Executor::collect(
            Arc::clone(&scan) as Arc<dyn PhysicalOperator>,
            ctx.task_ctx(),
            4,
        )
        .await?;
        let mut ks = output
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        ks.sort_unstable();
        let expected = (0..4 * BIG_FILE_ROWS)
            .filter(|k| k % 8192 < 256)
            .collect::<Vec<_>>();
        assert_eq!(ks, expected);
        let row_filtered = metric(&scan, "row_filtered_units");
        let native = metric(&scan, "natively_filtered_units");
        assert_eq!(row_filtered + native, 16);
        assert!(native >= 1, "{native} units without row filters");
        // A single thread, or a few units per thread: no exploration.
        let gstate = execution_state(&scan, &ctx, 16)?;
        let gstate = downcast_state::<ParquetScanState>(gstate.as_any(), "test")?;
        assert!(!gstate.row_filters.as_ref().unwrap().explores);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scan_predicates_keep_what_may_fail_out_of_row_filters() -> Result<()> {
        let dir = table();
        let ctx = SessionContext::new_with_config(test_session_config());
        register_table(&ctx, dir.path()).await?;
        let kinds = async |sql: &str| -> Result<Vec<(String, ConjunctKind)>> {
            let scan = folded_scan(&ctx, sql).await?;
            Ok(scan
                .predicate
                .candidates
                .iter()
                .map(|candidate| (candidate.expr.to_string(), candidate.kind))
                .collect())
        };
        // `a > 0` may be evaluated in the row filter, `100 / a > 2` may fail
        // (on `a = 0`) and is evaluated natively only, after `a <> 0`;
        // `random()` is volatile.
        let candidates =
            kinds("SELECT id FROM p WHERE a > 0 AND 100 / a > 2 AND random() < 2.0")
                .await?;
        assert_eq!(
            candidates,
            [("a@1 > 0".to_string(), ConjunctKind::Folded)],
            "{candidates:?}"
        );
        // Also those of partition columns (which DataFusion replaces by the
        // values of the file).
        let candidates =
            kinds("SELECT id FROM p WHERE year > a + 2000 AND s = 'x'").await?;
        assert_eq!(candidates.len(), 2, "{candidates:?}");
        Ok(())
    }
}
