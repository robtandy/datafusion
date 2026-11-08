// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`ParquetScanSource`]: a parallel, morsel-driven scan of Parquet files.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_physical_plan::ExecutionPlan;

use super::{FileScan, file_scan_config};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator, SourceResult,
};

/// A source that scans Parquet files on all threads of a pipeline.
///
/// It replaces a `DataSourceExec` whose [`FileScanConfig`] scans Parquet
/// files ([`ParquetSource`]). Instead of reading the file group of a
/// partition on every thread, all files of all file groups form a single
/// queue that the threads share: a thread that needs work claims the next
/// file. Files larger than twice the range size (see
/// [`Self::with_range_bytes`]) are split into byte ranges first, each of
/// which reads the row groups whose midpoint it contains, so a single large
/// file is read by many threads too.
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
        })
    }

    /// Set the size, in bytes, of the byte ranges that files are split into:
    /// a file (or byte range of a file) of at least twice this size is split
    /// into ranges of at least this size. `0` disables splitting.
    ///
    /// By default, this is the maximum of
    /// `datafusion.optimizer.repartition_file_min_size` and 16 MiB, and files
    /// are only split if `datafusion.optimizer.repartition_file_scans` is
    /// enabled. Files are never split if the scan has a limit, must preserve
    /// the order of its files, or a file has extensions (such as a
    /// `ParquetAccessPlan`).
    pub fn with_range_bytes(self, range_bytes: u64) -> Self {
        Self {
            scan: self.scan.with_range_bytes(range_bytes),
        }
    }

    /// The configuration of the scan.
    pub fn config(&self) -> &FileScanConfig {
        &self.scan.config
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
        self.scan.get_global_source_state(ctx)
    }

    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        self.scan.get_local_source_state(ctx, gstate)
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        self.scan.get_data(ctx, gstate, lstate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

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
    use crate::test_utils::{register_test_tables, test_session_config};

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
        // By default the files are too small to be split.
        let files = source
            .config()
            .file_groups
            .iter()
            .map(|g| g.len())
            .sum::<usize>();
        assert_eq!(files, 4);
        assert_eq!(source.get_global_source_state(&exec_ctx)?.max_threads(), 4);

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
