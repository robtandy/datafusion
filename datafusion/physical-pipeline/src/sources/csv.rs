// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`CsvScanSource`]: a parallel scan of CSV files.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_csv::source::CsvSource;
use datafusion_physical_plan::ExecutionPlan;

use super::{FileScan, file_scan_config};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator, SourceResult,
};

/// A source that scans CSV files on all threads of a pipeline.
///
/// It replaces a `DataSourceExec` whose [`FileScanConfig`] scans CSV files
/// ([`CsvSource`]). Like [`ParquetScanSource`](super::ParquetScanSource), all
/// files of all file groups form a single queue that the threads share, and
/// every thread reads the files it claims with the
/// [`Morselizer`](datafusion_datasource::morsel::Morselizer) of the
/// [`CsvSource`] (which opens the files with DataFusion's CSV opener), so
/// the projection, partition columns, header, delimiters and limit of the
/// scan are those of DataFusion.
///
/// Large uncompressed files are split into byte ranges (see
/// [`Self::with_range_bytes`]); every range reads the lines that start in
/// it. Files are not split if values may contain newlines
/// (`newlines_in_values`), as then lines cannot be found from a byte
/// offset. A scan that must preserve the order of its files reads every
/// file group, in order, on a single thread. See the
/// [module documentation](super) for the rows the source produces.
#[derive(Debug)]
pub struct CsvScanSource {
    scan: FileScan,
}

impl CsvScanSource {
    /// Create a source that scans the CSV files of `plan`.
    ///
    /// Returns an error if `plan` does not scan CSV files.
    pub fn try_new(plan: &DataSourceExec) -> Result<Self> {
        let config = file_scan_config::<CsvSource>(plan, "CsvScanSource", "CSV")?;
        Ok(Self {
            scan: FileScan::new("CsvScanSource", config.clone(), plan.schema()),
        })
    }

    /// Create a source that scans the CSV files of `config`.
    ///
    /// Returns an error if the file source of `config` is not a
    /// [`CsvSource`].
    pub fn try_from_config(config: FileScanConfig) -> Result<Self> {
        if !config.file_source().is::<CsvSource>() {
            return internal_err!(
                "CsvScanSource requires a FileScanConfig with a CsvSource, got {}",
                config.file_source().file_type()
            );
        }
        let schema = config.projected_schema()?;
        Ok(Self {
            scan: FileScan::new("CsvScanSource", config, schema),
        })
    }

    /// Set the size, in bytes, of the byte ranges that files are split into:
    /// a file (or byte range of a file) of at least twice this size is split
    /// into ranges of at least this size. `0` disables splitting.
    ///
    /// The default is the same as for
    /// [`ParquetScanSource::with_range_bytes`](super::ParquetScanSource::with_range_bytes).
    /// Compressed files, and files whose values may contain newlines, are
    /// never split.
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

impl PhysicalOperator for CsvScanSource {
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

    use arrow::array::{AsArray, RecordBatch};
    use arrow::datatypes::{DataType, Int64Type};
    use datafusion::prelude::{CsvReadOptions, SessionContext};
    use datafusion_datasource::file_compression_type::FileCompressionType;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_datasource_parquet::source::ParquetSource;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use tempfile::TempDir;

    use crate::executor::Executor;
    use crate::sources::test_harness::{
        LowerScan, assert_native_scans_byte_eq_with,
        assert_native_scans_byte_eq_with_config, assert_native_scans_byte_eq_with_setup,
        data_batch, data_schema, write_csv,
    };
    use crate::sources::try_create_scan_source;
    use crate::test_utils::{register_test_tables, test_session_config};

    /// Write CSV files with a header, partitioned by `part`: one large file,
    /// a small one, one with a single row and one without rows.
    fn write_tables(dir: &Path) -> Result<()> {
        write_csv(
            &dir.join("c/part=0/large.csv"),
            &[data_batch(0..2500), data_batch(2500..6000)],
        )?;
        write_csv(&dir.join("c/part=1/small.csv"), &[data_batch(6000..6100)])?;
        write_csv(&dir.join("c/part=2/one.csv"), &[data_batch(6100..6101)])?;
        write_csv(&dir.join("c/part=3/empty.csv"), &[data_batch(0..0)])?;
        write_csv(&dir.join("single.csv"), &[data_batch(0..3000)])?;
        // Values with newlines.
        std::fs::create_dir_all(dir.join("multiline"))?;
        std::fs::write(
            dir.join("multiline/data.csv"),
            "id,note\n1,\"first\nline\"\n2,plain\n3,\"x\ny\nz\"\n4,\n",
        )?;
        Ok(())
    }

    async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<()> {
        register_test_tables(ctx)?;
        ctx.register_csv(
            "c",
            dir.join("c").to_str().unwrap(),
            CsvReadOptions::new()
                .table_partition_cols(vec![("part".to_string(), DataType::Int32)]),
        )
        .await?;
        ctx.register_csv(
            "single",
            dir.join("single.csv").to_str().unwrap(),
            CsvReadOptions::new(),
        )
        .await?;
        ctx.register_csv(
            "multiline",
            dir.join("multiline").to_str().unwrap(),
            CsvReadOptions::new().newlines_in_values(true),
        )
        .await
    }

    fn tables() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_tables(dir.path()).unwrap();
        dir
    }

    const QUERIES: &[&str] = &[
        "SELECT * FROM c",
        "SELECT id, s, part FROM c",
        "SELECT part FROM c",
        "SELECT id, a FROM c WHERE a > 40",
        "SELECT * FROM c WHERE part = 1",
        "SELECT part, COUNT(*), SUM(a), MIN(s), MAX(d) FROM c GROUP BY part",
        "SELECT c.id, t2.name FROM c JOIN t2 ON c.a = t2.k",
        "SELECT id, c FROM c ORDER BY c DESC NULLS LAST, id LIMIT 10",
        "SELECT * FROM single",
        // A limit on a single file: its first rows.
        "SELECT * FROM single LIMIT 25",
        "SELECT * FROM multiline",
    ];

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_scans() {
        let dir = tables();
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        for sql in QUERIES {
            assert_native_scans_byte_eq_with_setup(sql, &setup).await;
        }
        for target_partitions in [1, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in &QUERIES[..4] {
                assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup)
                    .await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_byte_ranges() {
        // Split the files into ranges of about 1000 bytes: every line is read
        // by the range in which it starts (the header only by the first one).
        let dir = tables();
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        let lower: LowerScan = |plan| match plan.downcast_ref::<DataSourceExec>() {
            Some(exec) if exec.downcast_to_file_source::<CsvSource>().is_some() => {
                let source = CsvScanSource::try_new(exec)?.with_range_bytes(1000);
                Ok(Some(Arc::new(source)))
            }
            _ => try_create_scan_source(plan),
        };
        for sql in QUERIES {
            assert_native_scans_byte_eq_with(sql, test_session_config(), &setup, lower)
                .await;
        }
    }

    /// The ids of `batches` (whose first column is `id`), sorted.
    fn ids(batches: &[RecordBatch]) -> Vec<i64> {
        let mut ids = batches
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

    async fn scan_source(ctx: &SessionContext, table: &str) -> Result<CsvScanSource> {
        let plan = ctx
            .sql(&format!("SELECT * FROM {table}"))
            .await?
            .create_physical_plan()
            .await?;
        CsvScanSource::try_new(plan.downcast_ref::<DataSourceExec>().unwrap())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn splits_only_files_it_can_split() -> Result<()> {
        let dir = tables();
        let ctx = SessionContext::new_with_config(test_session_config());
        register_tables(&ctx, dir.path()).await?;
        let exec_ctx = ExecutionContext::new(ctx.task_ctx(), 0, 4);

        let size = std::fs::metadata(dir.path().join("single.csv"))?.len();
        let source = scan_source(&ctx, "single").await?.with_range_bytes(1000);
        let units = source.get_global_source_state(&exec_ctx)?.max_threads();
        assert_eq!(units, (size / 1000) as usize, "file size {size}");
        let source: Arc<dyn PhysicalOperator> = Arc::new(source);
        for threads in [1, 4] {
            let output =
                Executor::collect(Arc::clone(&source), ctx.task_ctx(), threads).await?;
            assert_eq!(ids(&output), (0..3000).collect::<Vec<_>>());
            assert!(output.iter().all(|b| b.num_rows() <= 2048));
        }

        // Values with newlines cannot be split: lines cannot be found from
        // byte offsets.
        let source = scan_source(&ctx, "multiline").await?.with_range_bytes(4);
        assert_eq!(source.get_global_source_state(&exec_ctx)?.max_threads(), 1);
        let output = Executor::collect(Arc::new(source), ctx.task_ctx(), 4).await?;
        assert_eq!(ids(&output), vec![1, 2, 3, 4]);

        // Neither can compressed files.
        ctx.sql(&format!(
            "COPY (SELECT * FROM single) TO '{}' STORED AS CSV OPTIONS ('format.compression' 'gzip')",
            dir.path().join("gzip/data.csv.gz").display()
        ))
        .await?
        .collect()
        .await?;
        ctx.register_csv(
            "gzip",
            dir.path().join("gzip").to_str().unwrap(),
            CsvReadOptions::new()
                .file_compression_type(FileCompressionType::GZIP)
                .file_extension(".csv.gz"),
        )
        .await?;
        let source = scan_source(&ctx, "gzip").await?.with_range_bytes(100);
        assert_eq!(source.get_global_source_state(&exec_ctx)?.max_threads(), 1);
        let output = Executor::collect(Arc::new(source), ctx.task_ctx(), 4).await?;
        assert_eq!(ids(&output), (0..3000).collect::<Vec<_>>());
        let setup = async |ctx: &SessionContext| {
            register_tables(ctx, dir.path()).await?;
            ctx.register_csv(
                "gzip",
                dir.path().join("gzip").to_str().unwrap(),
                CsvReadOptions::new()
                    .file_compression_type(FileCompressionType::GZIP)
                    .file_extension(".csv.gz"),
            )
            .await
        };
        assert_native_scans_byte_eq_with_setup("SELECT * FROM gzip", &setup).await;
        assert_native_scans_byte_eq_with_setup("SELECT * FROM gzip LIMIT 10", &setup)
            .await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limit_applies_to_all_threads() -> Result<()> {
        let dir = tables();
        let ctx = SessionContext::new_with_config(test_session_config());
        register_tables(&ctx, dir.path()).await?;
        let scan = scan_source(&ctx, "c").await?;
        for limit in [0, 1, 99, 2049, 6101, 7000] {
            let config = FileScanConfigBuilder::from(scan.config().clone())
                .with_limit(Some(limit))
                .build();
            let source: Arc<dyn PhysicalOperator> =
                Arc::new(CsvScanSource::try_from_config(config)?);
            for threads in [1, 4] {
                let output =
                    Executor::collect(Arc::clone(&source), ctx.task_ctx(), threads)
                        .await?;
                let ids = ids(&output);
                assert_eq!(ids.len(), limit.min(6101), "{limit} on {threads} threads");
                assert!(ids.iter().all(|id| (0..6101).contains(id)));
            }
        }
        Ok(())
    }

    #[test]
    fn requires_a_csv_scan() {
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(ParquetSource::new(data_schema())),
        )
        .build();
        let err = CsvScanSource::try_from_config(config.clone()).unwrap_err();
        assert!(err.to_string().contains("CsvSource"), "{err}");
        let err =
            CsvScanSource::try_new(&DataSourceExec::new(Arc::new(config))).unwrap_err();
        assert!(err.to_string().contains("CSV files"), "{err}");
    }
}
