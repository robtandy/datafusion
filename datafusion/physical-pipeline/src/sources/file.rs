// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`FileScanSource`]: a parallel scan of the files of any file format, e.g.
//! JSON, Avro and Arrow IPC files.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::ExecutionPlan;

use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};

use super::FileScan;
use super::batches::chunk_memory;
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state_mut,
};

/// A source that scans the files of a `DataSourceExec` of any file format
/// (any `FileSource`, e.g. JSON, Avro and Arrow IPC files) on all threads of
/// a pipeline.
///
/// Like [`ParquetScanSource`](super::ParquetScanSource) and
/// [`CsvScanSource`](super::CsvScanSource), the files of all file groups form
/// a single queue that the threads share, and every thread reads the files it
/// claims with the [`Morselizer`](datafusion_datasource::morsel::Morselizer)
/// of the `FileSource` of the scan, which DataFusion's `FileStream` reads them
/// with too: the projection, partition columns, schema adaptation and limit of
/// the scan are those of DataFusion.
///
/// The memory of the batch that a thread emits the chunks of is reserved in
/// the memory pool of the session.
///
/// The files are read as DataFusion planned them: the byte ranges into which
/// DataFusion split large files (for the formats that can read byte ranges,
/// see `FileSource::repartitioned`, e.g. newline-delimited JSON) are claimed
/// separately, and no file is split further, as not every format can read
/// any byte range of a file (e.g. JSON arrays or Arrow IPC streams cannot). A
/// scan that must preserve the order of its files reads every file group, in
/// order, on a single thread. See the [module documentation](super) for the
/// rows the source produces.
#[derive(Debug)]
pub struct FileScanSource {
    scan: FileScan,
}

impl FileScanSource {
    /// Create a source that scans the files of `plan`.
    ///
    /// Returns an error if `plan` does not scan files.
    pub fn try_new(plan: &DataSourceExec) -> Result<Self> {
        let Some(config) = plan.data_source().downcast_ref::<FileScanConfig>() else {
            return internal_err!(
                "FileScanSource requires a DataSourceExec that scans files"
            );
        };
        Ok(Self::new(config.clone(), plan.schema()))
    }

    /// Create a source that scans the files of `config`.
    pub fn try_from_config(config: FileScanConfig) -> Result<Self> {
        let schema = config.projected_schema()?;
        Ok(Self::new(config, schema))
    }

    fn new(config: FileScanConfig, schema: SchemaRef) -> Self {
        Self {
            scan: FileScan::new("FileScanSource", config, schema).with_range_bytes(0),
        }
    }

    /// Lower `plan` to a [`FileScanSource`] if it is a `DataSourceExec` that
    /// scans files.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        match plan.downcast_ref::<DataSourceExec>() {
            Some(exec) if exec.data_source().is::<FileScanConfig>() => {
                Self::try_new(exec).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// The configuration of the scan.
    pub fn config(&self) -> &FileScanConfig {
        &self.scan.config
    }
}

impl PhysicalOperator for FileScanSource {
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
        let scan = self.scan.get_local_source_state(ctx, gstate)?;
        let reservation =
            MemoryConsumer::new("FileScanSource").register(ctx.task_ctx.memory_pool());
        Ok(Box::new(FileScanSourceState { scan, reservation }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state =
            downcast_state_mut::<FileScanSourceState>(lstate.as_any_mut(), self.name())?;
        let result = self.scan.get_data(ctx, gstate, state.scan.as_mut())?;
        match &result {
            // A chunk is a slice of the batch that the thread read, whose
            // buffers it shares: the memory of the batch, which the scan
            // holds until it emitted all of its chunks.
            SourceResult::Chunk(chunk) if !chunk.is_empty() => {
                state.reservation.try_resize(chunk_memory(chunk))?;
            }
            SourceResult::Finished => {
                state.reservation.free();
            }
            _ => {}
        }
        Ok(result)
    }
}

/// The local source state of a [`FileScanSource`]: the state of its scan,
/// and the memory of the batch that the thread emits the chunks of, which is
/// reserved in the memory pool of the session.
struct FileScanSourceState {
    scan: Box<dyn LocalSourceState>,
    reservation: MemoryReservation,
}

impl LocalSourceState for FileScanSourceState {}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    use arrow::array::{
        AsArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use arrow_ipc::writer::StreamWriter;
    use datafusion::datasource::file_format::options::ArrowReadOptions;
    use datafusion::prelude::{JsonReadOptions, SessionContext};
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_plan::ExecutionPlanProperties;

    use crate::executor::Executor;
    use crate::operators::lowering_tests::{
        assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit,
    };
    use crate::test_utils::{register_test_tables, test_session_config};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("a", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("c", DataType::Float64, true),
        ]))
    }

    /// The rows with the ids `ids`, with nulls (and `-0.0` in `c`).
    fn batch(ids: std::ops::Range<i64>) -> RecordBatch {
        let a = ids
            .clone()
            .map(|i| (i % 11 != 3).then_some(((i * 7) % 101) as i32 - 50));
        let s = ids
            .clone()
            .map(|i| (i % 13 != 5).then(|| format!("value-{}", i % 37)));
        let c = ids.clone().map(|i| match i {
            _ if i % 17 == 2 => None,
            _ if i % 89 == 7 => Some(-0.0),
            _ => Some(((i * 31) % 1000) as f64 / 8.0 - 60.0),
        });
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(ids)),
                Arc::new(Int32Array::from_iter(a)),
                Arc::new(StringArray::from_iter(s)),
                Arc::new(Float64Array::from_iter(c)),
            ],
        )
        .unwrap()
    }

    /// Write `batches` to the Arrow IPC stream file `path`.
    fn write_arrow_stream(path: &Path, batches: &[RecordBatch]) -> Result<()> {
        std::fs::create_dir_all(path.parent().unwrap())?;
        let file = std::fs::File::create(path)?;
        let mut writer = StreamWriter::try_new(file, &schema())?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.finish()?;
        Ok(())
    }

    /// Write the files of the tables of [`register_tables`] to `dir`: JSON
    /// and Arrow IPC files, partitioned by `part` (a large file, a small one,
    /// one with a single row and one without rows), an Arrow IPC stream and a
    /// JSON array.
    async fn write_tables(dir: &Path) -> Result<()> {
        let ctx = SessionContext::new();
        let parts = [(0, 0..5000), (1, 5000..5100), (2, 5100..5101), (3, 0..0)];
        for (part, ids) in parts {
            ctx.register_batch("part_rows", batch(ids))?;
            for (format, extension) in [("JSON", "json"), ("ARROW", "arrow")] {
                let path = dir.join(format!("{extension}/part={part}/data.{extension}"));
                std::fs::create_dir_all(path.parent().unwrap())?;
                let sql =
                    format!("COPY part_rows TO '{}' STORED AS {format}", path.display());
                ctx.sql(&sql).await?.collect().await?;
            }
            ctx.deregister_table("part_rows")?;
        }
        write_arrow_stream(
            &dir.join("stream/data.arrow"),
            &[batch(0..3000), batch(3000..3500)],
        )?;
        std::fs::create_dir_all(dir.join("array"))?;
        std::fs::write(
            dir.join("array/data.json"),
            r#"[{"id": 1, "note": "one"}, {"id": 2, "note": null}, {"id": 3}]"#,
        )?;
        Ok(())
    }

    /// Register the tables of the files of [`write_tables`] in `dir`: `j`
    /// (JSON) and `a` (Arrow IPC files) partitioned by `part`, `stream` (an
    /// Arrow IPC stream) and `array` (a JSON array).
    async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<()> {
        register_test_tables(ctx)?;
        let part = || vec![("part".to_string(), DataType::Int32)];
        ctx.register_json(
            "j",
            dir.join("json").to_str().unwrap(),
            JsonReadOptions::default().table_partition_cols(part()),
        )
        .await?;
        ctx.register_arrow(
            "a",
            dir.join("arrow").to_str().unwrap(),
            ArrowReadOptions::default().table_partition_cols(part()),
        )
        .await?;
        ctx.register_arrow(
            "stream",
            dir.join("stream").to_str().unwrap(),
            ArrowReadOptions::default(),
        )
        .await?;
        ctx.register_json(
            "array",
            dir.join("array").to_str().unwrap(),
            JsonReadOptions::default().newline_delimited(false),
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scans_of_json_and_arrow_files_are_native() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_tables(dir.path()).await?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        for table in ["j", "a"] {
            for sql in [
                format!("SELECT * FROM {table}"),
                format!("SELECT id, s, part FROM {table}"),
                format!("SELECT part FROM {table}"),
                format!("SELECT id, a FROM {table} WHERE a > 40"),
                format!("SELECT * FROM {table} WHERE part = 1"),
                format!(
                    "SELECT part, COUNT(*), SUM(a), MIN(s), MAX(id) FROM {table} \
                     GROUP BY part"
                ),
                format!(
                    "SELECT {table}.id, t2.name FROM {table} JOIN t2 ON {table}.a = t2.k"
                ),
                format!(
                    "SELECT id, c FROM {table} ORDER BY c DESC NULLS LAST, id LIMIT 10"
                ),
                format!("SELECT COUNT(*) FROM (SELECT * FROM {table} LIMIT 25)"),
            ] {
                for target_partitions in [1, 4, 16] {
                    let config =
                        test_session_config().with_target_partitions(target_partitions);
                    let root = assert_lowered(&sql, config, &setup).await;
                    assert_eq!(count::<FileScanSource>(&root), 1, "{sql}");
                }
            }
        }
        // Arrow IPC streams and JSON arrays, which cannot be read in byte
        // ranges: DataFusion does not split Arrow IPC streams even with a
        // tiny `repartition_file_min_size` (it fails to scan the ranges of
        // the JSON arrays that it splits), and neither does the native scan.
        for (sql, min_size) in [
            ("SELECT * FROM stream", 1),
            ("SELECT s, COUNT(*) AS n FROM stream GROUP BY s", 1),
            ("SELECT * FROM array", 10 * 1024 * 1024),
            ("SELECT note, id FROM array WHERE id > 1", 10 * 1024 * 1024),
        ] {
            for target_partitions in [1, 4] {
                let mut config =
                    test_session_config().with_target_partitions(target_partitions);
                config.options_mut().optimizer.repartition_file_min_size = min_size;
                let root = assert_lowered(sql, config, &setup).await;
                assert_eq!(count::<FileScanSource>(&root), 1, "{sql}");
            }
        }
        Ok(())
    }

    /// The leaf of the plan of `sql`.
    async fn scan_of(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let mut scan = None;
        plan.apply(|node| {
            if node.children().is_empty() {
                scan = Some(Arc::clone(node));
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(scan.unwrap())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_byte_ranges_of_json_files_are_read_once() -> Result<()> {
        // DataFusion splits the large JSON file into byte ranges, which the
        // threads claim separately.
        let dir = tempfile::tempdir()?;
        write_tables(dir.path()).await?;
        let mut config = test_session_config().with_target_partitions(8);
        config.options_mut().optimizer.repartition_file_min_size = 1000;
        let ctx = SessionContext::new_with_config(config.clone());
        register_tables(&ctx, dir.path()).await?;
        let scan = scan_of(&ctx, "SELECT id FROM j WHERE part = 0").await?;
        assert!(scan.output_partitioning().partition_count() > 1);
        let source: Arc<dyn PhysicalOperator> =
            Arc::new(FileScanSource::try_lower(&scan)?.unwrap());
        for threads in [1, 3, 8] {
            let output =
                Executor::collect(Arc::clone(&source), ctx.task_ctx(), threads).await?;
            let mut ids = output
                .iter()
                .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            assert_eq!(ids, (0..5000).collect::<Vec<_>>(), "{threads} threads");
        }
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        assert_lowered("SELECT * FROM j", config, setup).await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_batches_of_a_thread_reserve_memory() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_tables(dir.path()).await?;
        for table in ["j", "a", "stream", "array"] {
            let sql = format!("SELECT * FROM {table}");
            // A pool of one byte cannot hold a batch.
            let ctx = pipeline_context_with_memory_limit(1);
            register_tables(&ctx, dir.path()).await?;
            let root = lowered(&ctx, &sql).await?;
            assert_eq!(count::<FileScanSource>(&root), 1, "{sql}");
            assert_resources_exhausted(&ctx, &sql).await;
            // A pool that holds the batches of the threads.
            let ctx = pipeline_context_with_memory_limit(64 * 1024 * 1024);
            register_tables(&ctx, dir.path()).await?;
            let batches = ctx.sql(&sql).await?.collect().await?;
            assert!(!batches.is_empty(), "{sql}");
            assert_eq!(ctx.runtime_env().memory_pool.reserved(), 0, "{sql}");
        }
        Ok(())
    }

    #[test]
    fn requires_a_scan_of_files() -> Result<()> {
        let config = MemorySourceConfig::try_new(&[vec![]], schema(), None)?;
        let exec = DataSourceExec::new(Arc::new(config));
        assert!(FileScanSource::try_new(&exec).is_err());
        let plan: Arc<dyn ExecutionPlan> = Arc::new(exec);
        assert!(FileScanSource::try_lower(&plan)?.is_none());
        Ok(())
    }
}
