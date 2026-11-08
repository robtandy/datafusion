// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`MemoryScanSource`]: a parallel scan of in-memory batches.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::nested_struct::adapt_batch_to_schema;
use datafusion_common::{Result, internal_err};
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::{DataSource, DataSourceExec};

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state,
};

/// A source that scans in-memory batches on all threads of a pipeline.
///
/// It replaces a `DataSourceExec` that scans a [`MemorySourceConfig`]
/// (e.g. a `MemTable` or a `VALUES` list). All batches of all partitions are
/// cut into slices of at most [`DEFAULT_VECTOR_SIZE`] rows, which form a
/// single work queue: every call of
/// [`get_data`](PhysicalOperator::get_data), on any thread, claims the next
/// slice through an atomic cursor and returns it as a zero-copy
/// [`DataChunk`]. So the threads share the work evenly however the rows are
/// distributed over the partitions, even if a single partition holds all of
/// them.
///
/// The source produces exactly the rows of DataFusion's `MemoryStream`s:
/// the projection is applied to every batch, batches whose schema is
/// stricter than the output schema are adapted to it, and a `fetch` limits
/// every partition to its first `fetch` rows.
///
/// The order of the rows is preserved (by reading the slices on a single
/// thread) for a scan with a single partition and an output ordering, see
/// [`Self::with_preserve_order`].
#[derive(Debug)]
pub struct MemoryScanSource {
    schema: SchemaRef,
    /// The projected batches of all partitions, in partition order. Only
    /// batches that contribute rows are kept.
    batches: Vec<RecordBatch>,
    /// For every batch, whether it must be adapted to `schema`.
    adapt: Vec<bool>,
    /// The work queue: slices of at most [`DEFAULT_VECTOR_SIZE`] rows of
    /// `batches`, in order.
    slices: Vec<BatchSlice>,
    num_partitions: usize,
    num_rows: usize,
    fetch: Option<usize>,
    preserve_order: bool,
}

/// The rows `offset..offset + len` of `batches[batch]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchSlice {
    batch: usize,
    offset: usize,
    len: usize,
}

impl MemoryScanSource {
    /// Create a source that scans `partitions`, whose batches have the
    /// schema `schema` (before `projection`), like
    /// [`MemorySourceConfig::try_new`]. The output schema is `schema` with
    /// `projection` applied. If `fetch` is set, every partition produces at
    /// most `fetch` rows.
    pub fn try_new(
        partitions: &[Vec<RecordBatch>],
        schema: SchemaRef,
        projection: Option<&[usize]>,
        fetch: Option<usize>,
    ) -> Result<Self> {
        let schema = match projection {
            Some(columns) => Arc::new(schema.project(columns)?),
            None => schema,
        };
        Self::try_new_with_output_schema(partitions, schema, projection, fetch)
    }

    /// Create a source that scans `config`.
    ///
    /// The order of the rows is preserved if `config` has a single
    /// partition and an output ordering.
    pub fn try_from_config(config: &MemorySourceConfig) -> Result<Self> {
        let eq_properties = config.eq_properties();
        let source = Self::try_new_with_output_schema(
            config.partitions(),
            Arc::clone(eq_properties.schema()),
            config.projection().as_deref(),
            config.fetch(),
        )?;
        let ordered =
            config.partitions().len() <= 1 && eq_properties.output_ordering().is_some();
        Ok(source.with_preserve_order(ordered))
    }

    /// Create the source of `plan`, a `DataSourceExec` that scans a
    /// [`MemorySourceConfig`], see [`Self::try_from_config`].
    pub fn try_from_exec(plan: &DataSourceExec) -> Result<Self> {
        match plan.data_source().downcast_ref::<MemorySourceConfig>() {
            Some(config) => Self::try_from_config(config),
            None => internal_err!(
                "MemoryScanSource requires a DataSourceExec that scans a MemorySourceConfig"
            ),
        }
    }

    /// Whether to preserve the order of the rows: if `true`, the slices are
    /// read in order by a single thread.
    pub fn with_preserve_order(mut self, preserve_order: bool) -> Self {
        self.preserve_order = preserve_order;
        self
    }

    fn try_new_with_output_schema(
        partitions: &[Vec<RecordBatch>],
        schema: SchemaRef,
        projection: Option<&[usize]>,
        fetch: Option<usize>,
    ) -> Result<Self> {
        let mut batches = vec![];
        let mut adapt = vec![];
        let mut slices = vec![];
        let mut num_rows = 0;
        for partition in partitions {
            // Like `MemoryStream::with_fetch`: the first `fetch` rows of
            // every partition.
            let mut remaining = fetch.unwrap_or(usize::MAX);
            for batch in partition {
                let rows = batch.num_rows().min(remaining);
                if rows == 0 {
                    continue;
                }
                remaining -= rows;
                num_rows += rows;
                let batch = match projection {
                    Some(columns) => batch.project(columns)?,
                    None => batch.clone(),
                };
                // Like `MemoryStream`, which adapts the batches that are
                // stricter than its schema (e.g. in the nullability of
                // nested fields) and emits all others as they are.
                let batch_schema = batch.schema_ref();
                adapt.push(
                    batch_schema.as_ref() != schema.as_ref()
                        && schema.contains(batch_schema),
                );
                let index = batches.len();
                batches.push(batch);
                for offset in (0..rows).step_by(DEFAULT_VECTOR_SIZE) {
                    slices.push(BatchSlice {
                        batch: index,
                        offset,
                        len: (rows - offset).min(DEFAULT_VECTOR_SIZE),
                    });
                }
            }
        }
        Ok(Self {
            schema,
            batches,
            adapt,
            slices,
            num_partitions: partitions.len(),
            num_rows,
            fetch,
            preserve_order: false,
        })
    }
}

/// Global source state of a [`MemoryScanSource`]: the next slice to claim.
struct MemoryScanGlobalState {
    next: AtomicUsize,
    max_threads: usize,
}

impl GlobalSourceState for MemoryScanGlobalState {
    fn max_threads(&self) -> usize {
        self.max_threads
    }
}

impl PhysicalOperator for MemoryScanSource {
    fn name(&self) -> &str {
        "MemoryScanSource"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        let fetch = self
            .fetch
            .map(|fetch| format!(", fetch={fetch}"))
            .unwrap_or_default();
        let order = if self.preserve_order {
            ", preserve_order=true"
        } else {
            ""
        };
        format!(
            "MemoryScanSource: partitions={}, rows={}, chunks={}{fetch}{order}",
            self.num_partitions,
            self.num_rows,
            self.slices.len()
        )
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let max_threads = if self.preserve_order {
            1
        } else {
            self.slices.len().max(1)
        };
        Ok(Arc::new(MemoryScanGlobalState {
            next: AtomicUsize::new(0),
            max_threads,
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let gstate =
            downcast_state::<MemoryScanGlobalState>(gstate.as_any(), self.name())?;
        let index = gstate.next.fetch_add(1, Ordering::Relaxed);
        let Some(slice) = self.slices.get(index) else {
            return Ok(SourceResult::Finished);
        };
        let batch = &self.batches[slice.batch];
        let chunk = if slice.len == batch.num_rows() {
            batch.clone()
        } else {
            batch.slice(slice.offset, slice.len)
        };
        let chunk = if self.adapt[slice.batch] {
            adapt_batch_to_schema(chunk, &self.schema)?
        } else {
            chunk
        };
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(chunk)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use arrow::array::{AsArray, Int32Array, Int64Array, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field, Fields, Int64Type, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_datasource_csv::source::CsvSource;
    use datafusion_execution::TaskContext;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::ExecutionPlan;

    use crate::bridge::BridgeOperator;
    use crate::executor::Executor;
    use crate::operator::EmptyState;
    use crate::sources::test_harness::{
        assert_native_scans_byte_eq, assert_native_scans_byte_eq_with_config,
        assert_native_scans_byte_eq_with_setup,
    };
    use crate::test_utils::{register_test_tables, test_session_config};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
        ]))
    }

    fn batch(values: std::ops::Range<i64>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(values.clone())),
                Arc::new(StringArray::from_iter(
                    values.map(|v| (v % 3 != 0).then(|| format!("s{v}"))),
                )),
            ],
        )
        .unwrap()
    }

    /// One huge partition and 15 empty partitions.
    fn skewed_partitions(rows: i64) -> Vec<Vec<RecordBatch>> {
        let mut partitions = vec![vec![batch(0..rows)]];
        partitions.resize(16, vec![]);
        partitions
    }

    fn values(batches: &[RecordBatch]) -> Vec<i64> {
        let mut values = batches
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect::<Vec<_>>();
        values.sort_unstable();
        values
    }

    fn task_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default())
    }

    /// Call `get_data` of `source` on `threads` simulated threads, one call
    /// per thread in turn, until all of them are finished. Returns the
    /// values every thread produced.
    fn round_robin(source: &dyn PhysicalOperator, threads: usize) -> Vec<Vec<i64>> {
        let ctx = ExecutionContext::new(task_ctx(), 0, threads);
        let gstate = source.get_global_source_state(&ctx).unwrap();
        let mut lstates = (0..threads)
            .map(|thread| {
                let ctx = ExecutionContext::new(task_ctx(), thread, threads);
                source
                    .get_local_source_state(&ctx, gstate.as_ref())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut produced = vec![vec![]; threads];
        let mut finished = vec![false; threads];
        while finished.iter().any(|finished| !finished) {
            for thread in 0..threads {
                if finished[thread] {
                    continue;
                }
                let ctx = ExecutionContext::new(task_ctx(), thread, threads);
                match source
                    .get_data(&ctx, gstate.as_ref(), lstates[thread].as_mut())
                    .unwrap()
                {
                    SourceResult::Chunk(chunk) => {
                        assert!(chunk.len() <= DEFAULT_VECTOR_SIZE);
                        let batch = chunk.into_record_batch().unwrap();
                        produced[thread]
                            .extend(batch.column(0).as_primitive::<Int64Type>().values());
                    }
                    SourceResult::Finished => finished[thread] = true,
                    SourceResult::Blocked(interrupt) => {
                        panic!("unexpected {interrupt:?}")
                    }
                }
            }
        }
        produced
    }

    #[test]
    fn batches_are_cut_into_slices_of_at_most_vector_size() -> Result<()> {
        let partitions = vec![
            vec![batch(0..5000), batch(5000..5000), batch(5000..5100)],
            vec![],
            vec![batch(5100..5100 + 2048)],
        ];
        let source = MemoryScanSource::try_new(&partitions, schema(), None, None)?;
        let lengths = source
            .slices
            .iter()
            .map(|slice| (slice.batch, slice.offset, slice.len))
            .collect::<Vec<_>>();
        // The empty batch contributes no slice.
        assert_eq!(
            lengths,
            vec![
                (0, 0, 2048),
                (0, 2048, 2048),
                (0, 4096, 904),
                (1, 0, 100),
                (2, 0, 2048)
            ]
        );
        assert_eq!(source.num_rows, 7148);
        assert_eq!(
            source.description(),
            "MemoryScanSource: partitions=3, rows=7148, chunks=5"
        );
        let ctx = ExecutionContext::new(task_ctx(), 0, 4);
        assert_eq!(source.get_global_source_state(&ctx)?.max_threads(), 5);
        Ok(())
    }

    #[test]
    fn threads_share_a_single_huge_partition() -> Result<()> {
        let partitions = skewed_partitions(10_000);
        let native = MemoryScanSource::try_new(&partitions, schema(), None, None)?;
        let produced = round_robin(&native, 4);
        // Five slices of the single non-empty partition, claimed in turn.
        let rows = produced.iter().map(Vec::len).collect::<Vec<_>>();
        assert_eq!(rows, vec![2048 + 1808, 2048, 2048, 2048]);
        let mut all = produced.concat();
        all.sort_unstable();
        assert_eq!(all, (0..10_000).collect::<Vec<_>>());

        // The bridge pins partitions to threads: the thread that claims the
        // huge partition reads all of it.
        let plan: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&partitions, schema(), None)?;
        let bridge = BridgeOperator::new(plan);
        let rows = round_robin(&bridge, 4)
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>();
        assert_eq!(rows, vec![10_000, 0, 0, 0]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn executes_on_many_threads() -> Result<()> {
        let mut partitions = skewed_partitions(20_000);
        partitions[3] = vec![batch(20_000..20_005), batch(20_005..23_000)];
        let source: Arc<dyn PhysicalOperator> = Arc::new(MemoryScanSource::try_new(
            &partitions,
            schema(),
            None,
            None,
        )?);
        for threads in [1, 3, 16] {
            let output =
                Executor::collect(Arc::clone(&source), task_ctx(), threads).await?;
            assert_eq!(values(&output), (0..23_000).collect::<Vec<_>>());
            assert!(output.iter().all(|b| b.num_rows() <= DEFAULT_VECTOR_SIZE));
            assert!(output.iter().all(|b| b.schema() == schema()));
        }
        // An empty source.
        let empty: Arc<dyn PhysicalOperator> = Arc::new(MemoryScanSource::try_new(
            &[vec![], vec![]],
            schema(),
            None,
            None,
        )?);
        assert!(Executor::collect(empty, task_ctx(), 4).await?.is_empty());
        Ok(())
    }

    #[test]
    fn fetch_limits_every_partition() -> Result<()> {
        // Like DataFusion's `MemoryStream`: the first 10 rows of every
        // partition.
        let partitions = vec![
            vec![batch(0..4), batch(4..3000)],
            vec![batch(3000..3005)],
            vec![],
            vec![batch(4000..4000), batch(4000..4100)],
        ];
        let source = MemoryScanSource::try_new(&partitions, schema(), None, Some(10))?;
        let mut produced = round_robin(&source, 2).concat();
        produced.sort_unstable();
        let mut expected = (0..10)
            .chain(3000..3005)
            .chain(4000..4010)
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(produced, expected);
        assert!(source.description().ends_with("fetch=10"));

        let source = MemoryScanSource::try_new(&partitions, schema(), None, Some(0))?;
        assert!(source.slices.is_empty());
        assert!(round_robin(&source, 2).concat().is_empty());
        Ok(())
    }

    #[test]
    fn projection() -> Result<()> {
        let partitions = vec![vec![batch(0..3000)]];
        let source =
            MemoryScanSource::try_new(&partitions, schema(), Some(&[1, 0]), None)?;
        let expected = Arc::new(schema().project(&[1, 0])?);
        assert_eq!(source.schema(), expected);
        let ctx = ExecutionContext::new(task_ctx(), 0, 1);
        let gstate = source.get_global_source_state(&ctx)?;
        let SourceResult::Chunk(chunk) =
            source.get_data(&ctx, gstate.as_ref(), &mut EmptyState)?
        else {
            panic!("expected a chunk");
        };
        assert_eq!(chunk.schema(), &expected);
        assert_eq!(chunk.len(), DEFAULT_VECTOR_SIZE);
        assert_eq!(chunk.column(1).as_primitive::<Int64Type>().value(7), 7);

        // An empty projection keeps the row count (e.g. for `COUNT(*)`).
        let source = MemoryScanSource::try_new(&partitions, schema(), Some(&[]), None)?;
        assert!(source.schema().fields().is_empty());
        let gstate = source.get_global_source_state(&ctx)?;
        let mut rows = vec![];
        while let SourceResult::Chunk(chunk) =
            source.get_data(&ctx, gstate.as_ref(), &mut EmptyState)?
        {
            assert_eq!(chunk.num_columns(), 0);
            rows.push(chunk.len());
        }
        assert_eq!(rows, vec![2048, 952]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_ordered_partition_is_read_in_order() -> Result<()> {
        let ordering =
            LexOrdering::new(vec![PhysicalSortExpr::new_default(col("v", &schema())?)])
                .unwrap();
        let config = MemorySourceConfig::try_new(
            &[vec![batch(0..5000), batch(5000..9000)]],
            schema(),
            None,
        )?
        .try_with_sort_information(vec![ordering.clone()])?;
        let source = MemoryScanSource::try_from_config(&config)?;
        assert!(source.preserve_order);
        assert!(source.description().ends_with("preserve_order=true"));
        let ctx = ExecutionContext::new(task_ctx(), 0, 4);
        assert_eq!(source.get_global_source_state(&ctx)?.max_threads(), 1);
        let output = Executor::collect(Arc::new(source), task_ctx(), 4).await?;
        let in_order = output
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(in_order, (0..9000).collect::<Vec<_>>());

        // Unordered, or with several partitions, all threads read.
        let config = MemorySourceConfig::try_new(
            &[vec![batch(0..5000)], vec![batch(5000..9000)]],
            schema(),
            None,
        )?
        .try_with_sort_information(vec![ordering])?;
        let source = MemoryScanSource::try_from_config(&config)?;
        assert!(!source.preserve_order);
        assert_eq!(source.get_global_source_state(&ctx)?.max_threads(), 5);
        assert_eq!(
            source
                .with_preserve_order(true)
                .get_global_source_state(&ctx)?
                .max_threads(),
            1
        );
        Ok(())
    }

    #[test]
    fn requires_a_memory_scan() {
        let plan = DataSourceExec::new(Arc::new(
            FileScanConfigBuilder::new(
                ObjectStoreUrl::local_filesystem(),
                Arc::new(CsvSource::new(schema())),
            )
            .build(),
        ));
        let err = MemoryScanSource::try_from_exec(&plan).unwrap_err();
        assert!(err.to_string().contains("MemorySourceConfig"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_scans_of_the_test_tables() {
        for sql in [
            "SELECT * FROM t1",
            "SELECT * FROM t2",
            "SELECT * FROM t_empty",
            "SELECT id, h, a FROM t1",
            "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
            "SELECT d, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh, MAX(f) AS mf FROM t1 GROUP BY d",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t2.k FROM t1 FULL OUTER JOIN t2 ON t1.a = t2.k",
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2",
            "SELECT * FROM t1 ORDER BY id",
            "SELECT id, c FROM t1 ORDER BY c NULLS FIRST, id LIMIT 30",
            "SELECT COUNT(*) FROM (SELECT * FROM t1 LIMIT 37)",
            "SELECT * FROM (VALUES (1, 'a'), (2, NULL), (NULL, 'c')) AS v(x, y)",
        ] {
            assert_native_scans_byte_eq(sql).await;
        }
        // Both scans of a join are lowered.
        assert_eq!(
            assert_native_scans_byte_eq(
                "SELECT t1.id, t2.w FROM t1 JOIN t2 ON t1.a = t2.k"
            )
            .await,
            2
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_skewed_partitions_and_thread_counts() {
        let partitions = skewed_partitions(30_000);
        let setup = async |ctx: &SessionContext| {
            register_test_tables(ctx)?;
            ctx.register_table(
                "skewed",
                Arc::new(MemTable::try_new(schema(), partitions.clone())?),
            )?;
            Ok(())
        };
        for target_partitions in [1, 3, 4, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT * FROM skewed",
                "SELECT s, v FROM skewed WHERE v % 7 = 3",
                "SELECT s, COUNT(*), SUM(v) FROM skewed GROUP BY s",
                "SELECT * FROM t1",
                "SELECT e, COUNT(DISTINCT a) FROM t1 GROUP BY e",
            ] {
                assert_native_scans_byte_eq_with_config(sql, config.clone(), &setup)
                    .await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_limits() {
        // A `LIMIT` over a single partition is pushed into the scan, which
        // then produces exactly DataFusion's first rows, on all threads.
        // (Without an `ORDER BY`, the rows an `OFFSET` skips depend on the
        // order in which the threads read the slices, as with several
        // partitions in DataFusion.)
        let setup = async |ctx: &SessionContext| {
            let single = vec![vec![batch(0..2500), batch(2500..9000)]];
            ctx.register_table("single", Arc::new(MemTable::try_new(schema(), single)?))?;
            ctx.register_table(
                "skewed",
                Arc::new(MemTable::try_new(schema(), skewed_partitions(10_000))?),
            )?;
            Ok(())
        };
        for sql in [
            "SELECT * FROM single LIMIT 5000",
            "SELECT * FROM single ORDER BY v LIMIT 10 OFFSET 2495",
            "SELECT v FROM single LIMIT 0",
            "SELECT COUNT(*), MIN(v) FROM (SELECT v FROM skewed LIMIT 3000)",
        ] {
            assert_native_scans_byte_eq_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_schemas_with_metadata_and_stricter_batches() {
        // The table schema is nullable and has metadata; the batches are not
        // nullable, have nested fields that are not nullable and no
        // metadata. DataFusion adapts them to the table schema.
        let metadata = HashMap::from([("origin".to_string(), "test".to_string())]);
        let inner = Fields::from(vec![Field::new("x", DataType::Int32, true)]);
        let table_schema = Arc::new(
            Schema::new(vec![
                Field::new("v", DataType::Int64, true).with_metadata(metadata.clone()),
                Field::new("st", DataType::Struct(inner), true),
            ])
            .with_metadata(metadata),
        );
        let strict_inner = Fields::from(vec![Field::new("x", DataType::Int32, false)]);
        let batch_schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, false),
            Field::new("st", DataType::Struct(strict_inner.clone()), false),
        ]));
        let batch = RecordBatch::try_new(
            batch_schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..3000)),
                Arc::new(StructArray::new(
                    strict_inner,
                    vec![Arc::new(Int32Array::from_iter_values(0..3000))],
                    None,
                )),
            ],
        )
        .unwrap();
        let setup = async |ctx: &SessionContext| {
            let partitions =
                vec![vec![batch.slice(0, 2500)], vec![batch.slice(2500, 500)]];
            ctx.register_table(
                "strict",
                Arc::new(MemTable::try_new(Arc::clone(&table_schema), partitions)?),
            )?;
            Ok(())
        };
        for sql in [
            "SELECT * FROM strict",
            "SELECT st, v FROM strict",
            "SELECT st FROM strict WHERE v > 2000",
        ] {
            assert_native_scans_byte_eq_with_setup(sql, &setup).await;
        }
    }
}
