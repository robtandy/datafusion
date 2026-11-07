// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalPlaceholderRow`: source for `PlaceholderRowExec`.
//!
//! DataFusion plans a `PlaceholderRowExec` for a relation that produces a
//! single row whose values are never read: the input of a projection
//! without a `FROM` clause (`SELECT 1, 'hello'`), the result of a global
//! aggregation without aggregate functions, and the input of the constants
//! that replace an aggregation answered from statistics
//! (`SELECT COUNT(*) FROM t`). It produces one row per output partition.
//!
//! [`PhysicalPlaceholderRow`] produces exactly the batches of
//! `PlaceholderRowExec`, as chunks: one `Null` column named `placeholder_{i}`
//! per field of its schema, adapted to that schema the way the `MemoryStream`
//! of `PlaceholderRowExec` adapts them (only if the schema has the same
//! fields, with more metadata or nullability). For the usual empty schema,
//! these are chunks without columns that only carry a row count. For a
//! schema with other fields, the chunks do not have the schema of the
//! operator, unlike the chunks of other operators: neither do the batches of
//! `PlaceholderRowExec` (nor the chunks of a
//! [`BridgeOperator`](crate::bridge::BridgeOperator) executing one). So the
//! operators above see exactly the columns they see in DataFusion.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{ArrayRef, NullArray, RecordBatch, RecordBatchOptions};
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_common::nested_struct::adapt_batch_to_schema;
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state,
};

/// A source that produces `rows` placeholder rows (one by default), like a
/// `PlaceholderRowExec` with schema `schema` and `rows` partitions.
///
/// See the [module documentation](self) for the chunks it produces.
#[derive(Debug)]
pub struct PhysicalPlaceholderRow {
    schema: SchemaRef,
    rows: usize,
}

impl PhysicalPlaceholderRow {
    /// A source of a single row with schema `schema`.
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema, rows: 1 }
    }

    /// Produce `rows` rows instead of one. A `PlaceholderRowExec` with `n`
    /// partitions produces `n` rows, one per partition.
    pub fn with_rows(mut self, rows: usize) -> Self {
        self.rows = rows;
        self
    }

    /// Lower a DataFusion `PlaceholderRowExec` to a source that produces its
    /// rows, one per output partition. Returns `Ok(None)` for any other node.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        if !plan.is::<PlaceholderRowExec>() {
            return Ok(None);
        }
        let rows = plan.output_partitioning().partition_count();
        Ok(Some(Self::new(plan.schema()).with_rows(rows)))
    }

    /// The number of rows this source produces.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// A chunk of `len` placeholder rows: the batch of `PlaceholderRowExec`
    /// (`PlaceholderRowExec::data`), adapted to the schema of the operator
    /// under the same condition and in the same way as its `MemoryStream`
    /// adapts it.
    fn chunk(&self, len: usize) -> Result<DataChunk> {
        let fields = (0..self.schema.fields().len())
            .map(|i| Field::new(format!("placeholder_{i}"), DataType::Null, true))
            .collect::<Fields>();
        let columns = fields
            .iter()
            .map(|_| Arc::new(NullArray::new(len)) as ArrayRef)
            .collect();
        let options = RecordBatchOptions::new().with_row_count(Some(len));
        let batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields)),
            columns,
            &options,
        )?;
        let batch = if batch.schema_ref().as_ref() != self.schema.as_ref()
            && self.schema.contains(batch.schema_ref())
        {
            adapt_batch_to_schema(batch, &self.schema)?
        } else {
            batch
        };
        Ok(DataChunk::from_record_batch(batch))
    }
}

/// The global source state of a [`PhysicalPlaceholderRow`].
#[derive(Debug)]
struct PlaceholderRowState {
    /// The number of rows handed out so far (possibly more than `rows`).
    next: AtomicUsize,
    rows: usize,
}

impl GlobalSourceState for PlaceholderRowState {
    fn max_threads(&self) -> usize {
        self.rows.div_ceil(DEFAULT_VECTOR_SIZE).max(1)
    }
}

impl PhysicalOperator for PhysicalPlaceholderRow {
    fn name(&self) -> &str {
        "PhysicalPlaceholderRow"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        if self.rows == 1 {
            "PhysicalPlaceholderRow".to_string()
        } else {
            format!("PhysicalPlaceholderRow: rows={}", self.rows)
        }
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
        Ok(Arc::new(PlaceholderRowState {
            next: AtomicUsize::new(0),
            rows: self.rows,
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<PlaceholderRowState>(gstate.as_any(), self.name())?;
        let start = state.next.fetch_add(DEFAULT_VECTOR_SIZE, Ordering::Relaxed);
        if start >= state.rows {
            return Ok(SourceResult::Finished);
        }
        let len = (state.rows - start).min(DEFAULT_VECTOR_SIZE);
        Ok(SourceResult::Chunk(self.chunk(len)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use arrow::datatypes::{TimeUnit, UnionFields, UnionMode};
    use datafusion_execution::TaskContext;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::empty::EmptyExec;

    use crate::operators::union::test_support::{
        assert_native_plan_byte_eq, assert_parity, assert_parity_with_config, count, run,
    };
    use crate::test_utils::test_session_config;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn placeholder_produces_its_rows() -> Result<()> {
        let schema = Arc::new(Schema::empty());
        for rows in [0, 1, 3, 2048, 2049, 5000] {
            for threads in [1, 4] {
                let source =
                    PhysicalPlaceholderRow::new(Arc::clone(&schema)).with_rows(rows);
                assert_eq!(source.rows(), rows);
                let output = run(Arc::new(source), threads).await?;
                assert_eq!(
                    output.iter().map(RecordBatch::num_rows).sum::<usize>(),
                    rows,
                    "{rows} rows, {threads} threads"
                );
                assert!(output.iter().all(|batch| {
                    (1..=DEFAULT_VECTOR_SIZE).contains(&batch.num_rows())
                        && batch.schema() == schema
                }));
            }
        }
        Ok(())
    }

    /// The outcome of executing a `PlaceholderRowExec` with schema `schema`
    /// with DataFusion, and a [`PhysicalPlaceholderRow`] with that schema
    /// with the pipeline engine.
    async fn outcomes(
        schema: &SchemaRef,
    ) -> (Result<Vec<RecordBatch>>, Result<Vec<RecordBatch>>) {
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(PlaceholderRowExec::new(Arc::clone(schema)));
        let datafusion = collect(plan, Arc::new(TaskContext::default())).await;
        let native =
            run(Arc::new(PhysicalPlaceholderRow::new(Arc::clone(schema))), 4).await;
        (datafusion, native)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn placeholder_rows_are_the_batches_of_placeholder_row_exec() -> Result<()> {
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
        let fields = vec![
            Field::new("nullable", DataType::Int32, true),
            Field::new("count", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
            Field::new("view", DataType::Utf8View, true),
            Field::new("d", DataType::Decimal128(12, 2), false),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                false,
            ),
            Field::new_list("list", Field::new_list_field(DataType::Int32, true), false),
            Field::new_struct(
                "struct",
                vec![Field::new("x", DataType::Int32, false)],
                false,
            ),
            Field::new_dictionary("dict", DataType::Int32, DataType::Utf8, false),
            Field::new(
                "u",
                DataType::Union(UnionFields::empty(), UnionMode::Sparse),
                false,
            ),
        ];
        let placeholder = |nullable, metadata: HashMap<String, String>| {
            Field::new("placeholder_0", DataType::Null, nullable).with_metadata(metadata)
        };
        for schema in [
            Schema::empty(),
            Schema::new(fields.clone()),
            Schema::new(fields).with_metadata(metadata.clone()),
            // Schemas that only add metadata or nullability to the placeholder
            // fields: the batches are adapted to them.
            Schema::new(vec![placeholder(true, HashMap::new())]),
            Schema::new(vec![placeholder(true, metadata.clone())])
                .with_metadata(metadata.clone()),
            Schema::new(vec![placeholder(false, HashMap::new())]),
        ] {
            let schema = Arc::new(schema);
            let (expected, actual) = outcomes(&schema).await;
            let (expected, actual) = (expected?, actual?);
            assert_eq!(actual, expected, "{schema:?}");
            assert_eq!(actual[0].num_rows(), 1);
            assert_eq!(
                PhysicalPlaceholderRow::new(Arc::clone(&schema)).schema(),
                schema
            );
        }

        // DataFusion cannot adapt its batch without columns to an empty schema
        // with metadata (the adapted batch loses its row count), and neither
        // can the native source.
        let schema = Arc::new(Schema::empty().with_metadata(metadata));
        let (expected, actual) = outcomes(&schema).await;
        let message = "must either specify a row count or at least one column";
        assert!(expected.unwrap_err().to_string().contains(message));
        assert!(actual.unwrap_err().to_string().contains(message));
        Ok(())
    }

    #[test]
    fn lowering_of_placeholder_rows() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "count(*)",
            DataType::Int64,
            false,
        )]));
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(PlaceholderRowExec::new(Arc::clone(&schema)));
        let source = PhysicalPlaceholderRow::try_lower(&plan)?.expect("a source");
        assert_eq!(source.schema(), schema);
        assert_eq!(source.rows(), 1);
        assert_eq!(source.description(), "PhysicalPlaceholderRow");
        assert!(source.is_source() && source.parallel_source());
        assert!(source.children().is_empty());

        // One row per partition.
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(PlaceholderRowExec::new(Arc::clone(&schema)).with_partitions(3));
        let source = PhysicalPlaceholderRow::try_lower(&plan)?.expect("a source");
        assert_eq!(source.rows(), 3);
        assert_eq!(source.description(), "PhysicalPlaceholderRow: rows=3");

        let empty: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(schema));
        assert!(PhysicalPlaceholderRow::try_lower(&empty)?.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_constant_queries() {
        for sql in [
            "SELECT 1, 'hello'",
            "SELECT 1 AS one, 'hello' AS greeting, NULL AS nothing",
            "SELECT 1 + 2 AS three, CAST(NULL AS INT) AS n, 'x' || 'y' AS xy, \
             [1, 2] AS l, -0.0 AS z, DATE '2024-02-29' AS d",
            "SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3",
            // Aggregations answered from statistics: constants above a
            // placeholder row with the schema of the aggregation.
            "SELECT COUNT(*) FROM t1",
            "SELECT COUNT(*) AS n FROM t_empty",
        ] {
            let root = assert_parity(sql).await;
            assert!(count::<PhysicalPlaceholderRow>(&root) > 0, "{sql}");
            for target_partitions in [1, 16] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                assert_parity_with_config(sql, config).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_placeholder_plans() -> Result<()> {
        // Placeholder rows as the result of a plan, which SQL does not
        // produce: one row per partition, with schemas with fields too (for
        // an empty schema with metadata, DataFusion fails, see
        // `placeholder_rows_are_the_batches_of_placeholder_row_exec`).
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
        let schemas = [
            Schema::empty(),
            Schema::new(vec![Field::new("v", DataType::Int32, false)]),
            Schema::new(vec![
                Field::new("a", DataType::Utf8, true),
                Field::new("b", DataType::Float64, false),
            ])
            .with_metadata(metadata),
            // The schema of the placeholder batches of DataFusion.
            Schema::new(vec![Field::new("placeholder_0", DataType::Null, true)]),
        ];
        for schema in schemas {
            let schema = Arc::new(schema);
            for partitions in [1, 3] {
                let make = || -> Result<Arc<dyn ExecutionPlan>> {
                    Ok(Arc::new(
                        PlaceholderRowExec::new(Arc::clone(&schema))
                            .with_partitions(partitions),
                    ))
                };
                let root = assert_native_plan_byte_eq(make, false).await;
                assert!(root.is::<PhysicalPlaceholderRow>());
            }
        }
        Ok(())
    }
}
