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
//! [`PhysicalPlaceholderRow`] is the native source for it. Its schema is
//! almost always empty: it then produces chunks without columns that only
//! carry a row count, exactly like `PlaceholderRowExec`. For a schema with
//! fields, `PlaceholderRowExec` emits `NULL` columns of type `Null`, which
//! do not match its own schema. The chunks of this source always have the
//! schema of the operator, as the pipeline engine requires: nullable fields
//! are `NULL`, non-nullable fields hold the default value of their type
//! (`0`, `''`, ...), so that the rows are valid for the operators above,
//! which never read them.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{ArrayRef, new_null_array};
use arrow::datatypes::{Field, SchemaRef};
use datafusion_common::{Result, ScalarValue};
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state,
};

/// A source that produces `rows` placeholder rows (one by default).
///
/// See the [module documentation](self) for the values of the rows.
#[derive(Debug)]
pub struct PhysicalPlaceholderRow {
    schema: SchemaRef,
    rows: usize,
}

impl PhysicalPlaceholderRow {
    /// A source of a single row with schema `schema`.
    ///
    /// Fails if the default value of the type of a non-nullable field cannot
    /// be created.
    pub fn try_new(schema: SchemaRef) -> Result<Self> {
        let source = Self { schema, rows: 1 };
        // Fail now rather than during the execution.
        source.chunk(1)?;
        Ok(source)
    }

    /// Produce `rows` rows instead of one. A `PlaceholderRowExec` with `n`
    /// partitions produces `n` rows, one per partition.
    pub fn with_rows(mut self, rows: usize) -> Self {
        self.rows = rows;
        self
    }

    /// Lower a DataFusion `PlaceholderRowExec` to a source that produces its
    /// rows, one per output partition. Returns `Ok(None)` for any other node,
    /// and for a `PlaceholderRowExec` whose rows this source cannot create
    /// (see [`Self::try_new`]), which can be executed by DataFusion instead.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        if !plan.is::<PlaceholderRowExec>() {
            return Ok(None);
        }
        let rows = plan.output_partitioning().partition_count();
        Ok(Self::try_new(plan.schema())
            .ok()
            .map(|source| source.with_rows(rows)))
    }

    /// The number of rows this source produces.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// A chunk of `len` placeholder rows.
    fn chunk(&self, len: usize) -> Result<DataChunk> {
        if self.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(Arc::clone(&self.schema), len);
        }
        let columns = self
            .schema
            .fields()
            .iter()
            .map(|field| placeholder_column(field, len))
            .collect::<Result<Vec<_>>>()?;
        DataChunk::try_new(Arc::clone(&self.schema), columns, None)
    }
}

/// `len` placeholder values for `field`: `NULL`s if it is nullable, the
/// default value of its type otherwise.
fn placeholder_column(field: &Field, len: usize) -> Result<ArrayRef> {
    if field.is_nullable() {
        Ok(new_null_array(field.data_type(), len))
    } else {
        ScalarValue::new_default(field.data_type())?.to_array_of_size(len)
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

    use arrow::array::{Array, RecordBatch};
    use arrow::datatypes::{DataType, Schema, TimeUnit, UnionFields, UnionMode};
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
                    PhysicalPlaceholderRow::try_new(Arc::clone(&schema))?.with_rows(rows);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn placeholder_rows_have_the_schema_of_the_operator() -> Result<()> {
        let schema = Arc::new(
            Schema::new(vec![
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
                Field::new("flag", DataType::Boolean, false),
                Field::new_list(
                    "list",
                    Field::new_list_field(DataType::Int32, true),
                    false,
                ),
                Field::new_struct(
                    "struct",
                    vec![
                        Field::new("x", DataType::Int32, false),
                        Field::new("y", DataType::Utf8, true),
                    ],
                    false,
                ),
                Field::new_dictionary("dict", DataType::Int32, DataType::Utf8, false),
                Field::new("null", DataType::Null, true),
            ])
            .with_metadata(HashMap::from([("key".to_string(), "value".to_string())])),
        );
        let source = PhysicalPlaceholderRow::try_new(Arc::clone(&schema))?.with_rows(3);
        let output = run(Arc::new(source), 4).await?;
        assert_eq!(output.len(), 1);
        let batch = &output[0];
        // A valid batch: the non-nullable columns have no nulls.
        assert_eq!(batch.schema(), schema);
        assert_eq!(batch.num_rows(), 3);
        for (field, column) in schema.fields().iter().zip(batch.columns()) {
            let nulls = if field.is_nullable() { 3 } else { 0 };
            assert_eq!(column.logical_null_count(), nulls, "{}", field.name());
        }
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

        // A non-nullable field without a default value: DataFusion executes
        // the node instead.
        let no_default = Arc::new(Schema::new(vec![Field::new(
            "u",
            DataType::Union(UnionFields::empty(), UnionMode::Sparse),
            false,
        )]));
        assert!(PhysicalPlaceholderRow::try_new(Arc::clone(&no_default)).is_err());
        let plan: Arc<dyn ExecutionPlan> = Arc::new(PlaceholderRowExec::new(no_default));
        assert!(PhysicalPlaceholderRow::try_lower(&plan)?.is_none());
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
        // Placeholder rows that SQL does not produce: one row per partition.
        for partitions in [1, 3] {
            let make = move || -> Result<Arc<dyn ExecutionPlan>> {
                Ok(Arc::new(
                    PlaceholderRowExec::new(Arc::new(Schema::empty()))
                        .with_partitions(partitions),
                ))
            };
            let root = assert_native_plan_byte_eq(make, false).await;
            assert!(root.is::<PhysicalPlaceholderRow>());
        }
        Ok(())
    }
}
