// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Differential test harness: proves that the pipeline engine produces
//! exactly the same results as default DataFusion.
//!
//! Every query is run on two [`SessionContext`]s that are configured
//! identically, except that the second one plans queries with
//! [`PipelineQueryPlanner`]. [`assert_byte_eq_with_datafusion`] then checks
//! that
//!
//! 1. the output schemas are exactly equal (field names, data types,
//!    nullability and metadata, at the field and at the schema level), both
//!    for the plans and for every produced batch, and
//! 2. the Arrow IPC serialization of the results is byte-for-byte equal.
//!
//! Before serialization, the output of each engine is brought into a
//! canonical form, which removes the differences that are not part of the
//! result itself:
//!
//! * the batches are concatenated into a single batch, so batch boundaries
//!   do not matter;
//! * if the query has no top-level `ORDER BY` (see
//!   [`has_top_level_order_by`]), the rows are sorted, so row order does not
//!   matter; with an `ORDER BY` the rows are compared in their exact order;
//! * the arrays are rebuilt through the Arrow row format, which normalizes
//!   their physical layout (bytes under null slots, buffer layout of view
//!   arrays, offsets of sliced arrays, dictionaries are replaced by their
//!   values). Values themselves are compared bit for bit, e.g. `-0.0` and
//!   `0.0` or different `NaN`s are different.
//!
//! Queries must have a deterministic result: no `LIMIT` without a total
//! `ORDER BY`, no `ORDER BY` with ties, and no floating point aggregates
//! whose result depends on the summation order.
//!
//! # Usage
//!
//! Unit tests in this crate use the harness directly:
//!
//! ```ignore
//! #[tokio::test]
//! async fn filter() {
//!     crate::test_utils::assert_byte_eq_with_datafusion("SELECT a FROM t1 WHERE a > 3").await;
//! }
//! ```
//!
//! The queries can use the tables registered by [`register_test_tables`].
//! Use [`assert_byte_eq_with_datafusion_with_setup`] to register other
//! tables.
//!
//! Integration tests (`tests/*.rs`) can include this file as a module: it
//! only uses the public API of the crate and dev-dependencies.
//!
//! ```ignore
//! #[path = "../src/test_utils.rs"]
//! mod test_utils;
//! ```

use std::fmt::Write;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow::util::pretty::pretty_format_batches;
use arrow_ipc::writer::StreamWriter;
use arrow_row::{RowConverter, SortField};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::physical_planner::DefaultPhysicalPlanner;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::LogicalPlan;
use datafusion_physical_pipeline::{PipelineQueryPlanner, PipelineRootExec};
use datafusion_physical_plan::{ExecutionPlan, collect};

/// The configuration of both sessions used by the harness.
pub fn test_session_config() -> SessionConfig {
    SessionConfig::new().with_target_partitions(4)
}

/// A `SessionContext` that runs queries with default DataFusion.
pub fn datafusion_context(config: SessionConfig) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(Arc::new(RuntimeEnv::default()))
        .with_default_features()
        .build();
    SessionContext::new_with_state(state)
}

/// A `SessionContext` that runs queries with the pipeline engine.
pub fn pipeline_context(config: SessionConfig) -> SessionContext {
    let planner = PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(Arc::new(RuntimeEnv::default()))
        .with_default_features()
        .with_query_planner(Arc::new(planner))
        .build();
    SessionContext::new_with_state(state)
}

/// The result of running a query on one engine.
#[derive(Debug)]
pub struct EngineOutput {
    /// The physical plan that was executed.
    pub plan: Arc<dyn ExecutionPlan>,
    /// The output schema of the plan.
    pub schema: SchemaRef,
    /// The output of the plan.
    pub batches: Vec<RecordBatch>,
    /// Whether the query has a top-level `ORDER BY`.
    pub ordered: bool,
}

/// Plan and run `sql` on `ctx`.
pub async fn run_sql(ctx: &SessionContext, sql: &str) -> Result<EngineOutput> {
    let df = ctx.sql(sql).await?;
    let ordered = has_top_level_order_by(df.logical_plan());
    let plan = df.create_physical_plan().await?;
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    Ok(EngineOutput {
        schema: plan.schema(),
        plan,
        batches,
        ordered,
    })
}

/// Returns `true` if the result of `plan` has a defined order, i.e. its
/// root (looking through projections, limits and aliases) is a `Sort`.
pub fn has_top_level_order_by(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Sort(_) => true,
        LogicalPlan::Projection(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::SubqueryAlias(_) => plan
            .inputs()
            .first()
            .is_some_and(|input| has_top_level_order_by(input)),
        _ => false,
    }
}

/// Run `sql` with default DataFusion and with the pipeline engine, after
/// registering [`register_test_tables`] in both sessions.
///
/// Returns the output batches of DataFusion and of the pipeline engine.
pub async fn run_sql_with_both_engines(
    sql: &str,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
    run_sql_with_both_engines_with_setup(sql, async |ctx: &SessionContext| {
        register_test_tables(ctx)
    })
    .await
}

/// Like [`run_sql_with_both_engines`], but `setup` registers the tables
/// used by the query (it is called once for each session).
pub async fn run_sql_with_both_engines_with_setup<F>(
    sql: &str,
    setup: F,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)>
where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    let (expected, actual) = run_engines(sql, test_session_config(), setup).await?;
    Ok((expected.batches, actual.batches))
}

/// Run `sql` on both engines, configured with `config` and set up with
/// `setup`. Fails if the pipeline engine did not plan the query.
pub async fn run_engines<F>(
    sql: &str,
    config: SessionConfig,
    setup: F,
) -> Result<(EngineOutput, EngineOutput)>
where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    let datafusion = datafusion_context(config.clone());
    setup(&datafusion).await?;
    let pipeline = pipeline_context(config);
    setup(&pipeline).await?;

    let expected = run_sql(&datafusion, sql).await?;
    let actual = run_sql(&pipeline, sql).await?;
    if !actual.plan.is::<PipelineRootExec>() {
        return Err(DataFusionError::Internal(format!(
            "The pipeline engine did not plan the query, its root is {}",
            actual.plan.name()
        )));
    }
    Ok((expected, actual))
}

/// Assert that `sql` produces byte-identical results on default DataFusion
/// and on the pipeline engine. The query can use the tables of
/// [`register_test_tables`].
pub async fn assert_byte_eq_with_datafusion(sql: &str) {
    assert_byte_eq_with_datafusion_with_setup(sql, async |ctx: &SessionContext| {
        register_test_tables(ctx)
    })
    .await;
}

/// Like [`assert_byte_eq_with_datafusion`], but `setup` registers the tables
/// used by the query (it is called once for each session).
pub async fn assert_byte_eq_with_datafusion_with_setup<F>(sql: &str, setup: F)
where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    assert_byte_eq_with_datafusion_with_config(sql, test_session_config(), setup).await;
}

/// Like [`assert_byte_eq_with_datafusion_with_setup`], with a custom session
/// configuration for both engines.
pub async fn assert_byte_eq_with_datafusion_with_config<F>(
    sql: &str,
    config: SessionConfig,
    setup: F,
) where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    let (expected, actual) = match run_engines(sql, config, setup).await {
        Ok(outputs) => outputs,
        Err(e) => panic!("Failed to run query on both engines: {e}\nquery: {sql}"),
    };
    if let Err(message) = compare_outputs(&expected, &actual) {
        panic!("Results differ: {message}\nquery: {sql}");
    }
}

/// Compare the output of DataFusion (`expected`) with the output of the
/// pipeline engine (`actual`), see the [module documentation](self).
pub fn compare_outputs(
    expected: &EngineOutput,
    actual: &EngineOutput,
) -> std::result::Result<(), String> {
    if expected.schema != actual.schema {
        return Err(format!(
            "plan schemas differ\nexpected: {:#?}\nactual: {:#?}",
            expected.schema, actual.schema
        ));
    }
    // DataFusion's batches are the reference for the schema of the batches.
    let batch_schema = expected
        .batches
        .first()
        .map_or_else(|| Arc::clone(&expected.schema), |batch| batch.schema());
    if let Some(batch) = actual
        .batches
        .iter()
        .find(|batch| batch.schema() != batch_schema)
    {
        return Err(format!(
            "batch schemas differ\nexpected: {batch_schema:#?}\nactual: {:#?}",
            batch.schema()
        ));
    }

    let ordered = expected.ordered;
    let canonical = |batches: &[RecordBatch]| {
        canonical_batch(&batch_schema, batches, ordered).map_err(|e| e.to_string())
    };
    let expected_batch = canonical(&expected.batches)?;
    let actual_batch = canonical(&actual.batches)?;
    let expected_bytes = ipc_bytes(&expected_batch).map_err(|e| e.to_string())?;
    let actual_bytes = ipc_bytes(&actual_batch).map_err(|e| e.to_string())?;
    if expected_bytes != actual_bytes {
        return Err(format!(
            "IPC bytes differ (ordered: {ordered}, rows: {} vs {})\nexpected:\n{}\nactual:\n{}",
            expected_batch.num_rows(),
            actual_batch.num_rows(),
            preview(&expected_batch),
            preview(&actual_batch)
        ));
    }
    Ok(())
}

/// The canonical Arrow IPC stream serialization of `batches` (with schema
/// `schema`), see the [module documentation](self).
pub fn canonical_ipc_bytes(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    ordered: bool,
) -> Result<Vec<u8>> {
    ipc_bytes(&canonical_batch(schema, batches, ordered)?)
}

/// Concatenate `batches`, sort the rows unless `ordered`, and normalize the
/// physical layout of the arrays.
pub fn canonical_batch(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    ordered: bool,
) -> Result<RecordBatch> {
    let batch = concat_batches(schema, batches)?;
    if batch.num_columns() == 0 {
        return Ok(batch);
    }
    let sort_fields = schema
        .fields()
        .iter()
        .map(|field| SortField::new(field.data_type().clone()))
        .collect::<Vec<_>>();
    if !RowConverter::supports_fields(&sort_fields) {
        return sort_by_display(&batch, ordered);
    }
    let converter = RowConverter::new(sort_fields)?;
    let rows = converter.convert_columns(batch.columns())?;
    let mut indices = (0..rows.num_rows()).collect::<Vec<_>>();
    if !ordered {
        indices.sort_unstable_by(|&a, &b| rows.row(a).cmp(&rows.row(b)));
    }
    let columns = converter.convert_rows(indices.iter().map(|&i| rows.row(i)))?;
    // Dictionaries are decoded to their value type: adjust the schema.
    let fields = schema
        .fields()
        .iter()
        .zip(&columns)
        .map(|(field, column)| {
            Field::new(
                field.name(),
                column.data_type().clone(),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone())
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        schema, columns, &options,
    )?)
}

/// Fallback for types the row format does not support: sort by the display
/// representation of the rows.
fn sort_by_display(batch: &RecordBatch, ordered: bool) -> Result<RecordBatch> {
    if ordered {
        return Ok(batch.clone());
    }
    let options = FormatOptions::default().with_null("NULL");
    let formatters = batch
        .columns()
        .iter()
        .map(|column| ArrayFormatter::try_new(column.as_ref(), &options))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let keys = (0..batch.num_rows())
        .map(|row| {
            formatters
                .iter()
                .map(|formatter| formatter.value(row).to_string())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut indices = (0..batch.num_rows()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
    let indices = UInt32Array::from_iter_values(indices.into_iter().map(|i| i as u32));
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<ArrayRef>, _>>()?;
    Ok(RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

fn ipc_bytes(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut bytes = vec![];
    let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema())?;
    writer.write(batch)?;
    writer.finish()?;
    drop(writer);
    Ok(bytes)
}

fn preview(batch: &RecordBatch) -> String {
    const MAX_ROWS: usize = 20;
    let rows = batch.num_rows().min(MAX_ROWS);
    let mut out = pretty_format_batches(&[batch.slice(0, rows)]).map_or_else(
        |e| format!("<cannot format batch: {e}>"),
        |table| table.to_string(),
    );
    if batch.num_rows() > MAX_ROWS {
        let _ = write!(out, "\n... {} more rows", batch.num_rows() - MAX_ROWS);
    }
    out
}

/// Register the default test tables in `ctx`:
///
/// * `t1`: 5000 rows in 4 skewed partitions (one of them empty), with
///   columns of many types and nulls: `id BIGINT NOT NULL` (unique, `0..5000`),
///   `a INT`, `b BIGINT`, `c DOUBLE` (with `NaN` and `-0.0`), `d VARCHAR`,
///   `e BOOLEAN`, `f DATE`, `g DECIMAL(12, 2)`, `h` (`Utf8View`) and `ts`
///   (`TIMESTAMP`).
/// * `t2`: 150 rows in 3 partitions: `k INT` (joins with `t1.a`, with
///   duplicates and nulls), `name VARCHAR` and `w BIGINT`.
/// * `t_empty`: a table with the schema of `t1` and no rows.
pub fn register_test_tables(ctx: &SessionContext) -> Result<()> {
    let t1 = test_table_t1()?;
    let schema = t1[0][0].schema();
    ctx.register_table("t1", Arc::new(MemTable::try_new(Arc::clone(&schema), t1)?))?;
    let t2 = test_table_t2()?;
    let t2_schema = t2[0][0].schema();
    ctx.register_table("t2", Arc::new(MemTable::try_new(t2_schema, t2)?))?;
    ctx.register_table(
        "t_empty",
        Arc::new(MemTable::try_new(schema, vec![vec![]])?),
    )?;
    Ok(())
}

/// The partitions of table `t1` of [`register_test_tables`].
pub fn test_table_t1() -> Result<Vec<Vec<RecordBatch>>> {
    use arrow::array::{
        BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array, Int64Array,
        StringArray, StringViewArray, TimestampNanosecondArray,
    };
    use arrow::datatypes::{DataType, TimeUnit};

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int64, true),
        Field::new("c", DataType::Float64, true),
        Field::new("d", DataType::Utf8, true),
        Field::new("e", DataType::Boolean, true),
        Field::new("f", DataType::Date32, true),
        Field::new("g", DataType::Decimal128(12, 2), true),
        Field::new("h", DataType::Utf8View, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ]));
    let words = ["alpha", "beta", "gamma", "delta", "epsilon", ""];
    let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
        let id = Int64Array::from_iter_values(ids.clone());
        let a = Int32Array::from_iter(
            ids.clone()
                .map(|i| (i % 13 != 7).then_some(((i * 7) % 101) as i32 - 50)),
        );
        let b = Int64Array::from_iter(
            ids.clone()
                .map(|i| (i % 17 != 3).then_some((i * 1_000_003) % 9973)),
        );
        let c = Float64Array::from_iter(ids.clone().map(|i| match i {
            _ if i % 19 == 11 => None,
            _ if i % 211 == 0 => Some(f64::NAN),
            _ if i % 97 == 5 => Some(-0.0),
            _ if i % 89 == 1 => Some(0.0),
            _ => Some(((i * 31) % 1000) as f64 / 8.0 - 60.0),
        }));
        let d = StringArray::from_iter(
            ids.clone()
                .map(|i| (i % 23 != 4).then(|| words[(i % 6) as usize])),
        );
        let e = BooleanArray::from_iter(
            ids.clone().map(|i| (i % 29 != 2).then_some(i % 3 == 0)),
        );
        let f = Date32Array::from_iter(
            ids.clone()
                .map(|i| (i % 31 != 9).then_some(18_000 + (i % 400) as i32)),
        );
        let g =
            Decimal128Array::from_iter(ids.clone().map(|i| {
                (i % 37 != 1).then_some(i128::from((i * 37) % 100_000) - 50_000)
            }))
            .with_precision_and_scale(12, 2)?;
        let h = StringViewArray::from_iter(ids.clone().map(|i| {
            (i % 41 != 6).then(|| format!("a fairly long view string {:04}", i % 300))
        }));
        let ts = TimestampNanosecondArray::from_iter(ids.map(|i| {
            (i % 43 != 8).then_some(1_700_000_000_000_000_000 + i * 3_600_000_000_123)
        }));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(id),
            Arc::new(a),
            Arc::new(b),
            Arc::new(c),
            Arc::new(d),
            Arc::new(e),
            Arc::new(f),
            Arc::new(g),
            Arc::new(h),
            Arc::new(ts),
        ];
        Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
    };
    // Skewed partitions: large batches (sliced by the pipeline engine),
    // small batches, an empty partition and a partition with an empty batch.
    Ok(vec![
        vec![batch(0..2600)?, batch(2600..3000)?],
        vec![],
        vec![batch(3000..3100)?, batch(3100..4990)?, batch(4990..4990)?],
        vec![batch(4990..5000)?],
    ])
}

/// The partitions of table `t2` of [`register_test_tables`].
pub fn test_table_t2() -> Result<Vec<Vec<RecordBatch>>> {
    use arrow::array::{Int32Array, Int64Array, StringArray};
    use arrow::datatypes::DataType;

    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("w", DataType::Int64, true),
    ]));
    let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
        // Keys -60..=60 with duplicates for the even keys and some nulls.
        let k = Int32Array::from_iter(
            ids.clone()
                .map(|i| (i % 11 != 10).then_some(((i * 5) % 121) as i32 - 60)),
        );
        let name = StringArray::from_iter(
            ids.clone()
                .map(|i| (i % 7 != 3).then(|| format!("name-{}", i % 40))),
        );
        let w = Int64Array::from_iter(ids.map(|i| (i % 5 != 1).then_some(i * i - 70)));
        Ok(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(k), Arc::new(name), Arc::new(w)],
        )?)
    };
    Ok(vec![
        vec![batch(0..50)?],
        vec![batch(50..120)?],
        vec![batch(120..150)?],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use arrow::array::{
        DictionaryArray, Float64Array, Int32Array, StringArray, StringViewArray,
    };
    use arrow::buffer::{NullBuffer, ScalarBuffer};
    use arrow::datatypes::{DataType, Int32Type};

    fn int_batch(values: &[Option<i32>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.to_vec()))])
            .unwrap()
    }

    fn bytes(batches: &[RecordBatch], ordered: bool) -> Vec<u8> {
        canonical_ipc_bytes(&batches[0].schema(), batches, ordered).unwrap()
    }

    fn output(batches: Vec<RecordBatch>, ordered: bool) -> EngineOutput {
        let schema = batches[0].schema();
        EngineOutput {
            plan: Arc::new(datafusion_physical_plan::empty::EmptyExec::new(Arc::clone(
                &schema,
            ))),
            schema,
            batches,
            ordered,
        }
    }

    #[test]
    fn canonical_bytes_ignore_batch_boundaries_and_row_order() {
        let one = int_batch(&[Some(3), None, Some(1), Some(2), Some(1)]);
        let split = [
            int_batch(&[Some(1), Some(2)]),
            int_batch(&[]),
            int_batch(&[Some(1), None, Some(3)]),
        ];
        assert_eq!(
            bytes(std::slice::from_ref(&one), false),
            bytes(&split, false)
        );
        // With an ORDER BY the row order matters.
        assert_ne!(bytes(std::slice::from_ref(&one), true), bytes(&split, true));
        let same_order = [one.slice(0, 2), one.slice(2, 3)];
        assert_eq!(bytes(&[one], true), bytes(&same_order, true));
    }

    #[test]
    fn canonical_bytes_detect_different_values() {
        let expected = bytes(&[int_batch(&[Some(1), Some(2), None])], false);
        assert_ne!(
            expected,
            bytes(&[int_batch(&[Some(1), Some(3), None])], false)
        );
        assert_ne!(
            expected,
            bytes(&[int_batch(&[Some(1), Some(2), Some(0)])], false)
        );
        assert_ne!(expected, bytes(&[int_batch(&[Some(1), Some(2)])], false));

        // Floats are compared bit for bit.
        let floats = |values: Vec<Option<f64>>| {
            let schema =
                Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, true)]));
            let batch =
                RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))])
                    .unwrap();
            bytes(&[batch], false)
        };
        assert_ne!(floats(vec![Some(0.0)]), floats(vec![Some(-0.0)]));
        assert_ne!(floats(vec![Some(f64::NAN)]), floats(vec![None]));
        assert_eq!(
            floats(vec![Some(f64::NAN), Some(1.0)]),
            floats(vec![Some(1.0), Some(f64::NAN)])
        );
    }

    #[test]
    fn canonical_bytes_normalize_physical_layout() {
        // Different values under null slots.
        let nulls = NullBuffer::from(vec![true, false, true]);
        let a = Int32Array::new(ScalarBuffer::from(vec![1, 0, 3]), Some(nulls.clone()));
        let b = Int32Array::new(ScalarBuffer::from(vec![1, 99, 3]), Some(nulls));
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        let batch = |array: Int32Array| {
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap()
        };
        assert_eq!(bytes(&[batch(a)], true), bytes(&[batch(b)], true));

        // View arrays with different buffer layouts.
        let long = |i: usize| format!("a string that does not fit inline {i}");
        let values = (0..10).map(long).collect::<Vec<_>>();
        let compact = StringViewArray::from_iter_values(values[2..6].iter());
        let sliced = StringViewArray::from_iter_values(values.iter()).slice(2, 4);
        let schema =
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
        let batch = |array: StringViewArray| {
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap()
        };
        assert_eq!(
            bytes(&[batch(compact)], true),
            bytes(&[batch(sliced.clone())], true)
        );
        assert_eq!(
            bytes(
                &[batch(sliced.slice(0, 2)), batch(sliced.slice(2, 2))],
                true
            ),
            bytes(&[batch(sliced)], true)
        );

        // Dictionaries with different dictionary values.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )]));
        let dict = |keys: Vec<i32>, values: Vec<&str>| {
            let array = DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(keys),
                Arc::new(StringArray::from(values)),
            )
            .unwrap();
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap()
        };
        assert_eq!(
            bytes(&[dict(vec![0, 1, 0], vec!["x", "y"])], true),
            bytes(&[dict(vec![2, 0, 2], vec!["y", "unused", "x"])], true)
        );
    }

    #[test]
    fn compare_outputs_checks_schemas() {
        let batch = int_batch(&[Some(1)]);
        assert!(
            compare_outputs(
                &output(vec![batch.clone()], false),
                &output(vec![batch.clone()], false)
            )
            .is_ok()
        );

        // Nullability.
        let not_null =
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let other =
            RecordBatch::try_new(not_null, vec![Arc::clone(batch.column(0))]).unwrap();
        let err = compare_outputs(
            &output(vec![batch.clone()], false),
            &output(vec![other], false),
        )
        .unwrap_err();
        assert!(err.contains("plan schemas differ"), "{err}");

        // Field metadata.
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
        let with_metadata = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int32, true).with_metadata(metadata.clone()),
        ]));
        let other =
            RecordBatch::try_new(with_metadata, vec![Arc::clone(batch.column(0))])
                .unwrap();
        assert!(
            compare_outputs(
                &output(vec![batch.clone()], false),
                &output(vec![other], false)
            )
            .is_err()
        );

        // Schema metadata of the batches (the plan schemas are equal).
        let expected = output(vec![batch.clone()], false);
        let mut actual = output(vec![batch.clone()], false);
        let schema = Arc::new(batch.schema().as_ref().clone().with_metadata(metadata));
        actual.batches = vec![batch.with_schema(schema).unwrap()];
        let err = compare_outputs(&expected, &actual).unwrap_err();
        assert!(err.contains("batch schemas differ"), "{err}");
    }

    #[test]
    fn compare_outputs_reports_differences() {
        let err = compare_outputs(
            &output(vec![int_batch(&[Some(1), Some(2)])], false),
            &output(vec![int_batch(&[Some(1), Some(5)])], false),
        )
        .unwrap_err();
        assert!(err.contains("IPC bytes differ"), "{err}");
        assert!(err.contains('5'), "{err}");
        assert!(
            compare_outputs(
                &output(vec![int_batch(&[Some(2), Some(1)])], false),
                &output(vec![int_batch(&[Some(1)]), int_batch(&[Some(2)])], false),
            )
            .is_ok()
        );
        assert!(
            compare_outputs(
                &output(vec![int_batch(&[Some(2), Some(1)])], true),
                &output(vec![int_batch(&[Some(1)]), int_batch(&[Some(2)])], true),
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn detects_top_level_order_by() -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        register_test_tables(&ctx)?;
        for (sql, ordered) in [
            ("SELECT id FROM t1", false),
            ("SELECT id FROM t1 ORDER BY id", true),
            ("SELECT a FROM t1 ORDER BY id DESC", true),
            ("SELECT id FROM t1 ORDER BY id LIMIT 3", true),
            // DataFusion ignores an ORDER BY without LIMIT in a subquery.
            (
                "SELECT * FROM (SELECT id FROM t1 ORDER BY id) LIMIT 3",
                false,
            ),
            (
                "SELECT * FROM (SELECT id FROM t1 ORDER BY id LIMIT 10) AS s LIMIT 3",
                true,
            ),
            ("SELECT id FROM t1 LIMIT 3", false),
            ("SELECT d, COUNT(*) FROM t1 GROUP BY d", false),
            ("SELECT id FROM t1 UNION ALL SELECT id FROM t1", false),
        ] {
            let df = ctx.sql(sql).await?;
            assert_eq!(
                has_top_level_order_by(df.logical_plan()),
                ordered,
                "{sql}\n{}",
                df.logical_plan().display_indent()
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn pipeline_context_plans_with_pipeline_root() -> Result<()> {
        let ctx = pipeline_context(test_session_config());
        register_test_tables(&ctx)?;
        let output = run_sql(&ctx, "SELECT id, a FROM t1 WHERE a > 0").await?;
        let root = output
            .plan
            .downcast_ref::<PipelineRootExec>()
            .expect("PipelineRootExec");
        let metrics = root.execution_metrics().expect("the plan was executed");
        let rows: usize = output.batches.iter().map(RecordBatch::num_rows).sum();
        let (_, collector) = metrics
            .iter()
            .find(|(operator, _)| operator.name() == "CollectorSink")
            .expect("CollectorSink metrics");
        assert_eq!(collector.sink_rows(), rows);
        Ok(())
    }

    #[tokio::test]
    async fn run_sql_with_both_engines_returns_both_results() -> Result<()> {
        let (expected, actual) =
            run_sql_with_both_engines("SELECT id, d FROM t1 WHERE e").await?;
        let rows = |batches: &[RecordBatch]| {
            batches.iter().map(RecordBatch::num_rows).sum::<usize>()
        };
        assert!(rows(&expected) > 0);
        assert_eq!(rows(&expected), rows(&actual));

        let (expected, actual) = run_sql_with_both_engines_with_setup(
            "SELECT * FROM my_table",
            async |ctx: &SessionContext| {
                ctx.register_batch("my_table", int_batch(&[Some(1), None]))?;
                Ok(())
            },
        )
        .await?;
        assert_eq!(expected, actual);
        assert!(
            run_sql_with_both_engines("SELECT * FROM missing")
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    #[should_panic(expected = "Failed to run query on both engines")]
    async fn assert_byte_eq_panics_on_invalid_query() {
        assert_byte_eq_with_datafusion("SELECT no_such_column FROM t1").await;
    }

    async fn assert_all_byte_eq(queries: &[&str]) {
        for sql in queries {
            assert_byte_eq_with_datafusion(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_scans_filters_and_projections() {
        assert_all_byte_eq(&[
            "SELECT * FROM t1",
            "SELECT * FROM t2",
            "SELECT * FROM t_empty",
            "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
            "SELECT id, a + b AS s, c * 2 AS c2, upper(d) AS u, NOT e AS ne FROM t1",
            "SELECT id, CASE WHEN a IS NULL THEN 'none' WHEN a < 0 THEN 'neg' ELSE 'pos' END AS sign FROM t1",
            "SELECT id, CAST(g AS DOUBLE) AS gd, f + INTERVAL '1 day' AS next_day, ts FROM t1 WHERE f > DATE '1970-01-01'",
            "SELECT id, length(h) AS len, concat(d, h) AS cat FROM t1 WHERE h LIKE '%01%'",
            "SELECT id FROM t1 WHERE c IS NULL OR c = 0.0 OR isnan(c)",
            "SELECT * FROM t1 WHERE id < 0",
            "SELECT 1 AS one, 'hello' AS greeting, NULL AS nothing",
            "SELECT * FROM (VALUES (1, 'a'), (2, NULL), (NULL, 'c')) AS v(x, y)",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_aggregates_and_windows() {
        assert_all_byte_eq(&[
            "SELECT COUNT(*), COUNT(a), SUM(b), MIN(c), MAX(d), AVG(g) FROM t1",
            "SELECT COUNT(*), SUM(a) FROM t_empty",
            "SELECT d, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh, MAX(f) AS mf FROM t1 GROUP BY d",
            "SELECT e, f, COUNT(DISTINCT a) AS da FROM t1 GROUP BY e, f",
            "SELECT DISTINCT d, e FROM t1",
            "SELECT a, COUNT(*) FROM t1 GROUP BY a HAVING COUNT(*) > 50",
            "SELECT d, SUM(g) FROM t1 GROUP BY ROLLUP (d)",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
            "SELECT id, SUM(b) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
            "SELECT id, LAG(a) OVER (ORDER BY id) AS prev, RANK() OVER (PARTITION BY e ORDER BY a, id) AS r FROM t1",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_joins_and_subqueries() {
        assert_all_byte_eq(&[
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t2.w FROM t1 LEFT JOIN t2 ON t1.a = t2.k AND t2.w > 100",
            "SELECT t1.id, t2.k FROM t1 FULL OUTER JOIN t2 ON t1.a = t2.k",
            "SELECT t2.k, t2.name FROM t2 WHERE k IN (SELECT a FROM t1 WHERE e)",
            "SELECT t2.k FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k)",
            "SELECT t2.k, (SELECT COUNT(*) FROM t1 WHERE t1.a = t2.k) AS cnt FROM t2",
            "SELECT a.k, b.k FROM t2 a CROSS JOIN t2 b WHERE a.w < 0",
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2",
            "SELECT d FROM t1 UNION SELECT name FROM t2",
            "SELECT id FROM t1 EXCEPT SELECT id FROM t1 WHERE a > 0",
            "WITH x AS (SELECT a, COUNT(*) AS n FROM t1 GROUP BY a) SELECT x.a, x.n, t2.name FROM x JOIN t2 ON x.a = t2.k",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_ordered_results() {
        assert_all_byte_eq(&[
            "SELECT * FROM t1 ORDER BY id",
            "SELECT id, a FROM t1 ORDER BY id DESC LIMIT 25",
            "SELECT id, c FROM t1 ORDER BY c NULLS FIRST, id",
            "SELECT id, h FROM t1 ORDER BY h DESC NULLS LAST, id LIMIT 100 OFFSET 7",
            "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY d",
            "SELECT * FROM t_empty ORDER BY id",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t1.id, t2.name, t2.w LIMIT 50",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_with_custom_setup_and_metadata() {
        let metadata = HashMap::from([("origin".to_string(), "test".to_string())]);
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("v", DataType::Int32, true).with_metadata(metadata.clone()),
                Field::new("s", DataType::Utf8View, false),
            ])
            .with_metadata(metadata),
        );
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter(
                    (0..3000).map(|i| (i % 3 != 0).then_some(i)),
                )),
                Arc::new(StringViewArray::from_iter_values(
                    (0..3000).map(|i| format!("value number {i}")),
                )),
            ],
        )
        .unwrap();
        let setup = async |ctx: &SessionContext| {
            let partitions =
                vec![vec![batch.slice(0, 2500)], vec![batch.slice(2500, 500)]];
            ctx.register_table(
                "meta",
                Arc::new(MemTable::try_new(Arc::clone(&schema), partitions)?),
            )?;
            Ok(())
        };
        for sql in [
            "SELECT * FROM meta",
            "SELECT v, s FROM meta WHERE v > 100 ORDER BY v",
            "SELECT s, v + 1 AS w FROM meta",
        ] {
            assert_byte_eq_with_datafusion_with_setup(sql, &setup).await;
        }

        // Single partition plans and a single thread.
        assert_byte_eq_with_datafusion_with_config(
            "SELECT * FROM meta",
            test_session_config().with_target_partitions(1),
            &setup,
        )
        .await;
    }
}
