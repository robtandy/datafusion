// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Differential test harness: proves that the pipeline engine produces the
//! same results as default DataFusion.
//!
//! Every query is run on two [`SessionContext`]s that are configured
//! identically, except that the second one plans queries with
//! [`PipelineQueryPlanner`]. The results of both engines are then compared
//! in one of two ways:
//!
//! * **by meaning**, with [`assert_same_results_as_datafusion`], as the
//!   correctness policy of the crate requires (`docs/compatibility.md`): the
//!   same SQL results, with exactly the same schema. New tests use this
//!   comparison.
//! * **byte for byte**, with [`assert_byte_eq_with_datafusion`], which
//!   predates the policy. It still works for queries whose results are
//!   unambiguous, which many tests use.
//!
//! # Comparison by meaning
//!
//! [`assert_same_results_as_datafusion`] (and its `_with_setup`,
//! `_with_config` and `_with_options` variants) checks that
//!
//! 1. both engines succeed, or both fail. If only one of them fails, the test
//!    fails and reports both outcomes, unless the caller allows error
//!    differences ([`ComparisonOptions::allow_error_differences`]). If both
//!    fail, their errors may differ: which error is raised is the engine's
//!    own.
//! 2. the output schemas are exactly equal: field names, data types,
//!    nullability and metadata, at the field and at the schema level, for
//!    the plans and for every batch of the pipeline engine.
//! 3. the rows are the same:
//!    * as a multiset if the query has no top-level `ORDER BY` (see
//!      [`has_top_level_order_by`]);
//!    * in order if it has one, except that the rows that tie on the
//!      `ORDER BY` keys may come in any order among themselves. Under
//!      `ORDER BY … LIMIT`, if the `LIMIT` cuts some of the rows that tie
//!      with the last row, the last rows may be any of the rows that tie
//!      with it, with their multiplicities; the same holds for the first
//!      rows and the `OFFSET`;
//!    * their values are compared by meaning, not by representation: batch
//!      boundaries, dictionary and run-end encoding, views, offsets and the
//!      bytes under nulls do not matter;
//!    * floating-point values (also inside lists, structs, maps and unions)
//!      are equal within a relative tolerance:
//!      [`ComparisonOptions::DEFAULT_FLOAT_TOLERANCE`] (`1e-9`) by default,
//!      see [`ComparisonOptions::with_float_tolerance`]. `NaN` equals `NaN`
//!      and `-0.0` equals `0.0`, but `NULL` only equals `NULL`. `ORDER BY`
//!      keys that are equal within twice the tolerance tie, as the keys of
//!      each engine may be off by the tolerance.
//!
//! To know which rows tie and which rows `LIMIT` and `OFFSET` cut, DataFusion
//! also runs the query without its `LIMIT` and `OFFSET`, with the values of
//! its `ORDER BY` keys after its output columns (also of the keys that the
//! query does not select), and both results are compared with these sorted
//! rows, its [`SortedReference`] (see [`result_order`]). The rows that tie on
//! the keys before a volatile key (e.g. `random()`) may come in any order. If
//! the sorted rows cannot be computed (e.g. for a `LIMIT` that is not a
//! literal), the rows are compared in their exact order.
//!
//! Queries must still define their results: no `LIMIT` without `ORDER BY`,
//! and no aggregates that depend on the order of their input without an
//! `ORDER BY` (`ARRAY_AGG`, `STRING_AGG`, `FIRST_VALUE`, ...).
//!
//! # Byte equality
//!
//! [`assert_byte_eq_with_datafusion`] checks that
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
//!     crate::test_utils::assert_same_results_as_datafusion("SELECT a FROM t1 WHERE a > 3").await;
//! }
//! ```
//!
//! The queries can use the tables registered by [`register_test_tables`].
//! Use [`assert_same_results_as_datafusion_with_setup`] to register other
//! tables, [`assert_same_results_as_datafusion_with_config`] to configure the
//! sessions, and [`assert_same_results_as_datafusion_with_options`] to
//! configure the comparison. [`run_both_engines`] and
//! [`EngineOutcomes::compare`] are the building blocks of these assertions,
//! for tests that also inspect the outputs or the plans.
//!
//! Integration tests (`tests/*.rs`) can include this file as a module: it
//! only uses the public API of the crate and dev-dependencies.
//!
//! ```ignore
//! #[path = "../src/test_utils.rs"]
//! mod test_utils;
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, RecordBatch, RecordBatchOptions, RunArray, UInt32Array,
};
use arrow::compute::{cast, concat_batches, take};
use arrow::datatypes::{
    ArrowNativeType, DataType, Field, Float16Type, Float32Type, Float64Type, Int16Type,
    Int32Type, Int64Type, RunEndIndexType, Schema, SchemaRef,
};
use arrow::downcast_dictionary_array;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow::util::pretty::pretty_format_batches;
use arrow_ipc::writer::StreamWriter;
use arrow_row::{RowConverter, SortField};
use datafusion::dataframe::DataFrame;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::physical_planner::DefaultPhysicalPlanner;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{
    Column, DataFusionError, Result, ScalarValue, internal_err, not_impl_err,
};
use datafusion_expr::logical_plan::{
    FetchType, Projection, SkipType, Sort, SubqueryAlias,
};
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_physical_pipeline::{PipelineQueryPlanner, PipelineRootExec};
use datafusion_physical_plan::{ExecutionPlan, collect, displayable};

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
    run_dataframe(ctx, ctx.sql(sql).await?).await
}

/// Plan and run `df`, created by `ctx`, on `ctx`.
async fn run_dataframe(ctx: &SessionContext, df: DataFrame) -> Result<EngineOutput> {
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
///
/// New tests use [`assert_same_results_as_datafusion`] instead, which
/// compares the results by meaning, as the correctness policy requires: byte
/// equality is not required, and this assertion fails on results that are
/// correct but differ, e.g., in floating-point rounding or in which rows tie
/// under `ORDER BY … LIMIT`. It still works for queries whose results are
/// unambiguous (see the [module documentation](self#byte-equality)).
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
/// pipeline engine (`actual`) byte for byte, see the
/// [module documentation](self#byte-equality). [`compare_results`] compares
/// them by meaning.
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
///
/// The batches are converted to the row format one by one, and the rows
/// converted back to arrays together: concatenating the batches first would
/// merge dictionaries, which can overflow narrow dictionary keys although
/// every batch is valid.
pub fn canonical_batch(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    ordered: bool,
) -> Result<RecordBatch> {
    let num_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if schema.fields().is_empty() {
        return Ok(concat_batches(schema, batches)?);
    }
    let sort_fields = schema
        .fields()
        .iter()
        .map(|field| SortField::new(field.data_type().clone()))
        .collect::<Vec<_>>();
    if !RowConverter::supports_fields(&sort_fields) {
        return sort_by_display(schema, batches, ordered);
    }
    let converter = RowConverter::new(sort_fields)?;
    let mut rows = converter.empty_rows(num_rows, 0);
    for batch in batches {
        converter.append(&mut rows, batch.columns())?;
    }
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
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    Ok(RecordBatch::try_new_with_options(
        schema, columns, &options,
    )?)
}

/// Fallback for types the row format does not support: decode dictionary
/// columns (so that concatenating cannot overflow their keys), concatenate,
/// and sort by the display representation of the rows.
fn sort_by_display(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    ordered: bool,
) -> Result<RecordBatch> {
    let decoded_type = |data_type: &DataType| match data_type {
        DataType::Dictionary(_, value_type) => value_type.as_ref().clone(),
        other => other.clone(),
    };
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            Field::new(
                field.name(),
                decoded_type(field.data_type()),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone())
        })
        .collect::<Vec<_>>();
    let decoded_schema =
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let decoded = batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .map(|column| {
                    Ok(cast(column.as_ref(), &decoded_type(column.data_type()))?)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(RecordBatch::try_new_with_options(
                Arc::clone(&decoded_schema),
                columns,
                &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    let batch = concat_batches(&decoded_schema, &decoded)?;
    if ordered {
        return Ok(batch);
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

/// How [`assert_same_results_as_datafusion_with_options`] (and
/// [`EngineOutcomes::compare`]) compare the outcomes of both engines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparisonOptions {
    /// The relative tolerance of floating-point values, see
    /// [`ComparisonOptions::with_float_tolerance`]. Defaults to
    /// [`ComparisonOptions::DEFAULT_FLOAT_TOLERANCE`].
    pub float_tolerance: f64,
    /// Whether the comparison accepts that one engine fails while the other
    /// does not, see [`ComparisonOptions::with_allow_error_differences`].
    /// Defaults to `false`.
    pub allow_error_differences: bool,
}

impl ComparisonOptions {
    /// The default relative tolerance of floating-point values.
    pub const DEFAULT_FLOAT_TOLERANCE: f64 = 1e-9;

    /// Compare floating-point values (`Float16`, `Float32` and `Float64`, also
    /// inside lists, structs, maps and unions) within the relative tolerance
    /// `tolerance`: `a` and `b` are equal if
    /// `|a - b| <= tolerance * max(|a|, |b|)`. In any case, `NaN` equals
    /// `NaN`, `-0.0` equals `0.0`, and an infinity only equals itself.
    ///
    /// # Panics
    ///
    /// If `tolerance` is negative or not finite.
    pub fn with_float_tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance >= 0.0,
            "invalid float tolerance {tolerance}"
        );
        self.float_tolerance = tolerance;
        self
    }

    /// Accept (`true`) that one engine fails while the other does not, e.g.
    /// for queries whose errors come from rows that an engine may never
    /// evaluate. The results are compared only if both engines succeed.
    pub fn with_allow_error_differences(mut self, allow: bool) -> Self {
        self.allow_error_differences = allow;
        self
    }
}

impl Default for ComparisonOptions {
    fn default() -> Self {
        Self {
            float_tolerance: Self::DEFAULT_FLOAT_TOLERANCE,
            allow_error_differences: false,
        }
    }
}

/// Assert that `sql` has the same results on default DataFusion and on the
/// pipeline engine, compared by meaning (see
/// [the module documentation](self#comparison-by-meaning)). The query can use
/// the tables of [`register_test_tables`].
///
/// This is the assertion of new tests, rather than
/// [`assert_byte_eq_with_datafusion`].
///
/// The assertion fails if `sql` is not a valid query, or if only one engine
/// fails. If both engines fail, it passes, as their errors may differ (see
/// [`EngineOutcomes::compare`]); tests that expect the query to succeed can
/// check the outcomes of [`run_both_engines`].
pub async fn assert_same_results_as_datafusion(sql: &str) {
    assert_same_results_as_datafusion_with_setup(sql, async |ctx: &SessionContext| {
        register_test_tables(ctx)
    })
    .await;
}

/// Like [`assert_same_results_as_datafusion`], but `setup` registers the
/// tables used by the query (it is called once for each session, see
/// [`run_both_engines`]).
pub async fn assert_same_results_as_datafusion_with_setup<F>(sql: &str, setup: F)
where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    assert_same_results_as_datafusion_with_config(sql, test_session_config(), setup)
        .await;
}

/// Like [`assert_same_results_as_datafusion_with_setup`], with a custom
/// session configuration for both engines.
pub async fn assert_same_results_as_datafusion_with_config<F>(
    sql: &str,
    config: SessionConfig,
    setup: F,
) where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    assert_same_results_as_datafusion_with_options(
        sql,
        config,
        setup,
        ComparisonOptions::default(),
    )
    .await;
}

/// Like [`assert_same_results_as_datafusion_with_config`], with custom
/// [`ComparisonOptions`]: another floating-point tolerance, or accepting that
/// one engine fails while the other does not.
pub async fn assert_same_results_as_datafusion_with_options<F>(
    sql: &str,
    config: SessionConfig,
    setup: F,
    options: ComparisonOptions,
) where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    let outcomes = match run_both_engines(sql, config, setup).await {
        Ok(outcomes) => outcomes,
        Err(e) => panic!("Failed to run query on both engines: {e}\nquery: {sql}"),
    };
    if let Err(message) = outcomes.compare(&options) {
        let plan = outcomes.pipeline.as_ref().map_or_else(
            |_| String::new(),
            |output| {
                format!(
                    "\npipeline plan:\n{}",
                    displayable(output.plan.as_ref()).indent(true)
                )
            },
        );
        panic!("Results differ: {message}\nquery: {sql}{plan}");
    }
}

/// The outcomes of a query on both engines, see [`run_both_engines`].
#[derive(Debug)]
pub struct EngineOutcomes {
    /// The outcome of default DataFusion: its output, or its error.
    pub datafusion: Result<EngineOutput>,
    /// The outcome of the pipeline engine: its output, or its error.
    pub pipeline: Result<EngineOutput>,
    /// How the rows of the result are ordered, see [`result_order`].
    pub order: ResultOrder,
}

impl EngineOutcomes {
    /// Compare the outcomes by meaning, see
    /// [the module documentation](self#comparison-by-meaning). On a
    /// difference, returns its description.
    ///
    /// If both engines fail, the outcomes are the same: which error is raised
    /// is the engine's own. If only one engine fails, the outcomes differ,
    /// unless `options` allow error differences.
    pub fn compare(
        &self,
        options: &ComparisonOptions,
    ) -> std::result::Result<(), String> {
        match (&self.datafusion, &self.pipeline) {
            (Ok(expected), Ok(actual)) => {
                compare_results(expected, actual, &self.order, options)
            }
            (Err(_), Err(_)) => Ok(()),
            _ if options.allow_error_differences => Ok(()),
            (Err(error), Ok(output)) => Err(format!(
                "DataFusion failed, but the pipeline engine did not (see \
                 `ComparisonOptions::allow_error_differences`)\n\
                 DataFusion failed with: {error}\n\
                 the pipeline engine returned {}",
                describe_output(output)
            )),
            (Ok(output), Err(error)) => Err(format!(
                "the pipeline engine failed, but DataFusion did not (see \
                 `ComparisonOptions::allow_error_differences`)\n\
                 DataFusion returned {}\n\
                 the pipeline engine failed with: {error}",
                describe_output(output)
            )),
        }
    }
}

/// Run `sql` on both engines, configured with `config` and set up with
/// `setup` (called for the session of DataFusion first, then for the session
/// of the pipeline engine), and return their outcomes, with the
/// [`result_order`] of the query (for which DataFusion runs the query again
/// if it has a top-level `ORDER BY`).
///
/// Fails if `setup` fails, if `sql` is not a valid query (DataFusion cannot
/// plan it logically), or if the pipeline engine did not plan the query.
pub async fn run_both_engines<F>(
    sql: &str,
    config: SessionConfig,
    setup: F,
) -> Result<EngineOutcomes>
where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    let datafusion = datafusion_context(config.clone());
    setup(&datafusion).await?;
    let pipeline = pipeline_context(config);
    setup(&pipeline).await?;

    let df = datafusion.sql(sql).await?;
    let plan = df.logical_plan().clone();
    let datafusion_outcome = run_dataframe(&datafusion, df).await;
    let pipeline_outcome = run_sql(&pipeline, sql).await;
    if let Ok(output) = &pipeline_outcome
        && !output.plan.is::<PipelineRootExec>()
    {
        return Err(DataFusionError::Internal(format!(
            "The pipeline engine did not plan the query, its root is {}",
            output.plan.name()
        )));
    }
    let order = result_order(&datafusion, &plan).await;
    Ok(EngineOutcomes {
        datafusion: datafusion_outcome,
        pipeline: pipeline_outcome,
        order,
    })
}

/// How the rows of the result of a query are ordered, see [`result_order`].
#[derive(Debug, Clone)]
pub enum ResultOrder {
    /// The query has no top-level `ORDER BY` (see [`has_top_level_order_by`]):
    /// the rows of its results are compared as a multiset.
    Unordered,
    /// The query has a top-level `ORDER BY`: the rows of its results are
    /// compared with DataFusion's sorted rows of the query.
    Sorted(SortedReference),
    /// The query has a top-level `ORDER BY`, but its [`SortedReference`]
    /// could not be computed, for the given reason: the rows of its results
    /// are compared in their exact order.
    Exact(String),
}

/// DataFusion's sorted rows of a query with a top-level `ORDER BY`, which the
/// results of the query are compared with: they show which rows tie on the
/// `ORDER BY` keys, also on the keys that the query does not select, and
/// which tied rows the `LIMIT` or the `OFFSET` of the query cut.
#[derive(Debug, Clone)]
pub struct SortedReference {
    /// The sorted rows of the query without its `LIMIT` and `OFFSET`: the
    /// output columns of the query, followed by the values of
    /// [`SortedReference::keys`].
    pub batches: Vec<RecordBatch>,
    /// The number of output columns of `batches`.
    pub columns: usize,
    /// The `ORDER BY` keys whose values follow the output columns
    /// (displayed): all of them, up to the first volatile key (e.g.
    /// `random()`), as the rows that tie on the keys before it may come in
    /// any order.
    pub keys: Vec<String>,
    /// The number of sorted rows that the query skips (`OFFSET`).
    pub skip: usize,
    /// The maximum number of sorted rows that the query returns (`LIMIT`).
    pub fetch: Option<usize>,
}

/// How the rows of the result of `plan`, planned by `ctx`, are ordered. For a
/// query with a top-level `ORDER BY`, `ctx` runs `plan` without its `LIMIT`
/// and `OFFSET`, with the values of its `ORDER BY` keys after its output
/// columns, to compute its [`SortedReference`].
pub async fn result_order(ctx: &SessionContext, plan: &LogicalPlan) -> ResultOrder {
    if !has_top_level_order_by(plan) {
        return ResultOrder::Unordered;
    }
    match sorted_reference(ctx, plan).await {
        Ok(reference) => ResultOrder::Sorted(reference),
        Err(e) => ResultOrder::Exact(e.to_string()),
    }
}

async fn sorted_reference(
    ctx: &SessionContext,
    plan: &LogicalPlan,
) -> Result<SortedReference> {
    let (reference, keys, window) = reference_plan(plan)?;
    let output = plan.schema().fields();
    let fields = reference.schema().fields();
    if fields.len() != output.len() + keys.len()
        || output
            .iter()
            .zip(fields)
            .any(|(output, field)| output.data_type() != field.data_type())
    {
        return internal_err!(
            "the sorted rows do not have the output columns of the query"
        );
    }
    let batches = ctx.execute_logical_plan(reference).await?.collect().await?;
    Ok(SortedReference {
        batches,
        columns: output.len(),
        keys,
        skip: window.skip,
        fetch: window.fetch,
    })
}

/// The rows of its sorted rows that a query returns.
#[derive(Debug, Clone, Copy)]
struct Window {
    skip: usize,
    fetch: Option<usize>,
}

impl Window {
    /// The rows of this window that a `LIMIT fetch OFFSET skip` returns.
    fn then(self, skip: usize, fetch: Option<usize>) -> Self {
        let remaining = self.fetch.map(|rows| rows.saturating_sub(skip));
        Self {
            skip: self.skip.saturating_add(skip),
            fetch: match (remaining, fetch) {
                (Some(remaining), Some(fetch)) => Some(remaining.min(fetch)),
                (remaining, fetch) => remaining.or(fetch),
            },
        }
    }
}

/// The name of the column of the values of the `i`th `ORDER BY` key in the
/// plan of a [`SortedReference`].
fn key_column(i: usize) -> String {
    format!("__test_utils_order_by_key_{i}")
}

/// The plan of the [`SortedReference`] of `plan`, a query with a top-level
/// `ORDER BY` (see [`has_top_level_order_by`]): `plan` without the limits
/// above its top-level sort, with the values of the `ORDER BY` keys after its
/// output columns. Also returns the keys (displayed), and the window of the
/// sorted rows that `plan` returns.
fn reference_plan(plan: &LogicalPlan) -> Result<(LogicalPlan, Vec<String>, Window)> {
    match plan {
        LogicalPlan::Sort(sort) => {
            // The values of a volatile key cannot be computed again.
            let keys = sort
                .expr
                .iter()
                .map(|sort_expr| &sort_expr.expr)
                .take_while(|key| !key.is_volatile())
                .collect::<Vec<_>>();
            let mut exprs = sort
                .input
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect::<Vec<_>>();
            exprs.extend(
                keys.iter()
                    .enumerate()
                    .map(|(i, &key)| key.clone().alias(key_column(i))),
            );
            let input = Projection::try_new(exprs, Arc::clone(&sort.input))?;
            let sorted = LogicalPlan::Sort(Sort {
                expr: sort.expr.clone(),
                input: Arc::new(LogicalPlan::Projection(input)),
                fetch: None,
            });
            let keys = keys.iter().map(ToString::to_string).collect();
            let window = Window {
                skip: 0,
                fetch: sort.fetch,
            };
            Ok((sorted, keys, window))
        }
        LogicalPlan::Limit(limit) => {
            let (input, keys, window) = reference_plan(&limit.input)?;
            let (SkipType::Literal(skip), FetchType::Literal(fetch)) =
                (limit.get_skip_type()?, limit.get_fetch_type()?)
            else {
                return not_impl_err!("a LIMIT or an OFFSET that is not a literal");
            };
            Ok((input, keys, window.then(skip, fetch)))
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let (input, keys, window) = reference_plan(&alias.input)?;
            let alias = SubqueryAlias::try_new(Arc::new(input), alias.alias.clone())?;
            Ok((LogicalPlan::SubqueryAlias(alias), keys, window))
        }
        LogicalPlan::Projection(projection) => {
            let (input, keys, window) = reference_plan(&projection.input)?;
            let mut exprs = projection.expr.clone();
            for i in 0..keys.len() {
                let (qualifier, field) = input
                    .schema()
                    .qualified_field_with_unqualified_name(&key_column(i))?;
                exprs.push(Expr::Column(Column::new(qualifier.cloned(), field.name())));
            }
            let projection = Projection::try_new(exprs, Arc::new(input))?;
            Ok((LogicalPlan::Projection(projection), keys, window))
        }
        _ => internal_err!("the query has no top-level ORDER BY"),
    }
}

/// Compare the output of DataFusion (`expected`) with the output of the
/// pipeline engine (`actual`) by meaning, see
/// [the module documentation](self#comparison-by-meaning). On a difference,
/// returns its description.
///
/// The rows are compared as `order` requires (see [`result_order`];
/// [`EngineOutput::ordered`] is not used). With a [`SortedReference`], the
/// result of DataFusion is compared with it too, which checks that the
/// reference is consistent with the query.
pub fn compare_results(
    expected: &EngineOutput,
    actual: &EngineOutput,
    order: &ResultOrder,
    options: &ComparisonOptions,
) -> std::result::Result<(), String> {
    compare_schemas(expected, actual)?;
    let expected_rows = result_rows(&expected.batches)
        .map_err(|e| format!("cannot read the result of DataFusion: {e}"))?;
    let actual_rows = result_rows(&actual.batches)
        .map_err(|e| format!("cannot read the result of the pipeline engine: {e}"))?;
    let tolerance = options.float_tolerance;
    let columns = expected
        .schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let engines = ("DataFusion", "the pipeline engine");
    let rows = (expected_rows.as_slice(), actual_rows.as_slice());
    if expected_rows.len() != actual_rows.len() {
        let difference = multiset_difference(&expected_rows, &actual_rows, tolerance);
        return Err(format!(
            "the results have different numbers of rows: {} (DataFusion) and {} \
             (pipeline engine)\n{}",
            expected_rows.len(),
            actual_rows.len(),
            difference.describe(&columns, engines, rows, 0)
        ));
    }
    match order {
        ResultOrder::Unordered => {
            let difference = multiset_difference(&expected_rows, &actual_rows, tolerance);
            if difference.is_empty() {
                return Ok(());
            }
            Err(format!(
                "the rows differ (compared as a multiset, as the query has no top-level \
                 ORDER BY)\n{}",
                difference.describe(&columns, engines, rows, 0)
            ))
        }
        ResultOrder::Sorted(reference) => {
            if reference.columns != expected.schema.fields().len() {
                return Err(format!(
                    "the sorted reference has {} output columns, but the query has {}",
                    reference.columns,
                    expected.schema.fields().len()
                ));
            }
            let sorted = reference
                .reference_rows(2.0 * tolerance)
                .map_err(|e| format!("cannot read the sorted reference: {e}"))?;
            let names = (columns.as_str(), reference.keys.join(", "));
            let names = (names.0, names.1.as_str());
            let reference_engines = ("the sorted reference of DataFusion", "DataFusion");
            compare_sorted(reference_engines, &expected_rows, &sorted, names, tolerance)
                .map_err(|e| {
                    format!(
                        "the result of DataFusion does not match its sorted reference \
                         (the query without its LIMIT and OFFSET, with the values of its \
                         ORDER BY keys): is the result of the query defined?\n{e}"
                    )
                })?;
            compare_sorted(engines, &actual_rows, &sorted, names, tolerance)
        }
        ResultOrder::Exact(reason) => {
            compare_in_exact_order(rows, &columns, reason, tolerance)
        }
    }
}

/// Check that the schemas of the plans are equal, and that every batch of the
/// pipeline engine has the output schema (or the schema of the batches of
/// DataFusion, if they do not have the output schema).
fn compare_schemas(
    expected: &EngineOutput,
    actual: &EngineOutput,
) -> std::result::Result<(), String> {
    if expected.schema != actual.schema {
        return Err(format!(
            "the output schemas differ: {}\nDataFusion: {:#?}\npipeline engine: {:#?}",
            schema_difference(&expected.schema, &actual.schema),
            expected.schema,
            actual.schema
        ));
    }
    let datafusion_batches = expected.batches.first().map(RecordBatch::schema);
    if let Some(schema) = actual
        .batches
        .iter()
        .map(RecordBatch::schema)
        .find(|schema| {
            *schema != actual.schema && Some(schema) != datafusion_batches.as_ref()
        })
    {
        return Err(format!(
            "the schema of a batch of the pipeline engine is not the output schema: \
             {}\noutput schema: {:#?}\nbatch schema: {:#?}",
            schema_difference(&actual.schema, &schema),
            actual.schema,
            schema
        ));
    }
    Ok(())
}

/// Describe the first difference of two schemas.
fn schema_difference(expected: &Schema, actual: &Schema) -> String {
    let (expected_fields, actual_fields) = (expected.fields(), actual.fields());
    if expected_fields.len() != actual_fields.len() {
        return format!(
            "{} vs {} fields",
            expected_fields.len(),
            actual_fields.len()
        );
    }
    for (e, a) in expected_fields.iter().zip(actual_fields) {
        let name = e.name();
        if name != a.name() {
            return format!("a field is named {name:?} vs {:?}", a.name());
        }
        if e.data_type() != a.data_type() {
            let (e_type, a_type) = (e.data_type().to_string(), a.data_type().to_string());
            return if e_type != a_type {
                format!("field {name:?} has the type {e_type} vs {a_type}")
            } else {
                format!(
                    "field {name:?} has the type {:?} vs {:?}",
                    e.data_type(),
                    a.data_type()
                )
            };
        }
        if e.is_nullable() != a.is_nullable() {
            let nullable = |nullable| if nullable { "nullable" } else { "not nullable" };
            return format!(
                "field {name:?} is {} vs {}",
                nullable(e.is_nullable()),
                nullable(a.is_nullable())
            );
        }
        if e.metadata() != a.metadata() {
            return format!(
                "field {name:?} has the metadata {:?} vs {:?}",
                e.metadata(),
                a.metadata()
            );
        }
        if e != a {
            return format!("field {name:?} differs: {e:?} vs {a:?}");
        }
    }
    format!(
        "the schema metadata is {:?} vs {:?}",
        expected.metadata(),
        actual.metadata()
    )
}

/// The rows of a [`SortedReference`] that results are compared with.
struct ReferenceRows {
    /// The output columns of the sorted rows that the query returns.
    rows: Vec<Row>,
    /// The values of the keys of these rows.
    keys: Vec<Row>,
    /// The output columns of the sorted rows that tie with the first row
    /// that the query returns, if the `OFFSET` skips some of them: the first
    /// rows of a result may be any of them.
    first_ties: Option<Vec<Row>>,
    /// The same for the last row that the query returns, if the `LIMIT` cuts
    /// some of the rows that tie with it.
    last_ties: Option<Vec<Row>>,
}

impl SortedReference {
    /// The rows that results are compared with, where keys that are equal
    /// within `tie_tolerance` tie.
    fn reference_rows(&self, tie_tolerance: f64) -> Result<ReferenceRows> {
        let width = self.columns + self.keys.len();
        if let Some(batch) = self
            .batches
            .iter()
            .find(|batch| batch.num_columns() != width)
        {
            return internal_err!(
                "a batch of the sorted rows has {} columns instead of {width}",
                batch.num_columns()
            );
        }
        let sorted = BatchRows::new(&self.batches);
        let (values, keys) = (0..self.columns, self.columns..width);
        let len = sorted.len();
        let start = self.skip.min(len);
        let end = self
            .fetch
            .map_or(len, |fetch| start.saturating_add(fetch).min(len));
        let ties = |a: usize, b: usize| -> Result<bool> {
            let (a, b) = (sorted.row(a, keys.clone())?, sorted.row(b, keys.clone())?);
            Ok(rows_equal(&a, &b, tie_tolerance))
        };
        // The output columns of the sorted rows that tie with `row`.
        let group = |row: usize| -> Result<Vec<Row>> {
            let (mut first, mut last) = (row, row);
            while first > 0 && ties(first - 1, first)? {
                first -= 1;
            }
            while last + 1 < len && ties(last, last + 1)? {
                last += 1;
            }
            (first..=last)
                .map(|row| sorted.row(row, values.clone()))
                .collect()
        };
        let skips_ties = start > 0 && start < end && ties(start - 1, start)?;
        let cuts_ties = start < end && end < len && ties(end - 1, end)?;
        Ok(ReferenceRows {
            rows: (start..end)
                .map(|row| sorted.row(row, values.clone()))
                .collect::<Result<_>>()?,
            keys: (start..end)
                .map(|row| sorted.row(row, keys.clone()))
                .collect::<Result<_>>()?,
            first_ties: skips_ties.then(|| group(start)).transpose()?,
            last_ties: cuts_ties.then(|| group(end - 1)).transpose()?,
        })
    }
}

/// Random access to the rows of batches.
struct BatchRows<'a> {
    batches: &'a [RecordBatch],
    /// The number of rows of the batches up to each batch (included).
    ends: Vec<usize>,
}

impl<'a> BatchRows<'a> {
    fn new(batches: &'a [RecordBatch]) -> Self {
        let mut end = 0;
        let ends = batches
            .iter()
            .map(|batch| {
                end += batch.num_rows();
                end
            })
            .collect();
        Self { batches, ends }
    }

    fn len(&self) -> usize {
        self.ends.last().copied().unwrap_or(0)
    }

    /// The values of the columns `columns` of row `row`.
    fn row(&self, row: usize, columns: Range<usize>) -> Result<Row> {
        let batch = self.ends.partition_point(|&end| end <= row);
        let first = batch
            .checked_sub(1)
            .map_or(0, |previous| self.ends[previous]);
        let batch = &self.batches[batch];
        columns
            .map(|column| Value::try_new(batch.column(column).as_ref(), row - first))
            .collect()
    }
}

/// Compare the rows `result` of `names.1` with the rows `reference` of the
/// sorted reference `names.0`, see [`ResultOrder::Sorted`]: the rows that tie
/// on the keys of the reference may come in any order among themselves, and
/// the rows that tie with a row that the `LIMIT` or the `OFFSET` cuts next to
/// may be any of the sorted rows that tie with it.
fn compare_sorted(
    (reference_name, result_name): (&str, &str),
    result: &[Row],
    reference: &ReferenceRows,
    (columns, keys): (&str, &str),
    tolerance: f64,
) -> std::result::Result<(), String> {
    if result.len() != reference.rows.len() {
        return Err(format!(
            "{result_name} returned {}, but the query returns {} of {reference_name}",
            count_rows(result.len()),
            count_rows(reference.rows.len())
        ));
    }
    let (ordering, on_keys) = if keys.is_empty() {
        (
            "the first ORDER BY key is volatile, so the rows may come in any order"
                .to_string(),
            "on the ORDER BY keys, the first of which is volatile".to_string(),
        )
    } else {
        (
            format!(
                "the result is ordered by ({keys}), rows that tie on these keys may \
                 come in any order"
            ),
            format!("on the ORDER BY keys ({keys})"),
        )
    };
    let runs = tie_runs(&reference.keys, 2.0 * tolerance);
    let last = runs.len().saturating_sub(1);
    for (i, run) in runs.into_iter().enumerate() {
        let (rows, differ, these_tie) = if run.len() == 1 {
            (format!("row {}", run.start), "differs", "this row ties")
        } else {
            (
                format!("rows {}..{}", run.start, run.end),
                "differ",
                "these rows tie",
            )
        };
        let actual = &result[run.clone()];
        let cut = match (&reference.first_ties, &reference.last_ties) {
            (_, Some(ties)) if i == last => {
                Some((ties, "last", "after which the LIMIT cuts the sorted rows"))
            }
            (Some(ties), _) if i == 0 => {
                Some((ties, "first", "before which the OFFSET skips sorted rows"))
            }
            _ => None,
        };
        if let Some((ties, which, cut)) = cut {
            // Any of the sorted rows that tie with the row next to the cut,
            // with their multiplicities.
            let unexpected = multiset_difference(ties, actual, tolerance).unexpected;
            if !unexpected.is_empty() {
                let all = (0..ties.len()).collect::<Vec<_>>();
                return Err(format!(
                    "{rows} of {result_name} {differ}: {these_tie} with the {which} row \
                     of the result {on_keys}, {cut}, so they may be any {} of the {} \
                     sorted rows of {reference_name} that tie with it (with their \
                     multiplicities)\ncolumns: ({columns}){}{}",
                    count_rows(run.len()),
                    ties.len(),
                    list_rows(
                        &format!(
                            "{} of {result_name} not among them",
                            count_rows(unexpected.len())
                        ),
                        actual,
                        &unexpected,
                        Some(run.start)
                    ),
                    list_rows(
                        &format!("the sorted rows of {reference_name} that tie with it"),
                        ties,
                        &all,
                        None
                    )
                ));
            }
        } else {
            let expected = &reference.rows[run.clone()];
            let difference = multiset_difference(expected, actual, tolerance);
            if !difference.is_empty() {
                return Err(format!(
                    "{rows} of {result_name} {differ} ({ordering})\n{}",
                    difference.describe(
                        columns,
                        (reference_name, result_name),
                        (expected, actual),
                        run.start
                    )
                ));
            }
        }
    }
    Ok(())
}

/// Compare the rows of both engines in their exact order, see
/// [`ResultOrder::Exact`].
fn compare_in_exact_order(
    (expected, actual): (&[Row], &[Row]),
    columns: &str,
    reason: &str,
    tolerance: f64,
) -> std::result::Result<(), String> {
    let Some(row) = (0..expected.len())
        .find(|&row| !rows_equal(&expected[row], &actual[row], tolerance))
    else {
        return Ok(());
    };
    let difference = Difference {
        missing: vec![0],
        unexpected: vec![0],
    };
    Err(format!(
        "row {row} differs (the rows are compared in their exact order, as the sorted \
         rows of the query could not be computed: {reason})\n{}",
        difference.describe(
            columns,
            ("DataFusion", "the pipeline engine"),
            (&expected[row..=row], &actual[row..=row]),
            row
        )
    ))
}

/// The ranges of the runs of consecutive rows whose `keys` are equal within
/// `tolerance`.
fn tie_runs(keys: &[Row], tolerance: f64) -> Vec<Range<usize>> {
    let mut runs = vec![];
    let mut start = 0;
    for (i, pair) in keys.windows(2).enumerate() {
        if !rows_equal(&pair[0], &pair[1], tolerance) {
            runs.push(start..i + 1);
            start = i + 1;
        }
    }
    if !keys.is_empty() {
        runs.push(start..keys.len());
    }
    runs
}

/// The rows of a result, see [`Value`].
fn result_rows(batches: &[RecordBatch]) -> Result<Vec<Row>> {
    let mut rows = vec![];
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(|column| Value::try_new(column.as_ref(), row))
                    .collect::<Result<Row>>()?,
            );
        }
    }
    Ok(rows)
}

/// A row of a result.
type Row = Vec<Value>;

/// A value of a result, independent of its physical representation:
/// dictionary and run-end encoding, views, offsets and the bytes under nulls
/// do not matter. Floating-point values are kept apart from the other values,
/// as they are compared within a tolerance.
#[derive(Debug, Clone)]
enum Value {
    Null,
    /// A `Float16`, `Float32` or `Float64` value.
    Float(f64),
    /// Any other scalar value (not null), compared exactly.
    Scalar(ScalarValue),
    /// The elements of a list (of any list type), or the entries of a map.
    List(Vec<Value>),
    /// The fields of a struct, or the type id and the value of a union.
    Struct(Vec<Value>),
}

impl Value {
    /// The value of `array` at `row`.
    fn try_new(array: &dyn Array, row: usize) -> Result<Self> {
        if array.data_type() == &DataType::Null || array.is_null(row) {
            return Ok(Self::Null);
        }
        Ok(match array.data_type() {
            DataType::Float16 => {
                Self::Float(array.as_primitive::<Float16Type>().value(row).to_f64())
            }
            DataType::Float32 => {
                Self::Float(f64::from(array.as_primitive::<Float32Type>().value(row)))
            }
            DataType::Float64 => {
                Self::Float(array.as_primitive::<Float64Type>().value(row))
            }
            DataType::List(_) => {
                let list = array.as_list::<i32>();
                let offsets = &list.value_offsets()[row..=row + 1];
                Self::list(list.values(), offsets[0].as_usize()..offsets[1].as_usize())?
            }
            DataType::LargeList(_) => {
                let list = array.as_list::<i64>();
                let offsets = &list.value_offsets()[row..=row + 1];
                Self::list(list.values(), offsets[0].as_usize()..offsets[1].as_usize())?
            }
            DataType::ListView(_) => {
                let list = array.as_list_view::<i32>();
                let start = list.value_offsets()[row].as_usize();
                Self::list(
                    list.values(),
                    start..start + list.value_sizes()[row].as_usize(),
                )?
            }
            DataType::LargeListView(_) => {
                let list = array.as_list_view::<i64>();
                let start = list.value_offsets()[row].as_usize();
                Self::list(
                    list.values(),
                    start..start + list.value_sizes()[row].as_usize(),
                )?
            }
            DataType::FixedSizeList(_, size) => {
                let list = array.as_fixed_size_list();
                let start = list.value_offset_at(row);
                Self::list(list.values(), start..start + size.as_usize())?
            }
            DataType::Map(_, _) => {
                let map = array.as_map();
                let offsets = &map.value_offsets()[row..=row + 1];
                Self::list(map.entries(), offsets[0].as_usize()..offsets[1].as_usize())?
            }
            DataType::Struct(_) => Self::Struct(
                array
                    .as_struct()
                    .columns()
                    .iter()
                    .map(|column| Self::try_new(column.as_ref(), row))
                    .collect::<Result<_>>()?,
            ),
            DataType::Union(_, _) => {
                let union = array.as_union();
                let type_id = union.type_id(row);
                let value = Self::try_new(
                    union.child(type_id).as_ref(),
                    union.value_offset(row),
                )?;
                Self::Struct(vec![Self::Scalar(ScalarValue::Int8(Some(type_id))), value])
            }
            DataType::Dictionary(_, _) => downcast_dictionary_array!(
                array => Self::try_new(array.values().as_ref(), array.keys().value(row).as_usize())?,
                _ => unreachable!("not a dictionary")
            ),
            DataType::RunEndEncoded(run_ends, _) => match run_ends.data_type() {
                DataType::Int16 => Self::run_value(array.as_run::<Int16Type>(), row)?,
                DataType::Int32 => Self::run_value(array.as_run::<Int32Type>(), row)?,
                DataType::Int64 => Self::run_value(array.as_run::<Int64Type>(), row)?,
                other => {
                    return Err(DataFusionError::NotImplemented(format!(
                        "run ends of type {other}"
                    )));
                }
            },
            _ => Self::Scalar(ScalarValue::try_from_array(array, row)?),
        })
    }

    /// The values of `values` at `rows`, as a list.
    fn list(values: &dyn Array, rows: Range<usize>) -> Result<Self> {
        Ok(Self::List(
            rows.map(|row| Self::try_new(values, row))
                .collect::<Result<_>>()?,
        ))
    }

    fn run_value<R: RunEndIndexType>(array: &RunArray<R>, row: usize) -> Result<Self> {
        Self::try_new(array.values().as_ref(), array.get_physical_index(row))
    }

    /// Whether `self` and `other` are equal, comparing floating-point values
    /// within the relative tolerance `tolerance`.
    fn equals(&self, other: &Self, tolerance: f64) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Float(a), Self::Float(b)) => floats_equal(*a, *b, tolerance),
            (Self::Scalar(a), Self::Scalar(b)) => a == b,
            (Self::List(a), Self::List(b)) | (Self::Struct(a), Self::Struct(b)) => {
                rows_equal(a, b, tolerance)
            }
            _ => false,
        }
    }

    /// Whether `self` and `other` are equal but (maybe) for their
    /// floating-point values, see [`Shape`].
    fn same_shape(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) | (Self::Float(_), Self::Float(_)) => true,
            (Self::Scalar(a), Self::Scalar(b)) => a == b,
            (Self::List(a), Self::List(b)) | (Self::Struct(a), Self::Struct(b)) => {
                Shape(a) == Shape(b)
            }
            _ => false,
        }
    }

    /// Hash `self` but its floating-point values, see [`Shape`].
    fn hash_shape<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Null | Self::Float(_) => {}
            Self::Scalar(scalar) => scalar.hash(state),
            Self::List(values) | Self::Struct(values) => Shape(values).hash(state),
        }
    }

    fn has_floats(&self) -> bool {
        match self {
            Self::Float(_) => true,
            Self::List(values) | Self::Struct(values) => {
                values.iter().any(Self::has_floats)
            }
            Self::Null | Self::Scalar(_) => false,
        }
    }

    /// Append the floating-point values of `self` to `floats`.
    fn collect_floats(&self, floats: &mut Vec<f64>) {
        match self {
            Self::Float(value) => floats.push(*value),
            Self::List(values) | Self::Struct(values) => {
                for value in values {
                    value.collect_floats(floats);
                }
            }
            Self::Null | Self::Scalar(_) => {}
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values = |f: &mut fmt::Formatter<'_>, values: &[Self]| {
            for (i, value) in values.iter().enumerate() {
                write!(f, "{}{value}", if i == 0 { "" } else { ", " })?;
            }
            Ok(())
        };
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Float(value) => write!(f, "{value:?}"),
            Self::Scalar(scalar) => match scalar.try_as_str() {
                Some(Some(string)) => write!(f, "{string:?}"),
                _ => write!(f, "{scalar}"),
            },
            Self::List(elements) => {
                write!(f, "[")?;
                values(f, elements)?;
                write!(f, "]")
            }
            Self::Struct(fields) => {
                write!(f, "{{")?;
                values(f, fields)?;
                write!(f, "}}")
            }
        }
    }
}

/// Values (a row, or the elements of a list or a struct) without their
/// floating-point values: values that are equal within a tolerance have the
/// same shape.
struct Shape<'a>(&'a [Value]);

impl PartialEq for Shape<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().zip(other.0).all(|(a, b)| a.same_shape(b))
    }
}

impl Eq for Shape<'_> {}

impl Hash for Shape<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for value in self.0 {
            value.hash_shape(state);
        }
    }
}

/// Whether floating-point values `a` and `b` are equal within the relative
/// tolerance `tolerance`, see [`ComparisonOptions::with_float_tolerance`].
fn floats_equal(a: f64, b: f64, tolerance: f64) -> bool {
    if a == b {
        // Also `-0.0` and `0.0`, and infinities of the same sign.
        return true;
    }
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    (a - b).abs() <= tolerance * a.abs().max(b.abs())
}

fn rows_equal(a: &[Value], b: &[Value], tolerance: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.equals(b, tolerance))
}

/// The rows of two multisets of rows that the other multiset does not have,
/// see [`multiset_difference`].
#[derive(Debug, Default)]
struct Difference {
    /// The indices of the expected rows that are not in the actual rows.
    missing: Vec<usize>,
    /// The indices of the actual rows that are not in the expected rows.
    unexpected: Vec<usize>,
}

impl Difference {
    fn is_empty(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty()
    }

    /// Describe the difference of the rows `expected` of `names.0` and
    /// `actual` of `names.1`, which are the rows `first..` of results with
    /// the columns `columns`.
    fn describe(
        &self,
        columns: &str,
        (expected_name, actual_name): (&str, &str),
        (expected, actual): (&[Row], &[Row]),
        first: usize,
    ) -> String {
        let mut out = format!("columns: ({columns})");
        if !self.missing.is_empty() {
            let title = format!(
                "{} of {expected_name} missing from the result of {actual_name}",
                count_rows(self.missing.len())
            );
            out += &list_rows(&title, expected, &self.missing, Some(first));
        }
        if !self.unexpected.is_empty() {
            let title = format!(
                "{} of {actual_name} not in the result of {expected_name}",
                count_rows(self.unexpected.len())
            );
            out += &list_rows(&title, actual, &self.unexpected, Some(first));
        }
        out
    }
}

/// List the rows `indices` of `rows` (up to 10 of them) under `title`.
/// `first` is the position of `rows[0]` in its result, if `rows` are rows of
/// a result.
fn list_rows(
    title: &str,
    rows: &[Row],
    indices: &[usize],
    first: Option<usize>,
) -> String {
    const MAX_ROWS: usize = 10;
    let mut out = format!("\n{title}:");
    for &i in indices.iter().take(MAX_ROWS) {
        let values = rows[i]
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = match first {
            Some(first) => write!(out, "\n  row {}: ({values})", first + i),
            None => write!(out, "\n  ({values})"),
        };
    }
    if indices.len() > MAX_ROWS {
        let _ = write!(out, "\n  ... {} more", indices.len() - MAX_ROWS);
    }
    out
}

/// The difference of the multisets of rows `expected` and `actual`, whose
/// floating-point values are compared within the relative tolerance
/// `tolerance`.
///
/// Rows that are equal within the tolerance have the same [`Shape`]: the rows
/// are grouped by their shapes, and the floating-point values of the rows of
/// each group are matched.
fn multiset_difference(expected: &[Row], actual: &[Row], tolerance: f64) -> Difference {
    if let ([e], [a]) = (expected, actual) {
        return if rows_equal(e, a, tolerance) {
            Difference::default()
        } else {
            Difference {
                missing: vec![0],
                unexpected: vec![0],
            }
        };
    }
    // The rows of `expected` and of `actual` of each shape.
    let mut groups: Vec<[Vec<usize>; 2]> = vec![];
    let mut group_of_shape = HashMap::new();
    for (side, rows) in [expected, actual].into_iter().enumerate() {
        for (i, row) in rows.iter().enumerate() {
            let group = *group_of_shape.entry(Shape(row)).or_insert_with(|| {
                groups.push(Default::default());
                groups.len() - 1
            });
            groups[group][side].push(i);
        }
    }

    let mut difference = Difference::default();
    for [expected_group, actual_group] in groups {
        let has_floats = expected_group
            .first()
            .map(|&row| &expected[row])
            .or_else(|| actual_group.first().map(|&row| &actual[row]))
            .is_some_and(|row| row.iter().any(Value::has_floats));
        if !has_floats {
            // The rows of the group are equal.
            let matched = expected_group.len().min(actual_group.len());
            difference.missing.extend(&expected_group[matched..]);
            difference.unexpected.extend(&actual_group[matched..]);
            continue;
        }
        let floats = |rows: &[Row], group: &[usize]| {
            group
                .iter()
                .map(|&row| {
                    let mut floats = vec![];
                    for value in &rows[row] {
                        value.collect_floats(&mut floats);
                    }
                    floats
                })
                .collect::<Vec<_>>()
        };
        let (missing, unexpected) = match_floats(
            &floats(expected, &expected_group),
            &floats(actual, &actual_group),
            tolerance,
        );
        difference
            .missing
            .extend(missing.into_iter().map(|i| expected_group[i]));
        difference
            .unexpected
            .extend(unexpected.into_iter().map(|i| actual_group[i]));
    }
    difference.missing.sort_unstable();
    difference.unexpected.sort_unstable();
    difference
}

/// Match the floating-point values of the rows of a group (vectors of the
/// same length): returns the indices of the `expected` and of the `actual`
/// vectors that a maximum matching of vectors that are equal within the
/// relative tolerance `tolerance` leaves unmatched.
///
/// Identical vectors are merged (with their multiplicities), the pairs of
/// vectors that are equal within the tolerance are searched among the
/// vectors whose values at one position are close, and a maximum flow
/// through these pairs matches the vectors: exactly, whatever the number of
/// rows.
fn match_floats(
    expected: &[Vec<f64>],
    actual: &[Vec<f64>],
    tolerance: f64,
) -> (Vec<usize>, Vec<usize>) {
    let (expected_vectors, expected_rows) = distinct_vectors(expected);
    let (actual_vectors, actual_rows) = distinct_vectors(actual);
    let pairs = equal_pairs(&expected_vectors, &actual_vectors, tolerance);
    let mut supply = expected_rows.iter().map(Vec::len).collect::<Vec<_>>();
    let mut demand = actual_rows.iter().map(Vec::len).collect::<Vec<_>>();
    max_flow(&pairs, &mut supply, &mut demand);
    // The rows of each distinct vector that are left over.
    let unmatched = |rows: &[Vec<usize>], left: &[usize]| {
        rows.iter()
            .zip(left)
            .flat_map(|(rows, &left)| rows[rows.len() - left..].iter().copied())
            .collect::<Vec<_>>()
    };
    (
        unmatched(&expected_rows, &supply),
        unmatched(&actual_rows, &demand),
    )
}

/// `value`, but `-0.0` is `0.0` and every `NaN` is [`f64::NAN`].
fn normalized(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value
    }
}

/// The distinct vectors of `vectors`, [`normalized`], and the indices of the
/// vectors that are equal to each.
fn distinct_vectors(vectors: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let mut distinct = vec![];
    let mut rows: Vec<Vec<usize>> = vec![];
    let mut index_of = HashMap::new();
    for (row, vector) in vectors.iter().enumerate() {
        let vector = vector
            .iter()
            .map(|&value| normalized(value))
            .collect::<Vec<_>>();
        let bits = vector
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let index = *index_of.entry(bits).or_insert_with(|| {
            distinct.push(vector);
            rows.push(vec![]);
            rows.len() - 1
        });
        rows[index].push(row);
    }
    (distinct, rows)
}

fn vectors_equal(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| floats_equal(*a, *b, tolerance))
}

/// For each of the vectors `expected`, the indices of the vectors `actual`
/// (all [`normalized`]) that are equal to it within the relative tolerance
/// `tolerance`.
///
/// The candidates are the vectors whose values are close at the position at
/// which `actual` has the most distinct values.
fn equal_pairs(
    expected: &[Vec<f64>],
    actual: &[Vec<f64>],
    tolerance: f64,
) -> Vec<Vec<usize>> {
    let width = actual.first().map_or(0, Vec::len);
    let position = (0..width)
        .max_by_key(|&position| {
            let values = actual.iter().map(|vector| vector[position].to_bits());
            values.collect::<HashSet<_>>().len()
        })
        .unwrap_or(0);
    let mut order = (0..actual.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| actual[a][position].total_cmp(&actual[b][position]));
    let values = order
        .iter()
        .map(|&a| actual[a][position])
        .collect::<Vec<_>>();
    expected
        .iter()
        .map(|vector| {
            if width == 0 {
                return vec![];
            }
            order[close_values(&values, vector[position], tolerance)]
                .iter()
                .copied()
                .filter(|&a| vectors_equal(vector, &actual[a], tolerance))
                .collect()
        })
        .collect()
}

/// The range of the sorted values `values` ([`normalized`]) that may be equal
/// to `value` within the relative tolerance `tolerance`.
fn close_values(values: &[f64], value: f64, tolerance: f64) -> Range<usize> {
    let (low, high) = if value.is_finite() {
        // `|a - b| <= tolerance * max(|a|, |b|)` bounds `|a - b|` by
        // `tolerance / (1 - tolerance) * |a|`, which is less than (with room
        // for rounding) `2 * tolerance * |a|` if `tolerance <= 1/4`.
        let radius = if tolerance <= 0.25 {
            2.0 * tolerance * value.abs()
        } else {
            f64::INFINITY
        };
        (value - radius, value + radius)
    } else {
        // An infinity only equals itself, and `NaN` only `NaN`.
        (value, value)
    };
    let start = values.partition_point(|v| v.total_cmp(&low).is_lt());
    let end = values.partition_point(|v| v.total_cmp(&high).is_le());
    start..end.max(start)
}

/// Push a maximum flow from the expected vectors, which supply `supply` rows
/// each, through the `pairs` of vectors that are equal, to the actual
/// vectors, which demand `demand` rows each. Leaves in `supply` and `demand`
/// the rows that are not matched.
///
/// A greedy flow is augmented along shortest paths (Edmonds-Karp): from an
/// expected vector with supply, forward through pairs, and backward through
/// pairs with flow, to an actual vector with demand.
fn max_flow(pairs: &[Vec<usize>], supply: &mut [usize], demand: &mut [usize]) {
    // The flow through each pair, and the pairs into each actual vector.
    let mut flow = pairs
        .iter()
        .map(|pairs| vec![0; pairs.len()])
        .collect::<Vec<_>>();
    let mut pairs_into = vec![vec![]; demand.len()];
    for (e, pairs) in pairs.iter().enumerate() {
        for (pair, &a) in pairs.iter().enumerate() {
            pairs_into[a].push((e, pair));
            let rows = supply[e].min(demand[a]);
            flow[e][pair] += rows;
            supply[e] -= rows;
            demand[a] -= rows;
        }
    }
    loop {
        // `reached_by[a]`: the pair through which an actual vector `a` is
        // reached. `backward_to[e]`: for an expected vector reached backward,
        // the actual vector and the pair with flow through which it is.
        let mut reached_by: Vec<Option<(usize, usize)>> = vec![None; demand.len()];
        let mut backward_to: Vec<Option<(usize, usize)>> = vec![None; supply.len()];
        let mut queued = supply.iter().map(|&rows| rows > 0).collect::<Vec<_>>();
        let mut queue = (0..supply.len())
            .filter(|&e| queued[e])
            .collect::<VecDeque<_>>();
        let mut end = None;
        'search: while let Some(e) = queue.pop_front() {
            for (pair, &a) in pairs[e].iter().enumerate() {
                if reached_by[a].is_some() {
                    continue;
                }
                reached_by[a] = Some((e, pair));
                if demand[a] > 0 {
                    end = Some(a);
                    break 'search;
                }
                for &(other, other_pair) in &pairs_into[a] {
                    if flow[other][other_pair] > 0 && !queued[other] {
                        queued[other] = true;
                        backward_to[other] = Some((a, other_pair));
                        queue.push_back(other);
                    }
                }
            }
        }
        let Some(end) = end else {
            return;
        };
        // The rows that the path can carry, then push them along it.
        let mut rows = demand[end];
        let mut a = end;
        loop {
            let (e, _) = reached_by[a].expect("the path reaches `a`");
            match backward_to[e] {
                Some((previous, pair)) => {
                    rows = rows.min(flow[e][pair]);
                    a = previous;
                }
                None => {
                    rows = rows.min(supply[e]);
                    break;
                }
            }
        }
        demand[end] -= rows;
        let mut a = end;
        loop {
            let (e, pair) = reached_by[a].expect("the path reaches `a`");
            flow[e][pair] += rows;
            match backward_to[e] {
                Some((previous, backward)) => {
                    flow[e][backward] -= rows;
                    a = previous;
                }
                None => {
                    supply[e] -= rows;
                    break;
                }
            }
        }
    }
}

/// The number of rows and the first rows of a result.
fn describe_output(output: &EngineOutput) -> String {
    let rows = output
        .batches
        .iter()
        .map(RecordBatch::num_rows)
        .sum::<usize>();
    format!(
        "{}:\n{}",
        count_rows(rows),
        preview_batches(&output.batches)
    )
}

/// `count` rows, in words.
fn count_rows(count: usize) -> String {
    format!("{count} {}", if count == 1 { "row" } else { "rows" })
}

/// The first rows of `batches`, formatted (without concatenating the
/// batches, which could overflow narrow dictionary keys).
fn preview_batches(batches: &[RecordBatch]) -> String {
    const MAX_ROWS: usize = 20;
    let mut slices = vec![];
    let mut rows = 0;
    for batch in batches {
        let take = batch.num_rows().min(MAX_ROWS - rows);
        slices.push(batch.slice(0, take));
        rows += take;
        if rows == MAX_ROWS {
            break;
        }
    }
    let mut out = pretty_format_batches(&slices).map_or_else(
        |e| format!("<cannot format the result: {e}>"),
        |table| table.to_string(),
    );
    let total = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if total > rows {
        let _ = write!(out, "\n... {} more rows", total - rows);
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
    let batch = |ids: Range<i64>| -> Result<RecordBatch> {
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
    let batch = |ids: Range<i64>| -> Result<RecordBatch> {
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

    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use arrow::array::{
        DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Float64Builder,
        Int8Array, Int32Array, Int64Array, ListArray, MapBuilder, StringArray,
        StringBuilder, StringViewArray, StructArray, UnionArray,
    };
    use arrow::buffer::{NullBuffer, ScalarBuffer};
    use arrow::datatypes::{Fields, Int8Type, UnionFields};

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

    /// Two batches of a `Dictionary<Int8, Utf8>` column with 100 distinct
    /// values each: together they have more values than an `Int8` key can
    /// address.
    fn narrow_dictionary_batches() -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        (0..2)
            .map(|b| {
                let ids = Int32Array::from_iter_values((0..100).map(|i| b * 100 + i));
                let strings = (0..100)
                    .map(|i| format!("value {}", b * 100 + i))
                    .collect::<Vec<_>>();
                let values = strings
                    .iter()
                    .map(String::as_str)
                    .collect::<DictionaryArray<Int8Type>>();
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(ids), Arc::new(values)],
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn canonical_bytes_of_narrow_dictionaries_over_several_batches() {
        let batches = narrow_dictionary_batches();
        for ordered in [false, true] {
            let one = canonical_batch(&batches[0].schema(), &batches, ordered).unwrap();
            assert_eq!(one.num_rows(), 200);
            // The same values in other batches and (unordered) another order.
            let reversed = batches.iter().rev().cloned().collect::<Vec<_>>();
            let split = batches
                .iter()
                .flat_map(|batch| [batch.slice(0, 30), batch.slice(30, 70)])
                .collect::<Vec<_>>();
            assert_eq!(bytes(&batches, ordered), bytes(&split, ordered));
            if ordered {
                assert_ne!(bytes(&batches, ordered), bytes(&reversed, ordered));
            } else {
                assert_eq!(bytes(&batches, ordered), bytes(&reversed, ordered));
            }
            assert!(
                compare_outputs(
                    &output(batches.clone(), ordered),
                    &output(split, ordered)
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn display_fallback_decodes_narrow_dictionaries() {
        let batches = narrow_dictionary_batches();
        let schema = batches[0].schema();
        let reversed = batches.iter().rev().cloned().collect::<Vec<_>>();
        let sorted = sort_by_display(&schema, &reversed, false).unwrap();
        assert_eq!(sorted.num_rows(), 200);
        assert_eq!(sorted.column(1).data_type(), &DataType::Utf8);
        assert_eq!(sorted, sort_by_display(&schema, &batches, false).unwrap());
        let in_order = sort_by_display(&schema, &reversed, true).unwrap();
        assert_eq!(
            in_order
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            100
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_narrow_dictionaries() {
        let batches = narrow_dictionary_batches();
        let setup = async |ctx: &SessionContext| {
            let partitions = batches.iter().map(|batch| vec![batch.clone()]).collect();
            ctx.register_table(
                "dict",
                Arc::new(MemTable::try_new(batches[0].schema(), partitions)?),
            )?;
            Ok(())
        };
        // (Sorting these batches fails in DataFusion itself, as merging
        // their dictionaries overflows the keys: the ordered comparison is
        // tested in `canonical_bytes_of_narrow_dictionaries_over_several_batches`.)
        for sql in [
            "SELECT * FROM dict",
            "SELECT d, id FROM dict WHERE id % 3 = 0",
        ] {
            assert_byte_eq_with_datafusion_with_setup(sql, &setup).await;
        }
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
            "WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 50) SELECT n FROM nums",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_dml_and_partitioning() {
        let setup = async |ctx: &SessionContext| {
            register_test_tables(ctx)?;
            ctx.sql("CREATE TABLE target (id BIGINT, d VARCHAR)")
                .await?
                .collect()
                .await?;
            Ok(())
        };
        assert_byte_eq_with_datafusion_with_setup(
            "INSERT INTO target SELECT id, d FROM t1 WHERE a > 0",
            &setup,
        )
        .await;
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT * FROM t1",
                "SELECT d, COUNT(*) FROM t1 GROUP BY d",
                "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t1.id, t2.name",
            ] {
                assert_byte_eq_with_datafusion_with_config(sql, config.clone(), &setup)
                    .await;
            }
        }
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

    // The comparison by meaning.

    /// A batch of nullable columns.
    fn batch_of(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter_with_nullable(
            columns.into_iter().map(|(name, array)| (name, array, true)),
        )
        .unwrap()
    }

    fn float_batch(values: &[Option<f64>]) -> RecordBatch {
        batch_of(vec![("f", Arc::new(Float64Array::from(values.to_vec())))])
    }

    /// A batch of `(k INT, v VARCHAR)` rows.
    fn kv_batch(rows: &[(i32, &str)]) -> RecordBatch {
        batch_of(vec![
            (
                "k",
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "v",
                Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.1))),
            ),
        ])
    }

    /// A batch of the values `v` of `(k, v)` rows.
    fn v_batch(rows: &[(i32, &str)]) -> RecordBatch {
        let v = StringArray::from_iter_values(rows.iter().map(|row| row.1));
        batch_of(vec![("v", Arc::new(v))])
    }

    /// The order of a query whose sorted rows (without its LIMIT and OFFSET)
    /// are `sorted` (with the columns of its output), ordered by the columns
    /// `keys`, of which it skips `skip` rows and returns at most `fetch`.
    fn sorted_by(
        sorted: &[RecordBatch],
        keys: &[&str],
        skip: usize,
        fetch: Option<usize>,
    ) -> ResultOrder {
        let columns = sorted[0].num_columns();
        let batches = sorted
            .iter()
            .map(|batch| {
                let schema = batch.schema();
                let mut fields = schema.fields().iter().cloned().collect::<Vec<_>>();
                let mut arrays = batch.columns().to_vec();
                for (i, key) in keys.iter().enumerate() {
                    let index = schema.index_of(key).unwrap();
                    let data_type = schema.field(index).data_type().clone();
                    fields.push(Arc::new(Field::new(key_column(i), data_type, true)));
                    arrays.push(Arc::clone(batch.column(index)));
                }
                RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
            })
            .collect();
        ResultOrder::Sorted(SortedReference {
            batches,
            columns,
            keys: keys.iter().map(ToString::to_string).collect(),
            skip,
            fetch,
        })
    }

    /// The order of a query that selects the values `v` of `(k, v)` rows,
    /// ordered by `k`: their sorted rows are `sorted` (without the LIMIT and
    /// OFFSET of the query, which returns at most `fetch` of them).
    fn sorted_by_k(sorted: &[(i32, &str)], fetch: Option<usize>) -> ResultOrder {
        let batch = kv_batch(sorted);
        let reference = batch_of(vec![
            ("v", Arc::clone(batch.column(1))),
            ("key", Arc::clone(batch.column(0))),
        ]);
        ResultOrder::Sorted(SortedReference {
            batches: vec![reference],
            columns: 1,
            keys: vec!["m.k".to_string()],
            skip: 0,
            fetch,
        })
    }

    /// Compare the results `expected` (of DataFusion) and `actual` (of the
    /// pipeline engine) by meaning.
    fn same_results(
        expected: Vec<RecordBatch>,
        actual: Vec<RecordBatch>,
        order: &ResultOrder,
    ) -> std::result::Result<(), String> {
        same_results_with(expected, actual, order, ComparisonOptions::default())
    }

    fn same_results_with(
        expected: Vec<RecordBatch>,
        actual: Vec<RecordBatch>,
        order: &ResultOrder,
        options: ComparisonOptions,
    ) -> std::result::Result<(), String> {
        compare_results(
            &output(expected, false),
            &output(actual, false),
            order,
            &options,
        )
    }

    const UNORDERED: &ResultOrder = &ResultOrder::Unordered;

    #[test]
    fn floats_are_equal_within_a_relative_tolerance() {
        let tolerance = ComparisonOptions::DEFAULT_FLOAT_TOLERANCE;
        for (a, b) in [
            (1.0, 1.0 + 1e-10),
            (-2.5e10, -2.5e10 * (1.0 + 5e-10)),
            (1e300, 1e300 * (1.0 - 1e-10)),
            (3e-300, 3e-300 * (1.0 + 1e-10)),
            (0.0, -0.0),
            (f64::NAN, -f64::NAN),
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        ] {
            assert!(floats_equal(a, b, tolerance), "{a:?} = {b:?}");
            assert!(floats_equal(b, a, tolerance), "{b:?} = {a:?}");
        }
        for (a, b) in [
            (1.0, 1.0 + 1e-8),
            (1.0, -1.0),
            // The tolerance is relative: only `-0.0` and `0.0` equal `0.0`.
            (0.0, 1e-300),
            (f64::NAN, 1.0),
            (f64::NAN, f64::INFINITY),
            (f64::INFINITY, f64::NEG_INFINITY),
            (f64::INFINITY, f64::MAX),
            (f64::MAX, -f64::MAX),
        ] {
            assert!(!floats_equal(a, b, tolerance), "{a:?} != {b:?}");
            assert!(!floats_equal(b, a, tolerance), "{b:?} != {a:?}");
        }
        // Other tolerances; 0 compares exactly, but for `NaN` and `-0.0`.
        assert!(floats_equal(1.0, 1.0 + 1e-7, 1e-6));
        assert!(!floats_equal(1.0, 1.0 + 1e-10, 0.0));
        assert!(floats_equal(-0.0, 0.0, 0.0));
        assert!(floats_equal(f64::NAN, f64::NAN, 0.0));
    }

    #[test]
    #[should_panic(expected = "invalid float tolerance -0.1")]
    fn float_tolerance_cannot_be_negative() {
        let _ = ComparisonOptions::default().with_float_tolerance(-0.1);
    }

    #[test]
    fn same_results_compare_floats_within_the_tolerance() {
        let expected = float_batch(&[
            Some(1.0),
            Some(-2.5e10),
            Some(3e-200),
            Some(f64::NAN),
            Some(-0.0),
            None,
            Some(f64::INFINITY),
        ]);
        let close = float_batch(&[
            Some(1.0 + 1e-12),
            Some(-2.5e10 * (1.0 - 5e-10)),
            Some(3e-200 * (1.0 + 1e-10)),
            Some(-f64::NAN),
            Some(0.0),
            None,
            Some(f64::INFINITY),
        ]);
        same_results(vec![expected], vec![close], UNORDERED).unwrap();

        // Beyond the default tolerance, but within a configured one.
        let one = || vec![float_batch(&[Some(1.0)])];
        let other = || vec![float_batch(&[Some(1.000001)])];
        let err = same_results(one(), other(), UNORDERED).unwrap_err();
        assert!(err.contains("row 0: (1.0)"), "{err}");
        assert!(err.contains("row 0: (1.000001)"), "{err}");
        let options = ComparisonOptions::default().with_float_tolerance(1e-5);
        same_results_with(one(), other(), UNORDERED, options).unwrap();

        // NULL, NaN and numbers are different values.
        for (a, b) in [
            (None, Some(f64::NAN)),
            (None, Some(0.0)),
            (Some(f64::NAN), Some(0.0)),
            (Some(f64::INFINITY), Some(f64::MAX)),
        ] {
            let (a, b) = (float_batch(&[a]), float_batch(&[b]));
            assert!(same_results(vec![a.clone()], vec![b.clone()], UNORDERED).is_err());
            assert!(same_results(vec![b], vec![a], UNORDERED).is_err());
        }

        // Float32 and Float16 values are compared as Float64 values.
        let narrow = |data_type: &DataType, values: Vec<f32>| {
            let array = cast(&Float32Array::from(values), data_type).unwrap();
            vec![batch_of(vec![("f", array)])]
        };
        let next = f32::from_bits(1.0f32.to_bits() + 1);
        for data_type in [DataType::Float32, DataType::Float16] {
            let values = vec![1.5, -0.0, f32::NAN, f32::NEG_INFINITY];
            let other_zero = vec![1.5, 0.0, f32::NAN, f32::NEG_INFINITY];
            same_results(
                narrow(&data_type, values),
                narrow(&data_type, other_zero),
                UNORDERED,
            )
            .unwrap();
            assert!(
                same_results(
                    narrow(&data_type, vec![1.5]),
                    narrow(&data_type, vec![1.75]),
                    UNORDERED
                )
                .is_err()
            );
        }
        let (one, next) = (
            narrow(&DataType::Float32, vec![1.0]),
            narrow(&DataType::Float32, vec![next]),
        );
        assert!(same_results(one.clone(), next.clone(), UNORDERED).is_err());
        let options = ComparisonOptions::default().with_float_tolerance(1e-6);
        same_results_with(one, next, UNORDERED, options).unwrap();
    }

    #[test]
    fn same_results_compare_unordered_rows_as_a_multiset() {
        let one = int_batch(&[Some(3), None, Some(1), Some(2), Some(1)]);
        let split = vec![
            int_batch(&[Some(1), Some(2)]),
            int_batch(&[]),
            int_batch(&[Some(1), None, Some(3)]),
        ];
        same_results(vec![one], split, UNORDERED).unwrap();

        // Duplicates count.
        let err = same_results(
            vec![int_batch(&[Some(1), Some(1), Some(2)])],
            vec![int_batch(&[Some(1), Some(2), Some(2)])],
            UNORDERED,
        )
        .unwrap_err();
        assert!(err.contains("compared as a multiset"), "{err}");
        assert!(
            err.contains(
                "1 row of DataFusion missing from the result of the pipeline engine:\n  \
                 row 1: (1)"
            ),
            "{err}"
        );
        assert!(
            err.contains(
                "1 row of the pipeline engine not in the result of DataFusion:\n  \
                 row 2: (2)"
            ),
            "{err}"
        );

        // Different numbers of rows.
        let err = same_results(
            vec![int_batch(&[Some(1)])],
            vec![int_batch(&[Some(1), None])],
            UNORDERED,
        )
        .unwrap_err();
        assert!(
            err.contains("different numbers of rows: 1 (DataFusion) and 2"),
            "{err}"
        );
        assert!(err.contains("row 1: (NULL)"), "{err}");
    }

    #[test]
    fn same_results_compare_ordered_rows_in_order_but_ties() {
        let rows = [(1, "a"), (2, "b"), (3, "c")];
        let expected = || vec![kv_batch(&rows)];
        let by_k = sorted_by(&expected(), &["k"], 0, None);
        let swapped = || vec![kv_batch(&[(2, "b"), (1, "a"), (3, "c")])];
        same_results(expected(), swapped(), UNORDERED).unwrap();
        let err = same_results(expected(), swapped(), &by_k).unwrap_err();
        assert!(
            err.starts_with(
                "row 0 of the pipeline engine differs (the result is ordered by (k)"
            ),
            "{err}"
        );
        assert!(err.contains("row 0: (2, \"b\")"), "{err}");
        let in_order = vec![kv_batch(&rows[..1]), kv_batch(&[]), kv_batch(&rows[1..])];
        same_results(expected(), in_order, &by_k).unwrap();

        // The rows that tie on the keys may come in any order.
        let ties = || {
            vec![kv_batch(&[
                (1, "a"),
                (1, "b"),
                (2, "c"),
                (2, "d"),
                (2, "e"),
            ])]
        };
        let other_order = || {
            vec![kv_batch(&[
                (1, "b"),
                (1, "a"),
                (2, "e"),
                (2, "c"),
                (2, "d"),
            ])]
        };
        let by_k = sorted_by(&ties(), &["k"], 0, None);
        same_results(ties(), other_order(), &by_k).unwrap();
        // ...also in the result of DataFusion.
        same_results(other_order(), ties(), &by_k).unwrap();
        // These rows do not tie on `(k, v)`.
        let by_kv = sorted_by(&ties(), &["k", "v"], 0, None);
        let err = same_results(ties(), other_order(), &by_kv).unwrap_err();
        assert!(
            err.starts_with(
                "row 0 of the pipeline engine differs (the result is ordered by (k, v)"
            ),
            "{err}"
        );
        // The rows that tie stay among the rows they tie with...
        let moved = vec![kv_batch(&[
            (1, "a"),
            (2, "c"),
            (1, "b"),
            (2, "d"),
            (2, "e"),
        ])];
        assert!(same_results(ties(), moved, &by_k).is_err());
        // ...and they are compared as a multiset.
        let other_rows = vec![kv_batch(&[
            (1, "a"),
            (1, "x"),
            (2, "c"),
            (2, "d"),
            (2, "e"),
        ])];
        let err = same_results(ties(), other_rows, &by_k).unwrap_err();
        assert!(
            err.starts_with("rows 0..2 of the pipeline engine differ"),
            "{err}"
        );
        assert!(err.contains("row 1: (1, \"b\")"), "{err}");
        assert!(err.contains("row 1: (1, \"x\")"), "{err}");
    }

    #[test]
    fn same_results_compare_the_rows_that_tie_at_a_limit_with_the_sorted_rows() {
        // The LIMIT cuts no rows: the rows are compared as usual.
        let two = || vec![kv_batch(&[(1, "a"), (2, "b")])];
        let limit_100 = sorted_by(&two(), &["k"], 0, Some(100));
        same_results(two(), two(), &limit_100).unwrap();
        let corrupted = vec![kv_batch(&[(1, "a"), (2, "BAD")])];
        let err = same_results(two(), corrupted, &limit_100).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine differs"),
            "{err}"
        );
        // The LIMIT cuts after the last row, but no row ties with it.
        let three = [kv_batch(&[(1, "a"), (2, "b"), (3, "c")])];
        let limit_2 = sorted_by(&three, &["k"], 0, Some(2));
        let corrupted = vec![kv_batch(&[(1, "a"), (2, "BAD")])];
        let err = same_results(two(), corrupted, &limit_2).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine differs"),
            "{err}"
        );

        // The LIMIT cuts the rows that tie with the last row: the last rows of
        // a result may be any of them, with their multiplicities.
        let sorted = [kv_batch(&[(1, "a"), (2, "b"), (2, "c"), (2, "d")])];
        let limit_3 = sorted_by(&sorted, &["k"], 0, Some(3));
        let expected = || vec![kv_batch(&[(1, "a"), (2, "b"), (2, "c")])];
        for actual in [
            [(1, "a"), (2, "b"), (2, "c")],
            [(1, "a"), (2, "d"), (2, "b")],
            [(1, "a"), (2, "c"), (2, "d")],
        ] {
            same_results(expected(), vec![kv_batch(&actual)], &limit_3).unwrap();
            // DataFusion's result may be any of them too.
            same_results(vec![kv_batch(&actual)], expected(), &limit_3).unwrap();
        }
        for (actual, unexpected) in [
            // A row that ties with the last row twice.
            ([(1, "a"), (2, "b"), (2, "b")], "row 2: (2, \"b\")"),
            // A row that does not tie with it.
            ([(1, "a"), (2, "x"), (2, "b")], "row 1: (2, \"x\")"),
            ([(1, "a"), (2, "b"), (3, "e")], "row 2: (3, \"e\")"),
        ] {
            let err =
                same_results(expected(), vec![kv_batch(&actual)], &limit_3).unwrap_err();
            assert!(
                err.starts_with(
                    "rows 1..3 of the pipeline engine differ: these rows tie with the \
                     last row of the result on the ORDER BY keys (k), after which the \
                     LIMIT cuts the sorted rows, so they may be any 2 rows of the 3 \
                     sorted rows of DataFusion that tie with it"
                ),
                "{err}"
            );
            assert!(
                err.contains(&format!(
                    "1 row of the pipeline engine not among them:\n  {unexpected}"
                )),
                "{err}"
            );
        }
        // The rows before them are compared as usual.
        let other_first = vec![kv_batch(&[(1, "z"), (2, "b"), (2, "c")])];
        let err = same_results(expected(), other_first, &limit_3).unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine differs"),
            "{err}"
        );
        // The result of DataFusion is compared with the sorted rows too.
        let corrupted = vec![kv_batch(&[(1, "a"), (2, "x"), (2, "b")])];
        let err = same_results(corrupted, expected(), &limit_3).unwrap_err();
        assert!(
            err.starts_with(
                "the result of DataFusion does not match its sorted reference"
            ),
            "{err}"
        );

        // The OFFSET skips some of the rows that tie with the first row.
        let sorted = [kv_batch(&[(1, "a"), (1, "b"), (2, "c"), (3, "d")])];
        let offset_1 = sorted_by(&sorted, &["k"], 1, Some(2));
        let expected = || vec![kv_batch(&[(1, "b"), (2, "c")])];
        same_results(expected(), vec![kv_batch(&[(1, "a"), (2, "c")])], &offset_1)
            .unwrap();
        for actual in [[(1, "x"), (2, "c")], [(1, "b"), (2, "z")]] {
            assert!(
                same_results(expected(), vec![kv_batch(&actual)], &offset_1).is_err()
            );
        }
        // The LIMIT and the OFFSET cut the same rows that tie.
        let sorted = [kv_batch(&[(5, "a"), (5, "b"), (5, "c"), (5, "d")])];
        let middle = sorted_by(&sorted, &["k"], 1, Some(2));
        let expected = || vec![kv_batch(&[(5, "b"), (5, "c")])];
        same_results(expected(), vec![kv_batch(&[(5, "d"), (5, "a")])], &middle).unwrap();
        let twice = vec![kv_batch(&[(5, "a"), (5, "a")])];
        assert!(same_results(expected(), twice, &middle).is_err());
    }

    #[test]
    fn same_results_see_the_ties_on_keys_that_are_not_selected() {
        // The query selects `v` of rows ordered by `k`: unique keys.
        let rows = [(1, "a"), (2, "b"), (3, "c")];
        let by_k = sorted_by_k(&rows, None);
        same_results(vec![v_batch(&rows)], vec![v_batch(&rows)], &by_k).unwrap();
        let reversed = [(3, "c"), (2, "b"), (1, "a")];
        let err = same_results(vec![v_batch(&rows)], vec![v_batch(&reversed)], &by_k)
            .unwrap_err();
        assert!(
            err.starts_with(
                "row 0 of the pipeline engine differs (the result is ordered by (m.k)"
            ),
            "{err}"
        );

        // Rows that tie on the key, which is not selected.
        let ties = [(1, "a"), (1, "b"), (2, "c")];
        let by_k = sorted_by_k(&ties, None);
        let swapped = [(1, "b"), (1, "a"), (2, "c")];
        same_results(vec![v_batch(&ties)], vec![v_batch(&swapped)], &by_k).unwrap();
        let moved = [(1, "a"), (2, "c"), (1, "b")];
        let err =
            same_results(vec![v_batch(&ties)], vec![v_batch(&moved)], &by_k).unwrap_err();
        assert!(
            err.starts_with("rows 0..2 of the pipeline engine differ"),
            "{err}"
        );

        // ...and that a LIMIT cuts.
        let sorted = [(1, "a"), (2, "b"), (2, "c"), (2, "d")];
        let limit_3 = sorted_by_k(&sorted, Some(3));
        let expected = || vec![v_batch(&sorted[..3])];
        let others = [(1, "a"), (2, "d"), (2, "b")];
        same_results(expected(), vec![v_batch(&others)], &limit_3).unwrap();
        for actual in [
            [(1, "a"), (2, "b"), (2, "b")],
            [(1, "a"), (2, "x"), (2, "b")],
        ] {
            let err =
                same_results(expected(), vec![v_batch(&actual)], &limit_3).unwrap_err();
            assert!(err.contains("not among them"), "{err}");
        }
    }

    #[test]
    fn same_results_accept_any_order_after_a_volatile_key() {
        // Without the values of a key (e.g. `random()`), all rows tie.
        let sorted = [kv_batch(&[(1, "a"), (2, "b"), (3, "c")])];
        let volatile = sorted_by(&sorted, &[], 0, None);
        let other_order = || vec![kv_batch(&[(3, "c"), (1, "a"), (2, "b")])];
        same_results(sorted.to_vec(), other_order(), &volatile).unwrap();
        let other_rows = vec![kv_batch(&[(3, "c"), (1, "a"), (2, "x")])];
        let err = same_results(sorted.to_vec(), other_rows, &volatile).unwrap_err();
        assert!(err.contains("the first ORDER BY key is volatile"), "{err}");
        // Under a LIMIT, the rows may be any of the sorted rows.
        let limit_2 = sorted_by(&sorted, &[], 0, Some(2));
        let expected = vec![kv_batch(&[(1, "a"), (2, "b")])];
        same_results(
            expected.clone(),
            vec![kv_batch(&[(3, "c"), (1, "a")])],
            &limit_2,
        )
        .unwrap();
        let twice = vec![kv_batch(&[(3, "c"), (3, "c")])];
        assert!(same_results(expected, twice, &limit_2).is_err());
    }

    #[test]
    fn same_results_compare_rows_in_their_exact_order_without_sorted_rows() {
        let exact = ResultOrder::Exact("a reason".to_string());
        let rows = || vec![kv_batch(&[(1, "a"), (1, "b")])];
        same_results(rows(), rows(), &exact).unwrap();
        let swapped = vec![kv_batch(&[(1, "b"), (1, "a")])];
        let err = same_results(rows(), swapped, &exact).unwrap_err();
        assert!(
            err.starts_with(
                "row 0 differs (the rows are compared in their exact order, as the \
                 sorted rows of the query could not be computed: a reason)"
            ),
            "{err}"
        );
    }

    #[test]
    fn same_results_compare_floats_nested_in_lists_and_structs() {
        let lists = |rows: Vec<Option<Vec<Option<f64>>>>| {
            vec![batch_of(vec![(
                "l",
                Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(rows)),
            )])]
        };
        let expected = || {
            vec![
                Some(vec![Some(1.0), Some(2.0)]),
                None,
                Some(vec![]),
                Some(vec![Some(f64::NAN), Some(-0.0), None]),
            ]
        };
        let mut close = expected();
        close[0] = Some(vec![Some(1.0 + 1e-12), Some(2.0 * (1.0 - 1e-12))]);
        close[3] = Some(vec![Some(f64::NAN), Some(0.0), None]);
        close.reverse();
        same_results(lists(expected()), lists(close), UNORDERED).unwrap();
        for (row, other) in [
            (0, Some(vec![Some(1.0), Some(2.1)])),
            (0, Some(vec![Some(1.0)])),
            (1, Some(vec![])),
            (3, Some(vec![Some(f64::NAN), Some(-0.0), Some(0.0)])),
        ] {
            let mut actual = expected();
            actual[row] = other;
            assert!(
                same_results(lists(expected()), lists(actual), UNORDERED).is_err(),
                "row {row}"
            );
        }

        // Structs of a float, a string and a list of floats.
        let structs = |rows: &[(f64, &str, f64)]| {
            let list = ListArray::from_iter_primitive::<Float64Type, _, _>(
                rows.iter().map(|row| Some(vec![Some(row.2)])),
            );
            let fields = Fields::from(vec![
                Field::new("x", DataType::Float64, true),
                Field::new("name", DataType::Utf8, true),
                Field::new("l", list.data_type().clone(), true),
            ]);
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.1))),
                Arc::new(list),
            ];
            let array = StructArray::new(fields, columns, None);
            vec![batch_of(vec![("s", Arc::new(array))])]
        };
        let expected = || structs(&[(1.0, "a", 10.0), (2.0, "b", 20.0)]);
        // Within the tolerance, in another order.
        let close =
            structs(&[(2.0 * (1.0 + 1e-12), "b", 20.0), (1.0, "a", 10.0 - 1e-11)]);
        same_results(expected(), close, UNORDERED).unwrap();
        for other in [
            structs(&[(1.0, "a", 10.0), (2.0, "c", 20.0)]),
            structs(&[(1.0, "a", 10.0), (2.1, "b", 20.0)]),
            structs(&[(1.0, "a", 10.0), (2.0, "b", 20.1)]),
        ] {
            let err = same_results(expected(), other, UNORDERED).unwrap_err();
            assert!(err.contains("row 1: ({2.0, \"b\", [20.0]})"), "{err}");
        }
    }

    #[test]
    fn same_results_compare_values_of_all_nested_types() {
        // Values of each type, and other values that are equal within the
        // tolerance, or different. Each array has the rows `[x, NULL]`.
        let large_lists = |x: f64| {
            let list = ListArray::from_iter_primitive::<Float64Type, _, _>([
                Some(vec![Some(x)]),
                None,
            ]);
            cast(&list, &DataType::new_large_list(DataType::Float64, true)).unwrap()
        };
        let list_views = |x: f64| {
            let list = ListArray::from_iter_primitive::<Float64Type, _, _>([
                Some(vec![Some(x)]),
                None,
            ]);
            let field = Field::new_list_field(DataType::Float64, true);
            cast(&list, &DataType::ListView(Arc::new(field))).unwrap()
        };
        let fixed_size_lists = |x: f64| -> ArrayRef {
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
                    [Some(vec![Some(x), Some(-x)]), None],
                    2,
                ),
            )
        };
        let maps = |x: f64| -> ArrayRef {
            let mut builder =
                MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
            builder.keys().append_value("x");
            builder.values().append_value(x);
            builder.append(true).unwrap();
            builder.append(false).unwrap();
            Arc::new(builder.finish())
        };
        let dictionaries = |x: f64| -> ArrayRef {
            // Other dictionaries for other values.
            let values = Float64Array::from(vec![x * 3.0, x]);
            let keys = Int8Array::from(vec![Some(1), None]);
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap(),
            )
        };
        let run_ends = |x: f64| -> ArrayRef {
            let run_ends = Int32Array::from(vec![1, 2]);
            let values = Float64Array::from(vec![Some(x), None]);
            Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap())
        };
        let unions = |x: f64| -> ArrayRef {
            let fields = UnionFields::try_new(
                [0, 1],
                [
                    Field::new("f", DataType::Float64, true),
                    Field::new("s", DataType::Utf8, true),
                ],
            )
            .unwrap();
            let children: Vec<ArrayRef> = vec![
                Arc::new(Float64Array::from(vec![Some(x), None])),
                Arc::new(StringArray::from(vec![Some("a"), None])),
            ];
            let type_ids = ScalarBuffer::from(vec![0, 1]);
            Arc::new(UnionArray::try_new(fields, type_ids, None, children).unwrap())
        };
        let arrays: [(&str, &dyn Fn(f64) -> ArrayRef); 7] = [
            ("large list", &large_lists),
            ("list view", &list_views),
            ("fixed size list", &fixed_size_lists),
            ("map", &maps),
            ("dictionary", &dictionaries),
            ("run-end encoded", &run_ends),
            ("union", &unions),
        ];
        for (name, array) in arrays {
            let batch = |x| vec![batch_of(vec![("a", array(x))])];
            same_results(batch(1.5), batch(1.5), UNORDERED).unwrap();
            same_results(batch(1.5), batch(1.5 * (1.0 + 1e-12)), UNORDERED)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let err = same_results(batch(1.5), batch(1.6), UNORDERED).unwrap_err();
            assert!(err.contains("1.6"), "{name}: {err}");
        }
    }

    /// A batch of `(x DOUBLE, y DOUBLE)` rows.
    fn xy_batch(rows: &[(f64, f64)]) -> RecordBatch {
        batch_of(vec![
            (
                "x",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "y",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.1))),
            ),
        ])
    }

    #[test]
    fn same_results_match_rows_whose_floats_are_close() {
        let close = 1.0 + 1e-12;
        // In their sorted order, `(1, 5)` would be paired with `(1, 3)`.
        let expected = || vec![xy_batch(&[(1.0, 5.0), (close, 3.0)])];
        let crossed = vec![xy_batch(&[(close, 5.0), (1.0, 3.0)])];
        same_results(expected(), crossed, UNORDERED).unwrap();
        let other = vec![xy_batch(&[(close, 5.0), (1.0, 3.5)])];
        let err = same_results(expected(), other, UNORDERED).unwrap_err();
        assert!(err.contains("row 1: (1.000000000001, 3.0)"), "{err}");
        assert!(err.contains("row 1: (1.0, 3.5)"), "{err}");

        // ORDER BY keys that are equal within the tolerance tie.
        let keyed = |rows: &[(f64, &str)]| {
            vec![batch_of(vec![
                (
                    "s",
                    Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.0)))
                        as ArrayRef,
                ),
                (
                    "v",
                    Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.1))),
                ),
            ])]
        };
        let expected = keyed(&[(1.0, "a"), (close, "b"), (2.0, "c")]);
        let by_s = sorted_by(&expected, &["s"], 0, None);
        let swapped = keyed(&[(1.0, "b"), (close, "a"), (2.0, "c")]);
        same_results(expected.clone(), swapped, &by_s).unwrap();
        let other_order = keyed(&[(2.0, "c"), (1.0, "b"), (close, "a")]);
        assert!(same_results(expected, other_order, &by_s).is_err());
    }

    #[test]
    fn same_results_match_any_number_of_rows_whose_floats_are_close() {
        let close = 1.0 + 1e-12;
        let repeat = |rows: &[(f64, f64)], times: usize| {
            let rows = rows.repeat(times);
            vec![xy_batch(&rows)]
        };
        // 1200 crossed pairs of rows.
        let expected = repeat(&[(1.0, 5.0), (close, 3.0)], 600);
        let crossed = repeat(&[(close, 5.0), (1.0, 3.0)], 600);
        same_results(expected.clone(), crossed, UNORDERED).unwrap();
        let other = repeat(&[(close, 5.0), (1.0, 3.5)], 600);
        let err = same_results(expected, other, UNORDERED).unwrap_err();
        assert!(
            err.contains("600 rows of DataFusion missing from the result"),
            "{err}"
        );

        // Rows that a greedy matching pairs the wrong way round: `u1` is
        // equal to `v1` and `v2`, but `u2` only to `v1`.
        let t = ComparisonOptions::DEFAULT_FLOAT_TOLERANCE;
        let (u1, u2) = ((1.0, 5.0), (1.0 + 0.5 * t, 5.0 * (1.0 - 1.8 * t)));
        let (v1, v2) = (
            (1.0 + 0.25 * t, 5.0 * (1.0 - 0.9 * t)),
            (1.0 + 0.25 * t, 5.0 * (1.0 + 0.9 * t)),
        );
        let expected = || repeat(&[u1, u2], 700);
        same_results(expected(), repeat(&[v1, v2], 700), UNORDERED).unwrap();
        let err =
            same_results(expected(), repeat(&[v2, v2], 700), UNORDERED).unwrap_err();
        assert!(
            err.contains("700 rows of DataFusion missing from the result"),
            "{err}"
        );
    }

    #[test]
    fn same_results_require_exactly_the_same_schemas() {
        let batch = int_batch(&[Some(1)]);
        same_results(vec![batch.clone()], vec![batch.clone()], UNORDERED).unwrap();
        let with_schema = |schema: Schema| {
            RecordBatch::try_new(Arc::new(schema), vec![Arc::clone(batch.column(0))])
                .unwrap()
        };
        let field = || Field::new("v", DataType::Int32, true);
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
        let int64 = batch_of(vec![("v", Arc::new(Int64Array::from(vec![1])))]);
        let lists = |nullable| {
            let list =
                ListArray::from_iter_primitive::<Int32Type, _, _>([Some([Some(1)])]);
            let field = Arc::new(Field::new_list_field(DataType::Int32, nullable));
            let (_, offsets, values, nulls) = list.into_parts();
            let list = ListArray::new(field, offsets, values, nulls);
            batch_of(vec![("v", Arc::new(list))])
        };
        for (expected, actual, difference) in [
            (
                &batch,
                with_schema(Schema::new(vec![Field::new("w", DataType::Int32, true)])),
                r#"a field is named "v" vs "w""#,
            ),
            (&batch, int64, r#"field "v" has the type Int32 vs Int64"#),
            (
                &batch,
                with_schema(Schema::new(vec![field().with_nullable(false)])),
                r#"field "v" is nullable vs not nullable"#,
            ),
            (
                &batch,
                with_schema(Schema::new(vec![field().with_metadata(metadata.clone())])),
                r#"field "v" has the metadata {} vs {"key": "value"}"#,
            ),
            (
                &batch,
                with_schema(Schema::new(vec![field()]).with_metadata(metadata.clone())),
                r#"the schema metadata is {} vs {"key": "value"}"#,
            ),
            (
                &batch,
                batch_of(vec![
                    ("v", Arc::clone(batch.column(0))),
                    ("w", Arc::clone(batch.column(0))),
                ]),
                "1 vs 2 fields",
            ),
            (&lists(true), lists(false), r#"field "v" has the type"#),
        ] {
            let err = same_results(vec![expected.clone()], vec![actual], UNORDERED)
                .unwrap_err();
            assert!(
                err.contains(&format!("the output schemas differ: {difference}")),
                "{err}"
            );
        }

        // The batches of the pipeline engine have the output schema, or the
        // schema of the batches of DataFusion.
        let options = ComparisonOptions::default();
        let expected = output(vec![batch.clone()], false);
        let mut actual = output(vec![batch.clone()], false);
        let schema = Arc::new(batch.schema().as_ref().clone().with_metadata(metadata));
        actual.batches = vec![batch.with_schema(schema).unwrap()];
        let err = compare_results(&expected, &actual, UNORDERED, &options).unwrap_err();
        assert!(
            err.contains(
                "the schema of a batch of the pipeline engine is not the output schema: \
                 the schema metadata is {} vs {\"key\": \"value\"}"
            ),
            "{err}"
        );
        let expected = EngineOutput {
            batches: actual.batches.clone(),
            ..expected
        };
        compare_results(&expected, &actual, UNORDERED, &options).unwrap();
    }

    #[test]
    fn same_results_require_both_engines_to_succeed_or_to_fail() {
        let failure = || {
            Err(DataFusionError::Execution(
                "Divide by zero error".to_string(),
            ))
        };
        let success = |value| Ok(output(vec![int_batch(&[Some(value)])], false));
        let outcomes = |datafusion, pipeline| EngineOutcomes {
            datafusion,
            pipeline,
            order: ResultOrder::Unordered,
        };
        let strict = ComparisonOptions::default();
        let lenient = strict.with_allow_error_differences(true);

        let datafusion_fails = outcomes(failure(), success(7));
        let err = datafusion_fails.compare(&strict).unwrap_err();
        assert!(
            err.starts_with("DataFusion failed, but the pipeline engine did not"),
            "{err}"
        );
        assert!(
            err.contains("DataFusion failed with: Execution error: Divide by zero error"),
            "{err}"
        );
        assert!(err.contains("the pipeline engine returned 1 row:"), "{err}");
        assert!(err.contains("| 7 |"), "{err}");
        datafusion_fails.compare(&lenient).unwrap();

        let pipeline_fails = outcomes(success(7), failure());
        let err = pipeline_fails.compare(&strict).unwrap_err();
        assert!(
            err.starts_with("the pipeline engine failed, but DataFusion did not"),
            "{err}"
        );
        assert!(err.contains("DataFusion returned 1 row:"), "{err}");
        assert!(err.contains("| 7 |"), "{err}");
        assert!(
            err.contains(
                "the pipeline engine failed with: Execution error: Divide by zero error"
            ),
            "{err}"
        );
        pipeline_fails.compare(&lenient).unwrap();

        // Which error is raised is the engine's own.
        let exhausted = DataFusionError::ResourcesExhausted("no memory".to_string());
        outcomes(failure(), Err(exhausted))
            .compare(&strict)
            .unwrap();
        // If both engines succeed, their results are compared.
        let err = outcomes(success(7), success(8))
            .compare(&lenient)
            .unwrap_err();
        assert!(err.contains("row 0: (8)"), "{err}");
    }

    #[test]
    fn same_results_ignore_the_physical_layout() {
        // Dictionaries over several batches, which cannot be concatenated.
        let batches = narrow_dictionary_batches();
        let reversed = batches.iter().rev().cloned().collect::<Vec<_>>();
        let split = batches
            .iter()
            .flat_map(|batch| [batch.slice(0, 30), batch.slice(30, 70)])
            .collect::<Vec<_>>();
        same_results(batches.clone(), reversed, UNORDERED).unwrap();
        let by_id = sorted_by(&batches, &["id"], 0, None);
        same_results(batches.clone(), split, &by_id).unwrap();

        // Dictionaries with other dictionary values.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )]));
        let dict = |keys: Vec<Option<i32>>, values: Vec<&str>| {
            let array = DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(keys),
                Arc::new(StringArray::from(values)),
            )
            .unwrap();
            vec![
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap(),
            ]
        };
        let expected = || dict(vec![Some(0), Some(0), Some(1), None], vec!["x", "y"]);
        let by_d = sorted_by(&expected(), &["d"], 0, None);
        same_results(
            expected(),
            dict(
                vec![Some(2), Some(2), Some(0), None],
                vec!["y", "unused", "x"],
            ),
            &by_d,
        )
        .unwrap();

        // Different values under null slots, and views of other buffers.
        let nulls = NullBuffer::from(vec![true, false, true]);
        let ints = |values: Vec<i32>| {
            let array = Int32Array::new(ScalarBuffer::from(values), Some(nulls.clone()));
            vec![batch_of(vec![("v", Arc::new(array))])]
        };
        same_results(ints(vec![1, 0, 3]), ints(vec![1, 99, 3]), UNORDERED).unwrap();
        let long = |i: usize| format!("a string that does not fit inline {i}");
        let values = (0..10).map(long).collect::<Vec<_>>();
        let views = |array: StringViewArray| vec![batch_of(vec![("s", Arc::new(array))])];
        same_results(
            views(StringViewArray::from_iter_values(values[2..6].iter())),
            views(StringViewArray::from_iter_values(values.iter()).slice(2, 4)),
            UNORDERED,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn result_order_of_queries() -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        register_test_tables(&ctx)?;
        // The keys, the number of output columns, the OFFSET, the LIMIT, and
        // the number of sorted rows.
        type Sorted<'a> = (&'a str, usize, usize, Option<usize>, usize);
        for (sql, expected) in [
            ("SELECT id FROM t1", None),
            ("SELECT id FROM t1 LIMIT 3", None),
            (
                "SELECT id, a FROM t1 ORDER BY a, id",
                Some(("t1.a, t1.id", 2, 0, None, 5000)),
            ),
            (
                "SELECT id AS x FROM t1 ORDER BY x DESC",
                Some(("x", 1, 0, None, 5000)),
            ),
            (
                "SELECT id, a FROM t1 ORDER BY 2 LIMIT 5 OFFSET 3",
                Some(("t1.a", 2, 3, Some(5), 5000)),
            ),
            (
                "SELECT d, id FROM t1 WHERE id < 100 ORDER BY d OFFSET 3",
                Some(("t1.d", 2, 3, None, 100)),
            ),
            // Keys that are not selected.
            (
                "SELECT a FROM t1 ORDER BY id LIMIT 3",
                Some(("t1.id", 1, 0, Some(3), 5000)),
            ),
            (
                "SELECT id, a FROM t1 ORDER BY a, b, id",
                Some(("t1.a, t1.b, t1.id", 2, 0, None, 5000)),
            ),
            (
                "SELECT a, b FROM t1 ORDER BY a + b",
                Some(("t1.a + t1.b", 2, 0, None, 5000)),
            ),
            (
                "SELECT a, MAX(c) FROM t1 GROUP BY a ORDER BY MAX(c) DESC LIMIT 3",
                Some(("max(t1.c)", 2, 0, Some(3), 102)),
            ),
            // A volatile key, and the keys after it, cannot be compared.
            (
                "SELECT id FROM t1 ORDER BY a, random(), id",
                Some(("t1.a", 1, 0, None, 5000)),
            ),
            // Through subqueries and nested limits.
            (
                "SELECT * FROM (SELECT id FROM t1 ORDER BY id LIMIT 10) AS s LIMIT 3",
                Some(("t1.id", 1, 0, Some(3), 5000)),
            ),
            (
                "SELECT * FROM (SELECT id, a FROM t1) AS s(x, y) ORDER BY y, x",
                Some(("s.y, s.x", 2, 0, None, 5000)),
            ),
            (
                "SELECT a FROM (SELECT a, id FROM t1 ORDER BY id LIMIT 20 OFFSET 1) \
                 LIMIT 5 OFFSET 2",
                Some(("t1.id", 1, 3, Some(5), 5000)),
            ),
            (
                "SELECT a FROM (SELECT a, id FROM t1 ORDER BY id LIMIT 4) LIMIT 5 OFFSET 2",
                Some(("t1.id", 1, 2, Some(2), 5000)),
            ),
        ] {
            let expected: Option<Sorted<'_>> = expected;
            let df = ctx.sql(sql).await?;
            let plan = df.logical_plan();
            let order = result_order(&ctx, plan).await;
            let sorted = match &order {
                ResultOrder::Unordered => None,
                ResultOrder::Sorted(reference) => Some(reference),
                ResultOrder::Exact(reason) => panic!("{sql}: {reason}"),
            };
            assert_eq!(sorted.is_some(), has_top_level_order_by(plan), "{sql}");
            let actual = sorted.map(|reference| {
                let rows = reference.batches.iter().map(RecordBatch::num_rows).sum();
                let keys = reference.keys.join(", ");
                (
                    keys,
                    reference.columns,
                    reference.skip,
                    reference.fetch,
                    rows,
                )
            });
            let actual = actual.as_ref().map(|(keys, columns, skip, fetch, rows)| {
                (keys.as_str(), *columns, *skip, *fetch, *rows)
            });
            assert_eq!(actual, expected, "{sql}\n{}", plan.display_indent());
        }

        // The sorted rows have the values of the keys after the output
        // columns.
        let df = ctx.sql("SELECT a FROM t1 ORDER BY id DESC LIMIT 3").await?;
        let ResultOrder::Sorted(reference) = result_order(&ctx, df.logical_plan()).await
        else {
            panic!("not sorted");
        };
        let rows = reference.reference_rows(0.0)?;
        let ids = rows
            .keys
            .iter()
            .map(|key| key[0].to_string())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["4999", "4998", "4997"].map(String::from));
        assert!(rows.first_ties.is_none() && rows.last_ties.is_none());

        // A LIMIT that is not a literal.
        let df = ctx.sql("SELECT id FROM t1 ORDER BY id LIMIT 1 + 1").await?;
        let order = result_order(&ctx, df.logical_plan()).await;
        assert!(
            matches!(&order, ResultOrder::Exact(reason) if reason.contains("not a literal")),
            "{order:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_over_the_test_tables() {
        for sql in [
            "SELECT * FROM t1",
            "SELECT * FROM t_empty",
            "SELECT d, COUNT(*) AS n, SUM(c) AS s, AVG(c) AS m, STDDEV(c) AS sd FROM t1 GROUP BY d",
            // Ties on the ORDER BY keys, also under LIMIT and OFFSET.
            "SELECT e, d, id FROM t1 ORDER BY e NULLS FIRST",
            "SELECT d, id FROM t1 ORDER BY d LIMIT 10",
            "SELECT d, a FROM t1 ORDER BY d DESC LIMIT 7 OFFSET 400",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t2.name LIMIT 30",
            "SELECT * FROM t1 ORDER BY id LIMIT 0",
            "SELECT * FROM t_empty ORDER BY id LIMIT 3",
            // Keys that are not selected, or functions of the output.
            "SELECT a FROM t1 ORDER BY id DESC LIMIT 5",
            "SELECT id FROM t1 ORDER BY d, a LIMIT 50",
            "SELECT a, b FROM t1 WHERE id < 100 ORDER BY a + b, a",
            "SELECT h FROM t1 ORDER BY e, random() LIMIT 20",
            // Floats in lists and structs.
            "SELECT id, make_array(c, c / 3) AS l, named_struct('c', c, 'd', d) AS s FROM t1",
        ] {
            assert_same_results_as_datafusion(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_with_setup_and_config() {
        let setup = async |ctx: &SessionContext| {
            register_test_tables(ctx)?;
            ctx.register_batch(
                "kv",
                kv_batch(&[(1, "a"), (2, "b"), (2, "c"), (3, "d")]),
            )?;
            Ok(())
        };
        assert_same_results_as_datafusion_with_setup(
            "SELECT k, v FROM kv ORDER BY k DESC LIMIT 3",
            &setup,
        )
        .await;
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT d, SUM(c) AS s, AVG(c) AS m FROM t1 GROUP BY d",
                "SELECT k, v, id FROM kv JOIN t1 ON kv.k = t1.a ORDER BY k LIMIT 40",
                "SELECT v FROM kv JOIN t1 ON kv.k = t1.a ORDER BY t1.id LIMIT 40",
            ] {
                assert_same_results_as_datafusion_with_config(
                    sql,
                    config.clone(),
                    &setup,
                )
                .await;
            }
        }
    }

    /// A setup that registers `datafusion` as table `m` in the session of
    /// DataFusion, and `pipeline` in the session of the pipeline engine (see
    /// [`run_both_engines`]).
    fn per_engine_tables(
        datafusion: RecordBatch,
        pipeline: RecordBatch,
    ) -> impl AsyncFn(&SessionContext) -> Result<()> {
        let sessions = AtomicUsize::new(0);
        async move |ctx: &SessionContext| {
            let batch = if sessions
                .fetch_add(1, AtomicOrdering::Relaxed)
                .is_multiple_of(2)
            {
                &datafusion
            } else {
                &pipeline
            };
            ctx.register_batch("m", batch.clone())?;
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_rounding_differences() {
        let table = |scale: f64| {
            let x = (0..100).map(|i| (f64::from(i) * 0.1 - 3.3) * scale);
            batch_of(vec![
                (
                    "k",
                    Arc::new(Int32Array::from_iter_values(0..100)) as ArrayRef,
                ),
                ("x", Arc::new(Float64Array::from_iter_values(x))),
            ])
        };
        // The values of the pipeline engine are rounded differently.
        let setup = per_engine_tables(table(1.0), table(1.0 + 1e-12));
        for sql in [
            "SELECT k, x FROM m",
            "SELECT k, x * 2 AS y FROM m ORDER BY x DESC LIMIT 10",
            "SELECT k % 7 AS g, SUM(x) AS s FROM m GROUP BY k % 7",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
            // Their bytes differ.
            let (expected, actual) = run_engines(sql, test_session_config(), &setup)
                .await
                .unwrap();
            assert!(compare_outputs(&expected, &actual).is_err(), "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_other_rows_that_tie() {
        // The tables of both engines have the same rows in other orders.
        let sorted = [(1, "a"), (2, "b"), (2, "c"), (2, "d"), (3, "e")];
        let shuffled = [(2, "d"), (3, "e"), (2, "c"), (1, "a"), (2, "b")];
        let setup = per_engine_tables(kv_batch(&sorted), kv_batch(&shuffled));
        for sql in [
            "SELECT k, v FROM m ORDER BY k LIMIT 3",
            "SELECT k, v FROM m ORDER BY k LIMIT 2 OFFSET 2",
            "SELECT v FROM m ORDER BY k",
            "SELECT v FROM m ORDER BY k LIMIT 3",
            "SELECT v FROM m ORDER BY k DESC LIMIT 2 OFFSET 1",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(expected = "Results differ: row 1 of the pipeline engine differs")]
    async fn assert_same_results_panics_on_a_corrupted_row_under_a_limit() {
        // The LIMIT cuts no rows.
        let setup = per_engine_tables(
            kv_batch(&[(1, "a"), (2, "b")]),
            kv_batch(&[(1, "a"), (2, "BAD")]),
        );
        assert_same_results_as_datafusion_with_setup(
            "SELECT k, v FROM m ORDER BY k LIMIT 100",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(
        expected = "1 row of the pipeline engine not among them:\n  row 2: (2, \"b\")"
    )]
    async fn assert_same_results_panics_on_a_tied_row_twice_under_a_limit() {
        let setup = per_engine_tables(
            kv_batch(&[(1, "a"), (2, "b"), (2, "c"), (2, "d")]),
            kv_batch(&[(1, "a"), (2, "b"), (2, "b"), (2, "b")]),
        );
        assert_same_results_as_datafusion_with_setup(
            "SELECT k, v FROM m ORDER BY k LIMIT 3",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(
        expected = "Results differ: row 0 of the pipeline engine differs (the result is ordered by (m.k)"
    )]
    async fn assert_same_results_panics_on_rows_out_of_the_order_of_keys_not_selected() {
        let setup = per_engine_tables(
            kv_batch(&[(1, "a"), (2, "b"), (3, "c")]),
            kv_batch(&[(1, "c"), (2, "b"), (3, "a")]),
        );
        assert_same_results_as_datafusion_with_setup(
            "SELECT v FROM m ORDER BY k",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(expected = "Results differ: the rows differ")]
    async fn assert_same_results_panics_on_different_rows() {
        let setup = per_engine_tables(
            int_batch(&[Some(1), Some(2)]),
            int_batch(&[Some(1), Some(3)]),
        );
        assert_same_results_as_datafusion_with_setup("SELECT v FROM m", &setup).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(
        expected = r#"Results differ: the output schemas differ: field "v" is nullable vs not nullable"#
    )]
    async fn assert_same_results_panics_on_different_schemas() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let not_null =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let setup = per_engine_tables(int_batch(&[Some(1)]), not_null);
        assert_same_results_as_datafusion_with_setup("SELECT v FROM m", &setup).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(
        expected = "Results differ: the pipeline engine failed, but DataFusion did not"
    )]
    async fn assert_same_results_panics_if_only_one_engine_fails() {
        let setup = per_engine_tables(
            int_batch(&[Some(1), Some(2)]),
            int_batch(&[Some(1), Some(0)]),
        );
        assert_same_results_as_datafusion_with_setup("SELECT 10 / v AS q FROM m", &setup)
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_may_allow_error_differences() {
        let sql = "SELECT 10 / v AS q FROM m";
        let (some, zero) = (&[Some(1), Some(2)], &[Some(1), Some(0)]);
        let options = ComparisonOptions::default().with_allow_error_differences(true);
        for (datafusion, pipeline) in [(some, zero), (zero, some)] {
            let setup = per_engine_tables(int_batch(datafusion), int_batch(pipeline));
            let outcomes = run_both_engines(sql, test_session_config(), &setup)
                .await
                .unwrap();
            assert_ne!(outcomes.datafusion.is_ok(), outcomes.pipeline.is_ok());
            assert_same_results_as_datafusion_with_options(
                sql,
                test_session_config(),
                &setup,
                options,
            )
            .await;
        }
        // If both engines fail, their errors may differ.
        let setup = per_engine_tables(int_batch(&[Some(0)]), int_batch(zero));
        assert_same_results_as_datafusion_with_setup(sql, &setup).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Failed to run query on both engines")]
    async fn assert_same_results_panics_on_invalid_query() {
        assert_same_results_as_datafusion("SELECT no_such_column FROM t1").await;
    }
}
