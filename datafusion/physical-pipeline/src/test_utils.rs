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
//!      [`has_top_level_order_by`], which also sees the `ORDER BY` that a
//!      `DISTINCT ON` absorbs);
//!    * in order if it has one: the result must be the rows that the query
//!      returns from an order of all its rows in which a row is only out of
//!      DataFusion's order relative to rows that it ties with. The keys are
//!      compared in their order (lexicographically, also the elements of
//!      lists and the fields of structs), and the keys of each engine may be
//!      off by the tolerance: another engine may have for a floating-point
//!      value any value of its precision that is equal to it within the
//!      tolerance. So rows tie if their `ORDER BY` keys are equal up to their
//!      first floating-point value that may vary within the tolerance, and
//!      the values that the engines may have for these values cross (either
//!      may be smaller), whatever their later keys. If they only meet (they
//!      may be equal, but neither smaller), the later keys decide.
//!      Floating-point values that the tolerance does not let vary (zeros,
//!      infinities and `NaN`s, and any value whose neighbors are farther than
//!      the tolerance, e.g. within a tolerance of `0`) are compared like other
//!      values: rows that are equal on them are ordered by their later keys.
//!      DataFusion sorts `-0.0` before `0.0`, but they are equal, so rows
//!      whose zeros only differ in sign may also be ordered by their later
//!      keys. Likewise, it sorts negative `NaN`s first and positive ones last,
//!      but `NaN = NaN`: the rows may also be in the order of the query whose
//!      top-level sort has all its `NaN`s positive, or all negative (see
//!      [`nan_alternatives`]).
//!      Ties are not transitive: rows that tie with a
//!      row may not tie with each other. So rows that tie may come in any
//!      order among themselves, and under `ORDER BY … LIMIT`, the last rows
//!      may be other rows that tie with them (with their multiplicities), as
//!      may, under `OFFSET`, the first rows;
//!    * their values are compared by meaning, not by representation: batch
//!      boundaries, dictionary and run-end encoding, views, offsets and the
//!      bytes under nulls do not matter;
//!    * floating-point values (also inside lists, structs, maps and unions)
//!      are equal within a relative tolerance:
//!      [`ComparisonOptions::DEFAULT_FLOAT_TOLERANCE`] (`1e-9`) by default,
//!      see [`ComparisonOptions::with_float_tolerance`]. `NaN` equals `NaN`
//!      and `-0.0` equals `0.0`, but `NULL` only equals `NULL`.
//!
//! To know which rows tie and which rows `LIMIT` and `OFFSET` cut, DataFusion
//! also sorts all rows of the top-level sort of the query (without its
//! `LIMIT` and `OFFSET`), with the values of its `ORDER BY` keys (also of the
//! keys that the query does not select), and both results are compared with
//! these sorted rows, its [`SortedReference`] (see [`result_order`]). The
//! output columns of the query (its expressions above the sort) are only
//! computed for the sorted rows that it may return: the rows that it returns,
//! and the rows whose keys may tie with theirs. So they are not evaluated on
//! rows that the query cuts, and the query returns no rows on which they fail
//! (e.g. on a division by zero). The keys from the
//! first key whose values depend on a volatile function (e.g. `random()`,
//! also through the columns of a subquery) are not compared, as their values
//! change from one run to the next: the rows that tie on the keys before it
//! may come in any order. If the sorted rows cannot be computed (e.g. for a
//! `LIMIT` that is not a constant), the rows are compared in their exact
//! order.
//!
//! The rows that the nested `ORDER BY … LIMIT`s of a query (below its
//! top-level sort, also in subqueries) return may also be other rows that tie
//! at their windows, and the rows that its `DISTINCT ON`s return for each
//! group may be any rows that tie for the first row of the group. If the results differ, DataFusion also runs the query
//! for other rows of its nested limits, and the result of the pipeline engine
//! may be any of these results (see [`alternative_outcomes`]). If the query
//! only has projections, filters, sorts, subquery aliases and inner joins
//! above its nested limits, the rows of the result of the pipeline engine
//! show which rows its nested limits returned, which may be any rows that
//! they may return. Otherwise (e.g. for an aggregate over them), one physical
//! plan of the query runs for up to 20,000 choices of their rows, of one or
//! several at once. Each occurrence of a nested limit may return other rows,
//! and the rows that a nested limit may return also depend on the rows of the
//! nested limits inside it: they are found for each choice of those rows.
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

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{self, Write};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow::array::{
    Array, ArrayRef, AsArray, RecordBatch, RecordBatchOptions, RunArray, UInt32Array,
    new_null_array,
};
use arrow::compute::{SortOptions, cast, concat_batches, take};
use arrow::datatypes::{
    ArrowNativeType, ArrowPrimitiveType, DataType, Field, FieldRef, Float16Type,
    Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, RunEndIndexType, Schema,
    SchemaRef, UInt32Type,
};
use arrow::downcast_dictionary_array;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow::util::pretty::pretty_format_batches;
use arrow_ipc::writer::StreamWriter;
use arrow_row::{RowConverter, SortField};
use datafusion::catalog::TableProvider;
use datafusion::catalog::streaming::StreamingTable;
use datafusion::dataframe::DataFrame;
use datafusion::datasource::{MemTable, provider_as_source};
use datafusion::execution::TaskContext;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::{SessionState, SessionStateBuilder};
use datafusion::functions::math::expr_fn::isnan;
use datafusion::physical_planner::DefaultPhysicalPlanner;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Column, DFSchema, DFSchemaRef, DataFusionError, JoinType, Result, ScalarValue,
    TableReference, internal_err, not_impl_err,
};
use datafusion_expr::logical_plan::{
    Distinct, DistinctOn, FetchType, Limit, Projection, SkipType, Sort, SubqueryAlias,
};
use datafusion_expr::{
    ColumnarValue, Expr, ExprSchemable, LogicalPlan, LogicalPlanBuilder, SortExpr,
};
use datafusion_physical_pipeline::{PipelineQueryPlanner, PipelineRootExec};
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::streaming::PartitionStream;
use datafusion_physical_plan::{
    ExecutionPlan, SendableRecordBatchStream, collect, displayable,
};
use itertools::Itertools;

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
        // `DISTINCT ON` absorbs the `ORDER BY` of its query.
        LogicalPlan::Distinct(Distinct::On(distinct)) => distinct.sort_expr.is_some(),
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
    let outcomes = match run_both_engines(sql, config, setup, &options).await {
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
    /// Other outputs of DataFusion that the query may return, with the order
    /// of their rows: for other rows that its nested `ORDER BY … LIMIT`s may
    /// return, see [`alternative_outcomes`]. Only computed if the output of
    /// the pipeline engine differs from the output of DataFusion.
    pub alternatives: Vec<(EngineOutput, ResultOrder)>,
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
                let difference = compare_results(expected, actual, &self.order, options);
                match difference {
                    // The schemas must be the schema of DataFusion, which the
                    // other results of the query have too.
                    Err(difference)
                        if compare_schemas(expected, actual).is_err()
                            || !self.alternatives.iter().any(
                                |(alternative, order)| {
                                    let alternative = EngineOutput {
                                        plan: Arc::clone(&alternative.plan),
                                        schema: Arc::clone(&expected.schema),
                                        batches: alternative.batches.clone(),
                                        ordered: alternative.ordered,
                                    };
                                    compare_results(&alternative, actual, order, options)
                                        .is_ok()
                                },
                            ) =>
                    {
                        Err(if self.alternatives.is_empty() {
                            difference
                        } else {
                            format!(
                                "{difference}
(nor is the result of the pipeline \
                                 engine one of the {} other results of the query for \
                                 other rows that tie at the windows of its nested ORDER \
                                 BY … LIMITs)",
                                self.alternatives.len()
                            )
                        })
                    }
                    _ => Ok(()),
                }
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
/// [`result_order`] of the query for the comparison `options` (for which
/// DataFusion sorts the rows of the query again if it has a top-level
/// `ORDER BY`). Compare the outcomes with the same options, see
/// [`EngineOutcomes::compare`].
///
/// Fails if `setup` fails, if `sql` is not a valid query (DataFusion cannot
/// plan it logically), or if the pipeline engine did not plan the query.
pub async fn run_both_engines<F>(
    sql: &str,
    config: SessionConfig,
    setup: F,
    options: &ComparisonOptions,
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
    let order = result_order(&datafusion, &plan, options).await;
    let alternatives = match (&datafusion_outcome, &pipeline_outcome) {
        (Ok(expected), Ok(actual))
            if compare_schemas(expected, actual).is_ok()
                && compare_results(expected, actual, &order, options).is_err() =>
        {
            let mut alternatives =
                alternative_outcomes(&datafusion, &plan, options, actual).await;
            alternatives.extend(nan_alternatives(&datafusion, &plan, options).await);
            alternatives
        }
        _ => vec![],
    };
    Ok(EngineOutcomes {
        datafusion: datafusion_outcome,
        pipeline: pipeline_outcome,
        order,
        alternatives,
    })
}

/// The name of the column of the lineage of the rows of the `i`th nested
/// limit of a query, see [`lineage_choices`].
fn lineage_column(i: usize) -> String {
    format!("__test_utils_lineage_{i}")
}

/// A node of a query whose rows may be other rows that tie (see
/// [`nested_limits`]): a nested `ORDER BY … LIMIT`, or a `DISTINCT ON`, which
/// returns one row of each group, among the rows that tie for its first row.
struct NestedLimit {
    /// Its plan, in the plan of the query.
    plan: LogicalPlan,
    kind: NestedKind,
}

/// The kind of a [`NestedLimit`], and the rows that it may return.
enum NestedKind {
    /// A nested `ORDER BY … LIMIT`: its sorted reference, whose sorted rows it
    /// returns, and these sorted rows.
    Limit {
        reference: SortedReference,
        sorted: SortedRows,
    },
    /// A `DISTINCT ON`: the rows of its input (its output columns, then the
    /// values of its `ON` expressions), in the order of its `ORDER BY`, the
    /// rows that may be the row of each group (positions), the sort options
    /// of its `ON` expressions in its `ORDER BY` (if any), and the rows that
    /// DataFusion returns.
    DistinctOn {
        rows: RecordBatch,
        groups: Vec<Vec<usize>>,
        order: Option<Vec<SortOptions>>,
        own: Vec<usize>,
    },
}

/// The name of the column of the values of the `j`th `ON` expression of a
/// `DISTINCT ON`, see [`NestedLimit::rows`].
fn on_column(j: usize) -> String {
    format!("__test_utils_distinct_on_{j}")
}

impl NestedLimit {
    /// The nested limit or the `DISTINCT ON` `plan` of a query, planned by
    /// `ctx`, with the rows that it may return within the tolerance
    /// `tolerance`.
    async fn new(
        ctx: &SessionContext,
        plan: LogicalPlan,
        tolerance: f64,
    ) -> Result<Self> {
        let LogicalPlan::Distinct(Distinct::On(distinct)) = &plan else {
            let reference = sorted_reference(ctx, &plan, tolerance).await?;
            let sorted = reference.sorted_rows(tolerance)?;
            return Ok(Self {
                plan,
                kind: NestedKind::Limit { reference, sorted },
            });
        };
        // The rows of its input with their output columns, the values of its
        // `ON` expressions, and the values of the rest of its `ORDER BY`.
        let sort_expr = distinct.sort_expr.clone().unwrap_or_default();
        let on = distinct.on_expr.len();
        let column = |name: &str| Expr::Column(Column::from_name(name));
        let mut exprs = distinct
            .select_expr
            .iter()
            .enumerate()
            .map(|(i, expr)| expr.clone().unalias().alias(format!("c{i}")))
            .collect::<Vec<_>>();
        let width = exprs.len();
        let mut order = vec![];
        for (j, expr) in distinct.on_expr.iter().enumerate() {
            exprs.push(expr.clone().unalias().alias(on_column(j)));
            let (asc, nulls_first) = sort_expr
                .get(j)
                .map_or((true, false), |sort| (sort.asc, sort.nulls_first));
            order.push(SortExpr::new(column(&on_column(j)), asc, nulls_first));
        }
        let mut rest = vec![];
        for (l, sort) in sort_expr.iter().enumerate().skip(on) {
            let name = format!("__test_utils_distinct_order_{l}");
            exprs.push(sort.expr.clone().unalias().alias(&name));
            order.push(SortExpr::new(column(&name), sort.asc, sort.nulls_first));
            rest.push(SortOptions {
                descending: !sort.asc,
                nulls_first: sort.nulls_first,
            });
        }
        let input = Projection::try_new(exprs, Arc::clone(&distinct.input))?;
        let sorted = LogicalPlan::Sort(Sort {
            expr: order,
            input: Arc::new(LogicalPlan::Projection(input)),
            fetch: None,
        });
        let batches = ctx
            .execute_logical_plan(sorted.clone())
            .await?
            .collect()
            .await?;
        let schema = batches.first().map_or_else(
            || Arc::new(sorted.schema().as_arrow().clone()),
            RecordBatch::schema,
        );
        let rows = concat_batches(&schema, &batches)?;
        let values = |row: usize, columns: Range<usize>| {
            columns
                .map(|column| Value::try_new(rows.column(column).as_ref(), row))
                .collect::<Result<Row>>()
        };
        // The groups of rows of equal values of the `ON` expressions, and the
        // rows of each group that may be first, as they tie on the rest of the
        // `ORDER BY`.
        let (mut groups, mut own) = (vec![], vec![]);
        let mut start = 0;
        while start < rows.num_rows() {
            let keys = values(start, width..width + on)?;
            let mut end = start + 1;
            while end < rows.num_rows()
                && rows_equal(&values(end, width..width + on)?, &keys, 0.0)
            {
                end += 1;
            }
            let later = (start..end)
                .map(|row| values(row, width + on..rows.num_columns()))
                .collect::<Result<Vec<_>>>()?;
            let (classes, tied) = classes_and_ties(&later, &rest, tolerance);
            let classes = classes
                .into_iter()
                .map(|class| (class.rows, class.keys))
                .collect::<Vec<_>>();
            let first = bounds(&classes, &tied);
            groups.push(
                classes
                    .iter()
                    .zip(first)
                    .filter(|(_, (first, _))| *first == 0)
                    .flat_map(|((rows, _), _)| rows.clone())
                    .map(|row| start + row)
                    .collect(),
            );
            own.push(start);
            start = end;
        }
        let order = distinct.sort_expr.as_ref().map(|_| {
            sort_expr[..on.min(sort_expr.len())]
                .iter()
                .map(|sort| SortOptions {
                    descending: !sort.asc,
                    nulls_first: sort.nulls_first,
                })
                .collect()
        });
        Ok(Self {
            plan,
            kind: NestedKind::DistinctOn {
                rows,
                groups,
                order,
                own,
            },
        })
    }

    /// The fields of the output schema of the node, with their qualifiers.
    fn fields(&self) -> Vec<(Option<TableReference>, FieldRef)> {
        self.plan
            .schema()
            .iter()
            .map(|(qualifier, field)| (qualifier.cloned(), Arc::clone(field)))
            .collect()
    }

    /// The fields of the rows of [`NestedLimit::columns`].
    fn row_fields(&self) -> Vec<(Option<TableReference>, FieldRef)> {
        let mut fields = self.fields();
        if let NestedKind::DistinctOn { rows, .. } = &self.kind {
            fields.extend(
                rows.schema()
                    .fields()
                    .iter()
                    .filter(|field| field.name().starts_with("__test_utils_distinct_on_"))
                    .map(|field| (None, Arc::clone(field))),
            );
        }
        fields
    }

    /// The output columns of the rows `rows` (positions), followed by the
    /// values of the `ON` expressions of a `DISTINCT ON`.
    fn columns(&self, rows: &[usize]) -> Result<Vec<ArrayRef>> {
        let positions = positions_array(rows)?;
        let (batch, width) = match &self.kind {
            NestedKind::Limit { reference, .. } => (
                reference.batches.first().ok_or_else(|| {
                    DataFusionError::Internal("no sorted rows".to_string())
                })?,
                reference.columns,
            ),
            NestedKind::DistinctOn { rows, .. } => (rows, self.row_fields().len()),
        };
        Ok(batch.columns()[..width]
            .iter()
            .map(|column| take(column, &positions, None))
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// The rows of `table` (of the columns of [`NestedLimit::columns`], then
    /// of the column `lineage`), as a plan of the output schema of the node,
    /// followed by the lineage: in the order of the `ON` expressions of a
    /// `DISTINCT ON` with an `ORDER BY`.
    fn plan_over(
        &self,
        table: Arc<dyn TableProvider>,
        lineage: Option<&str>,
    ) -> Result<LogicalPlan> {
        let mut fields = self.row_fields();
        if let Some(lineage) = lineage {
            fields.push((None, Arc::new(Field::new(lineage, DataType::UInt32, false))));
        }
        let rows = rows_plan(table, &fields)?;
        let NestedKind::DistinctOn { order, .. } = &self.kind else {
            return Ok(rows);
        };
        let rows = match order {
            Some(order) => LogicalPlan::Sort(Sort {
                expr: order
                    .iter()
                    .enumerate()
                    .map(|(j, options)| {
                        SortExpr::new(
                            Expr::Column(Column::from_name(on_column(j))),
                            !options.descending,
                            options.nulls_first,
                        )
                    })
                    .collect(),
                input: Arc::new(rows),
                fetch: None,
            }),
            None => rows,
        };
        // Without the values of the `ON` expressions.
        let mut exprs = self
            .fields()
            .iter()
            .map(|(qualifier, field)| {
                Expr::Column(Column::new(qualifier.clone(), field.name()))
            })
            .collect::<Vec<_>>();
        if let Some(lineage) = lineage {
            exprs.push(Expr::Column(Column::from_name(lineage)));
        }
        Ok(LogicalPlan::Projection(Projection::try_new(
            exprs,
            Arc::new(rows),
        )?))
    }

    /// The rows `rows` (positions), as a plan of the output schema of the
    /// node, followed by their positions in the column `lineage`.
    fn rows(&self, rows: &[usize], lineage: Option<&str>) -> Result<LogicalPlan> {
        let mut columns = self.columns(rows)?;
        let mut fields = self.row_fields();
        if let Some(lineage) = lineage {
            columns.push(Arc::new(positions_array(rows)?));
            fields.push((None, Arc::new(Field::new(lineage, DataType::UInt32, false))));
        }
        let batch = rows_batch(columns, &fields, self.plan.schema().metadata())?;
        let table = MemTable::try_new(batch.schema(), vec![vec![batch]])?;
        self.plan_over(Arc::new(table), lineage)
    }

    /// The rows that DataFusion returns (positions).
    fn own(&self) -> Vec<usize> {
        match &self.kind {
            NestedKind::Limit { sorted, .. } => sorted.window.clone().collect(),
            NestedKind::DistinctOn { own, .. } => own.clone(),
        }
    }

    /// The rows that the node may return (positions).
    fn candidates(&self) -> Vec<usize> {
        match &self.kind {
            NestedKind::Limit { sorted, .. } => candidates(sorted).0,
            NestedKind::DistinctOn { groups, .. } => {
                groups.iter().flatten().copied().collect()
            }
        }
    }

    /// Up to `max` choices of the rows that the node may return (each the
    /// positions of the rows, in their order), but for the rows that
    /// DataFusion returns: of the rows `required` and `optional` only (all
    /// rows if `None`), with as many rows of `required` as they may have, first.
    fn choices(
        &self,
        pools: Option<(&BTreeSet<usize>, &BTreeSet<usize>)>,
        max: usize,
    ) -> Vec<Vec<usize>> {
        let own = self.own();
        match &self.kind {
            NestedKind::Limit { sorted, .. } => {
                let Some((required, optional)) = pools else {
                    return window_choices(sorted, max);
                };
                let size = sorted.window.len();
                let mut choices = vec![];
                for count in (0..=required.len().min(size)).rev() {
                    for chosen in required.iter().copied().combinations(count).take(4096)
                    {
                        for rest in optional
                            .iter()
                            .copied()
                            .filter(|row| !chosen.contains(row))
                            .combinations(size - count)
                            .take(4096)
                        {
                            let mut choice =
                                chosen.iter().copied().chain(rest).collect::<Vec<_>>();
                            choice.sort_unstable();
                            if choice != own && is_window_choice(sorted, &choice) {
                                choices.push(choice);
                                if choices.len() == max {
                                    return choices;
                                }
                                // One completion of these rows is enough.
                                break;
                            }
                        }
                    }
                }
                choices
            }
            NestedKind::DistinctOn { groups, .. } => groups
                .iter()
                .map(|group| match pools {
                    None => group.clone(),
                    // The rows of `required` first.
                    Some((required, optional)) => group
                        .iter()
                        .copied()
                        .filter(|row| required.contains(row))
                        .chain(group.iter().copied().filter(|row| {
                            !required.contains(row) && optional.contains(row)
                        }))
                        .collect(),
                })
                .multi_cartesian_product()
                .filter(|choice| *choice != own)
                .take(max)
                .collect(),
        }
    }
}

/// The positions `rows`, as an array.
fn positions_array(rows: &[usize]) -> Result<UInt32Array> {
    rows.iter()
        .map(|&row| u32::try_from(row))
        .collect::<std::result::Result<UInt32Array, _>>()
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

/// The maximum number of choices of the rows of the nested limits of a query
/// for which [`alternative_outcomes`] runs its plan, if the rows of its result
/// do not show them.
const MAX_CHOICE_RUNS: usize = 20_000;

/// Other outputs of DataFusion for `plan`, planned by `ctx`, with the order of
/// their rows, to be compared with `options`, for other rows that the nested
/// `ORDER BY … LIMIT`s of the query (below its top-level sort, see
/// [`nested_limits`]) may return, as they tie at their windows: the query
/// runs with chosen rows in place of each of them.
///
/// If the query only has projections, filters, sorts, subquery aliases and
/// inner joins above its nested limits, the rows of `result` (the output of
/// the other engine) show which rows they returned (see
/// [`lineage_choices`]): the query runs for these rows, if they may be
/// returned. Otherwise, one physical plan runs for up to [`MAX_CHOICE_RUNS`]
/// choices of rows of the nested limits, of one or several of them at once
/// (see [`slot_choices`]), and the query runs again for those whose rows are
/// the rows of `result`.
pub async fn alternative_outcomes(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    options: &ComparisonOptions,
    result: &EngineOutput,
) -> Vec<(EngineOutput, ResultOrder)> {
    let tolerance = options.float_tolerance;
    // Each occurrence, if the rows that it may return are known.
    let mut nested = vec![];
    for occurrence in nested_limits(plan) {
        nested.push(
            possible_outputs(ctx, occurrence, tolerance, MAX_CHOICE_RUNS)
                .await
                .ok(),
        );
    }
    if nested.iter().all(Option::is_none) {
        return vec![];
    }
    // The query, with the chosen rows in place of each occurrence.
    let run = async |chosen: &[Option<&Rows>]| {
        let mut replacements = vec![];
        for (entry, rows) in nested.iter().zip(chosen) {
            replacements.push(match (entry, rows) {
                (Some(possible), Some(rows)) => {
                    let batch = rows_batch(
                        (*rows).clone(),
                        &possible.limit.row_fields(),
                        possible.limit.plan.schema().metadata(),
                    )?;
                    let table = MemTable::try_new(batch.schema(), vec![vec![batch]])?;
                    Some(possible.limit.plan_over(Arc::new(table), None)?)
                }
                _ => None,
            });
        }
        let replaced = replace_nested_limits(plan, &mut |i, _| {
            replacements.get(i).cloned().flatten()
        })?;
        let output =
            run_dataframe(ctx, DataFrame::new(ctx.state(), replaced.clone())).await?;
        let order = result_order(ctx, &replaced, options).await;
        Ok::<_, DataFusionError>((output, order))
    };
    let mut alternatives = vec![];
    // The rows of the result show the rows of the nested limits (maybe in
    // several ways, if rows of other keys have equal values), unless rows of
    // nested limits inside them decide which rows they may return.
    let traced = nested
        .iter()
        .map(|entry| {
            entry
                .as_ref()
                .filter(|possible| possible.traced)
                .map(|possible| &possible.limit)
        })
        .collect::<Option<Vec<_>>>();
    if let Some(limits) = traced
        && let Some(shown) = lineage_choices(ctx, plan, &limits, result, tolerance).await
    {
        for choices in shown {
            let Ok(rows) = limits
                .iter()
                .zip(&choices)
                .map(|(limit, choice)| limit.columns(choice))
                .collect::<Result<Vec<_>>>()
            else {
                continue;
            };
            let chosen = rows.iter().map(Some).collect::<Vec<_>>();
            if let Ok(alternative) = run(&chosen).await {
                alternatives.push(alternative);
            }
        }
        return alternatives;
    }
    let shown = slot_choices(ctx, plan, &nested, result, tolerance)
        .await
        .unwrap_or_default();
    for combination in shown {
        let chosen = nested
            .iter()
            .zip(&combination)
            .map(|(entry, index)| {
                entry
                    .as_ref()
                    .zip(*index)
                    .map(|(possible, i)| &possible.outputs[i])
            })
            .collect::<Vec<_>>();
        if let Ok(alternative) = run(&chosen).await {
            alternatives.push(alternative);
        }
    }
    alternatives
}

/// Other outputs of DataFusion for `plan`, planned by `ctx`, with the order of
/// their rows, to be compared with `options`, for other signs of the `NaN`s of
/// the floating-point keys of its top-level sort: DataFusion sorts negative
/// `NaN`s first and positive ones last, but `NaN = NaN`, so another engine may
/// sort all of them first or last. The query runs with these keys as
/// positive, then as negative `NaN`s.
pub async fn nan_alternatives(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    options: &ComparisonOptions,
) -> Vec<(EngineOutput, ResultOrder)> {
    let mut alternatives = vec![];
    for negative in [false, true] {
        let Ok(Some(normalized)) = normalize_nans(plan, negative) else {
            continue;
        };
        let Ok(output) =
            run_dataframe(ctx, DataFrame::new(ctx.state(), normalized.clone())).await
        else {
            continue;
        };
        let order = result_order(ctx, &normalized, options).await;
        alternatives.push((output, order));
    }
    alternatives
}

/// `plan` with its top-level sort by `CASE WHEN isnan(key) THEN NaN ELSE key
/// END` for each of its floating-point keys `key`, a negative `NaN` if
/// `negative`: `None` if it has none.
fn normalize_nans(plan: &LogicalPlan, negative: bool) -> Result<Option<LogicalPlan>> {
    match plan {
        LogicalPlan::Projection(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Limit(_) => {
            let Some(input) = normalize_nans(plan.inputs()[0], negative)? else {
                return Ok(None);
            };
            Ok(Some(plan.with_new_exprs(plan.expressions(), vec![input])?))
        }
        LogicalPlan::Sort(sort) => {
            let schema = sort.input.schema();
            let mut floats = false;
            let mut expr = vec![];
            for sort_expr in &sort.expr {
                let data_type = sort_expr.expr.get_type(schema)?;
                if !matches!(
                    data_type,
                    DataType::Float16 | DataType::Float32 | DataType::Float64
                ) {
                    expr.push(sort_expr.clone());
                    continue;
                }
                floats = true;
                let nan = if negative { -f64::NAN } else { f64::NAN };
                let nan = Expr::Literal(ScalarValue::Float64(Some(nan)), None)
                    .cast_to(&data_type, schema)?;
                let key = datafusion_expr::when(isnan(sort_expr.expr.clone()), nan)
                    .otherwise(sort_expr.expr.clone())?;
                expr.push(SortExpr {
                    expr: key,
                    ..sort_expr.clone()
                });
            }
            Ok(floats.then(|| {
                LogicalPlan::Sort(Sort {
                    expr,
                    input: Arc::clone(&sort.input),
                    fetch: sort.fetch,
                })
            }))
        }
        _ => Ok(None),
    }
}

/// The rows (the columns of [`NestedLimit::columns`]) that a nested limit or
/// a `DISTINCT ON` returns.
type Rows = Vec<ArrayRef>;

/// The rows that an occurrence of a nested limit or of a `DISTINCT ON` may
/// return, see [`possible_outputs`].
struct PossibleOutputs {
    /// The node, for the rows of the nested limits inside it that DataFusion
    /// returns.
    limit: NestedLimit,
    /// The rows that it may return, those of DataFusion first.
    outputs: Vec<Rows>,
    /// Whether these are the rows of `limit` (no nested limits inside it
    /// decide which rows it may return), so that the lineage of the rows of a
    /// query shows which rows it returned (see [`lineage_choices`]).
    traced: bool,
}

/// Up to `max` choices of the rows that the nested limit or the `DISTINCT ON`
/// `plan` of a query, planned by `ctx`, may return within the tolerance
/// `tolerance`: for each choice of the rows of the nested limits inside it
/// (up to `max` of them, recursively), the rows that it may return.
fn possible_outputs(
    ctx: &SessionContext,
    plan: LogicalPlan,
    tolerance: f64,
    max: usize,
) -> std::pin::Pin<Box<dyn Future<Output = Result<PossibleOutputs>> + Send + '_>> {
    Box::pin(async move {
        let inner = inner_limits(&plan);
        let limit = NestedLimit::new(ctx, plan.clone(), tolerance).await?;
        // The rows that it may return for the rows of the nested limits inside it
        // that DataFusion returns, those of DataFusion first.
        let mut outputs = vec![limit.columns(&limit.own())?];
        for choice in limit.choices(None, max) {
            outputs.push(limit.columns(&choice)?);
        }
        if inner.is_empty() {
            return Ok(PossibleOutputs {
                limit,
                outputs,
                traced: true,
            });
        }
        let mut inners = vec![];
        for occurrence in inner {
            inners.push(possible_outputs(ctx, occurrence, tolerance, max).await?);
        }
        let combinations = inners
            .iter()
            .map(|inner| 0..inner.outputs.len())
            .multi_cartesian_product()
            // The rows of DataFusion, which come first, are those of `limit`.
            .skip(1)
            .take(max);
        for combination in combinations {
            let mut replacements = vec![];
            for (inner, &i) in inners.iter().zip(&combination) {
                let batch = rows_batch(
                    inner.outputs[i].clone(),
                    &inner.limit.row_fields(),
                    inner.limit.plan.schema().metadata(),
                )?;
                let table = MemTable::try_new(batch.schema(), vec![vec![batch]])?;
                replacements.push(inner.limit.plan_over(Arc::new(table), None)?);
            }
            let replaced =
                replace_inner_limits(&plan, &mut |i, _| replacements.get(i).cloned())?;
            let Ok(variant) = NestedLimit::new(ctx, replaced, tolerance).await else {
                continue;
            };
            let own = variant.own();
            for choice in std::iter::once(own).chain(variant.choices(None, max)) {
                outputs.push(variant.columns(&choice)?);
                if outputs.len() >= max {
                    break;
                }
            }
            if outputs.len() >= max {
                break;
            }
        }
        Ok(PossibleOutputs {
            limit,
            outputs,
            traced: false,
        })
    })
}

/// Up to 16 choices of the rows of the occurrences `nested` of nested limits of
/// `plan` (each an index in their [`PossibleOutputs`], none for occurrences
/// whose rows are not known) for which the rows of the query are the rows of
/// `result` (the output of the other engine), as a multiset: of one or
/// several of them at once, up to [`MAX_CHOICE_RUNS`] choices in all. One
/// physical plan, whose nested limits read their rows from [`Slot`]s, runs
/// for each choice.
async fn slot_choices(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    nested: &[Option<PossibleOutputs>],
    result: &EngineOutput,
    tolerance: f64,
) -> Result<Vec<Vec<Option<usize>>>> {
    let result = result_rows(&result.batches)?;
    let mut slots = vec![];
    let mut tables = vec![];
    for entry in nested {
        let Some(possible) = entry else {
            slots.push(None);
            tables.push(None);
            continue;
        };
        let fields = possible.limit.row_fields();
        let empty = rows_batch(
            possible.limit.columns(&[])?,
            &fields,
            possible.limit.plan.schema().metadata(),
        )?;
        let slot = Arc::new(Slot {
            schema: empty.schema(),
            batch: Mutex::new(empty.clone()),
        });
        let partition = Arc::clone(&slot) as Arc<dyn PartitionStream>;
        let table = StreamingTable::try_new(empty.schema(), vec![partition])?;
        tables.push(Some(possible.limit.plan_over(Arc::new(table), None)?));
        slots.push(Some((slot, fields)));
    }
    let replaced =
        replace_nested_limits(plan, &mut |i, _| tables.get(i).cloned().flatten())?;
    let state = ctx.state();
    let physical = state.create_physical_plan(&replaced).await?;
    let indices = nested
        .iter()
        .map(|entry| match entry {
            Some(possible) => (0..possible.outputs.len()).map(Some).collect(),
            None => vec![None],
        })
        .collect::<Vec<Vec<Option<usize>>>>();
    let mut shown = vec![];
    // Not the rows of DataFusion, which are the first ones.
    for combination in indices
        .into_iter()
        .multi_cartesian_product()
        .skip(1)
        .take(MAX_CHOICE_RUNS)
    {
        for ((slot, entry), index) in slots.iter().zip(nested).zip(&combination) {
            if let (Some((slot, fields)), Some(possible), Some(i)) = (slot, entry, index)
            {
                *slot.batch.lock().expect("the slot is not poisoned") = rows_batch(
                    possible.outputs[*i].clone(),
                    fields,
                    possible.limit.plan.schema().metadata(),
                )?;
            }
        }
        let Ok(batches) =
            collect(reset_plan_states(Arc::clone(&physical))?, state.task_ctx()).await
        else {
            continue;
        };
        if multiset_difference(&result_rows(&batches)?, &result, tolerance).is_empty() {
            shown.push(combination);
            if shown.len() == 16 {
                break;
            }
        }
    }
    Ok(shown)
}

/// The rows that the nested limits `nested` of `plan` may have returned for
/// the other engine, shown by the rows of its `result`: up to 16 choices of
/// rows, each the positions of the sorted rows of each nested limit. `None`
/// if `plan` has other nodes than projections, filters, sorts (without a
/// fetch), subquery aliases and inner joins above its nested limits (e.g.
/// aggregates, limits or outer joins), so that its rows do not show them.
/// No choices if its rows show none that the nested limits may return.
///
/// The query runs with the rows that each nested limit may return in its
/// place (see [`window_choices`]), with their positions as their lineage,
/// which the nodes above keep: the rows of the query for some of these rows
/// are the rows whose lineage is some of these rows. The rows of `result`
/// show which rows each nested limit returned: those of the lineage of the
/// rows of the query that are rows of `result`.
async fn lineage_choices(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    nested: &[&NestedLimit],
    result: &EngineOutput,
    tolerance: f64,
) -> Option<Vec<Vec<Vec<usize>>>> {
    let candidates = nested
        .iter()
        .map(|limit| limit.candidates())
        .collect::<Vec<_>>();
    let mut replacements = vec![];
    for (i, (limit, candidates)) in nested.iter().zip(&candidates).enumerate() {
        replacements.push(limit.rows(candidates, Some(&lineage_column(i))).ok()?);
    }
    let replaced =
        replace_nested_limits(plan, &mut |i, _| replacements.get(i).cloned()).ok()?;
    let lineage = lineage_plan(&replaced).ok()??;
    // The lineage of every nested limit must reach the rows of the query (not
    // if a nested limit is in a subquery).
    let names = lineage
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<HashSet<_>>();
    if (0..nested.len()).any(|i| !names.contains(&lineage_column(i))) {
        return None;
    }
    let batches = ctx
        .execute_logical_plan(lineage)
        .await
        .ok()?
        .collect()
        .await
        .ok()?;
    // The values and the lineage of the rows of the query.
    let schema = batches.first()?.schema();
    let (mut values, mut lineages) = (vec![], vec![]);
    for (i, field) in schema.fields().iter().enumerate() {
        match (0..nested.len()).find(|&limit| field.name() == &lineage_column(limit)) {
            Some(limit) => lineages.push((i, limit)),
            None => values.push(i),
        }
    }
    let mut rows = vec![];
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let row_values = values
                .iter()
                .map(|&column| Value::try_new(batch.column(column).as_ref(), row))
                .collect::<Result<Row>>()
                .ok()?;
            let row_lineage = lineages
                .iter()
                .map(|&(column, limit)| {
                    let positions = batch.column(column).as_primitive::<UInt32Type>();
                    (limit, positions.value(row).as_usize())
                })
                .collect::<Vec<_>>();
            rows.push((row_values, row_lineage));
        }
    }
    let result = result_rows(&result.batches).ok()?;
    if rows.len().saturating_mul(result.len()) > 10_000_000 {
        return None;
    }
    // The rows of each nested limit that rows of the result show, and those
    // that no row of the query shows.
    let mut shown = vec![BTreeSet::new(); nested.len()];
    for (row_values, row_lineage) in &rows {
        if result
            .iter()
            .any(|row| rows_equal(row, row_values, tolerance))
        {
            for &(limit, position) in row_lineage {
                shown[limit].insert(position);
            }
        }
    }
    let mut silent = candidates.clone();
    for (_, row_lineage) in &rows {
        for &(limit, position) in row_lineage {
            silent[limit].retain(|&other| other != position);
        }
    }
    // The rows that each nested limit may return, among those.
    let choices = nested
        .iter()
        .zip(shown.iter().zip(&silent))
        .map(|(limit, (shown, silent))| {
            let silent = silent.iter().copied().collect::<BTreeSet<_>>();
            std::iter::once(limit.own())
                .filter(|own| {
                    own.iter()
                        .all(|row| shown.contains(row) || silent.contains(row))
                })
                .chain(limit.choices(Some((shown, &silent)), 4096))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut shown_choices = vec![];
    for choices in choices.into_iter().multi_cartesian_product().take(4096) {
        let selected = rows
            .iter()
            .filter(|(_, row_lineage)| {
                row_lineage.iter().all(|(limit, position)| {
                    choices[*limit].binary_search(position).is_ok()
                })
            })
            .map(|(row_values, _)| row_values.clone())
            .collect::<Vec<_>>();
        if multiset_difference(&selected, &result, tolerance).is_empty() {
            shown_choices.push(choices);
            if shown_choices.len() == 16 {
                break;
            }
        }
    }
    Some(shown_choices)
}

/// Whether the rows of `plan` have lineage columns (see [`lineage_column`]).
fn has_lineage(plan: &LogicalPlan) -> bool {
    plan.schema()
        .fields()
        .iter()
        .any(|field| field.name().starts_with("__test_utils_lineage_"))
}

/// `plan`, in which nested limits are replaced by their rows with their
/// lineage (see [`lineage_column`]), with the lineage columns below each node
/// after its columns. `None` if a node above a nested limit does not keep the
/// lineage of its rows: only projections, filters, sorts without a fetch,
/// subquery aliases and inner joins do.
fn lineage_plan(plan: &LogicalPlan) -> Result<Option<LogicalPlan>> {
    let mut inputs = vec![];
    for input in plan.inputs() {
        let Some(input) = lineage_plan(input)? else {
            return Ok(None);
        };
        inputs.push(input);
    }
    if !inputs.iter().any(has_lineage) {
        return Ok(Some(plan.clone()));
    }
    Ok(Some(match plan {
        LogicalPlan::Projection(projection) => {
            let input = inputs.remove(0);
            let mut exprs = projection.expr.clone();
            exprs.extend(
                input
                    .schema()
                    .iter()
                    .filter(|(_, field)| {
                        field.name().starts_with("__test_utils_lineage_")
                    })
                    .map(|(qualifier, field)| {
                        Expr::Column(Column::new(qualifier.cloned(), field.name()))
                    }),
            );
            LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(input))?)
        }
        LogicalPlan::Sort(sort) if sort.fetch.is_some() => return Ok(None),
        LogicalPlan::Join(join) if join.join_type != JoinType::Inner => return Ok(None),
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Join(_) => plan.with_new_exprs(plan.expressions(), inputs)?,
        _ => return Ok(None),
    }))
}

/// Whether `node` is a limit over a sort (through projections, subquery
/// aliases and limits), or a sort with a fetch.
fn limits_a_sort(node: &LogicalPlan) -> bool {
    match node {
        LogicalPlan::Sort(sort) => sort.fetch.is_some(),
        LogicalPlan::Limit(limit) => {
            let mut input = limit.input.as_ref();
            loop {
                match input {
                    LogicalPlan::Sort(_) => break true,
                    LogicalPlan::Projection(_)
                    | LogicalPlan::SubqueryAlias(_)
                    | LogicalPlan::Limit(_) => input = input.inputs()[0],
                    _ => break false,
                }
            }
        }
        _ => false,
    }
}

/// `plan` with the nested `ORDER BY … LIMIT`s that `replace` replaces (given
/// their index, see [`nested_limits`], and their plan).
fn replace_nested_limits(
    plan: &LogicalPlan,
    replace: &mut dyn FnMut(usize, &LogicalPlan) -> Option<LogicalPlan>,
) -> Result<LogicalPlan> {
    replace_ambiguous(plan, true, replace)
}

/// The nested limits and `DISTINCT ON`s inside the occurrence `plan` of a
/// nested limit or of a `DISTINCT ON` (see [`nested_limits`]).
fn inner_limits(plan: &LogicalPlan) -> Vec<LogicalPlan> {
    let mut limits = vec![];
    let _ = replace_inner_limits(plan, &mut |_, node| {
        limits.push(node.clone());
        None
    });
    limits
}

/// The occurrence `plan` of a nested limit or of a `DISTINCT ON`, with the
/// nested limits and `DISTINCT ON`s inside it that `replace` replaces (given
/// their index, see [`inner_limits`], and their plan).
fn replace_inner_limits(
    plan: &LogicalPlan,
    replace: &mut dyn FnMut(usize, &LogicalPlan) -> Option<LogicalPlan>,
) -> Result<LogicalPlan> {
    match plan {
        // All of those of its input.
        LogicalPlan::Distinct(Distinct::On(distinct)) => {
            let input = replace_ambiguous(&distinct.input, false, replace)?;
            Ok(LogicalPlan::Distinct(Distinct::On(DistinctOn::try_new(
                distinct.on_expr.clone(),
                distinct.select_expr.clone(),
                distinct.sort_expr.clone(),
                Arc::new(input),
            )?)))
        }
        _ => replace_ambiguous(plan, true, replace),
    }
}

/// `plan` with the nested limits and `DISTINCT ON`s that `replace` replaces,
/// but for the limits of its top-level path if `top_level` (see
/// [`replace_nested_limits`]).
fn replace_ambiguous(
    plan: &LogicalPlan,
    top_level_path: bool,
    replace: &mut dyn FnMut(usize, &LogicalPlan) -> Option<LogicalPlan>,
) -> Result<LogicalPlan> {
    // The limits above the top-level sort of the query, and the sort, which
    // the sorted reference of the query compares, come first.
    let mut top_level = 0;
    if top_level_path && has_top_level_order_by(plan) {
        let mut node = plan;
        loop {
            top_level += usize::from(limits_a_sort(node));
            match node {
                LogicalPlan::Projection(_)
                | LogicalPlan::SubqueryAlias(_)
                | LogicalPlan::Limit(_) => node = node.inputs()[0],
                _ => break,
            }
        }
    }
    let mut index = 0;
    Ok(plan
        .clone()
        .transform_down_with_subqueries(|node| {
            let distinct_on = matches!(node, LogicalPlan::Distinct(Distinct::On(_)));
            if !limits_a_sort(&node) && !distinct_on {
                return Ok(Transformed::no(node));
            }
            if top_level > 0 && !distinct_on {
                top_level -= 1;
                return Ok(Transformed::no(node));
            }
            index += 1;
            Ok(match replace(index - 1, &node) {
                Some(rows) => Transformed::new(rows, true, TreeNodeRecursion::Jump),
                None => Transformed::new(node, false, TreeNodeRecursion::Jump),
            })
        })?
        .data)
}

/// The nested `ORDER BY … LIMIT`s of `plan`, each occurrence of them: its
/// limits over sorts (through projections, subquery aliases and limits) and
/// its sorts with a fetch, below its top-level sort (see
/// [`has_top_level_order_by`]), also in subqueries, but not inside each
/// other.
fn nested_limits(plan: &LogicalPlan) -> Vec<LogicalPlan> {
    let mut limits = vec![];
    let _ = replace_nested_limits(plan, &mut |_, node| {
        limits.push(node.clone());
        None
    });
    limits
}

/// The rows of `columns` (of the fields `fields`), as a batch of columns
/// named `c0`, `c1`, …
fn rows_batch(
    columns: Vec<ArrayRef>,
    fields: &[(Option<TableReference>, FieldRef)],
    metadata: &arrow::datatypes::Metadata,
) -> Result<RecordBatch> {
    let schema = fields
        .iter()
        .enumerate()
        .map(|(i, (_, field))| field.as_ref().clone().with_name(format!("c{i}")))
        .collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(schema, metadata.clone())),
        columns,
    )?)
}

/// The rows of `table` (see [`rows_batch`]), as a plan of the fields
/// `fields`, which may have qualifiers.
fn rows_plan(
    table: Arc<dyn TableProvider>,
    fields: &[(Option<TableReference>, FieldRef)],
) -> Result<LogicalPlan> {
    let scan = LogicalPlanBuilder::scan(SORTED_ROWS, provider_as_source(table), None)?
        .build()?;
    let exprs = fields
        .iter()
        .enumerate()
        .map(|(i, (qualifier, field))| {
            Expr::Column(Column::new(Some(SORTED_ROWS), format!("c{i}")))
                .alias_qualified(qualifier.clone(), field.name())
        })
        .collect();
    Ok(LogicalPlan::Projection(Projection::try_new(
        exprs,
        Arc::new(scan),
    )?))
}

/// The sorted rows (`sorted`) that a query may return (positions), and of
/// those, the rows of its window that it returns for any choice of rows: the
/// rows whose output columns are computed, and whose classes may be at the
/// positions of its window, see [`ResultOrder::Sorted`].
fn candidates(sorted: &SortedRows) -> (Vec<usize>, Vec<usize>) {
    let window = sorted.window.clone();
    let class_of = class_of(sorted);
    let bounds = bounds(&sorted.classes, &sorted.tied);
    let candidates = (0..sorted.values.len())
        .filter(|&row| {
            let (first, last) = bounds[class_of[row]];
            sorted.values[row].is_some() && first < window.end && last >= window.start
        })
        .collect::<Vec<_>>();
    let fixed = candidates
        .iter()
        .copied()
        .filter(|&row| {
            window.contains(&row)
                && candidates.iter().all(|&other| {
                    other == row
                        || window.contains(&other)
                        || !ties(sorted, &class_of, row, other)
                })
        })
        .collect();
    (candidates, fixed)
}

/// The class of each sorted row of `sorted`.
fn class_of(sorted: &SortedRows) -> Vec<usize> {
    let mut class_of = vec![0; sorted.values.len()];
    for (class, (rows, _)) in sorted.classes.iter().enumerate() {
        class_of[rows.clone()].fill(class);
    }
    class_of
}

/// Whether the sorted rows `a` and `b` of `sorted` tie.
fn ties(sorted: &SortedRows, class_of: &[usize], a: usize, b: usize) -> bool {
    class_of[a] == class_of[b]
        || sorted.tied[class_of[a]].binary_search(&class_of[b]).is_ok()
}

/// Whether a query may return the sorted rows `choice` (positions, in their
/// order) of `sorted`: whether an order of all sorted rows in which rows are
/// only out of their sorted order relative to rows that they tie with has
/// them at the positions of the window. The rows that the `OFFSET` skips are
/// the first rows of the other positions.
fn is_window_choice(sorted: &SortedRows, choice: &[usize]) -> bool {
    let window = sorted.window.clone();
    if choice.len() != window.len()
        || choice
            .iter()
            .any(|&row| sorted.values.get(row).is_none_or(Option::is_none))
    {
        return false;
    }
    let class_of = class_of(sorted);
    // The skipped rows, before the chosen rows, and the cut rows, after them,
    // must tie with the chosen rows that they are out of order with.
    let skipped = (0..sorted.values.len())
        .filter(|row| choice.binary_search(row).is_err())
        .take(window.start)
        .collect::<Vec<_>>();
    choice.iter().all(|&row| {
        skipped
            .iter()
            .all(|&other| other < row || ties(sorted, &class_of, row, other))
            && (0..row).all(|other| {
                choice.binary_search(&other).is_ok()
                    || skipped.binary_search(&other).is_ok()
                    || ties(sorted, &class_of, row, other)
            })
    })
}

/// Up to `max` other choices of the sorted rows `sorted` that a query may
/// return than the rows of its window (each the positions of the rows, in
/// their order), see [`is_window_choice`].
fn window_choices(sorted: &SortedRows, max: usize) -> Vec<Vec<usize>> {
    let window = sorted.window.clone();
    let (candidates, fixed) = candidates(sorted);
    let free = candidates
        .iter()
        .copied()
        .filter(|row| fixed.binary_search(row).is_err())
        .collect::<Vec<_>>();
    let mut choices = vec![];
    for chosen in free
        .iter()
        .copied()
        .combinations(window.len().saturating_sub(fixed.len()))
        .take(100_000)
    {
        let mut choice = fixed.iter().copied().chain(chosen).collect::<Vec<_>>();
        choice.sort_unstable();
        if !choice.iter().copied().eq(window.clone()) && is_window_choice(sorted, &choice)
        {
            choices.push(choice);
            if choices.len() == max {
                break;
            }
        }
    }
    choices
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
    /// output columns of the query (null in the rows of
    /// [`SortedReference::uncomputed`]), followed by the values of
    /// [`SortedReference::keys`].
    pub batches: Vec<RecordBatch>,
    /// The number of output columns of `batches`.
    pub columns: usize,
    /// The `ORDER BY` keys whose values follow the output columns
    /// (displayed): all of them, up to the first key whose values depend on
    /// a volatile function (e.g. `random()`, also through the columns of a
    /// subquery), as the rows that tie on the keys before it may come in any
    /// order.
    pub keys: Vec<String>,
    /// The sort options of [`SortedReference::keys`] (ascending with nulls last
    /// if missing).
    pub sort_options: Vec<SortOptions>,
    /// The number of sorted rows that the query skips (`OFFSET`).
    pub skip: usize,
    /// The maximum number of sorted rows that the query returns (`LIMIT`).
    pub fetch: Option<usize>,
    /// The ranges of sorted rows whose output columns are not computed, with
    /// the reason: the query fails on them (e.g. on a division by zero above
    /// its `LIMIT`), or on the rows that it cannot return (whose
    /// keys cannot tie with the keys of the rows that it returns within
    /// [`SortedReference::tolerance`]: their output columns are only computed
    /// to describe differences). Results that have one of these rows differ
    /// from the reference.
    pub uncomputed: Vec<(Range<usize>, String)>,
    /// The floating-point tolerance within which the reference computes the
    /// output columns of the sorted rows that the query may return (see
    /// [`ComparisonOptions::float_tolerance`]). Results are compared with it
    /// within this tolerance at most.
    pub tolerance: f64,
}

/// How the rows of the result of `plan`, planned by `ctx`, are ordered, to be
/// compared with `options`. For a query with a top-level `ORDER BY`, `ctx`
/// sorts all rows of its top-level sort (without its `LIMIT` and `OFFSET`),
/// with the values of its `ORDER BY` keys, and computes the output columns of
/// the sorted rows that the query may return within the tolerance of
/// `options`, to compute its [`SortedReference`].
pub async fn result_order(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    options: &ComparisonOptions,
) -> ResultOrder {
    if !has_top_level_order_by(plan) {
        return ResultOrder::Unordered;
    }
    match sorted_reference(ctx, plan, options.float_tolerance).await {
        Ok(reference) => ResultOrder::Sorted(reference),
        Err(e) => ResultOrder::Exact(e.to_string()),
    }
}

async fn sorted_reference(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    tolerance: f64,
) -> Result<SortedReference> {
    let reference = reference_plan(plan, &ctx.state())?;
    let sorted = ctx
        .execute_logical_plan(reference.sort.clone())
        .await?
        .collect()
        .await?;
    let schema = sorted.first().map_or_else(
        || Arc::new(reference.sort.schema().as_arrow().clone()),
        RecordBatch::schema,
    );
    let sorted = concat_batches(&schema, &sorted)?;
    let width = reference.input.fields().len();
    if sorted.num_columns() != width + reference.keys.len() {
        return internal_err!("the sorted rows do not have the columns of the sort");
    }
    let rows = sorted.num_rows();
    let window = reference.window.rows(rows);
    let returnable = returnable_rows(
        &sorted,
        width,
        window.clone(),
        &reference.sort_options,
        tolerance,
    )?;
    let (mut outputs, mut failures) =
        reference.outputs(ctx, &sorted, &returnable).await?;
    // The other rows, only to describe differences (e.g. a row out of order):
    // the query cannot return them, so it may fail on them.
    let others = {
        let mut returnable = returnable.iter().peekable();
        (0..rows)
            .filter(|row| returnable.next_if_eq(&row).is_none())
            .collect::<Vec<_>>()
    };
    if !others.is_empty() {
        match reference.compute(ctx, &sorted, &others).await {
            Ok(batches) => outputs.extend(batches),
            Err(_) => failures.extend(others.iter().map(|&row| {
                let reason = "the query cannot return it, as its keys cannot tie \
                              with the keys of the rows that it returns within the \
                              tolerance";
                (row, reason.to_string())
            })),
        }
    }

    // The output columns of the sorted rows, at their positions.
    let output = plan.schema().fields();
    let mut index = vec![None; rows];
    let outputs = match outputs.first() {
        Some(first) => Some(concat_batches(&first.schema(), &outputs)?),
        None => None,
    };
    if let Some(outputs) = &outputs {
        let fields = outputs.schema().fields().clone();
        if fields.len() != output.len() + 1
            || output
                .iter()
                .zip(&fields)
                .any(|(output, field)| output.data_type() != field.data_type())
        {
            return internal_err!(
                "the sorted rows do not have the output columns of the query"
            );
        }
        let positions = outputs.column(output.len()).as_primitive::<UInt32Type>();
        for (i, &position) in positions.values().iter().enumerate() {
            let Some(slot @ None) = index.get_mut(position.as_usize()) else {
                return internal_err!(
                    "the output columns of a sorted row are computed twice"
                );
            };
            *slot = Some(u32::try_from(i).expect("rows fit in u32"));
        }
    }
    let failed = failures.iter().map(|(row, _)| *row).collect::<HashSet<_>>();
    if (0..rows).any(|row| index[row].is_none() && !failed.contains(&row)) {
        return internal_err!("the output columns of a sorted row are not computed");
    }
    let index = UInt32Array::from(index);
    let mut fields = vec![];
    let mut columns = vec![];
    for (i, field) in output.iter().enumerate() {
        columns.push(match &outputs {
            Some(outputs) => take(outputs.column(i), &index, None)?,
            None => new_null_array(field.data_type(), rows),
        });
        fields.push(Field::new(field.name(), field.data_type().clone(), true));
    }
    for i in width..sorted.num_columns() {
        fields.push(sorted.schema().field(i).clone());
        columns.push(Arc::clone(sorted.column(i)));
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;

    // The rows whose output columns are not computed, with the reason.
    let mut failures = failures.into_iter().collect::<HashMap<_, _>>();
    let mut uncomputed: Vec<(Range<usize>, String)> = vec![];
    for row in 0..rows {
        let Some(reason) = failures.remove(&row) else {
            continue;
        };
        match uncomputed.last_mut() {
            Some((range, last)) if range.end == row && *last == reason => range.end += 1,
            _ => uncomputed.push((row..row + 1, reason)),
        }
    }
    Ok(SortedReference {
        batches: vec![batch],
        columns: output.len(),
        keys: reference.keys,
        sort_options: reference.sort_options,
        skip: reference.window.skip,
        fetch: reference.window.fetch,
        uncomputed,
        tolerance,
    })
}

/// The sorted rows (`sorted`, whose columns from `width` are the values of the
/// `ORDER BY` keys) that a query may return if it returns the sorted rows
/// `window`, within the tolerance `tolerance`: the rows of the classes that
/// may be at the positions of `window`, as they tie with other classes (see
/// [`bounds`]).
fn returnable_rows(
    sorted: &RecordBatch,
    width: usize,
    window: Range<usize>,
    options: &[SortOptions],
    tolerance: f64,
) -> Result<Vec<usize>> {
    let keys = (0..sorted.num_rows())
        .map(|row| {
            sorted.columns()[width..]
                .iter()
                .map(|column| Value::try_new(column.as_ref(), row))
                .collect::<Result<Row>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let (classes, tied) = classes_and_ties(&keys, options, tolerance);
    let classes = classes
        .into_iter()
        .map(|class| (class.rows, class.keys))
        .collect::<Vec<_>>();
    Ok(classes
        .iter()
        .zip(bounds(&classes, &tied))
        .filter(|(_, (first, last))| *first < window.end && *last >= window.start)
        .flat_map(|((rows, _), _)| rows.clone())
        .collect())
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

    /// The positions of the rows of this window among `rows` sorted rows.
    fn rows(self, rows: usize) -> Range<usize> {
        let start = self.skip.min(rows);
        start
            ..self
                .fetch
                .map_or(rows, |fetch| start.saturating_add(fetch).min(rows))
    }
}

/// The name of the column of the values of the `i`th `ORDER BY` key in the
/// plan of a [`SortedReference`].
fn key_column(i: usize) -> String {
    format!("__test_utils_order_by_key_{i}")
}

/// The name of the table of the sorted rows whose output columns
/// [`ReferencePlan::compute`] computes.
const SORTED_ROWS: &str = "__test_utils_sorted_rows";

/// The name of the column of the positions of the sorted rows whose output
/// columns [`ReferencePlan::compute`] computes.
const SORTED_POSITION: &str = "__test_utils_sorted_position";

/// The plans of the [`SortedReference`] of a query with a top-level `ORDER BY`
/// (see [`has_top_level_order_by`]), see [`reference_plan`].
struct ReferencePlan {
    /// The top-level sort of the query without its `fetch`, over its input
    /// with the values of the `ORDER BY` keys after its columns (see
    /// [`key_column`]).
    sort: LogicalPlan,
    /// The schema of the input of the sort.
    input: DFSchemaRef,
    /// The `ORDER BY` keys whose values the sort has, see
    /// [`SortedReference::keys`].
    keys: Vec<String>,
    /// Their sort options.
    sort_options: Vec<SortOptions>,
    /// The projections and subquery aliases between the sort and the root of
    /// the query, from the sort up.
    above: Vec<LogicalPlan>,
    /// The window of the sorted rows that the query returns.
    window: Window,
}

/// A partition of one batch, which can be replaced from one run of a plan to
/// the next, see [`ReferencePlan::compute_each`].
#[derive(Debug)]
struct Slot {
    schema: SchemaRef,
    batch: Mutex<RecordBatch>,
}

impl PartitionStream for Slot {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _: Arc<TaskContext>) -> SendableRecordBatchStream {
        let batch = self.batch.lock().expect("the slot is not poisoned").clone();
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            futures::stream::iter([Ok(batch)]),
        ))
    }
}

impl ReferencePlan {
    /// The output columns of the sorted rows `rows` of `sorted` (the rows of
    /// [`ReferencePlan::sort`]), each followed by its position, and the rows
    /// on which the query fails, with the error. They are computed for all
    /// rows at once, or, if the query fails, for each row alone (see
    /// [`ReferencePlan::compute_each`]).
    async fn outputs(
        &self,
        ctx: &SessionContext,
        sorted: &RecordBatch,
        rows: &[usize],
    ) -> Result<(Vec<RecordBatch>, Vec<(usize, String)>)> {
        if rows.is_empty() {
            return Ok((vec![], vec![]));
        }
        if let Ok(batches) = self.compute(ctx, sorted, rows).await {
            return Ok((batches, vec![]));
        }
        // Without rows, the query must not fail: its errors come from its rows.
        self.compute(ctx, sorted, &[]).await?;
        self.compute_each(ctx, sorted, rows).await
    }

    /// The output columns of the sorted rows `rows` of `sorted`, each followed
    /// by its position ([`SORTED_POSITION`]): the projections and subquery
    /// aliases above the sort, over these rows.
    async fn compute(
        &self,
        ctx: &SessionContext,
        sorted: &RecordBatch,
        rows: &[usize],
    ) -> Result<Vec<RecordBatch>> {
        let batch = self.rows(sorted, rows)?;
        let table = MemTable::try_new(batch.schema(), vec![vec![batch]])?;
        let plan = self.over(Arc::new(table))?;
        ctx.execute_logical_plan(plan).await?.collect().await
    }

    /// The output columns of each of the sorted rows `rows` of `sorted` alone
    /// (each followed by its position), or the error of the query on it: one
    /// physical plan, over a table whose rows are one row of a [`Slot`], runs
    /// for each row.
    async fn compute_each(
        &self,
        ctx: &SessionContext,
        sorted: &RecordBatch,
        rows: &[usize],
    ) -> Result<(Vec<RecordBatch>, Vec<(usize, String)>)> {
        let batch = self.rows(sorted, rows)?;
        let slot = Arc::new(Slot {
            schema: batch.schema(),
            batch: Mutex::new(batch.slice(0, 0)),
        });
        let partition = Arc::clone(&slot) as Arc<dyn PartitionStream>;
        let table = StreamingTable::try_new(batch.schema(), vec![partition])?;
        let state = ctx.state();
        let plan = state
            .create_physical_plan(&self.over(Arc::new(table))?)
            .await?;
        let (mut outputs, mut failures) = (vec![], vec![]);
        for (i, &row) in rows.iter().enumerate() {
            *slot.batch.lock().expect("the slot is not poisoned") = batch.slice(i, 1);
            // Joins keep the rows of their build side from one run to the next.
            let batches = match reset_plan_states(Arc::clone(&plan)) {
                Ok(plan) => collect(plan, state.task_ctx()).await,
                Err(e) => Err(e),
            };
            match batches {
                Ok(batches) => outputs.extend(batches),
                Err(e) => failures.push((row, format!("the query fails on it: {e}"))),
            }
        }
        Ok((outputs, failures))
    }

    /// The input columns of the sorted rows `rows` of `sorted` (named `c0`,
    /// `c1`, …), followed by their positions ([`SORTED_POSITION`]).
    fn rows(&self, sorted: &RecordBatch, rows: &[usize]) -> Result<RecordBatch> {
        let positions = rows
            .iter()
            .map(|&row| u32::try_from(row))
            .collect::<std::result::Result<UInt32Array, _>>()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let schema = sorted.schema();
        let width = self.input.fields().len();
        let mut fields = vec![];
        let mut columns = vec![];
        for i in 0..width {
            fields.push(schema.field(i).clone().with_name(format!("c{i}")));
            columns.push(take(sorted.column(i), &positions, None)?);
        }
        fields.push(Field::new(SORTED_POSITION, DataType::UInt32, false));
        columns.push(Arc::new(positions) as ArrayRef);
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }

    /// The projections and subquery aliases above the sort, over the rows of
    /// `table` (see [`ReferencePlan::rows`]).
    fn over(&self, table: Arc<dyn TableProvider>) -> Result<LogicalPlan> {
        let scan =
            LogicalPlanBuilder::scan(SORTED_ROWS, provider_as_source(table), None)?
                .build()?;
        // The columns of the input of the sort, with their qualifiers.
        let column = |name: &str| Expr::Column(Column::new(Some(SORTED_ROWS), name));
        let exprs = self
            .input
            .iter()
            .enumerate()
            .map(|(i, (qualifier, field))| {
                column(&format!("c{i}")).alias_qualified(qualifier.cloned(), field.name())
            })
            .chain([column(SORTED_POSITION).alias(SORTED_POSITION)])
            .collect();
        let rows = LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(scan))?);
        self.above(rows)
    }

    /// The projections and subquery aliases above the sort over `rows`, which
    /// has the columns of the input of the sort and [`SORTED_POSITION`], with
    /// this column after their columns.
    fn above(&self, rows: LogicalPlan) -> Result<LogicalPlan> {
        self.above.iter().try_fold(rows, |input, node| {
            Ok(match node {
                LogicalPlan::Projection(projection) => {
                    let position = {
                        let (qualifier, field) = input
                            .schema()
                            .qualified_field_with_unqualified_name(SORTED_POSITION)?;
                        Expr::Column(Column::new(qualifier.cloned(), field.name()))
                    };
                    let mut exprs = projection.expr.clone();
                    exprs.push(position);
                    LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(input))?)
                }
                LogicalPlan::SubqueryAlias(alias) => LogicalPlan::SubqueryAlias(
                    SubqueryAlias::try_new(Arc::new(input), alias.alias.clone())?,
                ),
                _ => return internal_err!("a node above the sort: {}", node.display()),
            })
        })
    }
}

/// The [`ReferencePlan`] of `plan`, a query with a top-level `ORDER BY` (see
/// [`has_top_level_order_by`]). `state` evaluates the expressions of the
/// limits above its top-level sort.
fn reference_plan(plan: &LogicalPlan, state: &SessionState) -> Result<ReferencePlan> {
    match plan {
        LogicalPlan::Sort(sort) => {
            // The values of a key that depends on a volatile function, also
            // through the columns of the input of the sort (e.g. `r` in
            // `SELECT r FROM (SELECT random() AS r) ORDER BY r`), cannot be
            // computed again.
            let volatile = volatile_columns(&sort.input);
            let schema = sort.input.schema();
            let keys = sort
                .expr
                .iter()
                .map(|sort_expr| &sort_expr.expr)
                .take_while(|key| !depends_on_volatile(key, schema, &volatile))
                .collect::<Vec<_>>();
            let mut exprs = schema
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
            let sort_options = sort.expr[..keys.len()]
                .iter()
                .map(|sort_expr| SortOptions {
                    descending: !sort_expr.asc,
                    nulls_first: sort_expr.nulls_first,
                })
                .collect();
            Ok(ReferencePlan {
                sort: sorted,
                input: Arc::clone(schema),
                keys: keys.iter().map(ToString::to_string).collect(),
                sort_options,
                above: vec![],
                window: Window {
                    skip: 0,
                    fetch: sort.fetch,
                },
            })
        }
        LogicalPlan::Distinct(Distinct::On(distinct)) if distinct.sort_expr.is_some() => {
            // DataFusion orders the rows of `DISTINCT ON` by the expressions of
            // its `ORDER BY` that are its `ON` expressions.
            let sort_expr = distinct.sort_expr.as_deref().unwrap_or_default();
            let order = &sort_expr[..distinct.on_expr.len().min(sort_expr.len())];
            let volatile = volatile_columns(&distinct.input);
            let schema = distinct.input.schema();
            let keys = order
                .iter()
                .take_while(|sort_expr| {
                    !depends_on_volatile(&sort_expr.expr, schema, &volatile)
                })
                .collect::<Vec<_>>();
            let mut select = distinct.select_expr.clone();
            select.extend(
                keys.iter()
                    .enumerate()
                    .map(|(i, key)| key.expr.clone().alias(key_column(i))),
            );
            let with_keys = DistinctOn::try_new(
                distinct.on_expr.clone(),
                select,
                distinct.sort_expr.clone(),
                Arc::clone(&distinct.input),
            )?;
            let sorted = LogicalPlan::Sort(Sort {
                expr: keys
                    .iter()
                    .enumerate()
                    .map(|(i, key)| SortExpr {
                        expr: Expr::Column(Column::from_name(key_column(i))),
                        asc: key.asc,
                        nulls_first: key.nulls_first,
                    })
                    .collect(),
                input: Arc::new(LogicalPlan::Distinct(Distinct::On(with_keys))),
                fetch: None,
            });
            Ok(ReferencePlan {
                sort: sorted,
                input: Arc::clone(&distinct.schema),
                keys: keys.iter().map(|key| key.expr.to_string()).collect(),
                sort_options: keys
                    .iter()
                    .map(|key| SortOptions {
                        descending: !key.asc,
                        nulls_first: key.nulls_first,
                    })
                    .collect(),
                above: vec![],
                window: Window {
                    skip: 0,
                    fetch: None,
                },
            })
        }
        LogicalPlan::Limit(limit) => {
            let mut reference = reference_plan(&limit.input, state)?;
            // The limits may be constant expressions, e.g. `LIMIT 1 + 1`.
            let evaluate = |expr: &Option<Box<Expr>>| {
                expr.as_deref()
                    .map(|expr| {
                        Ok::<_, DataFusionError>(Box::new(constant(state, expr)?))
                    })
                    .transpose()
            };
            let limit = Limit {
                skip: evaluate(&limit.skip)?,
                fetch: evaluate(&limit.fetch)?,
                input: Arc::clone(&limit.input),
            };
            let (SkipType::Literal(skip), FetchType::Literal(fetch)) =
                (limit.get_skip_type()?, limit.get_fetch_type()?)
            else {
                return not_impl_err!("a LIMIT or an OFFSET that is not a constant");
            };
            reference.window = reference.window.then(skip, fetch);
            Ok(reference)
        }
        LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. })
        | LogicalPlan::Projection(Projection { input, .. }) => {
            let mut reference = reference_plan(input, state)?;
            reference.above.push(plan.clone());
            Ok(reference)
        }
        _ => internal_err!("the query has no top-level ORDER BY"),
    }
}

/// The value of `expr`, a constant expression, evaluated by `state`, as an
/// `Int64` literal.
fn constant(state: &SessionState, expr: &Expr) -> Result<Expr> {
    if let Expr::Literal(ScalarValue::Int64(_), _) = expr {
        return Ok(expr.clone());
    }
    let expr = state.create_physical_expr(expr.clone(), &DFSchema::empty())?;
    let options = RecordBatchOptions::new().with_row_count(Some(1));
    let batch =
        RecordBatch::try_new_with_options(Arc::new(Schema::empty()), vec![], &options)?;
    let value = match expr.evaluate(&batch)? {
        ColumnarValue::Scalar(value) => value,
        ColumnarValue::Array(array) => ScalarValue::try_from_array(array.as_ref(), 0)?,
    };
    Ok(Expr::Literal(value.cast_to(&DataType::Int64)?, None))
}

/// Whether the values of each output column of `plan` depend on a volatile
/// function (e.g. `random()`): running `plan` again may give other values.
fn volatile_columns(plan: &LogicalPlan) -> Vec<bool> {
    let width = plan.schema().fields().len();
    let columns = match plan {
        LogicalPlan::Projection(projection) => {
            let (input, schema) = (
                volatile_columns(&projection.input),
                projection.input.schema(),
            );
            projection
                .expr
                .iter()
                .map(|expr| depends_on_volatile(expr, schema, &input))
                .collect()
        }
        LogicalPlan::SubqueryAlias(alias) => volatile_columns(&alias.input),
        LogicalPlan::Aggregate(aggregate) => {
            let (input, schema) =
                (volatile_columns(&aggregate.input), aggregate.input.schema());
            aggregate
                .group_expr
                .iter()
                .chain(&aggregate.aggr_expr)
                .map(|expr| depends_on_volatile(expr, schema, &input))
                .collect()
        }
        LogicalPlan::Window(window) => {
            let (input, schema) =
                (volatile_columns(&window.input), window.input.schema());
            let window_columns = window
                .window_expr
                .iter()
                .map(|expr| depends_on_volatile(expr, schema, &input))
                .collect::<Vec<_>>();
            input.into_iter().chain(window_columns).collect()
        }
        LogicalPlan::Union(union) => {
            let inputs = union
                .inputs
                .iter()
                .map(|input| volatile_columns(input))
                .collect::<Vec<_>>();
            (0..width)
                .map(|i| inputs.iter().any(|input| input.get(i).is_none_or(|&v| v)))
                .collect()
        }
        // Nodes whose output columns are columns of their inputs.
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Repartition(_)
        | LogicalPlan::Distinct(Distinct::All(_))
        | LogicalPlan::Join(_) => {
            let inputs = plan
                .inputs()
                .into_iter()
                .map(|input| (input.schema(), volatile_columns(input)))
                .collect::<Vec<_>>();
            plan.schema()
                .iter()
                .map(|(qualifier, field)| {
                    let column = Column::new(qualifier.cloned(), field.name());
                    inputs
                        .iter()
                        .find_map(|(schema, volatile)| {
                            schema.maybe_index_of_column(&column).map(|i| volatile[i])
                        })
                        .unwrap_or(true)
                })
                .collect()
        }
        LogicalPlan::TableScan(_) | LogicalPlan::EmptyRelation(_) => vec![false; width],
        _ => vec![has_volatile_expressions(plan); width],
    };
    if columns.len() == width {
        columns
    } else {
        vec![has_volatile_expressions(plan); width]
    }
}

/// Whether the values of `expr` over rows of `schema`, whose columns are
/// `volatile` (see [`volatile_columns`]), depend on a volatile function.
fn depends_on_volatile(expr: &Expr, schema: &DFSchema, volatile: &[bool]) -> bool {
    calls_volatile_function(expr)
        || expr.column_refs().into_iter().any(|column| {
            schema
                .maybe_index_of_column(column)
                .is_none_or(|i| volatile[i])
        })
}

/// Whether `expr` calls a volatile function, also in its subqueries.
fn calls_volatile_function(expr: &Expr) -> bool {
    expr.exists(|expr| {
        Ok(expr.is_volatile_node()
            || match expr {
                Expr::ScalarSubquery(subquery) => {
                    has_volatile_expressions(&subquery.subquery)
                }
                Expr::Exists(exists) => {
                    has_volatile_expressions(&exists.subquery.subquery)
                }
                Expr::InSubquery(in_subquery) => {
                    has_volatile_expressions(&in_subquery.subquery.subquery)
                }
                Expr::SetComparison(comparison) => {
                    has_volatile_expressions(&comparison.subquery.subquery)
                }
                _ => false,
            })
    })
    .unwrap_or(true)
}

/// Whether an expression of `plan` or of its inputs calls a volatile function.
fn has_volatile_expressions(plan: &LogicalPlan) -> bool {
    plan.exists(|node| Ok(node.expressions().iter().any(calls_volatile_function)))
        .unwrap_or(true)
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
            if tolerance > reference.tolerance {
                return Err(format!(
                    "the sorted reference is computed for the float tolerance {}, but the \
                     results are compared within {tolerance}: compute it with the \
                     same options (see `run_both_engines`)",
                    reference.tolerance
                ));
            }
            let sorted = reference
                .sorted_rows(tolerance)
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
/// pipeline engine has exactly the output schema.
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
    if let Some(schema) = actual
        .batches
        .iter()
        .map(RecordBatch::schema)
        .find(|schema| *schema != actual.schema)
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

/// The rows of a [`SortedReference`], read for comparisons.
struct SortedRows {
    /// The output columns of all sorted rows, unless they are not computed
    /// (see [`SortedReference::uncomputed`]).
    values: Vec<Option<Row>>,
    /// The ranges of rows whose output columns are not computed, with the
    /// reason.
    uncomputed: Vec<(Range<usize>, String)>,
    /// The classes of the sorted rows: the ranges of rows whose keys are
    /// exactly equal up to their first floating-point value that may vary,
    /// and these keys (see [`decision_keys`]).
    classes: Vec<(Range<usize>, Row)>,
    /// For each class, whether its rows have keys after these keys.
    truncated: Vec<bool>,
    /// For each class, the other classes that it ties with, in their order
    /// (see [`classes_and_ties`]).
    tied: Vec<Vec<usize>>,
    /// The sorted rows that the query returns.
    window: Range<usize>,
}

impl SortedRows {
    /// The output columns of the sorted row `row`, which must be computed.
    fn computed(&self, row: usize) -> &Row {
        self.values[row]
            .as_ref()
            .expect("the output columns of the row are computed")
    }

    /// The output columns of the sorted row `row`, displayed, or why they are
    /// not computed.
    fn display(&self, row: usize) -> String {
        match &self.values[row] {
            Some(values) => values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            None => {
                let i = self.uncomputed.partition_point(|(rows, _)| rows.end <= row);
                let reason = self.uncomputed.get(i).map_or("", |(_, reason)| reason);
                format!("not computed: {reason}")
            }
        }
    }
}

/// The keys of a row (the values of its `ORDER BY` keys, `keys`) that decide
/// its order relative to rows of other keys, whether `keys` has values after
/// them, and whether their last floating-point value may vary: the keys up to
/// their first floating-point value that may vary (`may_vary`, included), in
/// the order in which they are compared (lexicographically, also the
/// elements of lists and the fields of structs).
///
/// The floating-point values of each engine may be off by the tolerance: of
/// two rows whose keys are exactly equal before their first floating-point
/// values that may vary, and whose first such values are close, either may be
/// first, whatever their later keys. Floating-point values that the tolerance
/// does not let vary (see [`may_vary`]) are compared like other values.
fn decision_keys(
    keys: &[Value],
    may_vary: &dyn Fn(f64, Precision) -> bool,
) -> (Row, bool, bool) {
    /// Push the values of `values` up to their first floating-point value
    /// that may vary (included) to `prefix`. Returns `None` if they have none,
    /// or whether they have values after it.
    fn push_prefix(
        values: &[Value],
        may_vary: &dyn Fn(f64, Precision) -> bool,
        prefix: &mut Vec<Value>,
    ) -> Option<bool> {
        for (i, value) in values.iter().enumerate() {
            let found = match value {
                Value::Float(float, precision) => {
                    prefix.push(value.clone());
                    may_vary(*float, *precision).then_some(false)
                }
                Value::List(elements) | Value::Struct(elements) => {
                    let mut inner = vec![];
                    let found = push_prefix(elements, may_vary, &mut inner);
                    prefix.push(match value {
                        Value::List(_) => Value::List(inner),
                        _ => Value::Struct(inner),
                    });
                    found
                }
                Value::Null | Value::Scalar(_) => {
                    prefix.push(value.clone());
                    None
                }
            };
            if let Some(truncated) = found {
                return Some(truncated || i + 1 < values.len());
            }
        }
        None
    }
    let mut prefix = vec![];
    let found = push_prefix(keys, may_vary, &mut prefix);
    (prefix, found.unwrap_or(false), found.is_some())
}

/// Whether the floating-point value `value` of the precision `precision` is
/// equal within the relative tolerance `tolerance` to other values of this
/// precision (see [`floats_equal`]), so that the value of another engine may
/// be smaller or larger. Zeros, infinities and `NaN`s are not (zeros unless
/// `tolerance` is at least 1), nor is any value if `tolerance` is smaller than
/// the distance to its neighbors (e.g. `0`, or the default tolerance for
/// `Float32` and `Float16` values).
fn may_vary(value: f64, precision: Precision, tolerance: f64) -> bool {
    value.is_finite()
        && precision
            .neighbors(value)
            .into_iter()
            .any(|other| floats_equal(value, other, tolerance))
}

/// The precision of a floating-point value, see [`Value::Float`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Precision {
    /// A `Float16` value.
    Half,
    /// A `Float32` value.
    Single,
    /// A `Float64` value.
    Double,
}

impl Precision {
    /// The value of this precision that is the nearest to `value`.
    fn round(self, value: f64) -> f64 {
        match self {
            Self::Double => value,
            // `f32` values are `f64` values.
            #[expect(clippy::cast_possible_truncation)]
            Self::Single => f64::from(value as f32),
            Self::Half => {
                <Float16Type as ArrowPrimitiveType>::Native::from_f64(value).to_f64()
            }
        }
    }

    /// The next larger and the next smaller value of this precision than
    /// `value`, a value of this precision.
    fn neighbors(self, value: f64) -> [f64; 2] {
        match self {
            Self::Double => [value.next_up(), value.next_down()],
            // `f32` values are `f64` values.
            #[expect(clippy::cast_possible_truncation)]
            Self::Single => {
                let value = value as f32;
                [f64::from(value.next_up()), f64::from(value.next_down())]
            }
            Self::Half => {
                type F16 = <Float16Type as ArrowPrimitiveType>::Native;
                let bits = F16::from_f64(value).to_bits();
                // The bits of the values of the sign of `value` follow their
                // magnitude; the neighbors of zeros are the smallest values.
                let (up, down) = if value == 0.0 {
                    (0x0001, 0x8001)
                } else if value > 0.0 {
                    (bits + 1, bits - 1)
                } else {
                    (bits - 1, bits + 1)
                };
                [F16::from_bits(up).to_f64(), F16::from_bits(down).to_f64()]
            }
        }
    }
}

/// A class of sorted rows, see [`classes_and_ties`].
#[derive(Debug, Clone)]
struct Class {
    /// The sorted rows of the class.
    rows: Range<usize>,
    /// Their [`decision_keys`].
    keys: Row,
    /// Whether the rows have keys after `keys`.
    truncated: bool,
    /// Whether all their keys are equal, not only `keys`.
    equal: bool,
}

/// The classes of the sorted rows whose `ORDER BY` keys are `keys` (of each
/// sorted row, in their order), and for each class the other classes that it
/// ties with (in their order), within the tolerance `tolerance`. `options`
/// are the sort options of the keys.
///
/// A class has the consecutive rows whose keys are exactly equal up to their
/// first floating-point value that may vary within the tolerance (their
/// [`decision_keys`]): they may come in any order, and the order of other
/// rows relative to them does not depend on their later keys. Unless the
/// values that the engines may have for this floating-point value meet
/// those of another class without crossing them: then the later keys of the
/// rows decide, and the rows of both classes are in classes of rows whose
/// keys are all equal.
fn classes_and_ties(
    keys: &[Row],
    options: &[SortOptions],
    tolerance: f64,
) -> (Vec<Class>, Vec<Vec<usize>>) {
    let decisions = keys
        .iter()
        .map(|keys| {
            decision_keys(keys, &|value, precision| {
                may_vary(value, precision, tolerance)
            })
        })
        .collect::<Vec<_>>();
    // The runs of the rows `rows` whose decision keys (or all keys) are equal.
    let runs = |rows: Range<usize>, all_keys: bool| {
        let mut classes: Vec<Class> = vec![];
        for row in rows {
            let (decision, truncated, _) = &decisions[row];
            match classes.last_mut() {
                Some(class)
                    if (if all_keys {
                        rows_equal(&keys[class.rows.start], &keys[row], 0.0)
                    } else {
                        rows_equal(&class.keys, decision, 0.0)
                    }) =>
                {
                    class.rows.end = row + 1;
                    class.truncated |= truncated;
                    class.equal &= rows_equal(&keys[class.rows.start], &keys[row], 0.0);
                }
                _ => classes.push(Class {
                    rows: row..row + 1,
                    keys: decision.clone(),
                    truncated: *truncated,
                    equal: true,
                }),
            }
        }
        classes
    };
    let classes = runs(0..keys.len(), false);
    let (tied, meeting) = tied_classes(&classes, keys, options, tolerance);
    if !meeting.contains(&true) {
        return (classes, tied);
    }
    let classes = classes
        .into_iter()
        .zip(meeting)
        .flat_map(|(class, meets)| {
            if meets && !class.equal {
                runs(class.rows, true)
            } else {
                vec![class]
            }
        })
        .collect::<Vec<_>>();
    let (tied, _) = tied_classes(&classes, keys, options, tolerance);
    (classes, tied)
}

/// The smallest and the largest values of the precision `precision` that are
/// equal to the finite value `value` within the relative tolerance
/// `tolerance` (see [`floats_equal`]): the values that another engine may
/// have for it. The zeros of both signs are one value, and, within a
/// tolerance of at least 1, all finite values are equal to each other.
fn interval(value: f64, precision: Precision, tolerance: f64) -> (f64, f64) {
    let value = normalized(value);
    if tolerance >= 1.0 {
        return (f64::MIN, f64::MAX);
    }
    let equal = |other: f64| other.is_finite() && floats_equal(value, other, tolerance);
    // Estimates of the bounds, which are `value * (1 - tolerance)` and
    // `value / (1 - tolerance)` but for rounding, moved to the last values
    // that are equal to `value`.
    let (near, far) = (value * (1.0 - tolerance), value / (1.0 - tolerance));
    let (low, high) = if value >= 0.0 {
        (near, far)
    } else {
        (far, near)
    };
    let (mut low, mut high) = (precision.round(low), precision.round(high));
    for _ in 0..64 {
        if high <= value || equal(high) {
            break;
        }
        high = precision.neighbors(high)[1];
    }
    for _ in 0..64 {
        let next = precision.neighbors(high)[0];
        if !equal(next) {
            break;
        }
        high = next;
    }
    for _ in 0..64 {
        if low >= value || equal(low) {
            break;
        }
        low = precision.neighbors(low)[0];
    }
    for _ in 0..64 {
        let next = precision.neighbors(low)[1];
        if !equal(next) {
            break;
        }
        low = next;
    }
    (low.min(value), high.max(value))
}

/// How the values that the engines may have for two floating-point values
/// (see [`interval`]) compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Meeting {
    /// Either may be smaller.
    Cross,
    /// They may be equal, but neither smaller than the other.
    Meet,
    /// They are always in the same order.
    Apart,
}

/// How the values that the engines may have for the floating-point values
/// `a` and `b` compare, within the tolerance `tolerance`. `NaN`s are equal.
fn meeting(a: (f64, Precision), b: (f64, Precision), tolerance: f64) -> Meeting {
    if a.0.is_nan() || b.0.is_nan() {
        return if a.0.is_nan() && b.0.is_nan() {
            Meeting::Meet
        } else {
            Meeting::Apart
        };
    }
    if !a.0.is_finite() || !b.0.is_finite() {
        return if a.0 == b.0 {
            Meeting::Meet
        } else {
            Meeting::Apart
        };
    }
    let ((a_low, a_high), (b_low, b_high)) =
        (interval(a.0, a.1, tolerance), interval(b.0, b.1, tolerance));
    // The largest value of the smaller one, and the smallest of the other.
    let (high, low) = if normalized(a.0) <= normalized(b.0) {
        (a_high, b_low)
    } else {
        (b_high, a_low)
    };
    if low < high {
        Meeting::Cross
    } else if low == high {
        Meeting::Meet
    } else {
        Meeting::Apart
    }
}

/// How a value of a key of a row may sort relative to the value of another
/// row, within the tolerance, see [`precedence`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Precedence {
    /// It may sort before it.
    Before,
    /// It may be equal to it, but not sort before it.
    Equal,
    /// It sorts after it.
    After,
}

/// How the value `later` of a key may sort relative to the value `earlier`,
/// in the values that the engines may have for them within the tolerance
/// `tolerance` (see [`interval`]), with the sort options `options` of the
/// key. Lists and structs are compared lexicographically.
fn precedence(
    later: &Value,
    earlier: &Value,
    options: SortOptions,
    tolerance: f64,
) -> Precedence {
    // The precedence of `later` if it compares as `ordering` to `earlier` in
    // ascending order.
    let ordered = |ordering: std::cmp::Ordering| match ordering {
        std::cmp::Ordering::Equal => Precedence::Equal,
        std::cmp::Ordering::Less if !options.descending => Precedence::Before,
        std::cmp::Ordering::Greater if options.descending => Precedence::Before,
        _ => Precedence::After,
    };
    let nulls = |later_is_null: bool| {
        if later_is_null == options.nulls_first {
            Precedence::Before
        } else {
            Precedence::After
        }
    };
    match (later, earlier) {
        (Value::Null, Value::Null) => Precedence::Equal,
        (Value::Null, _) => nulls(true),
        (_, Value::Null) => nulls(false),
        (Value::Float(y, y_precision), Value::Float(x, x_precision)) => {
            if !y.is_finite() || !x.is_finite() {
                return match meeting((*y, *y_precision), (*x, *x_precision), tolerance) {
                    Meeting::Meet => Precedence::Equal,
                    _ => ordered(y.total_cmp(x)),
                };
            }
            let (y_low, y_high) = interval(*y, *y_precision, tolerance);
            let (x_low, x_high) = interval(*x, *x_precision, tolerance);
            let (before, equal) = if options.descending {
                (y_high > x_low, y_high == x_low)
            } else {
                (y_low < x_high, y_low == x_high)
            };
            if before {
                Precedence::Before
            } else if equal {
                Precedence::Equal
            } else {
                Precedence::After
            }
        }
        (Value::Scalar(y), Value::Scalar(x)) => {
            y.partial_cmp(x).map_or(Precedence::After, ordered)
        }
        (Value::List(ys), Value::List(xs)) | (Value::Struct(ys), Value::Struct(xs)) => {
            for (y, x) in ys.iter().zip(xs) {
                match precedence(y, x, options, tolerance) {
                    Precedence::Equal => {}
                    other => return other,
                }
            }
            ordered(ys.len().cmp(&xs.len()))
        }
        _ => Precedence::After,
    }
}

/// How the row of `ORDER BY` keys `later` may sort relative to the row of
/// keys `earlier`, in the values that the engines may have for them within
/// the tolerance `tolerance` (see [`precedence`]): the first key that is not
/// equal decides. `options` are the sort options of the keys (ascending with
/// nulls last if missing).
fn row_precedence(
    later: &[Value],
    earlier: &[Value],
    options: &[SortOptions],
    tolerance: f64,
) -> Precedence {
    for (i, (y, x)) in later.iter().zip(earlier).enumerate() {
        let options = options.get(i).copied().unwrap_or(SortOptions {
            descending: false,
            nulls_first: false,
        });
        match precedence(y, x, options, tolerance) {
            Precedence::Equal => {}
            other => return other,
        }
    }
    Precedence::Equal
}

/// Whether the row of `ORDER BY` keys `later` may sort before the row of keys
/// `earlier`, or be equal to it, see [`row_precedence`].
fn may_precede(
    later: &[Value],
    earlier: &[Value],
    options: &[SortOptions],
    tolerance: f64,
) -> bool {
    row_precedence(later, earlier, options, tolerance) != Precedence::After
}

/// The first and the last positions of sorted rows at which the rows of each
/// class of `classes` may be, as they tie with the classes `tied` (see
/// [`tied_classes`]).
fn bounds(classes: &[(Range<usize>, Row)], tied: &[Vec<usize>]) -> Vec<(usize, usize)> {
    classes
        .iter()
        .enumerate()
        .map(|(class, (rows, _))| {
            let tied_rows = |range: Range<usize>| {
                tied[class]
                    .iter()
                    .filter(|other| range.contains(*other))
                    .map(|&other| classes[other].0.len())
                    .sum::<usize>()
            };
            (
                rows.start - tied_rows(0..class),
                rows.end - 1 + tied_rows(class + 1..classes.len()),
            )
        })
        .collect()
}

impl SortedReference {
    /// All sorted rows, in classes of rows whose keys are exactly equal up to
    /// their first floating-point value that may vary within the tolerance
    /// `tolerance` (see [`classes`]).
    fn sorted_rows(&self, tolerance: f64) -> Result<SortedRows> {
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
        let row = |batch: &RecordBatch, row: usize, columns: Range<usize>| {
            columns
                .map(|column| Value::try_new(batch.column(column).as_ref(), row))
                .collect::<Result<Row>>()
        };
        let rows = self
            .batches
            .iter()
            .flat_map(|batch| (0..batch.num_rows()).map(move |i| (batch, i)));
        let keys = rows
            .clone()
            .map(|(batch, i)| row(batch, i, self.columns..width))
            .collect::<Result<Vec<_>>>()?;
        let (classes, tied) = classes_and_ties(&keys, &self.sort_options, tolerance);
        let truncated = classes.iter().map(|class| class.truncated).collect();
        let classes = classes
            .into_iter()
            .map(|class| (class.rows, class.keys))
            .collect();
        let mut values = vec![];
        for (position, (batch, i)) in rows.enumerate() {
            let next = self
                .uncomputed
                .partition_point(|(rows, _)| rows.end <= position);
            let computed = self
                .uncomputed
                .get(next)
                .is_none_or(|(rows, _)| !rows.contains(&position));
            values.push(
                computed
                    .then(|| row(batch, i, 0..self.columns))
                    .transpose()?,
            );
        }
        let window = Window {
            skip: self.skip,
            fetch: self.fetch,
        }
        .rows(values.len());
        Ok(SortedRows {
            values,
            uncomputed: self.uncomputed.clone(),
            classes,
            truncated,
            tied,
            window,
        })
    }
}

/// For each class of sorted rows (`classes`, of the rows of keys `keys`), the
/// other classes whose keys tie with its keys (in their order), within the
/// tolerance `tolerance`; and the classes whose keys meet the keys of
/// another class, but whose ties with it depend on the later keys of their
/// rows, which are not all equal. `options` are the sort options of the keys.
///
/// The keys of the classes are their [`decision_keys`]: all but their last
/// floating-point value cannot vary within the tolerance. Of two classes, the
/// later one ties with the earlier one if its keys may sort before them, or
/// be equal to them (see [`row_precedence`]). That is only possible:
/// * if their keys have the same [`Shape`] (their values but their
///   floating-point values are equal), and the same floating-point values but
///   the last, which are close; or
/// * if they are equal up to a zero, which has another sign in each class:
///   DataFusion sorts `-0.0` before `0.0`, but they are equal, so the later
///   keys may order them otherwise.
// The keys of `Shape`s hold `ScalarValue`s, whose Arrow buffers have a `Mutex`
// for memory accounting with the `arrow-buffer/pool` feature (enabled by
// `--all-features`): it affects neither their equality nor their hashing. Not
// `expect`: the lint only fires with that feature.
#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
fn tied_classes(
    classes: &[Class],
    keys: &[Row],
    options: &[SortOptions],
    tolerance: f64,
) -> (Vec<Vec<usize>>, Vec<bool>) {
    // The floating-point values of each class, as bits but the last.
    let floats = |row: &[Value]| {
        let mut floats = vec![];
        for value in row {
            value.collect_floats_of_precision(&mut floats);
        }
        floats
    };
    let bits = |floats: &[(f64, Precision)]| {
        floats
            .iter()
            .map(|(value, _)| normalized(*value).to_bits())
            .collect::<Vec<_>>()
    };
    // The pairs of classes that may tie (each once, the earlier class first).
    let mut pairs = HashSet::new();
    // Of the same shape and floating-point values but the last, which are
    // within twice the tolerance (but for rounding) if their values may cross
    // or meet.
    let mut members_of: HashMap<ShapeAndFloats<'_>, Vec<usize>> = HashMap::new();
    let mut last = vec![0.0; classes.len()];
    for (index, class) in classes.iter().enumerate() {
        let mut floats = floats(&class.keys);
        let Some((value, _)) = floats.pop() else {
            continue;
        };
        last[index] = normalized(value);
        members_of
            .entry((Shape(&class.keys), bits(&floats)))
            .or_default()
            .push(index);
    }
    let reach = 2.0 * tolerance + 8.0 * f64::EPSILON;
    for members in members_of.values().filter(|members| members.len() > 1) {
        let values = members
            .iter()
            .map(|&class| vec![last[class]])
            .collect::<Vec<_>>();
        for (i, others) in equal_pairs(&values, &values, reach).into_iter().enumerate() {
            for j in others {
                let (a, b) = (members[i], members[j]);
                if a < b {
                    pairs.insert((a, b));
                }
            }
        }
    }
    // Equal up to the `n`th zero of their keys, which has another sign.
    let mut prefixes = vec![];
    for (index, class) in classes.iter().enumerate() {
        let zeros = floats(&class.keys)
            .iter()
            .filter(|(value, _)| *value == 0.0)
            .count();
        for n in 1..=zeros {
            let seen = std::cell::Cell::new(0);
            let (prefix, _, _) = decision_keys(&class.keys, &|value, _| {
                seen.set(seen.get() + usize::from(value == 0.0));
                value == 0.0 && seen.get() == n
            });
            let negative = floats(&prefix)
                .last()
                .is_some_and(|(value, _)| value.is_sign_negative());
            prefixes.push((index, prefix, negative));
        }
    }
    let mut signs_of: HashMap<ShapeAndFloats<'_>, Vec<(usize, bool)>> = HashMap::new();
    for (index, prefix, negative) in &prefixes {
        signs_of
            .entry((Shape(prefix), bits(&floats(prefix))))
            .or_default()
            .push((*index, *negative));
    }
    for members in signs_of.values() {
        for (i, &(a, a_negative)) in members.iter().enumerate() {
            for &(b, b_negative) in &members[i + 1..] {
                if a_negative != b_negative && a != b {
                    pairs.insert((a.min(b), a.max(b)));
                }
            }
        }
    }

    let mut tied = vec![vec![]; classes.len()];
    let mut meets = vec![false; classes.len()];
    for (a, b) in pairs {
        let (earlier, later) = (&classes[a], &classes[b]);
        let ties = match row_precedence(&later.keys, &earlier.keys, options, tolerance) {
            Precedence::Before => true,
            Precedence::After => false,
            // The values of their last floating-point values meet.
            Precedence::Equal if !earlier.truncated && !later.truncated => true,
            Precedence::Equal if earlier.equal && later.equal => may_precede(
                &keys[later.rows.start],
                &keys[earlier.rows.start],
                options,
                tolerance,
            ),
            Precedence::Equal => {
                meets[a] = true;
                meets[b] = true;
                false
            }
        };
        if ties {
            tied[a].push(b);
            tied[b].push(a);
        }
    }
    for others in &mut tied {
        others.sort_unstable();
        others.dedup();
    }
    (tied, meets)
}

/// The normalized floating-point values of `row`, as bits: rows of the same
/// [`Shape`] with the same bits are equal.
fn float_bits(row: &[Value]) -> Vec<u64> {
    let mut floats = vec![];
    for value in row {
        value.collect_floats(&mut floats);
    }
    floats
        .into_iter()
        .map(|value| normalized(value).to_bits())
        .collect()
}

/// Compare the rows `result` of `names.1` with the sorted rows of `names.0`,
/// see [`ResultOrder::Sorted`]: `result` must be the rows that the query
/// returns from an order of all sorted rows in which rows are only out of
/// their sorted order relative to rows that they tie with.
///
/// The sorted rows are divided into classes of rows whose keys are exactly
/// equal up to their first floating-point value that may vary within the
/// tolerance (their [`decision_keys`]).
/// The rows of a class may be at the positions of its rows, and of the rows
/// of the classes that it ties with. A maximum matching of the rows of
/// `result` to the sorted rows with their values (within the tolerance) at
/// their positions must match all rows of `result`. The classes are divided
/// into clusters, which ties do not cross: in a cluster whose classes all tie
/// with each other, rows may be in any order, so the matching is an order.
/// In other clusters, [`ClusterSearch`] searches which rows the `OFFSET`
/// skips, and an assignment of the rows of `result` to sorted rows, such that
/// every row of this order (the skipped rows, the rows of `result`, then the
/// rows that the `LIMIT` cuts) ties with the earlier rows of later classes.
// The rows of `Shape`s hold `ScalarValue`s, whose Arrow buffers have a
// `Mutex` for memory accounting with the `arrow-buffer/pool` feature (enabled
// by `--all-features`): it affects neither their equality nor their hashing.
// Not `expect`: the lint only fires with that feature.
#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
fn compare_sorted(
    (reference_name, result_name): (&str, &str),
    result: &[Row],
    sorted: &SortedRows,
    (columns, keys): (&str, &str),
    tolerance: f64,
) -> std::result::Result<(), String> {
    let window = sorted.window.clone();
    if result.len() != window.len() {
        return Err(format!(
            "{result_name} returned {}, but the query returns {} of the sorted rows of \
             {reference_name}",
            count_rows(result.len()),
            count_rows(window.len())
        ));
    }
    let classes = &sorted.classes;
    // The keys of each engine may be off by the tolerance. Ties are not
    // transitive: two rows that tie with a row may not tie with each other.
    let tied = &sorted.tied;
    let ties = |a: usize, b: usize| a == b || tied[a].binary_search(&b).is_ok();
    let display = |row: &Row| {
        row.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    // The keys of a class, `…` if its rows have later keys.
    let display_keys = |class: usize| {
        let keys = display(&classes[class].1);
        if sorted.truncated[class] {
            format!("{keys}, …")
        } else {
            keys
        }
    };
    let rule = if keys.is_empty() {
        "all rows tie, as the first ORDER BY key is volatile".to_string()
    } else {
        format!(
            "rows tie if their ORDER BY keys ({keys}) are equal up to their first \
             floating-point value that may vary within the tolerance (not a zero, an \
             infinity or NaN), and either of these values may be the smaller one \
             within the tolerance, whatever their later keys (if they may only be \
             equal, their later keys decide)"
        )
    };

    // The first and the last positions of sorted rows at which the rows of
    // each class may be.
    let bounds = bounds(classes, tied);

    // The groups of sorted rows of a class with equal values, and the groups
    // of each class and shape.
    let mut groups: Vec<(usize, usize, usize)> = vec![];
    let mut group_of = HashMap::new();
    let mut groups_of: HashMap<(usize, Shape<'_>), Vec<usize>> = HashMap::new();
    for (class, (rows, _)) in classes.iter().enumerate() {
        // The query returns no rows whose output columns are not computed.
        for row in rows.clone().filter(|&row| sorted.values[row].is_some()) {
            let values = sorted.computed(row);
            let group = *group_of
                .entry((class, Shape(values), float_bits(values)))
                .or_insert_with(|| {
                    groups_of
                        .entry((class, Shape(values)))
                        .or_default()
                        .push(groups.len());
                    groups.push((class, row, 0));
                    groups.len() - 1
                });
            groups[group].2 += 1;
        }
    }

    // The groups that each row of the result may be: their classes may be at
    // its position, and their values are its values.
    let mut by_first_position = (0..classes.len()).collect::<Vec<_>>();
    by_first_position.sort_by_key(|&class| bounds[class].0);
    let (mut next, mut active) = (0, vec![]);
    let mut candidates = Vec::with_capacity(result.len());
    for (i, row) in result.iter().enumerate() {
        let position = window.start + i;
        while next < classes.len() && bounds[by_first_position[next]].0 <= position {
            active.push(by_first_position[next]);
            next += 1;
        }
        active.retain(|&class| bounds[class].1 >= position);
        let floats = row.iter().any(Value::has_floats);
        let mut groups_of_row = vec![];
        for &class in &active {
            for &group in groups_of.get(&(class, Shape(row))).into_iter().flatten() {
                if !floats || rows_equal(sorted.computed(groups[group].1), row, tolerance)
                {
                    groups_of_row.push(group);
                }
            }
        }
        candidates.push(groups_of_row);
    }

    // A maximum matching of the rows of the result to the sorted rows.
    let mut supply = vec![1; result.len()];
    let mut demand = groups.iter().map(|group| group.2).collect::<Vec<_>>();
    max_flow(&candidates, &mut supply, &mut demand);
    if let Some(i) = supply.iter().position(|&rows| rows > 0) {
        let row = &result[i];
        let floats = row.iter().any(Value::has_floats);
        // Whether sorted rows of other positions have its values.
        let elsewhere = groups.iter().find(|&&(_, group_row, _)| {
            let values = sorted.computed(group_row);
            Shape(values) == Shape(row) && (!floats || rows_equal(values, row, tolerance))
        });
        let position = window.start + i;
        let position_class = classes
            .iter()
            .position(|(rows, _)| rows.contains(&position))
            .expect("the window has sorted rows");
        return Err(match elsewhere {
            Some(&(class, _, _)) if candidates[i].is_empty() => format!(
                "row {i} of {result_name} is out of order: the sorted rows of \
                 {reference_name} with its values may not be at its position \
                 ({rule})\ncolumns: ({columns})\n  row {i}: ({}), keys: ({})\n  sorted \
                 row {position} of {reference_name}: ({}), keys: ({})",
                display(row),
                display_keys(class),
                sorted.display(position),
                display_keys(position_class)
            ),
            _ => format!(
                "row {i} of {result_name} differs: it is not among the sorted rows of \
                 {reference_name} that the query may return at its position (with \
                 their multiplicities)\ncolumns: ({columns})\n  row {i}: ({})\n  sorted \
                 row {position} of {reference_name}: ({})",
                display(row),
                sorted.display(position)
            ),
        });
    }
    // In a cluster of classes that all tie with each other, the rows may be in
    // any order: the matching is an order. In other clusters, whose ties are
    // not transitive, search an order of the sorted rows of the cluster that
    // has the rows of the result at its positions.
    let sizes = classes
        .iter()
        .map(|(rows, _)| rows.len())
        .collect::<Vec<_>>();
    let mut budget = MAX_SEARCHED_STATES;
    for cluster in clusters(tied) {
        let (first, end) = (
            classes[cluster.start].0.start,
            classes[cluster.end - 1].0.end,
        );
        let positions = window.start.max(first)..window.end.min(end);
        if positions.is_empty()
            || cluster
                .clone()
                .all(|class| tied[class].len() == cluster.len() - 1)
        {
            continue;
        }
        let search = ClusterSearch {
            classes: cluster,
            first,
            skip: positions.start - first,
            sizes: &sizes,
            bounds: &bounds,
            groups: &groups,
            ties: &ties,
        };
        let rows = positions.start - window.start..positions.end - window.start;
        let Err(failure) = search.run(rows, &candidates, &mut budget) else {
            continue;
        };
        return Err(match failure {
            ClusterFailure::Values(i) => format!(
                "row {i} of {result_name} differs: it is not among the sorted rows of \
                 {reference_name} that the query may return at its position (with \
                 their multiplicities, and in an order in which rows are only out of \
                 their sorted order relative to rows that they tie with)\ncolumns: \
                 ({columns})\n  row {i}: ({})\n  sorted row {} of {reference_name}: \
                 ({})",
                display(&result[i]),
                window.start + i,
                sorted.display(window.start + i)
            ),
            ClusterFailure::Order {
                row,
                class,
                earlier,
                later,
            } => format!(
                "row {row} of {result_name} is out of order: the query orders it before \
                 row {earlier}, and their keys do not tie ({rule})\ncolumns: \
                 ({columns})\n  row {earlier}: ({}), keys: ({})\n  row {row}: ({}), \
                 keys: ({})",
                display(&result[earlier]),
                display_keys(later),
                display(&result[row]),
                display_keys(class)
            ),
            ClusterFailure::Skipped {
                row,
                class,
                skipped,
            } => format!(
                "row {row} of {result_name} is not among the rows that the query \
                 returns: the OFFSET skips sorted rows of {reference_name} that the \
                 query orders after it, and whose keys do not tie with its keys \
                 ({rule})\ncolumns: ({columns})\n  row {row}: ({}), keys: ({})\n  keys of \
                 the skipped rows: ({})",
                display(&result[row]),
                display_keys(class),
                display_keys(skipped)
            ),
            ClusterFailure::Cut { row, class, cut } => format!(
                "row {row} of {result_name} is not among the rows that the query \
                 returns: the LIMIT cuts sorted rows of {reference_name} that the query \
                 orders before it, and whose keys do not tie with its keys \
                 ({rule})\ncolumns: ({columns})\n  row {row}: ({}), keys: ({})\n  keys of \
                 the cut rows: ({})",
                display(&result[row]),
                display_keys(class),
                display_keys(cut)
            ),
            ClusterFailure::SkippedAfterCut { skipped, cut } => format!(
                "the rows of {result_name} are not among the rows that the query \
                 returns: the OFFSET would skip sorted rows of {reference_name} that \
                 the query orders after sorted rows that the LIMIT would cut, and \
                 whose keys do not tie with theirs ({rule})\n  keys of the skipped rows: \
                 ({})\n  keys of the cut rows: ({})",
                display_keys(skipped),
                display_keys(cut)
            ),
            ClusterFailure::NoOrder => format!(
                "the rows of {result_name} are not among the rows that the query \
                 returns: no rows of {reference_name} may be the rows that the OFFSET \
                 skips ({rule})"
            ),
            ClusterFailure::GaveUp => format!(
                "cannot compare the rows of {result_name} with the sorted rows of \
                 {reference_name}: there are too many orders of rows whose ORDER BY \
                 keys tie (more than {MAX_SEARCHED_STATES} states searched)"
            ),
        });
    }
    Ok(())
}

/// The number of states that [`ClusterSearch`] may search, for all clusters of
/// a comparison, before it gives up.
const MAX_SEARCHED_STATES: usize = 1_000_000;

/// The clusters of classes (`tied`, see [`tied_classes`]): the ranges of
/// classes whose keys only tie with the keys of classes of the same range. A
/// row is never out of its sorted order relative to a row of another cluster.
fn clusters(tied: &[Vec<usize>]) -> Vec<Range<usize>> {
    let mut clusters = vec![];
    let (mut start, mut reach) = (0, 0);
    for (class, others) in tied.iter().enumerate() {
        if class > reach {
            clusters.push(start..class);
            start = class;
        }
        reach = reach
            .max(class)
            .max(others.last().copied().unwrap_or(class));
    }
    if start < tied.len() {
        clusters.push(start..tied.len());
    }
    clusters
}

/// Why no order of the sorted rows of a cluster has the rows of a result at
/// its positions, see [`ClusterSearch`].
enum ClusterFailure {
    /// No sorted row with the values of this row of the result is left.
    Values(usize),
    /// Row `row` of the result, of class `class`, would come after row
    /// `earlier`, of class `later`, which it does not tie with.
    Order {
        row: usize,
        class: usize,
        earlier: usize,
        later: usize,
    },
    /// Row `row` of the result, of class `class`, would come after rows of
    /// class `skipped` that the OFFSET skips, which it does not tie with.
    Skipped {
        row: usize,
        class: usize,
        skipped: usize,
    },
    /// The LIMIT would cut rows of class `cut` that the query orders before
    /// row `row`, of class `class`, which they do not tie with.
    Cut {
        row: usize,
        class: usize,
        cut: usize,
    },
    /// The LIMIT would cut rows of class `cut` that the query orders before
    /// rows of class `skipped` that the OFFSET skips, which they do not tie
    /// with.
    SkippedAfterCut { skipped: usize, cut: usize },
    /// No rows of the cluster may be the rows that the OFFSET skips.
    NoOrder,
    /// The search took more states than its budget.
    GaveUp,
}

/// The search of an order of the sorted rows of a cluster of classes, whose
/// ties are not transitive, in which some rows of a result are at their
/// positions, see [`compare_sorted`].
struct ClusterSearch<'a> {
    /// The classes of the cluster.
    classes: Range<usize>,
    /// The position of the first sorted row of the cluster.
    first: usize,
    /// The number of rows of the cluster that the OFFSET skips.
    skip: usize,
    /// The number of rows of each class.
    sizes: &'a [usize],
    /// The first and the last positions at which the rows of each class may
    /// be.
    bounds: &'a [(usize, usize)],
    /// The groups of sorted rows: their class, a row and their number of rows.
    groups: &'a [(usize, usize, usize)],
    /// Whether the keys of two classes tie.
    ties: &'a dyn Fn(usize, usize) -> bool,
}

impl ClusterSearch<'_> {
    /// Search an order of the sorted rows of the cluster in which every row
    /// ties with the earlier rows of later classes: first the rows that the
    /// OFFSET skips, in the order of their classes (which rows they are
    /// matters, not their order), then the rows `rows` of the result, which
    /// are assigned, in order, one of their `candidates` (groups of sorted
    /// rows with their values that may be at their positions) with rows left,
    /// and finally the rows that the LIMIT cuts, in the order of their classes.
    ///
    /// A depth-first search, which remembers the states (a number of rows
    /// placed and the rows left) that lead to no order, and gives up after
    /// `budget` states. Returns the failure of the deepest dead end if there is
    /// no order.
    fn run(
        &self,
        rows: Range<usize>,
        candidates: &[Vec<usize>],
        budget: &mut usize,
    ) -> std::result::Result<(), ClusterFailure> {
        let rows = rows.collect::<Vec<_>>();
        let start = self.classes.start;
        let classes = self.classes.len();
        // The groups that are candidates of the rows of the result, the rows
        // left in them, and the candidates of each row among them, in the order
        // of their classes.
        let mut groups = vec![];
        let mut left = vec![];
        let mut local_of = HashMap::new();
        let mut local_candidates = Vec::with_capacity(rows.len());
        for &row in &rows {
            let mut local = candidates[row]
                .iter()
                .map(|&group| {
                    *local_of.entry(group).or_insert_with(|| {
                        groups.push(group);
                        left.push(self.groups[group].2);
                        groups.len() - 1
                    })
                })
                .collect::<Vec<_>>();
            local.sort_by_key(|&candidate| groups[candidate]);
            local_candidates.push(local);
        }
        let class_of = |local: usize| self.groups[groups[local]].0 - start;

        // The rows left of each class, the rows of each class that the OFFSET
        // skips and that the result has, and the first row of the result of
        // each class.
        let mut rows_left = self
            .classes
            .clone()
            .map(|class| self.sizes[class])
            .collect::<Vec<_>>();
        let mut skipped = vec![0; classes];
        let mut returned = vec![0; classes];
        let mut first_row: Vec<Option<usize>> = vec![None; classes];
        let steps = self.skip + rows.len();
        // The class placed at each step, its group for a row of the result,
        // and whether the row is the first row of the result of its class.
        let mut chosen: Vec<(usize, Option<usize>, bool)> = Vec::with_capacity(steps);
        let mut next = vec![0; steps];
        let mut dead_ends = HashSet::new();
        let mut deepest: Option<(usize, ClusterFailure)> = None;
        let record = |deepest: &mut Option<(usize, ClusterFailure)>,
                      depth: usize,
                      failure: ClusterFailure| {
            if deepest.as_ref().is_none_or(|(deepest, _)| depth > *deepest) {
                *deepest = Some((depth, failure));
            }
        };
        let (mut depth, mut entered) = (0, true);
        loop {
            let mut backtrack = false;
            if depth == steps {
                match self.check_cut(&rows_left, &skipped, &first_row) {
                    Ok(()) => return Ok(()),
                    Err(failure) => record(&mut deepest, depth, failure),
                }
                backtrack = true;
            } else {
                // The skipped rows are in the order of their classes.
                let previous = match depth {
                    1.. if depth < self.skip => chosen[depth - 1].0,
                    _ => 0,
                };
                if entered {
                    if *budget == 0 {
                        return Err(ClusterFailure::GaveUp);
                    }
                    *budget -= 1;
                    next[depth] = 0;
                    let state = (depth, previous, rows_left.clone(), left.clone());
                    backtrack = dead_ends.contains(&state);
                }
                if !backtrack {
                    let mut advanced = false;
                    if depth < self.skip {
                        // A row that the OFFSET skips, of a class that may be
                        // at its position.
                        let position = self.first + depth;
                        while previous + next[depth] < classes {
                            let class = previous + next[depth];
                            next[depth] += 1;
                            let (first, last) = self.bounds[start + class];
                            if rows_left[class] == 0
                                || position < first
                                || position > last
                            {
                                continue;
                            }
                            rows_left[class] -= 1;
                            skipped[class] += 1;
                            chosen.push((class, None, false));
                            advanced = true;
                            break;
                        }
                    } else {
                        let row = rows[depth - self.skip];
                        let mut any_left = false;
                        while let Some(&local) =
                            local_candidates[depth - self.skip].get(next[depth])
                        {
                            next[depth] += 1;
                            let class = class_of(local);
                            if left[local] == 0 || rows_left[class] == 0 {
                                continue;
                            }
                            any_left = true;
                            // The earlier rows of later classes, skipped or of
                            // the result, must tie with this row.
                            if let Some(later) = (class + 1..classes).find(|&later| {
                                skipped[later] + returned[later] > 0
                                    && !(self.ties)(start + class, start + later)
                            }) {
                                let failure = match first_row[later] {
                                    Some(earlier) => ClusterFailure::Order {
                                        row,
                                        class: start + class,
                                        earlier,
                                        later: start + later,
                                    },
                                    None => ClusterFailure::Skipped {
                                        row,
                                        class: start + class,
                                        skipped: start + later,
                                    },
                                };
                                record(&mut deepest, depth, failure);
                                continue;
                            }
                            left[local] -= 1;
                            rows_left[class] -= 1;
                            returned[class] += 1;
                            let first = first_row[class].is_none();
                            if first {
                                first_row[class] = Some(row);
                            }
                            chosen.push((class, Some(local), first));
                            advanced = true;
                            break;
                        }
                        if !any_left {
                            record(&mut deepest, depth, ClusterFailure::Values(row));
                        }
                    }
                    if advanced {
                        depth += 1;
                        entered = true;
                        continue;
                    }
                    dead_ends.insert((depth, previous, rows_left.clone(), left.clone()));
                    backtrack = true;
                }
            }
            if backtrack {
                if depth == 0 {
                    let failure = deepest.map(|(_, failure)| failure);
                    return Err(failure.unwrap_or(ClusterFailure::NoOrder));
                }
                depth -= 1;
                let (class, local, first) =
                    chosen.pop().expect("a row is placed at each step");
                rows_left[class] += 1;
                match local {
                    Some(local) => {
                        left[local] += 1;
                        returned[class] -= 1;
                        if first {
                            first_row[class] = None;
                        }
                    }
                    None => skipped[class] -= 1,
                }
                entered = false;
            }
        }
    }

    /// Check the rows that the LIMIT cuts, the rows left after the rows that
    /// the OFFSET skips and the rows of the result (whose first row of each
    /// class is `first_row`): in the order of their classes, they must tie with
    /// the earlier rows of later classes.
    fn check_cut(
        &self,
        rows_left: &[usize],
        skipped: &[usize],
        first_row: &[Option<usize>],
    ) -> std::result::Result<(), ClusterFailure> {
        let start = self.classes.start;
        for (cut, &rows) in rows_left.iter().enumerate() {
            if rows == 0 {
                continue;
            }
            if let Some(later) = (cut + 1..rows_left.len()).find(|&later| {
                (skipped[later] > 0 || first_row[later].is_some())
                    && !(self.ties)(start + cut, start + later)
            }) {
                return Err(match first_row[later] {
                    Some(row) => ClusterFailure::Cut {
                        row,
                        class: start + later,
                        cut: start + cut,
                    },
                    None => ClusterFailure::SkippedAfterCut {
                        skipped: start + later,
                        cut: start + cut,
                    },
                });
            }
        }
        Ok(())
    }
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
    /// A `Float16`, `Float32` or `Float64` value, and its precision.
    Float(f64, Precision),
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
                let value = array.as_primitive::<Float16Type>().value(row);
                Self::Float(value.to_f64(), Precision::Half)
            }
            DataType::Float32 => {
                let value = array.as_primitive::<Float32Type>().value(row);
                Self::Float(f64::from(value), Precision::Single)
            }
            DataType::Float64 => Self::Float(
                array.as_primitive::<Float64Type>().value(row),
                Precision::Double,
            ),
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
            (Self::Float(a, _), Self::Float(b, _)) => floats_equal(*a, *b, tolerance),
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
            (Self::Null, Self::Null) | (Self::Float(..), Self::Float(..)) => true,
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
            Self::Null | Self::Float(..) => {}
            Self::Scalar(scalar) => scalar.hash(state),
            Self::List(values) | Self::Struct(values) => Shape(values).hash(state),
        }
    }

    fn has_floats(&self) -> bool {
        match self {
            Self::Float(..) => true,
            Self::List(values) | Self::Struct(values) => {
                values.iter().any(Self::has_floats)
            }
            Self::Null | Self::Scalar(_) => false,
        }
    }

    /// Append the floating-point values of `self` to `floats`, with their
    /// precisions.
    fn collect_floats_of_precision(&self, floats: &mut Vec<(f64, Precision)>) {
        match self {
            Self::Float(value, precision) => floats.push((*value, *precision)),
            Self::List(values) | Self::Struct(values) => {
                for value in values {
                    value.collect_floats_of_precision(floats);
                }
            }
            Self::Null | Self::Scalar(_) => {}
        }
    }

    /// Append the floating-point values of `self` to `floats`.
    fn collect_floats(&self, floats: &mut Vec<f64>) {
        match self {
            Self::Float(value, _) => floats.push(*value),
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
            Self::Float(value, _) => write!(f, "{value:?}"),
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

/// The [`Shape`] of values, and some of their floating-point values, as bits.
type ShapeAndFloats<'a> = (Shape<'a>, Vec<u64>);

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
// The rows of `Shape`s hold `ScalarValue`s, whose Arrow buffers have a
// `Mutex` for memory accounting with the `arrow-buffer/pool` feature (enabled
// by `--all-features`): it affects neither their equality nor their hashing.
// Not `expect`: the lint only fires with that feature.
#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
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
            sort_options: vec![],
            skip,
            fetch,
            uncomputed: vec![],
            tolerance: f64::INFINITY,
        })
    }

    /// The order of a query that selects the values `v` of `(k, v)` rows,
    /// ordered by `k`: their sorted rows are `sorted` (without the LIMIT and
    /// OFFSET of the query, which skips `skip` of them and returns at most
    /// `fetch`).
    fn sorted_by_k(
        sorted: &[(i32, &str)],
        skip: usize,
        fetch: Option<usize>,
    ) -> ResultOrder {
        hidden_key_order(&kv_batch(sorted), skip, fetch)
    }

    /// The order of a query that selects the second column of the rows
    /// `sorted`, ordered by their first column (`m.k`), which it skips `skip`
    /// of and returns at most `fetch` of.
    fn hidden_key_order(
        sorted: &RecordBatch,
        skip: usize,
        fetch: Option<usize>,
    ) -> ResultOrder {
        let schema = sorted.schema();
        let reference = batch_of(vec![
            (
                schema.field(1).name().as_str(),
                Arc::clone(sorted.column(1)),
            ),
            ("key", Arc::clone(sorted.column(0))),
        ]);
        ResultOrder::Sorted(SortedReference {
            batches: vec![reference],
            columns: 1,
            keys: vec!["m.k".to_string()],
            sort_options: vec![],
            skip,
            fetch,
            uncomputed: vec![],
            tolerance: f64::INFINITY,
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
    fn floats_may_vary_within_a_tolerance_larger_than_the_distance_to_their_neighbors() {
        let t = 2e-9;
        for (value, precision, tolerance, varies) in [
            (1.0, Precision::Double, t, true),
            (-3.5e300, Precision::Double, t, true),
            (1e-310, Precision::Double, t, true),
            (1.0, Precision::Double, 0.0, false),
            (0.0, Precision::Double, t, false),
            (-0.0, Precision::Double, t, false),
            (0.0, Precision::Double, 1.0, true),
            (f64::INFINITY, Precision::Double, t, false),
            (f64::NEG_INFINITY, Precision::Double, 1.0, false),
            (f64::NAN, Precision::Double, 1.0, false),
            (1.0, Precision::Single, t, false),
            (1.0, Precision::Single, 2e-7, true),
            (-1.0, Precision::Single, 2e-7, true),
            (0.0, Precision::Single, 2e-7, false),
            (1.0, Precision::Half, 2e-7, false),
            (1.0, Precision::Half, 2e-3, true),
            (-1.0, Precision::Half, 2e-3, true),
            (0.0, Precision::Half, 2e-3, false),
        ] {
            assert_eq!(
                may_vary(value, precision, tolerance),
                varies,
                "{value} {precision:?} {tolerance}"
            );
        }
        // The neighbors are values of the precision.
        let eps = |bits: i32| 2f64.powi(-bits);
        assert_eq!(
            Precision::Double.neighbors(1.0),
            [1.0 + eps(52), 1.0 - eps(53)]
        );
        assert_eq!(
            Precision::Single.neighbors(1.0),
            [1.0 + eps(23), 1.0 - eps(24)]
        );
        assert_eq!(
            Precision::Half.neighbors(1.0),
            [1.0 + eps(10), 1.0 - eps(11)]
        );
        assert_eq!(
            Precision::Half.neighbors(-1.0),
            [-1.0 + eps(11), -1.0 - eps(10)]
        );
        assert_eq!(Precision::Half.neighbors(0.0), [eps(24), -eps(24)]);
        assert_eq!(Precision::Half.neighbors(-0.0), [eps(24), -eps(24)]);
    }

    #[test]
    fn intervals_of_floats_are_the_values_equal_within_the_tolerance() {
        // The values around `value` of the precision, `steps` neighbors away.
        let around = |value: f64, precision: Precision, steps: usize| {
            let (mut up, mut down) = (vec![value], vec![]);
            let (mut high, mut low) = (value, value);
            for _ in 0..steps {
                high = precision.neighbors(high)[0];
                low = precision.neighbors(low)[1];
                up.push(high);
                down.push(low);
            }
            down.into_iter().rev().chain(up).collect::<Vec<_>>()
        };
        for (precision, values, tolerances) in [
            (
                Precision::Double,
                vec![1.0, -1.0, 1.0f64.next_down(), 3.5e300, -1e-310],
                vec![0.0, 8e-17, 1.5e-16, 3e-16, 1e-15],
            ),
            (Precision::Single, vec![1.0, -2.5], vec![4e-8, 6e-8, 2e-7]),
            (Precision::Half, vec![1.0, -0.5], vec![4e-4, 1e-3, 3e-3]),
        ] {
            for (value, tolerance) in values.iter().copied().cartesian_product(tolerances)
            {
                let (low, high) = interval(value, precision, tolerance);
                // The interval is exactly the values that are equal to `value`.
                let equal = around(value, precision, 40)
                    .into_iter()
                    .filter(|&other| floats_equal(value, other, tolerance))
                    .collect::<Vec<_>>();
                assert_eq!(
                    (low, high),
                    (equal[0], equal[equal.len() - 1]),
                    "{value} {precision:?} {tolerance}"
                );
                // Values that cross, meet, or are apart, by brute force.
                for other in around(value, precision, 12) {
                    let others = around(other, precision, 40)
                        .into_iter()
                        .filter(|&x| floats_equal(other, x, tolerance))
                        .collect::<Vec<_>>();
                    let (smaller, larger) = if value <= other {
                        (&equal, &others)
                    } else {
                        (&others, &equal)
                    };
                    let expected = if smaller.iter().any(|a| larger.iter().any(|b| b < a))
                        || (value == other && equal.len() > 1)
                    {
                        Meeting::Cross
                    } else if smaller.iter().any(|a| larger.contains(a)) {
                        Meeting::Meet
                    } else {
                        Meeting::Apart
                    };
                    assert_eq!(
                        meeting((value, precision), (other, precision), tolerance),
                        expected,
                        "{value} {other} {precision:?} {tolerance}"
                    );
                }
            }
        }
        // Zeros of both signs are one value, which cannot vary below a
        // tolerance of 1.
        assert_eq!(interval(-0.0, Precision::Double, 1e-9), (0.0, 0.0));
        let zeros = ((0.0, Precision::Double), (-0.0, Precision::Double));
        assert_eq!(meeting(zeros.0, zeros.1, 1e-9), Meeting::Meet);
        let nan = (f64::NAN, Precision::Double);
        assert_eq!(meeting(nan, nan, 1e-9), Meeting::Meet);
        assert_eq!(meeting(nan, (1.0, Precision::Double), 1.0), Meeting::Apart);
        let infinity = (f64::INFINITY, Precision::Double);
        assert_eq!(meeting(infinity, infinity, 1e-9), Meeting::Meet);
        let max = (f64::MAX, Precision::Double);
        assert_eq!(meeting(infinity, max, 1e-9), Meeting::Apart);
    }

    #[test]
    fn precedence_follows_the_sort_options_of_the_keys() {
        let options = |descending, nulls_first| SortOptions {
            descending,
            nulls_first,
        };
        let int = |value: i32| Value::Scalar(ScalarValue::Int32(Some(value)));
        let float = |value: f64| Value::Float(value, Precision::Double);
        let (ascending, descending) = (options(false, false), options(true, true));
        let t = 1e-9;
        for (later, earlier, sort, expected) in [
            (int(1), int(2), ascending, Precedence::Before),
            (int(1), int(2), descending, Precedence::After),
            (int(2), int(2), ascending, Precedence::Equal),
            (Value::Null, int(2), ascending, Precedence::After),
            (
                Value::Null,
                int(2),
                options(false, true),
                Precedence::Before,
            ),
            (int(2), Value::Null, descending, Precedence::After),
            (Value::Null, Value::Null, descending, Precedence::Equal),
            // Close floats may be in either order, others not.
            (float(1.0 + 1e-9), float(1.0), ascending, Precedence::Before),
            (float(1.5), float(1.0), ascending, Precedence::After),
            (float(1.0), float(1.5), descending, Precedence::After),
            (float(0.0), float(-0.0), ascending, Precedence::Equal),
            (float(f64::NAN), float(1.0), ascending, Precedence::After),
            // Lists are compared element by element, then by their lengths.
            (
                Value::List(vec![int(1), int(3)]),
                Value::List(vec![int(1), int(2)]),
                ascending,
                Precedence::After,
            ),
            (
                Value::List(vec![int(1)]),
                Value::List(vec![int(1), int(2)]),
                ascending,
                Precedence::Before,
            ),
            (
                Value::List(vec![int(1)]),
                Value::List(vec![int(1), int(2)]),
                descending,
                Precedence::After,
            ),
        ] {
            assert_eq!(
                precedence(&later, &earlier, sort, t),
                expected,
                "{later} {earlier} {sort:?}"
            );
        }
        // The first key that is not equal decides; missing sort options are
        // ascending with nulls last.
        let row =
            |values: &[i32]| values.iter().map(|&value| int(value)).collect::<Row>();
        assert!(may_precede(&row(&[1, 1]), &row(&[1, 2]), &[], t));
        assert!(!may_precede(&row(&[1, 3]), &row(&[1, 2]), &[], t));
        assert!(may_precede(
            &row(&[1, 3]),
            &row(&[1, 2]),
            &[ascending, descending],
            t
        ));
        assert!(may_precede(&row(&[1, 2]), &row(&[1, 2]), &[], t));
    }

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
                "row 0 of the pipeline engine is out of order: the sorted rows of \
                 DataFusion with its values may not be at its position (rows tie if \
                 their ORDER BY keys (k) are equal up to their first floating-point \
                 value that may vary within the tolerance (not a zero, an infinity or \
                 NaN), and either of these values may be the smaller one within the \
                 tolerance, whatever their later keys (if they may only be equal, their \
                 later keys decide))"
            ),
            "{err}"
        );
        assert!(err.contains("row 0: (2, \"b\"), keys: (2)"), "{err}");
        assert!(
            err.contains("sorted row 0 of DataFusion: (1, \"a\"), keys: (1)"),
            "{err}"
        );
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
            err.starts_with("row 0 of the pipeline engine is out of order"),
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
        let err = same_results(ties(), moved, &by_k).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine is out of order"),
            "{err}"
        );
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
            err.starts_with(
                "row 1 of the pipeline engine differs: it is not among the sorted rows \
                 of DataFusion that the query may return at its position (with their \
                 multiplicities)"
            ),
            "{err}"
        );
        assert!(err.contains("row 1: (1, \"x\")"), "{err}");
        assert!(
            err.contains("sorted row 1 of DataFusion: (1, \"b\")"),
            "{err}"
        );
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
        for (actual, row) in [
            // A row that ties with the last row twice.
            ([(1, "a"), (2, "b"), (2, "b")], "row 2: (2, \"b\")"),
            // Rows that are not sorted rows.
            ([(1, "a"), (2, "x"), (2, "b")], "row 1: (2, \"x\")"),
            ([(1, "a"), (2, "b"), (3, "e")], "row 2: (3, \"e\")"),
        ] {
            let err =
                same_results(expected(), vec![kv_batch(&actual)], &limit_3).unwrap_err();
            let position = &row[..5];
            assert!(
                err.starts_with(&format!("{position} of the pipeline engine differs")),
                "{err}"
            );
            assert!(err.contains(row), "{err}");
        }
        // A sorted row that does not tie with the cut rows.
        let sorted = [kv_batch(&[(1, "a"), (2, "b"), (3, "c")])];
        let limit_2 = sorted_by(&sorted, &["k"], 0, Some(2));
        let after_the_cut = vec![kv_batch(&[(1, "a"), (3, "c")])];
        let err = same_results(two(), after_the_cut, &limit_2).unwrap_err();
        assert!(
            err.starts_with(
                "row 1 of the pipeline engine is out of order: the sorted rows of \
                 DataFusion with its values may not be at its position"
            ),
            "{err}"
        );
        assert!(err.contains("row 1: (3, \"c\"), keys: (3)"), "{err}");
        assert!(
            err.contains("sorted row 1 of DataFusion: (2, \"b\"), keys: (2)"),
            "{err}"
        );
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
        // A sorted row after the rows that the query returns, which does not
        // tie with them.
        let after_the_cut = vec![kv_batch(&[(1, "b"), (3, "d")])];
        let err = same_results(expected(), after_the_cut, &offset_1).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine is out of order"),
            "{err}"
        );
        // A sorted row that the OFFSET skips, which does not tie with them.
        let ascending = [kv_batch(&[(1, "a"), (2, "b"), (3, "c")])];
        let offset_2 = sorted_by(&ascending, &["k"], 2, None);
        let skipped = vec![kv_batch(&[(2, "b")])];
        let err =
            same_results(vec![kv_batch(&[(3, "c")])], skipped, &offset_2).unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine is out of order"),
            "{err}"
        );
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
        let by_k = sorted_by_k(&rows, 0, None);
        same_results(vec![v_batch(&rows)], vec![v_batch(&rows)], &by_k).unwrap();
        let reversed = [(3, "c"), (2, "b"), (1, "a")];
        let err = same_results(vec![v_batch(&rows)], vec![v_batch(&reversed)], &by_k)
            .unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine is out of order"),
            "{err}"
        );
        assert!(err.contains("ORDER BY keys (m.k)"), "{err}");
        assert!(err.contains("row 0: (\"c\"), keys: (3)"), "{err}");

        // Rows that tie on the key, which is not selected.
        let ties = [(1, "a"), (1, "b"), (2, "c")];
        let by_k = sorted_by_k(&ties, 0, None);
        let swapped = [(1, "b"), (1, "a"), (2, "c")];
        same_results(vec![v_batch(&ties)], vec![v_batch(&swapped)], &by_k).unwrap();
        let moved = [(1, "a"), (2, "c"), (1, "b")];
        let err =
            same_results(vec![v_batch(&ties)], vec![v_batch(&moved)], &by_k).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine is out of order"),
            "{err}"
        );

        // ...and that a LIMIT cuts.
        let sorted = [(1, "a"), (2, "b"), (2, "c"), (2, "d")];
        let limit_3 = sorted_by_k(&sorted, 0, Some(3));
        let expected = || vec![v_batch(&sorted[..3])];
        let others = [(1, "a"), (2, "d"), (2, "b")];
        same_results(expected(), vec![v_batch(&others)], &limit_3).unwrap();
        for actual in [
            [(1, "a"), (2, "b"), (2, "b")],
            [(1, "a"), (2, "x"), (2, "b")],
        ] {
            let err =
                same_results(expected(), vec![v_batch(&actual)], &limit_3).unwrap_err();
            assert!(err.contains("of the pipeline engine differs"), "{err}");
        }
    }

    #[test]
    fn same_results_assign_the_rows_to_the_sorted_rows_at_their_positions() {
        // The values of the rows of `k = 1` and `k = 3` are equal: the rows that
        // the query returns are those at its positions.
        let sorted = [(1, "a"), (2, "b"), (3, "a")];
        for (skip, fetch, rows) in [
            (2, None, &[(3, "a")][..]),
            (2, Some(1), &[(3, "a")]),
            (1, None, &[(2, "b"), (3, "a")]),
            (0, Some(1), &[(1, "a")]),
            (0, None, &sorted),
        ] {
            let order = sorted_by_k(&sorted, skip, fetch);
            same_results(vec![v_batch(rows)], vec![v_batch(rows)], &order)
                .unwrap_or_else(|e| panic!("{skip} {fetch:?}: {e}"));
        }
        let offset_2 = sorted_by_k(&sorted, 2, None);
        let other = vec![v_batch(&[(2, "b")])];
        let err = same_results(vec![v_batch(&[(3, "a")])], other, &offset_2).unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine is out of order"),
            "{err}"
        );
    }

    #[test]
    fn same_results_match_close_floats_across_classes_that_tie() {
        // The query selects `y` of rows ordered by `x`, whose keys tie: `1 -
        // 0.9e-9` can only be the `y` of the first row, and `1 + 0.9e-9` either.
        let sorted = xy_batch(&[(1.0, 1.0), (1.0 + 1.5e-9, 1.0 + 1.8e-9)]);
        let order = hidden_key_order(&sorted, 0, None);
        let y = |values: &[f64]| {
            let y = Float64Array::from(values.to_vec());
            vec![batch_of(vec![("y", Arc::new(y) as ArrayRef)])]
        };
        let expected = || y(&[1.0, 1.0 + 1.8e-9]);
        same_results(expected(), y(&[1.0 + 0.9e-9, 1.0 - 0.9e-9]), &order).unwrap();
        same_results(expected(), y(&[1.0 - 0.9e-9, 1.0 + 0.9e-9]), &order).unwrap();
        let err = same_results(expected(), y(&[1.0 - 0.9e-9, 1.0 - 0.9e-9]), &order)
            .unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine differs"),
            "{err}"
        );
    }

    /// `(x, y)` rows of the keys `x = 1 + t * [0, 1, 2, 3]`, with `t = 0.9e-9`,
    /// which tie with the next two keys, and the payloads `y = 1 + t * y`,
    /// which are equal within the tolerance to the payloads one `t` apart.
    fn chained_rows(y: [i32; 4]) -> RecordBatch {
        let t = 0.9e-9;
        let rows = (0..4)
            .map(|i| (1.0 + t * f64::from(i), 1.0 + t * f64::from(y[i as usize])))
            .collect::<Vec<_>>();
        xy_batch(&rows)
    }

    /// The payloads `y` of `chained_rows(y)`.
    fn chained_payloads(y: [i32; 4]) -> Vec<RecordBatch> {
        let y = y.map(|y| 1.0 + 0.9e-9 * f64::from(y));
        vec![batch_of(vec![(
            "y",
            Arc::new(Float64Array::from(y.to_vec())) as ArrayRef,
        )])]
    }

    #[test]
    fn same_results_search_the_orders_of_rows_whose_ties_are_not_transitive() {
        // Keys 1.5e-9 apart, which tie with the next one but not with the one
        // after, and rows of the same value: under OFFSET 1, the query returns
        // the rows of the last two keys, as the OFFSET may skip the first row,
        // but not the last, which does not tie with the first.
        let sorted = sv_batch(&[(1.0, "a"), (1.0 + 1.5e-9, "a"), (1.0 + 3e-9, "a")]);
        let a = |rows: usize| vec![v_batch(&vec![(0, "a"); rows])];
        for (skip, fetch, rows) in
            [(1, None, 2), (1, Some(1), 1), (2, None, 1), (0, Some(2), 2)]
        {
            let order = hidden_key_order(&sorted, skip, fetch);
            same_results(a(rows), a(rows), &order)
                .unwrap_or_else(|e| panic!("{skip} {fetch:?}: {e}"));
        }
        // Rows of other values: under OFFSET 1, `b, c`, `c, b` and `a, c` are
        // the last rows of orders in which rows are only out of their sorted
        // order relative to rows that they tie with, but `c, a`, `a, b` and
        // `b, a` are not.
        let sorted = sv_batch(&[(1.0, "a"), (1.0 + 1.5e-9, "b"), (1.0 + 3e-9, "c")]);
        let offset_1 = hidden_key_order(&sorted, 1, None);
        let values = |values: [&str; 2]| vec![v_batch(&values.map(|value| (0, value)))];
        for valid in [["b", "c"], ["c", "b"], ["a", "c"]] {
            same_results(values(["b", "c"]), values(valid), &offset_1)
                .unwrap_or_else(|e| panic!("{valid:?}: {e}"));
        }
        for invalid in [["c", "a"], ["a", "b"], ["b", "a"]] {
            let result = same_results(values(["b", "c"]), values(invalid), &offset_1);
            assert!(result.is_err(), "{invalid:?}");
        }

        // Payloads that are equal within the tolerance to the payloads of
        // several rows: of the assignments of `[2, 3, 6, 1]` to the rows of
        // `[5, 3, 1, 2]` at their positions, `[1, 3, 0, 2]` and `[3, 1, 0, 2]`
        // are out of order, but `[2, 1, 0, 3]` is in order.
        let order = hidden_key_order(&chained_rows([5, 3, 1, 2]), 0, None);
        let expected = || chained_payloads([5, 3, 1, 2]);
        same_results(expected(), chained_payloads([2, 3, 6, 1]), &order).unwrap();
        // The only assignment of `[0, 1, 5, 3]`, `[2, 3, 0, 1]`, is out of
        // order.
        let err =
            same_results(expected(), chained_payloads([0, 1, 5, 3]), &order).unwrap_err();
        assert!(
            err.starts_with(
                "row 2 of the pipeline engine is out of order: the query orders it \
                 before row 1, and their keys do not tie"
            ),
            "{err}"
        );
    }

    /// A batch of `(id BIGINT, k1 DOUBLE, k2 DOUBLE)` rows.
    fn id_keys_batch(rows: &[(i64, f64, f64)]) -> RecordBatch {
        batch_of(vec![
            (
                "id",
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "k1",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.1))),
            ),
            (
                "k2",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.2))),
            ),
        ])
    }

    /// The order of a query that selects the `id`s of the rows `sorted`
    /// (`id_keys_batch`), ordered by `k1, k2`, of which it skips `skip` rows
    /// and returns at most `fetch`.
    fn id_order(
        sorted: &[(i64, f64, f64)],
        skip: usize,
        fetch: Option<usize>,
    ) -> ResultOrder {
        ResultOrder::Sorted(SortedReference {
            batches: vec![id_keys_batch(sorted)],
            columns: 1,
            keys: vec!["m.k1".to_string(), "m.k2".to_string()],
            sort_options: vec![],
            skip,
            fetch,
            uncomputed: vec![],
            tolerance: f64::INFINITY,
        })
    }

    /// A result of `id`s.
    fn ids(ids: &[i64]) -> Vec<RecordBatch> {
        let ids = Int64Array::from(ids.to_vec());
        vec![batch_of(vec![("id", Arc::new(ids) as ArrayRef)])]
    }

    #[test]
    fn same_results_search_the_rows_that_the_offset_skips() {
        let t: f64 = 1e-9;
        // Rows `(id, k1, k2)` sorted on `(k1, k2)`, whose ids are their positions.
        // The first floating-point key `k1` decides which rows tie, whatever `k2`,
        // which is in the opposite order (but where `k1` is equal). The ties are
        // not transitive: in `chain`, each row ties with the next one, but not
        // with the one after.
        let chain = [1.0, 1.0 + 1.5 * t, 1.0 + 3.0 * t, 1.0 + 4.5 * t];
        // Row 0 ties with row 1, which ties with rows 2 and 3, which tie.
        let fork = [1.0, 1.0 + t, 1.0 + 2.5 * t, 1.0 + 2.8 * t];
        // Rows 0 and 1 are equal on `k1` (and in the order of `k2`), and tie with
        // row 2, which ties with row 3.
        let pair = [1.0, 1.0, 1.0 + 1.5 * t, 1.0 + 3.0 * t];
        let all = [1.0, 1.0 + 0.5 * t, 1.0 + t, 1.0 + 1.5 * t];
        for k1 in [&chain[..3], &chain, &fork, &pair, &all] {
            let n = k1.len();
            let k2 = |id: usize| {
                if id > 0 && k1[id] == k1[id - 1] {
                    1.0
                } else {
                    -(id as f64)
                }
            };
            let sorted = (0..n)
                .map(|id| (id as i64, k1[id], k2(id)))
                .collect::<Vec<_>>();
            let tie = |a: usize, b: usize| {
                let (a, b) = (k1[a], k1[b]);
                (a - b).abs() <= 2.0 * t * a.abs().max(b.abs())
            };
            // The orders of all rows in which rows are only out of their sorted
            // order relative to rows that they tie with.
            let orders = (0..n)
                .permutations(n)
                .filter(|order| {
                    (0..n).all(|i| {
                        (i + 1..n).all(|j| order[i] < order[j] || tie(order[i], order[j]))
                    })
                })
                .collect::<Vec<_>>();
            for (skip, fetch) in [
                (0, None),
                (0, Some(1)),
                (0, Some(2)),
                (1, None),
                (1, Some(1)),
                (1, Some(2)),
                (2, None),
                (2, Some(1)),
            ] {
                let window = skip.min(n)..fetch.map_or(n, |fetch| (skip + fetch).min(n));
                let valid = orders
                    .iter()
                    .map(|order| order[window.clone()].to_vec())
                    .collect::<HashSet<_>>();
                let order = id_order(&sorted, skip, fetch);
                let expected = window.clone().map(|id| id as i64).collect::<Vec<_>>();
                // Every result of distinct rows.
                for result in (0..n).permutations(window.len()) {
                    let ids_of_result =
                        result.iter().map(|&id| id as i64).collect::<Vec<_>>();
                    let comparison =
                        same_results(ids(&expected), ids(&ids_of_result), &order);
                    assert_eq!(
                        comparison.is_ok(),
                        valid.contains(&result),
                        "{k1:?} OFFSET {skip} LIMIT {fetch:?} {result:?}: {comparison:?}"
                    );
                }
            }
        }

        // With the keys of an earlier review, all rows tie on `k1`: `[0]` is the
        // second row of the order `[2, 0, 1]`.
        let sorted = [
            (0, 1.0, 1.0),
            (1, 1.0 + 0.1 * t, 1.0 + 3.0 * t),
            (2, 1.0 + 0.2 * t, 1.0 + 1.5 * t),
        ];
        same_results(ids(&[1]), ids(&[0]), &id_order(&sorted, 1, Some(1))).unwrap();
    }

    /// A batch of `(id BIGINT, k DOUBLE, s VARCHAR)` rows.
    fn id_k_s_batch(rows: &[(i64, f64, &str)]) -> RecordBatch {
        batch_of(vec![
            (
                "id",
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "k",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.1))),
            ),
            (
                "s",
                Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.2))),
            ),
        ])
    }

    #[test]
    fn same_results_order_rows_lexicographically_within_the_tolerance() {
        let rows = |rows: &[(i64, f64, &str)]| vec![id_k_s_batch(rows)];
        let close = 1.0 + 0.5e-9;
        // The values of `k` of the pipeline engine may be the values of
        // DataFusion swapped (within the tolerance), which reverses the order of
        // the rows on `(k, s)`, whatever `s`: with and without LIMIT 1.
        let sorted = || rows(&[(0, 1.0, "a"), (1, close, "b")]);
        let swapped = || rows(&[(1, 1.0, "b"), (0, close, "a")]);
        let by_k_s = sorted_by(&sorted(), &["k", "s"], 0, None);
        same_results(sorted(), swapped(), &by_k_s).unwrap();
        let first_by_k_s = sorted_by(&sorted(), &["k", "s"], 0, Some(1));
        let first = rows(&[(0, 1.0, "a")]);
        same_results(first.clone(), rows(&[(1, 1.0, "b")]), &first_by_k_s).unwrap();
        // Also if the values of `k` are equal.
        let equal = || rows(&[(0, 1.0, "a"), (1, 1.0, "b")]);
        let by_k_s = sorted_by(&equal(), &["k", "s"], 0, None);
        same_results(equal(), rows(&[(1, 1.0, "b"), (0, 1.0, "a")]), &by_k_s).unwrap();
        // On `(s, k)`, `s` decides, unless it is equal.
        let by_s_k = sorted_by(&sorted(), &["s", "k"], 0, None);
        let err = same_results(sorted(), swapped(), &by_s_k).unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine is out of order"),
            "{err}"
        );
        assert!(err.contains("keys: (\"b\", 1.0000000005)"), "{err}");
        let same_s = || rows(&[(0, 1.0, "a"), (1, close, "a")]);
        let by_s_k = sorted_by(&same_s(), &["s", "k"], 0, Some(1));
        same_results(rows(&[(0, 1.0, "a")]), rows(&[(1, 1.0, "a")]), &by_s_k).unwrap();
        // Values of `k` that are not close decide.
        let far = || rows(&[(0, 1.0, "a"), (1, 1.5, "b")]);
        let by_k_s = sorted_by(&far(), &["k", "s"], 0, None);
        let err = same_results(far(), rows(&[(1, 1.5, "b"), (0, 1.0, "a")]), &by_k_s)
            .unwrap_err();
        assert!(
            err.starts_with(
                "row 0 of the pipeline engine is out of order: the sorted rows of \
                 DataFusion with its values may not be at its position (rows tie if \
                 their ORDER BY keys (k, s) are equal up to their first floating-point \
                 value that may vary"
            ),
            "{err}"
        );
        // The keys of the rows are shown up to their first floating-point value.
        assert!(
            err.contains("row 0: (1, 1.5, \"b\"), keys: (1.5, …)"),
            "{err}"
        );
        let first_by_k_s = sorted_by(&far(), &["k", "s"], 0, Some(1));
        let err = same_results(first, rows(&[(1, 1.5, "b")]), &first_by_k_s).unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine is out of order"),
            "{err}"
        );

        // Lists are compared element by element.
        let lists = |lists: &[(i64, [f64; 2])]| {
            let ids = Int64Array::from_iter_values(lists.iter().map(|list| list.0));
            let values = lists
                .iter()
                .map(|list| Some(list.1.iter().copied().map(Some).collect::<Vec<_>>()));
            vec![batch_of(vec![
                ("id", Arc::new(ids) as ArrayRef),
                (
                    "l",
                    Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(values)),
                ),
            ])]
        };
        let sorted = lists(&[(0, [1.0, 2.0]), (1, [close, 1.0])]);
        let by_l = sorted_by(&sorted, &["l"], 0, None);
        let reversed = lists(&[(1, [1.0, 1.0]), (0, [close, 2.0])]);
        same_results(sorted.clone(), reversed, &by_l).unwrap();
        let sorted = lists(&[(0, [1.0, 2.0]), (1, [1.5, 1.0])]);
        let by_l = sorted_by(&sorted, &["l"], 0, None);
        let reversed = lists(&[(1, [1.5, 1.0]), (0, [1.0, 2.0])]);
        let err = same_results(sorted, reversed, &by_l).unwrap_err();
        assert!(err.contains("keys: ([1.5], …)"), "{err}");
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
        assert!(
            err.starts_with("row 2 of the pipeline engine differs"),
            "{err}"
        );
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
        let keyed = |rows: &[(f64, &str)]| vec![sv_batch(rows)];
        let expected = keyed(&[(1.0, "a"), (close, "b"), (2.0, "c")]);
        let by_s = sorted_by(&expected, &["s"], 0, None);
        let swapped = keyed(&[(1.0, "b"), (close, "a"), (2.0, "c")]);
        same_results(expected.clone(), swapped, &by_s).unwrap();
        let other_order = keyed(&[(2.0, "c"), (1.0, "b"), (close, "a")]);
        assert!(same_results(expected, other_order, &by_s).is_err());
    }

    /// A batch of `(s DOUBLE, v VARCHAR)` rows.
    fn sv_batch(rows: &[(f64, &str)]) -> RecordBatch {
        batch_of(vec![
            (
                "s",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "v",
                Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.1))),
            ),
        ])
    }

    /// The keys of `CLOSE_ROWS`: each ties with the next one (they are equal
    /// within twice the tolerance), but not with the one after.
    const CLOSE_KEYS: [f64; 4] = [1.0, 1.0 + 1.5e-9, 1.0 + 3e-9, 1.0 + 4.5e-9];

    /// The rows `order` of `CLOSE_KEYS` with the names `a`, `b`, `c`, `d`.
    fn close_rows(order: &[usize]) -> Vec<RecordBatch> {
        let names = ["a", "b", "c", "d"];
        let rows = order
            .iter()
            .map(|&i| (CLOSE_KEYS[i], names[i]))
            .collect::<Vec<_>>();
        vec![sv_batch(&rows)]
    }

    #[test]
    fn same_results_do_not_chain_the_ties_of_close_floats() {
        let sorted = close_rows(&[0, 1, 2, 3]);
        let by_s = sorted_by(&sorted, &["s"], 0, None);
        // Rows that tie may be out of their sorted order...
        for order in [[1, 0, 2, 3], [1, 0, 3, 2], [0, 2, 1, 3], [0, 1, 3, 2]] {
            same_results(sorted.clone(), close_rows(&order), &by_s)
                .unwrap_or_else(|e| panic!("{order:?}: {e}"));
        }
        // ...but not relative to rows that they do not tie with, although
        // each ties with the next.
        for (order, row) in [
            ([3, 2, 1, 0], 0),
            ([2, 1, 0, 3], 0),
            ([0, 3, 1, 2], 1),
            ([1, 2, 0, 3], 2),
        ] {
            let err =
                same_results(sorted.clone(), close_rows(&order), &by_s).unwrap_err();
            assert!(
                err.starts_with(&format!(
                    "row {row} of the pipeline engine is out of order"
                )),
                "{order:?}: {err}"
            );
        }

        // Under LIMIT 1, the first row may be one that ties with the first
        // sorted row, but not another.
        let limit_1 = sorted_by(&sorted, &["s"], 0, Some(1));
        same_results(close_rows(&[0]), close_rows(&[1]), &limit_1).unwrap();
        for row in [2, 3] {
            let err =
                same_results(close_rows(&[0]), close_rows(&[row]), &limit_1).unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{row}: {err}"
            );
            // DataFusion's result is compared with its sorted rows too.
            assert!(
                same_results(close_rows(&[row]), close_rows(&[0]), &limit_1).is_err()
            );
        }
        // Under LIMIT 2, the last row may be one that ties with all rows
        // before the LIMIT.
        let limit_2 = sorted_by(&sorted, &["s"], 0, Some(2));
        same_results(close_rows(&[0, 1]), close_rows(&[1, 0]), &limit_2).unwrap();
        same_results(close_rows(&[0, 1]), close_rows(&[0, 2]), &limit_2).unwrap();
        // Rows `b` and `c` may be first and second, but the LIMIT would then cut
        // row `a`, which does not tie with `c`.
        let err =
            same_results(close_rows(&[0, 1]), close_rows(&[1, 2]), &limit_2).unwrap_err();
        assert!(
            err.starts_with(
                "row 1 of the pipeline engine is not among the rows that the query \
                 returns: the LIMIT cuts sorted rows of DataFusion that the query orders \
                 before it, and whose keys do not tie with its keys"
            ),
            "{err}"
        );
        // Under OFFSET 3, the last row.
        let offset_3 = sorted_by(&sorted, &["s"], 3, None);
        same_results(close_rows(&[3]), close_rows(&[2]), &offset_3).unwrap();
        for row in [0, 1] {
            let err = same_results(close_rows(&[3]), close_rows(&[row]), &offset_3)
                .unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{row}: {err}"
            );
        }
        // Under OFFSET 1, rows `a`, `b` and `c` may each be at the positions of
        // `b`, `c` and `d`, but not all of them: the OFFSET may only skip row
        // `a` or row `b`, as row `d`, which does not tie with them, may not come
        // first.
        let offset_1 = sorted_by(&sorted, &["s"], 1, None);
        same_results(close_rows(&[1, 2, 3]), close_rows(&[0, 2, 3]), &offset_1).unwrap();
        let err = same_results(close_rows(&[1, 2, 3]), close_rows(&[0, 1, 2]), &offset_1)
            .unwrap_err();
        assert!(
            err.starts_with(
                "row 1 of the pipeline engine differs: it is not among the sorted rows \
                 of DataFusion that the query may return at its position (with their \
                 multiplicities, and in an order in which rows are only out of their \
                 sorted order relative to rows that they tie with)"
            ),
            "{err}"
        );
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

        // The batches of the pipeline engine have exactly the output schema,
        // even if the batches of DataFusion do not.
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
        let err = compare_results(&expected, &actual, UNORDERED, &options).unwrap_err();
        assert!(
            err.contains(
                "the schema of a batch of the pipeline engine is not the output"
            ),
            "{err}"
        );
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
            alternatives: vec![],
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
            // A volatile key, and the keys after it, cannot be compared...
            (
                "SELECT id FROM t1 ORDER BY a, random(), id",
                Some(("t1.a", 1, 0, None, 5000)),
            ),
            // ...also through the columns of a subquery.
            (
                "SELECT id FROM (SELECT id, random() AS r FROM t1) s ORDER BY r LIMIT 3",
                Some(("", 1, 0, Some(3), 5000)),
            ),
            (
                "SELECT id FROM (SELECT id, a, random() + 1 AS r FROM t1) s ORDER BY a, r, id",
                Some(("s.a", 1, 0, None, 5000)),
            ),
            (
                "SELECT id FROM (SELECT id, random() AS r FROM t1) s ORDER BY id DESC",
                Some(("s.id", 1, 0, None, 5000)),
            ),
            (
                "SELECT id FROM (SELECT id, random() AS r FROM t1) ORDER BY r",
                Some(("", 1, 0, None, 5000)),
            ),
            // Constant expressions in LIMIT and OFFSET.
            (
                "SELECT id FROM t1 ORDER BY id LIMIT 1 + 1 OFFSET 2 * 3",
                Some(("t1.id", 1, 6, Some(2), 5000)),
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
            let order = result_order(&ctx, plan, &ComparisonOptions::default()).await;
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
        let ResultOrder::Sorted(reference) =
            result_order(&ctx, df.logical_plan(), &ComparisonOptions::default()).await
        else {
            panic!("not sorted");
        };
        let tolerance = ComparisonOptions::DEFAULT_FLOAT_TOLERANCE;
        let rows = reference.sorted_rows(tolerance)?;
        assert_eq!(rows.window, 0..3);
        let ids = rows.classes[..3]
            .iter()
            .map(|(range, keys)| (range.clone(), keys[0].to_string()))
            .collect::<Vec<_>>();
        let expected = [(0..1, "4999"), (1..2, "4998"), (2..3, "4997")];
        assert_eq!(ids, expected.map(|(range, id)| (range, id.to_string())));

        // A LIMIT that is not a constant.
        let df = ctx.sql("SELECT id FROM t1 ORDER BY id LIMIT $1").await?;
        let order =
            result_order(&ctx, df.logical_plan(), &ComparisonOptions::default()).await;
        assert!(matches!(&order, ResultOrder::Exact(_)), "{order:?}");
        Ok(())
    }

    /// A batch of `(k INT, v INT)` rows.
    fn kv_int_batch(rows: &[(i32, i32)]) -> RecordBatch {
        batch_of(vec![
            (
                "k",
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "v",
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.1))),
            ),
        ])
    }

    /// The order of `sql` over the table `m` of the `(k, v)` rows `rows`.
    async fn order_over(rows: &[(i32, i32)], sql: &str) -> Result<ResultOrder> {
        let ctx = datafusion_context(test_session_config());
        ctx.register_batch("m", kv_int_batch(rows))?;
        let df = ctx.sql(sql).await?;
        Ok(result_order(&ctx, df.logical_plan(), &ComparisonOptions::default()).await)
    }

    #[tokio::test]
    async fn result_order_computes_the_output_columns_of_rows_that_the_query_may_return()
    -> Result<()> {
        let sql = "SELECT 10 / v AS x FROM (SELECT v FROM m ORDER BY k LIMIT 2) t";
        let x = |values: &[i64]| {
            let x = Int64Array::from(values.to_vec());
            vec![batch_of(vec![("x", Arc::new(x) as ArrayRef)])]
        };
        // The query cannot return the row of `v = 0`, whose key does not tie with
        // the keys of the rows that it returns: its output columns are not
        // computed (which would fail), and the rows that tie may come in any
        // order.
        let order = order_over(&[(1, 1), (1, 2), (2, 0)], sql).await?;
        let ResultOrder::Sorted(reference) = &order else {
            panic!("not sorted: {order:?}");
        };
        let cannot_return = "the query cannot return it, as its keys cannot tie with \
                             the keys of the rows that it returns within the tolerance";
        assert_eq!(reference.uncomputed, [(2..3, cannot_return.to_string())]);
        same_results(x(&[10, 5]), x(&[5, 10]), &order).unwrap();
        let err = same_results(x(&[10, 5]), x(&[10, 10]), &order).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine differs"),
            "{err}"
        );

        // The query fails on the row of `v = 0`, which ties with the rows that
        // it returns: it may only return the other rows.
        let order = order_over(&[(1, 1), (1, 2), (1, 0)], sql).await?;
        let ResultOrder::Sorted(reference) = &order else {
            panic!("not sorted: {order:?}");
        };
        let [(rows, reason)] = reference.uncomputed.as_slice() else {
            panic!("{:?}", reference.uncomputed);
        };
        assert_eq!(rows.len(), 1);
        assert!(reason.starts_with("the query fails on it: "), "{reason}");
        assert!(reason.contains("Divide by zero"), "{reason}");
        same_results(x(&[10, 5]), x(&[5, 10]), &order).unwrap();
        same_results(x(&[5, 10]), x(&[10, 5]), &order).unwrap();

        // The query fails on the second row that it returns, whatever the order.
        let order = order_over(&[(1, 1), (2, 0), (3, 5)], sql).await?;
        let ResultOrder::Sorted(reference) = &order else {
            panic!("not sorted: {order:?}");
        };
        // The query cannot return the last row either, but does not fail on it:
        // its output columns are computed, to describe differences.
        let [(rows, reason)] = reference.uncomputed.as_slice() else {
            panic!("{:?}", reference.uncomputed);
        };
        assert_eq!(*rows, 1..2);
        assert!(reason.starts_with("the query fails on it: "), "{reason}");
        let err = same_results(x(&[10, 2]), x(&[10, 2]), &order).unwrap_err();
        assert!(
            err.contains(
                "row 1 of DataFusion is out of order: the sorted rows of the sorted \
                 reference of DataFusion with its values may not be at its position"
            ),
            "{err}"
        );
        assert!(
            err.contains(
                "sorted row 1 of the sorted reference of DataFusion: (not computed: the \
                 query fails on it: "
            ),
            "{err}"
        );
        Ok(())
    }

    /// The order of `sql` over the table `m` of `batch`.
    async fn order_of(batch: RecordBatch, sql: &str) -> Result<ResultOrder> {
        order_with(batch, sql, &ComparisonOptions::default()).await
    }

    /// The order of `sql` over the table `m` of `batch`, to be compared with
    /// `options`.
    async fn order_with(
        batch: RecordBatch,
        sql: &str,
        options: &ComparisonOptions,
    ) -> Result<ResultOrder> {
        let ctx = datafusion_context(test_session_config());
        ctx.register_batch("m", batch)?;
        let df = ctx.sql(sql).await?;
        Ok(result_order(&ctx, df.logical_plan(), options).await)
    }

    #[tokio::test]
    async fn same_results_order_rows_by_later_keys_after_floats_that_cannot_vary()
    -> Result<()> {
        let m = || ids(&[1, 2]).remove(0);
        // Zeros, infinities and NaNs are only equal to equal values: the later
        // key `id` decides the order, with and without LIMIT 1.
        for float in ["0", "'-0'", "'Infinity'", "'-Infinity'", "'NaN'"] {
            let sql = format!("SELECT id FROM m ORDER BY CAST({float} AS DOUBLE), id");
            let order = order_of(m(), &sql).await?;
            assert!(matches!(order, ResultOrder::Sorted(_)), "{sql}: {order:?}");
            same_results(ids(&[1, 2]), ids(&[1, 2]), &order).unwrap();
            let err = same_results(ids(&[1, 2]), ids(&[2, 1]), &order).unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{sql}: {err}"
            );
            let sql = format!("{sql} LIMIT 1");
            let order = order_of(m(), &sql).await?;
            same_results(ids(&[1]), ids(&[1]), &order).unwrap();
            let err = same_results(ids(&[1]), ids(&[2]), &order).unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{sql}: {err}"
            );
        }
        // Without a tolerance, no value may vary, but within the tolerance the
        // values of either row may be smaller.
        let exact = ComparisonOptions::default().with_float_tolerance(0.0);
        for sql in [
            "SELECT id FROM m ORDER BY CAST(1 AS DOUBLE), id",
            "SELECT id FROM m ORDER BY CAST(1 AS DOUBLE), id LIMIT 1",
        ] {
            let order = order_of(m(), sql).await?;
            let (expected, actual) = if sql.ends_with("LIMIT 1") {
                (ids(&[1]), ids(&[2]))
            } else {
                (ids(&[1, 2]), ids(&[2, 1]))
            };
            let err = same_results_with(expected.clone(), actual.clone(), &order, exact)
                .unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{sql}: {err}"
            );
            same_results(expected, actual, &order).unwrap();
        }
        // Nor within a tolerance smaller than the distance to their neighbors,
        // which is `1.1e-16` below 1 and `2.2e-16` above (not within twice the
        // tolerance, as the value of each engine may only be off by the
        // tolerance).
        let fine = ComparisonOptions::default().with_float_tolerance(8e-17);
        for sql in [
            "SELECT id FROM m ORDER BY CAST(1 AS DOUBLE), id",
            "SELECT id FROM m ORDER BY CAST(1 AS DOUBLE), id LIMIT 1",
        ] {
            let order = order_with(m(), sql, &fine).await?;
            let (expected, actual) = if sql.ends_with("LIMIT 1") {
                (ids(&[1]), ids(&[2]))
            } else {
                (ids(&[1, 2]), ids(&[2, 1]))
            };
            same_results_with(expected.clone(), expected.clone(), &order, fine).unwrap();
            let err = same_results_with(expected, actual, &order, fine).unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{sql}: {err}"
            );
        }
        // `Float32` values cannot vary within the default tolerance, which is
        // smaller than the distance to their neighbors, but can within a larger
        // tolerance.
        for sql in [
            "SELECT id FROM m ORDER BY CAST(1 AS REAL), id",
            "SELECT id FROM m ORDER BY CAST(1 AS REAL), id LIMIT 1",
        ] {
            let order = order_of(m(), sql).await?;
            let (expected, actual) = if sql.ends_with("LIMIT 1") {
                (ids(&[1]), ids(&[2]))
            } else {
                (ids(&[1, 2]), ids(&[2, 1]))
            };
            let err = same_results(expected.clone(), actual.clone(), &order).unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{sql}: {err}"
            );
            // The reference is computed for the tolerance of the comparison.
            let coarse = ComparisonOptions::default().with_float_tolerance(1e-6);
            let err = same_results_with(expected.clone(), actual.clone(), &order, coarse)
                .unwrap_err();
            assert!(
                err.starts_with(
                    "the sorted reference is computed for the float tolerance \
                     0.000000001, but the results are compared within 0.000001"
                ),
                "{err}"
            );
            let order = order_with(m(), sql, &coarse).await?;
            same_results_with(expected, actual, &order, coarse).unwrap();
        }
        // Nor within `4e-8`: their neighbors of 1 are `6e-8` below and `1.2e-7`
        // above (within twice the tolerance below, but the value of each engine
        // may only be off by the tolerance).
        let fine = ComparisonOptions::default().with_float_tolerance(4e-8);
        for sql in [
            "SELECT id FROM m ORDER BY CAST(1 AS REAL), id",
            "SELECT id FROM m ORDER BY CAST(1 AS REAL), id LIMIT 1",
        ] {
            let order = order_with(m(), sql, &fine).await?;
            let (expected, actual) = if sql.ends_with("LIMIT 1") {
                (ids(&[1]), ids(&[2]))
            } else {
                (ids(&[1, 2]), ids(&[2, 1]))
            };
            same_results_with(expected.clone(), expected.clone(), &order, fine).unwrap();
            let err = same_results_with(expected, actual, &order, fine).unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{sql}: {err}"
            );
        }

        // Distinct keys that cannot vary are in their order: `1.next_down()`
        // and `1`, which are not within `8e-17` of their neighbors, but are
        // within twice that tolerance of each other.
        let below_one = 1.0f64.next_down();
        let distinct = || float_kv_batch(&[(below_one, 1), (1.0, 2)], 0, 0.0);
        let v = |values: &[i32]| {
            vec![int_batch(
                &values.iter().copied().map(Some).collect::<Vec<_>>(),
            )]
        };
        let fine = ComparisonOptions::default().with_float_tolerance(8e-17);
        for (sql, expected, actual) in [
            (
                "SELECT v FROM m ORDER BY k",
                [1, 2].as_slice(),
                [2, 1].as_slice(),
            ),
            ("SELECT v FROM m ORDER BY k LIMIT 1", &[1], &[2]),
            ("SELECT v FROM m ORDER BY k DESC", &[2, 1], &[1, 2]),
        ] {
            let order = order_with(distinct(), sql, &fine).await?;
            same_results_with(v(expected), v(expected), &order, fine).unwrap();
            let err =
                same_results_with(v(expected), v(actual), &order, fine).unwrap_err();
            assert!(
                err.starts_with("row 0 of the pipeline engine is out of order"),
                "{sql}: {err}"
            );
            // Within the default tolerance, they tie.
            let order = order_of(distinct(), sql).await?;
            same_results(v(expected), v(actual), &order).unwrap();
        }
        // The keys after a float that cannot vary are shown.
        let order =
            order_of(m(), "SELECT id FROM m ORDER BY CAST(0 AS DOUBLE), id").await?;
        let err = same_results(ids(&[1, 2]), ids(&[2, 1]), &order).unwrap_err();
        assert!(err.contains("row 0: (2), keys: (0.0, 2)"), "{err}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_order_rows_by_later_keys_after_floats_that_cannot_vary()
     {
        let setup = async |ctx: &SessionContext| {
            ctx.register_batch("m", ids(&[2, 1, 3]).remove(0))?;
            Ok(())
        };
        for sql in [
            "SELECT id FROM m ORDER BY CAST(0 AS DOUBLE), id",
            "SELECT id FROM m ORDER BY CAST(0 AS DOUBLE), id LIMIT 1",
            "SELECT id FROM m ORDER BY CAST('NaN' AS DOUBLE), id DESC LIMIT 2",
            "SELECT id FROM m ORDER BY CAST('Infinity' AS DOUBLE), id OFFSET 1",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_evaluate_no_expressions_on_rows_that_are_cut() {
        // The query divides by the `v` of the rows that tie on `k = 1`, in other
        // orders in the tables of both engines, but not by the `0` of the row
        // that its LIMIT, or its OFFSET, cuts.
        let setup = per_engine_tables(
            kv_int_batch(&[(1, 1), (1, 2), (2, 0)]),
            kv_int_batch(&[(1, 2), (1, 1), (2, 0)]),
        );
        for sql in [
            "SELECT 10 / v AS x FROM (SELECT v FROM m ORDER BY k LIMIT 2) t",
            "SELECT k, 10 / v AS x FROM (SELECT k, v FROM m ORDER BY k LIMIT 2)",
            "SELECT 10 / x AS y FROM (SELECT 10 / v AS x FROM \
             (SELECT v FROM m ORDER BY k DESC OFFSET 1)) t",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    /// A batch of `(k DOUBLE, v INT)` rows: `rows`, then `zeros` rows of
    /// `(zero_k, 0)`.
    fn float_kv_batch(rows: &[(f64, i32)], zeros: usize, zero_k: f64) -> RecordBatch {
        let rows = rows
            .iter()
            .copied()
            .chain(std::iter::repeat_n((zero_k, 0), zeros))
            .collect::<Vec<_>>();
        batch_of(vec![
            (
                "k",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "v",
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|row| row.1))),
            ),
        ])
    }

    #[tokio::test]
    async fn same_results_order_rows_by_later_keys_where_floats_only_meet() -> Result<()>
    {
        // Within `1.5e-16`, `1 - 2.2e-16` and `1` may only become equal (at
        // `1 - 1.1e-16`), not cross: the later key `v` decides, in its order.
        let fine = ComparisonOptions::default().with_float_tolerance(1.5e-16);
        let (a, b) = (1.0f64.next_down().next_down(), 1.0);
        let v = |values: &[i32]| {
            vec![int_batch(
                &values.iter().copied().map(Some).collect::<Vec<_>>(),
            )]
        };
        for (rows, sql, expected, other, valid) in [
            (
                [(a, 1), (b, 2)],
                "SELECT v FROM m ORDER BY k, v",
                [1, 2],
                [2, 1],
                false,
            ),
            (
                [(a, 2), (b, 1)],
                "SELECT v FROM m ORDER BY k, v",
                [2, 1],
                [1, 2],
                true,
            ),
            (
                [(a, 1), (b, 2)],
                "SELECT v FROM m ORDER BY k, v DESC",
                [1, 2],
                [2, 1],
                true,
            ),
            (
                [(a, 1), (b, 2)],
                "SELECT v FROM m ORDER BY k DESC, v",
                [2, 1],
                [1, 2],
                true,
            ),
            (
                [(a, 2), (b, 1)],
                "SELECT v FROM m ORDER BY k DESC, v",
                [1, 2],
                [2, 1],
                false,
            ),
            // Without later keys, the rows may be equal.
            (
                [(a, 1), (b, 2)],
                "SELECT v FROM m ORDER BY k",
                [1, 2],
                [2, 1],
                true,
            ),
        ] {
            let order = order_with(float_kv_batch(&rows, 0, 0.0), sql, &fine).await?;
            same_results_with(v(&expected), v(&expected), &order, fine).unwrap();
            let result = same_results_with(v(&expected), v(&other), &order, fine);
            assert_eq!(result.is_ok(), valid, "{rows:?} {sql}: {result:?}");
            if !valid {
                let err = result.unwrap_err();
                assert!(
                    err.starts_with("row 0 of the pipeline engine is out of order"),
                    "{sql}: {err}"
                );
            }
            // Within the default tolerance, the values cross.
            let order = order_of(float_kv_batch(&rows, 0, 0.0), sql).await?;
            same_results(v(&expected), v(&other), &order).unwrap();
        }
        // The rows of `k = a` may be in either order, as `a` may vary, but only
        // the one of `v = 3` may be after the row of `k = b`, which meets them.
        let rows = [(a, 3), (a, 1), (b, 2)];
        let order = order_with(
            float_kv_batch(&rows, 0, 0.0),
            "SELECT v FROM m ORDER BY k, v",
            &fine,
        )
        .await?;
        for (result, valid) in [
            ([1, 3, 2], true),
            ([3, 1, 2], true),
            ([1, 2, 3], true),
            ([2, 1, 3], false),
            ([3, 2, 1], false),
        ] {
            let comparison = same_results_with(v(&[1, 3, 2]), v(&result), &order, fine);
            assert_eq!(comparison.is_ok(), valid, "{result:?}: {comparison:?}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_evaluate_no_rows_out_of_the_reach_of_the_tolerance()
     {
        // The keys `2.0` of the 200 rows on which the query fails cannot tie
        // with the keys `1.0` of the rows that it returns, within the tolerance
        // of the comparison: the reference does not look for the rows on which
        // it fails among them.
        let sql = "SELECT 10 / v AS x FROM (SELECT v FROM m ORDER BY k LIMIT 2) t";
        let setup = per_engine_tables(
            float_kv_batch(&[(1.0, 1), (1.0, 2)], 200, 2.0),
            float_kv_batch(&[(1.0, 2), (1.0, 1)], 200, 2.0),
        );
        assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        let batch = float_kv_batch(&[(1.0, 1), (1.0, 2)], 200, 2.0);
        let order = order_of(batch, sql).await.unwrap();
        let ResultOrder::Sorted(reference) = &order else {
            panic!("not sorted: {order:?}");
        };
        let cannot_return = "the query cannot return it, as its keys cannot tie with the \
                             keys of the rows that it returns within the tolerance";
        assert_eq!(reference.uncomputed, [(2..202, cannot_return.to_string())]);
    }

    /// `(k, v)` rows that all tie on `k = 1`: `first`, 300 rows of `v = 0`, then
    /// `last`.
    fn rows_around_zeros(
        first: [(i32, i32); 2],
        last: [(i32, i32); 2],
    ) -> Vec<(i32, i32)> {
        first
            .into_iter()
            .chain(std::iter::repeat_n((1, 0), 300))
            .chain(last)
            .collect()
    }

    #[tokio::test]
    async fn result_order_finds_each_row_on_which_the_query_fails() -> Result<()> {
        // The query may return any 2 rows, but fails on the 300 rows of `v = 0`:
        // the reference finds each of them, and computes the output columns of
        // the others, also of the last ones.
        let rows = rows_around_zeros([(1, 1), (1, 2)], [(1, 4), (1, 5)]);
        let sql = "SELECT 100 / v AS x FROM (SELECT v FROM m ORDER BY k LIMIT 2) t";
        let order = order_over(&rows, sql).await?;
        let ResultOrder::Sorted(reference) = &order else {
            panic!("not sorted: {order:?}");
        };
        let mut failing = 0;
        for (rows, reason) in &reference.uncomputed {
            assert!(reason.starts_with("the query fails on it: "), "{reason}");
            failing += rows.len();
        }
        assert_eq!(failing, 300);
        let x = |values: &[i64]| {
            let x = Int64Array::from(values.to_vec());
            vec![batch_of(vec![("x", Arc::new(x) as ArrayRef)])]
        };
        for valid in [[50, 100], [100, 50], [25, 20], [20, 100]] {
            same_results(x(&[100, 50]), x(&valid), &order)
                .unwrap_or_else(|e| panic!("{valid:?}: {e}"));
        }
        let err = same_results(x(&[100, 50]), x(&[25, 25]), &order).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine differs"),
            "{err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_return_any_rows_that_tie_among_rows_that_fail() {
        // Both engines return the first 2 rows of their tables.
        let setup = per_engine_tables(
            kv_int_batch(&rows_around_zeros([(1, 1), (1, 2)], [(1, 4), (1, 5)])),
            kv_int_batch(&rows_around_zeros([(1, 4), (1, 5)], [(1, 1), (1, 2)])),
        );
        assert_same_results_as_datafusion_with_setup(
            "SELECT 100 / v AS x FROM (SELECT v FROM m ORDER BY k LIMIT 2) t",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_find_each_row_that_fails_in_plans_that_join() {
        // The scalar subquery joins the rows of the reference with the rows of
        // `c`: the reference computes each row of `m` alone in a plan that joins.
        let tables = per_engine_tables(
            kv_int_batch(&rows_around_zeros([(1, 1), (1, 2)], [(1, 4), (1, 5)])),
            kv_int_batch(&rows_around_zeros([(1, 4), (1, 5)], [(1, 1), (1, 2)])),
        );
        let setup = async |ctx: &SessionContext| {
            tables(ctx).await?;
            ctx.register_batch("c", int_batch(&[Some(1)]))?;
            Ok(())
        };
        let sql = "SELECT 100 / v + (SELECT count(*) FROM c) AS x \
                   FROM (SELECT v FROM m ORDER BY k LIMIT 2) t";
        assert_same_results_as_datafusion_with_setup(sql, &setup).await;

        // The reference finds the 300 rows on which the query fails, and
        // computes the others.
        let ctx = datafusion_context(test_session_config());
        let rows = rows_around_zeros([(1, 1), (1, 2)], [(1, 4), (1, 5)]);
        ctx.register_batch("m", kv_int_batch(&rows)).unwrap();
        ctx.register_batch("c", int_batch(&[Some(1)])).unwrap();
        let df = ctx.sql(sql).await.unwrap();
        let order =
            result_order(&ctx, df.logical_plan(), &ComparisonOptions::default()).await;
        let ResultOrder::Sorted(reference) = &order else {
            panic!("not sorted: {order:?}");
        };
        let failing = reference
            .uncomputed
            .iter()
            .map(|(rows, _)| rows.len())
            .sum::<usize>();
        assert_eq!(failing, 300, "{:?}", reference.uncomputed);
        let x = |values: &[i64]| {
            let x = Int64Array::from(values.to_vec());
            vec![batch_of(vec![("x", Arc::new(x) as ArrayRef)])]
        };
        same_results(x(&[101, 51]), x(&[26, 21]), &order).unwrap();
        let err = same_results(x(&[101, 51]), x(&[26, 26]), &order).unwrap_err();
        assert!(
            err.starts_with("row 1 of the pipeline engine differs"),
            "{err}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_compare_the_rows_of_distinct_on_in_their_order() -> Result<()> {
        // `DISTINCT ON` absorbs the `ORDER BY` of its query.
        let sql = "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k DESC, v";
        let order = order_over(&[(1, 10), (2, 20), (3, 30)], sql).await?;
        assert!(matches!(order, ResultOrder::Sorted(_)), "{order:?}");
        let rows = |rows: &[(i32, i32)]| vec![kv_int_batch(rows)];
        let sorted = [(3, 30), (2, 20), (1, 10)];
        same_results(rows(&sorted), rows(&sorted), &order).unwrap();
        let err = same_results(rows(&sorted), rows(&[(1, 10), (2, 20), (3, 30)]), &order)
            .unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine is out of order"),
            "{err}"
        );
        let order =
            order_over(&[(1, 10), (2, 20), (3, 30)], &format!("{sql} LIMIT 2")).await?;
        let err = same_results(rows(&sorted[..2]), rows(&[(2, 20), (3, 30)]), &order)
            .unwrap_err();
        assert!(
            err.starts_with("row 0 of the pipeline engine is out of order"),
            "{err}"
        );

        // End to end, of several rows of each key.
        let setup = per_engine_tables(
            kv_int_batch(&[(1, 10), (1, 5), (2, 20), (3, 30), (3, 1)]),
            kv_int_batch(&[(3, 1), (3, 30), (2, 20), (1, 5), (1, 10)]),
        );
        for sql in [
            sql,
            "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k, v DESC LIMIT 2",
            "SELECT v FROM (SELECT DISTINCT ON (k) k, v FROM m ORDER BY k DESC, v) s",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
        Ok(())
    }

    /// A batch of `(k DOUBLE, id BIGINT)` rows.
    fn k_id_batch(rows: &[(f64, i64)]) -> RecordBatch {
        batch_of(vec![
            (
                "k",
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.0)))
                    as ArrayRef,
            ),
            (
                "id",
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1))),
            ),
        ])
    }

    #[tokio::test]
    async fn same_results_order_rows_by_later_keys_after_zeros_of_other_signs()
    -> Result<()> {
        // DataFusion sorts `-0.0` before `0.0`, but they are equal: another
        // engine may order the rows by `id`, but zeros of the same sign keep
        // the order of `id`.
        for (rows, sql, expected, other, valid) in [
            (
                [(0.0, 1), (-0.0, 2)],
                "SELECT id FROM m ORDER BY k, id",
                [2, 1].as_slice(),
                [1, 2].as_slice(),
                true,
            ),
            (
                [(0.0, 1), (-0.0, 2)],
                "SELECT id FROM m ORDER BY k, id LIMIT 1",
                &[2],
                &[1],
                true,
            ),
            (
                [(0.0, 2), (-0.0, 1)],
                "SELECT id FROM m ORDER BY k, id",
                &[1, 2],
                &[2, 1],
                false,
            ),
            (
                [(0.0, 2), (-0.0, 1)],
                "SELECT id FROM m ORDER BY k DESC, id",
                &[2, 1],
                &[1, 2],
                true,
            ),
            (
                [(0.0, 1), (-0.0, 2)],
                "SELECT id FROM m ORDER BY k DESC, id",
                &[1, 2],
                &[2, 1],
                false,
            ),
            (
                [(0.0, 1), (0.0, 2)],
                "SELECT id FROM m ORDER BY k, id",
                &[1, 2],
                &[2, 1],
                false,
            ),
            (
                [(0.0, 1), (0.0, 2)],
                "SELECT id FROM m ORDER BY k, id LIMIT 1",
                &[1],
                &[2],
                false,
            ),
            // A zero of another sign after equal keys.
            (
                [(1.0, 1), (1.0, 2)],
                "SELECT id FROM m ORDER BY CAST(1 AS INT), CASE WHEN id = 2 THEN -0.0 ELSE 0.0 END, id",
                &[2, 1],
                &[1, 2],
                true,
            ),
        ] {
            let order = order_of(k_id_batch(&rows), sql).await?;
            same_results(ids(expected), ids(expected), &order)
                .unwrap_or_else(|e| panic!("{rows:?} {sql}: {e}"));
            let result = same_results(ids(expected), ids(other), &order);
            assert_eq!(result.is_ok(), valid, "{rows:?} {sql}: {result:?}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_zeros_of_other_signs() {
        // The keys of DataFusion are `0.0` and `-0.0`, those of the pipeline
        // engine `0.0` and `0.0`: it orders the rows by `id`.
        let setup = per_engine_tables(
            k_id_batch(&[(0.0, 1), (-0.0, 2)]),
            k_id_batch(&[(0.0, 1), (0.0, 2)]),
        );
        for sql in [
            "SELECT id FROM m ORDER BY k, id",
            "SELECT id FROM m ORDER BY k, id LIMIT 1",
            "SELECT id, k FROM m ORDER BY k, id",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_other_rows_of_nested_limits() {
        // The rows tie on `k`, in other orders in the tables of both engines: the
        // nested ORDER BY … LIMIT of each engine returns another row.
        let rows = |rows: &[(i64, f64)]| {
            k_id_batch(&rows.iter().map(|&(id, k)| (k, id)).collect::<Vec<_>>())
        };
        let setup =
            per_engine_tables(rows(&[(1, 0.5), (2, 0.5)]), rows(&[(2, 0.5), (1, 0.5)]));
        for sql in [
            "SELECT id FROM (SELECT id FROM m ORDER BY k LIMIT 1) s ORDER BY id",
            "SELECT id AS v FROM (SELECT id FROM m ORDER BY k LIMIT 1) s ORDER BY v",
            "SELECT id FROM (SELECT id FROM m ORDER BY k LIMIT 1) s WHERE id > 0",
            "SELECT sum(id) AS total FROM (SELECT id FROM m ORDER BY k LIMIT 1) s",
            "SELECT id FROM m WHERE id IN (SELECT id FROM m ORDER BY k LIMIT 1)",
            "SELECT id FROM (SELECT id FROM m ORDER BY k LIMIT 1 OFFSET 1) s ORDER BY id",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
        let setup = per_engine_tables(
            kv_int_batch(&[(0, 1), (0, 2)]),
            kv_int_batch(&[(0, 2), (0, 1)]),
        );
        assert_same_results_as_datafusion_with_setup(
            "SELECT v FROM (SELECT v FROM m ORDER BY k LIMIT 1) s ORDER BY v",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_other_rows_of_several_nested_limits() {
        // Both nested limits return another row in each engine, and any of the
        // 67 rows of `m` (beyond any number of choices to try): the rows of the
        // result show which.
        let (sessions, many) = (AtomicUsize::new(0), 1..=67);
        let setup = async |ctx: &SessionContext| {
            let datafusion = sessions
                .fetch_add(1, AtomicOrdering::Relaxed)
                .is_multiple_of(2);
            let (mut m, mut n) = (vec![(1, 1), (1, 2)], vec![(1, 10), (1, 20)]);
            let mut l = many.clone().map(|v| (1, v)).collect::<Vec<_>>();
            if !datafusion {
                m.reverse();
                n.reverse();
                l.reverse();
            }
            ctx.register_batch("m", kv_int_batch(&m))?;
            ctx.register_batch("n", kv_int_batch(&n))?;
            ctx.register_batch("l", kv_int_batch(&l))?;
            Ok(())
        };
        for sql in [
            "SELECT a.v AS a, b.v AS b \
             FROM (SELECT v FROM m ORDER BY k LIMIT 1) a \
             CROSS JOIN (SELECT v FROM n ORDER BY k LIMIT 1) b ORDER BY a",
            "SELECT a.v AS a, b.v AS b \
             FROM (SELECT v FROM m ORDER BY k LIMIT 1) a \
             JOIN (SELECT v FROM n ORDER BY k LIMIT 1) b ON a.v < b.v WHERE a.v > 0",
            // An equi-join, which only joins the rows of the same choices.
            "SELECT a.v AS a, b.w AS b \
             FROM (SELECT v FROM m ORDER BY k LIMIT 1) a \
             JOIN (SELECT v / 10 AS v, v AS w FROM n ORDER BY k LIMIT 1) b ON a.v = b.v \
             ORDER BY a",
            "SELECT a.v AS a, b.v AS b \
             FROM (SELECT v FROM m ORDER BY k LIMIT 1) a \
             JOIN (SELECT v - 9 AS v FROM n ORDER BY k LIMIT 1) b ON a.v = b.v",
            "SELECT v FROM (SELECT v FROM l ORDER BY k LIMIT 1) s ORDER BY v",
            "SELECT v + 1 AS w FROM (SELECT v FROM l ORDER BY k LIMIT 2) s ORDER BY w DESC",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test]
    async fn alternative_outcomes_choose_the_rows_of_each_occurrence_of_a_nested_limit()
    -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        ctx.register_batch("m", kv_int_batch(&[(1, 1), (1, 2)]))?;
        let options = ComparisonOptions::default();
        let ab = |rows: &[(i32, i32)]| {
            let column = |i: usize| {
                Arc::new(Int32Array::from_iter_values(
                    rows.iter().map(|row| [row.0, row.1][i]),
                )) as ArrayRef
            };
            output(
                vec![batch_of(vec![("a", column(0)), ("b", column(1))])],
                true,
            )
        };
        let limit = "(SELECT v FROM m ORDER BY k LIMIT 1)";
        for sql in [
            // The rows of the result show the rows of both occurrences.
            format!(
                "SELECT a.v AS a, b.v AS b FROM {limit} a CROSS JOIN {limit} b ORDER BY a, b"
            ),
            // They do not, through the aggregate.
            format!(
                "SELECT min(a.v) AS a, max(b.v) AS b FROM {limit} a CROSS JOIN {limit} b"
            ),
        ] {
            let plan = ctx.sql(&sql).await?.logical_plan().clone();
            for (valid, rows) in [(true, [(1, 2)]), (true, [(2, 1)]), (false, [(2, 3)])] {
                let result = ab(&rows);
                let alternatives =
                    alternative_outcomes(&ctx, &plan, &options, &result).await;
                let accepted = alternatives.iter().any(|(expected, order)| {
                    compare_results(expected, &result, order, &options).is_ok()
                });
                assert_eq!(accepted, valid, "{sql}: {rows:?}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_other_rows_of_nested_limits_below_aggregates()
     {
        // Any of the 67 rows that tie, in other orders in the tables of both
        // engines, through an aggregate (which does not keep the lineage of the
        // rows).
        let rows = (1..=67).map(|v| (1, v)).collect::<Vec<_>>();
        let reversed = rows.iter().rev().copied().collect::<Vec<_>>();
        let setup = per_engine_tables(kv_int_batch(&rows), kv_int_batch(&reversed));
        for sql in [
            "SELECT sum(v) AS total FROM (SELECT v FROM m ORDER BY k LIMIT 1) s",
            "SELECT count(*) AS n, max(v) AS top FROM (SELECT v FROM m ORDER BY k LIMIT 2) s",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_other_rows_that_tie_in_distinct_on() {
        // `DISTINCT ON` returns one row of each group, which may be any of the
        // rows that tie for the first row of the group.
        let setup = per_engine_tables(
            kv_int_batch(&[(0, 1), (0, 2), (1, 5), (1, 6), (1, 4)]),
            kv_int_batch(&[(1, 4), (1, 6), (0, 2), (0, 1), (1, 5)]),
        );
        for sql in [
            "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k LIMIT 1",
            "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k DESC",
            "SELECT DISTINCT ON (k) k, v FROM m",
            "SELECT v FROM (SELECT DISTINCT ON (k) k, v FROM m ORDER BY k) s ORDER BY v",
            "SELECT sum(v) AS total FROM (SELECT DISTINCT ON (k) k, v FROM m) s",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test]
    async fn alternative_outcomes_of_distinct_on_are_rows_that_tie_for_first()
    -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        ctx.register_batch("m", kv_int_batch(&[(0, 1), (0, 2), (1, 5)]))?;
        let options = ComparisonOptions::default();
        let kv = |rows: &[(i32, i32)]| output(vec![kv_int_batch(rows)], true);
        for (sql, rows, valid) in [
            (
                "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k",
                vec![(0, 2), (1, 5)],
                true,
            ),
            (
                "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k",
                vec![(0, 3), (1, 5)],
                false,
            ),
            (
                "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k",
                vec![(1, 5), (0, 2)],
                false,
            ),
            // `v` decides which row of the group is first.
            (
                "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k, v",
                vec![(0, 2), (1, 5)],
                false,
            ),
            (
                "SELECT DISTINCT ON (k) k, v FROM m ORDER BY k, v DESC",
                vec![(0, 2), (1, 5)],
                true,
            ),
        ] {
            let plan = ctx.sql(sql).await?.logical_plan().clone();
            let result = kv(&rows);
            let datafusion = ctx.sql(sql).await?;
            let expected = run_dataframe(&ctx, datafusion).await?;
            let order = result_order(&ctx, &plan, &options).await;
            let accepted = compare_results(&expected, &result, &order, &options).is_ok()
                || alternative_outcomes(&ctx, &plan, &options, &result)
                    .await
                    .iter()
                    .any(|(expected, order)| {
                        compare_results(expected, &result, order, &options).is_ok()
                    });
            assert_eq!(accepted, valid, "{sql}: {rows:?}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_other_rows_of_nested_limits_inside_others()
    {
        // The inner limit may return any 2 of the 3 rows that tie, so the outer
        // one `1` or `2`.
        let setup = per_engine_tables(
            kv_int_batch(&[(0, 1), (0, 2), (0, 3)]),
            kv_int_batch(&[(0, 3), (0, 2), (0, 1)]),
        );
        let inner = "(SELECT v FROM m ORDER BY k LIMIT 2) a";
        for sql in [
            format!(
                "SELECT sum(v) AS total FROM (SELECT v FROM {inner} ORDER BY v LIMIT 1) b"
            ),
            format!("SELECT v FROM (SELECT v FROM {inner} ORDER BY v LIMIT 1) b"),
        ] {
            assert_same_results_as_datafusion_with_setup(&sql, &setup).await;
        }
        // The outer limit may return any of the rows that tie, also for the rows
        // of the inner limit that DataFusion returns.
        let ctx = datafusion_context(test_session_config());
        ctx.register_batch("m", kv_int_batch(&[(0, 1), (0, 2), (0, 3)]))
            .unwrap();
        let sql = "SELECT sum(v) AS total FROM \
                   (SELECT v FROM (SELECT k, v FROM m ORDER BY v LIMIT 3) a ORDER BY k LIMIT 1) b";
        let plan = ctx.sql(sql).await.unwrap().logical_plan().clone();
        let options = ComparisonOptions::default();
        let expected = run_dataframe(&ctx, ctx.sql(sql).await.unwrap())
            .await
            .unwrap();
        let order = result_order(&ctx, &plan, &options).await;
        for (total, valid) in [(1, true), (2, true), (3, true), (4, false), (6, false)] {
            let values = Int64Array::from(vec![total]);
            let result = output(
                vec![batch_of(vec![("total", Arc::new(values) as ArrayRef)])],
                false,
            );
            let accepted = compare_results(&expected, &result, &order, &options).is_ok()
                || alternative_outcomes(&ctx, &plan, &options, &result)
                    .await
                    .iter()
                    .any(|(expected, order)| {
                        compare_results(expected, &result, order, &options).is_ok()
                    });
            assert_eq!(accepted, valid, "{sql}: {total:?}");
        }

        // But never `3`.
        let ctx = datafusion_context(test_session_config());
        ctx.register_batch("m", kv_int_batch(&[(0, 1), (0, 2), (0, 3)]))
            .unwrap();
        let sql = format!("SELECT v FROM (SELECT v FROM {inner} ORDER BY v LIMIT 1) b");
        let plan = ctx.sql(&sql).await.unwrap().logical_plan().clone();
        let options = ComparisonOptions::default();
        let expected = run_dataframe(&ctx, ctx.sql(&sql).await.unwrap())
            .await
            .unwrap();
        let order = result_order(&ctx, &plan, &options).await;
        for (v, valid) in [(1, true), (2, true), (3, false)] {
            let result = output(vec![int_batch(&[Some(v)])], false);
            let accepted = compare_results(&expected, &result, &order, &options).is_ok()
                || alternative_outcomes(&ctx, &plan, &options, &result)
                    .await
                    .iter()
                    .any(|(expected, order)| {
                        compare_results(expected, &result, order, &options).is_ok()
                    });
            assert_eq!(accepted, valid, "{v}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_complete_the_rows_of_nested_limits_that_are_filtered_out()
     {
        // The rows that the filter keeps show 2 of the 3 rows of the nested
        // limit, the third one is any row that the filter removes.
        let rows = (0..100).map(|v| (0, v)).collect::<Vec<_>>();
        let reversed = rows.iter().rev().copied().collect::<Vec<_>>();
        let setup = per_engine_tables(kv_int_batch(&rows), kv_int_batch(&reversed));
        assert_same_results_as_datafusion_with_setup(
            "SELECT v FROM (SELECT v FROM m ORDER BY k LIMIT 3) s WHERE v >= 98",
            &setup,
        )
        .await;
    }

    /// The outcomes of `sql` over the table `m` of `batch` in DataFusion, and
    /// `pipeline` as the output of the pipeline engine.
    async fn outcomes_of(
        batch: RecordBatch,
        sql: &str,
        pipeline: Vec<RecordBatch>,
    ) -> Result<EngineOutcomes> {
        let ctx = datafusion_context(test_session_config());
        ctx.register_batch("m", batch)?;
        let options = ComparisonOptions::default();
        let plan = ctx.sql(sql).await?.logical_plan().clone();
        let datafusion = run_dataframe(&ctx, ctx.sql(sql).await?).await?;
        let pipeline = output(pipeline, true);
        let order = result_order(&ctx, &plan, &options).await;
        let mut alternatives =
            alternative_outcomes(&ctx, &plan, &options, &pipeline).await;
        alternatives.extend(nan_alternatives(&ctx, &plan, &options).await);
        Ok(EngineOutcomes {
            datafusion: Ok(datafusion),
            pipeline: Ok(pipeline),
            order,
            alternatives,
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_nans_of_other_signs() -> Result<()> {
        // DataFusion sorts `-NaN` first and `NaN` last, but `NaN = NaN`: an engine
        // whose `NaN`s are all positive sorts them last, by their later keys.
        let nan = f64::NAN;
        for (datafusion, pipeline, sqls) in [
            (
                vec![(-nan, 2), (nan, 1)],
                vec![(nan, 2), (nan, 1)],
                vec![
                    "SELECT k, v FROM m ORDER BY k, v",
                    "SELECT v FROM m ORDER BY k, v LIMIT 1",
                ],
            ),
            (
                vec![(-nan, 3), (1.0, 2), (nan, 1)],
                vec![(nan, 3), (1.0, 2), (nan, 1)],
                vec![
                    "SELECT v FROM m ORDER BY k, v",
                    "SELECT k, v FROM m ORDER BY k DESC, v",
                ],
            ),
        ] {
            let setup = per_engine_tables(
                float_kv_batch(&datafusion, 0, 0.0),
                float_kv_batch(&pipeline, 0, 0.0),
            );
            for sql in sqls {
                assert_same_results_as_datafusion_with_setup(sql, &setup).await;
            }
        }
        // But the `NaN`s stay together, in the order of their later keys.
        let batch = float_kv_batch(&[(-nan, 3), (1.0, 2), (nan, 1)], 0, 0.0);
        let options = ComparisonOptions::default();
        for (result, valid) in [
            ([3, 2, 1], true),
            ([2, 1, 3], true),
            ([1, 3, 2], true),
            ([3, 1, 2], false),
        ] {
            let pipeline = vec![int_batch(&result.map(Some))];
            let outcomes =
                outcomes_of(batch.clone(), "SELECT v FROM m ORDER BY k, v", pipeline)
                    .await?;
            assert_eq!(outcomes.compare(&options).is_ok(), valid, "{result:?}");
        }
        Ok(())
    }

    /// A batch of `(k INT, v INT)` rows, whose schema and field `v` have
    /// metadata.
    fn kv_metadata_batch(rows: &[(i32, i32)]) -> RecordBatch {
        let batch = kv_int_batch(rows);
        let schema = batch.schema();
        let fields = schema
            .fields()
            .iter()
            .map(|field| {
                let metadata = HashMap::from([("unit".to_string(), "kg".to_string())]);
                if field.name() == "v" {
                    Arc::new(field.as_ref().clone().with_metadata(metadata))
                } else {
                    Arc::clone(field)
                }
            })
            .collect::<Vec<_>>();
        let metadata = HashMap::from([("source".to_string(), "test".to_string())]);
        let schema = Schema::new_with_metadata(fields, metadata);
        RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_require_the_schema_of_datafusion_with_other_rows_of_nested_limits()
    -> Result<()> {
        let sql = "SELECT * FROM (SELECT * FROM m ORDER BY k LIMIT 1) s ORDER BY v";
        let setup = per_engine_tables(
            kv_metadata_batch(&[(0, 1), (0, 2)]),
            kv_metadata_batch(&[(0, 2), (0, 1)]),
        );
        assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        // The other row that ties, with the metadata of the schema of DataFusion
        // or without it.
        let options = ComparisonOptions::default();
        let batch = kv_metadata_batch(&[(0, 1), (0, 2)]);
        let with = vec![kv_metadata_batch(&[(0, 2)])];
        let outcomes = outcomes_of(batch.clone(), sql, with).await?;
        outcomes.compare(&options).unwrap();
        let without = vec![kv_int_batch(&[(0, 2)])];
        let outcomes = outcomes_of(batch, sql, without).await?;
        let err = outcomes.compare(&options).unwrap_err();
        assert!(err.starts_with("the output schemas differ"), "{err}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(expected = "Results differ: row 0 of the pipeline engine differs")]
    async fn assert_same_results_panics_on_rows_that_nested_limits_cannot_return() {
        // The nested limit may return the row of `id = 1` or of `id = 2`, but not
        // of `id = 3`, whose key does not tie with theirs (in the table of the
        // pipeline engine, it is the only row of the smallest key).
        let setup = per_engine_tables(
            k_id_batch(&[(0.5, 1), (0.5, 2), (0.7, 3)]),
            k_id_batch(&[(0.7, 1), (0.7, 2), (0.5, 3)]),
        );
        assert_same_results_as_datafusion_with_setup(
            "SELECT id FROM (SELECT id FROM m ORDER BY k LIMIT 1) s ORDER BY id",
            &setup,
        )
        .await;
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

    /// A setup with the rows of `CLOSE_KEYS` in the session of DataFusion,
    /// and in the session of the pipeline engine with their keys in the other
    /// order.
    fn reversed_close_rows() -> impl AsyncFn(&SessionContext) -> Result<()> {
        let reversed = (0..4)
            .map(|i| (CLOSE_KEYS[3 - i], ["a", "b", "c", "d"][i]))
            .collect::<Vec<_>>();
        per_engine_tables(close_rows(&[0, 1, 2, 3]).remove(0), sv_batch(&reversed))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(
        expected = "Results differ: row 0 of the pipeline engine is out of order"
    )]
    async fn assert_same_results_panics_on_rows_out_of_order_by_close_floats() {
        let setup = reversed_close_rows();
        assert_same_results_as_datafusion_with_setup(
            "SELECT v FROM m ORDER BY s",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[should_panic(
        expected = "Results differ: row 0 of the pipeline engine is out of order: the sorted rows of DataFusion with its values may not be at its position"
    )]
    async fn assert_same_results_panics_on_another_first_row_by_close_floats() {
        let setup = reversed_close_rows();
        assert_same_results_as_datafusion_with_setup(
            "SELECT v FROM m ORDER BY s LIMIT 1",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_return_the_rows_at_the_offset() {
        // The values of the rows of `k = 1` and `k = 3` are equal.
        let rows = kv_batch(&[(1, "a"), (2, "b"), (3, "a")]);
        let setup = per_engine_tables(rows.clone(), rows);
        for sql in [
            "SELECT v FROM m ORDER BY k OFFSET 2",
            "SELECT v FROM m ORDER BY k LIMIT 1 OFFSET 2",
            "SELECT v FROM m ORDER BY k OFFSET 1",
            "SELECT v FROM m ORDER BY k DESC OFFSET 1",
            "SELECT v FROM m ORDER BY k DESC LIMIT 1 OFFSET 2",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_match_close_floats_of_rows_that_tie() {
        // Each value of `y` of the pipeline engine is within the tolerance of the
        // value of the other row of DataFusion, whose key ties with its key.
        let setup = per_engine_tables(
            xy_batch(&[(1.0, 1.0), (1.0 + 1.5e-9, 1.0 + 1.8e-9)]),
            xy_batch(&[(1.0, 1.0 + 0.9e-9), (1.0 + 1.5e-9, 1.0 - 0.9e-9)]),
        );
        assert_same_results_as_datafusion_with_setup(
            "SELECT y FROM m ORDER BY x",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_search_the_orders_of_rows_whose_ties_are_not_transitive()
     {
        // Keys that tie with the next one but not with the one after, and rows
        // of the same value.
        let rows = sv_batch(&[(1.0, "a"), (1.0 + 1.5e-9, "a"), (1.0 + 3e-9, "a")]);
        let setup = per_engine_tables(rows.clone(), rows);
        for sql in [
            "SELECT v FROM m ORDER BY s OFFSET 1",
            "SELECT v FROM m ORDER BY s LIMIT 1 OFFSET 1",
            "SELECT v FROM m ORDER BY s LIMIT 2",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
        // The payloads of the pipeline engine are only in order if they are
        // assigned to the rows of DataFusion of other keys.
        let setup =
            per_engine_tables(chained_rows([5, 3, 1, 2]), chained_rows([2, 3, 6, 1]));
        assert_same_results_as_datafusion_with_setup(
            "SELECT y FROM m ORDER BY x",
            &setup,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_search_the_rows_that_the_offset_skips() {
        let t = 1e-9;
        // The keys of the pipeline engine order the rows `[2, 0, 1]`, an order of
        // the keys of DataFusion in which rows are only out of their sorted order
        // relative to rows that they tie with.
        let setup = per_engine_tables(
            id_keys_batch(&[
                (0, 1.0, 1.0),
                (1, 1.0 + 0.1 * t, 1.0 + 3.0 * t),
                (2, 1.0 + 0.2 * t, 1.0 + 1.5 * t),
            ]),
            id_keys_batch(&[
                (0, 1.0 + 0.15 * t, 1.0),
                (1, 1.0 + 0.3 * t, 1.0 + 3.0 * t),
                (2, 1.0, 1.0 + 1.5 * t),
            ]),
        );
        for sql in [
            "SELECT id FROM m ORDER BY k1, k2 LIMIT 1 OFFSET 1",
            "SELECT id FROM m ORDER BY k1, k2 OFFSET 1",
            "SELECT id FROM m ORDER BY k1, k2",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_order_rows_lexicographically_within_the_tolerance()
     {
        // The values of `k` of the pipeline engine are the values of DataFusion
        // swapped, within the tolerance: it orders the rows `[1, 0]` on `(k, s)`.
        let close = 1.0 + 0.5e-9;
        let setup = per_engine_tables(
            id_k_s_batch(&[(0, 1.0, "a"), (1, close, "b")]),
            id_k_s_batch(&[(0, close, "a"), (1, 1.0, "b")]),
        );
        for sql in [
            "SELECT id, k, s FROM m ORDER BY k, s",
            "SELECT id, k, s FROM m ORDER BY k, s LIMIT 1",
            "SELECT id, s FROM m ORDER BY k, s",
            "SELECT id FROM m ORDER BY k, s LIMIT 1",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_evaluate_constant_limits() {
        // Rows that tie, in other orders in the tables of both engines.
        let setup = per_engine_tables(
            kv_batch(&[(5, "a"), (5, "b"), (5, "c")]),
            kv_batch(&[(5, "c"), (5, "b"), (5, "a")]),
        );
        for sql in [
            "SELECT k, v FROM m ORDER BY k LIMIT 1 + 1",
            "SELECT v FROM m ORDER BY k LIMIT 3 - 1 OFFSET 2 - 1",
        ] {
            assert_same_results_as_datafusion_with_setup(sql, &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_results_as_datafusion_accept_any_order_by_volatile_columns() {
        for sql in [
            "SELECT id FROM (SELECT id, random() AS r FROM t1) s ORDER BY r LIMIT 3",
            "SELECT id FROM (SELECT id, random() AS r FROM t1) ORDER BY r LIMIT 3",
            "SELECT id, d FROM (SELECT id, d, random() AS r FROM t1) s ORDER BY d, r LIMIT 10",
        ] {
            assert_same_results_as_datafusion(sql).await;
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
    #[should_panic(expected = "Results differ: row 2 of the pipeline engine differs")]
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
        expected = "Results differ: row 0 of the pipeline engine is out of order"
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
            let outcomes = run_both_engines(
                sql,
                test_session_config(),
                &setup,
                &ComparisonOptions::default(),
            )
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
