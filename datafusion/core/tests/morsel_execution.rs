// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(feature = "sql")]

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion::arrow::array::{
    BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion::catalog::Session;
use datafusion::common::{exec_err, plan_err};
use datafusion::datasource::{
    MemTable, TableProvider, TableType, ViewTable, empty::EmptyTable,
};
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::QueryPlanner;
use datafusion::logical_expr::{Expr, LogicalPlan, ScalarUDF, Volatility, create_udf};
use datafusion::morsel::MorselQueryPlanner;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, collect, displayable,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use tokio::sync::mpsc;
use tokio::task::Id;

const FILTERED_QUERY: &str =
    "SELECT id + 1 AS result, label FROM items WHERE threshold > 0";

fn morsel_context(workers: usize) -> SessionContext {
    morsel_context_with_config(workers, SessionConfig::new())
}

fn morsel_context_with_config(workers: usize, config: SessionConfig) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(workers).unwrap(),
        )))
        .build();
    SessionContext::new_with_state(state)
}

fn items(empty: bool) -> Result<Arc<MemTable>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("threshold", DataType::Int32, true),
        Field::new("unused", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    let mut partitions = vec![vec![], vec![]];
    if !empty {
        for i in 0..24 {
            partitions[i as usize % 2].push(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(vec![3 * i, 3 * i + 1, 3 * i + 2])),
                    Arc::new(Int32Array::from(vec![Some(10), None, Some(-1)])),
                    Arc::new(Int32Array::from(vec![0, 1, 2])),
                    Arc::new(StringArray::from(vec![
                        (i % 2 == 0).then_some("even"),
                        Some("null predicate"),
                        Some("false predicate"),
                    ])),
                ],
            )?);
        }
    }
    Ok(Arc::new(MemTable::try_new(schema, partitions)?))
}

async fn physical(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

#[tokio::test]
async fn sql_mode_selects_the_planner_for_new_queries() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_table("items", items(false)?)?;
    let scan = "SELECT id FROM items";
    let unsupported = "SELECT id FROM items ORDER BY id DESC";
    let before_set = ctx.sql(scan).await?;
    assert_ne!(physical(&ctx, scan).await?.name(), "MorselExec");

    ctx.sql("SET datafusion.execution.morsel_mode = 'prefer'")
        .await?
        .collect()
        .await?;
    assert_eq!(physical(&ctx, scan).await?.name(), "MorselExec");
    assert_ne!(
        before_set.create_physical_plan().await?.name(),
        "MorselExec"
    );
    ctx.sql(unsupported).await?.collect().await?;

    let control = ctx
        .sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    control.clone().collect().await?;
    let (state, _) = control.into_parts();
    let sorted = ctx.sql(unsupported).await?.into_unoptimized_plan();
    assert!(state.create_physical_plan(&sorted).await.is_err());

    assert_eq!(physical(&ctx, scan).await?.name(), "MorselExec");
    let count = "SELECT COUNT(*) FROM items WHERE threshold > 0";
    assert_eq!(physical(&ctx, count).await?.name(), "MorselExec");
    let err = physical(&ctx, unsupported).await.unwrap_err().to_string();
    assert!(
        err.contains("morsel_mode=required") && err.contains("Sort"),
        "{err}"
    );
    assert!(
        ctx.sql("SET datafusion.execution.morsel_mode = 'invalid'")
            .await
            .is_err()
    );
    assert_eq!(physical(&ctx, scan).await?.name(), "MorselExec");

    ctx.sql("RESET datafusion.execution.morsel_mode")
        .await?
        .collect()
        .await?;
    assert_ne!(physical(&ctx, scan).await?.name(), "MorselExec");
    Ok(())
}

#[tokio::test]
async fn explain_and_views_do_not_bypass_required_mode() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_table("items", items(false)?)?;
    ctx.sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    ctx.sql("CREATE VIEW totals AS SELECT SUM(id) AS total FROM items")
        .await?
        .collect()
        .await?;
    for query in [
        "SELECT * FROM totals",
        "SELECT id FROM items WHERE false",
        "EXPLAIN SELECT id FROM items ORDER BY id",
        "EXPLAIN FORMAT PGJSON SELECT * FROM totals",
        "EXPLAIN ANALYZE SELECT * FROM totals",
    ] {
        let err = physical(&ctx, query).await.unwrap_err().to_string();
        assert!(err.contains("morsel_mode=required"), "{query}: {err}");
    }

    let count = "SELECT COUNT(*) FROM items WHERE threshold > 0";
    for format in ["INDENT", "TREE"] {
        let explain = ctx
            .sql(&format!("EXPLAIN FORMAT {format} {count}"))
            .await?
            .collect()
            .await?;
        let explain = format!("{explain:?}");
        assert!(explain.contains("MorselExec"), "{explain}");
        assert!(explain.contains("CountSink"), "{explain}");
    }
    let analyzed = physical(&ctx, &format!("EXPLAIN ANALYZE {count}")).await?;
    assert_eq!(analyzed.children()[0].name(), "MorselExec");
    collect(analyzed, ctx.task_ctx()).await?;
    Ok(())
}

#[tokio::test]
async fn native_execution_errors_are_not_retried_with_stock_execution() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_batch(
        "input",
        RecordBatch::try_from_iter([("id", Arc::new(Int32Array::from(vec![1])) as _)])?,
    )?;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    ctx.register_udf(create_udf(
        "fail_native",
        vec![DataType::Int32],
        DataType::Int32,
        Volatility::Volatile,
        Arc::new(move |_| {
            observed.fetch_add(1, Ordering::Relaxed);
            exec_err!("test native execution error")
        }),
    ));
    for (i, mode) in ["prefer", "required"].into_iter().enumerate() {
        ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
            .await?;
        let plan = physical(&ctx, "SELECT fail_native(id) FROM input").await?;
        assert_eq!(plan.name(), "MorselExec");
        let err = collect(plan, ctx.task_ctx()).await.unwrap_err().to_string();
        assert!(err.contains("test native execution error"), "{err}");
        assert_eq!(calls.load(Ordering::Relaxed), i + 1);
    }
    Ok(())
}

#[derive(Debug)]
struct CustomPlanner;

#[async_trait::async_trait]
impl QueryPlanner for CustomPlanner {
    async fn create_physical_plan(
        &self,
        _: &LogicalPlan,
        _: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan_err!("custom planner invoked")
    }
}

#[tokio::test]
async fn required_mode_checks_explicit_planners() -> Result<()> {
    let state = SessionStateBuilder::new_with_default_features()
        .with_query_planner(Arc::new(CustomPlanner))
        .build();
    let custom = SessionContext::new_with_state(state);
    let explicit = morsel_context(1);
    for ctx in [&custom, &explicit] {
        ctx.register_table("items", items(false)?)?;
    }
    let scan = "SELECT id FROM items";
    assert_eq!(physical(&explicit, scan).await?.name(), "MorselExec");
    for mode in ["off", "prefer", "required"] {
        for ctx in [&custom, &explicit] {
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
        }
        let err = physical(&custom, scan).await.unwrap_err().to_string();
        if mode == "required" {
            assert!(
                err.contains("morsel_mode=required cannot use a custom"),
                "{err}"
            );
        } else {
            assert!(err.contains("custom planner invoked"), "{err}");
        }
    }
    custom
        .sql("SHOW datafusion.execution.morsel_mode")
        .await?
        .collect()
        .await?;
    assert_eq!(physical(&explicit, scan).await?.name(), "MorselExec");
    let err = physical(&explicit, "SELECT id FROM items ORDER BY id")
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("morsel_mode=required"), "{err}");
    Ok(())
}

fn sorted_rows(batches: &[RecordBatch]) -> Result<Vec<OwnedRow>> {
    let Some(first) = batches.first() else {
        return Ok(vec![]);
    };
    let converter = RowConverter::new(
        first
            .schema()
            .fields()
            .iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect(),
    )?;
    let mut rows = Vec::new();
    for batch in batches {
        rows.extend(
            converter
                .convert_columns(batch.columns())?
                .iter()
                .map(|row| row.owned()),
        );
    }
    rows.sort_unstable();
    Ok(rows)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lowers_memory_filter_and_projection() -> Result<()> {
    for empty in [false, true] {
        let table = items(empty)?;
        let default = SessionContext::new();
        default.register_table("items", table.clone())?;
        for workers in [1, 4] {
            let morsel = morsel_context(workers);
            morsel
                .sql("SET datafusion.execution.morsel_mode = 'required'")
                .await?;
            morsel.register_table("items", table.clone())?;
            for (sql, pruned_columns) in [
                (FILTERED_QUERY, Some("[0, 2]")),
                ("SELECT label FROM items WHERE threshold > 0", Some("[1]")),
                (
                    "SELECT 42 AS constant FROM items WHERE threshold > 0",
                    Some("[]"),
                ),
                ("SELECT label, id FROM items", None),
                ("SELECT 42 AS constant FROM items", None),
            ] {
                let native = physical(&morsel, sql).await?;
                let display = displayable(native.as_ref()).indent(false).to_string();
                assert!(display.starts_with("MorselExec:"), "{display}");
                let sink = if workers == 1 {
                    "-> ResultSink[inline]"
                } else {
                    "-> ResultSink[capacity=2]"
                };
                assert!(display.contains(sink), "{sql}: {display}");
                if let Some(columns) = pruned_columns {
                    let (_, filter) = display
                        .split_once("-> Filter[")
                        .unwrap_or_else(|| panic!("missing filter for {sql}: {display}"));
                    let (filter, _) =
                        filter.split_once("] -> Projection[").unwrap_or_else(|| {
                            panic!("missing projection after filter for {sql}: {display}")
                        });
                    if workers == 1 {
                        assert!(
                            filter.ends_with(&format!(", projection={columns}")),
                            "{sql}: {display}"
                        );
                    } else {
                        assert!(!filter.contains(", projection=["), "{sql}: {display}");
                    }
                }
                if sql == FILTERED_QUERY {
                    assert!(display.contains("projection=Some([0, 1, 3])"), "{display}");
                }
                assert_eq!(native.output_partitioning().partition_count(), 1);
                assert!(native.output_ordering().is_none());
                let expected = physical(&default, sql).await?;
                assert_ne!(expected.name(), "MorselExec");
                let schema = native.schema();
                assert_eq!(schema, expected.schema());
                let actual_batches = collect(native, morsel.task_ctx()).await?;
                let expected_batches = collect(expected, default.task_ctx()).await?;
                assert!(
                    actual_batches
                        .iter()
                        .all(|batch| batch.schema_ref() == &schema)
                );
                if sql == FILTERED_QUERY {
                    let rows: usize =
                        actual_batches.iter().map(RecordBatch::num_rows).sum();
                    assert_eq!(rows, if empty { 0 } else { 24 });
                }
                assert_eq!(
                    sorted_rows(&actual_batches)?,
                    sorted_rows(&expected_batches)?,
                    "{sql}"
                );
            }
        }
    }
    Ok(())
}

fn count_input_observer() -> (ScalarUDF, mpsc::UnboundedReceiver<Id>) {
    let (observed, scans) = mpsc::unbounded_channel();
    let udf = create_udf(
        "observe_predicate",
        vec![DataType::Boolean],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(move |args| {
            if let Some(worker) = tokio::task::try_id() {
                let _ = observed.send(worker);
            }
            Ok(args[0].clone())
        }),
    );
    (udf, scans)
}

#[tokio::test]
async fn lowers_filtered_count_star() -> Result<()> {
    for (empty, sql, count) in [
        (true, "SELECT COUNT(*) FROM items WHERE threshold > 0", 0),
        (false, "SELECT COUNT(*) FROM items WHERE threshold > 100", 0),
        (
            false,
            "SELECT COUNT(*) AS matching_rows FROM items WHERE observe_predicate(threshold > 0)",
            24,
        ),
    ] {
        let table = items(empty)?;
        let default = SessionContext::new();
        default.register_table("items", table.clone())?;
        default.register_udf(count_input_observer().0);
        let expected = physical(&default, sql).await?;
        let schema = expected.schema();
        let expected_batches = collect(expected, default.task_ctx()).await?;
        for workers in [1, 4] {
            let morsel = morsel_context(workers);
            morsel
                .sql("SET datafusion.execution.morsel_mode = 'required'")
                .await?;
            morsel.register_table("items", table.clone())?;
            let (observer, mut scans) = count_input_observer();
            morsel.register_udf(observer);
            let native = physical(&morsel, sql).await?;
            let display = displayable(native.as_ref()).indent(false).to_string();
            assert!(display.starts_with("MorselExec:"), "{sql}: {display}");
            assert!(display.contains("-> Filter["), "{sql}: {display}");
            assert!(
                display.contains("-> CountSink; FinalizedCountSource"),
                "{sql}: {display}"
            );
            assert_eq!(native.pipeline_behavior(), EmissionType::Final);
            assert_eq!(native.schema(), schema, "{sql}");
            let execute = || collect(Arc::clone(&native), morsel.task_ctx());
            let batches = execute().await?;
            assert_eq!(batches, expected_batches, "{sql}");
            assert_eq!(batches.len(), 1, "{sql}");
            assert_eq!(batches[0].schema(), schema, "{sql}");
            let counts = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(counts, &Int64Array::from(vec![count]), "{sql}");
            if count > 0 {
                let mut seen = HashSet::new();
                while let Ok(worker) = scans.try_recv() {
                    seen.insert(worker);
                }
                if workers == 1 {
                    assert_eq!(seen.len(), 1);
                } else {
                    assert!(seen.len() > 1, "expected workers to share the scan");
                    let (a, b) = tokio::try_join!(execute(), execute())?;
                    assert_eq!(a, batches);
                    assert_eq!(b, batches);
                }
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn native_memory_filters_propagate_udf_length_errors() -> Result<()> {
    let ctx = morsel_context(1);
    ctx.sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    ctx.register_table("items", items(false)?)?;
    ctx.register_udf(create_udf(
        "oversized_mask",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(|_| {
            Ok(datafusion::logical_expr::ColumnarValue::Array(Arc::new(
                BooleanArray::from(vec![true; 4]),
            )))
        }),
    ));
    for sql in [
        "SELECT 42 AS constant FROM items WHERE oversized_mask(id)",
        "SELECT COUNT(*) FROM items WHERE oversized_mask(id)",
    ] {
        let plan = physical(&ctx, sql).await?;
        assert_eq!(plan.name(), "MorselExec");
        let error = collect(plan, ctx.task_ctx()).await.unwrap_err().to_string();
        assert!(
            error.contains(
                "UDF oversized_mask returned a different number of rows than expected"
            ),
            "{error}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn falls_back_to_the_default_planner_for_the_whole_query() -> Result<()> {
    let table = items(false)?;
    let default = SessionContext::new();
    let morsel = morsel_context(4);
    for ctx in [&default, &morsel] {
        ctx.register_table("items", table.clone())?;
        ctx.register_table(
            "unknown",
            Arc::new(EmptyTable::new(Arc::new(Schema::new(vec![Field::new(
                "id",
                DataType::Int32,
                true,
            )])))),
        )?;
    }
    for sql in [
        "SELECT id FROM items ORDER BY id DESC",
        "SELECT id FROM items LIMIT 3",
        "SELECT COUNT(*) FROM items",
        "SELECT COUNT(label) FROM items WHERE threshold > 0",
        "SELECT COUNT(DISTINCT id) FROM items WHERE threshold > 0",
        "SELECT COUNT(*) FILTER (WHERE id > 0) FROM items WHERE threshold > 0",
        "SELECT label, COUNT(*) FROM items WHERE threshold > 0 GROUP BY label",
        "SELECT id + 1 FROM unknown WHERE id > 0",
        "SELECT id, (SELECT MAX(id) FROM items) FROM items",
    ] {
        let fallback = physical(&morsel, sql).await?;
        let expected = physical(&default, sql).await?;
        let actual_display = displayable(fallback.as_ref()).indent(false).to_string();
        assert!(
            !actual_display.contains("MorselExec"),
            "{sql}: {actual_display}"
        );
        assert_eq!(
            actual_display,
            displayable(expected.as_ref()).indent(false).to_string(),
            "{sql}"
        );
    }
    Ok(())
}

// Hide the view's logical plan so the physical planner must call scan.
#[derive(Debug)]
struct NotInlinedView {
    view: ViewTable,
    session_id: String,
}

#[async_trait::async_trait]
impl TableProvider for NotInlinedView {
    fn schema(&self) -> SchemaRef {
        self.view.schema()
    }

    fn table_type(&self) -> TableType {
        self.view.table_type()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        assert_eq!(state.session_id(), self.session_id);
        self.view.scan(state, projection, filters, limit).await
    }
}

#[tokio::test]
async fn fallback_keeps_nested_view_planning_on_the_default_engine() -> Result<()> {
    let config = SessionConfig::new().set_bool("datafusion.explain.show_sizes", false);
    let default = SessionContext::new_with_config(config.clone());
    let morsel = morsel_context_with_config(4, config.clone());
    let prefer = SessionContext::new_with_config(config);
    prefer
        .sql("SET datafusion.execution.morsel_mode = 'prefer'")
        .await?;
    let table = items(false)?;
    for ctx in [&default, &morsel, &prefer] {
        ctx.register_table("items", table.clone())?;
        let plan = ctx
            .sql("SELECT id FROM items")
            .await?
            .into_optimized_plan()?;
        ctx.register_table(
            "wrapped",
            Arc::new(NotInlinedView {
                view: ViewTable::new(plan, None),
                session_id: ctx.session_id(),
            }),
        )?;
    }
    let sql = "SELECT id FROM wrapped ORDER BY id DESC";
    let expected = physical(&default, sql).await?;
    for ctx in [&morsel, &prefer] {
        let fallback = physical(ctx, sql).await?;
        let display = displayable(fallback.as_ref()).indent(false).to_string();
        assert!(!display.contains("MorselExec"), "{display}");
        assert!(!display.contains("partition_sizes="), "{display}");
        assert_eq!(
            display,
            displayable(expected.as_ref()).indent(false).to_string()
        );
        assert!(
            display.contains("SortExec") && display.contains("DataSourceExec"),
            "{display}"
        );
        assert_eq!(
            physical(ctx, "SELECT id FROM items").await?.name(),
            "MorselExec"
        );
    }
    prefer
        .sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    let err = physical(&prefer, "SELECT id FROM wrapped")
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("unsupported table provider for wrapped"),
        "{err}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_and_concurrent_execution_is_isolated() -> Result<()> {
    let ctx = morsel_context(4);
    ctx.register_table("items", items(false)?)?;
    let plan = physical(&ctx, FILTERED_QUERY).await?;
    assert_eq!(plan.name(), "MorselExec");
    let execute = || collect(Arc::clone(&plan), ctx.task_ctx());
    let first = sorted_rows(&execute().await?)?;
    assert_eq!(first.len(), 24);
    let (a, b) = tokio::try_join!(execute(), execute())?;
    assert_eq!(sorted_rows(&a)?, first);
    assert_eq!(sorted_rows(&b)?, first);
    assert_eq!(sorted_rows(&execute().await?)?, first);
    Ok(())
}
