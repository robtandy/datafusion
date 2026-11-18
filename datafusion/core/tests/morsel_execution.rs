// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(feature = "sql")]

use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion::datasource::{MemTable, empty::EmptyTable};
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::morsel::MorselQueryPlanner;
use datafusion::physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, collect, displayable,
};
use datafusion::prelude::SessionContext;

const FILTERED_QUERY: &str =
    "SELECT id + 1 AS result, label FROM items WHERE threshold > 0";

fn morsel_context(workers: usize) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
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
            morsel.register_table("items", table.clone())?;
            for sql in [
                FILTERED_QUERY,
                "SELECT label, id FROM items",
                "SELECT 42 AS constant FROM items",
            ] {
                let native = physical(&morsel, sql).await?;
                let display = displayable(native.as_ref()).indent(false).to_string();
                assert!(display.starts_with("MorselExec:"), "{display}");
                if sql == FILTERED_QUERY {
                    assert!(display.contains("projection=Some([0, 1, 3])"), "{display}");
                    assert!(
                        display.contains("-> Filter[")
                            && display.contains("-> Projection[")
                            && display.contains("-> ResultSink["),
                        "{display}"
                    );
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
