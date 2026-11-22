// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(all(feature = "parquet", feature = "sql"))]

use std::fs::File;
use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::error::Result;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use tempfile::TempDir;

fn fixture() -> Result<TempDir> {
    fixture_with_files(2)
}

fn fixture_with_files(num_files: i32) -> Result<TempDir> {
    let dir = TempDir::new()?;
    for file in 0..num_files {
        let batch = RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(file * 8..file * 8 + 8)) as _,
            ),
            (
                "gate",
                Arc::new(Int32Array::from(vec![
                    Some(12),
                    None,
                    Some(8),
                    Some(15),
                    Some(1),
                    Some(4),
                    Some(25),
                    Some(3),
                ])) as _,
            ),
            (
                "payload",
                Arc::new(StringArray::from(vec![
                    "a", "b", "c", "d", "e", "f", "g", "h",
                ])) as _,
            ),
        ])?;
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("{file}.parquet")))?,
            batch.schema(),
            Some(properties),
        )?;
        writer.write(&batch)?;
        writer.close()?;
    }
    Ok(dir)
}

async fn context(dir: &TempDir, mode: &str, pushdown: bool) -> Result<SessionContext> {
    let mut config = SessionConfig::new().with_target_partitions(4);
    config.options_mut().execution.parquet.pushdown_filters = pushdown;
    // Exercise pruning after the footer, not listing-time file statistics.
    config.options_mut().execution.collect_statistics = false;
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet(
        "t",
        dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
        .await?;
    Ok(ctx)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn parallel_parquet_scan_claims_each_row_group_once() -> Result<()> {
    let file_count = 24;
    let dir = fixture_with_files(file_count)?;
    let native = context(&dir, "required", false).await?;
    native
        .sql("SET datafusion.execution.target_partitions = 8")
        .await?;
    let plan = native
        .sql("SELECT id FROM t")
        .await?
        .create_physical_plan()
        .await?;
    let display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(display.contains("workers=8, NativeParquet"), "{display}");
    let batches = collect(Arc::clone(&plan), native.task_ctx()).await?;
    let mut ids: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..file_count * 8).collect::<Vec<_>>());
    assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 96);
    Ok(())
}

#[tokio::test]
async fn filtered_count_uses_native_parquet_and_count_sink() -> Result<()> {
    let dir = fixture()?;
    let stock = context(&dir, "off", false).await?;
    let native = context(&dir, "required", false).await?;
    let sql = "SELECT COUNT(*) AS n FROM t WHERE gate > 10";
    let plan = native.sql(sql).await?.create_physical_plan().await?;
    let display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        display.contains("NativeParquet") && display.contains("CountSink"),
        "{display}"
    );
    let expected = stock.sql(sql).await?.collect().await?;
    assert_eq!(
        plan.schema().as_ref(),
        &Schema::new(vec![Field::new("n", DataType::Int64, false)])
    );
    let (actual, concurrent) = tokio::try_join!(
        collect(Arc::clone(&plan), native.task_ctx()),
        collect(Arc::clone(&plan), native.task_ctx()),
    )?;
    assert_eq!(actual, expected);
    assert_eq!(actual, concurrent);
    assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 12);
    assert_eq!(metric(plan.as_ref(), "count_finalizations"), 2);
    datafusion::assert_batches_eq!(
        ["+---+", "| n |", "+---+", "| 6 |", "+---+"],
        &actual
    );
    Ok(())
}

fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
    plan.metrics()
        .unwrap()
        .sum_by_name(name)
        .unwrap()
        .as_usize()
}

#[tokio::test]
async fn filtered_projection_matches_stock_with_or_without_row_pushdown() -> Result<()> {
    let dir = fixture()?;
    for pushdown in [false, true] {
        let stock = context(&dir, "off", pushdown).await?;
        let native = context(&dir, "required", pushdown).await?;
        let sql = "SELECT payload AS p, id FROM t WHERE gate > 10";
        let expected = stock.sql(sql).await?.create_physical_plan().await?;
        let actual = native.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(actual.schema(), expected.schema());
        let expected = collect(expected, stock.task_ctx()).await?;
        let batches = collect(Arc::clone(&actual), native.task_ctx()).await?;
        assert_eq!(
            batches_to_sort_string(&batches),
            batches_to_sort_string(&expected)
        );
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 6);
        let metrics = actual.metrics().unwrap();
        let Some(MetricValue::PruningMetrics {
            pruning_metrics, ..
        }) = metrics.sum_by_name("row_groups_pruned_statistics")
        else {
            panic!("expected native row-group pruning metrics: {metrics:?}");
        };
        assert_eq!(pruning_metrics.pruned(), 2);
        assert_eq!(metric(actual.as_ref(), "parquet_morsels_claimed"), 6);
    }
    Ok(())
}

#[tokio::test]
async fn parquet_filter_coalesces_across_row_groups() -> Result<()> {
    let dir = fixture()?;
    let native = context(&dir, "required", false).await?;
    native
        .sql("SET datafusion.execution.target_partitions = 1")
        .await?;
    native
        .sql("SET datafusion.execution.batch_size = 4")
        .await?;
    let sql = "SELECT payload AS p, id FROM t WHERE gate > 10";
    let plan = native.sql(sql).await?.create_physical_plan().await?;
    assert_eq!(plan.name(), "MorselExec");
    let batches = collect(Arc::clone(&plan), native.task_ctx()).await?;
    assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 6);
    assert_eq!(
        batches
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        [4, 2]
    );
    Ok(())
}

#[tokio::test]
async fn pruned_empty_and_zero_width_scans_preserve_row_counts() -> Result<()> {
    let dir = fixture()?;
    let native = context(&dir, "required", false).await?;
    let stock = context(&dir, "off", false).await?;
    let plan = native
        .sql("SELECT 7 AS x FROM t")
        .await?
        .create_physical_plan()
        .await?;
    let batches = collect(Arc::clone(&plan), native.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 16);
    assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 8);

    let empty = TempDir::new()?;
    let schema = Schema::new(vec![Field::new("gate", DataType::Int32, true)]);
    for ctx in [&stock, &native] {
        ctx.register_parquet(
            "empty",
            empty.path().to_str().unwrap(),
            ParquetReadOptions::default().schema(&schema),
        )
        .await?;
    }
    for sql in [
        "SELECT COUNT(*) AS n FROM t WHERE gate > 100",
        "SELECT COUNT(*) AS n FROM empty WHERE gate > 10",
    ] {
        let plan = native.sql(sql).await?.create_physical_plan().await?;
        let batches = collect(Arc::clone(&plan), native.task_ctx()).await?;
        assert_eq!(batches, stock.sql(sql).await?.collect().await?);
        datafusion::assert_batches_eq!(
            ["+---+", "| n |", "+---+", "| 0 |", "+---+"],
            &batches
        );
        assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 0);
        assert_eq!(metric(plan.as_ref(), "count_finalizations"), 1);
    }

    let sql = "SELECT COUNT(*) AS n FROM t WHERE gate % 100 = 99";
    let plan = native.sql(sql).await?.create_physical_plan().await?;
    let batches = collect(Arc::clone(&plan), native.task_ctx()).await?;
    assert_eq!(batches, stock.sql(sql).await?.collect().await?);
    assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 8);
    assert_eq!(metric(plan.as_ref(), "count_finalizations"), 1);
    Ok(())
}

#[tokio::test]
async fn parquet_projection_adapts_to_the_declared_table_schema() -> Result<()> {
    let dir = fixture()?;
    let native = context(&dir, "required", false).await?;
    let stock = context(&dir, "off", false).await?;
    let schema = Schema::new(vec![
        Field::new("gate", DataType::Int32, true),
        Field::new("missing", DataType::Utf8, true),
        Field::new("id", DataType::Int64, false),
    ]);
    for ctx in [&native, &stock] {
        ctx.register_parquet(
            "adapted",
            dir.path().to_str().unwrap(),
            ParquetReadOptions::default().schema(&schema),
        )
        .await?;
    }
    let sql = "SELECT missing, id FROM adapted WHERE gate > 10";
    let expected = stock.sql(sql).await?.create_physical_plan().await?;
    let actual = native.sql(sql).await?.create_physical_plan().await?;
    assert_eq!(actual.schema(), expected.schema());
    let expected = collect(expected, stock.task_ctx()).await?;
    let actual = collect(actual, native.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    Ok(())
}

#[tokio::test]
async fn unsupported_parquet_queries_delegate_only_in_prefer_mode() -> Result<()> {
    let dir = fixture()?;
    let stock = context(&dir, "off", false).await?;
    let prefer = context(&dir, "prefer", false).await?;
    let required = context(&dir, "required", false).await?;
    let partitioned = TempDir::new()?;
    std::fs::create_dir(partitioned.path().join("year=2026"))?;
    std::fs::copy(
        dir.path().join("0.parquet"),
        partitioned.path().join("year=2026/input.parquet"),
    )?;
    for ctx in [&stock, &prefer, &required] {
        ctx.register_parquet(
            "partitioned",
            partitioned.path().to_str().unwrap(),
            ParquetReadOptions::default()
                .table_partition_cols(vec![("year".into(), DataType::Int32)]),
        )
        .await?;
    }
    for (sql, reason) in [
        ("SELECT id FROM t ORDER BY id DESC", "Sort"),
        ("SELECT id FROM t OFFSET 1", "OFFSET"),
        ("SELECT id, year FROM partitioned", "partition columns"),
        ("SELECT input_file_name() FROM t", "projection expression"),
    ] {
        let err = required
            .sql(sql)
            .await?
            .create_physical_plan()
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("morsel_mode=required")
                && err.to_lowercase().contains(&reason.to_lowercase()),
            "{sql}: {err}"
        );
        let plan = prefer.sql(sql).await?.create_physical_plan().await?;
        assert!(
            !displayable(plan.as_ref())
                .indent(false)
                .to_string()
                .contains("MorselExec")
        );
        let actual = collect(plan, prefer.task_ctx()).await?;
        let expected = stock.sql(sql).await?.collect().await?;
        if reason == "OFFSET" {
            // The particular omitted row is unspecified without ORDER BY.
            assert_eq!(actual.iter().map(RecordBatch::num_rows).sum::<usize>(), 15);
        } else {
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected)
            );
        }
    }
    Ok(())
}
