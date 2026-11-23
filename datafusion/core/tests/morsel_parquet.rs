// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(all(feature = "parquet", feature = "sql"))]

use std::collections::HashSet;
use std::fs::File;
use std::sync::{Arc, Barrier};
use std::thread::ThreadId;
use std::time::Duration;

use arrow::array::{
    Array, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::{EnabledStatistics, WriterProperties};
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::StreamExt;
use parking_lot::Mutex;
use tempfile::TempDir;
use tokio::time::timeout;

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
        .and_then(|m| m.sum_by_name(name))
        .map_or(0, |v| v.as_usize())
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
async fn pushdown_filters_evaluates_parquet_row_filter_and_skips_unmatched_columns()
-> Result<()> {
    let dir = TempDir::new()?;
    let payload_text = "x".repeat(512);
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(Int32Array::from_iter_values(0..64)) as _),
        ("gate", Arc::new(Int32Array::from_iter_values(0..64)) as _),
        (
            "payload",
            Arc::new(StringArray::from(vec![payload_text.as_str(); 64])) as _,
        ),
    ])?;
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(16))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("0.parquet"))?,
        batch.schema(),
        Some(properties),
    )?;
    writer.write(&batch)?;
    writer.close()?;

    // Modulo predicate cannot be pruned by row-group min/max statistics, so
    // every row group survives to RowFilter and matches zero rows.
    let sql = "SELECT id, payload FROM t WHERE gate % 100 = 99";
    let without = context(&dir, "required", false).await?;
    let with = context(&dir, "required", true).await?;

    let plan_without = without.sql(sql).await?.create_physical_plan().await?;
    let plan_with = with.sql(sql).await?.create_physical_plan().await?;

    let batches_without = collect(Arc::clone(&plan_without), without.task_ctx()).await?;
    let batches_with = collect(Arc::clone(&plan_with), with.task_ctx()).await?;
    assert_eq!(
        batches_without
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        0
    );
    assert_eq!(
        batches_with
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        0
    );

    let display_with = displayable(plan_with.as_ref()).indent(false).to_string();
    assert!(
        display_with.contains("NativeParquet[t, predicate=")
            && !display_with.contains("Filter["),
        "expected pushed predicate in NativeParquet display: {display_with}"
    );
    assert_eq!(metric(plan_without.as_ref(), "pushdown_rows_pruned"), 0);
    assert_eq!(metric(plan_with.as_ref(), "pushdown_rows_pruned"), 64);
    assert!(
        metric(plan_with.as_ref(), "bytes_scanned")
            < metric(plan_without.as_ref(), "bytes_scanned"),
        "expected RowFilter to skip fetching non-predicate columns for unmatched row groups: with={} without={}",
        metric(plan_with.as_ref(), "bytes_scanned"),
        metric(plan_without.as_ref(), "bytes_scanned"),
    );
    Ok(())
}

#[tokio::test]
async fn parquet_filter_coalesces_across_row_groups() -> Result<()> {
    let dir = fixture()?;
    for pushdown in [false, true] {
        let native = context(&dir, "required", pushdown).await?;
        native
            .sql("SET datafusion.execution.target_partitions = 1")
            .await?;
        native
            .sql("SET datafusion.execution.batch_size = 4")
            .await?;
        let sql = "SELECT payload AS p, id FROM t WHERE gate > 10";
        let plan = native.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let display = displayable(plan.as_ref()).indent(false).to_string();
        if pushdown {
            assert!(
                display.contains("NativeParquet[t, predicate=gate@1 > 10]")
                    && !display.contains("Filter["),
                "{display}"
            );
        } else {
            assert!(display.contains("NativeParquet[t] -> Filter["), "{display}");
        }
        let batches = collect(Arc::clone(&plan), native.task_ctx()).await?;
        assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 6);
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            [4, 2],
            "pushdown={pushdown}"
        );
    }
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

#[tokio::test]
async fn pushdown_filters_parity_across_selectivities_nulls_and_count() -> Result<()> {
    let dir = TempDir::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("score", DataType::Int32, true),
        Field::new("delta", DataType::Int32, true),
        Field::new("payload", DataType::Int64, true),
    ]));

    // Write 2 files, each with 4 distinct row groups exercising:
    // - RG 0: fully matched for `score > 100` (all non-null values in 200..216)
    // - RG 1: partially matched with NULLs in both `score` and `delta`
    // - RG 2: all-pruned by stats for `score > 100` (values in 0..16)
    // - RG 3: straddles stats bounds, selectively filtered by modulo / NULL checks
    for file in 0..2i64 {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(16))
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("{file}.parquet")))?,
            Arc::clone(&schema),
            Some(properties),
        )?;
        let base = file * 64;
        type ScoreDeltaGroup = (Vec<Option<i32>>, Vec<Option<i32>>);
        let groups: Vec<ScoreDeltaGroup> = vec![
            (
                (0..16).map(|i| Some(200 + i)).collect(),
                (0..16).map(|i| Some(10 + i)).collect(),
            ),
            (
                (0..16)
                    .map(|i| if i % 4 == 0 { None } else { Some(i * 15) })
                    .collect(),
                (0..16)
                    .map(|i| if i % 3 == 0 { None } else { Some(i * 7) })
                    .collect(),
            ),
            (
                (0..16).map(Some).collect(),
                (0..16).map(|i| Some(i * 5)).collect(),
            ),
            (
                (0..16).map(|i| Some(50 + i * 10)).collect(),
                (0..16)
                    .map(|i| if i == 5 { None } else { Some(i * 13) })
                    .collect(),
            ),
        ];
        for (rg_idx, (scores, deltas)) in groups.into_iter().enumerate() {
            let start = base + (rg_idx as i64) * 16;
            let row_ids: Vec<i64> = (start..start + 16).collect();
            let payloads: Vec<Option<i64>> = row_ids
                .iter()
                .map(|&id| if id % 5 == 0 { None } else { Some(id * 11) })
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(row_ids)),
                    Arc::new(Int32Array::from(scores)),
                    Arc::new(Int32Array::from(deltas)),
                    Arc::new(Int64Array::from(payloads)),
                ],
            )?;
            writer.write(&batch)?;
            writer.flush()?;
        }
        writer.close()?;
    }

    let queries = [
        // Low selectivity multi-column arithmetic predicate
        "SELECT row_id, payload + delta AS projected FROM t WHERE (score + 3 * delta) % 1000 < 50",
        // High selectivity multi-column arithmetic predicate
        "SELECT row_id, payload + delta AS projected FROM t WHERE (score + 3 * delta) % 1000 < 650",
        // Mix of fully-matched, partially-matched (with NULLs), and stats-pruned RGs
        "SELECT row_id, payload FROM t WHERE score > 100",
        // Multi-conjunct predicate
        "SELECT row_id, payload FROM t WHERE score > 50 AND delta < 100",
        // Nullable predicates
        "SELECT row_id FROM t WHERE score IS NULL",
        "SELECT row_id, score FROM t WHERE score IS NOT NULL AND delta IS NULL",
        "SELECT row_id FROM t WHERE COALESCE(score, 0) > 100",
        // Filtered COUNT(*) across low, high, and all-filtered predicates
        "SELECT COUNT(*) AS n FROM t WHERE (score + 3 * delta) % 1000 < 50",
        "SELECT COUNT(*) AS n FROM t WHERE (score + 3 * delta) % 1000 < 650",
        "SELECT COUNT(*) AS n FROM t WHERE score > 100",
        "SELECT COUNT(*) AS n FROM t WHERE score % 1000 = 999",
    ];

    for pushdown in [false, true] {
        for reorder in [false, true] {
            let stock = context(&dir, "off", pushdown).await?;
            let native = context(&dir, "required", pushdown).await?;
            for ctx in [&stock, &native] {
                ctx.sql(&format!(
                    "SET datafusion.execution.parquet.reorder_filters = {reorder}"
                ))
                .await?;
            }
            for sql in queries {
                let expected_plan = stock.sql(sql).await?.create_physical_plan().await?;
                let actual_plan = native.sql(sql).await?.create_physical_plan().await?;
                assert_eq!(actual_plan.schema(), expected_plan.schema(), "{sql}");
                let expected = collect(expected_plan, stock.task_ctx()).await?;
                let actual = collect(Arc::clone(&actual_plan), native.task_ctx()).await?;
                assert_eq!(
                    batches_to_sort_string(&actual),
                    batches_to_sort_string(&expected),
                    "parity mismatch (pushdown={pushdown}, reorder={reorder}): {sql}"
                );
                if pushdown && sql == "SELECT row_id, payload FROM t WHERE score > 100" {
                    assert!(
                        metric(actual_plan.as_ref(), "row_filter_skipped_fully_matched")
                            >= 2,
                        "expected fully-matched RGs to skip RowFilter"
                    );
                    assert!(
                        metric(actual_plan.as_ref(), "pushdown_rows_pruned") > 0,
                        "expected non-fully-matched RGs to run RowFilter"
                    );
                }
                if pushdown
                    && sql
                        == "SELECT COUNT(*) AS n FROM t WHERE (score + 3 * delta) % 1000 < 50"
                {
                    let display =
                        displayable(actual_plan.as_ref()).indent(false).to_string();
                    assert!(
                        display.contains("NativeParquet[t, predicate=")
                            && display.contains("CountSink")
                            && !display.contains("Filter["),
                        "{display}"
                    );
                    assert!(metric(actual_plan.as_ref(), "pushdown_rows_pruned") > 0);
                }
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn pushdown_filters_keeps_residual_predicates_and_adapts_schema() -> Result<()> {
    let dir = TempDir::new()?;
    let person_fields = Fields::from(vec![Field::new("name", DataType::Utf8, true)]);
    let names = Arc::new(StringArray::from(vec![
        Some("a"),
        Some("b"),
        None,
        Some("d"),
        Some("e"),
        Some("f"),
        Some("g"),
        Some("h"),
    ]));
    let person = StructArray::try_new(
        person_fields.clone(),
        vec![names],
        Some(vec![true, false, true, true, true, false, true, true].into()),
    )?;
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(Int32Array::from_iter_values(0..8)) as _),
        (
            "gate",
            Arc::new(Int32Array::from(vec![
                Some(12),
                Some(15),
                None,
                Some(20),
                Some(3),
                Some(18),
                Some(25),
                Some(2),
            ])) as _,
        ),
        ("person", Arc::new(person) as _),
    ])?;
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(4))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("0.parquet"))?,
        batch.schema(),
        Some(properties),
    )?;
    writer.write(&batch)?;
    writer.close()?;

    for pushdown in [false, true] {
        let stock = context(&dir, "off", pushdown).await?;
        let native = context(&dir, "required", pushdown).await?;

        // Partially pushable: `gate > 10` pushes to RowFilter, whole-struct
        // `person IS NOT NULL` stays as a residual native Filter operator.
        for sql in [
            "SELECT id, person FROM t WHERE gate > 10 AND person IS NOT NULL",
            "SELECT COUNT(*) AS n FROM t WHERE gate > 10 AND person IS NOT NULL",
            "SELECT id FROM t WHERE person IS NOT NULL",
            "SELECT id FROM t WHERE gate > 10 AND random() >= 0.0",
        ] {
            let expected_plan = stock.sql(sql).await?.create_physical_plan().await?;
            let actual_plan = native.sql(sql).await?.create_physical_plan().await?;
            assert_eq!(actual_plan.schema(), expected_plan.schema(), "{sql}");
            let expected = collect(expected_plan, stock.task_ctx()).await?;
            let actual = collect(Arc::clone(&actual_plan), native.task_ctx()).await?;
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected),
                "pushdown={pushdown}: {sql}"
            );
            let display = displayable(actual_plan.as_ref()).indent(false).to_string();
            assert!(
                display.contains("Filter["),
                "expected residual Filter operator for {sql}: {display}"
            );
            if pushdown && sql.contains("gate > 10") {
                assert!(
                    metric(actual_plan.as_ref(), "pushdown_rows_pruned") > 0,
                    "expected pushable conjunct to execute in RowFilter for {sql}"
                );
            }
        }
    }

    // Schema adaptation (missing column + Int32->Int64 widening) under pushdown_filters=true.
    let adapted_schema = Schema::new(vec![
        Field::new("gate", DataType::Int32, true),
        Field::new("missing", DataType::Utf8, true),
        Field::new("id", DataType::Int64, false),
    ]);
    let stock = context(&dir, "off", true).await?;
    let native = context(&dir, "required", true).await?;
    for ctx in [&stock, &native] {
        ctx.register_parquet(
            "adapted",
            dir.path().to_str().unwrap(),
            ParquetReadOptions::default().schema(&adapted_schema),
        )
        .await?;
    }
    for sql in [
        "SELECT missing, id FROM adapted WHERE gate > 10",
        "SELECT id FROM adapted WHERE missing IS NULL",
        "SELECT id FROM adapted WHERE missing = 'never'",
        "SELECT id FROM adapted WHERE id > 3",
    ] {
        let expected = stock.sql(sql).await?.collect().await?;
        let actual = native.sql(sql).await?.collect().await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "{sql}"
        );
    }
    Ok(())
}

#[derive(Debug)]
struct BarrierGateUdf {
    signature: Signature,
    barrier: Arc<Barrier>,
    threads: Arc<Mutex<HashSet<ThreadId>>>,
}

impl PartialEq for BarrierGateUdf {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.barrier, &other.barrier)
    }
}

impl Eq for BarrierGateUdf {}

impl std::hash::Hash for BarrierGateUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.barrier).hash(state);
    }
}

impl ScalarUDFImpl for BarrierGateUdf {
    fn name(&self) -> &str {
        "barrier_gate"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        self.threads.lock().insert(std::thread::current().id());
        // Synchronize all 4 row groups concurrently. A single-file scan has
        // only 1 producer lane, so this barrier can only complete if RowFilter
        // executes concurrently on the 4 pipeline workers.
        self.barrier.wait();
        let arr = args.args[0].clone().into_array(args.number_rows)?;
        let ints = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        let mask: BooleanArray = ints.iter().map(|v| v.map(|x| x > 10)).collect();
        Ok(ColumnarValue::Array(Arc::new(mask)))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pushdown_filters_runs_row_filter_concurrently_on_workers_not_producer()
-> Result<()> {
    let dir = TempDir::new()?;
    // One file with 4 row groups: producer spawns min(workers, files) = 1 lane.
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(Int32Array::from_iter_values(0..16)) as _),
        (
            "gate",
            Arc::new(Int32Array::from_iter_values([
                1, 15, 2, 20, 3, 25, 4, 30, 5, 35, 6, 40, 7, 45, 8, 50,
            ])) as _,
        ),
        (
            "payload",
            Arc::new(StringArray::from(vec!["payload"; 16])) as _,
        ),
    ])?;
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(4))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("single.parquet"))?,
        batch.schema(),
        Some(properties),
    )?;
    writer.write(&batch)?;
    writer.close()?;

    let threads = Arc::new(Mutex::new(HashSet::new()));
    let udf = ScalarUDF::from(BarrierGateUdf {
        signature: Signature::exact(vec![DataType::Int32], Volatility::Immutable),
        barrier: Arc::new(Barrier::new(4)),
        threads: Arc::clone(&threads),
    });

    let native = context(&dir, "required", true).await?;
    native.register_udf(udf);
    let sql = "SELECT id, payload FROM t WHERE barrier_gate(gate)";
    let plan = native.sql(sql).await?.create_physical_plan().await?;
    let display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        display.contains("workers=4, NativeParquet[t, predicate=")
            && !display.contains("Filter["),
        "expected UDF predicate pushed into NativeParquet RowFilter: {display}"
    );

    let batches = timeout(
        Duration::from_secs(10),
        collect(Arc::clone(&plan), native.task_ctx()),
    )
    .await
    .expect("RowFilter deadlocked: must run concurrently across workers, not on 1 producer lane")?;

    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 3, 5, 7, 9, 11, 13, 15]);
    assert_eq!(metric(plan.as_ref(), "pushdown_rows_pruned"), 8);
    assert_eq!(metric(plan.as_ref(), "pushdown_rows_matched"), 8);
    assert_eq!(
        threads.lock().len(),
        4,
        "expected RowFilter to execute concurrently across 4 distinct worker threads"
    );
    Ok(())
}

#[tokio::test]
async fn pushdown_filters_skips_unmatched_pages_with_page_index() -> Result<()> {
    let dir = TempDir::new()?;
    let rows = 1024;
    // Put 0 and 1000 in every 32-row page so page min/max stats cannot prune
    // any page for `gate = 777`, leaving only RowFilter to narrow the selection
    // to page 0 and skip fetching pages 1..31 of `payload` via offset indexes.
    let gates: Vec<i32> = (0..rows)
        .map(|i| match i % 32 {
            0 => 0,
            1 => 1000,
            2 if i == 2 => 777,
            _ => 10,
        })
        .collect();
    let payloads: Vec<String> = (0..rows)
        .map(|i| format!("{i:08}-{}", "p".repeat(248)))
        .collect();
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(Int32Array::from_iter_values(0..rows)) as _),
        ("gate", Arc::new(Int32Array::from(gates)) as _),
        ("payload", Arc::new(StringArray::from(payloads)) as _),
    ])?;
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(rows as usize))
        .set_write_batch_size(16)
        .set_data_page_row_count_limit(32)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("pages.parquet"))?,
        batch.schema(),
        Some(properties),
    )?;
    writer.write(&batch)?;
    writer.close()?;

    let sql = "SELECT id, payload FROM t WHERE gate = 777";
    let stock_push = context(&dir, "off", true).await?;
    let native_no_push = context(&dir, "required", false).await?;
    let native_push = context(&dir, "required", true).await?;
    for ctx in [&stock_push, &native_no_push, &native_push] {
        ctx.sql("SET datafusion.execution.parquet.enable_page_index = true")
            .await?;
    }

    let stock_plan = stock_push.sql(sql).await?.create_physical_plan().await?;
    let no_push_plan = native_no_push
        .sql(sql)
        .await?
        .create_physical_plan()
        .await?;
    let push_plan = native_push.sql(sql).await?.create_physical_plan().await?;

    let expected = collect(Arc::clone(&stock_plan), stock_push.task_ctx()).await?;
    let without = collect(Arc::clone(&no_push_plan), native_no_push.task_ctx()).await?;
    let with = collect(Arc::clone(&push_plan), native_push.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&with),
        batches_to_sort_string(&expected)
    );
    assert_eq!(
        batches_to_sort_string(&without),
        batches_to_sort_string(&expected)
    );
    assert_eq!(with.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(metric(push_plan.as_ref(), "pushdown_rows_pruned"), 1023);
    assert_eq!(metric(push_plan.as_ref(), "pushdown_rows_matched"), 1);

    let bytes_no_push = metric(no_push_plan.as_ref(), "bytes_scanned");
    let bytes_push = metric(push_plan.as_ref(), "bytes_scanned");
    let bytes_stock_push = metric(stock_plan.as_ref(), "bytes_scanned");
    assert!(
        bytes_push < bytes_no_push / 2,
        "expected RowFilter + offset index to skip unmatched payload pages: push={bytes_push} no_push={bytes_no_push}"
    );
    assert_eq!(
        bytes_push, bytes_stock_push,
        "expected native pushdown bytes_scanned to match stock pushdown"
    );
    Ok(())
}

#[tokio::test]
async fn pushdown_filters_handles_worker_errors_and_early_cancellation() -> Result<()> {
    let dir = TempDir::new()?;
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(Int32Array::from_iter_values(0..16)) as _),
        (
            "gate",
            Arc::new(Int32Array::from_iter_values([
                5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            ])) as _,
        ),
    ])?;
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(4))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("0.parquet"))?,
        batch.schema(),
        Some(properties),
    )?;
    writer.write(&batch)?;
    writer.close()?;

    let native = context(&dir, "required", true).await?;
    // Division by zero inside pushed-down RowFilter must propagate cleanly without finalizing COUNT(*).
    let err_plan = native
        .sql("SELECT COUNT(*) AS n FROM t WHERE 100 / gate > 1")
        .await?
        .create_physical_plan()
        .await?;
    let err = collect(Arc::clone(&err_plan), native.task_ctx())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.to_lowercase().contains("divide by zero"),
        "expected divide-by-zero error from RowFilter, got: {err}"
    );
    assert_eq!(metric(err_plan.as_ref(), "count_finalizations"), 0);

    // Early stream cancellation on a multi-row-group pushdown scan.
    let multi = fixture_with_files(8)?;
    let native = context(&multi, "required", true).await?;
    let plan = native
        .sql("SELECT id, payload FROM t WHERE gate > 10")
        .await?
        .create_physical_plan()
        .await?;
    let mut stream = execute_stream(Arc::clone(&plan), native.task_ctx())?;
    let first = stream.next().await.expect("at least one batch")?;
    assert!(first.num_rows() > 0);
    drop(stream);
    Ok(())
}
