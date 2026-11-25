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
use datafusion::datasource::physical_plan::parquet::ParquetSplitConfig;
use datafusion::error::Result;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::{
    ColumnarValue, ScalarUDF, ScalarUDFImpl, Signature, Volatility, create_udf,
};
use datafusion::morsel::MorselMetrics;
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
        ("SELECT id FROM t LIMIT 3", "LIMIT"),
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
        } else if reason == "LIMIT" {
            // The particular returned rows are unspecified without ORDER BY.
            assert_eq!(actual.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
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

#[tokio::test]
async fn parquet_count_star_null_predicates_and_edge_row_groups() -> Result<()> {
    let dir = TempDir::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("gate", DataType::Int32, true),
        Field::new("delta", DataType::Int32, true),
    ]));

    // Write 2 files with 4 distinct row groups each:
    // RG 0: all NULL `gate`
    // RG 1: mixed NULL, matching (>10), and non-matching
    // RG 2: all non-matching (values in 0..4, stats-prunable for gate > 10)
    // RG 3: all matching (values in 20..24)
    for file in 0..2 {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4))
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("{file}.parquet")))?,
            Arc::clone(&schema),
            Some(properties),
        )?;
        let groups: [Vec<Option<i32>>; 4] = [
            vec![None, None, None, None],
            vec![Some(15), None, Some(5), Some(25)],
            vec![Some(0), Some(1), Some(2), Some(3)],
            vec![Some(20), Some(21), Some(22), Some(23)],
        ];
        for (rg, gates) in groups.into_iter().enumerate() {
            let base = file * 16 + (rg as i32) * 4;
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from_iter_values(base..base + 4)),
                    Arc::new(Int32Array::from(gates)),
                    Arc::new(Int32Array::from(vec![Some(1), None, Some(2), Some(3)])),
                ],
            )?;
            writer.write(&batch)?;
            writer.flush()?;
        }
        writer.close()?;
    }

    let empty = TempDir::new()?;
    let queries = [
        "SELECT COUNT(*) AS n FROM t WHERE gate > 10",
        "SELECT COUNT(*) AS n FROM t WHERE NOT (gate > 10)",
        "SELECT COUNT(*) AS n FROM t WHERE gate IS NULL",
        "SELECT COUNT(*) AS n FROM t WHERE gate IS NOT NULL",
        "SELECT COUNT(*) AS n FROM t WHERE gate = gate",
        "SELECT COUNT(*) AS n FROM t WHERE NULLIF(gate, gate) > 0",
        "SELECT COUNT(*) AS n FROM t WHERE COALESCE(gate, 0) > 10",
        "SELECT COUNT(*) AS n FROM t WHERE gate > 10 AND delta IS NULL",
        "SELECT COUNT(*) AS n FROM t WHERE gate % 100 = 99",
        "SELECT COUNT(*) + 5 AS n FROM t WHERE gate > 10",
        "SELECT COUNT(*) AS n FROM empty WHERE gate > 10",
    ];

    for pushdown in [false, true] {
        for partitions in [1, 4] {
            let stock = context(&dir, "off", pushdown).await?;
            let native = context(&dir, "required", pushdown).await?;
            for ctx in [&stock, &native] {
                ctx.sql(&format!(
                    "SET datafusion.execution.target_partitions = {partitions}"
                ))
                .await?;
                ctx.register_parquet(
                    "empty",
                    empty.path().to_str().unwrap(),
                    ParquetReadOptions::default().schema(schema.as_ref()),
                )
                .await?;
            }
            for sql in queries {
                let expected = stock.sql(sql).await?.collect().await?;
                let plan = native.sql(sql).await?.create_physical_plan().await?;
                let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
                assert_eq!(
                    actual, expected,
                    "parity mismatch (pushdown={pushdown}, partitions={partitions}): {sql}"
                );
                assert_eq!(metric(plan.as_ref(), "count_finalizations"), 1);
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn parquet_count_star_pushed_residual_errors_and_cancellation() -> Result<()> {
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
        person_fields,
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
                Some(0),
                Some(20),
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

    let fail_on_twenty = create_udf(
        "fail_on_twenty",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(|args| {
            let arr = args[0].clone().into_array(1)?;
            let ints = arr.as_any().downcast_ref::<Int32Array>().unwrap();
            if ints.iter().any(|v| v == Some(20)) {
                return datafusion::common::exec_err!("volatile residual failure");
            }
            let mask: BooleanArray = ints.iter().map(|v| v.map(|x| x > 10)).collect();
            Ok(ColumnarValue::Array(Arc::new(mask)))
        }),
    );

    for pushdown in [false, true] {
        let stock = context(&dir, "off", pushdown).await?;
        let native = context(&dir, "required", pushdown).await?;
        native.register_udf(fail_on_twenty.clone());

        for sql in [
            "SELECT COUNT(*) AS n FROM t WHERE gate > 10",
            "SELECT COUNT(*) AS n FROM t WHERE gate > 10 AND person IS NOT NULL",
            "SELECT COUNT(*) AS n FROM t WHERE gate > 10 AND random() >= 0.0",
        ] {
            let expected = stock.sql(sql).await?.collect().await?;
            let plan = native.sql(sql).await?.create_physical_plan().await?;
            let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
            assert_eq!(actual, expected, "pushdown={pushdown}: {sql}");
            assert_eq!(metric(plan.as_ref(), "count_finalizations"), 1);
        }

        // Division by zero inside direct-mask COUNT(*) (pushdown=false) or RowFilter (pushdown=true).
        let div_plan = native
            .sql("SELECT COUNT(*) AS n FROM t WHERE 100 / gate > 1")
            .await?
            .create_physical_plan()
            .await?;
        let err = collect(Arc::clone(&div_plan), native.task_ctx())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.to_lowercase().contains("divide by zero"),
            "pushdown={pushdown}: {err}"
        );
        assert_eq!(metric(div_plan.as_ref(), "count_finalizations"), 0);

        // Volatile residual UDF failure on direct-mask COUNT(*) (stays residual under pushdown=true).
        let res_plan = native
            .sql("SELECT COUNT(*) AS n FROM t WHERE gate > 10 AND fail_on_twenty(gate)")
            .await?
            .create_physical_plan()
            .await?;
        let display = displayable(res_plan.as_ref()).indent(false).to_string();
        assert!(
            display.contains("Filter["),
            "pushdown={pushdown}: {display}"
        );
        let err = collect(Arc::clone(&res_plan), native.task_ctx())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("volatile residual failure"),
            "pushdown={pushdown}: {err}"
        );
        assert_eq!(metric(res_plan.as_ref(), "count_finalizations"), 0);
    }

    // Early stream drop on multi-file Parquet COUNT(*) tears down cleanly.
    let multi = fixture_with_files(8)?;
    for pushdown in [false, true] {
        let native = context(&multi, "required", pushdown).await?;
        let plan = native
            .sql("SELECT COUNT(*) AS n FROM t WHERE gate > 10")
            .await?
            .create_physical_plan()
            .await?;
        let stream = execute_stream(Arc::clone(&plan), native.task_ctx())?;
        drop(stream);
    }
    Ok(())
}

#[tokio::test]
async fn parquet_filter_projection_prunes_filter_only_columns() -> Result<()> {
    let dir = fixture()?;
    for pushdown in [false, true] {
        for partitions in [1, 4] {
            let stock = context(&dir, "off", pushdown).await?;
            let native = context(&dir, "required", pushdown).await?;
            for ctx in [&stock, &native] {
                ctx.sql(&format!(
                    "SET datafusion.execution.target_partitions = {partitions}"
                ))
                .await?;
            }
            for (sql, expected_filter_proj) in [
                ("SELECT payload AS p, id FROM t WHERE gate > 10", "[0, 2]"),
                ("SELECT 42 AS c FROM t WHERE gate > 10", "[]"),
            ] {
                let expected = stock.sql(sql).await?.collect().await?;
                let plan = native.sql(sql).await?.create_physical_plan().await?;
                let display = displayable(plan.as_ref()).indent(false).to_string();
                if !pushdown {
                    assert!(
                        display
                            .contains(&format!(", projection={expected_filter_proj}]")),
                        "expected filter column pruning for {sql}: {display}"
                    );
                } else {
                    assert!(
                        !display.contains("Filter["),
                        "expected pushed predicate for {sql}: {display}"
                    );
                }
                let actual = collect(plan, native.task_ctx()).await?;
                assert_eq!(
                    batches_to_sort_string(&actual),
                    batches_to_sort_string(&expected),
                    "pushdown={pushdown}, partitions={partitions}: {sql}"
                );
            }
        }
    }
    Ok(())
}

fn split_fixture(
    files: usize,
    rgs_per_file: usize,
    rows_per_rg: usize,
    write_page_index: bool,
) -> Result<TempDir> {
    let dir = TempDir::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int32, true),
        Field::new("delta", DataType::Int32, true),
        Field::new("payload", DataType::Int64, true),
        Field::new("unused", DataType::Int64, false),
    ]));
    let stats = if write_page_index {
        EnabledStatistics::Page
    } else {
        EnabledStatistics::Chunk
    };
    let mut row_start = 0i64;
    for file in 0..files {
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_rg))
            .set_write_batch_size(35)
            .set_data_page_row_count_limit(70)
            .set_data_page_size_limit(256)
            .set_dictionary_page_size_limit(256)
            .set_dictionary_enabled(true)
            .set_statistics_enabled(stats)
            .set_offset_index_disabled(!write_page_index)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("part-{file:03}.parquet")))?,
            Arc::clone(&schema),
            Some(props),
        )?;
        for _ in 0..rgs_per_file {
            let ids: Vec<i64> = (row_start..row_start + rows_per_rg as i64).collect();
            let scores: Vec<Option<i32>> = ids
                .iter()
                .map(|&id| (id % 19 != 0).then_some(((id * 17) % 100) as i32))
                .collect();
            // Low cardinality so delta uses dictionary encoding with multiple pages.
            let deltas: Vec<Option<i32>> = ids
                .iter()
                .map(|&id| (id % 23 != 0).then_some((id % 7) as i32))
                .collect();
            let payloads: Vec<Option<i64>> = ids
                .iter()
                .map(|&id| (id % 11 != 0).then_some(id * 31 + 7))
                .collect();
            let unused: Vec<i64> = ids.iter().map(|&id| id * 101).collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(Int32Array::from(scores)),
                    Arc::new(Int32Array::from(deltas)),
                    Arc::new(Int64Array::from(payloads)),
                    Arc::new(Int64Array::from(unused)),
                ],
            )?;
            writer.write(&batch)?;
            writer.flush()?;
            row_start += rows_per_rg as i64;
        }
        writer.close()?;
    }
    Ok(dir)
}

#[expect(clippy::too_many_arguments)]
async fn split_context(
    dir: &TempDir,
    mode: &str,
    workers: usize,
    pushdown: bool,
    enable_page_index: bool,
    split_config: Option<ParquetSplitConfig>,
    metrics: Option<Arc<MorselMetrics>>,
    pool: Option<Arc<dyn MemoryPool>>,
) -> Result<SessionContext> {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(128);
    config.options_mut().execution.parquet.pushdown_filters = pushdown;
    config.options_mut().execution.parquet.enable_page_index = enable_page_index;
    config.options_mut().execution.collect_statistics = false;
    if let Some(split_cfg) = split_config {
        config.set_extension(Arc::new(split_cfg));
    }
    if let Some(m) = metrics {
        config.set_extension(m);
    }
    let ctx = if let Some(pool) = pool {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()?;
        SessionContext::new_with_config_rt(config, runtime)
    } else {
        SessionContext::new_with_config(config)
    };
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
async fn intra_rg_splitting_parity_across_boundaries_filters_nulls_and_pushdown()
-> Result<()> {
    // 2 files x 2 large RGs x 2400 rows = 9600 rows; 300 rows/slice -> 32 slices.
    let dir = split_fixture(2, 2, 2400, true)?;
    let split_cfg = ParquetSplitConfig {
        enabled: true,
        min_row_group_rows: 800,
        target_slice_rows: 300,
        ..Default::default()
    };

    for pushdown in [false, true] {
        let stock =
            split_context(&dir, "off", 8, pushdown, true, None, None, None).await?;
        let metrics = Arc::new(MorselMetrics::default());
        let native = split_context(
            &dir,
            "required",
            8,
            pushdown,
            true,
            Some(split_cfg.clone()),
            Some(Arc::clone(&metrics)),
            None,
        )
        .await?;

        // 1. Full projection across split boundaries with NULLs and dictionary column
        for sql in [
            "SELECT id, score, delta, payload FROM t",
            "SELECT id, payload + delta AS p FROM t WHERE (score + 3 * delta) % 10 < 3",
            "SELECT COUNT(*) AS n FROM t WHERE (score + 3 * delta) % 10 < 3",
            "SELECT id, score FROM t WHERE score > 50 AND random() >= 0.0",
        ] {
            let expected = stock.sql(sql).await?.collect().await?;
            let plan = native.sql(sql).await?.create_physical_plan().await?;
            let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected),
                "pushdown={pushdown}: {sql}"
            );
            assert_eq!(
                metric(plan.as_ref(), "parquet_morsels_claimed"),
                32,
                "pushdown={pushdown}: {sql}"
            );
            assert_eq!(
                metric(plan.as_ref(), "parquet_split_row_groups"),
                4,
                "pushdown={pushdown}: {sql}"
            );
            assert_eq!(
                metric(plan.as_ref(), "parquet_split_slices"),
                32,
                "pushdown={pushdown}: {sql}"
            );
            assert!(
                metric(plan.as_ref(), "parquet_split_reused_bytes") > 0,
                "expected page/dictionary reuse for pushdown={pushdown}: {sql}"
            );
            let runs = metrics.take_runs();
            assert_eq!(runs.len(), 1);
            assert_eq!(
                runs[0].sum_by_name("morsels_claimed").unwrap().as_usize(),
                32
            );
            assert_eq!(
                runs[0].sum_by_name("morsels_completed").unwrap().as_usize(),
                32
            );
        }

        // 2. Page-index-pruned sub-ranges (only a subset of slices survive!)
        for sql in [
            "SELECT id, score FROM t WHERE id >= 2000 AND id < 2100",
            "SELECT COUNT(*) AS n FROM t WHERE id >= 2000 AND id < 2100",
            "SELECT id, score FROM t WHERE (id >= 100 AND id < 200) OR (id >= 1800 AND id < 2200)",
            "SELECT COUNT(*) AS n FROM t WHERE id < 0",
        ] {
            let expected = stock.sql(sql).await?.collect().await?;
            let plan = native.sql(sql).await?.create_physical_plan().await?;
            let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected),
                "pruned pushdown={pushdown}: {sql}"
            );
            let runs = metrics.take_runs();
            assert_eq!(runs.len(), 1);
            assert_eq!(
                runs[0].sum_by_name("morsels_claimed").unwrap().as_usize(),
                runs[0].sum_by_name("morsels_completed").unwrap().as_usize()
            );
        }

        // 3. Metadata-only (0 projected columns) scan skips splitting
        let sql = "SELECT 42 AS c FROM t";
        let expected = stock.sql(sql).await?.collect().await?;
        let plan = native.sql(sql).await?.create_physical_plan().await?;
        let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected)
        );
        assert_eq!(metric(plan.as_ref(), "parquet_split_row_groups"), 0);
        let _ = metrics.take_runs();
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn intra_rg_splitting_enables_all_8_workers_on_4_large_row_groups() -> Result<()> {
    // 2 files x 2 large RGs = 4 row groups total across 8 workers.
    let dir = split_fixture(2, 2, 2400, true)?;

    // Control: default ParquetSplitConfig is opt-in (enabled == false), so
    // even with thresholds below the RG size at most 4 workers claim the 4 RGs.
    assert!(!ParquetSplitConfig::default().enabled);
    let disabled_metrics = Arc::new(MorselMetrics::default());
    let disabled_ctx = split_context(
        &dir,
        "required",
        8,
        false,
        true,
        Some(ParquetSplitConfig {
            min_row_group_rows: 800,
            target_slice_rows: 300,
            ..Default::default()
        }),
        Some(Arc::clone(&disabled_metrics)),
        None,
    )
    .await?;
    let disabled_plan = disabled_ctx
        .sql("SELECT COUNT(*) AS n FROM t WHERE score > 10")
        .await?
        .create_physical_plan()
        .await?;
    let expected = collect(Arc::clone(&disabled_plan), disabled_ctx.task_ctx()).await?;
    assert_eq!(metric(disabled_plan.as_ref(), "parquet_morsels_claimed"), 4);
    let disabled_runs = disabled_metrics.take_runs();
    let disabled_active_workers = disabled_runs[0]
        .iter()
        .filter(|m| m.value().name() == "morsels_claimed" && m.value().as_usize() > 0)
        .count();
    assert!(
        disabled_active_workers <= 4,
        "expected <= 4 active workers without splitting, got {disabled_active_workers}"
    );

    // Enabled: barrier inside a scalar UDF requires all 8 workers to claim and
    // run slices concurrently on the exact same 4-row-group table!
    let split_metrics = Arc::new(MorselMetrics::default());
    let split_ctx = split_context(
        &dir,
        "required",
        8,
        false,
        true,
        Some(ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 800,
            target_slice_rows: 300, // 32 slices across 8 workers
            ..Default::default()
        }),
        Some(Arc::clone(&split_metrics)),
        None,
    )
    .await?;
    let seen_threads = Arc::new(Mutex::new(HashSet::<ThreadId>::new()));
    let barrier = Arc::new(Barrier::new(8));
    let seen_threads_udf = Arc::clone(&seen_threads);
    let barrier_udf = Arc::clone(&barrier);
    let gate_udf = create_udf(
        "sync_eight_workers",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(move |args| {
            let first_for_thread =
                seen_threads_udf.lock().insert(std::thread::current().id());
            if first_for_thread {
                barrier_udf.wait();
            }
            let arr = args[0].clone().into_array(1)?;
            let ints = arr.as_any().downcast_ref::<Int32Array>().unwrap();
            let mask: BooleanArray = ints.iter().map(|v| v.map(|x| x > 10)).collect();
            Ok(ColumnarValue::Array(Arc::new(mask)))
        }),
    );
    split_ctx.register_udf(gate_udf);

    let split_plan = split_ctx
        .sql("SELECT COUNT(*) AS n FROM t WHERE sync_eight_workers(score)")
        .await?
        .create_physical_plan()
        .await?;
    let actual = timeout(
        Duration::from_secs(15),
        collect(Arc::clone(&split_plan), split_ctx.task_ctx()),
    )
    .await
    .expect("timed out waiting for all 8 workers to claim split slices")?;
    assert_eq!(actual, expected);
    assert_eq!(metric(split_plan.as_ref(), "parquet_morsels_claimed"), 32);
    assert_eq!(metric(split_plan.as_ref(), "parquet_split_row_groups"), 4);

    let split_runs = split_metrics.take_runs();
    assert_eq!(split_runs.len(), 1);
    let active_workers = split_runs[0]
        .iter()
        .filter(|m| m.value().name() == "morsels_claimed" && m.value().as_usize() > 0)
        .count();
    assert_eq!(active_workers, 8, "all 8 workers must claim split slices");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn intra_rg_splitting_fallbacks_and_mixed_split_unsplit_rgs() -> Result<()> {
    let split_cfg = ParquetSplitConfig {
        enabled: true,
        min_row_group_rows: 800,
        target_slice_rows: 300,
        ..Default::default()
    };

    // 1. Missing Page/Offset Index -> clean unsplit fallback with exact parity
    let no_index_dir = split_fixture(2, 2, 2400, false)?;
    for pushdown in [false, true] {
        let stock =
            split_context(&no_index_dir, "off", 4, pushdown, true, None, None, None)
                .await?;
        let native = split_context(
            &no_index_dir,
            "required",
            4,
            pushdown,
            true,
            Some(split_cfg.clone()),
            None,
            None,
        )
        .await?;
        let sql = "SELECT id, score FROM t WHERE score > 50";
        let expected = stock.sql(sql).await?.collect().await?;
        let plan = native.sql(sql).await?.create_physical_plan().await?;
        let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected)
        );
        assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 4);
        assert_eq!(metric(plan.as_ref(), "parquet_split_row_groups"), 0);
    }

    // 2. Indexed file with enable_page_index=false -> clean unsplit fallback
    let indexed_dir = split_fixture(2, 2, 2400, true)?;
    let native_pi_off = split_context(
        &indexed_dir,
        "required",
        4,
        false,
        false,
        Some(split_cfg.clone()),
        None,
        None,
    )
    .await?;
    let plan = native_pi_off
        .sql("SELECT id, score FROM t")
        .await?
        .create_physical_plan()
        .await?;
    let batches = collect(Arc::clone(&plan), native_pi_off.task_ctx()).await?;
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        9600
    );
    assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 4);
    assert_eq!(metric(plan.as_ref(), "parquet_split_row_groups"), 0);

    // 3. Mixed file: 1 large RG (2400 rows -> 8 slices) + 1 small RG (200 rows -> 1 unsplit)
    //    AND includes a Utf8 + Struct column to verify unsupported-type fallback vs
    //    fixed-width-only projection splitting on the exact same file!
    let mixed_dir = TempDir::new()?;
    let mixed_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
        Field::new(
            "nested",
            DataType::Struct(Fields::from(vec![Field::new("x", DataType::Int32, false)])),
            false,
        ),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2400))
        .set_write_batch_size(35)
        .set_data_page_row_count_limit(70)
        .set_data_page_size_limit(256)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(mixed_dir.path().join("mixed.parquet"))?,
        Arc::clone(&mixed_schema),
        Some(props),
    )?;
    for (start, len) in [(0i64, 2400usize), (2400i64, 200usize)] {
        let ids: Vec<i64> = (start..start + len as i64).collect();
        let scores: Vec<Option<i32>> = ids
            .iter()
            .map(|&i| (i % 7 != 0).then_some((i % 100) as i32))
            .collect();
        let labels: Vec<Option<String>> = ids
            .iter()
            .map(|&i| (i % 5 != 0).then_some(format!("v{i}")))
            .collect();
        let x_arr = Arc::new(Int32Array::from_iter_values(ids.iter().map(|&i| i as i32)));
        let nested = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Int32, false)),
            x_arr as _,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&mixed_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int32Array::from(scores)),
                Arc::new(StringArray::from(labels)),
                nested,
            ],
        )?;
        writer.write(&batch)?;
        writer.flush()?;
    }
    writer.close()?;

    for pushdown in [false, true] {
        let stock =
            split_context(&mixed_dir, "off", 4, pushdown, true, None, None, None).await?;
        let native = split_context(
            &mixed_dir,
            "required",
            4,
            pushdown,
            true,
            Some(split_cfg.clone()),
            None,
            None,
        )
        .await?;

        // Flat fixed-width projection -> RG 0 splits (8 slices) + RG 1 unsplit (1) = 9 morsels!
        let sql_flat = "SELECT id, score FROM t WHERE score > 20";
        let expected = stock.sql(sql_flat).await?.collect().await?;
        let plan = native.sql(sql_flat).await?.create_physical_plan().await?;
        let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected)
        );
        assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 9);
        assert_eq!(metric(plan.as_ref(), "parquet_split_row_groups"), 1);
        assert_eq!(metric(plan.as_ref(), "parquet_split_slices"), 8);

        // Unsupported string or struct columns -> clean unsplit fallback (2 morsels)
        for sql_unsupported in [
            "SELECT id, label FROM t WHERE score > 20",
            "SELECT id, nested FROM t WHERE score > 20",
        ] {
            let expected = stock.sql(sql_unsupported).await?.collect().await?;
            let plan = native
                .sql(sql_unsupported)
                .await?
                .create_physical_plan()
                .await?;
            let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected)
            );
            assert_eq!(metric(plan.as_ref(), "parquet_morsels_claimed"), 2);
            assert_eq!(metric(plan.as_ref(), "parquet_split_row_groups"), 0);
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn intra_rg_splitting_cancellation_and_mid_rg_error_cleanup() -> Result<()> {
    let dir = split_fixture(2, 2, 2400, true)?;
    let split_cfg = ParquetSplitConfig {
        enabled: true,
        min_row_group_rows: 800,
        target_slice_rows: 300,
        ..Default::default()
    };

    // 1. Early stream drop mid-RG releases all memory pool reservations and reports partial metrics
    for pushdown in [false, true] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128 << 20));
        let metrics = Arc::new(MorselMetrics::default());
        let native = split_context(
            &dir,
            "required",
            8,
            pushdown,
            true,
            Some(split_cfg.clone()),
            Some(Arc::clone(&metrics)),
            Some(Arc::clone(&pool)),
        )
        .await?;
        let plan = native
            .sql("SELECT id, score, delta FROM t WHERE score > 10")
            .await?
            .create_physical_plan()
            .await?;
        let mut stream = execute_stream(Arc::clone(&plan), native.task_ctx())?;
        let first = timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("timed out reading first batch")
            .expect("expected at least one batch")?;
        assert!(first.num_rows() > 0);
        drop(stream);

        // Wait briefly for aborted tasks to drop their reservations and publish partial report.
        timeout(Duration::from_secs(5), async {
            loop {
                if pool.reserved() == 0 && !metrics.take_runs().is_empty() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("timed out waiting for cancelled split scan to release reservations");
        assert_eq!(pool.reserved(), 0);
    }

    // 2. Mid-RG UDF error propagates cleanly and releases all reservations
    let fail_mid_rg = create_udf(
        "fail_mid_rg",
        vec![DataType::Int64],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(|args| {
            let arr = args[0].clone().into_array(1)?;
            let ids = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            // id=1502 has score=(1502*17)%100=34 > 10 and 1502%19 != 0.
            if ids.iter().any(|v| v == Some(1502)) {
                return datafusion::common::exec_err!("injected mid-rg filter failure");
            }
            let mask: BooleanArray = ids.iter().map(|v| v.map(|x| x >= 0)).collect();
            Ok(ColumnarValue::Array(Arc::new(mask)))
        }),
    );

    for pushdown in [false, true] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128 << 20));
        let native = split_context(
            &dir,
            "required",
            8,
            pushdown,
            true,
            Some(split_cfg.clone()),
            None,
            Some(Arc::clone(&pool)),
        )
        .await?;
        native.register_udf(fail_mid_rg.clone());
        let plan = native
            .sql("SELECT COUNT(*) AS n FROM t WHERE score > 10 AND fail_mid_rg(id)")
            .await?
            .create_physical_plan()
            .await?;
        let err = timeout(
            Duration::from_secs(10),
            collect(Arc::clone(&plan), native.task_ctx()),
        )
        .await
        .expect("timed out on mid-RG error")
        .unwrap_err()
        .to_string();
        assert!(err.contains("injected mid-rg filter failure"), "{err}");
        assert_eq!(metric(plan.as_ref(), "count_finalizations"), 0);
        timeout(Duration::from_secs(5), async {
            while pool.reserved() != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("reservations must return to zero after mid-RG error");
    }

    Ok(())
}

fn wide_six_col_split_fixture(
    files: usize,
    rgs_per_file: usize,
    rows_per_rg: usize,
) -> Result<TempDir> {
    let dir = TempDir::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int64, true),
        Field::new("c", DataType::Int64, false),
        Field::new("d", DataType::Int32, true),
        Field::new("e", DataType::Int64, true),
    ]));
    let mut row_start = 0i64;
    for file in 0..files {
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_rg))
            .set_write_batch_size(40)
            .set_data_page_row_count_limit(80)
            .set_data_page_size_limit(512)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("part-{file:03}.parquet")))?,
            Arc::clone(&schema),
            Some(props),
        )?;
        for _ in 0..rgs_per_file {
            let ids: Vec<i64> = (row_start..row_start + rows_per_rg as i64).collect();
            let a: Vec<Option<i32>> = ids
                .iter()
                .map(|&i| (i % 19 != 0).then_some(((i * 37) % 1000) as i32))
                .collect();
            let b: Vec<Option<i64>> = ids
                .iter()
                .map(|&i| (i % 17 != 0).then_some(i * 13 + 5))
                .collect();
            let c: Vec<i64> = ids.iter().map(|&i| i * 29 - 11).collect();
            let d: Vec<Option<i32>> = ids
                .iter()
                .map(|&i| (i % 23 != 0).then_some((i % 5) as i32))
                .collect();
            let e: Vec<Option<i64>> = ids
                .iter()
                .map(|&i| (i % 11 != 0).then_some(i * 41 + 3))
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(Int32Array::from(a)),
                    Arc::new(Int64Array::from(b)),
                    Arc::new(Int64Array::from(c)),
                    Arc::new(Int32Array::from(d)),
                    Arc::new(Int64Array::from(e)),
                ],
            )?;
            writer.write(&batch)?;
            writer.flush()?;
            row_start += rows_per_rg as i64;
        }
        writer.close()?;
    }
    Ok(dir)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn intra_rg_splitting_bounded_memory_pool_both_pushdown_modes() -> Result<()> {
    // 2 files x 2 RGs x 4800 rows, 6 fixed-width columns, 8 workers.
    let dir = wide_six_col_split_fixture(2, 2, 4800)?;
    let split_cfg = ParquetSplitConfig {
        enabled: true,
        min_row_group_rows: 800,
        target_slice_rows: 300,
        ..Default::default()
    };
    let queries = [
        "SELECT * FROM t WHERE a > 100",
        "SELECT COUNT(*) AS n FROM t WHERE a > 500 AND d = 1",
    ];

    for pushdown in [false, true] {
        let stock =
            split_context(&dir, "off", 8, pushdown, true, None, None, None).await?;
        for sql in queries {
            let expected = stock.sql(sql).await?.collect().await?;

            // 1. Bounded pool smaller than all in-flight slice estimates combined
            //    (only ~2-3 slices fit at once across 8 workers) must pace cleanly
            //    via backpressure, split all 4 RGs, and match stock.
            for round in 0..3 {
                let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(48 << 10));
                let native = split_context(
                    &dir,
                    "required",
                    8,
                    pushdown,
                    true,
                    Some(split_cfg.clone()),
                    None,
                    Some(Arc::clone(&pool)),
                )
                .await?;
                let plan = native.sql(sql).await?.create_physical_plan().await?;
                let actual = timeout(
                    Duration::from_secs(15),
                    collect(Arc::clone(&plan), native.task_ctx()),
                )
                .await
                .expect("timed out under bounded memory pool")?;
                assert_eq!(
                    batches_to_sort_string(&actual),
                    batches_to_sort_string(&expected),
                    "bounded pool pushdown={pushdown} round={round}: {sql}"
                );
                assert_eq!(
                    metric(plan.as_ref(), "parquet_split_row_groups"),
                    4,
                    "expected all 4 RGs to split under 48 KiB pool (pushdown={pushdown}): {sql}"
                );
                assert_eq!(pool.reserved(), 0);
            }

            // 2. Pool smaller than even a single slice must fall back cleanly to
            //    the unsplit reader before publishing slice 0 instead of erroring.
            let tiny_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024));
            let native = split_context(
                &dir,
                "required",
                8,
                pushdown,
                true,
                Some(split_cfg.clone()),
                None,
                Some(Arc::clone(&tiny_pool)),
            )
            .await?;
            let plan = native.sql(sql).await?.create_physical_plan().await?;
            let actual = timeout(
                Duration::from_secs(15),
                collect(Arc::clone(&plan), native.task_ctx()),
            )
            .await
            .expect("timed out on tiny-pool fallback")?;
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected),
                "tiny pool fallback pushdown={pushdown}: {sql}"
            );
            assert_eq!(
                metric(plan.as_ref(), "parquet_split_row_groups"),
                0,
                "expected clean unsplit fallback under 1 KiB pool (pushdown={pushdown}): {sql}"
            );
            assert_eq!(tiny_pool.reserved(), 0);
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn intra_rg_splitting_tight_buffer_cap_both_pushdown_modes() -> Result<()> {
    // 2 files x 2 RGs x 4800 rows, 6 fixed-width columns, 8 workers.
    // Tight split_buffer_cap_bytes forces frontier serialization across file
    // lanes AND slice-by-slice cap backpressure across ready/worker/last-slice
    // transitions for both plain and multi-conjunct pushed filter paths.
    let dir = wide_six_col_split_fixture(2, 2, 4800)?;
    let queries = [
        ("SELECT * FROM t WHERE a > 100", 48 << 10),
        (
            "SELECT COUNT(*) AS n FROM t WHERE a > 500 AND d = 1",
            16 << 10,
        ),
    ];

    for pushdown in [false, true] {
        let stock =
            split_context(&dir, "off", 8, pushdown, true, None, None, None).await?;
        for (sql, tight_cap) in queries {
            let expected = stock.sql(sql).await?.collect().await?;
            let split_cfg = ParquetSplitConfig {
                enabled: true,
                min_row_group_rows: 800,
                target_slice_rows: 300,
                split_buffer_cap_bytes: tight_cap,
                ..Default::default()
            };
            for round in 0..5 {
                let pool: Arc<dyn MemoryPool> =
                    Arc::new(GreedyMemoryPool::new(128 << 10));
                let native = split_context(
                    &dir,
                    "required",
                    8,
                    pushdown,
                    true,
                    Some(split_cfg.clone()),
                    None,
                    Some(Arc::clone(&pool)),
                )
                .await?;
                let plan = native.sql(sql).await?.create_physical_plan().await?;
                let actual = timeout(
                    Duration::from_secs(15),
                    collect(Arc::clone(&plan), native.task_ctx()),
                )
                .await
                .expect("timed out under tight split_buffer_cap_bytes")?;
                assert_eq!(
                    batches_to_sort_string(&actual),
                    batches_to_sort_string(&expected),
                    "tight cap={tight_cap} pushdown={pushdown} round={round}: {sql}"
                );
                assert_eq!(
                    metric(plan.as_ref(), "parquet_split_row_groups"),
                    4,
                    "expected all 4 RGs to split under tight cap={tight_cap} (pushdown={pushdown}, round={round}): {sql}"
                );
                assert_eq!(pool.reserved(), 0);
            }
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn intra_rg_splitting_small_pool_sweep_both_pushdown_modes() -> Result<()> {
    // 8 files x 1 RG x 4800 rows, id (Int64) + dictionary-encoded a, b, d with
    // 70-row pages, batch_size=1000, 8 workers: reproduces pool-unaware
    // multi-lane admission failures in the 6-24 KiB transition band where slice 0
    // fits individually but pinned dictionaries + later slices exceed the pool.
    let dir = TempDir::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("d", DataType::Int32, true),
    ]));
    let rows_per_rg = 4800usize;
    let mut row_start = 0i64;
    for file in 0..8usize {
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_rg))
            .set_write_batch_size(35)
            .set_data_page_row_count_limit(70)
            .set_data_page_size_limit(256)
            .set_dictionary_page_size_limit(256)
            .set_dictionary_enabled(true)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("part-{file:03}.parquet")))?,
            Arc::clone(&schema),
            Some(props),
        )?;
        let ids: Vec<i64> = (row_start..row_start + rows_per_rg as i64).collect();
        let a: Vec<Option<i32>> = ids
            .iter()
            .map(|&i| (i % 19 != 0).then_some(((i % 10) * 10) as i32))
            .collect();
        let b: Vec<Option<i32>> = ids
            .iter()
            .map(|&i| (i % 17 != 0).then_some((i % 11) as i32))
            .collect();
        let d: Vec<Option<i32>> = ids
            .iter()
            .map(|&i| (i % 23 != 0).then_some((i % 5) as i32))
            .collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int32Array::from(a)),
                Arc::new(Int32Array::from(b)),
                Arc::new(Int32Array::from(d)),
            ],
        )?;
        writer.write(&batch)?;
        writer.close()?;
        row_start += rows_per_rg as i64;
    }

    let split_cfg = ParquetSplitConfig {
        enabled: true,
        min_row_group_rows: 800,
        target_slice_rows: 300,
        ..Default::default()
    };

    // Sweep two workloads across 1..=64 KiB in 1 KiB steps:
    // 1) batch_size=1000, COUNT(*) WHERE a > 50 AND d = 1 (reproduces the exact
    //    multi-lane 6-8 KiB dictionary-pinning failure on pushdown=false).
    // 2) batch_size=128, SELECT id, b WHERE a > 50 AND d = 1 (exercises the full
    //    fallback -> single-lane -> multi-lane transition band for BOTH pushdown
    //    modes, including Phase-1 + Phase-2 admission).
    let cases = [
        (
            1000usize,
            "SELECT COUNT(*) AS n FROM t WHERE a > 50 AND d = 1",
            false,
        ),
        (128usize, "SELECT id, b FROM t WHERE a > 50 AND d = 1", true),
    ];

    for (batch_size, sql, require_split_both_modes) in cases {
        for pushdown in [false, true] {
            let mut stock_cfg = SessionConfig::new()
                .with_target_partitions(8)
                .with_batch_size(batch_size);
            stock_cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
            stock_cfg.options_mut().execution.parquet.enable_page_index = true;
            stock_cfg.options_mut().execution.collect_statistics = false;
            let stock = SessionContext::new_with_config(stock_cfg);
            stock
                .register_parquet(
                    "t",
                    dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
            let expected = stock.sql(sql).await?.collect().await?;
            let table = stock.table_provider("t").await?;

            let mut saw_unsplit = false;
            let mut saw_split = false;

            for pool_kib in 1usize..=64 {
                let repeats = if (6..=24).contains(&pool_kib) { 2 } else { 1 };
                for round in 0..repeats {
                    let pool: Arc<dyn MemoryPool> =
                        Arc::new(GreedyMemoryPool::new(pool_kib << 10));
                    let mut config = SessionConfig::new()
                        .with_target_partitions(8)
                        .with_batch_size(batch_size);
                    config.options_mut().execution.parquet.pushdown_filters = pushdown;
                    config.options_mut().execution.parquet.enable_page_index = true;
                    config.options_mut().execution.collect_statistics = false;
                    config.set_extension(Arc::new(split_cfg.clone()));
                    let runtime = RuntimeEnvBuilder::new()
                        .with_memory_pool(Arc::clone(&pool))
                        .build_arc()?;
                    let native = SessionContext::new_with_config_rt(config, runtime);
                    native.register_table("t", Arc::clone(&table))?;
                    native
                        .sql("SET datafusion.execution.morsel_mode = 'required'")
                        .await?;
                    let plan = native.sql(sql).await?.create_physical_plan().await?;
                    let actual = timeout(
                        Duration::from_secs(15),
                        collect(Arc::clone(&plan), native.task_ctx()),
                    )
                    .await
                    .unwrap_or_else(|_| {
                        panic!(
                            "timed out at pool={pool_kib}KiB batch={batch_size} pushdown={pushdown} round={round}: {sql}"
                        )
                    })
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed at pool={pool_kib}KiB batch={batch_size} pushdown={pushdown} round={round}: {sql}: {err}"
                        )
                    });
                    assert_eq!(
                        batches_to_sort_string(&actual),
                        batches_to_sort_string(&expected),
                        "row mismatch at pool={pool_kib}KiB batch={batch_size} pushdown={pushdown} round={round}: {sql}"
                    );
                    let split_rgs = metric(plan.as_ref(), "parquet_split_row_groups");
                    if split_rgs == 0 {
                        saw_unsplit = true;
                    } else {
                        saw_split = true;
                    }
                    assert_eq!(pool.reserved(), 0);
                }
            }
            if !pushdown || require_split_both_modes {
                assert!(
                    saw_unsplit && saw_split,
                    "expected sweep to cover both fallback and split regimes (batch={batch_size}, pushdown={pushdown}, saw_unsplit={saw_unsplit}, saw_split={saw_split})"
                );
            }
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parquet_group_count_parity_across_pushdown_nulls_empty_and_schema_adaptation()
-> Result<()> {
    let dir = TempDir::new()?;
    let person_fields = Fields::from(vec![Field::new("name", DataType::Utf8, true)]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("gate", DataType::Int32, true),
        Field::new("person", DataType::Struct(person_fields.clone()), true),
    ]));
    for file in 0..2i32 {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4))
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("{file}.parquet")))?,
            Arc::clone(&schema),
            Some(properties),
        )?;
        let base = file * 12;
        let ids: Vec<i32> = (base..base + 12).collect();
        let gates: Vec<Option<i32>> = vec![
            None,
            Some(12),
            Some(12),
            Some(5),
            None,
            Some(20),
            Some(12),
            Some(20),
            Some(5),
            None,
            Some(30),
            Some(5),
        ];
        let names = Arc::new(StringArray::from(vec![
            Some("a"),
            Some("b"),
            None,
            Some("d"),
            Some("e"),
            Some("f"),
            Some("g"),
            Some("h"),
            Some("i"),
            Some("j"),
            Some("k"),
            Some("l"),
        ]));
        let person = StructArray::try_new(
            person_fields.clone(),
            vec![names],
            Some(
                vec![
                    true, true, false, true, true, true, false, true, true, false, true,
                    true,
                ]
                .into(),
            ),
        )?;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(gates)),
                Arc::new(person),
            ],
        )?;
        writer.write(&batch)?;
        writer.close()?;
    }

    let empty = TempDir::new()?;
    let queries = [
        "SELECT gate, COUNT(*) AS n FROM t GROUP BY gate",
        "SELECT gate, COUNT(*) AS n FROM t WHERE id >= 4 GROUP BY gate",
        "SELECT gate, COUNT(*) AS n FROM t WHERE gate > 10 GROUP BY gate",
        "SELECT COUNT(*) + 1 AS n, gate FROM t WHERE gate > 10 AND person IS NOT NULL GROUP BY gate",
        "SELECT gate, COUNT(*) AS n FROM t WHERE gate > 10 AND random() >= 0.0 GROUP BY gate",
        "SELECT gate, COUNT(*) AS n FROM t WHERE gate > 100 GROUP BY gate",
        "SELECT gate, COUNT(*) AS n FROM empty GROUP BY gate",
    ];

    for pushdown in [false, true] {
        for workers in [1, 4] {
            let stock = context(&dir, "off", pushdown).await?;
            let native = context(&dir, "required", pushdown).await?;
            for ctx in [&stock, &native] {
                ctx.sql(&format!(
                    "SET datafusion.execution.target_partitions = {workers}"
                ))
                .await?;
                ctx.sql("SET datafusion.execution.batch_size = 2").await?;
                ctx.register_parquet(
                    "empty",
                    empty.path().to_str().unwrap(),
                    ParquetReadOptions::default().schema(schema.as_ref()),
                )
                .await?;
            }
            for sql in queries {
                let expected_plan = stock.sql(sql).await?.create_physical_plan().await?;
                let actual_plan = native.sql(sql).await?.create_physical_plan().await?;
                let display = displayable(actual_plan.as_ref()).indent(false).to_string();
                assert!(
                    display.contains("NativeParquet")
                        && display.contains("GroupCountSink")
                        && display.contains("FinalizedGroupCountSource"),
                    "pushdown={pushdown}, workers={workers}, {sql}: {display}"
                );
                assert_eq!(actual_plan.schema(), expected_plan.schema(), "{sql}");
                let expected = collect(expected_plan, stock.task_ctx()).await?;
                let actual = collect(Arc::clone(&actual_plan), native.task_ctx()).await?;
                assert!(actual.iter().all(|b| b.num_rows() <= 2), "{sql}");
                assert_eq!(
                    batches_to_sort_string(&actual),
                    batches_to_sort_string(&expected),
                    "pushdown={pushdown}, workers={workers}: {sql}"
                );
                assert_eq!(
                    metric(actual_plan.as_ref(), "group_count_finalizations"),
                    1,
                    "{sql}"
                );
                if sql.contains("empty") || sql.contains("gate > 100") {
                    assert_eq!(
                        actual.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        0
                    );
                }
            }
        }
    }

    // Schema adaptation (Int32 -> Int64 widening on GROUP BY key)
    let adapted_schema = Schema::new(vec![
        Field::new("gate", DataType::Int64, true),
        Field::new("id", DataType::Int32, false),
    ]);
    for pushdown in [false, true] {
        let stock = context(&dir, "off", pushdown).await?;
        let native = context(&dir, "required", pushdown).await?;
        for ctx in [&stock, &native] {
            ctx.register_parquet(
                "adapted",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default().schema(&adapted_schema),
            )
            .await?;
        }
        let sql = "SELECT gate, COUNT(*) AS n FROM adapted WHERE id >= 3 GROUP BY gate";
        let expected_plan = stock.sql(sql).await?.create_physical_plan().await?;
        let actual_plan = native.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(actual_plan.schema(), expected_plan.schema());
        let expected = collect(expected_plan, stock.task_ctx()).await?;
        let actual = collect(actual_plan, native.task_ctx()).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected)
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parquet_group_count_high_cardinality_concurrent_oom_error_and_cancellation()
-> Result<()> {
    let dir = TempDir::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("gate", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    for file in 0..8i64 {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(500))
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("{file}.parquet")))?,
            Arc::clone(&schema),
            Some(properties),
        )?;
        let base = file * 2_000;
        let keys: Vec<Option<i64>> = (base..base + 2_000)
            .map(|i| if i % 503 == 0 { None } else { Some(i) })
            .collect();
        let gates: Vec<i32> = (base..base + 2_000)
            .map(|i| if i == 4_321 { 0 } else { (i % 7) as i32 + 1 })
            .collect();
        let payloads: Vec<&str> = vec!["p"; 2_000];
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(Int32Array::from(gates)),
                Arc::new(StringArray::from(payloads)),
            ],
        )?;
        writer.write(&batch)?;
        writer.close()?;
    }

    let sql = "SELECT k, COUNT(*) AS n FROM t WHERE gate > 2 GROUP BY k";
    let stock = context(&dir, "off", true).await?;
    let expected = stock.sql(sql).await?.collect().await?;

    // 1. Concurrent multi-worker execution with bounded output batches & pool cleanup.
    for pushdown in [false, true] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
        let native = split_context(
            &dir,
            "required",
            4,
            pushdown,
            false,
            None,
            None,
            Some(Arc::clone(&pool)),
        )
        .await?;
        let plan = native.sql(sql).await?.create_physical_plan().await?;
        let (a, b) = tokio::try_join!(
            collect(Arc::clone(&plan), native.task_ctx()),
            collect(Arc::clone(&plan), native.task_ctx()),
        )?;
        for batches in [&a, &b] {
            assert!(batches.len() > 1);
            assert!(batches.iter().all(|batch| batch.num_rows() <= 128));
            assert_eq!(
                batches_to_sort_string(batches),
                batches_to_sort_string(&expected),
                "pushdown={pushdown}"
            );
        }
        assert_eq!(metric(plan.as_ref(), "group_count_finalizations"), 2);
        assert_eq!(pool.reserved(), 0);
    }

    // 2. Tight memory pool fails cleanly with ResourcesExhausted and releases all memory.
    for pushdown in [false, true] {
        let tight_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024));
        let native = split_context(
            &dir,
            "required",
            4,
            pushdown,
            false,
            None,
            None,
            Some(Arc::clone(&tight_pool)),
        )
        .await?;
        let plan = native
            .sql("SELECT k, COUNT(*) AS n FROM t GROUP BY k")
            .await?
            .create_physical_plan()
            .await?;
        let err = collect(Arc::clone(&plan), native.task_ctx())
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                datafusion::error::DataFusionError::ResourcesExhausted(_)
            ),
            "pushdown={pushdown}: expected ResourcesExhausted, got: {err}"
        );
        assert_eq!(metric(plan.as_ref(), "group_count_finalizations"), 0);
        assert_eq!(tight_pool.reserved(), 0);
    }

    // 3. Worker error during Parquet filter skips finalization and cleans up reservations.
    for pushdown in [false, true] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
        let native = split_context(
            &dir,
            "required",
            4,
            pushdown,
            false,
            None,
            None,
            Some(Arc::clone(&pool)),
        )
        .await?;
        let plan = native
            .sql("SELECT k, COUNT(*) AS n FROM t WHERE 100 / gate > 1 GROUP BY k")
            .await?
            .create_physical_plan()
            .await?;
        let err = collect(Arc::clone(&plan), native.task_ctx())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.to_lowercase().contains("divide by zero"),
            "pushdown={pushdown}: {err}"
        );
        assert_eq!(metric(plan.as_ref(), "group_count_finalizations"), 0);
        assert_eq!(pool.reserved(), 0);
    }

    // 4. Early stream cancellation frees all memory reservations.
    for pushdown in [false, true] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
        let native = split_context(
            &dir,
            "required",
            4,
            pushdown,
            false,
            None,
            None,
            Some(Arc::clone(&pool)),
        )
        .await?;
        let plan = native
            .sql("SELECT k, COUNT(*) AS n FROM t GROUP BY k")
            .await?
            .create_physical_plan()
            .await?;
        let mut stream = execute_stream(Arc::clone(&plan), native.task_ctx())?;
        let first = stream.next().await.expect("first batch")?;
        assert_eq!(first.num_rows(), 128);
        drop(stream);
        timeout(Duration::from_secs(5), async {
            while pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("pool reservation leaked after early cancellation");
    }

    // 5. Unsupported Parquet GROUP BY rejects in required mode and falls back in prefer mode.
    let prefer = context(&dir, "prefer", false).await?;
    let required = context(&dir, "required", false).await?;
    for bad_sql in [
        "SELECT k, COUNT(DISTINCT gate) FROM t GROUP BY k",
        "SELECT k, COUNT(*) FILTER (WHERE gate > 0) FROM t GROUP BY k",
        "SELECT k, MEDIAN(gate) FROM t GROUP BY k",
        "SELECT k, COUNT(*) FROM t GROUP BY ROLLUP(k)",
    ] {
        let err = required
            .sql(bad_sql)
            .await?
            .create_physical_plan()
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("morsel_mode=required"), "{bad_sql}: {err}");
        let fallback = prefer.sql(bad_sql).await?.create_physical_plan().await?;
        assert!(
            !displayable(fallback.as_ref())
                .indent(false)
                .to_string()
                .contains("MorselExec"),
            "{bad_sql}"
        );
        let actual = collect(fallback, prefer.task_ctx()).await?;
        let expected = stock.sql(bad_sql).await?.collect().await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "{bad_sql}"
        );
    }
    Ok(())
}
