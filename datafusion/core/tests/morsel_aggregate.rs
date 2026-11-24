// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(all(feature = "sql", feature = "parquet"))]

use std::collections::HashMap;
use std::fs::File;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, DictionaryArray, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray, StringViewArray, TimestampMicrosecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion::common::ScalarValue;
use datafusion::common::instant::Instant;
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
};
use datafusion::morsel::{MorselMetrics, MorselQueryPlanner};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::time::timeout;

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

async fn physical(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

fn sorted_rows(batches: &[RecordBatch]) -> Result<Vec<OwnedRow>> {
    let Some(first) = batches.first() else {
        return Ok(Vec::new());
    };
    let fields = first
        .schema()
        .fields()
        .iter()
        .map(|f| SortField::new(f.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let mut rows = Vec::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let converted = converter.convert_columns(batch.columns())?;
        rows.extend(converted.iter().map(|row| row.owned()));
    }
    rows.sort();
    Ok(rows)
}

fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
    plan.metrics()
        .and_then(|m| m.sum_by_name(name))
        .map_or(0, |v| v.as_usize())
}

fn rich_schema() -> Arc<Schema> {
    let mut meta = HashMap::new();
    meta.insert("table_owner".to_string(), "morsel_test".to_string());
    let mut key_meta = HashMap::new();
    key_meta.insert("col_tag".to_string(), "group_key".to_string());

    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("k_i8", DataType::Int8, true),
            Field::new("k_i16", DataType::Int16, true),
            Field::new("k_i32", DataType::Int32, true).with_metadata(key_meta),
            Field::new("k_i64", DataType::Int64, true),
            Field::new("k_u8", DataType::UInt8, true),
            Field::new("k_u16", DataType::UInt16, true),
            Field::new("k_u32", DataType::UInt32, true),
            Field::new("k_u64", DataType::UInt64, true),
            Field::new("k_f32", DataType::Float32, true),
            Field::new("k_f64", DataType::Float64, true),
            Field::new("k_dec", DataType::Decimal128(15, 2), true),
            Field::new("k_date", DataType::Date32, true),
            Field::new(
                "k_ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
            Field::new("k_str", DataType::Utf8, true),
            Field::new("k_lstr", DataType::LargeUtf8, true),
            Field::new("k_sview", DataType::Utf8View, true),
            Field::new("k_bool", DataType::Boolean, true),
            Field::new("v_i32", DataType::Int32, true),
            Field::new("v_i64", DataType::Int64, true),
            Field::new("v_u64", DataType::UInt64, true),
            Field::new("v_f64", DataType::Float64, true),
            Field::new("v_dec", DataType::Decimal128(15, 2), true),
            Field::new("v_str", DataType::Utf8, true),
            Field::new("v_nulls", DataType::Int64, true),
            Field::new("gate", DataType::Int32, true),
        ],
        meta,
    ))
}

fn rich_batches(empty: bool) -> Result<(Arc<Schema>, Vec<Vec<RecordBatch>>)> {
    let schema = rich_schema();
    if empty {
        return Ok((schema, vec![vec![], vec![]]));
    }
    let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(), Vec::new()];
    let labels = [Some("alpha"), Some("beta"), Some("gamma"), None, Some("")];
    let float_vals = [
        Some(1.5),
        Some(-0.0),
        Some(0.0),
        Some(f64::NAN),
        None,
        Some(-2.25),
        Some(10.0),
    ];

    for b in 0..8usize {
        let n = 17usize;
        let base = b * n;
        let mut k_i8 = Vec::with_capacity(n);
        let mut k_i16 = Vec::with_capacity(n);
        let mut k_i32 = Vec::with_capacity(n);
        let mut k_i64 = Vec::with_capacity(n);
        let mut k_u8 = Vec::with_capacity(n);
        let mut k_u16 = Vec::with_capacity(n);
        let mut k_u32 = Vec::with_capacity(n);
        let mut k_u64 = Vec::with_capacity(n);
        let mut k_f32 = Vec::with_capacity(n);
        let mut k_f64 = Vec::with_capacity(n);
        let mut k_dec = Vec::with_capacity(n);
        let mut k_date = Vec::with_capacity(n);
        let mut k_ts = Vec::with_capacity(n);
        let mut k_str = Vec::with_capacity(n);
        let mut k_lstr = Vec::with_capacity(n);
        let mut k_sview = Vec::with_capacity(n);
        let mut k_bool = Vec::with_capacity(n);
        let mut v_i32 = Vec::with_capacity(n);
        let mut v_i64 = Vec::with_capacity(n);
        let mut v_u64 = Vec::with_capacity(n);
        let mut v_f64 = Vec::with_capacity(n);
        let mut v_dec = Vec::with_capacity(n);
        let mut v_str = Vec::with_capacity(n);
        let mut v_nulls = Vec::with_capacity(n);
        let mut gate = Vec::with_capacity(n);

        for j in 0..n {
            let i = base + j;
            let grp = i % 5;
            k_i8.push((grp != 4).then_some((grp as i8) - 2));
            k_i16.push((grp != 4).then_some((grp as i16) * 100 - 200));
            k_i32.push((grp != 4).then_some(grp as i32));
            k_i64.push((grp != 4).then_some((grp as i64) * 10_000));
            k_u8.push((grp != 4).then_some(grp as u8));
            k_u16.push((grp != 4).then_some((grp as u16) * 10));
            k_u32.push((grp != 4).then_some((grp as u32) * 100));
            k_u64.push((grp != 4).then_some((grp as u64) * 1000));
            let fv = float_vals[i % float_vals.len()];
            k_f32.push(fv.map(|v| v as f32));
            k_f64.push(fv);
            k_dec.push((grp != 4).then_some((grp as i128) * 125 - 250));
            k_date.push((grp != 4).then_some(9000 + grp as i32));
            k_ts.push((grp != 4).then_some(1_700_000_000_000_000 + (grp as i64) * 1000));
            let lbl = labels[grp];
            k_str.push(lbl);
            k_lstr.push(lbl);
            k_sview.push(lbl);
            k_bool.push(match i % 3 {
                0 => Some(true),
                1 => Some(false),
                _ => None,
            });
            // Make group 3 have all-NULL values to test all-NULL group semantics.
            let is_val_null = grp == 3 || i.is_multiple_of(7);
            v_i32.push((!is_val_null).then_some((i as i32) * 3 - 50));
            v_i64.push((!is_val_null).then_some((i as i64) * 11 - 200));
            v_u64.push((!is_val_null).then_some((i as u64) * 7 + 1));
            v_f64.push((!is_val_null).then_some((i as f64) * 0.5 - 12.25));
            v_dec.push((!is_val_null).then_some((i as i128) * 103 - 500));
            v_str.push((!is_val_null).then_some(match i % 4 {
                0 => "aardvark",
                1 => "zebra",
                2 => "middle",
                _ => "long-string-for-view-and-heap-storage",
            }));
            v_nulls.push(None);
            gate.push(if i.is_multiple_of(11) {
                None
            } else {
                Some((i % 10) as i32)
            });
        }

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int8Array::from(k_i8)),
                Arc::new(Int16Array::from(k_i16)),
                Arc::new(Int32Array::from(k_i32)),
                Arc::new(Int64Array::from(k_i64)),
                Arc::new(UInt8Array::from(k_u8)),
                Arc::new(UInt16Array::from(k_u16)),
                Arc::new(UInt32Array::from(k_u32)),
                Arc::new(UInt64Array::from(k_u64)),
                Arc::new(Float32Array::from(k_f32)),
                Arc::new(Float64Array::from(k_f64)),
                Arc::new(Decimal128Array::from(k_dec).with_precision_and_scale(15, 2)?),
                Arc::new(Date32Array::from(k_date)),
                Arc::new(TimestampMicrosecondArray::from(k_ts)),
                Arc::new(StringArray::from(k_str)),
                Arc::new(LargeStringArray::from(k_lstr)),
                Arc::new(StringViewArray::from(k_sview)),
                Arc::new(BooleanArray::from(k_bool)),
                Arc::new(Int32Array::from(v_i32)),
                Arc::new(Int64Array::from(v_i64)),
                Arc::new(UInt64Array::from(v_u64)),
                Arc::new(Float64Array::from(v_f64)),
                Arc::new(Decimal128Array::from(v_dec).with_precision_and_scale(15, 2)?),
                Arc::new(StringArray::from(v_str)),
                Arc::new(Int64Array::from(v_nulls)),
                Arc::new(Int32Array::from(gate)),
            ],
        )?;
        partitions[b % 2].push(batch);
    }
    Ok((schema, partitions))
}

fn write_parquet_files(
    dir: &TempDir,
    schema: &Arc<Schema>,
    partitions: &[Vec<RecordBatch>],
    rows_per_rg: usize,
) -> Result<()> {
    for (file_idx, batches) in partitions.iter().enumerate() {
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_rg))
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("part-{file_idx}.parquet")))?,
            Arc::clone(schema),
            Some(props),
        )?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
    }
    Ok(())
}

async fn assert_parity(
    stock: &SessionContext,
    native: &SessionContext,
    sql: &str,
    expected_sink_token: &str,
) -> Result<()> {
    let stock_plan = physical(stock, sql).await?;
    let native_plan = physical(native, sql).await?;
    assert_eq!(native_plan.name(), "MorselExec", "{sql}");
    let display = displayable(native_plan.as_ref()).indent(false).to_string();
    assert!(
        display.contains(expected_sink_token),
        "{sql}: expected `{expected_sink_token}` in:\n{display}"
    );
    assert_eq!(
        native_plan.schema(),
        stock_plan.schema(),
        "{sql}: schema mismatch"
    );

    let expected = collect(stock_plan, stock.task_ctx()).await?;
    let actual = collect(Arc::clone(&native_plan), native.task_ctx()).await?;
    assert_eq!(sorted_rows(&actual)?, sorted_rows(&expected)?, "{sql}");

    let finalizations = match expected_sink_token {
        "CountSink" => metric(native_plan.as_ref(), "count_finalizations"),
        "GroupCountSink" => metric(native_plan.as_ref(), "group_count_finalizations"),
        _ => metric(native_plan.as_ref(), "aggregate_finalizations"),
    };
    assert_eq!(finalizations, 1, "{sql}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ungrouped_multi_aggregate_parity_memtable_and_parquet() -> Result<()> {
    let (schema, partitions) = rich_batches(false)?;
    let (empty_schema, empty_partitions) = rich_batches(true)?;

    let mem = Arc::new(MemTable::try_new(Arc::clone(&schema), partitions.clone())?);
    let empty_mem = Arc::new(MemTable::try_new(
        Arc::clone(&empty_schema),
        empty_partitions.clone(),
    )?);

    let dir = TempDir::new()?;
    write_parquet_files(&dir, &schema, &partitions, 24)?;
    let empty_dir = TempDir::new()?;
    write_parquet_files(&empty_dir, &empty_schema, &empty_partitions, 24)?;

    let queries = [
        ("SELECT COUNT(*) FROM t", "CountSink"),
        ("SELECT COUNT(*) FROM t WHERE gate > 3", "CountSink"),
        (
            "SELECT COUNT(*), COUNT(v_i32), COUNT(v_nulls), COUNT(1), COUNT(NULL) FROM t",
            "AggregateSink",
        ),
        (
            "SELECT SUM(v_i32), SUM(v_i64), SUM(v_u64), SUM(v_f64), SUM(v_dec), SUM(v_nulls) FROM t",
            "AggregateSink",
        ),
        (
            "SELECT MIN(v_i32), MAX(v_i32), MIN(v_f64), MAX(v_f64), MIN(v_dec), MAX(v_dec), MIN(k_date), MAX(k_date), MIN(v_str), MAX(v_str), MIN(k_bool), MAX(k_bool), MIN(v_nulls), MAX(v_nulls) FROM t",
            "AggregateSink",
        ),
        (
            "SELECT AVG(v_i32), AVG(v_i64), AVG(v_f64), AVG(v_dec), AVG(v_nulls) FROM t",
            "AggregateSink",
        ),
        (
            "SELECT COUNT(*) AS n, COUNT(v_dec) AS n_dec, SUM(v_dec) AS s_dec, AVG(v_dec) AS a_dec, MIN(v_str) AS lo, MAX(v_str) AS hi, COALESCE(SUM(v_nulls), 0) AS z FROM t WHERE gate >= 2",
            "AggregateSink",
        ),
        // Filter eliminates every row -> ungrouped aggregate must still emit 1 row.
        (
            "SELECT COUNT(*) AS n, COUNT(v_i64) AS nc, SUM(v_i64) AS s, MIN(v_str) AS lo, MAX(v_dec) AS hi, AVG(v_dec) AS a FROM t WHERE gate > 100",
            "AggregateSink",
        ),
        // Empty table -> ungrouped aggregate must emit 1 row.
        (
            "SELECT COUNT(*) AS n, COUNT(v_i64) AS nc, SUM(v_i64) AS s, MIN(v_str) AS lo, MAX(v_dec) AS hi, AVG(v_dec) AS a FROM empty_t",
            "AggregateSink",
        ),
        (
            "SELECT COUNT(*) + 10 AS n10, COALESCE(SUM(v_i64), -1) AS s FROM empty_t",
            "AggregateSink",
        ),
    ];

    // MemTable: workers = 1 and 4
    for workers in [1usize, 4] {
        let stock = SessionContext::new();
        let native = morsel_context(workers);
        native
            .sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        for ctx in [&stock, &native] {
            ctx.register_table("t", mem.clone())?;
            ctx.register_table("empty_t", empty_mem.clone())?;
        }
        for (sql, sink) in queries {
            assert_parity(&stock, &native, sql, sink).await?;
        }
    }

    // Parquet: pushdown_filters = false and true
    for pushdown in [false, true] {
        let mut cfg = SessionConfig::new().with_target_partitions(4);
        cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
        cfg.options_mut().execution.collect_statistics = false;
        let stock = SessionContext::new_with_config(cfg.clone());
        let native = morsel_context_with_config(4, cfg);
        native
            .sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        for ctx in [&stock, &native] {
            ctx.register_parquet(
                "t",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
            ctx.register_parquet(
                "empty_t",
                empty_dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        }
        for (sql, sink) in queries {
            assert_parity(&stock, &native, sql, sink).await?;
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn grouped_multi_aggregate_parity_all_key_types_memtable_and_parquet() -> Result<()>
{
    let (schema, partitions) = rich_batches(false)?;
    let (empty_schema, empty_partitions) = rich_batches(true)?;

    let mem = Arc::new(MemTable::try_new(Arc::clone(&schema), partitions.clone())?);
    let empty_mem = Arc::new(MemTable::try_new(
        Arc::clone(&empty_schema),
        empty_partitions.clone(),
    )?);

    let dir = TempDir::new()?;
    write_parquet_files(&dir, &schema, &partitions, 24)?;
    let empty_dir = TempDir::new()?;
    write_parquet_files(&empty_dir, &empty_schema, &empty_partitions, 24)?;

    let single_key_cols = [
        "k_i8", "k_i16", "k_i32", "k_i64", "k_u8", "k_u16", "k_u32", "k_u64", "k_f32",
        "k_f64", "k_dec", "k_date", "k_ts", "k_str", "k_lstr", "k_sview", "k_bool",
    ];

    let mut queries: Vec<(String, &'static str)> = single_key_cols
        .iter()
        .map(|col| {
            (
                format!(
                    "SELECT {col}, COUNT(*) AS n, COUNT(v_i64) AS nc, SUM(v_i64) AS s, \
                     MIN(v_dec) AS lo_dec, MAX(v_str) AS hi_str, AVG(v_dec) AS a_dec, \
                     AVG(v_f64) AS a_f64 \
                     FROM t GROUP BY {col}"
                ),
                "AggregateSink",
            )
        })
        .collect();

    // Multi-column keys, expression keys, filtered, empty, pure GROUP BY, and fast-path check.
    queries.extend([
        (
            "SELECT k_i32, COUNT(*) AS n FROM t GROUP BY k_i32".into(),
            "GroupCountSink",
        ),
        (
            "SELECT k_str, k_bool, COUNT(*) AS n, SUM(v_dec) AS s, AVG(v_dec) AS a, MIN(v_i32) AS lo, MAX(v_i32) AS hi FROM t GROUP BY k_str, k_bool".into(),
            "AggregateSink",
        ),
        (
            "SELECT k_date, k_dec, k_sview, COUNT(*) AS n, COUNT(v_nulls) AS nn, SUM(v_i64) AS s, AVG(v_f64) AS af FROM t WHERE gate >= 2 GROUP BY k_date, k_dec, k_sview".into(),
            "AggregateSink",
        ),
        (
            "SELECT k_i32 + 1 AS k_plus, lower(k_str) AS kl, SUM(v_i64) + 5 AS s5, COUNT(*) AS n FROM t GROUP BY k_i32 + 1, lower(k_str)".into(),
            "AggregateSink",
        ),
        // Pure GROUP BY / DISTINCT (0 aggregates)
        (
            "SELECT k_str, k_i32 FROM t GROUP BY k_str, k_i32".into(),
            "AggregateSink",
        ),
        // Grouped empty input -> 0 rows!
        (
            "SELECT k_str, k_bool, COUNT(*) AS n, SUM(v_dec) AS s, AVG(v_dec) AS a FROM t WHERE gate > 100 GROUP BY k_str, k_bool".into(),
            "AggregateSink",
        ),
        (
            "SELECT k_str, k_bool, COUNT(*) AS n, SUM(v_dec) AS s, AVG(v_dec) AS a FROM empty_t GROUP BY k_str, k_bool".into(),
            "AggregateSink",
        ),
    ]);

    // MemTable: workers = 1 and 4
    for workers in [1usize, 4] {
        let stock = SessionContext::new();
        let native = morsel_context(workers);
        native
            .sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        for ctx in [&stock, &native] {
            ctx.register_table("t", mem.clone())?;
            ctx.register_table("empty_t", empty_mem.clone())?;
        }
        for (sql, sink) in &queries {
            assert_parity(&stock, &native, sql, sink).await?;
        }
    }

    // Parquet: pushdown_filters = false and true
    for pushdown in [false, true] {
        let mut cfg = SessionConfig::new().with_target_partitions(4);
        cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
        cfg.options_mut().execution.collect_statistics = false;
        let stock = SessionContext::new_with_config(cfg.clone());
        let native = morsel_context_with_config(4, cfg);
        native
            .sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        for ctx in [&stock, &native] {
            ctx.register_parquet(
                "t",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
            ctx.register_parquet(
                "empty_t",
                empty_dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        }
        for (sql, sink) in &queries {
            assert_parity(&stock, &native, sql, sink).await?;
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dictionary_group_key_and_multi_arg_count_parity() -> Result<()> {
    let dict_values = StringArray::from(vec!["east", "west", "north"]);
    let dict_keys = Int32Array::from(vec![
        Some(0),
        Some(1),
        None,
        Some(0),
        Some(2),
        Some(1),
        None,
        Some(0),
    ]);
    let dict = DictionaryArray::<Int32Type>::try_new(dict_keys, Arc::new(dict_values))?;
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "region",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(dict) as ArrayRef,
            Arc::new(Int64Array::from(vec![
                Some(10),
                None,
                Some(5),
                Some(20),
                Some(3),
                Some(7),
                None,
                Some(30),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(2),
                None,
                Some(4),
                None,
                Some(6),
                None,
                Some(8),
            ])),
        ],
    )?;
    let table = Arc::new(MemTable::try_new(schema, vec![vec![batch]])?);
    let stock = SessionContext::new();
    let native = morsel_context(2);
    native
        .sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    for ctx in [&stock, &native] {
        ctx.register_table("t", table.clone())?;
    }
    for sql in [
        "SELECT region, COUNT(*), COUNT(a), COUNT(a, b), SUM(a), MIN(b), MAX(b), AVG(a) FROM t GROUP BY region",
        "SELECT COUNT(a, b), SUM(a + b) FROM t",
    ] {
        assert_parity(&stock, &native, sql, "AggregateSink").await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tpch_q6_and_q1_unordered_parity_memtable_and_parquet() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_quantity", DataType::Decimal128(15, 2), false),
        Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
        Field::new("l_discount", DataType::Decimal128(15, 2), false),
        Field::new("l_tax", DataType::Decimal128(15, 2), false),
        Field::new("l_returnflag", DataType::Utf8, false),
        Field::new("l_linestatus", DataType::Utf8, false),
        Field::new("l_shipdate", DataType::Date32, false),
    ]));

    let flags = ["A", "N", "N", "R"];
    let statuses = ["F", "F", "O", "F"];
    // 1994-01-01 is day 8766; 1998-09-02 is day 10471.
    let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for b in 0..16usize {
        let n = 256usize;
        let base = b * n;
        let qty: Vec<i128> = (0..n)
            .map(|j| (((base + j) % 50) as i128 + 1) * 100)
            .collect();
        let price: Vec<i128> = (0..n)
            .map(|j| (((base + j) % 900) as i128 + 100) * 100 + ((base + j) % 99) as i128)
            .collect();
        let disc: Vec<i128> = (0..n).map(|j| ((base + j) % 11) as i128).collect();
        let tax: Vec<i128> = (0..n).map(|j| ((base + j) % 9) as i128).collect();
        let rf: Vec<&str> = (0..n).map(|j| flags[(base + j) % 4]).collect();
        let ls: Vec<&str> = (0..n).map(|j| statuses[(base + j) % 4]).collect();
        let shipdate: Vec<i32> = (0..n)
            .map(|j| 8500 + (((base + j) * 13) % 2100) as i32)
            .collect();

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Decimal128Array::from(qty).with_precision_and_scale(15, 2)?),
                Arc::new(Decimal128Array::from(price).with_precision_and_scale(15, 2)?),
                Arc::new(Decimal128Array::from(disc).with_precision_and_scale(15, 2)?),
                Arc::new(Decimal128Array::from(tax).with_precision_and_scale(15, 2)?),
                Arc::new(StringArray::from(rf)),
                Arc::new(StringArray::from(ls)),
                Arc::new(Date32Array::from(shipdate)),
            ],
        )?;
        partitions[b % 4].push(batch);
    }

    let q6 = "
        SELECT SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
          AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR
          AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
          AND l_quantity < 24
    ";
    let q1_unordered = "
        SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) AS sum_qty,
            SUM(l_extendedprice) AS sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            AVG(l_quantity) AS avg_qty,
            AVG(l_extendedprice) AS avg_price,
            AVG(l_discount) AS avg_disc,
            COUNT(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-09-02'
        GROUP BY l_returnflag, l_linestatus
    ";

    // MemTable
    let mem = Arc::new(MemTable::try_new(Arc::clone(&schema), partitions.clone())?);
    let stock = SessionContext::new();
    let native = morsel_context(4);
    native
        .sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    for ctx in [&stock, &native] {
        ctx.register_table("lineitem", mem.clone())?;
    }
    assert_parity(&stock, &native, q6, "AggregateSink").await?;
    assert_parity(&stock, &native, q1_unordered, "AggregateSink").await?;

    // Parquet (both pushdown_filters=false and true)
    let dir = TempDir::new()?;
    write_parquet_files(&dir, &schema, &partitions, 512)?;
    for pushdown in [false, true] {
        let mut cfg = SessionConfig::new().with_target_partitions(4);
        cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
        cfg.options_mut().execution.collect_statistics = false;
        let stock = SessionContext::new_with_config(cfg.clone());
        let native = morsel_context_with_config(4, cfg);
        native
            .sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        for ctx in [&stock, &native] {
            ctx.register_parquet(
                "lineitem",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        }
        assert_parity(&stock, &native, q6, "AggregateSink").await?;
        assert_parity(&stock, &native, q1_unordered, "AggregateSink").await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn high_cardinality_skewed_concurrent_and_metrics() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("skewed", DataType::Utf8, true),
        Field::new("high_card", DataType::Utf8, false),
        Field::new("k2", DataType::Int32, false),
        Field::new("val", DataType::Decimal128(15, 2), true),
        Field::new("gate", DataType::Int32, false),
    ]));
    let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for b in 0..32usize {
        let n = 1024usize;
        let start = b * n;
        let skewed: Vec<Option<String>> = (start..start + n)
            .map(|i| {
                if !i.is_multiple_of(5) {
                    Some("hot_key".into())
                } else if i.is_multiple_of(50) {
                    None
                } else {
                    Some(format!("cold_{}", i % 32))
                }
            })
            .collect();
        let high_card: Vec<String> = (start..start + n)
            .map(|i| format!("grp_{:05}", (i * 2654435761) % 25_000))
            .collect();
        let k2: Vec<i32> = (start..start + n).map(|i| (i % 3) as i32).collect();
        let val: Vec<Option<i128>> = (start..start + n)
            .map(|i| (!i.is_multiple_of(13)).then_some((i as i128) * 17 - 1000))
            .collect();
        let gate: Vec<i32> = (start..start + n).map(|i| (i % 10) as i32).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(skewed)),
                Arc::new(StringArray::from(high_card)),
                Arc::new(Int32Array::from(k2)),
                Arc::new(Decimal128Array::from(val).with_precision_and_scale(15, 2)?),
                Arc::new(Int32Array::from(gate)),
            ],
        )?;
        partitions[b % 4].push(batch);
    }

    let queries = [
        "SELECT skewed, COUNT(*) AS n, COUNT(val) AS nv, SUM(val) AS s, AVG(val) AS a, MIN(val) AS lo, MAX(val) AS hi FROM big WHERE gate < 7 GROUP BY skewed",
        "SELECT high_card, COUNT(*) AS n, SUM(val) AS s, AVG(val) AS a FROM big GROUP BY high_card",
        "SELECT high_card, k2, COUNT(*) AS n, SUM(val) AS s, AVG(val) AS a FROM big GROUP BY high_card, k2",
        "SELECT COUNT(*) AS n, SUM(val) AS s, AVG(val) AS a, MIN(val) AS lo, MAX(val) AS hi FROM big WHERE gate >= 2",
    ];

    let table = Arc::new(MemTable::try_new(Arc::clone(&schema), partitions.clone())?);
    let dir = TempDir::new()?;
    write_parquet_files(&dir, &schema, &partitions, 2048)?;

    for (label, is_parquet, pushdown) in [
        ("memtable", false, false),
        ("parquet_no_pushdown", true, false),
        ("parquet_pushdown", true, true),
    ] {
        let metrics = Arc::new(MorselMetrics::default());
        let mut stock_cfg = SessionConfig::new()
            .with_batch_size(512)
            .with_target_partitions(4);
        stock_cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
        stock_cfg.options_mut().execution.collect_statistics = false;
        let native_cfg = stock_cfg.clone().with_extension(Arc::clone(&metrics));

        let stock = SessionContext::new_with_config(stock_cfg);
        let native = morsel_context_with_config(4, native_cfg);
        native
            .sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        for ctx in [&stock, &native] {
            if is_parquet {
                ctx.register_parquet(
                    "big",
                    dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
            } else {
                ctx.register_table("big", table.clone())?;
            }
        }

        for sql in queries {
            let stock_plan = physical(&stock, sql).await?;
            let native_plan = physical(&native, sql).await?;
            assert_eq!(native_plan.name(), "MorselExec", "{label}: {sql}");
            let display = displayable(native_plan.as_ref()).indent(false).to_string();
            assert!(
                display.contains("AggregateSink"),
                "{label}: {sql}: {display}"
            );
            if is_parquet {
                assert!(
                    display.contains("NativeParquet[big"),
                    "{label}: {sql}: {display}"
                );
            }
            let expected = sorted_rows(&collect(stock_plan, stock.task_ctx()).await?)?;
            let (a, b) = tokio::try_join!(
                collect(Arc::clone(&native_plan), native.task_ctx()),
                collect(Arc::clone(&native_plan), native.task_ctx()),
            )?;
            assert_eq!(sorted_rows(&a)?, expected, "{label}: {sql}");
            assert_eq!(sorted_rows(&b)?, expected, "{label}: {sql}");
            assert_eq!(
                metric(native_plan.as_ref(), "aggregate_finalizations"),
                2,
                "{label}: {sql}"
            );
            let runs = metrics.take_runs();
            assert_eq!(runs.len(), 2, "{label}: {sql}");
            for run in &runs {
                assert!(
                    run.sum_by_name("morsels_claimed").unwrap().as_usize() > 0,
                    "{label}: {sql}"
                );
                assert!(
                    run.sum_by_name("morsels_completed").unwrap().as_usize() > 0,
                    "{label}: {sql}"
                );
                assert!(
                    run.sum_by_name("morsel_peak_output_depth")
                        .unwrap()
                        .as_usize()
                        > 0,
                    "{label}: {sql}"
                );
            }
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn memory_limit_oom_worker_error_and_early_cancel_memtable_and_parquet()
-> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
        Field::new("gate", DataType::Int32, false),
    ]));
    let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for b in 0..16usize {
        let n = 1024usize;
        let start = b * n;
        let keys: Vec<String> = (start..start + n)
            .map(|i| format!("group-key-{i:08}-extra-payload"))
            .collect();
        let vals: Vec<i64> = (start..start + n).map(|i| i as i64).collect();
        let gate: Vec<i32> = (start..start + n).map(|i| (i % 5) as i32).collect();
        partitions[b % 4].push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(keys)),
                Arc::new(Int64Array::from(vals)),
                Arc::new(Int32Array::from(gate)),
            ],
        )?);
    }
    let mem = Arc::new(MemTable::try_new(Arc::clone(&schema), partitions.clone())?);
    let dir = TempDir::new()?;
    write_parquet_files(&dir, &schema, &partitions, 1024)?;

    let make_ctx = |pool: Arc<dyn MemoryPool>, pushdown: bool| -> SessionContext {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()
            .unwrap();
        let mut config = SessionConfig::new()
            .with_batch_size(128)
            .with_target_partitions(4);
        config.options_mut().execution.parquet.pushdown_filters = pushdown;
        config.options_mut().execution.collect_statistics = false;
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(config)
            .with_runtime_env(runtime)
            .with_query_planner(Arc::new(MorselQueryPlanner::new(
                NonZeroUsize::new(4).unwrap(),
            )))
            .build();
        SessionContext::new_with_state(state)
    };

    let sql = "SELECT k, COUNT(*) AS n, SUM(v) AS s, AVG(v) AS a, MIN(v) AS lo, MAX(v) AS hi FROM t GROUP BY k";
    let err_sql = "SELECT k, SUM(v) AS s FROM t WHERE 100 / gate > 1 GROUP BY k";

    for (label, is_parquet, pushdown) in [
        ("memtable", false, false),
        ("parquet_no_pushdown", true, false),
        ("parquet_pushdown", true, true),
    ] {
        // 1. Tight memory pool -> clean ResourcesExhausted, 0 finalizations, 0 leaked bytes.
        let tight: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128 * 1024));
        let ctx = make_ctx(Arc::clone(&tight), pushdown);
        ctx.sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        if is_parquet {
            ctx.register_parquet(
                "t",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        } else {
            ctx.register_table("t", mem.clone())?;
        }
        let plan = physical(&ctx, sql).await?;
        let err = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap_err();
        assert!(
            matches!(err, DataFusionError::ResourcesExhausted(_)),
            "{label}: expected ResourcesExhausted, got: {err}"
        );
        assert_eq!(
            metric(plan.as_ref(), "aggregate_finalizations"),
            0,
            "{label}"
        );
        assert_eq!(tight.reserved(), 0, "{label}");

        // 2. Worker runtime error -> skips finalization, 0 leaked bytes.
        let err_pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
        let ctx = make_ctx(Arc::clone(&err_pool), pushdown);
        ctx.sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        if is_parquet {
            ctx.register_parquet(
                "t",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        } else {
            ctx.register_table("t", mem.clone())?;
        }
        let plan = physical(&ctx, err_sql).await?;
        let err = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.to_lowercase().contains("divide by zero"),
            "{label}: {err}"
        );
        assert_eq!(
            metric(plan.as_ref(), "aggregate_finalizations"),
            0,
            "{label}"
        );
        assert_eq!(err_pool.reserved(), 0, "{label}");

        // 3. Early stream cancellation mid-emission -> frees all memory.
        let cancel_pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
        let ctx = make_ctx(Arc::clone(&cancel_pool), pushdown);
        ctx.sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?;
        if is_parquet {
            ctx.register_parquet(
                "t",
                dir.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        } else {
            ctx.register_table("t", mem.clone())?;
        }
        let plan = physical(&ctx, sql).await?;
        let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
        let first = stream.next().await.expect("first batch")?;
        assert_eq!(first.num_rows(), 128, "{label}");
        drop(stream);
        let deadline = Instant::now() + Duration::from_secs(5);
        timeout(Duration::from_secs(5), async {
            while cancel_pool.reserved() != 0 && Instant::now() < deadline {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(cancel_pool.reserved(), 0, "{label}");
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct CustomSumUdaf {
    signature: Signature,
}

impl CustomSumUdaf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Int64], Volatility::Immutable),
        }
    }
}

#[derive(Debug, Default)]
struct CustomSumAccumulator(i64);

impl Accumulator for CustomSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let arr = values[0].as_any().downcast_ref::<Int64Array>().unwrap();
        for v in arr.iter().flatten() {
            self.0 += v;
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Int64(Some(self.0)))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Int64(Some(self.0))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }
}

impl AggregateUDFImpl for CustomSumUdaf {
    fn name(&self) -> &str {
        "custom_sum"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn accumulator(
        &self,
        _acc_args: datafusion::logical_expr::function::AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(CustomSumAccumulator::default()))
    }
}

#[tokio::test]
async fn unsupported_aggregates_reject_in_required_and_fallback_in_prefer() -> Result<()>
{
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("a"),
                None,
            ])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), None])),
        ],
    )?;
    let table = Arc::new(MemTable::try_new(schema, vec![vec![batch]])?);

    let default = SessionContext::new();
    let prefer = morsel_context(2);
    let required = morsel_context(2);
    prefer
        .sql("SET datafusion.execution.morsel_mode = 'prefer'")
        .await?;
    required
        .sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    for ctx in [&default, &prefer, &required] {
        ctx.register_table("t", table.clone())?;
        ctx.register_udaf(AggregateUDF::from(CustomSumUdaf::new()));
    }

    for (sql, expected_reason) in [
        ("SELECT COUNT(DISTINCT v) FROM t", "DISTINCT"),
        (
            "SELECT COUNT(DISTINCT k), SUM(DISTINCT v) FROM t",
            "DISTINCT",
        ),
        ("SELECT k, SUM(DISTINCT v) FROM t GROUP BY k", "DISTINCT"),
        ("SELECT k, AVG(DISTINCT v) FROM t GROUP BY k", "DISTINCT"),
        ("SELECT SUM(v) FILTER (WHERE v > 10) FROM t", "FILTER"),
        (
            "SELECT k, COUNT(*) FILTER (WHERE v > 10) FROM t GROUP BY k",
            "FILTER",
        ),
        (
            "SELECT k, SUM(v) FROM t GROUP BY GROUPING SETS ((k), ())",
            "GROUPING SETS",
        ),
        (
            "SELECT k, SUM(v) FROM t GROUP BY ROLLUP(k)",
            "GROUPING SETS",
        ),
        ("SELECT k, SUM(v) FROM t GROUP BY CUBE(k)", "GROUPING SETS"),
        (
            "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 10",
            "only projection",
        ),
        ("SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k", "Sort"),
        ("SELECT k, SUM(v) FROM t GROUP BY k LIMIT 1", "LIMIT"),
        ("SELECT MEDIAN(v) FROM t", "median"),
        ("SELECT k, ARRAY_AGG(v) FROM t GROUP BY k", "array_agg"),
        ("SELECT k, STDDEV(v) FROM t GROUP BY k", "stddev"),
        ("SELECT custom_sum(v) FROM t", "custom_sum"),
        ("SELECT k, custom_sum(v) FROM t GROUP BY k", "custom_sum"),
    ] {
        let err = physical(&required, sql).await.unwrap_err().to_string();
        assert!(
            err.contains("morsel_mode=required")
                && err.to_lowercase().contains(&expected_reason.to_lowercase()),
            "{sql}: expected `{expected_reason}` in: {err}"
        );

        let fallback = physical(&prefer, sql).await?;
        assert!(
            !displayable(fallback.as_ref())
                .indent(false)
                .to_string()
                .contains("MorselExec"),
            "{sql}"
        );
        let expected = physical(&default, sql).await?;
        let actual_batches = collect(fallback, prefer.task_ctx()).await?;
        let expected_batches = collect(expected, default.task_ctx()).await?;
        if sql.contains("LIMIT 1") {
            assert_eq!(
                actual_batches
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
                1,
                "{sql}"
            );
        } else {
            assert_eq!(
                sorted_rows(&actual_batches)?,
                sorted_rows(&expected_batches)?,
                "{sql}"
            );
        }
    }
    Ok(())
}
