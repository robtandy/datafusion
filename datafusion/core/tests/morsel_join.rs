// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(all(feature = "sql", feature = "parquet"))]

use std::fs::File;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray, StringViewArray, StructArray, TimestampMicrosecondArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion::common::JoinType;
use datafusion::common::config::MorselMode;
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::morsel::{MorselMetrics, MorselQueryPlanner};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::time::timeout;

fn stock_context(workers: usize, batch_size: usize) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = MorselMode::Off;
    SessionContext::new_with_config(config)
}

fn morsel_context(
    workers: usize,
    batch_size: usize,
    mode: MorselMode,
    pushdown: bool,
) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = mode;
    config.options_mut().execution.parquet.pushdown_filters = pushdown;
    config.options_mut().execution.parquet.reorder_filters = pushdown;
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

async fn assert_native_parity(
    stock: &SessionContext,
    native: &SessionContext,
    sql: &str,
) -> Result<(Arc<dyn ExecutionPlan>, Vec<RecordBatch>)> {
    let stock_plan = physical(stock, sql).await?;
    let expected = collect(Arc::clone(&stock_plan), stock.task_ctx()).await?;

    let native_plan = physical(native, sql).await?;
    assert_eq!(
        native_plan.name(),
        "MorselExec",
        "expected native MorselExec for `{sql}`, got:\n{}",
        displayable(native_plan.as_ref()).indent(true)
    );
    let actual = collect(Arc::clone(&native_plan), native.task_ctx()).await?;

    assert_eq!(
        native_plan.schema().fields().len(),
        stock_plan.schema().fields().len(),
        "column count mismatch for `{sql}`"
    );
    for (nf, sf) in native_plan
        .schema()
        .fields()
        .iter()
        .zip(stock_plan.schema().fields().iter())
    {
        assert_eq!(nf.name(), sf.name(), "field name mismatch for `{sql}`");
        assert_eq!(
            nf.data_type(),
            sf.data_type(),
            "field type mismatch for `{sql}`"
        );
    }

    assert_eq!(
        sorted_rows(&actual)?,
        sorted_rows(&expected)?,
        "row parity mismatch for `{sql}`"
    );
    Ok((native_plan, actual))
}

fn write_parquet_files(
    batches: &[RecordBatch],
    files: usize,
    max_rg_rows: usize,
) -> Result<TempDir> {
    let dir = TempDir::new()?;
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(max_rg_rows))
        .build();
    if batches.is_empty() {
        return Ok(dir);
    }
    let schema = batches[0].schema();
    for file_idx in 0..files {
        let mut writer = ArrowWriter::try_new(
            File::create(dir.path().join(format!("part-{file_idx}.parquet")))?,
            Arc::clone(&schema),
            Some(props.clone()),
        )?;
        for (i, batch) in batches.iter().enumerate() {
            if i % files == file_idx {
                writer.write(batch)?;
            }
        }
        writer.close()?;
    }
    Ok(dir)
}

fn make_kv_table(
    rows: &[(Option<i32>, Option<&str>, i32)],
    chunk_size: usize,
) -> Result<(Arc<Schema>, Vec<RecordBatch>)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("v", DataType::Int32, false),
    ]));
    if rows.is_empty() {
        return Ok((schema, vec![]));
    }
    let mut batches = Vec::new();
    for chunk in rows.chunks(chunk_size.max(1)) {
        let ks: Vec<Option<i32>> = chunk.iter().map(|r| r.0).collect();
        let ss: Vec<Option<&str>> = chunk.iter().map(|r| r.1).collect();
        let vs: Vec<i32> = chunk.iter().map(|r| r.2).collect();
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(ks)),
                Arc::new(StringArray::from(ss)),
                Arc::new(Int32Array::from(vs)),
            ],
        )?);
    }
    Ok((schema, batches))
}

#[tokio::test]
async fn inner_join_parity_cardinality_nulls_duplicates_and_blowup() -> Result<()> {
    // 1-to-1, 1-to-many, many-to-many, NULL keys, unmatched keys, duplicate keys.
    let mut left_rows = Vec::new();
    // 1-to-1 keys: 1..=20
    for i in 1..=20 {
        left_rows.push((Some(i), Some("one"), i * 10));
    }
    // 1-to-many key 100 (1 row on left, 35 rows on right)
    left_rows.push((Some(100), Some("fan"), 1000));
    // Many-to-many key 200 (12 rows on left, 15 rows on right => 180 output rows)
    for i in 0..12 {
        left_rows.push((Some(200), Some("mm"), 2000 + i));
    }
    // Unmatched keys
    for i in 500..510 {
        left_rows.push((Some(i), Some("unmatched"), i));
    }
    // NULL keys (must NEVER match right NULL keys)
    for i in 0..8 {
        left_rows.push((None, None, -i));
        left_rows.push((Some(200), None, -100 - i));
        left_rows.push((None, Some("mm"), -200 - i));
    }

    let mut right_rows = Vec::new();
    for i in 1..=20 {
        right_rows.push((Some(i), Some("one"), i * 3));
    }
    for i in 0..35 {
        right_rows.push((Some(100), Some("fan"), 100 + i));
    }
    for i in 0..15 {
        right_rows.push((Some(200), Some("mm"), 3000 + i));
    }
    for i in 600..610 {
        right_rows.push((Some(i), Some("unmatched_r"), i));
    }
    for i in 0..8 {
        right_rows.push((None, None, i));
        right_rows.push((Some(200), None, 400 + i));
        right_rows.push((None, Some("mm"), 500 + i));
    }

    let (l_schema, l_batches) = make_kv_table(&left_rows, 7)?;
    let (r_schema, r_batches) = make_kv_table(&right_rows, 5)?;
    let l_dir = write_parquet_files(&l_batches, 2, 11)?;
    let r_dir = write_parquet_files(&r_batches, 2, 13)?;

    let queries = [
        // Basic inner join with pure projection
        "SELECT l.k, l.v, r.v AS rv FROM l JOIN r ON l.k = r.k",
        // Full SELECT * (duplicate column names across left and right)
        "SELECT * FROM l JOIN r ON l.k = r.k",
        // Multi-column key (Int32, Utf8) with partial NULLs
        "SELECT l.k, l.s, l.v, r.v AS rv FROM l JOIN r ON l.k = r.k AND l.s = r.s",
        // Post-join cross-table Filter (pushed to join.filter) + Projection
        "SELECT l.k, l.v + r.v AS total FROM l JOIN r ON l.k = r.k WHERE l.v + r.v > 500 AND l.v > 0 AND r.v < 3010",
        // Q19-style OR residual join filter
        "SELECT l.k, l.v, r.v AS rv FROM l JOIN r ON l.k = r.k WHERE (l.s = 'one' AND r.v > 20) OR (l.s = 'mm' AND l.v + r.v > 5010)",
        // Volatile post-join Filter (stays above Join)
        "SELECT l.k, l.v, r.v AS rv FROM l JOIN r ON l.k = r.k WHERE random() >= 0.0",
        // Post-join COUNT(*)
        "SELECT COUNT(*) AS n FROM l JOIN r ON l.k = r.k",
        "SELECT COUNT(*) AS n FROM l JOIN r ON l.k = r.k WHERE l.v + r.v > 500",
        // Post-join GROUP BY integer key COUNT(*) (GroupCountSink)
        "SELECT l.k, COUNT(*) AS n FROM l JOIN r ON l.k = r.k GROUP BY l.k",
        // Post-join multi-aggregate GROUP BY (AggregateSink)
        "SELECT l.s, COUNT(*) AS n, SUM(l.v) AS sv, MIN(r.v) AS mn, MAX(r.v) AS mx, AVG(r.v) AS av \
         FROM l JOIN r ON l.k = r.k GROUP BY l.s",
        // Post-join ungrouped multi-aggregate
        "SELECT COUNT(*) AS n, SUM(l.v + r.v) AS total, AVG(l.v) AS avg_l FROM l JOIN r ON l.k = r.k",
        // Self-join with table aliases
        "SELECT a.k, a.v AS va, b.v AS vb FROM l a JOIN l b ON a.k = b.k WHERE a.v < b.v",
        // Expression join key
        "SELECT l.k, r.k AS rk, l.v FROM l JOIN r ON l.k + 1 = r.k",
    ];

    // Test both MemTable and Parquet, and small batch_size=7 to force multi-batch
    // HaveMoreOutput blowup chunking on 1-to-many and many-to-many keys!
    for (source_kind, batch_size, workers, pushdown) in [
        ("mem_small_batch_1w", 7, 1, false),
        ("mem_small_batch_4w", 7, 4, false),
        ("mem_batch_1", 1, 2, false),
        ("parquet_no_pd_4w", 7, 4, false),
        ("parquet_pd_4w", 7, 4, true),
    ] {
        let stock = stock_context(workers, batch_size);
        let native = morsel_context(workers, batch_size, MorselMode::Required, pushdown);

        if source_kind.starts_with("mem") {
            for ctx in [&stock, &native] {
                ctx.register_table(
                    "l",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&l_schema),
                        vec![l_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "r",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&r_schema),
                        vec![r_batches.clone()],
                    )?),
                )?;
            }
        } else {
            for ctx in [&stock, &native] {
                ctx.register_parquet(
                    "l",
                    l_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "r",
                    r_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
            }
        }

        for sql in queries {
            let (plan, actual) = assert_native_parity(&stock, &native, sql).await?;
            let display = displayable(plan.as_ref()).indent(true).to_string();
            assert!(
                !display.contains(", filter="),
                "{source_kind}: INNER join residual filter must lower as post-probe Filter, not in-probe filter:\n{display}"
            );
            assert!(
                metric(plan.as_ref(), "join_build_finalizations") >= 1,
                "{source_kind}: `{sql}`"
            );
            // Verify HaveMoreOutput bounds every emitted batch for Result sink queries.
            if !sql.contains("COUNT(") && !sql.contains("SUM(") {
                for batch in &actual {
                    assert!(
                        batch.num_rows() <= batch_size,
                        "{source_kind}: batch exceeded batch_size ({} > {batch_size}) for `{sql}`",
                        batch.num_rows()
                    );
                }
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn inner_join_empty_build_empty_probe_and_build_side_selection() -> Result<()> {
    let (schema, nonempty) = make_kv_table(
        &[
            (Some(1), Some("a"), 10),
            (Some(2), Some("b"), 20),
            (Some(3), Some("c"), 30),
            (Some(4), Some("d"), 40),
            (Some(5), Some("e"), 50),
        ],
        2,
    )?;
    let (small_schema, small) =
        make_kv_table(&[(Some(2), Some("b"), 200), (Some(4), Some("d"), 400)], 1)?;

    let stock = stock_context(4, 16);
    let native = morsel_context(4, 16, MorselMode::Required, false);

    for ctx in [&stock, &native] {
        ctx.register_table(
            "big",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![nonempty.clone()],
            )?),
        )?;
        ctx.register_table(
            "small",
            Arc::new(MemTable::try_new(
                Arc::clone(&small_schema),
                vec![small.clone()],
            )?),
        )?;
        ctx.register_table(
            "empty_t",
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]])?),
        )?;
    }

    // Empty on right AND empty on left: both must short-circuit and match stock!
    for sql in [
        "SELECT big.k, empty_t.v FROM big JOIN empty_t ON big.k = empty_t.k",
        "SELECT empty_t.k, big.v FROM empty_t JOIN big ON empty_t.k = big.k",
        "SELECT COUNT(*) FROM big JOIN empty_t ON big.k = empty_t.k",
        "SELECT COUNT(*) FROM empty_t JOIN big ON empty_t.k = big.k",
        "SELECT big.k, COUNT(*) FROM big JOIN empty_t ON big.k = empty_t.k GROUP BY big.k",
        "SELECT SUM(big.v), AVG(big.v) FROM big JOIN empty_t ON big.k = empty_t.k",
        "SELECT COUNT(*) FROM empty_t a JOIN empty_t b ON a.k = b.k",
        // Non-empty build filtered to 0 rows
        "SELECT big.k, small.v FROM big JOIN small ON big.k = small.k WHERE small.v < 0",
        "SELECT COUNT(*) FROM big JOIN small ON big.k = small.k WHERE small.v < 0",
    ] {
        let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
        assert_eq!(
            metric(plan.as_ref(), "join_empty_build_short_circuits"),
            1,
            "expected empty-build short-circuit for `{sql}`"
        );
    }

    // Smaller table on Left vs Right: planner picks smaller side as build while preserving
    // exact (left, right) output column semantics!
    let (p_left_small, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT small.k, small.v AS sv, big.v AS bv FROM small JOIN big ON small.k = big.k",
    )
    .await?;
    let disp_left = displayable(p_left_small.as_ref()).indent(true).to_string();
    assert!(
        disp_left.contains("build=Left"),
        "expected smaller left side on build:\n{disp_left}"
    );

    let (p_right_small, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT big.k, big.v AS bv, small.v AS sv FROM big JOIN small ON big.k = small.k",
    )
    .await?;
    let disp_right = displayable(p_right_small.as_ref()).indent(true).to_string();
    assert!(
        disp_right.contains("build=Right"),
        "expected smaller right side on build:\n{disp_right}"
    );

    // 3-way join: big JOIN small JOIN small (aliased)
    assert_native_parity(
        &stock,
        &native,
        "SELECT big.k, s1.v AS v1, s2.v AS v2 \
         FROM big JOIN small s1 ON big.k = s1.k JOIN small s2 ON s1.k = s2.k",
    )
    .await?;

    Ok(())
}

#[tokio::test]
async fn inner_join_all_supported_key_types_and_mixed_memtable_parquet() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k_i8", DataType::Int8, true),
        Field::new("k_i16", DataType::Int16, true),
        Field::new("k_i32", DataType::Int32, true),
        Field::new("k_i64", DataType::Int64, true),
        Field::new("k_u8", DataType::UInt8, true),
        Field::new("k_u16", DataType::UInt16, true),
        Field::new("k_u32", DataType::UInt32, true),
        Field::new("k_u64", DataType::UInt64, true),
        Field::new("k_f32", DataType::Float32, true),
        Field::new("k_f64", DataType::Float64, true),
        Field::new("k_dec", DataType::Decimal128(12, 2), true),
        Field::new("k_date", DataType::Date32, true),
        Field::new(
            "k_ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("k_bool", DataType::Boolean, true),
        Field::new("k_str", DataType::Utf8, true),
        Field::new("k_lstr", DataType::LargeUtf8, true),
        Field::new("k_sview", DataType::Utf8View, true),
        Field::new("k_bin", DataType::Binary, true),
        Field::new("payload", DataType::Int32, false),
    ]));

    let make_batch = |neg_zero: bool, offset: i32| -> Result<RecordBatch> {
        let z32 = if neg_zero { -0.0f32 } else { 0.0f32 };
        let z64 = if neg_zero { -0.0f64 } else { 0.0f64 };
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int8Array::from(vec![Some(1), Some(2), None, Some(2)])),
            Arc::new(Int16Array::from(vec![Some(10), Some(20), None, Some(20)])),
            Arc::new(Int32Array::from(vec![
                Some(100),
                Some(200),
                None,
                Some(200),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(1000),
                Some(2000),
                None,
                Some(2000),
            ])),
            Arc::new(UInt8Array::from(vec![Some(1), Some(2), None, Some(2)])),
            Arc::new(UInt16Array::from(vec![Some(10), Some(20), None, Some(20)])),
            Arc::new(UInt32Array::from(vec![
                Some(100),
                Some(200),
                None,
                Some(200),
            ])),
            Arc::new(UInt64Array::from(vec![
                Some(1000),
                Some(2000),
                None,
                Some(2000),
            ])),
            Arc::new(Float32Array::from(vec![
                Some(z32),
                Some(1.5),
                None,
                Some(1.5),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(z64),
                Some(2.5),
                None,
                Some(2.5),
            ])),
            Arc::new(
                Decimal128Array::from(vec![Some(12345), Some(67890), None, Some(67890)])
                    .with_precision_and_scale(12, 2)?,
            ),
            Arc::new(Date32Array::from(vec![
                Some(19000),
                Some(19001),
                None,
                Some(19001),
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_000_000),
                Some(2_000_000),
                None,
                Some(2_000_000),
            ])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                None,
                Some(false),
            ])),
            Arc::new(StringArray::from(vec![
                Some("alpha"),
                Some("beta"),
                None,
                Some("beta"),
            ])),
            Arc::new(LargeStringArray::from(vec![
                Some("l_alpha"),
                Some("l_beta"),
                None,
                Some("l_beta"),
            ])),
            Arc::new(StringViewArray::from(vec![
                Some("v_alpha_long_string_exceeding_inline_len"),
                Some("v_beta"),
                None,
                Some("v_beta"),
            ])),
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(b"b1"),
                Some(b"b2"),
                None,
                Some(b"b2"),
            ])),
            Arc::new(Int32Array::from(vec![
                offset + 1,
                offset + 2,
                offset + 3,
                offset + 4,
            ])),
        ];
        Ok(RecordBatch::try_new(Arc::clone(&schema), cols)?)
    };

    let left_batches = vec![make_batch(true, 10)?, make_batch(false, 20)?];
    let right_batches = vec![make_batch(false, 100)?];
    let right_dir = write_parquet_files(&right_batches, 1, 2)?;

    let stock = stock_context(4, 4);
    let native = morsel_context(4, 4, MorselMode::Required, false);

    // Mixed MemTable (left) JOIN Parquet (right)
    for ctx in [&stock, &native] {
        ctx.register_table(
            "t1",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![left_batches.clone()],
            )?),
        )?;
        ctx.register_parquet(
            "t2",
            right_dir.path().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    }

    for col in [
        "k_i8", "k_i16", "k_i32", "k_i64", "k_u8", "k_u16", "k_u32", "k_u64", "k_f32",
        "k_f64", "k_dec", "k_date", "k_ts", "k_bool", "k_str", "k_lstr", "k_sview",
        "k_bin",
    ] {
        let sql = format!(
            "SELECT t1.payload AS p1, t2.payload AS p2 FROM t1 JOIN t2 ON t1.{col} = t2.{col}"
        );
        assert_native_parity(&stock, &native, &sql).await?;
    }

    // Multi-column mixed types + CAST join key
    for sql in [
        "SELECT t1.payload AS p1, t2.payload AS p2 \
         FROM t1 JOIN t2 ON t1.k_i32 = t2.k_i32 AND t1.k_str = t2.k_str AND t1.k_f64 = t2.k_f64",
        "SELECT t1.payload AS p1, t2.payload AS p2 \
         FROM t1 JOIN t2 ON CAST(t1.k_i32 AS BIGINT) * 10 = t2.k_i64",
    ] {
        assert_native_parity(&stock, &native, sql).await?;
    }

    Ok(())
}

#[tokio::test]
async fn unsupported_joins_reject_in_required_and_fallback_in_prefer() -> Result<()> {
    let struct_fields = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("v", DataType::Int32, false),
        Field::new("st", DataType::Struct(struct_fields.clone()), false),
    ]));
    let mk_struct = |vals: Vec<i32>| -> Result<ArrayRef> {
        Ok(Arc::new(StructArray::try_new(
            struct_fields.clone(),
            vec![Arc::new(Int32Array::from(vals))],
            None,
        )?))
    };
    let b1 = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            mk_struct(vec![1, 2, 3])?,
        ],
    )?;
    let b2 = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(2), Some(3), None])),
            Arc::new(Int32Array::from(vec![200, 300, 400])),
            mk_struct(vec![2, 3, 4])?,
        ],
    )?;

    let stock = stock_context(2, 16);
    let required = morsel_context(2, 16, MorselMode::Required, false);
    let prefer = morsel_context(2, 16, MorselMode::Prefer, false);

    for ctx in [&stock, &required, &prefer] {
        ctx.register_table(
            "a",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![b1.clone()]],
            )?),
        )?;
        ctx.register_table(
            "b",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![b2.clone()]],
            )?),
        )?;
    }

    let unsupported_queries = [
        // Mark join (disjunctive EXISTS)
        "SELECT a.k FROM a WHERE a.k > 0 OR EXISTS (SELECT 1 FROM b WHERE a.k = b.k)",
        // Non-equi outer / semi join without equi-join keys
        "SELECT a.k, b.v FROM a LEFT JOIN b ON a.v < b.v",
        "SELECT a.k FROM a WHERE EXISTS (SELECT 1 FROM b WHERE a.v < b.v)",
        // Correlated null-aware NOT IN
        "SELECT a.k FROM a WHERE a.k NOT IN (SELECT b.k FROM b WHERE b.v = a.v)",
        // Unsupported Struct join key
        "SELECT a.v, b.v AS bv FROM a JOIN b ON a.st = b.st",
    ];

    for sql in unsupported_queries {
        let err = physical(&required, sql).await.unwrap_err();
        assert!(
            matches!(err, DataFusionError::Plan(_))
                && err.to_string().contains("morsel_mode=required"),
            "expected clear morsel_mode=required plan error for `{sql}`, got: {err}"
        );

        let prefer_plan = physical(&prefer, sql).await?;
        assert_ne!(
            prefer_plan.name(),
            "MorselExec",
            "expected stock fallback in prefer mode for `{sql}`"
        );
        let actual = collect(prefer_plan, prefer.task_ctx()).await?;
        let expected = collect(physical(&stock, sql).await?, stock.task_ctx()).await?;
        assert_eq!(
            sorted_rows(&actual)?,
            sorted_rows(&expected)?,
            "prefer fallback parity mismatch for `{sql}`"
        );
    }

    // Verify morsel_mode=off never uses MorselExec for INNER JOIN.
    let off_plan = physical(&stock, "SELECT a.k, b.v FROM a JOIN b ON a.k = b.k").await?;
    assert_ne!(off_plan.name(), "MorselExec");

    Ok(())
}

#[tokio::test]
async fn hash_join_metrics_concurrent_runs_and_parquet_oom_cleanup() -> Result<()> {
    // Skewed keys + Parquet build and probe files.
    let mut build_rows = Vec::new();
    for i in 0..500 {
        build_rows.push((Some(i), Some("dim"), i * 7));
    }
    let mut probe_rows = Vec::new();
    for i in 0..2000 {
        // 50% hit hot key 42, rest spread across 0..500
        let k = if i % 2 == 0 { 42 } else { i % 500 };
        probe_rows.push((Some(k), Some("fact"), i));
    }
    let (_, b_batches) = make_kv_table(&build_rows, 125)?;
    let (_, p_batches) = make_kv_table(&probe_rows, 200)?;
    let b_dir = write_parquet_files(&b_batches, 2, 125)?;
    let p_dir = write_parquet_files(&p_batches, 4, 200)?;

    let sql = "SELECT p.k, p.v, b.v AS bv FROM p JOIN b ON p.k = b.k";

    // 1. Verify #138 per-run metrics report separate build and probe pipelines,
    //    and concurrent executions of the same plan are isolated and exact.
    let collector = Arc::new(MorselMetrics::default());
    let mut config = SessionConfig::new()
        .with_target_partitions(4)
        .with_batch_size(128)
        .with_extension(Arc::clone(&collector));
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(4).unwrap(),
        )))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet(
        "b",
        b_dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    ctx.register_parquet(
        "p",
        p_dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    let plan = physical(&ctx, sql).await?;
    assert_eq!(plan.name(), "MorselExec");

    let (r1, r2) = tokio::try_join!(
        collect(Arc::clone(&plan), ctx.task_ctx()),
        collect(Arc::clone(&plan), ctx.task_ctx()),
    )?;
    assert_eq!(sorted_rows(&r1)?, sorted_rows(&r2)?);
    assert_eq!(r1.iter().map(RecordBatch::num_rows).sum::<usize>(), 2000);
    assert_eq!(metric(plan.as_ref(), "join_build_finalizations"), 2);

    // Wait briefly for detached worker metrics drops to publish both runs.
    let runs = timeout(Duration::from_secs(5), async {
        loop {
            let got = collector.take_runs();
            if got.len() == 2 {
                return got;
            }
            if !got.is_empty() {
                let mut all = got;
                while all.len() < 2 {
                    tokio::task::yield_now().await;
                    all.extend(collector.take_runs());
                }
                return all;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("expected 2 run metric reports");

    // Verify EXPLAIN and EXPLAIN ANALYZE display native MorselExec join pipelines cleanly.
    let explain_batches = ctx.sql(&format!("EXPLAIN {sql}")).await?.collect().await?;
    let explain_text = format!("{explain_batches:?}");
    assert!(explain_text.contains("MorselExec"), "{explain_text}");
    assert!(explain_text.contains("HashJoinBuildSink"), "{explain_text}");
    assert!(explain_text.contains("HashJoinProbe"), "{explain_text}");
    ctx.sql(&format!("EXPLAIN ANALYZE {sql}"))
        .await?
        .collect()
        .await?;

    for run in &runs {
        let mut build_claimed = 0usize;
        let mut probe_claimed = 0usize;
        let mut saw_combine = false;
        let mut saw_finalize = false;
        for m in run.iter() {
            let pipeline_label = m
                .labels()
                .iter()
                .find(|l| l.name() == "pipeline")
                .map(|l| l.value());
            match (m.value().name(), pipeline_label) {
                ("morsels_claimed", Some("build")) => {
                    build_claimed += m.value().as_usize()
                }
                ("morsels_claimed", Some("probe")) => {
                    probe_claimed += m.value().as_usize()
                }
                ("morsel_join_combine_work", Some("build")) => saw_combine = true,
                ("morsel_join_finalize_work", Some("build")) => saw_finalize = true,
                _ => {}
            }
        }
        assert!(build_claimed > 0, "expected build morsels_claimed > 0");
        assert!(probe_claimed > 0, "expected probe morsels_claimed > 0");
        assert!(saw_combine, "expected morsel_join_combine_work");
        assert!(saw_finalize, "expected morsel_join_finalize_work");
    }

    // 2. Verify Parquet build OOM fails cleanly and leaves pool.reserved() == 0.
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut oom_cfg = SessionConfig::new()
        .with_target_partitions(4)
        .with_batch_size(128);
    oom_cfg.options_mut().execution.morsel_mode = MorselMode::Required;
    let oom_state = SessionStateBuilder::new_with_default_features()
        .with_config(oom_cfg)
        .with_runtime_env(runtime)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(4).unwrap(),
        )))
        .build();
    let oom_ctx = SessionContext::new_with_state(oom_state);
    oom_ctx
        .register_parquet(
            "b",
            b_dir.path().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    oom_ctx
        .register_parquet(
            "p",
            p_dir.path().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;

    let oom_plan = physical(&oom_ctx, sql).await?;
    assert_eq!(oom_plan.name(), "MorselExec");
    let mut stream = execute_stream(Arc::clone(&oom_plan), oom_ctx.task_ctx())?;
    let mut got_oom = false;
    while let Some(item) = timeout(Duration::from_secs(10), stream.next())
        .await
        .unwrap()
    {
        if let Err(err) = item {
            assert!(
                matches!(err, DataFusionError::ResourcesExhausted(_)),
                "expected ResourcesExhausted, got {err}"
            );
            got_oom = true;
            break;
        }
    }
    assert!(got_oom, "expected OOM on tight memory pool");
    drop(stream);
    drop(oom_plan);
    timeout(Duration::from_secs(5), async {
        while pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("memory pool reservations leaked after Parquet join OOM");

    Ok(())
}

fn extract_build_segments(plan_display: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut rest = plan_display;
    while let Some(start) = rest.find("[Build:") {
        let candidate = &rest[start..];
        let mut depth = 0usize;
        let mut end = candidate.len();
        for (idx, ch) in candidate.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = idx + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        segments.push(&candidate[..end]);
        rest = &candidate[end..];
    }
    segments
}

#[tokio::test]
async fn three_way_join_keeps_largest_non_join_input_on_probe() -> Result<()> {
    // TPC-H Q3-shaped 3-way join: small (10 rows) JOIN mid (80 rows) JOIN big (800 rows),
    // where the non-join input `big` is the largest side and must stay on probe rather than
    // being hashed into a `[Build: ...]` pipeline.
    let small_rows: Vec<(Option<i32>, Option<&str>, i32)> =
        (0..10).map(|i| (Some(i), Some("s"), i * 10)).collect();
    let mid_rows: Vec<(Option<i32>, Option<&str>, i32)> =
        (0..80).map(|i| (Some(i % 12), Some("m"), i)).collect();
    let big_rows: Vec<(Option<i32>, Option<&str>, i32)> =
        (0..800).map(|i| (Some(i % 90), Some("b"), i * 3)).collect();

    let (small_schema, small_batches) = make_kv_table(&small_rows, 5)?;
    let (mid_schema, mid_batches) = make_kv_table(&mid_rows, 16)?;
    let (big_schema, big_batches) = make_kv_table(&big_rows, 64)?;
    let small_dir = write_parquet_files(&small_batches, 1, 10)?;
    let mid_dir = write_parquet_files(&mid_batches, 2, 40)?;
    let big_dir = write_parquet_files(&big_batches, 4, 100)?;

    let queries = [
        "SELECT small.k AS sk, mid.v AS mv, big.v AS bv \
         FROM small \
         JOIN mid ON small.k = mid.k \
         JOIN big ON mid.v = big.k",
        "SELECT COUNT(*) AS n \
         FROM small \
         JOIN mid ON small.k = mid.k \
         JOIN big ON mid.v = big.k",
        "SELECT small.k AS sk, COUNT(*) AS n, SUM(big.v) AS total \
         FROM small \
         JOIN mid ON small.k = mid.k \
         JOIN big ON mid.v = big.k \
         WHERE big.v > 10 \
         GROUP BY small.k",
    ];

    for is_parquet in [false, true] {
        let stock = stock_context(4, 32);
        let native = morsel_context(4, 32, MorselMode::Required, false);

        if is_parquet {
            for ctx in [&stock, &native] {
                ctx.register_parquet(
                    "small",
                    small_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "mid",
                    mid_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "big",
                    big_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
            }
        } else {
            for ctx in [&stock, &native] {
                ctx.register_table(
                    "small",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&small_schema),
                        vec![small_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "mid",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&mid_schema),
                        vec![mid_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "big",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&big_schema),
                        vec![big_batches.clone()],
                    )?),
                )?;
            }
        }

        for sql in queries {
            let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
            let display = displayable(plan.as_ref()).indent(true).to_string();
            let build_segments = extract_build_segments(&display);
            assert_eq!(
                build_segments.len(),
                2,
                "expected 2 build pipelines for 3-way join:\n{display}"
            );
            for segment in &build_segments {
                assert!(
                    !segment.contains("big"),
                    "largest table `big` must not be placed in a [Build: ...] pipeline:\n{display}"
                );
            }
            let after_builds = display
                .rsplit_once("]; ")
                .map(|(_, tail)| tail)
                .unwrap_or(&display);
            assert!(
                after_builds.contains("big"),
                "expected `big` on outer probe pipeline:\n{display}"
            );
            assert_eq!(
                metric(plan.as_ref(), "join_build_finalizations"),
                2,
                "expected 2 build finalizations for 3-way join `{sql}`"
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn outer_joins_parity_left_right_full_empty_nulls_fanout_and_filters() -> Result<()>
{
    // Small side vs big side so BOTH build orientations (ProbeOuter and BuildOuter)
    // and FullOuter with multi-match fanout, NULL keys, all-NULL keys, and residual
    // Join::filter across tiny batch sizes are exercised!
    let small_rows = vec![
        (Some(1), Some("one"), 10),
        (Some(2), Some("two"), 20),
        (Some(2), Some("two"), 25),
        (Some(2), Some("two"), 30),
        (Some(3), Some("three"), 35),
        (Some(4), Some("only_s"), 40),
        (Some(5), Some("fail_f"), 50),
        (Some(5), Some("fail_f"), 55),
        (None, Some("null_s"), 60),
        (Some(2), None, 65),
    ];
    let mut big_rows = Vec::new();
    for i in 0..4 {
        big_rows.push((Some(1), Some("one"), 100 + i));
    }
    for i in 0..6 {
        big_rows.push((Some(2), Some("two"), 200 + i * 10));
    }
    for i in 0..5 {
        big_rows.push((Some(5), Some("fail_f"), 500 + i));
    }
    for i in 0..10 {
        big_rows.push((Some(100 + i), Some("only_b"), 1000 + i));
    }
    for i in 0..4 {
        big_rows.push((None, Some("null_b"), -10 - i));
        big_rows.push((Some(2), None, -50 - i));
    }
    let all_null_rows = vec![
        (None, Some("n1"), 1),
        (None, Some("n2"), 2),
        (None, None, 3),
    ];

    let (s_schema, s_batches) = make_kv_table(&small_rows, 3)?;
    let (b_schema, b_batches) = make_kv_table(&big_rows, 5)?;
    let (n_schema, n_batches) = make_kv_table(&all_null_rows, 2)?;
    let s_dir = write_parquet_files(&s_batches, 2, 4)?;
    let b_dir = write_parquet_files(&b_batches, 2, 7)?;
    let n_dir = write_parquet_files(&n_batches, 1, 2)?;

    let queries = [
        // LEFT OUTER both orientations (big LEFT small = ProbeOuter, small LEFT big = BuildOuter)
        "SELECT b.k, b.v, s.v AS sv FROM b LEFT OUTER JOIN s ON b.k = s.k",
        "SELECT s.k, s.v, b.v AS bv FROM s LEFT OUTER JOIN b ON s.k = b.k",
        // RIGHT OUTER both orientations
        "SELECT b.k, b.v, s.v AS sv FROM b RIGHT OUTER JOIN s ON b.k = s.k",
        "SELECT s.k, s.v, b.v AS bv FROM s RIGHT OUTER JOIN b ON s.k = b.k",
        // FULL OUTER both orientations
        "SELECT b.k AS bk, b.v, s.k AS sk, s.v AS sv FROM b FULL OUTER JOIN s ON b.k = s.k",
        "SELECT s.k AS sk, s.v, b.k AS bk, b.v AS bv FROM s FULL OUTER JOIN b ON s.k = b.k",
        // Composite key (Int32, Utf8) with partial NULLs
        "SELECT b.k, b.s, b.v, s.v AS sv FROM b LEFT JOIN s ON b.k = s.k AND b.s = s.s",
        "SELECT b.k, b.s, b.v, s.v AS sv FROM b RIGHT JOIN s ON b.k = s.k AND b.s = s.s",
        "SELECT b.k, b.s, b.v, s.k AS sk, s.v AS sv FROM b FULL OUTER JOIN s ON b.k = s.k AND b.s = s.s",
        // Residual Join::filter (some candidates fail, some pass, some keys fail all candidates!)
        "SELECT b.k, b.v, s.v AS sv FROM b LEFT OUTER JOIN s ON b.k = s.k AND s.v >= 25 AND b.v <> 500",
        "SELECT s.k, s.v, b.v AS bv FROM s LEFT OUTER JOIN b ON s.k = b.k AND b.v >= 220 AND s.v <> 50",
        "SELECT b.k, b.v, s.v AS sv FROM b RIGHT OUTER JOIN s ON b.k = s.k AND b.v >= 220 AND s.v <> 50",
        "SELECT s.k, s.v, b.v AS bv FROM s RIGHT OUTER JOIN b ON s.k = b.k AND s.v >= 25 AND b.v <> 500",
        "SELECT b.k AS bk, b.v, s.k AS sk, s.v AS sv FROM b FULL OUTER JOIN s ON b.k = s.k AND b.v + s.v > 240 AND b.k <> 5",
        "SELECT s.k AS sk, s.v, b.k AS bk, b.v AS bv FROM s FULL OUTER JOIN b ON s.k = b.k AND b.v + s.v > 240 AND s.k <> 5",
        // Pre-join filter (BatchView selection) + outer join
        "SELECT b.k, b.v, s.v AS sv FROM (SELECT * FROM b WHERE v >= 100) b LEFT OUTER JOIN s ON b.k = s.k",
        "SELECT b.k, b.v, s.v AS sv FROM (SELECT * FROM b WHERE v >= 100) b RIGHT OUTER JOIN s ON b.k = s.k",
        "SELECT b.k, b.v, s.v AS sv FROM (SELECT * FROM b WHERE v >= 100) b FULL OUTER JOIN s ON b.k = s.k",
        // All-NULL side
        "SELECT b.k, b.v, n.v AS nv FROM b LEFT OUTER JOIN n ON b.k = n.k",
        "SELECT b.k, b.v, n.v AS nv FROM b RIGHT OUTER JOIN n ON b.k = n.k",
        "SELECT b.k AS bk, b.v, n.k AS nk, n.v AS nv FROM b FULL OUTER JOIN n ON b.k = n.k",
        "SELECT n.k, n.v, b.v AS bv FROM n LEFT OUTER JOIN b ON n.k = b.k",
        // Empty side (empty on right, empty on left, both empty)
        "SELECT b.k, b.v, e.v AS ev FROM b LEFT OUTER JOIN e ON b.k = e.k",
        "SELECT e.k, e.v, b.v AS bv FROM e LEFT OUTER JOIN b ON e.k = b.k",
        "SELECT b.k, b.v, e.v AS ev FROM b RIGHT OUTER JOIN e ON b.k = e.k",
        "SELECT e.k, e.v, b.v AS bv FROM e RIGHT OUTER JOIN b ON e.k = b.k",
        "SELECT b.k AS bk, b.v, e.k AS ek, e.v AS ev FROM b FULL OUTER JOIN e ON b.k = e.k",
        "SELECT e.k AS ek, e.v, b.k AS bk, b.v AS bv FROM e FULL OUTER JOIN b ON e.k = b.k",
        // COUNT(*) and GROUP BY over outer joins
        "SELECT COUNT(*) AS n FROM b LEFT OUTER JOIN s ON b.k = s.k",
        "SELECT COUNT(*) AS n FROM b RIGHT OUTER JOIN s ON b.k = s.k",
        "SELECT COUNT(*) AS n FROM b FULL OUTER JOIN s ON b.k = s.k",
        "SELECT COUNT(*) AS n FROM b FULL OUTER JOIN s ON b.k = s.k AND b.v + s.v > 240",
        "SELECT s.k, COUNT(b.v) AS cnt, SUM(b.v) AS sm FROM s LEFT OUTER JOIN b ON s.k = b.k GROUP BY s.k",
        "SELECT b.k, COUNT(s.v) AS cnt FROM b FULL OUTER JOIN s ON b.k = s.k GROUP BY b.k",
        // IS NOT DISTINCT FROM (NullEqualsNull)
        "SELECT b.k AS bk, b.v, s.k AS sk, s.v AS sv FROM b LEFT OUTER JOIN s ON b.k IS NOT DISTINCT FROM s.k",
        "SELECT b.k AS bk, b.v, s.k AS sk, s.v AS sv FROM b FULL OUTER JOIN s ON b.k IS NOT DISTINCT FROM s.k",
    ];

    for (kind, batch_size, workers, pushdown) in [
        ("mem_bs1_1w", 1, 1, false),
        ("mem_bs2_4w", 2, 4, false),
        ("mem_bs7_4w", 7, 4, false),
        ("parquet_bs3_4w", 3, 4, false),
        ("parquet_pd_bs7_4w", 7, 4, true),
    ] {
        let stock = stock_context(workers, batch_size);
        let native = morsel_context(workers, batch_size, MorselMode::Required, pushdown);

        if kind.starts_with("mem") {
            for ctx in [&stock, &native] {
                ctx.register_table(
                    "s",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&s_schema),
                        vec![s_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "b",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&b_schema),
                        vec![b_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "n",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&n_schema),
                        vec![n_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "e",
                    Arc::new(MemTable::try_new(Arc::clone(&s_schema), vec![vec![]])?),
                )?;
            }
        } else {
            for ctx in [&stock, &native] {
                ctx.register_parquet(
                    "s",
                    s_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "b",
                    b_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "n",
                    n_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_table(
                    "e",
                    Arc::new(MemTable::try_new(Arc::clone(&s_schema), vec![vec![]])?),
                )?;
            }
        }

        for sql in queries {
            let (_, actual) = assert_native_parity(&stock, &native, sql).await?;
            if !sql.contains("COUNT(") && !sql.contains("SUM(") {
                for batch in &actual {
                    assert!(
                        batch.num_rows() <= batch_size,
                        "{kind}: batch exceeded batch_size ({} > {batch_size}) for `{sql}`",
                        batch.num_rows()
                    );
                }
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn semi_and_anti_joins_parity_including_right_semi_right_anti_and_filters()
-> Result<()> {
    let small_rows = vec![
        (Some(1), Some("a"), 10),
        (Some(2), Some("b"), 20),
        (Some(2), Some("b"), 25),
        (Some(2), Some("b"), 30),
        (Some(3), Some("c"), 35),
        (Some(4), Some("d"), 40),
        (None, Some("n"), 50),
    ];
    let mut big_rows = Vec::new();
    for i in 0..5 {
        big_rows.push((Some(1), Some("a"), 10 + i));
    }
    for i in 0..6 {
        big_rows.push((Some(2), Some("b"), 15 + i * 5));
    }
    for i in 0..6 {
        big_rows.push((Some(10 + i), Some("x"), 100 + i));
    }
    for i in 0..3 {
        big_rows.push((None, Some("n"), -1 - i));
    }

    let (s_schema, s_batches) = make_kv_table(&small_rows, 2)?;
    let (b_schema, b_batches) = make_kv_table(&big_rows, 4)?;
    let s_dir = write_parquet_files(&s_batches, 2, 3)?;
    let b_dir = write_parquet_files(&b_batches, 2, 5)?;

    let sql_queries = [
        // EXISTS (LeftSemi) both orientations: big EXISTS small (ProbeSemi) & small EXISTS big (BuildSemi)
        "SELECT big_t.k, big_t.v FROM big_t WHERE EXISTS (SELECT 1 FROM small_t WHERE small_t.k = big_t.k)",
        "SELECT small_t.k, small_t.v FROM small_t WHERE EXISTS (SELECT 1 FROM big_t WHERE big_t.k = small_t.k)",
        // IN (SELECT ...) (LeftSemi) both orientations
        "SELECT big_t.k, big_t.v FROM big_t WHERE big_t.k IN (SELECT small_t.k FROM small_t)",
        "SELECT small_t.k, small_t.v FROM small_t WHERE small_t.k IN (SELECT big_t.k FROM big_t)",
        // NOT EXISTS (LeftAnti, non-null-aware) both orientations: ProbeAnti & BuildAnti
        "SELECT big_t.k, big_t.v FROM big_t WHERE NOT EXISTS (SELECT 1 FROM small_t WHERE small_t.k = big_t.k)",
        "SELECT small_t.k, small_t.v FROM small_t WHERE NOT EXISTS (SELECT 1 FROM big_t WHERE big_t.k = small_t.k)",
        // EXISTS with residual Join::filter (both orientations + duplicate chains!)
        "SELECT big_t.k, big_t.v FROM big_t WHERE EXISTS (SELECT 1 FROM small_t WHERE small_t.k = big_t.k AND small_t.v > big_t.v)",
        "SELECT small_t.k, small_t.v FROM small_t WHERE EXISTS (SELECT 1 FROM big_t WHERE big_t.k = small_t.k AND big_t.v > small_t.v)",
        // NOT EXISTS with residual Join::filter (both orientations + duplicate chains!)
        "SELECT big_t.k, big_t.v FROM big_t WHERE NOT EXISTS (SELECT 1 FROM small_t WHERE small_t.k = big_t.k AND small_t.v > big_t.v)",
        "SELECT small_t.k, small_t.v FROM small_t WHERE NOT EXISTS (SELECT 1 FROM big_t WHERE big_t.k = small_t.k AND big_t.v > small_t.v)",
        // String key EXISTS / NOT EXISTS with residual filter
        "SELECT big_t.k, big_t.s, big_t.v FROM big_t WHERE EXISTS (SELECT 1 FROM small_t WHERE small_t.s = big_t.s AND small_t.v <> big_t.v)",
        "SELECT big_t.k, big_t.s, big_t.v FROM big_t WHERE NOT EXISTS (SELECT 1 FROM small_t WHERE small_t.s = big_t.s AND small_t.v <> big_t.v)",
        // Empty side
        "SELECT big_t.k, big_t.v FROM big_t WHERE EXISTS (SELECT 1 FROM empty_t WHERE empty_t.k = big_t.k)",
        "SELECT big_t.k, big_t.v FROM big_t WHERE NOT EXISTS (SELECT 1 FROM empty_t WHERE empty_t.k = big_t.k)",
        "SELECT empty_t.k, empty_t.v FROM empty_t WHERE EXISTS (SELECT 1 FROM big_t WHERE big_t.k = empty_t.k)",
        "SELECT empty_t.k, empty_t.v FROM empty_t WHERE NOT EXISTS (SELECT 1 FROM big_t WHERE big_t.k = empty_t.k)",
        // COUNT(*) and GROUP BY on top of SEMI / ANTI
        "SELECT COUNT(*) AS n FROM big_t WHERE EXISTS (SELECT 1 FROM small_t WHERE small_t.k = big_t.k)",
        "SELECT COUNT(*) AS n FROM small_t WHERE EXISTS (SELECT 1 FROM big_t WHERE big_t.k = small_t.k)",
        "SELECT COUNT(*) AS n FROM big_t WHERE NOT EXISTS (SELECT 1 FROM small_t WHERE small_t.k = big_t.k)",
        "SELECT COUNT(*) AS n FROM small_t WHERE NOT EXISTS (SELECT 1 FROM big_t WHERE big_t.k = small_t.k)",
        "SELECT small_t.k, COUNT(*) AS n FROM small_t WHERE NOT EXISTS (SELECT 1 FROM big_t WHERE big_t.k = small_t.k AND big_t.v > small_t.v) GROUP BY small_t.k",
    ];

    for (kind, batch_size, workers) in [
        ("mem_bs1_1w", 1, 1),
        ("mem_bs2_4w", 2, 4),
        ("parquet_bs3_4w", 3, 4),
    ] {
        let stock = stock_context(workers, batch_size);
        let native = morsel_context(workers, batch_size, MorselMode::Required, false);

        if kind.starts_with("mem") {
            for ctx in [&stock, &native] {
                ctx.register_table(
                    "small_t",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&s_schema),
                        vec![s_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "big_t",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&b_schema),
                        vec![b_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "empty_t",
                    Arc::new(MemTable::try_new(Arc::clone(&s_schema), vec![vec![]])?),
                )?;
            }
        } else {
            for ctx in [&stock, &native] {
                ctx.register_parquet(
                    "small_t",
                    s_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "big_t",
                    b_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_table(
                    "empty_t",
                    Arc::new(MemTable::try_new(Arc::clone(&s_schema), vec![vec![]])?),
                )?;
            }
        }

        for sql in sql_queries {
            assert_native_parity(&stock, &native, sql).await?;
        }

        // Explicitly test DataFrame JoinType::RightSemi and JoinType::RightAnti
        // in BOTH table orientations (small-left vs big-left) and WITH/WITHOUT residual filter!
        use datafusion::common::Column;
        use datafusion::logical_expr::Expr;
        for join_type in [
            JoinType::LeftSemi,
            JoinType::RightSemi,
            JoinType::LeftAnti,
            JoinType::RightAnti,
        ] {
            for (l_name, r_name) in [
                ("big_t", "small_t"),
                ("small_t", "big_t"),
                ("big_t", "empty_t"),
                ("empty_t", "big_t"),
            ] {
                for with_filter in [false, true] {
                    let build_plan = |ctx: SessionContext| async move {
                        let left = ctx.table(l_name).await?;
                        let right = ctx.table(r_name).await?;
                        let filter = with_filter.then(|| {
                            Expr::Column(Column::new(Some(l_name), "v"))
                                .lt(Expr::Column(Column::new(Some(r_name), "v")))
                        });
                        left.join(right, join_type, &["k"], &["k"], filter)?
                            .create_physical_plan()
                            .await
                    };
                    let stock_plan = build_plan(stock.clone()).await?;
                    let expected =
                        collect(Arc::clone(&stock_plan), stock.task_ctx()).await?;
                    let native_plan = build_plan(native.clone()).await?;
                    assert_eq!(
                        native_plan.name(),
                        "MorselExec",
                        "{kind}: expected MorselExec for {l_name} {join_type:?} {r_name} filter={with_filter}"
                    );
                    let actual =
                        collect(Arc::clone(&native_plan), native.task_ctx()).await?;
                    assert_eq!(
                        sorted_rows(&actual)?,
                        sorted_rows(&expected)?,
                        "{kind}: parity mismatch for {l_name} {join_type:?} {r_name} filter={with_filter}"
                    );
                }
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn null_aware_not_in_parity_all_null_and_empty_combinations() -> Result<()> {
    let probe_rows = vec![
        (Some(1), Some("a"), 10),
        (Some(2), Some("b"), 20),
        (Some(3), Some("c"), 30),
        (Some(4), Some("d"), 40),
        (None, Some("n1"), 50),
        (None, None, 60),
        (Some(5), None, 70),
    ];
    let build_no_null = vec![(Some(2), Some("b"), 200), (Some(4), Some("d"), 400)];
    let build_with_null = vec![
        (Some(2), Some("b"), 200),
        (None, Some("x"), 300),
        (Some(4), Some("d"), 400),
    ];
    let build_only_null = vec![(None, None, 100), (None, None, 101)];

    let (p_schema, p_batches) = make_kv_table(&probe_rows, 2)?;
    let (bn_schema, bn_batches) = make_kv_table(&build_no_null, 1)?;
    let (bw_schema, bw_batches) = make_kv_table(&build_with_null, 1)?;
    let (bo_schema, bo_batches) = make_kv_table(&build_only_null, 1)?;
    let p_dir = write_parquet_files(&p_batches, 2, 3)?;
    let bn_dir = write_parquet_files(&bn_batches, 2, 1)?;
    let bw_dir = write_parquet_files(&bw_batches, 2, 1)?;
    let bo_dir = write_parquet_files(&bo_batches, 1, 2)?;

    let queries = [
        // 1. Empty build: ALL probe rows pass (including probe NULLs!)
        (
            "empty_build",
            "SELECT p.k, p.v FROM p WHERE p.k NOT IN (SELECT k FROM empty_b)",
            false,
        ),
        (
            "empty_filtered_build",
            "SELECT p.k, p.v FROM p WHERE p.k NOT IN (SELECT k FROM build_with_null WHERE v < 0)",
            false,
        ),
        // 2. Non-empty build WITH NULL: short-circuits with 0 rows!
        (
            "build_with_null",
            "SELECT p.k, p.v FROM p WHERE p.k NOT IN (SELECT k FROM build_with_null)",
            true,
        ),
        (
            "build_only_null",
            "SELECT p.k, p.v FROM p WHERE p.k NOT IN (SELECT k FROM build_only_null)",
            true,
        ),
        (
            "build_with_null_count",
            "SELECT COUNT(*) AS n FROM p WHERE p.k NOT IN (SELECT k FROM build_with_null)",
            true,
        ),
        // 3. Non-empty build WITHOUT NULL: probe NULLs NEVER pass, non-matching non-NULLs pass!
        (
            "build_no_null",
            "SELECT p.k, p.v FROM p WHERE p.k NOT IN (SELECT k FROM build_no_null)",
            false,
        ),
        (
            "build_null_filtered_out",
            "SELECT p.k, p.v FROM p WHERE p.k NOT IN (SELECT k FROM build_with_null WHERE v <> 300)",
            false,
        ),
        (
            "build_no_null_count",
            "SELECT COUNT(*) AS n FROM p WHERE p.k NOT IN (SELECT k FROM build_no_null)",
            false,
        ),
        (
            "build_no_null_agg",
            "SELECT p.k, SUM(p.v) AS sv FROM p WHERE p.k NOT IN (SELECT k FROM build_no_null) GROUP BY p.k ORDER BY p.k",
            false,
        ),
        // 4. String key NOT IN (chained hash table path)
        (
            "str_empty",
            "SELECT p.s, p.v FROM p WHERE p.s NOT IN (SELECT s FROM empty_b)",
            false,
        ),
        (
            "str_no_null",
            "SELECT p.s, p.v FROM p WHERE p.s NOT IN (SELECT s FROM build_no_null)",
            false,
        ),
        (
            "str_only_null",
            "SELECT p.s, p.v FROM p WHERE p.s NOT IN (SELECT s FROM build_only_null)",
            true,
        ),
    ];

    for (kind, batch_size, workers) in [
        ("mem_bs1_1w", 1, 1),
        ("mem_bs2_4w", 2, 4),
        ("parquet_bs2_4w", 2, 4),
    ] {
        let stock = stock_context(workers, batch_size);
        let native = morsel_context(workers, batch_size, MorselMode::Required, false);

        if kind.starts_with("mem") {
            for ctx in [&stock, &native] {
                ctx.register_table(
                    "p",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&p_schema),
                        vec![p_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "build_no_null",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&bn_schema),
                        vec![bn_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "build_with_null",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&bw_schema),
                        vec![bw_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "build_only_null",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&bo_schema),
                        vec![bo_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "empty_b",
                    Arc::new(MemTable::try_new(Arc::clone(&bn_schema), vec![vec![]])?),
                )?;
            }
        } else {
            for ctx in [&stock, &native] {
                ctx.register_parquet(
                    "p",
                    p_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "build_no_null",
                    bn_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "build_with_null",
                    bw_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "build_only_null",
                    bo_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_table(
                    "empty_b",
                    Arc::new(MemTable::try_new(Arc::clone(&bn_schema), vec![vec![]])?),
                )?;
            }
        }

        for (label, sql, expect_null_sc) in queries {
            let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
            if expect_null_sc {
                assert_eq!(
                    metric(plan.as_ref(), "join_null_aware_short_circuits"),
                    1,
                    "{kind}/{label}: expected null-aware short-circuit for `{sql}`"
                );
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn cross_join_parity_single_row_multi_row_empty_and_filters() -> Result<()> {
    let one_row = vec![(Some(42), Some("scalar"), 100)];
    let small_rows = vec![
        (Some(1), Some("x"), 10),
        (Some(2), Some("y"), 20),
        (None, Some("z"), 30),
    ];
    let big_rows: Vec<(Option<i32>, Option<&str>, i32)> =
        (0..15).map(|i| (Some(i), Some("row"), i * 10)).collect();

    let (o_schema, o_batches) = make_kv_table(&one_row, 1)?;
    let (s_schema, s_batches) = make_kv_table(&small_rows, 2)?;
    let (b_schema, b_batches) = make_kv_table(&big_rows, 4)?;
    let o_dir = write_parquet_files(&o_batches, 1, 1)?;
    let s_dir = write_parquet_files(&s_batches, 1, 2)?;
    let b_dir = write_parquet_files(&b_batches, 2, 5)?;

    let queries = [
        // Single-row build (both right and left!)
        "SELECT b.k, b.v, o.v AS threshold FROM b CROSS JOIN o",
        "SELECT o.v AS threshold, b.k, b.v FROM o CROSS JOIN b",
        // Single-row cross join + residual filter (TPC-H Q11/Q15/Q22 decorrelated scalar shape)
        "SELECT b.k, b.v FROM b CROSS JOIN o WHERE b.v > o.v",
        "SELECT b.k, b.v FROM o CROSS JOIN b WHERE b.v > o.v",
        // Multi-row x multi-row cross join
        "SELECT b.k, b.v, s.k AS sk, s.v AS sv FROM b CROSS JOIN s",
        "SELECT s.k AS sk, s.v AS sv, b.k, b.v FROM s CROSS JOIN b",
        // Multi-row cross join + residual filter
        "SELECT b.k, b.v, s.v AS sv FROM b CROSS JOIN s WHERE b.v + s.v > 100",
        // Pre-filter (BatchView selection) + cross join
        "SELECT b.k, b.v, s.v AS sv FROM (SELECT * FROM b WHERE v % 20 = 0) b CROSS JOIN s",
        // Empty sides
        "SELECT b.k, e.v FROM b CROSS JOIN e",
        "SELECT e.k, b.v FROM e CROSS JOIN b",
        // COUNT(*) and GROUP BY over cross join
        "SELECT COUNT(*) AS n FROM b CROSS JOIN s",
        "SELECT COUNT(*) AS n FROM b CROSS JOIN o",
        "SELECT COUNT(*) AS n FROM b CROSS JOIN e",
        "SELECT s.s, COUNT(*) AS n, SUM(b.v) AS sm FROM b CROSS JOIN s GROUP BY s.s ORDER BY s.s",
        // Chained cross join + hash join
        "SELECT b.k, b.v, o.v AS ov, s.v AS sv FROM b CROSS JOIN o JOIN s ON b.k = s.k WHERE b.v < o.v",
    ];

    for (kind, batch_size, workers) in [
        ("mem_bs1_1w", 1, 1),
        ("mem_bs3_4w", 3, 4),
        ("mem_bs64_4w", 64, 4),
        ("parquet_bs4_4w", 4, 4),
    ] {
        let stock = stock_context(workers, batch_size);
        let native = morsel_context(workers, batch_size, MorselMode::Required, false);

        if kind.starts_with("mem") {
            for ctx in [&stock, &native] {
                ctx.register_table(
                    "o",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&o_schema),
                        vec![o_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "s",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&s_schema),
                        vec![s_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "b",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&b_schema),
                        vec![b_batches.clone()],
                    )?),
                )?;
                ctx.register_table(
                    "e",
                    Arc::new(MemTable::try_new(Arc::clone(&s_schema), vec![vec![]])?),
                )?;
            }
        } else {
            for ctx in [&stock, &native] {
                ctx.register_parquet(
                    "o",
                    o_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "s",
                    s_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "b",
                    b_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_table(
                    "e",
                    Arc::new(MemTable::try_new(Arc::clone(&s_schema), vec![vec![]])?),
                )?;
            }
        }

        for sql in queries {
            let (plan, actual) = assert_native_parity(&stock, &native, sql).await?;
            let display = displayable(plan.as_ref()).indent(true).to_string();
            assert!(
                display.contains("CrossJoinProbe"),
                "{kind}: expected CrossJoinProbe in display for `{sql}`:\n{display}"
            );
            if !sql.contains("COUNT(") && !sql.contains("SUM(") {
                for batch in &actual {
                    assert!(
                        batch.num_rows() <= batch_size,
                        "{kind}: cross join batch exceeded batch_size ({} > {batch_size}) for `{sql}`",
                        batch.num_rows()
                    );
                }
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn tpch_join_patterns_q4_q13_q16_q21_q22_parity() -> Result<()> {
    // Build compact TPC-H-shaped tables exercising Q4, Q13, Q16, Q21, Q22 join shapes
    // on both MemTable and Parquet!
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int64, false),
        Field::new("o_custkey", DataType::Int64, false),
        Field::new("o_orderstatus", DataType::Utf8, false),
        Field::new("o_orderdate", DataType::Int32, false),
        Field::new("o_orderpriority", DataType::Utf8, false),
        Field::new("o_comment", DataType::Utf8, false),
    ]));
    let orders_batch = RecordBatch::try_new(
        Arc::clone(&orders_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(Int64Array::from(vec![10, 10, 20, 20, 30, 40, 40, 50])),
            Arc::new(StringArray::from(vec![
                "F", "F", "F", "O", "F", "F", "F", "O",
            ])),
            Arc::new(Int32Array::from(vec![
                100, 105, 110, 115, 120, 125, 130, 135,
            ])),
            Arc::new(StringArray::from(vec![
                "1-URGENT", "2-HIGH", "1-URGENT", "3-MEDIUM", "2-HIGH", "1-URGENT",
                "2-HIGH", "3-MEDIUM",
            ])),
            Arc::new(StringArray::from(vec![
                "regular order",
                "very special big requests here",
                "normal comment",
                "another special requests note",
                "plain",
                "special requests",
                "ok",
                "fine",
            ])),
        ],
    )?;

    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int64, false),
        Field::new("l_suppkey", DataType::Int64, false),
        Field::new("l_commitdate", DataType::Int32, false),
        Field::new("l_receiptdate", DataType::Int32, false),
    ]));
    // Order 1: supp 1 (late), supp 2 (on-time) -> Q21 MATCH for supp 1!
    // Order 2: supp 1 (late), supp 2 (late) -> Q21 EXCLUDED (l3 fails NOT EXISTS)!
    // Order 3: supp 1 (late), supp 1 (on-time) -> Q21 EXCLUDED (only 1 supplier, fails EXISTS l2.suppkey <> l1.suppkey)!
    // Order 5: supp 2 (late), supp 3 (on-time), supp 1 (on-time) -> Q21 MATCH for supp 2!
    // Order 6: supp 1 (late), supp 3 (on-time) -> Q21 MATCH for supp 1!
    let lineitem_batch = RecordBatch::try_new(
        Arc::clone(&lineitem_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 2, 3, 3, 5, 5, 5, 6, 6, 7])),
            Arc::new(Int64Array::from(vec![1, 2, 1, 2, 1, 1, 2, 3, 1, 1, 3, 2])),
            Arc::new(Int32Array::from(vec![
                10, 20, 10, 10, 10, 20, 10, 20, 20, 10, 20, 20,
            ])),
            Arc::new(Int32Array::from(vec![
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            ])),
        ],
    )?;

    let customer_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int64, false),
        Field::new("c_phone", DataType::Utf8, false),
        Field::new("c_acctbal", DataType::Float64, false),
    ]));
    // Cust 60, 70, 80 have NO orders (Q13 count=0 & Q22 NOT EXISTS)!
    let customer_batch = RecordBatch::try_new(
        Arc::clone(&customer_schema),
        vec![
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80])),
            Arc::new(StringArray::from(vec![
                "13-111", "31-222", "23-333", "99-444", "13-555", "13-666", "31-777",
                "23-888",
            ])),
            Arc::new(Float64Array::from(vec![
                100.0, 200.0, 50.0, 300.0, 400.0, 250.0, 150.0, 500.0,
            ])),
        ],
    )?;

    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, true),
        Field::new("s_name", DataType::Utf8, false),
        Field::new("s_nationkey", DataType::Int64, false),
        Field::new("s_comment", DataType::Utf8, false),
    ]));
    let supplier_batch = RecordBatch::try_new(
        Arc::clone(&supplier_schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3), Some(4)])),
            Arc::new(StringArray::from(vec![
                "Supplier#1",
                "Supplier#2",
                "Supplier#3",
                "Supplier#4",
            ])),
            Arc::new(Int64Array::from(vec![1, 1, 2, 1])),
            Arc::new(StringArray::from(vec![
                "good",
                "has Customer Complaints",
                "good",
                "bad Customer bad Complaints",
            ])),
        ],
    )?;

    let nation_schema = Arc::new(Schema::new(vec![
        Field::new("n_nationkey", DataType::Int64, false),
        Field::new("n_name", DataType::Utf8, false),
    ]));
    let nation_batch = RecordBatch::try_new(
        Arc::clone(&nation_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["SAUDI ARABIA", "CANADA"])),
        ],
    )?;

    let partsupp_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int64, false),
        Field::new("ps_suppkey", DataType::Int64, true),
    ]));
    let partsupp_batch = RecordBatch::try_new(
        Arc::clone(&partsupp_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 2, 3, 3])),
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                None,
                Some(1),
            ])),
        ],
    )?;

    let avg_threshold_schema = Arc::new(Schema::new(vec![Field::new(
        "avg_bal",
        DataType::Float64,
        false,
    )]));
    let avg_threshold_batch = RecordBatch::try_new(
        Arc::clone(&avg_threshold_schema),
        vec![Arc::new(Float64Array::from(vec![200.0]))],
    )?;

    let o_dir = write_parquet_files(std::slice::from_ref(&orders_batch), 2, 4)?;
    let l_dir = write_parquet_files(std::slice::from_ref(&lineitem_batch), 2, 4)?;
    let c_dir = write_parquet_files(std::slice::from_ref(&customer_batch), 2, 4)?;
    let s_dir = write_parquet_files(std::slice::from_ref(&supplier_batch), 1, 4)?;
    let n_dir = write_parquet_files(std::slice::from_ref(&nation_batch), 1, 2)?;
    let ps_dir = write_parquet_files(std::slice::from_ref(&partsupp_batch), 1, 3)?;
    let avg_dir = write_parquet_files(std::slice::from_ref(&avg_threshold_batch), 1, 1)?;

    let queries = [
        // Q4 pattern: EXISTS (LEFT SEMI) + GROUP BY + ORDER BY
        "SELECT o_orderpriority, COUNT(*) AS order_count \
         FROM orders \
         WHERE o_orderdate >= 100 AND o_orderdate < 130 \
           AND EXISTS ( \
               SELECT 1 FROM lineitem \
               WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate \
           ) \
         GROUP BY o_orderpriority \
         ORDER BY o_orderpriority",
        // Q13 pattern: LEFT OUTER JOIN with residual Join::filter (o_comment NOT LIKE) + GROUP BY
        "SELECT c_custkey, COUNT(o_orderkey) AS c_count \
         FROM customer LEFT OUTER JOIN orders \
           ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' \
         GROUP BY c_custkey \
         ORDER BY c_custkey",
        // Q16 pattern: null-aware NOT IN with filter on subquery
        "SELECT ps_partkey, COUNT(*) AS cnt \
         FROM partsupp \
         WHERE ps_suppkey NOT IN ( \
             SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%' \
         ) \
         GROUP BY ps_partkey \
         ORDER BY ps_partkey",
        // Q21 pattern: multi-way INNER + EXISTS with <> residual filter + NOT EXISTS with <> AND > residual filter + TopN!
        "SELECT s_name, COUNT(*) AS numwait \
         FROM supplier, lineitem l1, orders, nation \
         WHERE s_suppkey = l1.l_suppkey \
           AND o_orderkey = l1.l_orderkey \
           AND o_orderstatus = 'F' \
           AND l1.l_receiptdate > l1.l_commitdate \
           AND EXISTS ( \
               SELECT 1 FROM lineitem l2 \
               WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey \
           ) \
           AND NOT EXISTS ( \
               SELECT 1 FROM lineitem l3 \
               WHERE l3.l_orderkey = l1.l_orderkey \
                 AND l3.l_suppkey <> l1.l_suppkey \
                 AND l3.l_receiptdate > l3.l_commitdate \
           ) \
           AND s_nationkey = n_nationkey \
           AND n_name = 'SAUDI ARABIA' \
         GROUP BY s_name \
         ORDER BY numwait DESC, s_name \
         LIMIT 10",
        // Q22 pattern: single-row CROSS JOIN threshold + NOT EXISTS (LEFT ANTI) + GROUP BY + ORDER BY
        "SELECT substring(c_phone from 1 for 2) AS cntrycode, \
                COUNT(*) AS numcust, \
                SUM(c_acctbal) AS totacctbal \
         FROM customer CROSS JOIN avg_threshold \
         WHERE substring(c_phone from 1 for 2) IN ('13', '31', '23') \
           AND c_acctbal > avg_bal \
           AND NOT EXISTS ( \
               SELECT 1 FROM orders WHERE o_custkey = c_custkey \
           ) \
         GROUP BY substring(c_phone from 1 for 2) \
         ORDER BY cntrycode",
    ];

    for (is_parquet, batch_size) in [(false, 2), (false, 16), (true, 3)] {
        let stock = stock_context(4, batch_size);
        let native = morsel_context(4, batch_size, MorselMode::Required, false);

        if !is_parquet {
            for ctx in [&stock, &native] {
                ctx.register_table(
                    "orders",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&orders_schema),
                        vec![vec![orders_batch.clone()]],
                    )?),
                )?;
                ctx.register_table(
                    "lineitem",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&lineitem_schema),
                        vec![vec![lineitem_batch.clone()]],
                    )?),
                )?;
                ctx.register_table(
                    "customer",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&customer_schema),
                        vec![vec![customer_batch.clone()]],
                    )?),
                )?;
                ctx.register_table(
                    "supplier",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&supplier_schema),
                        vec![vec![supplier_batch.clone()]],
                    )?),
                )?;
                ctx.register_table(
                    "nation",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&nation_schema),
                        vec![vec![nation_batch.clone()]],
                    )?),
                )?;
                ctx.register_table(
                    "partsupp",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&partsupp_schema),
                        vec![vec![partsupp_batch.clone()]],
                    )?),
                )?;
                ctx.register_table(
                    "avg_threshold",
                    Arc::new(MemTable::try_new(
                        Arc::clone(&avg_threshold_schema),
                        vec![vec![avg_threshold_batch.clone()]],
                    )?),
                )?;
            }
        } else {
            for ctx in [&stock, &native] {
                ctx.register_parquet(
                    "orders",
                    o_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "lineitem",
                    l_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "customer",
                    c_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "supplier",
                    s_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "nation",
                    n_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "partsupp",
                    ps_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
                ctx.register_parquet(
                    "avg_threshold",
                    avg_dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
            }
        }

        for sql in queries {
            assert_native_parity(&stock, &native, sql).await?;
        }
    }

    Ok(())
}

#[tokio::test]
async fn chained_mixed_joins_short_circuit_and_build_emit_metrics() -> Result<()> {
    let a_rows = vec![
        (Some(1), Some("a1"), 10),
        (Some(2), Some("a2"), 20),
        (Some(3), Some("a3"), 30),
        (None, Some("an"), 40),
    ];
    let b_rows = vec![
        (Some(2), Some("b2"), 200),
        (Some(4), Some("b4"), 400),
        (None, Some("bn"), 500),
    ];
    let c_rows = vec![
        (Some(2), Some("c2"), 2000),
        (Some(4), Some("c4"), 4000),
        (Some(5), Some("c5"), 5000),
        (None, Some("cn"), 6000),
    ];
    let big_rows: Vec<(Option<i32>, Option<&str>, i32)> = (0..40)
        .map(|i| (Some((i % 6) + 1), Some("big"), i))
        .collect();

    let (a_schema, a_batches) = make_kv_table(&a_rows, 2)?;
    let (b_schema, b_batches) = make_kv_table(&b_rows, 2)?;
    let (c_schema, c_batches) = make_kv_table(&c_rows, 2)?;
    let (big_schema, big_batches) = make_kv_table(&big_rows, 5)?;

    let stock = stock_context(4, 3);
    let collector = Arc::new(MorselMetrics::default());
    let mut cfg = SessionConfig::new()
        .with_target_partitions(4)
        .with_batch_size(3)
        .with_extension(Arc::clone(&collector));
    cfg.options_mut().execution.morsel_mode = MorselMode::Required;
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(cfg)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(4).unwrap(),
        )))
        .build();
    let native = SessionContext::new_with_state(state);

    for ctx in [&stock, &native] {
        ctx.register_table(
            "a",
            Arc::new(MemTable::try_new(
                Arc::clone(&a_schema),
                vec![a_batches.clone()],
            )?),
        )?;
        ctx.register_table(
            "b",
            Arc::new(MemTable::try_new(
                Arc::clone(&b_schema),
                vec![b_batches.clone()],
            )?),
        )?;
        ctx.register_table(
            "c",
            Arc::new(MemTable::try_new(
                Arc::clone(&c_schema),
                vec![c_batches.clone()],
            )?),
        )?;
        ctx.register_table(
            "big",
            Arc::new(MemTable::try_new(
                Arc::clone(&big_schema),
                vec![big_batches.clone()],
            )?),
        )?;
        ctx.register_table(
            "empty_t",
            Arc::new(MemTable::try_new(Arc::clone(&a_schema), vec![vec![]])?),
        )?;
    }

    let chained_queries = [
        // Earlier INNER empty build short-circuits probe, but later RIGHT OUTER MUST still emit c!
        "SELECT a.v AS av, c.v AS cv \
         FROM (SELECT a.k, a.v FROM a JOIN empty_t ON a.k = empty_t.k) a \
         RIGHT OUTER JOIN c ON a.k = c.k",
        // Later INNER empty build cancels earlier RIGHT OUTER completely
        "SELECT b.v AS bv, empty_t.v AS ev \
         FROM (SELECT b.k, b.v FROM a RIGHT OUTER JOIN b ON a.k = b.k) b \
         JOIN empty_t ON b.k = empty_t.k",
        // Chained RIGHT OUTER -> RIGHT OUTER (child pipeline 0 feeds probe 1 and marks bits in table 1!)
        "SELECT a.v AS av, b.v AS bv, c.v AS cv \
         FROM a RIGHT OUTER JOIN b ON a.k = b.k RIGHT OUTER JOIN c ON b.k = c.k",
        // Chained LEFT OUTER -> FULL OUTER
        "SELECT a.v AS av, b.v AS bv, c.v AS cv \
         FROM a LEFT OUTER JOIN b ON a.k = b.k FULL OUTER JOIN c ON b.k = c.k",
        // Nested RIGHT OUTER inside a build pipeline (big JOIN (a RIGHT OUTER JOIN b))
        "SELECT big.v AS bigv, sub.av, sub.bv \
         FROM big JOIN (SELECT a.v AS av, b.k AS bk, b.v AS bv FROM a RIGHT OUTER JOIN b ON a.k = b.k) sub \
           ON big.k = sub.bk",
        // Nested FULL OUTER with aggregate
        "SELECT COUNT(*) AS n \
         FROM a RIGHT OUTER JOIN b ON a.k = b.k RIGHT OUTER JOIN c ON b.k = c.k",
    ];

    for sql in chained_queries {
        assert_native_parity(&stock, &native, sql).await?;
    }

    // Drain previous metrics, then run a RIGHT OUTER join where `b` (3 rows) is build
    // and `big` (40 rows) is probe, and assert #138 per-pipeline metrics report
    // `build`, `probe`, AND `build_emit`!
    let _ = collector.take_runs();
    let (plan, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT big.v AS bigv, b.v AS bv FROM big RIGHT OUTER JOIN b ON big.k = b.k",
    )
    .await?;
    assert_eq!(metric(plan.as_ref(), "join_build_emission_pipelines"), 1);

    let runs = timeout(Duration::from_secs(5), async {
        loop {
            let got = collector.take_runs();
            if !got.is_empty() {
                return got;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("expected metrics run report");

    let mut build_claimed = 0usize;
    let mut probe_claimed = 0usize;
    let mut emit_claimed = 0usize;
    for m in runs[0].iter() {
        let label = m
            .labels()
            .iter()
            .find(|l| l.name() == "pipeline")
            .map(|l| l.value());
        if m.value().name() == "morsels_claimed" {
            match label {
                Some("build") => build_claimed += m.value().as_usize(),
                Some("probe") => probe_claimed += m.value().as_usize(),
                Some("build_emit") => emit_claimed += m.value().as_usize(),
                _ => {}
            }
        }
    }
    assert!(build_claimed > 0, "expected build morsels_claimed > 0");
    assert!(probe_claimed > 0, "expected probe morsels_claimed > 0");
    assert!(emit_claimed > 0, "expected build_emit morsels_claimed > 0");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_join_filter_minmax_prunes_parquet_row_groups_and_bytes_scanned()
-> Result<()> {
    // 16 sorted row groups of 10,000 rows each (k in 0..160_000), wide payload.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("val", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let mut fact_batches = Vec::with_capacity(16);
    for rg in 0..16i64 {
        let base = rg * 10_000;
        fact_batches.push(RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(base..base + 10_000)),
                Arc::new(Int64Array::from_iter_values(
                    (base..base + 10_000).map(|v| v * 3),
                )),
                Arc::new(StringArray::from_iter_values(
                    (base..base + 10_000)
                        .map(|v| format!("payload-{v:08}-padding-to-make-rg-wide")),
                )),
            ],
        )?);
    }
    let fact_dir = write_parquet_files(&fact_batches, 4, 10_000)?;

    // Selective build side: 500 keys strictly inside RG 5 (50_000..55_000),
    // above the default 150-value InList cap so both min/max AND Bloom activate.
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("weight", DataType::Int64, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..500i64).map(|i| 50_000 + i * 10),
            )),
            Arc::new(Int64Array::from_iter_values(0..500i64)),
        ],
    )?;
    let dim_dir = write_parquet_files(&[dim_batch], 1, 10_000)?;

    let make_ctx = |mode: &str, pushdown: bool, dyn_join: bool| {
        let fact_path = fact_dir.path().to_str().unwrap().to_owned();
        let dim_path = dim_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            cfg.options_mut()
                .optimizer
                .enable_join_dynamic_filter_pushdown = dyn_join;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("fact", &fact_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("dim", &dim_path, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let sql = "SELECT COUNT(*) AS n, SUM(f.val + d.weight) AS total \
               FROM fact f INNER JOIN dim d ON f.k = d.dk";

    for pushdown in [false, true] {
        let stock = make_ctx("off", pushdown, true).await?;
        let native_off = make_ctx("required", pushdown, false).await?;
        let native_on = make_ctx("required", pushdown, true).await?;

        let (plan_off, _) = assert_native_parity(&stock, &native_off, sql).await?;
        let (plan_on, _) = assert_native_parity(&stock, &native_on, sql).await?;

        let bytes_off = metric(plan_off.as_ref(), "bytes_scanned");
        let bytes_on = metric(plan_on.as_ref(), "bytes_scanned");
        assert!(
            bytes_on < bytes_off / 3,
            "pushdown={pushdown}: expected dynamic min/max pruning to slash bytes_scanned \
             (on={bytes_on}, off={bytes_off})"
        );
        let morsels_off = metric(plan_off.as_ref(), "parquet_morsels_claimed");
        let morsels_on = metric(plan_on.as_ref(), "parquet_morsels_claimed");
        assert!(
            morsels_on < morsels_off,
            "pushdown={pushdown}: expected fewer claimed row groups (on={morsels_on}, off={morsels_off})"
        );
        assert!(
            metric(plan_on.as_ref(), "join_dynamic_filters_pushed") > 0,
            "pushdown={pushdown}: expected join_dynamic_filters_pushed > 0"
        );
        assert_eq!(
            metric(plan_off.as_ref(), "join_dynamic_filters_pushed"),
            0,
            "pushdown={pushdown}: expected 0 dynamic filters when disabled"
        );
        if pushdown {
            // `k >= min AND k <= max AND bloom(k)` on the 1 surviving file must
            // group into 1 ArrowPredicate so `k` is decoded once, not 3x.
            assert_eq!(
                metric(plan_on.as_ref(), "row_filter_predicates"),
                1,
                "expected dynamic bounds + bloom on `k` to group into 1 RowFilter predicate"
            );
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_join_filter_bloom_filters_full_domain_parquet_probe_rows() -> Result<()>
{
    // 10 row groups of 10,000 rows each (k in 0..100_000), plus Date32, Utf8, and wide payload.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("skey", DataType::Utf8, false),
        Field::new("dkey", DataType::Date32, false),
        Field::new("k2", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let mut fact_batches = Vec::with_capacity(10);
    for rg in 0..10i64 {
        let base = rg * 10_000;
        fact_batches.push(RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(base..base + 10_000)),
                Arc::new(StringArray::from_iter_values(
                    (base..base + 10_000).map(|v| format!("key-{v:06}")),
                )),
                Arc::new(Date32Array::from_iter_values(
                    (base..base + 10_000).map(|v| (v % 3650) as i32),
                )),
                Arc::new(Int32Array::from_iter_values(
                    (base..base + 10_000).map(|v| (v % 1000) as i32),
                )),
                Arc::new(StringArray::from_iter_values(
                    (base..base + 10_000)
                        .map(|v| format!("wide-payload-column-value-{v:08}")),
                )),
            ],
        )?);
    }
    let fact_dir = write_parquet_files(&fact_batches, 2, 10_000)?;

    // 500 keys spanning the ENTIRE domain [0, 99_800] (every 200th key), so min/max
    // prunes ZERO row groups — all probe row pruning comes from the split-block Bloom filter!
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("dskey", DataType::Utf8, false),
        Field::new("ddkey", DataType::Date32, false),
        Field::new("dk2", DataType::Int32, false),
        Field::new("tag", DataType::Int64, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values((0..500i64).map(|i| i * 200))),
            Arc::new(StringArray::from_iter_values(
                (0..500i64).map(|i| format!("key-{:06}", i * 200)),
            )),
            Arc::new(Date32Array::from_iter_values(
                (0..500i64).map(|i| ((i * 200) % 3650) as i32),
            )),
            Arc::new(Int32Array::from_iter_values(
                (0..500i64).map(|i| ((i * 200) % 1000) as i32),
            )),
            Arc::new(Int64Array::from_iter_values(0..500i64)),
        ],
    )?;
    let dim_dir = write_parquet_files(&[dim_batch], 1, 10_000)?;

    // Also create a tiny 10-row dim table to exercise the InList tier (<= 150 distinct keys).
    let tiny_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..10i64).map(|i| i * 10_000 + 7),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..10i64).map(|i| format!("key-{:06}", i * 10_000 + 7)),
            )),
            Arc::new(Date32Array::from_iter_values(
                (0..10i64).map(|i| ((i * 10_000 + 7) % 3650) as i32),
            )),
            Arc::new(Int32Array::from_iter_values(
                (0..10i64).map(|i| ((i * 10_000 + 7) % 1000) as i32),
            )),
            Arc::new(Int64Array::from_iter_values(0..10i64)),
        ],
    )?;
    let tiny_dir = write_parquet_files(&[tiny_batch], 1, 10_000)?;

    let make_ctx = |mode: &str, pushdown: bool, dyn_join: bool| {
        let fact_path = fact_dir.path().to_str().unwrap().to_owned();
        let dim_path = dim_dir.path().to_str().unwrap().to_owned();
        let tiny_path = tiny_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            cfg.options_mut()
                .optimizer
                .enable_join_dynamic_filter_pushdown = dyn_join;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("fact", &fact_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("dim", &dim_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("tiny", &tiny_path, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let stock = make_ctx("off", true, true).await?;
    let native = make_ctx("required", true, true).await?;

    // 1. Int64 Bloom filter over full domain: prunes >95% of probe rows inside RowFilter!
    let (plan_int, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT f.k, f.payload, d.tag FROM fact f INNER JOIN dim d ON f.k = d.dk",
    )
    .await?;
    let pruned_int = metric(plan_int.as_ref(), "pushdown_rows_pruned");
    let matched_int = metric(plan_int.as_ref(), "pushdown_rows_matched");
    assert!(
        pruned_int > 95_000,
        "expected Int64 Bloom RowFilter to prune >95,000 rows, got {pruned_int}"
    );
    assert!(
        matched_int < 3_000,
        "expected Int64 Bloom RowFilter to pass <3,000 rows (500 true + <2% FP), got {matched_int}"
    );

    // 2. Utf8 string Bloom filter over full domain: prunes >95% of probe rows!
    let (plan_str, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT f.k, f.payload, d.tag FROM fact f INNER JOIN dim d ON f.skey = d.dskey",
    )
    .await?;
    assert!(
        metric(plan_str.as_ref(), "pushdown_rows_pruned") > 95_000,
        "expected Utf8 Bloom RowFilter to prune >95,000 rows"
    );

    // 3. Date32 Bloom filter!
    let (plan_date, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT COUNT(*) FROM fact f INNER JOIN dim d ON f.dkey = d.ddkey",
    )
    .await?;
    assert!(
        metric(plan_date.as_ref(), "pushdown_rows_pruned") > 50_000,
        "expected Date32 Bloom RowFilter to prune >50,000 rows"
    );

    // 4. Multi-column composite Bloom filter (k, k2)!
    let (plan_multi, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT f.k, f.payload, d.tag FROM fact f INNER JOIN dim d ON f.k = d.dk AND f.k2 = d.dk2",
    )
    .await?;
    assert!(
        metric(plan_multi.as_ref(), "pushdown_rows_pruned") > 95_000,
        "expected multi-column Bloom RowFilter to prune >95,000 rows"
    );

    // 5. Small build InList tier (10 exact keys -> zero false positives in RowFilter!).
    let (plan_inlist, _) = assert_native_parity(
        &stock,
        &native,
        "SELECT f.k, f.payload, t.tag FROM fact f INNER JOIN tiny t ON f.k = t.dk",
    )
    .await?;
    assert_eq!(
        metric(plan_inlist.as_ref(), "pushdown_rows_matched"),
        10,
        "expected exact InList RowFilter to match exactly 10 rows with 0 false positives"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_join_filter_semantics_parity_outer_semi_anti_nulls_empty_floats_and_chained()
-> Result<()> {
    // Fact Parquet table (4000 rows across 4 row groups) with NULLs, duplicates, and float edge cases.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("k2", DataType::Int64, true),
        Field::new("fk", DataType::Float64, true),
        Field::new("fv", DataType::Int64, false),
    ]));
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int64Array::from_iter(
                (0..4000i64).map(|i| if i % 10 == 0 { None } else { Some(i % 200) }),
            )),
            Arc::new(Int64Array::from_iter(
                (0..4000i64).map(|i| if i % 13 == 0 { None } else { Some(i % 25) }),
            )),
            Arc::new(Float64Array::from_iter((0..4000i64).map(|i| {
                match i % 20 {
                    0 => None,
                    1 => Some(f64::NAN),
                    2 => Some(-0.0),
                    3 => Some(0.0),
                    m => Some(m as f64),
                }
            }))),
            Arc::new(Int64Array::from_iter_values(0..4000i64)),
        ],
    )?;
    let fact_dir = write_parquet_files(&[fact_batch], 1, 1000)?;

    // Dim1 Parquet table (30 rows, keys 10..40 plus NULLs and float NaN/-0.0).
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, true),
        Field::new("dk2", DataType::Int64, true),
        Field::new("dfk", DataType::Float64, true),
        Field::new("dv", DataType::Int64, false),
    ]));
    let dim1_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter(
                (0..30i64).map(|i| if i % 7 == 0 { None } else { Some(10 + i) }),
            )),
            Arc::new(Int64Array::from_iter(
                (0..30i64).map(|i| if i % 5 == 0 { None } else { Some(i % 10) }),
            )),
            Arc::new(Float64Array::from_iter((0..30i64).map(|i| match i % 6 {
                0 => None,
                1 => Some(f64::NAN),
                2 => Some(0.0),
                3 => Some(-0.0),
                m => Some(m as f64),
            }))),
            Arc::new(Int64Array::from_iter_values(100..130i64)),
        ],
    )?;
    let dim1_dir = write_parquet_files(&[dim1_batch], 1, 100)?;

    // Dim2 Parquet table (20 rows, keys 15..35 plus NULLs).
    let dim2_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter(
                (0..20i64).map(|i| if i == 0 { None } else { Some(15 + i) }),
            )),
            Arc::new(Int64Array::from_iter_values(0..20i64)),
            Arc::new(Float64Array::from_iter_values((0..20i64).map(|i| i as f64))),
            Arc::new(Int64Array::from_iter_values(200..220i64)),
        ],
    )?;
    let dim2_dir = write_parquet_files(&[dim2_batch], 1, 100)?;

    // All-NULL build Parquet table (5 rows, all keys NULL!).
    let all_null_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from(vec![None; 5])),
            Arc::new(Int64Array::from(vec![None; 5])),
            Arc::new(Float64Array::from(vec![None; 5])),
            Arc::new(Int64Array::from_iter_values(300..305i64)),
        ],
    )?;
    let all_null_dir = write_parquet_files(&[all_null_batch], 1, 100)?;

    // Empty Parquet table (0 rows).
    let empty_batch = RecordBatch::new_empty(Arc::clone(&dim_schema));
    let empty_dir = write_parquet_files(&[empty_batch], 1, 100)?;

    for pushdown in [false, true] {
        let make_ctx = |mode: &str| {
            let paths = [
                ("fact", fact_dir.path().to_str().unwrap().to_owned()),
                ("dim1", dim1_dir.path().to_str().unwrap().to_owned()),
                ("dim2", dim2_dir.path().to_str().unwrap().to_owned()),
                ("all_null", all_null_dir.path().to_str().unwrap().to_owned()),
                ("empty_dim", empty_dir.path().to_str().unwrap().to_owned()),
            ];
            let mode = mode.to_owned();
            async move {
                let mut cfg = SessionConfig::new().with_target_partitions(4);
                cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
                cfg.options_mut().execution.parquet.pruning = true;
                cfg.options_mut().execution.collect_statistics = false;
                let ctx = SessionContext::new_with_config(cfg);
                for (name, path) in paths {
                    ctx.register_parquet(name, &path, ParquetReadOptions::default())
                        .await?;
                }
                ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                    .await?;
                Ok::<_, DataFusionError>(ctx)
            }
        };

        let stock = make_ctx("off").await?;
        let native = make_ctx("required").await?;

        let queries = [
            // Standard join types with NULLs on both sides
            "SELECT f.fv, d.dv FROM fact f INNER JOIN dim1 d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f LEFT OUTER JOIN dim1 d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f RIGHT OUTER JOIN dim1 d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f FULL OUTER JOIN dim1 d ON f.k = d.dk",
            "SELECT f.fv FROM fact f WHERE EXISTS (SELECT 1 FROM dim1 d WHERE d.dk = f.k)",
            "SELECT d.dv FROM dim1 d WHERE EXISTS (SELECT 1 FROM fact f WHERE f.k = d.dk)",
            "SELECT f.fv FROM fact f WHERE NOT EXISTS (SELECT 1 FROM dim1 d WHERE d.dk = f.k)",
            "SELECT d.dv FROM dim1 d WHERE NOT EXISTS (SELECT 1 FROM fact f WHERE f.k = d.dk)",
            // Null-aware NOT IN (with NULLs, non-NULLs, empty)
            "SELECT f.fv FROM fact f WHERE f.k NOT IN (SELECT dk FROM dim1)",
            "SELECT f.fv FROM fact f WHERE f.k NOT IN (SELECT dk FROM dim1 WHERE dk IS NOT NULL)",
            "SELECT f.fv FROM fact f WHERE f.k NOT IN (SELECT dk FROM empty_dim)",
            // IS NOT DISTINCT FROM (NullEqualsNull): single-key & multi-key
            "SELECT f.fv, d.dv FROM fact f INNER JOIN dim1 d ON f.k IS NOT DISTINCT FROM d.dk",
            "SELECT f.fv, d.dv FROM fact f INNER JOIN dim1 d \
             ON f.k IS NOT DISTINCT FROM d.dk AND f.k2 IS NOT DISTINCT FROM d.dk2",
            "SELECT f.fv, d.dv FROM fact f INNER JOIN all_null d ON f.k IS NOT DISTINCT FROM d.dk",
            // Empty build & all-NULL build across INNER / OUTER / ANTI
            "SELECT f.fv, d.dv FROM fact f INNER JOIN empty_dim d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f LEFT OUTER JOIN empty_dim d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f RIGHT OUTER JOIN empty_dim d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f INNER JOIN all_null d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f LEFT OUTER JOIN all_null d ON f.k = d.dk",
            "SELECT f.fv, d.dv FROM fact f RIGHT OUTER JOIN all_null d ON f.k = d.dk",
            "SELECT d.dv FROM all_null d WHERE NOT EXISTS (SELECT 1 FROM fact f WHERE f.k = d.dk)",
            // Float keys with NaN and -0.0 / +0.0
            "SELECT COUNT(*) FROM fact f INNER JOIN dim1 d ON f.fk = d.dfk",
            // Multi-column equi-join with NULLs
            "SELECT f.fv, d.dv FROM fact f INNER JOIN dim1 d ON f.k = d.dk AND f.k2 = d.dk2",
            // Chained INNER -> INNER (both push into fact)
            "SELECT f.fv, d1.dv, d2.dv FROM fact f \
             INNER JOIN dim1 d1 ON f.k = d1.dk \
             INNER JOIN dim2 d2 ON f.k2 = d2.dk2",
            // Chained RIGHT OUTER (BuildOuter) -> INNER: dim2 MUST NOT push past BuildOuter!
            "SELECT f.fv, d1.dv, d2.dv FROM fact f \
             RIGHT OUTER JOIN dim1 d1 ON f.k = d1.dk \
             INNER JOIN dim2 d2 ON d1.dk IS NOT DISTINCT FROM d2.dk",
            // Chained LEFT OUTER (ProbeOuter) -> INNER on fact column
            "SELECT f.fv, d1.dv, d2.dv FROM fact f \
             LEFT OUTER JOIN dim1 d1 ON f.k = d1.dk \
             INNER JOIN dim2 d2 ON f.k2 = d2.dk2",
            // Mixed-origin multi-key join (one key from fact, one key from dim1 build)
            "SELECT f.fv, d1.dv, d2.dv FROM fact f \
             INNER JOIN dim1 d1 ON f.k = d1.dk \
             INNER JOIN dim2 d2 ON f.k2 = d2.dk2 AND d1.dk = d2.dk",
        ];

        for sql in queries {
            assert_native_parity(&stock, &native, sql).await?;
        }

        // Verify ProbeOuter (LEFT OUTER with probe=fact) NEVER pushes dynamic filters into fact!
        let (left_plan, _) = assert_native_parity(
            &stock,
            &native,
            "SELECT f.fv, d.dv FROM fact f LEFT OUTER JOIN dim1 d ON f.k = d.dk",
        )
        .await?;
        assert_eq!(
            metric(left_plan.as_ref(), "join_dynamic_filters_pushed"),
            0,
            "pushdown={pushdown}: ProbeOuter (LEFT JOIN) must NOT push dynamic join filters into probe"
        );

        // Verify BuildOuter with all-NULL build short-circuits probe ParquetScan
        // (0 probe bytes scanned on `fact`) while still emitting all 5 unmatched build rows!
        let (all_null_right, batches) = assert_native_parity(
            &stock,
            &native,
            "SELECT f.fv, d.dv FROM fact f RIGHT OUTER JOIN all_null d ON f.k = d.dk",
        )
        .await?;
        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 5);
        assert_eq!(
            metric(
                all_null_right.as_ref(),
                "join_dynamic_filter_short_circuits"
            ),
            1,
            "pushdown={pushdown}: all-NULL BuildOuter build should short-circuit probe ParquetScan"
        );
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_join_filter_concurrent_repeated_runs_and_memory_cleanup() -> Result<()> {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..20_000i64)),
            Arc::new(Int64Array::from_iter_values(0..20_000i64)),
        ],
    )?;
    let fact_dir = write_parquet_files(&[fact_batch], 1, 5_000)?;

    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("dv", DataType::Int64, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..300i64).map(|i| 5_000 + i * 10),
            )),
            Arc::new(Int64Array::from_iter_values(0..300i64)),
        ],
    )?;
    let dim_dir = write_parquet_files(&[dim_batch], 1, 1_000)?;

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut cfg = SessionConfig::new().with_target_partitions(4);
    cfg.options_mut().execution.parquet.pushdown_filters = true;
    cfg.options_mut().execution.parquet.pruning = true;
    let ctx = SessionContext::new_with_config_rt(cfg, runtime);
    ctx.register_parquet(
        "fact",
        fact_dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    ctx.register_parquet(
        "dim",
        dim_dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    ctx.sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;

    let sql =
        "SELECT COUNT(*) AS n, SUM(f.v + d.dv) AS s FROM fact f JOIN dim d ON f.k = d.dk";
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let expected = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    let expected_rows = sorted_rows(&expected)?;

    // Execute the exact same prepared plan 4 times concurrently to prove zero
    // cross-run mutation of ParquetScan or dynamic join filters.
    let runs = futures::future::try_join_all(
        (0..4).map(|_| collect(Arc::clone(&plan), ctx.task_ctx())),
    )
    .await?;
    for got in runs {
        assert_eq!(sorted_rows(&got)?, expected_rows);
    }
    assert_eq!(
        pool.reserved(),
        0,
        "memory pool must be fully released after runs"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_join_filter_cascades_across_multi_join_build_chain() -> Result<()> {
    // Mirror TPC-H Q5's cascading star-schema join chain:
    //   `region (1 row: r_key = 2)` -> `nation (25 rows -> 5 rows)`
    //   -> `supplier (sorted by s_nationkey across 10 RGs)` -> `fact`
    // Reverse build order must execute `region` -> `nation` -> `supplier` -> `fact`
    // so `nation`'s 5 keys prune `supplier` row groups via dynamic InList/bounds
    // BEFORE `supplier` opens its Parquet row groups.
    let region_schema = Arc::new(Schema::new(vec![
        Field::new("r_key", DataType::Int64, false),
        Field::new("r_name", DataType::Utf8, false),
    ]));
    let region_batch = RecordBatch::try_new(
        region_schema,
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                "AFRICA",
                "AMERICA",
                "ASIA",
                "EUROPE",
                "MIDDLE EAST",
            ])),
        ],
    )?;
    let region_dir = write_parquet_files(&[region_batch], 1, 100)?;

    let nation_schema = Arc::new(Schema::new(vec![
        Field::new("n_key", DataType::Int64, false),
        Field::new("n_region", DataType::Int64, false),
        Field::new("n_name", DataType::Utf8, false),
    ]));
    let nation_batch = RecordBatch::try_new(
        nation_schema,
        vec![
            Arc::new(Int64Array::from_iter_values(0..25i64)),
            Arc::new(Int64Array::from_iter_values((0..25i64).map(|i| i / 5))),
            Arc::new(StringArray::from_iter_values(
                (0..25i64).map(|i| format!("NATION_{i:02}")),
            )),
        ],
    )?;
    let nation_dir = write_parquet_files(&[nation_batch], 1, 100)?;

    // Supplier sorted by `s_nation` across 25 row groups (1 RG per nation, 200 rows/RG),
    // with wide padding so pruning 20/25 RGs visibly reduces `bytes_scanned`.
    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_key", DataType::Int64, false),
        Field::new("s_nation", DataType::Int64, false),
        Field::new("s_pad", DataType::Utf8, false),
    ]));
    let mut supplier_batches = Vec::with_capacity(25);
    for nation in 0..25i64 {
        let base = nation * 200;
        supplier_batches.push(RecordBatch::try_new(
            Arc::clone(&supplier_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(base..base + 200)),
                Arc::new(Int64Array::from_iter_values(std::iter::repeat_n(
                    nation, 200,
                ))),
                Arc::new(StringArray::from_iter_values((0..200).map(|i| {
                    format!("supplier-padding-{nation:02}-{i:04}-xxxxxxxxxxxxxxxxxxxx")
                }))),
            ],
        )?);
    }
    let supplier_dir = write_parquet_files(&supplier_batches, 5, 200)?;

    // Fact table (20,000 rows across 4 RGs) joining `supplier` on `(f_skey, f_nation)`.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("f_skey", DataType::Int64, false),
        Field::new("f_nation", DataType::Int64, false),
        Field::new("f_val", DataType::Int64, false),
    ]));
    let mut fact_batches = Vec::with_capacity(4);
    for rg in 0..4i64 {
        let base = rg * 5_000;
        fact_batches.push(RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (base..base + 5_000).map(|i| i % 5_000),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (base..base + 5_000).map(|i| (i % 5_000) / 200),
                )),
                Arc::new(Int64Array::from_iter_values(base..base + 5_000)),
            ],
        )?);
    }
    let fact_dir = write_parquet_files(&fact_batches, 2, 5_000)?;

    let make_ctx = |mode: &str, dyn_join: bool| {
        let paths = [
            ("region", region_dir.path().to_str().unwrap().to_owned()),
            ("nation", nation_dir.path().to_str().unwrap().to_owned()),
            ("supplier", supplier_dir.path().to_str().unwrap().to_owned()),
            ("fact", fact_dir.path().to_str().unwrap().to_owned()),
        ];
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            cfg.options_mut()
                .optimizer
                .enable_join_dynamic_filter_pushdown = dyn_join;
            let ctx = SessionContext::new_with_config(cfg);
            for (name, path) in paths {
                ctx.register_parquet(name, &path, ParquetReadOptions::default())
                    .await?;
            }
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let sql = "SELECT n.n_name, SUM(f.f_val) AS total \
               FROM fact f \
               INNER JOIN supplier s ON f.f_skey = s.s_key AND f.f_nation = s.s_nation \
               INNER JOIN nation n ON s.s_nation = n.n_key \
               INNER JOIN region r ON n.n_region = r.r_key \
               WHERE r.r_name = 'ASIA' \
               GROUP BY n.n_name";

    let stock = make_ctx("off", true).await?;
    let native_off = make_ctx("required", false).await?;
    let native_on = make_ctx("required", true).await?;

    let (plan_off, _) = assert_native_parity(&stock, &native_off, sql).await?;
    let (plan_on, _) = assert_native_parity(&stock, &native_on, sql).await?;

    let bytes_off = metric(plan_off.as_ref(), "bytes_scanned");
    let bytes_on = metric(plan_on.as_ref(), "bytes_scanned");
    assert!(
        bytes_on < bytes_off,
        "expected cascading cross-build dynamic filters to reduce bytes_scanned \
         (on={bytes_on}, off={bytes_off})"
    );
    assert!(
        metric(plan_on.as_ref(), "join_dynamic_filters_pushed") >= 3,
        "expected cascading dynamic filters across region -> nation -> supplier -> fact"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_join_filter_preserves_cross_column_guards_on_cast_join_key() -> Result<()>
{
    // Regression test: `fact JOIN dim ON CAST(f.s AS BIGINT) = d.dk WHERE f.s <> '' AND f.kind = 'num'`
    // with `pushdown_filters = true` and `enable_join_dynamic_filter_pushdown = true`
    // must never evaluate `CAST(f.s AS BIGINT)` on rows rejected by `f.kind = 'num'`
    // (such as `'abcN'` rows) or `f.s <> ''`.
    let n = 4000usize;
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("s", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let s_vals: Vec<String> = (0..n)
        .map(|i| match i % 4 {
            0 => String::new(),
            1 => format!("abc{i}"),
            _ => format!("{}", 10 + (i % 500)),
        })
        .collect();
    let kind_vals: Vec<&str> = (0..n)
        .map(|i| if i % 4 == 1 { "txt" } else { "num" })
        .collect();
    let fact_batch = RecordBatch::try_new(
        fact_schema,
        vec![
            Arc::new(StringArray::from(s_vals)),
            Arc::new(StringArray::from(kind_vals)),
            Arc::new(Int64Array::from_iter_values(0..n as i64)),
        ],
    )?;
    let fact_dir = write_parquet_files(&[fact_batch], 2, 500)?;

    // Build dim with 200 keys (> 150 InList threshold so both bounds AND Bloom fire).
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("dv", DataType::Int64, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        dim_schema,
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..200i64).map(|i| 10 + i * 2),
            )),
            Arc::new(Int64Array::from_iter_values(0..200i64)),
        ],
    )?;
    let dim_dir = write_parquet_files(&[dim_batch], 1, 200)?;

    let sql = "SELECT COUNT(*) AS n, SUM(f.v + d.dv) AS total \
               FROM fact f INNER JOIN dim d ON CAST(f.s AS BIGINT) = d.dk \
               WHERE f.s <> '' AND f.kind = 'num'";

    for reorder in [false, true] {
        for dyn_join in [false, true] {
            let make_ctx = |mode: &str| {
                let fact_path = fact_dir.path().to_str().unwrap().to_owned();
                let dim_path = dim_dir.path().to_str().unwrap().to_owned();
                let mode = mode.to_owned();
                async move {
                    let mut cfg = SessionConfig::new().with_target_partitions(4);
                    cfg.options_mut().execution.parquet.pushdown_filters = true;
                    cfg.options_mut().execution.parquet.reorder_filters = reorder;
                    cfg.options_mut().execution.parquet.pruning = true;
                    cfg.options_mut().execution.collect_statistics = false;
                    cfg.options_mut()
                        .optimizer
                        .enable_join_dynamic_filter_pushdown = dyn_join;
                    let ctx = SessionContext::new_with_config(cfg);
                    ctx.register_parquet(
                        "fact",
                        &fact_path,
                        ParquetReadOptions::default(),
                    )
                    .await?;
                    ctx.register_parquet("dim", &dim_path, ParquetReadOptions::default())
                        .await?;
                    ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                        .await?;
                    Ok::<_, DataFusionError>(ctx)
                }
            };

            let stock = make_ctx("off").await?;
            let native = make_ctx("required").await?;
            let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
            if dyn_join {
                assert!(
                    metric(plan.as_ref(), "join_dynamic_filters_pushed") > 0,
                    "reorder={reorder}: expected dynamic filter pushdown on CAST join key"
                );
            }
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cross_aggregate_dynamic_join_filter_q17_and_q20_patterns_prune_before_aggregate()
-> Result<()> {
    // 16 sorted row groups of 10,000 rows each:
    // l_partkey in 0..16_000 (10 rows per partkey, clustered by RG: RG r holds partkeys [r*1000..(r+1)*1000)),
    // l_suppkey in (l_partkey % 200), l_quantity in 1..=20, l_extendedprice = 100..
    let li_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int64, false),
        Field::new("l_suppkey", DataType::Int64, false),
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Int64, false),
        Field::new("l_comment", DataType::Utf8, false),
    ]));
    let mut li_batches = Vec::with_capacity(16);
    for rg in 0..16i64 {
        let mut pks = Vec::with_capacity(10_000);
        let mut sks = Vec::with_capacity(10_000);
        let mut qtys = Vec::with_capacity(10_000);
        let mut prices = Vec::with_capacity(10_000);
        let mut comments = Vec::with_capacity(10_000);
        for i in 0..10_000i64 {
            let pk = rg * 1_000 + (i / 10);
            let sk = (pk % 200) + (i % 2);
            let qty = (i % 20) + 1;
            pks.push(pk);
            sks.push(sk);
            qtys.push(qty);
            prices.push(qty * 100 + (pk % 17));
            comments.push(format!("comment-{rg:02}-{i:05}-padding-wide-payload"));
        }
        li_batches.push(RecordBatch::try_new(
            Arc::clone(&li_schema),
            vec![
                Arc::new(Int64Array::from(pks)),
                Arc::new(Int64Array::from(sks)),
                Arc::new(Int64Array::from(qtys)),
                Arc::new(Int64Array::from(prices)),
                Arc::new(StringArray::from(comments)),
            ],
        )?);
    }
    let li_dir = write_parquet_files(&li_batches, 4, 10_000)?;

    // Part table: 16,000 parts, with only 120 parts inside RG 5 (partkeys 5_000..5_120)
    // matching p_brand = 'Brand#23' AND p_container = 'MED BOX'.
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_container", DataType::Utf8, false),
    ]));
    let part_batch = RecordBatch::try_new(
        Arc::clone(&part_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..16_000i64)),
            Arc::new(StringArray::from_iter_values((0..16_000i64).map(|k| {
                if (5_000..5_120).contains(&k) {
                    "Brand#23"
                } else {
                    "Brand#99"
                }
            }))),
            Arc::new(StringArray::from_iter_values((0..16_000i64).map(|k| {
                if (5_000..5_120).contains(&k) {
                    "MED BOX"
                } else {
                    "LG CASE"
                }
            }))),
        ],
    )?;
    let part_dir = write_parquet_files(&[part_batch], 1, 16_000)?;

    // Partsupp table: 2 suppliers per part, for all 16,000 parts (32,000 rows).
    let ps_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int64, false),
        Field::new("ps_suppkey", DataType::Int64, false),
        Field::new("ps_availqty", DataType::Int64, false),
    ]));
    let mut ps_pk = Vec::with_capacity(32_000);
    let mut ps_sk = Vec::with_capacity(32_000);
    let mut ps_avail = Vec::with_capacity(32_000);
    for pk in 0..16_000i64 {
        for j in 0..2i64 {
            ps_pk.push(pk);
            ps_sk.push((pk % 200) + j);
            ps_avail.push(30 + ((pk + j) % 40));
        }
    }
    let ps_batch = RecordBatch::try_new(
        Arc::clone(&ps_schema),
        vec![
            Arc::new(Int64Array::from(ps_pk)),
            Arc::new(Int64Array::from(ps_sk)),
            Arc::new(Int64Array::from(ps_avail)),
        ],
    )?;
    let ps_dir = write_parquet_files(&[ps_batch], 1, 16_000)?;

    let make_ctx = |mode: &str, pushdown: bool, dyn_join: bool| {
        let li_path = li_dir.path().to_str().unwrap().to_owned();
        let part_path = part_dir.path().to_str().unwrap().to_owned();
        let ps_path = ps_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            cfg.options_mut()
                .optimizer
                .enable_join_dynamic_filter_pushdown = dyn_join;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("part", &part_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("partsupp", &ps_path, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    // 1. Q20 pattern: multi-key GROUP BY (l_partkey, l_suppkey) feeding LeftSemi join
    //    against partsupp semi-joined with selective filtered part.
    let q20_sql = "SELECT ps.ps_partkey, ps.ps_suppkey \
                   FROM partsupp ps \
                   WHERE ps.ps_partkey IN ( \
                       SELECT p.p_partkey FROM part p \
                       WHERE p.p_brand = 'Brand#23' AND p.p_container = 'MED BOX' \
                   ) \
                   AND ps.ps_availqty > ( \
                       SELECT 0.5 * SUM(l.l_quantity) \
                       FROM lineitem l \
                       WHERE l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey \
                   ) \
                   ORDER BY ps.ps_partkey, ps.ps_suppkey";

    // 2. Q17 pattern: single-key GROUP BY l_partkey scalar subquery joined with
    //    lineitem INNER JOIN selective part.
    let q17_sql = "SELECT SUM(l.l_extendedprice) AS total_price \
                   FROM lineitem l, part p \
                   WHERE p.p_partkey = l.l_partkey \
                     AND p.p_brand = 'Brand#23' \
                     AND p.p_container = 'MED BOX' \
                     AND l.l_quantity < ( \
                         SELECT 0.5 * AVG(l2.l_quantity) \
                         FROM lineitem l2 \
                         WHERE l2.l_partkey = p.p_partkey \
                     )";

    // 3. GroupCountSink pattern: single-key GROUP BY l_partkey COUNT(*) joined with part.
    let group_count_sql = "SELECT p.p_partkey, g.cnt \
                           FROM part p \
                           INNER JOIN ( \
                               SELECT l_partkey, COUNT(*) AS cnt \
                               FROM lineitem \
                               GROUP BY l_partkey \
                           ) g ON p.p_partkey = g.l_partkey \
                           WHERE p.p_brand = 'Brand#23' AND p.p_container = 'MED BOX' \
                           ORDER BY p.p_partkey";

    for (label, sql) in [
        ("q20", q20_sql),
        ("q17", q17_sql),
        ("group_count", group_count_sql),
    ] {
        for pushdown in [false, true] {
            let stock = make_ctx("off", pushdown, true).await?;
            let native_off = make_ctx("required", pushdown, false).await?;
            let native_on = make_ctx("required", pushdown, true).await?;

            let (plan_off, _) = assert_native_parity(&stock, &native_off, sql).await?;
            let (plan_on, _) = assert_native_parity(&stock, &native_on, sql).await?;

            let bytes_off = metric(plan_off.as_ref(), "bytes_scanned");
            let bytes_on = metric(plan_on.as_ref(), "bytes_scanned");
            let morsels_off = metric(plan_off.as_ref(), "parquet_morsels_claimed");
            let morsels_on = metric(plan_on.as_ref(), "parquet_morsels_claimed");
            // `part` itself is scanned in both arms (~157 KB), while `lineitem`
            // has 16 row groups (15 of which must be pruned).
            assert!(
                bytes_on < (bytes_off * 2) / 3,
                "{label} pushdown={pushdown}: expected cross-Aggregate dynamic pushdown \
                 to slash bytes_scanned (on={bytes_on}, off={bytes_off})"
            );
            assert!(
                morsels_on < morsels_off / 2,
                "{label} pushdown={pushdown}: expected cross-Aggregate dynamic pushdown \
                 to prune lineitem row groups (on={morsels_on}, off={morsels_off})"
            );
            assert!(
                metric(plan_on.as_ref(), "join_dynamic_filters_pushed") > 0,
                "{label} pushdown={pushdown}: expected join_dynamic_filters_pushed > 0"
            );
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cross_aggregate_dynamic_join_filter_empty_build_short_circuits_stage()
-> Result<()> {
    let li_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int64, false),
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    let mut li_batches = Vec::with_capacity(8);
    for rg in 0..8i64 {
        let base = rg * 1_000;
        li_batches.push(RecordBatch::try_new(
            Arc::clone(&li_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..4_000i64).map(|i| base + (i % 1_000)),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..4_000i64).map(|i| (i % 10) + 1),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..4_000i64).map(|i| (i % 50) * 10),
                )),
            ],
        )?);
    }
    let li_dir = write_parquet_files(&li_batches, 2, 4_000)?;

    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_brand", DataType::Utf8, false),
    ]));
    let part_batch = RecordBatch::try_new(
        part_schema,
        vec![
            Arc::new(Int64Array::from_iter_values(0..500i64)),
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                "Brand#11", 500,
            ))),
        ],
    )?;
    let part_dir = write_parquet_files(&[part_batch], 1, 500)?;

    let make_ctx = |mode: &str| {
        let li_path = li_dir.path().to_str().unwrap().to_owned();
        let part_path = part_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("part", &part_path, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    // Build side `part WHERE p_brand = 'NO_MATCH'` is empty -> must short-circuit
    // the grouped Aggregate(lineitem) stage completely without scanning lineitem!
    let inner_sql = "SELECT p.p_partkey, a.total_qty \
                     FROM part p \
                     INNER JOIN ( \
                         SELECT l_partkey, SUM(l_quantity) AS total_qty \
                         FROM lineitem GROUP BY l_partkey \
                     ) a ON p.p_partkey = a.l_partkey \
                     WHERE p.p_brand = 'NO_MATCH'";

    let q17_empty_sql = "SELECT SUM(l.l_extendedprice) AS total_price \
                         FROM lineitem l, part p \
                         WHERE p.p_partkey = l.l_partkey \
                           AND p.p_brand = 'NO_MATCH' \
                           AND l.l_quantity < ( \
                               SELECT 0.5 * AVG(l2.l_quantity) \
                               FROM lineitem l2 \
                               WHERE l2.l_partkey = p.p_partkey \
                           )";

    for (label, sql, expected_agg_finalizations) in [
        ("inner_empty", inner_sql, 0usize),
        ("q17_empty", q17_empty_sql, 1usize), // outer ungrouped SUM still emits 1 NULL row; inner stage_0 skipped!
    ] {
        let stock = make_ctx("off").await?;
        let native = make_ctx("required").await?;
        let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
        assert!(
            metric(plan.as_ref(), "join_empty_build_short_circuits") > 0,
            "{label}: expected join_empty_build_short_circuits > 0"
        );
        // `lineitem` has 8 row groups; if stage_0 was skipped, total claimed morsels
        // can be at most `part`'s 1 RG (0 if `part` itself was pruned by row-group stats).
        assert!(
            metric(plan.as_ref(), "parquet_morsels_claimed") <= 1,
            "{label}: expected empty build to skip scanning lineitem stage (claimed={})",
            metric(plan.as_ref(), "parquet_morsels_claimed")
        );
        assert_eq!(
            metric(plan.as_ref(), "aggregate_finalizations"),
            expected_agg_finalizations,
            "{label}: expected grouped stage_0 aggregate to be skipped"
        );
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cross_aggregate_dynamic_join_filter_nulls_join_types_having_and_guards()
-> Result<()> {
    // Fact table with 8 row groups, unique `id`, NULL keys, and duplicates.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("k1", DataType::Int64, true),
        Field::new("k2", DataType::Int64, true),
        Field::new("val", DataType::Int64, false),
    ]));
    let mut fact_batches = Vec::with_capacity(8);
    for rg in 0..8i64 {
        let mut ids = Vec::with_capacity(500);
        let mut k1s = Vec::with_capacity(500);
        let mut k2s = Vec::with_capacity(500);
        let mut vals = Vec::with_capacity(500);
        for i in 0..500i64 {
            let k1 = if i % 50 == 0 {
                None
            } else {
                Some(rg * 100 + (i % 100))
            };
            let k2 = if i % 75 == 0 {
                None
            } else {
                Some((rg * 100 + (i % 100)) % 30)
            };
            ids.push(rg * 500 + i);
            k1s.push(k1);
            k2s.push(k2);
            vals.push((i % 25) + 1);
        }
        fact_batches.push(RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(k1s)),
                Arc::new(Int64Array::from(k2s)),
                Arc::new(Int64Array::from(vals)),
            ],
        )?);
    }
    let fact_dir = write_parquet_files(&fact_batches, 2, 500)?;

    // Dim1: small table with keys in RG 3 (300..315) PLUS a NULL key AND a non-matching key (9999).
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk1", DataType::Int64, true),
        Field::new("dk2", DataType::Int64, true),
        Field::new("tag", DataType::Int64, false),
    ]));
    let mut dk1 = vec![None, Some(9999)];
    let mut dk2 = vec![None, Some(1)];
    let mut tag = vec![0, 99];
    for k in 300..315i64 {
        dk1.push(Some(k));
        dk2.push(Some(k % 30));
        tag.push(k);
    }
    let dim1_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from(dk1)),
            Arc::new(Int64Array::from(dk2)),
            Arc::new(Int64Array::from(tag)),
        ],
    )?;
    let dim1_dir = write_parquet_files(&[dim1_batch], 1, 100)?;

    // Dim2: disjoint keys in RG 6 (600..615) for multi-consumer CTE guard testing.
    let dim2_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(600..615i64)),
            Arc::new(Int64Array::from_iter_values((600..615i64).map(|k| k % 30))),
            Arc::new(Int64Array::from_iter_values(600..615i64)),
        ],
    )?;
    let dim2_dir = write_parquet_files(&[dim2_batch], 1, 100)?;

    let make_ctx = |mode: &str, pushdown: bool, pg_dialect: bool| {
        let fact_path = fact_dir.path().to_str().unwrap().to_owned();
        let dim1_path = dim1_dir.path().to_str().unwrap().to_owned();
        let dim2_path = dim2_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            let ctx = SessionContext::new_with_config(cfg);
            if pg_dialect {
                ctx.sql("SET datafusion.sql_parser.dialect = 'PostgreSQL'")
                    .await?
                    .collect()
                    .await?;
            }
            ctx.register_parquet("fact", &fact_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("dim1", &dim1_path, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("dim2", &dim2_path, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    // (label, sql, expect_pushed, cte_mat)
    let cases = [
        // Safe: NULL keys with `=` (NULL group discarded cleanly).
        (
            "inner_nulls_eq",
            "SELECT d.dk1, a.s FROM dim1 d \
             INNER JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             ON d.dk1 = a.k1 ORDER BY d.dk1",
            true,
            false,
        ),
        // Safe: NULL keys with `IS NOT DISTINCT FROM` (NULL group preserved!).
        (
            "inner_nulls_indf",
            "SELECT d.dk1, a.s FROM dim1 d \
             INNER JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             ON d.dk1 IS NOT DISTINCT FROM a.k1 ORDER BY d.dk1",
            true,
            false,
        ),
        // Safe: multi-key GROUP BY with NULLs and IS NOT DISTINCT FROM.
        (
            "multi_key_nulls_indf",
            "SELECT d.dk1, d.dk2, a.s FROM dim1 d \
             INNER JOIN (SELECT k1, k2, SUM(val) AS s FROM fact GROUP BY k1, k2) a \
             ON d.dk1 IS NOT DISTINCT FROM a.k1 AND d.dk2 IS NOT DISTINCT FROM a.k2 \
             ORDER BY d.dk1, d.dk2",
            true,
            false,
        ),
        // Safe: BuildOuter (dim1 LEFT JOIN agg -> dim1 is build, agg is probe).
        (
            "build_outer",
            "SELECT d.dk1, d.tag, a.s FROM dim1 d \
             LEFT JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             ON d.dk1 = a.k1 ORDER BY d.tag",
            true,
            false,
        ),
        // Safe: BuildAnti (dim1 LEFT ANTI JOIN agg -> dim1 is build, agg is probe).
        (
            "build_anti",
            "SELECT d.dk1, d.tag FROM dim1 d \
             LEFT ANTI JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             ON d.dk1 = a.k1 ORDER BY d.tag",
            true,
            false,
        ),
        // Unsafe (must preserve all agg rows): ProbeOuter (agg LEFT JOIN dim1).
        (
            "probe_outer_no_push",
            "SELECT a.k1, a.s, d.tag FROM (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             LEFT JOIN dim1 d ON a.k1 = d.dk1 ORDER BY a.k1",
            false,
            false,
        ),
        // Unsafe (must preserve all agg rows): FullOuter.
        (
            "full_outer_no_push",
            "SELECT a.k1, a.s, d.tag FROM (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             FULL OUTER JOIN dim1 d ON a.k1 = d.dk1 ORDER BY a.k1, d.tag",
            false,
            false,
        ),
        // Unsafe (must preserve unmatched agg rows): ProbeAnti (agg LEFT ANTI JOIN dim1).
        (
            "probe_anti_no_push",
            "SELECT a.k1, a.s FROM (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             LEFT ANTI JOIN dim1 d ON a.k1 = d.dk1 ORDER BY a.k1",
            false,
            false,
        ),
        // Safe: HAVING on aggregate value with unmodified GROUP BY key column.
        (
            "having_unmodified_key",
            "SELECT d.dk1, a.s FROM dim1 d \
             INNER JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1 HAVING SUM(val) > 50) a \
             ON d.dk1 = a.k1 ORDER BY d.dk1",
            true,
            false,
        ),
        // Guard: HAVING + altered key expression (CASE depending on SUM(val)) MUST NOT push.
        (
            "having_altered_key_no_push",
            "SELECT d.dk1, a.s FROM dim1 d \
             INNER JOIN ( \
                 SELECT CASE WHEN SUM(val) > 50 THEN k1 ELSE 305 END AS ak, SUM(val) AS s \
                 FROM fact GROUP BY k1 HAVING SUM(val) > 10 \
             ) a ON d.dk1 = a.ak ORDER BY d.dk1, a.s",
            false,
            false,
        ),
        // Guard: ungrouped aggregate MUST NOT push (filtering input rows would change SUM(val)!).
        // Use a tiny VALUES build side so the ungrouped Aggregate stage sits on the probe side,
        // directly exercising the ungrouped-aggregate non-pushdown guard.
        (
            "ungrouped_agg_no_push",
            "SELECT d.dk1, a.total FROM (VALUES (305)) AS d(dk1) \
             INNER JOIN (SELECT SUM(val) - 51000 + 305 AS total FROM fact) a \
             ON d.dk1 = a.total ORDER BY d.dk1",
            false,
            false,
        ),
        // Guard: TopN / LIMIT over grouped aggregate MUST NOT push (would change top-K membership!).
        // Use a tiny VALUES build side so the TopN stage sits on the probe side.
        (
            "topn_agg_no_push",
            "SELECT d.dk1, a.s FROM (VALUES (305), (306)) AS d(dk1) \
             INNER JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1 ORDER BY s DESC, k1 ASC LIMIT 5) a \
             ON d.dk1 = a.k1 ORDER BY d.dk1",
            false,
            false,
        ),
        // Guard: Sort with OFFSET MUST NOT push (filtering before Sort changes which rows OFFSET skips!).
        // `dim1` keys live in RG 3 (k1 in 300..315, id in 1500..2000), so OFFSET 250 skips RG 0 rows
        // and keeps all matching rows; a pushed filter would prune RG 0..2 and skip the matching rows.
        (
            "sort_offset_no_push",
            "SELECT d.dk1, a.id FROM dim1 d \
             INNER JOIN (SELECT id, k1 FROM fact ORDER BY id OFFSET 250) a \
             ON d.dk1 = a.k1 ORDER BY d.dk1, a.id",
            false,
            false,
        ),
        // Guard: Grouped Aggregate + ORDER BY + OFFSET (PostSort::Sort with skip > 0) MUST NOT push.
        // Without pushdown, OFFSET 250 skips k1 in 0..249 and keeps all 15 groups in 300..314;
        // pushing dim1's filter before Aggregate would leave <= 15 groups and OFFSET 250 would return 0 rows!
        (
            "agg_sort_offset_no_push",
            "SELECT d.dk1, a.s FROM dim1 d \
             INNER JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1 ORDER BY k1 OFFSET 250) a \
             ON d.dk1 = a.k1 ORDER BY d.dk1",
            false,
            false,
        ),
        // Guard: GroupCount + ORDER BY + OFFSET (PostSort::Sort with skip > 0) MUST NOT push.
        (
            "group_count_sort_offset_no_push",
            "SELECT d.dk1, a.cnt FROM dim1 d \
             INNER JOIN (SELECT k1, COUNT(*) AS cnt FROM fact GROUP BY k1 ORDER BY k1 OFFSET 250) a \
             ON d.dk1 = a.k1 ORDER BY d.dk1",
            false,
            false,
        ),
        // Guard: multi-consumer materialized CTE with disjoint build sets MUST NOT push
        // dim1's filter into the shared CTE (which would starve dim2!).
        (
            "multi_consumer_cte_no_push",
            "WITH agg AS MATERIALIZED (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) \
             SELECT t1.k1 AS k1_a, t1.s AS s_a, t2.k1 AS k1_b, t2.s AS s_b \
             FROM (SELECT a.k1, a.s FROM agg a INNER JOIN dim1 d ON a.k1 = d.dk1) t1 \
             INNER JOIN (SELECT a.k1, a.s FROM agg a INNER JOIN dim2 d ON a.k1 = d.dk1) t2 \
             ON (t1.k1 % 10) = (t2.k1 % 10) \
             ORDER BY k1_a, k1_b",
            false,
            true,
        ),
    ];

    for (label, sql, expect_pushed, cte_mat) in cases {
        for pushdown in [false, true] {
            let stock = make_ctx("off", pushdown, cte_mat).await?;
            let native = make_ctx("required", pushdown, cte_mat).await?;
            let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
            let pushed = metric(plan.as_ref(), "join_dynamic_filters_pushed");
            if expect_pushed {
                assert!(
                    pushed > 0,
                    "{label} pushdown={pushdown}: expected join_dynamic_filters_pushed > 0, got 0"
                );
            } else {
                assert_eq!(
                    pushed, 0,
                    "{label} pushdown={pushdown}: expected join_dynamic_filters_pushed == 0, got {pushed}"
                );
            }
        }
    }

    // Concurrent & repeated execution of the same compiled cross-Aggregate plan
    // must be immutable and leak zero MemoryPool bytes.
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut cfg = SessionConfig::new().with_target_partitions(4);
    cfg.options_mut().execution.parquet.pushdown_filters = true;
    cfg.options_mut().execution.parquet.pruning = true;
    cfg.options_mut().execution.collect_statistics = false;
    let ctx = SessionContext::new_with_config_rt(cfg, runtime);
    ctx.register_parquet(
        "fact",
        fact_dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    ctx.register_parquet(
        "dim1",
        dim1_dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    ctx.sql("SET datafusion.execution.morsel_mode = 'required'")
        .await?;
    let plan = ctx
        .sql(
            "SELECT d.dk1, a.s FROM dim1 d \
             INNER JOIN (SELECT k1, SUM(val) AS s FROM fact GROUP BY k1) a \
             ON d.dk1 = a.k1 ORDER BY d.dk1",
        )
        .await?
        .create_physical_plan()
        .await?;

    let first = sorted_rows(&collect(Arc::clone(&plan), ctx.task_ctx()).await?)?;
    let futs = (0..4).map(|_| {
        let p = Arc::clone(&plan);
        let tc = ctx.task_ctx();
        async move {
            let batches = collect(p, tc).await?;
            sorted_rows(&batches)
        }
    });
    for got in futures::future::try_join_all(futs).await? {
        assert_eq!(got, first, "concurrent cross-Aggregate execution mismatch");
    }
    assert_eq!(
        pool.reserved(),
        0,
        "MemoryPool must have 0 reserved bytes after concurrent cross-Aggregate runs"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sibling_forward_dynamic_filter_propagation_prunes_sibling_build_q9_pattern()
-> Result<()> {
    // Q9-shaped multi-join: selective `part` (`p_partkey`, ~5% selective) and
    // large unfiltered `partsupp` (`(ps_suppkey, ps_partkey)`) both join
    // `lineitem` on `l_partkey`. Forward sibling dynamic filter propagation
    // must schedule `part` before `partsupp` and prune `partsupp` rows in Parquet.
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_name", DataType::Utf8, false),
    ]));
    let part_batch = RecordBatch::try_new(
        Arc::clone(&part_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=4_000i64)),
            Arc::new(StringArray::from_iter_values((1..=4_000i64).map(|k| {
                if k % 20 == 0 {
                    format!("bright green item {k}")
                } else {
                    format!("plain blue item {k}")
                }
            }))),
        ],
    )?;
    let part_dir = write_parquet_files(&[part_batch], 1, 2_000)?;

    let supp_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, false),
        Field::new("s_nationkey", DataType::Int64, false),
    ]));
    let supp_batch = RecordBatch::try_new(
        Arc::clone(&supp_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=200i64)),
            Arc::new(Int64Array::from_iter_values((1..=200i64).map(|k| k % 5))),
        ],
    )?;
    let supp_dir = write_parquet_files(&[supp_batch], 1, 200)?;

    let ps_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int64, false),
        Field::new("ps_suppkey", DataType::Int64, false),
        Field::new("ps_supplycost", DataType::Int64, false),
    ]));
    let mut ps_batches = Vec::with_capacity(4);
    for chunk in 0..4i64 {
        let mut pk = Vec::with_capacity(4_000);
        let mut sk = Vec::with_capacity(4_000);
        let mut sc = Vec::with_capacity(4_000);
        for p in (chunk * 1_000 + 1)..=((chunk + 1) * 1_000) {
            for s_off in 0..4i64 {
                pk.push(p);
                sk.push(((p + s_off * 37) % 200) + 1);
                sc.push((p * 7 + s_off) % 100);
            }
        }
        ps_batches.push(RecordBatch::try_new(
            Arc::clone(&ps_schema),
            vec![
                Arc::new(Int64Array::from(pk)),
                Arc::new(Int64Array::from(sk)),
                Arc::new(Int64Array::from(sc)),
            ],
        )?);
    }
    let ps_dir = write_parquet_files(&ps_batches, 2, 2_000)?;

    let li_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int64, false),
        Field::new("l_suppkey", DataType::Int64, false),
        Field::new("l_qty", DataType::Int64, false),
    ]));
    let mut li_batches = Vec::with_capacity(4);
    for chunk in 0..4i64 {
        let mut pk = Vec::with_capacity(5_000);
        let mut sk = Vec::with_capacity(5_000);
        let mut qty = Vec::with_capacity(5_000);
        for i in 0..5_000i64 {
            let p = ((chunk * 5_000 + i) % 4_000) + 1;
            let s_off = i % 4;
            pk.push(p);
            sk.push(((p + s_off * 37) % 200) + 1);
            qty.push((i % 10) + 1);
        }
        li_batches.push(RecordBatch::try_new(
            Arc::clone(&li_schema),
            vec![
                Arc::new(Int64Array::from(pk)),
                Arc::new(Int64Array::from(sk)),
                Arc::new(Int64Array::from(qty)),
            ],
        )?);
    }
    let li_dir = write_parquet_files(&li_batches, 2, 2_500)?;

    let make_ctx = |mode: &str, pushdown: bool, dyn_join: bool| {
        let part_p = part_dir.path().to_str().unwrap().to_owned();
        let supp_p = supp_dir.path().to_str().unwrap().to_owned();
        let ps_p = ps_dir.path().to_str().unwrap().to_owned();
        let li_p = li_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            cfg.options_mut()
                .optimizer
                .enable_join_dynamic_filter_pushdown = dyn_join;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("part", &part_p, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("supplier", &supp_p, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("partsupp", &ps_p, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("lineitem", &li_p, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let sql = "SELECT s.s_nationkey, SUM(l.l_qty * ps.ps_supplycost) AS total \
               FROM part p, supplier s, lineitem l, partsupp ps \
               WHERE s.s_suppkey = l.l_suppkey \
                 AND ps.ps_suppkey = l.l_suppkey \
                 AND ps.ps_partkey = l.l_partkey \
                 AND p.p_partkey = l.l_partkey \
                 AND p.p_name LIKE '%green%' \
               GROUP BY s.s_nationkey";

    let stock = make_ctx("off", true, true).await?;
    let native_on = make_ctx("required", true, true).await?;
    let native_off = make_ctx("required", true, false).await?;

    let (plan_on, _) = assert_native_parity(&stock, &native_on, sql).await?;
    let (plan_off, _) = assert_native_parity(&stock, &native_off, sql).await?;

    let pruned_on = metric(plan_on.as_ref(), "pushdown_rows_pruned");
    let pruned_off = metric(plan_off.as_ref(), "pushdown_rows_pruned");
    // Without sibling propagation, only `lineitem` (20K rows, ~19K pruned) + `part`
    // (4K, ~3.8K pruned) = ~22.8K pruned. With `part -> partsupp` sibling propagation,
    // `partsupp` (16K rows, ~15.2K pruned) pushes total pruned well above 30,000!
    assert!(
        pruned_on > pruned_off + 25_000,
        "expected sibling dynamic filter propagation to prune both lineitem and partsupp \
         (on={pruned_on}, off={pruned_off})"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn int_pair_and_dense_array_map_join_parity_nulls_duplicates_and_extremes()
-> Result<()> {
    // 1. 2-key integer FlatIntTable<(i32, i32)> and <(i64, i64)> with duplicates,
    //    NULLs, and all join types (Inner, Left, Right, Full, Semi, Anti, residual).
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int64, true),
        Field::new("k2", DataType::Int64, true),
        Field::new("i1", DataType::Int32, true),
        Field::new("i2", DataType::Int32, true),
        Field::new("bval", DataType::Int64, false),
    ]));
    let p_schema = Arc::new(Schema::new(vec![
        Field::new("pk1", DataType::Int64, true),
        Field::new("pk2", DataType::Int64, true),
        Field::new("pi1", DataType::Int32, true),
        Field::new("pi2", DataType::Int32, true),
        Field::new("pval", DataType::Int64, false),
    ]));
    let b_batch = RecordBatch::try_new(
        Arc::clone(&b_schema),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(1),
                Some(2),
                None,
                Some(3),
                Some(i64::MIN),
                Some(i64::MAX),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(10),
                Some(10),
                Some(20),
                Some(30),
                None,
                Some(i64::MAX),
                Some(i64::MIN),
            ])),
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(2),
                None,
                Some(3),
                Some(i32::MIN),
                Some(i32::MAX),
            ])),
            Arc::new(Int32Array::from(vec![
                Some(10),
                Some(10),
                Some(20),
                Some(30),
                None,
                Some(i32::MAX),
                Some(i32::MIN),
            ])),
            Arc::new(Int64Array::from(vec![100, 101, 200, 300, 400, 500, 600])),
        ],
    )?;
    let p_batch = RecordBatch::try_new(
        Arc::clone(&p_schema),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(2),
                Some(2),
                Some(99),
                None,
                Some(i64::MIN),
                Some(i64::MAX),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(10),
                Some(20),
                Some(21),
                Some(99),
                Some(30),
                Some(i64::MAX),
                Some(i64::MIN),
            ])),
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                Some(2),
                Some(99),
                None,
                Some(i32::MIN),
                Some(i32::MAX),
            ])),
            Arc::new(Int32Array::from(vec![
                Some(10),
                Some(20),
                Some(21),
                Some(99),
                Some(30),
                Some(i32::MAX),
                Some(i32::MIN),
            ])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7])),
        ],
    )?;

    // 2. Dense integer ArrayMap tables (unique + duplicate + negative keys)
    //    exercising build_key_probe_subst and contiguous zero-copy probe output.
    let dense_b_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, true),
        Field::new("dval", DataType::Int64, false),
    ]));
    let dense_p_schema = Arc::new(Schema::new(vec![
        Field::new("qk", DataType::Int64, true),
        Field::new("qval", DataType::Int64, false),
    ]));
    let dense_uniq_b = RecordBatch::try_new(
        Arc::clone(&dense_b_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(-500..1_500i64)),
            Arc::new(Int64Array::from_iter_values(0..2_000i64)),
        ],
    )?;
    let dense_dup_b = RecordBatch::try_new(
        Arc::clone(&dense_b_schema),
        vec![
            Arc::new(Int64Array::from(
                (0..2_000i64)
                    .map(|i| if i == 10 { None } else { Some((i % 400) - 200) })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from_iter_values(0..2_000i64)),
        ],
    )?;
    let dense_p = RecordBatch::try_new(
        Arc::clone(&dense_p_schema),
        vec![
            Arc::new(Int64Array::from(
                (0..4_000i64)
                    .map(|i| match i {
                        0 => Some(i64::MIN),
                        1 => Some(i64::MAX),
                        2 => None,
                        _ => Some((i % 2_200) - 600),
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from_iter_values(0..4_000i64)),
        ],
    )?;

    let stock = stock_context(4, 256);
    let native = morsel_context(4, 256, MorselMode::Required, false);
    for ctx in [&stock, &native] {
        ctx.register_table(
            "b",
            Arc::new(MemTable::try_new(
                Arc::clone(&b_schema),
                vec![vec![b_batch.clone()]],
            )?),
        )?;
        ctx.register_table(
            "p",
            Arc::new(MemTable::try_new(
                Arc::clone(&p_schema),
                vec![vec![p_batch.clone()]],
            )?),
        )?;
        ctx.register_table(
            "du",
            Arc::new(MemTable::try_new(
                Arc::clone(&dense_b_schema),
                vec![vec![dense_uniq_b.clone()]],
            )?),
        )?;
        ctx.register_table(
            "dd",
            Arc::new(MemTable::try_new(
                Arc::clone(&dense_b_schema),
                vec![vec![dense_dup_b.clone()]],
            )?),
        )?;
        ctx.register_table(
            "dp",
            Arc::new(MemTable::try_new(
                Arc::clone(&dense_p_schema),
                vec![vec![dense_p.clone()]],
            )?),
        )?;
    }

    let queries = [
        // Int64Pair
        "SELECT b.k1, b.k2, b.bval, p.pval FROM b INNER JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT b.k1, b.k2, b.bval, p.pval FROM b LEFT JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT b.k1, b.k2, b.bval, p.pval FROM b RIGHT JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT b.k1, b.k2, b.bval, p.pval FROM b FULL OUTER JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT b.k1, b.k2, b.bval FROM b WHERE EXISTS (SELECT 1 FROM p WHERE p.pk1 = b.k1 AND p.pk2 = b.k2)",
        "SELECT b.k1, b.k2, b.bval FROM b WHERE NOT EXISTS (SELECT 1 FROM p WHERE p.pk1 = b.k1 AND p.pk2 = b.k2)",
        "SELECT b.k1, p.pval FROM b INNER JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2 AND b.bval + p.pval > 105",
        // Int32Pair
        "SELECT b.i1, b.i2, b.bval, p.pval FROM b INNER JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2",
        "SELECT b.i1, b.i2, b.bval, p.pval FROM b LEFT JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2",
        "SELECT COUNT(*) FROM b INNER JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2",
        // Dense unique ArrayMap (q13-style LEFT OUTER + build_key_probe_subst + contiguous probe)
        "SELECT du.dk, COUNT(dp.qval) FROM du LEFT JOIN dp ON du.dk = dp.qk GROUP BY du.dk",
        "SELECT du.dk, dp.qval FROM du INNER JOIN dp ON du.dk = dp.qk",
        "SELECT du.dk, dp.qval FROM du RIGHT JOIN dp ON du.dk = dp.qk",
        "SELECT du.dk, dp.qval FROM du FULL OUTER JOIN dp ON du.dk = dp.qk",
        "SELECT du.dk FROM du WHERE EXISTS (SELECT 1 FROM dp WHERE dp.qk = du.dk)",
        "SELECT du.dk FROM du WHERE NOT EXISTS (SELECT 1 FROM dp WHERE dp.qk = du.dk)",
        "SELECT COUNT(*) FROM du INNER JOIN dp ON du.dk = dp.qk",
        // Dense duplicate ArrayMap + residual filter
        "SELECT dd.dk, dd.dval, dp.qval FROM dd INNER JOIN dp ON dd.dk = dp.qk",
        "SELECT dd.dk, dd.dval, dp.qval FROM dd LEFT JOIN dp ON dd.dk = dp.qk AND dd.dval > dp.qval",
        "SELECT dd.dk, dd.dval, dp.qval FROM dd FULL OUTER JOIN dp ON dd.dk = dp.qk",
        "SELECT dd.dk FROM dd WHERE EXISTS (SELECT 1 FROM dp WHERE dp.qk = dd.dk)",
    ];

    for sql in queries {
        assert_native_parity(&stock, &native, sql).await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_sided_range_and_projected_width_selects_filtered_narrow_build_side()
-> Result<()> {
    // Q10-shaped join: narrow date-range-filtered `orders` (2 ints + date) vs
    // wide unfiltered `customer` (4 wide string payload columns). Build-side
    // estimation must choose filtered `orders` as build side and push its
    // dynamic filter into `customer`.
    let cust_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int64, false),
        Field::new("c_name", DataType::Utf8, false),
        Field::new("c_address", DataType::Utf8, false),
        Field::new("c_phone", DataType::Utf8, false),
        Field::new("c_comment", DataType::Utf8, false),
    ]));
    let cust_batch = RecordBatch::try_new(
        Arc::clone(&cust_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=10_000i64)),
            Arc::new(StringArray::from_iter_values(
                (1..=10_000i64).map(|k| format!("Customer#{k:09}")),
            )),
            Arc::new(StringArray::from_iter_values(
                (1..=10_000i64).map(|k| format!("Address-padding-long-string-{k:09}")),
            )),
            Arc::new(StringArray::from_iter_values(
                (1..=10_000i64).map(|k| format!("15-555-{k:04}-{k:04}")),
            )),
            Arc::new(StringArray::from_iter_values((1..=10_000i64).map(|k| {
                format!("wide-customer-comment-payload-text-for-row-{k:09}")
            }))),
        ],
    )?;
    let cust_dir = write_parquet_files(&[cust_batch], 2, 2_500)?;

    let ord_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int64, false),
        Field::new("o_custkey", DataType::Int64, false),
        Field::new("o_orderdate", DataType::Date32, false),
        Field::new("o_pad", DataType::Utf8, false),
    ]));
    let mut ord_batches = Vec::with_capacity(4);
    for chunk in 0..4i64 {
        let base = chunk * 10_000;
        ord_batches.push(RecordBatch::try_new(
            Arc::clone(&ord_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(base + 1..=base + 10_000)),
                Arc::new(Int64Array::from_iter_values(
                    (base + 1..=base + 10_000).map(|i| (i % 10_000) + 1),
                )),
                Arc::new(Date32Array::from_iter_values(
                    (base + 1..=base + 10_000).map(|i| 8_000 + (i % 2_000) as i32),
                )),
                Arc::new(StringArray::from_iter_values(
                    (base + 1..=base + 10_000)
                        .map(|i| format!("unprojected-order-comment-padding-{i:09}")),
                )),
            ],
        )?);
    }
    let ord_dir = write_parquet_files(&ord_batches, 4, 10_000)?;

    let make_ctx = |mode: &str| {
        let cust_p = cust_dir.path().to_str().unwrap().to_owned();
        let ord_p = ord_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("customer", &cust_p, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("orders", &ord_p, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let sql = "SELECT c.c_custkey, c.c_name, c.c_address, c.c_phone, c.c_comment, o.o_orderkey \
               FROM customer c INNER JOIN orders o ON c.c_custkey = o.o_custkey \
               WHERE o.o_orderdate >= DATE '1993-10-01' AND o.o_orderdate < DATE '1993-11-01'";

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;
    let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        plan_str.contains("[Build: NativeParquet[orders"),
        "expected narrow date-range-filtered `orders` on build side, got:\n{plan_str}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn small_wide_utf8_dim_stays_build_side_against_large_narrow_int_fact() -> Result<()>
{
    // Repro: a 20K-row (~2.6 MB) dim projecting 6 wide Utf8 columns joined to
    // a 400K-row (~5.8 MB) narrow fact projecting 2 Int64 columns must keep
    // the smaller `dim` on the build side instead of double-counting width.
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("s1", DataType::Utf8, false),
        Field::new("s2", DataType::Utf8, false),
        Field::new("s3", DataType::Utf8, false),
        Field::new("s4", DataType::Utf8, false),
        Field::new("s5", DataType::Utf8, false),
        Field::new("s6", DataType::Utf8, false),
    ]));
    let n_dim = 20_000i64;
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..n_dim)),
            Arc::new(StringArray::from_iter_values(
                (0..n_dim).map(|i| format!("dim-s1-wide-value-{i:08}-abcdefghij")),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..n_dim).map(|i| format!("dim-s2-wide-value-{i:08}-klmnopqrst")),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..n_dim).map(|i| format!("dim-s3-wide-value-{i:08}-uvwxyz0123")),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..n_dim).map(|i| format!("dim-s4-wide-value-{i:08}-456789abcd")),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..n_dim).map(|i| format!("dim-s5-wide-value-{i:08}-efghijklmn")),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..n_dim).map(|i| format!("dim-s6-wide-value-{i:08}-opqrstuvwx")),
            )),
        ],
    )?;
    let dim_dir = write_parquet_files(&[dim_batch], 1, 10_000)?;

    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fk", DataType::Int64, false),
        Field::new("fval", DataType::Int64, false),
    ]));
    let mut fact_batches = Vec::with_capacity(4);
    for chunk in 0..4i64 {
        let base = chunk * 100_000;
        fact_batches.push(RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (base..base + 100_000).map(|i| i % 40_000),
                )),
                Arc::new(Int64Array::from_iter_values(base..base + 100_000)),
            ],
        )?);
    }
    let fact_dir = write_parquet_files(&fact_batches, 2, 100_000)?;

    let make_ctx = |mode: &str| {
        let dim_p = dim_dir.path().to_str().unwrap().to_owned();
        let fact_p = fact_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("dim", &dim_p, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("fact", &fact_p, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let sql = "SELECT COUNT(*), MIN(d.s1), MIN(d.s2), MIN(d.s3), MIN(d.s4), MIN(d.s5), MIN(d.s6), SUM(f.fval) \
               FROM dim d INNER JOIN fact f ON d.dk = f.fk";

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;
    let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        plan_str.contains("[Build: NativeParquet[dim"),
        "expected smaller wide `dim` to stay on build side, got:\n{plan_str}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn radix_int_pair_join_parity_all_types_nulls_and_residual() -> Result<()> {
    // 20,000 build rows (> RADIX_JOIN_THRESHOLD_ROWS = 16,384) forces 64-way
    // radix builds for both `Int32Pair` and `Int64Pair` across all join types,
    // NULLs, duplicates, and residual filters.
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int64, true),
        Field::new("k2", DataType::Int64, true),
        Field::new("i1", DataType::Int32, true),
        Field::new("i2", DataType::Int32, true),
        Field::new("bval", DataType::Int64, false),
    ]));
    let p_schema = Arc::new(Schema::new(vec![
        Field::new("pk1", DataType::Int64, true),
        Field::new("pk2", DataType::Int64, true),
        Field::new("pi1", DataType::Int32, true),
        Field::new("pi2", DataType::Int32, true),
        Field::new("pval", DataType::Int64, false),
    ]));

    let n_build = 20_000i64;
    let b_batch = RecordBatch::try_new(
        Arc::clone(&b_schema),
        vec![
            Arc::new(Int64Array::from(
                (0..n_build)
                    .map(|i| if i % 97 == 0 { None } else { Some(i % 15_000) })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..n_build)
                    .map(|i| {
                        if i % 101 == 0 {
                            None
                        } else {
                            Some((i % 15_000) * 3)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..n_build)
                    .map(|i| {
                        if i % 97 == 0 {
                            None
                        } else {
                            Some((i % 15_000) as i32)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..n_build)
                    .map(|i| {
                        if i % 101 == 0 {
                            None
                        } else {
                            Some(((i % 15_000) * 3) as i32)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from_iter_values(0..n_build)),
        ],
    )?;

    let n_probe = 24_000i64;
    let p_batch = RecordBatch::try_new(
        Arc::clone(&p_schema),
        vec![
            Arc::new(Int64Array::from(
                (0..n_probe)
                    .map(|i| {
                        if i % 89 == 0 {
                            None
                        } else {
                            Some((i * 2) % 20_000)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..n_probe)
                    .map(|i| {
                        if i % 103 == 0 {
                            None
                        } else {
                            Some(((i * 2) % 20_000) * 3)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..n_probe)
                    .map(|i| {
                        if i % 89 == 0 {
                            None
                        } else {
                            Some(((i * 2) % 20_000) as i32)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..n_probe)
                    .map(|i| {
                        if i % 103 == 0 {
                            None
                        } else {
                            Some((((i * 2) % 20_000) * 3) as i32)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from_iter_values(0..n_probe)),
        ],
    )?;

    let stock = stock_context(4, 512);
    let native = morsel_context(4, 512, MorselMode::Required, false);
    for ctx in [&stock, &native] {
        ctx.register_table(
            "b",
            Arc::new(MemTable::try_new(
                Arc::clone(&b_schema),
                vec![vec![b_batch.clone()]],
            )?),
        )?;
        ctx.register_table(
            "p",
            Arc::new(MemTable::try_new(
                Arc::clone(&p_schema),
                vec![vec![p_batch.clone()]],
            )?),
        )?;
    }

    let queries = [
        // Radix Int64Pair across all join types + residual filter
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b INNER JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b LEFT JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b RIGHT JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b FULL OUTER JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2",
        "SELECT COUNT(*), SUM(b.bval) FROM b WHERE EXISTS (SELECT 1 FROM p WHERE p.pk1 = b.k1 AND p.pk2 = b.k2)",
        "SELECT COUNT(*), SUM(b.bval) FROM b WHERE NOT EXISTS (SELECT 1 FROM p WHERE p.pk1 = b.k1 AND p.pk2 = b.k2)",
        "SELECT COUNT(*), SUM(p.pval) FROM p WHERE EXISTS (SELECT 1 FROM b WHERE b.k1 = p.pk1 AND b.k2 = p.pk2)",
        "SELECT COUNT(*), SUM(p.pval) FROM p WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.k1 = p.pk1 AND b.k2 = p.pk2)",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b INNER JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2 AND b.bval < p.pval",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b LEFT JOIN p ON b.k1 = p.pk1 AND b.k2 = p.pk2 AND b.bval < p.pval",
        // Radix Int32Pair across all join types + residual filter
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b INNER JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b LEFT JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b RIGHT JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b FULL OUTER JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2",
        "SELECT COUNT(*), SUM(b.bval) FROM b WHERE EXISTS (SELECT 1 FROM p WHERE p.pi1 = b.i1 AND p.pi2 = b.i2)",
        "SELECT COUNT(*), SUM(b.bval) FROM b WHERE NOT EXISTS (SELECT 1 FROM p WHERE p.pi1 = b.i1 AND p.pi2 = b.i2)",
        "SELECT COUNT(*), SUM(b.bval), SUM(p.pval) FROM b INNER JOIN p ON b.i1 = p.pi1 AND b.i2 = p.pi2 AND b.bval < p.pval",
    ];

    for sql in queries {
        assert_native_parity(&stock, &native, sql).await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn radix_partition_slice_bloom_prunes_filtered_utf8_build() -> Result<()> {
    // Filtered Utf8 build with 20,000 surviving rows (> RADIX_JOIN_THRESHOLD_ROWS = 16,384)
    // exercises `finalize_radix_all_in_memory` with lock-free partition-slice Bloom
    // construction (`bloom_radix_low_bits = Some(...)`).
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dskey", DataType::Utf8, false),
        Field::new("keep", DataType::Int64, false),
    ]));
    let mut dim_batches = Vec::with_capacity(4);
    for chunk in 0..4i64 {
        let base = chunk * 10_000;
        dim_batches.push(RecordBatch::try_new(
            Arc::clone(&dim_schema),
            vec![
                Arc::new(StringArray::from_iter_values(
                    (base..base + 10_000).map(|i| format!("k-{:07}", i * 5)),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (base..base + 10_000).map(|i| i % 2),
                )),
            ],
        )?);
    }
    let dim_dir = write_parquet_files(&dim_batches, 2, 10_000)?;

    // 100,000 probe rows across the full string domain `k-0000000..k-0199998`:
    // min/max prunes 0 RGs; only ~20% match the filtered 20K-row dim Bloom.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fskey", DataType::Utf8, false),
        Field::new("payload", DataType::Utf8, false),
        Field::new("val", DataType::Int64, false),
    ]));
    let mut fact_batches = Vec::with_capacity(10);
    for rg in 0..10i64 {
        let base = rg * 10_000;
        fact_batches.push(RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(StringArray::from_iter_values(
                    (base..base + 10_000).map(|i| format!("k-{:07}", i * 2)),
                )),
                Arc::new(StringArray::from_iter_values(
                    (base..base + 10_000).map(|i| format!("wide-fact-payload-{i:08}")),
                )),
                Arc::new(Int64Array::from_iter_values(base..base + 10_000)),
            ],
        )?);
    }
    let fact_dir = write_parquet_files(&fact_batches, 2, 10_000)?;

    let make_ctx = |mode: &str| {
        let dim_p = dim_dir.path().to_str().unwrap().to_owned();
        let fact_p = fact_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = false;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("dim", &dim_p, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("fact", &fact_p, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let sql = "SELECT COUNT(*), SUM(f.val), MIN(f.payload) \
               FROM fact f INNER JOIN dim d ON f.fskey = d.dskey \
               WHERE d.keep = 1";

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;
    let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
    let pruned = metric(plan.as_ref(), "pushdown_rows_pruned");
    assert!(
        pruned > 40_000,
        "expected radix partition-slice Bloom to prune >40,000 probe rows, got {pruned}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chained_double_short_circuit_skips_earlier_build_emission() -> Result<()> {
    // Repro: `p JOIN emptyA RIGHT JOIN b JOIN emptyA2 RIGHT JOIN c` has two
    // runtime-empty inner builds (`a1 WHERE v < 0`, `a2 WHERE v < 0`);
    // `short_circuit_op` must keep the later (`max`) op_idx so `b`'s build
    // emission is skipped and only `c`'s build emission runs.
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let non_empty = |lo: i64, hi: i64| -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(lo..hi)),
                Arc::new(Int64Array::from_iter_values(lo..hi)),
            ],
        )?)
    };

    let stock = stock_context(4, 16);
    let native = morsel_context(4, 16, MorselMode::Required, false);

    for ctx in [&stock, &native] {
        ctx.register_table(
            "p",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![non_empty(0, 100)?]],
            )?),
        )?;
        ctx.register_table(
            "a1",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![non_empty(0, 5)?]],
            )?),
        )?;
        ctx.register_table(
            "b",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![non_empty(0, 20)?]],
            )?),
        )?;
        ctx.register_table(
            "a2",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![non_empty(0, 5)?]],
            )?),
        )?;
        ctx.register_table(
            "c",
            Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                vec![vec![non_empty(0, 10)?]],
            )?),
        )?;
    }

    let sql = "SELECT p.v AS pv, b.v AS bv, c.v AS cv \
               FROM p \
               INNER JOIN (SELECT k, v FROM a1 WHERE v < 0) empty_a ON p.k = empty_a.k \
               RIGHT JOIN b ON p.k = b.k \
               INNER JOIN (SELECT k, v FROM a2 WHERE v < 0) empty_a2 ON b.k = empty_a2.k \
               RIGHT JOIN c ON b.k = c.k";

    let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
    assert_eq!(
        metric(plan.as_ref(), "join_empty_build_short_circuits"),
        2,
        "expected both empty inner builds to record short-circuits"
    );
    assert_eq!(
        metric(plan.as_ref(), "join_build_emission_pipelines"),
        1,
        "expected only 1 build_emission pipeline to run after double short-circuit"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stats_interval_selectivity_flips_q14_build_side_to_filtered_fact() -> Result<()>
{
    // Q14-like schema sized so `pipeline_filter_shift` (>> 2 = 25% selectivity)
    // still prefers building on `dim` (`fact_bytes ≈ 15x dim_bytes`, so
    // `fact * 3/4 > dim * 4`), whereas Parquet footer column-stats interval
    // selectivity (0.2% window) flips the build side to filtered `fact`.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("partkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Int32, false),
        Field::new("extprice", DataType::Int64, false),
        Field::new("discount", DataType::Int64, false),
    ]));
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..80_000i64)
                    .map(|i| (i.wrapping_mul(2_654_435_761) & 0x7fff_ffff) % 6_000),
            )),
            Arc::new(Int32Array::from_iter_values((0..80_000i32).map(|i| {
                ((i as u32).wrapping_mul(2_246_822_519) % 10_000) as i32
            }))),
            Arc::new(Int64Array::from_iter_values(
                (0..80_000i64)
                    .map(|i| (i.wrapping_mul(1_103_515_245) + 12_345) & 0x00ff_ffff),
            )),
            Arc::new(Int64Array::from_iter_values(
                (0..80_000i64).map(|i| ((i.wrapping_mul(16_807)) % 100) & 0x7f),
            )),
        ],
    )?;
    let fact_batches = vec![
        fact_batch.slice(0, 40_000),
        fact_batch.slice(40_000, 40_000),
    ];
    let fact_dir = write_parquet_files(&fact_batches, 2, 20_000)?;

    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_type", DataType::Utf8, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..6_000i64)),
            Arc::new(StringArray::from_iter_values((0..6_000i64).map(|i| {
                if i % 5 == 0 {
                    format!("PROMO STEEL {i:06}")
                } else {
                    format!("STANDARD BRASS {i:06}")
                }
            }))),
        ],
    )?;
    let dim_batches = vec![dim_batch.slice(0, 3_000), dim_batch.slice(3_000, 3_000)];
    let dim_dir = write_parquet_files(&dim_batches, 2, 1_500)?;

    let make_ctx = |mode: &str| {
        let fp = fact_dir.path().to_str().unwrap().to_owned();
        let dp = dim_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = true;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("fact", &fp, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("dim", &dp, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;

    // 0.2% shipdate window (~160 rows) vs 6,000-row `dim`:
    // Old `pipeline_filter_shift` (>> 2 = 25%) estimates `fact >> 2` > `dim` (builds `dim`),
    // while Parquet footer interval selectivity (0.2%) flips build side to `fact`.
    let q14_like = "SELECT SUM(CASE WHEN d.p_type LIKE 'PROMO%' THEN f.extprice * (100 - f.discount) ELSE 0 END), \
                           SUM(f.extprice * (100 - f.discount)) \
                    FROM fact f INNER JOIN dim d ON f.partkey = d.p_partkey \
                    WHERE f.shipdate >= 100 AND f.shipdate < 120";
    let (plan, _) = assert_native_parity(&stock, &native, q14_like).await?;
    let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));
    assert!(
        plan_str.contains("Build: NativeParquet[fact"),
        "expected filtered `fact` on build side, got:\n{plan_str}"
    );
    assert!(
        metric(plan.as_ref(), "join_dynamic_filters_pushed") > 0,
        "expected dynamic join filter pushed from filtered `fact` onto `dim`"
    );

    // Non-interval predicate (`Utf8` equality) on `dim` must still fall back cleanly
    // to `pipeline_filter_shift` and match Stock exactly.
    let string_filter_sql = "SELECT COUNT(*), SUM(f.extprice) \
                             FROM fact f INNER JOIN dim d ON f.partkey = d.p_partkey \
                             WHERE d.p_type = 'PROMO STEEL 000010'";
    assert_native_parity(&stock, &native, string_filter_sql).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transitive_equijoin_key_substitution_across_build_semi_and_inner() -> Result<()>
{
    // Three-table Parquet fixture where `ta` (20,000 rows) is always the probe side,
    // `tb` (3,000 rows) is the first join's build side, and filtered `tc` (~200 rows)
    // is the second join's build side whose probe key references `tb.bx`
    // (`JoinOutputCol::Build` of the first join!).
    let ta_schema = Arc::new(Schema::new(vec![
        Field::new("ax", DataType::Int64, true),
        Field::new("ay", DataType::Int64, true),
        Field::new("av", DataType::Int64, false),
        Field::new("apad", DataType::Utf8, false),
    ]));
    let ta_batch = RecordBatch::try_new(
        Arc::clone(&ta_schema),
        vec![
            Arc::new(Int64Array::from_iter((0..20_000i64).map(|i| {
                if i % 97 == 0 {
                    None
                } else {
                    Some((i * 7) % 1_000)
                }
            }))),
            Arc::new(Int64Array::from_iter((0..20_000i64).map(|i| {
                if i % 89 == 0 {
                    None
                } else {
                    Some((i * 13) % 200)
                }
            }))),
            Arc::new(Int64Array::from_iter_values(
                (0..20_000i64).map(|i| 1 + (i % 20)),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..20_000i64).map(|i| format!("ta-payload-{i:08}")),
            )),
        ],
    )?;
    let ta_batches = vec![ta_batch.slice(0, 10_000), ta_batch.slice(10_000, 10_000)];
    let ta_dir = write_parquet_files(&ta_batches, 2, 5_000)?;

    let tb_schema = Arc::new(Schema::new(vec![
        Field::new("bx", DataType::Int64, true),
        Field::new("by", DataType::Int64, true),
        Field::new("bv", DataType::Int64, false),
    ]));
    let tb_batch = RecordBatch::try_new(
        Arc::clone(&tb_schema),
        vec![
            Arc::new(Int64Array::from_iter((0..3_000i64).map(|i| {
                if i % 101 == 0 {
                    None
                } else {
                    Some((i * 11) % 1_000)
                }
            }))),
            Arc::new(Int64Array::from_iter(
                (0..3_000i64).map(|i| if i % 83 == 0 { None } else { Some(i % 200) }),
            )),
            Arc::new(Int64Array::from_iter_values(
                (0..3_000i64).map(|i| (i % 40) * 5),
            )),
        ],
    )?;
    let tb_batches = vec![tb_batch.slice(0, 1_500), tb_batch.slice(1_500, 1_500)];
    let tb_dir = write_parquet_files(&tb_batches, 2, 1_000)?;

    let tc_schema = Arc::new(Schema::new(vec![
        Field::new("cx", DataType::Int64, true),
        Field::new("cg", DataType::Int32, false),
        Field::new("cname", DataType::Utf8, false),
    ]));
    let tc_batch = RecordBatch::try_new(
        Arc::clone(&tc_schema),
        vec![
            Arc::new(Int64Array::from_iter(
                (0..1_000i64).map(|i| if i == 0 { None } else { Some(i) }),
            )),
            Arc::new(Int32Array::from_iter_values((0..1_000i32).map(|i| i % 5))),
            Arc::new(StringArray::from_iter_values(
                (0..1_000i64).map(|i| format!("Supplier#{i:04}")),
            )),
        ],
    )?;
    let tc_dir = write_parquet_files(&[tc_batch], 1, 1_000)?;

    let make_ctx = |mode: &str| {
        let ap = ta_dir.path().to_str().unwrap().to_owned();
        let bp = tb_dir.path().to_str().unwrap().to_owned();
        let cp = tc_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = true;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("ta", &ap, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("tb", &bp, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("tc", &cp, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;

    // 1. Inner join build-key equijoin substitution:
    //    `ta -> HashProbe(Inner, tb) -> HashProbe(Inner, tc on tb.bx = tc.cx)`.
    //    Baseline pushes `tb -> ta` (2 bounds; unfiltered tb skips Bloom? Wait: tb <= 16K so 3)
    //    + `tc -> tb` (3: lo, hi, bloom) = 6; PR substitutes `tb.bx -> ta.ax` and ALSO
    //    pushes `tc -> ta` (+3) = 9.
    let inner_sql = "SELECT COUNT(*), SUM(ta.av), SUM(tb.bv) \
                     FROM ta \
                     INNER JOIN tb ON ta.ax = tb.bx \
                     INNER JOIN tc ON tb.bx = tc.cx \
                     WHERE tc.cg = 3";
    let (inner_plan, _) = assert_native_parity(&stock, &native, inner_sql).await?;
    let inner_plan_str = format!("{}", displayable(inner_plan.as_ref()).indent(true));
    assert!(
        inner_plan_str.contains("NativeParquet[ta"),
        "expected ta probe plan, got:\n{inner_plan_str}"
    );
    let inner_pushed = metric(inner_plan.as_ref(), "join_dynamic_filters_pushed");
    assert_eq!(
        inner_pushed, 9,
        "expected 9 pushed dynamic filters (3 tb->ta + 3 tc->tb + 3 substituted tc->ta), got {inner_pushed}\n{inner_plan_str}"
    );

    // 2. BuildSemi build-key equijoin substitution (exact Q20 topology):
    //    `stage_0(ta GROUP BY ax, ay) -> HashProbe(BuildSemi, tb) -> HashProbe(tc on tb.bx = tc.cx)`.
    //    `ta` has ~20,000 groups (> `tb` 3,000 rows), so `tb` is on the BUILD side of BuildSemi
    //    and `stage_0(ta)` is on the PROBE side. Baseline cannot push `tc.cx` across BuildSemi
    //    into `ta`; PR substitutes `tb.bx -> ta.ax` and pushes `tc.cx` (+3) all the way to `ta`.
    let build_semi_sql = "SELECT c.cname \
                          FROM tc c \
                          WHERE c.cg = 3 \
                            AND c.cx IN ( \
                              SELECT b.bx \
                              FROM tb b \
                              WHERE b.bv > ( \
                                  SELECT SUM(a.av) \
                                  FROM ta a \
                                  WHERE a.ax = b.bx AND a.ay = b.by \
                              ) \
                            ) \
                          ORDER BY c.cname";
    let (semi_plan, _) = assert_native_parity(&stock, &native, build_semi_sql).await?;
    let semi_plan_str = format!("{}", displayable(semi_plan.as_ref()).indent(true));
    assert!(
        semi_plan_str.contains("HashJoinProbe[type=LeftSemi, keys=[ax, ay], build=Left")
            && semi_plan_str.contains("Build: NativeParquet[tb"),
        "expected BuildSemi (LeftSemi, build=Left) with tb on build side, got:\n{semi_plan_str}"
    );
    let semi_pushed = metric(semi_plan.as_ref(), "join_dynamic_filters_pushed");
    assert_eq!(
        semi_pushed, 11,
        "expected 11 pushed dynamic filters (5 tb->ta + 3 tc->tb + 3 substituted tc->ta vs baseline 8), got {semi_pushed}\n{semi_plan_str}"
    );

    // 3. Negative soundness checks reaching the exact same `JoinOutputCol::Build` branch
    //    (`ta` probe, `tb` build, downstream `tc` probing on `tb.bx`) for BuildOuter,
    //    FullOuter, BuildAnti, and NullEqualsNull — none may substitute `tc` onto `ta`
    //    and all must preserve exact parity with Stock (including unmatched & NULL rows).
    let build_outer_sql = "SELECT COUNT(*), SUM(tb.bv) \
                           FROM ta \
                           RIGHT JOIN tb ON ta.ax = tb.bx \
                           INNER JOIN tc ON tb.bx = tc.cx \
                           WHERE tc.cg = 3";
    let (bo_plan, _) = assert_native_parity(&stock, &native, build_outer_sql).await?;
    let bo_str = format!("{}", displayable(bo_plan.as_ref()).indent(true));
    assert!(
        bo_str.contains("HashJoinProbe[type=Right, keys=[ax], build=Right")
            && bo_str.contains("Build: NativeParquet[tb"),
        "expected BuildOuter (Right, build=Right) with tb on build side, got:\n{bo_str}"
    );
    // `tb -> ta` pushes only tb's own non-preserving bounds (2, since unfiltered tb's
    // key domain is saturated); downstream `tc` must NOT push onto `ta` or `tb`.
    assert_eq!(
        metric(bo_plan.as_ref(), "join_dynamic_filters_pushed"),
        2,
        "BuildOuter must not push downstream `tc` filter onto `ta` or `tb` (expected only tb->ta = 2)\n{bo_str}"
    );

    let full_outer_sql = "SELECT COUNT(*), SUM(tb.bv) \
                          FROM ta \
                          FULL OUTER JOIN tb ON ta.ax = tb.bx \
                          INNER JOIN tc ON tb.bx = tc.cx \
                          WHERE tc.cg = 3";
    let (fo_plan, _) = assert_native_parity(&stock, &native, full_outer_sql).await?;
    let fo_str = format!("{}", displayable(fo_plan.as_ref()).indent(true));
    assert!(
        fo_str.contains("HashJoinProbe[type=Full, keys=[ax], build=Right")
            && fo_str.contains("Build: NativeParquet[tb"),
        "expected FullOuter with tb on build side, got:\n{fo_str}"
    );
    assert_eq!(
        metric(fo_plan.as_ref(), "join_dynamic_filters_pushed"),
        0,
        "FullOuter must not push downstream filter onto probe scan `ta`"
    );

    let build_anti_sql = "SELECT COUNT(*), SUM(tb.bv) \
                          FROM tb \
                          INNER JOIN tc ON tb.bx = tc.cx \
                          WHERE tc.cg = 3 \
                            AND NOT EXISTS (SELECT 1 FROM ta WHERE ta.ax = tb.bx)";
    let (ba_plan, _) = assert_native_parity(&stock, &native, build_anti_sql).await?;
    let ba_str = format!("{}", displayable(ba_plan.as_ref()).indent(true));
    assert!(
        ba_str.contains("HashJoinProbe[type=LeftAnti, keys=[ax], build=Left")
            && ba_str.contains("Build: NativeParquet[tb"),
        "expected BuildAnti (LeftAnti, build=Left) with tb on build side, got:\n{ba_str}"
    );
    // `tc -> tb` pushes 3 filters onto `tb`, and `tb -> ta` is BuildAnti (pushes 3);
    // crucially `tc` must NOT be substituted onto `ta` through BuildAnti (6, never 9).
    assert_eq!(
        metric(ba_plan.as_ref(), "join_dynamic_filters_pushed"),
        6,
        "BuildAnti must not substitute downstream `tc` filter onto `ta` (expected 6, never 9)\n{ba_str}"
    );

    let null_eq_sql = "SELECT COUNT(*), SUM(ta.av), SUM(tb.bv) \
                       FROM ta \
                       INNER JOIN tb ON ta.ax IS NOT DISTINCT FROM tb.bx \
                       INNER JOIN tc ON tb.bx IS NOT DISTINCT FROM tc.cx \
                       WHERE tc.cg = 0";
    let (ne_plan, _) = assert_native_parity(&stock, &native, null_eq_sql).await?;
    let ne_str = format!("{}", displayable(ne_plan.as_ref()).indent(true));
    assert!(
        ne_str.contains("Build: NativeParquet[tb"),
        "expected tb on build side for NullEqualsNull check, got:\n{ne_str}"
    );
    // Under NullEqualsNull, `allow_equijoin_sub` is false so `tc` does not substitute onto `ta`:
    // only `tb -> ta` (2: `(range OR IS NULL)` + Bloom) + `tc -> tb` (2) = 4 (never 6).
    assert_eq!(
        metric(ne_plan.as_ref(), "join_dynamic_filters_pushed"),
        4,
        "NullEqualsNull must not substitute downstream `tc` filter onto `ta` (expected 4, never 6)\n{ne_str}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn static_range_filter_preserved_alongside_dynamic_bloom_on_same_column()
-> Result<()> {
    // Both a static user range filter AND a wider dynamic join Bloom filter target
    // the exact same column `f.k`. The filtered build (`dim WHERE d.keep = 1`) has
    // 200 distinct sparse keys (`103, 109, 115, ..., 1297`, step 6 -> > 150 InList
    // cap and not domain-saturated), so a real `JoinBloomFilterExpr` is built and
    // `build_redundant_dynamic_range_skip` actively runs.
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("pad", DataType::Utf8, false),
    ]));
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..15_000i64).map(|i| i % 1_500),
            )),
            Arc::new(Int64Array::from_iter_values(0..15_000i64)),
            Arc::new(StringArray::from_iter_values(
                (0..15_000i64).map(|i| format!("payload-{i:08}")),
            )),
        ],
    )?;
    let fact_batches = vec![fact_batch.slice(0, 7_500), fact_batch.slice(7_500, 7_500)];
    let fact_dir = write_parquet_files(&fact_batches, 2, 3_000)?;

    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("keep", DataType::Int64, false),
    ]));
    // 400 distinct keys in 100..=1297 (step 3); `keep = i % 2` leaves 200 distinct
    // filtered keys (`103, 109, ..., 1297`, min = 103, max = 1297, > 150 InList cap -> Bloom!).
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..400i64).map(|i| 100 + i * 3),
            )),
            Arc::new(Int64Array::from_iter_values((0..400i64).map(|i| i % 2))),
        ],
    )?;
    let dim_dir = write_parquet_files(&[dim_batch], 1, 400)?;

    let make_ctx = |mode: &str| {
        let fp = fact_dir.path().to_str().unwrap().to_owned();
        let dp = dim_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = true;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("fact", &fp, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("dim", &dp, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;

    // 1. Static bounds strictly tighter than dynamic [103, 1297]: dynamic bounds skip,
    //    while static [300, 900] must reject matching Bloom keys in [103..299] & [901..1297].
    let tighter_sql = "SELECT COUNT(*), MIN(f.k), MAX(f.k), SUM(f.v) \
                       FROM fact f INNER JOIN dim d ON f.k = d.dk \
                       WHERE d.keep = 1 AND f.k >= 300 AND f.k <= 900";
    let (p1, _) = assert_native_parity(&stock, &native, tighter_sql).await?;
    assert_eq!(
        metric(p1.as_ref(), "join_dynamic_filters_pushed"),
        3,
        "expected dynamic lo + hi + Bloom (3) for 200-key filtered build"
    );

    // 2. Static lower bound equal to dynamic min (`f.k >= 103`, plus reversed literal
    //    form `103 <= f.k`) alongside dynamic Bloom: static bound must be preserved.
    let equal_min_sql = "SELECT COUNT(*), MIN(f.k), MAX(f.k), SUM(f.v) \
                         FROM fact f INNER JOIN dim d ON f.k = d.dk \
                         WHERE d.keep = 1 AND f.k >= 103 AND f.k <= 600";
    let (p2, _) = assert_native_parity(&stock, &native, equal_min_sql).await?;
    assert_eq!(metric(p2.as_ref(), "join_dynamic_filters_pushed"), 3);

    let reversed_lit_sql = "SELECT COUNT(*), MIN(f.k), MAX(f.k), SUM(f.v) \
                            FROM fact f INNER JOIN dim d ON f.k = d.dk \
                            WHERE d.keep = 1 AND 103 <= f.k AND 600 >= f.k";
    let (p3, _) = assert_native_parity(&stock, &native, reversed_lit_sql).await?;
    assert_eq!(metric(p3.as_ref(), "join_dynamic_filters_pushed"), 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn radix_build_semi_anti_flat_int_and_prefilter_parity() -> Result<()> {
    // Build > RADIX_JOIN_THRESHOLD_ROWS (20,000 rows, sparse keys so it enters
    // 64-partition radix finalize and builds `RegisterBloom64`) and probe with
    // both `ProbeRows::All` (Q22-like) and `ProbeRows::Mask` (Q4-like) across
    // BuildSemi (EXISTS) and BuildAnti (NOT EXISTS), including duplicates,
    // NULLs, and composite 2-key Int64Pair.
    let build_schema = Arc::new(Schema::new(vec![
        Field::new("ok", DataType::Int64, true),
        Field::new("ok2", DataType::Int64, true),
        Field::new("ok32", DataType::Int32, true),
        Field::new("prio", DataType::Int32, false),
    ]));
    let n_build = 25_000i64;
    let b_ok: Vec<Option<i64>> = (0..n_build)
        .map(|i| {
            if i % 997 == 0 {
                None
            } else {
                // Sparse keys spanning [0..200_000_000] with duplicates every 17th row
                let base = if i % 17 == 0 { i - 1 } else { i };
                Some(base.wrapping_mul(10_007))
            }
        })
        .collect();
    let b_ok2: Vec<Option<i64>> = (0..n_build)
        .map(|i| if i % 991 == 0 { None } else { Some(i % 13) })
        .collect();
    let b_ok32: Vec<Option<i32>> = (0..n_build)
        .map(|i| {
            if i % 997 == 0 {
                None
            } else {
                Some(((i.wrapping_mul(10_007)) & 0x3fff_ffff) as i32)
            }
        })
        .collect();
    let b_prio: Vec<i32> = (0..n_build).map(|i| (i % 5) as i32).collect();
    let build_batch = RecordBatch::try_new(
        Arc::clone(&build_schema),
        vec![
            Arc::new(Int64Array::from(b_ok)),
            Arc::new(Int64Array::from(b_ok2)),
            Arc::new(Int32Array::from(b_ok32)),
            Arc::new(Int32Array::from(b_prio)),
        ],
    )?;
    let build_batches = vec![
        build_batch.slice(0, 12_500),
        build_batch.slice(12_500, 12_500),
    ];
    let build_dir = write_parquet_files(&build_batches, 2, 5_000)?;

    let probe_schema = Arc::new(Schema::new(vec![
        Field::new("lk", DataType::Int64, true),
        Field::new("lk2", DataType::Int64, true),
        Field::new("lk32", DataType::Int32, true),
        Field::new("cdate", DataType::Int32, false),
        Field::new("rdate", DataType::Int32, false),
    ]));
    let n_probe = 60_000i64;
    let p_lk: Vec<Option<i64>> = (0..n_probe)
        .map(|i| {
            if i % 1009 == 0 {
                None
            } else if i % 4 == 0 {
                Some((i % n_build).wrapping_mul(10_007))
            } else {
                Some(i.wrapping_mul(10_009) + 3)
            }
        })
        .collect();
    let p_lk2: Vec<Option<i64>> = (0..n_probe)
        .map(|i| if i % 1013 == 0 { None } else { Some(i % 13) })
        .collect();
    let p_lk32: Vec<Option<i32>> = (0..n_probe)
        .map(|i| {
            if i % 1009 == 0 {
                None
            } else if i % 4 == 0 {
                Some((((i % n_build).wrapping_mul(10_007)) & 0x3fff_ffff) as i32)
            } else {
                Some(((i.wrapping_mul(10_009) + 3) & 0x3fff_ffff) as i32)
            }
        })
        .collect();
    let p_cdate: Vec<i32> = (0..n_probe).map(|i| (i % 100) as i32).collect();
    let p_rdate: Vec<i32> = (0..n_probe).map(|i| ((i * 7 + 30) % 120) as i32).collect();
    let probe_batch = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int64Array::from(p_lk)),
            Arc::new(Int64Array::from(p_lk2)),
            Arc::new(Int32Array::from(p_lk32)),
            Arc::new(Int32Array::from(p_cdate)),
            Arc::new(Int32Array::from(p_rdate)),
        ],
    )?;
    let probe_batches = vec![
        probe_batch.slice(0, 30_000),
        probe_batch.slice(30_000, 30_000),
    ];
    let probe_dir = write_parquet_files(&probe_batches, 2, 10_000)?;

    let make_ctx = |mode: &str| {
        let bp = build_dir.path().to_str().unwrap().to_owned();
        let pp = probe_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = true;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("orders", &bp, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("lineitem", &pp, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;

    // 1. Q4-like BuildSemi (filtered build > 16,384 rows -> radix + Bloom +
    //    RegisterBloom64, static probe filter -> ProbeRows::Mask, and
    //    all-predicate probe columns -> ForceBulk + RegisterBloom64).
    let q4_like = "SELECT o.prio, COUNT(*) AS cnt \
                   FROM orders o \
                   WHERE o.prio < 4 \
                     AND EXISTS ( \
                       SELECT 1 FROM lineitem l \
                       WHERE l.lk = o.ok AND l.cdate < l.rdate \
                   ) \
                   GROUP BY o.prio ORDER BY o.prio";
    let (p_semi, _) = assert_native_parity(&stock, &native, q4_like).await?;
    assert!(
        metric(p_semi.as_ref(), "adaptive_bulk_decode_rgs") > 0,
        "expected Q4-like all-predicate radix probe to use BulkDecode + RegisterBloom64"
    );

    // 2. Q22-like BuildAnti (NOT EXISTS, single projected probe key -> ProbeRows::All).
    let q22_like = "SELECT o.prio, COUNT(*) AS cnt \
                    FROM orders o \
                    WHERE o.prio < 4 \
                      AND NOT EXISTS ( \
                        SELECT 1 FROM lineitem l WHERE l.lk = o.ok \
                    ) \
                    GROUP BY o.prio ORDER BY o.prio";
    let (p_anti, _) = assert_native_parity(&stock, &native, q22_like).await?;
    assert!(
        metric(p_anti.as_ref(), "adaptive_bulk_decode_rgs") > 0,
        "expected Q22-like all-predicate radix probe to use BulkDecode + RegisterBloom64"
    );

    // 3. Int32 BuildSemi and Int64Pair composite BuildAnti.
    let int32_semi = "SELECT o.prio, COUNT(*) AS cnt \
                      FROM orders o \
                      WHERE EXISTS ( \
                          SELECT 1 FROM lineitem l \
                          WHERE l.lk32 = o.ok32 AND l.cdate < l.rdate \
                      ) \
                      GROUP BY o.prio ORDER BY o.prio";
    assert_native_parity(&stock, &native, int32_semi).await?;

    let pair_anti = "SELECT o.prio, COUNT(*) AS cnt \
                     FROM orders o \
                     WHERE NOT EXISTS ( \
                         SELECT 1 FROM lineitem l \
                         WHERE l.lk = o.ok AND l.lk2 = o.ok2 \
                     ) \
                     GROUP BY o.prio ORDER BY o.prio";
    assert_native_parity(&stock, &native, pair_anti).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compact_span_dense_int_and_small_build_bulk_decode_parity() -> Result<()> {
    // Q11-like join: filtered 1-file supplier (1,000 rows over span [1..25,000],
    // 4% density -> compact DenseIntTable) joined with fixed-width Parquet
    // partsupp (no static filter on partsupp -> ForceBulk directly into ArrayMap).
    let supp_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, false),
        Field::new("s_nationkey", DataType::Int64, false),
    ]));
    let n_supp = 25_000i64;
    let supp_batch = RecordBatch::try_new(
        Arc::clone(&supp_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=n_supp)),
            Arc::new(Int64Array::from_iter_values((1..=n_supp).map(|i| i % 25))),
        ],
    )?;
    let supp_dir = write_parquet_files(&[supp_batch], 1, 25_000)?;

    let ps_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int64, false),
        Field::new("ps_suppkey", DataType::Int64, false),
        Field::new("ps_availqty", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Int64, false),
    ]));
    let n_ps = 60_000i64;
    let ps_batch = RecordBatch::try_new(
        Arc::clone(&ps_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..n_ps).map(|i| (i % 4_000) + 1),
            )),
            Arc::new(Int64Array::from_iter_values(
                (0..n_ps).map(|i| ((i.wrapping_mul(1_664_525) + 13) % n_supp) + 1),
            )),
            Arc::new(Int32Array::from_iter_values(
                (0..n_ps).map(|i| ((i % 50) + 1) as i32),
            )),
            Arc::new(Int64Array::from_iter_values(
                (0..n_ps).map(|i| (i % 1_000) + 10),
            )),
        ],
    )?;
    let ps_batches = vec![ps_batch.slice(0, 30_000), ps_batch.slice(30_000, 30_000)];
    let ps_dir = write_parquet_files(&ps_batches, 2, 10_000)?;

    let make_ctx = |mode: &str| {
        let sp = supp_dir.path().to_str().unwrap().to_owned();
        let pp = ps_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = true;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("supplier", &sp, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("partsupp", &pp, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let stock = make_ctx("off").await?;
    let native = make_ctx("required").await?;

    let sql = "SELECT ps.ps_partkey, SUM(ps.ps_supplycost * ps.ps_availqty) AS val \
               FROM partsupp ps INNER JOIN supplier s ON ps.ps_suppkey = s.s_suppkey \
               WHERE s.s_nationkey = 7 \
               GROUP BY ps.ps_partkey \
               ORDER BY val DESC, ps.ps_partkey ASC \
               LIMIT 20";
    let (plan, _) = assert_native_parity(&stock, &native, sql).await?;
    assert!(
        metric(plan.as_ref(), "adaptive_bulk_decode_rgs") >= 6,
        "expected all 6 partsupp RGs to run in ForceBulk against 4%-density small build, got {}",
        metric(plan.as_ref(), "adaptive_bulk_decode_rgs")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_independent_builds_short_circuit_and_narrow_payload_parity()
-> Result<()> {
    // Q16-like 2-build pipeline: multi-file `part` (narrow static payload:
    // p_brand/p_type/p_size in predicate, only 8-byte p_partkey skipped) +
    // 1-file `supplier` (independent small build -> concurrent build wave) +
    // multi-stage COUNT(DISTINCT ps_suppkey).
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    let n_part = 8_000i64;
    let part_batch = RecordBatch::try_new(
        Arc::clone(&part_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=n_part)),
            Arc::new(StringArray::from_iter_values(
                (1..=n_part).map(|i| format!("Brand#{}", (i % 10) + 1)),
            )),
            Arc::new(StringArray::from_iter_values((1..=n_part).map(|i| {
                if i % 7 == 0 {
                    format!("MEDIUM POLISHED STEEL {i}")
                } else {
                    format!("STANDARD BRASS {i}")
                }
            }))),
            Arc::new(Int32Array::from_iter_values(
                (1..=n_part).map(|i| ((i % 20) + 1) as i32),
            )),
        ],
    )?;
    let part_batches = vec![part_batch.slice(0, 4_000), part_batch.slice(4_000, 4_000)];
    let part_dir = write_parquet_files(&part_batches, 2, 2_000)?;

    let supp_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, true),
        Field::new("s_comment", DataType::Utf8, false),
    ]));
    let n_supp = 500i64;
    let supp_batch = RecordBatch::try_new(
        Arc::clone(&supp_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=n_supp)),
            Arc::new(StringArray::from_iter_values((1..=n_supp).map(|i| {
                if i % 25 == 0 {
                    format!("Customer {i} Complaints")
                } else {
                    format!("Normal supplier {i}")
                }
            }))),
        ],
    )?;
    let supp_dir = write_parquet_files(&[supp_batch], 1, 500)?;

    let supp_null_batch = RecordBatch::try_new(
        Arc::clone(&supp_schema),
        vec![
            Arc::new(Int64Array::from(
                (1..=n_supp)
                    .map(|i| if i == 25 { None } else { Some(i) })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from_iter_values((1..=n_supp).map(|i| {
                if i % 25 == 0 {
                    format!("Customer {i} Complaints")
                } else {
                    format!("Normal supplier {i}")
                }
            }))),
        ],
    )?;
    let supp_null_dir = write_parquet_files(&[supp_null_batch], 1, 500)?;

    let ps_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int64, false),
        Field::new("ps_suppkey", DataType::Int64, false),
    ]));
    let n_ps = 32_000i64;
    let ps_batch = RecordBatch::try_new(
        Arc::clone(&ps_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..n_ps).map(|i| (i % n_part) + 1),
            )),
            Arc::new(Int64Array::from_iter_values(
                (0..n_ps).map(|i| ((i.wrapping_mul(37)) % n_supp) + 1),
            )),
        ],
    )?;
    let ps_batches = vec![ps_batch.slice(0, 16_000), ps_batch.slice(16_000, 16_000)];
    let ps_dir = write_parquet_files(&ps_batches, 2, 8_000)?;

    let make_ctx = |mode: &str, supp_path: &str| {
        let pp = part_dir.path().to_str().unwrap().to_owned();
        let sp = supp_path.to_owned();
        let psp = ps_dir.path().to_str().unwrap().to_owned();
        let mode = mode.to_owned();
        async move {
            let mut cfg = SessionConfig::new().with_target_partitions(4);
            cfg.options_mut().execution.parquet.pushdown_filters = true;
            cfg.options_mut().execution.parquet.pruning = true;
            cfg.options_mut().execution.collect_statistics = true;
            let ctx = SessionContext::new_with_config(cfg);
            ctx.register_parquet("part", &pp, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("supplier", &sp, ParquetReadOptions::default())
                .await?;
            ctx.register_parquet("partsupp", &psp, ParquetReadOptions::default())
                .await?;
            ctx.sql(&format!("SET datafusion.execution.morsel_mode = '{mode}'"))
                .await?;
            Ok::<_, DataFusionError>(ctx)
        }
    };

    let supp_path = supp_dir.path().to_str().unwrap();
    let stock = make_ctx("off", supp_path).await?;
    let native = make_ctx("required", supp_path).await?;

    // 1. Normal Q16-like query: concurrent `part || supplier` build wave,
    //    narrow-static-payload ForceBulk on `part`, and multi-stage COUNT(DISTINCT).
    let q16_sql = "SELECT p.p_brand, p.p_type, p.p_size, COUNT(DISTINCT ps.ps_suppkey) AS cnt \
                   FROM partsupp ps, part p \
                   WHERE p.p_partkey = ps.ps_partkey \
                     AND p.p_brand <> 'Brand#5' \
                     AND p.p_type NOT LIKE 'MEDIUM POLISHED%' \
                     AND p.p_size IN (1, 3, 5, 7, 9, 11) \
                     AND ps.ps_suppkey NOT IN ( \
                         SELECT s.s_suppkey FROM supplier s \
                         WHERE s.s_comment LIKE '%Customer%Complaints%' \
                     ) \
                   GROUP BY p.p_brand, p.p_type, p.p_size \
                   ORDER BY cnt DESC, p.p_brand, p.p_type, p.p_size \
                   LIMIT 20";
    let (plan, _) = assert_native_parity(&stock, &native, q16_sql).await?;
    assert!(
        metric(plan.as_ref(), "adaptive_bulk_decode_rgs") >= 4,
        "expected narrow-payload part scan (4 RGs) to execute in BulkDecode, got {}",
        metric(plan.as_ref(), "adaptive_bulk_decode_rgs")
    );

    // 2. Empty inner build (`part` matches 0 rows) inside concurrent wave must
    //    short-circuit cleanly and abort the sibling `supplier` build.
    let empty_part_sql = "SELECT p.p_brand, COUNT(DISTINCT ps.ps_suppkey) AS cnt \
                          FROM partsupp ps, part p \
                          WHERE p.p_partkey = ps.ps_partkey \
                            AND p.p_size > 999 \
                            AND ps.ps_suppkey NOT IN ( \
                                SELECT s.s_suppkey FROM supplier s \
                                WHERE s.s_comment LIKE '%Customer%Complaints%' \
                            ) \
                          GROUP BY p.p_brand";
    let (empty_plan, _) = assert_native_parity(&stock, &native, empty_part_sql).await?;
    assert_eq!(
        metric(empty_plan.as_ref(), "join_empty_build_short_circuits"),
        1
    );

    // 3. Null-aware anti build with a NULL key inside concurrent wave must
    //    short-circuit cleanly and return 0 rows.
    let supp_null_path = supp_null_dir.path().to_str().unwrap();
    let stock_null = make_ctx("off", supp_null_path).await?;
    let native_null = make_ctx("required", supp_null_path).await?;
    let (null_plan, _) = assert_native_parity(&stock_null, &native_null, q16_sql).await?;
    assert_eq!(
        metric(null_plan.as_ref(), "join_null_aware_short_circuits"),
        1
    );
    Ok(())
}
