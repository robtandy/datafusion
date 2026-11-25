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
