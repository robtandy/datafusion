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
        "SELECT a.k, b.v FROM a LEFT JOIN b ON a.k = b.k",
        "SELECT a.k, b.v FROM a RIGHT JOIN b ON a.k = b.k",
        "SELECT a.k, b.v FROM a FULL OUTER JOIN b ON a.k = b.k",
        "SELECT a.k FROM a WHERE EXISTS (SELECT 1 FROM b WHERE a.k = b.k)",
        "SELECT a.k FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE a.k = b.k)",
        "SELECT a.k, b.v FROM a CROSS JOIN b",
        "SELECT a.k, b.v FROM a JOIN b ON a.v < b.v",
        "SELECT a.k, b.v FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k",
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
