// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(all(feature = "sql", feature = "parquet"))]

use std::fs::File;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int32Array, Int64Array, LargeStringArray, MapBuilder, RecordBatch, StringArray,
    StringBuilder, StringViewArray, TimestampMicrosecondArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion::common::config::MorselMode;
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::{Expr, Limit, LogicalPlan, SortExpr, col};
use datafusion::morsel::{MorselMetrics, MorselQueryPlanner};
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

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

fn morsel_context_with_pool(
    workers: usize,
    batch_size: usize,
    pool: Arc<dyn MemoryPool>,
    pushdown: bool,
) -> Result<SessionContext> {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    config.options_mut().execution.parquet.pushdown_filters = pushdown;
    config.options_mut().execution.parquet.reorder_filters = pushdown;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .build_arc()?;
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(workers).unwrap(),
        )))
        .build();
    Ok(SessionContext::new_with_state(state))
}

async fn physical(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

fn ordered_rows(batches: &[RecordBatch]) -> Result<Vec<OwnedRow>> {
    let Some(first) = batches.iter().find(|b| b.num_rows() > 0) else {
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
    Ok(rows)
}

fn ordered_projected_rows(
    batches: &[RecordBatch],
    col_indices: &[usize],
) -> Result<Vec<OwnedRow>> {
    let Some(first) = batches.iter().find(|b| b.num_rows() > 0) else {
        return Ok(Vec::new());
    };
    let fields = col_indices
        .iter()
        .map(|&idx| SortField::new(first.schema().field(idx).data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let mut rows = Vec::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let cols: Vec<ArrayRef> = col_indices
            .iter()
            .map(|&idx| Arc::clone(batch.column(idx)))
            .collect();
        let converted = converter.convert_columns(&cols)?;
        rows.extend(converted.iter().map(|row| row.owned()));
    }
    Ok(rows)
}

fn assert_schemas_match(
    native_plan: &Arc<dyn ExecutionPlan>,
    stock_plan: &Arc<dyn ExecutionPlan>,
    sql: &str,
) {
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
}

/// Asserts exact ordered row-by-row parity (without re-sorting!) between stock
/// DataFusion and native `MorselExec`.
async fn assert_ordered_parity(
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
    assert_schemas_match(&native_plan, &stock_plan, sql);

    let actual = collect(Arc::clone(&native_plan), native.task_ctx()).await?;
    assert_eq!(
        ordered_rows(&actual)?,
        ordered_rows(&expected)?,
        "ordered row parity mismatch for `{sql}`"
    );
    Ok((native_plan, actual))
}

/// For queries whose sort key has ties, verifies exact row count, exact
/// row-by-row sort-key sequence match, and (for full sort) exact multiset equality.
async fn assert_ordered_key_parity_with_ties(
    stock: &SessionContext,
    native: &SessionContext,
    sql: &str,
    key_cols: &[usize],
    check_full_multiset: bool,
) -> Result<()> {
    let stock_plan = physical(stock, sql).await?;
    let expected = collect(Arc::clone(&stock_plan), stock.task_ctx()).await?;

    let native_plan = physical(native, sql).await?;
    assert_eq!(
        native_plan.name(),
        "MorselExec",
        "expected native MorselExec for `{sql}`, got:\n{}",
        displayable(native_plan.as_ref()).indent(true)
    );
    assert_schemas_match(&native_plan, &stock_plan, sql);

    let actual = collect(Arc::clone(&native_plan), native.task_ctx()).await?;
    assert_eq!(
        ordered_projected_rows(&actual, key_cols)?,
        ordered_projected_rows(&expected, key_cols)?,
        "ordered sort-key sequence mismatch for `{sql}`"
    );
    if check_full_multiset {
        let mut actual_all = ordered_rows(&actual)?;
        let mut expected_all = ordered_rows(&expected)?;
        actual_all.sort();
        expected_all.sort();
        assert_eq!(
            actual_all, expected_all,
            "full multiset mismatch for `{sql}`"
        );
    }
    Ok(())
}

fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
    plan.metrics()
        .and_then(|m| m.sum_by_name(name))
        .map_or(0, |v| v.as_usize())
}

fn build_rich_dataset(
    num_rows: usize,
    chunk_size: usize,
) -> Result<(Arc<Schema>, Vec<RecordBatch>)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("k", DataType::Int64, true),
        Field::new("u", DataType::UInt64, true),
        Field::new("f32", DataType::Float32, true),
        Field::new("f64", DataType::Float64, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("ls", DataType::LargeUtf8, true),
        Field::new("sv", DataType::Utf8View, true),
        Field::new("b", DataType::Boolean, true),
        Field::new("d", DataType::Date32, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("dec", DataType::Decimal128(10, 2), true),
        Field::new("grp", DataType::Int32, true),
    ]));

    let mut ids = Vec::with_capacity(num_rows);
    let mut ks = Vec::with_capacity(num_rows);
    let mut us = Vec::with_capacity(num_rows);
    let mut f32s = Vec::with_capacity(num_rows);
    let mut f64s = Vec::with_capacity(num_rows);
    let mut ss: Vec<Option<String>> = Vec::with_capacity(num_rows);
    let mut bs = Vec::with_capacity(num_rows);
    let mut ds = Vec::with_capacity(num_rows);
    let mut tss = Vec::with_capacity(num_rows);
    let mut decs = Vec::with_capacity(num_rows);
    let mut grps = Vec::with_capacity(num_rows);

    let special_f64 = [
        Some(f64::NAN),
        Some(-f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(0.0),
        Some(-0.0),
        None,
    ];
    let special_f32 = [
        Some(f32::NAN),
        Some(-f32::NAN),
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
        Some(0.0),
        Some(-0.0),
        None,
    ];

    for i in 0..num_rows {
        let h = ((i as u64).wrapping_mul(2654435761) ^ ((i as u64) >> 3)) as i64;
        ids.push(i as i32);
        ks.push(if i % 9 == 0 {
            None
        } else {
            Some((h % 37) - 18)
        });
        us.push(if i % 11 == 0 {
            None
        } else {
            Some((h.unsigned_abs()) % 50)
        });
        f32s.push(if i < special_f32.len() {
            special_f32[i]
        } else if i % 13 == 0 {
            None
        } else {
            Some(((h % 200) as f32) / 10.0)
        });
        f64s.push(if i < special_f64.len() {
            special_f64[i]
        } else if i % 13 == 0 {
            None
        } else {
            Some(((h % 200) as f64) / 10.0)
        });
        ss.push(if i % 7 == 0 {
            None
        } else if i % 17 == 0 {
            Some(String::new())
        } else {
            Some(format!("val-{:03}-{}", (h.unsigned_abs() % 29), i % 5))
        });
        bs.push(if i % 5 == 0 { None } else { Some(i % 2 == 0) });
        ds.push(if i % 8 == 0 {
            None
        } else {
            Some(19000 + (h % 40) as i32)
        });
        tss.push(if i % 10 == 0 {
            None
        } else {
            Some(1_700_000_000_000_000 + (h % 1000) * 1_000)
        });
        decs.push(if i % 6 == 0 {
            None
        } else {
            Some(((h % 500) - 250) as i128)
        });
        grps.push(if i % 12 == 0 {
            None
        } else {
            Some((i % 8) as i32)
        });
    }

    let mut batches = Vec::new();
    for start in (0..num_rows).step_by(chunk_size.max(1)) {
        let end = (start + chunk_size).min(num_rows);
        let s_slice: Vec<Option<&str>> =
            ss[start..end].iter().map(|s| s.as_deref()).collect();
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(ids[start..end].to_vec())),
                Arc::new(Int64Array::from(ks[start..end].to_vec())),
                Arc::new(UInt64Array::from(us[start..end].to_vec())),
                Arc::new(Float32Array::from(f32s[start..end].to_vec())),
                Arc::new(Float64Array::from(f64s[start..end].to_vec())),
                Arc::new(StringArray::from(s_slice.clone())),
                Arc::new(LargeStringArray::from(s_slice.clone())),
                Arc::new(StringViewArray::from(s_slice)),
                Arc::new(BooleanArray::from(bs[start..end].to_vec())),
                Arc::new(Date32Array::from(ds[start..end].to_vec())),
                Arc::new(TimestampMicrosecondArray::from(tss[start..end].to_vec())),
                Arc::new(
                    Decimal128Array::from(decs[start..end].to_vec())
                        .with_precision_and_scale(10, 2)?,
                ),
                Arc::new(Int32Array::from(grps[start..end].to_vec())),
            ],
        )?);
    }
    Ok((schema, batches))
}

fn write_parquet_files(
    batches: &[RecordBatch],
    files: usize,
    max_rg_rows: usize,
    skewed: bool,
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
            let target = if skewed {
                if i % 4 == 0 { i % files } else { 0 }
            } else {
                i % files
            };
            if target == file_idx {
                writer.write(batch)?;
            }
        }
        writer.close()?;
    }
    Ok(dir)
}

#[tokio::test]
async fn topn_and_sort_memtable_exact_ordered_parity() -> Result<()> {
    let (schema, batches) = build_rich_dataset(240, 17)?;
    // Distribute batches across 4 partitions so multi-worker claiming is exercised.
    let mut partitions: Vec<Vec<RecordBatch>> = vec![vec![]; 4];
    for (i, b) in batches.into_iter().enumerate() {
        partitions[i % 4].push(b);
    }
    let table = Arc::new(MemTable::try_new(Arc::clone(&schema), partitions)?);
    let empty_table = Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]])?);

    for workers in [1usize, 4usize] {
        let stock = stock_context(workers, 11);
        let native = morsel_context(workers, 11, MorselMode::Required, false);
        for ctx in [&stock, &native] {
            ctx.register_table("t", table.clone())?;
            ctx.register_table("empty_t", empty_table.clone())?;
        }

        let queries = [
            // TopN edge cases: LIMIT 0, LIMIT 1, LIMIT K, LIMIT == rows, LIMIT > rows
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC LIMIT 0",
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC LIMIT 0 OFFSET 5",
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC LIMIT 1",
            "SELECT id, k, s FROM t ORDER BY k DESC, id DESC LIMIT 7",
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC LIMIT 240",
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC LIMIT 500",
            // OFFSET cases
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC LIMIT 10 OFFSET 0",
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC LIMIT 10 OFFSET 5",
            "SELECT id, k, s FROM t ORDER BY k DESC, id ASC LIMIT 15 OFFSET 230",
            "SELECT id, k, s FROM t ORDER BY k DESC, id ASC LIMIT 10 OFFSET 240",
            "SELECT id, k, s FROM t ORDER BY k DESC, id ASC LIMIT 10 OFFSET 300",
            // Full Sort with OFFSET (no LIMIT)
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC OFFSET 13",
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC OFFSET 240",
            // Full Sort (no LIMIT, no OFFSET)
            "SELECT id, k, s FROM t ORDER BY k ASC, id ASC",
            "SELECT id, k, s FROM t ORDER BY k DESC, id DESC",
            // All 4 ASC/DESC x NULLS FIRST/LAST combinations
            "SELECT id, k FROM t ORDER BY k ASC NULLS FIRST, id ASC LIMIT 25",
            "SELECT id, k FROM t ORDER BY k ASC NULLS LAST, id ASC LIMIT 25",
            "SELECT id, k FROM t ORDER BY k DESC NULLS FIRST, id ASC LIMIT 25",
            "SELECT id, k FROM t ORDER BY k DESC NULLS LAST, id ASC LIMIT 25",
            "SELECT id, k FROM t ORDER BY k ASC NULLS FIRST, id ASC",
            "SELECT id, k FROM t ORDER BY k DESC NULLS LAST, id ASC",
            // Floats including -NaN, NaN, -Inf, +Inf, -0.0, +0.0, NULL
            "SELECT id, f64 FROM t ORDER BY f64 ASC NULLS FIRST, id ASC LIMIT 20",
            "SELECT id, f64 FROM t ORDER BY f64 ASC NULLS LAST, id ASC LIMIT 20",
            "SELECT id, f64 FROM t ORDER BY f64 DESC NULLS FIRST, id ASC LIMIT 20",
            "SELECT id, f64 FROM t ORDER BY f64 DESC NULLS LAST, id ASC LIMIT 20",
            "SELECT id, f32 FROM t ORDER BY f32 DESC NULLS LAST, id ASC",
            // Strings (Utf8, LargeUtf8, Utf8View)
            "SELECT id, s FROM t ORDER BY s ASC NULLS FIRST, id ASC LIMIT 18 OFFSET 4",
            "SELECT id, ls FROM t ORDER BY ls DESC NULLS LAST, id ASC LIMIT 18",
            "SELECT id, sv FROM t ORDER BY sv ASC NULLS LAST, id DESC",
            // Multi-column composite keys & diverse types (UInt64, Boolean, Date32, Timestamp, Decimal128)
            "SELECT id, grp, k, s FROM t ORDER BY grp ASC NULLS LAST, k DESC NULLS FIRST, s ASC NULLS LAST, id ASC LIMIT 30 OFFSET 7",
            "SELECT id, b, u, dec FROM t ORDER BY b DESC NULLS FIRST, u ASC NULLS LAST, dec DESC NULLS LAST, id ASC",
            "SELECT id, d, ts FROM t ORDER BY d ASC NULLS FIRST, ts DESC NULLS LAST, id ASC LIMIT 15 OFFSET 3",
            // Computed sort expressions
            "SELECT id, k FROM t ORDER BY COALESCE(k, 0) * -3 + 1 DESC, id ASC LIMIT 12 OFFSET 2",
            "SELECT id, s FROM t ORDER BY length(s) DESC NULLS LAST, id ASC LIMIT 10",
            // Projection above Sort: sort key projected away, aliased, or computed
            "SELECT s FROM t ORDER BY k DESC NULLS LAST, id ASC LIMIT 15 OFFSET 3",
            "SELECT id AS row_num, k + 1 AS k1 FROM t ORDER BY k ASC NULLS FIRST, id ASC LIMIT 12",
            "SELECT s FROM t WHERE k IS NOT NULL AND k > -5 ORDER BY k DESC, id ASC LIMIT 10 OFFSET 2",
            "SELECT id, s FROM t WHERE k IS NOT NULL AND k > 0 ORDER BY s ASC NULLS LAST, id ASC",
            // Empty table
            "SELECT id, k FROM empty_t ORDER BY k DESC LIMIT 5",
            "SELECT id, k FROM empty_t ORDER BY k ASC LIMIT 5 OFFSET 2",
            "SELECT id, k FROM empty_t ORDER BY k ASC",
        ];

        for sql in queries {
            assert_ordered_parity(&stock, &native, sql).await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn topn_zero_limit_without_eliminate_limit_optimizer() -> Result<()> {
    let (schema, batches) = build_rich_dataset(64, 16)?;
    let table = Arc::new(MemTable::try_new(Arc::clone(&schema), vec![batches])?);

    // Disable EliminateLimit so LogicalPlan::Sort { fetch: Some(0) } reaches MorselQueryPlanner.
    let mut config = SessionConfig::new()
        .with_target_partitions(4)
        .with_batch_size(16);
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    let default_state = SessionStateBuilder::new_with_default_features().build();
    let rules: Vec<_> = default_state
        .optimizer()
        .rules
        .iter()
        .filter(|r| r.name() != "eliminate_limit")
        .cloned()
        .collect();
    let native_state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_optimizer_rules(rules)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(4).unwrap(),
        )))
        .build();
    let native = SessionContext::new_with_state(native_state);
    native.register_table("t", table.clone())?;

    let stock = stock_context(4, 16);
    stock.register_table("t", table)?;

    for sql in [
        "SELECT id, k FROM t ORDER BY k DESC LIMIT 0",
        "SELECT id, k FROM t ORDER BY k ASC LIMIT 0 OFFSET 5",
    ] {
        let (plan, batches) = assert_ordered_parity(&stock, &native, sql).await?;
        let display = displayable(plan.as_ref()).indent(true).to_string();
        assert!(
            display.contains("TopNSink") && display.contains("limit=0"),
            "expected TopNSink with limit=0 for `{sql}`, got:\n{display}"
        );
        assert!(
            plan.properties().output_ordering().is_some(),
            "expected output ordering preserved on TopNSink(limit=0)"
        );
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn topn_and_sort_ties_parity() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("v", DataType::Int32, false),
    ]));
    // 400 rows with only 5 distinct keys (+ NULLs) across 8 batches -> heavy ties!
    let mut batches = Vec::new();
    for b in 0..8 {
        let mut ks = Vec::new();
        let mut ss = Vec::new();
        let mut vs = Vec::new();
        for i in 0..50 {
            let row = b * 50 + i;
            let k = if row % 6 == 0 { None } else { Some(row % 5) };
            let s = k.map(|x| format!("g{x}"));
            ks.push(k);
            ss.push(s);
            vs.push(row);
        }
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(ks)),
                Arc::new(StringArray::from(ss)),
                Arc::new(Int32Array::from(vs)),
            ],
        )?);
    }
    let table = Arc::new(MemTable::try_new(schema, vec![batches])?);
    let stock = stock_context(4, 13);
    let native = morsel_context(4, 13, MorselMode::Required, false);
    for ctx in [&stock, &native] {
        ctx.register_table("ties", table.clone())?;
    }

    // Full sort with ties: key sequence + full multiset must match exactly.
    for sql in [
        "SELECT k, v FROM ties ORDER BY k ASC NULLS FIRST",
        "SELECT k, v FROM ties ORDER BY k ASC NULLS LAST",
        "SELECT k, v FROM ties ORDER BY k DESC NULLS FIRST",
        "SELECT k, v FROM ties ORDER BY k DESC NULLS LAST",
        "SELECT s, v FROM ties ORDER BY s DESC NULLS LAST",
    ] {
        assert_ordered_key_parity_with_ties(&stock, &native, sql, &[0], true).await?;
    }

    // TopN slicing right through tie groups: key sequence and row count must match exactly.
    for sql in [
        "SELECT k, v FROM ties ORDER BY k ASC NULLS FIRST LIMIT 100 OFFSET 25",
        "SELECT k, v FROM ties ORDER BY k DESC NULLS LAST LIMIT 75 OFFSET 10",
        "SELECT s, v FROM ties ORDER BY s ASC NULLS LAST LIMIT 50 OFFSET 60",
    ] {
        assert_ordered_key_parity_with_ties(&stock, &native, sql, &[0], false).await?;
    }
    Ok(())
}

#[tokio::test]
async fn topn_and_sort_parquet_parity_pushdown_and_skew() -> Result<()> {
    let (_schema, batches) = build_rich_dataset(360, 24)?;
    let dir_balanced = write_parquet_files(&batches, 4, 30, false)?;
    let dir_skewed = write_parquet_files(&batches, 4, 25, true)?;

    for pushdown in [false, true] {
        let stock = stock_context(4, 19);
        let native = morsel_context(4, 19, MorselMode::Required, pushdown);
        for ctx in [&stock, &native] {
            ctx.register_parquet(
                "pq_bal",
                dir_balanced.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
            ctx.register_parquet(
                "pq_skew",
                dir_skewed.path().to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        }

        for tbl in ["pq_bal", "pq_skew"] {
            let queries = [
                format!(
                    "SELECT id, k, s FROM {tbl} ORDER BY k DESC NULLS LAST, id ASC LIMIT 12"
                ),
                format!(
                    "SELECT id, k, s FROM {tbl} ORDER BY k ASC NULLS FIRST, id ASC LIMIT 15 OFFSET 7"
                ),
                format!("SELECT id, k FROM {tbl} ORDER BY k ASC, id ASC LIMIT 0"),
                format!("SELECT id, k FROM {tbl} ORDER BY k DESC, id ASC LIMIT 1000"),
                format!(
                    "SELECT id, f64, s FROM {tbl} WHERE k IS NOT NULL AND k > -10 ORDER BY f64 ASC NULLS LAST, s DESC NULLS FIRST, id ASC LIMIT 20 OFFSET 4"
                ),
                format!(
                    "SELECT id, s FROM {tbl} WHERE grp >= 2 ORDER BY s ASC NULLS LAST, id DESC"
                ),
                format!(
                    "SELECT id, k, dec FROM {tbl} ORDER BY dec DESC NULLS FIRST, id ASC OFFSET 10"
                ),
            ];
            for sql in queries {
                assert_ordered_parity(&stock, &native, &sql).await?;
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn topn_and_sort_post_group_by_and_post_hash_join_parity() -> Result<()> {
    let (_schema, fact_batches) = build_rich_dataset(320, 20)?;
    let fact_schema = fact_batches[0].schema();
    let fact_dir = write_parquet_files(&fact_batches, 4, 32, true)?;

    // Dimension table for HashJoin + Sort/TopN
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, true),
        Field::new("weight", DataType::Int64, false),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                None,
            ])),
            Arc::new(Int64Array::from(vec![100, -50, 25, 0, 75, -10, 40, 999])),
            Arc::new(StringArray::from(vec![
                Some("alpha"),
                Some("beta"),
                None,
                Some("delta"),
                Some("alpha"),
                Some("zeta"),
                Some("gamma"),
                Some("null_key"),
            ])),
        ],
    )?;
    let dim_dir = write_parquet_files(std::slice::from_ref(&dim_batch), 2, 4, false)?;

    let empty_dim = Arc::new(MemTable::try_new(Arc::clone(&dim_schema), vec![vec![]])?);
    let fact_mem = Arc::new(MemTable::try_new(fact_schema, vec![fact_batches])?);
    let dim_mem = Arc::new(MemTable::try_new(dim_schema, vec![vec![dim_batch]])?);

    let stock = stock_context(4, 17);
    let native = morsel_context(4, 17, MorselMode::Required, false);
    for ctx in [&stock, &native] {
        ctx.register_table("fact_mem", fact_mem.clone())?;
        ctx.register_table("dim_mem", dim_mem.clone())?;
        ctx.register_table("empty_dim", empty_dim.clone())?;
        ctx.register_parquet(
            "fact_pq",
            fact_dir.path().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "dim_pq",
            dim_dir.path().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    }

    for (f, d) in [("fact_mem", "dim_mem"), ("fact_pq", "dim_pq")] {
        let queries = [
            // Post-Count(*)
            format!("SELECT COUNT(*) AS n FROM {f} ORDER BY n DESC LIMIT 5"),
            // Post-GroupCount (integer key COUNT(*))
            format!(
                "SELECT grp, COUNT(*) AS n FROM {f} GROUP BY grp ORDER BY n DESC, grp ASC NULLS LAST LIMIT 5 OFFSET 1"
            ),
            format!(
                "SELECT grp, COUNT(*) AS n FROM {f} GROUP BY grp ORDER BY grp DESC NULLS FIRST"
            ),
            // Post-General Aggregate (SUM, MIN, MAX, AVG, COUNT)
            format!(
                "SELECT s, COUNT(*) AS c, SUM(k) AS sk, AVG(k) AS ak, ROUND(AVG(f64), 6) AS af FROM {f} GROUP BY s ORDER BY sk DESC NULLS LAST, s ASC NULLS FIRST LIMIT 10 OFFSET 2"
            ),
            format!(
                "SELECT grp, SUM(k) AS sk, MIN(k) AS mn, MAX(k) AS mx FROM {f} GROUP BY grp ORDER BY grp ASC NULLS LAST"
            ),
            // Post-HashJoin TopN and full Sort
            format!(
                "SELECT f.id, d.tag, COALESCE(f.k, 0) + d.weight AS total FROM {f} f JOIN {d} d ON f.grp = d.grp ORDER BY total DESC, f.id ASC LIMIT 15 OFFSET 4"
            ),
            format!(
                "SELECT f.id, d.tag, f.k FROM {f} f JOIN {d} d ON f.grp = d.grp WHERE f.k IS NOT NULL ORDER BY d.tag ASC NULLS LAST, f.id DESC"
            ),
            // HashJoin -> GROUP BY -> TopN and full Sort
            format!(
                "SELECT d.tag, COUNT(*) AS n, SUM(COALESCE(f.k, 0) + d.weight) AS total FROM {f} f JOIN {d} d ON f.grp = d.grp GROUP BY d.tag ORDER BY total DESC, d.tag ASC NULLS LAST LIMIT 4 OFFSET 1"
            ),
            format!(
                "SELECT d.grp, COUNT(*) AS n FROM {f} f JOIN {d} d ON f.grp = d.grp GROUP BY d.grp ORDER BY n DESC, d.grp ASC"
            ),
        ];
        for sql in queries {
            assert_ordered_parity(&stock, &native, &sql).await?;
        }
    }

    // Empty build-side short-circuit + TopN/Sort
    for sql in [
        "SELECT f.id, d.weight FROM fact_mem f JOIN empty_dim d ON f.grp = d.grp ORDER BY f.id DESC LIMIT 5",
        "SELECT f.id, d.weight FROM fact_mem f JOIN empty_dim d ON f.grp = d.grp ORDER BY f.id ASC",
    ] {
        let (_plan, batches) = assert_ordered_parity(&stock, &native, sql).await?;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn plan_properties_equivalence_and_ordering_reporting() -> Result<()> {
    let (schema, batches) = build_rich_dataset(32, 16)?;
    let table = Arc::new(MemTable::try_new(schema, vec![batches])?);
    let stock = stock_context(4, 16);
    let native = morsel_context(4, 16, MorselMode::Required, false);
    for ctx in [&stock, &native] {
        ctx.register_table("t", table.clone())?;
    }

    // Ordered root plans report exact output ordering and EmissionType::Final.
    for sql in [
        "SELECT id, k FROM t ORDER BY k DESC NULLS FIRST, id ASC NULLS LAST LIMIT 10",
        "SELECT id, k FROM t ORDER BY k ASC NULLS LAST, id DESC NULLS FIRST",
        "SELECT id AS row_id, k FROM t ORDER BY id DESC LIMIT 5",
        "SELECT grp, COUNT(*) AS n FROM t GROUP BY grp ORDER BY n DESC, grp ASC LIMIT 5",
        "SELECT grp, SUM(k) AS sk FROM t GROUP BY grp ORDER BY sk DESC NULLS LAST",
    ] {
        let native_plan = physical(&native, sql).await?;
        let stock_plan = physical(&stock, sql).await?;
        assert_eq!(native_plan.name(), "MorselExec");
        assert_eq!(
            native_plan.properties().emission_type,
            EmissionType::Final,
            "{sql}"
        );
        let native_ord = native_plan
            .properties()
            .output_ordering()
            .expect("expected output ordering on MorselExec");
        let stock_ord = stock_plan
            .properties()
            .output_ordering()
            .expect("expected output ordering on stock plan");
        assert_eq!(
            native_ord, stock_ord,
            "output ordering mismatch for `{sql}`"
        );
    }

    // Projection dropping the sort key must NOT claim an invalid output ordering.
    let dropped_key =
        physical(&native, "SELECT s FROM t ORDER BY k DESC LIMIT 5").await?;
    assert_eq!(dropped_key.name(), "MorselExec");
    assert_eq!(dropped_key.properties().emission_type, EmissionType::Final);
    assert!(
        dropped_key.properties().output_ordering().is_none(),
        "ordering must not be claimed when sort key is projected away"
    );

    // Unordered queries must remain unordered.
    for sql in [
        "SELECT id, k FROM t",
        "SELECT id, k FROM t WHERE k > 0",
        "SELECT grp, COUNT(*) FROM t GROUP BY grp",
        "SELECT COUNT(*) FROM t",
    ] {
        let plan = physical(&native, sql).await?;
        assert_eq!(plan.name(), "MorselExec");
        assert!(
            plan.properties().output_ordering().is_none(),
            "unordered query `{sql}` must not report output ordering"
        );
    }
    Ok(())
}

#[tokio::test]
async fn morsel_mode_required_and_prefer_for_unsupported_sort_limit_shapes() -> Result<()>
{
    let (schema, batches) = build_rich_dataset(40, 20)?;
    let table = Arc::new(MemTable::try_new(schema, vec![batches])?);

    // Table with an unsupported Map sort key type.
    let mut map_builder =
        MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    map_builder.keys().append_value("a");
    map_builder.values().append_value("1");
    map_builder.append(true)?;
    map_builder.keys().append_value("b");
    map_builder.values().append_value("2");
    map_builder.append(true)?;
    let map_arr = Arc::new(map_builder.finish()) as ArrayRef;
    let map_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("m", map_arr.data_type().clone(), true),
    ]));
    let map_batch = RecordBatch::try_new(
        Arc::clone(&map_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), map_arr],
    )?;
    let map_table = Arc::new(MemTable::try_new(map_schema, vec![vec![map_batch]])?);

    let required = morsel_context(2, 16, MorselMode::Required, false);
    let prefer = morsel_context(2, 16, MorselMode::Prefer, false);
    for ctx in [&required, &prefer] {
        ctx.register_table("t", table.clone())?;
        ctx.register_table("mt", map_table.clone())?;
    }

    for (sql, expected_substr) in [
        ("SELECT id FROM t LIMIT 3", "limit"),
        ("SELECT id FROM t OFFSET 2", "offset"),
        ("SELECT grp, COUNT(*) FROM t GROUP BY grp LIMIT 2", "limit"),
        (
            "SELECT * FROM (SELECT id, k FROM t ORDER BY id LIMIT 5) sub WHERE k > 0",
            "unsupported",
        ),
        (
            "SELECT a.id FROM (SELECT id, grp FROM t ORDER BY id LIMIT 5) a JOIN t b ON a.grp = b.grp",
            "unsupported",
        ),
        ("SELECT id FROM mt ORDER BY m", "unsupported sort key"),
    ] {
        let err = physical(&required, sql).await.unwrap_err().to_string();
        assert!(
            err.contains("morsel_mode=required")
                && err.to_lowercase().contains(expected_substr),
            "sql=`{sql}` expected `{expected_substr}` in: {err}"
        );
        // Under prefer mode, unsupported queries fall back to stock DataFusion
        // (note: Map sort may also error in stock SortExec if arrow doesn't sort Map,
        // so only assert non-MorselExec for queries stock can plan).
        if !sql.contains("mt") {
            let plan = physical(&prefer, sql).await?;
            let display = displayable(plan.as_ref()).indent(false).to_string();
            assert!(!display.contains("MorselExec"), "{sql}: {display}");
        }
    }

    // Non-literal LIMIT and OFFSET expressions must fail cleanly in required mode.
    let base_sort = required
        .sql("SELECT id, k FROM t ORDER BY id")
        .await?
        .into_optimized_plan()?;
    let non_literal_limit = LogicalPlan::Limit(Limit {
        skip: None,
        fetch: Some(Box::new(Expr::Placeholder(
            datafusion::logical_expr::expr::Placeholder::new_with_field(
                "$1".into(),
                None,
            ),
        ))),
        input: Arc::new(base_sort.clone()),
    });
    let err = required
        .state()
        .create_physical_plan(&non_literal_limit)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("morsel_mode=required") && err.contains("non-literal LIMIT"),
        "{err}"
    );

    let non_literal_offset = LogicalPlan::Limit(Limit {
        skip: Some(Box::new(Expr::Placeholder(
            datafusion::logical_expr::expr::Placeholder::new_with_field(
                "$1".into(),
                None,
            ),
        ))),
        fetch: None,
        input: Arc::new(base_sort.clone()),
    });
    let err = required
        .state()
        .create_physical_plan(&non_literal_offset)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("morsel_mode=required") && err.contains("non-literal OFFSET"),
        "{err}"
    );

    // Nested Sort must fail cleanly in required mode.
    let nested_sort = LogicalPlan::Sort(datafusion::logical_expr::Sort {
        expr: vec![SortExpr::new(col("id"), false, false)],
        input: Arc::new(base_sort),
        fetch: None,
    });
    let err = required
        .state()
        .create_physical_plan(&nested_sort)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("morsel_mode=required") && err.contains("nested Sort"),
        "{err}"
    );
    Ok(())
}

#[tokio::test]
async fn morsel_metrics_138_explain_and_oom_cleanup_end_to_end() -> Result<()> {
    let (schema, batches) = build_rich_dataset(200, 20)?;
    let dir = write_parquet_files(&batches, 4, 25, false)?;
    let table = Arc::new(MemTable::try_new(schema, vec![batches])?);

    // 1. Verify #138 MorselMetrics and EXPLAIN summary for TopN and Sort.
    let collector = Arc::new(MorselMetrics::default());
    let mut config = SessionConfig::new()
        .with_target_partitions(4)
        .with_batch_size(16)
        .with_extension(Arc::clone(&collector));
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(4).unwrap(),
        )))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t", table.clone())?;
    ctx.register_parquet(
        "pq",
        dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    let topn_plan = physical(
        &ctx,
        "SELECT id, k FROM pq ORDER BY k DESC, id ASC LIMIT 10 OFFSET 2",
    )
    .await?;
    let topn_display = displayable(topn_plan.as_ref()).indent(true).to_string();
    assert!(
        topn_display.contains("TopNSink")
            && topn_display.contains("FinalizedTopNSource")
            && topn_display.contains("limit=10")
            && topn_display.contains("offset=2"),
        "{topn_display}"
    );
    let _ = collect(Arc::clone(&topn_plan), ctx.task_ctx()).await?;
    assert_eq!(metric(topn_plan.as_ref(), "topn_finalizations"), 1);

    let sort_plan = physical(&ctx, "SELECT id, k FROM pq ORDER BY k ASC, id ASC").await?;
    let sort_display = displayable(sort_plan.as_ref()).indent(true).to_string();
    assert!(
        sort_display.contains("SortSink") && sort_display.contains("FinalizedSortSource"),
        "{sort_display}"
    );
    let _ = collect(Arc::clone(&sort_plan), ctx.task_ctx()).await?;
    assert_eq!(metric(sort_plan.as_ref(), "sort_finalizations"), 1);

    let runs = collector.take_runs();
    assert_eq!(runs.len(), 2);
    assert!(runs[0].sum_by_name("morsels_completed").unwrap().as_usize() > 0);
    assert!(runs[0].sum_by_name("morsel_topn_finalize_work").is_some());
    assert!(runs[1].sum_by_name("morsels_completed").unwrap().as_usize() > 0);
    assert!(runs[1].sum_by_name("morsel_sort_finalize_work").is_some());

    // 2. End-to-end OOM sweep over MemTable and Parquet: every failure returns
    // ResourcesExhausted and releases 100% of reserved pool memory.
    for sql in [
        "SELECT id, k, s FROM t ORDER BY k DESC, id ASC LIMIT 10",
        "SELECT id, k, s FROM t ORDER BY k ASC, id ASC",
        "SELECT grp, COUNT(*) AS n FROM t GROUP BY grp ORDER BY n DESC, grp ASC",
        "SELECT id, k, s FROM pq ORDER BY k ASC, id ASC",
    ] {
        for budget in [128usize, 1024usize] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(budget));
            let oom_ctx = morsel_context_with_pool(4, 16, Arc::clone(&pool), false)?;
            oom_ctx.register_table("t", table.clone())?;
            oom_ctx
                .register_parquet(
                    "pq",
                    dir.path().to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
            let plan = physical(&oom_ctx, sql).await?;
            let res = collect(plan, oom_ctx.task_ctx()).await;
            if let Err(err) = res {
                assert!(
                    matches!(err, DataFusionError::ResourcesExhausted(_)),
                    "sql=`{sql}` budget={budget}: expected ResourcesExhausted, got {err}"
                );
            }
            assert_eq!(
                pool.reserved(),
                0,
                "leaked memory for sql=`{sql}` budget={budget}"
            );
        }
    }
    Ok(())
}
