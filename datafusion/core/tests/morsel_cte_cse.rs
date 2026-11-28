// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Tests for Materialized CTE & Common Subplan (Stage) CSE in the experimental
//! morsel DAG planner and executor (#228).

#![cfg(all(feature = "sql", feature = "parquet"))]

use std::fs::File;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
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

fn days(year: i32, month: u32, day: u32) -> i32 {
    let y = year - i32::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u32;
    let m = month;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i32 - 719_468
}

fn write_parquet_table(
    dir: &TempDir,
    name: &str,
    batches: &[RecordBatch],
    max_rg_rows: usize,
) -> Result<String> {
    let table_dir = dir.path().join(name);
    std::fs::create_dir_all(&table_dir)?;
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(max_rg_rows))
        .build();
    for (idx, batch) in batches.iter().enumerate() {
        let path = table_dir.join(format!("{idx}.parquet"));
        let mut writer = ArrowWriter::try_new(
            File::create(path)?,
            batch.schema(),
            Some(props.clone()),
        )?;
        writer.write(batch)?;
        writer.close()?;
    }
    Ok(table_dir.to_string_lossy().into_owned())
}

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
    pushdown: bool,
    pool: Option<Arc<dyn MemoryPool>>,
    metrics: Option<Arc<MorselMetrics>>,
) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    config.options_mut().execution.parquet.pushdown_filters = pushdown;
    config.options_mut().execution.parquet.reorder_filters = pushdown;
    config
        .options_mut()
        .optimizer
        .enable_dynamic_filter_pushdown = true;
    config
        .options_mut()
        .optimizer
        .enable_join_dynamic_filter_pushdown = true;
    if let Some(m) = metrics {
        config = config.with_extension(m);
    }
    let mut builder = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(workers).unwrap(),
        )));
    if let Some(pool) = pool {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()
            .unwrap();
        builder = builder.with_runtime_env(runtime);
    }
    SessionContext::new_with_state(builder.build())
}

fn to_rows(batches: &[RecordBatch], sort: bool) -> Result<Vec<OwnedRow>> {
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
    let mut out = Vec::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let rows = converter.convert_columns(batch.columns())?;
        for row in rows.iter() {
            out.push(row.owned());
        }
    }
    if sort {
        out.sort();
    }
    Ok(out)
}

async fn assert_parity(
    stock: &SessionContext,
    native: &SessionContext,
    sql: &str,
    ordered: bool,
) -> Result<(Arc<dyn ExecutionPlan>, Vec<RecordBatch>)> {
    let plan = native.sql(sql).await?.create_physical_plan().await?;
    assert_eq!(plan.name(), "MorselExec", "expected MorselExec for `{sql}`");
    let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
    let expected = stock.sql(sql).await?.collect().await?;
    assert_eq!(
        to_rows(&actual, !ordered)?,
        to_rows(&expected, !ordered)?,
        "result mismatch for `{sql}`"
    );
    Ok((plan, actual))
}

async fn wait_for_zero_reserved(pool: &Arc<dyn MemoryPool>) {
    timeout(Duration::from_secs(5), async {
        while pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!("MemoryPool leaked {} bytes", pool.reserved());
    });
}

fn build_q15_fixture() -> Result<(TempDir, String, String)> {
    let dir = TempDir::new()?;

    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_suppkey", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
        Field::new("l_discount", DataType::Decimal128(15, 2), false),
        Field::new("l_shipdate", DataType::Date32, false),
    ]));
    // 2 files × 4 row groups each (32 rows / RG) = 8 row groups total across 256 rows.
    let mut li_batches = Vec::new();
    for file_idx in 0..2i64 {
        let mut suppkeys = Vec::new();
        let mut prices = Vec::new();
        let mut discounts = Vec::new();
        let mut dates = Vec::new();
        for i in 0..128i64 {
            let supp = (i % 8) + 1;
            let in_window = (i % 2) == 0;
            suppkeys.push(supp);
            // Make supp 5 the clear revenue leader.
            let price = if supp == 5 {
                500_000 + file_idx * 10_000 + i * 100
            } else {
                10_000 + supp * 1_000 + i * 10
            };
            prices.push(price as i128);
            discounts.push(5i128);
            dates.push(if in_window {
                days(1996, 1, 1) + (i % 85) as i32
            } else {
                days(1995, 6, 1) + (i % 30) as i32
            });
        }
        li_batches.push(RecordBatch::try_new(
            Arc::clone(&lineitem_schema),
            vec![
                Arc::new(Int64Array::from(suppkeys)) as ArrayRef,
                Arc::new(Decimal128Array::from(prices).with_precision_and_scale(15, 2)?),
                Arc::new(
                    Decimal128Array::from(discounts).with_precision_and_scale(15, 2)?,
                ),
                Arc::new(Date32Array::from(dates)),
            ],
        )?);
    }

    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, false),
        Field::new("s_name", DataType::Utf8, false),
        Field::new("s_address", DataType::Utf8, false),
        Field::new("s_phone", DataType::Utf8, false),
    ]));
    let supplier_batch = RecordBatch::try_new(
        supplier_schema,
        vec![
            Arc::new(Int64Array::from((1..=8).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(StringArray::from(
                (1..=8).map(|i| format!("Supplier#{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (1..=8).map(|i| format!("Addr#{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (1..=8).map(|i| format!("Phone#{i}")).collect::<Vec<_>>(),
            )),
        ],
    )?;

    let li_path = write_parquet_table(&dir, "lineitem", &li_batches, 32)?;
    let supp_path = write_parquet_table(&dir, "supplier", &[supplier_batch], 4)?;
    Ok((dir, li_path, supp_path))
}

#[tokio::test]
async fn q15_inlined_cte_executes_shared_stage_once_and_halves_scan_metrics() -> Result<()>
{
    let (_dir, li_path, supp_path) = build_q15_fixture()?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let metrics = Arc::new(MorselMetrics::default());
    let stock = stock_context(4, 32);
    let native = morsel_context(
        4,
        32,
        false,
        Some(Arc::clone(&pool)),
        Some(Arc::clone(&metrics)),
    );
    for ctx in [&stock, &native] {
        ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("supplier", &supp_path, ParquetReadOptions::default())
            .await?;
    }

    // Standard TPC-H Q15 (inlined non-MATERIALIZED CTE).
    let q15_sql = "
        WITH revenue0 (supplier_no, total_revenue) AS (
            SELECT
                l_suppkey,
                SUM(l_extendedprice * (1 - l_discount))
            FROM lineitem
            WHERE l_shipdate >= DATE '1996-01-01'
              AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
            GROUP BY l_suppkey
        )
        SELECT
            s_suppkey,
            s_name,
            s_address,
            s_phone,
            total_revenue
        FROM supplier, revenue0
        WHERE s_suppkey = supplier_no
          AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
        ORDER BY s_suppkey
    ";

    let (plan, batches) = assert_parity(&stock, &native, q15_sql, true).await?;
    let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(total_rows, 1, "Q15 should return the single top supplier");

    let explain = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        explain.contains("(shared, consumers=2)"),
        "expected shared stage in Q15 EXPLAIN, got:\n{explain}"
    );
    assert_eq!(
        explain.matches("NativeParquet[lineitem").count(),
        1,
        "expected lineitem to be scanned ONCE in Q15, got:\n{explain}"
    );

    // 1 aggregate for shared revenue0 + 1 for MAX(total_revenue) = 2 (NOT 3).
    let agg_fins = plan
        .metrics()
        .and_then(|m| m.sum_by_name("aggregate_finalizations"))
        .map_or(0, |v| v.as_usize());
    assert_eq!(
        agg_fins, 2,
        "Q15 should execute revenue0 aggregate only once (total 2, not 3)"
    );

    // Verify per-stage metrics report lineitem row-group claims once (8 RGs, NOT 16).
    let runs = metrics.take_runs();
    assert_eq!(runs.len(), 1);
    let cse_claims: usize = runs[0]
        .iter()
        .filter(|m| {
            m.value().name() == "morsels_claimed"
                && m.labels()
                    .iter()
                    .any(|l| l.name() == "pipeline" && l.value().starts_with("cse_"))
        })
        .map(|m| m.value().as_usize())
        .sum();
    assert_eq!(
        cse_claims, 8,
        "expected 8 lineitem row groups claimed once on shared cse stage, got {cse_claims}"
    );

    // Compare bytes_scanned against a single standalone scan of revenue0.
    let single_sql = "
        SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount))
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01'
          AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
        GROUP BY l_suppkey
    ";
    let single_plan = native.sql(single_sql).await?.create_physical_plan().await?;
    let _ = collect(Arc::clone(&single_plan), native.task_ctx()).await?;
    let q15_bytes = plan
        .metrics()
        .and_then(|m| m.sum_by_name("bytes_scanned"))
        .map_or(0, |v| v.as_usize());
    let single_bytes = single_plan
        .metrics()
        .and_then(|m| m.sum_by_name("bytes_scanned"))
        .map_or(0, |v| v.as_usize());
    let supp_plan = native
        .sql("SELECT s_suppkey, s_name, s_address, s_phone FROM supplier")
        .await?
        .create_physical_plan()
        .await?;
    let _ = collect(Arc::clone(&supp_plan), native.task_ctx()).await?;
    let supp_bytes = supp_plan
        .metrics()
        .and_then(|m| m.sum_by_name("bytes_scanned"))
        .map_or(0, |v| v.as_usize());
    assert!(
        single_bytes > 0 && q15_bytes <= single_bytes + supp_bytes,
        "Q15 bytes_scanned ({q15_bytes}) should not exceed single lineitem ({single_bytes}) + supplier ({supp_bytes})"
    );

    wait_for_zero_reserved(&pool).await;
    Ok(())
}

#[tokio::test]
async fn q11_duplicate_filtered_join_shares_stage_once() -> Result<()> {
    let dir = TempDir::new()?;

    let nation_schema = Arc::new(Schema::new(vec![
        Field::new("n_nationkey", DataType::Int64, false),
        Field::new("n_name", DataType::Utf8, false),
    ]));
    let nation_batch = RecordBatch::try_new(
        nation_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 7])) as ArrayRef,
            Arc::new(StringArray::from(vec!["FRANCE", "BRAZIL", "GERMANY"])),
        ],
    )?;

    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, false),
        Field::new("s_nationkey", DataType::Int64, false),
    ]));
    let supplier_batch = RecordBatch::try_new(
        supplier_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])) as ArrayRef,
            Arc::new(Int64Array::from(vec![7, 1, 7, 2, 7, 1])),
        ],
    )?;

    let partsupp_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int64, false),
        Field::new("ps_suppkey", DataType::Int64, false),
        Field::new("ps_availqty", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Decimal128(15, 2), false),
    ]));
    let mut ps_partkeys = Vec::new();
    let mut ps_suppkeys = Vec::new();
    let mut ps_qtys = Vec::new();
    let mut ps_costs = Vec::new();
    for i in 0..64i64 {
        ps_partkeys.push((i % 8) + 1);
        ps_suppkeys.push((i % 6) + 1);
        ps_qtys.push(10 + (i as i32 % 5) * 10);
        ps_costs.push(((i % 8) + 1) as i128 * 10_000);
    }
    let partsupp_batch = RecordBatch::try_new(
        partsupp_schema,
        vec![
            Arc::new(Int64Array::from(ps_partkeys)) as ArrayRef,
            Arc::new(Int64Array::from(ps_suppkeys)),
            Arc::new(Int32Array::from(ps_qtys)),
            Arc::new(Decimal128Array::from(ps_costs).with_precision_and_scale(15, 2)?),
        ],
    )?;

    let nation_path = write_parquet_table(&dir, "nation", &[nation_batch], 4)?;
    let supplier_path = write_parquet_table(&dir, "supplier", &[supplier_batch], 4)?;
    let partsupp_path = write_parquet_table(&dir, "partsupp", &[partsupp_batch], 16)?;

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let stock = stock_context(4, 16);
    let native = morsel_context(4, 16, true, Some(Arc::clone(&pool)), None);
    for ctx in [&stock, &native] {
        ctx.register_parquet("nation", &nation_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("supplier", &supplier_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("partsupp", &partsupp_path, ParquetReadOptions::default())
            .await?;
    }

    let q11_sql = "
        SELECT
            ps_partkey,
            SUM(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey
          AND s_nationkey = n_nationkey
          AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING SUM(ps_supplycost * ps_availqty) > (
            SELECT SUM(ps_supplycost * ps_availqty) * 0.1
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey
              AND s_nationkey = n_nationkey
              AND n_name = 'GERMANY'
        )
        ORDER BY value DESC
    ";

    let (plan, _) = assert_parity(&stock, &native, q11_sql, true).await?;
    let explain = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        explain.contains("(shared, consumers=2)"),
        "expected shared join stage in Q11 EXPLAIN, got:\n{explain}"
    );
    assert_eq!(
        explain.matches("NativeParquet[partsupp").count(),
        1,
        "expected partsupp to be scanned ONCE in Q11, got:\n{explain}"
    );
    let join_builds = plan
        .metrics()
        .and_then(|m| m.sum_by_name("join_build_finalizations"))
        .map_or(0, |v| v.as_usize());
    assert_eq!(
        join_builds, 2,
        "Q11 should build supplier and nation once each (total 2, not 4)"
    );
    wait_for_zero_reserved(&pool).await;
    Ok(())
}

#[tokio::test]
async fn explicit_materialized_cte_and_multi_consumer_fanout_2_and_3_consumers()
-> Result<()> {
    let (_dir, li_path, supp_path) = build_q15_fixture()?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let stock = stock_context(4, 32);
    let native = morsel_context(4, 32, false, Some(Arc::clone(&pool)), None);
    for ctx in [&stock, &native] {
        ctx.sql("SET datafusion.sql_parser.dialect = 'PostgreSQL'")
            .await?
            .collect()
            .await?;
        ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("supplier", &supp_path, ParquetReadOptions::default())
            .await?;
    }

    // 1. Explicit WITH ... AS MATERIALIZED with 3 consumers (main + 2 scalar subqueries).
    let mat_3_sql = "
        WITH rev AS MATERIALIZED (
            SELECT l_suppkey AS sno, SUM(l_extendedprice) AS total
            FROM lineitem
            GROUP BY l_suppkey
        )
        SELECT sno, total
        FROM rev
        WHERE total >= (SELECT MIN(total) FROM rev)
          AND total <= (SELECT MAX(total) FROM rev)
        ORDER BY sno
    ";
    let (plan, _) = assert_parity(&stock, &native, mat_3_sql, true).await?;
    let explain = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        explain.contains("Stage cte_rev (shared, consumers=3)"),
        "expected 3-consumer shared cte_rev stage, got:\n{explain}"
    );
    // 1 for cte_rev + 1 for MIN + 1 for MAX = 3.
    let agg_fins = plan
        .metrics()
        .and_then(|m| m.sum_by_name("aggregate_finalizations"))
        .map_or(0, |v| v.as_usize());
    assert_eq!(agg_fins, 3);

    // Concurrent execution of the same Arc<MorselExec> across 3 tasks.
    let expected = stock.sql(mat_3_sql).await?.collect().await?;
    let (r1, r2, r3) = tokio::try_join!(
        collect(Arc::clone(&plan), native.task_ctx()),
        collect(Arc::clone(&plan), native.task_ctx()),
        collect(Arc::clone(&plan), native.task_ctx()),
    )?;
    for r in [&r1, &r2, &r3] {
        assert_eq!(to_rows(r, false)?, to_rows(&expected, false)?);
    }

    // 2. Non-materialized inlined 3-way self-join over identical grouped aggregate.
    let inlined_3_sql = "
        WITH rev AS (
            SELECT l_suppkey AS sno, SUM(l_extendedprice) AS total
            FROM lineitem
            GROUP BY l_suppkey
        )
        SELECT a.sno, a.total, b.total AS b_total, c.total AS c_total
        FROM rev a
        JOIN rev b ON a.sno = b.sno
        JOIN rev c ON a.sno = c.sno
        ORDER BY a.sno
    ";
    let (plan3, _) = assert_parity(&stock, &native, inlined_3_sql, true).await?;
    let explain3 = displayable(plan3.as_ref()).indent(false).to_string();
    assert!(
        explain3.contains("(shared, consumers=3)"),
        "expected 3-consumer shared CSE stage, got:\n{explain3}"
    );
    assert_eq!(
        explain3.matches("NativeParquet[lineitem").count(),
        1,
        "expected lineitem scanned once for 3-way self-join, got:\n{explain3}"
    );
    let agg_fins3 = plan3
        .metrics()
        .and_then(|m| m.sum_by_name("aggregate_finalizations"))
        .map_or(0, |v| v.as_usize());
    assert_eq!(agg_fins3, 1);

    wait_for_zero_reserved(&pool).await;
    Ok(())
}

#[tokio::test]
async fn subquery_alias_and_output_alias_differences_still_deduplicate() -> Result<()> {
    let (_dir, li_path, _) = build_q15_fixture()?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let stock = stock_context(4, 32);
    let native = morsel_context(4, 32, false, Some(Arc::clone(&pool)), None);
    for ctx in [&stock, &native] {
        ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
            .await?;
    }

    let sql = "
        SELECT left_side.k1, left_side.rev_a, right_side.rev_b
        FROM (
            SELECT l_suppkey AS k1, SUM(l_extendedprice) AS rev_a
            FROM lineitem
            WHERE l_shipdate >= DATE '1996-01-01'
            GROUP BY l_suppkey
        ) left_side
        JOIN (
            SELECT l_suppkey AS k2, SUM(l_extendedprice) AS rev_b
            FROM lineitem
            WHERE l_shipdate >= DATE '1996-01-01'
            GROUP BY l_suppkey
        ) right_side
        ON left_side.k1 = right_side.k2
        ORDER BY left_side.k1
    ";
    let (plan, batches) = assert_parity(&stock, &native, sql, true).await?;
    let explain = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        explain.contains("(shared, consumers=2)"),
        "expected aliased identical subplans to share a stage, got:\n{explain}"
    );
    assert_eq!(explain.matches("NativeParquet[lineitem").count(), 1);
    assert_eq!(
        batches[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        vec!["k1", "rev_a", "rev_b"]
    );
    wait_for_zero_reserved(&pool).await;
    Ok(())
}

#[tokio::test]
async fn volatile_expressions_are_not_deduplicated() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from((0..64).map(|i| i % 4).collect::<Vec<_>>()))
                as ArrayRef,
            Arc::new(Int64Array::from((0..64).collect::<Vec<_>>())),
        ],
    )?;
    let mem = Arc::new(MemTable::try_new(schema, vec![vec![batch]])?);

    let native = morsel_context(4, 16, false, None, None);
    native
        .sql("SET datafusion.sql_parser.dialect = 'PostgreSQL'")
        .await?
        .collect()
        .await?;
    native.register_table("t", mem)?;

    // Inlined volatile subplans must NOT be deduplicated!
    let volatile_sql = "
        SELECT a.k, a.r AS r1, b.r AS r2
        FROM (SELECT k, SUM(CAST(v AS DOUBLE) + random()) AS r FROM t GROUP BY k) a
        JOIN (SELECT k, SUM(CAST(v AS DOUBLE) + random()) AS r FROM t GROUP BY k) b
          ON a.k = b.k
        ORDER BY a.k
    ";
    let plan = native
        .sql(volatile_sql)
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(plan.name(), "MorselExec");
    let explain = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        !explain.contains("(shared, consumers="),
        "volatile subplan must NOT be deduplicated, got:\n{explain}"
    );
    let batches = collect(Arc::clone(&plan), native.task_ctx()).await?;
    let agg_fins = plan
        .metrics()
        .and_then(|m| m.sum_by_name("aggregate_finalizations"))
        .map_or(0, |v| v.as_usize());
    assert_eq!(
        agg_fins, 2,
        "both volatile subplans must execute independently"
    );
    // Verify r1 != r2 on at least one row (probability of collision is ~0).
    let mut any_diff = false;
    for b in &batches {
        let r1 = b
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        let r2 = b
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        for i in 0..b.num_rows() {
            if (r1.value(i) - r2.value(i)).abs() > 1e-12 {
                any_diff = true;
            }
        }
    }
    assert!(
        any_diff,
        "independent volatile random() aggregates should produce distinct values"
    );

    // By contrast, an explicit WITH ... AS MATERIALIZED CTE DOES materialize random() once.
    let explicit_mat_sql = "
        WITH c AS MATERIALIZED (
            SELECT k, SUM(CAST(v AS DOUBLE) + random()) AS r FROM t GROUP BY k
        )
        SELECT a.k, a.r AS r1, b.r AS r2
        FROM c a
        JOIN c b ON a.k = b.k
        ORDER BY a.k
    ";
    let mat_plan = native
        .sql(explicit_mat_sql)
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(mat_plan.name(), "MorselExec");
    let mat_batches = collect(Arc::clone(&mat_plan), native.task_ctx()).await?;
    for b in &mat_batches {
        let r1 = b
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        let r2 = b
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        for i in 0..b.num_rows() {
            assert_eq!(
                r1.value(i),
                r2.value(i),
                "explicit MATERIALIZED CTE must return identical values to all consumers"
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn multi_consumer_dynamic_join_filter_isolation_prevents_corruption() -> Result<()>
{
    let dir = TempDir::new()?;

    // Multi-RG Parquet `fact` table with sorted keys 1..=200 across 10 RGs of 20 rows
    // so Parquet row-group pruning + dynamic join filters would aggressively prune
    // keys > 20 if a selective consumer's filter ever leaked into the shared stage!
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int64Array::from((1..=200).collect::<Vec<_>>())) as ArrayRef,
            // Key 200 has the largest value (200_000), key 1 has small value (1_000).
            Arc::new(Int64Array::from(
                (1..=200).map(|k| k * 1_000).collect::<Vec<_>>(),
            )),
        ],
    )?;

    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("tag", DataType::Utf8, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 199, 200])) as ArrayRef,
            Arc::new(StringArray::from(vec!["LOW", "OTHER", "OTHER", "HIGH"])),
        ],
    )?;

    let fact_path = write_parquet_table(&dir, "fact", &[fact_batch], 20)?;
    let dim_path = write_parquet_table(&dir, "dim", &[dim_batch], 4)?;

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let stock = stock_context(4, 32);
    // Enable full Parquet pushdown + dynamic join filter pushdown!
    let native = morsel_context(4, 32, true, Some(Arc::clone(&pool)), None);
    for ctx in [&stock, &native] {
        ctx.sql("SET datafusion.sql_parser.dialect = 'PostgreSQL'")
            .await?
            .collect()
            .await?;
        ctx.register_parquet("fact", &fact_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("dim", &dim_path, ParquetReadOptions::default())
            .await?;
    }

    // Case 1: Inlined CSE shared stage where Consumer 1 joins with highly selective
    // `dim WHERE tag = 'LOW'` (only k=1!) and Consumer 2 reads unfiltered MAX(total)
    // (at k=200!). If k=1 leaked into the shared stage, MAX(total) would be 1_000
    // instead of 200_000!
    let sql_inlined = "
        WITH agg AS (
            SELECT k, SUM(v) AS total FROM fact GROUP BY k
        )
        SELECT a.k, a.total, (SELECT MAX(total) FROM agg) AS global_max
        FROM agg a
        JOIN dim d ON a.k = d.dk
        WHERE d.tag = 'LOW'
        ORDER BY a.k
    ";
    let (plan1, batches1) = assert_parity(&stock, &native, sql_inlined, true).await?;
    let explain1 = displayable(plan1.as_ref()).indent(false).to_string();
    assert!(
        explain1.contains("(shared, consumers=2)"),
        "expected shared stage in Case 1, got:\n{explain1}"
    );
    assert_eq!(explain1.matches("NativeParquet[fact").count(), 1);
    let max_val = batches1[0]
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(max_val, 200_000);

    // Case 2: Two consumers with DISJOINT selective dynamic join filters (tag='LOW' -> k=1
    // vs tag='HIGH' -> k=200) reading the SAME shared stage!
    let sql_disjoint = "
        WITH agg AS (
            SELECT k, SUM(v) AS total FROM fact GROUP BY k
        )
        SELECT l.k AS low_k, l.total AS low_v, h.k AS high_k, h.total AS high_v
        FROM (SELECT a.k, a.total FROM agg a JOIN dim d ON a.k = d.dk WHERE d.tag = 'LOW') l
        JOIN (SELECT a.k, a.total FROM agg a JOIN dim d ON a.k = d.dk WHERE d.tag = 'HIGH') h
          ON l.total < h.total
    ";
    let (plan2, batches2) = assert_parity(&stock, &native, sql_disjoint, true).await?;
    let explain2 = displayable(plan2.as_ref()).indent(false).to_string();
    assert!(
        explain2.contains("(shared, consumers=2)"),
        "expected shared stage in Case 2, got:\n{explain2}"
    );
    assert_eq!(explain2.matches("NativeParquet[fact").count(), 1);
    assert_eq!(batches2.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);

    // Case 3: Explicit MATERIALIZED CTE with selective join + unfiltered consumer.
    let sql_mat = "
        WITH agg AS MATERIALIZED (
            SELECT k, SUM(v) AS total FROM fact GROUP BY k
        )
        SELECT a.k, a.total, (SELECT MAX(total) FROM agg) AS global_max
        FROM agg a
        JOIN dim d ON a.k = d.dk
        WHERE d.tag = 'LOW'
        ORDER BY a.k
    ";
    let (plan3, batches3) = assert_parity(&stock, &native, sql_mat, true).await?;
    let explain3 = displayable(plan3.as_ref()).indent(false).to_string();
    assert!(
        explain3.contains("Stage cte_agg (shared, consumers=2)"),
        "expected shared cte_agg stage in Case 3, got:\n{explain3}"
    );
    let max_val3 = batches3[0]
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(max_val3, 200_000);

    wait_for_zero_reserved(&pool).await;
    Ok(())
}

#[tokio::test]
async fn single_consumer_cross_aggregate_dynamic_pushdown_preserved() -> Result<()> {
    // Q17 shape: single-consumer grouped Aggregate over multi-RG Parquet `lineitem`
    // joined with selective `part` must still push dynamic filter across StageSource!
    let dir = TempDir::new()?;
    let li_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int64, false),
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    // 10 row groups of 20 rows each, sorted by l_partkey 1..=200.
    let li_batch = RecordBatch::try_new(
        li_schema,
        vec![
            Arc::new(Int64Array::from((1..=200).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Int64Array::from(vec![10; 200])),
            Arc::new(Int64Array::from(vec![100; 200])),
        ],
    )?;
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_brand", DataType::Utf8, false),
    ]));
    let part_batch = RecordBatch::try_new(
        part_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 100, 200])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "Brand#23", "Other", "Other", "Other",
            ])),
        ],
    )?;
    let li_path = write_parquet_table(&dir, "lineitem", &[li_batch], 20)?;
    let part_path = write_parquet_table(&dir, "part", &[part_batch], 4)?;

    let metrics = Arc::new(MorselMetrics::default());
    let stock = stock_context(4, 32);
    let native = morsel_context(4, 32, true, None, Some(Arc::clone(&metrics)));
    for ctx in [&stock, &native] {
        ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("part", &part_path, ParquetReadOptions::default())
            .await?;
    }

    let q17_shape = "
        SELECT SUM(l_extendedprice) AS total
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND l_quantity <= (
              SELECT AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey
          )
    ";
    let (plan, _) = assert_parity(&stock, &native, q17_shape, false).await?;
    let explain = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        !explain.contains("(shared, consumers="),
        "single-consumer Q17 stage must not be marked shared:\n{explain}"
    );
    // With cross-Aggregate dynamic pushdown active, stage_0 prunes 9 of 10 RGs
    // (claims strictly < 10 RGs!).
    let runs = metrics.take_runs();
    assert_eq!(runs.len(), 1);
    let stage0_claims: usize = runs[0]
        .iter()
        .filter(|m| {
            m.value().name() == "morsels_claimed"
                && m.labels()
                    .iter()
                    .any(|l| l.name() == "pipeline" && l.value() == "stage_0")
        })
        .map(|m| m.value().as_usize())
        .sum();
    assert!(
        stage0_claims < 10,
        "expected Q17 single-consumer stage_0 to prune row groups via dynamic filter, claimed {stage0_claims}/10"
    );
    Ok(())
}

#[tokio::test]
async fn shared_stage_memory_pool_oom_error_and_cancellation_cleanup() -> Result<()> {
    let dir = TempDir::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    // ~1.5 MB of unique group strings across 4096 groups so materializing the shared
    // stage exceeds a tiny 256 KB GreedyMemoryPool!
    let n = 4096i64;
    let payload: Vec<String> = (0..n).map(|i| format!("bad_{i:0>252}")).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from((0..n).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(StringArray::from(payload)),
        ],
    )?;
    let path = write_parquet_table(&dir, "big", &[batch], 512)?;

    let cse_sql = "
        WITH g AS (
            SELECT k, MAX(payload) AS p FROM big GROUP BY k
        )
        SELECT a.k, a.p
        FROM g a
        JOIN g b ON a.k = b.k
        ORDER BY a.k
    ";

    // 1. Tiny pool -> OOM cleanly without hanging or leaking.
    let tiny_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128 * 1024));
    let tiny_ctx = morsel_context(4, 256, false, Some(Arc::clone(&tiny_pool)), None);
    tiny_ctx
        .register_parquet("big", &path, ParquetReadOptions::default())
        .await?;
    let oom_plan = tiny_ctx.sql(cse_sql).await?.create_physical_plan().await?;
    let oom_res = timeout(
        Duration::from_secs(10),
        collect(oom_plan, tiny_ctx.task_ctx()),
    )
    .await
    .expect("OOM must not hang");
    assert!(
        matches!(oom_res, Err(DataFusionError::ResourcesExhausted(_))),
        "expected ResourcesExhausted, got: {oom_res:?}"
    );
    wait_for_zero_reserved(&tiny_pool).await;

    // 2. Early drop / cancellation mid-execution -> clean abort & 0 leaked bytes.
    let ok_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let ok_ctx = morsel_context(4, 64, false, Some(Arc::clone(&ok_pool)), None);
    ok_ctx
        .register_parquet("big", &path, ParquetReadOptions::default())
        .await?;
    let cancel_plan = ok_ctx.sql(cse_sql).await?.create_physical_plan().await?;
    {
        let mut stream = execute_stream(Arc::clone(&cancel_plan), ok_ctx.task_ctx())?;
        // Poll one batch then drop immediately.
        let _ = stream.next().await;
    }
    wait_for_zero_reserved(&ok_pool).await;

    // Drop before first poll.
    {
        let stream = execute_stream(Arc::clone(&cancel_plan), ok_ctx.task_ctx())?;
        drop(stream);
    }
    wait_for_zero_reserved(&ok_pool).await;

    // 3. Mid-execution runtime error inside shared stage (CAST invalid string to INT) ->
    //    both consumers abort cleanly with error, no hang, 0 leaked bytes.
    let err_sql = "
        WITH bad AS (
            SELECT k, MAX(CAST(payload AS BIGINT)) AS p FROM big GROUP BY k
        )
        SELECT a.k, a.p
        FROM bad a
        WHERE a.p = (SELECT MAX(p) FROM bad)
    ";
    let err_plan = ok_ctx.sql(err_sql).await?.create_physical_plan().await?;
    let err_res = timeout(
        Duration::from_secs(10),
        collect(err_plan, ok_ctx.task_ctx()),
    )
    .await
    .expect("error inside shared stage must not hang");
    assert!(
        err_res.is_err(),
        "expected execution error from invalid CAST"
    );
    wait_for_zero_reserved(&ok_pool).await;

    Ok(())
}

#[tokio::test]
async fn nested_multi_level_cse_and_deduplicated_scalar_subquery() -> Result<()> {
    let (_dir, li_path, _) = build_q15_fixture()?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let stock = stock_context(4, 32);
    let native = morsel_context(4, 32, false, Some(Arc::clone(&pool)), None);
    for ctx in [&stock, &native] {
        ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
            .await?;
    }

    // 1. Nested multi-level CSE: `a` shared inside `b`, and `b` shared in outer query.
    let nested_sql = "
        WITH a AS (
            SELECT l_suppkey AS k, SUM(l_extendedprice) AS rev
            FROM lineitem
            GROUP BY l_suppkey
        ),
        b AS (
            SELECT x.k, x.rev + y.rev AS doubled
            FROM a x
            JOIN a y ON x.k = y.k
        )
        SELECT b1.k, b1.doubled, b2.doubled AS doubled2
        FROM b b1
        JOIN b b2 ON b1.k = b2.k
        ORDER BY b1.k
    ";
    let (nested_plan, _) = assert_parity(&stock, &native, nested_sql, true).await?;
    let explain = displayable(nested_plan.as_ref()).indent(false).to_string();
    assert_eq!(
        explain.matches("NativeParquet[lineitem").count(),
        1,
        "nested multi-level CSE should scan lineitem only once, got:\n{explain}"
    );
    assert_eq!(
        nested_plan
            .metrics()
            .and_then(|m| m.sum_by_name("aggregate_finalizations"))
            .map_or(0, |v| v.as_usize()),
        1,
        "inner aggregate `a` should execute only once"
    );

    // 2. Identical scalar subquery referenced twice in WHERE is already deduplicated
    //    by SubqueryIndex (must execute aggregate once and produce exact parity).
    let repeated_sq = "
        SELECT l_suppkey
        FROM lineitem
        WHERE l_extendedprice >= (SELECT AVG(l_extendedprice) FROM lineitem)
          AND l_extendedprice <= (SELECT AVG(l_extendedprice) FROM lineitem) * 10
        ORDER BY l_suppkey
    ";
    let (sq_plan, _) = assert_parity(&stock, &native, repeated_sq, true).await?;
    assert_eq!(
        sq_plan
            .metrics()
            .and_then(|m| m.sum_by_name("aggregate_finalizations"))
            .map_or(0, |v| v.as_usize()),
        1,
        "repeated identical scalar subquery should execute only once"
    );

    wait_for_zero_reserved(&pool).await;
    Ok(())
}

#[tokio::test]
async fn distinct_topn_empty_shared_stage_and_subset_before_superset_join_cse()
-> Result<()> {
    let (_dir, li_path, supp_path) = build_q15_fixture()?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let stock = stock_context(4, 32);
    let native = morsel_context(4, 32, true, Some(Arc::clone(&pool)), None);
    for ctx in [&stock, &native] {
        ctx.register_parquet("lineitem", &li_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("supplier", &supp_path, ParquetReadOptions::default())
            .await?;
    }

    // 1. Duplicate DISTINCT subplan shares a single stage.
    let distinct_sql = "
        WITH d AS (
            SELECT DISTINCT l_suppkey FROM lineitem WHERE l_shipdate >= DATE '1996-01-01'
        )
        SELECT a.l_suppkey, (SELECT COUNT(*) FROM d) AS total_distinct
        FROM d a
        ORDER BY a.l_suppkey
    ";
    let (d_plan, _) = assert_parity(&stock, &native, distinct_sql, true).await?;
    let d_explain = displayable(d_plan.as_ref()).indent(false).to_string();
    assert!(
        d_explain.contains("(shared, consumers=2)"),
        "expected duplicate DISTINCT to share a stage, got:\n{d_explain}"
    );
    assert_eq!(d_explain.matches("NativeParquet[lineitem").count(), 1);

    // 2. Duplicate TopN (ORDER BY ... LIMIT) subplan shares a single stage.
    let topn_sql = "
        WITH top3 AS (
            SELECT s_suppkey, s_name FROM supplier ORDER BY s_suppkey DESC LIMIT 3
        )
        SELECT a.s_suppkey, a.s_name, b.s_name AS b_name
        FROM top3 a
        JOIN top3 b ON a.s_suppkey = b.s_suppkey
        ORDER BY a.s_suppkey
    ";
    let (t_plan, _) = assert_parity(&stock, &native, topn_sql, true).await?;
    let t_explain = displayable(t_plan.as_ref()).indent(false).to_string();
    assert!(
        t_explain.contains("(shared, consumers=2)"),
        "expected duplicate TopN to share a stage, got:\n{t_explain}"
    );
    assert_eq!(t_explain.matches("NativeParquet[supplier").count(), 1);

    // 3. Empty shared stage (0 rows) feeding 2 consumers + empty-build short-circuit.
    let empty_sql = "
        WITH empty_rev AS (
            SELECT l_suppkey AS sno, SUM(l_extendedprice) AS total
            FROM lineitem
            WHERE l_shipdate > DATE '2099-01-01'
            GROUP BY l_suppkey
        )
        SELECT a.sno, a.total, b.total AS b_total
        FROM empty_rev a
        JOIN empty_rev b ON a.sno = b.sno
    ";
    let (e_plan, e_batches) = assert_parity(&stock, &native, empty_sql, false).await?;
    let e_explain = displayable(e_plan.as_ref()).indent(false).to_string();
    assert!(
        e_explain.contains("(shared, consumers=2)"),
        "expected empty duplicate aggregate to share a stage, got:\n{e_explain}"
    );
    assert_eq!(
        e_batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        0
    );

    // 4. Projection-subsumed Join CSE where the NARROWER (subset) branch is on
    //    the LEFT (visited first) and the WIDER (superset) branch is on the RIGHT!
    let subset_first_sql = "
        SELECT a.s_suppkey, a.sum_p, b.s_name, b.sum_p AS sum_p2
        FROM (
            SELECT s.s_suppkey, SUM(l.l_extendedprice) AS sum_p
            FROM lineitem l
            JOIN supplier s ON l.l_suppkey = s.s_suppkey
            WHERE l.l_shipdate >= DATE '1996-01-01'
            GROUP BY s.s_suppkey
        ) a
        JOIN (
            SELECT s.s_suppkey, s.s_name, SUM(l.l_extendedprice) AS sum_p
            FROM lineitem l
            JOIN supplier s ON l.l_suppkey = s.s_suppkey
            WHERE l.l_shipdate >= DATE '1996-01-01'
            GROUP BY s.s_suppkey, s.s_name
        ) b ON a.s_suppkey = b.s_suppkey
        ORDER BY a.s_suppkey
    ";
    let (s_plan, _) = assert_parity(&stock, &native, subset_first_sql, true).await?;
    let s_explain = displayable(s_plan.as_ref()).indent(false).to_string();
    assert!(
        s_explain.contains("(shared, consumers=2)"),
        "expected subset-before-superset Join CSE to share a stage, got:\n{s_explain}"
    );
    assert_eq!(
        s_explain.matches("NativeParquet[lineitem").count(),
        1,
        "expected lineitem scanned once in subset-before-superset Join CSE, got:\n{s_explain}"
    );

    wait_for_zero_reserved(&pool).await;
    Ok(())
}

#[tokio::test]
async fn trivially_true_union_tag_filter_does_not_materialize_unfiltered_join_under_selective_dim()
-> Result<()> {
    // Regression test for TPC-DS Q4/Q11/Q74: pushing a literal tag filter
    // (`sale_type = 's'`) through `UNION ALL` eliminates the other branch but
    // leaves `partial_filters=[Boolean(true)]` on the surviving `customer`
    // scan. That trivially-true filter must NOT fool Join CSE into
    // materializing the unfiltered `customer ⋈ store_sales` join and blocking
    // per-consumer `date_dim` (`d_year = 2001` vs `d_year = 2002`) dynamic
    // join filter pruning.
    let dir = TempDir::new()?;

    let cust_schema = Arc::new(Schema::new(vec![
        Field::new("c_customer_sk", DataType::Int64, false),
        Field::new("c_customer_id", DataType::Utf8, false),
    ]));
    let cust_batch = RecordBatch::try_new(
        cust_schema,
        vec![
            Arc::new(Int64Array::from((1..=20).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(StringArray::from(
                (1..=20).map(|i| format!("CUST_{i:02}")).collect::<Vec<_>>(),
            )),
        ],
    )?;

    // 10 years (2000..=2009), 10 date keys per year -> 100 date_dim rows.
    let date_schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int64, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    let date_batch = RecordBatch::try_new(
        date_schema,
        vec![
            Arc::new(Int64Array::from((0..100).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Int32Array::from(
                (0..100).map(|d| 2000 + (d / 10)).collect::<Vec<_>>(),
            )),
        ],
    )?;

    // 10 row groups of 20 rows each, sorted by date_sk so each year occupies
    // exactly 1 row group (10% of the fact table).
    let make_sales = |cust_col: &str, date_col: &str, amt_col: &str, scale: i64| {
        let schema = Arc::new(Schema::new(vec![
            Field::new(cust_col, DataType::Int64, false),
            Field::new(date_col, DataType::Int64, false),
            Field::new(amt_col, DataType::Int64, false),
        ]));
        let mut csk = Vec::with_capacity(200);
        let mut dsk = Vec::with_capacity(200);
        let mut amt = Vec::with_capacity(200);
        for year_idx in 0..10i64 {
            for r in 0..20i64 {
                csk.push((r % 20) + 1);
                dsk.push(year_idx * 10 + (r % 10));
                amt.push(scale * (year_idx + 1) * (r + 1));
            }
        }
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(csk)) as ArrayRef,
                Arc::new(Int64Array::from(dsk)),
                Arc::new(Int64Array::from(amt)),
            ],
        )
    };
    let ss_batch = make_sales("ss_customer_sk", "ss_sold_date_sk", "ss_amt", 10)?;
    let ws_batch = make_sales("ws_customer_sk", "ws_sold_date_sk", "ws_amt", 20)?;

    let cust_path = write_parquet_table(&dir, "customer", &[cust_batch], 20)?;
    let date_path = write_parquet_table(&dir, "date_dim", &[date_batch], 100)?;
    let ss_path = write_parquet_table(&dir, "store_sales", &[ss_batch], 20)?;
    let ws_path = write_parquet_table(&dir, "web_sales", &[ws_batch], 20)?;

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let metrics = Arc::new(MorselMetrics::default());
    let stock = stock_context(4, 32);
    let native = morsel_context(
        4,
        32,
        true,
        Some(Arc::clone(&pool)),
        Some(Arc::clone(&metrics)),
    );
    for ctx in [&stock, &native] {
        ctx.register_parquet("customer", &cust_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("date_dim", &date_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("store_sales", &ss_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("web_sales", &ws_path, ParquetReadOptions::default())
            .await?;
    }

    let union_cte_sql = "
        WITH year_total AS (
            SELECT c_customer_id AS cid, d_year AS dyear, SUM(ss_amt) AS total, 's' AS sale_type
            FROM customer, store_sales, date_dim
            WHERE c_customer_sk = ss_customer_sk
              AND ss_sold_date_sk = d_date_sk
            GROUP BY c_customer_id, d_year
            UNION ALL
            SELECT c_customer_id AS cid, d_year AS dyear, SUM(ws_amt) AS total, 'w' AS sale_type
            FROM customer, web_sales, date_dim
            WHERE c_customer_sk = ws_customer_sk
              AND ws_sold_date_sk = d_date_sk
            GROUP BY c_customer_id, d_year
        )
        SELECT t1.cid, t1.total AS y1, t2.total AS y2
        FROM year_total t1, year_total t2
        WHERE t1.cid = t2.cid
          AND t1.sale_type = 's'
          AND t2.sale_type = 's'
          AND t1.dyear = 2001
          AND t2.dyear = 2002
        ORDER BY t1.cid
    ";

    let direct_sql = "
        SELECT t1.cid, t1.total AS y1, t2.total AS y2
        FROM (
            SELECT c_customer_id AS cid, SUM(ss_amt) AS total
            FROM customer, store_sales, date_dim
            WHERE c_customer_sk = ss_customer_sk
              AND ss_sold_date_sk = d_date_sk
              AND d_year = 2001
            GROUP BY c_customer_id
        ) t1
        JOIN (
            SELECT c_customer_id AS cid, SUM(ss_amt) AS total
            FROM customer, store_sales, date_dim
            WHERE c_customer_sk = ss_customer_sk
              AND ss_sold_date_sk = d_date_sk
              AND d_year = 2002
            GROUP BY c_customer_id
        ) t2 ON t1.cid = t2.cid
        ORDER BY t1.cid
    ";

    let (union_plan, _) = assert_parity(&stock, &native, union_cte_sql, true).await?;
    let union_explain = displayable(union_plan.as_ref()).indent(false).to_string();
    assert!(
        !union_explain.contains("(shared, consumers="),
        "trivially-true UNION ALL tag filter must not create a shared stage over unfiltered fact⋈dim join, got:\n{union_explain}"
    );
    let union_runs = metrics.take_runs();
    assert_eq!(union_runs.len(), 1);
    let union_claims: usize = union_runs[0]
        .sum_by_name("morsels_claimed")
        .map_or(0, |v| v.as_usize());

    let (direct_plan, _) = assert_parity(&stock, &native, direct_sql, true).await?;
    let direct_runs = metrics.take_runs();
    assert_eq!(direct_runs.len(), 1);
    let direct_claims: usize = direct_runs[0]
        .sum_by_name("morsels_claimed")
        .map_or(0, |v| v.as_usize());

    assert_eq!(
        union_claims,
        direct_claims,
        "UNION ALL tagged CTE must preserve the exact same dynamic-filter RG pruning as direct query (union={union_claims}, direct={direct_claims})\nunion plan:\n{union_explain}\ndirect plan:\n{}",
        displayable(direct_plan.as_ref()).indent(false)
    );
    let union_bytes = union_plan
        .metrics()
        .and_then(|m| m.sum_by_name("bytes_scanned"))
        .map_or(0, |v| v.as_usize());
    let direct_bytes = direct_plan
        .metrics()
        .and_then(|m| m.sum_by_name("bytes_scanned"))
        .map_or(0, |v| v.as_usize());
    assert_eq!(
        union_bytes, direct_bytes,
        "expected identical bytes_scanned with dynamic filter pruning preserved"
    );

    wait_for_zero_reserved(&pool).await;
    Ok(())
}
