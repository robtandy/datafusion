// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Deterministic out-of-core spilling, radix-partitioning, parity, skew, error,
//! and zero-leak cleanup tests for the experimental morsel engine (`GROUP BY`,
//! `HashJoin`, `Sort`).

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use arrow::array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    StringViewBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::config::MorselMode;
use datafusion::common::hash_utils::{RandomState, create_hashes};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::memory_pool::{
    FairSpillPool, GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use datafusion::morsel::MorselQueryPlanner;
use datafusion::physical_plan::{ExecutionPlan, collect, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_common_runtime::SpawnedTask;
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::time::timeout;

fn stock_context(workers: usize, batch_size: usize) -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    SessionContext::new_with_config(config)
}

fn morsel_context(
    workers: usize,
    batch_size: usize,
    runtime: Arc<RuntimeEnv>,
) -> Result<SessionContext> {
    morsel_context_with_pushdown(workers, batch_size, runtime, false)
}

fn morsel_context_with_pushdown(
    workers: usize,
    batch_size: usize,
    runtime: Arc<RuntimeEnv>,
    pushdown_filters: bool,
) -> Result<SessionContext> {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(workers).unwrap(),
        )))
        .build();
    Ok(SessionContext::new_with_state(state))
}

fn runtime_with_pool(
    pool: Arc<dyn MemoryPool>,
    max_temp_bytes: Option<u64>,
) -> Result<Arc<RuntimeEnv>> {
    let mut dm = DiskManagerBuilder::default().with_mode(DiskManagerMode::OsTmpDirectory);
    if let Some(limit) = max_temp_bytes {
        dm = dm.with_max_temp_directory_size(limit);
    }
    RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .with_disk_manager_builder(dm)
        .build_arc()
}

fn spilled_bytes(plan: &dyn ExecutionPlan) -> usize {
    plan.metrics().and_then(|m| m.spilled_bytes()).unwrap_or(0)
}

fn spill_count(plan: &dyn ExecutionPlan) -> usize {
    plan.metrics().and_then(|m| m.spill_count()).unwrap_or(0)
}

fn assert_zero_leaks(runtime: &RuntimeEnv, label: &str) {
    assert_eq!(
        runtime.memory_pool.reserved(),
        0,
        "{label}: MemoryPool leaked {} bytes",
        runtime.memory_pool.reserved()
    );
    let progress = runtime.disk_manager.spilling_progress();
    assert_eq!(
        progress.active_files_count, 0,
        "{label}: DiskManager leaked {} active spill files",
        progress.active_files_count
    );
    assert_eq!(
        progress.current_bytes, 0,
        "{label}: DiskManager leaked {} active spill bytes",
        progress.current_bytes
    );
}

async fn wait_zero_leaks(runtime: &RuntimeEnv, label: &str) {
    timeout(Duration::from_secs(5), async {
        while runtime.memory_pool.reserved() != 0
            || runtime.disk_manager.spilling_progress().active_files_count != 0
            || runtime.disk_manager.spilling_progress().current_bytes != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "{label}: resources leaked (mem={}, files={}, disk_bytes={})",
            runtime.memory_pool.reserved(),
            runtime.disk_manager.spilling_progress().active_files_count,
            runtime.disk_manager.spilling_progress().current_bytes
        )
    });
}

fn high_cardinality_group_table(
    num_batches: usize,
    rows_per_batch: usize,
) -> Arc<MemTable> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k_int", DataType::Int64, true),
        Field::new("k_str", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
        Field::new("gate", DataType::Int32, false),
    ]));
    let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for b in 0..num_batches {
        let mut k_int = Vec::with_capacity(rows_per_batch);
        let mut k_str = Vec::with_capacity(rows_per_batch);
        let mut v = Vec::with_capacity(rows_per_batch);
        let mut gate = Vec::with_capacity(rows_per_batch);
        for i in 0..rows_per_batch {
            let gid = (b * rows_per_batch + i) % (num_batches * rows_per_batch / 2);
            if gid.is_multiple_of(1009) {
                k_int.push(None);
                k_str.push(None);
            } else {
                k_int.push(Some(gid as i64));
                k_str.push(Some(format!("group-key-{gid:08}-padding-payload-xyz")));
            }
            v.push(if i.is_multiple_of(17) {
                None
            } else {
                Some(((gid * 7 + i) % 10_000) as i64)
            });
            gate.push(i32::from(!(b == num_batches / 2 && i == 10)));
        }
        partitions[b % 4].push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(k_int)),
                    Arc::new(StringArray::from(k_str)),
                    Arc::new(Int64Array::from(v)),
                    Arc::new(Int32Array::from(gate)),
                ],
            )
            .unwrap(),
        );
    }
    Arc::new(MemTable::try_new(schema, partitions).unwrap())
}

fn large_join_tables(
    build_rows: usize,
    probe_rows: usize,
    batch_rows: usize,
) -> (Arc<MemTable>, Arc<MemTable>) {
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Int64, true),
        Field::new("bstr", DataType::Utf8, true),
        Field::new("bval", DataType::Utf8, true),
        Field::new("bgate", DataType::Int32, false),
    ]));
    let p_schema = Arc::new(Schema::new(vec![
        Field::new("pk", DataType::Int64, true),
        Field::new("pstr", DataType::Utf8, true),
        Field::new("pval", DataType::Utf8, true),
        Field::new("pgate", DataType::Int32, false),
    ]));

    let mut b_parts: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for (idx, start) in (0..build_rows).step_by(batch_rows).enumerate() {
        let len = (build_rows - start).min(batch_rows);
        let mut keys = Vec::with_capacity(len);
        let mut strs = Vec::with_capacity(len);
        let mut vals = Vec::with_capacity(len);
        let mut gates = Vec::with_capacity(len);
        for i in start..start + len {
            if i % 251 == 0 {
                keys.push(None);
                strs.push(None);
            } else {
                let k = ((i * 31) % (build_rows * 7 / 10 + 1)) as i64;
                keys.push(Some(k));
                strs.push(Some(format!("k-{k:06}")));
            }
            vals.push(Some(format!(
                "build-payload-{i:08}-padding-bytes-0123456789"
            )));
            gates.push(i32::from(i != build_rows / 2));
        }
        b_parts[idx % 4].push(
            RecordBatch::try_new(
                Arc::clone(&b_schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(StringArray::from(strs)),
                    Arc::new(StringArray::from(vals)),
                    Arc::new(Int32Array::from(gates)),
                ],
            )
            .unwrap(),
        );
    }

    let mut p_parts: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for (idx, start) in (0..probe_rows).step_by(batch_rows).enumerate() {
        let len = (probe_rows - start).min(batch_rows);
        let mut keys = Vec::with_capacity(len);
        let mut strs = Vec::with_capacity(len);
        let mut vals = Vec::with_capacity(len);
        let mut gates = Vec::with_capacity(len);
        for i in start..start + len {
            if i % 199 == 0 {
                keys.push(None);
                strs.push(None);
            } else {
                let k = ((i * 17) % (build_rows + 500)) as i64;
                keys.push(Some(k));
                strs.push(Some(format!("k-{k:06}")));
            }
            vals.push(Some(format!(
                "probe-payload-{i:08}-padding-bytes-abcdefghij"
            )));
            gates.push(i32::from(i != probe_rows / 2));
        }
        p_parts[idx % 4].push(
            RecordBatch::try_new(
                Arc::clone(&p_schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(StringArray::from(strs)),
                    Arc::new(StringArray::from(vals)),
                    Arc::new(Int32Array::from(gates)),
                ],
            )
            .unwrap(),
        );
    }

    (
        Arc::new(MemTable::try_new(b_schema, b_parts).unwrap()),
        Arc::new(MemTable::try_new(p_schema, p_parts).unwrap()),
    )
}

fn large_sort_table(num_batches: usize, rows_per_batch: usize) -> Arc<MemTable> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("sort_key", DataType::Int64, true),
        Field::new("payload", DataType::Utf8View, true),
        Field::new("gate", DataType::Int32, false),
    ]));
    let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for b in 0..num_batches {
        let mut ids = Vec::with_capacity(rows_per_batch);
        let mut keys = Vec::with_capacity(rows_per_batch);
        let mut payload = StringViewBuilder::with_capacity(rows_per_batch);
        let mut gate = Vec::with_capacity(rows_per_batch);
        for i in 0..rows_per_batch {
            let row = (b * rows_per_batch + i) as i64;
            ids.push(row);
            if row % 97 == 0 {
                keys.push(None);
            } else {
                keys.push(Some(
                    (row.wrapping_mul(6364136223846793005).wrapping_add(1)) % 50_000,
                ));
            }
            payload.append_value(format!(
                "sort-row-{row:08}-payload-padding-0123456789abcdef0123456789abcdef"
            ));
            gate.push(i32::from(!(b == num_batches / 2 && i == 10)));
        }
        partitions[b % 4].push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(payload.finish()),
                    Arc::new(Int32Array::from(gate)),
                ],
            )
            .unwrap(),
        );
    }
    Arc::new(MemTable::try_new(schema, partitions).unwrap())
}

fn write_parquet_table(
    dir: &TempDir,
    prefix: &str,
    batches: &[RecordBatch],
) -> Result<String> {
    let subdir = dir.path().join(prefix);
    std::fs::create_dir_all(&subdir)?;
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2048))
        .build();
    for (idx, chunk) in batches.chunks(4).enumerate() {
        let path = subdir.join(format!("part_{idx}.parquet"));
        let file = std::fs::File::create(&path)?;
        let mut writer =
            ArrowWriter::try_new(file, chunk[0].schema(), Some(props.clone()))?;
        for b in chunk {
            writer.write(b)?;
        }
        writer.close()?;
    }
    Ok(subdir.to_str().unwrap().to_owned())
}

fn assert_exact_order(
    expected: Vec<RecordBatch>,
    actual: Vec<RecordBatch>,
    label: &str,
) -> Result<()> {
    let exp_rows: Vec<RecordBatch> =
        expected.into_iter().filter(|b| b.num_rows() > 0).collect();
    let act_rows: Vec<RecordBatch> =
        actual.into_iter().filter(|b| b.num_rows() > 0).collect();
    if exp_rows.is_empty() {
        assert!(act_rows.is_empty(), "{label}: expected empty result");
        return Ok(());
    }
    let exp_cat = arrow::compute::concat_batches(&exp_rows[0].schema(), &exp_rows)?;
    let act_cat = arrow::compute::concat_batches(&act_rows[0].schema(), &act_rows)?;
    assert_eq!(act_cat, exp_cat, "{label}: exact order parity mismatch");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn group_by_spills_under_tight_greedy_pool_with_exact_parity() -> Result<()> {
    let table = high_cardinality_group_table(64, 1024);
    let stock = stock_context(4, 512);
    stock.register_table("t", table.clone())?;

    for sql in [
        "SELECT k_str, COUNT(*) AS n, SUM(v) AS s, MIN(v) AS lo, MAX(v) AS hi FROM t GROUP BY k_str",
        "SELECT k_int, COUNT(*) AS n, SUM(v) AS s, AVG(CAST(v AS DOUBLE)) AS a FROM t GROUP BY k_int",
        "SELECT k_int, COUNT(*) AS n FROM t GROUP BY k_int",
        "SELECT k_int, k_str, COUNT(DISTINCT v) AS dv, COUNT(DISTINCT gate) AS dg, SUM(v) AS s FROM t GROUP BY k_int, k_str",
        "SELECT k_str, SUM(v) AS s FROM t GROUP BY k_str ORDER BY s DESC NULLS LAST, k_str ASC",
    ] {
        let expected = stock.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("t", table.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .map_err(|e| {
                DataFusionError::Execution(format!("sql=`{sql}` failed: {e}"))
            })?;

        if sql.contains("ORDER BY") {
            assert_exact_order(expected, actual, sql)?;
        } else {
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected),
                "parity mismatch for sql=`{sql}`"
            );
        }
        assert!(
            spilled_bytes(plan.as_ref()) > 0,
            "expected spilled_bytes > 0 for sql=`{sql}` under 1 MiB pool"
        );
        assert!(
            spill_count(plan.as_ref()) > 0,
            "expected spill_count > 0 for sql=`{sql}` under 1 MiB pool"
        );
        drop(plan);
        assert_zero_leaks(&runtime, sql);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hash_join_spills_all_join_types_under_tight_greedy_pool() -> Result<()> {
    let (build_tbl, probe_tbl) = large_join_tables(24_000, 32_000, 1024);
    let stock = stock_context(4, 512);
    stock.register_table("b", build_tbl.clone())?;
    stock.register_table("p", probe_tbl.clone())?;

    let queries = [
        (
            "inner_int",
            "SELECT b.bk, b.bval, p.pval FROM b INNER JOIN p ON b.bk = p.pk",
        ),
        (
            "left_outer",
            "SELECT b.bk, b.bval, p.pval FROM b LEFT JOIN p ON b.bk = p.pk",
        ),
        (
            "right_outer",
            "SELECT b.bk, b.bval, p.pval FROM b RIGHT JOIN p ON b.bk = p.pk",
        ),
        (
            "full_outer",
            "SELECT b.bk, b.bval, p.pval FROM b FULL OUTER JOIN p ON b.bk = p.pk",
        ),
        (
            "left_semi",
            "SELECT b.bk, b.bval FROM b WHERE EXISTS (SELECT 1 FROM p WHERE b.bk = p.pk)",
        ),
        (
            "right_semi",
            "SELECT p.pk, p.pval FROM p WHERE EXISTS (SELECT 1 FROM b WHERE b.bk = p.pk)",
        ),
        (
            "left_anti",
            "SELECT b.bk, b.bval FROM b WHERE NOT EXISTS (SELECT 1 FROM p WHERE b.bk = p.pk)",
        ),
        (
            "right_anti",
            "SELECT p.pk, p.pval FROM p WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.bk = p.pk)",
        ),
        (
            "multi_key_string_and_residual_filter",
            "SELECT b.bk, b.bstr, p.pval FROM b LEFT JOIN p ON b.bk = p.pk AND b.bstr = p.pstr AND b.bval < p.pval",
        ),
        (
            "null_equals_null_is_not_distinct_from",
            "SELECT b.bk, b.bval, p.pval FROM b INNER JOIN p ON b.bk IS NOT DISTINCT FROM p.pk AND b.bstr IS NOT DISTINCT FROM p.pstr",
        ),
    ];

    for (label, sql) in queries {
        let expected = stock.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1536 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("b", build_tbl.clone())?;
        ctx.register_table("p", probe_tbl.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec", "{label}");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;

        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "{label}: parity mismatch"
        );
        assert!(
            spilled_bytes(plan.as_ref()) > 0,
            "{label}: expected spilled_bytes > 0 under tight pool"
        );
        drop(plan);
        assert_zero_leaks(&runtime, label);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hash_join_extreme_duplicate_skew_bnl_fallback_all_join_types() -> Result<()> {
    // 100% single-key duplicate skew on build (12,000 rows all k = 42 with wide
    // payloads -> ~2 MiB in ONE radix partition that no hash split can divide!)
    // joined against probe rows containing both k = 42 (matches) and k = 99
    // (unmatched) under a 1 MiB GreedyMemoryPool -> forces the BNL sub-pass!
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Int64, false),
        Field::new("btag", DataType::Int32, false),
        Field::new("bpad", DataType::Utf8, false),
    ]));
    let p_schema = Arc::new(Schema::new(vec![
        Field::new("pk", DataType::Int64, false),
        Field::new("ptag", DataType::Int32, false),
        Field::new("ppad", DataType::Utf8, false),
    ]));

    let pad = "x".repeat(160);
    let mut b_batches = Vec::new();
    for b in 0..24usize {
        let len = 500;
        let ks = vec![42i64; len];
        let tags: Vec<i32> = (0..len).map(|i| ((b * len + i) % 10) as i32).collect();
        let pads = vec![pad.as_str(); len];
        b_batches.push(RecordBatch::try_new(
            Arc::clone(&b_schema),
            vec![
                Arc::new(Int64Array::from(ks)),
                Arc::new(Int32Array::from(tags)),
                Arc::new(StringArray::from(pads)),
            ],
        )?);
    }
    // Make `p` have more rows than `b` (15,000 > 12,000) so `choose_build_side`
    // places the 100% single-key-skewed `b` (~2.2 MiB) on the BUILD side,
    // forcing `b` into one overflowing radix partition -> BNL hash sub-pass!
    let mut p_batches = Vec::new();
    for pb in 0..30usize {
        let len = 500;
        let mut pk = Vec::with_capacity(len);
        let mut ptag = Vec::with_capacity(len);
        let mut ppad = Vec::with_capacity(len);
        for i in 0..len {
            let id = pb * len + i;
            if id < 6 {
                // 6 probe rows matching key 42 with varied tags (some matching,
                // some filtered out by `b.btag = p.ptag`)
                pk.push(42i64);
                ptag.push(if id % 2 == 0 { (id % 10) as i32 } else { 99 });
            } else {
                pk.push(10_000 + id as i64);
                ptag.push((id % 10) as i32);
            }
            ppad.push(format!("p-{id:05}"));
        }
        p_batches.push(RecordBatch::try_new(
            Arc::clone(&p_schema),
            vec![
                Arc::new(Int64Array::from(pk)),
                Arc::new(Int32Array::from(ptag)),
                Arc::new(StringArray::from(ppad)),
            ],
        )?);
    }

    let b_tbl = Arc::new(MemTable::try_new(b_schema, vec![b_batches])?);
    let p_tbl = Arc::new(MemTable::try_new(p_schema, vec![p_batches])?);

    let stock = stock_context(2, 256);
    stock.register_table("b", b_tbl.clone())?;
    stock.register_table("p", p_tbl.clone())?;

    for (label, sql) in [
        (
            "skew_inner_count",
            "SELECT COUNT(b.bpad), MIN(b.bpad) FROM b INNER JOIN p ON b.bk = p.pk",
        ),
        (
            "skew_inner_filtered",
            "SELECT COUNT(b.bpad), MIN(b.bpad) FROM b INNER JOIN p ON b.bk = p.pk AND b.btag = p.ptag",
        ),
        (
            "skew_right_outer",
            "SELECT p.pk, p.ptag, COUNT(b.bpad) FROM b RIGHT JOIN p ON b.bk = p.pk AND b.btag = p.ptag GROUP BY p.pk, p.ptag",
        ),
        (
            "skew_left_outer",
            "SELECT b.btag, COUNT(p.ptag), MIN(b.bpad) FROM b LEFT JOIN p ON b.bk = p.pk AND b.btag = p.ptag GROUP BY b.btag",
        ),
        (
            "skew_right_semi",
            "SELECT p.pk, p.ptag, p.ppad FROM p WHERE EXISTS (SELECT 1 FROM b WHERE b.bk = p.pk AND b.btag = p.ptag AND b.bpad <> p.ppad)",
        ),
        (
            "skew_right_anti",
            "SELECT p.pk, p.ptag, p.ppad FROM p WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.bk = p.pk AND b.btag = p.ptag AND b.bpad <> p.ppad)",
        ),
        (
            "skew_left_semi_count",
            "SELECT COUNT(b.bpad), MIN(b.bpad) FROM b WHERE EXISTS (SELECT 1 FROM p WHERE b.bk = p.pk AND b.btag = p.ptag)",
        ),
        (
            "skew_left_anti_count",
            "SELECT COUNT(b.bpad), MIN(b.bpad) FROM b WHERE NOT EXISTS (SELECT 1 FROM p WHERE b.bk = p.pk AND b.btag = p.ptag)",
        ),
    ] {
        let expected = stock.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(2, 256, Arc::clone(&runtime))?;
        ctx.register_table("b", b_tbl.clone())?;
        ctx.register_table("p", p_tbl.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec", "{label}");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;

        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "{label}: BNL skew parity mismatch"
        );
        assert!(
            spilled_bytes(plan.as_ref()) > 0,
            "{label}: expected spilled_bytes > 0"
        );
        drop(plan);
        assert_zero_leaks(&runtime, label);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sort_spills_under_tight_greedy_pool_with_exact_parity() -> Result<()> {
    let table = large_sort_table(64, 1024);
    let stock = stock_context(4, 512);
    stock.register_table("s", table.clone())?;

    for (label, sql) in [
        (
            "asc_nulls_last",
            "SELECT id, sort_key, payload FROM s ORDER BY sort_key ASC NULLS LAST, id ASC",
        ),
        (
            "desc_nulls_first_offset_limit",
            "SELECT id, sort_key, payload FROM s ORDER BY sort_key DESC NULLS FIRST, id DESC LIMIT 1000 OFFSET 500",
        ),
    ] {
        let expected = stock.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1536 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("s", table.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec", "{label}");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;

        assert_exact_order(expected, actual, label)?;
        if !sql.contains("LIMIT") {
            assert!(
                spilled_bytes(plan.as_ref()) > 0,
                "{label}: expected spilled_bytes > 0 under 1.5 MiB pool"
            );
            assert!(
                spill_count(plan.as_ref()) > 0,
                "{label}: expected spill_count > 0 under 1.5 MiB pool"
            );
        }
        drop(plan);
        wait_zero_leaks(&runtime, label).await;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parquet_spilled_join_group_by_and_float_nan_parity() -> Result<()> {
    // Verify Parquet native scan + dynamic join filter soundness over M ∪ S
    // (bounds cover all spilled partitions without dropping valid rows!), plus
    // float -0.0 / NaN GROUP BY and HashJoin parity under spill.
    let dir = TempDir::new()?;
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Float64, true),
        Field::new("bval", DataType::Utf8, false),
    ]));
    let p_schema = Arc::new(Schema::new(vec![
        Field::new("pk", DataType::Float64, true),
        Field::new("pval", DataType::Utf8, false),
    ]));

    let mut b_batches = Vec::new();
    let mut p_batches = Vec::new();
    for chunk in 0..20usize {
        let len = 1024;
        let mut bk = Vec::with_capacity(len);
        let mut bv = Vec::with_capacity(len);
        let mut pk = Vec::with_capacity(len);
        let mut pv = Vec::with_capacity(len);
        for i in 0..len {
            let id = chunk * len + i;
            let f = match id % 200 {
                0 => Some(f64::NAN),
                1 => Some(-0.0),
                2 => Some(0.0),
                3 => None,
                _ => Some((id % 12_000) as f64),
            };
            bk.push(f);
            bv.push(format!("bp-{id:08}-padding-0123456789abcdef"));
            pk.push(if id % 5 == 0 {
                Some((id + 50_000) as f64)
            } else {
                f
            });
            pv.push(format!("pp-{id:08}-padding-fedcba9876543210"));
        }
        b_batches.push(RecordBatch::try_new(
            Arc::clone(&b_schema),
            vec![
                Arc::new(Float64Array::from(bk)),
                Arc::new(StringArray::from(bv)),
            ],
        )?);
        p_batches.push(RecordBatch::try_new(
            Arc::clone(&p_schema),
            vec![
                Arc::new(Float64Array::from(pk)),
                Arc::new(StringArray::from(pv)),
            ],
        )?);
    }

    let b_path = write_parquet_table(&dir, "b", &b_batches)?;
    let p_path = write_parquet_table(&dir, "p", &p_batches)?;

    let stock = stock_context(4, 512);
    stock
        .register_parquet("b", &b_path, ParquetReadOptions::default())
        .await?;
    stock
        .register_parquet("p", &p_path, ParquetReadOptions::default())
        .await?;

    let sql =
        "SELECT b.bk, COUNT(*) AS cnt FROM b INNER JOIN p ON b.bk = p.pk GROUP BY b.bk";
    let expected = stock.sql(sql).await?.collect().await?;

    for pushdown in [false, true] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(2 * 1024 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context_with_pushdown(4, 512, Arc::clone(&runtime), pushdown)?;
        ctx.register_parquet("b", &b_path, ParquetReadOptions::default())
            .await?;
        ctx.register_parquet("p", &p_path, ParquetReadOptions::default())
            .await?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "parquet pushdown={pushdown}: parity mismatch"
        );
        assert!(
            spilled_bytes(plan.as_ref()) > 0,
            "parquet pushdown={pushdown}: expected spilled_bytes > 0"
        );
        drop(plan);
        assert_zero_leaks(&runtime, &format!("parquet_pushdown_{pushdown}"));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn spill_cleanup_on_early_drop_worker_error_disk_quota_and_fair_pool() -> Result<()>
{
    let group_tbl = high_cardinality_group_table(48, 1024);
    let (join_b, join_p) = large_join_tables(20_000, 24_000, 1024);
    let sort_tbl = large_sort_table(48, 1024);

    // 1. Early stream drop after 1 batch leaves zero leaked files or bytes
    //    across GROUP BY, HashJoin, and Sort!
    for (label, sql, budget) in [
        (
            "early_drop_group_by",
            "SELECT k_str, COUNT(*) AS n, SUM(v) AS s FROM g GROUP BY k_str",
            1024 * 1024,
        ),
        (
            "early_drop_hash_join",
            "SELECT b.bk, b.bval, p.pval FROM b INNER JOIN p ON b.bk = p.pk",
            1536 * 1024,
        ),
        (
            "early_drop_sort",
            "SELECT id, sort_key, payload FROM s ORDER BY sort_key ASC, id ASC",
            1536 * 1024,
        ),
    ] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(budget));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 128, Arc::clone(&runtime))?;
        ctx.register_table("g", group_tbl.clone())?;
        ctx.register_table("b", join_b.clone())?;
        ctx.register_table("p", join_p.clone())?;
        ctx.register_table("s", sort_tbl.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
        let first = timeout(Duration::from_secs(15), stream.next())
            .await
            .expect("stream timed out")
            .expect("expected first batch")?;
        assert!(first.num_rows() > 0, "{label}");
        drop(stream);
        drop(plan);
        wait_zero_leaks(&runtime, label).await;
    }

    // 2. Worker runtime error (division by zero) mid-spill cleans up all spill
    //    files and MemoryPool bytes across GROUP BY, HashJoin, and Sort!
    for (label, sql, budget) in [
        (
            "worker_err_group_by",
            "SELECT k_str, SUM(v) FROM g WHERE 100 / gate > 1 GROUP BY k_str",
            1024 * 1024,
        ),
        (
            "worker_err_hash_join_probe",
            "SELECT b.bk, p.pval FROM b INNER JOIN p ON b.bk = p.pk WHERE 100 / p.pgate > 1",
            1536 * 1024,
        ),
        (
            "worker_err_sort",
            "SELECT id, sort_key, payload FROM s WHERE 100 / gate > 1 ORDER BY sort_key ASC",
            1536 * 1024,
        ),
    ] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(budget));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 256, Arc::clone(&runtime))?;
        ctx.register_table("g", group_tbl.clone())?;
        ctx.register_table("b", join_b.clone())?;
        ctx.register_table("p", join_p.clone())?;
        ctx.register_table("s", sort_tbl.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let err = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap_err();
        assert!(
            !matches!(err, DataFusionError::ResourcesExhausted(_)),
            "{label}: expected division-by-zero error, got: {err}"
        );
        drop(plan);
        wait_zero_leaks(&runtime, label).await;
    }

    // 3. Disk quota exhaustion returns error and leaks 0 files / 0 bytes across
    //    GROUP BY, HashJoin, and Sort!
    for (label, sql, budget) in [
        (
            "disk_quota_group_by",
            "SELECT k_str, COUNT(*) AS n, SUM(v) AS s FROM g GROUP BY k_str",
            1024 * 1024,
        ),
        (
            "disk_quota_hash_join",
            "SELECT b.bk, b.bval, p.pval FROM b INNER JOIN p ON b.bk = p.pk",
            1536 * 1024,
        ),
        (
            "disk_quota_sort",
            "SELECT id, sort_key, payload FROM s ORDER BY sort_key ASC, id ASC",
            1536 * 1024,
        ),
    ] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(budget));
        let runtime = runtime_with_pool(Arc::clone(&pool), Some(32 * 1024))?;
        let ctx = morsel_context(4, 256, Arc::clone(&runtime))?;
        ctx.register_table("g", group_tbl.clone())?;
        ctx.register_table("b", join_b.clone())?;
        ctx.register_table("p", join_p.clone())?;
        ctx.register_table("s", sort_tbl.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let err = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                DataFusionError::ResourcesExhausted(_) | DataFusionError::IoError(_)
            ) && err.to_string().contains("max_temp_directory_size"),
            "{label}: expected disk quota error, got: {err}"
        );
        drop(plan);
        wait_zero_leaks(&runtime, label).await;
    }

    // 4. FairSpillPool parity across GROUP BY, HashJoin, and Sort!
    {
        let stock = stock_context(4, 512);
        stock.register_table("g", group_tbl.clone())?;
        stock.register_table("b", join_b.clone())?;
        stock.register_table("p", join_p.clone())?;
        stock.register_table("s", sort_tbl.clone())?;

        for (label, sql) in [
            (
                "fair_group_by",
                "SELECT k_str, COUNT(*) AS n, SUM(v) AS s FROM g GROUP BY k_str",
            ),
            (
                "fair_hash_join",
                "SELECT b.bk, b.bval, p.pval FROM b LEFT JOIN p ON b.bk = p.pk",
            ),
            (
                "fair_sort",
                "SELECT id, sort_key, payload FROM s ORDER BY sort_key ASC, id ASC",
            ),
        ] {
            let expected = stock.sql(sql).await?.collect().await?;
            let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(2 * 1024 * 1024));
            let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
            let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
            ctx.register_table("g", group_tbl.clone())?;
            ctx.register_table("b", join_b.clone())?;
            ctx.register_table("p", join_p.clone())?;
            ctx.register_table("s", sort_tbl.clone())?;

            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
            if sql.contains("ORDER BY") {
                assert_exact_order(expected, actual, label)?;
            } else {
                assert_eq!(
                    batches_to_sort_string(&actual),
                    batches_to_sort_string(&expected),
                    "{label}"
                );
            }
            drop(plan);
            assert_zero_leaks(&runtime, label);
        }
    }

    Ok(())
}

/// Fault-injecting MemoryPool that allows the first `MorselSpillViability`
/// probe to succeed and rejects any subsequent viability probes (simulating a
/// pool transiently filled by a concurrent query at sort/TopN finalize time).
#[derive(Debug)]
struct RejectSecondViabilityPool {
    inner: GreedyMemoryPool,
    viability_calls: AtomicUsize,
}

impl RejectSecondViabilityPool {
    fn new(pool_size: usize) -> Self {
        Self {
            inner: GreedyMemoryPool::new(pool_size),
            viability_calls: AtomicUsize::new(0),
        }
    }
}

impl std::fmt::Display for RejectSecondViabilityPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RejectSecondViabilityPool({})", self.inner)
    }
}

impl MemoryPool for RejectSecondViabilityPool {
    fn name(&self) -> &str {
        "RejectSecondViabilityPool"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        if reservation.consumer().name() == "MorselSpillViability" {
            let n = self.viability_calls.fetch_add(1, Ordering::SeqCst);
            if n >= 1 {
                return Err(DataFusionError::ResourcesExhausted(
                    "injected transient pool exhaustion during second viability probe"
                        .into(),
                ));
            }
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sort_and_topn_spilled_finalize_reuses_spill_context_under_transient_pool_pressure()
-> Result<()> {
    let table = large_sort_table(48, 1024);
    let stock = stock_context(4, 512);
    stock.register_table("s", table.clone())?;

    for (label, sql) in [
        (
            "sort_transient_viability_reject",
            "SELECT id, sort_key, payload FROM s ORDER BY sort_key ASC NULLS LAST, id ASC",
        ),
        (
            "topn_large_limit_transient_viability_reject",
            "SELECT id, sort_key, payload FROM s ORDER BY sort_key DESC NULLS FIRST, id ASC LIMIT 30000",
        ),
    ] {
        let expected = stock.sql(sql).await?.collect().await?;
        let pool: Arc<dyn MemoryPool> =
            Arc::new(RejectSecondViabilityPool::new(1536 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("s", table.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec", "{label}");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;

        assert_exact_order(expected, actual, label)?;
        assert!(
            spilled_bytes(plan.as_ref()) > 0,
            "{label}: expected spilled_bytes > 0"
        );
        drop(plan);
        wait_zero_leaks(&runtime, label).await;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fair_spill_pool_radix_join_into_group_by_and_sort_does_not_starve_downstream()
-> Result<()> {
    // 20,000 build rows (> RADIX_JOIN_THRESHOLD_ROWS = 16,384) guarantees a
    // 64-way radix HashJoin feeding downstream spillable sinks under FairSpillPool.
    let (build_tbl, probe_tbl) = large_join_tables(20_000, 24_000, 1024);
    let stock = stock_context(4, 512);
    stock.register_table("b", build_tbl.clone())?;
    stock.register_table("p", probe_tbl.clone())?;

    // 1. Join -> GROUP BY at generous 32 MiB FairSpillPool must NOT spill
    //    (proves resident radix partitions do not register 64 spillable consumers).
    {
        let sql = "SELECT b.bk, COUNT(*) AS cnt, MIN(p.pval) AS lo FROM b INNER JOIN p ON b.bk = p.pk GROUP BY b.bk";
        let expected = stock.sql(sql).await?.collect().await?;
        let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(32 * 1024 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("b", build_tbl.clone())?;
        ctx.register_table("p", probe_tbl.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "FairSpillPool 32 MiB Join -> GROUP BY parity mismatch"
        );
        assert_eq!(
            spilled_bytes(plan.as_ref()),
            0,
            "FairSpillPool 32 MiB Join -> GROUP BY should not spill when resident partitions use one non-spillable consumer"
        );
        drop(plan);
        assert_zero_leaks(&runtime, "fair_32mb_join_group_by");
    }

    // 2. Join -> ORDER BY at tight 4 MiB FairSpillPool succeeds cleanly via
    //    spilling without starving downstream Sort.
    {
        let sql = "SELECT b.bk, p.pk, p.pval FROM b INNER JOIN p ON b.bk = p.pk ORDER BY b.bk ASC, p.pval ASC";
        let expected = stock.sql(sql).await?.collect().await?;
        let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(4 * 1024 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("b", build_tbl.clone())?;
        ctx.register_table("p", probe_tbl.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert_exact_order(expected, actual, "fair_4mb_join_order_by")?;
        drop(plan);
        assert_zero_leaks(&runtime, "fair_4mb_join_order_by");
    }
    Ok(())
}

/// Exact inverse of `hash_join::mix_u64` (SplitMix64 finalizer) used to craft
/// deterministic `Int64` keys that collide on chosen radix hash bit windows.
fn unmix_u64(mut x: u64) -> u64 {
    x ^= (x >> 31) ^ (x >> 62);
    x = x.wrapping_mul(0x3196_42b2_d24d_8ec3);
    x ^= (x >> 27) ^ (x >> 54);
    x = x.wrapping_mul(0x96de_1b17_3f11_9089);
    x ^= (x >> 30) ^ (x >> 60);
    x
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn group_by_recursive_radix_split_and_sorted_spill_fallback_parity() -> Result<()> {
    // 1. Int64 keys colliding on L0 [51..57) (varied L1 [47..51) -> triggers
    //    recursive radix split) AND keys colliding on all radix levels [43..57)
    //    (non-progressing skew -> triggers `finalize_sorted_spill_fallback` and
    //    `rows_equal_on_keys` across duplicate boundary rows!).
    let int_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]));

    let make_colliding_int_table = |mask_all_levels: bool| -> Arc<MemTable> {
        let num_keys = 16_000usize;
        let mut keys_pool = Vec::with_capacity(num_keys);
        for i in 0..num_keys {
            let desired_hash = if mask_all_levels {
                // Zero bits [43..57) so L0, L1, and L2 all map to bucket 0.
                ((i as u64) & ((1u64 << 43) - 1)) | (((i as u64) & 0x7f) << 57)
            } else {
                // Zero L0 bits [51..57), spread L1 bits [47..51) uniformly.
                let l1 = ((i as u64) & 0xf) << 47;
                l1 | ((i as u64) & ((1u64 << 43) - 1)) | (((i as u64) & 0x7f) << 57)
            };
            keys_pool.push(unmix_u64(desired_hash) as i64);
        }

        let mut parts: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
        for rep in 0..2usize {
            for (chunk_idx, chunk) in keys_pool.chunks(1024).enumerate() {
                let mut ks: Vec<Option<i64>> = Vec::with_capacity(chunk.len() + 2);
                let mut vs: Vec<Option<i64>> = Vec::with_capacity(chunk.len() + 2);
                ks.push(None);
                vs.push(Some(7));
                for (j, &k) in chunk.iter().enumerate() {
                    ks.push(Some(k));
                    vs.push(Some(((j + rep * 13) % 97) as i64));
                }
                parts[(rep * 16 + chunk_idx) % 4].push(
                    RecordBatch::try_new(
                        Arc::clone(&int_schema),
                        vec![
                            Arc::new(Int64Array::from(ks)),
                            Arc::new(Int64Array::from(vs)),
                        ],
                    )
                    .unwrap(),
                );
            }
        }
        Arc::new(MemTable::try_new(Arc::clone(&int_schema), parts).unwrap())
    };

    for (mode_label, mask_all_levels) in [
        ("int_recursive_l1_split", false),
        ("int_sorted_spill_fallback", true),
    ] {
        let tbl = make_colliding_int_table(mask_all_levels);
        let stock = stock_context(4, 256);
        stock.register_table("t", tbl.clone())?;

        for (q_label, sql) in [
            ("group_count", "SELECT k, COUNT(*) AS n FROM t GROUP BY k"),
            (
                "direct_grouped",
                "SELECT k, COUNT(*) AS n, SUM(v) AS s, MIN(v) AS lo, MAX(v) AS hi, AVG(CAST(v AS DOUBLE)) AS a FROM t GROUP BY k",
            ),
        ] {
            let expected = stock.sql(sql).await?.collect().await?;
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(512 * 1024));
            let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
            let ctx = morsel_context(4, 256, Arc::clone(&runtime))?;
            ctx.register_table("t", tbl.clone())?;

            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            assert_eq!(plan.name(), "MorselExec");
            let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
            assert_eq!(
                batches_to_sort_string(&actual),
                batches_to_sort_string(&expected),
                "{mode_label}/{q_label}: parity mismatch"
            );
            assert!(
                spilled_bytes(plan.as_ref()) > 0,
                "{mode_label}/{q_label}: expected spilled_bytes > 0"
            );
            drop(plan);
            assert_zero_leaks(&runtime, &format!("{mode_label}_{q_label}"));
        }
    }

    // 2. Multi-column `Grouped` keys colliding on `GROUP_HASH_SEED` bits [47..57)
    //    (L0 + L1 identical -> forces non-progressing split -> `finalize_sorted_spill_fallback`
    //    and multi-column `rows_equal_on_keys` including NULLs and duplicate rows!).
    const GROUP_HASH_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
    let rs = RandomState::with_seed(GROUP_HASH_SEED);
    let target_keys = 3_500usize;
    let mut colliding_ints: Vec<i64> = Vec::with_capacity(target_keys);
    let mut colliding_strs: Vec<String> = Vec::with_capacity(target_keys);
    let pad = "p".repeat(120);
    let mut base = 0i64;
    let mut hashes = vec![0u64; 8192];
    while colliding_ints.len() < target_keys {
        let cand_ints: Vec<i64> = (base..base + 8192).collect();
        let cand_strs: Vec<String> = cand_ints
            .iter()
            .map(|k| format!("ck-{k:08}-{pad}"))
            .collect();
        let a0: ArrayRef = Arc::new(Int64Array::from(cand_ints.clone()));
        let a1: ArrayRef = Arc::new(StringArray::from(cand_strs.clone()));
        hashes.fill(0);
        create_hashes(&[a0, a1], &rs, &mut hashes)?;
        for (i, &h) in hashes.iter().enumerate() {
            // Collide on bits [47..57) (both L0 [51..57) and L1 [47..51) == 0).
            if (h >> 47).trailing_zeros() >= 10 {
                colliding_ints.push(cand_ints[i]);
                colliding_strs.push(cand_strs[i].clone());
                if colliding_ints.len() >= target_keys {
                    break;
                }
            }
        }
        base += 8192;
    }

    let grp_schema = Arc::new(Schema::new(vec![
        Field::new("k_int", DataType::Int64, true),
        Field::new("k_str", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let mut parts: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for rep in 0..2usize {
        for (chunk_idx, start) in (0..colliding_ints.len()).step_by(512).enumerate() {
            let end = (start + 512).min(colliding_ints.len());
            let mut ki: Vec<Option<i64>> = Vec::with_capacity(end - start + 1);
            let mut ks: Vec<Option<&str>> = Vec::with_capacity(end - start + 1);
            let mut vs: Vec<Option<i64>> = Vec::with_capacity(end - start + 1);
            ki.push(None);
            ks.push(None);
            vs.push(Some(3));
            for i in start..end {
                ki.push(Some(colliding_ints[i]));
                ks.push(Some(colliding_strs[i].as_str()));
                vs.push(Some(((i + rep * 11) % 53) as i64));
            }
            parts[(rep * 8 + chunk_idx) % 4].push(RecordBatch::try_new(
                Arc::clone(&grp_schema),
                vec![
                    Arc::new(Int64Array::from(ki)),
                    Arc::new(StringArray::from(ks)),
                    Arc::new(Int64Array::from(vs)),
                ],
            )?);
        }
    }
    let grp_tbl = Arc::new(MemTable::try_new(grp_schema, parts)?);
    let stock = stock_context(4, 128);
    stock.register_table("g", grp_tbl.clone())?;
    let sql =
        "SELECT k_int, k_str, COUNT(*) AS n, SUM(v) AS s FROM g GROUP BY k_int, k_str";
    let expected = stock.sql(sql).await?.collect().await?;

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(512 * 1024));
    let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
    let ctx = morsel_context(4, 128, Arc::clone(&runtime))?;
    ctx.register_table("g", grp_tbl)?;

    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    assert_eq!(plan.name(), "MorselExec");
    let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected),
        "grouped multi-column sorted fallback parity mismatch"
    );
    assert!(
        spilled_bytes(plan.as_ref()) > 0,
        "grouped multi-column sorted fallback: expected spilled_bytes > 0"
    );
    drop(plan);
    assert_zero_leaks(&runtime, "grouped_sorted_fallback");
    Ok(())
}

/// Spawn a lightweight sampler that tracks the peak `active_files_count` on
/// `runtime.disk_manager` while a query executes.
async fn collect_with_peak_active_files(
    plan: Arc<dyn ExecutionPlan>,
    ctx: &SessionContext,
    runtime: &Arc<RuntimeEnv>,
) -> Result<(Vec<RecordBatch>, usize)> {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicUsize::new(0));
    let stop_bg = Arc::clone(&stop);
    let peak_bg = Arc::clone(&peak);
    let rt_bg = Arc::clone(runtime);
    let sampler = SpawnedTask::spawn(async move {
        while !stop_bg.load(Ordering::Relaxed) {
            let cur = rt_bg.disk_manager.spilling_progress().active_files_count;
            peak_bg.fetch_max(cur, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_micros(500)).await;
        }
    });
    let res = collect(plan, ctx.task_ctx()).await;
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join().await;
    Ok((res?, peak.load(Ordering::Relaxed)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn peak_active_spill_files_stay_bounded_across_group_by_hash_join_and_sort()
-> Result<()> {
    // 1. High-cardinality GROUP BY with many spill flushes: peak active spill
    //    files must scale with partition count (<= 64 L0), NOT partitions x workers x flushes.
    {
        let table = high_cardinality_group_table(96, 1024);
        let stock = stock_context(4, 512);
        stock.register_table("t", table.clone())?;
        let sql = "SELECT k_str, COUNT(*) AS n, SUM(v) AS s FROM t GROUP BY k_str";
        let expected = stock.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("t", table)?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let (actual, peak_files) =
            collect_with_peak_active_files(Arc::clone(&plan), &ctx, &runtime).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
        );
        let total_files = spill_count(plan.as_ref());
        assert!(
            spilled_bytes(plan.as_ref()) > 0,
            "expected GROUP BY to spill"
        );
        assert!(
            peak_files <= 128 && total_files <= 128,
            "GROUP BY open spill files unbounded: peak_active={peak_files}, total_created={total_files}"
        );
        drop(plan);
        assert_zero_leaks(&runtime, "bounded_files_group_by");
    }

    // 2. Spilled HashJoin (build + probe): peak active spill files bounded by
    //    O(partitions) (`<= 64` build + `<= 64` probe + `<= 1` null-key).
    {
        let (build_tbl, probe_tbl) = large_join_tables(28_000, 36_000, 1024);
        let stock = stock_context(4, 512);
        stock.register_table("b", build_tbl.clone())?;
        stock.register_table("p", probe_tbl.clone())?;
        let sql = "SELECT b.bk, b.bval, p.pval FROM b LEFT JOIN p ON b.bk = p.pk";
        let expected = stock.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1536 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("b", build_tbl)?;
        ctx.register_table("p", probe_tbl)?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let (actual, peak_files) =
            collect_with_peak_active_files(Arc::clone(&plan), &ctx, &runtime).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
        );
        let total_files = spill_count(plan.as_ref());
        assert!(
            spilled_bytes(plan.as_ref()) > 0,
            "expected HashJoin to spill"
        );
        assert!(
            peak_files <= 160 && total_files <= 160,
            "HashJoin open spill files unbounded: peak_active={peak_files}, total_created={total_files}"
        );
        drop(plan);
        assert_zero_leaks(&runtime, "bounded_files_hash_join");
    }

    // 3. Spilled Sort across many small batches: periodic per-worker compaction
    //    bounds peak active spill files to `O(workers)`.
    {
        let table = large_sort_table(128, 1024);
        let stock = stock_context(4, 512);
        stock.register_table("s", table.clone())?;
        let sql = "SELECT id, sort_key, payload FROM s ORDER BY sort_key ASC NULLS LAST, id ASC";
        let expected = stock.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1536 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("s", table)?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let (actual, peak_files) =
            collect_with_peak_active_files(Arc::clone(&plan), &ctx, &runtime).await?;
        assert_exact_order(expected, actual, "bounded_files_sort")?;
        assert!(spilled_bytes(plan.as_ref()) > 0, "expected Sort to spill");
        assert!(
            peak_files <= 128,
            "Sort peak active spill files unbounded: peak_active={peak_files}"
        );
        drop(plan);
        wait_zero_leaks(&runtime, "bounded_files_sort").await;
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn topn_spill_truncates_runs_to_limit_instead_of_spilling_full_input() -> Result<()>
{
    // 128 batches x 1024 rows (~11 MiB raw payload) -> verifies TopN neither
    // falsely spills on small `LIMIT 10` nor spills the full input when a
    // larger `LIMIT` does trigger spill fallback.
    let table = large_sort_table(128, 1024);
    let stock = stock_context(4, 512);
    stock.register_table("s", table.clone())?;

    // 1. `LIMIT 10` under a 2 MiB pool must NOT spill the 11 MiB input.
    {
        let sql = "SELECT id, sort_key, payload FROM s ORDER BY sort_key DESC NULLS LAST, id DESC LIMIT 10";
        let expected = stock.sql(sql).await?.collect().await?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(2 * 1024 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("s", table.clone())?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert_exact_order(expected, actual, "topn_limit10_2mb")?;
        assert_eq!(
            spilled_bytes(plan.as_ref()),
            0,
            "LIMIT 10 under 2 MiB pool should not spill (spilled {} bytes)",
            spilled_bytes(plan.as_ref())
        );
        drop(plan);
        assert_zero_leaks(&runtime, "topn_limit10_2mb");
    }

    // 2. `LIMIT 150 OFFSET 50` (limit = 200 << 4096-row batches) over ~11 MiB
    //    input under a tight 512 KiB pool: even when TopN spills, every run is
    //    truncated to `<= 200` rows so spilled bytes stay tiny (< 2 MiB vs ~11 MiB input).
    {
        let wide_batches = large_sort_table(32, 4096);
        let stock_wide = stock_context(4, 512);
        stock_wide.register_table("sw", wide_batches.clone())?;
        let sql = "SELECT id, sort_key, payload FROM sw ORDER BY sort_key DESC NULLS FIRST, id DESC LIMIT 150 OFFSET 50";
        let expected = stock_wide.sql(sql).await?.collect().await?;

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(512 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("sw", wide_batches)?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let (actual, peak_files) =
            collect_with_peak_active_files(Arc::clone(&plan), &ctx, &runtime).await?;
        assert_exact_order(expected, actual, "topn_limit150_512kb")?;
        let spilled = spilled_bytes(plan.as_ref());
        assert!(
            spilled < 2 * 1024 * 1024,
            "TopN LIMIT 150 OFFSET 50 spilled {spilled} bytes (expected < 2 MiB on 11 MiB input)"
        );
        assert!(
            peak_files <= 128,
            "TopN peak active spill files unbounded: {peak_files}"
        );
        drop(plan);
        wait_zero_leaks(&runtime, "topn_limit150_512kb").await;
    }

    // 3. Large `LIMIT 2000 OFFSET 500` under a tight 768 KiB pool genuinely
    //    spills, keeps total spilled bytes well below full-input sort (~25 MiB),
    //    and keeps peak active spill files bounded.
    {
        let sql = "SELECT id, sort_key, payload FROM s ORDER BY sort_key DESC NULLS FIRST, id DESC LIMIT 2000 OFFSET 500";
        let expected = stock.sql(sql).await?.collect().await?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(768 * 1024));
        let runtime = runtime_with_pool(Arc::clone(&pool), None)?;
        let ctx = morsel_context(4, 512, Arc::clone(&runtime))?;
        ctx.register_table("s", table)?;

        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let (actual, peak_files) =
            collect_with_peak_active_files(Arc::clone(&plan), &ctx, &runtime).await?;
        assert_exact_order(expected, actual, "topn_limit2000_768kb")?;
        let spilled = spilled_bytes(plan.as_ref());
        assert!(
            spilled > 0,
            "expected large TopN under 768 KiB pool to spill"
        );
        assert!(
            spilled < 12 * 1024 * 1024,
            "TopN LIMIT 2000 OFFSET 500 spilled {spilled} bytes (expected < 12 MiB)"
        );
        assert!(
            peak_files <= 128,
            "TopN peak active spill files unbounded: {peak_files}"
        );
        drop(plan);
        wait_zero_leaks(&runtime, "topn_limit2000_768kb").await;
    }

    Ok(())
}
