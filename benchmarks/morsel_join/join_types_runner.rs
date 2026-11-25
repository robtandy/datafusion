// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Paired release benchmark comparing stock DataFusion (`morsel_mode=off`)
//! vs native morsel joins (`morsel_mode=required`) across SF=1 and SF=5
//! TPC-H-shaped workloads covering Semi, Anti, Null-Aware NOT IN,
//! Left/Right/Full Outer (with residual Join::filter and child pipeline),
//! CrossJoin, and Inner baseline (#190).

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::config::MorselMode;
use datafusion::morsel::MorselMetrics;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const BATCH_SIZE: usize = 8192;

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn customer_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int64, true),
        Field::new("c_mktsegment", DataType::Int32, false),
        Field::new("c_acctbal", DataType::Int64, false),
    ]))
}

fn orders_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int64, false),
        Field::new("o_custkey", DataType::Int64, true),
        Field::new("o_suppkey", DataType::Int64, true),
        Field::new("o_totalprice", DataType::Int64, false),
        Field::new("o_comment", DataType::Utf8, false),
    ]))
}

fn supplier_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, true),
        Field::new("s_flag", DataType::Int32, false),
    ]))
}

fn threshold_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "min_price",
        DataType::Int64,
        false,
    )]))
}

fn write_parquet_files<F>(
    dir: &Path,
    num_files: usize,
    total_rows: usize,
    rg_rows: usize,
    schema: Arc<Schema>,
    mut make_batch: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(usize, usize) -> Result<RecordBatch, Box<dyn Error>>,
{
    std::fs::create_dir_all(dir)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(rg_rows))
        .build();
    let rows_per_file = total_rows.div_ceil(num_files);
    let mut offset = 0usize;
    for f_idx in 0..num_files {
        if offset >= total_rows {
            break;
        }
        let file_end = (offset + rows_per_file).min(total_rows);
        let mut writer = ArrowWriter::try_new(
            File::create(dir.join(format!("part-{f_idx:02}.parquet")))?,
            Arc::clone(&schema),
            Some(props.clone()),
        )?;
        while offset < file_end {
            let len = (file_end - offset).min(BATCH_SIZE);
            let batch = make_batch(offset, len)?;
            writer.write(&batch)?;
            offset += len;
        }
        writer.close()?;
    }
    Ok(())
}

fn summarize_batches(batches: &[RecordBatch]) -> (usize, i128) {
    let mut rows = 0usize;
    let mut checksum = 0i128;
    for batch in batches {
        rows += batch.num_rows();
        for col in batch.columns() {
            if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        checksum = checksum.wrapping_add(a.value(i) as i128);
                    }
                }
            } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        checksum = checksum.wrapping_add(a.value(i) as i128);
                    }
                }
            }
        }
    }
    (rows, checksum)
}

fn median_ms(samples: &[f64]) -> f64 {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

fn make_ctx(
    mode: MorselMode,
    workers: usize,
    metrics: Option<Arc<MorselMetrics>>,
) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(BATCH_SIZE);
    let exec = &mut config.options_mut().execution;
    exec.morsel_mode = mode;
    exec.parquet.pushdown_filters = false;
    exec.parquet.pruning = false;
    if let Some(m) = metrics {
        config = config.with_extension(m);
    }
    SessionContext::new_with_config(config)
}

async fn run_one(
    ctx: &SessionContext,
    sql: &str,
    expect_morsel: bool,
) -> Result<(f64, usize, i128, String), Box<dyn Error>> {
    let df = ctx.sql(sql).await?;
    let plan = df.create_physical_plan().await?;
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    if expect_morsel {
        if plan.name() != "MorselExec" {
            return Err(format!("expected MorselExec, got:\n{plan_str}").into());
        }
    } else if plan.name() == "MorselExec" {
        return Err(format!("expected stock plan, got MorselExec:\n{plan_str}").into());
    }
    let start = Instant::now();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (rows, checksum) = summarize_batches(&batches);
    Ok((elapsed_ms, rows, checksum, plan_str))
}

fn main() -> Result<(), Box<dyn Error>> {
    let sf: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let workers: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let warmups: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let repeats: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(workers)
        .enable_all()
        .build()?;
    rt.block_on(run_bench(sf, workers, warmups, repeats))
}

async fn run_bench(
    sf: usize,
    workers: usize,
    warmups: usize,
    repeats: usize,
) -> Result<(), Box<dyn Error>> {
    // TPC-H proportional sizes:
    // SF=1: customer=150K, orders=1.5M, supplier=10K, threshold=1
    // SF=5: customer=750K, orders=7.5M, supplier=50K, threshold=1
    let cust_rows = 150_000 * sf;
    let ord_rows = 1_500_000 * sf;
    let supp_rows = 10_000 * sf;
    eprintln!(
        "Generating SF={sf} Parquet datasets: customer={cust_rows}, orders={ord_rows}, supplier={supp_rows}..."
    );

    let tmp = TempDir::new()?;
    let cust_dir = tmp.path().join("customer");
    let ord_dir = tmp.path().join("orders");
    let supp_dir = tmp.path().join("supplier");
    let supp_null_dir = tmp.path().join("supplier_null");
    let thresh_dir = tmp.path().join("threshold");

    let c_schema = customer_schema();
    write_parquet_files(
        &cust_dir,
        4,
        cust_rows,
        50_000,
        Arc::clone(&c_schema),
        |start, len| {
            let mut keys = Vec::with_capacity(len);
            let mut segs = Vec::with_capacity(len);
            let mut bals = Vec::with_capacity(len);
            for i in start..start + len {
                keys.push(Some(i as i64));
                segs.push((i % 5) as i32);
                bals.push(((splitmix64(i as u64) % 10_000) as i64) - 1_000);
            }
            Ok(RecordBatch::try_new(
                Arc::clone(&c_schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(Int32Array::from(segs)),
                    Arc::new(Int64Array::from(bals)),
                ],
            )?)
        },
    )?;

    let o_schema = orders_schema();
    // ~20% of customers have NO orders (for Q13/Q22/RIGHT/FULL OUTER unmatched rows),
    // ~2% of orders have "special requests" comments (for Q13 residual filter).
    let active_cust = (cust_rows * 4) / 5;
    write_parquet_files(
        &ord_dir,
        16,
        ord_rows,
        125_000,
        Arc::clone(&o_schema),
        |start, len| {
            let mut okeys = Vec::with_capacity(len);
            let mut ckeys = Vec::with_capacity(len);
            let mut skeys = Vec::with_capacity(len);
            let mut prices = Vec::with_capacity(len);
            let mut comments = Vec::with_capacity(len);
            for i in start..start + len {
                let h = splitmix64((i as u64).wrapping_add(0xBEEF));
                okeys.push(i as i64);
                let ck = if h % 200 == 0 {
                    // ~0.5% unmatched order customer keys (for FULL OUTER)
                    Some((cust_rows + (h as usize % 5_000)) as i64)
                } else {
                    Some((h as usize % active_cust) as i64)
                };
                ckeys.push(ck);
                skeys.push(Some(((h >> 8) as usize % supp_rows) as i64));
                prices.push(((h >> 16) % 100_000) as i64);
                let cmt = if h % 50 == 0 {
                    "order with special pending requests"
                } else {
                    "standard fulfilled order comment"
                };
                comments.push(cmt);
            }
            Ok(RecordBatch::try_new(
                Arc::clone(&o_schema),
                vec![
                    Arc::new(Int64Array::from(okeys)),
                    Arc::new(Int64Array::from(ckeys)),
                    Arc::new(Int64Array::from(skeys)),
                    Arc::new(Int64Array::from(prices)),
                    Arc::new(StringArray::from(comments)),
                ],
            )?)
        },
    )?;

    let s_schema = supplier_schema();
    write_parquet_files(
        &supp_dir,
        2,
        supp_rows,
        10_000,
        Arc::clone(&s_schema),
        |start, len| {
            let mut skeys = Vec::with_capacity(len);
            let mut flags = Vec::with_capacity(len);
            for i in start..start + len {
                skeys.push(Some(i as i64));
                flags.push(if i % 20 == 0 { 1 } else { 0 });
            }
            Ok(RecordBatch::try_new(
                Arc::clone(&s_schema),
                vec![
                    Arc::new(Int64Array::from(skeys)),
                    Arc::new(Int32Array::from(flags)),
                ],
            )?)
        },
    )?;
    write_parquet_files(
        &supp_null_dir,
        2,
        supp_rows,
        10_000,
        Arc::clone(&s_schema),
        |start, len| {
            let mut skeys = Vec::with_capacity(len);
            let mut flags = Vec::with_capacity(len);
            for i in start..start + len {
                skeys.push(if i == 100 { None } else { Some(i as i64) });
                flags.push(if i % 20 == 0 { 1 } else { 0 });
            }
            Ok(RecordBatch::try_new(
                Arc::clone(&s_schema),
                vec![
                    Arc::new(Int64Array::from(skeys)),
                    Arc::new(Int32Array::from(flags)),
                ],
            )?)
        },
    )?;

    let t_schema = threshold_schema();
    write_parquet_files(&thresh_dir, 1, 1, 1, Arc::clone(&t_schema), |_, _| {
        Ok(RecordBatch::try_new(
            Arc::clone(&t_schema),
            vec![Arc::new(Int64Array::from(vec![80_000]))],
        )?)
    })?;

    let stock = make_ctx(MorselMode::Off, workers, None);
    let metrics = Arc::new(MorselMetrics::default());
    let native = make_ctx(MorselMode::Required, workers, Some(Arc::clone(&metrics)));

    for ctx in [&stock, &native] {
        ctx.register_parquet(
            "customer",
            cust_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "orders",
            ord_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "supplier",
            supp_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "supplier_null",
            supp_null_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "threshold",
            thresh_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    }

    let workloads = [
        (
            "inner_baseline",
            "SELECT c.c_mktsegment, COUNT(*) AS n, SUM(o.o_totalprice) AS total \
             FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey \
             GROUP BY c.c_mktsegment",
        ),
        (
            "left_semi_exists",
            "SELECT COUNT(*) AS n, SUM(o.o_totalprice) AS total \
             FROM orders o \
             WHERE EXISTS ( \
                 SELECT 1 FROM customer c \
                 WHERE c.c_custkey = o.o_custkey AND c.c_mktsegment = 0 \
             )",
        ),
        (
            "left_anti_not_exists",
            "SELECT c.c_mktsegment, COUNT(*) AS n, SUM(c.c_acctbal) AS total \
             FROM customer c \
             WHERE NOT EXISTS ( \
                 SELECT 1 FROM orders o WHERE o.o_custkey = c.c_custkey \
             ) \
             GROUP BY c.c_mktsegment",
        ),
        (
            "not_in_null_aware",
            "SELECT COUNT(*) AS n, SUM(o.o_totalprice) AS total \
             FROM orders o \
             WHERE o.o_suppkey NOT IN ( \
                 SELECT s.s_suppkey FROM supplier s WHERE s.s_flag = 1 \
             )",
        ),
        (
            "not_in_null_short_circuit",
            "SELECT COUNT(*) AS n \
             FROM orders o \
             WHERE o.o_suppkey NOT IN ( \
                 SELECT s.s_suppkey FROM supplier_null s WHERE s.s_flag = 1 \
             )",
        ),
        (
            "left_outer_q13_filter",
            "SELECT c.c_mktsegment, COUNT(o.o_orderkey) AS cnt \
             FROM customer c LEFT OUTER JOIN orders o \
               ON c.c_custkey = o.o_custkey AND o.o_comment NOT LIKE '%special%requests%' \
             GROUP BY c.c_mktsegment",
        ),
        (
            "right_outer_child_emit",
            "SELECT COUNT(*) AS n, COUNT(o.o_orderkey) AS matched_o \
             FROM orders o RIGHT OUTER JOIN customer c ON o.o_custkey = c.c_custkey",
        ),
        (
            "full_outer_child_emit",
            "SELECT COUNT(*) AS n, COUNT(o.o_orderkey) AS matched_o, COUNT(c.c_custkey) AS matched_c \
             FROM orders o FULL OUTER JOIN customer c ON o.o_custkey = c.c_custkey",
        ),
        (
            "cross_join_scalar",
            "SELECT COUNT(*) AS n, SUM(o.o_totalprice) AS total \
             FROM orders o CROSS JOIN threshold t \
             WHERE o.o_totalprice > t.min_price",
        ),
    ];

    println!(
        "SF={sf} workers={workers} batch_size={BATCH_SIZE} warmups={warmups} repeats={repeats}"
    );
    println!(
        "| Workload | Stock median ms (min-max) | Morsel median ms (min-max) | Speedup | Output rows | Pipelines (B/P/E) |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | --- |");

    for (name, sql) in workloads {
        let (_, s_rows, s_sum, _) = run_one(&stock, sql, false).await?;
        let (_, m_rows, m_sum, _) = run_one(&native, sql, true).await?;
        if s_rows != m_rows || s_sum != m_sum {
            return Err(format!(
                "PARITY MISMATCH on SF={sf}/{name}: stock=({s_rows}, {s_sum}) vs morsel=({m_rows}, {m_sum})"
            )
            .into());
        }
        for _ in 1..warmups {
            let _ = run_one(&stock, sql, false).await?;
            let _ = run_one(&native, sql, true).await?;
        }
        let _ = metrics.take_runs();

        let mut stock_ms = Vec::with_capacity(repeats);
        let mut morsel_ms = Vec::with_capacity(repeats);
        for r in 0..repeats {
            if r % 2 == 0 {
                stock_ms.push(run_one(&stock, sql, false).await?.0);
                morsel_ms.push(run_one(&native, sql, true).await?.0);
            } else {
                morsel_ms.push(run_one(&native, sql, true).await?.0);
                stock_ms.push(run_one(&stock, sql, false).await?.0);
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let runs = metrics.take_runs();
        let (mut b_m, mut p_m, mut e_m) = (0usize, 0usize, 0usize);
        if let Some(last) = runs.last() {
            for m in last.iter() {
                let pipe = m
                    .labels()
                    .iter()
                    .find(|l| l.name() == "pipeline")
                    .map(|l| l.value());
                if m.value().name() == "morsels_claimed" {
                    match pipe {
                        Some("build") => b_m += m.value().as_usize(),
                        Some("probe") => p_m += m.value().as_usize(),
                        Some("build_emit") => e_m += m.value().as_usize(),
                        _ => {}
                    }
                }
            }
        }

        let s_med = median_ms(&stock_ms);
        let s_min = stock_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let s_max = stock_ms.iter().copied().fold(0.0f64, f64::max);
        let m_med = median_ms(&morsel_ms);
        let m_min = morsel_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let m_max = morsel_ms.iter().copied().fold(0.0f64, f64::max);
        let speedup = s_med / m_med;
        println!(
            "| `{name}` | {s_med:.1} ({s_min:.1}-{s_max:.1}) | {m_med:.1} ({m_min:.1}-{m_max:.1}) | {speedup:.2}x | {m_rows} | {b_m} / {p_m} / {e_m} |"
        );
    }

    Ok(())
}
