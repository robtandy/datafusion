// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Paired release benchmark comparing stock DataFusion (`morsel_mode=off`)
//! vs native two-pipeline morsel HashJoin (`morsel_mode=required`) on
//! balanced and skewed MemTable and Parquet probe inputs (#158).

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::config::MorselMode;
use datafusion::datasource::MemTable;
use datafusion::morsel::MorselMetrics;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const BATCH_SIZE: usize = 8192;
const DIM_ROWS: usize = 100_000;
const FACT_ROWS: usize = 4_000_000;
const RG_ROWS: usize = 125_000;

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn dim_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("grp", DataType::Int32, false),
        Field::new("weight", DataType::Int64, false),
        Field::new("tag", DataType::Utf8, false),
    ]))
}

fn fact_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("k", DataType::Int64, true),
        Field::new("val", DataType::Int64, false),
    ]))
}

fn build_dim_batches() -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    let schema = dim_schema();
    let tags = ["north", "south", "east", "west"];
    let mut batches = Vec::new();
    let mut start = 0usize;
    while start < DIM_ROWS {
        let len = (DIM_ROWS - start).min(BATCH_SIZE);
        let mut ks = Vec::with_capacity(len);
        let mut grps = Vec::with_capacity(len);
        let mut weights = Vec::with_capacity(len);
        let mut tag_vals = Vec::with_capacity(len);
        for i in start..start + len {
            ks.push(Some(i as i64));
            grps.push((i % 64) as i32);
            weights.push(((splitmix64(i as u64) % 1000) as i64) - 250);
            tag_vals.push(tags[i % tags.len()]);
        }
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ks)),
                Arc::new(Int32Array::from(grps)),
                Arc::new(Int64Array::from(weights)),
                Arc::new(StringArray::from(tag_vals)),
            ],
        )?);
        start += len;
    }
    Ok(batches)
}

fn build_fact_batch(
    start: usize,
    len: usize,
    skewed: bool,
) -> Result<RecordBatch, Box<dyn Error>> {
    let schema = fact_schema();
    let mut ids = Vec::with_capacity(len);
    let mut ks = Vec::with_capacity(len);
    let mut vals = Vec::with_capacity(len);
    for i in start..start + len {
        let h = splitmix64((i as u64).wrapping_add(0x5EED));
        ids.push(i as i64);
        let key = if h % 100 == 0 {
            None
        } else if h % 20 == 1 {
            // ~5% unmatched keys
            Some((DIM_ROWS as i64) + ((h % 10_000) as i64))
        } else if skewed && (h % 2 == 0) {
            // 50% of probe rows hit 10 hot build keys
            Some((h % 10) as i64)
        } else {
            Some((h % (DIM_ROWS as u64)) as i64)
        };
        ks.push(key);
        vals.push(((h >> 16) % 10_000) as i64);
    }
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(ks)),
            Arc::new(Int64Array::from(vals)),
        ],
    )?)
}

fn write_parquet_dir(
    dir: &Path,
    batches_per_file: &[Vec<RecordBatch>],
    max_rg_rows: usize,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(max_rg_rows))
        .build();
    for (idx, batches) in batches_per_file.iter().enumerate() {
        let schema = batches[0].schema();
        let mut writer = ArrowWriter::try_new(
            File::create(dir.join(format!("part-{idx:02}.parquet")))?,
            schema,
            Some(props.clone()),
        )?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
    }
    Ok(())
}

fn build_fact_layout(skewed: bool) -> Result<Vec<Vec<RecordBatch>>, Box<dyn Error>> {
    // Balanced: 16 equal files of 250,000 rows (2 row groups of 125,000 each).
    // Skewed: 1 hot file of 1,750,000 rows (14 row groups of 125,000) + 15 small files of 150,000 rows.
    let file_sizes: Vec<usize> = if skewed {
        let mut sizes = vec![1_750_000usize];
        sizes.extend(std::iter::repeat_n(150_000usize, 15));
        sizes
    } else {
        vec![FACT_ROWS / 16; 16]
    };
    let mut files = Vec::with_capacity(file_sizes.len());
    let mut offset = 0usize;
    for file_rows in file_sizes {
        let mut file_batches = Vec::new();
        let mut rem = file_rows;
        while rem > 0 {
            let chunk = rem.min(BATCH_SIZE);
            file_batches.push(build_fact_batch(offset, chunk, skewed)?);
            offset += chunk;
            rem -= chunk;
        }
        files.push(file_batches);
    }
    assert_eq!(offset, FACT_ROWS);
    Ok(files)
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
    let workers: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let warmups: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let repeats: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(workers)
        .enable_all()
        .build()?;
    rt.block_on(run_bench(workers, warmups, repeats))
}

async fn run_bench(
    workers: usize,
    warmups: usize,
    repeats: usize,
) -> Result<(), Box<dyn Error>> {
    eprintln!(
        "Generating datasets: dim={DIM_ROWS} rows, fact_balanced={FACT_ROWS} rows, fact_skewed={FACT_ROWS} rows..."
    );
    let dim_batches = build_dim_batches()?;
    let bal_files = build_fact_layout(false)?;
    let skew_files = build_fact_layout(true)?;

    let tmp = TempDir::new()?;
    let dim_dir = tmp.path().join("dim");
    let bal_dir = tmp.path().join("fact_balanced");
    let skew_dir = tmp.path().join("fact_skewed");

    // Split dim into 4 Parquet files
    let mut dim_per_file: Vec<Vec<RecordBatch>> = vec![Vec::new(); 4];
    for (i, b) in dim_batches.iter().enumerate() {
        dim_per_file[i % 4].push(b.clone());
    }
    write_parquet_dir(&dim_dir, &dim_per_file, 25_000)?;
    write_parquet_dir(&bal_dir, &bal_files, RG_ROWS)?;
    write_parquet_dir(&skew_dir, &skew_files, RG_ROWS)?;

    let stock = make_ctx(MorselMode::Off, workers, None);
    let metrics = Arc::new(MorselMetrics::default());
    let native = make_ctx(MorselMode::Required, workers, Some(Arc::clone(&metrics)));

    // Register MemTables (with 16 partitions matching file layout)
    let dim_mem_parts: Vec<Vec<RecordBatch>> = dim_per_file.clone();
    for ctx in [&stock, &native] {
        ctx.register_table(
            "dim_mem",
            Arc::new(MemTable::try_new(dim_schema(), dim_mem_parts.clone())?),
        )?;
        ctx.register_table(
            "fact_bal_mem",
            Arc::new(MemTable::try_new(fact_schema(), bal_files.clone())?),
        )?;
        ctx.register_table(
            "fact_skew_mem",
            Arc::new(MemTable::try_new(fact_schema(), skew_files.clone())?),
        )?;
        ctx.register_parquet(
            "dim_pq",
            dim_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "fact_bal_pq",
            bal_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "fact_skew_pq",
            skew_dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    }

    let workloads = [
        (
            "mem_balanced_proj",
            "SELECT f.id, d.weight + f.val AS total FROM fact_bal_mem f JOIN dim_mem d ON f.k = d.k",
        ),
        (
            "mem_skewed_proj",
            "SELECT f.id, d.weight + f.val AS total FROM fact_skew_mem f JOIN dim_mem d ON f.k = d.k",
        ),
        (
            "mem_skewed_count",
            "SELECT COUNT(*) AS n FROM fact_skew_mem f JOIN dim_mem d ON f.k = d.k",
        ),
        (
            "parquet_balanced_agg",
            "SELECT d.grp, COUNT(*) AS n, SUM(f.val + d.weight) AS total FROM fact_bal_pq f JOIN dim_pq d ON f.k = d.k GROUP BY d.grp",
        ),
        (
            "parquet_skewed_agg",
            "SELECT d.grp, COUNT(*) AS n, SUM(f.val + d.weight) AS total FROM fact_skew_pq f JOIN dim_pq d ON f.k = d.k GROUP BY d.grp",
        ),
        (
            "parquet_balanced_count",
            "SELECT COUNT(*) AS n FROM fact_bal_pq f JOIN dim_pq d ON f.k = d.k WHERE f.val > 0",
        ),
        (
            "parquet_skewed_proj",
            "SELECT f.id, d.weight + f.val AS total FROM fact_skew_pq f JOIN dim_pq d ON f.k = d.k WHERE f.val % 10 = 0",
        ),
    ];

    println!(
        "workers={workers} batch_size={BATCH_SIZE} warmups={warmups} repeats={repeats}"
    );
    println!(
        "| Workload | Stock median ms (min-max) | Morsel median ms (min-max) | Speedup | Output rows | Build/Probe morsels |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | --- |");

    for (name, sql) in workloads {
        // Parity check + warmup
        let (_, s_rows, s_sum, _) = run_one(&stock, sql, false).await?;
        let (_, m_rows, m_sum, m_plan) = run_one(&native, sql, true).await?;
        if s_rows != m_rows || s_sum != m_sum {
            return Err(format!(
                "PARITY MISMATCH on {name}: stock=({s_rows}, {s_sum}) vs morsel=({m_rows}, {m_sum})"
            )
            .into());
        }
        assert!(
            m_plan.contains("HashJoinBuildSink") && m_plan.contains("HashJoinProbe"),
            "missing join operators in plan for {name}:\n{m_plan}"
        );
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

        // Wait briefly for last run metric drop
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let runs = metrics.take_runs();
        let (mut b_morsels, mut p_morsels) = (0usize, 0usize);
        if let Some(last) = runs.last() {
            for m in last.iter() {
                let pipe = m
                    .labels()
                    .iter()
                    .find(|l| l.name() == "pipeline")
                    .map(|l| l.value());
                if m.value().name() == "morsels_claimed" {
                    match pipe {
                        Some("build") => b_morsels += m.value().as_usize(),
                        Some("probe") => p_morsels += m.value().as_usize(),
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
            "| `{name}` | {s_med:.1} ({s_min:.1}-{s_max:.1}) | {m_med:.1} ({m_min:.1}-{m_max:.1}) | {speedup:.2}x | {m_rows} | {b_morsels} / {p_morsels} |"
        );
    }

    Ok(())
}
