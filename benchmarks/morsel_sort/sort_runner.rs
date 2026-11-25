// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Paired release benchmark comparing stock DataFusion (`morsel_mode=off`)
//! vs native parallel TopN and Sort (`morsel_mode=required`) on uniform and
//! skewed Parquet scans (integer and `Utf8View` keys) and Parquet HashJoin
//! inputs under a bounded 1 GiB memory pool (#159).

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    StringViewArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::config::MorselMode;
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryPool, UnboundedMemoryPool,
};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::SessionStateBuilder;
use datafusion::morsel::MorselMetrics;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const WRITE_BATCH_SIZE: usize = 8192;
const TOTAL_ROWS: usize = 4_000_000;
const DIM_ROWS: usize = 400_000;
const RG_ROWS: usize = 125_000;
const DEFAULT_POOL_MIB: usize = 0; // 0 = UnboundedMemoryPool for paired speed; dedicated 1 GiB check runs separately

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, true),
        Field::new("metric", DataType::Float64, true),
        Field::new("grp", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn dim_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("weight", DataType::Int64, false),
    ]))
}

fn fact_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("k", DataType::Int64, false),
    ]))
}

fn build_scan_batch(
    start: usize,
    len: usize,
    skewed: bool,
) -> Result<RecordBatch, Box<dyn Error>> {
    let schema = table_schema();
    let prefixes = [
        "region-north-alpha",
        "region-south-beta",
        "region-east-gamma",
        "region-west-delta",
        "region-central-eps",
        "region-edge-zeta",
    ];
    let mut ids = Vec::with_capacity(len);
    let mut scores = Vec::with_capacity(len);
    let mut metrics = Vec::with_capacity(len);
    let mut grps = Vec::with_capacity(len);
    let mut names: Vec<Option<String>> = Vec::with_capacity(len);

    for i in start..start + len {
        let h = splitmix64((i as u64).wrapping_add(0x5EED_159));
        ids.push(i as i64);
        let score = if h % 100 == 0 {
            None
        } else if skewed && (h % 2 == 0) {
            Some(((h % 10) as i64) * 1_000_000)
        } else {
            Some((h % 10_000_000) as i64)
        };
        scores.push(score);
        let m = if h % 100 == 1 {
            None
        } else {
            Some(((h >> 16) % 1_000_000) as f64 / 100.0)
        };
        metrics.push(m);
        grps.push(((h >> 8) % 64) as i32);
        let name = if h % 100 == 2 {
            None
        } else {
            // >12 bytes so Parquet Utf8View stores out-of-line data buffers.
            let p = prefixes[(h as usize) % prefixes.len()];
            let suffix = (h >> 24) % 100_000;
            Some(format!("{p}-{suffix:05}-item"))
        };
        names.push(name);
    }

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(scores)),
            Arc::new(Float64Array::from(metrics)),
            Arc::new(Int32Array::from(grps)),
            Arc::new(StringArray::from(names)),
        ],
    )?)
}

fn write_scan_parquet_dataset(dir: &Path, skewed: bool) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(RG_ROWS))
        .build();

    let file_sizes: Vec<usize> = if skewed {
        let mut sizes = vec![1_750_000usize];
        sizes.extend(std::iter::repeat_n(150_000usize, 15));
        sizes
    } else {
        vec![TOTAL_ROWS / 16; 16]
    };

    let mut offset = 0usize;
    let schema = table_schema();
    for (file_idx, &size) in file_sizes.iter().enumerate() {
        let mut writer = ArrowWriter::try_new(
            File::create(dir.join(format!("part-{file_idx:02}.parquet")))?,
            Arc::clone(&schema),
            Some(props.clone()),
        )?;
        let mut remaining = size;
        while remaining > 0 {
            let n = remaining.min(WRITE_BATCH_SIZE);
            let batch = build_scan_batch(offset, n, skewed)?;
            writer.write(&batch)?;
            offset += n;
            remaining -= n;
        }
        writer.close()?;
    }
    Ok(())
}

fn write_join_parquet_datasets(
    dim_dir: &Path,
    fact_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dim_dir)?;
    std::fs::create_dir_all(fact_dir)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(RG_ROWS))
        .build();

    // 400K-row dimension table across 4 files with >12-byte strings (read as Utf8View).
    let d_schema = dim_schema();
    let per_dim_file = DIM_ROWS / 4;
    let mut dim_offset = 0usize;
    for file_idx in 0..4 {
        let mut writer = ArrowWriter::try_new(
            File::create(dim_dir.join(format!("dim-{file_idx:02}.parquet")))?,
            Arc::clone(&d_schema),
            Some(props.clone()),
        )?;
        let mut remaining = per_dim_file;
        while remaining > 0 {
            let n = remaining.min(WRITE_BATCH_SIZE);
            let mut ks = Vec::with_capacity(n);
            let mut names = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);
            for i in dim_offset..dim_offset + n {
                let h = splitmix64((i as u64).wrapping_add(0xD1A_159));
                ks.push(i as i64);
                names.push(format!(
                    "dim-entity-{:06}-{:08x}-utf8view-name",
                    h % 1_000_000,
                    (h >> 20) as u32
                ));
                weights.push(((h % 1000) as i64) - 250);
            }
            let batch = RecordBatch::try_new(
                Arc::clone(&d_schema),
                vec![
                    Arc::new(Int64Array::from(ks)),
                    Arc::new(StringArray::from(names)),
                    Arc::new(Int64Array::from(weights)),
                ],
            )?;
            writer.write(&batch)?;
            dim_offset += n;
            remaining -= n;
        }
        writer.close()?;
    }

    // 4M-row fact table across 16 files probing the 400K-row dimension.
    let f_schema = fact_schema();
    let per_fact_file = TOTAL_ROWS / 16;
    let mut fact_offset = 0usize;
    for file_idx in 0..16 {
        let mut writer = ArrowWriter::try_new(
            File::create(fact_dir.join(format!("fact-{file_idx:02}.parquet")))?,
            Arc::clone(&f_schema),
            Some(props.clone()),
        )?;
        let mut remaining = per_fact_file;
        while remaining > 0 {
            let n = remaining.min(WRITE_BATCH_SIZE);
            let mut ids = Vec::with_capacity(n);
            let mut ks = Vec::with_capacity(n);
            for i in fact_offset..fact_offset + n {
                let h = splitmix64((i as u64).wrapping_add(0xFAC7_159));
                ids.push(i as i64);
                ks.push((h % (DIM_ROWS as u64)) as i64);
            }
            let batch = RecordBatch::try_new(
                Arc::clone(&f_schema),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(Int64Array::from(ks)),
                ],
            )?;
            writer.write(&batch)?;
            fact_offset += n;
            remaining -= n;
        }
        writer.close()?;
    }
    Ok(())
}

/// Position-sensitive ordered checksum across all output batches.
fn ordered_key_digest(batches: &[RecordBatch], key_cols: &[usize]) -> (usize, u64) {
    let mut rows = 0usize;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: u64| {
        hash ^= v;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for batch in batches {
        let n = batch.num_rows();
        rows += n;
        for &c in key_cols {
            let col = batch.column(c);
            if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                for i in 0..n {
                    if a.is_null(i) {
                        mix(0xffff_ffff_ffff_fffe);
                    } else {
                        mix(a.value(i) as u64);
                    }
                }
            } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
                for i in 0..n {
                    if a.is_null(i) {
                        mix(0xffff_ffff_ffff_fffe);
                    } else {
                        mix(a.value(i) as i64 as u64);
                    }
                }
            } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
                for i in 0..n {
                    if a.is_null(i) {
                        mix(0xffff_ffff_ffff_fffe);
                    } else {
                        mix(a.value(i).to_bits());
                    }
                }
            } else if let Some(a) = col.as_any().downcast_ref::<StringViewArray>() {
                for i in 0..n {
                    if a.is_null(i) {
                        mix(0xffff_ffff_ffff_fffe);
                    } else {
                        let s = a.value(i);
                        mix(s.len() as u64);
                        for &b in s.as_bytes() {
                            mix(b as u64);
                        }
                    }
                }
            } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..n {
                    if a.is_null(i) {
                        mix(0xffff_ffff_ffff_fffe);
                    } else {
                        let s = a.value(i);
                        mix(s.len() as u64);
                        for &b in s.as_bytes() {
                            mix(b as u64);
                        }
                    }
                }
            }
        }
    }
    (rows, hash)
}

async fn make_context(
    workers: usize,
    batch_size: usize,
    pool_bytes: usize,
    mode: MorselMode,
    metrics: Option<Arc<MorselMetrics>>,
    uniform_dir: &Path,
    skewed_dir: &Path,
    dim_dir: &Path,
    fact_dir: &Path,
) -> Result<(SessionContext, Arc<dyn MemoryPool>), Box<dyn Error>> {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = mode;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config.options_mut().execution.parquet.pruning = false;
    if let Some(m) = metrics {
        config = config.with_extension(m);
    }
    let pool: Arc<dyn MemoryPool> = if pool_bytes == 0 {
        Arc::new(UnboundedMemoryPool::default())
    } else {
        Arc::new(GreedyMemoryPool::new(pool_bytes))
    };
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_runtime_env(runtime)
        .build();
    let ctx = SessionContext::new_with_state(state);
    for (name, path) in [
        ("pq_uniform", uniform_dir),
        ("pq_skewed", skewed_dir),
        ("dim_pq", dim_dir),
        ("fact_pq", fact_dir),
    ] {
        ctx.register_parquet(
            name,
            path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    }
    Ok((ctx, pool))
}

struct Workload {
    name: &'static str,
    sql: &'static str,
    expected_sink: &'static str,
    key_cols: &'static [usize],
}

const WORKLOADS: &[Workload] = &[
    Workload {
        name: "topn_uniform_limit100",
        sql: "SELECT id, score, name FROM pq_uniform ORDER BY score DESC NULLS LAST, id ASC LIMIT 100",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
    },
    Workload {
        name: "topn_skewed_limit100",
        sql: "SELECT id, score, name FROM pq_skewed ORDER BY score DESC NULLS LAST, id ASC LIMIT 100",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
    },
    Workload {
        name: "topn_utf8view_limit100",
        sql: "SELECT id, name, score FROM pq_uniform ORDER BY name ASC NULLS LAST, id ASC LIMIT 100",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
    },
    Workload {
        name: "topn_join_utf8view_100",
        sql: "SELECT d.name, f.id, d.weight FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY d.name, f.id LIMIT 100",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
    },
    Workload {
        name: "topn_limit_100k",
        sql: "SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST, id ASC LIMIT 100000",
        expected_sink: "TopNSink",
        key_cols: &[1, 0],
    },
    Workload {
        name: "topn_limit_1m",
        sql: "SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST, id ASC LIMIT 1000000",
        expected_sink: "TopNSink",
        key_cols: &[1, 0],
    },
    Workload {
        name: "topn_limit_gt_rows",
        sql: "SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST, id ASC LIMIT 10000000",
        expected_sink: "TopNSink",
        key_cols: &[1, 0],
    },
    Workload {
        name: "sort_uniform_single_key",
        sql: "SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST",
        expected_sink: "SortSink",
        key_cols: &[1],
    },
    Workload {
        name: "sort_skewed_single_key",
        sql: "SELECT id, score FROM pq_skewed ORDER BY score ASC NULLS LAST",
        expected_sink: "SortSink",
        key_cols: &[1],
    },
    Workload {
        name: "sort_utf8view_key",
        sql: "SELECT id, name FROM pq_uniform ORDER BY name ASC NULLS LAST, id ASC",
        expected_sink: "SortSink",
        key_cols: &[1, 0],
    },
    Workload {
        name: "sort_join_id",
        sql: "SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY f.id",
        expected_sink: "SortSink",
        key_cols: &[1, 0],
    },
    Workload {
        name: "sort_join_utf8view",
        sql: "SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY d.name, f.id",
        expected_sink: "SortSink",
        key_cols: &[0, 1],
    },
];

fn median_ms(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    let min = samples[0];
    let max = samples[samples.len() - 1];
    let mid = samples.len() / 2;
    let med = if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        0.5 * (samples[mid - 1] + samples[mid])
    };
    (med, min, max)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let workers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let warmups: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let batch_size: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let pool_mib: usize = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_POOL_MIB);
    let pool_bytes = pool_mib * 1024 * 1024;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let tmp = TempDir::new()?;
        let uniform_dir = tmp.path().join("pq_uniform");
        let skewed_dir = tmp.path().join("pq_skewed");
        let dim_dir = tmp.path().join("dim_pq");
        let fact_dir = tmp.path().join("fact_pq");
        eprintln!(
            "Generating {TOTAL_ROWS}-row scan datasets and {DIM_ROWS}x{TOTAL_ROWS} join Parquet datasets..."
        );
        write_scan_parquet_dataset(&uniform_dir, false)?;
        write_scan_parquet_dataset(&skewed_dir, true)?;
        write_join_parquet_datasets(&dim_dir, &fact_dir)?;

        let collector = Arc::new(MorselMetrics::default());
        let (stock_ctx, stock_pool) = make_context(
            workers,
            batch_size,
            pool_bytes,
            MorselMode::Off,
            None,
            &uniform_dir,
            &skewed_dir,
            &dim_dir,
            &fact_dir,
        )
        .await?;
        let (morsel_ctx, morsel_pool) = make_context(
            workers,
            batch_size,
            pool_bytes,
            MorselMode::Required,
            Some(Arc::clone(&collector)),
            &uniform_dir,
            &skewed_dir,
            &dim_dir,
            &fact_dir,
        )
        .await?;

        // Verify 1 GiB bounded pool succeeds for join TopN and join full Sort under morsel_mode=required.
        if batch_size >= 1024 {
            let (m1g_ctx, m1g_pool) = make_context(
                workers,
                batch_size,
                1024 * 1024 * 1024,
                MorselMode::Required,
                None,
                &uniform_dir,
                &skewed_dir,
                &dim_dir,
                &fact_dir,
            )
            .await?;
            for sql in [
                "SELECT d.name, f.id, d.weight FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY d.name, f.id LIMIT 100",
                "SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY f.id",
                "SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY d.name, f.id",
            ] {
                let mp = m1g_ctx.sql(sql).await?.create_physical_plan().await?;
                let _ = collect(mp, m1g_ctx.task_ctx()).await?;
                assert_eq!(m1g_pool.reserved(), 0);
            }
            eprintln!("Verified all 3 Parquet join TopN/Sort queries succeed cleanly inside 1 GiB pool (batch_size={batch_size}).");
        }

        let pool_label = if pool_mib == 0 {
            "unbounded".to_owned()
        } else {
            format!("{pool_mib}MiB")
        };
        println!(
            "workers={workers} batch_size={batch_size} rows={TOTAL_ROWS} dim_rows={DIM_ROWS} pool={pool_label} warmups={warmups} rounds={rounds}"
        );
        println!(
            "| {:<24} | {:>24} | {:>24} | {:>7} | {:>10} | {:>8} |",
            "Workload",
            "Stock ms (min-max)",
            "Morsel ms (min-max)",
            "Speedup",
            "Out rows",
            "Morsels"
        );
        println!(
            "|{}|{}|{}|{}|{}|{}|",
            "-".repeat(26),
            "-".repeat(26),
            "-".repeat(26),
            "-".repeat(9),
            "-".repeat(12),
            "-".repeat(10)
        );

        for w in WORKLOADS {
            let stock_plan = stock_ctx.sql(w.sql).await?.create_physical_plan().await?;
            let stock_disp = displayable(stock_plan.as_ref()).indent(false).to_string();
            assert!(
                !stock_disp.contains("MorselExec") && stock_disp.contains("SortExec"),
                "stock arm for {} did not use SortExec:\n{stock_disp}",
                w.name
            );

            let morsel_plan = morsel_ctx.sql(w.sql).await?.create_physical_plan().await?;
            let morsel_disp = displayable(morsel_plan.as_ref()).indent(false).to_string();
            assert!(
                morsel_disp.contains("MorselExec")
                    && morsel_disp.contains(w.expected_sink),
                "morsel arm for {} missing MorselExec/{}:\n{morsel_disp}",
                w.name,
                w.expected_sink
            );

            let mut out_rows = 0usize;
            for _ in 0..warmups {
                let sp = stock_ctx.sql(w.sql).await?.create_physical_plan().await?;
                let sb = collect(sp, stock_ctx.task_ctx()).await?;
                let mp = morsel_ctx.sql(w.sql).await?.create_physical_plan().await?;
                let mb = collect(mp, morsel_ctx.task_ctx()).await?;
                let sd = ordered_key_digest(&sb, w.key_cols);
                let md = ordered_key_digest(&mb, w.key_cols);
                assert_eq!(
                    md, sd,
                    "ordered parity mismatch for {}: morsel={md:?} stock={sd:?}",
                    w.name
                );
                out_rows = md.0;
                assert_eq!(stock_pool.reserved(), 0);
                assert_eq!(morsel_pool.reserved(), 0);
                let _ = collector.take_runs();
            }

            let mut stock_ms = Vec::with_capacity(rounds);
            let mut morsel_ms = Vec::with_capacity(rounds);
            let mut last_morsels = 0usize;

            for r in 0..rounds {
                if r % 2 == 0 {
                    let sp = stock_ctx.sql(w.sql).await?.create_physical_plan().await?;
                    let t0 = Instant::now();
                    let sb = collect(sp, stock_ctx.task_ctx()).await?;
                    stock_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        sb.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        out_rows
                    );

                    let mp = morsel_ctx.sql(w.sql).await?.create_physical_plan().await?;
                    let t1 = Instant::now();
                    let mb = collect(mp, morsel_ctx.task_ctx()).await?;
                    morsel_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        mb.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        out_rows
                    );
                } else {
                    let mp = morsel_ctx.sql(w.sql).await?.create_physical_plan().await?;
                    let t1 = Instant::now();
                    let mb = collect(mp, morsel_ctx.task_ctx()).await?;
                    morsel_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        mb.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        out_rows
                    );

                    let sp = stock_ctx.sql(w.sql).await?.create_physical_plan().await?;
                    let t0 = Instant::now();
                    let sb = collect(sp, stock_ctx.task_ctx()).await?;
                    stock_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        sb.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        out_rows
                    );
                }
                assert_eq!(morsel_pool.reserved(), 0);
                let runs = collector.take_runs();
                if let Some(last) = runs.last() {
                    last_morsels = last
                        .sum_by_name("morsels_completed")
                        .map_or(0, |v| v.as_usize());
                }
            }

            let (s_med, s_min, s_max) = median_ms(&mut stock_ms);
            let (m_med, m_min, m_max) = median_ms(&mut morsel_ms);
            let speedup = s_med / m_med;

            println!(
                "| {:<24} | {:>6.1} ({:>5.1}-{:>5.1})       | {:>6.1} ({:>5.1}-{:>5.1})       | {:>6.2}x | {:>10} | {:>8} |",
                w.name, s_med, s_min, s_max, m_med, m_min, m_max, speedup, out_rows, last_morsels
            );
        }
        Ok::<(), Box<dyn Error>>(())
    })
}
