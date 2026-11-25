// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Paired release benchmark comparing stock DataFusion (`morsel_mode=off`)
//! vs native parallel TopN and Sort (`morsel_mode=required`) on uniform and
//! skewed Parquet inputs (#159).

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
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
const TOTAL_ROWS: usize = 4_000_000;
const RG_ROWS: usize = 125_000;

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
        Field::new("tag", DataType::Utf8, false),
    ]))
}

fn build_batch(
    start: usize,
    len: usize,
    skewed: bool,
) -> Result<RecordBatch, Box<dyn Error>> {
    let schema = table_schema();
    let tags = [
        "north-alpha",
        "south-beta",
        "east-gamma",
        "west-delta",
        "central-eps",
        "edge-zeta",
    ];
    let mut ids = Vec::with_capacity(len);
    let mut scores = Vec::with_capacity(len);
    let mut metrics = Vec::with_capacity(len);
    let mut grps = Vec::with_capacity(len);
    let mut tag_vals = Vec::with_capacity(len);

    for i in start..start + len {
        let h = splitmix64((i as u64).wrapping_add(0x5EED_159));
        ids.push(i as i64);
        let score = if h % 100 == 0 {
            None
        } else if skewed && (h % 2 == 0) {
            // 50% of rows collide on 10 hot score values
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
        tag_vals.push(tags[(h as usize) % tags.len()]);
    }

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(scores)),
            Arc::new(Float64Array::from(metrics)),
            Arc::new(Int32Array::from(grps)),
            Arc::new(StringArray::from(tag_vals)),
        ],
    )?)
}

fn write_parquet_dataset(dir: &Path, skewed: bool) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(RG_ROWS))
        .build();

    // Uniform: 16 equal files of 250,000 rows (2 row groups of 125,000 each).
    // Skewed: 1 hot file of 1,750,000 rows (14 row groups of 125,000) + 15 small files of 150,000 rows.
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
            let n = remaining.min(BATCH_SIZE);
            let batch = build_batch(offset, n, skewed)?;
            writer.write(&batch)?;
            offset += n;
            remaining -= n;
        }
        writer.close()?;
    }
    Ok(())
}

/// Position-sensitive ordered checksum across all output batches (verifies both
/// total row count and ordering of the first/key columns).
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
            } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..n {
                    if a.is_null(i) {
                        mix(0xffff_ffff_ffff_fffe);
                    } else {
                        mix(a.value(i).len() as u64);
                        for &b in a.value(i).as_bytes() {
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
    mode: MorselMode,
    metrics: Option<Arc<MorselMetrics>>,
    uniform_dir: &Path,
    skewed_dir: &Path,
) -> Result<SessionContext, Box<dyn Error>> {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(BATCH_SIZE);
    config.options_mut().execution.morsel_mode = mode;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config.options_mut().execution.parquet.pruning = false;
    if let Some(m) = metrics {
        config = config.with_extension(m);
    }
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet(
        "pq_uniform",
        uniform_dir.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    ctx.register_parquet(
        "pq_skewed",
        skewed_dir.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    Ok(ctx)
}

struct Workload {
    name: &'static str,
    sql: &'static str,
    expected_sink: &'static str,
    /// Columns whose ordered sequence is uniquely determined (checked position-sensitively).
    key_cols: &'static [usize],
}

const WORKLOADS: &[Workload] = &[
    Workload {
        name: "topn_uniform_limit100",
        sql: "SELECT id, score, tag FROM pq_uniform ORDER BY score DESC NULLS LAST, id ASC LIMIT 100",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
    },
    Workload {
        name: "topn_skewed_limit100",
        sql: "SELECT id, score, tag FROM pq_skewed ORDER BY score DESC NULLS LAST, id ASC LIMIT 100",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
    },
    Workload {
        name: "topn_uniform_offset",
        sql: "SELECT id, score, metric FROM pq_uniform ORDER BY score ASC NULLS FIRST, id ASC LIMIT 100 OFFSET 50",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
    },
    Workload {
        name: "topn_skewed_filtered",
        sql: "SELECT id, score, tag FROM pq_skewed WHERE grp < 32 ORDER BY score DESC NULLS LAST, id ASC LIMIT 100",
        expected_sink: "TopNSink",
        key_cols: &[0, 1, 2],
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
        name: "sort_uniform_multi_key",
        sql: "SELECT id, grp, score FROM pq_uniform ORDER BY grp ASC, score DESC NULLS LAST",
        expected_sink: "SortSink",
        key_cols: &[1, 2],
    },
    Workload {
        name: "sort_skewed_multi_key",
        sql: "SELECT id, grp, score FROM pq_skewed ORDER BY grp ASC, score DESC NULLS LAST",
        expected_sink: "SortSink",
        key_cols: &[1, 2],
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

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let tmp = TempDir::new()?;
        let uniform_dir = tmp.path().join("pq_uniform");
        let skewed_dir = tmp.path().join("pq_skewed");
        eprintln!("Generating {TOTAL_ROWS}-row uniform and skewed Parquet datasets...");
        write_parquet_dataset(&uniform_dir, false)?;
        write_parquet_dataset(&skewed_dir, true)?;

        let collector = Arc::new(MorselMetrics::default());
        let stock_ctx =
            make_context(workers, MorselMode::Off, None, &uniform_dir, &skewed_dir)
                .await?;
        let morsel_ctx = make_context(
            workers,
            MorselMode::Required,
            Some(Arc::clone(&collector)),
            &uniform_dir,
            &skewed_dir,
        )
        .await?;

        println!(
            "workers={workers} batch_size={BATCH_SIZE} rows={TOTAL_ROWS} warmups={warmups} rounds={rounds}"
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
        println!("|{}|{}|{}|{}|{}|{}|", "-".repeat(26), "-".repeat(26), "-".repeat(26), "-".repeat(9), "-".repeat(12), "-".repeat(10));

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
                morsel_disp.contains("MorselExec") && morsel_disp.contains(w.expected_sink),
                "morsel arm for {} missing MorselExec/{}:\n{morsel_disp}",
                w.name,
                w.expected_sink
            );

            // Warmup + exact ordered parity check
            let mut out_rows = 0usize;
            for _ in 0..warmups {
                let sb = collect(Arc::clone(&stock_plan), stock_ctx.task_ctx()).await?;
                let mb = collect(Arc::clone(&morsel_plan), morsel_ctx.task_ctx()).await?;
                let sd = ordered_key_digest(&sb, w.key_cols);
                let md = ordered_key_digest(&mb, w.key_cols);
                assert_eq!(
                    md, sd,
                    "ordered parity mismatch for {}: morsel={md:?} stock={sd:?}",
                    w.name
                );
                out_rows = md.0;
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
                        out_rows,
                        "stock row count changed on round {r} for {}",
                        w.name
                    );

                    let mp = morsel_ctx.sql(w.sql).await?.create_physical_plan().await?;
                    let t1 = Instant::now();
                    let mb = collect(mp, morsel_ctx.task_ctx()).await?;
                    morsel_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        mb.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        out_rows,
                        "morsel row count changed on round {r} for {}",
                        w.name
                    );
                } else {
                    let mp = morsel_ctx.sql(w.sql).await?.create_physical_plan().await?;
                    let t1 = Instant::now();
                    let mb = collect(mp, morsel_ctx.task_ctx()).await?;
                    morsel_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        mb.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        out_rows,
                        "morsel row count changed on round {r} for {}",
                        w.name
                    );

                    let sp = stock_ctx.sql(w.sql).await?.create_physical_plan().await?;
                    let t0 = Instant::now();
                    let sb = collect(sp, stock_ctx.task_ctx()).await?;
                    stock_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        sb.iter().map(RecordBatch::num_rows).sum::<usize>(),
                        out_rows,
                        "stock row count changed on round {r} for {}",
                        w.name
                    );
                }
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
