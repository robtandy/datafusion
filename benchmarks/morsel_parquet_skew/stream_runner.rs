// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::error::Error;
use std::io;
use std::sync::Arc;

use arrow::csv::Writer;
use datafusion::common::config::MorselMode;
use datafusion::common::instant::Instant;
use datafusion::datasource::physical_plan::parquet::ParquetSplitConfig;
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::morsel::MorselMetrics;
use datafusion::physical_plan::{ExecutionPlan, displayable, execute_stream};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::TryStreamExt;
use serde_json::json;

const BATCH_SIZE: usize = 8192;

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1" | "true" | "TRUE") => true,
        Some("0" | "false" | "FALSE") => false,
        _ => default,
    }
}

fn env_usize(name: &str) -> Result<Option<usize>, Box<dyn Error>> {
    match std::env::var(name) {
        Ok(val) if !val.is_empty() => Ok(Some(val.parse()?)),
        _ => Ok(None),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [operation, mode, workers, ddl, sql] = args.as_slice() else {
        return Err("usage: morsel_parquet_skew <query|measure> <off|required> <workers> <ddl> <sql>".into());
    };
    let csv = match operation.as_str() {
        "query" => true,
        "measure" => false,
        _ => return Err("expected query or measure".into()),
    };
    let mode = match mode.as_str() {
        "off" => MorselMode::Off,
        "required" => MorselMode::Required,
        _ => return Err("expected off or required".into()),
    };
    let workers: usize = workers.parse()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(workers)
        .enable_all()
        .build()?;
    runtime.block_on(run(csv, mode, workers, ddl, sql))
}

async fn run(
    csv: bool,
    mode: MorselMode,
    workers: usize,
    ddl: &str,
    sql: &str,
) -> Result<(), Box<dyn Error>> {
    let enable_page_index = env_bool("MORSEL_ENABLE_PAGE_INDEX", false);
    let pushdown_filters = env_bool("MORSEL_PUSHDOWN_FILTERS", false);
    let split_enabled = env_bool("MORSEL_SPLIT_ENABLED", true);
    let split_buffer_cap_bytes = env_usize("MORSEL_SPLIT_BUFFER_CAP_BYTES")?;
    let memory_pool_bytes = env_usize("MORSEL_MEMORY_POOL_BYTES")?;

    let mut config = SessionConfig::new()
        .with_batch_size(BATCH_SIZE)
        .with_target_partitions(workers)
        .with_collect_statistics(false)
        .with_repartition_file_scans(false);
    let execution = &mut config.options_mut().execution;
    execution.morsel_mode = mode;
    execution.parquet.pruning = false;
    execution.parquet.pushdown_filters = pushdown_filters;
    execution.parquet.enable_page_index = enable_page_index;
    execution.parquet.bloom_filter_on_read = false;
    if !split_enabled || split_buffer_cap_bytes.is_some() {
        let mut split_cfg = ParquetSplitConfig {
            enabled: split_enabled,
            ..Default::default()
        };
        if let Some(cap) = split_buffer_cap_bytes {
            split_cfg.split_buffer_cap_bytes = cap;
        }
        config.set_extension(Arc::new(split_cfg));
    }
    let metrics = Arc::new(MorselMetrics::default());
    if !csv {
        config.set_extension(Arc::clone(&metrics));
    }
    let ctx = if let Some(limit) = memory_pool_bytes {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(limit)))
            .build_arc()?;
        SessionContext::new_with_config_rt(config, runtime)
    } else {
        SessionContext::new_with_config(config)
    };
    let mut writer = csv.then(|| Writer::new(io::stdout().lock()));

    let start = Instant::now();
    ctx.sql(ddl).await?;
    let df = ctx.sql(sql).await?;
    let task_ctx = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await?;
    let plan_seconds = start.elapsed().as_secs_f64();
    let mut stream = execute_stream(Arc::clone(&plan), task_ctx)?;
    let mut output_rows = 0;
    while let Some(batch) = stream.try_next().await? {
        output_rows += batch.num_rows();
        if let Some(writer) = &mut writer {
            writer.write(&batch)?;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    drop(stream);
    if !csv {
        let runs = metrics.take_runs();
        let (claims, completions, decode_ms, process_ms, ready_wait_ms, pending_io_ms) =
            match (mode, runs.as_slice()) {
                (MorselMode::Required, [run]) => (
                    Some(worker_counts(run, "morsels_claimed", workers)?),
                    Some(worker_counts(run, "morsels_completed", workers)?),
                    Some(run_time_ms(run, "morsel_decode_work")),
                    Some(run_time_ms(run, "morsel_process_work")),
                    Some(run_time_ms(run, "morsel_ready_wait")),
                    Some(run_time_ms(run, "morsel_pending_io")),
                ),
                (MorselMode::Off, []) => (None, None, None, None, None, None),
                _ => {
                    return Err(format!(
                        "unexpected number of native reports: {}",
                        runs.len()
                    )
                    .into());
                }
            };
        println!(
            "{}",
            json!({
                "build_revision": env!("MORSEL_BENCH_REVISION"),
                "output_rows": output_rows,
                "seconds": seconds,
                "plan_seconds": plan_seconds,
                "plan": displayable(plan.as_ref()).indent(false).to_string(),
                "parquet_morsels_claimed": metric(plan.as_ref(), "parquet_morsels_claimed"),
                "count_finalizations": metric(plan.as_ref(), "count_finalizations"),
                "parquet_split_row_groups": metric(plan.as_ref(), "parquet_split_row_groups"),
                "parquet_unsplit_row_groups": metric(plan.as_ref(), "parquet_unsplit_row_groups"),
                "parquet_split_slices": metric(plan.as_ref(), "parquet_split_slices"),
                "parquet_split_selected_rows": metric(plan.as_ref(), "parquet_split_selected_rows"),
                "parquet_split_storage_requests": metric(plan.as_ref(), "parquet_split_storage_requests"),
                "parquet_split_data_bytes_returned": metric(plan.as_ref(), "parquet_split_data_bytes_returned"),
                "parquet_split_reused_bytes": metric(plan.as_ref(), "parquet_split_reused_bytes"),
                "parquet_split_peak_unique_compressed_bytes": metric(plan.as_ref(), "parquet_split_peak_unique_compressed_bytes"),
                "bytes_scanned": metric(plan.as_ref(), "bytes_scanned"),
                "worker_claims": claims,
                "worker_completions": completions,
                "morsel_decode_work_ms": decode_ms,
                "morsel_process_work_ms": process_ms,
                "morsel_ready_wait_ms": ready_wait_ms,
                "morsel_pending_io_ms": pending_io_ms,
            })
        );
    }
    Ok(())
}

fn run_time_ms(
    run: &datafusion::physical_plan::metrics::MetricsSet,
    name: &str,
) -> f64 {
    run.sum_by_name(name)
        .map_or(0.0, |v| v.as_usize() as f64 / 1e6)
}

fn worker_counts(
    run: &datafusion::physical_plan::metrics::MetricsSet,
    name: &str,
    workers: usize,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut counts = vec![None; workers];
    for metric in run.iter().filter(|metric| metric.value().name() == name) {
        let worker: usize = metric
            .labels()
            .iter()
            .find(|label| label.name() == "worker")
            .ok_or("missing worker label")?
            .value()
            .parse()?;
        let count = counts.get_mut(worker).ok_or("worker index out of bounds")?;
        if count.replace(metric.value().as_usize()).is_some() {
            return Err(format!("duplicate {name} for worker {worker}").into());
        }
    }
    counts
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| format!("missing {name} for one or more workers").into())
}

fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
    let own = plan
        .metrics()
        .and_then(|metrics| metrics.sum_by_name(name))
        .map_or(0, |value| value.as_usize());
    own + plan
        .children()
        .iter()
        .map(|child| metric(child.as_ref(), name))
        .sum::<usize>()
}
