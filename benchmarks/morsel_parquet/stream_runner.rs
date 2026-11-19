// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::error::Error;
use std::io;
use std::sync::Arc;

use arrow::csv::Writer;
use datafusion::common::config::MorselMode;
use datafusion::common::instant::Instant;
use datafusion::physical_plan::{ExecutionPlan, displayable, execute_stream};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::TryStreamExt;
use serde_json::json;

const BATCH_SIZE: usize = 8192;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [operation, mode, workers, ddl, sql] = args.as_slice() else {
        return Err("usage: morsel_parquet_stream <query|measure> <off|required> <workers> <ddl> <sql>".into());
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
    let mut config = SessionConfig::new()
        .with_batch_size(BATCH_SIZE)
        .with_target_partitions(workers)
        .with_collect_statistics(false)
        .with_repartition_file_scans(false);
    let execution = &mut config.options_mut().execution;
    execution.morsel_mode = mode;
    execution.parquet.pruning = false;
    execution.parquet.pushdown_filters = false;
    execution.parquet.enable_page_index = false;
    execution.parquet.bloom_filter_on_read = false;
    let ctx = SessionContext::new_with_config(config);
    let mut writer = csv.then(|| Writer::new(io::stdout().lock()));

    // Start before registration: schema inference reads the fixture's footers.
    let start = Instant::now();
    ctx.sql(ddl).await?;
    let df = ctx.sql(sql).await?;
    let task_ctx = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await?;
    let plan_seconds = start.elapsed().as_secs_f64();
    let mut stream = execute_stream(Arc::clone(&plan), task_ctx)?;
    let mut first_output_seconds = None;
    let mut first_8192_seconds = None;
    let mut output_rows = 0;
    let mut output_batches = 0;
    while let Some(batch) = stream.try_next().await? {
        if batch.num_rows() != 0 {
            let before = output_rows;
            output_rows += batch.num_rows();
            output_batches += 1;
            let reached_8192 = before < BATCH_SIZE && output_rows >= BATCH_SIZE;
            if output_batches == 1 || reached_8192 {
                let elapsed = start.elapsed().as_secs_f64();
                if output_batches == 1 {
                    first_output_seconds = Some(elapsed);
                }
                if reached_8192 {
                    first_8192_seconds = Some(elapsed);
                }
            }
            if let Some(writer) = &mut writer {
                writer.write(&batch)?;
            }
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    if !csv {
        println!(
            "{}",
            json!({
                "build_revision": env!("MORSEL_BENCH_REVISION"),
                "output_rows": output_rows,
                "output_batches": output_batches,
                "first_output_seconds": first_output_seconds,
                "first_8192_seconds": first_8192_seconds,
                "seconds": seconds,
                "plan_seconds": plan_seconds,
                "plan": displayable(plan.as_ref()).indent(false).to_string(),
                "parquet_morsels_claimed": metric(plan.as_ref(), "parquet_morsels_claimed"),
                "count_finalizations": metric(plan.as_ref(), "count_finalizations"),
                "bytes_scanned": metric(plan.as_ref(), "bytes_scanned"),
            })
        );
    }
    Ok(())
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
