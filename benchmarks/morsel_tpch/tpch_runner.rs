// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::error::Error;
use std::io;
use std::path::Path;
use std::sync::Arc;

use arrow::ipc::writer::StreamWriter;
use datafusion::common::config::MorselMode;
use datafusion::common::instant::Instant;
use datafusion::physical_plan::{ExecutionPlan, displayable, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::TryStreamExt;
use serde_json::json;

const BATCH_SIZE: usize = 8192;

#[derive(Clone, Copy)]
enum Operation {
    Query,
    Measure,
    Plan,
}

struct RunRequest<'a> {
    operation: Operation,
    mode: MorselMode,
    mode_str: &'a str,
    profile: ScanProfile,
    workers: usize,
    data_dir: &'a Path,
    tables: &'a [String],
    sql: &'a str,
}

#[derive(Clone, Copy)]
enum ScanProfile {
    NoPushdown,
    Default,
    Pruning,
    Pushdown,
}

impl ScanProfile {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "no_pushdown" => Ok(Self::NoPushdown),
            "default" => Ok(Self::Default),
            "pruning" => Ok(Self::Pruning),
            "pushdown" => Ok(Self::Pushdown),
            _ => Err(format!(
                "unknown profile '{value}'; expected no_pushdown, default, pruning, or pushdown"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::NoPushdown => "no_pushdown",
            Self::Default => "default",
            Self::Pruning => "pruning",
            Self::Pushdown => "pushdown",
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [operation, mode, profile, workers, data_dir, tables_csv, sql] = args.as_slice()
    else {
        return Err(
            "usage: morsel_tpch_runner <query|measure|plan> <off|required> <no_pushdown|default|pruning|pushdown> <workers> <data_dir> <tables_csv> <sql>"
                .into(),
        );
    };
    let operation = match operation.as_str() {
        "query" => Operation::Query,
        "measure" => Operation::Measure,
        "plan" => Operation::Plan,
        _ => return Err("expected query, measure, or plan".into()),
    };
    let mode_str = mode.as_str();
    let mode = match mode_str {
        "off" => MorselMode::Off,
        "required" => MorselMode::Required,
        _ => return Err("expected off or required".into()),
    };
    let profile = ScanProfile::parse(profile)?;
    let workers: usize = workers.parse()?;
    let tables: Vec<String> = tables_csv
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect();
    if tables.is_empty() {
        return Err("tables_csv must contain at least one table name".into());
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(workers)
        .enable_all()
        .build()?;
    runtime.block_on(run(RunRequest {
        operation,
        mode,
        mode_str,
        profile,
        workers,
        data_dir: Path::new(data_dir),
        tables: &tables,
        sql,
    }))
}

fn build_config(mode: MorselMode, profile: ScanProfile, workers: usize) -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_batch_size(BATCH_SIZE)
        .with_target_partitions(workers);
    match profile {
        ScanProfile::NoPushdown => {
            config = config
                .with_collect_statistics(false)
                .with_repartition_file_scans(false);
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pruning = false;
            execution.parquet.pushdown_filters = false;
            execution.parquet.enable_page_index = false;
            execution.parquet.bloom_filter_on_read = false;
        }
        ScanProfile::Default => {
            config.options_mut().execution.morsel_mode = mode;
        }
        ScanProfile::Pruning => {
            config = config.with_repartition_file_scans(false);
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pushdown_filters = false;
        }
        ScanProfile::Pushdown => {
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pushdown_filters = true;
            execution.parquet.reorder_filters = true;
        }
    }
    config
}

async fn register_tables(
    ctx: &SessionContext,
    data_dir: &Path,
    tables: &[String],
) -> Result<(), Box<dyn Error>> {
    for table in tables {
        let dir = data_dir.join(table);
        let dir_str = dir
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 table path: {}", dir.display()))?;
        let location = if dir_str.ends_with('/') {
            dir_str.to_owned()
        } else {
            format!("{dir_str}/")
        };
        ctx.register_parquet(table, &location, ParquetReadOptions::default())
            .await?;
    }
    Ok(())
}

async fn run(req: RunRequest<'_>) -> Result<(), Box<dyn Error>> {
    let RunRequest {
        operation,
        mode,
        mode_str,
        profile,
        workers,
        data_dir,
        tables,
        sql,
    } = req;
    let ctx = SessionContext::new_with_config(build_config(mode, profile, workers));

    if matches!(operation, Operation::Plan) {
        register_tables(&ctx, data_dir, tables).await?;
        let outcome = match ctx.sql(sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => json!({
                    "build_revision": env!("MORSEL_BENCH_REVISION"),
                    "mode": mode_str,
                    "profile": profile.as_str(),
                    "admitted": true,
                    "plan": displayable(plan.as_ref()).indent(false).to_string(),
                    "error": null,
                }),
                Err(err) => json!({
                    "build_revision": env!("MORSEL_BENCH_REVISION"),
                    "mode": mode_str,
                    "profile": profile.as_str(),
                    "admitted": false,
                    "plan": null,
                    "error": err.to_string(),
                }),
            },
            Err(err) => json!({
                "build_revision": env!("MORSEL_BENCH_REVISION"),
                "mode": mode_str,
                "profile": profile.as_str(),
                "admitted": false,
                "plan": null,
                "error": err.to_string(),
            }),
        };
        println!("{outcome}");
        return Ok(());
    }

    // Start before registration: schema inference reads table footers, matching DuckDB
    // which reads Parquet footers during query planning/execution.
    let start = Instant::now();
    register_tables(&ctx, data_dir, tables).await?;
    let register_seconds = start.elapsed().as_secs_f64();
    let df = ctx.sql(sql).await?;
    let task_ctx = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await?;
    let plan_done = start.elapsed().as_secs_f64();
    let plan_seconds = plan_done - register_seconds;

    let exec_start = Instant::now();
    let mut stream = execute_stream(Arc::clone(&plan), task_ctx)?;
    let mut writer = match operation {
        Operation::Query => {
            Some(StreamWriter::try_new(io::stdout().lock(), &plan.schema())?)
        }
        Operation::Measure => None,
        Operation::Plan => unreachable!(),
    };
    let mut first_output_seconds = None;
    let mut first_8192_seconds = None;
    let mut output_rows = 0usize;
    let mut output_batches = 0usize;
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
    if let Some(mut writer) = writer {
        writer.finish()?;
        return Ok(());
    }
    let exec_seconds = exec_start.elapsed().as_secs_f64();
    let seconds = start.elapsed().as_secs_f64();

    println!(
        "{}",
        json!({
            "build_revision": env!("MORSEL_BENCH_REVISION"),
            "mode": mode_str,
            "profile": profile.as_str(),
            "output_rows": output_rows,
            "output_batches": output_batches,
            "register_seconds": register_seconds,
            "plan_seconds": plan_seconds,
            "exec_seconds": exec_seconds,
            "first_output_seconds": first_output_seconds,
            "first_8192_seconds": first_8192_seconds,
            "seconds": seconds,
            "plan": displayable(plan.as_ref()).indent(false).to_string(),
            "parquet_morsels_claimed": metric(plan.as_ref(), "parquet_morsels_claimed"),
            "count_finalizations": metric(plan.as_ref(), "count_finalizations"),
            "bytes_scanned": metric(plan.as_ref(), "bytes_scanned"),
        })
    );
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
