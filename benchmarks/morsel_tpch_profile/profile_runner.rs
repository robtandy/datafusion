// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::Path;
use std::sync::Arc;

use arrow::ipc::writer::StreamWriter;
use datafusion::common::config::MorselMode;
use datafusion::common::instant::Instant;
use datafusion::morsel::MorselMetrics;
use datafusion::physical_plan::{ExecutionPlan, displayable, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::TryStreamExt;
use serde_json::{Value, json};

#[derive(Clone, Copy)]
enum Operation {
    Query,
    Measure,
    MeasureWarm,
    MeasureExecLoop,
}

#[derive(Clone, Copy)]
enum ScanProfile {
    NoPushdown,
    Default,
    Pruning,
    Pushdown,
    PushdownNoRepartition,
    PushdownNoReorder,
}

impl ScanProfile {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "no_pushdown" => Ok(Self::NoPushdown),
            "default" => Ok(Self::Default),
            "pruning" => Ok(Self::Pruning),
            "pushdown" => Ok(Self::Pushdown),
            "pushdown_no_repartition" => Ok(Self::PushdownNoRepartition),
            "pushdown_no_reorder" => Ok(Self::PushdownNoReorder),
            _ => Err(format!(
                "unknown profile '{value}'; expected no_pushdown, default, pruning, \
                 pushdown, pushdown_no_repartition, or pushdown_no_reorder"
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
            Self::PushdownNoRepartition => "pushdown_no_repartition",
            Self::PushdownNoReorder => "pushdown_no_reorder",
        }
    }
}

struct RunRequest<'a> {
    operation: Operation,
    mode: MorselMode,
    mode_str: &'a str,
    profile: ScanProfile,
    workers: usize,
    batch_size: usize,
    collect_metrics: bool,
    data_dir: &'a Path,
    tables: &'a [String],
    sql: &'a str,
    warmups: usize,
    repeats: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 9 || args.len() > 11 {
        return Err(
            "usage: morsel_tpch_profile_runner <query|measure|measure_warm|measure_exec_loop> \
             <off|required> <profile> <workers> <batch_size> <metrics:on|off> \
             <data_dir> <tables_csv> <sql> [warmups] [repeats]"
                .into(),
        );
    }
    let operation = match args[0].as_str() {
        "query" => Operation::Query,
        "measure" => Operation::Measure,
        "measure_warm" => Operation::MeasureWarm,
        "measure_exec_loop" => Operation::MeasureExecLoop,
        other => return Err(format!("unknown operation '{other}'").into()),
    };
    let mode_str = args[1].as_str();
    let mode = match mode_str {
        "off" => MorselMode::Off,
        "required" => MorselMode::Required,
        other => return Err(format!("expected off or required, got '{other}'").into()),
    };
    let profile = ScanProfile::parse(&args[2])?;
    let workers: usize = args[3].parse()?;
    let batch_size: usize = args[4].parse()?;
    let collect_metrics = match args[5].as_str() {
        "on" => true,
        "off" => false,
        other => return Err(format!("expected metrics on|off, got '{other}'").into()),
    };
    let data_dir = Path::new(&args[6]);
    let tables: Vec<String> = args[7]
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect();
    if tables.is_empty() {
        return Err("tables_csv must contain at least one table name".into());
    }
    let sql = args[8].as_str();
    let warmups: usize = args.get(9).map_or(Ok(1), |v| v.parse())?;
    let repeats: usize = args.get(10).map_or(Ok(7), |v| v.parse())?;

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
        batch_size,
        collect_metrics,
        data_dir,
        tables: &tables,
        sql,
        warmups,
        repeats,
    }))
}

fn build_config(
    mode: MorselMode,
    profile: ScanProfile,
    workers: usize,
    batch_size: usize,
    metrics: Option<Arc<MorselMetrics>>,
) -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_batch_size(batch_size)
        .with_target_partitions(workers);
    if let Some(m) = metrics {
        config = config.with_extension(m);
    }
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
            if matches!(mode, MorselMode::Required) {
                config = config.with_repartition_file_scans(false);
            }
            config.options_mut().execution.morsel_mode = mode;
        }
        ScanProfile::Pruning => {
            config = config.with_repartition_file_scans(false);
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pushdown_filters = false;
        }
        ScanProfile::Pushdown => {
            if matches!(mode, MorselMode::Required) {
                config = config.with_repartition_file_scans(false);
            }
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pushdown_filters = true;
            execution.parquet.reorder_filters = true;
        }
        ScanProfile::PushdownNoRepartition => {
            config = config.with_repartition_file_scans(false);
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pushdown_filters = true;
            execution.parquet.reorder_filters = true;
        }
        ScanProfile::PushdownNoReorder => {
            config = config.with_repartition_file_scans(false);
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pushdown_filters = true;
            execution.parquet.reorder_filters = false;
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

struct DrainOutcome {
    output_rows: usize,
    output_batches: usize,
    first_output_seconds: Option<f64>,
    first_8192_seconds: Option<f64>,
    exec_seconds: f64,
    exec_cpu_seconds: f64,
}

async fn drain_plan(
    plan: &Arc<dyn ExecutionPlan>,
    task_ctx: Arc<datafusion::execution::TaskContext>,
    clock_origin: Instant,
    write_ipc: bool,
) -> Result<DrainOutcome, Box<dyn Error>> {
    let exec_start = Instant::now();
    let cpu_start = cpu_seconds();
    let mut stream = execute_stream(Arc::clone(plan), task_ctx)?;
    let mut writer = if write_ipc {
        Some(StreamWriter::try_new(io::stdout().lock(), &plan.schema())?)
    } else {
        None
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
            let reached_8192 = before < 8192 && output_rows >= 8192;
            if output_batches == 1 || reached_8192 {
                let elapsed = clock_origin.elapsed().as_secs_f64();
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
    }
    let exec_seconds = exec_start.elapsed().as_secs_f64();
    let exec_cpu_seconds = cpu_seconds() - cpu_start;
    drop(stream);
    Ok(DrainOutcome {
        output_rows,
        output_batches,
        first_output_seconds,
        first_8192_seconds,
        exec_seconds,
        exec_cpu_seconds,
    })
}

fn take_morsel_runs(metrics: Option<&Arc<MorselMetrics>>) -> Vec<Vec<Value>> {
    let Some(metrics) = metrics else {
        return Vec::new();
    };
    metrics
        .take_runs()
        .iter()
        .map(|run| {
            run.iter()
                .map(|m| {
                    json!({
                        "name": m.value().name(),
                        "worker": m.labels().iter().find(|l| l.name() == "worker").map(|l| l.value()),
                        "value": m.value().as_usize(),
                    })
                })
                .collect()
        })
        .collect()
}

fn collect_plan_metrics(plan: &dyn ExecutionPlan, out: &mut BTreeMap<String, usize>) {
    if let Some(metrics) = plan.metrics() {
        for m in metrics.iter() {
            let name = m.value().name();
            *out.entry(name.to_owned()).or_insert(0) += m.value().as_usize();
        }
    }
    for child in plan.children() {
        collect_plan_metrics(child.as_ref(), out);
    }
}

fn plan_metrics_map(plan: &dyn ExecutionPlan) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    collect_plan_metrics(plan, &mut map);
    map
}

async fn run(req: RunRequest<'_>) -> Result<(), Box<dyn Error>> {
    let RunRequest {
        operation,
        mode,
        mode_str,
        profile,
        workers,
        batch_size,
        collect_metrics,
        data_dir,
        tables,
        sql,
        warmups,
        repeats,
    } = req;
    let metrics = collect_metrics.then(|| Arc::new(MorselMetrics::default()));
    let ctx = SessionContext::new_with_config(build_config(
        mode,
        profile,
        workers,
        batch_size,
        metrics.clone(),
    ));

    match operation {
        Operation::Query => {
            register_tables(&ctx, data_dir, tables).await?;
            let df = ctx.sql(sql).await?;
            let task_ctx = Arc::new(df.task_ctx());
            let plan = df.create_physical_plan().await?;
            drain_plan(&plan, task_ctx, Instant::now(), true).await?;
            return Ok(());
        }
        Operation::Measure => {
            let start = Instant::now();
            let cpu_start = cpu_seconds();
            register_tables(&ctx, data_dir, tables).await?;
            let register_seconds = start.elapsed().as_secs_f64();
            let df = ctx.sql(sql).await?;
            let task_ctx = Arc::new(df.task_ctx());
            let plan = df.create_physical_plan().await?;
            let plan_seconds = start.elapsed().as_secs_f64() - register_seconds;
            let outcome = drain_plan(&plan, task_ctx, start, false).await?;
            let seconds = start.elapsed().as_secs_f64();
            let total_cpu_seconds = cpu_seconds() - cpu_start;
            let pmetrics = plan_metrics_map(plan.as_ref());
            let morsel_runs = take_morsel_runs(metrics.as_ref());
            println!(
                "{}",
                json!({
                    "build_revision": env!("MORSEL_BENCH_REVISION"),
                    "operation": "measure",
                    "mode": mode_str,
                    "profile": profile.as_str(),
                    "workers": workers,
                    "batch_size": batch_size,
                    "collect_metrics": collect_metrics,
                    "output_rows": outcome.output_rows,
                    "output_batches": outcome.output_batches,
                    "register_seconds": register_seconds,
                    "plan_seconds": plan_seconds,
                    "exec_seconds": outcome.exec_seconds,
                    "first_output_seconds": outcome.first_output_seconds,
                    "first_8192_seconds": outcome.first_8192_seconds,
                    "seconds": seconds,
                    "cpu_seconds": total_cpu_seconds,
                    "exec_cpu_seconds": outcome.exec_cpu_seconds,
                    "plan": displayable(plan.as_ref()).indent(false).to_string(),
                    "parquet_morsels_claimed": pmetrics.get("parquet_morsels_claimed").copied().unwrap_or(0),
                    "count_finalizations": pmetrics.get("count_finalizations").copied().unwrap_or(0),
                    "bytes_scanned": pmetrics.get("bytes_scanned").copied().unwrap_or(0),
                    "plan_metrics": pmetrics,
                    "morsel_runs": morsel_runs,
                })
            );
        }
        Operation::MeasureWarm => {
            let reg_start = Instant::now();
            register_tables(&ctx, data_dir, tables).await?;
            let register_seconds = reg_start.elapsed().as_secs_f64();
            for _ in 0..warmups {
                let df = ctx.sql(sql).await?;
                let task_ctx = Arc::new(df.task_ctx());
                let plan = df.create_physical_plan().await?;
                drain_plan(&plan, task_ctx, Instant::now(), false).await?;
                let _ = take_morsel_runs(metrics.as_ref());
            }
            let mut iterations = Vec::with_capacity(repeats);
            let mut last_plan_str = String::new();
            for _ in 0..repeats {
                let iter_start = Instant::now();
                let df = ctx.sql(sql).await?;
                let task_ctx = Arc::new(df.task_ctx());
                let plan = df.create_physical_plan().await?;
                let plan_seconds = iter_start.elapsed().as_secs_f64();
                let outcome = drain_plan(&plan, task_ctx, iter_start, false).await?;
                let total_seconds = iter_start.elapsed().as_secs_f64();
                let pmetrics = plan_metrics_map(plan.as_ref());
                let morsel_runs = take_morsel_runs(metrics.as_ref());
                last_plan_str = displayable(plan.as_ref()).indent(false).to_string();
                iterations.push(json!({
                    "plan_seconds": plan_seconds,
                    "exec_seconds": outcome.exec_seconds,
                    "total_seconds": total_seconds,
                    "exec_cpu_seconds": outcome.exec_cpu_seconds,
                    "output_rows": outcome.output_rows,
                    "output_batches": outcome.output_batches,
                    "first_output_seconds": outcome.first_output_seconds,
                    "first_8192_seconds": outcome.first_8192_seconds,
                    "parquet_morsels_claimed": pmetrics.get("parquet_morsels_claimed").copied().unwrap_or(0),
                    "count_finalizations": pmetrics.get("count_finalizations").copied().unwrap_or(0),
                    "bytes_scanned": pmetrics.get("bytes_scanned").copied().unwrap_or(0),
                    "plan_metrics": pmetrics,
                    "morsel_runs": morsel_runs,
                }));
            }
            println!(
                "{}",
                json!({
                    "build_revision": env!("MORSEL_BENCH_REVISION"),
                    "operation": "measure_warm",
                    "mode": mode_str,
                    "profile": profile.as_str(),
                    "workers": workers,
                    "batch_size": batch_size,
                    "collect_metrics": collect_metrics,
                    "register_seconds": register_seconds,
                    "plan": last_plan_str,
                    "iterations": iterations,
                })
            );
        }
        Operation::MeasureExecLoop => {
            let reg_start = Instant::now();
            register_tables(&ctx, data_dir, tables).await?;
            let register_seconds = reg_start.elapsed().as_secs_f64();
            let plan_start = Instant::now();
            let df = ctx.sql(sql).await?;
            let plan = df.create_physical_plan().await?;
            let plan_seconds = plan_start.elapsed().as_secs_f64();
            for _ in 0..warmups {
                let task_ctx = ctx.task_ctx();
                drain_plan(&plan, task_ctx, Instant::now(), false).await?;
                let _ = take_morsel_runs(metrics.as_ref());
            }
            let loop_start = Instant::now();
            let loop_cpu_start = cpu_seconds();
            let mut iterations = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let task_ctx = ctx.task_ctx();
                let outcome = drain_plan(&plan, task_ctx, Instant::now(), false).await?;
                let morsel_runs = take_morsel_runs(metrics.as_ref());
                iterations.push(json!({
                    "exec_seconds": outcome.exec_seconds,
                    "exec_cpu_seconds": outcome.exec_cpu_seconds,
                    "output_rows": outcome.output_rows,
                    "output_batches": outcome.output_batches,
                    "first_output_seconds": outcome.first_output_seconds,
                    "first_8192_seconds": outcome.first_8192_seconds,
                    "morsel_runs": morsel_runs,
                }));
            }
            let loop_wall_seconds = loop_start.elapsed().as_secs_f64();
            let loop_cpu_seconds = cpu_seconds() - loop_cpu_start;
            let pmetrics = plan_metrics_map(plan.as_ref());
            println!(
                "{}",
                json!({
                    "build_revision": env!("MORSEL_BENCH_REVISION"),
                    "operation": "measure_exec_loop",
                    "mode": mode_str,
                    "profile": profile.as_str(),
                    "workers": workers,
                    "batch_size": batch_size,
                    "collect_metrics": collect_metrics,
                    "register_seconds": register_seconds,
                    "plan_seconds": plan_seconds,
                    "loop_wall_seconds": loop_wall_seconds,
                    "loop_cpu_seconds": loop_cpu_seconds,
                    "plan": displayable(plan.as_ref()).indent(false).to_string(),
                    "plan_metrics": pmetrics,
                    "iterations": iterations,
                })
            );
        }
    }
    Ok(())
}

// Linux CLOCK_PROCESS_CPUTIME_ID (2): process CPU across async and blocking threads.
fn cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(id: i32, tp: *mut Timespec) -> i32;
    }
    let mut tp = Timespec { sec: 0, nsec: 0 };
    assert_eq!(unsafe { clock_gettime(2, &mut tp) }, 0);
    tp.sec as f64 + tp.nsec as f64 / 1e9
}
