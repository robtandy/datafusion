// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io;
use std::path::Path;
use std::sync::Arc;

use arrow::ipc::writer::StreamWriter;
use datafusion::common::config::MorselMode;
use datafusion::common::instant::Instant;
use datafusion::morsel::MorselMetrics;
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::{ExecutionPlan, displayable, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
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
    PushdownNoRepartition,
}

impl ScanProfile {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "no_pushdown" => Ok(Self::NoPushdown),
            "default" => Ok(Self::Default),
            "pruning" => Ok(Self::Pruning),
            "pushdown" => Ok(Self::Pushdown),
            "pushdown_no_repartition" => Ok(Self::PushdownNoRepartition),
            _ => Err(format!(
                "unknown profile '{value}'; expected no_pushdown, default, pruning, pushdown, or pushdown_no_repartition"
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
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [operation, mode, profile, workers, data_dir, tables_csv, sql] = args.as_slice()
    else {
        return Err(
            "usage: morsel_tpch_runner <query|measure|plan> <off|required> <no_pushdown|default|pruning|pushdown|pushdown_no_repartition> <workers> <data_dir> <tables_csv> <sql>"
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

fn build_config(
    mode: MorselMode,
    profile: ScanProfile,
    workers: usize,
    metrics: Option<Arc<MorselMetrics>>,
) -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_batch_size(BATCH_SIZE)
        .with_target_partitions(workers);
    if let Some(metrics) = metrics {
        config = config.with_extension(metrics);
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
            config.options_mut().execution.morsel_mode = mode;
        }
        ScanProfile::Pruning => {
            config = config.with_repartition_file_scans(false);
            let execution = &mut config.options_mut().execution;
            execution.morsel_mode = mode;
            execution.parquet.pushdown_filters = false;
        }
        ScanProfile::Pushdown => {
            if mode == MorselMode::Required {
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
    let collect_metrics =
        matches!(operation, Operation::Measure) && env_bool("MORSEL_METRICS", true);
    let metrics = collect_metrics.then(|| Arc::new(MorselMetrics::default()));
    let ctx = SessionContext::new_with_config(build_config(
        mode,
        profile,
        workers,
        metrics.clone(),
    ));

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
    let exec_cpu_start = process_cpu_seconds();
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
    let exec_cpu_seconds = match (exec_cpu_start, process_cpu_seconds()) {
        (Some(before), Some(after)) => Some(after - before),
        _ => None,
    };
    let seconds = start.elapsed().as_secs_f64();
    drop(stream);

    let native = match (mode, metrics.as_ref()) {
        (MorselMode::Required, Some(metrics)) => {
            let runs = metrics.take_runs();
            match runs.as_slice() {
                [run] => Some(NativeReport::from_run(run, workers)?),
                _ => {
                    return Err(format!(
                        "unexpected number of native reports: {}",
                        runs.len()
                    )
                    .into());
                }
            }
        }
        (MorselMode::Off, Some(metrics)) => {
            let runs = metrics.take_runs();
            if !runs.is_empty() {
                return Err(format!(
                    "unexpected native reports under morsel_mode=off: {}",
                    runs.len()
                )
                .into());
            }
            None
        }
        (MorselMode::Prefer, _) => unreachable!("runner only accepts off or required"),
        (_, None) => None,
    };
    let mut operator_compute_ms = BTreeMap::new();
    collect_operator_compute_ms(plan.as_ref(), &mut operator_compute_ms);

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
            "exec_cpu_seconds": exec_cpu_seconds,
            "first_output_seconds": first_output_seconds,
            "first_8192_seconds": first_8192_seconds,
            "seconds": seconds,
            "plan": displayable(plan.as_ref()).indent(false).to_string(),
            "parquet_morsels_claimed": metric(plan.as_ref(), "parquet_morsels_claimed"),
            "count_finalizations": metric(plan.as_ref(), "count_finalizations"),
            "aggregate_finalizations": metric(plan.as_ref(), "aggregate_finalizations"),
            "join_build_finalizations": metric(plan.as_ref(), "join_build_finalizations"),
            "sort_finalizations": metric(plan.as_ref(), "sort_finalizations"),
            "topn_finalizations": metric(plan.as_ref(), "topn_finalizations"),
            "bytes_scanned": metric(plan.as_ref(), "bytes_scanned"),
            "worker_claims": native.as_ref().map(|r| &r.claims),
            "worker_completions": native.as_ref().map(|r| &r.completions),
            "morsel_decode_work_ms": native.as_ref().map(|r| r.decode_ms),
            "morsel_process_work_ms": native.as_ref().map(|r| r.process_ms),
            "morsel_ready_wait_ms": native.as_ref().map(|r| r.ready_wait_ms),
            "morsel_output_wait_ms": native.as_ref().map(|r| r.output_wait_ms),
            "morsel_pending_io_ms": native.as_ref().map(|r| r.pending_io_ms),
            "morsel_join_combine_work_ms": native.as_ref().map(|r| r.join_combine_ms),
            "morsel_join_finalize_work_ms": native.as_ref().map(|r| r.join_finalize_ms),
            "morsel_sort_finalize_work_ms": native.as_ref().map(|r| r.sort_finalize_ms),
            "morsel_topn_finalize_work_ms": native.as_ref().map(|r| r.topn_finalize_ms),
            "morsel_peak_ready_depth": native.as_ref().map(|r| r.peak_ready_depth),
            "morsel_peak_output_depth": native.as_ref().map(|r| r.peak_output_depth),
            "morsel_pipelines": native.as_ref().map(|r| r.pipelines_json()),
            "operator_compute_ms": operator_compute_ms,
        })
    );
    Ok(())
}

struct PipelineReport {
    claims: Vec<usize>,
    completions: Vec<usize>,
    decode_ms: f64,
    process_ms: f64,
    ready_wait_ms: f64,
    output_wait_ms: f64,
    join_combine_ms: f64,
    join_finalize_ms: f64,
    sort_finalize_ms: f64,
    topn_finalize_ms: f64,
}

impl PipelineReport {
    fn new(workers: usize) -> Self {
        Self {
            claims: vec![0; workers],
            completions: vec![0; workers],
            decode_ms: 0.0,
            process_ms: 0.0,
            ready_wait_ms: 0.0,
            output_wait_ms: 0.0,
            join_combine_ms: 0.0,
            join_finalize_ms: 0.0,
            sort_finalize_ms: 0.0,
            topn_finalize_ms: 0.0,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        json!({
            "worker_claims": self.claims,
            "worker_completions": self.completions,
            "decode_work_ms": self.decode_ms,
            "process_work_ms": self.process_ms,
            "ready_wait_ms": self.ready_wait_ms,
            "output_wait_ms": self.output_wait_ms,
            "join_combine_work_ms": self.join_combine_ms,
            "join_finalize_work_ms": self.join_finalize_ms,
            "sort_finalize_work_ms": self.sort_finalize_ms,
            "topn_finalize_work_ms": self.topn_finalize_ms,
        })
    }
}

struct NativeReport {
    claims: Vec<usize>,
    completions: Vec<usize>,
    decode_ms: f64,
    process_ms: f64,
    ready_wait_ms: f64,
    output_wait_ms: f64,
    pending_io_ms: f64,
    join_combine_ms: f64,
    join_finalize_ms: f64,
    sort_finalize_ms: f64,
    topn_finalize_ms: f64,
    peak_ready_depth: usize,
    peak_output_depth: usize,
    pipelines: BTreeMap<String, PipelineReport>,
}

impl NativeReport {
    fn from_run(run: &MetricsSet, workers: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            claims: worker_counts(run, "morsels_claimed", workers)?,
            completions: worker_counts(run, "morsels_completed", workers)?,
            decode_ms: run_time_ms(run, "morsel_decode_work"),
            process_ms: run_time_ms(run, "morsel_process_work"),
            ready_wait_ms: run_time_ms(run, "morsel_ready_wait"),
            output_wait_ms: run_time_ms(run, "morsel_output_wait"),
            pending_io_ms: run_time_ms(run, "morsel_pending_io"),
            join_combine_ms: run_time_ms(run, "morsel_join_combine_work"),
            join_finalize_ms: run_time_ms(run, "morsel_join_finalize_work"),
            sort_finalize_ms: run_time_ms(run, "morsel_sort_finalize_work"),
            topn_finalize_ms: run_time_ms(run, "morsel_topn_finalize_work"),
            peak_ready_depth: run_usize(run, "morsel_peak_ready_depth"),
            peak_output_depth: run_usize(run, "morsel_peak_output_depth"),
            pipelines: collect_pipeline_reports(run, workers)?,
        })
    }

    fn pipelines_json(&self) -> BTreeMap<String, serde_json::Value> {
        self.pipelines
            .iter()
            .map(|(name, report)| (name.clone(), report.to_json()))
            .collect()
    }
}

fn collect_pipeline_reports(
    run: &MetricsSet,
    workers: usize,
) -> Result<BTreeMap<String, PipelineReport>, Box<dyn Error>> {
    let mut pipelines: BTreeMap<String, PipelineReport> = BTreeMap::new();
    let mut seen_worker_counts: BTreeSet<(&str, String, usize)> = BTreeSet::new();
    for metric in run.iter() {
        let name = metric.value().name();
        let pipeline_label = metric
            .labels()
            .iter()
            .find(|label| label.name() == "pipeline")
            .map(|label| label.value().to_owned());
        let worker_label = metric
            .labels()
            .iter()
            .find(|label| label.name() == "worker")
            .map(|label| label.value().parse::<usize>())
            .transpose()?;

        let is_pipeline_metric = matches!(
            name,
            "morsels_claimed"
                | "morsels_completed"
                | "morsel_decode_work"
                | "morsel_process_work"
                | "morsel_ready_wait"
                | "morsel_output_wait"
                | "morsel_join_combine_work"
                | "morsel_join_finalize_work"
                | "morsel_sort_finalize_work"
                | "morsel_topn_finalize_work"
        );
        if !is_pipeline_metric {
            continue;
        }
        let pipeline = pipeline_label.unwrap_or_else(|| "main".to_owned());
        let entry = pipelines
            .entry(pipeline.clone())
            .or_insert_with(|| PipelineReport::new(workers));
        match name {
            "morsels_claimed" => {
                let worker = worker_label.ok_or("missing worker label")?;
                if !seen_worker_counts.insert(("morsels_claimed", pipeline, worker)) {
                    return Err(
                        format!("duplicate morsels_claimed for worker {worker}").into()
                    );
                }
                *entry
                    .claims
                    .get_mut(worker)
                    .ok_or("worker index out of bounds")? += metric.value().as_usize();
            }
            "morsels_completed" => {
                let worker = worker_label.ok_or("missing worker label")?;
                if !seen_worker_counts.insert(("morsels_completed", pipeline, worker)) {
                    return Err(format!(
                        "duplicate morsels_completed for worker {worker}"
                    )
                    .into());
                }
                *entry
                    .completions
                    .get_mut(worker)
                    .ok_or("worker index out of bounds")? += metric.value().as_usize();
            }
            "morsel_decode_work" => {
                entry.decode_ms += metric.value().as_usize() as f64 / 1e6
            }
            "morsel_process_work" => {
                entry.process_ms += metric.value().as_usize() as f64 / 1e6
            }
            "morsel_ready_wait" => {
                entry.ready_wait_ms += metric.value().as_usize() as f64 / 1e6
            }
            "morsel_output_wait" => {
                entry.output_wait_ms += metric.value().as_usize() as f64 / 1e6
            }
            "morsel_join_combine_work" => {
                entry.join_combine_ms += metric.value().as_usize() as f64 / 1e6
            }
            "morsel_join_finalize_work" => {
                entry.join_finalize_ms += metric.value().as_usize() as f64 / 1e6
            }
            "morsel_sort_finalize_work" => {
                entry.sort_finalize_ms += metric.value().as_usize() as f64 / 1e6
            }
            "morsel_topn_finalize_work" => {
                entry.topn_finalize_ms += metric.value().as_usize() as f64 / 1e6
            }
            _ => unreachable!(),
        }
    }
    Ok(pipelines)
}

fn run_usize(run: &MetricsSet, name: &str) -> usize {
    run.sum_by_name(name).map_or(0, |v| v.as_usize())
}

fn run_time_ms(run: &MetricsSet, name: &str) -> f64 {
    run.sum_by_name(name)
        .map_or(0.0, |v| v.as_usize() as f64 / 1e6)
}

fn worker_counts(
    run: &MetricsSet,
    name: &str,
    workers: usize,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut seen: BTreeSet<(Option<String>, usize)> = BTreeSet::new();
    let mut counts = vec![None; workers];
    for metric in run.iter().filter(|metric| metric.value().name() == name) {
        let worker: usize = metric
            .labels()
            .iter()
            .find(|label| label.name() == "worker")
            .ok_or("missing worker label")?
            .value()
            .parse()?;
        let pipeline = metric
            .labels()
            .iter()
            .find(|label| label.name() == "pipeline")
            .map(|label| label.value().to_owned());
        if !seen.insert((pipeline.clone(), worker)) {
            return Err(format!(
                "duplicate {name} for worker {worker} (pipeline={pipeline:?})"
            )
            .into());
        }
        let count = counts.get_mut(worker).ok_or("worker index out of bounds")?;
        *count = Some(count.unwrap_or(0) + metric.value().as_usize());
    }
    counts
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| format!("missing {name} for one or more workers").into())
}

fn collect_operator_compute_ms(
    plan: &dyn ExecutionPlan,
    out: &mut BTreeMap<String, f64>,
) {
    if let Some(metrics) = plan.metrics()
        && let Some(elapsed) = metrics.elapsed_compute()
    {
        *out.entry(plan.name().to_owned()).or_insert(0.0) += elapsed as f64 / 1e6;
    }
    for child in plan.children() {
        collect_operator_compute_ms(child.as_ref(), out);
    }
}

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn process_cpu_seconds() -> Option<f64> {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(id: i32, tp: *mut Timespec) -> i32;
    }
    // Linux CLOCK_PROCESS_CPUTIME_ID (2): process CPU across all worker threads.
    let mut tp = Timespec { sec: 0, nsec: 0 };
    if unsafe { clock_gettime(2, &raw mut tp) } == 0 {
        Some(tp.sec as f64 + tp.nsec as f64 / 1e9)
    } else {
        None
    }
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
fn process_cpu_seconds() -> Option<f64> {
    None
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
