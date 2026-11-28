// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![recursion_limit = "256"]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use arrow::ipc::writer::StreamWriter;
use datafusion::common::config::MorselMode;
use datafusion::common::instant::Instant;
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
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

fn memory_limit_bytes() -> Result<Option<usize>, Box<dyn Error>> {
    if let Ok(raw) = std::env::var("MORSEL_MEMORY_LIMIT_BYTES") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let bytes: usize = trimmed.parse().map_err(|err| {
                format!("invalid MORSEL_MEMORY_LIMIT_BYTES '{trimmed}': {err}")
            })?;
            if bytes == 0 {
                return Err("MORSEL_MEMORY_LIMIT_BYTES must be positive".into());
            }
            return Ok(Some(bytes));
        }
    }
    if let Ok(raw) = std::env::var("MORSEL_MEMORY_LIMIT_MB") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let mb: usize = trimmed.parse().map_err(|err| {
                format!("invalid MORSEL_MEMORY_LIMIT_MB '{trimmed}': {err}")
            })?;
            if mb == 0 {
                return Err("MORSEL_MEMORY_LIMIT_MB must be positive".into());
            }
            let bytes = mb
                .checked_mul(1024 * 1024)
                .ok_or("MORSEL_MEMORY_LIMIT_MB overflow")?;
            return Ok(Some(bytes));
        }
    }
    Ok(None)
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
    statements: &'a [String],
    stmt_index: usize,
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

fn load_statements(sql_arg: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let raw = if let Some(path) = sql_arg.strip_prefix('@') {
        fs::read_to_string(path)
            .map_err(|err| format!("failed to read SQL file '{path}': {err}"))?
    } else {
        sql_arg.to_owned()
    };
    let trimmed = raw.trim();
    if trimmed.starts_with('[')
        && let Ok(stmts) = serde_json::from_str::<Vec<String>>(trimmed)
    {
        let non_empty: Vec<String> = stmts
            .into_iter()
            .map(|s| s.trim().trim_end_matches(';').trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        if non_empty.is_empty() {
            return Err("JSON statement list must not be empty".into());
        }
        return Ok(non_empty);
    }
    let stmt = trimmed.trim_end_matches(';').trim().to_owned();
    if stmt.is_empty() {
        return Err("SQL statement must not be empty".into());
    }
    Ok(vec![stmt])
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (operation, mode, profile, workers, data_dir, tables_csv, sql_arg, stmt_index) =
        match args.as_slice() {
            [op, mode, prof, w, dir, tables, sql] => {
                (op, mode, prof, w, dir, tables, sql, 0usize)
            }
            [op, mode, prof, w, dir, tables, sql, idx] => {
                (op, mode, prof, w, dir, tables, sql, idx.parse::<usize>()?)
            }
            _ => {
                return Err(
                    "usage: morsel_tpcds_runner <query|measure|plan> <off|required> <no_pushdown|default|pruning|pushdown|pushdown_no_repartition> <workers> <data_dir> <tables_csv> <sql|@json_file> [stmt_index]"
                        .into(),
                );
            }
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
    let statements = load_statements(sql_arg)?;
    if matches!(operation, Operation::Query) && stmt_index >= statements.len() {
        return Err(format!(
            "stmt_index {stmt_index} out of bounds for {} statements",
            statements.len()
        )
        .into());
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
        statements: &statements,
        stmt_index,
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

async fn run_plan(
    ctx: &SessionContext,
    mode_str: &str,
    profile: ScanProfile,
    statements: &[String],
) -> Result<(), Box<dyn Error>> {
    let mut stmt_reports = Vec::with_capacity(statements.len());
    let mut all_admitted = true;
    let mut first_error: Option<String> = None;
    let mut joined_plans = Vec::with_capacity(statements.len());

    for (idx, sql) in statements.iter().enumerate() {
        match ctx.sql(sql).await {
            Ok(df) => {
                let logical_plan = df
                    .clone()
                    .into_optimized_plan()
                    .ok()
                    .map(|p| p.display_indent().to_string());
                match df.create_physical_plan().await {
                    Ok(plan) => {
                        let plan_str =
                            displayable(plan.as_ref()).indent(false).to_string();
                        joined_plans.push(plan_str.clone());
                        stmt_reports.push(json!({
                            "index": idx,
                            "admitted": true,
                            "logical_plan": logical_plan,
                            "plan": plan_str,
                            "error": null,
                        }));
                    }
                    Err(err) => {
                        let err_str = err.to_string();
                        all_admitted = false;
                        if first_error.is_none() {
                            first_error = Some(err_str.clone());
                        }
                        stmt_reports.push(json!({
                            "index": idx,
                            "admitted": false,
                            "logical_plan": logical_plan,
                            "plan": null,
                            "error": err_str,
                        }));
                    }
                }
            }
            Err(err) => {
                let err_str = err.to_string();
                all_admitted = false;
                if first_error.is_none() {
                    first_error = Some(err_str.clone());
                }
                stmt_reports.push(json!({
                    "index": idx,
                    "admitted": false,
                    "logical_plan": null,
                    "plan": null,
                    "error": err_str,
                }));
            }
        }
    }

    let outcome = json!({
        "build_revision": env!("MORSEL_BENCH_REVISION"),
        "mode": mode_str,
        "profile": profile.as_str(),
        "statement_count": statements.len(),
        "admitted": all_admitted,
        "plan": if all_admitted { Some(joined_plans.join("\n---\n")) } else { None },
        "error": first_error,
        "statements": stmt_reports,
    });
    println!("{outcome}");
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
        statements,
        stmt_index,
    } = req;
    let collect_metrics =
        matches!(operation, Operation::Measure) && env_bool("MORSEL_METRICS", true);
    let metrics = collect_metrics.then(|| Arc::new(MorselMetrics::default()));
    let config = build_config(mode, profile, workers, metrics.clone());
    let mem_limit = memory_limit_bytes()?;
    let ctx = if let Some(limit_bytes) = mem_limit {
        let runtime_env = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(limit_bytes)))
            .with_disk_manager_builder(
                DiskManagerBuilder::default().with_mode(DiskManagerMode::OsTmpDirectory),
            )
            .build_arc()?;
        SessionContext::new_with_config_rt(config, runtime_env)
    } else {
        SessionContext::new_with_config(config)
    };

    if matches!(operation, Operation::Plan) {
        register_tables(&ctx, data_dir, tables).await?;
        return run_plan(&ctx, mode_str, profile, statements).await;
    }

    if matches!(operation, Operation::Query) {
        register_tables(&ctx, data_dir, tables).await?;
        let df = ctx.sql(&statements[stmt_index]).await?;
        let task_ctx = Arc::new(df.task_ctx());
        let plan = df.create_physical_plan().await?;
        let mut stream = execute_stream(Arc::clone(&plan), task_ctx)?;
        let mut writer = StreamWriter::try_new(io::stdout().lock(), &plan.schema())?;
        while let Some(batch) = stream.try_next().await? {
            if batch.num_rows() != 0 {
                writer.write(&batch)?;
            }
        }
        writer.finish()?;
        return Ok(());
    }

    // Operation::Measure: start before registration so schema inference from Parquet footers
    // matches DuckDB (lazy footer reads during query) and ClickHouse (CREATE VIEW + SELECT).
    let start = Instant::now();
    register_tables(&ctx, data_dir, tables).await?;
    let register_seconds = start.elapsed().as_secs_f64();

    let mut plan_seconds = 0.0f64;
    let mut exec_seconds = 0.0f64;
    let mut exec_cpu_seconds_opt = Some(0.0f64);
    let mut first_output_seconds = None;
    let mut first_8192_seconds = None;
    let mut output_rows = 0usize;
    let mut output_batches = 0usize;
    let mut statement_rows = Vec::with_capacity(statements.len());
    let mut executed_plans: Vec<Arc<dyn ExecutionPlan>> =
        Vec::with_capacity(statements.len());

    for sql in statements {
        let p_start = Instant::now();
        let df = ctx.sql(sql).await?;
        let task_ctx = Arc::new(df.task_ctx());
        let plan = df.create_physical_plan().await?;
        plan_seconds += p_start.elapsed().as_secs_f64();

        let e_start = Instant::now();
        let cpu_before = process_cpu_seconds();
        let mut stream = execute_stream(Arc::clone(&plan), task_ctx)?;
        let mut stmt_rows = 0usize;
        while let Some(batch) = stream.try_next().await? {
            if batch.num_rows() != 0 {
                let before = output_rows;
                output_rows += batch.num_rows();
                stmt_rows += batch.num_rows();
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
            }
        }
        drop(stream);
        exec_seconds += e_start.elapsed().as_secs_f64();
        match (exec_cpu_seconds_opt, cpu_before, process_cpu_seconds()) {
            (Some(acc), Some(b), Some(a)) => exec_cpu_seconds_opt = Some(acc + (a - b)),
            _ => exec_cpu_seconds_opt = None,
        }
        statement_rows.push(stmt_rows);
        executed_plans.push(plan);
    }
    let seconds = start.elapsed().as_secs_f64();

    let native = match (mode, metrics.as_ref()) {
        (MorselMode::Required, Some(metrics)) => {
            // Drain native reports after the timer has stopped; give spawned
            // coordinator tasks a brief window to finish dropping RunMetrics
            // (e.g., on empty-build short-circuits).
            let mut runs = metrics.take_runs();
            let wait_start = Instant::now();
            while runs.len() < statements.len() && wait_start.elapsed().as_millis() < 200
            {
                tokio::task::yield_now().await;
                let more = metrics.take_runs();
                if more.is_empty() {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                } else {
                    runs.extend(more);
                }
            }
            if runs.len() != statements.len() {
                return Err(format!(
                    "unexpected number of native reports: {} (expected {})",
                    runs.len(),
                    statements.len()
                )
                .into());
            }
            Some(NativeReport::from_runs(&runs, workers)?)
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
    let mut plan_strings = Vec::with_capacity(executed_plans.len());
    for plan in &executed_plans {
        collect_operator_compute_ms(plan.as_ref(), &mut operator_compute_ms);
        plan_strings.push(displayable(plan.as_ref()).indent(false).to_string());
    }

    let sum_metric = |name: &str| -> usize {
        executed_plans
            .iter()
            .map(|p| metric(p.as_ref(), name))
            .sum()
    };
    let sum_spill = |extract: fn(&MetricsSet) -> Option<usize>| -> usize {
        executed_plans
            .iter()
            .map(|p| spill_metric(p.as_ref(), extract))
            .sum()
    };

    println!(
        "{}",
        json!({
            "build_revision": env!("MORSEL_BENCH_REVISION"),
            "mode": mode_str,
            "profile": profile.as_str(),
            "statement_count": statements.len(),
            "statement_rows": statement_rows,
            "output_rows": output_rows,
            "output_batches": output_batches,
            "register_seconds": register_seconds,
            "plan_seconds": plan_seconds,
            "exec_seconds": exec_seconds,
            "exec_cpu_seconds": exec_cpu_seconds_opt,
            "first_output_seconds": first_output_seconds,
            "first_8192_seconds": first_8192_seconds,
            "seconds": seconds,
            "plan": plan_strings.join("\n---\n"),
            "plans": plan_strings,
            "parquet_morsels_claimed": sum_metric("parquet_morsels_claimed"),
            "count_finalizations": sum_metric("count_finalizations"),
            "group_count_finalizations": sum_metric("group_count_finalizations"),
            "aggregate_finalizations": sum_metric("aggregate_finalizations"),
            "join_build_finalizations": sum_metric("join_build_finalizations"),
            "join_empty_build_short_circuits": sum_metric("join_empty_build_short_circuits"),
            "join_null_aware_short_circuits": sum_metric("join_null_aware_short_circuits"),
            "join_dynamic_filters_pushed": sum_metric("join_dynamic_filters_pushed"),
            "join_dynamic_filter_short_circuits": sum_metric("join_dynamic_filter_short_circuits"),
            "adaptive_row_filter_rgs": sum_metric("adaptive_row_filter_rgs"),
            "adaptive_bulk_decode_rgs": sum_metric("adaptive_bulk_decode_rgs"),
            "adaptive_fast_path_bulk_rgs": sum_metric("adaptive_fast_path_bulk_rgs"),
            "adaptive_mode_switches": sum_metric("adaptive_mode_switches"),
            "sort_finalizations": sum_metric("sort_finalizations"),
            "topn_finalizations": sum_metric("topn_finalizations"),
            "bytes_scanned": sum_metric("bytes_scanned"),
            "memory_limit_bytes": mem_limit,
            "spill_count": sum_spill(|m| m.spill_count()),
            "spilled_bytes": sum_spill(|m| m.spilled_bytes()),
            "spilled_rows": sum_spill(|m| m.spilled_rows()),
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
    fn from_runs(runs: &[MetricsSet], workers: usize) -> Result<Self, Box<dyn Error>> {
        let mut combined = Self {
            claims: vec![0; workers],
            completions: vec![0; workers],
            decode_ms: 0.0,
            process_ms: 0.0,
            ready_wait_ms: 0.0,
            output_wait_ms: 0.0,
            pending_io_ms: 0.0,
            join_combine_ms: 0.0,
            join_finalize_ms: 0.0,
            sort_finalize_ms: 0.0,
            topn_finalize_ms: 0.0,
            peak_ready_depth: 0,
            peak_output_depth: 0,
            pipelines: BTreeMap::new(),
        };
        let multi = runs.len() > 1;
        for (idx, run) in runs.iter().enumerate() {
            let claims = worker_counts(run, "morsels_claimed", workers)?;
            let completions = worker_counts(run, "morsels_completed", workers)?;
            for (dst, src) in combined.claims.iter_mut().zip(claims) {
                *dst += src;
            }
            for (dst, src) in combined.completions.iter_mut().zip(completions) {
                *dst += src;
            }
            combined.decode_ms += run_time_ms(run, "morsel_decode_work");
            combined.process_ms += run_time_ms(run, "morsel_process_work");
            combined.ready_wait_ms += run_time_ms(run, "morsel_ready_wait");
            combined.output_wait_ms += run_time_ms(run, "morsel_output_wait");
            combined.pending_io_ms += run_time_ms(run, "morsel_pending_io");
            combined.join_combine_ms += run_time_ms(run, "morsel_join_combine_work");
            combined.join_finalize_ms += run_time_ms(run, "morsel_join_finalize_work");
            combined.sort_finalize_ms += run_time_ms(run, "morsel_sort_finalize_work");
            combined.topn_finalize_ms += run_time_ms(run, "morsel_topn_finalize_work");
            combined.peak_ready_depth = combined
                .peak_ready_depth
                .max(run_usize(run, "morsel_peak_ready_depth"));
            combined.peak_output_depth = combined
                .peak_output_depth
                .max(run_usize(run, "morsel_peak_output_depth"));
            for (pname, report) in collect_pipeline_reports(run, workers)? {
                let key = if multi {
                    format!("s{idx}:{pname}")
                } else {
                    pname
                };
                combined.pipelines.insert(key, report);
            }
        }
        Ok(combined)
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
    let mut counts = vec![0usize; workers];
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
        *count += metric.value().as_usize();
    }
    Ok(counts)
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

fn spill_metric<F>(plan: &dyn ExecutionPlan, extract: F) -> usize
where
    F: Fn(&MetricsSet) -> Option<usize> + Copy,
{
    let own = plan
        .metrics()
        .and_then(|metrics| extract(&metrics))
        .unwrap_or(0);
    own + plan
        .children()
        .iter()
        .map(|child| spill_metric(child.as_ref(), extract))
        .sum::<usize>()
}
