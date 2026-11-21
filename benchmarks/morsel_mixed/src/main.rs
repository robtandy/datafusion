// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

mod run;
mod workload;

use std::io::{self, Write};
use std::time::Duration;

use anyhow::{Result, bail, ensure};
use clap::{Parser, ValueEnum};
use datafusion::common::runtime::SpawnedTask;
use serde::Serialize;
use tokio::runtime::Builder;

use workload::{Query, QueryMetadata};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Backend {
    Off,
    Required,
}

#[derive(Debug, Parser, Serialize)]
#[command(
    about = "Measure DataFusion small-query latency with and without shared-runtime scans"
)]
struct Args {
    /// Require native morsels, or use only stock execution.
    #[arg(long, value_enum, default_value = "off")]
    morsel_mode: Backend,
    /// Tokio async worker threads, excluding the arrival driver.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    workers: u32,
    /// Tokio blocking thread limit (separate from async workers).
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    blocking_threads: u32,
    /// DataFusion target partitions for both sessions and generated tables.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    partitions: u32,
    #[arg(long, default_value_t = 8192, value_parser = clap::value_parser!(u32).range(1..=65536))]
    batch_size: u32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(1..=64_000_000))]
    small_rows: u32,
    #[arg(long, default_value_t = 4_194_304, value_parser = clap::value_parser!(u32).range(1..=64_000_000))]
    large_rows: u32,
    /// Optional Parquet file/directory to register as `small`.
    #[arg(long)]
    small_fixture: Option<String>,
    /// Optional Parquet file/directory to register as `large`.
    #[arg(long)]
    large_fixture: Option<String>,
    /// Override foreground SQL (default scans and counts matches in `small`).
    #[arg(long)]
    small_sql: Option<String>,
    /// Override background SQL (default scans and sums a computed value in `large`).
    #[arg(long)]
    large_sql: Option<String>,
    /// Expected number of result rows (alternative to --small-expected-scalar).
    #[arg(long, conflicts_with = "small_expected_scalar")]
    small_expected_rows: Option<usize>,
    /// Expected single BIGINT result (e.g. COUNT(*)).
    #[arg(long, allow_negative_numbers = true)]
    small_expected_scalar: Option<i64>,
    /// Expected number of result rows (alternative to --large-expected-scalar).
    #[arg(long, conflicts_with = "large_expected_scalar")]
    large_expected_rows: Option<usize>,
    /// Expected single BIGINT result (e.g. COUNT(*)).
    #[arg(long, allow_negative_numbers = true)]
    large_expected_scalar: Option<i64>,
    /// Concurrent background lanes, one scan in flight per lane.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=32))]
    background_lanes: u32,
    /// Number of waiting foreground requests allowed, excluding the active query.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..=1024))]
    queue_capacity: u32,
    /// Sequential warm-up queries per workload, before the first pair.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(0..=100))]
    warmup: u32,
    /// Number of isolated/mixed pairs, alternating which mode runs first.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=50))]
    repetitions: u32,
    /// Start the first pair with the mixed phase rather than the isolated phase.
    #[arg(long)]
    mixed_first: bool,
    /// Foreground requests scheduled per run (including drops).
    #[arg(long, default_value_t = 200, value_parser = clap::value_parser!(u32).range(1..=50_000))]
    requests: u32,
    /// Foreground interarrival time.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..=60_000))]
    interval_ms: u64,
    /// Foreground latency above this threshold is marked overdue.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u64).range(1..=600_000))]
    overdue_ms: u64,
    /// Cooperative query safety timeout; includes foreground queueing.
    #[arg(long, default_value_t = 10_000, value_parser = clap::value_parser!(u64).range(1..=600_000))]
    timeout_ms: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.requests * args.repetitions <= 200_000,
        "too many samples"
    );
    ensure!(
        args.interval_ms * u64::from(args.requests) <= 3_600_000,
        "offer window exceeds one hour"
    );
    let runtime = Builder::new_multi_thread()
        .worker_threads(args.workers as usize)
        .max_blocking_threads(args.blocking_threads as usize)
        .enable_all()
        .build()?;
    runtime.block_on(benchmark(args))
}

#[derive(Serialize)]
struct Output<'a> {
    schema_version: u32,
    build_revision: Option<&'static str>,
    backend: &'static str,
    valid: bool,
    config: Args,
    small: &'a QueryMetadata,
    large: &'a QueryMetadata,
    runtime: RuntimeMetadata,
    argv: Vec<String>,
    runs: Vec<run::Report>,
}

#[derive(Serialize)]
struct RuntimeMetadata {
    morsel_mode: Backend,
    available_parallelism: Option<usize>,
    linux_cpus_allowed_list: Option<String>,
    driver_threads: usize,
}

async fn benchmark(args: Args) -> Result<()> {
    let timeout = Duration::from_millis(args.timeout_ms);
    let small = Query::new(&args, false).await?;
    let large = Query::new(&args, true).await?;
    for query in [&small, &large] {
        let query = query.clone();
        let warmup = args.warmup;
        SpawnedTask::spawn(async move { query.warm_up(warmup, timeout).await }).await??;
    }

    let mut runs = Vec::new();
    for repetition in 0..args.repetitions {
        let mixed_first = args.mixed_first ^ (repetition % 2 != 0);
        for mixed in [mixed_first, !mixed_first] {
            runs.push(
                run::measure(&args, repetition, &small, mixed.then_some(&large)).await?,
            );
        }
    }
    let valid = runs.iter().all(|run| run.valid);
    let morsel_mode = args.morsel_mode;
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    serde_json::to_writer(
        &mut stdout,
        &Output {
            schema_version: 1,
            build_revision: option_env!("MORSEL_BENCH_REVISION"),
            backend: match morsel_mode {
                Backend::Off => "datafusion_stock",
                Backend::Required => "datafusion_native",
            },
            valid,
            config: args,
            small: small.metadata(),
            large: large.metadata(),
            runtime: RuntimeMetadata {
                morsel_mode,
                available_parallelism: std::thread::available_parallelism()
                    .ok()
                    .map(|n| n.get()),
                linux_cpus_allowed_list: linux_cpu_list(),
                driver_threads: 1,
            },
            argv: std::env::args().collect(),
            runs,
        },
    )?;
    writeln!(stdout)?;
    stdout.flush()?;
    if !valid {
        bail!(
            "invalid run: see JSON for query failures, missing overlap, or scan limit exhaustion"
        );
    }
    Ok(())
}

fn linux_cpu_list() -> Option<String> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(|list| list.trim().to_owned())
}
