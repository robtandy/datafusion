// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! SQL execution-only A/B benchmark for the fork's in-memory morsel pipelines.
//! See `benchmarks/morsel_execution.md` for the protocol and baseline setup.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::instant::Instant;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::{ExecutionPlan, displayable, execute_stream};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::TryStreamExt;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tokio::runtime::Builder;

const BATCH_ROWS: usize = 4_096;
const BATCHES: usize = 2_048;
const SEED: u64 = 127;
const WARMUPS: usize = 5;
const REPETITIONS: usize = 31;
const MODE: &str = "datafusion.execution.morsel_mode";

#[derive(Clone, Copy)]
enum Query {
    Projection,
    Count,
}

impl Query {
    fn label(self) -> &'static str {
        match self {
            Self::Projection => "projection",
            Self::Count => "count",
        }
    }

    fn sql(self) -> &'static str {
        match self {
            Self::Projection => "SELECT id + 1 AS result FROM items WHERE score < 64",
            Self::Count => "SELECT COUNT(*) AS result FROM items WHERE score < 64",
        }
    }
}

fn fixture(batch_counts: &[usize]) -> Result<(Arc<MemTable>, Vec<bool>)> {
    let rows = batch_counts.iter().sum::<usize>() * BATCH_ROWS;
    let mut selected = Vec::with_capacity(rows);
    let mut rng = SmallRng::seed_from_u64(SEED);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]));
    let mut partitions = Vec::with_capacity(batch_counts.len());
    let mut start = 0;
    for &batches in batch_counts {
        let mut partition = Vec::with_capacity(batches);
        for _ in 0..batches {
            let ids = Int64Array::from_iter_values(start..start + BATCH_ROWS as i64);
            let scores = Int64Array::from_iter_values((0..BATCH_ROWS).map(|_| {
                let score = i64::from(rng.random::<u8>());
                selected.push(score < 64);
                score
            }));
            partition.push(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(ids), Arc::new(scores)],
            )?);
            start += BATCH_ROWS as i64;
        }
        partitions.push(partition);
    }
    Ok((Arc::new(MemTable::try_new(schema, partitions)?), selected))
}

async fn prepare(
    table: Arc<MemTable>,
    workers: usize,
    query: Query,
    mode: &str,
) -> Result<(SessionContext, Arc<dyn ExecutionPlan>)> {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(BATCH_ROWS);
    if mode == "apache-main" {
        assert!(!config.options().entries().iter().any(|e| e.key == MODE));
    } else {
        config.options_mut().set(MODE, mode)?;
    }
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("items", table)?;
    let plan = ctx.sql(query.sql()).await?.create_physical_plan().await?;
    let explain = displayable(plan.as_ref()).indent(false).to_string();
    eprintln!("{mode}, {workers} workers, {}:\n{explain}", query.label());
    if mode == "required" {
        assert_eq!(plan.name(), "MorselExec");
        assert!(
            explain.contains(&format!("workers={workers},")),
            "{explain}"
        );
        assert!(explain.contains("-> Filter["), "{explain}");
        assert_eq!(explain.contains("CountSink"), matches!(query, Query::Count));
    } else {
        assert!(!explain.contains("MorselExec"), "{explain}");
    }
    Ok((ctx, plan))
}

async fn verify(
    ctx: &SessionContext,
    plan: &Arc<dyn ExecutionPlan>,
    query: Query,
    selected: &[bool],
    output_rows: usize,
) -> Result<()> {
    let plan = reset_plan_states(Arc::clone(plan))?;
    let mut stream = execute_stream(plan, ctx.task_ctx())?;
    let mut remaining = selected.to_vec();
    let mut rows = 0;
    while let Some(batch) = stream.try_next().await? {
        assert_eq!(batch.num_columns(), 1);
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.null_count(), 0);
        for &value in values.values() {
            match query {
                Query::Projection => {
                    let id = usize::try_from(value - 1).unwrap();
                    assert!(
                        std::mem::take(&mut remaining[id]),
                        "unexpected or duplicate id {id}"
                    );
                }
                Query::Count => assert_eq!(value, output_rows as i64),
            }
        }
        rows += batch.num_rows();
    }
    assert_eq!(
        rows,
        if matches!(query, Query::Count) {
            1
        } else {
            output_rows
        }
    );
    Ok(())
}

async fn measure(
    ctx: &SessionContext,
    plan: &Arc<dyn ExecutionPlan>,
    expected_rows: usize,
) -> Result<(Duration, Duration)> {
    let plan = reset_plan_states(Arc::clone(plan))?;
    let task = ctx.task_ctx();
    let start = Instant::now();
    let mut stream = execute_stream(plan, task)?;
    let mut first = None;
    let mut rows = 0;
    while let Some(batch) = stream.try_next().await? {
        if first.is_none() && batch.num_rows() > 0 {
            first = Some(start.elapsed());
        }
        rows += black_box(batch).num_rows();
    }
    let total = start.elapsed();
    assert_eq!(rows, expected_rows);
    Ok((first.expect("expected a non-empty result"), total))
}

fn report(
    layout: &str,
    workers: usize,
    query: Query,
    mode: &str,
    rows: usize,
    samples: &[(Duration, Duration)],
) {
    let stats = |mut times: Vec<Duration>| {
        times.sort_unstable();
        let last = times.len() - 1;
        [times[last / 2], times[last / 10], times[last * 9 / 10]]
    };
    let first = stats(samples.iter().map(|&(first, _)| first).collect());
    let total = stats(samples.iter().map(|&(_, total)| total).collect());
    let ms = |[median, p10, p90]: [Duration; 3]| {
        format!(
            "{:.3} [{:.3}, {:.3}]",
            median.as_secs_f64() * 1e3,
            p10.as_secs_f64() * 1e3,
            p90.as_secs_f64() * 1e3
        )
    };
    let throughput = rows as f64 / total[0].as_secs_f64() / 1e6;
    println!(
        "| {layout} | {workers} | {} | {mode} | {throughput:.1} | {} | {} |",
        query.label(),
        ms(first),
        ms(total)
    );
}

async fn run_case(
    layout: &str,
    batch_counts: &[usize],
    modes: &[&str],
    verify_only: bool,
) -> Result<()> {
    let workers = batch_counts.len();
    let (table, selected) = fixture(batch_counts)?;
    let output_rows = selected.iter().filter(|&&keep| keep).count();
    eprintln!(
        "{layout}: batches per partition = {batch_counts:?}, selected rows = {output_rows}"
    );
    for query in [Query::Projection, Query::Count] {
        let mut plans: Vec<(SessionContext, Arc<dyn ExecutionPlan>)> = Vec::new();
        for &mode in modes {
            let (ctx, plan) = prepare(Arc::clone(&table), workers, query, mode).await?;
            if let Some((_, stock)) = plans.first() {
                assert_eq!(plan.schema(), stock.schema());
            }
            verify(&ctx, &plan, query, &selected, output_rows).await?;
            plans.push((ctx, plan));
        }
        if verify_only {
            continue;
        }
        let mut samples = vec![Vec::with_capacity(REPETITIONS); plans.len()];
        let expected_rows = if matches!(query, Query::Count) {
            1
        } else {
            output_rows
        };
        for rep in 0..WARMUPS + REPETITIONS {
            // Alternate which engine runs first, using the same resident inputs.
            for offset in 0..plans.len() {
                let index = (offset + rep) % plans.len();
                let (ctx, plan) = &plans[index];
                let sample = measure(ctx, plan, expected_rows).await?;
                if rep >= WARMUPS {
                    samples[index].push(sample);
                }
            }
        }
        for (&mode, samples) in modes.iter().zip(&samples) {
            report(layout, workers, query, mode, selected.len(), samples);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut verify_only = false;
    let mut bench = false;
    let mut modes: &[_] = &["off", "required"];
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--bench" => bench = true,
            "--test" | "--verify" => verify_only = true,
            "--apache-main" => modes = &["apache-main"],
            _ => panic!("unknown argument: {arg}"),
        }
    }
    // Cargo's test invocation (without --bench) runs just the small parity check.
    let verify_only = verify_only || !bench;
    let n = if verify_only { 16 } else { BATCHES };
    if !verify_only {
        println!(
            "| Layout | Workers | Query | Mode | Input Mrows/s | First ms | Total ms |"
        );
        println!("| --- | ---: | --- | --- | ---: | ---: | ---: |");
    }
    for (layout, partitions) in [
        ("uniform", vec![n]),
        ("uniform", vec![n / 4; 4]),
        ("skewed", vec![13 * n / 16, n / 16, n / 16, n / 16]),
    ] {
        let runtime = Builder::new_multi_thread()
            .worker_threads(partitions.len())
            .build()?;
        // The drain runs on the same worker pool as the query, not on a separate
        // block_on thread that could give either engine an extra CPU.
        runtime
            .block_on(runtime.spawn(async move {
                run_case(layout, &partitions, modes, verify_only).await
            }))
            .expect("benchmark task failed")?;
    }
    Ok(())
}
