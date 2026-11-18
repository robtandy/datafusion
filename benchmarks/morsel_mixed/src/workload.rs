// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use datafusion::arrow::array::{Array, Int64Array, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::config::MorselMode;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SQLOptions;
use datafusion::physical_plan::execute_stream;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::StreamExt;
use serde::Serialize;
use tokio::sync::oneshot;
use tokio::time::{Instant, timeout_at};

use crate::Args;

const SMALL_SQL: &str = "SELECT COUNT(*) FROM small WHERE v % 17 = 0";
const LARGE_SQL: &str = "SELECT SUM((v * v + v * 13) % 1000033) FROM large";

#[derive(Clone)]
pub(crate) struct Query {
    ctx: SessionContext,
    metadata: QueryMetadata,
}

#[derive(Clone, Serialize)]
pub(crate) struct QueryMetadata {
    sql: String,
    fixture: Option<String>,
    expected: Expected,
    generated_rows: Option<u32>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Expected {
    Rows(usize),
    Scalar(i64),
}

pub(crate) struct Digest {
    rows: usize,
    scalar: Option<i128>,
}

pub(crate) enum Activity<'a> {
    Warmup,
    Foreground(&'a AtomicUsize),
    Background(&'a AtomicUsize, Option<oneshot::Sender<()>>),
}

#[derive(Default)]
pub(crate) struct Progress {
    pub(crate) plan_ready: Option<Instant>,
    pub(crate) execution_started: Option<Instant>,
    pub(crate) active_scans: usize,
}

impl Query {
    pub(crate) async fn new(args: &Args, large: bool) -> Result<Self> {
        let (table, default_sql, sql, fixture, rows, expected_rows, expected_scalar) =
            if large {
                (
                    "large",
                    LARGE_SQL,
                    &args.large_sql,
                    &args.large_fixture,
                    args.large_rows,
                    args.large_expected_rows,
                    args.large_expected_scalar,
                )
            } else {
                (
                    "small",
                    SMALL_SQL,
                    &args.small_sql,
                    &args.small_fixture,
                    args.small_rows,
                    args.small_expected_rows,
                    args.small_expected_scalar,
                )
            };
        let expectation = expected_rows
            .map(Expected::Rows)
            .or(expected_scalar.map(Expected::Scalar));
        ensure!(
            expectation.is_some() || (sql.is_none() && fixture.is_none()),
            "custom {table} workload needs --{table}-expected-rows or --{table}-expected-scalar"
        );
        let mut config = SessionConfig::new()
            .with_target_partitions(args.partitions as usize)
            .with_batch_size(args.batch_size as usize);
        config.options_mut().execution.morsel_mode = MorselMode::Off;
        let ctx = SessionContext::new_with_config(config);
        let expected = if let Some(path) = fixture {
            ctx.register_parquet(table, path, ParquetReadOptions::default())
                .await?;
            expectation.expect("checked custom workload expectation")
        } else {
            let (mem, scalar) = generate(args, rows, large)?;
            ctx.register_table(table, Arc::new(mem))?;
            expectation.unwrap_or(Expected::Scalar(scalar))
        };
        Ok(Self {
            ctx,
            metadata: QueryMetadata {
                sql: sql.clone().unwrap_or_else(|| default_sql.to_owned()),
                fixture: fixture.clone(),
                expected,
                generated_rows: fixture.is_none().then_some(rows),
            },
        })
    }

    pub(crate) fn metadata(&self) -> &QueryMetadata {
        &self.metadata
    }

    pub(crate) async fn warm_up(&self, count: u32, timeout: Duration) -> Result<()> {
        for _ in 0..count {
            let result = self
                .run(
                    Instant::now() + timeout,
                    Activity::Warmup,
                    &mut Progress::default(),
                )
                .await;
            self.validate(result)
                .with_context(|| format!("warming up {}", self.metadata.sql))?;
        }
        Ok(())
    }

    pub(crate) async fn run(
        &self,
        deadline: Instant,
        activity: Activity<'_>,
        progress: &mut Progress,
    ) -> Result<Digest> {
        ensure!(
            Instant::now() < deadline,
            "request timed out before planning"
        );
        timeout_at(deadline, self.drain(activity, progress))
            .await
            .context("query timed out")?
    }

    async fn drain(
        &self,
        activity: Activity<'_>,
        progress: &mut Progress,
    ) -> Result<Digest> {
        let options = SQLOptions::new()
            .with_allow_ddl(false)
            .with_allow_dml(false)
            .with_allow_copy(false)
            .with_allow_statements(false);
        let df = self
            .ctx
            .sql_with_options(&self.metadata.sql, options)
            .await?;
        let task_ctx = Arc::new(df.task_ctx());
        let plan = df.create_physical_plan().await?;
        progress.plan_ready = Some(Instant::now());
        let mut stream = execute_stream(plan, task_ctx)?;
        let (_scan, ready) = match activity {
            Activity::Background(active, ready) => (Some(ActiveScan::new(active)), ready),
            Activity::Foreground(active) => {
                progress.active_scans = active.load(Ordering::SeqCst);
                (None, None)
            }
            Activity::Warmup => (None, None),
        };
        progress.execution_started = Some(Instant::now());
        let mut ready = ready;
        let mut digest = Digest {
            rows: 0,
            scalar: None,
        };
        while let Some(batch) = poll_fn(|cx| {
            let poll = stream.poll_next_unpin(cx);
            if let Some(ready) = ready.take() {
                let _ = ready.send(());
            }
            poll
        })
        .await
        {
            let batch = batch?;
            digest.rows += batch.num_rows();
            if matches!(self.metadata.expected, Expected::Scalar(_))
                && batch.num_rows() == 1
                && batch.num_columns() == 1
            {
                let col = batch.column(0);
                if !col.is_null(0) {
                    digest.scalar = col
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .map(|a| i128::from(a.value(0)))
                        .or_else(|| {
                            col.as_any()
                                .downcast_ref::<UInt64Array>()
                                .map(|a| i128::from(a.value(0)))
                        });
                }
            }
        }
        Ok(digest)
    }

    pub(crate) fn validate(&self, result: Result<Digest>) -> Result<()> {
        let digest = result?;
        match self.metadata.expected {
            Expected::Rows(rows) => ensure!(
                digest.rows == rows,
                "expected {rows} result rows, got {}",
                digest.rows
            ),
            Expected::Scalar(scalar) => ensure!(
                digest.rows == 1 && digest.scalar == Some(i128::from(scalar)),
                "expected one row with integer {scalar}, got {} rows and scalar {:?}",
                digest.rows,
                digest.scalar
            ),
        }
        Ok(())
    }
}

fn generate(args: &Args, rows: u32, large: bool) -> Result<(MemTable, i64)> {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let mut rng = args.seed;
    let mut scalar = 0;
    let mut remaining = rows;
    let mut partitions = Vec::new();
    for partitions_left in (1..=args.partitions).rev() {
        let mut part_rows = remaining.div_ceil(partitions_left);
        remaining -= part_rows;
        let mut batches = Vec::new();
        while part_rows > 0 {
            let len = part_rows.min(args.batch_size);
            part_rows -= len;
            let array = Int64Array::from_iter_values((0..len).map(|_| {
                let v = (splitmix64(&mut rng) % 1_000_000) as i64;
                scalar += if large {
                    (v * v + v * 13) % 1_000_033
                } else {
                    i64::from(v % 17 == 0)
                };
                v
            }));
            batches.push(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(array)],
            )?);
        }
        partitions.push(batches);
    }
    Ok((MemTable::try_new(schema, partitions)?, scalar))
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

struct ActiveScan<'a>(&'a AtomicUsize);

impl<'a> ActiveScan<'a> {
    fn new(active: &'a AtomicUsize) -> Self {
        active.fetch_add(1, Ordering::SeqCst);
        Self(active)
    }
}

impl Drop for ActiveScan<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
