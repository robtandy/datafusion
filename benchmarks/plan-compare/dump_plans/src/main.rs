// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Dumps the pipeline engine's plans of the queries of `queries.sql` (the 47
//! queries of `datafusion/physical-pipeline/benches/pipeline_vs_default.rs`)
//! over the Parquet files of `materialize.sh`, for the comparison with
//! DuckDB's plans (`differ.py`, see `../README.md`).
//!
//! Plans and runs every query as the benchmark's `parquet` source does (the
//! same session configuration, see [`session_context`]), with profiling on,
//! and writes the JSON dump of its profile (`QueryProfile::to_json`, whose
//! schema is that of `datafusion/physical-pipeline/docs/profiling.md`) to
//! `<out>/<query>.json`: by default the plan-shape part of the dump only (see
//! [`plan_shape`]), or with `--full` the whole dump, with all timings. The
//! paths of the files in the descriptions of the scans are relative to the
//! data directory (as in DuckDB's plans), so that the dumps do not depend on
//! where the data is.
//!
//! ```text
//! cargo run --locked --manifest-path benchmarks/plan-compare/dump_plans/Cargo.toml -- \
//!     [--data-dir DIR] [--queries FILE] [--out DIR] [--threads N] [--full] [--set KEY=VALUE]...
//! ```
//!
//! `--set` sets an option of the session (e.g.
//! `--set datafusion.sql_parser.dialect=PostgreSQL`), to dump the plans of
//! other configurations; the plans of `pipeline/` are those of the
//! benchmark's configuration, without `--set`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::error::{DataFusionError, Result};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::collect;
use datafusion::physical_planner::DefaultPhysicalPlanner;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_physical_pipeline::aggregate::PhysicalHashAggregate;
use datafusion_physical_pipeline::operators::adaptive_filter::TrustedFunctions;
use datafusion_physical_pipeline::{
    PipelineQueryPlanner, PipelineRootExec, ProfilingOptions,
};
use serde_json::{Map, Value};

/// The tables of the benchmark: a directory of Parquet files in the data
/// directory (as in `capture.py`).
const TABLES: [&str; 9] = [
    "region", "nation", "supplier", "part", "partsupp", "customer", "orders", "lineitem",
    "hits",
];

/// The metrics of the operators that the plan-shape dump keeps: the rows
/// that every operator consumed and produced, which, unlike the timings, the
/// morsels and the bytes, do not depend on the scheduling of the threads
/// (but for operators that stop early, e.g. limits).
const ROW_METRICS: [&str; 2] = ["rows_in", "rows_out"];

/// The command line, see the module documentation.
struct Args {
    data_dir: PathBuf,
    queries: PathBuf,
    out: PathBuf,
    threads: usize,
    full: bool,
    /// The options of the session to set: `(key, value)`.
    options: Vec<(String, String)>,
}

impl Args {
    fn parse() -> Result<Self> {
        // The directory of plan-compare, the parent of this package.
        let here = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the package is in benchmarks/plan-compare")
            .to_path_buf();
        let mut args = Self {
            data_dir: here.join("data"),
            queries: here.join("queries.sql"),
            out: here.join("pipeline"),
            threads: 16,
            full: false,
            options: vec![],
        };
        let mut argv = std::env::args().skip(1);
        while let Some(arg) = argv.next() {
            let mut value = || {
                argv.next().ok_or_else(|| {
                    DataFusionError::Configuration(format!("{arg} needs a value"))
                })
            };
            match arg.as_str() {
                "--data-dir" => args.data_dir = value()?.into(),
                "--queries" => args.queries = value()?.into(),
                "--out" => args.out = value()?.into(),
                "--threads" => {
                    args.threads = value()?.parse().map_err(|_| {
                        DataFusionError::Configuration(
                            "--threads needs a number".to_string(),
                        )
                    })?
                }
                "--full" => args.full = true,
                "--set" => {
                    let option = value()?;
                    let (key, value) = option.split_once('=').ok_or_else(|| {
                        DataFusionError::Configuration(format!(
                            "--set needs KEY=VALUE, not {option}"
                        ))
                    })?;
                    args.options.push((key.to_string(), value.to_string()));
                }
                _ => {
                    return Err(DataFusionError::Configuration(format!(
                        "unknown argument {arg}: [--data-dir DIR] [--queries FILE] \
                         [--out DIR] [--threads N] [--full] [--set KEY=VALUE]..."
                    )));
                }
            }
        }
        Ok(args)
    }
}

/// The (name, SQL) of the queries of `queries.sql`: every query is on the
/// lines after a line `-- <name>`, up to the line that ends with `;`, as
/// `PIPELINE_BENCH_MODE=queries` prints them (`parse_queries` of
/// `capture.py`).
fn parse_queries(script: &str) -> Result<Vec<(String, String)>> {
    let mut queries = vec![];
    let mut name: Option<String> = None;
    let mut sql: Vec<&str> = vec![];
    for line in script.lines() {
        match &name {
            None => {
                if let Some(query) = line.strip_prefix("-- ") {
                    name = Some(query.trim().to_string());
                    sql.clear();
                } else if !line.trim().is_empty() {
                    return Err(DataFusionError::Configuration(format!(
                        "unexpected line {line:?} in the queries"
                    )));
                }
            }
            Some(query) => {
                if let Some(last) = line.strip_suffix(';') {
                    sql.push(last);
                    queries.push((query.clone(), sql.join("\n")));
                    name = None;
                } else {
                    sql.push(line);
                }
            }
        }
    }
    if let Some(query) = name {
        return Err(DataFusionError::Configuration(format!(
            "query {query} does not end with `;`"
        )));
    }
    Ok(queries)
}

/// A session of the pipeline engine configured as the sessions of the
/// benchmark (`BenchConfig::session_config` and `pipeline_context` of
/// `src/test_utils.rs`): `threads` target partitions, trusting DataFusion's
/// built-in scalar and aggregate functions, planned by `PipelineQueryPlanner`
/// over DataFusion's default planner; with profiling on, and `options`.
fn session_context(
    threads: usize,
    options: &[(String, String)],
) -> Result<SessionContext> {
    let trusted = TrustedFunctions::new(&datafusion::functions::all_default_functions());
    let aggregates = PhysicalHashAggregate::trusted_aggregate_functions(
        &datafusion::functions_aggregate::all_default_aggregate_functions(),
    );
    let mut config = SessionConfig::new()
        .with_target_partitions(threads)
        .with_option_extension(trusted)
        .with_option_extension(aggregates)
        .with_option_extension(ProfilingOptions::new(true));
    for (key, value) in options {
        config.options_mut().set(key, value)?;
    }
    let planner = PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(Arc::new(RuntimeEnv::default()))
        .with_default_features()
        .with_query_planner(Arc::new(planner))
        .build();
    Ok(SessionContext::new_with_state(state))
}

/// Keep `fields` of the JSON object `value`, in their order in the dump.
fn keep(value: &Value, fields: &[&str]) -> Value {
    let object = value.as_object().expect("the dump has objects");
    let mut kept = Map::new();
    for (name, field) in object {
        if fields.contains(&name.as_str()) {
            kept.insert(name.clone(), field.clone());
        }
    }
    Value::Object(kept)
}

/// The plan-shape part of the dump `profile`: the fields of the operator
/// tree, its pipelines and its meta pipelines (in the schema of the dump)
/// without their metrics, the stages and the threads of the pipelines and
/// the metrics of the execution, but with the rows that every operator
/// consumed and produced ([`ROW_METRICS`]).
fn plan_shape(profile: &Value) -> Value {
    let mut shape = keep(profile, &["schema", "version", "root"]);
    let shape_fields = shape.as_object_mut().expect("an object");
    let operators = profile["operators"]
        .as_array()
        .expect("operators")
        .iter()
        .map(|operator| {
            let mut kept = keep(operator, &["id", "kind", "name", "inputs", "pipelines"]);
            let metrics = keep(&operator["metrics"], &ROW_METRICS);
            kept.as_object_mut()
                .expect("an object")
                .insert("metrics".to_string(), metrics);
            kept
        })
        .collect();
    shape_fields.insert("operators".to_string(), Value::Array(operators));
    let pipelines = profile["pipelines"]
        .as_array()
        .expect("pipelines")
        .iter()
        .map(|pipeline| {
            keep(
                pipeline,
                &[
                    "id",
                    "meta_pipeline",
                    "source",
                    "operators",
                    "sink",
                    "dependencies",
                    "meta_dependencies",
                ],
            )
        })
        .collect();
    shape_fields.insert("pipelines".to_string(), Value::Array(pipelines));
    let meta_pipelines = profile["meta_pipelines"]
        .as_array()
        .expect("meta pipelines")
        .iter()
        .map(|meta| keep(meta, &["id", "sink", "pipelines", "children", "parent"]))
        .collect();
    shape_fields.insert("meta_pipelines".to_string(), Value::Array(meta_pipelines));
    shape
}

/// Plan and run `sql` on `ctx`, and return the JSON dump of its profile.
async fn profile(ctx: &SessionContext, sql: &str) -> Result<Value> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let Some(root) = plan.downcast_ref::<PipelineRootExec>() else {
        return Err(DataFusionError::Plan(format!(
            "the plan is not lowered to the pipeline engine, its root is a {}",
            plan.name()
        )));
    };
    collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    let profile = root.profile().ok_or_else(|| {
        DataFusionError::Execution("the execution has no profile".to_string())
    })?;
    Ok(profile.to_json())
}

/// Make the paths of the files of the data directory in the `name`s (the
/// descriptions) of the operators of `dump` relative to it: the scans show
/// the paths of the object store (of the local file system: absolute paths
/// without their leading `/`).
fn relative_paths(dump: &mut Value, data_dir: &str) {
    let prefix = format!("{}/", data_dir.trim_start_matches('/'));
    let operators = dump["operators"].as_array_mut().expect("operators");
    for operator in operators {
        if let Some(name) = operator["name"].as_str() {
            operator["name"] = Value::String(name.replace(&prefix, ""));
        }
    }
}

async fn run(args: &Args) -> Result<()> {
    let ctx = session_context(args.threads, &args.options)?;
    let data_dir = std::fs::canonicalize(&args.data_dir).map_err(|error| {
        DataFusionError::Configuration(format!(
            "no data directory {}: run materialize.sh ({error})",
            args.data_dir.display()
        ))
    })?;
    let data_dir_name = data_dir.to_str().ok_or_else(|| {
        DataFusionError::Configuration(format!(
            "not a UTF-8 path: {}",
            data_dir.display()
        ))
    })?;
    for table in TABLES {
        let path = data_dir.join(table);
        if !path.is_dir() {
            return Err(DataFusionError::Configuration(format!(
                "no table {table} in {}: run materialize.sh",
                data_dir.display()
            )));
        }
        let path = path.to_str().ok_or_else(|| {
            DataFusionError::Configuration(format!(
                "not a UTF-8 path: {}",
                path.display()
            ))
        })?;
        ctx.register_parquet(table, path, ParquetReadOptions::default())
            .await?;
    }
    let queries = parse_queries(&std::fs::read_to_string(&args.queries)?)?;
    std::fs::create_dir_all(&args.out)?;
    for (name, sql) in &queries {
        let dump = profile(&ctx, sql)
            .await
            .map_err(|error| error.context(format!("query {name}")))?;
        let mut dump = if args.full { dump } else { plan_shape(&dump) };
        relative_paths(&mut dump, data_dir_name);
        let mut json = serde_json::to_string_pretty(&dump)
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        json.push('\n');
        std::fs::write(args.out.join(format!("{name}.json")), json)?;
    }
    // Remove the dumps of queries that are no longer in the list.
    for entry in std::fs::read_dir(&args.out)? {
        let path = entry?.path();
        let stale = path
            .extension()
            .is_some_and(|extension| extension == "json")
            && !queries.iter().any(|(name, _)| {
                path.file_stem().is_some_and(|stem| stem == name.as_str())
            });
        if stale {
            std::fs::remove_file(&path)?;
        }
    }
    println!(
        "Wrote the plans of {} queries to {}",
        queries.len(),
        args.out.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    // As the benchmark: a runtime with as many worker threads as threads.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build()?;
    runtime.block_on(run(&args))
}
