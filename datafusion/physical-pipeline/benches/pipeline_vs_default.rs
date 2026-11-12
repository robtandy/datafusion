// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Wall-clock comparison of default DataFusion and the pipeline engine.
//!
//! Runs the 22 TPC-H queries and ClickBench-style queries (on a synthetic
//! `hits` table) multi-threaded, on a default `SessionContext` and on a
//! `SessionContext` that plans queries with `PipelineQueryPlanner`, both with
//! `target_partitions` set to the number of threads, and prints the execution
//! time of every query on both engines, the speedup, and how much of the
//! query the pipeline engine runs with native operators. Before timing a
//! query, it checks that both engines return the same rows (the benchmark
//! fails at the end if they do not).
//!
//! The sessions of both engines trust DataFusion's built-in functions (see
//! `TrustedFunctions`), so that the native operators can evaluate the calls of
//! the functions that they know, and DataFusion's built-in aggregate functions
//! (see `PhysicalHashAggregate::trusted_aggregate_functions`), so that the
//! rows of the native aggregations of high-cardinality groups bypass their
//! thread-local tables, as a session of the pipeline engine would, and the
//! strings of all tables are `Utf8View`, the string type of
//! DataFusion's SQL tables and Parquet scans (DataFusion's ClickBench reads
//! `hits.parquet` so).
//!
//! The TPC-H data and queries are those of the parity suite
//! (`tests/tpch_parity.rs`), which this file includes as a module. As the
//! crate has no `harness = false` benchmark targets, the benchmark is the
//! libtest test `pipeline_vs_default`, which `cargo bench` runs with
//! `--test` (the filter skips the parity tests of the included module):
//!
//! ```text
//! cargo bench -p datafusion-physical-pipeline --bench pipeline_vs_default -- --test pipeline_vs_default
//! ```
//!
//! or, faster to build (without LTO):
//!
//! ```text
//! cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench pipeline_vs_default -- pipeline_vs_default
//! ```
//!
//! The report is written to stdout, also when libtest captures the output
//! of tests: a table of all measurements, followed by the same table in
//! Markdown (as in `docs/architecture.md`). The benchmark is configured with
//! environment variables:
//!
//! | Variable                    | Meaning                                                          | Default (optimized / debug build) |
//! | --------------------------- | ---------------------------------------------------------------- | --------------------------------- |
//! | `PIPELINE_BENCH_SCALE`      | TPC-H scale factor                                               | `0.1` / `0.01`                    |
//! | `PIPELINE_BENCH_HITS`       | rows of the `hits` table                                         | `1000000` / `20000`               |
//! | `PIPELINE_BENCH_ITERATIONS` | timed executions of every query on every engine                  | `5` / `1`                         |
//! | `PIPELINE_BENCH_THREADS`    | threads (and target partitions)                                  | available parallelism             |
//! | `PIPELINE_BENCH_SOURCE`     | `memory` (`MemTable`s), `parquet` (files in a `TempDir`), `both` | `both`                            |
//! | `PIPELINE_BENCH_QUERIES`    | comma-separated names of the queries to run, e.g. `q1,q18,hits4` | all                               |
//! | `PIPELINE_BENCH_TRUSTED_AGGREGATES` | whether the sessions trust DataFusion's built-in aggregate functions (`true` / `false`) | `true`        |
//! | `PIPELINE_BENCH_DATA_DIR`   | directory of the Parquet files, which is kept (see below)        | a `TempDir`                       |
//! | `PIPELINE_BENCH_MODE`       | `run` (the benchmark), `queries` or `data` (see below)           | `run`                             |
//!
//! With `PIPELINE_BENCH_DATA_DIR`, the `parquet` source writes its files to
//! that directory instead of a `TempDir`, and keeps them, so that other
//! engines can read the same data (`benchmarks/plan-compare` captures
//! DuckDB's plans over them): the files of every table in a directory named
//! after the table, one file per thread (`<dir>/<table>/part-<i>.parquet`).
//! The Parquet files that an earlier run left in the directories of the
//! tables are removed first. A relative path is relative to the directory of
//! the package (`datafusion/physical-pipeline`), where cargo runs the
//! benchmark. Two modes do not run the queries:
//!
//! * `PIPELINE_BENCH_MODE=queries` prints the queries (of
//!   `PIPELINE_BENCH_QUERIES`) as a SQL script: every query on the lines
//!   after a comment line with its name (`-- q1`), terminated by `;`.
//! * `PIPELINE_BENCH_MODE=data` writes the Parquet files of the tables (of
//!   `PIPELINE_BENCH_SCALE`, `PIPELINE_BENCH_HITS` and
//!   `PIPELINE_BENCH_THREADS`), as the `parquet` source does, to
//!   `PIPELINE_BENCH_DATA_DIR`, which it needs, and prints the rows and files
//!   of every table.

#[path = "../tests/tpch_parity.rs"]
mod tpch_parity;

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::instant::Instant;
use datafusion_common::{DataFusionError, Result};
use datafusion_physical_pipeline::aggregate::PhysicalHashAggregate;
use datafusion_physical_pipeline::operators::adaptive_filter::TrustedFunctions;
use datafusion_physical_plan::collect;
use tempfile::TempDir;

use crate::tpch_parity::Lowering;
use crate::tpch_parity::test_utils::{
    EngineOutput, compare_outputs, datafusion_context, has_top_level_order_by,
    pipeline_context,
};
use crate::tpch_parity::tpch::{self, TableLayout, TpchData};

/// A synthetic ClickBench `hits` table and ClickBench-style queries.
///
/// The table has the columns of ClickBench's `hits` that the queries use (its
/// strings as `Utf8View`, as DataFusion reads them from `hits.parquet`), with
/// skewed distributions: a few users, counters, regions, search phrases
/// and URLs account for most of the rows, most rows have no search phrase,
/// no advertising engine and no mobile phone model. The queries are those of
/// ClickBench, with tie-breakers added to their `ORDER BY`s (so that their
/// results are deterministic and can be compared) and fewer sums in `hits29`.
mod hits {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, Date32Array, Int16Array, Int32Array, Int64Array, RecordBatch,
        StringViewArray,
    };
    use arrow::datatypes::{Field, Schema};
    use datafusion_common::Result;

    use crate::tpch_parity::tpch::{Rng, date};

    /// The first `EventTime`: 2013-07-01 00:00:00 UTC, in seconds.
    const FIRST_EVENT_TIME: i64 = 1_372_636_800;
    /// The number of seconds of July 2013.
    const SECONDS: i64 = 31 * 86_400;
    /// The most frequent counter, which the last ClickBench queries look at.
    const HOT_COUNTER: i32 = 62;

    const PHONE_MODELS: [&str; 8] = [
        "iPhone 5",
        "iPhone 4S",
        "Galaxy S4",
        "Galaxy S III",
        "Lumia 920",
        "Nexus 4",
        "HTC One",
        "Xperia Z",
    ];
    const RESOLUTION_WIDTHS: [i16; 8] = [800, 1024, 1280, 1366, 1440, 1600, 1920, 2560];
    const SEARCH_WORDS: [&str; 12] = [
        "cheap",
        "flights",
        "weather",
        "news",
        "music",
        "video",
        "recipe",
        "games",
        "maps",
        "hotel",
        "football",
        "translate",
    ];

    /// An index in `0..n`, skewed towards 0: index `i` is about
    /// `(n / (i + 1))^(2 / 3)` times as frequent as index `n - 1`.
    fn skewed(rng: &mut Rng, n: i64) -> i64 {
        let unit = (rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        ((unit * unit * unit * n as f64) as i64).min(n - 1)
    }

    /// The `hits` table with `rows` rows.
    pub(crate) fn generate(rows: usize) -> Result<RecordBatch> {
        let mut rng = Rng::new(11);
        let rows_i64 = rows as i64;
        let users = (rows_i64 / 8).max(1);
        let phrases = (rows_i64 / 20).max(1);
        let urls = (rows_i64 / 10).max(1);
        let mut watch_ids = Vec::with_capacity(rows);
        let mut user_ids = Vec::with_capacity(rows);
        let mut counter_ids = Vec::with_capacity(rows);
        let mut region_ids = Vec::with_capacity(rows);
        let mut client_ips = Vec::with_capacity(rows);
        let mut event_times = Vec::with_capacity(rows);
        let mut event_dates = Vec::with_capacity(rows);
        let mut adv_engine_ids = Vec::with_capacity(rows);
        let mut search_engine_ids = Vec::with_capacity(rows);
        let mut search_phrases = Vec::with_capacity(rows);
        let mut url_column = Vec::with_capacity(rows);
        let mut titles = Vec::with_capacity(rows);
        let mut is_refresh = Vec::with_capacity(rows);
        let mut dont_count_hits = Vec::with_capacity(rows);
        let mut resolution_widths = Vec::with_capacity(rows);
        let mut mobile_phones = Vec::with_capacity(rows);
        let mut mobile_phone_models = Vec::with_capacity(rows);
        for row in 0..rows_i64 {
            // Distinct, scattered ids.
            watch_ids
                .push(((row as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 1) as i64);
            let user = skewed(&mut rng, users);
            user_ids.push(
                ((user as u64 + 7).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) >> 2) as i64,
            );
            let counter = skewed(&mut rng, 500) as i32;
            counter_ids.push(if counter == 0 {
                HOT_COUNTER
            } else {
                counter + 100
            });
            region_ids.push(skewed(&mut rng, 300) as i32);
            client_ips.push(rng.int(0, 1 << 30) as i32);
            let event_time = FIRST_EVENT_TIME + rng.int(0, SECONDS - 1);
            event_times.push(event_time);
            event_dates.push(
                date(2013, 7, 1) + ((event_time - FIRST_EVENT_TIME) / 86_400) as i32,
            );
            adv_engine_ids.push(if rng.chance(950) {
                0
            } else {
                rng.int(1, 30) as i16
            });
            let search_engine = if rng.chance(700) {
                0
            } else {
                rng.int(1, 20) as i16
            };
            search_engine_ids.push(search_engine);
            search_phrases.push(if search_engine == 0 && rng.chance(950) {
                String::new()
            } else {
                let phrase = skewed(&mut rng, phrases);
                format!(
                    "{} {} {phrase}",
                    SEARCH_WORDS[(phrase % 12) as usize],
                    SEARCH_WORDS[(phrase / 12 % 12) as usize]
                )
            });
            let url = skewed(&mut rng, urls);
            url_column.push(if url % 31 == 3 {
                format!("https://www.google.com/search?q=item{url}")
            } else {
                format!("https://site{}.example.com/page/{url}", url % 997)
            });
            titles.push(if url % 37 == 5 {
                format!("Google results for item {url}")
            } else {
                format!("Example page {url}")
            });
            is_refresh.push(i16::from(rng.chance(100)));
            dont_count_hits.push(i16::from(rng.chance(50)));
            resolution_widths.push(RESOLUTION_WIDTHS[rng.int(0, 7) as usize]);
            if rng.chance(850) {
                mobile_phones.push(0);
                mobile_phone_models.push("");
            } else {
                mobile_phones.push(rng.int(1, 10) as i16);
                mobile_phone_models.push(PHONE_MODELS[rng.int(0, 7) as usize]);
            }
        }
        let columns: Vec<(&str, ArrayRef)> = vec![
            ("WatchID", Arc::new(Int64Array::from(watch_ids))),
            ("UserID", Arc::new(Int64Array::from(user_ids))),
            ("CounterID", Arc::new(Int32Array::from(counter_ids))),
            ("RegionID", Arc::new(Int32Array::from(region_ids))),
            ("ClientIP", Arc::new(Int32Array::from(client_ips))),
            ("EventTime", Arc::new(Int64Array::from(event_times))),
            ("EventDate", Arc::new(Date32Array::from(event_dates))),
            ("AdvEngineID", Arc::new(Int16Array::from(adv_engine_ids))),
            (
                "SearchEngineID",
                Arc::new(Int16Array::from(search_engine_ids)),
            ),
            (
                "SearchPhrase",
                Arc::new(StringViewArray::from_iter_values(search_phrases)),
            ),
            (
                "URL",
                Arc::new(StringViewArray::from_iter_values(url_column)),
            ),
            ("Title", Arc::new(StringViewArray::from_iter_values(titles))),
            ("IsRefresh", Arc::new(Int16Array::from(is_refresh))),
            ("DontCountHits", Arc::new(Int16Array::from(dont_count_hits))),
            (
                "ResolutionWidth",
                Arc::new(Int16Array::from(resolution_widths)),
            ),
            ("MobilePhone", Arc::new(Int16Array::from(mobile_phones))),
            (
                "MobilePhoneModel",
                Arc::new(StringViewArray::from_iter_values(mobile_phone_models)),
            ),
        ];
        let schema = Schema::new(
            columns
                .iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), false))
                .collect::<Vec<_>>(),
        );
        Ok(RecordBatch::try_new(
            Arc::new(schema),
            columns.into_iter().map(|(_, array)| array).collect(),
        )?)
    }

    /// ClickBench-style queries, named after the ClickBench query they are
    /// derived from.
    pub(crate) const QUERIES: &[(&str, &str)] = &[
        ("hits0", "SELECT COUNT(*) FROM hits"),
        (
            "hits1",
            r#"SELECT COUNT(*) FROM hits WHERE "AdvEngineID" <> 0"#,
        ),
        (
            "hits2",
            r#"SELECT SUM("AdvEngineID"), COUNT(*), AVG("ResolutionWidth") FROM hits"#,
        ),
        ("hits4", r#"SELECT COUNT(DISTINCT "UserID") FROM hits"#),
        (
            "hits5",
            r#"SELECT COUNT(DISTINCT "SearchPhrase") FROM hits"#,
        ),
        (
            "hits6",
            r#"SELECT MIN("EventDate"), MAX("EventDate") FROM hits"#,
        ),
        (
            "hits7",
            r#"SELECT "AdvEngineID", COUNT(*) FROM hits WHERE "AdvEngineID" <> 0
               GROUP BY "AdvEngineID" ORDER BY COUNT(*) DESC, "AdvEngineID""#,
        ),
        (
            "hits8",
            r#"SELECT "RegionID", COUNT(DISTINCT "UserID") AS u FROM hits
               GROUP BY "RegionID" ORDER BY u DESC, "RegionID" LIMIT 10"#,
        ),
        (
            "hits10",
            r#"SELECT "MobilePhoneModel", COUNT(DISTINCT "UserID") AS u FROM hits
               WHERE "MobilePhoneModel" <> '' GROUP BY "MobilePhoneModel"
               ORDER BY u DESC, "MobilePhoneModel" LIMIT 10"#,
        ),
        (
            "hits12",
            r#"SELECT "SearchPhrase", COUNT(*) AS c FROM hits WHERE "SearchPhrase" <> ''
               GROUP BY "SearchPhrase" ORDER BY c DESC, "SearchPhrase" LIMIT 10"#,
        ),
        (
            "hits14",
            r#"SELECT "SearchEngineID", "SearchPhrase", COUNT(*) AS c FROM hits
               WHERE "SearchPhrase" <> '' GROUP BY "SearchEngineID", "SearchPhrase"
               ORDER BY c DESC, "SearchEngineID", "SearchPhrase" LIMIT 10"#,
        ),
        (
            "hits15",
            r#"SELECT "UserID", COUNT(*) FROM hits GROUP BY "UserID"
               ORDER BY COUNT(*) DESC, "UserID" LIMIT 10"#,
        ),
        (
            "hits16",
            r#"SELECT "UserID", "SearchPhrase", COUNT(*) FROM hits
               GROUP BY "UserID", "SearchPhrase"
               ORDER BY COUNT(*) DESC, "UserID", "SearchPhrase" LIMIT 10"#,
        ),
        (
            "hits20",
            r#"SELECT COUNT(*) FROM hits WHERE "URL" LIKE '%google%'"#,
        ),
        (
            "hits21",
            r#"SELECT "SearchPhrase", MIN("URL"), COUNT(*) AS c FROM hits
               WHERE "URL" LIKE '%google%' AND "SearchPhrase" <> ''
               GROUP BY "SearchPhrase" ORDER BY c DESC, "SearchPhrase" LIMIT 10"#,
        ),
        (
            "hits23",
            r#"SELECT * FROM hits WHERE "URL" LIKE '%google%'
               ORDER BY "EventTime", "WatchID" LIMIT 10"#,
        ),
        (
            "hits25",
            r#"SELECT "SearchPhrase" FROM hits WHERE "SearchPhrase" <> ''
               ORDER BY "SearchPhrase" LIMIT 10"#,
        ),
        (
            "hits27",
            r#"SELECT "CounterID", AVG(length("URL")) AS l, COUNT(*) AS c FROM hits
               WHERE "URL" <> '' GROUP BY "CounterID" HAVING COUNT(*) > 100
               ORDER BY l DESC, "CounterID" LIMIT 25"#,
        ),
        (
            "hits29",
            r#"SELECT SUM("ResolutionWidth"), SUM("ResolutionWidth" + 1),
                      SUM("ResolutionWidth" + 2), SUM("ResolutionWidth" + 3),
                      SUM("ResolutionWidth" + 4), SUM("ResolutionWidth" + 5),
                      SUM("ResolutionWidth" + 6), SUM("ResolutionWidth" + 7),
                      SUM("ResolutionWidth" + 8), SUM("ResolutionWidth" + 9)
               FROM hits"#,
        ),
        (
            "hits30",
            r#"SELECT "SearchEngineID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"),
                      AVG("ResolutionWidth")
               FROM hits WHERE "SearchPhrase" <> '' GROUP BY "SearchEngineID", "ClientIP"
               ORDER BY c DESC, "SearchEngineID", "ClientIP" LIMIT 10"#,
        ),
        (
            "hits32",
            r#"SELECT "WatchID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"),
                      AVG("ResolutionWidth")
               FROM hits GROUP BY "WatchID", "ClientIP"
               ORDER BY c DESC, "WatchID" LIMIT 10"#,
        ),
        (
            "hits33",
            r#"SELECT "URL", COUNT(*) AS c FROM hits GROUP BY "URL"
               ORDER BY c DESC, "URL" LIMIT 10"#,
        ),
        (
            "hits35",
            r#"SELECT "ClientIP", "ClientIP" - 1, "ClientIP" - 2, "ClientIP" - 3, COUNT(*) AS c
               FROM hits GROUP BY "ClientIP", "ClientIP" - 1, "ClientIP" - 2, "ClientIP" - 3
               ORDER BY c DESC, "ClientIP" LIMIT 10"#,
        ),
        (
            "hits36",
            r#"SELECT "URL", COUNT(*) AS PageViews FROM hits
               WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-01'
                 AND "EventDate" <= '2013-07-31' AND "DontCountHits" = 0
                 AND "IsRefresh" = 0 AND "URL" <> ''
               GROUP BY "URL" ORDER BY PageViews DESC, "URL" LIMIT 10"#,
        ),
        (
            "hits42",
            r#"SELECT DATE_TRUNC('minute', to_timestamp_seconds("EventTime")) AS M,
                      COUNT(*) AS PageViews
               FROM hits
               WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-14'
                 AND "EventDate" <= '2013-07-15' AND "IsRefresh" = 0
                 AND "DontCountHits" = 0
               GROUP BY DATE_TRUNC('minute', to_timestamp_seconds("EventTime"))
               ORDER BY M LIMIT 10 OFFSET 100"#,
        ),
    ];
}

/// The configuration of the benchmark, see the module documentation.
#[derive(Debug)]
struct BenchConfig {
    scale: f64,
    hits: usize,
    iterations: usize,
    threads: usize,
    sources: Vec<SourceKind>,
    queries: Option<Vec<String>>,
    /// Whether the sessions trust DataFusion's built-in aggregate functions.
    trusted_aggregates: bool,
    /// The directory of the Parquet files (`PIPELINE_BENCH_DATA_DIR`), if
    /// they are kept.
    data_dir: Option<PathBuf>,
    mode: Mode,
}

/// What the benchmark does (`PIPELINE_BENCH_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Measure the queries on both engines and print the report.
    Run,
    /// Print the queries, see [`query_list`].
    Queries,
    /// Write the Parquet files of the tables, see [`write_data`].
    Data,
}

/// Where the tables of the benchmark are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    /// `MemTable`s, with as many partitions as threads.
    Memory,
    /// Parquet files in a temporary directory (or in
    /// `PIPELINE_BENCH_DATA_DIR`), as many per table as threads.
    Parquet,
}

impl SourceKind {
    fn name(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Parquet => "parquet",
        }
    }
}

/// The value of the environment variable `name`, or `default`.
fn env_or<T: FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("invalid value of {name}: {value}")),
        Err(_) => default,
    }
}

impl BenchConfig {
    fn from_env() -> Self {
        // Unoptimized builds (e.g. `cargo test --all-targets`) only check
        // that the benchmark runs.
        let optimized = !cfg!(debug_assertions);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let source = env_or("PIPELINE_BENCH_SOURCE", "both".to_string());
        let sources = match source.to_lowercase().as_str() {
            "memory" => vec![SourceKind::Memory],
            "parquet" => vec![SourceKind::Parquet],
            "both" => vec![SourceKind::Memory, SourceKind::Parquet],
            other => panic!(
                "invalid value of PIPELINE_BENCH_SOURCE: {other} (memory, parquet or both)"
            ),
        };
        let mode = env_or("PIPELINE_BENCH_MODE", "run".to_string());
        let mode = match mode.to_lowercase().as_str() {
            "run" => Mode::Run,
            "queries" => Mode::Queries,
            "data" => Mode::Data,
            other => panic!(
                "invalid value of PIPELINE_BENCH_MODE: {other} (run, queries or data)"
            ),
        };
        let queries = std::env::var("PIPELINE_BENCH_QUERIES").ok().map(|queries| {
            queries
                .split(',')
                .map(|query| query.trim().to_lowercase())
                .filter(|query| !query.is_empty())
                .collect()
        });
        Self {
            scale: env_or("PIPELINE_BENCH_SCALE", if optimized { 0.1 } else { 0.01 }),
            hits: env_or(
                "PIPELINE_BENCH_HITS",
                if optimized { 1_000_000 } else { 20_000 },
            ),
            iterations: env_or(
                "PIPELINE_BENCH_ITERATIONS",
                if optimized { 5 } else { 1 },
            )
            .max(1),
            threads: env_or("PIPELINE_BENCH_THREADS", threads).max(1),
            sources,
            queries,
            trusted_aggregates: env_or("PIPELINE_BENCH_TRUSTED_AGGREGATES", true),
            data_dir: std::env::var_os("PIPELINE_BENCH_DATA_DIR")
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from),
            mode,
        }
    }

    /// The tables of the benchmark: the TPC-H tables of scale factor
    /// `scale` and the `hits` table of `hits` rows.
    fn tables(&self) -> Result<Vec<(&'static str, RecordBatch)>> {
        let tpch = TpchData::generate(self.scale)?;
        let mut tables = tpch.tables().to_vec();
        tables.push(("hits", hits::generate(self.hits)?));
        Ok(tables)
    }

    /// The queries to run: name and SQL.
    fn queries(&self) -> Vec<(String, &'static str)> {
        let tpch = tpch::QUERIES
            .iter()
            .enumerate()
            .map(|(i, sql)| (format!("q{}", i + 1), *sql));
        let hits = hits::QUERIES
            .iter()
            .map(|(name, sql)| ((*name).to_string(), *sql));
        tpch.chain(hits)
            .filter(|(name, _)| {
                self.queries
                    .as_ref()
                    .is_none_or(|queries| queries.contains(name))
            })
            .collect()
    }

    /// The configuration of the sessions of both engines: `threads` target
    /// partitions, trusting DataFusion's built-in functions, and its built-in
    /// aggregate functions unless `PIPELINE_BENCH_TRUSTED_AGGREGATES` is
    /// `false` (see the module documentation).
    fn session_config(&self) -> SessionConfig {
        let trusted =
            TrustedFunctions::new(&datafusion::functions::all_default_functions());
        let config = SessionConfig::new()
            .with_target_partitions(self.threads)
            .with_option_extension(trusted);
        if !self.trusted_aggregates {
            return config;
        }
        let aggregates = PhysicalHashAggregate::trusted_aggregate_functions(
            &datafusion::functions_aggregate::all_default_aggregate_functions(),
        );
        config.with_option_extension(aggregates)
    }
}

/// The layout of the tables of the benchmark: as many partitions (or
/// Parquet files) per table as `threads`, of batches of 8192 rows.
fn table_layout(threads: usize) -> TableLayout {
    TableLayout::Even {
        partitions: threads,
        batch_size: 8_192,
    }
}

/// The maximum number of rows of the row groups of the Parquet files.
const ROW_GROUP_ROWS: usize = 128 * 1024;

/// Write the Parquet files of `tables` with `layout` to `dir`, the files of
/// every table in a directory named after the table (see
/// [`tpch::write_parquet`]), and return them. First removes the Parquet files
/// in the directories of the tables (that an earlier run left there, e.g.
/// with more threads), so that they hold the files of `tables` only.
fn write_parquet_tables(
    tables: &[(&'static str, RecordBatch)],
    layout: &TableLayout,
    dir: &Path,
) -> Result<Vec<tpch::ParquetFile>> {
    let files = tpch::encode_parquet(tables, layout, ROW_GROUP_ROWS)?;
    for (table, _) in tables {
        let table_dir = dir.join(table);
        if !table_dir.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&table_dir)? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "parquet")
            {
                std::fs::remove_file(&path)?;
            }
        }
    }
    tpch::write_parquet(&files, dir)?;
    Ok(files)
}

/// The directory of the Parquet files of [`SourceKind::Parquet`].
enum ParquetDir {
    /// A temporary directory, deleted with the tables.
    Temporary(TempDir),
    /// `PIPELINE_BENCH_DATA_DIR`, which is kept.
    Kept(PathBuf),
}

impl ParquetDir {
    fn path(&self) -> &Path {
        match self {
            Self::Temporary(dir) => dir.path(),
            Self::Kept(dir) => dir,
        }
    }
}

/// The tables of the benchmark, registered in the sessions of both engines.
struct Tables {
    tables: Vec<(&'static str, RecordBatch)>,
    layout: TableLayout,
    /// The directory of the Parquet files, for [`SourceKind::Parquet`].
    parquet: Option<ParquetDir>,
}

impl Tables {
    /// The tables of `source`, whose Parquet files are written to
    /// `data_dir`, or to a temporary directory if it is `None`.
    fn try_new(
        tables: Vec<(&'static str, RecordBatch)>,
        threads: usize,
        source: SourceKind,
        data_dir: Option<&Path>,
    ) -> Result<Self> {
        let layout = table_layout(threads);
        let parquet = match source {
            SourceKind::Memory => None,
            SourceKind::Parquet => {
                let dir = match data_dir {
                    Some(dir) => ParquetDir::Kept(dir.to_path_buf()),
                    None => ParquetDir::Temporary(tempfile::tempdir()?),
                };
                write_parquet_tables(&tables, &layout, dir.path())?;
                Some(dir)
            }
        };
        Ok(Self {
            tables,
            layout,
            parquet,
        })
    }

    async fn register(&self, ctx: &SessionContext) -> Result<()> {
        match &self.parquet {
            None => tpch::register_memtables(ctx, &self.tables, &self.layout),
            Some(dir) => {
                let names = self
                    .tables
                    .iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>();
                tpch::register_parquet(ctx, dir.path(), &names).await
            }
        }
    }
}

/// Plan and execute `sql` on `ctx`; returns the output and the execution
/// time (without planning).
async fn execute(ctx: &SessionContext, sql: &str) -> Result<(EngineOutput, Duration)> {
    let df = ctx.sql(sql).await?;
    let ordered = has_top_level_order_by(df.logical_plan());
    let plan = df.create_physical_plan().await?;
    let start = Instant::now();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    let elapsed = start.elapsed();
    let output = EngineOutput {
        schema: plan.schema(),
        plan,
        batches,
        ordered,
    };
    Ok((output, elapsed))
}

/// The measurements of a query on both engines.
struct Measurement {
    source: SourceKind,
    query: String,
    rows: usize,
    datafusion: Timings,
    pipeline: Timings,
    lowering: Lowering,
    /// Why the results of the engines differ, if they do.
    mismatch: Option<String>,
}

/// The execution times of a query on an engine.
struct Timings(Vec<Duration>);

impl Timings {
    fn median(&self) -> Duration {
        let mut times = self.0.clone();
        times.sort_unstable();
        times[times.len() / 2]
    }

    fn min(&self) -> Duration {
        self.0.iter().min().copied().unwrap_or_default()
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// Measure `sql` on both engines: run it once on each engine (to warm up,
/// and to compare the results), then `iterations` times on each engine,
/// alternately.
async fn measure(
    source: SourceKind,
    query: &str,
    sql: &str,
    datafusion: &SessionContext,
    pipeline: &SessionContext,
    iterations: usize,
) -> Result<Measurement> {
    let (mut expected, _) = execute(datafusion, sql).await?;
    let (mut actual, _) = execute(pipeline, sql).await?;
    // Compare the rows regardless of their order: at other scale factors, the
    // TPC-H queries may have ties in their `ORDER BY`s, whose order is not
    // deterministic (the parity suite checks the order).
    expected.ordered = false;
    actual.ordered = false;
    let mismatch = compare_outputs(&expected, &actual).err();
    let lowering = Lowering::of(&actual.plan)?;
    let rows = expected.batches.iter().map(RecordBatch::num_rows).sum();
    let mut datafusion_times = Vec::with_capacity(iterations);
    let mut pipeline_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        datafusion_times.push(execute(datafusion, sql).await?.1);
        pipeline_times.push(execute(pipeline, sql).await?.1);
    }
    Ok(Measurement {
        source,
        query: query.to_string(),
        rows,
        datafusion: Timings(datafusion_times),
        pipeline: Timings(pipeline_times),
        lowering,
        mismatch,
    })
}

/// Run the benchmark and return its measurements.
async fn run(config: &BenchConfig) -> Result<Vec<Measurement>> {
    let queries = config.queries();
    if queries.is_empty() {
        return Err(DataFusionError::Configuration(
            "PIPELINE_BENCH_QUERIES selects no query".to_string(),
        ));
    }
    let tables = config.tables()?;
    let mut measurements = vec![];
    for &source in &config.sources {
        let tables = Tables::try_new(
            tables.clone(),
            config.threads,
            source,
            config.data_dir.as_deref(),
        )?;
        let datafusion = datafusion_context(config.session_config());
        tables.register(&datafusion).await?;
        let pipeline = pipeline_context(config.session_config());
        tables.register(&pipeline).await?;
        for (query, sql) in &queries {
            let measurement = Box::pin(measure(
                source,
                query,
                sql,
                &datafusion,
                &pipeline,
                config.iterations,
            ))
            .await?;
            measurements.push(measurement);
        }
    }
    Ok(measurements)
}

/// Format the report of `measurements`.
fn report(config: &BenchConfig, measurements: &[Measurement]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\nPipeline engine vs. default DataFusion: TPC-H SF {} and {} ClickBench-style hits, \
         {} threads, median [min] of {} executions in ms, trusted aggregate functions: {}",
        config.scale,
        config.hits,
        config.threads,
        config.iterations,
        config.trusted_aggregates
    );
    let _ = writeln!(
        out,
        "(native: native operators / operators of the lowered plan; parity: both engines \
         return the same rows)"
    );
    let _ = writeln!(
        out,
        "{:<8} {:<7} {:>9} {:>20} {:>20} {:>8} {:>9}  parity",
        "source", "query", "rows", "DataFusion", "pipeline", "speedup", "native"
    );
    let cell = |timings: &Timings| {
        format!(
            "{:.2} [{:.2}]",
            millis(timings.median()),
            millis(timings.min())
        )
    };
    for measurement in measurements {
        let datafusion = millis(measurement.datafusion.median());
        let pipeline = millis(measurement.pipeline.median());
        let _ = writeln!(
            out,
            "{:<8} {:<7} {:>9} {:>20} {:>20} {:>7.2}x {:>9}  {}",
            measurement.source.name(),
            measurement.query,
            measurement.rows,
            cell(&measurement.datafusion),
            cell(&measurement.pipeline),
            datafusion / pipeline.max(f64::MIN_POSITIVE),
            format!(
                "{}/{}",
                measurement.lowering.native_operators(),
                measurement.lowering.operators
            ),
            if measurement.mismatch.is_some() {
                "MISMATCH"
            } else {
                "ok"
            }
        );
    }
    for source in &config.sources {
        let measurements = measurements
            .iter()
            .filter(|measurement| measurement.source == *source)
            .collect::<Vec<_>>();
        let summary = Summary::of(&measurements);
        let _ = writeln!(
            out,
            "{:<8} total: DataFusion {:.2} ms, pipeline {:.2} ms, speedup {:.2}x \
             (geometric mean {:.2}x), fully native {}/{} queries",
            source.name(),
            summary.datafusion,
            summary.pipeline,
            summary.datafusion / summary.pipeline.max(f64::MIN_POSITIVE),
            summary.geometric_mean_speedup,
            summary.fully_native,
            measurements.len()
        );
    }
    for measurement in measurements {
        if !measurement.lowering.bridged.is_empty() && measurement.lowering.operators > 1
        {
            let _ = writeln!(
                out,
                "{} {}: bridged {}",
                measurement.source.name(),
                measurement.query,
                measurement.lowering.bridged.join("; ")
            );
        }
        if let Some(mismatch) = &measurement.mismatch {
            let _ = writeln!(
                out,
                "{} {}: the results differ: {mismatch}",
                measurement.source.name(),
                measurement.query
            );
        }
    }
    out
}

/// The measurements of `report`, as Markdown tables: the median times on
/// both engines, the speedup and the native and all operators of the lowered
/// plan of every query, on every source, and the totals of every source.
fn markdown_report(config: &BenchConfig, measurements: &[Measurement]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\nTPC-H SF {} and {} ClickBench-style hits, {} threads, median of {} \
         executions in ms (native: native / all operators of the lowered plan):\n",
        config.scale, config.hits, config.threads, config.iterations
    );
    let mut header = "| Query | Rows |".to_string();
    let mut separator = "| ----- | ---: |".to_string();
    for source in &config.sources {
        let name = source.name();
        let _ = write!(
            header,
            " {name} DataFusion | {name} pipeline | {name} speedup | {name} native |"
        );
        separator.push_str(" ---: | ---: | ---: | ---: |");
    }
    let _ = writeln!(out, "{header}\n{separator}");
    // The queries, in the order of their first measurement.
    let mut queries: Vec<&str> = vec![];
    for measurement in measurements {
        if !queries.contains(&measurement.query.as_str()) {
            queries.push(&measurement.query);
        }
    }
    for query in queries {
        let of_query = measurements
            .iter()
            .filter(|measurement| measurement.query == query)
            .collect::<Vec<_>>();
        let rows = of_query.first().map_or(0, |measurement| measurement.rows);
        let mut row = format!("| {query} | {rows} |");
        for source in &config.sources {
            match of_query
                .iter()
                .find(|measurement| measurement.source == *source)
            {
                Some(measurement) => {
                    let datafusion = millis(measurement.datafusion.median());
                    let pipeline = millis(measurement.pipeline.median());
                    let _ = write!(
                        row,
                        " {datafusion:.2} | {pipeline:.2} | {:.2}x | {}/{} |",
                        datafusion / pipeline.max(f64::MIN_POSITIVE),
                        measurement.lowering.native_operators(),
                        measurement.lowering.operators,
                    );
                }
                None => row.push_str(" | | | |"),
            }
        }
        let _ = writeln!(out, "{row}");
    }
    let _ = writeln!(
        out,
        "\n| Source | DataFusion (ms) | Pipeline (ms) | Speedup | Geometric mean speedup | Fully native queries |"
    );
    let _ = writeln!(
        out,
        "| ------ | --------------: | ------------: | ------: | ---------------------: | -------------------: |"
    );
    for source in &config.sources {
        let measurements = measurements
            .iter()
            .filter(|measurement| measurement.source == *source)
            .collect::<Vec<_>>();
        let summary = Summary::of(&measurements);
        let _ = writeln!(
            out,
            "| {} | {:.2} | {:.2} | {:.2}x | {:.2}x | {}/{} |",
            source.name(),
            summary.datafusion,
            summary.pipeline,
            summary.datafusion / summary.pipeline.max(f64::MIN_POSITIVE),
            summary.geometric_mean_speedup,
            summary.fully_native,
            measurements.len()
        );
    }
    out
}

/// The totals of the measurements of a source.
struct Summary {
    /// The sum of the medians on DataFusion, in ms.
    datafusion: f64,
    /// The sum of the medians on the pipeline engine, in ms.
    pipeline: f64,
    /// The geometric mean of the speedups of the medians.
    geometric_mean_speedup: f64,
    /// The number of queries without `BridgeOperator`s.
    fully_native: usize,
}

impl Summary {
    fn of(measurements: &[&Measurement]) -> Self {
        let total = |timings: fn(&Measurement) -> &Timings| {
            measurements
                .iter()
                .map(|measurement| millis(timings(measurement).median()))
                .sum::<f64>()
        };
        let log_speedups = measurements
            .iter()
            .map(|measurement| {
                let datafusion = millis(measurement.datafusion.median());
                let pipeline = millis(measurement.pipeline.median());
                (datafusion.max(f64::MIN_POSITIVE) / pipeline.max(f64::MIN_POSITIVE)).ln()
            })
            .sum::<f64>();
        Self {
            datafusion: total(|measurement| &measurement.datafusion),
            pipeline: total(|measurement| &measurement.pipeline),
            geometric_mean_speedup: (log_speedups / measurements.len().max(1) as f64)
                .exp(),
            fully_native: measurements
                .iter()
                .filter(|measurement| measurement.lowering.bridged.is_empty())
                .count(),
        }
    }
}

/// The queries of `config` as a SQL script (`PIPELINE_BENCH_MODE=queries`):
/// every query on the lines after a comment line with its name, terminated
/// by `;` (none of the queries contains a `;`), and followed by an empty line.
fn query_list(config: &BenchConfig) -> Result<String> {
    let queries = config.queries();
    if queries.is_empty() {
        return Err(DataFusionError::Configuration(
            "PIPELINE_BENCH_QUERIES selects no query".to_string(),
        ));
    }
    let mut out = String::new();
    for (name, sql) in queries {
        let _ = writeln!(out, "-- {name}\n{sql};\n");
    }
    Ok(out)
}

/// Write the Parquet files of the tables of `config` to
/// `PIPELINE_BENCH_DATA_DIR`, as [`SourceKind::Parquet`] does
/// (`PIPELINE_BENCH_MODE=data`); returns the rows, files and bytes of every
/// table.
fn write_data(config: &BenchConfig) -> Result<String> {
    let dir = config.data_dir.as_deref().ok_or_else(|| {
        DataFusionError::Configuration(
            "PIPELINE_BENCH_MODE=data needs PIPELINE_BENCH_DATA_DIR".to_string(),
        )
    })?;
    let tables = config.tables()?;
    let files = write_parquet_tables(&tables, &table_layout(config.threads), dir)?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\nParquet files of TPC-H SF {} and {} ClickBench-style hits, {} per table, in {}:",
        config.scale,
        config.hits,
        config.threads,
        dir.display()
    );
    let _ = writeln!(
        out,
        "{:<9} {:>9} {:>6} {:>10}",
        "table", "rows", "files", "bytes"
    );
    for (table, batch) in &tables {
        let (count, bytes) = files
            .iter()
            .filter(|file| file.table == *table)
            .fold((0, 0), |(count, bytes), file| {
                (count + 1, bytes + file.bytes.len())
            });
        let _ = writeln!(
            out,
            "{table:<9} {:>9} {count:>6} {bytes:>10}",
            batch.num_rows()
        );
    }
    Ok(out)
}

/// Write `text` to stdout, bypassing the output capture of libtest.
fn print_uncaptured(text: &str) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(text.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

/// The benchmark, see the module documentation.
#[test]
fn pipeline_vs_default() -> Result<()> {
    let config = BenchConfig::from_env();
    match config.mode {
        Mode::Run => {}
        // The leading newline separates the output from the test name that
        // libtest may have printed.
        Mode::Queries => return print_uncaptured(&format!("\n{}", query_list(&config)?)),
        Mode::Data => return print_uncaptured(&write_data(&config)?),
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.threads)
        .enable_all()
        .build()?;
    let measurements = runtime.block_on(run(&config))?;
    // Bypass the output capture of libtest: the report is the purpose of
    // the benchmark.
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(report(&config, &measurements).as_bytes())?;
    stdout.write_all(markdown_report(&config, &measurements).as_bytes())?;
    stdout.flush()?;
    let mismatches = measurements
        .iter()
        .filter(|measurement| measurement.mismatch.is_some())
        .map(|measurement| format!("{} {}", measurement.source.name(), measurement.query))
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "The engines return different results for {}",
        mismatches.join(", ")
    );
    Ok(())
}

/// Tests of the modes that write the data and print the queries of the
/// benchmark for other engines (`benchmarks/plan-compare`).
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::File;

    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;

    /// A configuration of `mode` over small tables, on 3 threads.
    fn config(mode: Mode, data_dir: Option<PathBuf>) -> BenchConfig {
        BenchConfig {
            scale: 0.001,
            hits: 1_000,
            iterations: 1,
            threads: 3,
            sources: vec![SourceKind::Parquet],
            queries: None,
            trusted_aggregates: true,
            data_dir,
            mode,
        }
    }

    /// The content of the files in the directories of `dir`, by directory
    /// and file name.
    fn files(dir: &Path) -> Result<BTreeMap<(String, String), Vec<u8>>> {
        let mut files = BTreeMap::new();
        for table in std::fs::read_dir(dir)? {
            let table = table?;
            for file in std::fs::read_dir(table.path())? {
                let file = file?;
                let name = (
                    table.file_name().to_string_lossy().into_owned(),
                    file.file_name().to_string_lossy().into_owned(),
                );
                files.insert(name, std::fs::read(file.path())?);
            }
        }
        Ok(files)
    }

    /// The `(name, SQL)` of the queries of a script of [`query_list`], split
    /// as `benchmarks/plan-compare/capture.py` splits it.
    fn parse_query_list(script: &str) -> Vec<(String, String)> {
        let mut queries = vec![];
        let mut lines = script.lines();
        while let Some(line) = lines.next() {
            let Some(name) = line.strip_prefix("-- ") else {
                assert!(line.is_empty(), "unexpected line {line:?}");
                continue;
            };
            let mut sql = vec![];
            for line in lines.by_ref() {
                match line.strip_suffix(';') {
                    Some(last) => {
                        sql.push(last);
                        break;
                    }
                    None => sql.push(line),
                }
            }
            queries.push((name.to_string(), sql.join("\n")));
        }
        queries
    }

    #[test]
    fn data_mode_writes_the_files_of_the_parquet_source_to_the_data_dir() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let data_dir = dir.path().join("data");
        // A file of an earlier run with more threads, and a file that is not
        // a Parquet file.
        std::fs::create_dir_all(data_dir.join("region"))?;
        std::fs::write(data_dir.join("region").join("part-7.parquet"), b"stale")?;
        std::fs::write(data_dir.join("region").join("notes.txt"), b"kept")?;
        let config = config(Mode::Data, Some(data_dir.clone()));
        let report = write_data(&config)?;

        // The data directory holds the files that the parquet source writes
        // to a temporary directory (and the file that is not a Parquet file).
        let tables = config.tables()?;
        let source =
            Tables::try_new(tables.clone(), config.threads, SourceKind::Parquet, None)?;
        let expected = files(source.parquet.as_ref().unwrap().path())?;
        let mut actual = files(&data_dir)?;
        let notes = actual.remove(&("region".to_string(), "notes.txt".to_string()));
        assert_eq!(notes.as_deref(), Some(b"kept".as_slice()));
        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (file, bytes) in &expected {
            assert!(actual[file] == *bytes, "{file:?} differs");
        }

        // One file per table per thread, with all rows of the table.
        assert_eq!(tables.len(), 9);
        for (table, batch) in &tables {
            let mut rows = 0;
            for i in 0..config.threads {
                let path = data_dir.join(table).join(format!("part-{i}.parquet"));
                let reader = SerializedFileReader::new(File::open(path)?)?;
                rows += reader.metadata().file_metadata().num_rows();
            }
            assert_eq!(rows, batch.num_rows() as i64, "{table}");
            let line = format!("{table:<9} {:>9} {:>6} ", batch.num_rows(), 3);
            assert!(report.contains(&line), "{line:?} not in {report}");
        }
        Ok(())
    }

    #[test]
    fn data_mode_needs_a_data_dir() {
        let error = write_data(&config(Mode::Data, None)).unwrap_err();
        assert!(
            error.to_string().contains("PIPELINE_BENCH_DATA_DIR"),
            "{error}"
        );
    }

    #[test]
    fn queries_mode_prints_the_22_tpch_and_25_clickbench_queries() -> Result<()> {
        let config = config(Mode::Queries, None);
        let queries = parse_query_list(&query_list(&config)?);
        let expected = config.queries();
        assert_eq!(queries.len(), 47);
        assert_eq!(expected.len(), 47);
        for ((name, sql), (expected_name, expected_sql)) in queries.iter().zip(&expected)
        {
            assert_eq!(name, expected_name);
            assert_eq!(sql, expected_sql);
        }
        let names = queries.iter().map(|(name, _)| name.as_str());
        assert_eq!(
            names.clone().filter(|name| name.starts_with('q')).count(),
            22
        );
        assert_eq!(names.filter(|name| name.starts_with("hits")).count(), 25);

        // Of the queries of `PIPELINE_BENCH_QUERIES` only.
        let mut config = config;
        config.queries = Some(vec!["q18".to_string(), "hits42".to_string()]);
        let queries = parse_query_list(&query_list(&config)?);
        let names = queries
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["q18", "hits42"]);
        config.queries = Some(vec![]);
        assert!(query_list(&config).is_err());
        Ok(())
    }
}
