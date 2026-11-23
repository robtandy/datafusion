// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use serde_json::Value;

#[test]
fn stock_pair_drains_queries_and_overlaps_scans() {
    let output = Command::new(env!("CARGO_BIN_EXE_morsel-mixed"))
        .args([
            "--workers",
            "2",
            "--partitions",
            "2",
            "--large-rows",
            "262144",
            "--repetitions",
            "1",
            "--requests",
            "40",
            "--interval-ms",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("benchmark JSON report");
    let isolated = &report["runs"][0];
    let mixed = &report["runs"][1];
    assert_eq!(isolated["mode"], "isolated");
    assert_eq!(mixed["mode"], "mixed");
    assert_eq!(isolated["overlap_requests"], 0);
    assert!(mixed["overlap_requests"].as_u64().unwrap() > 0);
    assert_eq!(mixed["background"]["lanes_polled"], 3);
    assert!(mixed["background"]["completed"].as_u64().unwrap() >= 3);
    for run in [isolated, mixed] {
        assert_eq!(run["offered"], 40);
        assert_eq!(run["failed"], 0);
        for timing in [
            "latency",
            "offer_lag",
            "queue_wait",
            "service_time",
            "planning",
        ] {
            assert_eq!(run[timing]["samples"], run["completed"], "{timing}");
        }
        assert_eq!(
            run["completed"].as_u64().unwrap() + run["dropped"].as_u64().unwrap(),
            40
        );
        let requests = run["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 40);
        for (id, request) in requests.iter().enumerate() {
            assert_eq!(request["scheduled_ns"], id * 4_000_000);
        }
        let overlapped = requests
            .iter()
            .filter(|r| {
                r["status"] == "completed"
                    && r["active_scans_at_execution_start"].as_u64().unwrap() > 0
            })
            .count();
        assert_eq!(run["overlap_requests"], overlapped);
    }
}

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new(mode: &str) -> Self {
        let path = std::env::temp_dir()
            .join(format!("morsel-mixed-{}-{mode}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_fixture(dir: &Path, name: &str, rows: i64) -> PathBuf {
    let path = dir.join(name);
    let batch = RecordBatch::try_from_iter([(
        "v",
        Arc::new(Int64Array::from_iter_values(0..rows)) as _,
    )])
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(8192))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props))
            .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

fn parquet_count_command(mode: &str, small: &Path, large: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_morsel-mixed"));
    command.args([
        "--morsel-mode",
        mode,
        "--workers",
        "2",
        "--partitions",
        "2",
        "--small-fixture",
        small.to_str().unwrap(),
        "--large-fixture",
        large.to_str().unwrap(),
        "--small-expected-scalar",
        "61",
        "--large-expected-scalar",
        "30841",
        "--small-sql",
        "SELECT COUNT(*) FROM small WHERE v % 17 = 0",
        "--large-sql",
        "SELECT COUNT(*) FROM large WHERE v % 17 = 0",
        "--repetitions",
        "1",
        "--requests",
        "40",
        "--interval-ms",
        "4",
    ]);
    command
}

#[test]
fn stock_parquet_pair_reports_stock_plans_and_offer_rates() {
    let dir = FixtureDir::new("off");
    let small = write_fixture(&dir.0, "small.parquet", 1024);
    let large = write_fixture(&dir.0, "large.parquet", 524288);
    let output = parquet_count_command("off", &small, &large)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["runtime"]["morsel_mode"], "off");
    for table in ["small", "large"] {
        let plan = report[table]["plan"].as_str().unwrap();
        assert!(!plan.contains("MorselExec"), "{plan}");
        assert!(
            plan.lines().any(|line| {
                line.contains("DataSourceExec:") && line.contains("file_type=parquet")
            }),
            "{plan}"
        );
    }
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 2);
    for run in runs {
        assert_eq!(run["offered"], 40);
        for (rate, count) in [("completion_rate", "completed"), ("drop_rate", "dropped")]
        {
            let fraction =
                run[count].as_f64().unwrap() / run["offered"].as_f64().unwrap();
            let reported = run[rate].as_f64().unwrap();
            assert!(
                (reported - fraction).abs() <= f64::EPSILON,
                "{rate}: {reported} != {fraction}"
            );
        }
    }
}

#[test]
fn required_parquet_pair_checks_both_paths_and_rejects_unsupported_sql() {
    let dir = FixtureDir::new("required");
    let small = write_fixture(&dir.0, "small.parquet", 1024);
    let large = write_fixture(&dir.0, "large.parquet", 524288);
    let allowed = parquet_count_command("required", &small, &large)
        .arg("--mixed-first")
        .output()
        .unwrap();
    assert!(
        allowed.status.success(),
        "{}",
        String::from_utf8_lossy(&allowed.stderr)
    );
    let report: Value = serde_json::from_slice(&allowed.stdout).unwrap();
    assert_eq!(report["runtime"]["morsel_mode"], "required");
    for table in ["small", "large"] {
        let plan = report[table]["plan"].as_str().unwrap();
        assert!(plan.starts_with("MorselExec: workers=2, "), "{plan}");
        assert!(plan.contains(&format!("NativeParquet[{table}]")), "{plan}");
    }
    assert_eq!(report["runs"][0]["mode"], "mixed");
    assert!(report["runs"][0]["overlap_requests"].as_u64().unwrap() > 0);
    assert_eq!(report["runs"][1]["mode"], "isolated");

    let denied = Command::new(env!("CARGO_BIN_EXE_morsel-mixed"))
        .args([
            "--morsel-mode",
            "required",
            "--small-fixture",
            small.to_str().unwrap(),
            "--small-sql",
            "SELECT v FROM small LIMIT 1",
            "--small-expected-rows",
            "1",
            "--warmup",
            "0",
        ])
        .output()
        .unwrap();
    assert!(!denied.status.success());
    assert!(String::from_utf8_lossy(&denied.stderr).contains("LIMIT"));
}

#[test]
fn pushdown_filters_and_morsel_metrics_flags_work_on_parquet() {
    let dir = FixtureDir::new("pushdown");
    let small = write_fixture(&dir.0, "small.parquet", 1024);
    let large = write_fixture(&dir.0, "large.parquet", 524288);
    let output = parquet_count_command("required", &small, &large)
        .args(["--pushdown-filters", "--morsel-metrics"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["config"]["pushdown_filters"], true);
    assert_eq!(report["config"]["morsel_metrics"], true);
    for table in ["small", "large"] {
        let plan = report[table]["plan"].as_str().unwrap();
        assert!(
            plan.starts_with(&format!(
                "MorselExec: workers=2, NativeParquet[{table}, predicate="
            )),
            "{plan}"
        );
        assert!(plan.contains("CountSink"), "{plan}");
    }
    let isolated = &report["runs"][0];
    let mixed = &report["runs"][1];
    assert!(
        !isolated["morsel_metrics"]["foreground"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        isolated["morsel_metrics"]["background"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let bg_metrics = mixed["morsel_metrics"]["background"].as_array().unwrap();
    assert!(!bg_metrics.is_empty());
    assert!(
        bg_metrics
            .iter()
            .all(|m| m["morsels_completed"].as_u64().unwrap() > 0)
    );
}

#[test]
fn wrong_expected_row_count_is_reported_after_drain() {
    let output = Command::new(env!("CARGO_BIN_EXE_morsel-mixed"))
        .args([
            "--workers",
            "2",
            "--partitions",
            "2",
            "--large-rows",
            "262144",
            "--small-rows",
            "1025",
            "--batch-size",
            "64",
            "--small-sql",
            "SELECT v FROM small",
            "--small-expected-rows",
            "1024",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--requests",
            "3",
            "--interval-ms",
            "4",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("invalid run still emits JSON");
    let isolated = &report["runs"][0];
    assert_eq!(report["valid"], false);
    assert_eq!(isolated["failed"], 3);
    assert_eq!(isolated["latency"]["samples"], 0);
    for request in isolated["requests"].as_array().unwrap() {
        assert_eq!(request["error"], "expected 1024 result rows, got 1025");
    }
}
