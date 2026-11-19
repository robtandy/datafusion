// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::process::Command;

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
