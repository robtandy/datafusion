# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Reproducible SF=5 profiler for TPC-H Q4, Q11, Q16, and Q22 (#216).

Captures:
- Physical plans (`plan`) and Stock `EXPLAIN ANALYZE` operator metrics
- Arm PMUv3 hardware counters (`perf stat`)
- Leaf CPU symbol attribution (`perf record` + `perf report`)
- Summary tables from `benchmarks/morsel_tpch/harness.py` JSONL output
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
TPCH_DIR = REPO_ROOT / "benchmarks" / "morsel_tpch"
sys.path.insert(0, str(TPCH_DIR))

import harness  # noqa: E402

QUERIES = ("q4", "q11", "q16", "q22")
ARMS = (
    ("stock_default", "off", "default"),
    ("stock_pushdown", "off", "pushdown"),
    ("morsel_no_pushdown", "required", "no_pushdown"),
    ("morsel_pushdown", "required", "pushdown"),
)
PERF_EVENTS = "cycles:u,instructions:u,r17:u,task-clock,context-switches,cpu-migrations"


def parse_perf_stat(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw_val, _, event = (
            parts[0].strip(),
            parts[1].strip(),
            parts[2].strip(),
        )
        if not raw_val or raw_val.startswith("<"):
            continue
        try:
            metrics[event] = float(raw_val)
        except ValueError:
            continue
    return metrics


def cmd_perf_stat(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.perf_lib_dir:
        env["LD_LIBRARY_PATH"] = args.perf_lib_dir
    records = []
    for q in QUERIES:
        wl = harness.WORKLOADS[q]
        sql = harness.render_workload_sql(wl, scale_factor=5.0)
        tables = ",".join(wl.tables)
        for arm_name, mode, profile in ARMS:
            reps = []
            for rep in range(1, args.repeats + 1):
                stat_file = out_dir / f"{q}_{arm_name}_r{rep}.stat.csv"
                cmd = [
                    "taskset",
                    "-c",
                    args.cpus,
                    args.perf,
                    "stat",
                    "-x,",
                    "-o",
                    str(stat_file),
                    "-e",
                    PERF_EVENTS,
                    "--",
                    args.runner,
                    "measure",
                    mode,
                    profile,
                    str(args.threads),
                    args.data_dir,
                    tables,
                    sql,
                ]
                proc = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, check=True
                )
                trial = json.loads(proc.stdout)
                hw = parse_perf_stat(stat_file.read_text())
                reps.append({"trial": trial, "hw": hw})
            med_idx = sorted(
                range(len(reps)), key=lambda i: reps[i]["trial"]["seconds"]
            )[len(reps) // 2]
            med = reps[med_idx]
            cycles = med["hw"].get("cycles:u", 0.0)
            insns = med["hw"].get("instructions:u", 0.0)
            l2_refills = med["hw"].get("r17:u", 0.0)
            rec = {
                "query": q,
                "arm": arm_name,
                "seconds": med["trial"]["seconds"],
                "exec_seconds": med["trial"]["exec_seconds"],
                "exec_cpu_seconds": med["trial"]["exec_cpu_seconds"],
                "cycles_u": cycles,
                "instructions_u": insns,
                "ipc": (insns / cycles) if cycles > 0 else 0.0,
                "l2d_refills_u": l2_refills,
                "l2d_mpki": ((l2_refills / insns * 1000.0) if insns > 0 else 0.0),
                "context_switches": med["hw"].get("context-switches", 0.0),
                "cpu_migrations": med["hw"].get("cpu-migrations", 0.0),
                "task_clock_ms": med["hw"].get("task-clock", 0.0),
            }
            records.append(rec)
            print(json.dumps(rec))
    (out_dir / "perf_stat_summary.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


def cmd_perf_record(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.perf_lib_dir:
        env["LD_LIBRARY_PATH"] = args.perf_lib_dir
    for q in QUERIES:
        wl = harness.WORKLOADS[q]
        sql = harness.render_workload_sql(wl, scale_factor=5.0)
        tables = ",".join(wl.tables)
        for arm_name, mode, profile in (
            ("stock_default", "off", "default"),
            ("morsel_no_pushdown", "required", "no_pushdown"),
            ("morsel_pushdown", "required", "pushdown"),
        ):
            pdata = out_dir / f"{q}_{arm_name}.perf.data"
            loop = (
                f"for i in $(seq 1 {args.loop_reps}); do "
                f'"$1" measure "$2" "$3" {args.threads} "$4" "$5" "$6" >/dev/null; done'
            )
            cmd = [
                "taskset",
                "-c",
                args.cpus,
                args.perf,
                "record",
                "-F",
                "3999",
                "-e",
                "cycles:u",
                "-o",
                str(pdata),
                "--",
                "bash",
                "-c",
                loop,
                "_",
                args.runner,
                mode,
                profile,
                args.data_dir,
                tables,
                sql,
            ]
            subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            rep = subprocess.run(
                [
                    args.perf,
                    "report",
                    "-i",
                    str(pdata),
                    "--stdio",
                    "--no-children",
                    "--percent-limit",
                    "0.8",
                    "--demangle",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            (out_dir / f"{q}_{arm_name}.report.txt").write_text(rep.stdout)
            samples = [
                line
                for line in rep.stdout.splitlines()
                if line.startswith("# Samples:") or line.startswith("# Total Lost")
            ]
            top = [
                line
                for line in rep.stdout.splitlines()
                if re.match(r"^\s+\d+\.\d+%", line)
            ][:10]
            print(f"\n=== {q.upper()} [{arm_name}] {' | '.join(samples)} ===")
            for line in top:
                print(line[:160])


def cmd_summarize(args: argparse.Namespace) -> None:
    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in Path(args.jsonl).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("record_type") == "trial" and rec.get("case") in QUERIES:
            rows[(rec["case"], rec["arm"])].append(rec)
    for q in QUERIES:
        print(f"\n=== {q.upper()} ===")
        for arm in (
            "stock_default",
            "stock_pushdown",
            "morsel_no_pushdown",
            "morsel_pushdown",
            "duckdb_default",
        ):
            trials = rows[(q, arm)]
            if not trials:
                continue
            secs = [t["seconds"] * 1000.0 for t in trials]
            med_t = sorted(trials, key=lambda t: t["seconds"])[len(trials) // 2]
            reg = (med_t.get("register_seconds") or 0.0) * 1000.0
            pln = (med_t.get("plan_seconds") or 0.0) * 1000.0
            exe = (med_t.get("exec_seconds") or 0.0) * 1000.0
            cpu = (med_t.get("exec_cpu_seconds") or 0.0) * 1000.0
            par = (cpu / exe) if exe > 0 and cpu > 0 else 0.0
            bs = med_t.get("bytes_scanned")
            bs_mb = f"{bs / 1048576.0:.2f} MiB" if bs is not None else "n/a"
            print(
                f"  {arm:18s}: {statistics.median(secs):6.1f} ms "
                f"({min(secs):5.1f}–{max(secs):5.1f}) "
                f"reg={reg:4.1f} plan={pln:4.1f} exec={exe:6.1f} "
                f"cpu={cpu:7.1f} par={par:4.2f}x bytes={bs_mb}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stat = sub.add_parser("perf-stat")
    p_stat.add_argument("--runner", required=True)
    p_stat.add_argument("--data-dir", required=True)
    p_stat.add_argument("--out-dir", required=True)
    p_stat.add_argument("--perf", default="perf")
    p_stat.add_argument("--perf-lib-dir", default="")
    p_stat.add_argument("--threads", type=int, default=8)
    p_stat.add_argument("--cpus", default="0-7")
    p_stat.add_argument("--repeats", type=int, default=3)

    p_rec = sub.add_parser("perf-record")
    p_rec.add_argument("--runner", required=True)
    p_rec.add_argument("--data-dir", required=True)
    p_rec.add_argument("--out-dir", required=True)
    p_rec.add_argument("--perf", default="perf")
    p_rec.add_argument("--perf-lib-dir", default="")
    p_rec.add_argument("--threads", type=int, default=8)
    p_rec.add_argument("--cpus", default="0-7")
    p_rec.add_argument("--loop-reps", type=int, default=5)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("jsonl")

    args = parser.parse_args()
    if args.cmd == "perf-stat":
        cmd_perf_stat(args)
    elif args.cmd == "perf-record":
        cmd_perf_record(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)


if __name__ == "__main__":
    main()
