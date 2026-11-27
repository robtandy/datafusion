# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Reproducible SF=5 profiler for TPC-H Q4, Q11, Q16, and Q22 (#216).

Subcommands:
- `plans`: capture physical plans (`plan`) across all 4 DataFusion arms plus
  Stock `EXPLAIN ANALYZE` operator metrics (`row_pushdown_eval_time`,
  `HashJoinExec`, `RepartitionExec`, `array_map_created_count`, `build_mem_used`).
- `perf-stat`: Arm PMUv3 hardware counters (`cycles:u`, `instructions:u`,
  `r17:u` L2d cache refills, IPC, MPKI).
- `perf-record`: leaf CPU symbol attribution (`3,999 Hz cycles:u`).
- `ablations`: counterbalanced scratch ablations + phase timelines + PyArrow
  logical table parity check (requires a runner built with `ablations.patch`).
- `summarize`: summary tables from `benchmarks/morsel_tpch/harness.py` JSONL.
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

ABLATION_SUITE: dict[str, list[tuple[str, str, str, dict[str, str]]]] = {
    "q4": [
        ("baseline_pushdown", "required", "pushdown", {}),
        ("bloom_fast_neon", "required", "pushdown", {"MORSEL_DIAG_BLOOM_FAST": "1"}),
        (
            "fast_build_semi",
            "required",
            "pushdown",
            {"MORSEL_DIAG_FAST_BUILD_SEMI": "1"},
        ),
        (
            "bloom_fast+fast_semi",
            "required",
            "pushdown",
            {"MORSEL_DIAG_BLOOM_FAST": "1", "MORSEL_DIAG_FAST_BUILD_SEMI": "1"},
        ),
        (
            "force_bulk+keep_dyn+bloom_fast+fast_semi",
            "required",
            "pushdown",
            {
                "MORSEL_DIAG_ADAPTIVE_MODE": "force_bulk",
                "MORSEL_DIAG_KEEP_DYN_IN_BULK": "1",
                "MORSEL_DIAG_BLOOM_FAST": "1",
                "MORSEL_DIAG_FAST_BUILD_SEMI": "1",
            },
        ),
        ("no_pushdown", "required", "no_pushdown", {}),
        (
            "no_pushdown+fast_semi",
            "required",
            "no_pushdown",
            {"MORSEL_DIAG_FAST_BUILD_SEMI": "1"},
        ),
    ],
    "q11": [
        ("stock_default", "off", "default", {}),
        ("baseline_pushdown", "required", "pushdown", {}),
        ("bloom_fast_neon", "required", "pushdown", {"MORSEL_DIAG_BLOOM_FAST": "1"}),
        (
            "force_bulk",
            "required",
            "pushdown",
            {"MORSEL_DIAG_ADAPTIVE_MODE": "force_bulk"},
        ),
        ("force_rf", "required", "pushdown", {"MORSEL_DIAG_ADAPTIVE_MODE": "force_rf"}),
        (
            "force_bulk+keep_dyn",
            "required",
            "pushdown",
            {
                "MORSEL_DIAG_ADAPTIVE_MODE": "force_bulk",
                "MORSEL_DIAG_KEEP_DYN_IN_BULK": "1",
            },
        ),
        (
            "force_bulk+keep_dyn+bloom_fast",
            "required",
            "pushdown",
            {
                "MORSEL_DIAG_ADAPTIVE_MODE": "force_bulk",
                "MORSEL_DIAG_KEEP_DYN_IN_BULK": "1",
                "MORSEL_DIAG_BLOOM_FAST": "1",
            },
        ),
    ],
    "q16": [
        ("stock_default", "off", "default", {}),
        ("baseline_pushdown", "required", "pushdown", {}),
        ("stage_cap16", "required", "pushdown", {"MORSEL_DIAG_STAGE_CAP": "16"}),
        ("stage_cap32", "required", "pushdown", {"MORSEL_DIAG_STAGE_CAP": "32"}),
        ("stage_cap64", "required", "pushdown", {"MORSEL_DIAG_STAGE_CAP": "64"}),
        (
            "concurrent_builds",
            "required",
            "pushdown",
            {"MORSEL_DIAG_CONCURRENT_BUILDS": "1"},
        ),
        ("small_bulk", "required", "pushdown", {"MORSEL_DIAG_SMALL_BULK": "1"}),
        (
            "stage_cap16+small_bulk",
            "required",
            "pushdown",
            {"MORSEL_DIAG_STAGE_CAP": "16", "MORSEL_DIAG_SMALL_BULK": "1"},
        ),
        (
            "concurrent+small_bulk+cap64+bloom_fast",
            "required",
            "pushdown",
            {
                "MORSEL_DIAG_CONCURRENT_BUILDS": "1",
                "MORSEL_DIAG_SMALL_BULK": "1",
                "MORSEL_DIAG_STAGE_CAP": "64",
                "MORSEL_DIAG_BLOOM_FAST": "1",
            },
        ),
    ],
    "q22": [
        ("baseline_pushdown", "required", "pushdown", {}),
        ("bloom_fast_neon", "required", "pushdown", {"MORSEL_DIAG_BLOOM_FAST": "1"}),
        (
            "fast_build_semi",
            "required",
            "pushdown",
            {"MORSEL_DIAG_FAST_BUILD_SEMI": "1"},
        ),
        (
            "bloom_fast+fast_semi",
            "required",
            "pushdown",
            {"MORSEL_DIAG_BLOOM_FAST": "1", "MORSEL_DIAG_FAST_BUILD_SEMI": "1"},
        ),
        ("small_bulk", "required", "pushdown", {"MORSEL_DIAG_SMALL_BULK": "1"}),
    ],
}


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


def cmd_plans(args: argparse.Namespace) -> None:
    import pyarrow.ipc as ipc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in QUERIES:
        wl = harness.WORKLOADS[q]
        sql = harness.render_workload_sql(wl, scale_factor=5.0)
        tables = ",".join(wl.tables)
        print(f"\n================ {q.upper()} ================")
        for arm_name, mode, profile in ARMS:
            proc = subprocess.run(
                [
                    args.runner,
                    "plan",
                    mode,
                    profile,
                    str(args.threads),
                    args.data_dir,
                    tables,
                    sql,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            obj = json.loads(proc.stdout)
            plan_txt = obj["plan"] or ""
            (out_dir / f"{q}_{arm_name}_plan.txt").write_text(plan_txt)
            print(f"  saved {q}_{arm_name}_plan.txt (admitted={obj['admitted']})")

        for arm_name, mode, profile in (
            ("stock_default", "off", "default"),
            ("stock_pushdown", "off", "pushdown"),
        ):
            ea_sql = f"EXPLAIN ANALYZE {sql}"
            proc = subprocess.run(
                [
                    "taskset",
                    "-c",
                    args.cpus,
                    args.runner,
                    "query",
                    mode,
                    profile,
                    str(args.threads),
                    args.data_dir,
                    tables,
                    ea_sql,
                ],
                capture_output=True,
                check=True,
            )
            tbl = ipc.open_stream(proc.stdout).read_all()
            chunks = []
            for ptype, pstr in zip(
                tbl.column(0).to_pylist(), tbl.column(1).to_pylist()
            ):
                chunks.append(f"--- {ptype} ---\n{pstr}")
            full_txt = "\n".join(chunks) + "\n"
            (out_dir / f"{q}_{arm_name}_explain_analyze.txt").write_text(full_txt)
            print(f"  saved {q}_{arm_name}_explain_analyze.txt")
            for line in full_txt.splitlines():
                stripped = line.strip()
                if any(
                    tok in stripped
                    for tok in (
                        "HashJoinExec:",
                        "RepartitionExec:",
                        "DataSourceExec:",
                        "FilterExec:",
                        "SortPreservingMergeExec:",
                    )
                ):
                    hits = []
                    for pat in (
                        r"elapsed_compute=[^,\]]+",
                        r"repartition_time=[^,\]]+",
                        r"row_pushdown_eval_time=[^,\]]+",
                        r"array_map_created_count=[^,\]]+",
                        r"build_mem_used=[^,\]]+",
                        r"output_rows=[^,\]]+",
                    ):
                        m = re.search(pat, stripped)
                        if m:
                            hits.append(m.group(0))
                    op_name = stripped.split(":", 1)[0]
                    if hits:
                        print(f"    [{arm_name}] {op_name}: {', '.join(hits)}")


def cmd_ablations(args: argparse.Namespace) -> None:
    import pyarrow.ipc as ipc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        tuple(q.strip() for q in args.queries.split(",") if q.strip())
        if args.queries != "all"
        else QUERIES
    )

    def read_table(
        mode: str, profile: str, tables: str, sql: str, extra: dict[str, str]
    ):
        env = os.environ.copy()
        env.update(extra)
        proc = subprocess.run(
            [
                args.runner,
                "query",
                mode,
                profile,
                str(args.threads),
                args.data_dir,
                tables,
                sql,
            ],
            env=env,
            capture_output=True,
            check=True,
        )
        return ipc.open_stream(proc.stdout).read_all()

    results: dict[str, dict] = {}
    for q in selected:
        wl = harness.WORKLOADS[q]
        sql = harness.render_workload_sql(wl, scale_factor=5.0)
        tables = ",".join(wl.tables)
        arms = ABLATION_SUITE[q]
        ref_tbl = read_table("required", "pushdown", tables, sql, {})

        # 1. Verify exact logical Arrow table parity for every ablation toggle.
        for name, mode, profile, extra in arms:
            got = read_table(mode, profile, tables, sql, extra)
            if not got.equals(ref_tbl):
                raise AssertionError(
                    f"Arrow table parity mismatch on {q} {name}: "
                    f"{got.num_rows} vs {ref_tbl.num_rows} rows"
                )

        # 2. Capture one MORSEL_DIAG_PHASES=1 timeline per Morsel ablation toggle.
        phases: dict[str, str] = {}
        for name, mode, profile, extra in arms:
            if mode != "required":
                phases[name] = ""
                continue
            env = os.environ.copy()
            env.update(extra)
            env["MORSEL_DIAG_PHASES"] = "1"
            proc = subprocess.run(
                [
                    "taskset",
                    "-c",
                    args.cpus,
                    args.runner,
                    "measure",
                    mode,
                    profile,
                    str(args.threads),
                    args.data_dir,
                    tables,
                    sql,
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            phases[name] = proc.stderr.strip()

        # 3. Run counterbalanced timed repeats without phase stderr logging.
        per_arm: dict[str, list[dict]] = {name: [] for name, _, _, _ in arms}
        for rd in range(args.repeats):
            order = arms if rd % 2 == 0 else list(reversed(arms))
            for name, mode, profile, extra in order:
                env = os.environ.copy()
                env.update(extra)
                proc = subprocess.run(
                    [
                        "taskset",
                        "-c",
                        args.cpus,
                        args.runner,
                        "measure",
                        mode,
                        profile,
                        str(args.threads),
                        args.data_dir,
                        tables,
                        sql,
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                per_arm[name].append(json.loads(proc.stdout))

        base_exec = statistics.median(
            t["exec_seconds"] * 1000.0 for t in per_arm["baseline_pushdown"]
        )
        base_cpu = statistics.median(
            t["exec_cpu_seconds"] * 1000.0 for t in per_arm["baseline_pushdown"]
        )
        print(
            f"\n=== {q.upper()} ABLATIONS "
            f"({args.repeats} counterbalanced rounds, parity OK) ==="
        )
        q_out = {}
        for name, _, _, _ in arms:
            ts = per_arm[name]
            execs = [t["exec_seconds"] * 1000.0 for t in ts]
            cpus = [t["exec_cpu_seconds"] * 1000.0 for t in ts]
            med_t = sorted(ts, key=lambda t: t["exec_seconds"])[len(ts) // 2]
            me = statistics.median(execs)
            mc = statistics.median(cpus)
            print(
                f"  {name:42s}: exec={me:5.1f}ms ({me - base_exec:+5.1f}) "
                f"cpu={mc:6.1f}ms ({mc - base_cpu:+6.1f}) par={mc / me:4.2f}x "
                f"wait={med_t.get('morsel_ready_wait_ms') or 0:5.1f} "
                f"RF/Bk={med_t.get('adaptive_row_filter_rgs')}/"
                f"{med_t.get('adaptive_bulk_decode_rgs')}"
            )
            q_out[name] = {
                "exec_med_ms": round(me, 2),
                "exec_delta_ms": round(me - base_exec, 2),
                "cpu_med_ms": round(mc, 2),
                "cpu_delta_ms": round(mc - base_cpu, 2),
                "par": round(mc / me, 2),
                "ready_wait_ms": round(med_t.get("morsel_ready_wait_ms") or 0.0, 2),
                "rf_rgs": med_t.get("adaptive_row_filter_rgs"),
                "bulk_rgs": med_t.get("adaptive_bulk_decode_rgs"),
                "phases": phases[name].splitlines(),
            }
        results[q] = q_out
    (out_dir / "ablations_summary.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


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

    p_plans = sub.add_parser("plans")
    p_plans.add_argument("--runner", required=True)
    p_plans.add_argument("--data-dir", required=True)
    p_plans.add_argument("--out-dir", required=True)
    p_plans.add_argument("--threads", type=int, default=8)
    p_plans.add_argument("--cpus", default="0-7")

    p_abl = sub.add_parser("ablations")
    p_abl.add_argument("--runner", required=True)
    p_abl.add_argument("--data-dir", required=True)
    p_abl.add_argument("--out-dir", required=True)
    p_abl.add_argument("--threads", type=int, default=8)
    p_abl.add_argument("--cpus", default="0-7")
    p_abl.add_argument("--repeats", type=int, default=5)
    p_abl.add_argument("--queries", default="all")

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
    if args.cmd == "plans":
        cmd_plans(args)
    elif args.cmd == "ablations":
        cmd_ablations(args)
    elif args.cmd == "perf-stat":
        cmd_perf_stat(args)
    elif args.cmd == "perf-record":
        cmd_perf_record(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)


if __name__ == "__main__":
    main()
