# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Capture Arm PMUv3 hardware counters and demangled symbol profiles for TPC-H Q9, Q10, Q13 (#214)."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
from pathlib import Path
import statistics
import subprocess
import sys

EVENTS = "cycles:u,instructions:u,r17:u,task-clock,context-switches,cpu-migrations"

QUERIES: dict[str, tuple[str, str]] = {
    "q9": (
        "part,supplier,lineitem,partsupp,orders,nation",
        "benchmarks/queries/q9.sql",
    ),
    "q10": ("customer,orders,lineitem,nation", "benchmarks/queries/q10.sql"),
    "q13": ("customer,orders", "benchmarks/queries/q13.sql"),
}

ARMS: dict[str, tuple[str, str]] = {
    "stock_default": ("off", "default"),
    "stock_pushdown": ("off", "pushdown"),
    "morsel_no_pushdown": ("required", "no_pushdown"),
    "morsel_pushdown": ("required", "pushdown"),
}


def table_parquet_files(data_dir: Path, tables_csv: str) -> list[Path]:
    files: list[Path] = []
    for table in tables_csv.split(","):
        files.extend(sorted((data_dir / table.strip()).glob("*.parquet")))
    return files


def check_warm_cache(files: list[Path]) -> tuple[int, int]:
    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    libc = ctypes.CDLL(libc_name, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = 0
    resident_pages = 0
    for path in files:
        size = path.stat().st_size
        if size == 0:
            continue
        npages = (size + page_size - 1) // page_size
        with path.open("rb") as fh:
            mm = mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_COPY)
            try:
                buf = (ctypes.c_ubyte * npages)()
                addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), buf)
                if rc != 0:
                    err = ctypes.get_errno()
                    raise OSError(err, f"mincore failed for {path}")
                total_pages += npages
                resident_pages += sum(1 for b in buf if b & 1)
            finally:
                mm.close()
    return resident_pages, total_pages


def require_warm_cache(files: list[Path]) -> tuple[int, int]:
    res, tot = check_warm_cache(files)
    if res != tot:
        raise RuntimeError(f"Parquet pages not 100% resident: {res}/{tot}")
    return res, tot


def parse_perf_stat(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw_val, event = parts[0].strip(), parts[2].strip()
        if not raw_val or raw_val.startswith("<"):
            continue
        try:
            metrics[event] = float(raw_val)
        except ValueError:
            continue
    return metrics


def demangle_symbols(symbols: list[str]) -> list[str]:
    if not symbols:
        return []
    proc = subprocess.run(
        ["llvm-cxxfilt"],
        input="\n".join(symbols) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    out = proc.stdout.splitlines()
    return out if len(out) == len(symbols) else symbols


def parse_perf_report(report_text: str) -> dict:
    total_samples = 0
    lost_samples = 0
    event_count = 0
    raw_entries: list[tuple[float, str]] = []
    for line in report_text.splitlines():
        s = line.strip()
        if s.startswith("# Total Lost Samples:"):
            lost_samples = int(s.split(":", 1)[1].strip())
        elif s.startswith("# Samples:"):
            tok = s.split(":", 1)[1].strip().split()[0]
            mult = 1
            if tok.endswith("K"):
                mult = 1_000
                tok = tok[:-1]
            elif tok.endswith("M"):
                mult = 1_000_000
                tok = tok[:-1]
            total_samples = int(float(tok) * mult)
        elif s.startswith("# Event count"):
            event_count = int(s.split(":", 1)[1].strip())
        elif s and not s.startswith("#") and "%" in s:
            parts = s.split(None, 2)
            if len(parts) >= 3 and parts[0].endswith("%"):
                pct = float(parts[0][:-1])
                rest = parts[2].strip()
                # Strip trailing IPC columns if present
                sym = rest.split()[0]
                raw_entries.append((pct, sym))

    demangled = demangle_symbols([sym for _, sym in raw_entries])
    entries = [
        {"pct": pct, "raw": raw, "symbol": dem}
        for (pct, raw), dem in zip(raw_entries, demangled)
    ]
    return {
        "total_samples": total_samples,
        "lost_samples": lost_samples,
        "event_count": event_count,
        "symbols": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--perf", type=str, required=True)
    ap.add_argument("--perf-lib-dir", type=str, default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--stat-repeats", type=int, default=3)
    ap.add_argument("--queries", type=str, default="q9,q10,q13")
    ap.add_argument(
        "--arms",
        type=str,
        default="stock_default,stock_pushdown,morsel_no_pushdown,morsel_pushdown",
    )
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.perf_lib_dir:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{args.perf_lib_dir}:{prev}" if prev else args.perf_lib_dir
        )
    env["MORSEL_METRICS"] = "1"

    query_list = [q.strip() for q in args.queries.split(",") if q.strip()]
    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]

    summary: dict[str, dict] = {}
    for q in query_list:
        tables_csv, sql_rel = QUERIES[q]
        sql = (args.repo_root / sql_rel).read_text()
        files = table_parquet_files(args.data_dir, tables_csv)
        summary[q] = {}

        for arm in arm_list:
            mode, profile = ARMS[arm]
            cmd = [
                str(args.runner),
                "measure",
                mode,
                profile,
                str(args.threads),
                str(args.data_dir),
                tables_csv,
                sql,
            ]
            # 1 warmup
            subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )

            stat_runs = []
            for rep in range(args.stat_repeats):
                stat_path = args.out_dir / f"{q}.{arm}.{rep}.stat"
                res_pre, tot_pre = require_warm_cache(files)
                proc = subprocess.run(
                    [
                        args.perf,
                        "stat",
                        "-x,",
                        "-o",
                        str(stat_path),
                        "-e",
                        EVENTS,
                        "--",
                        *cmd,
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=120,
                )
                res_post, tot_post = require_warm_cache(files)
                runner_json = json.loads(proc.stdout)
                hw = parse_perf_stat(stat_path)
                stat_runs.append(
                    {
                        "rep": rep,
                        "wall_ms": runner_json["seconds"] * 1000.0,
                        "exec_ms": runner_json["exec_seconds"] * 1000.0,
                        "exec_cpu_ms": (runner_json.get("exec_cpu_seconds") or 0.0)
                        * 1000.0,
                        "hw": hw,
                        "pages": [res_pre, tot_pre, res_post, tot_post],
                    }
                )

            # Median hardware counters
            hw_keys = sorted({k for r in stat_runs for k in r["hw"]})
            hw_med = {
                k: statistics.median([r["hw"][k] for r in stat_runs if k in r["hw"]])
                for k in hw_keys
            }
            cycles = hw_med.get("cycles:u", 0.0)
            insns = hw_med.get("instructions:u", 0.0)
            l2_refill = hw_med.get("r17:u", 0.0)
            ipc = insns / cycles if cycles > 0 else 0.0
            l2_mpki = (l2_refill * 1000.0 / insns) if insns > 0 else 0.0

            # perf record
            perf_data = args.out_dir / f"{q}.{arm}.perf"
            report_path = args.out_dir / f"{q}.{arm}.report"
            require_warm_cache(files)
            rec_proc = subprocess.run(
                [
                    args.perf,
                    "record",
                    "-q",
                    "-F",
                    "3999",
                    "-e",
                    "cycles:u",
                    "-o",
                    str(perf_data),
                    "--",
                    *cmd,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=120,
            )
            require_warm_cache(files)
            rec_json = json.loads(rec_proc.stdout)
            rep_proc = subprocess.run(
                [
                    args.perf,
                    "report",
                    "--stdio",
                    "--no-children",
                    "--sort",
                    "symbol",
                    "-i",
                    str(perf_data),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=120,
            )
            report_path.write_text(rep_proc.stdout)
            parsed = parse_perf_report(rep_proc.stdout)

            summary[q][arm] = {
                "wall_ms_median": statistics.median([r["wall_ms"] for r in stat_runs]),
                "exec_ms_median": statistics.median([r["exec_ms"] for r in stat_runs]),
                "exec_cpu_ms_median": statistics.median(
                    [r["exec_cpu_ms"] for r in stat_runs]
                ),
                "hw_median": hw_med,
                "ipc": ipc,
                "l2_refill_mpki": l2_mpki,
                "stat_runs": stat_runs,
                "record_wall_ms": rec_json["seconds"] * 1000.0,
                "record_samples": parsed["total_samples"],
                "record_lost_samples": parsed["lost_samples"],
                "record_event_count": parsed["event_count"],
                "top_symbols": parsed["symbols"][:40],
            }
            print(
                f"{q} {arm:20s}: wall={summary[q][arm]['wall_ms_median']:6.1f}ms "
                f"cycles={cycles/1e9:5.2f}G insns={insns/1e9:5.2f}G IPC={ipc:.2f} "
                f"L2_refill={l2_refill/1e6:6.1f}M ({l2_mpki:5.2f} MPKI) "
                f"samples={parsed['total_samples']} lost={parsed['lost_samples']}",
                flush=True,
            )

    (args.out_dir / "perf_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
