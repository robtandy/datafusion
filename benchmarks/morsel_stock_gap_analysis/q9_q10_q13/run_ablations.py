# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Run reproducible EXPLAIN ANALYZE VERBOSE, phase/sub-phase traces, and parity-checked scratch ablations for TPC-H Q9, Q10, Q13 (#214)."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import subprocess

QUERIES: dict[str, tuple[str, str]] = {
    "q9": (
        "part,supplier,lineitem,partsupp,orders,nation",
        "benchmarks/queries/q9.sql",
    ),
    "q10": ("customer,orders,lineitem,nation", "benchmarks/queries/q10.sql"),
    "q13": ("customer,orders", "benchmarks/queries/q13.sql"),
}

ABLATION_CASES: dict[str, list[tuple[str, str, str, dict[str, str]]]] = {
    "q9": [
        ("stock_default", "off", "default", {}),
        ("morsel_no_pd", "required", "no_pushdown", {}),
        ("morsel_pd_base", "required", "pushdown", {}),
        (
            "morsel_pd+NO_UNFILT_BLOOM",
            "required",
            "pushdown",
            {"MORSEL_NO_UNFILTERED_BLOOM": "1"},
        ),
        (
            "morsel_pd+NO_RADIX_CLONE",
            "required",
            "pushdown",
            {"MORSEL_NO_RADIX_CLONE": "1"},
        ),
        (
            "morsel_pd+ASYNC_DROP",
            "required",
            "pushdown",
            {"MORSEL_ASYNC_DROP_TABLES": "1"},
        ),
        (
            "morsel_pd+ALL_Q9_FIXES",
            "required",
            "pushdown",
            {
                "MORSEL_NO_UNFILTERED_BLOOM": "1",
                "MORSEL_NO_RADIX_CLONE": "1",
                "MORSEL_ASYNC_DROP_TABLES": "1",
            },
        ),
    ],
    "q10": [
        ("stock_pushdown", "off", "pushdown", {}),
        ("morsel_pd_base", "required", "pushdown", {}),
        (
            "morsel_pd+NO_UNFILT_BLOOM",
            "required",
            "pushdown",
            {"MORSEL_NO_UNFILTERED_BLOOM": "1"},
        ),
        (
            "morsel_pd+NO_RADIX_CLONE",
            "required",
            "pushdown",
            {"MORSEL_NO_RADIX_CLONE": "1"},
        ),
        (
            "morsel_pd+FIX_BUILD_EST",
            "required",
            "pushdown",
            {"MORSEL_Q10_FIX_BUILD_EST": "1"},
        ),
        (
            "morsel_pd+ALL_Q10_FIXES",
            "required",
            "pushdown",
            {
                "MORSEL_Q10_FIX_BUILD_EST": "1",
                "MORSEL_NO_UNFILTERED_BLOOM": "1",
                "MORSEL_NO_RADIX_CLONE": "1",
            },
        ),
    ],
    "q13": [
        ("stock_default", "off", "default", {}),
        ("morsel_no_pd_base", "required", "no_pushdown", {}),
        ("morsel_pd_base", "required", "pushdown", {}),
        (
            "morsel_pd+NO_UNFILT_BLOOM",
            "required",
            "pushdown",
            {"MORSEL_NO_UNFILTERED_BLOOM": "1"},
        ),
        (
            "morsel_pd+NO_RADIX_CLONE",
            "required",
            "pushdown",
            {"MORSEL_NO_RADIX_CLONE": "1"},
        ),
        (
            "morsel_pd+SKIP_KEY_GATHER",
            "required",
            "pushdown",
            {"MORSEL_SKIP_KEY_GATHER": "1"},
        ),
        (
            "morsel_pd+ALL_Q13_FIXES",
            "required",
            "pushdown",
            {
                "MORSEL_NO_UNFILTERED_BLOOM": "1",
                "MORSEL_NO_RADIX_CLONE": "1",
                "MORSEL_SKIP_KEY_GATHER": "1",
            },
        ),
        (
            "morsel_no_pd+NO_CLONE+SKIP_GATHER",
            "required",
            "no_pushdown",
            {
                "MORSEL_NO_RADIX_CLONE": "1",
                "MORSEL_SKIP_KEY_GATHER": "1",
            },
        ),
    ],
}

PARITY_TARGETS: dict[str, dict[str, str]] = {
    "q9": {
        "MORSEL_NO_UNFILTERED_BLOOM": "1",
        "MORSEL_NO_RADIX_CLONE": "1",
        "MORSEL_ASYNC_DROP_TABLES": "1",
    },
    "q10": {
        "MORSEL_Q10_FIX_BUILD_EST": "1",
        "MORSEL_NO_UNFILTERED_BLOOM": "1",
        "MORSEL_NO_RADIX_CLONE": "1",
    },
    "q13": {
        "MORSEL_NO_UNFILTERED_BLOOM": "1",
        "MORSEL_NO_RADIX_CLONE": "1",
        "MORSEL_SKIP_KEY_GATHER": "1",
    },
}


def require_warm_cache(data_dir: Path, tables_csv: str) -> tuple[int, int]:
    files: list[Path] = []
    for table in tables_csv.split(","):
        files.extend(sorted((data_dir / table.strip()).glob("*.parquet")))
    mincore = ctypes.CDLL(None, use_errno=True).mincore
    resident = total = 0
    for path in files:
        with (
            path.open("rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY) as mem,
        ):
            pages = (len(mem) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            buf = (ctypes.c_ubyte * pages)()
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
            if mincore(ctypes.c_void_p(addr), ctypes.c_size_t(len(mem)), buf) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed", str(path))
            resident += sum(p & 1 for p in buf)
            total += pages
    if resident != total:
        raise RuntimeError(f"Parquet pages not 100% resident: {resident}/{total}")
    return resident, total


def run_analyze(
    runner: Path,
    data_dir: Path,
    out_dir: Path,
    repo_root: Path,
    threads: int,
    query_list: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        ("stock_default", "off", "default"),
        ("stock_pushdown", "off", "pushdown"),
        ("morsel_no_pushdown", "required", "no_pushdown"),
        ("morsel_pushdown", "required", "pushdown"),
    ]
    env = os.environ.copy()
    env["MORSEL_METRICS"] = "1"
    env["MORSEL_PHASE_TRACE"] = "1"
    env["MORSEL_SUBPHASE_TRACE"] = "1"

    for q in query_list:
        tables_csv, sql_rel = QUERIES[q]
        sql = (repo_root / sql_rel).read_text()
        for arm, mode, profile in arms:
            # 1 warmup
            subprocess.run(
                [
                    str(runner),
                    "measure",
                    mode,
                    profile,
                    str(threads),
                    str(data_dir),
                    tables_csv,
                    sql,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=120,
            )
            require_warm_cache(data_dir, tables_csv)
            proc = subprocess.run(
                [
                    str(runner),
                    "analyze",
                    mode,
                    profile,
                    str(threads),
                    str(data_dir),
                    tables_csv,
                    sql,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=120,
            )
            require_warm_cache(data_dir, tables_csv)
            (out_dir / f"{q}_{arm}.analyze.txt").write_text(proc.stdout)
            (out_dir / f"{q}_{arm}.phases.txt").write_text(proc.stderr)
            print(f"=== {q.upper()} {arm} ===", flush=True)
            if proc.stderr.strip():
                print(proc.stderr.strip(), flush=True)


def run_ablate(
    runner: Path,
    data_dir: Path,
    out_dir: Path,
    repo_root: Path,
    threads: int,
    repeats: int,
    query_list: list[str],
    skip_parity: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_parity:
        import pyarrow.ipc as ipc

        for q in query_list:
            tables_csv, sql_rel = QUERIES[q]
            sql = (repo_root / sql_rel).read_text()
            ref_bytes = subprocess.check_output(
                [
                    str(runner),
                    "query",
                    "off",
                    "default",
                    str(threads),
                    str(data_dir),
                    tables_csv,
                    sql,
                ],
                timeout=120,
            )
            env = os.environ.copy()
            env.update(PARITY_TARGETS[q])
            got_bytes = subprocess.check_output(
                [
                    str(runner),
                    "query",
                    "required",
                    "pushdown",
                    str(threads),
                    str(data_dir),
                    tables_csv,
                    sql,
                ],
                env=env,
                timeout=120,
            )
            t_ref = ipc.open_stream(ref_bytes).read_all()
            t_got = ipc.open_stream(got_bytes).read_all()
            if (
                t_ref.num_rows != t_got.num_rows
                or t_ref.to_pydict() != t_got.to_pydict()
            ):
                raise AssertionError(f"Parity mismatch for {q}")
            print(
                f"PARITY OK: {q} ({t_got.num_rows} rows exact match vs stock_default)",
                flush=True,
            )

    results: dict[str, dict[str, list[dict]]] = {}
    for q in query_list:
        tables_csv, sql_rel = QUERIES[q]
        sql = (repo_root / sql_rel).read_text()
        variants = ABLATION_CASES[q]
        results[q] = {name: [] for name, _, _, _ in variants}
        for name, mode, profile, extra in variants:
            env = os.environ.copy()
            env["MORSEL_METRICS"] = "1"
            env.update(extra)
            subprocess.run(
                [
                    str(runner),
                    "measure",
                    mode,
                    profile,
                    str(threads),
                    str(data_dir),
                    tables_csv,
                    sql,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=120,
            )
        for rep in range(repeats):
            order = variants if rep % 2 == 0 else list(reversed(variants))
            for name, mode, profile, extra in order:
                env = os.environ.copy()
                env["MORSEL_METRICS"] = "1"
                env.update(extra)
                require_warm_cache(data_dir, tables_csv)
                raw = subprocess.check_output(
                    [
                        str(runner),
                        "measure",
                        mode,
                        profile,
                        str(threads),
                        str(data_dir),
                        tables_csv,
                        sql,
                    ],
                    env=env,
                    text=True,
                    timeout=120,
                )
                require_warm_cache(data_dir, tables_csv)
                results[q][name].append(json.loads(raw))

        print(f"\n=== {q.upper()} ABLATIONS ({repeats} repeats) ===", flush=True)
        for name, _, _, _ in variants:
            trials = sorted(results[q][name], key=lambda x: x["seconds"])
            lo = trials[0]["seconds"] * 1000.0
            med = trials[len(trials) // 2]["seconds"] * 1000.0
            hi = trials[-1]["seconds"] * 1000.0
            m = trials[len(trials) // 2]
            print(
                f"  {name:35s}: wall={med:6.1f} ({lo:6.1f}–{hi:6.1f}) "
                f"exec={m['exec_seconds']*1000:6.1f} cpu={(m.get('exec_cpu_seconds') or 0)*1000:6.1f} "
                f"dec={m.get('morsel_decode_work_ms') or 0:6.1f} proc={m.get('morsel_process_work_ms') or 0:6.1f} "
                f"jf={m.get('morsel_join_finalize_work_ms') or 0:5.1f} "
                f"RF={m.get('adaptive_row_filter_rgs')}/{m.get('adaptive_bulk_decode_rgs')}",
                flush=True,
            )

    (out_dir / "ablations.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["analyze", "ablate"])
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--queries", type=str, default="q9,q10,q13")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    query_list = [q.strip() for q in args.queries.split(",") if q.strip()]
    if args.command == "analyze":
        run_analyze(
            args.runner,
            args.data_dir,
            args.out_dir,
            args.repo_root,
            args.threads,
            query_list,
        )
    else:
        run_ablate(
            args.runner,
            args.data_dir,
            args.out_dir,
            args.repo_root,
            args.threads,
            args.repeats,
            query_list,
            args.skip_parity,
        )


if __name__ == "__main__":
    main()
