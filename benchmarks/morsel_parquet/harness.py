# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import argparse
import ctypes
import json
import mmap
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as pq

SEED = 0x5EED
LIMITS = {"low": 50, "high": 650}
SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("score", pa.int32()),
        pa.field("delta", pa.int32()),
        pa.field("payload", pa.int64()),
        pa.field("unused", pa.int64(), nullable=False),
    ]
)


def mix(ids, salt):
    # SplitMix64 finalizer, with intentional wraparound in uint64 arithmetic.
    z = ids.astype(np.uint64) + np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def scores(ids):
    bits = mix(ids, SEED)
    return (bits & 32767).astype(np.int32), ((bits >> 32) & 32767).astype(np.int32)


def payloads(ids):
    return (mix(ids, SEED ^ (1 << 60)) >> 16).astype(np.int64)


def generate(path, files, groups, group_rows):
    path.mkdir(parents=True)
    for file in range(files):
        with pq.ParquetWriter(
            path / f"part-{file:06d}.parquet",
            SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        ) as writer:
            for group in range(groups):
                start = (file * groups + group) * group_rows
                ids = np.arange(start, start + group_rows, dtype=np.int64)
                score, delta = scores(ids)
                table = pa.Table.from_arrays(
                    [
                        pa.array(ids),
                        pa.array(score, mask=ids % 19 == 0),
                        pa.array(delta, mask=ids % 23 == 0),
                        pa.array(payloads(ids), mask=ids % 11 == 0),
                        pa.array((mix(ids, SEED ^ (1 << 61)) >> 16).astype(np.int64)),
                    ],
                    schema=SCHEMA,
                )
                writer.write_table(table, row_group_size=group_rows)


def fixture_files(path):
    files = sorted(path.iterdir())
    if not files or any(
        not p.is_file() or not p.match("part-*.parquet") for p in files
    ):
        raise ValueError(
            f"expected a directory containing only part-*.parquet files: {path}"
        )
    return files


def workloads():
    queries = {}
    for selectivity, limit in LIMITS.items():
        clause = f"FROM fixture WHERE (score + 3 * delta) % 1000 < {limit}"
        queries[f"scan_{selectivity}"] = (
            f"SELECT row_id, payload + delta AS projected {clause}"
        )
        queries[f"count_{selectivity}"] = f"SELECT COUNT(*) AS n {clause}"
    return queries


@dataclass(frozen=True)
class Measurement:
    output_rows: int
    seconds: float
    plan: str
    first_output_seconds: float | None = None


class StockCli:
    def __init__(self, binary, path, threads, timeout):
        self.binary = binary.resolve(strict=True)
        self.ddl = f"CREATE EXTERNAL TABLE fixture STORED AS PARQUET LOCATION '{path.resolve().as_uri()}/'"
        self.timeout = timeout
        self.env = {
            k: v for k, v in os.environ.items() if not k.startswith("DATAFUSION_")
        }
        self.env.update(
            TOKIO_WORKER_THREADS=str(threads),
            DATAFUSION_EXECUTION_TARGET_PARTITIONS=str(threads),
            DATAFUSION_EXECUTION_BATCH_SIZE="8192",
            DATAFUSION_EXPLAIN_FORMAT="indent",
        )

    def query(self, sql, output=subprocess.PIPE):
        return self._query(sql, output, "csv")

    def _query(self, sql, output, output_format):
        result = subprocess.run(
            [
                str(self.binary),
                "--quiet",
                "--format",
                output_format,
                "-c",
                self.ddl,
                sql,
            ],
            env=self.env,
            stdout=output,
            timeout=self.timeout,
        )
        if result.returncode:
            if result.stdout is not None:
                detail = result.stdout
            else:
                output.seek(max(0, output.tell() - 4096))
                detail = output.read()
            raise RuntimeError(
                f"DataFusion CLI failed ({result.returncode}): {detail.decode(errors='replace')}"
            )
        return result.stdout

    def measure(self, sql):
        # The full metrics plan can exceed Python's CSV field size limit.
        result = self._query(
            f"EXPLAIN ANALYZE VERBOSE {sql}", subprocess.PIPE, "nd-json"
        )
        rows = (json.loads(line) for line in result.splitlines())
        metrics = {
            row["plan_type"]: row["plan"]
            for row in rows
            if row["plan_type"] in {"Output Rows", "Duration", "Plan with Metrics"}
        }
        duration, unit = re.fullmatch(
            r"([0-9.]+)(s|ms|µs|ns)", metrics["Duration"]
        ).groups()
        seconds = float(duration) * {"s": 1, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9}[unit]
        return Measurement(
            int(metrics["Output Rows"]), seconds, metrics["Plan with Metrics"]
        )


def expected_mask(rows, limit):
    mask = np.empty(rows, dtype=bool)
    for start in range(0, rows, 1 << 18):
        ids = np.arange(start, min(start + (1 << 18), rows), dtype=np.int64)
        score, delta = scores(ids)
        mask[start : start + len(ids)] = (
            (ids % 19 != 0) & (ids % 23 != 0) & ((score + 3 * delta) % 1000 < limit)
        )
    return mask


def verify(backend, rows):
    counts = {}
    queries = workloads()
    for selectivity, limit in LIMITS.items():
        remaining = expected_mask(rows, limit)
        expected = int(np.count_nonzero(remaining))
        counts[selectivity] = expected
        result = backend.query(queries[f"count_{selectivity}"]).decode().splitlines()
        if result != ["n", str(expected)]:
            raise ValueError(
                f"{selectivity} COUNT(*) expected {expected}, got {result}"
            )
        with tempfile.TemporaryFile() as output:
            backend.query(queries[f"scan_{selectivity}"], output)
            if output.tell() == 0:  # CLI omits even the CSV header on an empty result.
                if expected:
                    raise ValueError(f"{selectivity} scan is missing rows")
                continue
            output.seek(0)
            reader = arrow_csv.open_csv(
                output,
                convert_options=arrow_csv.ConvertOptions(
                    column_types={"row_id": pa.int64(), "projected": pa.int64()},
                    null_values=[""],
                ),
            )
            if reader.schema.names != ["row_id", "projected"]:
                raise ValueError(f"unexpected scan schema: {reader.schema}")
            for batch in reader:
                ids = batch.column(0).to_numpy(zero_copy_only=False)
                if (
                    batch.column(0).null_count
                    or np.any(ids < 0)
                    or np.any(ids >= rows)
                    or len(np.unique(ids)) != len(ids)
                    or not np.all(remaining[ids])
                ):
                    raise ValueError(
                        f"{selectivity} scan returned unexpected or duplicate row IDs"
                    )
                _, delta = scores(ids)
                projected = pa.array(payloads(ids) + delta, mask=ids % 11 == 0)
                if not batch.column(1).equals(projected):
                    raise ValueError(
                        f"{selectivity} scan returned incorrect projected values"
                    )
                remaining[ids] = False
            if remaining.any():
                raise ValueError(f"{selectivity} scan is missing rows")
    return counts


def cache_pages(files):
    """Sample Linux file page-cache residency without faulting in the data."""
    if platform.system() != "Linux":
        return None
    mincore = ctypes.CDLL(None, use_errno=True).mincore
    resident = total = 0
    for path in files:
        with path.open("rb") as file, mmap.mmap(
            file.fileno(), 0, access=mmap.ACCESS_COPY
        ) as mem:
            pages = (len(mem) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            status = (ctypes.c_ubyte * pages)()
            address = ctypes.addressof(ctypes.c_char.from_buffer(mem))
            if (
                mincore(ctypes.c_void_p(address), ctypes.c_size_t(len(mem)), status)
                != 0
            ):
                raise OSError(ctypes.get_errno(), "mincore failed", str(path))
            resident += sum(p & 1 for p in status)
            total += pages
    return {"resident": resident, "total": total}


def bench(backend, files, rows, warmups, repeats):
    queries = workloads()
    for rep in range(1 - warmups, repeats + 1):
        for name, query in queries.items():
            before = cache_pages(files) if rep > 0 else None
            result = backend.measure(query)
            if rep > 0:
                print(
                    json.dumps(
                        {
                            "case": name,
                            "rep": rep,
                            "output_rows": result.output_rows,
                            "seconds": result.seconds,
                            "input_rows_per_s": round(rows / result.seconds),
                            "first_output_seconds": result.first_output_seconds,
                            "cache_pages_before": before,
                            "cache_pages_after": cache_pages(files),
                        }
                    ),
                    flush=True,
                )


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="Opt-in stock DataFusion Parquet workload"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("path", type=Path)
    gen.add_argument("--files", type=positive, default=2)
    gen.add_argument("--groups", type=positive, default=3, help="row groups per file")
    gen.add_argument(
        "--group-rows", type=positive, default=1000, help="rows per row group"
    )
    for mode in ("verify", "bench"):
        command = commands.add_parser(mode)
        command.add_argument("path", type=Path)
        command.add_argument("--cli", type=Path, required=True)
        command.add_argument("--threads", type=positive, default=4)
        command.add_argument(
            "--timeout", type=positive, default=120, help="seconds per CLI query"
        )
        if mode == "bench":
            command.add_argument(
                "--revision", required=True, help="git revision used to build the CLI"
            )
            command.add_argument(
                "--warmups", type=int, default=1, help="unmeasured rounds"
            )
            command.add_argument(
                "--repeats", type=positive, default=3, help="measured rounds"
            )
    args = parser.parse_args()
    if args.command == "generate":
        if args.files < 2 or args.groups < 2 or args.group_rows > 64 * 1024 * 1024:
            parser.error(
                "need at least two files and two row groups/file (max. 64 Mi rows/group)"
            )
        generate(args.path, args.files, args.groups, args.group_rows)
        return

    backend = StockCli(args.cli, args.path, args.threads, args.timeout)
    files = fixture_files(args.path)
    metadata = [pq.read_metadata(path) for path in files]
    rows = sum(m.num_rows for m in metadata)
    if args.command == "verify":
        print(
            json.dumps({"input_rows": rows, "verified_matches": verify(backend, rows)})
        )
    else:
        if args.warmups < 0:
            parser.error("--warmups must be non-negative")
        version = subprocess.check_output(
            [str(backend.binary), "--version"], timeout=args.timeout, text=True
        )
        print(
            json.dumps(
                {
                    "backend": "stock",
                    "version": version.strip(),
                    "build_revision": args.revision,
                    "host": platform.node(),
                    "arch": platform.machine(),
                    "kernel": platform.release(),
                    "cpu_affinity": (
                        sorted(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity")
                        else None
                    ),
                    "tokio_workers": args.threads,
                    "target_partitions": args.threads,
                    "batch_size": 8192,
                    "seed": SEED,
                    "input_rows": rows,
                    "row_groups": sum(m.num_row_groups for m in metadata),
                    "files": len(files),
                    "parquet_bytes": sum(path.stat().st_size for path in files),
                    "page_size": mmap.PAGESIZE,
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                }
            ),
            flush=True,
        )
        bench(backend, files, rows, args.warmups, args.repeats)


if __name__ == "__main__":
    main()
