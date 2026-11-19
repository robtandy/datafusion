# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import io
import json
import mmap
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from harness import StockCli, bench, fixture_files, generate, verify, workloads


class SmokeTest(unittest.TestCase):
    def test_ab_evidence_rejects_fallback_and_missing_count_finalization(self):
        from harness import validate_path

        stock = {"plan": "DataSourceExec: file_type=parquet\n"}
        native = {
            "plan": "MorselExec: workers=2, NativeParquet[fixture] -> CountSink",
            "parquet_morsels_claimed": 6,
            "count_finalizations": 1,
        }
        validate_path(stock, "off", "scan_low", 6)
        validate_path(native, "required", "count_low", 6)
        for evidence in (
            stock,
            dict(native, parquet_morsels_claimed=0),
            dict(native, count_finalizations=0),
            dict(native, plan=native["plan"].replace("CountSink", "ResultSink")),
        ):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                validate_path(evidence, "required", "count_low", 6)
        with self.assertRaises(ValueError):
            validate_path(native, "off", "count_low", 6)

    def test_ab_rejects_unverified_cache_regimes(self):
        from harness import prepare_cache

        with patch("harness.cache_pages", return_value={"resident": 1, "total": 2}):
            for regime in ("warm", "page-cache-cold"):
                with self.subTest(regime=regime), self.assertRaisesRegex(
                    ValueError, f"cannot verify {regime}"
                ):
                    prepare_cache([], regime)

    @unittest.skipUnless(
        sys.platform == "linux"
        and os.getenv("MORSEL_PARQUET_RUNNER")
        and os.getenv("MORSEL_BENCH_REVISION"),
        "set MORSEL_PARQUET_RUNNER and MORSEL_BENCH_REVISION for the Linux A/B smoke test",
    )
    def test_ab_verifies_both_plans_before_recording_stream_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture with spaces"
            generate(path, 2, 3, 1000)
            result = subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("harness.py")),
                    "ab",
                    str(path),
                    "--runner",
                    os.environ["MORSEL_PARQUET_RUNNER"],
                    "--revision",
                    os.environ["MORSEL_BENCH_REVISION"],
                    "--threads",
                    "2",
                    "--timeout",
                    "30",
                    "--cache",
                    "warm",
                    "--warmups",
                    "0",
                    "--repeats",
                    "2",
                ],
                text=True,
                timeout=120,
            )
            header, *trials = (json.loads(line) for line in result.splitlines())
            self.assertEqual(header["verified_matches"], {"low": 279, "high": 3506})
            self.assertEqual(len(header["evidence"]), 8)
            expected = {
                "scan_low": 279,
                "count_low": 1,
                "scan_high": 3506,
                "count_high": 1,
            }
            self.assertEqual(len(trials), 16)
            for index in range(0, len(trials), 2):
                first, second = trials[index : index + 2]
                self.assertEqual(first["case"], second["case"])
                self.assertEqual(first["rep"], second["rep"])
                modes = (
                    ["off", "required"] if first["rep"] == 1 else ["required", "off"]
                )
                self.assertEqual([first["mode"], second["mode"]], modes)
                for trial in (first, second):
                    self.assertEqual(trial["output_rows"], expected[trial["case"]])
                    self.assertGreater(trial["first_output_seconds"], 0)
                    self.assertLessEqual(
                        trial["first_output_seconds"], trial["seconds"]
                    )
                    pages = trial["cache_pages_before"]
                    self.assertGreater(pages["total"], 0)
                    self.assertEqual(pages["resident"], pages["total"])

    def test_fixture_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = (Path(tmp) / name for name in ("first", "second"))
            for path in (first, second):
                generate(path, 2, 3, 1000)
            files = fixture_files(first)
            self.assertEqual(len(files), 2)
            for file in files:
                self.assertEqual(file.read_bytes(), (second / file.name).read_bytes())
                parquet = pq.ParquetFile(file)
                self.assertEqual(parquet.metadata.num_rows, 3000)
                self.assertEqual(parquet.num_row_groups, 3)
                for index in range(3):
                    group = parquet.metadata.row_group(index)
                    self.assertEqual(group.num_rows, 1000)
                    for column in range(group.num_columns):
                        self.assertEqual(group.column(column).compression, "ZSTD")
                        self.assertTrue(group.column(column).is_stats_set)
                self.assertEqual(
                    parquet.schema_arrow.names,
                    ["row_id", "score", "delta", "payload", "unused"],
                )
            self.assertEqual(
                pq.read_table(files[0]).slice(0, 2).to_pylist(),
                [
                    dict(
                        row_id=0,
                        score=None,
                        delta=None,
                        payload=None,
                        unused=51272947919262,
                    ),
                    dict(
                        row_id=1,
                        score=21778,
                        delta=2950,
                        payload=2845013198969,
                        unused=97442737571047,
                    ),
                ],
            )

    @unittest.skipUnless(
        os.getenv("MORSEL_STOCK_CLI"), "set MORSEL_STOCK_CLI to test the CLI"
    )
    def test_stock_results_and_timed_drain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture with spaces"
            generate(path, 2, 3, 1000)
            backend = StockCli(Path(os.environ["MORSEL_STOCK_CLI"]), path, 2, 30)
            self.assertEqual(verify(backend, 6000), {"low": 279, "high": 3506})

            class WrongProjection(StockCli):
                def query(self, sql, *args):
                    return super().query(
                        sql.replace("payload + delta", "payload + delta + 1"), *args
                    )

            with self.assertRaisesRegex(ValueError, "incorrect projected values"):
                verify(WrongProjection(backend.binary, path, 2, 30), 6000)

            expected = {
                "scan_low": 279,
                "scan_high": 3506,
                "count_low": 1,
                "count_high": 1,
            }
            for name, query in workloads().items():
                with self.subTest(name=name):
                    result = backend.measure(query)
                    self.assertEqual(result.output_rows, expected[name])
                    self.assertGreater(result.seconds, 0)
                    self.assertIsNone(result.first_output_seconds)
                    self.assertIn("parquet", result.plan.lower())
                    self.assertIn("% 1000", result.plan)
                    if name.startswith("scan"):
                        self.assertIn("payload", result.plan)
                        self.assertIn("projected", result.plan)

    @unittest.skipUnless(
        os.getenv("MORSEL_STOCK_CLI"), "set MORSEL_STOCK_CLI to test the CLI"
    )
    def test_bench_with_large_verbose_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Full metrics repeat file paths, so this exceeds the 128 KiB CSV field limit.
            path = Path(tmp) / ("many-parts-" + "x" * 180)
            generate(path, 24, 2, 1000)
            files = fixture_files(path)
            backend = StockCli(Path(os.environ["MORSEL_STOCK_CLI"]), path, 8, 30)
            measurement = backend.measure(workloads()["scan_high"])
            self.assertEqual(measurement.output_rows, 28400)
            self.assertIn("projected", measurement.plan)

            with redirect_stdout(io.StringIO()) as output:
                bench(backend, files, 48000, 0, 1)
            trials = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len(trials), 4)
            self.assertEqual(
                {trial["case"]: trial["output_rows"] for trial in trials},
                {"scan_low": 2191, "count_low": 1, "scan_high": 28400, "count_high": 1},
            )
            pages = sum(
                (p.stat().st_size + mmap.PAGESIZE - 1) // mmap.PAGESIZE for p in files
            )
            for trial in trials:
                self.assertEqual(trial["rep"], 1)
                self.assertGreater(trial["seconds"], 0)
                for key in ("cache_pages_before", "cache_pages_after"):
                    if sys.platform == "linux":
                        self.assertEqual(trial[key]["total"], pages)
                        self.assertLessEqual(trial[key]["resident"], pages)
                    else:
                        self.assertIsNone(trial[key])
